"""
RELION-free 3D reconstruction by direct Fourier inversion (backprojection), implemented in PyTorch.

It is a Python alternative to ``relion_reconstruct --ctf --pad 2 [--sym X] [--fom_weighting]`` that does not
require a RELION installation. It follows the algorithm of ``relion_reconstruct`` (RELION >= 3.1, which skips the
iterative gridding correction by default):

1. Each particle image is Fourier transformed at its original box size N, translated by
   ``(rlnOriginXAngst, rlnOriginYAngst)`` with a phase shift and multiplied by its CTF (RELION's CTF convention,
   including phase shift, B-factor and scale factor when present). If figure-of-merit weighting is requested,
   both the image and the CTF^2 term are multiplied by ``rlnParticleFigureOfMerit``.
2. The central slice is rotated by the particle Euler matrix and inserted with trilinear weights into a padded
   ``(pad*N)^3`` half-Fourier volume. ``sum(CTF^2)`` is accumulated in the same way.
3. Hermitian symmetry is enforced on the ``kx = 0`` plane and point-group symmetry is applied to the accumulated
   data and weights by trilinear interpolation (RELION's ``BackProjector::symmetrise``). Point groups are built from
   the same generators as RELION, so their orientation follows RELION's conventions.
4. Fourier inversion: ``F = sum(CTF*img) / max(sum(CTF^2), shellMean(sum(CTF^2)) / 1000)``, restricted to Nyquist.
5. Inverse FFT, cropping to N, gridding correction (division by ``sinc^2(r / (pad*N))``) and RELION's soft
   spherical mask (radius N/2, 3 pixel cosine edge, background subtracted).

Not supported (an error is raised or a warning printed): optical aberrations in the optics table (beam tilt,
odd/even Zernikes, anisotropic magnification), phase-flipped or CTF-premultiplied particles, point groups with
mirrors or inversion (e.g. Cnv, Dnh, Td, Oh) and helical symmetry.

Command line usage:

```
python -m cesped.utils.backprojection --starFname particles.star --particlesDir /path/to/stacks \
    --outFname vol.mrc --symmetry C1
```
"""
import math
import os
import re
import warnings
from typing import Optional, Union

import mrcfile
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.spatial.transform import Rotation
from starstack.particlesStar import ParticlesStarSet

from cesped.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION, \
    RELION_PRED_POSE_CONFIDENCE_NAME

_UNSUPPORTED_OPTICS_LABELS = ["rlnBeamTiltX", "rlnBeamTiltY", "rlnOddZernike", "rlnEvenZernike",
                              "rlnMagMat00", "rlnMagMat01", "rlnMagMat10", "rlnMagMat11"]
_UNSUPPORTED_OPTICS_FLAGS = ["rlnCtfDataArePhaseFlipped", "rlnCtfDataAreCtfPremultiplied"]


def relion_ctf_rfft(box_size: int, sampling_rate: float, dfu: torch.Tensor, dfv: torch.Tensor,
                    dfang: torch.Tensor, volt: float, cs: float, amp_contrast: float,
                    phase_shift: Union[float, torch.Tensor] = 0., bfactor: Union[float, torch.Tensor] = 0.,
                    scale: Union[float, torch.Tensor] = 1., device=None,
                    dtype=torch.float64) -> torch.Tensor:
    """
    CTF following RELION's ``CTF::getCTF`` convention, sampled on an rFFT grid whose first (row, y) axis is
    fftshifted.

    Args:
        box_size (int): Image side, in pixels
        sampling_rate (float): Å/pixel
        dfu, dfv (Tensor): [B] defoci in Å (positive = underfocus)
        dfang (Tensor): [B] astigmatism angle in degrees, measured from the x (column) axis
        volt (float): Acceleration voltage in kV
        cs (float): Spherical aberration in mm
        amp_contrast (float): Amplitude contrast (Q0)
        phase_shift: [B] or scalar, in degrees
        bfactor: [B] or scalar, in Å^2
        scale: [B] or scalar
    Returns:
        Tensor: [B, box_size, box_size//2+1]
    """
    ky = torch.fft.fftshift(torch.fft.fftfreq(box_size, d=sampling_rate, device=device, dtype=dtype))
    kx = torch.fft.rfftfreq(box_size, d=sampling_rate, device=device, dtype=dtype)
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    u2 = kx ** 2 + ky ** 2
    ang = torch.atan2(ky, kx)

    def _col(v):
        return torch.as_tensor(v, device=device, dtype=dtype).reshape(-1, 1, 1)

    dfu, dfv, dfang = _col(dfu), _col(dfv), torch.deg2rad(_col(dfang))
    volt_v = volt * 1e3
    lam = 12.2643247 / math.sqrt(volt_v * (1. + volt_v * 0.978466e-6))
    cs_a = cs * 1e7
    k1 = math.pi * lam
    k2 = math.pi / 2. * cs_a * lam ** 3
    k3 = math.atan(amp_contrast / math.sqrt(1 - amp_contrast ** 2))
    k5 = torch.deg2rad(_col(phase_shift))
    defocus = -0.5 * (dfu + dfv) - 0.5 * (dfu - dfv) * torch.cos(2 * (ang - dfang))
    argument = k1 * defocus * u2 + k2 * u2 ** 2 - k5 - k3
    return -_col(scale) * torch.sin(argument) * torch.exp(-_col(bfactor) / 4. * u2)


# Generators (rotation order, axis) of RELION's point groups (SymList::fill_symmetry_class in symmetries.cpp)
_RELION_GENERATORS = {
    "T": [(3, (0., 0., 1.)), (2, (0., 0.816496, 0.577350))],
    "O": [(3, (.5773502, .5773502, .5773502)), (4, (0., 0., 1.))],
    "I": [(2, (0., 0., 1.)), (5, (0.525731114, 0., 0.850650807)), (3, (0., 0.356822076, 0.934172364))],
    "I1": [(2, (1., 0., 0.)), (5, (0.85065080702670, 0., -0.5257311142635)), (3, (0.9341723640, 0.3568220765, 0.))],
    "I3": [(2, (-0.5257311143, 0., 0.8506508070)), (5, (0., 0., 1.)),
           (3, (-0.4911234778630044, 0.3568220764705179, 0.7946544753759428))],
    "I4": [(2, (0.5257311143, 0., 0.8506508070)), (5, (0.8944271932547096, 0., 0.4472135909903704)),
           (3, (0.4911234778630044, 0.3568220764705179, 0.7946544753759428))],
}
_RELION_GENERATORS["I2"] = _RELION_GENERATORS["I"]
_GROUP_ORDERS = {"T": 12, "O": 24, "I": 60, "I1": 60, "I2": 60, "I3": 60, "I4": 60}


def get_symmetry_matrices(symmetry: str) -> torch.Tensor:
    """
    The rotation matrices of a point group, as a [K,3,3] float64 tensor (identity first), built from the same
    generators as RELION so that the orientation of the symmetry axes matches RELION's conventions.

    Args:
        symmetry (str): C1, Cn, Dn, T, O, I (=I2), I1, I2, I3 or I4. Groups with mirrors or inversion are not supported
    """
    sym = symmetry.upper().strip()
    match = re.fullmatch(r"([CD])(\d+)", sym)
    if match:
        kind, n = match.group(1), int(match.group(2))
        if n < 1:
            raise ValueError(f"Invalid symmetry {symmetry}")
        generators = [(n, (0., 0., 1.))] if n > 1 else []
        if kind == "D":
            generators.append((2, (1., 0., 0.)))
        order = n * (2 if kind == "D" else 1)
    elif sym in _RELION_GENERATORS:
        generators, order = _RELION_GENERATORS[sym], _GROUP_ORDERS[sym]
    else:
        raise ValueError(f"Symmetry {symmetry} is not supported by the python reconstruction. Supported: C<n>, D<n>, "
                         f"T, O, I, I1, I2, I3, I4. Use the relion backend for other groups")

    gens = [Rotation.from_rotvec(2 * np.pi / fold * np.asarray(axis) / np.linalg.norm(axis)).as_matrix()
            for fold, axis in generators]
    group = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                m = h @ g
                if not any(np.abs(m - x).max() < 1e-4 for x in group):
                    group.append(m)
                    new.append(m)
        frontier = new
        if len(group) > order:
            break
    if len(group) != order:
        raise RuntimeError(f"Error building symmetry group {symmetry}: {len(group)} elements, expected {order}")
    return torch.as_tensor(np.stack(group))


def _relion_padded_size(box_size: int, padding_factor: float) -> int:
    """RELION's padded size: ROUND(padding_factor * box_size), made even."""
    size = int(math.floor(padding_factor * box_size + 0.5))
    return size + size % 2


def _trilinear_corners(zyx: torch.Tensor):
    """Yields (corner_indices [M,3] long, weights [M]) for the 8 trilinear neighbours of zyx [M,3] (float)."""
    base = torch.floor(zyx)
    frac = zyx - base
    base = base.long()
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                off = torch.tensor([dz, dy, dx], device=zyx.device)
                w = torch.where(off.bool(), frac, 1. - frac).prod(-1)
                yield base + off, w


class FourierReconstructor:
    """
    Accumulates backprojected particles in Fourier space and computes the reconstruction.

    ```python
    rec = FourierReconstructor(box_size=128, sampling_rate=1.5, symmetry="C1")
    rec.backproject(imgs, rotMats, shiftsXYAngs, ctf=ctf, weights=fom)
    vol = rec.reconstruct()
    ```
    """

    def __init__(self, box_size: int, sampling_rate: float, symmetry: str = "C1", padding_factor: float = 2.,
                 regularization_ratio: float = 1000., device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            box_size (int): The particle image side, in pixels
            sampling_rate (float): Å/pixel
            symmetry (str): Point symmetry (C1, Cn, Dn, T, O, I, I1-I4), with RELION's axis conventions
            padding_factor (float): Oversampling of the 3D Fourier grid (RELION --pad). 2 by default.
            regularization_ratio (float): Fourier weights are floored to shellMean(weights)/regularization_ratio, \
                as RELION does (1000)
            device: torch device used for the computations
            dtype: Floating point type of the accumulators (float32 or float64)
        """
        self.box_size = int(box_size)
        self.sampling_rate = float(sampling_rate)
        self.symmetry = symmetry.upper()
        self.regularization_ratio = float(regularization_ratio)
        self.device = torch.device(device)
        self.dtype = dtype

        # As RELION: the padded size is even and the effective padding factor is recomputed from it
        self.pad_size = _relion_padded_size(self.box_size, padding_factor)
        self.padding_factor = self.pad_size / self.box_size  # image Fourier pixel -> padded volume Fourier pixel
        # The accumulation grid has a margin of 2 voxels around the centred axes, so that all trilinear neighbours
        # of any point up to Nyquist (insertion and symmetry interpolation) are inside the array
        self.grid_size = self.pad_size + 4
        self._center = self.grid_size // 2
        self._grid_shape = (self.grid_size, self.grid_size, self.pad_size // 2 + 3)

        self.data = torch.zeros((2,) + self._grid_shape, dtype=dtype, device=self.device)  # real, imag
        self.weight = torch.zeros(self._grid_shape, dtype=dtype, device=self.device)

        self.sym_matrices = get_symmetry_matrices(self.symmetry).to(device=self.device, dtype=dtype)
        self._init_plane()

    def _init_plane(self):
        """Pre-computes the coordinates of the central slice pixels that are within Nyquist."""
        N = self.box_size
        ky = torch.fft.fftshift(torch.fft.fftfreq(N, device=self.device, dtype=torch.float64)) * N
        kx = torch.fft.rfftfreq(N, device=self.device, dtype=torch.float64) * N
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        r2 = (kx ** 2 + ky ** 2).flatten()
        r_max = N // 2
        self._plane_idxs = torch.nonzero(r2 <= r_max ** 2).squeeze(-1)  # Same criterion as RELION
        coords = torch.stack([kx.flatten(), ky.flatten(), torch.zeros_like(r2)], -1)  # x, y, z
        self._plane_coords = coords[self._plane_idxs].to(self.dtype)  # [P, 3] in image Fourier pixels

    def _image_to_rfft(self, imgs: torch.Tensor) -> torch.Tensor:
        """Real-space images [B,N,N] -> rfft [B,N,N//2+1], with the image centre (N//2) at the origin and rows
        fftshifted"""
        imgs = torch.fft.ifftshift(imgs, dim=(-2, -1))
        fimgs = torch.fft.rfftn(imgs, dim=(-2, -1))
        return torch.fft.fftshift(fimgs, dim=(-2,))

    def _shift(self, fimgs: torch.Tensor, shiftsXYPix: torch.Tensor) -> torch.Tensor:
        """Translates the images by +shift pixels (RELION's convention for rlnOriginX/YAngst)."""
        N = self.box_size
        ky = torch.fft.fftshift(torch.fft.fftfreq(N, device=self.device, dtype=torch.float64))
        kx = torch.fft.rfftfreq(N, device=self.device, dtype=torch.float64)
        sx = shiftsXYPix[:, 0].to(torch.float64).reshape(-1, 1, 1)
        sy = shiftsXYPix[:, 1].to(torch.float64).reshape(-1, 1, 1)
        phase = -2 * torch.pi * (kx[None, None, :] * sx + ky[None, :, None] * sy)
        return fimgs * torch.polar(torch.ones_like(phase), phase).to(fimgs.dtype)

    def backproject(self, imgs: torch.Tensor, rotMats: torch.Tensor, shiftsXYAngs: torch.Tensor,
                    ctf: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None):
        """
        Inserts a batch of particles into the Fourier accumulators.

        Args:
            imgs (Tensor): [B,N,N] real space images
            rotMats (Tensor): [B,3,3] rotation matrices from the Relion Euler angles \
                (``cesped.utils.angles.euler_angles_to_matrix(angles, "ZYZ")``, i.e. the transpose of RELION's \
                projection matrix)
            shiftsXYAngs (Tensor): [B,2] rlnOriginXAngst, rlnOriginYAngst
            ctf (Optional[Tensor]): [B,N,N//2+1] CTF from ``relion_ctf_rfft``. If None, no CTF correction is applied
            weights (Optional[Tensor]): [B] per-particle weights (e.g. figure of merit). NaN weights are set to 0 \
                (RELION would propagate them into a NaN map) and negative weights are kept, as in RELION
        """
        cdtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        imgs = imgs.to(self.device, self.dtype)
        rotMats = rotMats.to(self.device, self.dtype)
        shiftsXYAngs = shiftsXYAngs.to(self.device, self.dtype)

        fimgs = self._image_to_rfft(imgs).to(cdtype)
        fimgs = self._shift(fimgs, shiftsXYAngs / self.sampling_rate)
        if ctf is not None:
            ctf = ctf.to(self.device, self.dtype)
            fimgs = fimgs * ctf
            ctf2 = ctf ** 2
        else:
            ctf2 = torch.ones(fimgs.shape, dtype=self.dtype, device=self.device)
        if weights is not None:
            weights = torch.nan_to_num(weights.to(self.device, self.dtype), nan=0.)
            fimgs = fimgs * weights.reshape(-1, 1, 1)
            ctf2 = ctf2 * weights.reshape(-1, 1, 1)

        B = imgs.shape[0]
        fimgs = fimgs.reshape(B, -1)[:, self._plane_idxs]  # [B, P]
        ctf2 = ctf2.reshape(B, -1)[:, self._plane_idxs]  # [B, P]
        values = torch.stack([fimgs.real, fimgs.imag, ctf2], -1)  # [B, P, 3]

        # Rotate the central slice: 3D frequency (x,y,z) = rotMat @ (kx, ky, 0), in padded voxel units
        coords = torch.einsum("bij,pj->bpi", rotMats, self._plane_coords) * self.padding_factor  # [B, P, 3]
        # Friedel: only the kx>=0 half is stored. Points with kx<0 are inserted at -k with conjugated value
        sign = torch.where(coords[..., 0] < 0, -1., 1.).to(self.dtype)
        coords = coords * sign.unsqueeze(-1)
        values = torch.cat([values[..., :1], values[..., 1:2] * sign.unsqueeze(-1), values[..., 2:]], -1)

        self._insert(coords.reshape(-1, 3), values.reshape(-1, 3))

    def _xyz_to_zyx_index(self, coords: torch.Tensor) -> torch.Tensor:
        c = self._center
        return torch.stack([coords[:, 2] + c, coords[:, 1] + c, coords[:, 0]], -1)

    def _insert(self, coords: torch.Tensor, values: torch.Tensor):
        """Trilinear insertion. coords [M,3] x,y,z in padded Fourier voxels; values [M,3] (real, imag, ctf2)."""
        D, H, W = self._grid_shape
        dims = torch.tensor([D, H, W], device=self.device)
        flat_data = self.data.view(2, -1)
        flat_weight = self.weight.view(-1)
        for idx, w in _trilinear_corners(self._xyz_to_zyx_index(coords)):
            valid = ((idx >= 0) & (idx < dims)).all(-1)  # Always true within Nyquist thanks to the margin
            flat_idx = torch.where(valid, idx[:, 0] * (H * W) + idx[:, 1] * W + idx[:, 2], 0)
            w = torch.where(valid, w, 0.)
            flat_data.index_add_(1, flat_idx, (values[:, :2] * w[:, None]).T)
            flat_weight.index_add_(0, flat_idx, values[:, 2] * w)

    def _enforce_hermitian_symmetry(self):
        """On the kx=0 plane, F(0,ky,kz) and F(0,-ky,-kz) are conjugate. Both are set to their (conj.) sum."""
        sl = (slice(1, None), slice(1, None), 0)  # index i in [1, G-1] <-> mirror G-i, with centre G/2
        real, imag, w = self.data[(0,) + sl], self.data[(1,) + sl], self.weight[sl]
        self.data[(0,) + sl] = real + real.flip(-1, -2)
        self.data[(1,) + sl] = imag - imag.flip(-1, -2)
        self.weight[sl] = w + w.flip(-1, -2)

    def _slab_coords(self, z0: int, z1: int):
        """x,y,z coordinates (padded voxels, centred) and r^2 of the grid planes z0:z1."""
        G, n_x = self.grid_size, self._grid_shape[-1]
        z = torch.arange(z0, z1, device=self.device, dtype=self.dtype) - self._center
        y = torch.arange(G, device=self.device, dtype=self.dtype) - self._center
        x = torch.arange(n_x, device=self.device, dtype=self.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return xx, yy, zz, xx ** 2 + yy ** 2 + zz ** 2

    def _symmetrise(self, slab: int = 8):
        """RELION's applyPointGroupSymmetry: adds the data/weights interpolated at R@k for every non-identity R."""
        if self.sym_matrices.shape[0] == 1:
            return
        D, H, W = self._grid_shape
        dims = torch.tensor([D, H, W], device=self.device)
        rmax2 = int(math.floor((self.box_size // 2) * self.padding_factor + 0.5)) ** 2  # ROUND(r_max*pad)^2
        src_data, src_weight = self.data.view(2, -1), self.weight.view(-1)
        sum_data, sum_weight = self.data.clone(), self.weight.clone()
        for z0 in range(0, D, slab):
            z1 = min(z0 + slab, D)
            xx, yy, zz, r2 = self._slab_coords(z0, z1)
            inside = r2 <= rmax2
            if not inside.any():
                continue
            xyz = torch.stack([xx[inside], yy[inside], zz[inside]], -1)
            acc = torch.zeros((xyz.shape[0], 3), dtype=self.dtype, device=self.device)
            for R in self.sym_matrices[1:]:
                rxyz = xyz @ R.T
                sign = torch.where(rxyz[:, 0] < 0, -1., 1.).to(self.dtype)
                rxyz = rxyz * sign[:, None]
                for idx, w in _trilinear_corners(self._xyz_to_zyx_index(rxyz)):
                    valid = ((idx >= 0) & (idx < dims)).all(-1)
                    flat_idx = torch.where(valid, idx[:, 0] * (H * W) + idx[:, 1] * W + idx[:, 2], 0)
                    w = torch.where(valid, w, 0.)
                    acc[:, 0] += src_data[0, flat_idx] * w
                    acc[:, 1] += src_data[1, flat_idx] * w * sign
                    acc[:, 2] += src_weight[flat_idx] * w
            sum_data[0, z0:z1][inside] += acc[:, 0]
            sum_data[1, z0:z1][inside] += acc[:, 1]
            sum_weight[z0:z1][inside] += acc[:, 2]
        self.data, self.weight = sum_data, sum_weight

    def reconstruct(self, apply_soft_mask: bool = True, mask_edge_width: float = 3., slab: int = 16) -> np.ndarray:
        """
        Computes the real space volume from the accumulated data. It can only be called once.

        Args:
            apply_soft_mask (bool): Apply RELION's soft spherical mask (radius N/2, cosine edge)
            mask_edge_width (float): Soft edge width, in pixels
            slab (int): Number of Fourier planes processed at once (lower it to reduce memory)
        Returns:
            np.ndarray: [N,N,N] float32 volume
        """
        self._enforce_hermitian_symmetry()
        self._symmetrise()

        N, P, D = self.box_size, self.pad_size, self.grid_size
        r_max_pad = self.padding_factor * (N // 2)
        n_shells = N // 2 + 2

        # Frequency-dependent regularization (RELION): weight >= shell_mean(weight)/1000, shells in original pixels
        shell_sum = torch.zeros(n_shells, dtype=torch.float64, device=self.device)
        shell_cnt = torch.zeros(n_shells, dtype=torch.float64, device=self.device)
        for z0 in range(0, D, slab):
            _, _, _, r2 = self._slab_coords(z0, min(z0 + slab, D))
            r = torch.sqrt(r2)
            inside = r < r_max_pad
            shell = torch.floor(r[inside] / self.padding_factor).long().clamp_max(n_shells - 1)
            shell_sum.index_add_(0, shell, self.weight[z0:z0 + slab][inside].to(torch.float64))
            shell_cnt.index_add_(0, shell, torch.ones_like(shell, dtype=torch.float64))
        floor = (shell_sum / shell_cnt.clamp_min(1.) / self.regularization_ratio).to(self.dtype)

        # Fourier volume without the accumulation margin: fftshifted (z, y) rfft of the padded size
        cdtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        lo = self._center - P // 2
        fvol = torch.zeros((P, P, P // 2 + 1), dtype=cdtype, device=self.device)
        for z0 in range(lo, lo + P, slab):
            z1 = min(z0 + slab, lo + P)
            _, _, _, r2 = self._slab_coords(z0, z1)
            r = torch.sqrt(r2)[:, lo:lo + P, :P // 2 + 1]
            shell = torch.floor(r / self.padding_factor).long().clamp_max(n_shells - 1)
            w = self.weight[z0:z1, lo:lo + P, :P // 2 + 1]
            w = torch.maximum(w, floor[shell])
            valid = (r < r_max_pad) & (w > 1e-20)
            d = self.data[:, z0:z1, lo:lo + P, :P // 2 + 1]
            fvol[z0 - lo:z1 - lo] = torch.where(valid, torch.complex(d[0], d[1]) / torch.where(valid, w, 1.), 0)
        del self.data, self.weight  # Free memory before the inverse FFT

        fvol = torch.fft.ifftshift(fvol, dim=(0, 1))
        vol = torch.fft.irfftn(fvol, s=(P, P, P), dim=(0, 1, 2))
        del fvol
        vol = torch.fft.fftshift(vol, dim=(0, 1, 2))

        # Window to the original box. With torch's normalised irfftn, the scale already matches relion_reconstruct
        start = P // 2 - N // 2
        vol = vol[start:start + N, start:start + N, start:start + N].to(torch.float64)

        # Gridding correction for trilinear interpolation (RELION's radial sinc^2)
        coords = torch.arange(N, device=self.device, dtype=torch.float64) - N // 2
        rr = torch.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
        vol = vol / torch.sinc(rr / P) ** 2

        if apply_soft_mask:
            vol = soft_mask_outside_map(vol, radius=N / 2., edge_width=mask_edge_width, rr=rr)
        return vol.to(torch.float32).cpu().numpy()


def soft_mask_outside_map(vol: torch.Tensor, radius: float, edge_width: float = 3.,
                          rr: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RELION's softMaskOutsideMap: replace the outside by the background mean, with a raised-cosine transition."""
    if rr is None:
        N = vol.shape[-1]
        coords = torch.arange(N, device=vol.device, dtype=vol.dtype) - N // 2
        rr = torch.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    radius_p = radius + edge_width
    # Background weight: 0 inside the sphere, raised cosine in the edge, 1 outside
    bg_w = 0.5 + 0.5 * torch.cos(torch.pi * (radius_p - rr) / edge_width)
    bg_w = torch.where(rr < radius, 0., torch.where(rr > radius_p, 1., bg_w)).to(vol.dtype)
    bg_mean = (vol * bg_w).sum() / bg_w.sum().clamp_min(1e-12)
    return (1. - bg_w) * vol + bg_w * bg_mean


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _check_star(md: pd.DataFrame, optics: pd.DataFrame, correctCtf: bool, starFname: str):
    for flag in _UNSUPPORTED_OPTICS_FLAGS:
        if flag in optics.columns and optics[flag].astype(int).any():
            raise NotImplementedError(f"{flag}=1 in {starFname} is not supported by the python reconstruction. "
                                      f"Use the relion backend")
    present = [c for c in _UNSUPPORTED_OPTICS_LABELS if c in optics.columns or c in md.columns]
    if present:
        warnings.warn(f"The following optical parameters are ignored by the python reconstruction: {present}")
    if "rlnImageSize" in optics.columns and optics["rlnImageSize"].nunique() > 1:
        raise ValueError(f"All optics groups in {starFname} must have the same rlnImageSize")
    if "rlnImagePixelSize" in optics.columns and optics["rlnImagePixelSize"].astype(float).nunique() > 1:
        raise ValueError(f"All optics groups in {starFname} must have the same rlnImagePixelSize")
    if "rlnOpticsGroup" in md.columns and "rlnOpticsGroup" in optics.columns:
        unknown = set(md["rlnOpticsGroup"].astype(int)) - set(optics["rlnOpticsGroup"].astype(int))
        if unknown:
            raise ValueError(f"Particles in {starFname} refer to optics groups not in the optics table: {unknown}")
    missing = [c for c in RELION_ANGLES_NAMES if c not in md.columns]
    if correctCtf:
        missing += [c for c in ("rlnDefocusU", "rlnDefocusV") if c not in md.columns]
        missing += [c for c in ("rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast")
                    if c not in optics.columns]
    if missing:
        raise ValueError(f"Missing columns in {starFname}: {missing}")


def reconstruct_starfile(starFname: str, outFname: Optional[str] = None, particlesDir: Optional[str] = None,
                         symmetry: str = "C1", correctCtf: bool = True, useFomWeighting: bool = False,
                         paddingFactor: float = 2., batchSize: int = 64, device: str = "auto",
                         verbose: bool = True) -> tuple[np.ndarray, float]:
    """
    Reconstructs a volume from the particles and poses stored in a RELION starfile, without using RELION.

    Follows ``relion_reconstruct --i starFname --o outFname --sym symmetry --pad paddingFactor [--ctf]
    [--fom_weighting]``. Results agree closely with RELION but are not bit-identical.

    Args:
        starFname (str): The starfile with the particles and their poses
        outFname (Optional[str]): The .mrc file where the volume will be saved. If None, it is not saved
        particlesDir (Optional[str]): The root directory for the rlnImageName paths. Default: the starfile directory \
            (note that relion_reconstruct resolves them relative to the working directory instead)
        symmetry (str): The point symmetry: C1, Cn, Dn, T, O, I, I1-I4 (RELION conventions)
        correctCtf (bool): Correct the CTF (RELION --ctf)
        useFomWeighting (bool): Weight particles by rlnParticleFigureOfMerit (RELION --fom_weighting)
        paddingFactor (float): Fourier oversampling factor (RELION --pad)
        batchSize (int): Number of particles processed at once
        device (str): "auto", "cpu", "cuda" or "cuda:N"
        verbose (bool): Show a progress bar
    Returns:
        tuple[np.ndarray, float]: The volume and its sampling rate
    """
    if not starFname:
        raise ValueError("starFname is required")
    starFname = os.path.expanduser(starFname)
    if particlesDir is None:
        particlesDir = os.path.dirname(os.path.abspath(starFname))
    particlesDir = os.path.expanduser(particlesDir)
    device = _resolve_device(device)

    pset = ParticlesStarSet(starFname=starFname, particlesDir=particlesDir)
    md = pset.particles_md
    optics = pset.optics_md
    sampling_rate = float(pset.sampling_rate)
    box_size = int(pset.particle_shape[-1])

    _check_star(md, optics, correctCtf, starFname)
    if useFomWeighting and RELION_PRED_POSE_CONFIDENCE_NAME not in md.columns:
        warnings.warn(f"{RELION_PRED_POSE_CONFIDENCE_NAME} not found in {starFname}. FOM weighting disabled")
        useFomWeighting = False
    optics = optics.set_index("rlnOpticsGroup") if "rlnOpticsGroup" in optics.columns else optics

    reconstructor = FourierReconstructor(box_size, sampling_rate, symmetry=symmetry, padding_factor=paddingFactor,
                                         device=device)
    from cesped.utils.angles import euler_angles_to_matrix

    n = len(pset)
    for start in tqdm.tqdm(range(0, n, batchSize), desc="Backprojecting", disable=not verbose):
        end = min(start + batchSize, n)
        imgs, batch_md = pset[start:end]
        batch_md = batch_md if isinstance(batch_md, pd.DataFrame) else md.iloc[start:end]
        imgs = torch.as_tensor(np.asarray(imgs, dtype=np.float32))
        angles = torch.as_tensor(batch_md[RELION_ANGLES_NAMES].values.astype(np.float64))
        rotMats = euler_angles_to_matrix(torch.deg2rad(angles), RELION_EULER_CONVENTION)
        shifts = torch.as_tensor(batch_md[RELION_SHIFTS_NAMES].values.astype(np.float64)) \
            if all(c in batch_md.columns for c in RELION_SHIFTS_NAMES) else torch.zeros(end - start, 2)
        weights = torch.as_tensor(batch_md[RELION_PRED_POSE_CONFIDENCE_NAME].values.astype(np.float64)) \
            if useFomWeighting else None
        ctf = None
        if correctCtf:
            ctf = _batch_ctf(batch_md, optics, box_size, sampling_rate, device)
        reconstructor.backproject(imgs, rotMats, shifts, ctf=ctf, weights=weights)

    vol = reconstructor.reconstruct()
    if outFname is not None:
        mrcfile.write(outFname, vol, voxel_size=sampling_rate, overwrite=True)
    return vol, sampling_rate


def _batch_ctf(batch_md: pd.DataFrame, optics: pd.DataFrame, box_size: int, sampling_rate: float,
               device) -> torch.Tensor:
    def _col(name, default):
        if name in batch_md.columns:
            return torch.as_tensor(batch_md[name].values.astype(np.float64))
        return torch.full((len(batch_md),), float(default), dtype=torch.float64)

    groups = batch_md["rlnOpticsGroup"].values.astype(int) if "rlnOpticsGroup" in batch_md.columns \
        else np.full(len(batch_md), -1)
    ctf = torch.empty((len(batch_md), box_size, box_size // 2 + 1), dtype=torch.float64, device=device)
    for group in np.unique(groups):
        sel = torch.as_tensor(np.where(groups == group)[0])
        opt = optics.loc[group] if group in optics.index else optics.iloc[0]  # -1: no optics group column
        ctf[sel.to(device)] = relion_ctf_rfft(
            box_size, sampling_rate,
            dfu=_col("rlnDefocusU", 0.)[sel], dfv=_col("rlnDefocusV", 0.)[sel],
            dfang=_col("rlnDefocusAngle", 0.)[sel],
            volt=float(opt["rlnVoltage"]), cs=float(opt["rlnSphericalAberration"]),
            amp_contrast=float(opt["rlnAmplitudeContrast"]),
            phase_shift=_col("rlnPhaseShift", 0.)[sel], bfactor=_col("rlnCtfBfactor", 0.)[sel],
            scale=_col("rlnCtfScalefactor", 1.)[sel], device=device)
    return ctf


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call

    parse_function_and_call(reconstruct_starfile)
