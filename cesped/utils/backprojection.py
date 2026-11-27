"""
RELION-free 3D reconstruction by direct Fourier inversion (backprojection), implemented in PyTorch.

It is a Python alternative to ``relion_reconstruct --ctf --pad 2 [--sym X] [--fom_weighting]`` that does not
require a RELION installation. It mirrors the default algorithm used by ``relion_reconstruct`` (RELION >= 3.1,
which skips the iterative gridding correction by default):

1. Each particle image is Fourier transformed at its original box size N, translated by
   ``(rlnOriginXAngst, rlnOriginYAngst)`` with a phase shift and multiplied by its CTF (RELION's CTF convention,
   including phase shift, B-factor and scale factor when present). If figure-of-merit weighting is requested,
   both the image and the CTF^2 term are multiplied by ``rlnParticleFigureOfMerit``.
2. The central slice is rotated by the particle Euler matrix (and by every operator of the point group) and
   inserted with trilinear weights into a padded ``(pad*N)^3`` half-Fourier volume. ``sum(CTF^2)`` is accumulated
   in the same way.
3. Hermitian symmetry is enforced on the ``kx = 0`` plane.
4. Fourier inversion: ``F = sum(CTF*img) / max(sum(CTF^2), shellMean(sum(CTF^2)) / 1000)``, restricted to Nyquist.
5. Inverse FFT, cropping to N, gridding correction (division by ``sinc^2(r / (pad*N))``) and RELION's soft
   spherical mask (radius N/2, 3 pixel cosine edge, background subtracted).

Optical aberrations stored in the optics table (beam tilt, odd/even Zernikes, anisotropic magnification) are NOT
modelled; a warning is printed when they are present.

The trilinear-insertion approach was adapted from cryoPARES (https://github.com/rsanchezgarc/cryoPARES), which in
turn derives from torch-fourier-slice (https://github.com/teamtomo/torch-fourier-slice,
BSD 3-Clause License, Copyright (c) 2023 Alister Burt). See THIRD-PARTY-LICENSES.

Command line usage:

```
python -m cesped.utils.backprojection --starFname particles.star --particlesDir /path/to/stacks \
    --outFname vol.mrc --symmetry C1
```
"""
import math
import os
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


def get_symmetry_matrices(symmetry: str) -> torch.Tensor:
    """The rotation matrices of the point group, as a [K,3,3] tensor (scipy convention)."""
    return torch.as_tensor(Rotation.create_group(symmetry.upper()).as_matrix())


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
            symmetry (str): Point symmetry (e.g. C1, C4, D2)
            padding_factor (float): Oversampling of the 3D Fourier grid (RELION --pad). 2 by default.
            regularization_ratio (float): Fourier weights are floored to shellMean(weights)/regularization_ratio, \
                as RELION does (1000)
            device: torch device used for the computations
            dtype: Floating point type of the accumulators
        """
        self.box_size = int(box_size)
        self.sampling_rate = float(sampling_rate)
        self.symmetry = symmetry.upper()
        self.padding_factor = float(padding_factor)
        self.regularization_ratio = float(regularization_ratio)
        self.device = torch.device(device)
        self.dtype = dtype

        # Padded size, even. The accumulation grid has a margin of 1 voxel (on both sides of the centred axes)
        # so that all trilinear neighbours of any sample up to Nyquist are inside the array
        self.pad_size = 2 * int(round(self.padding_factor * self.box_size / 2.))
        self._scale = self.pad_size / self.box_size  # image Fourier pixel -> padded volume Fourier pixel
        self.grid_size = self.pad_size + 2
        self._center = self.grid_size // 2
        n_x = self.pad_size // 2 + 2
        self._grid_shape = (self.grid_size, self.grid_size, n_x)

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
        self._corner_offsets = torch.tensor([[dz, dy, dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                                            device=self.device, dtype=torch.long)  # z,y,x

    def _image_to_rfft(self, imgs: torch.Tensor) -> torch.Tensor:
        """Real-space images [B,N,N] -> rfft [B,N,N//2+1], with the image centre at the origin and fftshifted rows"""
        imgs = torch.fft.fftshift(imgs, dim=(-2, -1))
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
            weights (Optional[Tensor]): [B] per-particle weights (e.g. figure of merit)
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
            weights = torch.nan_to_num(weights.to(self.device, self.dtype), nan=0.).clamp_min(0.)
            fimgs = fimgs * weights.reshape(-1, 1, 1)
            ctf2 = ctf2 * weights.reshape(-1, 1, 1)

        B = imgs.shape[0]
        fimgs = fimgs.reshape(B, -1)[:, self._plane_idxs]  # [B, P]
        ctf2 = ctf2.reshape(B, -1)[:, self._plane_idxs]  # [B, P]
        values = torch.stack([fimgs.real, fimgs.imag, ctf2], -1)  # [B, P, 3]

        # Symmetry expansion. A symmetric volume V(x) = sum_k V(R_k x), which is equivalent to inserting each image
        # with the projection matrix A @ R_k (RELION convention), i.e. with rotMat_k = R_k^T @ rotMat
        K = self.sym_matrices.shape[0]
        if K > 1:
            rotMats = (self.sym_matrices.transpose(-1, -2)[None, ...] @ rotMats[:, None, ...]).reshape(-1, 3, 3)
            values = values.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, *values.shape[1:])

        # Rotate the central slice: 3D frequency (x,y,z) = rotMat @ (kx, ky, 0), in padded voxel units
        coords = torch.einsum("bij,pj->bpi", rotMats, self._plane_coords) * self._scale  # [BK, P, 3] x,y,z
        # Friedel: only the kx>=0 half is stored. Points with kx<0 are inserted at -k with conjugated value
        neg = coords[..., 0] < 0
        sign = torch.where(neg, -1., 1.).to(self.dtype)
        coords = coords * sign.unsqueeze(-1)
        values = torch.cat([values[..., :1], values[..., 1:2] * sign.unsqueeze(-1), values[..., 2:]], -1)

        self._insert(coords.reshape(-1, 3), values.reshape(-1, 3))

    def _insert(self, coords: torch.Tensor, values: torch.Tensor):
        """Trilinear insertion. coords [M,3] x,y,z in padded Fourier voxels; values [M,3] (real, imag, ctf2)."""
        c = self._center
        zyx = torch.stack([coords[:, 2] + c, coords[:, 1] + c, coords[:, 0]], -1)
        base = torch.floor(zyx)
        frac = zyx - base
        base = base.long()
        D, H, W = self._grid_shape
        flat_vol = torch.cat([self.data.reshape(2, -1), self.weight.reshape(1, -1)], 0)
        for off in self._corner_offsets:
            idx = base + off
            w = torch.where(off.bool(), frac, 1. - frac).prod(-1)  # [M]
            # The margin of the grid guarantees that indices are in bounds within Nyquist. Guard anyway.
            valid = ((idx >= 0) & (idx < torch.tensor([D, H, W], device=idx.device))).all(-1)
            flat_idx = idx[:, 0] * (H * W) + idx[:, 1] * W + idx[:, 2]
            flat_idx = torch.where(valid, flat_idx, torch.zeros_like(flat_idx))
            w = torch.where(valid, w, torch.zeros_like(w))
            flat_vol.index_add_(1, flat_idx, (values * w[:, None]).T)
        self.data = flat_vol[:2].reshape(self.data.shape)
        self.weight = flat_vol[2].reshape(self.weight.shape)

    def _enforce_hermitian_symmetry(self):
        """On the kx=0 plane, F(0,ky,kz) and F(0,-ky,-kz) are conjugate. Both are set to their (conj.) sum."""
        sl = (slice(1, None), slice(1, None), 0)  # index i in [1, G-1] <-> mirror G-i, with centre G/2
        data = self.data[(slice(None),) + sl]
        mirror = data.flip(-1, -2)
        self.data[(0,) + sl] = data[0] + mirror[0]
        self.data[(1,) + sl] = data[1] - mirror[1]
        w = self.weight[sl]
        self.weight[sl] = w + w.flip(-1, -2)

    def _shell_radius(self) -> torch.Tensor:
        """Radius of each voxel of the (cropped) half Fourier grid, in padded voxels."""
        G, n_x = self.grid_size, self._grid_shape[-1]
        z = torch.arange(G, device=self.device, dtype=torch.float64) - self._center
        x = torch.arange(n_x, device=self.device, dtype=torch.float64)
        return torch.sqrt(z[:, None, None] ** 2 + z[None, :, None] ** 2 + x[None, None, :] ** 2)

    def reconstruct(self, apply_soft_mask: bool = True, mask_edge_width: float = 3.) -> np.ndarray:
        """
        Computes the real space volume from the accumulated data.

        Args:
            apply_soft_mask (bool): Apply RELION's soft spherical mask (radius N/2, cosine edge)
            mask_edge_width (float): Soft edge width, in pixels
        Returns:
            np.ndarray: [N,N,N] float32 volume
        """
        self._enforce_hermitian_symmetry()
        r = self._shell_radius()
        r_max_pad = self._scale * (self.box_size // 2)
        inside = r < r_max_pad

        # Frequency-dependent regularization (RELION): weight >= shell_mean(weight)/1000, shells in original pixels
        shell = torch.floor(r / self._scale).long().clamp_max(self.box_size)
        n_shells = self.box_size + 1
        weight = self.weight.to(torch.float64)
        shell_sum = torch.zeros(n_shells, dtype=torch.float64, device=self.device)
        shell_cnt = torch.zeros(n_shells, dtype=torch.float64, device=self.device)
        shell_sum.index_add_(0, shell[inside], weight[inside])
        shell_cnt.index_add_(0, shell[inside], torch.ones_like(weight[inside]))
        shell_mean = shell_sum / shell_cnt.clamp_min(1.)
        weight = torch.maximum(weight, shell_mean[shell] / self.regularization_ratio)

        valid = inside & (weight > 1e-20)
        fvol = torch.zeros(self._grid_shape, dtype=torch.complex128, device=self.device)
        data = self.data.to(torch.float64)
        fvol[valid] = torch.complex(data[0][valid], data[1][valid]) / weight[valid]

        # Remove the accumulation margin -> fftshifted rfft volume of the padded size
        P = self.pad_size
        lo = self._center - P // 2
        fvol = fvol[lo:lo + P, lo:lo + P, :P // 2 + 1]
        fvol = torch.fft.ifftshift(fvol, dim=(0, 1))
        vol = torch.fft.irfftn(fvol, s=(P, P, P), dim=(0, 1, 2))
        vol = torch.fft.fftshift(vol, dim=(0, 1, 2))

        # Window to the original box. With torch's normalised irfftn, the scale already matches relion_reconstruct
        N = self.box_size
        start = P // 2 - N // 2
        vol = vol[start:start + N, start:start + N, start:start + N]

        # Gridding correction for trilinear interpolation (RELION's radial sinc^2)
        coords = torch.arange(N, device=self.device, dtype=torch.float64) - N // 2
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
        rr = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
        vol = vol / torch.sinc(rr / P) ** 2

        if apply_soft_mask:
            vol = soft_mask_outside_map(vol, radius=N / 2., edge_width=mask_edge_width, rr=rr)
        return vol.to(torch.float32).cpu().numpy()


def soft_mask_outside_map(vol: torch.Tensor, radius: float, edge_width: float = 3.,
                          rr: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RELION's softMaskOutsideMap: subtract the mean of the background and apply a raised-cosine spherical mask."""
    if rr is None:
        N = vol.shape[-1]
        coords = torch.arange(N, device=vol.device, dtype=vol.dtype) - N // 2
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
        rr = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
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


def reconstruct_starfile(starFname: str, outFname: Optional[str] = None, particlesDir: Optional[str] = None,
                         symmetry: str = "C1", correctCtf: bool = True, useFomWeighting: bool = False,
                         paddingFactor: float = 2., batchSize: int = 64, device: str = "auto",
                         verbose: bool = True) -> tuple[np.ndarray, float]:
    """
    Reconstructs a volume from the particles and poses stored in a RELION starfile, without using RELION.

    Equivalent to ``relion_reconstruct --i starFname --o outFname --sym symmetry --pad paddingFactor [--ctf]
    [--fom_weighting]``.

    Args:
        starFname (str): The starfile with the particles and their poses
        outFname (Optional[str]): The .mrc file where the volume will be saved. If None, it is not saved
        particlesDir (Optional[str]): The root directory for the rlnImageName paths. Default: the starfile directory
        symmetry (str): The point symmetry (e.g. C1, C4, D2)
        correctCtf (bool): Correct the CTF (RELION --ctf)
        useFomWeighting (bool): Weight particles by rlnParticleFigureOfMerit (RELION --fom_weighting)
        paddingFactor (float): Fourier oversampling factor (RELION --pad)
        batchSize (int): Number of particles processed at once
        device (str): "auto", "cpu", "cuda" or "cuda:N"
        verbose (bool): Show a progress bar
    Returns:
        tuple[np.ndarray, float]: The volume and its sampling rate
    """
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

    present = [c for c in _UNSUPPORTED_OPTICS_LABELS if c in optics.columns or c in md.columns]
    if present:
        warnings.warn(f"The following optical parameters are ignored by the Python reconstruction: {present}")
    if useFomWeighting and RELION_PRED_POSE_CONFIDENCE_NAME not in md.columns:
        warnings.warn(f"{RELION_PRED_POSE_CONFIDENCE_NAME} not found in {starFname}. FOM weighting disabled")
        useFomWeighting = False

    optics = optics.set_index("rlnOpticsGroup") if "rlnOpticsGroup" in optics.columns else optics
    for col in ("rlnImagePixelSize",):
        if col in optics.columns and not np.allclose(optics[col].values.astype(float), sampling_rate):
            raise ValueError("All optics groups must have the same pixel size")

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
        else np.ones(len(batch_md), dtype=int)
    ctf = torch.empty((len(batch_md), box_size, box_size // 2 + 1), dtype=torch.float64, device=device)
    for group in np.unique(groups):
        sel = np.where(groups == group)[0]
        opt = optics.loc[group] if group in optics.index else optics.iloc[0]
        sel_t = torch.as_tensor(sel)
        ctf[sel_t.to(device)] = relion_ctf_rfft(
            box_size, sampling_rate,
            dfu=_col("rlnDefocusU", 0.)[sel_t], dfv=_col("rlnDefocusV", 0.)[sel_t],
            dfang=_col("rlnDefocusAngle", 0.)[sel_t],
            volt=float(opt["rlnVoltage"]), cs=float(opt["rlnSphericalAberration"]),
            amp_contrast=float(opt["rlnAmplitudeContrast"]),
            phase_shift=_col("rlnPhaseShift", 0.)[sel_t], bfactor=_col("rlnCtfBfactor", 0.)[sel_t],
            scale=_col("rlnCtfScalefactor", 1.)[sel_t], device=device)
    return ctf


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call

    parse_function_and_call(reconstruct_starfile)
