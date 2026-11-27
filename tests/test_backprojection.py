"""
Tests for the RELION-free reconstruction (cesped.utils.backprojection). They use small synthetic data and do not
require RELION or downloads. Agreement with relion_reconstruct itself was checked on the TEST entry (see release notes).
"""
import os
import tempfile
from unittest import TestCase

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from cesped.constants import RELION_EULER_CONVENTION
from cesped.datamanager.ctf import compute_ctf
from cesped.utils.angles import euler_angles_to_matrix
from cesped.utils.backprojection import FourierReconstructor, relion_ctf_rfft, reconstruct_starfile

BOX = 32
SAMPLING = 2.0


def _blob_volume(symmetry="C1", seed=0):
    """A smooth, asymmetric (unless symmetrized) volume made of Gaussian blobs."""
    rng = np.random.default_rng(seed)
    c = np.arange(BOX) - BOX // 2
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    xyz = np.stack([xx, yy, zz], -1).astype(np.float64)
    centers = rng.uniform(-7, 7, size=(5, 3))
    amps = rng.uniform(0.5, 1.5, size=5)
    ops = Rotation.create_group(symmetry).as_matrix()
    vol = np.zeros((BOX,) * 3)
    for op in ops:
        for cen, a in zip(centers, amps):
            vol += a * np.exp(-((xyz - op @ cen) ** 2).sum(-1) / (2 * 2.0 ** 2))
    return torch.as_tensor(vol, dtype=torch.float32)


def _project(vol, rotMats):
    """Real-space projections following RELION's convention: img(p) = sum_z V(rotMat @ (x, y, z))."""
    c = torch.arange(BOX, dtype=torch.float32) - BOX // 2
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    p = torch.stack([xx, yy, zz], -1).reshape(-1, 3)  # image coords x, y and depth z
    imgs = []
    for R in rotMats.float():
        r = p @ R.T  # volume coords (x, y, z)
        grid = (r + BOX // 2 + 0.5) / BOX * 2 - 1  # (x, y, z) normalized for align_corners=False
        sampled = F.grid_sample(vol[None, None], grid.reshape(1, BOX, BOX, BOX, 3), align_corners=False,
                                mode="bilinear", padding_mode="zeros")
        imgs.append(sampled[0, 0].sum(0))  # sum over depth (z)
    return torch.stack(imgs)


def _sphere_corr(a, b):
    c = np.arange(BOX) - BOX // 2
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    m = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2) < BOX // 2 - 2
    return np.corrcoef(np.asarray(a)[m], np.asarray(b)[m])[0, 1]


def _random_eulers(n, seed):
    return Rotation.random(n, random_state=seed).as_euler(RELION_EULER_CONVENTION, degrees=True)


class TestCTF(TestCase):
    def test_compute_ctf_matches_relion_convention_with_astigmatism(self):
        """Regression test: cesped's compute_ctf used rows as x, mis-orienting astigmatism (fixed after v26.09.0)."""
        N = 64
        for dfang in (0., 30., 90., 137.):
            kw = dict(dfu=torch.tensor([15000.]), dfv=torch.tensor([12000.]), dfang=torch.tensor([dfang]))
            ref = relion_ctf_rfft(N, 1.5, volt=300, cs=2.7, amp_contrast=0.1, **kw)[0].float()
            ctf = compute_ctf(N, 1.5, kw["dfu"], kw["dfv"], kw["dfang"], 300., 2.7, 0.1, 0., None, device="cpu")
            ctf = ctf.reshape(N, N)[:, N // 2:]  # fftshifted full grid -> kx >= 0 columns
            self.assertLess((ctf - ref[:, :N // 2]).abs().max().item(), 0.02, f"dfang={dfang}")

    def test_relion_ctf_dc_and_bounds(self):
        ctf = relion_ctf_rfft(64, 1.5, torch.tensor([20000.]), torch.tensor([18000.]), torch.tensor([45.]),
                              300., 2.7, 0.1)
        self.assertLessEqual(ctf.abs().max().item(), 1.0 + 1e-6)
        self.assertAlmostEqual(ctf[0, 32, 0].item(), 0.1, places=4)  # DC: -sin(-atan(Q0/sqrt(1-Q0^2))) = +Q0


class TestFourierReconstructor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol = _blob_volume()
        cls.eulers = _random_eulers(1500, seed=1)
        cls.rotMats = euler_angles_to_matrix(torch.deg2rad(torch.as_tensor(cls.eulers)), RELION_EULER_CONVENTION)
        cls.imgs = _project(cls.vol, cls.rotMats)

    def _reconstruct(self, imgs, shifts=None, ctf=None, weights=None, symmetry="C1", rotMats=None):
        rec = FourierReconstructor(BOX, SAMPLING, symmetry=symmetry)
        rotMats = self.rotMats if rotMats is None else rotMats
        shifts = torch.zeros(len(imgs), 2) if shifts is None else shifts
        for i in range(0, len(imgs), 256):
            sl = slice(i, i + 256)
            rec.backproject(imgs[sl], rotMats[sl], shifts[sl], ctf=None if ctf is None else ctf[sl],
                            weights=None if weights is None else weights[sl])
        return rec.reconstruct()

    def test_roundtrip(self):
        out = self._reconstruct(self.imgs)
        self.assertGreater(_sphere_corr(out, self.vol), 0.98)
        scale = (out * self.vol.numpy()).sum() / (out ** 2).sum()
        self.assertAlmostEqual(scale, 1.0, delta=0.15)  # Same magnitude as the input

    def test_shifts(self):
        """Images translated by -origin and reconstructed with rlnOrigin=+origin must give the same volume."""
        rng = np.random.default_rng(2)
        shiftsPix = torch.as_tensor(rng.integers(-3, 4, size=(len(self.imgs), 2)), dtype=torch.float32)
        shifted = torch.stack([torch.roll(img, shifts=(-int(s[1]), -int(s[0])), dims=(0, 1))
                               for img, s in zip(self.imgs, shiftsPix)])
        out = self._reconstruct(shifted, shifts=shiftsPix * SAMPLING)
        self.assertGreater(_sphere_corr(out, self.vol), 0.98)
        wrong = self._reconstruct(shifted, shifts=-shiftsPix * SAMPLING)
        self.assertLess(_sphere_corr(wrong, self.vol), 0.9)

    def test_ctf_correction(self):
        n = len(self.imgs)
        rng = np.random.default_rng(3)
        dfs = torch.as_tensor(rng.uniform(8000, 20000, n))
        ctf = relion_ctf_rfft(BOX, SAMPLING, dfs, dfs - 500, torch.as_tensor(rng.uniform(0, 180, n)),
                              300., 2.7, 0.1)
        fimgs = torch.fft.fftshift(torch.fft.rfftn(torch.fft.fftshift(self.imgs, dim=(-2, -1)), dim=(-2, -1)),
                                   dim=(-2,))
        fimgs = fimgs * ctf
        corrupted = torch.fft.ifftshift(torch.fft.irfftn(torch.fft.ifftshift(fimgs, dim=(-2,)), s=(BOX, BOX)),
                                        dim=(-2, -1)).float()
        out = self._reconstruct(corrupted, ctf=ctf)
        self.assertGreater(_sphere_corr(out, self.vol), 0.95)

    def test_weights(self):
        """Down-weighting garbage particles must recover the volume."""
        imgs = self.imgs.clone()
        bad = torch.arange(len(imgs)) % 3 == 0
        imgs[bad] = torch.randn_like(imgs[bad]) * imgs.std() * 5
        weights = torch.where(bad, 0., 1.)
        good = self._reconstruct(imgs, weights=weights)
        unweighted = self._reconstruct(imgs)
        self.assertGreater(_sphere_corr(good, self.vol), 0.97)
        self.assertGreater(_sphere_corr(good, self.vol), _sphere_corr(unweighted, self.vol))

    def test_symmetry(self):
        vol = _blob_volume("C4", seed=4)
        eulers = _random_eulers(400, seed=5)
        rotMats = euler_angles_to_matrix(torch.deg2rad(torch.as_tensor(eulers)), RELION_EULER_CONVENTION)
        imgs = _project(vol, rotMats)
        out = self._reconstruct(imgs, symmetry="C4", rotMats=rotMats)
        self.assertGreater(_sphere_corr(out, vol), 0.97)


class TestReconstructStarfile(TestCase):
    def test_starfile_roundtrip(self):
        vol = _blob_volume(seed=6)
        eulers = _random_eulers(800, seed=7)
        rotMats = euler_angles_to_matrix(torch.deg2rad(torch.as_tensor(eulers)), RELION_EULER_CONVENTION)
        imgs = _project(vol, rotMats).numpy().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            mrcfile.write(os.path.join(tmpdir, "stack.mrcs"), imgs, voxel_size=SAMPLING)
            n = len(imgs)
            particles = pd.DataFrame({
                "rlnImageName": [f"{i + 1:06d}@stack.mrcs" for i in range(n)],
                "rlnAngleRot": eulers[:, 0], "rlnAngleTilt": eulers[:, 1], "rlnAnglePsi": eulers[:, 2],
                "rlnOriginXAngst": 0., "rlnOriginYAngst": 0., "rlnOpticsGroup": 1,
                "rlnParticleFigureOfMerit": 1.,
            })
            optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"],
                                   "rlnVoltage": [300.], "rlnSphericalAberration": [2.7],
                                   "rlnAmplitudeContrast": [0.1], "rlnImagePixelSize": [SAMPLING],
                                   "rlnImageSize": [BOX], "rlnImageDimensionality": [2]})
            starFname = os.path.join(tmpdir, "particles.star")
            starfile.write({"optics": optics, "particles": particles}, starFname)
            outFname = os.path.join(tmpdir, "out.mrc")
            out, sr = reconstruct_starfile(starFname, outFname, symmetry="C1", correctCtf=False,
                                           useFomWeighting=True, device="cpu", verbose=False)
            self.assertEqual(sr, SAMPLING)
            with mrcfile.open(outFname) as f:
                self.assertAlmostEqual(float(f.voxel_size.x), SAMPLING, places=4)
                np.testing.assert_allclose(f.data, out)
            self.assertGreater(_sphere_corr(out, vol), 0.98)


class TestEvaluatorBackend(TestCase):
    def test_invalid_backend(self):
        from cesped.evaluateEntry import Evaluator
        with self.assertRaises(ValueError):
            Evaluator("SO3", reconstructionBackend="foo")

    def test_python_backend_does_not_need_relion(self):
        from cesped.evaluateEntry import Evaluator
        ev = Evaluator("SO3", relionBinDir=None, mpirun=None, relionSingularity=None, reconstructionBackend="python")
        self.assertEqual(ev.reconstructionBackend, "python")
        with self.assertRaises(ValueError):
            Evaluator("SO3", relionBinDir=None, mpirun=None, relionSingularity=None, reconstructionBackend="relion")
