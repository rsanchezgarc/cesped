import os
import shutil
import tempfile
from unittest import TestCase

import pandas as pd
import starfile
from cesped.particlesDataset import ParticlesDataset

benchmarkDir = "/tmp/cryoSupervisedDataset/"
testTargetName = "TEST"
class TestParticlesDataset(TestCase):
    def test__download(self):
        ds = ParticlesDataset(testTargetName, 0, benchmarkDir=benchmarkDir)
        print(len(ds))
        ds = ParticlesDataset(testTargetName, 1, benchmarkDir=benchmarkDir)
        print(len(ds))
        iid, img, (rotMat, xyShiftAngs, confidence), metadata = ds[0]
        print([x.shape for x in [img, rotMat, xyShiftAngs]])
        self.assertEqual(img.shape, (1, 232,232))

    def test_addNewEntry(self):
        ds = ParticlesDataset(testTargetName, 0, benchmarkDir=benchmarkDir)
        starFname = ds.starFname
        newTargetName="NewTarget_test_registerNewEntry"
        newTargetDir = os.path.join(benchmarkDir, newTargetName)
        if os.path.exists(newTargetDir):
            shutil.rmtree(newTargetDir)
        ParticlesDataset.addNewEntryLocally(starFname, particlesRootDir=os.path.split(starFname)[0],
                                            newTargetName=newTargetName, halfset=0, symmetry=ds.symmetry,
                                            benchmarkDir=benchmarkDir)
        ds = ParticlesDataset(newTargetName, 0, benchmarkDir=benchmarkDir)
        print(len(ds))
        print(ParticlesDataset.getLocallyAvailableEntries(benchmarkDir))

    def test_merge_star_files_combines_headers(self):
        from cesped.particlesDataset import _merge_star_files

        with tempfile.TemporaryDirectory() as tmpdir:
            star0 = {
                "optics": pd.DataFrame({"rlnVoltage": [300]}),
                "particles": pd.DataFrame({"rlnImageName": ["0"], "rlnCoordinateX": [1]}),
            }
            star1 = {
                "optics": pd.DataFrame({"rlnVoltage": [200], "rlnSphericalAberration": [2.0]}),
                "particles": pd.DataFrame({"rlnImageName": ["1"], "rlnCoordinateY": [2]}),
            }
            star0_fname = os.path.join(tmpdir, "p0.star")
            star1_fname = os.path.join(tmpdir, "p1.star")
            merged_fname = os.path.join(tmpdir, "merged.star")

            starfile.write(star0, star0_fname)
            starfile.write(star1, star1_fname)

            _merge_star_files([star0_fname, star1_fname], merged_fname)

            merged = starfile.read(merged_fname)
            particles = merged["particles"] if isinstance(merged, dict) else merged
            self.assertEqual(len(particles), 2)
            self.assertIn("rlnCoordinateX", particles.columns)
            self.assertIn("rlnCoordinateY", particles.columns)

            optics = merged.get("optics") if isinstance(merged, dict) else None
            self.assertIsNotNone(optics)
            self.assertEqual(len(optics), 2)
