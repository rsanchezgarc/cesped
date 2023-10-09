import os
import shutil
from unittest import TestCase

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
        self.assertEqual(img.shape, (1, 284,284))

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