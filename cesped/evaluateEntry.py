"""
This module allows for benchmark entry evaluation given the inferred poses.
"""
import os
import shutil
import subprocess
import os.path as osp
import tempfile
import warnings
from typing import Literal, Optional, Tuple, Union

import mrcfile
import numpy as np
import starfile

from cesped.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_ORI_POSE_CONFIDENCE_NAME, defaultBenchmarkDir, relionBinDir, mpirunCmd
from cesped.particlesDataset import ParticlesDataset
from cesped.utils.anglesStats import computeAngularError
from cesped.utils.volumeStats import compute_stats


class Evaluator():
    resolution_threshold: float = 0.5

    def __init__(self, predictionType: Literal["S2", "SO3", "SO3xR2"], usePredConfidence: bool = True,
                 benchmarkDir: Union[str, os.PathLike] = defaultBenchmarkDir, relionBinDir=relionBinDir,
                 mpirun=mpirunCmd, n_cpus=1, wdir=None, verbose=True):

        self.benchmarkDir = benchmarkDir
        self.predictionType = predictionType
        self.usePredConfidence = usePredConfidence
        self.relionBinDir = relionBinDir if relionBinDir else ""
        self.mpirun = mpirun
        self.n_cpus = n_cpus
        self.wdir = wdir
        self.verbose = verbose

    def reconstruct(self, starFname: str, outname: str, particlesDir: str, symmetry: str = "c1",
                    cleanExisting: bool = True) -> Tuple[
        np.ndarray, float]:
        """
        Reconstruct a volume from the information contained in a starfile.
        Args:
            starFname: The starfile with the information to be reconstructed
            outname: The .mrc file were the reconstructed volume will be saved
            particlesDir (str): particlesDir
            symmetry: The point symmetry of the dataset.
            cleanExisting: True if an existing file should be removed
        Returns:
            Tuple[np.ndarray, float]:
        """
        exists = osp.exists(outname)
        if exists and cleanExisting:
            os.unlink(outname)

        if not exists:
            cmd = []
            if self.n_cpus > 1:
                cmd += self.mpirun.split() + ["-np", str(self.n_cpus),
                                              osp.join(self.relionBinDir, "relion_reconstruct_mpi")]
            else:
                cmd += [osp.join(self.relionBinDir, "relion_reconstruct")]
            with tempfile.NamedTemporaryFile(suffix=".mrc") as f:
                cmd += ["--i", starFname, "--o", f.name, "--ctf", "--sym", symmetry.lower()]
                if self.usePredConfidence:
                    data = starfile.read(starFname)
                    if RELION_PRED_POSE_CONFIDENCE_NAME in data["particles"]:
                        cmd += ["--fom_weighting"]
                wdir = particlesDir if particlesDir else osp.dirname(starFname)
                if self.verbose:
                    print(" ".join(cmd))
                subprocess.run(cmd, cwd=wdir, check=True, capture_output=not self.verbose)
                f.seek(0)
                shutil.copyfile(f.name, outname)

        with mrcfile.open(outname) as f:
            data = f.data.copy()
            sampling_rate = float(f.voxel_size.x)
        return data, sampling_rate

    def computeAvgMap(self, starFname0: str, starFname1: str, particlesDir: str,
                      outbasename: str, symmetry: str, cleanExisting: bool=True) \
            -> Tuple[np.ndarray, float, float, float]:
        """
        Reconstruct each of the half-datasets starFname0 and starFname0 and averages the reconstruction to
        obtain the final average map
        Args:
            starFname0 (str):
            starFname1 (str):
            particlesDir (str):
            outbasename (str):
            symmetry (str):
            cleanExisting (bool): True if an existing file should be removed

        Returns:

        """
        name0 = osp.join(self.wdir, outbasename + "_0.mrc")
        data0, sr0 = self.reconstruct(starFname0, name0, particlesDir, symmetry=symmetry, cleanExisting=cleanExisting)

        name1 = osp.join(self.wdir, outbasename + "_1.mrc")
        data1, sr1 = self.reconstruct(starFname1, name1, particlesDir, cleanExisting=cleanExisting)
        assert sr0 == sr1, "Error, the sampling rate of the two datasets is different"
        corr, resolt = compute_stats(name0, name1, samplingRate=sr0,
                                     resolution_threshold=self.resolution_threshold)

        resolt = resolt[0]
        outvol = .5 * (data0 + data1)
        if not outbasename.endswith(".mrc"):
            outbasename += ".mrc"
        mrcfile.write(osp.join(self.wdir, outbasename), outvol, voxel_size=sr0, overwrite=True)

        return outvol, sr0, corr, resolt

    def preparePredStar(self, mainStarFname: str, predStarFname: str, outname: str, symmetry: str):

        if self.predictionType == "S2":
            keys = RELION_ANGLES_NAMES[:2].copy()
        elif self.predictionType == "SO3":
            keys = RELION_ANGLES_NAMES.copy()
        elif self.predictionType == "SO3xR2":
            keys = RELION_ANGLES_NAMES + RELION_SHIFTS_NAMES.copy()
        else:
            raise ValueError(f"Error, predictionType option {self.predictionType} is not valid")
        if self.usePredConfidence:
            keys += [RELION_PRED_POSE_CONFIDENCE_NAME]

        mainData = starfile.read(mainStarFname)
        if RELION_PRED_POSE_CONFIDENCE_NAME not in mainData["particles"]:
            mainData["particles"][RELION_PRED_POSE_CONFIDENCE_NAME] = 1.
        predData = starfile.read(predStarFname)
        if RELION_PRED_POSE_CONFIDENCE_NAME not in predData["particles"]:
            predData["particles"][RELION_PRED_POSE_CONFIDENCE_NAME] = 1.

        predAngles = predData["particles"][RELION_ANGLES_NAMES].values
        gtAngles = mainData["particles"][RELION_ANGLES_NAMES].values
        assert predAngles.shape[0] == gtAngles.shape[0], ("Error, mismatch in the number of predicted particles and gt"
                                                          "particles")
        meanAngularError, wMeanAngularError, totalConf = computeAngularError(
            predAngles,gtAngles,
            confidence=mainData["particles"][RELION_ORI_POSE_CONFIDENCE_NAME].values,
            symmetry=symmetry)

        meanAbsShiftError = np.sum(np.abs(predData["particles"][RELION_SHIFTS_NAMES].values -
                                          mainData["particles"][RELION_SHIFTS_NAMES].values), -1)
        result = mainData.copy()
        result["particles"] = result["particles"].copy()
        result["particles"][keys] = predData["particles"][keys]
        if outname is not None:
            starfile.write(result, outname)
        return result, meanAngularError, wMeanAngularError, meanAbsShiftError, totalConf

    def runEvaluate(self, targetName, half0PredsFname, half1PredsFname):

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.wdir is None:
                wdir = tmpdir
            else:
                wdir = self.wdir

            os.makedirs(wdir, exist_ok=True)
            ps0 = ParticlesDataset(targetName, halfset=0, benchmarkDir=self.benchmarkDir)
            ps1 = ParticlesDataset(targetName, halfset=1, benchmarkDir=self.benchmarkDir)

            gt_map, gt_sampling, gt_cor, gt_resolut = self.computeAvgMap(ps0.starFname, ps1.starFname,
                                                                         particlesDir=ps0.datadir, outbasename="gt",
                                                                         symmetry=ps0.symmetry, cleanExisting=False)
            prepStarFname0 = osp.join(tmpdir, "prepStarFname0.star")
            _, meanAngularError0, wMeanAngularError0, meanAbsShiftError0, totalConf0 = \
                self.preparePredStar(ps0.starFname, half0PredsFname, prepStarFname0, ps0.symmetry)
            prepStarFname1 = osp.join(tmpdir, "prepStarFname1.star")
            _, meanAngularError1, wMeanAngularError1, meanAbsShiftError1, totalConf1 = \
                self.preparePredStar(ps1.starFname, half1PredsFname, prepStarFname1, ps1.symmetry)

            pred_map, pred_sampling, pred_corr, pred_resolut = self.computeAvgMap(prepStarFname0, prepStarFname1,
                                                                                  particlesDir=ps0.datadir,
                                                                                  outbasename="pred",
                                                                                  symmetry=ps0.symmetry,
                                                                                  cleanExisting=True)
            if self.verbose:
                print("Computing statistics...")
            mapVsGT_cor, (mapVsGT_resolt, *_) = compute_stats(gt_map, pred_map, samplingRate=gt_sampling,
                                                              resolution_threshold=self.resolution_threshold)

            meanAngularError = .5 * (meanAngularError0 + meanAngularError1)
            wMeanAngularError = (totalConf0 * wMeanAngularError0 + totalConf1 * wMeanAngularError1) / (
                    totalConf0 + totalConf1)
            meanAbsShiftError = .5 * (meanAbsShiftError0 + meanAbsShiftError1)

            metrics = dict(meanAngularError=meanAngularError, wMeanAngularError=wMeanAngularError,
                           meanAbsShiftError=meanAbsShiftError,
                           GT_correlation=gt_cor, GT_resolution=gt_resolut,
                           half2half_resolution=pred_resolut,
                           mapVsGT_correlaton=mapVsGT_cor,
                           mapVsGT_resolution=mapVsGT_resolt)
            if self.verbose:
                print(f"""
> EVALUATION for target:  {targetName})
GT_correlation:           {gt_cor}
GT_resolution (Å):        {gt_resolut}
> RESULTS
mean_angular_error (°):   {meanAngularError} 
w_mean_angular_error (°): {wMeanAngularError} 
mean_abs_shift_error (Å): {meanAbsShiftError}
half2half_resolution (Å): {pred_resolut}
mapVsGT_correlaton:       {mapVsGT_cor}
mapVsGT_resolution (Å):   {mapVsGT_resolt}

            """)
            return metrics


def evaluate(targetName: str, half0PredsFname: str, half1PredsFname: str,
             predictionType: Literal["S2", "SO3", "SO3xR2"], usePredConfidence: bool = True,
             benchmarkDir: str = defaultBenchmarkDir, relionBinDir: Optional[str] = relionBinDir,
             mpirun: Optional[str] = mpirunCmd,
             n_cpus: int = 1, wdir: Optional[str] = None):
    """

    Args:
        targetName (str): The name of the target to use. It is also the basename of \
            the directory where the data is.
        half0PredsFname (str): The starfile for the half0 of the data with predicted poses
        half1PredsFname (str): The starfile for the half1 of the data with predicted poses
        predictionType (Literal[S2, SO3, SO3xR2]): The type of the predicted pose. S2 if only \
            the first two euler angles were predicted (cones). SO3 if all the 3 euler angles were predicted. \
            SO3xR2 if both the euler angles and the particle shifts (translations) were predicted. Depending on the
            type of prediction, the ground-truth values will be used to fill in the missing information
        usePredConfidence (bool): If true, particles are weighted by the predicted confidence at reconstruction time
        benchmarkDir (str): The root directory where the datasets are downloaded.
        relionBinDir (Optional[str]): The Relion bin directory
        mpirun (Optional[str]): The mpriun command. Required inf n_cpus >0
        n_cpus (int): The number of cpus used in the calculations
        wdir (Optional[str]): The directory where partical computations will be stored. Used as cache for the GT \
         computations. If None, a temporary directory will be used instead

    Returns:
            Dict[str,float]: A dictionary with the computed metrics

    """

    half0PredsFname = osp.expanduser(half0PredsFname)
    assert osp.exists(half0PredsFname), f"Error half0PredsFname: {half0PredsFname} not found!"

    half1PredsFname = osp.expanduser(half1PredsFname)
    assert osp.exists(half1PredsFname), f"Error half1PredsFname: {half1PredsFname} not found!"

    with tempfile.TemporaryDirectory() as tmpdir:
        if wdir is None:
            wdir = tmpdir
        evaluator = Evaluator(predictionType, usePredConfidence=usePredConfidence, benchmarkDir=benchmarkDir,
                              relionBinDir=relionBinDir, mpirun=mpirun, n_cpus=n_cpus, wdir=wdir)
        return evaluator.runEvaluate(targetName, half0PredsFname, half1PredsFname)


def _test():
    targetName = "TEST"
    benchmarkDir = "/tmp/cryoSupervisedDataset"
    half0PredsFname = "/tmp/results_half0.star"
    half1PredsFname = "/tmp/results_half1.star"
    evaluate(targetName, half0PredsFname, half1PredsFname, predictionType="SO3", usePredConfidence=True,
             benchmarkDir=benchmarkDir,
             n_cpus=1, wdir="/tmp/pruebaReconstruct")


if __name__ == "__main__":
    # _test()
    from argParseFromDoc import parse_function_and_call

    parse_function_and_call(evaluate)

"""
--targetName TEST --half0PredsFname /tmp/results_half0.star --half1PredsFname /tmp/results.star --predictionType SO3
"""
