"""
This module allows for benchmark entry evaluation given the inferred poses.
"""
import json
import os
import shutil
import subprocess
import os.path as osp
import tempfile
from typing import Literal, Optional, Tuple, Union, Dict

import mrcfile
import numpy as np
import starfile

from cesped.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME, \
    RELION_ORI_POSE_CONFIDENCE_NAME, defaultBenchmarkDir, relionBinDir, mpirunCmd, relionSingularity, RELION_IMAGE_FNAME
from cesped.particlesDataset import ParticlesDataset
from cesped.utils.anglesStats import computeAngularError
from cesped.utils.volumeStats import compute_stats
from cesped.zenodo.downloadFromZenodo import download_mask


class Evaluator():

    def __init__(self, predictionType: Literal["S2", "SO3", "SO3xR2"], usePredConfidence: bool = True,
                 benchmarkDir: Union[str, os.PathLike] = defaultBenchmarkDir,
                 relionBinDir:Optional[str]=relionBinDir, mpirun:Optional[str]=mpirunCmd,
                 relionSingularity:Optional[str]=relionSingularity,
                 n_cpus=1, wdir=None, verbose=True, use_gt_mask=True):
        """

        Args:
            predictionType (Literal[S2, SO3, SO3xR2]): The type of the predicted pose. S2 if only \
                the first two euler angles were predicted (cones). SO3 if all the 3 euler angles were predicted. \
                SO3xR2 if both the euler angles and the particle shifts (translations) were predicted. Depending on the
                type of prediction, the ground-truth values will be used to fill in the missing information            usePredConfidence:
            benchmarkDir (str): The root directory where the datasets are downloaded.
            relionBinDir (Optional[str]): The Relion bin directory. If not provided, it is read from configs/defaultRelionConfig.yaml
            mpirun (Optional[str]): The mpriun command. Required inf n_cpus >1. If not provided, it is read from configs/defaultRelionConfig.yaml
            relionSingularity (Optional[str]): A built singularity image from relionSingularity.def that let you install and run relion easily
            n_cpus (int): The number of cpus used in the calculations
            wdir (str): The directory where intermediate results will be saved. It acts as a cache as well
            verbose (bool): Print to stdout? Set it to true if using via command line
            use_gt_mask (bool): Should computations use ground-truth masks? It is recommended to set it to true. In \
                that case, ground truth mask will be downloaded before computations.
        """

        self.benchmarkDir = benchmarkDir
        self.predictionType = predictionType
        self.usePredConfidence = usePredConfidence
        self.relionBinDir = os.path.abspath(os.path.expanduser(relionBinDir)) if relionBinDir else ""
        self.mpirun = mpirun
        self.relionSingularity = os.path.abspath(os.path.expanduser(relionSingularity)) if relionSingularity else None
        self.n_cpus = n_cpus
        self.wdir = wdir
        self.verbose = verbose
        self.use_gt_mask = use_gt_mask

    def reconstruct(self, starFname: str, outname: str, particlesDir: str, symmetry: str = "c1",
                    cleanExisting: bool = True) -> Tuple[np.ndarray, float]:
        """
        Reconstruct a volume from the information contained in a starfile.
        Args:
            starFname: The starfile with the information to be reconstructed
            outname: The .mrc file where the reconstructed volume will be saved
            particlesDir (str): particlesDir
            symmetry: The point symmetry of the dataset.
            cleanExisting: True if an existing file should be removed
        Returns:
            Tuple[np.ndarray, float]:
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exists = osp.isfile(outname)
            if exists and cleanExisting:
                os.unlink(outname)
                exists = False
            if not exists:
                cmd = []
                wdir = particlesDir if particlesDir else osp.dirname(starFname)
                if self.relionSingularity:
                    cmd += [self.relionSingularity, str(self.n_cpus)]
                    destination = osp.join(tmpdir, osp.basename(wdir))
                    shutil.copytree(wdir, destination)
                    baseStar = osp.basename(starFname)
                    shutil.copy(starFname, destination)
                    starFname = osp.join(destination, baseStar)
                    wdir = destination
                else:
                    if self.n_cpus > 1:
                        cmd += self.mpirun.split() + ["-np", str(self.n_cpus),
                                                      osp.join(self.relionBinDir, "relion_reconstruct_mpi")]
                    else:
                        cmd += [osp.join(self.relionBinDir, "relion_reconstruct")]

                with tempfile.NamedTemporaryFile(suffix=".mrc") as f:
                    cmd += ["--i", starFname, "--o", f.name, "--ctf", "--sym", symmetry.lower(), "--pad", "2.0"]
                    if self.usePredConfidence:
                        data = starfile.read(starFname)
                        if RELION_PRED_POSE_CONFIDENCE_NAME in data["particles"]:
                            cmd += ["--fom_weighting"]
                    if self.verbose:
                        print(" ".join(cmd))
                    subprocess.run(cmd, cwd=wdir, check=True, capture_output=not self.verbose)
                    f.seek(0)
                    shutil.copyfile(f.name, outname)
            else:
                print(f"Reusing file {outname}")
            with mrcfile.open(outname) as f:
                data = f.data.copy()
                sampling_rate = float(f.voxel_size.x)
        return data, sampling_rate

    def computeMask(self, volFname: str, maskFname: str, lowpass_res: float = 15.0, volThr: float = 0.015,
                    mask_pix_exten: int = 3, mask_pix_soft: int = 4):
        """
        Computes a mask for a volume and stores it in maskFname using relion_mask_create

        Args:
            starFname: The starfile with the information to be reconstructed
            maskFname: The .mrc file where the mask volume will be saved
            lowpass_res (float): Low pass filtering before applying any opperation, in Angstroms
            volThr (float): volume threshold to compute the mask
            mask_pix_exten (int): How many pixels the mask should be extended after thresholding
            mask_pix_soft (int): How many pixels the mask should be softened after extension
        Returns:
            Tuple[np.ndarray, float]:
        """
        cmd = (f"{osp.join(self.relionBinDir, 'relion_mask_create')}  --o {maskFname} "
               f"--i  {volFname} --ini_threshold  {volThr} "
               f"--extend_inimask  {mask_pix_exten} "
               f"--width_soft_edge {mask_pix_exten} mask_pix_soft {mask_pix_soft} "
               f"--lowpass  {lowpass_res} --j {self.n_cpus} ")

        if self.verbose:
            print(" ".join(cmd))
        subprocess.run(cmd.split(), check=True, capture_output=not self.verbose)
        with mrcfile.open(maskFname) as f:
            return f.data.copy()

    def computeAvgMap(self, starFname0: str, starFname1: str, particlesDir: str,
                      outbasename: str, symmetry: str, cleanExisting: bool = True, mask: Optional[str] = None):
        """
        Reconstruct each of the half-datasets starFname0 and starFname0 and averages the reconstruction to
        obtain the final average map
        Args:
            starFname0 (str): The filename with the predicted poses for the first half of the dataset
            starFname1 (str):
            particlesDir (str):
            outbasename (str):
            symmetry (str):
            cleanExisting (bool): True if an existing file should be removed
            maskFname (str): The Fname with a mask
        Returns:

        """
        name0 = osp.join(self.wdir, outbasename + "_0.mrc")
        data0, sr0 = self.reconstruct(starFname0, name0, particlesDir, symmetry=symmetry, cleanExisting=cleanExisting)

        name1 = osp.join(self.wdir, outbasename + "_1.mrc")
        data1, sr1 = self.reconstruct(starFname1, name1, particlesDir, cleanExisting=cleanExisting)
        assert sr0 == sr1, "Error, the sampling rate of the two datasets is different"

        (corr, m_corr), resolt = compute_stats(name0, name1, maskOrFname=mask, samplingRate=sr0,
                                               resolution_threshold=0.143)

        resolt0143, m_resolt0143 = resolt[:2]

        _, resolt05 = compute_stats(name0, name1, maskOrFname=mask, samplingRate=sr0,
                                    resolution_threshold=0.5)
        resolt05, m_resolt05 = resolt05[:2]

        outvol = .5 * (data0 + data1)
        if not outbasename.endswith(".mrc"):
            outbasename += ".mrc"
        mrcfile.write(osp.join(self.wdir, outbasename), outvol, voxel_size=sr0, overwrite=True)

        if self.use_gt_mask:
            corr = m_corr
            resolt0143 = m_resolt0143
            resolt05 = m_resolt05
        return outvol, sr0, corr, resolt0143, resolt05

    def preparePredStar(self, referStarFname: str, predStarFname: str, outname: str, symmetry: str):
        """
        Transplants pose parameters not predicted by the model from the ground-truth pose data. Then compute
        angular errors comparing transplated poses with ground-truth poses

        Args:
            referStarFname (str): The file with the ground-truth poses
            predStarFname (str): The file with the predicted poses
            outname (str): The name for the transplated poses.
            symmetry (str): The point symmetry of the dataset.

        Returns:

        """

        if self.predictionType == "S2":
            predicted_keys = RELION_ANGLES_NAMES[:2].copy()
        elif self.predictionType == "SO3":
            predicted_keys = RELION_ANGLES_NAMES.copy()
        elif self.predictionType == "SO3xR2":
            predicted_keys = RELION_ANGLES_NAMES.copy() + RELION_SHIFTS_NAMES.copy()
        else:
            raise ValueError(f"Error, predictionType option {self.predictionType} is not valid")
        if self.usePredConfidence:
            predicted_keys += [RELION_PRED_POSE_CONFIDENCE_NAME]

        referData = starfile.read(referStarFname)
        if RELION_PRED_POSE_CONFIDENCE_NAME not in referData["particles"]:
            referData["particles"][RELION_PRED_POSE_CONFIDENCE_NAME] = 1.
        predData = starfile.read(predStarFname)
        if RELION_PRED_POSE_CONFIDENCE_NAME not in predData["particles"]:
            predData["particles"][RELION_PRED_POSE_CONFIDENCE_NAME] = 1.

        referData["particles"] = referData["particles"].sort_values(by=RELION_IMAGE_FNAME).reset_index(drop=True)
        predData["particles"] = predData["particles"].sort_values(by=RELION_IMAGE_FNAME).reset_index(drop=True)
        assert referData["particles"][RELION_IMAGE_FNAME].equals(predData["particles"][RELION_IMAGE_FNAME]), \
            (f"Error, there is a mismatch between the ids of the predicted {predStarFname} data and the benchmark"
             f" data {referStarFname}")

        predAngles = predData["particles"][RELION_ANGLES_NAMES].values
        gtAngles = referData["particles"][RELION_ANGLES_NAMES].values
        assert predAngles.shape[0] == gtAngles.shape[0], ("Error, mismatch in the number of predicted particles and gt"
                                                          f"particles for {referStarFname} {predStarFname}")
        meanAngularError, wMeanAngularError, totalConf = computeAngularError(
            predAngles, gtAngles,
            confidence=referData["particles"][RELION_ORI_POSE_CONFIDENCE_NAME].values,
            symmetry=symmetry)

        n_elems = len(referData["particles"][RELION_SHIFTS_NAMES].values)
        shiftsRMSE = np.linalg.norm(predData["particles"][RELION_SHIFTS_NAMES].values -
                                    referData["particles"][RELION_SHIFTS_NAMES].values) / np.sqrt(n_elems)

        result = referData.copy()
        result["particles"] = result["particles"].copy()
        result["particles"][predicted_keys] = predData["particles"][predicted_keys]
        if outname is not None:
            starfile.write(result, outname)
        return result, meanAngularError, wMeanAngularError, shiftsRMSE, totalConf

    def runEvaluate(self, targetName, half0PredsFname, half1PredsFname, rm_prev_reconstructions, ignore_symmetry):

        with (tempfile.TemporaryDirectory() as tmpdir):
            if self.wdir is None:
                wdir = tmpdir
            else:
                wdir = self.wdir
                os.makedirs(wdir, exist_ok=True)

            os.makedirs(wdir, exist_ok=True)
            ps0 = ParticlesDataset(targetName, halfset=0, benchmarkDir=self.benchmarkDir)
            ps1 = ParticlesDataset(targetName, halfset=1, benchmarkDir=self.benchmarkDir)

            # GET MASK
            if self.use_gt_mask:
                print("Downloading mask")
                mask_fname = osp.join(wdir, "mask.mrc")
                download_mask(targetName, mask_fname)
            else:
                mask_fname = None

            symmetry = ps0.symmetry if not ignore_symmetry else "C1"
            gt_map, gt_sampling, gt_cor, gt_resolt0143, gt_resolt05 = self.computeAvgMap(ps0.starFname, ps1.starFname,
                                                                                         particlesDir=ps0.datadir,
                                                                                         outbasename="gt",
                                                                                         symmetry=symmetry,
                                                                                         cleanExisting=False,
                                                                                         mask=mask_fname)
            prepStarFname0 = osp.join(tmpdir, "prepStarFname0.star")
            _, meanAngularError0, wMeanAngularError0, shiftsRMSE0, totalConf0 = \
                self.preparePredStar(ps0.starFname, half0PredsFname, prepStarFname0, symmetry)
            prepStarFname1 = osp.join(tmpdir, "prepStarFname1.star")
            _, meanAngularError1, wMeanAngularError1, shiftsRMSE1, totalConf1 = \
                self.preparePredStar(ps1.starFname, half1PredsFname, prepStarFname1, symmetry)

            meanAngularError = .5 * (meanAngularError0 + meanAngularError1)
            wMeanAngularError = (totalConf0 * wMeanAngularError0 + totalConf1 * wMeanAngularError1) / (
                    totalConf0 + totalConf1)
            shiftsRMSE = .5 * (shiftsRMSE0 + shiftsRMSE1)

            assert ps0.datadir == ps1.datadir
            pred_map, pred_sampling, pred_corr, pred_resolut, pred_resolt05 = self.computeAvgMap(
                prepStarFname0, prepStarFname1,
                particlesDir=ps0.datadir, outbasename="pred",
                symmetry=symmetry, cleanExisting=rm_prev_reconstructions, mask=mask_fname)

            if self.verbose:
                print("Computing statistics...")
            mapVsGT_cor, (mapVsGT_resolt, m_mapVsGT_resolt, *_) = compute_stats(gt_map, pred_map,
                                                                                samplingRate=gt_sampling,
                                                                                resolution_threshold=0.143,
                                                                                maskOrFname=mask_fname)
            _, (mapVsGT_resolt05, m_mapVsGT_resolt05, *_) = compute_stats(gt_map, pred_map,
                                                                          samplingRate=gt_sampling,
                                                                          resolution_threshold=0.5,
                                                                          maskOrFname=mask_fname)
            cor_diff = (gt_cor - mapVsGT_cor) #/ gt_cor

            res_diff05 = mapVsGT_resolt05 - gt_resolt05
            res_diff = mapVsGT_resolt - gt_resolt0143


            metrics = dict(meanAngularError=meanAngularError, wMeanAngularError=wMeanAngularError,
                           shiftsRMSE=shiftsRMSE,
                           GT_correlation=gt_cor,
                           GT_resolution0143=gt_resolt0143, GT_resolution05=gt_resolt05,
                           half2half_resolution=pred_resolut, half2half_resolution05=pred_resolt05,
                           half2half_correlation=pred_corr,
                           mapVsGT_correlaton_masked=mapVsGT_cor[1], mapVsGT_correlaton_unmasked=mapVsGT_cor[0],
                           mapVsGT_resolution=mapVsGT_resolt, mapVsGT_resolution05=mapVsGT_resolt05,
                           )

            with open(osp.join(self.wdir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            report_str = f"""
> EVALUATION for target:                  {targetName}
GT_correlation:                           {gt_cor}
GT_resolution (Å) (th=0.143, 0.5):        {gt_resolt0143}  {gt_resolt05}
> RESULTS
#Predictions
mean_angular_error (°):                   {meanAngularError} 
w_mean_angular_error (°):                 {wMeanAngularError} 
shifts_RMSE (Å):                          {shiftsRMSE}
#Reconstruction
half2half_correlation:                    {pred_corr}
half2half_resolution (Å) (th=0.143, 0.5): {pred_resolut}  {pred_resolt05}
mapVsGT_correlaton (masked, unmasked):    {"  ".join(reversed([str(x) for x in mapVsGT_cor]))}
mapVsGT_resolution (Å) (th=0.143, 0.5)    {mapVsGT_resolt}  {mapVsGT_resolt05}
#Reconstruction differences
cor_diff (%) (masked, unmasked):          {"  ".join(reversed([str(x * 100) for x in cor_diff]))}
res_diff (Å) th=0.143, 0.5):              {res_diff}  {res_diff05}
            """

            if self.verbose:
                print(report_str)
            return metrics


def evaluate(targetName: str, half0PredsFname: str, half1PredsFname: str,
             predictionType: Literal["S2", "SO3", "SO3xR2"], usePredConfidence: bool = True,
             benchmarkDir: str = defaultBenchmarkDir, relionBinDir: Optional[str] = relionBinDir,
             mpirun: Optional[str] = mpirunCmd, relionSingularity: Optional[str] = relionSingularity,
             rm_prev_reconstructions: bool = True,
             ignore_symmetry: bool = False, use_gt_mask:bool=True,
             n_cpus: int = 1, outdir: Optional[str] = None) -> Dict[str, float]:
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
        relionBinDir (Optional[str]): The Relion bin directory. If not provided, it is read from configs/defaultRelionConfig.yaml
        mpirun (Optional[str]): The mpriun command. Required inf n_cpus >1. If not provided, it is read from configs/defaultRelionConfig.yaml
        relionSingularity (Optional[str]): A built singularity image from relionSingularity.def that let you install and run relion easily
        rm_prev_reconstructions (bool): If True, the reconstructions from predicted angles stored in outdir will \
        be recomputed
        ignore_symmetry (bool): If True, reconstruct ignoring symmetry
        use_gt_mask (bool): If True, computations are performed using the ground truth mask that will be downloaded.
        n_cpus (int): The number of cpus used in the calculations
        outdir (Optional[str]): The directory where computations will be stored. Used as cache for the GT \
         computations. If None, a temporary directory will be used instead

    Returns:
            Dict[str,float]: A dictionary with the computed metrics

    """

    half0PredsFname = osp.expanduser(half0PredsFname)
    assert osp.exists(half0PredsFname), f"Error half0PredsFname: {half0PredsFname} not found!"

    half1PredsFname = osp.expanduser(half1PredsFname)
    assert osp.exists(half1PredsFname), f"Error half1PredsFname: {half1PredsFname} not found!"

    with tempfile.TemporaryDirectory() as tmpdir:
        if outdir is None:
            outdir = tmpdir
        evaluator = Evaluator(predictionType, usePredConfidence=usePredConfidence, benchmarkDir=benchmarkDir,
                              relionBinDir=relionBinDir, mpirun=mpirun, relionSingularity=relionSingularity,
                              n_cpus=n_cpus, wdir=outdir,
                              use_gt_mask=use_gt_mask)
        return evaluator.runEvaluate(targetName, half0PredsFname, half1PredsFname,
                                     rm_prev_reconstructions, ignore_symmetry)


def _test():
    targetName = "TEST"
    benchmarkDir = "/tmp/cryoSupervisedDataset"
    half0PredsFname = "/tmp/results_half0.star"
    half1PredsFname = "/tmp/results_half1.star"
    evaluate(targetName, half0PredsFname, half1PredsFname, predictionType="SO3", usePredConfidence=True,
             benchmarkDir=benchmarkDir, n_cpus=1, outdir="/tmp/pruebaReconstruct")


if __name__ == "__main__":
    # _test()
    from argParseFromDoc import parse_function_and_call
    parse_function_and_call(evaluate)

"""
--targetName TEST --half0PredsFname /tmp/results_half0.star --half1PredsFname /tmp/results.star --predictionType SO3
"""
