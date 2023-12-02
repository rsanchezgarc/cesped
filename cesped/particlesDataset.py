from itertools import chain

import functools
import json
import os.path
import warnings
from os import PathLike
import os.path as osp

from tqdm import tqdm
from typing import Union, Literal, Optional, List, Tuple, Any, Dict

warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it "
                                  "is not possible to uniquely determine all angles.")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are"
                                          " still Beta.")


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from starstack.particlesStar import ParticlesStarSet
from torch.utils.data import Dataset

from cesped.constants import RELION_EULER_CONVENTION, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, \
    RELION_ORI_POSE_CONFIDENCE_NAME, RELION_PRED_POSE_CONFIDENCE_NAME, default_configs_dir, defaultBenchmarkDir

from cesped.datamanager.augmentations import Augmenter
from cesped.zenodo.bechmarkUrls import NAME_PARTITION_TO_RECORID
from cesped.datamanager.ctf import apply_ctf
from cesped.utils.tensors import data_to_numpy
from cesped.zenodo.downloadFromZenodo import download_record, getDoneFname

"""
This module implements the ParticlesDataset class. A Pytorch Dataset for dealing with Cryo-EM particles 
in the CESPED benchmark
"""


class ParticlesDataset(Dataset):
    """
    ParticlesDataset: A Pytorch Dataset for dealing with Cryo-EM particles in the CESPED benchmark.<br>
    It can download data automatically

    ```python
    #Loads the halfset 0 for the benchmark entry named "TEST"
    ds = ParticlesDataset(targetName="TEST", halfset=0, benchmarkDir="/tmp/cryoSupervisedDataset/")
    ```
    and each particle can be acessed as usually
    ```python
    img, rotMat, xyShiftAngs, confidence, metadata = ds[0]
    ```
    <br>
    """

    def __init__(self, targetName: Union[PathLike, str],
                 halfset: Literal[0, 1],
                 benchmarkDir: str = defaultBenchmarkDir,
                 image_size: Optional[int] = None,
                 apply_perImg_normalization: bool = True,
                 ctf_correction: Literal["none", "phase_flip"] = "phase_flip",
                 image_size_factor_for_crop: float = 0.25,
                 ):
        """
        ##Builder

        Args:
            targetName (Union[PathLike, str]): The name of the target to use. It is also the basename of \
            the directory where the data is.
            halfset (Literal[0, 1]): The second parameter.
            benchmarkDir (str): The root directory where the datasets are downloaded.
            image_size (Optional[int]): The final size of the image (after cropping). If None, keep the original size
            apply_perImg_normalization (bool): Apply cryo-EM per-image normalization. I = (I-noiseMean)/noiseStd.
            ctf_correction (Literal[none, phase_flip]): phase_flip will correct amplitude inversion due to defocus
            image_size_factor_for_crop (float): Fraction of the image size to be cropped. Final size of the image \
            is origSize*(1-image_size_factor_for_crop). It is important because particles in cryo-EM tend to \
            be only 50% to 25% of the total area of the image.

        """

        super().__init__()
        assert halfset in [0, 1], f"Error, data halfset should be 0 or 1. Currently it is {halfset}"
        self.targetName = targetName
        self.halfset = halfset
        self.benchmarkDir = osp.expanduser(benchmarkDir)
        self._image_size = image_size
        self.apply_perImg_normalization = apply_perImg_normalization
        assert ctf_correction in ["none", "phase_flip"]
        self.ctf_correction = ctf_correction

        self._particles = None
        self._symmetry = None
        self._lock = torch.multiprocessing.RLock()
        self._augmenter = None

        assert 0 <= image_size_factor_for_crop <= 0.5
        self.image_size_factor_for_crop = image_size_factor_for_crop


    @property
    def image_size(self):
        """The image size in pixels"""
        if self._image_size is None:
            return self.particles.particle_shape[-1]
        else:
            return self._image_size
    @property
    def datadir(self):
        """ the directory where the target is stored"""
        return osp.join(self.benchmarkDir, self.targetName)

    @property
    def starFname(self):
        """ the particles star filename"""
        return osp.join(self.datadir, f"particles_{self.halfset}.star")

    @property
    def stackFname(self):
        """ the particles mrcs filename"""
        return osp.join(self.datadir, f"particles_{self.halfset}.mrcs")

    @property
    def particles(self):
        """
        a starstack.particlesStar.ParticlesStarSet representing the loaded particles
        """
        if self._particles is None:
            if not self._is_avaible():
                self._download()
            self._particles = ParticlesStarSet(starFname=self.starFname, particlesDir=self.datadir)
        return self._particles
    @property
    def symmetry(self):
        """The point symmetry of the dataset"""
        if not self._is_avaible():
            self._download()
        if self._symmetry is None:
            with open(osp.join(self.datadir, f"info_{self.halfset}.json")) as f:
                self._symmetry = json.load(f)["symmetry"].upper()
        return self._symmetry

    @property
    def sampling_rate(self):
        """The particle image sampling rate in A/pixels"""
        return self.particles.sampling_rate

    @property
    def augmenter(self):
        """The data augmentator object to be applied"""
        return self._augmenter

    @augmenter.setter
    def augmenter(self, augmenterObj:Augmenter):
        """

        Args:
            augmenter: he data augmentator object to be applied

        """
        self._augmenter = augmenterObj

    @classmethod
    def addNewEntryLocally(cls, starFname: Union[str, PathLike], particlesRootDir: Union[str, PathLike],
                           newTargetName: Union[str, PathLike], halfset: Literal[0, 1], symmetry:str,
                           benchmarkDir: Union[str, PathLike] = defaultBenchmarkDir):
        """

        Adds a new dataset to the local copy of the CESPED benchmark. It rearanges the starfile content and copies the
        stack of images. Notice that it duplicates your particle images.
        Args:
            starFname (Union[str, PathLike]): The star filename with the particles to be added to the local benchmark
            particlesRootDir (Union[str, PathLike]): The root directory that is referred in the starFname (e.g. Relion project dir).
            newTargetName (Union[str, PathLike]): The name of the target to use.
            halfset (Literal[0, 1]): The second parameter.
            symmetry (str): The point symmetry of the dataset
            benchmarkDir (str): The root directory where the datasets are downloaded.

        """

        stack = ParticlesStarSet(starFname=starFname, particlesDir=particlesRootDir)
        assert isinstance(stack[len(stack) - 1][0], np.ndarray), "Error, there is some problem reading your data"
        newTargetDir = os.path.join(benchmarkDir, newTargetName)
        os.makedirs(newTargetDir, exist_ok=True)
        newStarFname = os.path.join(newTargetDir, f"particles_{halfset}.star")
        newStackFname = os.path.join(newTargetDir, f"particles_{halfset}.mrcs")
        stack.save(newStarFname, newStackFname)

        with open(os.path.join(newTargetDir, f"info_{halfset}.json"), "w") as f: #TODO: move name template to download/uploadFromZenodo
            json.dump({"symmetry": symmetry.upper()}, f)

        with open(getDoneFname(newTargetDir, halfset), "w") as f:
            f.write("%s\n" % newTargetName)

    @classmethod
    def getCESPEDEntries(cls) -> List[Tuple[str, int]]:
        """
        Returns the list of available entries in benchmarkDir

        Returns:
            List[Tuple[str,int]]: the list of available entries in benchmarkDir

        """
        return list(NAME_PARTITION_TO_RECORID.keys())
    @classmethod
    def getLocallyAvailableEntries(cls, benchmarkDir: Union[str, PathLike] = defaultBenchmarkDir) -> List[Tuple[str, int]]:
        """
        Returns the list of available entries in benchmarkDir
        Args:
            benchmarkDir (str): The root directory where the datasets are downloaded.

        Returns:
            List[Tuple[str,int]]: the list of available entries in benchmarkDir

        """
        avail = []
        for dirname in os.listdir(benchmarkDir):
            if os.path.isfile(dirname):
                continue
            if os.path.exists(os.path.join(benchmarkDir, dirname, "particles_0.mrcs")):
                avail.append((dirname, 0))
            if os.path.exists(os.path.join(benchmarkDir, dirname, "particles_1.mrcs")):
                avail.append((dirname, 1))
        return avail

    def _download(self):
        assert (self.targetName, self.halfset) in NAME_PARTITION_TO_RECORID, (f"Error, unknown target and/or "
                                                                              f"halfset {self.targetName} {self.halfset}")
        download_record(NAME_PARTITION_TO_RECORID[self.targetName, self.halfset],
                        destination_dir=self.datadir)
        #Validation
        pset = ParticlesStarSet(starFname=self.starFname, particlesDir=self.datadir)
        for i in tqdm(range(len(pset)), desc="Checking downloaded dataset"):
            img, md = pset[i]
            assert len(img.shape) == 2, f"Error, there were problems downloading the target {self.targetName}"

    def _is_avaible(self):
        return osp.isfile(getDoneFname(self.datadir,
                                       NAME_PARTITION_TO_RECORID.get((self.targetName, self.halfset), self.halfset)))

    @functools.lru_cache(1)
    def _getParticleNormalizationMask(self, particleNumPixels: int,
                                      normalizationRadiusPixels: Optional[int] = None,
                                      device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
        """
        Gets a mask with 1s in the corners and 0 for the center

        Args:
            particleNumPixels: The number of pixels of the particle
            normalizationRadiusPixels: The number of pixels of the radius of the particle (inner circle)
            device: The torch device

        Returns:
            torch.Tensor: The mask with 1s in the corners of the image and 0s for the center
        """

        radius = particleNumPixels // 2
        if normalizationRadiusPixels is None:
            normalizationRadiusPixels = radius
        ies, jes = torch.meshgrid(
            torch.linspace(-1 * radius, 1 * radius, particleNumPixels, dtype=torch.float32),
            torch.linspace(-1 * radius, 1 * radius, particleNumPixels, dtype=torch.float32),
            indexing="ij"
        )
        r = (ies ** 2 + jes ** 2) ** 0.5
        _normalizationMask = (r > normalizationRadiusPixels)
        _normalizationMask = _normalizationMask.to(device)
        return _normalizationMask

    def _normalize(self, img):
        """

        Args:
            img: 1XSxS tensor

        Returns:

        """
        backgroundMask = self._getParticleNormalizationMask(img.shape[-1])
        noiseRegion = img[:, backgroundMask]
        meanImg = noiseRegion.mean()
        stdImg = noiseRegion.std()
        return (img - meanImg) / stdImg

    def getIdx(self, item: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """

        Args:
            item: a particle index

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]. The raw image before any pre-processing \
             as an LxL np.array and the metadata as dictionary
        """
        with self._lock:
            try:
                return self.particles[item]
            except ValueError:
                print(f"Error retrieving item {item}")
                raise

    def resizeImage(self, img):

        ori_pixelSize = float(self.particles.optics_md["rlnImagePixelSize"].item())
        particle_size_after_crop = int(self.particles.optics_md["rlnImageSize"].item() * (1 - self.image_size_factor_for_crop))

        desired_sampling_rate = float(self.particles.optics_md["rlnImagePixelSize"].item() * particle_size_after_crop / self.image_size)

        img, pad_info, crop_info = resize_and_padCrop_tensorBatch(img.unsqueeze(0),
                                                                  ori_pixelSize,
                                                                  desired_sampling_rate, self.image_size,
                                                                  padding_mode="constant")
        img = img.squeeze(0)
        return img
    def __getitem(self, item):
        img, md_row = self.getIdx(item)
        iid = md_row["rlnImageName"]

        img = torch.FloatTensor(img).unsqueeze(0)

        if self.apply_perImg_normalization:
            img = self._normalize(img)

        if self.ctf_correction != "none":
            ctf, wimg = apply_ctf(img, float(self.particles.optics_md["rlnImagePixelSize"].item()),
                                  dfu=md_row["rlnDefocusU"], dfv=md_row["rlnDefocusV"],
                                  dfang=md_row["rlnDefocusAngle"],
                                  volt=float(self.particles.optics_md["rlnVoltage"][0]),
                                  cs=float(self.particles.optics_md["rlnSphericalAberration"][0]),
                                  w=float(self.particles.optics_md["rlnAmplitudeContrast"][0]),
                                  mode=self.ctf_correction)
            wimg = torch.clamp(wimg, img.min(), img.max())
            wimg = torch.nan_to_num(wimg, nan=img.mean())
            # img = torch.concat([img, wimg], dim=0)
            img = wimg

        if img.isnan().any():
            raise RuntimeError(f"Error, img with idx {item} is NAN")

        img = self.resizeImage(img)

        degEuler = torch.FloatTensor([md_row[name] for name in RELION_ANGLES_NAMES])
        xyShiftAngs = torch.FloatTensor([md_row[name] for name in RELION_SHIFTS_NAMES])

        if self.augmenter is not None:
            img, degEuler, shift, _ = self.augmenter(img, #1xSxS image expected
                                                     degEuler,
                                                     shiftFraction=xyShiftAngs / (self.image_size * self.sampling_rate))
            xyShiftAngs = shift * (self.image_size*self.sampling_rate)

        r = R.from_euler(RELION_EULER_CONVENTION, degEuler, degrees=True)
        if self.symmetry.upper() != "C1":
            r = r.reduce(R.create_group(self.symmetry.upper()))
        rotMat = r.as_matrix()
        rotMat = torch.FloatTensor(rotMat)
        confidence = torch.FloatTensor([md_row.get(RELION_ORI_POSE_CONFIDENCE_NAME, 1)])

        return iid, img, (rotMat, xyShiftAngs, confidence), md_row.to_dict()

    def __getitem__(self, item):
        return self.__getitem(item)


    def __len__(self):
        return len(self.particles)

    def updateMd(self, ids: List[str], angles: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 shifts: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 confidence: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 angles_format: Literal["rotmat", "zyzEulerDegs"] = "rotmat",
                 shifts_format: Literal["Angst"] = "Angst"):
        """
        Updates the metadata of the particles with selected ids

        Args:
            ids (List[str]): The ids of the entries to be updated e.g. ["1@particles_0.mrcs", "2@particles_0.mrcs]
            angles (Optional[Union[torch.Tensor, np.ndarray]]): The particle pose angles to update
            shifts (Optional[Union[torch.Tensor, np.ndarray]]): The particle shifts
            confidence (Optional[Union[torch.Tensor, np.ndarray]]): The prediction confidence
            angles_format (Literal[rotmat, zyzEulerDegs]): The format for the argument angles
            shifts_format (Literal[rotmat, zyzEulerDegs]): The format for the argument shifts

        """

        assert angles_format in ["rotmat", "zyzEulerDegs"], \
            'Error, angle_format should be in ["rotmat", "zyzEulerDegs"]'

        assert shifts_format in ["Angst"], \
            'Error, shifts_format should be in ["Angst"]'

        col2val = {}

        if angles is not None:
            angles = data_to_numpy(angles)
            if angles_format == "rotmat":
                r = R.from_matrix(angles)
                rots, tilts, psis = r.as_euler(RELION_EULER_CONVENTION, degrees=True).T
            else:
                rots, tilts, psis = [angles[:, i] for i in range(3)]

            col2val.update({ #RELION_ANGLES_NAMES
                RELION_ANGLES_NAMES[0]: rots,
                RELION_ANGLES_NAMES[1]: tilts,
                RELION_ANGLES_NAMES[2]: psis
            })

        if shifts is not None:
            shifts = data_to_numpy(shifts)
            col2val.update({
                RELION_SHIFTS_NAMES[0]: shifts[:, 0],
                RELION_SHIFTS_NAMES[1]: shifts[:, 1],
            })

        if confidence is not None:
            confidence = data_to_numpy(confidence)
            col2val.update({
                RELION_PRED_POSE_CONFIDENCE_NAME: confidence,
            })
        assert col2val, "Error, no editing values were provided"
        self.particles.updateMd(ids=ids, colname2change=col2val)

    def saveMd(self, fname: Union[str, os.PathLike], overwrite: bool = True):
        """
        Saves the metadata of the current PartcilesDataset as a starfile
        Args:
            fname: The name of the file were the metadata will be saved
            overwrite: If true, overwrites the file fname if already exists

        """
        assert fname.endswith(".star"), "Error, metadata files will be saved as star files. Change extension to .star"
        self.particles.save(starFname=fname, overwrite=overwrite)


def resize_and_padCrop_tensorBatch(array, current_sampling_rate, new_sampling_rate, new_n_pixels=None, padding_mode='reflect'):

    ndims = array.ndim - 2
    if isinstance(array, np.ndarray):
        wasNumpy = True
        array = torch.from_numpy(array)

    else:
        wasNumpy = False

    if isinstance(current_sampling_rate, tuple):
        current_sampling_rate = torch.tensor(current_sampling_rate)
    if isinstance(new_sampling_rate, tuple):
        new_sampling_rate = torch.tensor(new_sampling_rate)

    scaleFactor = current_sampling_rate / new_sampling_rate
    if isinstance(scaleFactor, (int,float)):
        scaleFactor = (scaleFactor,) * ndims
    else:
        scaleFactor = tuple(scaleFactor)
    # Resize the array
    if ndims == 2:
        mode = 'bilinear'
    elif ndims == 3:
        mode = 'trilinear'
    else:
        raise ValueError(f"Option not valid. ndims={ndims}")
    resampled_array = torch.nn.functional.interpolate(array, scale_factor=scaleFactor, mode=mode, antialias=False)
    pad_width = []
    crop_positions = []
    if new_n_pixels is not None:
        if isinstance(new_n_pixels, int):
            new_n_pixels = [new_n_pixels] * ndims
        for i in range(ndims):
            new_n_pix = new_n_pixels[i]
            old_n_pix = resampled_array.shape[i+2]
            if new_n_pix < old_n_pix:
                # Crop the tensor
                crop_start = (old_n_pix - new_n_pix) // 2
                resampled_array = resampled_array.narrow(i+2, crop_start, new_n_pix)
                crop_positions.append((crop_start, crop_start+new_n_pix))
            elif new_n_pix > old_n_pix:
                # Pad the tensor
                pad_before = (new_n_pix - old_n_pix) // 2
                pad_after = new_n_pix - old_n_pix - pad_before
                pad_width.extend((pad_before, pad_after))

        if len(pad_width) > 0:
            resampled_array = torch.nn.functional.pad(resampled_array, pad_width, mode=padding_mode)


    if wasNumpy:
        resampled_array = resampled_array.numpy()
    return resampled_array, pad_width, crop_positions


if __name__ == "__main__":
    from argparse import ArgumentParser
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml"))

    parser = ArgumentParser(description="Dataset utility")
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    # Create parser for 'visualize' mode
    visualize_parser = subparsers.add_parser("visualize", help="Run dataset visualization")

    visualize_parser.add_argument("-t", "--targetName", help="The target to use", type=str, required=True)
    visualize_parser.add_argument("-p", "--halfset", help="The halfset to use", choices=["0", "1"], default="0")
    visualize_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str,
                                  default=defaultBenchmarkDir)
    visualize_parser.add_argument("-s", "--image_size", help="The desired image size", type=int,
                                  default=cfg.data.image_size)
    visualize_parser.add_argument("-c", "--image_size_factor_for_crop", help="Percentage of image to crop", type=float,
                                  default=cfg.data.image_size_factor_for_crop)
    visualize_parser.add_argument("-f", "--phase_flippling", help="Apply phase flippling", action="store_true")
    visualize_parser.add_argument("-i", "--channels_to_show", help="Channels of the images to show", type=int,
                                  nargs="+")

    # Create parser for 'add_entry' mode
    add_entry_parser = subparsers.add_parser("add_entry", help="Run add new entry locally")
    add_entry_parser.add_argument("--newTargetName", help="The name of the new target to use", type=str,
                                  required=True)
    add_entry_parser.add_argument("-p", "--halfset", help="The halfset to use", choices=["0", "1"],
                                  required=True)
    add_entry_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str,
                                  default=defaultBenchmarkDir)
    add_entry_parser.add_argument("--starFname",
                                  help="The star filename with the particles to be added to the local benchmark",
                                  type=str, required=True)
    add_entry_parser.add_argument("--particlesRootDir", help="The root directory referred to in the starFname",
                                  type=str, required=True)
    add_entry_parser.add_argument("--symmetry", help="The point symmetry of the dataset", type=str, required=True)


    list_entries_parser = subparsers.add_parser("list_entries", help="List available entries")
    list_entries_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str, default=defaultBenchmarkDir)
    list_entries_parser.add_argument("--remote", help="List remote entries instead of local", action="store_true")


    args = parser.parse_args()

    if args.mode == "visualize":
        # Run dataset visualization
        ps = ParticlesDataset(targetName=args.targetName,
                              halfset=int(args.halfset),
                              benchmarkDir=args.benchmarkDir,
                              image_size=args.image_size,
                              ctf_correction="phase_flip" if args.phase_flippling else "none",
                              image_size_factor_for_crop=args.image_size_factor_for_crop
                              )
        import matplotlib.pyplot as plt

        channels_to_show = args.channels_to_show if args.channels_to_show else [0]
        for elem in ps:
            iid, img, *_ = elem
            assert 1 <= len(channels_to_show) <= 4, "Error, at least one channel required and no more than 4"
            f, axes = plt.subplots(1, len(channels_to_show), squeeze=False)
            for j, c in enumerate(channels_to_show):
                axes[0, c].imshow(img[c, ...], cmap="gray")
            plt.show()
            plt.close()
        print("Done")

    elif args.mode == "add_entry":
        # Run addNewEntryLocally
        ParticlesDataset.addNewEntryLocally(starFname=args.starFname,
                                            particlesRootDir=args.particlesRootDir,
                                            newTargetName=args.newTargetName,
                                            halfset=int(args.halfset),
                                            symmetry=args.symmetry,
                                            benchmarkDir=args.benchmarkDir)
        print(
            f"Successfully added new entry {args.newTargetName} with halfset {args.halfset} to benchmark "
            f"directory {args.benchmarkDir}.")

    elif args.mode == "list_entries":
        # List available entries
        if args.remote:
            remote_entries = ParticlesDataset.getCESPEDEntries()
            print("Available for donwload entries:", remote_entries)
        else:
            local_entries = ParticlesDataset.getLocallyAvailableEntries(benchmarkDir=args.benchmarkDir)
            print("Locally available entries:", local_entries)
    else:
        raise ValueError("Error, option is not valid")

    """
    Reunning example
    
    particlesDataset visualize --targetName TEST --halfset 0
    """