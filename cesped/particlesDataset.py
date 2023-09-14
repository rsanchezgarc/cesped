import functools
import json
import os.path
import warnings
from os import PathLike
import os.path as osp
from typing import Union, Literal, Optional, List, Tuple


import numpy as np
import torch
from lightning import pytorch as pl
from scipy.spatial.transform import Rotation as R
from starstack.particlesStar import ParticlesStarSet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import CenterCrop, Resize, Compose

from cesped import default_configs_dir, defaultBenchmarkDir
from cesped.zenodo.bechmarkUrls import ROOT_URL_PATTERN, NAME_PARTITION_TO_RECORID
from cesped.utils.ctf import apply_ctf
from cesped.utils.tensors import data_to_numpy
from cesped.zenodo.downloadFromZenodo import download_record, getDoneFname

warnings.filterwarnings("ignore",
                        "Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")

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

    RELION_EULER_CONVENTION: str ="ZYZ"
    """ Euler convention used by Relion. Rot, Tilt and Psi angles"""
    RELION_ANGLES_NAMES: List[str] = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    """ Euler angles names in Relion. Rot, Tilt and Psi correspond to rotations on Z, Y and Z"""
    RELION_SHIFTS_NAMES: List[str] = ['rlnOriginXAngst', 'rlnOriginYAngst']
    """ Image shifts names in Relion. They are measured in Å (taking into account the sampling rate (aka pixel size) """
    RELION_POSE_CONFIDENCE_NAME: str = 'rlnParticleFigureOfMerit'
    """ The name of the metadata field used to weight the particles for the volume reconstruction"""

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
        self.benchmarkDir = benchmarkDir
        self.image_size = image_size
        self.apply_perImg_normalization = apply_perImg_normalization
        assert ctf_correction in ["none", "phase_flip"]
        self.ctf_correction = ctf_correction
        self._particles = None
        self._symmetry = None

        assert 0 <= image_size_factor_for_crop < 0.5
        _preprocessing = []
        if image_size_factor_for_crop > 0:
            imgShape = [int(s * (1 - image_size_factor_for_crop)) for s in self.particles.particle_shape]
            _preprocessing += [CenterCrop(size=imgShape)]
        if image_size is not None:
            _preprocessing += [Resize(image_size, antialias=True)]
        self._preprocessing = Compose(_preprocessing)

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
        if self._symmetry is None:
            with open(osp.join(self.datadir, f"info_{self.halfset}.json")) as f:
                self._symmetry = json.load(f)["symmetry"].upper()
        return self._symmetry

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
        #TODO: Add symmetry
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
        avail = []
        return avail
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
                        destination_dir=self.datadir,
                        root_url=ROOT_URL_PATTERN)

    def _is_avaible(self):
        return osp.isfile(getDoneFname(self.datadir, self.halfset))

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
        backgroundMask = self._getParticleNormalizationMask(img.shape[-1])
        noiseRegion = img[backgroundMask]
        meanImg = noiseRegion.mean()
        stdImg = noiseRegion.std()
        return (img - meanImg) / stdImg

    def __getitem__(self, item):
        img, metadata = self.particles[item]
        img = torch.from_numpy(img)
        if self.apply_perImg_normalization:
            img = self._normalize(img)

        rotMat = R.from_euler(self.RELION_EULER_CONVENTION, [metadata[name] for name in self.RELION_ANGLES_NAMES],
                              degrees=True).as_matrix()
        rotMat = torch.from_numpy(rotMat.astype(np.float32))
        xyShiftAngs = torch.FloatTensor([metadata[name] for name in self.RELION_SHIFTS_NAMES])
        confidence = torch.FloatTensor([metadata["rlnMaxValueProbDistribution"]])
        if self.ctf_correction != "none":
            ctf, wimg = apply_ctf(img, self.particles.sampling_rate, dfu=metadata["rlnDefocusU"],
                                  dfv=metadata["rlnDefocusV"],
                                  dfang=metadata["rlnDefocusAngle"],
                                  volt=float(self.particles.optics_md["rlnVoltage"][0]),
                                  cs=float(self.particles.optics_md["rlnSphericalAberration"][0]),
                                  w=float(self.particles.optics_md["rlnAmplitudeContrast"][0]),
                                  mode=self.ctf_correction)
            wimg = torch.clamp(wimg, img.min(), img.max())
            wimg = torch.nan_to_num(wimg, nan=img.mean())
            img = wimg

        img = img.unsqueeze(0)
        img = self._preprocessing(img)
        # TODO: Implement filter banks
        return img, rotMat, xyShiftAngs, confidence, metadata.to_dict()

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
                rots, tilts, psis = r.as_euler(self.RELION_EULER_CONVENTION, degrees=True).T
            else:
                rots, tilts, psis = [angles[:, i] for i in range(3)]

            col2val.update({ #RELION_ANGLES_NAMES
                self.RELION_ANGLES_NAMES[0]: rots,
                self.RELION_ANGLES_NAMES[0]: tilts,
                self.RELION_ANGLES_NAMES[0]: psis
            })

        if shifts is not None:
            shifts = data_to_numpy(shifts)
            col2val.update({
                self.RELION_SHIFTS_NAMES[0]: shifts[:, 0],
                self.RELION_SHIFTS_NAMES[1]: shifts[:, 1],
            })

        if confidence is not None:
            confidence = data_to_numpy(confidence)
            col2val.update({
                self.RELION_POSE_CONFIDENCE_NAME: confidence,
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
        self.particles.save(starFname=fname, overwrite=True)


class ParticlesDataModule(pl.LightningDataModule):
    """
    ParticlesDataModule: A LightningDataModule that wraps a ParticlesDataset
    """

    def __init__(self, targetName: Union[PathLike, str], halfset: Literal[0, 1], image_size: int,
                 benchmarkDir: str = defaultBenchmarkDir, apply_perImg_normalization: bool = True,
                 ctf_correction: Literal["none", "phase_flip"] = "phase_flip", image_size_factor_for_crop: float = 0.25,
                 train_validation_split: Tuple[float, float] = (0.7, 0.3), batch_size: int = 8,
                 num_data_workers: int = 0):
        """
        ##Builder

        Args:
            targetName (Union[PathLike, str]): The name of the target to use. It is also the basename of \
            the directory where the data is.
            halfset (Literal[0, 1]): The second parameter.
            image_size (bool): The final size of the image (after cropping).
            benchmarkDir (str): The root directory where the datasets are downloaded.
            apply_perImg_normalization (bool): Apply cryo-EM per-image normalization. I = (I-noiseMean)/noiseStd.
            ctf_correction (Literal[none, phase_flip]): phase_flip will correct amplitude inversion due to defocus
            image_size_factor_for_crop (float): Fraction of the image size to be cropped. Final size of the image \
            is origSize*(1-image_size_factor_for_crop). It is important because particles in cryo-EM tend to \
            be only 50% to 25% of the total area of the image.
            train_validation_split (List[float]):
            batch_size (int): The batch size
            num_data_workers (int): The number of workers for data loading. Set it to 0 to use the same thread as the model

        """

        super().__init__()
        self.save_hyperparameters()
        self.targetName = targetName
        self.halfset = halfset
        self.benchmarkDir = benchmarkDir
        self.image_size = image_size
        self.apply_perImg_normalization = apply_perImg_normalization
        self.ctf_correction = ctf_correction
        self.image_size_factor_for_crop = image_size_factor_for_crop
        self.train_validation_split = train_validation_split
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers

        self._dataset = None

    @property
    def symmetry(self):
        """The point symmetry of the dataset"""
        return self.dataset.symmetry

    @property
    def dataset(self) -> ParticlesDataset:
        """The particles dataset"""
        if self._dataset is None:
            self._dataset = ParticlesDataset(self.targetName, halfset=self.halfset, benchmarkDir=self.benchmarkDir,
                                image_size=self.image_size)
        return self._dataset

    def _create_dataloader(self, partitionName: Optional[str]):
        dataset = self.dataset
        if partitionName in ["train", "val"]:
            assert self.train_validation_split is not None, "Error, self.train_validation_split required"
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, self.train_validation_split)
            if partitionName == "train":
                warnings.warn("IMPLEMENT DATA AUGMENTATION LOGIC")
                dataset = train_dataset
            else:
                dataset = val_dataset
        dl = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True if partitionName == "train" else False,
            num_workers=self.num_data_workers,
            persistent_workers=True if self.num_data_workers > 0 else False)

        return dl

    def train_dataloader(self):
        return self._create_dataloader(partitionName="train")

    def val_dataloader(self):
        return self._create_dataloader(partitionName="val")

    def test_dataloader(self):
        return self._create_dataloader(partitionName="test")

    def predict_dataloader(self):
        return self._create_dataloader(partitionName=None)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml"))

    parser = ArgumentParser("Visualize dataset")
    parser.add_argument("-t", "--targetName", help="The target to visualize", type=str, required=True)
    parser.add_argument("-p", "--halfset", help="The target to visualize", choices=[0, 1], default=0)
    parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str,
                        default=cfg.data.benchmarkDir)
    parser.add_argument("-s", "--image_size", help="The desired image size", type=int,
                        default=cfg.data.image_size)
    parser.add_argument("-c", "--image_size_factor_for_crop", help="Percentage of image to crop",
                        type=float, default=cfg.data.image_size_factor_for_crop)
    parser.add_argument("-f", "--phase_flippling", help="Apply phase flippling", action="store_true")
    parser.add_argument("-i", "--channels_to_show", help="Channels of the images to show",
                        type=int, nargs="+")
    args = parser.parse_args()

    ps = ParticlesDataset(targetName=args.targetName,
                          halfset=args.halfset,
                          benchmarkDir=args.benchmarkDir,
                          image_size=args.image_size,
                          ctf_correction="phase_flip" if args.phase_flippling else "none",
                          image_size_factor_for_crop=args.image_size_factor_for_crop
                          )
    import matplotlib.pyplot as plt

    channels_to_show = args.channels_to_show if args.channels_to_show else [0]
    for elem in ps:
        img, *_ = elem
        assert 1 <= len(channels_to_show) <= 4, "Error, at least one channel required and no more than 4"
        f, axes = plt.subplots(1, len(channels_to_show), squeeze=False)
        for j, c in enumerate(channels_to_show):
            axes[0, c].imshow(img[c, ...], cmap="gray")
        plt.show()
        plt.close()
    print("Done")
