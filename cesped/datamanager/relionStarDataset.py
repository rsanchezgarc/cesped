import functools
import os
import warnings

import torch
import numpy as np
import os.path as osp

from scipy.spatial.transform import Rotation as R
from starstack.particlesStar import ParticlesStarSet
from torch.utils.data import Dataset
from typing import Union, Literal, Optional, List, Tuple, Any, Dict
from os import PathLike

warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it "
                                  "is not possible to uniquely determine all angles.")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces")

from cesped.constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION, \
    RELION_ORI_POSE_CONFIDENCE_NAME, RELION_PRED_POSE_CONFIDENCE_NAME
from cesped.datamanager.augmentations import AugmenterBase
from cesped.datamanager.ctf import correct_ctf
from cesped.utils.tensors import data_to_numpy


class ParticlesRelionStarDataset(Dataset):
    """
    ParticlesRelionStarDataset: A Pytorch Dataset for dealing with Cryo-EM particles in relion star format.<br>
    It can download data automatically

    ```python
    #Loads the halfset 0 for the benchmark entry named "TEST"
    ds = ParticlesRelionStarDataset(starFname="/tmp/cryoSupervisedDataset/particles.star", rootDir="/tmp/cryoSupervisedDataset/", symmetry="c1)
    ```
    and each particle can be acessed as usually
    ```python
    img, rotMat, xyShiftAngs, confidence, metadata = ds[0]
    ```
    <br>
    """

    def __init__(self, starFname: Union[PathLike, str],
                 rootDir: str,
                 symmetry: str,
                 image_size: Optional[int] = None,
                 perImg_normalization: Literal["none", "noiseStats", "subtractMean"] = "noiseStats",
                 ctf_correction: Literal["none", "phase_flip", "ctf_multiply"] = "none",
                 image_size_factor_for_crop: float = 0.,
                 ):
        """
        ##Builder

        Args:
            starFname (Union[PathLike, str]): The star filename to use
            rootDir (str): The root directory where the stack files are
            symmetry (str): The point symmetry of the macromolecule
            image_size (Optional[int]): The final size of the image (after cropping). If None, keep the original size
            perImg_normalization (Literal["none", "noiseStats", "subtractMean"]) The normalization to apply to the images
                Default is "noiseStats" --> I = (I-noiseMean)/noiseStd, computing the noise statistics using a circular mask
            ctf_correction (Literal[none, phase_flip]): phase_flip will correct amplitude inversion due to defocus
            image_size_factor_for_crop (float): Fraction of the image size to be cropped. Final size of the image \
                is origSize*(1-image_size_factor_for_crop). It is important because particles in cryo-EM tend to \
                be only 50% to 25% of the total area of the image.

        """

        super().__init__()
        self._starFname = starFname
        self._datadir = osp.expanduser(rootDir)
        self._image_size = image_size

        assert perImg_normalization in ["none", "noiseStats", "subtractMean"]
        if perImg_normalization == "none":
            self._normalize = self._normalizeNone
        elif perImg_normalization == "noiseStats":
            self._normalize = self._normalizeNoiseStats
        elif perImg_normalization == "subtractMean":
            self._normalize = self._normalizeSubtractMean
        else:
            ValueError(f"Error, perImg_normalization {perImg_normalization} wrong option")

        assert ctf_correction in ["none", "phase_flip", "ctf_multiply"]
        if ctf_correction == "none":
            self._correctCtf = self._correctCtfNone
        elif ctf_correction in ["ctf_multiply", "phase_flip"]:
            self._correctCtf = self._correctCtfPhase
        else:
            ValueError(f"Error, perImg_normalization {ctf_correction} wrong option")

        self.ctf_correction = ctf_correction
        self._symmetry = symmetry.upper()

        assert 0 <= image_size_factor_for_crop <= 0.5
        self.image_size_factor_for_crop = image_size_factor_for_crop

        self._lock = torch.multiprocessing.RLock()
        self._particles = None
        self._augmenter = None


    @property
    def image_size(self) -> int:
        """The image size in pixels"""
        if self._image_size is None:
            return self.particles.particle_shape[-1]
        else:
            return self._image_size


    @property
    def particles(self) -> ParticlesStarSet:
        """
        a starstack.particlesStar.ParticlesStarSet representing the loaded particles
        """
        if self._particles is None:
            self._particles = ParticlesStarSet(starFname=self._starFname, particlesDir=self._datadir)
        return self._particles


    @property
    def sampling_rate(self) -> float:
        """The particle image sampling rate in A/pixels"""
        return self.particles.sampling_rate

    @property
    def augmenter(self) -> AugmenterBase:
        """The data augmentator object to be applied"""
        return self._augmenter

    @augmenter.setter
    def augmenter(self, augmenterObj:AugmenterBase):
        """

        Args:
            augmenter: he data augmentator object to be applied

        """
        self._augmenter = augmenterObj

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

    def _normalizeNoiseStats(self, img):
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

    def _normalizeSubtractMean(self, img):
        return (img - img.mean())

    def _normalizeNone(self, img):
        return img

    def _correctCtfPhase(self, img, md_row):
        ctf, wimg = correct_ctf(img, float(self.particles.optics_md["rlnImagePixelSize"].item()),
                                dfu=md_row["rlnDefocusU"], dfv=md_row["rlnDefocusV"], dfang=md_row["rlnDefocusAngle"],
                                volt=float(self.particles.optics_md["rlnVoltage"][0]),
                                cs=float(self.particles.optics_md["rlnSphericalAberration"][0]),
                                w=float(self.particles.optics_md["rlnAmplitudeContrast"][0]), mode=self.ctf_correction)
        wimg = torch.clamp(wimg, img.min(), img.max())
        wimg = torch.nan_to_num(wimg, nan=img.mean())
        # img = torch.concat([img, wimg], dim=0)
        return wimg

    def _correctCtfNone(self, img, md_row):
        return img


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
        particle_size_after_crop = int(
            self.particles.optics_md["rlnImageSize"].item() * (1 - self.image_size_factor_for_crop))

        desired_sampling_rate = float(
            self.particles.optics_md["rlnImagePixelSize"].item() * particle_size_after_crop / self.image_size)

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

        img = self._normalize(img)

        img = self._correctCtf(img, md_row)

        if img.isnan().any():
            raise RuntimeError(f"Error, img with idx {item} is NAN")

        img = self.resizeImage(img)

        degEuler = torch.FloatTensor([md_row[name] for name in RELION_ANGLES_NAMES])
        xyShiftAngs = torch.FloatTensor([md_row[name] for name in RELION_SHIFTS_NAMES])

        if self.augmenter is not None:
            img, degEuler, shift, _ = self.augmenter(img,  # 1xSxS image expected
                                                     degEuler,
                                                     shiftFraction=xyShiftAngs / (self.image_size * self.sampling_rate))
            xyShiftAngs = shift * (self.image_size * self.sampling_rate)

        r = R.from_euler(RELION_EULER_CONVENTION, degEuler, degrees=True)
        if self._symmetry != "C1":
            r = r.reduce(R.create_group(self._symmetry.upper()))
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

            col2val.update({  # RELION_ANGLES_NAMES
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

    def saveMd(self, fname: Union[str, PathLike], overwrite: bool = True):
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
    
    parser = ArgumentParser(description="Visualize dataset relion starDataset")
    parser.add_argument("-f", "--filename", type=str, help="The starfile to visualize", required=True)
    parser.add_argument("-d", "--dirname", type=str, help="The root directory for the particle stacks", required=False, default=None)
    parser.add_argument("-s", "--symmetry", type=str, help="The point symmetry of the particle", required=True, default=None)
    parser.add_argument("-b", "--resize_box", type=str, help="The desired image box size", required=False, default=None)
    parser.add_argument("-n", "--normalization_type", help="The normalization type", choices=["none", "noiseStats", "subtractMean"], default="noiseStats", required=False)
    parser.add_argument("-c", "--ctf_correction", type=str, help="The ctf correction type", choices=["none", "phase_flip", "ctf_multiply"], default="none", required=False)
    parser.add_argument("-t", "--image_size_factor_for_crop", type=float, help="The  image_size_factor_for_crop", default=0., required=False)
    args = parser.parse_args()
    parts = ParticlesRelionStarDataset(starFname=args.filename,
                 rootDir=args.filename,
                 symmetry=args.symmetry,
                 image_size=args.resize_box,
                 perImg_normalization=args.normalization_type,
                 ctf_correction=args.ctf_correction,
                 image_size_factor_for_crop=args.image_size_factor_for_crop,
                 )

    from matplotlib import pyplot as plt    
    channels_to_show = [0]             
    for elem in parts:
        iid, img, *_ = elem
        print(img.shape)
        assert 1 <= len(channels_to_show) <= 4, "Error, at least one channel required and no more than 4"
        f, axes = plt.subplots(1, len(channels_to_show), squeeze=False)
        for j, c in enumerate(channels_to_show):
            axes[0, c].imshow(img[c, ...], cmap="gray")
        plt.show()
        plt.close()
    print("Done")


