import os.path as osp
import functools
import random
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import RandomErasing
import torchvision.transforms.functional as transformF
from cesped.constants import default_configs_dir


# TODO: Implement augmentations in a better way, defining custom torchvision operations so that they can be used in batch mode seamingly.

class Augmenter:
    def __init__(self,
                 configFname:Optional[str] = None,
                 min_n_augm_per_img: Optional[int] = None,
                 max_n_augm_per_img: Optional[int] = None):

        augmentConfig = self.read_config(configFname)
        self._augmentationTypes = augmentConfig.operations
        self.augmentationTypes = self._augmentationTypes.copy()  # We have self._augmentationTypes in case we want to reset probs

        self.min_n_augm_per_img = augmentConfig.min_n_augm_per_img if (
                    min_n_augm_per_img is None) else min_n_augm_per_img
        self.max_n_augm_per_img = augmentConfig.max_n_augm_per_img if (
                    max_n_augm_per_img is None) else max_n_augm_per_img

        self.probSchedulers = {name: Scheduler(vals.get("probScheduler")).generate() for name, vals in
                               self.augmentationTypes.items()}

        self.augmentation_count = 0

    @classmethod
    def read_config(cls, fname=None):
        if fname is None:
            return OmegaConf.load(osp.join(default_configs_dir, "defaultDataAugmentation.yaml"))
        else:
            return OmegaConf.load(osp.join(fname))

    @functools.lru_cache(1)
    def _getRandomEraser(self, **kwargs):
        return RandomErasing(p=1., **kwargs)

    def _randomErase(self, img, **kwargs):
        eraser = self._getRandomEraser(**kwargs)
        return eraser(img)

    def _get_nrounds(self):
        return random.randint(self.min_n_augm_per_img, self.max_n_augm_per_img)

    def _get_rand(self):
        return random.random()

    def applyAugmentation(self, imgs, degEulerList, shiftFractionList):
        if len(imgs.shape) > 3: #TODO: Better batch mode
            transformed_batch = []
            degEulerList_ = []
            shiftFractionList_ = []
            applied_transforms_ = []
            for img, euler, shift in zip(imgs, degEulerList, shiftFractionList):
                (transformed_img, euler, shift,
                 applied_transforms) = self._applyAugmentation(img, euler, shift)
                transformed_batch.append(transformed_img)
                degEulerList_.append(euler)
                shiftFractionList_.append(shift)
                applied_transforms_ += [applied_transforms]
            return torch.stack(transformed_batch, dim=0), torch.stack(degEulerList_, dim=0), \
                torch.stack(shiftFractionList_, dim=0), applied_transforms_
        else:
            return self._applyAugmentation(imgs, degEulerList, shiftFractionList)

    def _applyAugmentation(self, img, degEuler, shiftFraction):
        """

        Args:
            img: A tensor of shape 1XLxL
            degEuler:
            shiftFraction:

        Returns:

        """
        img = img.clone()
        applied_transforms = []
        n_rounds = self._get_nrounds()
        for round in range(n_rounds):
            for aug, v in self.augmentationTypes.items():
                aug_kwargs = v["kwargs"]
                p = self.probSchedulers[aug](v["p"], self.augmentation_count)
                if self._get_rand() < p:
                    if aug == "randomGaussNoise":
                        scale = random.random() * aug_kwargs["scale"]
                        applied_transforms.append((aug, dict(scale=scale)))
                        img += torch.randn_like(img) * scale
                    elif aug == "randomUnifNoise":
                        scale = random.random() * aug_kwargs["scale"]
                        img += (torch.rand_like(img) - 0.5) * scale
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "inPlaneRotations90":
                        rotOrder = random.randint(0, 3)
                        img = torch.rot90(img, rotOrder, [-2, -1])
                        degEuler[-1] = (degEuler[-1] + 90. * rotOrder) % 360
                        applied_transforms.append((aug, dict(rotOrder=rotOrder)))

                    elif aug == "inPlaneRotations":
                        randDeg = (torch.rand(1) - 0.5) * aug_kwargs["maxDegrees"]
                        img, theta = rotTransImage(img.unsqueeze(0), randDeg, translationFract=torch.zeros(1),
                                                   scaling=1)
                        img = img.squeeze(0)
                        degEuler[-1] = (degEuler[-1] + randDeg.item()) % 360
                        applied_transforms.append((aug, dict(randDeg=randDeg)))

                    elif aug == "inPlaneShifts":  # It is important to do rotations before shifts

                        randFractionShifts = (torch.rand(2) - 0.5) * aug_kwargs["maxShiftFraction"]
                        img = rotTransImage(img.unsqueeze(0), torch.zeros(1), translationFract=randFractionShifts,
                                            scaling=1)[0].squeeze(0)
                        shiftFraction += randFractionShifts
                        applied_transforms.append((aug, dict(randFractionShifts=randFractionShifts)))

                    elif aug == "sizePerturbation":
                        scale = 1 + (random.random() - 0.5) * aug_kwargs["maxSizeFraction"]
                        img = rotTransImage(img.unsqueeze(0), torch.zeros(1), translationFract=torch.zeros(2),
                                            scaling=torch.FloatTensor([scale]))[0].squeeze(0)
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "gaussianBlur":
                        scale = 1e-3 + (1 + (random.random() - 0.5) * aug_kwargs["scale"])
                        img = transformF.gaussian_blur(img, kernel_size=3 + 2 * int(scale), sigma=scale)
                        applied_transforms.append((aug, dict(scale=scale)))

                    elif aug == "erasing":
                        kwargs = {k: tuple(v) for k, v in aug_kwargs.items()}
                        img = self._randomErase(img, **kwargs)
                        applied_transforms.append((aug, dict(kwargs=kwargs)))
                    else:
                        raise ValueError(f"Error, unknown augmentation {aug}")
        self.augmentation_count += 1
        return img, degEuler, shiftFraction, applied_transforms

    def __call__(self, img, eulersDeg, shiftFraction):
        return self.applyAugmentation(img, eulersDeg, shiftFraction)



def rotTransImage(image, degrees, translationFract, scaling=1., padding_mode='reflection',
                  interpolation_mode="bilinear", rotation_first=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param image: BxCxNxN
    :param degrees:
    :param translationFract: The translation to be applied as a fraction of the total size in pixels
    :param scaling:
    :param padding_mode:
    :param interpolation_mode:
    :param rotation_first: if using to compute Relion alignment parameters, set it to True
    :return:
    """
    align_corners = True  # If set to True, the extrema (-1 and 1) are considered as referring to the center points of the input’s corner pixels. If set to False, they are instead considered as referring to the corner points of the input’s corner pixels, making the sampling more resolution agnostic.
    assert ((-1 < translationFract) & (translationFract < 1)).all(), \
        (f"Error, translation should be provided as a fraction of the image."
         f" {translationFract.min()} {translationFract.max()} ")
    radians = torch.deg2rad(degrees)
    cosAngle = torch.cos(radians)
    sinAngle = torch.sin(radians)

    # theta = torch.stack([cosAngle, -sinAngle, translation[..., 0:1], sinAngle, cosAngle, translation[..., 1:2]], -1).view(-1, 2, 3)

    noTransformation = torch.eye(3).unsqueeze(0).repeat(sinAngle.shape[0], 1, 1).to(sinAngle.device)
    rotMat = noTransformation.clone()
    rotMat[:, :2, :2] = torch.stack([cosAngle, -sinAngle, sinAngle, cosAngle], -1).view(-1, 2, 2)

    transMat = noTransformation.clone()
    transMat[:, :2, -1] = translationFract

    if rotation_first:
        theta = torch.bmm(rotMat, transMat)[:, :2, :]
    else:
        theta = torch.bmm(transMat, rotMat)[:, :2, :]

    # raise NotImplementedError("TODO: check if this is how to do it, rotTrans rather than transRot")
    if scaling != 1:
        theta[:, 0, 0] *= scaling
        theta[:, 1, 1] *= scaling

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    # Generate the grid for the transformation
    grid = F.affine_grid(
        theta,
        size=image.shape,
        align_corners=align_corners,
    )

    # Perform the affine transformation with automatic padding
    image = F.grid_sample(
        image,
        grid,
        padding_mode=padding_mode,
        align_corners=align_corners,
        mode=interpolation_mode

    )
    return image, theta

def _generate_scheduler(schedulerInfo):
    if schedulerInfo is None:
        def _identity(x, current_step):
            return x
        return _identity
    else:
        schedulerName = schedulerInfo["type"]
        schedulerKwargs = schedulerInfo["kwargs"]
        if schedulerName == "linear_up":
            maxProb = schedulerKwargs.get("max_prob")
            scheduler_steps = schedulerKwargs.get("scheduler_steps")

            def linear_up(p, current_step):
                # Linearly increase from 0 to p over scheduler_steps
                increment = (maxProb - p) / scheduler_steps
                new_p = min(p + increment * current_step, maxProb)
                return new_p

            return linear_up
        elif schedulerName == "linear_down":
            scheduler_steps = schedulerKwargs.get("scheduler_steps")
            minProb = schedulerKwargs.get("min_prob")

            def linear_down(p, current_step):
                # Linearly decrease from p to min_prob over scheduler_steps
                decrement = (p - minProb) / scheduler_steps
                new_p = max(p - decrement * current_step, minProb)
                return new_p

            return linear_down
        else:
            raise NotImplementedError(f"False {schedulerName} is not valid")

class Scheduler:
    def __init__(self, schedulerInfo):
        self.schedulerInfo = schedulerInfo

    def identity(self, x, current_step):
        return x

    def linear_up(self, p, current_step):
        maxProb = self.schedulerInfo["kwargs"].get("max_prob")
        scheduler_steps = self.schedulerInfo["kwargs"].get("scheduler_steps")
        increment = (maxProb - p) / scheduler_steps
        return min(p + increment * current_step, maxProb)

    def linear_down(self, p, current_step):
        scheduler_steps = self.schedulerInfo["kwargs"].get("scheduler_steps")
        minProb = self.schedulerInfo["kwargs"].get("min_prob")
        decrement = (p - minProb) / scheduler_steps
        return max(p - decrement * current_step, minProb)

    def generate(self):
        if self.schedulerInfo is None:
            return self.identity
        else:
            schedulerName = self.schedulerInfo["type"]
            if schedulerName == "linear_up":
                return self.linear_up
            elif schedulerName == "linear_down":
                return self.linear_down
            else:
                raise NotImplementedError(f"{schedulerName} is not valid")

if __name__ == "__main__":
    augmentKwargs = Augmenter.read_config()
    print(augmentKwargs)
    # augmentKwargs["operations"] = {'randomGaussNoise': {'kwargs': {'scale': 0.5}, 'p': 1.}}
    # augmentKwargs["operations"] = {'randomUnifNoise': {'kwargs': {'scale': 2}, 'p': 1.}}
    # augmentKwargs["operations"] = {'inPlaneRotations90': {'kwargs': {'scale': 2}, 'p': 1.}}

    from cesped.particlesDataset import ParticlesDataset

    # dataset = ParticlesDataset("TEST", 0)
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(root="/tmp/cifcar", transform=ToTensor(), download=True)

    augmenter = Augmenter(**augmentKwargs)
    # augmenter._get_rand = lambda: 0
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)
    for batch in dl:
        # _, img, (rotmat, shiftAngs, conf), *_ = batch; shiftFrac = shifstAngs/(2*datset.image_size)
        img, *_ = batch[0].unsqueeze(1)
        eulers = torch.from_numpy(Rotation.random(batch[0].shape[0]).as_matrix().astype(np.float32));
        shiftFrac = torch.zeros(batch[0].shape[0], 2)
        print(img.shape)
        img_, eulers_, shiftFrac_, applied_transforms = augmenter.applyAugmentation(img, eulers, shiftFrac)
        from matplotlib import pyplot as plt

        f, axes = plt.subplots(1, 2)
        for i in range(img.shape[0]):
            print(applied_transforms[i])
            axes.flat[0].imshow(img[i, ...].permute(1, 2, 0))  # , cmap="gray")
            axes.flat[1].imshow(img_[i, ...].permute(1, 2, 0))  # , cmap="gray")
            plt.show()
            print()
            print()
        # break
