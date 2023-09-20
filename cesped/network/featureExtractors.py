import re

import torch
import torchvision
from torch import nn

from cesped.utils.gaussianFilters import GaussianFilterBank


class ResNetImageEncoder(nn.Module):
    def __init__(self, resnetName: str = "resnet50", load_imagenetweights: bool = True, sigma0: float = 1.,
                 sigma1: float = 2):
        """

        Args:
            resnetName (str): The name of the torchvision.models resnet to be used
            sigma0 (float): The standard deviation of the 1st filter to expand the number of channels from 1 to 3
            sigma1 (float): The standard deviation of the 2nd filter to expand the number of channels from 1 to 3
        """
        super().__init__()

        cls = getattr(torchvision.models, resnetName)
        num = re.findall(r"resnet(\d+)", resnetName, re.IGNORECASE)[0]
        resnet = nn.Sequential(*list(cls(weights=None if not load_imagenetweights \
                                else getattr(torchvision.models, f"ResNet{num}_Weights").IMAGENET1K_V2
                                         ).children())[:-2])
        filterBank = GaussianFilterBank(1, [0, int(2 * sigma0) + 1, int(2 * sigma1) + 1],
                                        sigma_values=[0, sigma0, sigma1])
        self.imageEncoder = nn.Sequential(filterBank, resnet)

    def forward(self, x):
        return self.imageEncoder(x)


if __name__ == "__main__":
    model = ResNetImageEncoder()
    print(model)
