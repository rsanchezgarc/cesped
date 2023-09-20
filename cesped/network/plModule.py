from typing import Optional, Union

import torch
import torchvision
import lightning.pytorch as pl
from torch import nn

from cesped.network.featureExtractors import ResNetImageEncoder
from cesped.network.image2sphere import I2S


class PlModel(pl.LightningModule):
    def __init__(self, image_size: int, feature_extractor: Union[ResNetImageEncoder],
                 true_symmetry: str, symmetry: Optional[str] = None, lr: float = 1e-4,
                 lmax: int = 6, s2_fdim: int = 512, so3_fdim: int = 16, hp_order_so3: int = 2):
        """

        Args:
            image_size: The size of the image, e.g. 256 pixels
            feature_extractor: A vision model that takes an image of shape 1 x image_size x image_size and returns an \
                image os shape C x K x K, with K< image_size
            true_symmetry (str): The true symmetry of the dataset. Required to compute evaluation metrics
            symmetry (str): The symmetry to be applied during training
            lr: The learning rate
            lmax (int): The frequency of the signal
            s2_fdim (int): The number of filters for the S2 convolution
            so3_fdim (int): The number of filters for the SO(3) convolution
            hp_order_so3 (int): The grid size for the probability calculation
        """
        super().__init__()
        self.save_hyperparameters(ignore=['feature_extractor'])

        self.lr = lr
        self.true_symmetry = true_symmetry
        self.symmetry = symmetry
        if self.symmetry is None:
            self.symmetry = self.true_symmetry

        # Resnet before the global average pooling operation.
        self.imageEncoder = feature_extractor

        example = torch.rand(1, 1, image_size, image_size)

        imageEncoderOutputShape = self.imageEncoder(example).shape[1:]
        self.model = I2S(imageEncoder=self.imageEncoder, imageEncoderOutputShape=imageEncoderOutputShape,
                         true_symmetry=self.true_symmetry,
                         symmetry=self.symmetry,
                         lmax=lmax,
                         s2_fdim=s2_fdim, so3_fdim=so3_fdim, hp_order_so3=hp_order_so3)

    def _step(self, batch, batch_idx):
        idd, imgs, (rotMats, shifts, conf), *_ = batch
        loss, error_rads, pred_rotmats, maxprob, probs = self.model.compute_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprob, probs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        self.log("loss", loss.mean(), prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        self.log("val_loss", loss.mean(), prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("val_error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        metadata = batch[-1]
        return pred_rotmats, maxprob, metadata

    def configure_optimizers(self):
        # TODO: Move this part to LightningCLI
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                                  factor=0.5,
                                                                  min_lr=self.lr * 1e-3,
                                                                  cooldown=1,
                                                                  patience=5)
        return {
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }
