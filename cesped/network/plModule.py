from typing import Any, Optional

import torch
import torchvision
import lightning.pytorch as pl
from torch import nn

from cesped.network.image2sphere import I2S
from cesped.utils.gaussianFilters import GaussianFilterBank


class PlModel(pl.LightningModule):
    def __init__(self, image_size:int, batch_size:int, true_symmetry:str, symmetry:Optional[str]=None,
                 lr:float=1e-4, lmax:int=6, s2_fdim:int=512, so3_fdim:int=16, hp_order_so3:int=2):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.true_symmetry = true_symmetry
        self.symmetry = symmetry
        if self.symmetry is None:
            self.symmetry = self.true_symmetry
        self.save_hyperparameters()

        # Resnet before the global average pooling operation.
        resnet = nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])
        filterBank = GaussianFilterBank(1, [0, 5, 9], sigma_values=[0, 1, 2])
        self.imageEncoder = nn.Sequential(filterBank, resnet)
        #TODO: Check the new filterBank
        example = torch.rand(1, 1, image_size, image_size)

        imageEncoderOutputShape = self.imageEncoder(example).shape[1:]
        self.model = I2S(imageEncoder=self.imageEncoder, imageEncoderOutputShape=imageEncoderOutputShape,
                         symmetry=self.symmetry,
                         lmax=lmax,
                         s2_fdim=s2_fdim, so3_fdim=so3_fdim, hp_order_so3=hp_order_so3)

    def _step(self, batch, batch_idx):
        idd, imgs, (rotMats, shifts, conf), *_ = batch
        loss, error_rads, pred_rotmats, maxprob, probs = self.model.compute_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprob, probs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        self.log("loss", loss.mean(), prog_bar=True, batch_size=self.batch_size)
        self.log("error_rads", ang_error.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        self.log("val_loss", loss.mean(), prog_bar=True)
        self.log("val_error_rads", ang_error.mean(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        metadata = batch[-1]
        return pred_rotmats, maxprob, metadata

    def configure_optimizers(self):
        #TODO: Move this part to LightningCLI
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
