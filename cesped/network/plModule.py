from typing import Union

import torch
import lightning.pytorch as pl

from cesped.network.featureExtractors import ResNetImageEncoder
from cesped.network.image2sphere import I2S


class PlModel(pl.LightningModule):
    def __init__(self, image_size: int, feature_extractor: Union[ResNetImageEncoder],
                 symmetry: str,
                 lmax: int = 6, s2_fdim: int = 512, so3_fdim: int = 16,
                 hp_order_projector: int = 2,
                 hp_order_s2: int = 2,
                 hp_order_so3: int = 3,
                 rand_fraction_points_to_project: float = .5):
        """

        Args:
            image_size: The size of the image, e.g. 256 pixels
            feature_extractor: A vision model that takes an image of shape 1 x image_size x image_size and returns an \
                image os shape C x K x K, with K< image_size
            symmetry (str): The symmetry to be applied during training
            lmax (int): The frequency of the signal
            s2_fdim (int): The number of filters for the S2 convolution
            so3_fdim (int): The number of filters for the SO(3) convolution
            hp_order_so3 (int): The grid size for the probability calculation
            rand_fraction_points_to_project (float): The fraction of points in the HEALPIX grid to be used in the \
             hemisphere. Works similarly to dropout. Smaller numbers should reduce overfitting.
        """
        super().__init__()
        # self.save_hyperparameters(ignore=['feature_extractor']) #Not needed since we are using CLI

        self.symmetry = symmetry

        example = torch.rand(1, feature_extractor.in_channels, image_size, image_size)
        imageEncoderOutputShape = feature_extractor(example).shape[1:]

        self.model = I2S(imageEncoder=feature_extractor, imageEncoderOutputShape=imageEncoderOutputShape,
                         symmetry=self.symmetry, lmax=lmax,
                         s2_fdim=s2_fdim, so3_fdim=so3_fdim,
                         hp_order_projector=hp_order_projector,
                         hp_order_s2=hp_order_s2,
                         hp_order_so3=hp_order_so3,
                         rand_fraction_points_to_project=rand_fraction_points_to_project)

    def _step(self, batch, batch_idx):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        loss, error_rads, pred_rotmats, maxprob, probs = self.model.forward_and_loss(imgs, rotMats, conf)
        return loss, error_rads, pred_rotmats, maxprob, probs

    def training_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True, batch_size=pred_rotmats.shape[0])
        self.log("error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ang_error, pred_rotmats, maxprob, probs = self._step(batch, batch_idx)
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        self.log("val_error_degs", torch.rad2deg(ang_error.mean()), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=pred_rotmats.shape[0], sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        idd, imgs, (rotMats, shifts, conf), metadata = self.resolve_batch(batch)
        grid_signal, pred_rotmats, maxprob, probs = self.model(imgs)
        if rotMats is not None:
            errors = self.model.rotation_error_rads(rotMats, pred_rotmats)
            errors = torch.rad2deg(errors)
            metadata["pred_degs_error"] = errors.detach().cpu().numpy()
        return idd, (pred_rotmats, maxprob), metadata


    def resolve_batch(self, batch):

        idd, imgs, (rotMats, shifts, conf), metadata = batch

        # # #Modified for the old datamanager
        # imgs = batch.particle
        # idd = str(batch.itemIdx)
        # rotMats = batch.poseRepresentation
        # shifts = batch.shiftsFraction * 1.5 #1.5 is the sampling rate, only for debug here #TODO: Remove this
        # conf = batch.poseProbability

        return idd, imgs, (rotMats, shifts, conf), metadata