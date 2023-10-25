import sys

import tempfile

import os
import os.path as osp
from omegaconf import OmegaConf
from typing import Dict, Any, List

import torch
from lightning import Callback, Trainer
from lightning.pytorch.cli import LightningCLI

_ckpt_path_argname = "ckpt_path"
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.add_argument("--n_threads_torch", default=4, type=int, help="Number of inter_operation threads")
        parser.add_argument(f"--{_ckpt_path_argname}", type=str, help="The model checkpoint to load")

        parser.link_arguments("data.image_size", "model.image_size")
        parser.link_arguments("data.symmetry", "model.symmetry", apply_on="instantiate")



    # def before_instantiate_classes(self) -> None:
    #     self.config["model"]["image_size"] = self.config["data"]["image_size"]

    def _instantiate_trainer(self, config: Dict[str, Any], callbacks: List[Callback]) -> Trainer:
        assert config["logger"] in [None, False], ("Error, do not provide info about the logger as config. Modify"
                                          " MyLightningCLI instead")
        if config["logger"] is not False:
            from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
            saveDir = config["default_root_dir"]
            logger1 = TensorBoardLogger(saveDir)
            _, version = os.path.split(logger1.log_dir)
            logger2 = CSVLogger(saveDir, version=version)
            config["logger"] = [logger1, logger2]
        return super()._instantiate_trainer(config, callbacks)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        super().instantiate_classes()

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        torch.set_num_threads(self.config["n_threads_torch"])

    @property
    def ckpt_path(self):
        return self.config.get(_ckpt_path_argname)

class CheckpointLoader():
    def __init__(self, config_fnames):

        self._config_fnames = config_fnames
        self.conf = {}
        for confFname in self._config_fnames:
            self.conf.update(OmegaConf.load(confFname))

        self.ckpt_path = None
        self._tmpdir = None
        self._new_config_fname = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = self._tmpdir.name

        if f"--{_ckpt_path_argname}" in sys.argv:
            checkpointName = sys.argv[sys.argv.index(f"--{_ckpt_path_argname}") + 1]
            self.ckpt_path = osp.abspath(osp.expanduser(checkpointName))
            model_config_fname = osp.join(osp.dirname(osp.dirname(self.ckpt_path)), "config.yaml")

            if not osp.isfile(model_config_fname):
                raise RuntimeError(f"Error, {model_config_fname} does not exist")

            self.conf.update(dict(model=OmegaConf.load(model_config_fname)["model"]))

            self._new_config_fname = osp.join(tmpdir_path, "final_config.yaml")
            with open(self._new_config_fname, "w") as f:
                OmegaConf.save(self.conf, f)
            self.ckpt_found = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._tmpdir.cleanup()

    @property
    def config_fnames(self):
        if self._new_config_fname is None:
            return self._config_fnames
        else:
            return [self._new_config_fname]