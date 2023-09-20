import os
from typing import Dict, Any, List

import torch
from lightning import Callback, Trainer
from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.add_argument("--n_threads_torch", default=4, type=int, help="Number of inter_operation threads")

        parser.link_arguments("data.image_size", "model.image_size")

        parser.link_arguments("data.symmetry", "model.true_symmetry", apply_on="instantiate")

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

    def before_instantiate_classes(self):
        torch.set_num_threads(self.config["n_threads_torch"])