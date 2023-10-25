"""
This module is used to infer the pose of a benchmark entry given a trained model
Use it as CLI as <br>
    ```
python -m cesped.inferEntry --data.targetName TEST \
--data.halfset 0 \
--outFname /tmp/results.star \
--ckpt_path /tmp/supervised/lightning_logs/version_1/checkpoints/last.ckpt
    ```
"""

import shutil
import atexit
import os
import pickle

import os.path as osp
import sys

import argparse
import torch
from lightning import Callback, Trainer
from typing import Dict, Any, List

from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningArgumentParser, ArgsType

from cesped.constants import default_configs_dir
from cesped.utils.cliBuilder import MyLightningCLI, _ckpt_path_argname, CheckpointLoader
from cesped.network.plModule import PlModel
from cesped.datamanager.plDataset import ParticlesDataModule

_outname_argname = "outFname"


class CustomPredWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("epoch")
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        batch_indices = batch_indices[0]
        count = 0
        with open(osp.join(self.output_dir, f"serialized_data_{trainer.global_rank}.pkl"), "wb") as f:
            for predBatch, idxsBatch in zip(predictions, batch_indices):
                ids, (pred_rotmats, maxprob), metadata = predBatch
                pickle.dump(predBatch, f)
                count += 1

        print(f"Partial results written by rank {trainer.global_rank} ({count})")
        trainer.strategy.barrier()

    def iter_results(self):
        for fname in os.listdir(self.output_dir):
            with open(osp.join(self.output_dir, fname), "rb") as f:
                while True:
                    try:
                        item = pickle.load(f)
                        yield item
                    except EOFError:
                        break


class _MyInferLightningCLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(f"--{_outname_argname}", type=str, help="The name of the output file, ended in .star",
                            required=True)


    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:

        #If the arguments provided by the user are wrong, we want to display help
        #wihtout generic arguments


        for act in parser._actions:
            if act.dest in [_outname_argname, _ckpt_path_argname, "data.halfset", "data.targetName"]:
                act.required = True
            elif act.dest in ["trainer.accelerator", "trainer.devices", "trainer.limit_predict_batches",
                            "data.batch_size", "data.num_data_workers"]:
                #argument not required but configurable
                pass
            else:
                #argument that is internally loaded from checkpoint
                act.help = argparse.SUPPRESS

        super().parse_arguments(parser, args)

    def _instantiate_trainer(self, config: Dict[str, Any], callbacks: List[Callback]) -> Trainer:

        outFname = self.config[_outname_argname]
        outDir = osp.join(osp.dirname(outFname), "_tmp_" + osp.basename(osp.splitext(outFname)[0]))
        os.makedirs(outDir, exist_ok=True)

        def remove_temp_dir(dir_path):
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed temporary directory: {dir_path}")
                except OSError:
                    pass
        atexit.register(remove_temp_dir, outDir)

        resultsWriter = CustomPredWriter(outDir)
        self.resultsWriter = resultsWriter
        callbacks += [resultsWriter]
        return super()._instantiate_trainer(config, callbacks)


if __name__ == "__main__":

    config_fnames = [
        osp.join(default_configs_dir, "defaultDataConfig.yaml"),
        osp.join(default_configs_dir, "defaultInferenceConfig.yaml"),
    ]


    with CheckpointLoader(config_fnames) as cpk:  # TODO: Check if this works
        print(f"Reading configuration from {cpk.config_fnames}")

        cli = _MyInferLightningCLI(model_class=PlModel, datamodule_class=ParticlesDataModule,
                                   save_config_callback=None,
                                   parser_kwargs={"default_env": True, "parser_mode": "omegaconf",
                                                  "default_config_files": cpk.config_fnames},
                                   run=False)
        outFname = cli.config["outFname"]
        assert outFname.endswith(".star"), (f'Error, --outFname {outFname} not valid. Needs '
                                            f'to end in .star')
        assert osp.isdir(osp.dirname(outFname)), (f'Error, --outFname {outFname} not valid. The directory does not '
                                                  f'exits!')

        cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=cli.ckpt_path, return_predictions=False)

        particlesDataset = cli.datamodule.createDataset()

        cli.trainer.strategy.barrier()
        if cli.trainer.is_global_zero:
            for ids, (pred_rotmats, maxprob), metadata in cli.resultsWriter.iter_results():
                particlesDataset.updateMd(ids=ids, angles=pred_rotmats,
                                          shifts=torch.zeros(pred_rotmats.shape[0], 2, device=pred_rotmats.device),
                                          confidence=maxprob,
                                          angles_format="rotmat")
            particlesDataset.saveMd(outFname)

    """
    
--data.targetName TEST \
--data.halfset 1 \
--outFname /tmp/results.star \
--ckpt_path /tmp/supervised/lightning_logs/version_0/checkpoints/last.ckpt

"""
