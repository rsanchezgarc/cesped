import os
import os.path as osp
import shutil
import subprocess
import sys

import psutil
from omegaconf import OmegaConf

from cesped.constants import default_configs_dir
from cesped.utils.cliBuilder import MyLightningCLI, CheckpointLoader
from cesped.network.plModule import PlModel

# from cesped.datamanager.dataManager import ParticlesDataModule
from cesped.datamanager.plDataset import ParticlesDataModule



_skip_other_half_inference = "skip_other_half_inference"
class _TrainCLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(f"--{_skip_other_half_inference}", action="store_true", help="If true, "
                                             "after training finishes, inference is NOT called on the other half dataset")
def _copyCodeForReproducibility(logdir):
    """
    Copy the code to the logdir so that reproducibility is ensured

    Args:
        logdir: The directory were the code will be saved

    Returns:

    """
    copycodedir = osp.join(logdir, "code")
    os.makedirs(copycodedir, exist_ok=True)
    copycodedir = osp.join(copycodedir, "cesped")

    modulePath = os.path.abspath(sys.modules[__name__].__file__)
    rootPath = osp.dirname(osp.dirname(modulePath))

    for root, dirs, files in os.walk(rootPath):
        # Iterate through all folders
        for directory in dirs:
            # Create the corresponding directory in the target path
            source_folder = os.path.join(root, directory)
            target_folder = source_folder.replace(rootPath, copycodedir)
            os.makedirs(target_folder, exist_ok=True)
        # Iterate through all Python files
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                # Copy the Python file to the corresponding directory in the target path
                source_file = os.path.join(root, file)
                target_file = source_file.replace(rootPath, copycodedir)
                shutil.copy2(source_file, target_file)

    # Copy the command
    fname = osp.join(logdir, "command.txt")
    with open(fname, "w") as f:
        f.write(" ".join(sys.argv))


if __name__ == "__main__":


    #The order matters in config_fnames Load first Data, then Model and lastly trainer.
    config_fnames = [osp.join(default_configs_dir, "defaultDataConfig.yaml"),
                     osp.join(default_configs_dir, "defaultModelConfig.yaml"),
                     osp.join(default_configs_dir, "defaultTrainerConfig.yaml"),
                     osp.join(default_configs_dir, "defaultOptimizerConfig.yaml"),
                     ]

    with CheckpointLoader(config_fnames) as cpk:
        cli = _TrainCLI(model_class=PlModel,
                        datamodule_class=ParticlesDataModule,
                        parser_kwargs={"default_env": True, "parser_mode":"omegaconf",
                                            "default_config_files": cpk.config_fnames},
                        run=False)
        trainer = cli.trainer
        logdir = cli.trainer.log_dir
        if trainer.is_global_zero:
            _copyCodeForReproducibility(logdir)

        trainer.fit(cli.model, cli.datamodule, ckpt_path=cpk.ckpt_path)

        cli.trainer.strategy.barrier()
        if not cli.config.get(_skip_other_half_inference, False) and trainer.is_global_zero:
            otherHalf = (cli.config["data"]["halfset"] + 1) % 2
            print(f"Running inference on the halfset {otherHalf} ")

            cmd = [
                "python", "-m",  "cesped.inferEntry",
                "--data.targetName" , cli.config["data"]["targetName"],
                "--data.halfset", str(otherHalf),
                "--data.num_data_workers", str(cli.config["data"]["num_data_workers"]),
                "--data.batch_size", str(cli.config["data"]["batch_size"]),
                "--trainer.devices", "1", #Multiple devices will cause issues with ports.
                "--outFname", osp.join(logdir, f"predictions_{otherHalf}.star"),
                "--ckpt_path", getattr(cli.trainer.checkpoint_callback, "best_model_path", "best")]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=osp.dirname(osp.dirname(__file__)))



"""
-c ../configs/defaultTrainerConfig.yaml -c ../configs/defaultDataConfig.yaml -c ./configs/defaultModelConfig.yaml \ 
--data.targetName TEST --data.targetName TEST --data.halfset 0
"""