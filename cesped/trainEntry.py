import os
import os.path as osp
import shutil
import sys

import psutil

from cesped.constants import default_configs_dir
from cesped.utils.cliBuilder import MyLightningCLI
from cesped.network.plModule import PlModel
from cesped.particlesDataset import ParticlesDataModule

def _copyCodeForReproducibility(logdir):
    """
    Copy the code to the logdir so that reproducibility is ensured

    Args:
        logdir: The directory were the code will be saved

    Returns:

    """
    copycodedir = osp.join(logdir, "code")
    os.makedirs(copycodedir, exist_ok=True)
    copycodedir = osp.join(copycodedir, "cryoSolver")

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
            if file.endswith(".py"):
                # Copy the Python file to the corresponding directory in the target path
                source_file = os.path.join(root, file)
                target_file = source_file.replace(rootPath, copycodedir)
                shutil.copy2(source_file, target_file)

    # Copy the command
    fname = osp.join(logdir, "command.txt")
    with open(fname, "w") as f:
        f.write(" ".join(sys.argv))

if __name__ == "__main__":

    # os.environ["PL_CONFIG"] let's you to set config files using env variables

    #The order matters in config_fnames Load first Data, then Model and lastly trainer.
    config_fnames = [osp.join(default_configs_dir, "defaultDataConfig.yaml"),
                     osp.join(default_configs_dir, "defaultModelConfig.yaml"),
                     osp.join(default_configs_dir, "defaultTrainerConfig.yaml"),
                     osp.join(default_configs_dir, "defaultOptimizerConfig.yaml"),
                     ]

    cli = MyLightningCLI(model_class=PlModel, datamodule_class=ParticlesDataModule,
                         parser_kwargs={"default_env": True, "parser_mode":"omegaconf",
                                        "default_config_files": config_fnames},
                         run=False)
    trainer = cli.trainer
    logdir = cli.trainer.log_dir
    if trainer.is_global_zero:
        _copyCodeForReproducibility(logdir)
    trainer.fit(cli.model, cli.datamodule)

"""
-c ../configs/defaultTrainerConfig.yaml -c ../configs/defaultDataConfig.yaml -c ./configs/defaultModelConfig.yaml \ 
--data.targetName TEST --data.targetName TEST --data.halfset 0
"""