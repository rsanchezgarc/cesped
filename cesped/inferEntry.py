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
import sys
import os.path as osp
import tempfile
from cesped.constants import default_configs_dir
from cesped.utils.cliBuilder import MyLightningCLI, _ckpt_path_argname, CheckpointLoader
from cesped.network.plModule import PlModel
from cesped.datamanager.plDataset import ParticlesDataModule


class _MyInferLightningCLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(f"--outFname", type=str, help="The name of the output file, ended in .star")



if __name__ == "__main__":
    from omegaconf import OmegaConf

    config_fnames = [
        osp.join(default_configs_dir, "defaultDataConfig.yaml"),
        osp.join(default_configs_dir, "defaultInferenceConfig.yaml"),
    ]
    with CheckpointLoader(config_fnames) as cpk: #TODO: Check if this works
        cli = _MyInferLightningCLI(model_class=PlModel, datamodule_class=ParticlesDataModule,
                                   save_config_callback=None,
                                   parser_kwargs={"default_env": True,"parser_mode":"omegaconf",
                                                 "default_config_files": cpk.config_fnames},
                                   run=False)
        outFname = cli.config["outFname"]
        assert outFname.endswith(".star"), (f'Error, --outFname {outFname} not valid. Needs '
                                                          f'to end in .star')
        assert osp.isdir(osp.dirname(outFname)), (f'Error, --outFname {outFname} not valid. Needs '
                                                          f'to end in .star')
        preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=cli.ckpt_path)
        particlesDataset = cli.datamodule.createDataset()
        for ids, (pred_rotmats, maxprob), metadata in preds:
            particlesDataset.updateMd(ids=ids, angles=pred_rotmats, shifts=None, confidence=maxprob,
                                      angles_format="rotmat")
        particlesDataset.saveMd(outFname)

    """
    
--data.targetName TEST \
--data.halfset 1 \
--outFname /tmp/results.star \
--ckpt_path /tmp/supervised/lightning_logs/version_0/checkpoints/last.ckpt

"""