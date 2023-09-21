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
from cesped.utils.cliBuilder import MyLightningCLI
from cesped.network.plModule import PlModel
from cesped.particlesDataset import ParticlesDataModule

_ckpt_path_argname = "ckpt_path"

class _MyInferLightningCLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(f"--{_ckpt_path_argname}", type=str, help="The model checkpoint to load")
        parser.add_argument(f"--outFname", type=str, help="The name of the output file, ended in .star")



if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = {}
    conf.update(OmegaConf.load(osp.join(default_configs_dir, "defaultInferenceConfig.yaml")))
    conf.update(OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml")))

    with tempfile.TemporaryDirectory() as tmpdir:
        if f"--{_ckpt_path_argname}" in sys.argv:
            checkpointName = sys.argv[sys.argv.index(f"--{_ckpt_path_argname}") + 1]
            checkpointName = osp.abspath(osp.expanduser(checkpointName))
            model_config_fname = osp.join(osp.dirname(osp.dirname(checkpointName)), "config.yaml")
            assert osp.isfile(model_config_fname), f"Error, {model_config_fname} does not exists"
            conf.update(dict(model=OmegaConf.load(model_config_fname)["model"]))
        else:
            raise RuntimeError(f"Error, --{_ckpt_path_argname} argument missing")

        config_fname = osp.join(tmpdir, "final_config.yaml")
        with open(config_fname, "w") as f:
            OmegaConf.save(conf, f)

        cli = _MyInferLightningCLI(model_class=PlModel, datamodule_class=ParticlesDataModule,
                                   save_config_callback=None,
                                   parser_kwargs={"default_env": True,"parser_mode":"omegaconf",
                                                 "default_config_files": [config_fname]},
                                   run=False)
        outFname = cli.config["outFname"]
        assert outFname.endswith(".star"), (f'Error, --outFname {outFname} not valid. Needs '
                                                          f'to end in .star')
        assert osp.isdir(osp.dirname(outFname)), (f'Error, --outFname {outFname} not valid. Needs '
                                                          f'to end in .star')
        preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=cli.config[_ckpt_path_argname])
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