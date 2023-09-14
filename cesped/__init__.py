"""
This module provides utilities to load and train models for Cryo-EM Supervised Pose Inference models
"""
import os
import os.path as osp
from omegaconf import OmegaConf


default_configs_dir = os.environ.get("CESPED_CONFIGDIR",
                                     osp.abspath(osp.join(osp.dirname(osp.dirname(__file__)), "configs")))

_defaultDataConf = OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml"))

defaultBenchmarkDir = osp.expanduser(_defaultDataConf["data"]["benchmarkDir"])
"""The deafult benchmark directory, where entries will be saved"""

_defaultRelionConf = OmegaConf.load(osp.join(default_configs_dir, "defaultRelionConfig.yaml"))

relionBinDir = osp.expanduser(_defaultRelionConf["Relion"]["relionBinDir"])
"""The Relion bin directory"""

mpirunCmd = _defaultRelionConf["Relion"]["mpirun"]
"""The mpirun command to execute relion with several workers"""
