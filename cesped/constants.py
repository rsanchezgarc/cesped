import os
from os import path as osp
from typing import List

from omegaconf import OmegaConf

RELION_EULER_CONVENTION: str = "ZYZ"
""" Euler convention used by Relion. Rot, Tilt and Psi angles"""
RELION_ANGLES_NAMES: List[str] = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
""" Euler angles names in Relion. Rot, Tilt and Psi correspond to rotations on Z, Y and Z"""
RELION_SHIFTS_NAMES: List[str] = ['rlnOriginXAngst', 'rlnOriginYAngst']
""" Image shifts names in Relion. They are measured in Å (taking into account the sampling rate (aka pixel size) """
RELION_PRED_POSE_CONFIDENCE_NAME: str = 'rlnParticleFigureOfMerit'
""" The name of the metadata field used to weight the particles for the volume reconstruction"""
RELION_ORI_POSE_CONFIDENCE_NAME: str = 'rlnMaxValueProbDistribution'
""" The name of the metadata field with the estimated pose probability"""


_dirname = osp.dirname(__file__)
_filename = osp.join(osp.abspath(_dirname), "configs")
default_configs_dir = os.environ.get("CESPED_CONFIGDIR", _filename)
"""The directory where the default configuration files are. You can set the environmental variable CESPED_CONFIGDIR
to the configs directory
"""

_defaultDataConf = OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml"))
defaultBenchmarkDir = osp.expanduser(_defaultDataConf["data"]["benchmarkDir"])
"""The deafult benchmark directory, where entries will be saved"""

_defaultRelionConf = OmegaConf.load(osp.join(default_configs_dir, "defaultRelionConfig.yaml"))
relionBinDir = osp.expanduser(_defaultRelionConf["Relion"]["relionBinDir"])
"""The Relion bin directory used to compute reconstruction in evaluation"""

mpirunCmd = osp.expanduser(_defaultRelionConf["Relion"]["mpirun"])
"""The mpirun command to execute relion with several workers"""
