from typing import List

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