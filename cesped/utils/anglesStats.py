from typing import Union, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from cesped.constants import RELION_EULER_CONVENTION
from cesped.network.image2sphere import compute_trace, compute_symmetry_group_matrices, rotation_error_rads


def closest_symmetric_rotation(rot_pred, rot_true, symmetry):
    '''
    rot_true, rot_pred are tensors of shape (B,3,3)
    symmetry: the name of the point symmetry group.
    returns the closest symmetric rotation matrices to rot_true, tensor of shape (B,3,3)
    '''
    assert rot_pred.shape[0] == rot_true.shape[0], "Error, different batch size for rot_pred and rot_true"
    assert symmetry.upper() != "C1", "Error c1 is not supported"
    symmetry_group = compute_symmetry_group_matrices(symmetry)

    B = rot_true.shape[0]  # Batch size

    # Expand dimensions to prepare for broadcasting
    rot_true_expanded = rot_true[:, None, ...]

    # Compute all possible symmetric rotations of rot_pred
    all_pred_rotations = symmetry_group[None, ...] @ rot_pred[:, None, ...]

    # Compute traces for all combinations
    traces = compute_trace(all_pred_rotations, rot_true_expanded)

    # Compute errors for all combinations
    errors = torch.arccos(torch.clamp((traces - 1) / 2, -1, 1))

    # Find the index of the minimum error for each sample in the batch
    min_error_indices = torch.argmin(errors, dim=1)

    # Select the closest symmetric rotation for each sample in the batch
    closest_rotations = all_pred_rotations[torch.arange(B), min_error_indices]
    return closest_rotations

def computeAngularError(predEulers, trueEulers, confidence=None, symmetry="c1") \
    -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """

    Args:
        predEulers:
        trueEulers:
        confidence:
        symmetry:

    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]: The angular error and weighted angular\
          error in degrees.
    """
    a_isTensor = torch.is_tensor(predEulers)
    b_isTensor = torch.is_tensor(predEulers)
    assert a_isTensor and b_isTensor or not (a_isTensor and b_isTensor), "Error, both inputs should be of the same type"
    predRotM = R.from_euler(RELION_EULER_CONVENTION, predEulers, degrees=True).as_matrix()
    trueRotM = R.from_euler(RELION_EULER_CONVENTION, trueEulers, degrees=True).as_matrix()

    predRotM = torch.from_numpy(predRotM.astype(np.float32))
    trueRotM = torch.from_numpy(trueRotM.astype(np.float32))
    if symmetry.upper() != "C1":
        predRotM = closest_symmetric_rotation(predRotM, trueRotM, symmetry=symmetry)
    error = rotation_error_rads(predRotM, trueRotM)
    error = torch.rad2deg(error)
    w_error = torch.ones_like(error) * torch.nan
    if confidence is not None:
        totalConf = confidence.sum()
        confidence = torch.from_numpy(confidence.astype(np.float32))
        w_error = (confidence * error).sum() / totalConf
    else:
        totalConf = None

    error = error.mean()
    if not a_isTensor:
        error = error.numpy()
        w_error = w_error.numpy()
    return error, w_error, totalConf
