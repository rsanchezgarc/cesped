import numpy as np
import torch


def data_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x  # Do nothing if it's already a NumPy array

    if isinstance(x, torch.Tensor):
        # Move tensor to CPU if it's on GPU
        if x.is_cuda:
            x = x.cpu()

        # Detach it from the computation graph if it's attached
        if x.requires_grad:
            x = x.detach()

        return x.numpy()

    raise TypeError(f"Input type {type(x)} is not supported. Only PyTorch tensors and NumPy arrays are valid.")


