# Re-run the necessary code blocks after the code execution state was reset

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GaussianFilterBank(nn.Module):
    """
    A bank of gaussianl filters implemented in Pytorch
    """
    def __init__(self, in_channels, kernel_sizes, sigma_values, out_dim=None):
        super(GaussianFilterBank, self).__init__()
        max_kernel_size = max(kernel_sizes)

        self.in_channels = in_channels

        # Create a list to hold the Gaussian kernels
        kernels = []
        for kernel_size, sigma in zip(kernel_sizes, sigma_values):
            # Create 1D Gaussian kernel

            if sigma > 0:
                kernel_1D = self.gaussian_kernel(kernel_size, sigma)
                # Create 2D Gaussian kernel
                kernel_2D = kernel_1D[:, None] * kernel_1D[None, :]
            else:
                kernel_2D = torch.tensor([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]])
                kernel_size = 3

            # Pad kernel with zeros to have the same size as max_kernel_size
            padding = max_kernel_size - kernel_size
            pad_left = padding // 2
            pad_right = padding - pad_left
            kernel_2D = F.pad(kernel_2D, (pad_left, pad_right, pad_left, pad_right))

            # Add channel dimension and append to the list
            kernels.append(kernel_2D[None, None, :, :])

        # Stack the Gaussian kernels into a single tensor
        all_kernels = torch.cat(kernels, dim=0)

        # Repeat the kernels for each channel in the image
        all_kernels = all_kernels.repeat(self.in_channels, 1, 1, 1)

        # Register as a buffer so PyTorch can recognize it as a model parameter
        self.register_buffer('all_kernels', all_kernels)

        if out_dim and out_dim != self.all_kernels.shape[0]:
            self.lastLayer = nn.Conv2d(in_channels=self.all_kernels.shape[0], out_channels=out_dim,
                                       kernel_size=max(kernel_sizes), padding="same")
        else:
            self.lastLayer = nn.Identity()
    @staticmethod
    def gaussian_kernel(size, sigma):
        coords = torch.arange(size).float()
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def forward(self, x):
        # Perform depthwise convolution
        x = F.conv2d(x, self.all_kernels, groups=self.in_channels, padding="same")
        return self.lastLayer(x)

def _visual_test():
    # Create a synthetic "realistic" image using gradients and random noise
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    x, y = np.meshgrid(x, y)
    import skimage
    image_realistic = skimage.data.camera() #np.sin(x * np.pi) * np.sin(y * np.pi) + 0.2 * np.random.rand(64, 64)

    # Convert to PyTorch tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image_realistic, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Initialize GaussianFilterBank
    kernel_sizes = [0, 3, 11, 19]
    sigma_values = [0, 0.8, 2.0, 4.]
    gaussian_filter_bank = GaussianFilterBank(1, kernel_sizes, sigma_values)

    # Apply the Gaussian filters
    filtered_images = gaussian_filter_bank(image_tensor)

    # Plotting the original and filtered images
    fig, axs = plt.subplots(1, len(kernel_sizes), figsize=(15, 15))

    # Plot filtered images
    for i in range(len(kernel_sizes)):
        axs[i ].imshow(filtered_images[0, i].detach().numpy(), cmap='gray')
        axs[i ].set_title(f"Filtered with {kernel_sizes[i]}x{kernel_sizes[i]} kernel")
        axs[i].axis('off')

    plt.show()

if __name__ == "__main__":
    _visual_test()
