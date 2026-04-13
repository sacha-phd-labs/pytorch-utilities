import torch
import torch.nn.functional as F

def gaussian_kernel_2d(kernel_size: int, sigma: float, device="cpu"):
    """
    Returns a 2D Gaussian kernel normalized to sum to 1.
    Shape: (1, 1, k, k) → ready for conv2d
    """
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, kernel_size, kernel_size)

def gaussian_kernel_1d(sigma, kernel_size, device):
    """
    Returns: (1, 1, 1, K) kernel for grouped conv
    """
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2  # (K,)

    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()

    return g.view(1, 1, 1, kernel_size)


def apply_gaussian_psf_reflect(image, sigma):
    k = int(6 * sigma + 1) | 1
    pad = k // 2

    image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    kernel = gaussian_kernel_2d(k, sigma, device=image.device)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)

    return F.conv2d(image, kernel, groups=image.shape[1])

def apply_rowwise_gaussian_psf(image, sigma, kernel_size=None):
    """
    image: (B, C, H, W)
    sigma: single scalar sigma shared across all rows
    """
    B, C, H, W = image.shape
    device = image.device

    if kernel_size is None:
        kernel_size = int(6 * float(sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Build kernel
    kernel = gaussian_kernel_1d(torch.as_tensor(sigma, device=device), kernel_size, device)

    # Reshape image so each row becomes its own channel group
    x = image.permute(0, 2, 1, 3)   # (B, H, C, W)
    x = x.reshape(1, B * H * C, 1, W)  # treat each row as separate channel

    # Repeat kernel for batch, rows, and channels
    kernel = kernel.repeat(B * H * C, 1, 1, 1)  # (B*H*C,1,1,K)

    x = F.pad(x, (kernel_size // 2, kernel_size // 2, 0, 0), mode="reflect")
    out = F.conv2d(x, kernel, groups=B * H * C)

    # Reshape back
    out = out.view(B, H, C, W).permute(0, 2, 1, 3)  # (B,C,H,W)
    return out

def apply_columnwise_gaussian_psf(image, sigma, kernel_size=None):
    return apply_rowwise_gaussian_psf(image.transpose(2, 3), sigma, kernel_size).transpose(2, 3)
