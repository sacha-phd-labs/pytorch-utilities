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

def gaussian_kernel_1d_per_row(sigmas, kernel_size, device):
    """
    sigmas: (H,) tensor
    Returns: (H, 1, 1, K) kernels for grouped conv
    """
    H = sigmas.shape[0]
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2  # (K,)

    kernels = []
    for sigma in sigmas:
        g = torch.exp(-(ax**2) / (2 * sigma**2))
        g = g / g.sum()
        kernels.append(g)

    kernels = torch.stack(kernels)              # (H, K)
    return kernels.view(H, 1, 1, kernel_size)   # (H,1,1,K)


def apply_gaussian_psf_reflect(image, sigma):
    k = int(6 * sigma + 1) | 1
    pad = k // 2

    image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    kernel = gaussian_kernel_2d(k, sigma, device=image.device)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)

    return F.conv2d(image, kernel, groups=image.shape[1])

def apply_rowwise_gaussian_psf(image, sigmas, kernel_size=None):
    """
    image: (B, C, H, W)
    sigmas: (H,) — one sigma per row
    """
    B, C, H, W = image.shape
    device = image.device

    if kernel_size is None:
        max_sigma = sigmas.max().item()
        kernel_size = int(6 * max_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Build kernels
    kernels = gaussian_kernel_1d_per_row(sigmas.to(device), kernel_size, device)
    # (H,1,1,K)

    # Reshape image so each row becomes its own channel group
    x = image.permute(0, 2, 1, 3)   # (B, H, C, W)
    x = x.reshape(1, B * H * C, 1, W)  # treat each row as separate channel

    # Repeat kernels for batch and channels
    kernels = kernels.repeat(B * C, 1, 1, 1)  # (B*C*H,1,1,K)

    padding = (0, kernel_size // 2)  # pad only in W

    x = F.pad(x, (kernel_size // 2, kernel_size // 2, 0, 0), mode="reflect")
    out = F.conv2d(x, kernels, groups=B * H * C)

    # Reshape back
    out = out.view(B, H, C, W).permute(0, 2, 1, 3)  # (B,C,H,W)
    return out
