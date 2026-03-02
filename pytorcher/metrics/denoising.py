"""
Denoising metrics implemented with PyTorch only.

This file provides two metrics implemented using torch operations:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (fast torch-based implementation)

Both classes inherit from the project's Metric base class and accept
batch tensors in (B, C, H, W) format. Inputs are converted to float tensors
and moved to the same device for calculations.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from pytorcher.metrics import Metric


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert input to torch float tensor.

    Accepts numpy arrays or torch tensors. Returns a torch.float32 tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(dtype=torch.float32)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, dtype: torch.dtype = torch.float32, device=None):
    """Create 2D gaussian kernel for SSIM (separable).

    Returns a tensor of shape (1, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel2d = g[:, None] @ g[None, :]
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.unsqueeze(0).unsqueeze(0)


class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio computed with torch ops.

    Args:
        name: optional metric name
        max_val: maximum possible pixel value (e.g., 1.0 or 255)
        eps: small epsilon to avoid divide-by-zero
    """

    def __init__(self, name: Optional[str] = None, max_val: float = 1.0, eps: float = 1e-10, bkg_val: Optional[float] = None):
        super().__init__(name or "psnr")
        self.max_val = float(max_val)
        self.eps = float(eps)
        self.bkg_val = bkg_val

    def update_state(self, y_true, y_pred, sample_weight=None):
        #
        if self.bkg_val is not None:
            mask = y_true > self.bkg_val
            y_true = torch.where(mask, y_true, torch.nan)
            y_pred = torch.where(mask, y_pred, torch.nan)
        # Convert to torch float tensors
        y_true_t = _as_float_tensor(y_true)
        y_pred_t = _as_float_tensor(y_pred)

        # Ensure shape (B, C, H, W)
        if y_true_t.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            y_true_t = y_true_t.unsqueeze(0)
        if y_pred_t.dim() == 3:
            y_pred_t = y_pred_t.unsqueeze(0)

        # Move predictions to same device as targets
        y_pred_t = y_pred_t.to(device=y_true_t.device)

        # compute per-image MSE over channels and spatial dims
        mse = torch.nanmean((y_true_t - y_pred_t) ** 2, dim=(1, 2, 3))
        psnr = 20.0 * torch.log10(torch.tensor(self.max_val, dtype=mse.dtype, device=mse.device) / torch.sqrt(mse + self.eps))

        batch_total = float(torch.sum(psnr).item())
        batch_count = float(psnr.numel())

        self._total += batch_total
        self._count += batch_count


class SSIM(Metric):
    """
    Structural Similarity Index (SSIM) implemented using torch operations.

    This implementation follows the standard windowed SSIM formula using a
    separable Gaussian window and computes mean SSIM per image across
    channels and spatial dimensions.

    Args:
        name: optional metric name
        max_val: maximum possible pixel value (e.g., 1.0 or 255)
        window_size: size of gaussian window (odd integer, default 11)
        sigma: gaussian sigma used for window (default 1.5)
    """

    def __init__(self, name: Optional[str] = None, K1: float = 0.01, K2: float = 0.03, L: float = 1.0, window_size: int = 11, sigma: float = 1.5, bkg_val: Optional[float] = None):
        super().__init__(name or "ssim")
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.L = float(L)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.bkg_val = bkg_val

    def nanvar(self, x: torch.Tensor, correction=1):

        dims = (1, 2, 3)

        mask = ~torch.isnan(x)
        count = mask.sum(dim=dims, keepdim=True)

        mean = torch.nanmean(x, dim=dims, keepdim=True)

        X_filled = torch.where(mask, x, mean)

        var = torch.sum((X_filled - mean) ** 2 * mask, dim=dims, keepdim=True) / (count - correction)

        return var

    def masked_cov(self, img1, img2, mask, correction=1):
        dims = (1, 2, 3)
        mask = mask.bool()

        n = mask.sum(dim=dims, keepdim=True)  # (B,1,1,1)

        # Zero out invalid pixels
        x = torch.where(mask, img1, torch.zeros_like(img1))
        y = torch.where(mask, img2, torch.zeros_like(img2))

        mean_x = x.sum(dim=dims, keepdim=True) / n.clamp(min=1)
        mean_y = y.sum(dim=dims, keepdim=True) / n.clamp(min=1)

        cov = ((x - mean_x) * (y - mean_y) * mask).sum(dim=dims, keepdim=True)

        cov = cov / (n - correction).clamp(min=1)

        return cov

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply background mask if specified
        if self.bkg_val is not None:
            mask = y_true != self.bkg_val
            y_true = torch.where(mask, y_true, torch.nan)
            y_pred = torch.where(mask, y_pred, torch.nan)

        # Convert to float tensors
        X = _as_float_tensor(y_true)
        Y = _as_float_tensor(y_pred)

        if X.dim() == 3:
            X = X.unsqueeze(0)
        if Y.dim() == 3:
            Y = Y.unsqueeze(0)

        Y = Y.to(device=X.device)

        # ensure shape (B, C, H, W)
        if X.shape[1] != Y.shape[1]:
            # try to handle (B, H, W, C) layout by permuting
            if X.dim() == 4 and X.shape[-1] == Y.shape[1]:
                X = X.permute(0, 3, 1, 2)
            if Y.dim() == 4 and Y.shape[-1] == X.shape[1]:
                Y = Y.permute(0, 3, 1, 2)

        C1 = (self.K1 * self.L) ** 2
        C2 = (self.K2 * self.L) ** 2

        mu1 = torch.nanmean(X, dim=(1, 2, 3), keepdim=True)
        mu2 = torch.nanmean(Y, dim=(1, 2, 3), keepdim=True)
        sigma1_sq = self.nanvar(X, correction=0)
        sigma2_sq = self.nanvar(Y, correction=0)
        sigma12 = self.masked_cov(X, Y, mask=mask, correction=0)

        ssim_per_image = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        batch_total = float(torch.sum(ssim_per_image).item())
        batch_count = float(ssim_per_image.numel())

        self._total += batch_total
        self._count += batch_count
