from pytorcher.metrics import Metric
from typing import Optional
import torch

from pytorcher.metrics.base import _as_float_tensor

from torchmetrics.functional.image.dists import deep_image_structure_and_texture_similarity
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity

class PerceptualCommons(Metric):

    def prepare(self, y_true, y_pred, sample_weight = None):
        # Convert to torch float tensors
        y_true_t = _as_float_tensor(y_true)
        y_pred_t = _as_float_tensor(y_pred)

        # Ensure shape (B, 3, H, W) : DISTS's VGG is trained on natural RGB images
        assert y_true_t.ndim == y_pred_t.ndim == 4, "Expected 4D tensors (B, C, H, W)"
        if y_pred_t.size(1) == 1:  # If grayscale, repeat channels to make 3
            y_pred_t = y_pred_t.repeat(1, 3, 1, 1)
        if y_true_t.size(1) == 1:
            y_true_t = y_true_t.repeat(1, 3, 1, 1)
        assert y_true_t.size(1) == y_pred_t.size(1) == 3, "Expected 3 channels (RGB)"            

        # Move predictions to same device as targets
        y_pred_t = y_pred_t.to(device=y_true_t.device)

        return y_true_t, y_pred_t

class DISTS(PerceptualCommons):
    """
    DISTS metric for image quality assessment.

    Args:
        name: optional metric name
        eps: small epsilon to avoid divide-by-zero
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "dists")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_t, y_pred_t = self.prepare(y_true, y_pred, sample_weight)

        with torch.no_grad():
            batch_dists = deep_image_structure_and_texture_similarity(y_pred_t, y_true_t, reduction="sum")
        self._total += batch_dists.item()
        self._count += y_pred_t.size(0)

class LPIPS(PerceptualCommons):
    """
    LPIPS metric for image quality assessment.

    Args:
        name: optional metric name
        eps: small epsilon to avoid divide-by-zero
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "lpips")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_t, y_pred_t = self.prepare(y_true, y_pred, sample_weight)

        with torch.no_grad():
            batch_lpips = learned_perceptual_image_patch_similarity(y_pred_t, y_true_t, reduction="sum", net_type="alex")
        self._total += batch_lpips.item()
        self._count += y_pred_t.size(0)

if __name__ == "__main__":
    import time
    # Example usage
    metric = LPIPS()
    torch.manual_seed(0)
    y_true = torch.rand(8, 1, 160, 160)  # batch of 8 images
    y_pred = torch.rand(8, 1, 160, 160)
    start_time = time.time()
    metric.update_state(y_true, y_pred)
    end_time = time.time()
    print(f"LPIPS: {metric.result()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")