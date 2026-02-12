from pytorcher.metrics import PSNR, SSIM
from pytorcher.utils.processing import normalize_batch 
from tools.image.metrics import PSNR as PSNR_np, SSIM as SSIM_np
import numpy as np
import torch


class TestDenoisingMetrics:

    def get_dummy_image(self, seed=42):
        np.random.seed(seed)
        im1 = np.zeros((64, 64)) # dummy image
        im1[16:48, 16:48] = 1.0 # add a square
        im2 = np.where(im1 > 0, im1 + 0.1 * np.random.rand(64, 64), im1) # add noise only to the square
        mask = im1 > 0.0
        return im1, im2, mask

    def test_psnr(self):
        im1, im2, mask = self.get_dummy_image()
        psnr_np = PSNR_np(im1.copy(), im2.copy(), normalize=True, mask=mask)
        im1 = torch.tensor(im1).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        im2 = torch.tensor(im2).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        psnr_torch = PSNR(max_val=1.0)
        psnr_torch.update_state(normalize_batch(im1), normalize_batch(im2))
        psnr_torch = psnr_torch.result()
        assert np.isclose(psnr_np, psnr_torch, atol=1e-5), f"PSNR mismatch: {psnr_np} vs {psnr_torch}"

    def test_ssim(self):
        im1, im2, mask = self.get_dummy_image()
        ssim_np = SSIM_np(im1.copy(), im2.copy(), mask=mask)
        im1 = torch.tensor(im1).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        im2 = torch.tensor(im2).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0) # shape (1, 1, H, W)
        ssim_torch = SSIM(L=1.0)
        ssim_torch.update_state(normalize_batch(im1), normalize_batch(im2))
        ssim_torch = ssim_torch.result()
        assert np.isclose(ssim_np, ssim_torch, atol=1e-5), f"SSIM mismatch: {ssim_np} vs {ssim_torch}"

if __name__ == "__main__":
    test = TestDenoisingMetrics()
    test.test_psnr()
    test.test_ssim()
    print("All tests passed!")
