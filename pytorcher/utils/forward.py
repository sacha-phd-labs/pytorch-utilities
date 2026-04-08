import torch
import numpy as np
try:
    from torch_radon import Radon
except ImportError:
    print("torch_radon is not installed. Please install it to use PetForwardRadon.")

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_columnwise_gaussian_psf

class PetForwardRadon(torch.nn.Module):

    def __init__(
            self,
            n_angles=300,
            scanner_radius_mm=300,
            gaussian_PSF_fwhm_mm=4.0,
            voxel_size_mm=2.0
    ):
        super(PetForwardRadon, self).__init__()
        self.n_angles = n_angles
        self.scanner_radius_mm = scanner_radius_mm
        self.gaussian_PSF_fwhm_mm = gaussian_PSF_fwhm_mm
        self.voxel_size_mm = voxel_size_mm
        if not isinstance(self.voxel_size_mm, (list, tuple)):
            self.voxel_size_mm = [self.voxel_size_mm] * 2
        #
        assert torch.cuda.is_available(), "CUDA is not available. PetForwardRadon requires CUDA support."
        self.to(torch.device('cuda'))

    def get_radon_operator(self, image):
        self.theta = torch.linspace(0., np.pi, self.n_angles, dtype=image.dtype, device=image.device)
        self.in_size = image.shape[-1]
        self.radon = Radon(
            resolution=image.shape[-1],
            angles=self.theta,
            clip_to_circle=True,
            det_count=image.shape[-1],
            det_spacing=1.0
        )

    def radon_forward(self, image):
        if not hasattr(self, 'radon') or self.in_size != image.shape[-1]:
            self.get_radon_operator(image)
        out = self.radon.forward(image)
        return out.transpose(-2, -1)

    def pad_(self, img):
        # Pad img to fit in scanner radius
        img_size = torch.tensor(img.shape).max()
        if 2.0 * self.scanner_radius_mm >= img_size * max(self.voxel_size_mm):
            pad_x = int(
                (2.0 * self.scanner_radius_mm - img.shape[-2] * self.voxel_size_mm[0]) / (2 * self.voxel_size_mm[0])
            )
            pad_y = int(
                (2.0 * self.scanner_radius_mm - img.shape[-1] * self.voxel_size_mm[1]) / (2 * self.voxel_size_mm[1])
            )
            img = torch.nn.functional.pad(
                img, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0
            )
        else:
            raise ValueError("Image size exceeds scanner field of view.")
        return img
    
    def forward(self, image, attenuation_map=None, scale=None, voxel_size_mm=None):
        if voxel_size_mm is not None:
            if not isinstance(voxel_size_mm, (list, tuple)):
                voxel_size_mm = [voxel_size_mm] * 2
            self.voxel_size_mm = voxel_size_mm
        assert self.voxel_size_mm is not None, "Voxel size must be specified either during initialization or in the forward method."
        #
        if attenuation_map is not None:
            assert image.shape == attenuation_map.shape, "Image and attenuation map must have the same shape."

        # Image domain PSF
        if self.gaussian_PSF_fwhm_mm is not None:
            sigma_mm = self.gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
            image = apply_gaussian_psf_reflect(image, sigma_mm / self.voxel_size_mm[0]) # assuming square pixels

        # Pad image to fit in scanner radius
        image = self.pad_(image)

        # Radon transform
        if not hasattr(self, 'radon') or self.in_size != image.shape[-1]:
            self.get_radon_operator(image)
        sinogram = self.radon_forward(image)

        # Apply attenuation if provided
        if attenuation_map is not None:
            assert self.voxel_size_mm[0] == self.voxel_size_mm[1], "Voxel size must be isotropic for attenuation application."
            # We pass the attenuation map through the same padding as the image to ensure they are aligned
            attenuation_map = self.pad_(attenuation_map)
            # The attenuation map is typically in units of cm^-1, so we need to convert it to mm^-1 for homogeneous units
            attenuation_scale_factor = 0.01 # cm^-1 to mm^-1
            if self.gaussian_PSF_fwhm_mm is not None:
                # In clinical context, the attenuation map is obtained from a CT scan and smoothed with the same PSF as the image
                # to imitate the PSF of the PET system.
                sigma_mm = self.gaussian_PSF_fwhm_mm / (2.0 * (torch.log(torch.tensor(2.0)))**0.5)
                attenuation_map = apply_gaussian_psf_reflect(attenuation_map, sigma_mm / self.voxel_size_mm[0]) # assuming square pixels
            # The attenuation along each ray can be computed by applying the radon transform to the attenuation map.
            att_sino = self.radon_forward(attenuation_map * attenuation_scale_factor)
            # The expected counts along each ray are then scaled by the exponential of the negative attenuation, following the Beer-Lambert law.
            sinogram = sinogram * torch.exp(-att_sino * self.voxel_size_mm[0]) # assuming square pixels

        # PSF in sinogram domain
        if self.gaussian_PSF_fwhm_mm is not None: 
            sigma_mm = self.gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
            # 1D sigma in sinogram domain on each row
            bin_widths_mm = torch.abs(self.voxel_size_mm[0] * torch.cos(self.theta * torch.pi / 180.0)) + torch.abs(self.voxel_size_mm[1] * torch.sin(self.theta * torch.pi / 180.0))
            sigma_sino_mm = sigma_mm / bin_widths_mm
            sinogram = apply_columnwise_gaussian_psf(sinogram, sigmas=sigma_sino_mm)

        # Scale sinogram to match expected counts
        if scale is not None:
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale, device=sinogram.device)
            if scale.ndim == 0:
                scale = scale.repeat(sinogram.shape[0]) # make it a vector of shape (batch_size,)
            sinogram = sinogram * scale.view(-1, 1, 1, 1) # scale each row accordingly
    
        return sinogram


if __name__ == "__main__":

    device = torch.device('cuda')

    from pytorcher.utils.fbp import iradon

    image = torch.ones((2, 16, 160, 160), device=device)
    # image[:, :, 32:96, 32:96] = 10.0
    attenuation_map = torch.zeros((2, 16, 160, 160), device=device)

    scanner_radius_mm = 300
    voxel_size_mm = 2.0

    pet_forward = PetForwardRadon(
        n_angles=300,
        scanner_radius_mm=scanner_radius_mm,
        gaussian_PSF_fwhm_mm=4.0,
        voxel_size_mm=voxel_size_mm
    )

    sinogram = pet_forward.forward(image, attenuation_map=attenuation_map, scale=0.02)

    print(sinogram.sum())

    sinogram = torch.poisson(sinogram)

    recon = iradon(sinogram, theta=pet_forward.theta.cpu().numpy(), output_size=image.shape[-1], filter='ramp', circle=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image[0, 0, ...].cpu().squeeze(), cmap='gray_r')
    ax[0].set_title('Original Image')

    ax[1].imshow(sinogram[0, 0, ...].cpu().squeeze(), cmap='gray')
    ax[1].set_title('Sinogram')

    ax[2].imshow(recon[0, 0, ...].cpu().squeeze(), cmap='gray_r')
    ax[2].set_title('FBP Reconstruction')

    plt.show()