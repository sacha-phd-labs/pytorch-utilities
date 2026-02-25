import torch

from deepinv.physics.functional import Radon as DeepInvRadon

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_columnwise_gaussian_psf

class PetForwardRadon(torch.nn.Module):

    def __init__(
            self,
            n_angles=300,
            scanner_radius_mm=300,
            gaussian_PSF_fwhm_mm=4.0,
            voxel_size_mm=2.0,
            device=None
    ):
        super(PetForwardRadon, self).__init__()
        self.n_angles = n_angles
        self.scanner_radius_mm = scanner_radius_mm
        self.gaussian_PSF_fwhm_mm = gaussian_PSF_fwhm_mm
        self.voxel_size_mm = voxel_size_mm
        if not isinstance(self.voxel_size_mm, (list, tuple)):
            self.voxel_size_mm = [self.voxel_size_mm] * 2
        #
        if device is not None:
            self.to(device)

    def get_radon_operator(self, image):
        self.theta = torch.linspace(0., 180., self.n_angles, dtype=image.dtype, device=image.device)
        self.in_size = image.shape[-1]
        self.radon = DeepInvRadon(
            in_size=image.shape[-1],
            theta=self.theta,
            circle=False,
        )

    def pad_(self, img):
        # Pad img to fit in scanner radius
        img_size = torch.tensor(img.shape).max()
        if torch.sqrt(torch.tensor(2.0)) * self.scanner_radius_mm >= img_size * max(self.voxel_size_mm):
            pad_x = int(
                (torch.sqrt(torch.tensor(2.0)) * self.scanner_radius_mm - img.shape[-2] * self.voxel_size_mm[0]) / (2 * self.voxel_size_mm[0])
            )
            pad_y = int(
                (torch.sqrt(torch.tensor(2.0)) * self.scanner_radius_mm - img.shape[-1] * self.voxel_size_mm[1]) / (2 * self.voxel_size_mm[1])
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
        sinogram = self.radon.forward(image)

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
            att_sino = self.radon.forward(attenuation_map * attenuation_scale_factor)
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

    device = torch.device('cpu')

    from tools.image.castor import read_castor_binary_file
    import os

    from pytorcher.utils.fbp import FBPReconstructor

    # Example usage
    # dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"
    # image = read_castor_binary_file(os.path.join(dest_path, 'object', 'gt_web_after_scaling.hdr'), reader='numpy')
    # image = torch.from_numpy(image).unsqueeze(0).float().to(device) # shape (1, 1, H, W)
    # attenuation_map = read_castor_binary_file(os.path.join(dest_path, 'object', 'attenuat_brain_phantom.hdr'), reader='numpy')
    # attenuation_map = torch.from_numpy(attenuation_map).unsqueeze(0).float().to(device)

    image = torch.zeros((1, 16, 128, 128), device=device)
    image[:, :, 32:96, 32:96] = 1.0
    attenuation_map = torch.zeros((1, 16, 128, 128), device=device)

    scanner_radius_mm = 300
    voxel_size_mm = 2.0

    pet_forward = PetForwardRadon(
        n_angles=300,
        scanner_radius_mm=scanner_radius_mm,
        gaussian_PSF_fwhm_mm=4.0,
        voxel_size_mm=voxel_size_mm,
        device=device
    )

    sinogram = pet_forward.forward(image, attenuation_map=attenuation_map, scale=0.02)

    sinogram = torch.poisson(sinogram)

    sinogram = sinogram.transpose(-2, -1) # shape (1, A, D)

    fbp_reconstructor = FBPReconstructor(
        n_angles=300,
        scanner_radius_mm=scanner_radius_mm,
        voxel_size_mm=voxel_size_mm,
        image_size=image.shape[-2:],
        device=device
    )

    recon = fbp_reconstructor.forward(sinogram)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image.cpu().squeeze(), cmap='gray_r')
    ax[0].set_title('Original Image')

    ax[1].imshow(sinogram.cpu().squeeze(), cmap='gray')
    ax[1].set_title('Sinogram')

    ax[2].imshow(recon.cpu().squeeze(), cmap='gray_r')
    ax[2].set_title('FBP Reconstruction')

    plt.show()