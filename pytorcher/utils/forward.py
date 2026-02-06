import torch

from deepinv.physics.functional import Radon as DeepInvRadon

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_rowwise_gaussian_psf

def pet_forward_radon(
        image,
        attenuation_map=None,
        n_angles=300,
        scanner_radius_mm=300,
        gaussian_PSF_fwhm_mm=None,
        voxel_size_mm=1.0,
        scale=None # scale = acquisition_time * np.log(2) / half_life
):
    #
    if not isinstance(voxel_size_mm, (list, tuple)):
        voxel_size_mm = [voxel_size_mm] * 2
    
    def pad_(img):
        # Pad img to fit in scanner radius
        img_size = torch.tensor(img.shape).max()
        if torch.sqrt(torch.tensor(2.0)) * scanner_radius_mm >= img_size * max(voxel_size_mm):
            pad_x = int(
                (torch.sqrt(torch.tensor(2.0)) * scanner_radius_mm - img.shape[-2] * voxel_size_mm[0]) / (2 * voxel_size_mm[0])
            )
            pad_y = int(
                (torch.sqrt(torch.tensor(2.0)) * scanner_radius_mm - img.shape[-1] * voxel_size_mm[1]) / (2 * voxel_size_mm[1])
            )
            img = torch.nn.functional.pad(
                img, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0
            )
        else:
            raise ValueError("Image size exceeds scanner field of view.")
        return img
    
    # Pad image to fit in scanner radius
    image = pad_(image)

    # Image domain PSF
    if gaussian_PSF_fwhm_mm is not None:
        sigma_mm = gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
        image = apply_gaussian_psf_reflect(image, sigma_mm)

    # Radon transform
    theta = torch.linspace(0., 180., n_angles, dtype=image.dtype, device=image.device)
    radon = DeepInvRadon(
        in_size=image.shape[-1],
        theta=theta,
        circle=True,
    )
    sinogram = radon.forward(image)

    # Apply attenuation if provided
    if attenuation_map is not None:
        assert voxel_size_mm[0] == voxel_size_mm[1], "Voxel size must be isotropic for attenuation application."
        # We pass the attenuation map through the same padding as the image to ensure they are aligned
        attenuation_map = pad_(attenuation_map)
        # The attenuation map is typically in units of cm^-1, so we need to convert it to mm^-1 for homogeneous units
        attenuation_scale_factor = 0.01 # cm^-1 to mm^-1
        if gaussian_PSF_fwhm_mm is not None:
            # In clinical context, the attenuation map is obtained from a CT scan and smoothed with the same PSF as the image
            # to imitate the PSF of the PET system.
            sigma_mm = gaussian_PSF_fwhm_mm / (2.0 * (torch.log(torch.tensor(2.0)))**0.5)
            attenuation_map = apply_gaussian_psf_reflect(attenuation_map, sigma_mm)
        # The attenuation along each ray can be computed by applying the radon transform to the attenuation map.
        att_sino = radon.forward(attenuation_map * attenuation_scale_factor)
        # The expected counts along each ray are then scaled by the exponential of the negative attenuation, following the Beer-Lambert law.
        sinogram = sinogram * torch.exp(-att_sino * voxel_size_mm[0]) # assuming square pixels

    # Transpose sinogram to have shape (angles, bins)
    sinogram = sinogram.transpose(-2, -1)

    # PSF in sinogram domain
    if gaussian_PSF_fwhm_mm is not None: 
        sigma_mm = gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
        # 1D sigma in sinogram domain on each row
        bin_widths_mm = torch.abs(voxel_size_mm[0] * torch.cos(theta * torch.pi / 180.0)) + torch.abs(voxel_size_mm[1] * torch.sin(theta * torch.pi / 180.0))
        sigma_sino_mm = sigma_mm / bin_widths_mm
        sinogram = apply_rowwise_gaussian_psf(sinogram, sigmas=sigma_sino_mm)

    # Scale sinogram to match expected counts
    if scale is not None:
        sinogram = sinogram * scale

    return sinogram


if __name__ == "__main__":

    cuda_available = torch.cuda.is_available()
    assert cuda_available, "CUDA is required to run this example."
    device = torch.device('cuda')

    from tools.image.castor import read_castor_binary_file
    import os

    from pytorcher.utils.fbp import deepinv_iradon

    # Example usage
    dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"
    image = read_castor_binary_file(os.path.join(dest_path, 'object', 'gt_web_after_scaling.hdr'), reader='numpy')
    image = torch.from_numpy(image).unsqueeze(0).float().to(device) # shape (1, 1, H, W)
    attenuation_map = read_castor_binary_file(os.path.join(dest_path, 'object', 'attenuat_brain_phantom.hdr'), reader='numpy')
    attenuation_map = torch.from_numpy(attenuation_map).unsqueeze(0).float().to(device)
    sinogram = pet_forward(
        image=image,
        attenuation_map=attenuation_map,
        n_angles=300,
        scanner_radius_mm=300,
        gaussian_PSF_fwhm_mm=4.0,
        voxel_size_mm=2.0,
        scale=2e-2 # approximately 1e6 counts
    )

    backprojected_image = deepinv_iradon(sinogram.transpose(-2, -1) / 2e-2, filter=True, circle=True, out_size=image.shape[-1])
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Backprojected image shape: {backprojected_image.shape}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image.cpu().squeeze(), cmap='gray_r')
    ax[0].set_title('Original Image')

    ax[1].imshow(sinogram.cpu().squeeze(), cmap='gray')
    ax[1].set_title('Sinogram')

    ax[2].imshow(backprojected_image.cpu().squeeze(), cmap='gray_r')
    ax[2].set_title('Backprojected Image')

    plt.show()