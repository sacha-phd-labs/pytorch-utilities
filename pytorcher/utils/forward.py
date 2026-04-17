import math

import torch
import torch.nn.functional as F
import numpy as np
try:
    from torch_radon import Radon
except ImportError:
    print("torch_radon is not installed. Please install it to use PetForwardRadon.")

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_columnwise_gaussian_psf

def get_fourier_filter(size, filter_name, device, dtype=torch.float32):
    freqs = ( 2 * torch.fft.fftfreq(size, device=device).abs()).to(dtype)

    if filter_name == "ramp":
        filt = freqs
    elif filter_name == "shepp-logan":
        filt = freqs * torch.sinc(freqs / 2)
    elif filter_name == "cosine":
        filt = freqs * torch.cos(math.pi * freqs / 2)
    elif filter_name == "hamming":
        filt = freqs * (0.54 + 0.46 * torch.cos(math.pi * freqs))
    elif filter_name == "hann":
        filt = freqs * (1 + torch.cos(math.pi * freqs)) / 2
    elif filter_name is None:
        filt = torch.ones_like(freqs)
    else:
        raise ValueError("Unknown filter")

    return filt[:, None]  # column-wise

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
        #

    def get_radon_operator(self, in_size, dtype, device):
        self.theta = torch.linspace(0., np.pi, self.n_angles, dtype=dtype, device=device)
        self.in_size = in_size
        self.radon = Radon(
            resolution=in_size,
            angles=self.theta,
            clip_to_circle=True,
            det_count=in_size,
            det_spacing=1.0
        )

    def radon_forward(self, image):
        if not hasattr(self, 'radon') or self.in_size != image.shape[-1]:
            self.get_radon_operator(image.shape[-1], image.dtype, image.device)
        out = self.radon.forward(image)
        return out.transpose(-2, -1)

    def _broadcast_scale(self, tensor, scale):
        if scale is None:
            return tensor

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=tensor.dtype)
        else:
            scale = scale.to(device=tensor.device, dtype=tensor.dtype)

        leading_shape = tensor.shape[:-2]
        if scale.ndim == 0:
            scale = scale.expand(*leading_shape) if len(leading_shape) > 0 else scale
        elif tuple(scale.shape) != tuple(leading_shape):
            scale = scale.reshape(*leading_shape)

        view_shape = (*leading_shape, 1, 1)
        return tensor * scale.view(*view_shape)
    
    def filter_sinogram(self, sinogram, filter_name="ramp"):

        if sinogram.ndim < 2:
            raise ValueError("sinogram must be (..., detectors, angles)")

        leading_shape = sinogram.shape[:-2]
        num_detectors = sinogram.shape[-2]
        num_angles = sinogram.shape[-1]
        device = sinogram.device
        dtype = sinogram.dtype

        # Flatten leading dims to apply one FFT per sinogram
        sinogram = sinogram.reshape(-1, num_detectors, num_angles)

        # Padding
        padded_size = max(64, 2 ** math.ceil(math.log2(2 * num_detectors)))
        pad = padded_size - num_detectors
        img = torch.nn.functional.pad(sinogram, (0, 0, 0, pad))  # (Nflat, padded, A)

        # Fourier filtering
        fourier_filter = get_fourier_filter(padded_size, filter_name, device, dtype=dtype)  # (padded, 1)
        proj_fft = torch.fft.fft(img, dim=1) * fourier_filter.unsqueeze(0)  # broadcast over Nflat
        radon_filtered = torch.real(torch.fft.ifft(proj_fft, dim=1))[:, :num_detectors]

        return radon_filtered.reshape(*leading_shape, num_detectors, num_angles)


    def radon_backward(self, sinogram, output_size=None, scale=None, filter_name='ramp'):

        num_angles = sinogram.shape[-1]
        sino_resolution = sinogram.shape[-2]
        if not hasattr(self, 'radon') or self.in_size != sino_resolution:
            self.get_radon_operator(sino_resolution, sinogram.dtype, sinogram.device)

        if scale is not None:
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale, device=sinogram.device, dtype=sinogram.dtype)
            else:
                scale = scale.to(device=sinogram.device, dtype=sinogram.dtype)

            leading_shape = sinogram.shape[:-2]
            if scale.ndim == 0:
                scale = scale.expand(*leading_shape) if len(leading_shape) > 0 else scale
            elif tuple(scale.shape) != tuple(leading_shape):
                scale = scale.reshape(*leading_shape)

            sinogram = sinogram / scale.view(*leading_shape, 1, 1)

        # Apply filter in Fourier domain
        if filter_name is not None:
            radon_filtered = self.filter_sinogram(sinogram, filter_name=filter_name)
        else:
            radon_filtered = sinogram
        out = self.radon.backward(radon_filtered.transpose(-2, -1))

        # Normalize by angular sampling step.
        # - Filtered FBP follows torch_radon's convention: pi / (2 * N_angles)
        if filter_name is not None:
            out = out * (math.pi / (2 * num_angles))

        # Crop to original image size if padding was applied in forward projection
        if output_size is not None and output_size < out.shape[-1]:
            crop_start = (out.shape[-1] - output_size) // 2
            out = out[..., crop_start:crop_start + output_size, crop_start:crop_start + output_size]

        return out

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
            self.get_radon_operator(image.shape[-1], image.dtype, image.device)
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
            bin_widths_mm = self.voxel_size_mm[0] #torch.abs(self.voxel_size_mm[0] * torch.cos(self.theta * torch.pi / 180.0)) + torch.abs(self.voxel_size_mm[1] * torch.sin(self.theta * torch.pi / 180.0))
            sigma_sino_mm = sigma_mm / bin_widths_mm
            sinogram = apply_columnwise_gaussian_psf(sinogram, sigma=sigma_sino_mm)

        # # Scale sinogram to match expected counts
        if scale is not None:
            sinogram = self._broadcast_scale(sinogram, scale)
    
        return sinogram


if __name__ == "__main__":

    device = torch.device('cuda')

    from pytorcher.utils.fbp import iradon
    from tools.image.castor import read_castor_binary_file
    import os

    path=os.path.join(os.getenv('WORKSPACE'), 'data/brain_web_phantom')
    image_path = os.path.join(path, 'object', 'gt_web_after_scaling.hdr')
    att_path = os.path.join(path, 'object', 'attenuat_brain_phantom.hdr')

    image = read_castor_binary_file(image_path)
    attenuation_map = read_castor_binary_file(att_path)

    image = torch.from_numpy(image).unsqueeze(0).float().to(device)
    attenuation_map = torch.from_numpy(attenuation_map).unsqueeze(0).float().to(device)


    path=os.path.join(os.getenv('WORKSPACE'), 'data/brain_web_phantom')
    simu_dest_path = os.path.join(path, 'simu')

    nf_prompt = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_nfpt.s.hdr')).squeeze()
    _, meta = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_pt.s.hdr'), return_metadata=True)
    scale = float(meta['scale_factor'])
    true_sino = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_t.s.hdr')).squeeze()
    true_sino = torch.from_numpy(true_sino).float().to(device).unsqueeze(0).unsqueeze(0)

    scanner_radius_mm = 300
    voxel_size_mm = 2.0

    pet_forward = PetForwardRadon(
        n_angles=300,
        scanner_radius_mm=scanner_radius_mm,
        gaussian_PSF_fwhm_mm=4.0,
        voxel_size_mm=voxel_size_mm
    )
    #
    sinogram = pet_forward.forward(image, attenuation_map=attenuation_map, scale=scale)
    #
    back_projection = pet_forward.radon_backward(true_sino, filter_name='ramp', scale=scale, output_size=image.shape[-1])

    #
    print("distance between forward projection and true sinogram:", torch.norm(sinogram.cpu() - true_sino.cpu()) )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image[0, 0, ...].cpu().squeeze(), cmap='gray_r')
    ax[0].set_title('Original Image')

    ax[1].imshow(np.abs(sinogram[0, 0, ...].cpu().squeeze() - true_sino.cpu().squeeze()), cmap='gray')
    ax[1].set_title('Difference between Forward Projection and True Sinogram')

    ax[2].imshow(back_projection[0, 0, ...].cpu().squeeze(), cmap='gray_r')
    ax[2].set_title('Back Projection')

    plt.show()