import torch, math

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_columnwise_gaussian_psf

from pet_simulator import ParallelViewProjector2D

class PetSystem(torch.nn.Module):

    def __init__(
            self,
            projector_type='parallelproj_parallel',
            projector_config={},
            scatter_component=0.36,
            scatter_sigma=4.0, # in pixels
            random_component=0.50,
            gaussian_PSF=4.0,
            device='cuda'
        ):
        super().__init__()
        # update config to enforce use of torch
        projector_config['array_compat'] = 'torch'

        self.device = device
        self.to(device)
        
        self.projector_config = projector_config
        self.projector_type = projector_type
        self.get_projection_operator()

        self.voxel_size_mm = projector_config.get('voxel_size_mm', None)

        self.scatter_component = scatter_component
        self.scatter_sigma = scatter_sigma
        self.random_component = random_component
        self.gaussian_PSF_fwhm_mm = gaussian_PSF


    def get_projection_operator(self):

        if self.projector_type == 'parallelproj_parallel':
            self.proj = ParallelViewProjector2D(**self.projector_config)
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")
        
        self.proj.dev = self.device

    def project(self, img, scale=None):

        if img.ndim == 4 and self.projector_type == 'parallelproj_parallel':
            img = img.squeeze(1) # remove channel dimension if exists
            out = self.proj.project(img, scale=scale)
            out = out.unsqueeze(1) # add channel dimension back
        else:
            out = self.proj.project(img, scale=scale)

        return out
    
    def forward(self, image, attenuation_map=None, scale=None):

        #
        if attenuation_map is not None:
            assert image.shape == attenuation_map.shape, "Image and attenuation map must have the same shape."

        # Image domain PSF
        if self.gaussian_PSF_fwhm_mm is not None:
            sigma_mm = self.gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
            image = apply_gaussian_psf_reflect(image, sigma_mm / self.voxel_size_mm[0]) # assuming square pixels

        # Projection
        sinogram = self.project(image, scale=scale)

        # Apply attenuation if provided
        if attenuation_map is not None:

            assert self.voxel_size_mm[0] == self.voxel_size_mm[1], "Voxel size must be isotropic for attenuation application."

            # The attenuation map is typically in units of cm^-1, so we need to convert it to mm^-1 for homogeneous units
            attenuation_scale_factor = 0.01 # cm^-1 to mm^-1

            if self.gaussian_PSF_fwhm_mm is not None:
                # In clinical context, the attenuation map is obtained from a CT scan and smoothed with the same PSF as the image
                # to imitate the PSF of the PET system.
                sigma_mm = self.gaussian_PSF_fwhm_mm / (2.0 * (torch.log(torch.tensor(2.0)))**0.5)
                attenuation_map = apply_gaussian_psf_reflect(attenuation_map, sigma_mm / self.voxel_size_mm[0]) # assuming square pixels

            # The attenuation along each ray can be computed by applying the radon transform to the attenuation map.
            att_sino = self.project(attenuation_map * attenuation_scale_factor)

            # The expected counts along each ray are then scaled by the exponential of the negative attenuation, following the Beer-Lambert law.
            sinogram = sinogram * torch.exp(-att_sino * self.voxel_size_mm[0]) # assuming square pixels

        # PSF in sinogram domain
        if self.gaussian_PSF_fwhm_mm is not None: 
            sigma_mm = self.gaussian_PSF_fwhm_mm / (4.0 * (torch.log(torch.tensor(2.0)))**0.5)
            # 1D sigma in sinogram domain on each row
            bin_widths_mm = self.voxel_size_mm[0]
            sigma_sino_mm = sigma_mm / bin_widths_mm
            sinogram = apply_columnwise_gaussian_psf(sinogram, sigma=sigma_sino_mm)

        return sinogram
    
    def get_fourier_filter(self, size, filter_name, device, dtype=torch.float32):
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
        fourier_filter = self.get_fourier_filter(padded_size, filter_name, device, dtype=dtype)  # (padded, 1)
        proj_fft = torch.fft.fft(img, dim=1) * fourier_filter.unsqueeze(0)  # broadcast over Nflat
        radon_filtered = torch.real(torch.fft.ifft(proj_fft, dim=1))[:, :num_detectors]

        return radon_filtered.reshape(*leading_shape, num_detectors, num_angles)

    def adjoint(self, sino, filter_name="ramp", scale=None):

        sino = self.filter_sinogram(sino, filter_name=filter_name)

        if sino.ndim == 4 and self.projector_type == 'parallelproj_parallel':
            sino = sino.squeeze(1) # remove channel dimension if exists
            out = self.proj.adjoint(sino, scale=scale) / self.proj.adjoint(torch.ones_like(sino), scale=None) # apply sensitivity correction
            out = out.unsqueeze(1) # add channel dimension back
        else:
            raise ValueError(f"Unknown combination of projector type and sinogram dimensions: {self.projector_type} with sino.ndim={sino.ndim}")

        return out

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    device = torch.device('cuda')

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

    _, meta = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_pt.s.hdr'), return_metadata=True)
    scale = float(meta['scale_factor'])

    system = PetSystem(
        projector_type='parallelproj_parallel',
        projector_config={
            'num_angles': 300,
            'scanner_radius_mm': 300,
            'img_shape': image.shape[-2:],
            'voxel_size_mm': (2.0, 2.0)
        },
        gaussian_PSF=4.0,
        device=device
    )

    sinogram = system(image, attenuation_map=attenuation_map, scale=scale)

    sinogram = torch.poisson(sinogram) # add Poisson noise to simulate realistic PET data

    bp = system.adjoint(sinogram, scale=scale, filter_name='ramp')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sinogram.cpu().squeeze(), cmap='gray')
    ax[0].set_title('Simulated Sinogram')
    ax[1].imshow(bp.cpu().squeeze(), cmap='gray_r')
    ax[1].set_title('Backprojected Image')
    plt.show()
