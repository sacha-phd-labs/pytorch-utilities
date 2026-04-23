import numpy
import torch, math

from pytorcher.utils.filters import apply_gaussian_psf_reflect, apply_columnwise_gaussian_psf

from pet_simulator import ParallelViewProjector2D

class PetProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, projector, scale=None):

        ctx.projector = projector
        ctx.scale = scale

        img_detached = image.detach()

        if img_detached.ndim == 4 and isinstance(projector, ParallelViewProjector2D):
            img_detached = img_detached.squeeze(1) # remove channel dimension if exists
            out = projector.project(img_detached, scale=scale)
            out = out.unsqueeze(1) # add channel dimension back
        else:
            out = projector.project(img_detached, scale=scale)

        return out
    
    @staticmethod
    def backward(ctx, grad_output):

        projector = ctx.projector
        scale = ctx.scale

        grad_output_detached = grad_output.detach()

        if grad_output_detached.ndim == 4 and isinstance(projector, ParallelViewProjector2D):
            grad_output_detached = grad_output_detached.squeeze(1) # remove channel dimension if exists
            out = projector.adjoint(grad_output_detached, scale=scale)
            out = out.unsqueeze(1) # add channel dimension back
        else:
            raise ValueError(f"Unknown combination of projector type and grad_output dimensions: {type(projector)} with grad_output.ndim={grad_output_detached.ndim}")

        return out, None, None

class PetSystem(torch.nn.Module):

    def __init__(
            self,
            projector_type='parallelproj_parallel',
            projector_config={},
            gaussian_PSF=4.0,
            device='cuda'
        ):
        super().__init__()
        # update config to enforce use of torch
        projector_config['array_compat'] = 'torch'

        self.device = device
        
        self.projector_config = projector_config
        self.projector_type = projector_type
        self.get_projection_operator()

        self.voxel_size_mm = projector_config.get('voxel_size_mm', None)

        self.gaussian_PSF_fwhm_mm = gaussian_PSF


    def get_projection_operator(self):

        if self.projector_type == 'parallelproj_parallel':
            self.proj = ParallelViewProjector2D(**self.projector_config)
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")
        
        self.proj.dev = self.device

    def project(self, image, scale=None):

        return PetProjection.apply(image, self.proj, scale)
    
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

    @staticmethod
    def filter_sinogram(sinogram, filter_name="ramp"):

        def get_fourier_filter( size, filter_name, device, dtype=torch.float32):
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
    
    def forward_adjoint(self, sinogram, attenuation_map=None, scale=None):
        with torch.enable_grad():
            x_0 = torch.zeros((sinogram.shape[0], sinogram.shape[1], self.proj.img_shape[0], self.proj.img_shape[1]), device=sinogram.device, requires_grad=True)
            Ax_0 = self.forward(x_0, attenuation_map=attenuation_map, scale=scale)
            prod_Ax_y = (Ax_0 * sinogram).sum()
            grad_Ax = torch.autograd.grad(prod_Ax_y, x_0)[0]
        return grad_Ax
    
    def fbp(self, sinogram, filter_name='ramp', attenuation_map=None, scale=None):

        # filter sinogram
        sinogram = self.filter_sinogram(sinogram, filter_name=filter_name)

        # backproject
        sensitivity = self.forward_adjoint(torch.ones_like(sinogram), attenuation_map=attenuation_map, scale=None)
        reconstructed_image = self.forward_adjoint(sinogram, attenuation_map=attenuation_map, scale=scale) / sensitivity

        return reconstructed_image

    def mlem(self, sinogram, num_it=10, attenuation_map=None, scale=None, corr=None, eps=1e-8):

        if corr is None:
            corr = torch.zeros_like(sinogram)

        # Compute snesitivity
        sensitivity = self.forward_adjoint(torch.ones_like(sinogram))

        x_0 = torch.ones((sinogram.shape[0], sinogram.shape[1], self.proj.img_shape[0], self.proj.img_shape[1]), device=sinogram.device)

        for _ in range(num_it):

            # E-step
            y_hat = self.forward(x_0, attenuation_map=attenuation_map, scale=scale) + corr

            # M-step
            x_0 = x_0 * self.forward_adjoint(sinogram / (y_hat + eps)) / sensitivity

        return x_0

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

    prompt, meta = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_pt.s.hdr'), return_metadata=True)
    scale = float(meta['scale_factor'])
    random = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_rd.s.hdr'))
    scatter = read_castor_binary_file(os.path.join(simu_dest_path, 'simu', 'simu_sc.s.hdr'))
    corr = random + scatter

    prompt = torch.from_numpy(prompt).unsqueeze(0).float().to(device)
    corr = torch.from_numpy(corr).unsqueeze(0).float().to(device)

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

    bp = system.fbp(sinogram, filter_name='ramp', attenuation_map=attenuation_map, scale=scale)

    recon_mlem = system.mlem(prompt, attenuation_map=attenuation_map, scale=scale, corr=corr, num_it=10)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(prompt.cpu().squeeze(), cmap='gray')
    ax[0].set_title('Prompt Sinogram')
    ax[1].imshow(recon_mlem.cpu().squeeze(), cmap='gray_r')
    ax[1].set_title('MLEM Image')
    plt.show()
