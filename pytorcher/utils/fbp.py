import torch, math

from deepinv.physics.functional import IRadon as DeepInvIRadon

def get_fourier_filter(size, filter_name, device):
    freqs = torch.fft.fftfreq(size, device=device).abs()

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

def interp1d_linear(t, xp, fp):
    """
    t  : (H, W)
    xp : (D,)
    fp : (N, D)
    returns: (N, H, W)
    """
    N, D = fp.shape
    H, W = t.shape

    t_flat = t.reshape(-1)  # (H*W)

    # Find indices
    idx = torch.searchsorted(xp, t_flat)
    idx0 = torch.clamp(idx - 1, 0, D - 1)
    idx1 = torch.clamp(idx, 0, D - 1)

    x0 = xp[idx0]
    x1 = xp[idx1]

    y0 = fp[:, idx0]  # (N, H*W)
    y1 = fp[:, idx1]

    denom = (x1 - x0)
    denom[denom == 0] = 1  # avoid divide by zero

    w = (t_flat - x0) / denom  # (H*W)

    out = y0 + (y1 - y0) * w  # (N, H*W)

    return out.view(N, H, W)

def iradon(
    sinogram,          # (N, detectors, angles)
    output_size=None,
    theta= None,
    filter="ramp",
    circle=True,
):
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(0)  # (1, D, A)

    if sinogram.ndim != 3:
        raise ValueError("sinogram must be (N, detectors, angles)")

    device = sinogram.device
    dtype = sinogram.dtype
    N, num_detectors, num_angles = sinogram.shape

    if theta is None:
        theta = torch.linspace(0, 180, num_angles, device=device)

    theta = torch.deg2rad(theta)

    if output_size is None:
        output_size = num_detectors if circle else int(
            math.floor(math.sqrt(num_detectors**2 / 2))
        )

    # Padding
    padded_size = max(64, 2 ** math.ceil(math.log2(2 * num_detectors)))
    pad = padded_size - num_detectors
    img = torch.nn.functional.pad(sinogram, (0, 0, 0, pad))  # (N, padded, A)

    # Fourier filtering
    fourier_filter = get_fourier_filter(padded_size, filter, device)  # (padded, 1)
    proj_fft = torch.fft.fft(img, dim=1) * fourier_filter  # broadcast over N
    radon_filtered = torch.real(torch.fft.ifft(proj_fft, dim=1))[:, :num_detectors]

    # Reconstruction grid
    recon = torch.zeros((N, output_size, output_size), device=device, dtype=dtype)
    radius = output_size // 2

    xpr, ypr = torch.meshgrid(
        torch.arange(output_size, device=device),
        torch.arange(output_size, device=device),
        indexing="ij"
    )
    xpr = xpr - radius
    ypr = ypr - radius

    detector_x = torch.arange(num_detectors, device=device) - num_detectors // 2

    # Backprojection 
    for i in range(num_angles):
        angle = theta[i]
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)  # (H, W)

        # (N, D)
        col = radon_filtered[:, :, i]

        # interp must support batch now
        recon += interp1d_linear(t, detector_x, col)

    # Circle mask
    if circle:
        mask = (xpr**2 + ypr**2) > radius**2
        recon[:, mask] = 0.0

    return recon * math.pi / num_angles

def deepinv_iradon(
        sinogram, # (N, detectors, angles)
        in_size=None,
        out_size=None,
        theta=None,
        circle=True,
        filter=True
    ):
    """
    Wrapper for deepinv.physics.functional.IRadon
    """
    assert filter == True, "Using unfiltered backprojection is not recommended for PET reconstruction. Please set filter=True."
    #
    if in_size is None:
        in_size = sinogram.shape[-1]
    if out_size is None:
        out_size = sinogram.shape[-2]
    # Pad sinogram if not square
    if sinogram.shape[-1] != sinogram.shape[-2]:
        max_side = max(sinogram.shape[-1], sinogram.shape[-2])
        pad_y = (max_side - sinogram.shape[-2]) // 2
        pad_x = (max_side - sinogram.shape[-1]) // 2
        sinogram = torch.nn.functional.pad(
            sinogram, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0
        )
    if theta is None:
        theta = torch.linspace(0., 180., sinogram.shape[-2], dtype=sinogram.dtype, device=sinogram.device)
    iradon = DeepInvIRadon(
        in_size=in_size,
        out_size=out_size,
        theta=theta,
        circle=circle,
        device=sinogram.device,
        use_filter=filter
    )
    image = iradon.forward(sinogram)
    # Crop back to original size if padded
    if image.shape[-1] != out_size:
        crop_x = (image.shape[-1] - out_size) // 2
        crop_y = (image.shape[-2] - out_size) // 2
        image = image[:, :, crop_y:crop_y+out_size, crop_x:crop_x+out_size]
    return image

if __name__ == "__main__":

    from tools.image.castor import read_castor_binary_file
    # import matplotlib.pyplot as plt

    # sino = read_castor_binary_file("/workspace/data/brain_web_phantom/simu/simu_pt.s.hdr").squeeze().transpose()
    # sino = torch.from_numpy(sino)  # [1, A, D]
    # angles = torch.linspace(0, math.pi, 300, device=sino.device)

    # recon = iradon_torch(sino, angles, out_size=160, circle=False)
    # plt.imshow(recon.cpu(), cmap="gray")
    # plt.show()


    from skimage.transform import radon, iradon as skimage_iradon
    import numpy as np

    import matplotlib.pyplot as plt

    img = read_castor_binary_file("/workspace/data/brain_web_phantom/object/gt_web_after_scaling.hdr").squeeze()

    theta = np.linspace(0, 180, 180, endpoint=False)
    sino = radon(img.copy(), theta=theta, circle=True)

    recon_np = skimage_iradon(sino, theta=theta, circle=True)
    #
    sino_tensor = torch.from_numpy(sino).float().unsqueeze(0) # shape (1, D, A)
    recon_torch = iradon(sino_tensor).cpu().numpy()

    sino_tensor = sino_tensor.unsqueeze(1) # shape (1, 1, D, A)
    recon_torch_deepinv = deepinv_iradon(sino_tensor).cpu().numpy().squeeze()

    print(np.mean(np.abs(recon_np - recon_torch.squeeze())))

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(recon_np, cmap='gray')
    ax[1].set_title('Reconstructed Image (skimage)')
    ax[2].imshow(recon_torch.squeeze(), cmap='gray')
    ax[2].set_title('Reconstructed Image (PyTorch)')
    ax[3].imshow(recon_torch_deepinv, cmap='gray')
    ax[3].set_title('Reconstructed Image (DeepInv)')
    plt.show()
