import torch, math

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
    sinogram,          # (..., detectors, angles)
    output_size=None,
    theta= None,
    filter="ramp",
    circle=True,
):
    if sinogram.ndim < 2:
        raise ValueError("sinogram must be (..., detectors, angles)")

    squeeze_batch = False
    if sinogram.ndim == 2:
        sinogram = sinogram.unsqueeze(0)  # (1, D, A)
        squeeze_batch = True

    device = sinogram.device
    dtype = sinogram.dtype
    leading_shape = sinogram.shape[:-2]
    num_detectors, num_angles = sinogram.shape[-2:]
    sinogram = sinogram.reshape(-1, num_detectors, num_angles)
    N = sinogram.shape[0]

    if theta is None:
        theta = torch.linspace(0, torch.pi, num_angles, device=device, dtype=dtype)
    else:
        theta = torch.as_tensor(theta, device=device, dtype=dtype)
        if theta.numel() != num_angles:
            raise ValueError("theta length must match sinogram angles dimension")
        # Accept both degree and radian conventions.
        if torch.max(torch.abs(theta)) > 2 * torch.pi + 1e-6:
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

    # Backprojection (parallel over all angles and all batches)
    cos_t = torch.cos(theta).view(num_angles, 1, 1)
    sin_t = torch.sin(theta).view(num_angles, 1, 1)
    t = ypr.unsqueeze(0) * cos_t - xpr.unsqueeze(0) * sin_t  # (A, H, W)

    t_flat = t.reshape(num_angles, -1)  # (A, H*W)
    idx = torch.searchsorted(detector_x, t_flat)
    idx0 = torch.clamp(idx - 1, 0, num_detectors - 1)
    idx1 = torch.clamp(idx, 0, num_detectors - 1)

    x0 = detector_x[idx0]
    x1 = detector_x[idx1]
    denom = (x1 - x0)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    w = (t_flat - x0) / denom  # (A, H*W)

    # (N, D, A) -> (N, A, D)
    rf = radon_filtered.permute(0, 2, 1)
    idx0_exp = idx0.unsqueeze(0).expand(N, -1, -1)
    idx1_exp = idx1.unsqueeze(0).expand(N, -1, -1)

    y0 = torch.gather(rf, 2, idx0_exp)
    y1 = torch.gather(rf, 2, idx1_exp)

    interp = y0 + (y1 - y0) * w.unsqueeze(0)  # (N, A, H*W)
    recon = interp.sum(dim=1).view(N, output_size, output_size)

    # Circle mask
    if circle:
        mask = (xpr**2 + ypr**2) > radius**2
        recon[:, mask] = 0.0

    recon = recon * math.pi / num_angles
    recon = recon.view(*leading_shape, output_size, output_size)

    if squeeze_batch:
        return recon.squeeze(0)

    # Crop back to original size 
    return recon