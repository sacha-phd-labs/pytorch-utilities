import torch
import math


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


def interp1d_linear(x, xp, fp):
    """
    x: arbitrary shape
    xp: [N]
    fp: [N]
    """
    idx = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]))

    idx0 = (idx - 1).clamp(0, xp.numel() - 1)
    idx1 = idx.clamp(0, xp.numel() - 1)

    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]

    denom = (x1 - x0)
    denom[denom == 0] = 1

    w = (x - x0) / denom
    return y0 + w * (y1 - y0)


def iradon(
    radon_image: torch.Tensor,
    theta: torch.Tensor | None = None,
    output_size: int | None = None,
    filter_name="ramp",
    circle=True,
):
    """
    Faithful PyTorch port of skimage.transform.iradon
    radon_image: [detectors, angles]
    theta: degrees
    """

    if radon_image.ndim != 2:
        raise ValueError("radon_image must be 2D")

    device = radon_image.device
    dtype = radon_image.dtype

    num_detectors, num_angles = radon_image.shape

    if theta is None:
        theta = torch.linspace(
            0, 180, num_angles, endpoint=False, device=device
        )

    theta = torch.deg2rad(theta)

    if output_size is None:
        output_size = num_detectors if circle else int(
            math.floor(math.sqrt(num_detectors**2 / 2))
        )

    # Padding to power of two
    padded_size = max(64, 2 ** math.ceil(math.log2(2 * num_detectors)))
    pad = padded_size - num_detectors
    img = torch.nn.functional.pad(radon_image, (0, 0, 0, pad))

    # Fourier filtering
    fourier_filter = get_fourier_filter(padded_size, filter_name, device)
    proj_fft = torch.fft.fft(img, dim=0) * fourier_filter
    radon_filtered = torch.real(torch.fft.ifft(proj_fft, dim=0))[:num_detectors]

    # Reconstruction grid
    recon = torch.zeros((output_size, output_size), device=device, dtype=dtype)
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
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)

        col = radon_filtered[:, i]
        recon += interp1d_linear(t, detector_x, col)

    if circle:
        mask = (xpr**2 + ypr**2) > radius**2
        recon[mask] = 0.0

    return recon * math.pi / (num_angles)



if __name__ == "__main__":

    from tools.image.castor import read_castor_binary_file
    # import matplotlib.pyplot as plt

    # sino = read_castor_binary_file("/workspace/data/brain_web_phantom/simu/simu_pt.s.hdr").squeeze().transpose()
    # sino = torch.from_numpy(sino)  # [1, A, D]
    # angles = torch.linspace(0, math.pi, 300, device=sino.device)

    # recon = iradon_torch(sino, angles, output_size=160, circle=False)
    # plt.imshow(recon.cpu(), cmap="gray")
    # plt.show()


    from skimage.transform import radon, iradon
    import numpy as np

    import matplotlib.pyplot as plt

    img = read_castor_binary_file("/workspace/data/brain_web_phantom/object/gt_web_after_scaling.hdr").squeeze()

    theta = np.linspace(0, 180, 180, endpoint=False)
    sino = radon(img.copy(), theta=theta, circle=True)

    recon_np = iradon(sino, theta=theta, circle=True)
    recon_torch = iradon(torch.tensor(sino), torch.tensor(theta)).cpu().numpy()

    print(np.mean(np.abs(recon_np - recon_torch)))

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(recon_np, cmap='gray')
    ax[1].set_title('Reconstructed Image (skimage)')
    ax[2].imshow(recon_torch, cmap='gray')
    ax[2].set_title('Reconstructed Image (PyTorch)')
    plt.show()
