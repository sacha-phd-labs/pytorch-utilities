import torch.nn as nn
import torch.nn.functional as F 

class SeparableConv2d(nn.Module):
    """
    Source - https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
    Posted by Poe Dator
    Retrieved 11/5/2025, License - CC-BY-SA 4.0
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def mix_pad_sinogram(x, pad_width, radial_dim=1, radial_padding_mode="constant"):
    # Circular padding along angular dimension (assumed to be dimension 2 or 3)
    angular_dim = 2 if radial_dim == 1 else 3
    pad_angular = pad_width[1]
    if angular_dim == 2:
        padding = (0, 0, pad_angular, pad_angular)
    else:
        padding = (pad_angular, pad_angular, 0, 0)
    x = F.pad(x, padding, mode='circular')
    # Radial padding along radial dimension
    radial_dim = 2 if radial_dim == 0 else 3
    pad_radial = pad_width[0]
    if radial_dim == 2:
        padding = (0, 0, pad_radial, pad_radial)
    else:
        padding = (pad_radial, pad_radial, 0, 0)
    x = F.pad(x, padding, mode=radial_padding_mode)
    return x

class SinogramConv2d(nn.Module):
    """
    2D convolution layer for sinograms.
    Apply circular padding along the angular dimension and regular padding along the radial dimension
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, radial_padding_mode="constant", radial_dim=1, **kwargs):
        super(SinogramConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.radial_padding_mode = radial_padding_mode
        self.radial_dim = radial_dim

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=0, **kwargs) # No padding here, will be applied manually in forward

    def forward(self, x):
        # Mixed padding
        pad_width = (self.conv.kernel_size[0] // 2, self.conv.kernel_size[1] // 2)
        x = mix_pad_sinogram(x, pad_width, radial_dim=self.radial_dim, radial_padding_mode=self.radial_padding_mode)
        # Convolution
        out = self.conv(x)
        return out
    
class SinogramSeparableConv2d(nn.Module):
    """
    Separable convolution layer for sinograms.
    Apply circular padding along the angular dimension and regular padding along the radial dimension
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, radial_padding_mode="constant", radial_dim=1, **kwargs):
        super(SinogramSeparableConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.radial_padding_mode = radial_padding_mode
        self.radial_dim = radial_dim

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=0, **kwargs) # No padding here, will be applied manually in forward
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        # Mixed padding
        pad_width = (self.depthwise.kernel_size[0] // 2, self.depthwise.kernel_size[1] // 2)
        x = mix_pad_sinogram(x, pad_width, radial_dim=self.radial_dim, radial_padding_mode=self.radial_padding_mode)
        # Depthwise convolution
        out = self.depthwise(x)
        # Pointwise convolution
        out = self.pointwise(out)
        return out