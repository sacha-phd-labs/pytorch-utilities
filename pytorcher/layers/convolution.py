
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.nn import Conv2d

class SeparableConv2d(nn.Module):
    """
    Source - https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
    Posted by Poe Dator
    Retrieved 11/5/2025, License - CC-BY-SA 4.0
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = Conv2d(in_channels, out_channels, 
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

        kwargs.update({'padding': 0})  # No padding here, will be applied manually in forward
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, **kwargs) # No padding here, will be applied manually in forward

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

        kwargs.pop('padding', None)  # No padding here, will be applied manually in forward
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=0, **kwargs)
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
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, layer_type='Conv2d', residual=False, **kwargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        assert layer_type in globals().keys(), f"Layer type '{layer_type}' is not recognized."
        conv_layer = globals()[layer_type]
        
        self.residual = residual
        self.double_conv = nn.Sequential(
            conv_layer(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_layer(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.act = nn.ReLU(inplace=True)
        if residual and in_channels != out_channels:
            # If residual and dimensions do not match, add a 1x1 convolution to add a learnable projection and match dimensions
            conv_layer_type_no_separable = layer_type.replace('Separable', '')
            conv_layer = globals()[conv_layer_type_no_separable]
            self.residual_conv = conv_layer(in_channels, out_channels, kernel_size=1, bias=False, **kwargs)

    def forward(self, x):
        if self.residual:
            residual = x
            if hasattr(self, 'residual_conv'):
                residual = self.residual_conv(x)
            x = self.double_conv(x) + residual
        else:
            x = self.double_conv(x)
        x = self.act(x)
        return x

    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, layer_type='Conv2d', residual=False, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, layer_type=layer_type, residual=residual, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class ResizeConv(nn.Module):
    """Interpolation followed by convolution for image-to-image translation."""
    
    def __init__(self, in_channels, scale_factor=2, mode='bilinear', layer_type='Conv2d', residual=False, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.convs = nn.Sequential(
            DoubleConv(in_channels, in_channels * 2, layer_type=layer_type, residual=False, **kwargs),
            DoubleConv(in_channels * 2, in_channels, layer_type=layer_type, residual=residual, **kwargs),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        x = self.convs(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, layer_type='Conv2d', residual=False, **kwargs):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, layer_type=layer_type, residual=residual)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, layer_type=layer_type, residual=residual)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer_type='Conv2d', **kwargs):
        super(OutConv, self).__init__()
        conv_layer = globals()[conv_layer_type]
        self.conv = conv_layer(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = nn.Sigmoid()(x)
        return x