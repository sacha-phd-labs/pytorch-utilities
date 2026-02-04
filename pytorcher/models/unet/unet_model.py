""" Full assembly of the parts to form the complete network """

from pytorcher.layers import *
from pytorcher.utils.processing import normalize_batch, rescale_batch

class UNet(nn.Module):
    def __init__(
                 self,
                 n_channels,
                 n_classes,
                 global_conv=32,
                 n_levels=3,
                 bilinear=False,
                 conv_layer_type='Conv2d',
                 residual=False,
                 normalize_input=False,
                 output_size='same'
        ):
        """
        :param n_channels: Number of input channels
        :param n_classes: Number of output channels
        :param bilinear: Whether to use bilinear upsampling or transposed convolutions
        :param conv_layer_type: Type of convolutional layer to use ('Conv2d', 'SeparableConv2d', 'SinogramConv2d', 'SinogramSeparableConv2d'). Standard convolutions will always be used for upsampling layers, pre-concvolution layers, and the output layer.
        :param residual: Whether to use residual connections in double convolution blocks.
        :param normalize_input: Whether to normalize the input in the range [0, 1] before feeding it to the network. If set to True, output will be rescaled accordingly to match the input scale.
        :param output_size: 'same' to have the same output size as input, (H, W) to enforce specific output size with resize after the last convolution.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.normalize_input = normalize_input
        assert n_levels in [3, 4], "Only 3 or 4 levels are supported currently."
        self.n_levels = n_levels
        #
        conv_layer_type_no_separable = conv_layer_type.replace('Separable', '') # Ensure that separable convolutions are not used in initial, upsampling, and output layers
        self.inc = (DoubleConv(n_channels, global_conv, layer_type=conv_layer_type_no_separable, residual=residual)) # Initial layer uses standard convolution
        #
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(1,n_levels+1):
            j = n_levels - i + 1
            if i < n_levels:
                self.downs.append(Down(global_conv*(2**(i-1)), global_conv*(2**i), layer_type=conv_layer_type, residual=residual))
                self.ups.append(Up(global_conv*(2**j), (global_conv*(2**(j-1))) // factor, bilinear, layer_type=conv_layer_type, residual=residual))
            else:
                self.downs.append(Down(global_conv*(2**(i-1)), (global_conv*(2**i)) // factor, layer_type=conv_layer_type, residual=residual))
                self.ups.append(Up(global_conv*(2**j), (global_conv*(2**(j-1))), bilinear, layer_type=conv_layer_type, residual=residual))
        #
        self.outc = (OutConv(global_conv, n_classes, conv_layer_type=conv_layer_type_no_separable)) # Output layer uses standard convolution
        #
        self.output_size = output_size

    def forward(self, x):
        if self.normalize_input:
            x, x_mins, x_maxs = normalize_batch(x, return_min_max=True)
        x1 = self.inc(x)
        downs_outputs = []
        x_temp = x1
        for down in self.downs:
            x_temp = down(x_temp)
            downs_outputs.append(x_temp)
        x_temp = downs_outputs[-1]
        for i, up in enumerate(self.ups):
            if i < self.n_levels - 1:
                x_temp = up(x_temp, downs_outputs[-(i+2)])
            else:
                x_temp = up(x_temp, x1)
        x = x_temp
        logits = self.outc(x)
        if hasattr(self, 'output_size') and isinstance(self.output_size, tuple):
            logits = torch.nn.functional.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        if self.normalize_input:
            logits = rescale_batch(logits, x_mins, x_maxs)
        return logits

if __name__ == '__main__':

    from torchsummary import summary

    unet = UNet(n_channels=1, n_classes=1, global_conv=64, n_levels=4, bilinear=True, conv_layer_type='Conv2d', residual=True, normalize_input=True)
    unet.eval()
    # summary(unet, (1, 300, 300), device="cpu")

    test_tensor = torch.randn(1, 1, 300, 300)
    with torch.no_grad():
        output = unet(test_tensor)
    print(f"Output shape: {output.shape}")

    # torch.onnx.export(unet, (torch.zeros(1, 1, 300, 300)), f='unet.onnx', input_names=['input'], output_names=['output'])