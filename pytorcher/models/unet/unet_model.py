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
                 residual=False
        ):
        """
        :param n_channels: Number of input channels
        :param n_classes: Number of output channels
        :param global_conv: Number of convolutional filters in the first level. This number will be doubled at each subsequent level.
        :param n_levels: Number of levels in the U-Net. Supported values are 3 and 4.
        :param bilinear: Whether to use bilinear upsampling or transposed convolutions
        :param conv_layer_type: Type of convolutional layer to use ('Conv2d', 'SeparableConv2d', 'SinogramConv2d', 'SinogramSeparableConv2d'). Standard convolutions will always be used for upsampling layers, pre-concvolution layers, and the output layer.
        :param residual: Whether to use residual connections in double convolution blocks.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
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
                bottleneck = [Down(global_conv*(2**(i-1)), (global_conv*(2**i)) // factor, layer_type=conv_layer_type, residual=residual)]
                in_out_ratio = self.get_in_out_ratio()
                if in_out_ratio != (1.0,1.0):
                    bottleneck.append(ResizeConv((global_conv*(2**i)) // factor, scale_factor=in_out_ratio, mode='bilinear', layer_type=conv_layer_type_no_separable, residual=residual))
                self.downs.append(nn.Sequential(*bottleneck))
                self.ups.append(Up(global_conv*(2**j), (global_conv*(2**(j-1))), bilinear, layer_type=conv_layer_type, residual=residual))
        #
        self.outc = (OutConv(global_conv, n_classes, conv_layer_type=conv_layer_type_no_separable)) # Output layer uses standard convolution

    def compute_skip_connection(self, x, **kwargs):
        """
        Placeholder for any specific processing of skip connections before concatenation in the decoder.
        By default, it returns the input as is, but this method can be overriden in subclasses to implement specific processing if needed.
        """
        x = nn.Identity()(x)
        return x
    
    def get_in_out_ratio(self, size=(300, 300), **kwargs):
        """
        Computes the ratio of input to output spatial dimensions based on the operation defined in skip connections.
        This is used to determine if any resizing is needed in the bottleneck layer.
        If resizing is needed, the ResizeConv layer with extra convolutions will be added in the bottleneck to learn a better mapping between the encoder and decoder features.
        """
        dummy_input = torch.zeros(1, 1, *size)
        dummy_output = self.compute_skip_connection(dummy_input, **kwargs)
        in_out_ratio = (dummy_output.shape[2] / dummy_input.shape[2], dummy_output.shape[3] / dummy_input.shape[3])
        return in_out_ratio

    def forward(self, x, **kwargs):
        x1 = self.inc(x)
        skip_connections = [x1, ]
        x_temp = x1
        #
        # ENCODER
        for i, down in enumerate(self.downs):
            x_temp = down(x_temp)
            if i < self.n_levels - 1:
                skip_connections.append(x_temp)
        #
        # SKIP CONNECTIONS
        for i, skip_connection in enumerate(skip_connections):
            skip_connections[i] = self.compute_skip_connection(skip_connection, **kwargs)
        #
        # DECODER
        for i, up in enumerate(self.ups):
            if i < self.n_levels:
                x_temp = up(x_temp, skip_connections[-(i+1)])
        x = x_temp
        logits = self.outc(x)
        return logits

if __name__ == '__main__':

    from torchsummary import summary

    unet = UNet(n_channels=1, n_classes=1, global_conv=32, n_levels=4, bilinear=True, conv_layer_type='Conv2d', residual=True)
    unet.eval()
    summary(unet, (1, 300, 300), device="cpu")

    test_tensor = torch.randn(1, 1, 300, 300)
    with torch.no_grad():
        output = unet(test_tensor)
    print(f"Output shape: {output.shape}")

    # torch.onnx.export(unet, (torch.zeros(1, 1, 300, 300)), f='unet.onnx', input_names=['input'], output_names=['output'])