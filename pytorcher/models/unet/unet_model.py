""" Full assembly of the parts to form the complete network """

from pytorcher.models.unet.unet_parts import *
from pytorcher.utils.processing import normalize_batch, rescale_batch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, global_conv=32, n_levels=3, bilinear=False, layer_type='Conv2d', normalize_input=False):
        """
        :param n_channels: Number of input channels
        :param n_classes: Number of output channels
        :param bilinear: Whether to use bilinear upsampling or transposed convolutions
        :param layer_type: Type of convolutional layer to use ('standard' or 'separable'). Standard convolutions will always be used for upsampling layers, pre-concvolution layers, and the output layer.
        :param normalize_input: Whether to normalize the input in the range [0, 1] before feeding it to the network. If set to True, output will be rescaled accordingly to match the input scale.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.normalize_input = normalize_input
        assert n_levels in [3, 4], "Only 3 or 4 levels are supported currently."
        self.n_levels = n_levels
        #
        layer_type_no_separable = layer_type.replace('Separable', '') # Ensure that separable convolutions are not used in initial, upsampling, and output layers
        self.inc = (DoubleConv(n_channels, global_conv, layer_type=layer_type_no_separable)) # Initial layer uses standard convolution
        #
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(1,n_levels+1):
            j = n_levels - i + 1
            if i < n_levels:
                self.downs.append(Down(global_conv*(2**(i-1)), global_conv*(2**i), layer_type=layer_type))
                self.ups.append(Up(global_conv*(2**j), (global_conv*(2**(j-1))) // factor, bilinear, layer_type=layer_type))
            else:
                self.downs.append(Down(global_conv*(2**(i-1)), (global_conv*(2**i)) // factor, layer_type=layer_type))
                self.ups.append(Up(global_conv*(2**j), (global_conv*(2**(j-1))), bilinear, layer_type=layer_type))
        #
        self.outc = (OutConv(global_conv, n_classes, layer_type=layer_type_no_separable)) # Output layer uses standard convolution

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
        if self.normalize_input:
            logits = rescale_batch(logits, x_mins, x_maxs)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

if __name__ == '__main__':

    from torchsummary import summary

    unet = UNet(n_channels=1, n_classes=1, global_conv=32, n_levels=4, bilinear=True, layer_type='SinogramConv2d', normalize_input=True)
    unet.eval()
    summary(unet, (1, 300, 300), device="cpu")

    torch.onnx.export(unet, (torch.zeros(1, 1, 300, 300)), f='unet.onnx', input_names=['input'], output_names=['output'])
