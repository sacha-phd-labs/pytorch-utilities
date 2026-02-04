import pytest
import torch

from pytorcher.models import UNet

class TestUnetModel:
    @pytest.mark.parametrize("conv_layer_type, residual", [
        ('Conv2d', False),
        ('Conv2d', True),
        ('SeparableConv2d', False),
        ('SeparableConv2d', True),
        ('SinogramConv2d', False),
        ('SinogramConv2d', True),
        ('SinogramSeparableConv2d', False),
        ('SinogramSeparableConv2d', True),
    ])
    def test_unet_model_initialization(self, conv_layer_type, residual, n_channels=1, n_classes=1, n_levels=3, global_conv=32, bilinear=True):
        model = UNet(n_channels=n_channels,
                          n_classes=n_classes,
                          n_levels=n_levels,
                          global_conv=global_conv,
                          conv_layer_type=conv_layer_type,
                          residual=residual,
                          bilinear=bilinear)
        #
        dummy_tensor = torch.randn(1, n_channels, 128, 128)
        with torch.no_grad():
            output = model(dummy_tensor)
        assert output.shape == (1, n_classes, 128, 128), f"Output shape mismatch: expected {(1, n_classes, 128, 128)}, got {output.shape}"