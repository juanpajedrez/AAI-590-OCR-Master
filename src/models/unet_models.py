'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the U-Net architectures for text detection tasks:
    - UNetResNet18Model     (~14.3M params) -> ResNet18 encoder + U-Net decoder.
    - UNetMobileNetV2Model  (~6.6M params)  -> MobileNetV2 encoder + U-Net decoder.
Reference: https://arxiv.org/abs/1505.04597
'''

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNReLU, ResNet18Encoder, MobileNetV2Encoder


class UNetDecoderBlock(nn.Module):
    '''
    Class that defines a U-Net decoder block: bilinear 2x upsampling,
    concatenation with the encoder skip feature, and two 3x3 convolutions.

    Args:
        m (int): number of input channels (from the previous decoder stage).
        skip (int): number of channels of the encoder skip connection (0 if none).
        n (int): number of output channels.
    '''

    def __init__(self, m: int, skip: int, n: int) -> None:
        super().__init__()
        self.conv_block_1 = ConvBNReLU(m=m + skip, n=n, kernel_size=3,
                                       stride=1, padding=1)
        self.conv_block_2 = ConvBNReLU(m=n, n=n, kernel_size=3,
                                       stride=1, padding=1)

    def forward(self, x, skip=None) -> torch.Tensor:
        # Bilinear 2x upsampling (avoids checkerboard artifacts of ConvTranspose2d)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetDecoder(nn.Module):
    '''
    Class that defines the full U-Net decoder over 5 encoder feature maps.

    Args:
        encoder_channels (List[int]): channels of [f1, f2, f3, f4, f5].
        decoder_channels (List[int]): output channels of the 5 decoder blocks.
    '''

    def __init__(self, encoder_channels: List[int],
                 decoder_channels: List[int] = [256, 128, 64, 32, 16]) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels
        d5, d4, d3, d2, d1 = decoder_channels

        self.decoder_block_5 = UNetDecoderBlock(m=c5, skip=c4, n=d5)
        self.decoder_block_4 = UNetDecoderBlock(m=d5, skip=c3, n=d4)
        self.decoder_block_3 = UNetDecoderBlock(m=d4, skip=c2, n=d3)
        self.decoder_block_2 = UNetDecoderBlock(m=d3, skip=c1, n=d2)
        self.decoder_block_1 = UNetDecoderBlock(m=d2, skip=0, n=d1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4, f5 = features
        x = self.decoder_block_5(f5, f4)   # 1/16 resolution
        x = self.decoder_block_4(x, f3)    # 1/8 resolution
        x = self.decoder_block_3(x, f2)    # 1/4 resolution
        x = self.decoder_block_2(x, f1)    # 1/2 resolution
        x = self.decoder_block_1(x, None)  # full resolution
        return x


class UNetResNet18Model(nn.Module):
    '''
    Class that defines the full U-Net architecture with a ResNet18 encoder
    (~14.3M parameters).

    Args:
        Cin (int): number of input channels for the encoder.
        N (int): number of output channels (1 binary / num_classes semantic).
    '''

    def __init__(self, Cin: int = 3, N: int = 1) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(Cin=Cin)
        self.decoder = UNetDecoder(encoder_channels=self.encoder.out_channels,
                                   decoder_channels=[256, 128, 64, 32, 16])
        # 3x3 segmentation head that maps the last decoder features to N logits
        self.segmentation_head = nn.Conv2d(in_channels=16, out_channels=N,
                                           kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1))

    def forward(self, x) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)
        return self.segmentation_head(x)


class UNetMobileNetV2Model(nn.Module):
    '''
    Class that defines the full U-Net architecture with a MobileNetV2 encoder
    (~6.6M parameters).

    Args:
        Cin (int): number of input channels for the encoder.
        N (int): number of output channels (1 binary / num_classes semantic).
    '''

    def __init__(self, Cin: int = 3, N: int = 1) -> None:
        super().__init__()
        # include_top_conv=True appends the paper's final 1x1 conv to 1280
        # channels, giving the decoder a wide bottleneck (~6.6M total params)
        self.encoder = MobileNetV2Encoder(Cin=Cin, output_stride=32,
                                          include_top_conv=True)
        self.decoder = UNetDecoder(encoder_channels=self.encoder.out_channels,
                                   decoder_channels=[256, 128, 64, 32, 16])
        self.segmentation_head = nn.Conv2d(in_channels=16, out_channels=N,
                                           kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1))

    def forward(self, x) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)
        return self.segmentation_head(x)
