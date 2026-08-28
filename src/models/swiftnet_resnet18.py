'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the SwiftNet architecture with a ResNet18 backbone for text
detection tasks (~11.8M params): a lightweight single-scale model with a
Spatial Pyramid Pooling bottleneck and a slim 128-channel upsampling decoder.
Reference: "In Defense of Pre-trained ImageNet Architectures for Real-time
Semantic Segmentation of Road-driving Images" https://arxiv.org/abs/1903.08469
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNReLU, ResNet18Encoder


class SpatialPyramidPooling(nn.Module):
    '''
    Class that defines the SwiftNet SPP bottleneck: the 1/32 feature map is
    average-pooled onto several grid sizes, each grid is projected with a
    1x1 convolution, upsampled back, concatenated with a projection of the
    input, and blended into `n` output channels.

    Args:
        m (int): number of input channels (512 for ResNet18).
        n (int): number of output channels (128 in the paper).
        grids (tuple): pyramid grid sizes to pool onto.
        level_channels (int): channels of each pooled pyramid level.
    '''

    def __init__(self, m: int = 512, n: int = 128,
                 grids: tuple = (8, 4, 2, 1), level_channels: int = 42) -> None:
        super().__init__()
        self.grids = grids

        # 1x1 projection of the un-pooled input feature
        self.input_conv = ConvBNReLU(m=m, n=n, kernel_size=1, stride=1, padding=0)

        # One 1x1 projection per pyramid level (applied after pooling)
        self.level_convs = nn.ModuleList([
            ConvBNReLU(m=n, n=level_channels, kernel_size=1, stride=1, padding=0)
            for _ in grids
        ])

        # Final blending convolution over the concatenated pyramid
        self.fuse_conv = ConvBNReLU(m=n + len(grids) * level_channels, n=n,
                                    kernel_size=1, stride=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        x = self.input_conv(x)
        levels = [x]
        for grid, conv in zip(self.grids, self.level_convs):
            pooled = F.adaptive_avg_pool2d(x, output_size=grid)
            pooled = conv(pooled)
            pooled = F.interpolate(pooled, size=x.shape[2:],
                                   mode="bilinear", align_corners=True)
            levels.append(pooled)
        return self.fuse_conv(torch.cat(levels, dim=1))


class SwiftNetUpsampleBlock(nn.Module):
    '''
    Class that defines the SwiftNet lightweight decoder module: bilinear 2x
    upsampling, addition of a 1x1-projected encoder skip connection, and one
    3x3 blending convolution.

    Args:
        skip (int): number of channels of the encoder skip feature.
        n (int): number of decoder channels (128 in the paper).
    '''

    def __init__(self, skip: int, n: int = 128) -> None:
        super().__init__()
        self.skip_conv = ConvBNReLU(m=skip, n=n, kernel_size=1, stride=1, padding=0)
        self.blend_conv = ConvBNReLU(m=n, n=n, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = x + self.skip_conv(skip)
        return self.blend_conv(x)


class SwiftNetResNet18Model(nn.Module):
    '''
    Class that defines the full SwiftNet architecture with a ResNet18
    encoder, an SPP bottleneck at 1/32 resolution, and three 128-channel
    upsampling modules back to 1/4 resolution before the final 4x upsampling.

    Args:
        Cin (int): number of input channels for the encoder.
        N (int): number of output channels (1 binary / num_classes semantic).
        decoder_channels (int): width of the slim decoder (default 128).
    '''

    def __init__(self, Cin: int = 3, N: int = 1, decoder_channels: int = 128) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(Cin=Cin)
        _, c2, c3, c4, c5 = self.encoder.out_channels

        # SPP bottleneck on the 1/32 feature map
        self.spp = SpatialPyramidPooling(m=c5, n=decoder_channels)

        # Slim upsampling path with lateral skip connections
        self.upsample_16 = SwiftNetUpsampleBlock(skip=c4, n=decoder_channels)  # to 1/16
        self.upsample_8 = SwiftNetUpsampleBlock(skip=c3, n=decoder_channels)   # to 1/8
        self.upsample_4 = SwiftNetUpsampleBlock(skip=c2, n=decoder_channels)   # to 1/4

        # Segmentation head at 1/4 resolution, then 4x upsampling
        self.segmentation_head = nn.Conv2d(in_channels=decoder_channels,
                                           out_channels=N, kernel_size=(1, 1),
                                           stride=(1, 1), padding=(0, 0))

    def forward(self, x) -> torch.Tensor:
        _, f2, f3, f4, f5 = self.encoder(x)

        d = self.spp(f5)                 # (B, 128, H/32, W/32)
        d = self.upsample_16(d, f4)      # (B, 128, H/16, W/16)
        d = self.upsample_8(d, f3)       # (B, 128, H/8, W/8)
        d = self.upsample_4(d, f2)       # (B, 128, H/4, W/4)

        logits = self.segmentation_head(d)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=True)
