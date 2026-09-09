'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the FPN (Feature Pyramid Network) architecture with a ResNet18
encoder for text detection tasks (~13-14M params).
References:
    - FPN: https://arxiv.org/abs/1612.03144
    - Panoptic FPN (segmentation branch): https://arxiv.org/abs/1901.02446
'''

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet18Encoder


class FPNLateralBlock(nn.Module):
    '''
    Class that defines the FPN top-down pathway block: a 1x1 lateral
    convolution on the encoder feature plus the 2x upsampled coarser
    pyramid feature.

    Args:
        m (int): number of input channels of the encoder skip feature.
        pyramid_channels (int): number of channels of every pyramid level.
    '''

    def __init__(self, m: int, pyramid_channels: int = 256) -> None:
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels=m, out_channels=pyramid_channels,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x, skip) -> torch.Tensor:
        # Upsample the coarser pyramid feature and add the lateral projection
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x + self.lateral_conv(skip)


class FPNSegmentationBlock(nn.Module):
    '''
    Class that defines the Panoptic-FPN segmentation branch block: each
    pyramid level is processed by (3x3 conv -> GroupNorm -> ReLU -> 2x upsample)
    repeated until it reaches 1/4 of the input resolution.

    Args:
        m (int): number of input channels (pyramid channels).
        n (int): number of output channels (segmentation channels).
        num_upsamples (int): how many 2x upsampling steps to reach 1/4 scale.
    '''

    def __init__(self, m: int, n: int, num_upsamples: int = 0) -> None:
        super().__init__()
        blocks = []
        for i in range(max(1, num_upsamples)):
            in_ch = m if i == 0 else n
            blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=n, kernel_size=(3, 3),
                          stride=(1, 1), padding=(1, 1), bias=False),
                nn.GroupNorm(num_groups=32, num_channels=n),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                if i < num_upsamples else nn.Identity()
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x) -> torch.Tensor:
        return self.blocks(x)


class FPNResNet18Model(nn.Module):
    '''
    Class that defines the full FPN architecture with a ResNet18 encoder.
    The encoder features C2..C5 are merged in a top-down pyramid P2..P5
    (256 channels), each level is refined to 128 segmentation channels at
    1/4 scale, summed, and upsampled to full resolution.

    Args:
        Cin (int): number of input channels for the encoder.
        N (int): number of output channels (1 binary / num_classes semantic).
        pyramid_channels (int): channels of the pyramid levels (default 256).
        segmentation_channels (int): channels of the segmentation branch (default 128).
    '''

    def __init__(self, Cin: int = 3, N: int = 1,
                 pyramid_channels: int = 256,
                 segmentation_channels: int = 128) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(Cin=Cin)
        c1, c2, c3, c4, c5 = self.encoder.out_channels

        # Top of the pyramid: 1x1 projection of the deepest feature (C5 -> P5)
        self.p5_conv = nn.Conv2d(in_channels=c5, out_channels=pyramid_channels,
                                 kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Top-down pathway with lateral connections (C4 -> P4, C3 -> P3, C2 -> P2)
        self.p4_block = FPNLateralBlock(m=c4, pyramid_channels=pyramid_channels)
        self.p3_block = FPNLateralBlock(m=c3, pyramid_channels=pyramid_channels)
        self.p2_block = FPNLateralBlock(m=c2, pyramid_channels=pyramid_channels)

        # Segmentation branch: refine every level to 1/4 of the input resolution
        self.seg_block_5 = FPNSegmentationBlock(m=pyramid_channels, n=segmentation_channels, num_upsamples=3)
        self.seg_block_4 = FPNSegmentationBlock(m=pyramid_channels, n=segmentation_channels, num_upsamples=2)
        self.seg_block_3 = FPNSegmentationBlock(m=pyramid_channels, n=segmentation_channels, num_upsamples=1)
        self.seg_block_2 = FPNSegmentationBlock(m=pyramid_channels, n=segmentation_channels, num_upsamples=0)

        # Final classifier: dropout + 1x1 conv, then 4x upsample to full resolution
        self.dropout = nn.Dropout2d(p=0.2)
        self.segmentation_head = nn.Conv2d(in_channels=segmentation_channels,
                                           out_channels=N, kernel_size=(1, 1),
                                           stride=(1, 1), padding=(0, 0))

    def forward(self, x) -> torch.Tensor:
        _, f2, f3, f4, f5 = self.encoder(x)

        # Build the pyramid top-down
        p5 = self.p5_conv(f5)
        p4 = self.p4_block(p5, f4)
        p3 = self.p3_block(p4, f3)
        p2 = self.p2_block(p3, f2)

        # Merge all levels at 1/4 resolution by summation
        s = self.seg_block_5(p5) + self.seg_block_4(p4) \
            + self.seg_block_3(p3) + self.seg_block_2(p2)

        s = self.dropout(s)
        s = self.segmentation_head(s)

        # Upsample from 1/4 back to full input resolution
        return F.interpolate(s, scale_factor=4, mode="bilinear", align_corners=True)
