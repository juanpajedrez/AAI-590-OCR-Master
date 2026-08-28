'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the DeepLabV3 architecture with a MobileNetV2 backbone for text
detection tasks (~5-6M params): dilated MobileNetV2 at output stride 16
followed by an ASPP (Atrous Spatial Pyramid Pooling) head.
References:
    - DeepLabV3: https://arxiv.org/abs/1706.05587
    - MobileNetV2: https://arxiv.org/abs/1801.04381
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNReLU, MobileNetV2Encoder


class ASPP(nn.Module):
    '''
    Class that defines the Atrous Spatial Pyramid Pooling module: parallel
    1x1 and dilated 3x3 convolutions plus global image pooling, concatenated
    and projected to `n` channels.

    Args:
        m (int): number of input channels (320 for MobileNetV2).
        n (int): number of output channels (256 in the paper).
        rates (tuple): dilation rates of the atrous branches (output stride 16).
    '''

    def __init__(self, m: int = 320, n: int = 256,
                 rates: tuple = (6, 12, 18)) -> None:
        super().__init__()

        # Branch 1: simple 1x1 convolution
        self.branch_1x1 = ConvBNReLU(m=m, n=n, kernel_size=1, stride=1, padding=0)

        # Branches 2-4: 3x3 atrous convolutions with increasing rates
        self.branch_atrous = nn.ModuleList([
            ConvBNReLU(m=m, n=n, kernel_size=3, stride=1,
                       padding=rate, dilation=rate)
            for rate in rates
        ])

        # Branch 5: image-level global average pooling
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            ConvBNReLU(m=m, n=n, kernel_size=1, stride=1, padding=0)
        )

        # Projection of the concatenated branches with dropout
        self.project = nn.Sequential(
            ConvBNReLU(m=n * (2 + len(rates)), n=n, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x) -> torch.Tensor:
        size = x.shape[2:]
        branches = [self.branch_1x1(x)]
        for branch in self.branch_atrous:
            branches.append(branch(x))
        pooled = self.branch_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=True)
        branches.append(pooled)
        return self.project(torch.cat(branches, dim=1))


class DeepLabV3MobileNetV2Model(nn.Module):
    '''
    Class that defines the full DeepLabV3 architecture with a MobileNetV2
    backbone at output stride 16 (last stage dilated), an ASPP head, and a
    16x bilinear upsampling back to full resolution.

    Args:
        Cin (int): number of input channels for the encoder.
        N (int): number of output channels (1 binary / num_classes semantic).
        aspp_channels (int): number of ASPP output channels (default 256).
    '''

    def __init__(self, Cin: int = 3, N: int = 1, aspp_channels: int = 256) -> None:
        super().__init__()

        # Dilated backbone: final feature is (B, 320, H/16, W/16)
        self.encoder = MobileNetV2Encoder(Cin=Cin, output_stride=16)
        c5 = self.encoder.out_channels[-1]

        # ASPP head with rates (6, 12, 18) for output stride 16
        self.aspp = ASPP(m=c5, n=aspp_channels, rates=(6, 12, 18))

        # Classifier: one 3x3 refinement conv + 1x1 projection to N logits
        self.head_conv = ConvBNReLU(m=aspp_channels, n=aspp_channels,
                                    kernel_size=3, stride=1, padding=1)
        self.segmentation_head = nn.Conv2d(in_channels=aspp_channels,
                                           out_channels=N, kernel_size=(1, 1),
                                           stride=(1, 1), padding=(0, 0))

    def forward(self, x) -> torch.Tensor:
        features = self.encoder(x)
        d = self.aspp(features[-1])          # (B, 256, H/16, W/16)
        logits = self.segmentation_head(self.head_conv(d))
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=True)
