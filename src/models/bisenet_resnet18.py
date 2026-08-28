'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the BiSeNet (Bilateral Segmentation Network) V1 architecture with a
ResNet18 backbone for text detection tasks (~13M params).
Reference: https://arxiv.org/abs/1808.00897
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNReLU, ResNet18Encoder


class SpatialPath(nn.Module):
    '''
    Class that defines the BiSeNet Spatial Path: three stride-2 convolutions
    that preserve rich spatial detail at 1/8 resolution with wide channels.

    Args:
        Cin (int): number of input channels.
        n (int): number of output channels (128 in the paper).
    '''

    def __init__(self, Cin: int = 3, n: int = 128) -> None:
        super().__init__()
        self.conv_7x7 = ConvBNReLU(m=Cin, n=64, kernel_size=7, stride=2, padding=3)
        self.conv_3x3_1 = ConvBNReLU(m=64, n=64, kernel_size=3, stride=2, padding=1)
        self.conv_3x3_2 = ConvBNReLU(m=64, n=64, kernel_size=3, stride=2, padding=1)
        self.conv_1x1 = ConvBNReLU(m=64, n=n, kernel_size=1, stride=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_7x7(x)      # 1/2 resolution
        x = self.conv_3x3_1(x)    # 1/4 resolution
        x = self.conv_3x3_2(x)    # 1/8 resolution
        return self.conv_1x1(x)   # (B, 128, H/8, W/8)


class AttentionRefinementModule(nn.Module):
    '''
    Class that defines the BiSeNet ARM: global average pooling produces a
    channel attention vector that re-weights the input feature map.

    Args:
        m (int): number of input channels.
        n (int): number of output channels.
    '''

    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(m=m, n=n, kernel_size=3, stride=1, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        attention = self.attention(x)
        return x * attention


class FeatureFusionModule(nn.Module):
    '''
    Class that defines the BiSeNet FFM: concatenates the spatial and context
    features, projects them, and applies a channel attention residual.

    Args:
        m (int): number of input channels (spatial + context concatenated).
        n (int): number of output channels.
    '''

    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(m=m, n=n, kernel_size=1, stride=1, padding=0)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=n, out_channels=n // 4, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=n // 4, out_channels=n, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_spatial, x_context) -> torch.Tensor:
        x = torch.cat([x_spatial, x_context], dim=1)
        x = self.conv(x)
        attention = self.attention(x)
        return x + x * attention


class BiSeNetResNet18Model(nn.Module):
    '''
    Class that defines the full BiSeNet V1 architecture with a ResNet18
    context path. The Spatial Path keeps detail at 1/8 resolution while the
    Context Path (ResNet18 + global pooling + ARMs) provides a large
    receptive field; both are merged by the Feature Fusion Module.

    Args:
        Cin (int): number of input channels for both paths.
        N (int): number of output channels (1 binary / num_classes semantic).
    '''

    def __init__(self, Cin: int = 3, N: int = 1) -> None:
        super().__init__()

        # Spatial path: (B, 128, H/8, W/8)
        self.spatial_path = SpatialPath(Cin=Cin, n=128)

        # Context path: ResNet18 backbone with ARMs at 1/16 and 1/32
        self.context_path = ResNet18Encoder(Cin=Cin)
        _, _, _, c4, c5 = self.context_path.out_channels

        self.arm_16 = AttentionRefinementModule(m=c4, n=128)
        self.arm_32 = AttentionRefinementModule(m=c5, n=128)

        # Global context tail from the deepest feature
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            ConvBNReLU(m=c5, n=128, kernel_size=1, stride=1, padding=0)
        )

        # Refinement convolutions after each context upsampling
        self.refine_32 = ConvBNReLU(m=128, n=128, kernel_size=3, stride=1, padding=1)
        self.refine_16 = ConvBNReLU(m=128, n=128, kernel_size=3, stride=1, padding=1)

        # Feature fusion of spatial (128) + context (128) channels
        self.ffm = FeatureFusionModule(m=256, n=256)

        # Segmentation head at 1/8 resolution, then 8x upsampling
        self.head_conv = ConvBNReLU(m=256, n=256, kernel_size=3, stride=1, padding=1)
        self.segmentation_head = nn.Conv2d(in_channels=256, out_channels=N,
                                           kernel_size=(1, 1), stride=(1, 1),
                                           padding=(0, 0))

    def forward(self, x) -> torch.Tensor:
        # Spatial path detail features
        x_spatial = self.spatial_path(x)

        # Context path multi-scale features
        _, _, _, f4, f5 = self.context_path(x)

        # Global average pooling context, broadcast onto the 1/32 feature
        x_global = self.global_context(f5)

        # 1/32 branch: ARM + global context, upsample to 1/16 and refine
        x_32 = self.arm_32(f5) + x_global
        x_32 = F.interpolate(x_32, size=f4.shape[2:], mode="bilinear", align_corners=True)
        x_32 = self.refine_32(x_32)

        # 1/16 branch: ARM + upsampled 1/32 branch, upsample to 1/8 and refine
        x_16 = self.arm_16(f4) + x_32
        x_16 = F.interpolate(x_16, size=x_spatial.shape[2:], mode="bilinear", align_corners=True)
        x_context = self.refine_16(x_16)

        # Fuse both paths and predict at 1/8 resolution
        x_fused = self.ffm(x_spatial, x_context)
        logits = self.segmentation_head(self.head_conv(x_fused))

        # Upsample from 1/8 back to full input resolution
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=True)
