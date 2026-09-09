'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Contains the from-scratch backbone encoders shared by the IEEE benchmarking
architectures:
    - ResNet18Encoder   -> used by U-Net, FPN, BiSeNet, and SwiftNet variants.
    - MobileNetV2Encoder -> used by DeepLabV3 and U-Net MobileNetV2 variants.

Both encoders return the multi-scale feature maps needed by the decoders:
    [f1 (1/2), f2 (1/4), f3 (1/8), f4 (1/16), f5 (1/32)]
'''

from typing import List
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    '''
    Standard Conv2d -> BatchNorm2d -> ReLU block used across all architectures.

    Args:
        m (int): number of input channels.
        n (int): number of output channels.
        kernel_size (int): convolution kernel size.
        stride (int): convolution stride.
        padding (int): convolution padding.
        dilation (int): convolution dilation.
        groups (int): convolution groups (used for depthwise convolutions).
        relu6 (bool): if True, uses ReLU6 (MobileNetV2 convention) instead of ReLU.
    '''

    def __init__(self, m: int, n: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1, groups: int = 1,
                 relu6: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=m, out_channels=n, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU6() if relu6 else nn.ReLU()
        )

    def forward(self, x) -> torch.Tensor:
        return self.block(x)


class ResNetBasicBlock(nn.Module):
    '''
    Class that defines the BasicBlock of the ResNet18 architecture
    (two 3x3 convolutions with an identity or projected skip connection).
    Reference: https://arxiv.org/abs/1512.03385

    Args:
        m (int): number of input channels.
        n (int): number of output channels.
        stride (int): stride of the first convolution (2 halves the resolution).
    '''

    def __init__(self, m: int, n: int, stride: int = 1) -> None:
        super().__init__()

        # First 3x3 convolution (possibly downsampling)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=m, out_channels=n, kernel_size=(3, 3),
                      stride=(stride, stride), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # Second 3x3 convolution (no activation before the residual add)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True)
        )

        # Projection skip connection when shape changes, identity otherwise
        if stride != 1 or m != n:
            self.skip_conn = nn.Sequential(
                nn.Conv2d(in_channels=m, out_channels=n, kernel_size=(1, 1),
                          stride=(stride, stride), padding=(0, 0), bias=False),
                nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                               affine=True, track_running_stats=True)
            )
        else:
            self.skip_conn = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = out + self.skip_conn(x)
        return self.relu(out)


class ResNet18Encoder(nn.Module):
    '''
    Class that defines the full ResNet18 feature-extractor backbone from scratch
    (no fully connected head), returning multi-scale feature maps.
    Reference: https://arxiv.org/abs/1512.03385

    Feature maps returned for an input of shape (B, Cin, H, W):
        f1: (B,  64, H/2,  W/2)   -> after stem conv (before max pooling)
        f2: (B,  64, H/4,  W/4)   -> after layer1
        f3: (B, 128, H/8,  W/8)   -> after layer2
        f4: (B, 256, H/16, W/16)  -> after layer3
        f5: (B, 512, H/32, W/32)  -> after layer4

    Args:
        Cin (int): number of input channels (3 for RGB document images).
    '''

    # Output channels at each stage, useful for building decoders
    out_channels: List[int] = [64, 64, 128, 256, 512]

    def __init__(self, Cin: int = 3) -> None:
        super().__init__()

        # Stem: 7x7/2 convolution followed by 3x3/2 max pooling
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels=Cin, out_channels=64, kernel_size=(7, 7),
                      stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Four residual stages, two BasicBlocks each (ResNet18 configuration)
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(m=64, n=64, stride=1),
            ResNetBasicBlock(m=64, n=64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(m=64, n=128, stride=2),
            ResNetBasicBlock(m=128, n=128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(m=128, n=256, stride=2),
            ResNetBasicBlock(m=256, n=256, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(m=256, n=512, stride=2),
            ResNetBasicBlock(m=512, n=512, stride=1)
        )

    def forward(self, x) -> List[torch.Tensor]:
        f1 = self.stem_conv(x)          # (B, 64, H/2, W/2)
        f2 = self.layer1(self.max_pool(f1))  # (B, 64, H/4, W/4)
        f3 = self.layer2(f2)            # (B, 128, H/8, W/8)
        f4 = self.layer3(f3)            # (B, 256, H/16, W/16)
        f5 = self.layer4(f4)            # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4, f5]


class InvertedResidual(nn.Module):
    '''
    Class that defines the MobileNetV2 inverted residual block:
    1x1 expansion -> 3x3 depthwise -> 1x1 linear projection.
    Reference: https://arxiv.org/abs/1801.04381

    Args:
        m (int): number of input channels.
        n (int): number of output channels.
        stride (int): stride of the depthwise convolution.
        expand_ratio (int): channel expansion factor t of the paper.
        dilation (int): dilation of the depthwise convolution (used to keep
            spatial resolution for DeepLabV3, output stride 16).
    '''

    def __init__(self, m: int, n: int, stride: int = 1,
                 expand_ratio: int = 6, dilation: int = 1) -> None:
        super().__init__()
        hidden = m * expand_ratio
        self.use_residual = (stride == 1 and m == n)

        layers = []
        # 1x1 pointwise expansion (skipped when t=1, first bottleneck)
        if expand_ratio != 1:
            layers.append(ConvBNReLU(m=m, n=hidden, kernel_size=1, stride=1,
                                     padding=0, relu6=True))
        # 3x3 depthwise convolution (groups = hidden channels)
        layers.append(ConvBNReLU(m=hidden, n=hidden, kernel_size=3, stride=stride,
                                 padding=dilation, dilation=dilation,
                                 groups=hidden, relu6=True))
        # 1x1 linear projection (no activation, "linear bottleneck")
        layers.append(nn.Conv2d(in_channels=hidden, out_channels=n,
                                kernel_size=(1, 1), stride=(1, 1),
                                padding=(0, 0), bias=False))
        layers.append(nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1,
                                     affine=True, track_running_stats=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV2Encoder(nn.Module):
    '''
    Class that defines the full MobileNetV2 feature-extractor backbone from
    scratch (no classification head), returning multi-scale feature maps.
    Reference: https://arxiv.org/abs/1801.04381

    Bottleneck configuration (t, c, n, s) follows Table 2 of the paper:
        (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
        (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)

    Feature maps returned for an input of shape (B, Cin, H, W):
        f1: (B,  16, H/2,  W/2)
        f2: (B,  24, H/4,  W/4)
        f3: (B,  32, H/8,  W/8)
        f4: (B,  96, H/16, W/16)
        f5: (B, 320, H/32, W/32)  (H/16 when output_stride=16)

    Args:
        Cin (int): number of input channels (3 for RGB document images).
        output_stride (int): 32 for standard encoders (U-Net), 16 for
            DeepLabV3 (last downsampling replaced by dilation=2).
        include_top_conv (bool): if True, appends the final 1x1 convolution
            to 1280 channels of the original paper on top of f5 (used by the
            U-Net variant as a wide bottleneck).
    '''

    def __init__(self, Cin: int = 3, output_stride: int = 32,
                 include_top_conv: bool = False) -> None:
        super().__init__()
        assert output_stride in [16, 32], "output_stride must be 16 or 32"

        # When output_stride=16, the last stride-2 stage keeps stride 1
        # and uses dilation 2 instead (DeepLabV3 convention)
        last_stride = 2 if output_stride == 32 else 1
        last_dilation = 1 if output_stride == 32 else 2

        # Stem: 3x3/2 convolution to 32 channels
        self.stem = ConvBNReLU(m=Cin, n=32, kernel_size=3, stride=2,
                               padding=1, relu6=True)

        # Stage 1 -> f1 at 1/2 resolution (t=1, c=16, n=1, s=1)
        self.stage1 = InvertedResidual(m=32, n=16, stride=1, expand_ratio=1)

        # Stage 2 -> f2 at 1/4 resolution (t=6, c=24, n=2, s=2)
        self.stage2 = nn.Sequential(
            InvertedResidual(m=16, n=24, stride=2, expand_ratio=6),
            InvertedResidual(m=24, n=24, stride=1, expand_ratio=6)
        )

        # Stage 3 -> f3 at 1/8 resolution (t=6, c=32, n=3, s=2)
        self.stage3 = nn.Sequential(
            InvertedResidual(m=24, n=32, stride=2, expand_ratio=6),
            InvertedResidual(m=32, n=32, stride=1, expand_ratio=6),
            InvertedResidual(m=32, n=32, stride=1, expand_ratio=6)
        )

        # Stage 4 -> f4 at 1/16 resolution (t=6, c=64, n=4, s=2) + (t=6, c=96, n=3, s=1)
        self.stage4 = nn.Sequential(
            InvertedResidual(m=32, n=64, stride=2, expand_ratio=6),
            InvertedResidual(m=64, n=64, stride=1, expand_ratio=6),
            InvertedResidual(m=64, n=64, stride=1, expand_ratio=6),
            InvertedResidual(m=64, n=64, stride=1, expand_ratio=6),
            InvertedResidual(m=64, n=96, stride=1, expand_ratio=6),
            InvertedResidual(m=96, n=96, stride=1, expand_ratio=6),
            InvertedResidual(m=96, n=96, stride=1, expand_ratio=6)
        )

        # Stage 5 -> f5 at 1/32 (or dilated 1/16) resolution
        # (t=6, c=160, n=3, s=2) + (t=6, c=320, n=1, s=1)
        self.stage5 = nn.Sequential(
            InvertedResidual(m=96, n=160, stride=last_stride,
                             expand_ratio=6, dilation=last_dilation),
            InvertedResidual(m=160, n=160, stride=1,
                             expand_ratio=6, dilation=last_dilation),
            InvertedResidual(m=160, n=160, stride=1,
                             expand_ratio=6, dilation=last_dilation),
            InvertedResidual(m=160, n=320, stride=1,
                             expand_ratio=6, dilation=last_dilation)
        )

        # Optional final 1x1 convolution to 1280 channels (paper's last layer)
        if include_top_conv:
            self.top_conv = ConvBNReLU(m=320, n=1280, kernel_size=1,
                                       stride=1, padding=0, relu6=True)
            self.out_channels = [16, 24, 32, 96, 1280]
        else:
            self.top_conv = nn.Identity()
            self.out_channels = [16, 24, 32, 96, 320]

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)        # (B, 32, H/2, W/2)
        f1 = self.stage1(x)     # (B, 16, H/2, W/2)
        f2 = self.stage2(f1)    # (B, 24, H/4, W/4)
        f3 = self.stage3(f2)    # (B, 32, H/8, W/8)
        f4 = self.stage4(f3)    # (B, 96, H/16, W/16)
        f5 = self.stage5(f4)    # (B, 320, H/32 or H/16, W/32 or W/16)
        f5 = self.top_conv(f5)  # (B, 1280, ...) when include_top_conv=True
        return [f1, f2, f3, f4, f5]
