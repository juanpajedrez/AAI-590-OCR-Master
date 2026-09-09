'''
Author: Juan Pablo Triana Martinez
Date: 2026-08-27
Factory that builds any of the benchmarked segmentation architectures by
name, so training scripts and notebooks can select them with a single flag.
'''

import torch.nn as nn

from .linknet_model import LinknetModel
from .unet_models import UNetResNet18Model, UNetMobileNetV2Model
from .fpn_resnet18 import FPNResNet18Model
from .bisenet_resnet18 import BiSeNetResNet18Model
from .swiftnet_resnet18 import SwiftNetResNet18Model
from .deeplabv3_mobilenetv2 import DeepLabV3MobileNetV2Model

# Registry of architecture name -> model class. Every class shares the
# constructor signature (Cin, N) and maps (B, Cin, H, W) -> (B, N, H, W).
ARCH_REGISTRY = {
    "linknet-resnet": LinknetModel,
    "unet-resnet18": UNetResNet18Model,
    "fpn-resnet18": FPNResNet18Model,
    "bisenet-resnet18": BiSeNetResNet18Model,
    "swiftnet-resnet18": SwiftNetResNet18Model,
    "deeplabv3-mobilenetv2": DeepLabV3MobileNetV2Model,
    "unet-mobilenetv2": UNetMobileNetV2Model,
}

ARCH_CHOICES = tuple(ARCH_REGISTRY.keys())


def build_model(arch: str = "linknet-resnet", Cin: int = 3, N: int = 1) -> nn.Module:
    '''
    Builds a segmentation model by architecture name.

    Args:
        arch (str): one of ARCH_CHOICES (default "linknet-resnet").
        Cin (int): number of input channels (3 for RGB document images).
        N (int): number of output channels (1 binary / num_classes semantic).

    Returns:
        nn.Module: the instantiated segmentation model.
    '''
    if arch not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(ARCH_CHOICES)}"
        )
    return ARCH_REGISTRY[arch](Cin=Cin, N=N)
