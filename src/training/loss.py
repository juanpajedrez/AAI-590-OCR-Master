'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-24
The following contains the PyTorch loss functions to perform:
    binary text detection -> binary semantic segmentation.
    multi-class PDF region detection -> multi-class segmentation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    '''
    Class that will be used to simulate the 
    Dice Loss function for text detection, both in the
    binary and semantic segmentation case.
    '''
    def __init__(self, smooth=1e-7, ignore_background: bool = False):
        '''
        Args:
            - smooth (float): float value to avoid divition by zero when using the loss function
            - ignore_background (bool): boolean flag in order to ignore class 0 -> background for semantic
            segmentation.
        '''
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: B x C x H x W
        targets: B x H x W (class indices) or B x 1 x H x W (binary)
        """
        if preds.shape[1] == 1:
            # Binary case
            probs = torch.sigmoid(preds)
            targets = targets.float()
            intersection = (probs * targets).sum(dim=(2,3))
            union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()
        else:
            # Multi-class case
            if targets.dim() == 3:  # B x H x W
                targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0,3,1,2).float()
            probs = F.softmax(preds, dim=1)

            if self.ignore_background:
                probs = probs[:,1:,...]   # skip class 0
                targets = targets[:,1:,...]

            intersection = (probs * targets).sum(dim=(2,3))
            union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()