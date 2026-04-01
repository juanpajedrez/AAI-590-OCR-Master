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


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation:
        - Binary: BCEWithLogits + Dice
        - Multi-class: CrossEntropy + Dice
    """

    def __init__(
        self,
        binary: bool = True,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        ignore_background: bool = False
    ):
        """
        Args:
            - binary (bool): whether task is binary segmentation
            - weight_ce (float): weight for CE/BCE loss
            - weight_dice (float): weight for Dice loss
            - ignore_background (bool): ignore class 0 in Dice (multi-class only)
        """
        super().__init__()

        self.binary = binary
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        # Loss components
        if self.binary:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(ignore_background=ignore_background)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: B x C x H x W (logits)
            targets:
                - Binary: B x 1 x H x W or B x H x W
                - Multi-class: B x H x W (class indices)

        Returns:
            combined loss (scalar)
        """

        # ---- CE / BCE LOSS ----
        if self.binary:
            # Ensure same shape
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # B x 1 x H x W
            targets = targets.float()

            ce = self.ce_loss(preds, targets)

        else:
            # CrossEntropy expects class indices (no one-hot)
            ce = self.ce_loss(preds, targets.long())

        # ---- DICE LOSS ----
        dice = self.dice_loss(preds, targets)

        # ---- COMBINED ----
        loss = self.weight_ce * ce + self.weight_dice * dice

        return loss