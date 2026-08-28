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
            if targets.dim() == 3:  # B x H x W -> B x 1 x H x W
                targets = targets.unsqueeze(1)
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


class SegCrossEntropyLoss(nn.Module):
    """
    Pure cross-entropy loss for segmentation, exposed with the same
    (preds, targets) signature as DiceLoss and CombinedLoss:
        - Binary: BCEWithLogitsLoss
        - Multi-class: CrossEntropyLoss

    Named SegCrossEntropyLoss (not CrossEntropyLoss) so it does not shadow
    torch.nn.CrossEntropyLoss for callers doing `from src.training import *`.
    """

    def __init__(self, binary: bool = True):
        """
        Args:
            - binary (bool): whether task is binary segmentation
        """
        super().__init__()

        self.binary = binary

        if self.binary:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: B x C x H x W (logits)
            targets:
                - Binary: B x 1 x H x W or B x H x W
                - Multi-class: B x H x W (class indices)

        Returns:
            cross-entropy loss (scalar)
        """
        if self.binary:
            # BCEWithLogits needs float targets with the same shape as preds
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # B x 1 x H x W
            return self.ce_loss(preds, targets.float())

        # CrossEntropy expects class indices (no one-hot)
        return self.ce_loss(preds, targets.long())


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
        ignore_background: bool = False,
        smooth: float = 1e-7
    ):
        """
        Args:
            - binary (bool): whether task is binary segmentation
            - weight_ce (float): weight for CE/BCE loss
            - weight_dice (float): weight for Dice loss
            - ignore_background (bool): ignore class 0 in Dice (multi-class only)
            - smooth (float): smoothing factor handed to the Dice component
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

        self.dice_loss = DiceLoss(smooth=smooth, ignore_background=ignore_background)

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


# Valid values for the --loss_fn CLI flag of the training scripts.
LOSS_FN_CHOICES = ("dice", "cross-entropy", "combined")


def get_loss_fn(
    loss_name: str = "combined",
    binary: bool = True,
    smooth: float = 1e-7,
    weight_ce: float = 0.5,
    weight_dice: float = 0.5,
    ignore_background: bool = False
) -> nn.Module:
    '''
    Factory that builds a segmentation loss by name, so training scripts can
    swap losses from the CLI without any other code change.

    Every returned module accepts the exact same (preds, targets) call used by
    src.training.train: preds are raw logits B x C x H x W, targets are
    B x 1 x H x W floats (binary) or B x H x W class indices (multi-class).

    Args:
        - loss_name (str): one of LOSS_FN_CHOICES -> "dice", "cross-entropy", "combined".
          Underscores and casing are normalized ("Cross_Entropy" works).
        - binary (bool): whether task is binary segmentation
        - smooth (float): smoothing factor for the Dice component (unused by "cross-entropy")
        - weight_ce (float): weight for CE/BCE loss (only used by "combined")
        - weight_dice (float): weight for Dice loss (only used by "combined")
        - ignore_background (bool): ignore class 0 in Dice (multi-class only). Matches
          CombinedLoss behaviour: it affects the Dice term only, never the CE term.

    Returns:
        nn.Module: the requested loss function.

    Raises:
        ValueError: if loss_name is not one of LOSS_FN_CHOICES.
    '''
    name = loss_name.strip().lower().replace("_", "-")

    if name not in LOSS_FN_CHOICES:
        raise ValueError(
            f"Unknown loss_fn '{loss_name}'. Expected one of {list(LOSS_FN_CHOICES)}."
        )

    if name == "dice":
        return DiceLoss(smooth=smooth, ignore_background=ignore_background)

    if name == "cross-entropy":
        return SegCrossEntropyLoss(binary=binary)

    return CombinedLoss(
        binary=binary,
        weight_ce=weight_ce,
        weight_dice=weight_dice,
        ignore_background=ignore_background,
        smooth=smooth,
    )