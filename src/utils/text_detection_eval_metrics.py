import torch
import numpy as np
from typing import Dict
from scipy import ndimage

EPS = 1e-7


# =========================================================
# 🔹 Logits → Predictions
# =========================================================

def logits_to_binary_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def logits_to_multiclass_mask(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


# =========================================================
# 🔹 Pixel-level Binary Metrics
# =========================================================

def get_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:

    preds = logits_to_binary_mask(logits, threshold)

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = torch.sum((preds == 1) & (targets == 1)).float()
    fp = torch.sum((preds == 1) & (targets == 0)).float()
    fn = torch.sum((preds == 0) & (targets == 1)).float()
    tn = torch.sum((preds == 0) & (targets == 0)).float()

    if (tp + fp + fn) == 0:
        iou = torch.tensor(1.0)
        dice = torch.tensor(1.0)
    else:
        iou = tp / (tp + fp + fn + EPS)
        dice = (2 * tp) / (2 * tp + fp + fn + EPS)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1.item(),
        "iou_pixel": iou.item(),
        "dice_pixel": dice.item(),
    }


# =========================================================
# 🔹 Pixel-level Semantic Metrics
# =========================================================

def get_semantic_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = False,
    reduction: str = "macro",
) -> Dict[str, float]:

    preds = logits_to_multiclass_mask(logits)

    class_range = range(num_classes)
    if ignore_background:
        class_range = range(1, num_classes)

    tps, fps, fns = [], [], []

    for c in class_range:
        pred_c = (preds == c)
        target_c = (targets == c)

        tp = torch.sum(pred_c & target_c).float()
        fp = torch.sum(pred_c & ~target_c).float()
        fn = torch.sum(~pred_c & target_c).float()

        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    tps = torch.stack(tps)
    fps = torch.stack(fps)
    fns = torch.stack(fns)

    if reduction == "micro":
        tp = tps.sum()
        fp = fps.sum()
        fn = fns.sum()

        iou = tp / (tp + fp + fn + EPS)
        dice = (2 * tp) / (2 * tp + fp + fn + EPS)

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)

    else:
        iou = torch.mean(tps / (tps + fps + fns + EPS))
        dice = torch.mean((2 * tps) / (2 * tps + fps + fns + EPS))

        precision = torch.mean(tps / (tps + fps + EPS))
        recall = torch.mean(tps / (tps + fns + EPS))

    f1 = (2 * precision * recall) / (precision + recall + EPS)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1.item(),
        "iou_pixel": iou.item(),
        "dice_pixel": dice.item(),
    }


# =========================================================
# 🔹 Region-level Metrics (Dice Loss, Intersection, Union, IoU) for Binary and Semantic
# =========================================================

import torch
import torch.nn.functional as F
from typing import Union

def binary_region_metrics(
    logits:torch.Tensor,
    targets:torch.Tensor,
    smooth: Union[int, float] = 1e-7):

    """
    Computes the Dice Loss for binary segmentation.
    Args:
        logits: Tensor of predictions (batch_size, 1, H, W).
        targets: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(logits)
    
    # Calculate intersection and union
    intersection = (pred * targets).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    compute_metrics = {
        "IoU_region": intersection.mean() / union.mean(),
        "DsC_region": 1 - dice.mean()
    }

    # Return Dice Loss
    return compute_metrics

def multiclass_region_metrics(
        logits : torch.Tensor,
        targets : torch.Tensor,
        ignore_background:bool = False,
        smooth : Union[int, float] = 1e-7):
    """
    Computes Dice Loss for multi-class segmentation.
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    pred = F.softmax(logits, dim=1)  # Convert logits to probabilities
    num_classes = pred.shape[1]  # Number of classes (C)
    dice = 0  # Initialize Dice loss accumulator
    iou = 0
    
    if ignore_background:
        classes = range(1, num_classes) # class 0 is background, so we ignore it
    else:
        classes = range(num_classes) 

    compute_metrics = {}

    for c in classes:  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c
        target_c = targets[:, c]  # Ground truth for class c
        
        intersection : torch.Tensor = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union : torch.Tensor = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        iou += intersection / union

        compute_metrics["IoU_class_" + str(c)] = iou
        dice_c = (2. * intersection + smooth) / (union + smooth)
        compute_metrics["DsC_class_"+ str(c)] = dice_c
        dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

    compute_metrics["IoU_region"] = iou / num_classes
    compute_metrics["DsC_region"] = dice / num_classes

    # Return Dice Loss
    return compute_metrics

def get_region_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: Union[int, float] = 1e-7,
    is_binary: bool = True,
    ignore_background: bool = False,
) -> Dict[str, float]:
    """
    Region-level (connected-component) precision, recall, and F1.

    Binary mode  (is_binary=True):
        Applies sigmoid + threshold to logits (B x 1 x H x W), extracts connected
        components, and greedily matches predicted regions to GT regions.

    Semantic mode (is_binary=False):
        Applies softmax(dim=1) → argmax(dim=1) to logits (B x C x H x W), then
        runs the same greedy matching independently for each class and returns
        macro-averaged region metrics.  num_classes is required.

    Args:
        logits:           Raw model outputs (before activation).
        targets:          Binary mask (B x 1 x H x W) or class-index map (B x H x W).
        threshold:        Sigmoid threshold used in binary mode.
        iou_threshold:    Minimum mask-IoU to count a match as TP.
        is_binary:        True → binary mode, False → semantic mode.
        num_classes:      Number of classes; required when is_binary=False.
        ignore_background: When is_binary=False, skip class 0 if True.

    Returns:
        Dict with region_precision, region_recall, region_f1, tp, fp, fn.
    """

    if is_binary:
        binary_metrics = binary_region_metrics(logits=logits,
                                               targets=targets,
                                               smooth=smooth)
        return binary_metrics
    else:
        multiclass_metrics = multiclass_region_metrics(logits=logits,
                                                       targets=targets,
                                                       ignore_background=ignore_background,
                                                       smooth=smooth,
                                                       )
        return multiclass_metrics