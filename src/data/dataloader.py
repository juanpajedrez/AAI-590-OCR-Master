'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-18 (updated 2026-04-01)
The following contains the PyTorch Dataloader classes for OCR
Text Detection -> linknet-resnet model
Text Recognition -> ViTSTR Model
'''

from pathlib import Path
from typing import Union, Tuple, List
from torchvision import transforms
from .dataset import TextDetectionDataset
from torch.utils.data import DataLoader


def get_dataloaders_text_detection(
        data_path: Path,
        batch_size: int,
        dataset_name: str,
        mask_type: str,
        num_workers: int = 0,
        pin_memory: bool = False,
        new_height: int = 1024,
        new_width: int = 1024,
        train_transform: Union[transforms.Compose, None] = None,
        val_transform: Union[transforms.Compose, None] = None,
        test_transform: Union[transforms.Compose, None] = None,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Build train, val, and test DataLoaders for text-detection tasks.

    Args:
        data_path (Path): Pathlib path where data folder is located.
        batch_size (int): Number of samples per batch.
        dataset_name (str): Name of the subset dataset folder.
        mask_type (str): "binary-text" or "semantic-layout".
        num_workers (int): DataLoader worker processes (default: 0).
        pin_memory (bool): Pin tensors to memory (default: False).
        new_height (int): Target image height after resize (default: 1024).
        new_width (int): Target image width after resize (default: 1024).

        train_transform (Compose | None):
            Transform applied to training images. When None an augmented
            default is built automatically:
                Resize → ColorJitter → ToTensor → Normalize
            where mean/std are computed from the training split to avoid
            data leakage.

        val_transform (Compose | None):
            Transform applied to validation images. When None a clean
            inference default is built:
                Resize → ToTensor → Normalize(train_mean, train_std)

        test_transform (Compose | None):
            Transform applied to test images. Same default as val_transform.

        jitter_brightness/contrast/saturation/hue (float):
            ColorJitter parameters used only when train_transform is None.
            Defaults: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1.

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader).
    '''

    # ------------------------------------------------------------------ #
    # Build missing transforms                                             #
    # ------------------------------------------------------------------ #
    if train_transform is None or val_transform is None or test_transform is None:
        # Lazy import to avoid loading heavy dependencies when transforms
        # are provided explicitly by the caller.
        from ..utils.data_utils import compute_train_mean_std

        print("[INFO] Computing per-channel mean and std from training split...")
        mean, std = compute_train_mean_std(
            data_path=data_path,
            dataset_name=dataset_name,
            new_height=new_height,
            new_width=new_width,
        )
        print(f"[INFO] Training mean : {[round(m, 4) for m in mean]}")
        print(f"[INFO] Training std  : {[round(s, 4) for s in std]}")

        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize((new_height, new_width)),
                transforms.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        # Val and test share the same no-augmentation pipeline
        inference_transform = transforms.Compose([
            transforms.Resize((new_height, new_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if val_transform is None:
            val_transform = inference_transform
        if test_transform is None:
            test_transform = inference_transform

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    train_detection_dataset = TextDetectionDataset(
        data_path=data_path,
        split_analyze="train",
        dataset_name=dataset_name,
        mask_type=mask_type,
        new_height=new_height,
        new_width=new_width,
        transform=train_transform,
    )

    val_detection_dataset = TextDetectionDataset(
        data_path=data_path,
        split_analyze="val",
        dataset_name=dataset_name,
        mask_type=mask_type,
        new_height=new_height,
        new_width=new_width,
        transform=val_transform,
    )

    test_detection_dataset = TextDetectionDataset(
        data_path=data_path,
        split_analyze="test",
        dataset_name=dataset_name,
        mask_type=mask_type,
        new_height=new_height,
        new_width=new_width,
        transform=test_transform,
    )

    # ------------------------------------------------------------------ #
    # DataLoaders                                                          #
    # ------------------------------------------------------------------ #
    train_dataloader = DataLoader(
        dataset=train_detection_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        dataset=val_detection_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dataloader = DataLoader(
        dataset=test_detection_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader, test_dataloader
