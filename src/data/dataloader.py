'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-18
The following contains the PyTorch Dataloader classes for OCR
Text Detection -> linknet-resnet model
Text Recognition -> ViTSTR Model
'''

from pathlib import Path
from typing import Union
from torchvision import transforms
from .dataset import TextDetectionDataset
from torch.utils.data import DataLoader

def get_dataloaders_text_detection(
        data_path:Path,
        batch_size:int,
        dataset_name:str,
        mask_type:str,
        num_workers:int = 0,
        pin_memory: bool = False,
        new_height: int = 1024,
        new_width: int = 1024,
        transform : Union[transforms.Compose, None] = None):
    '''
    Args:
        - data_path (Path): Pathlib path where data folder is located
        - batch_size (int): Number of samples per batch in the Dataloaders
        - dataset_name (str): Name of the subset dataset that will be used for all training.
        - mask_type (str): = "binary-text",
        - num_workers (int) = 0: Number of workers per Dataloader, use os.cpu_count() when adding them
        - pin_memory (bool) = False: Boolean flag to pin torch.Tensors inside the batches to specific memory, normally set to False.
        - new_height (int) = 1024: Integer value that sets the new dimensions of the new height, default is 1024
        - new_width (int) = 1024: Integer value that sets the new dimensions of the new width, default is 1024
        - transform (Union[torchvision.Transforms.Compose, None]) =None : Parameters to pass a torchvision.transform class or None, automatically sets it inside with the torch.utils.Dataset
    '''
    # Let's get all train, val, and test datasets
    train_detection_dataset = TextDetectionDataset(data_path = data_path,
                split_analyze = "train",
                dataset_name = dataset_name,
                mask_type = mask_type,
                new_height = new_height,
                new_width = new_width,
                transform=transform)

    val_detection_dataset = TextDetectionDataset(data_path = data_path,
                split_analyze = "val",
                dataset_name = dataset_name,
                mask_type = mask_type,
                new_height = new_height,
                new_width = new_width,
                transform=transform)

    test_detection_dataset = TextDetectionDataset(data_path = data_path,
                split_analyze = "test",
                dataset_name = dataset_name,
                mask_type = mask_type,
                new_height = new_height,
                new_width = new_width,
                transform=transform)
    
    # Let's get the train, val, and test dataloaders
    train_dataloader = DataLoader(dataset=train_detection_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    # Let's create the DataLoader
    val_dataloader = DataLoader(dataset=val_detection_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    # Let's create the DataLoader
    test_dataloader = DataLoader(dataset=test_detection_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_dataloader, val_dataloader, test_dataloader