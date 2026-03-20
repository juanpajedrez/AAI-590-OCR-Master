'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-18
The following contains the Pytorch dataset classes
'''

# Import all necessary torch modules
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Import Image and path readings
from PIL import Image
from pathlib import Path
import json

# Import typing and get math masks
from typing import List, Union, Tuple
from collections import defaultdict
import math

class TextDetectionDataset(Dataset):
    '''
    Customized class that reads a subset
    Dataset previously extracted,
    and be used for training, validation, and evaluation.
    '''
    def __init__(self,
                data_path:Path,
                split_analyze:str,
                dataset_name:str,
                mask_type:str,
                new_height:int,
                new_width:int,
                transform: Union[transforms.Compose, None]):
        '''
        Args:
            - data_path (Path) : Pathlib path where all the data folder path is located
            - split_analyze (str): String that will determine to get train, val, or test
            - dataset_name (str): String name of the subset dataset.
            - mask_type (str): String of what mask to get the following. binary-text for obtaining the extra -> text bboxes masks.
                semantic-layout obtains the pdf layout bboxes masks.
            - new_height (int): integer for the new H dimension to be resized.
            - new_width (int): integer for the new W dimension to be resized.
            - transform (None or transforms.Compose): torchvision.transforms.Compose that applies vision transformations,
                # NOTE: if set to None, a new transform set with torch.toTensor() + torch.Resize(new_height, new_width) would be done.
                and if you add customized of transform, MAKE SURE new height and new width are set the same dimensions you want in your
                desired transform, otherwise the bboxes would be scaled improperly.
        '''

        super().__init__()
        
        # let's assign the data_path and see if it exists
        assert data_path.exists(), "The data folder doesn't exist, please check that it exists"
        self.data_path = data_path

        # Let's assign the dataset name and see that the subset folder works
        assert isinstance(dataset_name, str), "Add a valid string name for the datasetname"
        self.dataset_name = dataset_name

        self.subset_data_path = data_path / dataset_name
        assert self.subset_data_path.exists(), "Subset data folder doesn't exist, which means it doesn't"

        assert split_analyze in ["train", "val", "test"], "Please type a valud split, either train, val, or test"
        self.split = split_analyze

        assert mask_type in ["binary-text", "semantic-layout"], "Please, write either binary-text or semantic-layout"
        self.mask_type = mask_type

        assert isinstance(new_height, int) and new_height <= 1025, "Assign a valid integer for the new_height"
        self.new_height = new_height

        # Bug fix 4: assertion was checking new_height instead of new_width
        assert isinstance(new_width, int) and new_width <= 1025, "Assign a valid integer for the new_width"
        self.new_width = new_width

        # Let's construct the dictionaroes of DocLayNet_core to read from
        json_split = split_analyze + ".json"
        json_path = self.subset_data_path / "COCO" / json_split
        with open(json_path, "r") as f:
            coco_dict = json.load(f)

        # Assign the JSON extra path (only needed for binary-text mask type)
        self.extra_json_folder_path = self.data_path / self.dataset_name / "JSON"
        if self.mask_type == "binary-text":
            assert self.extra_json_folder_path.exists(), "binary-text mask requested, but JSON folder is missing from dataset, please check"

        # Obtain all neccssary COCO json variables
        self.coco_images = coco_dict["images"]
        self.coco_annotations = coco_dict["annotations"]
        self.coco_categories = coco_dict["categories"]

        # Build a lookup dict for O(1) annotation access per image_id
        self._ann_by_image_id = defaultdict(list)
        for ann in self.coco_annotations:
            self._ann_by_image_id[ann["image_id"]].append(ann)

        # Set the transform
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((new_height, new_width)),
                transforms.ToTensor()
            ])

    def get_mask(self, 
            img_name:str,
            annotations: List[dict[str, Union[int, str]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Instance method that will return the image tensor and its corresponding mask tensor.
        '''

        # Let's instantiate the mask
        if self.mask_type == "semantic-layout":
            mask = torch.zeros((self.new_height, self.new_width), dtype = torch.long)
        else:
            mask = torch.zeros((1, self.new_height, self.new_width), dtype = torch.float32)

        # Let's get the image path directly to the PNG folder
        img_path = self.subset_data_path / "PNG" / img_name
        pil_image = Image.open(img_path).convert("RGB")
        
        # Bug fix 2: PIL .size returns (width, height), not (height, width)
        original_width, original_height = pil_image.size

        # Convert to tensor
        tensor_image = self.transform(pil_image)

        y_scale = self.new_height / original_height
        x_scale = self.new_width / original_width

        # Now let's get the mask
        for ann in annotations:
            x, y, w, h = ann["bbox"] # [x, y, w, h]

            # Now let's scale them properly
            x = x * x_scale
            y = y * y_scale 
            w = w * x_scale
            h = h * y_scale

            # Now let's floor them to remove decimals
            xmin = int(math.floor(x))
            ymin = int(math.floor(y))
            xmax = int(math.ceil(x + w))
            ymax = int(math.ceil(y + h))

            # Bug fix 3: clip against the new (resized) dimensions, not the original ones
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(self.new_width - 1, xmax)
            ymax = min(self.new_height - 1, ymax)

            if self.mask_type == "semantic-layout":
                # Let's get the supercategory id, i.e: Caption, Footnote, etc.
                cat_id = ann["category_id"]
                mask[ymin:ymax+1, xmin:xmax+1] = cat_id
            else:
                mask[0, ymin:ymax+1, xmin:xmax+1] = 1.0

        # Bug fix 1: return was inside the for loop — moved outside so all annotations are drawn
        return tensor_image, mask

    def __len__(self):
        '''
        Returns the total number of samples
        '''
        return len(self.coco_images)

    def __getitem__(self, idx):
        '''
        Returns one sample of data, data and asmk (X, X_mask)
        '''
        # Get the selected image and image id
        sel_coco_image_dict = self.coco_images[idx]
        image_id = sel_coco_image_dict["id"]
        img_name = sel_coco_image_dict["file_name"]

        # Get the annotations
        if self.mask_type == "semantic-layout":
            sel_annotations = self._ann_by_image_id[image_id]

        elif self.mask_type == "binary-text":
            hash_name = sel_coco_image_dict["file_name"].split(".")[0]
            
            # extra_json_name to access inside JSON folder
            extra_json_name = hash_name + ".json"

            with open(self.extra_json_folder_path / extra_json_name, "r") as f:
                extra_dict = json.load(f)
            
            # Select the annotations pertinent to cells where text is located
            sel_annotations = extra_dict["cells"]
        
        # Let's get the image and mask
        sel_img, sel_mask = self.get_mask(
            img_name=img_name,
            annotations=sel_annotations)

        return sel_img, sel_mask, sel_coco_image_dict