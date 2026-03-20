from PIL import Image
from typing import Union, List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from pathlib import Path
import numpy as np
import torch
import json

class MetadataRetriever:
    '''
    Class that will retrieve the most relevant metadata
    features from the train, val, and test.json files coming
    inside the COCO folder of a sub-dataset.
    '''
    def __init__(self,
                 data_path: Path,
                 dataset_name:str,
                 split_analyze:str = "train"):
        '''
        Args:
            - data_path (Path): Pathlib path towards the data folder path
            - dataset_name (str): string name of the dataset name.
        '''
        self.data_path = data_path
        self.dataset_name = dataset_name

        # Let's get the COCO folder
        self.coco_folder_path = self.data_path / self.dataset_name / "COCO"
        assert self.coco_folder_path.exists(), f"the COCO folder inside the dataset named: {self.dataset_name} doesn't exist inside {self.data_path}, please take a look."

        assert split_analyze in ["train", "val", "test"], "Please type a valud split, either train, val, or test"
        self.split = split_analyze

        if split_analyze == "train":
            self.json_to_analyze = "train.json"
        if split_analyze == "test":
            self.json_to_analyze = "test.json"
        if split_analyze == "val":
            self.json_to_analyze = "val.json"
    
    def get_metadata_supercategories(self) -> List[dict[str, Union[int, str]]]:
        '''
        Instance method that will retrieve the available supercategories
        inside the .json file, in the COCO folder. 
        '''
        # Let's get the coco_path and categories key
        json_path = self.coco_folder_path / self.json_to_analyze
        with open(json_path, "r") as f:
            coco_dict = json.load(f)
        return coco_dict["categories"]

    def get_metadata_images(self) -> List[dict[str, Union[int, str]]]:
        '''
        Instance method that will retrieve the available images metadata
        inside the .json file, in the COCO folder. 
        '''
        # Let's get the coco_path and images key
        json_path = self.coco_folder_path / self.json_to_analyze
        with open(json_path, "r") as f:
            coco_dict = json.load(f)
        return coco_dict["images"]

    def get_metadata_annotations(self) -> List[dict[str, Union[int, str]]]:
        '''
        Instance method that will retrieve the available annotations metadata
        inside the .json file, in the COCO folder. 
        '''
        # Let's get the coco_path and annotations key
        json_path = self.coco_folder_path / self.json_to_analyze
        with open(json_path, "r") as f:
            coco_dict = json.load(f)
        return coco_dict["annotations"]

    def get_metadata_spec_image_id(self, img_id:int) ->Tuple[List[dict[str, Union[int, str]]], List[dict[str, Union[int, str]]]]:
        '''
        Instance method that will be used to retrieve the metadata
        from the core folders, these are inside COCO folder.
        Arg:
            - img_id (int): Integer that will be used to retrieve the specific image
        '''

        # Retrieve the images_dict and annotations
        images_dict = self.get_metadata_images()
        annotations_dict = self.get_metadata_annotations()

        # Let's check wether the image_id exists, else assert that please add a valid one
        available_image_ids = [x["id"] for x in images_dict]
        assert img_id in available_image_ids, f"Passed image id -> {img_id} does not exist inside {self.json_to_analyze}, inside {self.dataset_name} dataset. double check"
        
        # Select the image dictionary correspondent to the selected image id
        sel_image_dict = [img_dict for img_dict in images_dict if img_dict["id"] == img_id]

        # Select the annotations correspondent to the selected image id
        sel_annotations_dict = [annotation for annotation in annotations_dict if annotation["image_id"] == img_id]
        
        return sel_image_dict, sel_annotations_dict

    def get_metadata_extra_image_id(self, img_id:int) -> Tuple[List[dict[str, Union[int, str]]], List[dict[str, Union[int, str]]]]:
        '''
        Instance method that will retrieve the metadata from the extra
        folders, these are inside JSON folder.
        '''
        extra_json_folder_path = self.data_path / self.dataset_name / "JSON"
        assert extra_json_folder_path.exists(), f"JSON folder doesnt exist inside {self.dataset_name}"

        # Retrieve the images_dict and annotations
        images_dict = self.get_metadata_images()  

        # Let's check wether the image_id exists, else assert that please add a valid one
        available_image_ids = [x["id"] for x in images_dict]
        assert img_id in available_image_ids, f"Passed image id -> {img_id} does not exist inside {self.json_to_analyze}, inside {self.dataset_name} dataset. double check"

        # Select the image dictionary correspondent to the selected image id
        sel_image_dict = [img_dict for img_dict in images_dict if img_dict["id"] == img_id]
        hash_name :str = sel_image_dict[0]["file_name"].split(".")[0]

        # extra_json_name to access inside JSON folder
        extra_json_name = hash_name + ".json"

        with open(extra_json_folder_path / extra_json_name, "r") as f:
            extra_dict = json.load(f)
        
        return extra_dict["metadata"], extra_dict["cells"]

def plot_image_with_annotations_bbox(image_path, annotations, core:bool = True):
    """Plot image with bounding boxes from annotations."""

    # Load image
    image = Image.open(image_path)
    print(image.size)
    print("Image range values:", image.getextrema())
    fig, ax = plt.subplots(1, figsize = (12, 12))
    ax.imshow(image)

    # Add bounding boxes
    for ann in annotations:
        bbox = ann["bbox"]
        # COCO format: [x_min, y_min, width, height]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if core:
            ax.text(bbox[0], bbox[1] - 10, ann["category_id"], color='blue', fontsize=12, weight='bold')
    ax.axis("off")

    plt.show()

def plot_image_with_annotations_segmentation(image_path, annotations, core:bool = True):
    """Plot image with segmentation polygons from annotations."""

    # Load image
    image = Image.open(image_path)
    print(image.size)
    print("Image range values:", image.getextrema())

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    # Add segmentation polygons
    for ann in annotations:
        segmentations = ann["segmentation"]

        # COCO can have multiple polygons per object
        for seg in segmentations:
            # Convert flat list → (N, 2) array
            poly = np.array(seg).reshape(-1, 2)

            polygon = patches.Polygon(
                poly,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(polygon)

        # Optional: label using first point of first polygon
        if core:
            if len(segmentations) > 0:
                first_poly = np.array(segmentations[0]).reshape(-1, 2)
                x, y = first_poly[0]
                ax.text(x, y - 10, ann["category_id"], color='blue', fontsize=12, weight='bold')

    ax.axis("off")
    plt.show()

def plot_tensor_with_annotations_bbox(image_tensor:torch.Tensor, annotations, core:bool = True):
    """Plot image with bounding boxes from annotations."""

    fig, ax = plt.subplots(1, figsize = (12, 12))
    ax.imshow(image_tensor.permute(1, 2, 0).cpu()) # Permute from C * H * W -> H * W * C

    # Add bounding boxes
    for ann in annotations:
        bbox = ann["bbox"]
        # COCO format: [x_min, y_min, width, height]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if core:
            ax.text(bbox[0], bbox[1] - 10, ann["category_id"], color='blue', fontsize=12, weight='bold')
    ax.axis("off")

    plt.show()

def load_image(
    img_path: Path,
    annotations: List[Dict[str, Union[int, float, str]]],
    new_height: Union[int, None] = None,
    new_width: Union[int, None] = None,
    core: bool = True
) -> Tuple[torch.Tensor, List[Dict[str, Union[int, float, str]]]]:
    """
    Loads an image, converts to tensor, optionally resizes it, and rescales bounding boxes.
    bbox format assumed: [x, y, width, height]
    """

    # Load image
    pil_image = Image.open(img_path).convert("RGB")

    # Convert to tensor
    tensor_image = transforms.ToTensor()(pil_image)  # (C, H, W)

    original_height, original_width = tensor_image.shape[1], tensor_image.shape[2]

    # If resizing is requested
    if new_height is not None and new_width is not None:

        # Compute scaling factors
        y_scale = new_height / original_height
        x_scale = new_width / original_width

        # Resize image
        resize_transform = transforms.Resize((new_height, new_width))
        tensor_image = resize_transform(tensor_image)

        # Update bounding boxes
        updated_annotations = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, w, h]

            x, y, w, h = bbox

            new_bbox = [
                x * x_scale,
                y * y_scale,
                w * x_scale,
                h * y_scale
            ]

            updated_ann = ann.copy()
            updated_ann["bbox"] = new_bbox
            updated_annotations.append(updated_ann)

    else:
        # No resizing → keep annotations as is
        updated_annotations = annotations

    # Optional plotting
    plot_tensor_with_annotations_bbox(
        image_tensor=tensor_image,
        annotations=updated_annotations,
        core=core
    )

    return tensor_image, updated_annotations