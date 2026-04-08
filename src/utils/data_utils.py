from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
from typing import Union, List, Tuple
import random
import json
from datetime import datetime

def download_raw_data(
    data_folder_path: Path,
    filename_link:str,
    chunk_size:int = 16) -> Path:
    '''
    The following function will download the raw data from a privided link and save it in the
    specified data_folder_path
    Args:
        data_folder_path (Path): Path where the raw data will be saved.
        filename_link (str): The link to the raw data file.
    '''

    # Create the data_folder_path if it doesn't exist
    data_folder_path.mkdir(parents=True, exist_ok=True)

    # Let's create a raw folder inside the data folder
    raw_folder_path = data_folder_path / "raw"
    raw_folder_path.mkdir(parents=True, exist_ok=True)

    # Download the file and save it in the raw folder
    file_name = filename_link.split("/")[-1] # Select the last split, filename
    file_path = raw_folder_path / file_name

    if file_path.exists():
        print(f"[INFO] File already exists: {file_path}")
        return file_path
    else:
        print(f"[INFO] Downloading raw data from: {filename_link}")

        #Stream Donbwload the file
        with requests.get(filename_link, stream=True) as response:
            response.raise_for_status()

            # Find the total file size and chunk size
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = chunk_size * 1024 * 1024 # 16 MB

            # Open the file and write the chunks to it
            with open(file_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=file_name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(f"[INFO] Raw data downloaded at: {file_path}")
    return file_path

def extract_raw_data(
    data_folder_path:Path,
    zip_file_data_path: Union[None, Path]) -> Path:
    '''
    The following function will extract the raw data from a zip file
    into a specified data folder path
    Args:
        data_folder_path (Path): Path where the extracted data will be saved.
        zip_file_data_path (Path or None): Path where the raw data zip file is located.
        if None, then returns message there is no zip_file_data_path assigned
    '''

    if zip_file_data_path is not None:
        # Create a folder for extracted data
        extracted_data_path = data_folder_path / "extracted"
        extracted_data_path.mkdir(parents = True, exist_ok=True)

        # Let's obtain the zip local file name
        file_name = zip_file_data_path.stem

        # Let's now extract the data from zipfile
        with zipfile.ZipFile(zip_file_data_path, "r") as zip_ref:
            s = zip_ref.infolist()

            # Iterate over the zip file contents and extract them
            for member in tqdm(s, desc="Extracting files", total = (len(s))):
                output_path = extracted_data_path / file_name / member.filename
                if output_path.exists():
                    continue
                zip_ref.extract(member, extracted_data_path/ file_name)
        return extracted_data_path / file_name
    else:
        print(f"[INFO] .zip file has been set to None, check for existence inside {data_folder_path}")

class ObtainSubSample:
    '''
    This class will perform the following:
        1. Class will read from zip file the `.json` train, val, and test from `raw/DocLayNet_core.zip/COCO`.
        2. Class will proceed depending on the set `n_samples_train`, `n_samples_val`, and `n_samples_test`, to extract with a specified `seed`, the correspoding `hash` keys.
        3. Class will then save these `train_hash_keys`, `val_hash_keys`, and `test_hash_keys`, and stream download ONLY those from `raw/DocLayNet_core.zip/COCO`.
        4. Class will create a `metadata.txt` file containing all of the previous information, and any additional detail. 
    '''
    def __init__(self,
                data_folder_path:Path,
                n_samples_train:int,
                n_samples_val:int,
                n_samples_test:int,
                subsample_name:str,
                local_doclaynet_zip_core_path: Path,
                local_doclaynet_zip_extra_path: Union[Path, None],
                added_extra:bool = True,
                seed:int = 42):
        '''
        Args:
            - data_folder_path (Path): Pathlib Parameter where data folder is located.
            - n_samples_train (int): Parameter that sets the total samples retrieved from train dataset, found inside `train.json` of `DocLayNet_core/COCO`.
            - n_samples_val (int): Parameter that sets the total samples retrieved from train dataset, found inside `val.json` of `DocLayNet_core/COCO`.
            - n_samples_test (int): Parameter that sets the total samples retrieved from train dataset, found inside `test.json` of `DocLayNet_core/COCO`.
            - subsample_name (str): Parameter, which will name the folder containing the following.
            - local_doclaynet_zip_core_path (Path): Pathlib parameters where DocLayNet_core.zip file local is located.
            - local_doclaynet_zip_extra_path (Path or None): Pathlib parameter where where DocLayNet_extra.zip file local is located, if None, added_extra turns False.
            - added_extra (bool) = True parameter, which will tell wether to add the extra files with folders: `JSON` and `PDF`.
            - seed (int): Integer parameter that sets the random sampling from the machine.
        
        Key different functionalities:
            - Constructor will take the `local_doclaynet_zip_core_path: Path` parameter, pointing to where the `DocLayNet_core` path is located.
            - Constructor will take the `local_doclaynet_zip_extra_path: Union[Path, None]` parameter, pointing to where the `DocLayNet_extra` path is located. if `added_extra` is False, this would be ignored.
        '''
        # Assertions that data_folder_path and local_doclaynet_zip_core_path are Paths
        assert isinstance(data_folder_path, Path), "Please, add an appropiate data folder pathlib path"
        assert isinstance(local_doclaynet_zip_core_path, Path), "Please, add an appropiate DocLayNet_zip core pathlib path"
        
        # Assertions to make sure that all paths exist, else raise a message that they are not existent
        assert data_folder_path.exists(), f"{data_folder_path} doesn't exist, please check your data folder structure"
        assert local_doclaynet_zip_core_path.exists(), f"{local_doclaynet_zip_core_path} doesn't exist, please check your data folder structure"
        assert local_doclaynet_zip_extra_path.exists(), f"{local_doclaynet_zip_extra_path} doesn't exist, please check your data folder structure"

        # Let's add some parameters
        self.data_folder_path = data_folder_path
        self.subsample_name = subsample_name + "_seed_" + str(seed)
        self.added_extra = added_extra
        self.seed = seed

        ### ============================== NOTE ========================== ###
        ### This is gonna be used with ChatGPT generated code to integrate moving files
        # Set the paths
        self.local_doclaynet_zip_core_path = local_doclaynet_zip_core_path
        self.local_doclaynet_zip_extra_path = local_doclaynet_zip_extra_path
        ### =============================================================== ###

        # Set the random seed
        self.seed = seed
        random.seed(seed)

        # Set the subsample directories
        self.set_subsample_directories()

        # Get the coco_files
        self.coco_files = self.get_coco_files()

        # Let's obtain the max_number_train_samples
        self.max_num_train_samples = len(self.coco_files["train.json"]["images"])
        self.max_num_val_samples = len(self.coco_files["val.json"]["images"])
        self.max_num_test_samples = len(self.coco_files["test.json"]["images"])

        # Double check that they are set properly
        self.n_samples_train = self.set_num_samples(n_samples_train, self.max_num_train_samples)
        self.n_samples_val = self.set_num_samples(n_samples_val, self.max_num_val_samples)
        self.n_samples_test = self.set_num_samples(n_samples_test, self.max_num_test_samples)

        # Get the coco_key_dictionary containing all json files information
        self.json_coco_dict = self.get_subsample_coco_files_dict()
        self.set_subsample_coco_files()

    def set_subsample_directories(self):
        '''
        Public instance method that sets the directories where all
        the new subsample of data will be located.
        '''
        # Let's create a data folder to
        self.extracted_data_path = self.data_folder_path / self.subsample_name
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        # Let's create the directories
        self.coco_path = self.extracted_data_path / "COCO"
        self.coco_path.mkdir(parents=True, exist_ok=True)
        self.png_path = self.extracted_data_path / "PNG"
        self.png_path.mkdir(parents=True, exist_ok=True)
        if self.added_extra:
            self.json_extra_path = self.extracted_data_path / "JSON"
            self.json_extra_path.mkdir(parents=True, exist_ok=True)
            self.pdf_extra_path = self.extracted_data_path / "PDF"
            self.pdf_extra_path.mkdir(parents=True, exist_ok=True)
    
    def get_coco_files(self) -> dict[str, dict[str, str]]:
        '''
        Public instance method that gets the .json train, val, and test
        json files and retrieves dictionaries
        '''
        # Let's now extract the json files from the .zip core
        with zipfile.ZipFile(self.local_doclaynet_zip_core_path, "r") as zip_ref:
            s = zip_ref.infolist()

            # Iterate over the zip file contents and extract them
            coco_files = {}
            for member in s:
                if ".json" in member.filename:
                    print(f"[INFO] Extracting {member.filename.split("/")[-1]} for {self.subsample_name}...")
                    with zip_ref.open(member) as f:
                        coco = json.load(f)
                        coco_files[member.filename.split("/")[-1]] = coco
                    print(f"[INFO] Extracted {member.filename.split("/")[-1]} for {self.subsample_name}")
                else:
                    continue
        return coco_files

    def set_num_samples(self, n_samples_passed:int, max_num_passed:int) -> int:
        '''
        Public Instance method that, depending on the comparison 
        between n_samples_passed and max_num_passed, perform:
        - if n_samples_passed, return an error that it must be greater than 0
        - if n_samples_passed <= max_num_passed, n_samples passed stays the same
        - otherwise n_samples_passed is set to to max_num_passed
        '''
        assert n_samples_passed > 0, f"Please add a valid positive value of n_samples_passed"
        if n_samples_passed <= max_num_passed:
            return n_samples_passed
        else:
            return min(n_samples_passed, max_num_passed)
    
    def get_subsample_coco_files_dict(self) -> dict[str, dict[str, List[dict]]]:
        '''
        Public instanec method that returns the json files
        in a dictionary format to be uploaded to the new 
        ''' 

        test_coco_holder = {}

        # Let's iterate acorss each key
        for key in self.coco_files.keys():
            # We are creating a new dictionary notebook 
            json_coco_dict = {}

            # Let's add the categories as we know
            print(f"[INFO] Getting categories to: {key}...")
            json_coco_dict["categories"] = self.coco_files[key]["categories"]

            # Let's now sample according to if its train, val, or test
            if "train" in key:
                print(f"[INFO] Getting subsampled images, size: {self.n_samples_train}, to: {key}...")
                images_sampled = random.sample(self.coco_files[key]["images"], self.n_samples_train)
            elif "val" in key:
                print(f"[INFO] Getting subsampled images, size: {self.n_samples_val}, to: {key}...")
                images_sampled = random.sample(self.coco_files[key]["images"], self.n_samples_val)
            elif "test" in key:
                print(f"[INFO] Getting subsampled images, size: {self.n_samples_test}, to: {key}...")
                images_sampled = random.sample(self.coco_files[key]["images"], self.n_samples_test)
            else:
                print(f"[INFO] Getting entire image set, size: {len(self.coco_files[key]["images"])}, to: {key}...")
                images_sampled = self.coco_files[key]["images"]
            
            # Assign them properly
            json_coco_dict["images"] = images_sampled
            print(f"[INFO] Obtained subsampled complete!")

            # Obtain selected image ids inside a set
            print(f"[INFO] Getting annotations from subsample...")
            sel_images_sampled_ids = set([x["id"] for x in images_sampled])

            # Retrieve the annotations_sampled from sel_images_sampled_ids
            annotations_sampled = [x for x in self.coco_files[key]["annotations"] if x["image_id"] in sel_images_sampled_ids]

            # Add them properly
            json_coco_dict["annotations"] = annotations_sampled
            print(f"[INFO] Obtained annotations!")

            # Append them to test_coco_holder
            test_coco_holder[key] = json_coco_dict
        
        return test_coco_holder

    def set_subsample_coco_files(self):
        '''
        Public instance method that will now write the new .json files inside the folder
        '''
        for coco_filename in self.coco_files.keys():
            # Output path for new filename
            output_path = self.coco_path / coco_filename

            # Data selection
            sel_data = self.json_coco_dict[coco_filename]

            # Now data upload
            with open(output_path, "w") as f:
                print(f"[INFO] Adding subsample .json file, {coco_filename}..")
                json.dump(sel_data, f, indent = 4)
                print(f"[INFO] Added subsample .json file, {coco_filename}!")

    def extract_core_files(self):
        '''
        Public instance method that will now extract the following:
        1. from DocLayNet_core.zip path, extract the PNG files
        '''

        # Let's get the hash keys that are from the subsample
        for key in self.json_coco_dict:
            
            # Let's get the data from specific key (i.e: train.json, val.json, test.json)
            data_sel = self.json_coco_dict[key]

            # Let's get the sample hashes
            subsample_hashes = set([x["file_name"].replace(".png", "") for x in data_sel["images"]])

            # Let's do first the Doclaynet_core.zip to get the .png files
            print(f"[INFO]: Extracting core files...")
            with zipfile.ZipFile(self.local_doclaynet_zip_core_path) as zip_core:
                s = zip_core.infolist()

                for member in tqdm(s, desc=f"Extracting subsample", total = len(s)):
                    if "__MACOSX" in member.filename:
                        continue
                    if member.is_dir():
                        continue
                    if ".png" in member.filename:             
                        # Get the filename inside zip without path
                        file_name = member.filename.split("/")[-1].replace(".png", "")

                        # if file_name is inside the subsample_hashes, write -> target from DoclayNet_core.zip
                        if file_name in subsample_hashes:
                            download_filename = file_name + ".png"
                            output_path = self.png_path / download_filename

                            # If previously downloaded, continue
                            if output_path.exists():
                                continue
                            with zip_core.open(member) as source, open(output_path, "wb") as target:
                                target.write(source.read())
            print(f"[INFO]: Core files extracted!")

    def extract_extra_files(self):
        '''
        Public instance method that will now extract the following:
        1. from DocLayNet_core.zip path, extract the PNG files
        '''

        if self.added_extra:
            # Let's get the hash keys that are from the subsample
            for key in self.json_coco_dict:
                
                # Let's get the data from specific key (i.e: train.json, val.json, test.json)
                data_sel = self.json_coco_dict[key]

                # Let's get the sample hashes
                subsample_hashes = set([x["file_name"].replace(".png", "") for x in data_sel["images"]])

                # Let's do first the Doclaynet_extra.zip to get the .png files
                print(f"[INFO]: Extracting extra files...")
                with zipfile.ZipFile(self.local_doclaynet_zip_extra_path) as zip_core:
                    s = zip_core.infolist()

                    for member in tqdm(s, desc=f"Extracting subsample", total = len(s)):
                        if "__MACOSX" in member.filename:
                            continue

                        if member.is_dir():
                            continue

                        if ".json" in member.filename:             
                            # Get the filename inside zip without path
                            file_name = member.filename.split("/")[-1].replace(".json", "")

                            # if file_name is inside the subsample_hashes, write -> target from DoclayNet_core.zip
                            if file_name in subsample_hashes:
                                download_filename = file_name + ".json"
                                output_path = self.json_extra_path / download_filename

                                # If previously downloaded, continue
                                if output_path.exists():
                                    continue
                                with zip_core.open(member) as source, open(output_path, "wb") as target:
                                    target.write(source.read())

                        if ".pdf" in member.filename:             
                            # Get the filename inside zip without path
                            file_name = member.filename.split("/")[-1].replace(".pdf", "")

                            # if file_name is inside the subsample_hashes, write -> target from DoclayNet_core.zip
                            if file_name in subsample_hashes:
                                download_filename = file_name + ".pdf"
                                output_path = self.pdf_extra_path / download_filename

                                # If previously downloaded, continue
                                if output_path.exists():
                                    continue
                                with zip_core.open(member) as source, open(output_path, "wb") as target:
                                    target.write(source.read())
                print(f"[INFO]: extra files extracted!")
        else:
            print(f"[INFO]: Extra files is set to False -> {self.added_extra}")

    def write_metadata(self):
        '''
        Public instance method that writes a metadata.txt file
        containing information about the subsample creation.
        '''

        metadata_path = self.extracted_data_path / "metadata.txt"

        # Count extracted files
        num_png = len(list(self.png_path.glob("*.png")))

        num_json_extra = 0
        num_pdf_extra = 0

        if self.added_extra:
            num_json_extra = len(list(self.json_extra_path.glob("*.json")))
            num_pdf_extra = len(list(self.pdf_extra_path.glob("*.pdf")))

        # Metadata text
        metadata_text = f"""
        ==============================
        DocLayNet Subsample Metadata
        ==============================

        Subsample Name:
        {self.subsample_name}

        Creation Timestamp:
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Random Seed:
        {self.seed}

        ------------------------------
        Dataset Sizes
        ------------------------------

        Max Train Samples: {self.max_num_train_samples}
        Max Val Samples: {self.max_num_val_samples}
        Max Test Samples: {self.max_num_test_samples}

        Selected Train Samples: {self.n_samples_train}
        Selected Val Samples: {self.n_samples_val}
        Selected Test Samples: {self.n_samples_test}

        ------------------------------
        Files Extracted
        ------------------------------

        PNG Images Extracted: {num_png}

        Extra Files Enabled: {self.added_extra}

        JSON Extra Files Extracted: {num_json_extra}
        PDF Extra Files Extracted: {num_pdf_extra}

        ------------------------------
        Dataset Paths
        ------------------------------

        Core Dataset Zip:
        {self.local_doclaynet_zip_core_path}

        Extra Dataset Zip:
        {self.local_doclaynet_zip_extra_path}

        Output Dataset Directory:
        {self.extracted_data_path}

        ------------------------------
        Directory Structure
        ------------------------------

        COCO Annotations: {self.coco_path}
        PNG Images: {self.png_path}
        """

        if self.added_extra:
            metadata_text += f"""
            Extra JSON Files: {self.json_extra_path}
            Extra PDF Files: {self.pdf_extra_path}
            """

        # Write file
        with open(metadata_path, "w") as f:
            print("[INFO] Writing metadata file...")
            f.write(metadata_text.strip())

        print(f"[INFO] Metadata file written to {metadata_path}")


def compute_train_mean_std(
    data_path: Path,
    dataset_name: str,
    new_height: int,
    new_width: int,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std from the training split of a subset dataset.
    Only training images are used to avoid data leakage into val/test.

    Uses a single memory-efficient pass over the training images:
    no full dataset is loaded into RAM at once.

    Args:
        data_path:    Path to the project data folder.
        dataset_name: Name of the subset dataset folder inside data_path.
        new_height:   Target image height (must match the resize used at training time).
        new_width:    Target image width  (must match the resize used at training time).

    Returns:
        mean: List[float] of length 3 (RGB channels), values in [0, 1].
        std:  List[float] of length 3 (RGB channels), values in [0, 1].
    """
    import torch
    from PIL import Image
    from torchvision import transforms as T

    json_path = data_path / dataset_name / "COCO" / "train.json"
    with open(json_path, "r") as f:
        coco_dict = json.load(f)

    png_path = data_path / dataset_name / "PNG"

    to_tensor = T.Compose([
        T.Resize((new_height, new_width)),
        T.ToTensor(),
    ])

    # Online single-pass: accumulate sum and sum-of-squares per channel
    n_pixels = 0
    running_sum    = torch.zeros(3)
    running_sum_sq = torch.zeros(3)

    for img_info in tqdm(coco_dict["images"], desc=f"[compute_train_mean_std] scanning {dataset_name}/train"):
        pil_img = Image.open(png_path / img_info["file_name"]).convert("RGB")
        t = to_tensor(pil_img)          # (3, H, W), values in [0, 1]
        n = t.shape[1] * t.shape[2]     # pixels per image
        running_sum    += t.sum(dim=[1, 2])
        running_sum_sq += (t ** 2).sum(dim=[1, 2])
        n_pixels += n

    mean = running_sum / n_pixels
    # clamp prevents tiny negative variance from floating-point rounding
    std  = torch.sqrt(torch.clamp(running_sum_sq / n_pixels - mean ** 2, min=0))

    return mean.tolist(), std.tolist()
