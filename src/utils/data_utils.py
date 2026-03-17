from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
from typing import Union

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