'''
Author: Juan Pablo Triana Martinez 
Date: 2026-03-17
Script that will download the zip files of DocLayNet_core and DocLayNet_zip files
from the following links:

"core": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
"extra": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip"
'''

import sys
from pathlib import Path
import os
import argparse

# Add the string directly of AAI-590-OCR-Master for proper src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all data_utils.
from src.utils import data_utils

if __name__ == "__main__":
    ''' 
    Let's add 4 argument:
        - core_download : bool -> default True. -> flag to download core data.
        - extra_download: bool -> default True. -> flag to download zip data.
        - core_extract: bool -> default False. -> flag to extract ALL core data from .zip.
        - extra_extract: bool -> default False. -> flag to extract ALL extra data from .zip.
    '''

    # Let's get an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-crd",
                        "--core_download",
                        type = bool,
                        default = True,
                        help = "flag to download core data")
    parser.add_argument("-exd",
                    "--extra_download",
                    type = bool,
                    default = True,
                    help = "flag to download zip data")
    parser.add_argument("-cre",
                    "--core_extract",
                    type = bool,
                    default = False,
                    help = "flag to extract core data")
    parser.add_argument("-exe",
                    "--extra_extract",
                    type = bool,
                    default = False,
                    help = "flag to extract zip data")
    
    args = parser.parse_args()

    # Setup the data folder path
    data_folder_path = Path().cwd() / "data"

    # Read a dictionary containing both the links
    internet_links = {
        "core": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
        "extra": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip"
    }

    if args.core_download:
        doclaynet_core_path = data_utils.download_raw_data(
        data_folder_path=data_folder_path,
        filename_link=internet_links["core"])
    else:
        doclaynet_core_path = None
    
    if args.extra_download:
        doclaynet_extra_path = data_utils.download_raw_data(
        data_folder_path=data_folder_path,
        filename_link=internet_links["extra"])
    else:
        doclaynet_extra_path = None
    
    if args.core_extract:
        doclaynet_core_extracted_path = data_utils.extract_raw_data(
        data_folder_path=data_folder_path,
        zip_file_data_path=doclaynet_core_path)
    
    if args.extra_extract:
        doclaynet_extra_extracted_path = data_utils.extract_raw_data(
        data_folder_path = data_folder_path,
        zip_file_data_path = doclaynet_extra_path
)