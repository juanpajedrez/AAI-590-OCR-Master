'''
Author: Juan Pablo Triana Martinez 
Date: 2026-03-17
Script that, with downloaded DocLayNet_core and DocLayNet_extra .zip files
target -> write specified subsample of data
'''

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from data_utils de subsampler class
from src.utils.data_utils import ObtainSubSample

if __name__ == "__main__":
    # Setup the data folder path
    data_folder_path = Path().cwd() / "data"

    # Obtain the DocLayNet_core and DocLayNet_extra .zip default paths
    default_doclaynet_zip_core_path = data_folder_path / "raw" / "DocLayNet_core.zip"
    default_doclaynet_zip_extra_path = data_folder_path / "raw" / "DocLayNet_extra.zip"

    # Let's get an argument parser for each argument
    parser = argparse.ArgumentParser()

    # Add parser arguments for subsampling metadata
    parser.add_argument("--n_samples_train",
                        type=int,
                        default=1000,
                        help="number of train samples to sample from .zip")
    parser.add_argument("--n_samples_val",
                        type=int,
                        default=250,
                        help="number of val samples to sample from .zip")
    parser.add_argument("--n_samples_test",
                        type=int,
                        default=100,
                        help="number of test samples to sample from .zip")
    parser.add_argument("--subsample_name",
                    type=str,
                    default="test_subsample",
                    help="string name for subsampled dataset")
    parser.add_argument("--seed",
                        type = int,
                        default=42,
                        help = "computer random seed for setting system random sampling randomness")
    parser.add_argument("--extra",
                        type=bool,
                        default=True,
                        help = "boolean flag to subsample also from DocLayNet_extra.zip")

    # Add parser arguments for paths
    parser.add_argument("-dp",
                        "--data_path",
                        type =  str,
                        default=str(data_folder_path),
                        help="string to data folder path")
    parser.add_argument("--zip_core_path",
                        type =  str,
                        default=str(default_doclaynet_zip_core_path),
                        help="string to zip core path")
    parser.add_argument("--zip_extra_path",
                        type =  str,
                        default=str(default_doclaynet_zip_extra_path),
                        help="string to zip extra path")

    # obtain arguments
    args = parser.parse_args()

    subsampler = ObtainSubSample(
        data_folder_path=Path(args.data_path),
        n_samples_train=args.n_samples_train,
        n_samples_val=args.n_samples_val,
        n_samples_test=args.n_samples_test,
        subsample_name=args.subsample_name,
        local_doclaynet_zip_core_path=Path(args.zip_core_path),
        local_doclaynet_zip_extra_path=Path(args.zip_extra_path),
        added_extra=args.extra,
        seed=args.seed)
    
    # Exract files and write metadata
    subsampler.extract_core_files()
    subsampler.extract_extra_files()
    subsampler.write_metadata()