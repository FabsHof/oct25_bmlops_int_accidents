import os
from datetime import datetime
from os import path
from typing import Callable, Optional
import kagglehub as kh

from src.utils import logging

import shutil

def download_data(
    raw_data_path: str,
    download_func: Optional[Callable[[], str]],
    add_timestamp: bool = False,
) -> str:
    '''
    Download data, using the provided download function.

    Args:
        raw_data_path: Destination directory for the downloaded data
        download_func: Function to perform the download.
        add_timestamp: Whether to add a timestamp directory level to raw_data_path

    Returns:
        Path to the downloaded dataset

    Raises:
        ValueError: If raw_data_path is empty or download_func is empty
        FileNotFoundError: If the download fails or source file doesn't exist
    '''
    if not raw_data_path:
        raise ValueError('raw_data_path cannot be empty')
    if not download_func:
        raise ValueError('download_func cannot be empty')

    logging.info('>>> Downloading dataset ...')
    file_path = download_func()

    # Move downloaded file to target directory
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Downloaded file not found: {file_path}')
    if add_timestamp:
        now = datetime.now()
        timestamp = f'{now.strftime("%Y%m%d_%H%M")}'
        raw_data_path = path.join(raw_data_path, timestamp)
    os.makedirs(raw_data_path, exist_ok=True)
    new_file_path = path.join(raw_data_path, os.path.basename(file_path))

    if os.path.isdir(file_path):
        shutil.copytree(file_path, new_file_path)
    else:
        shutil.copy2(file_path, new_file_path)

    logging.info('>>> Data saved to: %s', new_file_path)
    return new_file_path


def main() -> None:
    '''Main entry point for downloading data.'''
    def download_kaggle_data() -> str:
        dataset_id = 'ahmedlahlou/accidents-in-france-from-2005-to-2016'
        return kh.dataset_download(dataset_id)

    root_dir = path.join(path.dirname(path.abspath(__file__)), '..', '..')
    raw_data_path = path.join(root_dir, os.getenv('RAW_DATA_PATH', 'data/raw/'))
    download_data(raw_data_path, download_kaggle_data, add_timestamp=True)

if __name__ == "__main__":
    main()