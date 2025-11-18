import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv


# load environment variables
load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME")        
RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw")


os.makedirs(raw_path, exist_ok=True)



# download raw data from kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_files(DATASET_NAME, path=RAW_PATH, unzip=True)
print(f"Dataset heruntergeladen nach: {raw_path}")

