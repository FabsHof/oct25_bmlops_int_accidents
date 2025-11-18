import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- 1. Lade Variablen aus .env ---
load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME")        
RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw")

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", 5432)

os.makedirs(RAW_PATH, exist_ok=True)

# --- 2. Kaggle Dataset herunterladen ---
api = KaggleApi()
api.authenticate()
api.dataset_download_files(DATASET_NAME, path=RAW_PATH, unzip=True)
print(f"Dataset heruntergeladen nach: {RAW_PATH}")

# # --- 3. CSV in PostgreSQL laden ---
# # Hier nehmen wir an, dass die CSV-Datei einen festen Namen hat, z. B. US_Accidents.csv
# csv_file = os.path.join(RAW_PATH, "US_Accidents.csv")  # <-- anpassen, falls anders
# df = pd.read_csv(csv_file)

# # PostgreSQL Verbindung
# engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# # CSV in die Tabelle 'accidents' importieren (Tabelle wird erstellt oder ersetzt)
# df.to_sql("accidents", engine, if_exists="replace", index=False)
# print("CSV erfolgreich in die Tabelle 'accidents' geladen!")
