

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


# load env
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")


# connection to database
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# load csv files
csv_files = ["caracteristics.csv", "vehicles.csv", "users.csv", "holidays.csv", "places.csv"]

for file in csv_files:
    path = os.path.join("data/raw", file)
    df = pd.read_csv(path, encoding="ISO-8859-1")  #terminal check: file -i data/raw/caracteristics.csv    ->   data/raw/caracteristics.csv: text/csv; charset=iso-8859-1
    table_name = os.path.splitext(file)[0]  # name der Tabelle = dateiname ohne .csv
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"{file} loaded succesfully in table {table_name} ")
