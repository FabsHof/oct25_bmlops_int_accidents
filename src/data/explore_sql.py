from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

# load env
load_dotenv()


DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")



def load_table(table_name):
    # user = os.getenv("POSTGRES_USER")
    # pw = os.getenv("POSTGRES_PASSWORD")
    # db = os.getenv("POSTGRES_DB")

    # connection to database
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


    return pd.read_sql(f"SELECT * FROM {table_name};", engine)


if __name__ == "__main__":
    df = load_table("accidents_full")
    print(df.head())
    print(df.info())













