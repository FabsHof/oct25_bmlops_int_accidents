
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

engine=create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

df_ca=pd.read_sql(f"SELECT * FROM caracteristics;", engine)
df_us=pd.read_sql(f"SELECT * FROM users;", engine)
df_ve=pd.read_sql(f"SELECT * FROM vehicles;", engine)
df_pl=pd.read_sql(f"SELECT * FROM places;", engine)

print(df_ca.head())
print(df_us.head())
print(df_ve.head())
print(df_pl.head())

df1 = pd.merge(df_ca, df_pl, how="outer", on="Num_Acc")
df2 = pd.merge(df_ve, df_us, how='outer', on="Num_Acc")

initial_data = pd.merge(df1, df2, how='outer', on="Num_Acc")

columns = ['an', 'mois', 'catu', 'grav', 'sexe', 'an_nais', 'trajet', 'secu', 'lum', 'atm', 'catr', 'surf']

df_preprocessed = initial_data[columns]
df_preprocessed.columns = ['Year', 'Month', 'User category', 'Severity', 'Sex', 'Year of birth', 'Trip purpose', 'Securiy', 'Luminosity', 'Weather', 'Type of road', 'Road surface']

print(df_preprocessed.head())

df_preprocessed.to_sql(preprocessed, engine, if_exists="replace", index=False)