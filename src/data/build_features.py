from sqlalchemy import create_engine, inspect, text
import os
from dotenv import load_dotenv

# Environment variables laden
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# Verbindung zur Datenbank erstellen
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Tabellen, die gejoined werden sollen
tables = ["caracteristics", "vehicles", "users", "places"]
main_table = tables[0]  # Haupttabelle für Num_Acc

with engine.connect() as conn:
    inspector = inspect(engine)

    # Spalten für jede Tabelle abrufen
    columns = {}
    for table in tables:
        cols = inspector.get_columns(table)
        col_names = [c["name"] for c in cols]

        if table == main_table:
            # Num_Acc der Haupttabelle eindeutig referenzieren
            col_names_prefixed = [f'{main_table}."Num_Acc"'] + [f'{main_table}."{c}"' for c in col_names if c != "Num_Acc"]
        else:
            # andere Tabellen: alle außer Num_Acc aliasieren
            col_names_prefixed = [f'{table}."{c}" AS {table}__{c}' for c in col_names if c != "Num_Acc"]

        columns[table] = col_names_prefixed

    # SELECT-Teil zusammenbauen
    select_parts = []
    for table in tables:
        select_parts.extend(columns[table])

    # JOIN-Teil zusammenbauen
    joins = " ".join(
        [f"LEFT JOIN {t} ON {main_table}.\"Num_Acc\" = {t}.\"Num_Acc\"" for t in tables[1:]]
    )

    # Vollständiges SQL
    sql = f"""
    DROP TABLE IF EXISTS accidents_full;

    CREATE TABLE accidents_full AS
    SELECT
        {', '.join(select_parts)}
    FROM {main_table}
    {joins};
    """

    # SQL ausführen
    with engine.connect() as conn:
        conn.execute(text(sql))
    print("Tabelle 'accidents_full' erfolgreich erstellt!")
