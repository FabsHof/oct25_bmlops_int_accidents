import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection() -> connection:
    """
    Create and return a database connection using environment variables.
    
    Returns:
        connection: psycopg2 database connection object
        
    Raises:
        ValueError: If required environment variables are missing
        psycopg2.Error: If database connection fails
    """
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_HOST_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'accidents_db'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }
    
    # Validate required environment variables
    if not db_config['user'] or not db_config['password']:
        raise ValueError("POSTGRES_USER and POSTGRES_PASSWORD must be set in environment variables")
    
    logger.info(f"Connecting to database: {db_config['database']} at {db_config['host']}:{db_config['port']}")
    
    try:
        conn = psycopg2.connect(**db_config)
        logger.info("Database connection established successfully")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def find_latest_csv_directory(raw_data_path: str) -> Optional[Path]:
    """
    Find the latest directory containing CSV files in the raw data path.
    
    Args:
        raw_data_path: Base path to search for raw data directories
        
    Returns:
        Path to the latest directory containing CSV files, or None if not found
    """
    raw_path = Path(raw_data_path)
    
    if not raw_path.exists():
        logger.warning(f"Raw data path does not exist: {raw_data_path}")
        return None
    
    # Look for timestamped directories (e.g., 20251121_1318)
    csv_dirs = []
    for date_dir in raw_path.iterdir():
        if date_dir.is_dir():
            # Check subdirectories for CSV files
            for subdir in date_dir.iterdir():
                if subdir.is_dir():
                    csv_files = list(subdir.glob("*.csv"))
                    if csv_files:
                        csv_dirs.append(subdir)
    
    if not csv_dirs:
        logger.warning(f"No directories with CSV files found in {raw_data_path}")
        return None
    
    # Return the most recently modified directory
    latest_dir = max(csv_dirs, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest CSV directory: {latest_dir}")
    return latest_dir


def load_csv_to_dataframe(csv_path: Path, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with error handling.
    
    Args:
        csv_path: Path to the CSV file
        encoding: File encoding (default: 'utf-8', fallback to 'latin-1')
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: If CSV loading fails with both encodings
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        logger.info(f"Loaded {csv_path.name} with {len(df)} rows using {encoding} encoding")
        return df
    except UnicodeDecodeError:
        logger.warning(f"Failed to load {csv_path.name} with {encoding}, trying latin-1")
        df = pd.read_csv(csv_path, encoding='latin-1')
        logger.info(f"Loaded {csv_path.name} with {len(df)} rows using latin-1 encoding")
        return df


def store_caracteristics(conn: connection, df: pd.DataFrame) -> int:
    """
    Store caracteristics data in the raw_caracteristics table using batch inserts.
    
    Args:
        conn: Database connection
        df: DataFrame containing caracteristics data
        
    Returns:
        Number of rows inserted
    """
    logger.info("Storing caracteristics data...")
    
    # Column mapping: CSV column -> DB column (lowercase)
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    
    # Helper function to convert to int or None
    def to_int_or_none(val):
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    
    # Prepare data for batch insert - use vectorized operations
    # Convert numeric columns, replacing NaN with None
    for col in ['num_acc', 'an', 'mois', 'jour', 'hrmn', 'lum', 'agg', 'int', 'atm', 'col', 'com', 'lat', 'dep']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].round().astype('Int64').replace({pd.NA: None})
    
    # Convert string columns
    df_clean['adr'] = df_clean['adr'].astype(str).replace({'nan': None})
    df_clean['gps'] = df_clean['gps'].astype(str).replace({'nan': None})
    df_clean['long'] = df_clean['long'].astype(str).replace({'nan': None})
    
    # Convert to list of tuples - much faster than iterrows
    data_tuples = [tuple(x) for x in df_clean[['num_acc', 'an', 'mois', 'jour', 'hrmn', 'lum', 'agg', 'int', 
                                                 'atm', 'col', 'com', 'adr', 'gps', 'lat', 'long', 'dep']].values]
    
    cursor = conn.cursor()
    
    try:
        # Use execute_batch for efficient batch inserts
        execute_batch(cursor, """
            INSERT INTO raw_caracteristics 
            (num_acc, an, mois, jour, hrmn, lum, agg, int, atm, col, com, adr, gps, lat, long, dep)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (num_acc) DO NOTHING
        """, data_tuples, page_size=1000)
        
        conn.commit()
        inserted_count = len(data_tuples)
        logger.info(f"Successfully stored {inserted_count} caracteristics records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing caracteristics data: {e}")
        raise
    finally:
        cursor.close()


def store_holidays(conn: connection, df: pd.DataFrame) -> int:
    """
    Store holidays data in the raw_holidays table.
    
    Args:
        conn: Database connection
        df: DataFrame containing holidays data
        
    Returns:
        Number of rows inserted
    """
    logger.info("Storing holidays data...")
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    
    # Convert string columns
    df_clean['ds'] = df_clean['ds'].astype(str).replace({'nan': None})
    df_clean['holiday'] = df_clean['holiday'].astype(str).replace({'nan': None})
    
    # Convert to list of tuples
    data_tuples = [tuple(x) for x in df_clean[['ds', 'holiday']].values]
    
    cursor = conn.cursor()
    
    try:
        execute_batch(cursor, """
            INSERT INTO raw_holidays (ds, holiday)
            VALUES (%s, %s)
        """, data_tuples, page_size=1000)
        
        conn.commit()
        inserted_count = len(data_tuples)
        logger.info(f"Successfully stored {inserted_count} holidays records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing holidays data: {e}")
        raise
    finally:
        cursor.close()


def store_places(conn: connection, df: pd.DataFrame) -> int:
    """
    Store places data in the raw_places table.
    
    Args:
        conn: Database connection
        df: DataFrame containing places data
        
    Returns:
        Number of rows inserted
    """
    logger.info("Storing places data...")
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    
    # Convert numeric columns - round floats first to avoid casting errors
    for col in ['num_acc', 'catr', 'voie', 'v1', 'circ', 'nbv', 'pr', 'pr1', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].round().astype('Int64').replace({pd.NA: None})
    
    # Convert string columns
    df_clean['v2'] = df_clean['v2'].astype(str).replace({'nan': None})
    
    # Convert to list of tuples
    data_tuples = [tuple(x) for x in df_clean[['num_acc', 'catr', 'voie', 'v1', 'v2', 'circ', 'nbv', 'pr', 'pr1', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1']].values]
    
    cursor = conn.cursor()
    
    try:
        execute_batch(cursor, """
            INSERT INTO raw_places 
            (num_acc, catr, voie, v1, v2, circ, nbv, pr, pr1, vosp, prof, plan, lartpc, larrout, surf, infra, situ, env1)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (num_acc) DO NOTHING
        """, data_tuples, page_size=1000)
        
        conn.commit()
        inserted_count = len(data_tuples)
        logger.info(f"Successfully stored {inserted_count} places records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing places data: {e}")
        raise
    finally:
        cursor.close()


def store_users(conn: connection, df: pd.DataFrame) -> int:
    """
    Store users data in the raw_users table.
    
    Args:
        conn: Database connection
        df: DataFrame containing users data
        
    Returns:
        Number of rows inserted
    """
    logger.info("Storing users data...")
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    
    # Helper function to convert to int or None
    def to_int_or_none(val):
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    
    # Convert numeric columns
    for col in ['num_acc', 'place', 'catu', 'grav', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp', 'an_nais']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].round().astype('Int64').replace({pd.NA: None})
    
    # Convert string columns
    df_clean['num_veh'] = df_clean['num_veh'].astype(str).replace({'nan': None})
    
    # Convert to list of tuples
    data_tuples = [tuple(x) for x in df_clean[['num_acc', 'place', 'catu', 'grav', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp', 'an_nais', 'num_veh']].values]
    
    cursor = conn.cursor()
    
    try:
        execute_batch(cursor, """
            INSERT INTO raw_users 
            (num_acc, place, catu, grav, sexe, trajet, secu, locp, actp, etatp, an_nais, num_veh)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, data_tuples, page_size=1000)
        
        conn.commit()
        inserted_count = len(data_tuples)
        logger.info(f"Successfully stored {inserted_count} users records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing users data: {e}")
        raise
    finally:
        cursor.close()


def store_vehicles(conn: connection, df: pd.DataFrame) -> int:
    """
    Store vehicles data in the raw_vehicles table.
    
    Args:
        conn: Database connection
        df: DataFrame containing vehicles data
        
    Returns:
        Number of rows inserted
    """
    logger.info("Storing vehicles data...")
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    
    # Helper function to convert to int or None
    def to_int_or_none(val):
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    
    # Convert numeric columns
    for col in ['num_acc', 'senc', 'catv', 'occutc', 'obs', 'obsm', 'choc', 'manv']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].round().astype('Int64').replace({pd.NA: None})
    
    # Convert string columns
    df_clean['num_veh'] = df_clean['num_veh'].astype(str).replace({'nan': None})
    
    # Convert to list of tuples
    data_tuples = [tuple(x) for x in df_clean[['num_acc', 'senc', 'catv', 'occutc', 'obs', 'obsm', 'choc', 'manv', 'num_veh']].values]
    
    cursor = conn.cursor()
    
    try:
        execute_batch(cursor, """
            INSERT INTO raw_vehicles 
            (num_acc, senc, catv, occutc, obs, obsm, choc, manv, num_veh)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (num_acc, num_veh) DO NOTHING
        """, data_tuples, page_size=1000)
        
        conn.commit()
        inserted_count = len(data_tuples)
        logger.info(f"Successfully stored {inserted_count} vehicles records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing vehicles data: {e}")
        raise
    finally:
        cursor.close()


def store_data(
    raw_data_path: Optional[str] = None,
    db_connection: Optional[connection] = None
) -> dict:
    """
    Main function to load CSV files and store them in the database.
    
    Args:
        raw_data_path: Path to raw data directory (uses env var DATA_RAW_PATH if not provided)
        db_connection: Existing database connection (creates new one if not provided)
        
    Returns:
        Dictionary with counts of inserted records for each table
        
    Raises:
        ValueError: If raw_data_path is invalid or CSV files not found
        Exception: If database operations fail
    """
    logger.info("Starting data storage process...")
    
    # Get raw data path from argument or environment variable
    if raw_data_path is None:
        raw_data_path = os.getenv('DATA_RAW_PATH', 'data/raw')
    
    if not raw_data_path:
        raise ValueError("raw_data_path must be provided or DATA_RAW_PATH must be set")
    
    # Find the latest CSV directory
    csv_dir = find_latest_csv_directory(raw_data_path)
    if csv_dir is None:
        raise ValueError(f"No CSV files found in {raw_data_path}")
    
    # Define CSV file mappings
    csv_files = {
        'caracteristics': csv_dir / 'caracteristics.csv',
        'holidays': csv_dir / 'holidays.csv',
        'places': csv_dir / 'places.csv',
        'users': csv_dir / 'users.csv',
        'vehicles': csv_dir / 'vehicles.csv'
    }
    
    # Verify all required CSV files exist
    missing_files = [name for name, path in csv_files.items() if not path.exists()]
    if missing_files:
        raise ValueError(f"Missing CSV files: {', '.join(missing_files)}")
    
    # Create or use provided database connection
    close_connection = False
    if db_connection is None:
        conn = get_db_connection()
        close_connection = True
    else:
        conn = db_connection
    
    results = {}
    
    try:
        # Load and store caracteristics (must be first due to foreign key constraints)
        df_caracteristics = load_csv_to_dataframe(csv_files['caracteristics'])
        results['caracteristics'] = store_caracteristics(conn, df_caracteristics)
        
        # Load and store holidays
        df_holidays = load_csv_to_dataframe(csv_files['holidays'])
        results['holidays'] = store_holidays(conn, df_holidays)
        
        # Load and store places
        df_places = load_csv_to_dataframe(csv_files['places'])
        results['places'] = store_places(conn, df_places)
        
        # Load and store users
        df_users = load_csv_to_dataframe(csv_files['users'])
        results['users'] = store_users(conn, df_users)
        
        # Load and store vehicles
        df_vehicles = load_csv_to_dataframe(csv_files['vehicles'])
        results['vehicles'] = store_vehicles(conn, df_vehicles)
        
        logger.info("Data storage completed successfully!")
        logger.info(f"Summary: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during data storage: {e}")
        raise
    finally:
        if close_connection and conn:
            conn.close()
            logger.info("Database connection closed")


def main():
    """Entry point for the script."""
    try:
        results = store_data()
        print("\n=== Data Storage Summary ===")
        for table, count in results.items():
            print(f"{table}: {count} records inserted")
        print("============================\n")
    except Exception as e:
        logger.error(f"Failed to store data: {e}")
        raise


if __name__ == "__main__":
    main()