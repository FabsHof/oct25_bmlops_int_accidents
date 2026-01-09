"""
Unified data storage module supporting both full batch and chunked incremental loading.

This module provides functions to load CSV data into the database either all at once
or in configurable chunks to simulate data evolution.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from psycopg2.extensions import connection
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import argparse

from src.utils import logging
from src.utils.database import (
    get_db_connection,
    initialize_progress_tracking,
    get_progress_status,
    update_progress,
    reset_progress as db_reset_progress
)
from src.utils.data_processing import prepare_table_data, load_dataframe_chunk

# Load environment variables
load_dotenv()

# Default chunk size for incremental loading
DEFAULT_CHUNK_SIZE = 10000

# SQL insert statements for each table
INSERT_QUERIES = {
    'caracteristics': """
        INSERT INTO raw_caracteristics 
        (num_acc, an, mois, jour, hrmn, lum, agg, int, atm, col, com, adr, gps, lat, long, dep)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (num_acc) DO NOTHING
    """,
    'holidays': """
        INSERT INTO raw_holidays (ds, holiday)
        VALUES (%s, %s)
    """,
    'places': """
        INSERT INTO raw_places 
        (num_acc, catr, voie, v1, v2, circ, nbv, pr, pr1, vosp, prof, plan, lartpc, larrout, surf, infra, situ, env1)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (num_acc) DO NOTHING
    """,
    'users': """
        INSERT INTO raw_users 
        (num_acc, place, catu, grav, sexe, trajet, secu, locp, actp, etatp, an_nais, num_veh)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
    'vehicles': """
        INSERT INTO raw_vehicles 
        (num_acc, senc, catv, occutc, obs, obsm, choc, manv, num_veh)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (num_acc, num_veh) DO NOTHING
    """
}


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
        logging.warning(f"Raw data path does not exist: {raw_data_path}")
        return None
    
    # Look for timestamped directories (e.g., 20251121_1318)
    csv_dirs = []
    for date_dir in raw_path.iterdir():
        if date_dir.is_dir():
            # First check if the date_dir itself has CSV files
            csv_files = list(date_dir.glob("*.csv"))
            if csv_files:
                csv_dirs.append(date_dir)
            else:
                # Check subdirectories for CSV files
                for subdir in date_dir.iterdir():
                    if subdir.is_dir():
                        csv_files = list(subdir.glob("*.csv"))
                        if csv_files:
                            csv_dirs.append(subdir)
    
    if not csv_dirs:
        logging.warning(f"No directories with CSV files found in {raw_data_path}")
        return None
    
    # Return the most recently modified directory
    latest_dir = max(csv_dirs, key=lambda p: p.stat().st_mtime)
    logging.info(f"Found latest CSV directory: {latest_dir}")
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
        logging.info(f"Loaded {csv_path.name} with {len(df)} rows using {encoding} encoding")
        return df
    except UnicodeDecodeError:
        logging.warning(f"Failed to load {csv_path.name} with {encoding}, trying latin-1")
        df = pd.read_csv(csv_path, encoding='latin-1')
        logging.info(f"Loaded {csv_path.name} with {len(df)} rows using latin-1 encoding")
        return df


def insert_table_data(conn: connection, table_name: str, data_tuples: list, batch_size: int = 1000) -> int:
    """
    Insert data tuples into a specific table.
    
    Args:
        conn: Database connection
        table_name: Name of the table to insert into
        data_tuples: List of tuples containing row data
        batch_size: Batch size for insert operations
        
    Returns:
        Number of rows inserted
    """
    if not data_tuples:
        logging.info(f"No data to insert for {table_name}")
        return 0
    
    cursor = conn.cursor()
    
    try:
        execute_batch(cursor, INSERT_QUERIES[table_name], data_tuples, page_size=batch_size)
        conn.commit()
        inserted_count = len(data_tuples)
        logging.info(f"Successfully inserted {inserted_count} {table_name} records")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logging.error(f"Error inserting {table_name} data: {e}")
        raise
    finally:
        cursor.close()


def store_table_full(conn: connection, table_name: str, csv_path: Path) -> int:
    """
    Store all data from a CSV file into a table (full batch load).
    
    Args:
        conn: Database connection
        table_name: Name of the table
        csv_path: Path to the CSV file
        
    Returns:
        Number of rows inserted
    """
    logging.info(f"Storing {table_name} data (full batch)...")
    
    # Load entire CSV
    df = load_csv_to_dataframe(csv_path)
    
    # Prepare data
    data_tuples = prepare_table_data(df, table_name)
    
    # Insert data
    return insert_table_data(conn, table_name, data_tuples)


def store_table_chunk(conn: connection, table_name: str, csv_path: Path, offset: int, chunk_size: int) -> int:
    """
    Store a chunk of data from a CSV file into a table.
    
    Args:
        conn: Database connection
        table_name: Name of the table
        csv_path: Path to the CSV file
        offset: Starting row (0-indexed, excluding header)
        chunk_size: Number of rows to load
        
    Returns:
        Number of rows inserted
    """
    logging.info(f"Loading {table_name} chunk: offset={offset}, size={chunk_size}")
    
    # Load chunk
    df = load_dataframe_chunk(csv_path, offset, chunk_size)
    
    if df.empty:
        logging.info(f"No more data to load for {table_name}")
        return 0
    
    # Prepare data
    data_tuples = prepare_table_data(df, table_name)
    
    # Insert data
    return insert_table_data(conn, table_name, data_tuples)


def ingest_data_full(
    raw_data_path: Optional[str] = None,
    db_connection: Optional[connection] = None
) -> Dict[str, int]:
    """
    Load all CSV files and store them in the database at once (full batch load).
    
    Args:
        raw_data_path: Path to raw data directory (uses env var DATA_RAW_PATH if not provided)
        db_connection: Existing database connection (creates new one if not provided)
        
    Returns:
        Dictionary with counts of inserted records for each table
        
    Raises:
        ValueError: If raw_data_path is invalid or CSV files not found
        Exception: If database operations fail
    """
    logging.info("Starting full batch data storage process...")
    
    # Get raw data path from argument or environment variable
    if raw_data_path is None:
        raw_data_path = os.getenv('RAW_DATA_PATH', 'data/raw')
    
    if not raw_data_path:
        raise ValueError("raw_data_path must be provided or RAW_DATA_PATH must be set")
    
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
        # Load tables in order (caracteristics first due to foreign keys)
        load_order = ['caracteristics', 'holidays', 'places', 'users', 'vehicles']
        
        for table_name in load_order:
            results[table_name] = store_table_full(conn, table_name, csv_files[table_name])
        
        logging.info("Data storage completed successfully!")
        logging.info(f"Summary: {results}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error during data storage: {e}")
        raise
    finally:
        if close_connection and conn:
            conn.close()
            logging.info("Database connection closed")


def load_next_chunk(raw_data_path: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
    """
    Load the next chunk of data for all tables that are not yet complete.
    
    Args:
        raw_data_path: Path to raw data directory (uses env var DATA_RAW_PATH if not provided)
        chunk_size: Number of rows to load per chunk (only used for initialization)
        
    Returns:
        Dictionary with results for each table and overall progress
    """
    logging.info("Loading next chunk of data...")
    
    # Get raw data path from argument or environment variable
    if raw_data_path is None:
        raw_data_path = os.getenv('RAW_DATA_PATH', 'data/raw')
    
    if not raw_data_path:
        raise ValueError("raw_data_path must be provided or RAW_DATA_PATH must be set")
    
    conn = get_db_connection()
    
    try:
        # Disable foreign key constraints for bulk loading
        cursor = conn.cursor()
        cursor.execute("SET session_replication_role = 'replica';")
        cursor.close()
        logging.info("Foreign key constraints temporarily disabled for bulk loading")
        
        # Get current progress
        progress = get_progress_status(conn)
        
        # If progress tracking is not initialized, initialize it
        if not progress:
            csv_dir = find_latest_csv_directory(raw_data_path)
            if csv_dir is None:
                raise ValueError(f"No CSV files found in {raw_data_path}")
            initialize_progress_tracking(conn, csv_dir, chunk_size)
            progress = get_progress_status(conn)
        else:
            # Check if chunk_size differs from stored value and update if needed
            stored_chunk_size = next(iter(progress.values()))['chunk_size']
            if chunk_size != stored_chunk_size:
                logging.info(f"Updating chunk size from {stored_chunk_size} to {chunk_size}")
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE data_ingestion_progress 
                        SET chunk_size = %s, last_updated = CURRENT_TIMESTAMP
                    """, (chunk_size,))
                    conn.commit()
                    logging.info("Chunk size updated successfully")
                    # Refresh progress with new chunk size
                    progress = get_progress_status(conn)
                finally:
                    cursor.close()
        
        # Get CSV directory from progress tracking
        csv_directory = None
        for table_progress in progress.values():
            if table_progress['csv_directory']:
                csv_directory = Path(table_progress['csv_directory'])
                break
        
        if csv_directory is None:
            raise ValueError("CSV directory not found in progress tracking")
        
        # Define the loading order (must load caracteristics first due to foreign keys)
        load_order = ['caracteristics', 'holidays', 'places', 'users', 'vehicles']
        
        results = {}
        all_complete = True
        
        # Check if caracteristics is complete before loading dependent tables
        caracteristics_complete = progress.get('caracteristics', {}).get('is_complete', False)
        
        for table_name in load_order:
            if table_name not in progress:
                logging.warning(f"No progress tracking for {table_name}, skipping")
                continue
            
            table_progress = progress[table_name]
            
            if table_progress['is_complete']:
                results[table_name] = {
                    'loaded': 0,
                    'message': 'Already complete',
                    'progress_percentage': 100
                }
                continue
            
            # Skip dependent tables if caracteristics is not complete
            # This prevents foreign key violations
            if table_name in ['places', 'users', 'vehicles'] and not caracteristics_complete:
                results[table_name] = {
                    'loaded': 0,
                    'message': 'Waiting for caracteristics to complete',
                    'progress_percentage': table_progress['progress_percentage']
                }
                continue
            
            all_complete = False
            
            csv_path = csv_directory / f"{table_name}.csv"
            
            if not csv_path.exists():
                logging.warning(f"CSV file not found: {csv_path}")
                results[table_name] = {
                    'loaded': 0,
                    'message': 'CSV file not found',
                    'progress_percentage': table_progress['progress_percentage']
                }
                continue
            
            # Load the next chunk
            offset = table_progress['rows_loaded']
            chunk_size_to_use = table_progress['chunk_size']
            
            rows_loaded = store_table_chunk(conn, table_name, csv_path, offset, chunk_size_to_use)
            
            # Update progress
            if rows_loaded > 0:
                update_progress(conn, table_name, rows_loaded)
            
            # Get updated progress
            updated_progress = get_progress_status(conn)
            table_updated = updated_progress[table_name]
            
            results[table_name] = {
                'loaded': rows_loaded,
                'total_loaded': table_updated['rows_loaded'],
                'total_rows': table_updated['total_rows'],
                'progress_percentage': table_updated['progress_percentage'],
                'is_complete': table_updated['is_complete']
            }
        
        return {
            'success': True,
            'tables': results,
            'all_complete': all_complete,
            'message': 'All data loaded successfully' if all_complete else 'Chunk loaded successfully'
        }
        
    except Exception as e:
        logging.error(f"Error loading next chunk: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to load data chunk'
        }
    finally:
        # Re-enable foreign key constraints
        try:
            cursor = conn.cursor()
            cursor.execute("SET session_replication_role = 'origin';")
            cursor.close()
            conn.commit()
        except Exception as e:
            logging.warning(f"Failed to re-enable foreign key constraints: {e}")
        if conn:
            conn.close()


def reset_progress(raw_data_path: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
    """
    Reset the progress tracking to start from the beginning.
    
    Args:
        raw_data_path: Path to raw data directory (uses env var DATA_RAW_PATH if not provided)
        chunk_size: Number of rows to load per chunk
        
    Returns:
        Dictionary with reset status
    """
    logging.info("Resetting data ingestion progress...")
    
    # Get raw data path from argument or environment variable
    if raw_data_path is None:
        raw_data_path = os.getenv('RAW_DATA_PATH', 'data/raw')
    
    conn = get_db_connection()
    
    try:
        # Find CSV directory
        csv_dir = find_latest_csv_directory(raw_data_path)
        if csv_dir is None:
            raise ValueError(f"No CSV files found in {raw_data_path}")
        
        # Reset progress
        db_reset_progress(conn, csv_dir, chunk_size)
        
        return {
            'success': True,
            'message': 'Progress reset successfully',
            'csv_directory': str(csv_dir)
        }
        
    except Exception as e:
        logging.error(f"Error resetting progress: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to reset progress'
        }
    finally:
        if conn:
            conn.close()


# Backward compatibility wrapper for tests
def ingest_data_chunk(
    raw_data_path: Optional[str] = None,
    db_connection: Optional[connection] = None
) -> Dict[str, int]:
    """
    Load next chunk of data (backward compatibility wrapper for tests).
    
    Note: db_connection parameter is ignored as load_next_chunk manages connections internally.
    
    Args:
        raw_data_path: Path to raw data directory
        db_connection: Ignored (kept for backward compatibility)
        
    Returns:
        Dictionary with counts of loaded records for each table
    """
    result = load_next_chunk(raw_data_path)
    
    if not result.get('success', False):
        raise Exception(result.get('message', 'Failed to load data chunk'))
    
    # Convert result format to match test expectations
    return {
        table_name: table_result.get('loaded', 0)
        for table_name, table_result in result.get('tables', {}).items()
    }


# Backward compatibility wrappers for individual table functions (for tests)
def store_caracteristics(conn: connection, df: pd.DataFrame) -> int:
    """
    Store caracteristics data (backward compatibility wrapper).
    
    Args:
        conn: Database connection
        df: DataFrame containing caracteristics data
        
    Returns:
        Number of rows inserted
    """
    data_tuples = prepare_table_data(df, 'caracteristics')
    return insert_table_data(conn, 'caracteristics', data_tuples)


def store_holidays(conn: connection, df: pd.DataFrame) -> int:
    """
    Store holidays data (backward compatibility wrapper).
    
    Args:
        conn: Database connection
        df: DataFrame containing holidays data
        
    Returns:
        Number of rows inserted
    """
    data_tuples = prepare_table_data(df, 'holidays')
    return insert_table_data(conn, 'holidays', data_tuples)


def store_places(conn: connection, df: pd.DataFrame) -> int:
    """
    Store places data (backward compatibility wrapper).
    
    Args:
        conn: Database connection
        df: DataFrame containing places data
        
    Returns:
        Number of rows inserted
    """
    data_tuples = prepare_table_data(df, 'places')
    return insert_table_data(conn, 'places', data_tuples)


def store_users(conn: connection, df: pd.DataFrame) -> int:
    """
    Store users data (backward compatibility wrapper).
    
    Args:
        conn: Database connection
        df: DataFrame containing users data
        
    Returns:
        Number of rows inserted
    """
    data_tuples = prepare_table_data(df, 'users')
    return insert_table_data(conn, 'users', data_tuples)


def store_vehicles(conn: connection, df: pd.DataFrame) -> int:
    """
    Store vehicles data (backward compatibility wrapper).
    
    Args:
        conn: Database connection
        df: DataFrame containing vehicles data
        
    Returns:
        Number of rows inserted
    """
    data_tuples = prepare_table_data(df, 'vehicles')
    return insert_table_data(conn, 'vehicles', data_tuples)


def main(ingestion_mode: str = 'full', chunk_size: int = None) -> None:
    """Entry point for the script (full or batch load)."""
    try:
        if ingestion_mode == 'chunked':
            logging.info("Loading next chunk of data...")
            if chunk_size is None:
                chunk_size = DEFAULT_CHUNK_SIZE
            logging.info(f"Using chunk size: {chunk_size}")
            result = load_next_chunk(chunk_size=chunk_size)
            if not result.get('success', False):
                logging.error(f"Chunk loading failed: {result.get('message')}")
            else:
                logging.info(f"Chunk loaded successfully: {result.get('tables')}")
                if result.get('all_complete', False):
                    logging.info("All data has been loaded!")
                else:
                    logging.info("Run the command again to load the next chunk.")
        else:
            logging.info("Starting full data ingestion...")
            results = ingest_data_full()
            logging.info(f"Full data ingestion completed: {results}")
    except Exception as e:
        logging.error(f"Failed to ingest data: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ingest data into the database (full or batch).")
    parser.add_argument(
        '--mode',
        choices=['full', 'chunked'],
        default='full',
        help="Ingestion mode: 'full' for full batch load, 'chunked' for incremental loading"
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for incremental loading (only used in 'chunked' mode)"
    )

    args = parser.parse_args()

    main(args.mode, args.chunk_size)
