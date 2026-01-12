"""
Database utilities for managing progress tracking and database operations.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import psycopg2
from psycopg2.extensions import connection
from dotenv import load_dotenv

from src.utils import logging

load_dotenv()


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
        'host': os.getenv('ACCIDENTS_POSTGRES_HOST', 'localhost'),
        'port': os.getenv('ACCIDENTS_POSTGRES_PORT', '5432'),
        'database': os.getenv('ACCIDENTS_POSTGRES_DB', 'accidents_db'),
        'user': os.getenv('ACCIDENTS_POSTGRES_USER'),
        'password': os.getenv('ACCIDENTS_POSTGRES_PASSWORD')
    }
    
    # Validate required environment variables
    if not db_config['user'] or not db_config['password']:
        raise ValueError("ACCIDENTS_POSTGRES_USER and ACCIDENTS_POSTGRES_PASSWORD must be set in environment variables")
    
    logging.info(f"Connecting to database: {db_config['database']} at {db_config['host']}:{db_config['port']}")
    
    try:
        conn = psycopg2.connect(**db_config)
        logging.info("Database connection established successfully")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise


def initialize_progress_tracking(conn: connection, csv_dir: Path, chunk_size: int = 1000) -> None:
    """
    Initialize the data_ingestion_progress table with metadata for all tables.
    Creates a new version for this ingestion cycle.
    
    Args:
        conn: Database connection
        csv_dir: Directory containing CSV files
        chunk_size: Number of rows to load per chunk
    """
    logging.info("Initializing progress tracking...")
    
    csv_files = {
        'caracteristics': csv_dir / 'caracteristics.csv',
        'holidays': csv_dir / 'holidays.csv',
        'places': csv_dir / 'places.csv',
        'users': csv_dir / 'users.csv',
        'vehicles': csv_dir / 'vehicles.csv'
    }
    
    cursor = conn.cursor()
    
    try:
        # Get the next version number
        cursor.execute("""
            SELECT COALESCE(MAX(version), 0) + 1 as next_version
            FROM data_ingestion_progress
        """)
        next_version = cursor.fetchone()[0]
        logging.info(f"Starting new ingestion cycle with version: {next_version}")
        
        for table_name, csv_path in csv_files.items():
            if csv_path.exists():
                # Get total row count (excluding header) with encoding fallback
                try:
                    with open(csv_path, encoding='utf-8') as f:
                        total_rows = sum(1 for _ in f) - 1
                except UnicodeDecodeError:
                    logging.warning(f"Failed to read {csv_path.name} with utf-8, trying latin-1")
                    with open(csv_path, encoding='latin-1') as f:
                        total_rows = sum(1 for _ in f) - 1
                
                # Always insert a new row for each new ingestion cycle
                cursor.execute("""
                    INSERT INTO data_ingestion_progress 
                    (table_name, rows_loaded, total_rows, chunk_size, csv_directory, is_complete, version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (table_name, 0, total_rows, chunk_size, str(csv_dir), False, next_version))
                
                logging.info(f"Initialized tracking for {table_name}: {total_rows} total rows (version {next_version})")
        
        conn.commit()
        logging.info(f"Progress tracking initialized successfully for version {next_version}")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error initializing progress tracking: {e}")
        raise
    finally:
        cursor.close()


def get_progress_status(conn: connection) -> Dict[str, Any]:
    """
    Get the current progress status for all tables in the most recent ingestion cycle.
    
    Args:
        conn: Database connection
        
    Returns:
        Dictionary with progress information for each table
    """
    cursor = conn.cursor()
    
    try:
        # Get only the most recent version entries for each table
        cursor.execute("""
            WITH latest_entries AS (
                SELECT DISTINCT ON (table_name)
                    table_name, rows_loaded, total_rows, chunk_size, 
                    last_updated, csv_directory, is_complete, version
                FROM data_ingestion_progress
                ORDER BY table_name, version DESC, id DESC
            )
            SELECT table_name, rows_loaded, total_rows, chunk_size, 
                   last_updated, csv_directory, is_complete, version
            FROM latest_entries
            ORDER BY table_name
        """)
        
        results = cursor.fetchall()
        
        progress = {}
        for row in results:
            table_name, rows_loaded, total_rows, chunk_size, last_updated, csv_directory, is_complete, version = row
            progress[table_name] = {
                'rows_loaded': rows_loaded,
                'total_rows': total_rows,
                'chunk_size': chunk_size,
                'last_updated': last_updated.isoformat() if last_updated else None,
                'csv_directory': csv_directory,
                'is_complete': is_complete,
                'version': version,
                'progress_percentage': (rows_loaded / total_rows * 100) if total_rows > 0 else 0
            }
        
        return progress
        
    finally:
        cursor.close()


def update_progress(conn: connection, table_name: str, rows_loaded: int) -> None:
    """
    Update the progress for a specific table in the current ingestion cycle.
    Updates the most recent entry for this table.
    
    Args:
        conn: Database connection
        table_name: Name of the table
        rows_loaded: Number of rows loaded in this operation
    """
    cursor = conn.cursor()
    
    try:
        # Update only the most recent entry for this table
        cursor.execute("""
            UPDATE data_ingestion_progress
            SET rows_loaded = rows_loaded + %s,
                last_updated = CURRENT_TIMESTAMP,
                is_complete = (rows_loaded + %s >= total_rows)
            WHERE id = (
                SELECT id FROM data_ingestion_progress
                WHERE table_name = %s
                ORDER BY version DESC, id DESC
                LIMIT 1
            )
        """, (rows_loaded, rows_loaded, table_name))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating progress for {table_name}: {e}")
        raise
    finally:
        cursor.close()


def reset_progress(conn: connection, csv_dir: Path, chunk_size: int = 1000) -> None:
    """
    Start a new ingestion cycle by initializing progress tracking with the next version number.
    Does not delete any existing data - just adds new rows for the new version.
    
    Args:
        conn: Database connection
        csv_dir: Directory containing CSV files
        chunk_size: Number of rows to load per chunk
    """
    logging.info("Starting new ingestion cycle...")
    
    try:
        # Initialize with next version number (no deletion, just append new rows)
        initialize_progress_tracking(conn, csv_dir, chunk_size)
        logging.info("New ingestion cycle initialized")
        
    except Exception as e:
        logging.error(f"Error starting new ingestion cycle: {e}")
        raise
