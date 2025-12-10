"""
Data transformation module for cleaning and preprocessing raw accident data.

This module loads data from raw database tables, performs cleaning operations,
and stores the processed data in the preprocessed_data table.
"""

import os
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from psycopg2.extensions import connection
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import argparse

from src.utils import logging
from src.utils.database import get_db_connection

load_dotenv()


def load_raw_data_from_db(conn: connection) -> Dict[str, pd.DataFrame]:
    """
    Load raw data from database tables into pandas DataFrames.
    
    Args:
        conn: Database connection
        
    Returns:
        Dictionary containing DataFrames for each table
    """
    logging.info("Loading raw data from database...")
    
    tables = {
        'caracteristics': 'SELECT * FROM raw_caracteristics',
        'holidays': 'SELECT * FROM raw_holidays',
        'places': 'SELECT * FROM raw_places',
        'users': 'SELECT * FROM raw_users',
        'vehicles': 'SELECT * FROM raw_vehicles'
    }
    
    dataframes = {}
    
    for table_name, query in tables.items():
        try:
            df = pd.read_sql_query(query, conn)
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            dataframes[table_name] = df
        except Exception as e:
            logging.error(f"Error loading {table_name}: {e}")
            raise
    
    return dataframes


def merge_raw_tables(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all raw tables using outer join on num_acc.
    
    Args:
        dataframes: Dictionary containing raw data DataFrames
        
    Returns:
        Merged DataFrame
    """
    logging.info("Merging raw tables...")
    
    # Start with caracteristics
    merged = dataframes['caracteristics'].copy()
    logging.info(f"Starting with caracteristics: {len(merged)} rows")
    
    # Merge places
    merged = pd.merge(merged, dataframes['places'], how='outer', on='num_acc')
    logging.info(f"After merging places: {len(merged)} rows")
    
    # Merge users
    merged = pd.merge(merged, dataframes['users'], how='outer', on='num_acc')
    logging.info(f"After merging users: {len(merged)} rows")
    
    # Merge vehicles (optional, depends on requirements)
    # Note: vehicles has multiple rows per num_acc, so we skip it for now
    # as the requirement focuses on user-level data
    
    logging.info(f"Final merged data: {len(merged)} rows, {len(merged.columns)} columns")
    
    return merged


def filter_metropolitan_areas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to keep only metropolitan areas (gps = 'M').
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    logging.info("Filtering for metropolitan areas (gps='M')...")
    
    initial_count = len(df)
    
    # Filter for metropolitan areas
    df_filtered = df[df['gps'] == 'M'].copy()
    
    filtered_count = len(df_filtered)
    logging.info(f"Filtered from {initial_count} to {filtered_count} rows ({filtered_count/initial_count*100:.1f}%)")
    
    return df_filtered


def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns and rename them to English.
    
    Relevant columns [F]: ['an', 'mois', 'hrmn', 'catu', 'grav', 'sexe', 'an_nais', 
                           'trajet', 'secu', 'lum', 'atm', 'catr', 'surf', 
                           'lat', 'long']
    Relevant columns [En]: ['Year', 'Month', 'Hour_Minute', 'User category', 'Severity', 'Sex', 
                            'Year of birth', 'Trip purpose', 'Security', 'Luminosity', 
                            'Weather', 'Type of road', 'Road surface', 
                            'Latitude', 'Longitude']
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with selected and renamed columns
    """
    logging.info("Selecting and renaming columns...")
    
    # Column mapping
    column_mapping = {
        'num_acc': 'num_acc',
        'user_id': 'raw_user_id',
        'an': 'Year',
        'mois': 'Month',
        'jour': 'jour',  # Keep for holiday matching, will be dropped later
        'hrmn': 'Hour_Minute',
        'catu': 'User_category',
        'grav': 'Severity',
        'sexe': 'Sex',
        'an_nais': 'Year_of_birth',
        'trajet': 'Trip_purpose',
        'secu': 'Security',
        'lum': 'Luminosity',
        'atm': 'Weather',
        'catr': 'Type_of_road',
        'surf': 'Road_surface',
        'lat': 'Latitude',
        'long': 'Longitude'
    }
    
    # Check which columns are available
    available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing_columns = set(column_mapping.keys()) - set(available_columns.keys())
    
    if missing_columns:
        logging.warning(f"Missing columns: {missing_columns}")
    
    # Select and rename columns
    df_selected = df[list(available_columns.keys())].copy()
    df_selected = df_selected.rename(columns=available_columns)
    
    logging.info(f"Selected {len(df_selected.columns)} columns")
    
    return df_selected


def clean_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing data in critical columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing data removed
    """
    logging.info("Removing rows with missing data...")
    
    initial_count = len(df)
    
    # Drop rows with any missing values
    df_cleaned = df.dropna().copy()
    
    cleaned_count = len(df_cleaned)
    removed_count = initial_count - cleaned_count
    
    logging.info(f"Removed {removed_count} rows with missing data ({removed_count/initial_count*100:.1f}%)")
    logging.info(f"Remaining rows: {cleaned_count}")
    
    return df_cleaned


def extract_hour_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour and minute from the Hour_Minute column.
    
    The hrmn field contains time in HMM or HHMM format (e.g., 715 = 7:15, 1430 = 14:30).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with Hour and Minute columns added
    """
    logging.info("Extracting hour and minute...")
    
    df = df.copy()
    
    if 'Hour_Minute' in df.columns:
        # Convert to string (no padding - we need to parse from the end)
        hrmn_str = df['Hour_Minute'].fillna(0).astype(int).astype(str)
        
        # Extract minute (last 2 digits) and hour (remaining characters)
        df['Minute'] = hrmn_str.str[-2:].astype(int)
        df['Hour'] = hrmn_str.str[:-2].replace('', '0').astype(int)
        
        # Validate hours (0-23) and minutes (0-59)
        invalid_hours = (df['Hour'] > 23) | (df['Hour'] < 0)
        invalid_minutes = (df['Minute'] > 59) | (df['Minute'] < 0)
        
        if invalid_hours.any() or invalid_minutes.any():
            logging.warning(f"Found {invalid_hours.sum()} invalid hours and {invalid_minutes.sum()} invalid minutes")
            # Set invalid times to NaN
            df.loc[invalid_hours, ['Hour', 'Minute']] = np.nan
            df.loc[invalid_minutes, ['Hour', 'Minute']] = np.nan
        
        # Drop the original Hour_Minute column
        df = df.drop(columns=['Hour_Minute'])
        
        logging.info(f"Extracted hour and minute successfully")
    
    return df


def convert_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert latitude and longitude to proper float format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with converted coordinates
    """
    logging.info("Converting coordinates...")
    
    df = df.copy()
    
    # Convert latitude
    if 'Latitude' in df.columns:
        # Handle string values like '-' and convert to numeric
        df['Latitude'] = pd.to_numeric(df['Latitude'].replace('-', np.nan), errors='coerce')
        # Convert from large integers to decimal degrees (divide by 10^5)
        df['Latitude'] = df['Latitude'] / 100000.0
    
    # Convert longitude
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'].replace('-', np.nan), errors='coerce')
        df['Longitude'] = df['Longitude'] / 100000.0
    
    logging.info("Coordinates converted successfully")
    
    return df


def add_holiday_column(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column indicating whether the accident occurred on a holiday.
    
    Args:
        df: Main DataFrame with Year, Month, and jour columns
        holidays_df: DataFrame containing holiday dates (ds column with YYYY-MM-DD format)
        
    Returns:
        DataFrame with holiday column added
    """
    logging.info("Adding holiday column...")
    
    df = df.copy()
    
    # Convert holiday dates to a set of date strings for efficient lookup
    if 'ds' in holidays_df.columns:
        # Convert ds column to datetime
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        holiday_dates = set(holidays_df['ds'].dt.date)
        
        logging.info(f"Loaded {len(holiday_dates)} unique holiday dates")
        
        # Create date column from Year, Month, and jour
        # Note: Year is 2-digit (16 = 2016), so we need to add 2000
        try:
            df['accident_date'] = pd.to_datetime(
                '20' + df['Year'].astype(str).str.zfill(2) + '-' +
                df['Month'].astype(str).str.zfill(2) + '-' +
                df['jour'].astype(str).str.zfill(2),
                format='%Y-%m-%d',
                errors='coerce'
            )
            
            # Check if accident date is in holiday dates
            df['holiday'] = df['accident_date'].dt.date.isin(holiday_dates)
            
            # Count matches
            holiday_count = df['holiday'].sum()
            total_count = len(df)
            logging.info(f"Matched {holiday_count} accidents ({holiday_count/total_count*100:.2f}%) to holidays")
            
            # Drop temporary date column
            df = df.drop(columns=['accident_date'])
            
        except Exception as e:
            logging.warning(f"Error matching holidays: {e}")
            logging.warning("Setting all holiday values to False")
            df['holiday'] = False
    else:
        logging.warning("'ds' column not found in holidays_df, setting all holiday values to False")
        df['holiday'] = False
    
    return df


def rearrange_severity_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rearrange severity values to be in ascending order.
    In the original data: 1=Unscathed, 2=Killed, 3=Hospitalized, 4=Light injury
    We want: 1=Unscathed, 2=Light injury, 3=Hospitalized, 4=Killed
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with rearranged severity values
    """
    logging.info("Rearranging severity values...")
    
    df = df.copy()
    
    if 'Severity' in df.columns:
        # Swap 2 and 4
        df['Severity'] = df['Severity'].replace({2: 4, 4: 2})
        logging.info("Severity values rearranged: 1=Unscathed, 2=Light injury, 3=Hospitalized, 4=Killed")
    
    return df


def transform_data_pipeline(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Execute the complete data transformation pipeline.
    
    Args:
        dataframes: Dictionary containing raw data DataFrames
        
    Returns:
        Cleaned and transformed DataFrame
    """
    logging.info("Starting data transformation pipeline...")
    
    # Step 1: Merge all tables
    merged_df = merge_raw_tables(dataframes)
    
    # Step 2: Filter for metropolitan areas
    filtered_df = filter_metropolitan_areas(merged_df)
    
    # Step 3: Select and rename relevant columns
    selected_df = select_and_rename_columns(filtered_df)
    
    # Step 4: Extract hour and minute from hrmn
    time_extracted_df = extract_hour_minute(selected_df)
    
    # Step 5: Convert coordinates
    converted_df = convert_coordinates(time_extracted_df)
    
    # Step 6: Remove rows with missing data
    cleaned_df = clean_missing_data(converted_df)
    
    # Step 7: Rearrange severity values
    final_df = rearrange_severity_values(cleaned_df)
    
    # Step 8: Add holiday column (requires jour)
    final_df = add_holiday_column(final_df, dataframes['holidays'])
    
    # Step 9: Drop temporary jour column (no longer needed)
    if 'jour' in final_df.columns:
        final_df = final_df.drop(columns=['jour'])
        logging.info("Dropped temporary 'jour' column")
    
    logging.info(f"Transformation pipeline completed: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    return final_df


def get_existing_records_hash(conn: connection) -> Dict[tuple, dict]:
    """
    Get hash of existing current records for change detection.
    
    Args:
        conn: Database connection
        
    Returns:
        Dictionary mapping (num_acc, raw_user_id) to record data
    """
    logging.info("Loading existing current records for change detection...")
    
    query = """
        SELECT num_acc, raw_user_id, year, month, hour, minute, user_category, 
               severity, sex, year_of_birth, trip_purpose, security, 
               luminosity, weather, type_of_road, road_surface, 
               latitude, longitude, holiday, record_id
        FROM clean_data 
        WHERE is_current = TRUE
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Create dictionary with composite key (num_acc, raw_user_id)
    records_dict = {}
    for _, row in df.iterrows():
        key = (row['num_acc'], row['raw_user_id'])
        records_dict[key] = row.to_dict()
    
    logging.info(f"Loaded {len(records_dict)} existing current records")
    return records_dict


def records_are_equal(existing: dict, new: pd.Series) -> bool:
    """
    Compare existing and new records to detect changes.
    
    Args:
        existing: Existing record as dictionary
        new: New record as pandas Series
        
    Returns:
        True if records are identical, False otherwise
    """
    # Compare all data columns (excluding metadata like record_id, timestamps, etc.)
    comparison_columns = [
        'year', 'month', 'hour', 'minute', 'user_category', 
        'severity', 'sex', 'year_of_birth', 'trip_purpose', 'security', 
        'luminosity', 'weather', 'type_of_road', 'road_surface', 
        'latitude', 'longitude', 'holiday'
    ]
    
    for col in comparison_columns:
        existing_val = existing.get(col)
        new_val = new.get(col)
        
        # Handle NaN comparisons
        if pd.isna(existing_val) and pd.isna(new_val):
            continue
        if pd.isna(existing_val) or pd.isna(new_val):
            return False
        
        # Handle float comparisons with tolerance
        if isinstance(existing_val, float) and isinstance(new_val, float):
            if not np.isclose(existing_val, new_val, rtol=1e-9, atol=1e-9):
                return False
        else:
            if existing_val != new_val:
                return False
    
    return True


def invalidate_old_records(conn: connection, record_ids: list) -> int:
    """
    Mark old records as no longer current (SCD Type 2).
    
    Args:
        conn: Database connection
        record_ids: List of record IDs to invalidate
        
    Returns:
        Number of records invalidated
    """
    if not record_ids:
        return 0
    
    logging.info(f"Invalidating {len(record_ids)} old records...")
    
    cursor = conn.cursor()
    
    try:
        update_query = """
            UPDATE clean_data 
            SET is_current = FALSE, 
                valid_to = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE record_id = ANY(%s) AND is_current = TRUE
        """
        
        cursor.execute(update_query, (record_ids,))
        conn.commit()
        invalidated_count = cursor.rowcount
        logging.info(f"Invalidated {invalidated_count} records")
        return invalidated_count
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error invalidating old records: {e}")
        raise
    finally:
        cursor.close()


def insert_preprocessed_data(conn: connection, df: pd.DataFrame, batch_size: int = 1000) -> Dict[str, int]:
    """
    Insert preprocessed data into the database using SCD Type 2.
    
    This function implements Slowly Changing Dimensions Type 2:
    - Existing records that haven't changed: skip
    - Existing records that have changed: mark old as is_current=FALSE, insert new version
    - New records: insert with is_current=TRUE
    
    Args:
        conn: Database connection
        df: DataFrame containing preprocessed data
        batch_size: Number of rows to insert per batch
        
    Returns:
        Dictionary with counts of inserted, updated, and unchanged records
    """
    logging.info(f"Processing {len(df)} rows for clean_data table with SCD Type 2...")
    
    # Prepare column mapping to match database schema
    column_mapping = {
        'num_acc': 'num_acc',
        'raw_user_id': 'raw_user_id',
        'Year': 'year',
        'Month': 'month',
        'Hour': 'hour',
        'Minute': 'minute',
        'User_category': 'user_category',
        'Severity': 'severity',
        'Sex': 'sex',
        'Year_of_birth': 'year_of_birth',
        'Trip_purpose': 'trip_purpose',
        'Security': 'security',
        'Luminosity': 'luminosity',
        'Weather': 'weather',
        'Type_of_road': 'type_of_road',
        'Road_surface': 'road_surface',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'holiday': 'holiday'
    }
    
    # Rename columns to match database schema
    df_to_insert = df.rename(columns=column_mapping)
    
    # Get existing records
    existing_records = get_existing_records_hash(conn)
    
    # Classify records into: new, changed, unchanged
    records_to_insert = []
    records_to_invalidate = []
    unchanged_count = 0
    
    for _, row in df_to_insert.iterrows():
        key = (row['num_acc'], row['raw_user_id'])
        
        if key in existing_records:
            # Record exists - check if it changed
            if records_are_equal(existing_records[key], row):
                # No change - skip
                unchanged_count += 1
            else:
                # Changed - invalidate old and insert new
                records_to_invalidate.append(existing_records[key]['record_id'])
                records_to_insert.append(row)
        else:
            # New record - insert
            records_to_insert.append(row)
    
    logging.info(f"Records analysis: {len(records_to_insert)} to insert, "
                f"{len(records_to_invalidate)} to update, {unchanged_count} unchanged")
    
    # Invalidate changed records
    invalidated_count = invalidate_old_records(conn, records_to_invalidate)
    
    # Insert new and changed records
    if records_to_insert:
        insert_query = """
            INSERT INTO clean_data 
            (num_acc, raw_user_id, year, month, hour, minute, user_category, severity, sex, 
             year_of_birth, trip_purpose, security, luminosity, weather, 
             type_of_road, road_surface, latitude, longitude, holiday,
             is_current, valid_from, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        
        # Convert records to list of tuples
        data_tuples = [
            (
                row['num_acc'], row['raw_user_id'], row['year'], row['month'], 
                row['hour'], row['minute'], row['user_category'], row['severity'], 
                row['sex'], row['year_of_birth'], row['trip_purpose'], row['security'], 
                row['luminosity'], row['weather'], row['type_of_road'], row['road_surface'], 
                row['latitude'], row['longitude'], row['holiday']
            )
            for row in records_to_insert
        ]
        
        cursor = conn.cursor()
        
        try:
            execute_batch(cursor, insert_query, data_tuples, page_size=batch_size)
            conn.commit()
            inserted_count = len(data_tuples)
            logging.info(f"Successfully inserted {inserted_count} clean records")
        except Exception as e:
            conn.rollback()
            logging.error(f"Error inserting clean data: {e}")
            raise
        finally:
            cursor.close()
    else:
        inserted_count = 0
        logging.info("No new records to insert")
    
    return {
        'inserted': inserted_count,
        'updated': invalidated_count,
        'unchanged': unchanged_count
    }


def clear_preprocessed_data(conn: connection) -> None:
    """
    Clear existing data from clean_data table using TRUNCATE for better performance.
    
    Args:
        conn: Database connection
    """
    logging.info("Clearing existing clean data...")
    
    cursor = conn.cursor()
    
    try:
        # Use TRUNCATE instead of DELETE for better performance and to reset sequences
        cursor.execute("TRUNCATE TABLE clean_data RESTART IDENTITY CASCADE")
        conn.commit()
        logging.info("Clean data cleared successfully")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error clearing clean data: {e}")
        raise
    finally:
        cursor.close()


def transform_data(clear_existing: bool = False) -> Dict[str, Any]:
    """
    Main function to transform raw data into clean preprocessed data.
    
    Args:
        clear_existing: Whether to clear existing preprocessed data before inserting
        
    Returns:
        Dictionary with transformation results
    """
    logging.info("Starting data transformation process...")
    
    conn = get_db_connection()
    
    try:
        # Step 1: Load raw data from database
        dataframes = load_raw_data_from_db(conn)
        
        # Step 2: Transform data
        transformed_df = transform_data_pipeline(dataframes)
        
        # Step 3: Clear existing clean data if requested
        if clear_existing:
            clear_preprocessed_data(conn)
        
        # Step 4: Insert clean data with SCD Type 2
        result_counts = insert_preprocessed_data(conn, transformed_df)
        
        logging.info("Data transformation completed successfully!")
        logging.info(f"Summary: {result_counts['inserted']} inserted, "
                    f"{result_counts['updated']} updated, {result_counts['unchanged']} unchanged")
        
        return {
            'success': True,
            'rows_processed': len(transformed_df),
            'rows_inserted': result_counts['inserted'],
            'rows_updated': result_counts['updated'],
            'rows_unchanged': result_counts['unchanged'],
            'message': 'Data transformation completed successfully'
        }
        
    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Data transformation failed'
        }
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")


def main(clear_existing: bool = False):
    """Entry point for the script."""
    try:
        result = transform_data(clear_existing=clear_existing)
        if not result.get('success', False):
            logging.error(f"Transformation failed: {result.get('message')}")
            raise Exception(result.get('error', 'Unknown error'))
        else:
            logging.info(f"Transformation successful: {result}")
    except Exception as e:
        logging.error(f"Failed to transform data: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform raw data into clean preprocessed data using SCD Type 2.",
        epilog="Note: This script implements Slowly Changing Dimensions Type 2. "
               "Unchanged records are skipped, changed records create new versions with history, "
               "and new records are inserted. Use --clear-all to truncate and start fresh."
    )
    parser.add_argument(
        '--clear-all',
        action='store_true',
        help="Clear all existing data before inserting (use for complete refresh)"
    )
    
    args = parser.parse_args()
    
    # With SCD Type 2, we typically don't clear existing data
    # unless explicitly requested for a complete refresh
    main(clear_existing=args.clear_all)