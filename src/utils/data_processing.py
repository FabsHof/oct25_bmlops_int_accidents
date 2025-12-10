"""
Data processing utilities for DataFrame transformations and type conversions.

This module provides reusable functions for cleaning and preparing data
for database insertion across different tables.
"""

import pandas as pd
from typing import List, Dict, Any


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by converting column names to lowercase.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with lowercase column names
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower()
    return df_clean


def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric type, handling NaN values.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted numeric columns
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].round().astype('Int64').replace({pd.NA: None})
    return df_clean


def convert_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to string type, replacing 'nan' with None.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted string columns
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).replace({'nan': None})
    return df_clean


def prepare_data_tuples(df: pd.DataFrame, columns: List[str]) -> List[tuple]:
    """
    Convert DataFrame to list of tuples for database insertion.
    
    Args:
        df: Input DataFrame
        columns: List of column names to extract in order
        
    Returns:
        List of tuples containing row data
    """
    return [tuple(x) for x in df[columns].values]


def load_dataframe_chunk(csv_path, offset: int, chunk_size: int, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load a chunk of data from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        offset: Starting row (0-indexed, excluding header)
        chunk_size: Number of rows to load
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        DataFrame with the loaded chunk
    """
    try:
        # Skip rows from 1 to offset (keeping header at 0)
        if offset > 0:
            df = pd.read_csv(csv_path, skiprows=range(1, offset + 1), nrows=chunk_size, encoding=encoding)
        else:
            df = pd.read_csv(csv_path, nrows=chunk_size, encoding=encoding)
        return df
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        if offset > 0:
            df = pd.read_csv(csv_path, skiprows=range(1, offset + 1), nrows=chunk_size, encoding='latin-1')
        else:
            df = pd.read_csv(csv_path, nrows=chunk_size, encoding='latin-1')
        return df


# Table-specific column configurations
TABLE_COLUMNS = {
    'caracteristics': {
        'numeric': ['num_acc', 'an', 'mois', 'jour', 'hrmn', 'lum', 'agg', 'int', 'atm', 'col', 'com', 'lat', 'dep'],
        'string': ['adr', 'gps', 'long'],
        'db_columns': ['num_acc', 'an', 'mois', 'jour', 'hrmn', 'lum', 'agg', 'int', 'atm', 'col', 'com', 'adr', 'gps', 'lat', 'long', 'dep']
    },
    'holidays': {
        'numeric': [],
        'string': ['ds', 'holiday'],
        'db_columns': ['ds', 'holiday']
    },
    'places': {
        'numeric': ['num_acc', 'catr', 'voie', 'v1', 'circ', 'nbv', 'pr', 'pr1', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1'],
        'string': ['v2'],
        'db_columns': ['num_acc', 'catr', 'voie', 'v1', 'v2', 'circ', 'nbv', 'pr', 'pr1', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'env1']
    },
    'users': {
        'numeric': ['num_acc', 'place', 'catu', 'grav', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp', 'an_nais'],
        'string': ['num_veh'],
        'db_columns': ['num_acc', 'place', 'catu', 'grav', 'sexe', 'trajet', 'secu', 'locp', 'actp', 'etatp', 'an_nais', 'num_veh']
    },
    'vehicles': {
        'numeric': ['num_acc', 'senc', 'catv', 'occutc', 'obs', 'obsm', 'choc', 'manv'],
        'string': ['num_veh'],
        'db_columns': ['num_acc', 'senc', 'catv', 'occutc', 'obs', 'obsm', 'choc', 'manv', 'num_veh']
    }
}


def prepare_table_data(df: pd.DataFrame, table_name: str) -> List[tuple]:
    """
    Prepare DataFrame for a specific table by applying appropriate transformations.
    
    Args:
        df: Input DataFrame
        table_name: Name of the table (must be in TABLE_COLUMNS)
        
    Returns:
        List of tuples ready for database insertion
        
    Raises:
        ValueError: If table_name is not recognized
    """
    if table_name not in TABLE_COLUMNS:
        raise ValueError(f"Unknown table: {table_name}. Must be one of {list(TABLE_COLUMNS.keys())}")
    
    config = TABLE_COLUMNS[table_name]
    
    # Clean column names
    df_clean = clean_dataframe_columns(df)
    
    # Convert numeric columns
    if config['numeric']:
        df_clean = convert_numeric_columns(df_clean, config['numeric'])
    
    # Convert string columns
    if config['string']:
        df_clean = convert_string_columns(df_clean, config['string'])
    
    # Prepare tuples
    return prepare_data_tuples(df_clean, config['db_columns'])
