"""
Accidents Data Pipeline DAG
===========================

This DAG handles only the data processing pipeline, from download to 
data splitting. Use this when you need to refresh data without retraining.

Pipeline Stages:
1. Download: Download raw data from Kaggle
2. Ingest: Load raw CSV data into PostgreSQL
3. Clean: Transform and clean data using SCD Type 2
4. Split: Assign train/validation/test splits
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import kagglehub as kh
from dotenv import load_dotenv

from airflow.sdk import dag, task, setup, teardown, Variable
from airflow.sdk.bases.operator import chain

from src.utils import logging
from src.utils.database import get_db_connection
from src.data.ingest_data import load_next_chunk, DEFAULT_CHUNK_SIZE
from src.data.clean_data import transform_data, assign_dataset_splits

load_dotenv()

# ########### Configuration ###########
RANDOM_STATE = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


@dag(
    dag_id="accidents_data_pipeline",
    schedule=None,  # Manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'data-pipeline', 'etl'],
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }
)
def accidents_data_pipeline():
    """Data Pipeline DAG - ETL for accidents data."""

    @setup
    def initialize():
        """Initialize the data pipeline configuration."""
        logging.info('Initializing Data Pipeline DAG')
        
        Variable.set('train_ratio', str(TRAIN_RATIO))
        Variable.set('val_ratio', str(VAL_RATIO))
        Variable.set('test_ratio', str(TEST_RATIO))
        Variable.set('random_state', str(RANDOM_STATE))
        
        return {
            'status': 'initialized',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'train_ratio': TRAIN_RATIO,
                'val_ratio': VAL_RATIO,
                'test_ratio': TEST_RATIO
            }
        }

    @task()
    def download_data() -> Dict[str, Any]:
        """Download raw accident data from Kaggle."""
        logging.info('Downloading dataset from Kaggle...')
        
        dataset_id = 'ahmedlahlou/accidents-in-france-from-2005-to-2016'
        
        try:
            downloaded_path = kh.dataset_download(dataset_id)
            logging.info(f'Downloaded to: {downloaded_path}')
        except Exception as e:
            raise Exception(f"Failed to download dataset: {e}")
        
        # Move to timestamped directory
        raw_data_path = os.getenv('RAW_DATA_PATH', '/app/data/raw/')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        target_dir = Path(raw_data_path) / timestamp
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy downloaded files
        src_path = Path(downloaded_path)
        if src_path.is_dir():
            for item in src_path.iterdir():
                dest = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(src_path, target_dir / src_path.name)
        
        logging.info(f'Data saved to: {target_dir}')
        
        return {
            'status': 'success',
            'source_path': str(downloaded_path),
            'target_path': str(target_dir),
            'timestamp': timestamp
        }

    @task()
    def ingest_data(download_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest raw CSV data into PostgreSQL database."""
        logging.info('Starting data ingestion (chunked mode)...')
        logging.info(f'Source: {download_result.get("target_path")}')
        
        total_records = {
            'caracteristics': 0,
            'places': 0,
            'users': 0,
            'vehicles': 0,
            'holidays': 0
        }
        chunks_processed = 0
        all_complete = False
        
        while not all_complete:
            result = load_next_chunk(chunk_size=DEFAULT_CHUNK_SIZE)
            
            if not result.get('success', False):
                error_msg = result.get('message', 'Unknown error')
                # Check if it's just "no more data" which is expected at the end
                if 'complete' in error_msg.lower() or 'no more' in error_msg.lower():
                    all_complete = True
                    break
                raise Exception(f"Ingestion failed: {error_msg}")
            
            # Accumulate counts
            tables = result.get('tables', {})
            for table, count in tables.items():
                total_records[table] = total_records.get(table, 0) + count
            
            chunks_processed += 1
            all_complete = result.get('all_complete', False)
            
            logging.info(f'Chunk {chunks_processed} processed: {tables}')
        
        logging.info(f'Data ingestion completed. Total: {total_records}')
        
        return {
            'status': 'success',
            'chunks_processed': chunks_processed,
            'total_records': total_records,
            'download_timestamp': download_result.get('timestamp')
        }

    @task()
    def clean_data(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and transform raw data using SCD Type 2 logic."""
        logging.info('Starting data cleaning and transformation...')
        logging.info(f'Processing {ingest_result.get("chunks_processed")} chunks of data')
        
        result = transform_data(clear_existing=False)
        
        if not result.get('success', False):
            raise Exception(f"Data transformation failed: {result.get('error', 'Unknown error')}")
        
        clean_stats = {
            'rows_processed': result.get('rows_processed', 0),
            'rows_inserted': result.get('rows_inserted', 0),
            'rows_updated': result.get('rows_updated', 0),
            'rows_unchanged': result.get('rows_unchanged', 0),
        }
        
        logging.info(f'Cleaning completed: {clean_stats}')
        
        return {
            'status': 'success',
            'stats': clean_stats,
            'message': result.get('message', 'Transformation completed')
        }

    @task()
    def assign_splits(clean_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assign train/validation/test splits using stratified sampling."""
        logging.info('Assigning dataset splits (stratified by severity)...')
        
        with get_db_connection() as conn:
            split_counts = assign_dataset_splits(
                conn,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO,
                random_state=RANDOM_STATE
            )
            
            logging.info(f'Split assignment completed: {split_counts}')
            
            total = sum(split_counts.values()) if split_counts else 0
            
            return {
                'status': 'success',
                'split_counts': split_counts,
                'total_records': total,
                'ratios': {
                    'train': TRAIN_RATIO,
                    'validation': VAL_RATIO,
                    'test': TEST_RATIO
                }
            }

    @task()
    def validate_data(split_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the processed data quality and splits."""
        logging.info('Validating processed data...')
        
        with get_db_connection() as conn:
            # Check total records
            total_query = "SELECT COUNT(*) as count FROM clean_data WHERE is_current = TRUE"
            total_df = pd.read_sql_query(total_query, conn)
            total_count = int(total_df['count'].iloc[0])
            
            # Check split distribution
            split_query = """
                SELECT dataset_split, COUNT(*) as count 
                FROM clean_data 
                WHERE is_current = TRUE 
                GROUP BY dataset_split
            """
            split_df = pd.read_sql_query(split_query, conn)
            split_dist = dict(zip(split_df['dataset_split'], split_df['count']))
            
            # Check class distribution per split
            class_query = """
                SELECT dataset_split, severity, COUNT(*) as count 
                FROM clean_data 
                WHERE is_current = TRUE 
                GROUP BY dataset_split, severity
                ORDER BY dataset_split, severity
            """
            class_df = pd.read_sql_query(class_query, conn)
            
            # Check for any NULL splits
            null_query = "SELECT COUNT(*) as count FROM clean_data WHERE is_current = TRUE AND dataset_split IS NULL"
            null_df = pd.read_sql_query(null_query, conn)
            null_count = int(null_df['count'].iloc[0])
            
            validation_result = {
                'status': 'success',
                'total_records': total_count,
                'split_distribution': split_dist,
                'null_splits': null_count,
                'class_distribution_by_split': class_df.to_dict('records'),
                'is_valid': null_count == 0 and total_count > 0
            }
            
            if not validation_result['is_valid']:
                logging.warning('Data validation issues detected!')
                logging.warning(f'Null splits: {null_count}, Total records: {total_count}')
            else:
                logging.info('Data validation passed!')
                logging.info(f'Total: {total_count}, Splits: {split_dist}')
            
            return validation_result

    @teardown()
    def finalize():
        """Clean up and finalize the data pipeline."""
        logging.info('Finalizing Data Pipeline DAG')
        
        try:
            Variable.delete('train_ratio')
            Variable.delete('val_ratio')
            Variable.delete('test_ratio')
            Variable.delete('random_state')
        except Exception as e:
            logging.warning(f'Cleanup warning: {e}')
        
        logging.info('Data pipeline completed successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # Define DAG Flow
    # ==============================
    init_task = initialize()
    download_task = download_data()
    ingest_task = ingest_data(download_task)
    clean_task = clean_data(ingest_task)
    split_task = assign_splits(clean_task)
    validate_task = validate_data(split_task)
    final_task = finalize()
    
    chain(init_task, download_task, ingest_task, clean_task, split_task, validate_task, final_task)


# Instantiate the DAG
accidents_data_pipeline()
