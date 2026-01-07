"""
Accidents ML Pipeline DAG
=========================

This DAG implements the complete ML pipeline for traffic accidents severity prediction,
including data processing, model training, evaluation, and drift monitoring.

Pipeline Stages:
1. Data Pipeline:
   - download_data: Download raw data from Kaggle
   - ingest_data: Ingest data into PostgreSQL database
   - clean_data: Clean and preprocess data with SCD Type 2
   - assign_splits: Assign train/validation/test splits

2. ML Pipeline:
   - setup_mlflow: Configure MLflow tracking
   - prepare_datasets: Load and version datasets
   - train_model: Train Random Forest with GridSearchCV
   - evaluate_model: Evaluate on validation and test sets
   - register_model: Compare with champion and register best model

3. Monitoring:
   - generate_drift_report: Generate data drift report using Evidently
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import mlflow
import kagglehub as kh
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from airflow.sdk import dag, task, setup, task_group, teardown, Variable
from airflow.sdk.bases.operator import chain

from src.utils import logging
from src.utils.database import get_db_connection
from src.data.ingest_data import load_next_chunk, DEFAULT_CHUNK_SIZE
from src.data.clean_data import transform_data, assign_dataset_splits
from src.models.metrics import (
    weighted_f1_score_metric,
    weighted_precision_metric,
    weighted_recall_metric,
    roc_auc_ovr_metric,
    per_class_f1_metric
)

load_dotenv()

# ########### Configuration ###########
MODEL_NAME = 'random_forest_model'
CHAMPION_MODEL_ALIAS = 'champion'
BEST_MODEL_METRIC = 'weighted_f1_score'
RANDOM_STATE = 42

CLASS_LABELS = {
    1: 'Unscathed',
    2: 'Light injury',
    3: 'Hospitalized wounded',
    4: 'Killed'
}

FEATURE_COLUMNS = [
    'year', 'month', 'hour', 'minute', 'user_category', 'sex',
    'year_of_birth', 'trip_purpose', 'security', 'luminosity',
    'weather', 'type_of_road', 'road_surface', 'latitude', 'longitude', 'holiday'
]
TARGET_COLUMN = 'severity'

NUMERICAL_FEATURES = [
    'year', 'month', 'hour', 'minute', 'year_of_birth', 'latitude', 'longitude'
]
CATEGORICAL_FEATURES = [
    'user_category', 'sex', 'trip_purpose', 'security',
    'luminosity', 'weather', 'type_of_road', 'road_surface', 'holiday'
]


# ########### Helper Functions ###########

def setup_mlflow_tracking() -> str:
    """Set up MLflow tracking and return tracking URI."""
    tracking_uri = f'http://{os.getenv("MLFLOW_HOST", "mlflow")}:{os.getenv("MLFLOW_PORT", "5001")}'
    mlflow.set_tracking_uri(tracking_uri)

    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ROOT_USER', 'minio_user')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_ROOT_PASSWORD', 'minio_password')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{os.getenv("MINIO_HOST", "minio")}:{os.getenv("MINIO_PORT", "9000")}'
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    mlflow.set_experiment('Car Accident Severity Prediction')
    logging.info(f'MLflow tracking set up at {tracking_uri}')
    logging.info(f'MLflow S3 endpoint: {os.environ["MLFLOW_S3_ENDPOINT_URL"]}')
    return tracking_uri


def load_training_data_from_db(dataset_split: str = 'train') -> pd.DataFrame:
    """Load training data from database for specified dataset split."""
    logging.info(f"Loading {dataset_split} data from database...")
    with get_db_connection() as conn:
        query = f"""
            SELECT {', '.join(FEATURE_COLUMNS + [TARGET_COLUMN])}
            FROM clean_data
            WHERE dataset_split = '{dataset_split}' AND is_current = TRUE
        """
        df = pd.read_sql_query(query, conn)
        logging.info(f"Loaded {len(df)} records for {dataset_split} set")
        
        # Log class distribution
        if TARGET_COLUMN in df.columns:
            class_dist = df[TARGET_COLUMN].value_counts().sort_index()
            logging.info(f"{dataset_split} set class distribution:")
            for severity, count in class_dist.items():
                class_name = CLASS_LABELS.get(severity, f'Class {severity}')
                logging.info(f"  {class_name} (severity={severity}): {count} ({count/len(df)*100:.1f}%)")
        
        return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features (X) and target (y) from dataframe."""
    df_clean = df.dropna()
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        logging.warning(f"Dropped {dropped} rows with missing values ({dropped/len(df)*100:.1f}%)")
    
    X = df_clean[FEATURE_COLUMNS].copy().astype('float64')
    y = df_clean[TARGET_COLUMN].copy()
    return X, y


# ########### DAG Definition ###########

@dag(
    dag_id="accidents_ml_pipeline",
    schedule='@daily',  # Daily trigger
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'ml-pipeline'],
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }
)
def accidents_ml_pipeline():
    """Main Accidents ML Pipeline DAG."""

    @setup
    def initialize():
        """Initialize environment variables and configurations for the DAG."""
        logging.info('Initializing Accidents ML Pipeline DAG')
        
        # Store configuration in Airflow Variables
        Variable.set('model_name', MODEL_NAME)
        Variable.set('champion_alias', CHAMPION_MODEL_ALIAS)
        Variable.set('best_model_metric', BEST_MODEL_METRIC)
        Variable.set('random_state', str(RANDOM_STATE))
        Variable.set('feature_columns', json.dumps(FEATURE_COLUMNS))
        Variable.set('target_column', TARGET_COLUMN)
        
        logging.info('Configuration stored in Airflow Variables')
        return {'status': 'initialized', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # Data Pipeline Tasks
    # ==============================
    @task_group()
    def data_pipeline():
        """Data ingestion and preprocessing pipeline."""

        @task()
        def download_data() -> Dict[str, Any]:
            """Download raw data from Kaggle."""
            logging.info('Downloading dataset from Kaggle...')
            
            dataset_id = 'ahmedlahlou/accidents-in-france-from-2005-to-2016'
            file_path = kh.dataset_download(dataset_id)
            
            raw_data_path = os.getenv('RAW_DATA_PATH', '/app/data/raw/')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            target_path = Path(raw_data_path) / timestamp
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Move downloaded file to target directory
            new_file_path = target_path / os.path.basename(file_path)
            shutil.move(file_path, new_file_path)
            
            logging.info(f'Data saved to: {new_file_path}')
            return {'file_path': str(new_file_path), 'timestamp': timestamp}

        @task()
        def ingest_data(download_result: Dict[str, Any]) -> Dict[str, Any]:
            """Ingest raw CSV data into the database."""
            logging.info('Starting data ingestion (chunked mode)...')
            
            all_complete = False
            total_loaded = {'caracteristics': 0, 'places': 0, 'users': 0, 'vehicles': 0, 'holidays': 0}
            
            while not all_complete:
                result = load_next_chunk(chunk_size=DEFAULT_CHUNK_SIZE)
                if not result.get('success', False):
                    raise Exception(f"Chunk loading failed: {result.get('message')}")
                
                for table, count in result.get('tables', {}).items():
                    total_loaded[table] = total_loaded.get(table, 0) + count
                
                all_complete = result.get('all_complete', False)
                logging.info(f"Chunk loaded: {result.get('tables')}")
            
            logging.info(f'Data ingestion completed. Total records: {total_loaded}')
            return {'status': 'completed', 'total_loaded': total_loaded}

        @task()
        def clean_data(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
            """Clean and preprocess raw data using SCD Type 2."""
            logging.info('Starting data cleaning and transformation...')
            
            result = transform_data(clear_existing=False)
            
            if not result.get('success', False):
                raise Exception(f"Data transformation failed: {result.get('error')}")
            
            logging.info(f'Data cleaning completed: {result}')
            return result

        @task()
        def assign_splits(clean_result: Dict[str, Any]) -> Dict[str, Any]:
            """Assign train/validation/test splits to the cleaned data."""
            logging.info('Assigning dataset splits...')
            
            with get_db_connection() as conn:
                result = assign_dataset_splits(
                    conn,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    random_state=RANDOM_STATE
                )
                logging.info(f'Dataset splits assigned: {result}')
                return {'status': 'completed', 'splits': result}

        # Define data pipeline flow
        download_result = download_data()
        ingest_result = ingest_data(download_result)
        clean_result = clean_data(ingest_result)
        splits_result = assign_splits(clean_result)
        
        return splits_result

    # ==============================
    # ML Pipeline Tasks
    # ==============================
    @task_group()
    def ml_pipeline():
        """Model training, evaluation, and registration pipeline."""

        @task()
        def prepare_datasets() -> Dict[str, Any]:
            """Load and prepare training, validation, and test datasets."""
            logging.info('Preparing datasets for training...')
            
            setup_mlflow_tracking()
            
            with get_db_connection() as conn:
                # Load metadata for versioning
                meta_query = """
                    SELECT id FROM data_ingestion_progress
                    ORDER BY id DESC LIMIT 1
                """
                meta_df = pd.read_sql_query(meta_query, conn)
                data_version = int(meta_df['id'].iloc[0]) if len(meta_df) > 0 else 1
            
            # Load datasets
            train_df = load_training_data_from_db('train')
            val_df = load_training_data_from_db('validation')
            test_df = load_training_data_from_db('test')
            
            # Log dataset info
            dataset_info = {
                'data_version': data_version,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'feature_columns': FEATURE_COLUMNS,
                'target_column': TARGET_COLUMN
            }
            
            logging.info(f'Datasets prepared: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
            return dataset_info

        @task()
        def train_model(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
            """Train Random Forest model with GridSearchCV hyperparameter tuning."""
            logging.info('Starting model training with GridSearchCV...')
            
            setup_mlflow_tracking()
            
            # Load training data
            train_df = load_training_data_from_db('train')
            X_train, y_train = prepare_features_and_target(train_df)
            
            with mlflow.start_run(run_name='model_training') as run:
                # Log dataset info
                mlflow.log_params({
                    'data_version': dataset_info['data_version'],
                    'train_size': dataset_info['train_size'],
                    'n_features': len(FEATURE_COLUMNS),
                })
                
                # Create versioned dataset
                train_data = pd.concat([X_train, y_train], axis=1)
                train_dataset = mlflow.data.from_pandas(
                    train_data,
                    name=f"training_data-v{dataset_info['data_version']}",
                    targets=TARGET_COLUMN,
                )
                mlflow.log_input(train_dataset, context="training")
                
                # Hyperparameter grid
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                }
                
                # Train with GridSearchCV
                model = RandomForestClassifier(random_state=RANDOM_STATE)
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                mlflow.log_params(best_params)
                
                # Log model
                model_info = mlflow.sklearn.log_model(
                    best_model,
                    MODEL_NAME,
                    registered_model_name=MODEL_NAME
                )
                
                logging.info(f'Model trained with best params: {best_params}')
                logging.info(f'Model URI: {model_info.model_uri}')
                
                return {
                    'run_id': run.info.run_id,
                    'model_uri': model_info.model_uri,
                    'model_version': model_info.registered_model_version,
                    'best_params': best_params,
                    'data_version': dataset_info['data_version']
                }

        @task()
        def evaluate_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate model on validation and test datasets."""
            logging.info('Evaluating model on validation and test sets...')
            
            setup_mlflow_tracking()
            model_uri = training_result['model_uri']
            data_version = training_result['data_version']
            
            # Load datasets
            val_df = load_training_data_from_db('validation')
            test_df = load_training_data_from_db('test')
            
            X_val, y_val = prepare_features_and_target(val_df)
            X_test, y_test = prepare_features_and_target(test_df)
            
            extra_metrics = [
                weighted_f1_score_metric,
                weighted_precision_metric,
                weighted_recall_metric,
                roc_auc_ovr_metric,
                per_class_f1_metric
            ]
            
            with mlflow.start_run(run_name='model_evaluation'):
                # Create versioned datasets
                val_data = pd.concat([X_val, y_val], axis=1)
                val_dataset = mlflow.data.from_pandas(
                    val_data,
                    name=f"validation_data-v{data_version}",
                    targets=TARGET_COLUMN,
                )
                
                test_data = pd.concat([X_test, y_test], axis=1)
                test_dataset = mlflow.data.from_pandas(
                    test_data,
                    name=f"test_data-v{data_version}",
                    targets=TARGET_COLUMN,
                )
                
                # Evaluate on validation set
                val_result = mlflow.models.evaluate(
                    model_uri,
                    val_dataset,
                    model_type="classifier",
                    evaluator_config={'log_explainer': True},
                    extra_metrics=extra_metrics
                )
                
                # Evaluate on test set
                test_result = mlflow.models.evaluate(
                    model_uri,
                    test_dataset,
                    model_type="classifier",
                    evaluator_config={'log_explainer': True},
                    extra_metrics=extra_metrics
                )
                
                eval_metrics = {
                    'validation': {k: float(v) for k, v in val_result.metrics.items() if isinstance(v, (int, float))},
                    'test': {k: float(v) for k, v in test_result.metrics.items() if isinstance(v, (int, float))}
                }
                
                logging.info(f'Validation metrics: {eval_metrics["validation"]}')
                logging.info(f'Test metrics: {eval_metrics["test"]}')
                
                return {
                    'model_uri': model_uri,
                    'model_version': training_result['model_version'],
                    'validation_metrics': eval_metrics['validation'],
                    'test_metrics': eval_metrics['test'],
                    'best_metric_value': test_result.metrics.get(BEST_MODEL_METRIC, 0.0)
                }

        @task()
        def register_and_alias_model(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
            """Compare with champion model and register as new champion if better."""
            logging.info('Comparing model with champion and registering...')
            
            setup_mlflow_tracking()
            
            model_uri = evaluation_result['model_uri']
            model_version = evaluation_result['model_version']
            current_metric = evaluation_result['best_metric_value']
            
            # Check for existing champion
            champion_model_uri = f'models:/{MODEL_NAME}@{CHAMPION_MODEL_ALIAS}'
            champion_exists = False
            champion_metric = None
            
            try:
                mlflow.models.get_model_info(champion_model_uri)
                champion_exists = True
                logging.info('Found existing champion model')
                
                # Evaluate champion on test set to get comparable metrics
                test_df = load_training_data_from_db('test')
                X_test, y_test = prepare_features_and_target(test_df)
                test_data = pd.concat([X_test, y_test], axis=1)
                
                extra_metrics = [
                    weighted_f1_score_metric,
                    weighted_precision_metric,
                    weighted_recall_metric,
                    roc_auc_ovr_metric,
                    per_class_f1_metric
                ]
                
                test_dataset = mlflow.data.from_pandas(
                    test_data,
                    name="test_data_champion_eval",
                    targets=TARGET_COLUMN,
                )
                
                champion_result = mlflow.models.evaluate(
                    champion_model_uri,
                    test_dataset,
                    model_type="classifier",
                    extra_metrics=extra_metrics
                )
                champion_metric = champion_result.metrics.get(BEST_MODEL_METRIC, 0.0)
                
            except Exception as e:
                logging.info(f'No existing champion model found: {e}')
                champion_exists = False
            
            # Compare and decide
            is_current_better = False
            if not champion_exists:
                is_current_better = True
            elif current_metric is not None and champion_metric is not None:
                is_current_better = current_metric > champion_metric
                logging.info(f'Comparing {BEST_MODEL_METRIC}: current={current_metric:.4f} vs champion={champion_metric:.4f}')
            
            result = {
                'model_version': model_version,
                'current_metric': current_metric,
                'champion_metric': champion_metric,
                'is_new_champion': is_current_better
            }
            
            if is_current_better:
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(
                    name=MODEL_NAME,
                    alias=CHAMPION_MODEL_ALIAS,
                    version=model_version
                )
                logging.info(f'Model version {model_version} is now the new champion!')
                result['status'] = 'promoted_to_champion'
            else:
                logging.info('Current model did not outperform champion. No changes made.')
                result['status'] = 'not_promoted'
            
            return result

        # Define ML pipeline flow
        dataset_info = prepare_datasets()
        training_result = train_model(dataset_info)
        evaluation_result = evaluate_model(training_result)
        registration_result = register_and_alias_model(evaluation_result)
        
        return registration_result

    # ==============================
    # Monitoring Tasks
    # ==============================
    @task_group()
    def monitoring_pipeline():
        """Drift detection and monitoring pipeline."""

        @task()
        def generate_drift_report() -> Dict[str, Any]:
            """Generate data drift report using Evidently."""
            logging.info('Generating drift report...')
            
            # Load reference data (training data)
            train_df = load_training_data_from_db('train')
            X_train, _ = prepare_features_and_target(train_df)
            
            # Load current production data (test data as proxy)
            test_df = load_training_data_from_db('test')
            X_test, _ = prepare_features_and_target(test_df)
            
            # Configure column mapping
            column_mapping = ColumnMapping(
                numerical_features=NUMERICAL_FEATURES,
                categorical_features=CATEGORICAL_FEATURES,
            )
            
            # Create drift report
            report = Report(metrics=[DataDriftPreset()])
            
            report.run(
                reference_data=X_train,
                current_data=X_test,
                column_mapping=column_mapping
            )
            
            # Save report
            reports_dir = Path(os.getenv('REPORTS_DIR', '/app/logs/drift_reports'))
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = reports_dir / f'drift_report_{timestamp}.html'
            report.save_html(str(report_path))
            
            # Extract drift metrics
            result_dict = report.as_dict()
            dataset_drift = result_dict.get('metrics', [{}])[0].get('result', {})
            
            drift_info = {
                'report_path': str(report_path),
                'timestamp': timestamp,
                'share_of_drifted_columns': dataset_drift.get('share_of_drifted_columns', 0.0),
                'dataset_drift_detected': dataset_drift.get('dataset_drift', False),
                'number_of_columns': dataset_drift.get('number_of_columns', 0),
                'number_of_drifted_columns': dataset_drift.get('number_of_drifted_columns', 0)
            }
            
            logging.info(f'Drift report generated: {report_path}')
            logging.info(f'Drift detected: {drift_info["dataset_drift_detected"]}')
            logging.info(f'Drifted columns: {drift_info["number_of_drifted_columns"]}/{drift_info["number_of_columns"]}')
            
            return drift_info

        return generate_drift_report()

    @teardown()
    def finalize():
        """Clean up Airflow Variables and finalize the DAG run."""
        logging.info('Finalizing Accidents ML Pipeline DAG')
        
        try:
            Variable.delete('model_name')
            Variable.delete('champion_alias')
            Variable.delete('best_model_metric')
            Variable.delete('random_state')
            Variable.delete('feature_columns')
            Variable.delete('target_column')
        except Exception as e:
            logging.warning(f'Warning during cleanup: {e}')
        
        logging.info('DAG run completed successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # Define DAG Flow
    # ==============================
    init_task = initialize()
    data_pipeline_task = data_pipeline()
    ml_pipeline_task = ml_pipeline()
    monitoring_task = monitoring_pipeline()
    final_task = finalize()
    
    # Chain tasks: init -> data -> ml -> monitoring -> finalize
    chain(init_task, data_pipeline_task)
    chain(data_pipeline_task, ml_pipeline_task)
    chain(ml_pipeline_task, monitoring_task)
    chain(monitoring_task, final_task)


# Instantiate the DAG
accidents_ml_pipeline()
