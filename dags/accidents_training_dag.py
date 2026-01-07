"""
Accidents Model Training DAG
============================

This DAG focuses exclusively on the ML training pipeline, assuming data
is already available in the database. Use this for retraining models
without running the full data pipeline.

Pipeline Stages:
1. Setup: Configure MLflow tracking
2. Dataset Preparation: Load and version datasets from database
3. Model Training: Train Random Forest with GridSearchCV
4. Model Evaluation: Evaluate on validation and test sets
5. Model Registration: Compare with champion and alias best model
6. Drift Monitoring: Generate drift report
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from airflow.sdk import dag, task, setup, task_group, teardown, Variable
from airflow.sdk.bases.operator import chain

from src.utils import logging
from src.utils.database import get_db_connection
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


def get_data_version() -> int:
    """Get the current data version from the database."""
    with get_db_connection() as conn:
        meta_query = """
            SELECT id FROM data_ingestion_progress
            ORDER BY id DESC LIMIT 1
        """
        meta_df = pd.read_sql_query(meta_query, conn)
        return int(meta_df['id'].iloc[0]) if len(meta_df) > 0 else 1


# ########### DAG Definition ###########

@dag(
    dag_id="accidents_model_training",
    schedule=None,  # Manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'training', 'ml-pipeline'],
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }
)
def accidents_model_training():
    """Model Training DAG - for retraining when data is already available."""

    @setup
    def initialize():
        """Initialize MLflow tracking and configuration."""
        logging.info('Initializing Model Training DAG')
        
        setup_mlflow_tracking()
        
        Variable.set('model_name', MODEL_NAME)
        Variable.set('champion_alias', CHAMPION_MODEL_ALIAS)
        Variable.set('best_model_metric', BEST_MODEL_METRIC)
        
        return {'status': 'initialized', 'timestamp': datetime.now().isoformat()}

    @task()
    def prepare_datasets() -> Dict[str, Any]:
        """Load and prepare training, validation, and test datasets with versioning."""
        logging.info('Preparing datasets for training...')
        
        setup_mlflow_tracking()
        data_version = get_data_version()
        
        train_df = load_training_data_from_db('train')
        val_df = load_training_data_from_db('validation')
        test_df = load_training_data_from_db('test')
        
        with mlflow.start_run(run_name='dataset_versioning'):
            # Log dataset metadata
            mlflow.log_params({
                'data_version': data_version,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'n_features': len(FEATURE_COLUMNS),
            })
            
            # Log class distribution
            for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                dist = df[TARGET_COLUMN].value_counts().to_dict()
                for severity, count in dist.items():
                    mlflow.log_metric(f'{split_name}_class_{severity}_count', count)
        
        dataset_info = {
            'data_version': data_version,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        }
        
        logging.info(f'Datasets prepared: {dataset_info}')
        return dataset_info

    @task()
    def train_model(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter tuning using GridSearchCV."""
        logging.info('Starting model training with GridSearchCV...')
        
        setup_mlflow_tracking()
        data_version = dataset_info['data_version']
        
        train_df = load_training_data_from_db('train')
        X_train, y_train = prepare_features_and_target(train_df)
        
        with mlflow.start_run(run_name='model_training') as run:
            # Create and log training dataset
            train_data = pd.concat([X_train, y_train], axis=1)
            train_dataset = mlflow.data.from_pandas(
                train_data,
                name=f"training_data-v{data_version}",
                targets=TARGET_COLUMN,
            )
            mlflow.log_input(train_dataset, context="training")
            
            mlflow.log_params({
                'data_version': data_version,
                'train_size': len(X_train),
                'n_features': len(FEATURE_COLUMNS),
            })
            
            # Hyperparameter grid
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }
            
            # Train with GridSearchCV
            model = RandomForestClassifier(random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Log best parameters
            mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
            mlflow.log_metric('cv_best_score', grid_search.best_score_)
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                best_model,
                MODEL_NAME,
                registered_model_name=MODEL_NAME
            )
            
            logging.info(f'Best params: {best_params}')
            logging.info(f'CV Best score: {grid_search.best_score_:.4f}')
            logging.info(f'Model URI: {model_info.model_uri}')
            
            return {
                'run_id': run.info.run_id,
                'model_uri': model_info.model_uri,
                'model_version': model_info.registered_model_version,
                'best_params': best_params,
                'cv_best_score': grid_search.best_score_,
                'data_version': data_version
            }

    @task()
    def evaluate_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model on validation and test datasets."""
        logging.info('Evaluating model...')
        
        setup_mlflow_tracking()
        model_uri = training_result['model_uri']
        data_version = training_result['data_version']
        
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
            # Create datasets
            val_data = pd.concat([X_val, y_val], axis=1)
            val_dataset = mlflow.data.from_pandas(
                val_data, name=f"validation_data-v{data_version}", targets=TARGET_COLUMN
            )
            
            test_data = pd.concat([X_test, y_test], axis=1)
            test_dataset = mlflow.data.from_pandas(
                test_data, name=f"test_data-v{data_version}", targets=TARGET_COLUMN
            )
            
            # Evaluate on validation
            logging.info('Evaluating on validation set...')
            val_result = mlflow.models.evaluate(
                model_uri, val_dataset, model_type="classifier",
                evaluator_config={'log_explainer': True},
                extra_metrics=extra_metrics
            )
            
            # Evaluate on test
            logging.info('Evaluating on test set...')
            test_result = mlflow.models.evaluate(
                model_uri, test_dataset, model_type="classifier",
                evaluator_config={'log_explainer': True},
                extra_metrics=extra_metrics
            )
            
            # Extract numeric metrics
            def extract_metrics(result):
                return {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
            
            val_metrics = extract_metrics(val_result)
            test_metrics = extract_metrics(test_result)
            
            logging.info(f'Validation {BEST_MODEL_METRIC}: {val_metrics.get(BEST_MODEL_METRIC, "N/A")}')
            logging.info(f'Test {BEST_MODEL_METRIC}: {test_metrics.get(BEST_MODEL_METRIC, "N/A")}')
            
            return {
                'model_uri': model_uri,
                'model_version': training_result['model_version'],
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_metric_value': test_metrics.get(BEST_MODEL_METRIC, 0.0)
            }

    @task()
    def register_and_alias_model(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with champion model and promote current model if better."""
        logging.info('Checking champion model and comparing...')
        
        setup_mlflow_tracking()
        
        model_version = evaluation_result['model_version']
        current_metric = evaluation_result['best_metric_value']
        
        # Check for existing champion
        champion_model_uri = f'models:/{MODEL_NAME}@{CHAMPION_MODEL_ALIAS}'
        champion_exists = False
        champion_metric = None
        
        extra_metrics = [
            weighted_f1_score_metric,
            weighted_precision_metric,
            weighted_recall_metric,
            roc_auc_ovr_metric,
            per_class_f1_metric
        ]
        
        try:
            mlflow.models.get_model_info(champion_model_uri)
            champion_exists = True
            logging.info('Found existing champion model, evaluating...')
            
            # Evaluate champion
            test_df = load_training_data_from_db('test')
            X_test, y_test = prepare_features_and_target(test_df)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            test_dataset = mlflow.data.from_pandas(
                test_data, name="test_data_champion_eval", targets=TARGET_COLUMN
            )
            
            champion_result = mlflow.models.evaluate(
                champion_model_uri, test_dataset, model_type="classifier",
                extra_metrics=extra_metrics
            )
            champion_metric = champion_result.metrics.get(BEST_MODEL_METRIC, 0.0)
            logging.info(f'Champion {BEST_MODEL_METRIC}: {champion_metric:.4f}')
            
        except Exception as e:
            logging.info(f'No champion model found: {e}')
        
        # Compare metrics
        is_better = False
        if not champion_exists:
            is_better = True
            logging.info('No champion exists, promoting current model')
        elif current_metric is not None and champion_metric is not None:
            is_better = current_metric > champion_metric
            logging.info(f'Current ({current_metric:.4f}) vs Champion ({champion_metric:.4f}): {"Better" if is_better else "Worse"}')
        
        result = {
            'model_version': model_version,
            'current_metric': current_metric,
            'champion_metric': champion_metric,
            'is_new_champion': is_better
        }
        
        if is_better:
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias=CHAMPION_MODEL_ALIAS,
                version=model_version
            )
            logging.info(f'Model v{model_version} promoted to champion!')
            result['status'] = 'promoted_to_champion'
        else:
            logging.info(f'Model v{model_version} not promoted (champion is better)')
            result['status'] = 'not_promoted'
        
        return result

    @task()
    def generate_drift_report(registration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data drift report comparing training and test distributions."""
        logging.info('Generating drift report...')
        
        train_df = load_training_data_from_db('train')
        test_df = load_training_data_from_db('test')
        
        X_train, _ = prepare_features_and_target(train_df)
        X_test, _ = prepare_features_and_target(test_df)
        
        column_mapping = ColumnMapping(
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        
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
        
        # Extract metrics
        result_dict = report.as_dict()
        dataset_drift = result_dict.get('metrics', [{}])[0].get('result', {})
        
        drift_info = {
            'report_path': str(report_path),
            'timestamp': timestamp,
            'drift_share': dataset_drift.get('share_of_drifted_columns', 0.0),
            'drift_detected': dataset_drift.get('dataset_drift', False),
            'n_columns': dataset_drift.get('number_of_columns', 0),
            'n_drifted': dataset_drift.get('number_of_drifted_columns', 0),
            'registration_status': registration_result.get('status', 'unknown')
        }
        
        logging.info(f'Drift report saved: {report_path}')
        logging.info(f'Drift detected: {drift_info["drift_detected"]} ({drift_info["n_drifted"]}/{drift_info["n_columns"]} columns)')
        
        return drift_info

    @teardown()
    def finalize():
        """Clean up and finalize the DAG run."""
        logging.info('Finalizing Model Training DAG')
        
        try:
            Variable.delete('model_name')
            Variable.delete('champion_alias')
            Variable.delete('best_model_metric')
        except Exception as e:
            logging.warning(f'Cleanup warning: {e}')
        
        logging.info('Model training pipeline completed successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # Define DAG Flow
    # ==============================
    init_task = initialize()
    dataset_task = prepare_datasets()
    train_task = train_model(dataset_task)
    eval_task = evaluate_model(train_task)
    register_task = register_and_alias_model(eval_task)
    drift_task = generate_drift_report(register_task)
    final_task = finalize()
    
    chain(init_task, dataset_task, train_task, eval_task, register_task, drift_task, final_task)


# Instantiate the DAG
accidents_model_training()
