"""
ML Utilities Module for Accidents Project

This module provides shared utility functions used across the Airflow DAGs
and training scripts, including:
- MLflow setup and configuration
- Database data loading for training datasets
- Feature and target preparation
- Data version management
- Pydantic BaseModel classes for prediction request/response
"""

import os
from typing import Tuple, Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

import pandas as pd
import mlflow
from dotenv import load_dotenv

from src.utils import logging
from src.utils.database import get_db_connection

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

HANDLE_MISSING_STRATEGY = 'drop'  # Options: 'drop', 'impute'


def setup_mlflow_tracking(use_localhost: bool = False) -> str:
    """
    Set up MLflow tracking and return tracking URI.
    
    Args:
        use_localhost: If True, use localhost for tracking (for local development).
                      If False, use container hostnames (for Airflow/Docker).
    
    Returns:
        The MLflow tracking URI
    """
    if use_localhost:
        tracking_uri = f'http://localhost:{os.getenv("MLFLOW_PORT", "5001")}'
        s3_endpoint = f'http://localhost:{os.getenv("MINIO_PORT", "9000")}'
    else:
        tracking_uri = f'http://{os.getenv("MLFLOW_HOST", "mlflow")}:{os.getenv("MLFLOW_PORT", "5001")}'
        s3_endpoint = f'http://{os.getenv("MINIO_HOST", "minio")}:{os.getenv("MINIO_PORT", "9000")}'
    
    mlflow.set_tracking_uri(tracking_uri)

    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ROOT_USER', 'minio_user')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_ROOT_PASSWORD', 'minio_password')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    mlflow.set_experiment('Car Accident Severity Prediction')
    logging.info(f'MLflow tracking set up at {tracking_uri}')
    logging.info(f'MLflow S3 endpoint: {os.environ["MLFLOW_S3_ENDPOINT_URL"]}')
    return tracking_uri


def load_training_data_from_db(
    dataset_split: str = 'train',
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Load training data from database for specified dataset split.
    
    Args:
        dataset_split: Dataset split to load ('train', 'validation', or 'test')
        feature_columns: List of feature columns to load. Defaults to FEATURE_COLUMNS.
        target_column: Target column to load. Defaults to TARGET_COLUMN.
    
    Returns:
        DataFrame containing the requested data
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    target_column = target_column or TARGET_COLUMN
    
    logging.info(f"Loading {dataset_split} data from database...")
    
    with get_db_connection() as conn:
        columns = feature_columns + [target_column]
        query = f"""
            SELECT {', '.join(columns)}
            FROM clean_data
            WHERE dataset_split = '{dataset_split}' AND is_current = TRUE
        """
        df = pd.read_sql_query(query, conn)
        logging.info(f"Loaded {len(df)} records for {dataset_split} set")
        
        # Log class distribution
        if target_column in df.columns:
            class_dist = df[target_column].value_counts().sort_index()
            logging.info(f"{dataset_split} set class distribution:")
            for severity, count in class_dist.items():
                class_name = CLASS_LABELS.get(severity, f'Class {severity}')
                logging.info(f"  {class_name} (severity={severity}): {count} ({count/len(df)*100:.1f}%)")
        
        return df


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    handle_missing: str = 'drop'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) from dataframe.
    
    Args:
        df: Input dataframe
        feature_columns: List of feature columns. Defaults to FEATURE_COLUMNS.
        target_column: Target column. Defaults to TARGET_COLUMN.
        handle_missing: How to handle missing values ('drop' or 'impute')
    
    Returns:
        Tuple of (features, target)
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    target_column = target_column or TARGET_COLUMN
    
    # Handle missing values
    if handle_missing == 'drop':
        df_clean = df.dropna()
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows with missing values ({dropped/len(df)*100:.1f}%)")
    elif handle_missing == 'impute':
        df_clean = df.fillna(df.median())
        logging.info("Imputed missing values with median")
    else:
        df_clean = df
    
    X = df_clean[feature_columns].copy().astype('float64')
    y = df_clean[target_column].copy()
    return X, y


def get_data_version() -> int:
    """
    Get the current data version from the database.
    
    Returns the latest version number from completed data ingestion entries.
    
    Returns:
        Latest version number from data_ingestion_progress table
    """
    with get_db_connection() as conn:
        meta_query = """
            SELECT version
            FROM data_ingestion_progress
            WHERE is_complete = TRUE
            ORDER BY last_updated DESC, id DESC
            LIMIT 1
        """
        meta_df = pd.read_sql_query(meta_query, conn)
        
        if len(meta_df) > 0 and meta_df['version'].iloc[0] is not None:
            version = int(meta_df['version'].iloc[0])
        else:
            # Fallback if no completed ingestions exist
            version = 1
        
        logging.info(f"Current data version: {version}")
        return version


def load_data_ingestion_metadata() -> pd.DataFrame:
    """
    Load the last data ingestion metadata from the database.
    
    Returns:
        DataFrame with the latest ingestion metadata
    """
    with get_db_connection() as conn:
        query = """
            SELECT *
            FROM data_ingestion_progress
            ORDER BY id DESC
            LIMIT 1
        """
        df = pd.read_sql_query(query, conn)
        if len(df) > 0:
            logging.info(f"Last metadata entry: {df.iloc[0].to_dict()}")
        return df


def create_versioned_dataset(
    data: pd.DataFrame,
    version: int,
    base_name: str,
    target_column: Optional[str] = None,
    source: Optional[str] = None
) -> mlflow.data.Dataset:
    """
    Create a versioned MLflow dataset with metadata.
    
    Args:
        data: The pandas DataFrame to version
        version: Version number for the dataset
        base_name: Base name for the dataset (e.g., 'training_data')
        target_column: Name of the target column. Defaults to TARGET_COLUMN.
        source: Optional source path for the dataset
    
    Returns:
        MLflow Dataset object
    """
    target_column = target_column or TARGET_COLUMN
    
    with mlflow.start_run(run_name=f'{base_name}_versioning', nested=True):
        dataset = mlflow.data.from_pandas(
            data,
            source=source,
            name=f"{base_name}-v{version}",
            targets=target_column,
        )
        mlflow.log_input(dataset, context="dataset_versioning")

        # Log version metadata
        mlflow.log_params({
            "dataset_version": version,
            "data_size": len(data),
            "features_count": len(data.columns) - 1,
            "target_distribution": data[target_column].value_counts().to_dict(),
        })

        # Log data quality metrics
        mlflow.log_metrics({
            "missing_values_pct": (data.isnull().sum().sum() / data.size) * 100,
            "duplicate_rows": data.duplicated().sum(),
            "target_balance": data[target_column].std(),
        })

    logging.info(f"Created versioned dataset: {base_name}-v{version} with {len(data)} records")
    return dataset


def evaluate_model(
    model_uri: str,
    dataset: mlflow.data.Dataset,
    extra_metrics: Optional[List] = None,
    model_type: str = "classifier"
) -> mlflow.models.EvaluationResult:
    """
    Evaluate a model on a given dataset with custom metrics.
    
    Args:
        model_uri: URI of the model to evaluate
        dataset: MLflow dataset to evaluate on
        extra_metrics: List of additional custom metrics
        model_type: Type of model (default: "classifier")
        
    Returns:
        Evaluation result containing metrics
    """
    from src.models.metrics import (
        weighted_f1_score_metric,
        weighted_precision_metric,
        weighted_recall_metric,
        roc_auc_ovr_metric,
        per_class_f1_metric
    )
    
    if extra_metrics is None:
        extra_metrics = [
            weighted_f1_score_metric,
            weighted_precision_metric,
            weighted_recall_metric,
            roc_auc_ovr_metric,
            per_class_f1_metric
        ]
    
    result = mlflow.models.evaluate(
        model_uri,
        dataset,
        model_type=model_type,
        evaluator_config={
            'log_explainer': False,
            'log_model_explainability': False,
            'explainability_algorithm': None,
            'log_shap_values': False,
        },
        extra_metrics=extra_metrics
    )
    
    logging.info(f"Model evaluation completed for {model_uri}")
    return result


def compare_with_champion(
    current_model_version: str,
    current_metric_value: float,
    test_dataset: mlflow.data.Dataset,
    model_name: Optional[str] = None,
    champion_alias: Optional[str] = None,
    best_metric: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare current model with champion and optionally promote.
    
    Args:
        current_model_version: Version of the current model
        current_metric_value: The metric value of the current model
        test_dataset: Test dataset for evaluating the champion
        model_name: Name of the registered model. Defaults to MODEL_NAME.
        champion_alias: Alias for champion model. Defaults to CHAMPION_MODEL_ALIAS.
        best_metric: Metric to compare. Defaults to BEST_MODEL_METRIC.
    
    Returns:
        Dictionary with comparison results and promotion status
    """
    model_name = model_name or MODEL_NAME
    champion_alias = champion_alias or CHAMPION_MODEL_ALIAS
    best_metric = best_metric or BEST_MODEL_METRIC
    
    champion_model_uri = f'models:/{model_name}@{champion_alias}'
    champion_exists = False
    champion_metric = None
    
    try:
        mlflow.models.get_model_info(champion_model_uri)
        champion_exists = True
        logging.info('Found existing champion model, evaluating...')
        
        champion_result = evaluate_model(champion_model_uri, test_dataset)
        champion_metric = champion_result.metrics.get(best_metric, 0.0)
        logging.info(f'Champion {best_metric}: {champion_metric:.4f}')
        
    except Exception as e:
        logging.info(f'No champion model found: {e}')
    
    # Compare metrics
    is_better = False
    if not champion_exists:
        is_better = True
        logging.info('No champion exists, promoting current model')
    elif current_metric_value is not None and champion_metric is not None:
        is_better = current_metric_value > champion_metric
        logging.info(f'Current ({current_metric_value:.4f}) vs Champion ({champion_metric:.4f}): {"Better" if is_better else "Worse"}')
    
    result = {
        'model_version': current_model_version,
        'current_metric': current_metric_value,
        'champion_metric': champion_metric,
        'is_new_champion': is_better
    }
    
    if is_better:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=current_model_version
        )
        logging.info(f'Model v{current_model_version} promoted to champion!')
        result['status'] = 'promoted_to_champion'
    else:
        logging.info(f'Model v{current_model_version} not promoted (champion is better)')
        result['status'] = 'not_promoted'
    
    return result

# Pydantic BaseModel classes for prediction request/response
class PredictionRequest(BaseModel):
    """Request model for accident severity prediction."""
    year: int = Field(..., description="Year of the accident", ge=2000, le=2100)
    month: int = Field(..., description="Month of the accident", ge=1, le=12)
    hour: int = Field(..., description="Hour of the accident", ge=0, le=23)
    minute: int = Field(..., description="Minute of the accident", ge=0, le=59)
    user_category: int = Field(..., description="User category (e.g., driver, passenger, pedestrian)")
    sex: int = Field(..., description="Sex of the user")
    year_of_birth: int = Field(..., description="Year of birth of the user", ge=1900, le=2100)
    trip_purpose: int = Field(..., description="Purpose of the trip")
    security: int = Field(..., description="Security equipment used")
    luminosity: int = Field(..., description="Luminosity conditions")
    weather: int = Field(..., description="Weather conditions")
    type_of_road: int = Field(..., description="Type of road")
    road_surface: int = Field(..., description="Road surface condition")
    latitude: float = Field(..., description="Latitude of the accident location")
    longitude: float = Field(..., description="Longitude of the accident location")
    holiday: int = Field(..., description="Holiday indicator (0 or 1)", ge=0, le=1)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year": 2023,
                "month": 6,
                "hour": 14,
                "minute": 30,
                "user_category": 1,
                "sex": 1,
                "year_of_birth": 1990,
                "trip_purpose": 1,
                "security": 1,
                "luminosity": 1,
                "weather": 1,
                "type_of_road": 1,
                "road_surface": 1,
                "latitude": 48.8566,
                "longitude": 2.3522,
                "holiday": 0
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response model for accident severity prediction."""
    prediction: int = Field(..., description="Predicted severity class (1-4)")
    prediction_label: str = Field(..., description="Human-readable severity label")
    probabilities: Dict[str, float] = Field(..., description="Probability for each severity class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    model_version: Optional[str] = Field(None, description="Version of the model used")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 3,
                "prediction_label": "Hospitalized wounded",
                "probabilities": {
                    "Unscathed": 0.1,
                    "Light injury": 0.2,
                    "Hospitalized wounded": 0.65,
                    "Killed": 0.05
                },
                "confidence": 0.65,
                "model_version": "accident_severity_rf_20251210_162637"
            }
        }
    )
