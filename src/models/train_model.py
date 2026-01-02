"""
Model Training Module for Accident Severity Prediction

This module implements the complete training pipeline for predicting accident severity
using a Random Forest Classifier. It includes:
- Data loading from database with static train/validation/test splits
- Feature engineering and preprocessing
- Model training with hyperparameters from config
- Comprehensive evaluation with multiple metrics
- Model versioning and persistence
"""

import os
from typing import Tuple
import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
BEST_MODEL_METRIC = 'weighted_f1_score'  # Metric used to select the best model for registration
MODEL_NAME = 'random_forest_model'
CHAMPION_MODEL_ALIAS = 'champion'

# Mapping of severity levels to descriptive names
CLASS_LABELS = {
    1: 'Unscathed',
    2: 'Light injury',
    3: 'Hospitalized wounded',
    4: 'Killed'
}

# Features and target configuration
FEATURE_COLUMNS = [
    'year',
    'month',
    'hour',
    'minute',
    'user_category',
    'sex',
    'year_of_birth',
    'trip_purpose',
    'security',
    'luminosity',
    'weather',
    'type_of_road',
    'road_surface',
    'latitude',
    'longitude',
    'holiday'
]
TARGET_COLUMN = 'severity'

# Data handling configuration
HANDLE_MISSING_STRATEGY = 'drop'  # Options: 'drop', 'impute'
RANDOM_STATE = 42

def setup_mlflow() -> None:
    '''Set up MLflow tracking and experiment.'''
    # Set tracking URI to point to MLflow server
    tracking_uri = f'http://localhost:{os.getenv("MLFLOW_PORT", "5001")}'
    mlflow.set_tracking_uri(tracking_uri)

    # Configure S3/MinIO credentials for artifact storage (host machine uses localhost endpoint)
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ROOT_USER', 'minio_user')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_ROOT_PASSWORD', 'minio_password')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://localhost:{os.getenv("MINIO_PORT", "9000")}'
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    mlflow.set_experiment('Car Accident Severity Prediction')
    logging.info(f'MLflow tracking set up at {mlflow.get_tracking_uri()}')
    logging.info(f'MLflow S3 endpoint: {os.environ["MLFLOW_S3_ENDPOINT_URL"]}')

def load_training_data(conn, dataset_split: str = 'train') -> pd.DataFrame:
    """
    Load training data from database for specified dataset split.
    
    Args:
        conn: Database connection
        dataset_split: Dataset split to load ('train', 'validation', or 'test')
        
    Returns:
        DataFrame containing the requested data
    """
    logging.info(f"Loading {dataset_split} data from database...")
    
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
    """
    Prepare features (X) and target (y) from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (features, target)
    """
    # Handle missing values
    if HANDLE_MISSING_STRATEGY == 'drop':
        df_clean = df.dropna()
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows with missing values ({dropped/len(df)*100:.1f}%)")
        df = df_clean
    elif HANDLE_MISSING_STRATEGY == 'impute':
        # Simple imputation with median for numeric columns
        df = df.fillna(df.median())
        logging.info("Imputed missing values with median")
    
    # Separate features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Convert all features to float64 to handle missing values and avoid MLflow schema warnings
    X = X.astype('float64')
    
    logging.info(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} rows")
    logging.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series ) -> mlflow.models.model.ModelInfo:
    """
    Train Random Forest Classifier with hyperparameter tuning, using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Model information logged to MLflow
    """
    logging.info("Training Random Forest Classifier...")
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Log best estimator and parameters
    tuned_random_forest = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    logging.info(f"Model parameters of tuned_random_forest: {best_parameters}")

    mlflow.log_params(best_parameters)
    model_info = mlflow.sklearn.log_model(
        tuned_random_forest, 
        MODEL_NAME, 
        registered_model_name=MODEL_NAME
    )

    return model_info

def evaluate_model(model_uri: str, dataset: mlflow.data.Dataset, model_type: str = "classifier") -> mlflow.models.EvaluationResult:
    """
    Evaluate a model on a given dataset with custom metrics.
    
    Args:
        model_uri: URI of the model to evaluate
        dataset: MLflow dataset to evaluate on
        model_type: Type of model (default: "classifier")
        
    Returns:
        Evaluation result containing metrics
    """
    return mlflow.models.evaluate(
        model_uri,
        dataset,
        model_type=model_type,
        evaluator_config={
            'log_explainer': True,
            'explainer_type': 'exact',
        },
        extra_metrics=[
            weighted_f1_score_metric,
            weighted_precision_metric,
            weighted_recall_metric,
            roc_auc_ovr_metric,
            per_class_f1_metric
        ]
    )

def train_model() -> None:
    """
    Main training function that orchestrates the entire training pipeline.
    
    Loads data from database, trains a Random Forest classifier with hyperparameter
    tuning, evaluates on validation and test sets, and registers the best model
    to MLflow.
    """
    logging.info("="*80)
    logging.info("STARTING MODEL TRAINING PIPELINE")
    logging.info("="*80)
    
    with get_db_connection() as conn:
    
        try:
            # Load data
            train_df = load_training_data(conn, 'train')
            val_df = load_training_data(conn, 'validation')
            test_df = load_training_data(conn, 'test')
            
            # Prepare features and targets
            X_train, y_train = prepare_features_and_target(train_df)
            X_val, y_val = prepare_features_and_target(val_df)
            X_test, y_test = prepare_features_and_target(test_df)

            # Log datasets to MLflow
            train_dataset = mlflow.data.from_pandas(
                pd.concat([X_train, y_train], axis=1),
                source='accidents_db.clean_data',
                name='Training Data',
                targets=TARGET_COLUMN
            )
            val_dataset = mlflow.data.from_pandas(
                pd.concat([X_val, y_val], axis=1),
                source='accidents_db.clean_data',
                name='Validation Data',
                targets=TARGET_COLUMN
            )
            test_dataset = mlflow.data.from_pandas(
                pd.concat([X_test, y_test], axis=1),
                source='accidents_db.clean_data',
                name='Test Data',
                targets=TARGET_COLUMN
            )
            
            with mlflow.start_run() as run:
                mlflow.log_inputs(
                    datasets=[train_dataset, val_dataset, test_dataset], 
                    contexts=['training', 'validation', 'test'],
                    tags_list=[None, None, None]
                    )
                current_model_info = train_random_forest(X_train, y_train)

                # Evaluate the model on validation set with custom metrics
                evaluate_model(
                    current_model_info.model_uri,
                    val_dataset,
                    model_type="classifier"
                )

                # Evaluate on test set with custom metrics
                current_model_evaluation = evaluate_model(
                    current_model_info.model_uri,
                    test_dataset,
                    model_type="classifier"
                )

            # Check for existing champion model
            champion_model_uri = f'models:/{MODEL_NAME}@{CHAMPION_MODEL_ALIAS}'
            try:
                champion_model_info = mlflow.models.get_model_info(champion_model_uri)
                logging.info("Found existing champion model")
            except Exception as e:
                logging.info(f"No existing champion model found: {e}")
                champion_model_info = None
            
            # If champion model exists, compare with current model and register if better
            champion_model = mlflow.sklearn.load_model(champion_model_uri) if champion_model_info is not None else None
            if champion_model is not None:
                champion_evaluation = evaluate_model(
                    champion_model_uri,
                    test_dataset,
                    model_type="classifier"
                )

            # Compare current model with champion model regarding BEST_MODEL_METRIC
            is_current_better = False
            if champion_model is None:
                is_current_better = True
            else:
                current_metric = current_model_evaluation.metrics.get(BEST_MODEL_METRIC)
                champion_metric = champion_evaluation.metrics.get(BEST_MODEL_METRIC)
                logging.info(f"Comparing {BEST_MODEL_METRIC} for champion: {champion_metric} and current: {current_metric}")
                if current_metric is not None and champion_metric is not None:
                    is_current_better = current_metric > champion_metric
            
            # add alias 'champion' to the current model if better than champion or no champion exists
            if is_current_better:
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(
                    name=MODEL_NAME,
                    alias=CHAMPION_MODEL_ALIAS,
                    version=current_model_info.registered_model_version
                )

                logging.info(f"Current model (version {current_model_info.registered_model_version}) is now the new champion.")
            else:
                logging.info("Current model did not outperform the champion model. No changes made to champion.")

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
    logging.info("Model training pipeline completed successfully.")

def main():
    """Entry point for the model training script."""
    logging.info("Starting training script...")
    setup_mlflow()

    try:
        train_model()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()