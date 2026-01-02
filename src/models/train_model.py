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
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import mlflow
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.utils import logging
from src.utils.database import get_db_connection

load_dotenv()


# Mapping of severity levels to descriptive names
CLASS_LABELS = {
    1: 'Unscathed',
    2: 'Light injury',
    3: 'Hospitalized wounded',
    4: 'Killed'
}

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
HANDLE_MISSING = 'drop'  # Options: 'drop', 'impute'
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
    if HANDLE_MISSING == 'drop':
        df_clean = df.dropna()
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows with missing values ({dropped/len(df)*100:.1f}%)")
        df = df_clean
    elif HANDLE_MISSING == 'impute':
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

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[RandomForestClassifier,mlflow.entities.model_registry.ModelVersion]:
    """
    Train Random Forest Classifier with hyperparameter tuning, using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Tuple of (trained RandomForestClassifier, model information)
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
    
    # Log best model
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    logging.info(f"Best model parameters: {best_parameters}")
    mlflow.log_params(best_parameters)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'rf_model_{timestamp}'
    model_info = mlflow.sklearn.log_model(best_model, model_name)

    train_evaluation = evaluate_model(best_model, X_train, y_train, dataset_prefix='train')
    val_evaluation = evaluate_model(best_model, X_val, y_val, dataset_prefix='val')
    
    logging.info(f"Training evaluation: {train_evaluation}")
    for key, value in train_evaluation.items():
        logging.info(f"Logging training metric {key}: {value}")
        mlflow.log_metric(f"train_{key}", value)

    logging.info(f"Validation evaluation: {val_evaluation}")
    for key, value in val_evaluation.items():
        logging.info(f"Logging validation metric {key}: {value}")
        mlflow.log_metric(f"val_{key}", value)

    return best_model, model_info

def evaluate_model(model: RandomForestClassifier, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  dataset_prefix: str = 'train') -> Dict[str, Any]:
    """
    Evaluate model on given dataset and compute all configured metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        dataset_prefix: Name of dataset for logging
        
    Returns:
        Dictionary containing all metrics
    """
    logging.info(f"Evaluating model on {dataset_prefix} set...")
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Compute primary metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y, y_pred)
    
    # Precision, Recall, F1-score (weighted average)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision
    metrics['recall_weighted'] = recall
    metrics['f1_weighted'] = f1
    
    # ROC-AUC (One-vs-Rest for multiclass)
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(
            y, y_pred_proba, multi_class='ovr', average='weighted'
        )
    except Exception as e:
        logging.warning(f"Could not compute ROC-AUC: {e}")
        metrics['roc_auc_ovr'] = None
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
    
    for idx, severity in enumerate(sorted(y.unique())):
        class_name = CLASS_LABELS.get(severity, f'Class {severity}')
        metrics[f'class_{class_name}_precision'] = float(precision_per_class[idx])
        metrics[f'class_{class_name}_recall'] = float(recall_per_class[idx])
        metrics[f'class_{class_name}_f1'] = float(f1_per_class[idx])
        metrics[f'class_{class_name}_support'] = int(support_per_class[idx])
        
    # Log metrics
    logging.info(f"\n{dataset_prefix.upper()} SET METRICS:")
    logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    logging.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    logging.info(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
    if metrics['roc_auc_ovr'] is not None:
        logging.info(f"  ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    
    return metrics

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
                model, model_info = train_random_forest(X_train, y_train, X_val, y_val)
                model_name = 'random_forest_model'
                mlflow.sklearn.log_model(model, name=model_name)
                logging.info(f"Model logged to MLflow with URI: runs:/{run.info.run_id}/{model_name}")

                # Evaluate on test set
                result = mlflow.models.evaluate(
                    model_info.model_uri,
                    test_dataset,
                    model_type="classifier",
                )
                logging.info(f"MLflow model evaluation result: {result}")

            # Query for the best model based on validation accuracy
            best_model = mlflow.search_logged_models(
                order_by=[{'field_name': 'metrics.val_accuracy', 'ascending': False}],
                max_results=1,
                output_format='pandas'
            )

            if not best_model.empty:
                best_model_info = best_model.iloc[0]
                logging.info(f"Best model found: Name={best_model_info.name}, Metrics={best_model_info.metrics}")
                
                # Register the best model
                mlflow.register_model(
                    model_uri=f"models:/best_{best_model_info.name}",
                    name="Best_Accident_Severity_Random_Forest"
                )
                logging.info("Best model registered successfully.")
            else:
                logging.warning("No best model found based on validation accuracy.")

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