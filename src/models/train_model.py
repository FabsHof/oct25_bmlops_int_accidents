"""
Model Training Module for Accident Severity Prediction

This module implements the complete training pipeline for predicting accident severity
using a Random Forest Classifier. It includes:
- Data loading from database with static train/validation/test splits
- Feature engineering and preprocessing
- Model training with hyperparameters from config
- Comprehensive evaluation with multiple metrics
- Model versioning and persistence

Note: This module can be run standalone for local development/testing.
For production use, prefer the Airflow DAGs which provide orchestration,
monitoring, and retry capabilities.
"""

from typing import Tuple

import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.utils import logging
from src.utils.database import get_db_connection
from src.utils.ml_utils import (
    MODEL_NAME,
    CHAMPION_MODEL_ALIAS,
    BEST_MODEL_METRIC,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    setup_mlflow_tracking,
    load_training_data_from_db,
    prepare_features_and_target,
    get_data_version,
    create_versioned_dataset,
    evaluate_model as evaluate_model_util,
)
from src.models.metrics import (
    weighted_f1_score_metric,
    weighted_precision_metric,
    weighted_recall_metric,
    roc_auc_ovr_metric,
    per_class_f1_metric
)

load_dotenv()


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> mlflow.models.model.ModelInfo:
    """
    Train Random Forest Classifier with hyperparameter tuning using GridSearchCV.
    
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
    
    tuned_random_forest = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    logging.info(f"Best parameters: {best_parameters}")
    logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    mlflow.log_params(best_parameters)
    mlflow.log_metric("cv_best_score", grid_search.best_score_)
    
    model_info = mlflow.sklearn.log_model(
        tuned_random_forest, 
        MODEL_NAME, 
        registered_model_name=MODEL_NAME
    )

    return model_info


def prepare_datasets() -> Tuple[
    Tuple[pd.DataFrame, pd.Series, mlflow.data.Dataset],
    Tuple[pd.DataFrame, pd.Series, mlflow.data.Dataset],
    Tuple[pd.DataFrame, pd.Series, mlflow.data.Dataset]
]:
    """
    Load and prepare training, validation, and test datasets.
    
    Returns:
        Tuple of ((X_train, y_train, train_dataset), (X_val, y_val, val_dataset), (X_test, y_test, test_dataset))
    """
    data_version = get_data_version()

    train_df = load_training_data_from_db('train')
    val_df = load_training_data_from_db('validation')
    test_df = load_training_data_from_db('test')
    
    X_train, y_train = prepare_features_and_target(train_df)
    X_val, y_val = prepare_features_and_target(val_df)
    X_test, y_test = prepare_features_and_target(test_df)

    train_dataset = create_versioned_dataset(
        pd.concat([X_train, y_train], axis=1), 
        version=data_version, 
        base_name='training_data',
        target_column=TARGET_COLUMN
    )

    val_dataset = create_versioned_dataset(
        pd.concat([X_val, y_val], axis=1), 
        version=data_version, 
        base_name='validation_data',
        target_column=TARGET_COLUMN
    )

    test_dataset = create_versioned_dataset(
        pd.concat([X_test, y_test], axis=1), 
        version=data_version, 
        base_name='test_data',
        target_column=TARGET_COLUMN
    )

    return (X_train, y_train, train_dataset), (X_val, y_val, val_dataset), (X_test, y_test, test_dataset)


def train_model() -> None:
    """
    Main training function that orchestrates the entire training pipeline.
    
    Loads data from database, trains a Random Forest classifier with hyperparameter
    tuning, evaluates on validation and test sets, and registers the best model
    to MLflow.
    """
    extra_metrics = [
        weighted_f1_score_metric,
        weighted_precision_metric,
        weighted_recall_metric,
        roc_auc_ovr_metric,
        per_class_f1_metric
    ]

    with mlflow.start_run(run_name='training') as parent_run:
        # Dataset preparation
        try:
            logging.info("Preparing datasets...")
            (X_train, y_train, train_dataset), (X_val, y_val, val_dataset), (X_test, y_test, test_dataset) = prepare_datasets()
        except Exception as e:
            logging.error(f"Error preparing datasets: {e}")
            raise
        
        # Model training
        try:
            with mlflow.start_run(run_name='model_training', nested=True):
                logging.info("Logging datasets to MLflow...")
                mlflow.log_inputs(
                    datasets=[train_dataset, val_dataset, test_dataset], 
                    contexts=['training', 'validation', 'test'],
                    tags_list=[None, None, None]
                )
                
                logging.info("Starting model training...")
                current_model_info = train_random_forest(X_train, y_train)
                logging.info(f"Model trained and logged with URI: {current_model_info.model_uri}")

                # Evaluate the model on validation and test sets
                logging.info("Evaluating model on validation dataset...")
                evaluate_model_util(current_model_info.model_uri, val_dataset, extra_metrics=extra_metrics)
                
                logging.info("Evaluating model on test dataset...")
                current_model_evaluation = evaluate_model_util(
                    current_model_info.model_uri, test_dataset, extra_metrics=extra_metrics
                )
                logging.info("Model evaluation completed.")
                
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

        # Model registration and champion comparison
        try:
            with mlflow.start_run(run_name='model_registration', nested=True):
                logging.info("Registering and comparing model with champion...")
                
                champion_model_uri = f'models:/{MODEL_NAME}@{CHAMPION_MODEL_ALIAS}'
                champion_model_info = None
                champion_evaluation = None
                
                try:
                    champion_model_info = mlflow.models.get_model_info(champion_model_uri)
                    logging.info("Found existing champion model")
                except Exception as e:
                    logging.info(f"No existing champion model found: {e}")

                if champion_model_info is not None:
                    champion_evaluation = evaluate_model_util(
                        champion_model_uri, test_dataset, extra_metrics=extra_metrics
                    )

                # Compare current model with champion
                is_current_better = False
                if champion_model_info is None:
                    is_current_better = True
                    logging.info("No champion exists, promoting current model")
                else:
                    current_metric = current_model_evaluation.metrics.get(BEST_MODEL_METRIC)
                    champion_metric = champion_evaluation.metrics.get(BEST_MODEL_METRIC)
                    logging.info(f"Comparing {BEST_MODEL_METRIC}: current={current_metric:.4f} vs champion={champion_metric:.4f}")
                    if current_metric is not None and champion_metric is not None:
                        is_current_better = current_metric > champion_metric
                
                if is_current_better:
                    client = mlflow.tracking.MlflowClient()
                    client.set_registered_model_alias(
                        name=MODEL_NAME,
                        alias=CHAMPION_MODEL_ALIAS,
                        version=current_model_info.registered_model_version
                    )
                    logging.info(f"Model version {current_model_info.registered_model_version} is now the new champion!")
                else:
                    logging.info("Current model did not outperform the champion. No changes made.")
                    
        except Exception as e:
            logging.error(f"Error during model registration: {e}")
            raise

    logging.info("Model training pipeline completed successfully.")


def main():
    """Entry point for the model training script."""
    logging.info("Starting training script...")
    
    # Use localhost for local development
    setup_mlflow_tracking(use_localhost=True)

    try:
        train_model()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
