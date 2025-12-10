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
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import joblib
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

from src.utils import logging
from src.utils.database import get_db_connection
from src.models.config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    VALIDATION_CONFIG,
    METRICS_CONFIG,
    PERSISTENCE_CONFIG
)


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
        SELECT {', '.join(DATASET_CONFIG['feature_columns'] + [DATASET_CONFIG['target_column']])}
        FROM clean_data
        WHERE dataset_split = '{dataset_split}' AND is_current = TRUE
    """
    
    df = pd.read_sql_query(query, conn)
    logging.info(f"Loaded {len(df)} records for {dataset_split} set")
    
    # Log class distribution
    if DATASET_CONFIG['target_column'] in df.columns:
        class_dist = df[DATASET_CONFIG['target_column']].value_counts().sort_index()
        logging.info(f"{dataset_split} set class distribution:")
        for severity, count in class_dist.items():
            class_name = METRICS_CONFIG['class_labels'].get(severity, f'Class {severity}')
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
    if DATASET_CONFIG['handle_missing'] == 'drop':
        df_clean = df.dropna()
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows with missing values ({dropped/len(df)*100:.1f}%)")
        df = df_clean
    elif DATASET_CONFIG['handle_missing'] == 'impute':
        # Simple imputation with median for numeric columns
        df = df.fillna(df.median())
        logging.info("Imputed missing values with median")
    
    # Separate features and target
    X = df[DATASET_CONFIG['feature_columns']].copy()
    y = df[DATASET_CONFIG['target_column']].copy()
    
    logging.info(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} rows")
    logging.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train Random Forest Classifier with configured hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Trained Random Forest model
    """
    logging.info("Training Random Forest Classifier...")
    logging.info(f"Hyperparameters: {MODEL_CONFIG['hyperparameters']}")
    
    # Initialize model
    model = RandomForestClassifier(**MODEL_CONFIG['hyperparameters'])
    
    # Train model
    model.fit(X_train, y_train)
    
    logging.info("Model training completed!")
    logging.info(f"Number of trees: {model.n_estimators}")
    logging.info(f"Number of features: {model.n_features_in_}")
    logging.info(f"Number of classes: {model.n_classes_}")
    
    return model


def evaluate_model(model: RandomForestClassifier, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  dataset_name: str = 'validation') -> Dict[str, Any]:
    """
    Evaluate model on given dataset and compute all configured metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        dataset_name: Name of dataset for logging
        
    Returns:
        Dictionary containing all metrics
    """
    logging.info(f"Evaluating model on {dataset_name} set...")
    
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
    
    metrics['per_class'] = {}
    for idx, severity in enumerate(sorted(y.unique())):
        class_name = METRICS_CONFIG['class_labels'].get(severity, f'Class {severity}')
        metrics['per_class'][class_name] = {
            'severity': int(severity),
            'precision': float(precision_per_class[idx]),
            'recall': float(recall_per_class[idx]),
            'f1_score': float(f1_per_class[idx]),
            'support': int(support_per_class[idx])
        }
    
    # Confusion matrix
    if METRICS_CONFIG['compute_confusion_matrix']:
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
    
    # Log metrics
    logging.info(f"\n{dataset_name.upper()} SET METRICS:")
    logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    logging.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    logging.info(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
    if metrics['roc_auc_ovr'] is not None:
        logging.info(f"  ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    
    logging.info(f"\n  Per-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        logging.info(f"    {class_name}:")
        logging.info(f"      Precision: {class_metrics['precision']:.4f}")
        logging.info(f"      Recall: {class_metrics['recall']:.4f}")
        logging.info(f"      F1-score: {class_metrics['f1_score']:.4f}")
        logging.info(f"      Support: {class_metrics['support']}")
    
    return metrics


def compute_feature_importance(model: RandomForestClassifier, 
                               feature_names: list) -> pd.DataFrame:
    """
    Compute and return feature importance scores.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature names and importance scores
    """
    logging.info("Computing feature importance...")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("\nTop 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        logging.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def save_model_artifacts(model: RandomForestClassifier,
                        feature_names: list,
                        train_metrics: Dict[str, Any],
                        val_metrics: Dict[str, Any],
                        feature_importance: pd.DataFrame,
                        version: Optional[str] = None) -> str:
    """
    Save model and all artifacts to disk.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        train_metrics: Training metrics
        val_metrics: Validation metrics
        feature_importance: Feature importance DataFrame
        version: Model version (auto-generated if None)
        
    Returns:
        Path to saved model directory
    """
    logging.info("Saving model artifacts...")
    
    # Generate version if not provided
    if version is None:
        if PERSISTENCE_CONFIG['versioning_strategy'] == 'timestamp':
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            version = 'v1'
    
    # Create model directory
    model_dir = Path(PERSISTENCE_CONFIG['model_dir']) / f"{MODEL_CONFIG['model_name']}_{version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / f"model.{PERSISTENCE_CONFIG['model_format']}"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to: {model_path}")
    
    # Save feature names
    feature_names_path = model_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'model_version': version,
        'timestamp': datetime.now().isoformat()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to: {metrics_path}")
    
    # Save feature importance
    importance_path = model_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logging.info(f"Feature importance saved to: {importance_path}")
    
    # Save configuration
    config_path = model_dir / "config.json"
    config = {
        'dataset_config': DATASET_CONFIG,
        'model_config': MODEL_CONFIG,
        'validation_config': VALIDATION_CONFIG,
        'metrics_config': {
            'primary_metrics': METRICS_CONFIG['primary_metrics'],
            'class_labels': METRICS_CONFIG['class_labels']
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved to: {config_path}")
    
    # Save model card / README
    readme_path = model_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# {MODEL_CONFIG['model_name']} - Version {version}\n\n")
        f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Model Type\n\n")
        f.write(f"{MODEL_CONFIG['model_type']}\n\n")
        f.write(f"## Validation Strategy\n\n")
        f.write(f"{VALIDATION_CONFIG['description']}\n\n")
        f.write(f"## Metrics\n\n")
        f.write(f"### Validation Set Performance\n\n")
        f.write(f"- **Accuracy:** {val_metrics['accuracy']:.4f}\n")
        f.write(f"- **Precision (weighted):** {val_metrics['precision_weighted']:.4f}\n")
        f.write(f"- **Recall (weighted):** {val_metrics['recall_weighted']:.4f}\n")
        f.write(f"- **F1-score (weighted):** {val_metrics['f1_weighted']:.4f}\n")
        if val_metrics.get('roc_auc_ovr'):
            f.write(f"- **ROC-AUC (OvR):** {val_metrics['roc_auc_ovr']:.4f}\n")
        f.write(f"\n### Per-Class Metrics\n\n")
        for class_name, class_metrics in val_metrics['per_class'].items():
            f.write(f"**{class_name}:**\n")
            f.write(f"- Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"- Recall: {class_metrics['recall']:.4f}\n")
            f.write(f"- F1-score: {class_metrics['f1_score']:.4f}\n\n")
        f.write(f"\n## Hyperparameters\n\n")
        f.write(f"```json\n{json.dumps(MODEL_CONFIG['hyperparameters'], indent=2)}\n```\n\n")
        f.write(f"\n## Features\n\n")
        f.write(f"Total features: {len(feature_names)}\n\n")
        for feat in feature_names:
            f.write(f"- {feat}\n")
    
    logging.info(f"Model card saved to: {readme_path}")
    logging.info(f"All artifacts saved to: {model_dir}")
    
    return str(model_dir)


def train_model(version: Optional[str] = None, 
                evaluate_test: bool = False) -> Dict[str, Any]:
    """
    Main training function that orchestrates the entire training pipeline.
    
    Args:
        version: Model version (auto-generated if None)
        evaluate_test: Whether to evaluate on test set (default: False)
        
    Returns:
        Dictionary with training results
    """
    logging.info("="*80)
    logging.info("STARTING MODEL TRAINING PIPELINE")
    logging.info("="*80)
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        # Load data
        train_df = load_training_data(conn, 'train')
        val_df = load_training_data(conn, 'validation')
        
        # Prepare features and targets
        X_train, y_train = prepare_features_and_target(train_df)
        X_val, y_val = prepare_features_and_target(val_df)
        
        # Train model
        model = train_random_forest(X_train, y_train)
        
        # Evaluate on training set
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, X_val, y_val, 'validation')
        
        # Compute feature importance
        feature_importance = compute_feature_importance(
            model, 
            DATASET_CONFIG['feature_columns']
        )
        
        # Save model artifacts
        model_dir = save_model_artifacts(
            model,
            DATASET_CONFIG['feature_columns'],
            train_metrics,
            val_metrics,
            feature_importance,
            version
        )
        
        # Optional: Evaluate on test set
        test_metrics = None
        if evaluate_test:
            test_df = load_training_data(conn, 'test')
            X_test, y_test = prepare_features_and_target(test_df)
            test_metrics = evaluate_model(model, X_test, y_test, 'test')
        
        logging.info("="*80)
        logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("="*80)
        
        return {
            'success': True,
            'model_dir': model_dir,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance.to_dict('records')
        }
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    finally:
        conn.close()


def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Train Random Forest model for predicting accident severity.',
        epilog='The model uses a static validation dataset tracked in the database.'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Model version (auto-generated if not provided)'
    )
    parser.add_argument(
        '--evaluate-test',
        action='store_true',
        help='Evaluate on test set after training (use sparingly)'
    )
    
    args = parser.parse_args()
    
    try:
        result = train_model(
            version=args.version,
            evaluate_test=args.evaluate_test
        )
        logging.info(f"Training result: {result}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()