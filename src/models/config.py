"""
Model Configuration for Accident Severity Prediction

This configuration defines all parameters for model training, validation, and evaluation.
It ensures reproducibility and clarity about the modeling approach.
"""

# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {
    # Dataset split ratios (should match the splits in database)
    'train_ratio': 0.6,
    'validation_ratio': 0.2,
    'test_ratio': 0.2,
    
    # Random seed for reproducibility
    'random_state': 42,
    
    # Features to use for training
    'feature_columns': [
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
    ],
    
    # Target variable
    'target_column': 'severity',
    
    # Handle missing values strategy
    'handle_missing': 'drop',  # Options: 'drop', 'impute'
}


# ============================================================
# MODEL CONFIGURATION - RANDOM FOREST CLASSIFIER
# ============================================================

MODEL_CONFIG = {
    'model_type': 'RandomForestClassifier',
    'model_name': 'accident_severity_rf',
    
    
    # Feature engineering
    'feature_engineering': {
        'scale_features': False,        # Random Forest doesn't require scaling
        'encode_categorical': False,    # Features are already numeric
        'create_interactions': False    # Keep it simple for now
    }
}


# ============================================================
# VALIDATION STRATEGY
# ============================================================

VALIDATION_CONFIG = {
    'strategy': 'static_split',
    'description': (
        'Uses a static validation dataset that never changes. '
        'The validation split is assigned once during data cleaning and '
        'tracked in the database via the dataset_split column. '
        'This ensures consistent evaluation across training runs.'
    ),
    
    # Validation dataset is identified by dataset_split = 'validation'
    'validation_source': 'database',
    'validation_filter': "dataset_split = 'validation' AND is_current = TRUE",
    
    # Cross-validation (optional, for hyperparameter tuning)
    'use_cross_validation': False,
    'cv_folds': 5,
    'cv_strategy': 'stratified',
}


# ============================================================
# EVALUATION METRICS
# ============================================================

METRICS_CONFIG = {
    # Primary metrics for model evaluation
    'primary_metrics': [
        'accuracy',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
        'roc_auc_ovr'  # One-vs-Rest for multiclass
    ],
    
    # Per-class metrics
    'per_class_metrics': [
        'precision',
        'recall',
        'f1_score'
    ],
    
    # Confusion matrix
    'compute_confusion_matrix': True,
    
    # Feature importance
    'compute_feature_importance': True,
    
    # Metrics description
    'metrics_description': {
        'accuracy': 'Overall accuracy - proportion of correct predictions',
        'precision_weighted': 'Weighted average precision across all classes',
        'recall_weighted': 'Weighted average recall across all classes',
        'f1_weighted': 'Weighted average F1-score across all classes',
        'roc_auc_ovr': 'Area under ROC curve using One-vs-Rest strategy',
    },
    
    # Class labels (severity levels)
    'class_labels': {
        1: 'Unscathed',
        2: 'Light injury',
        3: 'Hospitalized wounded',
        4: 'Killed'
    }
}


# ============================================================
# MODEL PERSISTENCE
# ============================================================

PERSISTENCE_CONFIG = {
    # Model versioning
    'versioning_strategy': 'timestamp',  # Options: 'timestamp', 'semantic', 'hash'
    
    # Model storage
    'model_dir': 'models/',
    'model_format': 'joblib',  # Options: 'joblib', 'pickle'
    
    # Model metadata
    'save_metadata': True,
    'metadata_format': 'json',
    
    # Model artifacts to save
    'save_artifacts': [
        'model',              # Trained model
        'feature_names',      # Feature names in correct order
        'scaler',             # If feature scaling is used
        'encoder',            # If categorical encoding is used
        'metrics',            # Training and validation metrics
        'confusion_matrix',   # Confusion matrix on validation set
        'feature_importance', # Feature importance scores
        'config'              # Complete configuration used for training
    ]
}


# ============================================================
# LOGGING AND MONITORING
# ============================================================

LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': 'logs/model_training.log',
    
    # Training progress
    'log_training_progress': True,
    'log_interval': 10,  # Log every N iterations
    
    # Metrics logging
    'log_metrics': True,
    'log_feature_importance': True,
}


# ============================================================
# EXPERIMENT TRACKING (Optional - for future MLflow integration)
# ============================================================

EXPERIMENT_CONFIG = {
    'use_mlflow': False,
    'experiment_name': 'accident_severity_prediction',
    'tracking_uri': None,  # Set to MLflow tracking server URI if available
}
