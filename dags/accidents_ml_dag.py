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

from datetime import datetime, timedelta
from typing import Dict, Any

# Only import lightweight modules at top level for fast DAG parsing
from airflow.sdk import dag, task, setup, teardown, Variable
from airflow.sdk.bases.operator import chain

# Heavy imports moved inside task functions to speed up DAG parsing
# Don't import pandas, mlflow, sklearn, or custom modules at top level


@dag(
    dag_id="accidents_ml_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'training', 'ml-pipeline'],
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
    }
)
def accidents_model_training():
    """Model Training DAG - for retraining when data is already available."""

    @setup
    def initialize():
        """Initialize MLflow tracking and configuration."""
        from src.utils import logging
        from src.utils.ml_utils import (
            MODEL_NAME,
            CHAMPION_MODEL_ALIAS,
            BEST_MODEL_METRIC
        )
        
        logging.info('Initializing Model Training DAG')
        
        # Don't call setup_mlflow_tracking() here - it connects to MLflow during DAG parsing
        # Each task calls it when executed instead
        
        Variable.set('model_name', MODEL_NAME)
        Variable.set('champion_alias', CHAMPION_MODEL_ALIAS)
        Variable.set('best_model_metric', BEST_MODEL_METRIC)
        
        return {'status': 'initialized', 'timestamp': datetime.now().isoformat()}

    @task()
    def prepare_datasets() -> Dict[str, Any]:
        """Load and prepare training, validation, and test datasets with versioning.
        
        Creates a parent MLflow run for the entire pipeline. All subsequent tasks
        will create nested runs under this parent.
        """
        import mlflow
        from src.utils import logging
        from src.utils.ml_utils import (
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            setup_mlflow_tracking,
            load_training_data_from_db,
            get_data_version
        )
        
        logging.info('Preparing datasets for training...')
        
        data_version = get_data_version()
        
        train_df = load_training_data_from_db('train')
        val_df = load_training_data_from_db('validation')
        test_df = load_training_data_from_db('test')
        
        # Setup MLflow right before using it
        setup_mlflow_tracking()
        
        # Create parent run for the entire ML pipeline
        with mlflow.start_run(run_name=f'ml_pipeline_v{data_version}') as parent_run:
            parent_run_id = parent_run.info.run_id
            
            mlflow.set_tag('pipeline_stage', 'complete_pipeline')
            mlflow.set_tag('data_version', data_version)
            
            # Create nested run for dataset versioning
            with mlflow.start_run(run_name='dataset_versioning', nested=True):
                mlflow.log_params({
                    'data_version': data_version,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'test_size': len(test_df),
                    'n_features': len(FEATURE_COLUMNS),
                })
                
                for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                    dist = df[TARGET_COLUMN].value_counts().to_dict()
                    for severity, count in dist.items():
                        mlflow.log_metric(f'{split_name}_class_{severity}_count', count)
        
        dataset_info = {
            'data_version': data_version,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'parent_run_id': parent_run_id,  # Pass to subsequent tasks for nesting
        }
        
        logging.info(f'Datasets prepared: {dataset_info}')
        logging.info(f'Parent MLflow run ID: {parent_run_id}')
        return dataset_info

    @task()
    def train_model(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter tuning using GridSearchCV."""
        import pandas as pd
        import mlflow
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from src.utils import logging
        from src.utils.ml_utils import (
            MODEL_NAME,
            RANDOM_STATE,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            setup_mlflow_tracking,
            load_training_data_from_db,
            prepare_features_and_target
        )
        
        logging.info('Starting model training with GridSearchCV...')
        
        data_version = dataset_info['data_version']
        
        train_df = load_training_data_from_db('train')
        X_train, y_train = prepare_features_and_target(train_df)
        
        # Setup MLflow right before using it
        setup_mlflow_tracking()
        
        parent_run_id = dataset_info.get('parent_run_id')
        
        # Resume parent run and create nested training run
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name='model_training', nested=True) as train_run:
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
                
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                }
                
                model = RandomForestClassifier(random_state=RANDOM_STATE)
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
                mlflow.log_metric('cv_best_score', grid_search.best_score_)
                
                model_info = mlflow.sklearn.log_model(
                    best_model,
                    MODEL_NAME,
                    registered_model_name=MODEL_NAME
                )
                
                training_run_id = train_run.info.run_id
                
                logging.info(f'Best params: {best_params}')
                logging.info(f'CV Best score: {grid_search.best_score_:.4f}')
                logging.info(f'Model URI: {model_info.model_uri}')
        
        return {
            'run_id': training_run_id,
            'parent_run_id': parent_run_id,
            'model_uri': model_info.model_uri,
            'model_version': model_info.registered_model_version,
            'best_params': best_params,
            'cv_best_score': grid_search.best_score_,
            'data_version': data_version
        }

    @task()
    def evaluate_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model on validation and test datasets."""
        import pandas as pd
        import mlflow
        from src.utils import logging
        from src.utils.ml_utils import (
            BEST_MODEL_METRIC,
            TARGET_COLUMN,
            setup_mlflow_tracking,
            load_training_data_from_db,
            prepare_features_and_target
        )
        from src.models.metrics import (
            weighted_f1_score_metric,
            weighted_precision_metric,
            weighted_recall_metric,
            roc_auc_ovr_metric,
            per_class_f1_metric
        )
        
        logging.info('Evaluating model...')
        
        model_uri = training_result['model_uri']
        data_version = training_result['data_version']
        
        val_df = load_training_data_from_db('validation')
        test_df = load_training_data_from_db('test')
        
        X_val, y_val = prepare_features_and_target(val_df)
        X_test, y_test = prepare_features_and_target(test_df)
        
        # Setup MLflow right before using it
        setup_mlflow_tracking()
        
        parent_run_id = training_result.get('parent_run_id')
        
        extra_metrics = [
            weighted_f1_score_metric,
            weighted_precision_metric,
            weighted_recall_metric,
            roc_auc_ovr_metric,
            per_class_f1_metric
        ]
        
        # Resume parent run and create nested evaluation run
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name='model_evaluation', nested=True):
                val_data = pd.concat([X_val, y_val], axis=1)
                val_dataset = mlflow.data.from_pandas(
                    val_data, name=f"validation_data-v{data_version}", targets=TARGET_COLUMN
                )
                
                test_data = pd.concat([X_test, y_test], axis=1)
                test_dataset = mlflow.data.from_pandas(
                    test_data, name=f"test_data-v{data_version}", targets=TARGET_COLUMN
                )
                
                # Evaluation config that disables all explainability features for performance
                # SHAP and other explanations are computed separately in generate_shap_explanations task
                fast_eval_config = {
                    'log_explainer': False,
                    'log_model_explainability': False,
                    'explainability_algorithm': None,
                }
                
                logging.info('Evaluating on validation set...')
                val_result = mlflow.models.evaluate(
                    model_uri, val_dataset, model_type="classifier",
                    evaluator_config=fast_eval_config,
                    extra_metrics=extra_metrics
                )
                
                logging.info('Evaluating on test set...')
                test_result = mlflow.models.evaluate(
                    model_uri, test_dataset, model_type="classifier",
                    evaluator_config=fast_eval_config,
                    extra_metrics=extra_metrics
                )
                
                def extract_metrics(result):
                    import math
                    metrics = {}
                    for k, v in result.metrics.items():
                        if isinstance(v, (int, float)):
                            # Convert NaN to None for JSON serialization
                            if math.isnan(v):
                                metrics[k] = None
                            else:
                                metrics[k] = float(v)
                    return metrics
                
                val_metrics = extract_metrics(val_result)
                test_metrics = extract_metrics(test_result)
                
                logging.info(f'Validation {BEST_MODEL_METRIC}: {val_metrics.get(BEST_MODEL_METRIC, "N/A")}')
                logging.info(f'Test {BEST_MODEL_METRIC}: {test_metrics.get(BEST_MODEL_METRIC, "N/A")}')
        
        return {
            'model_uri': model_uri,
            'model_version': training_result['model_version'],
            'parent_run_id': parent_run_id,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_metric_value': test_metrics.get(BEST_MODEL_METRIC, 0.0)
        }

    @task()
    def register_and_alias_model(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with champion model and promote current model if better."""
        import pandas as pd
        import mlflow
        from src.utils import logging
        from src.utils.ml_utils import (
            TARGET_COLUMN,
            setup_mlflow_tracking,
            load_training_data_from_db,
            prepare_features_and_target,
            compare_with_champion
        )
        
        logging.info('Checking champion model and comparing...')
        
        model_version = evaluation_result['model_version']
        current_metric = evaluation_result['best_metric_value']
        
        test_df = load_training_data_from_db('test')
        X_test, y_test = prepare_features_and_target(test_df)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Setup MLflow right before using it
        setup_mlflow_tracking()
        
        test_dataset = mlflow.data.from_pandas(
            test_data, name="test_data_champion_eval", targets=TARGET_COLUMN
        )
        
        result = compare_with_champion(
            current_model_version=model_version,
            current_metric_value=current_metric,
            test_dataset=test_dataset
        )
        
        return result

    @task()
    def generate_drift_report(registration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data drift report using DriftDetector from src/monitoring/drift.py."""
        import os
        from pathlib import Path
        from src.utils import logging
        from src.utils.ml_utils import (
            load_training_data_from_db,
            prepare_features_and_target
        )
        from src.monitoring.drift import DriftDetector
        
        logging.info('Generating drift report using DriftDetector...')
        
        train_df = load_training_data_from_db('train')
        test_df = load_training_data_from_db('test')
        
        X_train, _ = prepare_features_and_target(train_df)
        X_test, _ = prepare_features_and_target(test_df)
        
        # Use Airflow logs directory which is writable by workers
        reports_dir = os.getenv('REPORTS_DIR', '/opt/airflow/logs/drift_reports')
        
        # Ensure directory exists
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        
        drift_detector = DriftDetector(
            reference_data=X_train,
            reports_dir=reports_dir
        )
        
        report_path = drift_detector.generate_full_report(
            current_data=X_test,
            include_target_drift=False
        )
        
        drift_info = {
            'report_path': report_path,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'reference_samples': len(X_train),
            'current_samples': len(X_test),
            'registration_status': registration_result.get('status', 'unknown')
        }
        
        logging.info(f'Drift report saved: {report_path}')
        
        return drift_info

    @teardown()
    def finalize():
        """Clean up and finalize the DAG run."""
        from src.utils import logging
        
        logging.info('Finalizing Model Training DAG')
        
        try:
            Variable.delete('model_name')
            Variable.delete('champion_alias')
            Variable.delete('best_model_metric')
        except Exception as e:
            logging.warning(f'Cleanup warning: {e}')
        
        logging.info('Model training pipeline completed successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    @task()
    def generate_shap_explanations(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP-based model explanations and log to MLflow."""
        import os
        from pathlib import Path
        import mlflow
        from src.utils import logging
        from src.utils.ml_utils import (
            FEATURE_COLUMNS,
            setup_mlflow_tracking,
            load_training_data_from_db,
            prepare_features_and_target
        )
        
        logging.info('Generating SHAP explanations...')
        
        # Load data
        train_df = load_training_data_from_db('train')
        test_df = load_training_data_from_db('test')
        X_train, y_train = prepare_features_and_target(train_df)
        X_test, _ = prepare_features_and_target(test_df)
        
        # Setup MLflow and load the trained model
        setup_mlflow_tracking()
        
        model_uri = training_result['model_uri']
        parent_run_id = training_result.get('parent_run_id')
        model = mlflow.sklearn.load_model(model_uri)
        
        # Use Airflow logs directory for reports
        reports_dir = os.getenv('REPORTS_DIR', '/opt/airflow/logs/shap_reports')
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            from src.monitoring.explainability import SHAPExplainer
            
            explainer = SHAPExplainer(
                model=model,
                feature_names=FEATURE_COLUMNS,
                reports_dir=reports_dir
            )
            
            # Compute feature importance on training data
            train_importance = explainer.get_feature_importance(X_train)
            logging.info(f'Top 5 features: {list(train_importance.items())[:5]}')
            
            # Generate plots
            importance_plot = explainer.generate_importance_bar_plot(X_train, max_display=15)
            
            # Set reference and compute drift against test data
            explainer.set_reference_importance(X_train)
            drift_results = explainer.compute_importance_drift(X_test)
            drift_plot = explainer.generate_drift_comparison_plot(X_test, max_display=15)
            
            # Resume parent run and create nested SHAP run
            with mlflow.start_run(run_id=parent_run_id):
                with mlflow.start_run(run_name='shap_explanations', nested=True):
                    # Log importance as metrics
                    for feature, score in list(train_importance.items())[:10]:
                        safe_name = feature.replace(' ', '_').replace('-', '_')
                        mlflow.log_metric(f'shap_importance_{safe_name}', score)
                    
                    # Log plots as artifacts
                    mlflow.log_artifact(importance_plot, 'shap')
                    mlflow.log_artifact(drift_plot, 'shap')
            
            shap_info = {
                'feature_importance': train_importance,
                'importance_plot': importance_plot,
                'drift_plot': drift_plot,
                'top_changed_features': drift_results.get('top_changed_features', []),
                'status': 'success'
            }
            
            logging.info(f'SHAP explanations generated successfully')
            return shap_info
            
        except ImportError as e:
            logging.warning(f'SHAP not available: {e}. Skipping SHAP explanations.')
            return {'status': 'skipped', 'reason': str(e)}
        except Exception as e:
            logging.error(f'Error generating SHAP explanations: {e}')
            return {'status': 'error', 'error': str(e)}

    # ==============================
    # Define DAG Flow
    # ==============================
    init_task = initialize()
    dataset_task = prepare_datasets()
    train_task = train_model(dataset_task)
    eval_task = evaluate_model(train_task)
    register_task = register_and_alias_model(eval_task)
    drift_task = generate_drift_report(register_task)
    shap_task = generate_shap_explanations(train_task)
    final_task = finalize()
    
    # SHAP runs in parallel with evaluation/registration/drift
    chain(init_task, dataset_task, train_task, [eval_task, shap_task])
    chain(eval_task, register_task, drift_task, final_task)


accidents_model_training()
