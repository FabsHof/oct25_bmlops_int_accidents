"""
Accidents Complete ML Pipeline DAG
===================================

This DAG combines the complete data pipeline and ML pipeline for traffic accidents
severity prediction. It includes data processing, model training, evaluation,
SHAP explanations, and drift monitoring.

Pipeline Stages:
1. Data Pipeline:
   - download_data: Download raw data from Kaggle
   - ingest_data: Ingest data into PostgreSQL (chunked or full mode)
   - clean_data: Clean and preprocess data with SCD Type 2
   - assign_splits: Assign train/validation/test splits
   - validate_data: Validate processed data quality

2. ML Pipeline:
   - prepare_datasets: Load and version datasets from database
   - train_model: Train Random Forest with GridSearchCV
   - evaluate_model: Evaluate on validation and test sets
   - generate_shap_explanations: Generate SHAP-based model explanations
   - register_model: Compare with champion and alias best model

3. Monitoring:
   - generate_drift_report: Generate data drift report using Evidently
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Only import lightweight modules at top level for fast DAG parsing
from airflow.sdk import dag, task, setup, task_group, teardown, Variable
from airflow.sdk.bases.operator import chain

# Heavy imports moved inside task functions to speed up DAG parsing
# Don't import pandas, mlflow, sklearn, kagglehub, or custom modules at top level

# ########### Configuration ###########
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


@dag(
    dag_id="accidents_pipeline",
    schedule='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'ml-pipeline', 'complete'],
    params={
        "loading_mode": "chunked",  # "chunked" or "full"
        "chunk_size": 10000,  # Only used in chunked mode
    },
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }
)
def accidents_ml_pipeline():
    """Complete Accidents ML Pipeline DAG - Data + Training + Monitoring."""

    @setup
    def initialize():
        """Initialize environment variables and configurations for the DAG."""
        from src.utils import logging
        from src.utils.ml_utils import (
            MODEL_NAME,
            CHAMPION_MODEL_ALIAS,
            BEST_MODEL_METRIC,
            RANDOM_STATE
        )
        
        logging.info('Initializing Complete Accidents ML Pipeline DAG')
        
        Variable.set('model_name', MODEL_NAME)
        Variable.set('champion_alias', CHAMPION_MODEL_ALIAS)
        Variable.set('best_model_metric', BEST_MODEL_METRIC)
        Variable.set('random_state', str(RANDOM_STATE))
        Variable.set('train_ratio', str(TRAIN_RATIO))
        Variable.set('val_ratio', str(VAL_RATIO))
        Variable.set('test_ratio', str(TEST_RATIO))
        
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
            import kagglehub as kh
            from src.utils import logging
            
            logging.info('Downloading dataset from Kaggle...')
            
            dataset_id = 'ahmedlahlou/accidents-in-france-from-2005-to-2016'
            
            try:
                downloaded_path = kh.dataset_download(dataset_id)
                logging.info(f'Downloaded to: {downloaded_path}')
            except Exception as e:
                raise Exception(f"Failed to download dataset: {e}")
            
            raw_data_path = os.getenv('RAW_DATA_PATH', '/app/data/raw/')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            target_dir = Path(raw_data_path) / timestamp
            target_dir.mkdir(parents=True, exist_ok=True)
            
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
        def ingest_data(download_result: Dict[str, Any], **context) -> Dict[str, Any]:
            """Ingest raw CSV data into PostgreSQL database."""
            from src.utils import logging
            from src.data.ingest_data import load_next_chunk, ingest_data_full, DEFAULT_CHUNK_SIZE
            
            params = context.get('params', {})
            loading_mode = params.get('loading_mode', 'chunked')
            chunk_size = params.get('chunk_size', DEFAULT_CHUNK_SIZE)
            
            logging.info(f'Starting data ingestion ({loading_mode} mode)...')
            logging.info(f'Source: {download_result.get("target_path")}')
            
            if loading_mode == 'full':
                logging.info('Loading all data in one batch...')
                
                try:
                    results = ingest_data_full(raw_data_path=download_result.get('target_path'))
                    total_records = sum(results.values())
                    logging.info(f'Data ingestion completed. Total records: {total_records}')
                    
                    return {
                        'status': 'success',
                        'mode': 'full',
                        'total_records': results,
                        'download_timestamp': download_result.get('timestamp')
                    }
                except Exception as e:
                    logging.error(f'Full batch ingestion failed: {e}')
                    raise
            else:
                # Chunked loading - loads chunks until complete
                logging.info(f'Loading data in chunks (size={chunk_size})...')
                
                all_complete = False
                total_loaded = {}
                
                while not all_complete:
                    result = load_next_chunk(chunk_size=chunk_size)
                    
                    if not result.get('success', False):
                        error_msg = result.get('message', 'Unknown error')
                        raise Exception(f"Ingestion failed: {error_msg}")
                    
                    tables = result.get('tables', {})
                    for table, table_result in tables.items():
                        if isinstance(table_result, dict):
                            loaded = table_result.get('loaded', 0)
                        else:
                            loaded = table_result
                        total_loaded[table] = total_loaded.get(table, 0) + loaded
                    
                    all_complete = result.get('all_complete', False)
                    logging.info(f"Chunk loaded: {tables}")
                
                logging.info(f'Data ingestion completed. Total: {total_loaded}')
                
                return {
                    'status': 'success',
                    'mode': 'chunked',
                    'all_complete': True,
                    'total_records': total_loaded,
                    'download_timestamp': download_result.get('timestamp')
                }

        @task()
        def clean_data(ingest_result: Dict[str, Any], **context) -> Dict[str, Any]:
            """Clean and preprocess raw data using SCD Type 2."""
            from src.utils import logging
            from src.data.clean_data import transform_data, transform_data_chunked
            
            params = context.get('params', {})
            loading_mode = params.get('loading_mode', 'chunked')
            chunk_size = params.get('chunk_size', 10000)
            
            logging.info(f'Starting data cleaning and transformation ({loading_mode} mode)...')
            
            if loading_mode == 'chunked':
                logging.info(f'Using chunked transformation (chunk_size={chunk_size})')
                result = transform_data_chunked(clear_existing=False, chunk_size=chunk_size)
            else:
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
        def assign_splits_task(clean_result: Dict[str, Any]) -> Dict[str, Any]:
            """Assign train/validation/test splits using stratified sampling."""
            from src.utils import logging
            from src.utils.database import get_db_connection
            from src.data.clean_data import assign_dataset_splits
            from src.utils.ml_utils import RANDOM_STATE
            
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
            import pandas as pd
            from src.utils import logging
            from src.utils.database import get_db_connection
            
            logging.info('Validating processed data...')
            
            with get_db_connection() as conn:
                total_query = "SELECT COUNT(*) as count FROM clean_data WHERE is_current = TRUE"
                total_df = pd.read_sql_query(total_query, conn)
                total_count = int(total_df['count'].iloc[0])
                
                split_query = """
                    SELECT dataset_split, COUNT(*) as count 
                    FROM clean_data 
                    WHERE is_current = TRUE 
                    GROUP BY dataset_split
                """
                split_df = pd.read_sql_query(split_query, conn)
                split_dist = dict(zip(split_df['dataset_split'], split_df['count']))
                
                null_query = "SELECT COUNT(*) as count FROM clean_data WHERE is_current = TRUE AND dataset_split IS NULL"
                null_df = pd.read_sql_query(null_query, conn)
                null_count = int(null_df['count'].iloc[0])
                
                validation_result = {
                    'status': 'success',
                    'total_records': total_count,
                    'split_distribution': split_dist,
                    'null_splits': null_count,
                    'is_valid': null_count == 0 and total_count > 0
                }
                
                if not validation_result['is_valid']:
                    logging.warning('Data validation issues detected!')
                    logging.warning(f'Null splits: {null_count}, Total records: {total_count}')
                else:
                    logging.info('Data validation passed!')
                    logging.info(f'Total: {total_count}, Splits: {split_dist}')
                
                return validation_result

        download_result = download_data()
        ingest_result = ingest_data(download_result)
        clean_result = clean_data(ingest_result)
        splits_result = assign_splits_task(clean_result)
        validation_result = validate_data(splits_result)
        
        return validation_result

    # ==============================
    # ML Pipeline Tasks
    # ==============================
    @task_group()
    def ml_pipeline(data_result: Dict[str, Any]):
        """Model training, evaluation, and registration pipeline."""

        @task()
        def prepare_datasets(data_validation: Dict[str, Any]) -> Dict[str, Any]:
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
            
            # Log data validation status from previous pipeline
            logging.info(f'Data validation result: {data_validation.get("status", "unknown")}')
            logging.info('Preparing datasets for training...')
            
            data_version = get_data_version()
            
            train_df = load_training_data_from_db('train')
            val_df = load_training_data_from_db('validation')
            test_df = load_training_data_from_db('test')
            
            setup_mlflow_tracking()
            
            # Create parent run for the entire ML pipeline
            with mlflow.start_run(run_name=f'accidents_pipeline_v{data_version}') as parent_run:
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
            import math
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
                    
                    logging.info('Evaluating on validation set...')
                    val_result = mlflow.models.evaluate(
                        model_uri, val_dataset, model_type="classifier",
                        evaluator_config={'log_explainer': True},
                        extra_metrics=extra_metrics
                    )
                    
                    logging.info('Evaluating on test set...')
                    test_result = mlflow.models.evaluate(
                        model_uri, test_dataset, model_type="classifier",
                        evaluator_config={'log_explainer': True},
                        extra_metrics=extra_metrics
                    )
                    
                    def extract_metrics(result):
                        metrics = {}
                        for k, v in result.metrics.items():
                            if isinstance(v, (int, float)):
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
        def generate_shap_explanations(training_result: Dict[str, Any]) -> Dict[str, Any]:
            """Generate SHAP-based model explanations and log to MLflow as nested run."""
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
            
            train_df = load_training_data_from_db('train')
            test_df = load_training_data_from_db('test')
            X_train, _ = prepare_features_and_target(train_df)
            X_test, _ = prepare_features_and_target(test_df)
            
            setup_mlflow_tracking()
            
            model_uri = training_result['model_uri']
            parent_run_id = training_result.get('parent_run_id')
            model = mlflow.sklearn.load_model(model_uri)
            
            reports_dir = os.getenv('REPORTS_DIR', '/opt/airflow/logs/shap_reports')
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                from src.monitoring.explainability import SHAPExplainer
                
                explainer = SHAPExplainer(
                    model=model,
                    feature_names=FEATURE_COLUMNS,
                    reports_dir=reports_dir
                )
                
                train_importance = explainer.get_feature_importance(X_train)
                logging.info(f'Top 5 features: {list(train_importance.items())[:5]}')
                
                importance_plot = explainer.generate_importance_bar_plot(X_train, max_display=15)
                
                explainer.set_reference_importance(X_train)
                drift_results = explainer.compute_importance_drift(X_test)
                drift_plot = explainer.generate_drift_comparison_plot(X_test, max_display=15)
                
                # Resume parent run and create nested SHAP run
                with mlflow.start_run(run_id=parent_run_id):
                    with mlflow.start_run(run_name='shap_explanations', nested=True):
                        for feature, score in list(train_importance.items())[:10]:
                            safe_name = feature.replace(' ', '_').replace('-', '_')
                            mlflow.log_metric(f'shap_importance_{safe_name}', score)
                        
                        mlflow.log_artifact(importance_plot, 'shap')
                        mlflow.log_artifact(drift_plot, 'shap')
                
                shap_info = {
                    'feature_importance': train_importance,
                    'importance_plot': importance_plot,
                    'drift_plot': drift_plot,
                    'top_changed_features': drift_results.get('top_changed_features', []),
                    'status': 'success'
                }
                
                logging.info('SHAP explanations generated successfully')
                return shap_info
                
            except ImportError as e:
                logging.warning(f'SHAP not available: {e}. Skipping SHAP explanations.')
                return {'status': 'skipped', 'reason': str(e)}
            except Exception as e:
                logging.error(f'Error generating SHAP explanations: {e}')
                return {'status': 'error', 'error': str(e)}

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

        # Instantiate tasks - prepare_datasets uses data_result to create dependency
        dataset_info = prepare_datasets(data_result)
        training_result = train_model(dataset_info)
        eval_result = evaluate_model(training_result)
        shap_result = generate_shap_explanations(training_result)
        registration_result = register_and_alias_model(eval_result)
        
        # Chain: train → [eval, shap] → register
        chain(training_result, [eval_result, shap_result])
        chain(eval_result, registration_result)
        
        return registration_result

    # ==============================
    # Monitoring Tasks
    # ==============================
    @task_group()
    def monitoring_pipeline(registration_result: Dict[str, Any]):
        """Drift detection and monitoring pipeline."""

        @task()
        def generate_drift_report(reg_result: Dict[str, Any]) -> Dict[str, Any]:
            """Generate data drift report using DriftDetector from src/monitoring/drift.py."""
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
            
            reports_dir = os.getenv('REPORTS_DIR', '/opt/airflow/logs/drift_reports')
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
                'registration_status': reg_result.get('status', 'unknown')
            }
            
            logging.info(f'Drift report saved: {report_path}')
            
            return drift_info

        return generate_drift_report(registration_result)

    @teardown()
    def finalize():
        """Clean up Airflow Variables and finalize the DAG run."""
        from src.utils import logging
        
        logging.info('Finalizing Complete Accidents ML Pipeline DAG')
        
        try:
            Variable.delete('model_name')
            Variable.delete('champion_alias')
            Variable.delete('best_model_metric')
            Variable.delete('random_state')
            Variable.delete('train_ratio')
            Variable.delete('val_ratio')
            Variable.delete('test_ratio')
        except Exception as e:
            logging.warning(f'Warning during cleanup: {e}')
        
        logging.info('Complete pipeline finished successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # Define DAG Flow
    # ==============================
    init_task = initialize()
    data_pipeline_result = data_pipeline()
    ml_pipeline_result = ml_pipeline(data_pipeline_result)
    monitoring_result = monitoring_pipeline(ml_pipeline_result)
    final_task = finalize()
    
    # Chain the complete flow
    chain(init_task, data_pipeline_result, ml_pipeline_result, monitoring_result, final_task)


accidents_ml_pipeline()
