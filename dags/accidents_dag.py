"""
Accidents Complete Pipeline DAG

This DAG is the exact combination of the data pipeline (accidents_data_dag)
and the ML pipeline (accidents_ml_dag). The task code below matches those
two DAGs; it simply executes the data pipeline first, then the ML pipeline,
followed by drift reporting.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow.sdk import dag, task, setup, teardown, Variable
from airflow.sdk.bases.operator import chain
from airflow.sdk import TaskGroup
from src.data.ingest_data import reset_progress

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


@dag(
    dag_id="accidents_pipeline",
    schedule='@hourly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'accidents', 'data-pipeline', 'ml-pipeline', 'complete'],
    params={
        "loading_mode": "chunked",
        "chunk_size": 10000,
    },
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
    }
)
def accidents_pipeline():
    """Run data pipeline then ML pipeline using the same task code as the individual DAGs."""

    # ==============================
    # Data Pipeline (from accidents_data_dag.py)
    # ==============================
    @setup
    def initialize_data():
        from src.utils import logging
        from src.utils.ml_utils import RANDOM_STATE

        logging.info('Initializing Data Pipeline DAG')

        Variable.set('train_ratio', str(TRAIN_RATIO))
        Variable.set('val_ratio', str(VAL_RATIO))
        Variable.set('test_ratio', str(TEST_RATIO))
        Variable.set('random_state', str(RANDOM_STATE))

        return {
            'status': 'initialized',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'train_ratio': TRAIN_RATIO,
                'val_ratio': VAL_RATIO,
                'test_ratio': TEST_RATIO
            }
        }

    @task()
    def download_data() -> Dict[str, Any]:
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
    def reset_progress_tracking(download_result: Dict[str, Any], **context) -> Dict[str, Any]:
        """Reset ingestion progress to use the newly downloaded data directory."""
        from src.utils import logging
        from src.data.ingest_data import DEFAULT_CHUNK_SIZE
        
        params = context.get('params', {})
        chunk_size = params.get('chunk_size', DEFAULT_CHUNK_SIZE)
        target_path = download_result.get('target_path')
        
        logging.info(f'Resetting progress tracking for directory: {target_path}')
        
        result = reset_progress(raw_data_path=target_path, chunk_size=chunk_size)
        
        if not result.get('success', False):
            raise Exception(f"Failed to reset progress: {result.get('error', 'Unknown error')}")
        
        logging.info('Progress tracking reset successfully')
        
        return {
            'status': 'success',
            'csv_directory': result.get('csv_directory'),
            'chunk_size': chunk_size,
            'download_result': download_result
        }

    @task()
    def ingest_data(reset_result: Dict[str, Any], **context) -> Dict[str, Any]:
        from src.utils import logging
        from src.data.ingest_data import load_next_chunk, ingest_data_full, DEFAULT_CHUNK_SIZE

        params = context.get('params', {})
        loading_mode = params.get('loading_mode', 'chunked')
        chunk_size = params.get('chunk_size', DEFAULT_CHUNK_SIZE)
        download_result = reset_result.get('download_result', {})

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

        # Chunked loading - loads ONE chunk of data per DAG run
        logging.info(f'Loading next chunk (size={chunk_size})...')

        result = load_next_chunk(chunk_size=chunk_size)

        if not result.get('success', False):
            error_msg = result.get('message', 'Unknown error')
            raise Exception(f"Ingestion failed: {error_msg}")

        tables = result.get('tables', {})
        all_complete = result.get('all_complete', False)

        total_records = {}
        for table, table_result in tables.items():
            if isinstance(table_result, dict):
                total_records[table] = {
                    'loaded': table_result.get('loaded', 0),
                    'total_loaded': table_result.get('total_loaded', 0),
                    'progress': table_result.get('progress_percentage', 0)
                }
            else:
                total_records[table] = {'loaded': table_result}

        logging.info(f'Chunk loaded: {[(t, r.get("loaded", 0)) for t, r in total_records.items()]}')

        if all_complete:
            logging.info('All data has been loaded!')
        else:
            logging.info('Run the DAG again to load the next chunk.')

        return {
            'status': 'success',
            'mode': 'chunked',
            'all_complete': all_complete,
            'total_records': total_records,
            'download_timestamp': download_result.get('timestamp')
        }

    @task()
    def clean_data(ingest_result: Dict[str, Any], **context) -> Dict[str, Any]:
        from src.utils import logging
        from src.utils.database import get_db_connection
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

            class_query = """
                SELECT dataset_split, severity, COUNT(*) as count 
                FROM clean_data 
                WHERE is_current = TRUE 
                GROUP BY dataset_split, severity
                ORDER BY dataset_split, severity
            """
            class_df = pd.read_sql_query(class_query, conn)

            null_query = "SELECT COUNT(*) as count FROM clean_data WHERE is_current = TRUE AND dataset_split IS NULL"
            null_df = pd.read_sql_query(null_query, conn)
            null_count = int(null_df['count'].iloc[0])

            validation_result = {
                'status': 'success',
                'total_records': total_count,
                'split_distribution': split_dist,
                'null_splits': null_count,
                'class_distribution_by_split': class_df.to_dict('records'),
                'is_valid': null_count == 0 and total_count > 0
            }

            if not validation_result['is_valid']:
                logging.warning('Data validation issues detected!')
                logging.warning(f'Null splits: {null_count}, Total records: {total_count}')
            else:
                logging.info('Data validation passed!')
                logging.info(f'Total: {total_count}, Splits: {split_dist}')

            return validation_result

    @teardown()
    def finalize_data():
        from src.utils import logging

        logging.info('Finalizing Data Pipeline DAG')

        try:
            Variable.delete('train_ratio')
            Variable.delete('val_ratio')
            Variable.delete('test_ratio')
            Variable.delete('random_state')
        except Exception as e:
            logging.warning(f'Cleanup warning: {e}')

        logging.info('Data pipeline completed successfully.')
        return {'status': 'completed', 'timestamp': datetime.now().isoformat()}

    # ==============================
    # ML Pipeline (from accidents_ml_dag.py)
    # ==============================
    @setup
    def initialize_ml():
        from src.utils import logging
        from src.utils.ml_utils import (
            MODEL_NAME,
            CHAMPION_MODEL_ALIAS,
            BEST_MODEL_METRIC
        )

        logging.info('Initializing Model Training DAG')

        Variable.set('model_name', MODEL_NAME)
        Variable.set('champion_alias', CHAMPION_MODEL_ALIAS)
        Variable.set('best_model_metric', BEST_MODEL_METRIC)

        return {'status': 'initialized', 'timestamp': datetime.now().isoformat()}

    @task()
    def prepare_datasets() -> Dict[str, Any]:
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
            'parent_run_id': parent_run_id,
        }

        logging.info(f'Datasets prepared: {dataset_info}')
        logging.info(f'Parent MLflow run ID: {parent_run_id}')
        return dataset_info

    @task()
    def train_model(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
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

                # Evaluation config that disables all explainability features for performance
                # SHAP and other explanations are computed separately in generate_shap_explanations task
                # NOTE: These settings prevent duplicate metric insertion errors in MLflow backend
                fast_eval_config = {
                    'log_explainer': False,
                    'log_model_explainability': False,
                    'explainability_algorithm': None,
                    'log_shap_values': False,
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
    def register_and_alias_model(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
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

        # Pass through parent_run_id for downstream tasks
        result['parent_run_id'] = evaluation_result.get('parent_run_id')

        return result

    @task()
    def generate_drift_report(registration_result: Dict[str, Any]) -> Dict[str, Any]:
        import os
        from pathlib import Path
        import mlflow
        from src.utils import logging
        from src.utils.ml_utils import (
            setup_mlflow_tracking,
            load_training_data_from_db,
            prepare_features_and_target
        )
        from src.monitoring.drift import DriftDetector
        from src.monitoring.drift_reporter import compute_and_submit_drift

        logging.info('Generating optimized drift report...')

        train_df = load_training_data_from_db('train')
        test_df = load_training_data_from_db('test')

        X_train, _ = prepare_features_and_target(train_df)
        X_test, _ = prepare_features_and_target(test_df)

        reports_dir = os.getenv('REPORTS_DIR', '/opt/airflow/logs/drift_reports')
        Path(reports_dir).mkdir(parents=True, exist_ok=True)

        setup_mlflow_tracking()
        parent_run_id = registration_result.get('parent_run_id')

        # Optimized drift detection with:
        # - max_sample_size=2000 (vs 11252+3751 full datasets) for ~6x speedup
        # - save_html=False (skip HTML generation) for ~30% faster
        # - save_json=True (JSON export for efficient storage)
        # - log_to_mlflow=True (automatic metric/plot logging)
        drift_detector = DriftDetector(
            reference_data=X_train,
            reports_dir=reports_dir,
            max_sample_size=2000
        )

        # Generate report with MLflow integration
        drift_report = drift_detector.generate_full_report(
            current_data=X_test,
            include_target_drift=False,
            save_html=False,  # Skip HTML for performance
            save_json=True,   # Save JSON for programmatic access
            log_to_mlflow=True,  # Log metrics/plots to MLflow
            mlflow_run_id=parent_run_id
        )

        # Submit drift metrics to API/Prometheus
        drift_result = compute_and_submit_drift(drift_results=drift_report, logger=logging)
        
        drift_info = {
            'json_path': drift_report.get('json_path'),
            'drift_plot_path': drift_report.get('drift_plot_path'),
            'timestamp': drift_report.get('timestamp'),
            'reference_samples': drift_report.get('reference_samples'),
            'current_samples': drift_report.get('current_samples'),
            'computation_time': drift_report.get('computation_time_seconds'),
            'overall_drift_score': drift_report.get('overall_drift_score'),
            'feature_drift_scores': drift_report.get('feature_drift_scores'),
            'is_drift_detected': drift_report.get('is_drift_detected'),
            'metrics_submitted': drift_result.get('submitted', False),
            'registration_status': registration_result.get('status', 'unknown')
        }

        logging.info(f'Drift detection complete: score={drift_info["overall_drift_score"]:.3f}, '
                     f'time={drift_info["computation_time"]:.2f}s')

        return drift_info

    @task()
    def generate_shap_explanations(training_result: Dict[str, Any]) -> Dict[str, Any]:
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

        train_df = load_training_data_from_db('train')
        test_df = load_training_data_from_db('test')
        X_train, y_train = prepare_features_and_target(train_df)
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

            # Compute feature importance with optimized SHAP parameters
            # sample_size=500 (50% reduction), approximate=True (~10x faster), check_additivity=False (skip validation)
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

    @teardown()
    def finalize_ml():
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

    # ==============================
    # Flow (grouped by original DAGs)
    # ==============================
    with TaskGroup(group_id="data_pipeline", tooltip="Flow from accidents_data_dag.py") as data_pipeline_group:
        data_init = initialize_data()
        download = download_data()
        reset = reset_progress_tracking(download)
        ingest = ingest_data(reset)
        clean = clean_data(ingest)
        split = assign_splits_task(clean)
        validate = validate_data(split)
        data_done = finalize_data()

        chain(data_init, download, reset, ingest, clean, split, validate, data_done)

    with TaskGroup(group_id="ml_pipeline", tooltip="Flow from accidents_ml_dag.py") as ml_pipeline_group:
        ml_init = initialize_ml()
        datasets = prepare_datasets()
        train = train_model(datasets)
        eval = evaluate_model(train)
        shap = generate_shap_explanations(train)
        register = register_and_alias_model(eval)
        drift = generate_drift_report(register)
        ml_done = finalize_ml()

        # SHAP runs in parallel with evaluation/registration/drift
        chain(ml_init, datasets, train, [eval, shap])
        chain(eval, register, drift, ml_done)

    # Run ML pipeline after data pipeline completes
    data_done >> ml_init


accidents_pipeline()
