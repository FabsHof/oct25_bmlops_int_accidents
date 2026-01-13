"""
Data drift detection using Evidently.

This module provides drift detection capabilities for monitoring
data distribution changes between reference and production data.

Compatible with Evidently 0.7.0+
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
import logging
from pathlib import Path
from datetime import datetime

from src.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift between reference and current data using Evidently.
    
    This class provides methods to:
    - Set reference data for comparison
    - Analyze current data for drift
    - Generate drift reports
    - Update Prometheus metrics with drift scores
    
    Compatible with Evidently 0.7.0+
    """
    
    # Feature columns for drift detection
    NUMERICAL_FEATURES = [
        'year', 'month', 'hour', 'minute', 'year_of_birth',
        'latitude', 'longitude'
    ]
    
    CATEGORICAL_FEATURES = [
        'user_category', 'sex', 'trip_purpose', 'security',
        'luminosity', 'weather', 'type_of_road', 'road_surface', 'holiday'
    ]
    
    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.5,
        reports_dir: Optional[str] = None,
        max_sample_size: int = 2000
    ):
        """
        Initialize the drift detector with performance optimizations.
        
        Args:
            reference_data: Reference dataset for comparison
            drift_threshold: Threshold for drift detection (0-1)
            reports_dir: Directory to save drift reports
            max_sample_size: Maximum samples for drift computation (default 2000 for performance)
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reports_dir = Path(reports_dir) if reports_dir else Path("logs/drift_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.max_sample_size = max_sample_size
        
        # Buffer for collecting current data
        self._current_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Analyze drift every N predictions
    
    def _sample_data(self, df: pd.DataFrame, max_size: Optional[int] = None) -> pd.DataFrame:
        """
        Sample data for efficient drift detection.
        
        Args:
            df: Input DataFrame
            max_size: Maximum sample size (uses self.max_sample_size if None)
            
        Returns:
            Sampled DataFrame
        """
        sample_size = max_size or self.max_sample_size
        if len(df) > sample_size:
            sampled = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} from {len(df)} rows for drift detection")
            return sampled
        return df
    
    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Create an Evidently Dataset from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Evidently Dataset object
        """
        available_num = [col for col in self.NUMERICAL_FEATURES if col in df.columns]
        available_cat = [col for col in self.CATEGORICAL_FEATURES if col in df.columns]
        available_cols = available_num + available_cat
        
        # Create DataDefinition with explicit column types
        data_definition = DataDefinition(
            numerical_columns=available_num,
            categorical_columns=available_cat
        )
        
        return Dataset.from_pandas(df[available_cols], data_definition=data_definition)
    
    def set_reference_data(self, data: pd.DataFrame):
        """
        Set the reference dataset for drift comparison.
        
        Args:
            data: Reference DataFrame with feature columns
        """
        # Ensure all expected columns exist
        missing_cols = set(self.ALL_FEATURES) - set(data.columns)
        if missing_cols:
            logger.warning(f"Reference data missing columns: {missing_cols}")
        
        self.reference_data = data[
            [col for col in self.ALL_FEATURES if col in data.columns]
        ].copy()
        logger.info(f"Reference data set with {len(self.reference_data)} samples")
    
    def load_reference_from_csv(self, filepath: str):
        """
        Load reference data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        df = pd.read_csv(filepath)
        self.set_reference_data(df)
    
    def add_prediction_data(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Add prediction features to the current data buffer.
        
        When the buffer reaches the configured size, automatically
        triggers drift analysis.
        
        Args:
            features: Dictionary of feature values from a prediction request
            
        Returns:
            Drift analysis results if buffer is full, None otherwise
        """
        # Filter to only include known features
        filtered_features = {
            k: v for k, v in features.items()
            if k in self.ALL_FEATURES
        }
        self._current_buffer.append(filtered_features)
        
        if len(self._current_buffer) >= self._buffer_size:
            return self.analyze_drift()
        
        return None
    
    def analyze_drift(self) -> Dict[str, Any]:
        """
        Analyze drift between reference and current buffered data.
        
        Returns:
            Dictionary containing drift analysis results:
            - overall_drift: Overall dataset drift score
            - is_drift_detected: Boolean indicating if drift exceeds threshold
            - feature_drifts: Per-feature drift scores
            - report_path: Path to saved HTML report
        """
        if self.reference_data is None:
            logger.warning("No reference data set, skipping drift analysis")
            return {"error": "No reference data available"}
        
        if not self._current_buffer:
            logger.warning("No current data in buffer, skipping drift analysis")
            return {"error": "No current data available"}
        
        # Convert buffer to DataFrame
        current_data = pd.DataFrame(self._current_buffer)
        
        # Ensure columns match reference data
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                current_data[col] = np.nan
        
        current_data = current_data[self.reference_data.columns]
        
        # Create Evidently datasets
        ref_dataset = self._create_dataset(self.reference_data)
        cur_dataset = self._create_dataset(current_data)
        
        # Create and run Evidently report
        report = Report([DataDriftPreset()])
        
        try:
            # Evidently 0.7.0+: run(current_data, reference_data) returns a Snapshot
            snapshot = report.run(cur_dataset, ref_dataset)
        except Exception as e:
            logger.error(f"Error running drift analysis: {e}")
            return {"error": str(e)}
        
        # Extract results from snapshot
        result = snapshot.dict()
        
        # Parse drift metrics from Evidently 0.7.0 format
        overall_drift_share = 0.0
        is_drift_detected = False
        feature_drifts = {}
        
        # Navigate the result structure to extract metrics
        metrics = result.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            if "share_of_drifted_columns" in metric_result:
                overall_drift_share = metric_result.get("share_of_drifted_columns", 0.0)
                is_drift_detected = metric_result.get("dataset_drift", False)
            if "drift_by_columns" in metric_result:
                for feature, info in metric_result.get("drift_by_columns", {}).items():
                    if isinstance(info, dict):
                        feature_drifts[feature] = info.get("drift_score", 0.0)
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"drift_report_{timestamp}.html"
        snapshot.save_html(str(report_path))
        
        # Update Prometheus metrics
        try:
            metrics_collector = get_metrics_collector()
            metrics_collector.update_drift_metrics(
                overall_drift=overall_drift_share,
                feature_drifts=feature_drifts,
                is_drift_detected=is_drift_detected
            )
        except Exception as e:
            logger.warning(f"Could not update Prometheus metrics: {e}")
        
        # Clear buffer
        self._current_buffer = []
        
        analysis_result = {
            "overall_drift": overall_drift_share,
            "is_drift_detected": is_drift_detected,
            "feature_drifts": feature_drifts,
            "samples_analyzed": len(current_data),
            "report_path": str(report_path)
        }
        
        logger.info(f"Drift analysis complete: drift_detected={is_drift_detected}, score={overall_drift_share:.3f}")
        
        return analysis_result
    
    def generate_full_report(
        self,
        current_data: pd.DataFrame,
        include_target_drift: bool = False,
        target_column: Optional[str] = None,
        save_html: bool = False,
        save_json: bool = True,
        log_to_mlflow: bool = False,
        mlflow_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate optimized drift report with optional HTML/JSON export and MLflow logging.
        
        Performance optimizations:
        - Samples data to max_sample_size for faster computation
        - Skips HTML generation by default (can be enabled)
        - Exports JSON for efficient storage
        - Logs metrics and plots to MLflow
        
        Args:
            current_data: Current production data
            include_target_drift: Whether to include target drift analysis
            target_column: Name of the target column if analyzing target drift
            save_html: Whether to save HTML report (default False for performance)
            save_json: Whether to save JSON report (default True)
            log_to_mlflow: Whether to log metrics/plots to MLflow (default False)
            mlflow_run_id: MLflow run ID for logging (uses active run if None)
            
        Returns:
            Dict containing:
            - 'report_path': Path to saved HTML report (if save_html=True)
            - 'json_path': Path to saved JSON report (if save_json=True)
            - 'overall_drift_score': Share of drifted columns (0-1)
            - 'feature_drift_scores': Dict of per-feature drift scores
            - 'is_drift_detected': Boolean indicating if drift detected
            - 'reference_samples': Number of reference samples used
            - 'current_samples': Number of current samples used
        """
        import time
        start_time = time.time()
        
        if self.reference_data is None:
            raise ValueError("Reference data must be set before generating report")
        
        # Filter data to only include columns that exist in both
        available_cols = [col for col in self.reference_data.columns if col in current_data.columns]
        current_data_filtered = current_data[available_cols].copy()
        reference_data_filtered = self.reference_data[available_cols].copy()
        
        # Sample data for performance optimization
        reference_sampled = self._sample_data(reference_data_filtered)
        current_sampled = self._sample_data(current_data_filtered)
        
        # Create Evidently datasets
        ref_dataset = self._create_dataset(reference_sampled)
        cur_dataset = self._create_dataset(current_sampled)
        
        # Use DataDriftPreset for comprehensive report
        report = Report([DataDriftPreset()])
        
        # Evidently 0.7.0+: run(current_data, reference_data) returns a Snapshot
        logger.info("Computing drift metrics...")
        snapshot = report.run(cur_dataset, ref_dataset)
        computation_time = time.time() - start_time
        logger.info(f"Drift computation completed in {computation_time:.2f}s")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dict = {
            "timestamp": timestamp,
            "reference_samples": len(reference_sampled),
            "current_samples": len(current_sampled),
            "computation_time_seconds": computation_time
        }
        
        # Save HTML report (optional, for manual review)
        if save_html:
            html_path = self.reports_dir / f"drift_report_{timestamp}.html"
            snapshot.save_html(str(html_path))
            result_dict["report_path"] = str(html_path)
            logger.info(f"HTML report saved to {html_path}")
        
        # Save JSON report (default, for programmatic access)
        if save_json:
            json_path = self.reports_dir / f"drift_report_{timestamp}.json"
            json_str = snapshot.json()
            with open(json_path, 'w') as f:
                f.write(json_str)
            result_dict["json_path"] = str(json_path)
            logger.info(f"JSON report saved to {json_path}")
        
        # Extract drift metrics from snapshot
        snapshot_dict = snapshot.dict()
        overall_drift_score = 0.0
        is_drift_detected = False
        feature_drift_scores = {}
        
        # Parse drift metrics from Evidently result structure
        metrics = snapshot_dict.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            if "share_of_drifted_columns" in metric_result:
                overall_drift_score = metric_result.get("share_of_drifted_columns", 0.0)
                is_drift_detected = metric_result.get("dataset_drift", False)
            if "drift_by_columns" in metric_result:
                for feature, info in metric_result.get("drift_by_columns", {}).items():
                    if isinstance(info, dict):
                        feature_drift_scores[feature] = info.get("drift_score", 0.0)
        
        result_dict.update({
            "overall_drift_score": overall_drift_score,
            "feature_drift_scores": feature_drift_scores,
            "is_drift_detected": is_drift_detected
        })
        
        # Log to MLflow (optional)
        if log_to_mlflow:
            try:
                import mlflow
                import matplotlib.pyplot as plt
                
                with mlflow.start_run(run_id=mlflow_run_id, nested=True) if mlflow_run_id else mlflow.start_run(nested=True):
                    # Log scalar metrics
                    mlflow.log_metric("drift_overall_score", overall_drift_score)
                    mlflow.log_metric("drift_detected", 1.0 if is_drift_detected else 0.0)
                    mlflow.log_metric("drift_reference_samples", len(reference_sampled))
                    mlflow.log_metric("drift_current_samples", len(current_sampled))
                    mlflow.log_metric("drift_computation_time", computation_time)
                    
                    # Log top 10 feature drift scores
                    sorted_features = sorted(feature_drift_scores.items(), key=lambda x: x[1], reverse=True)
                    for feature, score in sorted_features[:10]:
                        safe_name = feature.replace(' ', '_').replace('-', '_')
                        mlflow.log_metric(f"drift_feature_{safe_name}", score)
                    
                    # Create and log drift bar plot
                    if feature_drift_scores:
                        top_features = sorted_features[:15]
                        features, scores = zip(*top_features)
                        
                        plt.figure(figsize=(10, 6))
                        plt.barh(range(len(features)), scores, align='center')
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('Drift Score')
                        plt.title('Top 15 Features by Drift Score')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        
                        plot_path = self.reports_dir / f"drift_features_{timestamp}.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        mlflow.log_artifact(str(plot_path), "drift")
                        result_dict["drift_plot_path"] = str(plot_path)
                    
                    # Log JSON report as artifact
                    if save_json and "json_path" in result_dict:
                        mlflow.log_artifact(result_dict["json_path"], "drift")
                    
                    logger.info("Drift metrics and plots logged to MLflow")
            except Exception as e:
                logger.warning(f"Could not log drift metrics to MLflow: {e}")
        
        logger.info(f"Drift detection complete: score={overall_drift_score:.3f}, detected={is_drift_detected}")
        
        return result_dict
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get the current status of the data buffer.
        
        Returns:
            Dictionary with buffer status information
        """
        return {
            "buffer_size": len(self._current_buffer),
            "max_buffer_size": self._buffer_size,
            "percentage_full": len(self._current_buffer) / self._buffer_size * 100,
            "has_reference_data": self.reference_data is not None,
            "reference_data_size": len(self.reference_data) if self.reference_data is not None else 0
        }


# Global drift detector instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """
    Get or create the global drift detector.
    
    Returns:
        DriftDetector instance
    """
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector
