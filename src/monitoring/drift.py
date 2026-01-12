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
        reports_dir: Optional[str] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference dataset for comparison
            drift_threshold: Threshold for drift detection (0-1)
            reports_dir: Directory to save drift reports
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reports_dir = Path(reports_dir) if reports_dir else Path("logs/drift_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer for collecting current data
        self._current_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Analyze drift every N predictions
    
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
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report and extract drift metrics.
        
        Args:
            current_data: Current production data
            include_target_drift: Whether to include target drift analysis
            target_column: Name of the target column if analyzing target drift
            
        Returns:
            Dict containing:
            - 'report_path': Path to saved HTML report
            - 'overall_drift_score': Share of drifted columns (0-1)
            - 'feature_drift_scores': Dict of per-feature drift scores
            - 'is_drift_detected': Boolean indicating if drift detected
        """
        if self.reference_data is None:
            raise ValueError("Reference data must be set before generating report")
        
        # Filter data to only include columns that exist in both
        available_cols = [col for col in self.reference_data.columns if col in current_data.columns]
        current_data_filtered = current_data[available_cols].copy()
        reference_data_filtered = self.reference_data[available_cols].copy()
        
        # Create Evidently datasets
        ref_dataset = self._create_dataset(reference_data_filtered)
        cur_dataset = self._create_dataset(current_data_filtered)
        
        # Use DataDriftPreset for comprehensive report
        report = Report([DataDriftPreset()])
        
        # Evidently 0.7.0+: run(current_data, reference_data) returns a Snapshot
        snapshot = report.run(cur_dataset, ref_dataset)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"full_drift_report_{timestamp}.html"
        snapshot.save_html(str(report_path))
        
        # Extract drift metrics from snapshot
        result = snapshot.dict()
        overall_drift_score = 0.0
        is_drift_detected = False
        feature_drift_scores = {}
        
        # Parse drift metrics from Evidently result structure
        metrics = result.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            if "share_of_drifted_columns" in metric_result:
                overall_drift_score = metric_result.get("share_of_drifted_columns", 0.0)
                is_drift_detected = metric_result.get("dataset_drift", False)
            if "drift_by_columns" in metric_result:
                for feature, info in metric_result.get("drift_by_columns", {}).items():
                    if isinstance(info, dict):
                        feature_drift_scores[feature] = info.get("drift_score", 0.0)
        
        logger.info(f"Full drift report saved to {report_path}")
        
        return {
            "report_path": str(report_path),
            "overall_drift_score": overall_drift_score,
            "feature_drift_scores": feature_drift_scores,
            "is_drift_detected": is_drift_detected
        }
    
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
