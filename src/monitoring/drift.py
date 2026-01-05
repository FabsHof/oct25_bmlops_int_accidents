"""
Data drift detection using Evidently.

This module provides drift detection capabilities for monitoring
data distribution changes between reference and production data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
import json
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
        
        # Configure column mapping for Evidently
        self.column_mapping = ColumnMapping(
            numerical_features=self.NUMERICAL_FEATURES,
            categorical_features=self.CATEGORICAL_FEATURES,
        )
        
        # Buffer for collecting current data
        self._current_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Analyze drift every N predictions
    
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
        
        # Create and run Evidently report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        try:
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
        except Exception as e:
            logger.error(f"Error running drift analysis: {e}")
            return {"error": str(e)}
        
        # Extract results
        result = report.as_dict()
        
        # Parse drift metrics
        dataset_drift = result.get("metrics", [{}])[0].get("result", {})
        drift_table = result.get("metrics", [{}])[1].get("result", {}) if len(result.get("metrics", [])) > 1 else {}
        
        overall_drift_share = dataset_drift.get("share_of_drifted_columns", 0.0)
        is_drift_detected = dataset_drift.get("dataset_drift", False)
        
        # Extract per-feature drift
        feature_drifts = {}
        drift_by_columns = drift_table.get("drift_by_columns", {})
        for feature, info in drift_by_columns.items():
            if isinstance(info, dict):
                feature_drifts[feature] = info.get("drift_score", 0.0)
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"drift_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        # Update Prometheus metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.update_drift_metrics(
            overall_drift=overall_drift_share,
            feature_drifts=feature_drifts,
            is_drift_detected=is_drift_detected
        )
        
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
    ) -> str:
        """
        Generate a comprehensive drift report.
        
        Args:
            current_data: Current production data
            include_target_drift: Whether to include target drift analysis
            target_column: Name of the target column if analyzing target drift
            
        Returns:
            Path to the saved HTML report
        """
        if self.reference_data is None:
            raise ValueError("Reference data must be set before generating report")
        
        metrics = [DataDriftPreset()]
        
        if include_target_drift and target_column:
            self.column_mapping.target = target_column
            metrics.append(TargetDriftPreset())
        
        report = Report(metrics=metrics)
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"full_drift_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"Full drift report saved to {report_path}")
        
        return str(report_path)
    
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
