"""
Monitoring module for ML model observability.

This module provides:
- Prometheus metrics for API monitoring
- Evidently integration for data drift detection
"""

from src.monitoring.metrics import (
    MetricsCollector,
    track_prediction,
    get_metrics_collector,
)
from src.monitoring.drift import (
    DriftDetector,
    get_drift_detector,
)

__all__ = [
    "MetricsCollector",
    "track_prediction",
    "get_metrics_collector",
    "DriftDetector",
    "get_drift_detector",
]
