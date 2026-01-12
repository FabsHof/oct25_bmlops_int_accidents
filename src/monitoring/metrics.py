"""
Prometheus metrics collector for ML model monitoring.

This module provides Prometheus metrics for tracking:
- Prediction requests and latency
- Prediction distribution by severity class
- Model confidence scores
- Data drift indicators
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import time
from typing import Dict, Optional
from functools import wraps


class MetricsCollector:
    """
    Centralized metrics collector for Prometheus monitoring.
    
    Provides metrics for API performance, model predictions, and data quality.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize the metrics collector.
        
        Args:
            registry: Optional custom Prometheus registry. Uses default if not provided.
        """
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of prediction requests',
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction request latency in seconds',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.prediction_errors = Counter(
            'prediction_errors_total',
            'Total number of prediction errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Prediction distribution metrics
        self.predictions_by_severity = Counter(
            'predictions_by_severity',
            'Predictions count by severity class',
            ['severity'],
            registry=self.registry
        )
        
        # Model confidence metrics
        self.prediction_confidence = Histogram(
            'prediction_confidence',
            'Distribution of prediction confidence scores',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Data drift metrics (updated by drift detector)
        self.data_drift_score = Gauge(
            'data_drift_score',
            'Overall data drift score (0-1, higher means more drift)',
            registry=self.registry
        )
        
        self.feature_drift = Gauge(
            'feature_drift',
            'Per-feature drift score',
            ['feature'],
            registry=self.registry
        )
        
        self.drift_detected = Gauge(
            'drift_detected',
            'Binary indicator if significant drift is detected (1=drift, 0=no drift)',
            registry=self.registry
        )
        
        # Model performance metrics (can be updated during evaluation)
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy on evaluation data',
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'model_f1_score',
            'Current model F1 score',
            ['class_label'],
            registry=self.registry
        )
    
    def record_prediction(
        self,
        severity_class: int,
        severity_label: str,
        confidence: float,
        latency: float
    ):
        """
        Record metrics for a single prediction.
        
        Args:
            severity_class: Predicted severity class (1-4)
            severity_label: Human-readable severity label
            confidence: Model confidence score
            latency: Prediction latency in seconds
        """
        self.predictions_total.inc()
        self.prediction_latency.observe(latency)
        self.predictions_by_severity.labels(severity=severity_label).inc()
        self.prediction_confidence.observe(confidence)
    
    def record_error(self, error_type: str):
        """
        Record a prediction error.
        
        Args:
            error_type: Type of error (e.g., 'validation', 'model', 'internal')
        """
        self.prediction_errors.labels(error_type=error_type).inc()
    
    def update_drift_metrics(
        self,
        overall_drift: float,
        feature_drifts: Dict[str, float],
        is_drift_detected: bool
    ):
        """
        Update data drift metrics.
        
        Args:
            overall_drift: Overall drift score (0-1)
            feature_drifts: Dict of feature name to drift score
            is_drift_detected: Whether significant drift was detected
        """
        self.data_drift_score.set(overall_drift)
        self.drift_detected.set(1 if is_drift_detected else 0)
        
        for feature, drift_value in feature_drifts.items():
            self.feature_drift.labels(feature=feature).set(drift_value)
    
    def update_model_metrics(self, accuracy: float, f1_scores: Dict[str, float]):
        """
        Update model performance metrics.
        
        Args:
            accuracy: Model accuracy
            f1_scores: Dict of class label to F1 score
        """
        self.model_accuracy.set(accuracy)
        for label, score in f1_scores.items():
            self.model_f1_score.labels(class_label=label).set(score)
    
    def get_metrics(self) -> bytes:
        """
        Generate Prometheus metrics output.
        
        Returns:
            Prometheus metrics in exposition format
        """
        return generate_latest(self.registry)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create the global metrics collector.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_prediction(func):
    """
    Decorator to track prediction metrics.
    
    Wraps a prediction function to automatically record latency and errors.
    The wrapped function should return a dict with 'prediction_label' and 'confidence' keys.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            
            # Extract metrics from result if available
            if isinstance(result, dict):
                collector.record_prediction(
                    severity_class=result.get('prediction', 0),
                    severity_label=result.get('prediction_label', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    latency=latency
                )
            
            return result
        except Exception as e:
            collector.record_error(type(e).__name__)
            raise
    
    return wrapper
