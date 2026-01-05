"""
Custom MLflow Metrics for Accident Severity Prediction

This module defines custom metrics for evaluating multiclass classification models
in the accident severity prediction task. These metrics integrate with MLflow's
evaluation framework and provide comprehensive performance assessment.
"""

from typing import Dict, Any, List, Union
import numpy as np
from mlflow.models import make_metric
from mlflow.metrics.base import MetricValue
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Mapping of severity levels to descriptive names
CLASS_LABELS = {
    1: 'Unscathed',
    2: 'Light injury',
    3: 'Hospitalized wounded',
    4: 'Killed'
}


def _weighted_f1_score_metric_fn(predictions: Union[List, np.ndarray], 
                                 targets: Union[List, np.ndarray], 
                                 metrics: Dict[str, Any]) -> MetricValue:
    """Calculate weighted F1 score for multiclass classification."""
    score = f1_score(targets, predictions, average='weighted', zero_division=0)
    return MetricValue(aggregate_results={'weighted_f1_score': score})


def _weighted_precision_metric_fn(predictions: Union[List, np.ndarray], 
                                  targets: Union[List, np.ndarray], 
                                  metrics: Dict[str, Any]) -> MetricValue:
    """Calculate weighted precision for multiclass classification."""
    score = precision_score(targets, predictions, average='weighted', zero_division=0)
    return MetricValue(aggregate_results={'weighted_precision': score})


def _weighted_recall_metric_fn(predictions: Union[List, np.ndarray], 
                               targets: Union[List, np.ndarray], 
                               metrics: Dict[str, Any]) -> MetricValue:
    """Calculate weighted recall for multiclass classification."""
    score = recall_score(targets, predictions, average='weighted', zero_division=0)
    return MetricValue(aggregate_results={'weighted_recall': score})


def _roc_auc_ovr_metric_fn(predictions: Union[List, np.ndarray], 
                           targets: Union[List, np.ndarray], 
                           metrics: Dict[str, Any]) -> MetricValue:
    """Calculate ROC-AUC using One-vs-Rest approach for multiclass classification."""
    try:
        # Get prediction probabilities from the model
        model_predictions = metrics.get('predictions', [])
        if hasattr(model_predictions[0], 'values') and len(model_predictions[0].values) > 1:
            # Extract probabilities for multiclass
            proba_matrix = [pred.values for pred in model_predictions]
            score = roc_auc_score(targets, proba_matrix, multi_class='ovr', average='weighted')
        else:
            # Fallback if probabilities not available
            score = None
    except Exception as e:
        score = None
    
    result = {'roc_auc_ovr': score} if score is not None else {'roc_auc_ovr': 0.0}
    return MetricValue(aggregate_results=result)


def _per_class_f1_metric_fn(predictions: Union[List, np.ndarray], 
                            targets: Union[List, np.ndarray], 
                            metrics: Dict[str, Any]) -> MetricValue:
    """Calculate F1 score for each class individually."""
    # Get unique classes from targets
    unique_classes = sorted(set(targets))
    results = {}
    
    for class_label in unique_classes:
        class_name = CLASS_LABELS.get(class_label, f'Class_{class_label}')
        # Calculate F1 for this specific class vs all others
        f1 = f1_score(targets, predictions, labels=[class_label], average=None, zero_division=0)
        if len(f1) > 0:
            results[f'class_{class_name}_f1'] = float(f1[0])
        else:
            results[f'class_{class_name}_f1'] = 0.0
    
    return MetricValue(aggregate_results=results)


# Create metric objects
weighted_f1_score_metric = make_metric(
    name='weighted_f1_score',
    eval_fn=_weighted_f1_score_metric_fn,
    greater_is_better=True,
    metric_details='Weighted F1 Score for multiclass classification'
)

weighted_precision_metric = make_metric(
    name='weighted_precision',
    eval_fn=_weighted_precision_metric_fn,
    greater_is_better=True,
    metric_details='Weighted Precision for multiclass classification'
)

weighted_recall_metric = make_metric(
    name='weighted_recall',
    eval_fn=_weighted_recall_metric_fn,
    greater_is_better=True,
    metric_details='Weighted Recall for multiclass classification'
)

roc_auc_ovr_metric = make_metric(
    name='roc_auc_ovr',
    eval_fn=_roc_auc_ovr_metric_fn,
    greater_is_better=True,
    metric_details='ROC-AUC using One-vs-Rest approach for multiclass classification'
)

per_class_f1_metric = make_metric(
    name='per_class_f1',
    eval_fn=_per_class_f1_metric_fn,
    greater_is_better=True,
    metric_details='F1 Score for each individual class'
)