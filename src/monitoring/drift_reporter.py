"""
Drift Reporter Utility

Handles computation of data drift scores and submission to Prometheus metrics API.
Reuses drift analysis results from Evidently reports and submits them for monitoring.
"""

import os
from typing import Dict, Tuple, Optional
from datetime import datetime
import requests
import pandas as pd
import json
from pathlib import Path


def extract_drift_from_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    logger=None
) -> Tuple[float, Dict[str, float]]:
    """
    Extract drift scores by analyzing reference vs current data using Evidently.
    
    Args:
        reference_data: Training/reference dataset (DataFrame)
        current_data: Current/test dataset (DataFrame)
        logger: Optional logger instance for messages
    
    Returns:
        Tuple of (overall_drift_score, feature_drift_scores_dict)
        - overall_drift_score: Share of drifted columns (0-1)
        - feature_drift_scores: Dict with drift score per feature
    """
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    
    try:
        # Define numerical and categorical features
        numerical_features = [
            'year', 'month', 'hour', 'minute', 'year_of_birth',
            'latitude', 'longitude'
        ]
        categorical_features = [
            'user_category', 'sex', 'trip_purpose', 'security',
            'luminosity', 'weather', 'type_of_road', 'road_surface', 'holiday'
        ]
        
        # Filter to columns that exist in both datasets
        available_cols = [col for col in reference_data.columns if col in current_data.columns]
        
        # Separate into numerical and categorical based on what exists
        available_num = [col for col in numerical_features if col in available_cols]
        available_cat = [col for col in categorical_features if col in available_cols]
        
        # Create Evidently DataDefinition
        data_definition = DataDefinition(
            numerical_columns=available_num,
            categorical_columns=available_cat
        )
        
        # Create datasets
        ref_dataset = Dataset.from_pandas(
            reference_data[available_cols],
            data_definition=data_definition
        )
        cur_dataset = Dataset.from_pandas(
            current_data[available_cols],
            data_definition=data_definition
        )
        
        # Run Evidently drift detection
        report = Report([DataDriftPreset()])
        snapshot = report.run(cur_dataset, ref_dataset)
        
        # Extract results
        result = snapshot.dict()
        overall_drift_share = 0.0
        feature_drift_scores = {}
        
        # Parse drift metrics from Evidently result structure
        metrics = result.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            if "share_of_drifted_columns" in metric_result:
                overall_drift_share = metric_result.get("share_of_drifted_columns", 0.0)
            if "drift_by_columns" in metric_result:
                for feature, info in metric_result.get("drift_by_columns", {}).items():
                    if isinstance(info, dict):
                        feature_drift_scores[feature] = info.get("drift_score", 0.0)
        
        if logger:
            logger.info(f'Extracted drift scores from Evidently: overall={overall_drift_share:.4f}')
        
        return overall_drift_share, feature_drift_scores
        
    except Exception as e:
        if logger:
            logger.warning(f'Could not extract drift from Evidently report: {e}. Using zero scores.')
        return 0.0, {}


def submit_drift_metrics_to_api(
    overall_drift_score: float,
    feature_drift_scores: Dict[str, float],
    logger=None
) -> bool:
    """
    Submit computed drift metrics to the API /metrics/drift endpoint.
    
    Args:
        overall_drift_score: Overall drift score (0-1)
        feature_drift_scores: Dict of per-feature drift scores
        logger: Optional logger instance for messages
    
    Returns:
        True if submission successful, False otherwise
    """
    api_host = os.getenv('API_HOST', 'http://api:8000')
    
    try:
        drift_metrics_payload = {
            'overall_drift_score': overall_drift_score,
            'feature_drift_scores': feature_drift_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        # Submit metrics to the API
        response = requests.post(
            f'{api_host}/metrics/drift',
            json=drift_metrics_payload,
            headers={'Authorization': 'Bearer airflow-drift-reporter'},
            timeout=10
        )
        
        if response.status_code == 200:
            if logger:
                logger.info(f'Drift metrics submitted to Prometheus: overall={overall_drift_score:.4f}')
            return True
        else:
            if logger:
                logger.warning(f'Failed to submit drift metrics: {response.status_code}')
            return False
            
    except Exception as e:
        if logger:
            logger.warning(f'Could not submit drift metrics to API: {e}. Drift will be unavailable in Grafana.')
        return False


def compute_and_submit_drift(
    reference_data: Optional[pd.DataFrame] = None,
    current_data: Optional[pd.DataFrame] = None,
    drift_results: Optional[Dict] = None,
    logger=None
) -> Dict:
    """
    Extract drift scores and submit to API.
    
    Can be called in two ways:
    1) With drift_results dict (pre-computed from DriftDetector) - preferred for reusing existing analysis
    2) With reference_data and current_data - will run Evidently analysis
    
    Args:
        reference_data: Training/reference dataset (optional if drift_results provided)
        current_data: Current/test dataset (optional if drift_results provided)
        drift_results: Pre-computed drift analysis results dict with keys:
            - 'overall_drift_score' or 'share_of_drifted_columns': float
            - 'feature_drift_scores' or 'drift_by_columns': dict
        logger: Optional logger instance
    
    Returns:
        Dict with:
        - 'overall_drift_score': float
        - 'feature_drift_scores': dict
        - 'submitted': bool
        - 'timestamp': str
    """
    # If drift_results provided, use those (preferred)
    if drift_results is not None:
        overall_score = drift_results.get(
            'overall_drift_score',
            drift_results.get('share_of_drifted_columns', 0.0)
        )
        feature_scores = drift_results.get(
            'feature_drift_scores',
            drift_results.get('drift_by_columns', {})
        )
        if logger:
            logger.info('Using pre-computed drift results from DriftDetector')
    # Otherwise, compute from data
    elif reference_data is not None and current_data is not None:
        overall_score, feature_scores = extract_drift_from_report(
            reference_data, current_data, logger
        )
    else:
        if logger:
            logger.warning('No drift data provided (either drift_results or reference/current data required)')
        return {
            'overall_drift_score': 0.0,
            'feature_drift_scores': {},
            'submitted': False,
            'timestamp': datetime.now().isoformat()
        }
    
    # Submit to API
    submitted = submit_drift_metrics_to_api(
        overall_score, feature_scores, logger
    )
    
    return {
        'overall_drift_score': overall_score,
        'feature_drift_scores': feature_scores,
        'submitted': submitted,
        'timestamp': datetime.now().isoformat()
    }
