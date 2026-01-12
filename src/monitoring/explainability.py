"""
Model Explainability using SHAP.

This module provides SHAP-based model explanations for:
- Feature importance (global explanations)
- Individual prediction explanations (local explanations)
- Feature importance drift tracking over time

Integrates with MLflow for logging SHAP artifacts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for tree-based models.
    
    Provides global and local explanations using SHAP values,
    and can compare feature importance between reference and current data.
    """
    
    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[List[str]] = None,
        reports_dir: Optional[str] = None
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model (sklearn estimator or similar)
            feature_names: List of feature column names
            reports_dir: Directory to save SHAP reports/plots
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )
        
        self.model = model
        self.feature_names = feature_names
        self.reports_dir = Path(reports_dir) if reports_dir else Path("logs/shap_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self._explainer: Optional[shap.Explainer] = None
        self._shap_values: Optional[np.ndarray] = None
        self._reference_importance: Optional[Dict[str, float]] = None
    
    def set_model(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Set or update the model for SHAP explanations.
        
        Args:
            model: Trained model
            feature_names: Optional list of feature names
        """
        self.model = model
        if feature_names:
            self.feature_names = feature_names
        self._explainer = None  # Reset explainer when model changes
        self._shap_values = None
    
    def _get_explainer(self, X: pd.DataFrame) -> shap.Explainer:
        """
        Get or create a SHAP explainer for the model.
        
        Args:
            X: Sample data for explainer initialization
            
        Returns:
            SHAP Explainer object
        """
        if self._explainer is None:
            if self.model is None:
                raise ValueError("Model must be set before creating explainer")
            
            # Use TreeExplainer for tree-based models (faster)
            try:
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer for tree-based model")
            except Exception:
                # Fallback to KernelExplainer for other models
                logger.info("Falling back to KernelExplainer")
                background = shap.sample(X, min(100, len(X)))
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background
                )
        
        return self._explainer
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = 1000
    ) -> np.ndarray:
        """
        Compute SHAP values for a dataset.
        
        Args:
            X: Feature DataFrame
            sample_size: Maximum samples to use (for performance)
            
        Returns:
            Array of SHAP values
        """
        # Sample if dataset is large
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} from {len(X)} rows for SHAP computation")
        else:
            X_sample = X
        
        explainer = self._get_explainer(X_sample)
        
        logger.info("Computing SHAP values...")
        self._shap_values = explainer.shap_values(X_sample)
        
        # For multi-class, shap_values is a list
        if isinstance(self._shap_values, list):
            logger.info(f"Computed SHAP values for {len(self._shap_values)} classes")
        else:
            logger.info(f"Computed SHAP values with shape {self._shap_values.shape}")
        
        return self._shap_values
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate global feature importance from SHAP values.
        
        Args:
            X: Feature DataFrame
            shap_values: Pre-computed SHAP values (optional)
            
        Returns:
            Dictionary of feature name to importance score
        """
        if shap_values is None:
            if self._shap_values is None:
                shap_values = self.compute_shap_values(X)
            else:
                shap_values = self._shap_values
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # List of 2D arrays (one per class) - older format
            stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
            mean_abs_shap = stacked.mean(axis=0)
        elif len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes) - TreeExplainer format
            # Take mean absolute value across samples and classes
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # 2D array: (samples, features) - single output
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dictionary
        feature_names = self.feature_names or X.columns.tolist()
        importance = {
            name: float(score) 
            for name, score in zip(feature_names, mean_abs_shap)
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def set_reference_importance(self, X_reference: pd.DataFrame):
        """
        Set reference feature importance for drift comparison.
        
        Args:
            X_reference: Reference dataset features
        """
        self._reference_importance = self.get_feature_importance(X_reference)
        logger.info(f"Set reference importance for {len(self._reference_importance)} features")
    
    def compute_importance_drift(
        self,
        X_current: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare feature importance between reference and current data.
        
        Args:
            X_current: Current dataset features
            
        Returns:
            Dictionary with drift analysis results
        """
        if self._reference_importance is None:
            raise ValueError("Reference importance not set. Call set_reference_importance first.")
        
        current_importance = self.get_feature_importance(X_current)
        
        # Calculate drift metrics
        drift_results = {
            "reference_importance": self._reference_importance,
            "current_importance": current_importance,
            "feature_drift": {},
            "rank_changes": {},
            "top_changed_features": []
        }
        
        # Get reference rankings
        ref_ranked = list(self._reference_importance.keys())
        cur_ranked = list(current_importance.keys())
        
        for feature in self._reference_importance:
            ref_imp = self._reference_importance.get(feature, 0)
            cur_imp = current_importance.get(feature, 0)
            
            # Absolute and relative change
            abs_change = cur_imp - ref_imp
            rel_change = (abs_change / ref_imp * 100) if ref_imp > 0 else 0
            
            # Rank change
            ref_rank = ref_ranked.index(feature) if feature in ref_ranked else -1
            cur_rank = cur_ranked.index(feature) if feature in cur_ranked else -1
            rank_change = ref_rank - cur_rank  # Positive = moved up in importance
            
            drift_results["feature_drift"][feature] = {
                "reference_importance": ref_imp,
                "current_importance": cur_imp,
                "absolute_change": abs_change,
                "relative_change_pct": rel_change
            }
            drift_results["rank_changes"][feature] = rank_change
        
        # Find top changed features
        sorted_by_change = sorted(
            drift_results["feature_drift"].items(),
            key=lambda x: abs(x[1]["relative_change_pct"]),
            reverse=True
        )
        drift_results["top_changed_features"] = [
            {"feature": f, **v} for f, v in sorted_by_change[:5]
        ]
        
        return drift_results
    
    def generate_summary_plot(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 15,
        class_index: int = 0
    ) -> str:
        """
        Generate and save a SHAP summary plot.
        
        Args:
            X: Feature DataFrame
            shap_values: Pre-computed SHAP values
            max_display: Maximum features to display
            class_index: Which class to plot (for multi-class)
            
        Returns:
            Path to saved plot
        """
        if shap_values is None:
            if self._shap_values is None:
                shap_values = self.compute_shap_values(X)
            else:
                shap_values = self._shap_values
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # List of 2D arrays (one per class)
            shap_values_plot = shap_values[class_index]
        elif len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes) - extract one class
            shap_values_plot = shap_values[:, :, class_index]
        else:
            # 2D array: (samples, features)
            shap_values_plot = shap_values
        
        # Get matching X data
        n_samples = shap_values_plot.shape[0]
        X_plot = X.head(n_samples) if len(X) >= n_samples else X
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            X_plot,
            max_display=max_display,
            show=False
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.reports_dir / f"shap_summary_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {plot_path}")
        return str(plot_path)
    
    def generate_importance_bar_plot(
        self,
        X: pd.DataFrame,
        max_display: int = 15
    ) -> str:
        """
        Generate and save a SHAP feature importance bar plot.
        
        Args:
            X: Feature DataFrame
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        importance = self.get_feature_importance(X)
        
        # Get top features
        top_features = list(importance.items())[:max_display]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('SHAP Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.reports_dir / f"shap_importance_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP importance plot saved to {plot_path}")
        return str(plot_path)
    
    def generate_drift_comparison_plot(
        self,
        X_current: pd.DataFrame,
        max_display: int = 15
    ) -> str:
        """
        Generate a comparison plot of feature importance drift.
        
        Args:
            X_current: Current dataset features
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if self._reference_importance is None:
            raise ValueError("Reference importance not set")
        
        current_importance = self.get_feature_importance(X_current)
        
        # Get top features from both
        all_features = set(list(self._reference_importance.keys())[:max_display])
        all_features.update(list(current_importance.keys())[:max_display])
        all_features = sorted(all_features, key=lambda x: self._reference_importance.get(x, 0), reverse=True)[:max_display]
        
        ref_values = [self._reference_importance.get(f, 0) for f in all_features]
        cur_values = [current_importance.get(f, 0) for f in all_features]
        
        x = np.arange(len(all_features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.barh(x - width/2, ref_values, width, label='Reference', color='steelblue')
        bars2 = ax.barh(x + width/2, cur_values, width, label='Current', color='coral')
        
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance: Reference vs Current')
        ax.set_yticks(x)
        ax.set_yticklabels(all_features)
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.reports_dir / f"shap_drift_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP drift comparison plot saved to {plot_path}")
        return str(plot_path)
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        class_index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X_single: Single row DataFrame with features
            class_index: Which class to explain (for multi-class)
            
        Returns:
            Dictionary with feature contributions
        """
        explainer = self._get_explainer(X_single)
        shap_values = explainer.shap_values(X_single)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # List of 2D arrays (one per class)
            sv = shap_values[class_index][0]
        elif len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes)
            sv = shap_values[0, :, class_index]
        else:
            # 2D array: (samples, features)
            sv = shap_values[0]
        
        feature_names = self.feature_names or X_single.columns.tolist()
        
        contributions = {
            name: float(value)
            for name, value in zip(feature_names, sv)
        }
        
        # Sort by absolute contribution
        contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        # Handle expected_value for different formats
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected = float(explainer.expected_value[class_index])
        else:
            expected = float(explainer.expected_value)
        
        return {
            "feature_contributions": contributions,
            "expected_value": expected,
            "class_index": class_index
        }
    
    def log_to_mlflow(
        self,
        X: pd.DataFrame,
        run_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Log SHAP artifacts to MLflow.
        
        Args:
            X: Feature DataFrame for computing SHAP values
            run_id: Optional MLflow run ID (uses active run if None)
            
        Returns:
            Dictionary of logged artifact paths
        """
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow not available, skipping SHAP logging")
            return {}
        
        logged_artifacts = {}
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Get feature importance
        importance = self.get_feature_importance(X, shap_values)
        
        # Log importance as JSON
        importance_path = self.reports_dir / "shap_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        
        # Generate plots
        summary_plot_path = self.generate_summary_plot(X, shap_values)
        importance_plot_path = self.generate_importance_bar_plot(X)
        
        # Log to MLflow
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(importance_path), "shap")
                mlflow.log_artifact(summary_plot_path, "shap")
                mlflow.log_artifact(importance_plot_path, "shap")
                
                # Log top feature importances as metrics
                for i, (feature, score) in enumerate(list(importance.items())[:10]):
                    mlflow.log_metric(f"shap_importance_{feature}", score)
        else:
            # Use active run
            mlflow.log_artifact(str(importance_path), "shap")
            mlflow.log_artifact(summary_plot_path, "shap")
            mlflow.log_artifact(importance_plot_path, "shap")
            
            for i, (feature, score) in enumerate(list(importance.items())[:10]):
                mlflow.log_metric(f"shap_importance_{feature}", score)
        
        logged_artifacts = {
            "importance_json": str(importance_path),
            "summary_plot": summary_plot_path,
            "importance_plot": importance_plot_path
        }
        
        logger.info(f"Logged SHAP artifacts to MLflow")
        return logged_artifacts


# Global SHAP explainer instance
_shap_explainer: Optional[SHAPExplainer] = None


def get_shap_explainer(
    model: Any = None,
    feature_names: Optional[List[str]] = None,
    reports_dir: Optional[str] = None
) -> SHAPExplainer:
    """
    Get or create a global SHAP explainer instance.
    
    Args:
        model: Optional model to set
        feature_names: Optional feature names
        reports_dir: Optional reports directory
        
    Returns:
        SHAPExplainer instance
    """
    global _shap_explainer
    
    if _shap_explainer is None:
        _shap_explainer = SHAPExplainer(
            model=model,
            feature_names=feature_names,
            reports_dir=reports_dir
        )
    elif model is not None:
        _shap_explainer.set_model(model, feature_names)
    
    return _shap_explainer
