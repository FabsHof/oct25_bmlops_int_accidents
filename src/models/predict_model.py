"""
Model Prediction Module for Accident Severity Prediction

This module implements the inference pipeline for predicting accident severity
using a trained Random Forest Classifier. It includes:
- Model loading from saved artifacts
- Input validation and feature alignment
- Prediction with probability scores
- Result formatting with human-readable labels
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import joblib
import numpy as np

from src.utils import logging


class AccidentSeverityPredictor:
    """
    Predictor class for accident severity inference.
    
    This class handles loading the trained model and making predictions
    on new input data.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize predictor by loading model and metadata.
        
        Args:
            model_dir: Path to directory containing model artifacts
                      (model.joblib, feature_names.json, config.json)
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If model artifacts are invalid
        """
        self.model_dir = Path(model_dir)
        
        # Validate directory exists
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load model artifacts
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        self.config = self._load_config()
        
        # Extract class labels from config
        self.class_labels = self.config.get('metrics_config', {}).get('class_labels', {})
        
        logging.info(f"Loaded model from {model_dir}")
        logging.info(f"Expected features: {len(self.feature_names)}")
        logging.info(f"Model classes: {self.model.n_classes_}")
    
    def _load_model(self) -> Any:
        """Load the trained model from joblib file."""
        model_path = self.model_dir / 'model.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            logging.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def _load_feature_names(self) -> List[str]:
        """Load feature names from JSON file."""
        feature_path = self.model_dir / 'feature_names.json'
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature names file not found: {feature_path}")
        
        try:
            with open(feature_path, 'r') as f:
                features = json.load(f)
            logging.info(f"Successfully loaded {len(features)} feature names")
            return features
        except Exception as e:
            raise ValueError(f"Failed to load feature names: {e}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        config_path = self.model_dir / 'config.json'
        if not config_path.exists():
            logging.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info("Successfully loaded model configuration")
            return config
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
            return {}
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate that input data contains all required features.
        
        Args:
            input_data: Dictionary with feature names as keys
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = set(self.feature_names) - set(input_data.keys())
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {sorted(missing_features)}. "
                f"Expected features: {self.feature_names}"
            )
        
        # Check for extra features (warning only)
        extra_features = set(input_data.keys()) - set(self.feature_names)
        if extra_features:
            logging.warning(f"Extra features will be ignored: {sorted(extra_features)}")
    
    def _prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features in the correct order for prediction.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            DataFrame with features in correct order
        """
        # Extract only the required features in the correct order
        feature_values = [input_data[feature] for feature in self.feature_names]
        
        # Create DataFrame with proper column names
        df = pd.DataFrame([feature_values], columns=self.feature_names)
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Dictionary with feature names as keys and values
                       Example: {'year': 2023, 'month': 6, 'hour': 14, ...}
        
        Returns:
            Dictionary containing:
                - prediction: Predicted severity class (integer)
                - prediction_label: Human-readable label
                - probabilities: Dictionary of class probabilities
                - confidence: Confidence score (max probability)
        
        Raises:
            ValueError: If input validation fails
        """
        logging.info("Starting prediction...")
        
        # Validate input
        self._validate_input(input_data)
        
        # Prepare features
        X = self._prepare_features(input_data)
        logging.info(f"Prepared features: {X.shape}")
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Format results
        result = {
            'prediction': int(prediction),
            'prediction_label': self._get_class_label(prediction),
            'probabilities': self._format_probabilities(probabilities),
            'confidence': float(np.max(probabilities))
        }
        
        logging.info(f"Prediction: {result['prediction_label']} (confidence: {result['confidence']:.4f})")
        
        return result
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple input samples.
        
        Args:
            input_data_list: List of dictionaries with feature values
        
        Returns:
            List of prediction result dictionaries
        """
        logging.info(f"Starting batch prediction for {len(input_data_list)} samples...")
        
        results = []
        for idx, input_data in enumerate(input_data_list):
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                logging.error(f"Error predicting sample {idx}: {e}")
                results.append({
                    'error': str(e),
                    'prediction': None,
                    'prediction_label': None,
                    'probabilities': None,
                    'confidence': None
                })
        
        logging.info(f"Completed batch prediction: {len(results)} results")
        return results
    
    def _get_class_label(self, severity: int) -> str:
        """
        Get human-readable label for severity class.
        
        Args:
            severity: Severity class number
            
        Returns:
            Human-readable label
        """
        # Convert to string key for lookup
        return self.class_labels.get(str(severity), f"Severity {severity}")
    
    def _format_probabilities(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Format probability array as dictionary with labels.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Dictionary mapping class labels to probabilities
        """
        result = {}
        for idx, prob in enumerate(probabilities):
            # Classes are 1-indexed in this dataset
            severity = idx + 1
            label = self._get_class_label(severity)
            result[label] = float(prob)
        
        return result


def predict_model(model_dir: str, input_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Convenience function for making predictions.
    
    Args:
        model_dir: Path to directory containing model artifacts
        input_data: Either a dictionary with feature values or a JSON string
    
    Returns:
        Dictionary containing prediction results
    """
    # Parse JSON string if needed
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
    
    # Create predictor and make prediction
    predictor = AccidentSeverityPredictor(model_dir)
    result = predictor.predict(input_data)
    
    return result


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Use the trained model to make predictions on the severity of a road accident.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the directory containing model artifacts (model.joblib, feature_names.json, config.json).'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        required=True,
        help='Input data for prediction in JSON format. Example: \'{"year": 2023, "month": 6, ...}\''
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save prediction results as JSON file.'
    )
    
    args = parser.parse_args()
    
    try:
        # Make prediction
        result = predict_model(args.model_dir, args.input_data)
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Predicted Severity: {result['prediction_label']} (class {result['prediction']})")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print("="*60 + "\n")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logging.info(f"Results saved to {output_path}")
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()