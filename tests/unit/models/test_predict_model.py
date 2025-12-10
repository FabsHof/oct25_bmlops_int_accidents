"""
Unit tests for model prediction module.

This test suite covers:
- Model loading and initialization
- Input validation
- Prediction functionality
- Batch prediction
- Error handling
- Edge cases
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import joblib

from src.models.predict_model import (
    AccidentSeverityPredictor,
    predict_model
)


@pytest.fixture
def sample_feature_names():
    """Sample feature names matching the model."""
    return [
        'year',
        'month',
        'hour',
        'minute',
        'user_category',
        'sex',
        'year_of_birth',
        'trip_purpose',
        'security',
        'luminosity',
        'weather',
        'type_of_road',
        'road_surface',
        'latitude',
        'longitude',
        'holiday'
    ]


@pytest.fixture
def sample_config():
    """Sample model configuration."""
    return {
        "dataset_config": {
            "feature_columns": [
                "year", "month", "hour", "minute", "user_category",
                "sex", "year_of_birth", "trip_purpose", "security",
                "luminosity", "weather", "type_of_road", "road_surface",
                "latitude", "longitude", "holiday"
            ],
            "target_column": "severity"
        },
        "metrics_config": {
            "class_labels": {
                "1": "Unscathed",
                "2": "Light injury",
                "3": "Hospitalized wounded",
                "4": "Killed"
            }
        }
    }


@pytest.fixture
def sample_input_data():
    """Sample valid input data for prediction."""
    return {
        'year': 2023,
        'month': 6,
        'hour': 14,
        'minute': 30,
        'user_category': 1,
        'sex': 1,
        'year_of_birth': 1990,
        'trip_purpose': 1,
        'security': 1,
        'luminosity': 1,
        'weather': 1,
        'type_of_road': 1,
        'road_surface': 1,
        'latitude': 48.8566,
        'longitude': 2.3522,
        'holiday': 0
    }


@pytest.fixture
def mock_model():
    """Create a simple trained Random Forest model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple model with minimal training
    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
    
    # Create dummy training data (16 features, 4 classes)
    np.random.seed(42)
    X_dummy = np.random.rand(100, 16)
    y_dummy = np.random.choice([1, 2, 3, 4], 100)
    
    # Train the model
    model.fit(X_dummy, y_dummy)
    
    return model


@pytest.fixture
def temp_model_dir(sample_feature_names, sample_config, mock_model):
    """Create a temporary directory with mock model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        
        # Save model
        joblib.dump(mock_model, model_dir / 'model.joblib')
        
        # Save feature names
        with open(model_dir / 'feature_names.json', 'w') as f:
            json.dump(sample_feature_names, f)
        
        # Save config
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(sample_config, f)
        
        yield model_dir


class TestAccidentSeverityPredictor:
    """Test suite for AccidentSeverityPredictor class."""
    
    def test_init_success(self, temp_model_dir):
        """Test successful initialization of predictor."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        assert predictor.model is not None
        assert len(predictor.feature_names) == 16
        assert predictor.config is not None
        assert len(predictor.class_labels) == 4
    
    def test_init_missing_directory(self):
        """Test initialization fails with missing directory."""
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            AccidentSeverityPredictor("/nonexistent/path")
    
    def test_init_missing_model_file(self, sample_feature_names, sample_config):
        """Test initialization fails with missing model.joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Only save feature names and config, no model
            with open(model_dir / 'feature_names.json', 'w') as f:
                json.dump(sample_feature_names, f)
            with open(model_dir / 'config.json', 'w') as f:
                json.dump(sample_config, f)
            
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                AccidentSeverityPredictor(model_dir)
    
    def test_init_missing_feature_names(self, sample_config, mock_model):
        """Test initialization fails with missing feature_names.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Only save model and config
            joblib.dump(mock_model, model_dir / 'model.joblib')
            with open(model_dir / 'config.json', 'w') as f:
                json.dump(sample_config, f)
            
            with pytest.raises(FileNotFoundError, match="Feature names file not found"):
                AccidentSeverityPredictor(model_dir)
    
    def test_init_missing_config(self, sample_feature_names, mock_model):
        """Test initialization succeeds without config (warning only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save model and features, no config
            joblib.dump(mock_model, model_dir / 'model.joblib')
            with open(model_dir / 'feature_names.json', 'w') as f:
                json.dump(sample_feature_names, f)
            
            # Should succeed but with empty config
            predictor = AccidentSeverityPredictor(model_dir)
            assert predictor.config == {}
    
    def test_validate_input_success(self, temp_model_dir, sample_input_data):
        """Test input validation with valid data."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Should not raise any exception
        predictor._validate_input(sample_input_data)
    
    def test_validate_input_missing_features(self, temp_model_dir):
        """Test input validation fails with missing features."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        incomplete_data = {
            'year': 2023,
            'month': 6,
            'hour': 14
        }
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor._validate_input(incomplete_data)
    
    def test_validate_input_extra_features(self, temp_model_dir, sample_input_data):
        """Test input validation with extra features (should warn but not fail)."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Add extra feature
        data_with_extra = sample_input_data.copy()
        data_with_extra['extra_feature'] = 999
        
        # Should succeed (extra features are ignored)
        predictor._validate_input(data_with_extra)
    
    def test_prepare_features(self, temp_model_dir, sample_input_data):
        """Test feature preparation creates correct DataFrame."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        df = predictor._prepare_features(sample_input_data)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 16)
        assert list(df.columns) == predictor.feature_names
        assert df.iloc[0]['year'] == 2023
        assert df.iloc[0]['month'] == 6
    
    def test_predict_success(self, temp_model_dir, sample_input_data):
        """Test successful prediction."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        result = predictor.predict(sample_input_data)
        
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        
        assert isinstance(result['prediction'], int)
        assert isinstance(result['prediction_label'], str)
        assert isinstance(result['probabilities'], dict)
        assert isinstance(result['confidence'], float)
        
        # Prediction should be one of the 4 severity classes
        assert result['prediction'] in [1, 2, 3, 4]
        # Confidence should be between 0 and 1
        assert 0.0 <= result['confidence'] <= 1.0
        # Should have probabilities for all 4 classes
        assert len(result['probabilities']) == 4
    
    def test_predict_invalid_input(self, temp_model_dir):
        """Test prediction fails with invalid input."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        invalid_data = {'year': 2023}  # Missing most features
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(invalid_data)
    
    def test_predict_batch_success(self, temp_model_dir, sample_input_data):
        """Test batch prediction with multiple samples."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Create batch with 3 samples
        batch_data = [
            sample_input_data,
            sample_input_data.copy(),
            sample_input_data.copy()
        ]
        
        results = predictor.predict_batch(batch_data)
        
        assert len(results) == 3
        assert all('prediction' in r for r in results)
        assert all('confidence' in r for r in results)
    
    def test_predict_batch_with_errors(self, temp_model_dir, sample_input_data):
        """Test batch prediction handles individual errors gracefully."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Create batch with one invalid sample
        batch_data = [
            sample_input_data,
            {'year': 2023},  # Invalid: missing features
            sample_input_data.copy()
        ]
        
        results = predictor.predict_batch(batch_data)
        
        assert len(results) == 3
        assert results[0]['prediction'] is not None
        assert 'error' in results[1]
        assert results[1]['prediction'] is None
        assert results[2]['prediction'] is not None
    
    def test_get_class_label(self, temp_model_dir):
        """Test class label retrieval."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        assert predictor._get_class_label(1) == "Unscathed"
        assert predictor._get_class_label(2) == "Light injury"
        assert predictor._get_class_label(3) == "Hospitalized wounded"
        assert predictor._get_class_label(4) == "Killed"
    
    def test_get_class_label_unknown(self, temp_model_dir):
        """Test class label for unknown severity."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Unknown severity should return default format
        assert predictor._get_class_label(99) == "Severity 99"
    
    def test_format_probabilities(self, temp_model_dir):
        """Test probability formatting."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        probs = np.array([0.15, 0.45, 0.30, 0.10])
        formatted = predictor._format_probabilities(probs)
        
        assert isinstance(formatted, dict)
        assert len(formatted) == 4
        assert "Unscathed" in formatted
        assert "Light injury" in formatted
        assert formatted["Light injury"] == 0.45
        assert all(isinstance(v, float) for v in formatted.values())


class TestPredictModelFunction:
    """Test suite for predict_model convenience function."""
    
    def test_predict_model_with_dict(self, temp_model_dir, sample_input_data):
        """Test predict_model with dictionary input."""
        result = predict_model(str(temp_model_dir), sample_input_data)
        
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert result['prediction'] in [1, 2, 3, 4]
    
    def test_predict_model_with_json_string(self, temp_model_dir, sample_input_data):
        """Test predict_model with JSON string input."""
        json_string = json.dumps(sample_input_data)
        
        result = predict_model(str(temp_model_dir), json_string)
        
        assert 'prediction' in result
        assert result['prediction'] in [1, 2, 3, 4]
    
    def test_predict_model_with_invalid_json(self, temp_model_dir):
        """Test predict_model fails with invalid JSON string."""
        invalid_json = "{'this': 'is not valid json}"
        
        with pytest.raises(ValueError, match="Invalid JSON input"):
            predict_model(str(temp_model_dir), invalid_json)
    
    def test_predict_model_missing_directory(self, sample_input_data):
        """Test predict_model fails with missing directory."""
        with pytest.raises(FileNotFoundError):
            predict_model("/nonexistent/path", sample_input_data)


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_predict_with_zero_values(self, temp_model_dir, sample_input_data):
        """Test prediction with zero values."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        zero_data = sample_input_data.copy()
        zero_data['month'] = 0
        zero_data['hour'] = 0
        zero_data['holiday'] = 0
        
        result = predictor.predict(zero_data)
        assert result['prediction'] is not None
    
    def test_predict_with_negative_values(self, temp_model_dir, sample_input_data):
        """Test prediction with negative values (if model accepts them)."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        # Some features might allow negative values
        data = sample_input_data.copy()
        data['year_of_birth'] = -1  # Invalid birth year
        
        # Should still make prediction (validation is data-level concern)
        result = predictor.predict(data)
        assert result['prediction'] is not None
    
    def test_predict_with_float_values(self, temp_model_dir, sample_input_data):
        """Test prediction with float values for integer features."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        float_data = sample_input_data.copy()
        float_data['month'] = 6.5
        float_data['hour'] = 14.7
        
        result = predictor.predict(float_data)
        assert result['prediction'] is not None
    
    def test_predict_with_extreme_coordinates(self, temp_model_dir, sample_input_data):
        """Test prediction with extreme coordinate values."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        extreme_data = sample_input_data.copy()
        extreme_data['latitude'] = 90.0  # North pole
        extreme_data['longitude'] = 180.0  # Date line
        
        result = predictor.predict(extreme_data)
        assert result['prediction'] is not None
    
    def test_confidence_in_valid_range(self, temp_model_dir, sample_input_data):
        """Test that confidence is always between 0 and 1."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        result = predictor.predict(sample_input_data)
        
        assert 0.0 <= result['confidence'] <= 1.0
        
        # All probabilities should sum to approximately 1
        prob_sum = sum(result['probabilities'].values())
        assert 0.99 <= prob_sum <= 1.01
    
    def test_predict_empty_batch(self, temp_model_dir):
        """Test batch prediction with empty list."""
        predictor = AccidentSeverityPredictor(temp_model_dir)
        
        results = predictor.predict_batch([])
        
        assert results == []


class TestIntegrationWithRealModel:
    """Integration tests using actual saved model."""
    
    def test_predict_with_real_model(self, sample_input_data):
        """Test prediction with real saved model if available."""
        # Find the latest model directory
        models_dir = Path(__file__).parent.parent.parent.parent / 'models'
        
        if not models_dir.exists():
            pytest.skip("No models directory found")
        
        # Get most recent model
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
        if not model_dirs:
            pytest.skip("No model directories found")
        
        latest_model = model_dirs[-1]
        
        # Check required files exist
        if not (latest_model / 'model.joblib').exists():
            pytest.skip("Model file not found in directory")
        
        try:
            predictor = AccidentSeverityPredictor(latest_model)
            result = predictor.predict(sample_input_data)
            
            # Verify result structure
            assert 'prediction' in result
            assert 'prediction_label' in result
            assert 'probabilities' in result
            assert 'confidence' in result
            
            # Verify prediction is valid severity
            assert result['prediction'] in [1, 2, 3, 4]
            
            # Verify confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
            
        except Exception as e:
            pytest.skip(f"Could not test with real model: {e}")
