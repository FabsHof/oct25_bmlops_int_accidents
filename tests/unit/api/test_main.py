"""
Unit tests for API main endpoints.

This test suite covers:
- API key authentication
- Prediction endpoint functionality
- Model info endpoint
- Error handling
- Request/response validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import os

from src.api.main import app, get_predictor, _predictor
from src.models.predict_model import AccidentSeverityPredictor


# Test client
client = TestClient(app)

# Sample test data
VALID_API_KEY = "test_api_key"
INVALID_API_KEY = "wrong_key"

SAMPLE_PREDICTION_REQUEST = {
    "year": 2023,
    "month": 6,
    "hour": 14,
    "minute": 30,
    "user_category": 1,
    "sex": 1,
    "year_of_birth": 1990,
    "trip_purpose": 1,
    "security": 1,
    "luminosity": 1,
    "weather": 1,
    "type_of_road": 1,
    "road_surface": 1,
    "latitude": 48.8566,
    "longitude": 2.3522,
    "holiday": 0
}

SAMPLE_PREDICTION_RESPONSE = {
    "prediction": 3,
    "prediction_label": "Hospitalized wounded",
    "probabilities": {
        "Unscathed": 0.1,
        "Light injury": 0.2,
        "Hospitalized wounded": 0.65,
        "Killed": 0.05
    },
    "confidence": 0.65
}


@pytest.fixture(autouse=True)
def setup_api_key():
    """Set up API key for all tests."""
    with patch.dict(os.environ, {"API_KEY": VALID_API_KEY}):
        yield


@pytest.fixture(autouse=True)
def reset_predictor():
    """Reset global predictor before each test."""
    import src.api.main
    src.api.main._predictor = None
    yield
    src.api.main._predictor = None


class TestAuthentication:
    """Test API key authentication."""
    
    def test_health_check_with_valid_api_key(self):
        """Test health check endpoint with valid API key."""
        response = client.get(f"/health?api_key={VALID_API_KEY}")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_health_check_without_api_key(self):
        """Test health check endpoint without API key."""
        response = client.get("/health")
        assert response.status_code == 401  # Unauthorized (missing API key)
    
    def test_health_check_with_invalid_api_key(self):
        """Test health check endpoint with invalid API key."""
        response = client.get(f"/health?api_key={INVALID_API_KEY}")
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_predict_without_api_key(self):
        """Test prediction endpoint without API key."""
        response = client.post("/predict", json=SAMPLE_PREDICTION_REQUEST)
        assert response.status_code == 401  # Unauthorized (missing API key)
    
    def test_predict_with_invalid_api_key(self):
        """Test prediction endpoint with invalid API key."""
        response = client.post(
            f"/predict?api_key={INVALID_API_KEY}",
            json=SAMPLE_PREDICTION_REQUEST
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestPredictionEndpoint:
    """Test prediction endpoint functionality."""
    
    @patch('src.api.main.setup_mlflow_tracking')
    @patch('src.api.main.mlflow.models.get_model_info')
    @patch('src.api.main.AccidentSeverityPredictor.from_mlflow_model')
    def test_predict_with_valid_request(self, mock_from_mlflow, mock_get_model_info, mock_setup_tracking):
        """Test successful prediction with valid request."""
        mock_setup_tracking.return_value = None
        mock_get_model_info.return_value = Mock(version=5)

        mock_model_dir = Mock()
        mock_model_dir.name = "random_forest_model_v5"

        mock_predictor = Mock()
        mock_predictor.model_dir = mock_model_dir
        mock_predictor.predict.return_value = SAMPLE_PREDICTION_RESPONSE
        mock_from_mlflow.return_value = mock_predictor
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=SAMPLE_PREDICTION_REQUEST
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 3
        assert data["prediction_label"] == "Hospitalized wounded"
        assert data["confidence"] == 0.65
        assert "probabilities" in data
        assert data["model_version"] == "random_forest_model_v5"
    
    @patch('src.api.main.setup_mlflow_tracking')
    @patch('src.api.main.mlflow.models.get_model_info')
    def test_predict_when_no_model_available(self, mock_get_model_info, mock_setup_tracking):
        """Test prediction when no model is available."""
        mock_setup_tracking.return_value = None
        mock_get_model_info.side_effect = Exception("no model")
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=SAMPLE_PREDICTION_REQUEST
        )
        
        assert response.status_code == 503
        assert "No trained model available" in response.json()["detail"]
    
    def test_predict_with_missing_features(self):
        """Test prediction with missing required features."""
        incomplete_request = {
            "year": 2023,
            "month": 6,
            "hour": 14
            # Missing other required fields
        }
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=incomplete_request
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_values(self):
        """Test prediction with invalid feature values."""
        invalid_request = SAMPLE_PREDICTION_REQUEST.copy()
        invalid_request["month"] = 13  # Invalid month
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.main.setup_mlflow_tracking')
    @patch('src.api.main.mlflow.models.get_model_info')
    @patch('src.api.main.AccidentSeverityPredictor.from_mlflow_model')
    def test_predict_with_prediction_error(self, mock_from_mlflow, mock_get_model_info, mock_setup_tracking):
        """Test prediction when model raises an error."""
        mock_setup_tracking.return_value = None
        mock_get_model_info.return_value = Mock(version=5)
        
        mock_model_dir = Mock()
        mock_model_dir.name = "random_forest_model_v5"
        
        mock_predictor = Mock()
        mock_predictor.model_dir = mock_model_dir
        mock_predictor.predict.side_effect = ValueError("Invalid input data")
        mock_from_mlflow.return_value = mock_predictor
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=SAMPLE_PREDICTION_REQUEST
        )
        
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    @patch('src.api.main.setup_mlflow_tracking')
    @patch('src.api.main.mlflow.models.get_model_info')
    @patch('src.api.main.AccidentSeverityPredictor.from_mlflow_model')
    def test_get_model_info(self, mock_from_mlflow, mock_get_model_info, mock_setup_tracking):
        """Test getting model information."""
        mock_setup_tracking.return_value = None
        mock_get_model_info.return_value = Mock(version=5)
        
        mock_model_dir = Mock()
        mock_model_dir.name = "random_forest_model_v5"
        mock_model_dir.__str__ = Mock(return_value="/path/to/model")

        mock_predictor = Mock()
        mock_predictor.model_dir = mock_model_dir
        mock_predictor.feature_names = ["year", "month", "hour"]
        mock_predictor.class_labels = {"1": "Unscathed", "2": "Light injury"}
        mock_predictor.config = {"some": "config"}
        mock_from_mlflow.return_value = mock_predictor
        
        response = client.get(f"/model/info?api_key={VALID_API_KEY}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "random_forest_model_v5"
        assert data["num_features"] == 3
        assert "feature_names" in data
        assert "class_labels" in data
        assert "config" in data
    
    @patch('src.api.main.setup_mlflow_tracking')
    @patch('src.api.main.mlflow.models.get_model_info')
    def test_get_model_info_no_model(self, mock_get_model_info, mock_setup_tracking):
        """Test getting model info when no model is available."""
        mock_setup_tracking.return_value = None
        mock_get_model_info.side_effect = Exception("no model")
        
        response = client.get(f"/model/info?api_key={VALID_API_KEY}")
        
        assert response.status_code == 503
        assert "No trained model available" in response.json()["detail"]


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Welcome" in response.json()["message"]


class TestRequestValidation:
    """Test request validation for prediction endpoint."""
    
    def test_year_validation(self):
        """Test year field validation."""
        invalid_request = SAMPLE_PREDICTION_REQUEST.copy()
        invalid_request["year"] = 1999  # Too old
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=invalid_request
        )
        assert response.status_code == 422
    
    def test_month_validation(self):
        """Test month field validation."""
        invalid_request = SAMPLE_PREDICTION_REQUEST.copy()
        invalid_request["month"] = 0  # Invalid
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=invalid_request
        )
        assert response.status_code == 422
    
    def test_holiday_validation(self):
        """Test holiday field validation."""
        invalid_request = SAMPLE_PREDICTION_REQUEST.copy()
        invalid_request["holiday"] = 2  # Must be 0 or 1
        
        response = client.post(
            f"/predict?api_key={VALID_API_KEY}",
            json=invalid_request
        )
        assert response.status_code == 422
