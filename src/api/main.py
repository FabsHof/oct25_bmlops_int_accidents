import time
from fastapi import FastAPI, Depends, HTTPException, Response, Request
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
from dotenv import load_dotenv
import mlflow
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from fastapi.security import OAuth2PasswordRequestForm
from src.auth.security import get_current_user, create_access_token, authenticate_user
from src.auth.schemas import Token


# Load environment variables
load_dotenv()

from src.models.predict_model import AccidentSeverityPredictor
from src.utils.ml_utils import (
    MODEL_NAME,
    CHAMPION_MODEL_ALIAS,
    FEATURE_COLUMNS,
    CLASS_LABELS,
    setup_mlflow_tracking,
)


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for accident severity prediction."""
    year: int = Field(..., description="Year of the accident", ge=2000, le=2100)
    month: int = Field(..., description="Month of the accident", ge=1, le=12)
    hour: int = Field(..., description="Hour of the accident", ge=0, le=23)
    minute: int = Field(..., description="Minute of the accident", ge=0, le=59)
    user_category: int = Field(..., description="User category (e.g., driver, passenger, pedestrian)")
    sex: int = Field(..., description="Sex of the user")
    year_of_birth: int = Field(..., description="Year of birth of the user", ge=1900, le=2100)
    trip_purpose: int = Field(..., description="Purpose of the trip")
    security: int = Field(..., description="Security equipment used")
    luminosity: int = Field(..., description="Luminosity conditions")
    weather: int = Field(..., description="Weather conditions")
    type_of_road: int = Field(..., description="Type of road")
    road_surface: int = Field(..., description="Road surface condition")
    latitude: float = Field(..., description="Latitude of the accident location")
    longitude: float = Field(..., description="Longitude of the accident location")
    holiday: int = Field(..., description="Holiday indicator (0 or 1)", ge=0, le=1)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    """Response model for accident severity prediction."""
    prediction: int = Field(..., description="Predicted severity class (1-4)")
    prediction_label: str = Field(..., description="Human-readable severity label")
    probabilities: Dict[str, float] = Field(..., description="Probability for each severity class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    model_version: Optional[str] = Field(None, description="Version of the model used")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 3,
                "prediction_label": "Hospitalized wounded",
                "probabilities": {
                    "Unscathed": 0.1,
                    "Light injury": 0.2,
                    "Hospitalized wounded": 0.65,
                    "Killed": 0.05
                },
                "confidence": 0.65,
                "model_version": "accident_severity_rf_20251210_162637"
            }
        }
    )


class DriftMetricsRequest(BaseModel):
    """Request model for submitting drift metrics from Airflow DAG."""
    overall_drift_score: float = Field(..., description="Overall data drift score (0-1)", ge=0, le=1)
    feature_drift_scores: Dict[str, float] = Field(..., description="Per-feature drift scores")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp of drift computation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_drift_score": 0.15,
                "feature_drift_scores": {
                    "year": 0.05,
                    "month": 0.12,
                    "hour": 0.08,
                    "user_category": 0.20
                },
                "timestamp": "2026-01-12T16:30:00Z"
            }
        }
    )


app = FastAPI(
    title="Road Accidents Severity Prediction API",
    description="API for predicting the severity of road accidents in France",
    version="1.0.0"
)

# Prometheus metrics registry and metrics
registry = CollectorRegistry()

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions served",
    registry=registry,
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
    registry=registry,
)

predictions_by_severity = Counter(
    "predictions_by_severity",
    "Predictions by severity label",
    ["severity"],
    registry=registry,
)

data_drift_score = Gauge(
    "data_drift_score",
    "Latest data drift score (0-1)",
    registry=registry,
)

feature_drift = Gauge(
    "feature_drift",
    "Feature-level drift score",
    ["feature"],
    registry=registry,
)

# Initialize drift metrics to zero so Grafana panels have values
data_drift_score.set(0.0)
for feature in FEATURE_COLUMNS:
    feature_drift.labels(feature=feature).set(0.0)

@app.on_event("startup")
def startup_event():
    """Initialize MLflow tracking on application startup."""
    setup_mlflow_tracking()

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(
        data={"sub": user.username}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

def get_predictor() -> AccidentSeverityPredictor:
    """
    Get or initialize the predictor with the best available model.
    
    Returns:
        Initialized AccidentSeverityPredictor
        
    Raises:
        HTTPException: If no model is available
    """
    champion_uri = f"models:/{MODEL_NAME}@{CHAMPION_MODEL_ALIAS}"

    try:
        model_info = mlflow.models.get_model_info(champion_uri)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Please train a model first."
        )

    try:
        predictor = AccidentSeverityPredictor.from_mlflow_model(
            champion_uri,
            model_version=str(model_info.version) if getattr(model_info, "version", None) else None,
            class_labels=CLASS_LABELS,
            fallback_feature_names=FEATURE_COLUMNS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )
    
    return predictor

@app.get('/')
def read_root():
    return {"message": "Welcome to the Road Accidents Severity Prediction API"}

@app.get('/health')
def health_check(current_user = Depends(get_current_user)):
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}


@app.post('/predict', tags=['model'], response_model=PredictionResponse)
def predict_severity(
    request: PredictionRequest,
    current_user = Depends(get_current_user)
) -> PredictionResponse:
    """
    Predict the severity of a road accident based on input features.
    
    This endpoint uses the best available trained model to predict
    the severity of a road accident given various features about
    the accident circumstances, location, and involved parties.
    
    Args:
        request: PredictionRequest containing all required features
        api_key: API key for authentication
        
    Returns:
        PredictionResponse with prediction results and probabilities
        
    Raises:
        HTTPException: If prediction fails or model is unavailable
    """
    start_time = time.perf_counter()
    try:
        # Get predictor (will initialize if needed)
        predictor = get_predictor()
        
        # Convert request to dictionary
        input_data = request.model_dump()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Add model version to response
        result['model_version'] = predictor.model_dir.name

        # Update Prometheus metrics
        predictions_total.inc()
        prediction_latency_seconds.observe(time.perf_counter() - start_time)
        predictions_by_severity.labels(severity=result.get('prediction_label', 'unknown')).inc()
        
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics for scraping.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")


@app.post("/metrics/drift")
async def submit_drift_metrics(drift_data: DriftMetricsRequest, current_user=Depends(get_current_user)):
    """
    Submit drift metrics from Airflow DAG to update Prometheus gauges.
    
    This endpoint is called by the DAG after drift detection to persist
    the computed drift scores in Prometheus so Grafana can visualize them.
    
    Args:
        drift_data: DriftMetricsRequest with overall and per-feature drift scores
        current_user: Authenticated user (required for security)
    
    Returns:
        Status confirmation message
    """
    try:
        # Update overall drift score gauge
        data_drift_score.set(drift_data.overall_drift_score)
        
        # Update per-feature drift gauges
        for feature, score in drift_data.feature_drift_scores.items():
            try:
                feature_drift.labels(feature=feature).set(score)
            except Exception as e:
                # Log but don't fail if a feature isn't recognized
                pass
        
        return {
            "status": "success",
            "message": "Drift metrics updated successfully",
            "overall_drift_score": drift_data.overall_drift_score,
            "features_updated": len(drift_data.feature_drift_scores)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update drift metrics: {str(e)}"
        )


@app.get('/model/info', tags=['model'])
def get_model_info( current_user = Depends(get_current_user)) -> Dict:
    """
    Get information about the currently loaded model.
    
    Returns:
        Dictionary with model metadata including version, features, and metrics
    """
    try:
        predictor = get_predictor()
        
        return {
            "model_version": predictor.model_dir.name,
            "model_path": str(predictor.model_dir),
            "num_features": len(predictor.feature_names),
            "feature_names": predictor.feature_names,
            "class_labels": predictor.class_labels,
            "config": predictor.config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

