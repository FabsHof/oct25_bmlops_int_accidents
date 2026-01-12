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


### Loading the API class objects

from src.utils.ml_utils import (
    PredictionRequest,
    PredictionResponse
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

