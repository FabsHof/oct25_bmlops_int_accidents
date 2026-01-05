from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.security import APIKeyQuery
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.ingest_data import (
    load_next_chunk,
    reset_progress
)
from src.utils.database import (
    get_db_connection,
    get_progress_status
)
from src.models.predict_model import (
    AccidentSeverityPredictor,
    get_best_model_dir
)
from src.monitoring.metrics import get_metrics_collector
from src.monitoring.drift import get_drift_detector


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


app = FastAPI(
    title="Road Accidents Severity Prediction API",
    description="API for predicting the severity of road accidents in France",
    version="1.0.0"
)

query_schema = APIKeyQuery(name="api_key")

# Global variable to cache the predictor
_predictor: Optional[AccidentSeverityPredictor] = None


def verify_api_key(api_key: str = Depends(query_schema)) -> str:
    """
    Verify the API key provided by the user.
    
    Args:
        api_key: API key from query parameter
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    expected_key = os.getenv("API_KEY")
    
    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API_KEY not configured on server"
        )
    
    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


def get_predictor() -> AccidentSeverityPredictor:
    """
    Get or initialize the predictor with the best available model.
    
    Returns:
        Initialized AccidentSeverityPredictor
        
    Raises:
        HTTPException: If no model is available
    """
    global _predictor
    
    if _predictor is None:
        model_dir = get_best_model_dir()
        
        if model_dir is None:
            raise HTTPException(
                status_code=503,
                detail="No trained model available. Please train a model first."
            )
        
        try:
            _predictor = AccidentSeverityPredictor(model_dir)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return _predictor

@app.get('/')
def read_root():
    return {"message": "Welcome to the Road Accidents Severity Prediction API"}

@app.get('/health')
def health_check(api_key: str = Depends(verify_api_key)):
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}


@app.get('/metrics', tags=['monitoring'])
def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics for scraping
    """
    metrics_collector = get_metrics_collector()
    return Response(
        content=metrics_collector.get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post('/predict', tags=['model'], response_model=PredictionResponse)
def predict_severity(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
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
    start_time = time.time()
    metrics_collector = get_metrics_collector()
    drift_detector = get_drift_detector()
    
    try:
        # Get predictor (will initialize if needed)
        predictor = get_predictor()
        
        # Convert request to dictionary
        input_data = request.model_dump()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Record metrics
        latency = time.time() - start_time
        metrics_collector.record_prediction(
            severity_class=result['prediction'],
            severity_label=result['prediction_label'],
            confidence=result['confidence'],
            latency=latency
        )
        
        # Add data to drift detector buffer
        drift_detector.add_prediction_data(input_data)
        
        # Add model version to response
        result['model_version'] = predictor.model_dir.name
        
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        metrics_collector.record_error('http_error')
        raise
    except Exception as e:
        metrics_collector.record_error('internal_error')
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get('/monitoring/drift', tags=['monitoring'])
def get_drift_status(api_key: str = Depends(verify_api_key)) -> Dict:
    """
    Get the current drift detection status.
    
    Returns:
        Dictionary with drift buffer status and latest drift analysis
    """
    drift_detector = get_drift_detector()
    return drift_detector.get_buffer_status()


@app.post('/monitoring/drift/analyze', tags=['monitoring'])
def trigger_drift_analysis(api_key: str = Depends(verify_api_key)) -> Dict:
    """
    Manually trigger drift analysis on buffered data.
    
    Returns:
        Drift analysis results
    """
    drift_detector = get_drift_detector()
    result = drift_detector.analyze_drift()
    return result


@app.get('/model/info', tags=['model'])
def get_model_info(api_key: str = Depends(verify_api_key)) -> Dict:
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


@app.get('/train', tags=['model'])
def train_model(api_key: str = Depends(verify_api_key)):
    # Placeholder for training logic
    return {"training": "Model training logic not yet implemented"}

@app.post('/data/ingest-chunk', tags=['data'])
def ingest_data_chunk(api_key: str = Depends(verify_api_key)):
    """
    Load the next chunk of data into the database.
    
    This endpoint simulates data evolution by loading data incrementally.
    Each call loads the next chunk for all tables that haven't completed yet.
    
    Returns:
        Dictionary with loading results and progress for each table
    """
    try:
        result = load_next_chunk()
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to load data chunk'))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data chunk: {str(e)}")

@app.get('/data/progress', tags=['data'])
def get_ingestion_progress(api_key: str = Depends(verify_api_key)):
    """
    Get the current progress of data ingestion for all tables.
    
    Returns:
        Dictionary with progress information including rows loaded, total rows,
        and completion percentage for each table
    """
    try:
        conn = get_db_connection()
        try:
            progress = get_progress_status(conn)
            
            if not progress:
                return {
                    'message': 'No data ingestion in progress. Use POST /data/ingest-chunk to start.',
                    'tables': {}
                }
            
            # Calculate overall progress
            total_rows_all = sum(p['total_rows'] for p in progress.values())
            loaded_rows_all = sum(p['rows_loaded'] for p in progress.values())
            overall_percentage = (loaded_rows_all / total_rows_all * 100) if total_rows_all > 0 else 0
            all_complete = all(p['is_complete'] for p in progress.values())
            
            return {
                'tables': progress,
                'overall': {
                    'total_rows': total_rows_all,
                    'loaded_rows': loaded_rows_all,
                    'progress_percentage': round(overall_percentage, 2),
                    'is_complete': all_complete
                }
            }
        finally:
            conn.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving progress: {str(e)}")

@app.post('/data/reset-progress', tags=['data'])
def reset_ingestion_progress(api_key: str = Depends(verify_api_key)):
    """
    Reset the data ingestion progress to start from the beginning.
    
    This will clear all progress tracking and allow restarting the
    incremental loading process from scratch.
    
    Returns:
        Dictionary with reset status
    """
    try:
        result = reset_progress()
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to reset progress'))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting progress: {str(e)}")