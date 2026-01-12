from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
from dotenv import load_dotenv
import mlflow

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


app = FastAPI(
    title="Road Accidents Severity Prediction API",
    description="API for predicting the severity of road accidents in France",
    version="1.0.0"
)

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
    try:
        # Get predictor (will initialize if needed)
        predictor = get_predictor()
        
        # Convert request to dictionary
        input_data = request.model_dump()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Add model version to response
        result['model_version'] = predictor.model_dir.name
        
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
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

