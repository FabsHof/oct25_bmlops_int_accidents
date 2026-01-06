"""
Centralized file for model utilities like the pydantic BaseModel class definitions
"""

from pydantic import BaseModel, Field, ConfigDict

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