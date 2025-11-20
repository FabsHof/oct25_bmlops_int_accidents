from fastapi import FastAPI, Depends
from fastapi.security import APIKeyQuery


app = FastAPI()

query_schema = APIKeyQuery(name="api_key")

@app.get('/')
def read_root():
    return {"message": "Welcome to the Road Accidents Severity Prediction API"}

@app.get('/health')
def health_check(api_key: str = Depends(query_schema)):
    return {"status": "healthy"}

@app.get('/predict', tags=['model'])
def predict_severity(api_key: str = Depends(query_schema)):
    # Placeholder for prediction logic
    return {"prediction": "Severity prediction logic not yet implemented"}

@app.get('/train', tags=['model'])
def train_model(api_key: str = Depends(query_schema)):
    # Placeholder for training logic
    return {"training": "Model training logic not yet implemented"}