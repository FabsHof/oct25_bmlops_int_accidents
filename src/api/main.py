from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyQuery

from src.data.ingest_data import (
    load_next_chunk,
    reset_progress
)
from src.utils.database import (
    get_db_connection,
    get_progress_status
)


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

@app.post('/data/ingest-chunk', tags=['data'])
def ingest_data_chunk(api_key: str = Depends(query_schema)):
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
def get_ingestion_progress(api_key: str = Depends(query_schema)):
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
def reset_ingestion_progress(api_key: str = Depends(query_schema)):
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