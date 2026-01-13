# Copilot Instructions - Road Accidents MLOps Project

## Project Overview
MLOps pipeline for predicting traffic accident severity in France. Multi-container Docker architecture with Airflow orchestration, MLflow tracking, FastAPI serving, and Streamlit frontend.

## Architecture & Data Flow
1. **ETL** (Airflow DAG) → Downloads from Kaggle → Cleans data → PostgreSQL (`accidents_db`)
2. **Training** (Airflow DAG) → Reads from DB → Trains RandomForest with GridSearchCV → Logs to MLflow
3. **Serving** (FastAPI) → Loads champion model from MLflow → Exposes `/predict` endpoint
4. **Monitoring** → Evidently drift detection → Prometheus metrics → Grafana dashboards

## Key Services (docker-compose.yml)
| Service | Port | Purpose |
|---------|------|---------|
| `airflow-apiserver` | 8080 | DAG orchestration UI |
| `mlflow` | 5001 | Experiment tracking, model registry |
| `api` | 8000 | FastAPI prediction service |
| `streamlit` | 8501 | User frontend |
| `accidents_db` | 5433 | Data storage (PostgreSQL) |
| `mlflow_minio` | 9000 | Artifact storage (S3-compatible) |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards |

## Developer Commands
```bash
make up       # Start all containers (docker compose up -d)
make down     # Stop containers
make build    # Rebuild with BuildKit
make logs     # Follow container logs
make clean    # Stop and remove volumes
make test     # Run pytest
```

## Git Workflow
- Single `main` branch as source of truth
- Create feature branches for all work
- Open PRs to merge into `main` via **squash merge**

## Airflow 3.x Notes
This project uses Airflow 3.x SDK with breaking changes from 2.x:
```python
# Use new SDK imports (not airflow.decorators)
from airflow.sdk import dag, task, setup, teardown, Variable

# CeleryExecutor with separate dag-processor service
# API server replaces webserver (port 8080)
```
DAG parsing timeout extended to 60s for slower machines (see `docker-compose.yml`).

## Code Conventions

### DAGs ([dags/](dags/))
- Heavy imports (pandas, mlflow, sklearn) go **inside task functions** to speed DAG parsing
- Use `@setup` decorator for initialization tasks
- Chain tasks explicitly with Airflow 3.x SDK: `from airflow.sdk import dag, task`

### ML Utilities ([src/utils/ml_utils.py](src/utils/ml_utils.py))
Central config source - always import constants from here:
```python
from src.utils.ml_utils import (
    MODEL_NAME,           # 'random_forest_model'
    CHAMPION_MODEL_ALIAS, # 'champion'
    FEATURE_COLUMNS,      # List of 16 features
    TARGET_COLUMN,        # 'severity'
    setup_mlflow_tracking,
    load_training_data_from_db,
)
```

### Database Access ([src/utils/database.py](src/utils/database.py))
```python
from src.utils.database import get_db_connection
conn = get_db_connection()  # Uses ACCIDENTS_POSTGRES_* env vars
```

### Logging
Use project logger, not print:
```python
from src.utils import logging
logging.info("Message")
```

### API Endpoints ([src/api/main.py](src/api/main.py))
- JWT auth via `/token` endpoint (OAuth2)
- Pydantic models: `PredictionRequest`, `PredictionResponse` in `ml_utils.py`
- Prometheus metrics exposed at `/metrics`

### Drift Detection ([src/monitoring/drift.py](src/monitoring/drift.py))
Uses Evidently 0.7+ with explicit `DataDefinition`:
```python
from src.monitoring.drift import DriftDetector
detector = DriftDetector(reference_data=train_df)
```

## Testing
Tests in `tests/unit/` mirror `src/` structure. Run:
```bash
pytest tests/                    # All tests
pytest tests/unit/api/           # API tests only
pytest -k "test_predict"         # Pattern match
```

## Environment Variables
Required in `.env` (see docker-compose for complete list):
- `ACCIDENTS_POSTGRES_*` - Data database credentials
- `MLFLOW_*` / `MINIO_*` - Tracking & artifact storage
- `SECRET_KEY`, `ALGORITHM` - JWT auth

## Feature Schema (16 columns)
`year`, `month`, `hour`, `minute`, `user_category`, `sex`, `year_of_birth`, `trip_purpose`, `security`, `luminosity`, `weather`, `type_of_road`, `road_surface`, `latitude`, `longitude`, `holiday`

Target: `severity` (1=Unscathed, 2=Light injury, 3=Hospitalized, 4=Killed)

## File Patterns
| Pattern | Location |
|---------|----------|
| DAG definitions | `dags/*.py` |
| Data processing | `src/data/` |
| Model training | `src/models/train_model.py` |
| API endpoints | `src/api/main.py` |
| Streamlit pages | `src/streamlit/pages/` |
| Docker configs | `Dockerfile.*` |
