# 🚗 Road accidents in France

The objective of this MLOps project is to build a MLOps pipeline with the aim of predicting the severity of road accidents in France. Predictions will be based on historical data.

## 🗂️ Project Organization

The project is structured as follows:
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

## 🛫 Prerequisites

> everything that has to be done once before starting development.

0. install Python 3.11 or higher.
1. install [UV](https://docs.astral.sh/uv/getting-started/installation/).
2. install [Docker](https://docs.docker.com/get-docker/).
3. create `.env.example` from `.env`-file and adapt values if needed

## ⌨️ Development Setup

> do this every time you start working on the project.

1. sync dependencies and enable virtual environment:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. start 🐳 Docker containers:
   ```bash
   make build  # improved build process
   make up     # start containers in detached mode
   ```
3. access services (see default credentials below, change in `.env` file if needed):
   - **Airflow UI**: `http://localhost:8080` (default credentials: `airflow` / `airflow`)
   - **Grafana**: `http://localhost:3000` (default credentials: `admin` / `admin`)
   - **MLflow UI**: `http://localhost:5001` (default credentials: `mlflow` / `mlflow`)
   - **MinIO UI**: `http://localhost:9000` (default credentials: `mini_user` / `mini_password`)


## 🪁 Airflow DAGs

The project includes the following Airflow DAGs for orchestrating workflows:

- [accidents data dag](./dags/accidents_data_dag.py): Manages chunked or full data ingestion. This allows for simulating data evolution over time.
- [accidents ml dag](./dags/accidents_ml_dag.py): Handles the machine learning pipeline, including data cleaning, dataset splitting, model training and evaluation.
- [accidents dag](./dags/accidents_dag.py): Orchestrates the ETL pipeline for data ingestion, cleaning, and model training.


## 📊 Model Training Details

### 🧪 Validation Strategy

- **Static train/validation/test splits**: 60% / 20% / 20%
- **Stratified sampling**: Ensures balanced class distribution
- **Database-tracked**: Split assignments stored in `clean_data.dataset_split` column
- **Reproducible**: Fixed random seed (42) for consistent splits

### 🧮 Model & Metrics

- **Model**: Random Forest Classifier with 100 trees
- **Metrics**: Accuracy, Precision, Recall, F1-score (weighted), ROC-AUC
- **Artifacts**: Model, metrics, feature importance, confusion matrix, config
