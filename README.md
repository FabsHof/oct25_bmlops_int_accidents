# 🚗 Road accidents in France

The objective of this MLOps project is to build a MLOps pipeline with the aim of predicting the severity of road accidents in France. Predictions will be based on historical data.

## ↪️ Architecture Overview

<!-- Mermaid.js script copied from the most up to date version in src/utils/schemas.py -->
```mermaid

%% %%{init: {"flowchart": {"curve": "curve"}}}%%
%% Choose between curve, linear, step, cardinal
%% default: curve

flowchart LR

    classDef transp fill:transparent,stroke:transparent;
    classDef user fill:#ff6f00,stroke:#b34700,stroke-width:3px,color:#ffffff;
    classDef airflow fill:#ffd54f,stroke:#b28704,stroke-width:3px,color:#000000;
    classDef api fill:#00c853,stroke:#007e33,stroke-width:3px,color:#ffffff;
    classDef db fill:#2962ff,stroke:#0039cb,stroke-width:3px,color:#ffffff;
    classDef mlflow fill:#00b0ff,stroke:#007bb2,stroke-width:3px,color:#ffffff;

    linkStyle default stroke:#000999,stroke-width:2px

    subgraph UA[USER APP]
        IFS[INTERACTIVE FEATURE INPUT]:::user
    end

    IFS --> |Features Input|EP



    subgraph MLF[MLFLOW]
        SR[STORE RUN]:::mlflow
        PLMS[PROD & LAST MODEL SCORE]:::mlflow
        UT[UPDATE TAGS]:::mlflow
        IPM[IDENTIFY PROD. MODEL]:::mlflow
    end



    IPM --> |Prod. Model|EP
    PLMS --> |Metrics|DPM





    subgraph API[MODEL API]
        ET[ENDPOINT /train]:::api
        EP[ENDPOINT /predict]:::api
    end


    EP --> |Query Prod. Model|IPM
    EP --> |Prediction|IFS
    ET -->|Data Query|FPD
    ET -->|Last Model & Metrics|SR



    subgraph DB[DATABASE]
        SRD[STORE RAW DATA]:::db
        SPD[STORE PREPROCESSED DATA]:::db
        FPD[FETCH PREPROCESSED DATA]:::db
    end

    FPD -->|Preprocessed Data|ET



    subgraph CAF[CRON / AIRFLOW]
        START[PROCESS START]:::airflow
        ETL[ETL]:::airflow
        TE[TRAIN & EVALUATE]:::airflow
        DPM[IDENTIFY PROD. MODEL]:::airflow
        END[PROCESS END<br>/ LOOP TO ETL]:::airflow
    end

    START --> ETL --> TE --> DPM --> END
    DPM -->|IF last model is better: Update Query|UT
    DPM --> |Last & Prod. Model Score Query|PLMS
    ETL --> |csv Files Query|DD
    ETL --> |Raw Data|SRD
    ETL --> |Preprocessed Data|SPD
    TE --> |Train New Model Query|ET
    linkStyle 8 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 9 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 10 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 11 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5

    

    subgraph KG[KAGGLE]
        DD[DATASET DOWNLOAD]:::db
    end

    DD --> |Raw csv Files|ETL



    subgraph LG[LEGEND]
        direction LR
        PROCESS
        D1[ ]:::transp
        D2[ ]:::transp
        D3[ ]:::transp
        D4[ ]:::transp
        D1 --> |DATA| D2
        D3 --> |PROCESS STEPS| D4
        linkStyle 20 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    end

%% =======================
%% CLICKABLE LINKS EXAMPLE
%% =======================

    %click DD "?page=training" "Test page"

```


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

0. install [UV](https://docs.astral.sh/uv/getting-started/installation/)
1. install python and its dependencies:
   ```bash
   uv sync
   ```
2. install [Docker](https://docs.docker.com/get-docker/).
3. create `.env` from the `.env.example` file and adapt values if needed

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

## Streamlit Presentation

1. **Start the API locally with:**
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. **Start the Streamlit App locally with:**
   ```bash
   PYTHONPATH=. streamlit run src/streamlit/streamlit_app.py
   ```
3. **The Streamlit App can be accessed at:\n**
   http://localhost:8501/