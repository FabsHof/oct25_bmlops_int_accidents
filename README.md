

# Accidents – MLOps Repository

This repository provides the foundation for an MLOps architecture designed for processing and analyzing severity in car accident datasets. It offers a containerized development environment and integrates automated data ingestion via the Kaggle API alongside a PostgreSQL database. This project includes a production-ready MLflow Tracking Server with MinIO for artifact storage and PostgreSQL for metadata.

---

## 📦 Project Overview

This project includes:

* Structured MLOps architecture (skeleton)
* Virtual Python environment managed with **uv**
* Container orchestration using **Docker & Docker Compose**
* Database integration via **PostgreSQL**
* Data ingestion through the **Kaggle API**
* Configurable environment variables using `.env`


---

## ✅ Requirements

Please ensure the following components are installed:

* Docker & Docker Compose
* Python >= 3.10
* uv (Python package manager)
* Kaggle account + Kaggle API token
* Git

---

## 🚀 Setup Guide

### 1. Clone the repository

```bash
git clone git@github.com:HanDBerlin/accidents.git
cd accidents
```

> Note: If your repository has a different name, adjust the folder name accordingly.

---

### 2. Create virtual environment (uv)

```bash
uv venv
source .venv/bin/activate  # Linux / Mac
# or
.venv\\Scripts\\activate     # Windows
```

---

### 3. Configure environment variables

Create a file named `.env` in the root directory with the following content:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=accidents_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

DATASET_NAME=ahmedlahlou/accidents-in-france-from-2005-to-2016
DATA_RAW_PATH=data/raw

MLFLOW_BACKEND_URI= postgresql+psycopg2://mlflow:mlflow@mlflow-db:5432/mlflow
MLFLOW_ARTIFACT_ROOT= s3://mlflow-artifacts
MLFLOW_HOST= 0.0.0.0
MLFLOW_PORT= 5000

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
```

🔧 These values can be adjusted individually depending on your local setup. In most cases, the default PostgreSQL values can remain unchanged unless there is a port conflict.

---

### 4. Configure Kaggle API

1. Download your `kaggle.json` file from your Kaggle account.
2. Place it in:

```bash
~/.kaggle/kaggle.json
```

3. Set correct permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

### 5. 🐳 Docker Setup & Usage

### First-Time Setup

When running the project for the first time, Docker images must be built before the containers can be started. This is also required whenever changes are made to the `Dockerfile` or the Docker Compose configuration.

This projects include the following containers:
| Container Name    | Description                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| **mlflow**        | MLflow Tracking Server used to log experiments, parameters, metrics, and models.                       |
| **mlflow-db**     | PostgreSQL database used by MLflow to store run metadata (parameters, metrics, tags, etc.).            |
| **minio**         | S3-compatible object storage used as the MLflow artifact store (stores model files, plots, artifacts). |
| **minio-console** | Browser-based UI for inspecting MinIO buckets and stored artifacts.                                    |
| **postgres**      | Main application database used by the MLOps project (e.g., for data pipelines or application state).   |


Run the following command:

```bash
docker compose up --build
```

### Normal Startup

After the initial Docker build has been completed, you can start the project containers using the standard command:

```bash
docker compose up -d
```


### Stopping the container

When you are finished working with the project, stop all running containers with:

```bash
docker compose down
```


---

### Accessing the MLflow UI

To open the MLflow Tracking interface, visit:

http://localhost:5000

The UI will start empty until you run your first experiment.
All tracked parameters, metrics, models, and artifacts will appear here.

---

### Accessing the MinIO Console

To browse MLflow artifacts stored in MinIO, open:

http://localhost:9001

Use the following default credentials:

Username: minioadmin
Password: minioadmin


---



## 📚 Technologies Used

* **Python**
* **Pandas** – Data processing
* **Docker & Docker Compose** – Containerization
* **PostgreSQL** – Persistent data storage
* **Kaggle API** – Data acquisition
* **uv** – Virtual environment & dependency management
* **.env** – Environment configuration management
* **MLflow** — Experiment tracking platform used to log parameters, metrics, models, and artifacts.
* **MinIO** — S3-compatible object storage used as the MLflow artifact store.
* **MinIO Console** — Web-based interface for browsing buckets and stored ML artifacts.

---





## 🚀 Getting Started: First Steps in Using This Repository

This section describes the initial workflow to run the data pipeline and prepare the foundation for further development such as preprocessing, exploration, and model training.

### Overview of the Initial Data Flow

The current pipeline is structured as follows:

1. **Ingest data from Kaggle**
2. **Load raw data into PostgreSQL**
3. **Initial preprocessing (foundation for further development)**

These steps are implemented in the `src/data` directory and represent the starting point for all downstream ML workflows.

---

### Step-by-Step Workflow

#### 1. Ingest dataset from Kaggle

This step downloads the dataset defined in your `.env` file via the Kaggle API and stores it in the raw data directory.


```bash
python src/data/ingest_kaggle.py
```
---

#### 2. Load data into PostgreSQL

This step takes the raw dataset and loads it into the configured PostgreSQL database.


```bash
python src/data/load_to_db.py
```

cRun initial preprocessing

The preprocessing script currently represents a basic foundation for further feature engineering and data transformation. 

```bash
python src/data/preprocess.py

```



#### 3.  Run a dummy experiment to test mlflow 

Access the MLflow container:

```bash
docker compose exec mlflow bash
```
Navigate to the mounted tests folder:
```bash
cd /app/tests
```
Execute the test script:
```bash
python test_mlflow.py
```


#### 4. View the results

MLflow UI:
Open http://localhost:5000 to see your experiment demo_experiment.

Minio Console (Artifacts):
Open http://localhost:9001 with your Minio credentials (from docker-compose.yml)