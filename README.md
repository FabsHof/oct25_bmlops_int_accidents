

# Accidents – MLOps Repository

This repository provides the foundation for an MLOps architecture designed for processing and analyzing severity in car accident datasets. It offers a containerized development environment and integrates automated data ingestion via the Kaggle API alongside a PostgreSQL database.

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

## 📚 Technologies Used

* **Python**
* **Pandas** – Data processing
* **Docker & Docker Compose** – Containerization
* **PostgreSQL** – Persistent data storage
* **Kaggle API** – Data acquisition
* **uv** – Virtual environment & dependency management
* **.env** – Environment configuration management

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

#### 3. Run initial preprocessing

The preprocessing script currently represents a basic foundation for further feature engineering and data transformation. 

```bash
python src/data/preprocess.py

```