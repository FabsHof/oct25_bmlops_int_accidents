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

## ⚙️ Setup

1. Install Python 3.11 or higher.
2. Install `uv` package manager from [uv package manager](https://uv.dev/).
3. Set up an environment variables file `.env` in the root directory (e.g., see `.env.example`).
3. Create a virtual environment:
   ```bash
   uv venv create .venv
   ```
4. Install dependencies:
   ```bash
   uv sync
   ```
5. Create and update environment variables in `.env` file as needed (see `.env.example`).

## ⌨️ Development

1. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```
2. Run the FastAPI application:
   ```bash
   make api_dev
    ```
3. Access the API documentation at `http://localhost:8000/docs` (find e.g. the API key in the `.env` file, set it via the "Authorize" button in the Swagger UI).

## 📊 Data Ingestion

This project supports two modes of data ingestion:

### Full Batch Loading
Load all data at once using the traditional ETL process:
```bash
make do_etl
```

### Chunked/Incremental Loading
Load data in chunks to simulate data evolution over time. This is useful for testing incremental model training and monitoring data arrival patterns.

- **Using Makefile:**
```bash
make ingest_data_chunked
```
- **Using the script directly:**
```bash
python -m src.data.ingest_data --mode chunked --chunk-size 10000
```
- **Using the API:**

1. Start the API server:
   ```bash
   make api_dev
   ```

2. Check ingestion progress:
   ```bash
   curl "http://localhost:8000/data/progress?api_key=YOUR_API_KEY"
   ```

3. Load the next chunk of data:
   ```bash
   curl -X POST "http://localhost:8000/data/ingest-chunk?api_key=YOUR_API_KEY"
   ```