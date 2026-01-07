"""
Example DAG for the Accidents MLOps Project
============================================

This DAG demonstrates a basic ML pipeline workflow that can be extended
for the traffic accidents prediction project.

DAG Tasks:
1. download_data - Download raw data from Kaggle
2. ingest_data - Ingest data into the database
3. clean_data - Clean and preprocess data
4. train_model - Train the ML model

To use this DAG:
1. Ensure all services are running (docker compose up -d)
2. Access Airflow UI at http://localhost:8080
3. Enable the DAG and trigger manually or wait for schedule
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Default arguments for all tasks
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id="accidents_ml_pipeline",
    default_args=default_args,
    description="ML Pipeline for Traffic Accidents Severity Prediction",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "accidents", "ml-pipeline"],
) as dag:

    # Task 1: Download data from Kaggle
    download_data = BashOperator(
        task_id="download_data",
        bash_command="cd /app && python -m src.data.download_data",
    )

    # Task 2: Ingest data into the database
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command="cd /app && python -m src.data.ingest_data --mode chunked",
    )

    # Task 3: Clean and preprocess data
    clean_data = BashOperator(
        task_id="clean_data",
        bash_command="cd /app && python -m src.data.clean_data",
    )

    # Task 4: Train the model
    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /app && python -m src.models.train_model",
    )

    # Define task dependencies
    download_data >> ingest_data >> clean_data >> train_model
