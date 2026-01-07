download_data:
	@echo ">>> Starting 'download_data' task"
	@python -m src.data.download_data
	@echo ">>> 'download_data' task completed"
transform_data:
	@echo ">>> Starting 'transform_data' task"
	@python -m src.data.transform_data
	@echo ">>> 'transform_data' task completed"
ingest_data_full:
	@echo ">>> Starting 'ingest_data_full' task"
	@python -m src.data.ingest_data --mode full
	@echo ">>> 'ingest_data_full' task completed"
ingest_data_chunked:
	@echo ">>> Starting 'ingest_data_chunked' task"
	@python -m src.data.ingest_data --mode chunked
	@echo ">>> 'ingest_data_chunked' task completed"
clean_data:
	@echo ">>> Starting 'clean_data' task"
	@python -m src.data.clean_data
	@echo ">>> 'clean_data' task completed"

assign_splits:
	@echo ">>> Starting 'assign_splits' task"
	@python -m src.data.clean_data --assign-splits
	@echo ">>> 'assign_splits' task completed"

train_model:
	@echo ">>> Starting 'train_model' task"
	@python -m src.models.train_model
	@echo ">>> 'train_model' task completed"

train_model_with_test:
	@echo ">>> Starting 'train_model_with_test' task"
	@python -m src.models.train_model --evaluate-test
	@echo ">>> 'train_model_with_test' task completed"

do_etl_full: download_data ingest_data_full clean_data
do_etl_chunked: download_data ingest_data_chunked clean_data
do_ml_pipeline: clean_data assign_splits train_model

api_dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
api_prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/

# ==============================
# Airflow targets
# ==============================
airflow-setup:
	@echo ">>> Setting up Airflow environment..."
	@chmod +x ./scripts/airflow-setup.sh
	@./scripts/airflow-setup.sh

airflow-init:
	@echo ">>> Initializing Airflow database..."
	@docker compose up airflow-init

airflow-up:
	@echo ">>> Starting Airflow services..."
	@docker compose up -d airflow_db airflow_redis airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer
	@echo ">>> Airflow services started. Access the UI at http://localhost:8080"

airflow-down:
	@echo ">>> Stopping Airflow services..."
	@docker compose stop airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer airflow_redis airflow_db

airflow-logs:
	@docker compose logs -f airflow-apiserver airflow-scheduler airflow-worker

airflow-restart:
	@echo ">>> Restarting Airflow services..."
	@docker compose restart airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer

# ==============================
# Docker targets
# ==============================
docker-build:
	@echo ">>> Building all Docker images..."
	@docker compose build

docker-up:
	@echo ">>> Starting all services..."
	@docker compose up -d

docker-down:
	@echo ">>> Stopping all services..."
	@docker compose down

docker-clean:
	@echo ">>> Removing all containers and volumes..."
	@docker compose down -v