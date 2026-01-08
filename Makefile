# ==============================
# API targets
# ==============================
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
	@echo ">>> Airflow services stopped."

# ==============================
# General targets
# ==============================
setup: airflow-setup