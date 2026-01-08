# ==============================
# Data targets
# ==============================
# TODO: tbc

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
up: 
	docker compose up -d
up_build: 
	docker compose up -d --build
down: 
	docker compose down
down_volumes: 
	docker compose down -v
logs: 
	docker compose logs -f