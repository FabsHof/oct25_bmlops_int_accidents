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

do_etl: download_data transform_data ingest_data_full

api_dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
api_prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/