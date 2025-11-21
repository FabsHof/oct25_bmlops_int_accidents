download_data:
	python src/data/download_data.py
transform_data:
	python src/data/transform_data.py
store_data:
	python src/data/store_data.py
do_etl: download_data transform_data store_data

api_dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
api_prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/