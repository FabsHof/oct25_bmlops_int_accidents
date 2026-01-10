
test:
	pytest tests/

# ==============================
# General targets
# ==============================
up: 
	docker compose up -d
down: 
	docker compose down
build:
	DOCKER_BUILDKIT=1 docker compose build --parallel
clean: 
	docker compose down -v
logs: 
	docker compose logs -f