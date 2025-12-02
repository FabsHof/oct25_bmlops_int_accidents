#!/bin/bash
# Exit immediately if any command fails
set -e

# Print message for logging
echo "Starting MLflow Server..."

BACKEND_URI=${MLFLOW_BACKEND_URI}
ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT}
HOST=${MLFLOW_HOST:-0.0.0.0}  # default fallback
PORT=${MLFLOW_PORT:-5000}

# Start the MLflow tracking server with the specified config
exec mlflow server \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host "$HOST" \
    --port "$PORT"
