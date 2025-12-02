# Use official slim Python image as base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install Python packages
COPY mlflow/requirements.txt   /app
RUN pip install --no-cache-dir -r  /app/requirements.txt

# Copy MLflow config and start script into the container
COPY mlflow/mlflow-config.yml  /app
COPY mlflow/mlflow-server.sh  /app
RUN chmod +x /app/mlflow-server.sh

# Expose MLflow default port
EXPOSE 5000

# Set default command to run when container starts
CMD ["/app/mlflow-server.sh"]
