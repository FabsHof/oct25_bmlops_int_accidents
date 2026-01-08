#!/bin/bash
# ==============================================================================
# Airflow Setup Script
# ==============================================================================
# This script initializes the Airflow environment for the project.
# It creates necessary directories, sets up environment variables, and
# prepares the Docker containers for Airflow.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Airflow Setup Script ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

echo "Note: Docker uses requirements-airflow.txt with minimal dependencies"
echo ""

# Create necessary directories for Airflow
echo ">>> Creating Airflow directories..."
mkdir -p ./dags
mkdir -p ./airflow/logs
mkdir -p ./airflow/plugins
mkdir -p ./airflow/config
echo "✓ Directories created"
echo ""

# Check if .env file exists, if not create from example
if [[ ! -f .env ]]; then
    echo ">>> .env file not found, creating from .env.example..."
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "✓ .env file created from .env.example"
    else
        echo "⚠ Warning: .env.example not found. Please create .env manually."
    fi
    echo ""
fi

# Set AIRFLOW_UID if not already present in .env
if ! grep -q "^AIRFLOW_UID=" .env 2>/dev/null; then
    echo ">>> Setting AIRFLOW_UID in .env..."
    # Use current user ID on Linux/macOS
    CURRENT_UID=$(id -u)
    echo "AIRFLOW_UID=$CURRENT_UID" >> .env
    echo "✓ AIRFLOW_UID set to $CURRENT_UID"
else
    echo ">>> AIRFLOW_UID already set in .env"
fi
echo ""

echo "🎉 Airflow setup is complete!"
