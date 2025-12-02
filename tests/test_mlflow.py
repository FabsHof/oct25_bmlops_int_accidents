# /Tests/_test_mlflow.py
import mlflow
import os

# -------------------------------
# 1️⃣ MLflow Tracking URI setzen
# -------------------------------
# Wenn Script innerhalb Container läuft, kann 'mlflow' als Hostname genutzt werden
mlflow.set_tracking_uri("http://mlflow:5000")

# -------------------------------
# 2️⃣ Experiment definieren
# -------------------------------
experiment_name = "demo_experiment"
mlflow.set_experiment(experiment_name)

# -------------------------------
# 3️⃣ Run starten
# -------------------------------
with mlflow.start_run():
    # Parameter loggen
    mlflow.log_param("param1", 42)
    mlflow.log_param("param2", "test_value")
    
    # Metriken loggen
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    # Artefakt loggen (Datei)
    artifact_path = "/app/tests/dummy_artifact.txt"
    with open(artifact_path, "w") as f:
        f.write("This is a dummy artifact for testing MLflow.\n")
    
    mlflow.log_artifact(artifact_path, artifact_path="artifacts")
    
    print(f"Run completed and logged to experiment '{experiment_name}'")
