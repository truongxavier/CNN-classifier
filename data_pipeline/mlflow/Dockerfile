# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Installer curl pour le healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY mlflow/requirements.txt /app/requirements.txt

# Copier les scripts nécessaires
COPY src/training/CNN_retraining_mlflow.py /app/data_pipeline/src/training/CNN_retraining_drift_mlflow.py

# Copier les datasets nécessaires
COPY data/preprocessed /app/data_pipeline/data/preprocessed

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Définir le point d'entrée pour démarrer MLflow
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "8080"]