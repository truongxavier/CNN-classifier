# Utiliser l'image Python
FROM python:3.10-slim
# Définir le répertoire de travail
WORKDIR /app
# Copier le fichier requirements.txt spécifique à MLflow
COPY data_pipeline/mlflow/requirements.txt /app/requirements.txt
# Copier les scripts nécessaires
COPY data_pipeline/src/training/CNN_retraining_mlflow.py /app/data_pipeline/src/training/CNN_retraining_drift_mlflow.py
# Copier les datasets nécessaires
COPY data_pipeline/data/preprocessed /app/data_pipeline/data/preprocessed
# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt
# Créer un répertoire pour les artefacts MLflow
RUN mkdir -p /mlartifacts
# Définir le point d'entrée pour démarrer MLflow
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlartifacts", "--host", "0.0.0.0", "--port", "8080"]