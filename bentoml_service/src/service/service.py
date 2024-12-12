import bentoml
from bentoml.io import Image, JSON, Text
import numpy as np
import tensorflow as tf
from security.jwt_middleware import JWTAuthenticationMiddleware
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv
from src.preprocessing.cnn_preprocessor import preprocess_image
from prometheus_client import Counter, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import time

# Définir les métriques Prometheus
PREDICTION_REQUEST_COUNT = Counter(
    'document_prediction_requests_total',
    'Total number of prediction requests',
    ['status']
)

PREDICTION_LATENCY = Histogram(
    'document_prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

CLASS_PREDICTION_COUNT = Counter(
    'document_class_predictions_total',
    'Total predictions per document class',
    ['predicted_class']
)

CONFIDENCE_SCORES = Summary(
    'document_prediction_confidence',
    'Confidence scores of predictions'
)

TOKEN_GENERATION_COUNT = Counter(
    'document_token_generation_total',
    'Total number of JWT token generations',
    ['status']
)

# Configuration initiale
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", None)
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY manquant dans le fichier .env.")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.WARNING)
logging.getLogger("bentoml").setLevel(logging.ERROR)
logging.getLogger("bentoml.runners").setLevel(logging.ERROR)

# Charger le modèle
model_ref = bentoml.models.get('document_classifier_model:latest')

class KerasModelRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = tf.keras.models.load_model(model_ref.path)

    @bentoml.Runnable.method(batchable=False)
    def predict(self, input_data):
        return self.model.predict(input_data)

model_runner = bentoml.Runner(KerasModelRunnable)
svc = bentoml.Service("document_classifier_service", runners=[model_runner])

# Les endpoints publics sont définis avant le middleware JWT

@svc.api(input=JSON(), output=JSON())
async def status_check(data):
    """Endpoint pour vérifier la disponibilité du service."""
    return {"status": "ok"}

@svc.api(input=JSON(), output=Text(), route="/custom_metrics")
async def custom_metrics(data: dict = None):
    """Endpoint pour exposer les métriques personnalisées."""
    metrics_data = generate_latest()
    return metrics_data.decode("utf-8")

# Ajout du middleware JWT après les endpoints publics et avant les endpoints sécurisés
svc.add_asgi_middleware(JWTAuthenticationMiddleware)

@svc.api(input=JSON(), output=JSON(), route="/generate_token")
async def generate_token(data):
    try:
        username = data.get("username")
        if not username:
            TOKEN_GENERATION_COUNT.labels(status="error").inc()
            return JSONResponse(
                {"error": "Le nom d'utilisateur est obligatoire."},
                status_code=400
            )
        
        payload = {
            "sub": username,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        TOKEN_GENERATION_COUNT.labels(status="success").inc()
        return {"token": token}
    except Exception as e:
        TOKEN_GENERATION_COUNT.labels(status="error").inc()
        logging.error(f"Erreur lors de la génération du token : {e}")
        return JSONResponse(
            {"error": "Erreur interne lors de la génération du token."},
            status_code=500
        )

@svc.api(input=Image(), output=JSON())
async def predict(image):
    start_time = time.time()
    try:
        image_array = preprocess_image(image)
        prediction = await model_runner.predict.async_run(image_array)
        
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        class_mapping = {0: 'Lettre', 1: 'Formulaire', 2: 'Email', 3: 'Manuscrit', 
                        4: 'Publicité', 5: 'Rapport Scientifique',
                        6: 'Publication Scientifique', 7: 'Spécification', 
                        8: 'Dossier', 9: 'Article de Presse',
                        10: 'Budget', 11: 'Facture', 12: 'Présentation', 
                        13: 'Questionnaire', 14: 'CV', 15: 'Mémo'}
        predicted_class_name = class_mapping.get(predicted_class, 'Classe inconnue')

        # Enregistrer les métriques
        PREDICTION_REQUEST_COUNT.labels(status="success").inc()
        CLASS_PREDICTION_COUNT.labels(predicted_class=predicted_class_name).inc()
        CONFIDENCE_SCORES.observe(confidence)
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return {'predicted_class': predicted_class_name, 'confidence': confidence}
    except Exception as e:
        PREDICTION_REQUEST_COUNT.labels(status="error").inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        logging.error(f"Erreur dans la prédiction : {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}