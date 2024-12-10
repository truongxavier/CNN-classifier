import bentoml
from bentoml.io import Image, JSON
import numpy as np
import tensorflow as tf
from security.jwt_middleware import JWTAuthenticationMiddleware
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv
from src.preprocessing.cnn_preprocessor import preprocess_image

# Charger les variables d'environnement
load_dotenv()

# Charger la clé secrète
SECRET_KEY = os.getenv("SECRET_KEY", None)
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY manquant dans le fichier .env.")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configurer le niveau de journalisation pour limiter les logs
logging.basicConfig(level=logging.WARNING)  # Options : DEBUG, INFO, WARNING, ERROR, CRITICAL
# Réduire les logs spécifiques à BentoML
logging.getLogger("bentoml").setLevel(logging.ERROR)
logging.getLogger("bentoml.runners").setLevel(logging.ERROR)

# Charger le modèle avec bentoml.models
model_ref = bentoml.models.get('document_classifier_model:latest')

# Définir un Runnable personnalisé
class KerasModelRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = tf.keras.models.load_model(model_ref.path)

    @bentoml.Runnable.method(batchable=False)
    def predict(self, input_data):
        return self.model.predict(input_data)

# Créer le runner à partir du Runnable personnalisé
model_runner = bentoml.Runner(KerasModelRunnable)

# Définir le service BentoML
svc = bentoml.Service("document_classifier_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON(), route="/generate_token")
async def generate_token(data):
    """
    Endpoint pour générer un token JWT.
    """
    try:
        username = data.get("username")
        if not username:
            return JSONResponse(
                {"error": "Le nom d'utilisateur est obligatoire."},
                status_code=400
            )
        
        payload = {
            "sub": username,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),  # Expiration après 1 heure
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token}
    except Exception as e:
        logging.error(f"Erreur lors de la génération du token : {e}")
        return JSONResponse(
            {"error": "Erreur interne lors de la génération du token."},
            status_code=500
        )

# Ajouter le middleware JWT
svc.add_asgi_middleware(JWTAuthenticationMiddleware)

# Définir le endpoint principal pour les prédictions
@svc.api(input=Image(), output=JSON())
async def predict(image):
    try:
        # Prétraitement de l'image
        image_array = preprocess_image(image)

        # Prédiction avec le modèle
        prediction = await model_runner.predict.async_run(image_array)

        # Post-traitement
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        class_mapping = {0: 'Lettre', 1: 'Formulaire', 2: 'Email', 3: 'Manuscrit', 4: 'Publicité', 5: 'Rapport Scientifique',
                         6: 'Publication Scientifique', 7: 'Spécification', 8: 'Dossier', 9: 'Article de Presse',
                         10: 'Budget', 11: 'Facture', 12: 'Présentation', 13: 'Questionnaire', 14: 'CV', 15: 'Mémo'}
        predicted_class_name = class_mapping.get(predicted_class, 'Classe inconnue')

        logging.info(f"Prédiction : classe={predicted_class_name}, confiance={confidence}")
        return {'predicted_class': predicted_class_name, 'confidence': confidence}
    except Exception as e:
        logging.error(f"Erreur dans la prédiction : {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Ajouter un endpoint de santé publique pour surveiller la disponibilité du service
@svc.api(input=JSON(), output=JSON())
async def status_check(data):
    """Endpoint pour vérifier la disponibilité du service."""
    return {"status": "ok"}

