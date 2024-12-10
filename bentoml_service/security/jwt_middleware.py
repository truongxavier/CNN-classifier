# security/jwt_middleware.py

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')  # Clé secrète depuis le fichier .env
LOG_FILE = os.getenv("LOG_FILE", "access_logs.log")  # Chemin du fichier de logs

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY manquant. Assurez-vous qu'il est défini dans votre fichier .env.")

# Configurer le logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(message)s",
)

class JWTAuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        log_details = {
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if request.url.path in ['/docs', '/openapi.json', '/healthz', '/generate_token', '/status_check']:
            return await call_next(request)

        auth_header = request.headers.get('Authorization')
        if auth_header is None:
            log_details.update({
                "status": "failed",
                "reason": "Authorization header missing"
            })
            logging.info(json.dumps(log_details))
            return JSONResponse({'detail': 'Authorization header missing'}, status_code=401)

        try:
            scheme, token = auth_header.strip().split(' ')
            if scheme.lower() != 'bearer':
                raise ValueError('Invalid authentication scheme')

            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.state.user = payload  # Stocker les informations utilisateur
            log_details.update({
                "status": "success",
                "user": payload["sub"]  # Ajouter l'utilisateur au log
            })
            logging.info(json.dumps(log_details))
        except jwt.ExpiredSignatureError:
            log_details.update({
                "status": "failed",
                "reason": "Token expired"
            })
            logging.info(json.dumps(log_details))
            return JSONResponse({'detail': 'Token expired'}, status_code=401)
        except jwt.InvalidTokenError:
            log_details.update({
                "status": "failed",
                "reason": "Invalid token"
            })
            logging.info(json.dumps(log_details))
            return JSONResponse({'detail': 'Invalid token'}, status_code=401)
        except Exception as e:
            log_details.update({
                "status": "failed",
                "reason": str(e)
            })
            logging.info(json.dumps(log_details))
            return JSONResponse({'detail': 'Authentication error'}, status_code=401)

        return await call_next(request)
