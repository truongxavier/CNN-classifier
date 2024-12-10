# security/auth.py

import jwt
from bentoml import HTTPRequest, HTTPResponse
from bentoml.exceptions import BentoMLException
import os
from dotenv import load_dotenv

load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

SECRET_KEY = os.getenv('SECRET_KEY')

def jwt_required(func):
    """
    Décorateur pour vérifier le JWT dans les en-têtes de la requête.
    """
    def wrapper(request: HTTPRequest):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise BentoMLException("Missing or invalid Authorization header")

        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.context['user'] = payload
        except jwt.ExpiredSignatureError:
            raise BentoMLException("Token expired")
        except jwt.InvalidTokenError:
            raise BentoMLException("Invalid token")

        return func(request)
    return wrapper
