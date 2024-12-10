# security/generate_token.py

import jwt
import datetime
import os
import argparse
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Charger la clé secrète et autres configurations
SECRET_KEY = os.getenv('SECRET_KEY', None)
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY manquant dans le fichier .env.")

TOKEN_EXPIRATION_HOURS = int(os.getenv('TOKEN_EXPIRATION_HOURS', 1))

def generate_token(username):
    """
    Génère un token JWT pour un utilisateur donné.
    
    Args:
        username (str): Le nom d'utilisateur pour lequel générer le token.
    
    Returns:
        str: Le token JWT signé.
    """
    payload = {
        'sub': username,
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRATION_HOURS)
    }
    try:
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération du token : {e}")
    return token

if __name__ == '__main__':
    username = 'utilisateur_test'  # Nom d'utilisateur par défaut
    token = generate_token(username)
    print(f'Token JWT pour {username}: {token}')
