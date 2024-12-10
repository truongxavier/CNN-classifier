import pytest
import requests
from pathlib import Path
import time

# URL de l'API
API_URL = "http://127.0.0.1:3000/predict"

# Token JWT pour les tests
AUTH_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1dGlsaXNhdGV1cl90ZXN0IiwiaWF0IjoxNzMzMjk1ODQ5LCJleHAiOjE3MzMyOTk0NDl9.KA_JVzBbOgHHHjOhTUd1MQUMBG9rF6IPDWqMEo11UcM"
)

# Chemins des images de test
IMAGE_PATHS = [
    Path("data/raw/__results___6_0.png"),  # Image valide
]

# Codes de statut attendus
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401


@pytest.fixture(scope="session", autouse=True)
def wait_for_server():
    """Attendre que le serveur soit prêt."""
    time.sleep(5)  # Attendre 5 secondes


@pytest.fixture(scope="session", autouse=True)
def check_api_availability():
    """Vérifiez que l'API est disponible avant les tests."""
    max_retries = 10
    for _ in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:3000/health")
            if response.status_code == 200:
                print("API disponible")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    pytest.fail("L'API n'est pas disponible après plusieurs tentatives.")


@pytest.fixture
def api_headers():
    """Fixture pour les en-têtes de requête."""
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def log_request_and_response(response):
    """Log des détails de la requête et de la réponse."""
    print(f"Requête : {response.request.method} {response.request.url}")
    print(f"En-têtes : {response.request.headers}")
    print(f"Réponse : {response.status_code} - {response.text}")


def validate_json_response(response, expected_status=HTTP_OK):
    """Valider une réponse JSON."""
    assert response.status_code == expected_status, (
        f"Statut inattendu : {response.status_code}, Contenu : {response.content}"
    )
    try:
        return response.json()
    except ValueError:
        pytest.fail(f"La réponse n'est pas au format JSON : {response.content}")


@pytest.mark.parametrize("image_path", IMAGE_PATHS)
def test_predict_endpoint(api_headers, image_path):
    """Tester l'endpoint /predict avec plusieurs images."""
    if not image_path.exists():
        pytest.fail(f"L'image de test {image_path} n'existe pas.")

    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(API_URL, headers=api_headers, files=files)
        log_request_and_response(response)

        if image_path.suffix in [".png", ".jpg", ".jpeg"]:
            json_response = validate_json_response(response)
            assert "predicted_class" in json_response
            assert "confidence" in json_response
            assert 0.0 <= json_response["confidence"] <= 1.0
            print(
                f"Test réussi : Image = {image_path.name}, "
                f"Classe prédite = {json_response['predicted_class']}, "
                f"Confiance = {json_response['confidence']}"
            )
        else:
            assert response.status_code == HTTP_BAD_REQUEST


def test_health_check():
    """Tester la disponibilité de l'API."""
    response = requests.get("http://127.0.0.1:3000/health")
    log_request_and_response(response)
    assert response.status_code == HTTP_OK


def test_auth_check(api_headers):
    """Tester l'authentification avec un token valide."""
    response = requests.get("http://127.0.0.1:3000/auth_check", headers=api_headers)
    log_request_and_response(response)
    assert response.status_code == HTTP_OK


def test_auth_check_invalid_token():
    """Tester l'authentification avec un token invalide."""
    headers = {"Authorization": "Bearer token_invalide"}
    response = requests.get("http://127.0.0.1:3000/auth_check", headers=headers)
    log_request_and_response(response)
    assert response.status_code == HTTP_UNAUTHORIZED


def test_auth_check_missing_token():
    """Tester l'authentification sans token."""
    response = requests.get("http://127.0.0.1:3000/auth_check")
    log_request_and_response(response)
    assert response.status_code == HTTP_UNAUTHORIZED


def test_missing_image(api_headers):
    """Tester l'endpoint sans image."""
    response = requests.post(API_URL, headers=api_headers, files={})
    log_request_and_response(response)
    assert response.status_code == HTTP_BAD_REQUEST


def test_corrupted_image(api_headers):
    """Tester l'API avec une image corrompue."""
    corrupted_image = Path("models/corrupted_image.png")
    if not corrupted_image.exists():
        pytest.skip("L'image corrompue est absente, test sauté.")

    with open(corrupted_image, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(API_URL, headers=api_headers, files=files)
        log_request_and_response(response)
        assert response.status_code == HTTP_BAD_REQUEST
