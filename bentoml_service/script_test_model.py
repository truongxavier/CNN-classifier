from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    Prétraitement de l'image pour le modèle CNN.
    
    - Convertit en niveaux de gris.
    - Redimensionne à 224x224 pixels.
    - Normalise les pixels entre 0 et 1.
    - Ajuste la forme pour correspondre à l'entrée du modèle.
    
    Args:
        image (PIL.Image): Image d'entrée.
    
    Returns:
        np.ndarray: Image prétraitée sous forme de tableau NumPy.
    """
    # Convertir en niveaux de gris
    image = image.convert('L')

    # Redimensionner à 224x224
    image = image.resize((224, 224))

    # Convertir en tableau NumPy et normaliser
    image_array = np.array(image).reshape(1, 224, 224, 1).astype('float32') / 255.0

    return image_array

# Charger le modèle
model_path = "bentoml_service/src/models/saved_modelcnn.keras"
model = load_model(model_path)

image_paths = [
    "bentoml_service/data/raw/email.png",
    "bentoml_service/data/raw/form.png",
    "bentoml_service/data/raw/file_folder.png"
]

for image_path in image_paths:
    img = Image.open(image_path)
    img_preprocessed = preprocess_image(img)
    prediction = model.predict(img_preprocessed)
    print(f"Prediction for {image_path}: {prediction}")