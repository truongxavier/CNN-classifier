import numpy as np
from PIL import Image
import tensorflow as tf

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
    # # Convertir en niveaux de gris
    # image = image.convert('L')

    # # Redimensionner à 224x224
    # image = image.resize((224, 224))

    # # Convertir en tableau NumPy et normaliser
    # image_array = np.array(image).reshape(1, 224, 224, 1).astype('float32') / 255.0

    image = image.convert('L')  # Convertir en niveaux de gris
    image = image.resize((224, 224))  # Redimensionner l'image à la taille requise
    image_tensor = np.expand_dims(np.array(image), axis=-1)  # Ajouter une dimension pour (224, 224, 1)
    image_tensor = np.expand_dims(image_tensor, axis=0)  # Ajouter une autre dimension pour batch (1, 224, 224, 1)
    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)

    return image_tensor
