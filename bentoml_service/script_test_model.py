import tensorflow as tf
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

# Charger le modèle
model_path = "bentoml_service/src/models/saved_modelcnn.keras"
model = load_model(model_path)

# Recompiler le modèle
learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction) * 100
    print(f"predicted_class for {image_path}: {predicted_class}")
    print(f"confidence_score for {image_path}: {confidence_score}")