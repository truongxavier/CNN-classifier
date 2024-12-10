import os
import tensorflow as tf
import numpy as np
from PIL import Image

def test_model_prediction():
    # Chemins
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../src/models/saved_modelcnn.keras")
    image_path = os.path.join(current_dir, "../data/raw/__results___6_0.png")

    # Vérifications des fichiers
    assert os.path.exists(model_path), "Modèle introuvable"
    assert os.path.exists(image_path), "Image introuvable"

    # Charger le modèle
    model = tf.keras.models.load_model(model_path, compile=False)

    # Prétraiter l'image
    image = Image.open(image_path).convert('L').resize((224, 224))
    image_array = np.array(image).reshape(1, 224, 224, 1).astype('float32') / 255.0

    # Effectuer la prédiction
    prediction = model.predict(image_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    # Vérifications
    assert prediction is not None, "La prédiction a échoué"
    assert prediction.shape[1] == 16, "Le modèle doit prédire 16 classes"
    assert 0 <= predicted_class < 16, "Classe prédite invalide"
