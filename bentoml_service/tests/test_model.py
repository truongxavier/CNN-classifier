import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Déterminez le chemin absolu du modèle
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../src/models/saved_modelcnn.keras")

def test_model_loading():
    """
    Test pour vérifier que le modèle peut être chargé correctement.
    """
    # Vérifiez si le fichier existe
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Tentez de charger le modèle
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model is not None, "Model loading failed"
    
def test_model_structure():
    """
    Test pour vérifier la structure du modèle (entrée, sortie, couches).
    """
    # Charger le modèle
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Vérifiez les couches du modèle
    # Vérifiez que le modèle a des couches
    assert len(model.layers) > 0, "The model has no layers"
    # Vérifiez la forme des entrées
    assert model.input_shape == (None, 224, 224, 1), f"Unexpected input shape: {model.input_shape}"
    # Vérifiez la forme des sorties
    assert model.output_shape == (None, 16), f"Unexpected output shape: {model.output_shape}"
