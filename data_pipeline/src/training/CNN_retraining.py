import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import mlflow
from mlflow.tracking import MlflowClient

import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Fine-tuning CNN with MLflow")
parser.add_argument("--model_index", type=int, required=True, help="Index of the model to fine-tune")
parser.add_argument("--unfreeze_layers", type=int, required=True, help="Number of layers to unfreeze for fine-tuning")
args = parser.parse_args()

# Désactiver les GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Définir l'URI de tracking MLflow
mlflow.set_tracking_uri("http://mlflow_server:9090")  # URI du serveur MLflow

# Vérifier la connexion au serveur MLflow
try:
    response = requests.get("http://mlflow_server:9090")
    if response.status_code == 200:
        print("Connexion au serveur MLflow réussie.")
    else:
        print(f"Le serveur MLflow a répondu avec un statut : {response.status_code}")
except Exception as e:
    raise RuntimeError(f"Impossible de se connecter au serveur MLflow : {e}")


#-------------------------------------------------------------------------------
# Déterminer dynamiquement `base_path`
#-------------------------------------------------------------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "preprocessed"))

#-------------------------------------------------------------------------------
# Fonction pour récupérer le dernier dataset basé sur le timestamp
#-------------------------------------------------------------------------------
def get_latest_dataset_path(base_path, dataset_prefix):
    datasets = [d for d in os.listdir(base_path) if d.startswith(dataset_prefix)]
    datasets.sort(reverse=True)  # Trier par ordre décroissant de date
    if datasets:
        return os.path.join(base_path, datasets[0])  # Retourner le plus récent
    else:
        raise FileNotFoundError(f"Aucun dataset trouvé avec le préfixe '{dataset_prefix}' dans {base_path}")

#-------------------------------------------------------------------------------
# Fonction pour construire le modèle VGG16 adapté aux images grayscale (224, 224, 1)
#-------------------------------------------------------------------------------
def build_vgg16_grayscale(input_shape=(224, 224, 1), num_classes=16):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)  # Grayscale vers pseudo-RGB
    x = base_model(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

#-------------------------------------------------------------------------------
# Fonction pour lister les modèles disponibles
#-------------------------------------------------------------------------------
def list_available_models(models_dir):
    models = []
    for dir_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, dir_name, "saved_modelcnn.keras")
        if os.path.exists(model_path):
            models.append(model_path)
    return models

#-------------------------------------------------------------------------------
# Identifier les derniers datasets
#-------------------------------------------------------------------------------
train_dataset_path = os.path.abspath(os.path.join(base_path, "split_2"))  # Utilisé comme train
val_dataset_path = os.path.abspath(os.path.join(base_path, "split_1"))  # Utilisé comme validation

print(f"Dataset d'entraînement : {train_dataset_path}")
print(f"Dataset de validation : {val_dataset_path}")

# Charger les datasets
train_dataset_tf = tf.data.Dataset.load(train_dataset_path).prefetch(tf.data.AUTOTUNE)
val_dataset_tf = tf.data.Dataset.load(val_dataset_path).prefetch(tf.data.AUTOTUNE)

# Configuration des chemins de sauvegarde avec un tag de date
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(base_path, f"../models/model_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

saved_model_path = os.path.join(output_dir, "saved_modelcnn.keras")
history_path = os.path.join(output_dir, "training_CNN_history.npy")
log_dir = os.path.join(output_dir, "logs")

# Prétraitement des données pour exclure image_ID
def preprocess_for_training(image, label):
    return image, label

train_dataset_tf = train_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_tf = val_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)

# Utilisation de cache() et prefetch() pour réduire 
train_dataset_tf = train_dataset_tf.cache().prefetch(tf.data.AUTOTUNE)
val_dataset_tf = val_dataset_tf.cache().prefetch(tf.data.AUTOTUNE)

#-------------------------------------------------------------------------------
# Charger un modèle existant pour le fine-tuning
#-------------------------------------------------------------------------------
models_dir = os.path.abspath(os.path.join(base_path, "..", "models"))
available_models = list_available_models(models_dir)

if not available_models:
    raise FileNotFoundError(f"Aucun modèle trouvé dans le répertoire : {models_dir}")

print("Modèles disponibles pour le fine-tuning :")
for idx, model_path in enumerate(available_models):
    print(f"{idx + 1}. {model_path}")

selected_model_idx = args.model_index - 1
loaded_model_path = available_models[selected_model_idx]

print(f"Chargement du modèle : {loaded_model_path}")
model = tf.keras.models.load_model(loaded_model_path)

# Nombre de couches à dégeler
num_layers_to_unfreeze = args.unfreeze_layers

# Geler toutes les couches
for layer in model.layers:
    layer.trainable = False

# Dégeler les dernières couches spécifiées
if num_layers_to_unfreeze > 0:
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

# Recompiler le modèle après avoir modifié trainable
learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 8
epochs = 3
steps_per_epoch = train_dataset_tf.cardinality().numpy() // batch_size
validation_steps = val_dataset_tf.cardinality().numpy() // batch_size

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

#-------------------------------------------------------------------------------
# Tracking MLflow
#-------------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"mlflow_run_{timestamp}" 

mlflow.set_experiment("fine_tuning_experiment")
with mlflow.start_run(run_name=run_name) as run:
    # Log des hyperparamètres
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_loaded_from", loaded_model_path)
    mlflow.log_param("num_layers_unfrozen", num_layers_to_unfreeze)

    # Entraîner le modèle
    history = model.fit(
        train_dataset_tf,
        validation_data=val_dataset_tf,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback, early_stopping]
    )

    # Sauvegarder le modèle
    model.save(saved_model_path)
    print(f"Modèle sauvegardé à : {saved_model_path}")

    # Sauvegarder l'historique
    np.save(history_path, history.history)
    print(f"Historique sauvegardé à : {history_path}")

    # Log du modèle avec MLflow
    mlflow.keras.log_model(model, artifact_path="model")

    # Récupération des métriques finales (dernière époque)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    mlflow.log_metric("final_accuracy", acc[-1])
    mlflow.log_metric("final_val_accuracy", val_acc[-1])
    mlflow.log_metric("final_loss", loss[-1])
    mlflow.log_metric("final_val_loss", val_loss[-1])

    # Visualisation des résultats
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    figure_path = os.path.join(output_dir, "training_results.png")
    plt.savefig(figure_path)
    plt.show()

    # Log de l'historique et de la figure en artefacts
    mlflow.log_artifact(history_path)
    mlflow.log_artifact(figure_path)

mlflow.end_run()