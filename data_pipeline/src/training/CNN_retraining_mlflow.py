import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime

import mlflow
from mlflow.tracking import MlflowClient

# Désactiver les GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Définir l'URI de tracking MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")  # URI du serveur MLflow

#-------------------------------------------------------------------------------
# Déterminer dynamiquement `base_path`
#-------------------------------------------------------------------------------
current_dir = os.path.basename(os.getcwd())
if current_dir == "mlflow":
    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data_pipeline", "data", "preprocessed"))
elif current_dir == "data_pipeline":
    base_path = os.path.abspath(os.path.join(os.getcwd(), "data", "preprocessed"))
else:
    raise RuntimeError("Le script doit être lancé depuis 'mlflow' ou 'data_pipeline'.")

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
train_dataset_path = get_latest_dataset_path(base_path, "val_tf_dataset_ID")  # Utilisé comme train
val_dataset_path = get_latest_dataset_path(base_path, "test_tf_dataset_ID")    

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
def preprocess_for_training(image, label, image_ID):
    return image, label

train_dataset_tf = train_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_tf = val_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)

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

selected_model_idx = int(input("Entrez le numéro du modèle à charger : ")) - 1
loaded_model_path = available_models[selected_model_idx]

print(f"Chargement du modèle : {loaded_model_path}")
model = tf.keras.models.load_model(loaded_model_path)

# Nombre de couches à dégeler
num_layers_to_unfreeze = int(input("Combien de couches voulez-vous dégeler pour le fine-tuning ? "))

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

batch_size = 16
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
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_loaded_from", loaded_model_path)
    mlflow.log_param("num_layers_unfrozen", num_layers_to_unfreeze)

    history = model.fit(
        train_dataset_tf,
        validation_data=val_dataset_tf,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback, early_stopping]
    )

    model.save(saved_model_path)
    np.save(history_path, history.history)

    mlflow.keras.log_model(model, artifact_path="model")

mlflow.end_run()
