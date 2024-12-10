import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime
import mlflow
import mlflow.keras
from mlflow import MlflowClient

# Désactiver les GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define tracking_uri
mlflow.set_tracking_uri("http://127.0.0.1:8080") 

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
# Paramétrage de lancement
#-------------------------------------------------------------------------------
# Chemin de base pour les datasets
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_pipeline", "data", "preprocessed"))

# Identifier les derniers datasets
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

#-------------------------------------------------------------------------------
# Construire le modèle VGG16 adapté aux images grayscale (224, 224, 1)
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

model = build_vgg16_grayscale()

# Compiler le modèle
initial_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print(f"Logs TensorBoard enregistrés dans : {log_dir}")

batch_size = 16
steps_per_epoch = train_dataset_tf.cardinality().numpy() // batch_size
validation_steps = val_dataset_tf.cardinality().numpy() // batch_size

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Prétraitement des données pour exclure image_ID
def preprocess_for_training(image, label, image_ID):
    # Exclure image_ID et ne garder que image et label
    return image, label

train_dataset_tf = train_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_tf = val_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)

#-------------------------------------------------------------------------------
# Tracking MLflow
#-------------------------------------------------------------------------------
mlflow.set_experiment("mon_experience_mlflow")  # Nom de l'expérience, modifiable
run_name = "first_beginning_run"
with mlflow.start_run(run_name=run_name) as run:
    # Log des hyperparamètres
    mlflow.log_param("initial_learning_rate", initial_learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", 2)
    mlflow.log_param("model", "VGG16_grayscale_transfer")

    # Entraîner le modèle
    history = model.fit(
        train_dataset_tf,
        validation_data=val_dataset_tf,
        epochs=2,
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

    # Récupération des métriques finales
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Log des métriques dans MLflow (dernière époque)
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
