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

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import TestColumnDrift
from evidently.pipeline.column_mapping import ColumnMapping
import json
import requests


# Désactiver les GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# (Optionnel) Définir l'URI de tracking MLflow
#mlflow.set_tracking_uri("http://127.0.0.1:8080")
# Définir l'URI de tracking MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # URI du serveur MLflow

# Vérifier la connexion au serveur MLflow
try:
    response = requests.get("http://127.0.0.1:8080")
    if response.status_code == 200:
        print("Connexion au serveur MLflow réussie.")
    else:
        print(f"Le serveur MLflow a répondu avec un statut : {response.status_code}")
except Exception as e:
    raise RuntimeError(f"Impossible de se connecter au serveur MLflow : {e}")

#-------------------------------------------------------------------------------
# Fonctions de monitoring drift
#-------------------------------------------------------------------------------
def prepare_data_for_drift(dataset_tf):
    """
    Convertit un dataset TensorFlow en format compatible avec Evidently
    """
    print("Extraction des données...")
    
    iterator = iter(dataset_tf)
    images = []
    labels = []
    
    while True:
        try:
            batch = next(iterator)
            if isinstance(batch, tuple):
                image_batch, label_batch = batch
                # Convertir le batch complet en numpy d'abord
                image_batch_np = image_batch.numpy()
                label_batch_np = label_batch.numpy()
                
                # Ajouter chaque élément du batch individuellement
                for i in range(len(image_batch_np)):
                    images.append(image_batch_np[i])
                    labels.append(int(label_batch_np[i]))
        except StopIteration:
            break
    
    print(f"Traitement de {len(images)} images...")
    
    # Calculer des caractéristiques statistiques
    image_features = []
    for img in images:
        features = {
            'mean_intensity': float(np.mean(img)),
            'std_intensity': float(np.std(img)),
            'min_intensity': float(np.min(img)),
            'max_intensity': float(np.max(img)),
            'median_intensity': float(np.median(img)),
            'q1_intensity': float(np.percentile(img, 25)),
            'q3_intensity': float(np.percentile(img, 75))
        }
        image_features.append(features)
    
    # Créer le DataFrame
    df = pd.DataFrame(image_features)
    df['target'] = labels
    
    print(f"DataFrame créé avec {len(df)} échantillons")
    print("Types des colonnes:", df.dtypes)
    return df

def generate_drift_report(reference_data, current_data, output_dir):
    """
    Génère un rapport complet de drift avec Evidently
    """

    # Vérifier que les données sont bien converties
    reference_data['target'] = reference_data['target'].astype(int)
    current_data['target'] = current_data['target'].astype(int)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = f"{output_dir}/drift_report_{timestamp}"
    os.makedirs(report_path, exist_ok=True)
    
    column_mapping = ColumnMapping(
        target='target',
        numerical_features=[
            'mean_intensity', 'std_intensity', 'min_intensity',
            'max_intensity', 'median_intensity', 'q1_intensity',
            'q3_intensity'
        ]
    )
    
    # Rapport de Data Drift
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    data_drift_report.save_html(f"{report_path}/data_drift_report.html")
    
    # Rapport de Target Drift
    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    target_drift_report.save_html(f"{report_path}/target_drift_report.html")
    
    # Tests de Drift
    drift_tests = TestSuite(tests=[
        DataDriftTestPreset(),
        TestColumnDrift(column_name="target")
    ])
    drift_tests.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    drift_tests.save_html(f"{report_path}/drift_tests.html")
    
     # Extraire les métriques de manière sécurisée
    try:
        metrics = {
            'data_drift_detected': int(drift_tests.tests[0].get_result().passed),  # Convertir en 0 ou 1
            'target_drift_detected': int(drift_tests.tests[1].get_result().passed),  # Convertir en 0 ou 1
        }
        
        # Ajouter des métriques supplémentaires si disponibles
        test_results = drift_tests.tests[0].get_result()
        if hasattr(test_results, 'metrics'):
            drift_share = sum(1 for t in test_results.metrics.values() if not t.passed) / len(test_results.metrics)
            metrics['share_of_drifted_columns'] = float(drift_share)
        else:
            metrics['share_of_drifted_columns'] = 0.0  # Valeur par défaut
            
    except Exception as e:
        print(f"Attention: Erreur lors de l'extraction des métriques: {str(e)}")
        # Fournir des valeurs par défaut numériques
        metrics = {
            'data_drift_detected': 0,
            'target_drift_detected': 0,
            'share_of_drifted_columns': 0.0
        }
    
    # S'assurer que toutes les valeurs sont numériques
    for key in metrics:
        if isinstance(metrics[key], bool):
            metrics[key] = int(metrics[key])
        elif not isinstance(metrics[key], (int, float)):
            metrics[key] = 0.0
    
    print("Métriques de drift calculées:", metrics)
    return metrics, report_path

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
# Fonction pour construire le modèle VGG16 adapté aux images grayscale
#-------------------------------------------------------------------------------
def build_vgg16_grayscale(input_shape=(224, 224, 1), num_classes=16):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)
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
# Paramétrage de lancement
#-------------------------------------------------------------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "preprocessed"))

# Identifier les derniers datasets
#datasets de réference pour le drift
reference_dataset_path = get_latest_dataset_path(base_path, "split_0")
#dataset de validation
val_dataset_path = get_latest_dataset_path(base_path, "split_14")

# on entraine sur les split 02 à 14
train_dataset_path = get_latest_dataset_path(base_path, "split_3")


print(f"Dataset d'entraînement : {train_dataset_path}")
print(f"Dataset de validation : {val_dataset_path}")

# Charger les datasets
train_dataset_tf = tf.data.Dataset.load(train_dataset_path).prefetch(tf.data.AUTOTUNE)
val_dataset_tf = tf.data.Dataset.load(val_dataset_path).prefetch(tf.data.AUTOTUNE)
reference_dataset_tf = tf.data.Dataset.load(reference_dataset_path).prefetch(tf.data.AUTOTUNE)

# Configuration des chemins de sauvegarde
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(base_path, f"../models/model_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

saved_model_path = os.path.join(output_dir, "saved_modelcnn.keras")
history_path = os.path.join(output_dir, "training_CNN_history.npy")
log_dir = os.path.join(output_dir, "logs")

# Prétraitement des données
# def preprocess_for_training(image, label, image_ID):
#     return image, label

# train_dataset_tf = train_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
# val_dataset_tf = val_dataset_tf.map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)

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

# Recompiler le modèle
learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print(f"Logs TensorBoard enregistrés dans : {log_dir}")

batch_size = 16
epochs = 10
steps_per_epoch = train_dataset_tf.cardinality().numpy() // batch_size
validation_steps = val_dataset_tf.cardinality().numpy() // batch_size

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

#-------------------------------------------------------------------------------
# Préparation des données pour le drift monitoring
#-------------------------------------------------------------------------------
print("Préparation des données pour le monitoring de drift...")
reference_data = prepare_data_for_drift(reference_dataset_tf)
current_data = prepare_data_for_drift(train_dataset_tf)

#-------------------------------------------------------------------------------
# Tracking MLflow avec monitoring de drift
#-------------------------------------------------------------------------------


mlflow.set_experiment("fine_tuning_experiment")
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"mlflow_run_drift_{timestamp}" 

with mlflow.start_run(run_name=run_name) as run:
    # Log des hyperparamètres
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_loaded_from", loaded_model_path)
    mlflow.log_param("num_layers_unfrozen", num_layers_to_unfreeze)
    mlflow.log_param("training_dataset_path", train_dataset_path)
    mlflow.log_param("validation_dataset_path", val_dataset_path)
    mlflow.log_param("reference_dataset_path", reference_dataset_path)


    # Générer et logger les rapports de drift avant l'entraînement
    print("Génération des rapports de drift...")
    drift_metrics, report_path = generate_drift_report(
        reference_data, 
        current_data, 
        output_dir
    )
    
    # Logger les métriques de drift
    for metric_name, metric_value in drift_metrics.items():
    # Vérifier que la valeur est valide avant de la logger
        if metric_value is not None and isinstance(metric_value, (int, float, bool)):
            # Convertir les booléens en int pour MLflow si nécessaire
            if isinstance(metric_value, bool):
                metric_value = int(metric_value)
            mlflow.log_metric(f"initial_{metric_name}", metric_value)
        else:
            print(f"Attention : La métrique {metric_name} n'a pas pu être loggée (valeur : {metric_value})")
    
    # Logger les rapports de drift comme artéfacts
    mlflow.log_artifacts(report_path, "drift_reports")

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

    # Récupération des métriques finales
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