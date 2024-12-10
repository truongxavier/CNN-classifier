import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
from prometheus_client import start_http_server, Summary, Gauge

# désactivation GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# problème d'incompatibilité avec XLA (Accelerated Linear Algebra), une bibliothèque 
# d'optimisation utilisée par TensorFlow pour compiler et exécuter efficacement les graphiques
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# -------------------------------------------------------------------------------
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
# Charger un modèle existant
#-------------------------------------------------------------------------------
models_dir = os.path.abspath(os.path.join("data", "models"))
available_models = list_available_models(models_dir)

if not available_models:
    raise FileNotFoundError(f"Aucun modèle trouvé dans le répertoire : {models_dir}")

print("Modèles disponibles pour l'évaluation :")
for idx, model_path in enumerate(available_models):
    print(f"{idx + 1}. {model_path}")

selected_model_idx = int(input("Entrez le numéro du modèle à charger : ")) - 1
model_path = available_models[selected_model_idx]

print(f"Chargement du modèle : {model_path}")
model = tf.keras.models.load_model(model_path)

#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------
test_dataset_tf_path = os.path.abspath(os.path.join("data", "preprocessed", "test_tf_dataset_ID_20241205-174750"))
output_file_path = os.path.abspath(os.path.join("data", "evaluation", "classification_vgg16_report.json"))

# Configurer un serveur Prometheus pour les métriques
start_http_server(8000)

# Métriques Prometheus
evaluation_duration = Summary("cnn_model_evaluation_duration_seconds", "Durée de l'évaluation du modèle")
test_accuracy_metric = Gauge("cnn_test_accuracy", "Précision du modèle sur le jeu de test")
test_loss_metric = Gauge("cnn_test_loss", "Perte du modèle sur le jeu de test")

#-------------------------------------------------------------------------------
# Charger le dataset TensorFlow de test
#-------------------------------------------------------------------------------
print("Chargement du dataset de test...")
test_dataset_tf = tf.data.Dataset.load(test_dataset_tf_path)

### problème avec les ID, mais ici on n'en pas besoin pour le CNN seul
# Prétraitement des données pour exclure image_ID
def preprocess_test_data(image, label, image_ID):
    # Ignore l'image_ID
    return image, label

test_dataset_tf = test_dataset_tf.map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
#-------------------------------------------------------------------------------
# Évaluation du modèle
#-------------------------------------------------------------------------------
@evaluation_duration.time()
def evaluate_model():
    print("Évaluation du modèle...")
    test_loss, test_acc = model.evaluate(test_dataset_tf)
    test_accuracy_metric.set(test_acc)
    test_loss_metric.set(test_loss)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    return test_loss, test_acc

test_loss, test_acc = evaluate_model()

#-------------------------------------------------------------------------------
# Prédictions
#-------------------------------------------------------------------------------
# Prédictions sur le dataset de test
print("Prédictions sur le dataset de test...")
y_true = []  # Pour stocker les vraies étiquettes
y_pred_probs = []  # Pour stocker les probabilités prédites pour chaque classe

for images, labels in test_dataset_tf:  # Le dataset contient uniquement `images` et `labels`
    y_true.extend(labels.numpy())  # Sauvegarder les vraies étiquettes
    y_pred_probs.extend(model.predict(images))  # Sauvegarder les probabilités prédites

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

#-------------------------------------------------------------------------------
# Calcul des métriques
#-------------------------------------------------------------------------------
n_classes = y_pred_probs.shape[1]
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

# Calcul des courbes ROC et AUC
fprs, tprs, aucs = {}, {}, {}
for i in range(n_classes):
    fprs[i], tprs[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    aucs[i] = auc(fprs[i], tprs[i])

# Matrice de confusion
y_pred_classes = np.argmax(y_pred_probs, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Rapport de classification
class_labels = [f"Class {i}" for i in range(n_classes)]
report_dict = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)

# Sauvegarder le rapport en JSON
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with open(output_file_path, "w") as f:
    json.dump(report_dict, f, indent=4)

print(f"Classification report saved to {output_file_path}")

#-------------------------------------------------------------------------------
# Visualisations
#-------------------------------------------------------------------------------
# Création d'un dossier pour les résultats
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join("evaluation_results", timestamp)
os.makedirs(results_dir, exist_ok=True)

# Matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# Courbes ROC
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fprs[i], tprs[i], label=f"Class {i} (AUC = {aucs[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "roc_curves.png"))
plt.close()

# Résumé des métriques globales
summary_report = {
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "f1_score": report_dict["accuracy"],
    "num_classes": n_classes,
}
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(summary_report, f, indent=4)

# Rapport détaillé
with open(os.path.join(results_dir, "report.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

print(f"Résultats sauvegardés dans : {results_dir}")
