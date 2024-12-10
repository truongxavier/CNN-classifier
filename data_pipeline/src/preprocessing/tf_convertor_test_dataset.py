import tensorflow as tf
from datasets import load_from_disk
from datetime import datetime
from tqdm import tqdm
import os

#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------

# Récupérer le chemin absolu du répertoire contenant ce script
# Base path défini au niveau de la racine 'data_pipeline'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Chemins relatifs vers les datasets locaux et le dossier de sauvegarde
input_path = os.path.join(base_path, "data/raw")
output_path = os.path.join(base_path, "data/preprocessed")

# Batch spécifique à convertir
batch_to_convert = "test_dataset_ID_resize"

# Assurer que le dossier de sauvegarde parent existe
os.makedirs(output_path, exist_ok=True)

# Ajouter un tag de date au dossier de sortie
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_batch_name = f"test_tf_dataset_ID_{timestamp}"
output_dataset_path = os.path.join(output_path, output_batch_name)

# Créer le dossier de sortie spécifique (si nécessaire)
os.makedirs(output_dataset_path, exist_ok=True)

# Charger le dataset
dataset_path = os.path.join(input_path, batch_to_convert)
dataset = load_from_disk(dataset_path)

#-------------------------------------------------------------------------------
# Affichage des informations
#-------------------------------------------------------------------------------
print(f"Dataset chargé depuis : {dataset_path}")
print(f"Le dataset sera sauvegardé dans : {output_dataset_path}")
print(f"Nombre d'images dans le dataset : {len(dataset)}")

#-------------------------------------------------------------------------------
# Convertir en tf.data.Dataset tout en incluant les image_ID
#-------------------------------------------------------------------------------
def to_tf_dataset(dataset, batch_size=32):
    def generator():
        for example in tqdm(dataset, total=len(dataset)):
            image = tf.expand_dims(example['image'], axis=-1)  # Ajouter une dimension de canal pour grayscale
            yield image, example['label'], example['image_ID']

    dataset_tf = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),  # Image grayscale avec un canal
            tf.TensorSpec(shape=(), dtype=tf.int64),  # Label
            tf.TensorSpec(shape=(), dtype=tf.string)  # Image ID
        )
    )

    # Batching et shuffling
    dataset_tf = dataset_tf.batch(batch_size).shuffle(buffer_size=1000)

    # Préchargement des données en avance pour optimiser les performances
    dataset_tf = dataset_tf.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_tf

# Générer le tf.data.Dataset
batch_size = 32
test_dataset_tf = to_tf_dataset(dataset, batch_size=batch_size)

#-------------------------------------------------------------------------------
# Prétraitement des images
#-------------------------------------------------------------------------------
def preprocess_images(image, label, image_ID):
    # Normalisation des images
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label, image_ID

# Application du prétraitement au dataset
test_dataset_tf = test_dataset_tf.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)

#-------------------------------------------------------------------------------
# Sauvegarde du dataset TensorFlow
#-------------------------------------------------------------------------------
test_dataset_save_path = output_dataset_path

test_dataset_tf.save(test_dataset_save_path)

print(f"Le dataset TensorFlow a été sauvegardé dans : {test_dataset_save_path}")
