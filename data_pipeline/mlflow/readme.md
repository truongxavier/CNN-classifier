#on se met dans le répertoire mlflow
cd mlflow

#on créé le env dans le répertoire mlflow
python3 -m venv mlfow_env  

source mlfow_env/bin/activate

#attention de bien faire un env, j'ai du chercher un bon moment pour avoir les version mlfow numpy et tensorfow compatible
pip install -r requirements.txt

#lancer le serveur avant d'entrainer le modèle
mlflow server --host 0.0.0.0 --port 8080

#supprimer au cas ou on veut tous remettre à zéros, ici on ne le fait pas
rm -r mlruns

#on se remet dans le répetoire parent
cd ..

#lancer l'entrainement avec mlflow
#modifier le nom du run à la main ligne 147 run_name = "second_run"
#dans ce fichiers le chemin de base a été changer pour aller chercher sur data_pipeline alors qu'on lance de mlflow
#j'ai entrainé sur le val et validé sur le test
#python mlflow/src/CNN_retraining_mlflow.py 
python3 data_pipeline/mlflow/src/CNN_retraining_drift_mlflow.py

#visualiser les résultats sur 
http://localhost:8080/

#le mlruns fournis contient 2 runs, un sur 1 epochs, lautre sur 3 epochs

remonté des données de training dans prometheus
python prometheus_exporter.py



pour le drift monitoring, plutôt que evidentdly qui n'est pas adapté au deep learning 
Je recommanderais :

TensorFlow Data Validation si vous utilisez déjà TFX
Deepchecks si vous voulez des fonctionnalités spécifiques à la computer vision
Weights & Biases si vous voulez du monitoring en temps réel avec de belles visualisations
Alibi Detect si vous voulez des algorithmes plus sophistiqués de détection de drift

Ces outils offrent plusieurs avantages par rapport à Evidently pour le deep learning d'images :

Support natif des tenseurs
Méthodes spécialisées pour les images
Meilleure gestion de la mémoire
Visualisations adaptées
Intégration plus facile avec TensorFlow
 CopyRetryClaude does not have the ability to run the code it generates yet.Claude can make mistakes. Please double-check responses.