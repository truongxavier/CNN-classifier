
import streamlit as st
from PIL import Image
import os
import requests
import json
import shutil
import tempfile
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------

image_path = "images"
name_app ='DocuMancer'
couleur_fond = '#2ED4DA'
couleur_police = '#382DD5'
#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# CSS
#-------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Styles pour le bouton de téléchargement */
    .stDownloadButton > button, .stDownloadButton > button * {
        background-color: #2ED4DA;
        color: #382DD5;
        border: none;
        padding: 2px 6px !importants;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 15px;
        font-family: Arial, sans-serif; /* Ajout de la police */
        font-weight: bold !important;
    }

    /* Styles pour les boutons LinkedIn */
    a.linkedin-button {
        display: inline-block; /* Changement de 'block' à 'inline-block' */
        background-color: #2ED4DA !important;
        color: #382DD5 !important;
        font-weight: bold;
        padding: 2px 6px;
        text-decoration: none !important;
        border: none;
        border-radius: 15px;
        text-align: center;
        margin: 10px auto;
        width: 200px;
    }

    /* Appliquer les styles aux différents états du lien */
    a.linkedin-button:link,
    a.linkedin-button:visited,
    a.linkedin-button:hover,
    a.linkedin-button:active {
        background-color: #2ED4DA !important;
        color: #382DD5 !important;
        text-decoration: none !important;
    }

    /* Styles pour la liste et les éléments de liste */
    ul.linkedin-list {
        list-style-type: none;
        padding: 0; /* Supprimer le padding par défaut */
        margin: 0;  /* Supprimer la marge par défaut */
        text-align: center; /* Centrer le contenu de la liste */
    }

    ul.linkedin-list li {
        display: block;
        margin-bottom: 2px; /* Ajouter de l'espace entre les boutons */
    }

    /* Justifier le texte des paragraphes */
    .content p, .content li {
        text-align: justify;
    }

    /* Styles pour le bloc de prédiction */
    .prediction-box {
        background-color: #2ED4DA;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .prediction-box h3 {
        margin: 0;
        color: #382DD5;
        text-align: center;
    }

    .prediction-box h3 span {
        font-weight: bold;
    }

    .prediction-box p {
        font-size: 18px;
        margin: 5px 0;
        color: #382DD5;
        text-align: center;
    }

    .prediction-box p span {
        font-weight: bold;
    }

    /* Centrer l'image dans la fenêtre modale en plein écran */
    div[role="dialog"] .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    </style>
    """, unsafe_allow_html=True)

# Icône de baguette magique (vous pouvez choisir n'importe quelle icône de Font Awesome ou une autre bibliothèque d'icônes)
icon = "🪄"  # Ici c'est l'emoji "baguette magique", mais on pourrait aussi utiliser une icône Font Awesome

# Créer le titre avec style
title_html = f"""
    <h1 style='text-align: center; color: #2ED4DA;'>{name_app} {icon}</h1>
"""
#-------------------------------------------------------------------------------
# CSS
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# méthode principale
#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Sidebar
    #---------------------------------------------------------------------------
    image_datascientest = Image.open(image_path + '/datascientest.png')
    st.sidebar.image(image_datascientest, use_container_width=True)
    st.sidebar.markdown(title_html, unsafe_allow_html=True)

    page = st.sidebar.radio('Aller à', [
        'Introduction au Projet',
        'Architecture du projet',
        'Les services exposés',
        'Entrainons le modèle',
        'Prédiction',
        'Conclusion & perspectives'
    ])
    st.sidebar.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '''
        <div style="background-color: #0E1117; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="margin-bottom: 20px;">MLOPS - Promotion Bootcamp MLE été 2024</h3>
            <h3>Participants</h3>
            <ul class="linkedin-list">
                <li>
                    <a href="https://www.linkedin.com/in/kévin-ory" target="_blank" class="linkedin-button">Kevin Ory</a>
                </li>               
                <li>
                    <a href="https://www.linkedin.com/in/truong-xavier" target="_blank" class="linkedin-button">Xavier Truong</a>
                </li>
            </ul>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #---------------------------------------------------------------------------
    # Sidebar
    #---------------------------------------------------------------------------


    #--------------------------------------------------------------------------- 
    # Introduction au Projet 
    #--------------------------------------------------------------------------- 
    if page == 'Introduction au Projet':
        # Affichage de l'image d'introduction
        st.image(image_path + '/Documancer3.gif', use_container_width=True)
        st.title('Présentation de ' + name_app)

        # Introduction et Contexte
        st.subheader('Introduction et Contexte du Projet')
        st.markdown('''
        Dans un monde où de nombreux secteurs comme la finance, la santé, et le juridique sont submergés de documents physiques et numériques, l'automatisation de leur gestion est un enjeu crucial.
        Notre projet s'est concentré sur la classification automatique de documents en utilisant des techniques avancées de deep learning.
        Le dataset que nous avons utilisé, RVL-CDIP, contient 400 000 images de documents répartis en 16 catégories (lettres, factures, mémos, etc.).
        ''')

        # Objectifs
        st.title('Objectifs du Projet DST')
        st.markdown('''
        L'objectif principal du projet est de développer un modèle performant capable de classer automatiquement les documents en fonction de leur type.
        Nous avions exploré plusieurs approches, notamment les réseaux convolutifs (CNN) pour les images, BERT pour l'analyse textuelle après OCR, et une approche multimodale combinant les deux pour optimiser les performances.
        ''')

        # Le dataset RVL-CDIP
        st.subheader('Le dataset RVL-CDIP')
        st.markdown('''
        Le dataset **RVL-CDIP** (Ryerson Vision Lab Complex Document Information Processing) est une ressource de référence dans le domaine de la classification de documents.
        Il contient **400 000 images de documents numérisés**, réparties en **16 catégories**, offrant une diversité qui permet de tester la capacité des modèles à reconnaître et différencier des types de documents variés.
        ''')

        # Structure des Données
        col1, col2 = st.columns([2, 1])  # La première colonne occupe 2/3, la seconde 1/3

        # Texte dans la colonne de gauche
        with col1:
            st.subheader('Structure des Données')
            st.markdown('''
            Le dataset est divisé en trois ensembles pour faciliter l'entraînement et l'évaluation des modèles :
            - **Ensemble d'entraînement** : 320 000 images utilisées pour ajuster les poids des modèles.
            - **Ensemble de validation** : 40 000 images permettant de vérifier la capacité du modèle à généraliser sur des données non vues.
            - **Ensemble de test** : 40 000 images pour évaluer la performance finale du modèle.
            ''')

        # Pie chart dans la colonne de droite
        with col2:
            # Création des données pour le pie chart
            labels = ['Entraînement', 'Validation', 'Test']
            sizes = [80, 10, 10]  # Proportions en pourcentage
            colors = ['#66b3ff', '#99ff99', '#ffcc99']
            explode = (0.1, 0, 0)  # Mettre en avant le segment de l'entraînement

            # Création du pie chart
            fig, ax = plt.subplots(figsize=(4, 4))  # Ajuste la taille du graphique
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
            ax.axis('equal')  # Assure que le pie chart est circulaire.

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)

        # Liste des catégories et image d'exemple
        st.markdown('''
        Chaque image est associée à une étiquette correspondant à sa classe parmi les 16 catégories suivantes :
        ''')

        # Division en deux colonnes
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('''
            1. Lettre  
            2. Formulaire  
            3. Messagerie électronique  
            4. Manuscrit  
            5. Publicité  
            6. Rapport scientifique  
            7. Publication scientifique  
            8. Spécification
            ''')

        with col2:
            st.markdown('''
            9. Dossier de fichiers  
            10. Article de presse  
            11. Budget  
            12. Facture  
            13. Présentation  
            14. Questionnaire  
            15. CV  
            16. Mémo
            ''')

        image_labels = Image.open(image_path + '/labels_example.png')
        st.image(image_labels, caption="Composition et exemples du dataset RVL-CDIP", use_container_width=True)

        # Développement de 3 stratégies
        st.subheader('Stratégies mises en place :')
        st.markdown('''
        Pour répondre à cette problématique, 3 stratégies ont été développées :

        - Modèle **CNN** : Classification de documents par les images -> accuracy 86%

        - Modèle **BERT** : Classification de documents par les textes -> accuracy 83%

        - Modèle **BERT-CNN** : Classification de documents par les images et les textes -> accuracy 90%
        ''')

        st.image(image_path + "/strategies.png", caption="Stratégies déployées", use_container_width=True)
        # st.image(image_path + "/cnn_workflow.png", caption="Workflow CNN", use_container_width=True)
        # st.image(image_path + "/bert.PNG", caption="Workflow BERT", use_container_width=True)
        # st.image(image_path + "/multimodal_workflow.png", caption="Workflow BERT-CNN", use_container_width=True)

        # Objectifs MLOPs
        st.title('Objectifs du Projet MLOPs')
        st.image(image_path + '/cnn_workflow.png', use_container_width=True)

        st.subheader('''Le modèle CNN: un modèle léger et robuste''')
        st.markdown('''
        Dans cette partie MLOps, nous avons choisi de déployer le modèle CNN car il est spécifiquement optimisé 
        pour analyser les informations visuelles des documents, avec une architecture plus légère 
        et une meilleure efficacité sur des tâches basées uniquement sur des images, 
        par rapport au modèle BERT ou multimodal qui nécessitent un traitement supplémentaire 
        des données textuelles.
        ''')

        st.image(image_path + "/cnn_w_vgg16.png", caption="architecture CNN VG16 full fine-tuning", use_container_width=True)
        

        st.markdown('''
        L'objectif ici a été de déployer notre modèle CNN pour garantir un système automatisé, 
        scalable et fiable pour classer des documents à partir de leurs caractéristiques visuelles. 
        - mise en production du modèle pour traiter des requêtes en temps réel
        - suivi des performances via des métriques (monitoring)
        - gestion des versions pour intégrer rapidement des mises à jour ou des retrainings
        - détection de dérives pour maintenir l’efficacité du modèle face à de nouvelles données.
        ''')
        
        


    #---------------------------------------------------------------------------
    # Architecture du projet
    #---------------------------------------------------------------------------
    elif page == 'Architecture du projet':

        st.title('Architecture du projet')
        st.markdown('''
        L'architecture du projet est basée sur une approche MLOPS, qui combine les pratiques de Machine Learning et DevOps.
                    
        ''')
        st.image(image_path + '/Architecture.jpg', use_container_width=True)
        st.title('Cycle mlpos')
        st.image(image_path + '/cyclemlops.jpg', use_container_width=True)
        st.title('Fine tuning du modèle pour réentrainement')
        st.image(image_path + '/finetuningmodel.jpg', use_container_width=True)
        st.title('Séquence de training')
        st.image(image_path + '/sequencetraining.jpg', use_container_width=True)
        st.title('Génération du drift')
        st.image(image_path + '/generationdrift.jpg', use_container_width=True)
        st.title('Flux du monitoring mis en place')
        st.image(image_path + '/fluxmonitoring.jpg', use_container_width=True)
    #---------------------------------------------------------------------------
    # Les services exposés
    #---------------------------------------------------------------------------
    elif page == 'Les services exposés':
        st.title('Les services exposés')
        
        # CSS modifié pour un thème sombre
        st.markdown("""
            <style>
            .service-container {
                background-color: #1E1E1E;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .service-item {
                background-color: #2D2D2D;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border: 1px solid #3D3D3D;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: transform 0.2s ease-in-out;
            }
            .service-item:hover {
                transform: translateX(5px);
                background-color: #333333;
            }
            .service-name {
                color: #2ED4DA;
                font-weight: bold;
                font-size: 1.1em;
            }
            .service-url {
                color: #969696;
            }
            .service-url a {
                color: #2ED4DA;
                text-decoration: none;
                transition: color 0.2s ease-in-out;
            }
            .service-url a:hover {
                color: #382DD5;
            }
            .service-description {
                background-color: #1E1E1E;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .service-description h3 {
                color: #2ED4DA;
                margin-bottom: 15px;
            }
            .service-description ul {
                list-style-type: none;
                padding-left: 0;
            }
            .service-description li {
                color: #FFFFFF;
                margin-bottom: 10px;
                padding-left: 20px;
                position: relative;
            }
            .service-description li::before {
                content: "•";
                color: #2ED4DA;
                position: absolute;
                left: 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Affichage des services avec icônes
        services = [
            {"name": "Grafana", "url": "http://localhost:3100", "icon": "📊"},
            {"name": "Prometheus", "url": "http://localhost:9090", "icon": "📈"},
            {"name": "Metrics Exporter", "url": "http://localhost:8000", "icon": "📤"},
            {"name": "MLflow", "url": "http://localhost:8080", "icon": "🔄"},
            {"name": "Drift Monitoring", "url": "http://localhost:8088", "icon": "📉"},
            {"name": "Bentoml metrics", "url": "http://localhost:3000/metrics", "icon": "📉"}
        ]
        
        st.markdown('<div class="service-container">', unsafe_allow_html=True)
        for service in services:
            st.markdown(f"""
                <div class="service-item">
                    <div>
                        <span class="service-name">{service['icon']} {service['name']}</span>
                    </div>
                    <span class="service-url">
                        <a href="{service['url']}" target="_blank">{service['url']}</a>
                    </span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="service-description">
            <h3>Description des services</h3>
            <ul>
                <li><strong>Grafana</strong> : Interface de visualisation pour le monitoring des métriques en temps réel. Permet de créer des tableaux de bord personnalisés pour suivre les performances du système.</li>
                <li><strong>Prometheus</strong> : Système de collecte et stockage des métriques. Assure la surveillance continue des performances et la disponibilité des services.</li>
                <li><strong>Metrics Exporter</strong> : Service d'exposition des métriques du modèle, permettant le suivi des performances et la santé du système de prédiction.</li>
                <li><strong>MLflow</strong> : Plateforme de gestion du cycle de vie ML, permettant le suivi des expériences, la comparaison des modèles et la gestion des déploiements.</li>
                <li><strong>Drift Monitoring</strong> : Système de surveillance de la dérive des données, assurant la détection précoce des anomalies et des changements dans les patterns de données.</li>
                <li><strongBentoml metrics</strong> : récupération des metrics bentoml pour le fonctionnement de l'API et des prédictions</li>    
            </ul>
        </div>
        """, unsafe_allow_html=True)

    #---------------------------------------------------------------------------
    # Entrainons le modèle
    #---------------------------------------------------------------------------
    elif page == 'Entrainons le modèle':
        st.title('Entrainons le modèle')
        
        st.title('Ré-entrainement du modèle de prédiction')
        st.image(image_path + '/VSCode_training.jpg', use_container_width=True)

    #---------------------------------------------------------------------------
    # Prédiction 
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Prédiction 
    #---------------------------------------------------------------------------
    elif page == 'Prédiction':
        st.image(image_path + '/Documancer.gif', use_container_width=True)
        st.title('Prédiction par nos modèles')
        st.markdown('''
        Les prédictions ont été réalisées à l'aide de plusieurs modèles décrits dans la stratégie déployée : CNN, BERT, et l'approche multimodale CNN-BERT.
        Vous pouvez visualiser ici les prédictions effectuées par chaque modèle sur un échantillon de documents.
        ''')

        def generate_token():
            url = "http://localhost:3000/generate_token"
            headers = {"Content-Type": "application/json"}
            data = {"username": "test_user"}
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    return response.json().get("token")
                else:
                    st.error("Erreur lors de la génération du token")
                    return None
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")
                return None

        def check_status():
            url = "http://localhost:3000/status_check"
            headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(url, headers=headers, json={})
                return response.status_code == 200
            except:
                return False

        def predict_image(uploaded_file, token):
            url = "http://localhost:3000/predict"
            headers = {
                "Authorization": f"Bearer {token}",
                "accept": "application/json"
            }
            
            try:
                # Vérifier les informations de l'image
                st.write(f"Type du fichier: {uploaded_file.type}")
                st.write(f"Taille du fichier: {len(uploaded_file.getvalue())} bytes")
                
                # Obtenir les bytes de l'image
                image_bytes = uploaded_file.getvalue()
                
                # Créer le fichier multipart
                files = {
                    "image": (
                        uploaded_file.name,
                        image_bytes,
                        uploaded_file.type
                    )
                }
                
                st.write("Envoi de la requête à l'API...")
                response = requests.post(url, headers=headers, files=files)
                st.write(f"Status code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    st.write("Réponse brute de l'API:", result)
                    return result
                else:
                    st.error(f"Erreur lors de la prédiction: {response.text}")
                    # Afficher les headers de la réponse pour le debug
                    st.write("Headers de la réponse:", dict(response.headers))
                    return None
            
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")
                # Afficher la stack trace complète
                import traceback
                st.error(f"Stack trace: {traceback.format_exc()}")
                return None
            

            
        # Vérifier le statut de l'API
        if not check_status():
            st.error("L'API n'est pas disponible. Veuillez vérifier que le serveur est en cours d'exécution.")
        else:
            st.success("API connectée et prête à l'emploi !")

            # Uploader un fichier image
            uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'])
            
            if uploaded_file is not None:
                # Afficher l'image
                image = Image.open(uploaded_file)
                st.image(image, caption='Image Importée', use_container_width=True)

                # Bouton pour lancer la prédiction
                if st.button("Lancer la prédiction"):
                    with st.spinner('Prédiction en cours...'):
                        # Générer un token
                        token = generate_token()
                        if token:
                            # Faire la prédiction directement avec le fichier uploadé
                            st.write("Token :", token)
                            prediction = predict_image(uploaded_file, token)
                            
                            if prediction:
                                st.markdown("""
                                <div class="prediction-box">
                                    <h3>Résultats de la prédiction</h3>
                                    <p>Type de document prédit : <span>{}</span></p>
                                    <p>Confiance : <span>{:.2f}%</span></p>
                                </div>
                                """.format(
                                    prediction.get("predicted_class", "Non disponible"),
                                    prediction.get("confidence", 0) * 100
                                ), unsafe_allow_html=True)

    #---------------------------------------------------------------------------
    # Conclusion & perspectives
    #---------------------------------------------------------------------------
    elif page == 'Conclusion & perspectives':
        st.title('Conclusion & perspectives')
        st.image(image_path + '/axeamélioration.jpg', use_container_width=True)
        st.markdown('''
        Le projet a démontré l'efficacité des modèles CNN classification de documents.
        Points forts de votre projet :

        MLflow pour le tracking des expériences
        BentoML pour le serving
        Stack de monitoring robuste avec Prometheus/Grafana
        Gestion du drift avec des rapports détaillés

        Métriques ML bien structurées
        Monitoring système
        Dashboards Grafana bien organisés
        Visualisation des performances du modèle

        Data drift

        Détection automatique
        Génération de rapports détaillés
        Interface de visualisation via Nginx
        ''')
        st.title('Ce qui manque')
        st.image(image_path + '/composantsmanquant.jpg', use_container_width=True)
        st.markdown('''
        Points d'amélioration :

        Manque de CI/CD
        Pas de tests automatisés
        Absence de Kubernetes
        Sécurité basique
        ''')
        st.image(image_path + '/cicd.jpg', use_container_width=True)

#-------------------------------------------------------------------------------
# Lancement
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()