
import streamlit as st
from PIL import Image
import os
import requests
import json
import shutil
import tempfile

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
        # st.image(image_path + '/Documancer.gif', use_container_width=True)
        st.image(image_path + '/Documancer3.gif', use_container_width=True)
        st.title('présentation de '+ name_app)
        st.subheader('Introduction et Contexte du Projet')
        st.markdown('''
        Dans un monde où de nombreux secteurs comme la finance, la santé, et le juridique sont submergés de documents physiques et numériques, l'automatisation de leur gestion est un enjeu crucial.
        Notre projet s'est concentré sur la classification automatique de documents en utilisant des techniques avancées de deep learning.
        Le dataset que nous avons utilisé, RVL-CDIP, contient 400 000 images de documents répartis en 16 catégories (lettres, factures, mémos, etc.).
        ''')
        st.subheader('Objectifs du Projet')
        st.markdown('''
        L'objectif principal du projet est de développer le MLOPS pour notre CNN , pour la classification de documents
        ''')
        st.image(image_path + '/cnn_workflow.png', use_container_width=True)
        
    #---------------------------------------------------------------------------
    # Architecture du projet
    #---------------------------------------------------------------------------
    elif page == 'Architecture du projet':

        st.title('Architecture du projet')
        st.markdown('''
        L'architecture du projet est basée sur une approche MLOPS, qui combine les pratiques de Machine Learning et DevOps.
                    
        ''')
        st.image(image_path + '/Architecture.jpg', use_container_width=True)
        st.image(image_path + '/cyclemlops.jpg', use_container_width=True)
        st.image(image_path + '/finetuningmodel.jpg', use_container_width=True)
        st.image(image_path + '/generationdrift.jpg', use_container_width=True)
        st.image(image_path + '/fluxmonitoring.jpg', use_container_width=True)
    #---------------------------------------------------------------------------
    # Les services exposés
    #---------------------------------------------------------------------------
    elif page == 'Les services exposés':
        st.title('Les services exposés')
        
        st.title('Architecture du projet')
        

    #---------------------------------------------------------------------------
    # Entrainons le modèle
    #---------------------------------------------------------------------------
    elif page == 'Entrainons le modèle':
        st.title('Entrainons le modèle')
        
           

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