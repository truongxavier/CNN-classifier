#!/bin/bash

# Chemin de base pour le projet
PROJECT_PATH=$(cd "$(dirname "$0")/.." && pwd)

# Configuration des symlinks
declare -A SYMLINKS=(
    # Base
    ["kubernetes/base/bento-deployment.yaml"]="bentoml_service/bentofile.yaml"
    # Monitoring
    ["kubernetes/monitoring/prometheus-deploy.yaml"]="monitoring/prometheus/prometheus.yml"
    ["kubernetes/monitoring/prometheus-alerts.yaml"]="monitoring/prometheus/alert_rules/drift_alerts.yml"
    ["kubernetes/monitoring/grafana-dashboards.yaml"]="monitoring/grafana/provisioning/dashboards/dashboard.yml"
    ["kubernetes/monitoring/grafana-datasources.yaml"]="monitoring/grafana/provisioning/datasources/datasources.yml"
    ["kubernetes/monitoring/grafana-deploy.yaml"]="monitoring/docker-compose.yml"
    ["kubernetes/monitoring/mlflow-compose.yaml"]="data_pipeline/mlflow/docker-compose.yml"
    # Network
    ["kubernetes/network/network-policy.yaml"]="config.yml"
)

# Créer les symlinks dans les modules appropriés
for link in "${!SYMLINKS[@]}"; do
    target="${SYMLINKS[$link]}"
    full_target="$PROJECT_PATH/$target"
    full_link="$PROJECT_PATH/$link"

    # Créer le répertoire parent si nécessaire
    parent_dir=$(dirname "$full_link")
    if [ ! -d "$parent_dir" ]; then
        mkdir -p "$parent_dir"
        echo "Répertoire créé : $parent_dir"
    fi

    # Supprimer un ancien symlink ou fichier
    if [ -e "$full_link" ] || [ -L "$full_link" ]; then
        rm "$full_link"
        echo "Ancien lien ou fichier supprimé : $full_link"
    fi

    # Vérifier si le fichier cible existe avant de créer le lien symbolique
    if [ -e "$full_target" ]; then
        ln -s "$full_target" "$full_link"
        echo "Symlink créé : $full_link -> $full_target"
    else
        echo "Fichier cible introuvable : $full_target. Symlink non créé."
    fi
done
