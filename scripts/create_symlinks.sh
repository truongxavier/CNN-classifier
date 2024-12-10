#!/bin/bash

# Chemin de base pour le projet
PROJECT_PATH=$(cd "$(dirname "$0")/.." && pwd)

# Symlinks configuration
declare -A SYMLINKS=(
    ["kubernetes/base/bento-deployment.yaml"]="../bentoml_service/src/bento-deployment.yaml"
    ["kubernetes/monitoring/prometheus-deploy.yaml"]="../monitoring/prometheus/prometheus.yml"
    ["kubernetes/monitoring/grafana-deploy.yaml"]="../monitoring/grafana/grafana-deploy.yaml"
)

# Créer les symlinks dans le bon module
for link in "${!SYMLINKS[@]}"; do
    target="${SYMLINKS[$link]}"
    full_target="$PROJECT_PATH/$target"
    full_link="$PROJECT_PATH/$link"

    # Créer le répertoire parent si nécessaire
    parent_dir=$(dirname "$full_link")
    if [ ! -d "$parent_dir" ]; then
        mkdir -p "$parent_dir"
        echo "Created directory: $parent_dir"
    fi

    # Supprimer un ancien symlink ou fichier
    if [ -e "$full_link" ] || [ -L "$full_link" ]; then
        rm "$full_link"
    fi

    # Créer le symlink
    ln -s "$full_target" "$full_link"
    echo "Created symlink: $full_link -> $full_target"
done
