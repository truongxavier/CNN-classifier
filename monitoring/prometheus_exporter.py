from prometheus_client import start_http_server, Gauge, Counter
import time
import os
import mlflow
from mlflow.tracking import MlflowClient

# Configuration du logging
def log_info(message):
    """Helper pour logger avec timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] INFO: {message}")

def log_error(message):
    """Helper pour logger les erreurs avec timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ERROR: {message}")

# Définition des métriques Prometheus
METRICS = {
    # Métriques d'entraînement
    'model_accuracy': Gauge('model_accuracy', 'Current model accuracy'),
    'model_loss': Gauge('model_loss', 'Current model loss'),
    'model_val_accuracy': Gauge('model_val_accuracy', 'Current validation accuracy'),
    'model_val_loss': Gauge('model_val_loss', 'Current validation loss'),
    
    # Métriques de drift
    'data_drift_detected': Gauge('data_drift_detected', 'Whether data drift was detected (0/1)'),
    'target_drift_detected': Gauge('target_drift_detected', 'Whether target drift was detected (0/1)'),
    'share_of_drifted_columns': Gauge('share_of_drifted_columns', 'Proportion of columns showing drift'),
    
    # Métriques d'entraînement supplémentaires
    'training_epochs_completed': Counter('training_epochs_completed', 'Number of completed training epochs'),
    'training_samples_processed': Counter('training_samples_processed', 'Number of training samples processed'),
}

def update_metrics(client, experiment_name, metrics_dict):
    """
    Mise à jour des métriques depuis MLflow
    """
    try:
        # Recherche de l'expérience
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            log_error(f"Expérience '{experiment_name}' non trouvée")
            return
        
        log_info(f"Expérience trouvée : {experiment.experiment_id}")
        
        # Recherche des runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        log_info(f"Nombre de runs trouvés : {len(runs)}")
        
        if not runs:
            log_error("Aucun run trouvé")
            return
            
        # Récupération du dernier run
        latest_run = runs[0]
        log_info(f"Dernier run ID : {latest_run.info.run_id}")
        log_info(f"Métriques disponibles : {latest_run.data.metrics}")
        
        # Mise à jour des métriques d'accuracy et loss
        if 'final_accuracy' in latest_run.data.metrics:
            value = float(latest_run.data.metrics['final_accuracy'])
            log_info(f"Setting model_accuracy to {value}")
            metrics_dict['model_accuracy'].set(value)
        
        if 'final_loss' in latest_run.data.metrics:
            value = float(latest_run.data.metrics['final_loss'])
            log_info(f"Setting model_loss to {value}")
            metrics_dict['model_loss'].set(value)
            
        if 'final_val_accuracy' in latest_run.data.metrics:
            value = float(latest_run.data.metrics['final_val_accuracy'])
            log_info(f"Setting model_val_accuracy to {value}")
            metrics_dict['model_val_accuracy'].set(value)
            
        if 'final_val_loss' in latest_run.data.metrics:
            value = float(latest_run.data.metrics['final_val_loss'])
            log_info(f"Setting model_val_loss to {value}")
            metrics_dict['model_val_loss'].set(value)
        
        # Mise à jour des métriques de drift
        if 'initial_data_drift_detected' in latest_run.data.metrics:
            value = int(latest_run.data.metrics['initial_data_drift_detected'])
            log_info(f"Setting data_drift_detected to {value}")
            metrics_dict['data_drift_detected'].set(value)
            
        if 'initial_target_drift_detected' in latest_run.data.metrics:
            value = int(latest_run.data.metrics['initial_target_drift_detected'])
            log_info(f"Setting target_drift_detected to {value}")
            metrics_dict['target_drift_detected'].set(value)
            
        if 'initial_share_of_drifted_columns' in latest_run.data.metrics:
            value = float(latest_run.data.metrics['initial_share_of_drifted_columns'])
            log_info(f"Setting share_of_drifted_columns to {value}")
            metrics_dict['share_of_drifted_columns'].set(value)
            
        # Mise à jour des compteurs
        if 'epochs' in latest_run.data.params:
            value = int(latest_run.data.params['epochs'])
            log_info(f"Incrementing training_epochs_completed by {value}")
            metrics_dict['training_epochs_completed'].inc(value)
            
        if 'batch_size' in latest_run.data.params:
            processed = int(latest_run.data.params['epochs']) * int(latest_run.data.params['batch_size'])
            log_info(f"Incrementing training_samples_processed by {processed}")
            metrics_dict['training_samples_processed'].inc(processed)
            
    except Exception as e:
        log_error(f"Erreur lors de la mise à jour des métriques : {str(e)}")

def main():
    # Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:8080")
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
    EXPERIMENT_NAME = "fine_tuning_experiment"
    
    log_info(f"Démarrage de l'exporteur avec URI MLflow: {MLFLOW_TRACKING_URI}")
    log_info(f"Port Prometheus: {PROMETHEUS_PORT}")
    log_info(f"Nom de l'expérience: {EXPERIMENT_NAME}")
    
    # Démarrage du serveur Prometheus
    start_http_server(PROMETHEUS_PORT)
    log_info(f"Serveur Prometheus démarré sur le port {PROMETHEUS_PORT}")
    
    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    log_info("Client MLflow configuré")
    
    while True:
        try:
            log_info("Début de la mise à jour des métriques")
            update_metrics(client, EXPERIMENT_NAME, METRICS)
            log_info("Mise à jour des métriques terminée")
            time.sleep(60)  # Attendre 1 minute avant la prochaine mise à jour
            
        except Exception as e:
            log_error(f"Erreur dans la boucle principale : {str(e)}")
            time.sleep(60)  # Attendre avant de réessayer

if __name__ == "__main__":
    main()