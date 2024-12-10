import os
import subprocess
import yaml
import socket

def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0

def check_ports(config_file):
    """Check the availability of ports based on a configuration file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    unavailable_ports = []
    for service, port in config["ports"].items():
        if not is_port_available(port):
            print(f"⚠️  Le port {port} est déjà utilisé par {service}.")
            unavailable_ports.append(service)
        else:
            print(f"✅ Le port {port} est disponible pour {service}.")
    return unavailable_ports

def execute_docker_compose(command, compose_file):
    """Execute docker-compose commands."""
    try:
        subprocess.run(
            ["docker-compose", "-f", compose_file, command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"✔️ Commande `docker-compose {command}` exécutée avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de `docker-compose {command}` : {e.stderr.decode()}")

def restart_services(compose_file):
    """Restart services using docker-compose."""
    print("🔄 Arrêt des services...")
    execute_docker_compose("down", compose_file)
    print("🔄 Redémarrage des services...")
    execute_docker_compose("up -d", compose_file)

# Main workflow
if __name__ == "__main__":
    config_file = "config.yml"
    compose_file = "monitoring/docker-compose.yml"

    # Check ports and restart services if necessary
    unavailable_ports = check_ports(config_file)
    if unavailable_ports:
        print("\nCertains ports sont occupés. Arrêt et redémarrage des services en cours...")
        restart_services(compose_file)
        print("\nVérification des ports après redémarrage :")
        check_ports(config_file)
    else:
        print("\nTous les ports sont disponibles. Aucun redémarrage nécessaire.")
