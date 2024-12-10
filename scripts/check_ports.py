import socket
import yaml

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

for service, port in config["ports"].items():
    if not is_port_available(port):
        print(f"⚠️  Le port {port} est déjà utilisé par {service}.")
    else:
        print(f"✅ Le port {port} est disponible pour {service}.")
