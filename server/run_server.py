from utils.server_com import Server
import signal
import time
import os
import yaml
import config

# We start by setting some paths
settings_path = os.path.join(config.PROJECT_DIR, "experiment-settings.yaml")

# Output some general information before we start
print("Experiment project directory: ", config.PROJECT_DIR) # should point to tile-management repo clone

# Read experiment settings
with open(settings_path, "r") as f:
    experiment_settings = yaml.safe_load(f)

server_settings = experiment_settings.get("server", "")
heartbeat_interval = experiment_settings.get("heartbeat_interval", "") + 10
messaging_port = server_settings.get("messaging_port", "")
sync_port = server_settings.get("sync_port", "")

server = Server(msg_port=messaging_port, sync_port=sync_port, heartbeat_timeout=heartbeat_interval, silent=False)

def handle_signal(signum, frame):
    print("\nReceived signal, stopping server...")
    server.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == "__main__":
    server.start()   # <-- non-blocking
    print("Server running in background thread.")

    # Main thread idle loop
    try:
        while server.running:
            time.sleep(30)
            server.broadcast("sync", "sync message payload")
    except KeyboardInterrupt:
        pass

    server.stop()
    server.join()
    print("Server terminated.")
