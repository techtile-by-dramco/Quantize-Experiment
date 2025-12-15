from utils.server_com import Server
import signal
import time
import os
import sys
import yaml
import config

# We start by setting some paths
settings_path = os.path.join(config.PROJECT_DIR, "experiment-settings.yaml")

# Output some general information before we start
print("Experiment project directory: ", config.PROJECT_DIR) # should point to tile-management repo clone

# Check if the tile-management repo is in the default location (no use in continuing if it's not)
if not config.check_tile_management_repo():
    sys.exit(config.ERRORS["REPO_ERROR"])

# Import code from the tile-management repo
sys.path.append(config.UTILS_DIR)
from ansible_utils import get_target_hosts, run_playbook

# Read experiment settings
with open(settings_path, "r") as f:
    experiment_settings = yaml.safe_load(f)

tiles = experiment_settings.get("tiles", "")
if len(tiles) == 0:
    print("The experiment doesn't target any tiles.")
    sys.exit(config.ERRORS["NO_TILES_ERROR"])

host_list = get_target_hosts(config.INVENTORY_PATH, limit=tiles, suppress_warnings=True)
tx_status = {host: {"tx-done": True} for host in host_list}

server_settings = experiment_settings.get("server", "")
heartbeat_interval = experiment_settings.get("heartbeat_interval", "") + 10
messaging_port = server_settings.get("messaging_port", "")
sync_port = server_settings.get("sync_port", "")

server = Server(msg_port=messaging_port, sync_port=sync_port, heartbeat_timeout=heartbeat_interval, silent=True)

def handle_signal(signum, frame):
    print("\nReceived signal, stopping server...")
    server.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def handle_tx_done(from_host, args):
    for h in tx_status:
        if h == from_host:
            tx_status[h]['tx-done'] = True

if __name__ == "__main__":
    server.on("tx-done", handle_tx_done)
    server.start()   # <-- non-blocking
    print("Server running in background thread.")

    duration = experiment_settings.get("duration", 10)

    # Main thread idle loop
    try:
        while server.running:
            connected_clients = server.get_connected()
            missing = [h for h in tx_status if h.encode() not in connected_clients]
            if missing:
                print("Waiting on hosts:", missing)
                # Make sure all are set to "tx-done"
                for h in tx_status:
                    tx_status[h]['tx-done'] = True
                time.sleep(1)
            else:
                # all hosts in tx_status are seen by the server
                all_done = all(v["tx-done"] for v in tx_status.values())

                if all_done: # restart tx
                    print("All hosts are ready")
                    print(f" -> tx-start with duration = {duration} s")
                    for h in tx_status:
                        tx_status[h]['tx-done'] = False
                    if server.running:
                        server.broadcast("tx-start", f"duration={duration}")
                    else:
                        print("server not running")
    except KeyboardInterrupt:
        pass

    server.stop()
    server.join()
    print("Server terminated.")
