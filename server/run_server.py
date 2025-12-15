#  ____  ____      _    __  __  ____ ___
# |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
# | | | | |_) |  / _ \ | |\/| | |  | | | |
# | |_| |  _ <  / ___ \| |  | | |__| |_| |
# |____/|_| \_\/_/   \_\_|  |_|\____\___/
#                           research group
#                             dramco.be/
# 
#    KU Leuven - Technology Campus Gent,
#    Gebroeders De Smetstraat 1,
#    B-9000 Gent, Belgium
# 
#          File: run_server.py
#       Created: 2025-12-12
#        Author: Geoffrey Ottoy
# 
#   Description: This script coordinates a distributed experiment by running
#     a background server that tracks a set of target hosts and their trans-
#     mission state. It waits until all expected hosts are connected and have
#     completed their current tx phase, then broadcasts a synchronized 
#     “tx-start” command to initiate the next transmission round. This process
#     repeats until the server is shut down.
#

from utils.server_com import Server
import signal
import time
import os
import sys
import yaml
import config

# ---------------------------------------------------------
# Set up paths and configuration
# ---------------------------------------------------------
# Construct the path to the experiment settings YAML file
settings_path = os.path.join(config.PROJECT_DIR, "experiment-settings.yaml")

# Output general info about project location
print("Experiment project directory: ", config.PROJECT_DIR)  # Should point to tile-management repo clone

# Check if the tile-management repository exists in the expected location
# Exit with an error code if not
if not config.check_tile_management_repo():
    sys.exit(config.ERRORS["REPO_ERROR"])

# Add utils directory to the system path to import additional helper functions
sys.path.append(config.UTILS_DIR)
from ansible_utils import get_target_hosts, run_playbook

# ---------------------------------------------------------
# Load experiment settings
# ---------------------------------------------------------
with open(settings_path, "r") as f:
    experiment_settings = yaml.safe_load(f)

# Determine which tiles are targeted by the experiment
tiles = experiment_settings.get("tiles", "")
if len(tiles) == 0:
    print("The experiment doesn't target any tiles.")
    sys.exit(config.ERRORS["NO_TILES_ERROR"])

# Retrieve the host list for the targeted tiles
host_list = get_target_hosts(config.INVENTORY_PATH, limit=tiles, suppress_warnings=True)

# Initialize the transmission status dictionary for each host
# "tx-done" indicates whether the host has completed its last transmission
tx_status = {host: {"tx-done": True} for host in host_list}

# ---------------------------------------------------------
# Configure server parameters
# ---------------------------------------------------------
server_settings = experiment_settings.get("server", "")
# Heartbeat interval for client liveness detection
heartbeat_interval = experiment_settings.get("heartbeat_interval", "") + 10
messaging_port = server_settings.get("messaging_port", "")
sync_port = server_settings.get("sync_port", "")

# Instantiate the server object
server = Server(
    msg_port=messaging_port,
    sync_port=sync_port,
    heartbeat_timeout=heartbeat_interval,
    silent=True  # suppress verbose server output
)

# ---------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------
def handle_signal(signum, frame):
    """Stop the server when a SIGINT or SIGTERM is received."""
    print("\nReceived signal, stopping server...")
    server.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ---------------------------------------------------------
# Callback function for "tx-done" messages from clients
# ---------------------------------------------------------
def handle_tx_done(from_host, args):
    """Mark the host's transmission as complete in the tx_status dict."""
    for h in tx_status:
        if h == from_host:
            tx_status[h]['tx-done'] = True

# ---------------------------------------------------------
# Main execution block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Register the "tx-done" callback with the server
    server.on("tx-done", handle_tx_done)

    # Start the server in a background thread (non-blocking)
    server.start()
    print("Server running in background thread.")

    # Retrieve experiment duration from settings (default to 10s)
    duration = experiment_settings.get("duration", 10)

    # ---------------------------------------------------------
    # Main loop: monitor client connections and manage transmissions
    # ---------------------------------------------------------
    try:
        while server.running:
            # Get currently connected clients from the server
            connected_clients = server.get_connected()

            # Check which hosts in tx_status are missing from the connected clients
            missing = [h for h in tx_status if h.encode() not in connected_clients]

            if missing:
                # If some hosts are not connected, print waiting message
                print("Waiting on hosts:", missing)

                # Ensure all tx-done flags are set to True
                # (i.e., no ongoing transmission)
                for h in tx_status:
                    tx_status[h]['tx-done'] = True

                # Sleep briefly before checking again
                time.sleep(1)
            else:
                # All hosts are connected, check if all transmissions are done
                all_done = all(v["tx-done"] for v in tx_status.values())

                if all_done:
                    # All hosts are ready for the next transmission
                    print("All hosts are ready")
                    print(f" -> tx-start with duration = {duration} s")

                    # Reset tx-done flags to False before starting a new transmission
                    for h in tx_status:
                        tx_status[h]['tx-done'] = False

                    # Broadcast "tx-start" message to all clients with the duration
                    if server.running:
                        server.broadcast("tx-start", f"duration={duration}")
                    else:
                        print("Server not running.")
                        
    except KeyboardInterrupt:
        # Catch Ctrl+C in main thread for clean shutdown
        pass

    # Stop and join the server thread before exiting
    server.stop()
    server.join()
    print("Server terminated.")
