#!/usr/bin/python3
# usage: sync_server.py <delay> <num_subscribers>

# VALUE "num_subscribers" --> IMPORTANT --> The server waits until all subscribers have sent their "alive" or ready message before starting a measurement.

import zmq
import time
import sys
import os
from datetime import datetime, UTC, timezone
# from helper import *
import numpy as np

# =============================================================================
#                           Experiment Configuration
# =============================================================================
host = "*"  # Host address to bind to. "*" means all available interfaces.
sync_port = "5557"  # Port used for synchronization messages.
alive_port = "5558"  # Port used for heartbeat/alive messages.
data_port = "5559"  # Port used for data transmission.
# =============================================================================
# =============================================================================

if len(sys.argv) > 1:
    delay = int(sys.argv[1])
    num_subscribers = int(sys.argv[2])
else:
    delay = 2
    num_subscribers = 42

# Creates a socket instance
context = zmq.Context()

sync_socket = context.socket(zmq.PUB)
# Binds the socket to a predefined port on localhost
sync_socket.bind("tcp://{}:{}".format(host, sync_port))

# Create a SUB socket to listen for subscribers
alive_socket = context.socket(zmq.REP)
alive_socket.bind("tcp://{}:{}".format(host, alive_port))

# Create a SUB socket to listen for subscribers
data_socket = context.socket(zmq.REP)
data_socket.bind("tcp://{}:{}".format(host, data_port))

# Measurement and experiment identifiers
meas_id = 0

# Unique ID for the experiment based on current UTC timestamp
unique_id = str(datetime.now(UTC).strftime("%Y%m%d%H%M%S"))

# Directory where this script is located
# script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# ZeroMQ poller setup
poller = zmq.Poller()
poller.register(
    alive_socket, zmq.POLLIN
)  # Register the alive socket to monitor incoming messages

# Track time of the last received message

# Maximum time to wait for messages before breaking out of the inner loop (10 minutes)
WAIT_TIMEOUT = 60.0 * 10.0

# Inform the user that the experiment is starting
print(f"Starting experiment: {unique_id}")

# Path setup for repo imports and data output
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
repo_root = os.path.dirname(parent_path)
sys.path.insert(0, repo_root)
output_path = os.path.join(parent_path, f"record/data/exp-{unique_id}.yml")

from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

settings = read_yaml_file("experiment-settings.yaml")
rfep = RFEP(settings["ep"]["ip"], settings["ep"]["port"])

CAPTURE_POWER_TIME = 5
prev_power = 0

def send_sync():
    """
    Synchronize measurement across all subscribers and handle their responses.
    
    Waits for incoming messages from all subscribers with a configurable timeout.
    Records received messages to a file and sends responses back. After all subscribers
    have reported or timeout is reached, broadcasts a synchronization signal with the
    current measurement ID and unique server identifier to trigger the next measurement cycle.
    
    Global variables modified:
        meas_id: Incremented after each synchronization cycle.
    """
    global meas_id, poller
    messages_received = 0
    new_msg_received = 0 

    while messages_received < num_subscribers:
        # Poll the socket for incoming messages with a 1-second timeout
        _socks = dict(poller.poll(1000))

        # If some messages were received but no new message comes within WAIT_TIMEOUT, break
        if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
            break

        if alive_socket in _socks and _socks[alive_socket] == zmq.POLLIN:
            # Record time when a new message is received
            new_msg_received = time.time()

            # Receive the message string from the subscriber
            _message = alive_socket.recv_string()
            messages_received += 1

            # Print received message and write it to the YAML file
            print(f"{_message} ({messages_received}/{num_subscribers})")
            f.write(f"     - {_message}\n")

            # Process the request (example placeholder)
            response = "Response from server"

            # Send response back to the subscriber
            alive_socket.send_string(response)

    # Wait a fixed delay before sending the next SYNC signal
    print(f"sending 'SYNC' message in {delay}s...")
    f.flush()
    time.sleep(delay)

    # Increment measurement ID for next iteration
    meas_id += 1

    # Broadcast synchronization message to all subscribers
    sync_socket.send_string(f"{meas_id} {unique_id}")  # str(meas_id)
    print(f"SYNC {meas_id}")


def collect_power(tx_time: float) -> float:
    max_samples = []

    # sleep till TX+1.0
    time.sleep(tx_time+1.0 - time.time())

    start_time = time.time()
    print(f"Collecting power measurements for {CAPTURE_POWER_TIME} seconds...")
    while CAPTURE_POWER_TIME > time.time() - start_time:
        d = rfep.get_data()
        if d is None:
            continue
        max_samples.append(d.pwr_pw)

    #take median of the max 10 power samples, median to avoid outliers
    max_samples = sorted(max_samples, reverse=True)[:10]
    return np.median(max_samples).item() # np.array to scalar 


def wait_till_tx_done(is_stronger: bool):
    # Wait for all subscribers to send a TX DONE MODE message
    print(f"Waiting for {num_subscribers} subscribers to send a TX DONE Mode ...")

    # Track number of messages received from subscribers
    messages_received = 0
    new_msg_received = 0

    max_starting_in = 0.0
    tx_updates = []

    while messages_received < num_subscribers:
        # Poll the socket for incoming messages with a 1-second timeout
        socks = dict(poller.poll(1000))

        # If some messages were received but no new message comes within WAIT_TIMEOUT, break
        if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
            break

        if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
            # Record time when a new message is received
            new_msg_received = time.time()

            # Receive the message string from the subscriber
            message = alive_socket.recv_string()
            messages_received += 1

            # Parse and log TX DONE payload: "<HOSTNAME> <applied_phase> <applied_delta>"
            parts = message.split()
            if len(parts) >= 4:
                host, applied_phase, applied_delta, starting_in = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                )
                if float(starting_in) > max_starting_in:
                    max_starting_in = float(starting_in)
                tx_updates.append((host, applied_phase, applied_delta))
                print(
                    f"{host} phase={applied_phase} delta={applied_delta} "
                    f"({messages_received}/{num_subscribers})"
                )
            else:
                print(f"{message} ({messages_received}/{num_subscribers})")

            # Send response back to the subscriber
            alive_socket.send_string(str(is_stronger))
    return max_starting_in, tx_updates

with open(output_path, "w") as f:
    # Write experiment metadata to the YAML file
    f.write(f"experiment: {unique_id}\n")
    f.write(f"num_subscribers: {num_subscribers}\n")
    f.write("measurments:\n")

    while True:
        # Wait for all subscribers to send a message
        print(f"Waiting for {num_subscribers} subscribers to send a message...")

        # Start a new measurement entry in the YAML file
        f.write(f"  - meas_id: {meas_id}\n")
        f.write("    active_tiles:\n")

        # Track number of messages received from subscribers
        send_sync()
        stronger = False
        f.write("    iterations:\n")

        for i in range(0, 100):
            next_tx_in, tx_updates = wait_till_tx_done(is_stronger=stronger)
            next_tx_time = time.time() + next_tx_in

            max_power = collect_power(next_tx_time)

            stronger = max_power > prev_power
            prev_power = max_power

            f.write(f"      - iter: {i}\n")
            f.write(f"        max_power_pw: {max_power}\n")
            f.write("        clients:\n")
            for host, applied_phase, applied_delta in tx_updates:
                f.write(f"          - host: {host}\n")
                f.write(f"            applied_phase: {applied_phase}\n")
                f.write(f"            applied_delta: {applied_delta}\n")
