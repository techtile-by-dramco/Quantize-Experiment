#!/usr/bin/python3
# usage: sync_server.py <delay> <num_subscribers>

# sudo fuser -k 50001/tcp

# VALUE "num_subscribers" --> IMPORTANT --> The server waits until all subscribers have sent their "alive" or ready message before starting a measurement.

import argparse
import zmq
import time
import sys
import os
from datetime import datetime
from helper import *
import json
import numpy as np

# =============================================================================
#                           Experiment Configuration
# =============================================================================
DEFAULT_HOST = "*"               # Host address to bind to. "*" means all available interfaces.
DEFAULT_SYNC_PORT = "5557"       # Port used for synchronization messages.
DEFAULT_ALIVE_PORT = "5558"      # Port used for heartbeat/alive messages.
DEFAULT_DATA_PORT = "5559"       # Port used for data transmission.
DEFAULT_PILOT_PORT =  "5560"  # Port used for PILOT transmission
DEFAULT_DELAY = 2                # Seconds to wait before sending SYNC
DEFAULT_SUBS = 42                # Expected subscribers
def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ sync server for GBWPT experiments.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: *)")
    parser.add_argument("--sync-port", default=DEFAULT_SYNC_PORT, help="Port for SYNC PUB (default: 5557)")
    parser.add_argument("--alive-port", default=DEFAULT_ALIVE_PORT, help="Port for alive/ready REP (default: 5558)")
    parser.add_argument("--data-port", default=DEFAULT_DATA_PORT, help="Port for data REP (default: 5559)")
    parser.add_argument(
        "--pilot-port",
        default=DEFAULT_PILOT_PORT,
        help="Port for Pilot REP (default: 5560)",
    )
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY, help="Delay before sending SYNC (seconds)")
    parser.add_argument("--num-pilots", type=int, default=DEFAULT_SUBS, help="Expected pilots before SYNC")
    parser.add_argument(
        "--num-subscribers",
        type=int,
        default=DEFAULT_SUBS,
        help="Expected subscribers before SYNC",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=60.0 * 10.0,
        help="Timeout in seconds to give up waiting for new ready messages once some arrived (default: 600s).",
    )
    return parser.parse_args()


args = parse_args()
delay = args.delay
num_subscribers = args.num_subscribers
num_pilots = args.num_pilots
host = args.host
sync_port = args.sync_port
alive_port = args.alive_port
data_port = args.data_port
pilot_port = args.pilot_port
# Maximum time to wait for messages before breaking out of the inner loop
WAIT_TIMEOUT = args.wait_timeout

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
unique_id = str(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

# Directory where this script is located
# script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# ZeroMQ alive_poller setup
alive_poller = zmq.Poller()
alive_poller.register(
    alive_socket, zmq.POLLIN
)  # Register the alive socket to monitor incoming messages

# Track time of the last received message
new_msg_received = 0
# Inform the user that the experiment is starting
print(f"Starting experiment: {unique_id}")

# Path to save the experiment data as a YAML file
current_file_path = os.path.abspath(__file__) 
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
output_path = os.path.join(parent_path, f"record/data/exp-{unique_id}.yml")


# Use ROUTER socket to allow delayed reply
router_socket = context.socket(zmq.ROUTER)
router_socket.bind(f"tcp://*:{pilot_port}")

# Poller setup
csi_poller = zmq.Poller()
csi_poller.register(router_socket, zmq.POLLIN)

# Data storage
identities = []
hostnames = []
csi_P1s = []
csi_P2s = []



with open(output_path, "w") as f:
    # Write experiment metadata to the YAML file
    f.write(f"experiment: {unique_id}\n")
    f.write(f"num_subscribers: {num_subscribers}\n")
    f.write(f"num_pilots: {num_pilots}\n")
    f.write(f"measurments:\n")

    while True:
        # Wait for all subscribers to send a message
        print(f"Waiting for {num_subscribers+num_pilots} subscribers to send a message...")

        # Start a new measurement entry in the YAML file
        f.write(f"  - meas_id: {meas_id}\n")
        f.write("    active_tiles:\n")

        # Track number of messages received from subscribers
        messages_received = 0
        start_processing = None

        ################## SYNC ###########################################

        while messages_received < num_subscribers + num_pilots:
            # Poll the socket for incoming messages with a 1-second timeout
            socks = dict(alive_poller.poll(1000))

            # If some messages were received but no new message comes within WAIT_TIMEOUT, break
            if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
                break

            if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
                # Record time when a new message is received
                new_msg_received = time.time()

                # Receive the message string from the subscriber
                message = alive_socket.recv_string()
                messages_received += 1

                # Print received message and write it to the YAML file
                print(f"{message} ({messages_received}/{num_subscribers})")
                f.write(f"     - {message}\n")

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

        ################## PILOT ###########################################
        # Clear for new round
        identities.clear()
        hostnames.clear()
        csi_P1s.clear()
        csi_P2s.clear()

        messages_received = 0
        start_time = time.time()

        # Receive all subscriber messages
        while messages_received < num_subscribers:
            socks = dict(csi_poller.poll(1000))
            if router_socket in socks and socks[router_socket] == zmq.POLLIN:
                identity, msg = router_socket.recv_multipart()
                msg_json = json.loads(msg.decode())

                hostname = msg_json.get("host")
                phi_P1 = float(msg_json.get("phi_P1", 0.0))
                phi_P2 = float(msg_json.get("phi_P2", 0.0))

                ampl_P1 = float(msg_json.get("ampl_P1", 0.0))
                ampl_P2 = float(msg_json.get("ampl_P2", 0.0))

                identities.append(identity)
                hostnames.append(hostname)
                csi_P1s.append(ampl_P1 * np.exp(1j * phi_P1))
                csi_P2s.append(ampl_P2 * np.exp(1j * phi_P2))

                messages_received += 1
                print(
                    "event=csi host=%s count=%d total=%d phi_P1=%.6f phi_P2=%.6f ampl_P1=%.6f ampl_P2=%.6f"
                    % (
                        hostname,
                        messages_received,
                        num_subscribers,
                        phi_P1,
                        phi_P2,
                        ampl_P1,
                        ampl_P2,
                    )
                )

                f.write(f"     - {hostname}\n")

        if messages_received == 0:
            continue

        angles = -np.angle(np.asarray(csi_P1s)) # MRT

        # Send individual replies to all identities
        for identity, bf_angle in zip(identities, angles):
            # delta_phase = np.angle(original_csi) - avg_phase
            # response = {"delta_phase": delta_phase, "avg_ampl": avg_ampl}
            reponse = {"phi_BF": bf_angle}
            router_socket.send_multipart([identity, json.dumps(reponse).encode()])

        f.flush()

        # Wait for all subscribers to send a TX MODE message
        print(f"Waiting for {num_subscribers} subscribers to send a TX Mode ...")

        # Track number of messages received from subscribers
        messages_received = 0
        start_processing = None

        while messages_received < num_subscribers:
            # Poll the socket for incoming messages with a 1-second timeout
            socks = dict(alive_poller.poll(1000))

            # If some messages were received but no new message comes within WAIT_TIMEOUT, break
            if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
                break

            if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
                # Record time when a new message is received
                new_msg_received = time.time()

                # Receive the message string from the subscriber
                message = alive_socket.recv_string()
                messages_received += 1

                # Print received message and write it to the YAML file
                print(f"{message} ({messages_received}/{num_subscribers})")

                # Process the request (example placeholder)
                response = "Response from server"

                # Send response back to the subscriber
                alive_socket.send_string(response)

        print(f"Wait 10s ...")

        time.sleep(10)

        # print(f"Measure phases")

        # save_phases()
