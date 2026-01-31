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
DEFAULT_PILOT_PORT = "5560"      # Port used for PILOT transmission
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
        help="Port for Pilot ROUTER (default: 5560)",
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

    # RZF regularization (lam=0 -> pure ZF)
    parser.add_argument(
        "--rzf-lam",
        type=float,
        default=1e-6,
        help="RZF regularization lambda (default: 1e-6). Use 0 for pure ZF.",
    )

    # Optional overall AP power scalar (after per-AP normalization)
    parser.add_argument(
        "--ap-power",
        type=float,
        default=1.0,
        help="Per-AP target power scalar after row-normalization (default: 1.0).",
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
WAIT_TIMEOUT = args.wait_timeout

RZF_LAM = float(args.rzf_lam)
AP_POWER = float(args.ap_power)

# Creates a socket instance
context = zmq.Context()

sync_socket = context.socket(zmq.PUB)
sync_socket.bind("tcp://{}:{}".format(host, sync_port))

alive_socket = context.socket(zmq.REP)
alive_socket.bind("tcp://{}:{}".format(host, alive_port))

data_socket = context.socket(zmq.REP)
data_socket.bind("tcp://{}:{}".format(host, data_port))

# Measurement and experiment identifiers
meas_id = 0
unique_id = str(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

alive_poller = zmq.Poller()
alive_poller.register(alive_socket, zmq.POLLIN)

new_msg_received = 0
print(f"Starting experiment: {unique_id}")

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


def compute_rzf_weights_from_conj_channels(g_list, lam, ap_power=1.0, debug=False):
    """
    Compute downlink RZF/ZF precoder W given G = H* (conjugated channels).

    Inputs:
      g_list: list of length K, each element is np.array shape (N,) complex, containing g_k = h_k* across APs.
      lam: RZF regularization (0 -> ZF)
      ap_power: scalar to scale per-AP power after normalization.
      debug: print effective channel diagnostics (H@W) before/after normalization.

    Returns:
      W: np.array shape (N, K) complex, per-AP weights for each user stream.
    """
    eps = 1e-12
    K = len(g_list)
    if K < 1:
        raise ValueError("g_list must contain at least 1 user channel vector")

    # Stack to G: (K, N), rows users, cols APs
    G = np.vstack([np.asarray(g, dtype=np.complex128) for g in g_list])  # (K, N)
    K_chk, N = G.shape
    if K_chk != K:
        raise RuntimeError("Internal shape error building G")

    # Recover H from G: since G = H*, then H = G*
    H = G.conj()  # (K, N)

    # GG = G* G^T  (K,K)
    GG = G.conj() @ G.T  # (K,K)

    # Invert (GG + lam I)
    A = np.linalg.inv(GG + lam * np.eye(K, dtype=np.complex128))  # (K,K)

    # W (unnormalized) = G^T A   (N,K)
    W = G.T @ A

    if debug:
        # Effective channel BEFORE normalization
        E0 = H @ W  # (K,K)
        diag0 = np.abs(np.diag(E0)) ** 2
        off0 = E0.copy()
        np.fill_diagonal(off0, 0.0)
        off0_pow = np.abs(off0) ** 2

        Pcols0 = np.sum(np.abs(W) ** 2, axis=0)  # (K,)

        print("---- DEBUG: before row-norm ----")
        print("lam =", lam)
        print("E0 = H@W (before norm):\n", E0)
        print("|diag(E0)|^2 =", diag0)
        print("|offdiag(E0)|^2 =\n", off0_pow)
        print("stream powers Pk = sum_n |w_nk|^2 =", Pcols0)
        if K >= 2:
            ratio_db = 10 * np.log10((Pcols0[0] + eps) / (Pcols0[1] + eps))
            print(f"P1/P2 ratio(dB) = {ratio_db:.2f}")

    # -----------------------------
    # Per-AP (row) normalization:
    # enforce sum_k |w_{n,k}|^2 = ap_power
    # -----------------------------
    row_norm = np.linalg.norm(W, axis=1, keepdims=True) + eps  # (N,1)
    W = W / row_norm
    if ap_power is not None:
        W = np.sqrt(float(ap_power)) * W

    if debug:
        # Effective channel AFTER normalization
        E1 = H @ W  # (K,K)
        diag1 = np.abs(np.diag(E1)) ** 2
        off1 = E1.copy()
        np.fill_diagonal(off1, 0.0)
        off1_pow = np.abs(off1) ** 2

        Pcols1 = np.sum(np.abs(W) ** 2, axis=0)  # (K,)

        # also verify per-AP constraint
        prow = np.sum(np.abs(W) ** 2, axis=1)
        print("---- DEBUG: after row-norm ----")
        print("E1 = H@W (after norm):\n", E1)
        print("|diag(E1)|^2 =", diag1)
        print("|offdiag(E1)|^2 =\n", off1_pow)
        print("stream powers Pk = sum_n |w_nk|^2 =", Pcols1)
        if K >= 2:
            ratio_db = 10 * np.log10((Pcols1[0] + eps) / (Pcols1[1] + eps))
            print(f"P1/P2 ratio(dB) = {ratio_db:.2f}")
        print("per-AP power check: min/mean/max =", float(np.min(prow)), float(np.mean(prow)), float(np.max(prow)))

    return W



with open(output_path, "w") as f:
    f.write(f"experiment: {unique_id}\n")
    f.write(f"num_subscribers: {num_subscribers}\n")
    f.write(f"num_pilots: {num_pilots}\n")
    f.write(f"rzf_lambda: {RZF_LAM}\n")
    f.write(f"ap_power: {AP_POWER}\n")
    f.write(f"measurments:\n")

    while True:
        print(f"Waiting for {num_subscribers + num_pilots} subscribers to send a message...")

        f.write(f"  - meas_id: {meas_id}\n")
        f.write("    active_tiles:\n")

        messages_received = 0

        ################## SYNC ###########################################
        while messages_received < num_subscribers + num_pilots:
            socks = dict(alive_poller.poll(1000))

            if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
                break

            if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
                new_msg_received = time.time()
                message = alive_socket.recv_string()
                messages_received += 1

                print(f"{message} ({messages_received}/{num_subscribers})")
                f.write(f"     - {message}\n")

                alive_socket.send_string("Response from server")

        print(f"sending 'SYNC' message in {delay}s...")
        f.flush()
        time.sleep(delay)

        meas_id += 1
        sync_socket.send_string(f"{meas_id} {unique_id}")
        print(f"SYNC {meas_id}")

        ################## PILOT ###########################################
        identities.clear()
        hostnames.clear()
        csi_P1s.clear()
        csi_P2s.clear()

        messages_received = 0

        # Receive all subscriber messages (CSI)
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

                # IMPORTANT:
                # You stated AP reports h*.
                # So we interpret: g_k = h_k* = ampl_k * exp(j * phi_Pk)
                g1 = ampl_P1 * np.exp(1j * phi_P1)
                g2 = ampl_P2 * np.exp(1j * phi_P2)

                identities.append(identity)
                hostnames.append(hostname)
                csi_P1s.append(g1)
                csi_P2s.append(g2)

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

        # =============================
        # Dual-user ZF / RZF using G=H*
        # =============================
        g1 = np.asarray(csi_P1s, dtype=np.complex128)  # (N,)
        g2 = np.asarray(csi_P2s, dtype=np.complex128)  # (N,)

        N = len(identities)
        if g1.size != N or g2.size != N:
            print(f"ERROR: size mismatch: g1={g1.size}, g2={g2.size}, identities={N}")
            continue

        # Compute W with row-normalization for fixed per-AP power
        try:
            W = compute_rzf_weights_from_conj_channels([g1, g2], lam=RZF_LAM, ap_power=AP_POWER, debug=True)
        except np.linalg.LinAlgError as e:
            print("ERROR: matrix inversion failed:", e)
            continue
        except Exception as e:
            print("ERROR: compute_rzf_weights_from_conj_channels failed:", e)
            continue

        w_u1 = W[:, 0]
        w_u2 = W[:, 1]
        P1 = float(np.sum(np.abs(w_u1)**2))
        P2 = float(np.sum(np.abs(w_u2)**2))
        print(f"stream powers: P1={P1:.3f}, P2={P2:.3f}, ratio(dB)={10*np.log10(P1/(P2+1e-12)):.2f}")

        # Reply: send full complex weights (re/im) so AP can apply amplitude+phase
        for idx, identity in enumerate(identities):
            response = {
                "w_u1_re": float(np.real(w_u1[idx])),
                "w_u1_im": float(np.imag(w_u1[idx])),
                "w_u2_re": float(np.real(w_u2[idx])),
                "w_u2_im": float(np.imag(w_u2[idx])),
            }
            router_socket.send_multipart([identity, json.dumps(response).encode()])

        f.flush()

        ################## TX MODE WAIT ####################################
        print(f"Waiting for {num_subscribers} subscribers to send a TX Mode ...")
        messages_received = 0

        while messages_received < num_subscribers:
            socks = dict(alive_poller.poll(1000))

            if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
                break

            if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
                new_msg_received = time.time()
                message = alive_socket.recv_string()
                messages_received += 1
                print(f"{message} ({messages_received}/{num_subscribers})")
                alive_socket.send_string("Response from server")

        print("Wait 10s ...")
        time.sleep(10)

        # save_phases()
