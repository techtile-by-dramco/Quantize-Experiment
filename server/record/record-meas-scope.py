#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************************** #
#                                       IMPORTS / PATHS                                      #
# ****************************************************************************************** #

import argparse
from time import sleep, time
from typing import Optional

from Positioner import PositionerClient
from TechtilePlotter.TechtilePlotter import TechtilePlotter
from TechtileScope import Scope
import atexit
import os
import signal
import sys
import threading
import json

import numpy as np
import zmq

# ****************************************************************************************** #
#                                           CONFIG                                           #
# ****************************************************************************************** #

SAVE_EVERY = 60.0  # seconds
FOLDER = "ZF-test1"  # subfolder inside data/where to save measurement data
TIMESTAMP = round(time())
DEFAULT_DURATION = None  # seconds, override via CLI

# ---------------------------
# EVM stream from client (ZMQ)
# ---------------------------
# Change this to your client IP (the machine running the demodulator PUB socket)
CLIENT_EVM_ADDR = "tcp://rpi-t03.local:52001"

# If no EVM packet received within this many seconds, record NaN
EVM_STALE_SEC = 2.0

# -------------------------------------------------
# Directory and file names
# -------------------------------------------------
server_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(server_dir)

# -------------------------------------------------
# lib imports
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(project_dir)
sys.path.insert(0, PROJECT_ROOT)
from lib.yaml_utils import read_yaml_file  # noqa: E402

# -------------------------------------------------
# config file
# -------------------------------------------------
settings = read_yaml_file("experiment-settings.yaml")

# -------------------------------------------------
# Data directory
# -------------------------------------------------
save_dir = os.path.abspath(os.path.join(server_dir, "../../data", FOLDER))
os.makedirs(save_dir, exist_ok=True)

# ****************************************************************************************** #
#                                      INITIALIZATION                                        #
# ****************************************************************************************** #

parser = argparse.ArgumentParser(description="Record energy profiler measurements.")
parser.add_argument(
    "--duration",
    dest="duration",
    type=str,
    help="Stop recording after a duration (e.g. '3h', '30m', '45s').",
)
parser.add_argument(
    "--load-existing",
    action="store_true",
    help="Load latest *_positions.npy and *_values.npy from the save folder and plot them.",
)
parser.add_argument(
    "--evm-addr",
    dest="evm_addr",
    type=str,
    default=CLIENT_EVM_ADDR,
    help="ZMQ address of client EVM publisher, e.g. tcp://192.168.0.10:50001",
)
args = parser.parse_args()


def _parse_duration(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip().lower()
    unit = value[-1]
    if unit in {"h", "m", "s"} and len(value) > 1:
        num = float(value[:-1])
        if unit == "h":
            return num * 3600.0
        if unit == "m":
            return num * 60.0
        return num
    return float(value)


max_duration = _parse_duration(args.duration) or DEFAULT_DURATION
positioner = PositionerClient(config=settings["positioning"], backend="zmq")
scope = Scope(config=settings["scope"])

import logging  # noqa: E402

scope.logger.setLevel(logging.ERROR)

plt = TechtilePlotter(realtime=True)

positions = []
values = []
bd_power = []
evm = []  # EVM percent per sample

# NOTE: keep original behavior if you want immediate save on start.
# If you prefer not to save immediately, change to: last_save = time()
last_save = 0
stop_requested = False

# ---------------------------
# EVM receiver state (thread)
# ---------------------------
latest_evm = float("nan")
latest_evm_t = 0.0  # local receive time (server clock)


def _evm_subscriber(evm_addr: str):
    """Background thread: receive EVM from client and keep only the latest value."""
    global latest_evm, latest_evm_t
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(evm_addr)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe all

    while True:
        try:
            s = sub.recv_string()
            d = json.loads(s)
            latest_evm = float(d.get("evm_pct", float("nan")))
            latest_evm_t = time()  # time of reception on server
        except Exception:
            # keep thread alive on malformed packets / transient network issues
            continue


# Start EVM subscriber thread immediately
threading.Thread(target=_evm_subscriber, args=(args.evm_addr,), daemon=True).start()


def _get_latest_evm_or_nan() -> float:
    """Return latest EVM if fresh, else NaN."""
    if latest_evm_t <= 0:
        return float("nan")
    if (time() - latest_evm_t) > EVM_STALE_SEC:
        return float("nan")
    return float(latest_evm)


def save_data():
    """Safely save measurement data to disk."""
    print("Saving data...")
    positions_snapshot = list(positions)
    values_snapshot = list(values)
    bd_power_snapshot = list(bd_power)
    evm_snapshot = list(evm)

    if len(positions_snapshot) != len(values_snapshot):
        print(
            "Warning: positions and values length mismatch:",
            len(positions_snapshot),
            len(values_snapshot),
        )

    if len(evm_snapshot) != len(positions_snapshot):
        print(
            "Warning: evm and positions length mismatch:",
            len(evm_snapshot),
            len(positions_snapshot),
        )

    positions_path = os.path.join(save_dir, f"{TIMESTAMP}_positions.npy")
    values_path = os.path.join(save_dir, f"{TIMESTAMP}_values.npy")
    bd_power_path = os.path.join(save_dir, f"{TIMESTAMP}_bd_power.npy")
    evm_path = os.path.join(save_dir, f"{TIMESTAMP}_evm.npy")

    _atomic_save_npy(positions_path, positions_snapshot)
    _atomic_save_npy(values_path, values_snapshot)
    _atomic_save_npy(bd_power_path, bd_power_snapshot)
    _atomic_save_npy(evm_path, evm_snapshot)
    print("Data saved.")


def _atomic_save_npy(final_path, data):
    """Write to a temp file, fsync, then replace to avoid partial writes."""
    tmp_path = f"{final_path}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _handle_signal(signum, frame):
    global stop_requested
    stop_requested = True


def _save_data_safe():
    try:
        save_data()
    except Exception as e:
        print("Failed to save data on exit:", e)


atexit.register(_save_data_safe)
signal.signal(signal.SIGTERM, _handle_signal)


def _load_all_snapshots(folder_path):
    positions_files = sorted([f for f in os.listdir(folder_path) if f.endswith("_positions.npy")])
    values_files = sorted([f for f in os.listdir(folder_path) if f.endswith("_values.npy")])
    if not positions_files or not values_files:
        return []
    pos_map = {
        name[: -len("_positions.npy")]: os.path.join(folder_path, name)
        for name in positions_files
    }
    val_map = {
        name[: -len("_values.npy")]: os.path.join(folder_path, name)
        for name in values_files
    }
    bases = sorted(set(pos_map) & set(val_map))
    return [(pos_map[b], val_map[b]) for b in bases]


def _load_existing_data():
    pairs = _load_all_snapshots(save_dir)
    if not pairs:
        print("No existing position/value snapshots found to load.")
        return

    total = 0
    loaded_evm_total = 0

    for pos_path, val_path in pairs:
        base = os.path.basename(pos_path)[: -len("_positions.npy")]
        evm_path = os.path.join(save_dir, f"{base}_evm.npy")

        try:
            existing_positions = np.load(pos_path, allow_pickle=True).tolist()
            existing_values = np.load(val_path, allow_pickle=True).tolist()
        except Exception as exc:
            print(f"Failed to load existing snapshots {pos_path}, {val_path}: {exc}")
            continue

        existing_evm = None
        if os.path.exists(evm_path):
            try:
                existing_evm = np.load(evm_path, allow_pickle=True).tolist()
            except Exception as exc:
                print(f"Failed to load existing evm snapshot {evm_path}: {exc}")
                existing_evm = None

        if len(existing_positions) != len(existing_values):
            print(
                "Warning: existing positions and values length mismatch:",
                len(existing_positions),
                len(existing_values),
            )

        positions.extend(existing_positions)
        values.extend(existing_values)

        # Replay plot using original logic (power stored in "d")
        for pos, d in zip(existing_positions, existing_values):
            plt.measurements_rt(pos.x, pos.y, pos.z, d)

        total += len(existing_positions)

        # EVM: load if present; otherwise fill NaN to keep indices aligned
        if existing_evm is None:
            evm.extend([float("nan")] * len(existing_positions))
        else:
            if len(existing_evm) != len(existing_positions):
                print(
                    "Warning: existing evm and positions length mismatch:",
                    len(existing_evm),
                    len(existing_positions),
                )
                if len(existing_evm) > len(existing_positions):
                    existing_evm = existing_evm[: len(existing_positions)]
                else:
                    existing_evm = existing_evm + [float("nan")] * (
                        len(existing_positions) - len(existing_evm)
                    )

            evm.extend(existing_evm)
            loaded_evm_total += len(existing_evm)

    print(f"Loaded {total} existing samples from {save_dir}.")
    if loaded_evm_total > 0:
        print(f"Loaded EVM for {loaded_evm_total} samples.")
    else:
        print("No existing EVM snapshots found; filled with NaN.")


# ****************************************************************************************** #
#                                           MAIN                                             #
# ****************************************************************************************** #
class scope_data(object):
    def __init__(self, pwr_pw):
        self.pwr_pw = pwr_pw


try:
    print("Starting positioner and RFEP...")
    print(f"Subscribing EVM from: {args.evm_addr}")
    positioner.start()

    if args.load_existing:
        _load_existing_data()

    start_time = time()

    while True:
        vals = scope.get_power_Watt() * 1e12
        pos = positioner.get_data()

        d1 = scope_data(vals[0])
        d2 = scope_data(vals[1])  # kept, even if not used later

        if vals[0] is not None and pos is not None:
            positions.append(pos)
            values.append(d1)
            bd_power.append(vals[1])

            # NEW: append EVM (percent) received from client (latest value)
            evm.append(_get_latest_evm_or_nan())

            plt.measurements_rt(pos.x, pos.y, pos.z, d1.pwr_pw / 1e6)
            print("x", end="", flush=True)
            print(vals[1])
        else:
            print(".", end="", flush=True)

        # Periodic autosave
        if time() - last_save >= SAVE_EVERY:
            _save_data_safe()
            last_save = time()

        sleep(0.1)
        if max_duration is not None and time() - start_time >= max_duration:
            print(f"Reached configured duration ({max_duration:.0f} s). Stopping.")
            break
        if stop_requested:
            print("Stop requested. Exiting loop...")
            break

except KeyboardInterrupt:
    print("\nCtrl+C received. Stopping measurement...")

except Exception as e:
    print("Unexpected error:", e)
    raise

finally:
    # ****************************************************************************************** #
    #                                           CLEANUP                                          #
    # ****************************************************************************************** #
    print("Cleaning up...")

    _save_data_safe()

    try:
        positioner.stop()
    except Exception:
        pass

    print("Shutdown complete.")
    sys.exit(0)
