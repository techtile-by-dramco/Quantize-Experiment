# ****************************************************************************************** #
#                                       IMPORTS / PATHS                                      #
# ****************************************************************************************** #

from Positioner import PositionerClient
from TechtilePlotter.TechtilePlotter import TechtilePlotter
import os
import atexit
import signal
from time import sleep, time
import numpy as np
import zmq
import sys

# ****************************************************************************************** #
#                                           CONFIG                                           #
# ****************************************************************************************** #

SAVE_EVERY = 60.0  # seconds
FOLDER = "energy-ball-MAX"  # subfolder inside data/ where to save measurement data
TIMESTAMP = round(time())

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
from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

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

positioner = PositionerClient(config=settings["positioning"], backend="zmq")
rfep = RFEP(settings["ep"]["ip"], settings["ep"]["port"])

context = zmq.Context()
iq_socket = context.socket(zmq.PUB)
iq_socket.bind("tcp://*:50001")

plt = TechtilePlotter(realtime=True)

positions = []
values = []
last_save = 0
stop_requested = False


def save_data():
    """Safely save measurement data to disk."""
    print("Saving data...")
    positions_snapshot = list(positions)
    values_snapshot = list(values)

    if len(positions_snapshot) != len(values_snapshot):
        print(
            "Warning: positions and values length mismatch:",
            len(positions_snapshot),
            len(values_snapshot),
        )

    positions_path = os.path.join(save_dir, f"{TIMESTAMP}_positions.npy")
    values_path = os.path.join(save_dir, f"{TIMESTAMP}_values.npy")

    _atomic_save_npy(positions_path, positions_snapshot)
    _atomic_save_npy(values_path, values_snapshot)
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

# ****************************************************************************************** #
#                                           MAIN                                             #
# ****************************************************************************************** #

try:
    print("Starting positioner and RFEP...")
    positioner.start()

    start_time = time()

    while True:
        d = rfep.get_data()
        pos = positioner.get_data()

        # print(d, pos)

        if d is not None and pos is not None:
            positions.append(pos)
            values.append(d)

            plt.measurements_rt(
                pos.x,
                pos.y,
                pos.z,
                d.pwr_pw / 1e6  # µW
            )
            print("x", end="", flush=True)
        else:
            print(".", end="", flush=True)

        # Periodic autosave
        if time() - last_save >= SAVE_EVERY:
            _save_data_safe()
            last_save = time()

        sleep(0.1)
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

    try:
        rfep.stop()
    except Exception:
        pass

    iq_socket.close()
    context.term()

    print("Shutdown complete.")
    sys.exit(0)
