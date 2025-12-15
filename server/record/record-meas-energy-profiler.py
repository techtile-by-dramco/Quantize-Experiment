# ****************************************************************************************** #
#                                       IMPORTS / PATHS                                      #
# ****************************************************************************************** #

from Positioner import PositionerClient
from TechtilePlotter.TechtilePlotter import TechtilePlotter
import os
from time import sleep, time
import numpy as np
import zmq
import sys

# ****************************************************************************************** #
#                                           CONFIG                                           #
# ****************************************************************************************** #

SAVE_EVERY = 60.0  # seconds
PREFIX = "quasi_multi_tone"
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
save_dir = os.path.abspath(os.path.join(server_dir, "../../data"))
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


def save_data():
    """Safely save measurement data to disk."""
    print("Saving data...")
    np.save(os.path.join(save_dir, f"{TIMESTAMP}_{PREFIX}_positions.npy"), positions)
    np.save(os.path.join(save_dir, f"{TIMESTAMP}_{PREFIX}_values.npy"), values)
    print("Data saved.")


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

        if d is not None and pos is not None:
            positions.append(pos)
            values.append(d)

            plt.measurements_rt(
                pos.x,
                pos.y,
                pos.z,
                d.pwr_pw / 1e6  # µW
            )

        # Periodic autosave
        if time() - last_save >= SAVE_EVERY:
            save_data()
            last_save = time()

        sleep(0.1)

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

    try:
        save_data()
    except Exception as e:
        print("Failed to save data on exit:", e)

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
