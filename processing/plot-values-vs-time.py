import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATA_DIR = "../data"
PREFIX = "adaptive-single-tone"
TIMESTAMP = 1765801272

# -------------------------------------------------
# Directory and file names
# -------------------------------------------------

server_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Server dir: {server_dir}")

# -------------------------------------------------
# lib imports
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(server_dir)
sys.path.insert(0, PROJECT_ROOT)
from lib.yaml_utils import read_yaml_file

# Data directory
DATA_DIR = os.path.abspath(os.path.join(server_dir, "../data"))

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print(f"Loading {DATA_DIR}")
values = np.load(f"{DATA_DIR}/{TIMESTAMP}-{PREFIX}-values.npy", allow_pickle=True)

print(values)

print(f"Loaded {len(values)} samples")

# -------------------------------------------------
# EXTRACT TIME + VALUE
# -------------------------------------------------

# Gebruik sample index als tijd
t = np.arange(len(values))

# Vermogen in µW
pwr_uw = np.array([v.pwr_pw / 1e6 for v in values])

# -------------------------------------------------
# PLOT
# -------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(t, pwr_uw)
plt.xlabel("Sample index")
plt.ylabel("Power [µW]")
plt.title("RFEP power over time")
plt.grid(True)
plt.tight_layout()
plt.show()
