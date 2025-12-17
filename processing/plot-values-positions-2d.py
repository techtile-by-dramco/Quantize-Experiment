from TechtilePlotter.TechtilePlotter import TechtilePlotter
import numpy as np
from Positioner import PositionerValues
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
import os
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATA_DIR = "../data"
PREFIX = "quasi_multi_tone"
TIMESTAMP = 1765907307

to_plot = ["20241107114752", "20241107124328", "20241107091548"]
cmap = "inferno"

wavelen = 3e8 / 920e6
zoom_val = 2

# Grid cell aggregation functions
STAT_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}

# -------------------------------------------------
# Directory and file names
# -------------------------------------------------

server_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Server dir: {server_dir}")

PROJECT_ROOT = os.path.dirname(server_dir)
sys.path.insert(0, PROJECT_ROOT)

from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

DATA_DIR = os.path.abspath(os.path.join(server_dir, "../data"))


# -------------------------------------------------
def compute_heatmap(values, grid_pos_ids, reducer):
    """
    values: 1D list/array with scalar values (same indexing as positions/o_values)
    grid_pos_ids: [Nx][Ny] list of lists containing indices into values
    reducer: np.mean / np.median / np.max / np.min
    """
    heatmap = np.full((len(grid_pos_ids), len(grid_pos_ids[0])), np.nan, dtype=float)
    x_bf = None
    y_bf = None

    for i_x, grid_along_y in enumerate(grid_pos_ids):
        for i_y, grid_along_xy_ids in enumerate(grid_along_y):
            if len(grid_along_xy_ids) == 0:
                continue

            heatmap[i_x, i_y] = reducer([values[_id] for _id in grid_along_xy_ids])

            # Keep your marker logic: cell containing sample id 0
            if 0 in grid_along_xy_ids:
                x_bf = i_x
                y_bf = i_y

    return heatmap, x_bf, y_bf


def plot_heatmaps_for_stat(tp, heatmap, xi, yi, x_bf, y_bf, stat_name, cmap, zoom_val, wavelen):
    # UE marker (only if available)
    ue_position = None
    if (x_bf is not None) and (y_bf is not None):
        ue_position = Rectangle(
            ((y_bf - 0.5) * zoom_val, (x_bf - 0.5) * zoom_val),
            1,
            1,
            fill=False,
            edgecolor="red",
            lw=3,
        )

    upsampled_heatmap = zoom(heatmap, zoom=zoom_val, order=1)

    # -----------------------
    # Linear plot (uW)
    # -----------------------
    fig, ax = plt.subplots()
    ax.set_title(f"{tp} | {stat_name} | uW")
    img_lin = ax.imshow(upsampled_heatmap, cmap=cmap, origin="lower")

    ax.set_xticks(
        zoom_val * np.arange(len(xi))[::4],
        labels=[f"{(x - xi[0]) / wavelen:.2f}" for x in xi][::4],
    )
    ax.set_yticks(
        zoom_val * np.arange(len(yi))[::4],
        labels=[f"{(y - yi[0]) / wavelen:.2f}" for y in yi][::4],
    )

    if ue_position is not None:
        ax.add_patch(ue_position)

    cbar = fig.colorbar(img_lin)
    cbar.ax.set_ylabel(f"{stat_name} power [uW]")

    ax.set_xlabel("distance in wavelengths")
    ax.set_ylabel("distance in wavelengths")
    fig.tight_layout()
    plt.show()

    # -----------------------
    # Log plot (dB)
    # NOTE: Keeps your original scaling idea; label is generic dB.
    # -----------------------
    fig, ax = plt.subplots()
    ax.set_title(f"{tp} | {stat_name} | dB")

    up = upsampled_heatmap.copy()
    up[up <= 0] = np.nan  # protect log

    img_db = ax.imshow(
        10 * np.log10(up / 10e3),  # keep your original scaling
        vmin=None,
        vmax=None,
        cmap=cmap,
        origin="lower",
    )

    ax.set_xticks(
        zoom_val * np.arange(len(xi))[::4],
        labels=[f"{(x - xi[0]) / wavelen:.2f}" for x in xi][::4],
    )
    ax.set_yticks(
        zoom_val * np.arange(len(yi))[::4],
        labels=[f"{(y - yi[0]) / wavelen:.2f}" for y in yi][::4],
    )

    cbar = fig.colorbar(img_db)
    cbar.ax.set_ylabel(f"{stat_name} power [dB]")

    ax.set_xlabel("distance in wavelengths")
    ax.set_ylabel("distance in wavelengths")
    fig.tight_layout()
    plt.show()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
for tp in to_plot:
    positions = np.load(f"{DATA_DIR}/{TIMESTAMP}_{PREFIX}_positions.npy", allow_pickle=True)
    o_values = np.load(f"{DATA_DIR}/{TIMESTAMP}_{PREFIX}_values.npy", allow_pickle=True)

    print(f"Processing {len(positions)} samples for tp={tp}")

    print("CHANGE OF X POSITION DUE TO QTM position not same as antenna position")
    y_positions = [p.y + 0.1 for p in positions]

    temp_pos_list = PositionerValues(positions)
    positions_list = PositionerValues.from_xyz(
        temp_pos_list.get_x_positions(), y_positions, temp_pos_list.get_z_positions()
    )

    # power in uW
    values = np.array([v.pwr_pw / 1e6 for v in o_values], dtype=float)
    print(f"MAX POWER (raw samples): {np.max(values):.2f} uW")

    grid_pos_ids, xi, yi = positions_list.group_in_grids(
        0.03, min_x=2.7, max_x=3.9, min_y=1.25, max_y=2.4
    )

    for stat_name, reducer in STAT_FUNCS.items():
        heatmap, x_bf, y_bf = compute_heatmap(values, grid_pos_ids, reducer)
        print(f"[{tp}] {stat_name}: max(cell)={np.nanmax(heatmap):.2f} uW")

        plot_heatmaps_for_stat(
            tp=tp,
            heatmap=heatmap,
            xi=xi,
            yi=yi,
            x_bf=x_bf,
            y_bf=y_bf,
            stat_name=stat_name,
            cmap=cmap,
            zoom_val=zoom_val,
            wavelen=wavelen,
        )

    # Uncomment if you want to stop after first tp
    # exit()
