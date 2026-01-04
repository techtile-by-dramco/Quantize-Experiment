"""
Aggregate all position/value pairs inside each subfolder of ../data
and plot a heatmap of mean power for the concatenated samples.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

wavelength = 3e8 / 920e6  # meters

GRID_RES = 0.1 * wavelength  # meters


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment-settings.yaml"))
CMAP = "inferno"

# Ensure pickle can resolve project modules referenced in saved arrays
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_target_from_settings(settings_path=SETTINGS_PATH):
    """Return target_location from experiment-settings.yaml as [x, y, z?]."""
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}
        target = settings.get("experiment_config", {}).get("target_location")
        if target is None:
            return None
        if isinstance(target, str):
            parts = [p.strip() for p in target.split(",") if p.strip()]
        elif isinstance(target, (list, tuple)):
            parts = list(target)
        else:
            return None
        vals = [float(p) for p in parts]
        return vals if len(vals) >= 2 else None
    except Exception as exc:
        print(f"Failed to load target_location from {settings_path}: {exc}", file=sys.stderr)
        return None


def target_rect_from_xyz(target_xyz, grid_res):
    """Rectangle of size grid_res centered on target x/y."""
    if not target_xyz or len(target_xyz) < 2:
        return None
    tx, ty = target_xyz[0], target_xyz[1]
    half = grid_res / 2
    return (tx - half, ty - half, grid_res, grid_res)


def load_folder(folder_path):
    """Load and concatenate all *_positions.npy and *_values.npy pairs in a folder."""
    positions_parts = []
    values_parts = []

    for name in sorted(os.listdir(folder_path)):
        if not name.endswith("_positions.npy"):
            continue
        base = name[: -len("_positions.npy")]
        pos_path = os.path.join(folder_path, name)
        val_path = os.path.join(folder_path, f"{base}_values.npy")
        if not os.path.exists(val_path):
            print(f"Skipping {base}: missing values file")
            continue
        positions_parts.append(np.load(pos_path, allow_pickle=True))
        values_parts.append(np.load(val_path, allow_pickle=True))

    if not positions_parts:
        raise ValueError(f"No position/value pairs found in {folder_path}")

    positions = np.concatenate(positions_parts)
    values = np.concatenate(values_parts)
    print(f"{os.path.basename(folder_path)}: merged {len(positions_parts)} pairs, {len(positions)} samples")
    return positions, values


def compute_heatmap(xs, ys, vs, grid_res):
    """Bin values onto a 2D grid and compute mean power per cell."""
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    x_edges = np.arange(min_x, max_x + grid_res, grid_res)
    y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    sums = np.zeros_like(heatmap, dtype=float)
    counts = np.zeros_like(heatmap, dtype=int)

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    for i_x, i_y, v in zip(xi, yi, vs):
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            sums[i_x, i_y] += v
            counts[i_x, i_y] += 1

    mask = counts > 0
    heatmap[mask] = sums[mask] / counts[mask]  # mean per cell
    return heatmap, counts, x_edges, y_edges, xi, yi


def plot_heatmap(folder, heatmap, counts, x_edges, y_edges, recent_cells=None, target_rect=None):
    """Render a heatmap with axes in meters."""
    fig, ax = plt.subplots()
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=CMAP,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_title(f"{os.path.basename(folder)} | mean power per cell [uW]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Mean power per cell [uW]")
    if recent_cells:
        for idx, (i_x, i_y) in enumerate(recent_cells):
            if 0 <= i_x < len(x_edges) - 1 and 0 <= i_y < len(y_edges) - 1:
                edgecolor = "lime" if idx == len(recent_cells) - 1 else "red"
                ax.add_patch(
                    plt.Rectangle(
                        (x_edges[i_x], y_edges[i_y]),
                        x_edges[i_x + 1] - x_edges[i_x],
                        y_edges[i_y + 1] - y_edges[i_y],
                        fill=False,
                        edgecolor=edgecolor,
                        linewidth=2,
                    )
                )
    if target_rect:
        x0, y0, w, h = target_rect
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor="cyan",
                linewidth=2,
                linestyle="--",
            )
        )
    # Optional: annotate with counts to show sample density per cell
    # for i_x in range(counts.shape[0]):
    #     for i_y in range(counts.shape[1]):
    #         cnt = counts[i_x, i_y]
    #         if cnt > 0:
    #             ax.text(
    #                 x_edges[i_x] + (x_edges[1] - x_edges[0]) / 2,
    #                 y_edges[i_y] + (y_edges[1] - y_edges[0]) / 2,
    #                 str(cnt),
    #                 color="white",
    #                 ha="center",
    #                 va="center",
    #                 fontsize=8,
    #             )
    fig.tight_layout()
    plt.savefig(os.path.join(folder, "heatmap.png"))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate positions/values per data subfolder and plot a heatmap."
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot heatmaps for all subfolders (default plots only the most recent).",
    )
    parser.add_argument(
        "--plot-movement",
        action="store_true",
        help="Overlay rectangles for the last 5 visited grid cells.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    target_vals = load_target_from_settings()
    target_rect = target_rect_from_xyz(target_vals, GRID_RES)

    # Sort subfolders by modification time (newest first) to view recent runs first
    folder_entries = []
    for name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(folder_path):
            folder_entries.append((os.path.getmtime(folder_path), name))

    if not folder_entries:
        raise ValueError(f"No subfolders found in {DATA_DIR}")

    folder_entries.sort(key=lambda x: x[0], reverse=True)
    for _, folder_name in folder_entries:
        folder_path = os.path.join(DATA_DIR, folder_name)
        try:
            positions, values = load_folder(folder_path)
        except ValueError as e:
            print(e)
            continue

        xs = np.array([p.x for p in positions], dtype=float)
        ys = np.array([p.y for p in positions], dtype=float)
        vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)  # uW

        heatmap, counts, x_edges, y_edges, xi, yi = compute_heatmap(xs, ys, vs, GRID_RES)

        recent_cells = None
        if args.plot_movement:
            # Take the last 5 distinct cells (most recent first), not just the last 5 samples.
            recent_cells = []
            seen = set()
            for cell in reversed(list(zip(xi, yi))):
                if cell in seen:
                    continue
                seen.add(cell)
                recent_cells.append(cell)
                if len(recent_cells) == 5:
                    break
            recent_cells.reverse()
        plot_heatmap(folder_path, heatmap, counts, x_edges, y_edges, recent_cells, target_rect)
        if not args.plot_all:
            break


if __name__ == "__main__":
    main()
