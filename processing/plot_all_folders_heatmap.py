"""
Aggregate all position/value pairs inside each subfolder of ../data
and plot a heatmap of mean power for the concatenated samples.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

WAVELENGTH = 3e8 / 920e6  # meters

GRID_RES = 0.08 * WAVELENGTH  # meters
SMALL_POWER_UW = 1e-8  # threshold for reporting tiny measurements (micro-watts)


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


def target_rect_from_xyz(target_xyz, rect_size=0.5 * WAVELENGTH):
    """Rectangle of fixed size (default 0.5 lambda) centered on target x/y."""
    if not target_xyz or len(target_xyz) < 2:
        return None
    tx, ty = target_xyz[0], target_xyz[1]
    half = rect_size / 2
    return (tx - half, ty - half, rect_size, rect_size)


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
        pos_arr = np.load(pos_path, allow_pickle=True)
        val_arr = np.load(val_path, allow_pickle=True)
        if len(pos_arr) != len(val_arr):
            min_len = min(len(pos_arr), len(val_arr))
            print(
                f"\033[91mWarning: {base} positions ({len(pos_arr)}) != values ({len(val_arr)}); "
                f"truncating both to {min_len}\033[0m"
            )
            pos_arr = pos_arr[:min_len]
            val_arr = val_arr[:min_len]
        positions_parts.append(pos_arr)
        values_parts.append(val_arr)

    if not positions_parts:
        raise ValueError(f"No position/value pairs found in {folder_path}")

    positions = np.concatenate(positions_parts)
    values = np.concatenate(values_parts)
    print(f"{os.path.basename(folder_path)}: merged {len(positions_parts)} pairs, {len(positions)} samples")
    return positions, values


def filter_small_values(folder_path, positions, values, vs, threshold=SMALL_POWER_UW):
    """
    Log and drop zero or near-zero power samples (threshold in uW).
    Returns filtered positions, values, and vs arrays.
    """
    zeros = vs == 0.0
    small = (vs > 0.0) & (vs < threshold)
    drop_mask = ~(zeros | small)

    reports = []
    if zeros.any():
        reports.append(f"{zeros.sum()} zeros")
    if small.any():
        reports.append(f"{small.sum()} below {threshold:.1e} uW (min {vs[small].min():.2e})")

    if reports:
        dropped = len(vs) - drop_mask.sum()
        print(f"{os.path.basename(folder_path)}: {', '.join(reports)} (removed {dropped})")
        return positions[drop_mask], values[drop_mask], vs[drop_mask]

    return positions, values, vs


def drop_consecutive_equal_values(positions, values):
    """
    Remove runs of consecutive measurements that have identical power.
    The same indices are removed from positions to keep arrays aligned.
    """
    if len(positions) != len(values):
        min_len = min(len(positions), len(values))
        print(
            f"Warning: length mismatch positions={len(positions)} values={len(values)}; truncating to {min_len}"
        )
        positions = positions[:min_len]
        values = values[:min_len]

    keep_idx = [0]
    last_power = values[0].pwr_pw
    for idx in range(1, len(values)):
        if values[idx].pwr_pw != last_power:
            keep_idx.append(idx)
            last_power = values[idx].pwr_pw

    if len(keep_idx) == len(values):
        return positions, values

    print(f"Dropped {len(values) - len(keep_idx)} consecutive duplicates (power).")
    return positions[keep_idx], values[keep_idx]


def heatmap_delta_db(curr_heatmap, base_heatmap):
    """
    Compute delta in dB: 10*log10(curr) - 10*log10(base).
    Cells with non-positive or NaN values in either map become NaN.
    """
    diff = np.full_like(curr_heatmap, np.nan, dtype=float)
    valid = (
        np.isfinite(curr_heatmap)
        & np.isfinite(base_heatmap)
        & (curr_heatmap > 0)
        & (base_heatmap > 0)
    )
    if not np.any(valid):
        return diff

    curr_db = np.zeros_like(curr_heatmap, dtype=float)
    base_db = np.zeros_like(base_heatmap, dtype=float)
    curr_db[valid] = 10 * np.log10(curr_heatmap[valid])
    base_db[valid] = 10 * np.log10(base_heatmap[valid])
    diff[valid] = curr_db[valid] - base_db[valid]
    return diff


def compute_heatmap(xs, ys, vs, grid_res, agg="median", x_edges=None, y_edges=None):
    """Bin values onto a 2D grid and compute mean or median power per cell."""
    if x_edges is None or y_edges is None:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        x_edges = np.arange(min_x, max_x + grid_res, grid_res)
        y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be either 'mean' or 'median'")
    sums = np.zeros_like(heatmap, dtype=float) if agg == "mean" else None
    cell_values = defaultdict(list) if agg == "median" else None
    counts = np.zeros_like(heatmap, dtype=int)

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    for i_x, i_y, v in zip(xi, yi, vs):
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            if agg == "mean":
                sums[i_x, i_y] += v
            else:
                cell_values[(i_x, i_y)].append(v)
            counts[i_x, i_y] += 1

    mask = counts > 0
    if agg == "median":
        for (i_x, i_y), values in cell_values.items():
            heatmap[i_x, i_y] = float(np.median(values))
        heatmap[~mask] = np.nan
    else:
        heatmap[mask] = sums[mask] / counts[mask]  # mean per cell
    return heatmap, counts, x_edges, y_edges, xi, yi


def plot_heatmap(folder, heatmap, counts, x_edges, y_edges, recent_cells=None, target_rect=None, agg="mean", show=True):
    """Render a heatmap with axes in meters."""
    fig, ax = plt.subplots()
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=CMAP,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    agg_label = "Median" if agg == "median" else "Mean"
    ax.set_title(f"{os.path.basename(folder)} | {agg_label.lower()} power per cell [uW]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(f"{agg_label} power per cell [uW]")
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
                edgecolor="green",
                linewidth=2,
                # linestyle="-",
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
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_diff_heatmap(folder, baseline_name, diff_map, x_edges, y_edges, target_rect=None, show=True):
    """Plot the difference vs baseline in dB (folder - baseline) on aligned grid."""
    vmax_db = 5 * np.log10(42) # we expect a sqrt(42) ~16x power gain at most
    vmin_db = -vmax_db
    fig, ax = plt.subplots()
    img = ax.imshow(
        diff_map.T,
        origin="lower",
        cmap="coolwarm",
        vmin=vmin_db,
        vmax=vmax_db,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_title(f"{os.path.basename(folder)} - {baseline_name} | delta power per cell [dB]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if target_rect:
        x0, y0, w, h = target_rect
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
        )
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Delta vs baseline [dB]")
    fig.tight_layout()
    out_name = f"heatmap_vs_{baseline_name}_dB.png"
    plt.savefig(os.path.join(folder, out_name))
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    parser.add_argument(
        "--drop-consecutive-equal",
        action="store_true",
        help="Filter out consecutive samples with identical power values.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots to disk without displaying them.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation used for heatmap cells (default: mean).",
    )
    parser.add_argument(
        "--baseline-folder",
        default="RANDOM",
        help="Folder name to use as baseline for delta heatmap (default: RANDOM). Set empty to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    target_vals = load_target_from_settings()
    target_rect = target_rect_from_xyz(target_vals)

    # Sort subfolders by modification time (newest first) to view recent runs first
    folder_entries = []
    for name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(folder_path):
            folder_entries.append((os.path.getmtime(folder_path), name))

    if not folder_entries:
        raise ValueError(f"No subfolders found in {DATA_DIR}")

    # Precompute baseline heatmap if requested
    baseline_heatmap = baseline_x_edges = baseline_y_edges = None
    baseline_agg = "mean"
    baseline_folder_name = args.baseline_folder.strip() if args.baseline_folder else ""
    if baseline_folder_name:
        baseline_path = os.path.join(DATA_DIR, baseline_folder_name)
        if os.path.isdir(baseline_path):
            try:
                base_positions, base_values = load_folder(baseline_path)
                if args.drop_consecutive_equal:
                    base_positions, base_values = drop_consecutive_equal_values(base_positions, base_values)
                base_vs = np.array([v.pwr_pw / 1e6 for v in base_values], dtype=float)
                base_positions, base_values, base_vs = filter_small_values(
                    baseline_path, base_positions, base_values, base_vs
                )
                base_xs = np.array([p.x for p in base_positions], dtype=float)
                base_ys = np.array([p.y for p in base_positions], dtype=float)
                baseline_heatmap, _, baseline_x_edges, baseline_y_edges, _, _ = compute_heatmap(
                    base_xs, base_ys, base_vs, GRID_RES, agg=baseline_agg
                )
            except Exception as exc:
                print(f"Failed to build baseline from {baseline_path}: {exc}")
                baseline_folder_name = ""
        else:
            print(f"Baseline folder not found: {baseline_path}")
            baseline_folder_name = ""

    folder_entries.sort(key=lambda x: x[0], reverse=True)
    for _, folder_name in folder_entries:
        folder_path = os.path.join(DATA_DIR, folder_name)
        try:
            positions, values = load_folder(folder_path)
        except ValueError as e:
            print(e)
            continue

        if args.drop_consecutive_equal:
            positions, values = drop_consecutive_equal_values(positions, values)

        vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)  # uW

        positions, values, vs = filter_small_values(folder_path, positions, values, vs)

        xs = np.array([p.x for p in positions], dtype=float)
        ys = np.array([p.y for p in positions], dtype=float)

        heatmap, counts, x_edges, y_edges, xi, yi = compute_heatmap(
            xs, ys, vs, GRID_RES, agg=args.agg
        )

        # If baseline available, compute aligned heatmap and plot delta
        if baseline_heatmap is not None and baseline_x_edges is not None and baseline_y_edges is not None:
            aligned_heatmap, _, _, _, _, _ = compute_heatmap(
                xs, ys, vs, GRID_RES, agg=baseline_agg, x_edges=baseline_x_edges, y_edges=baseline_y_edges
            )
            diff_map = heatmap_delta_db(aligned_heatmap, baseline_heatmap)
            plot_diff_heatmap(
                folder_path,
                baseline_folder_name,
                diff_map,
                baseline_x_edges,
                baseline_y_edges,
                target_rect=target_rect,
                show=not args.save_only,
            )

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
        plot_heatmap(
            folder_path,
            heatmap,
            counts,
            x_edges,
            y_edges,
            recent_cells,
            target_rect,
            agg=args.agg,
            show=not args.save_only,
        )
        if not args.plot_all:
            break


if __name__ == "__main__":
    main()
