"""
Aggregate all position/value pairs inside each subfolder of ../data
and plot a heatmap of mean power for the concatenated samples.

Extended:
- Also loads *_evm.npy (EVM percent per sample) and plots an EVM heatmap.
- Uses a dedicated binning function compute_evm_heatmap() (does NOT reuse compute_heatmap()).
- Uses a dedicated plot function plot_evm_heatmap() (no dBm plot for EVM).
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from scipy.ndimage import distance_transform_edt
    HAS_DT = True
except Exception:
    HAS_DT = False


class scope_data(object):
    def __init__(self, pwr_pw):
        self.pwr_pw = pwr_pw


WAVELENGTH = 3e8 / 920e6  # meters
GRID_RES = 0.04 * WAVELENGTH  # meters (default; overridden by --grid-res-lambda)
SMALL_POWER_UW = 1e-8  # threshold for reporting tiny measurements (micro-watts)
ZOOM_HALF_SIZE = 0.5 * WAVELENGTH  # meters, half-width/height for target zoom plots
DEFAULT_BASELINE_FOLDER = "RANDOM-1"

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment-settings.yaml"))
CMAP = "inferno"

# Ensure pickle can resolve project modules referenced in saved arrays
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_target_location(value, source_label):
    """Parse target_location as [x, y, z?] from a string or list."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        print(f"Warning: target_location in {source_label} has unsupported type; ignoring.")
        return None
    try:
        vals = [float(p) for p in parts]
    except Exception as exc:
        print(f"Warning: failed to parse target_location in {source_label}: {exc}")
        return None
    return vals if len(vals) >= 2 else None


def load_target_from_settings(settings_path=SETTINGS_PATH):
    """Return target_location from experiment-settings.yaml as [x, y, z?]."""
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}
        target = settings.get("experiment_config", {}).get("target_location")
        return parse_target_location(target, settings_path)
    except Exception as exc:
        print(f"Failed to load target_location from {settings_path}: {exc}", file=sys.stderr)
        return None


def target_rect_from_xyz(target_xyz, rect_size=0.2 * WAVELENGTH):
    """Rectangle of fixed size (default 0.2 lambda) centered on target x/y."""
    if not target_xyz or len(target_xyz) < 2:
        return None
    tx, ty = target_xyz[0], target_xyz[1]
    half = rect_size / 2
    return (tx - half, ty - half, rect_size, rect_size)


def pw_to_dbm(pw_values):
    """Convert power in pW to dBm (returns NaN for non-positive values)."""
    pw = np.asarray(pw_values, dtype=float)
    dbm = np.full_like(pw, np.nan, dtype=float)
    valid = pw > 0
    dbm[valid] = 10 * np.log10(pw[valid] * 1e-12 / 1e-3)
    return dbm


def load_folder_config(folder_path):
    """Load optional per-folder overrides from config.yml (returns a dict or {})."""
    config_path = os.path.join(folder_path, "config.yml")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            print(f"Warning: config.yml in {folder_path} is not a mapping; ignoring.")
            return {}
        return data
    except Exception as exc:
        print(f"Warning: failed to read config.yml in {folder_path}: {exc}")
        return {}


def heatmap_to_dbm(heatmap_uw, floor_watts=1e-15):
    """
    Convert heatmap values (uW) to dBm with a small floor to avoid -inf.
    NaNs in the input stay NaN in the output.
    """
    heatmap_uw = np.asarray(heatmap_uw, dtype=float)
    power_w = heatmap_uw * 1e-6
    power_w = np.where(np.isfinite(power_w) & (power_w > 0), power_w, floor_watts)
    dbm = 10 * np.log10(power_w / 1e-3)
    dbm[~np.isfinite(heatmap_uw)] = np.nan
    return dbm


def load_folder(folder_path):
    """
    Load and concatenate all *_positions.npy, *_values.npy, optional *_bd_power.npy, optional *_evm.npy.
    Returns: positions, values, bd_power (or None), evm (float array; NaN filled if missing).
    """
    positions_parts = []
    values_parts = []
    bd_power_parts = []
    evm_parts = []

    for name in sorted(os.listdir(folder_path)):
        if not name.endswith("_positions.npy"):
            continue
        base = name[: -len("_positions.npy")]
        pos_path = os.path.join(folder_path, name)
        val_path = os.path.join(folder_path, f"{base}_values.npy")
        bd_path = os.path.join(folder_path, f"{base}_bd_power.npy")
        evm_path = os.path.join(folder_path, f"{base}_evm.npy")

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

        # bd_power (optional)
        if os.path.exists(bd_path):
            bd_arr = np.load(bd_path, allow_pickle=True)
            if len(bd_arr) != len(pos_arr):
                min_len = min(len(bd_arr), len(pos_arr))
                print(
                    f"\033[91mWarning: {base} bd_power ({len(bd_arr)}) != positions ({len(pos_arr)}); "
                    f"truncating bd_power to {min_len}\033[0m"
                )
                bd_arr = bd_arr[:min_len]
            bd_power_parts.append(bd_arr)

        # evm (optional; NaN filled if missing)
        if os.path.exists(evm_path):
            evm_arr = np.load(evm_path, allow_pickle=True).astype(float)
            if len(evm_arr) != len(pos_arr):
                min_len = min(len(evm_arr), len(pos_arr))
                print(
                    f"\033[91mWarning: {base} evm ({len(evm_arr)}) != positions ({len(pos_arr)}); "
                    f"truncating all to {min_len}\033[0m"
                )
                pos_arr = pos_arr[:min_len]
                val_arr = val_arr[:min_len]
                evm_arr = evm_arr[:min_len]
            evm_parts.append(evm_arr)
        else:
            evm_parts.append(np.full(len(pos_arr), np.nan, dtype=float))

        positions_parts.append(pos_arr)
        values_parts.append(val_arr)

    if not positions_parts:
        raise ValueError(f"No position/value pairs found in {folder_path}")

    positions = np.concatenate(positions_parts)
    values = np.concatenate(values_parts)
    bd_power = np.concatenate(bd_power_parts) if bd_power_parts else None
    evm = np.concatenate(evm_parts) if evm_parts else None

    print(f"{os.path.basename(folder_path)}: merged {len(positions_parts)} pairs, {len(positions)} samples")
    return positions, values, bd_power, evm


def filter_small_values(folder_path, positions, values, vs, threshold=SMALL_POWER_UW):
    """
    Log and drop zero or near-zero power samples (threshold in uW).
    Returns filtered positions, values, and vs arrays, plus dropped count and report.
    """
    zeros = vs == 0.0
    small = (vs > 0.0) & (vs < threshold)
    drop_mask = ~(zeros | small)

    reports = []
    if zeros.any():
        reports.append(f"{zeros.sum()} zeros")
    if small.any():
        reports.append(f"{small.sum()} below {threshold:.1e} uW (min {vs[small].min():.2e})")

    dropped = len(vs) - int(drop_mask.sum())
    report = ", ".join(reports) if reports else ""
    if dropped:
        return positions[drop_mask], values[drop_mask], vs[drop_mask], dropped, report

    return positions, values, vs, 0, report


def drop_consecutive_equal_values(positions, values):
    """
    Remove runs of consecutive measurements that have identical power.
    The same indices are removed from positions to keep arrays aligned.
    """
    if len(positions) != len(values):
        min_len = min(len(positions), len(values))
        print(f"Warning: length mismatch positions={len(positions)} values={len(values)}; truncating to {min_len}")
        positions = positions[:min_len]
        values = values[:min_len]

    keep_idx = [0]
    last_power = values[0].pwr_pw
    for idx in range(1, len(values)):
        if values[idx].pwr_pw != last_power:
            keep_idx.append(idx)
            last_power = values[idx].pwr_pw

    if len(keep_idx) == len(values):
        return positions, values, 0

    dropped = len(values) - len(keep_idx)
    return positions[keep_idx], values[keep_idx], dropped


def drop_nonincreasing_timestamps(positions, values):
    """
    Drop samples whose position timestamp does not increase vs. the previous one.
    Assumes any duplicates/non-increasing timestamps occur consecutively.
    """
    if len(positions) != len(values):
        min_len = min(len(positions), len(values))
        print(f"Warning: length mismatch positions={len(positions)} values={len(values)}; truncating to {min_len}")
        positions = positions[:min_len]
        values = values[:min_len]

    if len(positions) <= 1:
        return positions, values, 0, {"equal": 0, "decrease": 0}

    keep_idx = [0]
    equal_count = 0
    decrease_count = 0
    last_t = getattr(positions[0], "t", None)
    for idx in range(1, len(positions)):
        curr_t = getattr(positions[idx], "t", None)
        if last_t is None or curr_t is None:
            keep_idx.append(idx)
            last_t = curr_t
            continue
        if curr_t > last_t:
            keep_idx.append(idx)
            last_t = curr_t
        elif curr_t == last_t:
            equal_count += 1
        else:
            decrease_count += 1

    dropped = len(positions) - len(keep_idx)
    return positions[keep_idx], values[keep_idx], dropped, {"equal": equal_count, "decrease": decrease_count}


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


def _target_mask(x_edges, y_edges, target_rect):
    if not target_rect:
        return None
    x0, y0, w, h = target_rect
    x1, y1 = x0 + w, y0 + h
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    mask_x = (xc >= x0) & (xc <= x1)
    mask_y = (yc >= y0) & (yc <= y1)
    if not mask_x.any() or not mask_y.any():
        return None
    return np.outer(mask_x, mask_y)


def gain_stats(curr, base, x_edges, y_edges, target_rect=None):
    """Return avg/max gain (linear and dB) vs baseline; optionally within target rect."""
    mask = np.isfinite(curr) & np.isfinite(base) & (curr > 0) & (base > 0)

    def _stats(mask_local):
        if mask_local is None or not np.any(mask_local):
            return None
        ratio = curr[mask_local] / base[mask_local]
        avg_lin = float(np.mean(ratio))
        max_lin = float(np.max(ratio))
        return {
            "avg_lin": avg_lin,
            "max_lin": max_lin,
            "avg_db": 10 * np.log10(avg_lin),
            "max_db": 10 * np.log10(max_lin),
        }

    global_stats = _stats(mask)
    target_mask = _target_mask(x_edges, y_edges, target_rect)
    target_stats = _stats(mask & target_mask) if target_mask is not None else None
    return global_stats, target_stats


def compute_heatmap(xs, ys, vs, grid_res, agg="median", x_edges=None, y_edges=None):
    """Bin values onto a 2D grid and compute an aggregate power per cell."""
    if x_edges is None or y_edges is None:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        x_edges = np.arange(min_x, max_x + grid_res, grid_res)
        y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    if agg not in {"mean", "median", "max", "min"}:
        raise ValueError("agg must be one of: mean, median, max, min")
    cell_values = defaultdict(list)
    counts = np.zeros_like(heatmap, dtype=int)

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    for i_x, i_y, v in zip(xi, yi, vs):
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            cell_values[(i_x, i_y)].append(v)
            counts[i_x, i_y] += 1

    agg_funcs = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    func = agg_funcs[agg]
    for (i_x, i_y), values in cell_values.items():
        if values:
            heatmap[i_x, i_y] = float(func(values))
    return heatmap, counts, x_edges, y_edges, xi, yi


# -----------------------------
# NEW: EVM binning (NO reuse of compute_heatmap)
# -----------------------------
def compute_evm_heatmap(xs, ys, evm_pct, grid_res, agg="mean", x_edges=None, y_edges=None):
    """
    Bin EVM values (percent) onto a 2D grid and compute an aggregate EVM per cell.
    This is a standalone implementation (does NOT call compute_heatmap()).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    evm_pct = np.asarray(evm_pct, dtype=float)

    if x_edges is None or y_edges is None:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        x_edges = np.arange(min_x, max_x + grid_res, grid_res)
        y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    counts = np.zeros_like(heatmap, dtype=int)

    if agg not in {"mean", "median", "max", "min"}:
        raise ValueError("agg must be one of: mean, median, max, min")

    agg_funcs = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    func = agg_funcs[agg]

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    cell_values = defaultdict(list)
    for i_x, i_y, v in zip(xi, yi, evm_pct):
        if not np.isfinite(v):
            continue
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            cell_values[(i_x, i_y)].append(float(v))
            counts[i_x, i_y] += 1

    for (i_x, i_y), vals in cell_values.items():
        if vals:
            heatmap[i_x, i_y] = float(func(vals))

    return heatmap, counts, x_edges, y_edges, xi, yi


def fill_empty_cells_nearest(heatmap):
    """Fill NaN cells with nearest non-NaN value using distance transform."""
    if not HAS_DT:
        print("Warning: scipy.ndimage.distance_transform_edt not available; cannot fill empty cells.")
        return heatmap
    mask = np.isnan(heatmap)
    if not mask.any():
        return heatmap
    filled = heatmap.copy()
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled[mask] = heatmap[tuple(idx[:, mask])]
    return filled


def plot_heatmap(
    folder,
    heatmap,
    counts,
    x_edges,
    y_edges,
    recent_cells=None,
    target_rect=None,
    agg="mean",
    cmin=None,
    cmax=None,
    vmin=None,
    vmax=None,
    show=True,
    save_bitmap=False,
    png_name="heatmap.png",
    bitmap_name="heatmap_bitmap.png",
):
    """Render power heatmaps with axes in meters (linear uW and dBm)."""

    def _draw(ax, add_axes=True):
        imshow_kwargs = dict(
            origin="lower",
            cmap=CMAP,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        if cmin is not None:
            imshow_kwargs["vmin"] = cmin
        if cmax is not None:
            imshow_kwargs["vmax"] = cmax
        img = ax.imshow(heatmap.T, **imshow_kwargs)
        ax.set_aspect("equal", adjustable="box")
        if add_axes:
            agg_label = "Median" if agg == "median" else "Mean"
            ax.set_title(f"{os.path.basename(folder)} | {agg_label.lower()} power per cell [uW]")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            cbar = plt.colorbar(img, ax=ax)
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
        if target_rect and add_axes:
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
        if not add_axes:
            ax.axis("off")
        return img

    fig, ax = plt.subplots()
    _draw(ax, add_axes=True)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, png_name))
    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_bitmap:
        fig2, ax2 = plt.subplots()
        _draw(ax2, add_axes=False)
        fig2.tight_layout(pad=0)
        plt.savefig(os.path.join(folder, bitmap_name), bbox_inches="tight", pad_inches=0)
        plt.close(fig2)

    # Counts heatmap
    fig_counts, ax_counts = plt.subplots()
    img_counts = ax_counts.imshow(
        counts.T,
        origin="lower",
        cmap="viridis",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax_counts.set_aspect("equal", adjustable="box")
    ax_counts.set_title(f"{os.path.basename(folder)} | samples per cell")
    ax_counts.set_xlabel("x [m]")
    ax_counts.set_ylabel("y [m]")
    cbar_counts = fig_counts.colorbar(img_counts, ax=ax_counts)
    cbar_counts.ax.set_ylabel("Samples per cell")
    if target_rect and show:
        x0, y0, w, h = target_rect
        ax_counts.add_patch(
            plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="green", linewidth=2)
        )
    fig_counts.tight_layout()
    plt.savefig(os.path.join(folder, "heatmap_counts.png"))
    if save_bitmap:
        fig_counts_bitmap, ax_counts_bitmap = plt.subplots()
        ax_counts_bitmap.imshow(
            counts.T,
            origin="lower",
            cmap="viridis",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        ax_counts_bitmap.set_aspect("equal", adjustable="box")
        ax_counts_bitmap.axis("off")
        fig_counts_bitmap.tight_layout(pad=0)
        plt.savefig(
            os.path.join(folder, "heatmap_counts_bitmap.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig_counts_bitmap)
    if show:
        plt.show()
    else:
        plt.close(fig_counts)

    # dBm plot
    heatmap_dbm = heatmap_to_dbm(heatmap)
    dbm_kwargs = dict(
        origin="lower",
        cmap=CMAP,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    if vmin is not None:
        dbm_kwargs["vmin"] = vmin
    if vmax is not None:
        dbm_kwargs["vmax"] = vmax

    fig_dbm, ax_dbm = plt.subplots()
    img_dbm = ax_dbm.imshow(heatmap_dbm.T, **dbm_kwargs)
    ax_dbm.set_aspect("equal", adjustable="box")
    ax_dbm.set_title(f"{os.path.basename(folder)} | power per cell [dBm]")
    ax_dbm.set_xlabel("x [m]")
    ax_dbm.set_ylabel("y [m]")
    cbar_dbm = fig_dbm.colorbar(img_dbm, ax=ax_dbm)
    cbar_dbm.ax.set_ylabel("Power per cell [dBm]")
    if target_rect:
        x0, y0, w, h = target_rect
        ax_dbm.add_patch(
            plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="green", linewidth=2)
        )
        ax_dbm.plot(
            x0 + w / 2.0,
            y0 + h / 2.0,
            marker="x",
            color="green",
            markersize=8,
            markeredgewidth=2,
        )
    fig_dbm.tight_layout()
    dbm_png_name = png_name.replace(".png", "_dBm.png")
    plt.savefig(os.path.join(folder, dbm_png_name))
    if show:
        plt.show()
    else:
        plt.close(fig_dbm)


# -----------------------------
# NEW: EVM plot (no dBm)
# -----------------------------
def plot_evm_heatmap(
    folder,
    heatmap,
    counts,
    x_edges,
    y_edges,
    agg="mean",
    target_rect=None,
    show=True,
    png_name="evm_heatmap.png",
):
    """Plot EVM heatmap (percent) + counts heatmap. No dBm conversion."""
    agg_label = "Median" if agg == "median" else "Mean"

    fig, ax = plt.subplots()
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=CMAP,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{os.path.basename(folder)} | {agg_label.lower()} EVM per cell [%]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(f"{agg_label} EVM per cell [%]")
    if target_rect:
        x0, y0, w, h = target_rect
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="green", linewidth=2)
        )
    fig.tight_layout()
    plt.savefig(os.path.join(folder, png_name))
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig2, ax2 = plt.subplots()
    img2 = ax2.imshow(
        counts.T,
        origin="lower",
        cmap="viridis",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title(f"{os.path.basename(folder)} | EVM samples per cell")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    cbar2 = fig2.colorbar(img2, ax=ax2)
    cbar2.ax.set_ylabel("Samples per cell")
    if target_rect:
        x0, y0, w, h = target_rect
        ax2.add_patch(
            plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="green", linewidth=2)
        )
    fig2.tight_layout()
    plt.savefig(os.path.join(folder, "evm_heatmap_counts.png"))
    if show:
        plt.show()
    else:
        plt.close(fig2)


def _cell_center(i_x, i_y, x_edges, y_edges):
    """Return center coordinate for a heatmap cell index."""
    cx = (x_edges[i_x] + x_edges[i_x + 1]) / 2.0
    cy = (y_edges[i_y] + y_edges[i_y + 1]) / 2.0
    return cx, cy


def clip_heatmap_to_window(heatmap, counts, x_edges, y_edges, center_xy, half_size):
    """Return a clipped heatmap/counts/edges around center_xy with given half_size."""
    if heatmap.size == 0 or center_xy is None:
        return None
    cx, cy = center_xy
    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size

    ix_start = max(np.searchsorted(x_edges, x0, side="right") - 1, 0)
    ix_end = min(np.searchsorted(x_edges, x1, side="left"), len(x_edges) - 1)
    iy_start = max(np.searchsorted(y_edges, y0, side="right") - 1, 0)
    iy_end = min(np.searchsorted(y_edges, y1, side="left"), len(y_edges) - 1)

    if ix_end <= ix_start or iy_end <= iy_start:
        return None

    clipped_heatmap = heatmap[ix_start:ix_end, iy_start:iy_end]
    clipped_counts = counts[ix_start:ix_end, iy_start:iy_end] if counts is not None else None
    clipped_x_edges = x_edges[ix_start:ix_end + 1]
    clipped_y_edges = y_edges[iy_start:iy_end + 1]
    return clipped_heatmap, clipped_counts, clipped_x_edges, clipped_y_edges


def write_folder_log(
    folder,
    heatmap,
    counts,
    x_edges,
    y_edges,
    target_xyz,
    agg,
    bd_power_pw=None,
):
    """Write a per-folder summary stats log (power-based)."""
    log_path = os.path.join(folder, "heatmap.txt")
    if np.isfinite(heatmap).any():
        max_idx = np.nanargmax(heatmap)
        i_x, i_y = np.unravel_index(max_idx, heatmap.shape)
        max_val = float(heatmap[i_x, i_y])
        max_x, max_y = _cell_center(i_x, i_y, x_edges, y_edges)
        max_count = int(counts[i_x, i_y])
    else:
        i_x = i_y = None
        max_val = float("nan")
        max_x = max_y = float("nan")
        max_count = 0

    total_samples = int(counts.sum())
    nonzero_counts = counts[counts > 0]
    min_count = int(nonzero_counts.min()) if nonzero_counts.size else 0
    max_count_all = int(nonzero_counts.max()) if nonzero_counts.size else 0

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"folder: {os.path.basename(folder)}\n")
        fh.write(f"aggregation: {agg}\n")
        fh.write(f"grid_res_m: {GRID_RES}\n")
        if target_xyz and len(target_xyz) >= 2:
            z_val = target_xyz[2] if len(target_xyz) > 2 else "n/a"
            fh.write(f"target_location: {target_xyz[0]:.6f}, {target_xyz[1]:.6f}, {z_val}\n")
        else:
            fh.write("target_location: n/a\n")
        if bd_power_pw is not None:
            bd_vals = np.asarray(bd_power_pw, dtype=float)
            bd_valid = bd_vals[np.isfinite(bd_vals) & (bd_vals > 0)]
            if bd_valid.size:
                bd_min_dbm = float(pw_to_dbm(bd_valid.min()))
                bd_max_dbm = float(pw_to_dbm(bd_valid.max()))
                bd_mean_dbm = float(pw_to_dbm(bd_valid.mean()))
                fh.write(f"bd_power_min_dBm: {bd_min_dbm:.2f}\n")
                fh.write(f"bd_power_max_dBm: {bd_max_dbm:.2f}\n")
                fh.write(f"bd_power_mean_dBm: {bd_mean_dbm:.2f}\n")
            else:
                fh.write("bd_power_min_dBm: n/a\n")
                fh.write("bd_power_max_dBm: n/a\n")
                fh.write("bd_power_mean_dBm: n/a\n")
        fh.write(f"max_power_uW: {max_val:.6f}\n")
        fh.write(f"max_cell_center_m: {max_x:.6f}, {max_y:.6f}\n")
        if i_x is not None and i_y is not None:
            fh.write(f"max_cell_index: {i_x}, {i_y}\n")
            fh.write(f"max_cell_count: {max_count}\n")
        fh.write(f"total_samples: {total_samples}\n")
        fh.write(f"cells_total: {int(counts.size)}\n")
        fh.write(f"cells_nonzero: {int(nonzero_counts.size)}\n")
        fh.write(f"min_cell_count: {min_count}\n")
        fh.write(f"max_cell_count_all: {max_count_all}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate positions/values per data subfolder and plot heatmaps."
    )
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot heatmaps for all subfolders (default plots only the most recent).")
    parser.add_argument("--plot-movement", action="store_true",
                        help="Overlay rectangles for the last 5 visited grid cells.")
    parser.add_argument("--drop-consecutive-equal", action="store_true",
                        help="Filter out consecutive samples with identical power values.")
    parser.add_argument("--drop-duplicate-timestamps", dest="drop_duplicate_timestamps",
                        action="store_true",
                        help="Enable filtering of samples where position timestamp does not increase (consecutive duplicates).")
    parser.set_defaults(drop_duplicate_timestamps=False)
    parser.add_argument("--save-only", action="store_true",
                        help="Save plots to disk without displaying them.")
    parser.add_argument("--agg", choices=["mean", "median", "max", "min"], default="mean",
                        help="Aggregation used for heatmap cells (default: mean).")
    parser.add_argument("--baseline-folder", default=None,
                        help="Folder name to use as baseline for delta heatmap (default: RANDOM-1). Set empty to disable.")
    parser.add_argument("--fill-empty", action="store_true",
                        help="Interpolate/fill empty cells with nearest non-empty value.")
    parser.add_argument("--grid-res-lambda", type=float,
                        help="Grid resolution as a fraction of wavelength (e.g., 0.08 for 0.08*lambda). Overrides default.")
    parser.add_argument("--vdmin", type=float, default=None,
                        help="Colormap minimum for baseline delta plots (dB).")
    parser.add_argument("--vdmax", type=float, default=None,
                        help="Colormap maximum for baseline delta plots (dB).")
    parser.add_argument("--cmin", type=float, default=None,
                        help="Colormap minimum for the linear uW heatmap.")
    parser.add_argument("--cmax", type=float, default=None,
                        help="Colormap maximum for the linear uW heatmap.")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Colormap minimum for the dBm plot.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Colormap maximum for the dBm plot.")
    return parser.parse_args()


def print_run_summary(args, target_rect, grid_res):
    rect_desc = (
        f"{target_rect[2]:.3f}x{target_rect[3]:.3f} m at ({target_rect[0]:.3f}, {target_rect[1]:.3f})"
        if target_rect
        else "none"
    )
    print(
        "\nRun configuration:\n"
        f"- data dir: {DATA_DIR}\n"
        f"- plot_all: {args.plot_all}\n"
        f"- save_only: {args.save_only}\n"
        f"- plot_movement: {args.plot_movement}\n"
        f"- agg: {args.agg}\n"
        f"- drop_duplicate_timestamps: {args.drop_duplicate_timestamps}\n"
        f"- drop_consecutive_equal: {args.drop_consecutive_equal}\n"
        f"- small-value filter threshold (uW): {SMALL_POWER_UW}\n"
        f"- target rectangle: {rect_desc}\n"
        f"- fill_empty: {args.fill_empty}\n"
        f"- grid_res: {grid_res:.4f} m\n"
        f"- vdmin/vdmax (baseline dB plots): {args.vdmin}/{args.vdmax}\n"
        f"- cmin/cmax (linear uW plots): {args.cmin}/{args.cmax}\n"
        f"- vmin/vmax (dBm plots): {args.vmin}/{args.vmax}\n"
    )


def main():
    args = parse_args()

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    grid_res = GRID_RES
    if args.grid_res_lambda:
        grid_res = float(args.grid_res_lambda) * WAVELENGTH

    target_vals = load_target_from_settings()
    target_rect = target_rect_from_xyz(target_vals) if target_vals else None
    print_run_summary(args, target_rect, grid_res)

    # Sort subfolders by modification time (newest first)
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
            positions, values, bd_power, evm = load_folder(folder_path)
        except ValueError as e:
            print(e)
            continue

        folder_config = load_folder_config(folder_path)
        folder_target_vals = None
        if "target_location" in folder_config:
            folder_target_vals = parse_target_location(
                folder_config.get("target_location"),
                os.path.join(folder_path, "config.yml"),
            )
        active_target_vals = folder_target_vals or target_vals
        active_target_rect = target_rect_from_xyz(active_target_vals) if active_target_vals else None

        # Optional filtering on timestamps / consecutive equals (power chain)
        start_count = len(values)
        if args.drop_duplicate_timestamps:
            positions, values, dropped_ts, _ = drop_nonincreasing_timestamps(positions, values)
            if evm is not None and len(evm) == start_count:
                # Keep EVM aligned with power filters
                keep_len = len(values)
                evm = np.asarray(evm, dtype=float)[:keep_len]

        if args.drop_consecutive_equal:
            positions, values, dropped_dup = drop_consecutive_equal_values(positions, values)
            if evm is not None:
                # Conservative: truncate to aligned length (we don't know keep indices here)
                evm = np.asarray(evm, dtype=float)[:len(values)]

        # Power values in uW
        vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)
        positions, values, vs, _, _ = filter_small_values(folder_path, positions, values, vs)

        # Align evm length after power filtering
        if evm is not None:
            evm = np.asarray(evm, dtype=float)
            if len(evm) != len(positions):
                min_len = min(len(evm), len(positions))
                evm = evm[:min_len]
                positions = positions[:min_len]
                values = values[:min_len]
                vs = vs[:min_len]
                if bd_power is not None:
                    bd_power = np.asarray(bd_power)[:min_len]

        xs = np.array([p.x for p in positions], dtype=float)
        ys = np.array([p.y for p in positions], dtype=float)

        # ---- Power heatmap (existing path) ----
        heatmap, counts, x_edges, y_edges, xi, yi = compute_heatmap(xs, ys, vs, grid_res, agg=args.agg)
        if args.fill_empty:
            heatmap = fill_empty_cells_nearest(heatmap)

        # Plot power heatmaps
        plot_heatmap(
            folder_path,
            heatmap,
            counts,
            x_edges,
            y_edges,
            recent_cells=None,
            target_rect=active_target_rect,
            agg=args.agg,
            cmin=args.cmin,
            cmax=args.cmax,
            vmin=args.vmin,
            vmax=args.vmax,
            show=not args.save_only,
            save_bitmap=False,
            png_name="heatmap.png",
            bitmap_name="heatmap_bitmap.png",
        )

        write_folder_log(
            folder_path,
            heatmap,
            counts,
            x_edges,
            y_edges,
            active_target_vals,
            args.agg,
            bd_power_pw=bd_power,
        )

        # ---- EVM heatmap (NEW; NO reuse of compute_heatmap) ----
        if evm is not None and np.isfinite(evm).any():
            valid = np.isfinite(evm)
            xs_e = xs[valid]
            ys_e = ys[valid]
            evm_v = evm[valid]

            if evm_v.size:
                evm_heatmap, evm_counts, ex_edges, ey_edges, _, _ = compute_evm_heatmap(
                    xs_e,
                    ys_e,
                    evm_v,
                    grid_res,
                    agg=args.agg,
                    x_edges=x_edges,   # align with power grid for direct comparison
                    y_edges=y_edges,
                )
                if args.fill_empty:
                    evm_heatmap = fill_empty_cells_nearest(evm_heatmap)

                plot_evm_heatmap(
                    folder_path,
                    evm_heatmap,
                    evm_counts,
                    ex_edges,
                    ey_edges,
                    agg=args.agg,
                    target_rect=active_target_rect,
                    show=not args.save_only,
                    png_name="evm_heatmap.png",
                )
            else:
                print(f"{folder_name}: EVM exists but all invalid after filtering.")
        else:
            print(f"{folder_name}: no valid EVM data found (missing file or all NaN).")

        if not args.plot_all:
            break


if __name__ == "__main__":
    main()
