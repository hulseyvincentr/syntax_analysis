#!/usr/bin/env python3
"""
Batch pre/post-procedure UMAP QC for all AFP lesion-study birds.

For each bird folder, this script looks for:

    /Volumes/my_own_SSD/updated_AreaX_outputs/<BIRD>/<BIRD>.npz

and uses the existing arrays:
    embedding_outputs : existing 2-D UMAP coordinates
    hdbscan_labels    : existing HDBSCAN cluster labels
    file_indices      : point -> source file index
    file_map          : file index -> source filename

Colors are loaded from:
    /Volumes/my_own_SSD/updated_AreaX_outputs/fixed_label_colors_50.json

Pre/post split dates are read directly from:
    /Users/mirandahulsey-vincent/Desktop/AFP_lesion_jsons/
        AFP_lesion_bird_metadata.json

For each bird, the split date is:
    metadata[bird]["lesion_surgery_date"]

The metadata are also used to label sham birds as "sham" and all other
lesion groups as "lesion".

OUTPUTS
-------
All generated plots and CSVs are written outside the Git repo:

    ~/Desktop/all_birds_pre_post_umap_QC/
        all_birds_qc_run_summary.csv
        USA5288/
            USA5288_pre_post_side_by_side.png
            USA5288_pre_gray_post_colored_overlay.png
            USA5288_cluster_centroids.png
            USA5288_cluster_shift_summary.csv
            USA5288_recording_date_summary.csv
        <other bird>/
            ...

The script processes birds sequentially so that only one large NPZ is loaded
at a time. If one bird fails, that error is recorded and the script continues
to the next bird.

Bird folders without a matching metadata entry are skipped rather than assigned
a guessed split date. Metadata birds without a discoverable bird-level NPZ are
reported in the Terminal output.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import re
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_ROOT_DIR = Path(
    "/Volumes/my_own_SSD/updated_AreaX_outputs"
)

DEFAULT_COLOR_JSON = (
    DEFAULT_ROOT_DIR / "fixed_label_colors_50.json"
)

DEFAULT_METADATA_JSON = Path(
    "/Users/mirandahulsey-vincent/Desktop/AFP_lesion_jsons/"
    "AFP_lesion_bird_metadata.json"
)

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "Desktop" / "all_birds_pre_post_umap_QC"
)

POINT_SIZE = 2.0
POINT_ALPHA = 0.55

PRE_GRAY = "0.72"
PRE_ALPHA = 0.18
POST_ALPHA = 0.65

ROBUST_CROP_PERCENT = 99.0

MIN_PRE_POINTS_FOR_CENTROID = 20
MIN_POST_POINTS_FOR_CENTROID = 20

NOISE_LABEL = -1
SHOW_NOISE = True


# ============================================================
# NPZ discovery
# ============================================================

def discover_bird_npzs(root_dir: Path) -> dict[str, Path]:
    """
    Discover one bird-level NPZ per immediate subfolder.

    Preferred organization:
        root_dir/BIRD/BIRD.npz

    If that exact file is absent but the folder contains exactly one .npz
    directly inside it, that single file is used as a fallback.

    This intentionally does NOT recursively search nested segment folders,
    which avoids accidentally collecting thousands of source-segment NPZs.
    """
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Root directory does not exist:\n  {root_dir}"
        )

    found: dict[str, Path] = {}

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        bird = folder.name
        preferred = folder / f"{bird}.npz"

        if preferred.exists():
            found[bird] = preferred
            continue

        direct_npzs = sorted(folder.glob("*.npz"))

        if len(direct_npzs) == 1:
            found[bird] = direct_npzs[0]

    return found


# ============================================================
# Bird metadata / split dates
# ============================================================

def load_bird_metadata(json_path: Path) -> dict[str, dict]:
    """
    Load the AFP lesion bird metadata JSON and parse each bird's
    lesion_surgery_date.

    Expected structure:
        {
            "USA5288": {
                "lesion_group": "medial_and_lateral",
                "lesion_surgery_date": "2024-04-09",
                ...
            },
            ...
        }

    A private helper field, "_split_date", is added as a normalized
    pandas Timestamp. A private "_event_label" field is also added:
        sham_saline / sham extent -> "sham"
        all other groups          -> "lesion"
    """
    if not json_path.exists():
        raise FileNotFoundError(
            "Bird metadata JSON was not found:\n"
            f"  {json_path}"
        )

    with open(json_path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            "Bird metadata JSON must contain a top-level object "
            "mapping bird IDs to metadata dictionaries."
        )

    parsed: dict[str, dict] = {}

    for bird, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(
                f"Metadata for {bird} is not a dictionary."
            )

        date_raw = info.get("lesion_surgery_date")

        if date_raw is None or str(date_raw).strip() == "":
            raise ValueError(
                f"{bird} is missing lesion_surgery_date in:\n"
                f"  {json_path}"
            )

        split_date = pd.to_datetime(
            str(date_raw).strip(),
            errors="coerce",
        )

        if pd.isna(split_date):
            raise ValueError(
                f"{bird} has an invalid lesion_surgery_date: "
                f"{date_raw!r}"
            )

        split_date = pd.Timestamp(split_date).normalize()

        lesion_group = str(
            info.get("lesion_group", "")
        ).strip()

        lesion_extent_class = str(
            info.get("lesion_extent_class", "")
        ).strip()

        if (
            lesion_group == "sham_saline"
            or lesion_extent_class == "sham"
        ):
            event_label = "sham"
        else:
            event_label = "lesion"

        enriched = dict(info)
        enriched["_split_date"] = split_date
        enriched["_event_label"] = event_label

        parsed[str(bird).strip()] = enriched

    return parsed


# ============================================================
# Date handling
# ============================================================

_SERIAL_RE = re.compile(
    r"^[^_]+_([0-9]+(?:\.[0-9]+)?)_"
)


def serial_date_from_filename(filename: str) -> pd.Timestamp:
    """
    Extract Excel/MATLAB-style serial date from filenames such as:

        USA5288_45382.42553504_3_31_11_49_13_segment_0.npz

    Returns a normalized pandas Timestamp.
    """
    basename = Path(filename).name
    match = _SERIAL_RE.search(basename)

    if match is None:
        raise ValueError(
            "Could not extract serial date from filename:\n"
            f"  {basename}"
        )

    serial = float(match.group(1))

    dt = (
        pd.Timestamp("1899-12-30")
        + pd.to_timedelta(serial, unit="D")
    )

    return dt.normalize()


def unpack_file_map_value(mapped_value) -> str:
    """
    file_map values are often one-element tuples, but tolerate strings or
    NumPy containers as well.
    """
    if isinstance(mapped_value, str):
        return mapped_value

    if isinstance(mapped_value, np.ndarray):
        if mapped_value.size == 0:
            raise ValueError("Encountered empty file_map array value.")
        return str(mapped_value.flat[0])

    if isinstance(mapped_value, (tuple, list)):
        if len(mapped_value) == 0:
            raise ValueError("Encountered empty file_map sequence value.")
        return str(mapped_value[0])

    return str(mapped_value)


def build_point_dates(file_indices, file_map):
    """
    Convert every UMAP point's source-file index into a recording date.
    """
    date_by_file_index = {}

    for idx, mapped_value in file_map.items():
        filename = unpack_file_map_value(mapped_value)

        date_by_file_index[int(idx)] = (
            serial_date_from_filename(filename)
        )

    try:
        dates = [
            date_by_file_index[int(i)]
            for i in file_indices
        ]
    except KeyError as exc:
        raise KeyError(
            f"file_indices references index {exc.args[0]}, "
            "but that index is absent from file_map."
        ) from exc

    return pd.DatetimeIndex(
        pd.to_datetime(dates)
    )


# ============================================================
# Figure 2 colors
# ============================================================

def load_fixed_colors(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(
            "Figure 2 color JSON was not found:\n"
            f"  {json_path}"
        )

    with open(json_path, "r") as f:
        return json.load(f)


def build_figure2_color_map(
    labels,
    fixed_colors: dict,
) -> dict[int, str]:
    """
    Map HDBSCAN syllable labels to the fixed Figure 2 palette.
    """
    color_map: dict[int, str] = {}

    for label in np.sort(np.unique(labels)):
        label = int(label)

        if label == NOISE_LABEL:
            color_map[label] = "0.75"
            continue

        key = str(label)

        if key not in fixed_colors:
            raise KeyError(
                f"Syllable label {label} is missing from "
                "fixed_label_colors_50.json."
            )

        color_map[label] = fixed_colors[key]

    return color_map


# ============================================================
# Plot helpers
# ============================================================

def robust_axis_limits(coords, percent=99.0):
    if len(coords) == 0:
        raise ValueError(
            "No finite pre/post coordinates available for plotting."
        )

    tail = (100.0 - percent) / 2.0

    x_low, x_high = np.percentile(
        coords[:, 0],
        [tail, 100.0 - tail],
    )

    y_low, y_high = np.percentile(
        coords[:, 1],
        [tail, 100.0 - tail],
    )

    x_span = max(x_high - x_low, 1e-9)
    y_span = max(y_high - y_low, 1e-9)

    x_pad = 0.04 * x_span
    y_pad = 0.04 * y_span

    return (
        (x_low - x_pad, x_high + x_pad),
        (y_low - y_pad, y_high + y_pad),
    )


def format_axis(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_colored_points(
    ax,
    coords,
    labels,
    mask,
    color_map,
    alpha=POINT_ALPHA,
):
    sub_coords = coords[mask]
    sub_labels = labels[mask]

    for label in np.unique(sub_labels):
        label = int(label)
        label_mask = sub_labels == label

        if label == NOISE_LABEL:
            if not SHOW_NOISE:
                continue
            alpha_this = 0.12
        else:
            alpha_this = alpha

        ax.scatter(
            sub_coords[label_mask, 0],
            sub_coords[label_mask, 1],
            s=POINT_SIZE,
            c=[color_map[label]],
            alpha=alpha_this,
            linewidths=0,
            rasterized=True,
        )


# ============================================================
# Cluster statistics
# ============================================================

def cluster_shift_summary(
    coords,
    labels,
    pre_mask,
    post_mask,
):
    rows = []

    for label in np.sort(np.unique(labels)):
        label = int(label)

        if label == NOISE_LABEL:
            continue

        pre_xy = coords[
            pre_mask & (labels == label)
        ]

        post_xy = coords[
            post_mask & (labels == label)
        ]

        n_pre = len(pre_xy)
        n_post = len(post_xy)

        if n_pre == 0 or n_post == 0:
            continue

        pre_centroid = pre_xy.mean(axis=0)
        post_centroid = post_xy.mean(axis=0)

        displacement = np.linalg.norm(
            post_centroid - pre_centroid
        )

        pre_radius = np.mean(
            np.linalg.norm(
                pre_xy - pre_centroid,
                axis=1,
            )
        )

        post_radius = np.mean(
            np.linalg.norm(
                post_xy - post_centroid,
                axis=1,
            )
        )

        rows.append(
            {
                "cluster": label,
                "n_pre": n_pre,
                "n_post": n_post,
                "pre_centroid_x": pre_centroid[0],
                "pre_centroid_y": pre_centroid[1],
                "post_centroid_x": post_centroid[0],
                "post_centroid_y": post_centroid[1],
                "centroid_displacement": displacement,
                "pre_mean_radius": pre_radius,
                "post_mean_radius": post_radius,
                "spread_ratio_post_over_pre":
                    post_radius / pre_radius
                    if pre_radius > 0
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Figures
# ============================================================

def make_side_by_side(
    bird,
    split_date,
    event_label,
    output_dir,
    coords,
    labels,
    pre_mask,
    post_mask,
    color_map,
    xlim,
    ylim,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )

    draw_colored_points(
        axes[0],
        coords,
        labels,
        pre_mask,
        color_map,
    )

    draw_colored_points(
        axes[1],
        coords,
        labels,
        post_mask,
        color_map,
    )

    format_axis(
        axes[0],
        xlim,
        ylim,
        (
            f"{bird} pre-{event_label}\n"
            f"n = {pre_mask.sum():,} points"
        ),
    )

    format_axis(
        axes[1],
        xlim,
        ylim,
        (
            f"{bird} post-{event_label}\n"
            f"n = {post_mask.sum():,} points"
        ),
    )

    fig.suptitle(
        f"Split date: {split_date.date()}",
        fontsize=10,
    )

    out = (
        output_dir
        / f"{bird}_pre_post_side_by_side.png"
    )

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


def make_overlay(
    bird,
    event_label,
    output_dir,
    coords,
    labels,
    pre_mask,
    post_mask,
    color_map,
    xlim,
    ylim,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 7.0),
        constrained_layout=True,
    )

    pre_xy = coords[pre_mask]

    ax.scatter(
        pre_xy[:, 0],
        pre_xy[:, 1],
        s=POINT_SIZE,
        c=PRE_GRAY,
        alpha=PRE_ALPHA,
        linewidths=0,
        rasterized=True,
    )

    draw_colored_points(
        ax,
        coords,
        labels,
        post_mask,
        color_map,
        alpha=POST_ALPHA,
    )

    format_axis(
        ax,
        xlim,
        ylim,
        (
            f"{bird}: pre-{event_label} gray, "
            f"post-{event_label} cluster-colored"
        ),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PRE_GRAY,
            markeredgecolor=PRE_GRAY,
            label=f"Pre-{event_label}",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            label=f"Post-{event_label} (cluster-colored)",
            markersize=7,
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
    )

    out = (
        output_dir
        / f"{bird}_pre_gray_post_colored_overlay.png"
    )

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


def make_centroid_plot(
    bird,
    event_label,
    output_dir,
    coords,
    color_map,
    summary,
    xlim,
    ylim,
):
    """
    Pre centroid  = larger open circle.
    Post centroid = smaller filled square.

    Markers are plotted at the true centroid positions. If they coincide
    exactly, the larger open circle remains visible around the square.
    """
    fig, ax = plt.subplots(
        figsize=(7.5, 7.0),
        constrained_layout=True,
    )

    # Faint full-UMAP context.
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.0,
        c="0.87",
        alpha=0.10,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    for _, row in summary.iterrows():
        if (
            row["n_pre"] < MIN_PRE_POINTS_FOR_CENTROID
            or row["n_post"] < MIN_POST_POINTS_FOR_CENTROID
        ):
            continue

        label = int(row["cluster"])
        color = color_map[label]

        x0 = row["pre_centroid_x"]
        y0 = row["pre_centroid_y"]

        x1 = row["post_centroid_x"]
        y1 = row["post_centroid_y"]

        ax.scatter(
            x0,
            y0,
            s=95,
            marker="o",
            facecolors="white",
            edgecolors=[color],
            linewidths=1.8,
            zorder=5,
        )

        ax.scatter(
            x1,
            y1,
            s=42,
            marker="s",
            c=[color],
            edgecolors="black",
            linewidths=0.55,
            zorder=6,
        )

    format_axis(
        ax,
        xlim,
        ylim,
        (
            f"{bird} pre- and post-{event_label} "
            "cluster centroids"
        ),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="0.25",
            markeredgewidth=1.8,
            label=f"Pre-{event_label} centroid",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="0.25",
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=f"Post-{event_label} centroid",
            markersize=6,
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
    )

    out = (
        output_dir
        / f"{bird}_cluster_centroids.png"
    )

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


# ============================================================
# One bird
# ============================================================

def process_bird(
    bird,
    npz_path,
    split_date,
    event_label,
    fixed_colors,
    output_root,
):
    output_dir = output_root / bird
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 72)
    print(f"{bird}")
    print("=" * 72)
    print(f"NPZ:        {npz_path}")
    print(f"Split date: {split_date.date()}")
    print(f"Event:      {event_label}")

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as z:

        required_keys = {
            "embedding_outputs",
            "hdbscan_labels",
            "file_indices",
            "file_map",
        }

        missing = required_keys - set(z.files)

        if missing:
            raise KeyError(
                "NPZ is missing required keys: "
                + ", ".join(sorted(missing))
            )

        coords = np.asarray(
            z["embedding_outputs"],
            dtype=float,
        )

        labels = np.asarray(
            z["hdbscan_labels"],
            dtype=int,
        )

        file_indices = np.asarray(
            z["file_indices"],
            dtype=int,
        )

        file_map_obj = z["file_map"]

        if file_map_obj.shape == ():
            file_map = file_map_obj.item()
        else:
            raise ValueError(
                "Expected file_map to be a scalar object array "
                "containing a dict."
            )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"embedding_outputs should be Nx2; got {coords.shape}"
        )

    n = len(coords)

    for name, arr in [
        ("hdbscan_labels", labels),
        ("file_indices", file_indices),
    ]:
        if len(arr) != n:
            raise ValueError(
                f"{name} has {len(arr):,} rows, "
                f"but embedding_outputs has {n:,}."
            )

    point_dates = build_point_dates(
        file_indices,
        file_map,
    )

    valid_coords = np.all(
        np.isfinite(coords),
        axis=1,
    )

    pre_mask = np.asarray(
        point_dates < split_date
    ) & valid_coords

    post_mask = np.asarray(
        point_dates > split_date
    ) & valid_coords

    split_day_mask = np.asarray(
        point_dates == split_date
    ) & valid_coords

    if pre_mask.sum() == 0:
        raise ValueError(
            "No pre-procedure points were found."
        )

    if post_mask.sum() == 0:
        raise ValueError(
            "No post-procedure points were found."
        )

    print(f"Loaded:      {n:,} UMAP points")
    print(f"Pre:         {pre_mask.sum():,}")
    print(f"Split day:   {split_day_mask.sum():,} excluded")
    print(f"Post:        {post_mask.sum():,}")

    # Per-date QC table.
    date_summary = (
        pd.DataFrame(
            {
                "date": point_dates,
                "period": np.where(
                    point_dates < split_date,
                    "pre",
                    np.where(
                        point_dates > split_date,
                        "post",
                        "split_day",
                    ),
                ),
            }
        )
        .groupby(["date", "period"])
        .size()
        .reset_index(name="n_points")
        .sort_values("date")
    )

    date_summary.to_csv(
        output_dir
        / f"{bird}_recording_date_summary.csv",
        index=False,
    )

    color_map = build_figure2_color_map(
        labels,
        fixed_colors,
    )

    analysis_mask = pre_mask | post_mask

    xlim, ylim = robust_axis_limits(
        coords[analysis_mask],
        ROBUST_CROP_PERCENT,
    )

    summary = cluster_shift_summary(
        coords,
        labels,
        pre_mask,
        post_mask,
    )

    if not summary.empty:
        summary = summary.sort_values(
            "centroid_displacement",
            ascending=False,
        )

    summary.to_csv(
        output_dir
        / f"{bird}_cluster_shift_summary.csv",
        index=False,
    )

    side_path = make_side_by_side(
        bird,
        split_date,
        event_label,
        output_dir,
        coords,
        labels,
        pre_mask,
        post_mask,
        color_map,
        xlim,
        ylim,
    )

    overlay_path = make_overlay(
        bird,
        event_label,
        output_dir,
        coords,
        labels,
        pre_mask,
        post_mask,
        color_map,
        xlim,
        ylim,
    )

    centroid_path = make_centroid_plot(
        bird,
        event_label,
        output_dir,
        coords,
        color_map,
        summary,
        xlim,
        ylim,
    )

    print("Saved:")
    print(f"  {side_path}")
    print(f"  {overlay_path}")
    print(f"  {centroid_path}")

    return {
        "bird": bird,
        "npz_path": str(npz_path),
        "split_date": split_date.date().isoformat(),
        "event_label": event_label,
        "n_total_points": int(n),
        "n_pre_points": int(pre_mask.sum()),
        "n_split_day_points": int(split_day_mask.sum()),
        "n_post_points": int(post_mask.sum()),
        "n_unique_hdbscan_labels":
            int(len(np.unique(labels))),
        "n_centroid_clusters":
            int(len(summary)),
        "status": "ok",
        "error": "",
        "output_dir": str(output_dir),
    }


# ============================================================
# CLI / batch
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create pre/post UMAP QC plots for every discovered bird NPZ, "
            "using lesion_surgery_date from the AFP lesion metadata JSON."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=(
            "Root updated_AreaX_outputs directory. "
            f"Default: {DEFAULT_ROOT_DIR}"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_JSON,
        help=(
            "AFP lesion bird metadata JSON containing lesion_surgery_date. "
            f"Default: {DEFAULT_METADATA_JSON}"
        ),
    )

    parser.add_argument(
        "--colors",
        type=Path,
        default=DEFAULT_COLOR_JSON,
        help=(
            "Figure 2 fixed label-color JSON. "
            f"Default: {DEFAULT_COLOR_JSON}"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Desktop/output directory containing one folder per bird. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--birds",
        nargs="+",
        default=None,
        help=(
            "Optional subset of bird IDs to process, e.g. "
            "--birds USA5288 USA5325 R08"
        ),
    )

    return parser.parse_args()


def empty_run_row(
    bird,
    npz_path,
    status,
    error,
    metadata_entry=None,
    output_dir="",
):
    """
    Construct a consistent run-summary row for skipped/error birds.
    """
    metadata_entry = metadata_entry or {}

    split_date = metadata_entry.get("_split_date")
    if isinstance(split_date, pd.Timestamp):
        split_date_text = split_date.date().isoformat()
    else:
        split_date_text = ""

    return {
        "bird": bird,
        "npz_path": str(npz_path),
        "split_date": split_date_text,
        "event_label": metadata_entry.get("_event_label", ""),
        "lesion_group": metadata_entry.get("lesion_group", ""),
        "lesion_group_detailed":
            metadata_entry.get("lesion_group_detailed", ""),
        "lesion_extent_class":
            metadata_entry.get("lesion_extent_class", ""),
        "lesion_hit_type":
            metadata_entry.get("lesion_hit_type", ""),
        "n_total_points": np.nan,
        "n_pre_points": np.nan,
        "n_split_day_points": np.nan,
        "n_post_points": np.nan,
        "n_unique_hdbscan_labels": np.nan,
        "n_centroid_clusters": np.nan,
        "status": status,
        "error": error,
        "output_dir": str(output_dir),
    }


def main():
    args = parse_args()

    discovered = discover_bird_npzs(
        args.root
    )

    if not discovered:
        raise RuntimeError(
            "No bird-level NPZ files were discovered under:\n"
            f"  {args.root}"
        )

    metadata = load_bird_metadata(
        args.metadata
    )

    fixed_colors = load_fixed_colors(
        args.colors
    )

    print(
        f"\nDiscovered {len(discovered)} bird NPZ file(s)."
    )
    print(
        f"Loaded metadata for {len(metadata)} bird(s) from:\n"
        f"  {args.metadata}"
    )

    # Report metadata birds for which no bird-level NPZ was discovered.
    metadata_without_npz = sorted(
        set(metadata) - set(discovered)
    )

    if metadata_without_npz:
        print(
            "\nMetadata bird(s) with no discovered bird-level NPZ:"
        )
        for bird in metadata_without_npz:
            print(f"  - {bird}")

    selected = discovered

    if args.birds:
        requested = set(args.birds)

        unknown = sorted(
            requested - set(discovered)
        )

        if unknown:
            print(
                "\nWARNING: requested bird(s) not discovered: "
                + ", ".join(unknown)
            )

        selected = {
            bird: path
            for bird, path in discovered.items()
            if bird in requested
        }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_rows = []

    for i, (bird, npz_path) in enumerate(
        selected.items(),
        start=1,
    ):
        print(
            f"\nProcessing bird {i}/{len(selected)}: {bird}"
        )

        if bird not in metadata:
            msg = (
                "Bird is missing from AFP lesion metadata JSON; "
                "split date was not guessed."
            )

            print(f"SKIP: {msg}")

            run_rows.append(
                empty_run_row(
                    bird=bird,
                    npz_path=npz_path,
                    status="skipped",
                    error=msg,
                )
            )

            continue

        meta = metadata[bird]

        split_date = meta["_split_date"]
        event_label = meta["_event_label"]

        print(
            "Metadata:"
        )
        print(
            f"  lesion_group: {meta.get('lesion_group', '')}"
        )
        print(
            f"  surgery date: {split_date.date()}"
        )
        print(
            f"  plot label:   {event_label}"
        )

        try:
            result = process_bird(
                bird=bird,
                npz_path=npz_path,
                split_date=split_date,
                event_label=event_label,
                fixed_colors=fixed_colors,
                output_root=args.output_root,
            )

            # Add useful metadata fields to the all-birds summary CSV.
            result["lesion_group"] = meta.get(
                "lesion_group",
                "",
            )
            result["lesion_group_detailed"] = meta.get(
                "lesion_group_detailed",
                "",
            )
            result["lesion_extent_class"] = meta.get(
                "lesion_extent_class",
                "",
            )
            result["lesion_hit_type"] = meta.get(
                "lesion_hit_type",
                "",
            )

            run_rows.append(result)

        except Exception as exc:
            print(
                f"\nERROR while processing {bird}: {exc}"
            )

            traceback.print_exc()

            run_rows.append(
                empty_run_row(
                    bird=bird,
                    npz_path=npz_path,
                    status="error",
                    error=str(exc),
                    metadata_entry=meta,
                    output_dir=args.output_root / bird,
                )
            )

        finally:
            # Release references between very large bird NPZs.
            plt.close("all")
            gc.collect()

    run_summary = pd.DataFrame(run_rows)

    # Put key metadata columns near the front when possible.
    preferred_columns = [
        "bird",
        "lesion_group",
        "lesion_group_detailed",
        "lesion_extent_class",
        "lesion_hit_type",
        "split_date",
        "event_label",
        "npz_path",
        "n_total_points",
        "n_pre_points",
        "n_split_day_points",
        "n_post_points",
        "n_unique_hdbscan_labels",
        "n_centroid_clusters",
        "status",
        "error",
        "output_dir",
    ]

    if not run_summary.empty:
        remaining = [
            c for c in run_summary.columns
            if c not in preferred_columns
        ]

        run_summary = run_summary.reindex(
            columns=[
                c for c in preferred_columns
                if c in run_summary.columns
            ] + remaining
        )

    summary_path = (
        args.output_root
        / "all_birds_qc_run_summary.csv"
    )

    run_summary.to_csv(
        summary_path,
        index=False,
    )

    n_ok = int(
        (run_summary["status"] == "ok").sum()
    ) if not run_summary.empty else 0

    n_skipped = int(
        (run_summary["status"] == "skipped").sum()
    ) if not run_summary.empty else 0

    n_error = int(
        (run_summary["status"] == "error").sum()
    ) if not run_summary.empty else 0

    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"Successful: {n_ok}")
    print(f"Skipped:    {n_skipped}")
    print(f"Errors:     {n_error}")
    print(f"\nRun summary:\n  {summary_path}")
    print(f"\nOutput root:\n  {args.output_root}")


if __name__ == "__main__":
    main()
