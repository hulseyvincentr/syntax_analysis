#!/usr/bin/env python3
"""
USA5288 pre- vs post-lesion UMAP QC.

Uses the existing TweetyBERT/HDBSCAN outputs stored in USA5288.npz:
    embedding_outputs : existing 2-D UMAP coordinates
    hdbscan_labels    : existing HDBSCAN cluster labels
    fixed_label_colors_50.json : exact Figure 2 syllable-label colors
    file_indices      : point -> source file index
    file_map          : file index -> source filename

USA5288 lesion date: 2024-04-09
Pre-lesion  = dates < 2024-04-09
Post-lesion = dates > 2024-04-09
Lesion day itself is excluded.

Outputs:
    USA5288_pre_post_umap_QC/
        USA5288_pre_post_side_by_side.png
        USA5288_pre_gray_post_colored_overlay.png
        USA5288_cluster_centroids.png
        USA5288_cluster_shift_summary.csv
        USA5288_recording_date_summary.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

NPZ_PATH = Path(
    "/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz"
)

LESION_DATE = pd.Timestamp("2024-04-09")

# Exact syllable-label palette used for Figure 2.
FIG2_COLOR_JSON = Path(
    "/Volumes/my_own_SSD/updated_AreaX_outputs/fixed_label_colors_50.json"
)

OUTPUT_DIR = Path.home() / "Desktop" / "USA5288_pre_post_umap_QC"

POINT_SIZE = 2.0
POINT_ALPHA = 0.55

PRE_GRAY = "0.72"
PRE_ALPHA = 0.18
POST_ALPHA = 0.65

ROBUST_CROP_PERCENT = 99.0

MIN_PRE_POINTS_FOR_SHIFT = 20
MIN_POST_POINTS_FOR_SHIFT = 20

NOISE_LABEL = -1
SHOW_NOISE = True


# ============================================================
# Date handling
# ============================================================

def serial_date_from_filename(filename):
    """
    Extract Excel/MATLAB-style serial date from filenames like:

    USA5288_45382.42553504_3_31_11_49_13_segment_0.npz

    Returns a normalized pandas Timestamp.
    """
    serial = float(Path(filename).name.split("_")[1])

    dt = (
        pd.Timestamp("1899-12-30")
        + pd.to_timedelta(serial, unit="D")
    )

    return dt.normalize()


def build_point_dates(file_indices, file_map):
    """
    Convert every UMAP point's file index into a recording date.
    """
    date_by_file_index = {}

    for idx, mapped_value in file_map.items():
        # Values are tuples such as:
        # ('USA5288_45382...segment_0.npz',)
        filename = mapped_value[0]
        date_by_file_index[int(idx)] = serial_date_from_filename(filename)

    point_dates = pd.to_datetime(
        [date_by_file_index[int(i)] for i in file_indices]
    )

    return pd.DatetimeIndex(point_dates), date_by_file_index


# ============================================================
# HDBSCAN colors
# ============================================================

def build_figure2_color_map(labels, json_path):
    """
    Load the exact syllable-label colors used for Figure 2.

    The JSON is expected to map string labels such as "0", "1", ..., "26"
    to matplotlib-compatible colors (typically hex strings).

    HDBSCAN noise (-1) is not a syllable category and is plotted in gray.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            "Figure 2 color JSON was not found:\n"
            f"  {json_path}\n\n"
            "Confirm that the external SSD is mounted and that "
            "fixed_label_colors_50.json is present."
        )

    with open(json_path, "r") as f:
        fixed_colors = json.load(f)

    color_map = {}

    for label in np.sort(np.unique(labels)):
        label = int(label)

        if label == NOISE_LABEL:
            color_map[label] = "0.75"
            continue

        key = str(label)

        if key not in fixed_colors:
            raise KeyError(
                f"Syllable label {label} is missing from:\n"
                f"  {json_path}"
            )

        color_map[label] = fixed_colors[key]

    return color_map


# ============================================================
# Plot helpers
# ============================================================

def robust_axis_limits(coords, percent=99.0):
    tail = (100.0 - percent) / 2.0

    x_low, x_high = np.percentile(
        coords[:, 0],
        [tail, 100.0 - tail],
    )

    y_low, y_high = np.percentile(
        coords[:, 1],
        [tail, 100.0 - tail],
    )

    x_pad = 0.04 * (x_high - x_low)
    y_pad = 0.04 * (y_high - y_low)

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
            c=[color_map[int(label)]],
            alpha=alpha_this,
            linewidths=0,
            rasterized=True,
        )


# ============================================================
# Cluster statistics
# ============================================================

def cluster_shift_summary(coords, labels, pre_mask, post_mask):
    rows = []

    for label in np.sort(np.unique(labels)):
        if label == NOISE_LABEL:
            continue

        pre_xy = coords[pre_mask & (labels == label)]
        post_xy = coords[post_mask & (labels == label)]

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
                "cluster": int(label),
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
        f"USA5288 pre-lesion\nn = {pre_mask.sum():,} points",
    )

    format_axis(
        axes[1],
        xlim,
        ylim,
        f"USA5288 post-lesion\nn = {post_mask.sum():,} points",
    )

    out = OUTPUT_DIR / "USA5288_pre_post_side_by_side.png"

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


def make_overlay(
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

    # Pre-lesion: gray reference distribution
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

    # Post-lesion: existing cluster colors
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
        "USA5288: pre-lesion gray, post-lesion cluster-colored",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PRE_GRAY,
            markeredgecolor=PRE_GRAY,
            label="Pre-lesion",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            label="Post-lesion (cluster-colored)",
            markersize=7,
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
    )

    out = OUTPUT_DIR / "USA5288_pre_gray_post_colored_overlay.png"

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


def make_centroid_plot(
    coords,
    labels,
    pre_mask,
    post_mask,
    color_map,
    summary,
    xlim,
    ylim,
):
    """
    Plot pre- and post-lesion cluster centroids without arrows or labels.

    Styling
    -------
    Pre centroid:
        larger open circle with the cluster's Figure 2 color as the edge.

    Post centroid:
        smaller filled square using the same cluster color, with a thin
        black edge.

    Both markers are plotted at their TRUE centroid coordinates. If the
    centroids overlap exactly, the larger open pre-lesion circle remains
    visible around the smaller post-lesion square, so no artificial spatial
    offset is required.
    """

    fig, ax = plt.subplots(
        figsize=(7.5, 7.0),
        constrained_layout=True,
    )

    # Very faint full-UMAP context.
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
            row["n_pre"] < MIN_PRE_POINTS_FOR_SHIFT
            or row["n_post"] < MIN_POST_POINTS_FOR_SHIFT
        ):
            continue

        label = int(row["cluster"])
        color = color_map[label]

        x0 = row["pre_centroid_x"]
        y0 = row["pre_centroid_y"]

        x1 = row["post_centroid_x"]
        y1 = row["post_centroid_y"]

        # Pre-lesion centroid: larger open circle.
        # Draw first so the post square can sit on top while leaving the
        # colored ring visible if the two centroids coincide.
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

        # Post-lesion centroid: smaller filled square.
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
        "USA5288 pre- and post-lesion cluster centroids",
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
            label="Pre centroid",
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
            label="Post centroid",
            markersize=6,
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
    )

    out = OUTPUT_DIR / "USA5288_cluster_centroids.png"

    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading:")
    print(NPZ_PATH)

    with np.load(
        NPZ_PATH,
        allow_pickle=True,
    ) as z:

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

        file_map = z["file_map"].item()

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

    print(f"\nLoaded {n:,} UMAP points.")

    point_dates, date_by_file_index = build_point_dates(
        file_indices,
        file_map,
    )

    pre_mask = np.asarray(
        point_dates < LESION_DATE
    )

    post_mask = np.asarray(
        point_dates > LESION_DATE
    )

    lesion_day_mask = np.asarray(
        point_dates == LESION_DATE
    )

    valid_coords = np.all(
        np.isfinite(coords),
        axis=1,
    )

    pre_mask &= valid_coords
    post_mask &= valid_coords
    lesion_day_mask &= valid_coords

    print("\nRecording split:")
    print(
        f"  pre-lesion points:   {pre_mask.sum():,}"
    )
    print(
        f"  lesion-day points:   {lesion_day_mask.sum():,} "
        "(excluded)"
    )
    print(
        f"  post-lesion points:  {post_mask.sum():,}"
    )

    # Recording-date QC table
    date_summary = (
        pd.DataFrame(
            {
                "date": point_dates,
                "period": np.where(
                    point_dates < LESION_DATE,
                    "pre",
                    np.where(
                        point_dates > LESION_DATE,
                        "post",
                        "lesion_day",
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
        OUTPUT_DIR / "USA5288_recording_date_summary.csv",
        index=False,
    )

    # Exact Figure 2 syllable-label colors
    color_map = build_figure2_color_map(
        labels,
        FIG2_COLOR_JSON,
    )

    print("\nFigure 2 color mapping:")
    print(
        f"  unique HDBSCAN labels: {len(np.unique(labels))}"
    )
    print(
        f"  colors loaded from: {FIG2_COLOR_JSON}"
    )

    # Crop based only on observations assigned pre or post.
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

    summary = summary.sort_values(
        "centroid_displacement",
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "USA5288_cluster_shift_summary.csv",
        index=False,
    )

    side = make_side_by_side(
        coords,
        labels,
        pre_mask,
        post_mask,
        color_map,
        xlim,
        ylim,
    )

    overlay = make_overlay(
        coords,
        labels,
        pre_mask,
        post_mask,
        color_map,
        xlim,
        ylim,
    )

    centroids = make_centroid_plot(
        coords,
        labels,
        pre_mask,
        post_mask,
        color_map,
        summary,
        xlim,
        ylim,
    )

    print("\nSaved:")
    print(f"  {side.resolve()}")
    print(f"  {overlay.resolve()}")
    print(f"  {centroids.resolve()}")

    print(
        f"  {(OUTPUT_DIR / 'USA5288_cluster_shift_summary.csv').resolve()}"
    )
    print(
        f"  {(OUTPUT_DIR / 'USA5288_recording_date_summary.csv').resolve()}"
    )

    print("\nLargest centroid shifts:")

    show_cols = [
        "cluster",
        "n_pre",
        "n_post",
        "centroid_displacement",
        "spread_ratio_post_over_pre",
    ]

    print(
        summary[show_cols]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
