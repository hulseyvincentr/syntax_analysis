#!/usr/bin/env python3
"""
Visualize pre- vs post-lesion TweetyBERT UMAP structure.

Outputs
-------
1. pre_post_umap_side_by_side.png
   Pre and post plotted separately with identical axes and cluster colors.

2. pre_gray_post_colored_overlay.png
   Pre-lesion points shown in light gray, post-lesion points colored by cluster.

3. cluster_centroid_shifts.png
   Arrows from each cluster's pre-lesion centroid to its post-lesion centroid.

4. cluster_shift_summary.csv
   Cluster-level counts, centroids, centroid displacement, and spread.

IMPORTANT
---------
Pre and post must be represented in the SAME UMAP coordinate system.
Do not fit separate UMAP models to pre and post data.

The script tries several common NPZ key names automatically. If it cannot
identify the correct arrays, it will print the available NPZ keys so that
you can set them explicitly in the CONFIG section.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

NPZ_PATH = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz")

OUTPUT_DIR = Path("./USA5288_pre_post_umap_QC")

# If your NPZ contains dates/timestamps rather than a pre/post array,
# enter the lesion date here.
#
# Example:
# LESION_DATE = "2025-03-14"
#
LESION_DATE = None

# Optional JSON mapping:
# {
#   "0": "#1f77b4",
#   "1": "#ff7f0e",
#   ...
# }
#
# Set to None to generate a deterministic discrete color map.
COLOR_MAP_JSON = None

# Plotting
POINT_SIZE = 2.0
POINT_ALPHA = 0.55
PRE_GRAY = "0.72"
PRE_ALPHA = 0.20
OVERLAY_POST_ALPHA = 0.65

# Crop to the central 99% of points on each UMAP axis.
ROBUST_CROP_PERCENT = 99.0

# Only calculate centroid shifts for clusters represented in BOTH periods.
MIN_PRE_POINTS_FOR_SHIFT = 20
MIN_POST_POINTS_FOR_SHIFT = 20

# HDBSCAN commonly uses -1 for noise.
NOISE_LABELS = {-1}

# If True, noise points are shown faintly.
SHOW_NOISE = True


# ============================================================
# OPTIONAL EXPLICIT KEY OVERRIDES
# ============================================================

UMAP_KEY = "embedding_outputs"
FEATURE_KEY = None
CLUSTER_KEY = "hdbscan_labels"
PERIOD_KEY = None
DATE_KEY = None
DAY_FROM_LESION_KEY = None


# ============================================================
# Candidate NPZ keys
# ============================================================

UMAP_CANDIDATES = [
    "umap",
    "umap_coords",
    "umap_embedding",
    "embedding_2d",
    "umap_projection",
]

FEATURE_CANDIDATES = [
    "embedding",
    "embeddings",
    "latent",
    "latents",
    "features",
    "hidden_states",
]

CLUSTER_CANDIDATES = [
    "cluster_labels",
    "hdbscan_labels",
    "labels",
    "syllable_labels",
    "cluster",
]

PERIOD_CANDIDATES = [
    "period",
    "pre_post",
    "epoch",
    "lesion_period",
    "condition",
    "is_post",
]

DATE_CANDIDATES = [
    "date",
    "dates",
    "datetime",
    "datetimes",
    "timestamp",
    "timestamps",
]

DAY_FROM_LESION_CANDIDATES = [
    "day_from_lesion",
    "days_from_lesion",
    "relative_day",
    "days_relative_to_lesion",
    "lesion_day",
]


# ============================================================
# Helpers
# ============================================================

def print_npz_inventory(npz):
    print("\nAvailable NPZ arrays:")
    print("-" * 70)

    for key in npz.files:
        arr = np.asarray(npz[key])
        print(
            f"{key:35s} "
            f"shape={str(arr.shape):18s} "
            f"dtype={arr.dtype}"
        )

    print("-" * 70)


def choose_key(npz, explicit_key, candidates, description):
    if explicit_key is not None:
        if explicit_key not in npz.files:
            raise KeyError(
                f"{description} key '{explicit_key}' was not found."
            )
        return explicit_key

    for key in candidates:
        if key in npz.files:
            print(f"{description}: using '{key}'")
            return key

    return None


def normalize_period_strings(values):
    strings = np.asarray(values).astype(str)
    cleaned = np.char.lower(np.char.strip(strings))

    pre_terms = {
        "pre",
        "pre-lesion",
        "prelesion",
        "before",
        "baseline",
        "0",
        "false",
    }

    post_terms = {
        "post",
        "post-lesion",
        "postlesion",
        "after",
        "1",
        "true",
    }

    is_pre = np.array([x in pre_terms for x in cleaned])
    is_post = np.array([x in post_terms for x in cleaned])

    if not np.all(is_pre | is_post):
        unknown = np.unique(cleaned[~(is_pre | is_post)])

        raise ValueError(
            "Could not interpret all period labels.\n"
            f"Unknown labels: {unknown}"
        )

    return is_pre, is_post


def determine_pre_post_masks(npz, n_points):
    period_key = choose_key(
        npz,
        PERIOD_KEY,
        PERIOD_CANDIDATES,
        "Period",
    )

    if period_key is not None:
        values = np.asarray(npz[period_key]).squeeze()

        if len(values) != n_points:
            raise ValueError(
                f"Period array '{period_key}' has {len(values)} rows, "
                f"but UMAP has {n_points}."
            )

        if values.dtype == bool:
            is_post = values
            is_pre = ~values
            return is_pre, is_post

        if np.issubdtype(values.dtype, np.number):
            unique = set(np.unique(values))
            if unique.issubset({0, 1}):
                is_post = values.astype(bool)
                is_pre = ~is_post
                return is_pre, is_post

        return normalize_period_strings(values)

    day_key = choose_key(
        npz,
        DAY_FROM_LESION_KEY,
        DAY_FROM_LESION_CANDIDATES,
        "Relative lesion day",
    )

    if day_key is not None:
        days = np.asarray(npz[day_key]).astype(float).squeeze()

        if len(days) != n_points:
            raise ValueError(
                f"Relative-day array '{day_key}' has {len(days)} rows, "
                f"but UMAP has {n_points}."
            )

        is_pre = days < 0
        is_post = days > 0
        return is_pre, is_post

    date_key = choose_key(
        npz,
        DATE_KEY,
        DATE_CANDIDATES,
        "Date",
    )

    if date_key is not None:
        if LESION_DATE is None:
            raise ValueError(
                f"Found dates in '{date_key}', but LESION_DATE is None.\n"
                "Enter the lesion date in the CONFIG section."
            )

        dates = pd.to_datetime(
            np.asarray(npz[date_key]).astype(str)
        )

        lesion_date = pd.Timestamp(LESION_DATE)

        if len(dates) != n_points:
            raise ValueError(
                f"Date array '{date_key}' has {len(dates)} rows, "
                f"but UMAP has {n_points}."
            )

        is_pre = dates < lesion_date
        is_post = dates > lesion_date

        return np.asarray(is_pre), np.asarray(is_post)

    raise RuntimeError(
        "\nCould not determine pre/post status.\n"
        "Set PERIOD_KEY, DAY_FROM_LESION_KEY, or DATE_KEY manually."
    )


def get_umap_coordinates(npz):
    umap_key = choose_key(
        npz,
        UMAP_KEY,
        UMAP_CANDIDATES,
        "UMAP",
    )

    if umap_key is not None:
        coords = np.asarray(npz[umap_key])

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"'{umap_key}' does not look like 2D UMAP coordinates: "
                f"{coords.shape}"
            )

        return coords[:, :2]

    feature_key = choose_key(
        npz,
        FEATURE_KEY,
        FEATURE_CANDIDATES,
        "Feature embedding",
    )

    if feature_key is None:
        raise RuntimeError(
            "Could not find UMAP coordinates or a feature matrix."
        )

    features = np.asarray(npz[feature_key])

    if features.ndim != 2:
        raise ValueError(
            f"Feature array '{feature_key}' should be 2D, "
            f"but has shape {features.shape}."
        )

    print(
        "\nNo precomputed UMAP coordinates found.\n"
        "Fitting ONE UMAP to all pre + post observations together..."
    )

    try:
        import umap
    except ImportError:
        raise ImportError(
            "Install umap-learn first:\n"
            "    conda install -c conda-forge umap-learn"
        )

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )

    return reducer.fit_transform(features)


def get_cluster_labels(npz, n_points):
    cluster_key = choose_key(
        npz,
        CLUSTER_KEY,
        CLUSTER_CANDIDATES,
        "Cluster label",
    )

    if cluster_key is None:
        warnings.warn(
            "No cluster-label array found. "
            "All observations will be treated as one cluster."
        )

        return np.zeros(n_points, dtype=int)

    labels = np.asarray(npz[cluster_key]).squeeze()

    if len(labels) != n_points:
        raise ValueError(
            f"Cluster array '{cluster_key}' has {len(labels)} rows, "
            f"but UMAP has {n_points}."
        )

    return labels


def build_color_map(labels):
    unique_clusters = [
        x for x in np.unique(labels)
        if x not in NOISE_LABELS
    ]

    if COLOR_MAP_JSON is not None:
        with open(COLOR_MAP_JSON, "r") as f:
            raw = json.load(f)

        colors = {}

        for cluster in unique_clusters:
            key = str(cluster)

            if key not in raw:
                raise KeyError(
                    f"Cluster {cluster} is missing from color map."
                )

            colors[cluster] = raw[key]

        return colors

    cmap = plt.get_cmap("tab20", max(len(unique_clusters), 1))

    return {
        cluster: cmap(i)
        for i, cluster in enumerate(unique_clusters)
    }


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


def draw_cluster_points(
    ax,
    coords,
    labels,
    mask,
    colors,
    alpha=0.6,
    gray=False,
):
    subset_labels = labels[mask]
    subset_coords = coords[mask]

    if gray:
        ax.scatter(
            subset_coords[:, 0],
            subset_coords[:, 1],
            s=POINT_SIZE,
            c=PRE_GRAY,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        return

    for cluster in np.unique(subset_labels):
        cmask = subset_labels == cluster

        if cluster in NOISE_LABELS:
            if not SHOW_NOISE:
                continue

            color = "0.82"
            cluster_alpha = 0.15
        else:
            color = colors[cluster]
            cluster_alpha = alpha

        ax.scatter(
            subset_coords[cmask, 0],
            subset_coords[cmask, 1],
            s=POINT_SIZE,
            c=[color],
            alpha=cluster_alpha,
            linewidths=0,
            rasterized=True,
        )


def format_umap_axis(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title(title)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def cluster_shift_summary(
    coords,
    labels,
    is_pre,
    is_post,
):
    rows = []

    for cluster in np.unique(labels):
        if cluster in NOISE_LABELS:
            continue

        pre_mask = is_pre & (labels == cluster)
        post_mask = is_post & (labels == cluster)

        pre_xy = coords[pre_mask]
        post_xy = coords[post_mask]

        n_pre = len(pre_xy)
        n_post = len(post_xy)

        if n_pre == 0 or n_post == 0:
            continue

        pre_centroid = np.mean(pre_xy, axis=0)
        post_centroid = np.mean(post_xy, axis=0)

        displacement = np.linalg.norm(
            post_centroid - pre_centroid
        )

        pre_spread = np.mean(
            np.linalg.norm(
                pre_xy - pre_centroid,
                axis=1,
            )
        )

        post_spread = np.mean(
            np.linalg.norm(
                post_xy - post_centroid,
                axis=1,
            )
        )

        rows.append(
            {
                "cluster": cluster,
                "n_pre": n_pre,
                "n_post": n_post,
                "pre_centroid_x": pre_centroid[0],
                "pre_centroid_y": pre_centroid[1],
                "post_centroid_x": post_centroid[0],
                "post_centroid_y": post_centroid[1],
                "centroid_displacement": displacement,
                "pre_mean_radius": pre_spread,
                "post_mean_radius": post_spread,
                "spread_ratio_post_over_pre":
                    post_spread / pre_spread
                    if pre_spread > 0
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_side_by_side(
    coords,
    labels,
    is_pre,
    is_post,
    colors,
    xlim,
    ylim,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        constrained_layout=True,
    )

    draw_cluster_points(
        axes[0],
        coords,
        labels,
        is_pre,
        colors,
        alpha=POINT_ALPHA,
    )

    draw_cluster_points(
        axes[1],
        coords,
        labels,
        is_post,
        colors,
        alpha=POINT_ALPHA,
    )

    format_umap_axis(
        axes[0],
        xlim,
        ylim,
        f"Pre-lesion\nn = {is_pre.sum():,}",
    )

    format_umap_axis(
        axes[1],
        xlim,
        ylim,
        f"Post-lesion\nn = {is_post.sum():,}",
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_overlay(
    coords,
    labels,
    is_pre,
    is_post,
    colors,
    xlim,
    ylim,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 7.0),
        constrained_layout=True,
    )

    draw_cluster_points(
        ax,
        coords,
        labels,
        is_pre,
        colors,
        alpha=PRE_ALPHA,
        gray=True,
    )

    draw_cluster_points(
        ax,
        coords,
        labels,
        is_post,
        colors,
        alpha=OVERLAY_POST_ALPHA,
    )

    format_umap_axis(
        ax,
        xlim,
        ylim,
        "Pre- vs post-lesion UMAP",
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PRE_GRAY,
            label="Pre-lesion",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="0.15",
            label="Post-lesion, cluster-colored",
            markersize=7,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        loc="best",
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_centroid_shifts(
    coords,
    labels,
    is_pre,
    is_post,
    colors,
    summary,
    xlim,
    ylim,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 7.0),
        constrained_layout=True,
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.0,
        c="0.88",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
    )

    for _, row in summary.iterrows():
        cluster = row["cluster"]

        if (
            row["n_pre"] < MIN_PRE_POINTS_FOR_SHIFT
            or row["n_post"] < MIN_POST_POINTS_FOR_SHIFT
        ):
            continue

        x0 = row["pre_centroid_x"]
        y0 = row["pre_centroid_y"]

        x1 = row["post_centroid_x"]
        y1 = row["post_centroid_y"]

        color = colors[cluster]

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linewidth=1.6,
                alpha=0.9,
            ),
        )

        ax.scatter(
            x0,
            y0,
            s=42,
            facecolors="white",
            edgecolors=[color],
            linewidths=1.5,
            zorder=5,
        )

        ax.scatter(
            x1,
            y1,
            s=42,
            c=[color],
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
        )

        ax.text(
            x1,
            y1,
            f" {cluster}",
            fontsize=7,
            color=color,
        )

    format_umap_axis(
        ax,
        xlim,
        ylim,
        "Within-cluster pre → post centroid shifts",
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="0.25",
            label="Pre centroid",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            label="Post centroid",
            markersize=7,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        loc="best",
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nLoading:\n{NPZ_PATH}\n")

    npz = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    print_npz_inventory(npz)

    coords = get_umap_coordinates(npz)

    n_points = len(coords)

    print(f"\nUMAP observations: {n_points:,}")

    labels = get_cluster_labels(
        npz,
        n_points,
    )

    is_pre, is_post = determine_pre_post_masks(
        npz,
        n_points,
    )

    finite = np.all(
        np.isfinite(coords),
        axis=1,
    )

    valid = finite & (is_pre | is_post)

    coords = coords[valid]
    labels = labels[valid]
    is_pre = is_pre[valid]
    is_post = is_post[valid]

    print(
        f"\nPre-lesion observations:  {is_pre.sum():,}"
    )

    print(
        f"Post-lesion observations: {is_post.sum():,}"
    )

    print(
        f"Clusters: "
        f"{len(set(np.unique(labels)) - NOISE_LABELS)}"
    )

    colors = build_color_map(labels)

    xlim, ylim = robust_axis_limits(
        coords,
        ROBUST_CROP_PERCENT,
    )

    summary = cluster_shift_summary(
        coords,
        labels,
        is_pre,
        is_post,
    )

    summary = summary.sort_values(
        "centroid_displacement",
        ascending=False,
    )

    summary_path = (
        OUTPUT_DIR /
        "cluster_shift_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_side_by_side(
        coords,
        labels,
        is_pre,
        is_post,
        colors,
        xlim,
        ylim,
        OUTPUT_DIR /
        "pre_post_umap_side_by_side.png",
    )

    plot_overlay(
        coords,
        labels,
        is_pre,
        is_post,
        colors,
        xlim,
        ylim,
        OUTPUT_DIR /
        "pre_gray_post_colored_overlay.png",
    )

    plot_centroid_shifts(
        coords,
        labels,
        is_pre,
        is_post,
        colors,
        summary,
        xlim,
        ylim,
        OUTPUT_DIR /
        "cluster_centroid_shifts.png",
    )

    print("\nDone.")

    print(f"\nOutputs written to:\n{OUTPUT_DIR.resolve()}")

    print("\nCluster shifts with largest centroid displacement:")

    cols = [
        "cluster",
        "n_pre",
        "n_post",
        "centroid_displacement",
        "spread_ratio_post_over_pre",
    ]

    print(
        summary[cols]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
