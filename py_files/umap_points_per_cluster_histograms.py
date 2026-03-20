#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_points_per_cluster_histograms.py

Create histograms of total raw points per cluster for every NPZ file in a root
folder, plus one aggregate histogram across all birds.

What it does
------------
- Finds NPZ files in a root directory
- Reads a cluster-label array from each NPZ (default: hdbscan_labels)
- Counts the number of points in each cluster (default excludes negative/noise labels)
- Saves one CSV of cluster counts per bird
- Saves one full-range histogram and one low-count zoom histogram per bird
- Saves one combined CSV and one aggregate full-range / zoom histogram across birds
- Also saves an extra focused low-end histogram (default: 0–2500) with finer bins

This script only uses total raw cluster sizes. It does NOT compute pre/post or
any equalized group sizes.

Default outputs
---------------
/Volumes/my_own_SSD/points_per_cluster_histograms/
  per_bird_csv/
    <animal_id>_cluster_point_counts.csv
  per_bird_plots/
    <animal_id>_cluster_point_count_histogram_total_raw.png
    <animal_id>_cluster_point_count_histogram_total_raw_zoom_0_to_5000.png
    <animal_id>_cluster_point_count_histogram_total_raw_zoom_0_to_2500.png
  aggregate/
    all_birds_cluster_point_counts.csv
    all_birds_cluster_point_count_histogram_total_raw.png
    all_birds_cluster_point_count_histogram_total_raw_zoom_0_to_5000.png
    all_birds_cluster_point_count_histogram_total_raw_zoom_0_to_2500.png
  run_manifest.csv

Example
-------
python umap_points_per_cluster_histograms.py \
  --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --out-dir "/Volumes/my_own_SSD/points_per_cluster_histograms" \
  --recursive \
  --cluster-key "hdbscan_labels"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _find_npz_files(root_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in root_dir.rglob("*.npz") if p.is_file())
    return sorted(p for p in root_dir.glob("*.npz") if p.is_file())


# -----------------------------------------------------------------------------
# Cluster counting
# -----------------------------------------------------------------------------


def _load_cluster_labels(npz_path: Path, cluster_key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if cluster_key not in data.files:
            raise KeyError(
                f"Cluster key '{cluster_key}' not found in {npz_path.name}. "
                f"Available keys: {list(data.files)}"
            )
        labels = np.asarray(data[cluster_key]).ravel()
    return labels


def _cluster_counts_from_labels(
    labels: np.ndarray,
    *,
    include_noise: bool,
) -> pd.DataFrame:
    if labels.size == 0:
        return pd.DataFrame(columns=["cluster", "n_points_total_raw"])

    # Try a numeric view first because HDBSCAN labels are usually integers.
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric.notna().all():
        vals = numeric.to_numpy(dtype=int)
        if not include_noise:
            vals = vals[vals >= 0]
        if vals.size == 0:
            return pd.DataFrame(columns=["cluster", "n_points_total_raw"])
        unique, counts = np.unique(vals, return_counts=True)
        df = pd.DataFrame({
            "cluster": unique.astype(int),
            "n_points_total_raw": counts.astype(int),
        })
        return df.sort_values("cluster").reset_index(drop=True)

    # Fallback for non-numeric labels.
    ser = pd.Series(labels, dtype="object")
    if not include_noise:
        ser = ser[~ser.isin([-1, "-1", "noise", "Noise", None])]
    vc = ser.value_counts(dropna=True).sort_index()
    if vc.empty:
        return pd.DataFrame(columns=["cluster", "n_points_total_raw"])
    return pd.DataFrame({
        "cluster": vc.index.astype(str),
        "n_points_total_raw": vc.values.astype(int),
    }).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Histogram helpers
# -----------------------------------------------------------------------------


def _integer_bin_edges(vals: np.ndarray, target_bin_width: int, hard_max_bins: int = 140) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    vmin = float(np.floor(np.nanmin(vals)))
    vmax = float(np.ceil(np.nanmax(vals)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([0.0, 1.0], dtype=float)
    if vmax <= vmin:
        return np.array([vmin - 0.5, vmin + 0.5], dtype=float)

    width = max(1, int(target_bin_width))
    n_bins = int(np.ceil((vmax - vmin + 1) / width))
    if n_bins > int(hard_max_bins):
        width = max(1, int(np.ceil((vmax - vmin + 1) / int(hard_max_bins))))

    start = np.floor(vmin / width) * width
    stop = np.ceil((vmax + 1) / width) * width
    edges = np.arange(start, stop + width, width, dtype=float)
    if edges.size < 2:
        edges = np.array([start, start + width], dtype=float)
    return edges


def _auto_full_bin_width(vals: np.ndarray) -> int:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 100

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    spread = max(1.0, vmax - vmin)

    # Aim for ~50-80 bins on the full histogram while keeping round integer widths.
    raw_width = spread / 60.0
    nice_choices = np.array([
        10, 20, 25, 50,
        100, 200, 250, 500,
        1000, 2000, 2500, 5000,
        10000, 20000, 25000, 50000,
    ], dtype=float)
    return int(nice_choices[np.argmin(np.abs(nice_choices - raw_width))])


def _plot_histogram(
    values: Iterable[float],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    dpi: int,
    target_bin_width: int,
    x_max: Optional[float] = None,
    add_rug: bool = True,
) -> None:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return

    if x_max is not None:
        vals = vals[vals <= float(x_max)]
        if vals.size == 0:
            return

    edges = _integer_bin_edges(vals, int(target_bin_width))

    plt.figure(figsize=(9.2, 5.8))
    counts, _, _ = plt.hist(vals, bins=edges, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Number of clusters")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    if add_rug and len(vals) <= 250:
        y_rug = np.full(len(vals), -max(float(np.max(counts)) * 0.02, 0.02))
        plt.scatter(vals, y_rug, marker="|", s=160, alpha=0.55)
        ymin, ymax = plt.ylim()
        plt.ylim(min(float(np.min(y_rug)) * 1.8, ymin), ymax)

    if x_max is not None:
        plt.xlim(left=0, right=float(x_max))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main batch logic
# -----------------------------------------------------------------------------


def _process_one_npz(
    npz_path: Path,
    *,
    cluster_key: str,
    include_noise: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    animal_id = _infer_animal_id(npz_path)

    try:
        labels = _load_cluster_labels(npz_path, cluster_key=cluster_key)
        counts_df = _cluster_counts_from_labels(labels, include_noise=include_noise)
        if counts_df.empty:
            return counts_df, {
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "ok": False,
                "message": "No valid clusters found after filtering.",
            }

        counts_df.insert(0, "animal_id", animal_id)
        counts_df.insert(1, "npz_path", str(npz_path))
        counts_df["rank_by_size_desc"] = counts_df["n_points_total_raw"].rank(
            method="first", ascending=False
        ).astype(int)
        return counts_df, {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "ok": True,
            "message": f"Found {len(counts_df)} clusters.",
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "ok": False,
            "message": f"{type(exc).__name__}: {exc}",
        }


def run_batch(
    *,
    root_dir: Path,
    out_dir: Path,
    recursive: bool,
    cluster_key: str,
    include_noise: bool,
    zoom_max: int,
    zoom_bin_width: int,
    focused_zoom_max: int,
    focused_zoom_bin_width: int,
    dpi: int,
) -> Dict[str, Path]:
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    per_bird_csv_dir = out_dir / "per_bird_csv"
    per_bird_plots_dir = out_dir / "per_bird_plots"
    aggregate_dir = out_dir / "aggregate"
    _safe_mkdir(per_bird_csv_dir)
    _safe_mkdir(per_bird_plots_dir)
    _safe_mkdir(aggregate_dir)

    npz_files = _find_npz_files(root_dir, recursive=recursive)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {root_dir}")

    all_counts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    created_paths: Dict[str, Path] = {}

    for npz_path in npz_files:
        counts_df, manifest = _process_one_npz(
            npz_path,
            cluster_key=cluster_key,
            include_noise=include_noise,
        )
        manifest_rows.append(manifest)
        animal_id = manifest["animal_id"]

        if counts_df.empty:
            continue

        counts_csv = per_bird_csv_dir / f"{animal_id}_cluster_point_counts.csv"
        counts_df.to_csv(counts_csv, index=False)
        created_paths[f"{animal_id}_csv"] = counts_csv

        vals = counts_df["n_points_total_raw"].to_numpy(dtype=float)
        full_width = _auto_full_bin_width(vals)

        full_plot = per_bird_plots_dir / f"{animal_id}_cluster_point_count_histogram_total_raw.png"
        _plot_histogram(
            vals,
            out_path=full_plot,
            title=f"{animal_id}: total raw points per cluster",
            xlabel="Total raw points per cluster",
            dpi=dpi,
            target_bin_width=full_width,
        )
        created_paths[f"{animal_id}_full_plot"] = full_plot

        zoom_plot = per_bird_plots_dir / f"{animal_id}_cluster_point_count_histogram_total_raw_zoom_0_to_{int(zoom_max)}.png"
        _plot_histogram(
            vals,
            out_path=zoom_plot,
            title=f"{animal_id}: total raw points per cluster (zoom 0–{int(zoom_max)})",
            xlabel="Total raw points per cluster",
            dpi=dpi,
            target_bin_width=zoom_bin_width,
            x_max=float(zoom_max),
        )
        created_paths[f"{animal_id}_zoom_plot"] = zoom_plot

        focused_zoom_plot = per_bird_plots_dir / (
            f"{animal_id}_cluster_point_count_histogram_total_raw_zoom_0_to_{int(focused_zoom_max)}.png"
        )
        _plot_histogram(
            vals,
            out_path=focused_zoom_plot,
            title=f"{animal_id}: total raw points per cluster (zoom 0–{int(focused_zoom_max)})",
            xlabel="Total raw points per cluster",
            dpi=dpi,
            target_bin_width=focused_zoom_bin_width,
            x_max=float(focused_zoom_max),
        )
        created_paths[f"{animal_id}_focused_zoom_plot"] = focused_zoom_plot

        all_counts.append(counts_df)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "run_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    created_paths["run_manifest"] = manifest_csv

    if not all_counts:
        return created_paths

    combined_df = pd.concat(all_counts, ignore_index=True)
    combined_csv = aggregate_dir / "all_birds_cluster_point_counts.csv"
    combined_df.to_csv(combined_csv, index=False)
    created_paths["aggregate_csv"] = combined_csv

    vals_all = combined_df["n_points_total_raw"].to_numpy(dtype=float)
    full_width_all = _auto_full_bin_width(vals_all)

    full_plot_all = aggregate_dir / "all_birds_cluster_point_count_histogram_total_raw.png"
    _plot_histogram(
        vals_all,
        out_path=full_plot_all,
        title="All birds: total raw points per cluster",
        xlabel="Total raw points per cluster",
        dpi=dpi,
        target_bin_width=full_width_all,
    )
    created_paths["aggregate_full_plot"] = full_plot_all

    zoom_plot_all = aggregate_dir / f"all_birds_cluster_point_count_histogram_total_raw_zoom_0_to_{int(zoom_max)}.png"
    _plot_histogram(
        vals_all,
        out_path=zoom_plot_all,
        title=f"All birds: total raw points per cluster (zoom 0–{int(zoom_max)})",
        xlabel="Total raw points per cluster",
        dpi=dpi,
        target_bin_width=zoom_bin_width,
        x_max=float(zoom_max),
    )
    created_paths["aggregate_zoom_plot"] = zoom_plot_all

    focused_zoom_plot_all = aggregate_dir / (
        f"all_birds_cluster_point_count_histogram_total_raw_zoom_0_to_{int(focused_zoom_max)}.png"
    )
    _plot_histogram(
        vals_all,
        out_path=focused_zoom_plot_all,
        title=f"All birds: total raw points per cluster (zoom 0–{int(focused_zoom_max)})",
        xlabel="Total raw points per cluster",
        dpi=dpi,
        target_bin_width=focused_zoom_bin_width,
        x_max=float(focused_zoom_max),
    )
    created_paths["aggregate_focused_zoom_plot"] = focused_zoom_plot_all

    return created_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot per-bird and aggregate histograms of total raw points per cluster from NPZ files."
    )
    p.add_argument("--root-dir", required=True, help="Root folder containing NPZ files.")
    p.add_argument(
        "--out-dir",
        default="/Volumes/my_own_SSD/points_per_cluster_histograms",
        help="Output folder for CSVs and plots.",
    )
    p.add_argument(
        "--cluster-key",
        default="hdbscan_labels",
        help="NPZ key containing cluster labels. Default: hdbscan_labels",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under root-dir for .npz files.",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="Include negative/noise labels such as -1. Default excludes them.",
    )
    p.add_argument(
        "--zoom-max",
        type=int,
        default=5000,
        help="Upper x-limit for zoomed low-count histograms. Default: 5000",
    )
    p.add_argument(
        "--zoom-bin-width",
        type=int,
        default=100,
        help="Bin width for the 0-to-zoom-max histograms. Default: 100",
    )
    p.add_argument(
        "--focused-zoom-max",
        type=int,
        default=2500,
        help="Upper x-limit for the extra focused low-count histograms. Default: 2500",
    )
    p.add_argument(
        "--focused-zoom-bin-width",
        type=int,
        default=50,
        help="Bin width for the extra focused 0-to-focused-zoom-max histograms. Default: 50",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution. Default: 200",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    created = run_batch(
        root_dir=Path(args.root_dir),
        out_dir=Path(args.out_dir),
        recursive=bool(args.recursive),
        cluster_key=str(args.cluster_key),
        include_noise=bool(args.include_noise),
        zoom_max=int(args.zoom_max),
        zoom_bin_width=int(args.zoom_bin_width),
        focused_zoom_max=int(args.focused_zoom_max),
        focused_zoom_bin_width=int(args.focused_zoom_bin_width),
        dpi=int(args.dpi),
    )

    print("Done.")
    for key, path in created.items():
        print(f"[{key}] {path}")


if __name__ == "__main__":
    main()
