#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_bc_bin_sweep.py

Sweep discrete Bhattacharyya-coefficient histogram bin counts for a single
bird's NPZ file while holding the equalized point subsets and UMAP coordinates
fixed across the sweep.

This version:
- equalizes pre-early vs pre-late and post-early vs post-late using one shared
  final N across all four groups in a cluster
- computes three BC curves on those frozen equalized coordinates:
    1) pre early vs late
    2) post early vs late
    3) pre vs post (where pre = pre early + pre late, post = post early + post late)
- plots a 2x4 cluster figure with:
    pre early | pre late | pre early-vs-late overlap | pre-vs-post overlap
    post early| post late| post early-vs-late overlap|      (same panel)
- optionally matches the overlap-display resolution to the BC bin count so the
  displayed overlap looks more intuitive relative to the metric.

Outputs
-------
/Volumes/my_own_SSD/bc_bin_sweep/
  clusters_bc_bin_20/
    <animal>_cluster_variance_bc_summary.csv
    clusters/
      <animal>_cluster<id>_early_late_equal_only_bcbin_20.png
  clusters_bc_bin_30/
    ...
  <animal>_bc_bin_sweep_all_metrics.csv
  <animal>_bc_bin_sweep_bc_only.csv
  <animal>_bc_bin_sweep_manifest.csv
  <animal>_bc_bin_sweep_plot_manifest.csv
  <animal>_cluster_point_counts.csv
  bc_vs_bin_plots/
  cluster_point_count_plots/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import argparse
import importlib
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]



def _build_bins(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("bc-step must be > 0")
    if stop < start:
        raise ValueError("bc-stop must be >= bc-start")
    return list(range(int(start), int(stop) + 1, int(step)))



def _load_single_bird_module(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod



def _subsample_indices(indices: np.ndarray, max_n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if max_n is None or len(idx) <= int(max_n):
        return np.sort(idx.copy())
    chosen = rng.choice(idx, size=int(max_n), replace=False)
    return np.sort(np.asarray(chosen, dtype=int))



def _choose_equal_indices(indices: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if n_keep <= 0 or len(idx) == 0:
        return np.array([], dtype=int)
    if len(idx) <= int(n_keep):
        return np.sort(idx.copy())
    chosen = rng.choice(idx, size=int(n_keep), replace=False)
    return np.sort(np.asarray(chosen, dtype=int))



def _collect_bc_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "animal_id",
        "cluster",
        "bc_bins",
        "cluster_change",
        "bc_pre_early_vs_late_equal_groups",
        "bc_post_early_vs_late_equal_groups",
        "bc_pre_vs_post_equal_groups",
        "n_pre_early_raw",
        "n_pre_late_raw",
        "n_post_early_raw",
        "n_post_late_raw",
        "n_pre_pair_equal",
        "n_post_pair_equal",
        "n_shared_equal",
        "summary_csv",
        "run_out_dir",
    ]
    keep = [c for c in desired if c in df.columns]
    return df[keep].copy()



def _prepare_cluster_point_count_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty or "cluster" not in combined_df.columns:
        return pd.DataFrame()

    sort_cols = [c for c in ["cluster", "bc_bins"] if c in combined_df.columns]
    base = combined_df.sort_values(sort_cols).drop_duplicates(subset=["cluster"], keep="first").copy()

    numeric_cols = [
        "n_pre_early_raw",
        "n_pre_late_raw",
        "n_post_early_raw",
        "n_post_late_raw",
        "n_pre_pair_equal",
        "n_post_pair_equal",
        "n_shared_equal",
    ]
    for col in numeric_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if {"n_pre_early_raw", "n_pre_late_raw", "n_post_early_raw", "n_post_late_raw"}.issubset(base.columns):
        base["n_total_points_raw"] = (
            base["n_pre_early_raw"].fillna(0)
            + base["n_pre_late_raw"].fillna(0)
            + base["n_post_early_raw"].fillna(0)
            + base["n_post_late_raw"].fillna(0)
        )

    keep = [
        c
        for c in [
            "animal_id",
            "cluster",
            "cluster_change",
            "n_pre_early_raw",
            "n_pre_late_raw",
            "n_post_early_raw",
            "n_post_late_raw",
            "n_total_points_raw",
            "n_pre_pair_equal",
            "n_post_pair_equal",
            "n_shared_equal",
        ]
        if c in base.columns
    ]
    return base[keep].copy()



def _integer_bin_edges(vals: np.ndarray, target_bin_width: int, hard_max_bins: int = 120) -> np.ndarray:
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



def _plot_hist(
    values: pd.Series,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    dpi: int,
    target_bin_width: int,
    x_max: Optional[float] = None,
) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return
    if x_max is not None:
        vals = vals[vals <= float(x_max)]
        if len(vals) == 0:
            return

    bin_edges = _integer_bin_edges(vals, target_bin_width=int(target_bin_width))

    plt.figure(figsize=(8.8, 5.8))
    counts, _, _ = plt.hist(vals, bins=bin_edges, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Number of clusters")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    if len(vals) <= 80:
        y_rug = np.full(len(vals), -max(counts.max() * 0.02, 0.02))
        plt.scatter(vals, y_rug, marker="|", s=160, alpha=0.65)
        ymin, ymax = plt.ylim()
        plt.ylim(min(y_rug.min() * 1.8, ymin), ymax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()



def _plot_cluster_point_count_histograms(
    *,
    cluster_counts_df: pd.DataFrame,
    out_dir: Path,
    animal_id: str,
    dpi: int,
) -> Dict[str, Path]:
    created: Dict[str, Path] = {}
    if cluster_counts_df.empty:
        return created
    _safe_mkdir(out_dir)

    specs = [
        ("n_total_points_raw", "total raw", "Total raw points per cluster"),
        ("n_pre_pair_equal", "pre_pair_equal", "Pre pair equalized points per group"),
        ("n_post_pair_equal", "post_pair_equal", "Post pair equalized points per group"),
        ("n_shared_equal", "shared_equal", "Shared equalized points per group"),
    ]
    for col, stem, xlabel in specs:
        if col not in cluster_counts_df.columns:
            continue
        p_full = out_dir / f"{animal_id}_cluster_point_count_histogram_{stem}.png"
        _plot_hist(
            cluster_counts_df[col],
            out_path=p_full,
            title=f"{animal_id}: {stem.replace('_', ' ')} across clusters",
            xlabel=xlabel,
            dpi=dpi,
            target_bin_width=500,
        )
        created[f"hist_{stem}"] = p_full

        p_zoom = out_dir / f"{animal_id}_cluster_point_count_histogram_{stem}_zoom_0_to_5000.png"
        _plot_hist(
            cluster_counts_df[col],
            out_path=p_zoom,
            title=f"{animal_id}: {stem.replace('_', ' ')} across clusters (0–5000 zoom)",
            xlabel=xlabel,
            dpi=dpi,
            target_bin_width=100,
            x_max=5000,
        )
        created[f"hist_{stem}_zoom_0_to_5000"] = p_zoom

    return created



def _plot_bc_vs_bin_curves(
    *,
    bc_df: pd.DataFrame,
    out_dir: Path,
    animal_id: str,
    dpi: int,
) -> Dict[str, Path]:
    created: Dict[str, Path] = {}
    if bc_df.empty or "cluster" not in bc_df.columns or "bc_bins" not in bc_df.columns:
        return created

    _safe_mkdir(out_dir)
    metric_cols = [
        c
        for c in [
            "bc_pre_early_vs_late_equal_groups",
            "bc_post_early_vs_late_equal_groups",
            "bc_pre_vs_post_equal_groups",
        ]
        if c in bc_df.columns
    ]
    if not metric_cols:
        return created

    for cluster_id, sub in bc_df.groupby("cluster", sort=True):
        sdf = sub.sort_values("bc_bins").copy()
        plt.figure(figsize=(8.5, 5.5))
        plotted_any = False
        for col in metric_cols:
            y = pd.to_numeric(sdf[col], errors="coerce")
            if y.notna().any():
                plt.plot(sdf["bc_bins"], y, marker="o", label=col)
                plotted_any = True
        if not plotted_any:
            plt.close()
            continue
        plt.xlabel("Bhattacharyya coefficient bin count")
        plt.ylabel("Bhattacharyya coefficient")
        plt.title(f"{animal_id}: cluster {cluster_id} BC vs. bin count")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        out_path = out_dir / f"{animal_id}_cluster{cluster_id}_bc_vs_bin.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        created[f"cluster_{cluster_id}_bc_vs_bin"] = out_path

    return created


@dataclass
class PreparedCluster:
    cluster_id: Any
    cluster_change: str
    pre_early_xy: np.ndarray
    pre_late_xy: np.ndarray
    post_early_xy: np.ndarray
    post_late_xy: np.ndarray
    pre_early_dates: List[pd.Timestamp]
    pre_late_dates: List[pd.Timestamp]
    post_early_dates: List[pd.Timestamp]
    post_late_dates: List[pd.Timestamp]
    treatment_date: pd.Timestamp
    n_pre_early_raw: int
    n_pre_late_raw: int
    n_post_early_raw: int
    n_post_late_raw: int
    n_pre_pair_equal: int
    n_post_pair_equal: int
    n_shared_equal: int


# -----------------------------------------------------------------------------
# Cached preparation
# -----------------------------------------------------------------------------


def _prepare_cluster_embeddings(
    *,
    raw_all: np.ndarray,
    points_df: pd.DataFrame,
    treatment_date: pd.Timestamp,
    cfg,
    mod,
) -> List[PreparedCluster]:
    cluster_values = sorted(points_df["cluster"].dropna().unique(), key=lambda x: (str(type(x)), x))
    prepared: List[PreparedCluster] = []

    for cluster_id in cluster_values:
        cdf = points_df.loc[points_df["cluster"] == cluster_id].copy()
        group_point_indices = mod._build_cluster_group_indices(cdf, method=cfg.early_late_split_method)

        n_pre_early_raw = len(group_point_indices["pre_early"])
        n_pre_late_raw = len(group_point_indices["pre_late"])
        n_post_early_raw = len(group_point_indices["post_early"])
        n_post_late_raw = len(group_point_indices["post_late"])

        pre_present_any = (n_pre_early_raw + n_pre_late_raw) > 0
        post_present_any = (n_post_early_raw + n_post_late_raw) > 0
        cluster_change = mod._cluster_change_label(pre_present_any, post_present_any)

        rng = mod._rng(int(cfg.random_seed))
        capped_indices: Dict[str, np.ndarray] = {}
        for key in ["pre_early", "pre_late", "post_early", "post_late"]:
            capped_indices[key] = _subsample_indices(group_point_indices[key], cfg.max_points_per_period, rng)

        n_pre_pair_equal = min(len(capped_indices["pre_early"]), len(capped_indices["pre_late"]))
        n_post_pair_equal = min(len(capped_indices["post_early"]), len(capped_indices["post_late"]))
        n_shared_equal = min(n_pre_pair_equal, n_post_pair_equal)

        if n_shared_equal <= 0:
            continue

        eq_indices = {
            key: _choose_equal_indices(capped_indices[key], n_shared_equal, rng)
            for key in ["pre_early", "pre_late", "post_early", "post_late"]
        }

        raw_groups = {key: raw_all[eq_indices[key]] for key in eq_indices}
        raw_for_embedding = np.vstack(
            [
                raw_groups["pre_early"],
                raw_groups["pre_late"],
                raw_groups["post_early"],
                raw_groups["post_late"],
            ]
        )
        embeds = mod._fit_2d_embedding(raw_for_embedding, cfg)

        n1 = len(raw_groups["pre_early"])
        n2 = len(raw_groups["pre_late"])
        n3 = len(raw_groups["post_early"])
        n4 = len(raw_groups["post_late"])

        pre_early_xy = embeds[:n1]
        pre_late_xy = embeds[n1:n1 + n2]
        post_early_xy = embeds[n1 + n2:n1 + n2 + n3]
        post_late_xy = embeds[n1 + n2 + n3:n1 + n2 + n3 + n4]

        prepared.append(
            PreparedCluster(
                cluster_id=cluster_id,
                cluster_change=cluster_change,
                pre_early_xy=np.asarray(pre_early_xy, dtype=float),
                pre_late_xy=np.asarray(pre_late_xy, dtype=float),
                post_early_xy=np.asarray(post_early_xy, dtype=float),
                post_late_xy=np.asarray(post_late_xy, dtype=float),
                pre_early_dates=mod._extract_group_dates(points_df, eq_indices["pre_early"]),
                pre_late_dates=mod._extract_group_dates(points_df, eq_indices["pre_late"]),
                post_early_dates=mod._extract_group_dates(points_df, eq_indices["post_early"]),
                post_late_dates=mod._extract_group_dates(points_df, eq_indices["post_late"]),
                treatment_date=treatment_date,
                n_pre_early_raw=int(n_pre_early_raw),
                n_pre_late_raw=int(n_pre_late_raw),
                n_post_early_raw=int(n_post_early_raw),
                n_post_late_raw=int(n_post_late_raw),
                n_pre_pair_equal=int(n_pre_pair_equal),
                n_post_pair_equal=int(n_post_pair_equal),
                n_shared_equal=int(n_shared_equal),
            )
        )

    return prepared



def _prepare_cache(
    *,
    mod,
    cfg,
) -> Dict[str, Any]:
    raw_all, _, points_df, treatment_date, animal_id = mod._load_point_table(cfg)
    prepared = _prepare_cluster_embeddings(
        raw_all=raw_all,
        points_df=points_df,
        treatment_date=treatment_date,
        cfg=cfg,
        mod=mod,
    )
    return {
        "animal_id": animal_id,
        "points_df": points_df,
        "treatment_date": treatment_date,
        "prepared": prepared,
    }


# -----------------------------------------------------------------------------
# Rendering per bin value from frozen embeddings
# -----------------------------------------------------------------------------


def _render_one_bin_from_cache(
    *,
    mod,
    cache: Dict[str, Any],
    cfg_template,
    bc_bin: int,
    out_dir: Path,
    match_overlap_bins_to_bc: bool,
    fixed_overlap_density_bins: int,
) -> Path:
    _safe_mkdir(out_dir)
    clusters_out_dir = out_dir / "clusters"
    _safe_mkdir(clusters_out_dir)

    animal_id = str(cache["animal_id"])
    prepared_clusters: List[PreparedCluster] = cache["prepared"]

    cfg_plot = cfg_template
    cfg_plot.bc_bins = int(bc_bin)
    cfg_plot.overlap_density_bins = int(bc_bin) if match_overlap_bins_to_bc else int(fixed_overlap_density_bins)

    rows: List[Dict[str, Any]] = []
    for prepared in prepared_clusters:
        pre_all_xy = mod._stack_nonempty_xy(prepared.pre_early_xy, prepared.pre_late_xy)
        post_all_xy = mod._stack_nonempty_xy(prepared.post_early_xy, prepared.post_late_xy)

        bc_pre = mod._bhattacharyya_coefficient_2d(prepared.pre_early_xy, prepared.pre_late_xy, bins=int(bc_bin))
        bc_post = mod._bhattacharyya_coefficient_2d(prepared.post_early_xy, prepared.post_late_xy, bins=int(bc_bin))
        bc_prepost = mod._bhattacharyya_coefficient_2d(pre_all_xy, post_all_xy, bins=int(bc_bin))

        umap_pair = mod._compute_summary_for_pair(pre_all_xy, post_all_xy)

        save_path = clusters_out_dir / f"{animal_id}_cluster{prepared.cluster_id}_early_late_equal_only_bcbin_{int(bc_bin)}.png"
        mod._plot_cluster_figure(
            save_path=save_path,
            animal_id=animal_id,
            cluster_id=prepared.cluster_id,
            treatment_date=prepared.treatment_date,
            pre_early_xy=prepared.pre_early_xy,
            pre_late_xy=prepared.pre_late_xy,
            post_early_xy=prepared.post_early_xy,
            post_late_xy=prepared.post_late_xy,
            pre_early_dates=prepared.pre_early_dates,
            pre_late_dates=prepared.pre_late_dates,
            post_early_dates=prepared.post_early_dates,
            post_late_dates=prepared.post_late_dates,
            bc_pre_early_vs_late=float(bc_pre),
            bc_post_early_vs_late=float(bc_post),
            bc_pre_vs_post=float(bc_prepost),
            centroid_shift_value=float(umap_pair["centroid_shift"]),
            post_over_pre_rms=float(umap_pair["post_over_pre_rms_radius"]),
            post_over_pre_trace=float(umap_pair["post_over_pre_trace_cov"]),
            cfg=cfg_plot,
            label_suffix="equal early/late pair UMAPs",
        )

        rows.append(
            {
                "animal_id": animal_id,
                "cluster": prepared.cluster_id,
                "cluster_change": prepared.cluster_change,
                "bc_pre_early_vs_late_equal_groups": float(bc_pre),
                "bc_post_early_vs_late_equal_groups": float(bc_post),
                "bc_pre_vs_post_equal_groups": float(bc_prepost),
                "n_pre_early_raw": prepared.n_pre_early_raw,
                "n_pre_late_raw": prepared.n_pre_late_raw,
                "n_post_early_raw": prepared.n_post_early_raw,
                "n_post_late_raw": prepared.n_post_late_raw,
                "n_pre_pair_equal": prepared.n_pre_pair_equal,
                "n_post_pair_equal": prepared.n_post_pair_equal,
                "n_shared_equal": prepared.n_shared_equal,
                "centroid_shift_umap_equal_groups": float(umap_pair["centroid_shift"]),
                "pre_rms_radius_umap_equal_groups": float(umap_pair["pre_rms_radius"]),
                "post_rms_radius_umap_equal_groups": float(umap_pair["post_rms_radius"]),
                "post_over_pre_rms_radius_umap_equal_groups": float(umap_pair["post_over_pre_rms_radius"]),
                "pre_trace_cov_umap_equal_groups": float(umap_pair["pre_trace_cov"]),
                "post_trace_cov_umap_equal_groups": float(umap_pair["post_trace_cov"]),
                "post_over_pre_trace_cov_umap_equal_groups": float(umap_pair["post_over_pre_trace_cov"]),
                "overlap_visualization_bins": int(cfg_plot.overlap_density_bins),
                "figure_path": str(save_path),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / f"{animal_id}_cluster_variance_bc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    return summary_csv


# -----------------------------------------------------------------------------
# Main sweep routine
# -----------------------------------------------------------------------------


def run_bc_bin_sweep(
    *,
    npz_path: Path,
    metadata_xlsx: Path,
    out_root: Path,
    module_name: str,
    bc_values: Sequence[int],
    array_key: str,
    cluster_key: str,
    file_key: str,
    vocalization_key: str,
    file_map_key: str,
    include_noise: bool,
    only_cluster_id: Optional[int],
    vocalization_only: bool,
    min_points_per_period: int,
    max_points_per_period: Optional[int],
    max_points_per_period_for_plot: Optional[int],
    exclude_treatment_day_from_post: bool,
    early_late_split_method: str,
    random_seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    dpi: int,
    overlap_density_bins: int,
    overlap_density_gamma: float,
    metadata_sheet: Optional[str],
    match_overlap_bins_to_bc: bool,
    plot_bc_curves: bool,
    make_count_histograms: bool,
) -> Dict[str, Path]:
    del min_points_per_period, max_points_per_period_for_plot

    npz_path = Path(npz_path)
    metadata_xlsx = Path(metadata_xlsx)
    out_root = Path(out_root)
    _safe_mkdir(out_root)

    animal_id = _infer_animal_id(npz_path)
    mod = _load_single_bird_module(module_name)
    UMAPEarlyLateConfig = mod.UMAPEarlyLateConfig

    cfg_template = UMAPEarlyLateConfig(
        npz_path=npz_path,
        metadata_xlsx=metadata_xlsx,
        array_key=str(array_key),
        cluster_key=str(cluster_key),
        file_key=str(file_key),
        vocalization_key=str(vocalization_key),
        file_map_key=str(file_map_key),
        include_noise=bool(include_noise),
        only_cluster_id=only_cluster_id,
        vocalization_only=bool(vocalization_only),
        min_points_per_period=0,
        max_points_per_period=(int(max_points_per_period) if max_points_per_period is not None else None),
        max_points_per_period_for_plot=None,
        exclude_treatment_day_from_post=bool(exclude_treatment_day_from_post),
        early_late_split_method=str(early_late_split_method),
        random_seed=int(random_seed),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        out_dir=out_root,
        out_prefix=animal_id,
        dpi=int(dpi),
        bc_bins=int(bc_values[0]) if len(bc_values) else 100,
        overlap_density_bins=int(overlap_density_bins),
        overlap_density_gamma=float(overlap_density_gamma),
        metadata_sheet=metadata_sheet,
    )

    print(f"[sweep] animal_id: {animal_id}")
    print(f"[sweep] npz: {npz_path}")
    print(f"[sweep] metadata: {metadata_xlsx}")
    print(f"[sweep] out root: {out_root}")
    print(f"[sweep] bc bins: {list(bc_values)}")
    print("[sweep] Preparing frozen equalized subsets and UMAP coordinates...")
    cache = _prepare_cache(mod=mod, cfg=cfg_template)
    print(f"[sweep] prepared clusters: {len(cache['prepared'])}")

    manifest_rows: List[Dict[str, Any]] = []
    combined_dfs: List[pd.DataFrame] = []

    for bc_bin in bc_values:
        run_out_dir = out_root / f"clusters_bc_bin_{int(bc_bin)}"
        _safe_mkdir(run_out_dir)

        print(f"\n[sweep] Rendering bc_bins={bc_bin}")
        print(f"[sweep] Output dir: {run_out_dir}")

        try:
            summary_csv = _render_one_bin_from_cache(
                mod=mod,
                cache=cache,
                cfg_template=cfg_template,
                bc_bin=int(bc_bin),
                out_dir=run_out_dir,
                match_overlap_bins_to_bc=bool(match_overlap_bins_to_bc),
                fixed_overlap_density_bins=int(overlap_density_bins),
            )

            manifest_row = {
                "animal_id": animal_id,
                "bc_bins": int(bc_bin),
                "status": "ok",
                "npz_path": str(npz_path),
                "out_dir": str(run_out_dir),
                "summary_csv": str(summary_csv),
                "error": "",
                "overlap_visualization_bins": int(bc_bin) if match_overlap_bins_to_bc else int(overlap_density_bins),
            }

            if Path(summary_csv).exists():
                df = pd.read_csv(summary_csv)
                df["bc_bins"] = int(bc_bin)
                df["summary_csv"] = str(summary_csv)
                df["run_out_dir"] = str(run_out_dir)
                combined_dfs.append(df)
                manifest_row["n_cluster_rows"] = int(len(df))
            else:
                manifest_row["n_cluster_rows"] = 0
                manifest_row["status"] = "failed"
                manifest_row["error"] = f"Summary CSV not found: {summary_csv}"

            manifest_rows.append(manifest_row)

        except Exception as e:
            manifest_rows.append(
                {
                    "animal_id": animal_id,
                    "bc_bins": int(bc_bin),
                    "status": "failed",
                    "npz_path": str(npz_path),
                    "out_dir": str(run_out_dir),
                    "summary_csv": "",
                    "n_cluster_rows": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "overlap_visualization_bins": int(bc_bin) if match_overlap_bins_to_bc else int(overlap_density_bins),
                }
            )
            print(f"[sweep] FAILED for bc_bins={bc_bin}: {type(e).__name__}: {e}")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["bc_bins"]).reset_index(drop=True)
    manifest_csv = out_root / f"{animal_id}_bc_bin_sweep_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    combined_df = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()
    if not combined_df.empty:
        combined_df = combined_df.sort_values([c for c in ["animal_id", "cluster", "bc_bins"] if c in combined_df.columns]).reset_index(drop=True)

    combined_csv = out_root / f"{animal_id}_bc_bin_sweep_all_metrics.csv"
    combined_df.to_csv(combined_csv, index=False)

    bc_only_df = _collect_bc_columns(combined_df) if not combined_df.empty else pd.DataFrame()
    bc_only_csv = out_root / f"{animal_id}_bc_bin_sweep_bc_only.csv"
    bc_only_df.to_csv(bc_only_csv, index=False)

    plot_manifest_rows: List[Dict[str, str]] = []

    if plot_bc_curves and not bc_only_df.empty:
        bc_plot_dir = out_root / "bc_vs_bin_plots"
        bc_plots = _plot_bc_vs_bin_curves(bc_df=bc_only_df, out_dir=bc_plot_dir, animal_id=animal_id, dpi=dpi)
        for key, path in bc_plots.items():
            plot_manifest_rows.append({"plot_kind": key, "path": str(path)})

    cluster_counts_csv = out_root / f"{animal_id}_cluster_point_counts.csv"
    if not combined_df.empty:
        cluster_counts_df = _prepare_cluster_point_count_table(combined_df)
        cluster_counts_df.to_csv(cluster_counts_csv, index=False)
    else:
        cluster_counts_df = pd.DataFrame()
        cluster_counts_df.to_csv(cluster_counts_csv, index=False)

    if make_count_histograms and not cluster_counts_df.empty:
        count_plot_dir = out_root / "cluster_point_count_plots"
        count_plots = _plot_cluster_point_count_histograms(
            cluster_counts_df=cluster_counts_df,
            out_dir=count_plot_dir,
            animal_id=animal_id,
            dpi=dpi,
        )
        for key, path in count_plots.items():
            plot_manifest_rows.append({"plot_kind": key, "path": str(path)})

    plot_manifest_csv = out_root / f"{animal_id}_bc_bin_sweep_plot_manifest.csv"
    pd.DataFrame(plot_manifest_rows).to_csv(plot_manifest_csv, index=False)

    print("\n[sweep] Done.")
    print(f"[sweep] Manifest: {manifest_csv}")
    print(f"[sweep] Combined all-metrics CSV: {combined_csv}")
    print(f"[sweep] BC-only CSV: {bc_only_csv}")
    print(f"[sweep] Cluster point counts CSV: {cluster_counts_csv}")
    print(f"[sweep] Plot manifest: {plot_manifest_csv}")
    print(f"[sweep] Successful runs: {(manifest_df['status'] == 'ok').sum() if len(manifest_df) else 0} / {len(manifest_df)}")

    return {
        "manifest_csv": manifest_csv,
        "combined_csv": combined_csv,
        "bc_only_csv": bc_only_csv,
        "cluster_counts_csv": cluster_counts_csv,
        "plot_manifest_csv": plot_manifest_csv,
        "out_root": out_root,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_bc_bin_sweep.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--npz-path", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-root", default="/Volumes/my_own_SSD/bc_bin_sweep", type=str)

    p.add_argument(
        "--module-name",
        default="umap_pre_post_early_late_cluster_variance_bc",
        type=str,
        help="Module that defines UMAPEarlyLateConfig and helper functions",
    )

    p.add_argument("--bc-start", default=20, type=int)
    p.add_argument("--bc-stop", default=120, type=int)
    p.add_argument("--bc-step", default=10, type=int)

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--vocalization-key", default="vocalization", type=str)
    p.add_argument("--file-map-key", default="file_map", type=str)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--only-cluster-id", default=None, type=int)
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)
    p.add_argument("--plot-max-points-per-period", default=None, type=int)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument(
        "--early-late-split-method",
        default="file_median",
        choices=["file_median", "file_half"],
    )

    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--dpi", default=200, type=int)
    p.add_argument(
        "--overlap-density-bins",
        default=180,
        type=int,
        help="Used only when --no-match-overlap-bins-to-bc is set",
    )
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)

    p.add_argument("--metadata-sheet", default=None, type=str)

    p.add_argument(
        "--no-match-overlap-bins-to-bc",
        action="store_true",
        help="Keep overlap visualization bins fixed instead of matching each BC bin count",
    )
    p.add_argument(
        "--no-plot-bc-curves",
        action="store_true",
        help="Skip the BC-vs-bin summary plots",
    )
    p.add_argument(
        "--no-count-histograms",
        action="store_true",
        help="Skip cluster point-count histograms",
    )

    return p



def main() -> None:
    args = _build_arg_parser().parse_args()

    bc_values = _build_bins(args.bc_start, args.bc_stop, args.bc_step)

    run_bc_bin_sweep(
        npz_path=Path(args.npz_path),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_root=Path(args.out_root),
        module_name=str(args.module_name),
        bc_values=bc_values,
        array_key=str(args.array_key),
        cluster_key=str(args.cluster_key),
        file_key=str(args.file_key),
        vocalization_key=str(args.vocalization_key),
        file_map_key=str(args.file_map_key),
        include_noise=bool(args.include_noise),
        only_cluster_id=(int(args.only_cluster_id) if args.only_cluster_id is not None else None),
        vocalization_only=not bool(args.no_vocalization_only),
        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=(int(args.max_points_per_period) if args.max_points_per_period is not None else None),
        max_points_per_period_for_plot=(int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None),
        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        early_late_split_method=str(args.early_late_split_method),
        random_seed=int(args.seed),
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        dpi=int(args.dpi),
        overlap_density_bins=int(args.overlap_density_bins),
        overlap_density_gamma=float(args.overlap_density_gamma),
        metadata_sheet=args.metadata_sheet,
        match_overlap_bins_to_bc=not bool(args.no_match_overlap_bins_to_bc),
        plot_bc_curves=not bool(args.no_plot_bc_curves),
        make_count_histograms=not bool(args.no_count_histograms),
    )


if __name__ == "__main__":
    main()
