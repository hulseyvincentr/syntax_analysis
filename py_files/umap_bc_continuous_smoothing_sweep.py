#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_bc_continuous_smoothing_sweep.py

Sweep a KDE-smoothed, continuous-style Bhattacharyya coefficient for a single
bird's NPZ file, using the same point loading / cluster splitting / UMAP /
figure layout as:
    umap_pre_post_early_late_cluster_variance_bc.py

What this script does
---------------------
- Dynamically imports + reloads the existing single-bird UMAP/cluster module.
- Reuses that module's helpers for:
    * loading the NPZ + metadata
    * splitting points into pre/post early/late groups
    * fitting the 2D embedding
    * plotting the 2x4 cluster figures
- Replaces the histogram-based BC with a KDE-based, continuous-style overlap
  estimate computed on a 2D evaluation grid.
- Sweeps over smoothing bandwidth values instead of histogram bin counts.
- Saves one folder per bandwidth under, by default:
      /Volumes/my_own_SSD/bc_continuous_smoothing_sweep
  with names like:
      clusters_kde_bw_0p10
      clusters_kde_bw_0p20
      ...
- Saves combined sweep CSVs and BC-vs-bandwidth plots.

Important note
--------------
The "continuous" BC here is a numerical grid approximation of:
    BC(p, q) = integral sqrt(p(x) q(x)) dx
where p and q are 2D KDE density estimates in embedding space.
It is therefore sensitive to KDE bandwidth and, to a lesser extent, grid size.

Example
-------
python /Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/umap_bc_continuous_smoothing_sweep.py \
  --npz-path "/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443.npz" \
  --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-root "/Volumes/my_own_SSD/bc_continuous_smoothing_sweep" \
  --bw-start 0.10 \
  --bw-stop 1.00 \
  --bw-step 0.10 \
  --seed -1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import importlib
import math
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]



def _sanitize_for_filename(value: Any) -> str:
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"



def _cluster_sort_key(value: Any) -> Tuple[int, Any]:
    try:
        if pd.isna(value):
            return (2, "")
    except Exception:
        pass

    try:
        f = float(value)
        if math.isfinite(f):
            return (0, f)
    except Exception:
        pass

    return (1, str(value))



def _bandwidth_slug(bw: float) -> str:
    return f"{float(bw):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")



def _build_float_sweep(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("bw-step must be > 0")
    if stop < start:
        raise ValueError("bw-stop must be >= bw-start")

    values: List[float] = []
    cur = float(start)
    stop = float(stop)
    step = float(step)
    tol = abs(step) * 1e-9
    while cur <= stop + tol:
        values.append(round(cur, 10))
        cur += step
    return values



def _load_single_bird_module(module_name: str):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod


# -----------------------------------------------------------------------------
# Continuous BC (KDE + numerical integration on a shared grid)
# -----------------------------------------------------------------------------

def _ensure_xy(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected array of shape (n, 2), got {arr.shape}")
    if len(arr) == 0:
        return np.empty((0, 2), dtype=float)
    keep = np.isfinite(arr).all(axis=1)
    return arr[keep]



def _build_eval_grid(
    xy_groups: Sequence[np.ndarray],
    *,
    bandwidth: float,
    grid_size: int,
    pad_fraction: float,
) -> Tuple[np.ndarray, float, float]:
    nonempty = [
        _ensure_xy(arr)
        for arr in xy_groups
        if arr is not None and len(np.asarray(arr)) > 0
    ]
    if not nonempty:
        raise ValueError("No non-empty XY arrays were provided for KDE grid construction.")

    both = np.vstack(nonempty)
    xmin, ymin = both.min(axis=0)
    xmax, ymax = both.max(axis=0)

    xr = float(xmax - xmin)
    yr = float(ymax - ymin)
    if xr <= 0:
        xr = 1.0
    if yr <= 0:
        yr = 1.0

    xpad = max(float(pad_fraction) * xr, 3.0 * float(bandwidth))
    ypad = max(float(pad_fraction) * yr, 3.0 * float(bandwidth))

    xs = np.linspace(xmin - xpad, xmax + xpad, int(grid_size))
    ys = np.linspace(ymin - ypad, ymax + ypad, int(grid_size))
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    return grid, dx, dy



def _kde_prob_on_grid(
    xy: np.ndarray,
    *,
    grid: np.ndarray,
    dx: float,
    dy: float,
    bandwidth: float,
    kernel: str,
    grid_size: int,
) -> np.ndarray:
    xy = _ensure_xy(xy)
    if len(xy) == 0:
        return np.zeros((int(grid_size), int(grid_size)), dtype=float)

    kde = KernelDensity(kernel=str(kernel), bandwidth=float(bandwidth))
    kde.fit(xy)
    log_d = kde.score_samples(grid)
    dens = np.exp(log_d).reshape(int(grid_size), int(grid_size))

    mass = float(dens.sum() * dx * dy)
    if not np.isfinite(mass) or mass <= 0:
        return np.zeros((int(grid_size), int(grid_size)), dtype=float)

    return dens / mass



def _continuous_bc_from_probs(p: np.ndarray, q: np.ndarray, *, dx: float, dy: float) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"Probability grids must have the same shape, got {p.shape} vs {q.shape}")

    bc = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))) * dx * dy)
    if not np.isfinite(bc):
        return np.nan
    return float(np.clip(bc, 0.0, 1.0))



def _compute_continuous_bc_group_set(
    *,
    xy_map: Dict[str, np.ndarray],
    bandwidth: float,
    kernel: str,
    grid_size: int,
    pad_fraction: float,
) -> Dict[str, float]:
    required = ["pre_early", "pre_late", "post_early", "post_late"]
    if not all(k in xy_map for k in required):
        missing = [k for k in required if k not in xy_map]
        raise KeyError(f"Missing required XY groups for continuous BC: {missing}")

    pre_all = np.vstack([xy_map["pre_early"], xy_map["pre_late"]]) if (len(xy_map["pre_early"]) or len(xy_map["pre_late"])) else np.empty((0, 2), dtype=float)
    post_all = np.vstack([xy_map["post_early"], xy_map["post_late"]]) if (len(xy_map["post_early"]) or len(xy_map["post_late"])) else np.empty((0, 2), dtype=float)

    out = {
        "continuous_bc_pre_early_vs_late": np.nan,
        "continuous_bc_post_early_vs_late": np.nan,
        "continuous_bc_pre_vs_post": np.nan,
    }

    nonempty_groups = [
        arr for arr in [xy_map["pre_early"], xy_map["pre_late"], xy_map["post_early"], xy_map["post_late"], pre_all, post_all]
        if len(arr) > 0
    ]
    if len(nonempty_groups) < 2:
        return out

    grid, dx, dy = _build_eval_grid(
        nonempty_groups,
        bandwidth=float(bandwidth),
        grid_size=int(grid_size),
        pad_fraction=float(pad_fraction),
    )

    prob_map = {
        "pre_early": _kde_prob_on_grid(xy_map["pre_early"], grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
        "pre_late": _kde_prob_on_grid(xy_map["pre_late"], grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
        "post_early": _kde_prob_on_grid(xy_map["post_early"], grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
        "post_late": _kde_prob_on_grid(xy_map["post_late"], grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
        "pre_all": _kde_prob_on_grid(pre_all, grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
        "post_all": _kde_prob_on_grid(post_all, grid=grid, dx=dx, dy=dy, bandwidth=bandwidth, kernel=kernel, grid_size=grid_size),
    }

    if len(xy_map["pre_early"]) > 0 and len(xy_map["pre_late"]) > 0:
        out["continuous_bc_pre_early_vs_late"] = _continuous_bc_from_probs(prob_map["pre_early"], prob_map["pre_late"], dx=dx, dy=dy)
    if len(xy_map["post_early"]) > 0 and len(xy_map["post_late"]) > 0:
        out["continuous_bc_post_early_vs_late"] = _continuous_bc_from_probs(prob_map["post_early"], prob_map["post_late"], dx=dx, dy=dy)
    if len(pre_all) > 0 and len(post_all) > 0:
        out["continuous_bc_pre_vs_post"] = _continuous_bc_from_probs(prob_map["pre_all"], prob_map["post_all"], dx=dx, dy=dy)

    return out


# -----------------------------------------------------------------------------
# Plotting the sweep summary
# -----------------------------------------------------------------------------

def _metric_specs() -> List[Tuple[str, str]]:
    return [
        ("continuous_bc_pre_early_vs_late", "Pre early vs late"),
        ("continuous_bc_post_early_vs_late", "Post early vs late"),
        ("continuous_bc_pre_vs_post", "Pre vs post"),
        ("continuous_bc_pre_early_vs_late_equal_groups", "Pre early vs late (equal groups)"),
        ("continuous_bc_post_early_vs_late_equal_groups", "Post early vs late (equal groups)"),
        ("continuous_bc_pre_vs_post_equal_groups", "Pre vs post (equal groups)"),
    ]



def _available_metric_specs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for col, label in _metric_specs():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            specs.append((col, label))
    return specs



def _collect_bc_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "animal_id",
        "cluster",
        "kde_bandwidth",
        "cluster_change",
        "continuous_bc_pre_early_vs_late",
        "continuous_bc_post_early_vs_late",
        "continuous_bc_pre_vs_post",
        "continuous_bc_pre_early_vs_late_equal_groups",
        "continuous_bc_post_early_vs_late_equal_groups",
        "continuous_bc_pre_vs_post_equal_groups",
        "summary_csv",
        "run_out_dir",
    ]
    keep = [c for c in desired if c in df.columns]
    return df[keep].copy()



def _plot_bc_vs_bandwidth_curves(
    *,
    bc_df: pd.DataFrame,
    out_dir: Path,
    animal_id: str,
    dpi: int = 200,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    _safe_mkdir(out_dir)

    if bc_df.empty:
        raise ValueError("BC dataframe is empty; nothing to plot.")
    if "cluster" not in bc_df.columns or "kde_bandwidth" not in bc_df.columns:
        raise ValueError("BC dataframe must contain at least 'cluster' and 'kde_bandwidth' columns.")

    work_df = bc_df.copy()
    work_df["kde_bandwidth"] = pd.to_numeric(work_df["kde_bandwidth"], errors="coerce")
    work_df = work_df.loc[work_df["kde_bandwidth"].notna()].copy()
    if work_df.empty:
        raise ValueError("No valid numeric kde_bandwidth values found in the sweep CSV.")

    metric_specs = _available_metric_specs(work_df)
    if not metric_specs:
        raise ValueError("No plottable continuous BC metric columns were found in the sweep CSV.")

    created: Dict[str, Path] = {}
    cluster_values = sorted(work_df["cluster"].dropna().unique().tolist(), key=_cluster_sort_key)

    for cluster in cluster_values:
        sub = work_df.loc[work_df["cluster"] == cluster].copy().sort_values("kde_bandwidth")
        if sub.empty:
            continue

        plt.figure(figsize=(8.5, 5.5))
        plotted_any = False
        for metric_col, metric_label in metric_specs:
            y = pd.to_numeric(sub[metric_col], errors="coerce")
            valid = y.notna() & sub["kde_bandwidth"].notna()
            if not valid.any():
                continue
            plotted_any = True
            plt.plot(
                sub.loc[valid, "kde_bandwidth"],
                y.loc[valid],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=metric_label,
            )

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("KDE bandwidth (smoothing)")
        plt.ylabel("Continuous Bhattacharyya coefficient")
        plt.title(f"{animal_id} | cluster {cluster} | continuous BC vs smoothing")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()

        plot_path = out_dir / f"{animal_id}_cluster_{_sanitize_for_filename(cluster)}_continuous_bc_vs_bandwidth.png"
        plt.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
        plt.close()
        created[f"cluster::{cluster}"] = plot_path

    for metric_col, metric_label in metric_specs:
        metric_df = work_df[["cluster", "kde_bandwidth", metric_col]].copy()
        metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors="coerce")
        metric_df = metric_df.loc[metric_df[metric_col].notna()].copy()
        if metric_df.empty:
            continue

        plt.figure(figsize=(8.8, 5.8))
        plotted_any = False
        for cluster in cluster_values:
            sub = metric_df.loc[metric_df["cluster"] == cluster].copy().sort_values("kde_bandwidth")
            if sub.empty:
                continue
            plotted_any = True
            plt.plot(
                sub["kde_bandwidth"],
                sub[metric_col],
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=f"cluster {cluster}",
            )

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("KDE bandwidth (smoothing)")
        plt.ylabel("Continuous Bhattacharyya coefficient")
        plt.title(f"{animal_id} | {metric_label} | continuous BC vs smoothing")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.25)
        if len(cluster_values) <= 20:
            plt.legend(frameon=False, fontsize=8, ncol=2)
        plt.tight_layout()

        plot_path = out_dir / f"{animal_id}_{metric_col}_continuous_bc_vs_bandwidth_all_clusters.png"
        plt.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
        plt.close()
        created[f"metric::{metric_col}"] = plot_path

    return created


# -----------------------------------------------------------------------------
# Per-cluster continuous-BC analysis
# -----------------------------------------------------------------------------

def _analyze_one_cluster_continuous(
    *,
    mod,
    cluster_id: Any,
    raw_all: np.ndarray,
    points_df: pd.DataFrame,
    treatment_date: pd.Timestamp,
    animal_id: str,
    cfg,
    clusters_out_dir: Path,
    bandwidth: float,
    kde_kernel: str,
    kde_grid_size: int,
    kde_pad_fraction: float,
) -> Dict[str, Any]:
    cdf = points_df.loc[points_df["cluster"] == cluster_id].copy()

    group_point_indices = mod._build_cluster_group_indices(cdf, method=cfg.early_late_split_method)
    group_raw = mod._build_group_arrays(raw_all, cdf, group_point_indices)

    n_pre_total_raw = len(group_raw["pre_all"])
    n_post_total_raw = len(group_raw["post_all"])
    n_pre_early_raw = len(group_raw["pre_early"])
    n_pre_late_raw = len(group_raw["pre_late"])
    n_post_early_raw = len(group_raw["post_early"])
    n_post_late_raw = len(group_raw["post_late"])

    pre_present_any = n_pre_total_raw > 0
    post_present_any = n_post_total_raw > 0

    row: Dict[str, Any] = {
        "animal_id": animal_id,
        "cluster": cluster_id,
        "cluster_change": mod._cluster_change_label(pre_present_any, post_present_any),
        "pre_present_any": bool(pre_present_any),
        "post_present_any": bool(post_present_any),
        "kde_bandwidth": float(bandwidth),
        "kde_kernel": str(kde_kernel),
        "kde_grid_size": int(kde_grid_size),
        "kde_pad_fraction": float(kde_pad_fraction),

        "n_pre_total_raw": int(n_pre_total_raw),
        "n_post_total_raw": int(n_post_total_raw),
        "n_pre_early_raw": int(n_pre_early_raw),
        "n_pre_late_raw": int(n_pre_late_raw),
        "n_post_early_raw": int(n_post_early_raw),
        "n_post_late_raw": int(n_post_late_raw),

        "continuous_bc_pre_early_vs_late": np.nan,
        "continuous_bc_post_early_vs_late": np.nan,
        "continuous_bc_pre_vs_post": np.nan,

        "centroid_shift_raw": np.nan,
        "pre_rms_radius_raw": np.nan,
        "post_rms_radius_raw": np.nan,
        "post_over_pre_rms_radius": np.nan,
        "pre_trace_cov_raw": np.nan,
        "post_trace_cov_raw": np.nan,
        "post_over_pre_trace_cov": np.nan,

        "centroid_shift_umap": np.nan,
        "pre_rms_radius_umap": np.nan,
        "post_rms_radius_umap": np.nan,
        "post_over_pre_rms_radius_umap": np.nan,

        "continuous_bc_pre_early_vs_late_equal_groups": np.nan,
        "continuous_bc_post_early_vs_late_equal_groups": np.nan,
        "continuous_bc_pre_vs_post_equal_groups": np.nan,

        "centroid_shift_raw_equal_groups": np.nan,
        "pre_rms_radius_raw_equal_groups": np.nan,
        "post_rms_radius_raw_equal_groups": np.nan,
        "post_over_pre_rms_radius_equal_groups": np.nan,
        "pre_trace_cov_raw_equal_groups": np.nan,
        "post_trace_cov_raw_equal_groups": np.nan,
        "post_over_pre_trace_cov_equal_groups": np.nan,

        "centroid_shift_umap_equal_groups": np.nan,
        "pre_rms_radius_umap_equal_groups": np.nan,
        "post_rms_radius_umap_equal_groups": np.nan,
        "post_over_pre_rms_radius_umap_equal_groups": np.nan,
    }

    raw_pre_all = mod._stack_raw(group_raw["pre_early"], group_raw["pre_late"])
    raw_post_all = mod._stack_raw(group_raw["post_early"], group_raw["post_late"])
    raw_pair = mod._compute_summary_for_pair(raw_pre_all, raw_post_all)
    row["centroid_shift_raw"] = raw_pair["centroid_shift"]
    row["pre_rms_radius_raw"] = raw_pair["pre_rms_radius"]
    row["post_rms_radius_raw"] = raw_pair["post_rms_radius"]
    row["post_over_pre_rms_radius"] = raw_pair["post_over_pre_rms_radius"]
    row["pre_trace_cov_raw"] = raw_pair["pre_trace_cov"]
    row["post_trace_cov_raw"] = raw_pair["post_trace_cov"]
    row["post_over_pre_trace_cov"] = raw_pair["post_over_pre_trace_cov"]

    local_rng = mod._rng(cfg.random_seed)
    group_raw_plot: Dict[str, np.ndarray] = {}
    for key in ["pre_early", "pre_late", "post_early", "post_late"]:
        group_raw_plot[key] = mod._subsample_array(group_raw[key], cfg.max_points_per_period, local_rng)

    all_groups_present = all(len(group_raw_plot[k]) > 0 for k in ["pre_early", "pre_late", "post_early", "post_late"])

    if all_groups_present:
        raw_for_embedding = np.vstack(
            [
                group_raw_plot["pre_early"],
                group_raw_plot["pre_late"],
                group_raw_plot["post_early"],
                group_raw_plot["post_late"],
            ]
        )
        embeds = mod._fit_2d_embedding(raw_for_embedding, cfg)

        n1 = len(group_raw_plot["pre_early"])
        n2 = len(group_raw_plot["pre_late"])
        n3 = len(group_raw_plot["post_early"])
        n4 = len(group_raw_plot["post_late"])

        xy_pre_early = embeds[:n1]
        xy_pre_late = embeds[n1:n1 + n2]
        xy_post_early = embeds[n1 + n2:n1 + n2 + n3]
        xy_post_late = embeds[n1 + n2 + n3:n1 + n2 + n3 + n4]

        xy_pre_all = mod._stack_nonempty_xy(xy_pre_early, xy_pre_late)
        xy_post_all = mod._stack_nonempty_xy(xy_post_early, xy_post_late)

        bc_raw = _compute_continuous_bc_group_set(
            xy_map={
                "pre_early": xy_pre_early,
                "pre_late": xy_pre_late,
                "post_early": xy_post_early,
                "post_late": xy_post_late,
            },
            bandwidth=float(bandwidth),
            kernel=str(kde_kernel),
            grid_size=int(kde_grid_size),
            pad_fraction=float(kde_pad_fraction),
        )
        row.update(bc_raw)

        umap_pair = mod._compute_summary_for_pair(xy_pre_all, xy_post_all)
        row["centroid_shift_umap"] = umap_pair["centroid_shift"]
        row["pre_rms_radius_umap"] = umap_pair["pre_rms_radius"]
        row["post_rms_radius_umap"] = umap_pair["post_rms_radius"]
        row["post_over_pre_rms_radius_umap"] = umap_pair["post_over_pre_rms_radius"]

        raw_fig_path = clusters_out_dir / f"{cfg.out_prefix}_cluster{cluster_id}_pre_post_early_late_umap_continuous_bc_raw_groups.png"
        mod._plot_cluster_figure(
            save_path=raw_fig_path,
            animal_id=animal_id,
            cluster_id=cluster_id,
            treatment_date=treatment_date,
            pre_early_xy=xy_pre_early,
            pre_late_xy=xy_pre_late,
            post_early_xy=xy_post_early,
            post_late_xy=xy_post_late,
            pre_early_dates=mod._extract_group_dates(points_df, group_point_indices["pre_early"]),
            pre_late_dates=mod._extract_group_dates(points_df, group_point_indices["pre_late"]),
            post_early_dates=mod._extract_group_dates(points_df, group_point_indices["post_early"]),
            post_late_dates=mod._extract_group_dates(points_df, group_point_indices["post_late"]),
            bc_pre_early_vs_late=row["continuous_bc_pre_early_vs_late"],
            bc_post_early_vs_late=row["continuous_bc_post_early_vs_late"],
            bc_pre_vs_post=row["continuous_bc_pre_vs_post"],
            centroid_shift_value=row["centroid_shift_umap"],
            post_over_pre_rms=row["post_over_pre_rms_radius_umap"],
            post_over_pre_trace=row["post_over_pre_trace_cov"],
            cfg=cfg,
            label_suffix=f"raw-groups | continuous KDE BC | bw={bandwidth:g}",
        )

        n_each, eq_idx = mod._balanced_group_indices(group_raw_plot, seed=cfg.random_seed)
        if n_each is not None and n_each > 0:
            eq_pre_early_raw = group_raw_plot["pre_early"][eq_idx["pre_early"]]
            eq_pre_late_raw = group_raw_plot["pre_late"][eq_idx["pre_late"]]
            eq_post_early_raw = group_raw_plot["post_early"][eq_idx["post_early"]]
            eq_post_late_raw = group_raw_plot["post_late"][eq_idx["post_late"]]

            eq_pre_all_raw = mod._stack_raw(eq_pre_early_raw, eq_pre_late_raw)
            eq_post_all_raw = mod._stack_raw(eq_post_early_raw, eq_post_late_raw)

            raw_eq_pair = mod._compute_summary_for_pair(eq_pre_all_raw, eq_post_all_raw)
            row["centroid_shift_raw_equal_groups"] = raw_eq_pair["centroid_shift"]
            row["pre_rms_radius_raw_equal_groups"] = raw_eq_pair["pre_rms_radius"]
            row["post_rms_radius_raw_equal_groups"] = raw_eq_pair["post_rms_radius"]
            row["post_over_pre_rms_radius_equal_groups"] = raw_eq_pair["post_over_pre_rms_radius"]
            row["pre_trace_cov_raw_equal_groups"] = raw_eq_pair["pre_trace_cov"]
            row["post_trace_cov_raw_equal_groups"] = raw_eq_pair["post_trace_cov"]
            row["post_over_pre_trace_cov_equal_groups"] = raw_eq_pair["post_over_pre_trace_cov"]

            eq_raw_for_embedding = np.vstack([eq_pre_early_raw, eq_pre_late_raw, eq_post_early_raw, eq_post_late_raw])
            eq_embeds = mod._fit_2d_embedding(eq_raw_for_embedding, cfg)

            e1 = len(eq_pre_early_raw)
            e2 = len(eq_pre_late_raw)
            e3 = len(eq_post_early_raw)
            e4 = len(eq_post_late_raw)

            eq_xy_pre_early = eq_embeds[:e1]
            eq_xy_pre_late = eq_embeds[e1:e1 + e2]
            eq_xy_post_early = eq_embeds[e1 + e2:e1 + e2 + e3]
            eq_xy_post_late = eq_embeds[e1 + e2 + e3:e1 + e2 + e3 + e4]

            eq_xy_pre_all = mod._stack_nonempty_xy(eq_xy_pre_early, eq_xy_pre_late)
            eq_xy_post_all = mod._stack_nonempty_xy(eq_xy_post_early, eq_xy_post_late)

            bc_eq = _compute_continuous_bc_group_set(
                xy_map={
                    "pre_early": eq_xy_pre_early,
                    "pre_late": eq_xy_pre_late,
                    "post_early": eq_xy_post_early,
                    "post_late": eq_xy_post_late,
                },
                bandwidth=float(bandwidth),
                kernel=str(kde_kernel),
                grid_size=int(kde_grid_size),
                pad_fraction=float(kde_pad_fraction),
            )
            row["continuous_bc_pre_early_vs_late_equal_groups"] = bc_eq["continuous_bc_pre_early_vs_late"]
            row["continuous_bc_post_early_vs_late_equal_groups"] = bc_eq["continuous_bc_post_early_vs_late"]
            row["continuous_bc_pre_vs_post_equal_groups"] = bc_eq["continuous_bc_pre_vs_post"]

            umap_eq_pair = mod._compute_summary_for_pair(eq_xy_pre_all, eq_xy_post_all)
            row["centroid_shift_umap_equal_groups"] = umap_eq_pair["centroid_shift"]
            row["pre_rms_radius_umap_equal_groups"] = umap_eq_pair["pre_rms_radius"]
            row["post_rms_radius_umap_equal_groups"] = umap_eq_pair["post_rms_radius"]
            row["post_over_pre_rms_radius_umap_equal_groups"] = umap_eq_pair["post_over_pre_rms_radius"]

            eq_fig_path = clusters_out_dir / f"{cfg.out_prefix}_cluster{cluster_id}_pre_post_early_late_umap_continuous_bc_equal_groups.png"
            mod._plot_cluster_figure(
                save_path=eq_fig_path,
                animal_id=animal_id,
                cluster_id=cluster_id,
                treatment_date=treatment_date,
                pre_early_xy=eq_xy_pre_early,
                pre_late_xy=eq_xy_pre_late,
                post_early_xy=eq_xy_post_early,
                post_late_xy=eq_xy_post_late,
                pre_early_dates=mod._extract_group_dates(points_df, group_point_indices["pre_early"]),
                pre_late_dates=mod._extract_group_dates(points_df, group_point_indices["pre_late"]),
                post_early_dates=mod._extract_group_dates(points_df, group_point_indices["post_early"]),
                post_late_dates=mod._extract_group_dates(points_df, group_point_indices["post_late"]),
                bc_pre_early_vs_late=row["continuous_bc_pre_early_vs_late_equal_groups"],
                bc_post_early_vs_late=row["continuous_bc_post_early_vs_late_equal_groups"],
                bc_pre_vs_post=row["continuous_bc_pre_vs_post_equal_groups"],
                centroid_shift_value=row["centroid_shift_umap_equal_groups"],
                post_over_pre_rms=row["post_over_pre_rms_radius_umap_equal_groups"],
                post_over_pre_trace=row["post_over_pre_trace_cov_equal_groups"],
                cfg=cfg,
                label_suffix=f"equal-sized four-group plot | continuous KDE BC | bw={bandwidth:g}",
            )

    return row


# -----------------------------------------------------------------------------
# Sweep runner
# -----------------------------------------------------------------------------

def run_continuous_bc_smoothing_sweep(
    *,
    npz_path: Path,
    metadata_xlsx: Path,
    out_root: Path,
    module_name: str,
    bandwidth_values: Sequence[float],
    kde_kernel: str,
    kde_grid_size: int,
    kde_pad_fraction: float,
    make_plots: bool,
    array_key: str = "predictions",
    cluster_key: str = "hdbscan_labels",
    file_key: str = "file_indices",
    vocalization_key: str = "vocalization",
    file_map_key: str = "file_map",
    include_noise: bool = False,
    only_cluster_id: Optional[int] = None,
    vocalization_only: bool = True,
    min_points_per_period: int = 200,
    max_points_per_period: Optional[int] = None,
    max_points_per_period_for_plot: Optional[int] = None,
    exclude_treatment_day_from_post: bool = False,
    early_late_split_method: str = "file_median",
    random_seed: int = 0,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    dpi: int = 200,
    overlap_density_bins: int = 180,
    overlap_density_gamma: float = 0.55,
    metadata_sheet: Optional[str] = None,
) -> Dict[str, Path]:
    npz_path = Path(npz_path)
    metadata_xlsx = Path(metadata_xlsx)
    out_root = Path(out_root)
    _safe_mkdir(out_root)

    animal_id = _infer_animal_id(npz_path)
    mod = _load_single_bird_module(module_name)

    if not hasattr(mod, "UMAPEarlyLateConfig"):
        raise AttributeError(f"Module {module_name!r} does not define UMAPEarlyLateConfig")

    needed_helpers = [
        "_load_point_table",
        "_build_cluster_group_indices",
        "_build_group_arrays",
        "_compute_summary_for_pair",
        "_extract_group_dates",
        "_balanced_group_indices",
        "_subsample_array",
        "_rng",
        "_cluster_change_label",
        "_stack_raw",
        "_stack_nonempty_xy",
        "_fit_2d_embedding",
        "_plot_cluster_figure",
    ]
    missing = [name for name in needed_helpers if not hasattr(mod, name)]
    if missing:
        raise AttributeError(
            f"Module {module_name!r} is missing helper(s) required by the continuous BC sweep: {missing}"
        )

    # Load once so all bandwidth runs use the same points.
    base_cfg = mod.UMAPEarlyLateConfig(
        npz_path=Path(npz_path),
        metadata_xlsx=Path(metadata_xlsx),
        array_key=str(array_key),
        cluster_key=str(cluster_key),
        file_key=str(file_key),
        vocalization_key=str(vocalization_key),
        file_map_key=str(file_map_key),
        include_noise=bool(include_noise),
        only_cluster_id=only_cluster_id,
        vocalization_only=bool(vocalization_only),
        min_points_per_period=int(min_points_per_period),
        max_points_per_period=(int(max_points_per_period) if max_points_per_period is not None else None),
        max_points_per_period_for_plot=(
            int(max_points_per_period_for_plot) if max_points_per_period_for_plot is not None else None
        ),
        exclude_treatment_day_from_post=bool(exclude_treatment_day_from_post),
        early_late_split_method=str(early_late_split_method),
        random_seed=int(random_seed),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        out_dir=Path(out_root),
        out_prefix=str(animal_id),
        dpi=int(dpi),
        bc_bins=100,  # unused here; retained for compatibility with plotting helper
        overlap_density_bins=int(overlap_density_bins),
        overlap_density_gamma=float(overlap_density_gamma),
        metadata_sheet=metadata_sheet,
    )

    raw_all, _clusters, points_df, treatment_date, animal_id = mod._load_point_table(base_cfg)
    if len(points_df) == 0:
        raise ValueError("No usable points remain after filtering.")

    cluster_values = sorted(points_df["cluster"].dropna().unique(), key=_cluster_sort_key)

    manifest_rows: List[Dict[str, Any]] = []
    combined_dfs: List[pd.DataFrame] = []

    for bandwidth in bandwidth_values:
        bandwidth = float(bandwidth)
        bw_slug = _bandwidth_slug(bandwidth)
        run_out_dir = out_root / f"clusters_kde_bw_{bw_slug}"
        clusters_out_dir = run_out_dir / "clusters"
        _safe_mkdir(clusters_out_dir)

        try:
            cfg = mod.UMAPEarlyLateConfig(**vars(base_cfg))
            cfg.out_dir = run_out_dir
            cfg.out_prefix = animal_id
            _safe_mkdir(cfg.out_dir)
            _safe_mkdir(clusters_out_dir)

            rows: List[Dict[str, Any]] = []
            for cluster_id in cluster_values:
                print(f"[continuous-sweep] {animal_id}: bw={bandwidth:g}, cluster {cluster_id}")
                row = _analyze_one_cluster_continuous(
                    mod=mod,
                    cluster_id=cluster_id,
                    raw_all=raw_all,
                    points_df=points_df,
                    treatment_date=treatment_date,
                    animal_id=animal_id,
                    cfg=cfg,
                    clusters_out_dir=clusters_out_dir,
                    bandwidth=bandwidth,
                    kde_kernel=kde_kernel,
                    kde_grid_size=kde_grid_size,
                    kde_pad_fraction=kde_pad_fraction,
                )
                rows.append(row)

            run_df = pd.DataFrame(rows).sort_values(["animal_id", "cluster"]).reset_index(drop=True)
            summary_csv = run_out_dir / f"{animal_id}_cluster_variance_continuous_bc_summary.csv"
            run_df.to_csv(summary_csv, index=False)

            run_df["summary_csv"] = str(summary_csv)
            run_df["run_out_dir"] = str(run_out_dir)
            combined_dfs.append(run_df)

            manifest_rows.append(
                {
                    "kde_bandwidth": float(bandwidth),
                    "status": "ok",
                    "npz_path": str(npz_path),
                    "out_dir": str(run_out_dir),
                    "summary_csv": str(summary_csv),
                    "n_cluster_rows": int(len(run_df)),
                    "error": "",
                    "traceback": "",
                }
            )
            print(f"[continuous-sweep] OK for bandwidth={bandwidth:g}: {summary_csv}")

        except Exception as e:
            manifest_rows.append(
                {
                    "kde_bandwidth": float(bandwidth),
                    "status": "failed",
                    "npz_path": str(npz_path),
                    "out_dir": str(run_out_dir),
                    "summary_csv": "",
                    "n_cluster_rows": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[continuous-sweep] FAILED for bandwidth={bandwidth:g}: {type(e).__name__}: {e}")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["kde_bandwidth"]).reset_index(drop=True)
    manifest_csv = out_root / f"{animal_id}_continuous_bc_smoothing_sweep_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        sort_cols = [c for c in ["animal_id", "cluster", "kde_bandwidth"] if c in combined_df.columns]
        if sort_cols:
            combined_df = combined_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        combined_df = pd.DataFrame()

    combined_csv = out_root / f"{animal_id}_continuous_bc_smoothing_sweep_all_metrics.csv"
    combined_df.to_csv(combined_csv, index=False)

    bc_only_df = _collect_bc_columns(combined_df) if len(combined_df) else pd.DataFrame()
    bc_only_csv = out_root / f"{animal_id}_continuous_bc_smoothing_sweep_bc_only.csv"
    bc_only_df.to_csv(bc_only_csv, index=False)

    outputs: Dict[str, Path] = {
        "manifest_csv": manifest_csv,
        "combined_csv": combined_csv,
        "bc_only_csv": bc_only_csv,
        "out_root": out_root,
    }

    if make_plots and len(bc_only_df):
        plot_dir = out_root / "continuous_bc_vs_bandwidth_plots"
        created = _plot_bc_vs_bandwidth_curves(
            bc_df=bc_only_df,
            out_dir=plot_dir,
            animal_id=animal_id,
            dpi=int(dpi),
        )
        plot_manifest = pd.DataFrame(
            [
                {"kind": key.split("::", 1)[0], "name": key.split("::", 1)[1], "plot_path": str(path)}
                for key, path in created.items()
            ]
        )
        plot_manifest_csv = out_root / f"{animal_id}_continuous_bc_smoothing_sweep_plot_manifest.csv"
        plot_manifest.to_csv(plot_manifest_csv, index=False)
        outputs["plot_dir"] = plot_dir
        outputs["plot_manifest_csv"] = plot_manifest_csv

    print("\n[continuous-sweep] Done.")
    print(f"[continuous-sweep] Manifest: {manifest_csv}")
    print(f"[continuous-sweep] Combined all-metrics CSV: {combined_csv}")
    print(f"[continuous-sweep] BC-only CSV: {bc_only_csv}")
    if len(manifest_df):
        print(f"[continuous-sweep] Successful runs: {(manifest_df['status'] == 'ok').sum()} / {len(manifest_df)}")

    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_bc_continuous_smoothing_sweep.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--npz-path", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument(
        "--out-root",
        default="/Volumes/my_own_SSD/bc_continuous_smoothing_sweep",
        type=str,
    )

    p.add_argument(
        "--module-name",
        default="umap_pre_post_early_late_cluster_variance_bc",
        type=str,
        help="Module that defines UMAPEarlyLateConfig and the helper functions reused here.",
    )

    p.add_argument("--bw-start", default=0.10, type=float)
    p.add_argument("--bw-stop", default=1.00, type=float)
    p.add_argument("--bw-step", default=0.10, type=float)
    p.add_argument(
        "--bw-values",
        default=None,
        type=str,
        help="Optional comma-separated explicit bandwidth values. Overrides --bw-start/stop/step.",
    )
    p.add_argument("--kde-kernel", default="gaussian", choices=["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])
    p.add_argument("--kde-grid-size", default=180, type=int)
    p.add_argument("--kde-pad-fraction", default=0.03, type=float)
    p.add_argument("--no-plot-bc-curves", action="store_true")

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

    p.add_argument("--seed", default=0, type=int, help="Use -1 for no seed / faster UMAP")
    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--overlap-density-bins", default=180, type=int)
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)
    p.add_argument("--metadata-sheet", default=None, type=str)

    return p



def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.bw_values:
        bandwidth_values = [float(x.strip()) for x in str(args.bw_values).split(",") if x.strip()]
    else:
        bandwidth_values = _build_float_sweep(args.bw_start, args.bw_stop, args.bw_step)

    if any(float(v) <= 0 for v in bandwidth_values):
        raise ValueError("All bandwidth values must be > 0.")
    if int(args.kde_grid_size) < 25:
        raise ValueError("kde-grid-size should usually be at least 25.")

    run_continuous_bc_smoothing_sweep(
        npz_path=Path(args.npz_path),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_root=Path(args.out_root),
        module_name=str(args.module_name),
        bandwidth_values=bandwidth_values,
        kde_kernel=str(args.kde_kernel),
        kde_grid_size=int(args.kde_grid_size),
        kde_pad_fraction=float(args.kde_pad_fraction),
        make_plots=not bool(args.no_plot_bc_curves),
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
        max_points_per_period_for_plot=(
            int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None
        ),
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
    )


if __name__ == "__main__":
    main()
