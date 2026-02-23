#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_syllable_clusters.py

Levina–Bickel (LB) intrinsic dimension estimation for HDBSCAN clusters in .npz files.

This script supports TWO workflows:

A) Scalar cluster dimensionality (optionally with k-sweep plots)
---------------------------------------------------------------
- For each NPZ:
    - For each HDBSCAN cluster:
        - Compute a single LB intrinsic dimension estimate (scalar)
- Optionally run a kNN sweep (k ~ 2..min(100, n-1)) per cluster and save plots.

B) Pre vs Post treatment dimensionality per cluster + visualization
-------------------------------------------------------------------
- Uses a metadata Excel to get each animal's treatment date + lesion hit type
- Splits timebins into pre vs post treatment
- Computes LB dimension per cluster for pre and post
- Visualizes Δ dimension (post - pre) grouped by:
    1) Lesion hit type group:
         - Combined (visible ML + not visible)
         - Area X visible (single hit)
         - sham saline injection
    2) Variance tier:
         - High variance syllables (top 30% by variance within animal)
         - Low variance syllables (remaining 70%)

Inputs expected in each NPZ
---------------------------
Required keys:
  - array_key (default "predictions"): shape (N, D)
  - cluster_key / label_key (default "hdbscan_labels"): shape (N,)

Optional but recommended keys:
  - file_key (default "file"): shape (N,), used to parse recording dates if no date array exists
  - date_key (optional): a direct date array key if available
  - syllable_key (default "ground_truth_labels"): shape (N,), used to map cluster -> dominant syllable label
  - vocalization_key (default "vocalization"): shape (N,), if present we can filter to vocalization==1

Dependencies
------------
numpy, pandas, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Utilities
# ============================================================
def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _nan_agg(a: np.ndarray, method: str, axis=None):
    method = method.lower()
    if method == "mean":
        return np.nanmean(a, axis=axis)
    if method == "median":
        return np.nanmedian(a, axis=axis)
    raise ValueError(f"Unknown aggregation method: {method!r} (use 'mean' or 'median')")


def _make_nn(n_neighbors: int, n_jobs: int) -> NearestNeighbors:
    """sklearn versions vary in whether NearestNeighbors accepts n_jobs."""
    try:
        return NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=n_jobs)
    except TypeError:
        return NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")


def _write_rows_csv(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(csv_path, "w", newline="") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _choose_tick_positions(k_arr: np.ndarray, tick_every: int) -> np.ndarray:
    tick_every = max(1, int(tick_every))
    ticks = k_arr[::tick_every]
    if ticks.size == 0:
        ticks = k_arr
    if ticks[-1] != k_arr[-1]:
        ticks = np.concatenate([ticks, [k_arr[-1]]])
    return ticks


def _format_k_ticks_with_percent(k_ticks: np.ndarray, n_cluster: int) -> List[str]:
    labels: List[str] = []
    for k in k_ticks:
        pct = 100.0 * float(k) / float(n_cluster)
        labels.append(f"{int(k)}\n{int(round(pct))}%")
    return labels


def _jitter(x: float, n: int, scale: float = 0.06, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return x + rng.uniform(-scale, scale, size=n)


# ============================================================
# Levina–Bickel MLE core
# ============================================================
def _compute_knn_logs(
    X: np.ndarray,
    k_max: int,
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute kNN distances up to k_max.

    Returns:
      - T: distances to 1..k_max nearest neighbors (excluding self), shape (n, k_max)
      - logT: log(clamped(T)), shape (n, k_max)
      - cumsum_logT: cumulative sum of logT along neighbors, shape (n, k_max)
    """
    X = _safe_2d(X)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points; got n={n}")
    if k_max > n - 1:
        raise ValueError(f"k_max={k_max} but n={n}; need k_max <= n-1")

    nn = _make_nn(n_neighbors=k_max + 1, n_jobs=n_jobs)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    T = distances[:, 1:]  # drop self-distance (0)
    logT = np.log(np.maximum(T, eps))
    cumsum_logT = np.cumsum(logT, axis=1)
    return T, logT, cumsum_logT


def levina_bickel_mle_scalar(
    X: np.ndarray,
    k1: int = 10,
    k2: int = 20,
    k_agg: str = "mean",        # aggregate across k per point
    point_agg: str = "median",  # aggregate across points
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> Tuple[float, int]:
    """
    Single LB scalar estimate for dataset X.

    Returns:
      (m_hat, k2_used)
    """
    X = _safe_2d(X)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need >=3 points; got n={n}")
    if k1 < 2:
        raise ValueError("k1 must be >= 2")

    k2_used = min(int(k2), n - 1)
    if k2_used < k1:
        raise ValueError(f"k2_used={k2_used} < k1={k1}; cluster too small for requested k-range.")

    _, logT, cumsum_logT = _compute_knn_logs(X, k_max=k2_used, eps=eps, n_jobs=n_jobs)
    ks = np.arange(k1, k2_used + 1)

    m_point_by_k = np.full((n, len(ks)), np.nan, dtype=float)

    # Identity:
    # sum_{j=1}^{k-1} log(T_k/T_j) = (k-1)*log(T_k) - sum_{j=1}^{k-1} log(T_j)
    with np.errstate(divide="ignore", invalid="ignore"):
        for idx, k in enumerate(ks):
            logT_k = logT[:, k - 1]
            sum_logT_j = cumsum_logT[:, k - 2]
            log_sums = (k - 1) * logT_k - sum_logT_j

            valid = np.isfinite(log_sums) & (log_sums > 0)
            m_k = np.full(n, np.nan, dtype=float)
            m_k[valid] = (k - 1) / log_sums[valid]
            m_point_by_k[:, idx] = m_k

    m_per_point = _nan_agg(m_point_by_k, k_agg, axis=1)
    m_hat = float(_nan_agg(m_per_point, point_agg, axis=0))
    return m_hat, k2_used


def levina_bickel_knn_sweep(
    X: np.ndarray,
    k_min_requested: int = 0,
    k_max_requested: int = 100,
    k_step: int = 1,
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep LB m_hat(k) for ONE dataset X.

    Requested range can start at 0, but LB requires k>=2:
      k_min_used = max(2, k_min_requested)
      k_max_used = min(k_max_requested, n-1)

    Returns:
      k_arr, m_hat_arr, frac_valid_arr
    """
    X = _safe_2d(X)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points; got n={n}")

    k_min_used = max(2, int(k_min_requested))
    k_max_used = min(int(k_max_requested), n - 1)
    if k_max_used < k_min_used:
        raise ValueError(f"Cluster too small for sweep: n={n}, k_min_used={k_min_used}, k_max_used={k_max_used}")

    k_arr = np.arange(k_min_used, k_max_used + 1, int(k_step), dtype=int)
    _, logT, cumsum_logT = _compute_knn_logs(X, k_max=int(k_arr.max()), eps=eps, n_jobs=n_jobs)

    m_hat_list: List[float] = []
    frac_valid_list: List[float] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        for k in k_arr:
            logT_k = logT[:, k - 1]
            sum_logT_j = cumsum_logT[:, k - 2]
            log_sums = (k - 1) * logT_k - sum_logT_j

            valid = np.isfinite(log_sums) & (log_sums > 0)
            m_k = np.full(n, np.nan, dtype=float)
            m_k[valid] = (k - 1) / log_sums[valid]

            m_hat_list.append(float(_nan_agg(m_k, point_agg)))
            frac_valid_list.append(float(np.mean(valid)))

    return k_arr, np.asarray(m_hat_list, dtype=float), np.asarray(frac_valid_list, dtype=float)


# ============================================================
# Scalar workflow: per-cluster LB + optional sweeps
# ============================================================
def plot_cluster_sweep(
    k_arr: np.ndarray,
    m_hat_arr: np.ndarray,
    n_cluster: int,
    cluster_id: int,
    point_agg: str,
    tick_every: int = 5,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_arr, m_hat_arr, marker="o")
    ax.set_xlabel("k (KNN)\n(k as % of cluster size)")
    ax.set_ylabel(f"Estimated intrinsic dimension (LB, point_agg={point_agg})")
    ax.set_title(f"Levina–Bickel k sweep (cluster {cluster_id}, n={n_cluster})")
    ax.grid(True, alpha=0.3)

    k_ticks = _choose_tick_positions(k_arr, tick_every=tick_every)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels(_format_k_ticks_with_percent(k_ticks, n_cluster=n_cluster), fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved cluster sweep plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_aggregate_curve_across_clusters(
    k_global: np.ndarray,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
    n_clusters_arr: np.ndarray,
    title: str,
    tick_every: int = 5,
    min_clusters_per_k: int = 1,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    mask = np.isfinite(mean_arr) & (n_clusters_arr >= int(min_clusters_per_k))
    if not np.any(mask):
        print("Aggregate curve: no k values met min_clusters_per_k; skipping plot.")
        return

    k = k_global[mask]
    mean = mean_arr[mask]
    std = std_arr[mask]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(k, mean, marker="o", label="Mean across clusters")
    ax.fill_between(k, mean - std, mean + std, alpha=0.2, label="±1 std")
    ax.set_xlabel("k (KNN)")
    ax.set_ylabel("Estimated intrinsic dimension (LB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    k_ticks = _choose_tick_positions(k, tick_every=tick_every)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels([str(int(x)) for x in k_ticks], fontsize=8)

    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved aggregate curve plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def estimate_lb_mle_for_npz_clusters(
    npz_path: str | Path,
    array_key: str = "predictions",
    label_key: str = "hdbscan_labels",
    include_noise: bool = False,
    min_cluster_size: int = 10,
    k1: int = 10,
    k2: int = 20,
    k_agg: str = "mean",
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Compute a scalar LB MLE per HDBSCAN cluster in one NPZ.
    """
    npz_path = Path(npz_path)
    rows: List[Dict[str, Any]] = []

    try:
        data = np.load(npz_path, allow_pickle=True)
        if array_key not in data or label_key not in data:
            missing = [k for k in (array_key, label_key) if k not in data]
            raise KeyError(f"Missing keys in npz: {missing}. Keys present: {list(data.keys())}")

        X = _safe_2d(data[array_key])
        labels = np.asarray(data[label_key])

        if X.shape[0] != labels.shape[0]:
            raise ValueError(f"X has n={X.shape[0]} rows but labels has n={labels.shape[0]} entries")

        cluster_ids = np.unique(labels)
        if not include_noise:
            cluster_ids = cluster_ids[cluster_ids != -1]

        for cid in cluster_ids:
            mask = (labels == cid)
            n_c = int(mask.sum())

            if n_c < max(3, int(min_cluster_size)):
                rows.append({
                    "npz_path": str(npz_path),
                    "cluster_id": int(cid),
                    "n_cluster": n_c,
                    "m_hat": "",
                    "k1": int(k1),
                    "k2_used": "",
                    "array_key": array_key,
                    "label_key": label_key,
                    "error": f"skipped: n_cluster<{min_cluster_size}",
                })
                continue

            Xc = X[mask]
            try:
                m_hat, k2_used = levina_bickel_mle_scalar(
                    Xc, k1=int(k1), k2=int(k2),
                    k_agg=str(k_agg), point_agg=str(point_agg),
                    eps=float(eps), n_jobs=int(n_jobs),
                )
                rows.append({
                    "npz_path": str(npz_path),
                    "cluster_id": int(cid),
                    "n_cluster": n_c,
                    "m_hat": float(m_hat),
                    "k1": int(k1),
                    "k2_used": int(k2_used),
                    "array_key": array_key,
                    "label_key": label_key,
                    "error": "",
                })
            except Exception as e:
                rows.append({
                    "npz_path": str(npz_path),
                    "cluster_id": int(cid),
                    "n_cluster": n_c,
                    "m_hat": "",
                    "k1": int(k1),
                    "k2_used": "",
                    "array_key": array_key,
                    "label_key": label_key,
                    "error": f"{type(e).__name__}: {e}",
                })

    except Exception as e:
        rows.append({
            "npz_path": str(npz_path),
            "cluster_id": "",
            "n_cluster": "",
            "m_hat": "",
            "k1": int(k1),
            "k2_used": "",
            "array_key": array_key,
            "label_key": label_key,
            "error": f"{type(e).__name__}: {e}",
        })

    return rows


def run_sweeps_for_npz(
    npz_path: str | Path,
    out_dir: str | Path,
    array_key: str = "predictions",
    label_key: str = "hdbscan_labels",
    include_noise: bool = False,
    min_cluster_size: int = 10,
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
    show: bool = False,
    k_min: int = 0,
    k_max: int = 100,
    k_step: int = 1,
    tick_every: int = 5,
    min_clusters_per_k: int = 1,
) -> Dict[str, str]:
    """
    For ONE NPZ:
      - run per-cluster sweeps (save PNG+CSV per cluster)
      - compute and plot aggregate curve across clusters
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = npz_path.stem

    data = np.load(npz_path, allow_pickle=True)
    if array_key not in data or label_key not in data:
        raise KeyError(f"Missing keys in {npz_path.name}: need {array_key!r} and {label_key!r}. Present: {list(data.keys())}")

    X = _safe_2d(data[array_key])
    labels = np.asarray(data[label_key])
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X has n={X.shape[0]} rows but labels has n={labels.shape[0]} entries")

    cluster_ids = np.unique(labels)
    if not include_noise:
        cluster_ids = cluster_ids[cluster_ids != -1]

    # sort clusters by size desc
    cluster_sizes = [(int(cid), int(np.sum(labels == cid))) for cid in cluster_ids]
    cluster_sizes.sort(key=lambda t: t[1], reverse=True)

    cluster_sweeps: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    summary_rows: List[Dict[str, Any]] = []

    cluster_dir = out_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for cid, n_c in cluster_sizes:
        if n_c < max(3, int(min_cluster_size)):
            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "k_min_used": "",
                "k_max_used": "",
                "point_agg": point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"skipped: n_cluster<{min_cluster_size}",
            })
            continue

        Xc = X[labels == cid]
        try:
            k_arr, m_hat_arr, frac_valid_arr = levina_bickel_knn_sweep(
                Xc,
                k_min_requested=k_min,
                k_max_requested=k_max,
                k_step=k_step,
                point_agg=point_agg,
                eps=eps,
                n_jobs=n_jobs,
            )

            # save per-cluster CSV
            csv_path = cluster_dir / f"{base}_cluster{cid}_{array_key}_sweep_{ts}.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("k,m_hat,frac_valid\n")
                for i in range(len(k_arr)):
                    f.write(f"{int(k_arr[i])},{m_hat_arr[i]},{frac_valid_arr[i]}\n")

            # save per-cluster plot
            plot_path = cluster_dir / f"{base}_cluster{cid}_{array_key}_sweep_{ts}.png"
            plot_cluster_sweep(
                k_arr=k_arr,
                m_hat_arr=m_hat_arr,
                n_cluster=n_c,
                cluster_id=cid,
                point_agg=point_agg,
                tick_every=tick_every,
                save_path=str(plot_path),
                show=show,
            )

            cluster_sweeps[cid] = (k_arr, m_hat_arr, frac_valid_arr, n_c)

            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "k_min_used": int(k_arr.min()) if len(k_arr) else "",
                "k_max_used": int(k_arr.max()) if len(k_arr) else "",
                "point_agg": point_agg,
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "error": "",
            })

        except Exception as e:
            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "k_min_used": "",
                "k_max_used": "",
                "point_agg": point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"{type(e).__name__}: {e}",
            })

    sweep_summary_csv = out_dir / f"{base}_{array_key}_sweep_summary_{ts}.csv"
    _write_rows_csv(sweep_summary_csv, summary_rows)

    # Aggregate curve across clusters
    k_min_global = max(2, int(k_min))
    k_max_global = int(k_max)
    k_global = np.arange(k_min_global, k_max_global + 1, int(k_step), dtype=int)

    mean_list: List[float] = []
    std_list: List[float] = []
    median_list: List[float] = []
    n_clusters_list: List[int] = []

    for k in k_global:
        vals: List[float] = []
        for _, (k_arr, m_hat_arr, _, _) in cluster_sweeps.items():
            if k >= int(k_arr.min()) and k <= int(k_arr.max()):
                idx = int(k - int(k_arr.min()))
                if 0 <= idx < len(m_hat_arr) and np.isfinite(m_hat_arr[idx]):
                    vals.append(float(m_hat_arr[idx]))

        if len(vals) == 0:
            mean_list.append(np.nan)
            std_list.append(np.nan)
            median_list.append(np.nan)
            n_clusters_list.append(0)
        else:
            arr = np.asarray(vals, dtype=float)
            mean_list.append(float(np.mean(arr)))
            std_list.append(float(np.std(arr)))
            median_list.append(float(np.median(arr)))
            n_clusters_list.append(int(arr.size))

    mean_arr = np.asarray(mean_list, dtype=float)
    std_arr = np.asarray(std_list, dtype=float)
    median_arr = np.asarray(median_list, dtype=float)
    n_clusters_arr = np.asarray(n_clusters_list, dtype=int)

    agg_csv = out_dir / f"{base}_{array_key}_aggregate_curve_{ts}.csv"
    with open(agg_csv, "w", newline="") as f:
        f.write("k,mean,std,median,n_clusters\n")
        for i in range(len(k_global)):
            f.write(f"{int(k_global[i])},{mean_arr[i]},{std_arr[i]},{median_arr[i]},{int(n_clusters_arr[i])}\n")

    agg_plot = out_dir / f"{base}_{array_key}_aggregate_curve_{ts}.png"
    plot_aggregate_curve_across_clusters(
        k_global=k_global,
        mean_arr=mean_arr,
        std_arr=std_arr,
        n_clusters_arr=n_clusters_arr,
        title=f"Aggregate Levina–Bickel curve across clusters ({base}, array={array_key})",
        tick_every=tick_every,
        min_clusters_per_k=min_clusters_per_k,
        save_path=str(agg_plot),
        show=show,
    )

    return {
        "sweep_summary_csv": str(sweep_summary_csv),
        "aggregate_csv": str(agg_csv),
        "aggregate_plot": str(agg_plot),
        "out_dir": str(out_dir),
    }


def run_root_directory_lb_mle(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    array_key: str = "predictions",
    label_key: str = "hdbscan_labels",
    include_noise: bool = False,
    min_cluster_size: int = 10,
    k1: int = 10,
    k2: int = 20,
    k_agg: str = "mean",
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
    save_per_npz: bool = True,
    do_sweeps: bool = False,
    sweep_show: bool = False,
    sweep_k_min: int = 0,
    sweep_k_max: int = 100,
    sweep_k_step: int = 1,
    sweep_tick_every: int = 5,
    sweep_min_clusters_per_k: int = 1,
) -> str:
    """
    Wrapper: compute scalar LB MLE per cluster for every NPZ under root_dir.
    Optionally also run sweeps + plots per NPZ.
    """
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / "lb_cluster_mle_outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(root_dir.rglob("*.npz") if recursive else root_dir.glob("*.npz"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    master_rows: List[Dict[str, Any]] = []

    for npz_path in npz_paths:
        print(f"[LB scalar] processing: {npz_path}")
        rows = estimate_lb_mle_for_npz_clusters(
            npz_path=npz_path,
            array_key=array_key,
            label_key=label_key,
            include_noise=include_noise,
            min_cluster_size=min_cluster_size,
            k1=k1,
            k2=k2,
            k_agg=k_agg,
            point_agg=point_agg,
            eps=eps,
            n_jobs=n_jobs,
        )
        master_rows.extend(rows)

        if save_per_npz:
            per_dir = out_dir / npz_path.stem
            per_dir.mkdir(parents=True, exist_ok=True)
            per_csv = per_dir / f"{npz_path.stem}_cluster_mle_{ts}.csv"
            _write_rows_csv(per_csv, rows)
            print(f"  saved per-npz scalar CSV: {per_csv}")

        if do_sweeps:
            print(f"[LB sweep] processing: {npz_path}")
            per_dir = out_dir / npz_path.stem / "sweeps"
            per_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_sweeps_for_npz(
                    npz_path=npz_path,
                    out_dir=per_dir,
                    array_key=array_key,
                    label_key=label_key,
                    include_noise=include_noise,
                    min_cluster_size=min_cluster_size,
                    point_agg=point_agg,
                    eps=eps,
                    n_jobs=n_jobs,
                    show=sweep_show,
                    k_min=sweep_k_min,
                    k_max=sweep_k_max,
                    k_step=sweep_k_step,
                    tick_every=sweep_tick_every,
                    min_clusters_per_k=sweep_min_clusters_per_k,
                )
            except Exception as e:
                print(f"  sweep ERROR for {npz_path.name}: {type(e).__name__}: {e}")

    master_csv = out_dir / f"LB_master_cluster_mle_{ts}.csv"
    _write_rows_csv(master_csv, master_rows)
    print(f"\n[LB] MASTER CSV saved: {master_csv}")
    return str(master_csv)


# ============================================================
# Pre/Post workflow: metadata + variance tiers + plots
# ============================================================
def load_metadata_treatment_and_hit_type(
    metadata_xlsx: str | Path,
    sheet_name: str = "metadata_with_hit_type",
    animal_col: str = "Animal ID",
    date_col: str = "Treatment date",
    hit_col: str = "Lesion hit type",
) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)
    for c in (animal_col, date_col, hit_col):
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} in {sheet_name!r}. Columns present: {list(df.columns)}")

    out = df[[animal_col, date_col, hit_col]].dropna(subset=[animal_col, date_col]).drop_duplicates().copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    out[animal_col] = out[animal_col].astype(str)
    out = out.groupby(animal_col, as_index=False).first()
    return out


def map_hit_type_to_group(hit_type: str) -> str:
    if not isinstance(hit_type, str):
        return "Unknown"
    s = hit_type.lower()

    if "sham" in s:
        return "sham saline injection"
    if "single hit" in s:
        return "Area X visible (single hit)"
    if ("medial+lateral" in s) or ("medial + lateral" in s) or ("not visible" in s) or ("large lesion" in s):
        return "Combined (visible ML + not visible)"
    return "Other"


def build_variance_tier_map(
    stats_csv: str | Path,
    animal_col: str = "Animal ID",
    syll_col: str = "Syllable",
    group_col: str = "Group",
    group_value: str = "Post",
    variance_col: str = "Pre_Variance_ms2",
    high_quantile: float = 0.70,  # top 30% => >= 70th percentile
) -> Dict[Tuple[str, int], str]:
    df = pd.read_csv(stats_csv)
    for c in (animal_col, syll_col, group_col, variance_col):
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} in stats CSV. Columns present: {list(df.columns)}")

    df = df[df[group_col] == group_value].copy()
    df = df.dropna(subset=[animal_col, syll_col, variance_col])
    df[animal_col] = df[animal_col].astype(str)
    df[syll_col] = df[syll_col].astype(int)
    df[variance_col] = pd.to_numeric(df[variance_col], errors="coerce")
    df = df.dropna(subset=[variance_col])

    tier_map: Dict[Tuple[str, int], str] = {}
    for animal_id, g in df.groupby(animal_col):
        vals = g[[syll_col, variance_col]].dropna()
        if vals.empty:
            continue
        thresh = float(vals[variance_col].quantile(high_quantile))
        for _, row in vals.iterrows():
            syll = int(row[syll_col])
            v = float(row[variance_col])
            tier_map[(animal_id, syll)] = "high" if v >= thresh else "low"
    return tier_map


_DATE_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})"),  # 2024-04-09 / 2024_04_09
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),          # 20240409
]


def _parse_date_from_string(s: str) -> Optional[date]:
    if not isinstance(s, str):
        return None
    for pat in _DATE_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                y = int(m.group("y"))
                mo = int(m.group("m"))
                d = int(m.group("d"))
                return date(y, mo, d)
            except Exception:
                return None
    return None


def infer_point_dates_from_npz(
    data: np.lib.npyio.NpzFile,
    n: int,
    date_key: Optional[str] = None,
    file_key: str = "file",
    preferred_date_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Return an array of python datetime.date (dtype=object), length n.

    Strategy:
      1) If date_key is provided and exists, use it.
      2) Else try preferred_date_keys.
      3) Else parse from file_key strings.
    """
    if preferred_date_keys is None:
        preferred_date_keys = ["recording_date", "date", "date_time", "datetime", "timestamp", "t", "times"]

    keys_to_try: List[str] = []
    if date_key:
        keys_to_try.append(date_key)
    for k in preferred_date_keys:
        if k not in keys_to_try:
            keys_to_try.append(k)

    for k in keys_to_try:
        if k in data:
            arr = data[k]
            if np.issubdtype(arr.dtype, np.datetime64):
                dt64 = arr.astype("datetime64[D]")
                return np.array([pd.Timestamp(x).date() for x in dt64], dtype=object)

            if arr.dtype.kind in ("U", "S", "O"):
                s = pd.Series(arr.astype(str))
                dt_parsed = pd.to_datetime(s, errors="coerce")
                if dt_parsed.notna().mean() > 0.80:
                    return np.array([d.date() if pd.notna(d) else None for d in dt_parsed], dtype=object)

            try:
                s = pd.Series(arr)
                dt_parsed = pd.to_datetime(s, errors="coerce")
                if dt_parsed.notna().mean() > 0.80:
                    return np.array([d.date() if pd.notna(d) else None for d in dt_parsed], dtype=object)
            except Exception:
                pass

    if file_key in data:
        files = data[file_key]
        if len(files) != n:
            raise ValueError(
                f"Found {file_key!r} but length {len(files)} != n={n}. "
                f"Provide a usable date_key instead."
            )

        out = np.empty(n, dtype=object)
        n_ok = 0
        for i, f in enumerate(files):
            d = _parse_date_from_string(str(f))
            out[i] = d
            if d is not None:
                n_ok += 1

        if n_ok / max(1, n) < 0.80:
            raise ValueError(
                f"Could not parse dates reliably from {file_key!r}: only {n_ok}/{n} parsed. "
                f"Provide a date_key in the NPZ."
            )
        return out

    raise KeyError("Could not infer dates: no usable date key found and file_key not present.")


def split_pre_post_masks(
    point_dates: np.ndarray,
    treatment_date: date,
    include_treatment_day_in_post: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if include_treatment_day_in_post:
        pre = np.array([(d is not None) and (d < treatment_date) for d in point_dates], dtype=bool)
        post = np.array([(d is not None) and (d >= treatment_date) for d in point_dates], dtype=bool)
    else:
        pre = np.array([(d is not None) and (d <= treatment_date) for d in point_dates], dtype=bool)
        post = np.array([(d is not None) and (d > treatment_date) for d in point_dates], dtype=bool)
    return pre, post


def cluster_mode_syllable(
    cluster_ids: np.ndarray,
    syllable_labels: np.ndarray,
    cluster_id: int,
    ignore_values: Tuple[int, ...] = (-1,),
) -> Optional[int]:
    mask = (cluster_ids == cluster_id)
    if not np.any(mask):
        return None
    vals = np.asarray(syllable_labels[mask])
    if vals.size == 0:
        return None
    if ignore_values:
        vals = vals[~np.isin(vals.astype(int), np.array(ignore_values, dtype=int))]
    if vals.size == 0:
        return None
    uniq, counts = np.unique(vals.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])


@dataclass
class PrePostConfig:
    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    syllable_key: str = "ground_truth_labels"
    file_key: str = "file"
    date_key: Optional[str] = None

    include_noise: bool = False
    use_vocalization_only: bool = True
    vocalization_key: str = "vocalization"

    min_points_per_period: int = 50
    include_treatment_day_in_post: bool = True

    # LB params
    k1: int = 10
    k2: int = 20
    k_agg: str = "mean"
    point_agg: str = "median"
    eps: float = 1e-12
    n_jobs: int = 1

    # optional speed cap
    max_points_per_cluster_period: Optional[int] = None
    subsample_seed: int = 0


def animal_id_from_npz_path(npz_path: Path, metadata_animals: Optional[set] = None) -> str:
    stem = npz_path.stem
    if metadata_animals and stem in metadata_animals:
        return stem
    tok = stem.split("_")[0]
    if metadata_animals and tok in metadata_animals:
        return tok
    return stem


def compute_pre_post_dimensionality_for_npz(
    npz_path: str | Path,
    treatment_date: date,
    hit_type_raw: str,
    hit_group: str,
    variance_tier_map: Dict[Tuple[str, int], str],
    config: PrePostConfig,
    animal_id: str,
) -> List[Dict[str, Any]]:
    npz_path = Path(npz_path)
    rows: List[Dict[str, Any]] = []

    try:
        data = np.load(npz_path, allow_pickle=True)

        if config.array_key not in data:
            raise KeyError(f"Missing array_key {config.array_key!r} in {npz_path.name}")
        if config.cluster_key not in data:
            raise KeyError(f"Missing cluster_key {config.cluster_key!r} in {npz_path.name}")

        X = _safe_2d(data[config.array_key])
        cluster_ids = np.asarray(data[config.cluster_key])
        n = X.shape[0]
        if cluster_ids.shape[0] != n:
            raise ValueError(f"X has n={n} but clusters has n={cluster_ids.shape[0]}")

        keep_mask = np.ones(n, dtype=bool)
        if config.use_vocalization_only and (config.vocalization_key in data):
            voc = np.asarray(data[config.vocalization_key]).astype(int)
            if voc.shape[0] == n:
                keep_mask &= (voc == 1)

        point_dates = infer_point_dates_from_npz(
            data=data, n=n, date_key=config.date_key, file_key=config.file_key
        )
        pre_mask, post_mask = split_pre_post_masks(
            point_dates=point_dates,
            treatment_date=treatment_date,
            include_treatment_day_in_post=config.include_treatment_day_in_post,
        )
        pre_mask &= keep_mask
        post_mask &= keep_mask

        syllable_labels = None
        if config.syllable_key in data:
            tmp = np.asarray(data[config.syllable_key])
            if tmp.shape[0] == n:
                syllable_labels = tmp

        unique_clusters = np.unique(cluster_ids)
        if not config.include_noise:
            unique_clusters = unique_clusters[unique_clusters != -1]

        rng = np.random.RandomState(config.subsample_seed)

        for cid in unique_clusters:
            cid = int(cid)
            c_mask = (cluster_ids == cid)

            idx_pre = np.where(c_mask & pre_mask)[0]
            idx_post = np.where(c_mask & post_mask)[0]
            n_pre = int(idx_pre.size)
            n_post = int(idx_post.size)

            syll = None
            if syllable_labels is not None:
                syll = cluster_mode_syllable(cluster_ids, syllable_labels, cid, ignore_values=(-1,))

            var_tier = "unknown"
            if syll is not None:
                var_tier = variance_tier_map.get((animal_id, int(syll)), "unknown")

            if (n_pre < config.min_points_per_period) or (n_post < config.min_points_per_period):
                rows.append({
                    "npz_path": str(npz_path),
                    "animal_id": animal_id,
                    "cluster_id": cid,
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "m_pre": "",
                    "m_post": "",
                    "delta_m": "",
                    "hit_type_raw": hit_type_raw,
                    "hit_group": hit_group,
                    "syllable_label": "" if syll is None else int(syll),
                    "variance_tier": var_tier,
                    "error": f"skipped: insufficient points (min={config.min_points_per_period})",
                })
                continue

            if config.max_points_per_cluster_period is not None:
                cap = int(config.max_points_per_cluster_period)
                if n_pre > cap:
                    idx_pre = rng.choice(idx_pre, size=cap, replace=False)
                    n_pre = int(idx_pre.size)
                if n_post > cap:
                    idx_post = rng.choice(idx_post, size=cap, replace=False)
                    n_post = int(idx_post.size)

            try:
                m_pre, _ = levina_bickel_mle_scalar(
                    X[idx_pre],
                    k1=config.k1, k2=config.k2,
                    k_agg=config.k_agg, point_agg=config.point_agg,
                    eps=config.eps, n_jobs=config.n_jobs,
                )
                m_post, _ = levina_bickel_mle_scalar(
                    X[idx_post],
                    k1=config.k1, k2=config.k2,
                    k_agg=config.k_agg, point_agg=config.point_agg,
                    eps=config.eps, n_jobs=config.n_jobs,
                )

                rows.append({
                    "npz_path": str(npz_path),
                    "animal_id": animal_id,
                    "cluster_id": cid,
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "m_pre": float(m_pre),
                    "m_post": float(m_post),
                    "delta_m": float(m_post - m_pre),
                    "hit_type_raw": hit_type_raw,
                    "hit_group": hit_group,
                    "syllable_label": "" if syll is None else int(syll),
                    "variance_tier": var_tier,
                    "error": "",
                })

            except Exception as e:
                rows.append({
                    "npz_path": str(npz_path),
                    "animal_id": animal_id,
                    "cluster_id": cid,
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "m_pre": "",
                    "m_post": "",
                    "delta_m": "",
                    "hit_type_raw": hit_type_raw,
                    "hit_group": hit_group,
                    "syllable_label": "" if syll is None else int(syll),
                    "variance_tier": var_tier,
                    "error": f"{type(e).__name__}: {e}",
                })

    except Exception as e:
        rows.append({
            "npz_path": str(npz_path),
            "animal_id": animal_id,
            "cluster_id": "",
            "n_pre": "",
            "n_post": "",
            "m_pre": "",
            "m_post": "",
            "delta_m": "",
            "hit_type_raw": hit_type_raw,
            "hit_group": hit_group,
            "syllable_label": "",
            "variance_tier": "unknown",
            "error": f"{type(e).__name__}: {e}",
        })

    return rows


def plot_delta_by_hit_and_variance(
    df: pd.DataFrame,
    out_png: str | Path,
    title: str = "Δ intrinsic dimension (post − pre) by lesion hit type and variance tier",
    hit_order: Optional[List[str]] = None,
    tier_order: Optional[List[str]] = None,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if hit_order is None:
        hit_order = [
            "sham saline injection",
            "Area X visible (single hit)",
            "Combined (visible ML + not visible)",
        ]
    if tier_order is None:
        tier_order = ["high", "low"]

    dd = df.copy()
    dd["delta_m"] = pd.to_numeric(dd["delta_m"], errors="coerce")
    dd = dd.dropna(subset=["delta_m"])
    dd = dd[dd["variance_tier"].isin(tier_order)]
    dd = dd[dd["hit_group"].isin(hit_order)]

    fig, axes = plt.subplots(len(hit_order), 1, figsize=(9, 10), sharex=True)
    if len(hit_order) == 1:
        axes = [axes]

    for ax, hit in zip(axes, hit_order):
        sub = dd[dd["hit_group"] == hit]
        data = []
        ns = []
        for tier in tier_order:
            v = sub[sub["variance_tier"] == tier]["delta_m"].values.astype(float)
            data.append(v)
            ns.append(v.size)

        bp = ax.boxplot(data, labels=[f"{t}\n(n={n})" for t, n in zip(tier_order, ns)], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.5)

        for i, v in enumerate(data, start=1):
            if v.size:
                ax.scatter(_jitter(i, v.size, seed=42 + i), v, s=18, alpha=0.7)

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(hit)
        ax.set_ylabel("Δ dimension")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def plot_pre_vs_post_scatter(
    df: pd.DataFrame,
    out_png: str | Path,
    title: str = "Pre vs Post intrinsic dimension by lesion hit type (marker = variance tier)",
    hit_order: Optional[List[str]] = None,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if hit_order is None:
        hit_order = [
            "sham saline injection",
            "Area X visible (single hit)",
            "Combined (visible ML + not visible)",
        ]

    dd = df.copy()
    dd["m_pre"] = pd.to_numeric(dd["m_pre"], errors="coerce")
    dd["m_post"] = pd.to_numeric(dd["m_post"], errors="coerce")
    dd = dd.dropna(subset=["m_pre", "m_post"])
    dd = dd[dd["hit_group"].isin(hit_order)]

    fig, axes = plt.subplots(1, len(hit_order), figsize=(15, 4.8), sharex=True, sharey=True)
    if len(hit_order) == 1:
        axes = [axes]

    marker_map = {"high": "o", "low": "s", "unknown": "x"}

    all_vals = np.concatenate([dd["m_pre"].values, dd["m_post"].values]).astype(float)
    if all_vals.size:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        pad = 0.05 * (hi - lo + 1e-9)
        lims = (lo - pad, hi + pad)
    else:
        lims = (0, 1)

    for ax, hit in zip(axes, hit_order):
        sub = dd[dd["hit_group"] == hit]
        for tier, mk in marker_map.items():
            ssub = sub[sub["variance_tier"] == tier]
            if ssub.empty:
                continue
            ax.scatter(ssub["m_pre"], ssub["m_post"], s=30, alpha=0.8, marker=mk, label=tier)

        ax.plot(lims, lims, linestyle="--", linewidth=1)
        ax.set_title(hit)
        ax.set_xlabel("Pre dimension")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Post dimension")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def run_root_directory_pre_post_analysis(
    root_dir: str | Path,
    metadata_xlsx: str | Path,
    stats_csv: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    metadata_sheet: str = "metadata_with_hit_type",
    config: Optional[PrePostConfig] = None,
    variance_high_quantile: float = 0.70,
    stats_group_value: str = "Post",
    stats_variance_col: str = "Pre_Variance_ms2",
) -> Dict[str, str]:
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / "lb_pre_post_dimensionality"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = PrePostConfig()

    md = load_metadata_treatment_and_hit_type(
        metadata_xlsx=metadata_xlsx,
        sheet_name=metadata_sheet,
        animal_col="Animal ID",
        date_col="Treatment date",
        hit_col="Lesion hit type",
    )
    md["hit_group"] = md["Lesion hit type"].apply(map_hit_type_to_group)
    md_lookup = md.set_index("Animal ID")[["Treatment date", "Lesion hit type", "hit_group"]].to_dict(orient="index")
    metadata_animals = set(md["Animal ID"].astype(str).tolist())

    tier_map = build_variance_tier_map(
        stats_csv=stats_csv,
        animal_col="Animal ID",
        syll_col="Syllable",
        group_col="Group",
        group_value=stats_group_value,
        variance_col=stats_variance_col,
        high_quantile=variance_high_quantile,
    )

    npz_paths = sorted(root_dir.rglob("*.npz") if recursive else root_dir.glob("*.npz"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows: List[Dict[str, Any]] = []

    for npz_path in npz_paths:
        animal_id = animal_id_from_npz_path(npz_path, metadata_animals=metadata_animals)
        if animal_id not in md_lookup:
            all_rows.append({
                "npz_path": str(npz_path),
                "animal_id": animal_id,
                "cluster_id": "",
                "n_pre": "",
                "n_post": "",
                "m_pre": "",
                "m_post": "",
                "delta_m": "",
                "hit_type_raw": "",
                "hit_group": "",
                "syllable_label": "",
                "variance_tier": "unknown",
                "error": "skipped: animal_id not found in metadata excel",
            })
            continue

        treatment_date = md_lookup[animal_id]["Treatment date"]
        hit_type_raw = md_lookup[animal_id]["Lesion hit type"]
        hit_group = md_lookup[animal_id]["hit_group"]

        print(f"[pre/post] {animal_id}  ({hit_group})  file={npz_path.name}")

        rows = compute_pre_post_dimensionality_for_npz(
            npz_path=npz_path,
            treatment_date=treatment_date,
            hit_type_raw=hit_type_raw,
            hit_group=hit_group,
            variance_tier_map=tier_map,
            config=config,
            animal_id=animal_id,
        )
        all_rows.extend(rows)

    out_csv = out_dir / f"lb_pre_post_cluster_dimensionality_{ts}.csv"
    _write_rows_csv(out_csv, all_rows)
    print(f"\nSaved results CSV: {out_csv}")

    df = pd.DataFrame(all_rows)

    plot1 = out_dir / f"delta_dim_by_hit_and_variance_{ts}.png"
    plot_delta_by_hit_and_variance(df, plot1)

    plot2 = out_dir / f"pre_vs_post_scatter_by_hit_{ts}.png"
    plot_pre_vs_post_scatter(df, plot2)

    return {
        "results_csv": str(out_csv),
        "plot_delta": str(plot1),
        "plot_scatter": str(plot2),
        "out_dir": str(out_dir),
    }


# ============================================================
# CLI (two subcommands)
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Levina–Bickel intrinsic dimension for HDBSCAN clusters in NPZ files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- scalar ----
    p_scalar = sub.add_parser("scalar", help="Compute scalar LB per cluster (optionally sweeps).")
    p_scalar.add_argument("--root-dir", type=str, required=True)
    p_scalar.add_argument("--out-dir", type=str, default=None)
    p_scalar.add_argument("--recursive", action="store_true")

    p_scalar.add_argument("--array-key", type=str, default="predictions")
    p_scalar.add_argument("--label-key", type=str, default="hdbscan_labels")
    p_scalar.add_argument("--include-noise", action="store_true")
    p_scalar.add_argument("--min-cluster-size", type=int, default=10)

    p_scalar.add_argument("--k1", type=int, default=10)
    p_scalar.add_argument("--k2", type=int, default=20)
    p_scalar.add_argument("--k-agg", type=str, default="mean", choices=["mean", "median"])
    p_scalar.add_argument("--point-agg", type=str, default="median", choices=["mean", "median"])

    p_scalar.add_argument("--do-sweeps", action="store_true")
    p_scalar.add_argument("--sweep-show", action="store_true")
    p_scalar.add_argument("--sweep-k-min", type=int, default=0)
    p_scalar.add_argument("--sweep-k-max", type=int, default=100)
    p_scalar.add_argument("--sweep-k-step", type=int, default=1)
    p_scalar.add_argument("--sweep-tick-every", type=int, default=5)
    p_scalar.add_argument("--sweep-min-clusters-per-k", type=int, default=3)

    # ---- prepost ----
    p_pp = sub.add_parser("prepost", help="Compute pre vs post LB per cluster using metadata + variance tiers.")
    p_pp.add_argument("--root-dir", type=str, required=True)
    p_pp.add_argument("--metadata-xlsx", type=str, required=True)
    p_pp.add_argument("--stats-csv", type=str, required=True)
    p_pp.add_argument("--out-dir", type=str, default=None)
    p_pp.add_argument("--recursive", action="store_true")

    p_pp.add_argument("--array-key", type=str, default="predictions")
    p_pp.add_argument("--cluster-key", type=str, default="hdbscan_labels")
    p_pp.add_argument("--syllable-key", type=str, default="ground_truth_labels")
    p_pp.add_argument("--file-key", type=str, default="file")
    p_pp.add_argument("--date-key", type=str, default=None)

    p_pp.add_argument("--include-noise", action="store_true")
    p_pp.add_argument("--min-points-per-period", type=int, default=50)
    p_pp.add_argument("--no-vocalization-only", action="store_true")
    p_pp.add_argument("--vocalization-key", type=str, default="vocalization")

    p_pp.add_argument("--k1", type=int, default=10)
    p_pp.add_argument("--k2", type=int, default=20)
    p_pp.add_argument("--k-agg", type=str, default="mean", choices=["mean", "median"])
    p_pp.add_argument("--point-agg", type=str, default="median", choices=["mean", "median"])

    p_pp.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p_pp.add_argument("--max-points-per-cluster-period", type=int, default=None)

    p_pp.add_argument("--variance-high-quantile", type=float, default=0.70)
    p_pp.add_argument("--stats-group-value", type=str, default="Post")
    p_pp.add_argument("--stats-variance-col", type=str, default="Pre_Variance_ms2")
    p_pp.add_argument("--metadata-sheet", type=str, default="metadata_with_hit_type")

    args = p.parse_args()

    if args.cmd == "scalar":
        master_csv = run_root_directory_lb_mle(
            root_dir=args.root_dir,
            out_dir=args.out_dir,
            recursive=args.recursive,
            array_key=args.array_key,
            label_key=args.label_key,
            include_noise=args.include_noise,
            min_cluster_size=args.min_cluster_size,
            k1=args.k1,
            k2=args.k2,
            k_agg=args.k_agg,
            point_agg=args.point_agg,
            do_sweeps=args.do_sweeps,
            sweep_show=args.sweep_show,
            sweep_k_min=args.sweep_k_min,
            sweep_k_max=args.sweep_k_max,
            sweep_k_step=args.sweep_k_step,
            sweep_tick_every=args.sweep_tick_every,
            sweep_min_clusters_per_k=args.sweep_min_clusters_per_k,
        )
        print(master_csv)

    elif args.cmd == "prepost":
        cfg = PrePostConfig(
            array_key=args.array_key,
            cluster_key=args.cluster_key,
            syllable_key=args.syllable_key,
            file_key=args.file_key,
            date_key=args.date_key,
            include_noise=args.include_noise,
            min_points_per_period=args.min_points_per_period,
            use_vocalization_only=(not args.no_vocalization_only),
            vocalization_key=args.vocalization_key,
            k1=args.k1,
            k2=args.k2,
            k_agg=args.k_agg,
            point_agg=args.point_agg,
            include_treatment_day_in_post=(not args.exclude_treatment_day_from_post),
            max_points_per_cluster_period=args.max_points_per_cluster_period,
        )

        paths = run_root_directory_pre_post_analysis(
            root_dir=args.root_dir,
            metadata_xlsx=args.metadata_xlsx,
            stats_csv=args.stats_csv,
            out_dir=args.out_dir,
            recursive=args.recursive,
            metadata_sheet=args.metadata_sheet,
            config=cfg,
            variance_high_quantile=args.variance_high_quantile,
            stats_group_value=args.stats_group_value,
            stats_variance_col=args.stats_variance_col,
        )
        print("\nOutputs:")
        for k, v in paths.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


"""
===============================================================================
Spyder Console Sample Usage (copy/paste)
===============================================================================
"""

"""
-------------------------
Setup + import/reload
-------------------------
"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import Levina_Bickel_syllable_clusters as lb
importlib.reload(lb)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")

"""
-------------------
Scalar-only (fast)
-------------------
Computes ONE LB intrinsic-dimension estimate per cluster per NPZ,
and writes:
  - per-NPZ CSVs under: <root>/lb_cluster_mle_outputs/<npz_stem>/
  - one master CSV under: <root>/lb_cluster_mle_outputs/
"""
master_csv = lb.run_root_directory_lb_mle(
    root_dir=root_dir,
    out_dir=root_dir / "lb_cluster_mle_outputs",
    recursive=True,
    array_key="predictions",
    label_key="hdbscan_labels",
    include_noise=False,
    min_cluster_size=10,
    k1=10, k2=20,
    k_agg="mean",
    point_agg="median",
    do_sweeps=False,
)
print(master_csv)

"""
-----------------------------------------
Scalar + sweeps (heavier; lots of plots)
-----------------------------------------
Also runs per-cluster sweeps (k ~ 2..min(100, n-1)) and saves:
  - per-cluster sweep PNG + CSV
  - per-NPZ aggregate curve across clusters (PNG + CSV)
All saved under:
  <root>/lb_cluster_mle_outputs/<npz_stem>/sweeps/
"""
master_csv = lb.run_root_directory_lb_mle(
    root_dir=root_dir,
    out_dir=root_dir / "lb_cluster_mle_outputs",
    recursive=True,
    array_key="predictions",
    label_key="hdbscan_labels",
    include_noise=False,
    min_cluster_size=10,
    k1=10, k2=20,
    k_agg="mean",
    point_agg="median",
    do_sweeps=True,
    sweep_show=False,
    sweep_k_min=0,
    sweep_k_max=100,
    sweep_k_step=1,
    sweep_tick_every=5,
    sweep_min_clusters_per_k=3,
)
print(master_csv)

"""
-----------------------------------------------------------
Pre vs Post treatment dimensionality + hit-type + variance
-----------------------------------------------------------
Writes:
  - results CSV
  - delta plot (post-pre) by hit type and variance tier
  - pre vs post scatter by hit type
Default outputs under:
  <root>/lb_pre_post_dimensionality/
"""
metadata_xlsx = root_dir / "Area_X_lesion_metadata_with_hit_types.xlsx"
stats_csv = root_dir / "usage_balanced_phrase_duration_stats.csv"

cfg = lb.PrePostConfig(
    array_key="predictions",
    cluster_key="hdbscan_labels",
    syllable_key="ground_truth_labels",  # used to map cluster -> syllable for variance tier
    file_key="file",                     # parse dates if no date_key exists
    date_key=None,                       # set if your NPZ already has a date array key
    include_noise=False,
    use_vocalization_only=True,
    min_points_per_period=50,
    include_treatment_day_in_post=True,
    k1=10, k2=20,
    k_agg="mean",
    point_agg="median",
    max_points_per_cluster_period=5000,  # optional speed cap
)

paths = lb.run_root_directory_pre_post_analysis(
    root_dir=root_dir,
    metadata_xlsx=metadata_xlsx,
    stats_csv=stats_csv,
    out_dir=root_dir / "lb_pre_post_dimensionality",
    recursive=True,
    config=cfg,
    variance_high_quantile=0.70,         # top 30% high variance
    stats_group_value="Post",
    stats_variance_col="Pre_Variance_ms2",
    metadata_sheet="metadata_with_hit_type",
)
print(paths)