#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Levina_Bickel_parameter_sweep.py

KNN parameter sweep for Levina–Bickel (LB) intrinsic dimension, per HDBSCAN cluster.

What this script does
---------------------
For each NPZ under --root-dir:
  - For each HDBSCAN cluster (optionally excluding noise):
      - Compute LB intrinsic dimension estimates across a k range (k sweep).
      - Save per-cluster CSV + PNG.
  - Also compute an "aggregate curve" across clusters for that NPZ (mean ± std).

Speed: computer_power profiles
------------------------------
This script can tune parallelism with --computer-power:
  - laptop: sequential across NPZ files; allow sklearn threading within the process.
  - studio: parallelize across NPZ files using a ProcessPool; keep NN single-threaded per process.
  - auto: chooses based on CPU count.

Because sweeps can be *very* expensive for large clusters, you can optionally cap
cluster size with --max-points-per-cluster (subsample without replacement) to
get a fast, representative sweep.

Dependencies
------------
numpy, pandas, matplotlib, scikit-learn

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Safer for headless / multiprocessing runs
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# Compute profile (computer_power)
# ============================================================
@dataclass(frozen=True)
class ComputeProfile:
    name: str
    max_workers: int
    nn_jobs: int


def _resolve_profile(computer_power: str = "auto", workers: Optional[int] = None) -> ComputeProfile:
    cpu = int(os.cpu_count() or 8)
    cp = (computer_power or "auto").strip().lower()

    if cp == "auto":
        cp = "laptop" if cpu <= 8 else "studio"

    if cp not in {"laptop", "studio"}:
        raise ValueError("computer_power must be one of: auto, laptop, studio")

    if cp == "laptop":
        max_workers = 1
        nn_jobs = -1
    else:
        max_workers = min(max(2, cpu - 1), 6)
        nn_jobs = 1

    if workers is not None:
        max_workers = max(1, int(workers))
        if max_workers > 1:
            nn_jobs = 1

    return ComputeProfile(name=cp, max_workers=max_workers, nn_jobs=nn_jobs)


# ============================================================
# Utilities
# ============================================================
def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _nan_agg(a: np.ndarray, method: str, axis=None):
    method = str(method).lower()
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
        csv_path.write_text("")
        return

    headers = list(rows[0].keys())
    import csv
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
        pct = 100.0 * float(k) / float(max(1, n_cluster))
        labels.append(f"{int(k)}\n{int(round(pct))}%")
    return labels


# ============================================================
# Levina–Bickel MLE sweep core
# ============================================================
def _compute_knn_logs(
    X: np.ndarray,
    k_max: int,
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _safe_2d(X)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points; got n={n}")
    if k_max > n - 1:
        raise ValueError(f"k_max={k_max} but n={n}; need k_max <= n-1")

    nn = _make_nn(n_neighbors=k_max + 1, n_jobs=int(n_jobs))
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    T = distances[:, 1:]
    logT = np.log(np.maximum(T, eps))
    cumsum_logT = np.cumsum(logT, axis=1)
    return T, logT, cumsum_logT


def levina_bickel_knn_sweep(
    X: np.ndarray,
    k_min_requested: int = 2,
    k_max_requested: int = 100,
    k_step: int = 1,
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep LB m_hat(k) for ONE dataset X.

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
# Plotting
# ============================================================
def plot_cluster_sweep(
    k_arr: np.ndarray,
    m_hat_arr: np.ndarray,
    n_cluster: int,
    cluster_id: int,
    point_agg: str,
    tick_every: int = 5,
    save_path: Optional[str] = None,
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
) -> None:
    mask = np.isfinite(mean_arr) & (n_clusters_arr >= int(min_clusters_per_k))
    if not np.any(mask):
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
    plt.close(fig)


# ============================================================
# One-NPZ sweep runner
# ============================================================
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
    k_min: int = 2,
    k_max: int = 100,
    k_step: int = 1,
    tick_every: int = 5,
    min_clusters_per_k: int = 3,
    max_points_per_cluster: Optional[int] = None,
    subsample_seed: int = 0,
) -> Dict[str, str]:
    """For ONE NPZ:
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
        raise KeyError(
            f"Missing keys in {npz_path.name}: need {array_key!r} and {label_key!r}. Present: {list(data.keys())}"
        )

    X = _safe_2d(data[array_key])
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    labels = np.asarray(data[label_key])
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X has n={X.shape[0]} rows but labels has n={labels.shape[0]} entries")

    cluster_ids = np.unique(labels)
    if not include_noise:
        cluster_ids = cluster_ids[cluster_ids != -1]

    # Sort clusters by size desc
    cluster_sizes = [(int(cid), int(np.sum(labels == cid))) for cid in cluster_ids]
    cluster_sizes.sort(key=lambda t: t[1], reverse=True)

    rng = np.random.RandomState(int(subsample_seed))

    cluster_dir = out_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    cluster_sweeps: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for cid, n_c in cluster_sizes:
        if n_c < max(3, int(min_cluster_size)):
            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "n_used": "",
                "k_min_used": "",
                "k_max_used": "",
                "point_agg": point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"skipped: n_cluster<{min_cluster_size}",
            })
            continue

        idx = np.where(labels == cid)[0]
        n_used = n_c
        if max_points_per_cluster is not None and n_c > int(max_points_per_cluster):
            idx = rng.choice(idx, size=int(max_points_per_cluster), replace=False)
            n_used = int(idx.size)

        Xc = X[idx]

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
            )

            cluster_sweeps[cid] = (k_arr, m_hat_arr, frac_valid_arr, n_used)

            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "n_used": n_used,
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
                "n_used": n_used,
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
    )

    return {
        "sweep_summary_csv": str(sweep_summary_csv),
        "aggregate_csv": str(agg_csv),
        "aggregate_plot": str(agg_plot),
        "out_dir": str(out_dir),
    }


# ============================================================
# Parallel NPZ processing wrapper
# ============================================================
def _sweep_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    npz_path = payload["npz_path"]
    out_dir = payload["out_dir"]

    try:
        res = run_sweeps_for_npz(
            npz_path=npz_path,
            out_dir=out_dir,
            array_key=payload["array_key"],
            label_key=payload["label_key"],
            include_noise=payload["include_noise"],
            min_cluster_size=payload["min_cluster_size"],
            point_agg=payload["point_agg"],
            eps=payload["eps"],
            n_jobs=payload["n_jobs"],
            k_min=payload["k_min"],
            k_max=payload["k_max"],
            k_step=payload["k_step"],
            tick_every=payload["tick_every"],
            min_clusters_per_k=payload["min_clusters_per_k"],
            max_points_per_cluster=payload.get("max_points_per_cluster"),
            subsample_seed=payload.get("subsample_seed", 0),
        )
        return {
            "npz_path": str(npz_path),
            **res,
            "error": "",
        }
    except Exception as e:
        return {
            "npz_path": str(npz_path),
            "sweep_summary_csv": "",
            "aggregate_csv": "",
            "aggregate_plot": "",
            "out_dir": str(out_dir),
            "error": f"{type(e).__name__}: {e}",
        }


def run_root_directory_sweeps(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    array_key: str = "predictions",
    label_key: str = "hdbscan_labels",
    include_noise: bool = False,
    min_cluster_size: int = 10,
    point_agg: str = "median",
    eps: float = 1e-12,
    k_min: int = 2,
    k_max: int = 100,
    k_step: int = 1,
    tick_every: int = 5,
    min_clusters_per_k: int = 3,
    max_points_per_cluster: Optional[int] = None,
    subsample_seed: int = 0,
    computer_power: str = "auto",
    workers: Optional[int] = None,
    n_jobs: int = 0,
) -> str:
    """Run sweeps for every NPZ under root_dir, saving a master summary CSV."""
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / "lb_knn_sweep_outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = _resolve_profile(computer_power=computer_power, workers=workers)

    # If user didn't set n_jobs, use profile default.
    nn_jobs = profile.nn_jobs if int(n_jobs) == 0 else int(n_jobs)
    # If using multiple processes, force single-threaded NN per process.
    if profile.max_workers > 1:
        nn_jobs = 1

    npz_paths = sorted(root_dir.rglob("*.npz") if recursive else root_dir.glob("*.npz"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"\n[LB sweep] computer_power={profile.name} | max_workers={profile.max_workers} | nn_jobs={nn_jobs} | "
        f"max_points_per_cluster={max_points_per_cluster}\n"
    )

    payloads: List[Dict[str, Any]] = []
    for npz_path in npz_paths:
        per_dir = out_dir / npz_path.stem / "sweeps"
        per_dir.mkdir(parents=True, exist_ok=True)
        payloads.append({
            "npz_path": str(npz_path),
            "out_dir": str(per_dir),
            "array_key": array_key,
            "label_key": label_key,
            "include_noise": include_noise,
            "min_cluster_size": int(min_cluster_size),
            "point_agg": point_agg,
            "eps": float(eps),
            "n_jobs": int(nn_jobs),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "k_step": int(k_step),
            "tick_every": int(tick_every),
            "min_clusters_per_k": int(min_clusters_per_k),
            "max_points_per_cluster": (None if max_points_per_cluster is None else int(max_points_per_cluster)),
            "subsample_seed": int(subsample_seed),
        })

    master_rows: List[Dict[str, Any]] = []

    if profile.max_workers <= 1 or len(payloads) <= 1:
        for p in payloads:
            print(f"[sweep] {Path(p['npz_path']).name}")
            master_rows.append(_sweep_worker(p))
    else:
        with ProcessPoolExecutor(max_workers=profile.max_workers) as ex:
            futs = [ex.submit(_sweep_worker, p) for p in payloads]
            for fut in as_completed(futs):
                master_rows.append(fut.result())

    master_csv = out_dir / f"LB_sweep_master_{ts}.csv"
    _write_rows_csv(master_csv, master_rows)
    print(f"\nSaved MASTER CSV: {master_csv}")
    return str(master_csv)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Levina–Bickel kNN parameter sweep per HDBSCAN cluster.")

    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--array-key", type=str, default="predictions")
    p.add_argument("--label-key", type=str, default="hdbscan_labels")
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--min-cluster-size", type=int, default=10)

    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=100)
    p.add_argument("--k-step", type=int, default=1)
    p.add_argument("--point-agg", type=str, default="median", choices=["mean", "median"])
    p.add_argument("--tick-every", type=int, default=5)
    p.add_argument("--min-clusters-per-k", type=int, default=3)

    p.add_argument(
        "--max-points-per-cluster",
        type=int,
        default=None,
        help="Optional subsampling cap per cluster (e.g. 20000) for faster sweeps",
    )
    p.add_argument("--subsample-seed", type=int, default=0)

    # Speed knobs
    p.add_argument("--computer-power", type=str, default="auto", choices=["auto", "laptop", "studio"])
    p.add_argument("--workers", type=int, default=None, help="Override number of parallel worker processes")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="NearestNeighbors n_jobs (0=auto from --computer-power; ignored when workers>1)",
    )

    args = p.parse_args()

    master_csv = run_root_directory_sweeps(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        recursive=args.recursive,
        array_key=args.array_key,
        label_key=args.label_key,
        include_noise=args.include_noise,
        min_cluster_size=args.min_cluster_size,
        point_agg=args.point_agg,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        tick_every=args.tick_every,
        min_clusters_per_k=args.min_clusters_per_k,
        max_points_per_cluster=args.max_points_per_cluster,
        subsample_seed=args.subsample_seed,
        computer_power=args.computer_power,
        workers=args.workers,
        n_jobs=args.n_jobs,
    )

    print(master_csv)


if __name__ == "__main__":
    main()
