#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_parameter_sweep.py

(Yes, "sweel" matches the filename you requested. Feel free to rename to "..._sweep.py".)

Purpose
-------
Run Levina–Bickel (LB) intrinsic-dimension *kNN parameter sweeps* for HDBSCAN clusters
inside .npz files.

What this script does
---------------------
For each NPZ:
  • For each HDBSCAN cluster (optionally excluding noise):
      - Compute LB m_hat(k) for k in [k_min..k_max] (LB requires k>=2 and k<=n-1)
      - Save:
          - per-cluster sweep CSV: k, m_hat, frac_valid
          - per-cluster sweep PNG
  • Aggregate across clusters:
      - mean / std / median of m_hat(k) across clusters (for each k)
      - Save aggregate curve CSV + PNG

Inputs expected in each NPZ
---------------------------
Required keys:
  - array_key (default "predictions"): (N, D)
  - label_key (default "hdbscan_labels"): (N,)

Dependencies
------------
numpy, matplotlib, scikit-learn

Usage (CLI)
-----------
python Levina_Bickel_parameter_sweel.py --root-dir /path/to/npzs --recursive \
  --array-key predictions --label-key hdbscan_labels --out-dir /path/to/out \
  --k-min 2 --k-max 30 --min-cluster-size 10

Spyder usage is included at bottom (triple-quoted block).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Helpers
# ============================================================
def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _make_nn(n_neighbors: int, n_jobs: int) -> NearestNeighbors:
    """sklearn versions vary in whether NearestNeighbors accepts n_jobs."""
    try:
        return NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=n_jobs)
    except TypeError:
        return NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")


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


def _write_rows_csv(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        csv_path.write_text("")
        return

    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Levina–Bickel sweep core
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

    point_agg: 'mean' or 'median' over points for each k.

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

    def _agg(a: np.ndarray) -> float:
        if point_agg.lower() == "mean":
            return float(np.nanmean(a))
        if point_agg.lower() == "median":
            return float(np.nanmedian(a))
        raise ValueError("point_agg must be 'mean' or 'median'")

    m_hat_list: List[float] = []
    frac_valid_list: List[float] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        for k in k_arr:
            # sum_{j=1}^{k-1} log(T_k/T_j) = (k-1)*log(T_k) - sum_{j=1}^{k-1} log(T_j)
            logT_k = logT[:, k - 1]
            sum_logT_j = cumsum_logT[:, k - 2]
            log_sums = (k - 1) * logT_k - sum_logT_j

            valid = np.isfinite(log_sums) & (log_sums > 0)
            m_k = np.full(n, np.nan, dtype=float)
            m_k[valid] = (k - 1) / log_sums[valid]

            m_hat_list.append(_agg(m_k))
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
    title_prefix: str = "Levina–Bickel k sweep",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_arr, m_hat_arr, marker="o")
    ax.set_xlabel("k (KNN)\n(k as % of cluster size)")
    ax.set_ylabel(f"Estimated intrinsic dimension (LB, point_agg={point_agg})")
    ax.set_title(f"{title_prefix} (cluster {cluster_id}, n={n_cluster})")
    ax.grid(True, alpha=0.3)

    k_ticks = _choose_tick_positions(k_arr, tick_every=tick_every)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels(_format_k_ticks_with_percent(k_ticks, n_cluster=n_cluster), fontsize=8)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
    show: bool = False,
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved aggregate curve plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# NPZ-level sweep
# ============================================================
@dataclass
class SweepConfig:
    array_key: str = "predictions"
    label_key: str = "hdbscan_labels"
    include_noise: bool = False
    min_cluster_size: int = 10
    point_agg: str = "median"
    eps: float = 1e-12
    n_jobs: int = 1

    # sweep params
    k_min: int = 0
    k_max: int = 100
    k_step: int = 1
    tick_every: int = 5
    show: bool = False
    min_clusters_per_k: int = 3


def run_sweeps_for_npz(
    npz_path: str | Path,
    out_dir: str | Path,
    cfg: SweepConfig,
) -> Dict[str, str]:
    """
    For ONE NPZ:
      - run per-cluster sweeps (save PNG+CSV per cluster)
      - compute and plot aggregate curve across clusters

    Returns paths to summary CSV + aggregate CSV/PNG.
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = npz_path.stem

    data = np.load(npz_path, allow_pickle=True)
    if cfg.array_key not in data or cfg.label_key not in data:
        raise KeyError(
            f"Missing keys in {npz_path.name}: need {cfg.array_key!r} and {cfg.label_key!r}. "
            f"Present: {list(data.keys())}"
        )

    X = _safe_2d(data[cfg.array_key])
    labels = np.asarray(data[cfg.label_key])
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X has n={X.shape[0]} rows but labels has n={labels.shape[0]} entries")

    cluster_ids = np.unique(labels)
    if not cfg.include_noise:
        cluster_ids = cluster_ids[cluster_ids != -1]

    # sort clusters by size desc
    cluster_sizes = [(int(cid), int(np.sum(labels == cid))) for cid in cluster_ids]
    cluster_sizes.sort(key=lambda t: t[1], reverse=True)

    cluster_sweeps: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    summary_rows: List[Dict[str, Any]] = []

    cluster_dir = out_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for cid, n_c in cluster_sizes:
        if n_c < max(3, int(cfg.min_cluster_size)):
            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "k_min_used": "",
                "k_max_used": "",
                "point_agg": cfg.point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"skipped: n_cluster<{cfg.min_cluster_size}",
            })
            continue

        Xc = X[labels == cid]
        try:
            k_arr, m_hat_arr, frac_valid_arr = levina_bickel_knn_sweep(
                Xc,
                k_min_requested=cfg.k_min,
                k_max_requested=cfg.k_max,
                k_step=cfg.k_step,
                point_agg=cfg.point_agg,
                eps=cfg.eps,
                n_jobs=cfg.n_jobs,
            )

            # save per-cluster CSV
            csv_path = cluster_dir / f"{base}_cluster{cid}_{cfg.array_key}_sweep_{ts}.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("k,m_hat,frac_valid\n")
                for i in range(len(k_arr)):
                    f.write(f"{int(k_arr[i])},{m_hat_arr[i]},{frac_valid_arr[i]}\n")

            # save per-cluster plot
            plot_path = cluster_dir / f"{base}_cluster{cid}_{cfg.array_key}_sweep_{ts}.png"
            plot_cluster_sweep(
                k_arr=k_arr,
                m_hat_arr=m_hat_arr,
                n_cluster=n_c,
                cluster_id=cid,
                point_agg=cfg.point_agg,
                tick_every=cfg.tick_every,
                title_prefix=f"Levina–Bickel k sweep ({base})",
                save_path=str(plot_path),
                show=cfg.show,
            )

            cluster_sweeps[cid] = (k_arr, m_hat_arr, frac_valid_arr, n_c)

            summary_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": cid,
                "n_cluster": n_c,
                "k_min_used": int(k_arr.min()) if len(k_arr) else "",
                "k_max_used": int(k_arr.max()) if len(k_arr) else "",
                "point_agg": cfg.point_agg,
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
                "point_agg": cfg.point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"{type(e).__name__}: {e}",
            })

    sweep_summary_csv = out_dir / f"{base}_{cfg.array_key}_sweep_summary_{ts}.csv"
    _write_rows_csv(sweep_summary_csv, summary_rows)

    # Aggregate curve across clusters
    k_min_global = max(2, int(cfg.k_min))
    k_max_global = int(cfg.k_max)
    k_global = np.arange(k_min_global, k_max_global + 1, int(cfg.k_step), dtype=int)

    mean_list: List[float] = []
    std_list: List[float] = []
    median_list: List[float] = []
    n_clusters_list: List[int] = []

    # For each k, collect the m_hat(k) from each cluster that has that k available.
    for k in k_global:
        vals: List[float] = []
        for _, (k_arr, m_hat_arr, _, _) in cluster_sweeps.items():
            if k < int(k_arr.min()) or k > int(k_arr.max()):
                continue
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

    agg_csv = out_dir / f"{base}_{cfg.array_key}_aggregate_curve_{ts}.csv"
    with open(agg_csv, "w", newline="") as f:
        f.write("k,mean,std,median,n_clusters\n")
        for i in range(len(k_global)):
            f.write(f"{int(k_global[i])},{mean_arr[i]},{std_arr[i]},{median_arr[i]},{int(n_clusters_arr[i])}\n")

    agg_plot = out_dir / f"{base}_{cfg.array_key}_aggregate_curve_{ts}.png"
    plot_aggregate_curve_across_clusters(
        k_global=k_global,
        mean_arr=mean_arr,
        std_arr=std_arr,
        n_clusters_arr=n_clusters_arr,
        title=f"Aggregate Levina–Bickel curve across clusters ({base}, array={cfg.array_key})",
        tick_every=cfg.tick_every,
        min_clusters_per_k=cfg.min_clusters_per_k,
        save_path=str(agg_plot),
        show=cfg.show,
    )

    return {
        "sweep_summary_csv": str(sweep_summary_csv),
        "aggregate_csv": str(agg_csv),
        "aggregate_plot": str(agg_plot),
        "out_dir": str(out_dir),
    }


def run_root_directory_sweeps(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    cfg: Optional[SweepConfig] = None,
) -> str:
    """
    Run sweeps for every NPZ under root_dir.

    Returns: path to a master summary CSV that concatenates per-NPZ sweep summaries.
    """
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / "lb_cluster_sweeps"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = SweepConfig()

    npz_paths = sorted(root_dir.rglob("*.npz") if recursive else root_dir.glob("*.npz"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    master_rows: List[Dict[str, Any]] = []

    for npz_path in npz_paths:
        print(f"[LB sweep] processing: {npz_path}")
        per_dir = out_dir / npz_path.stem / "sweeps"
        per_dir.mkdir(parents=True, exist_ok=True)
        try:
            paths = run_sweeps_for_npz(npz_path=npz_path, out_dir=per_dir, cfg=cfg)
            # Load that per-NPZ summary back into master
            df = pd.read_csv(paths["sweep_summary_csv"])
            df["npz_stem"] = npz_path.stem
            master_rows.extend(df.to_dict(orient="records"))
        except Exception as e:
            master_rows.append({
                "npz_path": str(npz_path),
                "cluster_id": "",
                "n_cluster": "",
                "k_min_used": "",
                "k_max_used": "",
                "point_agg": cfg.point_agg,
                "plot_path": "",
                "csv_path": "",
                "error": f"{type(e).__name__}: {e}",
                "npz_stem": npz_path.stem,
            })

    master_csv = out_dir / f"LB_master_sweep_summaries_{ts}.csv"
    _write_rows_csv(master_csv, master_rows)
    print(f"\n[LB sweep] MASTER summary saved: {master_csv}")
    return str(master_csv)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Levina–Bickel kNN parameter sweeps for HDBSCAN clusters in NPZ files.")
    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--array-key", type=str, default="predictions")
    p.add_argument("--label-key", type=str, default="hdbscan_labels")
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--min-cluster-size", type=int, default=10)

    p.add_argument("--point-agg", type=str, default="median", choices=["mean", "median"])
    p.add_argument("--n-jobs", type=int, default=1)

    p.add_argument("--k-min", type=int, default=0)
    p.add_argument("--k-max", type=int, default=100)
    p.add_argument("--k-step", type=int, default=1)
    p.add_argument("--tick-every", type=int, default=5)
    p.add_argument("--show", action="store_true")
    p.add_argument("--min-clusters-per-k", type=int, default=3)

    args = p.parse_args()

    cfg = SweepConfig(
        array_key=args.array_key,
        label_key=args.label_key,
        include_noise=args.include_noise,
        min_cluster_size=args.min_cluster_size,
        point_agg=args.point_agg,
        n_jobs=args.n_jobs,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        tick_every=args.tick_every,
        show=args.show,
        min_clusters_per_k=args.min_clusters_per_k,
    )

    master_csv = run_root_directory_sweeps(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        recursive=args.recursive,
        cfg=cfg,
    )
    print(master_csv)


if __name__ == "__main__":
    main()




# =============================================================================
# Spyder Console Sample Usage (copy/paste)
# =============================================================================
#
# from pathlib import Path
# import sys, importlib
#
# code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
# if str(code_dir) not in sys.path:
#     sys.path.insert(0, str(code_dir))
#
# import Levina_Bickel_parameter_sweep as lbs
# importlib.reload(lbs)
#
# root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
#
# cfg = lbs.SweepConfig(
#     array_key="predictions",
#     label_key="hdbscan_labels",
#     include_noise=False,
#     min_cluster_size=10,
#     point_agg="median",
#     n_jobs=1,
#     k_min=0,
#     k_max=30,
#     k_step=1,
#     tick_every=5,
#     show=False,
#     min_clusters_per_k=3,
# )
#
# master_csv = lbs.run_root_directory_sweeps(
#     root_dir=root_dir,
#     out_dir=root_dir / "lb_cluster_sweeps",
#     recursive=True,
#     cfg=cfg,
# )
# print(master_csv)
