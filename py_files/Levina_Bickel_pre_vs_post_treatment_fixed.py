#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_pre_vs_post_treatment.py

Compute Levina–Bickel intrinsic dimensionality (LB-MLE) per cluster PRE vs POST treatment,
using large NPZ files that contain pointwise embeddings/activations + per-point cluster labels.

Designed for TweetyBERT-style NPZs that contain:
    - predictions (N, D)           (default array_key)
    - hdbscan_labels (N,)          (default cluster_key)
    - ground_truth_labels (N,)     (optional; used only to report a "mode syllable" per cluster)
    - vocalization (N,)            (optional; used to filter to vocalization-only by default)
    - file_indices (N,) and file_map (dict)  (used to map each point to a source file/time)

Date parsing
------------
For TweetyBERT segment filenames like:
  USA5288_45382.42553504_3_31_11_49_13_segment_0.npz

We first parse the Excel-serial day number (e.g., 45382.4255...) which maps cleanly to
calendar datetime (origin 1899-12-30). If that fails, we fall back to parsing the
explicit month/day/hour/min/sec tokens.

Outputs
-------
Inside <root-dir>/lb_pre_post_dimensionality_k{K}/ (or --out-dir):
    - lb_pre_post_cluster_dimensionality_k{K}_<timestamp>.csv
    - pre_vs_post_scatter_by_hit_k{K}_<timestamp>.png
    - delta_dim_by_hit_type_k{K}_<timestamp>.png

Example (single bird folder):
    python Levina_Bickel_pre_vs_post_treatment.py \
      --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288" \
      --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
      --metadata-sheet "animal_hit_type_summary" \
      --computer-power pro \
      --workers 1 \
      --k 15

Notes on performance
--------------------
- `--workers` parallelizes ACROSS NPZ files (processes).
- `--n-jobs` parallelizes WITHIN an NPZ file for nearest-neighbor search (threads).
- If you pass 0 for these, defaults are chosen based on `--computer-power` (laptop vs pro).

Author: (generated/maintained with ChatGPT)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import argparse
import time
import re
import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Optional sklearn (required for LB as implemented here) ---
try:
    from sklearn.neighbors import NearestNeighbors
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# ---------------------------
# Config / defaults
# ---------------------------

@dataclass
class PrePostConfig:
    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    syllable_key: str = "ground_truth_labels"
    file_key: str = "file_indices"
    date_key: str = "date"  # only used if present in NPZ; otherwise file_indices+file_map are used
    include_noise: bool = False
    vocalization_only: bool = True
    vocalization_key: str = "vocalization"
    k: int = 15
    n_jobs: int = 0  # threads for NearestNeighbors; 0 => auto by computer_power
    min_points_per_period: int = 50
    include_treatment_day_in_post: bool = True
    max_points_per_cluster_period: Optional[int] = None  # None => auto by computer_power
    point_agg: str = "mean"  # reserved; currently LB returns an aggregate per cluster-period
    verbose_dates: bool = False


# ---------------------------
# Utilities
# ---------------------------

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _safe_2d(X: Any) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _is_scalar_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _animal_id_from_npz_path(npz_path: Union[str, Path]) -> str:
    p = Path(npz_path)
    # Typical: <root>/<ANIMAL>.npz
    stem = p.stem
    # If stem includes extra tokens, take the first token that looks like USA#### or similar
    m = re.search(r"(USA\d+)", stem)
    return m.group(1) if m else stem

def _canonical_hit_group(hit_type: str) -> str:
    """
    Map arbitrary metadata "Lesion hit type" strings into the 3 groups you want.
    """
    s = (hit_type or "").strip().lower()
    if "sham" in s:
        return "sham saline injection"
    if "single" in s:
        return "Area X visible (single hit)"
    # combined group: visible medial+lateral OR large not visible (and similar)
    if ("medial" in s and "lateral" in s) or ("ml" in s and "visible" in s) or ("not visible" in s) or ("large" in s):
        return "Combined (visible ML + not visible)"
    # fallback: if it contains "visible" but not single, still put in combined
    if "visible" in s:
        return "Combined (visible ML + not visible)"
    return "Combined (visible ML + not visible)"

# ---------------------------
# CSV + plotting
# ---------------------------

def _write_rows_csv(csv_path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    csv_path = Path(csv_path)
    _ensure_dir(csv_path.parent)
    if not rows:
        # still write a header-only file
        pd.DataFrame().to_csv(csv_path, index=False)
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def _savefig(fig: plt.Figure, out_path: Union[str, Path], dpi: int = 200) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Date parsing (file_map -> filename -> datetime/date)
# ---------------------------

_EXCEL_SERIAL_PATTERN = re.compile(r"_(?P<serial>\d{4,6}(?:\.\d{1,18})?)_")
# e.g., USA5508_45570.28601193_10_5_7_56_41_segment_1.npz  (month/day/hour/min/sec after serial)
_MDHMS_PATTERN = re.compile(
    r"^(?P<animal>[A-Za-z0-9]+)_(?P<serial>\d{4,6}(?:\.\d+)?)_(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<h>\d{1,2})_(?P<min>\d{1,2})_(?P<s>\d{1,2})"
)

def _excel_serial_to_datetime(serial_days: float) -> datetime.datetime:
    """
    Excel serial day number -> datetime, using the common pandas convention:
    origin='1899-12-30', unit='D'
    """
    # Avoid pandas dependency here; do it with datetime/timedelta.
    origin = datetime.datetime(1899, 12, 30)
    return origin + datetime.timedelta(days=float(serial_days))

def _normalize_filename_obj(x: Any) -> str:
    """
    file_map values can be:
      - a string path/filename
      - a tuple/list where element 0 is filename/path
      - an object array
      - rarely, dict-like structures
    Return a best-effort filename string.
    """
    if x is None:
        return ""
    if isinstance(x, (str, Path)):
        return str(x)
    if isinstance(x, dict):
        # try common keys
        for k in ("path", "file", "filename", "name"):
            if k in x:
                return _normalize_filename_obj(x[k])
        # fallback: first value
        try:
            return _normalize_filename_obj(next(iter(x.values())))
        except Exception:
            return ""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        return _normalize_filename_obj(x[0])
    if isinstance(x, np.ndarray):
        if x.shape == ():
            try:
                return _normalize_filename_obj(x.item())
            except Exception:
                return str(x)
        if x.size >= 1:
            return _normalize_filename_obj(x.flat[0])
    return str(x)

def _parse_date_from_string(file_str: str, year_default: int) -> Optional[datetime.date]:
    """
    Returns a DATE (not datetime) for comparison to treatment_date.
    """
    if not file_str:
        return None
    base = Path(str(file_str)).name

    # 1) Excel serial day number
    m = _EXCEL_SERIAL_PATTERN.search(base)
    if m:
        try:
            serial = float(m.group("serial"))
            dt = _excel_serial_to_datetime(serial)
            return dt.date()
        except Exception:
            pass

    # 2) Explicit month/day/hour/min/sec tokens
    parts = base.split("_")
    # Optional year override if a 4-digit year token appears
    year = year_default
    for p in parts:
        if p.isdigit() and len(p) == 4:
            y = int(p)
            if 1990 <= y <= 2100:
                year = y
                break

    mm = _MDHMS_PATTERN.match(base)
    if mm:
        try:
            month = int(mm.group("m"))
            day = int(mm.group("d"))
            # hour/min/sec are available, but for PRE/POST date comparisons, date() is sufficient
            return datetime.date(year, month, day)
        except Exception:
            return None

    # 3) very lenient fallback: try to find _M_D_ near the start after serial
    try:
        # base: animal_serial_M_D_H_M_S_...
        # parts: [animal, serial, M, D, H, MIN, S, ...]
        if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
            month = int(parts[2])
            day = int(parts[3])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return datetime.date(year, month, day)
    except Exception:
        pass

    return None

def infer_point_dates_from_npz(
    data: np.lib.npyio.NpzFile,
    *,
    N: int,
    file_key: str,
    date_key: str,
    year_default: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[int, str], float]:
    """
    Return:
      dates: (N,) np.datetime64[D] (NaT where unknown)
      file_map: {file_index -> filename}
      parsed_fraction: fraction of unique file indices successfully parsed
    """
    # If NPZ already has per-point dates, use them.
    if date_key in data.files:
        arr = data[date_key]
        try:
            # accept datetime64, strings, ints
            if np.issubdtype(arr.dtype, np.datetime64):
                dates = arr.astype("datetime64[D]")
                return dates, {}, 1.0
            # strings like YYYY-MM-DD
            if arr.dtype.kind in ("U", "S", "O"):
                dt = pd.to_datetime(arr, errors="coerce")
                dates = dt.to_numpy(dtype="datetime64[D]")
                return dates, {}, float(np.mean(~np.isnat(dates)))
            # numeric (maybe excel serial per point)
            if np.issubdtype(arr.dtype, np.number):
                dt = pd.to_datetime(arr.astype(float), unit="D", origin="1899-12-30", errors="coerce")
                dates = dt.to_numpy(dtype="datetime64[D]")
                return dates, {}, float(np.mean(~np.isnat(dates)))
        except Exception:
            # fall through to file_indices+file_map
            pass

    # file_indices + file_map route
    if file_key not in data.files or "file_map" not in data.files:
        dates = np.full((N,), np.datetime64("NaT"), dtype="datetime64[D]")
        return dates, {}, 0.0

    file_indices = np.asarray(data[file_key]).astype(int)
    if file_indices.shape[0] != N:
        raise ValueError(f"{file_key} has shape {file_indices.shape}, expected first dim {N}")

    fm_arr = data["file_map"]
    fm = fm_arr.item() if getattr(fm_arr, "shape", None) == () else fm_arr

    mapping: Dict[int, str] = {}
    if isinstance(fm, dict):
        for k, v in fm.items():
            try:
                idx = int(k)
            except Exception:
                continue
            mapping[idx] = Path(_normalize_filename_obj(v)).name
    elif isinstance(fm, (list, tuple, np.ndarray)):
        for idx, v in enumerate(list(fm)):
            mapping[int(idx)] = Path(_normalize_filename_obj(v)).name
    else:
        # unknown structure
        dates = np.full((N,), np.datetime64("NaT"), dtype="datetime64[D]")
        return dates, {}, 0.0

    uniq = np.unique(file_indices)
    idx_to_date = {}
    parsed = 0
    for idx in uniq:
        fname = mapping.get(int(idx), "")
        d = _parse_date_from_string(fname, year_default=year_default)
        if d is None:
            idx_to_date[int(idx)] = np.datetime64("NaT")
        else:
            idx_to_date[int(idx)] = np.datetime64(d)
            parsed += 1

    parsed_fraction = parsed / max(1, uniq.size)

    if verbose:
        print(f"[dates] file_indices+file_map: parsed {parsed}/{uniq.size} indices ({100*parsed_fraction:.1f}%)")
        if parsed < uniq.size:
            # show a few failures
            failed = [mapping.get(int(i), "") for i in uniq if np.isnat(idx_to_date[int(i)])]
            if failed:
                print("  examples that FAILED date parsing (first 10):")
                for ex in failed[:10]:
                    print("   -", ex)

    dates = np.full((N,), np.datetime64("NaT"), dtype="datetime64[D]")
    # map per-point
    for idx, dt64 in idx_to_date.items():
        mask = (file_indices == int(idx))
        if mask.any():
            dates[mask] = dt64

    return dates, mapping, parsed_fraction

# ---------------------------
# Levina–Bickel estimator
# ---------------------------

def levina_bickel_mle(X: np.ndarray, k: int = 15, n_jobs: int = 1, eps: float = 1e-12) -> float:
    """
    Levina & Bickel (2005) maximum likelihood estimator of intrinsic dimension.

    X: (n, d)
    k: number of neighbors (typical 10-30). Requires n >= k+1.
    Returns scalar dimension estimate (float), or np.nan if not computable.
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for Levina-Bickel estimator (NearestNeighbors).")

    X = _safe_2d(X)
    n = X.shape[0]
    if n < max(3, k + 1):
        return float("nan")

    k_use = min(int(k), n - 1)
    if k_use < 2:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)  # (n, k+1) includes self at 0
    dists = dists[:, 1:]  # drop self -> (n, k_use)

    # If a row has all zeros (duplicate points), it will be invalid.
    T_k = dists[:, -1]
    if np.all(T_k <= 0):
        return float("nan")

    log_Tk = np.log(np.maximum(T_k, eps))
    log_Tj = np.log(np.maximum(dists[:, :-1], eps))
    # sum_{j=1}^{k-1} log(T_k/T_j) = (k-1)*log(T_k) - sum log(T_j)
    log_sums = (k_use - 1) * log_Tk - np.sum(log_Tj, axis=1)

    valid = np.isfinite(log_sums) & (log_sums > 0)
    if not np.any(valid):
        return float("nan")

    inv_m = log_sums[valid] / (k_use - 1)
    m_i = 1.0 / inv_m
    m_hat = float(np.mean(m_i))
    return m_hat

# ---------------------------
# Cluster helper
# ---------------------------

def _mode_label(labels: np.ndarray) -> str:
    """
    Return the most frequent label as a string (ties broken by smallest numeric).
    """
    if labels.size == 0:
        return ""
    # Handle numeric labels
    if np.issubdtype(labels.dtype, np.number):
        lbl = labels.astype(int)
        # ignore negatives for mode label
        lbl = lbl[lbl >= 0]
        if lbl.size == 0:
            return ""
        counts = np.bincount(lbl)
        return str(int(np.argmax(counts)))
    # Strings / objects
    vals, counts = np.unique(labels.astype(str), return_counts=True)
    return str(vals[int(np.argmax(counts))])

# ---------------------------
# Core computation for one NPZ
# ---------------------------

def compute_pre_post_dimensionality_for_npz(
    npz_path: Union[str, Path],
    treatment_date: datetime.date,
    hit_group: str,
    cfg: PrePostConfig,
    *,
    rng_seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows (dicts) – one per cluster that has enough points in BOTH pre and post.
    """
    npz_path = Path(npz_path)
    t0 = time.perf_counter()

    data = np.load(npz_path, allow_pickle=True)

    # load the main array
    if cfg.array_key not in data.files:
        return [{
            "animal_id": _animal_id_from_npz_path(npz_path),
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "error": f"Missing array_key '{cfg.array_key}'",
        }]

    X = _safe_2d(data[cfg.array_key])
    N = X.shape[0]

    # cluster labels
    if cfg.cluster_key not in data.files:
        return [{
            "animal_id": _animal_id_from_npz_path(npz_path),
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "error": f"Missing cluster_key '{cfg.cluster_key}'",
        }]
    cluster_labels = np.asarray(data[cfg.cluster_key]).reshape(-1)
    if cluster_labels.shape[0] != N:
        return [{
            "animal_id": _animal_id_from_npz_path(npz_path),
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "error": f"cluster labels shape {cluster_labels.shape} does not match N={N}",
        }]

    # optional syllable labels (for reporting only)
    syllable_labels = None
    if cfg.syllable_key in data.files:
        sl = np.asarray(data[cfg.syllable_key]).reshape(-1)
        if sl.shape[0] == N:
            syllable_labels = sl

    # infer per-point dates
    dates, file_map, parsed_frac = infer_point_dates_from_npz(
        data, N=N, file_key=cfg.file_key, date_key=cfg.date_key,
        year_default=int(treatment_date.year), verbose=cfg.verbose_dates
    )

    if parsed_frac < 0.5:
        return [{
            "animal_id": _animal_id_from_npz_path(npz_path),
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "treatment_date": str(treatment_date),
            "error": f"Could not assign dates reliably from NPZ (parsed {parsed_frac*100:.1f}% of file indices).",
        }]

    t_dt64 = np.datetime64(treatment_date)
    if cfg.include_treatment_day_in_post:
        pre_mask = dates < t_dt64
        post_mask = dates >= t_dt64
    else:
        pre_mask = dates < t_dt64
        post_mask = dates > t_dt64

    # optional vocalization-only filter
    if cfg.vocalization_only and (cfg.vocalization_key in data.files):
        voc = np.asarray(data[cfg.vocalization_key]).reshape(-1)
        if voc.shape[0] == N:
            voc_mask = (voc.astype(int) == 1)
            pre_mask = pre_mask & voc_mask
            post_mask = post_mask & voc_mask

    # Determine cap (subsampling per cluster per period)
    cap = cfg.max_points_per_cluster_period

    rng = np.random.default_rng(rng_seed)

    # cluster ids
    uniq_clusters = np.unique(cluster_labels.astype(int))
    if not cfg.include_noise:
        uniq_clusters = uniq_clusters[uniq_clusters != -1]

    rows: List[Dict[str, Any]] = []
    animal_id = _animal_id_from_npz_path(npz_path)

    for c in uniq_clusters:
        c = int(c)
        idx_pre = np.where((cluster_labels == c) & pre_mask)[0]
        idx_post = np.where((cluster_labels == c) & post_mask)[0]

        if idx_pre.size < cfg.min_points_per_period or idx_post.size < cfg.min_points_per_period:
            continue

        if cap is not None:
            if idx_pre.size > cap:
                idx_pre = rng.choice(idx_pre, size=cap, replace=False)
            if idx_post.size > cap:
                idx_post = rng.choice(idx_post, size=cap, replace=False)

        X_pre = X[idx_pre]
        X_post = X[idx_post]

        syl_mode = ""
        if syllable_labels is not None:
            try:
                syl_mode = _mode_label(syllable_labels[np.where(cluster_labels == c)[0]])
            except Exception:
                syl_mode = ""

        err = ""
        m_pre = float("nan")
        m_post = float("nan")
        try:
            m_pre = float(levina_bickel_mle(X_pre, k=cfg.k, n_jobs=max(1, int(cfg.n_jobs))))
            m_post = float(levina_bickel_mle(X_post, k=cfg.k, n_jobs=max(1, int(cfg.n_jobs))))
        except Exception as e:
            err = str(e)

        delta = (m_post - m_pre) if (math.isfinite(m_pre) and math.isfinite(m_post)) else float("nan")

        rows.append({
            "animal_id": animal_id,
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "cluster_id": c,
            "mode_syllable": syl_mode,
            "n_pre": int(idx_pre.size),
            "n_post": int(idx_post.size),
            "dim_pre": m_pre if math.isfinite(m_pre) else "",
            "dim_post": m_post if math.isfinite(m_post) else "",
            "delta_dim": delta if math.isfinite(delta) else "",
            "treatment_date": str(treatment_date),
            "error": err,
        })

    t1 = time.perf_counter()
    # annotate runtime in each row (helps profiling)
    if rows:
        for r in rows:
            r["npz_seconds"] = round(t1 - t0, 4)
    else:
        rows.append({
            "animal_id": animal_id,
            "npz_file": npz_path.name,
            "hit_group": hit_group,
            "treatment_date": str(treatment_date),
            "error": "No clusters had enough points in both pre and post after filtering.",
            "npz_seconds": round(t1 - t0, 4),
        })
    return rows

# ---------------------------
# Plotting
# ---------------------------

_GROUP_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Combined (visible ML + not visible)",
]

def plot_pre_post_scatter_by_group(df: pd.DataFrame, out_path: Union[str, Path], k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    fig.suptitle(f"Pre vs Post intrinsic dimension by lesion hit type (k={k})")

    any_points = False
    for ax, group in zip(axes, _GROUP_ORDER):
        sub = df[df["hit_group"] == group].copy()
        # keep only rows with numeric dim_pre/dim_post
        sub["dim_pre"] = pd.to_numeric(sub.get("dim_pre"), errors="coerce")
        sub["dim_post"] = pd.to_numeric(sub.get("dim_post"), errors="coerce")
        sub = sub.dropna(subset=["dim_pre", "dim_post"])
        ax.set_title(group)
        ax.set_xlabel("Pre dimension")
        if ax is axes[0]:
            ax.set_ylabel("Post dimension")
        # diagonal
        ax.plot([0, 1], [0, 1], linestyle="--")
        if sub.empty:
            ax.text(0.5, 0.5, "No points", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        any_points = True
        ax.scatter(sub["dim_pre"].values, sub["dim_post"].values, s=12, alpha=0.8)
        # autoscale with a little padding
        xmin = float(np.nanmin(sub["dim_pre"].values))
        xmax = float(np.nanmax(sub["dim_pre"].values))
        ymin = float(np.nanmin(sub["dim_post"].values))
        ymax = float(np.nanmax(sub["dim_post"].values))
        lo = min(xmin, ymin)
        hi = max(xmax, ymax)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        # redraw diagonal in same limits
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")

    if not any_points:
        fig.text(0.5, 0.9, "No valid cluster estimates to plot.", ha="center", va="center")

    _savefig(fig, out_path)

def plot_delta_box_by_group(df: pd.DataFrame, out_path: Union[str, Path], k: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(f"Δ intrinsic dimension (post − pre) by lesion hit type (k={k})")
    ax.set_ylabel("Δ intrinsic dimension (post − pre)")

    # numeric
    d = df.copy()
    d["delta_dim"] = pd.to_numeric(d.get("delta_dim"), errors="coerce")
    d = d.dropna(subset=["delta_dim"])

    if d.empty:
        ax.text(0.5, 0.5, "No valid cluster estimates to plot.", ha="center", va="center")
        _savefig(fig, out_path)
        return

    data = []
    labels = []
    for group in _GROUP_ORDER:
        vals = d.loc[d["hit_group"] == group, "delta_dim"].values
        if vals.size:
            data.append(vals)
            labels.append(f"{group}\n(n={vals.size})")
        else:
            data.append(np.array([]))
            labels.append(f"{group}\n(n=0)")

    # Matplotlib >=3.9 prefers tick_labels over labels; use tick_labels to avoid warnings
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    ax.axhline(0.0, linestyle="--")

    _savefig(fig, out_path)

# ---------------------------
# Metadata loading
# ---------------------------

def _read_metadata_table(metadata_xlsx: Union[str, Path], sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)
    # normalize colnames
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fuzzy: contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in str(c).lower():
                return c
    return None

def build_animal_metadata_map(metadata_xlsx: Union[str, Path], sheet_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict:
      animal_id -> {"treatment_date": date, "hit_type": str, "hit_group": str}
    """
    df = _read_metadata_table(metadata_xlsx, sheet_name)
    col_animal = _find_col(df, ["animal_id", "Animal ID", "AnimalID", "bird", "Bird ID"])
    col_treat = _find_col(df, ["treatment_date", "Treatment date", "surgery_date", "Surgery date", "treatment"])
    col_hit = _find_col(df, ["lesion hit type", "Lesion hit type", "hit_type", "Hit type"])

    if col_animal is None or col_treat is None:
        raise ValueError(
            f"Could not find required columns in metadata sheet '{sheet_name}'. "
            f"Need animal id + treatment date. Found columns: {list(df.columns)}"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        animal = str(row[col_animal]).strip()
        if not animal or animal.lower() == "nan":
            continue
        td = row[col_treat]
        # parse treatment date
        treat_date: Optional[datetime.date] = None
        if isinstance(td, datetime.datetime):
            treat_date = td.date()
        elif isinstance(td, datetime.date):
            treat_date = td
        else:
            try:
                treat_date = pd.to_datetime(td).date()
            except Exception:
                treat_date = None

        hit_type = str(row[col_hit]).strip() if (col_hit is not None and not _is_scalar_nan(row[col_hit])) else ""
        hit_group = _canonical_hit_group(hit_type)

        if treat_date is None:
            continue

        out[animal] = {"treatment_date": treat_date, "hit_type": hit_type, "hit_group": hit_group}
    return out

# ---------------------------
# Root runner
# ---------------------------

def _choose_defaults_by_power(computer_power: str, workers: int, n_jobs: int, cap: Optional[int]) -> Tuple[int, int, Optional[int]]:
    """
    Apply auto defaults when args are 0/None.
    """
    cpu = os.cpu_count() or 4
    power = (computer_power or "laptop").lower()

    # workers: across files (processes)
    if workers == 0:
        if power == "pro":
            workers = max(1, min(8, cpu // 2))
        else:
            workers = 1

    # n_jobs: threads within NN search
    if n_jobs == 0:
        if power == "pro":
            n_jobs = max(1, min(8, cpu))
        else:
            n_jobs = max(1, min(2, cpu))

    # cap
    if cap is None:
        cap = 20000 if power == "pro" else 10000

    return workers, n_jobs, cap

def _find_npz_files(root_dir: Union[str, Path], recursive: bool) -> List[Path]:
    root_dir = Path(root_dir)
    if root_dir.is_file() and root_dir.suffix.lower() == ".npz":
        return [root_dir]
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)
    if recursive:
        return sorted(root_dir.rglob("*.npz"))
    else:
        return sorted(root_dir.glob("*.npz"))

def run_root_directory_pre_post_treatment(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    metadata_sheet: str,
    out_dir: Optional[Union[str, Path]],
    recursive: bool,
    cfg: PrePostConfig,
    computer_power: str,
    workers: int,
) -> Dict[str, str]:
    """
    Process all NPZ files found under root_dir.
    """
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / f"lb_pre_post_dimensionality_k{cfg.k}"
    out_dir = _ensure_dir(out_dir)

    animal_meta = build_animal_metadata_map(metadata_xlsx, metadata_sheet)

    npz_files = _find_npz_files(root_dir, recursive=recursive)

    # filter to those that have metadata
    jobs: List[Tuple[Path, datetime.date, str]] = []
    for npz in npz_files:
        aid = _animal_id_from_npz_path(npz)
        if npz.stem != aid:
            continue  # skip segment/aux NPZs like USA####_..._segment_*.npz
        if aid not in animal_meta:
            continue
        md = animal_meta[aid]
        jobs.append((npz, md["treatment_date"], md["hit_group"]))

    # announce
    print(f"[pre/post] NPZ files with metadata: {len(jobs)} (workers={workers}, n_jobs={cfg.n_jobs}, cap={cfg.max_points_per_cluster_period})")

    all_rows: List[Dict[str, Any]] = []
    t_all0 = time.perf_counter()

    if workers <= 1 or len(jobs) <= 1:
        for npz, tdate, hit_group in jobs:
            print(f"[pre/post] { _animal_id_from_npz_path(npz) } ({hit_group}) file={npz.name}")
            rows = compute_pre_post_dimensionality_for_npz(npz, tdate, hit_group, cfg)
            # print per-file time if available
            if rows and "npz_seconds" in rows[0]:
                print(f"[pre/post] done: {npz.name} in {rows[0]['npz_seconds']} s")
            all_rows.extend(rows)
    else:
        # parallel across files
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # NOTE: On macOS, this requires being in __main__ (we are).
        def _worker(npz_path_str: str, tdate_iso: str, hit_group_str: str, cfg_dict: dict) -> List[Dict[str, Any]]:
            npz_p = Path(npz_path_str)
            tdate = datetime.date.fromisoformat(tdate_iso)
            cfg_local = PrePostConfig(**cfg_dict)
            return compute_pre_post_dimensionality_for_npz(npz_p, tdate, hit_group_str, cfg_local)

        cfg_dict = cfg.__dict__.copy()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for npz, tdate, hit_group in jobs:
                futs.append(ex.submit(_worker, str(npz), tdate.isoformat(), hit_group, cfg_dict))
            for fut in as_completed(futs):
                rows = fut.result()
                all_rows.extend(rows)

    t_all1 = time.perf_counter()
    print(f"[pre/post] TOTAL time: {t_all1 - t_all0:.2f} s")

    # Save results CSV
    ts = _now_tag()
    out_csv = out_dir / f"lb_pre_post_cluster_dimensionality_k{cfg.k}_{ts}.csv"
    _write_rows_csv(out_csv, all_rows)
    print(f"Saved results CSV: {out_csv}")

    # Build plots from valid cluster rows
    df = pd.DataFrame(all_rows)
    # keep only cluster rows (cluster_id exists)
    if "cluster_id" in df.columns:
        df_plot = df.copy()
        # filter out error-only rows (where dim_pre missing)
        df_plot["dim_pre"] = pd.to_numeric(df_plot.get("dim_pre"), errors="coerce")
        df_plot["dim_post"] = pd.to_numeric(df_plot.get("dim_post"), errors="coerce")
        df_plot["delta_dim"] = pd.to_numeric(df_plot.get("delta_dim"), errors="coerce")
        df_plot = df_plot.dropna(subset=["dim_pre", "dim_post"], how="any")
    else:
        df_plot = pd.DataFrame()

    plot_scatter = out_dir / f"pre_vs_post_scatter_by_hit_k{cfg.k}_{ts}.png"
    plot_delta = out_dir / f"delta_dim_by_hit_type_k{cfg.k}_{ts}.png"

    if df_plot.empty:
        # write empty-ish plots with messages
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid cluster estimates to plot.", ha="center", va="center")
        _savefig(fig, plot_scatter)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid cluster estimates to plot.", ha="center", va="center")
        _savefig(fig, plot_delta)
        print(f"Saved plot (empty): {plot_delta}")
        print(f"Saved plot (empty): {plot_scatter}")
    else:
        plot_pre_post_scatter_by_group(df_plot, plot_scatter, k=cfg.k)
        plot_delta_box_by_group(df_plot, plot_delta, k=cfg.k)
        print(f"Saved plot: {plot_delta}")
        print(f"Saved plot: {plot_scatter}")

    return {
        "results_csv": str(out_csv),
        "plot_delta": str(plot_delta),
        "plot_scatter": str(plot_scatter),
        "out_dir": str(out_dir),
    }

# ---------------------------
# CLI
# ---------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str, required=True, help="Folder containing NPZ(s), or a single .npz file.")
    p.add_argument("--metadata-xlsx", type=str, required=True, help="Excel file containing treatment dates + hit types.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <root-dir>/lb_pre_post_dimensionality_k{K}).")
    p.add_argument("--recursive", action="store_true", help="Search for NPZs recursively under root-dir.")
    p.add_argument("--metadata-sheet", type=str, default="animal_hit_type_summary", help="Sheet name in metadata xlsx.")
    p.add_argument("--array-key", type=str, default="predictions", help="Pointwise array key in NPZ (NxD).")
    p.add_argument("--cluster-key", type=str, default="hdbscan_labels", help="Cluster label key in NPZ (N,).")
    p.add_argument("--syllable-key", type=str, default="ground_truth_labels", help="Optional syllable label key (N,) for reporting.")
    p.add_argument("--file-key", type=str, default="file_indices", help="Key for per-point file indices (N,).")
    p.add_argument("--date-key", type=str, default="date", help="Optional per-point date key in NPZ. If absent, uses file_map.")
    p.add_argument("--include-noise", action="store_true", help="Include cluster label -1.")
    p.add_argument("--no-vocalization-only", action="store_true", help="Do NOT filter to vocalization==1.")
    p.add_argument("--vocalization-key", type=str, default="vocalization", help="Vocalization key in NPZ (N,).")

    p.add_argument("--k", type=int, default=15, help="Neighbors for LB-MLE.")
    p.add_argument("--point-agg", type=str, default="mean", choices=["mean", "median"], help="Reserved (currently unused).")
    p.add_argument("--n-jobs", type=int, default=0, help="Threads for NearestNeighbors (0 => auto by computer-power).")
    p.add_argument("--min-points-per-period", type=int, default=50, help="Min points per cluster per period.")
    p.add_argument("--exclude-treatment-day-from-post", action="store_true", help="If set, treatment day is excluded from post (post is strictly after).")
    p.add_argument("--max-points-per-cluster-period", type=int, default=None, help="Subsample cap per cluster per period (None => auto by computer-power).")
    p.add_argument("--computer-power", type=str, default="laptop", choices=["laptop", "pro"], help="Sets default workers/n_jobs/caps when not provided.")
    p.add_argument("--workers", type=int, default=0, help="Processes across NPZ files (0 => auto by computer-power; 1 => serial).")
    p.add_argument("--verbose-dates", action="store_true", help="Print date-parsing diagnostics.")
    return p

def main() -> None:
    args = _build_arg_parser().parse_args()

    if not _HAVE_SKLEARN:
        raise SystemExit("ERROR: scikit-learn is required (pip/conda install scikit-learn).")

    workers, n_jobs, cap = _choose_defaults_by_power(
        args.computer_power, args.workers, args.n_jobs, args.max_points_per_cluster_period
    )

    cfg = PrePostConfig(
        array_key=args.array_key,
        cluster_key=args.cluster_key,
        syllable_key=args.syllable_key,
        file_key=args.file_key,
        date_key=args.date_key,
        include_noise=bool(args.include_noise),
        vocalization_only=not bool(args.no_vocalization_only),
        vocalization_key=args.vocalization_key,
        k=int(args.k),
        n_jobs=int(n_jobs),
        min_points_per_period=int(args.min_points_per_period),
        include_treatment_day_in_post=not bool(args.exclude_treatment_day_from_post),
        max_points_per_cluster_period=int(cap) if cap is not None else None,
        verbose_dates=bool(args.verbose_dates),
    )

    paths = run_root_directory_pre_post_treatment(
        root_dir=args.root_dir,
        metadata_xlsx=args.metadata_xlsx,
        metadata_sheet=args.metadata_sheet,
        out_dir=args.out_dir,
        recursive=bool(args.recursive),
        cfg=cfg,
        computer_power=args.computer_power,
        workers=int(workers),
    )

    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
