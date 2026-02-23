#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_pre_vs_post_treatment.py

Purpose
-------
Compare pre vs post-treatment (lesion) intrinsic dimensionality within each HDBSCAN
cluster using the Levina–Bickel estimator at a FIXED k (default k=15).

This script:
  • Finds NPZs under a root directory (each NPZ named by animal_id, or starting with animal_id)
  • Uses an Excel metadata sheet to fetch:
      - Treatment date
      - Lesion hit type (mapped into 3 groups)
  • Splits timebins into pre vs post by comparing each timebin's recording date to the treatment date
      - Per timebin date is inferred from a date array key if present, otherwise parsed from the "file" strings
  • For each cluster:
      - Computes m_pre and m_post using LB at k=15 (single k)
      - Saves a results CSV
      - Makes summary plots (by lesion hit type)

Optional (variance tiers)
-------------------------
If you provide a phrase-duration stats CSV (usage_balanced_phrase_duration_stats.csv),
the script can label each cluster as "high" or "low" variance using the dominant syllable
label in that cluster (top 30% variance within animal = "high").
If stats_csv is None, variance_tier will be "unknown" and tier-specific plots are skipped.

Inputs expected in each NPZ
---------------------------
Required keys:
  - array_key (default "predictions"): (N, D)
  - cluster_key (default "hdbscan_labels"): (N,)

Optional keys:
  - file_key (default "file"): (N,) strings used to parse recording date if no date_key exists
  - date_key: if your NPZ already stores dates/timestamps in an array
  - syllable_key (default "ground_truth_labels"): (N,) used to map cluster -> dominant syllable label
  - vocalization_key (default "vocalization"): (N,) if present and enabled, filters to vocalization==1

Dependencies
------------
numpy, pandas, matplotlib, scikit-learn, openpyxl (for reading Excel)

CLI
---
python Levina_Bickel_pre_vs_post_treatment.py --root-dir /path/to/npzs --recursive \
  --metadata-xlsx /path/to/Area_X_lesion_metadata_with_hit_types.xlsx \
  --out-dir /path/to/out \
  --k 15

Spyder usage is included at bottom (triple-quoted block).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime
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


def _jitter(x: float, n: int, scale: float = 0.06, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return x + rng.uniform(-scale, scale, size=n)


# ============================================================
# Levina–Bickel at a single k
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


def levina_bickel_single_k(
    X: np.ndarray,
    k: int = 15,
    point_agg: str = "median",
    eps: float = 1e-12,
    n_jobs: int = 1,
) -> float:
    """
    LB intrinsic dimension estimate at a single k.

    For each point x_i:
      m_i(k) = (k-1) / sum_{j=1}^{k-1} log(T_k / T_j)

    Aggregate across points using point_agg (mean or median).

    Notes:
      - Requires k >= 2 and k <= n-1.
      - If some points produce invalid log_sums (<=0), those points are ignored (NaN).
    """
    X = _safe_2d(X)
    n = X.shape[0]
    k = int(k)

    if k < 2:
        raise ValueError("k must be >= 2")
    if n < k + 1:
        raise ValueError(f"Need n >= k+1; got n={n}, k={k}")

    _, logT, cumsum_logT = _compute_knn_logs(X, k_max=k, eps=eps, n_jobs=n_jobs)

    # sum_{j=1}^{k-1} log(T_k/T_j) = (k-1)*log(T_k) - sum_{j=1}^{k-1} log(T_j)
    logT_k = logT[:, k - 1]
    sum_logT_j = cumsum_logT[:, k - 2]
    log_sums = (k - 1) * logT_k - sum_logT_j

    valid = np.isfinite(log_sums) & (log_sums > 0)
    m_k = np.full(n, np.nan, dtype=float)
    m_k[valid] = (k - 1) / log_sums[valid]

    if point_agg.lower() == "mean":
        return float(np.nanmean(m_k))
    if point_agg.lower() == "median":
        return float(np.nanmedian(m_k))
    raise ValueError("point_agg must be 'mean' or 'median'")


# ============================================================
# Metadata + mapping
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


# ============================================================
# Optional variance tiers from stats CSV
# ============================================================
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


# ============================================================
# Date inference from NPZ
# ============================================================
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

    # Try date arrays first
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

    # Fall back: parse from file strings
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


def animal_id_from_npz_path(npz_path: Path, metadata_animals: Optional[set] = None) -> str:
    stem = npz_path.stem
    if metadata_animals and stem in metadata_animals:
        return stem
    tok = stem.split("_")[0]
    if metadata_animals and tok in metadata_animals:
        return tok
    return stem


# ============================================================
# Main pre/post computation
# ============================================================
@dataclass
class PrePostConfig:
    # NPZ keys
    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    syllable_key: str = "ground_truth_labels"
    file_key: str = "file"
    date_key: Optional[str] = None

    # filtering
    include_noise: bool = False
    use_vocalization_only: bool = True
    vocalization_key: str = "vocalization"

    # split rules
    include_treatment_day_in_post: bool = True

    # LB params
    k: int = 15
    point_agg: str = "median"
    eps: float = 1e-12
    n_jobs: int = 1

    # minimums / speed caps
    min_points_per_period: int = 50
    max_points_per_cluster_period: Optional[int] = None
    subsample_seed: int = 0


def compute_pre_post_for_npz(
    npz_path: str | Path,
    treatment_date: date,
    hit_type_raw: str,
    hit_group: str,
    animal_id: str,
    cfg: PrePostConfig,
    variance_tier_map: Optional[Dict[Tuple[str, int], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute pre/post LB for each cluster within one NPZ.
    """
    npz_path = Path(npz_path)
    rows: List[Dict[str, Any]] = []

    data = np.load(npz_path, allow_pickle=True)
    if cfg.array_key not in data:
        raise KeyError(f"Missing array_key {cfg.array_key!r} in {npz_path.name}")
    if cfg.cluster_key not in data:
        raise KeyError(f"Missing cluster_key {cfg.cluster_key!r} in {npz_path.name}")

    X = _safe_2d(data[cfg.array_key])
    cluster_ids = np.asarray(data[cfg.cluster_key])
    n = X.shape[0]
    if cluster_ids.shape[0] != n:
        raise ValueError(f"X has n={n} but clusters has n={cluster_ids.shape[0]}")

    keep_mask = np.ones(n, dtype=bool)
    if cfg.use_vocalization_only and (cfg.vocalization_key in data):
        voc = np.asarray(data[cfg.vocalization_key]).astype(int)
        if voc.shape[0] == n:
            keep_mask &= (voc == 1)

    point_dates = infer_point_dates_from_npz(
        data=data, n=n, date_key=cfg.date_key, file_key=cfg.file_key
    )
    pre_mask, post_mask = split_pre_post_masks(
        point_dates=point_dates,
        treatment_date=treatment_date,
        include_treatment_day_in_post=cfg.include_treatment_day_in_post,
    )
    pre_mask &= keep_mask
    post_mask &= keep_mask

    syllable_labels = None
    if cfg.syllable_key in data:
        tmp = np.asarray(data[cfg.syllable_key])
        if tmp.shape[0] == n:
            syllable_labels = tmp

    unique_clusters = np.unique(cluster_ids)
    if not cfg.include_noise:
        unique_clusters = unique_clusters[unique_clusters != -1]

    rng = np.random.RandomState(cfg.subsample_seed)

    # Ensure min_points_per_period is also feasible for k
    min_needed = max(int(cfg.min_points_per_period), int(cfg.k) + 1)

    for cid in unique_clusters:
        cid = int(cid)
        c_mask = (cluster_ids == cid)

        idx_pre = np.where(c_mask & pre_mask)[0]
        idx_post = np.where(c_mask & post_mask)[0]
        n_pre = int(idx_pre.size)
        n_post = int(idx_post.size)

        # dominant syllable label (optional, for variance tiers)
        syll = None
        if syllable_labels is not None:
            syll = cluster_mode_syllable(cluster_ids, syllable_labels, cid, ignore_values=(-1,))

        var_tier = "unknown"
        if variance_tier_map is not None and syll is not None:
            var_tier = variance_tier_map.get((animal_id, int(syll)), "unknown")

        if (n_pre < min_needed) or (n_post < min_needed):
            rows.append({
                "npz_path": str(npz_path),
                "animal_id": animal_id,
                "cluster_id": cid,
                "n_pre": n_pre,
                "n_post": n_post,
                "k": int(cfg.k),
                "m_pre": "",
                "m_post": "",
                "delta_m": "",
                "hit_type_raw": hit_type_raw,
                "hit_group": hit_group,
                "syllable_label": "" if syll is None else int(syll),
                "variance_tier": var_tier,
                "error": f"skipped: insufficient points (need >= {min_needed} per period for k={cfg.k})",
            })
            continue

        if cfg.max_points_per_cluster_period is not None:
            cap = int(cfg.max_points_per_cluster_period)
            if n_pre > cap:
                idx_pre = rng.choice(idx_pre, size=cap, replace=False)
                n_pre = int(idx_pre.size)
            if n_post > cap:
                idx_post = rng.choice(idx_post, size=cap, replace=False)
                n_post = int(idx_post.size)

        try:
            m_pre = levina_bickel_single_k(
                X[idx_pre], k=cfg.k, point_agg=cfg.point_agg, eps=cfg.eps, n_jobs=cfg.n_jobs
            )
            m_post = levina_bickel_single_k(
                X[idx_post], k=cfg.k, point_agg=cfg.point_agg, eps=cfg.eps, n_jobs=cfg.n_jobs
            )

            rows.append({
                "npz_path": str(npz_path),
                "animal_id": animal_id,
                "cluster_id": cid,
                "n_pre": n_pre,
                "n_post": n_post,
                "k": int(cfg.k),
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
                "k": int(cfg.k),
                "m_pre": "",
                "m_post": "",
                "delta_m": "",
                "hit_type_raw": hit_type_raw,
                "hit_group": hit_group,
                "syllable_label": "" if syll is None else int(syll),
                "variance_tier": var_tier,
                "error": f"{type(e).__name__}: {e}",
            })

    return rows


# ============================================================
# Plotting
# ============================================================
def plot_delta_by_hit_type(
    df: pd.DataFrame,
    out_png: str | Path,
    title: str = "Δ intrinsic dimension (post − pre) by lesion hit type",
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
    dd["delta_m"] = pd.to_numeric(dd["delta_m"], errors="coerce")
    dd = dd.dropna(subset=["delta_m"])
    dd = dd[dd["hit_group"].isin(hit_order)]

    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    for hit in hit_order:
        v = dd[dd["hit_group"] == hit]["delta_m"].values.astype(float)
        data.append(v)
        labels.append(f"{hit}\n(n={v.size})")

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.5)

    for i, v in enumerate(data, start=1):
        if v.size:
            ax.scatter(_jitter(i, v.size, seed=100 + i), v, s=18, alpha=0.7)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Δ dimension")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def plot_pre_vs_post_scatter_by_hit_type(
    df: pd.DataFrame,
    out_png: str | Path,
    title: str = "Pre vs Post intrinsic dimension by lesion hit type",
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
        ax.scatter(sub["m_pre"], sub["m_post"], s=26, alpha=0.8)
        ax.plot(lims, lims, linestyle="--", linewidth=1)
        ax.set_title(hit)
        ax.set_xlabel("Pre dimension")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Post dimension")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


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

    if dd.empty:
        print("Variance-tier plot: no rows with variance tiers; skipping.")
        return

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


# ============================================================
# Root-directory wrapper
# ============================================================
def run_root_directory_pre_post_treatment(
    root_dir: str | Path,
    metadata_xlsx: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    metadata_sheet: str = "metadata_with_hit_type",
    stats_csv: Optional[str | Path] = None,  # optional
    variance_high_quantile: float = 0.70,
    stats_group_value: str = "Post",
    stats_variance_col: str = "Pre_Variance_ms2",
    cfg: Optional[PrePostConfig] = None,
) -> Dict[str, str]:
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / f"lb_pre_post_dimensionality_k15"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = PrePostConfig()

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

    variance_tier_map: Optional[Dict[Tuple[str, int], str]] = None
    if stats_csv is not None:
        variance_tier_map = build_variance_tier_map(
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
                "k": int(cfg.k),
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

        try:
            rows = compute_pre_post_for_npz(
                npz_path=npz_path,
                treatment_date=treatment_date,
                hit_type_raw=hit_type_raw,
                hit_group=hit_group,
                animal_id=animal_id,
                cfg=cfg,
                variance_tier_map=variance_tier_map,
            )
            all_rows.extend(rows)
        except Exception as e:
            all_rows.append({
                "npz_path": str(npz_path),
                "animal_id": animal_id,
                "cluster_id": "",
                "n_pre": "",
                "n_post": "",
                "k": int(cfg.k),
                "m_pre": "",
                "m_post": "",
                "delta_m": "",
                "hit_type_raw": hit_type_raw,
                "hit_group": hit_group,
                "syllable_label": "",
                "variance_tier": "unknown",
                "error": f"{type(e).__name__}: {e}",
            })

    out_csv = out_dir / f"lb_pre_post_cluster_dimensionality_k{int(cfg.k)}_{ts}.csv"
    _write_rows_csv(out_csv, all_rows)
    print(f"\nSaved results CSV: {out_csv}")

    df = pd.DataFrame(all_rows)

    plot_delta = out_dir / f"delta_dim_by_hit_type_k{int(cfg.k)}_{ts}.png"
    plot_delta_by_hit_type(df, plot_delta)

    plot_scatter = out_dir / f"pre_vs_post_scatter_by_hit_k{int(cfg.k)}_{ts}.png"
    plot_pre_vs_post_scatter_by_hit_type(df, plot_scatter)

    # Optional tier plot
    plot_delta_tier = ""
    if stats_csv is not None:
        plot_delta_tier_path = out_dir / f"delta_dim_by_hit_and_variance_k{int(cfg.k)}_{ts}.png"
        plot_delta_by_hit_and_variance(df, plot_delta_tier_path)
        plot_delta_tier = str(plot_delta_tier_path)

    return {
        "results_csv": str(out_csv),
        "plot_delta_by_hit": str(plot_delta),
        "plot_scatter_by_hit": str(plot_scatter),
        "plot_delta_by_hit_and_variance": plot_delta_tier,
        "out_dir": str(out_dir),
    }


# ============================================================
# CLI
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Pre vs post-treatment Levina–Bickel intrinsic dimension per cluster (fixed k).")
    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--metadata-xlsx", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--metadata-sheet", type=str, default="metadata_with_hit_type")

    p.add_argument("--array-key", type=str, default="predictions")
    p.add_argument("--cluster-key", type=str, default="hdbscan_labels")
    p.add_argument("--syllable-key", type=str, default="ground_truth_labels")
    p.add_argument("--file-key", type=str, default="file")
    p.add_argument("--date-key", type=str, default=None)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--no-vocalization-only", action="store_true")
    p.add_argument("--vocalization-key", type=str, default="vocalization")

    p.add_argument("--k", type=int, default=15)
    p.add_argument("--point-agg", type=str, default="median", choices=["mean", "median"])
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-points-per-period", type=int, default=50)
    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument("--max-points-per-cluster-period", type=int, default=None)

    # optional variance tiers
    p.add_argument("--stats-csv", type=str, default=None)
    p.add_argument("--variance-high-quantile", type=float, default=0.70)
    p.add_argument("--stats-group-value", type=str, default="Post")
    p.add_argument("--stats-variance-col", type=str, default="Pre_Variance_ms2")

    args = p.parse_args()

    cfg = PrePostConfig(
        array_key=args.array_key,
        cluster_key=args.cluster_key,
        syllable_key=args.syllable_key,
        file_key=args.file_key,
        date_key=args.date_key,
        include_noise=args.include_noise,
        use_vocalization_only=(not args.no_vocalization_only),
        vocalization_key=args.vocalization_key,
        include_treatment_day_in_post=(not args.exclude_treatment_day_from_post),
        k=args.k,
        point_agg=args.point_agg,
        n_jobs=args.n_jobs,
        min_points_per_period=args.min_points_per_period,
        max_points_per_cluster_period=args.max_points_per_cluster_period,
    )

    paths = run_root_directory_pre_post_treatment(
        root_dir=args.root_dir,
        metadata_xlsx=args.metadata_xlsx,
        out_dir=args.out_dir,
        recursive=args.recursive,
        metadata_sheet=args.metadata_sheet,
        stats_csv=args.stats_csv,
        variance_high_quantile=args.variance_high_quantile,
        stats_group_value=args.stats_group_value,
        stats_variance_col=args.stats_variance_col,
        cfg=cfg,
    )

    print("\nOutputs:")
    for k, v in paths.items():
        if v:
            print(f"  {k}: {v}")


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
# import Levina_Bickel_pre_vs_post_treatment as lbpp
# importlib.reload(lbpp)
#
# root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
# metadata_xlsx = root_dir / "Area_X_lesion_metadata_with_hit_types.xlsx"
#
# cfg = lbpp.PrePostConfig(
#     array_key="predictions",
#     cluster_key="hdbscan_labels",
#     syllable_key="ground_truth_labels",
#     file_key="file",
#     date_key=None,                 # set if your NPZ has a date array key
#     include_noise=False,
#     use_vocalization_only=True,
#     min_points_per_period=50,
#     include_treatment_day_in_post=True,
#     k=15,
#     point_agg="median",
#     max_points_per_cluster_period=5000,  # optional speed cap
# )
#
# # Optional variance tiers:
# # stats_csv = root_dir / "usage_balanced_phrase_duration_stats.csv"
# stats_csv = None
#
# paths = lbpp.run_root_directory_pre_post_treatment(
#     root_dir=root_dir,
#     metadata_xlsx=metadata_xlsx,
#     stats_csv=stats_csv,
#     out_dir=root_dir / "lb_pre_post_dimensionality_k15",
#     recursive=True,
#     cfg=cfg,
#     variance_high_quantile=0.70,     # top 30% high variance
#     stats_group_value="Post",
#     stats_variance_col="Pre_Variance_ms2",
#     metadata_sheet="metadata_with_hit_type",
# )
#
# print(paths)
