#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_pre_vs_post_treatment.py

Compute Levina–Bickel intrinsic dimensionality estimates PRE vs POST treatment
from "bird seasonality" style NPZs (with file_indices + file_map).

Key expectations for each NPZ
-----------------------------
Required keys:
  - file_indices   (N,) int   : per-point file index
  - file_map       (0-d object holding dict) : maps file_index -> filename (often tuple)
  - hdbscan_labels (N,) int   : cluster id per point (default cluster key)
  - predictions    (N, D) float : high-dim features to estimate ID from (default array key)
Optional keys:
  - vocalization   (N,) int : 1 for vocalization; default keeps only 1s

How PRE/POST is determined
--------------------------
We parse a datetime per *source file* referenced in file_map, then label points
by their file_index as pre/post relative to a treatment date.

Robust datetime parsing:
  1) If filename has an Excel-serial-style token as parts[1] (e.g. 45383.35),
     we interpret it as days since 1899-12-30 (Excel default) including fraction.
  2) Else we parse parts[2:7] as month, day, hour, minute, second, using a year:
        - any explicit 4-digit year token in the filename, else
        - a default year (defaults to treatment_date.year if known, else 2024).

Metadata
--------
We look up, per animal_id:
  - treatment date (from a sheet that has "Treatment date", usually "metadata")
  - lesion hit type (from "animal_hit_type_summary" or similar)

If the sheet you pass via --metadata-sheet does NOT contain "Treatment date",
we will automatically also read the "metadata" sheet to get treatment dates.

Outputs
-------
- CSV of per-cluster pre/post ID estimates
- Scatter plot (pre vs post) split by 3 lesion-hit-type groups
- Boxplot of Δ(post - pre) split by group

Example
-------
python Levina_Bickel_pre_vs_post_treatment.py \
  --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "animal_hit_type_summary" \
  --computer-power pro \
  --workers 4 \
  --k 15
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import argparse
import csv
import math
import os
import time
import datetime as _dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn for KNN distances
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PrePostConfig:
    root_dir: Path
    metadata_xlsx: Path
    metadata_sheet: str

    out_dir: Optional[Path] = None
    stats_csv: Optional[Path] = None  # optional existing combined CSV to use/extend
    hit_type_col: Optional[str] = None  # optional override for hit type column in metadata sheet
    treatment_date_col: Optional[str] = None  # optional override for treatment date column in metadata sheet
    recursive: bool = False

    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"
    date_key: str = "file_map"  # only used for naming; NPZ uses file_map

    include_noise: bool = False  # include cluster label -1
    vocalization_only: bool = True
    vocalization_key: str = "vocalization"

    k: int = 15
    point_agg: str = "mean"  # mean or median of local m_i estimates
    n_jobs: int = 0          # threads for NearestNeighbors (0 => auto)
    workers: int = 0         # processes across NPZ files (0 => auto)

    min_points_per_period: int = 200  # min points required in pre and post for a cluster
    exclude_treatment_day_from_post: bool = False
    max_points_per_cluster_period: Optional[int] = None  # subsample cap per (cluster, period)

    computer_power: str = "laptop"  # laptop or pro

    # optional overrides
    override_treatment_date: Optional[str] = None  # YYYY-MM-DD
    override_hit_type: Optional[str] = None

    verbose_dates: bool = False


# -----------------------------
# Small helpers
# -----------------------------
def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _unwrap_file_map_value(v: Any) -> str:
    """
    file_map often stores values like ('filename.npz',) or ['filename.npz', ...]
    Return a best-effort string path.
    """
    x = v
    # peel single-element containers
    for _ in range(4):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        else:
            break
    return str(x)


def _infer_animal_id(npz_path: Path) -> str:
    # Prefer stem (USA5288 from USA5288.npz). If stem has extra suffix, keep first token.
    stem = npz_path.stem
    # common: USA5288 or USA5288_embeddings
    return stem.split("_")[0]


# -----------------------------
# Date parsing
# -----------------------------
_EXCEL_ORIGIN = _dt.datetime(1899, 12, 30)


def _parse_excel_serial_days(token: str) -> Optional[_dt.datetime]:
    """
    Convert an Excel "serial days" token (float string) into datetime.
    Example: 45383.35215933 -> 2024-04-01 ...
    """
    try:
        x = float(token)
    except Exception:
        return None
    if not (10000.0 <= x <= 80000.0):
        # sanity bounds: ~1927 to ~2119
        return None
    try:
        return _EXCEL_ORIGIN + _dt.timedelta(days=float(x))
    except Exception:
        return None


def parse_datetime_from_filename(file_path: str | Path, *, year_default: int = 2024) -> Optional[_dt.datetime]:
    """
    Parses filenames like:
      USA5288_45383.35215933_4_1_9_46_55_segment_1.npz

    Strategy:
      1) If parts[1] looks like Excel serial days -> use that (keeps sub-day time)
      2) Else use parts[2:7] as month/day/hour/min/sec with year_default,
         unless a 4-digit year token exists.
    """
    base = Path(file_path).name
    parts = base.split("_")

    if len(parts) >= 2:
        dt = _parse_excel_serial_days(parts[1])
        if dt is not None:
            return dt

    # override year if a 4-digit year is present
    year = year_default
    for p in parts:
        if p.isdigit() and len(p) == 4:
            y = int(p)
            if 1990 <= y <= 2100:
                year = y
                break

    try:
        month, day, hour, minute, second = map(int, parts[2:7])
        return _dt.datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def _treatment_dt_from_any(value: Any) -> Optional[_dt.datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime.combine(value, _dt.time(0, 0, 0))
    # pandas Timestamp
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
    except Exception:
        pass
    # string
    try:
        s = str(value)
        # accept YYYY-MM-DD or full ISO
        return _dt.datetime.fromisoformat(s)
    except Exception:
        return None


# -----------------------------
# Metadata loading
# -----------------------------
def _normalize_hit_type(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


import re


def _map_hit_type_to_group(hit_type: str) -> str:
    """
    Map raw hit types to the 3 groups you want.
    """
    if hit_type is None:
        return "unknown"
    norm = _normalize_hit_type(str(hit_type))

    if "sham" in norm:
        return "sham saline injection"

    if "singlehit" in norm:
        return "Area X visible (single hit)"

    # combined group
    if ("medial" in norm and "lateral" in norm) or ("ml" in norm and "visible" in norm):
        return "Combined (visible ML + not visible)"
    if "notvisible" in norm or ("large" in norm and "notvisible" in norm):
        return "Combined (visible ML + not visible)"

    # fallback: keep original for debugging
    return str(hit_type)


def _find_column(
    df: pd.DataFrame,
    *,
    preferred: List[str],
    contains_all: Optional[List[str]] = None,
    contains_any: Optional[List[str]] = None,
) -> Optional[str]:
    cols = list(df.columns)
    norm = {c: str(c).strip().lower().replace("_", " ") for c in cols}

    for pref in preferred:
        p = pref.strip().lower().replace("_", " ")
        for c, n in norm.items():
            if n == p:
                return c

    if contains_all:
        keys = [k.lower() for k in contains_all]
        for c, n in norm.items():
            if all(k in n for k in keys):
                return c

    if contains_any:
        keys = [k.lower() for k in contains_any]
        for c, n in norm.items():
            if any(k in n for k in keys):
                return c

    return None


def build_animal_metadata_map(metadata_xlsx: Path, metadata_sheet: str, *, verbose: bool = False,
                             hit_type_col: Optional[str] = None, treatment_date_col: Optional[str] = None
                             ) -> Dict[str, Dict[str, Any]]:
    """Load animal_id -> metadata dict.

    Returns dict values with keys:
      animal_id, treatment_date (datetime or None), hit_type, hit_type_norm, group
    """
    metadata_xlsx = Path(metadata_xlsx)

    df_main = pd.read_excel(metadata_xlsx, sheet_name=metadata_sheet)

    animal_col = _find_column(df_main, preferred=["Animal ID", "animal_id"], contains_any=["animal id", "animal"])
    if animal_col is None:
        raise ValueError(f"Could not find Animal ID column in sheet '{metadata_sheet}'.")

    # Treatment date column (may be absent)
    treat_col = treatment_date_col or _find_column(df_main, preferred=["Treatment date", "treatment_date"], contains_all=["treatment", "date"])

    # Hit type: strongly prefer Lesion hit type
    if hit_type_col is not None:
        hit_col = hit_type_col
    else:
        hit_col = _find_column(df_main, preferred=["Lesion hit type", "lesion_hit_type"], contains_all=["lesion", "hit", "type"])
        if hit_col is None:
            hit_col = _find_column(df_main, preferred=[], contains_all=["hit", "type"])

    # If no treatment date column, fall back to 'metadata' sheet
    df_meta = None
    animal_col_meta = None
    treat_col_meta = None
    if treat_col is None:
        try:
            df_meta = pd.read_excel(metadata_xlsx, sheet_name="metadata")
            animal_col_meta = _find_column(df_meta, preferred=["Animal ID", "animal_id"], contains_any=["animal id", "animal"])
            treat_col_meta = _find_column(df_meta, preferred=["Treatment date", "treatment_date"], contains_all=["treatment", "date"])
        except Exception:
            df_meta = None

    out: Dict[str, Dict[str, Any]] = {}

    for _, row in df_main.iterrows():
        animal = str(row.get(animal_col, "")).strip()
        if not animal:
            continue

        td = _treatment_dt_from_any(row.get(treat_col)) if treat_col is not None else None

        hit = row.get(hit_col) if hit_col is not None else None
        # Guard: if we picked a generic column with values like 'bilateral', prefer a lesion hit type if present
        if hit_col is not None:
            s = str(hit).strip().lower() if hit is not None else ""
            if s in {"bilateral", "unilateral", "unknown", "nan", ""}:
                better = _find_column(df_main, preferred=["Lesion hit type", "lesion_hit_type"], contains_all=["lesion", "hit", "type"])
                if better is not None:
                    hit = row.get(better, hit)
                    hit_col = better

        hit_type = str(hit) if hit is not None and str(hit).strip() != "" else None
        hit_norm = _normalize_hit_type(hit_type) if hit_type is not None else None
        group = _map_hit_type_to_group(hit_norm) if hit_norm is not None else "unknown"

        out[animal] = {
            "animal_id": animal,
            "treatment_date": td,
            "hit_type": hit_type,
            "hit_type_norm": hit_norm,
            "group": group,
        }

    if df_meta is not None and animal_col_meta is not None and treat_col_meta is not None:
        for _, row in df_meta.iterrows():
            animal = str(row.get(animal_col_meta, "")).strip()
            if not animal:
                continue
            td = _treatment_dt_from_any(row.get(treat_col_meta))
            if animal not in out:
                out[animal] = {"animal_id": animal}
            if out[animal].get("treatment_date") is None:
                out[animal]["treatment_date"] = td

    if verbose:
        n_td = sum(1 for v in out.values() if v.get("treatment_date") is not None)
        print(f"[meta] loaded {len(out)} animals; treatment dates available for {n_td}")
    return out

def levina_bickel_id(
    X: np.ndarray,
    *,
    k: int = 15,
    n_jobs: int = 1,
    point_agg: str = "mean",
    seed: int = 0,
) -> float:
    """
    Compute Levina–Bickel MLE intrinsic dimension estimate for points X.

    Returns NaN if it cannot compute (too few points, degenerate distances).
    """
    n = int(X.shape[0])
    if n <= k:
        return float("nan")

    # nearest neighbors (k+1 to include self at distance 0)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)  # (n, k+1)
    # drop self
    D = dists[:, 1:]  # (n, k)

    eps = 1e-12
    rk = D[:, -1] + eps
    # avoid log(0) for duplicated points
    denom = D[:, :-1] + eps
    logs = np.log((rk[:, None]) / denom)  # (n, k-1)
    s = np.sum(logs, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        m = (k - 1) / s

    m = m[np.isfinite(m)]
    if m.size == 0:
        return float("nan")

    if point_agg == "median":
        return float(np.median(m))
    return float(np.mean(m))


# -----------------------------
# NPZ processing
# -----------------------------
def _choose_defaults(computer_power: str) -> Dict[str, Any]:
    cpu = os.cpu_count() or 8
    if computer_power == "pro":
        return {
            "workers": max(1, min(8, cpu - 1)),
            "n_jobs": max(1, min(8, cpu)),
            "cap": 20000,
        }
    # laptop
    return {
        "workers": 1,
        "n_jobs": 2,
        "cap": 5000,
    }


def _find_npz_files(root_dir: Path, recursive: bool) -> List[Path]:
    """Find candidate bird-level NPZ files.

    Heuristics:
      - Prefer files named like 'USA####.npz'
      - Prefer when file stem matches its parent directory name
      - Skip obvious per-segment NPZs (contain '_segment_')
    """
    root_dir = Path(root_dir)
    paths = list(root_dir.rglob("*.npz")) if recursive else list(root_dir.glob("*.npz"))

    out: List[Path] = []
    for p in paths:
        name = p.name.lower()
        if "_segment_" in name or "segment_" in name:
            continue

        stem = p.stem
        parent = p.parent.name

        if re.fullmatch(r"USA\d+", stem):
            out.append(p)
            continue

        if stem == parent and len(stem) >= 3:
            out.append(p)
            continue

    if not out:
        out = [p for p in paths if "_segment_" not in p.name.lower()]

    return sorted(set(out))

def _load_npz_keys(npz_path: Path, keys: Iterable[str]) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
        else:
            out[k] = None
    return out


def _build_file_datetime_map(
    file_map_obj: Any,
    *,
    year_default: int,
    verbose: bool = False,
) -> Dict[int, _dt.datetime]:
    """
    Returns {file_index:int -> datetime} for those that successfully parse.
    """
    # file_map is often a 0-d object array holding a dict
    fm = file_map_obj
    if isinstance(fm, np.ndarray) and fm.shape == ():
        try:
            fm = fm.item()
        except Exception:
            pass

    mapping: Dict[int, Any] = {}
    if isinstance(fm, dict):
        mapping = {int(k): v for k, v in fm.items()}
    elif isinstance(fm, (list, tuple)):
        mapping = {int(i): v for i, v in enumerate(fm)}
    else:
        raise ValueError(f"Unexpected file_map structure: {type(fm)}")

    dt_map: Dict[int, _dt.datetime] = {}
    for idx, v in mapping.items():
        fp = _unwrap_file_map_value(v)
        dt = parse_datetime_from_filename(fp, year_default=year_default)
        if dt is not None:
            dt_map[int(idx)] = dt

    if verbose:
        ok = len(dt_map)
        total = len(mapping)
        print(f"[dates] file_indices+file_map: parsed {ok}/{total} indices ({(100*ok/max(1,total)):.1f}%)")
        if ok < total:
            # show a few failures
            bad = []
            for idx, v in list(mapping.items())[:50]:
                fp = _unwrap_file_map_value(v)
                if int(idx) not in dt_map:
                    bad.append(fp)
            if bad:
                print("[dates] examples that FAILED parsing (first 10):")
                for b in bad[:10]:
                    print("   ", b)

    return dt_map


def _split_pre_post_masks(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    treatment_date: _dt.datetime,
    *,
    exclude_treatment_day_from_post: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (pre_mask, post_mask) over points.
    Uses file_index -> datetime, then broadcasts to points via integer indexing.
    """
    fi = file_indices.astype(int)
    max_idx = int(fi.max()) if fi.size else 0
    file_dt_arr = np.array([None] * (max_idx + 1), dtype=object)
    for idx, dt in file_dt_map.items():
        if 0 <= idx <= max_idx:
            file_dt_arr[idx] = dt

    # points with missing dt will be excluded
    point_dt = file_dt_arr[fi]  # object array (N,)

    # define boundaries
    t0 = _dt.datetime.combine(treatment_date.date(), _dt.time(0, 0, 0))
    t1 = _dt.datetime.combine(treatment_date.date(), _dt.time(23, 59, 59, 999999))

    pre = np.array([(d is not None and d < t0) for d in point_dt], dtype=bool)

    if exclude_treatment_day_from_post:
        post = np.array([(d is not None and d > t1) for d in point_dt], dtype=bool)
    else:
        post = np.array([(d is not None and d >= t0) for d in point_dt], dtype=bool)

    return pre, post


def process_one_npz(npz_path: Path, meta: Dict[str, Any], cfg: PrePostConfig) -> List[Dict[str, Any]]:
    """
    Process a single NPZ and return rows for the output CSV.
    """
    t_start = time.time()

    animal_id = _infer_animal_id(npz_path)
    hit_type = cfg.override_hit_type if cfg.override_hit_type is not None else meta.get("hit_type", None)
    group = _map_hit_type_to_group(hit_type) if hit_type is not None else meta.get("group", "unknown")

    # treatment date
    td = meta.get("treatment_date", None)
    if cfg.override_treatment_date:
        td = _treatment_dt_from_any(cfg.override_treatment_date)
    if td is None:
        print(f"[skip] {animal_id}: missing treatment date (metadata + override).")
        return []

    # load arrays
    needed = [cfg.array_key, cfg.cluster_key, cfg.file_key, "file_map"]
    if cfg.vocalization_only:
        needed.append(cfg.vocalization_key)

    arrs = _load_npz_keys(npz_path, needed)
    X = arrs[cfg.array_key]
    clusters = arrs[cfg.cluster_key]
    file_indices = arrs[cfg.file_key]
    file_map_obj = arrs["file_map"]

    if X is None or clusters is None or file_indices is None or file_map_obj is None:
        print(f"[skip] {animal_id}: NPZ missing one of required keys: {needed}")
        return []

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        print(f"[skip] {animal_id}: array_key {cfg.array_key} is not 2D array.")
        return []

    N = int(X.shape[0])
    if clusters.shape[0] != N or file_indices.shape[0] != N:
        print(f"[skip] {animal_id}: mismatched array lengths.")
        return []

    # optional vocalization mask
    base_mask = np.ones(N, dtype=bool)
    if cfg.vocalization_only and arrs.get(cfg.vocalization_key, None) is not None:
        voc = arrs[cfg.vocalization_key]
        try:
            base_mask &= (voc.astype(int) == 1)
        except Exception:
            pass

    # parse datetimes from file_map
    year_default = td.year if isinstance(td, _dt.datetime) else 2024
    file_dt_map = _build_file_datetime_map(file_map_obj, year_default=year_default, verbose=cfg.verbose_dates)

    pre_mask, post_mask = _split_pre_post_masks(
        file_indices=file_indices,
        file_dt_map=file_dt_map,
        treatment_date=td,
        exclude_treatment_day_from_post=cfg.exclude_treatment_day_from_post,
    )

    pre_mask &= base_mask
    post_mask &= base_mask

    if cfg.verbose_dates:
        print(f"[split] {animal_id}: pre points={int(pre_mask.sum()):,}  post points={int(post_mask.sum()):,}  (treatment={td.date()})")

    # determine labels to iterate
    labels = np.unique(clusters.astype(int))
    if not cfg.include_noise:
        labels = labels[labels != -1]

    # cap logic
    rng = np.random.default_rng(0)
    cap = cfg.max_points_per_cluster_period

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        lab = int(lab)

        idx_pre = np.where(pre_mask & (clusters.astype(int) == lab))[0]
        idx_post = np.where(post_mask & (clusters.astype(int) == lab))[0]

        if idx_pre.size < max(cfg.min_points_per_period, cfg.k + 1):
            continue
        if idx_post.size < max(cfg.min_points_per_period, cfg.k + 1):
            continue

        # subsample
        if cap is not None:
            if idx_pre.size > cap:
                idx_pre = rng.choice(idx_pre, size=cap, replace=False)
            if idx_post.size > cap:
                idx_post = rng.choice(idx_post, size=cap, replace=False)

        X_pre = X[idx_pre]
        X_post = X[idx_post]

        d_pre = levina_bickel_id(X_pre, k=cfg.k, n_jobs=cfg.n_jobs, point_agg=cfg.point_agg)
        d_post = levina_bickel_id(X_post, k=cfg.k, n_jobs=cfg.n_jobs, point_agg=cfg.point_agg)

        if not (np.isfinite(d_pre) and np.isfinite(d_post)):
            continue

        rows.append({
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "hit_type": hit_type if hit_type is not None else "",
            "group": group,
            "treatment_date": td.date().isoformat(),
            "cluster_label": lab,
            "k": cfg.k,
            "n_pre": int(idx_pre.size),
            "n_post": int(idx_post.size),
            "pre_dim": float(d_pre),
            "post_dim": float(d_post),
            "delta_dim": float(d_post - d_pre),
        })

    dt = time.time() - t_start
    print(f"[pre/post] done: {npz_path.name} in {dt:.2f} s  (clusters={len(rows)})")
    return rows


# -----------------------------
# Plotting
# -----------------------------
_GROUP_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Combined (visible ML + not visible)",
]


def _scatter_pre_post_by_group(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(title, y=0.98)

    have_any = False
    for ax, grp in zip(axes, _GROUP_ORDER):
        sub = df[df["group"] == grp]
        ax.set_title(grp)
        ax.set_xlabel("Pre dimension")
        if ax is axes[0]:
            ax.set_ylabel("Post dimension")
        ax.grid(True, alpha=0.3)
        ax.plot([0, 1], [0, 1], linestyle="--")  # will autoscale after scatter

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
            continue

        have_any = True
        ax.scatter(sub["pre_dim"], sub["post_dim"], s=14, alpha=0.7)

    if have_any:
        # autoscale nicely based on data range
        allv = pd.concat([df["pre_dim"], df["post_dim"]], axis=0).values
        lo = float(np.nanmin(allv))
        hi = float(np.nanmax(allv))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.05 * (hi - lo)
            for ax in axes:
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _box_delta_by_group(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Δ intrinsic dimension (post − pre)")
    ax.grid(True, axis="y", alpha=0.3)

    data = []
    labels = []
    for grp in _GROUP_ORDER:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            data.append([])
        else:
            data.append(sub["delta_dim"].values)
        labels.append(f"{grp}\n(n={len(sub)})")

    if all(len(x) == 0 for x in data):
        ax.text(0.5, 0.5, "No valid cluster estimates to plot.", ha="center", va="center", transform=ax.transAxes)
    else:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        ax.axhline(0, linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Driver
# -----------------------------


def _safe_float_series(x: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _delta_stats_by_group(df: pd.DataFrame, out_txt: Path) -> None:
    """Test Δ distributions across groups and write a summary text file."""
    out_txt = Path(out_txt)

    # Keep only rows with finite delta
    d = df.copy()
    d["delta_dim"] = pd.to_numeric(d.get("delta_dim"), errors="coerce")
    d = d[np.isfinite(d["delta_dim"].to_numpy(dtype=float))]

    groups = sorted(d["group"].dropna().unique().tolist())
    gdata = {g: _safe_float_series(d.loc[d["group"] == g, "delta_dim"]) for g in groups}
    gdata = {g: v for g, v in gdata.items() if v.size > 0}

    lines: List[str] = []
    k_val = int(d["k"].iloc[0]) if ("k" in d.columns and len(d)) else -1
    lines.append(f"Delta intrinsic dimension (post - pre) stats (k={k_val})")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    for g, v in gdata.items():
        lines.append(f"{g}: n={v.size}  mean={float(np.mean(v)):.4f}  median={float(np.median(v)):.4f}")
    lines.append("")
    try:
        from scipy import stats as _scipy_stats  # type: ignore
        have = True
    except Exception:
        have = False

    if len(gdata) >= 2 and have:
        H, p = _scipy_stats.kruskal(*[v for v in gdata.values()])
        lines.append(f"Kruskal-Wallis across groups: H={H:.4f}, p={p:.6g}")
        lines.append("")
        keys = list(gdata.keys())
        ps = []
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                U, p2 = _scipy_stats.mannwhitneyu(gdata[a], gdata[b], alternative='two-sided')
                pairs.append((a, b, float(U), float(p2)))
                ps.append(float(p2))
        # Holm correction
        ps = np.array(ps, dtype=float)
        order = np.argsort(ps)
        m = len(ps)
        adj = np.empty_like(ps)
        for rank, idx in enumerate(order):
            adj[idx] = min(1.0, ps[idx] * (m - rank))
        lines.append("Pairwise Mann–Whitney U (two-sided), Holm-adjusted p-values:")
        for (a, b, U, p2), p_adj in zip(pairs, adj):
            lines.append(f"  {a} vs {b}: U={U:.2f}, p={p2:.6g}, p_holm={p_adj:.6g}")
    else:
        lines.append("Not enough groups with data for statistical tests (or SciPy unavailable).")

    out_txt.write_text("\n".join(lines))


def _scatter_pre_post_colorcoded_by_group(df: pd.DataFrame, out_png: Path, *, title: str, k: int) -> None:
    out_png = Path(out_png)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_xlabel("Pre dimension")
    ax.set_ylabel("Post dimension")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)

    groups = sorted(df["group"].dropna().unique().tolist())
    any_points = False
    for g in groups:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        any_points = True
        ax.scatter(sub["pre_dim"].to_numpy(float), sub["post_dim"].to_numpy(float), s=20, alpha=0.8, label=g)

    if not any_points:
        ax.text(0.5, 0.5, "No points", ha="center", va="center", fontsize=14)
    else:
        ax.legend(frameon=False, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _make_plots_and_stats(df: pd.DataFrame, out_dir: Path, k: int) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If group labels look wrong/missing, rebuild them from hit_type(_norm).
    expected = {
        "sham saline injection",
        "Area X visible (single hit)",
        "Combined (visible ML + not visible)",
    }
    if "group" not in df.columns:
        df = df.copy()
        df["group"] = "unknown"

    present = set(df["group"].dropna().astype(str).unique().tolist())
    if not (present & expected):
        if "hit_type_norm" in df.columns:
            df = df.copy()
            df["group"] = df["hit_type_norm"].apply(lambda x: _map_hit_type_to_group(_normalize_hit_type(x) if pd.notna(x) else ""))
        elif "hit_type" in df.columns:
            df = df.copy()
            df["group"] = df["hit_type"].apply(lambda x: _map_hit_type_to_group(_normalize_hit_type(x) if pd.notna(x) else ""))

    plot_scatter = out_dir / f"pre_vs_post_scatter_by_hit_k{k}.png"
    plot_delta = out_dir / f"delta_dim_by_hit_type_k{k}.png"
    plot_scatter_color = out_dir / f"pre_vs_post_scatter_colorcoded_by_hit_type_k{k}.png"
    stats_txt = out_dir / f"delta_dim_stats_by_hit_type_k{k}.txt"

    _scatter_pre_post_by_group(df, plot_scatter, title=f"Pre vs Post intrinsic dimension by hit type (k={k})")
    _box_delta_by_group(df, plot_delta, title=f"Δ intrinsic dimension (post − pre) by hit type (k={k})")
    _scatter_pre_post_colorcoded_by_group(df, plot_scatter_color, title=f"Pre vs Post intrinsic dimension (color-coded by hit type) (k={k})", k=k)

    try:
        _delta_stats_by_group(df, stats_txt)
    except Exception as e:
        stats_txt.write_text(f"Failed to compute stats: {e}\n")

    return {
        "plot_scatter": str(plot_scatter),
        "plot_delta": str(plot_delta),
        "plot_scatter_color": str(plot_scatter_color),
        "stats_txt": str(stats_txt),
    }

def run_root_directory_pre_post_treatment(cfg: PrePostConfig, *, force_recompute: bool = False, resume: bool = True) -> Dict[str, str]:
    root_dir = Path(cfg.root_dir)
    out_dir = Path(cfg.out_dir) if cfg.out_dir is not None else root_dir / f"lb_pre_post_dimensionality_k{cfg.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = Path(cfg.stats_csv) if cfg.stats_csv is not None else out_dir / f"lb_pre_post_cluster_dimensionality_k{cfg.k}.csv"

    # Build metadata map
    animal_meta = build_animal_metadata_map(
        metadata_xlsx=Path(cfg.metadata_xlsx),
        metadata_sheet=cfg.metadata_sheet,
        verbose=cfg.verbose_dates,
        hit_type_col=cfg.hit_type_col,
        treatment_date_col=cfg.treatment_date_col,
    )

    # Find NPZs
    npz_paths = _find_npz_files(root_dir, recursive=cfg.recursive)

    existing_df: Optional[pd.DataFrame] = None
    processed_npz: set[str] = set()
    if combined_csv.exists() and not force_recompute:
        try:
            existing_df = pd.read_csv(combined_csv)
            if "npz_path" in existing_df.columns:
                processed_npz = set(existing_df["npz_path"].astype(str).unique().tolist())
            print(f"[cache] found existing combined CSV: {combined_csv}  (rows={len(existing_df)})")

            # Refresh hit_type/group columns from the metadata map (useful if an older CSV was generated
            # with a different hit-type column, e.g., 'bilateral' instead of lesion hit type).
            try:
                if "animal_id" in existing_df.columns and len(animal_meta) > 0:
                    meta_df = pd.DataFrame([{
                        "animal_id": k,
                        "hit_type_meta": v.get("hit_type"),
                        "hit_type_norm_meta": v.get("hit_type_norm"),
                        "group_meta": v.get("group"),
                    } for k, v in animal_meta.items()])
                    if len(meta_df) > 0:
                        tmp = existing_df.merge(meta_df, on="animal_id", how="left")
                        for col, mcol in [("hit_type", "hit_type_meta"), ("hit_type_norm", "hit_type_norm_meta"), ("group", "group_meta")]:
                            if mcol in tmp.columns:
                                if col in tmp.columns:
                                    tmp[col] = tmp[mcol].where(tmp[mcol].notna() & (tmp[mcol].astype(str).str.len() > 0), tmp[col])
                                else:
                                    tmp[col] = tmp[mcol]
                        existing_df = tmp.drop(columns=[c for c in ["hit_type_meta", "hit_type_norm_meta", "group_meta"] if c in tmp.columns])
            except Exception:
                pass
        except Exception:
            existing_df = None
            processed_npz = set()

    # If no NPZs (e.g., running on a machine without the data), but CSV exists, just plot.
    if (not npz_paths) and (existing_df is not None):
        plot_paths = _make_plots_and_stats(existing_df, out_dir, cfg.k)
        plot_paths["results_csv"] = str(combined_csv)
        plot_paths["out_dir"] = str(out_dir)
        return plot_paths

    # Determine compute list
    if existing_df is not None and (not resume):
        # Plot only
        plot_paths = _make_plots_and_stats(existing_df, out_dir, cfg.k)
        plot_paths["results_csv"] = str(combined_csv)
        plot_paths["out_dir"] = str(out_dir)
        return plot_paths

    todo = []
    for p in npz_paths:
        if (not force_recompute) and (str(p) in processed_npz):
            continue
        todo.append(p)

    # If nothing new to compute, plot existing
    if existing_df is not None and (not todo):
        plot_paths = _make_plots_and_stats(existing_df, out_dir, cfg.k)
        plot_paths["results_csv"] = str(combined_csv)
        plot_paths["out_dir"] = str(out_dir)
        return plot_paths

    # Decide workers/n_jobs/cap defaults
    cfg_eff = cfg
    workers = cfg.workers
    n_jobs = cfg.n_jobs
    cap = cfg.max_points_per_cluster_period

    if cfg.computer_power.lower() == "pro":
        if workers == 0:
            workers = max(1, (os.cpu_count() or 8) // 2)
        if n_jobs == 0:
            n_jobs = max(1, (os.cpu_count() or 8) // 2)
        if cap is None:
            cap = 20000
    else:
        if workers == 0:
            workers = 1
        if n_jobs == 0:
            n_jobs = 2
        if cap is None:
            cap = 15000

    cfg_eff = PrePostConfig(**{**cfg.__dict__, "workers": workers, "n_jobs": n_jobs, "max_points_per_cluster_period": cap})

    print(f"[pre/post] NPZ files found: {len(npz_paths)}  (todo={len(todo)}, workers={workers}, n_jobs={n_jobs}, cap={cap})")

    all_rows: List[Dict[str, Any]] = []
    if existing_df is not None and resume and not force_recompute:
        all_rows.extend(existing_df.to_dict(orient="records"))

    t0 = time.time()

    def _run_one(p: Path) -> List[Dict[str, Any]]:
        animal_id = _infer_animal_id(p)
        meta = animal_meta.get(animal_id, {"animal_id": animal_id, "treatment_date": None, "hit_type": None, "hit_type_norm": None, "group": "unknown"})
        return process_one_npz(p, meta, cfg_eff)

    if workers <= 1 or len(todo) <= 1:
        for p in todo:
            all_rows.extend(_run_one(p))
    else:
        import concurrent.futures as _fut
        with _fut.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, p): p for p in todo}
            for fu in _fut.as_completed(futs):
                p = futs[fu]
                try:
                    all_rows.extend(fu.result())
                except Exception as e:
                    print(f"[error] {p}: {e}")

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=[
            "animal_id", "hit_type", "hit_type_norm", "group", "npz_path",
            "k", "cluster_label", "n_pre", "n_post", "pre_dim", "post_dim", "delta_dim",
        ])

    # Write ONE combined CSV
    df.to_csv(combined_csv, index=False)
    print(f"[pre/post] TOTAL time: {time.time() - t0:.2f} s")
    print(f"Saved combined CSV: {combined_csv}")

    # Plots + stats
    plot_paths = _make_plots_and_stats(df, out_dir, cfg.k)
    plot_paths["results_csv"] = str(combined_csv)
    plot_paths["out_dir"] = str(out_dir)
    return plot_paths

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Levina_Bickel_pre_vs_post_treatment.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=str, help="Directory containing NPZ(s) (bird folder or parent).")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Excel metadata with treatment dates and hit types.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str,
                   help="Sheet to read hit types from (treatment dates may be pulled from 'metadata' sheet if missing here).")
    p.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <root-dir>/lb_pre_post_dimensionality_k<k>/).")
    p.add_argument("--recursive", action="store_true", help="Recursively search for NPZs under --root-dir.")

    p.add_argument("--array-key", default="predictions", type=str, help="Key in NPZ for high-dim data used for ID.")
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str, help="Key in NPZ for cluster labels.")
    p.add_argument("--file-key", default="file_indices", type=str, help="Key in NPZ for file indices.")
    p.add_argument("--include-noise", action="store_true", help="Include HDBSCAN noise label (-1).")

    p.add_argument("--no-vocalization-only", action="store_true",
                   help="Do NOT restrict to vocalization==1 (if vocalization key exists).")
    p.add_argument("--vocalization-key", default="vocalization", type=str, help="Key in NPZ for vocalization mask.")

    p.add_argument("--k", default=15, type=int, help="k for Levina–Bickel (kNN).")
    p.add_argument("--point-agg", default="mean", choices=["mean", "median"],
                   help="Aggregate local m_i estimates with mean or median.")
    p.add_argument("--n-jobs", default=0, type=int, help="Threads for NearestNeighbors (0 => auto by --computer-power).")
    p.add_argument("--workers", default=0, type=int, help="Processes across NPZ files (0 => auto by --computer-power; 1 => serial).")
    p.add_argument("--min-points-per-period", default=200, type=int, help="Min points required in both pre and post for a cluster.")
    p.add_argument("--exclude-treatment-day-from-post", action="store_true",
                   help="If set, POST starts the day *after* treatment day (treatment day excluded).")
    p.add_argument("--max-points-per-cluster-period", default=None, type=int,
                   help="Subsample cap per cluster per period (None => auto by --computer-power).")

    p.add_argument("--computer-power", default="laptop", choices=["laptop", "pro"],
                   help="Sets defaults for workers/n_jobs/caps when not provided.")

    # debugging / override hooks
    p.add_argument("--treatment-date", default=None, type=str,
                   help="Override treatment date for ALL birds (YYYY-MM-DD or ISO datetime).")
    p.add_argument("--hit-type", default=None, type=str, help="Override hit type for ALL birds (mainly for debugging).")
    p.add_argument("--verbose-dates", action="store_true", help="Print date parsing diagnostics and pre/post counts.")

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = PrePostConfig(
        root_dir=Path(args.root_dir),
        metadata_xlsx=Path(args.metadata_xlsx),
        metadata_sheet=args.metadata_sheet,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        stats_csv=Path(args.stats_csv) if args.stats_csv else None,
        hit_type_col=args.hit_type_col,
        treatment_date_col=args.treatment_date_col,
        recursive=bool(args.recursive),

        array_key=args.array_key,
        cluster_key=args.cluster_key,
        file_key=args.file_key,

        include_noise=bool(args.include_noise),
        vocalization_only=not bool(args.no_vocalization_only),
        vocalization_key=args.vocalization_key,

        k=int(args.k),
        point_agg=str(args.point_agg),
        n_jobs=int(args.n_jobs),
        workers=int(args.workers),

        min_points_per_period=int(args.min_points_per_period),
        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        max_points_per_cluster_period=args.max_points_per_cluster_period,

        computer_power=str(args.computer_power),

        override_treatment_date=args.treatment_date,
        override_hit_type=args.hit_type,

        verbose_dates=bool(args.verbose_dates),
    )

    paths = run_root_directory_pre_post_treatment(cfg, force_recompute=args.force_recompute, resume=(not args.no_resume))
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
