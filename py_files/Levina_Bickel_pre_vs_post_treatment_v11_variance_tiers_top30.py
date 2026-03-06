#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_pre_vs_post_treatment_balanced_early_late.py

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
python Levina_Bickel_pre_vs_post_treatment_balanced_early_late.py \
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

# Optional SciPy for stats on boxplots
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False



# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PrePostConfig:
    root_dir: Path
    metadata_xlsx: Path
    metadata_sheet: str

    out_dir: Optional[Path] = None
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

    # NEW: balance sample sizes within each comparison by subsampling both sides to min(nA, nB)
    balance_pair_samples: bool = True
    random_seed: int = 0

    # NEW: early/late splits within PRE and within POST (based on file datetime)
    split_early_late: bool = True
    early_late_split_method: str = "file_median"  # file_median or file_half

    computer_power: str = "laptop"  # laptop or pro

    # optional overrides
    override_treatment_date: Optional[str] = None  # YYYY-MM-DD
    override_hit_type: Optional[str] = None

    # Optional: variance tiers (e.g., high-variance vs low-variance syllables)
    # If variance_csv is provided, rows/plots will be duplicated for:
    #   - high variance: top `variance_top_pct` percent of syllables within each bird
    #   - low variance: remaining syllables
    variance_csv: Optional[Path] = None
    variance_top_pct: float = 30.0
    variance_animal_col: str = "Animal ID"
    variance_label_col: str = "Syllable"
    variance_value_col: str = "Pre_Variance_IQR_ms2"  # fallback candidates are tried if missing
    variance_group_filter: Optional[str] = "Post"  # only used if variance CSV has a 'Group' column

    verbose_dates: bool = False


# -----------------------------
# Small helpers
# -----------------------------
def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _coerce_int_label(x: Any) -> Optional[int]:
    """
    Best-effort coercion of a label to int.
    Returns None if it cannot be parsed as an int.
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        # allow things like "3.0"
        if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
            return int(float(s))
    except Exception:
        return None
    return None


def load_variance_tier_maps(
    variance_csv: Path,
    *,
    top_pct: float,
    animal_col: str = "Animal ID",
    label_col: str = "Syllable",
    value_col: str = "Pre_Variance_IQR_ms2",
    group_filter: Optional[str] = "Post",
) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], float], Dict[str, int]]:
    """
    Build (animal_id, label)->tier and (animal_id, label)->variance maps from a variance CSV.

    Tiers are assigned PER BIRD:
      - "high": top `top_pct` percent of labels by variance value
      - "low": everything else
      - "unknown": used later if a label is missing from the CSV map

    If a 'Group' column is present and `group_filter` is not None, rows are filtered
    to Group == group_filter before tiering (default "Post", because that row carries
    the Pre_* aggregate stats in your CSV).
    """
    dfv = pd.read_csv(variance_csv)

    # Optional group filter
    if group_filter is not None and "Group" in dfv.columns:
        dfv = dfv[dfv["Group"].astype(str) == str(group_filter)].copy()

    # Resolve columns robustly
    if animal_col not in dfv.columns:
        for c in ["Animal ID", "animal_id", "AnimalID", "bird", "Bird", "bird_id"]:
            if c in dfv.columns:
                animal_col = c
                break
    if label_col not in dfv.columns:
        for c in ["Syllable", "syllable", "label", "Label", "syllable_label", "mapped_syllable"]:
            if c in dfv.columns:
                label_col = c
                break

    # variance value column: use requested, else try candidates
    value_candidates = [value_col, "Pre_Variance_IQR_ms2", "Pre_Variance_ms2", "Variance_ms2", "Std_ms", "Pre_Std_ms"]
    value_col_use = None
    for c in value_candidates:
        if c in dfv.columns:
            value_col_use = c
            break
    if value_col_use is None:
        raise ValueError(f"Could not find a variance column in {variance_csv.name}. Tried: {value_candidates}")

    tier_map: Dict[Tuple[str, int], str] = {}
    val_map: Dict[Tuple[str, int], float] = {}
    counts = {"high": 0, "low": 0, "unknown": 0}

    # Coerce and drop unusable rows
    dfv = dfv[[animal_col, label_col, value_col_use]].copy()
    dfv[animal_col] = dfv[animal_col].astype(str)

    dfv["__label_int__"] = dfv[label_col].apply(_coerce_int_label)
    dfv = dfv[dfv["__label_int__"].notna()].copy()
    dfv["__label_int__"] = dfv["__label_int__"].astype(int)

    dfv["__var__"] = pd.to_numeric(dfv[value_col_use], errors="coerce")

    for aid, sub in dfv.groupby(animal_col):
        vals = sub["__var__"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        q = float(100.0 - float(top_pct))
        q = min(100.0, max(0.0, q))
        thr = float(np.nanpercentile(vals, q))

        for _, r in sub.iterrows():
            lab = int(r["__label_int__"])
            v = float(r["__var__"]) if np.isfinite(r["__var__"]) else float("nan")
            key = (str(aid), lab)
            val_map[key] = v
            if np.isfinite(v) and v >= thr:
                tier_map[key] = "high"
                counts["high"] += 1
            elif np.isfinite(v):
                tier_map[key] = "low"
                counts["low"] += 1

    return tier_map, val_map, counts


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
        return "Lateral hit only"

    # combined group
    if ("medial" in norm and "lateral" in norm) or ("ml" in norm and "visible" in norm):
        return "Medial Lateral hit, combined with large lesion"
    if "notvisible" in norm or ("large" in norm and "notvisible" in norm):
        return "Medial Lateral hit, combined with large lesion"

    # fallback: keep original for debugging
    return str(hit_type)


def build_animal_metadata_map(metadata_xlsx: Path, metadata_sheet: str, *, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Returns {animal_id: {"treatment_date": datetime|None, "hit_type": str|None, "group": str}}
    """
    xlsx = Path(metadata_xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"metadata_xlsx not found: {xlsx}")

    # read the requested sheet (usually hit type summary)
    df_main = pd.read_excel(xlsx, sheet_name=metadata_sheet)

    # try to find animal id col
    animal_col = None
    for c in df_main.columns:
        if str(c).strip().lower() in ("animal id", "animal_id", "animalid", "bird", "bird id"):
            animal_col = c
            break
    if animal_col is None:
        # fallback: first col
        animal_col = df_main.columns[0]

    # hit type column (prefer 'Lesion hit type' over e.g. 'Medial Area X hit type')
    hit_col = None
    cols = list(df_main.columns)

    def _cl(x: object) -> str:
        return str(x).strip().lower()

    # 1) exact match(s)
    for c in cols:
        if _cl(c) == "lesion hit type":
            hit_col = c
            break

    # 2) contains both 'lesion' and 'hit type'
    if hit_col is None:
        for c in cols:
            cl = _cl(c)
            if ("lesion" in cl) and ("hit type" in cl):
                hit_col = c
                break

    # 3) any 'hit type' column, but avoid 'medial' if possible
    if hit_col is None:
        candidates = [c for c in cols if "hit type" in _cl(c)]
        if candidates:
            non_medial = [c for c in candidates if "medial" not in _cl(c)]
            hit_col = (non_medial[0] if non_medial else candidates[0])

    if verbose:
        print(f"[meta] hit_type column: {hit_col!r}")

    # treatment date might be present here, but often it's in "metadata"
    treat_col = None
    for c in df_main.columns:
        if "treatment date" in str(c).lower():
            treat_col = c
            break

    # always also try metadata sheet for treatment dates (more reliable)
    df_meta = None
    try:
        df_meta = pd.read_excel(xlsx, sheet_name="metadata")
    except Exception:
        df_meta = None

    meta_animal_col = None
    meta_treat_col = None
    if df_meta is not None:
        for c in df_meta.columns:
            if str(c).strip().lower() in ("animal id", "animal_id", "animalid", "bird", "bird id"):
                meta_animal_col = c
                break
        for c in df_meta.columns:
            if "treatment date" in str(c).lower():
                meta_treat_col = c
                break

    out: Dict[str, Dict[str, Any]] = {}

    if verbose:
        print(f"[meta] metadata_sheet={metadata_sheet!r}  animal_id_col={animal_col!r}  hit_type_col={hit_col!r}")
        print(f"[meta] metadata sheet 'metadata'  animal_id_col={meta_animal_col!r}  treatment_date_col={meta_treat_col!r}")

    # build hit type mapping from df_main
    for _, row in df_main.iterrows():
        aid = str(row.get(animal_col, "")).strip()
        if not aid:
            continue

        hit = None
        if hit_col is not None:
            hit = row.get(hit_col, None)

        td = None
        if treat_col is not None:
            td = row.get(treat_col, None)

        out[aid] = {
            "hit_type": None if (hit is None or (isinstance(hit, float) and np.isnan(hit))) else str(hit),
            "treatment_date": _treatment_dt_from_any(td),
        }

    # overlay treatment dates from metadata sheet if missing
    if df_meta is not None and meta_animal_col is not None and meta_treat_col is not None:
        # take first non-null per animal (metadata has multiple rows per animal)
        for aid, sub in df_meta.groupby(df_meta[meta_animal_col].astype(str).str.strip()):
            if not aid:
                continue
            td_series = sub[meta_treat_col].dropna()
            td_val = td_series.iloc[0] if len(td_series) else None
            td_dt = _treatment_dt_from_any(td_val)
            if aid not in out:
                out[aid] = {"hit_type": None, "treatment_date": td_dt}
            else:
                if out[aid].get("treatment_date") is None and td_dt is not None:
                    out[aid]["treatment_date"] = td_dt

    # add group label
    for aid, meta in out.items():
        meta["group"] = _map_hit_type_to_group(meta.get("hit_type", None))

    return out


# -----------------------------
# Intrinsic dimension (Levina–Bickel)
# -----------------------------
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
    pat = "**/*.npz" if recursive else "*.npz"
    files = [p for p in root_dir.glob(pat) if p.is_file()]
    # ignore small segment npzs if user points at a directory with both (optional)
    # keep only "top-level bird npz" where filename equals folder name or starts with USA
    return sorted(files)


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

def _balanced_subsample_pair(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    rng: np.random.Generator,
    cap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Balance sample sizes without replacement.

    Target size = min(len(idx_a), len(idx_b), cap(if provided)).
    Returns: (idx_a_sub, idx_b_sub, n_target, n_a_raw, n_b_raw)
    """
    n_a_raw = int(idx_a.size)
    n_b_raw = int(idx_b.size)
    n_target = min(n_a_raw, n_b_raw)
    if cap is not None:
        n_target = min(n_target, int(cap))

    if n_target <= 0:
        return idx_a[:0], idx_b[:0], 0, n_a_raw, n_b_raw

    if idx_a.size > n_target:
        idx_a = rng.choice(idx_a, size=n_target, replace=False)
    if idx_b.size > n_target:
        idx_b = rng.choice(idx_b, size=n_target, replace=False)

    return idx_a, idx_b, int(n_target), n_a_raw, n_b_raw


def _split_early_late_masks_by_file(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    period_mask: np.ndarray,
    *,
    method: str = "file_median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a period mask into early/late by file datetime (point-level masks).

    method:
      - "file_median": early = files with dt <= median(dt), late = dt > median(dt)
      - "file_half":   early = first half of files sorted by dt, late = second half

    If there are <2 dated files in the period, returns (period_mask, all-False).
    """
    fi = file_indices.astype(int)
    files = np.unique(fi[period_mask])

    file_dts: List[Tuple[int, _dt.datetime]] = []
    for f in files:
        dt = file_dt_map.get(int(f), None)
        if dt is not None:
            file_dts.append((int(f), dt))

    if len(file_dts) < 2:
        return period_mask.copy(), np.zeros_like(period_mask, dtype=bool)

    file_dts.sort(key=lambda t: t[1])

    if method == "file_half":
        mid = len(file_dts) // 2
        early_files = {f for f, _ in file_dts[:mid]}
        late_files = {f for f, _ in file_dts[mid:]}
    else:
        ts = np.array([dt.timestamp() for _, dt in file_dts], dtype=float)
        med = float(np.median(ts))
        early_files = {f for (f, dt) in file_dts if dt.timestamp() <= med}
        late_files = {f for (f, dt) in file_dts if dt.timestamp() > med}

        # fallback if ties put everything in early
        if len(late_files) == 0:
            mid = len(file_dts) // 2
            early_files = {f for f, _ in file_dts[:mid]}
            late_files = {f for f, _ in file_dts[mid:]}

    early_mask = period_mask & np.isin(fi, np.fromiter(early_files, dtype=int))
    late_mask = period_mask & np.isin(fi, np.fromiter(late_files, dtype=int))
    return early_mask, late_mask



def process_one_npz(
    npz_path: Path,
    meta: Dict[str, Any],
    cfg: PrePostConfig,
    variance_tier_map: Optional[Dict[Tuple[str, int], str]] = None,
    variance_value_map: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[Dict[str, Any]]:
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
    rng = np.random.default_rng(int(cfg.random_seed))
    cap = cfg.max_points_per_cluster_period

    # Optional early/late splits (point-level masks)
    pre_early_mask = pre_late_mask = None
    post_early_mask = post_late_mask = None
    if cfg.split_early_late:
        pre_early_mask, pre_late_mask = _split_early_late_masks_by_file(
            file_indices=file_indices,
            file_dt_map=file_dt_map,
            period_mask=pre_mask,
            method=cfg.early_late_split_method,
        )
        post_early_mask, post_late_mask = _split_early_late_masks_by_file(
            file_indices=file_indices,
            file_dt_map=file_dt_map,
            period_mask=post_mask,
            method=cfg.early_late_split_method,
        )

    need = max(cfg.min_points_per_period, cfg.k + 1)

    def _compute_pair_row(
        *,
        lab: int,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        period_a: str,
        period_b: str,
        comparison: str,
    ) -> Optional[Dict[str, Any]]:
        idx_a = np.where(mask_a & (clusters.astype(int) == lab))[0]
        idx_b = np.where(mask_b & (clusters.astype(int) == lab))[0]

        if idx_a.size < need or idx_b.size < need:
            return None

        # balance + cap
        if cfg.balance_pair_samples:
            idx_a, idx_b, n_target, n_a_raw, n_b_raw = _balanced_subsample_pair(idx_a, idx_b, rng=rng, cap=cap)
            if n_target < need:
                return None
        else:
            n_a_raw, n_b_raw = int(idx_a.size), int(idx_b.size)
            if cap is not None:
                if idx_a.size > cap:
                    idx_a = rng.choice(idx_a, size=cap, replace=False)
                if idx_b.size > cap:
                    idx_b = rng.choice(idx_b, size=cap, replace=False)

        X_a = X[idx_a]
        X_b = X[idx_b]

        d_a = levina_bickel_id(X_a, k=cfg.k, n_jobs=cfg.n_jobs, point_agg=cfg.point_agg)
        d_b = levina_bickel_id(X_b, k=cfg.k, n_jobs=cfg.n_jobs, point_agg=cfg.point_agg)

        if not (np.isfinite(d_a) and np.isfinite(d_b)):
            return None

        row: Dict[str, Any] = {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "hit_type": hit_type if hit_type is not None else "",
            "group": group,
            "treatment_date": td.date().isoformat(),
            "cluster_label": lab,
            "comparison": comparison,
            "period_a": period_a,
            "period_b": period_b,
            "k": cfg.k,
            "n_a_raw": int(n_a_raw),
            "n_b_raw": int(n_b_raw),
            "n_a": int(idx_a.size),
            "n_b": int(idx_b.size),
            "a_dim": float(d_a),
            "b_dim": float(d_b),
            "delta_dim": float(d_b - d_a),
        }

        # Optional: attach variance tier/value for this (bird, label)
        if variance_tier_map is not None:
            key = (animal_id, int(lab))
            tier = variance_tier_map.get(key, "unknown")
            v = float("nan")
            if variance_value_map is not None and key in variance_value_map:
                try:
                    v = float(variance_value_map[key])
                except Exception:
                    v = float("nan")
            row.update({
                "variance_tier": tier,
                "variance_value": v,
                "variance_top_pct": float(cfg.variance_top_pct),
            })


        # Backwards-compatible columns for the main pre_vs_post comparison
        if comparison == "pre_vs_post":
            row.update({
                "n_pre": int(idx_a.size),
                "n_post": int(idx_b.size),
                "pre_dim": float(d_a),
                "post_dim": float(d_b),
            })

        return row

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        lab = int(lab)

        r = _compute_pair_row(
            lab=lab,
            mask_a=pre_mask,
            mask_b=post_mask,
            period_a="pre",
            period_b="post",
            comparison="pre_vs_post",
        )
        if r is not None:
            rows.append(r)

        if cfg.split_early_late and pre_early_mask is not None and pre_late_mask is not None:
            r = _compute_pair_row(
                lab=lab,
                mask_a=pre_early_mask,
                mask_b=pre_late_mask,
                period_a="pre_early",
                period_b="pre_late",
                comparison="pre_early_vs_late",
            )
            if r is not None:
                rows.append(r)

        if cfg.split_early_late and post_early_mask is not None and post_late_mask is not None:
            r = _compute_pair_row(
                lab=lab,
                mask_a=post_early_mask,
                mask_b=post_late_mask,
                period_a="post_early",
                period_b="post_late",
                comparison="post_early_vs_late",
            )
            if r is not None:
                rows.append(r)

        # NEW: transition comparison across lesion boundary using late PRE vs early POST
        if cfg.split_early_late and pre_late_mask is not None and post_early_mask is not None:
            r = _compute_pair_row(
                lab=lab,
                mask_a=pre_late_mask,
                mask_b=post_early_mask,
                period_a="pre_late",
                period_b="post_early",
                comparison="late_pre_vs_early_post",
            )
            if r is not None:
                rows.append(r)


    dt = time.time() - t_start
    print(f"[pre/post] done: {npz_path.name} in {dt:.2f} s  (rows={len(rows)})")
    return rows


# -----------------------------
# Plotting
# -----------------------------
_GROUP_ORDER = [
    "sham saline injection",
    "Lateral hit only",
    "Medial Lateral hit, combined with large lesion",
]


def _pretty_group_name(grp: str) -> str:
    """Human-friendly label wrapping for plot titles/ticks."""
    if grp == "Medial Lateral hit, combined with large lesion":
        return "Medial Lateral hit,\ncombined with large lesion"
    return grp

def _scatter_pair_by_group(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(title, y=0.98)

    have_any = False
    for ax, grp in zip(axes, _GROUP_ORDER):
        sub = df[df["group"] == grp]
        ax.set_title(_pretty_group_name(grp))
        ax.set_xlabel(x_label)
        if ax is axes[0]:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.plot([0, 1], [0, 1], linestyle="--")  # will autoscale after scatter

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
            continue

        have_any = True
        ax.scatter(sub[x_col], sub[y_col], s=14, alpha=0.7)

    if have_any:
        allv = pd.concat([df[x_col], df[y_col]], axis=0).to_numpy(dtype=float)
        allv = allv[np.isfinite(allv)]
        if allv.size:
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


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (strong FWER control)."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = float(pvals[idx])
        factor = (m - rank)
        p_adj = min(1.0, p * factor)
        running_max = max(running_max, p_adj)  # ensure monotone
        adj[idx] = running_max
    return [float(x) for x in adj]


def _rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann–Whitney U (range [-1, 1])."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return float(1.0 - (2.0 * u) / (n1 * n2))


def _annotate_sig(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    """Draw a bracket between x1 and x2 at height y, with label text."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=9)


def _box_delta_by_group(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
    out_stats_txt: Optional[Path] = None,
    ylabel: str = "Δ intrinsic dimension (B − A)",
    alpha: float = 0.05,
    correction: str = "holm",
) -> None:
    """
    Boxplot of per-cluster Δ(post-pre) split by lesion-hit-type group, plus stats.

    Stats computed on the same values that appear in the boxplot (cluster-level):
      - Omnibus: Kruskal–Wallis (if >=2 groups with n>=2)
      - Pairwise: Mann–Whitney U, Holm-corrected by default (or Bonferroni)

    Also writes a sensitivity analysis collapsing to per-bird mean Δ (bird-level),
    because per-cluster rows are not strictly independent within bird.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    data: List[np.ndarray] = []
    labels: List[str] = []
    grp_vals: Dict[str, np.ndarray] = {}

    for grp in _GROUP_ORDER:
        sub = df[df["group"] == grp]
        vals = sub["delta_dim"].to_numpy(dtype=float) if len(sub) else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]
        grp_vals[grp] = vals
        data.append(vals)
        labels.append(f"{_pretty_group_name(grp)}\n(n={len(vals)})")

    stats_lines: List[str] = []
    stats_lines.append(f"{title}")
    stats_lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    stats_lines.append("")
    stats_lines.append("=== Cluster-level (matches boxplot) ===")
    for grp in _GROUP_ORDER:
        v = grp_vals[grp]
        if len(v) == 0:
            stats_lines.append(f"{grp}: n=0")
        else:
            stats_lines.append(f"{grp}: n={len(v)}  mean={float(np.mean(v)):.6g}  median={float(np.median(v)):.6g}")

    if all(len(x) == 0 for x in data):
        ax.text(0.5, 0.5, "No valid cluster estimates to plot.", ha="center", va="center", transform=ax.transAxes)
        if out_stats_txt is not None:
            out_stats_txt.write_text("\n".join(stats_lines) + "\n\n(No data)\n")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1)

    # --- stats + annotations (cluster-level) ---
    if not _HAVE_SCIPY:
        stats_lines.append("")
        stats_lines.append("SciPy not available: skipping significance tests.")
    else:
        # valid groups for tests: require >=2 points
        valid = [(i, grp, v) for i, (grp, v) in enumerate(grp_vals.items(), start=1) if len(v) >= 2]
        if len(valid) < 2:
            stats_lines.append("")
            stats_lines.append("Not enough data for tests (need >=2 groups with n>=2).")
        else:
            # omnibus
            try:
                H, p_omni = _scipy_stats.kruskal(*[v for _, _, v in valid])
                stats_lines.append("")
                stats_lines.append(f"Kruskal–Wallis across groups: H={float(H):.6g}, p={float(p_omni):.6g}")
            except Exception as e:
                stats_lines.append("")
                stats_lines.append(f"Kruskal–Wallis failed: {e}")

            # pairwise MWU
            from itertools import combinations
            pairs = list(combinations(valid, 2))
            raw_ps: List[float] = []
            results = []
            for (i1, g1, v1), (i2, g2, v2) in pairs:
                try:
                    mwu = _scipy_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    p = float(mwu.pvalue)
                    u = float(mwu.statistic)
                    rbc = _rank_biserial_from_u(u, len(v1), len(v2))
                except Exception:
                    p = float("nan")
                    u = float("nan")
                    rbc = float("nan")
                raw_ps.append(p)
                results.append((i1, g1, len(v1), i2, g2, len(v2), u, p, rbc))

            # multiple-comparisons correction
            m = len(raw_ps)
            if correction.lower().startswith("bonf"):
                adj_ps = [min(1.0, (p * m)) if np.isfinite(p) else float("nan") for p in raw_ps]
                corr_name = "Bonferroni"
            else:
                # default Holm
                adj_ps = _holm_adjust([p if np.isfinite(p) else 1.0 for p in raw_ps])
                # restore NaNs where appropriate
                adj_ps = [float("nan") if not np.isfinite(p) else float(a) for p, a in zip(raw_ps, adj_ps)]
                corr_name = "Holm"

            stats_lines.append("")
            stats_lines.append(f"Pairwise Mann–Whitney U (two-sided), {corr_name}-corrected:")
            for (i1, g1, n1, i2, g2, n2, u, p, rbc), p_adj in zip(results, adj_ps):
                stats_lines.append(
                    f"  {g1} (n={n1}) vs {g2} (n={n2}): U={u:.6g}, p={p:.6g}, p_adj={p_adj:.6g}, rank-biserial={rbc:.4f}"
                )

            # annotate on plot (only the 3 comparisons max, so it stays readable)
            # Use corrected p-values for stars.
            # Determine y-scale
            all_vals = np.concatenate([x for x in data if len(x)], axis=0)
            y_max = float(np.nanmax(all_vals))
            y_min = float(np.nanmin(all_vals))
            y_rng = y_max - y_min
            if not np.isfinite(y_rng) or y_rng <= 0:
                y_rng = 1.0
            base = y_max + 0.05 * y_rng
            step = 0.08 * y_rng
            h = 0.02 * y_rng

            # bracket order: sham-vs-visible, sham-vs-combined, visible-vs-combined (in x positions 1,2,3)
            # Only annotate if both groups had n>=2 and p_adj is finite.
            for k, ((i1, g1, n1, i2, g2, n2, u, p, rbc), p_adj) in enumerate(zip(results, adj_ps)):
                if not np.isfinite(p_adj):
                    continue
                label = f"{_p_to_stars(p_adj)} (p={p_adj:.3g})" if p_adj < 0.1 else f"{_p_to_stars(p_adj)}"
                _annotate_sig(ax, i1, i2, base + k * step, h, label)

            ax.set_ylim(top=base + max(1, len(results)) * step + 0.1 * y_rng)

    # --- sensitivity analysis: bird-level collapsed ---
    stats_lines.append("")
    stats_lines.append("=== Bird-level sensitivity (mean Δ per bird) ===")
    if "animal_id" in df.columns:
        df_b = df.copy()
        df_b = df_b[np.isfinite(df_b["delta_dim"].to_numpy(dtype=float))]
        df_b = df_b.groupby(["animal_id", "group"], as_index=False)["delta_dim"].mean()

        grp_vals_b: Dict[str, np.ndarray] = {}
        for grp in _GROUP_ORDER:
            v = df_b.loc[df_b["group"] == grp, "delta_dim"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            grp_vals_b[grp] = v
            if len(v) == 0:
                stats_lines.append(f"{grp}: n_birds=0")
            else:
                stats_lines.append(f"{grp}: n_birds={len(v)}  mean={float(np.mean(v)):.6g}  median={float(np.median(v)):.6g}")

        if _HAVE_SCIPY:
            valid_b = [(grp, v) for grp, v in grp_vals_b.items() if len(v) >= 2]
            if len(valid_b) >= 2:
                try:
                    H, p_omni = _scipy_stats.kruskal(*[v for _, v in valid_b])
                    stats_lines.append(f"Kruskal–Wallis (bird-level): H={float(H):.6g}, p={float(p_omni):.6g}")
                except Exception as e:
                    stats_lines.append(f"Kruskal–Wallis (bird-level) failed: {e}")

                from itertools import combinations
                pairs_b = list(combinations(valid_b, 2))
                raw_ps_b = []
                for (g1, v1), (g2, v2) in pairs_b:
                    try:
                        mwu = _scipy_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                        raw_ps_b.append(float(mwu.pvalue))
                    except Exception:
                        raw_ps_b.append(float("nan"))

                if correction.lower().startswith("bonf"):
                    adj_b = [min(1.0, p * len(raw_ps_b)) if np.isfinite(p) else float("nan") for p in raw_ps_b]
                    corr_name_b = "Bonferroni"
                else:
                    adj_tmp = _holm_adjust([p if np.isfinite(p) else 1.0 for p in raw_ps_b])
                    adj_b = [float("nan") if not np.isfinite(p) else float(a) for p, a in zip(raw_ps_b, adj_tmp)]
                    corr_name_b = "Holm"

                stats_lines.append(f"Pairwise MWU (bird-level), {corr_name_b}-corrected:")
                for ((g1, v1), (g2, v2)), p, p_adj in zip(pairs_b, raw_ps_b, adj_b):
                    stats_lines.append(f"  {g1} (n={len(v1)}) vs {g2} (n={len(v2)}): p={p:.6g}, p_adj={p_adj:.6g}")
            else:
                stats_lines.append("Not enough birds for tests (need >=2 groups with n_birds>=2).")
        else:
            stats_lines.append("SciPy not available: skipping bird-level tests.")
    else:
        stats_lines.append("No animal_id column found; cannot compute bird-level sensitivity.")

    if out_stats_txt is not None:
        out_stats_txt.write_text("\n".join(stats_lines) + "\n")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _paired_pre_drift_vs_post_drift_birdlevel(
    df_bird: pd.DataFrame,
    out_png: Path,
    title: str,
    out_stats_txt: Optional[Path] = None,
    ylabel: str = "Δ intrinsic dimension (late − early)",
) -> None:
    """
    Paired comparison (WITHIN BIRD) of:
      1) pre drift  = (late pre − early pre)
      2) post drift = (late post − early post)

    Each point/line = one bird (mean across clusters within that bird and comparison).
    Faceted by lesion-hit-type group.

    Requires df_bird to contain comparisons:
      - "pre_early_vs_late"  with delta_dim = late_pre - early_pre
      - "post_early_vs_late" with delta_dim = late_post - early_post
    """
    if "comparison" not in df_bird.columns:
        return

    pre = df_bird[df_bird["comparison"] == "pre_early_vs_late"][["animal_id", "group", "delta_dim"]].copy()
    post = df_bird[df_bird["comparison"] == "post_early_vs_late"][["animal_id", "group", "delta_dim"]].copy()
    pre = pre.rename(columns={"delta_dim": "delta_pre"})
    post = post.rename(columns={"delta_dim": "delta_post"})

    merged = pd.merge(pre, post, on=["animal_id", "group"], how="inner")
    if len(merged) == 0:
        return

    # stats text
    stats_lines: List[str] = []
    stats_lines.append(title)
    stats_lines.append("Paired within-bird comparison: delta_post vs delta_pre")
    stats_lines.append("delta_pre  = (late pre − early pre)")
    stats_lines.append("delta_post = (late post − early post)")
    stats_lines.append("")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle(title, y=0.98)

    all_vals = np.concatenate([
        merged["delta_pre"].to_numpy(dtype=float),
        merged["delta_post"].to_numpy(dtype=float),
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return
    lo = float(np.nanmin(all_vals))
    hi = float(np.nanmax(all_vals))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.1 * (hi - lo)

    x_pre = 1.0
    x_post = 2.0

    for ax, grp in zip(axes, _GROUP_ORDER):
        sub = merged[merged["group"] == grp].copy()
        sub = sub[np.isfinite(sub["delta_pre"].to_numpy(dtype=float)) & np.isfinite(sub["delta_post"].to_numpy(dtype=float))]

        ax.set_title(_pretty_group_name(grp))
        ax.set_xticks([x_pre, x_post])
        ax.set_xticklabels(["pre-lesion", "post-lesion"])
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, linestyle="--", linewidth=1)

        if ax is axes[0]:
            ax.set_ylabel(ylabel)

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No birds", ha="center", va="center", transform=ax.transAxes)
            continue

        y_pre = sub["delta_pre"].to_numpy(dtype=float)
        y_post = sub["delta_post"].to_numpy(dtype=float)

        # boxplots (bird-level distributions)
        bp = ax.boxplot(
            [y_pre, y_post],
            positions=[x_pre, x_post],
            widths=0.5,
            patch_artist=True,
            showfliers=True,
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.6)

        # paired lines/points (one per bird)
        # small deterministic jitter so overlapping points are visible
        n = len(sub)
        jitter = (np.linspace(-0.06, 0.06, n) if n > 1 else np.array([0.0]))
        for j, (yp, yq) in enumerate(zip(y_pre, y_post)):
            ax.plot([x_pre + jitter[j], x_post + jitter[j]], [yp, yq], alpha=0.6, linewidth=1)
            ax.scatter([x_pre + jitter[j], x_post + jitter[j]], [yp, yq], s=18, alpha=0.85)

        # paired test within group (post drift vs pre drift)
        p_val = float("nan")
        if _HAVE_SCIPY and len(y_pre) >= 3:
            try:
                # Wilcoxon signed-rank on paired differences
                d = y_post - y_pre
                # If all diffs are zero, wilcoxon can fail; guard it.
                if np.any(np.abs(d) > 0):
                    w = _scipy_stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                    p_val = float(w.pvalue)
            except Exception:
                p_val = float("nan")

        stats_lines.append(f"{grp}: n_birds={len(y_pre)}  mean_pre={float(np.mean(y_pre)):.6g}  mean_post={float(np.mean(y_post)):.6g}  mean_diff(post-pre)={float(np.mean(y_post - y_pre)):.6g}  paired_p={p_val:.6g}")

        # annotate p-value on the plot (simple text)
        if np.isfinite(p_val):
            label = f"{_p_to_stars(p_val)} (p={p_val:.3g})"
        else:
            label = "paired test n/a"
        ax.text(0.5, 0.95, label, ha="center", va="top", transform=ax.transAxes)

        ax.set_ylim(lo - pad, hi + pad)

    if out_stats_txt is not None:
        out_stats_txt.write_text("\n".join(stats_lines) + "\n")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Driver
# -----------------------------



def _box_delta_by_group_birdlevel(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
    out_stats_txt: Optional[Path] = None,
    ylabel: str = "Δ intrinsic dimension (B − A)",
    alpha: float = 0.05,
    correction: str = "holm",
) -> None:
    """
    Bird-level boxplot of Δ split by lesion-hit-type group.

    Each point is ONE bird (Δ is the bird-mean across clusters for a given comparison).
    This avoids pseudo-replication from treating multiple clusters within a bird as independent.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    data: List[np.ndarray] = []
    tick_labels: List[str] = []
    grp_vals: Dict[str, np.ndarray] = {}

    for grp in _GROUP_ORDER:
        sub = df[df["group"] == grp]
        vals = sub["delta_dim"].to_numpy(dtype=float) if len(sub) else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]
        grp_vals[grp] = vals
        data.append(vals)
        tick_labels.append(f"{_pretty_group_name(grp)}\n(n={len(vals)} birds)")

    stats_lines: List[str] = []
    stats_lines.append(f"{title}")
    stats_lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    stats_lines.append("")
    stats_lines.append("=== Bird-level (each point = 1 bird) ===")
    for grp in _GROUP_ORDER:
        v = grp_vals[grp]
        if len(v) == 0:
            stats_lines.append(f"{grp}: n=0")
        else:
            stats_lines.append(f"{grp}: n={len(v)}  mean={float(np.mean(v)):.6g}  median={float(np.median(v)):.6g}")

    if all(len(x) == 0 for x in data):
        ax.text(0.5, 0.5, "No valid bird estimates to plot.", ha="center", va="center", transform=ax.transAxes)
        if out_stats_txt is not None:
            out_stats_txt.write_text("\n".join(stats_lines) + "\n\n(No data)\n")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1)

    if not _HAVE_SCIPY:
        stats_lines.append("")
        stats_lines.append("SciPy not available: skipping significance tests.")
    else:
        valid = [(i, grp, v) for i, (grp, v) in enumerate(grp_vals.items(), start=1) if len(v) >= 2]
        if len(valid) < 2:
            stats_lines.append("")
            stats_lines.append("Not enough data for tests (need >=2 groups with n>=2).")
        else:
            try:
                H, p_omni = _scipy_stats.kruskal(*[v for _, _, v in valid])
                stats_lines.append("")
                stats_lines.append(f"Kruskal–Wallis across groups: H={float(H):.6g}, p={float(p_omni):.6g}")
            except Exception as e:
                stats_lines.append("")
                stats_lines.append(f"Kruskal–Wallis failed: {e}")

            from itertools import combinations
            pairs = list(combinations(valid, 2))
            raw_ps: List[float] = []
            results = []
            for (i1, g1, v1), (i2, g2, v2) in pairs:
                try:
                    mwu = _scipy_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    p = float(mwu.pvalue)
                    u = float(mwu.statistic)
                    rbc = _rank_biserial_from_u(u, len(v1), len(v2))
                except Exception:
                    p = float("nan")
                    u = float("nan")
                    rbc = float("nan")
                raw_ps.append(p)
                results.append((i1, g1, len(v1), i2, g2, len(v2), u, p, rbc))

            m = len(raw_ps)
            if correction.lower().startswith("bonf"):
                adj_ps = [min(1.0, (p * m)) if np.isfinite(p) else float("nan") for p in raw_ps]
                corr_name = "Bonferroni"
            else:
                adj_ps = _holm_adjust([p if np.isfinite(p) else 1.0 for p in raw_ps])
                adj_ps = [float("nan") if not np.isfinite(p) else float(a) for p, a in zip(raw_ps, adj_ps)]
                corr_name = "Holm"

            stats_lines.append("")
            stats_lines.append(f"Pairwise Mann–Whitney U (two-sided), {corr_name}-corrected:")
            for (i1, g1, n1, i2, g2, n2, u, p, rbc), p_adj in zip(results, adj_ps):
                stats_lines.append(
                    f"  {g1} (n={n1}) vs {g2} (n={n2}): U={u:.6g}, p={p:.6g}, p_adj={p_adj:.6g}, rank-biserial={rbc:.4f}"
                )

            all_vals = np.concatenate([x for x in data if len(x)], axis=0)
            y_max = float(np.nanmax(all_vals))
            y_min = float(np.nanmin(all_vals))
            y_rng = y_max - y_min
            if not np.isfinite(y_rng) or y_rng <= 0:
                y_rng = 1.0
            base = y_max + 0.05 * y_rng
            step = 0.08 * y_rng
            h = 0.02 * y_rng

            for k, ((i1, g1, n1, i2, g2, n2, u, p, rbc), p_adj) in enumerate(zip(results, adj_ps)):
                if not np.isfinite(p_adj):
                    continue
                label = f"{_p_to_stars(p_adj)} (p={p_adj:.3g})" if p_adj < 0.1 else f"{_p_to_stars(p_adj)}"
                _annotate_sig(ax, i1, i2, base + k * step, h, label)

            ax.set_ylim(top=base + max(1, len(results)) * step + 0.1 * y_rng)

    if out_stats_txt is not None:
        out_stats_txt.write_text("\n".join(stats_lines) + "\n")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def run_root_directory_pre_post_treatment(cfg: PrePostConfig) -> Dict[str, str]:
    defaults = _choose_defaults(cfg.computer_power)

    workers = cfg.workers if cfg.workers and cfg.workers > 0 else defaults["workers"]
    n_jobs = cfg.n_jobs if cfg.n_jobs and cfg.n_jobs > 0 else defaults["n_jobs"]
    cap = cfg.max_points_per_cluster_period if cfg.max_points_per_cluster_period is not None else defaults["cap"]

    # avoid oversubscription: if multiple processes, reduce threads
    if workers > 1 and n_jobs > 1:
        n_jobs = max(1, int(math.floor(n_jobs / workers)))

    # update an effective cfg (dataclass is frozen, so recreate)
    cfg_eff = PrePostConfig(**{**cfg.__dict__, "workers": workers, "n_jobs": n_jobs, "max_points_per_cluster_period": cap})

    root_dir = cfg_eff.root_dir
    out_dir = cfg_eff.out_dir if cfg_eff.out_dir is not None else (root_dir / f"lb_balanced_early_late_dimensionality_k{cfg_eff.k}")
    _safe_mkdir(out_dir)

    # locate NPZs
    npz_files = _find_npz_files(root_dir, cfg_eff.recursive)
    if len(npz_files) == 0:
        print(f"[warn] No NPZ files found under: {root_dir}")
    # load metadata
    animal_meta = build_animal_metadata_map(cfg_eff.metadata_xlsx, cfg_eff.metadata_sheet, verbose=cfg_eff.verbose_dates)

    # Optional: load variance tiers for high/low-variance stratification
    variance_tier_map: Optional[Dict[Tuple[str, int], str]] = None
    variance_value_map: Optional[Dict[Tuple[str, int], float]] = None
    if cfg_eff.variance_csv is not None:
        try:
            variance_tier_map, variance_value_map, counts = load_variance_tier_maps(
                cfg_eff.variance_csv,
                top_pct=cfg_eff.variance_top_pct,
                animal_col=cfg_eff.variance_animal_col,
                label_col=cfg_eff.variance_label_col,
                value_col=cfg_eff.variance_value_col,
                group_filter=cfg_eff.variance_group_filter,
            )
            print(f"[variance tiers] top_pct={cfg_eff.variance_top_pct:.1f}% -> counts: {counts}")
        except Exception as e:
            print("[warn] Failed to load variance CSV / tiers:", repr(e))
            variance_tier_map = None
            variance_value_map = None


    # decide which NPZs to process
    tasks: List[Tuple[Path, Dict[str, Any]]] = []
    for p in npz_files:
        aid = _infer_animal_id(p)
        meta = animal_meta.get(aid, {"hit_type": None, "treatment_date": None, "group": "unknown"})
        # if treatment date missing and no override, skip later in process_one_npz (but count here)
        tasks.append((p, meta))

    print(f"[pre/post] NPZ files found: {len(npz_files)}  (workers={cfg_eff.workers}, n_jobs={cfg_eff.n_jobs}, cap={cfg_eff.max_points_per_cluster_period})")

    t0 = time.time()
    all_rows: List[Dict[str, Any]] = []

    if cfg_eff.workers <= 1 or len(tasks) <= 1:
        for p, meta in tasks:
            all_rows.extend(process_one_npz(p, meta, cfg_eff, variance_tier_map, variance_value_map))
    else:
        # multiprocessing across NPZs
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # NOTE: cfg_eff must be picklable; dataclass is fine.
        with ProcessPoolExecutor(max_workers=cfg_eff.workers) as ex:
            futs = [ex.submit(process_one_npz, p, meta, cfg_eff, variance_tier_map, variance_value_map) for p, meta in tasks]
            for fut in as_completed(futs):
                try:
                    rows = fut.result()
                except Exception as e:
                    print("[error] worker failed:", repr(e))
                    continue
                all_rows.extend(rows)

    total_dt = time.time() - t0
    print(f"[pre/post] TOTAL time: {total_dt:.2f} s")

    # write CSV
    stamp = _now_stamp()
    out_csv = out_dir / f"lb_balanced_early_late_cluster_dimensionality_k{cfg_eff.k}_{stamp}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved results CSV: {out_csv}")

    # plots (per comparison)
    paths: Dict[str, str] = {}

    if len(df) == 0:
        # make placeholder empties (helps pipeline)
        plot_scatter = out_dir / f"pre_vs_post_scatter_by_hit_k{cfg_eff.k}_{stamp}.png"
        plot_delta = out_dir / f"pre_vs_post_delta_by_hit_type_k{cfg_eff.k}_{stamp}.png"
        stats_txt = out_dir / f"pre_vs_post_delta_stats_k{cfg_eff.k}_{stamp}.txt"

        _scatter_pair_by_group(
            df,
            plot_scatter,
            title=f"Pre vs Post intrinsic dimension by hit type (k={cfg_eff.k})",
            x_col="pre_dim" if "pre_dim" in df.columns else "a_dim",
            y_col="post_dim" if "post_dim" in df.columns else "b_dim",
            x_label="Pre dimension",
            y_label="Post dimension",
        )
        _box_delta_by_group(
            df,
            plot_delta,
            title=f"Δ intrinsic dimension (post − pre) by hit type (k={cfg_eff.k})",
            out_stats_txt=stats_txt,
            ylabel="Δ intrinsic dimension (post − pre)",
        )
        paths.update({
            "pre_vs_post_plot_delta": str(plot_delta),
            "pre_vs_post_plot_scatter": str(plot_scatter),
            "pre_vs_post_delta_stats": str(stats_txt),
        })
        print(f"Saved plot (empty): {plot_delta}")
        print(f"Saved plot (empty): {plot_scatter}")
    else:
        # Bird-level summary: collapse cluster rows to one row per bird per comparison
        # (each point = mean across clusters within that bird)

        def _compute_df_bird(dfin: pd.DataFrame) -> pd.DataFrame:
            return (
                dfin.groupby(["comparison", "animal_id", "group", "hit_type", "treatment_date"], as_index=False)
                   .agg(
                       a_dim=("a_dim", "mean"),
                       b_dim=("b_dim", "mean"),
                       delta_dim=("delta_dim", "mean"),
                       n_clusters=("cluster_label", "nunique"),
                   )
            )

        def _with_suffix(stem: str, suffix: str) -> str:
            return f"{stem}_{suffix}" if suffix else stem

        def _make_plots_for(dfin: pd.DataFrame, suffix: str = "", tier_descr: str = "") -> None:
            if len(dfin) == 0:
                return

            df_b = _compute_df_bird(dfin)

            def _plot_one_for(
                comp: str,
                x_label: str,
                y_label: str,
                delta_ylabel: str,
            ) -> None:
                sub = dfin[dfin["comparison"] == comp] if "comparison" in dfin.columns else dfin
                if len(sub) == 0:
                    return

                plot_scatter = out_dir / f"{_with_suffix(comp + '_scatter_by_hit', suffix)}_k{cfg_eff.k}_{stamp}.png"
                plot_delta = out_dir / f"{_with_suffix(comp + '_delta_by_hit_type', suffix)}_k{cfg_eff.k}_{stamp}.png"
                stats_txt = out_dir / f"{_with_suffix(comp + '_delta_stats', suffix)}_k{cfg_eff.k}_{stamp}.txt"

                title_extra = f" ({tier_descr})" if tier_descr else ""
                _scatter_pair_by_group(
                    sub,
                    plot_scatter,
                    title=f"{comp} intrinsic dimension by hit type{title_extra} (k={cfg_eff.k})",
                    x_col="a_dim",
                    y_col="b_dim",
                    x_label=x_label,
                    y_label=y_label,
                )
                _box_delta_by_group(
                    sub,
                    plot_delta,
                    title=f"{delta_ylabel} by hit type{title_extra} (k={cfg_eff.k})",
                    out_stats_txt=stats_txt,
                    ylabel=delta_ylabel,
                )

                paths.update({
                    f"{_with_suffix(comp, suffix)}_plot_delta": str(plot_delta),
                    f"{_with_suffix(comp, suffix)}_plot_scatter": str(plot_scatter),
                    f"{_with_suffix(comp, suffix)}_delta_stats": str(stats_txt),
                })

                # Bird-level plots (each point = 1 bird; mean across clusters)
                sub_bird = df_b[df_b["comparison"] == comp]
                if len(sub_bird) > 0:
                    plot_scatter_bird = out_dir / f"{_with_suffix(comp + '_scatter_by_hit_birdlevel', suffix)}_k{cfg_eff.k}_{stamp}.png"
                    plot_delta_bird = out_dir / f"{_with_suffix(comp + '_delta_by_hit_type_birdlevel', suffix)}_k{cfg_eff.k}_{stamp}.png"
                    stats_txt_bird = out_dir / f"{_with_suffix(comp + '_delta_stats_birdlevel', suffix)}_k{cfg_eff.k}_{stamp}.txt"

                    _scatter_pair_by_group(
                        sub_bird,
                        plot_scatter_bird,
                        title=f"{comp} intrinsic dimension by hit type (bird means){title_extra} (k={cfg_eff.k})",
                        x_col="a_dim",
                        y_col="b_dim",
                        x_label=x_label,
                        y_label=y_label,
                    )
                    _box_delta_by_group_birdlevel(
                        sub_bird,
                        plot_delta_bird,
                        title=f"{delta_ylabel} by hit type (bird means){title_extra} (k={cfg_eff.k})",
                        out_stats_txt=stats_txt_bird,
                        ylabel=delta_ylabel,
                    )

                    paths.update({
                        f"{_with_suffix(comp, suffix)}_plot_delta_birdlevel": str(plot_delta_bird),
                        f"{_with_suffix(comp, suffix)}_plot_scatter_birdlevel": str(plot_scatter_bird),
                        f"{_with_suffix(comp, suffix)}_delta_stats_birdlevel": str(stats_txt_bird),
                    })

            _plot_one_for("pre_vs_post", "Pre dimension", "Post dimension", "Δ intrinsic dimension (post − pre)")
            _plot_one_for("pre_early_vs_late", "Early pre dimension", "Late pre dimension", "Δ intrinsic dimension (late pre − early pre)")
            _plot_one_for("post_early_vs_late", "Early post dimension", "Late post dimension", "Δ intrinsic dimension (late post − early post)")
            _plot_one_for("late_pre_vs_early_post", "Late pre dimension", "Early post dimension", "Δ intrinsic dimension (early post − late pre)")

            # Paired within-bird comparison of drift (pre vs post)
            paired_png = out_dir / f"{_with_suffix('paired_pre_drift_vs_post_drift_by_hit_birdlevel', suffix)}_k{cfg_eff.k}_{stamp}.png"
            paired_txt = out_dir / f"{_with_suffix('paired_pre_drift_vs_post_drift_stats_birdlevel', suffix)}_k{cfg_eff.k}_{stamp}.txt"
            title_extra = f" ({tier_descr})" if tier_descr else ""
            _paired_pre_drift_vs_post_drift_birdlevel(
                df_b,
                paired_png,
                title=f"Paired drift comparison within birds by hit type{title_extra} (k={cfg_eff.k})",
                out_stats_txt=paired_txt,
                ylabel="Δ intrinsic dimension (late − early)",
            )
            paths.update({
                f"{_with_suffix('paired_pre_drift_vs_post_drift_plot_birdlevel', suffix)}": str(paired_png),
                f"{_with_suffix('paired_pre_drift_vs_post_drift_stats_birdlevel', suffix)}": str(paired_txt),
            })

        _make_plots_for(df, suffix="", tier_descr="")

        if "variance_tier" in df.columns:
            high = df[df["variance_tier"] == "high"].copy()
            low = df[df["variance_tier"] == "low"].copy()
            if len(high) > 0:
                _make_plots_for(high, suffix=f"highvar_top{int(cfg_eff.variance_top_pct)}", tier_descr=f"high variance (top {cfg_eff.variance_top_pct:.0f}%)")
            if len(low) > 0:
                _make_plots_for(low, suffix="lowvar_rest", tier_descr="low variance (rest)")




    return {
        "results_csv": str(out_csv),
        **paths,
        "out_dir": str(out_dir),
    }


# -----------------------------
# CLI
# -----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Levina_Bickel_pre_vs_post_treatment_balanced_early_late.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=str, help="Directory containing NPZ(s) (bird folder or parent).")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Excel metadata with treatment dates and hit types.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str,
                   help="Sheet to read hit types from (treatment dates may be pulled from 'metadata' sheet if missing here).")
    p.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <root-dir>/lb_balanced_early_late_dimensionality_k<k>/).")
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

    p.add_argument("--no-balance-pairs", action="store_true",
                   help="Do NOT balance sample sizes to min(nA, nB) within each comparison.")
    p.add_argument("--no-split-early-late", action="store_true",
                   help="Do NOT compute early/late comparisons within PRE and within POST.")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"],
                   help="How to split early vs late within a period (by file datetime).")
    p.add_argument("--random-seed", default=0, type=int, help="Random seed for subsampling/balancing.")

    p.add_argument("--computer-power", default="laptop", choices=["laptop", "pro"],
                   help="Sets defaults for workers/n_jobs/caps when not provided.")


    # Optional: variance tiers (high-variance vs low-variance syllables)
    p.add_argument("--variance-csv", default=None, type=str,
                   help="CSV with per-bird per-label variance (e.g., usage_balanced_phrase_duration_stats.csv). "
                        "If provided, the script will also output a second set of plots for high-variance (top --variance-top-pct) "
                        "and low-variance (remaining) labels within each bird.")
    p.add_argument("--variance-top-pct", default=30.0, type=float,
                   help="Percent (per bird) of labels considered 'high variance'.")
    p.add_argument("--variance-animal-col", default="Animal ID", type=str, help="Animal ID column name in --variance-csv.")
    p.add_argument("--variance-label-col", default="Syllable", type=str, help="Label/syllable column name in --variance-csv.")
    p.add_argument("--variance-value-col", default="Pre_Variance_IQR_ms2", type=str,
                   help="Variance column name in --variance-csv (fallbacks are tried if missing).")
    p.add_argument("--variance-group-filter", default="Post", type=str,
                   help="If the variance CSV has a 'Group' column, only rows with this group are used to define tiers. "
                        "Set to empty string '' to disable filtering.")

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

        variance_csv=Path(args.variance_csv) if args.variance_csv else None,
        variance_top_pct=float(args.variance_top_pct),
        variance_animal_col=str(args.variance_animal_col),
        variance_label_col=str(args.variance_label_col),
        variance_value_col=str(args.variance_value_col),
        variance_group_filter=(str(args.variance_group_filter) if str(args.variance_group_filter).strip() != "" else None),

        override_treatment_date=args.treatment_date,
        override_hit_type=args.hit_type,

        verbose_dates=bool(args.verbose_dates),
    )

    paths = run_root_directory_pre_post_treatment(cfg)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
