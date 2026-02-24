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
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.axhline(0, linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Driver
# -----------------------------
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
    out_dir = cfg_eff.out_dir if cfg_eff.out_dir is not None else (root_dir / f"lb_pre_post_dimensionality_k{cfg_eff.k}")
    _safe_mkdir(out_dir)

    # locate NPZs
    npz_files = _find_npz_files(root_dir, cfg_eff.recursive)
    if len(npz_files) == 0:
        print(f"[warn] No NPZ files found under: {root_dir}")
    # load metadata
    animal_meta = build_animal_metadata_map(cfg_eff.metadata_xlsx, cfg_eff.metadata_sheet, verbose=cfg_eff.verbose_dates)

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
            all_rows.extend(process_one_npz(p, meta, cfg_eff))
    else:
        # multiprocessing across NPZs
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # NOTE: cfg_eff must be picklable; dataclass is fine.
        with ProcessPoolExecutor(max_workers=cfg_eff.workers) as ex:
            futs = [ex.submit(process_one_npz, p, meta, cfg_eff) for p, meta in tasks]
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
    out_csv = out_dir / f"lb_pre_post_cluster_dimensionality_k{cfg_eff.k}_{stamp}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved results CSV: {out_csv}")

    # plots
    plot_scatter = out_dir / f"pre_vs_post_scatter_by_hit_k{cfg_eff.k}_{stamp}.png"
    plot_delta = out_dir / f"delta_dim_by_hit_type_k{cfg_eff.k}_{stamp}.png"

    if len(df) == 0:
        # make placeholder empties (helps pipeline)
        _scatter_pre_post_by_group(df, plot_scatter, title=f"Pre vs Post intrinsic dimension by hit type (k={cfg_eff.k})")
        _box_delta_by_group(df, plot_delta, title=f"Δ intrinsic dimension (post − pre) by hit type (k={cfg_eff.k})")
        print(f"Saved plot (empty): {plot_delta}")
        print(f"Saved plot (empty): {plot_scatter}")
    else:
        _scatter_pre_post_by_group(df, plot_scatter, title=f"Pre vs Post intrinsic dimension by hit type (k={cfg_eff.k})")
        _box_delta_by_group(df, plot_delta, title=f"Δ intrinsic dimension (post − pre) by hit type (k={cfg_eff.k})")
        print(f"Saved plot: {plot_delta}")
        print(f"Saved plot: {plot_scatter}")

    return {
        "results_csv": str(out_csv),
        "plot_delta": str(plot_delta),
        "plot_scatter": str(plot_scatter),
        "out_dir": str(out_dir),
    }


# -----------------------------
# CLI
# -----------------------------
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

    paths = run_root_directory_pre_post_treatment(cfg)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
