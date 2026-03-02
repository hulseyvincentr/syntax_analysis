#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bhattacharyya_pre_vs_post_cluster.py

Compute Bhattacharyya coefficient (BC) between PRE vs POST points *within the same cluster*
(per bird NPZ), using a file_indices + file_map pre/post split.

BC is computed under a multivariate Gaussian approximation:
  - Fit N(mu_pre, Sigma_pre) and N(mu_post, Sigma_post)
  - Bhattacharyya distance:
        DB = 1/8 (dmu^T Sigma^-1 dmu) + 1/2 ln( det(Sigma) / sqrt(det(Sigma_pre) det(Sigma_post)) )
        where Sigma = (Sigma_pre + Sigma_post)/2
  - Bhattacharyya coefficient:
        BC = exp(-DB)

Outputs:
  - CSV with one row per (animal_id, cluster_label) that has enough points in PRE and POST.

Example:
  python bhattacharyya_pre_vs_post_cluster.py \
    --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
    --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
    --metadata-sheet "animal_hit_type_summary" \
    --array-key "predictions" \
    --cluster-key "hdbscan_labels" \
    --file-key "file_indices" \
    --cov-type diag \
    --min-points-per-period 500 \
    --computer-power pro
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import argparse
import datetime as _dt
import math
import os
import re
import time

import numpy as np
import pandas as pd

# Optional shrinkage covariance
try:
    from sklearn.covariance import LedoitWolf  # type: ignore
    _HAVE_LW = True
except Exception:
    _HAVE_LW = False


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PrePostBCConfig:
    root_dir: Path
    metadata_xlsx: Path
    metadata_sheet: str

    out_dir: Optional[Path] = None
    recursive: bool = False

    array_key: str = "predictions"      # or "embedding_outputs" for UMAP-space overlap
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"

    include_noise: bool = False
    vocalization_only: bool = True
    vocalization_key: str = "vocalization"

    min_points_per_period: int = 200
    max_points_per_cluster_period: Optional[int] = None  # subsample cap per (cluster, period)

    cov_type: str = "diag"              # diag | full | ledoitwolf
    eps: float = 1e-6                   # covariance regularization

    workers: int = 0
    computer_power: str = "laptop"      # laptop | pro

    exclude_treatment_day_from_post: bool = False

    override_treatment_date: Optional[str] = None  # YYYY-MM-DD or ISO datetime
    override_hit_type: Optional[str] = None
    verbose_dates: bool = False


# -----------------------------
# Small helpers
# -----------------------------
_EXCEL_ORIGIN = _dt.datetime(1899, 12, 30)

def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]

def _unwrap_file_map_value(v: Any) -> str:
    x = v
    for _ in range(4):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        else:
            break
    return str(x)

def _parse_excel_serial_days(token: str) -> Optional[_dt.datetime]:
    try:
        x = float(token)
    except Exception:
        return None
    # Typical Excel serial day ranges for recordings (loose bounds)
    if not (10000.0 <= x <= 80000.0):
        return None
    try:
        return _EXCEL_ORIGIN + _dt.timedelta(days=float(x))
    except Exception:
        return None

def parse_datetime_from_filename(file_path: str | Path, *, year_default: int = 2024) -> Optional[_dt.datetime]:
    """
    Tries a couple filename conventions:
      1) <bird>_<excel_serial_days>_...  => uses Excel origin
      2) <bird>_<something>_<month>_<day>_<hour>_<min>_<sec>_... and optionally a 4-digit year token
    """
    base = Path(file_path).name
    parts = base.split("_")

    if len(parts) >= 2:
        dt = _parse_excel_serial_days(parts[1])
        if dt is not None:
            return dt

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
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
    except Exception:
        pass
    try:
        return _dt.datetime.fromisoformat(str(value))
    except Exception:
        return None


# -----------------------------
# Metadata loading
# -----------------------------
def _normalize_hit_type(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _map_hit_type_to_group(hit_type: str) -> str:
    if hit_type is None:
        return "unknown"
    norm = _normalize_hit_type(str(hit_type))
    if "sham" in norm:
        return "sham saline injection"
    if "singlehit" in norm:
        return "Area X visible (single hit)"
    if ("medial" in norm and "lateral" in norm) or ("ml" in norm and "visible" in norm):
        return "Combined (visible ML + not visible)"
    if "notvisible" in norm or ("large" in norm and "notvisible" in norm):
        return "Combined (visible ML + not visible)"
    return str(hit_type)

def build_animal_metadata_map(metadata_xlsx: Path, metadata_sheet: str) -> Dict[str, Dict[str, Any]]:
    xlsx = Path(metadata_xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"metadata_xlsx not found: {xlsx}")

    df_main = pd.read_excel(xlsx, sheet_name=metadata_sheet)

    # Find animal ID column
    animal_col = None
    for c in df_main.columns:
        if str(c).strip().lower() in ("animal id", "animal_id", "animalid", "bird", "bird id"):
            animal_col = c
            break
    if animal_col is None:
        animal_col = df_main.columns[0]

    # Find lesion hit type column
    hit_col = None
    for c in df_main.columns:
        if str(c).strip().lower() == "lesion hit type":
            hit_col = c
            break
    if hit_col is None:
        for c in df_main.columns:
            cl = str(c).strip().lower()
            if ("lesion" in cl) and ("hit type" in cl):
                hit_col = c
                break
    if hit_col is None:
        candidates = [c for c in df_main.columns if "hit type" in str(c).lower()]
        if candidates:
            non_medial = [c for c in candidates if "medial" not in str(c).lower()]
            hit_col = non_medial[0] if non_medial else candidates[0]

    # Find treatment date column (if present here)
    treat_col = None
    for c in df_main.columns:
        if "treatment date" in str(c).lower():
            treat_col = c
            break

    # Also try to read treatment date from sheet named "metadata" if available
    df_meta = None
    try:
        df_meta = pd.read_excel(xlsx, sheet_name="metadata")
    except Exception:
        df_meta = None

    meta_animal_col, meta_treat_col = None, None
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
    for _, row in df_main.iterrows():
        aid = str(row.get(animal_col, "")).strip()
        if not aid:
            continue
        hit = row.get(hit_col, None) if hit_col is not None else None
        td = row.get(treat_col, None) if treat_col is not None else None
        out[aid] = {
            "hit_type": None if (hit is None or (isinstance(hit, float) and np.isnan(hit))) else str(hit),
            "treatment_date": _treatment_dt_from_any(td),
        }

    if df_meta is not None and meta_animal_col is not None and meta_treat_col is not None:
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

    for aid, meta in out.items():
        meta["group"] = _map_hit_type_to_group(meta.get("hit_type", None))

    return out


# -----------------------------
# NPZ + pre/post split helpers
# -----------------------------
def _find_npz_files(root_dir: Path, recursive: bool) -> List[Path]:
    pat = "**/*.npz" if recursive else "*.npz"
    return sorted([p for p in root_dir.glob(pat) if p.is_file()])

def _load_npz_keys(npz_path: Path, keys: Iterable[str]) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = d[k] if k in d else None
    return out

def _build_file_datetime_map(file_map_obj: Any, *, year_default: int, verbose: bool = False) -> Dict[int, _dt.datetime]:
    fm = file_map_obj
    if isinstance(fm, np.ndarray) and fm.shape == ():
        try:
            fm = fm.item()
        except Exception:
            pass

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
        ok, total = len(dt_map), len(mapping)
        print(f"[dates] parsed {ok}/{total} file_map entries ({(100*ok/max(1,total)):.1f}%)")

    return dt_map

def _split_pre_post_masks(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    treatment_date: _dt.datetime,
    *,
    exclude_treatment_day_from_post: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    fi = file_indices.astype(int)
    max_idx = int(fi.max()) if fi.size else 0
    file_dt_arr = np.array([None] * (max_idx + 1), dtype=object)
    for idx, dt in file_dt_map.items():
        if 0 <= idx <= max_idx:
            file_dt_arr[idx] = dt

    point_dt = file_dt_arr[fi]

    t0 = _dt.datetime.combine(treatment_date.date(), _dt.time(0, 0, 0))
    t1 = _dt.datetime.combine(treatment_date.date(), _dt.time(23, 59, 59, 999999))

    pre = np.array([(d is not None and d < t0) for d in point_dt], dtype=bool)
    if exclude_treatment_day_from_post:
        post = np.array([(d is not None and d > t1) for d in point_dt], dtype=bool)
    else:
        post = np.array([(d is not None and d >= t0) for d in point_dt], dtype=bool)

    return pre, post


# -----------------------------
# Bhattacharyya coefficient (Gaussian)
# -----------------------------
def _cov_est(X: np.ndarray, cov_type: str) -> np.ndarray:
    cov_type = cov_type.lower()
    if cov_type == "diag":
        v = np.var(X, axis=0, ddof=1)
        return np.diag(v)
    if cov_type == "ledoitwolf":
        if not _HAVE_LW:
            raise RuntimeError("cov_type=ledoitwolf requested but sklearn.covariance.LedoitWolf not available.")
        lw = LedoitWolf().fit(X)
        return lw.covariance_
    # full
    return np.cov(X, rowvar=False, ddof=1)

def _slogdet_psd(A: np.ndarray) -> Tuple[float, float]:
    sign, logdet = np.linalg.slogdet(A)
    return float(sign), float(logdet)

def bhattacharyya_gaussian(
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    cov_type: str = "diag",
    eps: float = 1e-6,
    max_tries: int = 6,
) -> Tuple[float, float]:
    """
    Returns (BC, DB) where DB is Bhattacharyya distance and BC = exp(-DB).
    """
    if X0.ndim != 2 or X1.ndim != 2:
        return float("nan"), float("nan")
    if X0.shape[1] != X1.shape[1]:
        return float("nan"), float("nan")
    d = int(X0.shape[1])

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    dmu = (mu1 - mu0).reshape(-1, 1)

    try:
        S0 = _cov_est(X0, cov_type)
        S1 = _cov_est(X1, cov_type)
    except Exception:
        return float("nan"), float("nan")

    I = np.eye(d, dtype=float)

    reg = float(eps)
    for _ in range(max_tries):
        S0r = S0 + reg * I
        S1r = S1 + reg * I
        Sr = 0.5 * (S0r + S1r)

        s0, ld0 = _slogdet_psd(S0r)
        s1, ld1 = _slogdet_psd(S1r)
        s, ld = _slogdet_psd(Sr)

        if s0 > 0 and s1 > 0 and s > 0:
            try:
                Sinv = np.linalg.inv(Sr)
            except Exception:
                reg *= 10.0
                continue

            quad = float((dmu.T @ Sinv @ dmu).squeeze())
            term1 = 0.125 * quad
            term2 = 0.5 * (ld - 0.5 * (ld0 + ld1))
            DB = term1 + term2

            if np.isfinite(DB):
                BC = float(np.exp(-DB))
                return BC, float(DB)

        reg *= 10.0

    return float("nan"), float("nan")


# -----------------------------
# Processing one NPZ
# -----------------------------
def process_one_npz(npz_path: Path, meta: Dict[str, Any], cfg: PrePostBCConfig) -> List[Dict[str, Any]]:
    t_start = time.time()

    animal_id = _infer_animal_id(npz_path)
    hit_type = cfg.override_hit_type if cfg.override_hit_type is not None else meta.get("hit_type", None)
    group = _map_hit_type_to_group(hit_type) if hit_type is not None else meta.get("group", "unknown")

    td = meta.get("treatment_date", None)
    if cfg.override_treatment_date:
        td = _treatment_dt_from_any(cfg.override_treatment_date)
    if td is None:
        print(f"[skip] {animal_id}: missing treatment date.")
        return []

    needed = [cfg.array_key, cfg.cluster_key, cfg.file_key, "file_map"]
    if cfg.vocalization_only:
        needed.append(cfg.vocalization_key)

    arrs = _load_npz_keys(npz_path, needed)
    X = arrs[cfg.array_key]
    clusters = arrs[cfg.cluster_key]
    file_indices = arrs[cfg.file_key]
    file_map_obj = arrs["file_map"]

    if X is None or clusters is None or file_indices is None or file_map_obj is None:
        print(f"[skip] {animal_id}: NPZ missing required keys: {needed}")
        return []

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        print(f"[skip] {animal_id}: {cfg.array_key} is not a 2D array.")
        return []

    N = int(X.shape[0])
    if clusters.shape[0] != N or file_indices.shape[0] != N:
        print(f"[skip] {animal_id}: mismatched array lengths.")
        return []

    base_mask = np.ones(N, dtype=bool)
    if cfg.vocalization_only and arrs.get(cfg.vocalization_key, None) is not None:
        voc = arrs[cfg.vocalization_key]
        try:
            base_mask &= (voc.astype(int) == 1)
        except Exception:
            pass

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
        print(f"[split] {animal_id}: pre={int(pre_mask.sum()):,} post={int(post_mask.sum()):,} (treatment={td.date()})")

    labels = np.unique(clusters.astype(int))
    if not cfg.include_noise:
        labels = labels[labels != -1]

    rng = np.random.default_rng(0)
    cap = cfg.max_points_per_cluster_period

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        lab = int(lab)

        idx_pre = np.where(pre_mask & (clusters.astype(int) == lab))[0]
        idx_post = np.where(post_mask & (clusters.astype(int) == lab))[0]

        if idx_pre.size < cfg.min_points_per_period:
            continue
        if idx_post.size < cfg.min_points_per_period:
            continue

        if cap is not None:
            if idx_pre.size > cap:
                idx_pre = rng.choice(idx_pre, size=cap, replace=False)
            if idx_post.size > cap:
                idx_post = rng.choice(idx_post, size=cap, replace=False)

        X_pre = X[idx_pre]
        X_post = X[idx_post]

        bc, db = bhattacharyya_gaussian(X_pre, X_post, cov_type=cfg.cov_type, eps=cfg.eps)
        if not (np.isfinite(bc) and np.isfinite(db)):
            continue

        rows.append({
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "hit_type": hit_type if hit_type is not None else "",
            "group": group,
            "treatment_date": td.date().isoformat(),
            "cluster_label": lab,
            "array_key": cfg.array_key,
            "cov_type": cfg.cov_type,
            "eps": float(cfg.eps),
            "n_pre": int(idx_pre.size),
            "n_post": int(idx_post.size),
            "bhattacharyya_coeff": float(bc),
            "bhattacharyya_dist": float(db),
        })

    dt = time.time() - t_start
    print(f"[BC] done: {npz_path.name} in {dt:.2f} s  (clusters={len(rows)})")
    return rows


# -----------------------------
# Driver
# -----------------------------
def _choose_defaults(computer_power: str) -> Dict[str, Any]:
    cpu = os.cpu_count() or 8
    if computer_power == "pro":
        return {"workers": max(1, min(8, cpu - 1)), "cap": 20000}
    return {"workers": 1, "cap": 5000}

def run_root_directory(cfg: PrePostBCConfig) -> Dict[str, str]:
    defaults = _choose_defaults(cfg.computer_power)
    workers = cfg.workers if cfg.workers and cfg.workers > 0 else defaults["workers"]
    cap = cfg.max_points_per_cluster_period if cfg.max_points_per_cluster_period is not None else defaults["cap"]

    cfg_eff = PrePostBCConfig(**{**cfg.__dict__, "workers": workers, "max_points_per_cluster_period": cap})

    root_dir = cfg_eff.root_dir
    out_dir = cfg_eff.out_dir if cfg_eff.out_dir is not None else (root_dir / f"bhattacharyya_pre_post_{cfg_eff.array_key}")
    _safe_mkdir(out_dir)

    npz_files = _find_npz_files(root_dir, cfg_eff.recursive)
    if len(npz_files) == 0:
        print(f"[warn] No NPZ files found under: {root_dir}")

    animal_meta = build_animal_metadata_map(cfg_eff.metadata_xlsx, cfg_eff.metadata_sheet)

    tasks: List[Tuple[Path, Dict[str, Any]]] = []
    for p in npz_files:
        aid = _infer_animal_id(p)
        meta = animal_meta.get(aid, {"hit_type": None, "treatment_date": None, "group": "unknown"})
        tasks.append((p, meta))

    print(f"[BC] NPZ files found: {len(npz_files)} (workers={cfg_eff.workers}, cap={cfg_eff.max_points_per_cluster_period}, cov={cfg_eff.cov_type})")

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    if cfg_eff.workers <= 1 or len(tasks) <= 1:
        for p, meta in tasks:
            all_rows.extend(process_one_npz(p, meta, cfg_eff))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
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
    print(f"[BC] TOTAL time: {total_dt:.2f} s")

    stamp = _now_stamp()
    out_csv = out_dir / f"bhattacharyya_pre_post_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    return {"results_csv": str(out_csv), "out_dir": str(out_dir)}


# -----------------------------
# CLI
# -----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhattacharyya_pre_vs_post_cluster.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=str, help="Directory containing NPZ(s) (bird folder or parent).")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Excel metadata with treatment dates and hit types.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str, help="Sheet to read hit types from.")
    p.add_argument("--out-dir", default=None, type=str, help="Output directory.")
    p.add_argument("--recursive", action="store_true", help="Recursively search for NPZs.")

    p.add_argument("--array-key", default="predictions", type=str, help="Key in NPZ for points (e.g., predictions or embedding_outputs).")
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str, help="Key in NPZ for cluster labels.")
    p.add_argument("--file-key", default="file_indices", type=str, help="Key in NPZ for file indices.")
    p.add_argument("--include-noise", action="store_true", help="Include HDBSCAN noise label (-1).")

    p.add_argument("--no-vocalization-only", action="store_true", help="Do NOT restrict to vocalization==1.")
    p.add_argument("--vocalization-key", default="vocalization", type=str, help="Key in NPZ for vocalization mask.")

    p.add_argument("--min-points-per-period", default=200, type=int, help="Min points required in both pre and post for a cluster.")
    p.add_argument("--max-points-per-cluster-period", default=None, type=int, help="Subsample cap per cluster per period.")

    p.add_argument("--cov-type", default="diag", choices=["diag", "full", "ledoitwolf"], help="Covariance estimator.")
    p.add_argument("--eps", default=1e-6, type=float, help="Covariance regularization strength.")

    p.add_argument("--exclude-treatment-day-from-post", action="store_true", help="POST starts day after treatment day.")

    p.add_argument("--computer-power", default="laptop", choices=["laptop", "pro"], help="Sets defaults for workers/caps.")
    p.add_argument("--workers", default=0, type=int, help="Processes across NPZ files (0 => auto).")

    p.add_argument("--treatment-date", default=None, type=str, help="Override treatment date for ALL birds (YYYY-MM-DD or ISO datetime).")
    p.add_argument("--hit-type", default=None, type=str, help="Override hit type for ALL birds (debugging).")
    p.add_argument("--verbose-dates", action="store_true", help="Print date parsing diagnostics and pre/post counts.")
    return p

def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = PrePostBCConfig(
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

        min_points_per_period=int(args.min_points_per_period),
        max_points_per_cluster_period=args.max_points_per_cluster_period,

        cov_type=str(args.cov_type),
        eps=float(args.eps),

        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),

        computer_power=str(args.computer_power),
        workers=int(args.workers),

        override_treatment_date=args.treatment_date,
        override_hit_type=args.hit_type,
        verbose_dates=bool(args.verbose_dates),
    )

    paths = run_root_directory(cfg)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
