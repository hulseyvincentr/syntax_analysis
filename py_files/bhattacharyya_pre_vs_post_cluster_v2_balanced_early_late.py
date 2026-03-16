#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bhattacharyya_pre_vs_post_cluster_v2_balanced_early_late.py

Extends your original bhattacharyya_pre_vs_post_cluster.py by adding:
1) Balanced sample sizes per comparison (subsample both periods to min(nA, nB), optionally capped).
2) Additional comparisons:
     - pre_vs_post                  (post vs pre)
     - pre_early_vs_late            (late pre vs early pre)
     - post_early_vs_late           (late post vs early post)
     - late_pre_vs_early_post       (early post vs late pre)
3) Clearer group names (for plots + CSV):
     - "Lateral hit only"
     - "Medial Lateral hit, combined with large lesion"
4) Optional plots + stats (can be disabled with --no-plots):
     - For each comparison: boxplot of Bhattacharyya coefficient (BC) by hit type group
     - Bird-level paired plot comparing within-pre drift vs within-post drift:
           BC(pre early vs late)  vs  BC(post early vs late)
       with x-axis labels "pre-lesion" and "post-lesion".
     - Per-bird cluster-level paired plots using ALL syllables:
           For each bird, plot paired BC values for each cluster
           comparing pre_early_vs_late vs post_early_vs_late.

BC is computed under a multivariate Gaussian approximation:
  - Fit N(mu_A, Sigma_A) and N(mu_B, Sigma_B)
  - Bhattacharyya distance (DB):
        DB = 1/8 (dmu^T Sigma^-1 dmu) + 1/2 ln( det(Sigma) / sqrt(det(Sigma_A) det(Sigma_B)) )
        where Sigma = (Sigma_A + Sigma_B)/2
  - Bhattacharyya coefficient (BC):
        BC = exp(-DB)

Outputs:
  - CSV with one row per (animal_id, cluster_label, comparison) that has enough points in both periods.
  - If plotting enabled: PNGs + TXT stats files in output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence
import argparse
import datetime as _dt
import os
import re
import time
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    from sklearn.covariance import LedoitWolf  # type: ignore
    _HAVE_LW = True
except Exception:
    _HAVE_LW = False


@dataclass(frozen=True)
class PrePostBCConfig:
    root_dir: Path
    metadata_xlsx: Path
    metadata_sheet: str

    out_dir: Optional[Path] = None
    recursive: bool = False

    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"

    include_noise: bool = False
    vocalization_only: bool = True
    vocalization_key: str = "vocalization"

    min_points_per_period: int = 200
    max_points_per_cluster_period: Optional[int] = None

    balance_pair_samples: bool = True

    split_early_late: bool = True
    early_late_split_method: str = "file_median"  # file_median | file_half

    cov_type: str = "diag"  # diag | full | ledoitwolf
    eps: float = 1e-6

    workers: int = 0
    computer_power: str = "laptop"  # laptop | pro

    exclude_treatment_day_from_post: bool = False

    override_treatment_date: Optional[str] = None
    override_hit_type: Optional[str] = None
    verbose_dates: bool = False

    variance_csv: Optional[Path] = None
    variance_top_pct: float = 30.0

    make_plots: bool = True


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
    if not (10000.0 <= x <= 80000.0):
        return None
    try:
        return _EXCEL_ORIGIN + _dt.timedelta(days=float(x))
    except Exception:
        return None


def _add_variance_tier_column(
    df: pd.DataFrame,
    variance_csv: Path,
    top_pct: float = 30.0,
    *,
    df_animal_col: str = "animal_id",
    df_syllable_col: str = "cluster_label",
) -> pd.DataFrame:
    """
    Add df["variance_tier"] in {"high", "low", "unknown"} based on per-bird syllable variance.

    "High" = syllables in the top `top_pct` percent variance (per bird).
    "Low"  = the remaining syllables with variance values (per bird).
    "Unknown" = syllables not found in the variance CSV for that bird.

    Heuristic for variance value per (bird, syllable):
      1) Use first non-null "Pre_Variance_ms2" if available
      2) Else use mean "Variance_ms2" over rows where Group contains "Pre" (case-insensitive)
      3) Else use mean "Variance_ms2" over all rows
    """
    if df.empty:
        out = df.copy()
        out["variance_tier"] = pd.Series(dtype="object")
        return out

    top_pct = float(top_pct)
    if top_pct <= 0 or top_pct >= 100:
        raise ValueError(f"--variance-top-pct must be between 0 and 100 (got {top_pct})")

    v = pd.read_csv(variance_csv)

    cols = {c.strip(): c for c in v.columns}

    def _pick(cands: Sequence[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        low = {c.lower(): c for c in v.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    col_animal = _pick(["Animal ID", "animal_id", "AnimalID", "bird_id"])
    col_syll = _pick(["Syllable", "syllable", "cluster_label", "label"])
    col_group = _pick(["Group", "group", "Period", "period"])
    col_prevar = _pick(["Pre_Variance_ms2", "pre_variance_ms2", "PreVariance_ms2"])
    col_var = _pick(["Variance_ms2", "variance_ms2", "Variance", "variance"])

    if col_animal is None or col_syll is None:
        raise ValueError(
            "variance CSV must include bird and syllable columns (e.g., 'Animal ID' and 'Syllable')."
        )
    if col_prevar is None and col_var is None:
        raise ValueError(
            "variance CSV must include a variance column (e.g., 'Pre_Variance_ms2' and/or 'Variance_ms2')."
        )

    idx_cols = [col_animal, col_syll]

    if col_prevar is not None:
        s_pre = (
            v.dropna(subset=[col_prevar])
            .groupby(idx_cols, dropna=False)[col_prevar]
            .first()
        )
    else:
        s_pre = pd.Series(dtype=float)

    if col_var is not None:
        if col_group is not None:
            pre_mask = v[col_group].astype(str).str.contains("pre", case=False, na=False)
        else:
            pre_mask = pd.Series(False, index=v.index)

        if pre_mask.any():
            s_pre_mean = v.loc[pre_mask].groupby(idx_cols, dropna=False)[col_var].mean()
        else:
            s_pre_mean = pd.Series(dtype=float)

        s_all_mean = v.groupby(idx_cols, dropna=False)[col_var].mean()
    else:
        s_pre_mean = pd.Series(dtype=float)
        s_all_mean = pd.Series(dtype=float)

    s_val = s_pre
    if not s_pre_mean.empty:
        s_val = s_val.combine_first(s_pre_mean)
    if not s_all_mean.empty:
        s_val = s_val.combine_first(s_all_mean)
    s_val.name = "variance_value"

    vmap = s_val.reset_index().rename(columns={col_animal: "animal_id", col_syll: "cluster_label"})
    vmap["animal_id"] = vmap["animal_id"].astype(str)
    vmap["cluster_label"] = pd.to_numeric(vmap["cluster_label"], errors="coerce")
    vmap = vmap.dropna(subset=["cluster_label"]).copy()
    vmap["cluster_label"] = vmap["cluster_label"].astype(int)

    q = 100.0 - top_pct

    def _pct(s: pd.Series) -> float:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.percentile(arr, q))

    vmap["variance_thresh"] = vmap.groupby("animal_id")["variance_value"].transform(_pct)
    vmap["variance_tier"] = np.where(
        np.isfinite(vmap["variance_thresh"]) & (vmap["variance_value"] >= vmap["variance_thresh"]),
        "high",
        "low",
    )

    out = df.copy()
    out[df_animal_col] = out[df_animal_col].astype(str)
    out[df_syllable_col] = pd.to_numeric(out[df_syllable_col], errors="coerce")

    vmap2 = vmap.rename(columns={"animal_id": "__var_animal_id", "cluster_label": "__var_cluster_label"})
    out = out.merge(
        vmap2[["__var_animal_id", "__var_cluster_label", "variance_tier"]],
        how="left",
        left_on=[df_animal_col, df_syllable_col],
        right_on=["__var_animal_id", "__var_cluster_label"],
    )
    out = out.drop(columns=["__var_animal_id", "__var_cluster_label"], errors="ignore")
    out["variance_tier"] = out["variance_tier"].fillna("unknown")

    try:
        counts = out["variance_tier"].value_counts(dropna=False).to_dict()
        print(f"[variance tiers] top_pct={top_pct:.1f}% -> counts: {counts}")
    except Exception:
        pass

    return out


def parse_datetime_from_filename(file_path: str | Path, *, year_default: int = 2024) -> Optional[_dt.datetime]:
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


def _normalize_hit_type(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _map_hit_type_to_group(hit_type: str) -> str:
    if hit_type is None:
        return "unknown"
    norm = _normalize_hit_type(str(hit_type))
    if "sham" in norm:
        return "sham saline injection"
    if "singlehit" in norm or ("lateral" in norm and "hit" in norm and "medial" not in norm):
        return "Lateral hit only"
    if ("medial" in norm and "lateral" in norm) or ("ml" in norm and "visible" in norm):
        return "Medial Lateral hit, combined with large lesion"
    if "notvisible" in norm or ("large" in norm and "lesion" in norm):
        return "Medial Lateral hit, combined with large lesion"
    return str(hit_type)


def build_animal_metadata_map(metadata_xlsx: Path, metadata_sheet: str) -> Dict[str, Dict[str, Any]]:
    xlsx = Path(metadata_xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"metadata_xlsx not found: {xlsx}")

    df_main = pd.read_excel(xlsx, sheet_name=metadata_sheet)

    animal_col = None
    for c in df_main.columns:
        if str(c).strip().lower() in ("animal id", "animal_id", "animalid", "bird", "bird id"):
            animal_col = c
            break
    if animal_col is None:
        animal_col = df_main.columns[0]

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

    treat_col = None
    for c in df_main.columns:
        if "treatment date" in str(c).lower():
            treat_col = c
            break

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
        print(f"[dates] parsed {ok}/{total} file_map entries ({(100 * ok / max(1, total)):.1f}%)")

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


def _split_early_late_masks_by_file(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    period_mask: np.ndarray,
    *,
    method: str = "file_median",
) -> Tuple[np.ndarray, np.ndarray]:
    fi = file_indices.astype(int)
    files = np.unique(fi[period_mask])

    file_dts = []
    for f in files:
        dt = file_dt_map.get(int(f), None)
        if dt is not None:
            file_dts.append((int(f), dt))

    if len(file_dts) < 2:
        return period_mask.copy(), np.zeros_like(period_mask, dtype=bool)

    file_dts.sort(key=lambda t: t[1])

    method = (method or "file_median").lower()
    if method == "file_half":
        mid = len(file_dts) // 2
        early_files = {f for f, _ in file_dts[:mid]}
        late_files = {f for f, _ in file_dts[mid:]}
    else:
        ts = np.array([dt.timestamp() for _, dt in file_dts], dtype=float)
        med = float(np.median(ts))
        early_files = {f for (f, dt) in file_dts if dt.timestamp() <= med}
        late_files = {f for (f, dt) in file_dts if dt.timestamp() > med}
        if len(late_files) == 0:
            mid = len(file_dts) // 2
            early_files = {f for f, _ in file_dts[:mid]}
            late_files = {f for f, _ in file_dts[mid:]}

    early_mask = period_mask & np.isin(fi, np.fromiter(early_files, dtype=int))
    late_mask = period_mask & np.isin(fi, np.fromiter(late_files, dtype=int))
    return early_mask, late_mask


def _balanced_subsample_pair(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    rng: np.random.Generator,
    cap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
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


def _cov_est(X: np.ndarray, cov_type: str, *, eps: float) -> np.ndarray:
    cov_type = cov_type.lower()
    if cov_type == "diag":
        v = np.var(X, axis=0, ddof=1)
        v = np.where(np.isfinite(v), v, 0.0)
        v = np.maximum(v, float(eps))
        return np.diag(v)
    if cov_type == "ledoitwolf":
        if not _HAVE_LW:
            raise RuntimeError("cov_type=ledoitwolf requested but sklearn.covariance.LedoitWolf not available.")
        lw = LedoitWolf().fit(X)
        return lw.covariance_
    return np.cov(X, rowvar=False, ddof=1)


def _slogdet_psd(A: np.ndarray) -> Tuple[float, float]:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
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
    if X0.ndim != 2 or X1.ndim != 2:
        return float("nan"), float("nan")
    if X0.shape[1] != X1.shape[1]:
        return float("nan"), float("nan")
    d = int(X0.shape[1])

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    dmu = (mu1 - mu0).reshape(-1, 1)

    try:
        S0 = _cov_est(X0, cov_type, eps=eps)
        S1 = _cov_est(X1, cov_type, eps=eps)
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


_GROUP_ORDER = [
    "sham saline injection",
    "Lateral hit only",
    "Medial Lateral hit, combined with large lesion",
]

_COMP_ORDER = [
    ("pre_vs_post", "pre-lesion vs post-lesion"),
    ("pre_early_vs_late", "within pre-lesion (early vs late)"),
    ("post_early_vs_late", "within post-lesion (early vs late)"),
    ("late_pre_vs_early_post", "late pre-lesion vs early post-lesion"),
]


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _mannwhitney_safe(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return float("nan")

    if np.all(a == a.flat[0]) and np.all(b == b.flat[0]) and a.flat[0] == b.flat[0]:
        return 1.0

    try:
        res = stats.mannwhitneyu(a, b, alternative=alternative, method="auto")
        return float(res.pvalue)
    except TypeError:
        try:
            res = stats.mannwhitneyu(a, b, alternative=alternative)
            return float(res.pvalue)
        except Exception:
            return 1.0
    except Exception:
        return 1.0


def _holm_bonferroni(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    if p.size == 0:
        return p

    p_clean = np.where(np.isfinite(p), p, 1.0)

    m = int(p_clean.size)
    order = np.argsort(p_clean)
    p_sorted = p_clean[order]

    adj_sorted = np.empty_like(p_sorted)
    prev = 0.0
    for i, pv in enumerate(p_sorted):
        mult = float(m - i)
        val = mult * float(pv)
        if val < prev:
            val = prev
        prev = val
        adj_sorted[i] = 1.0 if val > 1.0 else val

    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def _add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black", clip_on=False)
    text_pad = max(h * 0.5, 0.012)
    ax.text((x1 + x2) / 2.0, y + h + text_pad, text, ha="center", va="bottom", fontsize=11, clip_on=False)


def _boxplot_bc_by_group(
    title: str,
    ylabel: str,
    groups: List[str],
    data: List[np.ndarray],
    out_png: Path,
    *,
    show_stats: bool = True,
    stats_mode: str = "all_pairs",
    title_y: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    try:
        ax.boxplot(data, tick_labels=groups, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=groups, showfliers=True)

    if title_y is not None:
        ax.set_title(title, pad=18, y=title_y)
    else:
        ax.set_title(title, pad=18)

    ax.set_ylabel(ylabel)
    ax.grid(False)

    mins_maxs = []
    for v in data:
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            mins_maxs.append((float(arr.min()), float(arr.max())))
    if mins_maxs:
        ymin = min(mn for mn, _mx in mins_maxs)
        ymax = max(mx for _mn, mx in mins_maxs)
    else:
        ymin, ymax = 0.0, 1.0

    y_lower = min(-0.05, ymin - 0.05)
    y_upper_base = max(1.05, ymax + 0.05)

    if show_stats and len(groups) >= 2:
        x_positions = list(range(1, len(groups) + 1))
        pairs: List[Tuple[int, int, str, str, float]] = []

        if stats_mode == "vs_first":
            for j in range(1, len(groups)):
                p = _mannwhitney_safe(data[0], data[j])
                pairs.append((0, j, groups[0], groups[j], p))
        else:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    p = _mannwhitney_safe(data[i], data[j])
                    pairs.append((i, j, groups[i], groups[j], p))

        pvals = np.array([p for *_rest, p in pairs], dtype=float)
        p_holm = _holm_bonferroni(pvals)

        y_span = max(1e-9, ymax - y_lower)
        base = ymax + max(0.05, 0.06 * y_span)
        h = max(0.025, 0.03 * y_span)
        step = max(0.08, 0.10 * y_span)

        spans = [(abs(j - i), k) for k, (i, j, *_rest) in enumerate(pairs)]
        spans.sort()

        for level, (_span, k) in enumerate(spans):
            i, j, *_rest = pairs[k]
            p_adj = float(p_holm[k])
            stars = _p_to_stars(p_adj)
            label = f"{stars} (p={p_adj:.3g})" if stars != "n.s." else f"n.s. (p={p_adj:.3g})"
            y = base + level * step
            _add_sig_bracket(ax, x_positions[i], x_positions[j], y, h, label)

        text_pad = max(0.015, 0.6 * h)
        top_needed = base + (len(pairs) - 1) * step + h + text_pad + max(0.06, 0.06 * y_span)
        ax.set_ylim(y_lower, min(1.40, max(y_upper_base, top_needed)))
    else:
        ax.set_ylim(y_lower, min(1.40, y_upper_base))

    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _write_group_stats(df: pd.DataFrame, out_txt: Path, title: str) -> None:
    lines: List[str] = [title, ""]
    for g in _GROUP_ORDER:
        sub = df[df["group"] == g]["bhattacharyya_coeff"].dropna()
        if len(sub) == 0:
            continue
        lines.append(
            f"[{g}] n={len(sub)} mean={sub.mean():.6g} median={sub.median():.6g} std={sub.std(ddof=1):.6g}"
        )
    lines.append("")

    if _HAVE_SCIPY:
        groups = [df[df["group"] == g]["bhattacharyya_coeff"].dropna().values for g in _GROUP_ORDER]
        groups = [x for x in groups if len(x) > 0]
        if len(groups) >= 2:
            try:
                if len(groups) >= 3:
                    H, p = stats.kruskal(*groups)
                    lines.append(f"Kruskal–Wallis: H={H:.6g} p={p:.6g}")

                present = [(g, df[df["group"] == g]["bhattacharyya_coeff"].dropna().values) for g in _GROUP_ORDER]
                present = [(g, v) for g, v in present if len(v) > 0]

                if len(present) >= 2:
                    pairs = []
                    pvals = []
                    for i in range(len(present)):
                        for j in range(i + 1, len(present)):
                            g1, v1 = present[i]
                            g2, v2 = present[j]
                            U, p2 = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                            pairs.append((g1, g2, float(U)))
                            pvals.append(float(p2))

                    m = len(pvals)
                    order = np.argsort(pvals)
                    p_adj = [None] * m
                    for rank, idx in enumerate(order):
                        p_adj[idx] = min(1.0, pvals[idx] * (m - rank))

                    lines.append("Pairwise Mann–Whitney U (Holm-adjusted):")
                    for (g1, g2, U), p_raw, p_holm in zip(pairs, pvals, p_adj):
                        lines.append(f"  {g1} vs {g2}: U={U:.6g} p={p_raw:.6g} p_holm={p_holm:.6g}")
            except Exception as e:
                lines.append(f"[warn] stats failed: {repr(e)}")
    else:
        lines.append("[note] SciPy not available; skipping statistical tests.")

    out_txt.write_text("\n".join(lines) + "\n")


def _paired_drift_plot_birdlevel(
    df: pd.DataFrame,
    out_png: Path,
    out_txt: Path,
    *,
    title: str = "Paired drift BC comparison within birds by hit type",
) -> None:
    pre = (
        df[df["comparison"] == "pre_early_vs_late"]
        .groupby(["animal_id", "group"], as_index=False)["bhattacharyya_coeff"]
        .mean()
    )
    post = (
        df[df["comparison"] == "post_early_vs_late"]
        .groupby(["animal_id", "group"], as_index=False)["bhattacharyya_coeff"]
        .mean()
    )
    merged = pd.merge(pre, post, on=["animal_id", "group"], suffixes=("_pre", "_post"))

    stats_lines: List[str] = [title, "Metric: Bhattacharyya coefficient (higher = more similar)", ""]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(title, y=0.98)

    for ax, g in zip(axes, _GROUP_ORDER):
        sub = merged[merged["group"] == g].copy()
        ax.set_title(g)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No birds", ha="center", va="center", transform=ax.transAxes)
            stats_lines.append(f"[{g}] n_birds=0")
            continue

        y1 = sub["bhattacharyya_coeff_pre"].values
        y2 = sub["bhattacharyya_coeff_post"].values

        ax.boxplot([y1, y2], positions=[1, 2], widths=0.5, showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["pre-lesion", "post-lesion"])

        for _, r in sub.iterrows():
            ax.plot([1, 2], [r["bhattacharyya_coeff_pre"], r["bhattacharyya_coeff_post"]], alpha=0.8)
            ax.scatter([1, 2], [r["bhattacharyya_coeff_pre"], r["bhattacharyya_coeff_post"]], s=25)

        ax.set_ylim(0.0, 1.05)

        if _HAVE_SCIPY and len(sub) >= 2:
            try:
                w, p = stats.wilcoxon(y1, y2, alternative="two-sided", zero_method="wilcox")
                stats_lines.append(f"[{g}] n_birds={len(sub)} Wilcoxon(pre vs post drift BC): W={w:.6g} p={p:.6g}")
                ax.text(
                    0.5,
                    0.95,
                    f"n.s. (p={p:.3g})" if p >= 0.05 else f"* (p={p:.3g})",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                )
            except Exception as e:
                stats_lines.append(f"[{g}] n_birds={len(sub)} Wilcoxon failed: {repr(e)}")
        else:
            stats_lines.append(f"[{g}] n_birds={len(sub)} (SciPy missing or n<2)")

    axes[0].set_ylabel("Bhattacharyya coefficient (BC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    out_txt.write_text("\n".join(stats_lines) + "\n")


def _paired_cluster_plots_by_bird(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    array_key: str,
    cov_type: str,
    stamp: str,
    title_suffix: str = "all syllables",
) -> None:
    """
    Make one paired plot per bird using ALL cluster-level BC values.

    Each line corresponds to one cluster that has both a pre_early_vs_late BC and
    a post_early_vs_late BC.
    """
    pre = (
        df[df["comparison"] == "pre_early_vs_late"]
        [["animal_id", "group", "cluster_label", "bhattacharyya_coeff"]]
        .rename(columns={"bhattacharyya_coeff": "bhattacharyya_coeff_pre"})
    )
    post = (
        df[df["comparison"] == "post_early_vs_late"]
        [["animal_id", "group", "cluster_label", "bhattacharyya_coeff"]]
        .rename(columns={"bhattacharyya_coeff": "bhattacharyya_coeff_post"})
    )
    merged = pd.merge(pre, post, on=["animal_id", "group", "cluster_label"], how="inner")

    if len(merged) == 0:
        print(f"[skip per-bird plots] no overlapping pre/post early-late cluster BC values for {title_suffix}")
        return

    per_bird_dir = out_dir / "per_bird_cluster_paired_BC"
    per_bird_dir.mkdir(parents=True, exist_ok=True)

    for (animal_id, group), sub in merged.groupby(["animal_id", "group"], sort=True):
        sub = sub.sort_values("cluster_label").copy()
        if len(sub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y1 = sub["bhattacharyya_coeff_pre"].to_numpy(dtype=float)
        y2 = sub["bhattacharyya_coeff_post"].to_numpy(dtype=float)

        for _, r in sub.iterrows():
            y_pre = float(r["bhattacharyya_coeff_pre"])
            y_post = float(r["bhattacharyya_coeff_post"])
            ax.plot([1, 2], [y_pre, y_post], alpha=0.75, linewidth=1.1)
            ax.scatter([1, 2], [y_pre, y_post], s=28)

            if len(sub) <= 20:
                ax.text(2.04, y_post, str(int(r["cluster_label"])), fontsize=8, va="center")

        vals = np.concatenate([y1, y2]) if len(sub) > 0 else np.array([0.0, 1.0])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            y_min, y_max = 0.0, 1.0
        else:
            y_min = float(vals.min())
            y_max = float(vals.max())

        ax.set_xlim(0.85, 2.18 if len(sub) <= 20 else 2.05)
        ax.set_ylim(min(0.0, y_min - 0.05), min(1.05, max(1.05, y_max + 0.05)))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["pre-lesion", "post-lesion"])
        ax.set_ylabel("Bhattacharyya coefficient (BC)")
        ax.set_title(f"{animal_id}: cluster-level paired drift BC\n{group} ({title_suffix})", pad=14)
        ax.text(0.02, 0.98, f"n clusters={len(sub)}", transform=ax.transAxes, ha="left", va="top")

        safe_animal = re.sub(r"[^A-Za-z0-9._-]+", "_", str(animal_id))
        out_png = per_bird_dir / f"{safe_animal}_cluster_paired_drift_BC_{array_key}_{cov_type}_{stamp}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


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

    try:
        arrs = _load_npz_keys(npz_path, needed)
    except (BadZipFile, OSError, ValueError) as e:
        print(f"[skip] {animal_id}: failed to load NPZ ({npz_path.name}): {e}")
        return []

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

    pre_early = pre_late = None
    post_early = post_late = None
    if cfg.split_early_late:
        pre_early, pre_late = _split_early_late_masks_by_file(
            file_indices, file_dt_map, pre_mask, method=cfg.early_late_split_method
        )
        post_early, post_late = _split_early_late_masks_by_file(
            file_indices, file_dt_map, post_mask, method=cfg.early_late_split_method
        )

    labels = np.unique(clusters.astype(int))
    if not cfg.include_noise:
        labels = labels[labels != -1]

    rng = np.random.default_rng(0)
    cap = cfg.max_points_per_cluster_period

    def _compute_pair(
        lab: int,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        period_a: str,
        period_b: str,
        comparison: str,
    ) -> Optional[Dict[str, Any]]:
        idx_a = np.where(mask_a & (clusters.astype(int) == lab))[0]
        idx_b = np.where(mask_b & (clusters.astype(int) == lab))[0]

        if idx_a.size < cfg.min_points_per_period or idx_b.size < cfg.min_points_per_period:
            return None

        if cfg.balance_pair_samples:
            idx_a, idx_b, n_target, n_a_raw, n_b_raw = _balanced_subsample_pair(
                idx_a, idx_b, rng=rng, cap=cap
            )
            if n_target < cfg.min_points_per_period:
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

        bc, db = bhattacharyya_gaussian(X_a, X_b, cov_type=cfg.cov_type, eps=cfg.eps)
        if not (np.isfinite(bc) and np.isfinite(db)):
            return None

        return {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "hit_type": hit_type if hit_type is not None else "",
            "group": group,
            "treatment_date": td.date().isoformat(),
            "cluster_label": int(lab),
            "array_key": cfg.array_key,
            "cov_type": cfg.cov_type,
            "eps": float(cfg.eps),
            "comparison": comparison,
            "period_a": period_a,
            "period_b": period_b,
            "n_a_raw": int(n_a_raw),
            "n_b_raw": int(n_b_raw),
            "n_a": int(idx_a.size),
            "n_b": int(idx_b.size),
            "bhattacharyya_coeff": float(bc),
            "bhattacharyya_dist": float(db),
        }

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        lab = int(lab)

        r = _compute_pair(lab, pre_mask, post_mask, "pre", "post", "pre_vs_post")
        if r is not None:
            rows.append(r)

        if cfg.split_early_late and pre_early is not None and pre_late is not None:
            r = _compute_pair(lab, pre_early, pre_late, "pre_early", "pre_late", "pre_early_vs_late")
            if r is not None:
                rows.append(r)

        if cfg.split_early_late and post_early is not None and post_late is not None:
            r = _compute_pair(lab, post_early, post_late, "post_early", "post_late", "post_early_vs_late")
            if r is not None:
                rows.append(r)

        if cfg.split_early_late and (pre_late is not None) and (post_early is not None):
            r = _compute_pair(lab, pre_late, post_early, "pre_late", "post_early", "late_pre_vs_early_post")
            if r is not None:
                rows.append(r)

    dt = time.time() - t_start
    print(f"[BC] done: {npz_path.name} in {dt:.2f} s  (rows={len(rows)})")
    return rows


def _choose_defaults(computer_power: str) -> Dict[str, Any]:
    cpu = os.cpu_count() or 8
    if computer_power == "pro":
        return {"workers": max(1, min(8, max(1, cpu - 1))), "cap": 20000}
    return {"workers": 1, "cap": 5000}


def run_root_directory(cfg: PrePostBCConfig) -> Dict[str, str]:
    defaults = _choose_defaults(cfg.computer_power)
    workers = cfg.workers if cfg.workers and cfg.workers > 0 else defaults["workers"]
    cap = cfg.max_points_per_cluster_period if cfg.max_points_per_cluster_period is not None else defaults["cap"]

    cfg_eff = PrePostBCConfig(**{**cfg.__dict__, "workers": workers, "max_points_per_cluster_period": cap})

    root_dir = cfg_eff.root_dir
    out_dir = cfg_eff.out_dir if cfg_eff.out_dir is not None else (root_dir / f"bhattacharyya_comparisons_{cfg_eff.array_key}")
    _safe_mkdir(out_dir)

    npz_files = _find_npz_files(root_dir, cfg_eff.recursive)
    animal_meta = build_animal_metadata_map(cfg_eff.metadata_xlsx, cfg_eff.metadata_sheet)

    tasks: List[Tuple[Path, Dict[str, Any]]] = []
    for p in npz_files:
        aid = _infer_animal_id(p)
        meta = animal_meta.get(aid, {"hit_type": None, "treatment_date": None, "group": "unknown"})
        tasks.append((p, meta))

    print(
        f"[BC] NPZ files found: {len(npz_files)} (workers={cfg_eff.workers}, cap={cfg_eff.max_points_per_cluster_period}, "
        f"cov={cfg_eff.cov_type}, balance={cfg_eff.balance_pair_samples}, split_early_late={cfg_eff.split_early_late})"
    )

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
    out_csv = out_dir / f"bhattacharyya_comparisons_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.csv"
    df = pd.DataFrame(all_rows)

    if cfg_eff.variance_csv is not None and len(df) > 0:
        df = _add_variance_tier_column(df, cfg_eff.variance_csv, cfg_eff.variance_top_pct)

    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    if cfg_eff.make_plots and len(df) > 0:
        # Always make per-bird cluster paired plots using ALL syllables
        _paired_cluster_plots_by_bird(
            df,
            out_dir,
            array_key=cfg_eff.array_key,
            cov_type=cfg_eff.cov_type,
            stamp=stamp,
            title_suffix="all syllables",
        )

        plot_specs: List[Tuple[str, pd.DataFrame]] = []
        if (cfg_eff.variance_csv is not None) and ("variance_tier" in df.columns):
            for tier_label in ("high", "low"):
                df_tier = df[df["variance_tier"] == tier_label].copy()
                if len(df_tier) > 0:
                    plot_specs.append((tier_label, df_tier))
        else:
            plot_specs.append(("all", df))

        for tier_label, df_plot in plot_specs:
            if tier_label == "all":
                tier_dir = out_dir
                tier_descr = "all syllables"
            else:
                tier_dir = out_dir / f"variance_{tier_label}_top{int(cfg_eff.variance_top_pct)}"
                tier_dir.mkdir(parents=True, exist_ok=True)
                if tier_label == "high":
                    tier_descr = f"high-variance syllables (top {cfg_eff.variance_top_pct:.0f}%)"
                else:
                    tier_descr = f"low-variance syllables (bottom {100.0 - cfg_eff.variance_top_pct:.0f}%)"

                tier_csv = tier_dir / f"bhattacharyya_comparisons_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}_{tier_label}.csv"
                df_plot.to_csv(tier_csv, index=False)

            for comp_key, comp_title in _COMP_ORDER:
                sub = df_plot[df_plot["comparison"] == comp_key].copy()
                if len(sub) == 0:
                    continue

                png = tier_dir / f"{comp_key}_bc_by_hit_type_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.png"
                txt = tier_dir / f"{comp_key}_bc_stats_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.txt"

                groups = []
                data = []
                for g in _GROUP_ORDER:
                    arr = sub.loc[sub["group"] == g, "bhattacharyya_coeff"].dropna().to_numpy()
                    if arr.size:
                        groups.append(f"{g}\n(n={arr.size})")
                        data.append(arr)

                if len(data) >= 2:
                    _boxplot_bc_by_group(
                        title=f"BC by hit type: {comp_title} ({tier_descr})",
                        ylabel="Bhattacharyya coefficient (BC)",
                        groups=groups,
                        data=data,
                        out_png=png,
                        show_stats=True,
                        stats_mode="all_pairs",
                        title_y=1.08,
                    )
                    _write_group_stats(sub, txt, title=f"BC by hit type: {comp_title} ({tier_descr})")
                else:
                    print(f"[skip plot] {comp_key} ({tier_label}) not enough groups with data")

            paired_png = tier_dir / (
                f"paired_drift_BC_pre_vs_post_by_hit_birdlevel_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.png"
            )
            paired_txt = tier_dir / (
                f"paired_drift_BC_pre_vs_post_by_hit_birdlevel_{cfg_eff.array_key}_{cfg_eff.cov_type}_{stamp}.txt"
            )
            _paired_drift_plot_birdlevel(
                df_plot,
                paired_png,
                paired_txt,
                title=f"Paired drift BC comparison within birds by hit type ({tier_descr})",
            )

    return {"results_csv": str(out_csv), "out_dir": str(out_dir)}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhattacharyya_pre_vs_post_cluster_v2_balanced_early_late.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root-dir", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str)
    p.add_argument("--out-dir", default=None, type=str)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--include-noise", action="store_true")

    p.add_argument("--no-vocalization-only", action="store_true")
    p.add_argument("--vocalization-key", default="vocalization", type=str)

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-cluster-period", default=None, type=int)
    p.add_argument("--no-balance-pairs", action="store_true")

    p.add_argument("--no-split-early-late", action="store_true")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"])

    p.add_argument("--cov-type", default="diag", choices=["diag", "full", "ledoitwolf"])
    p.add_argument("--eps", default=1e-6, type=float)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")

    p.add_argument("--computer-power", default="laptop", choices=["laptop", "pro"])
    p.add_argument("--workers", default=0, type=int)

    p.add_argument("--treatment-date", default=None, type=str)
    p.add_argument("--hit-type", default=None, type=str)
    p.add_argument("--verbose-dates", action="store_true")

    p.add_argument("--no-plots", action="store_true")

    p.add_argument(
        "--variance-csv",
        default=None,
        help="CSV with per-bird per-syllable variance values (used to split into high/low variance tiers).",
    )
    p.add_argument(
        "--variance-top-pct",
        type=float,
        default=30.0,
        help="Top percentage (per bird) to label as 'high variance' (default 30 = top 30%).",
    )
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

        balance_pair_samples=not bool(args.no_balance_pairs),
        split_early_late=not bool(args.no_split_early_late),
        early_late_split_method=str(args.early_late_split_method),

        cov_type=str(args.cov_type),
        eps=float(args.eps),

        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),

        computer_power=str(args.computer_power),
        workers=int(args.workers),

        override_treatment_date=args.treatment_date,
        override_hit_type=args.hit_type,
        verbose_dates=bool(args.verbose_dates),

        variance_csv=(Path(args.variance_csv) if args.variance_csv else None),
        variance_top_pct=float(args.variance_top_pct),

        make_plots=not bool(args.no_plots),
    )

    paths = run_root_directory(cfg)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()