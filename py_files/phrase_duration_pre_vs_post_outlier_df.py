# -*- coding: utf-8 -*-
# phrase_duration_pre_vs_post_outlier_df.py
"""
Adds OUTLIER/Tail analysis for:
  (A) Phrase durations (ms)  ✅ (already implemented)
  (B) Per-song variance of phrase durations (ms^2)  ✅ NEW

Important note on what “variance outliers” means here:
- For each *song row* and each syllable, we compute the variance of that syllable’s phrase durations
  within that song (requires >=2 phrases; otherwise NaN).
- That gives a DISTRIBUTION of variances across songs in each group, which is what we run
  quantiles/fences/outlier probabilities on (and compare Pre vs Post).

Outputs add:
- variance_group_summary_df
- variance_comparison_df
- variance_outliers_df_ALL.csv
- variance figures (saved alongside duration figures)

Author: you + ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import math
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Optional: auto-merge builder
# ──────────────────────────────────────────────────────────────────────────────
try:
    from merge_annotations_from_split_songs import build_decoded_with_split_labels
except Exception as e:
    build_decoded_with_split_labels = None
    _MERGE_IMPORT_ERR = e


__all__ = [
    "OutlierDFResult",
    "run_phrase_duration_pre_vs_post_outlier_df",
    "run_batch_phrase_duration_outlier_df_from_excel",
]


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
TITLE_FS = 16
LABEL_FS = 13
TICK_FS = 11


def _pretty_axes(ax, x_rotation: int = 0):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    if x_rotation:
        for lab in ax.get_xticklabels():
            lab.set_rotation(x_rotation)
            lab.set_horizontalalignment("right")


def _savefig(fig: plt.Figure, outpath: Path, *, show: bool):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, transparent=False)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class OutlierDFResult:
    animal_id: str
    syllable_labels: List[str]

    # Duration (ms) tables
    group_summary_df: pd.DataFrame
    comparison_df: pd.DataFrame
    outliers_df: pd.DataFrame

    # Per-syllable basic stats (duration distribution)
    phrase_duration_stats_df: pd.DataFrame

    # NEW: Variance (ms^2) outlier tables (per-song variance distributions)
    variance_group_summary_df: pd.DataFrame
    variance_comparison_df: pd.DataFrame
    variance_outliers_df: pd.DataFrame

    # optional save paths
    group_summary_csv: Optional[Path] = None
    comparison_csv: Optional[Path] = None
    outliers_csv: Optional[Path] = None
    phrase_duration_stats_csv: Optional[Path] = None

    variance_group_summary_csv: Optional[Path] = None
    variance_comparison_csv: Optional[Path] = None
    variance_outliers_csv: Optional[Path] = None

    # figure paths
    figure_paths: Dict[str, Path] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities: dates, parsing, labels, durations extraction
# ──────────────────────────────────────────────────────────────────────────────
def _parse_date_like(s: Union[str, pd.Timestamp, None]) -> Union[pd.Timestamp, pd.NaT]:
    if s is None:
        return pd.NaT
    if isinstance(s, pd.Timestamp):
        return s
    s2 = str(s).replace(".", "-").replace("/", "-")
    return pd.to_datetime(s2, errors="coerce")


def _choose_datetime_series(df: pd.DataFrame) -> pd.Series:
    dt = df.get("Recording DateTime", pd.Series([pd.NaT] * len(df), index=df.index)).copy()
    for cand in ("recording_datetime", "creation_date"):
        if cand in df.columns:
            need = dt.isna()
            if need.any():
                dt.loc[need] = pd.to_datetime(df.loc[need, cand], errors="coerce")
    if "Date" in df.columns and "Time" in df.columns:
        need = dt.isna()
        if need.any():
            combo = pd.to_datetime(
                df.loc[need, "Date"].astype(str).str.replace(".", "-", regex=False)
                + " " + df.loc[need, "Time"].astype(str),
                errors="coerce",
            )
            dt.loc[need] = combo
    return dt


def _infer_animal_id(df: Optional[pd.DataFrame], decoded_path: Optional[Path]) -> str:
    if df is not None:
        for col in ["animal_id", "Animal", "Animal ID"]:
            if col in df.columns and pd.notna(df[col]).any():
                val = str(df[col].dropna().iloc[0]).strip()
                if val:
                    return val
        if "file_name" in df.columns and df["file_name"].notna().any():
            import re

            for s in df["file_name"].astype(str):
                m = re.search(r"(usa\d{4,6})", s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).upper()
    if decoded_path is not None:
        tok = decoded_path.stem.split("_")[0]
        if tok:
            return tok
    return "unknown_animal"


def _maybe_parse_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(obj)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None


def _row_ms_per_bin(row: pd.Series) -> float | None:
    for k in ["time_bin_ms", "timebin_ms", "bin_ms", "ms_per_bin"]:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return None


def _is_timebin_col(colname: str) -> bool:
    c = (colname or "").lower()
    return ("timebin" in c) or c.endswith("_bins") or ("bin" in c)


def _extract_durations_from_spans(spans, *, ms_per_bin: float | None, treat_as_bins: bool) -> List[float]:
    out: List[float] = []
    if spans is None:
        return out

    if (
        isinstance(spans, (list, tuple))
        and len(spans) == 2
        and all(isinstance(x, (int, float)) for x in spans)
    ):
        spans = [spans]
    if isinstance(spans, dict):
        spans = [spans]
    if not isinstance(spans, (list, tuple)):
        return out

    for item in spans:
        on = off = None
        using_bins = False

        if isinstance(item, dict):
            if "onset_ms" in item or "on" in item:
                on = item.get("onset_ms", item.get("on"))
                off = item.get("offset_ms", item.get("off"))
            elif "onset_bin" in item or "on_bin" in item:
                on = item.get("onset_bin", item.get("on_bin"))
                off = item.get("offset_bin", item.get("off_bin"))
                using_bins = True

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            on, off = item[:2]
            using_bins = False

        try:
            dur = float(off) - float(on)
            if dur < 0:
                continue
            if treat_as_bins or using_bins:
                if ms_per_bin:
                    dur *= float(ms_per_bin)
                else:
                    continue
            out.append(dur)
        except Exception:
            continue

    return out


def _collect_phrase_durations_per_song(row: pd.Series, spans_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    raw = row.get(spans_col, None)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return out

    d = _maybe_parse_dict(raw) if not isinstance(raw, dict) else raw
    if not isinstance(d, dict):
        return out

    mpb = _row_ms_per_bin(row)
    treat_as_bins = _is_timebin_col(spans_col)

    for lbl, spans in d.items():
        vals = _extract_durations_from_spans(spans, ms_per_bin=mpb, treat_as_bins=treat_as_bins)
        if vals:
            out[str(lbl)] = vals
    return out


def _find_best_spans_column(df: pd.DataFrame) -> str | None:
    for c in ["syllable_onsets_offsets_ms_dict", "syllable_onsets_offsets_ms", "onsets_offsets_ms_dict"]:
        if c in df.columns:
            return c
    for c in ["syllable_onsets_offsets_timebins", "syllable_onsets_offsets_timebins_dict"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_dict") and df[c].notna().any():
            return c
    return None


def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    labs: List[str] = []
    for d in df[dict_col].dropna():
        if isinstance(d, dict):
            labs.extend(list(map(str, d.keys())))
        else:
            dd = _maybe_parse_dict(d)
            if isinstance(dd, dict):
                labs.extend(list(map(str, dd.keys())))
    labs = list(dict.fromkeys(labs))
    try:
        as_int = [int(x) for x in labs]
        order = np.argsort(as_int)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)


def _build_durations_table(df: pd.DataFrame, labels: Sequence[str]) -> tuple[pd.DataFrame, str | None]:
    col = _find_best_spans_column(df)
    if col is None:
        return df.copy(), None
    per_song = df.apply(lambda r: _collect_phrase_durations_per_song(r, col), axis=1)
    out = df.copy()
    for lbl in labels:
        out[f"syllable_{lbl}_durations"] = per_song.apply(lambda d: d.get(str(lbl), []))
    return out, col


def _durations_array(df_subset: pd.DataFrame, lbl: str) -> np.ndarray:
    col = f"syllable_{lbl}_durations"
    if col not in df_subset.columns:
        return np.array([], dtype=float)
    s = pd.to_numeric(df_subset[col].explode(), errors="coerce").dropna()
    if s.empty:
        return np.array([], dtype=float)
    return s.to_numpy(dtype=float)


def _collect_all_durations(df_subset: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    parts = []
    for lbl in labels:
        arr = _durations_array(df_subset, str(lbl))
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: per-song variance (ms^2) per syllable
# ──────────────────────────────────────────────────────────────────────────────
def _variance_from_list(vals) -> float:
    """
    Variance of phrase durations WITHIN a song for one syllable.
    - If <2 phrases, returns NaN (not enough to define variance).
    """
    if not isinstance(vals, (list, tuple, np.ndarray)):
        return float("nan")
    if len(vals) < 2:
        return float("nan")
    arr = pd.to_numeric(pd.Series(list(vals)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(np.var(arr, ddof=1))


def _add_song_variance_columns(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """
    Adds columns:
      syllable_{lbl}_songvar_ms2 : variance within each song (row) for that syllable
      syllable_{lbl}_songn       : number of phrases used for that variance
    """
    out = df.copy()
    for lbl in labels:
        dcol = f"syllable_{lbl}_durations"
        if dcol not in out.columns:
            out[f"syllable_{lbl}_songvar_ms2"] = np.nan
            out[f"syllable_{lbl}_songn"] = 0
            continue
        out[f"syllable_{lbl}_songn"] = out[dcol].apply(lambda v: int(len(v)) if isinstance(v, (list, tuple)) else 0)
        out[f"syllable_{lbl}_songvar_ms2"] = out[dcol].apply(_variance_from_list)
    return out


def _songvar_array(df_subset: pd.DataFrame, lbl: str) -> np.ndarray:
    col = f"syllable_{lbl}_songvar_ms2"
    if col not in df_subset.columns:
        return np.array([], dtype=float)
    s = pd.to_numeric(df_subset[col], errors="coerce").dropna()
    if s.empty:
        return np.array([], dtype=float)
    return s.to_numpy(dtype=float)


def _collect_all_songvars(df_subset: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    parts = []
    for lbl in labels:
        arr = _songvar_array(df_subset, str(lbl))
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Basic per-syllable stats table (duration distribution)
# ──────────────────────────────────────────────────────────────────────────────
def _build_phrase_duration_stats_df(
    early_pre: pd.DataFrame,
    late_pre: pd.DataFrame,
    post_g: pd.DataFrame,
    labels: Sequence[str],
) -> pd.DataFrame:
    groups = [("Early Pre", early_pre), ("Late Pre", late_pre), ("Post", post_g)]
    rows = []
    for gname, df_subset in groups:
        if df_subset is None or df_subset.empty:
            continue
        for lbl in labels:
            arr = _durations_array(df_subset, str(lbl))
            n = int(arr.size)
            if n == 0:
                continue
            mean = float(arr.mean())
            median = float(np.median(arr))
            if n > 1:
                std = float(arr.std(ddof=1))
                var = float(arr.var(ddof=1))
                sem = float(std / math.sqrt(n))
            else:
                std = 0.0
                var = 0.0
                sem = 0.0
            rows.append(
                dict(
                    Group=gname,
                    Syllable=str(lbl),
                    N_phrases=n,
                    Mean_ms=mean,
                    SEM_ms=sem,
                    Median_ms=median,
                    Std_ms=std,
                    Variance_ms2=var,
                )
            )
    return pd.DataFrame(
        rows,
        columns=["Group", "Syllable", "N_phrases", "Mean_ms", "SEM_ms", "Median_ms", "Std_ms", "Variance_ms2"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Outlier / tail metrics
# ──────────────────────────────────────────────────────────────────────────────
def _safe_quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _compute_tail_outlier_metrics(
    arr: np.ndarray,
    *,
    tail_q1: float = 0.90,
    tail_q2: float = 0.95,
) -> Dict[str, float]:
    """
    Quantiles, tail spreads, tail-heaviness proxy, and outlier fence counts/fractions
    using fences derived from THIS arr.
    """
    out: Dict[str, float] = {}
    n = int(arr.size)
    out["n"] = float(n)
    if n == 0:
        for k in [
            "q25", "q50", "q75", "q90", "q95",
            "iqr",
            "upper_tail_spread_q90_q50",
            "upper_tail_spread_q95_q50",
            "upper_tail_spread_q90_q75",
            "upper_tail_spread_q95_q75",
            "tail_heaviness_proxy_q95",
            "mild_lower_fence", "mild_upper_fence",
            "extreme_lower_fence", "extreme_upper_fence",
            "n_mild_low", "n_mild_high", "n_mild_total",
            "n_extreme_low", "n_extreme_high", "n_extreme_total",
            "p_mild_low", "p_mild_high", "p_mild_total",
            "p_extreme_low", "p_extreme_high", "p_extreme_total",
        ]:
            out[k] = float("nan")
        return out

    q25 = _safe_quantile(arr, 0.25)
    q50 = _safe_quantile(arr, 0.50)
    q75 = _safe_quantile(arr, 0.75)
    q90 = _safe_quantile(arr, tail_q1)
    q95 = _safe_quantile(arr, tail_q2)

    iqr = q75 - q25
    out["q25"] = q25
    out["q50"] = q50
    out["q75"] = q75
    out["q90"] = q90
    out["q95"] = q95
    out["iqr"] = iqr

    out["upper_tail_spread_q90_q50"] = q90 - q50
    out["upper_tail_spread_q95_q50"] = q95 - q50
    out["upper_tail_spread_q90_q75"] = q90 - q75
    out["upper_tail_spread_q95_q75"] = q95 - q75

    denom = (q75 - q50)
    eps = 1e-12
    out["tail_heaviness_proxy_q95"] = (q95 - q75) / (denom if abs(denom) > eps else eps)

    mild_lower = q25 - 1.5 * iqr
    mild_upper = q75 + 1.5 * iqr
    extreme_lower = q25 - 3.0 * iqr
    extreme_upper = q75 + 3.0 * iqr

    out["mild_lower_fence"] = mild_lower
    out["mild_upper_fence"] = mild_upper
    out["extreme_lower_fence"] = extreme_lower
    out["extreme_upper_fence"] = extreme_upper

    mild_low = float(np.sum(arr < mild_lower))
    mild_high = float(np.sum(arr > mild_upper))
    extreme_low = float(np.sum(arr < extreme_lower))
    extreme_high = float(np.sum(arr > extreme_upper))

    out["n_mild_low"] = mild_low
    out["n_mild_high"] = mild_high
    out["n_mild_total"] = mild_low + mild_high

    out["n_extreme_low"] = extreme_low
    out["n_extreme_high"] = extreme_high
    out["n_extreme_total"] = extreme_low + extreme_high

    out["p_mild_low"] = (mild_low / n) if n else float("nan")
    out["p_mild_high"] = (mild_high / n) if n else float("nan")
    out["p_mild_total"] = ((mild_low + mild_high) / n) if n else float("nan")

    out["p_extreme_low"] = (extreme_low / n) if n else float("nan")
    out["p_extreme_high"] = (extreme_high / n) if n else float("nan")
    out["p_extreme_total"] = ((extreme_low + extreme_high) / n) if n else float("nan")

    return out


def _apply_fences_and_count(
    arr: np.ndarray,
    *,
    mild_lower: float,
    mild_upper: float,
    extreme_lower: float,
    extreme_upper: float,
) -> Dict[str, float]:
    """
    Count outliers in arr using PREDEFINED fences (e.g., fences from pooled pre).
    """
    out: Dict[str, float] = {}
    n = int(arr.size)
    out["n"] = float(n)
    if n == 0 or not np.isfinite(mild_lower) or not np.isfinite(mild_upper):
        for k in [
            "n_mild_low", "n_mild_high", "n_mild_total",
            "n_extreme_low", "n_extreme_high", "n_extreme_total",
            "p_mild_low", "p_mild_high", "p_mild_total",
            "p_extreme_low", "p_extreme_high", "p_extreme_total",
        ]:
            out[k] = float("nan")
        return out

    mild_low = float(np.sum(arr < mild_lower))
    mild_high = float(np.sum(arr > mild_upper))
    extreme_low = float(np.sum(arr < extreme_lower))
    extreme_high = float(np.sum(arr > extreme_upper))

    out["n_mild_low"] = mild_low
    out["n_mild_high"] = mild_high
    out["n_mild_total"] = mild_low + mild_high

    out["n_extreme_low"] = extreme_low
    out["n_extreme_high"] = extreme_high
    out["n_extreme_total"] = extreme_low + extreme_high

    out["p_mild_low"] = (mild_low / n) if n else float("nan")
    out["p_mild_high"] = (mild_high / n) if n else float("nan")
    out["p_mild_total"] = ((mild_low + mild_high) / n) if n else float("nan")

    out["p_extreme_low"] = (extreme_low / n) if n else float("nan")
    out["p_extreme_high"] = (extreme_high / n) if n else float("nan")
    out["p_extreme_total"] = ((extreme_low + extreme_high) / n) if n else float("nan")

    return out


def _risk_metrics(p_post: float, p_pre: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not (np.isfinite(p_post) and np.isfinite(p_pre)):
        out["rr"] = float("nan")
        out["rd"] = float("nan")
        out["log_rr"] = float("nan")
        return out
    out["rd"] = float(p_post - p_pre)
    if p_pre == 0.0:
        rr = float("inf") if p_post > 0 else 1.0
    else:
        rr = float(p_post / p_pre)
    out["rr"] = rr
    out["log_rr"] = float(np.log(rr)) if np.isfinite(rr) and rr > 0 else float("nan")
    return out


def _bootstrap_rr_rd(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: Optional[int] = 0,
    continuity: float = 0.5,
) -> Dict[str, float]:
    """
    Bootstrap CI for RR/RD for EXTREME TOTAL exceedance.
    Each draw:
      - resample pre/post
      - derive extreme fences from bootstrap PRE
      - count extreme exceedance in both
      - p = (k + continuity) / (n + 2*continuity)
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, float] = {}

    n_pre = int(pre_arr.size)
    n_post = int(post_arr.size)
    if n_pre < 5 or n_post < 5:
        for k in ["rr_med", "rr_ci_lo", "rr_ci_hi", "rd_med", "rd_ci_lo", "rd_ci_hi"]:
            out[k] = float("nan")
        return out

    rr_list: List[float] = []
    rd_list: List[float] = []

    for _ in range(int(n_boot)):
        bs_pre = rng.choice(pre_arr, size=n_pre, replace=True)
        bs_post = rng.choice(post_arr, size=n_post, replace=True)

        q25 = _safe_quantile(bs_pre, 0.25)
        q75 = _safe_quantile(bs_pre, 0.75)
        iqr = q75 - q25
        extreme_lower = q25 - 3.0 * iqr
        extreme_upper = q75 + 3.0 * iqr

        k_pre = float(np.sum((bs_pre < extreme_lower) | (bs_pre > extreme_upper)))
        k_post = float(np.sum((bs_post < extreme_lower) | (bs_post > extreme_upper)))

        p_pre = float((k_pre + continuity) / (n_pre + 2.0 * continuity))
        p_post = float((k_post + continuity) / (n_post + 2.0 * continuity))

        rd_list.append(p_post - p_pre)
        rr_list.append(p_post / p_pre if p_pre > 0 else float("inf"))

    rr = np.asarray(rr_list, dtype=float)
    rd = np.asarray(rd_list, dtype=float)

    out["rr_med"] = float(np.nanmedian(rr))
    out["rr_ci_lo"] = float(np.nanpercentile(rr, 2.5))
    out["rr_ci_hi"] = float(np.nanpercentile(rr, 97.5))

    out["rd_med"] = float(np.nanmedian(rd))
    out["rd_ci_lo"] = float(np.nanpercentile(rd, 2.5))
    out["rd_ci_hi"] = float(np.nanpercentile(rd, 97.5))
    return out


def _bootstrap_delta_quantile(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    *,
    tau: float,
    n_boot: int = 2000,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out: Dict[str, float] = {}

    n_pre = int(pre_arr.size)
    n_post = int(post_arr.size)
    if n_pre < 5 or n_post < 5:
        out["delta_q"] = float("nan")
        out["delta_q_ci_lo"] = float("nan")
        out["delta_q_ci_hi"] = float("nan")
        return out

    deltas: List[float] = []
    for _ in range(int(n_boot)):
        bs_pre = rng.choice(pre_arr, size=n_pre, replace=True)
        bs_post = rng.choice(post_arr, size=n_post, replace=True)
        deltas.append(_safe_quantile(bs_post, tau) - _safe_quantile(bs_pre, tau))

    d = np.asarray(deltas, dtype=float)
    out["delta_q"] = float(np.nanmedian(d))
    out["delta_q_ci_lo"] = float(np.nanpercentile(d, 2.5))
    out["delta_q_ci_hi"] = float(np.nanpercentile(d, 97.5))
    return out


def _try_quantile_regression_effect(pre_arr: np.ndarray, post_arr: np.ndarray, *, tau: float) -> Dict[str, float]:
    out: Dict[str, float] = {"qr_coef": float("nan"), "qr_pvalue": float("nan")}
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        return out

    y = np.concatenate([pre_arr, post_arr], axis=0)
    treat = np.concatenate([np.zeros_like(pre_arr), np.ones_like(post_arr)], axis=0)
    if y.size < 30:
        return out

    X = sm.add_constant(treat)
    try:
        mod = sm.QuantReg(y, X)
        res = mod.fit(q=tau)
        out["qr_coef"] = float(res.params[1])
        if hasattr(res, "pvalues") and len(res.pvalues) > 1:
            out["qr_pvalue"] = float(res.pvalues[1])
        return out
    except Exception:
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def _sort_labels(labels: Sequence[str]) -> List[str]:
    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))
    return sorted([str(x) for x in labels], key=_key)


def _plot_quantiles_per_syllable(
    group_summary_df: pd.DataFrame,
    labels: Sequence[str],
    *,
    groups: Sequence[str],
    tail_q1: float,
    tail_q2: float,
    outpath: Path,
    title: str,
    ylabel: str,
    show: bool,
):
    labels_sorted = _sort_labels(labels)
    x = np.arange(len(labels_sorted), dtype=float)

    nG = max(1, len(groups))
    offsets = np.linspace(-0.25, 0.25, nG) if nG > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels_sorted)), 6))

    for gi, g in enumerate(groups):
        sub = group_summary_df[group_summary_df["Group"] == g].copy().set_index("Syllable")

        q25 = np.array([sub.get("within_q25", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
        q50 = np.array([sub.get("within_q50", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
        q75 = np.array([sub.get("within_q75", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
        q90 = np.array([sub.get("within_q90", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
        q95 = np.array([sub.get("within_q95", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)

        xx = x + offsets[gi]

        for i in range(len(labels_sorted)):
            if np.isfinite(q25[i]) and np.isfinite(q75[i]):
                ax.plot([xx[i], xx[i]], [q25[i], q75[i]], linewidth=2.0, alpha=0.9)

        ax.scatter(xx, q50, s=24, marker="o", alpha=0.95, label=f"{g} median")
        ax.scatter(xx, q90, s=20, marker="^", alpha=0.75, label=f"{g} Q{int(tail_q1*100)}")
        ax.scatter(xx, q95, s=20, marker="s", alpha=0.75, label=f"{g} Q{int(tail_q2*100)}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Syllable")
    ax.set_title(title, fontsize=TITLE_FS)
    _pretty_axes(ax, x_rotation=90 if len(labels_sorted) > 20 else 0)
    ax.legend(frameon=False, ncols=2, fontsize=9)
    _savefig(fig, outpath, show=show)


def _plot_metric_by_group(
    group_summary_df: pd.DataFrame,
    labels: Sequence[str],
    *,
    groups: Sequence[str],
    metric_col: str,
    ylabel: str,
    outpath: Path,
    title: str,
    show: bool,
    y0_line: bool = False,
):
    labels_sorted = _sort_labels(labels)
    x = np.arange(len(labels_sorted), dtype=float)
    nG = max(1, len(groups))
    offsets = np.linspace(-0.25, 0.25, nG) if nG > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels_sorted)), 5.5))

    for gi, g in enumerate(groups):
        sub = group_summary_df[group_summary_df["Group"] == g].copy().set_index("Syllable")
        yy = np.array([sub.get(metric_col, pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
        ax.scatter(x + offsets[gi], yy, s=26, alpha=0.85, label=g)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Syllable")
    ax.set_title(title, fontsize=TITLE_FS)
    if y0_line:
        ax.axhline(0.0, linewidth=1, linestyle="--", alpha=0.5)
    _pretty_axes(ax, x_rotation=90 if len(labels_sorted) > 20 else 0)
    ax.legend(frameon=False)
    _savefig(fig, outpath, show=show)


def _plot_rr_with_ci(
    comparison_df: pd.DataFrame,
    labels: Sequence[str],
    *,
    comparison_name: str,
    outpath: Path,
    title: str,
    show: bool,
    log_scale: bool = True,
):
    labels_sorted = _sort_labels(labels)
    sub = comparison_df[comparison_df["Comparison"] == comparison_name].copy().set_index("Syllable")

    rr = np.array([sub.get("rr_med", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    lo = np.array([sub.get("rr_ci_lo", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    hi = np.array([sub.get("rr_ci_hi", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)

    x = np.arange(len(labels_sorted), dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels_sorted)), 5.5))
    yerr = np.vstack([rr - lo, hi - rr])
    ax.errorbar(x, rr, yerr=yerr, fmt="o", capsize=3, alpha=0.85)
    ax.axhline(1.0, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel("Risk Ratio (Post / Pre)")
    ax.set_xlabel("Syllable")
    ax.set_title(title, fontsize=TITLE_FS)
    if log_scale and np.nanmin(rr) > 0:
        ax.set_yscale("log")
    _pretty_axes(ax, x_rotation=90 if len(labels_sorted) > 20 else 0)
    _savefig(fig, outpath, show=show)


def _plot_rd_with_ci(
    comparison_df: pd.DataFrame,
    labels: Sequence[str],
    *,
    comparison_name: str,
    outpath: Path,
    title: str,
    show: bool,
):
    labels_sorted = _sort_labels(labels)
    sub = comparison_df[comparison_df["Comparison"] == comparison_name].copy().set_index("Syllable")

    rd = np.array([sub.get("rd_med", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    lo = np.array([sub.get("rd_ci_lo", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    hi = np.array([sub.get("rd_ci_hi", pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)

    x = np.arange(len(labels_sorted), dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels_sorted)), 5.5))
    yerr = np.vstack([rd - lo, hi - rd])
    ax.errorbar(x, rd, yerr=yerr, fmt="o", capsize=3, alpha=0.85)
    ax.axhline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel("Risk Difference (Post − Pre)")
    ax.set_xlabel("Syllable")
    ax.set_title(title, fontsize=TITLE_FS)
    _pretty_axes(ax, x_rotation=90 if len(labels_sorted) > 20 else 0)
    _savefig(fig, outpath, show=show)


def _plot_delta_q_with_ci(
    comparison_df: pd.DataFrame,
    labels: Sequence[str],
    *,
    comparison_name: str,
    delta_col: str,
    lo_col: str,
    hi_col: str,
    outpath: Path,
    title: str,
    ylabel: str,
    show: bool,
):
    labels_sorted = _sort_labels(labels)
    sub = comparison_df[comparison_df["Comparison"] == comparison_name].copy().set_index("Syllable")

    dq = np.array([sub.get(delta_col, pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    lo = np.array([sub.get(lo_col, pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)
    hi = np.array([sub.get(hi_col, pd.Series()).get(lbl, np.nan) for lbl in labels_sorted], dtype=float)

    x = np.arange(len(labels_sorted), dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels_sorted)), 5.5))
    yerr = np.vstack([dq - lo, hi - dq])
    ax.errorbar(x, dq, yerr=yerr, fmt="o", capsize=3, alpha=0.85)
    ax.axhline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Syllable")
    ax.set_title(title, fontsize=TITLE_FS)
    _pretty_axes(ax, x_rotation=90 if len(labels_sorted) > 20 else 0)
    _savefig(fig, outpath, show=show)


def _plot_pooled_quantiles_and_fences(
    *,
    pooled_metrics: Dict[str, Dict[str, float]],
    outpath: Path,
    title: str,
    ylabel: str,
    show: bool,
):
    groups = list(pooled_metrics.keys())
    x = np.arange(len(groups), dtype=float)

    q25 = np.array([pooled_metrics[g].get("q25", np.nan) for g in groups], dtype=float)
    q50 = np.array([pooled_metrics[g].get("q50", np.nan) for g in groups], dtype=float)
    q75 = np.array([pooled_metrics[g].get("q75", np.nan) for g in groups], dtype=float)
    q90 = np.array([pooled_metrics[g].get("q90", np.nan) for g in groups], dtype=float)
    q95 = np.array([pooled_metrics[g].get("q95", np.nan) for g in groups], dtype=float)

    mild_lo = np.array([pooled_metrics[g].get("mild_lower_fence", np.nan) for g in groups], dtype=float)
    mild_hi = np.array([pooled_metrics[g].get("mild_upper_fence", np.nan) for g in groups], dtype=float)
    ext_lo = np.array([pooled_metrics[g].get("extreme_lower_fence", np.nan) for g in groups], dtype=float)
    ext_hi = np.array([pooled_metrics[g].get("extreme_upper_fence", np.nan) for g in groups], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.8))

    for i in range(len(groups)):
        if np.isfinite(q25[i]) and np.isfinite(q75[i]):
            ax.plot([x[i], x[i]], [q25[i], q75[i]], linewidth=4, alpha=0.9)
    ax.scatter(x, q50, s=40, marker="o", label="Median")
    ax.scatter(x, q90, s=34, marker="^", label="Q90")
    ax.scatter(x, q95, s=34, marker="s", label="Q95")

    for i in range(len(groups)):
        if np.isfinite(mild_lo[i]) and np.isfinite(mild_hi[i]):
            ax.plot([x[i] - 0.18, x[i] + 0.18], [mild_lo[i], mild_lo[i]], linestyle="--", linewidth=1.2, alpha=0.7)
            ax.plot([x[i] - 0.18, x[i] + 0.18], [mild_hi[i], mild_hi[i]], linestyle="--", linewidth=1.2, alpha=0.7)
        if np.isfinite(ext_lo[i]) and np.isfinite(ext_hi[i]):
            ax.plot([x[i] - 0.18, x[i] + 0.18], [ext_lo[i], ext_lo[i]], linestyle=":", linewidth=1.4, alpha=0.8)
            ax.plot([x[i] - 0.18, x[i] + 0.18], [ext_hi[i], ext_hi[i]], linestyle=":", linewidth=1.4, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_FS)
    _pretty_axes(ax, x_rotation=0)
    ax.legend(frameon=False)
    _savefig(fig, outpath, show=show)


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_phrase_duration_pre_vs_post_outlier_df(
    *,
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,
    decoded_database_json: Optional[Union[str, Path]] = None,
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    treatment_date: Union[str, pd.Timestamp] = None,
    grouping_mode: str = "explicit",
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    tail_q1: float = 0.90,
    tail_q2: float = 0.95,
    n_boot: int = 2000,
    bootstrap_seed: Optional[int] = 0,
    do_quantile_regression: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    save_csvs: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
    plot_dir_name: str = "outlier_figures",
    animal_id_override: Optional[str] = None,
) -> OutlierDFResult:

    # 1) Decide source DF
    df = None
    if premerged_annotations_df is not None:
        df = premerged_annotations_df.copy()
    elif premerged_annotations_path is not None:
        p = Path(premerged_annotations_path)
        if not p.exists():
            raise FileNotFoundError(f"premerged_annotations_path not found: {p}")
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".json", ".ndjson"}:
            try:
                df = pd.read_json(p)
            except ValueError:
                df = pd.read_json(p, lines=True)
        else:
            raise ValueError(f"Unsupported file type for premerged_annotations_path: {p.suffix}")
    else:
        if build_decoded_with_split_labels is None:
            raise ImportError(
                "merge_annotations_from_split_songs.build_decoded_with_split_labels could not be imported.\n"
                f"Original import error: {_MERGE_IMPORT_ERR}"
            )
        if decoded_database_json is None or song_detection_json is None:
            raise ValueError("To auto-merge, you must provide both decoded_database_json and song_detection_json.")
        decoded_database_json = Path(decoded_database_json)
        song_detection_json = Path(song_detection_json)

        ann = build_decoded_with_split_labels(
            decoded_database_json=decoded_database_json,
            song_detection_json=song_detection_json,
            only_song_present=True,
            compute_durations=True,
            add_recording_datetime=True,
            songs_only=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=segment_index_offset,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
        )
        df = ann.annotations_appended_df.copy()

    if df is None or df.empty:
        raise ValueError("No data available after reading/merging annotations.")

    # 2) Output dir + animal id
    decoded_path = Path(decoded_database_json) if decoded_database_json else None
    if output_dir is None:
        base = (decoded_path.parent if decoded_path is not None else Path.cwd())
        output_dir = base / "phrase_durations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    animal_id = animal_id_override or _infer_animal_id(df, decoded_path)

    # 3) Datetime + sort
    dt = _choose_datetime_series(df)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    # 4) Labels
    if restrict_to_labels is None:
        col_hint = _find_best_spans_column(df)
        if col_hint is None:
            raise ValueError("Could not locate a syllable spans column (e.g., 'syllable_onsets_offsets_ms_dict').")
        labels = _collect_unique_labels_sorted(df, col_hint)
    else:
        labels = [str(x) for x in restrict_to_labels]

    # 5) Build per-syllable duration lists
    df, spans_col = _build_durations_table(df, labels)
    if spans_col is None:
        raise ValueError("Could not build duration columns: no spans column found.")

    # NEW: add per-song variance columns
    df = _add_song_variance_columns(df, labels)

    # 6) Split by treatment date
    t_date = _parse_date_like(treatment_date)
    if pd.isna(t_date):
        raise ValueError("Valid treatment_date is required (e.g., '2025-05-22').")
    pre_all = df[df["_dt"] < t_date].copy()
    post_all = df[df["_dt"] >= t_date].copy()

    # 7) Balanced groups
    if grouping_mode == "explicit":
        early_n, late_n, post_n = map(int, [early_group_size, late_group_size, post_group_size])
        late_pre = pre_all.tail(late_n)
        early_pre = pre_all.iloc[max(0, len(pre_all) - late_n - early_n): max(0, len(pre_all) - late_n)]
        post_g = post_all.head(post_n)
    else:
        n = min(len(pre_all) // 2, len(post_all))
        if n <= 0:
            late_pre = pre_all.iloc[0:0]
            early_pre = pre_all.iloc[0:0]
            post_g = post_all.iloc[0:0]
        else:
            late_pre = pre_all.tail(n)
            early_pre = pre_all.iloc[len(pre_all) - 2 * n: len(pre_all) - n]
            post_g = post_all.head(n)

    # 8) Basic per-syllable duration stats
    phrase_duration_stats_df = _build_phrase_duration_stats_df(early_pre, late_pre, post_g, labels)

    # ──────────────────────────────────────────────────────────────────────────
    # A) DURATION outlier tables (same as before)
    # ──────────────────────────────────────────────────────────────────────────
    groups: List[Tuple[str, pd.DataFrame]] = [
        ("Early Pre", early_pre),
        ("Late Pre", late_pre),
        ("Post", post_g),
        ("Pre(Pooled)", pre_all),
        ("Post(All)", post_all),
    ]

    pooled_pre_fences: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        arr_pre = _durations_array(pre_all, str(lbl))
        m_pre = _compute_tail_outlier_metrics(arr_pre, tail_q1=tail_q1, tail_q2=tail_q2)
        pooled_pre_fences[str(lbl)] = dict(
            mild_lower_fence=m_pre.get("mild_lower_fence", float("nan")),
            mild_upper_fence=m_pre.get("mild_upper_fence", float("nan")),
            extreme_lower_fence=m_pre.get("extreme_lower_fence", float("nan")),
            extreme_upper_fence=m_pre.get("extreme_upper_fence", float("nan")),
        )

    group_rows: List[Dict[str, object]] = []
    for gname, gdf in groups:
        for lbl in labels:
            arr = _durations_array(gdf, str(lbl))
            base = {"AnimalID": animal_id, "Group": gname, "Syllable": str(lbl)}

            n = int(arr.size)
            if n == 0:
                base.update(dict(N_phrases=0))
            else:
                base.update(dict(N_phrases=n, Mean_ms=float(arr.mean()), Median_ms=float(np.median(arr))))

            tm = _compute_tail_outlier_metrics(arr, tail_q1=tail_q1, tail_q2=tail_q2)
            within = {f"within_{k}": v for k, v in tm.items() if k != "n"}

            f = pooled_pre_fences.get(str(lbl), {})
            applied = _apply_fences_and_count(
                arr,
                mild_lower=float(f.get("mild_lower_fence", float("nan"))),
                mild_upper=float(f.get("mild_upper_fence", float("nan"))),
                extreme_lower=float(f.get("extreme_lower_fence", float("nan"))),
                extreme_upper=float(f.get("extreme_upper_fence", float("nan"))),
            )
            applied = {f"pooledpre_{k}": v for k, v in applied.items() if k != "n"}
            base_f = {f"pooledpre_{k}": v for k, v in f.items()}

            row = {}
            row.update(base)
            row.update(within)
            row.update(base_f)
            row.update(applied)
            group_rows.append(row)

    group_summary_df = pd.DataFrame(group_rows)

    comparisons: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("Post vs Pre(Pooled)", post_all, pre_all),
        ("Post vs Early Pre", post_g, early_pre),
        ("Post vs Late Pre", post_g, late_pre),
        ("Late Pre vs Early Pre", late_pre, early_pre),
    ]

    comp_rows: List[Dict[str, object]] = []
    for cname, post_df, pre_df in comparisons:
        for lbl in labels:
            arr_post = _durations_array(post_df, str(lbl))
            arr_pre = _durations_array(pre_df, str(lbl))

            f = pooled_pre_fences.get(str(lbl), {})
            mild_lower = float(f.get("mild_lower_fence", float("nan")))
            mild_upper = float(f.get("mild_upper_fence", float("nan")))
            extreme_lower = float(f.get("extreme_lower_fence", float("nan")))
            extreme_upper = float(f.get("extreme_upper_fence", float("nan")))

            pre_ev = _apply_fences_and_count(arr_pre, mild_lower=mild_lower, mild_upper=mild_upper,
                                            extreme_lower=extreme_lower, extreme_upper=extreme_upper)
            post_ev = _apply_fences_and_count(arr_post, mild_lower=mild_lower, mild_upper=mild_upper,
                                             extreme_lower=extreme_lower, extreme_upper=extreme_upper)

            row: Dict[str, object] = dict(
                AnimalID=animal_id,
                Comparison=cname,
                Syllable=str(lbl),
                N_pre=int(arr_pre.size),
                N_post=int(arr_post.size),
                p_pre_extreme_total=pre_ev.get("p_extreme_total", float("nan")),
                p_post_extreme_total=post_ev.get("p_extreme_total", float("nan")),
            )

            bs = _bootstrap_rr_rd(arr_pre, arr_post, n_boot=n_boot, seed=bootstrap_seed, continuity=0.5)
            row.update(bs)

            dq90 = _bootstrap_delta_quantile(arr_pre, arr_post, tau=tail_q1, n_boot=n_boot, seed=bootstrap_seed)
            dq95 = _bootstrap_delta_quantile(arr_pre, arr_post, tau=tail_q2, n_boot=n_boot, seed=bootstrap_seed)
            row.update(
                {
                    f"delta_q{int(tail_q1*100)}": dq90["delta_q"],
                    f"delta_q{int(tail_q1*100)}_ci_lo": dq90["delta_q_ci_lo"],
                    f"delta_q{int(tail_q1*100)}_ci_hi": dq90["delta_q_ci_hi"],
                    f"delta_q{int(tail_q2*100)}": dq95["delta_q"],
                    f"delta_q{int(tail_q2*100)}_ci_lo": dq95["delta_q_ci_lo"],
                    f"delta_q{int(tail_q2*100)}_ci_hi": dq95["delta_q_ci_hi"],
                }
            )

            if do_quantile_regression and cname == "Post vs Pre(Pooled)":
                qr90 = _try_quantile_regression_effect(arr_pre, arr_post, tau=tail_q1)
                qr95 = _try_quantile_regression_effect(arr_pre, arr_post, tau=tail_q2)
                row.update(
                    {
                        f"qr_coef_q{int(tail_q1*100)}": qr90["qr_coef"],
                        f"qr_pvalue_q{int(tail_q1*100)}": qr90["qr_pvalue"],
                        f"qr_coef_q{int(tail_q2*100)}": qr95["qr_coef"],
                        f"qr_pvalue_q{int(tail_q2*100)}": qr95["qr_pvalue"],
                    }
                )
            comp_rows.append(row)

    comparison_df = pd.DataFrame(comp_rows)

    group_out = group_summary_df.copy()
    group_out.insert(0, "RowType", "duration_group_metrics")
    comp_out = comparison_df.copy()
    comp_out.insert(0, "RowType", "duration_comparison_metrics")
    outliers_df = pd.concat([group_out, comp_out], ignore_index=True, sort=False)

    # ──────────────────────────────────────────────────────────────────────────
    # B) NEW: VARIANCE (ms^2) outlier tables using per-song variance distributions
    # ──────────────────────────────────────────────────────────────────────────
    pooled_pre_fences_var: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        var_pre = _songvar_array(pre_all, str(lbl))
        m_pre = _compute_tail_outlier_metrics(var_pre, tail_q1=tail_q1, tail_q2=tail_q2)
        pooled_pre_fences_var[str(lbl)] = dict(
            mild_lower_fence=m_pre.get("mild_lower_fence", float("nan")),
            mild_upper_fence=m_pre.get("mild_upper_fence", float("nan")),
            extreme_lower_fence=m_pre.get("extreme_lower_fence", float("nan")),
            extreme_upper_fence=m_pre.get("extreme_upper_fence", float("nan")),
        )

    var_group_rows: List[Dict[str, object]] = []
    for gname, gdf in groups:
        for lbl in labels:
            arr = _songvar_array(gdf, str(lbl))  # <-- per-song variances
            base = {"AnimalID": animal_id, "Group": gname, "Syllable": str(lbl), "Measure": "song_variance_ms2"}

            n = int(arr.size)
            base["N_songs_with_var"] = n
            base["Mean_var_ms2"] = float(arr.mean()) if n else float("nan")
            base["Median_var_ms2"] = float(np.median(arr)) if n else float("nan")

            tm = _compute_tail_outlier_metrics(arr, tail_q1=tail_q1, tail_q2=tail_q2)
            within = {f"within_{k}": v for k, v in tm.items() if k != "n"}

            f = pooled_pre_fences_var.get(str(lbl), {})
            applied = _apply_fences_and_count(
                arr,
                mild_lower=float(f.get("mild_lower_fence", float("nan"))),
                mild_upper=float(f.get("mild_upper_fence", float("nan"))),
                extreme_lower=float(f.get("extreme_lower_fence", float("nan"))),
                extreme_upper=float(f.get("extreme_upper_fence", float("nan"))),
            )
            applied = {f"pooledpre_{k}": v for k, v in applied.items() if k != "n"}
            base_f = {f"pooledpre_{k}": v for k, v in f.items()}

            row = {}
            row.update(base)
            row.update(within)
            row.update(base_f)
            row.update(applied)
            var_group_rows.append(row)

    variance_group_summary_df = pd.DataFrame(var_group_rows)

    var_comp_rows: List[Dict[str, object]] = []
    for cname, post_df, pre_df in comparisons:
        for lbl in labels:
            arr_post = _songvar_array(post_df, str(lbl))
            arr_pre = _songvar_array(pre_df, str(lbl))

            row: Dict[str, object] = dict(
                AnimalID=animal_id,
                Comparison=cname,
                Syllable=str(lbl),
                Measure="song_variance_ms2",
                N_pre=int(arr_pre.size),
                N_post=int(arr_post.size),
            )

            # Bootstrap RR/RD for EXTREME TOTAL (variance outliers)
            bs = _bootstrap_rr_rd(arr_pre, arr_post, n_boot=n_boot, seed=bootstrap_seed, continuity=0.5)
            row.update(bs)

            # ΔQ90 and ΔQ95 for variance
            dq90 = _bootstrap_delta_quantile(arr_pre, arr_post, tau=tail_q1, n_boot=n_boot, seed=bootstrap_seed)
            dq95 = _bootstrap_delta_quantile(arr_pre, arr_post, tau=tail_q2, n_boot=n_boot, seed=bootstrap_seed)
            row.update(
                {
                    f"delta_q{int(tail_q1*100)}": dq90["delta_q"],
                    f"delta_q{int(tail_q1*100)}_ci_lo": dq90["delta_q_ci_lo"],
                    f"delta_q{int(tail_q1*100)}_ci_hi": dq90["delta_q_ci_hi"],
                    f"delta_q{int(tail_q2*100)}": dq95["delta_q"],
                    f"delta_q{int(tail_q2*100)}_ci_lo": dq95["delta_q_ci_lo"],
                    f"delta_q{int(tail_q2*100)}_ci_hi": dq95["delta_q_ci_hi"],
                }
            )

            if do_quantile_regression and cname == "Post vs Pre(Pooled)":
                qr90 = _try_quantile_regression_effect(arr_pre, arr_post, tau=tail_q1)
                qr95 = _try_quantile_regression_effect(arr_pre, arr_post, tau=tail_q2)
                row.update(
                    {
                        f"qr_coef_q{int(tail_q1*100)}": qr90["qr_coef"],
                        f"qr_pvalue_q{int(tail_q1*100)}": qr90["qr_pvalue"],
                        f"qr_coef_q{int(tail_q2*100)}": qr95["qr_coef"],
                        f"qr_pvalue_q{int(tail_q2*100)}": qr95["qr_pvalue"],
                    }
                )

            var_comp_rows.append(row)

    variance_comparison_df = pd.DataFrame(var_comp_rows)

    var_group_out = variance_group_summary_df.copy()
    var_group_out.insert(0, "RowType", "variance_group_metrics")
    var_comp_out = variance_comparison_df.copy()
    var_comp_out.insert(0, "RowType", "variance_comparison_metrics")
    variance_outliers_df = pd.concat([var_group_out, var_comp_out], ignore_index=True, sort=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Saving CSVs
    # ──────────────────────────────────────────────────────────────────────────
    group_csv = comp_csv = out_csv = stats_csv = None
    v_group_csv = v_comp_csv = v_out_csv = None

    if save_csvs:
        group_csv = output_dir / f"{animal_id}_group_summary_outlier_metrics_DURATION.csv"
        comp_csv = output_dir / f"{animal_id}_prepost_comparison_outlier_metrics_DURATION.csv"
        out_csv = output_dir / f"{animal_id}_OUTLIERS_df_ALL_DURATION.csv"
        stats_csv = output_dir / f"{animal_id}_phrase_duration_stats_df.csv"

        v_group_csv = output_dir / f"{animal_id}_group_summary_outlier_metrics_VARIANCE.csv"
        v_comp_csv = output_dir / f"{animal_id}_prepost_comparison_outlier_metrics_VARIANCE.csv"
        v_out_csv = output_dir / f"{animal_id}_OUTLIERS_df_ALL_VARIANCE.csv"

        group_summary_df.to_csv(group_csv, index=False)
        comparison_df.to_csv(comp_csv, index=False)
        outliers_df.to_csv(out_csv, index=False)
        phrase_duration_stats_df.to_csv(stats_csv, index=False)

        variance_group_summary_df.to_csv(v_group_csv, index=False)
        variance_comparison_df.to_csv(v_comp_csv, index=False)
        variance_outliers_df.to_csv(v_out_csv, index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Figures (duration + variance)
    # ──────────────────────────────────────────────────────────────────────────
    fig_paths: Dict[str, Path] = {}
    if make_plots:
        fig_dir = output_dir / plot_dir_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_groups = ["Early Pre", "Late Pre", "Post"]
        main_comp = "Post vs Pre(Pooled)"

        q90 = int(tail_q1 * 100)
        q95 = int(tail_q2 * 100)

        # ----- Duration figures -----
        p1 = fig_dir / f"{animal_id}_DUR_quantiles_per_syllable.png"
        _plot_quantiles_per_syllable(
            group_summary_df, labels, groups=plot_groups, tail_q1=tail_q1, tail_q2=tail_q2,
            outpath=p1,
            title=f"{animal_id} — Duration quantiles per syllable (IQR+median+tails)",
            ylabel="Phrase Duration (ms)",
            show=show_plots,
        )
        fig_paths["dur_quantiles"] = p1

        p2 = fig_dir / f"{animal_id}_DUR_tail_spread_Q{q95}_minus_Q50.png"
        _plot_metric_by_group(
            group_summary_df, labels, groups=plot_groups,
            metric_col="within_upper_tail_spread_q95_q50",
            ylabel=f"Q{q95} − Q50 (ms)",
            outpath=p2,
            title=f"{animal_id} — Duration upper-tail spread per syllable",
            show=show_plots,
            y0_line=True,
        )
        fig_paths["dur_tail_spread"] = p2

        p3 = fig_dir / f"{animal_id}_DUR_tail_heaviness_proxy.png"
        _plot_metric_by_group(
            group_summary_df, labels, groups=plot_groups,
            metric_col="within_tail_heaviness_proxy_q95",
            ylabel="(Q95 − Q75) / (Q75 − Q50)",
            outpath=p3,
            title=f"{animal_id} — Duration tail-heaviness proxy per syllable",
            show=show_plots,
        )
        fig_paths["dur_tail_heaviness"] = p3

        p4 = fig_dir / f"{animal_id}_DUR_p_extreme_total_pooledpre.png"
        _plot_metric_by_group(
            group_summary_df, labels, groups=plot_groups,
            metric_col="pooledpre_p_extreme_total",
            ylabel="Probability",
            outpath=p4,
            title=f"{animal_id} — Duration extreme-outlier probability (pooled-pre fences)",
            show=show_plots,
        )
        fig_paths["dur_p_extreme_total"] = p4

        p5 = fig_dir / f"{animal_id}_DUR_RR_extreme_total_bootstrap.png"
        _plot_rr_with_ci(
            comparison_df, labels, comparison_name=main_comp,
            outpath=p5,
            title=f"{animal_id} — Duration RR(extreme total) + bootstrap CI\n({main_comp})",
            show=show_plots,
            log_scale=True,
        )
        fig_paths["dur_rr_extreme_boot"] = p5

        p6 = fig_dir / f"{animal_id}_DUR_RD_extreme_total_bootstrap.png"
        _plot_rd_with_ci(
            comparison_df, labels, comparison_name=main_comp,
            outpath=p6,
            title=f"{animal_id} — Duration RD(extreme total) + bootstrap CI\n({main_comp})",
            show=show_plots,
        )
        fig_paths["dur_rd_extreme_boot"] = p6

        p7 = fig_dir / f"{animal_id}_DUR_delta_Q{q90}_bootstrap.png"
        _plot_delta_q_with_ci(
            comparison_df, labels, comparison_name=main_comp,
            delta_col=f"delta_q{q90}", lo_col=f"delta_q{q90}_ci_lo", hi_col=f"delta_q{q90}_ci_hi",
            outpath=p7,
            title=f"{animal_id} — Duration ΔQ{q90} (Post−Pre) + bootstrap CI\n({main_comp})",
            ylabel=f"ΔQ{q90} (ms)",
            show=show_plots,
        )
        fig_paths[f"dur_delta_q{q90}"] = p7

        p8 = fig_dir / f"{animal_id}_DUR_delta_Q{q95}_bootstrap.png"
        _plot_delta_q_with_ci(
            comparison_df, labels, comparison_name=main_comp,
            delta_col=f"delta_q{q95}", lo_col=f"delta_q{q95}_ci_lo", hi_col=f"delta_q{q95}_ci_hi",
            outpath=p8,
            title=f"{animal_id} — Duration ΔQ{q95} (Post−Pre) + bootstrap CI\n({main_comp})",
            ylabel=f"ΔQ{q95} (ms)",
            show=show_plots,
        )
        fig_paths[f"dur_delta_q{q95}"] = p8

        pooled_metrics_dur = {}
        for gname, gdf in [("Early Pre", early_pre), ("Late Pre", late_pre), ("Post", post_g), ("Pre(Pooled)", pre_all), ("Post(All)", post_all)]:
            pooled_arr = _collect_all_durations(gdf, labels)
            pooled_metrics_dur[gname] = _compute_tail_outlier_metrics(pooled_arr, tail_q1=tail_q1, tail_q2=tail_q2)

        p9 = fig_dir / f"{animal_id}_DUR_POOLED_quantiles_and_fences.png"
        _plot_pooled_quantiles_and_fences(
            pooled_metrics=pooled_metrics_dur,
            outpath=p9,
            title=f"{animal_id} — Duration pooled quantiles + fences by group",
            ylabel="Phrase Duration (ms)",
            show=show_plots,
        )
        fig_paths["dur_pooled_quantiles_fences"] = p9

        # ----- NEW: Variance figures (per-song variance distribution) -----
        v1 = fig_dir / f"{animal_id}_VAR_quantiles_per_syllable.png"
        _plot_quantiles_per_syllable(
            variance_group_summary_df, labels, groups=plot_groups, tail_q1=tail_q1, tail_q2=tail_q2,
            outpath=v1,
            title=f"{animal_id} — Song-variance quantiles per syllable (IQR+median+tails)",
            ylabel="Per-song variance of phrase duration (ms²)",
            show=show_plots,
        )
        fig_paths["var_quantiles"] = v1

        v2 = fig_dir / f"{animal_id}_VAR_tail_spread_Q{q95}_minus_Q50.png"
        _plot_metric_by_group(
            variance_group_summary_df, labels, groups=plot_groups,
            metric_col="within_upper_tail_spread_q95_q50",
            ylabel=f"Q{q95} − Q50 (ms²)",
            outpath=v2,
            title=f"{animal_id} — Song-variance upper-tail spread per syllable",
            show=show_plots,
            y0_line=True,
        )
        fig_paths["var_tail_spread"] = v2

        v3 = fig_dir / f"{animal_id}_VAR_tail_heaviness_proxy.png"
        _plot_metric_by_group(
            variance_group_summary_df, labels, groups=plot_groups,
            metric_col="within_tail_heaviness_proxy_q95",
            ylabel="(Q95 − Q75) / (Q75 − Q50)",
            outpath=v3,
            title=f"{animal_id} — Song-variance tail-heaviness proxy per syllable",
            show=show_plots,
        )
        fig_paths["var_tail_heaviness"] = v3

        v4 = fig_dir / f"{animal_id}_VAR_p_extreme_total_pooledpre.png"
        _plot_metric_by_group(
            variance_group_summary_df, labels, groups=plot_groups,
            metric_col="pooledpre_p_extreme_total",
            ylabel="Probability",
            outpath=v4,
            title=f"{animal_id} — Song-variance extreme-outlier probability (pooled-pre fences)",
            show=show_plots,
        )
        fig_paths["var_p_extreme_total"] = v4

        v5 = fig_dir / f"{animal_id}_VAR_RR_extreme_total_bootstrap.png"
        _plot_rr_with_ci(
            variance_comparison_df, labels, comparison_name=main_comp,
            outpath=v5,
            title=f"{animal_id} — Song-variance RR(extreme total) + bootstrap CI\n({main_comp})",
            show=show_plots,
            log_scale=True,
        )
        fig_paths["var_rr_extreme_boot"] = v5

        v6 = fig_dir / f"{animal_id}_VAR_RD_extreme_total_bootstrap.png"
        _plot_rd_with_ci(
            variance_comparison_df, labels, comparison_name=main_comp,
            outpath=v6,
            title=f"{animal_id} — Song-variance RD(extreme total) + bootstrap CI\n({main_comp})",
            show=show_plots,
        )
        fig_paths["var_rd_extreme_boot"] = v6

        v7 = fig_dir / f"{animal_id}_VAR_delta_Q{q90}_bootstrap.png"
        _plot_delta_q_with_ci(
            variance_comparison_df, labels, comparison_name=main_comp,
            delta_col=f"delta_q{q90}", lo_col=f"delta_q{q90}_ci_lo", hi_col=f"delta_q{q90}_ci_hi",
            outpath=v7,
            title=f"{animal_id} — Song-variance ΔQ{q90} (Post−Pre) + bootstrap CI\n({main_comp})",
            ylabel=f"ΔQ{q90} (ms²)",
            show=show_plots,
        )
        fig_paths[f"var_delta_q{q90}"] = v7

        v8 = fig_dir / f"{animal_id}_VAR_delta_Q{q95}_bootstrap.png"
        _plot_delta_q_with_ci(
            variance_comparison_df, labels, comparison_name=main_comp,
            delta_col=f"delta_q{q95}", lo_col=f"delta_q{q95}_ci_lo", hi_col=f"delta_q{q95}_ci_hi",
            outpath=v8,
            title=f"{animal_id} — Song-variance ΔQ{q95} (Post−Pre) + bootstrap CI\n({main_comp})",
            ylabel=f"ΔQ{q95} (ms²)",
            show=show_plots,
        )
        fig_paths[f"var_delta_q{q95}"] = v8

        pooled_metrics_var = {}
        for gname, gdf in [("Early Pre", early_pre), ("Late Pre", late_pre), ("Post", post_g), ("Pre(Pooled)", pre_all), ("Post(All)", post_all)]:
            pooled_arr = _collect_all_songvars(gdf, labels)
            pooled_metrics_var[gname] = _compute_tail_outlier_metrics(pooled_arr, tail_q1=tail_q1, tail_q2=tail_q2)

        v9 = fig_dir / f"{animal_id}_VAR_POOLED_quantiles_and_fences.png"
        _plot_pooled_quantiles_and_fences(
            pooled_metrics=pooled_metrics_var,
            outpath=v9,
            title=f"{animal_id} — Song-variance pooled quantiles + fences by group",
            ylabel="Per-song variance of phrase duration (ms²)",
            show=show_plots,
        )
        fig_paths["var_pooled_quantiles_fences"] = v9

    return OutlierDFResult(
        animal_id=animal_id,
        syllable_labels=[str(x) for x in labels],
        group_summary_df=group_summary_df,
        comparison_df=comparison_df,
        outliers_df=outliers_df,
        phrase_duration_stats_df=phrase_duration_stats_df,
        variance_group_summary_df=variance_group_summary_df,
        variance_comparison_df=variance_comparison_df,
        variance_outliers_df=variance_outliers_df,
        group_summary_csv=group_csv,
        comparison_csv=comp_csv,
        outliers_csv=out_csv,
        phrase_duration_stats_csv=stats_csv,
        variance_group_summary_csv=v_group_csv,
        variance_comparison_csv=v_comp_csv,
        variance_outliers_csv=v_out_csv,
        figure_paths=fig_paths,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch wrapper: Excel-driven runs
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_phrase_duration_outlier_df_from_excel(
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    tail_q1: float = 0.90,
    tail_q2: float = 0.95,
    n_boot: int = 2000,
    bootstrap_seed: Optional[int] = 0,
    do_quantile_regression: bool = False,
    save_csvs: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, OutlierDFResult]:
    excel_path = Path(excel_path)
    json_root = Path(json_root)

    meta_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if id_col not in meta_df.columns:
        raise ValueError(f"Column '{id_col}' not found in Excel file: {excel_path}")
    if treatment_date_col not in meta_df.columns:
        raise ValueError(f"Column '{treatment_date_col}' not found in Excel file: {excel_path}")

    animal_to_tdate: Dict[str, Union[str, pd.Timestamp, None]] = {}
    for aid, group in meta_df.groupby(id_col):
        vals = group[treatment_date_col].dropna().unique()
        tdate = vals[0] if len(vals) > 0 else None
        animal_to_tdate[str(aid)] = tdate

    def _find_json_for_animal(
        root: Path,
        animal_id: str,
        decoded_suffix: str = "decoded_database.json",
        detect_suffix: str = "song_detection.json",
    ) -> Tuple[Optional[Path], Optional[Path]]:
        decoded_candidates = [p for p in root.rglob(f"*{animal_id}*{decoded_suffix}") if not p.name.startswith("._")]
        detect_candidates = [p for p in root.rglob(f"*{animal_id}*{detect_suffix}") if not p.name.startswith("._")]
        decoded_path = decoded_candidates[0] if decoded_candidates else None
        detect_path = detect_candidates[0] if detect_candidates else None
        return decoded_path, detect_path

    restrict_str = [str(x) for x in restrict_to_labels] if restrict_to_labels is not None else None

    results: Dict[str, OutlierDFResult] = {}
    for animal_id, tdate in animal_to_tdate.items():
        if tdate is None or (isinstance(tdate, float) and pd.isna(tdate)):
            print(f"[WARN] {animal_id}: no valid treatment date in Excel, skipping.")
            continue

        decoded_path, detect_path = _find_json_for_animal(json_root, animal_id)
        if decoded_path is None or detect_path is None:
            print(
                f"[WARN] {animal_id}: could not find both JSONs under {json_root}.\n"
                f"       decoded: {decoded_path}\n"
                f"       detect : {detect_path}"
            )
            continue

        outdir = decoded_path.parent / "figures" / "phrase_durations"
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN] {animal_id} | treatment_date={tdate} | decoded={decoded_path.name} | detect={detect_path.name}")

        res = run_phrase_duration_pre_vs_post_outlier_df(
            decoded_database_json=decoded_path,
            song_detection_json=detect_path,
            max_gap_between_song_segments=500,
            segment_index_offset=0,
            merge_repeated_syllables=True,
            repeat_gap_ms=10.0,
            repeat_gap_inclusive=False,
            treatment_date=tdate,
            grouping_mode=grouping_mode,
            early_group_size=early_group_size,
            late_group_size=late_group_size,
            post_group_size=post_group_size,
            restrict_to_labels=restrict_str,
            tail_q1=tail_q1,
            tail_q2=tail_q2,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
            do_quantile_regression=do_quantile_regression,
            output_dir=outdir,
            save_csvs=save_csvs,
            make_plots=make_plots,
            show_plots=show_plots,
            animal_id_override=animal_id,
        )
        results[animal_id] = res

    return results


"""
Example interactive usage (single animal):

from pathlib import Path
import phrase_duration_pre_vs_post_outlier_df as pdo

detect  = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443_song_detection.json")
decoded = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443_decoded_database.json")

outdir = decoded.parent / "figures" / "phrase_durations"
outdir.mkdir(parents=True, exist_ok=True)

res = pdo.run_phrase_duration_pre_vs_post_outlier_df(
    decoded_database_json=decoded,
    song_detection_json=detect,
    treatment_date="2024-04-30",
    grouping_mode="auto_balance",
    tail_q1=0.90,
    tail_q2=0.95,
    n_boot=2000,
    bootstrap_seed=0,
    output_dir=outdir,
    save_csvs=True,
    make_plots=True,
    show_plots=True,
)

print("Variance group summary head:")
print(res.variance_group_summary_df.head())
print("Saved variance figures:", [k for k in res.figure_paths if k.startswith("var_")])

"""
