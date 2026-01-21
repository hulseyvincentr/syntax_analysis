#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phrase_duration_stats_outliers.py

Tail / outlier statistics for compiled phrase-duration stats.

This version matches the columns in your helper output, e.g.:
  - 'Animal ID'
  - 'Group'              (e.g., 'Early Pre', 'Late Pre', 'Post')
  - 'Syllable'
  - 'N_phrases'
  - 'Mean_ms'
  - 'Variance_ms2'
  (plus other helper-derived columns that we ignore)

Core idea
---------
Within each (Animal ID, Group), you have one row per syllable (or unit). That
gives you a distribution across syllables for that bird+epoch.

We compute outlier/tail metrics on:
  1) log-variance of phrase durations  (log10(Variance_ms2) by default)
  2) CV = sqrt(Variance_ms2) / Mean_ms

We compute metrics two ways:
  - UNWEIGHTED: each syllable counts equally (rows)
  - WEIGHTED: syllables weighted by N_phrases (optional but often useful)

Outputs
-------
Saved into output_dir (default: next to compiled file):
  - logvar_long_flagged_unweighted.csv
  - logvar_long_flagged_weighted.csv
  - cv_long_flagged_unweighted.csv
  - cv_long_flagged_weighted.csv
  - per_bird_group_outlier_metrics_logvar.csv
  - per_bird_group_outlier_metrics_cv.csv
  - per_bird_prepost_outlier_summary.csv
  - group_comparisons_outlier_metrics.csv   (optional; requires SciPy)

Notes
-----
- “Outliers” here mean outliers across the *rows in your compiled table*
  (typically syllables) within each bird+group.
- If you want outliers across songs/bouts/days, you’ll need a song-level table.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List

import numpy as np
import pandas as pd

# Optional SciPy for group comparisons
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------
def _load_phrase_stats(
    compiled_stats_path: Union[str, Path],
    compiled_format: Optional[str] = None,
) -> pd.DataFrame:
    """Load compiled phrase-duration stats into a DataFrame."""
    path = Path(compiled_stats_path)

    if compiled_format is None:
        suf = path.suffix.lower()
        if suf == ".csv":
            compiled_format = "csv"
        elif suf == ".json":
            compiled_format = "json"
        elif suf == ".npz":
            compiled_format = "npz"
        else:
            raise ValueError(f"Cannot infer compiled_format from suffix {path.suffix!r}.")

    if compiled_format == "csv":
        df = pd.read_csv(path)
    elif compiled_format == "json":
        df = pd.read_json(path)
    elif compiled_format == "npz":
        arr = np.load(path, allow_pickle=True)
        if "phrase_stats" not in arr:
            raise KeyError("NPZ must contain an array named 'phrase_stats'.")
        df = pd.DataFrame(arr["phrase_stats"])
    else:
        raise ValueError(f"Unsupported compiled_format={compiled_format!r}")

    return df


def _load_metadata(
    metadata_excel_path: Union[str, Path],
    sheet_name: Union[int, str] = "metadata",
    id_col: str = "Animal ID",
) -> pd.DataFrame:
    """
    Load metadata from Excel.

    Required columns:
      - id_col (default 'Animal ID')
      - 'Treatment type'

    Returns a DataFrame with unique Animal IDs.
    """
    metadata_excel_path = Path(metadata_excel_path)
    meta_df = pd.read_excel(metadata_excel_path, sheet_name=sheet_name)

    if id_col not in meta_df.columns:
        raise KeyError(f"Column {id_col!r} not found in metadata Excel.")
    if "Treatment type" not in meta_df.columns:
        raise KeyError("Expected 'Treatment type' column in metadata Excel.")

    meta_df = meta_df[[id_col, "Treatment type"]].drop_duplicates(subset=id_col)
    return meta_df


def _classify_treatment_type(raw: Any) -> str:
    """
    Collapse free-text treatment descriptions into:
        'nma'   -> bilateral NMA lesion injections (experimental)
        'sham'  -> saline sham injections (control)
        'other' -> anything else
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "other"
    s = str(raw).strip().lower()
    if "sham" in s or "saline" in s:
        return "sham"
    if "nma" in s:
        return "nma"
    return "other"


# -----------------------------------------------------------------------------
# Weighted quantiles + weighted MAD helpers
# -----------------------------------------------------------------------------
def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Weighted quantiles for 1D arrays.
    quantiles in [0,1]. Returns same shape as quantiles.

    This uses the "CDF of weights" approach (no interpolation beyond step CDF).
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]

    if v.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]

    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    # CDF in [0,1]
    cdf = cw / total
    out = np.empty_like(quantiles, dtype=float)
    for i, q in enumerate(quantiles):
        if q <= 0:
            out[i] = v[0]
        elif q >= 1:
            out[i] = v[-1]
        else:
            idx = np.searchsorted(cdf, q, side="left")
            idx = min(max(idx, 0), v.size - 1)
            out[i] = v[idx]
    return out


def _median_and_mad(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Return (median, MAD). If weights provided, returns weighted median and weighted MAD,
    where weighted MAD = weighted median of |x - weighted_median|.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")

    if weights is None:
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        return med, mad

    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(w) & (w > 0)
    x = np.asarray(values, dtype=float)[mask]
    w = w[mask]
    if x.size == 0:
        return float("nan"), float("nan")

    med = float(_weighted_quantile(x, np.array([0.5]), w)[0])
    abs_dev = np.abs(x - med)
    mad = float(_weighted_quantile(abs_dev, np.array([0.5]), w)[0])
    return med, mad


# -----------------------------------------------------------------------------
# Build long-form distributions (per bird × group) for logvar and CV
# -----------------------------------------------------------------------------
def build_long_logvar(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syll_col: str = "Syllable",
    n_col: str = "N_phrases",
    var_col: str = "Variance_ms2",
    mean_col: str = "Mean_ms",
    keep_groups: Optional[List[str]] = None,
    log_mode: str = "log10",
    clip_lower: float = 1e-12,
) -> pd.DataFrame:
    """
    Long table with one row per syllable row:
      id_col, group_col, syll_col, n_col, mean_col, var_col, value_logvar
    """
    need = [id_col, group_col, syll_col, n_col, var_col]
    missing = [c for c in need if c not in phrase_df.columns]
    if missing:
        raise KeyError(f"Missing columns for logvar long build: {missing}")

    df = phrase_df.copy()
    if keep_groups is not None:
        df = df[df[group_col].isin(keep_groups)].copy()

    df = df.dropna(subset=[id_col, group_col, var_col]).copy()

    v = pd.to_numeric(df[var_col], errors="coerce").astype(float).clip(lower=clip_lower)
    if log_mode.lower() == "log10":
        df["value_logvar"] = np.log10(v)
        df.attrs["log_label"] = "log₁₀ variance (ms²)"
    elif log_mode.lower() in {"ln", "log"}:
        df["value_logvar"] = np.log(v)
        df.attrs["log_label"] = "ln variance (ms²)"
    else:
        raise ValueError(f"Unsupported log_mode={log_mode!r}; use 'log10' or 'ln'.")

    # Keep mean if present (optional, but helpful for debugging)
    cols = [id_col, group_col, syll_col, n_col]
    if mean_col in df.columns:
        cols.append(mean_col)
    cols += [var_col, "value_logvar"]
    return df[cols].copy()


def build_long_cv(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syll_col: str = "Syllable",
    n_col: str = "N_phrases",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    keep_groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Long table with one row per syllable row:
      id_col, group_col, syll_col, n_col, mean_col, var_col, value_cv
    """
    need = [id_col, group_col, syll_col, n_col, mean_col, var_col]
    missing = [c for c in need if c not in phrase_df.columns]
    if missing:
        raise KeyError(f"Missing columns for CV long build: {missing}")

    df = phrase_df.copy()
    if keep_groups is not None:
        df = df[df[group_col].isin(keep_groups)].copy()

    df = df.dropna(subset=[id_col, group_col, mean_col, var_col]).copy()

    mean_vals = pd.to_numeric(df[mean_col], errors="coerce").astype(float)
    var_vals = pd.to_numeric(df[var_col], errors="coerce").astype(float).clip(lower=0.0)

    mean_safe = mean_vals.replace(0.0, np.nan)
    df["value_cv"] = np.sqrt(var_vals) / mean_safe
    df = df.dropna(subset=["value_cv"]).copy()

    cols = [id_col, group_col, syll_col, n_col, mean_col, var_col, "value_cv"]
    return df[cols].copy()


# -----------------------------------------------------------------------------
# Outlier/tail summaries for a (possibly weighted) distribution
# -----------------------------------------------------------------------------
def summarize_distribution(
    values: pd.Series,
    weights: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Summarize a distribution into tail/outlier metrics.

    If weights provided, quantiles/median/MAD are weighted, and we also compute
    "weight fraction" of outliers.

    Returned metrics are used downstream for Tukey flags + exceedance rates.
    """
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n_rows = int(x.size)

    if weights is None:
        w = None
        wsum = float("nan")
    else:
        w_raw = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
        # Align weights to original index length; easiest: re-coerce from series and mask later
        # (We’ll re-mask using the same finite mask by rebuilding arrays below.)
        # We just compute w after removing non-finite x by re-filtering from the original series.
        x_full = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x_full)
        w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)[mask]
        w = w[np.isfinite(w) & (w > 0)]
        # If weights got out of sync (rare), fall back to None
        if w.size != n_rows:
            w = None
            wsum = float("nan")
        else:
            wsum = float(np.sum(w))

    if n_rows == 0:
        return pd.Series(
            {
                "n_rows": 0,
                "weight_sum": np.nan,
                "mean": np.nan,
                "std_pop": np.nan,
                "q1": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "q90": np.nan,
                "q95": np.nan,
                "q99": np.nan,
                "iqr": np.nan,
                "tukey_lo_1p5": np.nan,
                "tukey_hi_1p5": np.nan,
                "tukey_lo_3": np.nan,
                "tukey_hi_3": np.nan,
                "n_high_outliers_1p5": 0,
                "frac_high_outliers_1p5": np.nan,
                "weight_frac_high_outliers_1p5": np.nan,
                "mad": np.nan,
                "n_modz_abs_gt_3p5": 0,
                "weight_frac_modz_abs_gt_3p5": np.nan,
                "tail_ratio_95_75_over_75_50": np.nan,
            }
        )

    if w is None:
        mean = float(np.nanmean(x))
        std_pop = float(np.nanstd(x, ddof=0))
        q1, q50, q75, q90, q95, q99 = [float(np.nanquantile(x, q)) for q in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)]
        med, mad = _median_and_mad(x, weights=None)
    else:
        mean = float(np.average(x, weights=w))
        std_pop = float(np.sqrt(np.average((x - mean) ** 2, weights=w)))
        q1, q50, q75, q90, q95, q99 = _weighted_quantile(x, np.array([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]), w).astype(float).tolist()
        med, mad = _median_and_mad(x, weights=w)

    iqr = q75 - q1
    tukey_lo_1p5 = q1 - 1.5 * iqr
    tukey_hi_1p5 = q75 + 1.5 * iqr
    tukey_lo_3 = q1 - 3.0 * iqr
    tukey_hi_3 = q75 + 3.0 * iqr

    is_high_1p5 = x > tukey_hi_1p5
    n_high_1p5 = int(np.sum(is_high_1p5))
    frac_high_1p5 = float(n_high_1p5 / n_rows) if n_rows > 0 else float("nan")

    if w is None:
        weight_frac_high_1p5 = np.nan
    else:
        weight_frac_high_1p5 = float(np.sum(w[is_high_1p5]) / np.sum(w)) if np.sum(w) > 0 else float("nan")

    # Modified z-score: 0.6745*(x - median)/MAD
    if np.isfinite(mad) and mad > 0:
        modz = 0.6745 * (x - med) / mad
        is_modz = np.abs(modz) > 3.5
        n_modz_abs = int(np.sum(is_modz))
        if w is None:
            weight_frac_modz = np.nan
        else:
            weight_frac_modz = float(np.sum(w[is_modz]) / np.sum(w)) if np.sum(w) > 0 else float("nan")
    else:
        n_modz_abs = 0
        weight_frac_modz = np.nan

    denom = (q75 - q50)
    tail_ratio = float((q95 - q75) / denom) if np.isfinite(denom) and denom > 0 else float("nan")

    return pd.Series(
        {
            "n_rows": n_rows,
            "weight_sum": wsum,
            "mean": mean,
            "std_pop": std_pop,
            "q1": q1,
            "q50": q50,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "iqr": iqr,
            "tukey_lo_1p5": tukey_lo_1p5,
            "tukey_hi_1p5": tukey_hi_1p5,
            "tukey_lo_3": tukey_lo_3,
            "tukey_hi_3": tukey_hi_3,
            "n_high_outliers_1p5": n_high_1p5,
            "frac_high_outliers_1p5": frac_high_1p5,
            "weight_frac_high_outliers_1p5": weight_frac_high_1p5,
            "mad": mad,
            "n_modz_abs_gt_3p5": n_modz_abs,
            "weight_frac_modz_abs_gt_3p5": weight_frac_modz,
            "tail_ratio_95_75_over_75_50": tail_ratio,
        }
    )


def compute_bird_group_metrics(
    long_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    weighting_label: str = "unweighted",
) -> pd.DataFrame:
    """
    Compute per (bird, group) outlier metrics for value_col.
    If weight_col is provided, metrics use weights.
    """
    need = [id_col, group_col, value_col]
    missing = [c for c in need if c not in long_df.columns]
    if missing:
        raise KeyError(f"Missing columns for metrics: {missing}")

    if weight_col is not None and weight_col not in long_df.columns:
        raise KeyError(f"weight_col={weight_col!r} not found in long_df.")

    if weight_col is None:
        out = (
            long_df.groupby([id_col, group_col])[value_col]
            .apply(summarize_distribution)
            .reset_index()
        )
    else:
        def _apply(g: pd.DataFrame) -> pd.Series:
            return summarize_distribution(g[value_col], weights=g[weight_col])
        out = long_df.groupby([id_col, group_col]).apply(_apply).reset_index()

    out["weighting"] = weighting_label
    return out


def add_outlier_flags_and_modz(
    long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
    weight_col: Optional[str],
    prefix: str,
    weighting_label: str,
) -> pd.DataFrame:
    """
    Merge fences from metrics_df onto long_df and flag outlier rows + compute modz per row.
    """
    fence_cols = ["tukey_lo_1p5", "tukey_hi_1p5", "tukey_lo_3", "tukey_hi_3", "q50", "mad"]
    m = metrics_df[metrics_df["weighting"] == weighting_label].copy()

    need = [id_col, group_col] + fence_cols
    missing = [c for c in need if c not in m.columns]
    if missing:
        raise KeyError(f"metrics_df missing fence/center columns: {missing}")

    merged = long_df.merge(
        m[[id_col, group_col] + fence_cols],
        on=[id_col, group_col],
        how="left",
    )

    v = pd.to_numeric(merged[value_col], errors="coerce").astype(float)

    merged[f"{prefix}_is_low_outlier_1p5"] = v < merged["tukey_lo_1p5"]
    merged[f"{prefix}_is_high_outlier_1p5"] = v > merged["tukey_hi_1p5"]
    merged[f"{prefix}_is_low_outlier_3"] = v < merged["tukey_lo_3"]
    merged[f"{prefix}_is_high_outlier_3"] = v > merged["tukey_hi_3"]

    # Modified z-score using group median (q50) and MAD from metrics_df
    med = pd.to_numeric(merged["q50"], errors="coerce").astype(float)
    mad = pd.to_numeric(merged["mad"], errors="coerce").astype(float)

    merged[f"{prefix}_modz"] = np.nan
    ok = np.isfinite(v) & np.isfinite(med) & np.isfinite(mad) & (mad > 0)
    merged.loc[ok, f"{prefix}_modz"] = 0.6745 * (v[ok] - med[ok]) / mad[ok]
    merged[f"{prefix}_modz_abs_gt_3p5"] = np.abs(merged[f"{prefix}_modz"]) > 3.5

    merged["weighting"] = weighting_label
    return merged


def compute_post_exceedance_rates(
    long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
    weight_col: Optional[str],
    pre_group: str,
    post_group: str,
    prefix: str,
    weighting_label: str,
) -> pd.DataFrame:
    """
    For each bird, compute fraction of Post values exceeding that bird's Pre thresholds:
      - Pre q95
      - Pre Tukey high fence (1.5*IQR)

    Returns per-bird columns:
      {prefix}_post_exceed_pre_q95_rate
      {prefix}_post_exceed_pre_hi_fence_rate
      and weighted equivalents if weight_col is provided:
      {prefix}_post_exceed_pre_q95_weight_frac
      {prefix}_post_exceed_pre_hi_fence_weight_frac
    """
    m = metrics_df[metrics_df["weighting"] == weighting_label].copy()
    pre_m = m[m[group_col] == pre_group].copy()
    if pre_m.empty:
        raise RuntimeError(f"No metrics rows for pre_group={pre_group!r}, weighting={weighting_label!r}.")

    pre_thr = pre_m[[id_col, "q95", "tukey_hi_1p5"]].rename(
        columns={
            "q95": f"{prefix}_pre_q95",
            "tukey_hi_1p5": f"{prefix}_pre_hi_fence_1p5",
        }
    )

    post = long_df[long_df[group_col] == post_group].copy()
    if post.empty:
        raise RuntimeError(f"No rows in long_df for post_group={post_group!r}.")

    post = post.merge(pre_thr, on=id_col, how="left")

    def _rate(d: pd.DataFrame) -> pd.Series:
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        vals = vals[mask]
        if vals.size == 0:
            return pd.Series(
                {
                    f"{prefix}_post_exceed_pre_q95_rate": np.nan,
                    f"{prefix}_post_exceed_pre_hi_fence_rate": np.nan,
                    f"{prefix}_post_exceed_pre_q95_weight_frac": np.nan,
                    f"{prefix}_post_exceed_pre_hi_fence_weight_frac": np.nan,
                }
            )

        pre_q95 = float(d[f"{prefix}_pre_q95"].iloc[0]) if np.isfinite(d[f"{prefix}_pre_q95"].iloc[0]) else np.nan
        pre_hi = float(d[f"{prefix}_pre_hi_fence_1p5"].iloc[0]) if np.isfinite(d[f"{prefix}_pre_hi_fence_1p5"].iloc[0]) else np.nan

        r_q95 = float(np.mean(vals > pre_q95)) if np.isfinite(pre_q95) else np.nan
        r_hi = float(np.mean(vals > pre_hi)) if np.isfinite(pre_hi) else np.nan

        if weight_col is None or weight_col not in d.columns:
            return pd.Series(
                {
                    f"{prefix}_post_exceed_pre_q95_rate": r_q95,
                    f"{prefix}_post_exceed_pre_hi_fence_rate": r_hi,
                    f"{prefix}_post_exceed_pre_q95_weight_frac": np.nan,
                    f"{prefix}_post_exceed_pre_hi_fence_weight_frac": np.nan,
                }
            )

        w = pd.to_numeric(d[weight_col], errors="coerce").to_numpy(dtype=float)
        w = w[mask]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        wsum = float(np.sum(w))
        if wsum <= 0 or (not np.isfinite(pre_q95)) or (not np.isfinite(pre_hi)):
            return pd.Series(
                {
                    f"{prefix}_post_exceed_pre_q95_rate": r_q95,
                    f"{prefix}_post_exceed_pre_hi_fence_rate": r_hi,
                    f"{prefix}_post_exceed_pre_q95_weight_frac": np.nan,
                    f"{prefix}_post_exceed_pre_hi_fence_weight_frac": np.nan,
                }
            )

        wf_q95 = float(np.sum(w[vals > pre_q95]) / wsum) if np.isfinite(pre_q95) else np.nan
        wf_hi = float(np.sum(w[vals > pre_hi]) / wsum) if np.isfinite(pre_hi) else np.nan

        return pd.Series(
            {
                f"{prefix}_post_exceed_pre_q95_rate": r_q95,
                f"{prefix}_post_exceed_pre_hi_fence_rate": r_hi,
                f"{prefix}_post_exceed_pre_q95_weight_frac": wf_q95,
                f"{prefix}_post_exceed_pre_hi_fence_weight_frac": wf_hi,
            }
        )

    rates = post.groupby(id_col).apply(_rate).reset_index()
    rates["weighting"] = weighting_label
    rates = rates.merge(pre_thr, on=id_col, how="left")
    return rates


# -----------------------------------------------------------------------------
# Optional: group comparisons on outlier metrics (NMA vs sham)
# -----------------------------------------------------------------------------
def compare_groups_on_metrics(
    per_bird_df: pd.DataFrame,
    *,
    treat_col: str = "treat_class",
    weighting_col: str = "weighting",
    weighting_value: str = "unweighted",
    groups: Tuple[str, str] = ("nma", "sham"),
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare NMA vs sham on selected metric columns using:
      - Welch t-test (two-sided)
      - Mann–Whitney U (two-sided)

    Returns a tidy DataFrame with one row per metric.
    Requires SciPy.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available; cannot run group comparison tests.")

    g1, g2 = groups
    if treat_col not in per_bird_df.columns:
        raise KeyError(f"Missing treat_col={treat_col!r} in per_bird_df.")
    if weighting_col not in per_bird_df.columns:
        raise KeyError(f"Missing weighting_col={weighting_col!r} in per_bird_df.")

    df = per_bird_df.copy()
    df = df[df[weighting_col] == weighting_value].copy()
    df = df[df[treat_col].isin([g1, g2])].copy()

    if metrics is None:
        metrics = [
            "logvar_post_exceed_pre_q95_rate",
            "logvar_post_exceed_pre_hi_fence_rate",
            "cv_post_exceed_pre_q95_rate",
            "cv_post_exceed_pre_hi_fence_rate",
            "delta_logvar_q95",
            "delta_logvar_frac_high_outliers_1p5",
            "delta_cv_q95",
            "delta_cv_frac_high_outliers_1p5",
        ]

    rows: List[Dict[str, Any]] = []
    for m in metrics:
        if m not in df.columns:
            continue

        x1 = pd.to_numeric(df[df[treat_col] == g1][m], errors="coerce").to_numpy(dtype=float)
        x2 = pd.to_numeric(df[df[treat_col] == g2][m], errors="coerce").to_numpy(dtype=float)
        x1 = x1[np.isfinite(x1)]
        x2 = x2[np.isfinite(x2)]

        if x1.size < 2 or x2.size < 2:
            rows.append(
                {
                    "weighting": weighting_value,
                    "metric": m,
                    f"n_{g1}": int(x1.size),
                    f"n_{g2}": int(x2.size),
                    f"mean_{g1}": float(np.nanmean(x1)) if x1.size else np.nan,
                    f"mean_{g2}": float(np.nanmean(x2)) if x2.size else np.nan,
                    "welch_t_p": np.nan,
                    "mwu_p": np.nan,
                }
            )
            continue

        t_stat, t_p = _scipy_stats.ttest_ind(x1, x2, equal_var=False)
        u_stat, u_p = _scipy_stats.mannwhitneyu(x1, x2, alternative="two-sided")

        rows.append(
            {
                "weighting": weighting_value,
                "metric": m,
                f"n_{g1}": int(x1.size),
                f"n_{g2}": int(x2.size),
                f"mean_{g1}": float(np.nanmean(x1)),
                f"mean_{g2}": float(np.nanmean(x2)),
                "welch_t_p": float(t_p),
                "mwu_p": float(u_p),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Public wrapper
# -----------------------------------------------------------------------------
def run_phrase_outlier_stats(
    compiled_stats_path: Union[str, Path],
    metadata_excel_path: Union[str, Path],
    *,
    compiled_format: Optional[str] = None,
    metadata_sheet_name: Union[int, str] = "metadata",
    output_dir: Optional[Union[str, Path]] = None,
    # Column defaults that match your helper output:
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syll_col: str = "Syllable",
    n_col: str = "N_phrases",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    # Pre/post groups:
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    # If None: keep ALL groups present in the compiled table.
    keep_groups: Optional[List[str]] = None,
    log_mode: str = "log10",
    run_group_tests: bool = True,
) -> Dict[str, Any]:
    """
    High-level helper:
      1) Load compiled phrase stats + metadata
      2) Build long-form logvar and CV tables
      3) Compute per-bird-per-group tail/outlier metrics (unweighted + weighted by N_phrases)
      4) Flag outlier rows (Tukey + modz) in long tables
      5) Compute per-bird post exceedance rates over pre thresholds
      6) Compute per-bird pre/post delta summaries (q95 + outlier fractions)
      7) Save outputs to output_dir
      8) Optionally run NMA vs sham group comparisons (SciPy)
    """
    compiled_stats_path = Path(compiled_stats_path)
    if output_dir is None:
        output_dir = compiled_stats_path.parent / "phrase_duration_outlier_stats"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phrase_df = _load_phrase_stats(compiled_stats_path, compiled_format=compiled_format)

    # If keep_groups not specified, use all available groups
    if keep_groups is None:
        if group_col not in phrase_df.columns:
            raise KeyError(f"Compiled stats missing group_col={group_col!r}.")
        keep_groups = sorted([g for g in phrase_df[group_col].dropna().unique().tolist()])

    meta_df = _load_metadata(
        metadata_excel_path,
        sheet_name=metadata_sheet_name,
        id_col=id_col,
    ).copy()
    meta_df["treat_class"] = meta_df["Treatment type"].map(_classify_treatment_type)

    # --- Build long tables ---
    logvar_long = build_long_logvar(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        syll_col=syll_col,
        n_col=n_col,
        var_col=var_col,
        mean_col=mean_col,
        keep_groups=keep_groups,
        log_mode=log_mode,
    )
    cv_long = build_long_cv(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        syll_col=syll_col,
        n_col=n_col,
        mean_col=mean_col,
        var_col=var_col,
        keep_groups=keep_groups,
    )

    # --- Metrics: unweighted + weighted by N_phrases ---
    logvar_metrics_unw = compute_bird_group_metrics(
        logvar_long, id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=None, weighting_label="unweighted"
    )
    logvar_metrics_w = compute_bird_group_metrics(
        logvar_long, id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=n_col, weighting_label="weighted_by_N_phrases"
    )
    logvar_metrics = pd.concat([logvar_metrics_unw, logvar_metrics_w], ignore_index=True)

    cv_metrics_unw = compute_bird_group_metrics(
        cv_long, id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=None, weighting_label="unweighted"
    )
    cv_metrics_w = compute_bird_group_metrics(
        cv_long, id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=n_col, weighting_label="weighted_by_N_phrases"
    )
    cv_metrics = pd.concat([cv_metrics_unw, cv_metrics_w], ignore_index=True)

    # --- Flag outliers in long tables for each weighting mode ---
    logvar_long_flag_unw = add_outlier_flags_and_modz(
        logvar_long, logvar_metrics,
        id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=None, prefix="logvar", weighting_label="unweighted"
    )
    logvar_long_flag_w = add_outlier_flags_and_modz(
        logvar_long, logvar_metrics,
        id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=n_col, prefix="logvar", weighting_label="weighted_by_N_phrases"
    )

    cv_long_flag_unw = add_outlier_flags_and_modz(
        cv_long, cv_metrics,
        id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=None, prefix="cv", weighting_label="unweighted"
    )
    cv_long_flag_w = add_outlier_flags_and_modz(
        cv_long, cv_metrics,
        id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=n_col, prefix="cv", weighting_label="weighted_by_N_phrases"
    )

    # --- Exceedance rates (Post above Pre thresholds), both weighting modes ---
    logvar_ex_unw = compute_post_exceedance_rates(
        logvar_long, logvar_metrics,
        id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=None,
        pre_group=pre_group, post_group=post_group,
        prefix="logvar", weighting_label="unweighted"
    )
    logvar_ex_w = compute_post_exceedance_rates(
        logvar_long, logvar_metrics,
        id_col=id_col, group_col=group_col, value_col="value_logvar",
        weight_col=n_col,
        pre_group=pre_group, post_group=post_group,
        prefix="logvar", weighting_label="weighted_by_N_phrases"
    )
    logvar_exceed = pd.concat([logvar_ex_unw, logvar_ex_w], ignore_index=True)

    cv_ex_unw = compute_post_exceedance_rates(
        cv_long, cv_metrics,
        id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=None,
        pre_group=pre_group, post_group=post_group,
        prefix="cv", weighting_label="unweighted"
    )
    cv_ex_w = compute_post_exceedance_rates(
        cv_long, cv_metrics,
        id_col=id_col, group_col=group_col, value_col="value_cv",
        weight_col=n_col,
        pre_group=pre_group, post_group=post_group,
        prefix="cv", weighting_label="weighted_by_N_phrases"
    )
    cv_exceed = pd.concat([cv_ex_unw, cv_ex_w], ignore_index=True)

    # --- Per-bird pre/post deltas on a few key tail/outlier metrics ---
    def _extract_prepost(metrics_df: pd.DataFrame, prefix: str, weighting_value: str) -> pd.DataFrame:
        m = metrics_df[metrics_df["weighting"] == weighting_value].copy()
        pre = m[m[group_col] == pre_group].copy()
        post = m[m[group_col] == post_group].copy()

        keep_cols = [id_col, "q95", "q99", "frac_high_outliers_1p5", "weight_frac_high_outliers_1p5"]
        pre = pre[keep_cols].rename(
            columns={
                "q95": f"{prefix}_pre_q95",
                "q99": f"{prefix}_pre_q99",
                "frac_high_outliers_1p5": f"{prefix}_pre_frac_high_outliers_1p5",
                "weight_frac_high_outliers_1p5": f"{prefix}_pre_weight_frac_high_outliers_1p5",
            }
        )
        post = post[keep_cols].rename(
            columns={
                "q95": f"{prefix}_post_q95",
                "q99": f"{prefix}_post_q99",
                "frac_high_outliers_1p5": f"{prefix}_post_frac_high_outliers_1p5",
                "weight_frac_high_outliers_1p5": f"{prefix}_post_weight_frac_high_outliers_1p5",
            }
        )

        merged = pre.merge(post, on=id_col, how="inner")
        merged[f"delta_{prefix}_q95"] = merged[f"{prefix}_post_q95"] - merged[f"{prefix}_pre_q95"]
        merged[f"delta_{prefix}_q99"] = merged[f"{prefix}_post_q99"] - merged[f"{prefix}_pre_q99"]
        merged[f"delta_{prefix}_frac_high_outliers_1p5"] = (
            merged[f"{prefix}_post_frac_high_outliers_1p5"] - merged[f"{prefix}_pre_frac_high_outliers_1p5"]
        )
        merged[f"delta_{prefix}_weight_frac_high_outliers_1p5"] = (
            merged[f"{prefix}_post_weight_frac_high_outliers_1p5"] - merged[f"{prefix}_pre_weight_frac_high_outliers_1p5"]
        )
        merged["weighting"] = weighting_value
        return merged

    per_bird_logvar_unw = _extract_prepost(logvar_metrics, "logvar", "unweighted")
    per_bird_logvar_w = _extract_prepost(logvar_metrics, "logvar", "weighted_by_N_phrases")
    per_bird_cv_unw = _extract_prepost(cv_metrics, "cv", "unweighted")
    per_bird_cv_w = _extract_prepost(cv_metrics, "cv", "weighted_by_N_phrases")

    per_bird_unw = per_bird_logvar_unw.merge(per_bird_cv_unw, on=[id_col, "weighting"], how="outer")
    per_bird_w = per_bird_logvar_w.merge(per_bird_cv_w, on=[id_col, "weighting"], how="outer")
    per_bird = pd.concat([per_bird_unw, per_bird_w], ignore_index=True)

    # Merge exceedance and treatment class
    per_bird = per_bird.merge(logvar_exceed, on=[id_col, "weighting"], how="left")
    per_bird = per_bird.merge(cv_exceed, on=[id_col, "weighting"], how="left")
    per_bird = per_bird.merge(meta_df[[id_col, "Treatment type", "treat_class"]], on=id_col, how="left")

    # --- Save outputs ---
    paths = {
        "logvar_long_flagged_unweighted": output_dir / "logvar_long_flagged_unweighted.csv",
        "logvar_long_flagged_weighted": output_dir / "logvar_long_flagged_weighted.csv",
        "cv_long_flagged_unweighted": output_dir / "cv_long_flagged_unweighted.csv",
        "cv_long_flagged_weighted": output_dir / "cv_long_flagged_weighted.csv",
        "logvar_metrics": output_dir / "per_bird_group_outlier_metrics_logvar.csv",
        "cv_metrics": output_dir / "per_bird_group_outlier_metrics_cv.csv",
        "per_bird_summary": output_dir / "per_bird_prepost_outlier_summary.csv",
        "group_tests": output_dir / "group_comparisons_outlier_metrics.csv",
    }

    logvar_long_flag_unw.to_csv(paths["logvar_long_flagged_unweighted"], index=False)
    logvar_long_flag_w.to_csv(paths["logvar_long_flagged_weighted"], index=False)
    cv_long_flag_unw.to_csv(paths["cv_long_flagged_unweighted"], index=False)
    cv_long_flag_w.to_csv(paths["cv_long_flagged_weighted"], index=False)
    logvar_metrics.to_csv(paths["logvar_metrics"], index=False)
    cv_metrics.to_csv(paths["cv_metrics"], index=False)
    per_bird.to_csv(paths["per_bird_summary"], index=False)

    group_tests_df: Optional[pd.DataFrame] = None
    if run_group_tests and _HAVE_SCIPY:
        gt_unw = compare_groups_on_metrics(per_bird, weighting_value="unweighted")
        gt_w = compare_groups_on_metrics(per_bird, weighting_value="weighted_by_N_phrases")
        group_tests_df = pd.concat([gt_unw, gt_w], ignore_index=True)
        group_tests_df.to_csv(paths["group_tests"], index=False)

    return {
        "logvar_long_unweighted": logvar_long_flag_unw,
        "logvar_long_weighted": logvar_long_flag_w,
        "cv_long_unweighted": cv_long_flag_unw,
        "cv_long_weighted": cv_long_flag_w,
        "logvar_metrics": logvar_metrics,
        "cv_metrics": cv_metrics,
        "per_bird_summary": per_bird,
        "group_tests": group_tests_df,
        "paths": paths,
        "keep_groups": keep_groups,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute per-bird tail/outlier statistics for log-variance and CV (syllable-level distributions)."
    )
    parser.add_argument("compiled_stats_path", type=str, help="Path to compiled phrase-duration stats (CSV/JSON/NPZ).")
    parser.add_argument("metadata_excel_path", type=str, help="Path to metadata Excel (must contain Treatment type).")

    parser.add_argument(
        "--compiled_format",
        type=str,
        default=None,
        choices=["csv", "json", "npz"],
        help="File format of compiled stats (inferred if omitted).",
    )
    parser.add_argument(
        "--metadata_sheet_name",
        type=str,
        default="metadata",
        help="Sheet name or index for metadata Excel (default: 'metadata').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: phrase_duration_outlier_stats next to compiled file).",
    )

    # Column args (defaults match your helper output)
    parser.add_argument("--id_col", type=str, default="Animal ID")
    parser.add_argument("--group_col", type=str, default="Group")
    parser.add_argument("--syll_col", type=str, default="Syllable")
    parser.add_argument("--n_col", type=str, default="N_phrases")
    parser.add_argument("--mean_col", type=str, default="Mean_ms")
    parser.add_argument("--var_col", type=str, default="Variance_ms2")

    parser.add_argument("--pre_group", type=str, default="Late Pre")
    parser.add_argument("--post_group", type=str, default="Post")

    parser.add_argument(
        "--log_mode",
        type=str,
        default="log10",
        choices=["log10", "ln", "log"],
        help="Which log to use for variance (default: log10).",
    )
    parser.add_argument(
        "--run_group_tests",
        action="store_true",
        help="If set, run NMA vs sham comparisons on selected outlier metrics (requires SciPy).",
    )

    args = parser.parse_args()

    # Allow sheet name to be "0" or an actual string
    sheet_name: Union[int, str]
    try:
        sheet_name = int(args.metadata_sheet_name)
    except ValueError:
        sheet_name = args.metadata_sheet_name

    out_dir = Path(args.output_dir) if args.output_dir is not None else None

    res = run_phrase_outlier_stats(
        compiled_stats_path=args.compiled_stats_path,
        metadata_excel_path=args.metadata_excel_path,
        compiled_format=args.compiled_format,
        metadata_sheet_name=sheet_name,
        output_dir=out_dir,
        id_col=args.id_col,
        group_col=args.group_col,
        syll_col=args.syll_col,
        n_col=args.n_col,
        mean_col=args.mean_col,
        var_col=args.var_col,
        pre_group=args.pre_group,
        post_group=args.post_group,
        log_mode=args.log_mode,
        run_group_tests=args.run_group_tests,
    )

    print("\n[SAVED OUTPUTS]")
    for k, v in res["paths"].items():
        print(f"  {k}: {v}")

    print("\n[INFO] Groups included:", res["keep_groups"])

    if args.run_group_tests and (res["group_tests"] is None):
        print("\n[WARN] Group tests requested but SciPy was not available.")
    elif res["group_tests"] is not None:
        print("\n[GROUP TESTS] (head)")
        print(res["group_tests"].head(12).to_string(index=False))


"""
Example usage (Spyder / notebook):

from pathlib import Path
import importlib
import phrase_duration_stats_outliers as pdo
importlib.reload(pdo)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputsusage_balanced_phrase_duration_stats.csv")
excel_path   = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")

res = pdo.run_phrase_outlier_stats(
    compiled_stats_path=compiled_csv,
    metadata_excel_path=excel_path,
    output_dir=compiled_csv.parent / "phrase_duration_outlier_stats",
    pre_group="Late Pre",
    post_group="Post",
    run_group_tests=True,  # requires SciPy
)

print(res["per_bird_summary"].head())
"""
