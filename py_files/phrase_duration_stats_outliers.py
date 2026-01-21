#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phrase_duration_stats_outliers.py

Compute and SAVE tail/outlier statistics for phrase-duration variability, using
your compiled phrase-duration stats file (CSV/JSON/NPZ) + metadata Excel.

What this script does
---------------------
For each bird (Animal ID) and epoch/group (e.g., "Late Pre", "Post"), it builds
a *distribution* across rows in your compiled table (typically across syllables)
for two measures:

  1) log-variance of phrase duration (from Variance_ms2)
  2) coefficient of variation (CV = sqrt(Variance_ms2)/Mean_ms)

Then it computes per-bird-per-group tail/outlier metrics, including:
  - quantiles: q1, q50, q75, q90, q95, q99
  - IQR and Tukey fences (1.5*IQR + 3*IQR "extreme")
  - counts and fractions of high/low outliers (Tukey)
  - MAD + modified z-score counts (|modz| > 3.5)
  - tail heaviness ratios (e.g., (q95-q75)/(q75-q50))

And *pre/post exceedance* metrics per bird:
  - fraction of Post values above that bird's Pre q95
  - fraction of Post values above that bird's Pre Tukey high fence
  (computed separately for log-variance and CV)

Outputs (saved to output_dir)
-----------------------------
- logvar_long_with_outlier_flags.csv
- cv_long_with_outlier_flags.csv
- per_bird_group_outlier_metrics_logvar.csv
- per_bird_group_outlier_metrics_cv.csv
- per_bird_prepost_outlier_summary.csv
- group_comparisons_outlier_metrics.csv   (optional; if SciPy available)

Important note
--------------
These “outliers” are outliers *within the distribution represented by rows in
your compiled table* (often syllables). If you want outliers over songs/bouts/days,
you’ll need a song-level table instead of syllable-level summaries.

Author: ChatGPT (tail/outlier-focused companion to phrase_duration_stats.py)
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
    Load metadata directly from Excel.

    Expected columns (at minimum):
        id_col, 'Treatment type'

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


def _infer_label_col(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> Optional[str]:
    """
    Try to infer a syllable/label column for nicer outlier tables.
    Safe to return None if none found.
    """
    if candidates is None:
        candidates = [
            "Syllable",
            "Syllable label",
            "Syllable_label",
            "Label",
            "label",
            "syllable",
            "syllable_label",
            "Mapped label",
            "mapped_label",
            "mapped_syllable",
            "mapped_syllable_order",
        ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------------------------------------------------------
# Core: build long-form distributions (per bird × group) for logvar and CV
# -----------------------------------------------------------------------------
def build_long_logvar(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    var_col: str = "Variance_ms2",
    label_col: Optional[str] = None,
    keep_groups: Optional[List[str]] = None,
    log_mode: str = "log10",
    clip_lower: float = 1e-12,
) -> pd.DataFrame:
    """
    Return a long DataFrame with one row per original row, including:
        id_col
        group_col
        (optional) label_col
        value_logvar
        value_raw_variance
    """
    needed = [id_col, group_col, var_col]
    missing = [c for c in needed if c not in phrase_df.columns]
    if missing:
        raise KeyError(f"Missing columns for logvar long build: {missing}")

    df = phrase_df.copy()

    if keep_groups is not None:
        df = df[df[group_col].isin(keep_groups)].copy()

    df = df.dropna(subset=[var_col, id_col, group_col]).copy()
    var_vals = df[var_col].astype(float).clip(lower=clip_lower)

    if log_mode.lower() == "log10":
        df["value_logvar"] = np.log10(var_vals)
        df.attrs["log_label"] = "log₁₀ variance (ms²)"
    elif log_mode.lower() in {"ln", "log"}:
        df["value_logvar"] = np.log(var_vals)
        df.attrs["log_label"] = "ln variance (ms²)"
    else:
        raise ValueError(f"Unsupported log_mode={log_mode!r}; use 'log10' or 'ln'.")

    df["value_raw_variance"] = var_vals

    cols = [id_col, group_col, "value_logvar", "value_raw_variance"]
    if label_col is not None and label_col in df.columns:
        cols.insert(2, label_col)

    return df[cols].copy()


def build_long_cv(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    label_col: Optional[str] = None,
    keep_groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a long DataFrame with one row per original row, including:
        id_col
        group_col
        (optional) label_col
        value_cv
        value_raw_mean
        value_raw_variance
    """
    needed = [id_col, group_col, mean_col, var_col]
    missing = [c for c in needed if c not in phrase_df.columns]
    if missing:
        raise KeyError(f"Missing columns for CV long build: {missing}")

    df = phrase_df.copy()
    if keep_groups is not None:
        df = df[df[group_col].isin(keep_groups)].copy()

    df = df.dropna(subset=[mean_col, var_col, id_col, group_col]).copy()

    means = df[mean_col].astype(float)
    vars_ = df[var_col].astype(float).clip(lower=0.0)

    means_safe = means.replace(0.0, np.nan)
    df["value_cv"] = np.sqrt(vars_) / means_safe

    df["value_raw_mean"] = means
    df["value_raw_variance"] = vars_

    df = df.dropna(subset=["value_cv"]).copy()

    cols = [id_col, group_col, "value_cv", "value_raw_mean", "value_raw_variance"]
    if label_col is not None and label_col in df.columns:
        cols.insert(2, label_col)

    return df[cols].copy()


# -----------------------------------------------------------------------------
# Outlier/tail summaries for a 1D distribution
# -----------------------------------------------------------------------------
def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (unscaled)."""
    if x.size == 0:
        return float("nan")
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def summarize_distribution(x: pd.Series) -> pd.Series:
    """
    Summarize a distribution into tail/outlier metrics.
    Expects x to be numeric and already cleaned (finite preferred).
    """
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return pd.Series(
            {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
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
                "n_low_outliers_1p5": 0,
                "n_high_outliers_1p5": 0,
                "n_low_outliers_3": 0,
                "n_high_outliers_3": 0,
                "frac_high_outliers_1p5": np.nan,
                "median": np.nan,
                "mad": np.nan,
                "n_modz_abs_gt_3p5": 0,
                "n_modz_gt_3p5": 0,
                "tail_ratio_95_75_over_75_50": np.nan,
            }
        )

    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if n >= 2 else float("nan")

    q1 = float(np.nanquantile(arr, 0.25))
    q50 = float(np.nanquantile(arr, 0.50))
    q75 = float(np.nanquantile(arr, 0.75))
    q90 = float(np.nanquantile(arr, 0.90))
    q95 = float(np.nanquantile(arr, 0.95))
    q99 = float(np.nanquantile(arr, 0.99))

    iqr = q75 - q1
    tukey_lo_1p5 = q1 - 1.5 * iqr
    tukey_hi_1p5 = q75 + 1.5 * iqr
    tukey_lo_3 = q1 - 3.0 * iqr
    tukey_hi_3 = q75 + 3.0 * iqr

    n_low_1p5 = int(np.sum(arr < tukey_lo_1p5))
    n_high_1p5 = int(np.sum(arr > tukey_hi_1p5))
    n_low_3 = int(np.sum(arr < tukey_lo_3))
    n_high_3 = int(np.sum(arr > tukey_hi_3))
    frac_high_1p5 = float(n_high_1p5 / n) if n > 0 else float("nan")

    med = float(np.nanmedian(arr))
    mad = _mad(arr)

    # Modified z-score: 0.6745*(x - median)/MAD
    if np.isfinite(mad) and mad > 0:
        modz = 0.6745 * (arr - med) / mad
        n_modz_abs = int(np.sum(np.abs(modz) > 3.5))
        n_modz_hi = int(np.sum(modz > 3.5))
    else:
        n_modz_abs = 0
        n_modz_hi = 0

    denom = (q75 - q50)
    tail_ratio = float((q95 - q75) / denom) if np.isfinite(denom) and denom > 0 else float("nan")

    return pd.Series(
        {
            "n": n,
            "mean": mean,
            "std": std,
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
            "n_low_outliers_1p5": n_low_1p5,
            "n_high_outliers_1p5": n_high_1p5,
            "n_low_outliers_3": n_low_3,
            "n_high_outliers_3": n_high_3,
            "frac_high_outliers_1p5": frac_high_1p5,
            "median": med,
            "mad": mad,
            "n_modz_abs_gt_3p5": n_modz_abs,
            "n_modz_gt_3p5": n_modz_hi,
            "tail_ratio_95_75_over_75_50": tail_ratio,
        }
    )


def compute_bird_group_metrics(
    long_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Compute per (bird, group) summary metrics for the distribution in value_col.
    """
    needed = [id_col, group_col, value_col]
    missing = [c for c in needed if c not in long_df.columns]
    if missing:
        raise KeyError(f"Missing columns for bird-group metrics: {missing}")

    out = (
        long_df.groupby([id_col, group_col])[value_col]
        .apply(summarize_distribution)
        .reset_index()
    )
    return out


def add_outlier_flags(
    long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge Tukey fences from metrics_df onto long_df and flag outlier rows.
    """
    fence_cols = [
        "tukey_lo_1p5",
        "tukey_hi_1p5",
        "tukey_lo_3",
        "tukey_hi_3",
    ]
    needed = [id_col, group_col] + fence_cols
    missing = [c for c in needed if c not in metrics_df.columns]
    if missing:
        raise KeyError(f"metrics_df missing needed fence columns: {missing}")

    merged = long_df.merge(
        metrics_df[[id_col, group_col] + fence_cols],
        on=[id_col, group_col],
        how="left",
    )

    v = pd.to_numeric(merged[value_col], errors="coerce").astype(float)

    merged[f"{prefix}_is_low_outlier_1p5"] = v < merged["tukey_lo_1p5"]
    merged[f"{prefix}_is_high_outlier_1p5"] = v > merged["tukey_hi_1p5"]
    merged[f"{prefix}_is_low_outlier_3"] = v < merged["tukey_lo_3"]
    merged[f"{prefix}_is_high_outlier_3"] = v > merged["tukey_hi_3"]

    return merged


# -----------------------------------------------------------------------------
# Pre/Post exceedance metrics
# -----------------------------------------------------------------------------
def compute_post_exceedance_rates(
    long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    value_col: str,
    pre_group: str,
    post_group: str,
    prefix: str,
) -> pd.DataFrame:
    """
    For each bird, compute fraction of Post values exceeding Pre thresholds:
      - Pre q95
      - Pre Tukey high fence (1.5*IQR)

    Returns a per-bird DataFrame with columns:
      prefix_post_exceed_pre_q95_rate
      prefix_post_exceed_pre_hi_fence_rate
      prefix_pre_q95
      prefix_pre_hi_fence
    """
    # Extract per-bird pre thresholds from metrics_df (which is per bird-group)
    pre_metrics = metrics_df[metrics_df[group_col] == pre_group].copy()
    if pre_metrics.empty:
        raise RuntimeError(f"No rows found for pre_group={pre_group!r} in metrics_df.")

    pre_thr = pre_metrics[[id_col, "q95", "tukey_hi_1p5"]].rename(
        columns={
            "q95": f"{prefix}_pre_q95",
            "tukey_hi_1p5": f"{prefix}_pre_hi_fence_1p5",
        }
    )

    post = long_df[long_df[group_col] == post_group].copy()
    if post.empty:
        raise RuntimeError(f"No rows found for post_group={post_group!r} in long_df.")

    post = post.merge(pre_thr, on=id_col, how="left")
    post_val = pd.to_numeric(post[value_col], errors="coerce").astype(float)

    def _rate(d: pd.DataFrame) -> pd.Series:
        vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return pd.Series(
                {
                    f"{prefix}_post_exceed_pre_q95_rate": np.nan,
                    f"{prefix}_post_exceed_pre_hi_fence_rate": np.nan,
                }
            )
        pre_q95 = float(d[f"{prefix}_pre_q95"].iloc[0]) if np.isfinite(d[f"{prefix}_pre_q95"].iloc[0]) else np.nan
        pre_hi = float(d[f"{prefix}_pre_hi_fence_1p5"].iloc[0]) if np.isfinite(d[f"{prefix}_pre_hi_fence_1p5"].iloc[0]) else np.nan

        r1 = float(np.mean(vals > pre_q95)) if np.isfinite(pre_q95) else np.nan
        r2 = float(np.mean(vals > pre_hi)) if np.isfinite(pre_hi) else np.nan
        return pd.Series(
            {
                f"{prefix}_post_exceed_pre_q95_rate": r1,
                f"{prefix}_post_exceed_pre_hi_fence_rate": r2,
            }
        )

    rates = post.groupby(id_col).apply(_rate).reset_index()
    rates = rates.merge(pre_thr, on=id_col, how="left")
    return rates


# -----------------------------------------------------------------------------
# Optional: group comparisons on outlier metrics (NMA vs sham)
# -----------------------------------------------------------------------------
def compare_groups_on_metrics(
    per_bird_df: pd.DataFrame,
    *,
    treat_col: str = "treat_class",
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

    df = per_bird_df.copy()
    df = df[df[treat_col].isin([g1, g2])].copy()

    if metrics is None:
        # Reasonable defaults: exceedance and tail deltas
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
    id_col: str = "Animal ID",
    group_col: str = "Group",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    keep_groups: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    log_mode: str = "log10",
    run_group_tests: bool = True,
) -> Dict[str, Any]:
    """
    High-level helper:
      1) Loads compiled phrase stats and metadata
      2) Builds long-form logvar and CV tables
      3) Computes per-bird-per-group tail/outlier metrics
      4) Flags outlier rows (Tukey fences) in the long tables
      5) Computes per-bird post exceedance rates over pre thresholds
      6) Builds per-bird pre/post delta summaries
      7) Saves outputs to output_dir
      8) Optionally runs NMA vs sham group comparisons on selected metrics
    """
    compiled_stats_path = Path(compiled_stats_path)
    if output_dir is None:
        output_dir = compiled_stats_path.parent / "phrase_duration_outlier_stats"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phrase_df = _load_phrase_stats(compiled_stats_path, compiled_format=compiled_format)
    meta_df = _load_metadata(
        metadata_excel_path,
        sheet_name=metadata_sheet_name,
        id_col=id_col,
    ).copy()
    meta_df["treat_class"] = meta_df["Treatment type"].map(_classify_treatment_type)

    if keep_groups is None:
        keep_groups = [pre_group, post_group]

    if label_col is None:
        label_col = _infer_label_col(phrase_df)

    # --- Build long tables ---
    logvar_long = build_long_logvar(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        var_col=var_col,
        label_col=label_col,
        keep_groups=keep_groups,
        log_mode=log_mode,
    )
    cv_long = build_long_cv(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        mean_col=mean_col,
        var_col=var_col,
        label_col=label_col,
        keep_groups=keep_groups,
    )

    # --- Bird-group metrics ---
    logvar_metrics = compute_bird_group_metrics(
        logvar_long,
        id_col=id_col,
        group_col=group_col,
        value_col="value_logvar",
    )
    cv_metrics = compute_bird_group_metrics(
        cv_long,
        id_col=id_col,
        group_col=group_col,
        value_col="value_cv",
    )

    # --- Outlier flags on long tables ---
    logvar_long_flagged = add_outlier_flags(
        logvar_long,
        logvar_metrics,
        id_col=id_col,
        group_col=group_col,
        value_col="value_logvar",
        prefix="logvar",
    )
    cv_long_flagged = add_outlier_flags(
        cv_long,
        cv_metrics,
        id_col=id_col,
        group_col=group_col,
        value_col="value_cv",
        prefix="cv",
    )

    # --- Exceedance rates (Post above Pre thresholds) ---
    logvar_exceed = compute_post_exceedance_rates(
        logvar_long,
        logvar_metrics,
        id_col=id_col,
        group_col=group_col,
        value_col="value_logvar",
        pre_group=pre_group,
        post_group=post_group,
        prefix="logvar",
    )
    cv_exceed = compute_post_exceedance_rates(
        cv_long,
        cv_metrics,
        id_col=id_col,
        group_col=group_col,
        value_col="value_cv",
        pre_group=pre_group,
        post_group=post_group,
        prefix="cv",
    )

    # --- Per-bird pre/post summary with deltas on key tail/outlier metrics ---
    def _extract_prepost(metrics_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        pre = metrics_df[metrics_df[group_col] == pre_group].copy()
        post = metrics_df[metrics_df[group_col] == post_group].copy()

        keep_cols = [
            id_col,
            "q95",
            "q99",
            "frac_high_outliers_1p5",
            "n_high_outliers_1p5",
            "n",
            "tukey_hi_1p5",
        ]
        pre = pre[keep_cols].rename(
            columns={
                "q95": f"{prefix}_pre_q95",
                "q99": f"{prefix}_pre_q99",
                "frac_high_outliers_1p5": f"{prefix}_pre_frac_high_outliers_1p5",
                "n_high_outliers_1p5": f"{prefix}_pre_n_high_outliers_1p5",
                "n": f"{prefix}_pre_n",
                "tukey_hi_1p5": f"{prefix}_pre_hi_fence_1p5",
            }
        )
        post = post[keep_cols].rename(
            columns={
                "q95": f"{prefix}_post_q95",
                "q99": f"{prefix}_post_q99",
                "frac_high_outliers_1p5": f"{prefix}_post_frac_high_outliers_1p5",
                "n_high_outliers_1p5": f"{prefix}_post_n_high_outliers_1p5",
                "n": f"{prefix}_post_n",
                "tukey_hi_1p5": f"{prefix}_post_hi_fence_1p5",
            }
        )

        merged = pre.merge(post, on=id_col, how="inner")

        merged[f"delta_{prefix}_q95"] = merged[f"{prefix}_post_q95"] - merged[f"{prefix}_pre_q95"]
        merged[f"delta_{prefix}_q99"] = merged[f"{prefix}_post_q99"] - merged[f"{prefix}_pre_q99"]
        merged[f"delta_{prefix}_frac_high_outliers_1p5"] = (
            merged[f"{prefix}_post_frac_high_outliers_1p5"] - merged[f"{prefix}_pre_frac_high_outliers_1p5"]
        )
        return merged

    per_bird_logvar_prepost = _extract_prepost(logvar_metrics, prefix="logvar")
    per_bird_cv_prepost = _extract_prepost(cv_metrics, prefix="cv")

    per_bird = per_bird_logvar_prepost.merge(per_bird_cv_prepost, on=id_col, how="outer")
    per_bird = per_bird.merge(logvar_exceed, on=id_col, how="left")
    per_bird = per_bird.merge(cv_exceed, on=id_col, how="left")
    per_bird = per_bird.merge(meta_df[[id_col, "Treatment type", "treat_class"]], on=id_col, how="left")

    # --- Save outputs ---
    path_log_long = output_dir / "logvar_long_with_outlier_flags.csv"
    path_cv_long = output_dir / "cv_long_with_outlier_flags.csv"
    path_log_metrics = output_dir / "per_bird_group_outlier_metrics_logvar.csv"
    path_cv_metrics = output_dir / "per_bird_group_outlier_metrics_cv.csv"
    path_per_bird = output_dir / "per_bird_prepost_outlier_summary.csv"
    path_group_tests = output_dir / "group_comparisons_outlier_metrics.csv"

    logvar_long_flagged.to_csv(path_log_long, index=False)
    cv_long_flagged.to_csv(path_cv_long, index=False)
    logvar_metrics.to_csv(path_log_metrics, index=False)
    cv_metrics.to_csv(path_cv_metrics, index=False)
    per_bird.to_csv(path_per_bird, index=False)

    group_tests_df: Optional[pd.DataFrame] = None
    if run_group_tests and _HAVE_SCIPY:
        group_tests_df = compare_groups_on_metrics(per_bird, treat_col="treat_class")
        group_tests_df.to_csv(path_group_tests, index=False)

    return {
        "logvar_long": logvar_long_flagged,
        "cv_long": cv_long_flagged,
        "logvar_metrics": logvar_metrics,
        "cv_metrics": cv_metrics,
        "per_bird_summary": per_bird,
        "group_tests": group_tests_df,
        "paths": {
            "logvar_long": path_log_long,
            "cv_long": path_cv_long,
            "logvar_metrics": path_log_metrics,
            "cv_metrics": path_cv_metrics,
            "per_bird_summary": path_per_bird,
            "group_tests": path_group_tests if group_tests_df is not None else None,
        },
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute per-bird tail/outlier statistics for log-variance and CV (Late Pre vs Post)."
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
    parser.add_argument("--id_col", type=str, default="Animal ID")
    parser.add_argument("--group_col", type=str, default="Group")
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
        help="If set, run NMA vs sham group comparisons on selected outlier metrics (requires SciPy).",
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

    if res["group_tests"] is None and args.run_group_tests:
        print("\n[WARN] Group tests were requested but SciPy was not available.")
    elif res["group_tests"] is not None:
        print("\n[GROUP TESTS] (head)")
        print(res["group_tests"].head(10).to_string(index=False))


"""
Example (Spyder / notebook):

from pathlib import Path
import importlib
import phrase_duration_stats_outliers as pdo
importlib.reload(pdo)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
excel_path   = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")

res = pdo.run_phrase_outlier_stats(
    compiled_stats_path=compiled_csv,
    metadata_excel_path=excel_path,
    output_dir=compiled_csv.parent / "phrase_duration_outlier_stats",
    pre_group="Late Pre",
    post_group="Post",
    run_group_tests=True,   # requires scipy
)

print(res["per_bird_summary"].head())
print("Outlier long table (logvar):", res["paths"]["logvar_long"])
"""
