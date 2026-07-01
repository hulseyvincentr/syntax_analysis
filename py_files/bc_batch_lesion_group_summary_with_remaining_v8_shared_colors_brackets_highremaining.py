#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_batch_lesion_group_summary_with_remaining_v8_no_birdcount_clean_colors.py

Post-process a batch of per-bird *_cluster_bc_summary.csv files and make
lesion-hit-type comparison plots for three cluster sets:

  1) all qualifying clusters
  2) high-variance phrase-duration clusters (top-fraction set marked by the wrapper)
  3) remaining non-high-variance clusters (the complement of #2)

This version is tuned for publication-style consistency across figures:
  - combines complete + partial medial/lateral lesion groups into one
    "Medial and Lateral lesion" group
  - uses the shared teal/purple lesion-hit-type color scheme used across the manuscript figures
  - writes boxplot-only versions of the pre/post and delta figures
  - removes the extra "N birds" text from plot titles by default, so the
    p-value annotations have more room
  - places Pre/Post labels on the x-axis and lesion-group + p-value text below
    each Pre/Post pair
  - reserves vertical space above the boxplots so significance brackets do not
    overlap with boxes/whiskers
  - adds a Figure-4F-style medial+lateral-only plot comparing top 30% variance
    clusters against the remaining non-top-variance clusters

Expected input files are the per-bird outputs from bc_cluster_qc_and_summaries_*.py,
especially files named:

    <ANIMAL_ID>_cluster_bc_summary.csv

Required columns in each cluster summary:
    animal_id, cluster_id, passes_min_balanced_duration, is_high_variance_cluster,
    bc_pre, bc_post, bc_prepost

The script can also use method-specific columns if present, for example:
    bc_pre_selected_bins, bc_post_selected_bins, bc_prepost_selected_bins
    bc_pre_full_contiguous_selected_runs, ...
    bc_pre_run_weighted_full_contiguous, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon, kruskal, mannwhitneyu
except Exception:  # pragma: no cover
    wilcoxon = None
    kruskal = None
    mannwhitneyu = None


# -----------------------------------------------------------------------------
# Labels, colors, and BC method definitions
# -----------------------------------------------------------------------------

SET_LABELS = {
    "all_clusters": "All qualifying clusters",
    "high_variance_clusters": "Top 30% phrase-duration variance clusters",
    "remaining_non_high_variance_clusters": "Remaining non-top-variance clusters",
}

SET_FILENAME_TOKEN = {
    "all_clusters": "all_clusters",
    "high_variance_clusters": "high_variance_clusters",
    "remaining_non_high_variance_clusters": "remaining_non_high_variance_clusters",
}

COMBINED_ML_LABEL = "Medial and Lateral lesion"
LATERAL_LABEL = "Lateral lesion only"
SHAM_LABEL = "sham saline injection"
UNKNOWN_LABEL = "unknown"

DEFAULT_LESION_ORDER = [
    SHAM_LABEL,
    LATERAL_LABEL,
    COMBINED_ML_LABEL,
    UNKNOWN_LABEL,
]

# Shared manuscript palette.
# This matches the phrase-duration variability figures:
#   - sham: teal/green
#   - lateral-only lesion: light purple
#   - medial+lateral lesion pooled: medium purple
#
# Note: this BC plotting script intentionally pools complete + partial medial/lateral
# lesions into COMBINED_ML_LABEL via _normalize_hit_type().
DEFAULT_LESION_COLORS = {
    SHAM_LABEL: "#1B9E77",
    LATERAL_LABEL: "#A88BD9",
    COMBINED_ML_LABEL: "#7A4FB7",
    UNKNOWN_LABEL: "#4D4D4D",
}

METHOD_DEFS = {
    "selected_bins": (
        ["bc_pre_selected_bins", "bc_pre"],
        ["bc_post_selected_bins", "bc_post"],
        ["bc_prepost_selected_bins", "bc_prepost"],
        "Selected equal time bins",
    ),
    "full_contiguous_selected_runs": (
        ["bc_pre_full_contiguous_selected_runs"],
        ["bc_post_full_contiguous_selected_runs"],
        ["bc_prepost_full_contiguous_selected_runs"],
        "Full-contiguous selected runs",
    ),
    "run_weighted_full_contiguous": (
        ["bc_pre_run_weighted_full_contiguous"],
        ["bc_post_run_weighted_full_contiguous"],
        ["bc_prepost_run_weighted_full_contiguous"],
        "Run-weighted full-contiguous selected runs",
    ),
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_filename(s: Any) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "yes", "y", "t"])


def _normalize_animal_id(x: Any) -> str:
    return str(x).strip()


def _normalize_hit_type(x: Any) -> str:
    """Collapse raw metadata names into the three display groups."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return UNKNOWN_LABEL

    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)

    sham = {"sham saline injection", "sham", "saline", "sham control", "control"}
    lateral = {
        "lateral lesion only",
        "lateral hit only",
        "area x visible (single hit)",
        "single hit",
        "single lateral hit",
        "area x visible (lateral only)",
    }
    partial = {
        "partial medial and lateral lesion",
        "area x visible (medial+lateral hit)",
        "area x visible medial+lateral hit",
        "medial+lateral hit",
        "m+l hit",
        "medial+lateral",
        "medial lateral",
        "area x visible (medial + lateral hit)",
    }
    complete = {
        "complete medial and lateral lesion",
        "large lesion area x not visible",
        "large lesion, area x not visible",
        "area x not visible",
        "large lesion",
        "area x visible and large lesion area x not visible",
    }

    if s in sham:
        return SHAM_LABEL
    if s in lateral:
        return LATERAL_LABEL
    if s in partial or s in complete:
        return COMBINED_ML_LABEL

    # Conservative phrase catches.
    if "sham" in s or "saline" in s:
        return SHAM_LABEL
    if "single" in s or "lateral only" in s or "lateral hit only" in s:
        return LATERAL_LABEL
    if "large lesion" in s or "not visible" in s:
        return COMBINED_ML_LABEL
    if "medial" in s and "lateral" in s:
        return COMBINED_ML_LABEL

    return str(x).strip() or UNKNOWN_LABEL


def _load_metadata_hit_types(
    excel_path: Path,
    *,
    sheet_name: str,
    animal_col: str,
    hit_type_col: str,
) -> pd.DataFrame:
    excel_path = Path(excel_path)
    try:
        meta = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception:
        meta = pd.read_excel(excel_path, sheet_name=0)

    if animal_col not in meta.columns:
        raise ValueError(f"Metadata column {animal_col!r} not found. Found columns: {list(meta.columns)}")
    if hit_type_col not in meta.columns:
        raise ValueError(f"Metadata column {hit_type_col!r} not found. Found columns: {list(meta.columns)}")

    out = meta[[animal_col, hit_type_col]].copy()
    out = out.rename(columns={animal_col: "animal_id", hit_type_col: "raw_lesion_hit_type"})
    out["animal_id"] = out["animal_id"].map(_normalize_animal_id)
    out = out.dropna(subset=["animal_id"]).drop_duplicates(subset=["animal_id"], keep="first")
    out["lesion_hit_type"] = out["raw_lesion_hit_type"].map(_normalize_hit_type)
    return out[["animal_id", "raw_lesion_hit_type", "lesion_hit_type"]]


def _load_color_json(path: Optional[Path]) -> Dict[str, str]:
    """Load optional group colors, normalizing keys to the display group names."""
    if path is None:
        return dict(DEFAULT_LESION_COLORS)

    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    colors = dict(DEFAULT_LESION_COLORS)
    if isinstance(raw, dict):
        # Supports either {group: color} or {"colors": {group: color}}.
        if "colors" in raw and isinstance(raw["colors"], dict):
            raw = raw["colors"]
        for k, v in raw.items():
            if isinstance(v, str):
                colors[_normalize_hit_type(k)] = v
    return colors


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _available_methods(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    methods: List[str] = []
    for method in requested:
        pre_cands, post_cands, _, _ = METHOD_DEFS[method]
        if _find_first_existing_column(df, pre_cands) and _find_first_existing_column(df, post_cands):
            methods.append(method)
    return methods


def _ordered_groups(df: pd.DataFrame, order: Sequence[str]) -> List[str]:
    present = list(dict.fromkeys(df["lesion_hit_type"].astype(str)))
    return [g for g in order if g in present] + [g for g in present if g not in order]


def _jitter(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return rng.uniform(-scale, scale, size=n)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def _fmt_p_for_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def _stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _wrap_lesion_label(label: str) -> str:
    replacements = {
        SHAM_LABEL: "sham saline\ninjection",
        LATERAL_LABEL: "Lateral lesion\nonly",
        COMBINED_ML_LABEL: "Medial and\nLateral lesion",
    }
    return replacements.get(str(label), str(label).replace(" and ", " and\n"))


# -----------------------------------------------------------------------------
# Data loading and analysis tables
# -----------------------------------------------------------------------------

def collect_cluster_summaries(
    bc_root: Path,
    metadata_excel: Path,
    *,
    metadata_sheet: str,
    metadata_animal_col: str,
    metadata_hit_type_col: str,
    min_balanced_duration_s: Optional[float] = None,
) -> pd.DataFrame:
    bc_root = Path(bc_root)
    files = sorted(p for p in bc_root.rglob("*_cluster_bc_summary.csv") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No *_cluster_bc_summary.csv files found under: {bc_root}")

    parts: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue
        if df.empty:
            continue
        if "animal_id" not in df.columns:
            animal = p.name.replace("_cluster_bc_summary.csv", "")
            df["animal_id"] = animal
        df["source_csv"] = str(p)
        parts.append(df)

    if not parts:
        raise ValueError("No readable non-empty cluster summary CSVs found.")

    out = pd.concat(parts, ignore_index=True)
    out["animal_id"] = out["animal_id"].map(_normalize_animal_id)

    if "passes_min_balanced_duration" in out.columns:
        out = out[_bool_series(out["passes_min_balanced_duration"])].copy()

    if min_balanced_duration_s is not None and "balanced_duration_s_per_group" in out.columns:
        out["balanced_duration_s_per_group"] = pd.to_numeric(out["balanced_duration_s_per_group"], errors="coerce")
        out = out[out["balanced_duration_s_per_group"] >= float(min_balanced_duration_s)].copy()

    if "is_high_variance_cluster" not in out.columns:
        raise ValueError(
            "Cluster summaries do not contain 'is_high_variance_cluster'. "
            "Re-run the wrapper with --phrase-csv so high-vs-remaining sets can be defined."
        )
    out["is_high_variance_cluster"] = _bool_series(out["is_high_variance_cluster"])

    meta = _load_metadata_hit_types(
        metadata_excel,
        sheet_name=metadata_sheet,
        animal_col=metadata_animal_col,
        hit_type_col=metadata_hit_type_col,
    )
    out = out.merge(meta, on="animal_id", how="left")
    out["lesion_hit_type"] = out["lesion_hit_type"].fillna(UNKNOWN_LABEL)
    out["raw_lesion_hit_type"] = out["raw_lesion_hit_type"].fillna(UNKNOWN_LABEL)
    return out.reset_index(drop=True)


def build_analysis_tables(
    cluster_df: pd.DataFrame,
    *,
    methods: Sequence[str],
    bird_aggregate: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    set_masks = {
        "all_clusters": np.ones(len(cluster_df), dtype=bool),
        "high_variance_clusters": cluster_df["is_high_variance_cluster"].to_numpy(dtype=bool),
        "remaining_non_high_variance_clusters": (~cluster_df["is_high_variance_cluster"]).to_numpy(dtype=bool),
    }

    for method in methods:
        pre_cands, post_cands, prepost_cands, method_label = METHOD_DEFS[method]
        pre_col = _find_first_existing_column(cluster_df, pre_cands)
        post_col = _find_first_existing_column(cluster_df, post_cands)
        prepost_col = _find_first_existing_column(cluster_df, prepost_cands)
        if pre_col is None or post_col is None:
            print(f"[WARN] Skipping {method}: required BC columns not found.")
            continue

        for set_name, mask in set_masks.items():
            sub = cluster_df.loc[mask].copy()
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                bc_pre = pd.to_numeric(pd.Series([r.get(pre_col)]), errors="coerce").iloc[0]
                bc_post = pd.to_numeric(pd.Series([r.get(post_col)]), errors="coerce").iloc[0]
                bc_prepost = np.nan
                if prepost_col is not None:
                    bc_prepost = pd.to_numeric(pd.Series([r.get(prepost_col)]), errors="coerce").iloc[0]
                if not (np.isfinite(bc_pre) and np.isfinite(bc_post)):
                    continue

                rows.append({
                    "set_name": set_name,
                    "set_label": SET_LABELS.get(set_name, set_name),
                    "bc_method": method,
                    "bc_method_label": method_label,
                    "animal_id": r.get("animal_id"),
                    "cluster_id": r.get("cluster_id", r.get("cluster_label", r.get("label"))),
                    "cluster_token": r.get("cluster_token", r.get("cluster_id", r.get("cluster_label", r.get("label")))),
                    "lesion_hit_type": r.get("lesion_hit_type", UNKNOWN_LABEL),
                    "raw_lesion_hit_type": r.get("raw_lesion_hit_type", ""),
                    "bc_pre": float(bc_pre),
                    "bc_post": float(bc_post),
                    "bc_prepost": float(bc_prepost) if np.isfinite(bc_prepost) else np.nan,
                    "bc_delta_post_minus_pre": float(bc_post - bc_pre),
                    "source_csv": r.get("source_csv", ""),
                })

    cluster_long = pd.DataFrame(rows)
    if cluster_long.empty:
        raise ValueError("No finite BC rows available after filtering.")

    agg_func = np.nanmedian if bird_aggregate == "median" else np.nanmean
    bird_rows: List[Dict[str, Any]] = []
    group_cols = [
        "set_name",
        "set_label",
        "bc_method",
        "bc_method_label",
        "animal_id",
        "lesion_hit_type",
        "raw_lesion_hit_type",
    ]
    for keys, g in cluster_long.groupby(group_cols, dropna=False, sort=False):
        d = dict(zip(group_cols, keys))
        d.update({
            "n_clusters": int(len(g)),
            "bc_pre": float(agg_func(g["bc_pre"].to_numpy(dtype=float))),
            "bc_post": float(agg_func(g["bc_post"].to_numpy(dtype=float))),
            "bc_prepost": float(agg_func(g["bc_prepost"].to_numpy(dtype=float))) if np.isfinite(g["bc_prepost"]).any() else np.nan,
        })
        d["bc_delta_post_minus_pre"] = float(d["bc_post"] - d["bc_pre"])
        d["bird_aggregate"] = bird_aggregate
        bird_rows.append(d)

    bird_level = pd.DataFrame(bird_rows)
    return cluster_long, bird_level


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def _paired_pre_post_p(sub: pd.DataFrame) -> float:
    pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
    post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    if wilcoxon is None or mask.sum() < 2:
        return np.nan
    try:
        return float(wilcoxon(pre[mask], post[mask], alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _unpaired_delta_p_vs_sham(sub: pd.DataFrame, group: str, sham_label: str = SHAM_LABEL) -> float:
    if mannwhitneyu is None or group == sham_label:
        return np.nan
    a = pd.to_numeric(
        sub.loc[sub["lesion_hit_type"].astype(str) == group, "bc_delta_post_minus_pre"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    b = pd.to_numeric(
        sub.loc[sub["lesion_hit_type"].astype(str) == sham_label, "bc_delta_post_minus_pre"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def compute_stats(df: pd.DataFrame, *, level: str, order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["set_name", "set_label", "bc_method", "bc_method_label"]
    for keys, sub0 in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for lesion, sub in sub0.groupby("lesion_hit_type", sort=False):
            pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
            post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(sub["bc_delta_post_minus_pre"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(pre) & np.isfinite(post)
            p_w = _paired_pre_post_p(sub)
            rows.append({
                **base,
                "analysis_level": level,
                "lesion_hit_type": lesion,
                "n_observations": int(mask.sum()),
                "n_animals": int(sub["animal_id"].nunique()),
                "mean_bc_pre": float(np.nanmean(pre)),
                "mean_bc_post": float(np.nanmean(post)),
                "mean_delta_post_minus_pre": float(np.nanmean(delta)),
                "median_bc_pre": float(np.nanmedian(pre)),
                "median_bc_post": float(np.nanmedian(post)),
                "median_delta_post_minus_pre": float(np.nanmedian(delta)),
                "paired_pre_post_wilcoxon_p": p_w,
                "paired_pre_post_wilcoxon_p_formatted": _fmt_p(p_w),
                "stars": _stars_from_p(p_w),
            })

        if kruskal is not None:
            arrays: List[np.ndarray] = []
            labels: List[str] = []
            for lesion in _ordered_groups(sub0, order):
                vals = pd.to_numeric(
                    sub0.loc[sub0["lesion_hit_type"].astype(str) == lesion, "bc_delta_post_minus_pre"],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                if vals.size >= 2:
                    arrays.append(vals)
                    labels.append(lesion)
            if len(arrays) >= 2:
                try:
                    stat, p = kruskal(*arrays)
                except Exception:
                    stat, p = np.nan, np.nan
                rows.append({
                    **base,
                    "analysis_level": level,
                    "lesion_hit_type": "ACROSS_LESION_GROUPS",
                    "n_observations": int(sum(len(a) for a in arrays)),
                    "n_animals": int(sub0["animal_id"].nunique()),
                    "delta_kruskal_statistic": float(stat),
                    "delta_kruskal_p": float(p),
                    "delta_kruskal_p_formatted": _fmt_p(float(p)),
                    "delta_kruskal_groups": ";".join(labels),
                    "stars": _stars_from_p(float(p)),
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _boxplot_with_points(
    ax: plt.Axes,
    arrays: List[np.ndarray],
    positions: List[float],
    colors: List[str],
    *,
    width: float,
    point_size: float,
    rng: np.random.Generator,
    alpha: float = 0.75,
    show_points: bool = True,
) -> None:
    for arr, pos, color in zip(arrays, positions, colors):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        ax.boxplot(
            [arr],
            positions=[pos],
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.1, alpha=0.75),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )
        if show_points:
            x = pos + _jitter(arr.size, width * 0.24, rng)
            ax.scatter(x, arr, s=point_size, color=color, alpha=alpha, edgecolors="none", zorder=3)


def _boxplot_whisker_extents(arrays: Sequence[Sequence[float]], *, whis: float = 1.5) -> Tuple[Optional[float], Optional[float]]:
    lows: List[float] = []
    highs: List[float] = []
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size == 1:
            lows.append(float(vals[0]))
            highs.append(float(vals[0]))
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = float(q3 - q1)
        lo_fence = float(q1 - whis * iqr)
        hi_fence = float(q3 + whis * iqr)
        lower_candidates = vals[vals >= lo_fence]
        upper_candidates = vals[vals <= hi_fence]
        lows.append(float(np.min(lower_candidates)) if lower_candidates.size else float(np.min(vals)))
        highs.append(float(np.max(upper_candidates)) if upper_candidates.size else float(np.max(vals)))
    if not lows or not highs:
        return None, None
    return min(lows), max(highs)


def _ylim_from_box_whiskers(
    arrays: Sequence[Sequence[float]],
    *,
    include_zero: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    min_span: float = 0.04,
    extra_top_frac: float = 0.20,
) -> Tuple[float, float]:
    low, high = _boxplot_whisker_extents(arrays)
    if low is None or high is None:
        return (0.0, 1.0)
    if include_zero:
        low = min(float(low), 0.0)
        high = max(float(high), 0.0)
    span = float(high - low)
    if span < float(min_span):
        mid = 0.5 * (float(low) + float(high))
        span = float(min_span)
        low = mid - 0.5 * span
        high = mid + 0.5 * span
    ymin = float(low) - 0.035 * span
    ymax = float(high) + (0.10 + float(extra_top_frac)) * span
    if lower_bound is not None:
        ymin = max(float(lower_bound), ymin)
    if upper_bound is not None:
        ymax = min(float(upper_bound), ymax)
    if ymax <= ymin:
        ymax = ymin + span
    return ymin, ymax


def _auto_ylim(
    values: Sequence[float],
    *,
    include_zero: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    pad_frac: float = 0.12,
    min_span: float = 0.04,
    extra_top_frac: float = 0.0,
) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        ymin = float(np.nanmin(vals))
        ymax = float(np.nanmax(vals))
        if include_zero:
            ymin = min(ymin, 0.0)
            ymax = max(ymax, 0.0)
        span = ymax - ymin
        if span < float(min_span):
            mid = 0.5 * (ymin + ymax)
            span = float(min_span)
            ymin = mid - 0.5 * span
            ymax = mid + 0.5 * span
        pad = float(pad_frac) * span
        ymin -= pad
        ymax += pad + float(extra_top_frac) * span
    if lower_bound is not None:
        ymin = max(float(lower_bound), ymin)
    if upper_bound is not None:
        ymax = min(float(upper_bound), ymax)
    if ymax <= ymin:
        ymax = ymin + float(min_span)
    return ymin, ymax


def _draw_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str, *, h: float, fontsize: int = 13) -> None:
    if not text:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.1, clip_on=False, zorder=6)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fontsize, color="black", clip_on=False, zorder=7)


def _array_data_range_for_brackets(
    arrays: Sequence[Sequence[float]],
    *,
    boxplot_only: bool,
    include_zero: bool = False,
    min_span: float = 0.04,
) -> Tuple[float, float, float]:
    """Return data_low, data_high, data_span used for safe bracket placement.

    For boxplot-only panels, use boxplot whisker extent rather than hidden
    outliers, so the brackets sit above the visible boxplot elements.
    """
    if boxplot_only:
        low, high = _boxplot_whisker_extents(arrays)
        if low is None or high is None:
            low, high = 0.0, 1.0
    else:
        vals = np.concatenate([
            np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
            for a in arrays
            if len(np.asarray(a, dtype=float)) > 0
        ]) if arrays else np.asarray([], dtype=float)
        if vals.size == 0:
            low, high = 0.0, 1.0
        else:
            low, high = float(np.nanmin(vals)), float(np.nanmax(vals))

    if include_zero:
        low = min(float(low), 0.0)
        high = max(float(high), 0.0)

    span = float(high - low)
    if span < float(min_span):
        mid = 0.5 * (float(low) + float(high))
        span = float(min_span)
        low = mid - 0.5 * span
        high = mid + 0.5 * span
    return float(low), float(high), float(span)


def _set_ylim_with_top_bracket_zone(
    ax: plt.Axes,
    arrays: Sequence[Sequence[float]],
    *,
    boxplot_only: bool,
    include_zero: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    min_span: float = 0.04,
    bracket_rows: int = 1,
    bracket_top_padding_frac: float = 0.22,
) -> Tuple[float, float, float, float]:
    """Set y-limits with a reserved zone above the visible data for brackets.

    Returns ymin, ymax, bracket_y0, bracket_h.
    """
    data_low, data_high, data_span = _array_data_range_for_brackets(
        arrays,
        boxplot_only=boxplot_only,
        include_zero=include_zero,
        min_span=min_span,
    )

    bottom_pad = 0.06 * data_span
    top_pad = bracket_top_padding_frac * data_span + max(0, bracket_rows - 1) * 0.10 * data_span
    ymin = data_low - bottom_pad
    ymax = data_high + top_pad

    if lower_bound is not None:
        ymin = max(float(lower_bound), ymin)
    if upper_bound is not None:
        # Keep the requested upper bound as a soft ceiling, but never let it
        # crop the bracket zone.
        ymax = min(max(ymax, data_high + 0.18 * data_span), float(upper_bound))

    if ymax <= ymin:
        ymax = ymin + data_span

    ax.set_ylim(ymin, ymax)
    yr = ymax - ymin
    bracket_h = max(0.010, 0.020 * yr)

    # Put brackets in the upper part of the axes, clearly above the boxplots
    # but below the title.
    bracket_y0 = min(ymax - (bracket_rows + 1.2) * bracket_h, data_high + 0.08 * data_span)
    return ymin, ymax, bracket_y0, bracket_h


def _add_group_labels_below_prepost_ticks(
    ax: plt.Axes,
    *,
    groups: Sequence[str],
    pair_centers: Sequence[float],
    p_by_group: Mapping[str, float],
    group_fontsize: int = 14,
    p_fontsize: int = 13,
) -> None:
    """Add one centered lesion label and p-value below each Pre/Post pair.

    No bird-count text is added here. This is the main layout change that keeps
    the group labels from colliding with the p-value text.
    """
    for group, center in zip(groups, pair_centers):
        p = p_by_group.get(group, np.nan)
        p_text = f"pre/post p={_fmt_p_for_label(p)}" if np.isfinite(p) else "pre/post p=n/a"
        ax.text(
            center,
            -0.12,
            _wrap_lesion_label(group),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=group_fontsize,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            center,
            -0.27,
            p_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=p_fontsize,
            clip_on=False,
        )


def plot_pre_post_by_lesion(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    order: Sequence[str],
    level_label: str,
    dpi: int,
    boxplot_only: bool = False,
) -> None:
    """Plot pre/post BC by lesion group with brackets safely above boxes."""
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(0)
    fig_w = max(11.2, 2.25 * len(groups) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8.4))

    p_by_group: Dict[str, float] = {}
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    pair_centers: List[float] = []

    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        pre_vals = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        center = i * 1.18
        pair_centers.append(center)
        pre_x = center - 0.20
        post_x = center + 0.20
        arrays.extend([pre_vals, post_vals])
        positions.extend([pre_x, post_x])
        base = colors.get(group, "#808080")
        cols.extend([base, base])
        p_by_group[group] = _paired_pre_post_p(sub)

    _boxplot_with_points(
        ax,
        arrays,
        positions,
        cols,
        width=0.28,
        point_size=18 if level_label == "cluster" else 40,
        rng=rng,
        show_points=not bool(boxplot_only),
    )

    if not bool(boxplot_only) and len(df) <= 550:
        for _, r in df.iterrows():
            group = str(r["lesion_hit_type"])
            if group not in groups:
                continue
            center = pair_centers[groups.index(group)]
            ax.plot(
                [center - 0.20, center + 0.20],
                [r["bc_pre"], r["bc_post"]],
                color="0.78",
                linewidth=0.55,
                zorder=1,
            )

    _, _, bracket_y, bracket_h = _set_ylim_with_top_bracket_zone(
        ax,
        arrays,
        boxplot_only=bool(boxplot_only),
        include_zero=False,
        lower_bound=0.0,
        upper_bound=1.04,
        min_span=0.04,
        bracket_rows=1,
        bracket_top_padding_frac=0.34,
    )

    for i, group in enumerate(groups):
        center = pair_centers[i]
        _draw_sig_bracket(
            ax,
            center - 0.20,
            center + 0.20,
            bracket_y,
            _stars_from_p(p_by_group.get(group, np.nan)),
            h=bracket_h,
            fontsize=13,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Pre", "Post"] * len(groups), rotation=0, ha="center")
    _add_group_labels_below_prepost_ticks(
        ax,
        groups=groups,
        pair_centers=pair_centers,
        p_by_group=p_by_group,
        group_fontsize=14,
        p_fontsize=13,
    )
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=18)
    ax.set_title(title, fontsize=20, pad=42, y=1.055)
    ax.tick_params(axis="x", labelsize=16, pad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.36, top=0.78)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_delta_by_lesion(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    order: Sequence[str],
    level_label: str,
    dpi: int,
    boxplot_only: bool = False,
) -> None:
    """Plot delta BC by lesion group and compare lesion groups against sham."""
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(1)
    fig_w = max(10.0, 2.4 * len(groups) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 7.6))

    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        vals = pd.to_numeric(sub["bc_delta_post_minus_pre"], errors="coerce").to_numpy(dtype=float)
        arrays.append(vals)
        positions.append(i)
        cols.append(colors.get(group, "#808080"))

    _boxplot_with_points(
        ax,
        arrays,
        positions,
        cols,
        width=0.55,
        point_size=18 if level_label == "cluster" else 44,
        rng=rng,
        show_points=not bool(boxplot_only),
    )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.1)

    sham_present = SHAM_LABEL in groups
    p_vs_sham: Dict[str, float] = {g: np.nan for g in groups}
    if sham_present:
        for group in groups:
            if group != SHAM_LABEL:
                p_vs_sham[group] = _unpaired_delta_p_vs_sham(df, group, sham_label=SHAM_LABEL)

    n_brackets = sum(1 for g in groups if g != SHAM_LABEL and np.isfinite(p_vs_sham.get(g, np.nan)))
    _, _, bracket_y0, bracket_h = _set_ylim_with_top_bracket_zone(
        ax,
        arrays,
        boxplot_only=bool(boxplot_only),
        include_zero=True,
        min_span=0.04,
        bracket_rows=max(1, n_brackets),
        bracket_top_padding_frac=0.34 + 0.12 * max(0, n_brackets - 1),
    )

    if sham_present:
        sham_x = groups.index(SHAM_LABEL)
        comps: List[Tuple[int, int, str, float]] = []
        for group in groups:
            if group == SHAM_LABEL:
                continue
            p = p_vs_sham.get(group, np.nan)
            if np.isfinite(p):
                x = groups.index(group)
                comps.append((abs(x - sham_x), x, group, p))
        comps.sort(key=lambda t: t[0])
        y_cursor = bracket_y0
        for _, x, group, p in comps:
            _draw_sig_bracket(ax, x, sham_x, y_cursor, _stars_from_p(p), h=bracket_h, fontsize=13)
            y_cursor += 2.2 * bracket_h

    tick_labels: List[str] = []
    for g in groups:
        if g == SHAM_LABEL:
            tick_labels.append(_wrap_lesion_label(g))
        else:
            ptxt = _fmt_p_for_label(p_vs_sham.get(g, np.nan)) if np.isfinite(p_vs_sham.get(g, np.nan)) else "n/a"
            tick_labels.append(f"{_wrap_lesion_label(g)}\nvs sham p={ptxt}")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Post BC − Pre BC", fontsize=18)
    ax.set_title(title, fontsize=20, pad=38, y=1.045)
    ax.tick_params(axis="x", labelsize=14, pad=10)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.28, top=0.79)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def plot_medial_lateral_high_vs_remaining_prepost(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    level_label: str,
    dpi: int,
    boxplot_only: bool = True,
) -> None:
    """Plot medial+lateral lesion birds: high-variance clusters vs remaining clusters.

    This is the Figure 4F-style panel. It uses the same combined medial+lateral
    color as the shared palette.
    """
    _safe_mkdir(Path(out_png).parent)
    ml = df[df["lesion_hit_type"].astype(str) == COMBINED_ML_LABEL].copy()
    if ml.empty:
        return

    set_order = ["high_variance_clusters", "remaining_non_high_variance_clusters"]
    if not all(s in set(ml["set_name"].astype(str)) for s in set_order):
        return

    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(10.5, 7.6))

    set_display = {
        "high_variance_clusters": "Top 30%\nvariance clusters",
        "remaining_non_high_variance_clusters": "Remaining\nnon-top-variance clusters",
    }

    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    pair_centers: List[float] = []
    p_by_set: Dict[str, float] = {}

    ml_color = colors.get(COMBINED_ML_LABEL, "#7A4FB7")

    for i, set_name in enumerate(set_order):
        sub = ml[ml["set_name"].astype(str) == set_name].copy()
        if sub.empty:
            continue
        pre_vals = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)

        center = i * 1.70
        pair_centers.append(center)
        pre_x = center - 0.26
        post_x = center + 0.26

        arrays.extend([pre_vals, post_vals])
        positions.extend([pre_x, post_x])
        cols.extend([ml_color, ml_color])
        p_by_set[set_name] = _paired_pre_post_p(sub)

    if not arrays:
        plt.close(fig)
        return

    _boxplot_with_points(
        ax,
        arrays,
        positions,
        cols,
        width=0.42,
        point_size=18 if level_label == "cluster" else 44,
        rng=rng,
        show_points=not bool(boxplot_only),
    )

    if not bool(boxplot_only) and len(ml) <= 550:
        for _, r in ml.iterrows():
            set_name = str(r["set_name"])
            if set_name not in set_order:
                continue
            center = pair_centers[set_order.index(set_name)]
            ax.plot(
                [center - 0.26, center + 0.26],
                [r["bc_pre"], r["bc_post"]],
                color="0.78",
                linewidth=0.55,
                zorder=1,
            )

    _, _, bracket_y, bracket_h = _set_ylim_with_top_bracket_zone(
        ax,
        arrays,
        boxplot_only=bool(boxplot_only),
        include_zero=False,
        lower_bound=0.0,
        upper_bound=1.04,
        min_span=0.04,
        bracket_rows=1,
        bracket_top_padding_frac=0.36,
    )

    for i, set_name in enumerate(set_order):
        if i >= len(pair_centers):
            continue
        center = pair_centers[i]
        _draw_sig_bracket(
            ax,
            center - 0.26,
            center + 0.26,
            bracket_y,
            _stars_from_p(p_by_set.get(set_name, np.nan)),
            h=bracket_h,
            fontsize=13,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Pre", "Post"] * len(pair_centers), rotation=0, ha="center")

    # Labels below each pair.
    for center, set_name in zip(pair_centers, set_order):
        p = p_by_set.get(set_name, np.nan)
        p_text = f"pre/post p={_fmt_p_for_label(p)}" if np.isfinite(p) else "pre/post p=n/a"
        ax.text(
            center,
            -0.13,
            set_display.get(set_name, set_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=15,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            center,
            -0.29,
            p_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14,
            clip_on=False,
        )

    ax.set_ylabel("Bhattacharyya coefficient", fontsize=18)
    ax.set_title(title, fontsize=20, pad=42, y=1.055)
    ax.tick_params(axis="x", labelsize=16, pad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.36, top=0.78)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot orchestration
# -----------------------------------------------------------------------------

def make_all_plots(
    cluster_long: pd.DataFrame,
    bird_level: pd.DataFrame,
    out_dir: Path,
    *,
    colors: Dict[str, str],
    lesion_order: Sequence[str],
    dpi: int,
    include_bird_count_in_title: bool = False,
) -> List[Path]:
    paths: List[Path] = []
    for level_label, df in [("cluster", cluster_long), ("bird", bird_level)]:
        level_dir = Path(out_dir) / f"{level_label}_level"
        _safe_mkdir(level_dir)

        # Standard panels: all clusters, high-variance clusters, remaining clusters.
        for (set_name, method), sub in df.groupby(["set_name", "bc_method"], sort=False):
            if sub.empty:
                continue
            set_label = SET_LABELS.get(set_name, set_name)
            method_label = METHOD_DEFS.get(method, (None, None, None, method))[3]
            n_obs = len(sub)
            title_base = f"{level_label.capitalize()} level: {set_label}\n{method_label} | N={n_obs} {level_label}s"
            if include_bird_count_in_title:
                n_birds = sub["animal_id"].nunique()
                title_base += f", {n_birds} birds"

            token = f"{level_label}_level_{_clean_filename(method)}_{SET_FILENAME_TOKEN.get(set_name, _clean_filename(set_name))}"

            p1 = level_dir / f"{token}_pre_vs_post_by_lesion.png"
            plot_pre_post_by_lesion(sub, p1, title=title_base, colors=colors, order=lesion_order, level_label=level_label, dpi=dpi, boxplot_only=False)
            paths.append(p1)

            p1_box = level_dir / f"{token}_pre_vs_post_by_lesion_boxplots_only.png"
            plot_pre_post_by_lesion(sub, p1_box, title=title_base, colors=colors, order=lesion_order, level_label=level_label, dpi=dpi, boxplot_only=True)
            paths.append(p1_box)

            p2 = level_dir / f"{token}_delta_post_minus_pre_by_lesion.png"
            plot_delta_by_lesion(sub, p2, title=title_base, colors=colors, order=lesion_order, level_label=level_label, dpi=dpi, boxplot_only=False)
            paths.append(p2)

            p2_box = level_dir / f"{token}_delta_post_minus_pre_by_lesion_boxplots_only.png"
            plot_delta_by_lesion(sub, p2_box, title=title_base, colors=colors, order=lesion_order, level_label=level_label, dpi=dpi, boxplot_only=True)
            paths.append(p2_box)

        # Figure 4F-style special panel:
        # Medial+lateral lesion group only, top-variance clusters versus remaining clusters.
        for method, sub_method in df.groupby("bc_method", sort=False):
            method_label = METHOD_DEFS.get(method, (None, None, None, method))[3]
            ml = sub_method[sub_method["lesion_hit_type"].astype(str) == COMBINED_ML_LABEL]
            if ml.empty:
                continue
            n_units = ml["animal_id"].nunique() if level_label == "bird" else len(ml)
            title = (
                f"{level_label.capitalize()} level: {COMBINED_ML_LABEL}\n"
                f"Top 30% variance vs remaining clusters\n"
                f"{method_label} | N={n_units} {'birds' if level_label == 'bird' else 'clusters'}"
            )
            token = f"medial_lateral_only_{level_label}_level_{_clean_filename(method)}_high_vs_remaining_prepost"

            p_special = level_dir / f"{token}.png"
            plot_medial_lateral_high_vs_remaining_prepost(
                sub_method,
                p_special,
                title=title,
                colors=colors,
                level_label=level_label,
                dpi=dpi,
                boxplot_only=False,
            )
            paths.append(p_special)

            p_special_box = level_dir / f"{token}_boxplots_only.png"
            plot_medial_lateral_high_vs_remaining_prepost(
                sub_method,
                p_special_box,
                title=title,
                colors=colors,
                level_label=level_label,
                dpi=dpi,
                boxplot_only=True,
            )
            paths.append(p_special_box)

    return paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch lesion-hit-type BC summary plots, including remaining non-top-variance clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bc-root", required=True, help="Root folder containing per-bird *_cluster_bc_summary.csv files.")
    p.add_argument("--metadata-excel", required=True, help="Excel file with animal IDs and lesion hit types.")
    p.add_argument("--out-dir", default=None, help="Output folder. Defaults to <bc-root>/_batch_lesion_group_summaries.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--metadata-animal-col", default="Animal ID")
    p.add_argument("--metadata-hit-type-col", default="Lesion hit type")
    p.add_argument("--bc-method", choices=list(METHOD_DEFS.keys()) + ["all"], default="selected_bins")
    p.add_argument("--bird-aggregate", choices=["median", "mean"], default="median")
    p.add_argument(
        "--min-balanced-duration-s",
        type=float,
        default=None,
        help="Optional extra duration filter. Usually not needed because each cluster summary already has passes_min_balanced_duration.",
    )
    p.add_argument("--color-json", default=None, help="Optional lesion-group color JSON. Supports either {group: color} or {'colors': {group: color}}.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--include-bird-count-in-title",
        action="store_true",
        help="Add ', N birds' to the title. Default is off so the plot is cleaner and p-value text has more room.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bc_root = Path(args.bc_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else bc_root / "_batch_lesion_group_summaries"
    _safe_mkdir(out_dir)

    cluster_df = collect_cluster_summaries(
        bc_root,
        Path(args.metadata_excel).expanduser().resolve(),
        metadata_sheet=str(args.metadata_sheet),
        metadata_animal_col=str(args.metadata_animal_col),
        metadata_hit_type_col=str(args.metadata_hit_type_col),
        min_balanced_duration_s=args.min_balanced_duration_s,
    )

    requested_methods = list(METHOD_DEFS.keys()) if args.bc_method == "all" else [args.bc_method]
    methods = _available_methods(cluster_df, requested_methods)
    if not methods:
        raise ValueError(
            f"None of the requested BC methods are available. Requested: {requested_methods}. "
            f"Columns: {list(cluster_df.columns)}"
        )

    cluster_long, bird_level = build_analysis_tables(
        cluster_df,
        methods=methods,
        bird_aggregate=str(args.bird_aggregate),
    )

    colors = _load_color_json(Path(args.color_json).expanduser().resolve() if args.color_json else None)
    lesion_order = [g for g in DEFAULT_LESION_ORDER if g in set(cluster_long["lesion_hit_type"].astype(str))]
    lesion_order += [g for g in sorted(set(cluster_long["lesion_hit_type"].astype(str))) if g not in lesion_order]

    cluster_csv = out_dir / "bc_batch_cluster_level_long.csv"
    bird_csv = out_dir / "bc_batch_bird_level_summary.csv"
    cluster_long.to_csv(cluster_csv, index=False)
    bird_level.to_csv(bird_csv, index=False)
    print(f"[SAVE] {cluster_csv}")
    print(f"[SAVE] {bird_csv}")

    stats = pd.concat([
        compute_stats(cluster_long, level="cluster", order=lesion_order),
        compute_stats(bird_level, level="bird", order=lesion_order),
    ], ignore_index=True)
    stats_csv = out_dir / "bc_batch_lesion_group_stats.csv"
    stats.to_csv(stats_csv, index=False)
    print(f"[SAVE] {stats_csv}")

    paths = make_all_plots(
        cluster_long,
        bird_level,
        out_dir,
        colors=colors,
        lesion_order=lesion_order,
        dpi=int(args.dpi),
        include_bird_count_in_title=bool(args.include_bird_count_in_title),
    )
    for pth in paths:
        print(f"[SAVE] {pth}")

    manifest = pd.DataFrame({
        "bc_root": [str(bc_root)],
        "out_dir": [str(out_dir)],
        "n_cluster_summary_files": [cluster_df["source_csv"].nunique()],
        "n_animals": [cluster_long["animal_id"].nunique()],
        "n_cluster_level_rows": [len(cluster_long)],
        "n_bird_level_rows": [len(bird_level)],
        "bc_methods": [";".join(methods)],
        "sets": [";".join(SET_LABELS.keys())],
        "title_includes_bird_count": [bool(args.include_bird_count_in_title)],
        "color_sham": [colors.get(SHAM_LABEL, "")],
        "color_lateral_only": [colors.get(LATERAL_LABEL, "")],
        "color_combined_medial_lateral": [colors.get(COMBINED_ML_LABEL, "")],
        "color_unknown": [colors.get(UNKNOWN_LABEL, "")],
    })
    manifest_csv = out_dir / "bc_batch_lesion_group_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    print(f"[SAVE] {manifest_csv}")


if __name__ == "__main__":
    main()
