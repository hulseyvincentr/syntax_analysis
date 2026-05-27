#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_batch_lesion_group_summary_with_remaining_v11_source_transition_entropy.py

Post-process a batch of per-bird *_cluster_bc_summary.csv files and make
lesion-hit-type comparison plots for three cluster sets:

  1) all qualifying clusters
  2) high-variance phrase-duration clusters (top-fraction set marked by the wrapper)
  3) remaining non-high-variance clusters (the complement of #2)

This version is tuned for publication-style consistency across figures:
  - combines complete + partial medial/lateral lesion groups into one
    "Medial and Lateral lesion" group
  - uses a consistent gray/lavender color scheme
  - writes boxplot-only versions of the pre/post and delta figures
  - removes the extra "N birds" text from plot titles by default, so the
    p-value annotations have more room
  - places Pre/Post labels on the x-axis and lesion-group + p-value text below
    each Pre/Post pair
  - optionally computes bird-level mean per-syllable transition entropy
    H(next label | current label) from NPZ label sequences for high-variance
    phrase-duration clusters and for the remaining non-high-variance clusters, then plots those subsets with the same lesion
    colors and grouped Pre/Post layout

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
from datetime import date, datetime, timedelta
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

# Chosen to match the gray/lavender style of the current BC figures.
DEFAULT_LESION_COLORS = {
    SHAM_LABEL: "#D2D2D2",
    LATERAL_LABEL: "#DCD8EA",
    COMBINED_ML_LABEL: "#C6B6DC",
    UNKNOWN_LABEL: "#BDBDBD",
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
    """Plot pre/post BC by lesion group with p-values below each Pre/Post pair."""
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(0)
    fig_w = max(11.2, 2.25 * len(groups) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8.4))

    p_by_group: Dict[str, float] = {}
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    all_values: List[float] = []
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
        all_values.extend(pre_vals[np.isfinite(pre_vals)].tolist())
        all_values.extend(post_vals[np.isfinite(post_vals)].tolist())
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
            ax.plot([center - 0.20, center + 0.20], [r["bc_pre"], r["bc_post"]], color="0.78", linewidth=0.55, zorder=1)

    if bool(boxplot_only):
        ymin, ymax = _ylim_from_box_whiskers(arrays, include_zero=False, lower_bound=0.0, upper_bound=1.04, min_span=0.04, extra_top_frac=0.22)
    else:
        ymin, ymax = _auto_ylim(all_values, lower_bound=0.0, upper_bound=1.04, pad_frac=0.10, min_span=0.08, extra_top_frac=0.24)
    ax.set_ylim(ymin, ymax)
    yr = ymax - ymin
    bracket_h = max(0.012, 0.018 * yr)

    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        vals = pd.to_numeric(pd.concat([sub["bc_pre"], sub["bc_post"]]), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        if bool(boxplot_only):
            _, local_top0 = _boxplot_whisker_extents([sub["bc_pre"], sub["bc_post"]])
            local_top = float(local_top0) if local_top0 is not None else float(np.nanmax(vals))
        else:
            local_top = float(np.nanmax(vals))
        y = min(ymax - 3.0 * bracket_h, local_top + 0.05 * yr)
        center = pair_centers[i]
        _draw_sig_bracket(ax, center - 0.20, center + 0.20, y, _stars_from_p(p_by_group.get(group, np.nan)), h=bracket_h, fontsize=13)

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
    ax.set_title(title, fontsize=20, pad=34, y=1.045)
    ax.tick_params(axis="x", labelsize=16, pad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.36, top=0.81)
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
    all_values: List[float] = []
    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        vals = pd.to_numeric(sub["bc_delta_post_minus_pre"], errors="coerce").to_numpy(dtype=float)
        arrays.append(vals)
        positions.append(i)
        cols.append(colors.get(group, "#808080"))
        all_values.extend(vals[np.isfinite(vals)].tolist())

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
    if bool(boxplot_only):
        ymin, ymax = _ylim_from_box_whiskers(arrays, include_zero=True, min_span=0.04, extra_top_frac=0.28 + 0.16 * max(0, n_brackets - 1))
    else:
        ymin, ymax = _auto_ylim(all_values, include_zero=True, pad_frac=0.14, min_span=0.08, extra_top_frac=0.22 + 0.12 * max(0, n_brackets - 1))
    ax.set_ylim(ymin, ymax)
    yr = ymax - ymin
    bracket_h = max(0.010, 0.030 * yr)

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
        y_cursor = ymax - (n_brackets + 1.5) * bracket_h
        for _, x, group, p in comps:
            _draw_sig_bracket(ax, x, sham_x, y_cursor, _stars_from_p(p), h=bracket_h, fontsize=13)
            y_cursor += 2.1 * bracket_h

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
    ax.set_title(title, fontsize=20, pad=30, y=1.04)
    ax.tick_params(axis="x", labelsize=14, pad=10)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.28, top=0.82)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



# -----------------------------------------------------------------------------
# Optional transition entropy / sequence predictability analysis
# -----------------------------------------------------------------------------

TRANSITION_ENTROPY_SET_LABELS = {
    "all_clusters": "All syllables",
    "high_variance_clusters": "Top 30% phrase-duration variance clusters",
    "remaining_non_high_variance_clusters": "Remaining non-top-variance clusters",
}

TRANSITION_ENTROPY_SET_FILENAME_TOKEN = {
    "all_clusters": "all_syllables",
    "high_variance_clusters": "high_variance_clusters",
    "remaining_non_high_variance_clusters": "remaining_non_high_variance_clusters",
}


def _normalize_label_token(x: Any) -> str:
    """Normalize cluster/syllable labels so CSV ids and NPZ labels compare reliably."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if np.isfinite(float(x)) and float(x).is_integer():
            return str(int(x))
        return str(x).strip()
    s = str(x).strip()
    if s == "":
        return ""
    # Common CSV artifact: cluster labels saved as "3.0" while NPZ labels are 3.
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _label_array_to_tokens(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).astype(str)
    if arr.dtype.kind == "f":
        out = np.empty(arr.shape[0], dtype=object)
        finite = np.isfinite(arr)
        int_like = finite & (np.abs(arr - np.round(arr)) < 1e-9)
        out[int_like] = np.round(arr[int_like]).astype(np.int64).astype(str)
        out[~int_like] = arr[~int_like].astype(str)
        return out.astype(str)
    return np.asarray([_normalize_label_token(x) for x in arr], dtype=str)


def _load_treatment_dates(
    excel_path: Path,
    *,
    sheet_name: str,
    animal_col: str,
    treatment_date_col: str,
) -> pd.DataFrame:
    """Load one treatment date per animal from the metadata workbook."""
    excel_path = Path(excel_path)
    try:
        meta = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception:
        meta = pd.read_excel(excel_path, sheet_name=0)

    if animal_col not in meta.columns:
        raise ValueError(f"Metadata date column {animal_col!r} not found. Found columns: {list(meta.columns)}")
    if treatment_date_col not in meta.columns:
        raise ValueError(f"Metadata date column {treatment_date_col!r} not found. Found columns: {list(meta.columns)}")

    out = meta[[animal_col, treatment_date_col]].copy()
    out = out.rename(columns={animal_col: "animal_id", treatment_date_col: "treatment_date"})
    out["animal_id"] = out["animal_id"].map(_normalize_animal_id)
    out["treatment_date"] = pd.to_datetime(out["treatment_date"], errors="coerce").dt.date
    out = out.dropna(subset=["animal_id", "treatment_date"]).drop_duplicates(subset=["animal_id"], keep="first")
    return out


def _parse_date_from_text(text: Any) -> Optional[date]:
    """Parse common recording-date tokens from filenames or file_map entries."""
    if text is None:
        return None
    s = str(text)

    # ISO-like dates: 2024-04-01, 2024_04_01, 20240401.
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})[-_./]?([01]\d)[-_./]?([0-3]\d)(?!\d)", s):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue

    # Month-day-year tokens: 040124, 04-01-24, 04_01_2024.
    for m in re.finditer(r"(?<!\d)([01]?\d)[-_./]?([0-3]?\d)[-_./]?((?:20)?\d{2})(?!\d)", s):
        mm = int(m.group(1))
        dd = int(m.group(2))
        yy = int(m.group(3))
        if yy < 100:
            yy += 2000
        if not (2010 <= yy <= 2040):
            continue
        try:
            return date(yy, mm, dd)
        except ValueError:
            continue

    # Excel serial dates occasionally appear in exported filenames/metadata.
    for m in re.finditer(r"(?<!\d)(4\d{4}|5\d{4})(?!\d)", s):
        serial = int(m.group(1))
        try:
            d = date(1899, 12, 30) + timedelta(days=serial)
            if date(2010, 1, 1) <= d <= date(2040, 12, 31):
                return d
        except Exception:
            continue

    return None


def _load_npz_file_map(npz: np.lib.npyio.NpzFile, file_map_key: str) -> Dict[Any, str]:
    if file_map_key not in npz.files:
        return {}
    obj = npz[file_map_key]
    try:
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
    except Exception:
        pass

    out: Dict[Any, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out[k] = str(v)
            try:
                out[int(k)] = str(v)
            except Exception:
                pass
            out[str(k)] = str(v)
        return out

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            vals = obj.tolist()
        else:
            vals = list(obj)
    elif isinstance(obj, (list, tuple)):
        vals = list(obj)
    else:
        vals = []

    for i, v in enumerate(vals):
        out[i] = str(v)
        out[str(i)] = str(v)
    return out


def _lookup_file_map_text(file_map: Mapping[Any, str], file_idx: Any) -> str:
    if file_idx in file_map:
        return str(file_map[file_idx])
    try:
        i = int(file_idx)
        if i in file_map:
            return str(file_map[i])
        if str(i) in file_map:
            return str(file_map[str(i)])
    except Exception:
        pass
    s = str(file_idx)
    if s in file_map:
        return str(file_map[s])
    return s


def _find_npz_for_animal(npz_root: Path, animal_id: str) -> Optional[Path]:
    npz_root = Path(npz_root)
    animal_id = str(animal_id)
    candidates = [
        npz_root / animal_id / f"{animal_id}.npz",
        npz_root / f"{animal_id}.npz",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    hits = sorted(npz_root.rglob(f"{animal_id}.npz"))
    if hits:
        return hits[0]
    hits = sorted(p for p in npz_root.rglob("*.npz") if animal_id in p.name or animal_id in str(p.parent))
    return hits[0] if hits else None


def _period_mask_from_file_dates(
    *,
    file_indices: np.ndarray,
    file_map: Mapping[Any, str],
    treatment_date: date,
    treatment_day_assignment: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return boolean pre/post masks for each time bin, based on file dates."""
    file_indices = np.asarray(file_indices)
    unique_fids, inv = np.unique(file_indices, return_inverse=True)
    period_codes = np.zeros(unique_fids.shape[0], dtype=np.int8)  # -1 pre, 0 excluded/unknown, +1 post
    parsed_dates: Dict[str, Optional[date]] = {}
    n_unknown = 0

    for i, fid in enumerate(unique_fids):
        txt = _lookup_file_map_text(file_map, fid)
        d = _parse_date_from_text(txt)
        parsed_dates[str(fid)] = d
        if d is None:
            n_unknown += 1
            period_codes[i] = 0
            continue
        if d < treatment_date:
            period_codes[i] = -1
        elif d > treatment_date:
            period_codes[i] = 1
        else:
            if treatment_day_assignment == "pre":
                period_codes[i] = -1
            elif treatment_day_assignment == "post":
                period_codes[i] = 1
            else:
                period_codes[i] = 0

    by_bin = period_codes[inv]
    info = {
        "n_unique_file_indices": int(len(unique_fids)),
        "n_file_indices_with_unknown_date": int(n_unknown),
        "n_pre_bins": int(np.sum(by_bin == -1)),
        "n_post_bins": int(np.sum(by_bin == 1)),
        "treatment_date": str(treatment_date),
        "treatment_day_assignment": str(treatment_day_assignment),
    }
    return by_bin == -1, by_bin == 1, info


def _weighted_outgoing_transition_entropy(
    *,
    label_tokens: np.ndarray,
    source_tokens: Sequence[str],
    period_mask: np.ndarray,
    file_indices: Optional[np.ndarray] = None,
    include_noise: bool = False,
    destination_mode: str = "all",
) -> Dict[str, Any]:
    """Compute summary entropy statistics for selected source syllables.

    The returned field ``mean_source_entropy_bits`` is the unweighted mean of
    H(next label | current/source label) across the selected source labels that
    are actually observed in the requested period. ``tte_bits`` is also returned
    for reference, but downstream plots/statistics use the unweighted mean so
    transition-entropy outputs stay conceptually separate from TTE.
    """
    source_set = {str(x) for x in source_tokens if str(x) != ""}
    if not source_set:
        return {
            "tte_bits": np.nan,
            "n_source_labels_requested": 0,
            "n_source_labels_observed": 0,
            "n_transitions_used": 0,
            "mean_source_entropy_bits": np.nan,
            "median_source_entropy_bits": np.nan,
        }

    labels = np.asarray(label_tokens, dtype=str)
    if labels.shape[0] < 2:
        return {
            "tte_bits": np.nan,
            "n_source_labels_requested": len(source_set),
            "n_source_labels_observed": 0,
            "n_transitions_used": 0,
            "mean_source_entropy_bits": np.nan,
            "median_source_entropy_bits": np.nan,
        }

    src = labels[:-1]
    dst = labels[1:]
    valid = np.asarray(period_mask[:-1], dtype=bool) & np.asarray(period_mask[1:], dtype=bool)

    if file_indices is not None:
        f = np.asarray(file_indices)
        if f.shape[0] == labels.shape[0]:
            valid &= (f[:-1] == f[1:])

    if not include_noise:
        bad = {"-1", "nan", "None", ""}
        valid &= ~np.isin(src, list(bad))
        valid &= ~np.isin(dst, list(bad))

    valid &= np.isin(src, list(source_set))
    if destination_mode == "same_set":
        valid &= np.isin(dst, list(source_set))

    if not np.any(valid):
        return {
            "tte_bits": np.nan,
            "n_source_labels_requested": len(source_set),
            "n_source_labels_observed": 0,
            "n_transitions_used": 0,
            "mean_source_entropy_bits": np.nan,
            "median_source_entropy_bits": np.nan,
        }

    src_sel = src[valid]
    dst_sel = dst[valid]
    trans = pd.DataFrame({"src": src_sel, "dst": dst_sel})
    counts = trans.value_counts(["src", "dst"]).rename("n").reset_index()
    source_totals = counts.groupby("src", sort=False)["n"].sum()
    total_transitions = float(source_totals.sum())
    if total_transitions <= 0:
        tte = np.nan
        entropy_by_source = pd.Series(dtype=float)
    else:
        counts["source_total"] = counts["src"].map(source_totals)
        counts["p_b_given_a"] = counts["n"].astype(float) / counts["source_total"].astype(float)
        counts["entropy_term"] = -counts["p_b_given_a"] * np.log2(counts["p_b_given_a"])
        entropy_by_source = counts.groupby("src", sort=False)["entropy_term"].sum()
        p_a = source_totals.astype(float) / total_transitions
        tte = float((p_a * entropy_by_source).sum())

    return {
        "tte_bits": tte,
        "n_source_labels_requested": int(len(source_set)),
        "n_source_labels_observed": int(len(source_totals)),
        "n_transitions_used": int(total_transitions),
        "mean_source_entropy_bits": float(np.nanmean(entropy_by_source.to_numpy(dtype=float))) if len(entropy_by_source) else np.nan,
        "median_source_entropy_bits": float(np.nanmedian(entropy_by_source.to_numpy(dtype=float))) if len(entropy_by_source) else np.nan,
    }


def _collect_successor_tokens(
    *,
    label_tokens: np.ndarray,
    source_tokens: Sequence[str],
    period_mask: np.ndarray,
    file_indices: Optional[np.ndarray] = None,
    include_noise: bool = False,
    destination_mode: str = "all",
) -> List[str]:
    """Return the unique syllable labels that directly follow the selected source syllables.

    If source_tokens correspond to high-variance syllables, the returned labels are
    the *successor syllables* observed one step later in the same period. Those
    successor labels can then be treated as source labels in a second entropy
    calculation, which yields the successor transition entropy.
    """
    source_set = {str(x) for x in source_tokens if str(x) != ""}
    if not source_set:
        return []

    labels = np.asarray(label_tokens, dtype=str)
    if labels.shape[0] < 2:
        return []

    src = labels[:-1]
    dst = labels[1:]
    valid = np.asarray(period_mask[:-1], dtype=bool) & np.asarray(period_mask[1:], dtype=bool)

    if file_indices is not None:
        f = np.asarray(file_indices)
        if f.shape[0] == labels.shape[0]:
            valid &= (f[:-1] == f[1:])

    if not include_noise:
        bad = {"-1", "nan", "None", ""}
        valid &= ~np.isin(src, list(bad))
        valid &= ~np.isin(dst, list(bad))

    valid &= np.isin(src, list(source_set))
    if destination_mode == "same_set":
        valid &= np.isin(dst, list(source_set))

    if not np.any(valid):
        return []

    return sorted({str(x) for x in dst[valid].tolist() if str(x) != ""})


def _source_entropy_frame(
    *,
    label_tokens: np.ndarray,
    period_mask: np.ndarray,
    file_indices: Optional[np.ndarray] = None,
    include_noise: bool = False,
    allowed_sources: Optional[Sequence[str]] = None,
    allowed_destinations: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return one row per source label with H(next label | source label).

    This is the core cluster-level transition entropy table. Each row is one
    source cluster/syllable label in one period for one bird.
    """
    labels = np.asarray(label_tokens, dtype=str)
    columns = [
        "source_label",
        "source_entropy_bits",
        "n_source_transitions",
        "n_observed_destinations",
        "observed_destination_labels",
    ]
    if labels.shape[0] < 2:
        return pd.DataFrame(columns=columns)

    src = labels[:-1]
    dst = labels[1:]
    valid = np.asarray(period_mask[:-1], dtype=bool) & np.asarray(period_mask[1:], dtype=bool)

    if file_indices is not None:
        f = np.asarray(file_indices)
        if f.shape[0] == labels.shape[0]:
            valid &= (f[:-1] == f[1:])

    if not include_noise:
        bad = {"-1", "nan", "None", ""}
        valid &= ~np.isin(src, list(bad))
        valid &= ~np.isin(dst, list(bad))

    if allowed_sources is not None:
        allowed_source_set = {str(x) for x in allowed_sources if str(x) != ""}
        valid &= np.isin(src, list(allowed_source_set))

    if allowed_destinations is not None:
        allowed_dest_set = {str(x) for x in allowed_destinations if str(x) != ""}
        valid &= np.isin(dst, list(allowed_dest_set))

    if not np.any(valid):
        return pd.DataFrame(columns=columns)

    trans = pd.DataFrame({"src": src[valid], "dst": dst[valid]})
    counts = trans.value_counts(["src", "dst"]).rename("n").reset_index()
    source_totals = counts.groupby("src", sort=False)["n"].sum()
    counts["source_total"] = counts["src"].map(source_totals)
    counts["p_b_given_a"] = counts["n"].astype(float) / counts["source_total"].astype(float)
    counts["entropy_term"] = -counts["p_b_given_a"] * np.log2(counts["p_b_given_a"])

    entropy_by_source = counts.groupby("src", sort=False)["entropy_term"].sum()
    dests_by_source = counts.groupby("src", sort=False)["dst"].agg(lambda x: ";".join(sorted({str(v) for v in x})))
    n_dests_by_source = counts.groupby("src", sort=False)["dst"].nunique()

    out = pd.DataFrame({
        "source_label": entropy_by_source.index.astype(str),
        "source_entropy_bits": entropy_by_source.to_numpy(dtype=float),
        "n_source_transitions": entropy_by_source.index.map(source_totals).astype(int).to_numpy(),
        "n_observed_destinations": entropy_by_source.index.map(n_dests_by_source).astype(int).to_numpy(),
        "observed_destination_labels": entropy_by_source.index.map(dests_by_source).astype(str).to_numpy(),
    })
    return out


def _successors_by_source(
    *,
    label_tokens: np.ndarray,
    source_tokens: Sequence[str],
    period_mask: np.ndarray,
    file_indices: Optional[np.ndarray] = None,
    include_noise: bool = False,
    allowed_destinations: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """Map each selected source label to the unique labels that follow it."""
    source_set = {str(x) for x in source_tokens if str(x) != ""}
    out = {str(x): [] for x in source_set}
    if not source_set:
        return out

    labels = np.asarray(label_tokens, dtype=str)
    if labels.shape[0] < 2:
        return out

    src = labels[:-1]
    dst = labels[1:]
    valid = np.asarray(period_mask[:-1], dtype=bool) & np.asarray(period_mask[1:], dtype=bool)

    if file_indices is not None:
        f = np.asarray(file_indices)
        if f.shape[0] == labels.shape[0]:
            valid &= (f[:-1] == f[1:])

    if not include_noise:
        bad = {"-1", "nan", "None", ""}
        valid &= ~np.isin(src, list(bad))
        valid &= ~np.isin(dst, list(bad))

    valid &= np.isin(src, list(source_set))
    if allowed_destinations is not None:
        allowed_dest_set = {str(x) for x in allowed_destinations if str(x) != ""}
        valid &= np.isin(dst, list(allowed_dest_set))

    if not np.any(valid):
        return out

    tmp = pd.DataFrame({"src": src[valid], "dst": dst[valid]})
    for source, sub in tmp.groupby("src", sort=False):
        out[str(source)] = sorted({str(x) for x in sub["dst"].tolist() if str(x) != ""})
    return out


def _lookup_source_entropy(row_map: Mapping[str, Mapping[str, Any]], token: str, column: str, default: Any = np.nan) -> Any:
    info = row_map.get(str(token))
    if info is None:
        return default
    return info.get(column, default)


def _mean_entropy_for_tokens(row_map: Mapping[str, Mapping[str, Any]], tokens: Sequence[str]) -> Tuple[float, int, int]:
    """Return mean entropy, number of tokens requested, and number with finite entropy."""
    vals: List[float] = []
    for token in tokens:
        val = _lookup_source_entropy(row_map, str(token), "source_entropy_bits", np.nan)
        try:
            val_f = float(val)
        except Exception:
            val_f = np.nan
        if np.isfinite(val_f):
            vals.append(val_f)
    if len(vals) == 0:
        return np.nan, int(len(tokens)), 0
    return float(np.nanmean(vals)), int(len(tokens)), int(len(vals))


def build_transition_entropy_table(
    cluster_df: pd.DataFrame,
    metadata_excel: Path,
    *,
    npz_root: Path,
    metadata_date_sheet: str,
    metadata_date_animal_col: str,
    metadata_treatment_date_col: str,
    label_key: str = "hdbscan_labels",
    file_index_key: str = "file_indices",
    file_map_key: str = "file_map",
    treatment_day_assignment: str = "exclude",
    include_noise: bool = False,
    destination_mode: str = "all",
) -> pd.DataFrame:
    """Compute cluster-level pre/post transition entropy tables.

    Each output row is one selected cluster/syllable label from one bird. This
    avoids averaging by bird before plotting: the boxplots pool cluster-level
    transition entropy values within each lesion hit type.

    ``analysis_kind == 'source'`` gives H(next label | selected cluster).
    ``analysis_kind == 'successor'`` first finds labels that directly follow the
    selected cluster, then averages H(next label | successor label) across those
    successor labels for that selected cluster and period.
    """
    if destination_mode not in {"all", "same_set"}:
        raise ValueError("destination_mode must be 'all' or 'same_set'")

    treatment_dates = _load_treatment_dates(
        metadata_excel,
        sheet_name=metadata_date_sheet,
        animal_col=metadata_date_animal_col,
        treatment_date_col=metadata_treatment_date_col,
    )
    treatment_by_animal = dict(zip(treatment_dates["animal_id"], treatment_dates["treatment_date"]))

    rows: List[Dict[str, Any]] = []
    for animal_id, sub0 in cluster_df.groupby("animal_id", sort=False):
        animal_id = str(animal_id)
        treatment_date = treatment_by_animal.get(animal_id)
        if treatment_date is None:
            print(f"[WARN] Skipping entropy for {animal_id}: no treatment date found.")
            continue

        npz_path = _find_npz_for_animal(Path(npz_root), animal_id)
        if npz_path is None:
            print(f"[WARN] Skipping entropy for {animal_id}: no NPZ found under {npz_root}.")
            continue

        sub = sub0.copy()
        label_col = _find_first_existing_column(sub, ["cluster_id", "cluster_label", "label", "cluster_token"])
        if label_col is None:
            print(f"[WARN] Skipping entropy for {animal_id}: no cluster id/label column found in cluster summary.")
            continue

        sub["_cluster_token_norm"] = sub[label_col].map(_normalize_label_token)
        sub = sub[sub["_cluster_token_norm"].astype(str) != ""].copy()
        if sub.empty:
            print(f"[WARN] Skipping entropy for {animal_id}: cluster summary has no usable cluster labels.")
            continue

        high_tokens = sorted(sub.loc[sub["is_high_variance_cluster"], "_cluster_token_norm"].astype(str).unique().tolist())
        rem_tokens = sorted(sub.loc[~sub["is_high_variance_cluster"], "_cluster_token_norm"].astype(str).unique().tolist())
        all_tokens = sorted(sub["_cluster_token_norm"].astype(str).unique().tolist())
        token_info = {str(r["_cluster_token_norm"]): r.to_dict() for _, r in sub.drop_duplicates("_cluster_token_norm").iterrows()}

        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                if label_key not in npz.files:
                    print(f"[WARN] Skipping entropy for {animal_id}: label key {label_key!r} not in {npz_path.name}. Keys: {npz.files}")
                    continue
                if file_index_key not in npz.files:
                    print(f"[WARN] Skipping entropy for {animal_id}: file-index key {file_index_key!r} not in {npz_path.name}.")
                    continue
                labels_raw = np.asarray(npz[label_key])
                file_indices = np.asarray(npz[file_index_key])
                if labels_raw.shape[0] != file_indices.shape[0]:
                    print(
                        f"[WARN] Skipping entropy for {animal_id}: {label_key} length {labels_raw.shape[0]} "
                        f"does not match {file_index_key} length {file_indices.shape[0]}."
                    )
                    continue
                label_tokens = _label_array_to_tokens(labels_raw)
                file_map = _load_npz_file_map(npz, file_map_key)
        except Exception as e:
            print(f"[WARN] Skipping entropy for {animal_id}: could not load NPZ {npz_path}: {e}")
            continue

        pre_mask, post_mask, period_info = _period_mask_from_file_dates(
            file_indices=file_indices,
            file_map=file_map,
            treatment_date=treatment_date,
            treatment_day_assignment=treatment_day_assignment,
        )
        if not np.any(pre_mask) or not np.any(post_mask):
            print(f"[WARN] Entropy for {animal_id}: pre/post bins missing after date split: {period_info}")

        lesion_hit_type = str(sub["lesion_hit_type"].iloc[0]) if "lesion_hit_type" in sub.columns and len(sub) else UNKNOWN_LABEL
        raw_lesion_hit_type = str(sub["raw_lesion_hit_type"].iloc[0]) if "raw_lesion_hit_type" in sub.columns and len(sub) else ""

        # All-destination entropy for every observed source label. This is used
        # for successor entropy after identifying the successor labels.
        pre_all_entropy = _source_entropy_frame(
            label_tokens=label_tokens,
            period_mask=pre_mask,
            file_indices=file_indices,
            include_noise=include_noise,
        )
        post_all_entropy = _source_entropy_frame(
            label_tokens=label_tokens,
            period_mask=post_mask,
            file_indices=file_indices,
            include_noise=include_noise,
        )
        pre_all_map = {str(r["source_label"]): r for _, r in pre_all_entropy.iterrows()}
        post_all_map = {str(r["source_label"]): r for _, r in post_all_entropy.iterrows()}

        token_sets = {
            "all_clusters": all_tokens,
            "high_variance_clusters": high_tokens,
            "remaining_non_high_variance_clusters": rem_tokens,
        }

        for set_name, tokens in token_sets.items():
            same_set_destinations = tokens if destination_mode == "same_set" else None

            pre_source_entropy = _source_entropy_frame(
                label_tokens=label_tokens,
                period_mask=pre_mask,
                file_indices=file_indices,
                include_noise=include_noise,
                allowed_sources=tokens,
                allowed_destinations=same_set_destinations,
            )
            post_source_entropy = _source_entropy_frame(
                label_tokens=label_tokens,
                period_mask=post_mask,
                file_indices=file_indices,
                include_noise=include_noise,
                allowed_sources=tokens,
                allowed_destinations=same_set_destinations,
            )
            pre_source_map = {str(r["source_label"]): r for _, r in pre_source_entropy.iterrows()}
            post_source_map = {str(r["source_label"]): r for _, r in post_source_entropy.iterrows()}

            pre_successors = _successors_by_source(
                label_tokens=label_tokens,
                source_tokens=tokens,
                period_mask=pre_mask,
                file_indices=file_indices,
                include_noise=include_noise,
                allowed_destinations=same_set_destinations,
            )
            post_successors = _successors_by_source(
                label_tokens=label_tokens,
                source_tokens=tokens,
                period_mask=post_mask,
                file_indices=file_indices,
                include_noise=include_noise,
                allowed_destinations=same_set_destinations,
            )

            for token in tokens:
                info = token_info.get(str(token), {})
                is_high = bool(set_name == "high_variance_clusters")
                source_csv = info.get("source_csv", "")

                pre_source_value = _lookup_source_entropy(pre_source_map, token, "source_entropy_bits", np.nan)
                post_source_value = _lookup_source_entropy(post_source_map, token, "source_entropy_bits", np.nan)
                pre_source_transitions = _lookup_source_entropy(pre_source_map, token, "n_source_transitions", 0)
                post_source_transitions = _lookup_source_entropy(post_source_map, token, "n_source_transitions", 0)
                pre_source_dests = _lookup_source_entropy(pre_source_map, token, "observed_destination_labels", "")
                post_source_dests = _lookup_source_entropy(post_source_map, token, "observed_destination_labels", "")
                pre_source_ndests = _lookup_source_entropy(pre_source_map, token, "n_observed_destinations", 0)
                post_source_ndests = _lookup_source_entropy(post_source_map, token, "n_observed_destinations", 0)

                rows.append({
                    "analysis_level": "cluster",
                    "analysis_kind": "source",
                    "analysis_label": "Source transition entropy",
                    "set_name": set_name,
                    "set_label": TRANSITION_ENTROPY_SET_LABELS.get(set_name, set_name),
                    "animal_id": animal_id,
                    "cluster_id": str(token),
                    "is_high_variance_cluster": is_high,
                    "lesion_hit_type": lesion_hit_type,
                    "raw_lesion_hit_type": raw_lesion_hit_type,
                    "npz_path": str(npz_path),
                    "source_csv": str(source_csv),
                    "label_key": label_key,
                    "file_index_key": file_index_key,
                    "file_map_key": file_map_key,
                    "destination_mode": destination_mode,
                    "treatment_date": str(treatment_date),
                    "transition_entropy_pre": float(pre_source_value) if np.isfinite(pd.to_numeric(pd.Series([pre_source_value]), errors="coerce").iloc[0]) else np.nan,
                    "transition_entropy_post": float(post_source_value) if np.isfinite(pd.to_numeric(pd.Series([post_source_value]), errors="coerce").iloc[0]) else np.nan,
                    "transition_entropy_delta_post_minus_pre": (
                        float(post_source_value - pre_source_value)
                        if np.isfinite(pd.to_numeric(pd.Series([pre_source_value]), errors="coerce").iloc[0]) and np.isfinite(pd.to_numeric(pd.Series([post_source_value]), errors="coerce").iloc[0])
                        else np.nan
                    ),
                    "n_transitions_pre": int(pre_source_transitions) if pd.notna(pre_source_transitions) else 0,
                    "n_transitions_post": int(post_source_transitions) if pd.notna(post_source_transitions) else 0,
                    "n_observed_destinations_pre": int(pre_source_ndests) if pd.notna(pre_source_ndests) else 0,
                    "n_observed_destinations_post": int(post_source_ndests) if pd.notna(post_source_ndests) else 0,
                    "observed_destination_labels_pre": str(pre_source_dests),
                    "observed_destination_labels_post": str(post_source_dests),
                    **{f"date_split_{k}": v for k, v in period_info.items()},
                })

                pre_succ_tokens = pre_successors.get(str(token), [])
                post_succ_tokens = post_successors.get(str(token), [])
                pre_succ_mean, pre_n_succ, pre_n_succ_with_entropy = _mean_entropy_for_tokens(pre_all_map, pre_succ_tokens)
                post_succ_mean, post_n_succ, post_n_succ_with_entropy = _mean_entropy_for_tokens(post_all_map, post_succ_tokens)

                rows.append({
                    "analysis_level": "cluster",
                    "analysis_kind": "successor",
                    "analysis_label": "Successor transition entropy",
                    "set_name": set_name,
                    "set_label": TRANSITION_ENTROPY_SET_LABELS.get(set_name, set_name),
                    "animal_id": animal_id,
                    "cluster_id": str(token),
                    "is_high_variance_cluster": is_high,
                    "lesion_hit_type": lesion_hit_type,
                    "raw_lesion_hit_type": raw_lesion_hit_type,
                    "npz_path": str(npz_path),
                    "source_csv": str(source_csv),
                    "label_key": label_key,
                    "file_index_key": file_index_key,
                    "file_map_key": file_map_key,
                    "destination_mode": destination_mode,
                    "treatment_date": str(treatment_date),
                    "successor_labels_pre": ";".join(pre_succ_tokens),
                    "successor_labels_post": ";".join(post_succ_tokens),
                    "n_successor_labels_pre": int(pre_n_succ),
                    "n_successor_labels_post": int(post_n_succ),
                    "n_successor_labels_with_entropy_pre": int(pre_n_succ_with_entropy),
                    "n_successor_labels_with_entropy_post": int(post_n_succ_with_entropy),
                    "transition_entropy_pre": float(pre_succ_mean) if np.isfinite(pre_succ_mean) else np.nan,
                    "transition_entropy_post": float(post_succ_mean) if np.isfinite(post_succ_mean) else np.nan,
                    "transition_entropy_delta_post_minus_pre": (
                        float(post_succ_mean - pre_succ_mean)
                        if np.isfinite(pre_succ_mean) and np.isfinite(post_succ_mean)
                        else np.nan
                    ),
                    **{f"date_split_{k}": v for k, v in period_info.items()},
                })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[WARN] No cluster-level transition entropy rows were created.")
    return out


def _paired_pre_post_p_for_cols(sub: pd.DataFrame, pre_col: str, post_col: str) -> float:
    pre = pd.to_numeric(sub[pre_col], errors="coerce").to_numpy(dtype=float)
    post = pd.to_numeric(sub[post_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    if wilcoxon is None or mask.sum() < 2:
        return np.nan
    try:
        return float(wilcoxon(pre[mask], post[mask], alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def compute_transition_entropy_stats(te_df: pd.DataFrame, *, order: Sequence[str]) -> pd.DataFrame:
    """Group statistics for cluster-level source and successor transition entropy."""
    rows: List[Dict[str, Any]] = []
    if te_df.empty:
        return pd.DataFrame(rows)
    for (analysis_kind, set_name), sub0 in te_df.groupby(["analysis_kind", "set_name"], sort=False):
        set_label = TRANSITION_ENTROPY_SET_LABELS.get(set_name, set_name)
        analysis_label = str(sub0["analysis_label"].iloc[0]) if "analysis_label" in sub0.columns and len(sub0) else analysis_kind
        for lesion, sub in sub0.groupby("lesion_hit_type", sort=False):
            pre = pd.to_numeric(sub["transition_entropy_pre"], errors="coerce").to_numpy(dtype=float)
            post = pd.to_numeric(sub["transition_entropy_post"], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(sub["transition_entropy_delta_post_minus_pre"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(pre) & np.isfinite(post)
            p_w = _paired_pre_post_p_for_cols(sub, "transition_entropy_pre", "transition_entropy_post")
            rows.append({
                "analysis_level": "cluster",
                "analysis_kind": analysis_kind,
                "analysis_label": analysis_label,
                "set_name": set_name,
                "set_label": set_label,
                "lesion_hit_type": lesion,
                "metric": f"cluster_level_mean_{analysis_kind}_transition_entropy_bits",
                "n_clusters_with_pre_post": int(mask.sum()),
                "n_animals": int(sub.loc[mask, "animal_id"].nunique()) if "animal_id" in sub.columns else np.nan,
                "mean_transition_entropy_pre": float(np.nanmean(pre)) if np.isfinite(pre).any() else np.nan,
                "mean_transition_entropy_post": float(np.nanmean(post)) if np.isfinite(post).any() else np.nan,
                "mean_delta_post_minus_pre": float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan,
                "median_transition_entropy_pre": float(np.nanmedian(pre)) if np.isfinite(pre).any() else np.nan,
                "median_transition_entropy_post": float(np.nanmedian(post)) if np.isfinite(post).any() else np.nan,
                "median_delta_post_minus_pre": float(np.nanmedian(delta)) if np.isfinite(delta).any() else np.nan,
                "paired_pre_post_wilcoxon_p": p_w,
                "paired_pre_post_wilcoxon_p_formatted": _fmt_p(p_w),
                "stars": _stars_from_p(p_w),
            })
    return pd.DataFrame(rows)


def plot_transition_entropy_pre_post_by_lesion(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    y_label: str,
    colors: Dict[str, str],
    order: Sequence[str],
    dpi: int,
    boxplot_only: bool = False,
) -> None:
    """Plot cluster-level pre/post transition entropy by lesion group."""
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(2)
    fig_w = max(11.2, 2.25 * len(groups) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8.4))

    p_by_group: Dict[str, float] = {}
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    all_values: List[float] = []
    pair_centers: List[float] = []

    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group].copy()
        pre_vals = pd.to_numeric(sub["transition_entropy_pre"], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(sub["transition_entropy_post"], errors="coerce").to_numpy(dtype=float)
        center = i * 1.18
        pair_centers.append(center)
        pre_x = center - 0.20
        post_x = center + 0.20
        arrays.extend([pre_vals, post_vals])
        positions.extend([pre_x, post_x])
        base = colors.get(group, "#808080")
        cols.extend([base, base])
        all_values.extend(pre_vals[np.isfinite(pre_vals)].tolist())
        all_values.extend(post_vals[np.isfinite(post_vals)].tolist())
        p_by_group[group] = _paired_pre_post_p_for_cols(sub, "transition_entropy_pre", "transition_entropy_post")

    _boxplot_with_points(
        ax,
        arrays,
        positions,
        cols,
        width=0.28,
        point_size=42,
        rng=rng,
        show_points=False,
    )

    if not bool(boxplot_only):
        for i, group in enumerate(groups):
            sub = df[df["lesion_hit_type"].astype(str) == group].copy()
            base = colors.get(group, "#808080")
            center = pair_centers[i]
            pre_x = center - 0.20
            post_x = center + 0.20
            for _, r in sub.iterrows():
                pre = pd.to_numeric(pd.Series([r.get("transition_entropy_pre")]), errors="coerce").iloc[0]
                post = pd.to_numeric(pd.Series([r.get("transition_entropy_post")]), errors="coerce").iloc[0]
                if not (np.isfinite(pre) and np.isfinite(post)):
                    continue
                x0 = pre_x + float(rng.uniform(-0.04, 0.04))
                x1 = post_x + float(rng.uniform(-0.04, 0.04))
                ax.plot([x0, x1], [pre, post], color=base, alpha=0.40, linewidth=1.0, zorder=2)
                ax.scatter([x0], [pre], s=52, facecolors="none", edgecolors=base, linewidths=1.6, zorder=4)
                ax.scatter([x1], [post], s=52, facecolors=base, edgecolors=base, linewidths=1.0, zorder=4)

    if bool(boxplot_only):
        ymin, ymax = _ylim_from_box_whiskers(arrays, include_zero=False, lower_bound=0.0, min_span=0.10, extra_top_frac=0.22)
    else:
        ymin, ymax = _auto_ylim(all_values, lower_bound=0.0, pad_frac=0.10, min_span=0.20, extra_top_frac=0.24)
    ax.set_ylim(ymin, ymax)
    yr = ymax - ymin
    bracket_h = max(0.025, 0.018 * yr)

    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        vals = pd.to_numeric(pd.concat([sub["transition_entropy_pre"], sub["transition_entropy_post"]]), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        if bool(boxplot_only):
            lo_w, hi_w = _boxplot_whisker_extents([
                pd.to_numeric(sub["transition_entropy_pre"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(sub["transition_entropy_post"], errors="coerce").to_numpy(dtype=float),
            ])
            local_top = float(hi_w) if hi_w is not None else float(np.nanmax(vals))
        else:
            local_top = float(np.nanmax(vals))
        y = min(ymax - 2.4 * bracket_h, local_top + 0.05 * yr)
        center = pair_centers[i]
        _draw_sig_bracket(ax, center - 0.20, center + 0.20, y, _stars_from_p(p_by_group.get(group, np.nan)), h=bracket_h, fontsize=13)

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
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(title, fontsize=20, pad=34, y=1.045)
    ax.tick_params(axis="x", labelsize=16, pad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.36, top=0.81)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_transition_entropy_plots(
    tte_df: pd.DataFrame,
    out_dir: Path,
    *,
    colors: Dict[str, str],
    lesion_order: Sequence[str],
    dpi: int,
    include_bird_count_in_title: bool = False,
) -> List[Path]:
    paths: List[Path] = []
    if tte_df.empty:
        return paths
    entropy_dir = Path(out_dir) / "transition_entropy"
    _safe_mkdir(entropy_dir)
    ylabels = {
        "source": "Transition entropy per syllable (bits)",
        "successor": "Successor transition entropy per syllable (bits)",
    }
    title_prefix = {
        "source": "Cluster level: Transition Entropy",
        "successor": "Cluster level: Successor Transition Entropy",
    }
    for (analysis_kind, set_name), sub in tte_df.groupby(["analysis_kind", "set_name"], sort=False):
        if sub.empty:
            continue
        set_label = TRANSITION_ENTROPY_SET_LABELS.get(set_name, set_name)
        n_birds = sub["animal_id"].nunique()
        title = f"{title_prefix.get(analysis_kind, 'Bird level: Transition Entropy')}\n{set_label}"
        if include_bird_count_in_title:
            title += f" | N={n_birds} birds"
        token = f"cluster_level_{analysis_kind}_transition_entropy_{TRANSITION_ENTROPY_SET_FILENAME_TOKEN.get(set_name, _clean_filename(set_name))}"

        p1 = entropy_dir / f"{token}_pre_vs_post_by_lesion.png"
        plot_transition_entropy_pre_post_by_lesion(
            sub,
            p1,
            title=title,
            y_label=ylabels.get(analysis_kind, "Transition entropy per syllable (bits)"),
            colors=colors,
            order=lesion_order,
            dpi=dpi,
            boxplot_only=False,
        )
        paths.append(p1)

        p2 = entropy_dir / f"{token}_pre_vs_post_by_lesion_boxplots_only.png"
        plot_transition_entropy_pre_post_by_lesion(
            sub,
            p2,
            title=title,
            y_label=ylabels.get(analysis_kind, "Transition entropy per syllable (bits)"),
            colors=colors,
            order=lesion_order,
            dpi=dpi,
            boxplot_only=True,
        )
        paths.append(p2)
    return paths

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

    return paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute cluster-level source/successor transition entropy for high-variance phrase-duration "
            "clusters, remaining non-high-variance clusters, and all clusters/syllables. This transition-entropy-only "
            "script does not generate Bhattacharyya coefficient plots or statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cluster-summary-root",
        default=None,
        help=(
            "Root folder containing per-bird *_cluster_bc_summary.csv files. These files are used "
            "only to identify high-variance versus remaining clusters."
        ),
    )
    p.add_argument(
        "--bc-root",
        default=None,
        help=(
            "Backward-compatible alias for --cluster-summary-root. This script does not compute BC; "
            "it only reads the cluster summary files to define syllable subsets."
        ),
    )
    p.add_argument("--metadata-excel", required=True, help="Excel file with animal IDs, lesion hit types, and treatment dates.")
    p.add_argument("--out-dir", required=True, help="Output folder for transition entropy figures, CSVs, stats, and manifest.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--metadata-animal-col", default="Animal ID")
    p.add_argument("--metadata-hit-type-col", default="Lesion hit type")
    p.add_argument(
        "--min-balanced-duration-s",
        type=float,
        default=None,
        help="Optional extra duration filter applied to the cluster-summary rows before defining high-vs-remaining labels.",
    )
    p.add_argument("--color-json", default=None, help="Optional lesion-group color JSON. Supports either {group: color} or {'colors': {group: color}}.")

    # Transition-entropy analysis inputs.
    p.add_argument(
        "--npz-root",
        required=True,
        help="Root folder containing per-bird NPZ files with label and file-index sequences.",
    )
    p.add_argument("--label-key", default="hdbscan_labels", help="NPZ key containing the cluster/syllable labels used for transition entropy.")
    p.add_argument("--file-index-key", default="file_indices", help="NPZ key mapping each time bin to a source file/segment index.")
    p.add_argument("--file-map-key", default="file_map", help="NPZ key mapping file indices to filenames/segment names with dates.")
    p.add_argument("--metadata-date-sheet", default="metadata", help="Metadata sheet containing treatment dates.")
    p.add_argument("--metadata-date-animal-col", default="Animal ID", help="Animal ID column in the treatment-date metadata sheet.")
    p.add_argument("--metadata-treatment-date-col", default="Treatment date", help="Treatment-date column in the metadata workbook.")
    p.add_argument(
        "--treatment-day-assignment",
        choices=["exclude", "pre", "post"],
        default="exclude",
        help="How to handle recordings dated exactly on the treatment day for transition entropy.",
    )
    p.add_argument(
        "--entropy-destination-mode",
        choices=["all", "same_set"],
        default="all",
        help=(
            "For selected source syllables a, compute entropy over transitions to all destination syllables b "
            "or only to destinations in the same selected set. Default 'all' asks how predictable the next "
            "syllable is after a high-variance/stuttered syllable."
        ),
    )
    p.add_argument("--entropy-include-noise", action="store_true", help="Include noise label -1 in transition entropy calculations.")

    # Kept for backward compatibility with older commands; it has no effect now.
    p.add_argument(
        "--make-transition-entropy",
        action="store_true",
        help="Ignored. This script is transition-entropy-only and always runs the transition entropy analysis.",
    )
    p.add_argument(
        "--bc-method",
        default=None,
        help="Ignored. Kept only so older commands do not fail; no BC calculations are run.",
    )
    p.add_argument(
        "--bird-aggregate",
        default=None,
        help="Ignored. Kept only so older commands do not fail; no BC bird aggregation is run.",
    )

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--include-bird-count-in-title",
        action="store_true",
        help="Add ', N birds' to the title. Default is off so the plot is cleaner and p-value text has more room.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cluster_summary_root_arg = args.cluster_summary_root or args.bc_root
    if not cluster_summary_root_arg:
        raise ValueError("Please provide --cluster-summary-root. You can also use the backward-compatible alias --bc-root.")

    cluster_summary_root = Path(cluster_summary_root_arg).expanduser().resolve()
    metadata_excel = Path(args.metadata_excel).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _safe_mkdir(out_dir)

    print("[INFO] Transition-entropy-only mode: no Bhattacharyya coefficient plots or BC statistics will be generated. Source and successor transition-entropy outputs will be created.")
    print(f"[INFO] Reading cluster summaries only to define high-vs-remaining syllable sets: {cluster_summary_root}")

    cluster_df = collect_cluster_summaries(
        cluster_summary_root,
        metadata_excel,
        metadata_sheet=str(args.metadata_sheet),
        metadata_animal_col=str(args.metadata_animal_col),
        metadata_hit_type_col=str(args.metadata_hit_type_col),
        min_balanced_duration_s=args.min_balanced_duration_s,
    )

    colors = _load_color_json(Path(args.color_json).expanduser().resolve() if args.color_json else None)
    lesion_order = [g for g in DEFAULT_LESION_ORDER if g in set(cluster_df["lesion_hit_type"].astype(str))]
    lesion_order += [g for g in sorted(set(cluster_df["lesion_hit_type"].astype(str))) if g not in lesion_order]

    te_df = build_transition_entropy_table(
        cluster_df,
        metadata_excel,
        npz_root=npz_root,
        metadata_date_sheet=str(args.metadata_date_sheet),
        metadata_date_animal_col=str(args.metadata_date_animal_col),
        metadata_treatment_date_col=str(args.metadata_treatment_date_col),
        label_key=str(args.label_key),
        file_index_key=str(args.file_index_key),
        file_map_key=str(args.file_map_key),
        treatment_day_assignment=str(args.treatment_day_assignment),
        include_noise=bool(args.entropy_include_noise),
        destination_mode=str(args.entropy_destination_mode),
    )

    entropy_dir = out_dir / "transition_entropy"
    _safe_mkdir(entropy_dir)

    combined_csv = entropy_dir / "transition_entropy_cluster_level_summary.csv"
    te_df.to_csv(combined_csv, index=False)
    print(f"[SAVE] {combined_csv}")

    source_df = te_df[te_df["analysis_kind"].astype(str) == "source"].copy() if not te_df.empty and "analysis_kind" in te_df.columns else pd.DataFrame()
    successor_df = te_df[te_df["analysis_kind"].astype(str) == "successor"].copy() if not te_df.empty and "analysis_kind" in te_df.columns else pd.DataFrame()

    source_csv = entropy_dir / "source_transition_entropy_cluster_level_summary.csv"
    successor_csv = entropy_dir / "successor_transition_entropy_cluster_level_summary.csv"
    source_df.to_csv(source_csv, index=False)
    successor_df.to_csv(successor_csv, index=False)
    print(f"[SAVE] {source_csv}")
    print(f"[SAVE] {successor_csv}")

    te_stats = compute_transition_entropy_stats(te_df, order=lesion_order)
    combined_stats_csv = entropy_dir / "transition_entropy_lesion_group_stats.csv"
    te_stats.to_csv(combined_stats_csv, index=False)
    print(f"[SAVE] {combined_stats_csv}")

    source_stats = te_stats[te_stats["analysis_kind"].astype(str) == "source"].copy() if not te_stats.empty and "analysis_kind" in te_stats.columns else pd.DataFrame()
    successor_stats = te_stats[te_stats["analysis_kind"].astype(str) == "successor"].copy() if not te_stats.empty and "analysis_kind" in te_stats.columns else pd.DataFrame()
    source_stats_csv = entropy_dir / "source_transition_entropy_lesion_group_stats.csv"
    successor_stats_csv = entropy_dir / "successor_transition_entropy_lesion_group_stats.csv"
    source_stats.to_csv(source_stats_csv, index=False)
    successor_stats.to_csv(successor_stats_csv, index=False)
    print(f"[SAVE] {source_stats_csv}")
    print(f"[SAVE] {successor_stats_csv}")

    te_paths = make_transition_entropy_plots(
        te_df,
        out_dir,
        colors=colors,
        lesion_order=lesion_order,
        dpi=int(args.dpi),
        include_bird_count_in_title=bool(args.include_bird_count_in_title),
    )
    for pth in te_paths:
        print(f"[SAVE] {pth}")

    manifest = pd.DataFrame({
        "mode": ["source_and_successor_transition_entropy_cluster_level_with_all_syllables"],
        "cluster_summary_root": [str(cluster_summary_root)],
        "npz_root": [str(npz_root)],
        "out_dir": [str(out_dir)],
        "n_cluster_summary_files": [int(cluster_df["source_csv"].nunique()) if "source_csv" in cluster_df.columns else 0],
        "n_animals_in_cluster_summaries": [int(cluster_df["animal_id"].nunique()) if "animal_id" in cluster_df.columns else 0],
        "n_transition_entropy_rows": [int(len(te_df))],
        "n_clusters_with_transition_entropy": [int(te_df[["animal_id", "cluster_id"]].drop_duplicates().shape[0]) if isinstance(te_df, pd.DataFrame) and not te_df.empty and "cluster_id" in te_df.columns else 0],
        "n_animals_with_transition_entropy": [int(te_df["animal_id"].nunique()) if isinstance(te_df, pd.DataFrame) and not te_df.empty else 0],
        "sets": [";".join(TRANSITION_ENTROPY_SET_LABELS.keys())],
        "analysis_kinds": ["source;successor"],
        "lesion_order": [";".join(lesion_order)],
        "title_includes_bird_count": [bool(args.include_bird_count_in_title)],
        "transition_entropy_destination_mode": [str(args.entropy_destination_mode)],
        "metrics": ["cluster_level_source_transition_entropy_bits;cluster_level_successor_transition_entropy_bits"],
        "no_bc_outputs_generated": [True],
    })
    manifest_csv = entropy_dir / "transition_entropy_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    print(f"[SAVE] {manifest_csv}")

    print({
        "out_dir": str(out_dir),
        "transition_entropy_dir": str(entropy_dir),
        "summary_csv": str(combined_csv),
        "source_summary_csv": str(source_csv),
        "successor_summary_csv": str(successor_csv),
        "stats_csv_with_p_values": str(combined_stats_csv),
        "source_stats_csv_with_p_values": str(source_stats_csv),
        "successor_stats_csv_with_p_values": str(successor_stats_csv),
        "plots": [str(p) for p in te_paths],
    })


if __name__ == "__main__":
    main()
