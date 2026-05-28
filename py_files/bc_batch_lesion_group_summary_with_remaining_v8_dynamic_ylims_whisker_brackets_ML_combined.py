#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_batch_lesion_group_summary_with_remaining_v5_boxplots_only.py

Post-process a batch of per-bird *_cluster_bc_summary.csv files and make
lesion-hit-type comparison plots for three cluster sets:

  1) all qualifying clusters
  2) high-variance phrase-duration clusters (top-fraction set marked by the wrapper)
  3) remaining non-high-variance clusters (the complement of #2)

For each set, the script makes both:
  - cluster-level plots: every cluster is one point
  - bird-level plots: clusters are first aggregated within each bird, then each bird is one point

Expected input files are the per-bird outputs from bc_cluster_qc_and_summaries_*.py,
especially the files named:
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import wilcoxon, kruskal, mannwhitneyu
except Exception:  # pragma: no cover
    wilcoxon = None
    kruskal = None
    mannwhitneyu = None


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

# For the final lesion-group comparison, combine complete and partial
# medial+lateral lesion animals into one group. This keeps the batch plots focused
# on: medial+lateral lesion, lateral-only lesion, and sham controls.
COMBINED_ML_LABEL = "Medial and Lateral lesion"
LATERAL_LABEL = "Lateral lesion only"
SHAM_LABEL = "sham saline injection"

DEFAULT_LESION_ORDER = [
    SHAM_LABEL,
    LATERAL_LABEL,
    COMBINED_ML_LABEL,
    "unknown",
]

DEFAULT_LESION_COLORS = {
    COMBINED_ML_LABEL: "#4B148C",
    LATERAL_LABEL: "#9A8FBF",
    SHAM_LABEL: "#707070",
    "unknown": "#BDBDBD",
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


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_filename(s: str) -> str:
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
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
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
    if s in partial:
        return COMBINED_ML_LABEL
    if s in complete:
        return COMBINED_ML_LABEL

    # Catch partial phrases conservatively.
    if "sham" in s or "saline" in s:
        return SHAM_LABEL
    if "single" in s or "lateral only" in s or "lateral hit only" in s:
        return LATERAL_LABEL
    if "large lesion" in s or "not visible" in s:
        return COMBINED_ML_LABEL
    if "medial" in s and "lateral" in s:
        return COMBINED_ML_LABEL
    return str(x).strip() or "unknown"


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
    if path is None:
        return dict(DEFAULT_LESION_COLORS)
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    colors = dict(DEFAULT_LESION_COLORS)
    if isinstance(raw, dict):
        # Supports either {group: color} or {"colors": {group: color}}
        if "colors" in raw and isinstance(raw["colors"], dict):
            raw = raw["colors"]
        for k, v in raw.items():
            if isinstance(v, str):
                group = _normalize_hit_type(k)
                colors[str(group)] = v
    return colors


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _available_methods(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    methods = []
    for method in requested:
        pre_cands, post_cands, prepost_cands, _ = METHOD_DEFS[method]
        if _find_first_existing_column(df, pre_cands) and _find_first_existing_column(df, post_cands):
            methods.append(method)
    return methods


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

    parts = []
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
    out["lesion_hit_type"] = out["lesion_hit_type"].fillna("unknown")
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
                    "cluster_id": r.get("cluster_id"),
                    "cluster_token": r.get("cluster_token", r.get("cluster_id")),
                    "lesion_hit_type": r.get("lesion_hit_type", "unknown"),
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

    # Bird-level: aggregate clusters within each bird/set/method.
    agg_func = np.nanmedian if bird_aggregate == "median" else np.nanmean
    bird_rows: List[Dict[str, Any]] = []
    group_cols = ["set_name", "set_label", "bc_method", "bc_method_label", "animal_id", "lesion_hit_type", "raw_lesion_hit_type"]
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


def _ordered_groups(df: pd.DataFrame, order: Sequence[str]) -> List[str]:
    present = list(dict.fromkeys(df["lesion_hit_type"].astype(str)))
    return [g for g in order if g in present] + [g for g in present if g not in order]


def _jitter(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return rng.uniform(-scale, scale, size=n)


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
            boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.1, alpha=0.25),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )
        if bool(show_points):
            x = pos + _jitter(arr.size, width * 0.24, rng)
            ax.scatter(x, arr, s=point_size, color=color, alpha=alpha, edgecolors="none", zorder=3)



def _fmt_p_for_label(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def _wrap_lesion_label(label: str) -> str:
    """Make lesion labels readable when placed under grouped Pre/Post ticks."""
    s = str(label)
    replacements = {
        "sham saline injection": "sham saline\ninjection",
        "Lateral lesion only": "Lateral lesion\nonly",
        "Medial and Lateral lesion": "Medial and\nLateral lesion",
    }
    return replacements.get(s, s.replace(" and ", " and\n"))


def _add_group_labels_below_prepost_ticks(
    ax: plt.Axes,
    *,
    groups: Sequence[str],
    pair_centers: Sequence[float],
    p_by_group: Mapping[str, float],
    group_fontsize: int = 14,
    p_fontsize: int = 13,
) -> None:
    """Add one centered lesion label and p-value below each Pre/Post pair."""
    for group, center in zip(groups, pair_centers):
        group_text = _wrap_lesion_label(group)
        p = p_by_group.get(group, np.nan)
        p_text = f"pre/post p={_fmt_p_for_label(p).replace('p=', '')}" if np.isfinite(p) else "pre/post p=n/a"
        ax.text(
            center,
            -0.115,
            group_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=group_fontsize,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            center,
            -0.255,
            p_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=p_fontsize,
            clip_on=False,
        )


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


def _draw_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str, *, h: float = 0.025, fontsize: int = 11) -> None:
    if not text:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.1, clip_on=False, zorder=6)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fontsize, color="black", clip_on=False, zorder=7)


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
    """Return compact y-limits that show the observed data with modest padding."""
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ymin, ymax = (0.0, 1.0)
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



def _boxplot_whisker_limits(
    values: Sequence[float],
    *,
    whis: float = 1.5,
) -> Tuple[Optional[float], Optional[float]]:
    """Return the visible Tukey boxplot whisker limits for one array.

    This matches Matplotlib's default 1.5*IQR whiskers with showfliers=False.
    Hidden outlier points are therefore not allowed to stretch boxplot-only
    figure limits.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        v = float(vals[0])
        return v, v

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return float(np.min(vals)), float(np.max(vals))

    lo_fence = float(q1 - whis * iqr)
    hi_fence = float(q3 + whis * iqr)
    lower_candidates = vals[vals >= lo_fence]
    upper_candidates = vals[vals <= hi_fence]
    lo = float(np.min(lower_candidates)) if lower_candidates.size else float(np.min(vals))
    hi = float(np.max(upper_candidates)) if upper_candidates.size else float(np.max(vals))
    return lo, hi


def _boxplot_whisker_extents(
    arrays: Sequence[Sequence[float]],
    *,
    whis: float = 1.5,
) -> Tuple[Optional[float], Optional[float]]:
    """Return global min lower whisker and max upper whisker."""
    lows: List[float] = []
    highs: List[float] = []
    for arr in arrays:
        lo, hi = _boxplot_whisker_limits(arr, whis=whis)
        if lo is not None and hi is not None:
            lows.append(float(lo))
            highs.append(float(hi))
    if not lows or not highs:
        return None, None
    return min(lows), max(highs)


def _boxplot_top_whiskers(arrays: Sequence[Sequence[float]]) -> List[float]:
    """Return the top whisker for each plotted box."""
    tops: List[float] = []
    for arr in arrays:
        _, hi = _boxplot_whisker_limits(arr)
        tops.append(float(hi) if hi is not None else np.nan)
    return tops


def _data_extents(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return float(np.min(vals)), float(np.max(vals))


def _base_ylim_for_boxplot(
    arrays: Sequence[Sequence[float]],
    *,
    plotted_values: Optional[Sequence[float]] = None,
    include_zero: bool = False,
    use_whiskers_only: bool = True,
    min_span: float = 0.012,
    lower_pad_frac: float = 0.08,
    upper_pad_frac: float = 0.10,
    upper_bound: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """Compute a compact y-limit from the visible boxplot range.

    Returns ymin, ymax, low, high, span where low/high are the underlying range
    before padding. For boxplot-only figures, low/high are the visible whiskers.
    For point-overlay figures, low/high also include the overlaid raw points.
    """
    low, high = _boxplot_whisker_extents(arrays)
    if not use_whiskers_only and plotted_values is not None:
        raw_low, raw_high = _data_extents(plotted_values)
        if raw_low is not None and raw_high is not None:
            low = raw_low if low is None else min(float(low), float(raw_low))
            high = raw_high if high is None else max(float(high), float(raw_high))

    if low is None or high is None:
        low, high = 0.0, 1.0
    low = float(low)
    high = float(high)

    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)

    span = float(high - low)
    if span < float(min_span):
        mid = 0.5 * (low + high)
        span = float(min_span)
        low = mid - 0.5 * span
        high = mid + 0.5 * span

    ymin = low - float(lower_pad_frac) * span
    ymax = high + float(upper_pad_frac) * span
    if upper_bound is not None:
        ymax = min(float(upper_bound), ymax)
        if ymax <= ymin:
            ymax = ymin + span
    return ymin, ymax, low, high, span


def _finalize_ylim_with_brackets(
    *,
    ymin: float,
    ymax: float,
    bracket_tops: Sequence[float],
    span: float,
    upper_bound: Optional[float] = None,
    final_pad_frac: float = 0.06,
) -> Tuple[float, float]:
    """Expand y-limits enough to include all bracket labels."""
    finite_tops = [float(x) for x in bracket_tops if np.isfinite(x)]
    if finite_tops:
        ymax = max(float(ymax), max(finite_tops) + float(final_pad_frac) * float(span))
    if upper_bound is not None:
        # Keep a little slack above 1 for BC plots if brackets require it, but avoid
        # excessive empty space. This allows brackets above boxes whose whiskers are
        # near 1.0 without cropping the annotation.
        ymax = min(max(float(ymax), max(finite_tops) + 0.03 * span if finite_tops else float(ymax)), float(upper_bound))
    if ymax <= ymin:
        ymax = ymin + max(float(span), 0.01)
    return float(ymin), float(ymax)


def _prepost_group_bracket_specs(
    *,
    arrays: Sequence[Sequence[float]],
    groups: Sequence[str],
    pair_centers: Sequence[float],
    p_by_group: Mapping[str, float],
    span: float,
) -> List[Tuple[float, float, float, str, float]]:
    """Return pre/post bracket specs placed above the top whisker of each pair."""
    tops = _boxplot_top_whiskers(arrays)
    gap = max(0.010 * span, 0.0015)
    h = max(0.020 * span, 0.0015)
    specs: List[Tuple[float, float, float, str, float]] = []
    for i, group in enumerate(groups):
        idx0 = 2 * i
        idx1 = 2 * i + 1
        pair_top = np.nanmax([tops[idx0], tops[idx1]]) if idx1 < len(tops) else np.nan
        if not np.isfinite(pair_top):
            continue
        center = float(pair_centers[i])
        y = float(pair_top) + gap
        label = _stars_from_p(p_by_group.get(group, np.nan))
        specs.append((center - 0.20, center + 0.20, y, label, h))
    return specs


def _delta_bracket_specs(
    *,
    arrays: Sequence[Sequence[float]],
    groups: Sequence[str],
    p_vs_sham: Mapping[str, float],
    span: float,
) -> List[Tuple[float, float, float, str, float]]:
    """Return delta-vs-sham brackets stacked above the relevant top whiskers."""
    if SHAM_LABEL not in groups:
        return []
    tops = _boxplot_top_whiskers(arrays)
    sham_x = groups.index(SHAM_LABEL)
    gap = max(0.018 * span, 0.0015)
    h = max(0.030 * span, 0.0015)

    raw: List[Tuple[int, float, float, str, float]] = []
    for group in groups:
        if group == SHAM_LABEL:
            continue
        p = p_vs_sham.get(group, np.nan)
        if not np.isfinite(p):
            continue
        x = groups.index(group)
        involved_top = np.nanmax([tops[sham_x], tops[x]])
        if not np.isfinite(involved_top):
            continue
        label = _stars_from_p(p) or "n.s."
        raw.append((abs(x - sham_x), float(x), float(involved_top + gap), label, h))

    # Shorter comparisons lower, longer comparisons higher, while always staying
    # above the whiskers of the boxes being compared.
    raw.sort(key=lambda t: t[0])
    specs: List[Tuple[float, float, float, str, float]] = []
    previous_top = -np.inf
    for _, x, candidate_y, label, h in raw:
        y = max(float(candidate_y), float(previous_top) + gap)
        specs.append((float(sham_x), float(x), y, label, h))
        previous_top = y + h
    return specs


def _ylim_from_box_whiskers(
    arrays: Sequence[Sequence[float]],
    *,
    include_zero: bool = False,
    upper_bound: Optional[float] = None,
    min_span: float = 0.012,
    extra_top_frac: float = 0.10,
) -> Tuple[float, float]:
    """Backward-compatible compact y-limits based on visible boxplot whiskers."""
    ymin, ymax, _, _, _ = _base_ylim_for_boxplot(
        arrays,
        include_zero=include_zero,
        use_whiskers_only=True,
        min_span=min_span,
        lower_pad_frac=0.08,
        upper_pad_frac=extra_top_frac,
        upper_bound=upper_bound,
    )
    return ymin, ymax


def _with_p_label(group: str, p: float, *, comparison_prefix: str = "p") -> str:
    """Make a compact multi-line tick label with a p-value below the group name."""
    label = str(group)
    if np.isfinite(p):
        label += f"\n{comparison_prefix}={_fmt_p_for_label(p).replace('p=', '')}"
    else:
        label += f"\n{comparison_prefix}=n/a"
    return label


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
    """Plot pre/post BC by lesion group with paired-test p-values under groups."""
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(0)
    fig_w = max(11.2, 2.25 * len(groups) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8.4))

    p_by_group: Dict[str, float] = {}
    arrays, positions, cols = [], [], []
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

    if (not bool(boxplot_only)) and len(df) <= 550:
        for _, r in df.iterrows():
            group = str(r["lesion_hit_type"])
            if group not in groups:
                continue
            center = pair_centers[groups.index(group)]
            x0 = center - 0.20
            x1 = center + 0.20
            ax.plot([x0, x1], [r["bc_pre"], r["bc_post"]], color="0.78", linewidth=0.55, zorder=1)

    ymin, ymax, low, high, span = _base_ylim_for_boxplot(
        arrays,
        plotted_values=all_values,
        include_zero=False,
        use_whiskers_only=bool(boxplot_only),
        min_span=0.012 if bool(boxplot_only) else 0.020,
        lower_pad_frac=0.08,
        upper_pad_frac=0.08,
        upper_bound=None,
    )
    bracket_specs = _prepost_group_bracket_specs(
        arrays=arrays,
        groups=groups,
        pair_centers=pair_centers,
        p_by_group=p_by_group,
        span=span,
    )
    bracket_tops = [y + h for _, _, y, _, h in bracket_specs]
    ymin, ymax = _finalize_ylim_with_brackets(
        ymin=ymin,
        ymax=ymax,
        bracket_tops=bracket_tops,
        span=span,
        upper_bound=1.04,
        final_pad_frac=0.08,
    )
    ax.set_ylim(ymin, ymax)

    for x1, x2, y, label, h in bracket_specs:
        _draw_sig_bracket(ax, x1, x2, y, label, h=h, fontsize=13)

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

    arrays, positions, cols = [], [], []
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

    ymin, ymax, low, high, span = _base_ylim_for_boxplot(
        arrays,
        plotted_values=all_values,
        include_zero=True,
        use_whiskers_only=bool(boxplot_only),
        min_span=0.012 if bool(boxplot_only) else 0.020,
        lower_pad_frac=0.10,
        upper_pad_frac=0.08,
        upper_bound=None,
    )
    bracket_specs = _delta_bracket_specs(
        arrays=arrays,
        groups=groups,
        p_vs_sham=p_vs_sham,
        span=span,
    )
    bracket_tops = [y + h for _, _, y, _, h in bracket_specs]
    ymin, ymax = _finalize_ylim_with_brackets(
        ymin=ymin,
        ymax=ymax,
        bracket_tops=bracket_tops,
        span=span,
        upper_bound=None,
        final_pad_frac=0.08,
    )
    ax.set_ylim(ymin, ymax)

    for x1, x2, y, label, h in bracket_specs:
        _draw_sig_bracket(ax, x1, x2, y, label, h=h, fontsize=13)

    tick_labels = []
    for g in groups:
        if g == SHAM_LABEL:
            tick_labels.append(f"{g}")
        else:
            p = p_vs_sham.get(g, np.nan)
            ptxt = _fmt_p_for_label(p).replace('p=', '') if np.isfinite(p) else 'n/a'
            tick_labels.append(f"{g}\nvs sham p={ptxt}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Post BC − Pre BC", fontsize=18)
    ax.set_title(title, fontsize=20, pad=30, y=1.04)
    ax.tick_params(axis="x", labelsize=15, pad=10)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def compute_stats(df: pd.DataFrame, *, level: str, order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["set_name", "set_label", "bc_method", "bc_method_label"]
    for keys, sub0 in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for lesion, sub in sub0.groupby("lesion_hit_type", sort=False):
            pre = sub["bc_pre"].to_numpy(dtype=float)
            post = sub["bc_post"].to_numpy(dtype=float)
            delta = sub["bc_delta_post_minus_pre"].to_numpy(dtype=float)
            mask = np.isfinite(pre) & np.isfinite(post)
            p_w = np.nan
            if wilcoxon is not None and mask.sum() >= 2:
                try:
                    p_w = float(wilcoxon(pre[mask], post[mask], alternative="two-sided").pvalue)
                except Exception:
                    p_w = np.nan
            rows.append({
                **base,
                "analysis_level": level,
                "test_family": "paired_pre_vs_post_within_lesion_group",
                "lesion_hit_type": lesion,
                "comparison_group_a": "Pre",
                "comparison_group_b": "Post",
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

        # Pairwise lesion-group comparisons on delta vs sham.
        for lesion in _ordered_groups(sub0, order):
            if lesion == SHAM_LABEL:
                continue
            p_vs = _unpaired_delta_p_vs_sham(sub0, lesion, sham_label=SHAM_LABEL)
            a = pd.to_numeric(
                sub0.loc[sub0["lesion_hit_type"].astype(str) == lesion, "bc_delta_post_minus_pre"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            b = pd.to_numeric(
                sub0.loc[sub0["lesion_hit_type"].astype(str) == SHAM_LABEL, "bc_delta_post_minus_pre"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            rows.append({
                **base,
                "analysis_level": level,
                "test_family": "delta_post_minus_pre_vs_sham",
                "lesion_hit_type": f"{lesion}_VS_{SHAM_LABEL}",
                "comparison_group_a": lesion,
                "comparison_group_b": SHAM_LABEL,
                "n_observations": int(len(a) + len(b)),
                "n_animals": int(sub0.loc[sub0["lesion_hit_type"].astype(str).isin([lesion, SHAM_LABEL]), "animal_id"].nunique()),
                "mean_bc_pre": np.nan,
                "mean_bc_post": np.nan,
                "mean_delta_post_minus_pre": np.nan,
                "median_bc_pre": np.nan,
                "median_bc_post": np.nan,
                "median_delta_post_minus_pre": np.nan,
                "paired_pre_post_wilcoxon_p": np.nan,
                "paired_pre_post_wilcoxon_p_formatted": "",
                "delta_vs_sham_mannwhitney_p": p_vs,
                "delta_vs_sham_mannwhitney_p_formatted": _fmt_p(p_vs),
                "stars": _stars_from_p(p_vs),
            })

        # Across-lesion Kruskal-Wallis on delta.
        if kruskal is not None:
            arrays = []
            labels = []
            for lesion in _ordered_groups(sub0, order):
                vals = sub0.loc[sub0["lesion_hit_type"].astype(str) == lesion, "bc_delta_post_minus_pre"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
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
                    "test_family": "delta_post_minus_pre_across_lesion_groups",
                    "lesion_hit_type": "ACROSS_LESION_GROUPS",
                    "comparison_group_a": ";".join(labels),
                    "comparison_group_b": "",
                    "n_observations": int(sum(len(a) for a in arrays)),
                    "n_animals": int(sub0["animal_id"].nunique()),
                    "mean_bc_pre": np.nan,
                    "mean_bc_post": np.nan,
                    "mean_delta_post_minus_pre": np.nan,
                    "median_bc_pre": np.nan,
                    "median_bc_post": np.nan,
                    "median_delta_post_minus_pre": np.nan,
                    "paired_pre_post_wilcoxon_p": np.nan,
                    "paired_pre_post_wilcoxon_p_formatted": "",
                    "delta_kruskal_statistic": float(stat),
                    "delta_kruskal_p": float(p),
                    "delta_kruskal_p_formatted": _fmt_p(float(p)),
                    "delta_kruskal_groups": ";".join(labels),
                    "stars": _stars_from_p(float(p)),
                })
    return pd.DataFrame(rows)




def _short_set_label_for_ml_combined(set_name: str) -> str:
    """Readable labels for the Medial+Lateral high-vs-remaining combined plot."""
    labels = {
        "high_variance_clusters": "Top 30%\nvariance clusters",
        "remaining_non_high_variance_clusters": "Remaining\nnon-top-variance clusters",
    }
    return labels.get(str(set_name), str(set_name).replace("_", "\n"))


def plot_ml_high_vs_remaining_prepost(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    dpi: int,
    boxplot_only: bool = False,
) -> None:
    """Plot only Medial+Lateral lesion birds, with high-variance and remaining sets on one figure.

    The x-axis has two grouped categories:
        1) Top 30% phrase-duration variance clusters
        2) Remaining non-top-variance clusters

    Within each category, Pre and Post BC are shown as paired bird-level boxplots.
    This is intentionally a small add-on plot and does not change any of the main
    batch plots or statistics.
    """
    _safe_mkdir(Path(out_png).parent)

    set_order = ["high_variance_clusters", "remaining_non_high_variance_clusters"]
    available_sets = [s for s in set_order if s in set(df["set_name"].astype(str))]
    if not available_sets:
        print(f"[WARN] No high/remaining sets available for Medial+Lateral combined plot: {out_png}")
        return

    rng = np.random.default_rng(4)
    fig_w = max(9.6, 2.6 * len(available_sets) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8.2))

    base = colors.get(COMBINED_ML_LABEL, "#808080")
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    cols: List[str] = []
    all_values: List[float] = []
    pair_centers: List[float] = []
    p_by_set: Dict[str, float] = {}

    for i, set_name in enumerate(available_sets):
        sub = df[df["set_name"].astype(str) == set_name].copy()
        pre_vals = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        center = i * 1.35
        pair_centers.append(center)
        pre_x = center - 0.22
        post_x = center + 0.22

        arrays.extend([pre_vals, post_vals])
        positions.extend([pre_x, post_x])
        cols.extend([base, base])
        all_values.extend(pre_vals[np.isfinite(pre_vals)].tolist())
        all_values.extend(post_vals[np.isfinite(post_vals)].tolist())
        p_by_set[set_name] = _paired_pre_post_p(sub)

    _boxplot_with_points(
        ax,
        arrays,
        positions,
        cols,
        width=0.30,
        point_size=44,
        rng=rng,
        show_points=not bool(boxplot_only),
    )

    if not bool(boxplot_only):
        for i, set_name in enumerate(available_sets):
            sub = df[df["set_name"].astype(str) == set_name].copy()
            center = pair_centers[i]
            x0 = center - 0.22
            x1 = center + 0.22
            for _, r in sub.iterrows():
                pre = pd.to_numeric(pd.Series([r.get("bc_pre")]), errors="coerce").iloc[0]
                post = pd.to_numeric(pd.Series([r.get("bc_post")]), errors="coerce").iloc[0]
                if not (np.isfinite(pre) and np.isfinite(post)):
                    continue
                j0 = x0 + float(rng.uniform(-0.035, 0.035))
                j1 = x1 + float(rng.uniform(-0.035, 0.035))
                ax.plot([j0, j1], [pre, post], color=base, alpha=0.35, linewidth=1.0, zorder=1)
                ax.scatter([j0], [pre], s=52, facecolors="none", edgecolors=base, linewidths=1.5, zorder=4)
                ax.scatter([j1], [post], s=52, facecolors=base, edgecolors=base, linewidths=1.0, zorder=4)

    ymin, ymax, low, high, span = _base_ylim_for_boxplot(
        arrays,
        plotted_values=all_values,
        include_zero=False,
        use_whiskers_only=bool(boxplot_only),
        min_span=0.012 if bool(boxplot_only) else 0.020,
        lower_pad_frac=0.08,
        upper_pad_frac=0.08,
        upper_bound=None,
    )

    bracket_specs = _prepost_group_bracket_specs(
        arrays=arrays,
        groups=available_sets,
        pair_centers=pair_centers,
        p_by_group=p_by_set,
        span=span,
    )
    bracket_tops = [y + h for _, _, y, _, h in bracket_specs]
    ymin, ymax = _finalize_ylim_with_brackets(
        ymin=ymin,
        ymax=ymax,
        bracket_tops=bracket_tops,
        span=span,
        upper_bound=1.04,
        final_pad_frac=0.08,
    )
    ax.set_ylim(ymin, ymax)

    for x1, x2, y, label, h in bracket_specs:
        _draw_sig_bracket(ax, x1, x2, y, label, h=h, fontsize=13)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Pre", "Post"] * len(available_sets), rotation=0, ha="center")

    for set_name, center in zip(available_sets, pair_centers):
        ax.text(
            center,
            -0.115,
            _short_set_label_for_ml_combined(set_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14,
            linespacing=0.95,
            clip_on=False,
        )
        p = p_by_set.get(set_name, np.nan)
        p_text = f"pre/post p={_fmt_p_for_label(p).replace('p=', '')}" if np.isfinite(p) else "pre/post p=n/a"
        ax.text(
            center,
            -0.255,
            p_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
            clip_on=False,
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


def make_medial_lateral_combined_prepost_plots(
    bird_level: pd.DataFrame,
    out_dir: Path,
    *,
    colors: Dict[str, str],
    dpi: int,
) -> List[Path]:
    """Create the requested Medial+Lateral-only combined high-vs-remaining figure."""
    paths: List[Path] = []
    ml = bird_level[bird_level["lesion_hit_type"].astype(str) == COMBINED_ML_LABEL].copy()
    if ml.empty:
        print("[WARN] No Medial and Lateral lesion bird-level rows available for combined plot.")
        return paths

    ml = ml[ml["set_name"].astype(str).isin(["high_variance_clusters", "remaining_non_high_variance_clusters"])].copy()
    if ml.empty:
        print("[WARN] No high-variance or remaining rows available for Medial+Lateral combined plot.")
        return paths

    combined_dir = Path(out_dir) / "bird_level" / "medial_lateral_only_high_vs_remaining_prepost"
    _safe_mkdir(combined_dir)

    stats_rows: List[Dict[str, Any]] = []
    for (method, method_label), sub_method in ml.groupby(["bc_method", "bc_method_label"], sort=False):
        if sub_method.empty:
            continue
        n_birds = int(sub_method["animal_id"].nunique())
        title = (
            f"Bird level: Medial and Lateral lesion\n"
            f"Top 30% variance vs remaining clusters\n"
            f"{method_label} | N={n_birds} birds"
        )
        token = f"medial_lateral_only_bird_level_{_clean_filename(method)}_high_vs_remaining_prepost"

        p_full = combined_dir / f"{token}.png"
        plot_ml_high_vs_remaining_prepost(
            sub_method,
            p_full,
            title=title,
            colors=colors,
            dpi=dpi,
            boxplot_only=False,
        )
        paths.append(p_full)

        p_box = combined_dir / f"{token}_boxplots_only.png"
        plot_ml_high_vs_remaining_prepost(
            sub_method,
            p_box,
            title=title,
            colors=colors,
            dpi=dpi,
            boxplot_only=True,
        )
        paths.append(p_box)

        for set_name, sub_set in sub_method.groupby("set_name", sort=False):
            pre = pd.to_numeric(sub_set["bc_pre"], errors="coerce").to_numpy(dtype=float)
            post = pd.to_numeric(sub_set["bc_post"], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(sub_set["bc_delta_post_minus_pre"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(pre) & np.isfinite(post)
            p_w = _paired_pre_post_p(sub_set)
            stats_rows.append({
                "bc_method": method,
                "bc_method_label": method_label,
                "lesion_hit_type": COMBINED_ML_LABEL,
                "set_name": set_name,
                "set_label": SET_LABELS.get(set_name, set_name),
                "analysis_level": "bird",
                "n_animals": int(sub_set["animal_id"].nunique()),
                "n_observations_with_pre_post": int(mask.sum()),
                "mean_bc_pre": float(np.nanmean(pre)) if np.isfinite(pre).any() else np.nan,
                "mean_bc_post": float(np.nanmean(post)) if np.isfinite(post).any() else np.nan,
                "mean_delta_post_minus_pre": float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan,
                "median_bc_pre": float(np.nanmedian(pre)) if np.isfinite(pre).any() else np.nan,
                "median_bc_post": float(np.nanmedian(post)) if np.isfinite(post).any() else np.nan,
                "median_delta_post_minus_pre": float(np.nanmedian(delta)) if np.isfinite(delta).any() else np.nan,
                "paired_pre_post_wilcoxon_p": p_w,
                "paired_pre_post_wilcoxon_p_formatted": _fmt_p(p_w),
                "stars": _stars_from_p(p_w),
            })

    if stats_rows:
        stats_csv = combined_dir / "medial_lateral_only_high_vs_remaining_prepost_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
        print(f"[SAVE] {stats_csv}")

    return paths

def make_all_plots(
    cluster_long: pd.DataFrame,
    bird_level: pd.DataFrame,
    out_dir: Path,
    *,
    colors: Dict[str, str],
    lesion_order: Sequence[str],
    dpi: int,
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
            n_birds = sub["animal_id"].nunique()
            title_base = f"{level_label.capitalize()} level: {set_label}\n{method_label} | N={n_obs} {level_label}s, {n_birds} birds"
            token = f"{level_label}_level_{_clean_filename(method)}_{SET_FILENAME_TOKEN.get(set_name, _clean_filename(set_name))}"
            p1 = level_dir / f"{token}_pre_vs_post_by_lesion.png"
            plot_pre_post_by_lesion(
                sub,
                p1,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
            )
            paths.append(p1)

            p1_box = level_dir / f"{token}_pre_vs_post_by_lesion_boxplots_only.png"
            plot_pre_post_by_lesion(
                sub,
                p1_box,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
                boxplot_only=True,
            )
            paths.append(p1_box)

            p2 = level_dir / f"{token}_delta_post_minus_pre_by_lesion.png"
            plot_delta_by_lesion(
                sub,
                p2,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
            )
            paths.append(p2)

            p2_box = level_dir / f"{token}_delta_post_minus_pre_by_lesion_boxplots_only.png"
            plot_delta_by_lesion(
                sub,
                p2_box,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
                boxplot_only=True,
            )
            paths.append(p2_box)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch lesion-hit-type BC summary plots, including remaining non-top-variance clusters."
    )
    p.add_argument("--bc-root", required=True, help="Root folder containing per-bird *_cluster_bc_summary.csv files.")
    p.add_argument("--metadata-excel", required=True, help="Excel file with animal IDs and lesion hit types.")
    p.add_argument("--out-dir", default=None, help="Output folder. Defaults to <bc-root>/_batch_lesion_group_summaries.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--metadata-animal-col", default="Animal ID")
    p.add_argument("--metadata-hit-type-col", default="Lesion hit type")
    p.add_argument("--bc-method", choices=list(METHOD_DEFS.keys()) + ["all"], default="selected_bins")
    p.add_argument("--bird-aggregate", choices=["median", "mean"], default="median")
    p.add_argument("--min-balanced-duration-s", type=float, default=None, help="Optional extra duration filter. Usually not needed because each cluster summary already has passes_min_balanced_duration.")
    p.add_argument("--color-json", default=None, help="Optional lesion-group color JSON. Supports either {group: color} or {'colors': {group: color}}.")
    p.add_argument("--dpi", type=int, default=300)
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
        raise ValueError(f"None of the requested BC methods are available. Requested: {requested_methods}. Columns: {list(cluster_df.columns)}")

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
    )

    # Extra small add-on requested: one Medial+Lateral-only bird-level figure
    # that places the Top 30% variance Pre/Post boxes and the Remaining
    # Pre/Post boxes on the same figure. This does not alter the original plots.
    paths.extend(
        make_medial_lateral_combined_prepost_plots(
            bird_level,
            out_dir,
            colors=colors,
            dpi=int(args.dpi),
        )
    )

    for p in paths:
        print(f"[SAVE] {p}")

    # A compact manifest to remind you what was included.
    manifest = pd.DataFrame({
        "bc_root": [str(bc_root)],
        "out_dir": [str(out_dir)],
        "n_cluster_summary_files": [cluster_df["source_csv"].nunique()],
        "n_animals": [cluster_long["animal_id"].nunique()],
        "n_cluster_level_rows": [len(cluster_long)],
        "n_bird_level_rows": [len(bird_level)],
        "bc_methods": [";".join(methods)],
        "sets": [";".join(SET_LABELS.keys())],
    })
    manifest_csv = out_dir / "bc_batch_lesion_group_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    print(f"[SAVE] {manifest_csv}")


if __name__ == "__main__":
    main()
