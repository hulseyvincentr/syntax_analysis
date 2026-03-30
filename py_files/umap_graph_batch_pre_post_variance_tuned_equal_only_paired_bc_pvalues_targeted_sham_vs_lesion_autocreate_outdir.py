#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_graph_batch_pre_post_variance_tuned_equal_only.py

%run /Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/umap_graph_batch_pre_post_variance.py --summary-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/batch_umap_pre_post_variance/batch_cluster_variance_bc_equal_only_summary.csv" --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/umap_graph_batch_pre_post_variance"

Reduced, interpretable graphing script for the equal-groups-only batch outputs.

This version is aligned to:
    umap_pre_post_early_late_cluster_variance_bc_tuned_equal_only.py
and the batch runner:
    umap_batch_pre_post_variance_tuned_equal_only.py

Main questions
--------------
1) Overlap change (equal-groups only)
   - bc_pre_vs_post_equal_groups
   - bc_pre_early_vs_late_equal_groups
   - bc_post_early_vs_late_equal_groups

2) Mean / representational shift (equal-groups only)
   - centroid_shift_raw_equal_groups
   - centroid_shift_umap_equal_groups

3) Spread / variance change (equal-groups only)
   - post_over_pre_rms_radius_equal_groups
   - post_over_pre_rms_radius_umap_equal_groups

Also keeps direct paired pre-vs-post comparisons for the equal-group outputs where both
pre and post columns exist in the batch summary:
- Bhattacharyya coefficient: pre early-vs-late vs post early-vs-late
- RMS radius in latent space
- RMS radius in UMAP space

Note on centroid metrics:
- The batch summary stores centroid shift as a post-vs-pre change magnitude
  (e.g., centroid_shift_raw_equal_groups, centroid_shift_umap_equal_groups),
  not separate pre and post centroid values.
- So centroid metrics are still graphed as change metrics by lesion hit type,
  but there is no true pre-vs-post paired centroid boxplot unless the upstream
  single-bird summary is extended to save separate pre and post centroids.

Outputs
-------
out_dir/
  merged_cluster_level_with_hit_type.csv
  bird_level_mean_by_animal.csv

  cluster_level/
    <metric>_cluster_level_by_hit_type.png

  bird_level_mean/
    <metric>_bird_mean_by_hit_type.png

  paired_cluster_level/
    paired_<metric_name>_cluster_level_pre_vs_post_grouped_by_hit_type.png

  paired_bird_level_mean/
    paired_<metric_name>_bird_mean_pre_vs_post_grouped_by_hit_type.png

  stats/
    cluster_level_stats.csv
    bird_level_mean_stats.csv
    paired_cluster_level_pre_post_stats.csv
    paired_bird_level_mean_pre_post_stats.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_METRICS = [
    "bc_pre_vs_post_equal_groups",
    "bc_pre_early_vs_late_equal_groups",
    "bc_post_early_vs_late_equal_groups",
    "centroid_shift_raw_equal_groups",
    "centroid_shift_umap_equal_groups",
    "post_over_pre_rms_radius_equal_groups",
    "post_over_pre_rms_radius_umap_equal_groups",
]

PAIRED_METRIC_SPECS = [
    {
        "name": "bc_early_late_equal_groups",
        "pre_col": "bc_pre_early_vs_late_equal_groups",
        "post_col": "bc_post_early_vs_late_equal_groups",
        "ylabel": "Bhattacharyya coefficient (early vs late, equal groups)",
    },
    {
        "name": "rms_radius_TweetyBERT_latent_equal_groups",
        "pre_col": "pre_rms_radius_raw_equal_groups",
        "post_col": "post_rms_radius_raw_equal_groups",
        "ylabel": "RMS radius (TweetyBERT latent space, equal groups)",
    },
    {
        "name": "rms_radius_umap_equal_groups",
        "pre_col": "pre_rms_radius_umap_equal_groups",
        "post_col": "post_rms_radius_umap_equal_groups",
        "ylabel": "RMS radius (UMAP space, equal groups)",
    },
]

DEFAULT_HIT_TYPE_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Medial/Lateral visible + large lesion",
]


COLUMN_DISPLAY_ALIASES = {
    "centroid_shift_raw_equal_groups": "centroid_shift_TweetyBERT_latent_equal_groups",
    "pre_rms_radius_raw_equal_groups": "pre_rms_radius_TweetyBERT_latent_equal_groups",
    "post_rms_radius_raw_equal_groups": "post_rms_radius_TweetyBERT_latent_equal_groups",
    "post_over_pre_rms_radius_equal_groups": "post_over_pre_rms_radius_TweetyBERT_latent_equal_groups",
    "pre_trace_cov_raw_equal_groups": "pre_trace_cov_TweetyBERT_latent_equal_groups",
    "post_trace_cov_raw_equal_groups": "post_trace_cov_TweetyBERT_latent_equal_groups",
    "post_over_pre_trace_cov_equal_groups": "post_over_pre_trace_cov_TweetyBERT_latent_equal_groups",
}

COLUMN_YLABEL_ALIASES = {
    "bc_pre_vs_post_equal_groups": "Bhattacharyya coefficient (pre vs post, equal groups)",
    "bc_pre_early_vs_late_equal_groups": "Bhattacharyya coefficient (pre early vs late, equal groups)",
    "bc_post_early_vs_late_equal_groups": "Bhattacharyya coefficient (post early vs late, equal groups)",
    "centroid_shift_raw_equal_groups": "Centroid shift (TweetyBERT latent space, equal groups)",
    "centroid_shift_umap_equal_groups": "Centroid shift (UMAP space, equal groups)",
    "post_over_pre_rms_radius_equal_groups": "Post / pre RMS radius (TweetyBERT latent space, equal groups)",
    "post_over_pre_rms_radius_umap_equal_groups": "Post / pre RMS radius (UMAP space, equal groups)",
    "pre_rms_radius_raw_equal_groups": "Pre RMS radius (TweetyBERT latent space, equal groups)",
    "post_rms_radius_raw_equal_groups": "Post RMS radius (TweetyBERT latent space, equal groups)",
    "pre_rms_radius_umap_equal_groups": "Pre RMS radius (UMAP space, equal groups)",
    "post_rms_radius_umap_equal_groups": "Post RMS radius (UMAP space, equal groups)",
}


def _display_metric_name(metric: str) -> str:
    return COLUMN_DISPLAY_ALIASES.get(metric, metric)


def _display_ylabel(metric: str) -> str:
    return COLUMN_YLABEL_ALIASES.get(metric, _display_metric_name(metric))


# ----------------------------
# Utilities
# ----------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_filename(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in low:
            return low[key]
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _collapse_hit_type(raw_hit_type: str) -> str:
    """
    Collapse lesion hit types into:
      - sham saline injection
      - Area X visible (single hit)
      - Medial/Lateral visible + large lesion
    """
    norm = _normalize_text(raw_hit_type)

    if "sham" in norm:
        return "sham saline injection"

    if "singlehit" in norm:
        return "Area X visible (single hit)"

    if "medial" in norm and "lateral" in norm:
        return "Medial/Lateral visible + large lesion"

    if "large" in norm and "lesion" in norm:
        return "Medial/Lateral visible + large lesion"

    if "notvisible" in norm:
        return "Medial/Lateral visible + large lesion"

    return str(raw_hit_type)


def _infer_hit_type_order(values: Sequence[str]) -> List[str]:
    vals = [str(v) for v in values if pd.notna(v)]
    present = list(dict.fromkeys(vals))
    ordered = [v for v in DEFAULT_HIT_TYPE_ORDER if v in present]
    remaining = [v for v in present if v not in ordered]
    return ordered + remaining


def _target_contrast_pairs(groups: Sequence[str]) -> List[Tuple[str, str]]:
    wanted = [
        ("sham saline injection", "Area X visible (single hit)"),
        ("sham saline injection", "Medial/Lateral visible + large lesion"),
    ]
    present = set(groups)
    return [(a, b) for (a, b) in wanted if a in present and b in present]


def _jitter_positions(n: int, center: float, width: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return center + rng.uniform(-width, width, size=n)


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


def _format_p_label(p: float, sig_label: str) -> str:
    if not np.isfinite(p):
        return sig_label
    return f"{sig_label}\np={p:.3g}"


def _holm_bonferroni(pvals: Sequence[float]) -> np.ndarray:
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
        adj_sorted[i] = min(1.0, val)

    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def _add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, color="black", clip_on=False)
    text_pad = max(h * 0.4, 0.01)
    ax.text((x1 + x2) / 2.0, y + h + text_pad, text,
            ha="center", va="bottom", fontsize=11, clip_on=False)


def _mannwhitney_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")

    if not _HAVE_SCIPY:
        return float("nan"), float("nan")

    if np.all(a == a.flat[0]) and np.all(b == b.flat[0]) and a.flat[0] == b.flat[0]:
        return 0.0, 1.0

    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _wilcoxon_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), "no paired data"
    if a.size != b.size:
        return float("nan"), float("nan"), "paired lengths mismatch"
    if a.size < 2:
        return float("nan"), float("nan"), "n<2"

    if not _HAVE_SCIPY:
        return float("nan"), float("nan"), "scipy unavailable"

    diffs = a - b
    if np.allclose(diffs, 0):
        return 0.0, 1.0, "all paired differences are zero"

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), repr(e)


# ----------------------------
# Stats helpers
# ----------------------------

def _pairwise_stats_for_metric(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    groups: Sequence[str],
    level_label: str,
    contrast_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()

    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[np.isfinite(work[value_col])].copy()

    if len(work) == 0:
        return pd.DataFrame()

    group_arrays = []
    present_groups = []
    for g in groups:
        vals = work.loc[work[group_col] == g, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            present_groups.append(g)
            group_arrays.append(vals)

    if len(present_groups) == 0:
        return pd.DataFrame()

    omnibus_stat = float("nan")
    omnibus_p = float("nan")
    if _HAVE_SCIPY and len(group_arrays) >= 2:
        try:
            if len(group_arrays) >= 3:
                omnibus_stat, omnibus_p = map(float, stats.kruskal(*group_arrays))
            else:
                omnibus_stat, omnibus_p = float("nan"), float("nan")
        except Exception:
            omnibus_stat, omnibus_p = float("nan"), float("nan")

    pair_rows = []
    raw_ps = []

    if contrast_pairs is None:
        pairs = []
        for i in range(len(present_groups)):
            for j in range(i + 1, len(present_groups)):
                pairs.append((present_groups[i], present_groups[j]))
    else:
        present_set = set(present_groups)
        pairs = [(g1, g2) for (g1, g2) in contrast_pairs if g1 in present_set and g2 in present_set]

    for g1, g2 in pairs:
        a = work.loc[work[group_col] == g1, value_col].to_numpy(dtype=float)
        b = work.loc[work[group_col] == g2, value_col].to_numpy(dtype=float)

        stat, p_raw = _mannwhitney_safe(a, b)
        pair_rows.append({
            "level": level_label,
            "metric": value_col,
            "omnibus_test": "kruskal" if len(group_arrays) >= 3 else "",
            "omnibus_stat": omnibus_stat,
            "omnibus_p": omnibus_p,
            "pairwise_test": "mannwhitneyu",
            "group_1": g1,
            "group_2": g2,
            "n_1": int(len(a)),
            "n_2": int(len(b)),
            "pairwise_stat": stat,
            "p_raw": p_raw,
        })
        raw_ps.append(p_raw)

    if len(pair_rows) == 0:
        return pd.DataFrame()

    p_holm = _holm_bonferroni(raw_ps)
    for row, p_adj in zip(pair_rows, p_holm):
        row["p_holm"] = float(p_adj)
        row["sig_label"] = _p_to_stars(float(p_adj))

    return pd.DataFrame(pair_rows)


def _paired_stats_for_metric(
    df: pd.DataFrame,
    *,
    group_col: str,
    pre_col: str,
    post_col: str,
    groups: Sequence[str],
    metric_name: str,
    level_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if pre_col not in df.columns or post_col not in df.columns:
        return pd.DataFrame()

    raw_ps = []
    raw_idx = []

    for g in groups:
        sub = df.loc[df[group_col] == g, [pre_col, post_col]].copy()
        sub[pre_col] = pd.to_numeric(sub[pre_col], errors="coerce")
        sub[post_col] = pd.to_numeric(sub[post_col], errors="coerce")
        sub = sub[np.isfinite(sub[pre_col]) & np.isfinite(sub[post_col])].copy()

        a = sub[pre_col].to_numpy(dtype=float)
        b = sub[post_col].to_numpy(dtype=float)

        stat, p_raw, note = _wilcoxon_safe(a, b)

        row = {
            "level": level_label,
            "metric": metric_name,
            "group": g,
            "test": "wilcoxon",
            "n_units": int(len(sub)),
            "statistic": stat,
            "p_raw": p_raw,
            "note": note,
        }
        rows.append(row)

        if np.isfinite(p_raw):
            raw_ps.append(float(p_raw))
            raw_idx.append(len(rows) - 1)

    if len(rows) == 0:
        return pd.DataFrame()

    p_adj = np.full(len(rows), np.nan, dtype=float)
    if len(raw_ps) > 0:
        adj_vals = _holm_bonferroni(raw_ps)
        for idx, val in zip(raw_idx, adj_vals):
            p_adj[idx] = float(val)

    for i, row in enumerate(rows):
        row["p_holm"] = p_adj[i]
        row["sig_label"] = _p_to_stars(p_adj[i]) if np.isfinite(p_adj[i]) else "n/a"

    return pd.DataFrame(rows)


def _paired_between_group_stats(
    df: pd.DataFrame,
    *,
    group_col: str,
    pre_col: str,
    post_col: str,
    groups: Sequence[str],
    metric_name: str,
    level_label: str,
    contrast_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if pre_col not in df.columns or post_col not in df.columns:
        return pd.DataFrame()

    if contrast_pairs is None:
        contrast_pairs = _target_contrast_pairs(groups)

    raw_ps: List[float] = []
    raw_idx: List[int] = []

    for time_label, col in [("pre", pre_col), ("post", post_col)]:
        work = df[[group_col, col]].copy()
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work[np.isfinite(work[col])].copy()
        if len(work) == 0:
            continue

        present = set(work[group_col].astype(str).tolist())
        for g1, g2 in contrast_pairs:
            if g1 not in present or g2 not in present:
                continue
            a = work.loc[work[group_col] == g1, col].to_numpy(dtype=float)
            b = work.loc[work[group_col] == g2, col].to_numpy(dtype=float)
            stat, p_raw = _mannwhitney_safe(a, b)
            row = {
                "level": level_label,
                "metric": metric_name,
                "timepoint": time_label,
                "group_1": g1,
                "group_2": g2,
                "n_1": int(len(a)),
                "n_2": int(len(b)),
                "pairwise_test": "mannwhitneyu",
                "pairwise_stat": stat,
                "p_raw": p_raw,
            }
            rows.append(row)
            if np.isfinite(p_raw):
                raw_ps.append(float(p_raw))
                raw_idx.append(len(rows) - 1)

    if len(rows) == 0:
        return pd.DataFrame()

    p_adj = np.full(len(rows), np.nan, dtype=float)
    if len(raw_ps) > 0:
        adj_vals = _holm_bonferroni(raw_ps)
        for idx, val in zip(raw_idx, adj_vals):
            p_adj[idx] = float(val)

    for i, row in enumerate(rows):
        row["p_holm"] = p_adj[i]
        row["sig_label"] = _p_to_stars(p_adj[i]) if np.isfinite(p_adj[i]) else "n/a"

    return pd.DataFrame(rows)


# ----------------------------
# Plot helpers
# ----------------------------

def _box_strip_plot(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    groups: List[str],
    title: str,
    ylabel: str,
    out_png: Path,
    figsize=(10.5, 5.2),
    rng_seed: int = 0,
    stats_df: Optional[pd.DataFrame] = None,
    annotate_only_significant: bool = False,
) -> None:
    rng = np.random.default_rng(rng_seed)

    plot_groups = []
    plot_values = []
    plot_labels = []
    for g in groups:
        vals = pd.to_numeric(df.loc[df[group_col] == g, value_col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            plot_groups.append(g)
            plot_values.append(vals.to_numpy())
            plot_labels.append(f"{g}\n(n={len(vals)})")

    if len(plot_values) == 0:
        return

    fig, ax = plt.subplots(figsize=figsize)

    try:
        ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=False)
    except TypeError:
        ax.boxplot(plot_values, labels=plot_labels, showfliers=False)

    for i, vals in enumerate(plot_values, start=1):
        x = _jitter_positions(len(vals), i, 0.10, rng)
        ax.scatter(x, vals, s=20, alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Lesion hit type")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    all_vals = np.concatenate(plot_values) if len(plot_values) > 0 else np.array([0.0, 1.0])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))

    y_span = max(1e-9, ymax - ymin)
    y_lower = ymin - max(0.05 * y_span, 0.05)
    y_upper = ymax + max(0.08 * y_span, 0.08)

    pairs_to_draw = []
    if stats_df is not None and len(stats_df) > 0:
        pos_map = {g: i + 1 for i, g in enumerate(plot_groups)}

        required_cols = {"group_1", "group_2", "p_holm", "sig_label"}
        if required_cols.issubset(stats_df.columns):
            for _, row in stats_df.iterrows():
                g1 = str(row["group_1"])
                g2 = str(row["group_2"])

                if g1 not in pos_map or g2 not in pos_map:
                    continue

                p_adj = pd.to_numeric(pd.Series([row["p_holm"]]), errors="coerce").iloc[0]
                sig_label = str(row["sig_label"])
                bracket_label = _format_p_label(float(p_adj), sig_label)

                if annotate_only_significant:
                    if (not np.isfinite(p_adj)) or (p_adj >= 0.05) or (sig_label in ("n.s.", "n/a")):
                        continue

                x1 = float(pos_map[g1])
                x2 = float(pos_map[g2])
                if x1 > x2:
                    x1, x2 = x2, x1

                pairs_to_draw.append((x1, x2, bracket_label))

    if len(pairs_to_draw) > 0:
        pairs_to_draw = sorted(pairs_to_draw, key=lambda t: (t[1] - t[0], t[0]))

        base = ymax + max(0.07 * y_span, 0.07)
        h = max(0.03 * y_span, 0.03)
        step = max(0.16 * y_span, 0.16)

        for level, (x1, x2, label) in enumerate(pairs_to_draw):
            y = base + level * step
            _add_sig_bracket(ax, x1, x2, y, h, label)

        top_needed = base + (len(pairs_to_draw) - 1) * step + h + max(0.18 * y_span, 0.18)
        y_upper = max(y_upper, top_needed)

    ax.set_ylim(y_lower, y_upper)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paired_grouped_pre_post_boxplot(
    df: pd.DataFrame,
    *,
    group_col: str,
    pre_col: str,
    post_col: str,
    groups: List[str],
    ylabel: str,
    title: str,
    out_png: Path,
    stats_df: Optional[pd.DataFrame] = None,
    between_stats_df: Optional[pd.DataFrame] = None,
    rng_seed: int = 0,
) -> None:
    rng = np.random.default_rng(rng_seed)

    plot_groups = []
    plot_subs = []
    for g in groups:
        sub = df.loc[df[group_col] == g, [pre_col, post_col]].copy()
        sub[pre_col] = pd.to_numeric(sub[pre_col], errors="coerce")
        sub[post_col] = pd.to_numeric(sub[post_col], errors="coerce")
        sub = sub[np.isfinite(sub[pre_col]) & np.isfinite(sub[post_col])].copy()
        if len(sub) > 0:
            plot_groups.append(g)
            plot_subs.append(sub)

    if len(plot_subs) == 0:
        return

    fig, ax = plt.subplots(figsize=(4.2 * len(plot_groups), 5.8))

    positions_pre = []
    positions_post = []
    xticks = []
    xticklabels = []

    all_vals = []

    current = 1.0
    gap_within = 0.8
    gap_between = 1.6

    for sub, g in zip(plot_subs, plot_groups):
        y_pre = sub[pre_col].to_numpy(dtype=float)
        y_post = sub[post_col].to_numpy(dtype=float)

        p_pre = current
        p_post = current + gap_within

        positions_pre.append(p_pre)
        positions_post.append(p_post)

        ax.boxplot([y_pre], positions=[p_pre], widths=0.55, showfliers=False)
        ax.boxplot([y_post], positions=[p_post], widths=0.55, showfliers=False)

        for pre_v, post_v in zip(y_pre, y_post):
            ax.plot([p_pre, p_post], [pre_v, post_v], alpha=0.7)
            ax.scatter([p_pre, p_post], [pre_v, post_v], s=22)

        x_pre = _jitter_positions(len(y_pre), p_pre, 0.08, rng)
        x_post = _jitter_positions(len(y_post), p_post, 0.08, rng)
        ax.scatter(x_pre, y_pre, s=20, alpha=0.7)
        ax.scatter(x_post, y_post, s=20, alpha=0.7)

        xticks.extend([p_pre, p_post])
        xticklabels.extend([f"Pre\n{g}", "Post"])

        all_vals.extend(list(y_pre))
        all_vals.extend(list(y_post))

        current = p_post + gap_between

    all_vals = np.asarray(all_vals, dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    ymin = float(np.min(all_vals)) if all_vals.size else 0.0
    ymax = float(np.max(all_vals)) if all_vals.size else 1.0
    y_span = max(1e-9, ymax - ymin)
    y_lower = ymin - max(0.05 * y_span, 0.05)
    y_upper = ymax + max(0.16 * y_span, 0.12)

    stats_map = {}
    if stats_df is not None and len(stats_df) > 0:
        stats_map = {str(r["group"]): r for _, r in stats_df.iterrows()}

    within_base = ymax + max(0.06 * y_span, 0.05)
    h = max(0.03 * y_span, 0.03)

    for g, p_pre, p_post in zip(plot_groups, positions_pre, positions_post):
        row = stats_map.get(g, None)
        label = "n/a"
        if row is not None:
            p_adj = row.get("p_holm", np.nan)
            sig = row.get("sig_label", "n/a")
            if np.isfinite(p_adj):
                label = _format_p_label(float(p_adj), str(sig))
            else:
                label = str(sig)
        _add_sig_bracket(ax, p_pre, p_post, within_base, h, label)

    between_pairs_to_draw = []
    if between_stats_df is not None and len(between_stats_df) > 0:
        pre_map = {g: p for g, p in zip(plot_groups, positions_pre)}
        post_map = {g: p for g, p in zip(plot_groups, positions_post)}
        required_cols = {"timepoint", "group_1", "group_2", "p_holm", "sig_label"}
        if required_cols.issubset(between_stats_df.columns):
            for _, row in between_stats_df.iterrows():
                t = str(row["timepoint"]).lower()
                g1 = str(row["group_1"])
                g2 = str(row["group_2"])
                pos_map = pre_map if t == "pre" else post_map if t == "post" else None
                if pos_map is None or g1 not in pos_map or g2 not in pos_map:
                    continue
                p_adj = pd.to_numeric(pd.Series([row["p_holm"]]), errors="coerce").iloc[0]
                sig_label = str(row["sig_label"])
                bracket_label = _format_p_label(float(p_adj), sig_label) if np.isfinite(p_adj) else sig_label
                x1 = float(pos_map[g1])
                x2 = float(pos_map[g2])
                if x1 > x2:
                    x1, x2 = x2, x1
                between_pairs_to_draw.append((t, x1, x2, bracket_label))

    if len(between_pairs_to_draw) > 0:
        between_pairs_to_draw = sorted(between_pairs_to_draw, key=lambda t: (t[0], t[2] - t[1], t[1]))
        base = within_base + h + max(0.12 * y_span, 0.12)
        step = max(0.16 * y_span, 0.16)
        for level, (_, x1, x2, label) in enumerate(between_pairs_to_draw):
            y = base + level * step
            _add_sig_bracket(ax, x1, x2, y, h, label)
        top_needed = base + (len(between_pairs_to_draw) - 1) * step + h + max(0.18 * y_span, 0.18)
    else:
        top_needed = within_base + h + max(0.18 * y_span, 0.18)

    y_upper = max(y_upper, top_needed)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(y_lower, y_upper)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Load + merge
# ----------------------------

def load_batch_summary(summary_csv: Path) -> pd.DataFrame:
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            "Batch summary CSV not found: "
            f"{summary_csv}\n"
            "The graphing script can create the output folder automatically, but it still needs an existing batch summary CSV as input. "
            "Run the equal-only batch script first, or point --summary-csv to the actual summary file that was created."
        )
    df = pd.read_csv(summary_csv)
    if "animal_id" not in df.columns:
        raise KeyError("Summary CSV must contain an 'animal_id' column.")
    return df


def load_hit_type_metadata(
    metadata_xlsx: Path,
    *,
    sheet_name: str = "animal_hit_type_summary",
    animal_id_col: Optional[str] = None,
    hit_type_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)

    if animal_id_col is None:
        animal_id_col = _pick_column(df, ["Animal ID", "animal_id", "animal id", "Bird ID", "bird"])
    if hit_type_col is None:
        hit_type_col = _pick_column(df, ["Lesion hit type", "hit_type", "lesion hit type"])

    if animal_id_col is None:
        raise KeyError(f"Could not find animal ID column in sheet {sheet_name!r}.")
    if hit_type_col is None:
        raise KeyError(f"Could not find lesion hit type column in sheet {sheet_name!r}.")

    out = df[[animal_id_col, hit_type_col]].copy()
    out.columns = ["animal_id", "lesion_hit_type_raw"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["lesion_hit_type_raw"] = out["lesion_hit_type_raw"].astype(str).str.strip()
    out["lesion_hit_type"] = out["lesion_hit_type_raw"].map(_collapse_hit_type)
    out = out.drop_duplicates(subset=["animal_id"])
    return out


def merge_summary_with_hit_types(summary_df: pd.DataFrame, hit_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out = out.merge(hit_df, on="animal_id", how="left")
    out["lesion_hit_type"] = out["lesion_hit_type"].fillna("unknown")
    out["lesion_hit_type_raw"] = out["lesion_hit_type_raw"].fillna("unknown")
    return out


# ----------------------------
# Bird-level summaries
# ----------------------------

def make_bird_level_summary(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    keep_cols = ["animal_id", "lesion_hit_type"]
    use_cols = keep_cols + [m for m in metrics if m in df.columns]

    sub = df[use_cols].copy()
    for m in metrics:
        if m in sub.columns:
            sub[m] = pd.to_numeric(sub[m], errors="coerce")

    grouped = sub.groupby(["animal_id", "lesion_hit_type"], as_index=False)
    return grouped.mean(numeric_only=True)


# ----------------------------
# Main plotting routine
# ----------------------------

def graph_metrics_by_hit_type(
    summary_csv: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    *,
    metadata_sheet: str = "animal_hit_type_summary",
    metrics: Optional[Sequence[str]] = None,
    cluster_filter: Optional[str] = None,
    drop_unknown: bool = False,
) -> Dict[str, Path]:
    summary_csv = Path(summary_csv)
    metadata_xlsx = Path(metadata_xlsx)
    out_dir = Path(out_dir)

    _safe_mkdir(out_dir)
    cluster_dir = out_dir / "cluster_level"
    bird_mean_dir = out_dir / "bird_level_mean"
    paired_cluster_dir = out_dir / "paired_cluster_level"
    paired_bird_mean_dir = out_dir / "paired_bird_level_mean"
    stats_dir = out_dir / "stats"
    for p in [cluster_dir, bird_mean_dir, paired_cluster_dir, paired_bird_mean_dir, stats_dir]:
        _safe_mkdir(p)

    metrics = list(metrics) if metrics is not None else list(DEFAULT_METRICS)

    df = load_batch_summary(summary_csv)
    hit_df = load_hit_type_metadata(metadata_xlsx, sheet_name=metadata_sheet)
    merged = merge_summary_with_hit_types(df, hit_df)

    if cluster_filter is not None and "cluster_change" in merged.columns:
        merged = merged[merged["cluster_change"] == cluster_filter].copy()

    if drop_unknown:
        merged = merged[merged["lesion_hit_type"] != "unknown"].copy()

    hit_type_order = _infer_hit_type_order(merged["lesion_hit_type"].tolist())
    contrast_pairs = _target_contrast_pairs(hit_type_order)

    merged_csv = out_dir / "merged_cluster_level_with_hit_type.csv"
    merged.to_csv(merged_csv, index=False)

    # Include paired RMS columns in bird-level mean summary even if not in DEFAULT_METRICS
    bird_mean_metric_cols = list(dict.fromkeys(
        list(metrics)
        + [spec["pre_col"] for spec in PAIRED_METRIC_SPECS]
        + [spec["post_col"] for spec in PAIRED_METRIC_SPECS]
    ))
    bird_mean = make_bird_level_summary(merged, metrics=bird_mean_metric_cols)
    bird_mean_csv = out_dir / "bird_level_mean_by_animal.csv"
    bird_mean.to_csv(bird_mean_csv, index=False)

    cluster_stats_all: List[pd.DataFrame] = []
    bird_mean_stats_all: List[pd.DataFrame] = []
    paired_cluster_stats_all: List[pd.DataFrame] = []
    paired_bird_mean_stats_all: List[pd.DataFrame] = []
    paired_cluster_between_stats_all: List[pd.DataFrame] = []
    paired_bird_mean_between_stats_all: List[pd.DataFrame] = []

    # Standard by-hit-type plots for reduced main metrics
    for metric in metrics:
        if metric not in merged.columns:
            print(f"[skip] metric not found in CSV: {metric}")
            continue

        # cluster-level plot + stats
        df_metric = merged[["lesion_hit_type", metric]].copy()
        df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")
        df_metric = df_metric[np.isfinite(df_metric[metric])].copy()

        if len(df_metric) > 0:
            stats_df = _pairwise_stats_for_metric(
                df_metric,
                group_col="lesion_hit_type",
                value_col=metric,
                groups=hit_type_order,
                level_label="cluster_level",
                contrast_pairs=contrast_pairs,
            )

            metric_label = _display_metric_name(metric)
            metric_ylabel = _display_ylabel(metric)

            out_png = cluster_dir / f"{_clean_filename(metric_label)}_cluster_level_by_hit_type.png"
            _box_strip_plot(
                df_metric,
                group_col="lesion_hit_type",
                value_col=metric,
                groups=hit_type_order,
                title=f"{metric_label} by lesion hit type (cluster level)",
                ylabel=metric_ylabel,
                out_png=out_png,
                stats_df=stats_df,
            )

            if len(stats_df) > 0:
                cluster_stats_all.append(stats_df)

        # bird mean plot + stats
        if metric in bird_mean.columns:
            df_bm = bird_mean[["lesion_hit_type", metric]].copy()
            df_bm[metric] = pd.to_numeric(df_bm[metric], errors="coerce")
            df_bm = df_bm[np.isfinite(df_bm[metric])].copy()

            if len(df_bm) > 0:
                stats_df = _pairwise_stats_for_metric(
                    df_bm,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    level_label="bird_level_mean",
                    contrast_pairs=contrast_pairs,
                )

                metric_label = _display_metric_name(metric)
                metric_ylabel = _display_ylabel(metric)

                out_png = bird_mean_dir / f"{_clean_filename(metric_label)}_bird_mean_by_hit_type.png"

                _box_strip_plot(
                    df_bm,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    title=f"{metric_label} by lesion hit type (bird mean)",
                    ylabel=metric_ylabel,
                    out_png=out_png,
                    stats_df=stats_df,
                    annotate_only_significant=False,
                )

                if len(stats_df) > 0:
                    bird_mean_stats_all.append(stats_df)

    # Paired pre/post RMS-radius plots at cluster level and bird-mean level
    for spec in PAIRED_METRIC_SPECS:
        pre_col = spec["pre_col"]
        post_col = spec["post_col"]
        metric_name = spec["name"]
        ylabel = spec["ylabel"]

        # cluster level
        if pre_col in merged.columns and post_col in merged.columns:
            stats_df_cluster = _paired_stats_for_metric(
                merged,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                metric_name=metric_name,
                level_label="paired_cluster_level",
            )

            if len(stats_df_cluster) > 0:
                paired_cluster_stats_all.append(stats_df_cluster)

            between_stats_cluster = _paired_between_group_stats(
                merged,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                metric_name=metric_name,
                level_label="paired_cluster_level_between_groups",
                contrast_pairs=contrast_pairs,
            )
            if len(between_stats_cluster) > 0:
                paired_cluster_between_stats_all.append(between_stats_cluster)

            out_png = paired_cluster_dir / f"paired_{_clean_filename(metric_name)}_cluster_level_pre_vs_post_grouped_by_hit_type.png"
            _paired_grouped_pre_post_boxplot(
                merged,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                ylabel=ylabel,
                title=f"{metric_name}: pre vs post within lesion hit type (all clusters)",
                out_png=out_png,
                stats_df=stats_df_cluster,
                between_stats_df=between_stats_cluster,
            )
        else:
            print(f"[skip paired cluster] missing columns for {metric_name}: {pre_col}, {post_col}")

        # bird mean
        if pre_col in bird_mean.columns and post_col in bird_mean.columns:
            stats_df_bird = _paired_stats_for_metric(
                bird_mean,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                metric_name=metric_name,
                level_label="paired_bird_level_mean",
            )

            if len(stats_df_bird) > 0:
                paired_bird_mean_stats_all.append(stats_df_bird)

            between_stats_bird = _paired_between_group_stats(
                bird_mean,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                metric_name=metric_name,
                level_label="paired_bird_level_mean_between_groups",
                contrast_pairs=contrast_pairs,
            )
            if len(between_stats_bird) > 0:
                paired_bird_mean_between_stats_all.append(between_stats_bird)

            out_png = paired_bird_mean_dir / f"paired_{_clean_filename(metric_name)}_bird_mean_pre_vs_post_grouped_by_hit_type.png"
            _paired_grouped_pre_post_boxplot(
                bird_mean,
                group_col="lesion_hit_type",
                pre_col=pre_col,
                post_col=post_col,
                groups=hit_type_order,
                ylabel=ylabel,
                title=f"{metric_name}: pre vs post within lesion hit type (bird means)",
                out_png=out_png,
                stats_df=stats_df_bird,
                between_stats_df=between_stats_bird,
            )
        else:
            print(f"[skip paired bird mean] missing columns for {metric_name}: {pre_col}, {post_col}")

    # Save stats tables
    cluster_stats_csv = stats_dir / "cluster_level_stats.csv"
    bird_mean_stats_csv = stats_dir / "bird_level_mean_stats.csv"
    paired_cluster_stats_csv = stats_dir / "paired_cluster_level_pre_post_stats.csv"
    paired_bird_mean_stats_csv = stats_dir / "paired_bird_level_mean_pre_post_stats.csv"
    paired_cluster_between_stats_csv = stats_dir / "paired_cluster_level_between_hit_type_stats.csv"
    paired_bird_mean_between_stats_csv = stats_dir / "paired_bird_level_mean_between_hit_type_stats.csv"

    if len(cluster_stats_all) > 0:
        pd.concat(cluster_stats_all, ignore_index=True).to_csv(cluster_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(cluster_stats_csv, index=False)

    if len(bird_mean_stats_all) > 0:
        pd.concat(bird_mean_stats_all, ignore_index=True).to_csv(bird_mean_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(bird_mean_stats_csv, index=False)

    if len(paired_cluster_stats_all) > 0:
        pd.concat(paired_cluster_stats_all, ignore_index=True).to_csv(paired_cluster_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(paired_cluster_stats_csv, index=False)

    if len(paired_bird_mean_stats_all) > 0:
        pd.concat(paired_bird_mean_stats_all, ignore_index=True).to_csv(paired_bird_mean_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(paired_bird_mean_stats_csv, index=False)

    if len(paired_cluster_between_stats_all) > 0:
        pd.concat(paired_cluster_between_stats_all, ignore_index=True).to_csv(paired_cluster_between_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(paired_cluster_between_stats_csv, index=False)

    if len(paired_bird_mean_between_stats_all) > 0:
        pd.concat(paired_bird_mean_between_stats_all, ignore_index=True).to_csv(paired_bird_mean_between_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(paired_bird_mean_between_stats_csv, index=False)

    print(f"Saved merged CSV: {merged_csv}")
    print(f"Saved bird mean CSV: {bird_mean_csv}")
    print(f"Saved cluster stats CSV: {cluster_stats_csv}")
    print(f"Saved bird mean stats CSV: {bird_mean_stats_csv}")
    print(f"Saved paired cluster-level pre/post stats CSV: {paired_cluster_stats_csv}")
    print(f"Saved paired bird mean pre/post stats CSV: {paired_bird_mean_stats_csv}")
    print(f"Saved paired cluster-level between-hit-type stats CSV: {paired_cluster_between_stats_csv}")
    print(f"Saved paired bird mean between-hit-type stats CSV: {paired_bird_mean_between_stats_csv}")

    return {
        "merged_csv": merged_csv,
        "bird_mean_csv": bird_mean_csv,
        "cluster_stats_csv": cluster_stats_csv,
        "bird_mean_stats_csv": bird_mean_stats_csv,
        "paired_cluster_stats_csv": paired_cluster_stats_csv,
        "paired_bird_mean_stats_csv": paired_bird_mean_stats_csv,
        "paired_cluster_between_stats_csv": paired_cluster_between_stats_csv,
        "paired_bird_mean_between_stats_csv": paired_bird_mean_between_stats_csv,
        "out_dir": out_dir,
    }


# ----------------------------
# CLI
# ----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_graph_batch_pre_post_variance.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--summary-csv", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)

    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str)
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional list of main summary metrics to graph. If omitted, uses the reduced default set.",
    )
    p.add_argument(
        "--cluster-filter",
        default=None,
        type=str,
        help="Optional cluster_change filter, e.g. present_pre_and_post, appeared_after_treatment, disappeared_after_treatment",
    )
    p.add_argument("--drop-unknown", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_metrics_by_hit_type(
        summary_csv=Path(args.summary_csv),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=out_dir,
        metadata_sheet=args.metadata_sheet,
        metrics=args.metrics,
        cluster_filter=args.cluster_filter,
        drop_unknown=bool(args.drop_unknown),
    )


if __name__ == "__main__":
    main()