#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_graph_batch_pre_post_variance.py

%run /Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/umap_graph_batch_pre_post_variance.py --summary-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/batch_umap_pre_post_variance/batch_cluster_variance_bc_summary.csv" --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/umap_graph_batch_pre_post_variance"

Graph batch UMAP/variance summary metrics by lesion hit type, and make paired
pre-vs-post bird-level plots.

Inputs
------
1) batch_cluster_variance_bc_summary.csv
   Produced by umap_batch_pre_post_variance.py

2) Area_X_lesion_metadata_with_hit_types.xlsx
   Uses sheet "animal_hit_type_summary" by default, merging on Animal ID.

Outputs
-------
out_dir/
  merged_cluster_level_with_hit_type.csv
  bird_level_mean_by_animal.csv
  bird_level_median_by_animal.csv

  cluster_level/
    <metric>_cluster_level_by_hit_type.png

  bird_level_mean/
    <metric>_bird_mean_by_hit_type.png

  bird_level_median/
    <metric>_bird_median_by_hit_type.png

  paired_bird_level_mean/
    paired_<metric_name>_bird_mean_pre_vs_post_by_hit_type.png

  stats/
    cluster_level_stats.csv
    bird_level_mean_stats.csv
    bird_level_median_stats.csv
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
    # BC metrics
    "bc_pre_early_vs_late",
    "bc_post_early_vs_late",
    "bc_pre_vs_post",

    # raw/all-data centroid + spread metrics
    "pre_centroid_norm_raw",
    "post_centroid_norm_raw",
    "pre_centroid_dist_to_cluster_global_raw",
    "post_centroid_dist_to_cluster_global_raw",
    "centroid_shift_raw",
    "pre_rms_radius_raw",
    "post_rms_radius_raw",
    "post_over_pre_rms_radius",
    "pre_trace_cov_raw",
    "post_trace_cov_raw",
    "post_over_pre_trace_cov",

    # equal-group BC metrics
    "bc_pre_early_vs_late_equal_groups",
    "bc_post_early_vs_late_equal_groups",
    "bc_pre_vs_post_equal_groups",

    # equal-group centroid + spread metrics
    "pre_centroid_norm_raw_equal_groups",
    "post_centroid_norm_raw_equal_groups",
    "pre_centroid_dist_to_cluster_global_raw_equal_groups",
    "post_centroid_dist_to_cluster_global_raw_equal_groups",
    "centroid_shift_raw_equal_groups",
    "pre_rms_radius_raw_equal_groups",
    "post_rms_radius_raw_equal_groups",
    "post_over_pre_rms_radius_equal_groups",
    "pre_trace_cov_raw_equal_groups",
    "post_trace_cov_raw_equal_groups",
    "post_over_pre_trace_cov_equal_groups",
]

PAIRED_METRIC_SPECS = [
    {
        "name": "bhattacharyya_coefficient",
        "pre_col": "bc_pre_early_vs_late",
        "post_col": "bc_post_early_vs_late",
        "ylabel": "Bhattacharyya coefficient (BC)",
    },
    {
        "name": "centroid_distance_to_cluster_global",
        "pre_col": "pre_centroid_dist_to_cluster_global_raw",
        "post_col": "post_centroid_dist_to_cluster_global_raw",
        "ylabel": "Distance to cluster-global centroid",
    },
    {
        "name": "rms_radius",
        "pre_col": "pre_rms_radius_raw",
        "post_col": "post_rms_radius_raw",
        "ylabel": "RMS radius",
    },
    {
        "name": "trace_covariance",
        "pre_col": "pre_trace_cov_raw",
        "post_col": "post_trace_cov_raw",
        "ylabel": "Trace of covariance",
    },

    # equal-group versions
    {
        "name": "bhattacharyya_coefficient_equal_groups",
        "pre_col": "bc_pre_early_vs_late_equal_groups",
        "post_col": "bc_post_early_vs_late_equal_groups",
        "ylabel": "Bhattacharyya coefficient (BC), equal groups",
    },
    {
        "name": "centroid_distance_to_cluster_global_equal_groups",
        "pre_col": "pre_centroid_dist_to_cluster_global_raw_equal_groups",
        "post_col": "post_centroid_dist_to_cluster_global_raw_equal_groups",
        "ylabel": "Distance to cluster-global centroid, equal groups",
    },
    {
        "name": "rms_radius_equal_groups",
        "pre_col": "pre_rms_radius_raw_equal_groups",
        "post_col": "post_rms_radius_raw_equal_groups",
        "ylabel": "RMS radius, equal groups",
    },
    {
        "name": "trace_covariance_equal_groups",
        "pre_col": "pre_trace_cov_raw_equal_groups",
        "post_col": "post_trace_cov_raw_equal_groups",
        "ylabel": "Trace of covariance, equal groups",
    },
]

DEFAULT_HIT_TYPE_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Medial/Lateral visible + large lesion",
]


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

    for i in range(len(present_groups)):
        for j in range(i + 1, len(present_groups)):
            g1 = present_groups[i]
            g2 = present_groups[j]
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
    bird_col: str,
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
        sub = df.loc[df[group_col] == g, [bird_col, pre_col, post_col]].copy()
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
            "n_birds": int(len(sub)),
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
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paired_pre_post_plot(
    df: pd.DataFrame,
    *,
    group_col: str,
    bird_col: str,
    pre_col: str,
    post_col: str,
    groups: List[str],
    ylabel: str,
    title: str,
    out_png: Path,
    stats_df: Optional[pd.DataFrame] = None,
) -> None:
    n_panels = max(1, len(groups))
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 5.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    stats_map = {}
    if stats_df is not None and len(stats_df) > 0:
        stats_map = {str(r["group"]): r for _, r in stats_df.iterrows()}

    for ax, g in zip(axes, groups):
        sub = df.loc[df[group_col] == g, [bird_col, pre_col, post_col]].copy()
        sub[pre_col] = pd.to_numeric(sub[pre_col], errors="coerce")
        sub[post_col] = pd.to_numeric(sub[post_col], errors="coerce")
        sub = sub[np.isfinite(sub[pre_col]) & np.isfinite(sub[post_col])].copy()

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No birds", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Pre", "Post"])
            ax.set_title(g)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        y1 = sub[pre_col].to_numpy(dtype=float)
        y2 = sub[post_col].to_numpy(dtype=float)

        ax.boxplot([y1, y2], positions=[1, 2], widths=0.5, showfliers=False)
        for _, r in sub.iterrows():
            ax.plot([1, 2], [r[pre_col], r[post_col]], alpha=0.8)
            ax.scatter([1, 2], [r[pre_col], r[post_col]], s=22)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_title(g)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        stat_row = stats_map.get(g, None)
        if stat_row is not None:
            p_txt = stat_row.get("p_holm", np.nan)
            n_txt = stat_row.get("n_birds", len(sub))
            note = stat_row.get("note", "")
            sig = stat_row.get("sig_label", "n/a")
            if np.isfinite(p_txt):
                label = f"{sig} (p={p_txt:.3g}, n={int(n_txt)})"
            else:
                label = f"n/a (n={int(n_txt)})"
            if isinstance(note, str) and len(note) > 0:
                label = label + f"\n{note}"
            ax.text(0.5, 0.98, label, ha="center", va="top", transform=ax.transAxes, fontsize=10)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Load + merge
# ----------------------------

def load_batch_summary(summary_csv: Path) -> pd.DataFrame:
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
    agg: str = "mean",
) -> pd.DataFrame:
    keep_cols = ["animal_id", "lesion_hit_type"]
    use_cols = keep_cols + [m for m in metrics if m in df.columns]

    sub = df[use_cols].copy()
    for m in metrics:
        if m in sub.columns:
            sub[m] = pd.to_numeric(sub[m], errors="coerce")

    grouped = sub.groupby(["animal_id", "lesion_hit_type"], as_index=False)

    if agg == "median":
        return grouped.median(numeric_only=True)
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
    bird_median_dir = out_dir / "bird_level_median"
    paired_bird_mean_dir = out_dir / "paired_bird_level_mean"
    stats_dir = out_dir / "stats"
    for p in [cluster_dir, bird_mean_dir, bird_median_dir, paired_bird_mean_dir, stats_dir]:
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

    merged_csv = out_dir / "merged_cluster_level_with_hit_type.csv"
    merged.to_csv(merged_csv, index=False)

    bird_mean = make_bird_level_summary(merged, metrics=metrics, agg="mean")
    bird_mean_csv = out_dir / "bird_level_mean_by_animal.csv"
    bird_mean.to_csv(bird_mean_csv, index=False)

    bird_median = make_bird_level_summary(merged, metrics=metrics, agg="median")
    bird_median_csv = out_dir / "bird_level_median_by_animal.csv"
    bird_median.to_csv(bird_median_csv, index=False)

    cluster_stats_all: List[pd.DataFrame] = []
    bird_mean_stats_all: List[pd.DataFrame] = []
    bird_median_stats_all: List[pd.DataFrame] = []
    paired_bird_mean_stats_all: List[pd.DataFrame] = []

    # Standard by-hit-type plots
    for metric in metrics:
        if metric not in merged.columns:
            print(f"[skip] metric not found in CSV: {metric}")
            continue

        # cluster-level plot + stats
        df_metric = merged[["lesion_hit_type", metric]].copy()
        df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")
        df_metric = df_metric[np.isfinite(df_metric[metric])].copy()

        if len(df_metric) > 0:
            out_png = cluster_dir / f"{_clean_filename(metric)}_cluster_level_by_hit_type.png"
            _box_strip_plot(
                df_metric,
                group_col="lesion_hit_type",
                value_col=metric,
                groups=hit_type_order,
                title=f"{metric} by lesion hit type (cluster level)",
                ylabel=metric,
                out_png=out_png,
            )

            stats_df = _pairwise_stats_for_metric(
                df_metric,
                group_col="lesion_hit_type",
                value_col=metric,
                groups=hit_type_order,
                level_label="cluster_level",
            )
            if len(stats_df) > 0:
                cluster_stats_all.append(stats_df)

        # bird mean plot + stats
        if metric in bird_mean.columns:
            df_bm = bird_mean[["lesion_hit_type", metric]].copy()
            df_bm[metric] = pd.to_numeric(df_bm[metric], errors="coerce")
            df_bm = df_bm[np.isfinite(df_bm[metric])].copy()

            if len(df_bm) > 0:
                out_png = bird_mean_dir / f"{_clean_filename(metric)}_bird_mean_by_hit_type.png"
                _box_strip_plot(
                    df_bm,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    title=f"{metric} by lesion hit type (bird mean)",
                    ylabel=metric,
                    out_png=out_png,
                )

                stats_df = _pairwise_stats_for_metric(
                    df_bm,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    level_label="bird_level_mean",
                )
                if len(stats_df) > 0:
                    bird_mean_stats_all.append(stats_df)

        # bird median plot + stats
        if metric in bird_median.columns:
            df_bmed = bird_median[["lesion_hit_type", metric]].copy()
            df_bmed[metric] = pd.to_numeric(df_bmed[metric], errors="coerce")
            df_bmed = df_bmed[np.isfinite(df_bmed[metric])].copy()

            if len(df_bmed) > 0:
                out_png = bird_median_dir / f"{_clean_filename(metric)}_bird_median_by_hit_type.png"
                _box_strip_plot(
                    df_bmed,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    title=f"{metric} by lesion hit type (bird median)",
                    ylabel=metric,
                    out_png=out_png,
                )

                stats_df = _pairwise_stats_for_metric(
                    df_bmed,
                    group_col="lesion_hit_type",
                    value_col=metric,
                    groups=hit_type_order,
                    level_label="bird_level_median",
                )
                if len(stats_df) > 0:
                    bird_median_stats_all.append(stats_df)

    # Paired pre/post bird-level mean plots
    for spec in PAIRED_METRIC_SPECS:
        pre_col = spec["pre_col"]
        post_col = spec["post_col"]
        metric_name = spec["name"]
        ylabel = spec["ylabel"]

        if pre_col not in bird_mean.columns or post_col not in bird_mean.columns:
            print(f"[skip paired] missing columns for {metric_name}: {pre_col}, {post_col}")
            continue

        stats_df = _paired_stats_for_metric(
            bird_mean,
            group_col="lesion_hit_type",
            bird_col="animal_id",
            pre_col=pre_col,
            post_col=post_col,
            groups=hit_type_order,
            metric_name=metric_name,
            level_label="paired_bird_level_mean",
        )

        if len(stats_df) > 0:
            paired_bird_mean_stats_all.append(stats_df)

        out_png = paired_bird_mean_dir / f"paired_{_clean_filename(metric_name)}_bird_mean_pre_vs_post_by_hit_type.png"
        _paired_pre_post_plot(
            bird_mean,
            group_col="lesion_hit_type",
            bird_col="animal_id",
            pre_col=pre_col,
            post_col=post_col,
            groups=hit_type_order,
            ylabel=ylabel,
            title=f"{metric_name}: bird mean pre vs post by lesion hit type",
            out_png=out_png,
            stats_df=stats_df,
        )

    # Save stats tables
    cluster_stats_csv = stats_dir / "cluster_level_stats.csv"
    bird_mean_stats_csv = stats_dir / "bird_level_mean_stats.csv"
    bird_median_stats_csv = stats_dir / "bird_level_median_stats.csv"
    paired_bird_mean_stats_csv = stats_dir / "paired_bird_level_mean_pre_post_stats.csv"

    if len(cluster_stats_all) > 0:
        pd.concat(cluster_stats_all, ignore_index=True).to_csv(cluster_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(cluster_stats_csv, index=False)

    if len(bird_mean_stats_all) > 0:
        pd.concat(bird_mean_stats_all, ignore_index=True).to_csv(bird_mean_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(bird_mean_stats_csv, index=False)

    if len(bird_median_stats_all) > 0:
        pd.concat(bird_median_stats_all, ignore_index=True).to_csv(bird_median_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(bird_median_stats_csv, index=False)

    if len(paired_bird_mean_stats_all) > 0:
        pd.concat(paired_bird_mean_stats_all, ignore_index=True).to_csv(paired_bird_mean_stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(paired_bird_mean_stats_csv, index=False)

    print(f"Saved merged CSV: {merged_csv}")
    print(f"Saved bird mean CSV: {bird_mean_csv}")
    print(f"Saved bird median CSV: {bird_median_csv}")
    print(f"Saved cluster stats CSV: {cluster_stats_csv}")
    print(f"Saved bird mean stats CSV: {bird_mean_stats_csv}")
    print(f"Saved bird median stats CSV: {bird_median_stats_csv}")
    print(f"Saved paired bird mean pre/post stats CSV: {paired_bird_mean_stats_csv}")

    return {
        "merged_csv": merged_csv,
        "bird_mean_csv": bird_mean_csv,
        "bird_median_csv": bird_median_csv,
        "cluster_stats_csv": cluster_stats_csv,
        "bird_mean_stats_csv": bird_mean_stats_csv,
        "bird_median_stats_csv": bird_median_stats_csv,
        "paired_bird_mean_stats_csv": paired_bird_mean_stats_csv,
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
        help="Optional list of metrics to graph. If omitted, uses the default metric list.",
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

    graph_metrics_by_hit_type(
        summary_csv=Path(args.summary_csv),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=Path(args.out_dir),
        metadata_sheet=args.metadata_sheet,
        metrics=args.metrics,
        cluster_filter=args.cluster_filter,
        drop_unknown=bool(args.drop_unknown),
    )


if __name__ == "__main__":
    main()