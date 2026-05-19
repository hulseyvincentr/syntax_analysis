#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_delta_spread_analysis.py

Test the hypothesis that lesion effects cancel out at the bird-mean level because
some clusters/syllables become less stable while others become more stable.

Input
-----
A batch UMAP/BC summary CSV containing, at minimum:
    animal_id
    bc_pre_early_vs_late_equal_groups
    bc_post_early_vs_late_equal_groups

and your lesion metadata Excel file.

Core calculation
----------------
For each cluster row:
    delta_BC = post early-vs-late BC - pre early-vs-late BC

Interpretation:
    delta_BC < 0  : lower post early/late overlap; more post-lesion divergence
    delta_BC > 0  : higher post early/late overlap; less post-lesion divergence
    abs(delta_BC) : amount of reorganization, regardless of direction

This script computes bird-level summaries first, then compares birds across lesion
groups. That helps avoid treating many clusters from the same bird as independent
animals.

Outputs
-------
out_dir/
    cluster_level_delta_values.csv
    bird_level_delta_spread_summary.csv
    bird_level_delta_spread_group_stats.csv
    cluster_level_delta_distribution_group_stats.csv
    within_bird_pre_post_distribution_stats.csv

    figures/
        bird_level_mean_delta_BC_by_hit_type.png
        bird_level_median_delta_BC_by_hit_type.png
        bird_level_mean_abs_delta_BC_by_hit_type.png
        bird_level_median_abs_delta_BC_by_hit_type.png
        bird_level_sd_delta_BC_by_hit_type.png
        bird_level_iqr_delta_BC_by_hit_type.png
        bird_level_fraction_strongly_changed_by_hit_type.png
        bird_level_fraction_decreased_by_hit_type.png
        bird_level_fraction_increased_by_hit_type.png
        cluster_level_delta_BC_distribution_by_hit_type.png
        cluster_level_delta_BC_distribution_by_bird.png
        bird_level_delta_metrics_summary_panel.png
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
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


DEFAULT_PRE_COL = "bc_pre_early_vs_late_equal_groups"
DEFAULT_POST_COL = "bc_post_early_vs_late_equal_groups"

DEFAULT_HIT_TYPE_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Medial/Lateral visible + large lesion",
]

COLOR_BY_GROUP = {
    "sham saline injection": "#ff1a1a",
    "Area X visible (Lateral only)": "#c4b5fd",
    "Area X visible (Medial+Lateral hit)": "#6a3d9a",
    "large lesion Area X not visible": "#000000",
    "unknown": "#808080",
}


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
    Collapse raw metadata labels into the broad lesion groups used in your BC plots.
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


def _canonical_color_group(raw_hit_type: str) -> str:
    """
    Keep a more detailed lesion category for point colors.
    """
    norm = _normalize_text(raw_hit_type)

    if "sham" in norm:
        return "sham saline injection"

    if "medial" in norm and "lateral" in norm:
        return "Area X visible (Medial+Lateral hit)"

    if "singlehit" in norm:
        return "Area X visible (Lateral only)"

    if "lateralonly" in norm:
        return "Area X visible (Lateral only)"

    if ("lateral" in norm) and ("medial" not in norm) and ("large" not in norm) and ("notvisible" not in norm):
        return "Area X visible (Lateral only)"

    if ("large" in norm and "lesion" in norm) or ("notvisible" in norm):
        return "large lesion Area X not visible"

    return "unknown"


def _color_for_group(group: str) -> str:
    return COLOR_BY_GROUP.get(str(group), COLOR_BY_GROUP["unknown"])


def _infer_hit_type_order(values: Sequence[str]) -> List[str]:
    vals = [str(v) for v in values if pd.notna(v)]
    present = list(dict.fromkeys(vals))
    ordered = [v for v in DEFAULT_HIT_TYPE_ORDER if v in present]
    remaining = [v for v in present if v not in ordered]
    return ordered + remaining


def _target_contrast_pairs(groups: Sequence[str]) -> List[Tuple[str, str]]:
    """
    By default, compare sham against each lesion group present.
    """
    groups = list(groups)
    present = set(groups)
    sham = "sham saline injection"

    pairs: List[Tuple[str, str]] = []
    if sham in present:
        for g in groups:
            if g != sham:
                pairs.append((sham, g))

    return pairs


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
        val = float(m - i) * float(pv)
        if val < prev:
            val = prev
        prev = val
        adj_sorted[i] = min(1.0, val)

    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def _mannwhitney_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), "empty group"
    if not HAVE_SCIPY:
        return float("nan"), float("nan"), "scipy unavailable"

    if np.all(a == a.flat[0]) and np.all(b == b.flat[0]) and a.flat[0] == b.flat[0]:
        return 0.0, 1.0, "all values equal"

    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue), ""
    except TypeError:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), ""
    except Exception as e:
        return float("nan"), float("nan"), repr(e)


def _ks_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), "empty group"
    if not HAVE_SCIPY:
        return float("nan"), float("nan"), "scipy unavailable"

    try:
        res = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue), ""
    except TypeError:
        res = stats.ks_2samp(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), ""
    except Exception as e:
        return float("nan"), float("nan"), repr(e)


def _wilcoxon_one_sample_safe(x: np.ndarray) -> Tuple[float, float, str]:
    """
    One-sample Wilcoxon signed-rank test against 0.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan"), float("nan"), "no data"
    if x.size < 2:
        return float("nan"), float("nan"), "n<2"
    if not HAVE_SCIPY:
        return float("nan"), float("nan"), "scipy unavailable"
    if np.allclose(x, 0):
        return 0.0, 1.0, "all values are zero"

    try:
        stat, p = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), repr(e)


def _add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, color="black", clip_on=False)
    text_pad = max(h * 0.4, 0.01)
    ax.text((x1 + x2) / 2.0, y + h + text_pad, text,
            ha="center", va="bottom", fontsize=11, clip_on=False)


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
    out["lesion_color_group"] = out["lesion_hit_type_raw"].map(_canonical_color_group)
    out = out.drop_duplicates(subset=["animal_id"])
    return out


def load_and_prepare_data(
    summary_csv: Path,
    metadata_xlsx: Path,
    *,
    metadata_sheet: str,
    pre_col: str,
    post_col: str,
    cluster_filter: Optional[str],
    drop_unknown: bool,
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    if "animal_id" not in df.columns:
        raise KeyError("Summary CSV must contain an 'animal_id' column.")
    if pre_col not in df.columns:
        raise KeyError(f"Summary CSV does not contain pre column: {pre_col}")
    if post_col not in df.columns:
        raise KeyError(f"Summary CSV does not contain post column: {post_col}")

    df["animal_id"] = df["animal_id"].astype(str).str.strip()

    hit_df = load_hit_type_metadata(metadata_xlsx, sheet_name=metadata_sheet)
    merged = df.merge(hit_df, on="animal_id", how="left")

    merged["lesion_hit_type_raw"] = merged["lesion_hit_type_raw"].fillna("unknown")
    merged["lesion_hit_type"] = merged["lesion_hit_type"].fillna("unknown")
    merged["lesion_color_group"] = merged["lesion_color_group"].fillna("unknown")

    if cluster_filter is not None and "cluster_change" in merged.columns:
        merged = merged[merged["cluster_change"] == cluster_filter].copy()

    if drop_unknown:
        merged = merged[merged["lesion_hit_type"] != "unknown"].copy()

    merged[pre_col] = pd.to_numeric(merged[pre_col], errors="coerce")
    merged[post_col] = pd.to_numeric(merged[post_col], errors="coerce")

    valid = merged[np.isfinite(merged[pre_col]) & np.isfinite(merged[post_col])].copy()
    valid["delta_BC"] = valid[post_col] - valid[pre_col]
    valid["abs_delta_BC"] = np.abs(valid["delta_BC"])

    return valid


def compute_bird_level_summary(
    valid: pd.DataFrame,
    *,
    pre_col: str,
    post_col: str,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (animal_id, hit_type, color_group), sub in valid.groupby(
        ["animal_id", "lesion_hit_type", "lesion_color_group"], dropna=False
    ):
        delta = sub["delta_BC"].to_numpy(dtype=float)
        abs_delta = np.abs(delta)
        pre = sub[pre_col].to_numpy(dtype=float)
        post = sub[post_col].to_numpy(dtype=float)

        delta = delta[np.isfinite(delta)]
        abs_delta = abs_delta[np.isfinite(abs_delta)]
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]

        n = int(len(delta))
        if n == 0:
            continue

        n_decreased = int(np.sum(delta < -threshold))
        n_increased = int(np.sum(delta > threshold))
        n_strong = int(np.sum(np.abs(delta) > threshold))

        rows.append({
            "animal_id": animal_id,
            "lesion_hit_type": hit_type,
            "lesion_color_group": color_group,
            "n_clusters_valid": n,

            "mean_pre_BC": float(np.mean(pre)) if len(pre) else np.nan,
            "median_pre_BC": float(np.median(pre)) if len(pre) else np.nan,
            "mean_post_BC": float(np.mean(post)) if len(post) else np.nan,
            "median_post_BC": float(np.median(post)) if len(post) else np.nan,

            "mean_delta_BC": float(np.mean(delta)),
            "median_delta_BC": float(np.median(delta)),
            "mean_abs_delta_BC": float(np.mean(abs_delta)),
            "median_abs_delta_BC": float(np.median(abs_delta)),
            "sd_delta_BC": float(np.std(delta, ddof=1)) if n >= 2 else np.nan,
            "var_delta_BC": float(np.var(delta, ddof=1)) if n >= 2 else np.nan,
            "iqr_delta_BC": float(np.percentile(delta, 75) - np.percentile(delta, 25)) if n >= 2 else np.nan,

            "n_delta_decreased": n_decreased,
            "n_delta_increased": n_increased,
            "n_delta_strongly_changed": n_strong,
            "fraction_decreased": float(n_decreased / n),
            "fraction_increased": float(n_increased / n),
            "fraction_strongly_changed": float(n_strong / n),
            "threshold_abs_delta_BC": float(threshold),
        })

    return pd.DataFrame(rows)


def compute_bird_level_group_stats(
    bird_summary: pd.DataFrame,
    *,
    groups: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    contrast_pairs = _target_contrast_pairs(groups)

    raw_p_indices: List[int] = []
    raw_ps: List[float] = []

    for metric in metrics:
        if metric not in bird_summary.columns:
            continue

        # One-sample within-group test against zero for signed delta metrics only.
        if metric in ["mean_delta_BC", "median_delta_BC"]:
            for g in groups:
                vals = pd.to_numeric(
                    bird_summary.loc[bird_summary["lesion_hit_type"] == g, metric],
                    errors="coerce",
                ).to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                stat, p_raw, note = _wilcoxon_one_sample_safe(vals)
                row = {
                    "analysis": "within_group_test_against_zero",
                    "metric": metric,
                    "test": "one_sample_wilcoxon_signed_rank_vs_0",
                    "group_1": g,
                    "group_2": "",
                    "n_1": int(len(vals)),
                    "n_2": 0,
                    "statistic": stat,
                    "p_raw": p_raw,
                    "note": note,
                }
                rows.append(row)
                if np.isfinite(p_raw):
                    raw_p_indices.append(len(rows) - 1)
                    raw_ps.append(float(p_raw))

        # Between-group comparisons at bird level.
        for g1, g2 in contrast_pairs:
            a = pd.to_numeric(
                bird_summary.loc[bird_summary["lesion_hit_type"] == g1, metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            b = pd.to_numeric(
                bird_summary.loc[bird_summary["lesion_hit_type"] == g2, metric],
                errors="coerce",
            ).to_numpy(dtype=float)

            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]

            stat, p_raw, note = _mannwhitney_safe(a, b)
            row = {
                "analysis": "between_group_bird_level",
                "metric": metric,
                "test": "mannwhitneyu",
                "group_1": g1,
                "group_2": g2,
                "n_1": int(len(a)),
                "n_2": int(len(b)),
                "statistic": stat,
                "p_raw": p_raw,
                "note": note,
            }
            rows.append(row)
            if np.isfinite(p_raw):
                raw_p_indices.append(len(rows) - 1)
                raw_ps.append(float(p_raw))

    if len(rows) == 0:
        return pd.DataFrame()

    p_adj = np.full(len(rows), np.nan, dtype=float)
    if len(raw_ps) > 0:
        adj_vals = _holm_bonferroni(raw_ps)
        for idx, val in zip(raw_p_indices, adj_vals):
            p_adj[idx] = float(val)

    for i, row in enumerate(rows):
        row["p_holm"] = p_adj[i]
        row["sig_label"] = _p_to_stars(p_adj[i]) if np.isfinite(p_adj[i]) else "n/a"

    return pd.DataFrame(rows)


def compute_cluster_distribution_group_stats(
    valid: pd.DataFrame,
    *,
    groups: Sequence[str],
) -> pd.DataFrame:
    """
    Descriptive cluster-level comparison of delta distributions between groups.
    This is not animal-level inference, but useful for seeing the pooled pattern.
    """
    rows: List[Dict[str, object]] = []
    contrast_pairs = _target_contrast_pairs(groups)

    raw_p_indices: List[int] = []
    raw_ps: List[float] = []

    for metric in ["delta_BC", "abs_delta_BC"]:
        for g1, g2 in contrast_pairs:
            a = valid.loc[valid["lesion_hit_type"] == g1, metric].to_numpy(dtype=float)
            b = valid.loc[valid["lesion_hit_type"] == g2, metric].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]

            mw_stat, mw_p, mw_note = _mannwhitney_safe(a, b)
            row = {
                "analysis": "pooled_cluster_level_distribution",
                "metric": metric,
                "test": "mannwhitneyu",
                "group_1": g1,
                "group_2": g2,
                "n_1": int(len(a)),
                "n_2": int(len(b)),
                "statistic": mw_stat,
                "p_raw": mw_p,
                "note": mw_note,
            }
            rows.append(row)
            if np.isfinite(mw_p):
                raw_p_indices.append(len(rows) - 1)
                raw_ps.append(float(mw_p))

            ks_stat, ks_p, ks_note = _ks_safe(a, b)
            row = {
                "analysis": "pooled_cluster_level_distribution",
                "metric": metric,
                "test": "ks_2samp",
                "group_1": g1,
                "group_2": g2,
                "n_1": int(len(a)),
                "n_2": int(len(b)),
                "statistic": ks_stat,
                "p_raw": ks_p,
                "note": ks_note,
            }
            rows.append(row)
            if np.isfinite(ks_p):
                raw_p_indices.append(len(rows) - 1)
                raw_ps.append(float(ks_p))

    if len(rows) == 0:
        return pd.DataFrame()

    p_adj = np.full(len(rows), np.nan, dtype=float)
    if len(raw_ps) > 0:
        adj_vals = _holm_bonferroni(raw_ps)
        for idx, val in zip(raw_p_indices, adj_vals):
            p_adj[idx] = float(val)

    for i, row in enumerate(rows):
        row["p_holm"] = p_adj[i]
        row["sig_label"] = _p_to_stars(p_adj[i]) if np.isfinite(p_adj[i]) else "n/a"

    return pd.DataFrame(rows)


def compute_within_bird_distribution_stats(
    valid: pd.DataFrame,
    *,
    pre_col: str,
    post_col: str,
) -> pd.DataFrame:
    """
    For each bird separately, compare its pre cluster BC distribution vs post cluster BC distribution.
    This describes whether each bird's cluster distribution shifts.
    """
    rows: List[Dict[str, object]] = []

    for (animal_id, hit_type, color_group), sub in valid.groupby(
        ["animal_id", "lesion_hit_type", "lesion_color_group"], dropna=False
    ):
        pre = sub[pre_col].to_numpy(dtype=float)
        post = sub[post_col].to_numpy(dtype=float)
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]

        mw_stat, mw_p, mw_note = _mannwhitney_safe(pre, post)
        ks_stat, ks_p, ks_note = _ks_safe(pre, post)

        rows.append({
            "animal_id": animal_id,
            "lesion_hit_type": hit_type,
            "lesion_color_group": color_group,
            "n_clusters_valid": int(len(sub)),
            "test": "mannwhitneyu_pre_vs_post_distribution_within_bird",
            "statistic": mw_stat,
            "p_raw": mw_p,
            "note": mw_note,
        })
        rows.append({
            "animal_id": animal_id,
            "lesion_hit_type": hit_type,
            "lesion_color_group": color_group,
            "n_clusters_valid": int(len(sub)),
            "test": "ks_2samp_pre_vs_post_distribution_within_bird",
            "statistic": ks_stat,
            "p_raw": ks_p,
            "note": ks_note,
        })

    return pd.DataFrame(rows)


def _plot_bird_metric_by_group(
    bird_summary: pd.DataFrame,
    *,
    metric: str,
    groups: Sequence[str],
    ylabel: str,
    title: str,
    out_png: Path,
    stats_df: Optional[pd.DataFrame] = None,
    rng_seed: int = 0,
    add_zero_line: bool = False,
) -> None:
    rng = np.random.default_rng(rng_seed)

    plot_values: List[np.ndarray] = []
    plot_labels: List[str] = []
    plot_colors: List[List[str]] = []
    plot_groups: List[str] = []

    for g in groups:
        sub = bird_summary.loc[bird_summary["lesion_hit_type"] == g].copy()
        vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        vals = vals[mask]
        color_groups = sub.loc[mask, "lesion_color_group"].astype(str).tolist()

        if len(vals) > 0:
            plot_values.append(vals)
            plot_labels.append(f"{g}\n(n={len(vals)})")
            plot_colors.append([_color_for_group(cg) for cg in color_groups])
            plot_groups.append(g)

    if len(plot_values) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7.5, 3.4 * len(plot_values)), 5.4))

    try:
        bp = ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=False, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(plot_values, labels=plot_labels, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set(facecolor="white", alpha=0.9)

    for i, (vals, colors) in enumerate(zip(plot_values, plot_colors), start=1):
        x = _jitter_positions(len(vals), i, 0.08, rng)
        ax.scatter(x, vals, s=55, alpha=0.85, c=colors, edgecolors="none", zorder=3)

    if add_zero_line:
        ax.axhline(0, lw=1.2, ls="--", color="black", alpha=0.7)

    # Annotate between-group stats if available.
    if stats_df is not None and len(stats_df) > 0:
        sub_stats = stats_df[
            (stats_df["analysis"] == "between_group_bird_level")
            & (stats_df["metric"] == metric)
        ].copy()

        if len(sub_stats) > 0:
            pos_map = {g: i + 1 for i, g in enumerate(plot_groups)}
            all_vals = np.concatenate(plot_values)
            all_vals = all_vals[np.isfinite(all_vals)]
            ymin = float(np.min(all_vals)) if len(all_vals) else 0.0
            ymax = float(np.max(all_vals)) if len(all_vals) else 1.0
            y_span = max(1e-9, ymax - ymin)
            base = ymax + max(0.10 * y_span, 0.08)
            h = max(0.04 * y_span, 0.04)
            step = max(0.16 * y_span, 0.12)

            drawn = 0
            for _, row in sub_stats.iterrows():
                g1 = str(row["group_1"])
                g2 = str(row["group_2"])
                if g1 not in pos_map or g2 not in pos_map:
                    continue
                p_adj = pd.to_numeric(pd.Series([row.get("p_holm", np.nan)]), errors="coerce").iloc[0]
                sig = str(row.get("sig_label", "n/a"))
                label = sig if not np.isfinite(p_adj) else f"{sig}\np={p_adj:.3g}"
                x1 = float(pos_map[g1])
                x2 = float(pos_map[g2])
                if x1 > x2:
                    x1, x2 = x2, x1
                _add_sig_bracket(ax, x1, x2, base + drawn * step, h, label)
                drawn += 1

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


def _plot_cluster_delta_by_group(
    valid: pd.DataFrame,
    *,
    metric: str,
    groups: Sequence[str],
    ylabel: str,
    title: str,
    out_png: Path,
    stats_df: Optional[pd.DataFrame] = None,
    rng_seed: int = 0,
    add_zero_line: bool = True,
) -> None:
    rng = np.random.default_rng(rng_seed)

    plot_values: List[np.ndarray] = []
    plot_labels: List[str] = []
    plot_colors: List[List[str]] = []
    plot_groups: List[str] = []

    for g in groups:
        sub = valid.loc[valid["lesion_hit_type"] == g].copy()
        vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        vals = vals[mask]
        color_groups = sub.loc[mask, "lesion_color_group"].astype(str).tolist()

        if len(vals) > 0:
            plot_values.append(vals)
            plot_labels.append(f"{g}\n(n={len(vals)})")
            plot_colors.append([_color_for_group(cg) for cg in color_groups])
            plot_groups.append(g)

    if len(plot_values) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8.5, 3.8 * len(plot_values)), 5.8))

    try:
        bp = ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=False, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(plot_values, labels=plot_labels, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set(facecolor="white", alpha=0.72)

    for i, (vals, colors) in enumerate(zip(plot_values, plot_colors), start=1):
        x = _jitter_positions(len(vals), i, 0.12, rng)
        ax.scatter(x, vals, s=24, alpha=0.55, c=colors, edgecolors="none", zorder=3)

    if add_zero_line:
        ax.axhline(0, lw=1.2, ls="--", color="black", alpha=0.7)

    # Annotate pooled cluster-level between-group stats.
    if stats_df is not None and len(stats_df) > 0:
        sub_stats = stats_df[
            (stats_df["analysis"] == "pooled_cluster_level_distribution")
            & (stats_df["metric"] == metric)
            & (stats_df["test"] == "mannwhitneyu")
        ].copy()

        if len(sub_stats) > 0:
            pos_map = {g: i + 1 for i, g in enumerate(plot_groups)}
            all_vals = np.concatenate(plot_values)
            all_vals = all_vals[np.isfinite(all_vals)]
            ymin = float(np.min(all_vals)) if len(all_vals) else 0.0
            ymax = float(np.max(all_vals)) if len(all_vals) else 1.0
            y_span = max(1e-9, ymax - ymin)
            base = ymax + max(0.10 * y_span, 0.08)
            h = max(0.04 * y_span, 0.04)
            step = max(0.16 * y_span, 0.12)

            drawn = 0
            for _, row in sub_stats.iterrows():
                g1 = str(row["group_1"])
                g2 = str(row["group_2"])
                if g1 not in pos_map or g2 not in pos_map:
                    continue
                p_adj = pd.to_numeric(pd.Series([row.get("p_holm", np.nan)]), errors="coerce").iloc[0]
                sig = str(row.get("sig_label", "n/a"))
                label = sig if not np.isfinite(p_adj) else f"{sig}\np={p_adj:.3g}"
                x1 = float(pos_map[g1])
                x2 = float(pos_map[g2])
                if x1 > x2:
                    x1, x2 = x2, x1
                _add_sig_bracket(ax, x1, x2, base + drawn * step, h, label)
                drawn += 1

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


def _plot_cluster_delta_by_bird(
    valid: pd.DataFrame,
    *,
    groups: Sequence[str],
    out_png: Path,
    rng_seed: int = 0,
) -> None:
    """
    Plot per-cluster delta distributions separated by bird.
    Birds are ordered by lesion group, then median delta.
    """
    rng = np.random.default_rng(rng_seed)

    bird_order = []
    labels = []
    values = []
    colors = []

    for g in groups:
        sub_g = valid.loc[valid["lesion_hit_type"] == g].copy()
        if len(sub_g) == 0:
            continue
        med = sub_g.groupby("animal_id")["delta_BC"].median().sort_values()
        for animal_id in med.index:
            sub = sub_g.loc[sub_g["animal_id"] == animal_id]
            vals = sub["delta_BC"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            color_group = str(sub["lesion_color_group"].iloc[0])
            bird_order.append(animal_id)
            labels.append(f"{animal_id}\n{g}\n(n={len(vals)})")
            values.append(vals)
            colors.append(_color_for_group(color_group))

    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(values)), 6.0))

    try:
        bp = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set(facecolor="white", alpha=0.7)

    for i, (vals, c) in enumerate(zip(values, colors), start=1):
        x = _jitter_positions(len(vals), i, 0.10, rng)
        ax.scatter(x, vals, s=20, alpha=0.55, color=c, edgecolors="none", zorder=3)

    ax.axhline(0, lw=1.2, ls="--", color="black", alpha=0.7)
    ax.set_ylabel("Delta BC = post early-vs-late BC - pre early-vs-late BC")
    ax.set_xlabel("Bird")
    ax.set_title("Cluster-level delta BC distributions by bird")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_panel(
    bird_summary: pd.DataFrame,
    *,
    groups: Sequence[str],
    out_png: Path,
    rng_seed: int = 0,
) -> None:
    """
    Four-panel summary:
      mean signed delta
      mean absolute delta
      SD of delta
      fraction strongly changed
    """
    rng = np.random.default_rng(rng_seed)

    panel_metrics = [
        ("mean_delta_BC", "Mean signed delta BC", True),
        ("mean_abs_delta_BC", "Mean absolute delta BC", False),
        ("sd_delta_BC", "SD of delta BC", False),
        ("fraction_strongly_changed", "Fraction strongly changed", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()

    for ax, (metric, ylabel, add_zero) in zip(axes, panel_metrics):
        plot_values = []
        plot_labels = []
        plot_colors = []

        for g in groups:
            sub = bird_summary.loc[bird_summary["lesion_hit_type"] == g].copy()
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals)
            vals = vals[mask]
            color_groups = sub.loc[mask, "lesion_color_group"].astype(str).tolist()

            if len(vals) > 0:
                plot_values.append(vals)
                plot_labels.append(f"{g}\n(n={len(vals)})")
                plot_colors.append([_color_for_group(cg) for cg in color_groups])

        if len(plot_values) == 0:
            ax.axis("off")
            continue

        try:
            bp = ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=False, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(plot_values, labels=plot_labels, showfliers=False, patch_artist=True)

        for patch in bp["boxes"]:
            patch.set(facecolor="white", alpha=0.9)

        for i, (vals, colors) in enumerate(zip(plot_values, plot_colors), start=1):
            x = _jitter_positions(len(vals), i, 0.08, rng)
            ax.scatter(x, vals, s=45, alpha=0.85, c=colors, edgecolors="none", zorder=3)

        if add_zero:
            ax.axhline(0, lw=1.1, ls="--", color="black", alpha=0.7)

        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("Bird-level BC delta/spread metrics", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    summary_csv: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    metadata_sheet: str,
    pre_col: str,
    post_col: str,
    threshold: float,
    cluster_filter: Optional[str],
    drop_unknown: bool,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    _safe_mkdir(out_dir)
    fig_dir = out_dir / "figures"
    _safe_mkdir(fig_dir)

    valid = load_and_prepare_data(
        summary_csv=summary_csv,
        metadata_xlsx=metadata_xlsx,
        metadata_sheet=metadata_sheet,
        pre_col=pre_col,
        post_col=post_col,
        cluster_filter=cluster_filter,
        drop_unknown=drop_unknown,
    )

    groups = _infer_hit_type_order(valid["lesion_hit_type"].tolist())

    cluster_csv = out_dir / "cluster_level_delta_values.csv"
    valid.to_csv(cluster_csv, index=False)

    bird_summary = compute_bird_level_summary(
        valid,
        pre_col=pre_col,
        post_col=post_col,
        threshold=threshold,
    )
    bird_summary_csv = out_dir / "bird_level_delta_spread_summary.csv"
    bird_summary.to_csv(bird_summary_csv, index=False)

    bird_metrics = [
        "mean_delta_BC",
        "median_delta_BC",
        "mean_abs_delta_BC",
        "median_abs_delta_BC",
        "sd_delta_BC",
        "var_delta_BC",
        "iqr_delta_BC",
        "fraction_decreased",
        "fraction_increased",
        "fraction_strongly_changed",
    ]

    bird_stats = compute_bird_level_group_stats(
        bird_summary,
        groups=groups,
        metrics=bird_metrics,
    )
    bird_stats_csv = out_dir / "bird_level_delta_spread_group_stats.csv"
    bird_stats.to_csv(bird_stats_csv, index=False)

    cluster_stats = compute_cluster_distribution_group_stats(
        valid,
        groups=groups,
    )
    cluster_stats_csv = out_dir / "cluster_level_delta_distribution_group_stats.csv"
    cluster_stats.to_csv(cluster_stats_csv, index=False)

    within_bird_stats = compute_within_bird_distribution_stats(
        valid,
        pre_col=pre_col,
        post_col=post_col,
    )
    within_bird_stats_csv = out_dir / "within_bird_pre_post_distribution_stats.csv"
    within_bird_stats.to_csv(within_bird_stats_csv, index=False)

    # Bird-level figures.
    metric_specs = [
        ("mean_delta_BC", "Mean signed delta BC\n(post early-vs-late BC - pre early-vs-late BC)", "bird_level_mean_delta_BC_by_hit_type.png", True),
        ("median_delta_BC", "Median signed delta BC\n(post early-vs-late BC - pre early-vs-late BC)", "bird_level_median_delta_BC_by_hit_type.png", True),
        ("mean_abs_delta_BC", "Mean absolute delta BC", "bird_level_mean_abs_delta_BC_by_hit_type.png", False),
        ("median_abs_delta_BC", "Median absolute delta BC", "bird_level_median_abs_delta_BC_by_hit_type.png", False),
        ("sd_delta_BC", "SD of delta BC across clusters", "bird_level_sd_delta_BC_by_hit_type.png", False),
        ("iqr_delta_BC", "IQR of delta BC across clusters", "bird_level_iqr_delta_BC_by_hit_type.png", False),
        ("fraction_strongly_changed", f"Fraction of clusters with |delta BC| > {threshold:g}", "bird_level_fraction_strongly_changed_by_hit_type.png", False),
        ("fraction_decreased", f"Fraction of clusters with delta BC < -{threshold:g}", "bird_level_fraction_decreased_by_hit_type.png", False),
        ("fraction_increased", f"Fraction of clusters with delta BC > {threshold:g}", "bird_level_fraction_increased_by_hit_type.png", False),
    ]

    for metric, ylabel, filename, zero_line in metric_specs:
        _plot_bird_metric_by_group(
            bird_summary,
            metric=metric,
            groups=groups,
            ylabel=ylabel,
            title=metric,
            out_png=fig_dir / filename,
            stats_df=bird_stats,
            add_zero_line=zero_line,
        )

    # Pooled cluster-level distribution figures.
    _plot_cluster_delta_by_group(
        valid,
        metric="delta_BC",
        groups=groups,
        ylabel="Delta BC = post early-vs-late BC - pre early-vs-late BC",
        title="Pooled cluster-level delta BC distribution",
        out_png=fig_dir / "cluster_level_delta_BC_distribution_by_hit_type.png",
        stats_df=cluster_stats,
        add_zero_line=True,
    )

    _plot_cluster_delta_by_group(
        valid,
        metric="abs_delta_BC",
        groups=groups,
        ylabel="Absolute delta BC",
        title="Pooled cluster-level absolute delta BC distribution",
        out_png=fig_dir / "cluster_level_abs_delta_BC_distribution_by_hit_type.png",
        stats_df=cluster_stats,
        add_zero_line=False,
    )

    _plot_cluster_delta_by_bird(
        valid,
        groups=groups,
        out_png=fig_dir / "cluster_level_delta_BC_distribution_by_bird.png",
    )

    _plot_summary_panel(
        bird_summary,
        groups=groups,
        out_png=fig_dir / "bird_level_delta_metrics_summary_panel.png",
    )

    print()
    print("=" * 80)
    print("BC delta/spread analysis complete")
    print("=" * 80)
    print(f"Valid cluster rows: {len(valid)}")
    print(f"Birds: {bird_summary['animal_id'].nunique() if len(bird_summary) else 0}")
    print()
    print(f"Saved cluster-level delta values: {cluster_csv}")
    print(f"Saved bird-level summary:        {bird_summary_csv}")
    print(f"Saved bird-level stats:          {bird_stats_csv}")
    print(f"Saved cluster-level stats:       {cluster_stats_csv}")
    print(f"Saved within-bird stats:         {within_bird_stats_csv}")
    print(f"Saved figures in:                {fig_dir}")
    print()
    print("Main figures to inspect first:")
    print(f"  {fig_dir / 'bird_level_mean_abs_delta_BC_by_hit_type.png'}")
    print(f"  {fig_dir / 'bird_level_sd_delta_BC_by_hit_type.png'}")
    print(f"  {fig_dir / 'bird_level_fraction_strongly_changed_by_hit_type.png'}")
    print(f"  {fig_dir / 'cluster_level_delta_BC_distribution_by_bird.png'}")
    print()

    return {
        "cluster_csv": cluster_csv,
        "bird_summary_csv": bird_summary_csv,
        "bird_stats_csv": bird_stats_csv,
        "cluster_stats_csv": cluster_stats_csv,
        "within_bird_stats_csv": within_bird_stats_csv,
        "fig_dir": fig_dir,
        "out_dir": out_dir,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bc_delta_spread_analysis.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Calculate and plot bird-level BC delta/spread metrics.",
    )

    p.add_argument("--summary-csv", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)

    p.add_argument("--metadata-sheet", default="animal_hit_type_summary", type=str)
    p.add_argument("--pre-col", default=DEFAULT_PRE_COL, type=str)
    p.add_argument("--post-col", default=DEFAULT_POST_COL, type=str)
    p.add_argument(
        "--threshold",
        default=0.10,
        type=float,
        help="Threshold for calling a cluster strongly changed based on abs(delta_BC).",
    )
    p.add_argument(
        "--cluster-filter",
        default=None,
        type=str,
        help="Optional cluster_change filter, such as present_pre_and_post.",
    )
    p.add_argument("--drop-unknown", action="store_true")

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_analysis(
        summary_csv=Path(args.summary_csv),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=Path(args.out_dir),
        metadata_sheet=args.metadata_sheet,
        pre_col=args.pre_col,
        post_col=args.post_col,
        threshold=float(args.threshold),
        cluster_filter=args.cluster_filter,
        drop_unknown=bool(args.drop_unknown),
    )


if __name__ == "__main__":
    main()
