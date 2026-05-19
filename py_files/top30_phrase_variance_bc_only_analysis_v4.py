#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Bhattacharyya coefficient (BC) only for the top fraction of
high phrase-duration-variance syllables within each bird.

This script answers two questions using only the top 30% (or other chosen
fraction) highest post-lesion phrase-duration-variance syllables per bird:

1) Cluster-level question:
   Within each lesion hit type, do the distributions of pre vs post
   early-vs-late BC values differ for the selected high-variance syllables?

2) Bird-level question:
   After averaging BC across the selected high-variance syllables within each
   bird, do bird-level mean BC values differ across lesion hit types?

Outputs are BC-only: no RMS radius, centroid shift, or other acoustic-spread
metrics are computed or plotted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import itertools
import math
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

DEFAULT_BC_PRE_COL = "bc_pre_early_vs_late_equal_groups"
DEFAULT_BC_POST_COL = "bc_post_early_vs_late_equal_groups"
DEFAULT_HIT_TYPE_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    "Complete and partial Medial and Lateral lesion",
]
COLOR_BY_GROUP = {
    "sham saline injection": "#ff3b3b",
    "Area X visible (Lateral only)": "#c4b5fd",
    "Area X visible (Medial+Lateral hit)": "#6a3d9a",
    "large lesion Area X not visible": "#222222",
    "unknown": "#9e9e9e",
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _label_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def _collapse_hit_type(raw_hit_type: str) -> str:
    norm = _normalize_text(raw_hit_type)
    if "sham" in norm:
        return "sham saline injection"
    if "singlehit" in norm or "lateralonly" in norm:
        return "Lateral lesion only"
    if ("lateral" in norm and "medial" not in norm and "large" not in norm and "notvisible" not in norm):
        return "Lateral lesion only"
    if ("medial" in norm and "lateral" in norm) or ("large" in norm and "lesion" in norm) or ("notvisible" in norm):
        return "Complete and partial Medial and Lateral lesion"
    return "unknown"


def _canonical_color_group(raw_hit_type: str) -> str:
    norm = _normalize_text(raw_hit_type)
    if "sham" in norm:
        return "sham saline injection"
    if "medial" in norm and "lateral" in norm:
        return "Area X visible (Medial+Lateral hit)"
    if "singlehit" in norm or "lateralonly" in norm:
        return "Area X visible (Lateral only)"
    if ("lateral" in norm and "medial" not in norm and "large" not in norm and "notvisible" not in norm):
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


def _jitter_positions(n: int, center: float, width: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return center + rng.uniform(-width, width, size=n)


def _wrap_group_label(g: str) -> str:
    if g == "Complete and partial Medial and Lateral lesion":
        return "Complete and partial\nMedial and Lateral lesion"
    return g


def _clean_filename(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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


def _nanptp_safe(values: np.ndarray) -> float:
    """NumPy-version-safe nan-aware peak-to-peak range."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values) - np.nanmin(values))


def _format_p_label(p_value: float, sig_label: str) -> str:
    """Format significance annotation with stars plus adjusted p-value."""
    try:
        p_value = float(p_value)
    except Exception:
        return str(sig_label)
    if not np.isfinite(p_value):
        return str(sig_label)
    return f"{sig_label}\np={p_value:.3g}"


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


def _wilcoxon_paired_safe(pre: np.ndarray, post: np.ndarray) -> Tuple[float, float, str]:
    """
    Paired Wilcoxon signed-rank test for matched bird-level pre/post values.

    This is used for bird-level mean BC, where each bird contributes one
    pre value and one post value within a lesion hit type.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]

    if pre.size == 0:
        return float("nan"), float("nan"), "no paired data"
    if pre.size < 2:
        return float("nan"), float("nan"), "n<2"
    if not HAVE_SCIPY:
        return float("nan"), float("nan"), "scipy unavailable"

    diff = post - pre
    if np.allclose(diff, 0):
        return 0.0, 1.0, "all paired differences are zero"

    try:
        stat, p = stats.wilcoxon(post, pre, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), repr(e)


def _add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.6, color="black", clip_on=False)
    text_pad = max(h * 0.42, 0.010)
    ax.text(
        (x1 + x2) / 2.0,
        y + h + text_pad,
        text,
        ha="center",
        va="bottom",
        fontsize=16,
        clip_on=False,
        linespacing=1.05,
    )


def _add_panel_label_below_axis(ax, text: str, *, y: float = -0.18, fontsize: int = 15) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
    )

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
    if animal_id_col is None or hit_type_col is None:
        raise KeyError("Could not infer animal ID and lesion hit type columns from metadata.")

    out = df[[animal_id_col, hit_type_col]].copy()
    out.columns = ["animal_id", "lesion_hit_type_raw"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["lesion_hit_type_raw"] = out["lesion_hit_type_raw"].astype(str).str.strip()
    out["lesion_hit_type"] = out["lesion_hit_type_raw"].map(_collapse_hit_type)
    out["lesion_color_group"] = out["lesion_hit_type_raw"].map(_canonical_color_group)
    out = out.drop_duplicates(subset=["animal_id"])
    return out


def load_phrase_variance_table(
    phrase_csv: Path,
    *,
    animal_col: Optional[str],
    syllable_col: Optional[str],
    group_col: Optional[str],
    nphrases_col: Optional[str],
    variance_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(phrase_csv)
    if animal_col is None:
        animal_col = _pick_column(df, ["Animal ID", "animal_id", "animal id", "Bird ID", "bird"])
    if syllable_col is None:
        syllable_col = _pick_column(df, ["Syllable", "syllable", "cluster", "label"])
    if group_col is None:
        group_col = _pick_column(df, ["Group", "group", "period"])
    if nphrases_col is None:
        nphrases_col = _pick_column(df, ["N_phrases", "n_phrases", "nphrases", "count"])
    if variance_col is None:
        variance_col = _pick_column(df, ["Variance_ms2", "variance_ms2", "Post_Variance_ms2"])
    missing = [name for name, val in {
        "animal_col": animal_col,
        "syllable_col": syllable_col,
        "group_col": group_col,
        "nphrases_col": nphrases_col,
        "variance_col": variance_col,
    }.items() if val is None]
    if missing:
        raise KeyError(f"Could not infer required phrase-duration columns: {missing}")

    cols = {
        "animal_col": animal_col,
        "syllable_col": syllable_col,
        "group_col": group_col,
        "nphrases_col": nphrases_col,
        "variance_col": variance_col,
    }
    df = df.copy()
    df["animal_id"] = df[animal_col].astype(str).str.strip()
    df["syllable_key"] = df[syllable_col].map(_label_key)
    df["phrase_group"] = df[group_col].astype(str).str.strip()
    df["phrase_n_phrases"] = pd.to_numeric(df[nphrases_col], errors="coerce")
    df["phrase_variance_ms2"] = pd.to_numeric(df[variance_col], errors="coerce")
    return df, cols


def select_top_phrase_variance_syllables(
    phrase_df: pd.DataFrame,
    *,
    post_group_name: str,
    top_fraction: float,
    min_n_phrases: int,
) -> pd.DataFrame:
    if not (0 < top_fraction <= 1):
        raise ValueError("--top-fraction must be >0 and <=1")

    post = phrase_df.loc[
        phrase_df["phrase_group"].str.lower() == str(post_group_name).strip().lower()
    ].copy()
    post = post[np.isfinite(post["phrase_variance_ms2"])].copy()
    post = post[np.isfinite(post["phrase_n_phrases"])].copy()
    post = post[post["phrase_n_phrases"] >= min_n_phrases].copy()

    if len(post) == 0:
        raise ValueError(
            "No phrase-duration rows survived the post-group/min-n-phrases filters. "
            "Try lowering --min-n-phrases or check --post-group-name."
        )

    duplicate_counts = (
        post.groupby(["animal_id", "syllable_key"], dropna=False)
        .size()
        .rename("n_duplicate_post_rows_for_animal_syllable")
        .reset_index()
    )
    post = post.merge(
        duplicate_counts,
        on=["animal_id", "syllable_key"],
        how="left",
        validate="many_to_one",
    )

    post = (
        post.sort_values(
            ["animal_id", "syllable_key", "phrase_variance_ms2", "phrase_n_phrases"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["animal_id", "syllable_key"], keep="first")
        .copy()
    )
    post["duplicate_resolution"] = "max_post_phrase_variance_keep_one_row"

    out_parts: List[pd.DataFrame] = []
    for animal_id, sub in post.groupby("animal_id"):
        sub = sub.sort_values("phrase_variance_ms2", ascending=False).copy()
        n = len(sub)
        n_top = max(1, int(math.ceil(n * top_fraction)))
        sub["phrase_variance_rank_within_bird"] = np.arange(1, n + 1)
        sub["n_ranked_syllables_within_bird"] = n
        sub["n_top_syllables_within_bird"] = n_top
        sub["top_fraction"] = top_fraction
        sub["is_top_phrase_variance"] = False
        sub.iloc[:n_top, sub.columns.get_loc("is_top_phrase_variance")] = True
        out_parts.append(sub)

    return pd.concat(out_parts, ignore_index=True)


def load_bc_summary(
    bc_summary_csv: Path,
    *,
    animal_col: Optional[str],
    cluster_col: Optional[str],
    bc_pre_col: Optional[str],
    bc_post_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(bc_summary_csv)
    if animal_col is None:
        animal_col = _pick_column(df, ["animal_id", "Animal ID", "animal id", "bird"])
    if cluster_col is None:
        cluster_col = _pick_column(df, ["cluster", "syllable", "label"])
    if bc_pre_col is None:
        bc_pre_col = _pick_column(df, [DEFAULT_BC_PRE_COL, "bc_pre_early_late_equal_groups"])
    if bc_post_col is None:
        bc_post_col = _pick_column(df, [DEFAULT_BC_POST_COL, "bc_post_early_late_equal_groups"])
    missing = [name for name, val in {
        "animal_col": animal_col,
        "cluster_col": cluster_col,
        "bc_pre_col": bc_pre_col,
        "bc_post_col": bc_post_col,
    }.items() if val is None]
    if missing:
        raise KeyError(f"Could not infer required BC-summary columns: {missing}")

    cols = {
        "animal_col": animal_col,
        "cluster_col": cluster_col,
        "bc_pre_col": bc_pre_col,
        "bc_post_col": bc_post_col,
    }
    out = df[[animal_col, cluster_col, bc_pre_col, bc_post_col]].copy()
    out.columns = ["animal_id", "cluster", "bc_pre", "bc_post"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["syllable_key"] = out["cluster"].map(_label_key)
    out["bc_pre"] = pd.to_numeric(out["bc_pre"], errors="coerce")
    out["bc_post"] = pd.to_numeric(out["bc_post"], errors="coerce")
    out = out.drop_duplicates(subset=["animal_id", "syllable_key"], keep="first")
    return out, cols


def merge_phrase_bc_metadata(
    *,
    bc_df: pd.DataFrame,
    phrase_selected: pd.DataFrame,
    metadata_df: pd.DataFrame,
    drop_unknown: bool,
) -> pd.DataFrame:
    phrase_cols = [
        "animal_id",
        "syllable_key",
        "phrase_n_phrases",
        "phrase_variance_ms2",
        "phrase_variance_rank_within_bird",
        "n_ranked_syllables_within_bird",
        "n_top_syllables_within_bird",
        "is_top_phrase_variance",
        "top_fraction",
    ]
    merged = bc_df.merge(
        phrase_selected[phrase_cols],
        on=["animal_id", "syllable_key"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.merge(metadata_df, on="animal_id", how="left")
    merged["lesion_hit_type_raw"] = merged["lesion_hit_type_raw"].fillna("unknown")
    merged["lesion_hit_type"] = merged["lesion_hit_type"].fillna("unknown")
    merged["lesion_color_group"] = merged["lesion_color_group"].fillna("unknown")
    if drop_unknown:
        merged = merged[merged["lesion_hit_type"] != "unknown"].copy()
    merged["delta_bc_post_minus_pre"] = merged["bc_post"] - merged["bc_pre"]
    return merged


def compute_bird_level_summary(top_only: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (animal_id, hit_type, color_group), sub in top_only.groupby(
        ["animal_id", "lesion_hit_type", "lesion_color_group"], dropna=False
    ):
        row = {
            "animal_id": animal_id,
            "lesion_hit_type": hit_type,
            "lesion_color_group": color_group,
            "n_top_syllables_matched_to_bc": int(len(sub)),
            "mean_top_phrase_variance_ms2": float(np.nanmean(sub["phrase_variance_ms2"])),
            "bird_mean_bc_pre": float(np.nanmean(pd.to_numeric(sub["bc_pre"], errors="coerce"))),
            "bird_mean_bc_post": float(np.nanmean(pd.to_numeric(sub["bc_post"], errors="coerce"))),
            "bird_mean_delta_bc_post_minus_pre": float(np.nanmean(pd.to_numeric(sub["delta_bc_post_minus_pre"], errors="coerce"))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _add_holm_to_rows(rows: List[Dict[str, object]], raw_idx: List[int], raw_ps: List[float]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    p_adj = np.full(len(rows), np.nan, dtype=float)
    if raw_ps:
        adj_vals = _holm_bonferroni(raw_ps)
        for idx, val in zip(raw_idx, adj_vals):
            p_adj[idx] = float(val)
    for i, row in enumerate(rows):
        row["p_holm"] = p_adj[i]
        row["sig_label"] = _p_to_stars(p_adj[i]) if np.isfinite(p_adj[i]) else "n/a"
    return pd.DataFrame(rows)


def compute_cluster_level_stats(top_only: pd.DataFrame, *, groups: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    raw_idx: List[int] = []
    raw_ps: List[float] = []
    for g in groups:
        sub = top_only[top_only["lesion_hit_type"] == g].copy()
        pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        stat, p_raw, note = _mannwhitney_safe(pre, post)
        rows.append({
            "analysis": "cluster_level_top30_pre_vs_post_within_hit_type",
            "lesion_hit_type": g,
            "test": "mannwhitneyu",
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
            "pre_mean": float(np.mean(pre)) if len(pre) else np.nan,
            "post_mean": float(np.mean(post)) if len(post) else np.nan,
            "pre_median": float(np.median(pre)) if len(pre) else np.nan,
            "post_median": float(np.median(post)) if len(post) else np.nan,
            "statistic": stat,
            "p_raw": p_raw,
            "note": note,
        })
        if np.isfinite(p_raw):
            raw_idx.append(len(rows) - 1)
            raw_ps.append(float(p_raw))
    return _add_holm_to_rows(rows, raw_idx, raw_ps)


def compute_bird_level_between_hit_type_stats(bird_df: pd.DataFrame, *, groups: Sequence[str]) -> pd.DataFrame:
    metric_map = {
        "bird_mean_bc_pre": "Pre early-vs-late BC",
        "bird_mean_bc_post": "Post early-vs-late BC",
    }
    rows: List[Dict[str, object]] = []
    raw_idx: List[int] = []
    raw_ps: List[float] = []
    for value_col, metric_label in metric_map.items():
        for g1, g2 in itertools.combinations(groups, 2):
            a = pd.to_numeric(bird_df.loc[bird_df["lesion_hit_type"] == g1, value_col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(bird_df.loc[bird_df["lesion_hit_type"] == g2, value_col], errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            stat, p_raw, note = _mannwhitney_safe(a, b)
            rows.append({
                "analysis": "bird_level_top30_between_hit_type",
                "metric": metric_label,
                "value_col": value_col,
                "group_1": g1,
                "group_2": g2,
                "n_1": int(len(a)),
                "n_2": int(len(b)),
                "mean_1": float(np.mean(a)) if len(a) else np.nan,
                "mean_2": float(np.mean(b)) if len(b) else np.nan,
                "median_1": float(np.median(a)) if len(a) else np.nan,
                "median_2": float(np.median(b)) if len(b) else np.nan,
                "statistic": stat,
                "p_raw": p_raw,
                "note": note,
            })
            if np.isfinite(p_raw):
                raw_idx.append(len(rows) - 1)
                raw_ps.append(float(p_raw))
    return _add_holm_to_rows(rows, raw_idx, raw_ps)


def compute_bird_level_pre_post_within_hit_type_stats(
    bird_df: pd.DataFrame,
    *,
    groups: Sequence[str],
) -> pd.DataFrame:
    """
    Test whether bird-level mean pre early-vs-late BC differs from bird-level
    mean post early-vs-late BC within each lesion hit type.

    Uses a paired Wilcoxon signed-rank test because the pre/post values are
    paired within the same bird. Holm correction is applied across lesion hit
    types.
    """
    rows: List[Dict[str, object]] = []
    raw_idx: List[int] = []
    raw_ps: List[float] = []

    for g in groups:
        sub = bird_df[bird_df["lesion_hit_type"] == g].copy()
        pre = pd.to_numeric(sub["bird_mean_bc_pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["bird_mean_bc_post"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pre) & np.isfinite(post)
        pre = pre[mask]
        post = post[mask]
        diff = post - pre

        stat, p_raw, note = _wilcoxon_paired_safe(pre, post)

        rows.append({
            "analysis": "bird_level_top30_pre_vs_post_within_hit_type",
            "lesion_hit_type": g,
            "test": "paired_wilcoxon_signed_rank",
            "n_birds": int(len(pre)),
            "pre_mean": float(np.mean(pre)) if len(pre) else np.nan,
            "post_mean": float(np.mean(post)) if len(post) else np.nan,
            "pre_median": float(np.median(pre)) if len(pre) else np.nan,
            "post_median": float(np.median(post)) if len(post) else np.nan,
            "mean_post_minus_pre": float(np.mean(diff)) if len(diff) else np.nan,
            "median_post_minus_pre": float(np.median(diff)) if len(diff) else np.nan,
            "statistic": stat,
            "p_raw": p_raw,
            "note": note,
        })
        if np.isfinite(p_raw):
            raw_idx.append(len(rows) - 1)
            raw_ps.append(float(p_raw))

    return _add_holm_to_rows(rows, raw_idx, raw_ps)


def plot_cluster_level_top30(
    top_only: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    groups: Sequence[str],
    out_png: Path,
    show_p_values: bool = True,
) -> None:
    groups = [g for g in groups if g in set(top_only["lesion_hit_type"])]
    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 6.9), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    for ax, g in zip(axes, groups):
        sub = top_only[top_only["lesion_hit_type"] == g].copy()
        pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        data = [pre, post]
        colors = [_color_for_group(cg) for cg in sub["lesion_color_group"].astype(str).tolist()]

        try:
            bp = ax.boxplot(data, tick_labels=["Pre", "Post"], showfliers=False, patch_artist=True, widths=0.55)
        except TypeError:
            bp = ax.boxplot(data, labels=["Pre", "Post"], showfliers=False, patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set(facecolor="white", alpha=0.95, linewidth=1.5)
        for item in ["whiskers", "caps"]:
            for artist in bp[item]:
                artist.set(linewidth=1.3)
        for med in bp["medians"]:
            med.set(linewidth=2.0)

        if len(pre):
            pre_colors = colors[: len(pre)] if len(colors) >= len(pre) else ["#999999"] * len(pre)
            ax.scatter(_jitter_positions(len(pre), 1, 0.08, rng), pre, s=58, c=pre_colors, alpha=0.75, edgecolors="none")
        if len(post):
            post_colors = colors[: len(post)] if len(colors) >= len(post) else ["#999999"] * len(post)
            ax.scatter(_jitter_positions(len(post), 2, 0.08, rng), post, s=58, c=post_colors, alpha=0.75, edgecolors="none")

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", labelsize=15)

        _add_panel_label_below_axis(ax, f"{_wrap_group_label(g)}\n(n={len(sub)})", y=-0.16, fontsize=16)

        row = stats_df[stats_df["lesion_hit_type"] == g]
        if len(row):
            row = row.iloc[0]
            vals_all = np.concatenate([pre, post]) if (len(pre) + len(post)) else np.array([0.0, 1.0])
            y0 = float(np.nanmax(vals_all)) + 0.03 * max(1e-6, _nanptp_safe(vals_all) + 0.1)
            h = max(0.02, 0.035 * max(1e-6, _nanptp_safe(vals_all) + 0.1))
            label = _format_p_label(row.get("p_holm", np.nan), row["sig_label"]) if show_p_values else str(row["sig_label"])
            _add_sig_bracket(ax, 1, 2, y0, h, label)

    axes[0].set_ylabel("Bhattacharyya coefficient\n(early vs. late, each equal sample size)", fontsize=21)
    fig.suptitle(
        "Top 30% phrase-duration-variance syllables:\n"
        "cluster-level BC_early_late_equal_groups, pre vs post within lesion hit type",
        fontsize=25,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_bird_level_top30_means(
    bird_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    groups: Sequence[str],
    out_png: Path,
    show_p_values: bool = True,
) -> None:
    groups = [g for g in groups if g in set(bird_df["lesion_hit_type"])]
    if not groups:
        return

    panel_specs = [
        ("bird_mean_bc_pre", "Pre early-vs-late BC"),
        ("bird_mean_bc_post", "Post early-vs-late BC"),
    ]
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(8.4 * len(panel_specs), 6.4), sharey=True)
    if len(panel_specs) == 1:
        axes = [axes]
    rng = np.random.default_rng(1)

    for ax, (value_col, title) in zip(axes, panel_specs):
        plot_values = []
        plot_labels = []
        plot_colors = []
        present_groups = []
        for g in groups:
            sub = bird_df[bird_df["lesion_hit_type"] == g].copy()
            vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals)
            vals = vals[mask]
            if len(vals) == 0:
                continue
            color_groups = sub.loc[mask, "lesion_color_group"].astype(str).tolist()
            plot_values.append(vals)
            plot_labels.append(f"{_wrap_group_label(g)}\n(n={len(vals)})")
            plot_colors.append([_color_for_group(cg) for cg in color_groups])
            present_groups.append(g)

        try:
            bp = ax.boxplot(plot_values, tick_labels=plot_labels, showfliers=False, patch_artist=True, widths=0.55)
        except TypeError:
            bp = ax.boxplot(plot_values, labels=plot_labels, showfliers=False, patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set(facecolor="white", alpha=0.95, linewidth=1.5)
        for item in ["whiskers", "caps"]:
            for artist in bp[item]:
                artist.set(linewidth=1.3)
        for med in bp["medians"]:
            med.set(linewidth=2.0)

        for i, (vals, colors) in enumerate(zip(plot_values, plot_colors), start=1):
            ax.scatter(_jitter_positions(len(vals), i, 0.11, rng), vals, s=58, c=colors, alpha=0.82, edgecolors="none")

        ax.set_title(title, fontsize=19)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=15)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

        stat_sub = stats_df[stats_df["value_col"] == value_col].copy()
        stat_sub = stat_sub[np.isfinite(stat_sub["p_holm"])].copy()
        stat_sub = stat_sub.sort_values("p_holm", ascending=True)
        stat_sub = stat_sub[stat_sub["p_holm"] < 0.05].copy()

        if len(stat_sub):
            pos_map = {g: i + 1 for i, g in enumerate(present_groups)}
            all_vals = np.concatenate(plot_values)
            span = max(0.06, 0.10 * max(1e-6, _nanptp_safe(all_vals) + 0.05))
            y = float(np.nanmax(all_vals)) + 0.05 * max(1e-6, _nanptp_safe(all_vals) + 0.1)
            h = span * 0.35
            for _, row in stat_sub.iterrows():
                g1 = row["group_1"]
                g2 = row["group_2"]
                if g1 in pos_map and g2 in pos_map:
                    label = _format_p_label(row.get("p_holm", np.nan), row["sig_label"]) if show_p_values else str(row["sig_label"])
                    _add_sig_bracket(ax, pos_map[g1], pos_map[g2], y, h, label)
                    y += span

    axes[0].set_ylabel("Bird mean Bhattacharyya coefficient\n(top 30% phrase-variance syllables)", fontsize=20)
    fig.suptitle(
        "Bird-level mean BC for top 30% phrase-duration-variance syllables\nacross lesion hit types",
        fontsize=23,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_bird_level_top30_pre_vs_post_within_hit_type(
    bird_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    groups: Sequence[str],
    out_png: Path,
    show_p_values: bool = True,
) -> None:
    """
    Plot bird-level mean pre vs post early-vs-late BC for the top 30%
    phrase-variance syllables, separately for each lesion hit type.

    Each bird contributes one pre and one post mean. Lines connect paired
    pre/post values from the same bird. The p-value shown is the Holm-corrected
    paired Wilcoxon p-value within that lesion hit type.
    """
    groups = [g for g in groups if g in set(bird_df["lesion_hit_type"])]
    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 6.9), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    rng = np.random.default_rng(2)

    for ax, g in zip(axes, groups):
        sub = bird_df[bird_df["lesion_hit_type"] == g].copy()
        sub["bird_mean_bc_pre"] = pd.to_numeric(sub["bird_mean_bc_pre"], errors="coerce")
        sub["bird_mean_bc_post"] = pd.to_numeric(sub["bird_mean_bc_post"], errors="coerce")
        sub = sub[np.isfinite(sub["bird_mean_bc_pre"]) & np.isfinite(sub["bird_mean_bc_post"])].copy()

        pre = sub["bird_mean_bc_pre"].to_numpy(dtype=float)
        post = sub["bird_mean_bc_post"].to_numpy(dtype=float)
        colors = [_color_for_group(cg) for cg in sub["lesion_color_group"].astype(str).tolist()]

        data = [pre, post]
        try:
            bp = ax.boxplot(data, tick_labels=["Pre", "Post"], showfliers=False, patch_artist=True, widths=0.55)
        except TypeError:
            bp = ax.boxplot(data, labels=["Pre", "Post"], showfliers=False, patch_artist=True, widths=0.55)

        for patch in bp["boxes"]:
            patch.set(facecolor="white", alpha=0.95, linewidth=1.5)
        for item in ["whiskers", "caps"]:
            for artist in bp[item]:
                artist.set(linewidth=1.3)
        for med in bp["medians"]:
            med.set(linewidth=2.0)

        for pre_v, post_v, c in zip(pre, post, colors):
            ax.plot([1, 2], [pre_v, post_v], color=c, alpha=0.65, linewidth=1.3)
            ax.scatter([1, 2], [pre_v, post_v], color=c, s=58, alpha=0.88, edgecolors="none", zorder=3)

        if len(pre):
            ax.scatter(_jitter_positions(len(pre), 1, 0.06, rng), pre, s=38, c=colors, alpha=0.35, edgecolors="none", zorder=2)
        if len(post):
            ax.scatter(_jitter_positions(len(post), 2, 0.06, rng), post, s=38, c=colors, alpha=0.35, edgecolors="none", zorder=2)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=15)

        _add_panel_label_below_axis(ax, f"{_wrap_group_label(g)}\n(n={len(sub)} birds)", y=-0.16, fontsize=16)

        row = stats_df[stats_df["lesion_hit_type"] == g]
        if len(row):
            row = row.iloc[0]
            vals_all = np.concatenate([pre, post]) if (len(pre) + len(post)) else np.array([0.0, 1.0])
            y0 = float(np.nanmax(vals_all)) + 0.04 * max(1e-6, _nanptp_safe(vals_all) + 0.1)
            h = max(0.02, 0.035 * max(1e-6, _nanptp_safe(vals_all) + 0.1))
            label = _format_p_label(row.get("p_holm", np.nan), row["sig_label"]) if show_p_values else str(row["sig_label"])
            _add_sig_bracket(ax, 1, 2, y0, h, label)

    axes[0].set_ylabel(
        "Bird mean Bhattacharyya coefficient\n"
        "(top 30% phrase-variance syllables)",
        fontsize=20,
    )
    fig.suptitle(
        "Top 30% phrase-duration-variance syllables:\n"
        "bird-level mean BC_early_late_equal_groups, pre vs post within lesion hit type",
        fontsize=24,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run_analysis(
    *,
    phrase_csv: Path,
    bc_summary_csv: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    top_fraction: float,
    min_n_phrases: int,
    post_group_name: str,
    drop_unknown: bool,
    phrase_animal_col: Optional[str],
    phrase_syllable_col: Optional[str],
    phrase_group_col: Optional[str],
    phrase_nphrases_col: Optional[str],
    phrase_variance_col: Optional[str],
    bc_animal_col: Optional[str],
    bc_cluster_col: Optional[str],
    bc_pre_col: Optional[str],
    bc_post_col: Optional[str],
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    _safe_mkdir(out_dir)
    _safe_mkdir(fig_dir)

    phrase_df, phrase_cols = load_phrase_variance_table(
        phrase_csv,
        animal_col=phrase_animal_col,
        syllable_col=phrase_syllable_col,
        group_col=phrase_group_col,
        nphrases_col=phrase_nphrases_col,
        variance_col=phrase_variance_col,
    )
    phrase_selected = select_top_phrase_variance_syllables(
        phrase_df,
        post_group_name=post_group_name,
        top_fraction=top_fraction,
        min_n_phrases=min_n_phrases,
    )
    bc_df, bc_cols = load_bc_summary(
        bc_summary_csv,
        animal_col=bc_animal_col,
        cluster_col=bc_cluster_col,
        bc_pre_col=bc_pre_col,
        bc_post_col=bc_post_col,
    )
    metadata_df = load_hit_type_metadata(metadata_xlsx)
    merged = merge_phrase_bc_metadata(
        bc_df=bc_df,
        phrase_selected=phrase_selected,
        metadata_df=metadata_df,
        drop_unknown=drop_unknown,
    )
    top_only = merged[merged["is_top_phrase_variance"] == True].copy()
    if len(top_only) == 0:
        raise ValueError("After merging BC and phrase tables, zero top-variance rows remained.")

    groups = _infer_hit_type_order(top_only["lesion_hit_type"].dropna().tolist())
    bird_df = compute_bird_level_summary(top_only)
    cluster_stats = compute_cluster_level_stats(top_only, groups=groups)
    bird_stats = compute_bird_level_between_hit_type_stats(bird_df, groups=groups)
    bird_prepost_stats = compute_bird_level_pre_post_within_hit_type_stats(bird_df, groups=groups)

    selected_csv = out_dir / "selected_top_phrase_variance_syllables.csv"
    merged_csv = out_dir / "merged_top30_bc_rows.csv"
    bird_csv = out_dir / "bird_level_top30_bc_summary.csv"
    cluster_stats_csv = out_dir / "stats_cluster_level_top30_pre_vs_post.csv"
    bird_stats_csv = out_dir / "stats_bird_level_top30_between_hit_types.csv"
    bird_prepost_stats_csv = out_dir / "stats_bird_level_top30_pre_vs_post_within_hit_type.csv"
    manifest_csv = out_dir / "run_manifest.csv"
    fig_cluster = fig_dir / "cluster_level_top30_pre_vs_post_within_hit_type_bc_early_late_equal_groups_with_pvalues.png"
    fig_cluster_stars = fig_dir / "cluster_level_top30_pre_vs_post_within_hit_type_bc_early_late_equal_groups_stars_only.png"
    fig_bird = fig_dir / "bird_level_top30_mean_bc_across_hit_types_with_pvalues.png"
    fig_bird_stars = fig_dir / "bird_level_top30_mean_bc_across_hit_types_stars_only.png"
    fig_bird_prepost = fig_dir / "bird_level_top30_mean_pre_vs_post_within_hit_type_bc_early_late_equal_groups_with_pvalues.png"
    fig_bird_prepost_stars = fig_dir / "bird_level_top30_mean_pre_vs_post_within_hit_type_bc_early_late_equal_groups_stars_only.png"

    phrase_selected.to_csv(selected_csv, index=False)
    top_only.to_csv(merged_csv, index=False)
    bird_df.to_csv(bird_csv, index=False)
    cluster_stats.to_csv(cluster_stats_csv, index=False)
    bird_stats.to_csv(bird_stats_csv, index=False)
    bird_prepost_stats.to_csv(bird_prepost_stats_csv, index=False)

    manifest = pd.DataFrame([{
        "phrase_csv": str(phrase_csv),
        "bc_summary_csv": str(bc_summary_csv),
        "metadata_xlsx": str(metadata_xlsx),
        "out_dir": str(out_dir),
        "top_fraction": top_fraction,
        "min_n_phrases": min_n_phrases,
        "post_group_name": post_group_name,
        "drop_unknown": drop_unknown,
        "phrase_columns_used": str(phrase_cols),
        "bc_columns_used": str(bc_cols),
        "n_phrase_rows_total": int(len(phrase_df)),
        "n_post_ranked_rows": int(len(phrase_selected)),
        "n_top_rows_after_merge": int(len(top_only)),
        "n_birds_top_rows_after_merge": int(top_only["animal_id"].nunique()),
        "groups": ";".join(groups),
    }])
    manifest.to_csv(manifest_csv, index=False)

    plot_cluster_level_top30(top_only, cluster_stats, groups=groups, out_png=fig_cluster, show_p_values=True)
    plot_cluster_level_top30(top_only, cluster_stats, groups=groups, out_png=fig_cluster_stars, show_p_values=False)
    plot_bird_level_top30_means(bird_df, bird_stats, groups=groups, out_png=fig_bird, show_p_values=True)
    plot_bird_level_top30_means(bird_df, bird_stats, groups=groups, out_png=fig_bird_stars, show_p_values=False)
    plot_bird_level_top30_pre_vs_post_within_hit_type(
        bird_df,
        bird_prepost_stats,
        groups=groups,
        out_png=fig_bird_prepost,
        show_p_values=True,
    )
    plot_bird_level_top30_pre_vs_post_within_hit_type(
        bird_df,
        bird_prepost_stats,
        groups=groups,
        out_png=fig_bird_prepost_stars,
        show_p_values=False,
    )

    print()
    print("=" * 88)
    print("Top-variance BC-only analysis complete")
    print("=" * 88)
    print(f"Phrase rows loaded:                 {len(phrase_df)}")
    print(f"Post rows ranked:                  {len(phrase_selected)}")
    print(f"Top rows after BC merge:           {len(top_only)}")
    print(f"Birds represented:                 {top_only['animal_id'].nunique()}")
    print()
    print(f"Saved selected top syllables:      {selected_csv}")
    print(f"Saved merged top BC rows:          {merged_csv}")
    print(f"Saved bird-level summary:          {bird_csv}")
    print(f"Saved cluster-level stats:         {cluster_stats_csv}")
    print(f"Saved bird-level between-type stats: {bird_stats_csv}")
    print(f"Saved bird-level pre/post stats:    {bird_prepost_stats_csv}")
    print(f"Saved run manifest:                {manifest_csv}")
    print(f"Saved cluster-level figure (p):    {fig_cluster}")
    print(f"Saved cluster-level figure (*):    {fig_cluster_stars}")
    print(f"Saved bird between-type figure (p): {fig_bird}")
    print(f"Saved bird between-type figure (*): {fig_bird_stars}")
    print(f"Saved bird pre/post figure (p):    {fig_bird_prepost}")
    print(f"Saved bird pre/post figure (*):    {fig_bird_prepost_stars}")
    print()

    return {
        "selected_csv": selected_csv,
        "merged_csv": merged_csv,
        "bird_csv": bird_csv,
        "cluster_stats_csv": cluster_stats_csv,
        "bird_stats_csv": bird_stats_csv,
        "bird_prepost_stats_csv": bird_prepost_stats_csv,
        "manifest_csv": manifest_csv,
        "fig_cluster": fig_cluster,
        "fig_cluster_stars": fig_cluster_stars,
        "fig_bird": fig_bird,
        "fig_bird_stars": fig_bird_stars,
        "fig_bird_prepost": fig_bird_prepost,
        "fig_bird_prepost_stars": fig_bird_prepost_stars,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BC-only top-variance analysis for high phrase-duration-variance syllables."
    )
    p.add_argument("--phrase-csv", required=True, help="Path to phrase-duration summary CSV.")
    p.add_argument("--bc-summary-csv", required=True, help="Path to BC summary CSV.")
    p.add_argument("--metadata-xlsx", required=True, help="Path to metadata Excel file.")
    p.add_argument("--out-dir", required=True, help="Directory to save outputs.")

    p.add_argument("--top-fraction", type=float, default=0.30, help="Top fraction of post phrase-variance syllables within each bird (default: 0.30).")
    p.add_argument("--min-n-phrases", type=int, default=100, help="Minimum N_phrases in the phrase table to include a post syllable in ranking (default: 100).")
    p.add_argument("--post-group-name", default="Post", help="Name of the post-lesion group in the phrase CSV (default: Post).")
    p.add_argument("--drop-unknown", action="store_true", help="Drop birds with unknown lesion hit type.")

    p.add_argument("--phrase-animal-col", default=None)
    p.add_argument("--phrase-syllable-col", default=None)
    p.add_argument("--phrase-group-col", default=None)
    p.add_argument("--phrase-nphrases-col", default=None)
    p.add_argument("--phrase-variance-col", default=None)

    p.add_argument("--bc-animal-col", default=None)
    p.add_argument("--bc-cluster-col", default=None)
    p.add_argument("--bc-pre-col", default=DEFAULT_BC_PRE_COL)
    p.add_argument("--bc-post-col", default=DEFAULT_BC_POST_COL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        phrase_csv=Path(args.phrase_csv),
        bc_summary_csv=Path(args.bc_summary_csv),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=Path(args.out_dir),
        top_fraction=args.top_fraction,
        min_n_phrases=args.min_n_phrases,
        post_group_name=args.post_group_name,
        drop_unknown=bool(args.drop_unknown),
        phrase_animal_col=args.phrase_animal_col,
        phrase_syllable_col=args.phrase_syllable_col,
        phrase_group_col=args.phrase_group_col,
        phrase_nphrases_col=args.phrase_nphrases_col,
        phrase_variance_col=args.phrase_variance_col,
        bc_animal_col=args.bc_animal_col,
        bc_cluster_col=args.bc_cluster_col,
        bc_pre_col=args.bc_pre_col,
        bc_post_col=args.bc_post_col,
    )


if __name__ == "__main__":
    main()
