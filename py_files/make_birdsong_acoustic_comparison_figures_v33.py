#!/usr/bin/env python3
"""
Make higher-level comparison figures for pre/post-lesion birdsong acoustic analyses.

This script aggregates per-label outputs from two analysis families:

1) Pitch derivative / Wiener entropy pipeline
   Expected file per bird/label:
      *_per_segment_feature_summary_all_periods.csv

2) Spectrogram cross-correlation pipeline
   Expected files per bird/label:
      *_spectrogram_crosscorr_segments.csv
      *_pairwise_correlation_values.csv

It creates the figure types discussed for comparing acoustic structure:

A. Pooled pre/post ECDFs
B. Bird-level post-pre effect summaries
C. Baseline drift vs lesion-associated shift plots

The plotting/statistical unit is bird x syllable/cluster label, summarized to bird medians.
Raw syllable/segment values are used for distribution visualization and per-label effect-size calculation.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PRE_PERIODS = ["early_pre", "late_pre"]
POST_PERIODS = ["early_post", "late_post"]

PRETTY_FEATURE = {
    "mean_wiener_entropy": "Mean Wiener entropy",
    "median_wiener_entropy": "Median Wiener entropy",
    "q75_wiener_entropy": "75th percentile Wiener entropy",
    "q90_wiener_entropy": "90th percentile Wiener entropy",
    "q95_wiener_entropy": "95th percentile Wiener entropy",
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative (kHz/s)",
    "median_pitch_derivative_khz_per_s": "Median pitch derivative (kHz/s)",
    "q75_pitch_derivative_khz_per_s": "75th percentile pitch derivative (kHz/s)",
    "q90_pitch_derivative_khz_per_s": "90th percentile pitch derivative (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative (kHz/s)",
    "spectrogram_template_correlation": "Spectrogram correlation to late-pre template",
    "within_condition_pairwise_correlation": "Within-condition pairwise spectrogram correlation",
}

XLABEL = {
    "mean_wiener_entropy": "Mean Wiener entropy per segmented syllable",
    "q95_wiener_entropy": "95th percentile Wiener entropy per segmented syllable",
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative per segmented syllable (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative per segmented syllable (kHz/s)",
    "spectrogram_template_correlation": "Spectrogram correlation to late-pre template",
    "within_condition_pairwise_correlation": "Within-condition pairwise spectrogram correlation",
}


def clean_values(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def ecdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(clean_values(values))
    if x.size == 0:
        return x, np.array([], dtype=float)
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def parse_animal_label_from_name(path: Path, suffix: str) -> Tuple[str, str]:
    # Handles names like USA5288_label17_per_segment_feature_summary_all_periods.csv
    m = re.search(r"(?P<animal>.+?)_label(?P<label>.+?)_" + re.escape(suffix) + r"$", path.name)
    if m:
        return m.group("animal"), m.group("label")
    # Fallback: use parent folders.
    animal = path.parent.parent.name if path.parent.parent.name else "unknown"
    label = path.parent.name.split("_label")[-1] if "_label" in path.parent.name else "unknown"
    return animal, label


def period_values_from_df(df: pd.DataFrame, feature: str, inliers_only: bool) -> Dict[str, np.ndarray]:
    use = df.copy()
    if inliers_only and "pitch_contour_inlier" in use.columns:
        col = use["pitch_contour_inlier"]
        if col.dtype == bool:
            mask = col
        else:
            mask = col.astype(str).str.lower().isin(["true", "1", "yes"])
        use = use.loc[mask]
    out = {p: np.array([], dtype=float) for p in PERIOD_ORDER}
    if "period" not in use.columns or feature not in use.columns:
        return out
    for period in PERIOD_ORDER:
        out[period] = clean_values(use.loc[use["period"] == period, feature].to_numpy(dtype=float))
    return out


def combine_period_values(values: Dict[str, np.ndarray], periods: List[str]) -> np.ndarray:
    arrays = [clean_values(values.get(p, [])) for p in periods]
    arrays = [a for a in arrays if a.size > 0]
    return np.concatenate(arrays) if arrays else np.array([], dtype=float)


def distance_between(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    x = clean_values(x)
    y = clean_values(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    if metric == "wasserstein":
        if SCIPY_AVAILABLE:
            return float(stats.wasserstein_distance(x, y))
        # fallback: average quantile distance
        n = max(x.size, y.size)
        q = np.linspace(0, 1, n)
        return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))
    if metric == "ks":
        if SCIPY_AVAILABLE:
            return float(stats.ks_2samp(x, y, alternative="two-sided", mode="auto").statistic)
        vals = np.sort(np.unique(np.concatenate([x, y])))
        sx = np.sort(x)
        sy = np.sort(y)
        fx = np.searchsorted(sx, vals, side="right") / sx.size
        fy = np.searchsorted(sy, vals, side="right") / sy.size
        return float(np.max(np.abs(fx - fy)))
    if metric == "delta_median_abs":
        return float(abs(np.nanmedian(y) - np.nanmedian(x)))
    raise ValueError(f"Unknown distance metric: {metric}")


def stats_for_pre_post(pre: np.ndarray, post: np.ndarray) -> Dict[str, float]:
    pre = clean_values(pre)
    post = clean_values(post)
    row = {
        "n_pre": int(pre.size),
        "n_post": int(post.size),
        "median_pre": float(np.nanmedian(pre)) if pre.size else np.nan,
        "median_post": float(np.nanmedian(post)) if post.size else np.nan,
        "mean_pre": float(np.nanmean(pre)) if pre.size else np.nan,
        "mean_post": float(np.nanmean(post)) if post.size else np.nan,
        "delta_median_post_minus_pre": float(np.nanmedian(post) - np.nanmedian(pre)) if pre.size and post.size else np.nan,
        "delta_mean_post_minus_pre": float(np.nanmean(post) - np.nanmean(pre)) if pre.size and post.size else np.nan,
        "q90_pre": float(np.nanpercentile(pre, 90)) if pre.size else np.nan,
        "q90_post": float(np.nanpercentile(post, 90)) if post.size else np.nan,
        "delta_q90_post_minus_pre": float(np.nanpercentile(post, 90) - np.nanpercentile(pre, 90)) if pre.size and post.size else np.nan,
        "q95_pre": float(np.nanpercentile(pre, 95)) if pre.size else np.nan,
        "q95_post": float(np.nanpercentile(post, 95)) if post.size else np.nan,
        "delta_q95_post_minus_pre": float(np.nanpercentile(post, 95) - np.nanpercentile(pre, 95)) if pre.size and post.size else np.nan,
        "ks_statistic": np.nan,
        "ks_p_value": np.nan,
        "mannwhitney_p_value": np.nan,
        "wasserstein_distance": np.nan,
    }
    if SCIPY_AVAILABLE and pre.size > 0 and post.size > 0:
        try:
            ks = stats.ks_2samp(pre, post, alternative="two-sided", mode="auto")
            mw = stats.mannwhitneyu(pre, post, alternative="two-sided", method="auto")
            row["ks_statistic"] = float(ks.statistic)
            row["ks_p_value"] = float(ks.pvalue)
            row["mannwhitney_p_value"] = float(mw.pvalue)
            row["wasserstein_distance"] = float(stats.wasserstein_distance(pre, post))
        except Exception:
            pass
    return row


def add_label_row(rows: List[dict], raw_store: Dict[str, Dict[str, List[np.ndarray]]],
                  animal_id: str, label: str, feature: str, source: str,
                  values_by_period: Dict[str, np.ndarray], distance_metric: str,
                  lesion_post_period: str, min_values_per_group: int):
    early_pre = clean_values(values_by_period.get("early_pre", []))
    late_pre = clean_values(values_by_period.get("late_pre", []))
    early_post = clean_values(values_by_period.get("early_post", []))
    late_post = clean_values(values_by_period.get("late_post", []))
    pre = combine_period_values(values_by_period, PRE_PERIODS)
    post = combine_period_values(values_by_period, POST_PERIODS)
    lesion_post = {
        "combined_post": post,
        "early_post": early_post,
        "late_post": late_post,
    }[lesion_post_period]

    if pre.size < min_values_per_group or post.size < min_values_per_group:
        return
    if early_pre.size < min_values_per_group or late_pre.size < min_values_per_group or lesion_post.size < min_values_per_group:
        return

    row = {
        "animal_id": str(animal_id),
        "label": str(label),
        "feature": feature,
        "pretty_feature": PRETTY_FEATURE.get(feature, feature),
        "source": source,
        "n_early_pre": int(early_pre.size),
        "n_late_pre": int(late_pre.size),
        "n_early_post": int(early_post.size),
        "n_late_post": int(late_post.size),
        "baseline_distance": distance_between(early_pre, late_pre, distance_metric),
        "lesion_distance": distance_between(late_pre, lesion_post, distance_metric),
        "distance_metric": distance_metric,
        "lesion_post_period": lesion_post_period,
    }
    row["lesion_minus_baseline_distance"] = row["lesion_distance"] - row["baseline_distance"]
    row.update(stats_for_pre_post(pre, post))
    rows.append(row)

    raw_store.setdefault(feature, {"pre": [], "post": []})
    raw_store[feature]["pre"].append(pre)
    raw_store[feature]["post"].append(post)


def load_acoustic_rows(root: Path, features: List[str], inliers_only: bool,
                       distance_metric: str, lesion_post_period: str,
                       min_values_per_group: int,
                       raw_store: Dict[str, Dict[str, List[np.ndarray]]]) -> List[dict]:
    rows: List[dict] = []
    files = sorted(root.rglob("*_per_segment_feature_summary_all_periods.csv"))
    print(f"[INFO] Found {len(files)} acoustic feature summary files under {root}")
    for path in files:
        animal, label = parse_animal_label_from_name(path, "per_segment_feature_summary_all_periods.csv")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        if df.empty:
            continue
        if "animal_id" in df.columns and df["animal_id"].notna().any():
            animal = str(df["animal_id"].dropna().iloc[0])
        if "label" in df.columns and df["label"].notna().any():
            label = str(df["label"].dropna().iloc[0])
        for feature in features:
            if feature not in df.columns:
                continue
            vals = period_values_from_df(df, feature, inliers_only)
            add_label_row(rows, raw_store, animal, label, feature, "entropy_pitch", vals,
                          distance_metric, lesion_post_period, min_values_per_group)
    return rows


def load_crosscorr_rows(root: Path, distance_metric: str, lesion_post_period: str,
                        min_values_per_group: int,
                        raw_store: Dict[str, Dict[str, List[np.ndarray]]]) -> List[dict]:
    rows: List[dict] = []

    # Template correlation to late-pre template: one value per segmented syllable.
    seg_files = sorted(root.rglob("*_spectrogram_crosscorr_segments.csv"))
    print(f"[INFO] Found {len(seg_files)} spectrogram cross-correlation segment files under {root}")
    for path in seg_files:
        animal, label = parse_animal_label_from_name(path, "spectrogram_crosscorr_segments.csv")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        if df.empty or "corr_late_pre_template" not in df.columns or "period" not in df.columns:
            continue
        if "animal_id" in df.columns and df["animal_id"].notna().any():
            animal = str(df["animal_id"].dropna().iloc[0])
        if "label" in df.columns and df["label"].notna().any():
            label = str(df["label"].dropna().iloc[0])
        vals = {p: clean_values(df.loc[df["period"] == p, "corr_late_pre_template"].to_numpy(dtype=float)) for p in PERIOD_ORDER}
        add_label_row(rows, raw_store, animal, label, "spectrogram_template_correlation", "spectrogram_crosscorr", vals,
                      distance_metric, lesion_post_period, min_values_per_group)

    # Within-condition pairwise correlation: one value per pair sampled by the per-label script.
    pair_files = sorted(root.rglob("*_pairwise_correlation_values.csv"))
    print(f"[INFO] Found {len(pair_files)} pairwise correlation value files under {root}")
    for path in pair_files:
        animal, label = parse_animal_label_from_name(path, "pairwise_correlation_values.csv")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue
        if df.empty or "group" not in df.columns or "pairwise_spectrogram_correlation" not in df.columns:
            continue
        vals = {
            "early_pre": clean_values(df.loc[df["group"] == "early_pre", "pairwise_spectrogram_correlation"].to_numpy(dtype=float)),
            "late_pre": clean_values(df.loc[df["group"] == "late_pre", "pairwise_spectrogram_correlation"].to_numpy(dtype=float)),
            "early_post": clean_values(df.loc[df["group"] == "early_post", "pairwise_spectrogram_correlation"].to_numpy(dtype=float)),
            "late_post": clean_values(df.loc[df["group"] == "late_post", "pairwise_spectrogram_correlation"].to_numpy(dtype=float)),
        }
        add_label_row(rows, raw_store, animal, label, "within_condition_pairwise_correlation", "spectrogram_crosscorr", vals,
                      distance_metric, lesion_post_period, min_values_per_group)
    return rows


def bird_level_summary(per_label: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bird_rows = []
    stat_rows = []
    for feature, subf in per_label.groupby("feature"):
        bird_vals = []
        bird_lesion_minus_base = []
        for animal, sub in subf.groupby("animal_id"):
            r = {
                "animal_id": animal,
                "feature": feature,
                "pretty_feature": PRETTY_FEATURE.get(feature, feature),
                "source": sub["source"].iloc[0] if "source" in sub else "unknown",
                "n_labels": int(sub.shape[0]),
                "bird_median_delta_median_post_minus_pre": float(np.nanmedian(sub["delta_median_post_minus_pre"])),
                "bird_median_delta_q95_post_minus_pre": float(np.nanmedian(sub["delta_q95_post_minus_pre"])),
                "bird_median_baseline_distance": float(np.nanmedian(sub["baseline_distance"])),
                "bird_median_lesion_distance": float(np.nanmedian(sub["lesion_distance"])),
                "bird_median_lesion_minus_baseline_distance": float(np.nanmedian(sub["lesion_minus_baseline_distance"])),
            }
            bird_rows.append(r)
            bird_vals.append(r["bird_median_delta_median_post_minus_pre"])
            bird_lesion_minus_base.append(r["bird_median_lesion_minus_baseline_distance"])
        bird_vals = clean_values(bird_vals)
        bird_lesion_minus_base = clean_values(bird_lesion_minus_base)
        s = {
            "feature": feature,
            "pretty_feature": PRETTY_FEATURE.get(feature, feature),
            "n_birds": int(subf["animal_id"].nunique()),
            "median_bird_effect": float(np.nanmedian(bird_vals)) if bird_vals.size else np.nan,
            "median_lesion_minus_baseline": float(np.nanmedian(bird_lesion_minus_base)) if bird_lesion_minus_base.size else np.nan,
            "wilcoxon_effect_vs_zero_p": np.nan,
            "wilcoxon_lesion_minus_baseline_gt_zero_p": np.nan,
            "scipy_available": SCIPY_AVAILABLE,
        }
        if SCIPY_AVAILABLE and bird_vals.size > 0:
            try:
                s["wilcoxon_effect_vs_zero_p"] = float(stats.wilcoxon(bird_vals, alternative="two-sided", zero_method="wilcox").pvalue)
            except Exception:
                pass
        if SCIPY_AVAILABLE and bird_lesion_minus_base.size > 0:
            try:
                s["wilcoxon_lesion_minus_baseline_gt_zero_p"] = float(stats.wilcoxon(bird_lesion_minus_base, alternative="greater", zero_method="wilcox").pvalue)
            except Exception:
                pass
        stat_rows.append(s)
    return pd.DataFrame(bird_rows), pd.DataFrame(stat_rows)


def plot_pooled_ecdf(raw_store: Dict[str, Dict[str, List[np.ndarray]]], feature: str, out_png: Path, title_prefix: str):
    if feature not in raw_store:
        return
    pre = np.concatenate(raw_store[feature]["pre"]) if raw_store[feature]["pre"] else np.array([], dtype=float)
    post = np.concatenate(raw_store[feature]["post"]) if raw_store[feature]["post"] else np.array([], dtype=float)
    pre_x, pre_y = ecdf_xy(pre)
    post_x, post_y = ecdf_xy(post)
    if pre_x.size == 0 and post_x.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if pre_x.size:
        ax.step(pre_x, pre_y, where="post", lw=2.2, label=f"Pre-lesion (N={pre_x.size:,})")
    if post_x.size:
        ax.step(post_x, post_y, where="post", lw=2.2, label=f"Post-lesion (N={post_x.size:,})")
    st = stats_for_pre_post(pre, post)
    ax.set_title(
        f"{title_prefix} — ECDF pre vs post\n{PRETTY_FEATURE.get(feature, feature)}; "
        f"Δmedian={st['delta_median_post_minus_pre']:.3g}; KS={st['ks_statistic']:.3g}"
    )
    ax.set_xlabel(XLABEL.get(feature, PRETTY_FEATURE.get(feature, feature)))
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_bird_level(per_label: pd.DataFrame, bird_df: pd.DataFrame, feature: str, out_png: Path, title_prefix: str):
    sub = per_label.loc[per_label["feature"] == feature].copy()
    birds = bird_df.loc[bird_df["feature"] == feature].copy()
    if sub.empty or birds.empty:
        return
    birds = birds.sort_values("animal_id")
    bird_names = birds["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(bird_names)}
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for _, row in sub.iterrows():
        x = xmap.get(row["animal_id"])
        if x is None:
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature)) % 1000) / 1000.0 - 0.5) * 0.24
        ax.scatter(x + jitter, row["delta_median_post_minus_pre"], s=32, alpha=0.32)
    xs = np.arange(len(bird_names))
    ys = birds["bird_median_delta_median_post_minus_pre"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=105, marker="D", label="Bird median across selected labels")
    ax.axhline(0, linestyle="--", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(bird_names, rotation=30, ha="right")
    ax.set_ylabel("Post - pre effect\n(median value per bird × label)")
    ax.set_title(f"{title_prefix} — bird-level post-pre effects\n{PRETTY_FEATURE.get(feature, feature)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_baseline_vs_lesion(per_label: pd.DataFrame, bird_df: pd.DataFrame, feature: str, out_png: Path,
                            title_prefix: str, distance_metric: str, lesion_post_period: str):
    sub = per_label.loc[per_label["feature"] == feature].copy()
    birds = bird_df.loc[bird_df["feature"] == feature].copy()
    if sub.empty or birds.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    bird_names = sorted(sub["animal_id"].unique())
    cmap = plt.cm.get_cmap("tab10", max(1, len(bird_names)))
    colors = {b: cmap(i) for i, b in enumerate(bird_names)}
    xvals, yvals = [], []
    for _, row in sub.iterrows():
        x = row["baseline_distance"]
        y = row["lesion_distance"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=36, alpha=0.35, color=colors[row["animal_id"]])
    for _, row in birds.iterrows():
        x = row["bird_median_baseline_distance"]
        y = row["bird_median_lesion_distance"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=105, marker="D", color=colors.get(row["animal_id"], "black"), edgecolors="black")
        ax.text(x, y, row["animal_id"], fontsize=8, ha="left", va="bottom")
    if not xvals or not yvals:
        plt.close(fig)
        return
    data_min = min(min(xvals), min(yvals))
    data_max = max(max(xvals), max(yvals))
    span = data_max - data_min
    pad = max(0.05 * span, 0.02 * max(abs(data_min), abs(data_max), 1.0))
    lo = data_min - pad
    hi = data_max + pad
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], linestyle="--", lw=1.2, label="equal change")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    post_name = {"combined_post": "combined post", "early_post": "early post", "late_post": "late post"}[lesion_post_period]
    ax.set_xlabel(f"Baseline drift: distance(early pre, late pre)\n[{distance_metric}]")
    ax.set_ylabel(f"Lesion-associated shift: distance(late pre, {post_name})\n[{distance_metric}]")
    ax.set_title(f"{title_prefix} — baseline drift vs lesion-associated shift\n{PRETTY_FEATURE.get(feature, feature)}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_template_vs_pairwise(per_label: pd.DataFrame, out_png: Path, title_prefix: str):
    needed = ["spectrogram_template_correlation", "within_condition_pairwise_correlation"]
    if not all(f in set(per_label["feature"]) for f in needed):
        return
    a = per_label.loc[per_label["feature"] == needed[0], ["animal_id", "label", "delta_median_post_minus_pre"]].copy()
    b = per_label.loc[per_label["feature"] == needed[1], ["animal_id", "label", "delta_median_post_minus_pre"]].copy()
    merged = a.merge(b, on=["animal_id", "label"], suffixes=("_template", "_pairwise"))
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    ax.scatter(merged["delta_median_post_minus_pre_template"], merged["delta_median_post_minus_pre_pairwise"], s=65, alpha=0.8)
    for _, row in merged.iterrows():
        ax.text(row["delta_median_post_minus_pre_template"], row["delta_median_post_minus_pre_pairwise"],
                f"{row['animal_id']}:{row['label']}", fontsize=8, ha="left", va="bottom")
    ax.axhline(0, linestyle="--", lw=1.2)
    ax.axvline(0, linestyle="--", lw=1.2)
    ax.set_xlabel("Δ median correlation to late-pre template\n(post - pre)")
    ax.set_ylabel("Δ median within-condition pairwise correlation\n(post - pre)")
    ax.set_title(f"{title_prefix}\nTemplate shift vs within-condition stereotypy change")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate higher-level birdsong acoustic comparison figures.")
    p.add_argument("--acoustic-batch-root", default=None,
                   help="Batch root containing *_per_segment_feature_summary_all_periods.csv files from entropy/pitch pipeline.")
    p.add_argument("--crosscorr-batch-root", default=None,
                   help="Batch/root folder containing spectrogram cross-correlation outputs.")
    p.add_argument("--out-dir", required=True, help="Output folder for aggregate figures/tables.")
    p.add_argument("--acoustic-features", default="mean_wiener_entropy,q95_wiener_entropy,mean_pitch_derivative_khz_per_s,q95_pitch_derivative_khz_per_s",
                   help="Comma-separated acoustic per-segment columns to aggregate.")
    p.add_argument("--use-inliers-only", action="store_true", default=True,
                   help="Use only pitch_contour_inlier rows when available. Default on.")
    p.add_argument("--include-all-segments", dest="use_inliers_only", action="store_false",
                   help="Disable pitch-contour inlier filtering for the acoustic summaries.")
    p.add_argument("--distance-metric", default="wasserstein", choices=["wasserstein", "ks", "delta_median_abs"],
                   help="Distance metric for baseline-vs-lesion plots.")
    p.add_argument("--lesion-post-period", default="combined_post", choices=["combined_post", "early_post", "late_post"],
                   help="Post-lesion period pool used for lesion-associated shift.")
    p.add_argument("--min-values-per-group", type=int, default=5,
                   help="Minimum observations in each group for per-label comparisons.")
    p.add_argument("--title-prefix", default="Medial + Lateral birds",
                   help="Prefix used in plot titles.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_store: Dict[str, Dict[str, List[np.ndarray]]] = {}
    per_label_rows: List[dict] = []

    if args.acoustic_batch_root:
        acoustic_root = Path(args.acoustic_batch_root).expanduser().resolve()
        acoustic_features = [x.strip() for x in args.acoustic_features.split(",") if x.strip()]
        per_label_rows.extend(load_acoustic_rows(
            acoustic_root, acoustic_features, args.use_inliers_only,
            args.distance_metric, args.lesion_post_period, args.min_values_per_group,
            raw_store
        ))

    if args.crosscorr_batch_root:
        cross_root = Path(args.crosscorr_batch_root).expanduser().resolve()
        per_label_rows.extend(load_crosscorr_rows(
            cross_root, args.distance_metric, args.lesion_post_period,
            args.min_values_per_group, raw_store
        ))

    if not per_label_rows:
        raise RuntimeError("No valid per-label rows found. Check input roots and file patterns.")

    per_label = pd.DataFrame(per_label_rows)
    per_label_out = out_dir / "bird_label_acoustic_and_similarity_effects.csv"
    per_label.to_csv(per_label_out, index=False)
    print(f"[SAVED] {per_label_out}")

    bird_df, stat_df = bird_level_summary(per_label)
    bird_out = out_dir / "bird_level_acoustic_and_similarity_summary.csv"
    stat_out = out_dir / "bird_level_acoustic_and_similarity_stats.csv"
    bird_df.to_csv(bird_out, index=False)
    stat_df.to_csv(stat_out, index=False)
    print(f"[SAVED] {bird_out}")
    print(f"[SAVED] {stat_out}")

    features = list(per_label["feature"].drop_duplicates())
    for feature in features:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        plot_pooled_ecdf(raw_store, feature, out_dir / f"ecdf_pre_post_{safe}.png", args.title_prefix)
        plot_bird_level(per_label, bird_df, feature, out_dir / f"bird_level_summary_{safe}.png", args.title_prefix)
        plot_baseline_vs_lesion(per_label, bird_df, feature, out_dir / f"baseline_vs_lesion_{safe}.png",
                                args.title_prefix, args.distance_metric, args.lesion_post_period)

    plot_template_vs_pairwise(per_label, out_dir / "template_shift_vs_pairwise_stereotypy_change.png", args.title_prefix)

    readme = out_dir / "README_v33.txt"
    readme.write_text(
        "Higher-level acoustic comparison outputs.\n\n"
        "Main CSVs:\n"
        "- bird_label_acoustic_and_similarity_effects.csv: one row per bird x label x feature.\n"
        "- bird_level_acoustic_and_similarity_summary.csv: bird medians across selected labels.\n"
        "- bird_level_acoustic_and_similarity_stats.csv: across-bird Wilcoxon summaries.\n\n"
        "Main figures:\n"
        "- ecdf_pre_post_<feature>.png: pooled pre/post ECDFs.\n"
        "- bird_level_summary_<feature>.png: one point per label plus bird median diamonds.\n"
        "- baseline_vs_lesion_<feature>.png: baseline drift vs lesion-associated shift.\n"
        "- template_shift_vs_pairwise_stereotypy_change.png: cross-correlation-specific quadrant plot.\n\n"
        "Interpretation notes:\n"
        "For entropy/pitch derivative, positive post-pre effect means the feature increased after lesion.\n"
        "For spectrogram correlations, negative post-pre effect means reduced similarity/stereotypy after lesion.\n"
        "For baseline-vs-lesion plots, points above the diagonal mean lesion-associated shift exceeds baseline pre-lesion drift.\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE] Completed higher-level acoustic comparison figures.")


if __name__ == "__main__":
    main()
