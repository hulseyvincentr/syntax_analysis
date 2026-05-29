#!/usr/bin/env python3
"""
Aggregate per-label pitch/entropy analysis outputs into higher-level plots and statistics.

Designed to read output folders produced by
cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py
(or compatible descendants) and create:

1) Bird-level summary plots of post-pre acoustic effects across selected labels.
2) Lesion-associated shift vs baseline drift plots.
3) CSV tables of per-label and per-bird effect sizes.
4) Basic across-bird statistical summaries.

Default features:
- mean_wiener_entropy
- q95_wiener_entropy
- mean_pitch_derivative_khz_per_s
- q95_pitch_derivative_khz_per_s
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
    "segment_duration_s": "Syllable duration (s)",
}


def clean_values(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-label acoustic outputs into higher-level bird summary plots and lesion-vs-baseline analyses."
    )
    p.add_argument("--batch-out-root", required=True,
                   help="Root output folder from the batch run (e.g. .../pitch_entropy_high_phrase_variance_ML_birds_v30)")
    p.add_argument("--out-dir", default=None,
                   help="Directory for aggregated outputs. Default: <batch-out-root>/aggregate_v31")
    p.add_argument("--features", default="mean_wiener_entropy,q95_wiener_entropy,mean_pitch_derivative_khz_per_s,q95_pitch_derivative_khz_per_s",
                   help="Comma-separated per-segment summary columns to analyze.")
    p.add_argument("--use-inliers-only", action="store_true", default=True,
                   help="Use only segments with pitch_contour_inlier=True when available. Default: on.")
    p.add_argument("--include-all-segments", dest="use_inliers_only", action="store_false",
                   help="Disable pitch-contour inlier filtering in the aggregation step.")
    p.add_argument("--lesion-post-period", default="combined_post", choices=["combined_post", "early_post", "late_post"],
                   help="Which post-lesion pool to compare against late_pre for lesion-associated shift.")
    p.add_argument("--distance-metric", default="wasserstein", choices=["wasserstein", "ks", "delta_median_abs"],
                   help="Distance metric for baseline-vs-lesion plots.")
    p.add_argument("--min-values-per-group", type=int, default=5,
                   help="Minimum number of segments required in each comparison group.")
    p.add_argument("--selected-labels-csv", default=None,
                   help="Optional path to selected_high_phrase_duration_variance_labels.csv. If omitted, the script will use the copy inside batch-out-root when present.")
    p.add_argument("--title-prefix", default="Medial + Lateral birds",
                   help="Prefix for figure titles.")
    return p.parse_args()


def find_segment_summary_files(batch_root: Path) -> List[Path]:
    return sorted(batch_root.rglob("*_per_segment_feature_summary_all_periods.csv"))


def parse_animal_label_from_path(csv_path: Path) -> Tuple[str, str]:
    m = re.search(r"(?P<animal>[^/\\]+)_label(?P<label>.+?)_per_segment_feature_summary_all_periods\.csv$", csv_path.name)
    if m:
        return m.group("animal"), m.group("label")
    # fallback to folder names
    parts = csv_path.parts
    animal = parts[-3] if len(parts) >= 3 else "unknown"
    label = parts[-2].split("_label")[-1] if "_label" in parts[-2] else "unknown"
    return animal, label


def distance_between(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    x = clean_values(x)
    y = clean_values(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    if metric == "wasserstein":
        if SCIPY_AVAILABLE:
            return float(stats.wasserstein_distance(x, y))
        # simple fallback approximation
        xs = np.sort(x)
        ys = np.sort(y)
        n = max(xs.size, ys.size)
        q = np.linspace(0, 1, n)
        xq = np.quantile(xs, q)
        yq = np.quantile(ys, q)
        return float(np.mean(np.abs(xq - yq)))
    if metric == "ks":
        if SCIPY_AVAILABLE:
            return float(stats.ks_2samp(x, y, alternative="two-sided", mode="auto").statistic)
        # fallback ECDF max difference
        vals = np.sort(np.unique(np.concatenate([x, y])))
        fx = np.searchsorted(np.sort(x), vals, side="right") / x.size
        fy = np.searchsorted(np.sort(y), vals, side="right") / y.size
        return float(np.max(np.abs(fx - fy)))
    if metric == "delta_median_abs":
        return float(abs(np.nanmedian(y) - np.nanmedian(x)))
    raise ValueError(f"Unknown distance metric: {metric}")


def get_period_values(df: pd.DataFrame, feature: str, use_inliers_only: bool) -> Dict[str, np.ndarray]:
    out = {p: np.array([], dtype=float) for p in PERIOD_ORDER}
    use = df.copy()
    if use_inliers_only and "pitch_contour_inlier" in use.columns:
        vals = use["pitch_contour_inlier"]
        # tolerate bool / string / numeric
        if vals.dtype == bool:
            mask = vals
        else:
            mask = vals.astype(str).str.lower().isin(["true", "1", "yes"])
        use = use.loc[mask]
    for p in PERIOD_ORDER:
        sub = use.loc[use["period"] == p, feature] if feature in use.columns else pd.Series([], dtype=float)
        out[p] = clean_values(sub.to_numpy(dtype=float)) if len(sub) else np.array([], dtype=float)
    return out


def compute_label_feature_row(df: pd.DataFrame, animal_id: str, label: str, feature: str,
                              distance_metric: str, lesion_post_period: str,
                              min_values_per_group: int, use_inliers_only: bool) -> Dict[str, object] | None:
    vals = get_period_values(df, feature, use_inliers_only)
    ep = vals["early_pre"]
    lp = vals["late_pre"]
    epost = vals["early_post"]
    lpost = vals["late_post"]
    pre = np.concatenate([ep, lp]) if ep.size + lp.size > 0 else np.array([], dtype=float)
    post = np.concatenate([epost, lpost]) if epost.size + lpost.size > 0 else np.array([], dtype=float)
    lesion_post = {
        "combined_post": post,
        "early_post": epost,
        "late_post": lpost,
    }[lesion_post_period]

    if pre.size < min_values_per_group or post.size < min_values_per_group:
        return None
    if ep.size < min_values_per_group or lp.size < min_values_per_group or lesion_post.size < min_values_per_group:
        return None

    row = {
        "animal_id": animal_id,
        "label": label,
        "feature": feature,
        "pretty_feature": PRETTY_FEATURE.get(feature, feature),
        "n_early_pre": int(ep.size),
        "n_late_pre": int(lp.size),
        "n_early_post": int(epost.size),
        "n_late_post": int(lpost.size),
        "n_pre_combined": int(pre.size),
        "n_post_combined": int(post.size),
        "median_pre": float(np.nanmedian(pre)),
        "median_post": float(np.nanmedian(post)),
        "mean_pre": float(np.nanmean(pre)),
        "mean_post": float(np.nanmean(post)),
        "delta_median_post_minus_pre": float(np.nanmedian(post) - np.nanmedian(pre)),
        "delta_mean_post_minus_pre": float(np.nanmean(post) - np.nanmean(pre)),
        "q75_pre": float(np.nanpercentile(pre, 75)),
        "q75_post": float(np.nanpercentile(post, 75)),
        "q90_pre": float(np.nanpercentile(pre, 90)),
        "q90_post": float(np.nanpercentile(post, 90)),
        "q95_pre": float(np.nanpercentile(pre, 95)),
        "q95_post": float(np.nanpercentile(post, 95)),
        "delta_q75_post_minus_pre": float(np.nanpercentile(post, 75) - np.nanpercentile(pre, 75)),
        "delta_q90_post_minus_pre": float(np.nanpercentile(post, 90) - np.nanpercentile(pre, 90)),
        "delta_q95_post_minus_pre": float(np.nanpercentile(post, 95) - np.nanpercentile(pre, 95)),
        "pre_post_distance": float(distance_between(pre, post, distance_metric)),
        "baseline_distance": float(distance_between(ep, lp, distance_metric)),
        "lesion_distance": float(distance_between(lp, lesion_post, distance_metric)),
        "lesion_minus_baseline_distance": float(distance_between(lp, lesion_post, distance_metric) - distance_between(ep, lp, distance_metric)),
        "distance_metric": distance_metric,
        "lesion_post_period": lesion_post_period,
        "ks_statistic_pre_post": np.nan,
        "ks_p_value_pre_post": np.nan,
        "mannwhitney_p_value_pre_post": np.nan,
        "wasserstein_distance_pre_post": np.nan,
    }

    if SCIPY_AVAILABLE:
        ks = stats.ks_2samp(pre, post, alternative="two-sided", mode="auto")
        mw = stats.mannwhitneyu(pre, post, alternative="two-sided", method="auto")
        row["ks_statistic_pre_post"] = float(ks.statistic)
        row["ks_p_value_pre_post"] = float(ks.pvalue)
        row["mannwhitney_p_value_pre_post"] = float(mw.pvalue)
        row["wasserstein_distance_pre_post"] = float(stats.wasserstein_distance(pre, post))

    return row


def holm_adjust(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return adjusted
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, original_idx in enumerate(order, start=1):
        adj = (m - rank + 1) * p[original_idx]
        running = max(running, adj)
        adjusted[original_idx] = min(running, 1.0)
    return adjusted


def summarize_by_bird(per_label_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stat_rows = []
    for feature, subf in per_label_df.groupby("feature"):
        bird_values = []
        bird_baselines = []
        bird_lesions = []
        for animal, sub in subf.groupby("animal_id"):
            r = {
                "animal_id": animal,
                "feature": feature,
                "pretty_feature": PRETTY_FEATURE.get(feature, feature),
                "n_labels": int(sub.shape[0]),
                "median_delta_median_post_minus_pre": float(np.nanmedian(sub["delta_median_post_minus_pre"])),
                "median_delta_q95_post_minus_pre": float(np.nanmedian(sub["delta_q95_post_minus_pre"])),
                "median_pre_post_distance": float(np.nanmedian(sub["pre_post_distance"])),
                "median_baseline_distance": float(np.nanmedian(sub["baseline_distance"])),
                "median_lesion_distance": float(np.nanmedian(sub["lesion_distance"])),
                "median_lesion_minus_baseline_distance": float(np.nanmedian(sub["lesion_minus_baseline_distance"])),
                "mean_delta_median_post_minus_pre": float(np.nanmean(sub["delta_median_post_minus_pre"])),
                "mean_lesion_minus_baseline_distance": float(np.nanmean(sub["lesion_minus_baseline_distance"])),
            }
            rows.append(r)
            bird_values.append(r["median_delta_median_post_minus_pre"])
            bird_baselines.append(r["median_baseline_distance"])
            bird_lesions.append(r["median_lesion_distance"])

        bird_values = clean_values(bird_values)
        bird_baselines = clean_values(bird_baselines)
        bird_lesions = clean_values(bird_lesions)
        srow = {
            "feature": feature,
            "pretty_feature": PRETTY_FEATURE.get(feature, feature),
            "n_birds": int(len(subf["animal_id"].unique())),
            "median_of_bird_median_effects": float(np.nanmedian(bird_values)) if bird_values.size else np.nan,
            "median_of_bird_baseline_distances": float(np.nanmedian(bird_baselines)) if bird_baselines.size else np.nan,
            "median_of_bird_lesion_distances": float(np.nanmedian(bird_lesions)) if bird_lesions.size else np.nan,
            "wilcoxon_effect_vs_zero_p": np.nan,
            "wilcoxon_effect_vs_zero_stat": np.nan,
            "wilcoxon_lesion_gt_baseline_p": np.nan,
            "wilcoxon_lesion_gt_baseline_stat": np.nan,
            "scipy_available": SCIPY_AVAILABLE,
        }
        if SCIPY_AVAILABLE and bird_values.size > 0:
            try:
                w = stats.wilcoxon(bird_values, alternative="two-sided", zero_method="wilcox")
                srow["wilcoxon_effect_vs_zero_p"] = float(w.pvalue)
                srow["wilcoxon_effect_vs_zero_stat"] = float(w.statistic)
            except Exception:
                pass
        if SCIPY_AVAILABLE and bird_baselines.size > 0 and bird_lesions.size > 0 and bird_baselines.size == bird_lesions.size:
            try:
                w = stats.wilcoxon(bird_lesions, bird_baselines, alternative="greater", zero_method="wilcox")
                srow["wilcoxon_lesion_gt_baseline_p"] = float(w.pvalue)
                srow["wilcoxon_lesion_gt_baseline_stat"] = float(w.statistic)
            except Exception:
                pass
        stat_rows.append(srow)

    bird_df = pd.DataFrame(rows)
    stat_df = pd.DataFrame(stat_rows)
    if not stat_df.empty:
        for col in ["wilcoxon_effect_vs_zero_p", "wilcoxon_lesion_gt_baseline_p"]:
            stat_df[f"{col}_holm"] = holm_adjust(stat_df[col].tolist())
    return bird_df, stat_df


def scatter_bird_summary(per_label_df: pd.DataFrame, bird_df: pd.DataFrame, feature: str, out_png: Path, title_prefix: str):
    sub_labels = per_label_df.loc[per_label_df["feature"] == feature].copy()
    sub_birds = bird_df.loc[bird_df["feature"] == feature].copy()
    if sub_birds.empty:
        return
    sub_birds = sub_birds.sort_values("animal_id")
    birds = sub_birds["animal_id"].tolist()
    x_map = {b: i for i, b in enumerate(birds)}

    fig = plt.figure(figsize=(10, 6))
    # light individual label effects
    for _, row in sub_labels.iterrows():
        x = x_map.get(row["animal_id"], None)
        if x is None or not np.isfinite(row["delta_median_post_minus_pre"]):
            continue
        jitter = (hash((row["animal_id"], str(row["label"]))) % 1000) / 1000.0
        jitter = (jitter - 0.5) * 0.22
        plt.scatter(x + jitter, row["delta_median_post_minus_pre"], alpha=0.25, s=28)

    # bold bird medians
    xs = np.arange(len(birds))
    ys = sub_birds["median_delta_median_post_minus_pre"].to_numpy(dtype=float)
    plt.scatter(xs, ys, s=90, marker='D', label="Bird median across selected syllables")
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.xticks(xs, birds, rotation=30, ha="right")
    plt.ylabel("Post - pre effect (median feature value)")
    plt.title(f"{title_prefix} — bird-level summary\n{PRETTY_FEATURE.get(feature, feature)}")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def scatter_baseline_vs_lesion(per_label_df: pd.DataFrame, bird_df: pd.DataFrame, feature: str, out_png: Path,
                               title_prefix: str, distance_metric: str, lesion_post_period: str):
    sub_labels = per_label_df.loc[per_label_df["feature"] == feature].copy()
    sub_birds = bird_df.loc[bird_df["feature"] == feature].copy()
    if sub_labels.empty or sub_birds.empty:
        return

    fig = plt.figure(figsize=(7.2, 6.6))
    # color per bird
    birds = sorted(sub_labels["animal_id"].unique().tolist())
    cmap = plt.cm.get_cmap("tab10", len(birds))
    b2c = {b: cmap(i) for i, b in enumerate(birds)}

    maxv = 0.0
    for _, row in sub_labels.iterrows():
        x = row["baseline_distance"]
        y = row["lesion_distance"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        maxv = max(maxv, x, y)
        plt.scatter(x, y, s=34, alpha=0.4, color=b2c[row["animal_id"]])

    # overlay bird medians
    for _, row in sub_birds.iterrows():
        x = row["median_baseline_distance"]
        y = row["median_lesion_distance"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        maxv = max(maxv, x, y)
        plt.scatter(x, y, s=95, marker='D', color=b2c.get(row["animal_id"], None), edgecolors='black')
        plt.text(x, y, row["animal_id"], fontsize=8, ha="left", va="bottom")

    lim = maxv * 1.08 if maxv > 0 else 1.0
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, label="equal change")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    post_label = {
        "combined_post": "combined post",
        "early_post": "early post",
        "late_post": "late post",
    }[lesion_post_period]
    plt.xlabel(f"Baseline drift: distance(early pre, late pre)\n[{distance_metric}]")
    plt.ylabel(f"Lesion-associated shift: distance(late pre, {post_label})\n[{distance_metric}]")
    plt.title(f"{title_prefix} — baseline drift vs lesion-associated shift\n{PRETTY_FEATURE.get(feature, feature)}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_overview_multiplot(bird_df: pd.DataFrame, out_png: Path, title_prefix: str):
    features = bird_df["feature"].drop_duplicates().tolist()
    if not features:
        return
    n = len(features)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(12, 4.8 * nrows))
    for idx, feature in enumerate(features, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        sub = bird_df.loc[bird_df["feature"] == feature].sort_values("animal_id")
        xs = np.arange(sub.shape[0])
        ys = sub["median_delta_median_post_minus_pre"].to_numpy(dtype=float)
        ax.scatter(xs, ys, s=85)
        ax.axhline(0, linestyle="--", linewidth=1.1)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["animal_id"].tolist(), rotation=30, ha="right")
        ax.set_ylabel("Post - pre effect")
        ax.set_title(PRETTY_FEATURE.get(feature, feature))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{title_prefix} — bird-level median post-pre effects", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def main():
    args = parse_args()
    batch_root = Path(args.batch_out_root).expanduser().resolve()
    if args.out_dir is None:
        out_dir = batch_root / "aggregate_v31"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    features = [x.strip() for x in args.features.split(",") if x.strip()]

    summary_files = find_segment_summary_files(batch_root)
    if not summary_files:
        raise FileNotFoundError(f"No *_per_segment_feature_summary_all_periods.csv files found under {batch_root}")

    print(f"[INFO] Found {len(summary_files)} per-label segment summary files.")

    per_label_rows: List[Dict[str, object]] = []
    file_rows: List[Dict[str, object]] = []
    for csv_path in summary_files:
        animal_id, label = parse_animal_label_from_path(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue
        if df.empty:
            continue
        if "animal_id" in df.columns and df["animal_id"].notna().any():
            animal_id = str(df["animal_id"].dropna().iloc[0])
        if "label" in df.columns and df["label"].notna().any():
            label = str(df["label"].dropna().iloc[0])
        file_rows.append({"animal_id": animal_id, "label": label, "csv_path": str(csv_path), "n_rows": int(df.shape[0])})
        for feature in features:
            if feature not in df.columns:
                print(f"[WARN] {csv_path.name} missing feature column {feature}; skipping.")
                continue
            row = compute_label_feature_row(
                df=df,
                animal_id=animal_id,
                label=label,
                feature=feature,
                distance_metric=args.distance_metric,
                lesion_post_period=args.lesion_post_period,
                min_values_per_group=args.min_values_per_group,
                use_inliers_only=args.use_inliers_only,
            )
            if row is not None:
                per_label_rows.append(row)

    if not per_label_rows:
        raise RuntimeError("No valid per-label rows were computed. Check feature names, minimum group sizes, and input folders.")

    files_df = pd.DataFrame(file_rows)
    files_df.to_csv(out_dir / "input_segment_summary_files_used.csv", index=False)
    print(f"[SAVED] {out_dir / 'input_segment_summary_files_used.csv'}")

    per_label_df = pd.DataFrame(per_label_rows)
    per_label_out = out_dir / "bird_label_acoustic_effects.csv"
    per_label_df.to_csv(per_label_out, index=False)
    print(f"[SAVED] {per_label_out}")

    bird_df, stat_df = summarize_by_bird(per_label_df)
    bird_out = out_dir / "bird_level_acoustic_summary.csv"
    stat_out = out_dir / "bird_level_acoustic_stats.csv"
    bird_df.to_csv(bird_out, index=False)
    stat_df.to_csv(stat_out, index=False)
    print(f"[SAVED] {bird_out}")
    print(f"[SAVED] {stat_out}")

    # If available, copy / filter selected labels table for convenience.
    selected_csv = None
    if args.selected_labels_csv:
        candidate = Path(args.selected_labels_csv)
        if candidate.exists():
            selected_csv = candidate
    else:
        candidate = batch_root / "selected_high_phrase_duration_variance_labels.csv"
        if candidate.exists():
            selected_csv = candidate
    if selected_csv is not None:
        try:
            sel = pd.read_csv(selected_csv)
            sel.to_csv(out_dir / "selected_high_phrase_duration_variance_labels.csv", index=False)
            print(f"[SAVED] {out_dir / 'selected_high_phrase_duration_variance_labels.csv'}")
        except Exception as e:
            print(f"[WARN] Could not copy selected labels CSV: {e}")

    # plots
    make_overview_multiplot(bird_df, out_dir / "bird_level_summary_all_features.png", args.title_prefix)
    for feature in features:
        scatter_bird_summary(
            per_label_df=per_label_df,
            bird_df=bird_df,
            feature=feature,
            out_png=out_dir / f"bird_level_summary_{feature}.png",
            title_prefix=args.title_prefix,
        )
        scatter_baseline_vs_lesion(
            per_label_df=per_label_df,
            bird_df=bird_df,
            feature=feature,
            out_png=out_dir / f"baseline_vs_lesion_{feature}.png",
            title_prefix=args.title_prefix,
            distance_metric=args.distance_metric,
            lesion_post_period=args.lesion_post_period,
        )

    # small readme
    readme = out_dir / "README_aggregate_v31.txt"
    readme.write_text(
        "This folder contains higher-level aggregation outputs for per-label acoustic analyses.\n"
        "Main tables:\n"
        "- bird_label_acoustic_effects.csv: one row per bird × label × feature\n"
        "- bird_level_acoustic_summary.csv: bird-level medians across selected labels\n"
        "- bird_level_acoustic_stats.csv: across-bird Wilcoxon summaries\n\n"
        "Main figures:\n"
        "- bird_level_summary_<feature>.png\n"
        "- baseline_vs_lesion_<feature>.png\n"
        "- bird_level_summary_all_features.png\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE] Aggregation complete.")


if __name__ == "__main__":
    main()
