#!/usr/bin/env python3
"""
Make q99-q100 extreme-tail phrase-duration prolongation metrics and paper-style figures with all pairwise lesion-group comparisons.

Purpose
-------
This script is meant to address the reviewer-style question:
    Are the very longest phrase durations prolonged after lesion?

It adds two complementary q99-q100 extreme-tail metrics:
    1. tail_mean_q99_to_max_s
       Mean duration of the upper 1% of renditions, i.e. all values >= q99.
       This is usually the best main upper-tail metric because it captures rare
       long phrases but is less unstable than the single maximum.

    2. tail_range_max_minus_q99_s
       max - q99, i.e. the span from q99 to q100. This directly visualizes the
       extreme tail but is very sensitive to one outlier/annotation artifact.

It also computes q99, q99.5, q99.9, and max for QC.

Recommended use
---------------
Use the BALANCED animal-level metrics for inferential statistics. Use the RAW
syllable-level metrics/plots as QC to show that the 20-40 s phrases are really
present in the selected syllables.

Example
-------
python make_upper_tail_q95_q100_figures.py \
  --long-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/all_birds_phrase_duration_long_latepre_post_epochs.csv" \
  --selected-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/selected_syllables_used_for_prolongation.csv" \
  --bird-col "animal_id" \
  --syllable-col "syllable" \
  --epoch-col "analysis_epoch" \
  --duration-col "Phrase Duration (ms)" \
  --group-col "lesion_group" \
  --duration-unit ms \
  --pre-epoch-values pre \
  --post-epoch-values post \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/q99_q100_tail_figures" \
  --n-balance 200 \
  --n-perm 10000 \
  --seed 123
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# -----------------------------
# Style / labels
# -----------------------------
GROUP_ORDER = ["sham", "lateral_only", "medial_lateral"]
GROUP_LABELS = {
    "sham": "sham saline\ninjection",
    "lateral_only": "Lateral lesion\nonly",
    "medial_lateral": "Complete and partial\nmedial and lateral lesion",
}
GROUP_COLORS = {
    "sham": "#1b9e77",           # teal
    "lateral_only": "#b39ddb",   # light purple
    "medial_lateral": "#9575cd", # pooled purple
}
SCATTER_COLORS = {
    "complete_medial_lateral": "#5e1a9a",
    "partial_medial_lateral": "#8e6bd1",
    "lateral_only": "#b39ddb",
    "sham": "#1b9e77",
}
SCATTER_LABELS = {
    "complete_medial_lateral": "Complete Medial and Lateral lesion",
    "partial_medial_lateral": "Partial Medial and Lateral lesion",
    "lateral_only": "Lateral lesion only",
    "sham": "sham saline injection",
}

Y_LABELS = {
    "tail_mean_q99_to_max_s": "Mean phrase duration from q99-q100 (s)",
    "tail_range_max_minus_q99_s": "q100-q99 phrase-duration range (s)",
    "tail_range_q999_minus_q99_s": "q99.9-q99 phrase-duration range (s)",
    "q99_duration_s": "99th percentile phrase duration (s)",
    "q999_duration_s": "99.9th percentile phrase duration (s)",
    "max_duration_s": "Maximum phrase duration (s)",
}

DELTA_LABELS = {
    "delta_tail_mean_q99_to_max_s": "Δ mean q99-q100 phrase duration (s)",
    "delta_tail_range_max_minus_q99_s": "Δ q100-q99 phrase-duration range (s)",
    "delta_tail_range_q999_minus_q99_s": "Δ q99.9-q99 phrase-duration range (s)",
    "delta_q99_duration_s": "Δ 99th percentile phrase duration (s)",
    "delta_q999_duration_s": "Δ 99.9th percentile phrase duration (s)",
    "delta_max_duration_s": "Δ maximum phrase duration (s)",
}

PRE_POST_METRICS = [
    "tail_mean_q99_to_max_s",
    "tail_range_max_minus_q99_s",
    "tail_range_q999_minus_q99_s",
    "q99_duration_s",
    "q999_duration_s",
    "max_duration_s",
]
DELTA_METRICS = ["delta_" + m for m in PRE_POST_METRICS]


def clean_key(x) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("+", "_").replace("/", "_")


def normalize_group(x) -> str:
    s = str(x).strip().lower()
    if "sham" in s:
        return "sham"
    if "lateral" in s and ("only" in s or "single" in s) and "medial" not in s and "m+l" not in s:
        return "lateral_only"
    if "lateral hit only" in s:
        return "lateral_only"
    if "medial" in s or "m+l" in s or "large lesion" in s or "combined" in s:
        return "medial_lateral"
    # fall back to cleaned string
    return clean_key(s)


def infer_plot_group(group: str, bird: str | None = None) -> str:
    """For scatter colors. Without detailed metadata, pooled ML stays pooled."""
    g = normalize_group(group)
    return g


def split_values(s: str | Iterable[str]) -> set[str]:
    if isinstance(s, str):
        return {x.strip().lower() for x in s.split(",") if x.strip()}
    return {str(x).strip().lower() for x in s}


def p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def holm_bonferroni(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out.tolist()
    p = arr[mask]
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for i, pv in enumerate(p_sorted):
        adj = (m - i) * pv
        running = max(running, adj)
        adj_sorted[i] = min(running, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    out[mask] = adj
    return out.tolist()


def fdr_bh(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out.tolist()
    p = arr[mask]
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = p_sorted[i] * m / rank
        running = min(running, adj)
        adj_sorted[i] = min(running, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    out[mask] = adj
    return out.tolist()


def add_corrected_p_columns(df: pd.DataFrame, metric_col: str, p_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{p_col}_holm"] = np.nan
    df[f"{p_col}_fdr_bh"] = np.nan
    for metric, idx in df.groupby(metric_col).groups.items():
        idx = list(idx)
        raw = df.loc[idx, p_col].astype(float).tolist()
        df.loc[idx, f"{p_col}_holm"] = holm_bonferroni(raw)
        df.loc[idx, f"{p_col}_fdr_bh"] = fdr_bh(raw)
    return df


def set_paper_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(axis="both", width=1.8, length=6, labelsize=14)


def add_bracket(ax, x1, x2, y, text, h_frac=0.035):
    ymin, ymax = ax.get_ylim()
    h = (ymax - ymin) * h_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.2, text, ha="center", va="bottom", fontsize=15)


# -----------------------------
# Metrics
# -----------------------------
def tail_metrics(x: np.ndarray) -> dict[str, float]:
    """
    Compute extreme upper-tail metrics.

    Main q99-q100 metrics:
      - tail_mean_q99_to_max_s: mean of values >= q99
      - tail_range_max_minus_q99_s: max - q99
      - tail_range_q999_minus_q99_s: q99.9 - q99

    The q99.9 range is included as a less maximum-sensitive companion to q100-q99.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {m: np.nan for m in PRE_POST_METRICS}
    q95 = float(np.quantile(x, 0.95))
    q99 = float(np.quantile(x, 0.99))
    q995 = float(np.quantile(x, 0.995))
    q999 = float(np.quantile(x, 0.999))
    xmax = float(np.max(x))
    top99 = x[x >= q99]
    return {
        "q95_duration_s": q95,
        "q99_duration_s": q99,
        "q995_duration_s": q995,
        "q999_duration_s": q999,
        "max_duration_s": xmax,
        "tail_mean_q99_to_max_s": float(np.mean(top99)) if top99.size else np.nan,
        "tail_range_max_minus_q99_s": xmax - q99,
        "tail_range_q999_minus_q99_s": q999 - q99,
        "tail_n_q99_to_max": int(top99.size),
        "n": int(x.size),
    }


def balanced_epoch_metrics(x: np.ndarray, n: int, rng: np.random.Generator, n_balance: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or n < 2:
        return {m: np.nan for m in list(PRE_POST_METRICS) + ["q95_duration_s", "q99_duration_s", "q995_duration_s", "tail_n_q99_to_max", "n"]}
    reps = []
    replace = False
    for _ in range(n_balance):
        idx = rng.choice(x.size, size=n, replace=replace)
        reps.append(tail_metrics(x[idx]))
    out = {}
    keys = reps[0].keys()
    for k in keys:
        vals = np.array([r[k] for r in reps], dtype=float)
        out[k] = float(np.nanmean(vals))
    out["n"] = int(n)
    return out


def compute_syllable_metrics(df: pd.DataFrame, args, balanced: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    for (bird, group, syll), g in df.groupby(["bird", "group", "syllable"], sort=True):
        pre = g.loc[g["epoch"].eq("pre"), "duration_s"].to_numpy()
        post = g.loc[g["epoch"].eq("post"), "duration_s"].to_numpy()
        if len(pre) < args.min_renditions or len(post) < args.min_renditions:
            continue
        n_bal = min(len(pre), len(post))
        if balanced:
            pre_m = balanced_epoch_metrics(pre, n_bal, rng, args.n_balance)
            post_m = balanced_epoch_metrics(post, n_bal, rng, args.n_balance)
        else:
            pre_m = tail_metrics(pre)
            post_m = tail_metrics(post)
            pre_m["n"] = len(pre)
            post_m["n"] = len(post)

        row = {"bird": bird, "group": group, "syllable": syll, "n_pre_raw": len(pre), "n_post_raw": len(post), "n_balanced": n_bal}
        for k, v in pre_m.items():
            row[f"pre_{k}"] = v
        for k, v in post_m.items():
            row[f"post_{k}"] = v
        for m in PRE_POST_METRICS:
            row[f"delta_{m}"] = row.get(f"post_{m}", np.nan) - row.get(f"pre_{m}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_animals(syll_df: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
    metric_cols = [c for c in syll_df.columns if c.startswith("pre_") or c.startswith("post_") or c.startswith("delta_")]
    numeric_cols = metric_cols + ["n_pre_raw", "n_post_raw", "n_balanced"]
    rows = []
    for (bird, group), g in syll_df.groupby(["bird", "group"], sort=True):
        row = {"bird": bird, "group": group, "selected_syllables": g["syllable"].nunique()}
        for c in numeric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            if c.startswith("n_"):
                row[f"total_{c}"] = vals.sum()
            else:
                if agg == "mean":
                    row[c] = vals.mean()
                elif agg == "max":
                    row[c] = vals.max()
                else:
                    row[c] = vals.median()
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Stats
# -----------------------------
def signflip_p(vals: np.ndarray, rng: np.random.Generator, n_perm: int, alternative: str = "greater") -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    obs = float(np.mean(vals)) if vals.size else np.nan
    if vals.size == 0:
        return obs, np.nan
    signs = rng.choice([-1, 1], size=(n_perm, vals.size), replace=True)
    null = (signs * vals).mean(axis=1)
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return obs, float(p)


def labelshuffle_p(animal: pd.DataFrame, metric: str, group_a: str, group_b: str, rng: np.random.Generator, n_perm: int, alternative: str = "greater") -> tuple[float, float]:
    sub = animal[animal["group"].isin([group_a, group_b])].copy()
    vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy()
    groups = sub["group"].to_numpy()
    mask = np.isfinite(vals)
    vals = vals[mask]
    groups = groups[mask]
    obs = vals[groups == group_a].mean() - vals[groups == group_b].mean()
    count_a = np.sum(groups == group_a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(groups)
        null[i] = vals[perm == group_a].mean() - vals[perm == group_b].mean()
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return float(obs), float(p)


def make_stats(animal: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 999)
    group_rows = []
    comp_rows = []
    for metric in DELTA_METRICS:
        for group in GROUP_ORDER:
            vals = animal.loc[animal["group"].eq(group), metric].to_numpy()
            vals = vals[np.isfinite(vals)]
            obs, p = signflip_p(vals, rng, args.n_perm, args.alternative)
            group_rows.append({
                "metric": metric,
                "group": group,
                "n_birds": int(vals.size),
                "mean_delta": float(np.mean(vals)) if vals.size else np.nan,
                "median_delta": float(np.median(vals)) if vals.size else np.nan,
                "signflip_stat_mean_delta": obs,
                "signflip_p": p,
                "alternative": args.alternative,
            })
        pairwise = [("sham", "lateral_only"), ("sham", "medial_lateral"), ("lateral_only", "medial_lateral")]
        for a, b in pairwise:
            obs, p = labelshuffle_p(animal, metric, a, b, rng, args.n_perm, args.alternative)
            va = animal.loc[animal["group"].eq(a), metric].dropna().to_numpy()
            vb = animal.loc[animal["group"].eq(b), metric].dropna().to_numpy()
            comp_rows.append({
                "metric": metric,
                "comparison": f"{a} - {b}",
                "group_a": a,
                "group_b": b,
                "n_a": int(len(va)),
                "n_b": int(len(vb)),
                "mean_diff": float(np.mean(va) - np.mean(vb)) if len(va) and len(vb) else np.nan,
                "median_diff": float(np.median(va) - np.median(vb)) if len(va) and len(vb) else np.nan,
                "labelshuffle_stat_mean_diff": obs,
                "labelshuffle_p": p,
                "alternative": args.alternative,
            })
    return pd.DataFrame(group_rows), pd.DataFrame(comp_rows)


# -----------------------------
# Plots
# -----------------------------
def scatter_pre_post(animal: pd.DataFrame, metric: str, out_dir: Path, prefix: str):
    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    pre = f"pre_{metric}"
    post = f"post_{metric}"
    vals = pd.concat([animal[pre], animal[post]], ignore_index=True).dropna()
    lo = max(0, float(vals.min()) * 0.95) if len(vals) else 0
    hi = float(vals.max()) * 1.05 if len(vals) else 1
    if hi <= lo:
        hi = lo + 1

    for group in GROUP_ORDER:
        g = animal[animal["group"].eq(group)]
        ax.scatter(g[pre], g[post], s=55, color=GROUP_COLORS[group], alpha=0.9, edgecolor="none", label=GROUP_LABELS[group].replace("\n", " "))
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1.8, ls="--", alpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Late pre-lesion " + Y_LABELS.get(metric, metric).replace("Mean ", "mean ").replace("Maximum", "maximum"), fontsize=15)
    ax.set_ylabel("Post-lesion " + Y_LABELS.get(metric, metric).replace("Mean ", "mean ").replace("Maximum", "maximum"), fontsize=15)
    set_paper_axes(ax)
    ax.legend(frameon=False, fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pre_vs_post_{metric}.png", dpi=300)
    fig.savefig(out_dir / f"{prefix}_pre_vs_post_{metric}.pdf")
    plt.close(fig)


def paired_box(animal: pd.DataFrame, group_stats: pd.DataFrame, metric: str, out_dir: Path, prefix: str, args):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    centers = np.arange(len(GROUP_ORDER)) * 2.0
    width = 0.35
    rng = np.random.default_rng(22)
    all_y = []
    for i, group in enumerate(GROUP_ORDER):
        g = animal[animal["group"].eq(group)]
        color = GROUP_COLORS[group]
        pre_vals = g[f"pre_{metric}"].dropna().to_numpy()
        post_vals = g[f"post_{metric}"].dropna().to_numpy()
        all_y.extend(pre_vals.tolist() + post_vals.tolist())
        positions = [centers[i] - width / 1.5, centers[i] + width / 1.5]
        bp = ax.boxplot([pre_vals, post_vals], positions=positions, widths=width, patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)
            patch.set_facecolor("white" if j == 0 else color)
            patch.set_alpha(0.25 if j == 1 else 1.0)
        for elem in ["whiskers", "caps"]:
            for line in bp[elem]:
                line.set_color(color)
                line.set_linewidth(1.6)
        for line in bp["medians"]:
            line.set_color("#444444")
            line.set_linewidth(1.8)
        # paired points/lines
        for _, row in g.iterrows():
            x1, x2 = positions
            y1, y2 = row[f"pre_{metric}"], row[f"post_{metric}"]
            if np.isfinite(y1) and np.isfinite(y2):
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.25, lw=0.8)
                ax.scatter([x1, x2], [y1, y2], color=color, s=28, alpha=0.75, edgecolor="none")
        # within-group bracket
        p_col = "signflip_p" if getattr(args, "correction", "none") == "none" else f"signflip_p_{args.correction}"
        p = group_stats.loc[(group_stats["group"].eq(group)) & (group_stats["metric"].eq("delta_" + metric)), p_col]
        label = p_to_star(float(p.iloc[0])) if len(p) else "n.s."
        local_max = np.nanmax(np.r_[pre_vals, post_vals]) if len(pre_vals) or len(post_vals) else 1
        add_bracket(ax, positions[0], positions[1], local_max * 1.06, label, h_frac=0.025)
    ax.set_xticks(centers)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=13)
    ax.set_ylabel(Y_LABELS.get(metric, metric), fontsize=16)
    ymin = min(0, np.nanmin(all_y) * 0.95) if all_y else 0
    ymax = np.nanmax(all_y) * 1.25 if all_y else 1
    ax.set_ylim(ymin, ymax)
    set_paper_axes(ax)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#444444", label="Late Pre"), Patch(facecolor="#cccccc", edgecolor="#aaaaaa", label="Post")], frameon=False, fontsize=12, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_paired_box_{metric}.png", dpi=300)
    fig.savefig(out_dir / f"{prefix}_paired_box_{metric}.pdf")
    plt.close(fig)


def delta_box(animal: pd.DataFrame, comp_stats: pd.DataFrame, metric: str, out_dir: Path, prefix: str, args):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    centers = np.arange(len(GROUP_ORDER))
    all_y = []
    for i, group in enumerate(GROUP_ORDER):
        vals = animal.loc[animal["group"].eq(group), metric].dropna().to_numpy()
        all_y.extend(vals.tolist())
        color = GROUP_COLORS[group]
        bp = ax.boxplot([vals], positions=[i], widths=0.45, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_edgecolor(color)
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_linewidth(1.8)
        for elem in ["whiskers", "caps"]:
            for line in bp[elem]:
                line.set_color(color)
                line.set_linewidth(1.6)
        for line in bp["medians"]:
            line.set_color("#444444")
            line.set_linewidth(1.8)
        jitter = np.linspace(-0.08, 0.08, max(len(vals), 1))[:len(vals)]
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=45, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="#555555", lw=1.5, ls="--")
    ax.set_xticks(centers)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=13)
    ax.set_ylabel(DELTA_LABELS.get(metric, metric), fontsize=16)
    ymin = np.nanmin(all_y) if all_y else -1
    ymax = np.nanmax(all_y) if all_y else 1
    data_range = ymax - ymin
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(abs(ymax), 1.0)
    pad = data_range * 0.18
    top_extra = data_range * 0.55
    ax.set_ylim(ymin - pad * 0.35, ymax + top_extra)

    # Add all pairwise between-group comparisons
    pair_positions = {
        "sham - lateral_only": (0, 1),
        "sham - medial_lateral": (0, 2),
        "lateral_only - medial_lateral": (1, 2),
    }
    heights = [ymax + data_range * 0.10, ymax + data_range * 0.24, ymax + data_range * 0.38]
    for h, comp in zip(heights, ["sham - lateral_only", "sham - medial_lateral", "lateral_only - medial_lateral"]):
        p_col = "labelshuffle_p" if getattr(args, "correction", "none") == "none" else f"labelshuffle_p_{args.correction}"
        p = comp_stats.loc[(comp_stats["metric"].eq(metric)) & (comp_stats["comparison"].eq(comp)), p_col]
        label = p_to_star(float(p.iloc[0])) if len(p) else "n.s."
        x1, x2 = pair_positions[comp]
        add_bracket(ax, x1, x2, h, label, h_frac=0.020)

    set_paper_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_delta_by_group_{metric}.png", dpi=300)
    fig.savefig(out_dir / f"{prefix}_delta_by_group_{metric}.pdf")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", required=True)
    ap.add_argument("--selected-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bird-col", default="animal_id")
    ap.add_argument("--syllable-col", default="syllable")
    ap.add_argument("--epoch-col", default="analysis_epoch")
    ap.add_argument("--duration-col", default="Phrase Duration (ms)")
    ap.add_argument("--group-col", default="lesion_group")
    ap.add_argument("--duration-unit", choices=["ms", "s"], default="ms")
    ap.add_argument("--pre-epoch-values", default="pre")
    ap.add_argument("--post-epoch-values", default="post")
    ap.add_argument("--min-renditions", type=int, default=10)
    ap.add_argument("--n-balance", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="greater")
    ap.add_argument("--correction", choices=["none", "holm", "fdr_bh"], default="holm",
                    help="Multiple-comparison correction used for figure annotations. Applied within each metric across the 3 within-group tests or 3 pairwise group comparisons.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(args.long_csv)
    selected = pd.read_csv(args.selected_csv)

    # Normalize selected key columns.
    sel = selected.copy()
    if "bird" not in sel.columns:
        if args.bird_col in sel.columns:
            sel = sel.rename(columns={args.bird_col: "bird"})
        else:
            raise ValueError(f"selected CSV needs a bird column. Columns: {list(sel.columns)}")
    if "syllable" not in sel.columns:
        if args.syllable_col in sel.columns:
            sel = sel.rename(columns={args.syllable_col: "syllable"})
        else:
            raise ValueError(f"selected CSV needs a syllable column. Columns: {list(sel.columns)}")
    sel["bird"] = sel["bird"].astype(str)
    sel["syllable"] = sel["syllable"].astype(str)
    sel = sel[["bird", "syllable"]].drop_duplicates()

    df = long.rename(columns={
        args.bird_col: "bird",
        args.syllable_col: "syllable",
        args.epoch_col: "epoch_raw",
        args.duration_col: "duration_raw",
        args.group_col: "group_raw",
    }).copy()
    need = ["bird", "syllable", "epoch_raw", "duration_raw", "group_raw"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after renaming: {missing}. Available: {list(df.columns)}")

    pre_vals = split_values(args.pre_epoch_values)
    post_vals = split_values(args.post_epoch_values)
    epoch_lower = df["epoch_raw"].astype(str).str.strip().str.lower()
    df = df[epoch_lower.isin(pre_vals | post_vals)].copy()
    df["epoch"] = np.where(epoch_lower[epoch_lower.isin(pre_vals | post_vals)].isin(pre_vals), "pre", "post")
    df["bird"] = df["bird"].astype(str)
    df["syllable"] = df["syllable"].astype(str)
    df["group"] = df["group_raw"].map(normalize_group)
    df["duration_s"] = pd.to_numeric(df["duration_raw"], errors="coerce")
    if args.duration_unit == "ms":
        df["duration_s"] = df["duration_s"] / 1000.0
    df = df[np.isfinite(df["duration_s"]) & (df["duration_s"] >= 0)].copy()

    before = len(df)
    df = df.merge(sel.assign(selected=True), on=["bird", "syllable"], how="inner")
    print(f"[INFO] Kept {len(df):,}/{before:,} rows after selected-syllable filter.")
    print("[INFO] Selected rows by group x epoch:")
    print(pd.crosstab(df["group"], df["epoch"]).to_string())

    balanced = compute_syllable_metrics(df, args, balanced=True)
    raw = compute_syllable_metrics(df, args, balanced=False)
    animal_median = aggregate_animals(balanced, agg="median")
    animal_mean = aggregate_animals(balanced, agg="mean")
    animal_max = aggregate_animals(balanced, agg="max")

    balanced.to_csv(out_dir / "syllable_level_q99_q100_tail_metrics_balanced.csv", index=False)
    raw.to_csv(out_dir / "syllable_level_q99_q100_tail_metrics_raw.csv", index=False)
    animal_median.to_csv(out_dir / "animal_level_q99_q100_tail_metrics_median_across_syllables.csv", index=False)
    animal_mean.to_csv(out_dir / "animal_level_q99_q100_tail_metrics_mean_across_syllables.csv", index=False)
    animal_max.to_csv(out_dir / "animal_level_q99_q100_tail_metrics_max_across_syllables.csv", index=False)

    group_stats, comp_stats = make_stats(animal_median, args)
    group_stats = add_corrected_p_columns(group_stats, metric_col="metric", p_col="signflip_p")
    comp_stats = add_corrected_p_columns(comp_stats, metric_col="metric", p_col="labelshuffle_p")
    group_stats.to_csv(out_dir / "group_q99_q100_tail_stats_median_across_syllables.csv", index=False)
    comp_stats.to_csv(out_dir / "group_comparison_q99_q100_tail_stats_median_across_syllables.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for metric in PRE_POST_METRICS:
        scatter_pre_post(animal_median, metric, fig_dir, prefix="animal_median")
        paired_box(animal_median, group_stats, metric, fig_dir, prefix="animal_median", args=args)
    for metric in DELTA_METRICS:
        delta_box(animal_median, comp_stats, metric, fig_dir, prefix="animal_median", args=args)

    # QC raw syllable-level figures for the rare extreme values.
    raw_fig_dir = fig_dir / "raw_syllable_level_qc"
    raw_fig_dir.mkdir(exist_ok=True)
    for metric in ["tail_mean_q99_to_max_s", "tail_range_max_minus_q99_s", "tail_range_q999_minus_q99_s", "q99_duration_s", "q999_duration_s", "max_duration_s"]:
        scatter_pre_post(raw.rename(columns={}), metric, raw_fig_dir, prefix="raw_syllable")

    print(f"[DONE] Wrote q99-q100 extreme-tail metrics and figures to: {out_dir}")
    print("\n[CHECK] Top raw post max values:")
    cols = ["bird", "group", "syllable", "post_q99_duration_s", "post_tail_mean_q99_to_max_s", "post_q999_duration_s", "post_max_duration_s", "delta_tail_mean_q99_to_max_s", "delta_tail_range_max_minus_q99_s", "n_post_raw"]
    print(raw.sort_values("post_max_duration_s", ascending=False)[cols].head(20).to_string(index=False))
    print("\n[CHECK] Animal-level median-across-syllables tail deltas:")
    cols2 = ["bird", "group", "selected_syllables", "delta_tail_mean_q99_to_max_s", "delta_tail_range_max_minus_q99_s", "delta_q999_duration_s", "delta_max_duration_s"]
    print(animal_median.sort_values("delta_tail_mean_q99_to_max_s", ascending=False)[cols2].to_string(index=False))
    print("\n[CHECK] Group stats:")
    print(group_stats.to_string(index=False))
    print("\n[CHECK] Group comparisons:")
    print(comp_stats.to_string(index=False))


if __name__ == "__main__":
    main()
