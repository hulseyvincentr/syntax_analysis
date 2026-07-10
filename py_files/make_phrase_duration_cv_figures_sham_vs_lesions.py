#!/usr/bin/env python3
"""
make_phrase_duration_cv_figures_sham_vs_lesions.py

Calculate coefficient of variation (CV = SD / mean) for phrase durations and
make Figure-3-style plots with planned sham-vs-lesion comparisons.

Purpose
-------
This script checks whether lesion-associated phrase-duration variability remains
when variability is normalized by mean phrase duration.

It uses the same selected syllables as the prolongation analysis and computes:

    CV = phrase-duration SD / phrase-duration mean
    ΔCV = post-lesion CV - late-pre-lesion CV

Main outputs
------------
animal_syllable_cv_metrics.csv
animal_level_cv_metrics.csv
group_cv_stats.csv
group_comparison_cv_stats.csv
figures/paper_pre_vs_post_cv.pdf/png
figures/paper_paired_box_cv.pdf/png
figures/paper_delta_by_group_delta_cv.pdf/png

Recommended example
-------------------
python make_phrase_duration_cv_figures_sham_vs_lesions.py \
  --long-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/all_birds_phrase_duration_long_latepre_post_epochs.csv" \
  --selected-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/selected_syllables_used_for_prolongation.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "animal_hit_type_summary" \
  --metadata-bird-col "Animal ID" \
  --metadata-plot-col "Lesion hit type" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/cv_figures" \
  --n-balance 200 \
  --n-perm 10000 \
  --correction holm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# -----------------------------
# Paper-style colors / labels
# -----------------------------
PAPER_COLORS = {
    "complete_medial_lateral": "#4B0082",   # dark purple
    "partial_medial_lateral": "#7E57C2",    # medium purple
    "medial_lateral": "#8E6CCB",            # pooled ML purple
    "lateral_only": "#B39DDB",              # light purple
    "sham": "#1B9E77",                      # teal/green
}

PAPER_SCATTER_ORDER = [
    "complete_medial_lateral",
    "partial_medial_lateral",
    "medial_lateral",
    "lateral_only",
    "sham",
]

PAPER_SCATTER_LABELS = {
    "complete_medial_lateral": "Complete Medial and Lateral lesion",
    "partial_medial_lateral": "Partial Medial and Lateral lesion",
    "medial_lateral": "Complete and partial medial and lateral lesion",
    "lateral_only": "Lateral lesion only",
    "sham": "sham saline injection",
}

GROUP_ORDER = ["sham", "lateral_only", "medial_lateral"]
GROUP_LABELS = {
    "sham": "sham saline\ninjection",
    "lateral_only": "Lateral lesion\nonly",
    "medial_lateral": "Complete and partial\nmedial and lateral lesion",
}


# -----------------------------
# Helpers
# -----------------------------
def norm_value(x: object) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def normalize_group_value(x: object) -> str:
    v = norm_value(x)
    compact = v.replace(" ", "")
    if "sham" in v or "saline" in v:
        return "sham"
    if "medial" in v and "lateral" in v:
        return "medial_lateral"
    if "area x not visible" in v or "largelesion" in compact or "large lesion" in v or "complete" in v:
        return "medial_lateral"
    if "combined" in v or "m+l" in compact:
        return "medial_lateral"
    if "single hit" in v or ("lateral" in v and "only" in v):
        return "lateral_only"
    if compact in {"medial_lateral", "mediallateral", "ml"}:
        return "medial_lateral"
    if compact in {"lateral_only", "lateralonly", "lateral"}:
        return "lateral_only"
    return compact if compact else "unknown"


def normalize_plot_group_value(x: object) -> str:
    v = norm_value(x)
    compact = v.replace(" ", "")
    if "sham" in v or "saline" in v:
        return "sham"
    if "single hit" in v or "lateral only" in v or "lateral-only" in v:
        return "lateral_only"
    if "area x not visible" in v or "largelesion" in compact or "large lesion" in v or "complete" in v:
        return "complete_medial_lateral"
    if ("medial" in v and "lateral" in v) or "m+l" in compact or "medial+lateral" in v:
        return "partial_medial_lateral"
    return normalize_group_value(x)


def find_col(df: pd.DataFrame, requested: Optional[str], candidates: list[str], label: str) -> str:
    if requested:
        if requested in df.columns:
            return requested
        raise ValueError(f"Requested {label} '{requested}' not found. Available columns: {list(df.columns)}")

    norm_to_actual = {"".join(ch.lower() for ch in c if ch.isalnum()): c for c in df.columns}
    for c in candidates:
        key = "".join(ch.lower() for ch in c if ch.isalnum())
        if key in norm_to_actual:
            return norm_to_actual[key]
    raise ValueError(f"Could not find {label}. Available columns: {list(df.columns)}")


def split_values(s: str) -> set[str]:
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}


def p_to_label(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def apply_paper_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    h = 0.03 * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=10)


def add_identity(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    vals = np.concatenate([x[mask], y[mask]])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.3, color="#D62728", alpha=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


# -----------------------------
# p-value corrections
# -----------------------------
def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    p = pvals[mask]
    if len(p) == 0:
        return out

    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = np.empty(m)
    running = 0.0
    for i, pv in enumerate(p_sorted):
        adj = (m - i) * pv
        running = max(running, adj)
        adj_sorted[i] = min(running, 1.0)

    adj = np.empty(m)
    adj[order] = adj_sorted
    out[mask] = adj
    return out


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    p = pvals[mask]
    if len(p) == 0:
        return out

    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = p_sorted[i] * m / rank
        running = min(running, adj)
        adj_sorted[i] = min(running, 1.0)

    adj = np.empty(m)
    adj[order] = adj_sorted
    out[mask] = adj
    return out


def add_corrected_p_columns(df: pd.DataFrame, metric_col: str, p_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{p_col}_holm"] = np.nan
    df[f"{p_col}_fdr_bh"] = np.nan

    for metric, idx in df.groupby(metric_col).groups.items():
        idx = list(idx)
        p = df.loc[idx, p_col].astype(float).to_numpy()
        df.loc[idx, f"{p_col}_holm"] = holm_bonferroni(p)
        df.loc[idx, f"{p_col}_fdr_bh"] = fdr_bh(p)

    return df


# -----------------------------
# Metrics
# -----------------------------
def cv_metric(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {
            "mean_s": np.nan,
            "sd_s": np.nan,
            "cv": np.nan,
            "n": len(x),
        }

    mean_s = float(np.mean(x))
    sd_s = float(np.std(x, ddof=1))
    cv = sd_s / mean_s if mean_s > 0 else np.nan

    return {
        "mean_s": mean_s,
        "sd_s": sd_s,
        "cv": cv,
        "n": len(x),
    }


def balanced_pre_post_cv(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, n_balance: int, min_renditions: int) -> Optional[dict[str, float]]:
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]

    n = min(len(pre), len(post))
    if n < min_renditions:
        return None

    rows = []
    for _ in range(n_balance):
        pre_s = rng.choice(pre, size=n, replace=False)
        post_s = rng.choice(post, size=n, replace=False)

        pre_m = cv_metric(pre_s)
        post_m = cv_metric(post_s)

        row = {}
        for k, v in pre_m.items():
            row[f"pre_{k}"] = v
        for k, v in post_m.items():
            row[f"post_{k}"] = v

        row["delta_cv"] = row["post_cv"] - row["pre_cv"]
        row["delta_sd_s"] = row["post_sd_s"] - row["pre_sd_s"]
        row["delta_mean_s"] = row["post_mean_s"] - row["pre_mean_s"]
        rows.append(row)

    out = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    out["n_balanced"] = int(n)
    return out


def compute_syllable_metrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []

    for (bird, group, syll), g in df.groupby(["bird", "group", "syllable"], sort=True):
        pre = g.loc[g["epoch"].eq("pre"), "duration_s"].to_numpy()
        post = g.loc[g["epoch"].eq("post"), "duration_s"].to_numpy()

        if len(pre) < args.min_renditions or len(post) < args.min_renditions:
            continue

        metrics = balanced_pre_post_cv(pre, post, rng, args.n_balance, args.min_renditions)
        if metrics is None:
            continue

        rows.append({
            "bird": bird,
            "group": group,
            "syllable": syll,
            "n_pre_raw": len(pre),
            "n_post_raw": len(post),
            **metrics,
        })

    return pd.DataFrame(rows)


def aggregate_animals(syll: pd.DataFrame, agg: str) -> pd.DataFrame:
    rows = []
    metrics = [
        "pre_mean_s", "post_mean_s", "delta_mean_s",
        "pre_sd_s", "post_sd_s", "delta_sd_s",
        "pre_cv", "post_cv", "delta_cv",
    ]

    for (bird, group), g in syll.groupby(["bird", "group"], sort=True):
        row = {
            "bird": bird,
            "group": group,
            "selected_syllables": g["syllable"].nunique(),
            "total_n_pre_raw": int(g["n_pre_raw"].sum()),
            "total_n_post_raw": int(g["n_post_raw"].sum()),
            "total_n_balanced": int(g["n_balanced"].sum()),
        }

        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce")
            if agg == "mean":
                row[metric] = float(vals.mean())
            elif agg == "max":
                row[metric] = float(vals.max())
            else:
                row[metric] = float(vals.median())

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Stats
# -----------------------------
def signflip_p(vals: np.ndarray, rng: np.random.Generator, n_perm: int, alternative: str) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan

    obs = float(np.mean(vals))
    signs = rng.choice([-1, 1], size=(n_perm, len(vals)), replace=True)
    null = (signs * vals).mean(axis=1)

    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)

    return obs, float(p)


def labelshuffle_p(animal: pd.DataFrame, metric: str, group_a: str, group_b: str, rng: np.random.Generator, n_perm: int, alternative: str) -> tuple[float, float]:
    sub = animal[animal["group"].isin([group_a, group_b])].copy()

    vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy()
    groups = sub["group"].to_numpy()

    mask = np.isfinite(vals)
    vals = vals[mask]
    groups = groups[mask]

    if vals.size == 0 or np.sum(groups == group_a) == 0 or np.sum(groups == group_b) == 0:
        return np.nan, np.nan

    obs = float(vals[groups == group_a].mean() - vals[groups == group_b].mean())

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

    return obs, float(p)


def make_stats(animal: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 999)

    group_rows = []
    comp_rows = []

    metric = "delta_cv"

    for group in GROUP_ORDER:
        vals = animal.loc[animal["group"].eq(group), metric].dropna().to_numpy()
        obs, p = signflip_p(vals, rng, args.n_perm, args.alternative)
        group_rows.append({
            "metric": metric,
            "group": group,
            "n_birds": int(len(vals)),
            "mean_delta_cv": float(np.mean(vals)) if len(vals) else np.nan,
            "median_delta_cv": float(np.median(vals)) if len(vals) else np.nan,
            "signflip_stat_mean_delta": obs,
            "signflip_p": p,
            "alternative": args.alternative,
        })

    # Planned comparisons to sham only.
    # Comparisons are stored as lesion - sham, so alternative="greater" tests:
    #   lateral-only ΔCV > sham ΔCV
    #   medial+lateral ΔCV > sham ΔCV
    pairwise = [
        ("lateral_only", "sham"),
        ("medial_lateral", "sham"),
    ]

    between_alt = args.between_alternative
    for a, b in pairwise:
        obs, p = labelshuffle_p(animal, metric, a, b, rng, args.n_perm, between_alt)
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
            "alternative": between_alt,
        })

    group_stats = add_corrected_p_columns(pd.DataFrame(group_rows), metric_col="metric", p_col="signflip_p")
    comp_stats = add_corrected_p_columns(pd.DataFrame(comp_rows), metric_col="metric", p_col="labelshuffle_p")
    return group_stats, comp_stats


# -----------------------------
# Metadata
# -----------------------------
def merge_plot_metadata(animal: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    animal = animal.copy()
    animal["plot_group"] = animal["group"]

    if args.metadata_excel is None:
        return animal

    meta = pd.read_excel(args.metadata_excel, sheet_name=args.metadata_sheet)
    bird_col = find_col(meta, args.metadata_bird_col, ["Animal ID", "bird", "animal_id", "Bird ID"], "metadata bird column")
    plot_col = find_col(meta, args.metadata_plot_col, ["Lesion hit type", "lesion_group", "Treatment type"], "metadata plotting column")

    m = meta[[bird_col, plot_col]].copy()
    m.columns = ["bird", "plot_raw"]
    m["bird"] = m["bird"].astype(str)
    m["plot_group"] = m["plot_raw"].map(normalize_plot_group_value)
    m = m.drop_duplicates("bird")

    animal = animal.drop(columns=["plot_group"], errors="ignore").merge(m[["bird", "plot_group"]], on="bird", how="left")
    animal["plot_group"] = animal["plot_group"].fillna(animal["group"])
    return animal


# -----------------------------
# Plots
# -----------------------------
def get_p_col(base: str, correction: str) -> str:
    if correction == "none":
        return base
    return f"{base}_{correction}"


def plot_pre_vs_post_cv(animal: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    handles, labels = [], []
    present = set(animal["plot_group"])

    for pg in PAPER_SCATTER_ORDER:
        if pg not in present:
            continue
        g = animal[animal["plot_group"].eq(pg)]
        h = ax.scatter(g["pre_cv"], g["post_cv"], s=34, alpha=0.85, color=PAPER_COLORS.get(pg, "0.4"), edgecolor="none")
        handles.append(h)
        labels.append(PAPER_SCATTER_LABELS.get(pg, pg))

    add_identity(ax, animal["pre_cv"].to_numpy(), animal["post_cv"].to_numpy())

    ax.set_xlabel("Late pre-lesion coefficient of variation", fontsize=13)
    ax.set_ylabel("Post-lesion coefficient of variation", fontsize=13)
    apply_paper_style(ax)

    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(fig_dir / "paper_pre_vs_post_cv.png", dpi=450)
    fig.savefig(fig_dir / "paper_pre_vs_post_cv.pdf")
    plt.close(fig)


def plot_paired_box_cv(animal: pd.DataFrame, group_stats: pd.DataFrame, fig_dir: Path, correction: str) -> None:
    groups = [g for g in GROUP_ORDER if g in set(animal["group"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    pre_positions, post_positions = [], []
    xticks, xticklabels, all_vals = [], [], []
    p_col = get_p_col("signflip_p", correction)

    for i, group in enumerate(groups):
        center = i * 2.2 + 1.0
        pre_pos, post_pos = center - 0.22, center + 0.22
        pre_positions.append(pre_pos)
        post_positions.append(post_pos)
        xticks.append(center)
        xticklabels.append(GROUP_LABELS[group])

        color = PAPER_COLORS[group]
        g = animal[animal["group"].eq(group)]

        pre_vals = g["pre_cv"].dropna().to_numpy(dtype=float)
        post_vals = g["post_cv"].dropna().to_numpy(dtype=float)
        all_vals.extend(pre_vals.tolist() + post_vals.tolist())

        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=[pre_pos, post_pos],
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.25", "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        bp["boxes"][0].set(facecolor="white", edgecolor=color, linewidth=1.4)
        bp["boxes"][1].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.42)

        # paired bird-level points/lines
        for _, row in g.iterrows():
            if np.isfinite(row["pre_cv"]) and np.isfinite(row["post_cv"]):
                ax.plot([pre_pos, post_pos], [row["pre_cv"], row["post_cv"]], color=color, alpha=0.25, linewidth=0.8)
                ax.scatter([pre_pos, post_pos], [row["pre_cv"], row["post_cv"]], color=color, s=20, alpha=0.8, edgecolor="none", zorder=3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_ylabel("Coefficient of variation", fontsize=13)
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor="0.25", label="Late Pre"),
            Patch(facecolor="0.55", edgecolor="0.55", alpha=0.42, label="Post"),
        ],
        frameon=False,
        fontsize=10,
        loc="upper left",
        ncol=2,
    )
    apply_paper_style(ax)

    if all_vals:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(max(0, ymin - 0.08 * yr), ymax + 0.25 * yr)

        for i, group in enumerate(groups):
            rows = group_stats[(group_stats["metric"].eq("delta_cv")) & (group_stats["group"].eq(group))]
            p = float(rows.iloc[0][p_col]) if (not rows.empty and p_col in rows.columns) else np.nan
            g = animal[animal["group"].eq(group)]
            vals = pd.concat([g["pre_cv"], g["post_cv"]], ignore_index=True).dropna().to_numpy(float)
            if len(vals):
                add_bracket(ax, pre_positions[i], post_positions[i], float(np.nanmax(vals)) + 0.06 * yr, p_to_label(p))

    fig.tight_layout()
    fig.savefig(fig_dir / "paper_paired_box_cv.png", dpi=450)
    fig.savefig(fig_dir / "paper_paired_box_cv.pdf")
    plt.close(fig)


def plot_delta_cv(animal: pd.DataFrame, comp_stats: pd.DataFrame, fig_dir: Path, correction: str) -> None:
    groups = [g for g in GROUP_ORDER if g in set(animal["group"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    rng = np.random.default_rng(123)
    positions = np.arange(1, len(groups) + 1)
    all_vals = []
    p_col = get_p_col("labelshuffle_p", correction)

    for pos, group in zip(positions, groups):
        color = PAPER_COLORS[group]
        vals = animal.loc[animal["group"].eq(group), "delta_cv"].dropna().to_numpy(float)
        all_vals.extend(vals.tolist())

        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.25", "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        bp["boxes"][0].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.35)
        ax.scatter(
            np.full(len(vals), pos) + rng.normal(0, 0.045, len(vals)),
            vals,
            s=28,
            color=color,
            alpha=0.85,
            edgecolor="none",
            zorder=3,
        )

    ax.axhline(0, linestyle="--", linewidth=1.0, color="0.35")
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=11)
    ax.set_ylabel("Δ coefficient of variation", fontsize=13)
    apply_paper_style(ax)

    if all_vals:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.18 * yr, ymax + 0.34 * yr)

        # Planned sham-vs-lesion comparisons only.
        # Stats are stored as lesion - sham, but brackets are drawn from sham to each lesion group.
        pair_defs = [
            ("lateral_only", "sham"),
            ("medial_lateral", "sham"),
        ]
        heights = [ymax + 0.07 * yr, ymax + 0.18 * yr]

        for (a, b), y in zip(pair_defs, heights):
            if a in groups and b in groups:
                x1, x2 = sorted([groups.index(a) + 1, groups.index(b) + 1])
                rows = comp_stats[(comp_stats["metric"].eq("delta_cv")) & (comp_stats["group_a"].eq(a)) & (comp_stats["group_b"].eq(b))]
                p = float(rows.iloc[0][p_col]) if (not rows.empty and p_col in rows.columns) else np.nan
                add_bracket(ax, x1, x2, y, p_to_label(p))

    fig.tight_layout()
    fig.savefig(fig_dir / "paper_delta_by_group_delta_cv.png", dpi=450)
    fig.savefig(fig_dir / "paper_delta_by_group_delta_cv.pdf")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--long-csv", required=True, type=Path)
    ap.add_argument("--selected-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)

    ap.add_argument("--bird-col", default="animal_id")
    ap.add_argument("--syllable-col", default="syllable")
    ap.add_argument("--epoch-col", default="analysis_epoch")
    ap.add_argument("--duration-col", default="Phrase Duration (ms)")
    ap.add_argument("--group-col", default="lesion_group")
    ap.add_argument("--duration-unit", choices=["ms", "s"], default="ms")
    ap.add_argument("--pre-epoch-values", default="pre")
    ap.add_argument("--post-epoch-values", default="post")

    ap.add_argument("--metadata-excel", type=Path, default=None)
    ap.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    ap.add_argument("--metadata-bird-col", default="Animal ID")
    ap.add_argument("--metadata-plot-col", default="Lesion hit type")

    ap.add_argument("--agg", choices=["median", "mean", "max"], default="median",
                    help="How to aggregate selected syllables within animal.")
    ap.add_argument("--min-renditions", type=int, default=10)
    ap.add_argument("--n-balance", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="greater",
                    help="Within-group sign-flip alternative for testing post > pre CV.")
    ap.add_argument("--between-alternative", choices=["greater", "less", "two-sided"], default="greater",
                    help="Between-group label-shuffle alternative for planned sham-vs-lesion ΔCV comparisons. With default comparison ordering, greater tests lesion ΔCV > sham ΔCV.")
    ap.add_argument("--correction", choices=["none", "holm", "fdr_bh"], default="holm",
                    help="Which p-value column to use for figure brackets.")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load selected syllables.
    selected = pd.read_csv(args.selected_csv)
    sel = selected.copy()

    if "bird" not in sel.columns:
        if args.bird_col in sel.columns:
            sel = sel.rename(columns={args.bird_col: "bird"})
        elif "animal_id" in sel.columns:
            sel = sel.rename(columns={"animal_id": "bird"})
        elif "Animal ID" in sel.columns:
            sel = sel.rename(columns={"Animal ID": "bird"})
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

    # Load long table.
    long = pd.read_csv(args.long_csv)
    df = long.rename(columns={
        args.bird_col: "bird",
        args.syllable_col: "syllable",
        args.epoch_col: "epoch_raw",
        args.duration_col: "duration_raw",
        args.group_col: "group_raw",
    }).copy()

    needed = ["bird", "syllable", "epoch_raw", "duration_raw", "group_raw"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after renaming: {missing}. Available columns: {list(df.columns)}")

    pre_vals = split_values(args.pre_epoch_values)
    post_vals = split_values(args.post_epoch_values)
    epoch_lower = df["epoch_raw"].astype(str).str.strip().str.lower()
    keep = epoch_lower.isin(pre_vals | post_vals)

    df = df[keep].copy()
    df["epoch"] = np.where(epoch_lower[keep].isin(pre_vals), "pre", "post")
    df["bird"] = df["bird"].astype(str)
    df["syllable"] = df["syllable"].astype(str)
    df["group"] = df["group_raw"].map(normalize_group_value)
    df["duration_s"] = pd.to_numeric(df["duration_raw"], errors="coerce")

    if args.duration_unit == "ms":
        df["duration_s"] = df["duration_s"] / 1000.0

    df = df[np.isfinite(df["duration_s"]) & (df["duration_s"] >= 0)].copy()

    before = len(df)
    df = df.merge(sel.assign(selected=True), on=["bird", "syllable"], how="inner")

    print(f"[INFO] Kept {len(df):,}/{before:,} rows after selected-syllable filter.")
    print("[INFO] Selected rows by group x epoch:")
    print(pd.crosstab(df["group"], df["epoch"]).to_string())

    # Metrics.
    syll = compute_syllable_metrics(df, args)
    animal = aggregate_animals(syll, agg=args.agg)
    group_stats, comp_stats = make_stats(animal, args)

    animal = merge_plot_metadata(animal, args)

    # Save outputs.
    syll.to_csv(args.out_dir / "animal_syllable_cv_metrics.csv", index=False)
    animal.to_csv(args.out_dir / "animal_level_cv_metrics.csv", index=False)
    group_stats.to_csv(args.out_dir / "group_cv_stats.csv", index=False)
    comp_stats.to_csv(args.out_dir / "group_comparison_cv_stats.csv", index=False)

    # Plots.
    plot_pre_vs_post_cv(animal, fig_dir)
    plot_paired_box_cv(animal, group_stats, fig_dir, correction=args.correction)
    plot_delta_cv(animal, comp_stats, fig_dir, correction=args.correction)

    # Console summary.
    print("\n[INFO] Birds by statistical group:")
    print(animal.groupby("group").size().to_string())

    print("\n[INFO] Birds by plotting subgroup:")
    print(animal.groupby("plot_group").size().to_string())

    print("\n[CHECK] Animal-level CV metrics:")
    print(animal[["bird", "group", "selected_syllables", "pre_cv", "post_cv", "delta_cv"]].sort_values(["group", "bird"]).to_string(index=False))

    print("\n[CHECK] Within-group ΔCV stats:")
    print(group_stats.to_string(index=False))

    print("\n[CHECK] Planned sham-vs-lesion ΔCV comparisons:")
    print(comp_stats.to_string(index=False))

    print("\n[DONE] Wrote CV outputs to:", args.out_dir)
    print("[DONE] Figures:", fig_dir)


if __name__ == "__main__":
    main()
