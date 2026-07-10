#!/usr/bin/env python3
"""
make_prolongation_paper_style_figures.py

Remake manuscript-style prolongation figures from the outputs of
phrase_duration_prolongation_analysis.py, without re-running the resampling.

Creates Figure-3-style versions of:
  - pre vs post q99 phrase duration
  - pre vs post proportion above late-pre 99th percentile
  - paired pre/post boxplots for both metrics
  - delta-by-group boxplots for both metrics with all three pairwise lesion-group comparisons

Recommended use:
python make_prolongation_paper_style_figures.py \
  --animal-level-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/animal_level_prolongation_metrics.csv" \
  --group-stats-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/group_prolongation_stats.csv" \
  --comparison-stats-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/group_comparison_prolongation_stats.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "animal_hit_type_summary" \
  --metadata-bird-col "Animal ID" \
  --metadata-plot-col "Lesion hit type" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_prepost_top30/paper_style_figures"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

PAPER_GROUP_ORDER = ["sham", "lateral_only", "medial_lateral"]
PAPER_GROUP_LABELS = {
    "sham": "sham saline\ninjection",
    "lateral_only": "Lateral lesion\nonly",
    "medial_lateral": "Complete and partial\nmedial and lateral lesion",
}
PAPER_GROUP_FULL_LABELS = {
    "sham": "sham saline injection",
    "lateral_only": "Lateral lesion only",
    "medial_lateral": "Complete and partial medial and lateral lesion",
}
PAPER_METRIC_LABELS = {
    "q99_duration_s": "99th percentile phrase duration (s)",
    "prop_above_pre99": "proportion above late-pre 99th percentile",
}
FOCUS_BASES = ["q99_duration_s", "prop_above_pre99"]


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
    if "single hit" in v or "lateral" in v:
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


def apply_paper_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


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


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


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


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    h = 0.03 * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=10)


def get_group_p(group_stats: pd.DataFrame, metric: str, group: str, p_col: str = "signflip_p") -> float:
    rows = group_stats[(group_stats["metric"] == metric) & (group_stats["group"] == group)]
    if rows.empty or p_col not in rows.columns:
        return np.nan
    return float(rows.iloc[0][p_col])


def get_comparison_p(comparison_stats: pd.DataFrame, metric: str, a: str, b: str, p_col: str = "labelshuffle_p") -> float:
    rows = comparison_stats[(comparison_stats["metric"] == metric) & (comparison_stats["group_a"] == a) & (comparison_stats["group_b"] == b)]
    if rows.empty or p_col not in rows.columns:
        return np.nan
    return float(rows.iloc[0][p_col])


def merge_plot_metadata(bird: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    bird = bird.copy()
    bird["group"] = bird["group"].map(normalize_group_value)
    bird["plot_group"] = bird["group"]
    if args.metadata_excel is None:
        return bird
    meta = pd.read_excel(args.metadata_excel, sheet_name=args.metadata_sheet)
    bird_col = find_col(meta, args.metadata_bird_col, ["Animal ID", "bird", "animal_id", "Bird ID"], "metadata bird column")
    plot_col = find_col(meta, args.metadata_plot_col, ["Lesion hit type", "lesion_group", "Treatment type"], "metadata plotting column")
    m = meta[[bird_col, plot_col]].copy()
    m.columns = ["bird", "plot_raw"]
    m["bird"] = m["bird"].astype(str)
    m["plot_group"] = m["plot_raw"].map(normalize_plot_group_value)
    m = m.drop_duplicates("bird")
    bird = bird.drop(columns=["plot_group"], errors="ignore").merge(m[["bird", "plot_group"]], on="bird", how="left")
    bird["plot_group"] = bird["plot_group"].fillna(bird["group"])
    return bird


def pre_vs_post_scatter(bird: pd.DataFrame, base: str, out_dir: Path) -> None:
    pre_col = f"pre_{base}"
    post_col = f"post_{base}"
    label = PAPER_METRIC_LABELS[base]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    handles, labels = [], []
    present = set(bird["plot_group"])
    for pg in PAPER_SCATTER_ORDER:
        if pg not in present:
            continue
        g = bird[bird["plot_group"] == pg]
        h = ax.scatter(g[pre_col], g[post_col], s=34, alpha=0.85, color=PAPER_COLORS.get(pg, "0.4"), edgecolor="none")
        handles.append(h)
        labels.append(PAPER_SCATTER_LABELS.get(pg, pg))
    add_identity(ax, bird[pre_col].to_numpy(), bird[post_col].to_numpy())
    ax.set_xlabel(f"Late pre-lesion {label}", fontsize=13)
    ax.set_ylabel(f"Post-lesion {label}", fontsize=13)
    apply_paper_style(ax)
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_pre_vs_post_{safe_filename(base)}.png", dpi=450)
    fig.savefig(out_dir / f"paper_pre_vs_post_{safe_filename(base)}.pdf")
    plt.close(fig)


def paired_boxplot(bird: pd.DataFrame, group_stats: pd.DataFrame, base: str, out_dir: Path, group_p_col: str = "signflip_p") -> None:
    pre_col = f"pre_{base}"
    post_col = f"post_{base}"
    metric = f"delta_{base}"
    groups = [g for g in PAPER_GROUP_ORDER if g in set(bird["group"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    pre_positions, post_positions, xticks, xticklabels, all_vals = [], [], [], [], []
    for i, group in enumerate(groups):
        center = i * 2.2 + 1.0
        pre_pos, post_pos = center - 0.22, center + 0.22
        pre_positions.append(pre_pos)
        post_positions.append(post_pos)
        xticks.append(center)
        xticklabels.append(PAPER_GROUP_LABELS[group])
        color = PAPER_COLORS[group]
        g = bird[bird["group"] == group]
        pre_vals = g[pre_col].dropna().to_numpy(dtype=float)
        post_vals = g[post_col].dropna().to_numpy(dtype=float)
        all_vals.extend(pre_vals.tolist() + post_vals.tolist())
        bp = ax.boxplot([pre_vals, post_vals], positions=[pre_pos, post_pos], widths=0.32, patch_artist=True, showfliers=False,
                        medianprops={"color": "0.25", "linewidth": 1.2},
                        whiskerprops={"color": color, "linewidth": 1.2},
                        capprops={"color": color, "linewidth": 1.2})
        bp["boxes"][0].set(facecolor="white", edgecolor=color, linewidth=1.4)
        bp["boxes"][1].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.42)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_ylabel(PAPER_METRIC_LABELS[base][0].upper() + PAPER_METRIC_LABELS[base][1:], fontsize=13)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="0.25", label="Late Pre"), Patch(facecolor="0.55", edgecolor="0.55", alpha=0.42, label="Post")],
              frameon=False, fontsize=10, loc="upper left", ncol=2)
    apply_paper_style(ax)
    if all_vals:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.25 * yr)
        for i, group in enumerate(groups):
            g = bird[bird["group"] == group]
            vals = pd.concat([g[pre_col], g[post_col]], ignore_index=True).dropna().to_numpy(float)
            if len(vals):
                add_bracket(ax, pre_positions[i], post_positions[i], float(np.nanmax(vals)) + 0.06 * yr, p_to_label(get_group_p(group_stats, metric, group, p_col=group_p_col)))
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_paired_box_{safe_filename(base)}.png", dpi=450)
    fig.savefig(out_dir / f"paper_paired_box_{safe_filename(base)}.pdf")
    plt.close(fig)


def delta_by_group(bird: pd.DataFrame, comparison_stats: pd.DataFrame, base: str, out_dir: Path, comparison_p_col: str = "labelshuffle_p") -> None:
    metric = f"delta_{base}"
    groups = [g for g in PAPER_GROUP_ORDER if g in set(bird["group"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rng = np.random.default_rng(123)
    positions = np.arange(1, len(groups) + 1)
    all_vals = []
    for pos, group in zip(positions, groups):
        color = PAPER_COLORS[group]
        vals = bird.loc[bird["group"] == group, metric].dropna().to_numpy(float)
        all_vals.extend(vals.tolist())
        bp = ax.boxplot([vals], positions=[pos], widths=0.48, patch_artist=True, showfliers=False,
                        medianprops={"color": "0.25", "linewidth": 1.2},
                        whiskerprops={"color": color, "linewidth": 1.2},
                        capprops={"color": color, "linewidth": 1.2})
        bp["boxes"][0].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.35)
        ax.scatter(np.full(len(vals), pos) + rng.normal(0, 0.045, len(vals)), vals, s=28, color=color, alpha=0.85, edgecolor="none", zorder=3)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="0.35")
    ax.set_xticks(positions)
    ax.set_xticklabels([PAPER_GROUP_LABELS[g] for g in groups], fontsize=11)
    ax.set_ylabel("Δ " + PAPER_METRIC_LABELS[base], fontsize=13)
    apply_paper_style(ax)
    if all_vals:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        # extra vertical room for 3 brackets
        ax.set_ylim(ymin - 0.18 * yr, ymax + 0.46 * yr)
        pair_defs = [
            ("sham", "lateral_only"),
            ("sham", "medial_lateral"),
            ("lateral_only", "medial_lateral"),
        ]
        heights = [ymax + 0.07 * yr, ymax + 0.18 * yr, ymax + 0.29 * yr]
        for (a, b), y in zip(pair_defs, heights):
            if a in groups and b in groups:
                x1, x2 = groups.index(a) + 1, groups.index(b) + 1
                p = get_comparison_p(comparison_stats, metric, a, b, p_col=comparison_p_col)
                add_bracket(ax, x1, x2, y, p_to_label(p))
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_delta_by_group_{safe_filename(metric)}.png", dpi=450)
    fig.savefig(out_dir / f"paper_delta_by_group_{safe_filename(metric)}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal-level-csv", required=True, type=Path)
    ap.add_argument("--group-stats-csv", required=True, type=Path)
    ap.add_argument("--comparison-stats-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--metadata-excel", type=Path, default=None)
    ap.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    ap.add_argument("--metadata-bird-col", default="Animal ID")
    ap.add_argument("--metadata-plot-col", default="Lesion hit type")
    ap.add_argument("--group-p-col", default="signflip_p",
                    help="Column to use for within-group pre/post p-values in group stats CSV (e.g., signflip_p, signflip_p_holm, signflip_p_fdr_bh)")
    ap.add_argument("--comparison-p-col", default="labelshuffle_p",
                    help="Column to use for between-group pairwise p-values in comparison stats CSV (e.g., labelshuffle_p, labelshuffle_p_holm, labelshuffle_p_fdr_bh)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bird = pd.read_csv(args.animal_level_csv)
    group_stats = pd.read_csv(args.group_stats_csv)
    comparison_stats = pd.read_csv(args.comparison_stats_csv)
    bird = merge_plot_metadata(bird, args)

    print("[INFO] Birds by statistical group:")
    print(bird.groupby("group").size().to_string())
    print("[INFO] Birds by plotting subgroup:")
    print(bird.groupby("plot_group").size().to_string())

    for base in FOCUS_BASES:
        pre_vs_post_scatter(bird, base, args.out_dir)
        paired_boxplot(bird, group_stats, base, args.out_dir, group_p_col=args.group_p_col)
        delta_by_group(bird, comparison_stats, base, args.out_dir, comparison_p_col=args.comparison_p_col)

    print("[DONE] Wrote paper-style figures to:", args.out_dir)


if __name__ == "__main__":
    main()
