#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


GROUP_ORDER = [
    "sham saline injection",
    "Area X visible (single hit)",
    "Medial/Lateral visible + large lesion",
]

DISPLAY_LABELS: Dict[str, str] = {
    "sham saline injection": "sham saline injection",
    "Area X visible (single hit)": "Lateral lesion only",
    "Medial/Lateral visible + large lesion": "Complete and Partial Medial and Lateral lesion",
}

GROUP_COLORS: Dict[str, str] = {
    "sham saline injection": "#ff2d2d",
    "Area X visible (single hit)": "#c4b5fd",
    "Medial/Lateral visible + large lesion": "#6a3d9a",
}

SECONDARY_GROUP_COLORS: Dict[str, str] = {
    "Area X visible (Medial + Lateral)": "#6a3d9a",
    "Large lesion / Area X not visible": "#000000",
}


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h + max(0.01, h * 0.4), text, ha="center", va="bottom", fontsize=14, clip_on=False)


def _resolve_color(group_name: str) -> str:
    if group_name in GROUP_COLORS:
        return GROUP_COLORS[group_name]
    if group_name in SECONDARY_GROUP_COLORS:
        return SECONDARY_GROUP_COLORS[group_name]
    return "#666666"


def _normalize_group_label(s: str) -> str:
    x = str(s).strip()
    mapping = {
        "Area X visible (Lateral only)": "Area X visible (single hit)",
        "Area X visible (single hit)": "Area X visible (single hit)",
        "Lateral lesion only": "Area X visible (single hit)",
        "Area X visible (Medial + Lateral)": "Medial/Lateral visible + large lesion",
        "Large lesion / Area X not visible": "Medial/Lateral visible + large lesion",
        "Medial/Lateral visible + large lesion": "Medial/Lateral visible + large lesion",
        "Complete and Partial Medial and Lateral lesion": "Medial/Lateral visible + large lesion",
        "sham saline injection": "sham saline injection",
    }
    return mapping.get(x, x)


def _default_legend_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_legend.png")


def save_color_legend_png(
    legend_out_path: str | Path,
    *,
    title: str = "Lesion hit type color coding",
    fontsize: float = 15.0,
    title_fontsize: float = 16.0,
) -> Path:
    legend_out_path = Path(legend_out_path)
    legend_out_path.parent.mkdir(parents=True, exist_ok=True)

    handles = [
        Line2D([0], [0], color=GROUP_COLORS["sham saline injection"], lw=4, label=DISPLAY_LABELS["sham saline injection"]),
        Line2D([0], [0], color=GROUP_COLORS["Area X visible (single hit)"], lw=4, label=DISPLAY_LABELS["Area X visible (single hit)"]),
        Line2D([0], [0], color=GROUP_COLORS["Medial/Lateral visible + large lesion"], lw=4, label=DISPLAY_LABELS["Medial/Lateral visible + large lesion"]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=1,
        fontsize=fontsize,
        title=title,
        title_fontsize=title_fontsize,
        handlelength=2.8,
    )
    fig.savefig(legend_out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return legend_out_path


def make_boxplots_only(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    dataset_level: str = "cluster",
    group_col: str = "lesion_hit_type",
    pre_col: str = "bc_pre_early_vs_late_equal_groups",
    post_col: str = "bc_post_early_vs_late_equal_groups",
    title: str = "bc_early_late_equal_groups: pre vs post within lesion hit type",
    y_label: str = "Bhattacharyya coefficient (early vs late, each equal sample size)",
    showfliers: bool = False,
    figure_width: float = 16.0,
    figure_height: float = 7.0,
    title_fontsize: float = 18.0,
    axis_label_fontsize: float = 16.0,
    tick_label_fontsize: float = 14.0,
    panel_title_fontsize: float = 16.0,
) -> Path:
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    df = pd.read_csv(csv_path).copy()

    if group_col not in df.columns:
        raise ValueError(f"group column not found: {group_col}")
    if pre_col not in df.columns:
        raise ValueError(f"pre column not found: {pre_col}")
    if post_col not in df.columns:
        raise ValueError(f"post column not found: {post_col}")

    df[group_col] = df[group_col].astype(str).map(_normalize_group_label)
    df[pre_col] = pd.to_numeric(df[pre_col], errors="coerce")
    df[post_col] = pd.to_numeric(df[post_col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(figure_width, figure_height), sharey=True)
    fig.suptitle(
        f"{title} ({'all clusters' if dataset_level == 'cluster' else 'bird means'})",
        fontsize=title_fontsize,
        y=0.98,
    )

    overall_max = 0.0
    for g in GROUP_ORDER:
        sub = df[df[group_col] == g].copy()
        vals = np.concatenate([
            sub[pre_col].dropna().to_numpy(dtype=float),
            sub[post_col].dropna().to_numpy(dtype=float),
        ]) if len(sub) else np.array([], dtype=float)
        if vals.size:
            overall_max = max(overall_max, float(np.nanmax(vals)))

    for ax, g in zip(axes, GROUP_ORDER):
        sub = df[df[group_col] == g].copy()
        pre = sub[pre_col].dropna().to_numpy(dtype=float)
        post = sub[post_col].dropna().to_numpy(dtype=float)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(DISPLAY_LABELS.get(g, g), fontsize=panel_title_fontsize)

        if pre.size == 0 and post.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        bp = ax.boxplot(
            [pre, post],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=showfliers,
        )

        color = _resolve_color(g)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_linewidth(1.6)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.8)
        for whisker in bp["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.4)
        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.4)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre", "Post"], fontsize=tick_label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_label_fontsize)

        ymax = max(
            overall_max if np.isfinite(overall_max) else 1.0,
            float(np.nanmax(pre)) if pre.size else 0.0,
            float(np.nanmax(post)) if post.size else 0.0,
            1.0,
        )
        ax.set_ylim(0.4, min(1.25, ymax + 0.18))

        if HAVE_SCIPY and pre.size >= 2 and post.size >= 2:
            try:
                if dataset_level == "bird":
                    common = sub.dropna(subset=[pre_col, post_col]).copy()
                    _, p = stats.wilcoxon(
                        common[pre_col].to_numpy(dtype=float),
                        common[post_col].to_numpy(dtype=float),
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                else:
                    _, p = stats.mannwhitneyu(pre, post, alternative="two-sided")
                stars = p_to_stars(float(p))
                y = min(1.18, max(np.nanmax(pre), np.nanmax(post)) + 0.04)
                add_sig_bracket(ax, 1, 2, y, 0.025, stars)
            except Exception:
                pass

    axes[0].set_ylabel(y_label, fontsize=axis_label_fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make Bhattacharyya coefficient boxplots only from an existing CSV.")
    p.add_argument("--csv-path", required=True, type=str)
    p.add_argument("--out-path", required=True, type=str)
    p.add_argument("--legend-out-path", default=None, type=str)
    p.add_argument("--dataset-level", choices=["cluster", "bird"], default="cluster")
    p.add_argument("--group-col", default="lesion_hit_type", type=str)
    p.add_argument("--pre-col", default="bc_pre_early_vs_late_equal_groups", type=str)
    p.add_argument("--post-col", default="bc_post_early_vs_late_equal_groups", type=str)
    p.add_argument("--title", default="bc_early_late_equal_groups: pre vs post within lesion hit type", type=str)
    p.add_argument("--y-label", default="Bhattacharyya coefficient (early vs late, each equal sample size)", type=str)
    p.add_argument("--showfliers", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_path = Path(args.out_path)
    legend_out_path = Path(args.legend_out_path) if args.legend_out_path else _default_legend_path(out_path)

    make_boxplots_only(
        csv_path=args.csv_path,
        out_path=out_path,
        dataset_level=args.dataset_level,
        group_col=args.group_col,
        pre_col=args.pre_col,
        post_col=args.post_col,
        title=args.title,
        y_label=args.y_label,
        showfliers=args.showfliers,
    )
    save_color_legend_png(legend_out_path)

    print(f"Saved boxplot figure to: {out_path}")
    print(f"Saved legend figure to: {legend_out_path}")


if __name__ == "__main__":
    main()
