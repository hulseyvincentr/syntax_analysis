#!/usr/bin/env python3
"""
phrase_duration_robustness_summary_figure_v3_selected_labels.py

Updated robustness summary plotting script with:
- lesion-group colors matched to the Figure 3 / Figure 4 palette
- "selected" wording in the x-axis labels
- "late pre-lesion" and "post-lesion" wording
- cleaner manuscript-style formatting
- a combined 2-panel supplemental-style figure:
    Panel A: group comparison of median Δ phrase-duration SD
    Panel B: pooled medial+lateral late pre-lesion vs post-lesion dumbbell plot

It also writes separate panel images for convenience.

Output files
------------
Combined figure:
    robustness_summary_selectedlabels_2panel.png/.pdf

Separate panels:
    robustness_group_comparison_selectedlabels.png/.pdf
    robustness_pooledML_dumbbell_selectedlabels.png/.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_robustness_2026-07-01")
DEFAULT_CSV_CANDIDATES = [
    "phrase_duration_robustness_summary_table_with_pooledML_and_pooled_selection.csv",
    "phrase_duration_robustness_summary_table_with_pooledML.csv",
    "phrase_duration_robustness_summary_table.csv",
]

# Group labels used in the summary table
SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

# Ordered methods for x-axis
METHOD_ORDER = [
    "Post-selected top 20%",
    "Post-selected top 30%",
    "Post-selected top 40%",
    "Pooled pre+post-selected top 30%",
    "Pre-selected top 30% validation",
]

# Short x-axis labels designed to emphasize "selected"
METHOD_SHORT_LABELS = {
    "Post-selected top 20%": "Post-lesion\nselected\nTop 20%",
    "Post-selected top 30%": "Post-lesion\nselected\nTop 30%",
    "Post-selected top 40%": "Post-lesion\nselected\nTop 40%",
    "Pooled pre+post-selected top 30%": "Pooled\npre-lesion+\npost-lesion\nselected\nTop 30%",
    "Pre-selected top 30% validation": "Late Pre-lesion\nselected\nTop 30%\nvalidation",
}

# Approximate manuscript-matched palette
COLOR_MAP = {
    SHAM: "#1FA187",         # teal/green
    LATERAL: "#B39DDB",      # light purple
    PARTIAL_ML: "#7E57C2",   # medium purple
    COMPLETE_ML: "#4A148C",  # dark purple
    POOLED_ML: "#4D4D4D",    # charcoal pooled line, close to Figure 3E pooled group
}

POOLED_ML_PURPLE = "#5E35B1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make robustness summary figures with selected wording and manuscript-matched colors."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root robustness folder.")
    p.add_argument("--summary-csv", type=Path, default=None, help="Path to summary CSV. If omitted, auto-detect in --root.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: same folder as summary CSV.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--prefix", type=str, default="robustness")
    p.add_argument("--show", action="store_true", help="Show figures interactively.")
    p.add_argument(
        "--pooled-ml-color",
        choices=["charcoal", "purple"],
        default="charcoal",
        help="Use charcoal or purple for the pooled medial+lateral group.",
    )
    return p.parse_args()


def _find_summary_csv(root: Path) -> Path:
    for name in DEFAULT_CSV_CANDIDATES:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a robustness summary CSV. Tried:\n  "
        + "\n  ".join(str(root / n) for n in DEFAULT_CSV_CANDIDATES)
    )


def _normalize_method_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selection_method"] = df["selection_method"].astype(str)

    ordered = [m for m in METHOD_ORDER if m in set(df["selection_method"])]
    extras = [m for m in sorted(df["selection_method"].unique()) if m not in ordered]
    ordered = ordered + extras

    df["_method_rank"] = df["selection_method"].map({m: i for i, m in enumerate(ordered)})
    df = df.sort_values(["_method_rank", "selection_method"]).drop(columns="_method_rank")
    return df


def _sig_label(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "n.s."
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _set_method_ticks(ax, methods: Iterable[str]) -> None:
    methods = list(methods)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_SHORT_LABELS.get(m, m) for m in methods], fontsize=10)


def _apply_clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4, width=1)
    ax.grid(False)


def _get_color_map(pooled_ml_color: str) -> dict[str, str]:
    cmap = dict(COLOR_MAP)
    if pooled_ml_color == "purple":
        cmap[POOLED_ML] = POOLED_ML_PURPLE
    return cmap


def load_summary_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {
        "selection_method",
        "lesion_group",
        "pre_median_SD_ms",
        "post_median_SD_ms",
        "median_SD_delta_ms",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {sorted(missing)}")
    return _normalize_method_order(df)


def _subset_group(df: pd.DataFrame, group: str, methods: list[str]) -> pd.DataFrame:
    g = df[df["lesion_group"].astype(str) == group].copy()
    g["selection_method"] = g["selection_method"].astype(str)
    g = g.set_index("selection_method").reindex(methods).reset_index()
    return g


def plot_group_comparison_panel(ax, df: pd.DataFrame, color_map: dict[str, str]) -> None:
    keep_groups = [SHAM, LATERAL, POOLED_ML]
    sub = df[df["lesion_group"].astype(str).isin(keep_groups)].copy()
    methods = [m for m in METHOD_ORDER if m in set(sub["selection_method"].astype(str))]
    x = np.arange(len(methods))

    # Connect only the first 3 "post-lesion-selected" threshold points.
    post_idx = np.array([0, 1, 2], dtype=int)

    for group in keep_groups:
        g = _subset_group(sub, group, methods)
        y = g["median_SD_delta_ms"].to_numpy(dtype=float)

        # first three threshold-sensitivity points connected
        ax.plot(
            x[post_idx], y[post_idx],
            marker="o",
            markersize=7,
            linewidth=1.8,
            color=color_map[group],
            label=group,
        )

        # the two alternative selection methods shown as separate points
        for idx in [3, 4]:
            if idx < len(x):
                ax.scatter([x[idx]], [y[idx]], s=55, color=color_map[group], zorder=3)

    ax.axhline(0, linewidth=1.0, color="0.4")
    if len(x) >= 5:
        ax.axvline(2.5, linewidth=0.8, color="0.8", linestyle="--")

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median Δ phrase-duration SD\n(post-lesion - late pre-lesion, ms)")
    _apply_clean_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # Small group headers to help explain the x-axis logic.
    if len(x) >= 5:
        ylim = ax.get_ylim()
        ytxt = ylim[1] - 0.03 * (ylim[1] - ylim[0])
        ax.text(1.0, ytxt, "threshold sensitivity", ha="center", va="top", fontsize=9)
        ax.text(3.5, ytxt, "alternative selection checks", ha="center", va="top", fontsize=9)


def plot_pooled_ml_dumbbell_panel(ax, df: pd.DataFrame, color_map: dict[str, str]) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    pre = pooled["pre_median_SD_ms"].to_numpy(dtype=float)
    post = pooled["post_median_SD_ms"].to_numpy(dtype=float)

    color = color_map[POOLED_ML]
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [pre[i], post[i]], linewidth=1.6, color=color, alpha=0.8)

    ax.scatter(x, pre, s=65, facecolors="white", edgecolors=color, linewidths=1.7, label="Late Pre-lesion")
    ax.scatter(x, post, s=65, facecolors=color, edgecolors=color, linewidths=1.2, label="Post-lesion")

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median phrase-duration SD (ms)")
    _apply_clean_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")


def plot_pooled_ml_delta_panel(ax, df: pd.DataFrame, color_map: dict[str, str]) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    y = pooled["median_SD_delta_ms"].to_numpy(dtype=float)

    ax.plot(x[:3], y[:3], marker="o", markersize=7, linewidth=1.8, color=color_map[POOLED_ML])
    for idx in [3, 4]:
        if idx < len(x):
            ax.scatter([x[idx]], [y[idx]], s=55, color=color_map[POOLED_ML], zorder=3)

    ax.axhline(0, linewidth=1.0, color="0.4")
    if len(x) >= 5:
        ax.axvline(2.5, linewidth=0.8, color="0.8", linestyle="--")

    ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0
    ymin = np.nanmin(y) if np.isfinite(y).any() else -1.0
    yrange = ymax - ymin
    if not np.isfinite(yrange) or yrange <= 0:
        yrange = max(abs(ymax), 1.0)
    pad = 0.28 * yrange
    ax.set_ylim(ymin - 0.12 * yrange, ymax + pad)

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median Δ phrase-duration SD\n(post-lesion - late pre-lesion, ms)")
    _apply_clean_axes(ax)

    for i, (_, row) in enumerate(pooled.iterrows()):
        paired_p = row.get("paired_p_post_gt_pre", np.nan)
        welch_p = row.get("welch_p_delta_gt_sham", np.nan)
        label = f"paired {_sig_label(paired_p)}\nvs sham {_sig_label(welch_p)}"
        ax.text(i, y[i] + 0.05 * yrange, label, ha="center", va="bottom", fontsize=8)


def save_separate_panels(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, color_map: dict[str, str], show: bool = False) -> None:
    figA, axA = plt.subplots(figsize=(8.9, 4.9))
    plot_group_comparison_panel(axA, df, color_map)
    figA.tight_layout()
    pngA = out_dir / f"{prefix}_group_comparison_selectedlabels.png"
    pdfA = out_dir / f"{prefix}_group_comparison_selectedlabels.pdf"
    figA.savefig(pngA, dpi=dpi, bbox_inches="tight")
    figA.savefig(pdfA, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figA)

    figB, axB = plt.subplots(figsize=(8.6, 4.8))
    plot_pooled_ml_dumbbell_panel(axB, df, color_map)
    figB.tight_layout()
    pngB = out_dir / f"{prefix}_pooledML_dumbbell_selectedlabels.png"
    pdfB = out_dir / f"{prefix}_pooledML_dumbbell_selectedlabels.pdf"
    figB.savefig(pngB, dpi=dpi, bbox_inches="tight")
    figB.savefig(pdfB, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figB)

    # Also write the pooled ML delta panel, in case the user still wants it.
    figC, axC = plt.subplots(figsize=(8.6, 4.8))
    plot_pooled_ml_delta_panel(axC, df, color_map)
    figC.tight_layout()
    pngC = out_dir / f"{prefix}_pooledML_delta_selectedlabels.png"
    pdfC = out_dir / f"{prefix}_pooledML_delta_selectedlabels.pdf"
    figC.savefig(pngC, dpi=dpi, bbox_inches="tight")
    figC.savefig(pdfC, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figC)

    print(f"Wrote:\n  {pngA}\n  {pdfA}\n  {pngB}\n  {pdfB}\n  {pngC}\n  {pdfC}")


def save_combined_two_panel(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, color_map: dict[str, str], show: bool = False) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 9.5), constrained_layout=True)

    plot_group_comparison_panel(axes[0], df, color_map)
    plot_pooled_ml_dumbbell_panel(axes[1], df, color_map)

    axes[0].text(-0.08, 1.03, "A", transform=axes[0].transAxes, fontsize=16, fontweight="bold", va="top")
    axes[1].text(-0.08, 1.03, "B", transform=axes[1].transAxes, fontsize=16, fontweight="bold", va="top")

    png = out_dir / f"{prefix}_summary_selectedlabels_2panel.png"
    pdf = out_dir / f"{prefix}_summary_selectedlabels_2panel.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Wrote:\n  {png}\n  {pdf}")


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve() if args.summary_csv else _find_summary_csv(root)
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using summary CSV:\n  {summary_csv}")
    print(f"Writing plots to:\n  {out_dir}")

    df = load_summary_table(summary_csv)
    color_map = _get_color_map(args.pooled_ml_color)

    save_combined_two_panel(df, out_dir, args.prefix, args.dpi, color_map, show=args.show)
    save_separate_panels(df, out_dir, args.prefix, args.dpi, color_map, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
