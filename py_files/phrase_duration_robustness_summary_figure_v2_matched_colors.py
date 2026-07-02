#!/usr/bin/env python3
"""
phrase_duration_robustness_summary_figure_v2_matched_colors.py

Updated robustness summary plotting script with lesion-group colors chosen to
match the manuscript Figure 3 / Figure 4 palette as closely as possible.

Outputs
-------
1) robustness_pooledML_delta_summary_matchedcolors.png/.pdf
2) robustness_pooledML_pre_post_dumbbell_matchedcolors.png/.pdf
3) robustness_group_comparison_delta_summary_matchedcolors.png/.pdf

Color defaults
--------------
sham saline injection                  -> teal/green
Lateral lesion only                    -> light purple
Partial Medial and Lateral lesion      -> medium purple
Complete Medial and Lateral lesion     -> dark purple
Complete+partial medial+lateral pooled -> dark charcoal by default
                                         (can switch to purple with --pooled-ml-color purple)

The pooled M+L default is charcoal because that is visually close to the pooled
group styling in Figure 3E, while the subgroup colors are preserved from
Figure 3A / Figure 4-style lesion hit types.
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

# Group labels used in the robustness summary tables.
SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

# Ordered methods for x-axis.
METHOD_ORDER = [
    "Post-selected top 20%",
    "Post-selected top 30%",
    "Post-selected top 40%",
    "Pooled pre+post-selected top 30%",
    "Pre-selected top 30% validation",
]

METHOD_SHORT_LABELS = {
    "Post-selected top 20%": "Post-lesion\nTop 20%",
    "Post-selected top 30%": "Post-lesion\nTop 30%",
    "Post-selected top 40%": "Post-lesion\nTop 40%",
    "Pooled pre+post-selected top 30%": "Pooled\nPre-lesion+Post-lesion\nTop 30%",
    "Pre-selected top 30% validation": "Late Pre-lesion\nTop 30%\nvalidation",
}

# Approximate manuscript-matched palette.
# If you want exact manuscript hex values, replace these with the values used in your
# Figure 3 / Figure 4 scripts.
COLOR_MAP = {
    SHAM: "#1FA187",         # teal/green
    LATERAL: "#B39DDB",      # light purple
    PARTIAL_ML: "#7E57C2",   # medium purple
    COMPLETE_ML: "#4A148C",  # dark purple
    POOLED_ML: "#4D4D4D",    # charcoal (Figure 3E-like pooled ML line)
}

POOLED_ML_PURPLE = "#5E35B1"  # alternate pooled group color if you want pooled ML purple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make robustness summary figures with manuscript-matched lesion colors.")
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
        help="Use charcoal (Figure 3E-like) or purple for the pooled medial+lateral group.",
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


def _format_p(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "NA"
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


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
    ax.set_xticklabels([METHOD_SHORT_LABELS.get(m, m) for m in methods])


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


def plot_pooled_ml_delta(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, color_map: dict[str, str], show: bool = False) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    if pooled.empty:
        raise ValueError(f"No rows found for pooled M+L group: {POOLED_ML}")

    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    y = pooled["median_SD_delta_ms"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, y, marker="o", markersize=7, linewidth=1.8, color=color_map[POOLED_ML])
    ax.axhline(0, linewidth=1.0, color="0.4")

    ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0
    ymin = np.nanmin(y) if np.isfinite(y).any() else -1.0
    yrange = ymax - ymin
    if not np.isfinite(yrange) or yrange <= 0:
        yrange = max(abs(ymax), 1.0)
    pad = 0.30 * yrange
    ax.set_ylim(ymin - 0.15 * yrange, ymax + pad)

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median Δ phrase-duration SD\n(post-lesion - late pre-lesion, ms)")
    _apply_clean_axes(ax)

    # Compact significance annotations.
    for i, (_, row) in enumerate(pooled.iterrows()):
        paired_p = row.get("paired_p_post_gt_pre", np.nan)
        welch_p = row.get("welch_p_delta_gt_sham", np.nan)
        label = f"paired {_sig_label(paired_p)}\nvs sham {_sig_label(welch_p)}"
        ax.text(i, y[i] + 0.05 * yrange, label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    png = out_dir / f"{prefix}_pooledML_delta_summary_matchedcolors.png"
    pdf = out_dir / f"{prefix}_pooledML_delta_summary_matchedcolors.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote:\n  {png}\n  {pdf}")


def plot_pooled_ml_pre_post_dumbbell(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, color_map: dict[str, str], show: bool = False) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    if pooled.empty:
        raise ValueError(f"No rows found for pooled M+L group: {POOLED_ML}")

    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    pre = pooled["pre_median_SD_ms"].to_numpy(dtype=float)
    post = pooled["post_median_SD_ms"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    color = color_map[POOLED_ML]

    for i in range(len(x)):
        ax.plot([x[i], x[i]], [pre[i], post[i]], linewidth=1.6, color=color, alpha=0.8)

    # Pre = open marker; Post = filled marker
    ax.scatter(x, pre, s=60, facecolors="white", edgecolors=color, linewidths=1.5, label="Late Pre-lesion")
    ax.scatter(x, post, s=60, facecolors=color, edgecolors=color, linewidths=1.2, label="Post-lesion")

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median phrase-duration SD (ms)")
    _apply_clean_axes(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    png = out_dir / f"{prefix}_pooledML_pre_post_dumbbell_matchedcolors.png"
    pdf = out_dir / f"{prefix}_pooledML_pre_post_dumbbell_matchedcolors.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote:\n  {png}\n  {pdf}")


def plot_group_comparison_delta(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, color_map: dict[str, str], show: bool = False) -> None:
    keep_groups = [SHAM, LATERAL, POOLED_ML]
    sub = df[df["lesion_group"].astype(str).isin(keep_groups)].copy()
    if sub.empty:
        raise ValueError("No sham/lateral/pooled M+L rows found in summary table.")

    methods = [m for m in METHOD_ORDER if m in set(sub["selection_method"].astype(str))]
    if not methods:
        methods = list(dict.fromkeys(sub["selection_method"].astype(str).tolist()))

    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    for group in keep_groups:
        g = sub[sub["lesion_group"].astype(str) == group].copy()
        g["selection_method"] = g["selection_method"].astype(str)
        g = g.set_index("selection_method").reindex(methods).reset_index()
        y = g["median_SD_delta_ms"].to_numpy(dtype=float)
        x = np.arange(len(methods))
        ax.plot(
            x, y,
            marker="o",
            markersize=7,
            linewidth=1.8,
            label=group,
            color=color_map[group],
        )

    ax.axhline(0, linewidth=1.0, color="0.4")
    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median Δ phrase-duration SD\n(post-lesion - late pre-lesion, ms)")
    _apply_clean_axes(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    png = out_dir / f"{prefix}_group_comparison_delta_summary_matchedcolors.png"
    pdf = out_dir / f"{prefix}_group_comparison_delta_summary_matchedcolors.pdf"
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

    plot_pooled_ml_delta(df, out_dir, args.prefix, args.dpi, color_map, show=args.show)
    plot_pooled_ml_pre_post_dumbbell(df, out_dir, args.prefix, args.dpi, color_map, show=args.show)
    plot_group_comparison_delta(df, out_dir, args.prefix, args.dpi, color_map, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
