#!/usr/bin/env python3
"""
phrase_duration_robustness_summary_figure_v1.py

Generate supplemental robustness summary figures from a phrase-duration
robustness summary table.

Designed for the AFP_lesion_paper robustness outputs.

Inputs
------
A CSV table with columns like:
    selection_method
    lesion_group
    n_animals
    pre_median_SD_ms
    post_median_SD_ms
    median_SD_delta_ms
    paired_p_post_gt_pre
    welch_p_delta_gt_sham
    interpretation

Compatible with:
    phrase_duration_robustness_summary_table_with_pooledML.csv
or
    phrase_duration_robustness_summary_table_with_pooledML_and_pooled_selection.csv

Outputs
-------
1) robustness_pooledML_delta_summary.png/.pdf
   - pooled medial+lateral group only
   - x-axis = selection method
   - y-axis = median ΔSD (post - pre)
   - text annotation with paired p and Welch-vs-sham p

2) robustness_pooledML_pre_post_summary.png/.pdf
   - pooled medial+lateral group only
   - x-axis = selection method
   - y-axis = median SD
   - separate pre and post lines

3) robustness_group_comparison_delta_summary.png/.pdf
   - sham, lateral-only, pooled medial+lateral
   - x-axis = selection method
   - y-axis = median ΔSD (post - pre)

The script uses matplotlib defaults (no custom colors required).
"""

from __future__ import annotations

import argparse
import math
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

POOLED_ML = "Complete and partial medial and lateral lesion"
SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"

METHOD_ORDER = [
    "Post-selected top 20%",
    "Post-selected top 30%",
    "Post-selected top 40%",
    "Pooled pre+post-selected top 30%",
    "Pre-selected top 30% validation",
]

METHOD_SHORT_LABELS = {
    "Post-selected top 20%": "Post\nTop 20%",
    "Post-selected top 30%": "Post\nTop 30%",
    "Post-selected top 40%": "Post\nTop 40%",
    "Pooled pre+post-selected top 30%": "Pooled\nPre+Post\nTop 30%",
    "Pre-selected top 30% validation": "Pre\nTop 30%\nvalidation",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make robustness summary figures from summary CSV.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root robustness folder.")
    p.add_argument("--summary-csv", type=Path, default=None, help="Path to summary CSV. If omitted, auto-detect in --root.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: same folder as summary CSV.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--prefix", type=str, default="robustness")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
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


def plot_pooled_ml_delta(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, show: bool = False) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    if pooled.empty:
        raise ValueError(f"No rows found for pooled M+L group: {POOLED_ML}")

    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    y = pooled["median_SD_delta_ms"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.axhline(0, linewidth=1)

    ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0
    ymin = np.nanmin(y) if np.isfinite(y).any() else -1.0
    yrange = ymax - ymin
    if not np.isfinite(yrange) or yrange <= 0:
        yrange = max(abs(ymax), 1.0)
    pad = 0.28 * yrange
    ax.set_ylim(ymin - 0.15 * yrange, ymax + pad)

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median ΔSD (post - pre, ms)")
    ax.set_title("Robustness summary: pooled medial+lateral group")

    for i, (_, row) in enumerate(pooled.iterrows()):
        paired_p = row.get("paired_p_post_gt_pre", np.nan)
        welch_p = row.get("welch_p_delta_gt_sham", np.nan)
        label = (
            f"paired {_sig_label(paired_p)}\n"
            f"p={_format_p(paired_p)}\n"
            f"vs sham {_sig_label(welch_p)}\n"
            f"p={_format_p(welch_p)}"
        )
        ax.text(i, y[i] + 0.05 * yrange, label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    png = out_dir / f"{prefix}_pooledML_delta_summary.png"
    pdf = out_dir / f"{prefix}_pooledML_delta_summary.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote:\n  {png}\n  {pdf}")


def plot_pooled_ml_pre_post(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, show: bool = False) -> None:
    pooled = df[df["lesion_group"].astype(str) == POOLED_ML].copy()
    if pooled.empty:
        raise ValueError(f"No rows found for pooled M+L group: {POOLED_ML}")

    methods = pooled["selection_method"].astype(str).tolist()
    x = np.arange(len(methods))
    pre = pooled["pre_median_SD_ms"].to_numpy(dtype=float)
    post = pooled["post_median_SD_ms"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, pre, marker="o", linewidth=1.5, label="Pre median SD")
    ax.plot(x, post, marker="o", linewidth=1.5, label="Post median SD")

    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median SD (ms)")
    ax.set_title("Pooled medial+lateral group: pre vs post medians")
    ax.legend()

    fig.tight_layout()
    png = out_dir / f"{prefix}_pooledML_pre_post_summary.png"
    pdf = out_dir / f"{prefix}_pooledML_pre_post_summary.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote:\n  {png}\n  {pdf}")


def plot_group_comparison_delta(df: pd.DataFrame, out_dir: Path, prefix: str, dpi: int, show: bool = False) -> None:
    keep_groups = [SHAM, LATERAL, POOLED_ML]
    sub = df[df["lesion_group"].astype(str).isin(keep_groups)].copy()
    if sub.empty:
        raise ValueError("No sham/lateral/pooled M+L rows found in summary table.")

    methods = [m for m in METHOD_ORDER if m in set(sub["selection_method"].astype(str))]
    if not methods:
        methods = list(dict.fromkeys(sub["selection_method"].astype(str).tolist()))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for group in keep_groups:
        g = sub[sub["lesion_group"].astype(str) == group].copy()
        g["selection_method"] = g["selection_method"].astype(str)
        g = g.set_index("selection_method").reindex(methods).reset_index()
        y = g["median_SD_delta_ms"].to_numpy(dtype=float)
        x = np.arange(len(methods))
        ax.plot(x, y, marker="o", linewidth=1.5, label=group)

    ax.axhline(0, linewidth=1)
    _set_method_ticks(ax, methods)
    ax.set_ylabel("Median ΔSD (post - pre, ms)")
    ax.set_title("Robustness summary by lesion group")
    ax.legend()

    fig.tight_layout()
    png = out_dir / f"{prefix}_group_comparison_delta_summary.png"
    pdf = out_dir / f"{prefix}_group_comparison_delta_summary.pdf"
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

    plot_pooled_ml_delta(df, out_dir, args.prefix, args.dpi, show=args.show)
    plot_pooled_ml_pre_post(df, out_dir, args.prefix, args.dpi, show=args.show)
    plot_group_comparison_delta(df, out_dir, args.prefix, args.dpi, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
