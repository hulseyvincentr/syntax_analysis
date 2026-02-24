#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_graphs.py

Plot and test pre vs post Levina–Bickel intrinsic dimensionality outputs
produced by Levina_Bickel_pre_vs_post_treatment_v5.py.

Input CSV (expected columns; your file matches this):
    animal_id, npz_path, hit_type, group, treatment_date,
    cluster_label, k, n_pre, n_post, pre_dim, post_dim, delta_dim

Outputs (saved to --out-dir):
  1) pre_vs_post_dim_overlay.png
     Overlay scatter: x=pre_dim, y=post_dim, colored by hit_type, with y=x line.

  2) delta_dim_boxplot_by_group.png
     Boxplot of delta_dim by 3 groups:
        - sham saline injection
        - Area X visible (single hit)
        - Combined (visible ML + not visible)
     Includes significance brackets for:
        - visible_single vs sham
        - combined vs sham
     (Holm-corrected by default)

  3) stats_report.txt
     - Omnibus (Kruskal–Wallis) across the 3 groups (cluster-level rows)
     - Pairwise MWU tests with Holm correction (cluster-level rows)
     - Sensitivity: same tests on per-bird mean delta_dim (bird-level aggregation)

Notes
-----
• Cluster rows are not fully independent within a bird, so the bird-level analysis is
  a useful sensitivity check.
• By default we filter out clusters with too few points in either period, since
  small-n clusters can yield noisy ID estimates.

Run example
-----------
conda activate syntax_analysis
cd ~/Documents/allPythonCode/syntax_analysis/py_files/

python Levina_Bickel_graphs.py \
  --csv "/Volumes/my_own_SSD/updated_AreaX_outputs/lb_pre_post_dimensionality_k15/lb_pre_post_cluster_dimensionality_k15_20260224_115517.csv" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/lb_pre_post_dimensionality_k15/graphs" \
  --min-pre 200 --min-post 200
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for stats
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -----------------------------
# Canonical strings + colors
# -----------------------------
HT_SHAM = "sham saline injection"
HT_VISIBLE_SINGLE = "Area X visible (single hit)"
HT_VISIBLE_ML = "Area X visible (medial+lateral hit)"
HT_NOT_VISIBLE = "large lesion Area X not visible"
GROUP_COMBINED = "Combined (visible ML + not visible)"

GROUP_ORDER = [HT_SHAM, HT_VISIBLE_SINGLE, GROUP_COMBINED]

# Your preferred palette from the screenshot (approx)
HIT_TYPE_COLORS = {
    HT_VISIBLE_ML: "#7B4BB7",       # dark purple
    HT_VISIBLE_SINGLE: "#C9B6E4",   # light purple
    HT_NOT_VISIBLE: "#000000",      # black
    HT_SHAM: "#E31A1C",             # red
}

GROUP_COLORS = {
    HT_SHAM: HIT_TYPE_COLORS[HT_SHAM],
    HT_VISIBLE_SINGLE: HIT_TYPE_COLORS[HT_VISIBLE_SINGLE],
    GROUP_COMBINED: "#4B4B4B",  # distinct dark gray
}


# -----------------------------
# Utilities
# -----------------------------
def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            f"  missing: {missing}\n"
            f"  present: {list(df.columns)}"
        )


def _holm_adjust(pvals: List[float]) -> List[float]:
    """
    Holm–Bonferroni correction (step-down).
    Returns adjusted p-values in original order.
    """
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    prev = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, p[idx] * factor)
        val = max(val, prev)  # ensure non-decreasing in sorted order
        adj[idx] = val
        prev = val
    return adj.tolist()


def _p_to_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_sig_bracket(ax, x1: float, x2: float, y: float, text: str, dh_frac: float = 0.02, barh_frac: float = 0.01):
    """Draw a bracket between x1 and x2 at height y (data coords)."""
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    h = barh_frac * yr
    d = dh_frac * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
    ax.text((x1 + x2) / 2, y + h + d, text, ha="center", va="bottom")


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    return arr[np.isfinite(arr)]


# -----------------------------
# Stats
# -----------------------------
def _kruskal_if_possible(groups: Dict[str, np.ndarray]) -> Optional[Tuple[float, float]]:
    if not _HAVE_SCIPY:
        return None
    xs = [v for v in groups.values() if v.size > 0]
    if len(xs) < 2:
        return None
    try:
        H, p = _scipy_stats.kruskal(*xs)
        return float(H), float(p)
    except Exception:
        return None


def _mwu(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> Tuple[float, float, int, int]:
    """
    Mann–Whitney U test. Returns (U, p, n_a, n_b).
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available; cannot run MWU tests.")
    a = _finite(a)
    b = _finite(b)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), int(a.size), int(b.size)
    res = _scipy_stats.mannwhitneyu(a, b, alternative=alternative)
    return float(res.statistic), float(res.pvalue), int(a.size), int(b.size)


# -----------------------------
# Plots
# -----------------------------
def plot_pre_post_overlay(df: pd.DataFrame, out_path: Path, pre_col: str, post_col: str, hit_type_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter per hit-type
    for ht, sub in df.groupby(hit_type_col):
        if ht not in HIT_TYPE_COLORS:
            continue
        ax.scatter(
            sub[pre_col].to_numpy(),
            sub[post_col].to_numpy(),
            s=22,
            alpha=0.85,
            c=HIT_TYPE_COLORS[ht],
            label=ht,
            edgecolors="none",
        )

    # y=x line
    x = df[pre_col].to_numpy()
    y = df[post_col].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.any():
        mn = float(min(x[m].min(), y[m].min()))
        mx = float(max(x[m].max(), y[m].max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, color="red")

    ax.set_xlabel("Pre intrinsic dimensionality (Levina–Bickel)")
    ax.set_ylabel("Post intrinsic dimensionality (Levina–Bickel)")
    ax.set_title("Pre vs Post intrinsic dimensionality (colored by lesion hit type)")

    ax.legend(title="Hit type", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_delta_boxplot_and_tests(
    df: pd.DataFrame,
    out_path: Path,
    stats_path: Path,
    group_col: str,
    delta_col: str,
    alternative: str = "two-sided",
) -> None:
    # Keep only the 3 groups we want
    keep = df[df[group_col].isin(GROUP_ORDER)].copy()

    data = []
    for g in GROUP_ORDER:
        vals = _finite(keep.loc[keep[group_col] == g, delta_col].to_numpy())
        data.append(vals)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    bp = ax.boxplot(
        data,
        labels=GROUP_ORDER,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="k", linewidth=2),
    )

    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS.get(g, "#CCCCCC"))
        patch.set_alpha(0.6)
        patch.set_edgecolor("k")

    # Jittered points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if vals.size == 0:
            continue
        xj = i + rng.normal(0, 0.06, size=vals.size)
        ax.scatter(xj, vals, s=14, alpha=0.55, c="k", edgecolors="none")

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("delta_dim = post_dim − pre_dim")
    ax.set_title("Δ intrinsic dimensionality by lesion group")
    ax.tick_params(axis="x", labelrotation=25)

    # --------------------
    # Stats report
    # --------------------
    lines: List[str] = []
    lines.append("Levina–Bickel Δ intrinsic dimensionality stats\n")
    lines.append("================================================\n\n")
    lines.append(f"delta column: {delta_col}\n")
    lines.append(f"group column: {group_col}\n")
    lines.append(f"pairwise test: Mann–Whitney U ({alternative})\n")
    lines.append("multiple comparisons: Holm (2 planned comparisons)\n\n")

    # Cluster-level (rows)
    groups = {g: data[i] for i, g in enumerate(GROUP_ORDER)}
    kw = _kruskal_if_possible(groups)
    if kw is not None:
        H, p = kw
        lines.append(f"[Cluster-level] Kruskal–Wallis across 3 groups: H={H:.4g}, p={p:.6g}\n")
    else:
        lines.append("[Cluster-level] Kruskal–Wallis: (skipped; SciPy missing or insufficient data)\n")

    # Planned comparisons
    sham = groups[HT_SHAM]
    vis = groups[HT_VISIBLE_SINGLE]
    comb = groups[GROUP_COMBINED]

    planned = [
        (HT_VISIBLE_SINGLE, vis, HT_SHAM, sham),
        (GROUP_COMBINED, comb, HT_SHAM, sham),
    ]

    pvals = []
    raw = []
    for name_a, a, name_b, b in planned:
        if _HAVE_SCIPY:
            U, p, n_a, n_b = _mwu(a, b, alternative=alternative)
            raw.append((name_a, name_b, U, p, n_a, n_b))
            pvals.append(p)
        else:
            raw.append((name_a, name_b, float("nan"), float("nan"), int(a.size), int(b.size)))
            pvals.append(float("nan"))

    if _HAVE_SCIPY:
        padj = _holm_adjust(pvals)
        lines.append("\n[Cluster-level] Planned pairwise tests:\n")
        for (name_a, name_b, U, p, n_a, n_b), pa in zip(raw, padj):
            lines.append(f"  - {name_a} vs {name_b}: U={U:.4g}, p={p:.6g}, p_holm={pa:.6g} (n_a={n_a}, n_b={n_b})\n")
    else:
        lines.append("\n[Cluster-level] Pairwise tests skipped (SciPy missing).\n")

    # Bird-level sensitivity (mean delta per animal)
    lines.append("\n\n[Bird-level sensitivity] Mean delta_dim per bird, then same tests\n")
    lines.append("---------------------------------------------------------------\n")
    if "animal_id" not in df.columns:
        lines.append("  (skipped) no animal_id column in CSV.\n")
    else:
        bird = keep.groupby(["animal_id", group_col], as_index=False)[delta_col].mean()
        bird_data = {g: _finite(bird.loc[bird[group_col] == g, delta_col].to_numpy()) for g in GROUP_ORDER}
        kw2 = _kruskal_if_possible(bird_data)
        if kw2 is not None:
            H, p = kw2
            lines.append(f"  Kruskal–Wallis across 3 groups: H={H:.4g}, p={p:.6g}\n")
        else:
            lines.append("  Kruskal–Wallis: (skipped; SciPy missing or insufficient data)\n")

        if _HAVE_SCIPY:
            sham_b = bird_data[HT_SHAM]
            vis_b = bird_data[HT_VISIBLE_SINGLE]
            comb_b = bird_data[GROUP_COMBINED]

            planned_b = [
                (HT_VISIBLE_SINGLE, vis_b, HT_SHAM, sham_b),
                (GROUP_COMBINED, comb_b, HT_SHAM, sham_b),
            ]

            pvals_b = []
            raw_b = []
            for name_a, a, name_b, b in planned_b:
                U, p, n_a, n_b = _mwu(a, b, alternative=alternative)
                raw_b.append((name_a, name_b, U, p, n_a, n_b))
                pvals_b.append(p)

            padj_b = _holm_adjust(pvals_b)
            for (name_a, name_b, U, p, n_a, n_b), pa in zip(raw_b, padj_b):
                lines.append(f"  - {name_a} vs {name_b}: U={U:.4g}, p={p:.6g}, p_holm={pa:.6g} (n_a={n_a}, n_b={n_b})\n")
        else:
            lines.append("  Pairwise tests skipped (SciPy missing).\n")

    stats_path.write_text("".join(lines))

    # --------------------
    # Annotate plot brackets
    # --------------------
    if _HAVE_SCIPY:
        # Use Holm-adjusted p-values for the two planned comparisons
        padj = _holm_adjust(pvals)
        anns = [
            (1, 2, padj[0]),  # sham vs visible
            (1, 3, padj[1]),  # sham vs combined
        ]
        ymax = np.nanmax([np.nanmax(v) if v.size else np.nan for v in data])
        if np.isfinite(ymax):
            y0 = ymax + 0.15 * (abs(ymax) + 1.0)
            step = 0.12 * (abs(ymax) + 1.0)
            for j, (x1, x2, p) in enumerate(anns):
                _add_sig_bracket(ax, x1, x2, y0 + j * step, f"{_p_to_stars(p)} (p={p:.3g})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Graphs + stats for Levina–Bickel pre/post dimensionality outputs.")
    ap.add_argument("--csv", required=True, type=str, help="CSV from Levina_Bickel_pre_vs_post_treatment_v5.py")
    ap.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <csv_dir>/lb_graphs/)")

    ap.add_argument("--pre-col", default="pre_dim", type=str)
    ap.add_argument("--post-col", default="post_dim", type=str)
    ap.add_argument("--delta-col", default="delta_dim", type=str)
    ap.add_argument("--group-col", default="group", type=str)
    ap.add_argument("--hit-type-col", default="hit_type", type=str)

    ap.add_argument("--min-pre", default=200, type=int, help="Filter clusters with n_pre < this (0 disables).")
    ap.add_argument("--min-post", default=200, type=int, help="Filter clusters with n_post < this (0 disables).")
    ap.add_argument("--k", default=None, type=int, help="If set, keep only rows with this k.")

    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"],
                    help="MWU alternative hypothesis (default: two-sided).")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else (csv_path.parent / "lb_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Confirm required columns
    _ensure_cols(df, [
        "animal_id", "hit_type", "group",
        "n_pre", "n_post",
        args.pre_col, args.post_col, args.delta_col
    ])

    # Basic filtering
    df = df.copy()
    if args.k is not None and "k" in df.columns:
        df = df[df["k"] == args.k].copy()

    if args.min_pre > 0:
        df = df[df["n_pre"] >= args.min_pre].copy()
    if args.min_post > 0:
        df = df[df["n_post"] >= args.min_post].copy()

    # Keep only hit-types we know how to color (so the legend is clean)
    df = df[df[args.hit_type_col].isin(list(HIT_TYPE_COLORS.keys()))].copy()

    # Make sure group labels are one of the 3 groups; if not, map from hit_type
    if not set(GROUP_ORDER).issubset(set(df[args.group_col].unique())):
        # Some runs store group == hit_type; fix it by mapping hit_type -> 3-group label
        def _to_group(x: str) -> str:
            x = str(x).strip().lower()
            if "sham" in x and "saline" in x:
                return HT_SHAM
            if "visible" in x and "single" in x:
                return HT_VISIBLE_SINGLE
            if ("medial" in x and "lateral" in x) or ("not visible" in x) or ("large lesion" in x):
                return GROUP_COMBINED
            return str(x)
        df[args.group_col] = df[args.hit_type_col].apply(_to_group)

    # 1) Overlay scatter (linear axes)
    overlay_path = out_dir / "pre_vs_post_dim_overlay.png"
    plot_pre_post_overlay(df, overlay_path, pre_col=args.pre_col, post_col=args.post_col, hit_type_col=args.hit_type_col)

    # 2) Boxplot + stats
    box_path = out_dir / "delta_dim_boxplot_by_group.png"
    stats_path = out_dir / "stats_report.txt"
    plot_delta_boxplot_and_tests(
        df=df,
        out_path=box_path,
        stats_path=stats_path,
        group_col=args.group_col,
        delta_col=args.delta_col,
        alternative=args.alternative,
    )

    print(f"[OK] Saved outputs to: {out_dir}")
    print(f"  - Overlay scatter: {overlay_path}")
    print(f"  - Boxplot: {box_path}")
    print(f"  - Stats report: {stats_path}")


if __name__ == "__main__":
    main()
