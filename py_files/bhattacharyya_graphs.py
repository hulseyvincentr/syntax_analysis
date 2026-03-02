#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bhattacharyya_graphs.py

Graphs + stats for Bhattacharyya pre/post similarity outputs produced by:
  bhattacharyya_pre_vs_post_cluster.py

Input CSV (required) columns (from bhattacharyya_pre_vs_post_cluster.py):
  animal_id, hit_type, group, cluster_label, n_pre, n_post,
  bhattacharyya_coeff, bhattacharyya_dist
(+ optional: cov_type, array_key, etc.)

What this script outputs
-----------------------
Always:
  • bhattacharyya_coeff_boxplot_by_group_cluster.png   (cluster-level points)
  • bhattacharyya_dist_boxplot_by_group_cluster.png
  • bhattacharyya_coeff_boxplot_by_group_bird.png      (bird-level medians)
  • bhattacharyya_dist_boxplot_by_group_bird.png
  • stats_report_cluster.txt
  • stats_report_bird.txt
  • coeff_vs_dist_scatter_by_hit_type.png

Optional (if --phrase-stats-csv provided):
  • highVar_coeff_by_lesion_group.png (+ stats txt)
  • lowVar_coeff_by_lesion_group.png  (+ stats txt)
  • variance_vs_coeff_scatter.png

Notes
-----
• Planned group comparisons match your Levina–Bickel graphs:
    1) Area X visible (single hit) vs sham saline injection
    2) Combined (visible ML + not visible) vs sham saline injection
  using Mann–Whitney U and Holm correction (2 comparisons).

Example
-------
conda activate syntax_analysis
cd ~/Documents/allPythonCode/syntax_analysis/py_files/

python bhattacharyya_graphs.py \
  --csv "/Volumes/my_own_SSD/updated_AreaX_outputs/bhattacharyya_pre_post_predictions/bhattacharyya_pre_post_predictions_diag_20260224_123456.csv" \
  --min-pre 500 --min-post 500 \
  --alternative two-sided

(Optionally filter to a single array_key/cov_type if your directory contains multiple runs)
python bhattacharyya_graphs.py \
  --csv "...csv" \
  --array-key predictions \
  --cov-type diag
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
# Canonical strings + colors (match Levina_Bickel_graphs.py)
# -----------------------------
HT_SHAM = "sham saline injection"
HT_VISIBLE_SINGLE = "Area X visible (single hit)"
HT_VISIBLE_ML = "Area X visible (medial+lateral hit)"
HT_NOT_VISIBLE = "large lesion Area X not visible"
GROUP_COMBINED = "Combined (visible ML + not visible)"

GROUP_ORDER = [HT_SHAM, HT_VISIBLE_SINGLE, GROUP_COMBINED]

# Palette (approx from your screenshot / LB script)
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
def _ensure_cols(df: pd.DataFrame, cols: List[str], name: str = "DataFrame") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n"
            f"  missing: {missing}\n"
            f"  present: {list(df.columns)}"
        )


def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni correction (step-down)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    prev = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, p[idx] * factor)
        val = max(val, prev)  # ensure monotone
        adj[idx] = val
        prev = val
    return adj.tolist()


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "na"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    return arr[np.isfinite(arr)]


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    text: str,
    dh_frac: float = 0.05,
    barh_frac: float = 0.012,
) -> None:
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    h = barh_frac * yr
    d = dh_frac * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.6, c="k")
    ax.text((x1 + x2) / 2, y + h + d, text, ha="center", va="bottom")


def _boxplot(ax, data, labels, **kwargs):
    """Matplotlib 3.9 renamed boxplot(labels=...) -> boxplot(tick_labels=...)."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


# -----------------------------
# Stats helpers
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
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available.")
    a = _finite(a)
    b = _finite(b)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), int(a.size), int(b.size)
    res = _scipy_stats.mannwhitneyu(a, b, alternative=alternative)
    return float(res.statistic), float(res.pvalue), int(a.size), int(b.size)


# -----------------------------
# Group mapping fallbacks
# -----------------------------
def _normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _map_hit_type_to_group(hit_type: str) -> str:
    if hit_type is None:
        return "unknown"
    norm = _normalize(hit_type)
    if "sham" in norm:
        return HT_SHAM
    if "singlehit" in norm:
        return HT_VISIBLE_SINGLE
    if ("medial" in norm and "lateral" in norm) or ("ml" in norm and "visible" in norm):
        return GROUP_COMBINED
    if "notvisible" in norm or ("large" in norm and "notvisible" in norm):
        return GROUP_COMBINED
    return str(hit_type)


# -----------------------------
# Plotting + stats core
# -----------------------------
def _collect_group_arrays(df: pd.DataFrame, group_col: str, metric_col: str) -> Dict[str, np.ndarray]:
    keep = df[df[group_col].isin(GROUP_ORDER)].copy()
    out: Dict[str, np.ndarray] = {}
    for g in GROUP_ORDER:
        out[g] = _finite(pd.to_numeric(keep.loc[keep[group_col] == g, metric_col], errors="coerce").to_numpy())
    return out


def _write_group_stats_report(
    groups: Dict[str, np.ndarray],
    stats_path: Path,
    *,
    metric_col: str,
    level_label: str,
    alternative: str,
) -> List[float]:
    """
    Writes a stats report for:
      - Kruskal–Wallis across 3 groups (if SciPy)
      - Planned pairwise MWU tests vs sham + Holm correction
    Returns list of Holm-adjusted p-values in planned order [single vs sham, combined vs sham]
    """
    lines: List[str] = []
    lines.append(f"Bhattacharyya {metric_col} stats ({level_label})\n")
    lines.append("================================================\n\n")
    lines.append(f"metric column: {metric_col}\n")
    lines.append(f"pairwise test: Mann–Whitney U ({alternative})\n")
    lines.append("planned comparisons: (visible single vs sham), (combined vs sham)\n")
    lines.append("multiple comparisons: Holm (2 planned comparisons)\n\n")

    kw = _kruskal_if_possible(groups)
    if kw is not None:
        H, p = kw
        lines.append(f"Kruskal–Wallis across 3 groups: H={H:.4g}, p={p:.6g}\n")
    else:
        lines.append("Kruskal–Wallis: (skipped; SciPy missing or insufficient data)\n")

    sham = groups.get(HT_SHAM, np.array([]))
    vis = groups.get(HT_VISIBLE_SINGLE, np.array([]))
    comb = groups.get(GROUP_COMBINED, np.array([]))

    planned = [
        (HT_VISIBLE_SINGLE, vis, HT_SHAM, sham),
        (GROUP_COMBINED, comb, HT_SHAM, sham),
    ]

    pvals: List[float] = []
    raw: List[Tuple[str, str, float, float, int, int]] = []
    for name_a, a, name_b, b in planned:
        if _HAVE_SCIPY:
            U, p, n_a, n_b = _mwu(a, b, alternative=alternative)
            raw.append((name_a, name_b, U, p, n_a, n_b))
            pvals.append(p)
        else:
            raw.append((name_a, name_b, float("nan"), float("nan"), int(a.size), int(b.size)))
            pvals.append(float("nan"))

    padj = _holm_adjust(pvals) if _HAVE_SCIPY else [float("nan"), float("nan")]

    lines.append("\nPlanned pairwise tests:\n")
    for (name_a, name_b, U, p, n_a, n_b), pa in zip(raw, padj):
        lines.append(
            f"  - {name_a} vs {name_b}: U={U:.4g}, p={p:.6g}, p_holm={pa:.6g} "
            f"(n_a={n_a}, n_b={n_b})\n"
        )

    stats_path.write_text("".join(lines))
    return padj


def _boxplot_by_group(
    df: pd.DataFrame,
    out_path: Path,
    stats_path: Path,
    *,
    group_col: str,
    metric_col: str,
    ylabel: str,
    title: str,
    alternative: str,
    level_label: str,
) -> None:
    groups = _collect_group_arrays(df, group_col=group_col, metric_col=metric_col)

    data = [groups[g] for g in GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bp = _boxplot(
        ax,
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

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if vals.size == 0:
            continue
        xj = i + rng.normal(0, 0.06, size=vals.size)
        ax.scatter(xj, vals, s=14, alpha=0.55, c="k", edgecolors="none")

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.tick_params(axis="x", labelrotation=25)
    _despine(ax)

    padj = _write_group_stats_report(
        groups, stats_path,
        metric_col=metric_col,
        level_label=level_label,
        alternative=alternative,
    )

    # annotate brackets
    if _HAVE_SCIPY:
        ymax = np.nanmax([np.nanmax(v) if v.size else np.nan for v in data])
        if np.isfinite(ymax):
            base = abs(float(ymax)) + 1.0
            y0 = ymax + 0.20 * base
            step = 0.14 * base
            extra_top = y0 + (len(padj)) * step + 0.25 * base
            ylo, yhi = ax.get_ylim()
            if extra_top > yhi:
                ax.set_ylim(ylo, extra_top)

            anns = [(1, 2, padj[0]), (1, 3, padj[1])]
            for j, (x1, x2, p) in enumerate(anns):
                _add_sig_bracket(ax, x1, x2, y0 + j * step, f"{_p_to_stars(p)} (p={p:.3g})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_coeff_vs_dist_scatter(df: pd.DataFrame, out_path: Path, hit_type_col: str) -> None:
    """Sanity check: coefficient vs distance, colored by hit type."""
    sub = df.copy()
    sub["bhattacharyya_coeff"] = pd.to_numeric(sub["bhattacharyya_coeff"], errors="coerce")
    sub["bhattacharyya_dist"] = pd.to_numeric(sub["bhattacharyya_dist"], errors="coerce")
    sub = sub[np.isfinite(sub["bhattacharyya_coeff"].to_numpy()) & np.isfinite(sub["bhattacharyya_dist"].to_numpy())].copy()

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for ht, ss in sub.groupby(hit_type_col):
        c = HIT_TYPE_COLORS.get(ht, "#999999")
        ax.scatter(ss["bhattacharyya_dist"], ss["bhattacharyya_coeff"], s=18, alpha=0.6, c=c, edgecolors="none", label=ht)

    ax.set_xlabel("Bhattacharyya distance (DB)")
    ax.set_ylabel("Bhattacharyya coefficient (BC = exp(-DB))")
    ax.set_title("BC vs DB (colored by lesion hit type)", pad=14)
    _despine(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Hit type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Phrase variance optional tiering (copied in spirit from LB graphs)
# -----------------------------
def load_post_variance_table(
    phrase_csv: Path,
    phrase_group_col: str = "Group",
    post_group_value: str = "Post",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    variance_col: str = "Variance_ms2",
    n_phrases_col: str = "N_phrases",
) -> pd.DataFrame:
    ph = pd.read_csv(phrase_csv)
    _ensure_cols(ph, [phrase_group_col, animal_col, syllable_col, variance_col], name="Phrase stats CSV")

    ph = ph[ph[phrase_group_col].astype(str) == str(post_group_value)].copy()
    ph = ph.rename(columns={animal_col: "animal_id", syllable_col: "cluster_label"})
    ph["animal_id"] = ph["animal_id"].astype(str).str.strip()
    ph["cluster_label"] = pd.to_numeric(ph["cluster_label"], errors="coerce").astype("Int64")
    ph["post_variance_ms2"] = pd.to_numeric(ph[variance_col], errors="coerce")

    if n_phrases_col in ph.columns:
        ph["post_n_phrases"] = pd.to_numeric(ph[n_phrases_col], errors="coerce")
    else:
        ph["post_n_phrases"] = np.nan

    ph = ph[["animal_id", "cluster_label", "post_variance_ms2", "post_n_phrases"]].copy()
    ph = ph.dropna(subset=["cluster_label"]).copy()
    ph["cluster_label"] = ph["cluster_label"].astype(int)

    ph_agg = ph.groupby(["animal_id", "cluster_label"], as_index=False).agg(
        post_variance_ms2=("post_variance_ms2", "median"),
        post_n_phrases=("post_n_phrases", "max"),
    )
    return ph_agg


def add_high_variance_flag(post_var: pd.DataFrame, top_pct: float = 70.0, min_phrases: int = 20) -> pd.DataFrame:
    """Within each bird, mark syllables with POST variance >= percentile(top_pct) as high variance."""
    df = post_var.copy()
    df["_use"] = True
    if min_phrases > 0:
        df["_use"] = df["post_n_phrases"].fillna(0) >= min_phrases

    df["is_high_variance"] = False
    for aid, sub in df.groupby("animal_id"):
        sub_use = sub[sub["_use"] & np.isfinite(sub["post_variance_ms2"].to_numpy())]
        if sub_use.shape[0] < 3:
            continue
        thr = np.nanpercentile(sub_use["post_variance_ms2"].to_numpy(), top_pct)
        idx = sub.index[(sub["post_variance_ms2"] >= thr) & sub["_use"]]
        df.loc[idx, "is_high_variance"] = True

    return df.drop(columns=["_use"])


def plot_variance_vs_metric_scatter(
    merged: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    hit_type_col: str,
) -> None:
    df = merged.copy()
    df["post_variance_ms2"] = pd.to_numeric(df["post_variance_ms2"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[np.isfinite(df["post_variance_ms2"].to_numpy()) & np.isfinite(df[metric_col].to_numpy())].copy()

    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    low = df[~df["is_high_variance"].fillna(False).astype(bool)]
    high = df[df["is_high_variance"].fillna(False).astype(bool)]

    for ht, sub in low.groupby(hit_type_col):
        c = HIT_TYPE_COLORS.get(ht, "#999999")
        ax.scatter(sub["post_variance_ms2"], sub[metric_col], s=18, alpha=0.45, c=c, edgecolors="none")

    for ht, sub in high.groupby(hit_type_col):
        c = HIT_TYPE_COLORS.get(ht, "#999999")
        ax.scatter(sub["post_variance_ms2"], sub[metric_col], s=26, alpha=0.90, c=c,
                   edgecolors="k", linewidths=0.7, label=f"{ht} (high var)")

    ax.set_xlabel("Post phrase duration variance (ms$^2$)")
    ax.set_ylabel(metric_col)
    ax.set_title(f"Post variance vs {metric_col}\n(high-variance syllables highlighted)", pad=14)
    _despine(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Hit type")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Graphs + stats for Bhattacharyya pre/post similarity outputs.")
    ap.add_argument("--csv", required=True, type=str, help="Bhattacharyya CSV from bhattacharyya_pre_vs_post_cluster.py")
    ap.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <csv_dir>/bhattacharyya_graphs/)")

    ap.add_argument("--min-pre", default=0, type=int, help="Filter clusters with n_pre < this (0 disables).")
    ap.add_argument("--min-post", default=0, type=int, help="Filter clusters with n_post < this (0 disables).")

    ap.add_argument("--array-key", default=None, type=str, help="If provided and CSV has array_key, filter to this.")
    ap.add_argument("--cov-type", default=None, type=str, help="If provided and CSV has cov_type, filter to this.")

    ap.add_argument("--group-col", default="group", type=str)
    ap.add_argument("--hit-type-col", default="hit_type", type=str)
    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"],
                    help="Alternative hypothesis for MWU tests (default: two-sided).")

    # Optional phrase stats + tiering
    ap.add_argument("--phrase-stats-csv", default=None, type=str,
                    help="Optional phrase duration stats CSV to do high/low-variance tier comparisons.")
    ap.add_argument("--top-pct", default=70.0, type=float,
                    help="Percentile threshold WITHIN each bird for High variance (default 70 => top 30%).")
    ap.add_argument("--min-phrases", default=20, type=int,
                    help="Min POST N_phrases to include syllable when computing per-bird percentile threshold.")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else (csv_path.parent / "bhattacharyya_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    _ensure_cols(df, [
        "animal_id", args.hit_type_col, args.group_col, "cluster_label",
        "n_pre", "n_post", "bhattacharyya_coeff", "bhattacharyya_dist"
    ], name="Bhattacharyya CSV")

    # numeric coercion
    df = df.copy()
    df["n_pre"] = pd.to_numeric(df["n_pre"], errors="coerce")
    df["n_post"] = pd.to_numeric(df["n_post"], errors="coerce")
    df["bhattacharyya_coeff"] = pd.to_numeric(df["bhattacharyya_coeff"], errors="coerce")
    df["bhattacharyya_dist"] = pd.to_numeric(df["bhattacharyya_dist"], errors="coerce")

    # filters
    if args.min_pre > 0:
        df = df[df["n_pre"] >= args.min_pre].copy()
    if args.min_post > 0:
        df = df[df["n_post"] >= args.min_post].copy()

    if args.array_key and ("array_key" in df.columns):
        df = df[df["array_key"].astype(str) == str(args.array_key)].copy()
    if args.cov_type and ("cov_type" in df.columns):
        df = df[df["cov_type"].astype(str) == str(args.cov_type)].copy()

    # Ensure group labels exist; else map from hit_type
    if not set(GROUP_ORDER).issubset(set(df[args.group_col].astype(str).unique())):
        df[args.group_col] = df[args.hit_type_col].astype(str).apply(_map_hit_type_to_group)

    # Keep only hit-types we know how to color (for scatter / tier plots)
    df = df[df[args.hit_type_col].isin(list(HIT_TYPE_COLORS.keys()))].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check --min-pre/--min-post or your CSV content.")

    # --- Cluster-level plots ---
    _boxplot_by_group(
        df=df,
        out_path=out_dir / "bhattacharyya_coeff_boxplot_by_group_cluster.png",
        stats_path=out_dir / "stats_report_cluster_coeff.txt",
        group_col=args.group_col,
        metric_col="bhattacharyya_coeff",
        ylabel="Bhattacharyya coefficient (higher = more similar)",
        title="Pre vs Post similarity (Bhattacharyya coefficient)\nCluster-level values by lesion group",
        alternative=args.alternative,
        level_label="cluster-level",
    )

    _boxplot_by_group(
        df=df,
        out_path=out_dir / "bhattacharyya_dist_boxplot_by_group_cluster.png",
        stats_path=out_dir / "stats_report_cluster_dist.txt",
        group_col=args.group_col,
        metric_col="bhattacharyya_dist",
        ylabel="Bhattacharyya distance (lower = more similar)",
        title="Pre vs Post similarity (Bhattacharyya distance)\nCluster-level values by lesion group",
        alternative=args.alternative,
        level_label="cluster-level",
    )

    # --- Bird-level plots (median across clusters per bird) ---
    bird = (
        df.groupby(["animal_id", args.group_col, args.hit_type_col], as_index=False)
          .agg(
              bhattacharyya_coeff=("bhattacharyya_coeff", "median"),
              bhattacharyya_dist=("bhattacharyya_dist", "median"),
              n_clusters=("cluster_label", "nunique"),
          )
    )

    _boxplot_by_group(
        df=bird.rename(columns={args.group_col: "group"}),
        out_path=out_dir / "bhattacharyya_coeff_boxplot_by_group_bird.png",
        stats_path=out_dir / "stats_report_bird_coeff.txt",
        group_col="group",
        metric_col="bhattacharyya_coeff",
        ylabel="Median Bhattacharyya coefficient per bird",
        title="Pre vs Post similarity (Bhattacharyya coefficient)\nBird-level medians by lesion group",
        alternative=args.alternative,
        level_label="bird-level",
    )

    _boxplot_by_group(
        df=bird.rename(columns={args.group_col: "group"}),
        out_path=out_dir / "bhattacharyya_dist_boxplot_by_group_bird.png",
        stats_path=out_dir / "stats_report_bird_dist.txt",
        group_col="group",
        metric_col="bhattacharyya_dist",
        ylabel="Median Bhattacharyya distance per bird",
        title="Pre vs Post similarity (Bhattacharyya distance)\nBird-level medians by lesion group",
        alternative=args.alternative,
        level_label="bird-level",
    )

    # sanity scatter
    plot_coeff_vs_dist_scatter(df, out_dir / "coeff_vs_dist_scatter_by_hit_type.png", hit_type_col=args.hit_type_col)

    # --- Optional tiered variance analysis ---
    if args.phrase_stats_csv:
        phrase_csv = Path(args.phrase_stats_csv)
        if not phrase_csv.exists():
            raise FileNotFoundError(phrase_csv)

        post_var = load_post_variance_table(phrase_csv)
        post_var = add_high_variance_flag(post_var, top_pct=args.top_pct, min_phrases=args.min_phrases)

        merged = df.merge(post_var, how="inner", on=["animal_id", "cluster_label"], validate="many_to_one")
        merged["is_high_variance"] = merged["is_high_variance"].fillna(False).astype(bool)

        if merged.empty:
            raise ValueError(
                "After merging phrase stats with Bhattacharyya CSV, no rows remained. "
                "Check that phrase 'Syllable' IDs match 'cluster_label' and animal IDs match."
            )

        # High tier
        hi = merged[merged["is_high_variance"]].copy()
        _boxplot_by_group(
            df=hi,
            out_path=out_dir / "highVar_coeff_by_lesion_group.png",
            stats_path=out_dir / "highVar_coeff_by_lesion_group_stats.txt",
            group_col=args.group_col,
            metric_col="bhattacharyya_coeff",
            ylabel="Bhattacharyya coefficient (higher = more similar)",
            title="High-variance syllables: BC by lesion group",
            alternative=args.alternative,
            level_label="cluster-level (high variance)",
        )

        # Low tier
        lo = merged[~merged["is_high_variance"]].copy()
        _boxplot_by_group(
            df=lo,
            out_path=out_dir / "lowVar_coeff_by_lesion_group.png",
            stats_path=out_dir / "lowVar_coeff_by_lesion_group_stats.txt",
            group_col=args.group_col,
            metric_col="bhattacharyya_coeff",
            ylabel="Bhattacharyya coefficient (higher = more similar)",
            title="Low-variance syllables: BC by lesion group",
            alternative=args.alternative,
            level_label="cluster-level (low variance)",
        )

        plot_variance_vs_metric_scatter(
            merged=merged,
            out_path=out_dir / "variance_vs_coeff_scatter.png",
            metric_col="bhattacharyya_coeff",
            hit_type_col=args.hit_type_col,
        )

    print(f"[OK] Saved outputs to: {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print("  -", p.name)
    for p in sorted(out_dir.glob("*.txt")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
