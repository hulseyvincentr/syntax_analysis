#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levina_Bickel_graphs.py

Graphs + stats for Levina–Bickel intrinsic dimensionality outputs produced by:
  Levina_Bickel_pre_vs_post_treatment_v5.py

Core input CSV (required) columns:
    animal_id, hit_type, group, cluster_label, k, n_pre, n_post,
    pre_dim, post_dim, delta_dim

Optional phrase-duration stats CSV (for "stutter" / high-variance syllables):
    usage_balanced_phrase_duration_stats.csv

Expected columns in phrase CSV:
    Group (Pre/Post), Animal ID, Syllable, Variance_ms2, N_phrases
(Extra columns are fine.)

What this script outputs
-----------------------
Always (LB only):
  • pre_vs_post_dim_overlay.png
  • delta_dim_boxplot_by_group.png
  • stats_report.txt

If --phrase-stats-csv provided:
  • highVar_delta_dim_by_lesion_group.png (+ stats txt)
  • lowVar_delta_dim_by_lesion_group.png  (+ stats txt)
  • variance_vs_delta_dim_scatter.png

Also if --phrase-stats-csv provided:
  • dim_vs_logvar_scatter_panels.png
  • dim_variance_relationship_report.txt

NEW (requested): the dim_vs_logvar_scatter_panels.png now ANNOTATES each panel with:
  - Spearman rho, p
  - within-bird median rho, Wilcoxon p
  - MixedLM beta, p  (if fit succeeds)

Example
-------
conda activate syntax_analysis
cd ~/Documents/allPythonCode/syntax_analysis/py_files/

python Levina_Bickel_graphs.py \
  --csv "/Volumes/my_own_SSD/updated_AreaX_outputs/lb_pre_post_dimensionality_k15/lb_pre_post_cluster_dimensionality_k15_20260224_115517.csv" \
  --phrase-stats-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/lb_pre_post_dimensionality_k15/usage_balanced_phrase_duration_stats.csv" \
  --min-pre 200 --min-post 200 \
  --min-phrases 20 \
  --dim-predictor post_dim \
  --alternative two-sided
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

# Optional statsmodels for mixed-effects regression
try:
    import statsmodels.formula.api as smf  # type: ignore
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False


# -----------------------------
# Canonical strings + colors
# -----------------------------
HT_SHAM = "sham saline injection"
HT_VISIBLE_SINGLE = "Area X visible (single hit)"
HT_VISIBLE_ML = "Area X visible (medial+lateral hit)"
HT_NOT_VISIBLE = "large lesion Area X not visible"
GROUP_COMBINED = "Combined (visible ML + not visible)"

GROUP_ORDER = [HT_SHAM, HT_VISIBLE_SINGLE, GROUP_COMBINED]

# Palette (approx from your screenshot)
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


def _wilcoxon_1samp(x: np.ndarray, alternative: str = "two-sided") -> Tuple[float, float]:
    if not _HAVE_SCIPY:
        return float("nan"), float("nan")
    x = _finite(x)
    if x.size == 0:
        return float("nan"), float("nan")
    try:
        res = _scipy_stats.wilcoxon(x, zero_method="wilcox", alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        res = _scipy_stats.wilcoxon(x, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)


# -----------------------------
# Standard LB plots
# -----------------------------
def plot_pre_post_overlay(df: pd.DataFrame, out_path: Path, pre_col: str, post_col: str, hit_type_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

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

    x = df[pre_col].to_numpy()
    y = df[post_col].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.any():
        mn = float(min(x[m].min(), y[m].min()))
        mx = float(max(x[m].max(), y[m].max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, color="red")

    ax.set_xlabel("Pre intrinsic dimensionality (Levina–Bickel)")
    ax.set_ylabel("Post intrinsic dimensionality (Levina–Bickel)")
    ax.set_title("Pre vs Post intrinsic dimensionality (colored by lesion hit type)", pad=14)

    _despine(ax)
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
    keep = df[df[group_col].isin(GROUP_ORDER)].copy()

    data = []
    for g in GROUP_ORDER:
        vals = _finite(keep.loc[keep[group_col] == g, delta_col].to_numpy())
        data.append(vals)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
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

    ax.axhline(0, linestyle="--", linewidth=1, color="#1f77b4")
    ax.set_ylabel("delta_dim = post_dim − pre_dim")
    ax.set_title("Δ intrinsic dimensionality by lesion group", pad=18)
    ax.tick_params(axis="x", labelrotation=25)
    _despine(ax)

    # Stats report
    lines: List[str] = []
    lines.append("Levina–Bickel Δ intrinsic dimensionality stats\n")
    lines.append("================================================\n\n")
    lines.append(f"delta column: {delta_col}\n")
    lines.append(f"group column: {group_col}\n")
    lines.append(f"pairwise test: Mann–Whitney U ({alternative})\n")
    lines.append("multiple comparisons: Holm (2 planned comparisons)\n\n")

    groups = {g: data[i] for i, g in enumerate(GROUP_ORDER)}
    kw = _kruskal_if_possible(groups)
    if kw is not None:
        H, p = kw
        lines.append(f"[Cluster-level] Kruskal–Wallis across 3 groups: H={H:.4g}, p={p:.6g}\n")
    else:
        lines.append("[Cluster-level] Kruskal–Wallis: (skipped; SciPy missing or insufficient data)\n")

    sham = groups[HT_SHAM]
    vis = groups[HT_VISIBLE_SINGLE]
    comb = groups[GROUP_COMBINED]

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

    if _HAVE_SCIPY:
        padj = _holm_adjust(pvals)
        lines.append("\n[Cluster-level] Planned pairwise tests:\n")
        for (name_a, name_b, U, p, n_a, n_b), pa in zip(raw, padj):
            lines.append(f"  - {name_a} vs {name_b}: U={U:.4g}, p={p:.6g}, p_holm={pa:.6g} (n_a={n_a}, n_b={n_b})\n")
    else:
        lines.append("\n[Cluster-level] Pairwise tests skipped (SciPy missing).\n")

    stats_path.write_text("".join(lines))

    # annotate brackets
    if _HAVE_SCIPY:
        padj = _holm_adjust(pvals)
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


# -----------------------------
# Phrase variance loading + tiering
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
    """Return ONE row per (animal_id, cluster_label) for POST variance; collapse duplicates safely."""
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

    dup_mask = ph.duplicated(subset=["animal_id", "cluster_label"], keep=False)
    if dup_mask.any():
        print(f"[INFO] Phrase stats contains duplicated keys after filtering to Post ({dup_mask.sum()} rows duplicated). Collapsing duplicates.")

    ph_agg = ph.groupby(["animal_id", "cluster_label"], as_index=False).agg(
        post_variance_ms2=("post_variance_ms2", "median"),
        post_n_phrases=("post_n_phrases", "max"),
    )
    return ph_agg


def add_high_variance_flag(
    post_var: pd.DataFrame,
    top_pct: float = 70.0,
    min_phrases: int = 20,
) -> pd.DataFrame:
    """Within each animal_id, mark syllables with post_variance >= percentile(top_pct) as high variance."""
    df = post_var.copy()
    df["_use_for_threshold"] = True
    if min_phrases > 0:
        df["_use_for_threshold"] = df["post_n_phrases"].fillna(0) >= min_phrases

    df["is_high_variance"] = False
    for aid, sub in df.groupby("animal_id"):
        sub_use = sub[sub["_use_for_threshold"] & np.isfinite(sub["post_variance_ms2"].to_numpy())]
        if sub_use.shape[0] < 3:
            continue
        thr = np.nanpercentile(sub_use["post_variance_ms2"].to_numpy(), top_pct)
        idx = sub.index[(sub["post_variance_ms2"] >= thr) & sub["_use_for_threshold"]]
        df.loc[idx, "is_high_variance"] = True

    return df.drop(columns=["_use_for_threshold"])


# -----------------------------
# Tiered lesion-group comparison (high tier and low tier)
# -----------------------------
def plot_tier_group_boxplot_and_tests(
    merged: pd.DataFrame,
    out_path: Path,
    stats_path: Path,
    tier_mask: np.ndarray,
    tier_label: str,
    group_col: str = "group",
    delta_col: str = "delta_dim",
    alternative: str = "two-sided",
) -> None:
    df = merged.copy()
    df = df[df[group_col].isin(GROUP_ORDER)].copy()
    df = df[np.asarray(tier_mask, dtype=bool)].copy()
    df = df[np.isfinite(df[delta_col].to_numpy())].copy()

    data = []
    for g in GROUP_ORDER:
        data.append(_finite(df.loc[df[group_col] == g, delta_col].to_numpy()))

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

    ax.axhline(0, linestyle="--", linewidth=1, color="#1f77b4")
    ax.set_ylabel("delta_dim = post_dim − pre_dim")
    ax.set_title(f"Δ intrinsic dimensionality by lesion group\n{tier_label}", pad=18)
    ax.tick_params(axis="x", labelrotation=25)
    _despine(ax)

    # Stats report
    lines: List[str] = []
    lines.append(f"Tiered lesion-group comparison: {tier_label}\n")
    lines.append("================================================\n\n")
    lines.append(f"delta column: {delta_col}\n")
    lines.append(f"group column: {group_col}\n")
    lines.append(f"MWU alternative: {alternative}\n")
    lines.append("planned comparisons: (visible single vs sham), (combined vs sham)\n")
    lines.append("multiple comparisons: Holm (2 planned comparisons)\n\n")

    groups = {g: data[i] for i, g in enumerate(GROUP_ORDER)}
    kw = _kruskal_if_possible(groups)
    if kw is not None:
        H, p = kw
        lines.append(f"[Cluster-level] Kruskal–Wallis across 3 groups: H={H:.4g}, p={p:.6g}\n")
    else:
        lines.append("[Cluster-level] Kruskal–Wallis: (skipped; SciPy missing or insufficient data)\n")

    sham = groups[HT_SHAM]
    vis = groups[HT_VISIBLE_SINGLE]
    comb = groups[GROUP_COMBINED]

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

    if _HAVE_SCIPY:
        padj = _holm_adjust(pvals)
        lines.append("\n[Cluster-level] Planned pairwise tests:\n")
        for (name_a, name_b, U, p, n_a, n_b), pa in zip(raw, padj):
            lines.append(f"  - {name_a} vs {name_b}: U={U:.4g}, p={p:.6g}, p_holm={pa:.6g} (n_a={n_a}, n_b={n_b})\n")
    else:
        lines.append("\n[Cluster-level] Pairwise tests skipped (SciPy missing).\n")

    stats_path.write_text("".join(lines))

    # annotate plot brackets
    if _HAVE_SCIPY:
        padj = _holm_adjust(pvals)
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


def plot_variance_vs_delta_dim_scatter(
    merged: pd.DataFrame,
    out_path: Path,
    x_col: str = "post_variance_ms2",
    y_col: str = "delta_dim",
    hit_type_col: str = "hit_type",
) -> None:
    """Scatter of post variance vs delta_dim (sanity check)."""
    df = merged.copy()
    df = df[np.isfinite(df[x_col].to_numpy()) & np.isfinite(df[y_col].to_numpy())].copy()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    low = df[~df["is_high_variance"].fillna(False).astype(bool)]
    high = df[df["is_high_variance"].fillna(False).astype(bool)]

    for ht, sub in low.groupby(hit_type_col):
        c = HIT_TYPE_COLORS.get(ht, "#999999")
        ax.scatter(sub[x_col].to_numpy(), sub[y_col].to_numpy(), s=18, alpha=0.45, c=c, edgecolors="none")

    for ht, sub in high.groupby(hit_type_col):
        c = HIT_TYPE_COLORS.get(ht, "#999999")
        ax.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            s=26,
            alpha=0.90,
            c=c,
            edgecolors="k",
            linewidths=0.7,
            label=f"{ht} (high var)",
        )

    ax.axhline(0, linestyle="--", linewidth=1, color="#1f77b4")
    ax.set_xlabel("Post phrase duration variance (ms$^2$)")
    ax.set_ylabel("delta_dim = post_dim − pre_dim")
    ax.set_title("Post variance vs Δ intrinsic dimensionality\n(high-variance syllables highlighted)", pad=14)
    _despine(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Hit type")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Dimensionality–variance relationship tests
# -----------------------------
def _subset_masks_by_hit_type(df: pd.DataFrame, hit_type_col: str) -> Dict[str, np.ndarray]:
    """Masks used for the dim–variance relationship tests.

    IMPORTANT (per your interpretation):
      • The 'M+L hit' bucket includes BOTH:
          - Area X visible (medial+lateral hit)
          - large lesion Area X not visible
      • The pooled lesion bucket includes:
          - single hit + (M+L + not visible)
    """
    ht = df[hit_type_col].astype(str)
    ml_or_not_visible = ht.isin([HT_VISIBLE_ML, HT_NOT_VISIBLE]).to_numpy()
    lesion_pooled = ht.isin([HT_VISIBLE_SINGLE, HT_VISIBLE_ML, HT_NOT_VISIBLE]).to_numpy()

    return {
        HT_SHAM: (ht == HT_SHAM).to_numpy(),
        HT_VISIBLE_SINGLE: (ht == HT_VISIBLE_SINGLE).to_numpy(),
        f"{HT_VISIBLE_ML} (+ not visible)": ml_or_not_visible,
        "Lesion pooled (single + M+L + not visible)": lesion_pooled,
    }


def run_dim_variance_tests(
    merged: pd.DataFrame,
    out_report: Path,
    out_fig: Path,
    predictor_col: str,
    hit_type_col: str = "hit_type",
    animal_col: str = "animal_id",
    var_col: str = "post_variance_ms2",
    nphr_col: str = "post_n_phrases",
    npost_col: str = "n_post",
    alternative: str = "two-sided",
) -> None:
    """
    Tests whether larger intrinsic dimension (predictor_col) is associated with higher POST variance.
    Runs separately for sham, single hit, M+L(+not visible), pooled lesions.
    """
    df = merged.copy()

    _ensure_cols(df, [predictor_col, var_col, animal_col, hit_type_col], name="Merged table")
    df = df[np.isfinite(df[predictor_col].to_numpy()) & np.isfinite(df[var_col].to_numpy())].copy()

    df["log_var"] = np.log10(df[var_col].to_numpy() + 1e-12)
    if nphr_col in df.columns:
        df["log_nphr"] = np.log10(df[nphr_col].fillna(0).to_numpy() + 1.0)
    else:
        df["log_nphr"] = 0.0

    if npost_col in df.columns:
        df["log_npost"] = np.log10(pd.to_numeric(df[npost_col], errors="coerce").fillna(0).to_numpy() + 1.0)
    else:
        df["log_npost"] = 0.0

    masks = _subset_masks_by_hit_type(df, hit_type_col)

    # Report text
    lines: List[str] = []
    lines.append("Dimensionality–variance relationship tests\n")
    lines.append("=========================================\n\n")
    lines.append(f"Predictor column: {predictor_col}\n")
    lines.append("Outcome: log10(post_variance_ms2)\n")
    lines.append(f"Alternative for Wilcoxon on within-bird rho: {alternative}\n\n")
    lines.append("Methods:\n")
    lines.append("  A) Cluster-level Spearman rho (predictor vs log_var)\n")
    lines.append("  B) Within-bird Spearman rho per bird; Wilcoxon test on rho's vs 0\n")
    lines.append("  C) Mixed-effects regression (if statsmodels): log_var ~ predictor + log_nphr + log_npost + (1|bird)\n\n")

    # Figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8), sharey=True)
    axes = axes.ravel()
    fig.suptitle(f"{predictor_col} vs log10(post variance)", y=0.985)

    for ax, (name, mask) in zip(axes, masks.items()):
        sub = df[mask].copy()
        n_clusters = sub.shape[0]
        n_birds = sub[animal_col].nunique()

        lines.append(f"\n--- {name} ---\n")
        lines.append(f"n_clusters={n_clusters}, n_birds={n_birds}\n")

        ax.set_title(name)
        ax.set_xlabel(predictor_col)
        _despine(ax)

        if n_clusters < 5 or n_birds < 1:
            lines.append("Too little data; skipping.\n")
            ax.text(0.02, 0.98, "insufficient data", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            continue

        ax.scatter(sub[predictor_col].to_numpy(), sub["log_var"].to_numpy(), s=18, alpha=0.5, edgecolors="none")

        # Compute stats for both report and on-plot annotation
        rho_all = float("nan")
        p_all = float("nan")
        bird_rhos: List[float] = []
        med = float("nan")
        W = float("nan")
        pW = float("nan")
        beta = float("nan")
        se = float("nan")
        pm = float("nan")

        # Cluster-level Spearman
        if _HAVE_SCIPY:
            rho_all, p_all = _scipy_stats.spearmanr(
                sub[predictor_col].to_numpy(),
                sub["log_var"].to_numpy(),
                nan_policy="omit",
            )
            lines.append(f"[Cluster-level] Spearman rho={rho_all:.4g}, p={p_all:.6g}\n")
        else:
            lines.append("[Cluster-level] Spearman skipped (SciPy missing).\n")

        # Within-bird rhos + Wilcoxon
        if _HAVE_SCIPY:
            for bid, ss in sub.groupby(animal_col):
                x = ss[predictor_col].to_numpy()
                y = ss["log_var"].to_numpy()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 3:
                    continue
                r, _ = _scipy_stats.spearmanr(x[m], y[m])
                if np.isfinite(r):
                    bird_rhos.append(float(r))

            lines.append(f"[Within-bird] n_birds_with_rho={len(bird_rhos)}\n")
            if len(bird_rhos) >= 3:
                arr = np.asarray(bird_rhos, float)
                med = float(np.nanmedian(arr))
                W, pW = _wilcoxon_1samp(arr, alternative=alternative)
                lines.append(f"  median_rho={med:.4g}; Wilcoxon W={W:.4g}, p={pW:.6g}\n")
            else:
                lines.append("  (skipped) <3 birds with >=3 syllables.\n")
        else:
            lines.append("[Within-bird] skipped (SciPy missing).\n")

        # Mixed effects
        if _HAVE_SM and (n_birds >= 2):
            model_df = sub[[animal_col, "log_var", predictor_col, "log_nphr", "log_npost"]].copy()
            model_df = model_df.rename(columns={predictor_col: "pred"})
            model_df = model_df[np.isfinite(model_df["pred"].to_numpy()) & np.isfinite(model_df["log_var"].to_numpy())].copy()
            try:
                md = smf.mixedlm("log_var ~ pred + log_nphr + log_npost", model_df, groups=model_df[animal_col])
                mdf = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
                beta = float(mdf.params.get("pred", np.nan))
                se = float(mdf.bse.get("pred", np.nan))
                pm = float(mdf.pvalues.get("pred", np.nan))
                lines.append(f"[MixedLM] beta_pred={beta:.4g}, SE={se:.4g}, p={pm:.6g}\n")
            except Exception as e:
                lines.append(f"[MixedLM] failed: {e}\n")
        else:
            if not _HAVE_SM:
                lines.append("[MixedLM] skipped (statsmodels missing).\n")
            else:
                lines.append("[MixedLM] skipped (need >=2 birds).\n")

        # --- annotate panel with key stats ---
        annot = [f"n={n_clusters} (clusters), {n_birds} birds"]
        if _HAVE_SCIPY and np.isfinite(rho_all):
            annot.append(f"Spearman ρ={rho_all:.3f}, p={_fmt_p(p_all)}")
        if _HAVE_SCIPY and len(bird_rhos) >= 3 and np.isfinite(med):
            annot.append(f"Within-bird med ρ={med:.3f}, p={_fmt_p(pW)}")
        elif _HAVE_SCIPY:
            annot.append(f"Within-bird: n={len(bird_rhos)}")
        if _HAVE_SM and np.isfinite(beta):
            annot.append(f"MixedLM β={beta:.3f}, p={_fmt_p(pm)}")

        ax.text(
            0.02,
            0.98,
            "\n".join(annot),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
        )

    axes[0].set_ylabel("log10(post variance)")
    axes[2].set_ylabel("log10(post variance)")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    out_report.write_text("".join(lines))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Graphs + stats for Levina–Bickel pre/post dimensionality outputs.")
    ap.add_argument("--csv", required=True, type=str, help="LB CSV from Levina_Bickel_pre_vs_post_treatment_v5.py")
    ap.add_argument("--out-dir", default=None, type=str, help="Output directory (default: <csv_dir>/lb_graphs/)")

    ap.add_argument("--pre-col", default="pre_dim", type=str)
    ap.add_argument("--post-col", default="post_dim", type=str)
    ap.add_argument("--delta-col", default="delta_dim", type=str)
    ap.add_argument("--group-col", default="group", type=str)
    ap.add_argument("--hit-type-col", default="hit_type", type=str)

    ap.add_argument("--min-pre", default=200, type=int, help="Filter clusters with n_pre < this (0 disables).")
    ap.add_argument("--min-post", default=200, type=int, help="Filter clusters with n_post < this (0 disables).")
    ap.add_argument("--k", default=None, type=int, help="If set, keep only rows with this k (requires 'k' column).")

    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"],
                    help="Alternative hypothesis for MWU/Wilcoxon tests (default: two-sided).")

    # Phrase stats + tiering
    ap.add_argument("--phrase-stats-csv", default=None, type=str,
                    help="If provided, run tiered variance analysis + dim–variance tests using phrase duration stats CSV.")
    ap.add_argument("--top-pct", default=70.0, type=float,
                    help="Percentile threshold WITHIN each bird for High variance (default 70 => top 30%).")
    ap.add_argument("--min-phrases", default=20, type=int,
                    help="Min POST N_phrases to include syllable when computing per-bird percentile threshold, and in dim–variance tests.")

    # Relationship test predictor choice
    ap.add_argument("--dim-predictor", default="post_dim", choices=["post_dim", "delta_dim", "pre_dim"],
                    help="Which dimensionality metric to use as predictor for variance (default: post_dim).")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else (csv_path.parent / "lb_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    _ensure_cols(df, [
        "animal_id", args.hit_type_col, args.group_col,
        "cluster_label", "n_pre", "n_post",
        args.pre_col, args.post_col, args.delta_col
    ], name="LB CSV")

    df = df.copy()
    if args.k is not None and "k" in df.columns:
        df = df[df["k"] == args.k].copy()
    if args.min_pre > 0:
        df = df[df["n_pre"] >= args.min_pre].copy()
    if args.min_post > 0:
        df = df[df["n_post"] >= args.min_post].copy()

    # Keep only hit-types we know how to color
    df = df[df[args.hit_type_col].isin(list(HIT_TYPE_COLORS.keys()))].copy()

    # Ensure 3-group labels exist; else map from hit_type
    if not set(GROUP_ORDER).issubset(set(df[args.group_col].unique())):
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

    # Standard outputs
    overlay_path = out_dir / "pre_vs_post_dim_overlay.png"
    plot_pre_post_overlay(df, overlay_path, pre_col=args.pre_col, post_col=args.post_col, hit_type_col=args.hit_type_col)

    box_path = out_dir / "delta_dim_boxplot_by_group.png"
    stats_path = out_dir / "stats_report.txt"
    plot_delta_boxplot_and_tests(df=df, out_path=box_path, stats_path=stats_path,
                                 group_col=args.group_col, delta_col=args.delta_col, alternative=args.alternative)

    print(f"[OK] Saved standard outputs to: {out_dir}")
    print(f"  - Overlay scatter: {overlay_path}")
    print(f"  - Boxplot: {box_path}")
    print(f"  - Stats report: {stats_path}")

    # Phrase-stats-based analyses
    if args.phrase_stats_csv:
        phrase_csv = Path(args.phrase_stats_csv)
        if not phrase_csv.exists():
            raise FileNotFoundError(phrase_csv)

        post_var = load_post_variance_table(phrase_csv)
        if args.min_phrases > 0:
            post_var = post_var[post_var["post_n_phrases"].fillna(0) >= args.min_phrases].copy()

        post_var = add_high_variance_flag(post_var, top_pct=args.top_pct, min_phrases=0)  # already filtered above

        merged = df.merge(
            post_var,
            how="inner",
            on=["animal_id", "cluster_label"],
            validate="many_to_one",
        )
        merged["is_high_variance"] = merged["is_high_variance"].fillna(False).astype(bool)

        if merged.empty:
            raise ValueError(
                "After merging phrase stats with LB CSV, no rows remained. "
                "Check that phrase 'Syllable' IDs match LB 'cluster_label' and animal IDs match, "
                "or lower --min-phrases."
            )

        mask_hi = merged["is_high_variance"].to_numpy()
        mask_lo = ~mask_hi

        hv_box = out_dir / "highVar_delta_dim_by_lesion_group.png"
        hv_report = out_dir / "highVar_delta_dim_by_lesion_group_stats.txt"
        lv_box = out_dir / "lowVar_delta_dim_by_lesion_group.png"
        lv_report = out_dir / "lowVar_delta_dim_by_lesion_group_stats.txt"

        plot_tier_group_boxplot_and_tests(
            merged=merged,
            out_path=hv_box,
            stats_path=hv_report,
            tier_mask=mask_hi,
            tier_label="High-variance syllables (top 30% within bird)",
            group_col=args.group_col,
            delta_col=args.delta_col,
            alternative=args.alternative,
        )

        plot_tier_group_boxplot_and_tests(
            merged=merged,
            out_path=lv_box,
            stats_path=lv_report,
            tier_mask=mask_lo,
            tier_label="Low-variance syllables (bottom 70% within bird)",
            group_col=args.group_col,
            delta_col=args.delta_col,
            alternative=args.alternative,
        )

        hv_scatter = out_dir / "variance_vs_delta_dim_scatter.png"
        plot_variance_vs_delta_dim_scatter(
            merged=merged,
            out_path=hv_scatter,
            x_col="post_variance_ms2",
            y_col=args.delta_col,
            hit_type_col=args.hit_type_col,
        )

        pred_col = {"post_dim": args.post_col, "pre_dim": args.pre_col, "delta_dim": args.delta_col}[args.dim_predictor]
        rel_report = out_dir / "dim_variance_relationship_report.txt"
        rel_fig = out_dir / "dim_vs_logvar_scatter_panels.png"

        run_dim_variance_tests(
            merged=merged,
            out_report=rel_report,
            out_fig=rel_fig,
            predictor_col=pred_col,
            hit_type_col=args.hit_type_col,
            animal_col="animal_id",
            var_col="post_variance_ms2",
            nphr_col="post_n_phrases",
            npost_col="n_post",
            alternative=args.alternative,
        )

        print(f"[OK] Saved phrase-stats outputs to: {out_dir}")
        print(f"  - High tier group plot: {hv_box}")
        print(f"  - High tier stats:      {hv_report}")
        print(f"  - Low tier group plot:  {lv_box}")
        print(f"  - Low tier stats:       {lv_report}")
        print(f"  - Scatter (var vs Δ):   {hv_scatter}")
        print(f"  - Dim–variance report:  {rel_report}")
        print(f"  - Dim–variance figure:  {rel_fig}")


if __name__ == "__main__":
    main()
