#!/usr/bin/env python3
"""
plot_phrase_duration_quantile_results_Fig3style.py

Visualize the phrase-duration quantile analysis in the same lesion-hit-type
palette used for Figure 3.

INPUT
-----
Required:
    figure3_balanced_pair_metrics.csv

Optional:
    quantile_group_contrasts.csv
        If supplied, the pooled quantile-profile panel marks significant
        M+L-vs-control contrasts using the across-quantile Holm-adjusted
        p-values produced by phrase_duration_quantile_profile_analysis.py.

WHAT THIS SCRIPT MAKES
----------------------
1) Q75 detailed lesion-hit-type percentile boxplot + raw bird points
2) Q90 detailed lesion-hit-type percentile boxplot + raw bird points
3) Pooled Q10-Q90 quantile profile (sham, lateral-only, pooled M+L)
4) Q75 ECDF by detailed lesion hit type
5) A two-panel main-figure candidate:
       pooled quantile profile + detailed Q75 boxplot
6) CSVs containing bird-level detailed quantiles and the pairwise statistics
   used for boxplot annotations.

STATISTICAL ANNOTATIONS ON DETAILED BOX PLOTS
----------------------------------------------
For each displayed quantile, the default planned directional family is:
    Complete M+L > sham
    Partial M+L > sham
    Complete M+L > lateral-only
    Partial M+L > lateral-only

Exact bird-label permutation p-values are Holm-corrected across those four
comparisons WITHIN the displayed quantile. Only Holm-adjusted p < 0.05 is
bracketed by default.

Complete M+L vs Partial M+L is also calculated as a separate, two-sided
secondary contrast and saved to the CSV. It is annotated only if p < 0.05.

Important: these annotations do not replace the multiple-quantile correction
from the full quantile-profile analysis. The pooled profile preferentially
uses the across-quantile Holm p-values from quantile_group_contrasts.csv.

FIGURE 3 PALETTE
----------------
Complete M+L  #3F007D
Partial M+L   #7A4FB7
Lateral-only  #A88BD9
Sham          #1B9E77
Pooled M+L    #7A4FB7
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Canonical lesion groups and Figure 3 colors
# -----------------------------------------------------------------------------

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL = "Partial Medial and Lateral lesion"
COMPLETE = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

DETAIL_ORDER = [SHAM, LATERAL, PARTIAL, COMPLETE]
POOLED_ORDER = [SHAM, LATERAL, POOLED_ML]

COLORS = {
    COMPLETE: "#3F007D",
    PARTIAL: "#7A4FB7",
    LATERAL: "#A88BD9",
    SHAM: "#1B9E77",
    POOLED_ML: "#7A4FB7",
}

SHORT_LABELS = {
    SHAM: "Sham",
    LATERAL: "Lateral-only",
    PARTIAL: "Partial M+L",
    COMPLETE: "Complete M+L",
    POOLED_ML: "Medial+lateral",
}

BOX_LABELS = {
    SHAM: "Sham\nsaline",
    LATERAL: "Lateral\nonly",
    PARTIAL: "Partial\nM+L",
    COMPLETE: "Complete\nM+L",
}

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pair-metrics",
        type=Path,
        required=True,
        help="figure3_balanced_pair_metrics.csv",
    )
    p.add_argument(
        "--pooled-contrasts",
        type=Path,
        default=None,
        help="Optional quantile_group_contrasts.csv from the quantile analysis.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--main-quantile",
        type=float,
        default=0.75,
        help="Quantile for main detailed boxplot. Default: 0.75",
    )
    p.add_argument(
        "--supp-quantile",
        type=float,
        default=0.90,
        help="Quantile for supplemental detailed boxplot. Default: 0.90",
    )
    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
        help="Bird-level bootstrap reps for profile 95%% CIs. Default: 10000",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--annotate-ns",
        action="store_true",
        help="Also draw non-significant planned-comparison brackets on boxplots.",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def canonical_detail_group(value: object) -> str:
    s = " ".join(str(value).strip().split()).lower().replace("_", " ")
    if s in {"sham saline injection", "sham", "sham saline", "saline"}:
        return SHAM
    if s in {
        "lateral lesion only",
        "lateral only",
        "lateral-only",
        "lateral hit only",
    }:
        return LATERAL
    if "partial" in s and "medial" in s and "lateral" in s:
        return PARTIAL
    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE
    raise ValueError(f"Unrecognized lesion_group_detailed value: {value!r}")


def pooled_group(detail_group: str) -> str:
    return POOLED_ML if detail_group in {PARTIAL, COMPLETE} else detail_group


def qlabel(q: float) -> str:
    return f"Q{int(round(100*q))}"


def apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=10, width=1.0, length=4)


def save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int, show: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Bird-level quantiles
# -----------------------------------------------------------------------------

def build_bird_quantiles(pair_df: pd.DataFrame) -> pd.DataFrame:
    required = {"animal_id", "lesion_group_detailed", "delta_cv"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"Missing required pair-metrics columns: {sorted(missing)}")

    work = pair_df.copy()
    work["animal_id"] = work["animal_id"].astype(str).str.strip()
    work["detail_group"] = work["lesion_group_detailed"].map(canonical_detail_group)
    work["delta_cv"] = pd.to_numeric(work["delta_cv"], errors="coerce")
    work = work[np.isfinite(work["delta_cv"])].copy()

    rows = []
    for (bird, detail), sub in work.groupby(["animal_id", "detail_group"], sort=True):
        vals = sub["delta_cv"].to_numpy(dtype=float)
        row = {
            "animal_id": bird,
            "detail_group": detail,
            "detail_label": SHORT_LABELS[detail],
            "pooled_group": pooled_group(detail),
            "pooled_label": SHORT_LABELS[pooled_group(detail)],
            "n_qualifying_syllables": len(vals),
        }
        for q in QUANTILES:
            row[qlabel(q)] = float(np.quantile(vals, q, method="linear"))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


# -----------------------------------------------------------------------------
# Exact / Monte Carlo bird-label permutation tests
# -----------------------------------------------------------------------------

def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
    rng: np.random.Generator,
    max_exact: int = 200000,
    mc_reps: int = 200000,
) -> tuple[float, str, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    nx = len(x)
    total = math.comb(len(pooled), nx)

    def is_extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - 1e-12
        if alternative == "less":
            return value <= observed + 1e-12
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - 1e-12
        raise ValueError(alternative)

    if total <= max_exact:
        extreme = 0
        indices = np.arange(len(pooled))
        for combo in itertools.combinations(indices, nx):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(combo)] = True
            diff = float(np.mean(pooled[mask]) - np.mean(pooled[~mask]))
            extreme += int(is_extreme(diff))
        return extreme / total, "exact", total

    extreme = 0
    for _ in range(mc_reps):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
        extreme += int(is_extreme(diff))
    p = (extreme + 1) / (mc_reps + 1)
    return p, "monte_carlo", mc_reps


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def detailed_stats_for_quantile(
    bird_df: pd.DataFrame,
    q: float,
    seed: int,
) -> pd.DataFrame:
    col = qlabel(q)
    rng = np.random.default_rng(seed + int(round(q * 1000)))

    # Four directional, anatomically planned contrasts.
    primary = [
        (COMPLETE, SHAM, "greater"),
        (PARTIAL, SHAM, "greater"),
        (COMPLETE, LATERAL, "greater"),
        (PARTIAL, LATERAL, "greater"),
    ]

    rows = []
    for g1, g2, alt in primary:
        x = bird_df.loc[bird_df["detail_group"] == g1, col].to_numpy(float)
        y = bird_df.loc[bird_df["detail_group"] == g2, col].to_numpy(float)
        p, method, nperm = permutation_p(x, y, alt, rng)
        rows.append({
            "quantile": q,
            "quantile_label": col,
            "family": "four_planned_directional",
            "group1": g1,
            "group2": g2,
            "alternative": alt,
            "n_group1": len(x),
            "n_group2": len(y),
            "group1_mean": float(np.mean(x)),
            "group2_mean": float(np.mean(y)),
            "mean_difference": float(np.mean(x) - np.mean(y)),
            "p_raw": p,
            "permutation_method": method,
            "permutations_or_assignments": nperm,
        })

    adjusted = holm_adjust([r["p_raw"] for r in rows])
    for row, padj in zip(rows, adjusted):
        row["p_holm_within_quantile"] = float(padj)

    # Separate two-sided secondary question: complete vs partial.
    x = bird_df.loc[bird_df["detail_group"] == COMPLETE, col].to_numpy(float)
    y = bird_df.loc[bird_df["detail_group"] == PARTIAL, col].to_numpy(float)
    p, method, nperm = permutation_p(x, y, "two-sided", rng)
    rows.append({
        "quantile": q,
        "quantile_label": col,
        "family": "complete_vs_partial_secondary",
        "group1": COMPLETE,
        "group2": PARTIAL,
        "alternative": "two-sided",
        "n_group1": len(x),
        "n_group2": len(y),
        "group1_mean": float(np.mean(x)),
        "group2_mean": float(np.mean(y)),
        "mean_difference": float(np.mean(x) - np.mean(y)),
        "p_raw": p,
        "permutation_method": method,
        "permutations_or_assignments": nperm,
        "p_holm_within_quantile": np.nan,
    })

    return pd.DataFrame(rows)


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# -----------------------------------------------------------------------------
# Detailed lesion-hit-type boxplot
# -----------------------------------------------------------------------------

def plot_detailed_box(
    bird_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    q: float,
    out_dir: Path,
    dpi: int,
    seed: int,
    annotate_ns: bool,
    show: bool,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    col = qlabel(q)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
    else:
        fig = ax.figure

    positions = np.arange(1, len(DETAIL_ORDER) + 1)
    group_values = {
        g: bird_df.loc[bird_df["detail_group"] == g, col].dropna().to_numpy(float)
        for g in DETAIL_ORDER
    }

    # Packmor-like percentiles: 5 / 25 / 50 / 75 / 95.
    for xpos, group in zip(positions, DETAIL_ORDER):
        vals = group_values[group]
        if len(vals) == 0:
            continue
        q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])

        width = 0.54
        # box: Q25-Q75
        ax.add_patch(
            plt.Rectangle(
                (xpos - width / 2, q25),
                width,
                q75 - q25,
                facecolor=COLORS[group],
                edgecolor=COLORS[group],
                linewidth=1.4,
                alpha=0.22,
                zorder=1,
            )
        )
        # median
        ax.plot(
            [xpos - width / 2, xpos + width / 2],
            [q50, q50],
            color=COLORS[group],
            linewidth=2.0,
            zorder=2,
        )
        # whiskers 5-95
        ax.plot([xpos, xpos], [q5, q25], color=COLORS[group], linewidth=1.4, zorder=1)
        ax.plot([xpos, xpos], [q75, q95], color=COLORS[group], linewidth=1.4, zorder=1)
        cap = 0.18
        ax.plot([xpos-cap, xpos+cap], [q5, q5], color=COLORS[group], linewidth=1.4)
        ax.plot([xpos-cap, xpos+cap], [q95, q95], color=COLORS[group], linewidth=1.4)

    # Raw bird points
    rng = np.random.default_rng(seed + int(round(q * 1000)))
    for xpos, group in zip(positions, DETAIL_ORDER):
        vals = group_values[group]
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=42,
            facecolor=COLORS[group],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([BOX_LABELS[g] for g in DETAIL_ORDER])
    ax.set_ylabel(f"Bird-level {col} of ΔCV\n(Post − Late Pre)")
    if title:
        ax.set_title(title)
    apply_style(ax)

    # n labels below x-axis using axis coordinates
    for xpos, group in zip(positions, DETAIL_ORDER):
        n = len(group_values[group])
        ax.text(
            xpos,
            -0.13,
            f"n={n}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    # Significance brackets
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    data_max = max(np.max(v) for v in group_values.values() if len(v))
    base = max(data_max, 0) + 0.07 * span
    step = 0.10 * span

    group_pos = {g: i for i, g in enumerate(DETAIL_ORDER, start=1)}

    annotation_rows = []

    # Primary family uses Holm-adjusted p.
    primary = stats_df[stats_df["family"] == "four_planned_directional"].copy()
    for _, row in primary.sort_values("p_holm_within_quantile").iterrows():
        p = float(row["p_holm_within_quantile"])
        if p < 0.05 or annotate_ns:
            annotation_rows.append(
                (row["group1"], row["group2"], p, "Holm")
            )

    # Complete vs partial is a separate two-sided secondary contrast.
    secondary = stats_df[stats_df["family"] == "complete_vs_partial_secondary"]
    if not secondary.empty:
        row = secondary.iloc[0]
        p = float(row["p_raw"])
        if p < 0.05 or annotate_ns:
            annotation_rows.append(
                (row["group1"], row["group2"], p, "two-sided")
            )

    for k, (g1, g2, p, _) in enumerate(annotation_rows):
        x1, x2 = sorted([group_pos[g1], group_pos[g2]])
        y = base + k * step
        h = 0.022 * span
        ax.plot(
            [x1, x1, x2, x2],
            [y, y+h, y+h, y],
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        label = p_to_stars(p)
        ax.text((x1+x2)/2, y+h + 0.012*span, label, ha="center", va="bottom", fontsize=11)

    if annotation_rows:
        upper_needed = base + (len(annotation_rows)-1)*step + 0.08*span
        ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], upper_needed))

    if own_fig:
        fig.tight_layout()
        save(
            fig,
            out_dir,
            f"{col}_deltaCV_by_lesion_hit_type_percentile_boxplot",
            dpi,
            show,
        )
    return fig


# -----------------------------------------------------------------------------
# Pooled quantile profile
# -----------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def pooled_profile_summary(
    bird_df: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for group in POOLED_ORDER:
        sub = bird_df[bird_df["pooled_group"] == group]
        for q in QUANTILES:
            col = qlabel(q)
            vals = sub[col].dropna().to_numpy(float)
            lo, hi = bootstrap_mean_ci(vals, rng, bootstrap_reps)
            rows.append({
                "group": group,
                "quantile": q,
                "quantile_label": col,
                "mean": float(np.mean(vals)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_birds": len(vals),
            })
    return pd.DataFrame(rows)


def significance_lookup_from_pooled_contrasts(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "quantile_label",
        "comparison",
        "p_one_sided_holm_ml_primary_across_quantiles",
    }
    if not required.issubset(df.columns):
        raise ValueError(
            "pooled contrasts CSV does not contain the across-quantile Holm "
            "columns expected from phrase_duration_quantile_profile_analysis.py"
        )

    out = {}
    for _, row in df.iterrows():
        if row["comparison"] not in {"M+L vs sham", "M+L vs lateral-only"}:
            continue
        out[(str(row["quantile_label"]), str(row["comparison"]))] = float(
            row["p_one_sided_holm_ml_primary_across_quantiles"]
        )
    return out


def plot_pooled_profile(
    bird_df: pd.DataFrame,
    pooled_contrast_path: Path | None,
    out_dir: Path,
    bootstrap_reps: int,
    seed: int,
    dpi: int,
    show: bool,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    summary = pooled_profile_summary(bird_df, bootstrap_reps, seed)
    sig = significance_lookup_from_pooled_contrasts(pooled_contrast_path)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
    else:
        fig = ax.figure

    x = np.array([10, 25, 50, 75, 90], dtype=float)

    for group in POOLED_ORDER:
        sub = summary[summary["group"] == group].set_index("quantile_label")
        means = np.array([sub.loc[qlabel(q), "mean"] for q in QUANTILES])
        lows = np.array([sub.loc[qlabel(q), "ci_low"] for q in QUANTILES])
        highs = np.array([sub.loc[qlabel(q), "ci_high"] for q in QUANTILES])
        yerr = np.vstack([means - lows, highs - means])

        ax.errorbar(
            x,
            means,
            yerr=yerr,
            color=COLORS[group],
            marker="o",
            markersize=6.5,
            linewidth=2.1,
            elinewidth=1.4,
            capsize=3.5,
            label=SHORT_LABELS[group],
            zorder=3,
        )

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Within-bird quantile of syllable-level ΔCV")
    ax.set_ylabel("Post − Late Pre phrase-duration CV")
    if title:
        ax.set_title(title)
    apply_style(ax)
    ax.legend(frameon=False, loc="upper left")

    # Minimal significance strip inside the bottom of the axes:
    # filled star = across-quantile Holm p < .05.
    # It uses axes coordinates so the labels never spill outside the panel.
    if sig:
        ax.text(
            0.015, 0.090, "M+L vs sham",
            transform=ax.transAxes,
            ha="left", va="center", fontsize=8.3, color=COLORS[SHAM]
        )
        ax.text(
            0.015, 0.040, "M+L vs lateral",
            transform=ax.transAxes,
            ha="left", va="center", fontsize=8.3, color=COLORS[LATERAL]
        )

        x_min, x_max = ax.get_xlim()
        for xpos, q in zip(x, QUANTILES):
            label = qlabel(q)
            p_sham = sig.get((label, "M+L vs sham"), np.nan)
            p_lat = sig.get((label, "M+L vs lateral-only"), np.nan)
            xfrac = (xpos - x_min) / (x_max - x_min)

            if np.isfinite(p_sham) and p_sham < 0.05:
                ax.text(
                    xfrac, 0.090, "*",
                    transform=ax.transAxes,
                    color=COLORS[SHAM],
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                )
            if np.isfinite(p_lat) and p_lat < 0.05:
                ax.text(
                    xfrac, 0.040, "*",
                    transform=ax.transAxes,
                    color=COLORS[LATERAL],
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                )

    if own_fig:
        fig.tight_layout()
        save(
            fig,
            out_dir,
            "pooled_quantile_profile_with_across_quantile_significance",
            dpi,
            show,
        )
    return fig


# -----------------------------------------------------------------------------
# ECDF
# -----------------------------------------------------------------------------

def plot_ecdf(
    bird_df: pd.DataFrame,
    q: float,
    out_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    col = qlabel(q)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    for group in DETAIL_ORDER:
        vals = np.sort(
            bird_df.loc[bird_df["detail_group"] == group, col].dropna().to_numpy(float)
        )
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(
            vals,
            y,
            where="post",
            color=COLORS[group],
            linewidth=2.0,
            label=f"{SHORT_LABELS[group]} (n={len(vals)})",
        )
        ax.scatter(
            vals,
            y,
            s=22,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Bird-level {col} of ΔCV (Post − Late Pre)")
    ax.set_ylabel("Cumulative proportion of birds")
    ax.set_ylim(0, 1.04)
    apply_style(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    save(
        fig,
        out_dir,
        f"{col}_deltaCV_by_lesion_hit_type_ECDF",
        dpi,
        show,
    )


# -----------------------------------------------------------------------------
# Combined main-figure candidate
# -----------------------------------------------------------------------------

def plot_combined_main_candidate(
    bird_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    pooled_contrast_path: Path | None,
    q: float,
    out_dir: Path,
    bootstrap_reps: int,
    seed: int,
    dpi: int,
    annotate_ns: bool,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.2))

    plot_pooled_profile(
        bird_df,
        pooled_contrast_path,
        out_dir,
        bootstrap_reps,
        seed,
        dpi,
        False,
        ax=axes[0],
        title=None,
    )

    plot_detailed_box(
        bird_df,
        stats_df,
        q,
        out_dir,
        dpi,
        seed,
        annotate_ns,
        False,
        ax=axes[1],
        title=None,
    )

    axes[0].text(
        -0.12, 1.04, "A",
        transform=axes[0].transAxes,
        fontsize=14, fontweight="bold",
        va="top",
    )
    axes[1].text(
        -0.12, 1.04, "B",
        transform=axes[1].transAxes,
        fontsize=14, fontweight="bold",
        va="top",
    )

    fig.tight_layout(w_pad=2.0)
    save(
        fig,
        out_dir,
        f"main_candidate_quantile_profile_plus_{qlabel(q)}_hit_type_boxplot",
        dpi,
        show,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.main_quantile not in QUANTILES:
        raise ValueError(f"--main-quantile must be one of {QUANTILES}")
    if args.supp_quantile not in QUANTILES:
        raise ValueError(f"--supp-quantile must be one of {QUANTILES}")

    pair_path = args.pair_metrics.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    pair_df = pd.read_csv(pair_path)
    bird_df = build_bird_quantiles(pair_df)
    bird_df.to_csv(out_dir / "bird_quantiles_by_detailed_hit_type.csv", index=False)

    stats_frames = []
    for q in sorted(set([args.main_quantile, args.supp_quantile])):
        stats_frames.append(
            detailed_stats_for_quantile(bird_df, q, seed=args.seed)
        )
    detailed_stats = pd.concat(stats_frames, ignore_index=True)
    detailed_stats.to_csv(
        out_dir / "detailed_hit_type_pairwise_stats.csv",
        index=False,
    )

    # Individual figure files
    for q in [args.main_quantile, args.supp_quantile]:
        qstats = detailed_stats[detailed_stats["quantile"] == q]
        plot_detailed_box(
            bird_df,
            qstats,
            q,
            out_dir,
            args.dpi,
            args.seed,
            args.annotate_ns,
            args.show,
        )

    plot_pooled_profile(
        bird_df,
        args.pooled_contrasts.expanduser().resolve()
        if args.pooled_contrasts is not None else None,
        out_dir,
        args.bootstrap_reps,
        args.seed,
        args.dpi,
        args.show,
    )

    plot_ecdf(
        bird_df,
        args.main_quantile,
        out_dir,
        args.dpi,
        args.show,
    )

    main_stats = detailed_stats[
        detailed_stats["quantile"] == args.main_quantile
    ]
    plot_combined_main_candidate(
        bird_df,
        main_stats,
        args.pooled_contrasts.expanduser().resolve()
        if args.pooled_contrasts is not None else None,
        args.main_quantile,
        out_dir,
        args.bootstrap_reps,
        args.seed,
        args.dpi,
        args.annotate_ns,
        args.show,
    )

    print("\nBird counts:")
    for group in DETAIL_ORDER:
        n = int((bird_df["detail_group"] == group).sum())
        print(f"  {SHORT_LABELS[group]}: n={n}")

    print("\nDetailed statistics:")
    show_cols = [
        "quantile_label",
        "family",
        "group1",
        "group2",
        "alternative",
        "mean_difference",
        "p_raw",
        "p_holm_within_quantile",
    ]
    print(detailed_stats[show_cols].to_string(index=False))

    print("\nDone.")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
