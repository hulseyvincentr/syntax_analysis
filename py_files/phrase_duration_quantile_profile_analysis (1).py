#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phrase-duration quantile-profile analysis (no syllable selection)
===============================================================

Purpose
-------
Replace the top-X%-syllable selection step with bird-level quantiles of the
FULL distribution of qualifying syllable-level changes.

This script intentionally starts from the already-generated
``figure3_balanced_pair_metrics.csv`` table. Therefore the upstream Figure 3
pipeline is unchanged:

    * same Late Pre / Post windows
    * same usage/count balancing
    * same 200 balancing draws used to create the input table
    * same >=10 renditions per period qualification
    * same syllable-level Delta CV calculation

The ONLY analytical swap is downstream:

    OLD: rank syllables -> retain top fraction -> median Delta CV per bird
    NEW: keep every qualifying syllable -> Q10/Q25/Q50/Q75/Q90 Delta CV per bird

Each bird remains one independent observation. Between-group inference uses the
same bird-label permutation logic as the prior Figure 3 analysis: the test
statistic is mean(group 1 bird-level summary) - mean(group 2 bird-level summary).

Default pairwise tests, at every quantile:
    1. M+L > lateral-only
    2. M+L > sham
    3. lateral-only > sham

Holm correction is reported in three ways:
    * within_quantile: the same 3-comparison family used in the earlier analysis
    * all_quantile_tests: all 3 comparisons x all requested quantiles
    * ml_primary_across_quantiles: only the two M+L-vs-control contrasts across
      all requested quantiles (useful as a stricter sensitivity analysis)

The script also tests upper-tail amplification using Q75-Q50 and Q90-Q50. A
uniform repertoire-wide shift should move the median and upper tail together;
a concentrated upper-tail effect should increase these within-bird contrasts.

Outputs
-------
CSV:
    input_bird_audit.csv
    bird_quantile_values_long.csv
    bird_quantile_values_wide.csv
    group_quantile_summary.csv
    quantile_group_contrasts.csv
    upper_tail_bird_values.csv
    upper_tail_group_contrasts.csv

Text:
    quantile_analysis_summary.txt

Figures:
    quantile_profile_by_group.png
    quantile_contrast_profile.png

Example
-------
python phrase_duration_quantile_profile_analysis.py \
    figure3_balanced_pair_metrics.csv \
    --output Figure3_quantile_analysis \
    --metric delta_cv \
    --quantiles 0.10 0.25 0.50 0.75 0.90 \
    --bootstrap-reps 10000 \
    --seed 123

Dependencies
------------
numpy, pandas, matplotlib
(scientific tests are implemented directly; scipy is not required)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Canonical groups / display
# -----------------------------------------------------------------------------

ML = "medial_and_lateral"
LATERAL = "lateral_only"
SHAM = "sham_saline"

GROUP_ORDER = [SHAM, LATERAL, ML]
GROUP_LABELS = {
    SHAM: "Sham",
    LATERAL: "Lateral-only",
    ML: "Medial+lateral",
}

# Figure-3-style palette. These are only presentation choices and do not affect
# any statistic.
GROUP_COLORS = {
    SHAM: "#1FA187",
    LATERAL: "#B39DDB",
    ML: "#7E57C2",
}

PAIRWISE_COMPARISONS = [
    (ML, LATERAL, "M+L vs lateral-only"),
    (ML, SHAM, "M+L vs sham"),
    (LATERAL, SHAM, "Lateral-only vs sham"),
]

METRIC_LABELS = {
    "delta_cv": "Post - Late Pre phrase-duration CV",
    "delta_sd_s": "Post - Late Pre phrase-duration SD (s)",
    "delta_mean_s": "Post - Late Pre mean phrase duration (s)",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="No-selection quantile-profile analysis of syllable-level phrase-duration changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input_csv",
        type=Path,
        help="figure3_balanced_pair_metrics.csv from the unchanged upstream Figure 3 pipeline.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Figure3_quantile_analysis"),
        help="Output directory.",
    )
    p.add_argument(
        "--metric",
        default="delta_cv",
        help="Syllable-level change column to summarize by quantile.",
    )
    p.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 0.90],
        help="Bird-level quantiles to calculate from all qualifying syllables.",
    )
    p.add_argument(
        "--quantile-method",
        default="linear",
        choices=["linear", "lower", "higher", "midpoint", "nearest"],
        help="NumPy sample-quantile interpolation method.",
    )
    p.add_argument(
        "--min-required-phrases",
        type=int,
        default=10,
        help="Validation threshold only; rows are NOT reselected by this script.",
    )
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    p.add_argument(
        "--max-exact-assignments",
        type=int,
        default=200000,
        help="Use exact label permutation when the number of group-label assignments is <= this value.",
    )
    p.add_argument(
        "--monte-carlo-permutations",
        type=int,
        default=100000,
        help="Fallback label permutations when exact enumeration would be too large.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save PDF copies of the two figures.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def normalize_group(value: object) -> str:
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    compact = "".join(ch for ch in text if ch.isalnum())

    if compact in {"medialandlateral", "mediallateral", "ml"}:
        return ML
    if "medial" in text and "lateral" in text:
        return ML
    if "large lesion" in text or "area x not visible" in text:
        return ML

    if compact in {"lateralonly", "laterallesiononly", "lateralhitonly"}:
        return LATERAL
    if "lateral" in text and "only" in text:
        return LATERAL
    if "single" in text and "hit" in text:
        return LATERAL

    if compact in {"shamsaline", "sham", "saline"}:
        return SHAM
    if "sham" in text or "saline" in text:
        return SHAM

    return compact or "unknown"


def q_label(q: float) -> str:
    value = 100.0 * q
    if math.isclose(value, round(value), abs_tol=1e-10):
        return f"Q{int(round(value))}"
    return f"Q{value:g}"


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(x) for x in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def finite(values: Iterable[object]) -> np.ndarray:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    return x[np.isfinite(x)]


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = finite(values)
    if values.size == 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# -----------------------------------------------------------------------------
# Input loading / validation
# -----------------------------------------------------------------------------

def choose_group_column(df: pd.DataFrame) -> str:
    for col in ["lesion_group", "lesion_group_detailed", "display_group", "group"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a lesion-group column. Tried lesion_group, "
        "lesion_group_detailed, display_group, and group."
    )


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)
    required = {"animal_id", "syllable", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_col = choose_group_column(df)

    work = df.copy()
    work["animal_id"] = work["animal_id"].astype(str).str.strip()
    work["syllable"] = work["syllable"].astype(str).str.strip()
    work["analysis_group"] = work[group_col].map(normalize_group)
    work[args.metric] = pd.to_numeric(work[args.metric], errors="coerce")

    unknown = sorted(set(work["analysis_group"]) - {ML, LATERAL, SHAM})
    if unknown:
        raise ValueError(
            "Unrecognized analysis groups after normalization: " + ", ".join(unknown)
        )

    duplicates = work.duplicated(["animal_id", "syllable"], keep=False)
    if duplicates.any():
        example = work.loc[duplicates, ["animal_id", "syllable"]].head(10)
        raise ValueError(
            "Input contains duplicate animal_id x syllable rows. Example:\n"
            + example.to_string(index=False)
        )

    # This is an audit, not a new selection step. The upstream table should
    # already contain only qualifying syllables.
    for count_col in ["n_pre_phrases", "n_post_phrases"]:
        if count_col in work.columns:
            counts = pd.to_numeric(work[count_col], errors="coerce")
            bad = counts < args.min_required_phrases
            if bad.fillna(False).any():
                raise ValueError(
                    f"{int(bad.sum())} rows have {count_col} < "
                    f"{args.min_required_phrases}. This script is meant to start "
                    "from the already-qualified Figure 3 pair table; investigate "
                    "the upstream input rather than silently filtering here."
                )

    n_nonfinite = int((~np.isfinite(work[args.metric])).sum())
    if n_nonfinite:
        print(f"[WARN] Dropping {n_nonfinite} rows with non-finite {args.metric}.")
        work = work[np.isfinite(work[args.metric])].copy()

    # One group per bird is required for label permutation.
    group_counts = work.groupby("animal_id")["analysis_group"].nunique()
    bad_birds = group_counts[group_counts != 1]
    if not bad_birds.empty:
        raise ValueError(
            "Some birds map to more than one analysis group:\n" + bad_birds.to_string()
        )

    audit_rows = []
    for bird, g in work.groupby("animal_id", sort=True):
        row = {
            "animal_id": bird,
            "group": g["analysis_group"].iloc[0],
            "group_label": GROUP_LABELS[g["analysis_group"].iloc[0]],
            "n_qualifying_syllables": int(g["syllable"].nunique()),
            "metric": args.metric,
            "metric_min": float(g[args.metric].min()),
            "metric_median": float(g[args.metric].median()),
            "metric_max": float(g[args.metric].max()),
        }
        if "n_pre_phrases" in g.columns:
            row["min_n_pre_phrases"] = int(pd.to_numeric(g["n_pre_phrases"]).min())
        if "n_post_phrases" in g.columns:
            row["min_n_post_phrases"] = int(pd.to_numeric(g["n_post_phrases"]).min())
        if "n_pre_days" in g.columns:
            row["min_n_pre_days"] = int(pd.to_numeric(g["n_pre_days"]).min())
        if "n_post_days" in g.columns:
            row["min_n_post_days"] = int(pd.to_numeric(g["n_post_days"]).min())
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows)
    return work.reset_index(drop=True), audit


# -----------------------------------------------------------------------------
# Bird-level quantiles
# -----------------------------------------------------------------------------

def build_bird_quantiles(
    df: pd.DataFrame,
    metric: str,
    quantiles: Sequence[float],
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for bird, g in df.groupby("animal_id", sort=True):
        group = str(g["analysis_group"].iloc[0])
        values = finite(g[metric])
        if values.size == 0:
            continue

        for q in quantiles:
            value = float(np.quantile(values, q, method=method))
            rows.append(
                {
                    "animal_id": bird,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "metric": metric,
                    "quantile": float(q),
                    "quantile_label": q_label(q),
                    "n_qualifying_syllables": int(values.size),
                    "quantile_value": value,
                    "quantile_method": method,
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["quantile", "group", "animal_id"]).reset_index(drop=True)


def quantiles_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(
        index=["animal_id", "group", "group_label", "n_qualifying_syllables"],
        columns="quantile_label",
        values="quantile_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


# -----------------------------------------------------------------------------
# Bootstrap helpers
# -----------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = finite(values)
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(values, size=values.size, replace=True)
        draws[i] = float(np.mean(sample))
    return percentile_ci(draws, confidence=confidence)


def bootstrap_mean_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reps: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    x = finite(x)
    y = finite(y)
    if x.size == 0 or y.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        draws[i] = float(np.mean(xb) - np.mean(yb))
    return percentile_ci(draws, confidence=confidence)


def summarize_groups(
    bird_quantiles: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for (q, qlab, group), g in bird_quantiles.groupby(
        ["quantile", "quantile_label", "group"], sort=True
    ):
        values = finite(g["quantile_value"])
        lo, hi = bootstrap_mean_ci(
            values,
            reps=bootstrap_reps,
            seed=stable_seed(seed, "group_ci", q, group),
        )
        rows.append(
            {
                "quantile": float(q),
                "quantile_label": qlab,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "n_birds": int(values.size),
                "mean_bird_quantile": float(np.mean(values)),
                "median_bird_quantile": float(np.median(values)),
                "sd_bird_quantile": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
                "bootstrap_95ci_low_mean": lo,
                "bootstrap_95ci_high_mean": hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["quantile", "group"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Exact / Monte Carlo bird-label permutation tests
# -----------------------------------------------------------------------------

def permutation_test_mean_difference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Test mean(x) - mean(y); one-sided alternative is x > y."""
    x = finite(x)
    y = finite(y)
    if x.size < 1 or y.size < 1:
        return {
            "mean_difference_group1_minus_group2": math.nan,
            "p_one_sided": math.nan,
            "p_two_sided": math.nan,
            "permutation_method": "none",
            "permutations_or_assignments": 0,
            "possible_exact_assignments": 0,
        }

    pooled = np.concatenate([x, y])
    nx = x.size
    observed = float(np.mean(x) - np.mean(y))
    total_assignments = math.comb(pooled.size, nx)
    tol = 1e-12

    if total_assignments <= max_exact_assignments:
        extreme_one = 0
        extreme_two = 0
        total = 0
        all_idx = np.arange(pooled.size)
        for x_idx_tuple in itertools.combinations(range(pooled.size), nx):
            x_idx = np.fromiter(x_idx_tuple, dtype=int)
            mask = np.ones(pooled.size, dtype=bool)
            mask[x_idx] = False
            y_idx = all_idx[mask]
            null_diff = float(np.mean(pooled[x_idx]) - np.mean(pooled[y_idx]))
            extreme_one += int(null_diff >= observed - tol)
            extreme_two += int(abs(null_diff) >= abs(observed) - tol)
            total += 1
        p_one = extreme_one / total
        p_two = extreme_two / total
        method = "exact"
        used = total
    else:
        rng = np.random.default_rng(seed)
        extreme_one = 0
        extreme_two = 0
        for _ in range(monte_carlo_permutations):
            perm = rng.permutation(pooled)
            null_diff = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
            extreme_one += int(null_diff >= observed - tol)
            extreme_two += int(abs(null_diff) >= abs(observed) - tol)
        p_one = (extreme_one + 1) / (monte_carlo_permutations + 1)
        p_two = (extreme_two + 1) / (monte_carlo_permutations + 1)
        method = "monte_carlo"
        used = monte_carlo_permutations

    return {
        "mean_difference_group1_minus_group2": observed,
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "permutation_method": method,
        "permutations_or_assignments": int(used),
        "possible_exact_assignments": int(total_assignments),
    }


def build_quantile_group_contrasts(
    bird_quantiles: pd.DataFrame,
    *,
    bootstrap_reps: int,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for q in sorted(bird_quantiles["quantile"].unique()):
        qdf = bird_quantiles[bird_quantiles["quantile"] == q]
        qlab = q_label(float(q))

        for group1, group2, label in PAIRWISE_COMPARISONS:
            x = finite(qdf.loc[qdf["group"] == group1, "quantile_value"])
            y = finite(qdf.loc[qdf["group"] == group2, "quantile_value"])

            perm = permutation_test_mean_difference(
                x,
                y,
                max_exact_assignments=max_exact_assignments,
                monte_carlo_permutations=monte_carlo_permutations,
                seed=stable_seed(seed, "perm", q, group1, group2),
            )
            ci_low, ci_high = bootstrap_mean_difference_ci(
                x,
                y,
                reps=bootstrap_reps,
                seed=stable_seed(seed, "diff_ci", q, group1, group2),
            )

            rows.append(
                {
                    "quantile": float(q),
                    "quantile_label": qlab,
                    "comparison": label,
                    "group1": group1,
                    "group2": group2,
                    "alternative": "group1 > group2",
                    "n_group1": int(x.size),
                    "n_group2": int(y.size),
                    "group1_mean": float(np.mean(x)),
                    "group1_median": float(np.median(x)),
                    "group2_mean": float(np.mean(y)),
                    "group2_median": float(np.median(y)),
                    **perm,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                }
            )

    out = pd.DataFrame(rows)

    # Main correction: same three pairwise group comparisons within each quantile.
    out["p_one_sided_holm_within_quantile"] = np.nan
    out["p_two_sided_holm_within_quantile"] = np.nan
    for q, idx in out.groupby("quantile").groups.items():
        idx = list(idx)
        out.loc[idx, "p_one_sided_holm_within_quantile"] = holm_adjust(
            out.loc[idx, "p_one_sided"].to_numpy(float)
        )
        out.loc[idx, "p_two_sided_holm_within_quantile"] = holm_adjust(
            out.loc[idx, "p_two_sided"].to_numpy(float)
        )

    # Stricter sensitivity correction across the entire quantile-profile family.
    out["p_one_sided_holm_all_quantile_tests"] = holm_adjust(
        out["p_one_sided"].to_numpy(float)
    )

    # Another useful sensitivity family: the two M+L-vs-control contrasts across
    # all quantiles. Lateral-vs-sham is excluded because it is not part of the
    # primary anatomical claim.
    out["p_one_sided_holm_ml_primary_across_quantiles"] = np.nan
    primary = out["group1"].eq(ML) & out["group2"].isin([LATERAL, SHAM])
    out.loc[primary, "p_one_sided_holm_ml_primary_across_quantiles"] = holm_adjust(
        out.loc[primary, "p_one_sided"].to_numpy(float)
    )

    return out.sort_values(["quantile", "comparison"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Upper-tail amplification: Q75-Q50 and Q90-Q50
# -----------------------------------------------------------------------------

def find_quantile_column(wide: pd.DataFrame, target: float) -> str | None:
    label = q_label(target)
    return label if label in wide.columns else None


def build_upper_tail_values(wide: pd.DataFrame) -> pd.DataFrame:
    q50 = find_quantile_column(wide, 0.50)
    q75 = find_quantile_column(wide, 0.75)
    q90 = find_quantile_column(wide, 0.90)

    required = {"animal_id", "group", "group_label", "n_qualifying_syllables"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Internal error: wide quantile table missing {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        base = {
            "animal_id": row["animal_id"],
            "group": row["group"],
            "group_label": row["group_label"],
            "n_qualifying_syllables": row["n_qualifying_syllables"],
        }
        if q50 is not None and q75 is not None:
            rows.append(
                {
                    **base,
                    "tail_metric": "Q75-Q50",
                    "tail_value": float(row[q75] - row[q50]),
                }
            )
        if q50 is not None and q90 is not None:
            rows.append(
                {
                    **base,
                    "tail_metric": "Q90-Q50",
                    "tail_value": float(row[q90] - row[q50]),
                }
            )

    return pd.DataFrame(rows)


def build_upper_tail_contrasts(
    tail_values: pd.DataFrame,
    *,
    bootstrap_reps: int,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if tail_values.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for tail_metric, tdf in tail_values.groupby("tail_metric", sort=True):
        for group1, group2, label in PAIRWISE_COMPARISONS:
            x = finite(tdf.loc[tdf["group"] == group1, "tail_value"])
            y = finite(tdf.loc[tdf["group"] == group2, "tail_value"])
            perm = permutation_test_mean_difference(
                x,
                y,
                max_exact_assignments=max_exact_assignments,
                monte_carlo_permutations=monte_carlo_permutations,
                seed=stable_seed(seed, "tail_perm", tail_metric, group1, group2),
            )
            ci_low, ci_high = bootstrap_mean_difference_ci(
                x,
                y,
                reps=bootstrap_reps,
                seed=stable_seed(seed, "tail_ci", tail_metric, group1, group2),
            )
            rows.append(
                {
                    "tail_metric": tail_metric,
                    "comparison": label,
                    "group1": group1,
                    "group2": group2,
                    "alternative": "group1 > group2",
                    "n_group1": int(x.size),
                    "n_group2": int(y.size),
                    "group1_mean": float(np.mean(x)),
                    "group2_mean": float(np.mean(y)),
                    **perm,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                }
            )

    out = pd.DataFrame(rows)
    out["p_one_sided_holm_within_tail_metric"] = np.nan
    for metric, idx in out.groupby("tail_metric").groups.items():
        idx = list(idx)
        out.loc[idx, "p_one_sided_holm_within_tail_metric"] = holm_adjust(
            out.loc[idx, "p_one_sided"].to_numpy(float)
        )
    out["p_one_sided_holm_all_tail_tests"] = holm_adjust(
        out["p_one_sided"].to_numpy(float)
    )
    return out.sort_values(["tail_metric", "comparison"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def setup_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def plot_group_profile(
    group_summary: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    for group in GROUP_ORDER:
        g = group_summary[group_summary["group"] == group].sort_values("quantile")
        if g.empty:
            continue
        x = 100.0 * g["quantile"].to_numpy(float)
        y = g["mean_bird_quantile"].to_numpy(float)
        lo = g["bootstrap_95ci_low_mean"].to_numpy(float)
        hi = g["bootstrap_95ci_high_mean"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            markersize=6,
            capsize=3,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )

    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xticks(sorted(100.0 * group_summary["quantile"].unique()))
    ax.set_xlabel("Within-bird quantile of syllable-level change")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.legend(frameon=False)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_contrast_profile(
    contrasts: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    comparison_styles = [
        ("M+L vs lateral-only", "#7E57C2", "o"),
        ("M+L vs sham", "#1FA187", "s"),
    ]

    for comparison, color, marker in comparison_styles:
        g = contrasts[contrasts["comparison"] == comparison].sort_values("quantile")
        if g.empty:
            continue
        x = 100.0 * g["quantile"].to_numpy(float)
        y = g["mean_difference_group1_minus_group2"].to_numpy(float)
        lo = g["bootstrap_95ci_low"].to_numpy(float)
        hi = g["bootstrap_95ci_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            capsize=3,
            color=color,
            label=comparison,
        )

    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xticks(sorted(100.0 * contrasts["quantile"].unique()))
    ax.set_xlabel("Within-bird quantile of syllable-level change")
    ylabel = "Difference in bird-level quantile"
    if metric == "delta_cv":
        ylabel += " (Delta CV)"
    elif metric == "delta_sd_s":
        ylabel += " (Delta SD, s)"
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Summary text
# -----------------------------------------------------------------------------

def fmt_p(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    if x < 0.0001:
        return f"{x:.2e}"
    return f"{x:.4f}"


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    input_df: pd.DataFrame,
    audit: pd.DataFrame,
    contrasts: pd.DataFrame,
    tail_contrasts: pd.DataFrame,
) -> None:
    group_counts = audit.groupby("group")["animal_id"].nunique().to_dict()
    lines = [
        "Phrase-duration quantile-profile analysis",
        "=========================================",
        "",
        f"Input: {args.input_csv}",
        f"Metric: {args.metric}",
        f"Quantiles: {', '.join(q_label(q) for q in args.quantiles)}",
        f"Quantile method: {args.quantile_method}",
        f"Qualifying syllable rows: {len(input_df)}",
        f"Birds: {audit['animal_id'].nunique()}",
        f"  M+L: {group_counts.get(ML, 0)}",
        f"  Lateral-only: {group_counts.get(LATERAL, 0)}",
        f"  Sham: {group_counts.get(SHAM, 0)}",
        f"Qualifying syllables per bird: median={audit['n_qualifying_syllables'].median():.1f}, "
        f"range={audit['n_qualifying_syllables'].min()}-{audit['n_qualifying_syllables'].max()}",
        "",
        "Important design note",
        "---------------------",
        "No syllable ranking or top-fraction selection is performed here. Every row",
        "in the already-qualified Figure 3 pair-metrics table contributes to the",
        "within-bird Delta distribution. Each bird is then summarized by quantiles.",
        "",
        "Pointwise quantile contrasts",
        "----------------------------",
        "Effect = mean bird-level quantile(group1) - mean bird-level quantile(group2).",
        "The one-sided permutation alternative is group1 > group2.",
        "Holm-within-quantile is the direct analogue of the previous 3-comparison",
        "Holm family. The CSV also contains stricter across-quantile corrections.",
        "",
    ]

    for q in sorted(contrasts["quantile"].unique()):
        lines.append(q_label(float(q)))
        qdf = contrasts[contrasts["quantile"] == q]
        for comparison in ["M+L vs lateral-only", "M+L vs sham", "Lateral-only vs sham"]:
            row = qdf[qdf["comparison"] == comparison]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"  {comparison}: effect={r['mean_difference_group1_minus_group2']:.6g}, "
                f"one-sided p={fmt_p(r['p_one_sided'])}, "
                f"Holm-within-quantile={fmt_p(r['p_one_sided_holm_within_quantile'])}, "
                f"95% bootstrap CI=[{r['bootstrap_95ci_low']:.6g}, {r['bootstrap_95ci_high']:.6g}]"
            )
        lines.append("")

    if not tail_contrasts.empty:
        lines.extend([
            "Upper-tail amplification contrasts",
            "----------------------------------",
            "These test whether the upper tail is elevated relative to the median",
            "within the same bird (Q75-Q50 or Q90-Q50).",
            "",
        ])
        for metric in sorted(tail_contrasts["tail_metric"].unique()):
            lines.append(metric)
            mdf = tail_contrasts[tail_contrasts["tail_metric"] == metric]
            for comparison in ["M+L vs lateral-only", "M+L vs sham", "Lateral-only vs sham"]:
                row = mdf[mdf["comparison"] == comparison]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"  {comparison}: effect={r['mean_difference_group1_minus_group2']:.6g}, "
                    f"one-sided p={fmt_p(r['p_one_sided'])}, "
                    f"Holm-within-tail={fmt_p(r['p_one_sided_holm_within_tail_metric'])}, "
                    f"95% bootstrap CI=[{r['bootstrap_95ci_low']:.6g}, {r['bootstrap_95ci_high']:.6g}]"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.quantiles:
        raise ValueError("At least one quantile is required.")
    args.quantiles = sorted(set(float(q) for q in args.quantiles))
    if any(q <= 0 or q >= 1 for q in args.quantiles):
        raise ValueError("Quantiles must be strictly between 0 and 1.")
    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be at least 100.")
    if args.monte_carlo_permutations < 100:
        raise ValueError("--monte-carlo-permutations must be at least 100.")

    args.output.mkdir(parents=True, exist_ok=True)

    input_df, audit = load_input(args)
    audit.to_csv(args.output / "input_bird_audit.csv", index=False)

    bird_quantiles = build_bird_quantiles(
        input_df,
        args.metric,
        args.quantiles,
        args.quantile_method,
    )
    bird_quantiles.to_csv(args.output / "bird_quantile_values_long.csv", index=False)

    wide = quantiles_wide(bird_quantiles)
    wide.to_csv(args.output / "bird_quantile_values_wide.csv", index=False)

    group_summary = summarize_groups(
        bird_quantiles,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    group_summary.to_csv(args.output / "group_quantile_summary.csv", index=False)

    contrasts = build_quantile_group_contrasts(
        bird_quantiles,
        bootstrap_reps=args.bootstrap_reps,
        max_exact_assignments=args.max_exact_assignments,
        monte_carlo_permutations=args.monte_carlo_permutations,
        seed=args.seed,
    )
    contrasts.to_csv(args.output / "quantile_group_contrasts.csv", index=False)

    tail_values = build_upper_tail_values(wide)
    tail_values.to_csv(args.output / "upper_tail_bird_values.csv", index=False)

    tail_contrasts = build_upper_tail_contrasts(
        tail_values,
        bootstrap_reps=args.bootstrap_reps,
        max_exact_assignments=args.max_exact_assignments,
        monte_carlo_permutations=args.monte_carlo_permutations,
        seed=args.seed,
    )
    tail_contrasts.to_csv(args.output / "upper_tail_group_contrasts.csv", index=False)

    plot_group_profile(
        group_summary,
        metric=args.metric,
        out_path=args.output / "quantile_profile_by_group.png",
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )
    plot_contrast_profile(
        contrasts,
        metric=args.metric,
        out_path=args.output / "quantile_contrast_profile.png",
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )

    write_summary(
        args.output / "quantile_analysis_summary.txt",
        args=args,
        input_df=input_df,
        audit=audit,
        contrasts=contrasts,
        tail_contrasts=tail_contrasts,
    )

    print("[DONE] Quantile-profile analysis complete.")
    print(f"[DONE] Output directory: {args.output.resolve()}")
    print("[DONE] Key files:")
    print("       bird_quantile_values_long.csv")
    print("       quantile_group_contrasts.csv")
    print("       upper_tail_group_contrasts.csv")
    print("       quantile_profile_by_group.png")
    print("       quantile_contrast_profile.png")
    print("       quantile_analysis_summary.txt")


if __name__ == "__main__":
    main()
