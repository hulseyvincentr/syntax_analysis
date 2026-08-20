#!/usr/bin/env python3
"""Hierarchical analysis of pre/post phrase-duration variability.

This script analyzes the AFP lesion decoded-database JSON structure used by
``run_simple_variance_tests.py``. It implements the following prespecified
analysis hierarchy, with the bird as the independent experimental unit:

1. PRIMARY — all-syllable absolute variability
   For each eligible bird x syllable, calculate the log post/pre SD ratio and
   summarize each bird by the median across all eligible syllables.

2. ROBUSTNESS — all-syllable relative variability
   Calculate the corresponding log post/pre coefficient-of-variation (CV)
   ratio and summarize each bird by the median across eligible syllables.

3. SECONDARY — adaptive cross-screened burden
   Split recording dates within each bird into temporally balanced A/B halves.
   In A->B, screen syllables in A using a prespecified positive effect and an
   optional permissive day-permutation p-value threshold, then measure the
   selected syllables only in B. Reverse the roles for B->A. The signed burden
   for a direction is the sum of selected syllables' held-out effects divided
   by the number of syllables eligible in both halves. Unselected syllables
   therefore contribute zero, while selected syllables with negative held-out
   effects reduce the burden. Direction-level burdens are averaged to produce
   one cross-fitted value per bird.

4. DESCRIPTIVE DECOMPOSITION
   Report selected proportion, median held-out SD change, median held-out CV
   change, and median held-out mean-duration change.

For each bird-level endpoint, the script performs a three-group permutation
omnibus test and the three pairwise comparisons among medial+lateral,
lateral-only, and sham-saline birds. Pairwise two-sided p-values are adjusted
with Holm's method within each endpoint. The script also reports bootstrap
confidence intervals for group summaries and pairwise mean differences.

Expected directory structure
----------------------------
JSON_DIR/
    AFP_lesion_bird_metadata.json
    <lesion-group-folder>/<bird>/<bird>_decoded_database.json

Usage example (macOS)
---------------------
python run_phrase_variability_hierarchy.py \
    "$HOME/Desktop/AFP_lesion_jsons" \
    --output "../phrase_variability_hierarchy_results" \
    --seed 123

The default adaptive screen is based on increased SD, with a one-sided
recording-day permutation p <= 0.20 and a screening ratio > 1.0. To screen on
CV instead, add ``--screen-metric cv``.

Required package
----------------
numpy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


DIRECTIONS = (
    ("A_screen_B_test", "A", "B"),
    ("B_screen_A_test", "B", "A"),
)


# -----------------------------------------------------------------------------
# Argument parsing and validation
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hierarchical phrase-duration variability analysis."
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing AFP_lesion_bird_metadata.json and group folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phrase_variability_hierarchy_results"),
        help="Output directory (default: phrase_variability_hierarchy_results).",
    )
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--pre-start-day", type=int, default=-14)
    parser.add_argument("--pre-end-day", type=int, default=-1)
    parser.add_argument("--post-start-day", type=int, default=1)
    parser.add_argument("--post-end-day", type=int, default=14)

    parser.add_argument(
        "--min-full-phrases",
        type=int,
        default=20,
        help="Minimum pre and post phrases for full-data syllable metrics.",
    )
    parser.add_argument(
        "--min-full-days",
        type=int,
        default=2,
        help="Minimum pre and post recording dates for full-data metrics.",
    )
    parser.add_argument(
        "--min-screen-phrases",
        type=int,
        default=20,
        help="Minimum pre and post phrases in a screening half.",
    )
    parser.add_argument(
        "--min-screen-days",
        type=int,
        default=2,
        help="Minimum pre and post recording dates in a screening half.",
    )
    parser.add_argument(
        "--min-heldout-phrases",
        type=int,
        default=20,
        help="Minimum pre and post phrases in a held-out half.",
    )
    parser.add_argument(
        "--min-heldout-days",
        type=int,
        default=2,
        help="Minimum pre and post recording dates in a held-out half.",
    )

    parser.add_argument(
        "--screen-metric",
        choices=("sd", "cv"),
        default="sd",
        help=(
            "Metric used to nominate syllables in each screening half. "
            "Default: sd (recommended for absolute variability)."
        ),
    )
    parser.add_argument(
        "--screen-rule",
        choices=("effect_and_p", "effect_only"),
        default="effect_and_p",
        help="Adaptive screening rule (default: effect_and_p).",
    )
    parser.add_argument(
        "--screen-min-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum post/pre screening ratio, used with a strict > comparison. "
            "Default 1.0 requires a positive log change."
        ),
    )
    parser.add_argument(
        "--screen-p-threshold",
        type=float,
        default=0.20,
        help=(
            "One-sided day-permutation screening p threshold when "
            "--screen-rule effect_and_p is used (default: 0.20)."
        ),
    )

    parser.add_argument(
        "--medial-group", default="medial_and_lateral"
    )
    parser.add_argument("--lateral-group", default="lateral_only")
    parser.add_argument("--sham-group", default="sham_saline")

    parser.add_argument(
        "--max-exact-day-assignments",
        type=int,
        default=100_000,
        help="Maximum exact recording-day assignments per screen test.",
    )
    parser.add_argument(
        "--day-permutations",
        type=int,
        default=50_000,
        help="Monte Carlo day permutations when exact enumeration is too large.",
    )
    parser.add_argument(
        "--max-exact-group-assignments",
        type=int,
        default=200_000,
        help="Maximum exact group-label assignments.",
    )
    parser.add_argument(
        "--group-permutations",
        type=int,
        default=200_000,
        help="Monte Carlo group-label permutations for the omnibus test.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Bird-level bootstrap replicates for confidence intervals.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.json_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.json_dir}")
    if not args.pre_start_day <= args.pre_end_day < 0:
        raise ValueError("The pre-lesion window must end before day 0.")
    if not 0 < args.post_start_day <= args.post_end_day:
        raise ValueError("The post-lesion window must begin after day 0.")
    if args.screen_min_ratio <= 0:
        raise ValueError("--screen-min-ratio must be positive.")
    if not 0 < args.screen_p_threshold <= 1:
        raise ValueError("--screen-p-threshold must lie in (0, 1].")
    for name in (
        "min_full_phrases",
        "min_full_days",
        "min_screen_phrases",
        "min_screen_days",
        "min_heldout_phrases",
        "min_heldout_days",
        "day_permutations",
        "group_permutations",
        "bootstrap_replicates",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    text = "|".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


def label_sort_key(label: str) -> tuple[int, Any]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def mean_or_nan(values: Iterable[float]) -> float:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else math.nan


def median_or_nan(values: Iterable[float]) -> float:
    arr = finite_array(values)
    return float(np.median(arr)) if arr.size else math.nan


def exp_or_nan(value: float) -> float:
    return math.exp(value) if math.isfinite(value) else math.nan


def fmt(value: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_indices = np.where(np.isfinite(p))[0]
    if finite_indices.size == 0:
        return adjusted.tolist()

    order = finite_indices[np.argsort(p[finite_indices])]
    m = order.size
    running = 0.0
    for rank, index in enumerate(order, start=1):
        candidate = min(1.0, (m - rank + 1) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


# -----------------------------------------------------------------------------
# Date splitting and phrase-level metrics
# -----------------------------------------------------------------------------


def split_adjacent_dates(
    dates: Sequence[date], seed: int, bird: str, period: str
) -> dict[date, str]:
    """Create temporally balanced A/B halves by randomizing adjacent pairs."""
    ordered = sorted(set(dates))
    rng = stable_rng(seed, "date_split", bird, period)
    assignment: dict[date, str] = {}
    counts = {"A": 0, "B": 0}

    paired_end = len(ordered) - (len(ordered) % 2)
    for index in range(0, paired_end, 2):
        first, second = ordered[index], ordered[index + 1]
        if int(rng.integers(0, 2)) == 0:
            assignment[first], assignment[second] = "A", "B"
        else:
            assignment[first], assignment[second] = "B", "A"
        counts[assignment[first]] += 1
        counts[assignment[second]] += 1

    if len(ordered) % 2:
        leftover = ordered[-1]
        if counts["A"] < counts["B"]:
            fold = "A"
        elif counts["B"] < counts["A"]:
            fold = "B"
        else:
            fold = "A" if int(rng.integers(0, 2)) == 0 else "B"
        assignment[leftover] = fold

    return assignment


def flatten_days(days: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(day, dtype=float) for day in days if len(day)]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


@dataclass(frozen=True)
class Metrics:
    n_pre_phrases: int
    n_post_phrases: int
    n_pre_days: int
    n_post_days: int
    pre_mean: float
    post_mean: float
    pre_sd: float
    post_sd: float
    pre_variance: float
    post_variance: float
    pre_cv: float
    post_cv: float
    log_mean_ratio: float
    log_sd_ratio: float
    log_variance_ratio: float
    log_cv_ratio: float

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}n_pre_phrases": self.n_pre_phrases,
            f"{prefix}n_post_phrases": self.n_post_phrases,
            f"{prefix}n_pre_days": self.n_pre_days,
            f"{prefix}n_post_days": self.n_post_days,
            f"{prefix}pre_mean": self.pre_mean,
            f"{prefix}post_mean": self.post_mean,
            f"{prefix}pre_sd": self.pre_sd,
            f"{prefix}post_sd": self.post_sd,
            f"{prefix}pre_variance": self.pre_variance,
            f"{prefix}post_variance": self.post_variance,
            f"{prefix}pre_cv": self.pre_cv,
            f"{prefix}post_cv": self.post_cv,
            f"{prefix}log_mean_ratio": self.log_mean_ratio,
            f"{prefix}log_sd_ratio": self.log_sd_ratio,
            f"{prefix}log_variance_ratio": self.log_variance_ratio,
            f"{prefix}log_cv_ratio": self.log_cv_ratio,
        }


def calculate_metrics(
    pre_days: Sequence[Sequence[float]], post_days: Sequence[Sequence[float]]
) -> Metrics:
    pre = flatten_days(pre_days)
    post = flatten_days(post_days)

    pre_mean = float(np.mean(pre)) if pre.size else math.nan
    post_mean = float(np.mean(post)) if post.size else math.nan
    pre_sd = float(np.std(pre, ddof=1)) if pre.size >= 2 else math.nan
    post_sd = float(np.std(post, ddof=1)) if post.size >= 2 else math.nan
    pre_variance = pre_sd**2 if math.isfinite(pre_sd) else math.nan
    post_variance = post_sd**2 if math.isfinite(post_sd) else math.nan
    pre_cv = (
        pre_sd / pre_mean
        if math.isfinite(pre_sd) and math.isfinite(pre_mean) and pre_mean > 0
        else math.nan
    )
    post_cv = (
        post_sd / post_mean
        if math.isfinite(post_sd) and math.isfinite(post_mean) and post_mean > 0
        else math.nan
    )

    def log_ratio(post_value: float, pre_value: float) -> float:
        if (
            math.isfinite(post_value)
            and math.isfinite(pre_value)
            and post_value > 0
            and pre_value > 0
        ):
            return math.log(post_value / pre_value)
        return math.nan

    return Metrics(
        n_pre_phrases=int(pre.size),
        n_post_phrases=int(post.size),
        n_pre_days=len(pre_days),
        n_post_days=len(post_days),
        pre_mean=pre_mean,
        post_mean=post_mean,
        pre_sd=pre_sd,
        post_sd=post_sd,
        pre_variance=pre_variance,
        post_variance=post_variance,
        pre_cv=pre_cv,
        post_cv=post_cv,
        log_mean_ratio=log_ratio(post_mean, pre_mean),
        log_sd_ratio=log_ratio(post_sd, pre_sd),
        log_variance_ratio=log_ratio(post_variance, pre_variance),
        log_cv_ratio=log_ratio(post_cv, pre_cv),
    )


def metrics_eligible(metrics: Metrics, min_phrases: int, min_days: int) -> bool:
    return bool(
        metrics.n_pre_phrases >= min_phrases
        and metrics.n_post_phrases >= min_phrases
        and metrics.n_pre_days >= min_days
        and metrics.n_post_days >= min_days
        and math.isfinite(metrics.log_sd_ratio)
        and math.isfinite(metrics.log_cv_ratio)
        and math.isfinite(metrics.log_mean_ratio)
    )


def metric_value(metrics: Metrics, metric_name: str) -> float:
    if metric_name == "sd":
        return metrics.log_sd_ratio
    if metric_name == "cv":
        return metrics.log_cv_ratio
    raise ValueError(f"Unsupported screen metric: {metric_name}")


# -----------------------------------------------------------------------------
# Recording-day screening permutation test
# -----------------------------------------------------------------------------


def day_permutation_p(
    pre_days: Sequence[Sequence[float]],
    post_days: Sequence[Sequence[float]],
    metric_name: str,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, str, int, float]:
    """One-sided test for a larger post than pre SD or CV.

    Recording dates are the permutation units. All phrase observations from a
    date remain together. The observed assignment is included among exact
    assignments. Monte Carlo p-values use an add-one correction.
    """
    all_days = [list(day) for day in pre_days] + [list(day) for day in post_days]
    n_pre = len(pre_days)
    n_total = len(all_days)
    observed = metric_value(calculate_metrics(pre_days, post_days), metric_name)
    if not math.isfinite(observed) or n_pre == 0 or n_pre == n_total:
        return math.nan, "not_tested", 0, observed

    n_assignments = math.comb(n_total, n_pre)

    def statistic(chosen_pre: Sequence[int]) -> float:
        mask = np.zeros(n_total, dtype=bool)
        mask[list(chosen_pre)] = True
        perm_pre = [all_days[i] for i in range(n_total) if mask[i]]
        perm_post = [all_days[i] for i in range(n_total) if not mask[i]]
        return metric_value(calculate_metrics(perm_pre, perm_post), metric_name)

    if n_assignments <= max_exact_assignments:
        extreme = 0
        valid = 0
        for chosen in combinations(range(n_total), n_pre):
            value = statistic(chosen)
            if not math.isfinite(value):
                continue
            valid += 1
            extreme += int(value >= observed - 1e-15)
        return (
            extreme / valid if valid else math.nan,
            "exact",
            valid,
            observed,
        )

    extreme = 0
    valid = 0
    for _ in range(monte_carlo_permutations):
        chosen = rng.choice(n_total, size=n_pre, replace=False)
        value = statistic(chosen)
        if not math.isfinite(value):
            continue
        valid += 1
        extreme += int(value >= observed - 1e-15)
    p_value = (extreme + 1) / (valid + 1) if valid else math.nan
    return p_value, "monte_carlo", valid, observed


# -----------------------------------------------------------------------------
# Group-level permutation and bootstrap methods
# -----------------------------------------------------------------------------


def classical_anova_f(groups: Sequence[np.ndarray]) -> float:
    groups = [finite_array(group) for group in groups]
    if any(group.size == 0 for group in groups):
        return math.nan
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = sum(
        group.size * (float(np.mean(group)) - grand_mean) ** 2 for group in groups
    )
    ss_within = sum(
        float(np.sum((group - np.mean(group)) ** 2)) for group in groups
    )
    df_between = len(groups) - 1
    df_within = all_values.size - len(groups)
    if df_between <= 0 or df_within <= 0:
        return math.nan
    if ss_within <= 0:
        return math.inf if ss_between > 0 else 0.0
    return (ss_between / df_between) / (ss_within / df_within)


def welch_anova_f(groups: Sequence[np.ndarray]) -> tuple[float, str]:
    clean = [finite_array(group) for group in groups]
    if any(group.size < 2 for group in clean):
        return classical_anova_f(clean), "classical_F_fallback"
    variances = np.asarray([np.var(group, ddof=1) for group in clean], dtype=float)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        return classical_anova_f(clean), "classical_F_fallback"

    sizes = np.asarray([group.size for group in clean], dtype=float)
    means = np.asarray([np.mean(group) for group in clean], dtype=float)
    weights = sizes / variances
    total_weight = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / total_weight)
    k = len(clean)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
    correction_sum = float(
        np.sum(((1.0 - weights / total_weight) ** 2) / (sizes - 1.0))
    )
    denominator = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction_sum
    return numerator / denominator, "Welch_F"


def omnibus_permutation_test(
    values_by_group: dict[str, np.ndarray],
    group_order: Sequence[str],
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, str, int, str]:
    groups = [finite_array(values_by_group.get(group, [])) for group in group_order]
    if any(group.size == 0 for group in groups):
        return math.nan, math.nan, "not_tested", 0, "not_tested"

    observed, statistic_method = welch_anova_f(groups)
    pooled = np.concatenate(groups)
    sizes = [group.size for group in groups]
    n_total = pooled.size
    # Multinomial number of label assignments.
    n_assignments = math.factorial(n_total)
    for size in sizes:
        n_assignments //= math.factorial(size)

    def perm_stat(permuted: np.ndarray) -> float:
        split_groups: list[np.ndarray] = []
        start = 0
        for size in sizes:
            split_groups.append(permuted[start : start + size])
            start += size
        value, _ = welch_anova_f(split_groups)
        return value

    # Exact enumeration is practical only for very small problems. Generate
    # unique assignments recursively while keeping the final group implicit.
    if n_assignments <= max_exact_assignments and len(groups) == 3:
        extreme = 0
        valid = 0
        indices = np.arange(n_total)
        n1, n2, _ = sizes
        for chosen1 in combinations(indices, n1):
            mask1 = np.zeros(n_total, dtype=bool)
            mask1[list(chosen1)] = True
            remaining = indices[~mask1]
            for chosen2_local in combinations(range(remaining.size), n2):
                mask2 = np.zeros(remaining.size, dtype=bool)
                mask2[list(chosen2_local)] = True
                group1 = pooled[mask1]
                group2 = pooled[remaining[mask2]]
                group3 = pooled[remaining[~mask2]]
                value, _ = welch_anova_f([group1, group2, group3])
                if not math.isfinite(value) and not math.isinf(value):
                    continue
                valid += 1
                extreme += int(value >= observed - 1e-15)
        return observed, extreme / valid, "exact", valid, statistic_method

    extreme = 0
    valid = 0
    for _ in range(monte_carlo_permutations):
        value = perm_stat(rng.permutation(pooled))
        if not math.isfinite(value) and not math.isinf(value):
            continue
        valid += 1
        extreme += int(value >= observed - 1e-15)
    p_value = (extreme + 1) / (valid + 1) if valid else math.nan
    return observed, p_value, "monte_carlo", valid, statistic_method


def welch_t_statistic(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    x = finite_array(x)
    y = finite_array(y)
    if x.size < 2 or y.size < 2:
        return math.nan, "not_tested"
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    denominator = math.sqrt(vx / x.size + vy / y.size)
    if denominator > 0 and math.isfinite(denominator):
        return (float(np.mean(x) - np.mean(y))) / denominator, "Welch_t"
    # Fallback when both groups have zero within-group variance.
    difference = float(np.mean(x) - np.mean(y))
    return difference, "mean_difference_fallback"


def pairwise_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, str, int, str]:
    x_arr = finite_array(x)
    y_arr = finite_array(y)
    if x_arr.size < 2 or y_arr.size < 2:
        return math.nan, math.nan, "not_tested", 0, "not_tested"

    pooled = np.concatenate([x_arr, y_arr])
    n_x = x_arr.size
    observed, statistic_method = welch_t_statistic(x_arr, y_arr)
    n_assignments = math.comb(pooled.size, n_x)

    def is_extreme(value: float) -> bool:
        return abs(value) >= abs(observed) - 1e-15

    if n_assignments <= max_exact_assignments:
        extreme = 0
        valid = 0
        for chosen in combinations(range(pooled.size), n_x):
            mask = np.zeros(pooled.size, dtype=bool)
            mask[list(chosen)] = True
            value, _ = welch_t_statistic(pooled[mask], pooled[~mask])
            if not math.isfinite(value):
                continue
            valid += 1
            extreme += int(is_extreme(value))
        return observed, extreme / valid, "exact", valid, statistic_method

    extreme = 0
    valid = 0
    for _ in range(monte_carlo_permutations):
        permuted = rng.permutation(pooled)
        value, _ = welch_t_statistic(permuted[:n_x], permuted[n_x:])
        if not math.isfinite(value):
            continue
        valid += 1
        extreme += int(is_extreme(value))
    p_value = (extreme + 1) / (valid + 1) if valid else math.nan
    return observed, p_value, "monte_carlo", valid, statistic_method


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return math.nan, math.nan
    stats = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[index] = statistic(sample)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def bootstrap_difference_ci(
    x: Sequence[float],
    y: Sequence[float],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    x_arr = finite_array(x)
    y_arr = finite_array(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return math.nan, math.nan
    differences = np.empty(replicates, dtype=float)
    for index in range(replicates):
        x_sample = rng.choice(x_arr, size=x_arr.size, replace=True)
        y_sample = rng.choice(y_arr, size=y_arr.size, replace=True)
        differences[index] = float(np.mean(x_sample) - np.mean(y_sample))
    return (
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata_path = args.json_dir / "AFP_lesion_bird_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    database_paths = sorted(args.json_dir.glob("*/*/*decoded_database.json"))
    if not database_paths:
        raise FileNotFoundError(
            f"No files matched */*/*decoded_database.json under {args.json_dir}"
        )

    group_order = [args.medial_group, args.lateral_group, args.sham_group]
    pair_order = [
        (args.medial_group, args.lateral_group),
        (args.medial_group, args.sham_group),
        (args.lateral_group, args.sham_group),
    ]

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )

    input_audit_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    all_syllable_rows: list[dict[str, Any]] = []
    bird_all_rows: list[dict[str, Any]] = []
    bird_data: dict[str, dict[str, Any]] = {}

    # Read and organize every bird.
    for path in database_paths:
        bird = path.stem.removesuffix("_decoded_database")
        if bird not in metadata:
            raise KeyError(f"Bird {bird!r} is absent from {metadata_path}")
        group = str(metadata[bird]["lesion_group"])
        surgery = datetime.fromisoformat(
            metadata[bird]["lesion_surgery_date"]
        ).date()

        durations: dict[str, dict[str, dict[date, list[float]]]] = {
            "pre": defaultdict(lambda: defaultdict(list)),
            "post": defaultdict(lambda: defaultdict(list)),
        }

        results = json.loads(path.read_text()).get("results", [])
        for result in results:
            recording_date = datetime.fromisoformat(result["creation_date"]).date()
            relative_day = (recording_date - surgery).days
            if args.pre_start_day <= relative_day <= args.pre_end_day:
                period = "pre"
            elif args.post_start_day <= relative_day <= args.post_end_day:
                period = "post"
            else:
                continue

            for label, spans in result.get("syllable_onsets_offsets_ms", {}).items():
                durations[period][str(label)][recording_date].extend(
                    (float(end) - float(start)) / 1000.0 for start, end in spans
                )

        dates = {
            period: sorted(
                {
                    recording_date
                    for date_map in durations[period].values()
                    for recording_date in date_map
                }
            )
            for period in ("pre", "post")
        }
        assignments = {
            period: split_adjacent_dates(dates[period], args.seed, bird, period)
            for period in ("pre", "post")
        }
        for period in ("pre", "post"):
            for recording_date in dates[period]:
                date_rows.append(
                    {
                        "bird": bird,
                        "group": group,
                        "period": period,
                        "date": recording_date.isoformat(),
                        "relative_day": (recording_date - surgery).days,
                        "fold": assignments[period][recording_date],
                    }
                )

        common_labels = sorted(
            set(durations["pre"]) & set(durations["post"]),
            key=label_sort_key,
        )
        bird_data[bird] = {
            "group": group,
            "path": path,
            "durations": durations,
            "dates": dates,
            "assignments": assignments,
            "common_labels": common_labels,
        }

        eligible_full_rows: list[dict[str, Any]] = []
        for label in common_labels:
            pre_days = [
                values
                for _, values in sorted(durations["pre"][label].items())
            ]
            post_days = [
                values
                for _, values in sorted(durations["post"][label].items())
            ]
            metrics = calculate_metrics(pre_days, post_days)
            eligible = metrics_eligible(
                metrics, args.min_full_phrases, args.min_full_days
            )
            row = {
                "bird": bird,
                "group": group,
                "syllable": label,
                "eligible_full_analysis": eligible,
                **metrics.as_dict(),
                "mean_ratio": exp_or_nan(metrics.log_mean_ratio),
                "sd_ratio": exp_or_nan(metrics.log_sd_ratio),
                "variance_ratio": exp_or_nan(metrics.log_variance_ratio),
                "cv_ratio": exp_or_nan(metrics.log_cv_ratio),
            }
            all_syllable_rows.append(row)
            if eligible:
                eligible_full_rows.append(row)

        bird_all_rows.append(
            {
                "bird": bird,
                "group": group,
                "n_common_syllables": len(common_labels),
                "n_eligible_syllables": len(eligible_full_rows),
                "median_log_sd_ratio": median_or_nan(
                    row["log_sd_ratio"] for row in eligible_full_rows
                ),
                "median_log_variance_ratio": median_or_nan(
                    row["log_variance_ratio"] for row in eligible_full_rows
                ),
                "median_log_cv_ratio": median_or_nan(
                    row["log_cv_ratio"] for row in eligible_full_rows
                ),
                "median_log_mean_ratio": median_or_nan(
                    row["log_mean_ratio"] for row in eligible_full_rows
                ),
            }
        )

        input_audit_rows.append(
            {
                "bird": bird,
                "group": group,
                "database_path": str(path),
                "n_pre_dates": len(dates["pre"]),
                "n_post_dates": len(dates["post"]),
                "n_common_syllables": len(common_labels),
                "n_full_analysis_eligible_syllables": len(eligible_full_rows),
            }
        )

    # Adaptive reciprocal cross-screening.
    cross_syllable_rows: list[dict[str, Any]] = []
    direction_bird_rows: list[dict[str, Any]] = []
    screen_min_log_effect = math.log(args.screen_min_ratio)

    for direction, screen_fold, heldout_fold in DIRECTIONS:
        for bird in sorted(bird_data):
            info = bird_data[bird]
            group = info["group"]
            durations = info["durations"]
            assignments = info["assignments"]
            candidate_rows: list[dict[str, Any]] = []

            for label in info["common_labels"]:
                fold_metrics: dict[str, Metrics] = {}
                for fold in ("A", "B"):
                    pre_days = [
                        values
                        for recording_date, values in sorted(
                            durations["pre"][label].items()
                        )
                        if assignments["pre"].get(recording_date) == fold
                    ]
                    post_days = [
                        values
                        for recording_date, values in sorted(
                            durations["post"][label].items()
                        )
                        if assignments["post"].get(recording_date) == fold
                    ]
                    fold_metrics[fold] = calculate_metrics(pre_days, post_days)

                screen_metrics = fold_metrics[screen_fold]
                heldout_metrics = fold_metrics[heldout_fold]
                eligible_screen = metrics_eligible(
                    screen_metrics,
                    args.min_screen_phrases,
                    args.min_screen_days,
                )
                eligible_heldout = metrics_eligible(
                    heldout_metrics,
                    args.min_heldout_phrases,
                    args.min_heldout_days,
                )
                eligible_both = eligible_screen and eligible_heldout

                screen_p = math.nan
                p_method = "not_tested"
                n_day_assignments = 0
                if eligible_both:
                    screen_pre_days = [
                        values
                        for recording_date, values in sorted(
                            durations["pre"][label].items()
                        )
                        if assignments["pre"].get(recording_date) == screen_fold
                    ]
                    screen_post_days = [
                        values
                        for recording_date, values in sorted(
                            durations["post"][label].items()
                        )
                        if assignments["post"].get(recording_date) == screen_fold
                    ]
                    screen_p, p_method, n_day_assignments, _ = day_permutation_p(
                        screen_pre_days,
                        screen_post_days,
                        args.screen_metric,
                        args.max_exact_day_assignments,
                        args.day_permutations,
                        stable_rng(
                            args.seed,
                            "day_permutation",
                            bird,
                            direction,
                            label,
                            args.screen_metric,
                        ),
                    )

                screening_effect = (
                    metric_value(screen_metrics, args.screen_metric)
                    if eligible_both
                    else math.nan
                )
                effect_pass = bool(
                    math.isfinite(screening_effect)
                    and screening_effect > screen_min_log_effect
                )
                p_pass = bool(
                    math.isfinite(screen_p) and screen_p <= args.screen_p_threshold
                )
                selected = bool(
                    eligible_both
                    and effect_pass
                    and (
                        p_pass if args.screen_rule == "effect_and_p" else True
                    )
                )

                row = {
                    "bird": bird,
                    "group": group,
                    "direction": direction,
                    "screen_fold": screen_fold,
                    "heldout_fold": heldout_fold,
                    "syllable": label,
                    "eligible_screen": eligible_screen,
                    "eligible_heldout": eligible_heldout,
                    "eligible_both": eligible_both,
                    "screen_metric": args.screen_metric,
                    "screen_rule": args.screen_rule,
                    "screen_min_ratio": args.screen_min_ratio,
                    "screen_p_threshold": args.screen_p_threshold,
                    "screen_metric_log_effect": screening_effect,
                    "screen_metric_ratio": exp_or_nan(screening_effect),
                    "screen_day_permutation_p": screen_p,
                    "screen_p_method": p_method,
                    "n_day_assignments_or_permutations": n_day_assignments,
                    "screen_effect_pass": effect_pass,
                    "screen_p_pass": p_pass,
                    "selected": selected,
                    **screen_metrics.as_dict("screen_"),
                    **heldout_metrics.as_dict("heldout_"),
                    "heldout_mean_ratio": exp_or_nan(
                        heldout_metrics.log_mean_ratio
                    ),
                    "heldout_sd_ratio": exp_or_nan(heldout_metrics.log_sd_ratio),
                    "heldout_variance_ratio": exp_or_nan(
                        heldout_metrics.log_variance_ratio
                    ),
                    "heldout_cv_ratio": exp_or_nan(heldout_metrics.log_cv_ratio),
                }
                cross_syllable_rows.append(row)
                if eligible_both:
                    candidate_rows.append(row)

            selected_rows = [row for row in candidate_rows if row["selected"]]
            n_eligible = len(candidate_rows)
            n_selected = len(selected_rows)

            def burden(key: str) -> float:
                if n_eligible == 0:
                    return math.nan
                return float(
                    sum(float(row[key]) for row in selected_rows) / n_eligible
                )

            direction_bird_rows.append(
                {
                    "bird": bird,
                    "group": group,
                    "direction": direction,
                    "screen_fold": screen_fold,
                    "heldout_fold": heldout_fold,
                    "n_eligible_syllables": n_eligible,
                    "n_selected_syllables": n_selected,
                    "selected_proportion": (
                        n_selected / n_eligible if n_eligible else math.nan
                    ),
                    "selected_labels": ",".join(
                        str(row["syllable"]) for row in selected_rows
                    ),
                    "burden_log_sd_ratio": burden("heldout_log_sd_ratio"),
                    "burden_log_variance_ratio": burden(
                        "heldout_log_variance_ratio"
                    ),
                    "burden_log_cv_ratio": burden("heldout_log_cv_ratio"),
                    "burden_log_mean_ratio": burden("heldout_log_mean_ratio"),
                    "median_heldout_log_sd_ratio_selected": median_or_nan(
                        row["heldout_log_sd_ratio"] for row in selected_rows
                    ),
                    "median_heldout_log_variance_ratio_selected": median_or_nan(
                        row["heldout_log_variance_ratio"] for row in selected_rows
                    ),
                    "median_heldout_log_cv_ratio_selected": median_or_nan(
                        row["heldout_log_cv_ratio"] for row in selected_rows
                    ),
                    "median_heldout_log_mean_ratio_selected": median_or_nan(
                        row["heldout_log_mean_ratio"] for row in selected_rows
                    ),
                }
            )

    # Average reciprocal directions to one cross-fitted row per bird.
    crossfitted_rows: list[dict[str, Any]] = []
    rows_by_bird: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direction_bird_rows:
        rows_by_bird[row["bird"]].append(row)

    crossfit_metric_keys = [
        "selected_proportion",
        "burden_log_sd_ratio",
        "burden_log_variance_ratio",
        "burden_log_cv_ratio",
        "burden_log_mean_ratio",
        "median_heldout_log_sd_ratio_selected",
        "median_heldout_log_variance_ratio_selected",
        "median_heldout_log_cv_ratio_selected",
        "median_heldout_log_mean_ratio_selected",
    ]

    for bird in sorted(bird_data):
        rows = rows_by_bird.get(bird, [])
        by_direction = {row["direction"]: row for row in rows}
        output: dict[str, Any] = {
            "bird": bird,
            "group": bird_data[bird]["group"],
            "n_directions_available": len(rows),
            "A_screen_B_n_eligible": by_direction.get(
                "A_screen_B_test", {}
            ).get("n_eligible_syllables", math.nan),
            "A_screen_B_n_selected": by_direction.get(
                "A_screen_B_test", {}
            ).get("n_selected_syllables", math.nan),
            "B_screen_A_n_eligible": by_direction.get(
                "B_screen_A_test", {}
            ).get("n_eligible_syllables", math.nan),
            "B_screen_A_n_selected": by_direction.get(
                "B_screen_A_test", {}
            ).get("n_selected_syllables", math.nan),
        }
        for key in crossfit_metric_keys:
            output[f"A_screen_B_{key}"] = by_direction.get(
                "A_screen_B_test", {}
            ).get(key, math.nan)
            output[f"B_screen_A_{key}"] = by_direction.get(
                "B_screen_A_test", {}
            ).get(key, math.nan)
            output[f"crossfitted_{key}"] = mean_or_nan(
                row[key] for row in rows
            )
        crossfitted_rows.append(output)

    # Merge all bird-level endpoints into one analysis table.
    crossfitted_by_bird = {row["bird"]: row for row in crossfitted_rows}
    bird_endpoint_rows: list[dict[str, Any]] = []
    for row in bird_all_rows:
        cross = crossfitted_by_bird.get(row["bird"], {})
        bird_endpoint_rows.append({**row, **cross})

    # Prespecified endpoint hierarchy.
    burden_primary_key = (
        "crossfitted_burden_log_sd_ratio"
        if args.screen_metric == "sd"
        else "crossfitted_burden_log_cv_ratio"
    )
    endpoints = [
        (
            "all_syllable_absolute_variability",
            "primary",
            "median_log_sd_ratio",
        ),
        (
            "all_syllable_relative_variability",
            "robustness",
            "median_log_cv_ratio",
        ),
        (
            f"cross_screened_{args.screen_metric}_burden",
            "secondary",
            burden_primary_key,
        ),
        (
            "cross_screened_selected_proportion",
            "descriptive_component",
            "crossfitted_selected_proportion",
        ),
        (
            "cross_screened_heldout_sd_magnitude",
            "descriptive_component",
            "crossfitted_median_heldout_log_sd_ratio_selected",
        ),
        (
            "cross_screened_heldout_cv_magnitude",
            "descriptive_component",
            "crossfitted_median_heldout_log_cv_ratio_selected",
        ),
        (
            "cross_screened_heldout_mean_change",
            "descriptive_component",
            "crossfitted_median_heldout_log_mean_ratio_selected",
        ),
    ]

    group_descriptive_rows: list[dict[str, Any]] = []
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for endpoint_name, hierarchy_role, key in endpoints:
        values_by_group = {
            group: finite_array(
                finite_or_nan(row.get(key))
                for row in bird_endpoint_rows
                if row["group"] == group
            )
            for group in group_order
        }

        for group in group_order:
            values = values_by_group[group]
            mean_ci = bootstrap_ci(
                values,
                lambda array: float(np.mean(array)),
                args.bootstrap_replicates,
                stable_rng(args.seed, "bootstrap_group_mean", endpoint_name, group),
            )
            median_ci = bootstrap_ci(
                values,
                lambda array: float(np.median(array)),
                args.bootstrap_replicates,
                stable_rng(args.seed, "bootstrap_group_median", endpoint_name, group),
            )
            group_descriptive_rows.append(
                {
                    "endpoint": endpoint_name,
                    "hierarchy_role": hierarchy_role,
                    "column": key,
                    "group": group,
                    "n_birds": values.size,
                    "mean": float(np.mean(values)) if values.size else math.nan,
                    "mean_ci_low": mean_ci[0],
                    "mean_ci_high": mean_ci[1],
                    "median": float(np.median(values)) if values.size else math.nan,
                    "median_ci_low": median_ci[0],
                    "median_ci_high": median_ci[1],
                    "iqr_low": (
                        float(np.quantile(values, 0.25))
                        if values.size
                        else math.nan
                    ),
                    "iqr_high": (
                        float(np.quantile(values, 0.75))
                        if values.size
                        else math.nan
                    ),
                    "exp_mean": exp_or_nan(
                        float(np.mean(values)) if values.size else math.nan
                    ),
                    "exp_median": exp_or_nan(
                        float(np.median(values)) if values.size else math.nan
                    ),
                }
            )

        statistic, p_value, permutation_method, n_perm, statistic_method = (
            omnibus_permutation_test(
                values_by_group,
                group_order,
                args.max_exact_group_assignments,
                args.group_permutations,
                stable_rng(args.seed, "omnibus", endpoint_name),
            )
        )
        omnibus_rows.append(
            {
                "endpoint": endpoint_name,
                "hierarchy_role": hierarchy_role,
                "column": key,
                "groups": "|".join(group_order),
                "statistic": statistic,
                "statistic_method": statistic_method,
                "permutation_p": p_value,
                "permutation_method": permutation_method,
                "n_assignments_or_permutations": n_perm,
            }
        )

        endpoint_pair_rows: list[dict[str, Any]] = []
        for group1, group2 in pair_order:
            x = values_by_group[group1]
            y = values_by_group[group2]
            test_stat, p_raw, method, n_test, test_stat_method = (
                pairwise_permutation_test(
                    x,
                    y,
                    args.max_exact_group_assignments,
                    args.group_permutations,
                    stable_rng(
                        args.seed,
                        "pairwise",
                        endpoint_name,
                        group1,
                        group2,
                    ),
                )
            )
            mean_difference = (
                float(np.mean(x) - np.mean(y))
                if x.size and y.size
                else math.nan
            )
            median_difference = (
                float(np.median(x) - np.median(y))
                if x.size and y.size
                else math.nan
            )
            difference_ci = bootstrap_difference_ci(
                x,
                y,
                args.bootstrap_replicates,
                stable_rng(
                    args.seed,
                    "bootstrap_difference",
                    endpoint_name,
                    group1,
                    group2,
                ),
            )
            endpoint_pair_rows.append(
                {
                    "endpoint": endpoint_name,
                    "hierarchy_role": hierarchy_role,
                    "column": key,
                    "group1": group1,
                    "group2": group2,
                    "n_group1": x.size,
                    "n_group2": y.size,
                    "mean_group1": (
                        float(np.mean(x)) if x.size else math.nan
                    ),
                    "mean_group2": (
                        float(np.mean(y)) if y.size else math.nan
                    ),
                    "mean_difference_group1_minus_group2": mean_difference,
                    "mean_difference_ci_low": difference_ci[0],
                    "mean_difference_ci_high": difference_ci[1],
                    "median_difference_group1_minus_group2": median_difference,
                    "exp_mean_difference": exp_or_nan(mean_difference),
                    "test_statistic": test_stat,
                    "test_statistic_method": test_stat_method,
                    "two_sided_permutation_p_raw": p_raw,
                    "permutation_method": method,
                    "n_assignments_or_permutations": n_test,
                }
            )

        adjusted = holm_adjust(
            [row["two_sided_permutation_p_raw"] for row in endpoint_pair_rows]
        )
        for row, adjusted_p in zip(endpoint_pair_rows, adjusted):
            row["two_sided_permutation_p_holm"] = adjusted_p
            pairwise_rows.append(row)

    # Output tables.
    write_tsv(
        args.output / "input_audit.tsv",
        input_audit_rows,
        [
            "bird",
            "group",
            "database_path",
            "n_pre_dates",
            "n_post_dates",
            "n_common_syllables",
            "n_full_analysis_eligible_syllables",
        ],
    )
    write_tsv(
        args.output / "date_split_assignments.tsv",
        date_rows,
        ["bird", "group", "period", "date", "relative_day", "fold"],
    )
    write_tsv(
        args.output / "all_syllable_metrics.tsv",
        all_syllable_rows,
        list(all_syllable_rows[0]) if all_syllable_rows else [],
    )
    write_tsv(
        args.output / "bird_level_all_syllable.tsv",
        bird_all_rows,
        list(bird_all_rows[0]) if bird_all_rows else [],
    )
    write_tsv(
        args.output / "cross_screen_syllable_details.tsv",
        cross_syllable_rows,
        list(cross_syllable_rows[0]) if cross_syllable_rows else [],
    )
    write_tsv(
        args.output / "cross_screen_direction_bird.tsv",
        direction_bird_rows,
        list(direction_bird_rows[0]) if direction_bird_rows else [],
    )
    write_tsv(
        args.output / "cross_screen_crossfitted_bird.tsv",
        crossfitted_rows,
        list(crossfitted_rows[0]) if crossfitted_rows else [],
    )
    write_tsv(
        args.output / "bird_level_endpoints.tsv",
        bird_endpoint_rows,
        list(bird_endpoint_rows[0]) if bird_endpoint_rows else [],
    )
    write_tsv(
        args.output / "group_descriptives.tsv",
        group_descriptive_rows,
        list(group_descriptive_rows[0]) if group_descriptive_rows else [],
    )
    write_tsv(
        args.output / "group_omnibus_tests.tsv",
        omnibus_rows,
        list(omnibus_rows[0]) if omnibus_rows else [],
    )
    write_tsv(
        args.output / "group_pairwise_tests.tsv",
        pairwise_rows,
        list(pairwise_rows[0]) if pairwise_rows else [],
    )

    # Human-readable summary.
    summary_lines = [
        "PHRASE-DURATION VARIABILITY ANALYSIS HIERARCHY",
        "=" * 58,
        "",
        f"Bird databases found: {len(database_paths)}",
        "Group counts: "
        + ", ".join(
            f"{group}={sum(row['group'] == group for row in bird_endpoint_rows)}"
            for group in group_order
        ),
        f"Screening metric: {args.screen_metric}",
        f"Screening rule: {args.screen_rule}",
        f"Screening minimum ratio: > {args.screen_min_ratio}",
        (
            f"Screening p threshold: <= {args.screen_p_threshold}"
            if args.screen_rule == "effect_and_p"
            else "Screening p threshold: not used"
        ),
        "",
    ]

    for endpoint_name, hierarchy_role, key in endpoints:
        omnibus = next(row for row in omnibus_rows if row["endpoint"] == endpoint_name)
        summary_lines.extend(
            [
                f"{hierarchy_role.upper()}: {endpoint_name}",
                f"  Bird-level column: {key}",
                (
                    f"  Omnibus permutation p: {fmt(float(omnibus['permutation_p']))} "
                    f"({omnibus['statistic_method']}; {omnibus['permutation_method']})"
                ),
            ]
        )
        for group in group_order:
            desc = next(
                row
                for row in group_descriptive_rows
                if row["endpoint"] == endpoint_name and row["group"] == group
            )
            summary_lines.append(
                f"  {group}: n={desc['n_birds']}, mean={fmt(float(desc['mean']))}, "
                f"median={fmt(float(desc['median']))}"
            )
        for row in pairwise_rows:
            if row["endpoint"] != endpoint_name:
                continue
            summary_lines.append(
                f"  {row['group1']} vs {row['group2']}: "
                f"mean difference={fmt(float(row['mean_difference_group1_minus_group2']))}, "
                f"raw p={fmt(float(row['two_sided_permutation_p_raw']))}, "
                f"Holm p={fmt(float(row['two_sided_permutation_p_holm']))}"
            )
        summary_lines.append("")

    summary_lines.extend(
        [
            "INTERPRETATION NOTES",
            "- The primary inferential endpoint is the all-syllable median log SD ratio.",
            "- The all-syllable median log CV ratio is a robustness endpoint.",
            "- The cross-screened burden is a prespecified secondary endpoint.",
            "- Selected proportion and held-out magnitude are decomposition endpoints.",
            "- SD, variance, CV, and mean ratios are on a natural-log scale; exponentiate",
            "  a log ratio to obtain the corresponding post/pre multiplicative ratio.",
            "- Pairwise Holm correction is applied separately within each endpoint.",
            "- No correction is applied across hierarchy levels because their roles are",
            "  prespecified; only the primary endpoint should support the main claim.",
        ]
    )
    summary_text = "\n".join(summary_lines) + "\n"
    (args.output / "summary.txt").write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
