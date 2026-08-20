#!/usr/bin/env python3
"""Cross-screened localization of rare, extreme phrase prolongations.

This script is designed for the AFP lesion decoded-database JSON structure used
by the user's existing analysis scripts:

JSON_DIR/
    AFP_lesion_bird_metadata.json
    <lesion-group-folder>/<bird>/<bird>_decoded_database.json

Each decoded database must contain a top-level ``results`` list. Each result
must contain:

* ``creation_date``: an ISO-formatted recording timestamp/date
* ``syllable_onsets_offsets_ms``: {syllable_label: [[start_ms, end_ms], ...]}

The metadata JSON must map each bird ID to at least:

* ``lesion_group``
* ``lesion_surgery_date``

Analysis hierarchy
------------------
1. Define a syllable-specific extreme-duration threshold from EARLY pre-lesion
   phrase durations (default: 99th percentile).
2. Compare LATE pre-lesion with post-lesion tail behavior using:
      * extreme-event rate
      * conditional excess severity
      * tail burden = mean(max(0, duration - threshold))
3. Localize affected syllables using reciprocal two-way cross-screening:
      * split late-pre and post recording dates separately into balanced A/B
        folds, keeping all observations from a date together;
      * screen syllables in A and evaluate them only in B;
      * reverse A and B;
      * confirm candidates with held-out recording-day permutation tests and
        Holm correction within each bird/direction.
4. Produce bird-level localization endpoints and exact bird-label permutation
   comparisons among medial+lateral, lateral-only, and sham groups.
5. Produce supplemental all-syllable and whole-bird tail-burden summaries.
6. Produce sensitivity analyses for threshold choice, split seed, matched post
   windows, removal of the longest post occurrence, and requiring post extremes
   on at least two recording dates.

Primary outputs are CSV files so they can be inspected directly or uploaded for
validation. The only required third-party package is NumPy.

Example
-------
python run_extreme_phrase_localization.py \
    "$HOME/Desktop/AFP_lesion_jsons" \
    --output "$HOME/Desktop/extreme_phrase_localization_results" \
    --early-pre-start-day -28 --early-pre-end-day -15 \
    --late-pre-start-day -14 --late-pre-end-day -1 \
    --post-start-day 1 \
    --primary-quantile 0.99 \
    --seed 123

Important
---------
The default early-pre window (-28 to -15) is only a starting value. Set the
three period windows to match the periods used in the manuscript before
interpreting the results.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DIRECTIONS = (
    ("A_screen_B_test", "A", "B"),
    ("B_screen_A_test", "B", "A"),
)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Localize rare extreme phrase prolongations with reciprocal "
            "cross-screening and recording-day permutation tests."
        ),
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing AFP_lesion_bird_metadata.json and group folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extreme_phrase_localization_results"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=123)

    # Period windows, in days relative to surgery.
    parser.add_argument("--early-pre-start-day", type=int, default=-28)
    parser.add_argument("--early-pre-end-day", type=int, default=-15)
    parser.add_argument("--late-pre-start-day", type=int, default=-14)
    parser.add_argument("--late-pre-end-day", type=int, default=-1)
    parser.add_argument("--post-start-day", type=int, default=1)
    parser.add_argument(
        "--post-end-day",
        type=int,
        default=None,
        help="Last included post-lesion day; omit to use all available post days.",
    )

    # Extreme-event definition.
    parser.add_argument("--primary-quantile", type=float, default=0.99)
    parser.add_argument(
        "--sensitivity-quantiles",
        type=float,
        nargs="+",
        default=[0.95, 0.975, 0.99, 0.995],
    )

    # Eligibility.
    parser.add_argument("--min-early-pre-phrases", type=int, default=100)
    parser.add_argument("--min-late-pre-phrases", type=int, default=50)
    parser.add_argument("--min-post-phrases", type=int, default=50)
    parser.add_argument("--min-late-pre-days", type=int, default=4)
    parser.add_argument("--min-post-days", type=int, default=4)
    parser.add_argument("--min-fold-phrases", type=int, default=20)
    parser.add_argument("--min-fold-days", type=int, default=2)

    # Screening and held-out confirmation.
    parser.add_argument(
        "--screen-min-delta-burden-seconds",
        type=float,
        default=0.0,
        help="Candidates must have screening-fold post-minus-late-pre burden above this value.",
    )
    parser.add_argument(
        "--screen-min-post-extreme-events",
        type=int,
        default=1,
        help="Minimum screening-fold post extreme events required for nomination.",
    )
    parser.add_argument(
        "--screen-min-post-extreme-days",
        type=int,
        default=1,
        help="Minimum screening-fold post dates containing an extreme event.",
    )
    parser.add_argument(
        "--confirmation-family-alpha",
        type=float,
        default=0.05,
        help=(
            "Familywise alpha across both cross-screening directions. Each "
            "direction uses half this alpha after Holm correction."
        ),
    )

    # Permutations and bootstrap.
    parser.add_argument("--max-exact-day-assignments", type=int, default=100_000)
    parser.add_argument("--day-permutations", type=int, default=50_000)
    parser.add_argument("--max-exact-group-assignments", type=int, default=200_000)
    parser.add_argument("--group-permutations", type=int, default=200_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--group-statistic",
        choices=("mean", "median"),
        default="mean",
        help="Statistic used for exact group-label permutation contrasts.",
    )

    # Group labels in AFP_lesion_bird_metadata.json.
    parser.add_argument("--medial-group", default="medial_and_lateral")
    parser.add_argument("--lateral-group", default="lateral_only")
    parser.add_argument("--sham-group", default="sham_saline")

    # Sensitivity controls.
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the threshold/split/window/outlier sensitivity analyses.",
    )
    parser.add_argument(
        "--sensitivity-random-splits",
        type=int,
        default=5,
        help="Number of additional reproducible date-split seeds.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.json_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.json_dir}")
    if not (
        args.early_pre_start_day
        <= args.early_pre_end_day
        < args.late_pre_start_day
        <= args.late_pre_end_day
        < 0
    ):
        raise ValueError(
            "Require early-pre < late-pre < surgery with non-overlapping windows."
        )
    if args.post_start_day <= 0:
        raise ValueError("--post-start-day must be > 0.")
    if args.post_end_day is not None and args.post_end_day < args.post_start_day:
        raise ValueError("--post-end-day must be >= --post-start-day.")
    quantiles = [args.primary_quantile, *args.sensitivity_quantiles]
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("All quantiles must lie strictly between 0 and 1.")
    if not 0 < args.confirmation_family_alpha <= 1:
        raise ValueError("--confirmation-family-alpha must lie in (0, 1].")
    for name in (
        "min_early_pre_phrases",
        "min_late_pre_phrases",
        "min_post_phrases",
        "min_late_pre_days",
        "min_post_days",
        "min_fold_phrases",
        "min_fold_days",
        "screen_min_post_extreme_events",
        "screen_min_post_extreme_days",
        "max_exact_day_assignments",
        "day_permutations",
        "max_exact_group_assignments",
        "group_permutations",
        "bootstrap_replicates",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")
    if args.sensitivity_random_splits < 0:
        raise ValueError("--sensitivity-random-splits must be >= 0.")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    text = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


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


def max_or_nan(values: Iterable[float]) -> float:
    arr = finite_array(values)
    return float(np.max(arr)) if arr.size else math.nan


def label_sort_key(label: str) -> tuple[int, Any]:
    try:
        return (0, int(label))
    except (TypeError, ValueError):
        return (1, str(label))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_indices = np.where(np.isfinite(p))[0]
    if finite_indices.size == 0:
        return adjusted.tolist()
    ordered = finite_indices[np.argsort(p[finite_indices])]
    m = len(ordered)
    running = 0.0
    for rank, index in enumerate(ordered, start=1):
        candidate = min(1.0, (m - rank + 1) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> tuple[float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    n = len(x_arr)
    if n < 3 or np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return math.nan, n
    xr = rankdata_average(x_arr)
    yr = rankdata_average(y_arr)
    return float(np.corrcoef(xr, yr)[0, 1]), n


def fmt(value: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


# -----------------------------------------------------------------------------
# Data loading and period/date handling
# -----------------------------------------------------------------------------


def parse_iso_date(value: str) -> date:
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def classify_period(relative_day: int, args: argparse.Namespace) -> str | None:
    if args.early_pre_start_day <= relative_day <= args.early_pre_end_day:
        return "early_pre"
    if args.late_pre_start_day <= relative_day <= args.late_pre_end_day:
        return "late_pre"
    if relative_day >= args.post_start_day and (
        args.post_end_day is None or relative_day <= args.post_end_day
    ):
        return "post"
    return None


def split_adjacent_dates(
    dates: Sequence[date], seed: int, bird: str, period: str
) -> dict[date, str]:
    """Temporally balanced A/B split with adjacent dates separated across folds."""
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


def subset_day_map(
    day_map: Mapping[date, Sequence[float]],
    allowed_dates: set[date] | None = None,
    fold_assignment: Mapping[date, str] | None = None,
    fold: str | None = None,
) -> dict[date, list[float]]:
    result: dict[date, list[float]] = {}
    for recording_date, values in sorted(day_map.items()):
        if allowed_dates is not None and recording_date not in allowed_dates:
            continue
        if fold_assignment is not None and fold is not None:
            if fold_assignment.get(recording_date) != fold:
                continue
        clean = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0]
        if clean:
            result[recording_date] = clean
    return result


def remove_single_longest(day_map: Mapping[date, Sequence[float]]) -> dict[date, list[float]]:
    copied = {d: list(map(float, values)) for d, values in day_map.items()}
    best_date: date | None = None
    best_index: int | None = None
    best_value = -math.inf
    for recording_date, values in copied.items():
        for index, value in enumerate(values):
            if math.isfinite(value) and value > best_value:
                best_date, best_index, best_value = recording_date, index, value
    if best_date is not None and best_index is not None:
        copied[best_date].pop(best_index)
        if not copied[best_date]:
            del copied[best_date]
    return copied


def flatten_day_map(day_map: Mapping[date, Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(values, dtype=float) for values in day_map.values() if values]
    if not arrays:
        return np.asarray([], dtype=float)
    result = np.concatenate(arrays)
    return result[np.isfinite(result) & (result > 0)]


# -----------------------------------------------------------------------------
# Tail metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TailMetrics:
    n_phrases: int
    n_days: int
    n_extreme: int
    n_extreme_days: int
    event_rate: float
    burden_seconds_per_occurrence: float
    conditional_excess_median_seconds: float
    conditional_excess_mean_seconds: float
    max_duration_seconds: float
    max_excess_seconds: float
    mean_duration_seconds: float
    sd_duration_seconds: float
    cv_duration: float

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}n_phrases": self.n_phrases,
            f"{prefix}n_days": self.n_days,
            f"{prefix}n_extreme": self.n_extreme,
            f"{prefix}n_extreme_days": self.n_extreme_days,
            f"{prefix}event_rate": self.event_rate,
            f"{prefix}burden_seconds_per_occurrence": self.burden_seconds_per_occurrence,
            f"{prefix}burden_seconds_per_100": 100.0 * self.burden_seconds_per_occurrence,
            f"{prefix}conditional_excess_median_seconds": self.conditional_excess_median_seconds,
            f"{prefix}conditional_excess_mean_seconds": self.conditional_excess_mean_seconds,
            f"{prefix}max_duration_seconds": self.max_duration_seconds,
            f"{prefix}max_excess_seconds": self.max_excess_seconds,
            f"{prefix}mean_duration_seconds": self.mean_duration_seconds,
            f"{prefix}sd_duration_seconds": self.sd_duration_seconds,
            f"{prefix}cv_duration": self.cv_duration,
        }


def calculate_tail_metrics(
    day_map: Mapping[date, Sequence[float]], threshold_seconds: float
) -> TailMetrics:
    values = flatten_day_map(day_map)
    if values.size == 0 or not math.isfinite(threshold_seconds):
        return TailMetrics(
            n_phrases=int(values.size),
            n_days=len(day_map),
            n_extreme=0,
            n_extreme_days=0,
            event_rate=math.nan,
            burden_seconds_per_occurrence=math.nan,
            conditional_excess_median_seconds=math.nan,
            conditional_excess_mean_seconds=math.nan,
            max_duration_seconds=math.nan,
            max_excess_seconds=math.nan,
            mean_duration_seconds=math.nan,
            sd_duration_seconds=math.nan,
            cv_duration=math.nan,
        )
    excess = np.maximum(0.0, values - threshold_seconds)
    extreme = excess > 0
    extreme_excess = excess[extreme]
    n_extreme_days = 0
    for day_values in day_map.values():
        arr = np.asarray(day_values, dtype=float)
        if np.any(arr > threshold_seconds):
            n_extreme_days += 1
    mean_value = float(np.mean(values))
    sd_value = float(np.std(values, ddof=1)) if values.size >= 2 else math.nan
    cv_value = sd_value / mean_value if math.isfinite(sd_value) and mean_value > 0 else math.nan
    return TailMetrics(
        n_phrases=int(values.size),
        n_days=len(day_map),
        n_extreme=int(np.sum(extreme)),
        n_extreme_days=n_extreme_days,
        event_rate=float(np.mean(extreme)),
        burden_seconds_per_occurrence=float(np.mean(excess)),
        conditional_excess_median_seconds=(
            float(np.median(extreme_excess)) if extreme_excess.size else 0.0
        ),
        conditional_excess_mean_seconds=(
            float(np.mean(extreme_excess)) if extreme_excess.size else 0.0
        ),
        max_duration_seconds=float(np.max(values)),
        max_excess_seconds=float(np.max(excess)),
        mean_duration_seconds=mean_value,
        sd_duration_seconds=sd_value,
        cv_duration=cv_value,
    )


def tail_deltas(pre: TailMetrics, post: TailMetrics) -> dict[str, float]:
    return {
        "delta_event_rate": post.event_rate - pre.event_rate,
        "delta_event_rate_per_100": 100.0 * (post.event_rate - pre.event_rate),
        "delta_burden_seconds_per_occurrence": (
            post.burden_seconds_per_occurrence - pre.burden_seconds_per_occurrence
        ),
        "delta_burden_seconds_per_100": 100.0
        * (post.burden_seconds_per_occurrence - pre.burden_seconds_per_occurrence),
        "delta_conditional_excess_median_seconds": (
            post.conditional_excess_median_seconds
            - pre.conditional_excess_median_seconds
        ),
        "delta_mean_duration_seconds": (
            post.mean_duration_seconds - pre.mean_duration_seconds
        ),
        "delta_sd_duration_seconds": (
            post.sd_duration_seconds - pre.sd_duration_seconds
        ),
        "delta_cv_duration": post.cv_duration - pre.cv_duration,
    }


def full_eligibility(
    early: TailMetrics,
    late: TailMetrics,
    post: TailMetrics,
    args: argparse.Namespace,
) -> bool:
    return bool(
        early.n_phrases >= args.min_early_pre_phrases
        and late.n_phrases >= args.min_late_pre_phrases
        and post.n_phrases >= args.min_post_phrases
        and late.n_days >= args.min_late_pre_days
        and post.n_days >= args.min_post_days
        and math.isfinite(late.burden_seconds_per_occurrence)
        and math.isfinite(post.burden_seconds_per_occurrence)
    )


def fold_eligibility(late: TailMetrics, post: TailMetrics, args: argparse.Namespace) -> bool:
    return bool(
        late.n_phrases >= args.min_fold_phrases
        and post.n_phrases >= args.min_fold_phrases
        and late.n_days >= args.min_fold_days
        and post.n_days >= args.min_fold_days
        and math.isfinite(late.burden_seconds_per_occurrence)
        and math.isfinite(post.burden_seconds_per_occurrence)
    )


# -----------------------------------------------------------------------------
# Recording-day permutation and bird-label permutation methods
# -----------------------------------------------------------------------------


def day_permutation_p_tail_burden(
    late_day_map: Mapping[date, Sequence[float]],
    post_day_map: Mapping[date, Sequence[float]],
    threshold_seconds: float,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, str, int, float]:
    """One-sided test of post > late-pre tail burden, permuting whole dates."""
    late_days = [(d, list(v)) for d, v in sorted(late_day_map.items())]
    post_days = [(d, list(v)) for d, v in sorted(post_day_map.items())]
    all_days = late_days + post_days
    n_late = len(late_days)
    n_total = len(all_days)
    observed = (
        calculate_tail_metrics(post_day_map, threshold_seconds).burden_seconds_per_occurrence
        - calculate_tail_metrics(late_day_map, threshold_seconds).burden_seconds_per_occurrence
    )
    if not math.isfinite(observed) or n_late == 0 or n_late == n_total:
        return math.nan, "not_tested", 0, observed

    n_assignments = math.comb(n_total, n_late)

    def statistic(chosen_late: Sequence[int]) -> float:
        chosen = set(map(int, chosen_late))
        perm_late = {all_days[i][0]: all_days[i][1] for i in range(n_total) if i in chosen}
        perm_post = {all_days[i][0]: all_days[i][1] for i in range(n_total) if i not in chosen}
        late_metric = calculate_tail_metrics(perm_late, threshold_seconds)
        post_metric = calculate_tail_metrics(perm_post, threshold_seconds)
        return post_metric.burden_seconds_per_occurrence - late_metric.burden_seconds_per_occurrence

    if n_assignments <= max_exact_assignments:
        extreme = 0
        valid = 0
        for chosen in combinations(range(n_total), n_late):
            value = statistic(chosen)
            if not math.isfinite(value):
                continue
            valid += 1
            extreme += int(value >= observed - 1e-15)
        return (extreme / valid if valid else math.nan, "exact", valid, observed)

    extreme = 0
    valid = 0
    for _ in range(monte_carlo_permutations):
        chosen = rng.choice(n_total, size=n_late, replace=False)
        value = statistic(chosen)
        if not math.isfinite(value):
            continue
        valid += 1
        extreme += int(value >= observed - 1e-15)
    p_value = (extreme + 1) / (valid + 1) if valid else math.nan
    return p_value, "monte_carlo", valid, observed


def center(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(statistic)


def group_permutation_test(
    group1: Sequence[float],
    group2: Sequence[float],
    statistic: str,
    alternative: str,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, str, int]:
    x = finite_array(group1)
    y = finite_array(group2)
    if x.size == 0 or y.size == 0:
        return math.nan, math.nan, "not_tested", 0
    pooled = np.concatenate([x, y])
    n_x = x.size
    observed = center(x, statistic) - center(y, statistic)
    n_assignments = math.comb(len(pooled), n_x)

    def is_extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - 1e-15
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - 1e-15
        raise ValueError(alternative)

    if n_assignments <= max_exact_assignments:
        extreme = 0
        total = 0
        for chosen in combinations(range(len(pooled)), n_x):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(chosen)] = True
            value = center(pooled[mask], statistic) - center(pooled[~mask], statistic)
            extreme += int(is_extreme(value))
            total += 1
        return observed, extreme / total, "exact", total

    extreme = 0
    for _ in range(monte_carlo_permutations):
        permuted = rng.permutation(pooled)
        value = center(permuted[:n_x], statistic) - center(permuted[n_x:], statistic)
        extreme += int(is_extreme(value))
    return (
        observed,
        (extreme + 1) / (monte_carlo_permutations + 1),
        "monte_carlo",
        monte_carlo_permutations,
    )


def bootstrap_group_ci(
    values: Sequence[float], replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return math.nan, math.nan
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.choice(arr, size=arr.size, replace=True)
        draws[index] = float(np.mean(sample))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


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
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        xs = rng.choice(x_arr, size=x_arr.size, replace=True)
        ys = rng.choice(y_arr, size=y_arr.size, replace=True)
        draws[index] = float(np.mean(xs) - np.mean(ys))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# -----------------------------------------------------------------------------
# Analysis specification and one-spec analysis
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    quantile: float
    split_seed: int
    post_mode: str = "full"  # full or matched_late_pre_days
    remove_longest_post: bool = False
    min_screen_post_extreme_days: int = 1
    run_confirmation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.name,
            "quantile": self.quantile,
            "split_seed": self.split_seed,
            "post_mode": self.post_mode,
            "remove_longest_post": self.remove_longest_post,
            "min_screen_post_extreme_days": self.min_screen_post_extreme_days,
            "run_confirmation": self.run_confirmation,
        }


def choose_post_dates(
    all_post_dates: Sequence[date], late_pre_dates: Sequence[date], mode: str
) -> list[date]:
    ordered = sorted(set(all_post_dates))
    if mode == "full":
        return ordered
    if mode == "matched_late_pre_days":
        return ordered[: min(len(ordered), len(set(late_pre_dates)))]
    raise ValueError(f"Unknown post mode: {mode}")


def build_syllable_metrics_for_spec(
    bird: str,
    group: str,
    info: dict[str, Any],
    spec: AnalysisSpec,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Return full syllable rows, direction rows, bird summary, and event rows."""
    durations = info["durations"]
    late_dates_all = sorted(
        {d for label_map in durations["late_pre"].values() for d in label_map}
    )
    post_dates_all = sorted(
        {d for label_map in durations["post"].values() for d in label_map}
    )
    selected_post_dates = choose_post_dates(post_dates_all, late_dates_all, spec.post_mode)
    allowed_post = set(selected_post_dates)

    fold_assignment = {
        "late_pre": split_adjacent_dates(late_dates_all, spec.split_seed, bird, "late_pre"),
        "post": split_adjacent_dates(selected_post_dates, spec.split_seed, bird, "post"),
    }

    common_labels = sorted(
        set(durations["early_pre"])
        & set(durations["late_pre"])
        & set(durations["post"]),
        key=label_sort_key,
    )

    full_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    label_cache: dict[str, dict[str, Any]] = {}

    for label in common_labels:
        early_map = subset_day_map(durations["early_pre"][label])
        late_map = subset_day_map(durations["late_pre"][label])
        post_map = subset_day_map(durations["post"][label], allowed_dates=allowed_post)
        if spec.remove_longest_post:
            post_map = remove_single_longest(post_map)

        early_values = flatten_day_map(early_map)
        threshold = (
            float(np.quantile(early_values, spec.quantile))
            if early_values.size >= args.min_early_pre_phrases
            else math.nan
        )
        early_metrics = calculate_tail_metrics(early_map, threshold)
        late_metrics = calculate_tail_metrics(late_map, threshold)
        post_metrics = calculate_tail_metrics(post_map, threshold)
        eligible_full = full_eligibility(early_metrics, late_metrics, post_metrics, args)
        deltas = tail_deltas(late_metrics, post_metrics)

        row = {
            **spec.as_dict(),
            "bird": bird,
            "group": group,
            "syllable": label,
            "threshold_seconds": threshold,
            "eligible_full": eligible_full,
            **early_metrics.as_dict("early_pre_"),
            **late_metrics.as_dict("late_pre_"),
            **post_metrics.as_dict("post_"),
            **deltas,
        }
        full_rows.append(row)

        # Output every primary-threshold late-pre/post event above threshold.
        if spec.run_confirmation and math.isfinite(threshold):
            for period, day_map in (("late_pre", late_map), ("post", post_map)):
                assignment = fold_assignment[period]
                for recording_date, values in sorted(day_map.items()):
                    for occurrence_index, duration in enumerate(values):
                        if duration > threshold:
                            event_rows.append(
                                {
                                    **spec.as_dict(),
                                    "bird": bird,
                                    "group": group,
                                    "syllable": label,
                                    "period": period,
                                    "date": recording_date.isoformat(),
                                    "relative_day": (
                                        recording_date - info["surgery_date"]
                                    ).days,
                                    "fold": assignment.get(recording_date, ""),
                                    "occurrence_index_within_date": occurrence_index,
                                    "duration_seconds": duration,
                                    "threshold_seconds": threshold,
                                    "excess_seconds": duration - threshold,
                                }
                            )

        fold_data: dict[str, dict[str, Any]] = {}
        for fold in ("A", "B"):
            late_fold_map = subset_day_map(
                late_map,
                fold_assignment=fold_assignment["late_pre"],
                fold=fold,
            )
            post_fold_map = subset_day_map(
                post_map,
                fold_assignment=fold_assignment["post"],
                fold=fold,
            )
            late_fold_metrics = calculate_tail_metrics(late_fold_map, threshold)
            post_fold_metrics = calculate_tail_metrics(post_fold_map, threshold)
            fold_data[fold] = {
                "late_map": late_fold_map,
                "post_map": post_fold_map,
                "late": late_fold_metrics,
                "post": post_fold_metrics,
                "eligible": fold_eligibility(late_fold_metrics, post_fold_metrics, args),
                "deltas": tail_deltas(late_fold_metrics, post_fold_metrics),
            }
        label_cache[label] = {
            "threshold": threshold,
            "eligible_full": eligible_full,
            "fold_data": fold_data,
        }

    # Reciprocal screening and held-out evaluation.
    for direction, screen_fold, test_fold in DIRECTIONS:
        for label in common_labels:
            cache = label_cache[label]
            screen = cache["fold_data"][screen_fold]
            heldout = cache["fold_data"][test_fold]
            eligible_both = bool(screen["eligible"] and heldout["eligible"])
            screen_delta = finite_or_nan(
                screen["deltas"]["delta_burden_seconds_per_occurrence"]
            )
            selected = bool(
                eligible_both
                and screen_delta > args.screen_min_delta_burden_seconds
                and screen["post"].n_extreme >= args.screen_min_post_extreme_events
                and screen["post"].n_extreme_days
                >= max(
                    args.screen_min_post_extreme_days,
                    spec.min_screen_post_extreme_days,
                )
            )
            heldout_p = math.nan
            p_method = "not_tested"
            n_assignments = 0
            if selected and spec.run_confirmation:
                heldout_p, p_method, n_assignments, _ = day_permutation_p_tail_burden(
                    heldout["late_map"],
                    heldout["post_map"],
                    cache["threshold"],
                    args.max_exact_day_assignments,
                    args.day_permutations,
                    stable_rng(
                        args.seed,
                        "heldout_day_permutation",
                        spec.name,
                        bird,
                        direction,
                        label,
                    ),
                )
            direction_rows.append(
                {
                    **spec.as_dict(),
                    "bird": bird,
                    "group": group,
                    "direction": direction,
                    "screen_fold": screen_fold,
                    "test_fold": test_fold,
                    "syllable": label,
                    "threshold_seconds": cache["threshold"],
                    "eligible_screen": screen["eligible"],
                    "eligible_test": heldout["eligible"],
                    "eligible_both": eligible_both,
                    "selected_in_screen": selected,
                    **screen["late"].as_dict("screen_late_pre_"),
                    **screen["post"].as_dict("screen_post_"),
                    **{f"screen_{k}": v for k, v in screen["deltas"].items()},
                    **heldout["late"].as_dict("heldout_late_pre_"),
                    **heldout["post"].as_dict("heldout_post_"),
                    **{f"heldout_{k}": v for k, v in heldout["deltas"].items()},
                    "heldout_raw_p_one_sided": heldout_p,
                    "heldout_permutation_method": p_method,
                    "heldout_n_assignments_or_permutations": n_assignments,
                    "heldout_holm_p": math.nan,
                    "direction_alpha": args.confirmation_family_alpha / 2.0,
                    "confirmed_in_direction": False,
                }
            )

    # Holm correction among selected candidates separately within each direction.
    if spec.run_confirmation:
        for direction, _, _ in DIRECTIONS:
            indices = [
                index
                for index, row in enumerate(direction_rows)
                if row["direction"] == direction
                and row["selected_in_screen"]
                and math.isfinite(finite_or_nan(row["heldout_raw_p_one_sided"]))
            ]
            adjusted = holm_adjust(
                [finite_or_nan(direction_rows[index]["heldout_raw_p_one_sided"]) for index in indices]
            )
            for index, adjusted_p in zip(indices, adjusted):
                direction_rows[index]["heldout_holm_p"] = adjusted_p
                heldout_delta = finite_or_nan(
                    direction_rows[index][
                        "heldout_delta_burden_seconds_per_occurrence"
                    ]
                )
                direction_rows[index]["confirmed_in_direction"] = bool(
                    heldout_delta > 0
                    and math.isfinite(adjusted_p)
                    and adjusted_p <= args.confirmation_family_alpha / 2.0
                )

    # Bird-level cross-fitted summaries.
    eligible_labels = sorted(
        {
            row["syllable"]
            for row in direction_rows
            if row["eligible_both"]
        },
        key=label_sort_key,
    )
    direction_summaries: dict[str, dict[str, Any]] = {}
    for direction, _, _ in DIRECTIONS:
        rows = [row for row in direction_rows if row["direction"] == direction]
        eligible = [row for row in rows if row["eligible_both"]]
        selected = [row for row in eligible if row["selected_in_screen"]]
        confirmed = [row for row in selected if row["confirmed_in_direction"]]
        n_eligible = len(eligible)
        signed_burden = (
            sum(
                finite_or_nan(row["heldout_delta_burden_seconds_per_occurrence"])
                for row in selected
                if math.isfinite(
                    finite_or_nan(row["heldout_delta_burden_seconds_per_occurrence"])
                )
            )
            / n_eligible
            if n_eligible
            else math.nan
        )
        direction_summaries[direction] = {
            "n_eligible": n_eligible,
            "n_selected": len(selected),
            "n_confirmed": len(confirmed),
            "selected_fraction": len(selected) / n_eligible if n_eligible else math.nan,
            "confirmed_fraction": len(confirmed) / n_eligible if n_eligible else math.nan,
            "signed_burden": signed_burden,
            "median_selected_heldout": median_or_nan(
                row["heldout_delta_burden_seconds_per_occurrence"] for row in selected
            ),
            "max_selected_heldout": max_or_nan(
                row["heldout_delta_burden_seconds_per_occurrence"] for row in selected
            ),
        }

    confirmed_labels = sorted(
        {
            row["syllable"]
            for row in direction_rows
            if row["confirmed_in_direction"]
        },
        key=label_sort_key,
    )
    confirmed_effects: list[float] = []
    for label in confirmed_labels:
        # Descriptive severity among confirmed syllables. These values are
        # conditional on passing held-out confirmation and should not be treated
        # as an unbiased primary effect-size estimate; the cross-fitted signed
        # burden below is the less selection-biased magnitude endpoint.
        heldout_confirmed_effects = [
            finite_or_nan(row["heldout_delta_burden_seconds_per_occurrence"])
            for row in direction_rows
            if row["syllable"] == label and row["confirmed_in_direction"]
        ]
        heldout_confirmed_effects = [
            value for value in heldout_confirmed_effects if math.isfinite(value)
        ]
        if heldout_confirmed_effects:
            confirmed_effects.append(float(np.mean(heldout_confirmed_effects)))

    eligible_full_rows = [row for row in full_rows if row["eligible_full"]]
    all_syllable_median_delta_burden = median_or_nan(
        row["delta_burden_seconds_per_occurrence"] for row in eligible_full_rows
    )
    all_syllable_median_delta_event_rate = median_or_nan(
        row["delta_event_rate"] for row in eligible_full_rows
    )

    # Whole-bird burden pools all eligible syllables and all occurrences.
    whole_late_excess_sum = 0.0
    whole_post_excess_sum = 0.0
    whole_late_n = 0
    whole_post_n = 0
    for row in eligible_full_rows:
        whole_late_excess_sum += (
            row["late_pre_burden_seconds_per_occurrence"] * row["late_pre_n_phrases"]
        )
        whole_post_excess_sum += row["post_burden_seconds_per_occurrence"] * row["post_n_phrases"]
        whole_late_n += int(row["late_pre_n_phrases"])
        whole_post_n += int(row["post_n_phrases"])
    whole_late_burden = whole_late_excess_sum / whole_late_n if whole_late_n else math.nan
    whole_post_burden = whole_post_excess_sum / whole_post_n if whole_post_n else math.nan

    bird_summary = {
        **spec.as_dict(),
        "bird": bird,
        "group": group,
        "n_common_syllables": len(common_labels),
        "n_full_eligible_syllables": len(eligible_full_rows),
        "n_crossfit_eligible_unique_syllables": len(eligible_labels),
        "confirmed_syllable_count": len(confirmed_labels),
        "confirmed_syllable_labels": ";".join(confirmed_labels),
        "confirmed_fraction": (
            len(confirmed_labels) / len(eligible_labels) if eligible_labels else math.nan
        ),
        "median_confirmed_heldout_delta_burden_seconds_per_100": (
            100.0 * median_or_nan(confirmed_effects)
        ),
        "max_confirmed_heldout_delta_burden_seconds_per_100": (
            100.0 * max_or_nan(confirmed_effects)
        ),
        "crossfitted_selected_fraction": mean_or_nan(
            summary["selected_fraction"] for summary in direction_summaries.values()
        ),
        "crossfitted_signed_burden_seconds_per_100": 100.0
        * mean_or_nan(summary["signed_burden"] for summary in direction_summaries.values()),
        "crossfitted_median_selected_heldout_delta_burden_seconds_per_100": 100.0
        * mean_or_nan(
            summary["median_selected_heldout"] for summary in direction_summaries.values()
        ),
        "crossfitted_max_selected_heldout_delta_burden_seconds_per_100": 100.0
        * mean_or_nan(
            summary["max_selected_heldout"] for summary in direction_summaries.values()
        ),
        "all_syllable_median_delta_burden_seconds_per_100": 100.0
        * all_syllable_median_delta_burden,
        "all_syllable_median_delta_event_rate_per_100": 100.0
        * all_syllable_median_delta_event_rate,
        "whole_bird_late_pre_burden_seconds_per_100": 100.0 * whole_late_burden,
        "whole_bird_post_burden_seconds_per_100": 100.0 * whole_post_burden,
        "whole_bird_delta_burden_seconds_per_100": 100.0
        * (whole_post_burden - whole_late_burden),
        "n_selected_A_screen_B_test": direction_summaries["A_screen_B_test"]["n_selected"],
        "n_selected_B_screen_A_test": direction_summaries["B_screen_A_test"]["n_selected"],
        "n_confirmed_A_screen_B_test": direction_summaries["A_screen_B_test"]["n_confirmed"],
        "n_confirmed_B_screen_A_test": direction_summaries["B_screen_A_test"]["n_confirmed"],
    }
    return full_rows, direction_rows, bird_summary, event_rows


# -----------------------------------------------------------------------------
# Group summaries/tests and baseline predictors
# -----------------------------------------------------------------------------


def group_test_rows(
    bird_rows: list[dict[str, Any]],
    endpoints: Sequence[str],
    args: argparse.Namespace,
    analysis_family: str,
    spec_name: str,
) -> list[dict[str, Any]]:
    group_order = [args.medial_group, args.lateral_group, args.sham_group]
    pair_order = [
        (args.medial_group, args.lateral_group),
        (args.medial_group, args.sham_group),
        (args.lateral_group, args.sham_group),
    ]
    output: list[dict[str, Any]] = []
    for endpoint in endpoints:
        group_values = {
            group: finite_array(
                finite_or_nan(row.get(endpoint))
                for row in bird_rows
                if row.get("group") == group
            )
            for group in group_order
        }
        endpoint_rows: list[dict[str, Any]] = []
        for group1, group2 in pair_order:
            x = group_values[group1]
            y = group_values[group2]
            observed, p_greater, method, n_assignments = group_permutation_test(
                x,
                y,
                args.group_statistic,
                "greater",
                args.max_exact_group_assignments,
                args.group_permutations,
                stable_rng(args.seed, "group_test", analysis_family, spec_name, endpoint, group1, group2),
            )
            _, p_two_sided, _, _ = group_permutation_test(
                x,
                y,
                args.group_statistic,
                "two-sided",
                args.max_exact_group_assignments,
                args.group_permutations,
                stable_rng(args.seed, "group_test_two", analysis_family, spec_name, endpoint, group1, group2),
            )
            x_lo, x_hi = bootstrap_group_ci(
                x,
                args.bootstrap_replicates,
                stable_rng(args.seed, "bootstrap_group", analysis_family, spec_name, endpoint, group1),
            )
            y_lo, y_hi = bootstrap_group_ci(
                y,
                args.bootstrap_replicates,
                stable_rng(args.seed, "bootstrap_group", analysis_family, spec_name, endpoint, group2),
            )
            d_lo, d_hi = bootstrap_difference_ci(
                x,
                y,
                args.bootstrap_replicates,
                stable_rng(args.seed, "bootstrap_difference", analysis_family, spec_name, endpoint, group1, group2),
            )
            endpoint_rows.append(
                {
                    "analysis_family": analysis_family,
                    "spec_name": spec_name,
                    "endpoint": endpoint,
                    "group1": group1,
                    "group2": group2,
                    "alternative": f"{group1} > {group2}",
                    "group_statistic": args.group_statistic,
                    "n_group1": len(x),
                    "n_group2": len(y),
                    "group1_mean": mean_or_nan(x),
                    "group1_median": median_or_nan(x),
                    "group1_bootstrap_mean_ci_low": x_lo,
                    "group1_bootstrap_mean_ci_high": x_hi,
                    "group2_mean": mean_or_nan(y),
                    "group2_median": median_or_nan(y),
                    "group2_bootstrap_mean_ci_low": y_lo,
                    "group2_bootstrap_mean_ci_high": y_hi,
                    "observed_group1_minus_group2": observed,
                    "bootstrap_mean_difference_ci_low": d_lo,
                    "bootstrap_mean_difference_ci_high": d_hi,
                    "one_sided_p_raw": p_greater,
                    "one_sided_p_holm": math.nan,
                    "two_sided_p_raw": p_two_sided,
                    "two_sided_p_holm": math.nan,
                    "permutation_method": method,
                    "n_assignments_or_permutations": n_assignments,
                }
            )
        one_adjusted = holm_adjust([row["one_sided_p_raw"] for row in endpoint_rows])
        two_adjusted = holm_adjust([row["two_sided_p_raw"] for row in endpoint_rows])
        for row, p1, p2 in zip(endpoint_rows, one_adjusted, two_adjusted):
            row["one_sided_p_holm"] = p1
            row["two_sided_p_holm"] = p2
        output.extend(endpoint_rows)
    return output


def baseline_predictor_rows(
    primary_full_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictors = (
        "early_pre_mean_duration_seconds",
        "early_pre_sd_duration_seconds",
        "early_pre_cv_duration",
        "early_pre_n_phrases",
        "threshold_seconds",
    )
    outcome = "delta_burden_seconds_per_100"
    output: list[dict[str, Any]] = []
    birds = sorted({row["bird"] for row in primary_full_rows})
    for bird in birds:
        rows = [
            row
            for row in primary_full_rows
            if row["bird"] == bird and row["eligible_full"]
        ]
        group = rows[0]["group"] if rows else ""
        for predictor in predictors:
            rho, n = spearman_rho(
                [finite_or_nan(row.get(predictor)) for row in rows],
                [finite_or_nan(row.get(outcome)) for row in rows],
            )
            output.append(
                {
                    "bird": bird,
                    "group": group,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_syllables": n,
                    "spearman_rho": rho,
                }
            )
    return output


def leave_one_bird_out_rows(
    primary_bird_rows: list[dict[str, Any]],
    endpoints: Sequence[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pairs = [
        (args.medial_group, args.lateral_group),
        (args.medial_group, args.sham_group),
        (args.lateral_group, args.sham_group),
    ]
    output: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for group1, group2 in pairs:
            relevant = [
                row for row in primary_bird_rows if row["group"] in {group1, group2}
            ]
            for omitted in relevant:
                kept = [row for row in relevant if row["bird"] != omitted["bird"]]
                x = [
                    finite_or_nan(row.get(endpoint))
                    for row in kept
                    if row["group"] == group1
                ]
                y = [
                    finite_or_nan(row.get(endpoint))
                    for row in kept
                    if row["group"] == group2
                ]
                observed, p_value, method, n_assignments = group_permutation_test(
                    x,
                    y,
                    args.group_statistic,
                    "greater",
                    args.max_exact_group_assignments,
                    args.group_permutations,
                    stable_rng(
                        args.seed,
                        "leave_one_out",
                        endpoint,
                        group1,
                        group2,
                        omitted["bird"],
                    ),
                )
                output.append(
                    {
                        "endpoint": endpoint,
                        "group1": group1,
                        "group2": group2,
                        "omitted_bird": omitted["bird"],
                        "omitted_group": omitted["group"],
                        "n_group1_after_omission": len(finite_array(x)),
                        "n_group2_after_omission": len(finite_array(y)),
                        "observed_group1_minus_group2": observed,
                        "one_sided_p_raw": p_value,
                        "permutation_method": method,
                        "n_assignments_or_permutations": n_assignments,
                    }
                )
    return output


# -----------------------------------------------------------------------------
# Main
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

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )

    bird_data: dict[str, dict[str, Any]] = {}
    input_audit_rows: list[dict[str, Any]] = []

    for path in database_paths:
        bird = path.stem.removesuffix("_decoded_database")
        if bird not in metadata:
            raise KeyError(f"Bird {bird!r} is absent from {metadata_path}")
        group = str(metadata[bird]["lesion_group"])
        surgery = parse_iso_date(metadata[bird]["lesion_surgery_date"])
        durations: dict[str, dict[str, dict[date, list[float]]]] = {
            period: defaultdict(lambda: defaultdict(list))
            for period in ("early_pre", "late_pre", "post")
        }
        results = json.loads(path.read_text()).get("results", [])
        n_bad_spans = 0
        for result in results:
            recording_date = parse_iso_date(result["creation_date"])
            relative_day = (recording_date - surgery).days
            period = classify_period(relative_day, args)
            if period is None:
                continue
            for label, spans in result.get("syllable_onsets_offsets_ms", {}).items():
                for span in spans:
                    try:
                        start, end = float(span[0]), float(span[1])
                        duration_seconds = (end - start) / 1000.0
                    except (TypeError, ValueError, IndexError):
                        n_bad_spans += 1
                        continue
                    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
                        n_bad_spans += 1
                        continue
                    durations[period][str(label)][recording_date].append(duration_seconds)

        date_counts = {
            period: len(
                {
                    recording_date
                    for date_map in durations[period].values()
                    for recording_date in date_map
                }
            )
            for period in durations
        }
        common_labels = (
            set(durations["early_pre"])
            & set(durations["late_pre"])
            & set(durations["post"])
        )
        bird_data[bird] = {
            "bird": bird,
            "group": group,
            "surgery_date": surgery,
            "durations": durations,
            "database_path": path,
        }
        input_audit_rows.append(
            {
                "bird": bird,
                "group": group,
                "database_path": str(path),
                "surgery_date": surgery.isoformat(),
                "n_results_records": len(results),
                "n_early_pre_dates": date_counts["early_pre"],
                "n_late_pre_dates": date_counts["late_pre"],
                "n_post_dates": date_counts["post"],
                "n_common_syllables_all_three_periods": len(common_labels),
                "n_invalid_spans_skipped": n_bad_spans,
            }
        )

    primary_spec = AnalysisSpec(
        name="primary",
        quantile=args.primary_quantile,
        split_seed=args.seed,
        post_mode="full",
        remove_longest_post=False,
        min_screen_post_extreme_days=args.screen_min_post_extreme_days,
        run_confirmation=True,
    )

    primary_full_rows: list[dict[str, Any]] = []
    primary_direction_rows: list[dict[str, Any]] = []
    primary_bird_rows: list[dict[str, Any]] = []
    primary_event_rows: list[dict[str, Any]] = []
    for bird in sorted(bird_data):
        info = bird_data[bird]
        full_rows, direction_rows, bird_summary, event_rows = build_syllable_metrics_for_spec(
            bird, info["group"], info, primary_spec, args
        )
        primary_full_rows.extend(full_rows)
        primary_direction_rows.extend(direction_rows)
        primary_bird_rows.append(bird_summary)
        primary_event_rows.extend(event_rows)

    # Audit the exact primary date folds used for cross-screening.
    primary_fold_rows: list[dict[str, Any]] = []
    for bird in sorted(bird_data):
        info = bird_data[bird]
        durations = info["durations"]
        late_dates = sorted(
            {d for label_map in durations["late_pre"].values() for d in label_map}
        )
        post_dates = sorted(
            {d for label_map in durations["post"].values() for d in label_map}
        )
        assignments = {
            "late_pre": split_adjacent_dates(late_dates, args.seed, bird, "late_pre"),
            "post": split_adjacent_dates(post_dates, args.seed, bird, "post"),
        }
        for period, dates in (("late_pre", late_dates), ("post", post_dates)):
            for recording_date in dates:
                primary_fold_rows.append(
                    {
                        "bird": bird,
                        "group": info["group"],
                        "period": period,
                        "date": recording_date.isoformat(),
                        "relative_day": (recording_date - info["surgery_date"]).days,
                        "fold": assignments[period].get(recording_date, ""),
                    }
                )

    # One row per bird x syllable that was confirmed in at least one direction.
    primary_confirmed_rows: list[dict[str, Any]] = []
    confirmed_keys = sorted(
        {
            (row["bird"], row["syllable"])
            for row in primary_direction_rows
            if row["confirmed_in_direction"]
        },
        key=lambda item: (item[0], label_sort_key(item[1])),
    )
    for bird, label in confirmed_keys:
        rows = [
            row
            for row in primary_direction_rows
            if row["bird"] == bird and row["syllable"] == label
        ]
        confirmed = [row for row in rows if row["confirmed_in_direction"]]
        selected = [row for row in rows if row["selected_in_screen"]]
        primary_confirmed_rows.append(
            {
                "bird": bird,
                "group": rows[0]["group"],
                "syllable": label,
                "threshold_seconds": rows[0]["threshold_seconds"],
                "selected_directions": ";".join(row["direction"] for row in selected),
                "confirmed_directions": ";".join(row["direction"] for row in confirmed),
                "n_selected_directions": len(selected),
                "n_confirmed_directions": len(confirmed),
                "mean_confirmed_heldout_delta_burden_seconds_per_100": 100.0
                * mean_or_nan(
                    row["heldout_delta_burden_seconds_per_occurrence"]
                    for row in confirmed
                ),
                "max_confirmed_heldout_delta_burden_seconds_per_100": 100.0
                * max_or_nan(
                    row["heldout_delta_burden_seconds_per_occurrence"]
                    for row in confirmed
                ),
                "minimum_raw_heldout_p": min(
                    finite_or_nan(row["heldout_raw_p_one_sided"]) for row in confirmed
                ),
                "minimum_holm_heldout_p": min(
                    finite_or_nan(row["heldout_holm_p"]) for row in confirmed
                ),
            }
        )

    supplemental_bird_rows = [
        {
            "bird": row["bird"],
            "group": row["group"],
            "n_full_eligible_syllables": row["n_full_eligible_syllables"],
            "all_syllable_median_delta_burden_seconds_per_100": row[
                "all_syllable_median_delta_burden_seconds_per_100"
            ],
            "all_syllable_median_delta_event_rate_per_100": row[
                "all_syllable_median_delta_event_rate_per_100"
            ],
            "whole_bird_late_pre_burden_seconds_per_100": row[
                "whole_bird_late_pre_burden_seconds_per_100"
            ],
            "whole_bird_post_burden_seconds_per_100": row[
                "whole_bird_post_burden_seconds_per_100"
            ],
            "whole_bird_delta_burden_seconds_per_100": row[
                "whole_bird_delta_burden_seconds_per_100"
            ],
        }
        for row in primary_bird_rows
    ]

    localization_endpoints = (
        "confirmed_fraction",
        "crossfitted_signed_burden_seconds_per_100",
    )
    supplemental_endpoints = (
        "all_syllable_median_delta_burden_seconds_per_100",
        "all_syllable_median_delta_event_rate_per_100",
        "whole_bird_delta_burden_seconds_per_100",
    )
    primary_group_rows = group_test_rows(
        primary_bird_rows,
        localization_endpoints,
        args,
        analysis_family="primary_localization",
        spec_name="primary",
    )
    supplemental_group_rows = group_test_rows(
        primary_bird_rows,
        supplemental_endpoints,
        args,
        analysis_family="supplemental_repertoire",
        spec_name="primary",
    )
    predictor_rows = baseline_predictor_rows(primary_full_rows)
    loo_rows = leave_one_bird_out_rows(primary_bird_rows, localization_endpoints, args)

    # Sensitivity specifications use cross-fitted continuous endpoints but skip
    # the expensive per-candidate confirmation tests.
    sensitivity_specs: list[AnalysisSpec] = []
    sensitivity_bird_rows: list[dict[str, Any]] = []
    sensitivity_group_rows: list[dict[str, Any]] = []
    if not args.skip_sensitivity:
        for q in sorted(set(args.sensitivity_quantiles)):
            if abs(q - args.primary_quantile) < 1e-12:
                continue
            sensitivity_specs.append(
                AnalysisSpec(
                    name=f"quantile_{q:g}",
                    quantile=q,
                    split_seed=args.seed,
                )
            )
        sensitivity_specs.extend(
            [
                AnalysisSpec(
                    name="matched_post_days",
                    quantile=args.primary_quantile,
                    split_seed=args.seed,
                    post_mode="matched_late_pre_days",
                ),
                AnalysisSpec(
                    name="remove_longest_post_per_syllable",
                    quantile=args.primary_quantile,
                    split_seed=args.seed,
                    remove_longest_post=True,
                ),
                AnalysisSpec(
                    name="require_extremes_on_two_post_days",
                    quantile=args.primary_quantile,
                    split_seed=args.seed,
                    min_screen_post_extreme_days=2,
                ),
            ]
        )
        for index in range(args.sensitivity_random_splits):
            sensitivity_specs.append(
                AnalysisSpec(
                    name=f"alternate_split_{index + 1}",
                    quantile=args.primary_quantile,
                    split_seed=args.seed + index + 1,
                )
            )

        sensitivity_endpoints = (
            "crossfitted_selected_fraction",
            "crossfitted_signed_burden_seconds_per_100",
            "crossfitted_median_selected_heldout_delta_burden_seconds_per_100",
            "all_syllable_median_delta_burden_seconds_per_100",
            "whole_bird_delta_burden_seconds_per_100",
        )
        for spec in sensitivity_specs:
            spec_bird_rows: list[dict[str, Any]] = []
            for bird in sorted(bird_data):
                info = bird_data[bird]
                _, _, bird_summary, _ = build_syllable_metrics_for_spec(
                    bird, info["group"], info, spec, args
                )
                spec_bird_rows.append(bird_summary)
            sensitivity_bird_rows.extend(spec_bird_rows)
            sensitivity_group_rows.extend(
                group_test_rows(
                    spec_bird_rows,
                    sensitivity_endpoints,
                    args,
                    analysis_family="sensitivity",
                    spec_name=spec.name,
                )
            )

    # CSV outputs.
    write_csv(
        args.output / "input_audit.csv",
        input_audit_rows,
        [
            "bird",
            "group",
            "database_path",
            "surgery_date",
            "n_results_records",
            "n_early_pre_dates",
            "n_late_pre_dates",
            "n_post_dates",
            "n_common_syllables_all_three_periods",
            "n_invalid_spans_skipped",
        ],
    )
    write_csv(
        args.output / "primary_recording_day_folds.csv",
        primary_fold_rows,
        ["bird", "group", "period", "date", "relative_day", "fold"],
    )
    write_csv(
        args.output / "primary_all_syllable_tail_metrics.csv",
        primary_full_rows,
        list(primary_full_rows[0].keys()) if primary_full_rows else ["bird"],
    )
    write_csv(
        args.output / "primary_cross_screened_syllables.csv",
        primary_direction_rows,
        list(primary_direction_rows[0].keys()) if primary_direction_rows else ["bird"],
    )
    write_csv(
        args.output / "primary_extreme_events.csv",
        primary_event_rows,
        list(primary_event_rows[0].keys())
        if primary_event_rows
        else [
            "bird",
            "group",
            "syllable",
            "period",
            "date",
            "duration_seconds",
            "threshold_seconds",
            "excess_seconds",
        ],
    )
    write_csv(
        args.output / "primary_confirmed_syllables.csv",
        primary_confirmed_rows,
        list(primary_confirmed_rows[0].keys())
        if primary_confirmed_rows
        else [
            "bird",
            "group",
            "syllable",
            "threshold_seconds",
            "confirmed_directions",
        ],
    )
    write_csv(
        args.output / "primary_bird_localization_results.csv",
        primary_bird_rows,
        list(primary_bird_rows[0].keys()) if primary_bird_rows else ["bird"],
    )
    write_csv(
        args.output / "supplemental_bird_repertoire_results.csv",
        supplemental_bird_rows,
        list(supplemental_bird_rows[0].keys())
        if supplemental_bird_rows
        else ["bird"],
    )
    write_csv(
        args.output / "primary_localization_group_tests.csv",
        primary_group_rows,
        list(primary_group_rows[0].keys()) if primary_group_rows else ["endpoint"],
    )
    write_csv(
        args.output / "supplemental_repertoire_group_tests.csv",
        supplemental_group_rows,
        list(supplemental_group_rows[0].keys())
        if supplemental_group_rows
        else ["endpoint"],
    )
    write_csv(
        args.output / "baseline_predictor_correlations.csv",
        predictor_rows,
        list(predictor_rows[0].keys()) if predictor_rows else ["bird"],
    )
    write_csv(
        args.output / "leave_one_bird_out_primary_tests.csv",
        loo_rows,
        list(loo_rows[0].keys()) if loo_rows else ["endpoint"],
    )
    write_csv(
        args.output / "sensitivity_specifications.csv",
        [spec.as_dict() for spec in sensitivity_specs],
        list(sensitivity_specs[0].as_dict().keys())
        if sensitivity_specs
        else ["spec_name"],
    )
    write_csv(
        args.output / "sensitivity_bird_results.csv",
        sensitivity_bird_rows,
        list(sensitivity_bird_rows[0].keys()) if sensitivity_bird_rows else ["bird"],
    )
    write_csv(
        args.output / "sensitivity_group_tests.csv",
        sensitivity_group_rows,
        list(sensitivity_group_rows[0].keys())
        if sensitivity_group_rows
        else ["endpoint"],
    )

    # Human-readable summary.
    groups = [args.medial_group, args.lateral_group, args.sham_group]
    lines = [
        "Extreme phrase localization analysis",
        "===================================",
        f"Input directory: {args.json_dir.resolve()}",
        f"Birds read: {len(primary_bird_rows)}",
        (
            "Windows: early pre "
            f"[{args.early_pre_start_day}, {args.early_pre_end_day}], late pre "
            f"[{args.late_pre_start_day}, {args.late_pre_end_day}], post "
            f"[{args.post_start_day}, {args.post_end_day if args.post_end_day is not None else 'all'}]"
        ),
        f"Primary threshold quantile: {args.primary_quantile:g}",
        f"Direction-specific confirmation alpha: {args.confirmation_family_alpha / 2:g}",
        "",
        "Primary bird-level localization summaries",
        "-----------------------------------------",
    ]
    for group in groups:
        rows = [row for row in primary_bird_rows if row["group"] == group]
        lines.append(
            f"{group}: n={len(rows)}, median confirmed fraction="
            f"{fmt(median_or_nan(row['confirmed_fraction'] for row in rows), 4)}, "
            "median cross-fitted signed burden (s/100 occurrences)="
            f"{fmt(median_or_nan(row['crossfitted_signed_burden_seconds_per_100'] for row in rows), 4)}, "
            "median whole-bird delta burden (s/100 occurrences)="
            f"{fmt(median_or_nan(row['whole_bird_delta_burden_seconds_per_100'] for row in rows), 4)}"
        )
    lines.extend(["", "Primary planned group contrasts", "-------------------------------"])
    for row in primary_group_rows:
        if row["endpoint"] not in {
            "confirmed_fraction",
            "crossfitted_signed_burden_seconds_per_100",
        }:
            continue
        lines.append(
            f"{row['endpoint']}: {row['group1']} > {row['group2']}; "
            f"effect={fmt(finite_or_nan(row['observed_group1_minus_group2']))}, "
            f"raw p={fmt(finite_or_nan(row['one_sided_p_raw']))}, "
            f"Holm p={fmt(finite_or_nan(row['one_sided_p_holm']))}"
        )
    lines.extend(
        [
            "",
            "Interpretation reminder",
            "-----------------------",
            "confirmed_fraction is the fraction of cross-fit-eligible syllables that passed",
            "held-out day-level testing after Holm correction in either reciprocal direction.",
            "crossfitted_signed_burden gives unselected syllables zero weight and includes",
            "negative held-out effects among screened syllables, reducing winner's-curse bias.",
            "The all-syllable median and whole-bird burden answer different supplemental",
            "questions: typical-syllable change versus total behavioral tail burden.",
            "",
            f"Results written to: {args.output.resolve()}",
        ]
    )
    summary_text = "\n".join(lines)
    (args.output / "summary.txt").write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
