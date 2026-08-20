#!/usr/bin/env python3
"""Explore and rigorously test phrase-duration syllable cutoffs.

This standalone script analyzes the AFP lesion JSON directory structure:

    AFP_lesion_jsons/
        AFP_lesion_bird_metadata.json
        <group>/<bird>/<bird>_decoded_database.json

Each decoded database must contain a top-level ``results`` list. Each result must
contain ``creation_date`` and ``syllable_onsets_offsets_ms``. The metadata file
must map bird IDs to ``lesion_group`` and ``lesion_surgery_date``.

The script evaluates whether the phrase-duration variability effect is
concentrated in an upper-ranked subset of syllables and whether the current 30%
cutoff lies within a stable range. It includes four ranking designs:

1. full_pooled_variance
   Rank with average pre/post variance and estimate change in the same data.
   This most closely mirrors the current top-30% analysis and is exploratory.

2. full_pre_variance
   Rank using pre-lesion variance only, then estimate post-minus-pre change.

3. crossfit_pooled_variance
   Split recording dates into reciprocal A/B folds. Rank by pooled variance in
   one fold and estimate change in the held-out fold; reverse and average.

4. crossfit_pre_variance
   Rank by pre-lesion variance in one fold and estimate change in the held-out
   fold; reverse and average. This is the cleanest selection/estimation split.

At every requested cutoff, the script calculates bird-level medians for:

* selected-syllable delta SD and delta CV
* remaining-syllable delta SD and delta CV
* selected-minus-remaining contrasts

It then performs bird-level group comparisons, bootstrap confidence intervals,
pointwise exact/Monte Carlo permutation tests, and two global tests correcting
for the search across cutoffs:

* max-t: evidence for an especially affected upper subset at any cutoff
* mean-t: evidence for a stable effect across the full cutoff range

The bird is always the independent experimental unit. Group labels are
permuted only at the bird level.

Example
-------
python run_phrase_duration_cutoff_scan.py \
    "$HOME/Desktop/AFP_lesion_jsons" \
    --output "$HOME/Desktop/phrase_duration_cutoff_scan_results" \
    --seed 123

Dependencies
------------
Required: numpy
Optional for figures: matplotlib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


DEFAULT_FRACTIONS = tuple(np.round(np.arange(0.10, 0.6001, 0.05), 2))
DIRECTIONS = (
    ("A_screen_B_test", "A", "B"),
    ("B_screen_A_test", "B", "A"),
)
METHODS = (
    "full_pooled_variance",
    "full_pre_variance",
    "crossfit_pooled_variance",
    "crossfit_pre_variance",
)
ENDPOINTS = ("delta_sd", "delta_cv")
SUMMARY_TYPES = ("selected_median", "selected_minus_remaining")


@dataclass(frozen=True)
class PeriodMetrics:
    n_pre_phrases: int
    n_post_phrases: int
    n_pre_days: int
    n_post_days: int
    pre_mean: float
    post_mean: float
    pre_sd: float
    post_sd: float
    pre_cv: float
    post_cv: float
    pre_variance: float
    post_variance: float
    delta_sd: float
    delta_cv: float

    @property
    def pooled_variance(self) -> float:
        if not (math.isfinite(self.pre_variance) and math.isfinite(self.post_variance)):
            return math.nan
        return 0.5 * (self.pre_variance + self.post_variance)

    @property
    def valid(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                self.pre_mean,
                self.post_mean,
                self.pre_sd,
                self.post_sd,
                self.pre_cv,
                self.post_cv,
                self.pre_variance,
                self.post_variance,
                self.delta_sd,
                self.delta_cv,
            )
        )


# -----------------------------------------------------------------------------
# Arguments and validation
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a cutoff scan for phrase-duration variability effects.",
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing AFP_lesion_bird_metadata.json and group folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phrase_duration_cutoff_scan_results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Top-ranked fractions to retain.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pre-start-day", type=int, default=-14)
    parser.add_argument("--pre-end-day", type=int, default=-1)
    parser.add_argument("--post-start-day", type=int, default=1)
    parser.add_argument("--post-end-day", type=int, default=14)
    parser.add_argument(
        "--min-phrases",
        type=int,
        default=10,
        help="Minimum phrase occurrences in each period or held-out fold.",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=2,
        help="Minimum recording days in each period or held-out fold.",
    )
    parser.add_argument(
        "--min-eligible-syllables",
        type=int,
        default=3,
        help="Minimum eligible syllables required for a bird at a method/direction.",
    )
    parser.add_argument(
        "--balance-draws",
        type=int,
        default=200,
        help="Equal-count pre/post subsampling repetitions for SD/CV estimates.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
        help="Bird-level bootstrap replicates for pointwise contrast CIs.",
    )
    parser.add_argument(
        "--max-exact-assignments",
        type=int,
        default=2_000_000,
        help="Maximum label assignments enumerated exactly.",
    )
    parser.add_argument(
        "--monte-carlo-permutations",
        type=int,
        default=200_000,
        help="Permutations used when exact enumeration is too large.",
    )
    parser.add_argument(
        "--allow-one-crossfit-direction",
        action="store_true",
        help="Allow a cross-fitted bird estimate when only one reciprocal direction is available.",
    )
    parser.add_argument(
        "--medial-group",
        default="medial_and_lateral",
        help="Metadata label for the medial+lateral group.",
    )
    parser.add_argument(
        "--lateral-group",
        default="lateral_only",
        help="Metadata label for the lateral-only group.",
    )
    parser.add_argument(
        "--sham-group",
        default="sham_saline",
        help="Metadata label for the sham group.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[float]:
    if not args.json_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.json_dir}")
    if not args.pre_start_day <= args.pre_end_day < 0:
        raise ValueError("Pre-lesion range must end before day 0.")
    if not 0 < args.post_start_day <= args.post_end_day:
        raise ValueError("Post-lesion range must begin after day 0.")
    if args.min_phrases < 2:
        raise ValueError("--min-phrases must be at least 2.")
    if args.min_days < 1:
        raise ValueError("--min-days must be at least 1.")
    if args.min_eligible_syllables < 2:
        raise ValueError("--min-eligible-syllables must be at least 2.")
    if args.balance_draws < 1:
        raise ValueError("--balance-draws must be at least 1.")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative.")

    fractions = sorted(set(float(value) for value in args.fractions))
    if not fractions:
        raise ValueError("At least one cutoff fraction is required.")
    for fraction in fractions:
        if not 0 < fraction < 1:
            raise ValueError(f"Cutoffs must lie strictly between 0 and 1: {fraction}")
    if max(fractions) >= 1.0:
        raise ValueError("The largest cutoff must leave a remaining-syllable set.")
    return fractions


# -----------------------------------------------------------------------------
# Reproducible helpers
# -----------------------------------------------------------------------------


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    text = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


def label_sort_key(label: str) -> tuple[int, Any]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def parse_iso_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def safe_mean(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.mean(array)) if array.size else math.nan


def safe_median(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.median(array)) if array.size else math.nan


def sample_sd(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    value = float(np.std(values, ddof=1))
    return value if math.isfinite(value) else math.nan


def split_adjacent_dates(
    dates: Sequence[date], seed: int, bird: str, period: str
) -> dict[date, str]:
    """Split dates into balanced A/B folds while keeping adjacent dates paired."""
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


# -----------------------------------------------------------------------------
# Input loading and metric calculation
# -----------------------------------------------------------------------------


def find_database_paths(json_dir: Path) -> list[Path]:
    paths = sorted(json_dir.glob("*/*/*decoded_database.json"))
    if not paths:
        paths = sorted(json_dir.rglob("*decoded_database.json"))
    return paths


def load_metadata(path: Path) -> Mapping[str, Mapping[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"Expected a bird-keyed metadata object in {path}")
    return data


def flatten_day_values(day_map: Mapping[date, Sequence[float]]) -> np.ndarray:
    pieces = [np.asarray(values, dtype=float) for _, values in sorted(day_map.items())]
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def balanced_period_metrics(
    pre_day_map: Mapping[date, Sequence[float]],
    post_day_map: Mapping[date, Sequence[float]],
    min_phrases: int,
    min_days: int,
    n_draws: int,
    rng: np.random.Generator,
) -> PeriodMetrics | None:
    pre = flatten_day_values(pre_day_map)
    post = flatten_day_values(post_day_map)
    n_pre_days = sum(bool(values) for values in pre_day_map.values())
    n_post_days = sum(bool(values) for values in post_day_map.values())

    if (
        pre.size < min_phrases
        or post.size < min_phrases
        or n_pre_days < min_days
        or n_post_days < min_days
    ):
        return None

    n_balance = min(pre.size, post.size)
    if n_balance < 2:
        return None

    draw_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_draws):
        pre_draw = (
            pre
            if pre.size == n_balance
            else pre[rng.choice(pre.size, size=n_balance, replace=False)]
        )
        post_draw = (
            post
            if post.size == n_balance
            else post[rng.choice(post.size, size=n_balance, replace=False)]
        )

        pre_mean = float(np.mean(pre_draw))
        post_mean = float(np.mean(post_draw))
        pre_sd = sample_sd(pre_draw)
        post_sd = sample_sd(post_draw)
        if (
            not math.isfinite(pre_mean)
            or not math.isfinite(post_mean)
            or pre_mean <= 0
            or post_mean <= 0
            or not math.isfinite(pre_sd)
            or not math.isfinite(post_sd)
        ):
            continue

        pre_cv = pre_sd / pre_mean
        post_cv = post_sd / post_mean
        values = {
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "pre_sd": pre_sd,
            "post_sd": post_sd,
            "pre_cv": pre_cv,
            "post_cv": post_cv,
            "pre_variance": pre_sd * pre_sd,
            "post_variance": post_sd * post_sd,
            "delta_sd": post_sd - pre_sd,
            "delta_cv": post_cv - pre_cv,
        }
        if all(math.isfinite(value) for value in values.values()):
            for key, value in values.items():
                draw_values[key].append(value)

    if not draw_values["delta_sd"]:
        return None

    metrics = PeriodMetrics(
        n_pre_phrases=int(pre.size),
        n_post_phrases=int(post.size),
        n_pre_days=n_pre_days,
        n_post_days=n_post_days,
        pre_mean=safe_mean(draw_values["pre_mean"]),
        post_mean=safe_mean(draw_values["post_mean"]),
        pre_sd=safe_mean(draw_values["pre_sd"]),
        post_sd=safe_mean(draw_values["post_sd"]),
        pre_cv=safe_mean(draw_values["pre_cv"]),
        post_cv=safe_mean(draw_values["post_cv"]),
        pre_variance=safe_mean(draw_values["pre_variance"]),
        post_variance=safe_mean(draw_values["post_variance"]),
        delta_sd=safe_mean(draw_values["delta_sd"]),
        delta_cv=safe_mean(draw_values["delta_cv"]),
    )
    return metrics if metrics.valid else None


def metric_to_dict(prefix: str, metrics: PeriodMetrics | None) -> dict[str, Any]:
    fields = (
        "n_pre_phrases",
        "n_post_phrases",
        "n_pre_days",
        "n_post_days",
        "pre_mean",
        "post_mean",
        "pre_sd",
        "post_sd",
        "pre_cv",
        "post_cv",
        "pre_variance",
        "post_variance",
        "pooled_variance",
        "delta_sd",
        "delta_cv",
    )
    if metrics is None:
        return {f"{prefix}{field}": math.nan for field in fields}
    result: dict[str, Any] = {}
    for field in fields:
        result[f"{prefix}{field}"] = getattr(metrics, field)
    return result


def load_bird_data(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = args.json_dir / "AFP_lesion_bird_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = load_metadata(metadata_path)

    paths = find_database_paths(args.json_dir)
    if not paths:
        raise FileNotFoundError(
            f"No decoded database JSON files found under {args.json_dir}"
        )

    bird_data: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []

    for path in paths:
        stem = path.stem
        suffix = "_decoded_database"
        bird = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        if bird not in metadata:
            raise KeyError(f"Bird {bird!r} is missing from {metadata_path}")

        group = str(metadata[bird]["lesion_group"])
        surgery = parse_iso_date(str(metadata[bird]["lesion_surgery_date"]))
        durations: dict[str, dict[str, dict[date, list[float]]]] = {
            "pre": defaultdict(lambda: defaultdict(list)),
            "post": defaultdict(lambda: defaultdict(list)),
        }

        raw = json.loads(path.read_text())
        results = raw.get("results", [])
        n_valid_spans = 0
        for result in results:
            if "creation_date" not in result:
                continue
            recording_date = parse_iso_date(str(result["creation_date"]))
            relative_day = (recording_date - surgery).days
            if args.pre_start_day <= relative_day <= args.pre_end_day:
                period = "pre"
            elif args.post_start_day <= relative_day <= args.post_end_day:
                period = "post"
            else:
                continue

            spans_by_label = result.get("syllable_onsets_offsets_ms", {})
            if not isinstance(spans_by_label, dict):
                continue
            for raw_label, spans in spans_by_label.items():
                label = str(raw_label)
                for span in spans:
                    try:
                        start, end = span
                        duration_s = (float(end) - float(start)) / 1000.0
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(duration_s) and duration_s > 0:
                        durations[period][label][recording_date].append(duration_s)
                        n_valid_spans += 1

        all_dates: dict[str, list[date]] = {}
        assignments: dict[str, dict[date, str]] = {}
        for period in ("pre", "post"):
            dates = sorted(
                {
                    recording_date
                    for label_dates in durations[period].values()
                    for recording_date in label_dates
                }
            )
            all_dates[period] = dates
            assignments[period] = split_adjacent_dates(dates, args.seed, bird, period)
            for recording_date in dates:
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
            set(durations["pre"]) & set(durations["post"]), key=label_sort_key
        )
        bird_data[bird] = {
            "bird": bird,
            "group": group,
            "surgery": surgery,
            "path": path,
            "durations": durations,
            "assignments": assignments,
            "common_labels": common_labels,
        }
        audit_rows.append(
            {
                "bird": bird,
                "group": group,
                "surgery_date": surgery.isoformat(),
                "database_path": str(path),
                "n_results": len(results),
                "n_valid_phrase_spans_in_window": n_valid_spans,
                "n_pre_dates": len(all_dates["pre"]),
                "n_post_dates": len(all_dates["post"]),
                "n_common_syllables": len(common_labels),
            }
        )

    return bird_data, audit_rows, date_rows


def period_day_map(
    info: Mapping[str, Any], label: str, period: str, fold: str | None
) -> dict[date, list[float]]:
    day_map = info["durations"][period][label]
    if fold is None:
        return {recording_date: values for recording_date, values in day_map.items()}
    assignments = info["assignments"][period]
    return {
        recording_date: values
        for recording_date, values in day_map.items()
        if assignments.get(recording_date) == fold
    }


# -----------------------------------------------------------------------------
# Syllable metrics and cutoff summaries
# -----------------------------------------------------------------------------


def build_syllable_metrics(
    bird_data: Mapping[str, Mapping[str, Any]], args: argparse.Namespace
) -> tuple[
    dict[str, dict[str, PeriodMetrics]],
    dict[tuple[str, str], dict[str, PeriodMetrics]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    full_metrics: dict[str, dict[str, PeriodMetrics]] = defaultdict(dict)
    fold_metrics: dict[tuple[str, str], dict[str, PeriodMetrics]] = defaultdict(dict)
    full_rows: list[dict[str, Any]] = []
    crossfit_rows: list[dict[str, Any]] = []

    for bird in sorted(bird_data):
        info = bird_data[bird]
        group = str(info["group"])
        for label in info["common_labels"]:
            full = balanced_period_metrics(
                period_day_map(info, label, "pre", None),
                period_day_map(info, label, "post", None),
                args.min_phrases,
                args.min_days,
                args.balance_draws,
                stable_rng(args.seed, "balanced", bird, label, "full"),
            )
            if full is not None:
                full_metrics[bird][label] = full
            full_rows.append(
                {
                    "bird": bird,
                    "group": group,
                    "syllable": label,
                    "eligible": full is not None,
                    **metric_to_dict("", full),
                }
            )

            for fold in ("A", "B"):
                fold_value = balanced_period_metrics(
                    period_day_map(info, label, "pre", fold),
                    period_day_map(info, label, "post", fold),
                    args.min_phrases,
                    args.min_days,
                    args.balance_draws,
                    stable_rng(args.seed, "balanced", bird, label, fold),
                )
                if fold_value is not None:
                    fold_metrics[(bird, fold)][label] = fold_value

        for direction, screen_fold, test_fold in DIRECTIONS:
            labels = sorted(
                set(fold_metrics[(bird, screen_fold)])
                & set(fold_metrics[(bird, test_fold)]),
                key=label_sort_key,
            )
            for label in labels:
                screen = fold_metrics[(bird, screen_fold)][label]
                test = fold_metrics[(bird, test_fold)][label]
                crossfit_rows.append(
                    {
                        "bird": bird,
                        "group": group,
                        "direction": direction,
                        "screen_fold": screen_fold,
                        "test_fold": test_fold,
                        "syllable": label,
                        **metric_to_dict("screen_", screen),
                        **metric_to_dict("heldout_", test),
                    }
                )

    return full_metrics, fold_metrics, full_rows, crossfit_rows


def rank_score(metrics: PeriodMetrics, method: str) -> float:
    if method.endswith("pooled_variance"):
        return metrics.pooled_variance
    if method.endswith("pre_variance"):
        return metrics.pre_variance
    raise ValueError(f"Unknown method: {method}")


def cutoff_count(fraction: float, n_eligible: int) -> int:
    if n_eligible < 2:
        return 0
    requested = max(1, int(math.ceil(fraction * n_eligible)))
    return min(requested, n_eligible - 1)


def summarize_ranked_candidates(
    candidates: list[dict[str, Any]],
    fractions: Sequence[float],
    bird: str,
    group: str,
    method: str,
    direction: str,
    min_eligible_syllables: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        row
        for row in candidates
        if math.isfinite(float(row["rank_score"]))
        and math.isfinite(float(row["delta_sd"]))
        and math.isfinite(float(row["delta_cv"]))
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            label_sort_key(str(row["syllable"])),
        )
    )
    n_eligible = len(candidates)
    rank_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(candidates, start=1):
        rank_rows.append(
            {
                "method": method,
                "direction": direction,
                "bird": bird,
                "group": group,
                "syllable": row["syllable"],
                "rank": rank,
                "rank_percentile": rank / n_eligible if n_eligible else math.nan,
                "rank_score": row["rank_score"],
                "delta_sd": row["delta_sd"],
                "delta_cv": row["delta_cv"],
                "n_eligible_syllables": n_eligible,
            }
        )

    cutoff_rows: list[dict[str, Any]] = []
    if n_eligible < min_eligible_syllables:
        return rank_rows, cutoff_rows

    for fraction in fractions:
        n_selected = cutoff_count(fraction, n_eligible)
        if n_selected < 1 or n_selected >= n_eligible:
            continue
        selected = candidates[:n_selected]
        remaining = candidates[n_selected:]
        row: dict[str, Any] = {
            "method": method,
            "direction": direction,
            "fraction": fraction,
            "bird": bird,
            "group": group,
            "n_eligible_syllables": n_eligible,
            "n_selected_syllables": n_selected,
            "n_remaining_syllables": len(remaining),
            "selected_labels": ",".join(str(item["syllable"]) for item in selected),
            "remaining_labels": ",".join(str(item["syllable"]) for item in remaining),
            "selected_min_rank_score": float(selected[-1]["rank_score"]),
        }
        for endpoint in ENDPOINTS:
            selected_value = safe_median(float(item[endpoint]) for item in selected)
            remaining_value = safe_median(float(item[endpoint]) for item in remaining)
            row[f"selected_{endpoint}"] = selected_value
            row[f"remaining_{endpoint}"] = remaining_value
            row[f"selected_minus_remaining_{endpoint}"] = (
                selected_value - remaining_value
                if math.isfinite(selected_value) and math.isfinite(remaining_value)
                else math.nan
            )
        cutoff_rows.append(row)

    return rank_rows, cutoff_rows


def build_cutoff_estimates(
    bird_data: Mapping[str, Mapping[str, Any]],
    full_metrics: Mapping[str, Mapping[str, PeriodMetrics]],
    fold_metrics: Mapping[tuple[str, str], Mapping[str, PeriodMetrics]],
    fractions: Sequence[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rank_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []

    for bird in sorted(bird_data):
        group = str(bird_data[bird]["group"])

        for method in ("full_pooled_variance", "full_pre_variance"):
            candidates = [
                {
                    "syllable": label,
                    "rank_score": rank_score(metrics, method),
                    "delta_sd": metrics.delta_sd,
                    "delta_cv": metrics.delta_cv,
                }
                for label, metrics in full_metrics.get(bird, {}).items()
            ]
            method_ranks, method_cutoffs = summarize_ranked_candidates(
                candidates,
                fractions,
                bird,
                group,
                method,
                "full_data",
                args.min_eligible_syllables,
            )
            rank_rows.extend(method_ranks)
            direction_rows.extend(method_cutoffs)

        for method in ("crossfit_pooled_variance", "crossfit_pre_variance"):
            for direction, screen_fold, test_fold in DIRECTIONS:
                common_labels = sorted(
                    set(fold_metrics.get((bird, screen_fold), {}))
                    & set(fold_metrics.get((bird, test_fold), {})),
                    key=label_sort_key,
                )
                candidates = []
                for label in common_labels:
                    screen = fold_metrics[(bird, screen_fold)][label]
                    heldout = fold_metrics[(bird, test_fold)][label]
                    candidates.append(
                        {
                            "syllable": label,
                            "rank_score": rank_score(screen, method),
                            "delta_sd": heldout.delta_sd,
                            "delta_cv": heldout.delta_cv,
                        }
                    )
                method_ranks, method_cutoffs = summarize_ranked_candidates(
                    candidates,
                    fractions,
                    bird,
                    group,
                    method,
                    direction,
                    args.min_eligible_syllables,
                )
                rank_rows.extend(method_ranks)
                direction_rows.extend(method_cutoffs)

    # Full-data rows already are final bird estimates. Cross-fitted rows are
    # averaged across reciprocal directions.
    bird_rows: list[dict[str, Any]] = []
    for row in direction_rows:
        if str(row["method"]).startswith("full_"):
            bird_rows.append(dict(row))

    grouped_crossfit: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in direction_rows:
        if str(row["method"]).startswith("crossfit_"):
            grouped_crossfit[(str(row["method"]), float(row["fraction"]), str(row["bird"]))].append(row)

    for (method, fraction, bird), rows in sorted(grouped_crossfit.items()):
        available_directions = sorted(str(row["direction"]) for row in rows)
        if not args.allow_one_crossfit_direction and len(rows) != len(DIRECTIONS):
            continue
        base = {
            "method": method,
            "direction": "reciprocal_mean",
            "fraction": fraction,
            "bird": bird,
            "group": rows[0]["group"],
            "n_directions_available": len(rows),
            "directions_available": ",".join(available_directions),
            "n_eligible_syllables": safe_mean(
                float(row["n_eligible_syllables"]) for row in rows
            ),
            "n_selected_syllables": safe_mean(
                float(row["n_selected_syllables"]) for row in rows
            ),
            "n_remaining_syllables": safe_mean(
                float(row["n_remaining_syllables"]) for row in rows
            ),
            "selected_labels": " | ".join(
                f"{row['direction']}:{row['selected_labels']}" for row in rows
            ),
            "remaining_labels": " | ".join(
                f"{row['direction']}:{row['remaining_labels']}" for row in rows
            ),
            "selected_min_rank_score": safe_mean(
                float(row["selected_min_rank_score"]) for row in rows
            ),
        }
        for endpoint in ENDPOINTS:
            for prefix in ("selected_", "remaining_", "selected_minus_remaining_"):
                key = f"{prefix}{endpoint}"
                base[key] = safe_mean(float(row[key]) for row in rows)
        bird_rows.append(base)

    return rank_rows, direction_rows, bird_rows


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------


def welch_t_statistic(group1: np.ndarray, group2: np.ndarray) -> float:
    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]
    if group1.size == 0 or group2.size == 0:
        return math.nan
    difference = float(np.mean(group1) - np.mean(group2))
    variance1 = float(np.var(group1, ddof=1)) if group1.size > 1 else 0.0
    variance2 = float(np.var(group2, ddof=1)) if group2.size > 1 else 0.0
    standard_error = math.sqrt(variance1 / group1.size + variance2 / group2.size)
    if standard_error <= 0 or not math.isfinite(standard_error):
        if difference > 0:
            return math.inf
        if difference < 0:
            return -math.inf
        return 0.0
    return difference / standard_error


def is_extreme(value: float, observed: float, alternative: str) -> bool:
    if alternative == "greater":
        return value >= observed - 1e-15
    if alternative == "less":
        return value <= observed + 1e-15
    if alternative == "two-sided":
        return abs(value) >= abs(observed) - 1e-15
    raise ValueError(f"Unknown alternative: {alternative}")


def pointwise_permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str,
    max_exact_assignments: int,
    monte_carlo_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]
    if group1.size == 0 or group2.size == 0:
        return math.nan, "not_tested", 0

    pooled = np.concatenate([group1, group2])
    n_group1 = group1.size
    observed = float(np.mean(group1) - np.mean(group2))
    n_assignments = math.comb(pooled.size, n_group1)

    if n_assignments <= max_exact_assignments:
        extreme = 0
        total = 0
        indices = np.arange(pooled.size)
        for chosen in combinations(indices, n_group1):
            mask = np.zeros(pooled.size, dtype=bool)
            mask[list(chosen)] = True
            permuted = float(np.mean(pooled[mask]) - np.mean(pooled[~mask]))
            extreme += int(is_extreme(permuted, observed, alternative))
            total += 1
        return extreme / total, "exact", total

    extreme = 0
    for _ in range(monte_carlo_permutations):
        permuted = rng.permutation(pooled)
        statistic = float(
            np.mean(permuted[:n_group1]) - np.mean(permuted[n_group1:])
        )
        extreme += int(is_extreme(statistic, observed, alternative))
    return (
        (extreme + 1) / (monte_carlo_permutations + 1),
        "monte_carlo",
        monte_carlo_permutations,
    )


def bootstrap_mean_difference_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]
    if group1.size == 0 or group2.size == 0 or replicates <= 0:
        return math.nan, math.nan
    differences = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample1 = group1[rng.integers(0, group1.size, size=group1.size)]
        sample2 = group2[rng.integers(0, group2.size, size=group2.size)]
        differences[index] = float(np.mean(sample1) - np.mean(sample2))
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite_indices = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    if not finite_indices:
        return adjusted
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def summary_column(summary_type: str, endpoint: str) -> str:
    if summary_type == "selected_median":
        return f"selected_{endpoint}"
    if summary_type == "selected_minus_remaining":
        return f"selected_minus_remaining_{endpoint}"
    raise ValueError(summary_type)


def group_values(
    rows: Sequence[dict[str, Any]], group: str, value_column: str
) -> np.ndarray:
    return finite_array(
        float(row[value_column]) for row in rows if str(row["group"]) == group
    )


def comparisons(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    return [
        ("medial_and_lateral_vs_sham", args.medial_group, args.sham_group),
        ("medial_and_lateral_vs_lateral_only", args.medial_group, args.lateral_group),
        ("lateral_only_vs_sham", args.lateral_group, args.sham_group),
    ]


def pointwise_group_tests(
    bird_rows: Sequence[dict[str, Any]],
    fractions: Sequence[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for method in METHODS:
        for endpoint in ENDPOINTS:
            for summary_type in SUMMARY_TYPES:
                value_column = summary_column(summary_type, endpoint)
                for fraction in fractions:
                    subset = [
                        row
                        for row in bird_rows
                        if str(row["method"]) == method
                        and abs(float(row["fraction"]) - fraction) < 1e-12
                    ]
                    block_indices: list[int] = []
                    for comparison_name, group1_name, group2_name in comparisons(args):
                        group1 = group_values(subset, group1_name, value_column)
                        group2 = group_values(subset, group2_name, value_column)
                        contrast = (
                            float(np.mean(group1) - np.mean(group2))
                            if group1.size and group2.size
                            else math.nan
                        )
                        ci_low, ci_high = bootstrap_mean_difference_ci(
                            group1,
                            group2,
                            args.bootstrap_replicates,
                            stable_rng(
                                args.seed,
                                "bootstrap",
                                method,
                                endpoint,
                                summary_type,
                                fraction,
                                comparison_name,
                            ),
                        )
                        p_greater, test_method, n_assignments = pointwise_permutation_test(
                            group1,
                            group2,
                            "greater",
                            args.max_exact_assignments,
                            args.monte_carlo_permutations,
                            stable_rng(
                                args.seed,
                                "pointwise_greater",
                                method,
                                endpoint,
                                summary_type,
                                fraction,
                                comparison_name,
                            ),
                        )
                        p_two_sided, _, _ = pointwise_permutation_test(
                            group1,
                            group2,
                            "two-sided",
                            args.max_exact_assignments,
                            args.monte_carlo_permutations,
                            stable_rng(
                                args.seed,
                                "pointwise_two_sided",
                                method,
                                endpoint,
                                summary_type,
                                fraction,
                                comparison_name,
                            ),
                        )
                        output.append(
                            {
                                "method": method,
                                "endpoint": endpoint,
                                "summary_type": summary_type,
                                "fraction": fraction,
                                "is_30_percent": abs(fraction - 0.30) < 1e-12,
                                "comparison": comparison_name,
                                "group1": group1_name,
                                "group2": group2_name,
                                "n_group1_birds": int(group1.size),
                                "n_group2_birds": int(group2.size),
                                "mean_group1": safe_mean(group1),
                                "median_group1": safe_median(group1),
                                "mean_group2": safe_mean(group2),
                                "median_group2": safe_median(group2),
                                "mean_difference_group1_minus_group2": contrast,
                                "bootstrap_95_ci_low": ci_low,
                                "bootstrap_95_ci_high": ci_high,
                                "welch_t": welch_t_statistic(group1, group2),
                                "one_sided_permutation_p": p_greater,
                                "two_sided_permutation_p": p_two_sided,
                                "permutation_method": test_method,
                                "n_permutations_or_assignments": n_assignments,
                                "one_sided_p_holm_within_fraction": math.nan,
                                "two_sided_p_holm_within_fraction": math.nan,
                            }
                        )
                        block_indices.append(len(output) - 1)

                    one_sided_adjusted = holm_adjust(
                        [float(output[index]["one_sided_permutation_p"]) for index in block_indices]
                    )
                    two_sided_adjusted = holm_adjust(
                        [float(output[index]["two_sided_permutation_p"]) for index in block_indices]
                    )
                    for local_index, output_index in enumerate(block_indices):
                        output[output_index]["one_sided_p_holm_within_fraction"] = one_sided_adjusted[local_index]
                        output[output_index]["two_sided_p_holm_within_fraction"] = two_sided_adjusted[local_index]

    return output


def global_threshold_test(
    bird_rows: Sequence[dict[str, Any]],
    fractions: Sequence[float],
    method: str,
    endpoint: str,
    summary_type: str,
    comparison_name: str,
    group1_name: str,
    group2_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    value_column = summary_column(summary_type, endpoint)
    rows = [row for row in bird_rows if str(row["method"]) == method]

    # A common complete-case bird set keeps the permutation group sizes and the
    # population of birds fixed across the entire cutoff scan.
    values_by_bird: dict[str, dict[float, float]] = defaultdict(dict)
    group_by_bird: dict[str, str] = {}
    for row in rows:
        bird = str(row["bird"])
        group = str(row["group"])
        if group not in {group1_name, group2_name}:
            continue
        value = float(row[value_column])
        if math.isfinite(value):
            values_by_bird[bird][float(row["fraction"])] = value
            group_by_bird[bird] = group

    complete_birds = sorted(
        bird
        for bird, values in values_by_bird.items()
        if all(fraction in values for fraction in fractions)
    )
    group1_birds = [bird for bird in complete_birds if group_by_bird[bird] == group1_name]
    group2_birds = [bird for bird in complete_birds if group_by_bird[bird] == group2_name]

    empty_result = {
        "method": method,
        "endpoint": endpoint,
        "summary_type": summary_type,
        "comparison": comparison_name,
        "group1": group1_name,
        "group2": group2_name,
        "n_complete_group1_birds": len(group1_birds),
        "n_complete_group2_birds": len(group2_birds),
        "complete_case_birds": ",".join(complete_birds),
        "n_cutoffs": len(fractions),
        "observed_max_t": math.nan,
        "observed_mean_t": math.nan,
        "max_t_global_p": math.nan,
        "mean_t_global_p": math.nan,
        "max_t_95_percent_null_critical": math.nan,
        "permutation_method": "not_tested",
        "n_permutations_or_assignments": 0,
        "max_t_global_p_holm": math.nan,
        "mean_t_global_p_holm": math.nan,
    }
    if not group1_birds or not group2_birds:
        return empty_result

    matrix = np.asarray(
        [[values_by_bird[bird][fraction] for fraction in fractions] for bird in complete_birds],
        dtype=float,
    )
    labels = np.asarray([group_by_bird[bird] for bird in complete_birds], dtype=object)
    n_group1 = len(group1_birds)
    observed_mask = labels == group1_name
    observed_t = np.asarray(
        [
            welch_t_statistic(matrix[observed_mask, index], matrix[~observed_mask, index])
            for index in range(len(fractions))
        ],
        dtype=float,
    )
    finite_observed = observed_t[np.isfinite(observed_t)]
    if finite_observed.size == 0:
        return empty_result
    observed_max = float(np.max(finite_observed))
    observed_mean = float(np.mean(finite_observed))

    n_assignments = math.comb(len(complete_birds), n_group1)
    max_null: list[float] = []
    mean_null: list[float] = []

    def evaluate_mask(mask: np.ndarray) -> None:
        t_values = np.asarray(
            [
                welch_t_statistic(matrix[mask, index], matrix[~mask, index])
                for index in range(len(fractions))
            ],
            dtype=float,
        )
        finite = t_values[np.isfinite(t_values)]
        if finite.size:
            max_null.append(float(np.max(finite)))
            mean_null.append(float(np.mean(finite)))

    if n_assignments <= args.max_exact_assignments:
        indices = np.arange(len(complete_birds))
        for chosen in combinations(indices, n_group1):
            mask = np.zeros(len(complete_birds), dtype=bool)
            mask[list(chosen)] = True
            evaluate_mask(mask)
        method_name = "exact"
        n_evaluated = len(max_null)
        max_p = sum(value >= observed_max - 1e-15 for value in max_null) / n_evaluated
        mean_p = sum(value >= observed_mean - 1e-15 for value in mean_null) / n_evaluated
    else:
        rng = stable_rng(
            args.seed,
            "global",
            method,
            endpoint,
            summary_type,
            comparison_name,
        )
        indices = np.arange(len(complete_birds))
        for _ in range(args.monte_carlo_permutations):
            chosen = rng.choice(indices, size=n_group1, replace=False)
            mask = np.zeros(len(complete_birds), dtype=bool)
            mask[chosen] = True
            evaluate_mask(mask)
        method_name = "monte_carlo"
        n_evaluated = len(max_null)
        max_p = (sum(value >= observed_max - 1e-15 for value in max_null) + 1) / (n_evaluated + 1)
        mean_p = (sum(value >= observed_mean - 1e-15 for value in mean_null) + 1) / (n_evaluated + 1)

    result = dict(empty_result)
    result.update(
        {
            "observed_max_t": observed_max,
            "observed_mean_t": observed_mean,
            "max_t_global_p": float(max_p),
            "mean_t_global_p": float(mean_p),
            "max_t_95_percent_null_critical": float(np.quantile(max_null, 0.95)),
            "permutation_method": method_name,
            "n_permutations_or_assignments": n_evaluated,
        }
    )
    return result


def global_group_tests(
    bird_rows: Sequence[dict[str, Any]],
    fractions: Sequence[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for endpoint in ENDPOINTS:
            for summary_type in SUMMARY_TYPES:
                block_indices: list[int] = []
                for comparison_name, group1_name, group2_name in comparisons(args):
                    rows.append(
                        global_threshold_test(
                            bird_rows,
                            fractions,
                            method,
                            endpoint,
                            summary_type,
                            comparison_name,
                            group1_name,
                            group2_name,
                            args,
                        )
                    )
                    block_indices.append(len(rows) - 1)

                max_adjusted = holm_adjust(
                    [float(rows[index]["max_t_global_p"]) for index in block_indices]
                )
                mean_adjusted = holm_adjust(
                    [float(rows[index]["mean_t_global_p"]) for index in block_indices]
                )
                for local_index, row_index in enumerate(block_indices):
                    rows[row_index]["max_t_global_p_holm"] = max_adjusted[local_index]
                    rows[row_index]["mean_t_global_p_holm"] = mean_adjusted[local_index]
    return rows


# -----------------------------------------------------------------------------
# Threshold diagnostics and figures
# -----------------------------------------------------------------------------


def threshold_support_rows(
    pointwise_rows: Sequence[dict[str, Any]],
    global_rows: Sequence[dict[str, Any]],
    fractions: Sequence[float],
) -> list[dict[str, Any]]:
    global_lookup = {
        (
            str(row["method"]),
            str(row["endpoint"]),
            str(row["summary_type"]),
            str(row["comparison"]),
        ): row
        for row in global_rows
    }
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(row["method"]),
                str(row["endpoint"]),
                str(row["summary_type"]),
                str(row["comparison"]),
            )
            for row in pointwise_rows
        }
    )
    for key in keys:
        subset = sorted(
            [
                row
                for row in pointwise_rows
                if (
                    str(row["method"]),
                    str(row["endpoint"]),
                    str(row["summary_type"]),
                    str(row["comparison"]),
                )
                == key
            ],
            key=lambda row: float(row["fraction"]),
        )
        row30 = next(
            (row for row in subset if abs(float(row["fraction"]) - 0.30) < 1e-12),
            None,
        )
        neighborhood = [
            row for row in subset if 0.20 - 1e-12 <= float(row["fraction"]) <= 0.40 + 1e-12
        ]
        effects = finite_array(
            float(row["mean_difference_group1_minus_group2"]) for row in subset
        )
        neighborhood_effects = finite_array(
            float(row["mean_difference_group1_minus_group2"]) for row in neighborhood
        )
        global_row = global_lookup.get(key, {})
        effect30 = (
            float(row30["mean_difference_group1_minus_group2"])
            if row30 is not None
            else math.nan
        )
        maximum_effect = float(np.max(effects)) if effects.size else math.nan
        output.append(
            {
                "method": key[0],
                "endpoint": key[1],
                "summary_type": key[2],
                "comparison": key[3],
                "effect_at_30_percent": effect30,
                "one_sided_p_at_30_percent": (
                    float(row30["one_sided_permutation_p"]) if row30 else math.nan
                ),
                "bootstrap_95_ci_low_at_30_percent": (
                    float(row30["bootstrap_95_ci_low"]) if row30 else math.nan
                ),
                "bootstrap_95_ci_high_at_30_percent": (
                    float(row30["bootstrap_95_ci_high"]) if row30 else math.nan
                ),
                "maximum_observed_effect_across_cutoffs": maximum_effect,
                "effect_30_divided_by_max_effect": (
                    effect30 / maximum_effect
                    if math.isfinite(effect30) and math.isfinite(maximum_effect) and maximum_effect > 0
                    else math.nan
                ),
                "proportion_of_cutoffs_with_positive_effect": (
                    float(np.mean(effects > 0)) if effects.size else math.nan
                ),
                "all_effects_positive_from_20_to_40_percent": (
                    bool(neighborhood_effects.size)
                    and bool(np.all(neighborhood_effects > 0))
                ),
                "minimum_effect_20_to_40_percent": (
                    float(np.min(neighborhood_effects))
                    if neighborhood_effects.size
                    else math.nan
                ),
                "maximum_effect_20_to_40_percent": (
                    float(np.max(neighborhood_effects))
                    if neighborhood_effects.size
                    else math.nan
                ),
                "max_t_global_p": global_row.get("max_t_global_p", math.nan),
                "mean_t_global_p": global_row.get("mean_t_global_p", math.nan),
                "max_t_global_p_holm": global_row.get("max_t_global_p_holm", math.nan),
                "mean_t_global_p_holm": global_row.get("mean_t_global_p_holm", math.nan),
            }
        )
    return output


def make_plots(
    pointwise_rows: Sequence[dict[str, Any]],
    global_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] matplotlib is unavailable; skipping plots.", file=sys.stderr)
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    global_lookup = {
        (
            str(row["method"]),
            str(row["endpoint"]),
            str(row["summary_type"]),
            str(row["comparison"]),
        ): row
        for row in global_rows
    }
    plotted_comparisons = (
        "medial_and_lateral_vs_sham",
        "medial_and_lateral_vs_lateral_only",
    )

    for method in METHODS:
        for endpoint in ENDPOINTS:
            for summary_type in SUMMARY_TYPES:
                fig, ax = plt.subplots(figsize=(8.0, 5.5))
                any_data = False
                for comparison_name in plotted_comparisons:
                    subset = sorted(
                        [
                            row
                            for row in pointwise_rows
                            if str(row["method"]) == method
                            and str(row["endpoint"]) == endpoint
                            and str(row["summary_type"]) == summary_type
                            and str(row["comparison"]) == comparison_name
                        ],
                        key=lambda row: float(row["fraction"]),
                    )
                    if not subset:
                        continue
                    x = np.asarray([100.0 * float(row["fraction"]) for row in subset])
                    y = np.asarray(
                        [float(row["mean_difference_group1_minus_group2"]) for row in subset]
                    )
                    low = np.asarray([float(row["bootstrap_95_ci_low"]) for row in subset])
                    high = np.asarray([float(row["bootstrap_95_ci_high"]) for row in subset])
                    valid = np.isfinite(x) & np.isfinite(y)
                    if not np.any(valid):
                        continue
                    any_data = True
                    label = comparison_name.replace("_", " ")
                    ax.plot(x[valid], y[valid], marker="o", label=label)
                    band_valid = valid & np.isfinite(low) & np.isfinite(high)
                    if np.any(band_valid):
                        ax.fill_between(x[band_valid], low[band_valid], high[band_valid], alpha=0.18)

                if not any_data:
                    plt.close(fig)
                    continue

                ax.axhline(0.0, linewidth=1.0, linestyle="--")
                ax.axvline(30.0, linewidth=1.0, linestyle=":")
                ax.set_xlabel("Top-ranked syllables retained (%)")
                y_label = (
                    "Group difference in bird-level median ΔSD (s)"
                    if endpoint == "delta_sd"
                    else "Group difference in bird-level median ΔCV"
                )
                if summary_type == "selected_minus_remaining":
                    y_label = "Group difference in selected-minus-remaining " + (
                        "ΔSD (s)" if endpoint == "delta_sd" else "ΔCV"
                    )
                ax.set_ylabel(y_label)
                ax.set_title(
                    f"{method.replace('_', ' ')}\n{summary_type.replace('_', ' ')}"
                )
                ax.legend(frameon=False, fontsize=8)

                annotation_lines = []
                for comparison_name in plotted_comparisons:
                    global_row = global_lookup.get(
                        (method, endpoint, summary_type, comparison_name)
                    )
                    if global_row and math.isfinite(float(global_row["max_t_global_p"])):
                        short_name = (
                            "M+L vs sham"
                            if comparison_name.endswith("vs_sham")
                            else "M+L vs lateral"
                        )
                        annotation_lines.append(
                            f"{short_name}: max-t p={float(global_row['max_t_global_p']):.4g}, "
                            f"mean-t p={float(global_row['mean_t_global_p']):.4g}"
                        )
                if annotation_lines:
                    ax.text(
                        0.01,
                        0.01,
                        "\n".join(annotation_lines),
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=8,
                    )
                fig.tight_layout()
                stem = f"cutoff_scan_{method}_{endpoint}_{summary_type}"
                fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
                fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
                plt.close(fig)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def serialize_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(value) for key, value in row.items()})


def write_summary(
    path: Path,
    args: argparse.Namespace,
    fractions: Sequence[float],
    audit_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
) -> None:
    primary = [
        row
        for row in support_rows
        if row["method"] in {"full_pooled_variance", "crossfit_pooled_variance", "crossfit_pre_variance"}
        and row["endpoint"] == "delta_cv"
        and row["summary_type"] == "selected_median"
        and row["comparison"] in {
            "medial_and_lateral_vs_sham",
            "medial_and_lateral_vs_lateral_only",
        }
    ]
    lines = [
        "Phrase-duration cutoff scan",
        "============================",
        "",
        f"Input directory: {args.json_dir}",
        f"Birds loaded: {len(audit_rows)}",
        f"Cutoffs: {', '.join(f'{100*fraction:.0f}%' for fraction in fractions)}",
        f"Pre window: days {args.pre_start_day} to {args.pre_end_day}",
        f"Post window: days {args.post_start_day} to {args.post_end_day}",
        f"Minimum phrases per period/fold: {args.min_phrases}",
        f"Minimum days per period/fold: {args.min_days}",
        f"Equal-count balancing draws: {args.balance_draws}",
        "",
        "Interpretation guide",
        "--------------------",
        "* full_pooled_variance most closely reproduces the current pooled-variance selection, but selection and estimation use the same data.",
        "* full_pre_variance avoids post-lesion selection but still uses the same pre observations for ranking and change estimation.",
        "* crossfit_pooled_variance separates selection from estimation using recording-day folds.",
        "* crossfit_pre_variance combines pre-only ranking with held-out estimation and is the most conservative design.",
        "* max-t p corrects for searching for the strongest cutoff.",
        "* mean-t p tests whether the contrast is consistently positive across the cutoff range.",
        "* selected-minus-remaining directly tests concentration in the upper-ranked subset.",
        "",
        "Key ΔCV selected-syllable diagnostics",
        "--------------------------------------",
    ]
    for row in primary:
        lines.extend(
            [
                f"{row['method']} | {row['comparison']}",
                f"  Effect at 30%: {row['effect_at_30_percent']}",
                f"  Pointwise one-sided p at 30%: {row['one_sided_p_at_30_percent']}",
                f"  All effects positive from 20-40%: {row['all_effects_positive_from_20_to_40_percent']}",
                f"  30% effect / maximum observed effect: {row['effect_30_divided_by_max_effect']}",
                f"  Global max-t p: {row['max_t_global_p']}",
                f"  Global mean-t p: {row['mean_t_global_p']}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    fractions = validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading JSON data from {args.json_dir}")
    bird_data, audit_rows, date_rows = load_bird_data(args)
    print(f"[INFO] Loaded {len(bird_data)} birds")

    full_metrics, fold_metrics, full_metric_rows, crossfit_metric_rows = build_syllable_metrics(
        bird_data, args
    )
    print("[INFO] Built balanced full-data and fold-specific syllable metrics")

    rank_rows, direction_rows, bird_rows = build_cutoff_estimates(
        bird_data,
        full_metrics,
        fold_metrics,
        fractions,
        args,
    )
    print(f"[INFO] Built {len(bird_rows)} bird-by-cutoff estimates")

    pointwise_rows = pointwise_group_tests(bird_rows, fractions, args)
    print("[INFO] Finished pointwise bird-level group tests")

    global_rows = global_group_tests(bird_rows, fractions, args)
    print("[INFO] Finished global max-t and mean-t cutoff tests")

    support_rows = threshold_support_rows(pointwise_rows, global_rows, fractions)

    config = {
        "json_dir": str(args.json_dir),
        "output": str(args.output),
        "fractions": fractions,
        "seed": args.seed,
        "pre_start_day": args.pre_start_day,
        "pre_end_day": args.pre_end_day,
        "post_start_day": args.post_start_day,
        "post_end_day": args.post_end_day,
        "min_phrases": args.min_phrases,
        "min_days": args.min_days,
        "min_eligible_syllables": args.min_eligible_syllables,
        "balance_draws": args.balance_draws,
        "bootstrap_replicates": args.bootstrap_replicates,
        "max_exact_assignments": args.max_exact_assignments,
        "monte_carlo_permutations": args.monte_carlo_permutations,
        "require_both_crossfit_directions": not args.allow_one_crossfit_direction,
        "groups": {
            "medial": args.medial_group,
            "lateral": args.lateral_group,
            "sham": args.sham_group,
        },
        "methods": METHODS,
        "endpoints": ENDPOINTS,
        "summary_types": SUMMARY_TYPES,
    }
    (args.output / "analysis_config.json").write_text(json.dumps(config, indent=2))

    write_csv(args.output / "input_audit.csv", audit_rows)
    write_csv(args.output / "date_fold_assignments.csv", date_rows)
    write_csv(args.output / "syllable_metrics_full.csv", full_metric_rows)
    write_csv(args.output / "syllable_metrics_crossfit.csv", crossfit_metric_rows)
    write_csv(args.output / "syllable_rankings.csv", rank_rows)
    write_csv(args.output / "bird_direction_cutoff_estimates.csv", direction_rows)
    write_csv(args.output / "bird_cutoff_estimates.csv", bird_rows)
    write_csv(args.output / "pointwise_group_contrasts.csv", pointwise_rows)
    write_csv(args.output / "global_threshold_tests.csv", global_rows)
    write_csv(args.output / "threshold_support_summary.csv", support_rows)
    write_summary(args.output / "summary.txt", args, fractions, audit_rows, support_rows)

    if not args.no_plots:
        make_plots(pointwise_rows, global_rows, args.output)

    print(f"[DONE] Results written to {args.output}")


if __name__ == "__main__":
    main()
