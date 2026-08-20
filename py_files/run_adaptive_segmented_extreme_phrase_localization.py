#!/usr/bin/env python3
"""Segmented, cross-fitted analysis of sparse extreme phrase prolongations.

Designed for the AFP lesion decoded-database JSON structure:

JSON_ROOT/
    AFP_lesion_bird_metadata.json
    <group>/<bird>/<bird>_decoded_database.json

Each decoded database must contain a top-level ``results`` list. Each result must
contain ``creation_date`` and ``syllable_onsets_offsets_ms``. Metadata must map
bird IDs to ``lesion_group`` and ``lesion_surgery_date``.

Primary design
--------------
* Pre-lesion selection is availability-adaptive. Up to the final 14 singing
  days before surgery are retained. Birds with at least 8 selected pre-lesion
  singing days are split chronologically into early pre and late pre; when the
  number is odd, the extra day is assigned to early pre. Thus 9 available days
  become 5 early-pre and 4 late-pre days.
* Early post: calendar days +1 through +14.
* Late post: calendar days +15 through +28.
* Threshold: a syllable-specific early-pre duration quantile (default 97.5%).
* Daily tail burden: excess seconds above threshold per 100 occurrences,
  calculated separately for every recording day so days receive equal weight.
* Localization: reciprocal A/B cross-fitting. Late-pre dates are split across
  folds, and post dates are split within 7-day post blocks so both folds span
  early and late post-lesion time. Syllables are ranked in the screening fold;
  the top k are evaluated only in the held-out fold. Negative held-out values
  are retained.
* Bird-level inference: exact bird-label permutations, with the bird as the
  independent experimental unit.

The script writes detailed CSV files for period definitions, daily metrics,
all-syllable summaries, cross-fitted top-k localization, time-resolved effects,
group tests, within-group sign-flip tests, and sensitivity analyses.

Example (macOS)
---------------
python run_segmented_extreme_phrase_localization.py \
  "$HOME/Desktop/AFP_lesion_jsons" \
  --output "$HOME/Desktop/segmented_extreme_phrase_results" \
  --max-pre-singing-days 14 \
  --min-total-pre-singing-days 8 \
  --post-end-day 28 \
  --primary-quantile 0.975 \
  --primary-top-k 3 \
  --seed 123

Required package: numpy
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
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DIRECTIONS = (
    ("A_screen_B_test", "A", "B"),
    ("B_screen_A_test", "B", "A"),
)

POST_BLOCKS_PRIMARY = (
    ("post_days_1_7", 1, 7),
    ("post_days_8_14", 8, 14),
    ("post_days_15_21", 15, 21),
    ("post_days_22_28", 22, 28),
)


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run segmented, cross-fitted localization of rare extreme phrase "
            "prolongations with recording days weighted equally."
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
        default=Path("segmented_extreme_phrase_results"),
    )
    parser.add_argument("--seed", type=int, default=123)

    # Bird-level singing-day windows before surgery.
    parser.add_argument(
        "--max-pre-singing-days",
        type=int,
        default=14,
        help=(
            "Maximum number of most-recent pre-lesion singing days retained "
            "for the availability-adaptive primary analysis."
        ),
    )
    parser.add_argument(
        "--min-total-pre-singing-days",
        type=int,
        default=8,
        help=(
            "Minimum total selected pre-lesion singing days required for the "
            "adaptive early/late split. Eight days yield four early and four late."
        ),
    )
    # Fixed-window settings are retained for sensitivity analyses and backwards
    # compatibility. They do not control the adaptive primary split.
    parser.add_argument("--early-pre-singing-days", type=int, default=7)
    parser.add_argument("--late-pre-singing-days", type=int, default=7)
    parser.add_argument(
        "--min-early-pre-singing-days",
        type=int,
        default=2,
        help="Minimum early-pre days for fixed-window sensitivity specifications.",
    )
    parser.add_argument(
        "--min-late-pre-singing-days",
        type=int,
        default=4,
        help="Minimum late-pre days for fixed-window sensitivity specifications.",
    )

    # Post windows in calendar days relative to surgery.
    parser.add_argument("--post-start-day", type=int, default=1)
    parser.add_argument("--early-post-end-day", type=int, default=14)
    parser.add_argument("--late-post-start-day", type=int, default=15)
    parser.add_argument(
        "--post-end-day",
        type=int,
        default=28,
        help="Primary last included post-treatment calendar day.",
    )

    # Threshold and top-k localization.
    parser.add_argument("--primary-quantile", type=float, default=0.975)
    parser.add_argument("--primary-top-k", type=int, default=3)
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values reported under the primary period/threshold specification.",
    )

    # Syllable eligibility.
    parser.add_argument("--min-early-pre-phrases", type=int, default=50)
    parser.add_argument("--min-late-pre-phrases", type=int, default=30)
    parser.add_argument("--min-post-phrases", type=int, default=30)
    parser.add_argument("--min-late-pre-days-per-syllable", type=int, default=3)
    parser.add_argument("--min-post-days-per-syllable", type=int, default=4)
    parser.add_argument("--min-fold-phrases", type=int, default=15)
    parser.add_argument("--min-fold-days", type=int, default=2)
    parser.add_argument(
        "--require-both-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require both reciprocal cross-fitting directions for primary bird endpoints.",
    )

    # Group labels and inference.
    parser.add_argument("--medial-group", default="medial_and_lateral")
    parser.add_argument("--lateral-group", default="lateral_only")
    parser.add_argument("--sham-group", default="sham_saline")
    parser.add_argument(
        "--group-statistic",
        choices=("mean", "median"),
        default="mean",
    )
    parser.add_argument("--max-exact-group-assignments", type=int, default=200_000)
    parser.add_argument("--group-permutations", type=int, default=200_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)

    # Optional descriptive day-level p-values for selected syllables.
    parser.add_argument(
        "--selected-syllable-day-permutations",
        type=int,
        default=0,
        help=(
            "Optional descriptive held-out day permutations for selected syllables; "
            "0 skips them because they can be computationally expensive."
        ),
    )
    parser.add_argument("--max-exact-day-assignments", type=int, default=100_000)

    # Sensitivity analyses.
    parser.add_argument(
        "--sensitivity-quantiles",
        type=float,
        nargs="+",
        default=[0.95, 0.975, 0.99],
    )
    parser.add_argument(
        "--sensitivity-pre-singing-days",
        type=int,
        nargs="+",
        default=[5, 7, 10],
    )
    parser.add_argument(
        "--sensitivity-post-end-days",
        type=int,
        nargs="+",
        default=[21, 28],
    )
    parser.add_argument(
        "--sensitivity-split-seeds",
        type=int,
        default=5,
        help="Number of additional deterministic fold seeds.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.json_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.json_dir}")
    positive_ints = (
        "max_pre_singing_days",
        "min_total_pre_singing_days",
        "early_pre_singing_days",
        "late_pre_singing_days",
        "min_early_pre_singing_days",
        "min_late_pre_singing_days",
        "primary_top_k",
        "min_early_pre_phrases",
        "min_late_pre_phrases",
        "min_post_phrases",
        "min_late_pre_days_per_syllable",
        "min_post_days_per_syllable",
        "min_fold_phrases",
        "min_fold_days",
        "max_exact_group_assignments",
        "group_permutations",
        "bootstrap_replicates",
        "max_exact_day_assignments",
    )
    for name in positive_ints:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.min_total_pre_singing_days > args.max_pre_singing_days:
        raise ValueError(
            "--min-total-pre-singing-days cannot exceed --max-pre-singing-days"
        )
    if args.min_total_pre_singing_days < 2 * args.min_fold_days:
        raise ValueError(
            "--min-total-pre-singing-days must allow at least min-fold-days in "
            "both early and late halves."
        )
    if args.post_start_day < 1:
        raise ValueError("--post-start-day must be >= 1")
    if not (
        args.post_start_day
        <= args.early_post_end_day
        < args.late_post_start_day
        <= args.post_end_day
    ):
        raise ValueError(
            "Require post_start <= early_post_end < late_post_start <= post_end."
        )
    quantiles = [args.primary_quantile, *args.sensitivity_quantiles]
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("All quantiles must lie strictly between 0 and 1")
    if any(k < 1 for k in [args.primary_top_k, *args.top_k_values]):
        raise ValueError("All top-k values must be >= 1")
    if args.selected_syllable_day_permutations < 0:
        raise ValueError("--selected-syllable-day-permutations must be >= 0")
    if args.sensitivity_split_seeds < 0:
        raise ValueError("--sensitivity-split-seeds must be >= 0")


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def parse_iso_date(value: str) -> date:
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


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


def center(values: Sequence[float], statistic: str) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return math.nan
    if statistic == "mean":
        return float(np.mean(arr))
    if statistic == "median":
        return float(np.median(arr))
    raise ValueError(statistic)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return out.tolist()
    ordered = valid[np.argsort(p[valid])]
    m = len(ordered)
    running = 0.0
    for rank, index in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * p[index])
        running = max(running, adjusted)
        out[index] = running
    return out.tolist()


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


# -----------------------------------------------------------------------------
# Data structures and loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    quantile: float
    pre_mode: str
    max_pre_singing_days: int
    min_total_pre_singing_days: int
    early_pre_singing_days: int
    late_pre_singing_days: int
    post_end_day: int | None
    split_seed: int
    remove_longest_post_per_syllable: bool = False
    min_early_pre_phrases_override: int | None = None
    min_early_pre_days_override: int | None = None
    min_late_pre_days_override: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.name,
            "quantile": self.quantile,
            "pre_mode": self.pre_mode,
            "max_pre_singing_days": self.max_pre_singing_days,
            "min_total_pre_singing_days": self.min_total_pre_singing_days,
            "early_pre_singing_days_requested": self.early_pre_singing_days,
            "late_pre_singing_days_requested": self.late_pre_singing_days,
            "post_end_day": self.post_end_day,
            "split_seed": self.split_seed,
            "remove_longest_post_per_syllable": self.remove_longest_post_per_syllable,
            "min_early_pre_phrases_override": self.min_early_pre_phrases_override,
            "min_early_pre_days_override": self.min_early_pre_days_override,
            "min_late_pre_days_override": self.min_late_pre_days_override,
        }


@dataclass(frozen=True)
class PeriodSummary:
    n_days: int
    n_phrases: int
    n_extreme: int
    n_extreme_days: int
    pooled_event_rate: float
    pooled_burden_seconds_per_100: float
    mean_daily_event_rate_per_100: float
    median_daily_event_rate_per_100: float
    mean_daily_burden_seconds_per_100: float
    median_daily_burden_seconds_per_100: float
    conditional_excess_median_seconds: float
    conditional_excess_mean_seconds: float
    max_duration_seconds: float
    max_excess_seconds: float
    mean_duration_seconds: float
    sd_duration_seconds: float
    cv_duration: float

    def as_dict(self, prefix: str) -> dict[str, Any]:
        return {f"{prefix}{name}": value for name, value in self.__dict__.items()}


def load_data(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = args.json_dir / "AFP_lesion_bird_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    database_paths = sorted(args.json_dir.glob("*/*/*decoded_database.json"))
    if not database_paths:
        raise FileNotFoundError(
            f"No files matched */*/*decoded_database.json under {args.json_dir}"
        )

    bird_data: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for path in database_paths:
        bird = path.stem.removesuffix("_decoded_database")
        if bird not in metadata:
            raise KeyError(f"Bird {bird!r} is absent from {metadata_path}")
        group = str(metadata[bird]["lesion_group"])
        surgery = parse_iso_date(metadata[bird]["lesion_surgery_date"])
        durations: dict[str, dict[date, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        n_results = 0
        n_spans = 0
        for result in json.loads(path.read_text()).get("results", []):
            n_results += 1
            recording_date = parse_iso_date(result["creation_date"])
            for label, spans in result.get("syllable_onsets_offsets_ms", {}).items():
                for start, end in spans:
                    duration = (float(end) - float(start)) / 1000.0
                    if math.isfinite(duration) and duration > 0:
                        durations[str(label)][recording_date].append(duration)
                        n_spans += 1
        all_dates = sorted({d for label_map in durations.values() for d in label_map})
        bird_data[bird] = {
            "group": group,
            "surgery_date": surgery,
            "durations": durations,
            "all_dates": all_dates,
            "path": str(path),
        }
        audit_rows.append(
            {
                "bird": bird,
                "group": group,
                "surgery_date": surgery.isoformat(),
                "database_path": str(path),
                "n_results": n_results,
                "n_valid_phrase_spans": n_spans,
                "n_labels": len(durations),
                "n_recording_dates": len(all_dates),
                "first_recording_date": all_dates[0].isoformat() if all_dates else "",
                "last_recording_date": all_dates[-1].isoformat() if all_dates else "",
            }
        )
    return bird_data, audit_rows


# -----------------------------------------------------------------------------
# Period definitions and fold assignment
# -----------------------------------------------------------------------------


def choose_period_dates(
    info: Mapping[str, Any], spec: AnalysisSpec, args: argparse.Namespace
) -> dict[str, list[date]]:
    surgery: date = info["surgery_date"]
    all_dates: list[date] = info["all_dates"]
    pre_dates_all = sorted(d for d in all_dates if (d - surgery).days <= -1)

    if spec.pre_mode == "adaptive_segmented":
        selected_pre = pre_dates_all[-spec.max_pre_singing_days :]
        n_selected = len(selected_pre)
        # Earlier half receives the extra day when n is odd.
        n_early = (n_selected + 1) // 2
        early_pre = selected_pre[:n_early]
        late_pre = selected_pre[n_early:]
        pooled_pre = selected_pre
    elif spec.pre_mode == "fixed_segmented":
        late_pre = pre_dates_all[-spec.late_pre_singing_days :]
        remaining = pre_dates_all[: max(0, len(pre_dates_all) - len(late_pre))]
        early_pre = remaining[-spec.early_pre_singing_days :]
        pooled_pre = early_pre + late_pre
    elif spec.pre_mode == "pooled_crossfit":
        pooled_pre = pre_dates_all[-spec.max_pre_singing_days :]
        early_pre = []
        late_pre = []
    else:
        raise ValueError(f"Unsupported pre_mode: {spec.pre_mode}")

    post_dates = sorted(
        d
        for d in all_dates
        if (d - surgery).days >= args.post_start_day
        and (spec.post_end_day is None or (d - surgery).days <= spec.post_end_day)
    )
    early_post = [
        d
        for d in post_dates
        if args.post_start_day <= (d - surgery).days <= args.early_post_end_day
    ]
    late_post = [
        d
        for d in post_dates
        if (d - surgery).days >= args.late_post_start_day
    ]
    blocks: dict[str, list[date]] = {}
    for name, lo, hi in POST_BLOCKS_PRIMARY:
        blocks[name] = [d for d in post_dates if lo <= (d - surgery).days <= hi]
    if spec.post_end_day is None or spec.post_end_day > 28:
        blocks["post_days_29_plus"] = [
            d for d in post_dates if (d - surgery).days >= 29
        ]
    return {
        "pre": pooled_pre,
        "early_pre": early_pre,
        "late_pre": late_pre,
        "post": post_dates,
        "early_post": early_post,
        "late_post": late_post,
        **blocks,
    }


def split_adjacent_dates(
    dates: Sequence[date], seed: int, *parts: Any
) -> dict[date, str]:
    ordered = sorted(set(dates))
    rng = stable_rng(seed, "date_split", *parts)
    assignment: dict[date, str] = {}
    counts = {"A": 0, "B": 0}
    paired_end = len(ordered) - len(ordered) % 2
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


def build_fold_assignment(
    bird: str,
    period_dates: Mapping[str, Sequence[date]],
    spec: AnalysisSpec,
) -> dict[str, dict[date, str]]:
    if spec.pre_mode == "pooled_crossfit":
        pre_assignment = split_adjacent_dates(
            period_dates["pre"], spec.split_seed, bird, "pooled_pre"
        )
        late_pre_assignment: dict[date, str] = {}
    else:
        late_pre_assignment = split_adjacent_dates(
            period_dates["late_pre"], spec.split_seed, bird, "late_pre"
        )
        pre_assignment = {}

    post_assignment: dict[date, str] = {}
    global_counts = {"A": 0, "B": 0}
    ordered_blocks = [name for name, _, _ in POST_BLOCKS_PRIMARY]
    if "post_days_29_plus" in period_dates:
        ordered_blocks.append("post_days_29_plus")
    for block_name in ordered_blocks:
        dates = sorted(set(period_dates.get(block_name, [])))
        block_assignment = split_adjacent_dates(
            dates, spec.split_seed, bird, block_name
        )
        # Rebalance singleton blocks so successive one-date blocks do not all
        # land in the same fold.
        if len(dates) == 1:
            d = dates[0]
            if global_counts["A"] < global_counts["B"]:
                block_assignment[d] = "A"
            elif global_counts["B"] < global_counts["A"]:
                block_assignment[d] = "B"
        for d, fold in block_assignment.items():
            post_assignment[d] = fold
            global_counts[fold] += 1
    # Any post dates not represented by a named block (possible custom windows).
    missing = [d for d in period_dates["post"] if d not in post_assignment]
    if missing:
        extra = split_adjacent_dates(missing, spec.split_seed, bird, "post_extra")
        post_assignment.update(extra)
    return {
        "pre": pre_assignment,
        "late_pre": late_pre_assignment,
        "post": post_assignment,
    }


def period_date_rows(
    bird: str,
    group: str,
    surgery: date,
    period_dates: Mapping[str, Sequence[date]],
    folds: Mapping[str, Mapping[date, str]],
    spec: AnalysisSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if spec.pre_mode == "pooled_crossfit":
        periods = ("pre", "early_post", "late_post")
    else:
        periods = ("early_pre", "late_pre", "early_post", "late_post")
    for period in periods:
        for d in period_dates.get(period, []):
            fold = ""
            if period == "pre":
                fold = folds["pre"].get(d, "")
            elif period == "late_pre":
                fold = folds["late_pre"].get(d, "")
            elif "post" in period:
                fold = folds["post"].get(d, "")
            rows.append(
                {
                    **spec.as_dict(),
                    "bird": bird,
                    "group": group,
                    "period": period,
                    "date": d.isoformat(),
                    "relative_day": (d - surgery).days,
                    "fold": fold,
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Tail metrics
# -----------------------------------------------------------------------------


def clean_day_map(
    label_dates: Mapping[date, Sequence[float]], allowed_dates: Sequence[date]
) -> dict[date, list[float]]:
    allowed = set(allowed_dates)
    out: dict[date, list[float]] = {}
    for d, values in label_dates.items():
        if d not in allowed:
            continue
        clean = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0]
        if clean:
            out[d] = clean
    return dict(sorted(out.items()))


def filter_fold(
    day_map: Mapping[date, Sequence[float]],
    assignment: Mapping[date, str],
    fold: str,
) -> dict[date, list[float]]:
    return {
        d: list(map(float, values))
        for d, values in day_map.items()
        if assignment.get(d) == fold
    }


def flatten_day_map(day_map: Mapping[date, Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(values, dtype=float) for values in day_map.values() if values]
    if not arrays:
        return np.asarray([], dtype=float)
    arr = np.concatenate(arrays)
    return arr[np.isfinite(arr) & (arr > 0)]


def remove_longest(day_map: Mapping[date, Sequence[float]]) -> dict[date, list[float]]:
    copied = {d: list(map(float, values)) for d, values in day_map.items()}
    best: tuple[date, int] | None = None
    best_value = -math.inf
    for d, values in copied.items():
        for index, value in enumerate(values):
            if value > best_value:
                best_value = value
                best = (d, index)
    if best is not None:
        d, index = best
        copied[d].pop(index)
        if not copied[d]:
            del copied[d]
    return copied


def daily_tail_row(values: Sequence[float], threshold: float) -> dict[str, float | int]:
    arr = finite_array(values)
    arr = arr[arr > 0]
    if arr.size == 0 or not math.isfinite(threshold):
        return {
            "n_phrases": int(arr.size),
            "n_extreme": 0,
            "event_rate": math.nan,
            "event_rate_per_100": math.nan,
            "burden_seconds_per_100": math.nan,
            "conditional_excess_median_seconds": math.nan,
            "conditional_excess_mean_seconds": math.nan,
            "max_duration_seconds": math.nan,
            "max_excess_seconds": math.nan,
        }
    excess = np.maximum(0.0, arr - threshold)
    positive = excess[excess > 0]
    return {
        "n_phrases": int(arr.size),
        "n_extreme": int(positive.size),
        "event_rate": float(positive.size / arr.size),
        "event_rate_per_100": float(100.0 * positive.size / arr.size),
        "burden_seconds_per_100": float(100.0 * np.sum(excess) / arr.size),
        "conditional_excess_median_seconds": (
            float(np.median(positive)) if positive.size else math.nan
        ),
        "conditional_excess_mean_seconds": (
            float(np.mean(positive)) if positive.size else math.nan
        ),
        "max_duration_seconds": float(np.max(arr)),
        "max_excess_seconds": float(np.max(excess)),
    }


def summarize_period(
    day_map: Mapping[date, Sequence[float]], threshold: float
) -> PeriodSummary:
    values = flatten_day_map(day_map)
    daily = [daily_tail_row(v, threshold) for _, v in sorted(day_map.items())]
    if values.size == 0 or not math.isfinite(threshold):
        return PeriodSummary(
            n_days=len(day_map),
            n_phrases=int(values.size),
            n_extreme=0,
            n_extreme_days=0,
            pooled_event_rate=math.nan,
            pooled_burden_seconds_per_100=math.nan,
            mean_daily_event_rate_per_100=math.nan,
            median_daily_event_rate_per_100=math.nan,
            mean_daily_burden_seconds_per_100=math.nan,
            median_daily_burden_seconds_per_100=math.nan,
            conditional_excess_median_seconds=math.nan,
            conditional_excess_mean_seconds=math.nan,
            max_duration_seconds=math.nan,
            max_excess_seconds=math.nan,
            mean_duration_seconds=math.nan,
            sd_duration_seconds=math.nan,
            cv_duration=math.nan,
        )
    excess = np.maximum(0.0, values - threshold)
    positive = excess[excess > 0]
    daily_event = [finite_or_nan(row["event_rate_per_100"]) for row in daily]
    daily_burden = [finite_or_nan(row["burden_seconds_per_100"]) for row in daily]
    mean_duration = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size >= 2 else math.nan
    return PeriodSummary(
        n_days=len(day_map),
        n_phrases=int(values.size),
        n_extreme=int(positive.size),
        n_extreme_days=int(sum(int(row["n_extreme"]) > 0 for row in daily)),
        pooled_event_rate=float(positive.size / values.size),
        pooled_burden_seconds_per_100=float(100.0 * np.sum(excess) / values.size),
        mean_daily_event_rate_per_100=mean_or_nan(daily_event),
        median_daily_event_rate_per_100=median_or_nan(daily_event),
        mean_daily_burden_seconds_per_100=mean_or_nan(daily_burden),
        median_daily_burden_seconds_per_100=median_or_nan(daily_burden),
        conditional_excess_median_seconds=(
            float(np.median(positive)) if positive.size else math.nan
        ),
        conditional_excess_mean_seconds=(
            float(np.mean(positive)) if positive.size else math.nan
        ),
        max_duration_seconds=float(np.max(values)),
        max_excess_seconds=float(np.max(excess)),
        mean_duration_seconds=mean_duration,
        sd_duration_seconds=sd,
        cv_duration=(sd / mean_duration if mean_duration > 0 and math.isfinite(sd) else math.nan),
    )


def delta(post: PeriodSummary, pre: PeriodSummary, field: str) -> float:
    return finite_or_nan(getattr(post, field)) - finite_or_nan(getattr(pre, field))


def eligible_full(
    early: PeriodSummary,
    late: PeriodSummary,
    post: PeriodSummary,
    args: argparse.Namespace,
) -> bool:
    return bool(
        early.n_phrases >= args.min_early_pre_phrases
        and late.n_phrases >= args.min_late_pre_phrases
        and post.n_phrases >= args.min_post_phrases
        and late.n_days >= args.min_late_pre_days_per_syllable
        and post.n_days >= args.min_post_days_per_syllable
    )


def eligible_fold(
    late: PeriodSummary, post: PeriodSummary, args: argparse.Namespace
) -> bool:
    return bool(
        late.n_phrases >= args.min_fold_phrases
        and post.n_phrases >= args.min_fold_phrases
        and late.n_days >= args.min_fold_days
        and post.n_days >= args.min_fold_days
    )


# -----------------------------------------------------------------------------
# Permutation and bootstrap helpers
# -----------------------------------------------------------------------------


def group_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    statistic: str,
    alternative: str,
    max_exact: int,
    monte_carlo: int,
    rng: np.random.Generator,
) -> tuple[float, float, str, int]:
    x_arr = finite_array(x)
    y_arr = finite_array(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return math.nan, math.nan, "not_tested", 0
    pooled = np.concatenate([x_arr, y_arr])
    observed = center(x_arr, statistic) - center(y_arr, statistic)
    n_x = len(x_arr)
    n_assignments = math.comb(len(pooled), n_x)

    def extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - 1e-15
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - 1e-15
        raise ValueError(alternative)

    if n_assignments <= max_exact:
        count = 0
        total = 0
        for chosen in combinations(range(len(pooled)), n_x):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(chosen)] = True
            value = center(pooled[mask], statistic) - center(pooled[~mask], statistic)
            count += int(extreme(value))
            total += 1
        return observed, count / total, "exact", total

    count = 0
    for _ in range(monte_carlo):
        perm = rng.permutation(pooled)
        value = center(perm[:n_x], statistic) - center(perm[n_x:], statistic)
        count += int(extreme(value))
    return observed, (count + 1) / (monte_carlo + 1), "monte_carlo", monte_carlo


def exact_signflip(values: Sequence[float], alternative: str = "greater") -> tuple[float, float, int]:
    arr = finite_array(values)
    if arr.size == 0:
        return math.nan, math.nan, 0
    observed = float(np.mean(arr))
    count = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(arr)):
        value = float(np.mean(arr * np.asarray(signs)))
        if alternative == "greater":
            count += int(value >= observed - 1e-15)
        else:
            count += int(abs(value) >= abs(observed) - 1e-15)
        total += 1
    return observed, count / total, total


def bootstrap_mean_ci(
    values: Sequence[float], replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return math.nan, math.nan
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
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


def day_permutation_p(
    late_map: Mapping[date, Sequence[float]],
    post_map: Mapping[date, Sequence[float]],
    threshold: float,
    max_exact: int,
    monte_carlo: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    late_days = sorted(late_map)
    post_days = sorted(post_map)
    all_days = late_days + post_days
    n_late = len(late_days)
    if n_late < 1 or len(post_days) < 1:
        return math.nan, "not_tested", 0
    daily_values = {**{d: late_map[d] for d in late_days}, **{d: post_map[d] for d in post_days}}

    def statistic(chosen_indices: Sequence[int]) -> float:
        chosen = set(chosen_indices)
        late = {d: daily_values[d] for i, d in enumerate(all_days) if i in chosen}
        post = {d: daily_values[d] for i, d in enumerate(all_days) if i not in chosen}
        return delta(
            summarize_period(post, threshold),
            summarize_period(late, threshold),
            "mean_daily_burden_seconds_per_100",
        )

    observed = delta(
        summarize_period(post_map, threshold),
        summarize_period(late_map, threshold),
        "mean_daily_burden_seconds_per_100",
    )
    n_assignments = math.comb(len(all_days), n_late)
    if n_assignments <= max_exact:
        count = 0
        total = 0
        for chosen in combinations(range(len(all_days)), n_late):
            value = statistic(chosen)
            if math.isfinite(value):
                count += int(value >= observed - 1e-15)
                total += 1
        return (count / total if total else math.nan), "exact", total
    count = 0
    valid = 0
    for _ in range(monte_carlo):
        chosen = rng.choice(len(all_days), size=n_late, replace=False)
        value = statistic(chosen)
        if math.isfinite(value):
            count += int(value >= observed - 1e-15)
            valid += 1
    return ((count + 1) / (valid + 1) if valid else math.nan), "monte_carlo", valid


# -----------------------------------------------------------------------------
# One analysis specification
# -----------------------------------------------------------------------------


def analyze_bird(
    bird: str,
    info: Mapping[str, Any],
    spec: AnalysisSpec,
    top_k_values: Sequence[int],
    args: argparse.Namespace,
    detailed: bool,
) -> dict[str, Any]:
    group = str(info["group"])
    surgery: date = info["surgery_date"]
    durations: Mapping[str, Mapping[date, Sequence[float]]] = info["durations"]
    period_dates = choose_period_dates(info, spec, args)
    folds = build_fold_assignment(bird, period_dates, spec)

    min_early_days = (
        spec.min_early_pre_days_override
        if spec.min_early_pre_days_override is not None
        else args.min_early_pre_singing_days
    )
    min_late_days = (
        spec.min_late_pre_days_override
        if spec.min_late_pre_days_override is not None
        else args.min_late_pre_singing_days
    )
    min_early_phrases = (
        spec.min_early_pre_phrases_override
        if spec.min_early_pre_phrases_override is not None
        else args.min_early_pre_phrases
    )

    if spec.pre_mode == "adaptive_segmented":
        bird_pre_eligible = bool(
            len(period_dates["pre"]) >= spec.min_total_pre_singing_days
            and len(period_dates["early_pre"]) >= 1
            and len(period_dates["late_pre"]) >= 2 * args.min_fold_days
        )
    elif spec.pre_mode == "fixed_segmented":
        bird_pre_eligible = bool(
            len(period_dates["early_pre"]) >= min_early_days
            and len(period_dates["late_pre"]) >= min_late_days
            and len(period_dates["late_pre"]) >= 2 * args.min_fold_days
        )
    elif spec.pre_mode == "pooled_crossfit":
        pre_counts = {
            fold: sum(1 for d in period_dates["pre"] if folds["pre"].get(d) == fold)
            for fold in ("A", "B")
        }
        bird_pre_eligible = bool(
            len(period_dates["pre"]) >= spec.min_total_pre_singing_days
            and all(count >= args.min_fold_days for count in pre_counts.values())
        )
    else:
        raise ValueError(spec.pre_mode)

    all_pre_dates = sorted(
        d for d in info["all_dates"] if (d - surgery).days <= -1
    )
    selected_pre_dates = period_dates["pre"]
    pre_allocation_row = {
        **spec.as_dict(),
        "bird": bird,
        "group": group,
        "n_all_pre_singing_days_available": len(all_pre_dates),
        "n_selected_pre_singing_days": len(selected_pre_dates),
        "n_early_pre_singing_days": len(period_dates["early_pre"]),
        "n_late_pre_singing_days": len(period_dates["late_pre"]),
        "n_pre_fold_A_days": sum(
            1 for d in period_dates["pre"] if folds["pre"].get(d) == "A"
        ),
        "n_pre_fold_B_days": sum(
            1 for d in period_dates["pre"] if folds["pre"].get(d) == "B"
        ),
        "n_late_pre_fold_A_days": sum(
            1 for d in period_dates["late_pre"] if folds["late_pre"].get(d) == "A"
        ),
        "n_late_pre_fold_B_days": sum(
            1 for d in period_dates["late_pre"] if folds["late_pre"].get(d) == "B"
        ),
        "n_post_fold_A_days": sum(
            1 for d in period_dates["post"] if folds["post"].get(d) == "A"
        ),
        "n_post_fold_B_days": sum(
            1 for d in period_dates["post"] if folds["post"].get(d) == "B"
        ),
        "selected_pre_first_relative_day": (
            (selected_pre_dates[0] - surgery).days if selected_pre_dates else ""
        ),
        "selected_pre_last_relative_day": (
            (selected_pre_dates[-1] - surgery).days if selected_pre_dates else ""
        ),
        "early_pre_first_relative_day": (
            (period_dates["early_pre"][0] - surgery).days
            if period_dates["early_pre"]
            else ""
        ),
        "early_pre_last_relative_day": (
            (period_dates["early_pre"][-1] - surgery).days
            if period_dates["early_pre"]
            else ""
        ),
        "late_pre_first_relative_day": (
            (period_dates["late_pre"][0] - surgery).days
            if period_dates["late_pre"]
            else ""
        ),
        "late_pre_last_relative_day": (
            (period_dates["late_pre"][-1] - surgery).days
            if period_dates["late_pre"]
            else ""
        ),
        "bird_pre_window_eligible": bird_pre_eligible,
    }

    date_rows = (
        period_date_rows(bird, group, surgery, period_dates, folds, spec)
        if detailed
        else []
    )
    daily_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    label_cache: dict[str, dict[str, Any]] = {}
    for label in sorted(durations, key=label_sort_key):
        label_dates = durations[label]
        maps = {
            period: clean_day_map(label_dates, period_dates.get(period, []))
            for period in (
                "pre",
                "early_pre",
                "late_pre",
                "post",
                "early_post",
                "late_post",
                "post_days_1_7",
                "post_days_8_14",
                "post_days_15_21",
                "post_days_22_28",
                "post_days_29_plus",
            )
        }
        if spec.remove_longest_post_per_syllable:
            trimmed = remove_longest(maps["post"])
            maps["post"] = trimmed
            for period in (
                "early_post",
                "late_post",
                "post_days_1_7",
                "post_days_8_14",
                "post_days_15_21",
                "post_days_22_28",
                "post_days_29_plus",
            ):
                allowed = set(period_dates.get(period, []))
                maps[period] = {d: v for d, v in trimmed.items() if d in allowed}

        if spec.pre_mode != "pooled_crossfit":
            early_values = flatten_day_map(maps["early_pre"])
            threshold = (
                float(np.quantile(early_values, spec.quantile))
                if bird_pre_eligible and early_values.size >= min_early_phrases
                else math.nan
            )
            summaries = {
                period: summarize_period(day_map, threshold)
                for period, day_map in maps.items()
            }
            full_ok = bool(
                bird_pre_eligible
                and eligible_full(
                    summaries["early_pre"],
                    summaries["late_pre"],
                    summaries["post"],
                    args,
                )
                and summaries["early_pre"].n_phrases >= min_early_phrases
            )

            fold_cache: dict[str, dict[str, Any]] = {}
            for fold in ("A", "B"):
                late_map = filter_fold(maps["late_pre"], folds["late_pre"], fold)
                post_map = filter_fold(maps["post"], folds["post"], fold)
                early_post_map = filter_fold(
                    maps["early_post"], folds["post"], fold
                )
                late_post_map = filter_fold(
                    maps["late_post"], folds["post"], fold
                )
                block_maps = {
                    block_name: filter_fold(maps[block_name], folds["post"], fold)
                    for block_name, _, _ in POST_BLOCKS_PRIMARY
                }
                fold_summaries = {
                    "late_pre": summarize_period(late_map, threshold),
                    "post": summarize_period(post_map, threshold),
                    "early_post": summarize_period(early_post_map, threshold),
                    "late_post": summarize_period(late_post_map, threshold),
                    **{
                        name: summarize_period(day_map, threshold)
                        for name, day_map in block_maps.items()
                    },
                }
                fold_cache[fold] = {
                    "threshold": threshold,
                    "baseline_key": "late_pre",
                    "maps": {
                        "late_pre": late_map,
                        "post": post_map,
                        "early_post": early_post_map,
                        "late_post": late_post_map,
                        **block_maps,
                    },
                    "summaries": fold_summaries,
                    "eligible": eligible_fold(
                        fold_summaries["late_pre"], fold_summaries["post"], args
                    ),
                }
        else:
            # Fully reciprocal pooled-pre cross-fit. Each fold estimates its own
            # threshold from its own pre-lesion days; the opposite fold is used
            # for held-out estimation.
            pooled_values = flatten_day_map(maps["pre"])
            threshold = (
                float(np.quantile(pooled_values, spec.quantile))
                if bird_pre_eligible and pooled_values.size >= min_early_phrases
                else math.nan
            )
            summaries = {
                period: summarize_period(day_map, threshold)
                for period, day_map in maps.items()
            }
            full_ok = False
            fold_cache = {}
            for fold in ("A", "B"):
                pre_map = filter_fold(maps["pre"], folds["pre"], fold)
                pre_values = flatten_day_map(pre_map)
                fold_threshold = (
                    float(np.quantile(pre_values, spec.quantile))
                    if bird_pre_eligible
                    and len(pre_map) >= args.min_fold_days
                    and pre_values.size >= min_early_phrases
                    else math.nan
                )
                post_map = filter_fold(maps["post"], folds["post"], fold)
                early_post_map = filter_fold(
                    maps["early_post"], folds["post"], fold
                )
                late_post_map = filter_fold(
                    maps["late_post"], folds["post"], fold
                )
                block_maps = {
                    block_name: filter_fold(maps[block_name], folds["post"], fold)
                    for block_name, _, _ in POST_BLOCKS_PRIMARY
                }
                fold_summaries = {
                    "pre": summarize_period(pre_map, fold_threshold),
                    "post": summarize_period(post_map, fold_threshold),
                    "early_post": summarize_period(early_post_map, fold_threshold),
                    "late_post": summarize_period(late_post_map, fold_threshold),
                    **{
                        name: summarize_period(day_map, fold_threshold)
                        for name, day_map in block_maps.items()
                    },
                }
                pre_summary = fold_summaries["pre"]
                post_summary = fold_summaries["post"]
                fold_ok = bool(
                    math.isfinite(fold_threshold)
                    and pre_summary.n_phrases >= min_early_phrases
                    and post_summary.n_phrases >= args.min_fold_phrases
                    and pre_summary.n_days >= args.min_fold_days
                    and post_summary.n_days >= args.min_fold_days
                )
                fold_cache[fold] = {
                    "threshold": fold_threshold,
                    "baseline_key": "pre",
                    "maps": {
                        "pre": pre_map,
                        "post": post_map,
                        "early_post": early_post_map,
                        "late_post": late_post_map,
                        **block_maps,
                    },
                    "summaries": fold_summaries,
                    "eligible": fold_ok,
                }

        if detailed and spec.pre_mode != "pooled_crossfit" and math.isfinite(threshold):
            for period, day_map in maps.items():
                if period in ("pre", "post"):
                    continue
                for d, values in sorted(day_map.items()):
                    day_metrics = daily_tail_row(values, threshold)
                    fold = ""
                    if period == "late_pre":
                        fold = folds["late_pre"].get(d, "")
                    elif "post" in period:
                        fold = folds["post"].get(d, "")
                    daily_rows.append(
                        {
                            **spec.as_dict(),
                            "bird": bird,
                            "group": group,
                            "syllable": label,
                            "period": period,
                            "date": d.isoformat(),
                            "relative_day": (d - surgery).days,
                            "fold": fold,
                            "threshold_seconds": threshold,
                            **day_metrics,
                        }
                    )

        if spec.pre_mode != "pooled_crossfit":
            full_row = {
                **spec.as_dict(),
                "bird": bird,
                "group": group,
                "syllable": label,
                "bird_pre_window_eligible": bird_pre_eligible,
                "threshold_seconds": threshold,
                "eligible_full": full_ok,
                **summaries["early_pre"].as_dict("early_pre_"),
                **summaries["late_pre"].as_dict("late_pre_"),
                **summaries["early_post"].as_dict("early_post_"),
                **summaries["late_post"].as_dict("late_post_"),
                **summaries["post"].as_dict("post_"),
                "delta_late_pre_to_early_post_mean_daily_burden_seconds_per_100": delta(
                    summaries["early_post"],
                    summaries["late_pre"],
                    "mean_daily_burden_seconds_per_100",
                ),
                "delta_late_pre_to_late_post_mean_daily_burden_seconds_per_100": delta(
                    summaries["late_post"],
                    summaries["late_pre"],
                    "mean_daily_burden_seconds_per_100",
                ),
                "delta_late_pre_to_full_post_mean_daily_burden_seconds_per_100": delta(
                    summaries["post"],
                    summaries["late_pre"],
                    "mean_daily_burden_seconds_per_100",
                ),
                "delta_late_post_minus_early_post_mean_daily_burden_seconds_per_100": delta(
                    summaries["late_post"],
                    summaries["early_post"],
                    "mean_daily_burden_seconds_per_100",
                ),
                "delta_early_pre_to_late_pre_mean_daily_burden_seconds_per_100": delta(
                    summaries["late_pre"],
                    summaries["early_pre"],
                    "mean_daily_burden_seconds_per_100",
                ),
            }
            for block_name, _, _ in POST_BLOCKS_PRIMARY:
                full_row[
                    f"delta_late_pre_to_{block_name}_mean_daily_burden_seconds_per_100"
                ] = delta(
                    summaries[block_name],
                    summaries["late_pre"],
                    "mean_daily_burden_seconds_per_100",
                )
            if detailed:
                full_rows.append(full_row)

        label_cache[label] = {
            "threshold": threshold,
            "full_ok": full_ok,
            "maps": maps,
            "summaries": summaries,
            "folds": fold_cache,
        }

    direction_topk_rows: list[dict[str, Any]] = []
    for direction, screen_fold, test_fold in DIRECTIONS:
        candidates: list[dict[str, Any]] = []
        for label, cache in label_cache.items():
            screen = cache["folds"][screen_fold]
            test = cache["folds"][test_fold]
            both_ok = bool(screen["eligible"] and test["eligible"])
            screen_baseline = screen["baseline_key"]
            test_baseline = test["baseline_key"]
            screen_delta = delta(
                screen["summaries"]["post"],
                screen["summaries"][screen_baseline],
                "mean_daily_burden_seconds_per_100",
            )
            heldout = {
                "heldout_delta_full_post": delta(
                    test["summaries"]["post"],
                    test["summaries"][test_baseline],
                    "mean_daily_burden_seconds_per_100",
                ),
                "heldout_delta_early_post": delta(
                    test["summaries"]["early_post"],
                    test["summaries"][test_baseline],
                    "mean_daily_burden_seconds_per_100",
                ),
                "heldout_delta_late_post": delta(
                    test["summaries"]["late_post"],
                    test["summaries"][test_baseline],
                    "mean_daily_burden_seconds_per_100",
                ),
                "heldout_late_minus_early_post": delta(
                    test["summaries"]["late_post"],
                    test["summaries"]["early_post"],
                    "mean_daily_burden_seconds_per_100",
                ),
            }
            for block_name, _, _ in POST_BLOCKS_PRIMARY:
                heldout[f"heldout_delta_{block_name}"] = delta(
                    test["summaries"][block_name],
                    test["summaries"][test_baseline],
                    "mean_daily_burden_seconds_per_100",
                )
            candidates.append(
                {
                    "label": label,
                    "eligible_both": both_ok,
                    "screen_delta": screen_delta,
                    "screen": screen,
                    "test": test,
                    **heldout,
                }
            )
        eligible_candidates = [
            row
            for row in candidates
            if row["eligible_both"] and math.isfinite(row["screen_delta"])
        ]
        eligible_candidates.sort(
            key=lambda row: (-row["screen_delta"], label_sort_key(row["label"]))
        )
        for rank, row in enumerate(eligible_candidates, start=1):
            row["screen_rank"] = rank

        if detailed:
            max_requested_k = max(top_k_values)
            for row in candidates:
                rank = int(row.get("screen_rank", 0))
                selected_any = bool(rank and rank <= max_requested_k)
                day_p = math.nan
                day_p_method = "not_tested"
                day_p_n = 0
                test_baseline = row["test"]["baseline_key"]
                if selected_any and args.selected_syllable_day_permutations > 0:
                    day_p, day_p_method, day_p_n = day_permutation_p(
                        row["test"]["maps"][test_baseline],
                        row["test"]["maps"]["post"],
                        row["test"]["threshold"],
                        args.max_exact_day_assignments,
                        args.selected_syllable_day_permutations,
                        stable_rng(
                            args.seed,
                            "selected_syllable_day_p",
                            spec.name,
                            bird,
                            direction,
                            row["label"],
                        ),
                    )
                cross_rows.append(
                    {
                        **spec.as_dict(),
                        "bird": bird,
                        "group": group,
                        "direction": direction,
                        "screen_fold": screen_fold,
                        "test_fold": test_fold,
                        "syllable": row["label"],
                        "screen_threshold_seconds": row["screen"]["threshold"],
                        "test_threshold_seconds": row["test"]["threshold"],
                        "eligible_both": row["eligible_both"],
                        "screen_rank": rank if rank else "",
                        "selected_within_max_requested_k": selected_any,
                        "screen_delta_full_post_mean_daily_burden_seconds_per_100": row[
                            "screen_delta"
                        ],
                        **{
                            key + "_mean_daily_burden_seconds_per_100": value
                            for key, value in row.items()
                            if key.startswith("heldout_")
                            and isinstance(value, (int, float))
                        },
                        "heldout_day_permutation_p_one_sided": day_p,
                        "heldout_day_permutation_method": day_p_method,
                        "heldout_day_assignments_or_permutations": day_p_n,
                    }
                )

        for k in sorted(set(top_k_values)):
            selected = eligible_candidates[: min(k, len(eligible_candidates))]
            direction_row = {
                **spec.as_dict(),
                "bird": bird,
                "group": group,
                "direction": direction,
                "screen_fold": screen_fold,
                "test_fold": test_fold,
                "top_k_requested": k,
                "n_eligible_syllables": len(eligible_candidates),
                "n_selected_syllables": len(selected),
                "selected_syllable_labels": ";".join(
                    row["label"] for row in selected
                ),
                "mean_screen_delta_full_post_mean_daily_burden_seconds_per_100": mean_or_nan(
                    row["screen_delta"] for row in selected
                ),
                "mean_heldout_delta_full_post_mean_daily_burden_seconds_per_100": mean_or_nan(
                    row["heldout_delta_full_post"] for row in selected
                ),
                "mean_heldout_delta_early_post_mean_daily_burden_seconds_per_100": mean_or_nan(
                    row["heldout_delta_early_post"] for row in selected
                ),
                "mean_heldout_delta_late_post_mean_daily_burden_seconds_per_100": mean_or_nan(
                    row["heldout_delta_late_post"] for row in selected
                ),
                "mean_heldout_late_minus_early_post_mean_daily_burden_seconds_per_100": mean_or_nan(
                    row["heldout_late_minus_early_post"] for row in selected
                ),
            }
            for block_name, _, _ in POST_BLOCKS_PRIMARY:
                direction_row[
                    f"mean_heldout_delta_{block_name}_mean_daily_burden_seconds_per_100"
                ] = mean_or_nan(
                    row[f"heldout_delta_{block_name}"] for row in selected
                )
            direction_topk_rows.append(direction_row)

    crossfitted_rows: list[dict[str, Any]] = []
    for k in sorted(set(top_k_values)):
        rows = [
            row for row in direction_topk_rows if row["top_k_requested"] == k
        ]
        available = [
            row
            for row in rows
            if row["n_selected_syllables"] > 0
            and math.isfinite(
                finite_or_nan(
                    row[
                        "mean_heldout_delta_full_post_mean_daily_burden_seconds_per_100"
                    ]
                )
            )
        ]
        both_available = len(available) == 2
        primary_available = (
            both_available if args.require_both_directions else bool(available)
        )
        endpoint_names = [
            "mean_heldout_delta_full_post_mean_daily_burden_seconds_per_100",
            "mean_heldout_delta_early_post_mean_daily_burden_seconds_per_100",
            "mean_heldout_delta_late_post_mean_daily_burden_seconds_per_100",
            "mean_heldout_late_minus_early_post_mean_daily_burden_seconds_per_100",
            *[
                f"mean_heldout_delta_{name}_mean_daily_burden_seconds_per_100"
                for name, _, _ in POST_BLOCKS_PRIMARY
            ],
        ]
        row = {
            **spec.as_dict(),
            "bird": bird,
            "group": group,
            "top_k_requested": k,
            "n_directions_available": len(available),
            "both_directions_available": both_available,
            "primary_crossfit_eligible": primary_available,
            "n_eligible_syllables_A_screen_B_test": next(
                (
                    r["n_eligible_syllables"]
                    for r in rows
                    if r["direction"] == "A_screen_B_test"
                ),
                0,
            ),
            "n_eligible_syllables_B_screen_A_test": next(
                (
                    r["n_eligible_syllables"]
                    for r in rows
                    if r["direction"] == "B_screen_A_test"
                ),
                0,
            ),
            "selected_labels_A_screen_B_test": next(
                (
                    r["selected_syllable_labels"]
                    for r in rows
                    if r["direction"] == "A_screen_B_test"
                ),
                "",
            ),
            "selected_labels_B_screen_A_test": next(
                (
                    r["selected_syllable_labels"]
                    for r in rows
                    if r["direction"] == "B_screen_A_test"
                ),
                "",
            ),
        }
        for endpoint in endpoint_names:
            output_name = endpoint.replace("mean_heldout_", "crossfitted_")
            row[output_name] = (
                mean_or_nan(r[endpoint] for r in available)
                if primary_available
                else math.nan
            )
        crossfitted_rows.append(row)

    if spec.pre_mode != "pooled_crossfit":
        if detailed:
            eligible_full_rows = [
                row for row in full_rows if row.get("eligible_full")
            ]
        else:
            eligible_full_rows = [
                {
                    "syllable": label,
                    "eligible_full": cache["full_ok"],
                    "delta_late_pre_to_early_post_mean_daily_burden_seconds_per_100": delta(
                        cache["summaries"]["early_post"],
                        cache["summaries"]["late_pre"],
                        "mean_daily_burden_seconds_per_100",
                    ),
                    "delta_late_pre_to_late_post_mean_daily_burden_seconds_per_100": delta(
                        cache["summaries"]["late_post"],
                        cache["summaries"]["late_pre"],
                        "mean_daily_burden_seconds_per_100",
                    ),
                    "delta_late_pre_to_full_post_mean_daily_burden_seconds_per_100": delta(
                        cache["summaries"]["post"],
                        cache["summaries"]["late_pre"],
                        "mean_daily_burden_seconds_per_100",
                    ),
                    "delta_late_post_minus_early_post_mean_daily_burden_seconds_per_100": delta(
                        cache["summaries"]["late_post"],
                        cache["summaries"]["early_post"],
                        "mean_daily_burden_seconds_per_100",
                    ),
                }
                for label, cache in label_cache.items()
                if cache["full_ok"]
            ]

        positive_deltas = sorted(
            [
                max(
                    0.0,
                    finite_or_nan(
                        row[
                            "delta_late_pre_to_full_post_mean_daily_burden_seconds_per_100"
                        ]
                    ),
                )
                for row in eligible_full_rows
                if math.isfinite(
                    finite_or_nan(
                        row[
                            "delta_late_pre_to_full_post_mean_daily_burden_seconds_per_100"
                        ]
                    )
                )
            ],
            reverse=True,
        )
        total_positive = (
            float(np.sum(positive_deltas)) if positive_deltas else math.nan
        )
        concentration = {}
        for k in (1, 3, 5):
            concentration[f"top{k}_fraction_of_positive_syllable_burden"] = (
                float(np.sum(positive_deltas[:k]) / total_positive)
                if positive_deltas and total_positive > 0
                else math.nan
            )

        supplemental_row = {
            **spec.as_dict(),
            "bird": bird,
            "group": group,
            "n_selected_pre_singing_days": len(period_dates["pre"]),
            "n_early_pre_singing_days_available": len(period_dates["early_pre"]),
            "n_late_pre_singing_days_available": len(period_dates["late_pre"]),
            "n_early_post_recording_days": len(period_dates["early_post"]),
            "n_late_post_recording_days": len(period_dates["late_post"]),
            "n_full_post_recording_days": len(period_dates["post"]),
            "bird_pre_window_eligible": bird_pre_eligible,
            "n_full_eligible_syllables": len(eligible_full_rows),
            "median_all_syllable_delta_early_post": median_or_nan(
                row[
                    "delta_late_pre_to_early_post_mean_daily_burden_seconds_per_100"
                ]
                for row in eligible_full_rows
            ),
            "median_all_syllable_delta_late_post": median_or_nan(
                row[
                    "delta_late_pre_to_late_post_mean_daily_burden_seconds_per_100"
                ]
                for row in eligible_full_rows
            ),
            "median_all_syllable_delta_full_post": median_or_nan(
                row[
                    "delta_late_pre_to_full_post_mean_daily_burden_seconds_per_100"
                ]
                for row in eligible_full_rows
            ),
            "median_all_syllable_late_minus_early_post": median_or_nan(
                row[
                    "delta_late_post_minus_early_post_mean_daily_burden_seconds_per_100"
                ]
                for row in eligible_full_rows
            ),
            **concentration,
        }
    else:
        supplemental_row = {
            **spec.as_dict(),
            "bird": bird,
            "group": group,
            "n_selected_pre_singing_days": len(period_dates["pre"]),
            "n_early_pre_singing_days_available": 0,
            "n_late_pre_singing_days_available": 0,
            "n_early_post_recording_days": len(period_dates["early_post"]),
            "n_late_post_recording_days": len(period_dates["late_post"]),
            "n_full_post_recording_days": len(period_dates["post"]),
            "bird_pre_window_eligible": bird_pre_eligible,
            "n_full_eligible_syllables": 0,
            "median_all_syllable_delta_early_post": math.nan,
            "median_all_syllable_delta_late_post": math.nan,
            "median_all_syllable_delta_full_post": math.nan,
            "median_all_syllable_late_minus_early_post": math.nan,
            "top1_fraction_of_positive_syllable_burden": math.nan,
            "top3_fraction_of_positive_syllable_burden": math.nan,
            "top5_fraction_of_positive_syllable_burden": math.nan,
        }

    return {
        "date_rows": date_rows,
        "daily_rows": daily_rows,
        "full_rows": full_rows,
        "cross_rows": cross_rows,
        "direction_topk_rows": direction_topk_rows,
        "crossfitted_rows": crossfitted_rows,
        "supplemental_row": supplemental_row,
        "pre_allocation_row": pre_allocation_row,
    }

def analyze_spec(
    bird_data: Mapping[str, Mapping[str, Any]],
    spec: AnalysisSpec,
    top_k_values: Sequence[int],
    args: argparse.Namespace,
    detailed: bool,
) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {
        "date_rows": [],
        "daily_rows": [],
        "full_rows": [],
        "cross_rows": [],
        "direction_topk_rows": [],
        "crossfitted_rows": [],
        "supplemental_rows": [],
        "pre_allocation_rows": [],
    }
    for bird in sorted(bird_data):
        result = analyze_bird(
            bird, bird_data[bird], spec, top_k_values, args, detailed
        )
        for key in outputs:
            if key == "supplemental_rows":
                source_key = "supplemental_row"
            elif key == "pre_allocation_rows":
                source_key = "pre_allocation_row"
            else:
                source_key = key
            value = result[source_key]
            if isinstance(value, list):
                outputs[key].extend(value)
            else:
                outputs[key].append(value)
    return outputs


# -----------------------------------------------------------------------------
# Bird-level group summaries
# -----------------------------------------------------------------------------


def group_test_rows(
    bird_rows: list[dict[str, Any]],
    endpoints: Sequence[str],
    args: argparse.Namespace,
    family: str,
    spec_name: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if top_k is not None:
        bird_rows = [row for row in bird_rows if int(row.get("top_k_requested", -1)) == top_k]
    groups = [args.medial_group, args.lateral_group, args.sham_group]
    pairs = [
        (args.medial_group, args.lateral_group),
        (args.medial_group, args.sham_group),
        (args.lateral_group, args.sham_group),
    ]
    output: list[dict[str, Any]] = []
    for endpoint in endpoints:
        values = {
            group: finite_array(
                finite_or_nan(row.get(endpoint))
                for row in bird_rows
                if row.get("group") == group
            )
            for group in groups
        }
        endpoint_rows: list[dict[str, Any]] = []
        for group1, group2 in pairs:
            x, y = values[group1], values[group2]
            observed, p_one, method, n_assignments = group_permutation_test(
                x,
                y,
                args.group_statistic,
                "greater",
                args.max_exact_group_assignments,
                args.group_permutations,
                stable_rng(args.seed, "group", family, spec_name, top_k, endpoint, group1, group2),
            )
            _, p_two, _, _ = group_permutation_test(
                x,
                y,
                args.group_statistic,
                "two-sided",
                args.max_exact_group_assignments,
                args.group_permutations,
                stable_rng(args.seed, "group_two", family, spec_name, top_k, endpoint, group1, group2),
            )
            x_lo, x_hi = bootstrap_mean_ci(
                x,
                args.bootstrap_replicates,
                stable_rng(args.seed, "boot_group", family, spec_name, top_k, endpoint, group1),
            )
            y_lo, y_hi = bootstrap_mean_ci(
                y,
                args.bootstrap_replicates,
                stable_rng(args.seed, "boot_group", family, spec_name, top_k, endpoint, group2),
            )
            d_lo, d_hi = bootstrap_difference_ci(
                x,
                y,
                args.bootstrap_replicates,
                stable_rng(args.seed, "boot_diff", family, spec_name, top_k, endpoint, group1, group2),
            )
            endpoint_rows.append(
                {
                    "analysis_family": family,
                    "spec_name": spec_name,
                    "top_k": top_k if top_k is not None else "",
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
                    "one_sided_p_raw": p_one,
                    "one_sided_p_holm_across_three_contrasts": math.nan,
                    "two_sided_p_raw": p_two,
                    "two_sided_p_holm_across_three_contrasts": math.nan,
                    "primary_medial_vs_lateral_contrast": bool(
                        group1 == args.medial_group and group2 == args.lateral_group
                    ),
                    "permutation_method": method,
                    "n_assignments_or_permutations": n_assignments,
                }
            )
        adjusted_one = holm_adjust([row["one_sided_p_raw"] for row in endpoint_rows])
        adjusted_two = holm_adjust([row["two_sided_p_raw"] for row in endpoint_rows])
        for row, p1, p2 in zip(endpoint_rows, adjusted_one, adjusted_two):
            row["one_sided_p_holm_across_three_contrasts"] = p1
            row["two_sided_p_holm_across_three_contrasts"] = p2
        output.extend(endpoint_rows)
    return output


def within_group_signflip_rows(
    bird_rows: list[dict[str, Any]],
    endpoints: Sequence[str],
    args: argparse.Namespace,
    top_k: int,
    spec_name: str,
) -> list[dict[str, Any]]:
    rows = [row for row in bird_rows if int(row.get("top_k_requested", -1)) == top_k]
    output: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for group in (args.medial_group, args.lateral_group, args.sham_group):
            values = [
                finite_or_nan(row.get(endpoint))
                for row in rows
                if row.get("group") == group
            ]
            observed, p, n = exact_signflip(values, "greater")
            output.append(
                {
                    "spec_name": spec_name,
                    "top_k": top_k,
                    "endpoint": endpoint,
                    "group": group,
                    "n_birds": len(finite_array(values)),
                    "mean_bird_value": observed,
                    "one_sided_signflip_p_greater_than_zero": p,
                    "n_sign_assignments": n,
                }
            )
    return output


# -----------------------------------------------------------------------------
# Main and output
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["primary_pre_mode"] = "adaptive_segmented"
    config["adaptive_split_rule"] = (
        "Select up to the final max_pre_singing_days before surgery; require "
        "at least min_total_pre_singing_days; split chronologically with the "
        "extra odd-numbered day assigned to early pre."
    )
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )

    bird_data, audit_rows = load_data(args)
    primary_spec = AnalysisSpec(
        name="primary_adaptive_pre_split",
        quantile=args.primary_quantile,
        pre_mode="adaptive_segmented",
        max_pre_singing_days=args.max_pre_singing_days,
        min_total_pre_singing_days=args.min_total_pre_singing_days,
        early_pre_singing_days=args.early_pre_singing_days,
        late_pre_singing_days=args.late_pre_singing_days,
        post_end_day=args.post_end_day,
        split_seed=args.seed,
    )
    primary_top_k_values = sorted(set([*args.top_k_values, args.primary_top_k]))
    primary = analyze_spec(
        bird_data, primary_spec, primary_top_k_values, args, detailed=True
    )

    primary_endpoints = [
        "crossfitted_delta_full_post_mean_daily_burden_seconds_per_100",
        "crossfitted_delta_early_post_mean_daily_burden_seconds_per_100",
        "crossfitted_delta_late_post_mean_daily_burden_seconds_per_100",
        "crossfitted_late_minus_early_post_mean_daily_burden_seconds_per_100",
        *[
            f"crossfitted_delta_{name}_mean_daily_burden_seconds_per_100"
            for name, _, _ in POST_BLOCKS_PRIMARY
        ],
    ]
    primary_group_rows: list[dict[str, Any]] = []
    for k in primary_top_k_values:
        primary_group_rows.extend(
            group_test_rows(
                primary["crossfitted_rows"],
                primary_endpoints,
                args,
                "crossfitted_top_k",
                primary_spec.name,
                top_k=k,
            )
        )
    primary_signflip_rows = within_group_signflip_rows(
        primary["crossfitted_rows"],
        primary_endpoints,
        args,
        args.primary_top_k,
        primary_spec.name,
    )

    supplemental_endpoints = [
        "median_all_syllable_delta_early_post",
        "median_all_syllable_delta_late_post",
        "median_all_syllable_delta_full_post",
        "median_all_syllable_late_minus_early_post",
        "top1_fraction_of_positive_syllable_burden",
        "top3_fraction_of_positive_syllable_burden",
        "top5_fraction_of_positive_syllable_burden",
    ]
    supplemental_group_rows = group_test_rows(
        primary["supplemental_rows"],
        supplemental_endpoints,
        args,
        "supplemental_all_syllable",
        primary_spec.name,
    )

    # Sensitivity: change one component at a time from the adaptive primary.
    sensitivity_specs: list[AnalysisSpec] = []
    if not args.skip_sensitivity:
        def adaptive_spec(
            name: str,
            *,
            quantile: float | None = None,
            post_end_day: int | None | str = "primary",
            split_seed: int | None = None,
            remove_longest: bool = False,
        ) -> AnalysisSpec:
            resolved_post_end = (
                args.post_end_day if post_end_day == "primary" else post_end_day
            )
            return AnalysisSpec(
                name=name,
                quantile=args.primary_quantile if quantile is None else quantile,
                pre_mode="adaptive_segmented",
                max_pre_singing_days=args.max_pre_singing_days,
                min_total_pre_singing_days=args.min_total_pre_singing_days,
                early_pre_singing_days=args.early_pre_singing_days,
                late_pre_singing_days=args.late_pre_singing_days,
                post_end_day=resolved_post_end,
                split_seed=args.seed if split_seed is None else split_seed,
                remove_longest_post_per_syllable=remove_longest,
            )

        for q in sorted(set(args.sensitivity_quantiles)):
            if math.isclose(q, args.primary_quantile):
                continue
            sensitivity_specs.append(
                adaptive_spec(name=f"quantile_{q:g}", quantile=q)
            )

        # Fixed equal early/late windows test the adaptive allocation against
        # conventional prespecified windows.
        fixed_sizes = sorted(set([4, *args.sensitivity_pre_singing_days]))
        for n_days in fixed_sizes:
            sensitivity_specs.append(
                AnalysisSpec(
                    name=f"fixed_pre_windows_{n_days}_plus_{n_days}",
                    quantile=args.primary_quantile,
                    pre_mode="fixed_segmented",
                    max_pre_singing_days=2 * n_days,
                    min_total_pre_singing_days=2 * n_days,
                    early_pre_singing_days=n_days,
                    late_pre_singing_days=n_days,
                    post_end_day=args.post_end_day,
                    split_seed=args.seed,
                    min_early_pre_days_override=n_days,
                    min_late_pre_days_override=n_days,
                )
            )

        # This preserves the original 7-day late-pre window while allowing a
        # two-day early-pre threshold when at least 100 phrase occurrences are
        # available. It is included as a targeted sensitivity, not the primary.
        sensitivity_specs.append(
            AnalysisSpec(
                name="fixed_early2_late7_min100phrases",
                quantile=args.primary_quantile,
                pre_mode="fixed_segmented",
                max_pre_singing_days=9,
                min_total_pre_singing_days=9,
                early_pre_singing_days=2,
                late_pre_singing_days=7,
                post_end_day=args.post_end_day,
                split_seed=args.seed,
                min_early_pre_phrases_override=100,
                min_early_pre_days_override=2,
                min_late_pre_days_override=7,
            )
        )

        # Fully reciprocal pooled-pre cross-fit: pre A defines the threshold and
        # ranks syllables against post A, while pre/post B estimates the held-out
        # effect; then the folds reverse. This includes birds with at least eight
        # total pre-lesion singing days without requiring distinct early/late
        # threshold-training windows.
        sensitivity_specs.append(
            AnalysisSpec(
                name="pooled_pre_post_reciprocal_crossfit",
                quantile=args.primary_quantile,
                pre_mode="pooled_crossfit",
                max_pre_singing_days=args.max_pre_singing_days,
                min_total_pre_singing_days=args.min_total_pre_singing_days,
                early_pre_singing_days=0,
                late_pre_singing_days=0,
                post_end_day=args.post_end_day,
                split_seed=args.seed,
            )
        )

        for post_end in sorted(set(args.sensitivity_post_end_days)):
            if post_end == args.post_end_day:
                continue
            sensitivity_specs.append(
                adaptive_spec(
                    name=f"post_days_{args.post_start_day}_{post_end}",
                    post_end_day=post_end,
                )
            )
        sensitivity_specs.append(
            adaptive_spec(name="all_available_post_days", post_end_day=None)
        )
        sensitivity_specs.append(
            adaptive_spec(
                name="remove_longest_post_phrase_per_syllable",
                remove_longest=True,
            )
        )
        for offset in range(1, args.sensitivity_split_seeds + 1):
            sensitivity_specs.append(
                adaptive_spec(
                    name=f"split_seed_{args.seed + offset}",
                    split_seed=args.seed + offset,
                )
            )

    sensitivity_spec_rows: list[dict[str, Any]] = []
    sensitivity_pre_allocation_rows: list[dict[str, Any]] = []
    sensitivity_bird_rows: list[dict[str, Any]] = []
    sensitivity_group_rows: list[dict[str, Any]] = []
    for spec in sensitivity_specs:
        sensitivity_spec_rows.append(spec.as_dict())
        result = analyze_spec(
            bird_data, spec, [args.primary_top_k], args, detailed=False
        )
        sensitivity_pre_allocation_rows.extend(result["pre_allocation_rows"])
        sensitivity_bird_rows.extend(result["crossfitted_rows"])
        sensitivity_group_rows.extend(
            group_test_rows(
                result["crossfitted_rows"],
                primary_endpoints,
                args,
                "sensitivity_crossfitted_top_k",
                spec.name,
                top_k=args.primary_top_k,
            )
        )

    # Write outputs.
    write_csv(
        args.output / "input_audit.csv",
        audit_rows,
        [
            "bird", "group", "surgery_date", "database_path", "n_results",
            "n_valid_phrase_spans", "n_labels", "n_recording_dates",
            "first_recording_date", "last_recording_date",
        ],
    )
    write_csv(
        args.output / "primary_pre_window_allocation.csv",
        primary["pre_allocation_rows"],
        (
            list(primary["pre_allocation_rows"][0].keys())
            if primary["pre_allocation_rows"]
            else ["bird", "group", "pre_mode"]
        ),
    )
    write_csv(
        args.output / "primary_period_dates_and_folds.csv",
        primary["date_rows"],
        (
            list(primary["date_rows"][0].keys())
            if primary["date_rows"]
            else ["spec_name", "bird", "group", "period", "date", "fold"]
        ),
    )
    daily_fields = [
        "spec_name", "bird", "group", "syllable", "period", "date",
        "relative_day", "fold", "threshold_seconds", "n_phrases", "n_extreme",
        "event_rate", "event_rate_per_100", "burden_seconds_per_100",
        "conditional_excess_median_seconds", "conditional_excess_mean_seconds",
        "max_duration_seconds", "max_excess_seconds",
    ]
    write_csv(args.output / "primary_daily_tail_metrics.csv", primary["daily_rows"], daily_fields)

    if primary["full_rows"]:
        full_fields = list(primary["full_rows"][0].keys())
    else:
        full_fields = ["bird", "group", "syllable"]
    write_csv(
        args.output / "primary_all_syllable_segmented_metrics.csv",
        primary["full_rows"],
        full_fields,
    )
    if primary["cross_rows"]:
        cross_fields = list(primary["cross_rows"][0].keys())
    else:
        cross_fields = ["bird", "group", "direction", "syllable"]
    write_csv(
        args.output / "primary_cross_screened_syllable_ranks.csv",
        primary["cross_rows"],
        cross_fields,
    )
    if primary["direction_topk_rows"]:
        direction_fields = list(primary["direction_topk_rows"][0].keys())
    else:
        direction_fields = ["bird", "group", "direction", "top_k_requested"]
    write_csv(
        args.output / "primary_direction_specific_topk_results.csv",
        primary["direction_topk_rows"],
        direction_fields,
    )
    if primary["crossfitted_rows"]:
        bird_fields = list(primary["crossfitted_rows"][0].keys())
    else:
        bird_fields = ["bird", "group", "top_k_requested"]
    write_csv(
        args.output / "primary_crossfitted_topk_bird_results.csv",
        primary["crossfitted_rows"],
        bird_fields,
    )
    write_csv(
        args.output / "primary_crossfitted_topk_group_tests.csv",
        primary_group_rows,
        list(primary_group_rows[0].keys()) if primary_group_rows else ["endpoint"],
    )
    write_csv(
        args.output / "primary_within_group_signflip_tests.csv",
        primary_signflip_rows,
        list(primary_signflip_rows[0].keys()) if primary_signflip_rows else ["endpoint"],
    )
    write_csv(
        args.output / "supplemental_bird_segmented_results.csv",
        primary["supplemental_rows"],
        list(primary["supplemental_rows"][0].keys()) if primary["supplemental_rows"] else ["bird"],
    )
    write_csv(
        args.output / "supplemental_group_tests.csv",
        supplemental_group_rows,
        list(supplemental_group_rows[0].keys()) if supplemental_group_rows else ["endpoint"],
    )
    write_csv(
        args.output / "sensitivity_specifications.csv",
        sensitivity_spec_rows,
        list(sensitivity_spec_rows[0].keys()) if sensitivity_spec_rows else ["spec_name"],
    )
    write_csv(
        args.output / "sensitivity_pre_window_allocation.csv",
        sensitivity_pre_allocation_rows,
        (
            list(sensitivity_pre_allocation_rows[0].keys())
            if sensitivity_pre_allocation_rows
            else ["spec_name", "bird", "group", "pre_mode"]
        ),
    )
    write_csv(
        args.output / "sensitivity_crossfitted_topk_bird_results.csv",
        sensitivity_bird_rows,
        list(sensitivity_bird_rows[0].keys()) if sensitivity_bird_rows else ["spec_name"],
    )
    write_csv(
        args.output / "sensitivity_crossfitted_topk_group_tests.csv",
        sensitivity_group_rows,
        list(sensitivity_group_rows[0].keys()) if sensitivity_group_rows else ["spec_name"],
    )

    # Human-readable summary, centered on the primary top-k endpoint.
    primary_bird_rows = [
        row
        for row in primary["crossfitted_rows"]
        if int(row["top_k_requested"]) == args.primary_top_k
    ]
    main_endpoint = "crossfitted_delta_full_post_mean_daily_burden_seconds_per_100"
    delayed_endpoint = "crossfitted_late_minus_early_post_mean_daily_burden_seconds_per_100"
    lines = [
        "Adaptive segmented extreme phrase localization analysis",
        "=======================================================",
        f"Input directory: {args.json_dir.resolve()}",
        f"Birds read: {len(bird_data)}",
        (
            f"Primary pre rule: select up to the final {args.max_pre_singing_days} "
            f"singing days; require at least {args.min_total_pre_singing_days}; "
            "split chronologically with the extra odd day assigned to early pre"
        ),
        (
            f"Post windows: early +{args.post_start_day} to +{args.early_post_end_day}; "
            f"late +{args.late_post_start_day} to +{args.post_end_day}"
        ),
        f"Threshold: early-pre quantile {args.primary_quantile:g}",
        f"Primary localization: cross-fitted top {args.primary_top_k}",
        (
            "USA5443 pre allocation: "
            + next(
                (
                    f"{row['n_early_pre_singing_days']} early + "
                    f"{row['n_late_pre_singing_days']} late days; "
                    f"eligible={row['bird_pre_window_eligible']}"
                    for row in primary["pre_allocation_rows"]
                    if row["bird"] == "USA5443"
                ),
                "bird not found",
            )
        ),
        "",
        "Primary bird-level summaries",
        "----------------------------",
    ]
    for group in (args.medial_group, args.lateral_group, args.sham_group):
        rows = [row for row in primary_bird_rows if row["group"] == group]
        main_values = [finite_or_nan(row[main_endpoint]) for row in rows]
        delayed_values = [finite_or_nan(row[delayed_endpoint]) for row in rows]
        lines.append(
            f"{group}: n={len(finite_array(main_values))}, "
            f"median full-post held-out burden change={fmt(median_or_nan(main_values))} "
            "s/100 occurrences, "
            f"median late-minus-early post change={fmt(median_or_nan(delayed_values))}"
        )
    lines.extend(["", "Primary planned contrast", "------------------------"])
    match = next(
        (
            row
            for row in primary_group_rows
            if int(row["top_k"]) == args.primary_top_k
            and row["endpoint"] == main_endpoint
            and row["group1"] == args.medial_group
            and row["group2"] == args.lateral_group
        ),
        None,
    )
    if match:
        lines.append(
            f"{args.medial_group} > {args.lateral_group}: "
            f"effect={fmt(finite_or_nan(match['observed_group1_minus_group2']), 6)}, "
            f"one-sided p={fmt(finite_or_nan(match['one_sided_p_raw']), 6)}, "
            "Holm across all three group contrasts="
            f"{fmt(finite_or_nan(match['one_sided_p_holm_across_three_contrasts']), 6)}"
        )
    lines.extend(
        [
            "",
            "Interpretation notes",
            "--------------------",
            "The primary endpoint ranks syllables in one recording-day fold and",
            "measures the selected top-k syllables only in the held-out fold.",
            "Daily tail burden is averaged across recording days, so a day with",
            "many phrase occurrences does not automatically outweigh another day.",
            "The early-post, late-post, and four 7-day block endpoints use the",
            "same screen-selected syllables and therefore describe time course",
            "without reselecting a different subset in every post segment.",
            "",
            f"Results written to: {args.output.resolve()}",
        ]
    )
    summary = "\n".join(lines)
    (args.output / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
