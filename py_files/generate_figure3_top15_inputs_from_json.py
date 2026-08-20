#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Figure 3 period-statistics and phrase-occurrence CSVs from AFP JSONs.

This script reads the ``AFP_lesion_jsons`` directory directly and creates the
inputs expected by ``figure3_top15_SDscatter_deltaCV_timecourse_pngonly.py``.
It avoids relying on an older ``usage_balanced_phrase_duration_stats.csv`` whose
period definitions may not match the current analysis windows.

The two primary outputs are:

1. ``figure3_balanced_period_stats.csv``
   One row per eligible bird x syllable x analysis period (Late Pre or Post).
   Late-pre and post phrase counts are equalized within each bird x syllable,
   metrics are recalculated over repeated subsampling draws, and the draw-level
   estimates are averaged. This file should be supplied as ``--scatter-csv``.

2. ``figure3_phrase_duration_long.csv``
   One row per phrase occurrence, with recording date, relative day, duration,
   and period assignment. This file should be supplied as
   ``--duration-long-csv`` and is also sufficient to reconstruct the daily CV
   time course.

Additional outputs include a wide bird x syllable metric table, an input audit,
and a plain-text summary.

Expected input structure
------------------------
AFP_lesion_jsons/
    AFP_lesion_bird_metadata.json
    <group>/<bird>/<bird>_decoded_database.json

The metadata JSON must be keyed by bird and contain ``lesion_group`` and
``lesion_surgery_date``. Each decoded database must contain a top-level
``results`` list; each result should contain ``creation_date`` and a
``syllable_onsets_offsets_ms`` mapping from syllable label to [start, end] spans.

Example
-------
python generate_figure3_top15_inputs_from_json.py \
  "$HOME/Desktop/AFP_lesion_jsons" \
  --output "$HOME/Desktop/Figure3_top15_inputs" \
  --pre-start-day -14 --pre-end-day -1 \
  --post-start-day 1 --post-end-day 14 \
  --long-start-day -30 --long-end-day 30 \
  --min-period-phrases 10 \
  --min-period-days 2 \
  --balance-draws 200 \
  --seed 123

Required packages: numpy
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate balanced Late Pre/Post phrase-duration statistics and an "
            "occurrence-level long CSV directly from AFP lesion decoded databases."
        ),
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="AFP_lesion_jsons directory containing metadata and decoded databases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure3_top15_inputs"),
        help="Output directory.",
    )
    parser.add_argument("--pre-start-day", type=int, default=-14)
    parser.add_argument("--pre-end-day", type=int, default=-1)
    parser.add_argument("--post-start-day", type=int, default=1)
    parser.add_argument("--post-end-day", type=int, default=14)
    parser.add_argument(
        "--long-start-day",
        type=int,
        default=-30,
        help="First relative day retained in the occurrence-level long CSV.",
    )
    parser.add_argument(
        "--long-end-day",
        type=int,
        default=30,
        help="Last relative day retained in the occurrence-level long CSV.",
    )
    parser.add_argument(
        "--min-period-phrases",
        type=int,
        default=10,
        help="Minimum available phrases in both Late Pre and Post for eligibility.",
    )
    parser.add_argument(
        "--min-period-days",
        type=int,
        default=2,
        help="Minimum contributing recording days in both periods.",
    )
    parser.add_argument(
        "--balance-draws",
        type=int,
        default=200,
        help="Number of equal-count subsampling draws per bird x syllable.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--include-ineligible-period-rows",
        action="store_true",
        help=(
            "Also write unbalanced descriptive period rows for syllables that fail "
            "the minimum count/day criteria. These rows are marked eligible=False."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.json_dir.exists():
        raise FileNotFoundError(args.json_dir)
    if args.pre_start_day > args.pre_end_day:
        raise ValueError("--pre-start-day must be <= --pre-end-day")
    if args.post_start_day > args.post_end_day:
        raise ValueError("--post-start-day must be <= --post-end-day")
    if args.long_start_day > args.long_end_day:
        raise ValueError("--long-start-day must be <= --long-end-day")
    if args.pre_end_day >= 0:
        raise ValueError("The late-pre window should end before day 0.")
    if args.post_start_day <= 0:
        raise ValueError("The post window should begin after day 0.")
    if args.min_period_phrases < 2:
        raise ValueError("--min-period-phrases must be at least 2")
    if args.min_period_days < 1:
        raise ValueError("--min-period-days must be at least 1")
    if args.balance_draws < 1:
        raise ValueError("--balance-draws must be at least 1")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def parse_iso_date(value: Any) -> date:
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    # ``datetime.fromisoformat`` does not accept trailing Z on older Python.
    text = text.replace("Z", "+00:00")
    return datetime.fromisoformat(text).date()


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def label_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def find_database_paths(json_dir: Path) -> list[Path]:
    paths = sorted(json_dir.glob("*/*/*decoded_database.json"))
    if not paths:
        paths = sorted(json_dir.rglob("*decoded_database.json"))
    return paths


def flatten_day_map(day_map: Mapping[date, Sequence[float]]) -> np.ndarray:
    pieces = [np.asarray(values, dtype=float) for _, values in sorted(day_map.items()) if values]
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def sample_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    return float(np.std(values, ddof=1))


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def descriptive_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return {
            "mean_s": math.nan,
            "sd_s": math.nan,
            "variance_s2": math.nan,
            "cv": math.nan,
        }
    mean_s = float(np.mean(values))
    sd_s = sample_sd(values)
    variance_s2 = sd_s * sd_s if math.isfinite(sd_s) else math.nan
    cv = sd_s / mean_s if math.isfinite(sd_s) and mean_s > 0 else math.nan
    return {"mean_s": mean_s, "sd_s": sd_s, "variance_s2": variance_s2, "cv": cv}


# -----------------------------------------------------------------------------
# Equal-count period metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedPairMetrics:
    n_pre_phrases: int
    n_post_phrases: int
    n_pre_days: int
    n_post_days: int
    n_balance: int
    n_successful_draws: int
    pre_mean_s: float
    post_mean_s: float
    pre_sd_s: float
    post_sd_s: float
    pre_variance_s2: float
    post_variance_s2: float
    pre_cv: float
    post_cv: float
    delta_mean_s: float
    delta_sd_s: float
    delta_variance_s2: float
    delta_cv: float
    pooled_variance_s2: float


def balanced_pair_metrics(
    pre_day_map: Mapping[date, Sequence[float]],
    post_day_map: Mapping[date, Sequence[float]],
    *,
    min_phrases: int,
    min_days: int,
    draws: int,
    rng: np.random.Generator,
) -> BalancedPairMetrics | None:
    pre = flatten_day_map(pre_day_map)
    post = flatten_day_map(post_day_map)
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
    for _ in range(draws):
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
        if not all(math.isfinite(x) for x in [pre_mean, post_mean, pre_sd, post_sd]):
            continue
        if pre_mean <= 0 or post_mean <= 0:
            continue

        pre_variance = pre_sd * pre_sd
        post_variance = post_sd * post_sd
        pre_cv = pre_sd / pre_mean
        post_cv = post_sd / post_mean
        values = {
            "pre_mean_s": pre_mean,
            "post_mean_s": post_mean,
            "pre_sd_s": pre_sd,
            "post_sd_s": post_sd,
            "pre_variance_s2": pre_variance,
            "post_variance_s2": post_variance,
            "pre_cv": pre_cv,
            "post_cv": post_cv,
            "delta_mean_s": post_mean - pre_mean,
            "delta_sd_s": post_sd - pre_sd,
            "delta_variance_s2": post_variance - pre_variance,
            "delta_cv": post_cv - pre_cv,
            "pooled_variance_s2": 0.5 * (pre_variance + post_variance),
        }
        if all(math.isfinite(value) for value in values.values()):
            for key, value in values.items():
                draw_values[key].append(value)

    n_successful = len(draw_values["delta_cv"])
    if n_successful == 0:
        return None

    return BalancedPairMetrics(
        n_pre_phrases=int(pre.size),
        n_post_phrases=int(post.size),
        n_pre_days=int(n_pre_days),
        n_post_days=int(n_post_days),
        n_balance=int(n_balance),
        n_successful_draws=int(n_successful),
        pre_mean_s=safe_mean(draw_values["pre_mean_s"]),
        post_mean_s=safe_mean(draw_values["post_mean_s"]),
        pre_sd_s=safe_mean(draw_values["pre_sd_s"]),
        post_sd_s=safe_mean(draw_values["post_sd_s"]),
        pre_variance_s2=safe_mean(draw_values["pre_variance_s2"]),
        post_variance_s2=safe_mean(draw_values["post_variance_s2"]),
        pre_cv=safe_mean(draw_values["pre_cv"]),
        post_cv=safe_mean(draw_values["post_cv"]),
        delta_mean_s=safe_mean(draw_values["delta_mean_s"]),
        delta_sd_s=safe_mean(draw_values["delta_sd_s"]),
        delta_variance_s2=safe_mean(draw_values["delta_variance_s2"]),
        delta_cv=safe_mean(draw_values["delta_cv"]),
        pooled_variance_s2=safe_mean(draw_values["pooled_variance_s2"]),
    )


def period_row(
    *,
    bird: str,
    label: str,
    lesion_group: str,
    surgery_date: date,
    period: str,
    metrics: BalancedPairMetrics,
    pre_start_day: int,
    pre_end_day: int,
    post_start_day: int,
    post_end_day: int,
    source_database: Path,
) -> dict[str, Any]:
    if period == "Late Pre":
        n_phrases = metrics.n_pre_phrases
        n_days = metrics.n_pre_days
        mean_s = metrics.pre_mean_s
        sd_s = metrics.pre_sd_s
        variance_s2 = metrics.pre_variance_s2
        cv = metrics.pre_cv
        start_day, end_day = pre_start_day, pre_end_day
    elif period == "Post":
        n_phrases = metrics.n_post_phrases
        n_days = metrics.n_post_days
        mean_s = metrics.post_mean_s
        sd_s = metrics.post_sd_s
        variance_s2 = metrics.post_variance_s2
        cv = metrics.post_cv
        start_day, end_day = post_start_day, post_end_day
    else:
        raise ValueError(period)

    return {
        "Animal ID": bird,
        "Syllable": label,
        "Group": period,
        "N_phrases": n_phrases,
        "N_days": n_days,
        "N_balance": metrics.n_balance,
        "N_successful_draws": metrics.n_successful_draws,
        "Mean_ms": mean_s * 1000.0,
        "SD_ms": sd_s * 1000.0,
        "Variance_ms2": variance_s2 * 1_000_000.0,
        "CV": cv,
        "Mean_s": mean_s,
        "SD_s": sd_s,
        "Variance_s2": variance_s2,
        "lesion_group": lesion_group,
        "lesion_surgery_date": surgery_date.isoformat(),
        "period_start_day": start_day,
        "period_end_day": end_day,
        "balanced": True,
        "eligible": True,
        "source_database": str(source_database),
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


LONG_FIELDS = [
    "animal_id",
    "syllable",
    "recording_date",
    "creation_date",
    "relative_day",
    "Group",
    "phrase_duration_ms",
    "phrase_duration_s",
    "lesion_group",
    "lesion_surgery_date",
    "source_database",
]

PERIOD_FIELDS = [
    "Animal ID",
    "Syllable",
    "Group",
    "N_phrases",
    "N_days",
    "N_balance",
    "N_successful_draws",
    "Mean_ms",
    "SD_ms",
    "Variance_ms2",
    "CV",
    "Mean_s",
    "SD_s",
    "Variance_s2",
    "lesion_group",
    "lesion_surgery_date",
    "period_start_day",
    "period_end_day",
    "balanced",
    "eligible",
    "source_database",
]

PAIR_FIELDS = [
    "animal_id",
    "syllable",
    "lesion_group",
    "lesion_surgery_date",
    "n_pre_phrases",
    "n_post_phrases",
    "n_pre_days",
    "n_post_days",
    "n_balance",
    "n_successful_draws",
    "pre_mean_s",
    "post_mean_s",
    "delta_mean_s",
    "pre_sd_s",
    "post_sd_s",
    "delta_sd_s",
    "pre_variance_s2",
    "post_variance_s2",
    "delta_variance_s2",
    "pooled_variance_s2",
    "pre_cv",
    "post_cv",
    "delta_cv",
    "pre_mean_ms",
    "post_mean_ms",
    "pre_sd_ms",
    "post_sd_ms",
    "pre_variance_ms2",
    "post_variance_ms2",
    "pooled_variance_ms2",
    "pre_start_day",
    "pre_end_day",
    "post_start_day",
    "post_end_day",
    "source_database",
]

AUDIT_FIELDS = [
    "animal_id",
    "lesion_group",
    "lesion_surgery_date",
    "source_database",
    "n_results",
    "n_results_with_valid_date",
    "n_results_in_long_window",
    "n_results_in_pre_window",
    "n_results_in_post_window",
    "n_valid_phrase_spans",
    "n_invalid_phrase_spans",
    "n_phrase_rows_written",
    "n_labels_anywhere",
    "n_labels_in_pre",
    "n_labels_in_post",
    "n_labels_in_both",
    "n_eligible_pairs",
]


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata_path = args.json_dir / "AFP_lesion_bird_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected a bird-keyed metadata object in {metadata_path}")

    database_paths = find_database_paths(args.json_dir)
    if not database_paths:
        raise FileNotFoundError(
            f"No decoded databases were found under {args.json_dir}. Expected files ending in decoded_database.json."
        )

    long_path = args.output / "figure3_phrase_duration_long.csv"
    period_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    with long_path.open("w", newline="", encoding="utf-8") as long_handle:
        long_writer = csv.DictWriter(long_handle, fieldnames=LONG_FIELDS)
        long_writer.writeheader()

        for database_path in database_paths:
            bird = database_path.stem.removesuffix("_decoded_database")
            if bird not in metadata:
                raise KeyError(f"Bird {bird!r} is missing from {metadata_path}")
            bird_meta = metadata[bird]
            if not isinstance(bird_meta, Mapping):
                raise TypeError(f"Metadata entry for {bird!r} is not an object")
            if "lesion_surgery_date" not in bird_meta:
                raise KeyError(f"Bird {bird!r} lacks lesion_surgery_date in {metadata_path}")

            surgery_date = parse_iso_date(bird_meta["lesion_surgery_date"])
            lesion_group = str(bird_meta.get("lesion_group", "unknown"))
            payload = json.loads(database_path.read_text(encoding="utf-8"))
            results = payload.get("results", [])
            if not isinstance(results, list):
                raise TypeError(f"Top-level results in {database_path} is not a list")

            durations: dict[str, dict[str, dict[date, list[float]]]] = {
                "Late Pre": defaultdict(lambda: defaultdict(list)),
                "Post": defaultdict(lambda: defaultdict(list)),
            }
            labels_anywhere: set[str] = set()
            n_results_valid_date = 0
            n_results_long = 0
            n_results_pre = 0
            n_results_post = 0
            n_valid_spans = 0
            n_invalid_spans = 0
            n_long_rows = 0

            for result in results:
                try:
                    recording_date = parse_iso_date(result.get("creation_date"))
                except Exception:
                    continue
                n_results_valid_date += 1
                relative_day = (recording_date - surgery_date).days
                in_long = args.long_start_day <= relative_day <= args.long_end_day
                if in_long:
                    n_results_long += 1
                if args.pre_start_day <= relative_day <= args.pre_end_day:
                    analysis_period = "Late Pre"
                    n_results_pre += 1
                elif args.post_start_day <= relative_day <= args.post_end_day:
                    analysis_period = "Post"
                    n_results_post += 1
                else:
                    analysis_period = "Other"

                span_map = result.get("syllable_onsets_offsets_ms", {})
                if not isinstance(span_map, Mapping):
                    continue
                for raw_label, spans in span_map.items():
                    label = str(raw_label)
                    labels_anywhere.add(label)
                    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
                        continue
                    for span in spans:
                        try:
                            if not isinstance(span, Sequence) or len(span) < 2:
                                raise ValueError("malformed span")
                            start_ms = float(span[0])
                            end_ms = float(span[1])
                            duration_ms = end_ms - start_ms
                            if not math.isfinite(duration_ms) or duration_ms <= 0:
                                raise ValueError("non-positive duration")
                        except Exception:
                            n_invalid_spans += 1
                            continue

                        n_valid_spans += 1
                        duration_s = duration_ms / 1000.0
                        if analysis_period in {"Late Pre", "Post"}:
                            durations[analysis_period][label][recording_date].append(duration_s)
                        if in_long:
                            long_writer.writerow(
                                {
                                    "animal_id": bird,
                                    "syllable": label,
                                    "recording_date": recording_date.isoformat(),
                                    "creation_date": str(result.get("creation_date", "")),
                                    "relative_day": relative_day,
                                    "Group": analysis_period,
                                    "phrase_duration_ms": duration_ms,
                                    "phrase_duration_s": duration_s,
                                    "lesion_group": lesion_group,
                                    "lesion_surgery_date": surgery_date.isoformat(),
                                    "source_database": str(database_path),
                                }
                            )
                            n_long_rows += 1

            pre_labels = set(durations["Late Pre"])
            post_labels = set(durations["Post"])
            common_labels = sorted(pre_labels & post_labels, key=label_sort_key)
            n_eligible = 0

            for label in common_labels:
                pre_map = durations["Late Pre"][label]
                post_map = durations["Post"][label]
                metrics = balanced_pair_metrics(
                    pre_map,
                    post_map,
                    min_phrases=args.min_period_phrases,
                    min_days=args.min_period_days,
                    draws=args.balance_draws,
                    # Match the full-data metric seed used by the cutoff-scan script.
                    rng=stable_rng(args.seed, "balanced", bird, label, "full"),
                )
                if metrics is None:
                    if args.include_ineligible_period_rows:
                        for period, day_map, start_day, end_day in [
                            ("Late Pre", pre_map, args.pre_start_day, args.pre_end_day),
                            ("Post", post_map, args.post_start_day, args.post_end_day),
                        ]:
                            raw = flatten_day_map(day_map)
                            desc = descriptive_metrics(raw)
                            period_rows.append(
                                {
                                    "Animal ID": bird,
                                    "Syllable": label,
                                    "Group": period,
                                    "N_phrases": int(raw.size),
                                    "N_days": sum(bool(v) for v in day_map.values()),
                                    "N_balance": min(
                                        flatten_day_map(pre_map).size,
                                        flatten_day_map(post_map).size,
                                    ),
                                    "N_successful_draws": 0,
                                    "Mean_ms": desc["mean_s"] * 1000.0,
                                    "SD_ms": desc["sd_s"] * 1000.0,
                                    "Variance_ms2": desc["variance_s2"] * 1_000_000.0,
                                    "CV": desc["cv"],
                                    "Mean_s": desc["mean_s"],
                                    "SD_s": desc["sd_s"],
                                    "Variance_s2": desc["variance_s2"],
                                    "lesion_group": lesion_group,
                                    "lesion_surgery_date": surgery_date.isoformat(),
                                    "period_start_day": start_day,
                                    "period_end_day": end_day,
                                    "balanced": False,
                                    "eligible": False,
                                    "source_database": str(database_path),
                                }
                            )
                    continue

                n_eligible += 1
                period_rows.append(
                    period_row(
                        bird=bird,
                        label=label,
                        lesion_group=lesion_group,
                        surgery_date=surgery_date,
                        period="Late Pre",
                        metrics=metrics,
                        pre_start_day=args.pre_start_day,
                        pre_end_day=args.pre_end_day,
                        post_start_day=args.post_start_day,
                        post_end_day=args.post_end_day,
                        source_database=database_path,
                    )
                )
                period_rows.append(
                    period_row(
                        bird=bird,
                        label=label,
                        lesion_group=lesion_group,
                        surgery_date=surgery_date,
                        period="Post",
                        metrics=metrics,
                        pre_start_day=args.pre_start_day,
                        pre_end_day=args.pre_end_day,
                        post_start_day=args.post_start_day,
                        post_end_day=args.post_end_day,
                        source_database=database_path,
                    )
                )
                pair_rows.append(
                    {
                        "animal_id": bird,
                        "syllable": label,
                        "lesion_group": lesion_group,
                        "lesion_surgery_date": surgery_date.isoformat(),
                        **metrics.__dict__,
                        "pre_mean_ms": metrics.pre_mean_s * 1000.0,
                        "post_mean_ms": metrics.post_mean_s * 1000.0,
                        "pre_sd_ms": metrics.pre_sd_s * 1000.0,
                        "post_sd_ms": metrics.post_sd_s * 1000.0,
                        "pre_variance_ms2": metrics.pre_variance_s2 * 1_000_000.0,
                        "post_variance_ms2": metrics.post_variance_s2 * 1_000_000.0,
                        "pooled_variance_ms2": metrics.pooled_variance_s2 * 1_000_000.0,
                        "pre_start_day": args.pre_start_day,
                        "pre_end_day": args.pre_end_day,
                        "post_start_day": args.post_start_day,
                        "post_end_day": args.post_end_day,
                        "source_database": str(database_path),
                    }
                )

            audit_rows.append(
                {
                    "animal_id": bird,
                    "lesion_group": lesion_group,
                    "lesion_surgery_date": surgery_date.isoformat(),
                    "source_database": str(database_path),
                    "n_results": len(results),
                    "n_results_with_valid_date": n_results_valid_date,
                    "n_results_in_long_window": n_results_long,
                    "n_results_in_pre_window": n_results_pre,
                    "n_results_in_post_window": n_results_post,
                    "n_valid_phrase_spans": n_valid_spans,
                    "n_invalid_phrase_spans": n_invalid_spans,
                    "n_phrase_rows_written": n_long_rows,
                    "n_labels_anywhere": len(labels_anywhere),
                    "n_labels_in_pre": len(pre_labels),
                    "n_labels_in_post": len(post_labels),
                    "n_labels_in_both": len(common_labels),
                    "n_eligible_pairs": n_eligible,
                }
            )
            print(
                f"[INFO] {bird}: {n_eligible} eligible syllables; "
                f"{n_long_rows:,} occurrence rows written"
            )

    period_path = args.output / "figure3_balanced_period_stats.csv"
    pair_path = args.output / "figure3_balanced_pair_metrics.csv"
    audit_path = args.output / "figure3_input_audit.csv"
    write_csv(period_path, period_rows, PERIOD_FIELDS)
    write_csv(pair_path, pair_rows, PAIR_FIELDS)
    write_csv(audit_path, audit_rows, AUDIT_FIELDS)

    eligible_birds = sorted({str(row["animal_id"]) for row in pair_rows})
    summary_lines = [
        "Figure 3 input generation summary",
        "=================================",
        "",
        f"Input directory: {args.json_dir}",
        f"Decoded databases found: {len(database_paths)}",
        f"Birds with at least one eligible syllable: {len(eligible_birds)}",
        f"Eligible bird x syllable pairs: {len(pair_rows)}",
        f"Balanced period rows: {sum(bool(row.get('eligible')) for row in period_rows)}",
        f"Occurrence-level rows: {sum(int(row['n_phrase_rows_written']) for row in audit_rows):,}",
        "",
        f"Late Pre window: days {args.pre_start_day} to {args.pre_end_day}",
        f"Post window: days {args.post_start_day} to {args.post_end_day}",
        f"Long-table window: days {args.long_start_day} to {args.long_end_day}",
        f"Minimum phrases per period: {args.min_period_phrases}",
        f"Minimum days per period: {args.min_period_days}",
        f"Equal-count balancing draws: {args.balance_draws}",
        f"Seed: {args.seed}",
        "",
        "Use these files with the plotting script:",
        f"  --scatter-csv \"{period_path}\"",
        f"  --duration-long-csv \"{long_path}\"",
        "",
        "The scatter/statistics table contains one balanced Late Pre row and one",
        "balanced Post row for every eligible bird x syllable. The long table",
        "retains all valid phrase occurrences in the requested longitudinal window.",
    ]
    summary_path = args.output / "figure3_input_generation_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    config_path = args.output / "figure3_input_generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "json_dir": str(args.json_dir),
                "output": str(args.output),
                "pre_start_day": args.pre_start_day,
                "pre_end_day": args.pre_end_day,
                "post_start_day": args.post_start_day,
                "post_end_day": args.post_end_day,
                "long_start_day": args.long_start_day,
                "long_end_day": args.long_end_day,
                "min_period_phrases": args.min_period_phrases,
                "min_period_days": args.min_period_days,
                "balance_draws": args.balance_draws,
                "seed": args.seed,
                "include_ineligible_period_rows": args.include_ineligible_period_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[OK] Wrote Figure 3 input files:")
    for path in [period_path, long_path, pair_path, audit_path, summary_path, config_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
