"""Run two phrase-duration variance tests directly from the lesion JSON files.

Test 1 uses every eligible syllable, reduces each bird to one median log
post/pre variance ratio, and compares medial-lesion birds with all other birds.

Test 3 alternates pre and post recording dates between two folds. Each fold is
used once to select increased-variance syllables and once to measure the
selected syllables on held-out dates. The reported affected-syllable percentage
is the discovery-fold selection rate; held-out variance is reported separately.

Usage:
    python run_simple_variance_tests.py /path/to/AFP_lesion_jsons

Required package:
    numpy
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np


TEST1_MIN_PHRASES = 20
CROSSFIT_MIN_PHRASES = 25
CROSSFIT_MIN_DAYS = 2
HELDOUT_MIN_PHRASES = 20
FDR = 0.05


def sample_variance(stats):
    count, total, squares = stats.sum(axis=0)
    return (squares - total * total / count) / (count - 1)


def day_stats(days):
    return np.asarray(
        [[len(values), sum(values), sum(x * x for x in values)] for values in days]
    )


def log_variance_ratio(pre_days, post_days):
    pre_variance = sample_variance(day_stats(pre_days))
    post_variance = sample_variance(day_stats(post_days))
    return np.log(post_variance / pre_variance)


def day_permutation_p(pre_days, post_days):
    stats = day_stats(pre_days + post_days)
    observed = log_variance_ratio(pre_days, post_days)
    extreme = 0
    total = 0
    for chosen in combinations(range(len(stats)), len(pre_days)):
        mask = np.zeros(len(stats), dtype=bool)
        mask[list(chosen)] = True
        null = np.log(
            sample_variance(stats[~mask]) / sample_variance(stats[mask])
        )
        extreme += null >= observed - 1e-15
        total += 1
    return extreme / total


def bh_adjust(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return adjusted


def exact_group_test(medial, control):
    pooled = np.r_[medial, control]
    observed = abs(medial.mean() - control.mean())
    extreme = 0
    total = 0
    for chosen in combinations(range(len(pooled)), len(medial)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        difference = abs(pooled[mask].mean() - pooled[~mask].mean())
        extreme += difference >= observed - 1e-15
        total += 1
    return extreme / total


def signflip_test(values):
    observed = abs(values.mean())
    null = [
        abs(np.mean(values * np.asarray(signs)))
        for signs in product((-1, 1), repeat=len(values))
    ]
    return np.mean(np.asarray(null) >= observed - 1e-15)


def write_tsv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser(
    description="Run two non-circular phrase-duration variance tests."
)
parser.add_argument(
    "json_dir",
    type=Path,
    help="Directory containing AFP_lesion_bird_metadata.json and group folders.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("simple_variance_test_results"),
    help="Output directory (default: simple_variance_test_results).",
)
args = parser.parse_args()

metadata_path = args.json_dir / "AFP_lesion_bird_metadata.json"
assert metadata_path.exists(), metadata_path
metadata = json.loads(metadata_path.read_text())
database_paths = sorted(args.json_dir.glob("*/*/*decoded_database.json"))
assert database_paths, f"No decoded databases found under {args.json_dir}"

test1_syllable_rows = []
test1_bird_rows = []
crossfit_syllable_rows = []
crossfit_bird_rows = []

for path in database_paths:
    bird = path.stem.removesuffix("_decoded_database")
    group = metadata[bird]["lesion_group"]
    surgery = datetime.fromisoformat(
        metadata[bird]["lesion_surgery_date"]
    ).date()
    durations = {
        "pre": defaultdict(lambda: defaultdict(list)),
        "post": defaultdict(lambda: defaultdict(list)),
    }

    for result in json.loads(path.read_text())["results"]:
        date = datetime.fromisoformat(result["creation_date"]).date()
        day = (date - surgery).days
        period = "pre" if -14 <= day <= -1 else "post" if 1 <= day <= 14 else None
        if period is None:
            continue
        for label, spans in result["syllable_onsets_offsets_ms"].items():
            durations[period][label][date].extend(
                (end - start) / 1000 for start, end in spans
            )

    common_labels = sorted(
        durations["pre"].keys() & durations["post"].keys(), key=int
    )

    bird_test1 = []
    for label in common_labels:
        pre = np.concatenate(list(durations["pre"][label].values()))
        post = np.concatenate(list(durations["post"][label].values()))
        if len(pre) < TEST1_MIN_PHRASES or len(post) < TEST1_MIN_PHRASES:
            continue
        pre_sd = pre.std(ddof=1)
        if pre_sd == 0:
            continue
        pre_z = (pre - pre.mean()) / pre_sd
        post_z = (post - pre.mean()) / pre_sd
        pre_z_variance = pre_z.var(ddof=1)
        post_z_variance = post_z.var(ddof=1)
        variance_ratio = post_z_variance / pre_z_variance
        row = {
            "bird": bird,
            "group": group,
            "syllable": label,
            "n_pre": len(pre),
            "n_post": len(post),
            "pre_z_variance": pre_z_variance,
            "post_z_variance": post_z_variance,
            "variance_ratio": variance_ratio,
            "log_variance_ratio": np.log(variance_ratio),
        }
        bird_test1.append(row)
        test1_syllable_rows.append(row)

    assert bird_test1
    test1_bird_rows.append(
        {
            "bird": bird,
            "group": group,
            "n_syllables": len(bird_test1),
            "median_log_variance_ratio": np.median(
                [row["log_variance_ratio"] for row in bird_test1]
            ),
        }
    )

    dates = {
        period: sorted(
            {
                date
                for label in durations[period]
                for date in durations[period][label]
            }
        )
        for period in ("pre", "post")
    }
    date_fold = {
        period: {date: index % 2 for index, date in enumerate(dates[period])}
        for period in ("pre", "post")
    }
    fold_proportions = []
    fold_heldout_medians = []

    for discovery_fold in (0, 1):
        candidate_rows = []
        for label in common_labels:
            discovery = {
                period: [
                    values
                    for date, values in sorted(durations[period][label].items())
                    if date_fold[period][date] == discovery_fold
                ]
                for period in ("pre", "post")
            }
            if (
                sum(map(len, discovery["pre"])) < CROSSFIT_MIN_PHRASES
                or sum(map(len, discovery["post"])) < CROSSFIT_MIN_PHRASES
                or len(discovery["pre"]) < CROSSFIT_MIN_DAYS
                or len(discovery["post"]) < CROSSFIT_MIN_DAYS
            ):
                continue
            candidate_rows.append(
                {
                    "bird": bird,
                    "group": group,
                    "discovery_fold": discovery_fold,
                    "syllable": label,
                    "n_pre_discovery": sum(map(len, discovery["pre"])),
                    "n_post_discovery": sum(map(len, discovery["post"])),
                    "log_variance_ratio_discovery": log_variance_ratio(
                        discovery["pre"], discovery["post"]
                    ),
                    "day_permutation_p": day_permutation_p(
                        discovery["pre"], discovery["post"]
                    ),
                    "bh_q_within_bird": 0.0,
                    "selected": False,
                    "heldout_log_variance_ratio": np.nan,
                }
            )

        if not candidate_rows:
            continue
        adjusted = bh_adjust(
            [row["day_permutation_p"] for row in candidate_rows]
        )
        for row, q_value in zip(candidate_rows, adjusted):
            row["bh_q_within_bird"] = q_value
            row["selected"] = (
                q_value < FDR
                and row["log_variance_ratio_discovery"] > 0
            )
            if not row["selected"]:
                continue
            label = row["syllable"]
            heldout = {
                period: [
                    values
                    for date, values in sorted(durations[period][label].items())
                    if date_fold[period][date] != discovery_fold
                ]
                for period in ("pre", "post")
            }
            if (
                sum(map(len, heldout["pre"])) >= HELDOUT_MIN_PHRASES
                and sum(map(len, heldout["post"])) >= HELDOUT_MIN_PHRASES
            ):
                row["heldout_log_variance_ratio"] = log_variance_ratio(
                    heldout["pre"], heldout["post"]
                )

        crossfit_syllable_rows.extend(candidate_rows)
        selected = [row for row in candidate_rows if row["selected"]]
        fold_proportions.append(len(selected) / len(candidate_rows))
        heldout_values = [
            row["heldout_log_variance_ratio"]
            for row in selected
            if np.isfinite(row["heldout_log_variance_ratio"])
        ]
        if heldout_values:
            fold_heldout_medians.append(np.median(heldout_values))

    crossfit_bird_rows.append(
        {
            "bird": bird,
            "group": group,
            "mean_selected_proportion": (
                np.mean(fold_proportions) if fold_proportions else np.nan
            ),
            "mean_heldout_median_log_variance_ratio": (
                np.mean(fold_heldout_medians)
                if fold_heldout_medians
                else np.nan
            ),
        }
    )


def group_values(rows, key):
    medial = np.asarray(
        [
            row[key]
            for row in rows
            if row["group"] == "medial_and_lateral"
            and np.isfinite(row[key])
        ]
    )
    control = np.asarray(
        [
            row[key]
            for row in rows
            if row["group"] != "medial_and_lateral"
            and np.isfinite(row[key])
        ]
    )
    return medial, control


test1_medial, test1_control = group_values(
    test1_bird_rows, "median_log_variance_ratio"
)
test1_difference = test1_medial.mean() - test1_control.mean()

crossfit_medial, crossfit_control = group_values(
    crossfit_bird_rows, "mean_selected_proportion"
)
heldout_medial, heldout_control = group_values(
    crossfit_bird_rows, "mean_heldout_median_log_variance_ratio"
)

args.output.mkdir(parents=True, exist_ok=True)
write_tsv(args.output / "test1_syllables.tsv", test1_syllable_rows)
write_tsv(args.output / "test1_birds.tsv", test1_bird_rows)
write_tsv(args.output / "test3_crossfit_syllables.tsv", crossfit_syllable_rows)
write_tsv(args.output / "test3_crossfit_birds.tsv", crossfit_bird_rows)

summary = (
    "TEST 1: ALL-SYLLABLE BIRD-LEVEL TEST\n"
    f"Medial mean bird median log variance ratio: {test1_medial.mean():.6f}\n"
    f"No-medial mean: {test1_control.mean():.6f}\n"
    f"Difference: {test1_difference:.6f}\n"
    f"Ratio of variance changes: {np.exp(test1_difference):.6f}\n"
    f"Exact two-sided bird-label permutation p: "
    f"{exact_group_test(test1_medial, test1_control):.6f}\n\n"
    "TEST 3: TWO-FOLD CROSS-FITTED SELECTION\n"
    f"Medial mean discovery-selected proportion: {crossfit_medial.mean():.6f}\n"
    f"No-medial mean discovery-selected proportion: {crossfit_control.mean():.6f}\n"
    f"Difference: {crossfit_medial.mean() - crossfit_control.mean():.6f}\n"
    f"Exact two-sided bird-label permutation p: "
    f"{exact_group_test(crossfit_medial, crossfit_control):.6f}\n\n"
    "HELD-OUT CONFIRMATION OF SELECTED SYLLABLES\n"
    f"Medial mean held-out summary: {heldout_medial.mean():.6f}\n"
    f"No-medial mean held-out summary: {heldout_control.mean():.6f}\n"
    f"Medial-versus-no-medial p: "
    f"{exact_group_test(heldout_medial, heldout_control):.6f}\n"
    f"Medial sign-flip p versus zero: {signflip_test(heldout_medial):.6f}\n"
)
(args.output / "summary.txt").write_text(summary)
print(summary)
