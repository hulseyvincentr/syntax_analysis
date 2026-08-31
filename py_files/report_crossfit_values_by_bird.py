#!/usr/bin/env python3
"""
Report fold-level and bird-level values from run_simple_variance_tests.py.

This script reconstructs the two discovery-fold summaries for every bird from
test3_crossfit_syllables.tsv, checks them against test3_crossfit_birds.tsv, and
optionally audits which decoded-database files were discovered in the input
directory.

Usage
-----
Results only:
    python report_crossfit_values_by_bird.py simple_variance_test_results

Results plus input audit:
    python report_crossfit_values_by_bird.py simple_variance_test_results \
        --json-dir /path/to/AFP_lesion_jsons

Outputs are written into the results directory:
    crossfit_fold_values_by_bird.tsv
    crossfit_bird_values_detailed.tsv
    crossfit_input_audit.tsv          (only with --json-dir)
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def fmt(value: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show fold-level and final cross-fitted values for every bird."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory produced by run_simple_variance_tests.py.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help=(
            "Optional original input directory. When supplied, audit all files "
            "matching */*/*decoded_database.json and compare them with output birds."
        ),
    )
    args = parser.parse_args()

    syllable_path = args.results_dir / "test3_crossfit_syllables.tsv"
    bird_path = args.results_dir / "test3_crossfit_birds.tsv"

    syllable_rows = read_tsv(syllable_path)
    saved_bird_rows = read_tsv(bird_path)

    # Group every candidate syllable by bird and discovery fold.
    rows_by_bird_fold: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    groups_by_bird: dict[str, str] = {}

    for row in syllable_rows:
        bird = row["bird"]
        group = row["group"]
        fold = int(row["discovery_fold"])
        groups_by_bird[bird] = group
        rows_by_bird_fold[(bird, group, fold)].append(row)

    # The bird-level file may include birds with no Test 3 candidates.
    for row in saved_bird_rows:
        groups_by_bird[row["bird"]] = row["group"]

    fold_rows: list[dict[str, Any]] = []
    fold_summaries: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    for (bird, group, fold), rows in sorted(rows_by_bird_fold.items()):
        selected_rows = [row for row in rows if parse_bool(row["selected"])]
        heldout_values = finite(
            [parse_float(row.get("heldout_log_variance_ratio")) for row in selected_rows]
        )
        heldout_median = (
            statistics.median(heldout_values) if heldout_values else math.nan
        )

        selected_labels = ",".join(row["syllable"] for row in selected_rows)
        heldout_labels = ",".join(
            row["syllable"]
            for row in selected_rows
            if math.isfinite(parse_float(row.get("heldout_log_variance_ratio")))
        )

        summary = {
            "bird": bird,
            "group": group,
            "discovery_fold": fold,
            "n_candidate_syllables": len(rows),
            "n_selected_syllables": len(selected_rows),
            "selected_proportion": len(selected_rows) / len(rows),
            "selected_percent": 100.0 * len(selected_rows) / len(rows),
            "selected_syllable_labels": selected_labels,
            "n_selected_with_heldout_value": len(heldout_values),
            "heldout_syllable_labels": heldout_labels,
            "heldout_median_log_variance_ratio": heldout_median,
            "heldout_median_variance_ratio": (
                math.exp(heldout_median)
                if math.isfinite(heldout_median)
                else math.nan
            ),
        }
        fold_rows.append(summary)
        fold_summaries[bird][fold] = summary

    saved_by_bird = {row["bird"]: row for row in saved_bird_rows}
    detailed_bird_rows: list[dict[str, Any]] = []

    for bird in sorted(groups_by_bird):
        group = groups_by_bird[bird]
        folds = fold_summaries.get(bird, {})

        fold_props = [
            folds[fold]["selected_proportion"]
            for fold in sorted(folds)
        ]
        fold_heldout_medians = finite(
            [
                folds[fold]["heldout_median_log_variance_ratio"]
                for fold in sorted(folds)
            ]
        )

        recalculated_selected = (
            statistics.mean(fold_props) if fold_props else math.nan
        )
        recalculated_heldout = (
            statistics.mean(fold_heldout_medians)
            if fold_heldout_medians
            else math.nan
        )

        saved = saved_by_bird.get(bird, {})
        saved_selected = parse_float(saved.get("mean_selected_proportion"))
        saved_heldout = parse_float(
            saved.get("mean_heldout_median_log_variance_ratio")
        )

        def fold_value(fold: int, key: str) -> Any:
            return folds.get(fold, {}).get(key, math.nan)

        detailed_bird_rows.append(
            {
                "bird": bird,
                "group": group,
                "n_discovery_folds_with_candidates": len(folds),
                "fold0_n_candidates": fold_value(0, "n_candidate_syllables"),
                "fold0_n_selected": fold_value(0, "n_selected_syllables"),
                "fold0_selected_proportion": fold_value(
                    0, "selected_proportion"
                ),
                "fold0_selected_percent": fold_value(0, "selected_percent"),
                "fold0_selected_labels": folds.get(0, {}).get(
                    "selected_syllable_labels", ""
                ),
                "fold0_heldout_median_log_variance_ratio": fold_value(
                    0, "heldout_median_log_variance_ratio"
                ),
                "fold1_n_candidates": fold_value(1, "n_candidate_syllables"),
                "fold1_n_selected": fold_value(1, "n_selected_syllables"),
                "fold1_selected_proportion": fold_value(
                    1, "selected_proportion"
                ),
                "fold1_selected_percent": fold_value(1, "selected_percent"),
                "fold1_selected_labels": folds.get(1, {}).get(
                    "selected_syllable_labels", ""
                ),
                "fold1_heldout_median_log_variance_ratio": fold_value(
                    1, "heldout_median_log_variance_ratio"
                ),
                "recalculated_mean_selected_proportion": recalculated_selected,
                "recalculated_mean_selected_percent": (
                    100.0 * recalculated_selected
                    if math.isfinite(recalculated_selected)
                    else math.nan
                ),
                "saved_mean_selected_proportion": saved_selected,
                "selected_value_matches_saved": (
                    abs(recalculated_selected - saved_selected) < 1e-12
                    if (
                        math.isfinite(recalculated_selected)
                        and math.isfinite(saved_selected)
                    )
                    else (
                        not math.isfinite(recalculated_selected)
                        and not math.isfinite(saved_selected)
                    )
                ),
                "recalculated_mean_heldout_median_log_variance_ratio": (
                    recalculated_heldout
                ),
                "recalculated_mean_heldout_variance_ratio": (
                    math.exp(recalculated_heldout)
                    if math.isfinite(recalculated_heldout)
                    else math.nan
                ),
                "saved_mean_heldout_median_log_variance_ratio": saved_heldout,
                "heldout_value_matches_saved": (
                    abs(recalculated_heldout - saved_heldout) < 1e-12
                    if (
                        math.isfinite(recalculated_heldout)
                        and math.isfinite(saved_heldout)
                    )
                    else (
                        not math.isfinite(recalculated_heldout)
                        and not math.isfinite(saved_heldout)
                    )
                ),
            }
        )

    fold_output = args.results_dir / "crossfit_fold_values_by_bird.tsv"
    bird_output = args.results_dir / "crossfit_bird_values_detailed.tsv"
    write_tsv(fold_output, fold_rows)
    write_tsv(bird_output, detailed_bird_rows)

    print("\nCROSS-FITTED VALUES BY BIRD")
    print("=" * 90)
    for row in detailed_bird_rows:
        print(f"\n{row['bird']}  [{row['group']}]")
        for fold in (0, 1):
            candidates = row[f"fold{fold}_n_candidates"]
            selected = row[f"fold{fold}_n_selected"]
            proportion = row[f"fold{fold}_selected_proportion"]
            labels = row[f"fold{fold}_selected_labels"]

            if isinstance(candidates, float) and not math.isfinite(candidates):
                print(f"  Fold {fold}: no eligible candidate syllables")
            else:
                print(
                    f"  Fold {fold}: {selected}/{candidates} selected "
                    f"= {fmt(100.0 * proportion, 2)}%"
                )
                print(f"           selected labels: {labels or 'none'}")

        print(
            "  Final bird mean selected proportion: "
            f"{fmt(row['recalculated_mean_selected_proportion'])} "
            f"({fmt(row['recalculated_mean_selected_percent'], 2)}%)"
        )
        print(
            "  Final bird mean held-out log variance ratio: "
            f"{fmt(row['recalculated_mean_heldout_median_log_variance_ratio'])}"
        )

    print("\nFiles written:")
    print(f"  {fold_output}")
    print(f"  {bird_output}")

    if args.json_dir is not None:
        database_paths = sorted(
            args.json_dir.glob("*/*/*decoded_database.json")
        )
        output_birds = set(groups_by_bird)
        audit_rows: list[dict[str, Any]] = []

        for path in database_paths:
            bird = path.stem.removesuffix("_decoded_database")
            audit_rows.append(
                {
                    "bird": bird,
                    "database_path": str(path),
                    "found_by_original_glob": True,
                    "present_in_crossfit_bird_output": bird in output_birds,
                    "status": (
                        "present in output"
                        if bird in output_birds
                        else "found as input but absent from output"
                    ),
                }
            )

        input_birds = {
            path.stem.removesuffix("_decoded_database")
            for path in database_paths
        }
        for bird in sorted(output_birds - input_birds):
            audit_rows.append(
                {
                    "bird": bird,
                    "database_path": "",
                    "found_by_original_glob": False,
                    "present_in_crossfit_bird_output": True,
                    "status": "present in output but not found by current input glob",
                }
            )

        audit_output = args.results_dir / "crossfit_input_audit.tsv"
        write_tsv(audit_output, audit_rows)

        print("\nINPUT AUDIT")
        print("=" * 90)
        print(
            f"Decoded databases matched by */*/*decoded_database.json: "
            f"{len(database_paths)}"
        )
        for row in audit_rows:
            print(f"  {row['bird']}: {row['status']}")
            if row["database_path"]:
                print(f"      {row['database_path']}")
        print(f"\nAudit file written: {audit_output}")


if __name__ == "__main__":
    main()
