#!/usr/bin/env python3
"""Validate final Figure 4 bird counts, membership, and key p-values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd


EXPECTED_GROUP_COUNTS = {
    "sham saline injection": 4,
    "Lateral lesion only": 8,
    "Medial and Lateral lesion": 8,
}
EXPECTED_ML_BIRDS = {
    "USA5288", "USA5325", "USA5326", "USA5337",
    "USA5371", "USA5443", "USA5468", "USA5509",
}
EXPECTED_LATERAL_BIRDS = {
    "R08", "R09", "R10", "USA5336",
    "USA5347", "USA5483", "USA5499", "USA5510",
}
EXPECTED_P = {
    ("all_clusters", "sham saline injection"): 0.875,
    ("all_clusters", "Lateral lesion only"): 0.7421875,
    ("all_clusters", "Medial and Lateral lesion"): 0.0390625,
    ("high_variance_clusters", "Medial and Lateral lesion"): 0.0234375,
    ("remaining_non_high_variance_clusters", "Medial and Lateral lesion"): 0.15625,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate the final Figure 4 batch output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--strict", action="store_true", help="Exit nonzero when any expected result differs.")
    p.add_argument("--atol", type=float, default=1e-12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch = args.batch_dir.expanduser().resolve()
    bird_file = batch / "bc_batch_bird_level_summary.csv"
    stats_file = batch / "bc_batch_lesion_group_stats.csv"
    if not bird_file.is_file() or not stats_file.is_file():
        raise FileNotFoundError(f"Expected batch CSVs under {batch}")

    bird = pd.read_csv(bird_file)
    stats = pd.read_csv(stats_file)
    selected = bird[bird["bc_method"].astype(str) == "selected_bins"].copy()
    all_rows = selected[selected["set_name"].astype(str) == "all_clusters"]

    errors = []
    print("Group counts:")
    for group, expected in EXPECTED_GROUP_COUNTS.items():
        observed = int(all_rows.loc[all_rows["lesion_hit_type"].astype(str) == group, "animal_id"].nunique())
        print(f"  {group}: observed={observed}, expected={expected}")
        if observed != expected:
            errors.append(f"{group} count {observed} != {expected}")

    ml = set(all_rows.loc[all_rows["lesion_hit_type"].astype(str) == "Medial and Lateral lesion", "animal_id"].astype(str))
    lateral = set(all_rows.loc[all_rows["lesion_hit_type"].astype(str) == "Lateral lesion only", "animal_id"].astype(str))
    if ml != EXPECTED_ML_BIRDS:
        errors.append(f"M+L membership differs: {sorted(ml)}")
    if lateral != EXPECTED_LATERAL_BIRDS:
        errors.append(f"Lateral membership differs: {sorted(lateral)}")

    print("\nKey paired Wilcoxon p-values:")
    bird_stats = stats[
        (stats["analysis_level"].astype(str) == "bird")
        & (stats["bc_method"].astype(str) == "selected_bins")
        & (stats["test_family"].astype(str) == "paired_pre_vs_post_within_lesion_group")
    ]
    for key, expected in EXPECTED_P.items():
        set_name, group = key
        row = bird_stats[
            (bird_stats["set_name"].astype(str) == set_name)
            & (bird_stats["lesion_hit_type"].astype(str) == group)
        ]
        if len(row) != 1:
            errors.append(f"Expected one stats row for {key}, found {len(row)}")
            print(f"  {key}: missing or duplicated")
            continue
        observed = float(row.iloc[0]["paired_pre_post_wilcoxon_p"])
        match = np.isclose(observed, expected, rtol=0.0, atol=args.atol)
        print(f"  {set_name} | {group}: observed={observed:.10g}, expected={expected:.10g}, match={match}")
        if not match:
            errors.append(f"p-value {key}: {observed} != {expected}")

    print("\nValidation:", "PASS" if not errors else "FAIL")
    for error in errors:
        print("  -", error)
    if errors and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
