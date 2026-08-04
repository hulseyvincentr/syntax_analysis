#!/usr/bin/env python3
"""Compare per-cluster and bird-level selected-bin BC values between two roots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_ML_BIRDS = [
    "USA5288", "USA5325", "USA5326", "USA5337",
    "USA5371", "USA5443", "USA5468", "USA5509",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit two Figure 4 per-bird summary roots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("old_root", type=Path)
    p.add_argument("new_root", type=Path)
    p.add_argument("--birds", nargs="+", default=DEFAULT_ML_BIRDS)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def load(root: Path, bird: str, suffix: str) -> pd.DataFrame:
    path = root / bird / f"{bird}_cluster_bc_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "passes_min_balanced_duration" in df.columns:
        df = df[bool_series(df["passes_min_balanced_duration"])].copy()
    required = ["cluster_id", "is_high_variance_cluster", "bc_pre_selected_bins", "bc_post_selected_bins"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}")
    out = pd.DataFrame({
        "animal_id": bird,
        "cluster_id": df["cluster_id"].astype(str),
        f"high_variance_{suffix}": bool_series(df["is_high_variance_cluster"]),
        f"bc_pre_{suffix}": pd.to_numeric(df["bc_pre_selected_bins"], errors="coerce"),
        f"bc_post_{suffix}": pd.to_numeric(df["bc_post_selected_bins"], errors="coerce"),
    })
    return out[np.isfinite(out[f"bc_pre_{suffix}"]) & np.isfinite(out[f"bc_post_{suffix}"])].copy()


def main() -> None:
    args = parse_args()
    old_root = args.old_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    merged_parts: List[pd.DataFrame] = []
    for bird in args.birds:
        old = load(old_root, bird, "old")
        new = load(new_root, bird, "new")
        merged_parts.append(old.merge(new, on=["animal_id", "cluster_id"], how="outer", indicator=True))
    comparison = pd.concat(merged_parts, ignore_index=True)
    comparison["high_variance_match"] = comparison["high_variance_old"] == comparison["high_variance_new"]
    comparison["bc_pre_new_minus_old"] = comparison["bc_pre_new"] - comparison["bc_pre_old"]
    comparison["bc_post_new_minus_old"] = comparison["bc_post_new"] - comparison["bc_post_old"]

    print("Cluster presence:")
    print(comparison["_merge"].value_counts(dropna=False).to_string())
    matched = comparison[comparison["_merge"] == "both"]
    print("\nHigh-variance flags match:")
    print(matched["high_variance_match"].value_counts(dropna=False).to_string())
    print("\nMaximum absolute differences:")
    print("  Pre: ", matched["bc_pre_new_minus_old"].abs().max())
    print("  Post:", matched["bc_post_new_minus_old"].abs().max())
    print("\nBy bird:")
    summary = matched.groupby("animal_id").agg(
        n_matched_clusters=("cluster_id", "size"),
        all_high_variance_flags_match=("high_variance_match", "all"),
        max_abs_pre_difference=("bc_pre_new_minus_old", lambda x: x.abs().max()),
        max_abs_post_difference=("bc_post_new_minus_old", lambda x: x.abs().max()),
    )
    print(summary.to_string())

    if args.output_csv:
        output = args.output_csv.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output, index=False)
        print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
