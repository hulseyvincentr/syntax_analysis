#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_date_matched_umap_subset.py

Creates a restricted/date-matched version of your batch UMAP/BC summary CSV.

Main recommended mode:
    severe_2024_window

This keeps:
    sham:
        USA5283, USA5271

    medial/lateral-involved lesions close in time:
        USA5326, USA5325, USA5288, USA5337, USA5371, USA5443

Then you can feed the resulting subset CSV into your existing graphing script.
"""

from pathlib import Path
import argparse
import pandas as pd


DATE_MATCHED_SETS = {
    "severe_2024_window": {
        "description": (
            "Recommended sensitivity analysis: 2024 sham birds vs nearby "
            "medial/lateral-involved or large Area X lesion birds."
        ),
        "sham": ["USA5283", "USA5271"],
        "lesion": ["USA5326", "USA5325", "USA5288", "USA5337", "USA5371", "USA5443"],
    },
    "severe_1to1": {
        "description": (
            "Strict 1:1 date-matched severe-lesion comparison. "
            "Very small: 2 sham birds vs 2 medial/lateral lesion birds."
        ),
        "sham": ["USA5283", "USA5271"],
        "lesion": ["USA5326", "USA5325"],
    },
    "all_nma_1to1": {
        "description": (
            "All-sham 1:1 date-matched comparison against any NMA lesion. "
            "Includes 2025 single-hit/lateral-only lesion matches."
        ),
        "sham": ["USA5283", "USA5271", "USA5506", "USA5494"],
        "lesion": ["USA5326", "USA5325", "USA5499", "USA5483"],
    },
}


def _pick_column(df: pd.DataFrame, candidates):
    """Pick a column name robustly from possible alternatives."""
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_to_actual:
            return lower_to_actual[key]
    return None


def load_animal_metadata(metadata_xlsx: Path) -> pd.DataFrame:
    """
    Load one row per animal with treatment date and lesion hit type.

    This prefers the metadata_with_hit_type sheet because it contains both
    treatment dates and lesion hit types. If unavailable, it falls back to
    other sheets where possible.
    """
    xls = pd.ExcelFile(metadata_xlsx)

    if "metadata_with_hit_type" in xls.sheet_names:
        df = pd.read_excel(metadata_xlsx, sheet_name="metadata_with_hit_type")
    elif "metadata" in xls.sheet_names:
        df = pd.read_excel(metadata_xlsx, sheet_name="metadata")
    else:
        df = pd.read_excel(metadata_xlsx, sheet_name=xls.sheet_names[0])

    animal_col = _pick_column(df, ["Animal ID", "animal_id", "animal id", "Bird ID", "bird"])
    date_col = _pick_column(df, ["Treatment date", "treatment_date", "date"])
    hit_col = _pick_column(df, ["Lesion hit type", "hit_type", "lesion hit type", "Treatment type"])

    if animal_col is None:
        raise KeyError("Could not find an animal ID column in the metadata Excel file.")

    out_cols = [animal_col]
    if date_col is not None:
        out_cols.append(date_col)
    if hit_col is not None:
        out_cols.append(hit_col)

    out = df[out_cols].copy()
    rename = {animal_col: "animal_id"}
    if date_col is not None:
        rename[date_col] = "treatment_date"
    if hit_col is not None:
        rename[hit_col] = "lesion_hit_type"

    out = out.rename(columns=rename)
    out["animal_id"] = out["animal_id"].astype(str).str.strip()

    if "treatment_date" in out.columns:
        out["treatment_date"] = pd.to_datetime(out["treatment_date"], errors="coerce")
    else:
        out["treatment_date"] = pd.NaT

    if "lesion_hit_type" not in out.columns:
        out["lesion_hit_type"] = "unknown"

    # One row per animal. If there are multiple injection rows, keep the earliest
    # treatment date and first lesion-hit label.
    animal_summary = (
        out.sort_values(["animal_id", "treatment_date"])
        .groupby("animal_id", as_index=False)
        .agg(
            treatment_date=("treatment_date", "min"),
            lesion_hit_type=("lesion_hit_type", "first"),
        )
    )

    return animal_summary


def main():
    parser = argparse.ArgumentParser(
        description="Create a date-matched subset of the batch UMAP/BC summary CSV."
    )
    parser.add_argument("--summary-csv", required=True, type=str)
    parser.add_argument("--metadata-xlsx", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument(
        "--mode",
        choices=list(DATE_MATCHED_SETS.keys()),
        default="severe_2024_window",
        help="Which date-matched subset to create.",
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    metadata_xlsx = Path(args.metadata_xlsx)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    subset_info = DATE_MATCHED_SETS[args.mode]
    selected_sham = subset_info["sham"]
    selected_lesion = subset_info["lesion"]
    selected_animals = selected_sham + selected_lesion

    print()
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(subset_info["description"])
    print("=" * 80)
    print()

    summary_df = pd.read_csv(summary_csv)

    if "animal_id" not in summary_df.columns:
        raise KeyError("The summary CSV must contain an 'animal_id' column.")

    summary_df["animal_id"] = summary_df["animal_id"].astype(str).str.strip()

    subset_df = summary_df[summary_df["animal_id"].isin(selected_animals)].copy()

    missing_from_summary = sorted(set(selected_animals) - set(summary_df["animal_id"].unique()))
    present_in_summary = sorted(set(subset_df["animal_id"].unique()))

    print(f"Original summary rows: {len(summary_df)}")
    print(f"Subset summary rows:  {len(subset_df)}")
    print()
    print(f"Animals requested: {len(selected_animals)}")
    print(f"Animals found in summary CSV: {len(present_in_summary)}")
    print()

    if missing_from_summary:
        print("WARNING: These selected animals were not found in the summary CSV:")
        for animal in missing_from_summary:
            print(f"  - {animal}")
        print()

    subset_df.to_csv(out_csv, index=False)
    print(f"Saved subset CSV:")
    print(f"  {out_csv}")
    print()

    # Also save a small human-readable animal table next to the subset CSV.
    try:
        meta = load_animal_metadata(metadata_xlsx)
        meta_subset = meta[meta["animal_id"].isin(selected_animals)].copy()

        meta_subset["date_matched_subset_group"] = meta_subset["animal_id"].map(
            lambda x: "sham" if x in selected_sham else "lesion"
        )

        meta_subset = meta_subset.sort_values(
            ["date_matched_subset_group", "treatment_date", "animal_id"]
        )

        animal_table_csv = out_csv.with_name(out_csv.stem + "_animal_table.csv")
        meta_subset.to_csv(animal_table_csv, index=False)

        print("Selected animals:")
        print(meta_subset.to_string(index=False))
        print()
        print(f"Saved selected-animal table:")
        print(f"  {animal_table_csv}")
        print()

    except Exception as e:
        print("Could not summarize selected animals from metadata Excel.")
        print(f"Metadata summary error: {repr(e)}")
        print()

    print("Rows per animal in the subset CSV:")
    print(subset_df["animal_id"].value_counts().sort_index().to_string())
    print()
    print("Done.")


if __name__ == "__main__":
    main()
