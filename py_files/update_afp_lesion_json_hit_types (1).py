#!/usr/bin/env python3
"""Add detailed lesion hit-type metadata to AFP_lesion_bird_metadata.json.

The script reads the ``animal_hit_type_summary`` sheet from the lesion metadata
workbook and updates the bird-keyed JSON stored in an ``AFP_lesion_jsons``
directory. It preserves the existing coarse ``lesion_group`` field used by the
cutoff analyses and adds separate fields for plotting complete versus partial
medial+lateral lesions.

A timestamped backup is written before the JSON is modified.

Example
-------
python update_afp_lesion_json_hit_types.py \
  "$HOME/Desktop/AFP_lesion_jsons" \
  "$HOME/Downloads/Area_X_lesion_metadata_with_hit_types.xlsx"
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_FILENAME = "AFP_lesion_bird_metadata.json"
DEFAULT_SHEET = "animal_hit_type_summary"

REQUIRED_COLUMNS = [
    "Animal ID",
    "Area X visible in histology? (parsed)",
    "Medial Area X hit type",
    "Lateral Area X hit type",
    "Lesion hit type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update AFP_lesion_bird_metadata.json with detailed complete/partial "
            "lesion hit-type fields from the lesion metadata workbook."
        )
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="AFP_lesion_jsons directory containing AFP_lesion_bird_metadata.json.",
    )
    parser.add_argument(
        "metadata_excel",
        type=Path,
        help="Area_X_lesion_metadata_with_hit_types.xlsx workbook.",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help=f"Workbook sheet containing one row per bird (default: {DEFAULT_SHEET}).",
    )
    parser.add_argument(
        "--metadata-filename",
        default=METADATA_FILENAME,
        help=f"Bird metadata JSON filename (default: {METADATA_FILENAME}).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before overwriting the JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if a bird is present in one source but absent from the other. "
            "By default, unmatched birds are reported and matching birds are updated."
        ),
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def clean_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = clean_text(value).lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def classify_hit_type(raw_hit_type: str) -> tuple[str, str, str]:
    """Return (coarse_group, detailed_group, lesion_extent_class)."""
    text = clean_text(raw_hit_type).lower()

    if "sham" in text or ("saline" in text and "lesion" not in text):
        return "sham_saline", "sham saline injection", "sham"

    if "single hit" in text or "lateral" in text and "only" in text:
        return "lateral_only", "Lateral lesion only", "lateral_only"

    if "large lesion" in text or "not visible" in text:
        return (
            "medial_and_lateral",
            "Complete Medial and Lateral lesion",
            "complete",
        )

    if "medial+lateral" in text or ("medial" in text and "lateral" in text):
        return (
            "medial_and_lateral",
            "Partial Medial and Lateral lesion",
            "partial",
        )

    raise ValueError(f"Unrecognized lesion hit type: {raw_hit_type!r}")


def read_hit_type_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    xls = pd.ExcelFile(path)
    if sheet_name not in xls.sheet_names:
        raise ValueError(
            f"Sheet {sheet_name!r} not found in {path}. Available sheets: {xls.sheet_names}"
        )

    df = pd.read_excel(path, sheet_name=sheet_name)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet {sheet_name!r} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["Animal ID"]).copy()
    df["Animal ID"] = df["Animal ID"].map(clean_text)
    if df["Animal ID"].duplicated().any():
        duplicates = sorted(df.loc[df["Animal ID"].duplicated(False), "Animal ID"].unique())
        raise ValueError(f"Duplicate bird rows in {sheet_name!r}: {duplicates}")
    return df


def main() -> None:
    args = parse_args()
    json_dir = args.json_dir.expanduser().resolve()
    excel_path = args.metadata_excel.expanduser().resolve()
    metadata_path = json_dir / args.metadata_filename

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected a bird-keyed JSON object in {metadata_path}")

    hit_df = read_hit_type_table(excel_path, args.sheet)
    hit_rows = {row["Animal ID"]: row for _, row in hit_df.iterrows()}

    json_birds = set(metadata)
    excel_birds = set(hit_rows)
    missing_from_excel = sorted(json_birds - excel_birds)
    missing_from_json = sorted(excel_birds - json_birds)

    if args.strict and (missing_from_excel or missing_from_json):
        raise ValueError(
            "Bird mismatch between JSON and workbook. "
            f"Missing from workbook: {missing_from_excel}; "
            f"missing from JSON: {missing_from_json}"
        )

    if not args.no_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = metadata_path.with_name(
            f"{metadata_path.stem}.backup_before_hit_types_{timestamp}{metadata_path.suffix}"
        )
        shutil.copy2(metadata_path, backup_path)
        print(f"[OK] Backup written: {backup_path}")

    updated = 0
    detailed_counts: dict[str, int] = {}
    coarse_mismatches: list[tuple[str, str, str]] = []

    for bird in sorted(json_birds & excel_birds):
        row = hit_rows[bird]
        raw_hit_type = clean_text(row["Lesion hit type"])
        coarse_group, detailed_group, extent_class = classify_hit_type(raw_hit_type)

        bird_meta = metadata[bird]
        if not isinstance(bird_meta, dict):
            raise TypeError(f"Metadata entry for {bird!r} is not an object")

        existing_coarse = clean_text(bird_meta.get("lesion_group", ""))
        if existing_coarse and existing_coarse != coarse_group:
            coarse_mismatches.append((bird, existing_coarse, coarse_group))
        elif not existing_coarse:
            bird_meta["lesion_group"] = coarse_group

        # Preserve the existing coarse field and add detailed histology fields.
        bird_meta["lesion_hit_type"] = raw_hit_type
        bird_meta["lesion_group_detailed"] = detailed_group
        bird_meta["lesion_extent_class"] = extent_class
        bird_meta["area_x_visible_in_histology"] = clean_bool(
            row["Area X visible in histology? (parsed)"]
        )
        bird_meta["medial_area_x_hit_type"] = clean_text(row["Medial Area X hit type"])
        bird_meta["lateral_area_x_hit_type"] = clean_text(row["Lateral Area X hit type"])

        updated += 1
        detailed_counts[detailed_group] = detailed_counts.get(detailed_group, 0) + 1

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[OK] Updated {updated} bird entries in: {metadata_path}")
    print("[INFO] Detailed group counts:")
    for group, count in sorted(detailed_counts.items()):
        print(f"  {group}: {count}")

    if missing_from_excel:
        print(f"[WARN] Birds in JSON but not workbook ({len(missing_from_excel)}):")
        print("  " + ", ".join(missing_from_excel))
    if missing_from_json:
        print(f"[WARN] Birds in workbook but not JSON ({len(missing_from_json)}):")
        print("  " + ", ".join(missing_from_json))
    if coarse_mismatches:
        print("[WARN] Existing coarse lesion_group values were preserved despite mismatches:")
        for bird, existing, expected in coarse_mismatches:
            print(f"  {bird}: existing={existing!r}, workbook-implied={expected!r}")

    print("[NEXT] Regenerate the Figure 3 input CSVs so lesion_group_detailed is propagated.")


if __name__ == "__main__":
    main()
