from __future__ import annotations

import json
import re
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


JsonLike = Union[dict, list, str, int, float, bool, None]


# ───────────────────────────
# Public return container
# ───────────────────────────
@dataclass
class OrganizedDataset:
    """Container for the organized outputs."""
    organized_df: pd.DataFrame
    unique_dates: List[str]               # 'YYYY.MM.DD'
    unique_syllable_labels: List[str]     # sorted strings


# ───────────────────────────
# Parsing helpers
# ───────────────────────────
def parse_json_safe(value: JsonLike) -> dict:
    """
    Best-effort parse of JSON-like content that might be stored as:
      • dict (already parsed) → return as-is
      • JSON string (single quotes) → try json.loads with quote fix
      • Python literal string (e.g., "{'0': [..]}") → ast.literal_eval
      • NaN/empty/parse-fail → {}
    """
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}

    if not isinstance(value, str):
        # Unexpected type; do not throw—return {}
        return {}

    s = value.strip()
    if not s:
        return {}

    # Strip odd wrapping quotes
    if s.startswith("''") and s.endswith("''"):
        s = s[2:-2]
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1]

    if not s:
        return {}

    # Try JSON (normalize single to double quotes)
    try:
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def extract_animal_id_from_path(path: Union[str, Path]) -> Optional[str]:
    """
    Extract an animal ID like 'USA5288' from a path string.
    """
    m = re.search(r"(USA\d{4})", str(path))
    return m.group(1) if m else None


def find_recording_dates_and_times(file_path: Union[str, Path]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse a filename assumed to follow this underscore-delimited pattern:

        USA#### _ <ignored1> _ <MM> _ <DD> _ <HH> _ <MM> _ <SS>.wav

    Returns: (animal_id, "MM.DD", HH, MM, SS) with zero-padding, or (None, ..) on error.
    """
    try:
        name = str(file_path).split("/")[-1]
        parts = name.split("_")
        animal_id = parts[0]
        month = parts[2].zfill(2)
        day   = parts[3].zfill(2)
        hour  = parts[4].zfill(2)
        minute = parts[5].zfill(2)
        second = parts[6].replace(".wav", "").zfill(2)
        return animal_id, f"{month}.{day}", hour, minute, second
    except Exception:
        return None, None, None, None, None


def update_date_with_year(month_dot_day: Optional[str], subdirectory_dates: Dict[str, str]) -> Optional[str]:
    """
    Convert 'MM.DD' to 'YYYY.MM.DD' by matching against subdirectory creation dates
    provided as 'YYYY-MM-DD' in the metadata JSON.
    """
    if not month_dot_day or pd.isna(month_dot_day):
        return None

    try:
        month, day = month_dot_day.split(".")
    except ValueError:
        return None

    for _, iso_date in subdirectory_dates.items():
        try:
            year, json_month, json_day = iso_date.split("-")
        except ValueError:
            continue
        if json_month == month and json_day == day:
            return f"{year}.{month}.{day}"

    return None


def extract_syllable_order(
    label_to_intervals: dict,
    *,
    onset_index: int = 0
) -> List[str]:
    """
    Build a per-file syllable order by sorting all syllable intervals by onset time.

    Parameters
    ----------
    label_to_intervals : dict
        Mapping: syllable_label -> list of [onset_ms, offset_ms] (or similar).
    onset_index : int
        Which position within each interval holds the onset time. Default 0.

    Returns
    -------
    list[str]
        A list of syllable labels ordered by their onset times across the file.
    """
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return []

    onset_syllable_pairs: List[Tuple[float, str]] = []
    for syllable, intervals in label_to_intervals.items():
        if not isinstance(intervals, (list, tuple)):
            continue
        for interval in intervals:
            # Accept list/tuple intervals of length >= onset_index+1
            if isinstance(interval, (list, tuple)) and len(interval) > onset_index:
                onset_time = interval[onset_index]
                # Ignore NaNs/non-numerics silently
                try:
                    onset_time = float(onset_time)
                except (TypeError, ValueError):
                    continue
                onset_syllable_pairs.append((onset_time, syllable))

    onset_syllable_pairs.sort(key=lambda p: p[0])
    return [s for _, s in onset_syllable_pairs]


# ───────────────────────────
# Core builder
# ───────────────────────────
def build_organized_dataset(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    verbose: bool = False,
) -> OrganizedDataset:
    """
    Load decoded syllable JSON and creation metadata JSON, organize into a DataFrame with:
      • Animal ID, Date (as datetime), Hour/Minute/Second (strings)
      • Parsed syllable onset/offset dicts
      • 'syllables_present' and 'syllable_order' per file
    Also returns unique recording dates and unique syllable labels.

    Parameters
    ----------
    decoded_database_json : str | Path
        Path to the decoded database JSON (expects top-level key 'results' with a list of dicts).
    creation_metadata_json : str | Path
        Path to the metadata JSON containing 'treatment_date' and 'subdirectories'.
    verbose : bool
        If True, prints minimal progress messages.

    Returns
    -------
    OrganizedDataset
    """
    decoded_database_json = Path(decoded_database_json)
    creation_metadata_json = Path(creation_metadata_json)

    # Load creation metadata
    with creation_metadata_json.open("r") as f:
        meta = json.load(f)

    # Treatment date (optional downstream use)
    treatment_date_iso = meta.get("treatment_date", None)
    treatment_date_fmt = None
    if treatment_date_iso:
        try:
            treatment_date_fmt = datetime.strptime(treatment_date_iso, "%Y-%m-%d").strftime("%Y.%m.%d")
        except ValueError:
            treatment_date_fmt = None
    if verbose and treatment_date_fmt:
        print(f"[info] Treatment date: {treatment_date_fmt}")

    # Subdirectory dates: {subdir: 'YYYY-MM-DD'}
    subdirectory_dates = {
        subdir: data.get("subdirectory_creation_date", "")
        for subdir, data in meta.get("subdirectories", {}).items()
        if isinstance(data, dict)
    }

    # Load decoded data
    if verbose:
        print(f"[info] Reading decoded database: {decoded_database_json}")
    with decoded_database_json.open("r") as f:
        decoded_payload = json.load(f)

    results = decoded_payload.get("results", [])
    df = pd.DataFrame(results)

    # Parse potentially stringified dict columns
    for col in ("syllable_onsets_offsets_ms", "syllable_onsets_offsets_timebins"):
        if col in df.columns:
            df[col] = df[col].apply(parse_json_safe)

    # Build organized table
    organized = df.copy()
    for col in ["Animal ID", "Date", "Hour", "Minute", "Second"]:
        organized[col] = None

    if "file_name" not in organized.columns:
        raise KeyError("Expected column 'file_name' in decoded database JSON.")

    for i, file_path in enumerate(organized["file_name"]):
        animal_id, month_day, hh, mm, ss = find_recording_dates_and_times(file_path)
        organized.at[i, "Animal ID"] = animal_id
        organized.at[i, "Date"] = month_day
        organized.at[i, "Hour"] = hh
        organized.at[i, "Minute"] = mm
        organized.at[i, "Second"] = ss

    # Upgrade Date from 'MM.DD' to 'YYYY.MM.DD' using metadata
    organized["Date"] = organized["Date"].apply(lambda md: update_date_with_year(md, subdirectory_dates))
    organized["Date"] = pd.to_datetime(organized["Date"], format="%Y.%m.%d", errors="coerce")

    # Keep a dict copy for convenient access
    organized["syllable_onsets_offsets_ms_dict"] = organized.get("syllable_onsets_offsets_ms", {})

    # Unique syllable labels across the dataset
    unique_labels: set[str] = set()
    for row in organized["syllable_onsets_offsets_ms_dict"]:
        if isinstance(row, dict) and row:
            unique_labels.update(row.keys())
    unique_syllable_labels = sorted(unique_labels)

    # Per-file syllables present
    def _extract_syllables_present(v: dict) -> List[str]:
        return sorted(v.keys()) if isinstance(v, dict) else []

    organized["syllables_present"] = organized["syllable_onsets_offsets_ms_dict"].apply(_extract_syllables_present)

    # Per-file syllable order (sorted by onset time in ms)
    organized["syllable_order"] = organized["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: extract_syllable_order(d, onset_index=0)
    )

    # Unique recording dates as strings
    unique_dates = (
        organized["Date"]
        .dt.strftime("%Y.%m.%d")
        .dropna()
        .unique()
        .tolist()
    )
    unique_dates.sort()

    if verbose:
        print(f"[info] Rows: {len(organized)} | unique dates: {len(unique_dates)} | labels: {len(unique_syllable_labels)}")

    return OrganizedDataset(
        organized_df=organized,
        unique_dates=unique_dates,
        unique_syllable_labels=unique_syllable_labels,
    )


# ───────────────────────────
# Optional: quick CLI demo
# ───────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Organize decoded birdsong dataset.")
    p.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    p.add_argument("creation_metadata_json", type=str, help="Path to *_creation_data.json")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    p.add_argument("--out-csv", type=str, default="", help="Optional path to save organized CSV")
    args = p.parse_args()

    out = build_organized_dataset(
        decoded_database_json=args.decoded_database_json,
        creation_metadata_json=args.creation_metadata_json,
        verbose=args.verbose,
    )

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.organized_df.to_csv(args.out_csv, index=False)
        if args.verbose:
            print(f"[info] Saved: {args.out_csv}")
