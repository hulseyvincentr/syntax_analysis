# organize_decoded_with_durations.py
from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


JsonLike = Union[dict, list, str, int, float, bool, None]


@dataclass
class OrganizedDataset:
    """Container for the organized outputs."""
    organized_df: pd.DataFrame
    unique_dates: List[str]               # 'YYYY.MM.DD'
    unique_syllable_labels: List[str]     # sorted strings
    treatment_date: Optional[str] = None  # 'YYYY.MM.DD' or None


# ───────────────────────────
# Parsing helpers
# ───────────────────────────
def parse_json_safe(value: JsonLike) -> dict:
    """
    Best-effort parse of JSON-like content that might be stored as:
      • dict (already parsed) → return as-is
      • JSON string (single quotes) → try json.loads with quote normalization
      • Python literal string (e.g., "{'0': [..]}") → ast.literal_eval
      • NaN/empty/parse-fail → {}
    """
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}

    if not isinstance(value, str):
        return {}

    s = value.strip()
    if not s:
        return {}

    # Strip odd wrapping quotes
    if s.startswith("''") and s.endswith("''"):
        s = s[2:-2].strip()
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

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
    provided as 'YYYY-MM-DD' in the metadata JSON. Requires exact (month, day) match.
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
    """
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return []

    onset_syllable_pairs: List[Tuple[float, str]] = []
    for syllable, intervals in label_to_intervals.items():
        if not isinstance(intervals, (list, tuple)):
            continue
        for interval in intervals:
            if isinstance(interval, (list, tuple)) and len(interval) > onset_index:
                try:
                    onset_time = float(interval[onset_index])
                except (TypeError, ValueError):
                    continue
                onset_syllable_pairs.append((onset_time, syllable))

    onset_syllable_pairs.sort(key=lambda p: p[0])
    return [s for _, s in onset_syllable_pairs]


def calculate_syllable_durations_ms(label_to_intervals: dict, syllable_label: str) -> List[float]:
    """
    Return a list of durations (ms) for a given syllable label from a mapping:
        { 'label': [[onset_ms, offset_ms], ...], ... }
    """
    if not isinstance(label_to_intervals, dict):
        return []
    intervals = label_to_intervals.get(syllable_label, [])
    out: List[float] = []
    for itv in intervals:
        if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
            continue
        try:
            on, off = float(itv[0]), float(itv[1])
            out.append(off - on)
        except (TypeError, ValueError):
            continue
    return out


# ───────────────────────────
# Core builder (JSON only)
# ───────────────────────────
def build_organized_dataset_with_durations(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    only_song_present: bool = False,      # keep all rows by default
    compute_durations: bool = True,       # add syllable_<label>_durations columns
) -> OrganizedDataset:
    """
    Load decoded syllable JSON and creation metadata JSON, organize into a DataFrame with:
      • Animal ID, Date (as datetime), Hour/Minute/Second (strings)
      • Parsed syllable onset/offset dicts
      • 'syllables_present' and 'syllable_order' per file
      • (optional) per-label duration columns: 'syllable_<label>_durations' (ms)
    Returns OrganizedDataset with unique dates and labels.

    Notes
    -----
    - JSON only; expects top-level key 'results' (list of dicts) in decoded JSON.
    - Creation metadata JSON must contain 'subdirectories' with 'subdirectory_creation_date'.
    """
    decoded_database_json = Path(decoded_database_json)
    creation_metadata_json = Path(creation_metadata_json)

    # Load creation metadata
    with creation_metadata_json.open("r") as f:
        meta = json.load(f)

    # Treatment date (optional)
    treatment_date_iso = meta.get("treatment_date", None)
    treatment_date_fmt = None
    if treatment_date_iso:
        try:
            treatment_date_fmt = datetime.strptime(treatment_date_iso, "%Y-%m-%d").strftime("%Y.%m.%d")
        except ValueError:
            treatment_date_fmt = None

    # Subdirectory dates: {subdir: 'YYYY-MM-DD'}
    subdirectory_dates = {
        subdir: data.get("subdirectory_creation_date", "")
        for subdir, data in meta.get("subdirectories", {}).items()
        if isinstance(data, dict)
    }

    # Load decoded data (JSON only, expect 'results')
    with decoded_database_json.open("r") as f:
        decoded_payload = json.load(f)

    results = decoded_payload.get("results", [])
    df = pd.DataFrame(results)

    # Parse potentially stringified dict columns
    if "syllable_onsets_offsets_ms" in df.columns:
        df["syllable_onsets_offsets_ms"] = df["syllable_onsets_offsets_ms"].apply(parse_json_safe)
    if "syllable_onsets_offsets_timebins" in df.columns:
        df["syllable_onsets_offsets_timebins"] = df["syllable_onsets_offsets_timebins"].apply(parse_json_safe)

    organized = df.copy()

    # Optional filter
    if only_song_present and "song_present" in organized.columns:
        organized = organized[organized["song_present"] == True].reset_index(drop=True)

    # Ensure target columns
    for col in ["Animal ID", "Date", "Hour", "Minute", "Second"]:
        organized[col] = None

    # Filename-derived time fields
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

    # Dict copy for convenience
    organized["syllable_onsets_offsets_ms_dict"] = organized.get("syllable_onsets_offsets_ms", {})

    # Unique syllable labels across the dataset
    unique_labels: List[str] = []
    if "syllable_onsets_offsets_ms_dict" in organized.columns:
        label_set: set[str] = set()
        for row in organized["syllable_onsets_offsets_ms_dict"]:
            if isinstance(row, dict) and row:
                label_set.update(row.keys())
        unique_labels = sorted(label_set)

    # Per-file syllables present
    def _extract_syllables_present(v: dict) -> List[str]:
        return sorted(v.keys()) if isinstance(v, dict) else []

    organized["syllables_present"] = organized["syllable_onsets_offsets_ms_dict"].apply(_extract_syllables_present)

    # Per-file syllable order (sorted by onset time in ms)
    organized["syllable_order"] = organized["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: extract_syllable_order(d, onset_index=0)
    )

    # Optional: per-label durations (ms)
    if compute_durations and unique_labels:
        for lab in unique_labels:
            colname = f"syllable_{lab}_durations"
            organized[colname] = organized["syllable_onsets_offsets_ms_dict"].apply(
                lambda d: calculate_syllable_durations_ms(d, lab)
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

    return OrganizedDataset(
        organized_df=organized,
        unique_dates=unique_dates,
        unique_syllable_labels=unique_labels,
        treatment_date=treatment_date_fmt,
    )


"""
from organize_decoded_with_durations import build_organized_dataset_with_durations

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

out = build_organized_dataset_with_durations(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,   # or True to filter
    compute_durations=True,
)

# Now inspect `out.organized_df` in the Variable Explorer
# `out.unique_dates`, `out.unique_syllable_labels`, `out.treatment_date` are also available

"""