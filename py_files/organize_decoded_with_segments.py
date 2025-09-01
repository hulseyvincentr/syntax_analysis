# -*- coding: utf-8 -*-
# organize_decoded_with_segments.py
from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

JsonLike = Union[dict, list, str, int, float, bool, None]


# ───────────────────────────
# Public container
# ───────────────────────────
@dataclass
class OrganizedDataset:
    """
    Container for organized outputs (one row per annotated segment).
    """
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
      • JSON string (single/double quotes) → try json.loads (single→double normalization)
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


def _strip_ext(name: str) -> str:
    """
    Remove a single trailing extension like '.wav' from a filename if present.
    """
    if "." in name:
        base, ext = name.rsplit(".", 1)
        # be conservative: only strip common audio/texty extensions if present
        if ext.lower() in {"wav", "flac", "mp3", "json", "txt", "csv"}:
            return base
    return name


def parse_filename_with_segment(file_field: Union[str, Path]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Parse a filename assumed to be underscore-delimited with a trailing segment index:

        USA#### _ <ignored1> _ <MM> _ <DD> _ <HH> _ <MM> _ <SS> _ <SEG>

    Notes
    -----
    - Works whether the stored 'file_name' includes an extension or not. If
      present, the extension is ignored before parsing.
    - If there is **no** trailing segment index (legacy files), `segment` is None.
    - Returns: (animal_id, "MM.DD", HH, MM, SS, segment:int|None, file_stem)
      where file_stem is the base name **without** extension and **without** the trailing segment.

    Examples
    --------
    "USA5497_45444.26577576_6_1_7_22_57_0"         → segment 0
    "USA5497_..._7_22_57_1.wav"                    → segment 1
    "USA5497_..._7_22_57.wav"                      → segment None  (legacy)
    """
    try:
        raw = str(file_field).split("/")[-1]
        no_ext = _strip_ext(raw)
        parts = no_ext.split("_")

        if len(parts) < 7:
            return None, None, None, None, None, None, None

        # Always present fields in your scheme
        animal_id = parts[0]
        month     = parts[2].zfill(2)
        day       = parts[3].zfill(2)
        hour      = parts[4].zfill(2)
        minute    = parts[5].zfill(2)
        second    = parts[6].zfill(2)

        segment: Optional[int] = None
        file_stem: Optional[str] = no_ext

        # If we have an extra trailing numeric token, treat as segment
        if len(parts) >= 8 and parts[-1].isdigit():
            segment = int(parts[-1])
            # remove the trailing segment from the stem
            file_stem = "_".join(parts[:-1])
        else:
            # legacy (no explicit segment index)
            segment = None
            file_stem = no_ext

        return animal_id, f"{month}.{day}", hour, minute, second, segment, file_stem

    except Exception:
        return None, None, None, None, None, None, None


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


def extract_syllable_order(label_to_intervals: dict, *, onset_index: int = 0) -> List[str]:
    """
    Build a per-file syllable order by sorting all syllable intervals by onset time.
    """
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return []

    pairs: List[tuple[float, str]] = []
    for syl, intervals in label_to_intervals.items():
        if not isinstance(intervals, (list, tuple)):
            continue
        for itv in intervals:
            if isinstance(itv, (list, tuple)) and len(itv) > onset_index:
                try:
                    onset = float(itv[onset_index])
                except (TypeError, ValueError):
                    continue
                pairs.append((onset, syl))

    pairs.sort(key=lambda p: p[0])
    return [s for _, s in pairs]


def calculate_syllable_durations_ms(label_to_intervals: dict, syllable_label: str) -> List[float]:
    """
    Return durations (ms) for a given syllable label from:
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
# Core builder (JSON only; one row per segment)
# ───────────────────────────
def build_organized_segments_with_durations(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    only_song_present: bool = False,      # keep all rows by default
    compute_durations: bool = True,       # add syllable_<label>_durations columns
    add_recording_datetime: bool = True,  # handy combined timestamp
) -> OrganizedDataset:
    """
    Load decoded JSON (now possibly with multiple annotated segments per original file,
    expressed via a trailing segment index in 'file_name') and creation metadata JSON,
    then organize into a DataFrame with **one row per annotated segment**.

    Adds/keeps:
      • Animal ID, Date (datetime), Hour/Minute/Second (strings), Segment (int|NaN)
      • File Stem (string; base file without segment suffix or extension)
      • Parsed syllable onset/offset dicts
      • 'syllables_present' and 'syllable_order' per segment
      • (optional) per-label duration columns: 'syllable_<label>_durations' (ms)
      • (optional) Recording DateTime (datetime64[ns])

    Returns an OrganizedDataset with unique dates and labels.

    Expected JSONs
    --------------
    decoded_database_json:
        {
          "results": [
            {
              "file_name": "USA####_..._<MM>_<DD>_<HH>_<MM>_<SS>_<SEG?>",
              "song_present": true/false,
              "syllable_onsets_offsets_ms": {...},
              "syllable_onsets_offsets_timebins": {...},
              ...
            }, ...
          ]
        }

    creation_metadata_json:
        {
          "treatment_date": "YYYY-MM-DD" | null,
          "subdirectories": {
              "<subdir>": {"subdirectory_creation_date": "YYYY-MM-DD", ...},
              ...
          }
        }
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

    # Ensure target columns exist (don’t drop unknown original columns)
    for col in ["Animal ID", "Date", "Hour", "Minute", "Second", "Segment", "File Stem"]:
        if col not in organized.columns:
            organized[col] = None

    # Filename-derived fields (now with Segment and File Stem)
    if "file_name" not in organized.columns:
        raise KeyError("Expected column 'file_name' in decoded database JSON.")

    for i, file_field in enumerate(organized["file_name"]):
        animal_id, month_day, hh, mm, ss, segment, file_stem = parse_filename_with_segment(file_field)
        organized.at[i, "Animal ID"] = animal_id
        organized.at[i, "Date"]      = month_day
        organized.at[i, "Hour"]      = hh
        organized.at[i, "Minute"]    = mm
        organized.at[i, "Second"]    = ss
        organized.at[i, "Segment"]   = segment
        organized.at[i, "File Stem"] = file_stem

    # Upgrade Date from 'MM.DD' to 'YYYY.MM.DD' using metadata
    organized["Date"] = organized["Date"].apply(lambda md: update_date_with_year(md, subdirectory_dates))
    organized["Date"] = pd.to_datetime(organized["Date"], format="%Y.%m.%d", errors="coerce")

    # Optional combined timestamp
    if add_recording_datetime:
        def _mk_dt(row):
            d = row.get("Date")
            h = row.get("Hour")
            m = row.get("Minute")
            s = row.get("Second")
            if pd.isna(d) or d is None or any(val in (None, "",) for val in (h, m, s)):
                return pd.NaT
            try:
                # 'Date' is Timestamp like 2024-06-01; combine with HMS
                return datetime(
                    year=d.year, month=d.month, day=d.day,
                    hour=int(h), minute=int(m), second=int(s)
                )
            except Exception:
                return pd.NaT
        organized["Recording DateTime"] = organized.apply(_mk_dt, axis=1)

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

    # Per-segment syllables present
    def _extract_syllables_present(v: dict) -> List[str]:
        return sorted(v.keys()) if isinstance(v, dict) else []

    organized["syllables_present"] = organized["syllable_onsets_offsets_ms_dict"].apply(_extract_syllables_present)

    # Per-segment syllable order (sorted by onset time in ms)
    organized["syllable_order"] = organized["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: extract_syllable_order(d, onset_index=0)
    )

    # Optional: per-label durations (ms)
    if compute_durations and unique_labels:
        for lab in unique_labels:
            colname = f"syllable_{lab}_durations"
            organized[colname] = organized["syllable_onsets_offsets_ms_dict"].apply(
                lambda d, L=lab: calculate_syllable_durations_ms(d, L)
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

# Reorder so Segment sits beside file_name
    if "file_name" in organized.columns and "Segment" in organized.columns:
        cols = list(organized.columns)
        cols.remove("Segment")
        insert_at = cols.index("file_name") + 1
        cols = cols[:insert_at] + ["Segment"] + cols[insert_at:]
        organized = organized[cols]

    return OrganizedDataset(
        organized_df=organized,
        unique_dates=unique_dates,
        unique_syllable_labels=unique_labels,
        treatment_date=treatment_date_fmt,
    )

# ───────────────────────────
# Example usage (copy into your notebook/console)
# ───────────────────────────
"""
from organize_decoded_with_segments import build_organized_segments_with_durations

decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis/USA5507_RC4/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5507_RC5_Comp2_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis/USA5507_RC4/USA5507_RC4_metadata.json"

out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=True,   # or True to filter
    compute_durations=True,
)

df = out.organized_df
df.head()

# Columns of interest now include:
#   ['Animal ID','Date','Hour','Minute','Second','Segment','File Stem',
#    'Recording DateTime','syllables_present','syllable_order',
#    'syllable_<label>_durations', ...]
#
# You can group by base file/segment like:
# df.sort_values(['File Stem','Segment','Recording DateTime'])
"""
