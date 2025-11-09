# -*- coding: utf-8 -*-
# organize_decoded_serialTS_segments.py
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
    treatment_date: Optional[str] = None  # left as None (no metadata used)


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
        if ext.lower() in {"wav", "flac", "mp3", "json", "txt", "csv"}:
            return base
    return name


def excel_serial_to_timestamp(serial: Union[str, float, int]) -> Optional[pd.Timestamp]:
    """
    Convert an Excel serial day (1900 system) to a pandas Timestamp.
    Uses origin='1899-12-30' which matches Excel's 1900-based convention.
    Accepts strings or numbers; returns None if it cannot be parsed.
    """
    try:
        val = float(serial)
        # Pandas handles the fractional day → time-of-day for us
        ts = pd.to_datetime(val, unit="D", origin="1899-12-30")
        # Ensure tz-naive Timestamp
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts[0]
        return pd.Timestamp(ts)
    except Exception:
        return None


def parse_filename_with_excel_serial(file_field: Union[str, Path]) -> Tuple[
    Optional[str],  # animal_id
    Optional[float],# excel_serial
    Optional[int],  # segment index
    Optional[str],  # file_stem (without extension and without trailing segment)
]:
    """
    Parse filenames of the form:

        <ANIMAL> _ <EXCEL_SERIAL> _ <M> _ <D> _ <H> _ <M> _ <S> [_<SEG>]

    Examples
    --------
    "R08_45765.2741464_4_18_0_45_41"       → excel_serial=45765.2741464, segment=None
    "USA5497_45444.26577576_6_1_7_22_57_0" → excel_serial=45444.26577576, segment=0

    Notes
    -----
    - We DO NOT use the trailing tokens for time; they’re ignored in favor of the Excel serial.
    - Segment (final numeric token) is optional.
    - Returns (animal_id, serial_float, segment_or_None, file_stem_without_segment).
    """
    try:
        raw = str(file_field).split("/")[-1]
        no_ext = _strip_ext(raw)
        parts = no_ext.split("_")

        if len(parts) < 2:
            return None, None, None, None

        animal_id = parts[0]
        serial_str = parts[1]

        # Detect optional trailing segment:
        # [ANIMAL, SERIAL, M, D, H, M, S]  -> len == 7 (no segment)
        # If there's one more numeric token -> segment
        segment: Optional[int] = None
        if len(parts) >= 8 and parts[-1].isdigit():
            segment = int(parts[-1])
            file_stem = "_".join(parts[:-1])
        else:
            file_stem = no_ext

        try:
            serial_val = float(serial_str)
        except Exception:
            serial_val = None

        return animal_id, serial_val, segment, file_stem

    except Exception:
        return None, None, None, None


def build_organized_segments_with_durations(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path, None] = None,  # kept for backward-compatibility, unused
    *,
    only_song_present: bool = False,      # keep all rows by default
    compute_durations: bool = True,       # add syllable_<label>_durations columns
    add_recording_datetime: bool = True,  # always derived from Excel serial here
) -> OrganizedDataset:
    """
    Load the decoded JSON (possibly with multiple annotated segments per original file),
    and organize into a DataFrame with **one row per annotated segment**.

    IMPORTANT: Recording times are derived from the Excel serial found in the
    filename (the token immediately after the animal ID). No metadata file is needed.

    Adds/keeps:
      • Animal ID
      • Recording DateTime (from Excel serial)
      • Date (midnight-normalized pandas Timestamp), Hour/Minute/Second (zero-padded strings)
      • Segment (int|NaN) and File Stem (base without trailing segment or extension)
      • Parsed syllable onset/offset dicts
      • 'syllables_present' and 'syllable_order' per segment
      • (optional) per-label duration columns: 'syllable_<label>_durations' (ms)

    Expected decoded JSON
    ---------------------
    {
      "results": [
        {
          "file_name": "R08_45765.2741464_4_18_0_45_41[_SEG?]",
          "song_present": true/false,
          "syllable_onsets_offsets_ms": {...},
          "syllable_onsets_offsets_timebins": {...},
          ...
        }, ...
      ]
    }
    """
    decoded_database_json = Path(decoded_database_json)

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
    for col in ["Animal ID", "Date", "Hour", "Minute", "Second", "Segment", "File Stem", "Recording DateTime"]:
        if col not in organized.columns:
            organized[col] = None

    # Filename-derived fields using the Excel serial (helpers must exist in module)
    if "file_name" not in organized.columns:
        raise KeyError("Expected column 'file_name' in decoded database JSON.")

    # Process each file_name
    for i, file_field in enumerate(organized["file_name"]):
        animal_id, serial_val, segment, file_stem = parse_filename_with_excel_serial(file_field)

        # Base identity fields
        organized.at[i, "Animal ID"] = animal_id
        organized.at[i, "Segment"]   = segment
        organized.at[i, "File Stem"] = file_stem

        # Keep original upstream value for traceability
        raw_name = str(file_field).split("/")[-1]
        organized.at[i, "file_name_upstream"] = raw_name

        # ── Fix upstream ".wav" removal only if needed ─────────────────────────
        # If the base name already ends with ".wav" (case-insensitive), leave it as-is.
        # Otherwise, reconstruct "<file_stem>.wav". If parsing failed, add ".wav" only
        # if the base name lacks any extension.
        if raw_name.lower().endswith(".wav"):
            corrected_name = raw_name
        else:
            if file_stem:
                corrected_name = f"{file_stem}.wav"
            else:
                # Fallback: append .wav only if there is no extension present
                if "." in raw_name.rsplit("/", 1)[-1]:
                    corrected_name = raw_name  # some other extension present; leave as-is
                else:
                    corrected_name = raw_name + ".wav"

        # Overwrite file_name column with corrected base name
        organized.at[i, "file_name"] = corrected_name

        # Build datetime exclusively from the Excel serial
        ts = excel_serial_to_timestamp(serial_val) if (serial_val is not None) else None
        if add_recording_datetime and (ts is not None):
            organized.at[i, "Recording DateTime"] = ts
            # Fill temporary HMS; we’ll re-coerce below to guarantee types
            organized.at[i, "Date"]   = ts.normalize()
            organized.at[i, "Hour"]   = str(ts.hour).zfill(2)
            organized.at[i, "Minute"] = str(ts.minute).zfill(2)
            organized.at[i, "Second"] = str(ts.second).zfill(2)
        else:
            organized.at[i, "Recording DateTime"] = pd.NaT
            organized.at[i, "Date"]   = pd.NaT
            organized.at[i, "Hour"]   = None
            organized.at[i, "Minute"] = None
            organized.at[i, "Second"] = None

    # ── Enforce datetime dtype & regenerate Date/HMS consistently ─────────────
    organized["Recording DateTime"] = pd.to_datetime(organized["Recording DateTime"], errors="coerce")
    # Derive/normalize Date from Recording DateTime to avoid mixed types
    organized["Date"] = organized["Recording DateTime"].dt.normalize()

    # Derive HMS from Recording DateTime (only where present)
    hm_mask = organized["Recording DateTime"].notna()
    organized.loc[hm_mask, "Hour"]   = organized.loc[hm_mask, "Recording DateTime"].dt.hour.astype(int).astype(str).str.zfill(2)
    organized.loc[hm_mask, "Minute"] = organized.loc[hm_mask, "Recording DateTime"].dt.minute.astype(int).astype(str).str.zfill(2)
    organized.loc[hm_mask, "Second"] = organized.loc[hm_mask, "Recording DateTime"].dt.second.astype(int).astype(str).str.zfill(2)

    # Dict copy for convenience
    if "syllable_onsets_offsets_ms" in organized.columns:
        organized["syllable_onsets_offsets_ms_dict"] = organized["syllable_onsets_offsets_ms"]
    else:
        # broadcast empty dict if column absent
        organized["syllable_onsets_offsets_ms_dict"] = {}

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
    def _extract_syllable_order(label_to_intervals: dict, *, onset_index: int = 0) -> List[str]:
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

    organized["syllable_order"] = organized["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: _extract_syllable_order(d, onset_index=0)
    )

    # Optional: per-label durations (ms)
    def _calculate_syllable_durations_ms(label_to_intervals: dict, syllable_label: str) -> List[float]:
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

    if compute_durations and unique_labels:
        for lab in unique_labels:
            colname = f"syllable_{lab}_durations"
            organized[colname] = organized["syllable_onsets_offsets_ms_dict"].apply(
                lambda d, L=lab: _calculate_syllable_durations_ms(d, L)
            )

    # Unique recording dates as strings (robust to dtype)
    date_series = pd.to_datetime(organized["Date"], errors="coerce")
    unique_dates = (
        date_series.dropna()
        .dt.strftime("%Y.%m.%d")
        .unique()
        .tolist()
    )
    unique_dates.sort()

    # Reorder so Segment sits beside file_name
    if "file_name" in organized.columns and "Segment" in organized.columns:
        cols = list(organized.columns)
        if "Segment" in cols:
            cols.remove("Segment")
            insert_at = cols.index("file_name") + 1 if "file_name" in cols else 0
            cols = cols[:insert_at] + ["Segment"] + cols[insert_at:]
            organized = organized[cols]

    return OrganizedDataset(
        organized_df=organized,
        unique_dates=unique_dates,
        unique_syllable_labels=unique_labels,
        treatment_date=None,  # no metadata used
    )


# ───────────────────────────
# Example usage
# ───────────────────────────
"""
from pathlib import Path
from organized_decoded_serialTS_segments import build_organized_segments_with_durations

decoded = Path("/Volumes/my_own_ssd/2025_areax_lesion/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json")

out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    only_song_present=True,   # or False
    compute_durations=True,
)

df = out.organized_df.sort_values(["File Stem","Segment","Recording DateTime"])
df.head()

# Time fields now come from the Excel serial:
#   'Recording DateTime' (full timestamp)
#   'Date' (midnight-normalized)
#   'Hour','Minute','Second' (zero-padded strings)

# The module now also:
#   - Preserves original upstream 'file_name' in 'file_name_upstream'
#   - Rewrites 'file_name' to '<file_stem>.wav' ONLY if it didn't already end with '.wav'
"""
