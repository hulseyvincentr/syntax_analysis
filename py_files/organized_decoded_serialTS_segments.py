# -*- coding: utf-8 -*-
# organized_decoded_serialTS_segments.py
from __future__ import annotations

import json
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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
        ts = pd.to_datetime(val, unit="D", origin="1899-12-30")
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts[0]
        return pd.Timestamp(ts)
    except Exception:
        return None


def parse_filename_with_excel_serial(
    file_field: Union[str, Path]
) -> Tuple[
    Optional[str],  # animal_id
    Optional[float],  # excel_serial
    Optional[int],  # segment index
    Optional[str],  # file_stem (without extension and without trailing segment)
]:
    """
    Parse filenames in either of two styles.

    Old style (no leading index, serial as one token with decimal):
        <ANIMAL> _ <EXCEL_SERIAL> _ <M> _ <D> _ <H> _ <M> _ <S> [_<SEG>]

      Examples
      --------
      "R08_45765.2741464_4_18_0_45_41"
      "USA5497_45444.26577576_6_1_7_22_57_0"

    New style (leading index + double underscore, serial split into int + frac,
    and an explicit "segment_<N>" suffix):

        <IDX>__<ANIMAL>_<EXCEL_INT>_<EXCEL_FRAC>_<M>_<D>_<H>_<M>_<S>_segment_<SEG>

      Example
      -------
      "1__USA5271_45342_27408656_2_20_7_36_48_segment_0"

    Returns
    -------
    (animal_id, serial_float, segment_or_None, file_stem_without_segment)
    """
    try:
        raw = str(file_field).split("/")[-1]
        no_ext = _strip_ext(raw)

        # Handle new-style prefix like "1__USA5271_..."
        if "__" in no_ext:
            _, rest = no_ext.split("__", 1)
            parts = rest.split("_")
        else:
            parts = no_ext.split("_")

        if len(parts) < 2:
            return None, None, None, None

        animal_id = parts[0]

        segment: Optional[int] = None
        base_parts = parts

        # New-style: serial split into int + frac
        is_new_style_serial = (
            len(parts) >= 3
            and parts[1].isdigit()
            and parts[2].isdigit()
        )

        # Explicit "segment_<n>" suffix (new-style)
        if "segment" in parts:
            seg_idx = parts.index("segment")
            if seg_idx + 1 < len(parts) and parts[seg_idx + 1].isdigit():
                segment = int(parts[seg_idx + 1])
                base_parts = parts[:seg_idx]  # drop "segment" and index
        else:
            # Old style: last token is segment if not in new-style pattern
            if (not is_new_style_serial) and len(parts) >= 8 and parts[-1].isdigit():
                segment = int(parts[-1])
                base_parts = parts[:-1]

        # Serial string
        serial_str: Optional[str] = None
        if is_new_style_serial:
            # "<ANIMAL>_<INT>_<FRAC>_..."
            serial_str = f"{parts[1]}.{parts[2]}"
        elif len(parts) >= 2:
            serial_str = parts[1]

        try:
            serial_val = float(serial_str) if serial_str is not None else None
        except Exception:
            serial_val = None

        file_stem = "_".join(base_parts) if base_parts else None
        return animal_id, serial_val, segment, file_stem

    except Exception:
        return None, None, None, None


# ───────────────────────────
# Core organizer
# ───────────────────────────
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
    filename (the token immediately after the animal ID, possibly split into
    int+frac). No metadata file is needed.

    Adds/keeps:
      • Animal ID
      • Recording DateTime (from Excel serial)
      • Date (midnight-normalized pandas Timestamp), Hour/Minute/Second (zero-padded strings)
      • Segment (int|NaN) and File Stem (base without trailing segment or extension)
      • Parsed syllable onset/offset dicts
      • 'syllables_present' and 'syllable_order' per segment
      • (optional) per-label duration columns: 'syllable_<label>_durations' (ms)
    """
    decoded_database_json = Path(decoded_database_json)

    # Load decoded data (JSON only, expect 'results')
    with decoded_database_json.open("r") as f:
        decoded_payload = json.load(f)

    results = decoded_payload.get("results", [])
    df = pd.DataFrame(results)

    # ── Handle completely empty decoded files gracefully ────────────────────
    if df.empty:
        # Nothing to organize: return a clean, empty container instead of erroring.
        return OrganizedDataset(
            organized_df=df.copy(),
            unique_dates=[],
            unique_syllable_labels=[],
            treatment_date=None,
        )

    # Parse potentially stringified dict columns (older / variant schemas)
    if "syllable_onsets_offsets_ms" in df.columns:
        df["syllable_onsets_offsets_ms"] = df["syllable_onsets_offsets_ms"].apply(parse_json_safe)
    if "syllable_onsets_offsets_timebins" in df.columns:
        df["syllable_onsets_offsets_timebins"] = df["syllable_onsets_offsets_timebins"].apply(parse_json_safe)
    if "syllable_onsets_offsets_ms_dict" in df.columns:
        df["syllable_onsets_offsets_ms_dict"] = df["syllable_onsets_offsets_ms_dict"].apply(parse_json_safe)

    organized = df.copy()

    # Optional filter
    if only_song_present and "song_present" in organized.columns:
        organized = organized[organized["song_present"] == True].reset_index(drop=True)

    # Ensure target columns exist (don’t drop unknown original columns)
    for col in ["Animal ID", "Date", "Hour", "Minute", "Second", "Segment", "File Stem", "Recording DateTime"]:
        if col not in organized.columns:
            organized[col] = None

    # Filename-derived fields using the Excel serial
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

        # Fix upstream ".wav" removal only if needed
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

    # Enforce datetime dtype & regenerate Date/HMS consistently
    organized["Recording DateTime"] = pd.to_datetime(organized["Recording DateTime"], errors="coerce")
    organized["Date"] = organized["Recording DateTime"].dt.normalize()

    hm_mask = organized["Recording DateTime"].notna()
    organized.loc[hm_mask, "Hour"]   = (
        organized.loc[hm_mask, "Recording DateTime"].dt.hour.astype(int).astype(str).str.zfill(2)
    )
    organized.loc[hm_mask, "Minute"] = (
        organized.loc[hm_mask, "Recording DateTime"].dt.minute.astype(int).astype(str).str.zfill(2)
    )
    organized.loc[hm_mask, "Second"] = (
        organized.loc[hm_mask, "Recording DateTime"].dt.second.astype(int).astype(str).str.zfill(2)
    )

    # ── Normalise / build onset-offset dict column ──────────────────────────
    # Prefer an explicit dict column if present, otherwise fall back to the
    # older 'syllable_onsets_offsets_ms' column. If neither exists, we'll
    # optionally rebuild from syllable_*_durations below.
    if "syllable_onsets_offsets_ms_dict" in organized.columns:
        base_series = organized["syllable_onsets_offsets_ms_dict"].apply(parse_json_safe)
    elif "syllable_onsets_offsets_ms" in organized.columns:
        base_series = organized["syllable_onsets_offsets_ms"].apply(parse_json_safe)
    else:
        base_series = pd.Series([{} for _ in range(len(organized))], index=organized.index)

    organized["syllable_onsets_offsets_ms_dict"] = base_series

    # Helper to collect labels from the dict column
    def _collect_unique_labels(series: pd.Series) -> List[str]:
        label_set: set[str] = set()
        for row in series:
            if isinstance(row, dict) and row:
                label_set.update(row.keys())
        return sorted(label_set)

    unique_labels: List[str] = _collect_unique_labels(organized["syllable_onsets_offsets_ms_dict"])

    # ── Fallback: if dicts are empty but legacy duration columns exist,
    #    reconstruct pseudo intervals from syllable_*_durations. ────────────
    if not unique_labels:
        duration_cols = [
            c for c in organized.columns
            if isinstance(c, str) and c.startswith("syllable_") and c.endswith("_durations")
        ]

        if duration_cols:
            def _rebuild_intervals_from_durations(row: pd.Series) -> dict:
                out: Dict[str, List[List[float]]] = {}
                for col in duration_cols:
                    m = re.match(r"^syllable_(.+)_durations$", col)
                    if not m:
                        continue
                    label = m.group(1)
                    durs = row[col]
                    if not isinstance(durs, (list, tuple, np.ndarray)) or len(durs) == 0:
                        continue
                    # fabricate onsets as cumulative sum of previous durations
                    durs_list = list(durs)
                    onsets = np.cumsum([0.0] + durs_list[:-1])
                    out[label] = [[float(on), float(on + d)] for on, d in zip(onsets, durs_list)]
                return out

            organized["syllable_onsets_offsets_ms_dict"] = organized.apply(
                _rebuild_intervals_from_durations,
                axis=1,
            )
            unique_labels = _collect_unique_labels(organized["syllable_onsets_offsets_ms_dict"])

    # ── Per-segment syllables present ───────────────────────────────────────
    def _extract_syllables_present(v: dict) -> List[str]:
        return sorted(v.keys()) if isinstance(v, dict) else []

    organized["syllables_present"] = organized["syllable_onsets_offsets_ms_dict"].apply(
        _extract_syllables_present
    )

    # ── Per-segment syllable order (sorted by onset time in ms) ────────────
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

    # ── Optional: per-label durations (ms) computed from the dict ──────────
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

    # ── Unique recording dates as strings ──────────────────────────────────
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
        treatment_date=None,
    )

# ───────────────────────────
# Example usage
# ───────────────────────────
"""
from pathlib import Path
from organized_decoded_serialTS_segments import build_organized_segments_with_durations

decoded = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5271/USA5271_decoded_database.json")

out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    only_song_present=True,   # or False
    compute_durations=True,
)

df = out.organized_df.sort_values(["File Stem","Segment","Recording DateTime"])
df.head()

# For a new-style name like:
#   "1__USA5271_45342_27408656_2_20_7_36_48_segment_0"
# this will:
#   - store 'file_name_upstream' as the original string
#   - parse the Excel serial 45342.27408656
#   - set 'file_name' to:
#       "USA5271_45342_27408656_2_20_7_36_48.wav"

# Time fields now come from the Excel serial:
#   'Recording DateTime' (full timestamp)
#   'Date' (midnight-normalized)
#   'Hour','Minute','Second' (zero-padded strings)
"""
