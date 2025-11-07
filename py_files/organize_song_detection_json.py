# organize_song_detection_json.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import re
import json
from datetime import datetime, timedelta
import pandas as pd

__all__ = [
    "build_meta_dataframe",
    "filter_songs",
    "make_segments_long_table",
    "build_detected_song_segments",   # one-call wrapper → returns long table
    "filter_flagged_segments",        # helper → returns only flagged rows
    "print_flagged_summary",          # helper → prints counts + sample rows
]

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
_EXCEL_1900_EPOCH = datetime(1899, 12, 30)  # Excel "1900" date system epoch

def _excel_serial_to_datetime(serial: Union[str, float, int, None]) -> Optional[datetime]:
    """
    Convert an Excel '1900 date system' serial number to a naive datetime.
    Returns None if parsing fails.
    """
    if serial is None:
        return None
    try:
        serial_f = float(serial)
    except Exception:
        return None
    # Negative/NaN safety
    if not pd.notna(serial_f) or serial_f < 0:
        return None
    return _EXCEL_1900_EPOCH + timedelta(days=serial_f)

_FLOAT_IN_FILENAME = re.compile(r"(\d{4,}\.\d+)")  # e.g., 45765.26224434

def _parse_excel_serial_from_filename(filename: Optional[str]) -> Optional[float]:
    """
    Extract the first long float token from filename (e.g., '45765.26224434').
    Returns None if not found.
    """
    if not filename:
        return None
    m = _FLOAT_IN_FILENAME.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _ensure_list(obj):
    if obj is None:
        return None
    return obj if isinstance(obj, list) else [obj]

def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"true", "1", "yes", "y"}

def _load_records(json_input: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Accept a path to a .json file, a JSON string '[{...}, ...]', or a list of dicts.
    Returns a list of dicts.
    """
    if isinstance(json_input, list):
        return json_input

    s = str(json_input)
    if s.strip().startswith(("[", "{")):
        return json.loads(s)

    p = Path(s)
    with p.open("r") as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────
# Wide table builder
# ──────────────────────────────────────────────────────────────
def build_meta_dataframe(json_input: Union[str, Path, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Build a 'wide' table with columns:
      - filename (str)
      - song_present (bool)
      - spec_parameters (dict or None)
      - segments (list of dicts or None)
      - recording_datetime (Timestamp, from Excel serial in filename)
      - recording_date (YYYY-MM-DD string)
      - recording_time (HH:MM:SS.mmm string)
    """
    records = _load_records(json_input)

    rows = []
    for rec in records:
        filename        = rec.get("filename")
        song_present    = _as_bool(rec.get("song_present", False))
        spec_parameters = rec.get("spec_parameters")
        # tolerate accidental key 'semgents'
        segments        = rec.get("segments", rec.get("semgents"))

        # Parse Excel serial from filename → datetime/date/time
        serial = _parse_excel_serial_from_filename(filename)
        dt = _excel_serial_to_datetime(serial)
        if dt is not None:
            date_str = dt.date().isoformat()
            # Use milliseconds for precision from fractional day
            time_str = dt.time().isoformat(timespec="milliseconds")
            dt_ts = pd.Timestamp(dt)
        else:
            date_str = None
            time_str = None
            dt_ts = pd.NaT

        rows.append({
            "filename": filename,
            "song_present": song_present,
            "spec_parameters": spec_parameters,
            "segments": segments,
            "recording_datetime": dt_ts,
            "recording_date": date_str,
            "recording_time": time_str,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "song_present",
            "spec_parameters",
            "segments",
            "recording_datetime",
            "recording_date",
            "recording_time",
        ],
    )

def filter_songs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where song_present is True."""
    return df.loc[df["song_present"]].copy()

# ──────────────────────────────────────────────────────────────
# Long-form (one row per segment) with lengths + gap from previous segment
# ──────────────────────────────────────────────────────────────
def make_segments_long_table(
    df: pd.DataFrame,
    *,
    drop_no_segments: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 1000,   # ms, flag if gap < this value
) -> pd.DataFrame:
    """
    Build a long-form table with one row per segment.

    Columns produced:
      - filename
      - recording_datetime, recording_date, recording_time
      - song_present
      - segment_index (0-based within file)
      - onset_timebin, offset_timebin
      - onset_ms, offset_ms
      - duration_timebins (offset_timebin - onset_timebin, if both present)
      - duration_ms (offset_ms - onset_ms, if both present)
      - length_timebins (same as duration_timebins when non-negative else None)
      - length_ms (same as duration_ms when non-negative else None)
      - gap_from_prev_ms (onset_ms - previous segment's offset_ms, per filename; NaN for first)
      - potential_split_up_song (True if gap_from_prev_ms < max_gap_between_song_segments)
      - spec.* (flattened spec_parameters, if requested)
      - record_index (original row index in the wide table)
    """
    long_rows: List[Dict[str, Any]] = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        segments = row.get("segments")
        if segments is None:
            if not drop_no_segments:
                long_rows.append({
                    "filename": row.get("filename"),
                    "recording_datetime": row.get("recording_datetime"),
                    "recording_date": row.get("recording_date"),
                    "recording_time": row.get("recording_time"),
                    "song_present": row.get("song_present"),
                    "segment_index": None,
                    "onset_timebin": None,
                    "offset_timebin": None,
                    "onset_ms": None,
                    "offset_ms": None,
                    "duration_timebins": None,
                    "duration_ms": None,
                    "length_timebins": None,
                    "length_ms": None,
                    "spec_parameters": row.get("spec_parameters"),
                    "record_index": row_idx,
                })
            continue

        if not isinstance(segments, list):
            segments = _ensure_list(segments) or []

        if len(segments) == 0:
            if not drop_no_segments:
                long_rows.append({
                    "filename": row.get("filename"),
                    "recording_datetime": row.get("recording_datetime"),
                    "recording_date": row.get("recording_date"),
                    "recording_time": row.get("recording_time"),
                    "song_present": row.get("song_present"),
                    "segment_index": None,
                    "onset_timebin": None,
                    "offset_timebin": None,
                    "onset_ms": None,
                    "offset_ms": None,
                    "duration_timebins": None,
                    "duration_ms": None,
                    "length_timebins": None,
                    "length_ms": None,
                    "spec_parameters": row.get("spec_parameters"),
                    "record_index": row_idx,
                })
            continue

        for seg_idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

            on_tb  = seg.get("onset_timebin")
            off_tb = seg.get("offset_timebin")
            on_ms  = seg.get("onset_ms")
            off_ms = seg.get("offset_ms")

            # raw differences
            dur_tb = (off_tb - on_tb) if (on_tb is not None and off_tb is not None) else None
            dur_ms = (off_ms - on_ms) if (on_ms is not None and off_ms is not None) else None

            # non-negative lengths (None if negative or missing)
            len_tb = dur_tb if (isinstance(dur_tb, (int, float)) and dur_tb >= 0) else None
            len_ms = dur_ms if (isinstance(dur_ms, (int, float)) and dur_ms >= 0) else None

            long_rows.append({
                "filename": row.get("filename"),
                "recording_datetime": row.get("recording_datetime"),
                "recording_date": row.get("recording_date"),
                "recording_time": row.get("recording_time"),
                "song_present": row.get("song_present"),
                "segment_index": seg_idx,
                "onset_timebin": on_tb,
                "offset_timebin": off_tb,
                "onset_ms": on_ms,
                "offset_ms": off_ms,
                "duration_timebins": len_tb if len_tb is not None else None,
                "duration_ms": len_ms if len_ms is not None else None,
                "length_timebins": len_tb,
                "length_ms": len_ms,
                "spec_parameters": row.get("spec_parameters"),
                "record_index": row_idx,
            })

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df

    # Compute gap from previous segment within each file
    long_df = long_df.sort_values(["filename", "segment_index"], kind="mergesort")
    prev_offset = long_df.groupby("filename", dropna=False)["offset_ms"].shift(1)
    long_df["gap_from_prev_ms"] = long_df["onset_ms"] - prev_offset  # NaN for first of each file

    # Flag potential split-up songs using the configurable threshold
    long_df["potential_split_up_song"] = (long_df["gap_from_prev_ms"] < max_gap_between_song_segments).fillna(False)

    # Backward-compat alias for earlier misspelling (safe to remove later)
    long_df["potential_split_up_sopng"] = long_df["potential_split_up_song"]

    # Optionally flatten spec params into columns
    if flatten_spec_params and "spec_parameters" in long_df.columns:
        spec_flat = pd.json_normalize(long_df["spec_parameters"]).add_prefix("spec.")
        long_df = pd.concat([long_df.drop(columns=["spec_parameters"]), spec_flat], axis=1)

    return long_df

# ──────────────────────────────────────────────────────────────
# One-call convenience wrapper that returns ONLY the long table
# ──────────────────────────────────────────────────────────────
def build_detected_song_segments(
    json_input: Union[str, Path, List[Dict[str, Any]]],
    *,
    songs_only: bool = True,
    drop_no_segments: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 1000,   # ms
) -> pd.DataFrame:
    """
    Return ONLY the long-form table of detected song segments.
    """
    wide_df = build_meta_dataframe(json_input)
    df_in   = filter_songs(wide_df) if songs_only else wide_df
    return make_segments_long_table(
        df_in,
        drop_no_segments=drop_no_segments,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )

# ──────────────────────────────────────────────────────────────
# Helpers to get/print flagged rows
# ──────────────────────────────────────────────────────────────
def filter_flagged_segments(df: pd.DataFrame, *, flag_col: str = "potential_split_up_song") -> pd.DataFrame:
    """Return only rows where the flag column is True."""
    if flag_col not in df.columns and "potential_split_up_sopng" in df.columns:
        flag_col = "potential_split_up_sopng"
    return df.loc[df[flag_col].fillna(False)].copy()

def print_flagged_summary(
    df: pd.DataFrame,
    *,
    flag_col: str = "potential_split_up_song",
    max_rows_per_file: int = 10,
) -> None:
    """Print 'n out of x' and show a few flagged rows per filename."""
    if flag_col not in df.columns and "potential_split_up_sopng" in df.columns:
        flag_col = "potential_split_up_sopng"

    flagged = df.loc[df[flag_col].fillna(False)].copy()
    print(f"Number of potentially split up songs: {len(flagged)} out of {len(df)} segments")

    if flagged.empty:
        return

    flagged = flagged.sort_values(["filename", "segment_index"])
    cols_to_show = [
        "segment_index", "onset_ms", "offset_ms", "gap_from_prev_ms",
        "length_ms", "onset_timebin", "offset_timebin", "length_timebins",
        "recording_date", "recording_time",
    ]
    for fname, g in flagged.groupby("filename"):
        print(f"\n=== {fname} — {len(g)} flagged segment(s) ===")
        show_cols = [c for c in cols_to_show if c in g.columns]
        print(g[show_cols].head(max_rows_per_file).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# Optional: quick local run for testing (creates 'detected_song_segments')
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_json = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
    detected_song_segments = build_detected_song_segments(
        example_json,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=500,  # tweak here
    )
    print(f"SEGMENT rows: {detected_song_segments.shape[0]}")

    # Print summary + sample flagged rows
    print_flagged_summary(detected_song_segments)

    # Also expose 'flagged' in the Variable Explorer
    flagged = filter_flagged_segments(detected_song_segments)



"""
from organize_song_detection_json import (
    build_detected_song_segments,
    filter_flagged_segments,
    print_flagged_summary,
)

detected_song_segments = build_detected_song_segments(
    "/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json",
    max_gap_between_song_segments=500,
)

print(f"Number of potentially split up songs: {int(detected_song_segments['potential_split_up_song'].fillna(False).sum())} out of {len(detected_song_segments)} segments")

# Work with the flagged dataframe in Variable Explorer
flagged = filter_flagged_segments(detected_song_segments)
"""
