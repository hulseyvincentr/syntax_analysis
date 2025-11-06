# organize_song_detection_json.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import json
import pandas as pd

__all__ = [
    "build_meta_dataframe",
    "filter_songs",
    "make_segments_long_table",
    "build_detected_song_segments",   # ← main entrypoint for other files
]
 
# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
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
    """
    records = _load_records(json_input)

    rows = []
    for rec in records:
        filename        = rec.get("filename")
        song_present    = _as_bool(rec.get("song_present", False))
        spec_parameters = rec.get("spec_parameters")
        # tolerate accidental key 'semgents'
        segments        = rec.get("segments", rec.get("semgents"))

        rows.append({
            "filename": filename,
            "song_present": song_present,
            "spec_parameters": spec_parameters,
            "segments": segments,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "song_present",
            "spec_parameters",
            "segments",
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
) -> pd.DataFrame:
    """
    Build a long-form table with one row per segment.

    Columns produced:
      - filename
      - song_present
      - segment_index (0-based within file)
      - onset_timebin, offset_timebin
      - onset_ms, offset_ms
      - duration_timebins (offset_timebin - onset_timebin, if both present)
      - duration_ms (offset_ms - onset_ms, if both present)
      - length_timebins (same as duration_timebins when non-negative else None)
      - length_ms (same as duration_ms when non-negative else None)
      - gap_from_prev_ms (onset_ms - previous segment's offset_ms, per filename; NaN for first)
      - potential_split_up_song (True if gap_from_prev_ms < 1000 ms)
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
                "song_present": row.get("song_present"),
                "segment_index": seg_idx,
                "onset_timebin": on_tb,
                "offset_timebin": off_tb,
                "onset_ms": on_ms,
                "offset_ms": off_ms,
                "duration_timebins": dur_tb,
                "duration_ms": dur_ms,
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

    # Flag potential split-up songs: gap < 1000 ms
    # (NaN comparisons yield False automatically; first segments remain False)
    long_df["potential_split_up_song"] = long_df["gap_from_prev_ms"] < 1000

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
) -> pd.DataFrame:
    """
    Return ONLY the long-form table of detected song segments.
    """
    wide_df = build_meta_dataframe(json_input)
    df_in   = filter_songs(wide_df) if songs_only else wide_df
    return make_segments_long_table(
        df_in, drop_no_segments=drop_no_segments, flatten_spec_params=flatten_spec_params
    )

# ──────────────────────────────────────────────────────────────
# Optional: quick local run for testing (creates 'detected_song_segments')
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_json = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
    detected_song_segments = build_detected_song_segments(
        example_json, songs_only=True, flatten_spec_params=True
    )
    print(f"SEGMENT rows: {detected_song_segments.shape[0]}")

    # Show flagged potential split-up songs (gap < 1000 ms)
    flagged = detected_song_segments.loc[detected_song_segments["potential_split_up_song"]].copy()
    print(f"\nFiles with potential split-up songs: {flagged['filename'].nunique()} "
          f"(rows flagged: {len(flagged)})")

    # Print filename + segment rows that were flagged
    if not flagged.empty:
        cols_to_show = [
            "segment_index", "onset_ms", "offset_ms", "gap_from_prev_ms",
            "length_ms", "onset_timebin", "offset_timebin", "length_timebins"
        ]
        for fname, g in flagged.groupby("filename"):
            print(f"\n=== {fname} — {len(g)} flagged segment(s) ===")
            show_cols = [c for c in cols_to_show if c in g.columns]
            print(g[show_cols].to_string(index=False))

    # Quick head for sanity
    cols = [
        "filename", "segment_index",
        "onset_timebin", "offset_timebin",
        "onset_ms", "offset_ms",
        "length_timebins", "length_ms",
        "gap_from_prev_ms", "potential_split_up_song",
    ]
    extra = [c for c in detected_song_segments.columns if c.startswith("spec.")]
    show = [c for c in cols + extra if c in detected_song_segments.columns]
    print("\nPreview:\n", detected_song_segments[show].head(12).to_string(index=False))



"""
from organize_song_detection_json import build_detected_song_segments

detected_song_segments = build_detected_song_segments(
    "/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json",
    songs_only=True,
    flatten_spec_params=True,
    )

# Filter just the flagged rows if you want
flagged = detected_song_segments.loc[detected_song_segments["potential_split_up_song"]]
print(f"Number of potentially split up songs: {int(detected_song_segments['potential_split_up_song'].fillna(False).sum())} out of {len(detected_song_segments)} segments")















"""