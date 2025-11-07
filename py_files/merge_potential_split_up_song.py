# merge_potential_split_up_song.py
# -*- coding: utf-8 -*-
"""
Build dataframes that merge "split-up" song segments and/or keep non-flagged segments.

Relies on organize_song_detection_json.build_detected_song_segments to:
  - parse JSON into long-form segments
  - compute gap_from_prev_ms
  - flag potential split-up segments via potential_split_up_song

Carries datetime fields from the organizer:
  - recording_datetime (Timestamp)
  - recording_date (YYYY-MM-DD)
  - recording_time (HH:MM:SS.mmm)

NOTE: In detected_merged_songs, `segment_index` is a LIST:
  - singles → [idx]
  - merged runs → [idx_start, ..., idx_end]
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import pandas as pd

from organize_song_detection_json import build_detected_song_segments

__all__ = [
    "build_merged_split_up_songs",
    "build_detected_merged_songs",
    "build_detected_and_merged_songs",
]


def _resolve_flag_col(df: pd.DataFrame) -> str:
    if "potential_split_up_song" in df.columns:
        return "potential_split_up_song"
    if "potential_split_up_sopng" in df.columns:  # backward-compat
        return "potential_split_up_sopng"
    raise KeyError("Neither 'potential_split_up_song' nor 'potential_split_up_sopng' found in dataframe.")


def _first_idx_for_sort(x) -> int:
    """Return an integer sort key from `segment_index` which may be a list or a scalar."""
    try:
        if isinstance(x, (list, tuple)) and len(x):
            return int(x[0])
        return int(x)
    except Exception:
        return -1


def _merge_from_detected(detected: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From an already-built detected_song_segments dataframe, compute:
      - merged_runs_df: only contiguous merged runs (size >= 2)
      - detected_merged_df: singles + merged runs (replacing split rows)
        with `segment_index` as a LIST.
    """
    if detected.empty:
        return pd.DataFrame(), pd.DataFrame()

    flag_col = _resolve_flag_col(detected)
    base_cols = list(detected.columns)
    dt_cols = [c for c in ["recording_datetime", "recording_date", "recording_time"] if c in detected.columns]

    merged_rows: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []

    for fname, g in detected.groupby("filename", dropna=False):
        g = g.sort_values("segment_index").reset_index(drop=True)

        # Group id: start a new group whenever current row is NOT joining previous
        join_with_prev = g[flag_col].fillna(False).to_numpy(dtype=bool)
        grp_id = (~join_with_prev).cumsum()
        g = g.assign(_grp_id=grp_id)

        for _, grp in g.groupby("_grp_id", sort=True):
            if len(grp) == 1:
                # --- single (non-merged) row ---
                r = grp.iloc[0].to_dict()
                # make segment_index a list for consistency
                r["segment_index"] = [int(r["segment_index"])]
                r["was_merged"] = False
                r["merged_n_parts"] = 1
                r["merged_segment_indices"] = r["segment_index"].copy()
                r["merged_segment_index_start"] = r["segment_index"][0]
                r["merged_segment_index_end"] = r["segment_index"][0]
                for c in base_cols:
                    r.setdefault(c, None)
                combined_rows.append(r)
                continue

            # --- merged run (>= 2 rows) ---
            first = grp.iloc[0]
            last  = grp.iloc[-1]

            # Record for "merged runs only" table
            onset_tb  = first["onset_timebin"]
            offset_tb = last["offset_timebin"]
            onset_ms  = first["onset_ms"]
            offset_ms = last["offset_ms"]

            merged_len_tb = (offset_tb - onset_tb) if pd.notna(onset_tb) and pd.notna(offset_tb) else None
            merged_len_ms = (offset_ms - onset_ms) if pd.notna(onset_ms) and pd.notna(offset_ms) else None

            merged_rec: Dict[str, Any] = {
                "filename": fname,
                "merged_segment_start_index": int(first["segment_index"]),
                "merged_segment_end_index":   int(last["segment_index"]),
                "segment_indices":            list(grp["segment_index"].astype(int)),
                "n_parts":                    int(len(grp)),
                "merged_onset_timebin":       onset_tb,
                "merged_offset_timebin":      offset_tb,
                "merged_onset_ms":            onset_ms,
                "merged_offset_ms":           offset_ms,
                "merged_length_timebins":     merged_len_tb,
                "merged_length_ms":           merged_len_ms,
            }
            for c in dt_cols:
                merged_rec[c] = first.get(c)
            for c in [c for c in grp.columns if c.startswith("spec.")]:
                vals = grp[c].dropna().unique()
                merged_rec[c] = vals[0] if len(vals) == 1 else None
            merged_rows.append(merged_rec)

            # Record for "combined (singles + merged)" table
            comb = first.to_dict()
            # segment_index becomes the full list of indices in this merged run
            comb["segment_index"] = list(grp["segment_index"].astype(int))
            comb["was_merged"] = True
            comb["merged_n_parts"] = int(len(grp))
            comb["merged_segment_indices"] = comb["segment_index"].copy()
            comb["merged_segment_index_start"] = int(first["segment_index"])
            comb["merged_segment_index_end"] = int(last["segment_index"])

            # span full run
            comb["onset_timebin"] = first["onset_timebin"]
            comb["offset_timebin"] = last["offset_timebin"]
            comb["onset_ms"] = first["onset_ms"]
            comb["offset_ms"] = last["offset_ms"]

            if pd.notna(comb["onset_timebin"]) and pd.notna(comb["offset_timebin"]):
                comb["length_timebins"] = comb["offset_timebin"] - comb["onset_timebin"]
                comb["duration_timebins"] = comb["length_timebins"]
            else:
                comb["length_timebins"] = None
                comb["duration_timebins"] = None

            if pd.notna(comb["onset_ms"]) and pd.notna(comb["offset_ms"]):
                comb["length_ms"] = comb["offset_ms"] - comb["onset_ms"]
                comb["duration_ms"] = comb["length_ms"]
            else:
                comb["length_ms"] = None
                comb["duration_ms"] = None

            # first row's gap is the gap from the previous group
            comb["gap_from_prev_ms"] = first.get("gap_from_prev_ms")

            # merged rows are no longer considered split
            comb["potential_split_up_song"] = False
            if "potential_split_up_sopng" in detected.columns:
                comb["potential_split_up_sopng"] = False

            for c in base_cols:
                comb.setdefault(c, None)
            combined_rows.append(comb)

    merged_runs_df = pd.DataFrame(merged_rows)
    if not merged_runs_df.empty:
        merged_runs_df = merged_runs_df.sort_values(
            ["filename", "merged_segment_start_index", "merged_segment_end_index"]
        ).reset_index(drop=True)

    detected_merged_df = pd.DataFrame(combined_rows)
    if not detected_merged_df.empty:
        # sort by filename and the first index in the (possibly list) segment_index
        detected_merged_df["_sort_idx"] = detected_merged_df["segment_index"].apply(_first_idx_for_sort)
        detected_merged_df = detected_merged_df.sort_values(
            ["filename", "_sort_idx"]
        ).drop(columns=["_sort_idx"]).reset_index(drop=True)

    return merged_runs_df, detected_merged_df


# ──────────────────────────────────────────────────────────────
# Public entrypoints
# ──────────────────────────────────────────────────────────────
def build_merged_split_up_songs(
    json_input: Union[str, Path],
    *,
    songs_only: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 1000,
) -> pd.DataFrame:
    detected = build_detected_song_segments(
        json_input,
        songs_only=songs_only,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )
    merged_runs, _ = _merge_from_detected(detected)
    return merged_runs


def build_detected_merged_songs(
    json_input: Union[str, Path],
    *,
    songs_only: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 1000,
) -> pd.DataFrame:
    detected = build_detected_song_segments(
        json_input,
        songs_only=songs_only,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )
    _, detected_merged = _merge_from_detected(detected)
    return detected_merged


def build_detected_and_merged_songs(
    json_input: Union[str, Path],
    *,
    songs_only: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 1000,
) -> Dict[str, pd.DataFrame]:
    detected = build_detected_song_segments(
        json_input,
        songs_only=songs_only,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )
    merged_runs, detected_merged = _merge_from_detected(detected)
    return {
        "detected_song_segments": detected,
        "merged_split_up_songs": merged_runs,
        "detected_merged_songs": detected_merged,
    }


# ──────────────────────────────────────────────────────────────
# Quick local test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_json = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
    out = build_detected_and_merged_songs(
        example_json,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=500,
    )
    detected_song_segments = out["detected_song_segments"]
    merged_split_up_songs  = out["merged_split_up_songs"]
    detected_merged_songs  = out["detected_merged_songs"]

    print("Original detected rows:", len(detected_song_segments))
    print("Merged runs (size >= 2):", len(merged_split_up_songs))
    print("Detected + merged rows:", len(detected_merged_songs))
    print(
        detected_merged_songs[
            ["filename", "segment_index", "was_merged", "merged_n_parts", "merged_segment_indices",
             "onset_ms", "offset_ms", "length_ms", "recording_date", "recording_time"]
        ].head(12).to_string(index=False)
    )



"""
from merge_potential_split_up_song import build_detected_and_merged_songs

out = build_detected_and_merged_songs(
    "/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json",
    max_gap_between_song_segments=500,
)
detected_song_segments = out["detected_song_segments"]
merged_split_up_songs  = out["merged_split_up_songs"]
detected_merged_songs  = out["detected_merged_songs"]
"""
