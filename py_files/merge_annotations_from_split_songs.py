#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_annotations_from_split_songs.py

Ask for paths to:
  - song_detection JSON
  - song_annotation (decoded database) JSON

Then:
  1) Use merge_potential_split_up_song.build_detected_and_merged_songs to find split-up songs.
  2) Use organized_decoded_serialTS_segments.build_organized_segments_with_durations
     to build a per-segment annotations dataframe.
  3) For each detected row (single or merged run), append annotation segments together:
     - Find the matching (file_name, Segment) rows in the organized annotations df.
     - Time-shift each segment's annotation by its detected onset (ms) relative to the
       first segment in the run.
     - Concatenate intervals label-by-label; sort by onset.

Outputs:
  - organized_annotations.csv (per-segment)
  - decoded_with_split_labels.csv (singles + merged annotations)

Usage (CLI flags or interactive prompts):
  python merge_annotations_from_split_songs.py \
      --song-detection /path/to/R08_RC6_Comp2_song_detection.json \
      --annotations   /path/to/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json \
      --max-gap-ms 500 \
      --segment-index-offset 0
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import argparse
import json

import pandas as pd
import numpy as np

# ── Your modules (adjust names if your file names differ) ─────────────────────
from merge_potential_split_up_song import build_detected_and_merged_songs
from organized_decoded_serialTS_segments import build_organized_segments_with_durations


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for annotation appending
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_list(x) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [int(v) for v in x]
    try:
        return [int(x)]
    except Exception:
        return []

def _compute_syllable_order(label_to_intervals: Dict[str, List[List[float]]]) -> List[str]:
    """Return labels ordered by onset, globally across all intervals."""
    pairs: List[Tuple[float, str]] = []
    for lab, intervals in (label_to_intervals or {}).items():
        for itv in intervals or []:
            if isinstance(itv, (list, tuple)) and len(itv) >= 2:
                try:
                    on = float(itv[0])
                except Exception:
                    continue
                pairs.append((on, lab))
    pairs.sort(key=lambda p: p[0])
    return [lab for _, lab in pairs]

def _shift_and_extend(acc: Dict[str, List[List[float]]],
                      piece: Dict[str, List[List[float]]],
                      shift_ms: float) -> None:
    """Shift all intervals in `piece` by `shift_ms` and extend into `acc`."""
    if not isinstance(piece, dict):
        return
    for lab, intervals in piece.items():
        if not isinstance(intervals, (list, tuple)):
            continue
        out = acc.setdefault(lab, [])
        for itv in intervals:
            if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
                continue
            try:
                on = float(itv[0]) + float(shift_ms)
                off = float(itv[1]) + float(shift_ms)
            except Exception:
                continue
            out.append([on, off])

def _sort_intervals_inplace(d: Dict[str, List[List[float]]]) -> None:
    for lab, ivals in d.items():
        ivals.sort(key=lambda ab: (ab[0], ab[1] if len(ab) > 1 else ab[0]))

def _durations_by_label(d: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for lab, ivals in (d or {}).items():
        dur = []
        for itv in ivals:
            if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
                continue
            try:
                dif = float(itv[1]) - float(itv[0])
            except Exception:
                continue
            dur.append(dif)
        out[lab] = dur
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core merge
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MergeAnnotationsResult:
    organized_annotations_df: pd.DataFrame
    detected_song_segments_df: pd.DataFrame
    detected_merged_songs_df: pd.DataFrame
    annotations_appended_df: pd.DataFrame

def append_annotations_using_detection(
    *,
    annotations_df: pd.DataFrame,
    detected_song_segments: pd.DataFrame,
    detected_merged_songs: pd.DataFrame,
    segment_index_offset: int = 0,
) -> pd.DataFrame:
    """
    Build a "singles + merged" annotations dataframe using detection to decide which
    segments to append together. Times are shifted so the first segment in each group
    starts at 0 ms, and later segments are appended in timeline order.

    Joins on:
      detection filename  <-> annotations 'file_name'
      detection segment_index (per part)  <-> annotations 'Segment' (+ offset if needed)
    """
    ann = annotations_df.copy()

    # 'Segment' can be float/nullable; make a safe view for matching
    if "Segment" in ann.columns:
        ann["_Segment_int"] = pd.to_numeric(ann["Segment"], errors="coerce").astype("Int64")
    else:
        ann["_Segment_int"] = pd.Series([pd.NA] * len(ann), dtype="Int64")

    # Build lookup for per-segment onset_ms from detection (needed to compute shifts)
    seg_onset_map: Dict[Tuple[str, int], float] = {}
    if not detected_song_segments.empty:
        for _, r in detected_song_segments.iterrows():
            fname = str(r.get("filename"))
            try:
                sidx = int(r.get("segment_index"))
            except Exception:
                continue
            if pd.notna(r.get("onset_ms")):
                seg_onset_map[(fname, sidx)] = float(r["onset_ms"])

    rows_out: List[Dict[str, Any]] = []

    for _, det in detected_merged_songs.iterrows():
        fname = str(det.get("filename"))
        segs  = _ensure_list(det.get("segment_index"))
        if not segs:
            continue

        # Compute baseline (first onset) from detection
        # (if any onset missing, fall back to the minimum available)
        onsets = []
        for s in segs:
            key = (fname, s)
            if key in seg_onset_map:
                onsets.append(seg_onset_map[key])
        if onsets:
            baseline = min(onsets)
        else:
            baseline = 0.0  # if detection didn't carry onset_ms, append with no shifts

        # Accumulate combined annotations
        combined: Dict[str, List[List[float]]] = {}
        found_any = False
        found_rows: List[pd.Series] = []
        missing_segments: List[int] = []

        # Walk segments in order and append with shift
        for s in segs:
            ann_seg = s + int(segment_index_offset)
            # Match a single annotation row for this (file, segment)
            match = ann.loc[(ann["file_name"] == fname) & (ann["_Segment_int"] == ann_seg)]
            if match.empty:
                missing_segments.append(s)
                continue

            # If multiple rows (shouldn't typically happen), take the first by time
            mrow = match.sort_values("Recording DateTime", na_position="last").iloc[0]
            found_rows.append(mrow)

            # Compute shift from detection onset
            shift = 0.0
            key = (fname, s)
            if key in seg_onset_map:
                shift = float(seg_onset_map[key]) - float(baseline)

            # Pull the per-label intervals dict (already parsed by organizer)
            piece = mrow.get("syllable_onsets_offsets_ms_dict", {})
            _shift_and_extend(combined, piece, shift_ms=shift)
            found_any = True

        # Finalize combined intervals
        if found_any:
            _sort_intervals_inplace(combined)
            syllables_present = sorted(combined.keys())
            syllable_order = _compute_syllable_order(combined)
            durs = _durations_by_label(combined)
        else:
            combined = {}
            syllables_present = []
            syllable_order = []
            durs = {}

        # Build output record: copy key identity/timestamps from the first available row
        base = (found_rows[0] if found_rows else ann[(ann["file_name"] == fname)].head(1).squeeze())
        rec: Dict[str, Any] = {}

        rec["file_name"] = fname
        rec["Segment"]   = (segs[0] + int(segment_index_offset)) if len(segs) else None
        rec["was_merged"] = len(segs) >= 2
        rec["merged_n_parts"] = len(segs)
        rec["merged_segment_indices"] = segs
        rec["merged_segment_index_start"] = segs[0]
        rec["merged_segment_index_end"] = segs[-1]
        rec["missing_segments"] = missing_segments if missing_segments else None

        # Carry over useful context if available
        for c in ["Animal ID", "Recording DateTime", "Date", "Hour", "Minute", "Second", "File Stem"]:
            if isinstance(base, pd.Series) and c in base.index:
                rec[c] = base[c]
            else:
                rec[c] = None

        # Store merged annotation dicts + derived helpers
        rec["syllable_onsets_offsets_ms_dict"] = combined
        rec["syllables_present"] = syllables_present
        rec["syllable_order"] = syllable_order

        # Also add per-label duration columns like the organizer does
        for lab, arr in durs.items():
            rec[f"syllable_{lab}_durations"] = arr

        rows_out.append(rec)

    out_df = pd.DataFrame(rows_out)

    # Sort by file_name and first segment index for readability
    if not out_df.empty:
        out_df["_first_idx"] = out_df["merged_segment_indices"].apply(lambda L: L[0] if isinstance(L, list) and L else -1)
        out_df = out_df.sort_values(["file_name", "_first_idx"]).drop(columns=["_first_idx"]).reset_index(drop=True)

    return out_df


def build_decoded_with_split_labels(
    *,
    decoded_database_json: Union[str, Path],
    song_detection_json: Union[str, Path],
    only_song_present: bool = True,
    compute_durations: bool = True,
    add_recording_datetime: bool = True,
    songs_only: bool = True,
    flatten_spec_params: bool = True,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
) -> MergeAnnotationsResult:
    """
    Full pipeline: detection → merged groups, annotations → per-segment df,
    then append annotations across merged segments using detection timing.
    """
    # 1) Detection: both the raw per-segment and the combined "singles + merged"
    det = build_detected_and_merged_songs(
        song_detection_json,
        songs_only=songs_only,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )
    detected_song_segments = det["detected_song_segments"]
    detected_merged_songs  = det["detected_merged_songs"]

    # 2) Annotations organized per segment
    org = build_organized_segments_with_durations(
        decoded_database_json=decoded_database_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
        add_recording_datetime=add_recording_datetime,
    )
    annotations_df = org.organized_df.copy()

    # 3) Append annotations using detection groupings
    appended_df = append_annotations_using_detection(
        annotations_df=annotations_df,
        detected_song_segments=detected_song_segments,
        detected_merged_songs=detected_merged_songs,
        segment_index_offset=segment_index_offset,
    )

    return MergeAnnotationsResult(
        organized_annotations_df=annotations_df,
        detected_song_segments_df=detected_song_segments,
        detected_merged_songs_df=detected_merged_songs,
        annotations_appended_df=appended_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _prompt_if_missing(val: Optional[str], prompt: str) -> str:
    if val:
        return val
    return input(prompt).strip()

def main():
    p = argparse.ArgumentParser(description="Append annotation segments for split-up songs using detection timing.")
    p.add_argument("--song-detection", "-d", type=str, help="Path to song_detection JSON")
    p.add_argument("--annotations", "-a", type=str, help="Path to decoded annotations JSON")
    p.add_argument("--max-gap-ms", type=int, default=500, help="Max gap between segments to consider 'split up'")
    p.add_argument("--segment-index-offset", type=int, default=0,
                   help="If detection segment indices are 0-based but annotations are 1-based, use 1 here (or vice versa).")
    p.add_argument("--keep-all", action="store_true",
                   help="(No effect on saved outputs—kept for future options)")
    args = p.parse_args()

    song_detection_json = _prompt_if_missing(args.song_detection, "Path to song_detection JSON: ")
    annotations_json    = _prompt_if_missing(args.annotations,    "Path to decoded annotations JSON: ")

    res = build_decoded_with_split_labels(
        decoded_database_json=annotations_json,
        song_detection_json=song_detection_json,
        max_gap_between_song_segments=args.max_gap_ms,
        segment_index_offset=args.segment_index_offset,
        only_song_present=True,
        compute_durations=True,
        add_recording_datetime=True,
    )

    # Save outputs next to the annotations JSON
    outdir = Path(annotations_json).parent
    outdir.mkdir(parents=True, exist_ok=True)

    org_csv = outdir / "organized_annotations.csv"
    merged_csv = outdir / "decoded_with_split_labels.csv"

    res.organized_annotations_df.to_csv(org_csv, index=False)
    res.annotations_appended_df.to_csv(merged_csv, index=False)

    print("\n✅ Done.")
    print(f"  Organized per-segment annotations → {org_csv}")
    print(f"  Singles + merged (appended) annotations → {merged_csv}")
    print(f"  #detected segments: {len(res.detected_song_segments_df)}")
    print(f"  #detected rows (singles+merged): {len(res.detected_merged_songs_df)}")
    print(f"  #annotations (singles+merged) written: {len(res.annotations_appended_df)}")

if __name__ == "__main__":
    main()


"""
from pathlib import Path
import importlib

# Import & reload your script so recent edits are picked up
import merge_annotations_from_split_songs as mps
importlib.reload(mps)

# --- Edit these two paths ---
detect  = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
decoded = Path("/Volumes/my_own_ssd/2025_areax_lesion/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json")

# Run the pipeline:
# - merges split-up songs from detection
# - organizes annotations per segment
# - appends annotation segments across merged songs (time-shifted to align)
res = mps.build_decoded_with_split_labels(
    decoded_database_json=decoded,
    song_detection_json=detect,
    only_song_present=True,
    compute_durations=True,
    add_recording_datetime=True,
    songs_only=True,
    flatten_spec_params=True,
    max_gap_between_song_segments=500,   # adjust if needed
    segment_index_offset=0,              # use 1 if annotations are 1-based and detection is 0-based
)

# Handy handles
organized_df = res.organized_annotations_df          # per-segment annotations
detected_df  = res.detected_song_segments_df         # raw detected segments
merged_det   = res.detected_merged_songs_df          # detection after merging
appended_df  = res.annotations_appended_df           # singles + merged annotations (appended)

# Save next to the annotation JSON
outdir = decoded.parent
organized_df.to_csv(outdir / "organized_annotations.csv", index=False)
appended_df.to_csv(outdir / "decoded_with_split_labels.csv", index=False)

print("Saved:")
print(" -", outdir / "organized_annotations.csv")
print(" -", outdir / "decoded_with_split_labels.csv")

# Quick sanity checks
print("\nCounts:")
print("# detected segments:", len(detected_df))
print("# detected rows (singles+merged):", len(merged_det))
print("# annotations written (singles+merged):", len(appended_df))

# Peek the specific file you mentioned (should show merged indices like [1, 2])
fname = "R08_45765.26224434_4_18_7_17_4.wav"
subset = appended_df.loc[appended_df["file_name"] == fname,
                         ["file_name", "merged_segment_indices", "was_merged", "missing_segments"]]
print("\nCheck specific file:")
print(subset.head().to_string(index=False))



"""
