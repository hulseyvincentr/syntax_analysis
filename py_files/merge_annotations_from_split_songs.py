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
     - Match (filename, segment_index) to (file_name, Segment) (+ optional offset)
     - Shift each segment's annotation by detection onset so timelines align
     - Concatenate label intervals and sort by onset
  4) (Optional) Coalesce adjacent repeats of the same syllable if the gap between
     intervals is within a small threshold (e.g., < 10 ms).

Outputs:
  - organized_annotations.csv (per-segment)
  - decoded_with_split_labels.csv (singles + merged annotations)

CLI:
  python merge_annotations_from_split_songs.py \
      --song-detection /path/to/R08_RC6_Comp2_song_detection.json \
      --annotations   /path/to/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json \
      --max-gap-ms 500 \
      --segment-index-offset 0 \
      --merge-repeats \
      --repeat-gap-ms 10
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import argparse

import pandas as pd
import numpy as np

# ── Your modules (adjust names if your file names differ) ─────────────────────
from merge_potential_split_up_song import build_detected_and_merged_songs
from organized_decoded_serialTS_segments import (
    build_organized_segments_with_durations,
    parse_filename_with_excel_serial,
    excel_serial_to_timestamp,
)


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


def _shift_and_extend(
    acc: Dict[str, List[List[float]]],
    piece: Dict[str, List[List[float]]],
    shift_ms: float,
) -> None:
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


def _coalesce_adjacent_intervals_by_label(
    d: Dict[str, List[List[float]]],
    max_gap_ms: float = 10.0,
    inclusive: bool = False,
) -> None:
    """
    In-place: for each label, merge consecutive intervals if the silence/gap
    between them is small (e.g., < 10 ms). This is helpful when detection
    split a sustained/repeated syllable. Assumes intervals are already sorted.
    """
    if not isinstance(d, dict) or not d:
        return

    cmp = (lambda gap: gap <= max_gap_ms) if inclusive else (lambda gap: gap < max_gap_ms)

    for lab, ivals in list(d.items()):
        if not ivals:
            continue
        merged: List[List[float]] = []
        cur_start, cur_end = float(ivals[0][0]), float(ivals[0][1])

        for k in range(1, len(ivals)):
            s, e = float(ivals[k][0]), float(ivals[k][1])
            gap = s - cur_end
            if cmp(gap):
                # extend current interval
                cur_end = max(cur_end, e)
            else:
                merged.append([cur_start, cur_end])
                cur_start, cur_end = s, e
        merged.append([cur_start, cur_end])
        d[lab] = merged


def _durations_by_label(d: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for lab, ivals in (d or {}).items():
        dur: List[float] = []
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


def _canonical_file_key(name: Any) -> str:
    """
    Map any filename variant (old/new, with prefixes, underscores vs dots, etc.)
    to a canonical base key using parse_filename_with_excel_serial.

    We canonicalize to '<ANIMAL>_<EXCEL_SERIAL>' where EXCEL_SERIAL is the
    numeric Excel serial formatted with 8 decimal places. That way, e.g.:

      - 'USA5271_45342.27393766_2_20_7_36_33.wav'
      - 'USA5271_45342_27393766_2_20_7_36_33.wav'
      - '1__USA5271_45342_27393766_2_20_7_36_33_segment_0.wav'

    all become 'USA5271_45342.27393766'.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name)

    try:
        animal_id, serial_val, _, file_stem = parse_filename_with_excel_serial(s)
    except Exception:
        animal_id, serial_val, file_stem = None, None, None

    # Preferred: animal + numeric Excel serial, normalized to 8 decimal places
    if animal_id is not None and serial_val is not None:
        return f"{animal_id}_{float(serial_val):.8f}"

    # Fallback: if parse failed but file_stem exists, use that
    if file_stem:
        return str(file_stem)

    # Last resort: drop extension from raw string, or return raw
    if "." in s:
        base, ext = s.rsplit(".", 1)
        if ext.lower() in {"wav", "flac", "mp3"}:
            return base
    return s


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
    merge_repeated_syllables: bool = False,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
) -> pd.DataFrame:
    """
    Build a "singles + merged" annotations dataframe using detection to decide which
    segments to append together. Times are shifted so the first segment in each group
    starts at 0 ms, and later segments are appended in timeline order.

    Joins on a canonical file key derived from the detection 'filename' and the
    annotations 'file_name', plus the segment index (with optional offset).

    If `merge_repeated_syllables` is True, coalesces adjacent repeats of the same label
    when the between-interval gap is within the threshold (repeat_gap_ms).
    """
    ann = annotations_df.copy()

    # Ensure we have a file_name column (robustness for future organizer tweaks)
    if "file_name" not in ann.columns:
        for alt in ("filename", "File Name", "file", "wav_file", "wav_path", "path"):
            if alt in ann.columns:
                ann["file_name"] = ann[alt]
                break
    if "file_name" not in ann.columns:
        raise KeyError(
            f"annotations_df must contain a 'file_name' column (or a recognized "
            f"alternative); got columns: {list(ann.columns)}"
        )

    # Canonical filename key for robust matching across naming schemes
    ann["_canonical_file"] = ann["file_name"].apply(_canonical_file_key)

    # 'Segment' can be float/nullable; make a safe view for matching
    if "Segment" in ann.columns:
        ann["_Segment_int"] = pd.to_numeric(ann["Segment"], errors="coerce").astype("Int64")
    else:
        ann["_Segment_int"] = pd.Series([pd.NA] * len(ann), dtype="Int64")

    # Build lookup for per-segment onset_ms from detection (needed to compute shifts)
    seg_onset_map: Dict[Tuple[str, int], float] = {}
    if not detected_song_segments.empty:
        for _, r in detected_song_segments.iterrows():
            fname_raw = str(r.get("filename"))
            canon = _canonical_file_key(fname_raw)
            try:
                sidx = int(r.get("segment_index"))
            except Exception:
                continue
            if pd.notna(r.get("onset_ms")):
                seg_onset_map[(canon, sidx)] = float(r["onset_ms"])

    rows_out: List[Dict[str, Any]] = []

    for _, det in detected_merged_songs.iterrows():
        fname_raw = str(det.get("filename"))
        fname_canon = _canonical_file_key(fname_raw)
        segs = _ensure_list(det.get("segment_index"))
        if not segs:
            continue

        # Compute baseline (first onset) from detection using canonical key
        onsets: List[float] = []
        for s in segs:
            key = (fname_canon, s)
            if key in seg_onset_map:
                onsets.append(seg_onset_map[key])
        baseline = min(onsets) if onsets else 0.0

        # Accumulate combined annotations
        combined: Dict[str, List[List[float]]] = {}
        found_any = False
        found_rows: List[pd.Series] = []
        missing_segments: List[int] = []

        # Walk segments in order and append with shift
        for s in segs:
            ann_seg = s + int(segment_index_offset)
            match = ann.loc[
                (ann["_canonical_file"] == fname_canon)
                & (ann["_Segment_int"] == ann_seg)
            ]
            if match.empty:
                missing_segments.append(s)
                continue

            mrow = match.sort_values("Recording DateTime", na_position="last").iloc[0]
            found_rows.append(mrow)

            # shift using detection onset difference (canonical key)
            shift = 0.0
            key = (fname_canon, s)
            if key in seg_onset_map:
                shift = float(seg_onset_map[key]) - float(baseline)

            piece = mrow.get("syllable_onsets_offsets_ms_dict", {})
            _shift_and_extend(combined, piece, shift_ms=shift)
            found_any = True

        # Finalize combined intervals
        if found_any:
            _sort_intervals_inplace(combined)

            # Optional: coalesce repeated same-label intervals if tiny gap
            if merge_repeated_syllables:
                _coalesce_adjacent_intervals_by_label(
                    combined,
                    max_gap_ms=float(repeat_gap_ms),
                    inclusive=bool(repeat_gap_inclusive),
                )

            # Rebuild helpers after potential coalescing
            syllables_present = sorted(combined.keys())
            syllable_order = _compute_syllable_order(combined)
            durs = _durations_by_label(combined)
        else:
            combined = {}
            syllables_present = []
            syllable_order = []
            durs = {}

        # Build output record: copy key identity/timestamps from the first available row
        if found_rows:
            base = found_rows[0]
        else:
            # Fallback: try matching by canonical file only
            base = ann[ann["_canonical_file"] == fname_canon].head(1).squeeze()

        rec: Dict[str, Any] = {}
        # Use the original detection filename as the outward-facing name
        rec["file_name"] = fname_raw
        rec["Segment"] = (segs[0] + int(segment_index_offset)) if len(segs) else None
        rec["was_merged"] = len(segs) >= 2
        rec["merged_n_parts"] = len(segs)
        rec["merged_segment_indices"] = segs
        rec["merged_segment_index_start"] = segs[0]
        rec["merged_segment_index_end"] = segs[-1]
        rec["missing_segments"] = missing_segments if missing_segments else None

        for c in ["Animal ID", "Recording DateTime", "Date", "Hour", "Minute", "Second", "File Stem"]:
            if isinstance(base, pd.Series) and c in base.index:
                rec[c] = base[c]
            else:
                rec[c] = None

        rec["syllable_onsets_offsets_ms_dict"] = combined
        rec["syllables_present"] = syllables_present
        rec["syllable_order"] = syllable_order

        # Per-label duration columns like the organizer
        for lab, arr in durs.items():
            rec[f"syllable_{lab}_durations"] = arr

        rows_out.append(rec)

    out_df = pd.DataFrame(rows_out)

    # Sort by file_name and first segment index for readability
    if not out_df.empty:
        out_df["_first_idx"] = out_df["merged_segment_indices"].apply(
            lambda L: L[0] if isinstance(L, list) and L else -1
        )
        out_df = (
            out_df.sort_values(["file_name", "_first_idx"])
                  .drop(columns=["_first_idx"])
                  .reset_index(drop=True)
        )

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
    merge_repeated_syllables: bool = False,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
) -> MergeAnnotationsResult:
    """
    Full pipeline: detection → merged groups, annotations → per-segment df,
    then append annotations across merged segments using detection timing.
    Optionally coalesce adjacent repeats of the same syllable if the between-interval
    gap is tiny (e.g., < 10 ms).
    """
    # 1) Detection: both the raw per-segment and the combined "singles + merged"
    det = build_detected_and_merged_songs(
        song_detection_json,
        songs_only=songs_only,
        flatten_spec_params=flatten_spec_params,
        max_gap_between_song_segments=max_gap_between_song_segments,
    )
    detected_song_segments = det["detected_song_segments"]
    detected_merged_songs = det["detected_merged_songs"]

    # 2) Annotations organized per segment
    org = build_organized_segments_with_durations(
        decoded_database_json=decoded_database_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
        add_recording_datetime=add_recording_datetime,
    )
    annotations_df = org.organized_df.copy()

    # 3) Append annotations using detection groupings (+ optional coalescing)
    appended_df = append_annotations_using_detection(
        annotations_df=annotations_df,
        detected_song_segments=detected_song_segments,
        detected_merged_songs=detected_merged_songs,
        segment_index_offset=segment_index_offset,
        merge_repeated_syllables=merge_repeated_syllables,
        repeat_gap_ms=repeat_gap_ms,
        repeat_gap_inclusive=repeat_gap_inclusive,
    )

    # ── Safety: ensure Recording DateTime exists and is datetime ─────────────
    if "Recording DateTime" in appended_df.columns:
        # If everything is missing, try to reconstruct from file_name via Excel serial
        if appended_df["Recording DateTime"].isna().all():
            for idx, fname in appended_df["file_name"].items():
                _, serial_val, _, _ = parse_filename_with_excel_serial(str(fname))
                ts = excel_serial_to_timestamp(serial_val) if serial_val is not None else None
                if ts is not None:
                    appended_df.at[idx, "Recording DateTime"] = ts
        appended_df["Recording DateTime"] = pd.to_datetime(
            appended_df["Recording DateTime"], errors="coerce"
        )
    else:
        # Create from scratch if column is missing entirely
        ts_list: List[Optional[pd.Timestamp]] = []
        for fname in appended_df["file_name"]:
            _, serial_val, _, _ = parse_filename_with_excel_serial(str(fname))
            ts = excel_serial_to_timestamp(serial_val) if serial_val is not None else None
            ts_list.append(ts)
        appended_df["Recording DateTime"] = pd.to_datetime(ts_list, errors="coerce")

    # ── Rebuild Date / Hour / Minute / Second from Recording DateTime ───────
    if "Recording DateTime" in appended_df.columns:
        dt = appended_df["Recording DateTime"]
        appended_df["Date"] = dt.dt.normalize()
        appended_df["Hour"] = dt.dt.hour.astype("Int64").astype(str).str.zfill(2)
        appended_df["Minute"] = dt.dt.minute.astype("Int64").astype(str).str.zfill(2)
        appended_df["Second"] = dt.dt.second.astype("Int64").astype(str).str.zfill(2)

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
    p = argparse.ArgumentParser(
        description="Append annotation segments for split-up songs using detection timing."
    )
    p.add_argument("--song-detection", "-d", type=str, help="Path to song_detection JSON")
    p.add_argument("--annotations", "-a", type=str, help="Path to decoded annotations JSON")
    p.add_argument(
        "--max-gap-ms",
        type=int,
        default=500,
        help="Max gap between segments to consider 'split up'",
    )
    p.add_argument(
        "--segment-index-offset",
        type=int,
        default=0,
        help=(
            "If detection segment indices are 0-based but annotations are 1-based, "
            "use 1 here (or vice versa)."
        ),
    )
    # Options for repeat coalescing
    p.add_argument(
        "--merge-repeats",
        action="store_true",
        help="If set, coalesce adjacent repeats of the same syllable when tiny gaps exist.",
    )
    p.add_argument(
        "--repeat-gap-ms",
        type=float,
        default=10.0,
        help="Max silence between same-label intervals to coalesce (default 10.0 ms).",
    )
    p.add_argument(
        "--repeat-gap-inclusive",
        action="store_true",
        help="Use <= repeat-gap instead of < repeat-gap.",
    )
    args = p.parse_args()

    song_detection_json = _prompt_if_missing(args.song_detection, "Path to song_detection JSON: ")
    annotations_json = _prompt_if_missing(args.annotations, "Path to decoded annotations JSON: ")

    res = build_decoded_with_split_labels(
        decoded_database_json=annotations_json,
        song_detection_json=song_detection_json,
        max_gap_between_song_segments=args.max_gap_ms,
        segment_index_offset=args.segment_index_offset,
        only_song_present=True,
        compute_durations=True,
        add_recording_datetime=True,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
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
