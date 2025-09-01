# -*- coding: utf-8 -*-
# generate_validation_dataset.py
"""
Create two validation JSON files for USA1234, matching your prior schemas:

1) USA1234_metadata.json  (creation metadata format)
2) USA1234_decoded_database.json  (decoded database format)

AM/PM rules (10 songs/day = 5 AM + 5 PM):
- Jan 1 & 2:
    AM: syllable '0' only
    PM: '0' -> '1' only
- Jan 3 & 4:
    AM: half '0'->'1', half '0'->'2'
    PM: syllable '0' only

Syllable timing:
- Each syllable: 200 ms
- Gap between consecutive syllables: 10 ms
- Timebins = 10 ms; e.g., 200 ms spans 20 bins.

Notes:
- Uses year 2000 (pandas-safe) to avoid out-of-bounds issues later when
  converting to pandas Timestamps in organizer-style code.
- File names include a trailing segment index ("_0.wav") to match your parser.
- The "random-looking" middle token in file_name is just a deterministic counter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ANIMAL_ID = "USA1234"
YEAR = 2000  # pandas-safe synthetic year
DAYS = [1, 2, 3, 4]
SONGS_PER_DAY = 10
SONGS_AM = 5
SONGS_PM = SONGS_PER_DAY - SONGS_AM

SYL_MS = 200.0
GAP_MS = 10.0
BIN_MS = 10.0  # timebin resolution (10 ms)

# output files (written next to this script)
CREATION_OUT = Path("USA1234_metadata.json")
DECODED_OUT = Path("USA1234_decoded_database.json")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _ms_to_bins_span(on_ms: float, off_ms: float) -> List[int]:
    """Convert ms span to inclusive [start_bin, end_bin] indices with 10 ms bins."""
    start_bin = int(round(on_ms / BIN_MS))
    end_bin = int(round(off_ms / BIN_MS))
    return [start_bin, end_bin]


def _mk_syllables_one(label: str) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[int]]]]:
    """One syllable of SYL_MS at 0 ms; returns (ms_dict, timebins_dict)."""
    on = 0.0
    off = on + SYL_MS
    ms = {label: [[on, off]]}
    tb = {label: [_ms_to_bins_span(on, off)]}
    return ms, tb


def _mk_syllables_two(a: str, b: str) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[int]]]]:
    """Two syllables a then b with GAP_MS between; returns (ms_dict, timebins_dict)."""
    a_on = 0.0
    a_off = a_on + SYL_MS
    b_on = a_off + GAP_MS
    b_off = b_on + SYL_MS
    ms = {a: [[a_on, a_off]], b: [[b_on, b_off]]}
    tb = {a: [_ms_to_bins_span(a_on, a_off)], b: [_ms_to_bins_span(b_on, b_off)]}
    return ms, tb


def _am_patterns(day: int) -> List[List[str]]:
    """Return list of label sequences for AM (length = SONGS_AM)."""
    if day in (1, 2):
        return [["0"]] * SONGS_AM
    # day 3 & 4: half 0->1, half 0->2
    half = SONGS_AM // 2
    rem = SONGS_AM - half
    return [["0", "1"]] * half + [["0", "2"]] * rem


def _pm_patterns(day: int) -> List[List[str]]:
    """Return list of label sequences for PM (length = SONGS_PM)."""
    if day in (1, 2):
        return [["0", "1"]] * SONGS_PM
    # day 3 & 4: only '0'
    return [["0"]] * SONGS_PM


def _file_stem_token(i: int) -> str:
    """
    Deterministic “random-looking” token for the middle filename field.
    Format XXXXX.YYYYYYYY (digits only) to look like prior examples.
    """
    return f"{(45300 + i):05d}.{(41751298 + 87123 * i):08d}"[-14:]  # keep width-ish similar


def _mk_filename(dt: datetime, i: int, segment: int = 0, with_ext: bool = True) -> str:
    """
    Build file_name as in your examples:
      USA1234_<token>_<M>_<D>_<H>_<M>_<S>_<SEG>[.wav]
    """
    token = _file_stem_token(i)
    parts = [
        ANIMAL_ID,
        token,
        str(dt.month),
        str(dt.day),
        str(dt.hour),
        str(dt.minute),
        str(dt.second),
        str(segment),
    ]
    stem = "_".join(parts)
    return f"{stem}.wav" if with_ext else stem


# ── DATA BUILDERS ──────────────────────────────────────────────────────────────
@dataclass
class DecodedRow:
    file_name: str
    song_present: bool
    syllable_onsets_offsets_ms: Dict[str, List[List[float]]]
    syllable_onsets_offsets_timebins: Dict[str, List[List[int]]]


def build_decoded_results() -> List[Dict]:
    """
    Build results list for the decoded_database.json.
    10 songs/day → 5 AM (08:00..08:04), 5 PM (14:00..14:04) per day.
    """
    results: List[Dict] = []
    counter = 0

    for day in DAYS:
        day_base = datetime(YEAR, 1, day)

        # AM block
        am_start = day_base.replace(hour=8, minute=0, second=0, microsecond=0)
        for idx, patt in enumerate(_am_patterns(day)):
            rec_dt = am_start + timedelta(minutes=idx)
            if len(patt) == 1:
                ms, tb = _mk_syllables_one(patt[0])
            else:
                ms, tb = _mk_syllables_two(patt[0], patt[1])

            results.append(
                {
                    "file_name": _mk_filename(rec_dt, counter, segment=0, with_ext=True),
                    "song_present": True,
                    "syllable_onsets_offsets_ms": ms,
                    "syllable_onsets_offsets_timebins": tb,
                }
            )
            counter += 1

        # PM block
        pm_start = day_base.replace(hour=14, minute=0, second=0, microsecond=0)
        for idx, patt in enumerate(_pm_patterns(day)):
            rec_dt = pm_start + timedelta(minutes=idx)
            if len(patt) == 1:
                ms, tb = _mk_syllables_one(patt[0])
            else:
                ms, tb = _mk_syllables_two(patt[0], patt[1])

            results.append(
                {
                    "file_name": _mk_filename(rec_dt, counter, segment=0, with_ext=True),
                    "song_present": True,
                    "syllable_onsets_offsets_ms": ms,
                    "syllable_onsets_offsets_timebins": tb,
                }
            )
            counter += 1

    return results


def build_creation_metadata() -> Dict:
    """
    Build creation/metadata JSON:
    - treatment_date/type are synthetic placeholders
    - subdirectories: one per day with 'YYYY-MM-DD'
    """
    meta = {
        "treatment_date": f"{YEAR}-01-15",
        "treatment_type": "Mock treatment for testing",
        "subdirectories": {},
    }
    for i, day in enumerate(DAYS):
        iso = f"{YEAR}-01-{day:02d}"
        meta["subdirectories"][str(i)] = {
            "subdirectory_creation_date": iso,
            "unique_file_creation_dates": [iso],
        }
    return meta


def build_decoded_payload(results: List[Dict]) -> Dict:
    """
    Top-level structure for decoded file, matching prior example:
      { "metadata": {...}, "results": [...] }
    """
    payload = {
        "metadata": {
            "classifier_path": "experiments/TweetyBERT_Pretrain_LLB_AreaX_FallSong",
            "spec_dst_folder": "imgs/decoder_specs_inference_test",
            "output_path": "files/USA1234_decoded_database.json",
            "visualize": False,
            "dump_interval": 500,
            "apply_post_processing": True,
        },
        "results": results,
    }
    return payload


# ── WRITE FILES ────────────────────────────────────────────────────────────────
def main():
    # Build decoded "results"
    results = build_decoded_results()

    # Build and write creation metadata JSON
    creation_meta = build_creation_metadata()
    CREATION_OUT.write_text(json.dumps(creation_meta, indent=2))
    print(f"Wrote creation metadata → {CREATION_OUT.resolve()}")

    # Build and write decoded database JSON
    decoded_payload = build_decoded_payload(results)
    DECODED_OUT.write_text(json.dumps(decoded_payload, indent=2))
    print(f"Wrote decoded database → {DECODED_OUT.resolve()}")

    # Nice for interactive sessions (e.g., Spyder Variable Explorer)
    globals()["decoded_results"] = results
    globals()["creation_metadata"] = creation_meta


if __name__ == "__main__":
    main()
