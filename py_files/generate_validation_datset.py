# -*- coding: utf-8 -*-
# generate_validation_dataset.py
"""
Generate validation JSON for USA1234 with AM and PM blocks and segment-aware filenames.

What this creates (per day: 2000-01-01 and 2000-01-02):
- AM block: timestamps at 00:00..00:19 (20 stems), each with segments _0 and _1 → 40 segments
- PM block: timestamps at 12:00..12:19 (20 stems), each with segments _0 and _1 → 40 segments
Total per day = 80 segments; across two days = 160 segments.

Filename format (no extension):
  USA1234_<token>_<M>_<D>_<H>_<MIN>_<S>_<SEG>

`creation_date` is ISO8601 and identical for both segments of the same stem.

Requested pattern splits (applied across *segments* in each block):
- 2000-01-01 AM: all 0→1
- 2000-01-01 PM: half 0→1, half 0→2
- 2000-01-02 AM: equal thirds 0→1, 0→2, 0→3
- 2000-01-02 PM: all 0→2

Syllable timing:
- Each syllable = 200 ms
- Gap between syllables = 10 ms
- Timebins = 10 ms (inclusive bin spans)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# ── CONFIG ───────────────────────────────────────────────────────────────────
ANIMAL_ID = "USA1234"
YEAR = 2000
DAYS = [1, 2]                 # Jan 1 and Jan 2, 2000
STEMS_PER_BLOCK = 20          # ⇐ 20 stems → 40 segments per block (2 segments per stem)
SEGMENTS_PER_STEM = 2         # emit _0 and _1 per stem

SYL_MS = 200.0
GAP_MS = 10.0
BIN_MS = 10.0

# Output directory/files
OUTDIR = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CREATION_OUT = OUTDIR / "USA1234_metadata.json"
DECODED_OUT = OUTDIR / "USA1234_decoded_database.json"

# ── HELPERS ──────────────────────────────────────────────────────────────────
def _ms_to_bins_span(on_ms: float, off_ms: float) -> List[int]:
    start_bin = int(round(on_ms / BIN_MS))
    end_bin   = int(round(off_ms / BIN_MS))
    return [start_bin, end_bin]

def _mk_syllables_two(a: str, b: str):
    a_on, a_off = 0.0, SYL_MS
    b_on, b_off = a_off + GAP_MS, a_off + GAP_MS + SYL_MS
    ms = {a: [[a_on, a_off]], b: [[b_on, b_off]]}
    tb = {a: [_ms_to_bins_span(a_on, a_off)], b: [_ms_to_bins_span(b_on, b_off)]}
    return ms, tb

def _file_stem_token(stem_index: int) -> str:
    # Deterministic token; same for _0 and _1 of the same stem
    return f"{(45300 + stem_index):05d}.{(41751298 + 87123 * stem_index):08d}"[-14:]

def _mk_filename(dt: datetime, stem_index: int, seg: int) -> str:
    parts = [
        ANIMAL_ID,
        _file_stem_token(stem_index),
        str(dt.month),
        str(dt.day),
        str(dt.hour),
        str(dt.minute),
        str(dt.second),
        str(seg),
    ]
    return "_".join(parts)

def _patterns_for_block(total_segments: int, day: int, block: str) -> List[List[str]]:
    """
    Build a list of length `total_segments` containing label pairs [from, to]
    that realize the requested distribution per day/block.
    """
    if day == 1 and block == "AM":
        return [["0", "1"]] * total_segments
    if day == 1 and block == "PM":
        half = total_segments // 2
        rem  = total_segments - 2 * half
        patt = [["0", "1"]] * half + [["0", "2"]] * half
        if rem > 0:
            patt.append(["0", "1"])
        return patt
    if day == 2 and block == "AM":
        third = total_segments // 3
        rem   = total_segments - 3 * third
        patt  = [["0", "1"]] * third + [["0", "2"]] * third + [["0", "3"]] * third
        order = [["0", "1"], ["0", "2"], ["0", "3"]]
        for i in range(rem):
            patt.append(order[i % 3])
        return patt
    if day == 2 and block == "PM":
        return [["0", "2"]] * total_segments
    # Fallback (should not occur)
    return [["0", "1"]] * total_segments

# ── DATA BUILDERS ────────────────────────────────────────────────────────────
@dataclass
class DecodedRow:
    file_name: str
    creation_date: str
    song_present: bool
    syllable_onsets_offsets_ms: Dict[str, List[List[float]]]
    syllable_onsets_offsets_timebins: Dict[str, List[List[int]]]

def _emit_block(results: List[Dict], *, day_base: datetime, hour: int, day: int, block_name: str, stem_start: int) -> int:
    """
    Emit one block (AM or PM) of STEMS_PER_BLOCK stems, each with SEGMENTS_PER_STEM segments.
    Returns the next stem index after emitting.
    """
    start_dt = day_base.replace(hour=hour, minute=0, second=0, microsecond=0)

    total_segments = STEMS_PER_BLOCK * SEGMENTS_PER_STEM
    patterns = _patterns_for_block(total_segments, day=day, block=block_name)
    p = 0  # pattern cursor

    stem_index = stem_start
    for minute_idx in range(STEMS_PER_BLOCK):
        rec_dt = start_dt + timedelta(minutes=minute_idx)
        creation_date = rec_dt.isoformat()

        # Emit SEGMENTS_PER_STEM segments for this stem (same token & creation_date)
        for seg in range(SEGMENTS_PER_STEM):
            frm, to = patterns[p]
            p += 1
            ms, tb = _mk_syllables_two(frm, to)
            results.append(
                {
                    "file_name": _mk_filename(rec_dt, stem_index, seg=seg),
                    "creation_date": creation_date,
                    "song_present": True,
                    "syllable_onsets_offsets_ms": ms,
                    "syllable_onsets_offsets_timebins": tb,
                }
            )
        stem_index += 1

    return stem_index

def build_decoded_results() -> List[Dict]:
    results: List[Dict] = []
    stem_index = 0  # token index advances per stem (shared by _0 and _1)

    for day in DAYS:
        day_base = datetime(YEAR, 1, day, 0, 0, 0)

        # AM block (00:00..00:19) – first-half-of-day selection
        stem_index = _emit_block(
            results, day_base=day_base, hour=0, day=day, block_name="AM", stem_start=stem_index
        )

        # PM block (12:00..12:19) – last-half-of-day selection
        stem_index = _emit_block(
            results, day_base=day_base, hour=12, day=day, block_name="PM", stem_start=stem_index
        )

    return results

def build_creation_metadata() -> Dict:
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
    return {
        "metadata": {
            "classifier_path": "experiments/TweetyBERT_Pretrain_LLB_AreaX_FallSong",
            "spec_dst_folder": "imgs/decoder_specs_inference_test",
            "output_path": str(DECODED_OUT),
            "visualize": False,
            "dump_interval": 500,
            "apply_post_processing": True,
        },
        "results": results,
    }

# ── WRITE FILES ──────────────────────────────────────────────────────────────
def main():
    results = build_decoded_results()
    CREATION_OUT.write_text(json.dumps(build_creation_metadata(), indent=2))
    print(f"Wrote creation metadata → {CREATION_OUT.resolve()}")

    DECODED_OUT.write_text(json.dumps(build_decoded_payload(results), indent=2))
    print(f"Wrote decoded database → {DECODED_OUT.resolve()}")

    # Handy globals for interactive sessions
    globals()["decoded_results"] = results

if __name__ == "__main__":
    main()
