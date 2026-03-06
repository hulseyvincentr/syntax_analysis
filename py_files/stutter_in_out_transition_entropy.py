#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stutter_in_out_transition_entropy.py

Compute incoming and outgoing transition entropy (bits) for each syllable
*conditioned on that syllable being stuttered* (i.e., repeated consecutively).

What you get
------------
For each day (optional) and syllable:
  - n_stutter_events
  - incoming_entropy_bits  : entropy over labels that precede stutter-runs of this syllable
  - outgoing_entropy_bits  : entropy over labels that follow stutter-runs of this syllable
  - (optional) baseline for non-stutter occurrences of that syllable

Stutter definition
------------------
A stutter event is a contiguous run of the same token label with length >= stutter_min_run.
Example (token sequence):  1, 2, 2, 2, 3
- syllable "2" has one stutter event (run length 3), incoming=1, outgoing=3

Token sequences vs time-bin labels
----------------------------------
If your sequence column is time-bin labels (e.g., one label per 1 ms / 10 ms / etc),
a sustained syllable will appear as many repeated bins and would falsely look like stutter.

To handle that:
  --interpret-as-timebins
will convert time-bins -> token sequence via collapsing consecutive duplicates,
optionally dropping ignore labels (e.g. -1), then detecting stutter on the token sequence.

Data input options
------------------
A) Auto-merge from:
   --decoded-json  (e.g. USA5288_decoded_database.json)
   --song-detection-json (e.g. USA5288_song_detection.json)
   using merge_annotations_from_split_songs.build_decoded_with_split_labels

B) Use premerged table:
   --premerged (CSV/TSV/JSON/Parquet supported by pandas where applicable)

Filtering post-treatment
------------------------
Provide either:
  --treatment-date YYYY-MM-DD
or
  --metadata-xlsx + columns to lookup treatment date for your bird

Typical usage (USA5288)
------------------------
python stutter_in_out_transition_entropy.py \
  --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --bird "USA5288" \
  --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "metadata" \
  --animal-id-col "Animal ID" \
  --treatment-date-col "Treatment date" \
  --scope post \
  --seq-col labels \
  --interpret-as-timebins \
  --ignore-labels -1 \
  --stutter-min-run 2 \
  --compute-baseline \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/entropy_stutter_outputs" \
  --out-prefix "USA5288_stutter_inout"

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Parsing / utilities
# -----------------------------
def safe_parse_listlike(x: Any) -> Optional[List[Any]]:
    """Accept list/tuple/np.array or parse a string like '["1","2"]' or "['1','2']"."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = json.loads(s)
            return list(v) if isinstance(v, (list, tuple)) else None
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple)) else None
        except Exception:
            return None
    return None


def entropy_bits_from_counts(counts: Counter) -> float:
    """Shannon entropy in bits from a Counter."""
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    H = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        H -= p * math.log(p, 2)
    return H


def collapse_consecutive(seq: Sequence[Any]) -> List[Any]:
    """Collapse AAAABCC -> ABC."""
    out: List[Any] = []
    prev = object()
    for x in seq:
        if x != prev:
            out.append(x)
            prev = x
    return out


def parse_date_like(x: Any) -> Optional[pd.Timestamp]:
    """Parse YYYY-MM-DD, MM/DD/YYYY, etc. Returns pandas Timestamp (normalized)."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


def choose_datetime_series(df: pd.DataFrame) -> pd.Series:
    """
    Pick a datetime series from common columns, fallback to combining Date + Time when available.
    Returns pd.Series of dtype datetime64[ns] with NaT where unavailable.
    """
    candidates = [
        "Recording DateTime",
        "recording_datetime",
        "recordingDateTime",
        "recording_dt",
        "creation_date",
        "Creation Date",
        "datetime",
        "_dt",
    ]
    for c in candidates:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s

    # Try date-only columns
    date_candidates = ["recording_date", "Recording Date", "Date", "date"]
    time_candidates = ["recording_time", "Recording Time", "Time", "time"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if date_col and time_col:
        s = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        return s

    if date_col:
        s = pd.to_datetime(df[date_col], errors="coerce")
        return s

    # Give up
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def coerce_ignore_set(values: Optional[Iterable[str]]) -> Set[str]:
    """
    Normalize ignore labels to a set of strings.
    (We compare labels via str(label), so "-1" matches -1, "-1", etc.)
    """
    if not values:
        return set()
    return {str(v) for v in values}


def auto_pick_seq_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    """Pick sequence column: requested if present, else try common candidates."""
    if requested and requested in df.columns:
        return requested
    candidates = [
        "mapped_syllable_order",
        "syllable_order",
        "phrase_identities",
        "labels",              # common in merged timebin tables
        "time_bin_syllable_labels",
        "timebin_labels",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a sequence column. "
        f"Requested={requested!r}. Available columns include: {list(df.columns)[:30]} ..."
    )


def infer_interpret_as_timebins(seq_col: str, explicit: Optional[bool]) -> bool:
    """If user didn’t specify, infer: labels/timebin columns are likely time-bin sequences."""
    if explicit is not None:
        return explicit
    name = seq_col.lower()
    if "timebin" in name or "time_bin" in name:
        return True
    if name in {"labels"} or name.endswith("_labels"):
        return True
    return False


# -----------------------------
# Stutter: run detection
# -----------------------------
@dataclass(frozen=True)
class Run:
    label: Any
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_runs(seq: Sequence[Any]) -> List[Run]:
    """Return all contiguous runs in seq (including length 1)."""
    runs: List[Run] = []
    n = len(seq)
    i = 0
    while i < n:
        lab = seq[i]
        j = i
        while j + 1 < n and seq[j + 1] == lab:
            j += 1
        runs.append(Run(label=lab, start=i, end=j))
        i = j + 1
    return runs


# -----------------------------
# Core computation
# -----------------------------
def compute_stutter_in_out_entropy(
    df: pd.DataFrame,
    *,
    seq_col: str,
    dt_col: str = "_dt",
    group_by_day: bool = True,
    stutter_min_run: int = 2,
    mode: str = "events",  # "events" or "positions"
    ignore_labels: Optional[Iterable[str]] = None,
    interpret_as_timebins: bool = False,
    drop_ignored_after_tokenize: bool = True,
    compute_baseline_nonstutter: bool = False,
) -> pd.DataFrame:
    """
    Compute in/out transition entropy conditioned on stutter.

    mode="events" (recommended):
      - For each stutter run of label X (length>=threshold), record:
            incoming: label before run
            outgoing: label after run

    mode="positions":
      - For each position i inside a stutter run, record prev=seq[i-1], next=seq[i+1]
        (includes X->X transitions inside run, often dominates).

    interpret_as_timebins:
      - Convert per-bin labels -> token sequence by collapsing consecutive duplicates.
      - Optionally drop ignored labels after tokenization (recommended for -1/silence).
    """
    if mode not in {"events", "positions"}:
        raise ValueError("mode must be 'events' or 'positions'")

    ignore = coerce_ignore_set(ignore_labels)

    # key -> syllable -> Counter(prev/next)
    incoming_counts: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    outgoing_counts: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_stutter: Dict[Any, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # baseline
    incoming_counts_base: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    outgoing_counts_base: Dict[Any, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_base: Dict[Any, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def key_for_row(row: pd.Series) -> Any:
        if not group_by_day:
            return "__ALL__"
        dt = row.get(dt_col, pd.NaT)
        if pd.isna(dt):
            return "__NO_DATE__"
        return pd.Timestamp(dt).date().isoformat()

    for _, row in df.iterrows():
        raw = row.get(seq_col, None)
        seq = safe_parse_listlike(raw)
        if not seq:
            continue

        # Normalize to strings for stable comparisons / outputs
        seq = [str(x) for x in seq]

        if interpret_as_timebins:
            seq = collapse_consecutive(seq)
            if drop_ignored_after_tokenize and ignore:
                seq = [x for x in seq if x not in ignore]

        if not seq:
            continue

        # build stutter mask for baseline selection
        runs = find_runs(seq)
        stutter_mask = np.zeros(len(seq), dtype=bool)
        for r in runs:
            if r.label in ignore:
                continue
            if r.length >= stutter_min_run:
                stutter_mask[r.start : r.end + 1] = True

        gkey = key_for_row(row)

        if mode == "events":
            for r in runs:
                if r.label in ignore or r.length < stutter_min_run:
                    continue

                prev_lab = seq[r.start - 1] if r.start - 1 >= 0 else None
                next_lab = seq[r.end + 1] if r.end + 1 < len(seq) else None

                if prev_lab is not None and prev_lab not in ignore:
                    incoming_counts[gkey][r.label][prev_lab] += 1
                if next_lab is not None and next_lab not in ignore:
                    outgoing_counts[gkey][r.label][next_lab] += 1

                n_stutter[gkey][r.label] += 1

            if compute_baseline_nonstutter:
                for i, lab in enumerate(seq):
                    if lab in ignore or stutter_mask[i]:
                        continue
                    prev_lab = seq[i - 1] if i - 1 >= 0 else None
                    next_lab = seq[i + 1] if i + 1 < len(seq) else None
                    if prev_lab is not None and prev_lab not in ignore:
                        incoming_counts_base[gkey][lab][prev_lab] += 1
                    if next_lab is not None and next_lab not in ignore:
                        outgoing_counts_base[gkey][lab][next_lab] += 1
                    n_base[gkey][lab] += 1

        else:  # positions
            for i, lab in enumerate(seq):
                if lab in ignore or not stutter_mask[i]:
                    continue
                prev_lab = seq[i - 1] if i - 1 >= 0 else None
                next_lab = seq[i + 1] if i + 1 < len(seq) else None
                if prev_lab is not None and prev_lab not in ignore:
                    incoming_counts[gkey][lab][prev_lab] += 1
                if next_lab is not None and next_lab not in ignore:
                    outgoing_counts[gkey][lab][next_lab] += 1
                n_stutter[gkey][lab] += 1

            if compute_baseline_nonstutter:
                for i, lab in enumerate(seq):
                    if lab in ignore or stutter_mask[i]:
                        continue
                    prev_lab = seq[i - 1] if i - 1 >= 0 else None
                    next_lab = seq[i + 1] if i + 1 < len(seq) else None
                    if prev_lab is not None and prev_lab not in ignore:
                        incoming_counts_base[gkey][lab][prev_lab] += 1
                    if next_lab is not None and next_lab not in ignore:
                        outgoing_counts_base[gkey][lab][next_lab] += 1
                    n_base[gkey][lab] += 1

    rows: List[Dict[str, Any]] = []
    all_keys = sorted(n_stutter.keys(), key=str)
    for gkey in all_keys:
        labs = set(n_stutter[gkey].keys())
        if compute_baseline_nonstutter:
            labs |= set(n_base[gkey].keys())

        for lab in sorted(labs, key=str):
            inc = incoming_counts[gkey].get(lab, Counter())
            out = outgoing_counts[gkey].get(lab, Counter())

            r: Dict[str, Any] = {
                "group": gkey,
                "syllable": lab,
                "n_stutter": n_stutter[gkey].get(lab, 0),
                "incoming_entropy_bits": entropy_bits_from_counts(inc),
                "outgoing_entropy_bits": entropy_bits_from_counts(out),
                "incoming_counts": dict(inc),
                "outgoing_counts": dict(out),
            }

            if compute_baseline_nonstutter:
                inc_b = incoming_counts_base[gkey].get(lab, Counter())
                out_b = outgoing_counts_base[gkey].get(lab, Counter())
                r.update(
                    {
                        "n_nonstutter": n_base[gkey].get(lab, 0),
                        "incoming_entropy_bits_nonstutter": entropy_bits_from_counts(inc_b),
                        "outgoing_entropy_bits_nonstutter": entropy_bits_from_counts(out_b),
                        "incoming_counts_nonstutter": dict(inc_b),
                        "outgoing_counts_nonstutter": dict(out_b),
                    }
                )

            rows.append(r)

    return pd.DataFrame(rows)


# -----------------------------
# Loading inputs (merged or auto-merged)
# -----------------------------
_HAS_MERGE_BUILDER = False
_MERGE_IMPORT_ERR: Optional[Exception] = None
try:
    from merge_annotations_from_split_songs import build_decoded_with_split_labels  # type: ignore

    _HAS_MERGE_BUILDER = True
except Exception as e:
    build_decoded_with_split_labels = None  # type: ignore
    _MERGE_IMPORT_ERR = e


def find_file_recursive(root: Path, pattern: str) -> Optional[Path]:
    hits = list(root.rglob(pattern))
    return hits[0] if hits else None


def load_premerged(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".json"}:
        return pd.read_json(path)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    # fallback
    return pd.read_csv(path)


def load_merged_from_json(decoded_json: Path, song_det_json: Path) -> pd.DataFrame:
    if not _HAS_MERGE_BUILDER or build_decoded_with_split_labels is None:
        raise ImportError(
            "merge_annotations_from_split_songs.build_decoded_with_split_labels could not be imported.\n"
            f"Original import error: {_MERGE_IMPORT_ERR}"
        )

    ann = build_decoded_with_split_labels(
        decoded_database_json=decoded_json,
        song_detection_json=song_det_json,
        only_song_present=True,
        compute_durations=False,
        add_recording_datetime=True,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=0,
        segment_index_offset=0,
        merge_repeated_syllables=False,
        repeat_gap_ms=0,
        repeat_gap_inclusive=True,
    )
    df = ann.annotations_appended_df.copy()
    return df


def lookup_treatment_date_from_excel(
    metadata_xlsx: Path,
    *,
    sheet: str,
    animal_id_col: str,
    treatment_date_col: str,
    bird: str,
) -> Optional[pd.Timestamp]:
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet)
    if animal_id_col not in meta.columns or treatment_date_col not in meta.columns:
        raise ValueError(
            f"Excel is missing required columns. "
            f"Need {animal_id_col!r} and {treatment_date_col!r}. "
            f"Found: {list(meta.columns)}"
        )
    m = meta.loc[meta[animal_id_col].astype(str) == str(bird)]
    if m.empty:
        return None
    td = parse_date_like(m.iloc[0][treatment_date_col])
    return td


def filter_scope(df: pd.DataFrame, scope: str, treatment_date: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    scope: 'post', 'pre', 'all'
    Requires df['_dt'] present.
    """
    if scope == "all" or treatment_date is None:
        return df

    td = pd.Timestamp(treatment_date).normalize()
    if scope == "post":
        return df.loc[df["_dt"] >= td].copy()
    if scope == "pre":
        return df.loc[df["_dt"] < td].copy()
    raise ValueError("scope must be one of: post, pre, all")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute stutter-conditioned incoming/outgoing transition entropy per syllable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data selection
    ap.add_argument("--root-dir", type=str, default=None, help="Root directory containing bird folders (optional).")
    ap.add_argument("--bird", type=str, required=True, help="Animal ID, e.g. USA5288.")
    ap.add_argument("--premerged", type=str, default=None, help="Path to premerged annotations table (CSV/TSV/JSON/parquet).")

    ap.add_argument("--decoded-json", type=str, default=None, help="Path to *_decoded_database.json (optional).")
    ap.add_argument("--song-detection-json", type=str, default=None, help="Path to *_song_detection.json (optional).")

    # Treatment date filtering
    ap.add_argument("--treatment-date", type=str, default=None, help="Treatment date (YYYY-MM-DD).")
    ap.add_argument("--metadata-xlsx", type=str, default=None, help="Excel to lookup treatment date.")
    ap.add_argument("--metadata-sheet", type=str, default="metadata", help="Sheet name for treatment date lookup.")
    ap.add_argument("--animal-id-col", type=str, default="Animal ID", help="Column with animal IDs.")
    ap.add_argument("--treatment-date-col", type=str, default="Treatment date", help="Column with treatment dates.")
    ap.add_argument("--scope", type=str, default="post", choices=["post", "pre", "all"], help="Compute on post/pre/all rows.")

    # Sequence interpretation
    ap.add_argument("--seq-col", type=str, default=None, help="Column holding sequence (list or stringified list).")
    ap.add_argument("--interpret-as-timebins", action="store_true", help="Convert time-bin labels to token sequence before stutter detection.")
    ap.add_argument("--no-interpret-as-timebins", action="store_true", help="Force interpret_as_timebins=False.")
    ap.add_argument("--ignore-labels", nargs="*", default=["-1"], help="Labels to ignore (e.g., -1 silence).")
    ap.add_argument("--drop-ignored-after-tokenize", action="store_true", help="When interpreting as timebins, drop ignored labels after collapse.")
    ap.set_defaults(drop_ignored_after_tokenize=True)

    # Stutter / entropy options
    ap.add_argument("--stutter-min-run", type=int, default=2, help="Run length threshold (>=) to count as stutter.")
    ap.add_argument("--mode", type=str, default="events", choices=["events", "positions"], help="Events or positions conditioning.")
    ap.add_argument("--no-group-by-day", action="store_true", help="Aggregate across all days.")
    ap.add_argument("--compute-baseline", action="store_true", help="Also compute non-stutter baseline in/out entropy.")
    ap.add_argument("--min-events", type=int, default=1, help="Filter out rows with fewer than this many stutter events (per syllable).")

    # Outputs
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    ap.add_argument("--out-prefix", type=str, default=None, help="Prefix for output files.")
    ap.add_argument("--also-save-json", action="store_true", help="Also save counts as JSON (incoming/outgoing) per row.")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    bird = str(args.bird)

    # 1) Load table
    df: Optional[pd.DataFrame] = None

    if args.premerged:
        df = load_premerged(Path(args.premerged))
    else:
        # auto-merge
        root = Path(args.root_dir) if args.root_dir else None

        decoded_json = Path(args.decoded_json) if args.decoded_json else None
        song_det_json = Path(args.song_detection_json) if args.song_detection_json else None

        if decoded_json is None or song_det_json is None:
            if root is None:
                raise ValueError("If you don't pass --premerged, provide --root-dir so I can locate JSONs.")
            # common locations
            decoded_json = decoded_json or (
                root / bird / f"{bird}_decoded_database.json"
                if (root / bird / f"{bird}_decoded_database.json").exists()
                else find_file_recursive(root, f"{bird}_decoded_database.json")
            )
            song_det_json = song_det_json or (
                root / bird / f"{bird}_song_detection.json"
                if (root / bird / f"{bird}_song_detection.json").exists()
                else find_file_recursive(root, f"{bird}_song_detection.json")
            )

        if decoded_json is None or song_det_json is None:
            raise FileNotFoundError(
                "Could not find decoded/song_detection JSONs. "
                "Pass --decoded-json and --song-detection-json explicitly, or check --root-dir."
            )

        df = load_merged_from_json(decoded_json, song_det_json)

    if df is None or df.empty:
        raise ValueError("No rows loaded (df is empty).")

    # 2) Datetime + sort
    dt = choose_datetime_series(df)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    # 3) Treatment date
    treatment_date: Optional[pd.Timestamp] = parse_date_like(args.treatment_date)

    if treatment_date is None and args.metadata_xlsx:
        td = lookup_treatment_date_from_excel(
            Path(args.metadata_xlsx),
            sheet=args.metadata_sheet,
            animal_id_col=args.animal_id_col,
            treatment_date_col=args.treatment_date_col,
            bird=bird,
        )
        treatment_date = td

    if args.scope in {"post", "pre"} and treatment_date is None:
        raise ValueError("For scope=post/pre you must provide --treatment-date or --metadata-xlsx lookup.")

    df = filter_scope(df, args.scope, treatment_date)

    if df.empty:
        raise ValueError(f"No rows remain after scope={args.scope!r} filtering (bird={bird}).")

    # 4) Sequence column + interpretation choice
    seq_col = auto_pick_seq_col(df, args.seq_col)
    interpret_as_timebins = infer_interpret_as_timebins(
        seq_col,
        explicit=False if args.no_interpret_as_timebins else (True if args.interpret_as_timebins else None),
    )

    # 5) Compute
    out_df = compute_stutter_in_out_entropy(
        df,
        seq_col=seq_col,
        dt_col="_dt",
        group_by_day=not args.no_group_by_day,
        stutter_min_run=args.stutter_min_run,
        mode=args.mode,
        ignore_labels=args.ignore_labels,
        interpret_as_timebins=interpret_as_timebins,
        drop_ignored_after_tokenize=args.drop_ignored_after_tokenize,
        compute_baseline_nonstutter=args.compute_baseline,
    )

    if out_df.empty:
        print("No stutter events found (output is empty).")
        return

    # Filter min events
    out_df = out_df.loc[out_df["n_stutter"] >= int(args.min_events)].copy()
    if out_df.empty:
        print(f"No rows meet --min-events {args.min_events}.")
        return

    # Add helpful metadata columns
    out_df.insert(0, "animal_id", bird)
    out_df.insert(1, "scope", args.scope)
    out_df.insert(2, "seq_col", seq_col)
    out_df.insert(3, "interpret_as_timebins", interpret_as_timebins)

    # 6) Save outputs
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"{bird}_stutter_inout_{args.scope}"

    csv_path = out_dir / f"{prefix}.csv"
    out_df.to_csv(csv_path, index=False)

    # Also save an aggregated-across-days summary
    # (recompute with group_by_day=False to make sure counts sum correctly)
    summary_df = compute_stutter_in_out_entropy(
        df,
        seq_col=seq_col,
        dt_col="_dt",
        group_by_day=False,
        stutter_min_run=args.stutter_min_run,
        mode=args.mode,
        ignore_labels=args.ignore_labels,
        interpret_as_timebins=interpret_as_timebins,
        drop_ignored_after_tokenize=args.drop_ignored_after_tokenize,
        compute_baseline_nonstutter=args.compute_baseline,
    )
    if not summary_df.empty:
        summary_df.insert(0, "animal_id", bird)
        summary_df.insert(1, "scope", args.scope)
        summary_df.insert(2, "seq_col", seq_col)
        summary_df.insert(3, "interpret_as_timebins", interpret_as_timebins)
        summary_csv = out_dir / f"{prefix}_SUMMARY.csv"
        summary_df.to_csv(summary_csv, index=False)
    else:
        summary_csv = None

    if args.also_save_json:
        # Save counts only (friendly for downstream)
        json_rows = out_df[["animal_id", "group", "syllable", "incoming_counts", "outgoing_counts"]].to_dict("records")
        json_path = out_dir / f"{prefix}_counts.json"
        with open(json_path, "w") as f:
            json.dump(json_rows, f, indent=2)
    else:
        json_path = None

    # 7) Print
    print("\n=== Stutter in/out entropy computed ===")
    print(f"Bird: {bird}")
    if treatment_date is not None:
        print(f"Treatment date: {pd.Timestamp(treatment_date).date().isoformat()}")
    print(f"Scope: {args.scope}")
    print(f"Rows analyzed: {len(df)}")
    print(f"Sequence col: {seq_col}")
    print(f"Interpret as timebins: {interpret_as_timebins}")
    print(f"Mode: {args.mode}")
    print(f"Stutter min run: {args.stutter_min_run}")
    print(f"Ignore labels: {args.ignore_labels}")
    print(f"\nSaved: {csv_path}")
    if summary_csv is not None:
        print(f"Saved: {summary_csv}")
    if json_path is not None:
        print(f"Saved: {json_path}")
    print("======================================\n")


if __name__ == "__main__":
    main()
