#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stutter_in_out_entropy_by_duration.py

Compute incoming/outgoing transition entropy (bits) for each syllable
*conditioned on that syllable being "stuttered"*, where "stutter" is defined
by phrase duration exceeding a per-syllable baseline threshold.

Key idea
--------
Instead of detecting stutter as repeated tokens (A A A ...), we detect stutter
as a *single token with an unusually long phrase duration* (e.g., the JSON
already merged repeats into one longer span).

Stutter rule (default)
----------------------
For each syllable label L:
  baseline[L] = Pth percentile of PRE-treatment phrase durations for L (default P=95)
A token (L, duration_ms) is marked as stutter if:
  duration_ms > baseline[L]
(With optional fallback if baseline[L] cannot be estimated.)

Incoming/outgoing transitions (mode="events")
---------------------------------------------
For each stutter token of label L at index i in the song’s token sequence:
  incoming = label at i-1   (if exists, not ignored)
  outgoing = label at i+1   (if exists, not ignored)

Entropy
-------
incoming_entropy_bits(L)  = Shannon entropy over the distribution of incoming labels
outgoing_entropy_bits(L)  = Shannon entropy over the distribution of outgoing labels

Inputs
------
You can load data either by:
A) Auto-merge from:
   --decoded-json + --song-detection-json
   (or provide --root-dir and --bird so the script finds them)
B) A premerged table:
   --premerged

The script extracts ordered token events from a spans dict column such as
"syllable_onsets_offsets_ms_dict" by:
  - parsing the dict (json or python literal)
  - collecting (onset, offset, label) events across all labels
  - sorting by onset
  - duration_ms = offset - onset

If spans are in bins, it attempts to convert to ms using ms-per-bin fields.

Outputs
-------
Writes:
  <prefix>.csv           (per-day if group-by-day, else one group)
  <prefix>_SUMMARY.csv   (always aggregated across all days)

Typical usage
-------------
python stutter_in_out_entropy_by_duration.py \
  --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --bird "USA5288" \
  --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "metadata" \
  --animal-id-col "Animal ID" \
  --treatment-date-col "Treatment date" \
  --scope post \
  --baseline-percentile 95 \
  --min-baseline-n 20 \
  --fallback-baseline global \
  --ignore-labels -1 \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/entropy_stutter_outputs" \
  --out-prefix "USA5288_stutter_inout_by_duration"
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
def _maybe_parse_dict(x: Any) -> Optional[dict]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else None
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def _entropy_bits_from_counts(counts: Counter) -> float:
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


def _parse_date_like(x: Any) -> Optional[pd.Timestamp]:
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


def _choose_datetime_series(df: pd.DataFrame) -> pd.Series:
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

    date_candidates = ["recording_date", "Recording Date", "Date", "date"]
    time_candidates = ["recording_time", "Recording Time", "Time", "time"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if date_col and time_col:
        return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    if date_col:
        return pd.to_datetime(df[date_col], errors="coerce")

    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def _coerce_ignore_set(values: Optional[Iterable[str]]) -> Set[str]:
    if not values:
        return set()
    return {str(v) for v in values}


def _find_best_spans_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested

    candidates = [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_bins_dict",
        "syllable_onsets_offsets_timebins_dict",
        "syllable_onsets_offsets_dict",
        "syllable_onsets_offsets_ms",
        "syllable_spans_ms_dict",
        "syllable_spans_dict",
        "onsets_offsets_ms_dict",
        "onsets_offsets_dict",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # heuristic: any *_dict column with dict-like content
    dict_cols = [c for c in df.columns if c.endswith("_dict")]
    for c in dict_cols:
        series = df[c].dropna().head(50)
        ok = 0
        for v in series:
            d = _maybe_parse_dict(v)
            if isinstance(d, dict) and len(d) > 0:
                ok += 1
        if ok >= 3:
            return c

    raise ValueError(
        "Could not find a spans dict column. "
        "Pass --spans-col explicitly. "
        f"Columns include: {list(df.columns)[:40]} ..."
    )


def _row_ms_per_bin(row: pd.Series) -> Optional[float]:
    for c in ("time_bin_ms", "timebin_ms", "bin_ms", "ms_per_bin", "msPerBin"):
        if c in row.index:
            v = row[c]
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                v = float(v)
                if v > 0:
                    return v
            except Exception:
                continue
    return None


def _extract_events_from_spans_dict(
    spans_dict: dict,
    *,
    ms_per_bin: Optional[float],
    ignore: Set[str],
) -> List[Tuple[float, float, str]]:
    """
    Return list of (onset_ms, offset_ms, label_str) for all spans in dict.
    Supports span formats:
      - [on, off] or (on, off)
      - {"onset_ms":..., "offset_ms":...} (also onset/offset, on/off)
      - {"onset_bin":..., "offset_bin":...} (converted via ms_per_bin)
    """
    events: List[Tuple[float, float, str]] = []

    for lab, spans in spans_dict.items():
        lab_s = str(lab)
        if lab_s in ignore:
            continue
        if spans is None:
            continue

        # Normalize spans to list
        if isinstance(spans, (tuple, list)):
            span_list = list(spans)
        else:
            # Sometimes a single span dict is stored as dict
            span_list = [spans]

        for sp in span_list:
            on = off = None

            if isinstance(sp, dict):
                # ms keys
                for a, b in (("onset_ms", "offset_ms"), ("onset", "offset"), ("on", "off")):
                    if a in sp and b in sp:
                        on, off = sp[a], sp[b]
                        break

                # bin keys
                if on is None or off is None:
                    for a, b in (("onset_bin", "offset_bin"), ("onset_bins", "offset_bins"), ("on_bin", "off_bin")):
                        if a in sp and b in sp:
                            if ms_per_bin is None:
                                on = off = None
                            else:
                                try:
                                    on = float(sp[a]) * ms_per_bin
                                    off = float(sp[b]) * ms_per_bin
                                except Exception:
                                    on = off = None
                            break

            elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
                on, off = sp[0], sp[1]

            # Convert to floats if possible
            try:
                if on is None or off is None:
                    continue
                on_f = float(on)
                off_f = float(off)
                if not (off_f > on_f):
                    continue
                events.append((on_f, off_f, lab_s))
            except Exception:
                continue

    return events


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


def _find_file_recursive(root: Path, pattern: str) -> Optional[Path]:
    hits = list(root.rglob(pattern))
    return hits[0] if hits else None


def _load_premerged(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".json"}:
        return pd.read_json(path)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_merged_from_json(decoded_json: Path, song_det_json: Path) -> pd.DataFrame:
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
        merge_repeated_syllables=True,
        repeat_gap_ms=0,
        repeat_gap_inclusive=True,
    )
    return ann.annotations_appended_df.copy()


def _lookup_treatment_date_from_excel(
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
            f"Excel is missing required columns. Need {animal_id_col!r} and {treatment_date_col!r}. "
            f"Found: {list(meta.columns)}"
        )
    m = meta.loc[meta[animal_id_col].astype(str) == str(bird)]
    if m.empty:
        return None
    return _parse_date_like(m.iloc[0][treatment_date_col])


# -----------------------------
# Baseline + stutter-conditioned entropy
# -----------------------------
@dataclass(frozen=True)
class TokenEvent:
    label: str
    onset_ms: float
    offset_ms: float

    @property
    def duration_ms(self) -> float:
        return self.offset_ms - self.onset_ms


def _row_token_events(
    row: pd.Series,
    *,
    spans_col: str,
    ignore: Set[str],
) -> List[TokenEvent]:
    d = _maybe_parse_dict(row.get(spans_col, None))
    if not isinstance(d, dict) or not d:
        return []

    ms_per_bin = _row_ms_per_bin(row)
    events = _extract_events_from_spans_dict(d, ms_per_bin=ms_per_bin, ignore=ignore)
    if not events:
        return []

    events_sorted = sorted(events, key=lambda t: (t[0], t[1]))
    return [TokenEvent(label=lab, onset_ms=on, offset_ms=off) for (on, off, lab) in events_sorted]


def _build_baseline_thresholds(
    df_pre: pd.DataFrame,
    *,
    spans_col: str,
    ignore: Set[str],
    percentile: float,
    min_n: int,
    fallback: str,  # "global" or "none"
) -> Tuple[Dict[str, float], Optional[float], Dict[str, int]]:
    """
    Return (baseline_by_label, global_fallback, n_pre_by_label).
    baseline_by_label[label] may be NaN if insufficient samples and fallback="none".
    """
    per_label: Dict[str, List[float]] = defaultdict(list)

    for _, row in df_pre.iterrows():
        toks = _row_token_events(row, spans_col=spans_col, ignore=ignore)
        for t in toks:
            if t.label in ignore:
                continue
            per_label[t.label].append(float(t.duration_ms))

    n_pre = {lab: len(v) for lab, v in per_label.items()}

    baseline: Dict[str, float] = {}
    for lab, durs in per_label.items():
        if len(durs) >= min_n:
            baseline[lab] = float(np.percentile(durs, percentile))
        else:
            baseline[lab] = float("nan")

    global_fallback = None
    if fallback == "global":
        all_pre = [d for durs in per_label.values() for d in durs]
        if len(all_pre) > 0:
            global_fallback = float(np.percentile(all_pre, percentile))

    return baseline, global_fallback, n_pre


def compute_stutter_entropy_by_duration(
    df: pd.DataFrame,
    *,
    spans_col: str,
    treatment_date: pd.Timestamp,
    scope: str,
    baseline_percentile: float = 95.0,
    min_baseline_n: int = 20,
    fallback_baseline: str = "global",  # "global" or "none"
    ignore_labels: Optional[Iterable[str]] = None,
    group_by_day: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_group_df, summary_df) where summary_df is aggregated across all groups.
    """
    ignore = _coerce_ignore_set(ignore_labels)

    td = pd.Timestamp(treatment_date).normalize()

    # Pre rows for baseline
    df_pre = df.loc[df["_dt"] < td].copy()
    if df_pre.empty:
        raise ValueError("No PRE-treatment rows available to compute baseline thresholds.")

    baseline_by_label, global_fallback, n_pre_by_label = _build_baseline_thresholds(
        df_pre,
        spans_col=spans_col,
        ignore=ignore,
        percentile=baseline_percentile,
        min_n=min_baseline_n,
        fallback=fallback_baseline,
    )

    def _threshold_for(label: str) -> Optional[float]:
        thr = baseline_by_label.get(label, float("nan"))
        if not np.isnan(thr):
            return float(thr)
        if global_fallback is not None:
            return float(global_fallback)
        return None

    # Select scope rows for analysis
    if scope == "all":
        df_scope = df
    elif scope == "post":
        df_scope = df.loc[df["_dt"] >= td]
    elif scope == "pre":
        df_scope = df.loc[df["_dt"] < td]
    else:
        raise ValueError("scope must be one of: post, pre, all")

    if df_scope.empty:
        raise ValueError(f"No rows remain after scope={scope!r} filtering.")

    # Counts: group -> syllable -> Counter(prev/next)
    inc_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    out_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_stutter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dur_stutter: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def _group_key(dt_val: Any) -> str:
        if not group_by_day:
            return "__ALL__"
        if pd.isna(dt_val):
            return "__NO_DATE__"
        return pd.Timestamp(dt_val).date().isoformat()

    for _, row in df_scope.iterrows():
        toks = _row_token_events(row, spans_col=spans_col, ignore=ignore)
        if not toks:
            continue

        g = _group_key(row.get("_dt", pd.NaT))

        labels = [t.label for t in toks]
        durs = [t.duration_ms for t in toks]

        # mark stutter tokens by duration threshold for their label
        stutter_idx: List[int] = []
        for i, (lab, dur) in enumerate(zip(labels, durs)):
            thr = _threshold_for(lab)
            if thr is None:
                continue
            if float(dur) > float(thr):
                stutter_idx.append(i)

        for i in stutter_idx:
            lab = labels[i]
            prev_lab = labels[i - 1] if i - 1 >= 0 else None
            next_lab = labels[i + 1] if i + 1 < len(labels) else None

            if prev_lab is not None and prev_lab not in ignore:
                inc_counts[g][lab][prev_lab] += 1
            if next_lab is not None and next_lab not in ignore:
                out_counts[g][lab][next_lab] += 1

            n_stutter[g][lab] += 1
            dur_stutter[g][lab].append(float(durs[i]))

    # Build output DF (per group)
    rows: List[Dict[str, Any]] = []
    for g in sorted(set(n_stutter.keys()), key=str):
        for lab in sorted(n_stutter[g].keys(), key=str):
            inc = inc_counts[g].get(lab, Counter())
            out = out_counts[g].get(lab, Counter())

            thr = baseline_by_label.get(lab, float("nan"))
            thr_used = thr if not np.isnan(thr) else (global_fallback if global_fallback is not None else float("nan"))

            dlist = dur_stutter[g].get(lab, [])
            rows.append(
                {
                    "group": g,
                    "syllable": lab,
                    "n_stutter_events": n_stutter[g].get(lab, 0),
                    "incoming_entropy_bits": _entropy_bits_from_counts(inc),
                    "outgoing_entropy_bits": _entropy_bits_from_counts(out),
                    "baseline_threshold_ms": thr_used,
                    "n_pre_for_baseline": int(n_pre_by_label.get(lab, 0)),
                    "mean_stutter_duration_ms": float(np.mean(dlist)) if len(dlist) else float("nan"),
                    "median_stutter_duration_ms": float(np.median(dlist)) if len(dlist) else float("nan"),
                    "incoming_counts": dict(inc),
                    "outgoing_counts": dict(out),
                }
            )

    per_group_df = pd.DataFrame(rows)

    # Summary across all groups: recompute by concatenating counts (more accurate than averaging entropies)
    inc_all: Dict[str, Counter] = defaultdict(Counter)
    out_all: Dict[str, Counter] = defaultdict(Counter)
    n_all: Dict[str, int] = defaultdict(int)
    d_all: Dict[str, List[float]] = defaultdict(list)

    for g in n_stutter.keys():
        for lab, n in n_stutter[g].items():
            n_all[lab] += n
            inc_all[lab].update(inc_counts[g].get(lab, Counter()))
            out_all[lab].update(out_counts[g].get(lab, Counter()))
            d_all[lab].extend(dur_stutter[g].get(lab, []))

    srows: List[Dict[str, Any]] = []
    for lab in sorted(n_all.keys(), key=str):
        thr = baseline_by_label.get(lab, float("nan"))
        thr_used = thr if not np.isnan(thr) else (global_fallback if global_fallback is not None else float("nan"))
        dlist = d_all.get(lab, [])
        srows.append(
            {
                "group": "__ALL__",
                "syllable": lab,
                "n_stutter_events": int(n_all.get(lab, 0)),
                "incoming_entropy_bits": _entropy_bits_from_counts(inc_all.get(lab, Counter())),
                "outgoing_entropy_bits": _entropy_bits_from_counts(out_all.get(lab, Counter())),
                "baseline_threshold_ms": thr_used,
                "n_pre_for_baseline": int(n_pre_by_label.get(lab, 0)),
                "mean_stutter_duration_ms": float(np.mean(dlist)) if len(dlist) else float("nan"),
                "median_stutter_duration_ms": float(np.median(dlist)) if len(dlist) else float("nan"),
                "incoming_counts": dict(inc_all.get(lab, Counter())),
                "outgoing_counts": dict(out_all.get(lab, Counter())),
            }
        )

    summary_df = pd.DataFrame(srows)
    return per_group_df, summary_df


# -----------------------------
# CLI
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute stutter-conditioned in/out transition entropy using duration > baseline threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--root-dir", type=str, default=None)
    ap.add_argument("--bird", type=str, required=True)

    ap.add_argument("--premerged", type=str, default=None)
    ap.add_argument("--decoded-json", type=str, default=None)
    ap.add_argument("--song-detection-json", type=str, default=None)

    ap.add_argument("--spans-col", type=str, default=None, help="Spans dict column (auto-detect if omitted).")

    ap.add_argument("--treatment-date", type=str, default=None, help="YYYY-MM-DD (optional if metadata lookup is used).")
    ap.add_argument("--metadata-xlsx", type=str, default=None)
    ap.add_argument("--metadata-sheet", type=str, default="metadata")
    ap.add_argument("--animal-id-col", type=str, default="Animal ID")
    ap.add_argument("--treatment-date-col", type=str, default="Treatment date")
    ap.add_argument("--scope", type=str, default="post", choices=["post", "pre", "all"])

    ap.add_argument("--baseline-percentile", type=float, default=95.0)
    ap.add_argument("--min-baseline-n", type=int, default=20)
    ap.add_argument("--fallback-baseline", type=str, default="global", choices=["global", "none"])

    ap.add_argument("--ignore-labels", nargs="*", default=["-1"])
    ap.add_argument("--no-group-by-day", action="store_true")
    ap.add_argument("--min-events", type=int, default=1)

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default=None)

    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    bird = str(args.bird)

    # Load df
    if args.premerged:
        df = _load_premerged(Path(args.premerged))
    else:
        root = Path(args.root_dir) if args.root_dir else None
        decoded_json = Path(args.decoded_json) if args.decoded_json else None
        song_det_json = Path(args.song_detection_json) if args.song_detection_json else None

        if decoded_json is None or song_det_json is None:
            if root is None:
                raise ValueError("If you don't pass --premerged, provide --root-dir so I can locate JSONs.")
            decoded_json = decoded_json or (
                root / bird / f"{bird}_decoded_database.json"
                if (root / bird / f"{bird}_decoded_database.json").exists()
                else _find_file_recursive(root, f"{bird}_decoded_database.json")
            )
            song_det_json = song_det_json or (
                root / bird / f"{bird}_song_detection.json"
                if (root / bird / f"{bird}_song_detection.json").exists()
                else _find_file_recursive(root, f"{bird}_song_detection.json")
            )

        if decoded_json is None or song_det_json is None:
            raise FileNotFoundError(
                "Could not find decoded/song_detection JSONs. "
                "Pass --decoded-json and --song-detection-json explicitly, or check --root-dir."
            )

        df = _load_merged_from_json(decoded_json, song_det_json)

    if df.empty:
        raise ValueError("No rows loaded (df is empty).")

    # Datetime
    df = df.copy()
    df["_dt"] = _choose_datetime_series(df)
    df = df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if df.empty:
        raise ValueError("All rows had missing datetime; can't proceed.")

    # Treatment date
    treatment_date = _parse_date_like(args.treatment_date)
    if treatment_date is None and args.metadata_xlsx:
        treatment_date = _lookup_treatment_date_from_excel(
            Path(args.metadata_xlsx),
            sheet=args.metadata_sheet,
            animal_id_col=args.animal_id_col,
            treatment_date_col=args.treatment_date_col,
            bird=bird,
        )

    if treatment_date is None and args.scope in {"post", "pre"}:
        raise ValueError("Need --treatment-date or --metadata-xlsx to filter scope pre/post.")

    if treatment_date is None:
        # for scope=all we still need a baseline; use median date as a fallback split
        # (not recommended, but prevents hard crash)
        mid = df["_dt"].iloc[len(df) // 2]
        treatment_date = pd.Timestamp(mid).normalize()

    # Spans column
    spans_col = _find_best_spans_column(df, requested=args.spans_col)

    per_group, summary = compute_stutter_entropy_by_duration(
        df,
        spans_col=spans_col,
        treatment_date=pd.Timestamp(treatment_date),
        scope=args.scope,
        baseline_percentile=float(args.baseline_percentile),
        min_baseline_n=int(args.min_baseline_n),
        fallback_baseline=str(args.fallback_baseline),
        ignore_labels=args.ignore_labels,
        group_by_day=(not args.no_group_by_day),
    )

    if per_group.empty and summary.empty:
        print("No stutter events found (output is empty).")
        return

    # Filter min-events
    if not per_group.empty:
        per_group = per_group.loc[per_group["n_stutter_events"] >= int(args.min_events)].copy()
    if not summary.empty:
        summary = summary.loc[summary["n_stutter_events"] >= int(args.min_events)].copy()

    if per_group.empty and summary.empty:
        print(f"No rows meet --min-events {args.min_events}.")
        return

    # Add metadata cols
    def _decorate(df0: pd.DataFrame) -> pd.DataFrame:
        df0 = df0.copy()
        df0.insert(0, "animal_id", bird)
        df0.insert(1, "scope", args.scope)
        df0.insert(2, "spans_col", spans_col)
        df0.insert(3, "baseline_percentile", float(args.baseline_percentile))
        df0.insert(4, "min_baseline_n", int(args.min_baseline_n))
        df0.insert(5, "fallback_baseline", str(args.fallback_baseline))
        df0.insert(6, "treatment_date", pd.Timestamp(treatment_date).date().isoformat())
        return df0

    per_group = _decorate(per_group) if not per_group.empty else per_group
    summary = _decorate(summary) if not summary.empty else summary

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"{bird}_stutter_inout_by_duration_{args.scope}"

    csv_path = out_dir / f"{prefix}.csv"
    per_group.to_csv(csv_path, index=False)

    summary_path = out_dir / f"{prefix}_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Stutter (duration-threshold) in/out entropy computed ===")
    print(f"Bird: {bird}")
    print(f"Treatment date: {pd.Timestamp(treatment_date).date().isoformat()}")
    print(f"Scope: {args.scope}")
    print(f"Spans column: {spans_col}")
    print(f"Baseline percentile: {args.baseline_percentile}")
    print(f"Min baseline N: {args.min_baseline_n}")
    print(f"Fallback baseline: {args.fallback_baseline}")
    print(f"Ignore labels: {args.ignore_labels}")
    print(f"Group by day: {not args.no_group_by_day}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
