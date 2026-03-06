#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stutter_in_out_entropy_by_duration_with_plots.py

Compute stutter-conditioned incoming/outgoing transition entropy using:
  stutter token := phrase duration > per-syllable PRE-treatment baseline threshold

This version adds plotting:
  1) SUMMARY scatter: incoming vs outgoing entropy (size ~ # stutter events)
  2) SUMMARY bars: incoming entropy (sorted by # stutter events)
  3) SUMMARY bars: outgoing entropy (sorted by # stutter events)
  4) SUMMARY scatter: baseline threshold vs median stutter duration
  5) Per-day heatmap: stutter counts (dates x syllables)
  6) Per-day line: total stutter events per day

Outputs
-------
CSV:
  <prefix>.csv
  <prefix>_SUMMARY.csv

PNG:
  <prefix>__summary_entropy_scatter.png
  <prefix>__summary_incoming_entropy_bars.png
  <prefix>__summary_outgoing_entropy_bars.png
  <prefix>__summary_threshold_vs_stutter_duration.png
  <prefix>__daily_stutter_count_heatmap.png
  <prefix>__daily_total_stutter_events.png
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt


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
        "Could not find a spans dict column. Pass --spans-col explicitly. "
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
    events: List[Tuple[float, float, str]] = []

    for lab, spans in spans_dict.items():
        lab_s = str(lab)
        if lab_s in ignore:
            continue
        if spans is None:
            continue

        if isinstance(spans, (tuple, list)):
            span_list = list(spans)
        else:
            span_list = [spans]

        for sp in span_list:
            on = off = None

            if isinstance(sp, dict):
                for a, b in (("onset_ms", "offset_ms"), ("onset", "offset"), ("on", "off")):
                    if a in sp and b in sp:
                        on, off = sp[a], sp[b]
                        break

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
    ignore = _coerce_ignore_set(ignore_labels)

    td = pd.Timestamp(treatment_date).normalize()

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
# Plotting
# -----------------------------
def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_results(
    per_day_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    out_dir: Path,
    prefix: str,
    top_k_label: int = 10,
    min_events_for_plots: int = 1,
) -> List[Path]:
    saved: List[Path] = []

    if summary_df is None or summary_df.empty:
        return saved

    s = summary_df.copy()
    s = s.loc[s["n_stutter_events"] >= int(min_events_for_plots)].copy()
    if s.empty:
        return saved

    # sort syllables by stutter count
    s = s.sort_values("n_stutter_events", ascending=False).reset_index(drop=True)

    # 1) Summary scatter: incoming vs outgoing entropy
    x = s["incoming_entropy_bits"].astype(float).to_numpy()
    y = s["outgoing_entropy_bits"].astype(float).to_numpy()
    n = s["n_stutter_events"].astype(float).to_numpy()
    sizes = (np.sqrt(np.maximum(n, 0)) + 1.0) * 30.0

    plt.figure(figsize=(7.5, 6.0))
    plt.scatter(x, y, s=sizes, alpha=0.7)
    lo = np.nanmin(np.concatenate([x, y]))
    hi = np.nanmax(np.concatenate([x, y]))
    if np.isfinite(lo) and np.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    plt.xlabel("Incoming entropy (bits)")
    plt.ylabel("Outgoing entropy (bits)")
    plt.title("Stutter-conditioned entropy (per syllable)\n(size ∝ # stutter events)")

    for _, row in s.head(int(top_k_label)).iterrows():
        plt.text(float(row["incoming_entropy_bits"]), float(row["outgoing_entropy_bits"]), str(row["syllable"]), fontsize=8)

    p = out_dir / f"{prefix}__summary_entropy_scatter.png"
    _savefig(p)
    saved.append(p)

    # 2) Incoming entropy bars
    plt.figure(figsize=(12, 4.5))
    plt.bar([str(v) for v in s["syllable"]], s["incoming_entropy_bits"].astype(float))
    plt.xlabel("Syllable")
    plt.ylabel("Incoming entropy (bits)")
    plt.title("Incoming entropy (stutter-conditioned), sorted by # stutter events")
    plt.xticks(rotation=90)
    p = out_dir / f"{prefix}__summary_incoming_entropy_bars.png"
    _savefig(p)
    saved.append(p)

    # 3) Outgoing entropy bars
    plt.figure(figsize=(12, 4.5))
    plt.bar([str(v) for v in s["syllable"]], s["outgoing_entropy_bits"].astype(float))
    plt.xlabel("Syllable")
    plt.ylabel("Outgoing entropy (bits)")
    plt.title("Outgoing entropy (stutter-conditioned), sorted by # stutter events")
    plt.xticks(rotation=90)
    p = out_dir / f"{prefix}__summary_outgoing_entropy_bars.png"
    _savefig(p)
    saved.append(p)

    # 4) Threshold vs median stutter duration
    plt.figure(figsize=(7.5, 6.0))
    x2 = s["baseline_threshold_ms"].astype(float).to_numpy()
    y2 = s["median_stutter_duration_ms"].astype(float).to_numpy()
    plt.scatter(x2, y2, s=sizes, alpha=0.7)
    finite = np.isfinite(x2) & np.isfinite(y2)
    if finite.any():
        lo2 = float(np.nanmin(np.concatenate([x2[finite], y2[finite]])))
        hi2 = float(np.nanmax(np.concatenate([x2[finite], y2[finite]])))
        plt.plot([lo2, hi2], [lo2, hi2], linestyle="--", linewidth=1)
    plt.xlabel("Baseline threshold (ms)")
    plt.ylabel("Median stutter duration (ms)")
    plt.title("Duration threshold vs stutter duration\n(size ∝ # stutter events)")
    for _, row in s.head(int(top_k_label)).iterrows():
        plt.text(float(row["baseline_threshold_ms"]), float(row["median_stutter_duration_ms"]), str(row["syllable"]), fontsize=8)
    p = out_dir / f"{prefix}__summary_threshold_vs_stutter_duration.png"
    _savefig(p)
    saved.append(p)

    # Daily plots
    if per_day_df is None or per_day_df.empty:
        return saved

    d = per_day_df.copy()
    d = d.loc[d["n_stutter_events"] >= int(min_events_for_plots)].copy()
    if d.empty:
        return saved

    # ensure date order
    d["date"] = pd.to_datetime(d["group"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    # 5) Heatmap: daily stutter counts (dates x syllables)
    # order syllables by total in summary
    syll_order = [str(v) for v in s["syllable"]]
    pivot = d.pivot_table(index="date", columns="syllable", values="n_stutter_events", aggfunc="sum").fillna(0)
    # reindex columns to syll_order intersection
    cols = [c for c in syll_order if c in pivot.columns]
    pivot = pivot[cols] if cols else pivot

    plt.figure(figsize=(12, 5.5))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(label="Stutter events (count)")
    plt.yticks(range(len(pivot.index)), [dt.date().isoformat() for dt in pivot.index], fontsize=8)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=90, fontsize=8)
    plt.xlabel("Syllable")
    plt.ylabel("Date")
    plt.title("Daily stutter event counts (duration-threshold)")
    p = out_dir / f"{prefix}__daily_stutter_count_heatmap.png"
    _savefig(p)
    saved.append(p)

    # 6) Total stutter events per day
    totals = pivot.sum(axis=1)
    plt.figure(figsize=(10, 4.5))
    plt.plot(pivot.index, totals.to_numpy(), marker="o")
    plt.xlabel("Date")
    plt.ylabel("Total stutter events")
    plt.title("Total stutter events per day (duration-threshold)")
    plt.xticks(rotation=45)
    p = out_dir / f"{prefix}__daily_total_stutter_events.png"
    _savefig(p)
    saved.append(p)

    return saved



# -----------------------------
# Song-level entropy: stuttered vs non-stuttered songs
# -----------------------------
def _songlevel_weighted_conditional_entropy_bits(labels: List[str]) -> Tuple[float, float, int]:
    """
    Compute song-level outgoing and incoming entropy (bits) from a token label sequence.

    Outgoing: for each source syllable s, compute H(next | s), then weight by P(s) over transitions.
    Incoming: for each destination syllable d, compute H(prev | d), then weight by P(d) over transitions.

    Returns (H_out, H_in, n_transitions).
    """
    if labels is None or len(labels) < 2:
        return float("nan"), float("nan"), 0

    out_counts: Dict[str, Counter] = defaultdict(Counter)
    in_counts: Dict[str, Counter] = defaultdict(Counter)
    out_totals: Counter = Counter()
    in_totals: Counter = Counter()

    for i in range(len(labels) - 1):
        s = labels[i]
        t = labels[i + 1]
        out_counts[s][t] += 1
        in_counts[t][s] += 1
        out_totals[s] += 1
        in_totals[t] += 1

    n_trans = len(labels) - 1
    if n_trans <= 0:
        return float("nan"), float("nan"), 0

    H_out = 0.0
    for s, n_s in out_totals.items():
        H_out += (n_s / n_trans) * _entropy_bits_from_counts(out_counts[s])

    H_in = 0.0
    for d, n_d in in_totals.items():
        H_in += (n_d / n_trans) * _entropy_bits_from_counts(in_counts[d])

    return float(H_out), float(H_in), int(n_trans)


def compute_songlevel_entropy_stutter_vs_non(
    df: pd.DataFrame,
    *,
    spans_col: str,
    treatment_date: pd.Timestamp,
    scope: str,
    baseline_percentile: float,
    min_baseline_n: int,
    fallback_baseline: str,
    ignore_labels: Optional[Iterable[str]],
    song_stutter_min_events: int = 1,
    song_min_transitions: int = 5,
) -> pd.DataFrame:
    """
    Build a per-song table with:
      - H_out_song_bits, H_in_song_bits
      - n_transitions
      - n_stutter_tokens (duration > baseline threshold)
      - is_stutter_song (n_stutter_tokens >= song_stutter_min_events)
    """
    ignore = _coerce_ignore_set(ignore_labels)
    td = pd.Timestamp(treatment_date).normalize()

    df_pre = df.loc[df["_dt"] < td].copy()
    if df_pre.empty:
        raise ValueError("No PRE-treatment rows available to compute baseline thresholds.")

    baseline_by_label, global_fallback, _ = _build_baseline_thresholds(
        df_pre,
        spans_col=spans_col,
        ignore=ignore,
        percentile=float(baseline_percentile),
        min_n=int(min_baseline_n),
        fallback=str(fallback_baseline),
    )

    def _thr(label: str) -> Optional[float]:
        v = baseline_by_label.get(label, float("nan"))
        if not np.isnan(v):
            return float(v)
        if global_fallback is not None:
            return float(global_fallback)
        return None

    if scope == "all":
        df_scope = df
    elif scope == "post":
        df_scope = df.loc[df["_dt"] >= td]
    elif scope == "pre":
        df_scope = df.loc[df["_dt"] < td]
    else:
        raise ValueError("scope must be one of: post, pre, all")

    rows: List[Dict[str, Any]] = []
    for idx, row in df_scope.iterrows():
        toks = _row_token_events(row, spans_col=spans_col, ignore=ignore)
        if not toks:
            continue

        labels = [t.label for t in toks if t.label not in ignore]
        if len(labels) < 2:
            continue

        H_out, H_in, n_trans = _songlevel_weighted_conditional_entropy_bits(labels)
        if n_trans < int(song_min_transitions):
            continue

        n_stutter = 0
        for t in toks:
            if t.label in ignore:
                continue
            thr = _thr(t.label)
            if thr is None:
                continue
            if float(t.duration_ms) > float(thr):
                n_stutter += 1

        is_stutter_song = int(n_stutter) >= int(song_stutter_min_events)

        dt = row.get("_dt", pd.NaT)
        date = pd.Timestamp(dt).date().isoformat() if not pd.isna(dt) else "__NO_DATE__"

        song_id = None
        for c in ("file_name", "wav_file", "recording_file", "filename", "File", "wav_path"):
            if c in row.index and pd.notna(row[c]):
                song_id = str(row[c])
                break
        if song_id is None:
            song_id = str(idx)

        rows.append(
            {
                "date": date,
                "song_id": song_id,
                "n_transitions": int(n_trans),
                "H_out_song_bits": float(H_out),
                "H_in_song_bits": float(H_in),
                "n_stutter_tokens": int(n_stutter),
                "is_stutter_song": bool(is_stutter_song),
            }
        )

    return pd.DataFrame(rows)


def _box_with_jitter(values_a: np.ndarray, values_b: np.ndarray, labels: Tuple[str, str], title: str, ylabel: str, out_path: Path) -> None:
    a = values_a[np.isfinite(values_a)]
    b = values_b[np.isfinite(values_b)]

    plt.figure(figsize=(6.5, 5.0))
    plt.boxplot([a, b], labels=list(labels), showfliers=False)

    rng = np.random.default_rng(0)
    x1 = 1 + rng.normal(0, 0.04, size=len(a))
    x2 = 2 + rng.normal(0, 0.04, size=len(b))
    plt.scatter(x1, a, alpha=0.6, s=18)
    plt.scatter(x2, b, alpha=0.6, s=18)

    plt.title(title)
    plt.ylabel(ylabel)
    _savefig(out_path)


def plot_songlevel_stutter_vs_non(song_df: pd.DataFrame, *, out_dir: Path, prefix: str) -> List[Path]:
    saved: List[Path] = []
    if song_df is None or song_df.empty:
        return saved

    df = song_df.copy()
    df["is_stutter_song"] = df["is_stutter_song"].astype(bool)

    st = df.loc[df["is_stutter_song"]]
    ns = df.loc[~df["is_stutter_song"]]

    plt.figure(figsize=(7.0, 6.0))
    if not ns.empty:
        plt.scatter(ns["H_in_song_bits"], ns["H_out_song_bits"], alpha=0.7, label=f"Non-stutter (n={len(ns)})")
    if not st.empty:
        plt.scatter(st["H_in_song_bits"], st["H_out_song_bits"], alpha=0.7, label=f"Stutter (n={len(st)})")

    plt.xlabel("Song-level incoming entropy (bits)")
    plt.ylabel("Song-level outgoing entropy (bits)")
    plt.title("Song-level transition entropy\n(stuttered vs non-stuttered songs)")
    plt.legend(frameon=False)
    p = out_dir / f"{prefix}__songlevel_entropy_scatter.png"
    _savefig(p)
    saved.append(p)

    if st.empty or ns.empty:
        return saved

    p1 = out_dir / f"{prefix}__songlevel_outgoing_entropy_box.png"
    _box_with_jitter(
        ns["H_out_song_bits"].to_numpy(dtype=float),
        st["H_out_song_bits"].to_numpy(dtype=float),
        ("Non-stutter", "Stutter"),
        f"Outgoing entropy by song type (n={len(ns)} vs {len(st)})",
        "Song-level outgoing entropy (bits)",
        p1,
    )
    saved.append(p1)

    p2 = out_dir / f"{prefix}__songlevel_incoming_entropy_box.png"
    _box_with_jitter(
        ns["H_in_song_bits"].to_numpy(dtype=float),
        st["H_in_song_bits"].to_numpy(dtype=float),
        ("Non-stutter", "Stutter"),
        f"Incoming entropy by song type (n={len(ns)} vs {len(st)})",
        "Song-level incoming entropy (bits)",
        p2,
    )
    saved.append(p2)

    p3 = out_dir / f"{prefix}__songlevel_stutter_token_counts.png"
    _box_with_jitter(
        ns["n_stutter_tokens"].to_numpy(dtype=float),
        st["n_stutter_tokens"].to_numpy(dtype=float),
        ("Non-stutter", "Stutter"),
        "Stutter-token count by song type",
        "# duration-threshold stutter tokens",
        p3,
    )
    saved.append(p3)

    return saved


# -----------------------------
# CLI
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute stutter-conditioned in/out transition entropy using duration > baseline threshold, and plot results.",
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

    ap.add_argument("--no-plots", action="store_true", help="Skip plotting.")
    ap.add_argument("--plot-top-k-label", type=int, default=10, help="Annotate top-k syllables on scatter plots.")

    ap.add_argument("--song-stutter-min-events", type=int, default=1,
                    help="Define a 'stuttered song' as having at least this many duration-threshold stutter tokens.")
    ap.add_argument("--song-min-transitions", type=int, default=5,
                    help="Minimum # transitions in a song to compute song-level entropies.")

    ap.add_argument("--plot-only", action="store_true",
                    help="Skip computation and only plot from existing result CSVs.")
    ap.add_argument("--results-csv", type=str, default=None,
                    help="Per-day results CSV to plot (the <prefix>.csv output).")
    ap.add_argument("--summary-csv", type=str, default=None,
                    help="Summary CSV to plot (the <prefix>_SUMMARY.csv output).")

    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    bird = str(args.bird)

    # Plot-only mode: read existing CSV outputs and generate PNGs without recomputing.
    if args.plot_only:
        if not args.results_csv and not args.summary_csv:
            raise ValueError("In --plot-only mode, pass --results-csv and/or --summary-csv.")

        out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        per_day_df = pd.read_csv(args.results_csv) if args.results_csv else pd.DataFrame()
        summary_df = pd.read_csv(args.summary_csv) if args.summary_csv else pd.DataFrame()

        if (summary_df is None or summary_df.empty) and args.results_csv and not args.summary_csv:
            # try to infer summary path
            p = Path(args.results_csv)
            inferred = p.with_name(p.stem + "_SUMMARY.csv")
            if inferred.exists():
                summary_df = pd.read_csv(inferred)

        prefix = args.out_prefix
        if not prefix:
            if args.results_csv:
                prefix = Path(args.results_csv).stem
            elif args.summary_csv:
                prefix = Path(args.summary_csv).stem.replace("_SUMMARY", "")
            else:
                prefix = "stutter_inout_by_duration"

        plot_paths = plot_results(
            per_day_df,
            summary_df,
            out_dir=out_dir,
            prefix=prefix,
            top_k_label=int(args.plot_top_k_label),
            min_events_for_plots=int(args.min_events),
        )

        if plot_paths:
            print("\nPlots saved:")
            for p in plot_paths:
                print(f"  - {p}")
        else:
            print("No plots produced (empty data or filtered out by --min-events).")
        return

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
        mid = df["_dt"].iloc[len(df) // 2]
        treatment_date = pd.Timestamp(mid).normalize()

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

    plot_paths: List[Path] = []
    songlevel_plot_paths: List[Path] = []
    songlevel_csv_path: Optional[Path] = None

    # Song-level comparison: stuttered vs non-stuttered songs
    try:
        song_df = compute_songlevel_entropy_stutter_vs_non(
            df,
            spans_col=spans_col,
            treatment_date=pd.Timestamp(treatment_date),
            scope=args.scope,
            baseline_percentile=float(args.baseline_percentile),
            min_baseline_n=int(args.min_baseline_n),
            fallback_baseline=str(args.fallback_baseline),
            ignore_labels=args.ignore_labels,
            song_stutter_min_events=int(args.song_stutter_min_events),
            song_min_transitions=int(args.song_min_transitions),
        )
        if song_df is not None and not song_df.empty:
            songlevel_csv_path = out_dir / f"{prefix}__songlevel.csv"
            song_df.to_csv(songlevel_csv_path, index=False)
        else:
            song_df = pd.DataFrame()
    except Exception as e:
        song_df = pd.DataFrame()
        print(f"[WARN] song-level computation failed: {e}")

    if not args.no_plots:
        plot_paths = plot_results(
            per_group,
            summary,
            out_dir=out_dir,
            prefix=prefix,
            top_k_label=int(args.plot_top_k_label),
            min_events_for_plots=int(args.min_events),
        )
        if song_df is not None and not song_df.empty:
            songlevel_plot_paths = plot_songlevel_stutter_vs_non(song_df, out_dir=out_dir, prefix=prefix)

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
    if plot_paths:
        print("\nPlots saved:")
        for p in plot_paths:
            print(f"  - {p}")
    if songlevel_csv_path is not None:
        print(f"\nSong-level CSV saved:\n  - {songlevel_csv_path}")
    if songlevel_plot_paths:
        print("\nSong-level plots saved:")
        for p in songlevel_plot_paths:
            print(f"  - {p}")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
