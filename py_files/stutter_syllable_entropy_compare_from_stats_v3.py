#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stutter_syllable_entropy_compare_from_stats.py

Goal
----
Syllable-level comparison of transition entropy in two contexts:
  (A) "stuttered" occurrences of the syllable
  (B) "non-stuttered" occurrences of the syllable

Here, "stutter" is defined by phrase duration exceeding a threshold derived from a
lookup/stats table (e.g., usage_balanced_phrase_duration_stats.csv), rather than
counting repeated tokens.

Definitions
-----------
For each syllable L, define a duration threshold using PRE stats from the lookup table:

- threshold-method = q3_approx (default):
      thr_L = Pre_Median_ms + 0.5 * Pre_IQR_ms
  (approximate PRE 75th percentile / top quartile if distribution is roughly symmetric)

- threshold-method = median_plus_iqr:
      thr_L = Pre_Median_ms + 1.0 * Pre_IQR_ms

- threshold-method = median_plus_1p5iqr:
      thr_L = Pre_Median_ms + 1.5 * Pre_IQR_ms

Then, within the analysis scope (default POST),
each token event (L, duration_ms) is "stuttered" if duration_ms > thr_L.

For each syllable L, we compute incoming/outgoing neighbor distributions separately
for stuttered vs non-stuttered occurrences and compute Shannon entropy (bits):

Outgoing entropy:
  H_out_stutter(L) vs H_out_non(L)

Incoming entropy:
  H_in_stutter(L) vs H_in_non(L)

Optionally filter syllables to the top X% by a variance metric from the stats table
(e.g., Variance_ms2 for Group="Post"), so you can focus on high-variance syllables.

Inputs
------
Either:
  - --root-dir + --bird  (auto-locate decoded/song_detection JSONs and merge), or
  - --decoded-json + --song-detection-json, or
  - --premerged (CSV/TSV/JSON/parquet)

Also requires:
  - --stats-csv  (your syllable stats table)

Outputs
-------
CSV:
  <prefix>__syllable_compare.csv  (one row per syllable)

PNG (syllable-level plots):
  <prefix>__outgoing_non_vs_stutter.png
  <prefix>__incoming_non_vs_stutter.png
  <prefix>__outgoing_delta_bars.png
  <prefix>__incoming_delta_bars.png
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Small utils
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
    raise ValueError("Could not find spans dict column; pass --spans-col explicitly.")


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
        span_list = list(spans) if isinstance(spans, (tuple, list)) else [spans]
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
# Loading inputs
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
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_merged_from_json(decoded_json: Path, song_det_json: Path) -> pd.DataFrame:
    if not _HAS_MERGE_BUILDER or build_decoded_with_split_labels is None:
        raise ImportError(
            "Could not import merge builder (merge_annotations_from_split_songs.build_decoded_with_split_labels).\n"
            f"Original error: {_MERGE_IMPORT_ERR}"
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
    m = meta.loc[meta[animal_id_col].astype(str) == str(bird)]
    if m.empty:
        return None
    return _parse_date_like(m.iloc[0][treatment_date_col])


# -----------------------------
# Token events + compare entropies
# -----------------------------
@dataclass(frozen=True)
class TokenEvent:
    label: str
    onset_ms: float
    offset_ms: float
    @property
    def duration_ms(self) -> float:
        return self.offset_ms - self.onset_ms


def _row_token_events(row: pd.Series, *, spans_col: str, ignore: Set[str]) -> List[TokenEvent]:
    d = _maybe_parse_dict(row.get(spans_col, None))
    if not isinstance(d, dict) or not d:
        return []
    ms_per_bin = _row_ms_per_bin(row)
    events = _extract_events_from_spans_dict(d, ms_per_bin=ms_per_bin, ignore=ignore)
    if not events:
        return []
    events_sorted = sorted(events, key=lambda t: (t[0], t[1]))
    return [TokenEvent(label=lab, onset_ms=on, offset_ms=off) for (on, off, lab) in events_sorted]


def _compute_thresholds_from_stats(
    stats_df: pd.DataFrame,
    *,
    bird: str,
    group_label: str,
    group_col: str,
    bird_col: str,
    syll_col: str,
    pre_median_col: str,
    pre_iqr_col: str,
    threshold_method: str,
    iqr_mult: float,
    min_phrases_col: Optional[str],
    min_phrases: int,
) -> Dict[str, float]:
    sdf = stats_df.copy()
    # keep relevant bird
    sdf = sdf.loc[sdf[bird_col].astype(str) == str(bird)]
    # prefer the specified group rows (often Group="Post" contains the pre stats too)
    if group_col in sdf.columns:
        sdf_pref = sdf.loc[sdf[group_col].astype(str) == str(group_label)]
        if not sdf_pref.empty:
            sdf = sdf_pref

    thr: Dict[str, float] = {}
    for _, r in sdf.iterrows():
        lab = str(r[syll_col])
        pre_med = r.get(pre_median_col, np.nan)
        pre_iqr = r.get(pre_iqr_col, np.nan)
        if pd.isna(pre_med) or pd.isna(pre_iqr):
            continue
        if min_phrases_col and min_phrases_col in r.index:
            nph = r.get(min_phrases_col, np.nan)
            try:
                if not pd.isna(nph) and float(nph) < float(min_phrases):
                    continue
            except Exception:
                pass

        if threshold_method == "q3_approx":
            t = float(pre_med) + 0.5 * float(pre_iqr)
        elif threshold_method == "median_plus_iqr":
            t = float(pre_med) + 1.0 * float(pre_iqr)
        elif threshold_method == "median_plus_1p5iqr":
            t = float(pre_med) + float(iqr_mult) * float(pre_iqr)
        else:
            raise ValueError("threshold_method must be one of: q3_approx, median_plus_iqr, median_plus_1p5iqr")
        if t > 0:
            thr[lab] = t
    return thr


def _filter_syllables_by_variance(
    stats_df: pd.DataFrame,
    *,
    bird: str,
    group_label: str,
    group_col: str,
    bird_col: str,
    syll_col: str,
    variance_col: str,
    top_pct: float,
    min_phrases_col: str,
    min_phrases: int,
) -> Set[str]:
    sdf = stats_df.copy()
    sdf = sdf.loc[sdf[bird_col].astype(str) == str(bird)]
    if group_col in sdf.columns:
        sdf2 = sdf.loc[sdf[group_col].astype(str) == str(group_label)]
        if not sdf2.empty:
            sdf = sdf2

    if variance_col not in sdf.columns:
        raise ValueError(f"variance_col {variance_col!r} not found in stats table.")

    sdf = sdf.copy()
    if min_phrases_col in sdf.columns:
        sdf = sdf.loc[pd.to_numeric(sdf[min_phrases_col], errors="coerce") >= float(min_phrases)]

    vals = pd.to_numeric(sdf[variance_col], errors="coerce")
    sdf = sdf.loc[vals.notna()].copy()
    sdf[variance_col] = vals.loc[sdf.index]

    if sdf.empty:
        return set()

    cutoff = np.percentile(sdf[variance_col].to_numpy(), 100.0 - float(top_pct))
    keep = sdf.loc[sdf[variance_col] >= cutoff, syll_col].astype(str).tolist()
    return set(keep)


def syllable_entropy_compare(
    df: pd.DataFrame,
    *,
    spans_col: str,
    treatment_date: pd.Timestamp,
    scope: str,
    thresholds_ms: Dict[str, float],
    ignore_labels: Optional[Iterable[str]],
    restrict_syllables: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Returns one row per syllable with entropies in stutter vs non-stutter contexts.
    """
    ignore = _coerce_ignore_set(ignore_labels)
    td = pd.Timestamp(treatment_date).normalize()

    if scope == "all":
        df_scope = df
    elif scope == "post":
        df_scope = df.loc[df["_dt"] >= td]
    elif scope == "pre":
        df_scope = df.loc[df["_dt"] < td]
    else:
        raise ValueError("scope must be one of: pre, post, all")

    # Counts per syllable per condition
    inc_st: Dict[str, Counter] = defaultdict(Counter)
    out_st: Dict[str, Counter] = defaultdict(Counter)
    inc_ns: Dict[str, Counter] = defaultdict(Counter)
    out_ns: Dict[str, Counter] = defaultdict(Counter)
    n_st: Counter = Counter()
    n_ns: Counter = Counter()

    # Iterate songs (rows)
    for _, row in df_scope.iterrows():
        toks = _row_token_events(row, spans_col=spans_col, ignore=ignore)
        if not toks:
            continue
        labels = [t.label for t in toks]
        durs = [t.duration_ms for t in toks]

        for i, (lab, dur) in enumerate(zip(labels, durs)):
            if lab in ignore:
                continue
            if restrict_syllables is not None and lab not in restrict_syllables:
                continue

            thr = thresholds_ms.get(lab, None)
            if thr is None:
                continue

            prev_lab = labels[i - 1] if i - 1 >= 0 else None
            next_lab = labels[i + 1] if i + 1 < len(labels) else None

            is_st = float(dur) > float(thr)

            if is_st:
                n_st[lab] += 1
                if prev_lab is not None and prev_lab not in ignore:
                    inc_st[lab][prev_lab] += 1
                if next_lab is not None and next_lab not in ignore:
                    out_st[lab][next_lab] += 1
            else:
                n_ns[lab] += 1
                if prev_lab is not None and prev_lab not in ignore:
                    inc_ns[lab][prev_lab] += 1
                if next_lab is not None and next_lab not in ignore:
                    out_ns[lab][next_lab] += 1

    # Build table
    sylls = sorted(set(list(n_st.keys()) + list(n_ns.keys())), key=str)
    rows: List[Dict[str, Any]] = []
    for lab in sylls:
        H_in_st = _entropy_bits_from_counts(inc_st.get(lab, Counter()))
        H_out_st = _entropy_bits_from_counts(out_st.get(lab, Counter()))
        H_in_ns = _entropy_bits_from_counts(inc_ns.get(lab, Counter()))
        H_out_ns = _entropy_bits_from_counts(out_ns.get(lab, Counter()))
        rows.append(
            {
                "syllable": lab,
                "threshold_ms": float(thresholds_ms.get(lab, np.nan)),
                "n_stutter_tokens": int(n_st.get(lab, 0)),
                "n_nonstutter_tokens": int(n_ns.get(lab, 0)),
                "incoming_entropy_stutter_bits": float(H_in_st),
                "incoming_entropy_nonstutter_bits": float(H_in_ns),
                "outgoing_entropy_stutter_bits": float(H_out_st),
                "outgoing_entropy_nonstutter_bits": float(H_out_ns),
                "delta_in_bits": float(H_in_st - H_in_ns) if np.isfinite(H_in_st) and np.isfinite(H_in_ns) else float("nan"),
                "delta_out_bits": float(H_out_st - H_out_ns) if np.isfinite(H_out_st) and np.isfinite(H_out_ns) else float("nan"),
                "incoming_counts_stutter": dict(inc_st.get(lab, Counter())),
                "incoming_counts_nonstutter": dict(inc_ns.get(lab, Counter())),
                "outgoing_counts_stutter": dict(out_st.get(lab, Counter())),
                "outgoing_counts_nonstutter": dict(out_ns.get(lab, Counter())),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_syllable_compare(
    df_cmp: pd.DataFrame,
    *,
    out_dir: Path,
    prefix: str,
    label_top_k: int = 12,
    min_stutter_tokens: int = 1,
    include_delta_bars: bool = False,
) -> List[Path]:
    """
    Creates syllable-level scatter plots:
      - Outgoing: non-stutter (x) vs stutter (y)
      - Incoming: non-stutter (x) vs stutter (y)

    Optionally also creates delta bar plots if include_delta_bars=True.
    """
    saved: List[Path] = []
    if df_cmp is None or df_cmp.empty:
        return saved

    d = df_cmp.copy()
    d = d.loc[d["n_stutter_tokens"] >= int(min_stutter_tokens)].copy()
    if d.empty:
        return saved

    n = d["n_stutter_tokens"].to_numpy(dtype=float)
    sizes = (np.sqrt(np.maximum(n, 0)) + 1.0) * 28.0

    # OUTGOING scatter: x = non-stutter, y = stutter
    x = d["outgoing_entropy_nonstutter_bits"].to_numpy(dtype=float)
    y = d["outgoing_entropy_stutter_bits"].to_numpy(dtype=float)

    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(x, y, s=sizes, alpha=0.7)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(np.min(np.concatenate([x[finite], y[finite]])))
        hi = float(np.max(np.concatenate([x[finite], y[finite]])))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Outgoing entropy (non-stutter) [bits]")
    plt.ylabel("Outgoing entropy (stutter) [bits]")
    plt.title("Syllable-level outgoing entropy\nx=non-stutter, y=stutter (size ∝ # stutter tokens)")

    top = d.sort_values("n_stutter_tokens", ascending=False).head(int(label_top_k))
    for _, r in top.iterrows():
        plt.text(
            float(r["outgoing_entropy_nonstutter_bits"]),
            float(r["outgoing_entropy_stutter_bits"]),
            str(r["syllable"]),
            fontsize=8,
        )

    p = out_dir / f"{prefix}__outgoing_nonstutter_x_vs_stutter_y.png"
    _savefig(p)
    saved.append(p)

    # INCOMING scatter: x = non-stutter, y = stutter
    x2 = d["incoming_entropy_nonstutter_bits"].to_numpy(dtype=float)
    y2 = d["incoming_entropy_stutter_bits"].to_numpy(dtype=float)

    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(x2, y2, s=sizes, alpha=0.7)
    finite = np.isfinite(x2) & np.isfinite(y2)
    if finite.any():
        lo = float(np.min(np.concatenate([x2[finite], y2[finite]])))
        hi = float(np.max(np.concatenate([x2[finite], y2[finite]])))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Incoming entropy (non-stutter) [bits]")
    plt.ylabel("Incoming entropy (stutter) [bits]")
    plt.title("Syllable-level incoming entropy\nx=non-stutter, y=stutter (size ∝ # stutter tokens)")

    top = d.sort_values("n_stutter_tokens", ascending=False).head(int(label_top_k))
    for _, r in top.iterrows():
        plt.text(
            float(r["incoming_entropy_nonstutter_bits"]),
            float(r["incoming_entropy_stutter_bits"]),
            str(r["syllable"]),
            fontsize=8,
        )

    p = out_dir / f"{prefix}__incoming_nonstutter_x_vs_stutter_y.png"
    _savefig(p)
    saved.append(p)

    if include_delta_bars:
        # Delta bars (outgoing)
        d_sorted = d.sort_values("delta_out_bits", ascending=False)
        plt.figure(figsize=(12, 4.5))
        plt.bar([str(v) for v in d_sorted["syllable"]], d_sorted["delta_out_bits"].to_numpy(dtype=float))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Syllable")
        plt.ylabel("Δ outgoing entropy (stutter - non) [bits]")
        plt.title("Δ outgoing entropy by syllable")
        plt.xticks(rotation=90)
        p = out_dir / f"{prefix}__outgoing_delta_bars.png"
        _savefig(p)
        saved.append(p)

        # Delta bars (incoming)
        d_sorted = d.sort_values("delta_in_bits", ascending=False)
        plt.figure(figsize=(12, 4.5))
        plt.bar([str(v) for v in d_sorted["syllable"]], d_sorted["delta_in_bits"].to_numpy(dtype=float))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Syllable")
        plt.ylabel("Δ incoming entropy (stutter - non) [bits]")
        plt.title("Δ incoming entropy by syllable")
        plt.xticks(rotation=90)
        p = out_dir / f"{prefix}__incoming_delta_bars.png"
        _savefig(p)
        saved.append(p)

    return saved
# -----------------------------
# CLI
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Syllable-level: compare in/out entropy between stutter vs non-stutter occurrences using stats-derived thresholds."
    )
    ap.add_argument("--root-dir", type=str, default=None)
    ap.add_argument("--bird", type=str, required=True)

    ap.add_argument("--premerged", type=str, default=None)
    ap.add_argument("--decoded-json", type=str, default=None)
    ap.add_argument("--song-detection-json", type=str, default=None)

    ap.add_argument("--spans-col", type=str, default=None)

    ap.add_argument("--treatment-date", type=str, default=None)
    ap.add_argument("--metadata-xlsx", type=str, default=None)
    ap.add_argument("--metadata-sheet", type=str, default="metadata")
    ap.add_argument("--animal-id-col", type=str, default="Animal ID")
    ap.add_argument("--treatment-date-col", type=str, default="Treatment date")
    ap.add_argument("--scope", type=str, default="post", choices=["post", "pre", "all"])

    ap.add_argument("--stats-csv", type=str, required=True)
    ap.add_argument("--stats-group-label", type=str, default="Post")
    ap.add_argument("--stats-group-col", type=str, default="Group")
    ap.add_argument("--stats-bird-col", type=str, default="Animal ID")
    ap.add_argument("--stats-syllable-col", type=str, default="Syllable")

    ap.add_argument("--pre-median-col", type=str, default="Pre_Median_ms")
    ap.add_argument("--pre-iqr-col", type=str, default="Pre_IQR_ms")
    ap.add_argument("--threshold-method", type=str, default="q3_approx",
                    choices=["q3_approx", "median_plus_iqr", "median_plus_1p5iqr"])
    ap.add_argument("--iqr-mult", type=float, default=1.5)

    ap.add_argument("--min-phrases-col", type=str, default="N_phrases")
    ap.add_argument("--min-phrases", type=int, default=20)

    ap.add_argument("--filter-top-variance-pct", type=float, default=None,
                    help="If set (e.g. 30), keep only syllables in the top X%% of variance from the stats table.")
    ap.add_argument("--variance-col", type=str, default="Variance_ms2")

    ap.add_argument("--ignore-labels", nargs="*", default=["-1"])
    ap.add_argument("--min-stutter-tokens", type=int, default=5,
                    help="Only plot syllables with at least this many stutter tokens.")

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--label-top-k", type=int, default=12)

    ap.add_argument("--include-delta-bars", action="store_true",
                    help="Also save delta bar plots (Δ = stutter - non-stutter).")

    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    bird = str(args.bird)

    # Load merged annotations
    if args.premerged:
        df = _load_premerged(Path(args.premerged))
    else:
        root = Path(args.root_dir) if args.root_dir else None
        decoded_json = Path(args.decoded_json) if args.decoded_json else None
        song_det_json = Path(args.song_detection_json) if args.song_detection_json else None

        if decoded_json is None or song_det_json is None:
            if root is None:
                raise ValueError("Provide --root-dir (or explicit --decoded-json/--song-detection-json), or use --premerged.")
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
            raise FileNotFoundError("Could not locate decoded/song_detection JSONs; pass them explicitly or fix --root-dir.")
        df = _load_merged_from_json(decoded_json, song_det_json)

    if df.empty:
        raise ValueError("Loaded df is empty.")

    # Add datetime
    df = df.copy()
    df["_dt"] = _choose_datetime_series(df)
    df = df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if df.empty:
        raise ValueError("All rows missing datetime; cannot proceed.")

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
        raise ValueError("Need --treatment-date or --metadata-xlsx for scope pre/post.")
    if treatment_date is None:
        # fallback split if scope=all (still ok)
        treatment_date = pd.Timestamp(df["_dt"].iloc[len(df) // 2]).normalize()

    # Spans col
    spans_col = _find_best_spans_column(df, requested=args.spans_col)

    # Load stats
    stats = pd.read_csv(Path(args.stats_csv))
    thresholds = _compute_thresholds_from_stats(
        stats,
        bird=bird,
        group_label=args.stats_group_label,
        group_col=args.stats_group_col,
        bird_col=args.stats_bird_col,
        syll_col=args.stats_syllable_col,
        pre_median_col=args.pre_median_col,
        pre_iqr_col=args.pre_iqr_col,
        threshold_method=args.threshold_method,
        iqr_mult=float(args.iqr_mult),
        min_phrases_col=args.min_phrases_col,
        min_phrases=int(args.min_phrases),
    )
    if not thresholds:
        raise ValueError("No thresholds could be computed from stats table. Check column names / bird id / group label.")

    restrict: Optional[Set[str]] = None
    if args.filter_top_variance_pct is not None:
        restrict = _filter_syllables_by_variance(
            stats,
            bird=bird,
            group_label=args.stats_group_label,
            group_col=args.stats_group_col,
            bird_col=args.stats_bird_col,
            syll_col=args.stats_syllable_col,
            variance_col=args.variance_col,
            top_pct=float(args.filter_top_variance_pct),
            min_phrases_col=args.min_phrases_col,
            min_phrases=int(args.min_phrases),
        )
        if not restrict:
            raise ValueError("Variance filter produced 0 syllables; adjust --filter-top-variance-pct or min phrases.")
        # Also ensure we have thresholds for restricted syllables
        restrict = {s for s in restrict if s in thresholds}

    cmp_df = syllable_entropy_compare(
        df,
        spans_col=spans_col,
        treatment_date=pd.Timestamp(treatment_date),
        scope=args.scope,
        thresholds_ms=thresholds,
        ignore_labels=args.ignore_labels,
        restrict_syllables=restrict,
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"{bird}_syllable_entropy_compare_{args.scope}"

    out_csv = out_dir / f"{prefix}__syllable_compare.csv"
    cmp_df.to_csv(out_csv, index=False)

    plot_paths: List[Path] = []
    if not args.no_plots:
        plot_paths = plot_syllable_compare(
            cmp_df,
            out_dir=out_dir,
            prefix=prefix,
            label_top_k=int(args.label_top_k),
            min_stutter_tokens=int(args.min_stutter_tokens),
            include_delta_bars=bool(args.include_delta_bars),
        )

    print("\n=== Syllable-level stutter vs non-stutter entropy comparison ===")
    print(f"Bird: {bird}")
    print(f"Scope: {args.scope}")
    print(f"Spans column: {spans_col}")
    print(f"Treatment date: {pd.Timestamp(treatment_date).date().isoformat()}")
    print(f"Threshold method: {args.threshold_method} (iqr_mult={args.iqr_mult})")
    if args.filter_top_variance_pct is not None:
        print(f"Variance filter: top {args.filter_top_variance_pct}% by {args.variance_col} (kept {len(restrict or [])})")
    print(f"Saved CSV: {out_csv}")
    if plot_paths:
        print("Plots saved:")
        for p in plot_paths:
            print(f"  - {p}")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
