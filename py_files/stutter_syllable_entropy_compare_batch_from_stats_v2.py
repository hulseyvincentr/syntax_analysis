#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stutter_syllable_entropy_compare_batch_from_stats_v2.py

Batch syllable-level comparison (stutter vs non-stutter contexts) across birds,
restricted to TOP variance syllables per bird (default: top 30%).

Key changes vs v1
-----------------
1) Colors match your requested scheme:
   - Area X visible (single hit): light purple
   - Area X visible (medial+lateral hit): dark purple
   - large lesion Area X not visible: black
   - sham saline injection: red
   - unknown/other: gray

2) Reduced point-size inflation to reduce overlap:
   - size scales ~ log1p(#stutter_tokens) instead of sqrt
   - capped max marker size

3) By default, ONLY plots (and outputs) syllables in the top 30% of variance
   *for each bird* (variance looked up from your stats CSV).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


_HAS_MERGE_BUILDER = False
_MERGE_IMPORT_ERR: Optional[Exception] = None
try:
    from merge_annotations_from_split_songs import build_decoded_with_split_labels  # type: ignore
    _HAS_MERGE_BUILDER = True
except Exception as e:
    build_decoded_with_split_labels = None  # type: ignore
    _MERGE_IMPORT_ERR = e


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


def _find_birds_in_root(root: Path) -> Dict[str, Dict[str, Path]]:
    decoded_files = list(root.rglob("*_decoded_database.json"))
    birds: Dict[str, Dict[str, Path]] = {}
    pat = re.compile(r"(.+?)_decoded_database\.json$")
    for dec in decoded_files:
        m = pat.search(dec.name)
        if not m:
            continue
        bird = m.group(1)
        song = dec.with_name(f"{bird}_song_detection.json")
        if not song.exists():
            song_hits = list(root.rglob(f"{bird}_song_detection.json"))
            if song_hits:
                song = song_hits[0]
        if song.exists():
            birds[bird] = {"decoded": dec, "song_det": song}
    return birds


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
        return None
    m = meta.loc[meta[animal_id_col].astype(str) == str(bird)]
    if m.empty:
        return None
    return _parse_date_like(m.iloc[0][treatment_date_col])


def _build_lesion_type_map(
    metadata_xlsx: Optional[Path],
    *,
    sheet: str,
    animal_id_col: str,
    hit_type_col: str,
) -> Dict[str, str]:
    if metadata_xlsx is None:
        return {}
    try:
        meta = pd.read_excel(metadata_xlsx, sheet_name=sheet)
    except Exception:
        return {}
    if animal_id_col not in meta.columns or hit_type_col not in meta.columns:
        return {}
    d: Dict[str, str] = {}
    for _, r in meta.iterrows():
        a = str(r[animal_id_col])
        ht = r[hit_type_col]
        if pd.isna(ht):
            continue
        d[a] = str(ht)
    return d


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
    sdf = stats_df.loc[stats_df[bird_col].astype(str) == str(bird)].copy()
    if sdf.empty:
        return {}
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
    sdf = stats_df.loc[stats_df[bird_col].astype(str) == str(bird)].copy()
    if sdf.empty:
        return set()
    if group_col in sdf.columns:
        sdf2 = sdf.loc[sdf[group_col].astype(str) == str(group_label)]
        if not sdf2.empty:
            sdf = sdf2

    if variance_col not in sdf.columns:
        return set()

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

    inc_st: Dict[str, Counter] = defaultdict(Counter)
    out_st: Dict[str, Counter] = defaultdict(Counter)
    inc_ns: Dict[str, Counter] = defaultdict(Counter)
    out_ns: Dict[str, Counter] = defaultdict(Counter)
    n_st: Counter = Counter()
    n_ns: Counter = Counter()

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
            }
        )
    return pd.DataFrame(rows)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def lesion_color(lesion_type: str):
    s = (lesion_type or "").lower()
    if "single hit" in s:
        return "#c7b3ff"  # light purple
    if "medial+lateral" in s or "m+l" in s or "ml" in s:
        return "#5b2a86"  # dark purple
    if "not visible" in s and ("large lesion" in s or "lesion" in s):
        return "black"
    if "sham" in s:
        return "#d62728"  # red
    return "gray"


def _sizes_from_stutter_counts(n_stutter_tokens: np.ndarray) -> np.ndarray:
    n = np.maximum(n_stutter_tokens.astype(float), 0.0)
    sizes = (np.log1p(n) + 1.0) * 18.0
    return np.clip(sizes, 14.0, 220.0)


def plot_syllable_scatter(
    df_cmp: pd.DataFrame,
    *,
    direction: str,
    out_dir: Path,
    filename: str,
    title: str,
    colors: Sequence[Any],
    min_stutter_tokens: int = 1,
    legend: Optional[List[Tuple[str, Any]]] = None,
) -> Optional[Path]:
    if df_cmp is None or df_cmp.empty:
        return None
    d = df_cmp.copy()
    d = d.loc[d["n_stutter_tokens"] >= int(min_stutter_tokens)].copy()
    if d.empty:
        return None

    if direction == "outgoing":
        x = d["outgoing_entropy_nonstutter_bits"].to_numpy(dtype=float)
        y = d["outgoing_entropy_stutter_bits"].to_numpy(dtype=float)
        xlabel = "Outgoing entropy (non-stutter) [bits]"
        ylabel = "Outgoing entropy (stutter) [bits]"
    elif direction == "incoming":
        x = d["incoming_entropy_nonstutter_bits"].to_numpy(dtype=float)
        y = d["incoming_entropy_stutter_bits"].to_numpy(dtype=float)
        xlabel = "Incoming entropy (non-stutter) [bits]"
        ylabel = "Incoming entropy (stutter) [bits]"
    else:
        raise ValueError("direction must be incoming or outgoing")

    sizes = _sizes_from_stutter_counts(d["n_stutter_tokens"].to_numpy())

    colors_arr = np.asarray(list(colors))
    if len(colors_arr) == len(df_cmp):
        colors_arr = colors_arr[d.index]

    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(x, y, s=sizes, alpha=0.65, c=colors_arr, edgecolors="none")

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(np.min(np.concatenate([x[finite], y[finite]])))
        hi = float(np.max(np.concatenate([x[finite], y[finite]])))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend:
        handles = []
        labels = []
        for name, col in legend:
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=col))
            labels.append(name)
        plt.legend(handles, labels, frameon=False, fontsize=8, loc="best")

    p = out_dir / filename
    _savefig(p)
    return p


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Batch syllable-level: compare in/out entropy between stutter vs non-stutter occurrences using stats-derived thresholds.",
    )
    ap.add_argument("--root-dir", type=str, required=True)

    ap.add_argument("--birds", nargs="*", default=None)
    ap.add_argument("--spans-col", type=str, default=None)

    ap.add_argument("--metadata-xlsx", type=str, required=True)
    ap.add_argument("--metadata-sheet", type=str, default="metadata")
    ap.add_argument("--animal-id-col", type=str, default="Animal ID")
    ap.add_argument("--treatment-date-col", type=str, default="Treatment date")

    ap.add_argument("--hit-type-sheet", type=str, default="animal_hit_type_summary")
    ap.add_argument("--hit-type-col", type=str, default="Lesion hit type")

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

    ap.add_argument("--filter-top-variance-pct", type=float, default=30.0)
    ap.add_argument("--variance-col", type=str, default="Variance_ms2")

    ap.add_argument("--ignore-labels", nargs="*", default=["-1"])
    ap.add_argument("--min-stutter-tokens", type=int, default=20)

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default="syllable_stutter_vs_non")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "entropy_stutter_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseprefix = str(args.out_prefix)

    stats = pd.read_csv(Path(args.stats_csv))

    lesion_map = _build_lesion_type_map(
        Path(args.metadata_xlsx),
        sheet=args.hit_type_sheet,
        animal_id_col=args.animal_id_col,
        hit_type_col=args.hit_type_col,
    )

    bird_paths = _find_birds_in_root(root)
    bird_list = [str(b) for b in args.birds] if args.birds else sorted(bird_paths.keys(), key=str)
    if not bird_list:
        raise ValueError("No birds found. Either pass --birds or ensure root-dir contains *_decoded_database.json files.")

    all_rows: List[pd.DataFrame] = []
    processed: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for bird in bird_list:
        if bird not in bird_paths:
            skipped.append((bird, "missing decoded/song_detection JSONs in root-dir scan"))
            continue

        decoded_json = bird_paths[bird]["decoded"]
        song_det_json = bird_paths[bird]["song_det"]

        treatment_date = _lookup_treatment_date_from_excel(
            Path(args.metadata_xlsx),
            sheet=args.metadata_sheet,
            animal_id_col=args.animal_id_col,
            treatment_date_col=args.treatment_date_col,
            bird=bird,
        )
        if treatment_date is None and args.scope in {"post", "pre"}:
            skipped.append((bird, "missing treatment date in metadata"))
            continue
        if treatment_date is None:
            treatment_date = pd.Timestamp("1970-01-01")

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
            skipped.append((bird, "no thresholds found in stats table for this bird"))
            continue

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
        restrict = {s for s in restrict if s in thresholds}
        if not restrict:
            skipped.append((bird, "variance filter kept 0 syllables"))
            continue

        try:
            df = _load_merged_from_json(decoded_json, song_det_json)
        except Exception as e:
            skipped.append((bird, f"failed to merge JSONs: {e}"))
            continue
        if df.empty:
            skipped.append((bird, "merged df is empty"))
            continue

        df = df.copy()
        df["_dt"] = _choose_datetime_series(df)
        df = df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
        if df.empty:
            skipped.append((bird, "all rows missing datetime"))
            continue

        spans_col = _find_best_spans_column(df, requested=args.spans_col)

        if args.scope == "all" and (pd.Timestamp(treatment_date) == pd.Timestamp("1970-01-01")):
            treatment_date = pd.Timestamp(df["_dt"].iloc[len(df) // 2]).normalize()

        cmp_df = syllable_entropy_compare(
            df,
            spans_col=spans_col,
            treatment_date=pd.Timestamp(treatment_date),
            scope=args.scope,
            thresholds_ms=thresholds,
            ignore_labels=args.ignore_labels,
            restrict_syllables=restrict,
        )
        if cmp_df.empty:
            skipped.append((bird, "comparison dataframe empty"))
            continue

        lesion_type = lesion_map.get(bird, "UNKNOWN")
        cmp_df.insert(0, "animal_id", bird)
        cmp_df.insert(1, "lesion_type", lesion_type)
        cmp_df.insert(2, "scope", args.scope)
        cmp_df.insert(3, "spans_col", spans_col)
        cmp_df.insert(4, "variance_top_pct", float(args.filter_top_variance_pct))

        prefix = f"{bird}_{baseprefix}"
        out_csv = out_dir / f"{prefix}__syllable_compare.csv"
        cmp_df.to_csv(out_csv, index=False)

        if not args.no_plots and (not args.aggregate_only):
            col = lesion_color(lesion_type)
            colors = [col] * len(cmp_df)
            plot_syllable_scatter(
                cmp_df,
                direction="incoming",
                out_dir=out_dir,
                filename=f"{prefix}__incoming_nonstutter_x_vs_stutter_y.png",
                title=f"{bird} ({lesion_type}) incoming entropy\nx=non-stutter, y=stutter (top {args.filter_top_variance_pct:.0f}% variance)",
                colors=colors,
                min_stutter_tokens=int(args.min_stutter_tokens),
            )
            plot_syllable_scatter(
                cmp_df,
                direction="outgoing",
                out_dir=out_dir,
                filename=f"{prefix}__outgoing_nonstutter_x_vs_stutter_y.png",
                title=f"{bird} ({lesion_type}) outgoing entropy\nx=non-stutter, y=stutter (top {args.filter_top_variance_pct:.0f}% variance)",
                colors=colors,
                min_stutter_tokens=int(args.min_stutter_tokens),
            )

        all_rows.append(cmp_df)
        processed.append(bird)

    if not all_rows:
        print("No birds processed successfully.")
        if skipped:
            print("Skipped:")
            for b, why in skipped:
                print(f"  - {b}: {why}")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    all_csv = out_dir / f"{baseprefix}__ALL_BIRDS__syllable_compare.csv"
    all_df.to_csv(all_csv, index=False)

    if not args.no_plots:
        lesion_types = all_df["lesion_type"].astype(str).tolist()
        colors = [lesion_color(lt) for lt in lesion_types]
        legend = [
            ("Area X visible (medial+lateral hit)", lesion_color("Area X visible (medial+lateral hit)")),
            ("Area X visible (single hit)", lesion_color("Area X visible (single hit)")),
            ("large lesion Area X not visible", lesion_color("large lesion Area X not visible")),
            ("sham saline injection", lesion_color("sham saline injection")),
        ]

        plot_syllable_scatter(
            all_df,
            direction="incoming",
            out_dir=out_dir,
            filename=f"{baseprefix}__ALL_BIRDS__incoming_nonstutter_x_vs_stutter_y.png",
            title=f"ALL BIRDS incoming entropy\nx=non-stutter, y=stutter (top {args.filter_top_variance_pct:.0f}% variance per bird)",
            colors=colors,
            min_stutter_tokens=int(args.min_stutter_tokens),
            legend=legend,
        )
        plot_syllable_scatter(
            all_df,
            direction="outgoing",
            out_dir=out_dir,
            filename=f"{baseprefix}__ALL_BIRDS__outgoing_nonstutter_x_vs_stutter_y.png",
            title=f"ALL BIRDS outgoing entropy\nx=non-stutter, y=stutter (top {args.filter_top_variance_pct:.0f}% variance per bird)",
            colors=colors,
            min_stutter_tokens=int(args.min_stutter_tokens),
            legend=legend,
        )

    print("\n=== Batch syllable stutter vs non-stutter entropy (TOP variance syllables) ===")
    print(f"Root: {root}")
    print(f"Out dir: {out_dir}")
    print(f"Birds processed: {len(processed)}")
    print(f"Aggregate CSV: {all_csv}")
    if skipped:
        print(f"Birds skipped: {len(skipped)}")
        for b, why in skipped:
            print(f"  - {b}: {why}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
