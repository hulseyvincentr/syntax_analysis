#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_by_syllable.py

Compute:
  • Per-syllable transition entropy H_a (conditioned on preceding syllable a)
  • Total transition entropy TE (weighted average of H_a by syllable frequency P(a))

Definition:
  H_a = - Σ_i P(i|a) log P(i|a)
  TE  = Σ_a P(a) * H_a
(where P(a) is the frequency of syllable a as the *preceding* syllable in transitions)

This file is split-song aware:
  - merges split songs / appends annotations using detection timing
  - builds an organized dataframe from the merged table
  - computes pre vs post H_a and TE relative to treatment_date

Seasonality control:
  - Optional day-window restriction around lesion date:
      max_days_pre=40  -> include only [t-40, t)  (or <= t if include_treatment_day=True)
      max_days_post=40 -> include only (t, t+40]  (or >= t if include_treatment_day=True)

Plotting improvements:
  - Plot titles rendered as FIGURE suptitle (prevents overlap/cutoff)
  - Optional parameter line appended automatically:
        window: pre≤Xd, post≤Yd | min_transitions=Z
  - Optional soft-wrapping of long titles onto multiple lines

Extra analyses (ADDED):
  1) Neighbor histograms:
     - For any target syllable a, plot BOTH:
         • most common followers (next | a)
         • most common predecessors (prev | a)
     - Can batch-save for all syllables.

  2) Remaining-to-end:
     - For each syllable a, compute the mean number of syllables remaining in the song
       AFTER each occurrence of a (collapsed-repeats optional).
     - Plot pre vs post means for each syllable with connecting lines.
     - Includes an optional legend (requested).

  3) Variance-tier entropy plots:
     - Load a per-syllable phrase-duration variance table (CSV or Excel).
     - For a given animal_id, select the top 30% variance syllables (configurable).
     - Make a boxplot of H_a (pre vs post) for:
         • high-variance syllables
         • the remaining low-variance syllables

NOTE:
  - This module intentionally does NOT import organize_decoded_with_segments.py
    to avoid circular-import issues. It includes local versions of helper funcs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for standard paired tests
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Your merge pipeline (split-song aware)
from merge_annotations_from_split_songs import build_decoded_with_split_labels


# ──────────────────────────────────────────────────────────────────────────────
# Organizer-like helpers (local copy; avoids circular-import issues)
# ──────────────────────────────────────────────────────────────────────────────
JsonLike = Union[dict, list, str, int, float, bool, None]


def parse_json_safe(value: JsonLike) -> dict:
    """Best-effort parse of JSON-like content; returns {} on failure."""
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if not isinstance(value, str):
        return {}

    s = value.strip()
    if not s:
        return {}

    # Strip wrapping quotes
    if s.startswith("''") and s.endswith("''"):
        s = s[2:-2].strip()
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    if not s:
        return {}

    # Try JSON (normalize single to double quotes)
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass

    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def extract_syllable_order(label_to_intervals: dict, *, onset_index: int = 0) -> List[str]:
    """Build per-file syllable order by sorting all syllable intervals by onset time."""
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return []

    pairs: List[tuple[float, str]] = []
    for syl, intervals in label_to_intervals.items():
        if not isinstance(intervals, (list, tuple)):
            continue
        for itv in intervals:
            if isinstance(itv, (list, tuple)) and len(itv) > onset_index:
                try:
                    onset = float(itv[onset_index])
                except Exception:
                    continue
                pairs.append((onset, str(syl)))

    pairs.sort(key=lambda p: p[0])
    return [s for _, s in pairs]


def calculate_syllable_durations_ms(label_to_intervals: dict, syllable_label: str) -> List[float]:
    """Return durations (ms) for a given syllable label."""
    if not isinstance(label_to_intervals, dict):
        return []
    intervals = label_to_intervals.get(syllable_label, [])
    out: List[float] = []
    for itv in intervals:
        if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
            continue
        try:
            on, off = float(itv[0]), float(itv[1])
            out.append(off - on)
        except Exception:
            continue
    return out


@dataclass
class OrganizedDataset:
    organized_df: pd.DataFrame
    unique_dates: List[str]
    unique_syllable_labels: List[str]
    treatment_date: Optional[str] = None  # 'YYYY.MM.DD' or None


# ──────────────────────────────────────────────────────────────────────────────
# Sequence helpers
# ──────────────────────────────────────────────────────────────────────────────
def _coerce_listlike(x) -> List[str]:
    """Coerce syllable_order / syllables_present cell into list[str]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [str(v) for v in list(x)]
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, np.ndarray)):
                return [str(v) for v in list(obj)]
        except Exception:
            pass
        return [s]
    return [str(x)]


def _collapse_consecutive(labels: Sequence[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for lab in labels:
        if prev is None or lab != prev:
            out.append(lab)
            prev = lab
    return out


def _try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _sorted_labels_natural(labels: Sequence[str]) -> List[str]:
    labs = list({str(x) for x in labels})
    numeric = [(lab, _try_int(lab)) for lab in labs]
    if numeric and all(v is not None for _, v in numeric):
        return [lab for lab, _ in sorted(numeric, key=lambda t: int(t[1]))]
    return sorted(labs)


def _sanitize_for_filename(s: str) -> str:
    """Keep filenames portable."""
    s2 = str(s).strip().replace(" ", "_")
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        s2 = s2.replace(ch, "_")
    return s2


# ──────────────────────────────────────────────────────────────────────────────
# Distinct categorical colors (30+ unique)
# ──────────────────────────────────────────────────────────────────────────────
def _get_distinct_qual_colors(n: int) -> List[tuple]:
    pools: List[tuple] = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent"]:
        try:
            cmap = plt.get_cmap(cmap_name)
            if hasattr(cmap, "colors"):
                pools.extend(list(cmap.colors))  # type: ignore[attr-defined]
            else:
                pools.extend([cmap(i) for i in np.linspace(0, 1, 12)])
        except Exception:
            continue

    uniq: List[tuple] = []
    seen = set()
    for c in pools:
        key = tuple(np.round(np.array(c), 6))
        if key not in seen:
            seen.add(key)
            uniq.append(tuple(c))

    if len(uniq) >= n:
        return uniq[:n]

    cmap = plt.get_cmap("turbo")
    return [cmap(i) for i in np.linspace(0, 1, n)]


# ──────────────────────────────────────────────────────────────────────────────
# Title utilities
# ──────────────────────────────────────────────────────────────────────────────
def _soft_wrap_title(title: Optional[str], parts_per_line: int = 2) -> Optional[str]:
    """
    If title has no newline and uses " | " separators, split into multiple lines.
    Example with parts_per_line=2:
      "A | B | C | D" -> "A | B\nC | D"
    """
    if not title:
        return title
    if "\n" in title:
        return title
    if " | " not in title:
        return title

    parts = title.split(" | ")
    if len(parts) <= parts_per_line:
        return title

    lines = []
    for i in range(0, len(parts), parts_per_line):
        lines.append(" | ".join(parts[i : i + parts_per_line]))
    return "\n".join(lines)


def _append_param_line(
    title: Optional[str],
    *,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    min_transitions: Optional[int] = None,
) -> Optional[str]:
    """
    Append a new line like:
      window: pre≤40d, post≤40d | min_transitions=50
    Only includes items that are not None.
    """
    items: List[str] = []

    if (max_days_pre is not None) or (max_days_post is not None):
        pre_s = "NA" if max_days_pre is None else str(int(max_days_pre))
        post_s = "NA" if max_days_post is None else str(int(max_days_post))
        items.append(f"window: pre≤{pre_s}d, post≤{post_s}d")

    if min_transitions is not None:
        items.append(f"min_transitions={int(min_transitions)}")

    if not items:
        return title

    param_line = " | ".join(items)
    if not title:
        return param_line
    return f"{title}\n{param_line}"


# ──────────────────────────────────────────────────────────────────────────────
# Entropy core
# ──────────────────────────────────────────────────────────────────────────────
def _entropy_from_probs(probs: np.ndarray, log_base: float = 2.0) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return float("nan")

    if log_base == 2.0:
        logs = np.log2(probs)
    elif log_base == np.e:
        logs = np.log(probs)
    else:
        logs = np.log(probs) / np.log(log_base)

    return float(-(probs * logs).sum())


def compute_H_a_from_counts(
    follower_counts: Dict[str, int],
    log_base: float = 2.0,
) -> Tuple[float, Dict[str, float], Optional[float], int]:
    """
    follower_counts maps next_syllable -> count, conditioned on current syllable a.
    Returns (H, prob_map, max_H_over_observed_followers, n_transitions).
    """
    n = int(sum(follower_counts.values()))
    if n == 0:
        return float("nan"), {}, None, 0

    followers = sorted(follower_counts.keys())
    counts = np.array([follower_counts[f] for f in followers], dtype=float)
    probs = counts / counts.sum()

    H = _entropy_from_probs(probs, log_base=log_base)
    prob_map = {f: float(p) for f, p in zip(followers, probs)}

    c = len(followers)
    if c == 0:
        max_H = None
    else:
        if log_base == 2.0:
            max_H = float(np.log2(c))
        elif log_base == np.e:
            max_H = float(np.log(c))
        else:
            max_H = float(np.log(c) / np.log(log_base))

    return H, prob_map, max_H, n


# ──────────────────────────────────────────────────────────────────────────────
# Total transition entropy TE (weighted average of H_a)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TotalEntropyResult:
    group: str
    total_entropy: float
    log_base: float
    n_total_transitions: int
    n_unique_preceding_syllables: int
    P_of_a: Dict[str, float]              # P(a) over preceding syllables
    H_of_a: Dict[str, float]              # H_a for each a (nan if none)
    TE_terms: Dict[str, float]            # P(a) * H_a (nan skipped)
    notes: Optional[str] = None


def compute_total_transition_entropy_from_prev_counts(
    prev_to_next_counts: Dict[str, Dict[str, int]],
    *,
    log_base: float = 2.0,
) -> TotalEntropyResult:
    totals_by_a: Dict[str, int] = {a: int(sum(d.values())) for a, d in prev_to_next_counts.items()}
    n_total = int(sum(totals_by_a.values()))

    if n_total == 0:
        return TotalEntropyResult(
            group="unknown",
            total_entropy=float("nan"),
            log_base=log_base,
            n_total_transitions=0,
            n_unique_preceding_syllables=len(prev_to_next_counts),
            P_of_a={},
            H_of_a={},
            TE_terms={},
            notes="No transitions available (n_total_transitions=0).",
        )

    P_of_a: Dict[str, float] = {a: totals_by_a[a] / n_total for a in totals_by_a}
    H_of_a: Dict[str, float] = {}
    TE_terms: Dict[str, float] = {}

    TE = 0.0
    for a, next_counts in prev_to_next_counts.items():
        Ha, _probs, _maxH, _n_a = compute_H_a_from_counts(next_counts, log_base=log_base)
        H_of_a[a] = float(Ha)
        if np.isfinite(Ha) and a in P_of_a:
            term = float(P_of_a[a] * Ha)
            TE_terms[a] = term
            TE += term
        else:
            TE_terms[a] = float("nan")

    return TotalEntropyResult(
        group="unknown",
        total_entropy=float(TE),
        log_base=log_base,
        n_total_transitions=n_total,
        n_unique_preceding_syllables=len(prev_to_next_counts),
        P_of_a=P_of_a,
        H_of_a=H_of_a,
        TE_terms=TE_terms,
        notes=None,
    )


def compute_pre_post_total_transition_entropy(
    organized_df: pd.DataFrame,
    *,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    log_base: float = 2.0,
    order_col: str = "syllable_order",
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
) -> Tuple[TotalEntropyResult, TotalEntropyResult]:
    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    pre_counts_all = _bigram_counts_by_prev(pre_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)
    post_counts_all = _bigram_counts_by_prev(post_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)

    pre_TE = compute_total_transition_entropy_from_prev_counts(pre_counts_all, log_base=log_base)
    post_TE = compute_total_transition_entropy_from_prev_counts(post_counts_all, log_base=log_base)

    pre_TE.group = "pre"
    post_TE.group = "post"
    return pre_TE, post_TE


# ──────────────────────────────────────────────────────────────────────────────
# Treatment date: arg OR Excel lookup
# ──────────────────────────────────────────────────────────────────────────────
def infer_animal_id_from_merged_df(df: pd.DataFrame) -> Optional[str]:
    if "Animal ID" in df.columns:
        vals = df["Animal ID"].dropna().astype(str)
        if not vals.empty:
            uniq = vals.unique()
            if len(uniq) == 1:
                return uniq[0]
            return vals.mode().iloc[0]

    if "file_name" in df.columns:
        fn = df["file_name"].dropna().astype(str)
        if not fn.empty:
            prefixes = fn.apply(lambda s: s.split("_")[0])
            uniq = prefixes.unique()
            if len(uniq) == 1:
                return uniq[0]
            return prefixes.mode().iloc[0]

    return None


def _find_col_case_insensitive(df: pd.DataFrame, wanted: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for w in wanted:
        if w.lower() in lower_map:
            return lower_map[w.lower()]
    return None


def lookup_treatment_date_from_xlsx(
    metadata_xlsx: Union[str, Path],
    animal_id: str,
    sheet_name: Optional[str] = None,
) -> pd.Timestamp:
    metadata_xlsx = Path(metadata_xlsx)
    if not metadata_xlsx.exists():
        raise FileNotFoundError(metadata_xlsx)

    xls = pd.ExcelFile(metadata_xlsx)

    preferred: List[str] = []
    if sheet_name:
        preferred.append(sheet_name)

    for s in ["metadata_with_hit_type", "metadata", "Metadata", "Sheet1"]:
        if s in xls.sheet_names and s not in preferred:
            preferred.append(s)

    for s in xls.sheet_names:
        if s not in preferred:
            preferred.append(s)

    last_reason = None

    for sname in preferred:
        df = pd.read_excel(metadata_xlsx, sheet_name=sname)

        animal_col = _find_col_case_insensitive(df, ["Animal ID", "animal_id", "AnimalID"])
        date_col = _find_col_case_insensitive(
            df, ["Treatment date", "Treatment Date", "treatment_date", "Treatment_date"]
        )

        if not animal_col or not date_col:
            last_reason = f"Sheet '{sname}' missing required columns (Animal ID / Treatment date)."
            continue

        sub = df[df[animal_col].astype(str) == str(animal_id)].copy()
        if sub.empty:
            last_reason = f"Sheet '{sname}' has no rows for animal_id={animal_id}."
            continue

        dates = pd.to_datetime(sub[date_col], errors="coerce").dropna().dt.normalize().unique()
        if len(dates) == 0:
            last_reason = f"Sheet '{sname}' rows found but treatment dates could not be parsed."
            continue

        dates_sorted = sorted(pd.to_datetime(dates))
        if len(dates_sorted) > 1:
            print(
                f"[metadata lookup] Multiple treatment dates found for {animal_id} in sheet '{sname}': "
                f"{[d.date() for d in dates_sorted]}. Using earliest."
            )
        return pd.to_datetime(dates_sorted[0]).normalize()

    raise ValueError(
        f"Could not find a parsable treatment date for animal_id='{animal_id}' in {metadata_xlsx}. "
        f"Searched sheets: {preferred}. Last reason: {last_reason}"
    )


def resolve_treatment_date(
    *,
    treatment_date_arg: Optional[str],
    metadata_xlsx: Optional[Union[str, Path]],
    animal_id_arg: Optional[str],
    merged_df: pd.DataFrame,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.Timestamp, str, str]:
    if treatment_date_arg:
        t = pd.to_datetime(treatment_date_arg, errors="raise").normalize()
        animal_id_used = animal_id_arg or infer_animal_id_from_merged_df(merged_df) or "UNKNOWN"
        return t, animal_id_used, "arg"

    if not metadata_xlsx:
        raise ValueError("Provide either treatment_date_arg OR metadata_xlsx (Excel lookup).")

    animal_id_used = animal_id_arg or infer_animal_id_from_merged_df(merged_df)
    if not animal_id_used:
        raise ValueError("Could not infer animal_id from merged data. Please pass animal_id_arg explicitly.")

    t = lookup_treatment_date_from_xlsx(metadata_xlsx, animal_id_used, sheet_name=sheet_name)
    return t, animal_id_used, "metadata_xlsx"


# ──────────────────────────────────────────────────────────────────────────────
# Merge + organize-from-merged
# ──────────────────────────────────────────────────────────────────────────────
def build_merged_annotations_df(
    decoded_database_json: Union[str, Path],
    song_detection_json: Union[str, Path],
    *,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = False,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
) -> pd.DataFrame:
    res = build_decoded_with_split_labels(
        decoded_database_json=decoded_database_json,
        song_detection_json=song_detection_json,
        only_song_present=True,
        compute_durations=True,
        add_recording_datetime=True,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=max_gap_between_song_segments,
        segment_index_offset=segment_index_offset,
        merge_repeated_syllables=merge_repeated_syllables,
        repeat_gap_ms=repeat_gap_ms,
        repeat_gap_inclusive=repeat_gap_inclusive,
    )

    df = res.annotations_appended_df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "Recording DateTime" in df.columns:
        df["Date"] = pd.to_datetime(df["Recording DateTime"], errors="coerce").dt.normalize()
    else:
        raise KeyError("Merged annotations DF has neither 'Date' nor 'Recording DateTime'.")

    return df


def build_organized_from_merged_df(
    merged_df: pd.DataFrame,
    *,
    compute_durations: bool = True,
    add_recording_datetime: bool = True,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
) -> OrganizedDataset:
    df = merged_df.copy()

    if "syllable_onsets_offsets_ms_dict" not in df.columns:
        if "syllable_onsets_offsets_ms" in df.columns:
            df["syllable_onsets_offsets_ms_dict"] = df["syllable_onsets_offsets_ms"].apply(parse_json_safe)
        else:
            df["syllable_onsets_offsets_ms_dict"] = [{} for _ in range(len(df))]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "Recording DateTime" in df.columns:
        df["Date"] = pd.to_datetime(df["Recording DateTime"], errors="coerce").dt.normalize()
    else:
        df["Date"] = pd.NaT

    if add_recording_datetime and "Recording DateTime" in df.columns:
        df["Recording DateTime"] = pd.to_datetime(df["Recording DateTime"], errors="coerce")

    label_set: set[str] = set()
    for d in df["syllable_onsets_offsets_ms_dict"]:
        if isinstance(d, dict) and d:
            label_set.update([str(k) for k in d.keys()])
    unique_labels = _sorted_labels_natural(label_set)

    df["syllables_present"] = df["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: _sorted_labels_natural([str(k) for k in d.keys()]) if isinstance(d, dict) else []
    )
    df["syllable_order"] = df["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: extract_syllable_order(d, onset_index=0)
    )

    if compute_durations and unique_labels:
        for lab in unique_labels:
            df[f"syllable_{lab}_durations"] = df["syllable_onsets_offsets_ms_dict"].apply(
                lambda d, L=lab: calculate_syllable_durations_ms(d, L)
            )

    unique_dates = (
        pd.to_datetime(df["Date"], errors="coerce")
        .dt.strftime("%Y.%m.%d")
        .dropna()
        .unique()
        .tolist()
    )
    unique_dates.sort()

    treatment_fmt = None
    if treatment_date is not None:
        try:
            treatment_fmt = pd.to_datetime(treatment_date).strftime("%Y.%m.%d")
        except Exception:
            treatment_fmt = None

    return OrganizedDataset(
        organized_df=df,
        unique_dates=unique_dates,
        unique_syllable_labels=unique_labels,
        treatment_date=treatment_fmt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pre/Post split + sequences (supports ± day window)
# ──────────────────────────────────────────────────────────────────────────────
def split_pre_post(
    df: pd.DataFrame,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
    *,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(treatment_date).normalize()

    if "Date" not in df.columns:
        raise KeyError("Expected column 'Date' in organized_df.")

    dfx = df.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce").dt.normalize()
    dfx = dfx[dfx["Date"].notna()].copy()

    if include_treatment_day:
        pre = dfx[dfx["Date"] <= t].copy()
        post = dfx[dfx["Date"] >= t].copy()
    else:
        pre = dfx[dfx["Date"] < t].copy()
        post = dfx[dfx["Date"] > t].copy()

    if max_days_pre is not None:
        pre_start = t - pd.Timedelta(days=int(max_days_pre))
        pre = pre[pre["Date"] >= pre_start].copy()

    if max_days_post is not None:
        post_end = t + pd.Timedelta(days=int(max_days_post))
        post = post[post["Date"] <= post_end].copy()

    return pre, post


def get_syllable_order_sequences(df: pd.DataFrame, col: str = "syllable_order") -> List[List[str]]:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}'. Found: {list(df.columns)}")
    seqs: List[List[str]] = []
    for v in df[col].tolist():
        seq = _coerce_listlike(v)
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


# ──────────────────────────────────────────────────────────────────────────────
# H_a for ONE syllable
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EntropyResult:
    group: str
    target: str
    entropy: float
    n_transitions: int
    distinct_followers: int
    follower_counts: Dict[str, int]
    follower_probs: Dict[str, float]
    max_entropy_if_uniform_over_observed_followers: Optional[float]


def _follower_counts_for_target(
    sequences: Sequence[Sequence[str]],
    target: str,
    *,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    ignore = set(ignore_labels or [])
    counts: Dict[str, int] = {}
    for seq in sequences:
        labs = [str(x) for x in seq if str(x) not in ignore]
        if collapse_repeats:
            labs = _collapse_consecutive(labs)
        for cur, nxt in zip(labs[:-1], labs[1:]):
            if cur == target:
                counts[nxt] = counts.get(nxt, 0) + 1
    return counts


def _predecessor_counts_for_target(
    sequences: Sequence[Sequence[str]],
    target: str,
    *,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Counts of prev syllable given current == target."""
    ignore = set(ignore_labels or [])
    counts: Dict[str, int] = {}
    for seq in sequences:
        labs = [str(x) for x in seq if str(x) not in ignore]
        if collapse_repeats:
            labs = _collapse_consecutive(labs)
        for prev, cur in zip(labs[:-1], labs[1:]):
            if cur == target:
                counts[prev] = counts.get(prev, 0) + 1
    return counts


def compute_pre_post_Ha(
    organized_df: pd.DataFrame,
    *,
    target: str,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    log_base: float = 2.0,
    order_col: str = "syllable_order",
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
) -> Tuple[EntropyResult, EntropyResult]:
    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    pre_counts = _follower_counts_for_target(
        pre_seqs, target=target, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels
    )
    post_counts = _follower_counts_for_target(
        post_seqs, target=target, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels
    )

    pre_H, pre_probs, pre_maxH, pre_n = compute_H_a_from_counts(pre_counts, log_base=log_base)
    post_H, post_probs, post_maxH, post_n = compute_H_a_from_counts(post_counts, log_base=log_base)

    pre_res = EntropyResult(
        group="pre",
        target=target,
        entropy=pre_H,
        n_transitions=pre_n,
        distinct_followers=len(pre_probs),
        follower_counts=pre_counts,
        follower_probs=pre_probs,
        max_entropy_if_uniform_over_observed_followers=pre_maxH,
    )
    post_res = EntropyResult(
        group="post",
        target=target,
        entropy=post_H,
        n_transitions=post_n,
        distinct_followers=len(post_probs),
        follower_counts=post_counts,
        follower_probs=post_probs,
        max_entropy_if_uniform_over_observed_followers=post_maxH,
    )
    return pre_res, post_res


# ──────────────────────────────────────────────────────────────────────────────
# ALL syllables bigram counts (needed for Ha_all and TE)
# ──────────────────────────────────────────────────────────────────────────────
def _bigram_counts_by_prev(
    sequences: Sequence[Sequence[str]],
    *,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int]]:
    ignore = set(ignore_labels or [])
    out: Dict[str, Dict[str, int]] = {}

    for seq in sequences:
        labs = [str(x) for x in seq if str(x) not in ignore]
        if collapse_repeats:
            labs = _collapse_consecutive(labs)
        if len(labs) < 2:
            continue

        for a, b in zip(labs[:-1], labs[1:]):
            if a in ignore or b in ignore:
                continue
            row = out.setdefault(a, {})
            row[b] = row.get(b, 0) + 1

    return out


def compute_pre_post_Ha_all_syllables(
    organized_df: pd.DataFrame,
    *,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    log_base: float = 2.0,
    order_col: str = "syllable_order",
    min_transitions: int = 1,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
) -> pd.DataFrame:
    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    pre_counts_all = _bigram_counts_by_prev(pre_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)
    post_counts_all = _bigram_counts_by_prev(post_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)

    targets = set(pre_counts_all.keys()) | set(post_counts_all.keys())

    if "syllables_present" in organized_df.columns:
        for v in organized_df["syllables_present"].tolist():
            for lab in _coerce_listlike(v):
                targets.add(str(lab))

    ignore = set(ignore_labels or [])
    targets = {t for t in targets if t not in ignore}
    targets_sorted = _sorted_labels_natural(list(targets))

    rows: List[Dict[str, object]] = []
    for a in targets_sorted:
        pre_counts = pre_counts_all.get(a, {})
        post_counts = post_counts_all.get(a, {})

        Hpre, pre_probs, pre_maxH, npre = compute_H_a_from_counts(pre_counts, log_base=log_base)
        Hpost, post_probs, post_maxH, npost = compute_H_a_from_counts(post_counts, log_base=log_base)

        if (npre < min_transitions) and (npost < min_transitions):
            continue

        rows.append(
            dict(
                syllable=a,
                H_pre=Hpre,
                H_post=Hpost,
                delta=(Hpost - Hpre) if (np.isfinite(Hpre) and np.isfinite(Hpost)) else np.nan,
                n_pre=int(npre),
                n_post=int(npost),
                followers_pre=int(len(pre_probs)),
                followers_post=int(len(post_probs)),
                maxH_pre=pre_maxH,
                maxH_post=post_maxH,
            )
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    return out_df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Paired tests across syllables
# ──────────────────────────────────────────────────────────────────────────────
def _paired_diffs_from_ha_df(ha_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = pd.to_numeric(ha_df["H_pre"], errors="coerce").to_numpy(dtype=float)
    post = pd.to_numeric(ha_df["H_post"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(pre) & np.isfinite(post)
    pre2 = pre[ok]
    post2 = post[ok]
    diffs = post2 - pre2
    return pre2, post2, diffs


def _signflip_permutation_pvalue(
    diffs: np.ndarray,
    *,
    n_perm: int = 20000,
    seed: int = 0,
    two_sided: bool = True,
) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))

    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size), replace=True)
    perm_means = (signs * diffs[None, :]).mean(axis=1)

    if two_sided:
        p = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1))
    else:
        p = float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))
    return p


def paired_entropy_tests(
    ha_df: pd.DataFrame,
    *,
    n_perm: int = 20000,
    seed: int = 0,
) -> Dict[str, object]:
    pre, post, diffs = _paired_diffs_from_ha_df(ha_df)

    out: Dict[str, object] = {
        "n_paired_syllables": int(diffs.size),
        "mean_pre": float(np.mean(pre)) if pre.size else float("nan"),
        "mean_post": float(np.mean(post)) if post.size else float("nan"),
        "mean_diff_post_minus_pre": float(np.mean(diffs)) if diffs.size else float("nan"),
        "median_diff_post_minus_pre": float(np.median(diffs)) if diffs.size else float("nan"),
    }

    if diffs.size < 2:
        out["note"] = "Not enough paired syllables with finite pre+post entropy to test."
        return out

    if _HAVE_SCIPY:
        try:
            tt = _scipy_stats.ttest_rel(post, pre, nan_policy="omit")
            out["ttest_rel_p_two_sided"] = float(tt.pvalue)
            out["ttest_rel_t"] = float(tt.statistic)
        except Exception as e:
            out["ttest_rel_error"] = str(e)

        try:
            if np.allclose(diffs, 0):
                out["wilcoxon_p_two_sided"] = 1.0
                out["wilcoxon_stat"] = 0.0
                out["wilcoxon_note"] = "All paired differences are ~0."
            else:
                w = _scipy_stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                out["wilcoxon_p_two_sided"] = float(w.pvalue)
                out["wilcoxon_stat"] = float(w.statistic)
        except Exception as e:
            out["wilcoxon_error"] = str(e)

    out["signflip_perm_p_two_sided"] = float(
        _signflip_permutation_pvalue(diffs, n_perm=n_perm, seed=seed, two_sided=True)
    )

    return out


def _format_p(p: object) -> str:
    try:
        pv = float(p)  # type: ignore
    except Exception:
        return "NA"
    if not np.isfinite(pv):
        return "NA"
    if pv < 1e-4:
        return f"{pv:.1e}"
    return f"{pv:.4f}"


def format_test_summary(stats: Dict[str, object]) -> str:
    n = stats.get("n_paired_syllables", "NA")
    p_perm = _format_p(stats.get("signflip_perm_p_two_sided", np.nan))
    parts = [f"n={n}", f"perm p={p_perm}"]
    if "wilcoxon_p_two_sided" in stats:
        parts.append(f"wilc p={_format_p(stats.get('wilcoxon_p_two_sided'))}")
    if "ttest_rel_p_two_sided" in stats:
        parts.append(f"t p={_format_p(stats.get('ttest_rel_p_two_sided'))}")
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Plots: connected pre/post boxplot (all syllables or subset)
# ──────────────────────────────────────────────────────────────────────────────
def plot_Ha_boxplot_pre_post_connected(
    ha_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    show: bool = False,
    overlay_points: bool = True,
    connect_pairs: bool = True,
    annotate_stats: bool = True,
    stats_n_perm: int = 20000,
    stats_seed: int = 0,
    # parameter clarity in title
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    min_transitions: Optional[int] = None,
    wrap_title_parts_per_line: int = 2,
) -> Dict[str, object]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ha_df is None or ha_df.empty:
        raise ValueError("ha_df is empty; nothing to plot.")

    pre_all = pd.to_numeric(ha_df["H_pre"], errors="coerce").to_numpy(dtype=float)
    post_all = pd.to_numeric(ha_df["H_post"], errors="coerce").to_numpy(dtype=float)
    pre_all = pre_all[np.isfinite(pre_all)]
    post_all = post_all[np.isfinite(post_all)]

    if pre_all.size == 0 and post_all.size == 0:
        raise ValueError("No finite H_pre or H_post values to plot.")

    pre_p, post_p, _diffs = _paired_diffs_from_ha_df(ha_df)
    stats = paired_entropy_tests(ha_df, n_perm=stats_n_perm, seed=stats_seed)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.boxplot([pre_all, post_all], labels=["Pre", "Post"], showfliers=True, whis=1.5)

    if overlay_points:
        rng = np.random.default_rng(stats_seed)
        jitter = rng.normal(loc=0.0, scale=0.05, size=pre_p.size) if pre_p.size else np.array([])
        x1 = 1.0 + jitter
        x2 = 2.0 + jitter

        if connect_pairs and pre_p.size:
            for i in range(pre_p.size):
                if np.isfinite(pre_p[i]) and np.isfinite(post_p[i]):
                    ax.plot([x1[i], x2[i]], [pre_p[i], post_p[i]], linewidth=1, alpha=0.5)

        if pre_p.size:
            ax.scatter(x1, pre_p, s=20, alpha=0.85, color="C0", label="Pre")
        if post_p.size:
            ax.scatter(x2, post_p, s=20, alpha=0.85, color="C1", label="Post")

        ax.legend(frameon=False, loc="best")

    ax.set_ylabel("Transition entropy Hₐ (bits)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_stats:
        ax.text(0.02, 0.98, format_test_summary(stats), transform=ax.transAxes, va="top", ha="left", fontsize=9)

    title2 = _append_param_line(
        _soft_wrap_title(title, parts_per_line=wrap_title_parts_per_line),
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
        min_transitions=min_transitions,
    )
    if title2:
        fig.suptitle(title2, y=0.98, fontsize=10)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats


def plot_Ha_boxplot_pre_post_by_label(
    ha_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    show: bool = False,
    connect_pairs: bool = True,

    annotate_stats: bool = True,
    stats_n_perm: int = 20000,
    stats_seed: int = 0,

    legend_max_labels: int = 60,
    legend_fontsize: float = 7.0,
    legend_title_fontsize: float = 9.0,
    legend_ncol: int = 2,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),

    show_usage_panel: bool = True,
    connect_usage_pairs: bool = True,
    usage_log_scale: bool = True,

    # parameter clarity in title
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    min_transitions: Optional[int] = None,
    wrap_title_parts_per_line: int = 2,
) -> Dict[str, object]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ha_df is None or ha_df.empty:
        raise ValueError("ha_df is empty; nothing to plot.")

    df = ha_df.copy()
    df["syllable"] = df["syllable"].astype(str)
    df["H_pre"] = pd.to_numeric(df["H_pre"], errors="coerce")
    df["H_post"] = pd.to_numeric(df["H_post"], errors="coerce")
    df["n_pre"] = pd.to_numeric(df.get("n_pre", np.nan), errors="coerce")
    df["n_post"] = pd.to_numeric(df.get("n_post", np.nan), errors="coerce")

    pre_all = df["H_pre"].to_numpy(dtype=float)
    post_all = df["H_post"].to_numpy(dtype=float)
    pre_all = pre_all[np.isfinite(pre_all)]
    post_all = post_all[np.isfinite(post_all)]
    if pre_all.size == 0 and post_all.size == 0:
        raise ValueError("No finite H_pre or H_post values to plot.")

    df_paired_H = df[np.isfinite(df["H_pre"]) & np.isfinite(df["H_post"])].copy()
    uniq = _sorted_labels_natural(df_paired_H["syllable"].tolist())
    if len(uniq) == 0:
        raise ValueError("No paired syllables with finite H_pre and H_post to plot.")

    colors = _get_distinct_qual_colors(max(len(uniq), 30))
    color_map: Dict[str, tuple] = {lab: colors[i] for i, lab in enumerate(uniq)}

    stats = paired_entropy_tests(df, n_perm=stats_n_perm, seed=stats_seed)

    if show_usage_panel:
        fig, (axH, axN) = plt.subplots(
            1, 2, figsize=(9.8, 5.4), gridspec_kw={"width_ratios": [1.0, 1.0]}
        )
    else:
        fig, axH = plt.subplots(1, 1, figsize=(6.8, 5.4))
        axN = None

    rng = np.random.default_rng(stats_seed)

    # Entropy panel
    axH.boxplot([pre_all, post_all], labels=["Pre", "Post"], showfliers=True, whis=1.5)

    jitter = rng.normal(loc=0.0, scale=0.05, size=len(df_paired_H)) if len(df_paired_H) else np.array([])
    x1 = 1.0 + jitter
    x2 = 2.0 + jitter

    for i, (_, row) in enumerate(df_paired_H.iterrows()):
        lab = row["syllable"]
        c = color_map.get(lab, (0, 0, 0, 1))
        y1 = float(row["H_pre"])
        y2 = float(row["H_post"])

        if connect_pairs:
            axH.plot([x1[i], x2[i]], [y1, y2], linewidth=1, alpha=0.65, color=c)

        axH.scatter([x1[i]], [y1], s=26, alpha=0.92, color=c)
        axH.scatter([x2[i]], [y2], s=26, alpha=0.92, color=c)

    axH.set_ylabel("Transition entropy Hₐ (bits)")
    axH.spines["top"].set_visible(False)
    axH.spines["right"].set_visible(False)

    if annotate_stats:
        axH.text(0.02, 0.98, format_test_summary(stats), transform=axH.transAxes, va="top", ha="left", fontsize=9)

    # Usage panel
    if show_usage_panel and axN is not None:
        npre_all = df["n_pre"].to_numpy(dtype=float)
        npost_all = df["n_post"].to_numpy(dtype=float)
        npre_all = npre_all[np.isfinite(npre_all)]
        npost_all = npost_all[np.isfinite(npost_all)]

        if npre_all.size == 0 and npost_all.size == 0:
            axN.text(0.5, 0.5, "No n_pre/n_post available", ha="center", va="center")
        else:
            axN.boxplot([npre_all, npost_all], labels=["Pre", "Post"], showfliers=True, whis=1.5)

            df_counts = df[df["syllable"].isin(uniq)].copy().reset_index(drop=True)
            jitterN = rng.normal(loc=0.0, scale=0.05, size=len(df_counts)) if len(df_counts) else np.array([])
            xn1 = 1.0 + jitterN
            xn2 = 2.0 + jitterN

            for i, row in df_counts.iterrows():
                lab = row["syllable"]
                c = color_map.get(lab, (0, 0, 0, 1))
                npre = row["n_pre"]
                npost = row["n_post"]

                if np.isfinite(npre):
                    axN.scatter([xn1[i]], [float(npre)], s=26, alpha=0.92, color=c)
                if np.isfinite(npost):
                    axN.scatter([xn2[i]], [float(npost)], s=26, alpha=0.92, color=c)
                if connect_usage_pairs and np.isfinite(npre) and np.isfinite(npost):
                    axN.plot([xn1[i], xn2[i]], [float(npre), float(npost)], linewidth=1, alpha=0.65, color=c)

            axN.set_ylabel("Usage (nₐ transitions)")
            axN.set_title("Syllable usage (nₐ)", pad=12)
            axN.spines["top"].set_visible(False)
            axN.spines["right"].set_visible(False)

            if usage_log_scale:
                all_counts = []
                if npre_all.size:
                    all_counts.append(npre_all)
                if npost_all.size:
                    all_counts.append(npost_all)
                if all_counts:
                    mn = float(np.min(np.concatenate(all_counts)))
                    if mn > 0:
                        axN.set_yscale("log")
                    else:
                        axN.text(
                            0.02, 0.02, "log-scale skipped (counts include 0)",
                            transform=axN.transAxes, va="bottom", ha="left", fontsize=8
                        )

    # Figure legend
    if len(uniq) > 0:
        if len(uniq) > legend_max_labels:
            axH.text(
                0.02, 0.02,
                f"Legend omitted: {len(uniq)} syllables > legend_max_labels={legend_max_labels}",
                transform=axH.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
            )
        else:
            n_lookup = df.set_index("syllable")[["n_pre", "n_post"]]
            handles = []
            labels = []
            for lab in uniq:
                c = color_map[lab]
                npre = n_lookup.loc[lab, "n_pre"] if lab in n_lookup.index else np.nan
                npost = n_lookup.loc[lab, "n_post"] if lab in n_lookup.index else np.nan
                npre_s = "NA" if not np.isfinite(float(npre)) else str(int(float(npre)))
                npost_s = "NA" if not np.isfinite(float(npost)) else str(int(float(npost)))
                handles.append(plt.Line2D([0], [0], marker="o", linestyle="None", color=c, markersize=6))
                labels.append(f"{lab} (n_pre={npre_s}, n_post={npost_s})")

            fig.legend(
                handles,
                labels,
                frameon=False,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                title="Syllable label",
                ncol=int(max(1, legend_ncol)),
                fontsize=float(legend_fontsize),
                title_fontsize=float(legend_title_fontsize),
                borderaxespad=0.0,
                handletextpad=0.4,
                columnspacing=0.8,
            )

    title2 = _append_param_line(
        _soft_wrap_title(title, parts_per_line=wrap_title_parts_per_line),
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
        min_transitions=min_transitions,
    )
    if title2:
        fig.suptitle(title2, y=0.985, x=0.39, fontsize=10)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 0.90])
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Followers / Predecessors plots for ONE target syllable
# ──────────────────────────────────────────────────────────────────────────────
def plot_followers_pre_post(
    pre: EntropyResult,
    post: EntropyResult,
    out_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """Bar plot for P(next | target) pre vs post."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(pre.follower_probs.keys()) | set(post.follower_probs.keys()))
    pre_vals = [pre.follower_probs.get(k, 0.0) for k in keys]
    post_vals = [post.follower_probs.get(k, 0.0) for k in keys]

    x = np.arange(len(keys))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(keys)), 4.2))
    ax.bar(x - width / 2, pre_vals, width, label=f"Pre (H={pre.entropy:.2f}, n={pre.n_transitions})")
    ax.bar(x + width / 2, post_vals, width, label=f"Post (H={post.entropy:.2f}, n={post.n_transitions})")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("P(next | target)")
    ax.set_xlabel(f"Next syllable (given {pre.target})")
    ax.legend(loc="best", frameon=False)

    if title:
        ax.set_title(_soft_wrap_title(title, parts_per_line=2) or "")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_predecessors_pre_post(
    organized_df: pd.DataFrame,
    *,
    target: str,
    treatment_date: Union[str, pd.Timestamp],
    out_path: Union[str, Path],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    title: Optional[str] = None,
) -> None:
    """Bar plot for P(prev | target) pre vs post."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )
    pre_seqs = get_syllable_order_sequences(pre_df) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df) if not post_df.empty else []

    pre_counts = _predecessor_counts_for_target(pre_seqs, target=target, collapse_repeats=collapse_repeats,
                                               ignore_labels=ignore_labels)
    post_counts = _predecessor_counts_for_target(post_seqs, target=target, collapse_repeats=collapse_repeats,
                                                ignore_labels=ignore_labels)

    def _to_probs(d: Dict[str, int]) -> Dict[str, float]:
        n = float(sum(d.values()))
        if n <= 0:
            return {}
        return {k: v / n for k, v in d.items()}

    pre_probs = _to_probs(pre_counts)
    post_probs = _to_probs(post_counts)

    keys = sorted(set(pre_probs.keys()) | set(post_probs.keys()))
    pre_vals = [pre_probs.get(k, 0.0) for k in keys]
    post_vals = [post_probs.get(k, 0.0) for k in keys]

    x = np.arange(len(keys))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(keys)), 4.2))
    ax.bar(x - width / 2, pre_vals, width, label=f"Pre (n={sum(pre_counts.values())})")
    ax.bar(x + width / 2, post_vals, width, label=f"Post (n={sum(post_counts.values())})")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("P(prev | target)")
    ax.set_xlabel(f"Previous syllable (given {target})")
    ax.legend(loc="best", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(_soft_wrap_title(title, parts_per_line=2) or "")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_neighbors_hist_pre_post(
    organized_df: pd.DataFrame,
    *,
    target: str,
    treatment_date: Union[str, pd.Timestamp],
    out_path: Union[str, Path],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    top_k: Optional[int] = 12,
    title: Optional[str] = None,
) -> None:
    """
    One figure with TWO panels:
      Left:  P(prev | target)  (most common predecessors)
      Right: P(next | target)  (most common followers)

    If top_k is not None, show the top_k categories by total count (pre+post).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )
    pre_seqs = get_syllable_order_sequences(pre_df) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df) if not post_df.empty else []

    pre_prev = _predecessor_counts_for_target(pre_seqs, target=target, collapse_repeats=collapse_repeats,
                                              ignore_labels=ignore_labels)
    post_prev = _predecessor_counts_for_target(post_seqs, target=target, collapse_repeats=collapse_repeats,
                                               ignore_labels=ignore_labels)

    pre_next = _follower_counts_for_target(pre_seqs, target=target, collapse_repeats=collapse_repeats,
                                          ignore_labels=ignore_labels)
    post_next = _follower_counts_for_target(post_seqs, target=target, collapse_repeats=collapse_repeats,
                                           ignore_labels=ignore_labels)

    def _top_keys(pre_d: Dict[str, int], post_d: Dict[str, int]) -> List[str]:
        keys = sorted(set(pre_d) | set(post_d))
        totals = {k: int(pre_d.get(k, 0) + post_d.get(k, 0)) for k in keys}
        keys_sorted = sorted(keys, key=lambda k: totals[k], reverse=True)
        if top_k is not None:
            keys_sorted = keys_sorted[: int(top_k)]
        return keys_sorted

    prev_keys = _top_keys(pre_prev, post_prev)
    next_keys = _top_keys(pre_next, post_next)

    def _probs(counts: Dict[str, int], keys: List[str]) -> List[float]:
        n = float(sum(counts.values()))
        if n <= 0:
            return [0.0 for _ in keys]
        return [float(counts.get(k, 0)) / n for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 0.5 * (len(prev_keys) + len(next_keys))), 4.2))

    # Predecessors
    x1 = np.arange(len(prev_keys))
    w = 0.42
    ax1.bar(x1 - w / 2, _probs(pre_prev, prev_keys), w, label=f"Pre (n={sum(pre_prev.values())})")
    ax1.bar(x1 + w / 2, _probs(post_prev, prev_keys), w, label=f"Post (n={sum(post_prev.values())})")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(prev_keys, rotation=45, ha="right")
    ax1.set_ylabel("P(prev | target)")
    ax1.set_title(f"Predecessors of {target}")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Followers
    x2 = np.arange(len(next_keys))
    ax2.bar(x2 - w / 2, _probs(pre_next, next_keys), w, label=f"Pre (n={sum(pre_next.values())})")
    ax2.bar(x2 + w / 2, _probs(post_next, next_keys), w, label=f"Post (n={sum(post_next.values())})")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(next_keys, rotation=45, ha="right")
    ax2.set_ylabel("P(next | target)")
    ax2.set_title(f"Followers of {target}")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.legend(loc="best", frameon=False)

    if title:
        fig.suptitle(_soft_wrap_title(title, parts_per_line=2) or "", y=0.99, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_neighbors_hist_for_all_targets(
    organized_df: pd.DataFrame,
    *,
    treatment_date: Union[str, pd.Timestamp],
    out_dir: Union[str, Path],
    targets: Optional[Sequence[str]] = None,
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    top_k: Optional[int] = 12,
) -> List[Path]:
    """
    Batch-save neighbor histogram figures for many targets.
    Returns a list of saved paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        # Infer targets from syllable_order + syllables_present if available
        labs: set[str] = set()
        if "syllable_order" in organized_df.columns:
            for v in organized_df["syllable_order"].tolist():
                labs.update(_coerce_listlike(v))
        if "syllables_present" in organized_df.columns:
            for v in organized_df["syllables_present"].tolist():
                labs.update(_coerce_listlike(v))
        targets2 = _sorted_labels_natural(list(labs))
    else:
        targets2 = [str(t) for t in targets]

    saved: List[Path] = []
    for t in targets2:
        p = out_dir / f"neighbors__target_{_sanitize_for_filename(t)}.png"
        plot_neighbors_hist_pre_post(
            organized_df,
            target=str(t),
            treatment_date=treatment_date,
            out_path=p,
            include_treatment_day=include_treatment_day,
            collapse_repeats=collapse_repeats,
            ignore_labels=ignore_labels,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            top_k=top_k,
            title=f"Neighbors for target={t} (pre vs post)",
        )
        saved.append(p)
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Remaining-to-end computation (mean # syllables after each occurrence)
# ──────────────────────────────────────────────────────────────────────────────
def _remaining_counts_for_sequence(
    seq: Sequence[str],
    *,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, List[int]]:
    """
    For each occurrence of a syllable a at position i in the (possibly collapsed) sequence,
    record remaining = len(seq) - i - 1.
    Returns dict syllable -> list of remaining counts (one entry per occurrence).
    """
    ignore = set(ignore_labels or [])
    labs = [str(x) for x in seq if str(x) not in ignore]
    if collapse_repeats:
        labs = _collapse_consecutive(labs)
    out: Dict[str, List[int]] = {}
    L = len(labs)
    for i, a in enumerate(labs):
        rem = int(max(0, L - i - 1))
        out.setdefault(a, []).append(rem)
    return out


def compute_mean_remaining_to_end_pre_post(
    organized_df: pd.DataFrame,
    *,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
    order_col: str = "syllable_order",
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    min_occurrences: int = 1,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per syllable:
      syllable, mean_pre, mean_post, delta, n_occ_pre, n_occ_post
    """
    pre_df, post_df = split_pre_post(
        organized_df,
        treatment_date=treatment_date,
        include_treatment_day=include_treatment_day,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    def _accumulate(seqs: List[List[str]]) -> Dict[str, List[int]]:
        acc: Dict[str, List[int]] = {}
        for s in seqs:
            d = _remaining_counts_for_sequence(s, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)
            for k, v in d.items():
                acc.setdefault(k, []).extend(v)
        return acc

    pre_acc = _accumulate(pre_seqs)
    post_acc = _accumulate(post_seqs)

    targets = set(pre_acc) | set(post_acc)
    ignore = set(ignore_labels or [])
    targets = {t for t in targets if t not in ignore}
    targets_sorted = _sorted_labels_natural(list(targets))

    rows: List[Dict[str, object]] = []
    for a in targets_sorted:
        pre_vals = np.asarray(pre_acc.get(a, []), dtype=float)
        post_vals = np.asarray(post_acc.get(a, []), dtype=float)

        npre = int(pre_vals.size)
        npost = int(post_vals.size)

        if npre < min_occurrences and npost < min_occurrences:
            continue

        mean_pre = float(np.mean(pre_vals)) if npre else float("nan")
        mean_post = float(np.mean(post_vals)) if npost else float("nan")
        delta = (mean_post - mean_pre) if (np.isfinite(mean_pre) and np.isfinite(mean_post)) else float("nan")

        rows.append(
            dict(
                syllable=str(a),
                mean_pre=mean_pre,
                mean_post=mean_post,
                delta=delta,
                n_occ_pre=npre,
                n_occ_post=npost,
            )
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_mean_remaining_to_end_pre_post(
    rem_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    show: bool = False,
    connect_pairs: bool = True,
    legend: bool = True,
    legend_max_labels: int = 60,
    legend_fontsize: float = 7.0,
    legend_ncol: int = 2,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    stats_seed: int = 0,
) -> None:
    """
    Plot: each syllable is a line from mean_pre to mean_post (y-axis = mean remaining).
    Includes an optional legend (requested).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rem_df is None or rem_df.empty:
        raise ValueError("rem_df is empty; nothing to plot.")

    df = rem_df.copy()
    df["syllable"] = df["syllable"].astype(str)
    df["mean_pre"] = pd.to_numeric(df["mean_pre"], errors="coerce")
    df["mean_post"] = pd.to_numeric(df["mean_post"], errors="coerce")

    dfp = df[np.isfinite(df["mean_pre"]) & np.isfinite(df["mean_post"])].copy()
    if dfp.empty:
        raise ValueError("No rows with finite mean_pre and mean_post.")

    uniq = _sorted_labels_natural(dfp["syllable"].tolist())
    colors = _get_distinct_qual_colors(max(len(uniq), 30))
    color_map: Dict[str, tuple] = {lab: colors[i] for i, lab in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    rng = np.random.default_rng(stats_seed)
    jitter = rng.normal(loc=0.0, scale=0.05, size=len(dfp)) if len(dfp) else np.array([])
    x1 = 1.0 + jitter
    x2 = 2.0 + jitter

    handles = []
    labels = []

    for i, (_, row) in enumerate(dfp.iterrows()):
        lab = row["syllable"]
        c = color_map.get(lab, (0, 0, 0, 1))
        y1 = float(row["mean_pre"])
        y2 = float(row["mean_post"])

        if connect_pairs:
            ax.plot([x1[i], x2[i]], [y1, y2], linewidth=1, alpha=0.65, color=c)
        ax.scatter([x1[i]], [y1], s=26, alpha=0.92, color=c)
        ax.scatter([x2[i]], [y2], s=26, alpha=0.92, color=c)

        if legend and (len(uniq) <= legend_max_labels):
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="None", color=c, markersize=6))
            npre = row.get("n_occ_pre", np.nan)
            npost = row.get("n_occ_post", np.nan)
            npre_s = "NA" if not np.isfinite(float(npre)) else str(int(float(npre)))
            npost_s = "NA" if not np.isfinite(float(npost)) else str(int(float(npost)))
            labels.append(f"{lab} (n_pre={npre_s}, n_post={npost_s})")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean # syllables remaining to end")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(_soft_wrap_title(title, parts_per_line=2) or "", y=0.98, fontsize=11)

    if legend:
        if len(uniq) > legend_max_labels:
            ax.text(
                0.02, 0.02,
                f"Legend omitted: {len(uniq)} syllables > legend_max_labels={legend_max_labels}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
            )
        else:
            fig.legend(
                handles,
                labels,
                frameon=False,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                title="Syllable label",
                ncol=int(max(1, legend_ncol)),
                fontsize=float(legend_fontsize),
                borderaxespad=0.0,
                handletextpad=0.4,
                columnspacing=0.8,
            )

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 0.92] if legend else [0.0, 0.0, 1.0, 0.92])
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Variance-tier selection (load per-syllable phrase-duration variance)
# ──────────────────────────────────────────────────────────────────────────────
def load_variance_table(
    variance_path: Union[str, Path],
    *,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a variance table from CSV or Excel.

    Expected columns (case-insensitive):
      - Animal ID (or animal_id)
      - Syllable (or syllable)
      - Var_ms2 (or variance, var, etc.)

    The provided file can also include DayRel; if present, it can be used to focus on pre rows.
    """
    p = Path(variance_path)
    if not p.exists():
        raise FileNotFoundError(p)

    suf = p.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet_name or 0)
    elif suf in [".csv", ".tsv"]:
        df = pd.read_csv(p, sep="\t" if suf == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported variance file type: {p.suffix}. Use .csv or .xlsx/.xls")

    return df


def summarize_variance_by_syllable(
    variance_df: pd.DataFrame,
    *,
    animal_id: str,
    use_pre_only: bool = True,
    dayrel_col_candidates: Sequence[str] = ("DayRel", "dayrel", "day_rel", "day_relative"),
    animal_col_candidates: Sequence[str] = ("Animal ID", "animal_id", "AnimalID"),
    syll_col_candidates: Sequence[str] = ("Syllable", "syllable", "label"),
    var_col_candidates: Sequence[str] = ("Var_ms2", "var_ms2", "variance", "Var", "var"),
    agg: str = "median",
    min_rows_per_syllable: int = 5,
) -> pd.DataFrame:
    """
    Returns a per-syllable variance summary table with columns:
      syllable, var_summary, n_rows_used

    If DayRel is present and use_pre_only=True, uses rows with DayRel < 0.
    """
    df = variance_df.copy()

    a_col = _find_col_case_insensitive(df, list(animal_col_candidates))
    s_col = _find_col_case_insensitive(df, list(syll_col_candidates))
    v_col = _find_col_case_insensitive(df, list(var_col_candidates))
    d_col = _find_col_case_insensitive(df, list(dayrel_col_candidates))

    if a_col is None or s_col is None or v_col is None:
        raise KeyError(
            f"Variance table missing required columns. Found columns: {list(df.columns)}. "
            f"Need Animal ID, Syllable, and a variance column (e.g., Var_ms2)."
        )

    sub = df[df[a_col].astype(str) == str(animal_id)].copy()
    if sub.empty:
        raise ValueError(f"No rows for animal_id={animal_id} in variance table.")

    if use_pre_only and (d_col is not None):
        sub[d_col] = pd.to_numeric(sub[d_col], errors="coerce")
        sub = sub[np.isfinite(sub[d_col]) & (sub[d_col] < 0)].copy()

    sub["syllable"] = sub[s_col].astype(str)
    sub["_var_"] = pd.to_numeric(sub[v_col], errors="coerce")
    sub = sub[np.isfinite(sub["_var_"])].copy()

    if sub.empty:
        raise ValueError("Variance subset is empty after filtering/parsing.")

    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'.")

    grouped = sub.groupby("syllable")["_var_"]
    if agg == "median":
        var_sum = grouped.median()
    else:
        var_sum = grouped.mean()
    n_rows = grouped.size()

    out = pd.DataFrame({"syllable": var_sum.index.astype(str), "var_summary": var_sum.values, "n_rows_used": n_rows.values})
    out = out[out["n_rows_used"] >= int(min_rows_per_syllable)].copy()
    out = out.sort_values("var_summary", ascending=False).reset_index(drop=True)
    return out


def select_top_variance_syllables(
    variance_summary_df: pd.DataFrame,
    *,
    top_frac: float = 0.30,
) -> List[str]:
    """
    Select top fraction of syllables by var_summary.
    """
    if variance_summary_df is None or variance_summary_df.empty:
        return []
    top_frac = float(top_frac)
    top_frac = min(max(top_frac, 0.0), 1.0)
    n = int(np.ceil(top_frac * len(variance_summary_df)))
    n = max(n, 1) if len(variance_summary_df) else 0
    return variance_summary_df["syllable"].astype(str).tolist()[:n]


def plot_Ha_boxplots_by_variance_tier(
    ha_df: pd.DataFrame,
    *,
    variance_path: Union[str, Path],
    animal_id: str,
    out_dir: Union[str, Path],
    top_frac: float = 0.30,
    use_pre_only: bool = True,
    variance_sheet_name: Optional[str] = None,
    variance_agg: str = "median",
    min_rows_per_syllable: int = 5,
    title_prefix: Optional[str] = None,
    show: bool = False,
    # pass-through for title clarity
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,
    min_transitions: Optional[int] = None,
) -> Dict[str, object]:
    """
    Creates two figures in out_dir:
      - Ha_high_variance_pre_post.png
      - Ha_low_variance_pre_post.png

    Returns dict with selected labels and paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    var_df = load_variance_table(variance_path, sheet_name=variance_sheet_name)
    var_sum = summarize_variance_by_syllable(
        var_df,
        animal_id=animal_id,
        use_pre_only=use_pre_only,
        agg=variance_agg,
        min_rows_per_syllable=min_rows_per_syllable,
    )
    top_labels = set(select_top_variance_syllables(var_sum, top_frac=top_frac))

    df = ha_df.copy()
    df["syllable"] = df["syllable"].astype(str)

    high = df[df["syllable"].isin(top_labels)].copy()
    low = df[~df["syllable"].isin(top_labels)].copy()

    res: Dict[str, object] = {
        "animal_id": str(animal_id),
        "top_frac": float(top_frac),
        "n_top_syllables": int(len(top_labels)),
        "top_syllables": sorted(list(top_labels), key=lambda x: (_try_int(x) is None, _try_int(x) or 0, x)),
        "paths": {},
    }

    if high.empty:
        res["note_high"] = "No overlap between top-variance syllables and ha_df syllables."
    else:
        title_high = f"{title_prefix + ' | ' if title_prefix else ''}{animal_id} | Top {int(top_frac*100)}% variance syllables: Hₐ pre vs post"
        stats_high = plot_Ha_boxplot_pre_post_connected(
            high,
            out_path=out_dir / "Ha_high_variance_pre_post.png",
            title=title_high,
            show=show,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            min_transitions=min_transitions,
        )
        res["stats_high"] = stats_high
        res["paths"]["high"] = str(out_dir / "Ha_high_variance_pre_post.png")

    if low.empty:
        res["note_low"] = "Low-variance tier is empty after filtering."
    else:
        title_low = f"{title_prefix + ' | ' if title_prefix else ''}{animal_id} | Remaining low-variance syllables: Hₐ pre vs post"
        stats_low = plot_Ha_boxplot_pre_post_connected(
            low,
            out_path=out_dir / "Ha_low_variance_pre_post.png",
            title=title_low,
            show=show,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            min_transitions=min_transitions,
        )
        res["stats_low"] = stats_low
        res["paths"]["low"] = str(out_dir / "Ha_low_variance_pre_post.png")

    # Also save the summary table used
    var_sum.to_csv(out_dir / "variance_summary_by_syllable.csv", index=False)
    res["paths"]["variance_summary"] = str(out_dir / "variance_summary_by_syllable.csv")

    return res



# ──────────────────────────────────────────────────────────────────────────────
# High-level "do everything" wrapper (used by scripts and CLI)
# ──────────────────────────────────────────────────────────────────────────────
def make_entropy_by_syllable_outputs(
    *,
    decoded_database_json: Union[str, Path],
    song_detection_json: Union[str, Path],
    metadata_xlsx: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Optional[str] = None,
    treatment_date: Optional[str] = None,
    animal_id: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,

    include_treatment_day: bool = False,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,

    log_base: float = 2.0,
    min_transitions: int = 1,

    # optional seasonality windows (days relative to treatment date)
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,

    # optional neighbor histograms (prev+next) for each syllable
    make_neighbors_hist: bool = True,
    neighbors_top_k: Optional[int] = 12,
    neighbors_targets: Optional[Sequence[str]] = None,

    # optional remaining-to-end plot
    make_mean_remaining_to_end: bool = True,
    mean_remaining_min_occurrences: int = 10,
    mean_remaining_legend_max_labels: int = 60,

    # optional variance-tier entropy boxplots
    make_variance_tiers: bool = False,
    variance_path: Optional[Union[str, Path]] = None,
    variance_sheet_name: Optional[str] = None,
    variance_top_frac: float = 0.30,
    variance_use_pre_only: bool = True,
    variance_agg: str = "median",
    variance_min_rows_per_syllable: int = 5,

    # optional show plots interactively
    show: bool = False,
) -> Dict[str, object]:
    """
    End-to-end runner that:
      1) merges split songs + appends annotations using detection timing
      2) resolves treatment date (arg or Excel lookup)
      3) builds organized_df
      4) computes H_a for all syllables (pre vs post)
      5) saves:
           - Ha_all_syllables__boxplot_connected.png
           - Ha_all_syllables__boxplot_by_label__with_usage.png
           - mean_remaining_to_end_pre_vs_post.png (optional; includes legend)
           - neighbors histograms (optional)
           - variance-tier boxplots (optional)
           - TE_pre_post_summary.txt

    Returns dict with paths and key results (ha_df, TE summaries).
    """
    decoded_database_json = Path(decoded_database_json)
    song_detection_json = Path(song_detection_json)

    if out_dir is None:
        out_dir = song_detection_json.parent / "entropy_by_syllable_outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge (split-song aware)
    merged_df = build_merged_annotations_df(
        decoded_database_json=decoded_database_json,
        song_detection_json=song_detection_json,
    )

    # Resolve treatment date
    t_dt, animal_id_used, t_src = resolve_treatment_date(
        treatment_date_arg=treatment_date,
        metadata_xlsx=metadata_xlsx,
        animal_id_arg=animal_id,
        merged_df=merged_df,
        sheet_name=metadata_sheet_name,
    )

    # Organized DF
    organized = build_organized_from_merged_df(merged_df, treatment_date=t_dt)
    organized_df = organized.organized_df

    # H_a table
    ha_df = compute_pre_post_Ha_all_syllables(
        organized_df,
        treatment_date=t_dt,
        include_treatment_day=include_treatment_day,
        collapse_repeats=collapse_repeats,
        ignore_labels=ignore_labels,
        log_base=log_base,
        min_transitions=min_transitions,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    # TE summaries
    pre_TE, post_TE = compute_pre_post_total_transition_entropy(
        organized_df,
        treatment_date=t_dt,
        include_treatment_day=include_treatment_day,
        collapse_repeats=collapse_repeats,
        ignore_labels=ignore_labels,
        log_base=log_base,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
    )

    # Save TE summary
    te_path = out_dir / "TE_pre_post_summary.txt"
    with open(te_path, "w") as f:
        f.write(f"animal_id: {animal_id_used}\n")
        f.write(f"treatment_date: {t_dt.date()} (source={t_src})\n")
        f.write(f"log_base: {log_base}\n")
        f.write(f"collapse_repeats: {collapse_repeats}\n")
        f.write(f"ignore_labels: {ignore_labels}\n")
        f.write(f"window: max_days_pre={max_days_pre}, max_days_post={max_days_post}\n\n")

        f.write("PRE:\n")
        f.write(f"  TE: {pre_TE.total_entropy}\n")
        f.write(f"  n_total_transitions: {pre_TE.n_total_transitions}\n\n")

        f.write("POST:\n")
        f.write(f"  TE: {post_TE.total_entropy}\n")
        f.write(f"  n_total_transitions: {post_TE.n_total_transitions}\n\n")

        if np.isfinite(pre_TE.total_entropy) and np.isfinite(post_TE.total_entropy):
            f.write(f"DELTA (post - pre): {post_TE.total_entropy - pre_TE.total_entropy}\n")

    # Save main plots
    paths: Dict[str, str] = {}

    p1 = out_dir / "Ha_all_syllables__boxplot_connected.png"
    plot_Ha_boxplot_pre_post_connected(
        ha_df,
        out_path=p1,
        title=f"{animal_id_used} | Hₐ across syllables (pre vs post)",
        show=show,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
        min_transitions=min_transitions,
    )
    paths["Ha_boxplot_connected"] = str(p1)

    p2 = out_dir / "Ha_all_syllables__boxplot_by_label__with_usage.png"
    plot_Ha_boxplot_pre_post_by_label(
        ha_df,
        out_path=p2,
        title=f"{animal_id_used} | Per-syllable Hₐ (colored by syllable; legend shows n_pre/n_post)",
        show=show,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,
        min_transitions=min_transitions,
    )
    paths["Ha_boxplot_by_label_with_usage"] = str(p2)

    # Remaining-to-end plot (requested legend included)
    rem_df = None
    if make_mean_remaining_to_end:
        rem_df = compute_mean_remaining_to_end_pre_post(
            organized_df,
            treatment_date=t_dt,
            include_treatment_day=include_treatment_day,
            collapse_repeats=collapse_repeats,
            ignore_labels=ignore_labels,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            min_occurrences=mean_remaining_min_occurrences,
        )

        p3 = out_dir / "mean_remaining_to_end_pre_vs_post.png"
        plot_mean_remaining_to_end_pre_post(
            rem_df,
            out_path=p3,
            title=f"{animal_id_used} | Mean syllables remaining to end (pre vs post)",
            legend=True,
            legend_max_labels=mean_remaining_legend_max_labels,
            show=show,
        )
        paths["mean_remaining_to_end"] = str(p3)

    # Neighbors histograms (prev + next for each syllable)
    neigh_paths: List[str] = []
    if make_neighbors_hist:
        neigh_dir = out_dir / "neighbors_histograms"
        saved = plot_neighbors_hist_for_all_targets(
            organized_df,
            treatment_date=t_dt,
            out_dir=neigh_dir,
            targets=neighbors_targets,
            include_treatment_day=include_treatment_day,
            collapse_repeats=collapse_repeats,
            ignore_labels=ignore_labels,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            top_k=neighbors_top_k,
        )
        neigh_paths = [str(p) for p in saved]
        paths["neighbors_hist_dir"] = str(neigh_dir)

    # Variance-tier boxplots
    var_res: Optional[Dict[str, object]] = None
    if make_variance_tiers:
        if variance_path is None:
            raise ValueError("make_variance_tiers=True requires variance_path=...")

        var_res = plot_Ha_boxplots_by_variance_tier(
            ha_df,
            variance_path=variance_path,
            animal_id=animal_id_used,
            out_dir=out_dir / "variance_tiers",
            top_frac=variance_top_frac,
            use_pre_only=variance_use_pre_only,
            variance_sheet_name=variance_sheet_name,
            variance_agg=variance_agg,
            min_rows_per_syllable=variance_min_rows_per_syllable,
            title_prefix="Variance tier",
            show=show,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
            min_transitions=min_transitions,
        )
        paths["variance_tiers_dir"] = str(out_dir / "variance_tiers")

    # Save data tables
    ha_csv = out_dir / "Ha_pre_post_by_syllable.csv"
    ha_df.to_csv(ha_csv, index=False)
    paths["Ha_table_csv"] = str(ha_csv)

    if rem_df is not None and isinstance(rem_df, pd.DataFrame) and not rem_df.empty:
        rem_csv = out_dir / "mean_remaining_to_end_pre_post.csv"
        rem_df.to_csv(rem_csv, index=False)
        paths["mean_remaining_table_csv"] = str(rem_csv)

    return {
        "animal_id": animal_id_used,
        "treatment_date": str(pd.to_datetime(t_dt).date()),
        "treatment_date_source": t_src,
        "out_dir": str(out_dir),
        "paths": paths,
        "ha_df": ha_df,
        "rem_df": rem_df,
        "pre_TE": pre_TE,
        "post_TE": post_TE,
        "variance_tiers": var_res,
        "neighbors_hist_paths": neigh_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Compute per-syllable transition entropy H_a (pre vs post).")
    p.add_argument("--decoded", required=True, help="Path to *_decoded_database.json")
    p.add_argument("--detect", required=True, help="Path to *_song_detection.json")

    p.add_argument("--metadata-xlsx", default=None, help="Excel file with treatment dates (optional if --treatment-date given)")
    p.add_argument("--metadata-sheet", default=None, help="Sheet name in metadata-xlsx (optional)")

    p.add_argument("--treatment-date", default=None, help="Treatment date override (YYYY-MM-DD). If omitted, uses metadata-xlsx lookup.")
    p.add_argument("--animal-id", default=None, help="Animal ID override for metadata lookup")

    p.add_argument("--out-dir", default=None, help="Output directory (default: sibling 'entropy_by_syllable_outputs')")

    p.add_argument("--include-treatment-day", action="store_true", help="Include treatment day in both pre and post")
    p.add_argument("--no-collapse-repeats", action="store_true", help="Do NOT collapse consecutive repeats")
    p.add_argument("--ignore-label", action="append", default=None, help="Label(s) to ignore; can be passed multiple times")
    p.add_argument("--log-base", type=float, default=2.0, help="Log base for entropy (default 2 -> bits)")
    p.add_argument("--min-transitions", type=int, default=1, help="Minimum transitions for a syllable to be included")

    p.add_argument("--max-days-pre", type=int, default=None, help="Limit to days in [t-max_days_pre, t) (or ≤t if include-treatment-day)")
    p.add_argument("--max-days-post", type=int, default=None, help="Limit to days in (t, t+max_days_post] (or ≥t if include-treatment-day)")

    p.add_argument("--no-neighbors", action="store_true", help="Do NOT make per-target neighbor histograms")
    p.add_argument("--neighbors-top-k", type=int, default=12, help="Top K neighbors to show in histograms (default 12)")

    p.add_argument("--no-remaining", action="store_true", help="Do NOT make mean remaining-to-end plot")
    p.add_argument("--remaining-min-occ", type=int, default=10, help="Min occurrences per syllable for remaining-to-end plot")

    p.add_argument("--variance-path", default=None, help="CSV/XLSX variance table (enables variance-tier plots)")
    p.add_argument("--variance-sheet", default=None, help="Excel sheet name for variance table (optional)")
    p.add_argument("--variance-top-frac", type=float, default=0.30, help="Top fraction by variance (default 0.30)")
    p.add_argument("--variance-use-pre-only", action="store_true", help="Use DayRel<0 rows when available (default True).")
    p.add_argument("--variance-use-all", action="store_true", help="Override: do NOT restrict to pre rows even if DayRel present.")
    p.add_argument("--variance-agg", type=str, default="median", choices=["median", "mean"], help="How to summarize variance per syllable")
    p.add_argument("--variance-min-rows", type=int, default=5, help="Min rows per syllable in variance table to be eligible")

    p.add_argument("--show", action="store_true", help="Show figures interactively")
    return p


def main():
    args = _build_arg_parser().parse_args()

    make_variance = args.variance_path is not None
    variance_use_pre_only = True
    if args.variance_use_all:
        variance_use_pre_only = False
    elif args.variance_use_pre_only:
        variance_use_pre_only = True

    res = make_entropy_by_syllable_outputs(
        decoded_database_json=args.decoded,
        song_detection_json=args.detect,
        metadata_xlsx=args.metadata_xlsx,
        metadata_sheet_name=args.metadata_sheet,
        treatment_date=args.treatment_date,
        animal_id=args.animal_id,
        out_dir=args.out_dir,

        include_treatment_day=args.include_treatment_day,
        collapse_repeats=not args.no_collapse_repeats,
        ignore_labels=args.ignore_label,

        log_base=args.log_base,
        min_transitions=args.min_transitions,

        max_days_pre=args.max_days_pre,
        max_days_post=args.max_days_post,

        make_neighbors_hist=not args.no_neighbors,
        neighbors_top_k=args.neighbors_top_k,

        make_mean_remaining_to_end=not args.no_remaining,
        mean_remaining_min_occurrences=args.remaining_min_occ,

        make_variance_tiers=make_variance,
        variance_path=args.variance_path,
        variance_sheet_name=args.variance_sheet,
        variance_top_frac=args.variance_top_frac,
        variance_use_pre_only=variance_use_pre_only,
        variance_agg=args.variance_agg,
        variance_min_rows_per_syllable=args.variance_min_rows,

        show=args.show,
    )

    print(f"[entropy_by_syllable] Wrote outputs to: {res['out_dir']}")
    for k, v in res.get("paths", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT EXAMPLE USAGE (Spyder console)
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import importlib
import entropy_by_syllable as ebs
importlib.reload(ebs)

song_detection = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288_song_detection.json")
decoded        = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288_decoded_database.json")
metadata_xlsx  = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")

# Example variance file (CSV or Excel)
variance_table = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/aggregate_variance_points_by_bird_and_syllable.csv")

MAX_DAYS_PRE  = 40
MAX_DAYS_POST = 40
MIN_TRANSITIONS = 50

# 1) Merge split songs + append annotations using detection timing
merged_df = ebs.build_merged_annotations_df(
    decoded_database_json=decoded,
    song_detection_json=song_detection,
    max_gap_between_song_segments=500,
    segment_index_offset=0,
    merge_repeated_syllables=True,
    repeat_gap_ms=10,
    repeat_gap_inclusive=False,
)

# 2) Resolve treatment date (lookup from Excel for this animal)
treatment_dt, animal_id_used, src = ebs.resolve_treatment_date(
    treatment_date_arg=None,
    metadata_xlsx=metadata_xlsx,
    animal_id_arg=None,               # or "USA5336"
    merged_df=merged_df,
    sheet_name="metadata_with_hit_type",
)

# 3) Build organized dataframe from merged data
organized_df = ebs.build_organized_from_merged_df(
    merged_df,
    compute_durations=True,
    add_recording_datetime=True,
    treatment_date=treatment_dt,
).organized_df

# 4) H_a table (restricted to ±40 days)
ha_df = ebs.compute_pre_post_Ha_all_syllables(
    organized_df,
    treatment_date=treatment_dt,
    include_treatment_day=False,
    collapse_repeats=True,
    ignore_labels=["silence", "-"],
    log_base=2.0,
    min_transitions=MIN_TRANSITIONS,
    max_days_pre=MAX_DAYS_PRE,
    max_days_post=MAX_DAYS_POST,
)

out_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/entropy_new_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 5) Mean remaining-to-end plot (WITH legend)
rem_df = ebs.compute_mean_remaining_to_end_pre_post(
    organized_df,
    treatment_date=treatment_dt,
    include_treatment_day=False,
    collapse_repeats=True,
    ignore_labels=["silence", "-"],
    max_days_pre=MAX_DAYS_PRE,
    max_days_post=MAX_DAYS_POST,
    min_occurrences=10,
)
ebs.plot_mean_remaining_to_end_pre_post(
    rem_df,
    out_path=out_dir / "mean_remaining_to_end_pre_vs_post.png",
    title=f"{animal_id_used} | Mean syllables remaining to end (pre vs post) | treat={treatment_dt.date()}",
    legend=True,
    legend_max_labels=60,
)

# 6) Batch neighbor histograms (prev + next) for all targets
ebs.plot_neighbors_hist_for_all_targets(
    organized_df,
    treatment_date=treatment_dt,
    out_dir=out_dir / "neighbors_histograms",
    ignore_labels=["silence", "-"],
    max_days_pre=MAX_DAYS_PRE,
    max_days_post=MAX_DAYS_POST,
    top_k=12,
)

# 7) Variance-tier entropy plots (top 30% variance vs rest)
res_var = ebs.plot_Ha_boxplots_by_variance_tier(
    ha_df,
    variance_path=variance_table,
    animal_id=animal_id_used,
    out_dir=out_dir / "variance_tiers",
    top_frac=0.30,
    use_pre_only=True,           # uses DayRel<0 if present
    variance_agg="median",
    min_rows_per_syllable=5,
    title_prefix="Entropy vs variance tier",
    max_days_pre=MAX_DAYS_PRE,
    max_days_post=MAX_DAYS_POST,
    min_transitions=MIN_TRANSITIONS,
)
print(res_var)
"""
