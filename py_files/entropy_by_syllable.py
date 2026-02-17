#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_by_syllable.py

Compute transition entropy H_a for one bird, using merged annotations (split-song aware),
and compare PRE vs POST lesion.

Includes:
  • build_merged_annotations_df(...)              # merges split songs + appends annotations
  • resolve_treatment_date(...)                  # pass treatment date OR look it up in metadata Excel
  • build_organized_from_merged_df(...)          # builds organizer-style DF from merged output
  • compute_pre_post_Ha(...)                     # H_a for one target syllable
  • compute_pre_post_Ha_all_syllables(...)       # H_a for every syllable (as the preceding syllable)
  • plot_Ha_all_syllables_pre_post(...)          # paired plot or scatter (H_pre vs H_post)
  • plot_Ha_boxplot_pre_post_connected(...)      # boxplot w/ BLUE(pre) + ORANGE(post) points + connecting lines
  • plot_Ha_boxplot_pre_post_by_label(...)       # second boxplot: each syllable label gets its own color (+ lines)
  • plot_followers_pre_post(...)                 # follower distribution for ONE target syllable

Stats:
  • paired_entropy_tests(...) does paired tests across syllables (finite pre+post):
      - Wilcoxon signed-rank + paired t-test if SciPy is available
      - otherwise a sign-flip permutation test on mean difference

Important:
  - This module intentionally does NOT import organize_decoded_with_segments.py
    to avoid circular-import issues. It contains local copies of the small helper
    functions it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
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
    """
    Sort labels like ['0','1','10','2'] into numeric order when possible,
    otherwise fall back to lexicographic.
    """
    labs = list({str(x) for x in labels})
    numeric = [(lab, _try_int(lab)) for lab in labs]
    if numeric and all(v is not None for _, v in numeric):
        return [lab for lab, _ in sorted(numeric, key=lambda t: int(t[1]))]
    return sorted(labs)


# ──────────────────────────────────────────────────────────────────────────────
# Entropy
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
# Treatment date: arg OR Excel lookup
# ──────────────────────────────────────────────────────────────────────────────
def infer_animal_id_from_merged_df(df: pd.DataFrame) -> Optional[str]:
    """Infer Animal ID from merged DF using 'Animal ID' column or file_name prefix."""
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
    """
    Find treatment date for animal_id in an Excel file.
    Looks for columns (case-insensitive):
      - Animal ID
      - Treatment date (variants)

    If multiple dates exist, returns earliest.
    """
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
    """Returns (treatment_date_timestamp, animal_id_used, source='arg'|'metadata_xlsx')."""
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
    """
    Merge split songs + append annotations using detection timing.
    Returns the merged annotations dataframe (singles + merged).
    """
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
    """
    Build an OrganizedDataset from the merged dataframe.
    Adds:
      - syllable_onsets_offsets_ms_dict
      - syllables_present
      - syllable_order
      - syllable_<label>_durations (optional)
    """
    df = merged_df.copy()

    if "syllable_onsets_offsets_ms_dict" not in df.columns:
        if "syllable_onsets_offsets_ms" in df.columns:
            df["syllable_onsets_offsets_ms_dict"] = df["syllable_onsets_offsets_ms"].apply(parse_json_safe)
        else:
            df["syllable_onsets_offsets_ms_dict"] = [{} for _ in range(len(df))]

    # Normalize Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "Recording DateTime" in df.columns:
        df["Date"] = pd.to_datetime(df["Recording DateTime"], errors="coerce").dt.normalize()
    else:
        df["Date"] = pd.NaT

    if add_recording_datetime and "Recording DateTime" in df.columns:
        df["Recording DateTime"] = pd.to_datetime(df["Recording DateTime"], errors="coerce")

    # Unique labels across merged data
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
# Pre/Post split + compute H_a (single syllable)
# ──────────────────────────────────────────────────────────────────────────────
def split_pre_post(
    df: pd.DataFrame,
    treatment_date: Union[str, pd.Timestamp],
    include_treatment_day: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(treatment_date).normalize()
    if include_treatment_day:
        pre = df[df["Date"] <= t].copy()
        post = df[df["Date"] >= t].copy()
    else:
        pre = df[df["Date"] < t].copy()
        post = df[df["Date"] > t].copy()
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
) -> Tuple[EntropyResult, EntropyResult]:
    pre_df, post_df = split_pre_post(
        organized_df, treatment_date=treatment_date, include_treatment_day=include_treatment_day
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
# ALL syllables: compute H_a pre vs post
# ──────────────────────────────────────────────────────────────────────────────
def _bigram_counts_by_prev(
    sequences: Sequence[Sequence[str]],
    *,
    collapse_repeats: bool = True,
    ignore_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Returns nested dict:
      counts[prev][next] = number of transitions prev -> next
    """
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
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per syllable 'a', including:
      - H_pre, H_post
      - n_pre, n_post
      - followers_pre, followers_post
      - delta (post - pre)

    min_transitions filters syllables that have <min_transitions transitions in BOTH
    pre and post (keeps syllables that meet threshold in at least one side).
    """
    pre_df, post_df = split_pre_post(
        organized_df, treatment_date=treatment_date, include_treatment_day=include_treatment_day
    )

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    pre_counts_all = _bigram_counts_by_prev(pre_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)
    post_counts_all = _bigram_counts_by_prev(post_seqs, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)

    targets = set(pre_counts_all.keys()) | set(post_counts_all.keys())

    # also include labels that appear in the dataset even if they have no outgoing transitions
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
# Paired statistical tests across syllables (finite pre+post)
# ──────────────────────────────────────────────────────────────────────────────
def _paired_diffs_from_ha_df(ha_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pre, post, diffs=post-pre) restricted to syllables with finite pre and post.
    """
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
    """
    Paired sign-flip permutation test on mean(diffs).
    H0: diffs are symmetric about 0.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))

    # Fast: random sign matrix
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size), replace=True)
    perm_means = (signs * diffs[None, :]).mean(axis=1)

    if two_sided:
        p = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1))
    else:
        # one-sided: post > pre (mean diff > 0)
        p = float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))
    return p


def paired_entropy_tests(
    ha_df: pd.DataFrame,
    *,
    n_perm: int = 20000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Run paired tests across syllables (requires finite pre+post per syllable).
    Returns a dict with summary stats + available tests.

    Tests included:
      - SciPy: paired t-test + Wilcoxon signed-rank (if SciPy available)
      - Fallback: sign-flip permutation test on mean difference
    """
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
            tt = _scipy_stats.ttest_rel(post, pre, nan_policy="omit")  # post vs pre
            out["ttest_rel_p_two_sided"] = float(tt.pvalue)
            out["ttest_rel_t"] = float(tt.statistic)
        except Exception as e:
            out["ttest_rel_error"] = str(e)

        try:
            # Wilcoxon fails if all diffs are 0; handle gently
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

    # Always include a permutation-based p-value (robust, minimal assumptions)
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
    """
    Compact string for putting on plots.
    """
    n = stats.get("n_paired_syllables", "NA")
    p_perm = _format_p(stats.get("signflip_perm_p_two_sided", np.nan))
    parts = [f"n={n}", f"perm p={p_perm}"]
    if "wilcoxon_p_two_sided" in stats:
        parts.append(f"wilc p={_format_p(stats.get('wilcoxon_p_two_sided'))}")
    if "ttest_rel_p_two_sided" in stats:
        parts.append(f"t p={_format_p(stats.get('ttest_rel_p_two_sided'))}")
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting: ALL syllables (paired plot or scatter)
# ──────────────────────────────────────────────────────────────────────────────
def plot_Ha_all_syllables_pre_post(
    ha_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    style: str = "paired",  # "paired" or "scatter"
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    style="paired": x is syllable (categorical), y is H; show pre+post points per syllable
    style="scatter": x=H_pre, y=H_post with y=x reference line
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ha_df is None or ha_df.empty:
        raise ValueError("ha_df is empty; nothing to plot.")

    df = ha_df.copy()

    if style.lower() == "scatter":
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        x = df["H_pre"].to_numpy(dtype=float)
        y = df["H_post"].to_numpy(dtype=float)

        ax.scatter(x, y)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            lo = float(min(x[finite].min(), y[finite].min()))
            hi = float(max(x[finite].max(), y[finite].max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--")

        ax.set_xlabel("H_pre (bits)")
        ax.set_ylabel("H_post (bits)")
        if title:
            ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    # paired plot
    fig, ax = plt.subplots(figsize=(max(8.5, 0.35 * len(df)), 4.8))

    labels = df["syllable"].astype(str).tolist()
    x = np.arange(len(labels))

    pre = df["H_pre"].to_numpy(dtype=float)
    post = df["H_post"].to_numpy(dtype=float)

    for i in range(len(labels)):
        if np.isfinite(pre[i]) and np.isfinite(post[i]):
            ax.plot([x[i], x[i]], [pre[i], post[i]], linewidth=1)

    ax.scatter(x, pre, label="Pre")
    ax.scatter(x, post, label="Post")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Transition entropy Hₐ (bits)")
    ax.set_xlabel("Syllable a (preceding syllable)")
    ax.legend(frameon=False, loc="best")

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Box plots with connecting lines + stats
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
) -> Dict[str, object]:
    """
    Box plot comparing the distribution of per-syllable H_a values (pre vs post),
    with BLUE(pre) and ORANGE(post) points and (optionally) connecting lines
    for each syllable.

    Returns a stats dict from paired_entropy_tests(ha_df).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ha_df is None or ha_df.empty:
        raise ValueError("ha_df is empty; nothing to plot.")

    # full distributions (may include syllables with finite on only one side)
    pre_all = pd.to_numeric(ha_df["H_pre"], errors="coerce").to_numpy(dtype=float)
    post_all = pd.to_numeric(ha_df["H_post"], errors="coerce").to_numpy(dtype=float)
    pre_all = pre_all[np.isfinite(pre_all)]
    post_all = post_all[np.isfinite(post_all)]

    if pre_all.size == 0 and post_all.size == 0:
        raise ValueError("No finite H_pre or H_post values to plot.")

    # paired subset for connecting lines + stats
    pre_p, post_p, diffs = _paired_diffs_from_ha_df(ha_df)

    stats = paired_entropy_tests(ha_df, n_perm=stats_n_perm, seed=stats_seed)

    fig, ax = plt.subplots(figsize=(5.6, 4.9))

    # Boxplots
    ax.boxplot(
        [pre_all, post_all],
        labels=["Pre", "Post"],
        showfliers=True,
        whis=1.5,
    )

    # Overlay points and connecting lines for paired syllables
    if overlay_points:
        rng = np.random.default_rng(stats_seed)
        jitter = rng.normal(loc=0.0, scale=0.05, size=pre_p.size) if pre_p.size else np.array([])

        x1 = 1.0 + jitter
        x2 = 2.0 + jitter

        if connect_pairs and pre_p.size:
            for i in range(pre_p.size):
                if np.isfinite(pre_p[i]) and np.isfinite(post_p[i]):
                    ax.plot([x1[i], x2[i]], [pre_p[i], post_p[i]], linewidth=1, alpha=0.5)

        # Keep BLUE / ORANGE points (Matplotlib default cycle C0/C1)
        if pre_p.size:
            ax.scatter(x1, pre_p, s=20, alpha=0.85, color="C0", label="Pre")
        if post_p.size:
            ax.scatter(x2, post_p, s=20, alpha=0.85, color="C1", label="Post")

        ax.legend(frameon=False, loc="best")

    ax.set_ylabel("Transition entropy Hₐ (bits)")
    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_stats:
        msg = format_test_summary(stats)
        ax.text(
            0.02, 0.98,
            msg,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
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
    legend_max_labels: int = 20,
) -> Dict[str, object]:
    """
    Second box plot: same Pre/Post boxes, but each syllable's paired points
    are color-coded by syllable label (same color for pre and post), with
    connecting lines (optionally).

    Returns a stats dict from paired_entropy_tests(ha_df).
    """
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

    # paired subset for points/lines (finite pre & post)
    df = ha_df.copy()
    df["H_pre"] = pd.to_numeric(df["H_pre"], errors="coerce")
    df["H_post"] = pd.to_numeric(df["H_post"], errors="coerce")
    df = df[np.isfinite(df["H_pre"]) & np.isfinite(df["H_post"])].copy()
    df["syllable"] = df["syllable"].astype(str)

    stats = paired_entropy_tests(ha_df, n_perm=stats_n_perm, seed=stats_seed)

    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    ax.boxplot(
        [pre_all, post_all],
        labels=["Pre", "Post"],
        showfliers=True,
        whis=1.5,
    )

    # Map syllables to colors using a matplotlib colormap
    syllables = df["syllable"].tolist()
    uniq = _sorted_labels_natural(syllables)
    cmap = plt.get_cmap("tab20")  # good categorical default
    color_map: Dict[str, tuple] = {}
    for i, lab in enumerate(uniq):
        color_map[lab] = cmap(i % cmap.N)

    rng = np.random.default_rng(stats_seed)
    jitter = rng.normal(loc=0.0, scale=0.05, size=len(df)) if len(df) else np.array([])
    x1 = 1.0 + jitter
    x2 = 2.0 + jitter

    # Draw pairs
    for i, (_, row) in enumerate(df.iterrows()):
        lab = row["syllable"]
        c = color_map.get(lab, (0, 0, 0, 1))
        y1 = float(row["H_pre"])
        y2 = float(row["H_post"])

        if connect_pairs:
            ax.plot([x1[i], x2[i]], [y1, y2], linewidth=1, alpha=0.6, color=c)

        ax.scatter([x1[i]], [y1], s=22, alpha=0.9, color=c)
        ax.scatter([x2[i]], [y2], s=22, alpha=0.9, color=c)

    ax.set_ylabel("Transition entropy Hₐ (bits)")
    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_stats:
        msg = format_test_summary(stats)
        ax.text(
            0.02, 0.98,
            msg,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    # Optional legend (only if not too many labels)
    if len(uniq) <= legend_max_labels:
        handles = []
        labels = []
        for lab in uniq:
            h = plt.Line2D([0], [0], marker="o", linestyle="None", color=color_map[lab], markersize=6)
            handles.append(h)
            labels.append(lab)
        ax.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Plotting: ONE target syllable follower distributions (pre vs post)
# ──────────────────────────────────────────────────────────────────────────────
def plot_followers_pre_post(
    pre: EntropyResult,
    post: EntropyResult,
    out_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(pre.follower_probs.keys()) | set(post.follower_probs.keys()))
    pre_vals = [pre.follower_probs.get(k, 0.0) for k in keys]
    post_vals = [post.follower_probs.get(k, 0.0) for k in keys]

    x = np.arange(len(keys))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(keys)), 4.2))
    ax.bar(x - width / 2, pre_vals, width, label=f"Pre (H={pre.entropy:.3g}, n={pre.n_transitions})")
    ax.bar(x + width / 2, post_vals, width, label=f"Post (H={post.entropy:.3g}, n={post.n_transitions})")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("P(next | target)")
    ax.set_xlabel(f"Next syllable (given {pre.target})")
    ax.legend(loc="best", frameon=False)

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT EXAMPLE USAGE (Spyder console) — uses your exact paths
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import importlib
import entropy_by_syllable as ebs
importlib.reload(ebs)

song_detection = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288_song_detection.json")
decoded        = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288_decoded_database.json")
metadata_xlsx  = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")

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
    animal_id_arg=None,               # or "USA5288"
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

print("Animal:", animal_id_used, "| Treatment:", treatment_dt.date(), "| source:", src)

# 4) ALL syllables: compute table
ha_df = ebs.compute_pre_post_Ha_all_syllables(
    organized_df,
    treatment_date=treatment_dt,
    include_treatment_day=False,
    collapse_repeats=True,
    ignore_labels=["silence", "-"],
    log_base=2.0,
    min_transitions=5,   # optional filter
)

out_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/entropy_all_syllables")
out_dir.mkdir(parents=True, exist_ok=True)

# A) Box plot with BLUE/ORANGE points + connecting lines + stats annotation
stats1 = ebs.plot_Ha_boxplot_pre_post_connected(
    ha_df,
    out_path=out_dir / "Ha_all_syllables__boxplot_connected.png",
    title=f"{animal_id_used}: per-syllable H_a (pre vs post)",
    connect_pairs=True,
    annotate_stats=True,
)

print("Stats (connected boxplot):", stats1)

# B) Second box plot: each syllable label gets its own color (+ connecting lines)
stats2 = ebs.plot_Ha_boxplot_pre_post_by_label(
    ha_df,
    out_path=out_dir / "Ha_all_syllables__boxplot_by_label.png",
    title=f"{animal_id_used}: per-syllable H_a (colored by syllable)",
    connect_pairs=True,
    annotate_stats=True,
    legend_max_labels=20,   # set higher if you want a bigger legend
)

print("Stats (colored-by-label boxplot):", stats2)

# Save table and optional debug CSVs
ha_df.to_csv(out_dir / "Ha_all_syllables_pre_post.csv", index=False)
merged_df.to_csv(out_dir / "decoded_with_split_labels.csv", index=False)
organized_df.to_csv(out_dir / "organized_from_merged.csv", index=False)

# If you want the paired plot / scatter too:
ebs.plot_Ha_all_syllables_pre_post(
    ha_df,
    out_path=out_dir / "Ha_all_syllables__paired.png",
    style="paired",
    title=f"{animal_id_used}: H_a pre vs post (paired points per syllable)",
)
ebs.plot_Ha_all_syllables_pre_post(
    ha_df,
    out_path=out_dir / "Ha_all_syllables__scatter.png",
    style="scatter",
    title=f"{animal_id_used}: H_pre vs H_post (all syllables)",
)
"""
