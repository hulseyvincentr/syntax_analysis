#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_by_syllable.py

Compute transition entropy H_a for a single target syllable a, pre vs post lesion,
for ONE bird.

Pipeline:
  1) merge_annotations_from_split_songs.build_decoded_with_split_labels (JSONs -> merged DF)
  2) build "organized-from-merged" DF using functions from organize_decoded_with_segments.py
  3) compute H_a pre vs post using organized_df['syllable_order']

Treatment date:
  - Either pass --treatment-date YYYY-MM-DD
  - OR pass --metadata-xlsx and (optionally) --animal-id
    * If --animal-id not provided, infer from merged_df.

Example:
  python entropy_by_syllable.py \
    --song-detection /path/to/bird_song_detection.json \
    --decoded /path/to/bird_decoded_database.json \
    --metadata-xlsx /path/to/Area_X_lesion_metadata_with_hit_types.xlsx \
    --syllable 9 \
    --out-dir /path/to/out_entropy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import argparse
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your merge pipeline
from merge_annotations_from_split_songs import build_decoded_with_split_labels

# Organizer (your provided file)
from organize_decoded_with_segments import (
    OrganizedDataset,
    extract_syllable_order,
    calculate_syllable_durations_ms,
    parse_json_safe,
)


# -----------------------------
# Parsing helpers
# -----------------------------
def _coerce_listlike(x) -> List[str]:
    """Coerce 'syllable_order' cell into list[str]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [str(v) for v in list(x)]
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        # JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj]
        except Exception:
            pass
        # python literal
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


# -----------------------------
# Entropy
# -----------------------------
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


def follower_counts_for_target(
    sequences: Sequence[Sequence[str]],
    target: str,
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


def compute_H_a_from_counts(
    follower_counts: Dict[str, int],
    log_base: float = 2.0,
) -> Tuple[float, Dict[str, float], Optional[float], int]:
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


# -----------------------------
# Treatment date lookup (NEW)
# -----------------------------
def infer_animal_id_from_merged_df(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer Animal ID from merged_df. Prefer 'Animal ID' column; fallback to file_name prefix.
    """
    if "Animal ID" in df.columns:
        vals = df["Animal ID"].dropna().astype(str)
        if not vals.empty:
            uniq = vals.unique()
            if len(uniq) == 1:
                return uniq[0]
            # if multiple, pick the most common
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
    Read an Excel metadata sheet and return the Treatment date for the given animal_id.
    Uses columns (case-insensitive):
      - 'Animal ID'
      - 'Treatment date' (or variants)
    If multiple treatment dates exist (e.g., repeated rows per injection),
    returns the earliest date and prints a note.
    """
    metadata_xlsx = Path(metadata_xlsx)
    if not metadata_xlsx.exists():
        raise FileNotFoundError(metadata_xlsx)

    xls = pd.ExcelFile(metadata_xlsx)

    # Prefer user-specified; else try common sheets first
    preferred = []
    if sheet_name:
        preferred.append(sheet_name)
    for s in ["metadata_with_hit_type", "metadata", "animal_hit_type_summary"]:
        if s in xls.sheet_names and s not in preferred:
            preferred.append(s)
    # fallback: all sheets
    for s in xls.sheet_names:
        if s not in preferred:
            preferred.append(s)

    for sname in preferred:
        df = pd.read_excel(metadata_xlsx, sheet_name=sname)

        animal_col = _find_col_case_insensitive(df, ["Animal ID", "animal_id", "AnimalID"])
        date_col = _find_col_case_insensitive(df, ["Treatment date", "Treatment Date", "treatment_date", "Treatment_date"])

        if not animal_col or not date_col:
            continue

        sub = df[df[animal_col].astype(str) == str(animal_id)].copy()
        if sub.empty:
            continue

        dates = pd.to_datetime(sub[date_col], errors="coerce").dropna().dt.normalize().unique()
        if len(dates) == 0:
            continue

        dates_sorted = sorted(pd.to_datetime(dates))
        if len(dates_sorted) > 1:
            print(f"[metadata lookup] Multiple treatment dates found for {animal_id} in sheet '{sname}': "
                  f"{[d.date() for d in dates_sorted]}. Using earliest.")
        return pd.to_datetime(dates_sorted[0]).normalize()

    raise ValueError(
        f"Could not find a treatment date for animal_id='{animal_id}' in {metadata_xlsx} "
        f"(searched sheets: {preferred})."
    )


def resolve_treatment_date(
    *,
    treatment_date_arg: Optional[str],
    metadata_xlsx: Optional[Union[str, Path]],
    animal_id_arg: Optional[str],
    merged_df: pd.DataFrame,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.Timestamp, str]:
    """
    Returns (treatment_date_timestamp, animal_id_used)
    """
    if treatment_date_arg:
        t = pd.to_datetime(treatment_date_arg, errors="raise").normalize()
        # still try to infer animal_id for printing
        animal_id_used = animal_id_arg or infer_animal_id_from_merged_df(merged_df) or "UNKNOWN"
        return t, animal_id_used

    if not metadata_xlsx:
        raise ValueError("Provide either --treatment-date OR --metadata-xlsx (to look up treatment date).")

    animal_id_used = animal_id_arg or infer_animal_id_from_merged_df(merged_df)
    if not animal_id_used:
        raise ValueError(
            "Could not infer animal_id from merged data. Please pass --animal-id explicitly."
        )

    t = lookup_treatment_date_from_xlsx(metadata_xlsx, animal_id_used, sheet_name=sheet_name)
    return t, animal_id_used


# -----------------------------
# Merge + organize (from merged data)
# -----------------------------
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

    # Ensure Date exists and is normalized datetime
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
    Build an OrganizedDataset from the *merged* dataframe.
    Adds: syllables_present, syllable_order, syllable_<label>_durations.
    """
    df = merged_df.copy()

    if "syllable_onsets_offsets_ms_dict" not in df.columns:
        if "syllable_onsets_offsets_ms" in df.columns:
            df["syllable_onsets_offsets_ms_dict"] = df["syllable_onsets_offsets_ms"].apply(parse_json_safe)
        else:
            df["syllable_onsets_offsets_ms_dict"] = [{} for _ in range(len(df))]

    if "Date" not in df.columns:
        if "Recording DateTime" in df.columns:
            df["Date"] = pd.to_datetime(df["Recording DateTime"], errors="coerce").dt.normalize()
        else:
            df["Date"] = pd.NaT
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    if add_recording_datetime and "Recording DateTime" in df.columns:
        df["Recording DateTime"] = pd.to_datetime(df["Recording DateTime"], errors="coerce")

    # unique labels across merged data
    label_set: set[str] = set()
    for d in df["syllable_onsets_offsets_ms_dict"]:
        if isinstance(d, dict) and d:
            label_set.update([str(k) for k in d.keys()])
    unique_labels = sorted(label_set)

    df["syllables_present"] = df["syllable_onsets_offsets_ms_dict"].apply(
        lambda d: sorted([str(k) for k in d.keys()]) if isinstance(d, dict) else []
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


# -----------------------------
# pre/post split + entropy
# -----------------------------
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
    pre_df, post_df = split_pre_post(organized_df, treatment_date=treatment_date, include_treatment_day=include_treatment_day)

    pre_seqs = get_syllable_order_sequences(pre_df, col=order_col) if not pre_df.empty else []
    post_seqs = get_syllable_order_sequences(post_df, col=order_col) if not post_df.empty else []

    pre_counts = follower_counts_for_target(pre_seqs, target=target, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)
    post_counts = follower_counts_for_target(post_seqs, target=target, collapse_repeats=collapse_repeats, ignore_labels=ignore_labels)

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


# -----------------------------
# Plotting
# -----------------------------
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


# -----------------------------
# CLI
# -----------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute H_a for one syllable pre vs post (treatment date via arg OR metadata Excel).")
    p.add_argument("--song-detection", "-d", type=str, required=True, help="Path to song_detection JSON")
    p.add_argument("--decoded", "-a", type=str, required=True, help="Path to decoded_database JSON")

    # Treatment date options (NEW)
    p.add_argument("--treatment-date", type=str, default=None, help="Treatment date, e.g. 2024-04-03 (optional if using metadata-xlsx)")
    p.add_argument("--metadata-xlsx", type=str, default=None, help="Path to metadata Excel to look up treatment date (optional if passing treatment-date)")
    p.add_argument("--metadata-sheet", type=str, default=None, help="Optional sheet name (default tries metadata_with_hit_type then metadata)")
    p.add_argument("--animal-id", type=str, default=None, help="Animal ID to use for metadata lookup (optional; otherwise inferred)")

    p.add_argument("--syllable", type=str, required=True, help="Target syllable label a (string)")

    # merge params
    p.add_argument("--max-gap-ms", type=int, default=500, help="Max gap between segments to consider split-up")
    p.add_argument("--segment-index-offset", type=int, default=0, help="Offset between detection vs annotation segment indices")
    p.add_argument("--merge-repeats", action="store_true", help="Coalesce adjacent repeats within repeat-gap-ms during merge")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0, help="Gap threshold for coalescing repeats (ms)")
    p.add_argument("--repeat-gap-inclusive", action="store_true", help="Use <= repeat-gap-ms instead of <")

    # entropy options
    p.add_argument("--include-treatment-day", action="store_true", help="Include treatment day in BOTH pre and post")
    p.add_argument("--no-collapse-repeats", action="store_true", help="Do NOT collapse consecutive repeats in syllable_order before counting transitions")
    p.add_argument("--ignore-labels", type=str, default="", help="Comma-separated labels to ignore (e.g. 'silence,-')")
    p.add_argument("--log-base", type=str, default="2", help="Entropy log base: '2' (bits), 'e' (nats), or numeric like '10'")

    # outputs
    p.add_argument("--out-dir", type=str, default="", help="If set, save follower plot + results JSON + optional CSVs here")
    p.add_argument("--save-merged-csv", action="store_true", help="Save merged annotations CSV to out-dir")
    p.add_argument("--save-organized-csv", action="store_true", help="Save organized-from-merged CSV to out-dir")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ignore = [s.strip() for s in args.ignore_labels.split(",") if s.strip() != ""]
    collapse = not args.no_collapse_repeats

    if args.log_base.lower() == "e":
        log_base = np.e
        base_name = "ln"
    else:
        log_base = float(args.log_base)
        base_name = "log2" if log_base == 2.0 else f"log{log_base:g}"

    merged_df = build_merged_annotations_df(
        decoded_database_json=args.decoded,
        song_detection_json=args.song_detection,
        max_gap_between_song_segments=args.max_gap_ms,
        segment_index_offset=args.segment_index_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
    )

    # ✅ Resolve treatment date from arg OR metadata Excel
    treatment_dt, animal_id_used = resolve_treatment_date(
        treatment_date_arg=args.treatment_date,
        metadata_xlsx=args.metadata_xlsx,
        animal_id_arg=args.animal_id,
        merged_df=merged_df,
        sheet_name=args.metadata_sheet,
    )

    organized_ds = build_organized_from_merged_df(
        merged_df,
        compute_durations=True,
        add_recording_datetime=True,
        treatment_date=treatment_dt,
    )
    organized_df = organized_ds.organized_df

    pre, post = compute_pre_post_Ha(
        organized_df,
        target=str(args.syllable),
        treatment_date=treatment_dt,
        include_treatment_day=args.include_treatment_day,
        collapse_repeats=collapse,
        ignore_labels=ignore,
        log_base=log_base,
        order_col="syllable_order",
    )

    print("\n--- Entropy by syllable (H_a) ---")
    print(f"Animal ID = {animal_id_used}")
    print(f"Target a = {pre.target}")
    print(f"Entropy base = {base_name}")
    print(f"Treatment date = {treatment_dt.date()} (include_treatment_day={bool(args.include_treatment_day)})")

    def _print_res(r: EntropyResult) -> None:
        print(f"\n{r.group.upper()}:")
        print(f"  H_a = {r.entropy:.6g}")
        print(f"  n transitions from a = {r.n_transitions}")
        print(f"  distinct followers observed = {r.distinct_followers}")
        if r.max_entropy_if_uniform_over_observed_followers is not None:
            print(f"  max H_a if uniform over observed followers = {r.max_entropy_if_uniform_over_observed_followers:.6g}")

        top = sorted(r.follower_probs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        if top:
            print("  top followers (a -> b : P):")
            for b, p in top:
                print(f"    {r.target} -> {b}: {p:.4f}")

    _print_res(pre)
    _print_res(post)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.save_merged_csv:
            merged_csv = out_dir / "decoded_with_split_labels.csv"
            merged_df.to_csv(merged_csv, index=False)
            print(f"\nSaved merged annotations CSV: {merged_csv}")

        if args.save_organized_csv:
            org_csv = out_dir / "organized_from_merged.csv"
            organized_df.to_csv(org_csv, index=False)
            print(f"Saved organized-from-merged CSV: {org_csv}")

        plot_path = out_dir / f"followers_pre_post__a_{pre.target}.png"
        plot_followers_pre_post(pre, post, plot_path, title=f"Followers for a={pre.target} (pre vs post)")
        print(f"Saved follower plot: {plot_path}")

        results = {
            "animal_id": animal_id_used,
            "target": pre.target,
            "treatment_date": str(treatment_dt.date()),
            "treatment_date_source": ("arg" if args.treatment_date else "metadata_xlsx"),
            "metadata_xlsx": args.metadata_xlsx,
            "metadata_sheet": args.metadata_sheet,
            "log_base": ("e" if log_base == np.e else log_base),
            "include_treatment_day": bool(args.include_treatment_day),
            "collapse_consecutive_repeats": bool(collapse),
            "ignore_labels": ignore,
            "merge_params": {
                "max_gap_ms": int(args.max_gap_ms),
                "segment_index_offset": int(args.segment_index_offset),
                "merge_repeats": bool(args.merge_repeats),
                "repeat_gap_ms": float(args.repeat_gap_ms),
                "repeat_gap_inclusive": bool(args.repeat_gap_inclusive),
            },
            "organizer_summary": {
                "unique_dates": organized_ds.unique_dates,
                "unique_syllable_labels_n": len(organized_ds.unique_syllable_labels),
                "treatment_date_fmt": organized_ds.treatment_date,
            },
            "pre": asdict(pre),
            "post": asdict(post),
        }
        out_json = out_dir / f"Ha_pre_post__a_{pre.target}.json"
        out_json.write_text(json.dumps(results, indent=2))
        print(f"Saved results JSON: {out_json}")


if __name__ == "__main__":
    main()


"""
from pathlib import Path
import importlib
import entropy_by_syllable as ebs
importlib.reload(ebs)

song_detection = Path("/path/to/song_detection.json")
decoded = Path("/path/to/decoded_database.json")

merged_df = ebs.build_merged_annotations_df(
    decoded_database_json=decoded,
    song_detection_json=song_detection,
    merge_repeated_syllables=True,
    repeat_gap_ms=10,
)

treatment_dt, animal_id_used = ebs.resolve_treatment_date(
    treatment_date_arg="2024-04-03",
    metadata_xlsx=None,
    animal_id_arg=None,
    merged_df=merged_df,
)

organized_df = ebs.build_organized_from_merged_df(merged_df, treatment_date=treatment_dt).organized_df

pre, post = ebs.compute_pre_post_Ha(
    organized_df,
    target="9",
    treatment_date=treatment_dt,
)

print(animal_id_used, pre.entropy, post.entropy)


"""