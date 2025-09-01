# set_hour_and_batch_TTE.py
from __future__ import annotations
"""
Per-day aggregated TTE for first-N from time range 1 and last-N from time range 2.

Behavior:
  • Excludes any file whose syllable_order contains only ONE unique label
    if exclude_single_label=True (e.g., ['A','A','A'] gets excluded).
  • For each calendar day:
      - Select the first N files within Range1 and the last N files within Range2 (after filtering).
      - Aggregate transitions across those N files (per range) and compute TTE.
      - Plot a paired line plot (two x positions: Range1-firstN vs Range2-lastN),
        one line per day, with a legend labeling each day.
      - ALSO plot an AM/PM (Range1/Range2) trend across calendar days.

Returns:
  BatchTTEByDayResult with:
    - summary_df (day, n_r1, n_r2, tte_r1, tte_r2)
    - figure_path (paired points figure)
    - trend_figure_path (AM/PM trend across days)

Example:
--------
from pathlib import Path
from set_hour_and_batch_TTE import run_set_hour_and_batch_TTE_by_day

decoded = "/path/to/..._decoded_database.json"
meta    = "/path/to/..._metadata.json"

res = run_set_hour_and_batch_TTE_by_day(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    range1="05:00-07:00",
    range2="14:00-17:00",
    batch_size=10,
    min_required_per_range=1,  # require at least this many per range to include day
    only_song_present=True,
    exclude_single_label=True,
    save_dir=Path("./figures"),
    show=True,
)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import (segments-aware preferred; fall back to durations builder)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from organize_decoded_with_segments import (
        build_organized_segments_with_durations as build_organized_dataset_with_durations
    )
except ImportError:
    from organize_decoded_with_durations import (
        build_organized_dataset_with_durations  # type: ignore
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_pad(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return "00"
    return str(s).zfill(2)

def _parse_time_range(spec: str) -> Tuple[int, int]:
    """
    '5:00-7:00' or '05:00-07:00' → (start_min, end_min), half-open [start, end)
    """
    spec = spec.strip().replace(" ", "")
    if "-" not in spec:
        raise ValueError(f"Bad time range {spec!r}; expected 'HH:MM-HH:MM'.")
    a, b = spec.split("-", 1)
    def to_minutes(hhmm: str) -> int:
        if ":" not in hhmm:
            raise ValueError(f"Bad time token {hhmm!r}; expected 'HH:MM'.")
        hh, mm = hhmm.split(":")
        h, m = int(hh), int(mm)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"Hour/min out of range in {hhmm!r}.")
        return h * 60 + m
    start, end = to_minutes(a), to_minutes(b)
    if end <= start:
        raise ValueError(f"End must be after start in range {spec!r}.")
    return start, end

def _minutes_since_midnight(dt: pd.Timestamp) -> int:
    return int(dt.hour) * 60 + int(dt.minute)

def _bigrams(labels: Iterable[str]) -> List[Tuple[str, str]]:
    arr = list(labels)
    return [(arr[i], arr[i+1]) for i in range(len(arr) - 1)]

def _row_transition_frequencies(syllable_order: Iterable[str]) -> Dict[Tuple[str, str], int]:
    if not syllable_order:
        return {}
    from collections import Counter
    return dict(Counter(_bigrams(list(syllable_order))))

def _build_per_file_transitions(
    organized_df: pd.DataFrame,
    *,
    exclude_single_label: bool = True,
) -> pd.DataFrame:
    """
    Returns columns: date (date), date_time (Timestamp), transition_frequencies (dict).
    Keeps only rows with a valid timestamp and at least one transition.
    If exclude_single_label=True, also require >=2 unique labels in syllable_order.
    """
    df = organized_df.copy()
    for c in ["Date", "Hour", "Minute", "Second", "syllable_order"]:
        if c not in df.columns:
            df[c] = None

    # Compose timestamps (fallback 00 for missing H/M/S)
    dt_list = []
    for _, r in df.iterrows():
        d = r.get("Date", pd.NaT)
        if pd.isna(d):
            dt_list.append(pd.NaT)
            continue
        hh = _safe_pad(r.get("Hour")); mm = _safe_pad(r.get("Minute")); ss = _safe_pad(r.get("Second"))
        try:
            dt_list.append(pd.to_datetime(f"{pd.to_datetime(d).strftime('%Y-%m-%d')} {hh}:{mm}:{ss}"))
        except Exception:
            dt_list.append(pd.NaT)
    df["date_time"] = dt_list
    df["date"] = pd.to_datetime(df["date_time"]).dt.date

    # Build per-file transitions + unique label count
    df["transition_frequencies"] = df["syllable_order"].apply(_row_transition_frequencies)
    def _uniq_count(obj) -> int:
        try:
            return len(set(obj)) if obj is not None else 0
        except Exception:
            return 0
    df["n_unique_labels"] = df["syllable_order"].apply(_uniq_count)

    # Base validity: has timestamp + at least one transition
    ok = (~df["date_time"].isna()) & df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0)

    # Additional exclusion: only one unique label present
    if exclude_single_label:
        ok &= df["n_unique_labels"] >= 2

    filtered = df.loc[ok, ["date", "date_time", "transition_frequencies"]].reset_index(drop=True)
    return filtered

# ──────────────────────────────────────────────────────────────────────────────
# Entropy utilities
# ──────────────────────────────────────────────────────────────────────────────

def _generate_normalized_transition_matrix(
    transition_list: List[Tuple[str, str]],
    transition_counts: List[int],
    syllable_types: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(syllable_types)
    counts_mat = np.zeros((n, n), dtype=float)
    idx = {lab: i for i, lab in enumerate(syllable_types)}
    for (frm, to), cnt in zip(transition_list, transition_counts):
        if frm in idx and to in idx:
            counts_mat[idx[frm], idx[to]] += cnt
    row_sums = counts_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_mat = counts_mat / row_sums
    return norm_mat, counts_mat

def _calculate_transition_entropy(transition_matrix: np.ndarray, syllable_types: List[str]) -> Dict[str, float]:
    ent: Dict[str, float] = {}
    for i, lab in enumerate(syllable_types):
        probs = transition_matrix[i, :]
        probs = probs[probs > 0]
        ent[lab] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    return ent

def _calculate_total_transition_entropy(
    transition_counts_matrix: np.ndarray,
    syllable_types: List[str],
    transition_entropies: Dict[str, float],
) -> float:
    row_sums = transition_counts_matrix.sum(axis=1)
    total = row_sums.sum()
    probs = (row_sums / total) if total > 0 else np.zeros_like(row_sums)
    te = 0.0
    for i, lab in enumerate(syllable_types):
        te += float(probs[i]) * float(transition_entropies.get(lab, 0.0))
    return float(te)

def _aggregate_counts(dicts: Iterable[Dict[Tuple[str, str], int]]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for d in dicts:
        for tr, cnt in d.items():
            out[tr] = out.get(tr, 0) + cnt
    return out

def _tte_from_counts(counts: Dict[Tuple[str, str], int], syllable_types: List[str]) -> float:
    if not counts:
        return 0.0
    tlist = list(counts.keys())
    tcnts = list(counts.values())
    norm, cm = _generate_normalized_transition_matrix(tlist, tcnts, syllable_types)
    ent_by_syll = _calculate_transition_entropy(norm, syllable_types)
    return _calculate_total_transition_entropy(cm, syllable_types, ent_by_syll)

def _clean_for_filename(s: str) -> str:
    """Helper to sanitize strings for filenames."""
    return s.replace(":", "").replace("-", "_")

# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PerDayAggregate:
    day: pd.Timestamp
    n_r1: int
    n_r2: int
    tte_r1: float
    tte_r2: float

@dataclass
class BatchTTEByDayResult:
    animal_id: str
    summary_df: pd.DataFrame   # columns: day, n_r1, n_r2, tte_r1, tte_r2
    figure_path: Optional[Path] = None        # paired points figure
    trend_figure_path: Optional[Path] = None  # AM/PM trend across days

# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────

def run_set_hour_and_batch_TTE_by_day(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    range1: str,
    range2: str,
    batch_size: int,
    min_required_per_range: int = 1,  # require at least this many in each range to include day
    only_song_present: bool = False,
    exclude_single_label: bool = True,  # default True
    compute_durations: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> BatchTTEByDayResult:
    """
    For each day:
      - Select first `batch_size` files in Range1 and last `batch_size` files in Range2
        **after excluding single-label files** (if exclude_single_label=True).
      - Aggregate transitions per range and compute TTE using a GLOBAL syllable set
        built from all selected files across all days/ranges (consistent axes).
      - Plot paired points (Range1 vs Range2) with lines connecting each day's pair,
        and include a legend labeling each day.
      - ALSO plot an AM/PM (Range1/Range2) trend across calendar days.
    """
    # Build dataset
    od = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
    )
    organized = od.organized_df

    # Animal ID for titles/filenames
    animal_id = "unknown"
    if "Animal ID" in organized.columns:
        try:
            ids = organized["Animal ID"].dropna().astype(str).unique()
            if len(ids) == 1 and ids[0]:
                animal_id = ids[0]
        except Exception:
            pass

    # Per-file transitions (with single-label exclusion applied)
    pf = _build_per_file_transitions(organized, exclude_single_label=exclude_single_label)
    if pf.empty:
        print("No valid per-file transitions found after filtering.")
        return BatchTTEByDayResult(animal_id, pd.DataFrame(), None, None)

    # Parse ranges
    s1, e1 = _parse_time_range(range1)
    s2, e2 = _parse_time_range(range2)

    # Compute minutes since midnight and add day index
    pf["tod_min"] = pf["date_time"].apply(_minutes_since_midnight)

    # Select per day
    per_day = []
    selected_all_counts: List[Dict[Tuple[str, str], int]] = []  # for global syllable set

    for day, grp in pf.groupby("date"):
        r1 = grp[(grp["tod_min"] >= s1) & (grp["tod_min"] < e1)].sort_values("date_time")
        r2 = grp[(grp["tod_min"] >= s2) & (grp["tod_min"] < e2)].sort_values("date_time")

        first_r1 = r1.head(batch_size)
        last_r2  = r2.tail(batch_size)

        if len(first_r1) < min_required_per_range or len(last_r2) < min_required_per_range:
            # Skip this day if not enough data in either range
            continue

        counts_r1 = _aggregate_counts(first_r1["transition_frequencies"])
        counts_r2 = _aggregate_counts(last_r2["transition_frequencies"])

        # Save for building global syllable set later
        selected_all_counts.append(counts_r1)
        selected_all_counts.append(counts_r2)

        per_day.append((pd.to_datetime(day), len(first_r1), len(last_r2), counts_r1, counts_r2))

    if not per_day:
        print("No days met the selection criteria.")
        return BatchTTEByDayResult(animal_id, pd.DataFrame(), None, None)

    # Build a GLOBAL syllable set over all selected counts
    global_labels = set()
    for d in selected_all_counts:
        for (a, b) in d.keys():
            global_labels.add(a); global_labels.add(b)
    syllable_types = sorted(global_labels)
    if not syllable_types:
        print("Selected data had no transitions.")
        return BatchTTEByDayResult(animal_id, pd.DataFrame(), None, None)

    # Compute TTE per day using the global axes
    rows: List[PerDayAggregate] = []
    for day_ts, n1, n2, c1, c2 in per_day:
        tte1 = _tte_from_counts(c1, syllable_types)
        tte2 = _tte_from_counts(c2, syllable_types)
        rows.append(PerDayAggregate(day=day_ts, n_r1=n1, n_r2=n2, tte_r1=tte1, tte_r2=tte2))

    summary_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("day").reset_index(drop=True)

    # ── PLOT 1: Paired points with lines (one line per day) ───────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = [0, 1]
    x_labels = [f"{range1}\n(first N)", f"{range2}\n(last N)"]

    handles, labels = [], []
    for _, row in summary_df.iterrows():
        lbl = pd.to_datetime(row["day"]).strftime("%Y-%m-%d")
        h, = ax.plot(x_positions, [row["tte_r1"], row["tte_r2"]],
                     marker="o", linewidth=1.8, alpha=0.9, label=lbl)
        handles.append(h); labels.append(lbl)

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(f"{animal_id} – Per-day aggregated TTE\n"
                 f"First {batch_size} in {range1} vs Last {batch_size} in {range2}")
    ax.grid(False)
    # Hide top & right spines
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)

    # Legend outside on the right
    n = len(labels)
    ncol = 1 if n <= 14 else 2 if n <= 28 else 3
    ax.legend(handles, labels, title="Day",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.,
              frameon=False, ncol=ncol, fontsize=9)

    plt.tight_layout()

    fig_path: Optional[Path] = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"{animal_id}_PerDay_TTE_{_clean_for_filename(range1)}__{_clean_for_filename(range2)}__N{batch_size}.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved figure → {fig_path}")

        # ── PLOT 2: Sequential AM/PM labels per day (Day 1 AM, Day 1 PM, ...) ────
        fig2, ax2 = plt.subplots(figsize=(12, 6))
    
        # Pick friendly window names when they match AM/PM, otherwise show the spec
        def _win_name(start_min: int, end_min: int, spec: str) -> str:
            if start_min == 0 and end_min <= 720:    # 00:00–12:00
                return "AM"
            if start_min >= 720:                     # 12:00–24:00
                return "PM"
            return spec
    
        name_r1 = _win_name(s1, e1, range1)
        name_r2 = _win_name(s2, e2, range2)
    
        # Build interleaved labels/values: Day 1 AM, Day 1 PM, Day 2 AM, Day 2 PM, ...
        labels_seq, y_seq = [], []
        for i, row in summary_df.reset_index(drop=True).iterrows():
            day_idx = i + 1
            labels_seq += [f"Day {day_idx} {name_r1}", f"Day {day_idx} {name_r2}"]
            y_seq += [row["tte_r1"], row["tte_r2"]]
    
        x = np.arange(len(labels_seq))
        ax2.plot(x, y_seq, marker="o", linewidth=1.8, alpha=0.9)
    
        ax2.set_xlabel("Day / Window")
        ax2.set_ylabel("Total Transition Entropy (bits)")
        ax2.set_title(
            f"{animal_id} – AM/PM TTE sequence by day\n"
            f"First {batch_size} in {range1} vs Last {batch_size} in {range2}"
        )
    
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_seq, rotation=90, ha="right")
        ax2.grid(False)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        ax2.tick_params(top=False, right=False)
        plt.tight_layout()
    
        trend_fig_path: Optional[Path] = None
        if save_dir:
            trend_fig_path = (
                Path(save_dir) /
                f"{animal_id}_PerDay_TTE_AMPM_SEQ_{_clean_for_filename(range1)}__{_clean_for_filename(range2)}__N{batch_size}.png"
            )
            fig2.savefig(trend_fig_path, dpi=200)
            print(f"Saved sequential AM/PM figure → {trend_fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/..._decoded_database.json"
    meta    = "/path/to/..._metadata.json"
    _ = run_set_hour_and_batch_TTE_by_day(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        range1="05:00-07:00",
        range2="14:00-17:00",
        batch_size=10,
        min_required_per_range=1,
        only_song_present=True,
        exclude_single_label=True,  # exclude files with only one unique label
        save_dir=Path("/figures"),
        show=True,
    )


"""
# Paths to your decoded + metadata files
from pathlib import Path
import importlib, set_hour_and_batch_TTE   # import the module itself
importlib.reload(set_hour_and_batch_TTE)   # force Python to use the new code

from set_hour_and_batch_TTE import run_set_hour_and_batch_TTE_by_day
res = run_set_hour_and_batch_TTE_by_day(
    decoded_database_json="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/USA1234_decoded_database.json",
    creation_metadata_json="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/USA1234_metadata.json",
    range1="00:00-12:00",
    range2="12:00-23:59",
    batch_size=5,
    min_required_per_range=1,
    only_song_present=True,
    exclude_single_label=False,
    save_dir=Path("./figures"),
    show=True,
)
print("paired:", res.figure_path)
print("trend :", res.trend_figure_path)


"""