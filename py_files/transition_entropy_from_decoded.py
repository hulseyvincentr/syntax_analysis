# transition_entropy_from_decoded.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # only used to match your original import; layout not strictly needed

# ──────────────────────────────────────────────────────────────────────────────
# 1) Bring in your organizer
# ──────────────────────────────────────────────────────────────────────────────
from organize_decoded_with_durations import build_organized_dataset_with_durations

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helpers to convert organized_df → per-file transition dicts
# ──────────────────────────────────────────────────────────────────────────────

def bigrams(labels: Iterable[str]) -> List[Tuple[str, str]]:
    """Return ordered adjacent pairs: (l0->l1), (l1->l2), ..."""
    arr = list(labels)
    return [(arr[i], arr[i+1]) for i in range(len(arr) - 1)]

def row_transition_frequencies(syllable_order: Iterable[str]) -> Dict[Tuple[str, str], int]:
    """Count bigram transitions within a file."""
    if not syllable_order:
        return {}
    return dict(Counter(bigrams(list(syllable_order))))

def safe_pad(s: Optional[str]) -> str:
    """Zero-pad a possibly missing time component."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return "00"
    s = str(s)
    return s.zfill(2)

def build_annotation_dataframe(organized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a slim DataFrame with:
        - date_time: datetime64 (combining Date + Hour/Minute/Second)
        - transition_frequencies: dict[(from, to) -> count] per file
    Drops rows that cannot be timestamped or have no transitions.
    """
    df = organized_df.copy()

    # Ensure needed cols exist
    for c in ["Date", "Hour", "Minute", "Second", "syllable_order"]:
        if c not in df.columns:
            df[c] = None

    # Build a datetime per file; fallback to midnight if H/M/S missing
    # df["Date"] is already datetime64[ns] in your organizer (or NaT if unknown)
    out_dt = []
    for _, r in df.iterrows():
        d = r.get("Date", pd.NaT)
        if pd.isna(d):
            out_dt.append(pd.NaT)
            continue
        hh = safe_pad(r.get("Hour"))
        mm = safe_pad(r.get("Minute"))
        ss = safe_pad(r.get("Second"))
        try:
            out_dt.append(pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {hh}:{mm}:{ss}"))
        except Exception:
            out_dt.append(pd.NaT)

    df["date_time"] = out_dt

    # Build transition frequency dict per file from syllable_order
    df["transition_frequencies"] = df["syllable_order"].apply(row_transition_frequencies)

    # Keep only rows with valid timestamp and at least one transition
    ok = (~df["date_time"].isna()) & (df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0))
    return df.loc[ok, ["date_time", "transition_frequencies"]].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Entropy utilities (your original math, wrapped)
# ──────────────────────────────────────────────────────────────────────────────

def generate_normalized_transition_matrix(
    transition_list: List[Tuple[str, str]],
    transition_counts: List[int],
    syllable_types: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        normalized_transition_matrix: rows sum to 1 (unless row had all zeros),
        transition_counts_matrix: raw counts.
    """
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

def calculate_transition_entropy(transition_matrix: np.ndarray, syllable_types: List[str]) -> Dict[str, float]:
    """
    Per-syllable outgoing entropy H_b = - Σ p * log2 p for nonzero row probabilities.
    """
    ent: Dict[str, float] = {}
    for i, lab in enumerate(syllable_types):
        probs = transition_matrix[i, :]
        probs = probs[probs > 0]
        if probs.size:
            ent[lab] = float(-np.sum(probs * np.log2(probs)))
        else:
            ent[lab] = 0.0
    return ent

def calculate_total_transition_entropy(
    transition_counts_matrix: np.ndarray,
    syllable_types: List[str],
    transition_entropies: Dict[str, float],
) -> float:
    """
    Total TE = Σ_b P(b) * H_b, where P(b) is the fraction of all outgoing transitions
    that originate from syllable b (row-sum / total).
    """
    row_sums = transition_counts_matrix.sum(axis=1)
    total = row_sums.sum()
    if total > 0:
        probs = row_sums / total
    else:
        probs = np.zeros_like(row_sums)

    tot = 0.0
    for i, lab in enumerate(syllable_types):
        tot += probs[i] * transition_entropies.get(lab, 0.0)
    return float(tot)

def plot_total_transition_entropy(entropies_per_day: List[Tuple[date, float]], surgery_date: Optional[date]):
    days, entropies = zip(*sorted(entropies_per_day, key=lambda t: t[0])) if entropies_per_day else ([], [])
    if not days:
        print("No daily entropy values to plot.")
        return

    plt.figure(figsize=(20, 6))
    plt.scatter(days, entropies, s=100, edgecolor="black")  # default color

    if surgery_date is not None:
        plt.axvline(x=surgery_date, color="r", linestyle="--", label="Surgery Date")

    # Tick label formatting
    plt.xticks(ticks=days, labels=[pd.to_datetime(d).strftime("%Y-%m-%d") for d in days], rotation=90)

    plt.title("Total Transition Entropy Across Days", fontsize=16)
    plt.xlabel("Day", fontsize=14)
    plt.ylabel("Total Transition Entropy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if surgery_date is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main analysis (build → per-file transitions → per-day entropies)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_transitions_for_each_day_from_decoded(
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path,
    *,
    only_song_present: bool = False,
    compute_durations: bool = False,   # durations not needed for entropy; you can turn off
    surgery_date_override: Optional[str] = None,  # "YYYY-MM-DD" or None → use metadata's treatment_date
):
    """
    High-level driver that:
      1) builds the organized dataset,
      2) constructs per-file transition frequency dicts,
      3) aggregates by day,
      4) computes + plots total transition entropy per day.

    Returns: (organized_df, per_file_df, entropies_per_day, syllable_types)
    """

    # 1) Build the organized dataset
    od = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
    )

    organized = od.organized_df

    # Resolve surgery/treatment date
    # Prefer explicit override; else use od.treatment_date ("YYYY.MM.DD") if available
    if surgery_date_override:
        try:
            surgery_dt = pd.to_datetime(surgery_date_override).date()
        except Exception:
            surgery_dt = None
    else:
        if od.treatment_date:
            try:
                surgery_dt = datetime.strptime(od.treatment_date, "%Y.%m.%d").date()
            except Exception:
                surgery_dt = None
        else:
            surgery_dt = None

    # 2) Build slim per-file annotation dataframe: date_time + transition_frequencies
    per_file_df = build_annotation_dataframe(organized)

    if per_file_df.empty:
        print("No valid per-file transitions could be built (check dates or syllable_order).")
        return organized, per_file_df, [], []

    # 3) Determine a global, consistent set of syllable types from all transitions
    all_transition_frequencies: Dict[Tuple[str, str], int] = {}
    for freq_dict in per_file_df["transition_frequencies"]:
        for tr, cnt in freq_dict.items():
            all_transition_frequencies[tr] = all_transition_frequencies.get(tr, 0) + cnt

    syllable_types = sorted(set([t[0] for t in all_transition_frequencies] + [t[1] for t in all_transition_frequencies]))

    # 4) Aggregate by day and compute entropies
    per_file_df["day"] = per_file_df["date_time"].dt.date
    unique_days = sorted(per_file_df["day"].unique())

    entropies_per_day: List[Tuple[date, float]] = []
    for d in unique_days:
        day_df = per_file_df[per_file_df["day"] == d]

        day_counts: Dict[Tuple[str, str], int] = {}
        for freq_dict in day_df["transition_frequencies"]:
            for tr, cnt in freq_dict.items():
                day_counts[tr] = day_counts.get(tr, 0) + cnt

        transition_list = list(day_counts.keys())
        transition_counts = list(day_counts.values())

        norm_mat, counts_mat = generate_normalized_transition_matrix(
            transition_list, transition_counts, syllable_types
        )
        ent_by_syll = calculate_transition_entropy(norm_mat, syllable_types)
        total_te = calculate_total_transition_entropy(counts_mat, syllable_types, ent_by_syll)
        entropies_per_day.append((d, total_te))

    # 5) Plot
    plot_total_transition_entropy(entropies_per_day, surgery_dt)

    return organized, per_file_df, entropies_per_day, syllable_types

# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fill these with your paths
    decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
    meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

    # Optionally override the surgery date (else we'll use creation metadata's treatment_date if present)
    surgery_override_iso = None  # e.g., "2024-06-28"

    organized_df, per_file_df, entropies_per_day, syllable_types = analyze_transitions_for_each_day_from_decoded(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        compute_durations=False,
        surgery_date_override=surgery_override_iso,
    )

    # Inspect results in your Variable Explorer if running in Spyder:
    # - organized_df: full table from the organizer
    # - per_file_df:   per-file date_time + transition_frequencies
    # - entropies_per_day: list[(date, total_entropy)]
    # - syllable_types: consistent label order used for matrices
