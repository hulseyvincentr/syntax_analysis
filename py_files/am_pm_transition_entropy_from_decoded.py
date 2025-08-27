# -*- coding: utf-8 -*-
# am_pm_transition_entropy_from_decoded.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Bring in your organizer
# ──────────────────────────────────────────────────────────────────────────────
from organize_decoded_with_segments import (
    build_organized_segments_with_durations as build_organized_dataset_with_durations
)
# ──────────────────────────────────────────────────────────────────────────────
# Helpers: build transitions from syllable_order
# ──────────────────────────────────────────────────────────────────────────────

def bigrams(labels: Iterable[str]) -> List[Tuple[str, str]]:
    """Adjacent ordered pairs: (l0->l1), (l1->l2), ..."""
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
        - date_time: datetime64 (combining Date + Hour:Minute:Second)
        - day: date
        - am_pm: "AM" (00:00–11:59) or "PM" (12:00–23:59)
        - transition_frequencies: dict[(from, to) -> count] per file
    Only keeps rows with a valid timestamp and at least one transition.
    """
    df = organized_df.copy()

    for c in ["Date", "Hour", "Minute", "Second", "syllable_order"]:
        if c not in df.columns:
            df[c] = None

    # Build a datetime per file (fallback to 00:00:00 if H/M/S missing)
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
    df["day"] = df["date_time"].dt.date
    df["am_pm"] = np.where(df["date_time"].dt.hour < 12, "AM", "PM")

    df["transition_frequencies"] = df["syllable_order"].apply(row_transition_frequencies)

    ok = (
        ~df["date_time"].isna()
        & df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0)
    )
    return df.loc[ok, ["date_time", "day", "am_pm", "transition_frequencies"]].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Entropy utilities
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
    row_sums[row_sums == 0] = 1.0  # safe divide
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
    probs = (row_sums / total) if total > 0 else np.zeros_like(row_sums)

    tot = 0.0
    for i, lab in enumerate(syllable_types):
        tot += probs[i] * transition_entropies.get(lab, 0.0)
    return float(tot)

# ──────────────────────────────────────────────────────────────────────────────
# AM/PM analysis + plotting
# ──────────────────────────────────────────────────────────────────────────────

def analyze_am_pm_transitions_from_decoded(
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path,
    *,
    only_song_present: bool = False,
    compute_durations: bool = False,  # durations not required for entropy
    surgery_date_override: Optional[str] = None,  # "YYYY-MM-DD" or None → use metadata's treatment_date
    return_syllable_types: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Pipeline:
      1) Build organized dataset.
      2) Build per-file transitions with date_time + AM/PM tags.
      3) Compute daily AM and PM total transition entropies.
      4) Plot both series with an optional dashed red surgery date.

    Returns:
      organized_df,
      per_file_df (columns: date_time, day, am_pm, transition_frequencies),
      results_df (columns: day, am_pm, total_entropy)
      [+ syllable_types if return_syllable_types=True]
    """
    # 1) Build dataset
    od = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
    )
    organized = od.organized_df

    # Resolve surgery/treatment date
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

    # 2) Per-file transitions + AM/PM
    per_file_df = build_annotation_dataframe(organized)
    if per_file_df.empty:
        print("No valid per-file transitions (check dates or syllable_order).")
        if return_syllable_types:
            return organized, per_file_df, pd.DataFrame(), []
        return organized, per_file_df, pd.DataFrame()

    # 3) Global syllable set (consistent matrix axes)
    all_transition_frequencies: Dict[Tuple[str, str], int] = {}
    for freq_dict in per_file_df["transition_frequencies"]:
        for tr, cnt in freq_dict.items():
            all_transition_frequencies[tr] = all_transition_frequencies.get(tr, 0) + cnt
    syllable_types = sorted(set([t[0] for t in all_transition_frequencies] + [t[1] for t in all_transition_frequencies]))

    # Compute total TE for each (day, am_pm)
    results = []
    for (day, ampm), grp in per_file_df.groupby(["day", "am_pm"]):
        # Sum transitions over all files within that (day, AM/PM)
        counts: Dict[Tuple[str, str], int] = {}
        for freq_dict in grp["transition_frequencies"]:
            for tr, cnt in freq_dict.items():
                counts[tr] = counts.get(tr, 0) + cnt

        transition_list = list(counts.keys())
        transition_counts = list(counts.values())

        norm_mat, counts_mat = generate_normalized_transition_matrix(
            transition_list, transition_counts, syllable_types
        )
        ent_by_syll = calculate_transition_entropy(norm_mat, syllable_types)
        total_te = calculate_total_transition_entropy(counts_mat, syllable_types, ent_by_syll)
        results.append((day, ampm, total_te))

    results_df = pd.DataFrame(results, columns=["day", "am_pm", "total_entropy"]).sort_values(["day", "am_pm"]).reset_index(drop=True)

    # 4) Plot AM vs PM
    plot_am_pm_total_transition_entropy(results_df, surgery_dt)

    if return_syllable_types:
        return organized, per_file_df, results_df, syllable_types
    return organized, per_file_df, results_df

def plot_am_pm_total_transition_entropy(results_df: pd.DataFrame, surgery_date: Optional[date]):
    """
    Scatter both AM and PM series across days, optionally marking the surgery date.
    """
    if results_df.empty:
        print("No AM/PM entropy values to plot.")
        return

    # Ensure both series exist (some days may miss AM or PM)
    days_all = sorted(results_df["day"].unique())
    am = results_df[results_df["am_pm"] == "AM"].sort_values("day")
    pm = results_df[results_df["am_pm"] == "PM"].sort_values("day")

    plt.figure(figsize=(22, 7))
    # Scatter AM and PM; default colors; give distinct markers
    plt.scatter(am["day"], am["total_entropy"], marker="o", s=100, edgecolor="black", label="AM (00:00–11:59)")
    plt.scatter(pm["day"], pm["total_entropy"], marker="^", s=100, edgecolor="black", label="PM (12:00–23:59)")

    if surgery_date is not None:
        plt.axvline(x=surgery_date, color="r", linestyle="--", label="Surgery Date")

    plt.xticks(ticks=days_all, labels=[pd.to_datetime(d).strftime("%Y-%m-%d") for d in days_all], rotation=90)

    plt.title("Total Transition Entropy by Day and Half-Day", fontsize=16)
    plt.xlabel("Day", fontsize=14)
    plt.ylabel("Total Transition Entropy", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=12)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Update these to your paths
    decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5272/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5272_decoded_database.json"
    meta    = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5272/USA5272_metadata.json"

    # Optionally override the surgery date (else we'll use metadata's treatment_date if present)
    surgery_override_iso = None  # e.g., "2024-06-28"

    organized_df, per_file_df, ampm_entropy_df = analyze_am_pm_transitions_from_decoded(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        compute_durations=False,
        surgery_date_override=surgery_override_iso,
        return_syllable_types=False,
    )

    # You can save results if you like:
    # ampm_entropy_df.to_csv("am_pm_total_transition_entropy.csv", index=False)
"""
from am_pm_transition_entropy_from_decoded import analyze_am_pm_transitions_from_decoded

# Paths to your decoded + metadata files
decoded = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5507_RC4_Comp2/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5507_RC4_Comp2_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5507_RC4_Comp2/USA5507_RC4_Comp2_metadata.json"

# Optional: override the treatment/surgery date (otherwise pulled from metadata)
surgery_override_iso = None  # e.g., "2024-07-01"

# Run the analysis
organized_df, per_file_df, ampm_entropy_df = analyze_am_pm_transitions_from_decoded(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,   # set True to restrict to rows where song_present == True
    compute_durations=False,   # durations not needed for entropy
    surgery_date_override=surgery_override_iso,
    return_syllable_types=False,   # set True if you also want the global syllable label list back
)

# Inspect results
print("\nOrganized dataset shape:", organized_df.shape)
print("Per-file transitions shape:", per_file_df.shape)
print("AM/PM entropy summary:\n", ampm_entropy_df.head())

# If you want to save the entropy table for later analysis:
out_csv = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5507_RC4_Comp2/am_pm_total_transition_entropy.csv"
ampm_entropy_df.to_csv(out_csv, index=False)
print(f"Saved AM/PM entropy results → {out_csv}")


"""
