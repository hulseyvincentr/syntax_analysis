# transition_entropy_from_decoded.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # kept to match your original import; not strictly required

# ──────────────────────────────────────────────────────────────────────────────
# 1) Organizer import (prefer segments-aware; fall back to legacy durations)
# ──────────────────────────────────────────────────────────────────────────────
_USING_SEGMENTS = False
try:
    from organize_decoded_with_segments import (
        build_organized_segments_with_durations as _build_organized,
    )
    _USING_SEGMENTS = True
except ImportError:
    from organize_decoded_with_durations import (
        build_organized_dataset_with_durations as _build_organized,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2) Helpers to convert organized_df → per-file transition dicts
# ──────────────────────────────────────────────────────────────────────────────
def bigrams(labels: Iterable[str]) -> List[Tuple[str, str]]:
    """Return ordered adjacent pairs: (l0->l1), (l1->l2), ..."""
    arr = list(labels)
    return [(arr[i], arr[i + 1]) for i in range(len(arr) - 1)]


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
            out_dt.append(pd.to_datetime(f"{pd.to_datetime(d).strftime('%Y-%m-%d')} {hh}:{mm}:{ss}"))
        except Exception:
            out_dt.append(pd.NaT)

    df["date_time"] = out_dt
    df["transition_frequencies"] = df["syllable_order"].apply(row_transition_frequencies)

    ok = (
        ~df["date_time"].isna()
        & df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0)
    )
    return df.loc[ok, ["date_time", "transition_frequencies"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Entropy utilities
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
    """Per-syllable outgoing entropy H_b = - Σ p * log2 p for nonzero row probabilities."""
    ent: Dict[str, float] = {}
    for i, lab in enumerate(syllable_types):
        probs = transition_matrix[i, :]
        probs = probs[probs > 0]
        ent[lab] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
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


def plot_total_transition_entropy(
    entropies_per_day: List[Tuple[date, float]],
    surgery_date: Optional[date],
    save_path: Optional[Union[str, Path]] = None,
):
    days, entropies = zip(*sorted(entropies_per_day, key=lambda t: t[0])) if entropies_per_day else ([], [])
    if not days:
        print("No daily entropy values to plot.")
        return

    plt.figure(figsize=(20, 6))
    plt.scatter(days, entropies, s=100, edgecolor="black")

    if surgery_date is not None:
        plt.axvline(x=surgery_date, color="r", linestyle="--", label="Surgery Date")

    plt.xticks(
        ticks=days,
        labels=[pd.to_datetime(d).strftime("%Y-%m-%d") for d in days],
        rotation=90,
    )

    plt.title("Total Transition Entropy Across Days", fontsize=16)
    plt.xlabel("Day", fontsize=14)
    plt.ylabel("Total Transition Entropy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if surgery_date is not None:
        plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[transition_entropy] Saved plot → {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4) Main analysis (build → per-file transitions → per-day entropies)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_transitions_for_each_day_from_decoded(
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path,
    *,
    only_song_present: bool = False,
    compute_durations: bool = False,         # durations not required for entropy
    surgery_date_override: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
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
    if _USING_SEGMENTS:
        od = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=compute_durations,
            add_recording_datetime=True,   # ensure Date/Hour/Minute/Second are populated
        )
    else:
        od = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=compute_durations,
        )

    organized = od.organized_df

    # Best-effort animal ID
    try:
        animal_id = str(organized["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"

    # Resolve surgery/treatment date
    if surgery_date_override:
        try:
            surgery_dt = pd.to_datetime(surgery_date_override).date()
        except Exception:
            surgery_dt = None
    else:
        if getattr(od, "treatment_date", None):
            try:
                surgery_dt = datetime.strptime(od.treatment_date, "%Y.%m.%d").date()
            except Exception:
                surgery_dt = None
        else:
            surgery_dt = None

    # 2) Per-file transitions + timestamps
    per_file_df = build_annotation_dataframe(organized)
    if per_file_df.empty:
        print("No valid per-file transitions could be built (check dates or syllable_order).")
        return organized, per_file_df, [], []

    # 3) Global syllable set (consistent matrix axes)
    all_transition_frequencies: Dict[Tuple[str, str], int] = {}
    for freq_dict in per_file_df["transition_frequencies"]:
        for tr, cnt in freq_dict.items():
            all_transition_frequencies[tr] = all_transition_frequencies.get(tr, 0) + cnt
    syllable_types = sorted(set([t[0] for t in all_transition_frequencies] + [t[1] for t in all_transition_frequencies]))

    # 4) Aggregate by day → total transition entropy
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
    save_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{animal_id}_transition_entropy_daily.png"

    plot_total_transition_entropy(entropies_per_day, surgery_dt, save_path=save_path)

    return organized, per_file_df, entropies_per_day, syllable_types


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Update these with your paths
    decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5323_decoded_database.json"
    created = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/USA5323_metadata.json"


    # Optionally override the surgery date (else we'll use metadata's treatment_date if present)
    surgery_override_iso = None  # e.g., "2024-06-28"

    organized_df, per_file_df, entropies_per_day, syllable_types = analyze_transitions_for_each_day_from_decoded(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        compute_durations=False,
        surgery_date_override=surgery_override_iso,
        save_dir="figures",  # or None to skip saving
    )

    # Inspect:
    # - organized_df            (full table from the organizer)
    # - per_file_df             (date_time + transition_frequencies per file)
    # - entropies_per_day       (list[(date, total_entropy)])
    # - syllable_types          (label order used for matrices)
    
    
    """
    from transition_entropy_from_decoded import analyze_transitions_for_each_day_from_decoded
    from pathlib import Path
    
    # Input files
    decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5323_decoded_database.json"
    meta    = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/USA5323_metadata.json"
    
    # Output directory (plot will be saved here if provided)
    outdir = Path("/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Optional: override surgery/treatment date (otherwise metadata’s treatment_date will be used if present)
    surgery_override_iso = None  # e.g. "2024-06-28"
    
    # Run analysis
    organized_df, per_file_df, entropies_per_day, syllable_types = analyze_transitions_for_each_day_from_decoded(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,     # True → filter to rows where song_present == True
        compute_durations=False,     # not needed for entropy
        surgery_date_override=surgery_override_iso,
        save_dir=outdir,             # creates {animal_id}_transition_entropy_daily.png
    )
    
    # Inspect results
    print("Organized dataset shape:", organized_df.shape)
    print("Per-file transitions shape:", per_file_df.shape)
    print("Unique syllables:", syllable_types[:10], "...")
    print("First few entropy values:", entropies_per_day[:5])
    
    # If you’d like to save entropies into a CSV for further use:
    import pandas as pd
    entropy_df = pd.DataFrame(entropies_per_day, columns=["day", "total_entropy"])
    csv_path = outdir / "USA5323_transition_entropy_daily.csv"
    entropy_df.to_csv(csv_path, index=False)
    print(f"Saved entropy table → {csv_path}")

    
    """
