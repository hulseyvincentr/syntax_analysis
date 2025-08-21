# -*- coding: utf-8 -*-
# build_first_order_transitions.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from organize_decoded_with_durations import build_organized_dataset_with_durations


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _try_int_for_sort(x):
    """Sort labels numerically when possible, otherwise lexicographically."""
    try:
        return int(x)
    except Exception:
        return x

def _as_str_labels(labels: Iterable) -> List[str]:
    """Ensure all labels are strings (safe for DataFrame index/columns)."""
    return [str(l) for l in labels]


# ──────────────────────────────────────────────────────────────────────────────
# Core: build first-order transition matrices
# ──────────────────────────────────────────────────────────────────────────────
def build_first_order_transition_matrices(
    organized_df: pd.DataFrame,
    *,
    order_column: str = "syllable_order",
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build first-order transition **count** and **probability** matrices from an
    `organized_df` produced by `build_organized_dataset_with_durations`.

    Parameters
    ----------
    organized_df : pd.DataFrame
        Must contain a column with per-file syllable orders (lists), default "syllable_order".
    order_column : str
        Column name containing lists of syllable labels in temporal order.
    restrict_to_labels : Optional[Iterable[str]]
        If provided, only include transitions among this set of labels (others dropped).
    min_row_total : int
        Rows whose total outgoing transitions < min_row_total are zeroed (stays in index).

    Returns
    -------
    counts : pd.DataFrame
        Square matrix of transition counts (rows=current, cols=next).
    probs : pd.DataFrame
        Row-normalized transition probabilities (sum(row) == 1 or 0 if empty/zeroed).
    """
    if order_column not in organized_df.columns:
        raise KeyError(f"Expected column '{order_column}' in organized_df.")

    transition_counts = defaultdict(lambda: defaultdict(int))

    # Count adjacent transitions within each file's syllable order
    for order in organized_df[order_column]:
        if isinstance(order, list) and len(order) > 1:
            # ensure labels are strings for consistent indexing
            order_str = _as_str_labels(order)
            for i in range(len(order_str) - 1):
                curr_syll = order_str[i]
                next_syll = order_str[i + 1]
                transition_counts[curr_syll][next_syll] += 1

    # Collect set of labels observed anywhere (as strings)
    all_labels = set(transition_counts.keys()) | {
        s for d in transition_counts.values() for s in d
    }
    all_labels = _as_str_labels(all_labels)

    # Optional restriction
    if restrict_to_labels is not None:
        allowed = set(_as_str_labels(restrict_to_labels))
        all_labels = [l for l in all_labels if l in allowed]

    # Sorted labels (numeric when possible)
    sorted_labels = sorted(all_labels, key=_try_int_for_sort)

    # Build square count matrix
    counts = pd.DataFrame(0, index=sorted_labels, columns=sorted_labels, dtype=int)
    for curr_syll, nexts in transition_counts.items():
        if curr_syll not in counts.index:
            continue
        for next_syll, c in nexts.items():
            if next_syll in counts.columns:
                counts.loc[curr_syll, next_syll] = c

    # Apply minimum row-total rule (optional)
    if min_row_total > 0:
        bad_rows = counts.sum(axis=1) < min_row_total
        counts.loc[bad_rows, :] = 0

    # Row-normalize to probabilities
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0)

    return counts, probs


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_transition_matrix(
    probs: pd.DataFrame,
    *,
    title: Optional[str] = None,
    xlabel: str = "Next Syllable",
    ylabel: str = "Current Syllable",
    figsize: Tuple[float, float] = (10, 8),
    show: bool = True,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a row-stochastic first-order transition matrix (probabilities) as a heatmap.
    Uses a white→black grayscale (low→high).
    """
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        probs,
        cmap="binary",  # white (low) → black (high)
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Transition Probability"},
        square=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # readable tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    plt.tight_layout()

    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# High-level runner that ties everything together
# ──────────────────────────────────────────────────────────────────────────────
def run_first_order_transitions(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    only_song_present: bool = False,
    compute_durations: bool = False,   # not needed for transitions; keep for parity
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
    save_counts_csv: Optional[Union[str, Path]] = None,
    save_probs_csv: Optional[Union[str, Path]] = None,
    save_fig_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
    1) calls your organizer
    2) builds first-order transitions
    3) plots and optionally saves outputs

    Returns
    -------
    counts, probs : (pd.DataFrame, pd.DataFrame)
    """
    out = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=compute_durations,
    )
    organized_df = out.organized_df

    # Best-effort animal ID for plot title
    animal_id = None
    if "Animal ID" in organized_df.columns:
        # take first non-null, if any
        non_null = organized_df["Animal ID"].dropna()
        if not non_null.empty:
            animal_id = str(non_null.iloc[0])

    counts, probs = build_first_order_transition_matrices(
        organized_df,
        order_column="syllable_order",
        restrict_to_labels=restrict_to_labels,
        min_row_total=min_row_total,
    )

    # Save CSVs if requested
    if save_counts_csv:
        Path(save_counts_csv).parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(save_counts_csv)
    if save_probs_csv:
        Path(save_probs_csv).parent.mkdir(parents=True, exist_ok=True)
        probs.to_csv(save_probs_csv)

    # Title with available metadata
    title_bits = []
    if animal_id:
        title_bits.append(f"{animal_id}")
    title_bits.append("First-Order Syllable Transition Probabilities")
    if out.treatment_date:
        title_bits.append(f"(Treatment: {out.treatment_date})")
    title = " ".join(title_bits)

    plot_transition_matrix(
        probs,
        title=title,
        save_fig_path=save_fig_path,
        show=show_plot,
    )

    return counts, probs


# ──────────────────────────────────────────────────────────────────────────────
# Example (for Spyder/IPython)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/USA5288_decoded_database.json"
    meta    = "/path/to/USA5288_creation_data.json"
    counts, probs = run_first_order_transitions(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        restrict_to_labels=None,   # e.g., ['0','1','2'] to limit the matrix
        min_row_total=0,           # e.g., 5 to zero out sparse rows
        save_counts_csv=None,      # e.g., "out/first_order_counts.csv"
        save_probs_csv=None,       # e.g., "out/first_order_probs.csv"
        save_fig_path=None,        # e.g., "out/first_order_probs.png"
        show_plot=True,
    )

"""
from first_order_transitions import run_first_order_transitions

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

counts, probs = run_first_order_transitions(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,
    restrict_to_labels=None,           # or a subset like ['0','1','9']
    min_row_total=0,                   # e.g., 5 to suppress tiny rows
    #save_counts_csv="figures/first_order_counts.csv",
    #save_probs_csv="figures/first_order_probs.csv",
    save_fig_path="figures/first_order_probs.png",
    show_plot=True,
)

"""