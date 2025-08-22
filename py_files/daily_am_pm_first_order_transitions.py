# daily_am_pm_first_order_transitions.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from organize_decoded_with_durations import build_organized_dataset_with_durations


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _try_int_for_sort(x):
    try:
        return int(x)
    except Exception:
        return x

def _as_str_labels(labels: Iterable) -> List[str]:
    return [str(l) for l in labels]


# ──────────────────────────────────────────────────────────────────────────────
# Transition matrices
# ──────────────────────────────────────────────────────────────────────────────
def build_first_order_transition_matrices(
    organized_df: pd.DataFrame,
    *,
    order_column: str = "syllable_order",
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, probs) for first-order transitions."""
    if order_column not in organized_df.columns:
        raise KeyError(f"Expected column '{order_column}' in organized_df.")

    from collections import defaultdict
    transition_counts = defaultdict(lambda: defaultdict(int))

    for order in organized_df[order_column]:
        if isinstance(order, list) and len(order) > 1:
            order_str = _as_str_labels(order)
            for i in range(len(order_str) - 1):
                transition_counts[order_str[i]][order_str[i + 1]] += 1

    all_labels = set(transition_counts.keys()) | {s for d in transition_counts.values() for s in d}
    all_labels = _as_str_labels(all_labels)

    if restrict_to_labels is not None:
        allowed = set(_as_str_labels(restrict_to_labels))
        all_labels = [l for l in all_labels if l in allowed]

    if not all_labels:
        return (pd.DataFrame(dtype=int), pd.DataFrame(dtype=float))

    sorted_labels = sorted(all_labels, key=_try_int_for_sort)
    counts = pd.DataFrame(0, index=sorted_labels, columns=sorted_labels, dtype=int)

    for curr_syll, nexts in transition_counts.items():
        if curr_syll not in counts.index:
            continue
        for next_syll, c in nexts.items():
            if next_syll in counts.columns:
                counts.loc[curr_syll, next_syll] = c

    if min_row_total > 0 and not counts.empty:
        sparse = counts.sum(axis=1) < min_row_total
        counts.loc[sparse, :] = 0

    if counts.empty:
        return (counts, counts.astype(float))

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
    figsize: Tuple[float, float] = (9, 8),
    show: bool = True,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> None:
    if probs.empty:
        return
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        probs,
        cmap="binary", vmin=0.0, vmax=1.0,
        cbar_kws={"label": "Transition Probability"},
        square=True,
    )
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    plt.tight_layout()
    if save_fig_path:
        save_fig_path = Path(save_fig_path); save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# AM/PM runner
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_am_pm_first_order_transitions(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    only_song_present: bool = False,
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
    output_dir: Optional[Union[str, Path]] = None,  # base folder for outputs
    save_csv: bool = True,
    save_png: bool = True,
    show_plots: bool = True,
    enforce_consistent_order: bool = True,  # same label order across all days/halves
) -> Dict[str, Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]]:
    """
    For each date, compute AM (00:00–11:59) and PM (12:00–23:59) first-order
    transition matrices.

    Returns
    -------
    dict keyed by date_str:
      {
        "AM": {"counts": DF|None, "probs": DF|None, "figure_path": Path|None,
               "counts_csv": Path|None, "probs_csv": Path|None},
        "PM": {...}
      }
    """
    org = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=False,
    )
    df = org.organized_df.copy()
    if "Date" not in df.columns or "Hour" not in df.columns:
        raise KeyError("Organized DataFrame must include 'Date' and 'Hour' columns.")

    # Clean up Date/Hour
    df = df[df["Date"].notna()].copy()
    df["DateOnly"] = pd.to_datetime(df["Date"]).dt.normalize()
    hour_numeric = pd.to_numeric(df["Hour"], errors="coerce")
    df["HourNum"] = hour_numeric

    # Best-effort animal ID
    animal_id = None
    if "Animal ID" in df.columns:
        non_null = df["Animal ID"].dropna()
        if not non_null.empty:
            animal_id = str(non_null.iloc[0])

    # Output base
    output_dir = Path(output_dir) if output_dir else None
    results: Dict[str, Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]] = OrderedDict()

    # First pass: compute AM/PM per day and collect union of labels (for consistent ordering)
    union_labels = set(_as_str_labels(restrict_to_labels)) if restrict_to_labels else set()
    per_day_half_probs: Dict[str, Dict[str, pd.DataFrame]] = OrderedDict()  # date -> {"AM": probs, "PM": probs}

    for date_val, day_df in df.groupby("DateOnly", sort=True):
        date_str = pd.to_datetime(date_val).strftime("%Y.%m.%d")

        am_df = day_df[(day_df["HourNum"] >= 0) & (day_df["HourNum"] < 12)]
        pm_df = day_df[(day_df["HourNum"] >= 12) & (day_df["HourNum"] <= 23)]

        halves = {}
        for label, part_df in (("AM", am_df), ("PM", pm_df)):
            counts, probs = build_first_order_transition_matrices(
                part_df,
                order_column="syllable_order",
                restrict_to_labels=restrict_to_labels,
                min_row_total=min_row_total,
            )
            # skip halves with no transitions
            if counts.empty or (counts.values.sum() == 0):
                continue
            halves[label] = probs
            if enforce_consistent_order:
                union_labels.update(probs.index.tolist())
                union_labels.update(probs.columns.tolist())

        if halves:
            per_day_half_probs[date_str] = halves

    # Canonical label order
    if enforce_consistent_order:
        if restrict_to_labels is not None:
            canonical = list(_as_str_labels(restrict_to_labels))
        else:
            canonical = sorted(list(union_labels), key=_try_int_for_sort)
    else:
        canonical = None

    # Second pass: save & plot
    for date_str, halves in per_day_half_probs.items():
        results[date_str] = {"AM": {"counts": None, "probs": None, "figure_path": None,
                                    "counts_csv": None, "probs_csv": None},
                             "PM": {"counts": None, "probs": None, "figure_path": None,
                                    "counts_csv": None, "probs_csv": None}}

        for label in ("AM", "PM"):
            if label not in halves:
                continue
            probs = halves[label]
            if canonical is not None:
                probs = probs.reindex(index=canonical, columns=canonical, fill_value=0)

            # Build titles & paths
            title_bits = []
            if animal_id: title_bits.append(animal_id)
            title_bits.append(f"{date_str} {label} First-Order Transition Probabilities")
            if org.treatment_date: title_bits.append(f"(Treatment: {org.treatment_date})")
            title = " ".join(title_bits)

            fig_path = counts_csv = probs_csv = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{date_str}_{label}_first_order"
                if save_png:
                    fig_path = output_dir / f"{stem}_probs.png"
                if save_csv:
                    probs_csv = output_dir / f"{stem}_probs.csv"

            # Save probs CSV (counts not persisted here)
            if probs_csv: probs.to_csv(probs_csv)

            # Plot
            if fig_path or show_plots:
                plot_transition_matrix(
                    probs, title=title, save_fig_path=fig_path, show=show_plots
                )

            # Record outputs
            results[date_str][label] = {
                "counts": None,
                "probs": probs,
                "figure_path": fig_path,
                "counts_csv": counts_csv,
                "probs_csv": probs_csv,
            }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/USA5288_decoded_database.json"
    meta    = "/path/to/USA5288_creation_data.json"

    out = run_daily_am_pm_first_order_transitions(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        restrict_to_labels=None,                 # e.g., ['0','1','2'] to lock axes
        min_row_total=0,                         # e.g., 5 to suppress sparse rows
        output_dir="figures/daily_transitions_am_pm",
        save_csv=True,
        save_png=True,
        show_plots=True,
        enforce_consistent_order=True,
    )
    print(f"Generated AM/PM matrices for {len(out)} day(s).")


"""
from daily_am_pm_first_order_transitions import run_daily_am_pm_first_order_transitions

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

results = run_daily_am_pm_first_order_transitions(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,
    restrict_to_labels=None,   # or your canonical list to keep axes fixed
    min_row_total=0,           # try 5 to zero ultra-sparse rows
    output_dir="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures/daily_transitions_am_pm",
    save_csv=False,
    save_png=True,
    show_plots=True,
    enforce_consistent_order=True,
)

"""