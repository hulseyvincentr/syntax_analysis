# -*- coding: utf-8 -*-

# phrase_duration_birds_stats_df.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from phrase_duration_pre_vs_post_grouped import (
    run_batch_phrase_duration_from_excel,
    GroupedPlotsResult,
)

__all__ = [
    "BirdsPhraseDurationStats",
    "build_birds_phrase_duration_stats_df",
]


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BirdsPhraseDurationStats:
    """
    Container for phrase-duration statistics across multiple birds.

    Attributes
    ----------
    phrase_duration_stats_df : pd.DataFrame
        Big concatenated DataFrame with one row per (bird, group, syllable).
        Columns typically include:
            "Group", "Syllable", "N_phrases",
            "Mean_ms", "SEM_ms", "Median_ms",
            "Std_ms", "Variance_ms2", id_col
    per_animal_results : dict[str, GroupedPlotsResult]
        Mapping from animal ID -> full GroupedPlotsResult returned by
        phrase_duration_pre_vs_post_grouped.run_batch_phrase_duration_from_excel.
    """
    phrase_duration_stats_df: pd.DataFrame
    per_animal_results: Dict[str, GroupedPlotsResult]


# ──────────────────────────────────────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────────────────────────────────────
def build_birds_phrase_duration_stats_df(
    *,
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",    # or "explicit"
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,
) -> BirdsPhraseDurationStats:
    """
    Run phrase_duration_pre_vs_post_grouped for every bird listed in the Excel
    metadata sheet and stack the per-bird phrase-duration stats into one
    concatenated DataFrame ("big_df").

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel metadata sheet (with Animal IDs, treatment dates, etc.).
    json_root : str or Path
        Root directory containing the decoded/detected JSON files for each bird.
    sheet_name : int or str, default 0
        Excel sheet index or name.
    id_col : str, default "Animal ID"
        Column in the Excel sheet that uniquely identifies each bird.
    treatment_date_col : str, default "Treatment date"
        Column containing the treatment (lesion) date per animal.
    grouping_mode : {"auto_balance", "explicit"}, default "auto_balance"
        Passed through to run_batch_phrase_duration_from_excel and ultimately
        to run_phrase_duration_pre_vs_post_grouped.
    early_group_size, late_group_size, post_group_size : int
        Number of songs per group when grouping_mode="auto_balance".
    restrict_to_labels : sequence of str/int or None
        If not None, restrict analysis to these syllable labels.
    y_max_ms : float or None
        Optional upper limit for y-axis in plotting (passed through).
    show_plots : bool, default True
        Whether to display plots while running per-bird analysis.

    Returns
    -------
    BirdsPhraseDurationStats
        An object with:
            - phrase_duration_stats_df : big concatenated DataFrame
            - per_animal_results       : dict[animal_id, GroupedPlotsResult]
    """
    excel_path = Path(excel_path)
    json_root = Path(json_root)

    # Run the existing batch helper from phrase_duration_pre_vs_post_grouped
    batch_results: Dict[str, GroupedPlotsResult] = run_batch_phrase_duration_from_excel(
        excel_path=excel_path,
        json_root=json_root,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        grouping_mode=grouping_mode,
        early_group_size=early_group_size,
        late_group_size=late_group_size,
        post_group_size=post_group_size,
        restrict_to_labels=restrict_to_labels,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
    )

    # Stack per-bird phrase_duration_stats_df into one big DataFrame
    frames = []
    for animal_id, res in batch_results.items():
        stats = getattr(res, "phrase_duration_stats_df", None)
        if stats is None or stats.empty:
            continue

        stats = stats.copy()
        # Track which bird each row comes from
        stats[id_col] = animal_id
        frames.append(stats)

    if frames:
        big_df = pd.concat(frames, ignore_index=True)
    else:
        # Empty but with expected columns so downstream code doesn't break
        big_df = pd.DataFrame(
            columns=[
                "Group",
                "Syllable",
                "N_phrases",
                "Mean_ms",
                "SEM_ms",
                "Median_ms",
                "Std_ms",
                "Variance_ms2",
                id_col,
            ]
        )

    return BirdsPhraseDurationStats(
        phrase_duration_stats_df=big_df,
        per_animal_results=batch_results,
    )


"""
from pathlib import Path
import importlib

import phrase_duration_birds_stats_df as pb
importlib.reload(pb)  # pick up recent edits

# Paths to your metadata and JSON/NPZ root
excel_path = Path("/Users/mirandahulsey-vincent/Desktop/Area_X_lesion_metadata.xlsx")
json_root  = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs")

# Run phrase_duration_pre_vs_post_grouped across all birds and build big_df
res = pb.build_birds_phrase_duration_stats_df(
    excel_path=excel_path,
    json_root=json_root,
    sheet_name=0,                     # or the sheet name if not 0
    id_col="Animal ID",
    treatment_date_col="Treatment date",
    grouping_mode="auto_balance",     # or "explicit"
    early_group_size=100,
    late_group_size=100,
    post_group_size=100,
    restrict_to_labels=[str(i) for i in range(26)],  # 0..25 on x-axis
    y_max_ms=40000,
    show_plots=False,                 # True if you want the per-bird plots to pop up
)

# Big concatenated DataFrame: one row per (bird, group, syllable)
big_df = res.phrase_duration_stats_df

# Optional: inspect / save
print(big_df.head())

out_csv = json_root / "compiled_phrase_duration_stats_from_helper.csv"
big_df.to_csv(out_csv, index=False)
print(f"Saved big_df to: {out_csv}")

# Per-bird full results if you need them later
per_animal = res.per_animal_results
print(per_animal.keys())  # e.g., dict_keys(['R08', 'R09', 'R10', ...])



"""