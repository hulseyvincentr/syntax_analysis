# -*- coding: utf-8 -*-

# phrase_duration_birds_stats_df.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
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

        PLUS additional derived columns (filled on Post rows for each
        (bird, syllable)):

            "Pre_N_phrases"
            "Pre_Mean_ms"
            "Pre_Variance_ms2"
            "Pre_Std_ms"

            "Post_vs_Pre_Delta_Mean_ms"
            "Post_vs_Pre_Mean_Ratio"
            "Post_vs_Pre_Delta_Variance_ms2"
            "Post_vs_Pre_Variance_Ratio"

            "Post_Mean_Increased"
            "Post_Variance_Increased"
            "Post_Mean_Above_Pre_Mean_Plus_1SD"
            "Post_Mean_Above_Pre_Mean_Plus_2SD"

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

    In addition to the per-group statistics, this function computes, for each
    (Animal ID, Syllable):

        • a pooled "Pre" distribution combining Early Pre and Late Pre
          (weighted by N_phrases), with:
              - Pre_N_phrases
              - Pre_Mean_ms
              - Pre_Variance_ms2
              - Pre_Std_ms

        • comparison metrics between Post and pooled Pre:
              - Post_vs_Pre_Delta_Mean_ms
              - Post_vs_Pre_Mean_Ratio
              - Post_vs_Pre_Delta_Variance_ms2
              - Post_vs_Pre_Variance_Ratio

        • boolean flags indicating increased mean/variance and whether the
          Post mean lies outside the Pre mean ± 1 SD or ± 2 SD:
              - Post_Mean_Increased
              - Post_Variance_Increased
              - Post_Mean_Above_Pre_Mean_Plus_1SD
              - Post_Mean_Above_Pre_Mean_Plus_2SD

    These derived columns are populated on the Post rows for each
    (Animal ID, Syllable). Pre rows keep NaN / False in these columns.

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
            - phrase_duration_stats_df : big concatenated & annotated DataFrame
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

    if big_df.empty:
        return BirdsPhraseDurationStats(
            phrase_duration_stats_df=big_df,
            per_animal_results=batch_results,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Normalize group labels (e.g., "Early-Pre" -> "Early Pre") for safety
    # ──────────────────────────────────────────────────────────────────────
    big_df["Group"] = (
        big_df["Group"]
        .astype(str)
        .str.strip()
        .str.replace("-", " ", regex=False)
    )

    # ──────────────────────────────────────────────────────────────────────
    # Initialize new columns (NaN / False) that we'll fill on Post rows
    # ──────────────────────────────────────────────────────────────────────
    derived_float_cols = [
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Post_vs_Pre_Delta_Mean_ms",
        "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2",
        "Post_vs_Pre_Variance_Ratio",
    ]
    for col in derived_float_cols:
        big_df[col] = np.nan

    derived_bool_cols = [
        "Post_Mean_Increased",
        "Post_Variance_Increased",
        "Post_Mean_Above_Pre_Mean_Plus_1SD",
        "Post_Mean_Above_Pre_Mean_Plus_2SD",
    ]
    for col in derived_bool_cols:
        big_df[col] = False

    # ──────────────────────────────────────────────────────────────────────
    # Helper: pooled pre (Early+Late) mean/variance for each (bird, syllable)
    # ──────────────────────────────────────────────────────────────────────
    PRE_GROUPS = {"Early Pre", "Late Pre"}
    POST_GROUP = "Post"

    # Group by animal + syllable to compare pre vs post within each
    group_keys = [id_col, "Syllable"]
    grouped = big_df.groupby(group_keys)

    for (animal_id_val, syll_val), idx in grouped.groups.items():
        sub = big_df.loc[idx]

        # Extract pre and post rows
        pre_rows = sub[sub["Group"].isin(PRE_GROUPS)]
        post_rows = sub[sub["Group"] == POST_GROUP]

        if post_rows.empty or pre_rows.empty:
            # Need both pre and post to compute comparisons
            continue

        # Pre pooled N, mean, variance (weighted by N_phrases)
        n_pre = pre_rows["N_phrases"].fillna(0).astype(float).values
        mean_pre = pre_rows["Mean_ms"].astype(float).values
        var_pre = pre_rows["Variance_ms2"].astype(float).values

        N_pre_total = n_pre.sum()
        if N_pre_total <= 0:
            continue

        # Pooled pre mean
        pre_mean = float((n_pre * mean_pre).sum() / N_pre_total)

        # Pooled pre variance (unbiased), combining within-group variance and
        # between-group mean differences:
        #   S = Σ[(n_i - 1)*s_i^2 + n_i*(m_i - m)^2]
        #   s_pooled^2 = S / (N_total - 1)
        if N_pre_total > 1:
            S = 0.0
            for ni, mi, si2 in zip(n_pre, mean_pre, var_pre):
                if ni <= 0:
                    continue
                # (ni - 1)*si^2 term
                S += max(ni - 1, 0) * float(si2)
                # between-group contribution
                S += float(ni) * (float(mi) - pre_mean) ** 2
            pre_var = S / (N_pre_total - 1)
        else:
            # Only one phrase overall; degenerate variance
            pre_var = np.nan

        pre_std = float(np.sqrt(pre_var)) if pre_var >= 0 else np.nan

        # Post stats (there should typically be a single Post row)
        post_row = post_rows.iloc[0]
        post_mean = float(post_row["Mean_ms"])
        post_var = float(post_row["Variance_ms2"])

        # Derived comparisons
        delta_mean = post_mean - pre_mean
        mean_ratio = post_mean / pre_mean if pre_mean > 0 else np.nan

        delta_var = post_var - pre_var if not np.isnan(pre_var) else np.nan
        var_ratio = post_var / pre_var if pre_var > 0 else np.nan

        mean_increased = post_mean > pre_mean
        var_increased = post_var > pre_var if not np.isnan(pre_var) else False

        above_1sd = False
        above_2sd = False
        if not np.isnan(pre_std) and pre_std > 0:
            above_1sd = post_mean > (pre_mean + pre_std)
            above_2sd = post_mean > (pre_mean + 2.0 * pre_std)

        # Assign these values back to the Post row(s) for this (animal, syllable)
        post_idx = post_rows.index

        big_df.loc[post_idx, "Pre_N_phrases"] = N_pre_total
        big_df.loc[post_idx, "Pre_Mean_ms"] = pre_mean
        big_df.loc[post_idx, "Pre_Variance_ms2"] = pre_var
        big_df.loc[post_idx, "Pre_Std_ms"] = pre_std

        big_df.loc[post_idx, "Post_vs_Pre_Delta_Mean_ms"] = delta_mean
        big_df.loc[post_idx, "Post_vs_Pre_Mean_Ratio"] = mean_ratio
        big_df.loc[post_idx, "Post_vs_Pre_Delta_Variance_ms2"] = delta_var
        big_df.loc[post_idx, "Post_vs_Pre_Variance_Ratio"] = var_ratio

        big_df.loc[post_idx, "Post_Mean_Increased"] = mean_increased
        big_df.loc[post_idx, "Post_Variance_Increased"] = var_increased
        big_df.loc[post_idx, "Post_Mean_Above_Pre_Mean_Plus_1SD"] = above_1sd
        big_df.loc[post_idx, "Post_Mean_Above_Pre_Mean_Plus_2SD"] = above_2sd

    return BirdsPhraseDurationStats(
        phrase_duration_stats_df=big_df,
        per_animal_results=batch_results,
    )


"""
Example usage
-------------

from pathlib import Path
import importlib

import phrase_duration_birds_stats_df as pb
importlib.reload(pb)  # pick up recent edits

excel_path = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_lesion_metadata.xlsx")
json_root  = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/TweetyBERT_outputs/")

res = pb.build_birds_phrase_duration_stats_df(
    excel_path=excel_path,
    json_root=json_root,
    sheet_name=0,
    id_col="Animal ID",
    treatment_date_col="Treatment date",
    grouping_mode="auto_balance",
    early_group_size=100,
    late_group_size=100,
    post_group_size=100,
    restrict_to_labels=[str(i) for i in range(26)],
    y_max_ms=40000,
    show_plots=False,
)

big_df = res.phrase_duration_stats_df

out_csv = json_root / "compiled_phrase_duration_stats_with_prepost_metrics.csv"
big_df.to_csv(out_csv, index=False)
print(f"Saved big_df with metrics to: {out_csv}")

# Example: filter to Post rows where both mean & variance increased
interesting = big_df[
    (big_df["Group"] == "Post")
    & (big_df["Post_Mean_Increased"])
    & (big_df["Post_Variance_Increased"])
]

print(interesting[[ "Animal ID", "Syllable",
                    "Pre_Mean_ms", "Mean_ms",
                    "Pre_Variance_ms2", "Variance_ms2",
                    "Post_vs_Pre_Delta_Mean_ms",
                    "Post_vs_Pre_Delta_Variance_ms2"]].head())

"""
