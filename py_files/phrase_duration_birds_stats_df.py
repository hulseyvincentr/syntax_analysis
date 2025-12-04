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

            # Pooled Pre (Early+Late) stats:
            "Pre_N_phrases"
            "Pre_Mean_ms"
            "Pre_Variance_ms2"
            "Pre_Std_ms"
            "Pre_Median_ms"         # pooled Pre median (approx, from group medians)
            "Pre_IQR_ms"            # approximate Pre IQR, ~1.349 * Pre_Std_ms
            "Pre_Median_IQR_ms"     # IQR across Pre group medians
            "Pre_Variance_IQR_ms2"  # IQR across Pre group variances

            # Post vs pooled Pre comparisons (mean, variance, median):
            "Post_vs_Pre_Delta_Mean_ms"
            "Post_vs_Pre_Mean_Ratio"
            "Post_vs_Pre_Delta_Variance_ms2"
            "Post_vs_Pre_Variance_Ratio"
            "Post_vs_Pre_Delta_Median_ms"
            "Post_vs_Pre_Median_Ratio"

            # Boolean flags for increase / thresholds:
            "Post_Mean_Increased"
            "Post_Variance_Increased"
            "Post_Mean_Above_Pre_Mean_Plus_1SD"
            "Post_Mean_Above_Pre_Mean_Plus_2SD"

        Notes
        -----
        The pooled Pre mean/variance are computed exactly from the Early+Late
        Pre group means/variances and N_phrases (standard pooled variance
        formula). The pooled Pre median is an approximation based on group
        medians and N_phrases; the Pre_IQR_ms is an approximate IQR assuming
        a roughly normal distribution (IQR ≈ 1.349 * Std).

        The additional Pre_Median_IQR_ms and Pre_Variance_IQR_ms2 columns
        quantify how different the Pre groups are from each other in terms of
        their medians and variances, respectively (IQR across Pre groups).
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
              - Pre_Median_ms   (approximate pooled median based on group medians)
              - Pre_IQR_ms      (approximate IQR ≈ 1.349 * Pre_Std_ms)
              - Pre_Median_IQR_ms     (IQR across Pre group medians)
              - Pre_Variance_IQR_ms2  (IQR across Pre group variances)

        • comparison metrics between Post and pooled Pre:
              - Post_vs_Pre_Delta_Mean_ms
              - Post_vs_Pre_Mean_Ratio
              - Post_vs_Pre_Delta_Variance_ms2
              - Post_vs_Pre_Variance_Ratio
              - Post_vs_Pre_Delta_Median_ms
              - Post_vs_Pre_Median_Ratio

        • boolean flags indicating increased mean/variance and whether the
          Post mean lies outside the Pre mean ± 1 SD or ± 2 SD:
              - Post_Mean_Increased
              - Post_Variance_Increased
              - Post_Mean_Above_Pre_Mean_Plus_1SD
              - Post_Mean_Above_Pre_Mean_Plus_2SD

    These derived columns are populated on the Post rows for each
    (Animal ID, Syllable). Pre rows keep NaN / False in these columns.
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

        # Ensure N_phrases exists for compatibility with plotting
        if "N_phrases" not in stats.columns:
            if "n_phrases" in stats.columns:
                stats["N_phrases"] = stats["n_phrases"]
            elif "N" in stats.columns:
                stats["N_phrases"] = stats["N"]
            else:
                for cand in ["Durations_ms", "durations_ms", "Durations"]:
                    if cand in stats.columns:
                        stats["N_phrases"] = stats[cand].apply(
                            lambda v: len(v)
                            if isinstance(v, (list, np.ndarray))
                            else (len(v) if hasattr(v, "__len__") else np.nan)
                        )
                        break
                if "N_phrases" not in stats.columns:
                    stats["N_phrases"] = np.nan

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
        "Pre_Median_ms",          # pooled Pre median
        "Pre_IQR_ms",             # approximate pooled Pre IQR
        "Pre_Median_IQR_ms",      # IQR across Pre group medians
        "Pre_Variance_IQR_ms2",   # IQR across Pre group variances
        "Post_vs_Pre_Delta_Mean_ms",
        "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2",
        "Post_vs_Pre_Variance_Ratio",
        "Post_vs_Pre_Delta_Median_ms",
        "Post_vs_Pre_Median_Ratio",
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

    group_keys = [id_col, "Syllable"]
    grouped = big_df.groupby(group_keys)

    for (animal_id_val, syll_val), idx in grouped.groups.items():
        sub = big_df.loc[idx]

        pre_rows = sub[sub["Group"].isin(PRE_GROUPS)]
        post_rows = sub[sub["Group"] == POST_GROUP]

        if post_rows.empty or pre_rows.empty:
            continue

        n_pre = pre_rows["N_phrases"].fillna(0).astype(float).values
        mean_pre = pre_rows["Mean_ms"].astype(float).values
        var_pre = pre_rows["Variance_ms2"].astype(float).values

        N_pre_total = n_pre.sum()
        if N_pre_total <= 0:
            continue

        # Pooled pre mean
        pre_mean = float((n_pre * mean_pre).sum() / N_pre_total)

        # Pooled pre variance (unbiased)
        if N_pre_total > 1:
            S = 0.0
            for ni, mi, si2 in zip(n_pre, mean_pre, var_pre):
                if ni <= 0:
                    continue
                S += max(ni - 1, 0) * float(si2)
                S += float(ni) * (float(mi) - pre_mean) ** 2
            pre_var = S / (N_pre_total - 1)
        else:
            pre_var = np.nan

        pre_std = float(np.sqrt(pre_var)) if pre_var >= 0 else np.nan

        # Approximate pooled pre median based on group medians and N_phrases.
        try:
            med_pre = pre_rows["Median_ms"].astype(float).values
        except Exception:
            med_pre = np.array([], dtype=float)

        if med_pre.size and n_pre.size and N_pre_total > 0:
            pre_median = float(np.average(med_pre, weights=n_pre))
        else:
            pre_median = np.nan

        # Approximate pooled Pre IQR (assuming roughly normal distribution):
        if not np.isnan(pre_std):
            pre_iqr = float(1.349 * pre_std)
        else:
            pre_iqr = np.nan

        # IQR across Pre group medians and variances
        med_pre_clean = med_pre[~np.isnan(med_pre)] if med_pre.size else med_pre
        var_pre_clean = var_pre[~np.isnan(var_pre)] if var_pre.size else var_pre

        if med_pre_clean.size:
            pre_median_iqr = float(
                np.nanpercentile(med_pre_clean, 75) - np.nanpercentile(med_pre_clean, 25)
            )
        else:
            pre_median_iqr = np.nan

        if var_pre_clean.size:
            pre_variance_iqr = float(
                np.nanpercentile(var_pre_clean, 75) - np.nanpercentile(var_pre_clean, 25)
            )
        else:
            pre_variance_iqr = np.nan

        # Post stats (there should typically be a single Post row)
        post_row = post_rows.iloc[0]
        post_mean = float(post_row["Mean_ms"])
        post_var = float(post_row["Variance_ms2"])
        post_median = float(post_row["Median_ms"])

        # Derived comparisons: mean/variance
        delta_mean = post_mean - pre_mean
        mean_ratio = post_mean / pre_mean if pre_mean > 0 else np.nan

        delta_var = post_var - pre_var if not np.isnan(pre_var) else np.nan
        var_ratio = post_var / pre_var if pre_var > 0 else np.nan

        # Derived comparisons: median
        if not np.isnan(pre_median) and pre_median != 0:
            delta_median = post_median - pre_median
            median_ratio = post_median / pre_median
        else:
            delta_median = np.nan
            median_ratio = np.nan

        mean_increased = post_mean > pre_mean
        var_increased = post_var > pre_var if not np.isnan(pre_var) else False

        above_1sd = False
        above_2sd = False
        if not np.isnan(pre_std) and pre_std > 0:
            above_1sd = post_mean > (pre_mean + pre_std)
            above_2sd = post_mean > (pre_mean + 2.0 * pre_std)

        post_idx = post_rows.index

        big_df.loc[post_idx, "Pre_N_phrases"] = N_pre_total
        big_df.loc[post_idx, "Pre_Mean_ms"] = pre_mean
        big_df.loc[post_idx, "Pre_Variance_ms2"] = pre_var
        big_df.loc[post_idx, "Pre_Std_ms"] = pre_std
        big_df.loc[post_idx, "Pre_Median_ms"] = pre_median
        big_df.loc[post_idx, "Pre_IQR_ms"] = pre_iqr
        big_df.loc[post_idx, "Pre_Median_IQR_ms"] = pre_median_iqr
        big_df.loc[post_idx, "Pre_Variance_IQR_ms2"] = pre_variance_iqr

        big_df.loc[post_idx, "Post_vs_Pre_Delta_Mean_ms"] = delta_mean
        big_df.loc[post_idx, "Post_vs_Pre_Mean_Ratio"] = mean_ratio
        big_df.loc[post_idx, "Post_vs_Pre_Delta_Variance_ms2"] = delta_var
        big_df.loc[post_idx, "Post_vs_Pre_Variance_Ratio"] = var_ratio

        big_df.loc[post_idx, "Post_vs_Pre_Delta_Median_ms"] = delta_median
        big_df.loc[post_idx, "Post_vs_Pre_Median_Ratio"] = median_ratio

        big_df.loc[post_idx, "Post_Mean_Increased"] = mean_increased
        big_df.loc[post_idx, "Post_Variance_Increased"] = var_increased
        big_df.loc[post_idx, "Post_Mean_Above_Pre_Mean_Plus_1SD"] = above_1sd
        big_df.loc[post_idx, "Post_Mean_Above_Pre_Mean_Plus_2SD"] = above_2sd

    return BirdsPhraseDurationStats(
        phrase_duration_stats_df=big_df,
        per_animal_results=batch_results,
    )

    
"""
from pathlib import Path
import importlib

import phrase_duration_birds_stats_df as pb
importlib.reload(pb)  # pick up recent edits

# Paths to your metadata Excel and JSON root
excel_path = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")
json_root  = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")

# Run the batch phrase-duration analysis across all birds in the Excel sheet
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

# Big concatenated DataFrame with all the pre/post metrics
big_df = res.phrase_duration_stats_df
print("big_df shape:", big_df.shape)
print(big_df.head())

# Save to CSV for downstream plotting (e.g., phrase_and_metadata_plotting.py)
out_csv = json_root / "compiled_phrase_duration_stats_with_prepost_metrics.csv"
big_df.to_csv(out_csv, index=False)
print(f"Saved big_df with metrics to: {out_csv}")


"""
