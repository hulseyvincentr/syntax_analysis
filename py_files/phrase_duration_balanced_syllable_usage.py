# -*- coding: utf-8 -*-
"""
phrase_duration_balanced_syllable_usage.py

Compute phrase-duration statistics for *balanced syllable usage*
(Early-Pre, Late-Pre, Post groups) for one or many birds.

UPDATED FEATURES
----------------
- When a song-detection JSON is available, split-up songs are merged using
  `merge_annotations_from_split_songs.build_decoded_with_split_labels`,
  and phrase occurrences are built from the *merged* annotations.

- The batch-from-Excel function
  `run_balanced_syllable_usage_from_metadata_excel` assumes a per-bird
  folder layout (one subfolder per Animal ID under `decoded_root`)
  containing both JSONs, and writes a compiled CSV of the results with
  columns that match `compiled_phrase_duration_stats_with_prepost_metrics.csv`.

Typical usage (Spyder console)
------------------------------
Single bird (recommended, using merged songs):

    from pathlib import Path
    import importlib
    import phrase_duration_balanced_syllable_usage as pbsu
    importlib.reload(pbsu)

    decoded_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
    bird_id      = "USA5283"

    decoded_json   = decoded_root / bird_id / f"{bird_id}_decoded_database.json"
    detection_json = decoded_root / bird_id / f"{bird_id}_song_detection.json"
    tdate          = "2024-03-05"   # treatment date

    res = pbsu.run_phrase_duration_balanced_syllable_usage(
        decoded_database_json=decoded_json,
        treatment_date=tdate,
        group_size=None,              # None → auto-balance per syllable
        restrict_to_labels=None,
        animal_id=bird_id,
        song_detection_json=detection_json,  # <- use merged songs
    )

    stats_df = res.phrase_duration_stats_df
    stats_df.head()

Single bird (legacy JSON-only, no merged songs):

    res = pbsu.run_phrase_duration_balanced_syllable_usage(
        decoded_database_json=decoded_json,
        treatment_date=tdate,
        group_size=None,
        restrict_to_labels=None,
        animal_id=bird_id,
        song_detection_json=None,  # or omit this argument
    )

Batch across birds from Excel metadata (per-bird folders, merged songs):

    from pathlib import Path
    import importlib
    import phrase_duration_balanced_syllable_usage as pbsu
    importlib.reload(pbsu)

    decoded_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
    excel_path   = decoded_root / "Area_X_lesion_metadata.xlsx"

    folder_res = pbsu.run_balanced_syllable_usage_from_metadata_excel(
        excel_path=excel_path,
        decoded_root=decoded_root,
        sheet_name=0,                      # or a sheet name like "metadata"
        id_col="Animal ID",
        treatment_date_col="Treatment date",
        group_size=None,                   # None → auto-balance per syllable
        restrict_to_labels=None, #[str(i) for i in range(26)] if you just want the first 25 syllables
        glob_pattern="*decoded_database.json",  # used for decoded JSON search
        compiled_filename="usage_balanced_phrase_duration_stats.csv",
    )

    compiled_df   = folder_res.compiled_stats_df
    compiled_path = folder_res.compiled_stats_path

    print(compiled_df.head())
    print("Saved compiled CSV to:", compiled_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Any, Tuple

import json
import numpy as np
import pandas as pd

import merge_annotations_from_split_songs as mps

__all__ = [
    "BalancedSyllableUsageResult",
    "BalancedFolderResult",
    "build_phrase_occurrence_table",
    "run_phrase_duration_balanced_syllable_usage",
    "run_balanced_syllable_usage_from_metadata_excel",
]


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BalancedSyllableUsageResult:
    """Container for balanced phrase-duration stats for a single bird."""

    animal_id: Optional[str]
    treatment_date: pd.Timestamp
    phrase_duration_stats_df: pd.DataFrame
    balanced_occurrences_df: pd.DataFrame
    raw_phrases_df: pd.DataFrame  # one-row-per-phrase occurrences table


@dataclass
class BalancedFolderResult:
    """Container for multi-bird run via Excel metadata."""

    compiled_stats_df: pd.DataFrame
    compiled_stats_path: Path
    per_animal_results: Dict[str, BalancedSyllableUsageResult]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_label(label: Any) -> str:
    """Convert a label to a clean string (e.g. 1 → "1")."""
    if label is None:
        return ""
    return str(label).strip()


# ---- phrase-occurrence builders ---------------------------------------------


def build_phrase_occurrence_table(
    decoded_database_json: Union[str, Path],
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    """
    ORIGINAL JSON-BASED IMPLEMENTATION.

    Parse a TweetyBERT decoded_database JSON into a phrase-occurrence table
    with one row per (file, syllable occurrence).

    This is still used when `song_detection_json` is not provided to
    `run_phrase_duration_balanced_syllable_usage`.

    Returns a DataFrame with columns:
        "file_name", "creation_datetime",
        "Syllable", "onset_ms", "offset_ms", "duration_ms".
    """
    decoded_database_json = Path(decoded_database_json)
    with decoded_database_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    results = data.get("results", None)
    if results is None:
        # Some older formats might store the list at the top level
        if isinstance(data, list):
            results = data
        else:
            raise ValueError(
                f"JSON at {decoded_database_json} does not contain a 'results' list."
            )

    allowed_labels: Optional[set[str]] = None
    if restrict_to_labels is not None:
        allowed_labels = {str(x) for x in restrict_to_labels}

    rows = []
    for rec in results:
        if not isinstance(rec, dict):
            continue

        fname = rec.get("file_name") or rec.get("filename") or rec.get("file") or ""
        fname = str(fname)

        # Try multiple possible keys for creation date / datetime
        cdate = (
            rec.get("Recording DateTime")
            or rec.get("creation_date")
            or rec.get("creation_datetime")
            or rec.get("file_creation_date")
        )
        try:
            creation_dt = pd.to_datetime(cdate) if cdate is not None else pd.NaT
        except Exception:
            creation_dt = pd.NaT

        # Two common keys for the label→intervals mapping
        label_dict = (
            rec.get("syllable_onsets_offsets_ms_dict")
            or rec.get("syllable_onsets_offsets_ms")
            or {}
        )
        if not isinstance(label_dict, dict):
            continue

        for lab, intervals in label_dict.items():
            lab_str = _normalize_label(lab)
            if allowed_labels is not None and lab_str not in allowed_labels:
                continue
            if not isinstance(intervals, (list, tuple)):
                continue
            for itv in intervals:
                if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
                    continue
                try:
                    onset = float(itv[0])
                    offset = float(itv[1])
                except Exception:
                    continue
                dur = offset - onset
                if not np.isfinite(dur) or dur <= 0:
                    continue
                rows.append(
                    {
                        "file_name": fname,
                        "creation_datetime": creation_dt,
                        "Syllable": lab_str,
                        "onset_ms": onset,
                        "offset_ms": offset,
                        "duration_ms": dur,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "file_name",
                "creation_datetime",
                "Syllable",
                "onset_ms",
                "offset_ms",
                "duration_ms",
            ]
        )
    df = pd.DataFrame.from_records(rows)
    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df["creation_datetime"]):
        df["creation_datetime"] = pd.to_datetime(df["creation_datetime"], errors="coerce")
    return df


def _build_phrase_occurrences_from_annotations(
    annotations_df: pd.DataFrame,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    """
    Build a phrase-occurrence table from the `annotations_appended_df`
    produced by `merge_annotations_from_split_songs.build_decoded_with_split_labels`.

    This is analogous to `build_phrase_occurrence_table`, but it operates on
    the merged-song annotations instead of the raw JSON "results" list.

    Returns DataFrame with columns:
        "file_name", "creation_datetime",
        "Syllable", "onset_ms", "offset_ms", "duration_ms".
    """
    if annotations_df.empty:
        return pd.DataFrame(
            columns=[
                "file_name",
                "creation_datetime",
                "Syllable",
                "onset_ms",
                "offset_ms",
                "duration_ms",
            ]
        )

    # Candidate columns for spans dict and datetime / filename
    span_candidates = [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_ms",
        "syllable_onsets_offsets",
    ]
    dt_candidates = [
        "Recording DateTime",
        "Recording datetime",
        "Recording Datetime",
        "recording_datetime",
        "creation_datetime",
    ]
    fname_candidates = [
        "file_name",
        "Recording file name",
        "filename",
        "file",
    ]

    span_col = next((c for c in span_candidates if c in annotations_df.columns), None)
    if span_col is None:
        raise KeyError(
            f"Could not find a spans column in annotations; looked for: {span_candidates} "
            f"but columns are: {list(annotations_df.columns)}"
        )

    dt_col = next((c for c in dt_candidates if c in annotations_df.columns), None)
    fname_col = next((c for c in fname_candidates if c in annotations_df.columns), None)

    allowed_labels: Optional[set[str]] = None
    if restrict_to_labels is not None:
        allowed_labels = {str(x) for x in restrict_to_labels}

    rows: list[dict[str, Any]] = []

    for _, rec in annotations_df.iterrows():
        # filename
        if fname_col is not None:
            fname = rec.get(fname_col, "")
        else:
            fname = ""
        fname = "" if pd.isna(fname) else str(fname)

        # datetime
        cdate = rec.get(dt_col, None) if dt_col is not None else None
        try:
            creation_dt = pd.to_datetime(cdate) if cdate is not None else pd.NaT
        except Exception:
            creation_dt = pd.NaT

        label_dict = rec.get(span_col, {})
        if isinstance(label_dict, str):
            # Sometimes stored as JSON string
            try:
                label_dict = json.loads(label_dict)
            except Exception:
                label_dict = {}
        if not isinstance(label_dict, dict):
            continue

        for lab, intervals in label_dict.items():
            lab_str = _normalize_label(lab)
            if allowed_labels is not None and lab_str not in allowed_labels:
                continue
            if not isinstance(intervals, (list, tuple)):
                continue
            for itv in intervals:
                if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
                    continue
                try:
                    onset = float(itv[0])
                    offset = float(itv[1])
                except Exception:
                    continue
                dur = offset - onset
                if not np.isfinite(dur) or dur <= 0:
                    continue
                rows.append(
                    {
                        "file_name": fname,
                        "creation_datetime": creation_dt,
                        "Syllable": lab_str,
                        "onset_ms": onset,
                        "offset_ms": offset,
                        "duration_ms": dur,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "file_name",
                "creation_datetime",
                "Syllable",
                "onset_ms",
                "offset_ms",
                "duration_ms",
            ]
        )

    df = pd.DataFrame.from_records(rows)
    if not pd.api.types.is_datetime64_any_dtype(df["creation_datetime"]):
        df["creation_datetime"] = pd.to_datetime(df["creation_datetime"], errors="coerce")
    return df


# ---- baseline / balancing helpers -------------------------------------------


def _compute_pre_baseline_stats(
    raw_phrases_df: pd.DataFrame, treatment_datetime: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute baseline (pre-treatment) statistics for each syllable using *all*
    pre-treatment phrase occurrences (not just the balanced subset).

    Returns a DataFrame with one row per syllable containing:
        "Syllable",
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Pre_Median_ms",
        "Pre_IQR_ms",
        "Pre_Median_IQR_ms",
        "Pre_Variance_IQR_ms2".
    """
    cols = [
        "Syllable",
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Pre_Median_ms",
        "Pre_IQR_ms",
        "Pre_Median_IQR_ms",
        "Pre_Variance_IQR_ms2",
    ]

    if raw_phrases_df.empty:
        return pd.DataFrame(columns=cols)

    pre_df = raw_phrases_df[raw_phrases_df["creation_datetime"] < treatment_datetime]
    if pre_df.empty:
        return pd.DataFrame(columns=cols)

    records = []
    for syll, g in pre_df.groupby("Syllable"):
        durs = g["duration_ms"].to_numpy(dtype=float)
        n = durs.size
        if n == 0:
            continue

        mean = float(np.mean(durs))
        if n > 1:
            var = float(np.var(durs, ddof=1))
            std = float(np.std(durs, ddof=1))
        else:
            var = np.nan
            std = np.nan

        median = float(np.median(durs))

        if n >= 2:
            q1, q3 = np.percentile(durs, [25, 75])
            iqr = float(q3 - q1)
        else:
            iqr = np.nan

        # We don't have per-song medians here, so we reuse the IQR of durations
        median_iqr = iqr

        if n >= 2:
            dev2 = (durs - mean) ** 2
            vq1, vq3 = np.percentile(dev2, [25, 75])
            var_iqr = float(vq3 - vq1)
        else:
            var_iqr = np.nan

        records.append(
            {
                "Syllable": syll,
                "Pre_N_phrases": int(n),
                "Pre_Mean_ms": mean,
                "Pre_Variance_ms2": var,
                "Pre_Std_ms": std,
                "Pre_Median_ms": median,
                "Pre_IQR_ms": iqr,
                "Pre_Median_IQR_ms": median_iqr,
                "Pre_Variance_IQR_ms2": var_iqr,
            }
        )

    return pd.DataFrame.from_records(records, columns=cols)


def _balance_syllable_occurrences_for_groups(
    df_syll: pd.DataFrame,
    treatment_datetime: pd.Timestamp,
    group_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Given all occurrences for a single syllable (with `creation_datetime`
    and `duration_ms`), split into Early Pre, Late Pre, and Post groups with
    equal N_phrases in each group.

    N_phrases is chosen as the largest possible balanced sample:

        n_per_group = min( floor(N_pre / 2), N_post )

    where:
        N_pre  = total # of pre-treatment phrases
        N_post = total # of post-treatment phrases

    If `group_size` is not None, it acts as an upper cap, i.e.:

        n_per_group = min(n_per_group, group_size)

    Returns
    -------
    df : pd.DataFrame
        One row per balanced phrase occurrence, with columns including
        "Syllable", "creation_datetime", "duration_ms", "Group",
        "N_pre_total", "N_post_total", "N_used_per_group".
    """
    base_cols = list(df_syll.columns) + ["Group", "N_pre_total", "N_post_total", "N_used_per_group"]

    if df_syll.empty:
        return pd.DataFrame(columns=base_cols)

    df_s = df_syll.sort_values("creation_datetime").reset_index(drop=True)

    pre_mask = df_s["creation_datetime"] < treatment_datetime
    pre = df_s[pre_mask]
    post = df_s[~pre_mask]

    n_pre = len(pre)
    n_post = len(post)

    if n_pre < 2 or n_post < 1:
        return pd.DataFrame(columns=base_cols)

    half_pre = n_pre // 2
    if half_pre <= 0:
        return pd.DataFrame(columns=base_cols)

    n = min(half_pre, n_post)
    if group_size is not None:
        n = min(n, int(group_size))

    if n <= 0:
        return pd.DataFrame(columns=base_cols)

    # Early Pre: earliest n pre phrases
    early_pre = pre.iloc[:n].copy()
    early_pre["Group"] = "Early Pre"

    # Late Pre: latest n pre phrases
    late_pre = pre.iloc[-n:].copy()
    late_pre["Group"] = "Late Pre"

    # Post: earliest n post-treatment phrases
    post_sel = post.iloc[:n].copy()
    post_sel["Group"] = "Post"

    for df_part in (early_pre, late_pre, post_sel):
        df_part["N_pre_total"] = n_pre
        df_part["N_post_total"] = n_post
        df_part["N_used_per_group"] = n

    out = pd.concat([early_pre, late_pre, post_sel], ignore_index=True)
    return out


def _compute_stats_from_balanced_occurrences(balanced_df: pd.DataFrame) -> pd.DataFrame:
    """
    From balanced per-phrase rows (with 'Syllable', 'Group', 'duration_ms',
    and the N_pre_total / N_post_total / N_used_per_group columns), compute a
    one-row-per-(Syllable, Group) statistics table.
    """
    cols = [
        "Syllable",
        "Group",
        "N_phrases",
        "Mean_ms",
        "SEM_ms",
        "Median_ms",
        "Std_ms",
        "Variance_ms2",
        "N_pre_total",
        "N_post_total",
        "N_used_per_group",
    ]

    if balanced_df.empty:
        return pd.DataFrame(columns=cols)

    def _agg(group: pd.DataFrame) -> pd.Series:
        durs = group["duration_ms"].to_numpy(dtype=float)
        n = durs.size

        mean = float(np.mean(durs)) if n > 0 else np.nan
        if n > 1:
            std = float(np.std(durs, ddof=1))
            var = float(np.var(durs, ddof=1))
            sem = std / np.sqrt(n)
        else:
            std = np.nan
            var = np.nan
            sem = np.nan

        median = float(np.median(durs)) if n > 0 else np.nan

        return pd.Series(
            {
                "N_phrases": n,
                "Mean_ms": mean,
                "SEM_ms": sem,
                "Median_ms": median,
                "Std_ms": std,
                "Variance_ms2": var,
                "N_pre_total": group["N_pre_total"].iloc[0],
                "N_post_total": group["N_post_total"].iloc[0],
                "N_used_per_group": group["N_used_per_group"].iloc[0],
            }
        )

    stats = (
        balanced_df.groupby(["Syllable", "Group"], as_index=False)
        .apply(_agg)
    )
    # In recent pandas, groupby+apply can give a MultiIndex on columns; flatten if needed.
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = stats.columns.get_level_values(-1)

    # Ensure column order
    return stats[cols]


def _attach_pre_post_metrics(
    phrase_stats: pd.DataFrame, pre_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Given per-(Syllable, Group) stats and baseline pre_stats (per Syllable),
    add Pre_* columns and Post-vs-Pre comparison metrics.

    Only rows with Group == "Post" receive non-NaN Pre_* and delta/ratio
    values; other rows have NaNs for Pre_* and numeric comparison columns,
    and False for boolean flags.
    """
    df = phrase_stats.copy()

    baseline_cols = [
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Pre_Median_ms",
        "Pre_IQR_ms",
        "Pre_Median_IQR_ms",
        "Pre_Variance_IQR_ms2",
    ]
    for col in baseline_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Mapping from syllable -> baseline metrics
    if not pre_stats.empty:
        pre_idx = pre_stats.set_index("Syllable")
    else:
        pre_idx = pd.DataFrame(columns=["Syllable"]).set_index("Syllable")

    mask_post = df["Group"] == "Post"
    syll_post = df.loc[mask_post, "Syllable"]
    if not syll_post.empty and not pre_idx.empty:
        for col in baseline_cols:
            if col in pre_idx.columns:
                df.loc[mask_post, col] = syll_post.map(pre_idx[col])

    # Initialize Post-vs-Pre columns
    comp_cols_numeric = [
        "Post_vs_Pre_Delta_Mean_ms",
        "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2",
        "Post_vs_Pre_Variance_Ratio",
        "Post_vs_Pre_Delta_Median_ms",
        "Post_vs_Pre_Median_Ratio",
    ]
    comp_cols_bool = [
        "Post_Mean_Increased",
        "Post_Variance_Increased",
        "Post_Mean_Above_Pre_Mean_Plus_1SD",
        "Post_Mean_Above_Pre_Mean_Plus_2SD",
    ]

    for col in comp_cols_numeric:
        if col not in df.columns:
            df[col] = np.nan
    for col in comp_cols_bool:
        if col not in df.columns:
            df[col] = False

    # Compute metrics only where we have Post rows and a valid baseline
    valid = mask_post & df["Pre_Mean_ms"].notna()

    # Mean deltas/ratios
    df.loc[valid, "Post_vs_Pre_Delta_Mean_ms"] = (
        df.loc[valid, "Mean_ms"] - df.loc[valid, "Pre_Mean_ms"]
    )
    den_mean = df.loc[valid, "Pre_Mean_ms"].replace(0, np.nan)
    df.loc[valid, "Post_vs_Pre_Mean_Ratio"] = df.loc[valid, "Mean_ms"] / den_mean

    # Variance deltas/ratios
    df.loc[valid, "Post_vs_Pre_Delta_Variance_ms2"] = (
        df.loc[valid, "Variance_ms2"] - df.loc[valid, "Pre_Variance_ms2"]
    )
    den_var = df.loc[valid, "Pre_Variance_ms2"].replace(0, np.nan)
    df.loc[valid, "Post_vs_Pre_Variance_Ratio"] = (
        df.loc[valid, "Variance_ms2"] / den_var
    )

    # Median deltas/ratios
    df.loc[valid, "Post_vs_Pre_Delta_Median_ms"] = (
        df.loc[valid, "Median_ms"] - df.loc[valid, "Pre_Median_ms"]
    )
    den_med = df.loc[valid, "Pre_Median_ms"].replace(0, np.nan)
    df.loc[valid, "Post_vs_Pre_Median_Ratio"] = df.loc[valid, "Median_ms"] / den_med

    # Boolean flags
    df.loc[valid, "Post_Mean_Increased"] = (
        df.loc[valid, "Mean_ms"] > df.loc[valid, "Pre_Mean_ms"]
    )
    df.loc[valid, "Post_Variance_Increased"] = (
        df.loc[valid, "Variance_ms2"] > df.loc[valid, "Pre_Variance_ms2"]
    )

    std_valid = valid & df["Pre_Std_ms"].notna()
    df.loc[std_valid, "Post_Mean_Above_Pre_Mean_Plus_1SD"] = (
        df.loc[std_valid, "Mean_ms"]
        > df.loc[std_valid, "Pre_Mean_ms"] + df.loc[std_valid, "Pre_Std_ms"]
    )
    df.loc[std_valid, "Post_Mean_Above_Pre_Mean_Plus_2SD"] = (
        df.loc[std_valid, "Mean_ms"]
        > df.loc[std_valid, "Pre_Mean_ms"] + 2 * df.loc[std_valid, "Pre_Std_ms"]
    )

    return df


def _build_phrase_duration_stats_single_bird(
    raw_phrases_df: pd.DataFrame,
    treatment_datetime: pd.Timestamp,
    group_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    High-level helper: from raw phrase occurrences for a single bird, compute
    balanced per-group stats + baseline pre metrics + post-vs-pre comparisons.

    Parameters
    ----------
    raw_phrases_df : pd.DataFrame
        Output of `build_phrase_occurrence_table` (or
        `_build_phrase_occurrences_from_annotations`), one row per phrase.
    treatment_datetime : pd.Timestamp
        Treatment date/time; phrases with creation_datetime < this are "pre".
    group_size : int or None, optional
        If None (recommended), use the largest possible balanced sample size
        per syllable:

            n_per_group = min( floor(N_pre / 2), N_post )

        If an integer is given, it acts as an upper cap on n_per_group.

    Returns
    -------
    stats_final : pd.DataFrame
        One row per (Syllable, Group) with columns matching the per-bird part
        of `compiled_phrase_duration_stats_with_prepost_metrics.csv` (minus
        the Animal ID column).
    """
    if raw_phrases_df.empty:
        # Empty schema with all the expected columns
        cols_order = [
            "Group",
            "Syllable",
            "N_phrases",
            "Mean_ms",
            "SEM_ms",
            "Median_ms",
            "Std_ms",
            "Variance_ms2",
            "Pre_N_phrases",
            "Pre_Mean_ms",
            "Pre_Variance_ms2",
            "Pre_Std_ms",
            "Pre_Median_ms",
            "Pre_IQR_ms",
            "Pre_Median_IQR_ms",
            "Pre_Variance_IQR_ms2",
            "Post_vs_Pre_Delta_Mean_ms",
            "Post_vs_Pre_Mean_Ratio",
            "Post_vs_Pre_Delta_Variance_ms2",
            "Post_vs_Pre_Variance_Ratio",
            "Post_vs_Pre_Delta_Median_ms",
            "Post_vs_Pre_Median_Ratio",
            "Post_Mean_Increased",
            "Post_Variance_Increased",
            "Post_Mean_Above_Pre_Mean_Plus_1SD",
            "Post_Mean_Above_Pre_Mean_Plus_2SD",
        ]
        return pd.DataFrame(columns=cols_order)

    # 1) Balanced occurrences per syllable
    balanced_list = []
    for syll, g in raw_phrases_df.groupby("Syllable"):
        bal = _balance_syllable_occurrences_for_groups(
            g, treatment_datetime=treatment_datetime, group_size=group_size
        )
        if not bal.empty:
            balanced_list.append(bal)

    if balanced_list:
        balanced_df = pd.concat(balanced_list, ignore_index=True)
    else:
        balanced_df = pd.DataFrame(
            columns=list(raw_phrases_df.columns)
            + ["Group", "N_pre_total", "N_post_total", "N_used_per_group"]
        )

    base_stats = _compute_stats_from_balanced_occurrences(balanced_df)

    # 2) Baseline pre metrics from *all* pre occurrences
    pre_stats = _compute_pre_baseline_stats(raw_phrases_df, treatment_datetime)

    # 3) Attach baselines + Post-vs-Pre metrics
    stats_with_pre = _attach_pre_post_metrics(base_stats, pre_stats)

    # 4) Reorder into final schema (no Animal ID yet)
    cols_order = [
        "Group",
        "Syllable",
        "N_phrases",
        "Mean_ms",
        "SEM_ms",
        "Median_ms",
        "Std_ms",
        "Variance_ms2",
        # baseline
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Pre_Median_ms",
        "Pre_IQR_ms",
        "Pre_Median_IQR_ms",
        "Pre_Variance_IQR_ms2",
        # comparisons
        "Post_vs_Pre_Delta_Mean_ms",
        "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2",
        "Post_vs_Pre_Variance_Ratio",
        "Post_vs_Pre_Delta_Median_ms",
        "Post_vs_Pre_Median_Ratio",
        "Post_Mean_Increased",
        "Post_Variance_Increased",
        "Post_Mean_Above_Pre_Mean_Plus_1SD",
        "Post_Mean_Above_Pre_Mean_Plus_2SD",
    ]

    for col in cols_order:
        if col not in stats_with_pre.columns:
            stats_with_pre[col] = np.nan

    stats_final = stats_with_pre[cols_order].copy()
    return stats_final


# ──────────────────────────────────────────────────────────────────────────────
# Public API — single bird
# ──────────────────────────────────────────────────────────────────────────────


def run_phrase_duration_balanced_syllable_usage(
    *,
    decoded_database_json: Union[str, Path],
    treatment_date: Union[str, pd.Timestamp],
    group_size: Optional[int] = None,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    animal_id: Optional[str] = None,
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
) -> BalancedSyllableUsageResult:
    """
    Run balanced syllable-usage phrase-duration stats for a single bird.

    This function:

      1) If `song_detection_json` is provided, uses
         `merge_annotations_from_split_songs.build_decoded_with_split_labels`
         to merge split-up songs and build a phrase-occurrence table from the
         merged annotations.

         Otherwise, falls back to reading the TweetyBERT decoded_database
         JSON directly via `build_phrase_occurrence_table`.

      2) Builds a one-row-per-phrase table.
      3) Splits occurrences into Early-Pre, Late-Pre, Post balanced groups
         per syllable.
      4) Computes per-group phrase-duration stats.
      5) Computes baseline pre-treatment metrics using *all* pre-treatment
         phrase occurrences.
      6) Computes post-vs-pre deltas, ratios, and threshold flags.

    Parameters
    ----------
    decoded_database_json : path-like
        Path to the TweetyBERT decoded_database JSON.
    treatment_date : str or pd.Timestamp
        Treatment date; phrases with creation_datetime < this are "pre".
    group_size : int or None, optional
        If None (recommended), use the largest possible balanced sample size
        for each (syllable, bird):

            n_per_group = min( floor(N_pre / 2), N_post )

        If an integer is given, it acts as an upper cap on n_per_group.
    restrict_to_labels : sequence of hashable, optional
        If provided, only these labels (after string conversion) are included.
    animal_id : str, optional
        If provided, stored in the returned BalancedSyllableUsageResult.
    song_detection_json : path-like or None, optional
        If provided, used together with `decoded_database_json` to build
        merged-song annotations via `build_decoded_with_split_labels`.
        If None, the legacy JSON-only path is used.
    max_gap_between_song_segments : int, default 500
        Passed to `build_decoded_with_split_labels`.
    merge_repeated_syllables : bool, default True
        Passed to `build_decoded_with_split_labels`.
    repeat_gap_ms : float, default 10.0
        Passed to `build_decoded_with_split_labels`.
    repeat_gap_inclusive : bool, default False
        Passed to `build_decoded_with_split_labels`.

    Returns
    -------
    result : BalancedSyllableUsageResult
        Contains:
          - phrase_duration_stats_df : aggregated per-(Syllable, Group) stats
                                      with pre-baseline + post-vs-pre metrics
                                      (no Animal ID column yet).
          - balanced_occurrences_df  : per-phrase balanced rows
          - raw_phrases_df           : all phrase occurrences from JSON or
                                       merged annotations
    """
    decoded_database_json = Path(decoded_database_json)
    t_datetime = pd.to_datetime(treatment_date)

    # Raw phrase occurrences across the whole recording set
    if song_detection_json is not None:
        # Use merged songs via merge_annotations_from_split_songs
        song_detection_json = Path(song_detection_json)
        ann_res = mps.build_decoded_with_split_labels(
            decoded_database_json=decoded_database_json,
            song_detection_json=song_detection_json,
            only_song_present=True,
            compute_durations=True,
            add_recording_datetime=True,
            songs_only=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=0,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
        )
        annotations_df = ann_res.annotations_appended_df.copy()
        raw_phrases_df = _build_phrase_occurrences_from_annotations(
            annotations_df=annotations_df,
            restrict_to_labels=restrict_to_labels,
        )
    else:
        # Legacy JSON-only path
        raw_phrases_df = build_phrase_occurrence_table(
            decoded_database_json=decoded_database_json,
            restrict_to_labels=restrict_to_labels,
        )

    # Balanced per-group stats + baselines + comparisons
    phrase_stats_df = _build_phrase_duration_stats_single_bird(
        raw_phrases_df=raw_phrases_df,
        treatment_datetime=t_datetime,
        group_size=group_size,
    )

    # We also want to keep the balanced per-phrase rows
    balanced_list = []
    if not raw_phrases_df.empty:
        for syll, g in raw_phrases_df.groupby("Syllable"):
            bal = _balance_syllable_occurrences_for_groups(
                g, treatment_datetime=t_datetime, group_size=group_size
            )
            if not bal.empty:
                balanced_list.append(bal)
    if balanced_list:
        balanced_df = pd.concat(balanced_list, ignore_index=True)
    else:
        balanced_df = pd.DataFrame(
            columns=list(raw_phrases_df.columns)
            + ["Group", "N_pre_total", "N_post_total", "N_used_per_group"]
        )

    return BalancedSyllableUsageResult(
        animal_id=animal_id,
        treatment_date=t_datetime,
        phrase_duration_stats_df=phrase_stats_df,
        balanced_occurrences_df=balanced_df,
        raw_phrases_df=raw_phrases_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API — multi-bird via Excel
# ──────────────────────────────────────────────────────────────────────────────


def _find_decoded_and_detection_json(
    decoded_root: Union[str, Path],
    animal_id: str,
    glob_pattern_decoded: str = "*decoded_database.json",
    glob_pattern_detect: str = "*song_detection*.json",
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Search `decoded_root` for a decoded_database JSON and a song_detection
    JSON for the given `animal_id`.

    Preferred layout (per-bird subfolder):

        decoded_root/
            <animal_id>/
                <animal_id>_decoded_database.json
                <animal_id>_song_detection.json

    Falls back to a recursive search under decoded_root if the per-bird
    subfolder is not found.

    Filters out macOS "._" resource files.

    Returns
    -------
    decoded_path, detect_path : (Path or None, Path or None)
        Each is the first match (sorted lexicographically), or None if no
        match is found.
    """
    decoded_root = Path(decoded_root)

    decoded_candidates: list[Path] = []
    detect_candidates: list[Path] = []

    bird_dir = decoded_root / animal_id
    if bird_dir.is_dir():
        decoded_candidates.extend(
            p
            for p in bird_dir.glob(glob_pattern_decoded)
            if not p.name.startswith("._")
        )
        detect_candidates.extend(
            p
            for p in bird_dir.glob(glob_pattern_detect)
            if not p.name.startswith("._")
        )
    else:
        # Fallback: search entire tree for paths containing the animal_id
        for p in decoded_root.rglob(glob_pattern_decoded):
            if p.name.startswith("._"):
                continue
            if animal_id in str(p):
                decoded_candidates.append(p)
        for p in decoded_root.rglob(glob_pattern_detect):
            if p.name.startswith("._"):
                continue
            if animal_id in str(p):
                detect_candidates.append(p)

    decoded_path = None
    detect_path = None

    if decoded_candidates:
        decoded_candidates = sorted(decoded_candidates)
        decoded_path = decoded_candidates[0]
        if len(decoded_candidates) > 1:
            rels = [str(c.relative_to(decoded_root)) for c in decoded_candidates]
            print(
                f"[WARN] Multiple decoded JSON candidates for Animal ID '{animal_id}': {rels}. "
                f"Using the first."
            )

    if detect_candidates:
        detect_candidates = sorted(detect_candidates)
        detect_path = detect_candidates[0]
        if len(detect_candidates) > 1:
            rels = [str(c.relative_to(decoded_root)) for c in detect_candidates]
            print(
                f"[WARN] Multiple detection JSON candidates for Animal ID '{animal_id}': {rels}. "
                f"Using the first."
            )

    return decoded_path, detect_path


def run_balanced_syllable_usage_from_metadata_excel(
    *,
    excel_path: Union[str, Path],
    decoded_root: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    group_size: Optional[int] = None,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    glob_pattern: str = "*decoded_database.json",
    compiled_filename: str = "usage_balanced_phrase_duration_stats.csv",
) -> BalancedFolderResult:
    """
    Batch-run balanced syllable-usage phrase-duration stats for all birds
    listed in an Excel metadata sheet, and save a compiled CSV.

    Assumes `decoded_root` contains one subfolder per Animal ID, each with
    a decoded_database JSON and a song_detection JSON, e.g.:

        decoded_root/
            USA5283/
                USA5283_decoded_database.json
                USA5283_song_detection.json
            USA5288/
                USA5288_decoded_database.json
                USA5288_song_detection.json
            ...

    The output CSV has columns:

        "Group", "Syllable", "N_phrases",
        "Mean_ms", "SEM_ms", "Median_ms", "Std_ms", "Variance_ms2",
        <id_col>,
        "Pre_N_phrases", "Pre_Mean_ms", "Pre_Variance_ms2",
        "Pre_Std_ms", "Pre_Median_ms", "Pre_IQR_ms",
        "Pre_Median_IQR_ms", "Pre_Variance_IQR_ms2",
        "Post_vs_Pre_Delta_Mean_ms", "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2", "Post_vs_Pre_Variance_Ratio",
        "Post_vs_Pre_Delta_Median_ms", "Post_vs_Pre_Median_Ratio",
        "Post_Mean_Increased", "Post_Variance_Increased",
        "Post_Mean_Above_Pre_Mean_Plus_1SD",
        "Post_Mean_Above_Pre_Mean_Plus_2SD"

    which matches the structure of
    `compiled_phrase_duration_stats_with_prepost_metrics.csv`
    (aside from the exact string used for the ID column).

    Parameters
    ----------
    excel_path : path-like
        Excel metadata file containing at least the columns given by
        `id_col` and `treatment_date_col`.
    decoded_root : path-like
        Root folder under which per-bird decoded_database and
        song_detection JSONs live in subdirectories.
    sheet_name : int or str, default 0
        Excel sheet name or index.
    id_col : str, default "Animal ID"
        Column in the Excel sheet that stores the animal/bird ID.
    treatment_date_col : str, default "Treatment date"
        Column in the Excel sheet that stores the treatment date.
    group_size : int or None, optional
        Passed through to `run_phrase_duration_balanced_syllable_usage`.
        If None (recommended), the per-(bird, syllable) n_per_group is
        chosen as:

            n_per_group = min( floor(N_pre / 2), N_post )

        i.e. as large as possible while keeping groups balanced.
    restrict_to_labels : sequence of hashable, optional
        If provided, only these labels (after string conversion) are included.
    glob_pattern : str, default "*decoded_database.json"
        Pattern used when searching for decoded_database JSONs under
        `decoded_root`. (The detection JSON pattern is inferred as
        "*song_detection*.json".)
    compiled_filename : str, default "usage_balanced_phrase_duration_stats.csv"
        Name of the compiled CSV file to save under `decoded_root`.

    Returns
    -------
    result : BalancedFolderResult
        - compiled_stats_df  : concatenated stats across all birds
        - compiled_stats_path: path to the compiled CSV
        - per_animal_results : mapping {animal_id → BalancedSyllableUsageResult}
    """
    excel_path = Path(excel_path)
    decoded_root = Path(decoded_root)

    meta_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    if id_col not in meta_df.columns:
        raise KeyError(f"id_col='{id_col}' not found in Excel sheet columns.")

    if treatment_date_col not in meta_df.columns:
        raise KeyError(
            f"treatment_date_col='{treatment_date_col}' not found in Excel sheet columns."
        )

    per_animal_results: Dict[str, BalancedSyllableUsageResult] = {}
    compiled_frames: list[pd.DataFrame] = []

    for _, row in meta_df.iterrows():
        animal_raw = row.get(id_col)
        if pd.isna(animal_raw):
            continue
        animal_id = str(animal_raw).strip()
        if not animal_id:
            continue

        tdate_raw = row.get(treatment_date_col)
        if pd.isna(tdate_raw):
            print(f"[WARN] No treatment date for Animal ID '{animal_id}'. Skipping.")
            continue
        t_datetime = pd.to_datetime(tdate_raw)

        decoded_path, detect_path = _find_decoded_and_detection_json(
            decoded_root=decoded_root,
            animal_id=animal_id,
            glob_pattern_decoded=glob_pattern,
            glob_pattern_detect="*song_detection*.json",
        )
        if decoded_path is None or detect_path is None:
            print(
                f"[WARN] '{animal_id}': could not find both JSONs under {decoded_root / animal_id}.\n"
                f"       decoded: {decoded_path}\n"
                f"       detect : {detect_path}"
            )
            continue

        rel_dec = decoded_path.relative_to(decoded_root)
        rel_det = detect_path.relative_to(decoded_root)
        print(
            f"[RUN] {animal_id} | decoded={rel_dec} | detect={rel_det} | treatment_date={t_datetime}"
        )

        single_res = run_phrase_duration_balanced_syllable_usage(
            decoded_database_json=decoded_path,
            treatment_date=t_datetime,
            group_size=group_size,
            restrict_to_labels=restrict_to_labels,
            animal_id=animal_id,
            song_detection_json=detect_path,
        )

        if single_res.phrase_duration_stats_df.empty:
            print(f"[INFO] No balanced stats for '{animal_id}'. Skipping.")
            continue

        per_animal_results[animal_id] = single_res

        df_stats = single_res.phrase_duration_stats_df.copy()
        df_stats[id_col] = animal_id

        compiled_frames.append(df_stats)

    # Build compiled DataFrame
    if compiled_frames:
        compiled_df = pd.concat(compiled_frames, ignore_index=True)
    else:
        # Empty but with correct columns
        compiled_df = pd.DataFrame(
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
                "Pre_N_phrases",
                "Pre_Mean_ms",
                "Pre_Variance_ms2",
                "Pre_Std_ms",
                "Pre_Median_ms",
                "Pre_IQR_ms",
                "Pre_Median_IQR_ms",
                "Pre_Variance_IQR_ms2",
                "Post_vs_Pre_Delta_Mean_ms",
                "Post_vs_Pre_Mean_Ratio",
                "Post_vs_Pre_Delta_Variance_ms2",
                "Post_vs_Pre_Variance_Ratio",
                "Post_vs_Pre_Delta_Median_ms",
                "Post_vs_Pre_Median_Ratio",
                "Post_Mean_Increased",
                "Post_Variance_Increased",
                "Post_Mean_Above_Pre_Mean_Plus_1SD",
                "Post_Mean_Above_Pre_Mean_Plus_2SD",
            ]
        )

    # Ensure column order matches the reference CSV structure
    cols_order = [
        "Group",
        "Syllable",
        "N_phrases",
        "Mean_ms",
        "SEM_ms",
        "Median_ms",
        "Std_ms",
        "Variance_ms2",
        id_col,
        "Pre_N_phrases",
        "Pre_Mean_ms",
        "Pre_Variance_ms2",
        "Pre_Std_ms",
        "Pre_Median_ms",
        "Pre_IQR_ms",
        "Pre_Median_IQR_ms",
        "Pre_Variance_IQR_ms2",
        "Post_vs_Pre_Delta_Mean_ms",
        "Post_vs_Pre_Mean_Ratio",
        "Post_vs_Pre_Delta_Variance_ms2",
        "Post_vs_Pre_Variance_Ratio",
        "Post_vs_Pre_Delta_Median_ms",
        "Post_vs_Pre_Median_Ratio",
        "Post_Mean_Increased",
        "Post_Variance_Increased",
        "Post_Mean_Above_Pre_Mean_Plus_1SD",
        "Post_Mean_Above_Pre_Mean_Plus_2SD",
    ]
    for col in cols_order:
        if col not in compiled_df.columns:
            # For booleans, default False; for numerics, NaN.
            if col.startswith("Post_Mean_") or col.startswith("Post_Variance_"):
                compiled_df[col] = False
            else:
                compiled_df[col] = np.nan

    compiled_df = compiled_df[cols_order].copy()

    # Save compiled CSV
    compiled_path = decoded_root / compiled_filename
    compiled_df.to_csv(compiled_path, index=False)
    print(f"[SAVE] Compiled usage-balanced stats CSV: {compiled_path}")

    return BalancedFolderResult(
        compiled_stats_df=compiled_df,
        compiled_stats_path=compiled_path,
        per_animal_results=per_animal_results,
    )

"""
from pathlib import Path
import importlib
import phrase_duration_balanced_syllable_usage as pbsu

importlib.reload(pbsu)

decoded_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
excel_path   = decoded_root / "Area_X_lesion_metadata.xlsx"
folder_res = pbsu.run_balanced_syllable_usage_from_metadata_excel(
    excel_path=excel_path,
    decoded_root=decoded_root,
    sheet_name=0,                      # or a sheet name like "metadata"
    id_col="Animal ID",
    treatment_date_col="Treatment date",
    group_size=None,                   # None → auto-balance per (bird, syllable)
    restrict_to_labels=None,
    glob_pattern="*decoded_database.json",      # decoded JSON pattern
    compiled_filename="usage_balanced_phrase_duration_stats.csv",
)

compiled_df   = folder_res.compiled_stats_df
compiled_path = folder_res.compiled_stats_path

print(compiled_df.head())
print("Saved compiled CSV to:", compiled_path)

# You can also grab one bird's result:
per_animal = folder_res.per_animal_results
print(per_animal.keys())  # dict of {animal_id: BalancedSyllableUsageResult}


"""