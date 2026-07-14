#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure3_panelE_longevity_normalized_sd_cv_v4.py

Create medial+lateral-lesion-only longitudinal plots for the persistence of
phrase-duration variability. This companion script is intended to replace the
all-group raw-SD overlay previously used for Figure 3 Panel E.

Updates in v4
-------------
1. Summary-only plots use tighter y-axis limits based on the rolling median and
   IQR, which reduces excess white space below the curves.
2. The multiline y-axis labels, sparse symlog ticks, plain-number log tick
   labels, and summary-only timecourse versions from v3 are retained.

Why this version
----------------
1. Only pooled medial+lateral (M+L) lesion birds are shown in the main plots.
2. Each bird is normalized to its own late-pre-lesion baseline.
3. Birds are weighted equally in the rolling group summary: the script first
   calculates one rolling value per bird per day, then summarizes across birds.
4. The group summary can stop when fewer than a user-defined number of birds
   contribute, while individual trajectories continue.
5. Absolute SD, normalized SD, heatmap, small-multiple, and persistence outputs
   are generated.
6. CV plots are generated when a per-rendition long CSV is supplied, or when the
   daily variance CSV contains a daily mean-duration column.

Primary outputs
---------------
Tables
  panelE_ML_daily_SD_normalized.csv
  panelE_ML_SD_equal_weight_rolling_summary.csv
  panelE_ML_SD_persistence_summary.csv
  panelE_ML_daily_CV_normalized.csv                         [when CV available]
  panelE_ML_CV_equal_weight_rolling_summary.csv             [when CV available]
  panelE_ML_CV_persistence_summary.csv                      [when CV available]

Figures
  panelE_ML_SD_delta_timecourse.png/pdf                     recommended main
  panelE_ML_SD_percent_change_timecourse.png/pdf
  panelE_ML_SD_fold_change_timecourse.png/pdf
  panelE_ML_SD_delta_heatmap.png/pdf
  panelE_ML_SD_delta_small_multiples.png/pdf
  panelE_ML_SD_persistence_fraction_above_pre95.png/pdf
  panelE_ML_SD_persistence_timeline.png/pdf
  panelE_ML_CV_delta_timecourse.png/pdf                     recommended CV check
  panelE_ML_CV_fold_change_timecourse.png/pdf
  panelE_ML_CV_delta_heatmap.png/pdf
  panelE_ML_CV_persistence_fraction_above_pre95.png/pdf
  panelE_ML_CV_persistence_timeline.png/pdf
  panelE_ML_SD_fold_change_timecourse_summary_only.png/pdf   [new in v2]
  panelE_ML_CV_fold_change_timecourse_summary_only.png/pdf   [new in v2]

Recommended usage
-----------------
First run the existing Figure 3 script so it writes the pooled-pre/post-selected
animal × syllable pairs. Then run:

python figure3_panelE_longevity_normalized_sd_cv_v1.py \
  --daily-variance-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/batch_aligned_phrase_duration_variance_all.csv" \
  --selected-pairs-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_pooled_prepost_top30_ALLdaily_SDonly_seconds/panel_C_selected_animal_syllable_pairs.csv" \
  --duration-long-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/all_birds_phrase_duration_long_latepre_post_epochs.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_panelE_ML_longevity_SD_CV" \
  --baseline-start-day -14 \
  --baseline-end-day -1 \
  --x-min -30 \
  --x-max 30 \
  --smooth-window-days 7 \
  --min-animals-for-summary 3 \
  --no-show

If the long CSV already has a relative-day column, it will be detected. If it
only has a recording date, provide --long-date-col and the metadata workbook
must contain a treatment-date column.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


# -----------------------------------------------------------------------------
# Canonical group labels
# -----------------------------------------------------------------------------
SHAM_GROUP = "sham saline injection"
LATERAL_ONLY_GROUP = "Lateral lesion only"
COMPLETE_ML_GROUP = "Complete Medial and Lateral lesion"
PARTIAL_ML_GROUP = "Partial Medial and Lateral lesion"
POOLED_ML_GROUP = "Complete and partial medial and lateral lesion"
ML_GROUPS = {COMPLETE_ML_GROUP, PARTIAL_ML_GROUP, POOLED_ML_GROUP}

# Kept deliberately close to the existing Figure 3 appearance.
ML_LINE_COLOR = "#4D4D4D"
ML_INDIVIDUAL_COLOR = "#9E9E9E"
LESION_COLOR = "#D62728"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def norm_text(value: Any) -> str:
    return standardize_text(value).lower().replace("_", " ").replace("-", " ")


def normalize_group(value: Any) -> str:
    """Map old and new lesion labels onto the Figure 3 display groups."""
    s = norm_text(value)
    compact = s.replace(" ", "")
    if not s:
        return "unknown"
    if "sham" in s or ("saline" in s and "lesion" not in s):
        return SHAM_GROUP
    if "single hit" in s or "lateral only" in s or "lateral hit only" in s:
        return LATERAL_ONLY_GROUP
    if "area x not visible" in s or "large lesion" in s or "largelesion" in compact:
        return COMPLETE_ML_GROUP
    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE_ML_GROUP
    if ("medial" in s and "lateral" in s) or "m+l" in compact or "combined" in s:
        return PARTIAL_ML_GROUP
    if compact in {"ml", "mediallateral", "medial+lateral"}:
        return PARTIAL_ML_GROUP
    return standardize_text(value) or "unknown"


def pooled_group(value: Any) -> str:
    group = normalize_group(value)
    return POOLED_ML_GROUP if group in ML_GROUPS else group


def find_column(
    df: pd.DataFrame,
    requested: Optional[str],
    candidates: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> Optional[str]:
    if requested:
        if requested in df.columns:
            return requested
        raise ValueError(
            f"Requested {label} column {requested!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )

    normalized = {
        "".join(ch.lower() for ch in str(col) if ch.isalnum()): col
        for col in df.columns
    }
    for candidate in candidates:
        key = "".join(ch.lower() for ch in candidate if ch.isalnum())
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not identify the {label} column. Available columns: {list(df.columns)}"
        )
    return None


def pretty_axes(ax: plt.Axes, tick_fontsize: float = 12.5) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.0, length=4)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def plain_numeric_tick_formatter(x: float, pos: object = None) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    if abs(x) < 1e-10:
        return "0"
    ax = abs(x)
    if ax >= 100:
        txt = f"{x:.0f}"
    elif ax >= 10:
        txt = f"{x:.1f}"
    elif ax >= 1:
        txt = f"{x:.2f}"
    else:
        txt = f"{x:.3f}"
    return txt.rstrip("0").rstrip(".")


def apply_plain_log_tick_labels(ax: plt.Axes, axis: str = "y") -> None:
    formatter = FuncFormatter(plain_numeric_tick_formatter)
    locator = LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    if axis == "y":
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())


def apply_sparse_symlog_tick_labels(ax: plt.Axes, lo: float, hi: float) -> None:
    """Use a sparse, readable set of symlog ticks centered on zero."""
    preferred = [-1000, -100, -10, 0, 10, 100, 1000]
    ticks = [t for t in preferred if lo <= t <= hi]
    if 0 not in ticks:
        ticks.append(0)
    # If the range is small and no non-zero ticks survived, add a few small values.
    if len(ticks) <= 1:
        extra = [-10, -1, 0, 1, 10]
        ticks = [t for t in extra if lo <= t <= hi]
        if 0 not in ticks:
            ticks.append(0)
    ticks = sorted(set(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_numeric_tick_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save_figure(fig: plt.Figure, base_path: Path, dpi: int, show: bool) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)


def robust_limits(
    values: Iterable[float],
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    include: Optional[float] = None,
    pad_fraction: float = 0.08,
) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanpercentile(vals, low_percentile))
    hi = float(np.nanpercentile(vals, high_percentile))
    if include is not None:
        lo = min(lo, float(include))
        hi = max(hi, float(include))
    if hi <= lo:
        span = max(abs(lo), 1.0)
        lo -= 0.05 * span
        hi += 0.05 * span
    span = hi - lo
    return lo - pad_fraction * span, hi + pad_fraction * span


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
def _excel_sheets(path: Path, preferred: Optional[str]) -> list[str]:
    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)
    if preferred and preferred in sheets:
        return [preferred] + [s for s in sheets if s != preferred]
    return sheets


def load_metadata_maps(
    metadata_excel: Optional[Path],
    *,
    preferred_sheet: Optional[str],
    animal_col: str,
    hit_type_col: Optional[str],
    treatment_date_col: Optional[str],
) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    if metadata_excel is None:
        return {}, {}
    if not metadata_excel.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_excel}")

    hit_map: dict[str, str] = {}
    date_map: dict[str, pd.Timestamp] = {}

    for sheet in _excel_sheets(metadata_excel, preferred_sheet):
        df = pd.read_excel(metadata_excel, sheet_name=sheet)
        if animal_col not in df.columns:
            continue

        if not hit_map:
            hit_col = find_column(
                df,
                hit_type_col if hit_type_col in df.columns else None,
                ["Lesion hit type", "Hit type", "lesion_group", "Treatment type"],
                "metadata hit type",
                required=False,
            )
            if hit_col:
                for _, row in df[[animal_col, hit_col]].dropna(subset=[animal_col]).iterrows():
                    animal = standardize_text(row[animal_col])
                    if animal:
                        hit_map[animal] = pooled_group(row[hit_col])

        if not date_map:
            date_col = find_column(
                df,
                treatment_date_col if treatment_date_col in df.columns else None,
                ["Treatment date", "Lesion date", "Surgery date", "treatment_date"],
                "treatment date",
                required=False,
            )
            if date_col:
                temp = df[[animal_col, date_col]].copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                for _, row in temp.dropna(subset=[animal_col, date_col]).iterrows():
                    animal = standardize_text(row[animal_col])
                    if animal:
                        date_map[animal] = pd.Timestamp(row[date_col]).normalize()

        if hit_map and date_map:
            break

    return hit_map, date_map


# -----------------------------------------------------------------------------
# Selected animal × syllable pairs
# -----------------------------------------------------------------------------
def load_selected_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bird_col = find_column(
        df,
        None,
        ["animal_id", "Animal ID", "bird", "Bird ID"],
        "selected-pairs animal",
    )
    syllable_col = find_column(
        df,
        None,
        ["syllable", "Syllable", "label", "syllable_label"],
        "selected-pairs syllable",
    )
    out = df[[bird_col, syllable_col]].copy()
    out.columns = ["animal_id", "syllable"]
    out["animal_id"] = out["animal_id"].astype(str)
    out["syllable"] = out["syllable"].astype(str)
    return out.drop_duplicates().reset_index(drop=True)


def select_pairs_from_scatter(
    scatter_csv: Path,
    *,
    top_percentile: float,
    rank_on: str,
    pre_group: str,
    post_group: str,
    min_n_phrases: int,
) -> pd.DataFrame:
    df = pd.read_csv(scatter_csv)
    bird_col = find_column(df, None, ["Animal ID", "animal_id", "bird"], "scatter animal")
    syll_col = find_column(df, None, ["Syllable", "syllable", "label"], "scatter syllable")
    group_col = find_column(df, None, ["Group", "group", "epoch"], "scatter group")
    variance_col = find_column(
        df,
        None,
        ["Variance_ms2", "Variance (ms^2)", "variance_ms2", "variance"],
        "scatter variance",
    )
    n_col = find_column(
        df,
        None,
        ["N_phrases", "n_phrases", "N", "count"],
        "scatter phrase count",
        required=False,
    )

    work = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    work[variance_col] = pd.to_numeric(work[variance_col], errors="coerce")
    if n_col:
        work[n_col] = pd.to_numeric(work[n_col], errors="coerce")
        work = work[work[n_col].fillna(0) >= min_n_phrases]

    long = (
        work.groupby([bird_col, syll_col, group_col], dropna=False)[variance_col]
        .mean()
        .reset_index()
    )
    wide = long.pivot_table(
        index=[bird_col, syll_col], columns=group_col, values=variance_col, aggfunc="mean"
    ).reset_index()
    if pre_group not in wide.columns or post_group not in wide.columns:
        raise ValueError(
            f"Could not find both {pre_group!r} and {post_group!r} after pivoting {scatter_csv}."
        )
    wide = wide.dropna(subset=[pre_group, post_group]).copy()

    rank_on = rank_on.lower()
    if rank_on == "pre":
        wide["rank_metric"] = wide[pre_group]
    elif rank_on == "post":
        wide["rank_metric"] = wide[post_group]
    elif rank_on == "max":
        wide["rank_metric"] = wide[[pre_group, post_group]].max(axis=1)
    elif rank_on == "pooled":
        wide["rank_metric"] = wide[[pre_group, post_group]].mean(axis=1)
    else:
        raise ValueError("rank_on must be pre, post, max, or pooled")

    keep = []
    for _, g in wide.groupby(bird_col, dropna=False):
        threshold = np.nanpercentile(g["rank_metric"].to_numpy(dtype=float), top_percentile)
        keep.append(g[g["rank_metric"] >= threshold])
    selected = pd.concat(keep, ignore_index=True) if keep else wide.iloc[0:0].copy()
    selected = selected[[bird_col, syll_col]].copy()
    selected.columns = ["animal_id", "syllable"]
    selected["animal_id"] = selected["animal_id"].astype(str)
    selected["syllable"] = selected["syllable"].astype(str)
    return selected.drop_duplicates().reset_index(drop=True)


# -----------------------------------------------------------------------------
# Daily SD and CV tables
# -----------------------------------------------------------------------------
def prepare_daily_variance(
    path: Path,
    *,
    selected_pairs: pd.DataFrame,
    hit_map: dict[str, str],
    x_min: int,
    x_max: int,
    mean_col_requested: Optional[str],
    mean_unit: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(path)
    bird_col = find_column(df, None, ["animal_id", "Animal ID", "bird"], "daily animal")
    syll_col = find_column(df, None, ["syllable", "Syllable", "label"], "daily syllable")
    day_col = find_column(
        df,
        None,
        ["relative_day", "Days relative to lesion", "day_relative_to_lesion", "relative day"],
        "daily relative day",
    )
    variance_col = find_column(
        df,
        None,
        ["Variance (ms^2)", "Variance_ms2", "variance_ms2", "variance"],
        "daily variance",
    )
    group_col = find_column(
        df,
        None,
        ["lesion_group", "display_group", "Lesion hit type", "group"],
        "daily lesion group",
        required=False,
    )
    mean_col = find_column(
        df,
        mean_col_requested,
        [
            "Mean (ms)",
            "Mean_ms",
            "Mean phrase duration (ms)",
            "Mean duration (ms)",
            "mean_duration_ms",
            "mean_phrase_duration_ms",
            "mean_duration",
            "Mean",
        ],
        "daily mean duration",
        required=False,
    )

    work = df.rename(
        columns={bird_col: "animal_id", syll_col: "syllable", day_col: "relative_day", variance_col: "variance_ms2"}
    ).copy()
    work["animal_id"] = work["animal_id"].astype(str)
    work["syllable"] = work["syllable"].astype(str)
    work["relative_day"] = pd.to_numeric(work["relative_day"], errors="coerce")
    work["variance_ms2"] = pd.to_numeric(work["variance_ms2"], errors="coerce")
    work = work.dropna(subset=["relative_day", "variance_ms2"])
    work = work[(work["relative_day"] >= x_min) & (work["relative_day"] <= x_max)]
    work = work.merge(selected_pairs, on=["animal_id", "syllable"], how="inner")

    if group_col:
        if group_col in work.columns:
            work["group_raw"] = work[group_col]
        else:
            # group_col may have been renamed only if it collided, which is unlikely.
            work["group_raw"] = "unknown"
    else:
        work["group_raw"] = "unknown"

    work["display_group"] = [
        hit_map.get(animal, pooled_group(group))
        for animal, group in zip(work["animal_id"], work["group_raw"])
    ]
    work["display_group"] = work["display_group"].map(pooled_group)
    work["sd_s"] = np.sqrt(np.clip(work["variance_ms2"].to_numpy(dtype=float), 0, None)) / 1000.0

    sd_daily = (
        work.groupby(["display_group", "animal_id", "relative_day"], dropna=False)
        .agg(sd_s=("sd_s", "median"), n_selected_syllables=("syllable", "nunique"))
        .reset_index()
    )

    cv_daily: Optional[pd.DataFrame] = None
    if mean_col:
        # Recover the mean column from the original dataframe before groupby.
        mean_values = pd.to_numeric(df[mean_col], errors="coerce")
        if mean_unit == "s":
            mean_ms = mean_values * 1000.0
        else:
            mean_ms = mean_values
        df_with_mean = df.copy()
        df_with_mean["__mean_ms"] = mean_ms
        temp = df_with_mean.rename(
            columns={bird_col: "animal_id", syll_col: "syllable", day_col: "relative_day", variance_col: "variance_ms2"}
        )
        temp["animal_id"] = temp["animal_id"].astype(str)
        temp["syllable"] = temp["syllable"].astype(str)
        temp["relative_day"] = pd.to_numeric(temp["relative_day"], errors="coerce")
        temp["variance_ms2"] = pd.to_numeric(temp["variance_ms2"], errors="coerce")
        temp = temp.dropna(subset=["relative_day", "variance_ms2", "__mean_ms"])
        temp = temp[(temp["relative_day"] >= x_min) & (temp["relative_day"] <= x_max)]
        temp = temp.merge(selected_pairs, on=["animal_id", "syllable"], how="inner")
        if group_col and group_col in temp.columns:
            temp["group_raw"] = temp[group_col]
        else:
            temp["group_raw"] = "unknown"
        temp["display_group"] = [
            hit_map.get(animal, pooled_group(group))
            for animal, group in zip(temp["animal_id"], temp["group_raw"])
        ]
        temp["display_group"] = temp["display_group"].map(pooled_group)
        temp["cv"] = np.sqrt(np.clip(temp["variance_ms2"].to_numpy(dtype=float), 0, None)) / temp["__mean_ms"].to_numpy(dtype=float)
        temp.loc[~np.isfinite(temp["cv"]) | (temp["__mean_ms"] <= 0), "cv"] = np.nan
        cv_daily = (
            temp.groupby(["display_group", "animal_id", "relative_day"], dropna=False)
            .agg(cv=("cv", "median"), n_selected_syllables=("syllable", "nunique"))
            .reset_index()
        )

    return sd_daily, cv_daily


def prepare_daily_cv_from_long(
    path: Path,
    *,
    selected_pairs: pd.DataFrame,
    hit_map: dict[str, str],
    treatment_date_map: dict[str, pd.Timestamp],
    bird_col_requested: Optional[str],
    syllable_col_requested: Optional[str],
    day_col_requested: Optional[str],
    date_col_requested: Optional[str],
    duration_col_requested: Optional[str],
    group_col_requested: Optional[str],
    duration_unit: str,
    min_renditions: int,
    x_min: int,
    x_max: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    bird_col = find_column(df, bird_col_requested, ["animal_id", "Animal ID", "bird"], "long animal")
    syll_col = find_column(df, syllable_col_requested, ["syllable", "Syllable", "label"], "long syllable")
    duration_col = find_column(
        df,
        duration_col_requested,
        ["Phrase Duration (ms)", "phrase_duration_ms", "duration_ms", "duration", "Phrase Duration (s)"],
        "long phrase duration",
    )
    group_col = find_column(
        df,
        group_col_requested,
        ["lesion_group", "Lesion hit type", "group", "Treatment type"],
        "long lesion group",
        required=False,
    )
    day_col = find_column(
        df,
        day_col_requested,
        ["relative_day", "Days relative to lesion", "day_relative_to_lesion", "relative day"],
        "long relative day",
        required=False,
    )
    date_col = None
    if day_col is None:
        date_col = find_column(
            df,
            date_col_requested,
            ["Date", "date", "recording_date", "Recording date"],
            "long recording date",
            required=False,
        )
        if date_col is None:
            raise ValueError(
                "CV calculation needs either a relative-day column or a recording-date column. "
                f"Available columns: {list(df.columns)}"
            )
        if not treatment_date_map:
            raise ValueError(
                "The long CSV has no relative-day column, and no treatment-date map could be "
                "loaded from metadata. Provide --long-day-col or metadata containing Treatment date."
            )

    work = df.rename(columns={bird_col: "animal_id", syll_col: "syllable", duration_col: "duration_raw"}).copy()
    work["animal_id"] = work["animal_id"].astype(str)
    work["syllable"] = work["syllable"].astype(str)
    work["duration_s"] = pd.to_numeric(work["duration_raw"], errors="coerce")
    if duration_unit == "ms":
        work["duration_s"] = work["duration_s"] / 1000.0

    if day_col:
        work["relative_day"] = pd.to_numeric(work[day_col], errors="coerce")
    else:
        work["recording_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
        work["treatment_date"] = work["animal_id"].map(treatment_date_map)
        work["relative_day"] = (work["recording_date"] - work["treatment_date"]).dt.days

    if group_col:
        work["group_raw"] = work[group_col]
    else:
        work["group_raw"] = "unknown"
    work["display_group"] = [
        hit_map.get(animal, pooled_group(group))
        for animal, group in zip(work["animal_id"], work["group_raw"])
    ]
    work["display_group"] = work["display_group"].map(pooled_group)

    work = work[
        np.isfinite(work["duration_s"])
        & (work["duration_s"] > 0)
        & np.isfinite(work["relative_day"])
    ].copy()
    work = work[(work["relative_day"] >= x_min) & (work["relative_day"] <= x_max)]
    work = work.merge(selected_pairs, on=["animal_id", "syllable"], how="inner")

    per_syllable = (
        work.groupby(["display_group", "animal_id", "syllable", "relative_day"], dropna=False)["duration_s"]
        .agg(n="size", mean_s="mean", sd_s="std")
        .reset_index()
    )
    per_syllable = per_syllable[per_syllable["n"] >= min_renditions].copy()
    per_syllable["cv"] = per_syllable["sd_s"] / per_syllable["mean_s"]
    per_syllable.loc[~np.isfinite(per_syllable["cv"]) | (per_syllable["mean_s"] <= 0), "cv"] = np.nan

    daily = (
        per_syllable.groupby(["display_group", "animal_id", "relative_day"], dropna=False)
        .agg(
            cv=("cv", "median"),
            n_selected_syllables=("syllable", "nunique"),
            total_renditions=("n", "sum"),
        )
        .reset_index()
    )
    return daily


# -----------------------------------------------------------------------------
# Baseline normalization and equal-bird rolling summary
# -----------------------------------------------------------------------------
def add_baseline_normalization(
    daily: pd.DataFrame,
    *,
    value_col: str,
    baseline_start_day: int,
    baseline_end_day: int,
    min_baseline_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = daily.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["relative_day"] = pd.to_numeric(work["relative_day"], errors="coerce")
    work = work.dropna(subset=[value_col, "relative_day"])

    baseline_rows = []
    for animal, g in work.groupby("animal_id", sort=True):
        pre = g[
            (g["relative_day"] >= baseline_start_day)
            & (g["relative_day"] <= baseline_end_day)
        ][value_col].dropna().to_numpy(dtype=float)
        if pre.size < min_baseline_days:
            warnings.warn(
                f"Skipping {animal}: only {pre.size} baseline days in "
                f"[{baseline_start_day}, {baseline_end_day}], need {min_baseline_days}."
            )
            continue
        q25 = float(np.nanpercentile(pre, 25))
        q75 = float(np.nanpercentile(pre, 75))
        baseline_rows.append(
            {
                "animal_id": animal,
                "value_col": value_col,
                "baseline_start_day": baseline_start_day,
                "baseline_end_day": baseline_end_day,
                "n_baseline_days": int(pre.size),
                "baseline_mean": float(np.nanmean(pre)),
                "baseline_sd": float(np.nanstd(pre, ddof=1)) if pre.size >= 2 else np.nan,
                "baseline_median": float(np.nanmedian(pre)),
                "baseline_q25": q25,
                "baseline_q75": q75,
                "baseline_iqr": q75 - q25,
                "baseline_q95": float(np.nanpercentile(pre, 95)),
                "baseline_median_plus_iqr": float(np.nanmedian(pre) + (q75 - q25)),
            }
        )

    baseline = pd.DataFrame(baseline_rows)
    if baseline.empty:
        raise ValueError(
            f"No animals had at least {min_baseline_days} usable baseline days for {value_col}."
        )

    out = work.merge(baseline, on="animal_id", how="inner")
    out["delta"] = out[value_col] - out["baseline_median"]
    valid = np.isfinite(out["baseline_median"]) & (out["baseline_median"] > 0)
    out["percent_change"] = np.where(valid, 100.0 * out["delta"] / out["baseline_median"], np.nan)
    out["fold_change"] = np.where(valid, out[value_col] / out["baseline_median"], np.nan)
    out["log2_fold_change"] = np.where(out["fold_change"] > 0, np.log2(out["fold_change"]), np.nan)
    return out.sort_values(["animal_id", "relative_day"]).reset_index(drop=True), baseline


def equal_bird_rolling_summary(
    normalized: pd.DataFrame,
    *,
    metric_col: str,
    x_min: int,
    x_max: int,
    window_days: int,
    rolling_stat: str,
    min_observations_per_bird_window: int,
    min_animals_for_summary: int,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """One rolling value per bird per day, then median/IQR/CI across birds."""
    rng = np.random.default_rng(seed)
    half = max(int(window_days) // 2, 0)
    birds = sorted(normalized["animal_id"].astype(str).unique())
    rows = []

    for day in range(int(x_min), int(x_max) + 1):
        animal_values = []
        for bird in birds:
            g = normalized[normalized["animal_id"].astype(str).eq(bird)]
            window = g[(g["relative_day"] >= day - half) & (g["relative_day"] <= day + half)]
            # Never allow a rolling window to mix pre- and post-lesion observations.
            if day < 0:
                window = window[window["relative_day"] < 0]
            elif day > 0:
                window = window[window["relative_day"] > 0]
            else:
                window = window[window["relative_day"] == 0]
            vals = pd.to_numeric(window[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size < min_observations_per_bird_window:
                continue
            animal_value = float(np.nanmedian(vals) if rolling_stat == "median" else np.nanmean(vals))
            animal_values.append(animal_value)

        values = np.asarray(animal_values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < min_animals_for_summary:
            continue

        if bootstrap_reps > 0:
            boot = np.empty(bootstrap_reps, dtype=float)
            for i in range(bootstrap_reps):
                sample = rng.choice(values, size=values.size, replace=True)
                boot[i] = np.nanmedian(sample) if rolling_stat == "median" else np.nanmean(sample)
            ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan

        rows.append(
            {
                "relative_day": day,
                "metric_col": metric_col,
                "window_days": window_days,
                "rolling_stat": rolling_stat,
                "n_animals": int(values.size),
                "summary": float(np.nanmedian(values) if rolling_stat == "median" else np.nanmean(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Persistence metrics
# -----------------------------------------------------------------------------
def longest_consecutive_recorded_run(days: Sequence[int]) -> int:
    ordered = sorted(set(int(day) for day in days))
    if not ordered:
        return 0
    longest = current = 1
    for previous, current_day in zip(ordered[:-1], ordered[1:]):
        if current_day == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def persistence_summary(normalized: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    rows = []
    for animal, g in normalized.groupby("animal_id", sort=True):
        post = g[g["relative_day"] > 0].sort_values("relative_day").copy()
        pre = g[g["relative_day"] < 0].copy()
        if post.empty:
            continue

        baseline_median = float(g["baseline_median"].iloc[0])
        threshold_median_iqr = float(g["baseline_median_plus_iqr"].iloc[0])
        threshold_q95 = float(g["baseline_q95"].iloc[0])

        above_median = post[value_col] > baseline_median
        above_median_iqr = post[value_col] > threshold_median_iqr
        above_q95 = post[value_col] > threshold_q95
        above_q95_days = post.loc[above_q95, "relative_day"].astype(int).tolist()

        x = post["relative_day"].to_numpy(dtype=float)
        positive_delta = np.clip(post["delta"].to_numpy(dtype=float), 0, None)
        auc = float(np.trapezoid(positive_delta, x)) if len(x) >= 2 else float(positive_delta[0])
        recording_span = max(float(x.max() - x.min()), 1.0)

        rows.append(
            {
                "animal_id": animal,
                "value_col": value_col,
                "n_pre_days": int(pre["relative_day"].nunique()),
                "n_post_days": int(post["relative_day"].nunique()),
                "first_post_day": int(post["relative_day"].min()),
                "last_recorded_post_day": int(post["relative_day"].max()),
                "baseline_median": baseline_median,
                "baseline_median_plus_iqr": threshold_median_iqr,
                "baseline_q95": threshold_q95,
                "median_post_delta": float(np.nanmedian(post["delta"])),
                "median_post_percent_change": float(np.nanmedian(post["percent_change"])),
                "median_post_fold_change": float(np.nanmedian(post["fold_change"])),
                "n_post_days_above_baseline_median": int(above_median.sum()),
                "fraction_post_days_above_baseline_median": float(above_median.mean()),
                "n_post_days_above_baseline_median_plus_iqr": int(above_median_iqr.sum()),
                "fraction_post_days_above_baseline_median_plus_iqr": float(above_median_iqr.mean()),
                "n_post_days_above_pre_q95": int(above_q95.sum()),
                "fraction_post_days_above_pre_q95": float(above_q95.mean()),
                "longest_consecutive_recorded_days_above_pre_q95": longest_consecutive_recorded_run(above_q95_days),
                "last_day_above_pre_q95": max(above_q95_days) if above_q95_days else np.nan,
                "auc_positive_delta": auc,
                "auc_positive_delta_per_recording_day": auc / recording_span,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_timecourse(
    normalized: pd.DataFrame,
    summary: pd.DataFrame,
    out_base: Path,
    *,
    metric_col: str,
    y_label: str,
    reference_value: float,
    x_min: int,
    x_max: int,
    band: str,
    robust_low: float,
    robust_high: float,
    y_scale: str,
    raw_alpha: float,
    min_animals_for_summary: int,
    dpi: int,
    show: bool,
    include_individuals: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    if include_individuals:
        for _, g in normalized.groupby("animal_id", sort=True):
            g = g.sort_values("relative_day")
            ax.plot(
                g["relative_day"],
                g[metric_col],
                color=ML_INDIVIDUAL_COLOR,
                alpha=raw_alpha,
                linewidth=1.0,
                marker="o",
                markersize=2.0,
                markeredgewidth=0,
                zorder=1,
            )

    if not summary.empty:
        s = summary.sort_values("relative_day")
        if band == "bootstrap_ci" and s[["bootstrap_ci_low", "bootstrap_ci_high"]].notna().any().all():
            lower = s["bootstrap_ci_low"].to_numpy(dtype=float)
            upper = s["bootstrap_ci_high"].to_numpy(dtype=float)
            band_label = "95% bootstrap CI"
        else:
            lower = s["q25"].to_numpy(dtype=float)
            upper = s["q75"].to_numpy(dtype=float)
            band_label = "IQR across birds"
        x = s["relative_day"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color=ML_LINE_COLOR, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, s["summary"], color=ML_LINE_COLOR, linewidth=3.2, zorder=3)
    else:
        band_label = "IQR across birds"

    ax.axhline(reference_value, color="0.35", linestyle=":", linewidth=1.1, zorder=0)
    ax.axvline(0, color=LESION_COLOR, linestyle="--", linewidth=1.3, zorder=4)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Days relative to lesion", fontsize=17)
    ax.set_ylabel(y_label, fontsize=17)
    pretty_axes(ax, tick_fontsize=12.5)

    if include_individuals or summary.empty:
        values = pd.to_numeric(normalized[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
    else:
        summary_cols = ["summary", "q25", "q75"]
        values = pd.concat([
            pd.to_numeric(summary[c], errors="coerce")
            for c in summary_cols if c in summary.columns
        ], ignore_index=True).dropna().to_numpy(dtype=float)

    if y_scale == "log":
        positive = values[values > 0]
        if positive.size:
            if include_individuals or summary.empty:
                lo = float(np.nanpercentile(positive, robust_low))
                hi = float(np.nanpercentile(positive, robust_high))
                lower_pad = 0.90
                upper_pad = 1.10
            else:
                lo = float(np.nanmin(positive))
                hi = float(np.nanmax(positive))
                lower_pad = 0.97
                upper_pad = 1.06
            lo = min(lo, reference_value) if reference_value > 0 else lo
            hi = max(hi, reference_value) if reference_value > 0 else hi
            if hi > lo > 0:
                ax.set_yscale("log")
                ax.set_ylim(lo * lower_pad, hi * upper_pad)
                apply_plain_log_tick_labels(ax, axis="y")
    elif y_scale == "symlog":
        if include_individuals or summary.empty:
            lo, hi = robust_limits(values, low_percentile=robust_low, high_percentile=robust_high, include=reference_value)
        else:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            lo = min(lo, reference_value)
            hi = max(hi, reference_value)
            span = hi - lo
            pad = max(0.04 * span, 0.5)
            lo -= pad
            hi += pad
        linthresh = max(1.0, 0.03 * max(abs(lo), abs(hi)))
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(lo, hi)
        apply_sparse_symlog_tick_labels(ax, lo, hi)
    else:
        if include_individuals or summary.empty:
            lo, hi = robust_limits(values, low_percentile=robust_low, high_percentile=robust_high, include=reference_value)
        else:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            lo = min(lo, reference_value)
            hi = max(hi, reference_value)
            span = hi - lo
            pad = max(0.04 * span, 0.02)
            lo -= pad
            hi += pad
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(FuncFormatter(plain_numeric_tick_formatter))

    # Mark individual observations outside the robust display range rather than
    # silently hiding them. This keeps extreme late effects visible without
    # allowing one or two birds to compress the rest of the panel.
    y_min, y_max = ax.get_ylim()
    if include_individuals:
        high = normalized[pd.to_numeric(normalized[metric_col], errors="coerce") > y_max]
        low = normalized[pd.to_numeric(normalized[metric_col], errors="coerce") < y_min]
        if not high.empty:
            ax.scatter(high["relative_day"], np.full(len(high), y_max), marker="^", s=24,
                       facecolor="white", edgecolor=ML_LINE_COLOR, linewidth=0.8, zorder=5, clip_on=False)
        if not low.empty:
            ax.scatter(low["relative_day"], np.full(len(low), y_min), marker="v", s=24,
                       facecolor="white", edgecolor=ML_LINE_COLOR, linewidth=0.8, zorder=5, clip_on=False)

    handles = []
    if include_individuals:
        handles.append(Line2D([0], [0], color=ML_INDIVIDUAL_COLOR, alpha=min(0.65, raw_alpha * 3), linewidth=1.0, marker="o", markersize=3, label="individual M+L birds"))
    handles.extend([
        Line2D([0], [0], color=ML_LINE_COLOR, linewidth=3.2, label=f"equal-bird rolling median (n≥{min_animals_for_summary})"),
        Patch(facecolor=ML_LINE_COLOR, edgecolor="none", alpha=0.18, label=band_label),
        Line2D([0], [0], color=LESION_COLOR, linestyle="--", linewidth=1.3, label="lesion day"),
    ])
    ax.legend(handles=handles, frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(left=0.25, bottom=0.16)
    save_figure(fig, out_base, dpi=dpi, show=show)


def plot_heatmap(

    normalized: pd.DataFrame,
    out_base: Path,
    *,
    metric_col: str,
    colorbar_label: str,
    x_min: int,
    x_max: int,
    sort_mode: str,
    center: float,
    dpi: int,
    show: bool,
) -> None:
    work = normalized[(normalized["relative_day"] >= x_min) & (normalized["relative_day"] <= x_max)].copy()
    post = work[work["relative_day"] > 0]
    if sort_mode == "max_post":
        order = post.groupby("animal_id")[metric_col].max().sort_values(ascending=False).index.tolist()
    elif sort_mode == "recording_length":
        order = post.groupby("animal_id")["relative_day"].max().sort_values(ascending=False).index.tolist()
    elif sort_mode == "bird":
        order = sorted(work["animal_id"].astype(str).unique())
    else:
        order = post.groupby("animal_id")[metric_col].median().sort_values(ascending=False).index.tolist()

    days = list(range(x_min, x_max + 1))
    pivot = work.pivot_table(index="animal_id", columns="relative_day", values=metric_col, aggfunc="median")
    pivot = pivot.reindex(index=order, columns=days)
    data = pivot.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return

    low, high = np.nanpercentile(finite, [2.5, 97.5])
    span = max(abs(low - center), abs(high - center), 1e-12)
    norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)

    fig_height = max(3.2, 0.42 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    image = ax.imshow(masked, aspect="auto", interpolation="none", cmap=cmap, norm=norm)

    lesion_col = 0 - x_min
    ax.axvline(lesion_col - 0.5, color=LESION_COLOR, linestyle="--", linewidth=1.2)
    xtick_days = [d for d in range(x_min, x_max + 1, 5)]
    ax.set_xticks([d - x_min for d in xtick_days])
    ax.set_xticklabels(xtick_days)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Days relative to lesion", fontsize=15)
    ax.set_ylabel("M+L lesion bird", fontsize=15)
    ax.tick_params(axis="both", labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=9.5)
    fig.tight_layout()
    save_figure(fig, out_base, dpi=dpi, show=show)


def plot_small_multiples(
    normalized: pd.DataFrame,
    out_base: Path,
    *,
    metric_col: str,
    y_label: str,
    reference_value: float,
    x_min: int,
    x_max: int,
    dpi: int,
    show: bool,
) -> None:
    birds = sorted(normalized["animal_id"].astype(str).unique())
    if not birds:
        return
    n_cols = 3
    n_rows = math.ceil(len(birds) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 3.0 * n_rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()

    values = normalized[metric_col].dropna().to_numpy(dtype=float)
    y_lo, y_hi = robust_limits(values, low_percentile=1.0, high_percentile=99.0, include=reference_value)

    for ax, bird in zip(axes_array, birds):
        g = normalized[normalized["animal_id"].astype(str).eq(bird)].sort_values("relative_day")
        ax.plot(g["relative_day"], g[metric_col], color=ML_LINE_COLOR, linewidth=1.5, marker="o", markersize=2.2)
        ax.axhline(reference_value, color="0.45", linestyle=":", linewidth=0.9)
        ax.axvline(0, color=LESION_COLOR, linestyle="--", linewidth=0.9)
        ax.set_title(bird, fontsize=11.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        pretty_axes(ax, tick_fontsize=9.5)

    for ax in axes_array[len(birds):]:
        ax.axis("off")

    fig.supxlabel("Days relative to lesion", fontsize=15)
    fig.supylabel(y_label, fontsize=15, x=0.02)
    fig.tight_layout(rect=[0.035, 0.03, 1, 1])
    save_figure(fig, out_base, dpi=dpi, show=show)


def plot_persistence_fraction(
    persistence: pd.DataFrame,
    out_base: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    if persistence.empty:
        return
    plot_df = persistence.sort_values("fraction_post_days_above_pre_q95", ascending=True)
    fig_height = max(3.4, 0.42 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    y = np.arange(len(plot_df))
    ax.barh(y, 100.0 * plot_df["fraction_post_days_above_pre_q95"], color=ML_LINE_COLOR, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["animal_id"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Post-lesion recording days above pre-lesion 95th percentile (%)", fontsize=13)
    ax.set_ylabel("M+L lesion bird", fontsize=13)
    pretty_axes(ax, tick_fontsize=10.5)
    for index, (_, row) in enumerate(plot_df.iterrows()):
        percent = 100.0 * row["fraction_post_days_above_pre_q95"]
        if percent >= 82:
            text_x, ha, text_color = percent - 1.5, "right", "white"
        else:
            text_x, ha, text_color = percent + 1.5, "left", "black"
        ax.text(
            text_x,
            index,
            f"{int(row['n_post_days_above_pre_q95'])}/{int(row['n_post_days'])}",
            va="center",
            ha=ha,
            color=text_color,
            fontsize=9,
        )
    fig.tight_layout()
    save_figure(fig, out_base, dpi=dpi, show=show)


def plot_persistence_timeline(
    normalized: pd.DataFrame,
    out_base: Path,
    *,
    value_col: str,
    x_max: int,
    dpi: int,
    show: bool,
) -> None:
    post = normalized[(normalized["relative_day"] > 0) & (normalized["relative_day"] <= x_max)].copy()
    if post.empty:
        return
    post["above_q95"] = post[value_col] > post["baseline_q95"]
    order = (
        post.groupby("animal_id")["above_q95"].mean().sort_values(ascending=False).index.tolist()
    )
    y_map = {bird: i for i, bird in enumerate(order)}

    fig_height = max(3.4, 0.42 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    for _, row in post.iterrows():
        y = y_map[str(row["animal_id"])]
        if bool(row["above_q95"]):
            ax.scatter(row["relative_day"], y, s=30, color=ML_LINE_COLOR, edgecolor="none", zorder=3)
        else:
            ax.scatter(row["relative_day"], y, s=18, facecolor="white", edgecolor="0.70", linewidth=0.8, zorder=2)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Post-lesion day", fontsize=13)
    ax.set_ylabel("M+L lesion bird", fontsize=13)
    pretty_axes(ax, tick_fontsize=10.5)
    ax.grid(axis="x", alpha=0.15, linewidth=0.7)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=ML_LINE_COLOR, markeredgecolor="none", markersize=6, label="above pre-lesion 95th percentile"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="0.70", markersize=5, label="at or below threshold"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_base, dpi=dpi, show=show)


# -----------------------------------------------------------------------------
# Metric workflow
# -----------------------------------------------------------------------------
def run_metric_workflow(
    daily: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
    raw_label: str,
    out_dir: Path,
    baseline_start_day: int,
    baseline_end_day: int,
    min_baseline_days: int,
    x_min: int,
    x_max: int,
    smooth_window_days: int,
    rolling_stat: str,
    min_observations_per_bird_window: int,
    min_animals_for_summary: int,
    bootstrap_reps: int,
    seed: int,
    band: str,
    heatmap_sort: str,
    robust_low: float,
    robust_high: float,
    raw_alpha: float,
    dpi: int,
    show: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    ml = daily[daily["display_group"].isin(ML_GROUPS)].copy()
    ml["display_group"] = POOLED_ML_GROUP
    if ml.empty:
        raise ValueError(f"No pooled medial+lateral lesion rows were found for {value_col}.")

    normalized, baseline = add_baseline_normalization(
        ml,
        value_col=value_col,
        baseline_start_day=baseline_start_day,
        baseline_end_day=baseline_end_day,
        min_baseline_days=min_baseline_days,
    )
    norm_path = out_dir / f"panelE_ML_daily_{prefix}_normalized.csv"
    baseline_path = out_dir / f"panelE_ML_{prefix}_baseline_summary.csv"
    outputs[f"{prefix}_normalized_table"] = write_csv(normalized, norm_path)
    outputs[f"{prefix}_baseline_table"] = write_csv(baseline, baseline_path)

    persistence = persistence_summary(normalized, value_col=value_col)
    persistence_path = out_dir / f"panelE_ML_{prefix}_persistence_summary.csv"
    outputs[f"{prefix}_persistence_table"] = write_csv(persistence, persistence_path)

    if prefix == "SD":
        delta_label = "Δ phrase duration SD (s)\nfrom late-pre"
        pct_label = "Phrase duration SD\nchange from late-pre (%)"
        fold_label = "Phrase duration SD (s)\n/ late-pre baseline"
    else:
        delta_label = "Δ phrase duration CV\nfrom late-pre"
        pct_label = "Phrase duration CV\nchange from late-pre (%)"
        fold_label = "Phrase duration CV\n/ late-pre baseline"

    plot_specs = [
        ("delta", delta_label, 0.0, "linear"),
        ("percent_change", pct_label, 0.0, "symlog"),
        ("fold_change", fold_label, 1.0, "log"),
    ]

    for metric_col, y_label, reference, y_scale in plot_specs:
        summary = equal_bird_rolling_summary(
            normalized,
            metric_col=metric_col,
            x_min=x_min,
            x_max=x_max,
            window_days=smooth_window_days,
            rolling_stat=rolling_stat,
            min_observations_per_bird_window=min_observations_per_bird_window,
            min_animals_for_summary=min_animals_for_summary,
            bootstrap_reps=bootstrap_reps,
            seed=seed + {"delta": 0, "percent_change": 1000, "fold_change": 2000}[metric_col],
        )
        summary_path = out_dir / f"panelE_ML_{prefix}_{metric_col}_equal_weight_rolling_summary.csv"
        outputs[f"{prefix}_{metric_col}_summary"] = write_csv(summary, summary_path)

        base = out_dir / f"panelE_ML_{prefix}_{metric_col}_timecourse"
        plot_timecourse(
            normalized,
            summary,
            base,
            metric_col=metric_col,
            y_label=y_label,
            reference_value=reference,
            x_min=x_min,
            x_max=x_max,
            band=band,
            robust_low=robust_low,
            robust_high=robust_high,
            y_scale=y_scale,
            raw_alpha=raw_alpha,
            min_animals_for_summary=min_animals_for_summary,
            dpi=dpi,
            show=show,
            include_individuals=True,
        )
        outputs[f"{prefix}_{metric_col}_timecourse_png"] = base.with_suffix(".png")
        outputs[f"{prefix}_{metric_col}_timecourse_pdf"] = base.with_suffix(".pdf")

        summary_only_base = out_dir / f"panelE_ML_{prefix}_{metric_col}_timecourse_summary_only"
        plot_timecourse(
            normalized,
            summary,
            summary_only_base,
            metric_col=metric_col,
            y_label=y_label,
            reference_value=reference,
            x_min=x_min,
            x_max=x_max,
            band=band,
            robust_low=robust_low,
            robust_high=robust_high,
            y_scale=y_scale,
            raw_alpha=raw_alpha,
            min_animals_for_summary=min_animals_for_summary,
            dpi=dpi,
            show=show,
            include_individuals=False,
        )
        outputs[f"{prefix}_{metric_col}_timecourse_summary_only_png"] = summary_only_base.with_suffix(".png")
        outputs[f"{prefix}_{metric_col}_timecourse_summary_only_pdf"] = summary_only_base.with_suffix(".pdf")

    heatmap_base = out_dir / f"panelE_ML_{prefix}_delta_heatmap"
    plot_heatmap(
        normalized,
        heatmap_base,
        metric_col="delta",
        colorbar_label=f"Δ {raw_label}",
        x_min=x_min,
        x_max=x_max,
        sort_mode=heatmap_sort,
        center=0.0,
        dpi=dpi,
        show=show,
    )
    outputs[f"{prefix}_heatmap_png"] = heatmap_base.with_suffix(".png")
    outputs[f"{prefix}_heatmap_pdf"] = heatmap_base.with_suffix(".pdf")

    # Small multiples are most useful for the primary SD panel; they are also
    # generated for CV so the same heterogeneity check is available.
    small_base = out_dir / f"panelE_ML_{prefix}_delta_small_multiples"
    plot_small_multiples(
        normalized,
        small_base,
        metric_col="delta",
        y_label=f"Δ {raw_label}",
        reference_value=0.0,
        x_min=x_min,
        x_max=x_max,
        dpi=dpi,
        show=show,
    )
    outputs[f"{prefix}_small_multiples_png"] = small_base.with_suffix(".png")
    outputs[f"{prefix}_small_multiples_pdf"] = small_base.with_suffix(".pdf")

    fraction_base = out_dir / f"panelE_ML_{prefix}_persistence_fraction_above_pre95"
    plot_persistence_fraction(persistence, fraction_base, dpi=dpi, show=show)
    outputs[f"{prefix}_persistence_fraction_png"] = fraction_base.with_suffix(".png")
    outputs[f"{prefix}_persistence_fraction_pdf"] = fraction_base.with_suffix(".pdf")

    timeline_base = out_dir / f"panelE_ML_{prefix}_persistence_timeline"
    plot_persistence_timeline(
        normalized,
        timeline_base,
        value_col=value_col,
        x_max=x_max,
        dpi=dpi,
        show=show,
    )
    outputs[f"{prefix}_persistence_timeline_png"] = timeline_base.with_suffix(".png")
    outputs[f"{prefix}_persistence_timeline_pdf"] = timeline_base.with_suffix(".pdf")

    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create M+L-only baseline-normalized SD and CV timecourses, heatmaps, "
            "small multiples, and persistence summaries for Figure 3 Panel E."
        )
    )
    parser.add_argument("--daily-variance-csv", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--selected-pairs-csv", type=Path)
    selection.add_argument("--scatter-csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument("--metadata-excel", type=Path, default=None)
    parser.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    parser.add_argument("--metadata-animal-col", default="Animal ID")
    parser.add_argument("--metadata-hit-type-col", default="Lesion hit type")
    parser.add_argument("--metadata-treatment-date-col", default="Treatment date")

    parser.add_argument("--scatter-top-percentile", type=float, default=70.0)
    parser.add_argument("--rank-on", choices=["pre", "post", "max", "pooled"], default="pooled")
    parser.add_argument("--pre-group", default="Late Pre")
    parser.add_argument("--post-group", default="Post")
    parser.add_argument("--min-n-phrases", type=int, default=5)

    parser.add_argument(
        "--daily-mean-col",
        default=None,
        help="Optional mean-duration column in the daily variance CSV; enables CV without a long CSV.",
    )
    parser.add_argument("--daily-mean-unit", choices=["ms", "s"], default="ms")

    parser.add_argument(
        "--duration-long-csv",
        type=Path,
        default=None,
        help="Optional per-rendition phrase-duration CSV used to calculate daily CV.",
    )
    parser.add_argument("--long-bird-col", default=None)
    parser.add_argument("--long-syllable-col", default=None)
    parser.add_argument("--long-day-col", default=None)
    parser.add_argument("--long-date-col", default=None)
    parser.add_argument("--long-duration-col", default=None)
    parser.add_argument("--long-group-col", default=None)
    parser.add_argument("--long-duration-unit", choices=["ms", "s"], default="ms")
    parser.add_argument("--daily-cv-min-renditions", type=int, default=10)

    parser.add_argument("--baseline-start-day", type=int, default=-14)
    parser.add_argument("--baseline-end-day", type=int, default=-1)
    parser.add_argument("--min-baseline-days", type=int, default=3)
    parser.add_argument("--x-min", type=int, default=-30)
    parser.add_argument("--x-max", type=int, default=30)
    parser.add_argument("--smooth-window-days", type=int, default=7)
    parser.add_argument("--rolling-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--min-observations-per-bird-window", type=int, default=1)
    parser.add_argument(
        "--min-animals-for-summary",
        type=int,
        default=3,
        help=(
            "Minimum number of birds required to draw the group rolling summary at a day. "
            "Individual bird trajectories continue beyond this point."
        ),
    )
    parser.add_argument("--band", choices=["iqr", "bootstrap_ci"], default="iqr")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--heatmap-sort", choices=["median_post", "max_post", "recording_length", "bird"], default="median_post")
    parser.add_argument("--robust-y-low-percentile", type=float, default=1.0)
    parser.add_argument("--robust-y-high-percentile", type=float, default=97.5)
    parser.add_argument("--individual-line-alpha", type=float, default=0.28)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    show = bool(args.show and not args.no_show)
    out_dir = ensure_dir(args.out_dir)

    hit_map, treatment_date_map = load_metadata_maps(
        args.metadata_excel,
        preferred_sheet=args.metadata_sheet,
        animal_col=args.metadata_animal_col,
        hit_type_col=args.metadata_hit_type_col,
        treatment_date_col=args.metadata_treatment_date_col,
    )

    if args.selected_pairs_csv:
        selected_pairs = load_selected_pairs(args.selected_pairs_csv)
    else:
        selected_pairs = select_pairs_from_scatter(
            args.scatter_csv,
            top_percentile=args.scatter_top_percentile,
            rank_on=args.rank_on,
            pre_group=args.pre_group,
            post_group=args.post_group,
            min_n_phrases=args.min_n_phrases,
        )
    selected_path = out_dir / "panelE_selected_animal_syllable_pairs.csv"
    write_csv(selected_pairs, selected_path)

    sd_daily, cv_from_daily = prepare_daily_variance(
        args.daily_variance_csv,
        selected_pairs=selected_pairs,
        hit_map=hit_map,
        x_min=args.x_min,
        x_max=args.x_max,
        mean_col_requested=args.daily_mean_col,
        mean_unit=args.daily_mean_unit,
    )
    sd_source_path = out_dir / "panelE_daily_SD_source_table.csv"
    write_csv(sd_daily, sd_source_path)

    outputs: dict[str, Path] = {"selected_pairs": selected_path, "SD_source_table": sd_source_path}
    outputs.update(
        run_metric_workflow(
            sd_daily,
            value_col="sd_s",
            prefix="SD",
            raw_label="phrase duration SD (s)",
            out_dir=out_dir,
            baseline_start_day=args.baseline_start_day,
            baseline_end_day=args.baseline_end_day,
            min_baseline_days=args.min_baseline_days,
            x_min=args.x_min,
            x_max=args.x_max,
            smooth_window_days=args.smooth_window_days,
            rolling_stat=args.rolling_stat,
            min_observations_per_bird_window=args.min_observations_per_bird_window,
            min_animals_for_summary=args.min_animals_for_summary,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
            band=args.band,
            heatmap_sort=args.heatmap_sort,
            robust_low=args.robust_y_low_percentile,
            robust_high=args.robust_y_high_percentile,
            raw_alpha=args.individual_line_alpha,
            dpi=args.dpi,
            show=show,
        )
    )

    cv_daily = None
    if args.duration_long_csv:
        cv_daily = prepare_daily_cv_from_long(
            args.duration_long_csv,
            selected_pairs=selected_pairs,
            hit_map=hit_map,
            treatment_date_map=treatment_date_map,
            bird_col_requested=args.long_bird_col,
            syllable_col_requested=args.long_syllable_col,
            day_col_requested=args.long_day_col,
            date_col_requested=args.long_date_col,
            duration_col_requested=args.long_duration_col,
            group_col_requested=args.long_group_col,
            duration_unit=args.long_duration_unit,
            min_renditions=args.daily_cv_min_renditions,
            x_min=args.x_min,
            x_max=args.x_max,
        )
    elif cv_from_daily is not None:
        cv_daily = cv_from_daily

    if cv_daily is not None and not cv_daily.empty:
        cv_source_path = out_dir / "panelE_daily_CV_source_table.csv"
        write_csv(cv_daily, cv_source_path)
        outputs["CV_source_table"] = cv_source_path
        outputs.update(
            run_metric_workflow(
                cv_daily,
                value_col="cv",
                prefix="CV",
                raw_label="phrase duration CV",
                out_dir=out_dir,
                baseline_start_day=args.baseline_start_day,
                baseline_end_day=args.baseline_end_day,
                min_baseline_days=args.min_baseline_days,
                x_min=args.x_min,
                x_max=args.x_max,
                smooth_window_days=args.smooth_window_days,
                rolling_stat=args.rolling_stat,
                min_observations_per_bird_window=args.min_observations_per_bird_window,
                min_animals_for_summary=args.min_animals_for_summary,
                bootstrap_reps=args.bootstrap_reps,
                seed=args.seed + 10000,
                band=args.band,
                heatmap_sort=args.heatmap_sort,
                robust_low=args.robust_y_low_percentile,
                robust_high=args.robust_y_high_percentile,
                raw_alpha=args.individual_line_alpha,
                dpi=args.dpi,
                show=show,
            )
        )
    else:
        print(
            "[INFO] CV figures were skipped. Supply --duration-long-csv, or use a daily "
            "variance CSV containing a mean-duration column (optionally identify it with "
            "--daily-mean-col)."
        )

    manifest = pd.DataFrame(
        [{"output": key, "path": str(value)} for key, value in outputs.items()]
    )
    manifest_path = out_dir / "panelE_output_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("[OK] Wrote:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
