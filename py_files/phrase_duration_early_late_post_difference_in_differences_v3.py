#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phrase_duration_early_late_post_difference_in_differences_v3.py

Test whether the lesion-associated change from Late Pre to Post is greater than
the baseline change from Early Pre to Late Pre.

Primary within-bird contrast
----------------------------
For each metric:

    baseline_change = Late Pre - Early Pre
    lesion_change   = Post - Late Pre
    difference_in_differences = lesion_change - baseline_change

Positive difference-in-differences values indicate that the post-lesion change
was greater than the change already occurring during the pre-lesion baseline.

Metrics
-------
1. Phrase-duration SD, raw seconds
2. Phrase-duration SD, log2 proportional change
3. Phrase-duration CV, raw CV units
4. Phrase-duration CV, log2 proportional change

The script:
- uses the same top-30-percent pooled Late Pre/Post variance selection logic as
  the primary Figure 3 analysis, unless --selected-pairs-csv is supplied;
- requires Early Pre, Late Pre, and Post values for each retained
  animal-by-syllable pair;
- averages selected syllables within each bird by default, matching the current
  Figure 3 robustness workflow;
- pools partial and complete medial+lateral lesions;
- uses bird as the independent/resampling unit;
- calculates animal-level percentile bootstrap 95% confidence intervals;
- calculates one-sided sign-flip tests for DID > 0 within each group;
- compares pooled M+L versus lateral-only and pooled M+L versus sham using
  bootstrap confidence intervals and one-sided label-shuffle tests;
- applies Benjamini-Hochberg FDR correction within each metric family;
- saves PNG plots only;
- safely reports and skips lesion groups with no complete three-epoch data.

Expected input
--------------
The default column names match:

    /Volumes/my_own_SSD/updated_AreaX_outputs/
    usage_balanced_phrase_duration_stats.csv

Expected columns include variants of:
- Animal ID
- Syllable
- Group
- N_phrases
- Mean_ms
- Std_ms
- Variance_ms2

Recommended run
---------------
python phrase_duration_early_late_post_difference_in_differences.py \
  --stats-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_early_late_post_DID" \
  --top-fraction 0.30 \
  --within-animal-stat mean \
  --n-bootstrap 5000 \
  --n-permutations 10000 \
  --seed 123
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical lesion groups
# ---------------------------------------------------------------------------

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

PLOT_GROUPS = [SHAM, LATERAL, POOLED_ML]

GROUP_LABELS = {
    SHAM: "Sham",
    LATERAL: "Lateral only",
    POOLED_ML: "Pooled M+L",
    PARTIAL_ML: "Partial M+L",
    COMPLETE_ML: "Complete M+L",
}

GROUP_COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    POOLED_ML: "#6F4AB6",
    PARTIAL_ML: "#7A4FB7",
    COMPLETE_ML: "#3F007D",
}

METRIC_SPECS = {
    "sd_raw": {
        "baseline_col": "baseline_change_sd_s",
        "lesion_col": "lesion_change_sd_s",
        "did_col": "did_sd_s",
        "interval_ylabel": "Change in phrase-duration SD (s)",
        "did_ylabel": "Difference-in-differences in SD (s)",
        "title": "Phrase-duration SD",
        "stem": "SD_raw",
    },
    "sd_log2": {
        "baseline_col": "baseline_log2_change_sd",
        "lesion_col": "lesion_log2_change_sd",
        "did_col": "did_log2_sd",
        "interval_ylabel": "Log2 proportional change in phrase-duration SD",
        "did_ylabel": "Difference-in-differences in log2 SD change",
        "title": "Phrase-duration SD proportional change",
        "stem": "SD_log2",
    },
    "cv_raw": {
        "baseline_col": "baseline_change_cv",
        "lesion_col": "lesion_change_cv",
        "did_col": "did_cv",
        "interval_ylabel": "Change in phrase-duration CV",
        "did_ylabel": "Difference-in-differences in CV",
        "title": "Phrase-duration coefficient of variation",
        "stem": "CV_raw",
    },
    "cv_log2": {
        "baseline_col": "baseline_log2_change_cv",
        "lesion_col": "lesion_log2_change_cv",
        "did_col": "did_log2_cv",
        "interval_ylabel": "Log2 proportional change in phrase-duration CV",
        "did_ylabel": "Difference-in-differences in log2 CV change",
        "title": "Phrase-duration CV proportional change",
        "stem": "CV_log2",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether Post minus Late Pre exceeds Late Pre minus Early Pre "
            "for phrase-duration SD and CV."
        )
    )
    parser.add_argument("--stats-csv", required=True, type=Path)
    parser.add_argument("--metadata-excel", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument(
        "--selected-pairs-csv",
        type=Path,
        default=None,
        help=(
            "Optional exact animal-by-syllable selection from the primary Figure 3 "
            "analysis. When omitted, the script independently reproduces pooled "
            "Late Pre/Post top-fraction selection."
        ),
    )

    parser.add_argument("--metadata-sheet", default=None)
    parser.add_argument("--metadata-animal-col", default=None)
    parser.add_argument("--metadata-hit-type-col", default=None)

    parser.add_argument("--early-group", default="Early Pre")
    parser.add_argument("--late-group", default="Late Pre")
    parser.add_argument("--post-group", default="Post")

    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.30,
        help="Fraction of syllables selected per bird when no selected-pairs CSV is supplied.",
    )
    parser.add_argument(
        "--min-n-phrases",
        type=int,
        default=5,
        help="Minimum number of phrases required in each epoch for a retained syllable.",
    )
    parser.add_argument(
        "--within-animal-stat",
        choices=["mean", "median"],
        default="mean",
        help="How selected syllables are summarized within each bird.",
    )
    parser.add_argument(
        "--group-stat",
        choices=["median", "mean"],
        default="median",
        help="Statistic used for group bootstrap CIs and permutation tests.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


def norm_text(value: Any) -> str:
    return clean_text(value).lower().replace("_", " ").replace("-", " ")


def find_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested {label} column {requested!r} is not present.\n"
                f"Available columns: {list(df.columns)}"
            )
        return requested

    exact = {str(column): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    normalized = {norm_text(column): str(column) for column in df.columns}
    for candidate in candidates:
        key = norm_text(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not identify the {label} column.\n"
            f"Tried: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def canonical_group(value: Any) -> str:
    text = norm_text(value)

    if not text:
        return "unknown"

    if "sham" in text or "saline" in text:
        return SHAM

    if "complete and partial" in text or "partial and complete" in text:
        return POOLED_ML

    if "partial" in text and "medial" in text and "lateral" in text:
        return PARTIAL_ML

    if "complete" in text and "medial" in text and "lateral" in text:
        return COMPLETE_ML

    if "large lesion" in text and "area x not visible" in text:
        return COMPLETE_ML

    if "medial" in text and "lateral" in text:
        return POOLED_ML

    # In the lesion metadata, lateral-only birds may be described by the
    # histological shorthand "Area X visible (single hit)". These are the
    # lateral-only lesion cases: Area X remains visible because the medial
    # pathway was not included in the lesion.
    if (
        "area x visible" in text
        and (
            "single hit" in text
            or "lateral" in text
            or "medial" not in text
        )
    ):
        return LATERAL

    if "single hit" in text and "area x" in text:
        return LATERAL

    if "lateral" in text and "medial" not in text:
        return LATERAL

    if text in {"l", "lateral only", "area x visible"}:
        return LATERAL

    return clean_text(value)


def pooled_group(value: Any) -> str:
    group = canonical_group(value)
    if group in {PARTIAL_ML, COMPLETE_ML, POOLED_ML}:
        return POOLED_ML
    return group


def canonical_epoch(value: Any, early: str, late: str, post: str) -> str:
    text = norm_text(value)
    mappings = {
        norm_text(early): "Early Pre",
        norm_text(late): "Late Pre",
        norm_text(post): "Post",
        "early pre": "Early Pre",
        "early pre lesion": "Early Pre",
        "late pre": "Late Pre",
        "late pre lesion": "Late Pre",
        "post": "Post",
        "post lesion": "Post",
    }
    return mappings.get(text, clean_text(value))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize(values: np.ndarray, stat: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.nanmean(values))
    return float(np.nanmedian(values))


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Metadata and input preparation
# ---------------------------------------------------------------------------

def load_metadata(
    path: Path,
    *,
    sheet_requested: str | None,
    animal_col_requested: str | None,
    hit_col_requested: str | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata workbook not found:\n{path}")

    workbook = pd.ExcelFile(path)

    if sheet_requested:
        if sheet_requested not in workbook.sheet_names:
            raise ValueError(
                f"Metadata sheet {sheet_requested!r} was not found.\n"
                f"Available sheets: {workbook.sheet_names}"
            )
        sheets = [sheet_requested]
    else:
        preferred = [
            "animal_hit_type_summary",
            "metadata_with_hit_type",
            "metadata",
        ]
        sheets = [sheet for sheet in preferred if sheet in workbook.sheet_names]
        sheets += [sheet for sheet in workbook.sheet_names if sheet not in sheets]

    errors = []
    for sheet in sheets:
        try:
            raw = pd.read_excel(path, sheet_name=sheet)
            animal_col = find_column(
                raw,
                animal_col_requested,
                ["Animal ID", "animal_id", "bird", "Bird ID", "Animal"],
                "metadata animal",
            )
            hit_col = find_column(
                raw,
                hit_col_requested,
                [
                    "Lesion hit type",
                    "Lesion Hit Type",
                    "hit_type",
                    "display_group",
                    "Treatment Type",
                    "Treatment type",
                    "Group",
                ],
                "metadata lesion hit type",
            )

            out = raw[[animal_col, hit_col]].copy()
            out.columns = ["animal_id", "raw_hit_type"]
            out["animal_id"] = out["animal_id"].astype(str).str.strip()
            out["display_group_original"] = out["raw_hit_type"].apply(canonical_group)
            out["display_group"] = out["raw_hit_type"].apply(pooled_group)
            out = out[out["animal_id"].ne("")].drop_duplicates("animal_id")

            observed = set(out["display_group"])
            if not observed.intersection(set(PLOT_GROUPS)):
                raise ValueError(
                    f"Sheet {sheet!r} did not yield recognized lesion groups: {sorted(observed)}"
                )

            print(
                f"[INFO] Metadata group source: sheet={sheet!r}, "
                f"column={hit_col!r}"
            )
            print("[INFO] Normalized metadata group counts:")
            normalized_counts = (
                out["display_group"]
                .value_counts(dropna=False)
                .reindex(PLOT_GROUPS)
                .dropna()
                .astype(int)
            )
            for normalized_group, normalized_count in normalized_counts.items():
                print(
                    f"  {GROUP_LABELS[normalized_group]}: "
                    f"{normalized_count}"
                )
            return out
        except Exception as exc:
            errors.append(f"{sheet}: {exc}")

    raise ValueError(
        "Could not identify a usable metadata sheet.\n"
        + "\n".join(errors)
    )


def load_stats_long(
    path: Path,
    *,
    metadata: pd.DataFrame,
    early_group: str,
    late_group: str,
    post_group: str,
    min_n_phrases: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stats CSV not found:\n{path}")

    raw = pd.read_csv(path)

    animal_col = find_column(
        raw,
        None,
        ["Animal ID", "animal_id", "bird", "Bird ID"],
        "animal",
    )
    syllable_col = find_column(
        raw,
        None,
        ["Syllable", "syllable", "label", "syllable_label"],
        "syllable",
    )
    group_col = find_column(
        raw,
        None,
        ["Group", "group", "epoch", "analysis_epoch"],
        "epoch/group",
    )
    n_col = find_column(
        raw,
        None,
        ["N_phrases", "n_phrases", "N", "count"],
        "phrase count",
    )
    mean_col = find_column(
        raw,
        None,
        ["Mean_ms", "mean_ms", "Mean (ms)", "mean_duration_ms"],
        "mean duration",
    )
    sd_col = find_column(
        raw,
        None,
        ["Std_ms", "SD_ms", "std_ms", "Std (ms)", "sd_ms"],
        "standard deviation",
    )
    variance_col = find_column(
        raw,
        None,
        ["Variance_ms2", "Variance (ms^2)", "variance_ms2", "variance"],
        "variance",
    )

    work = raw[
        [animal_col, syllable_col, group_col, n_col, mean_col, sd_col, variance_col]
    ].copy()
    work.columns = [
        "animal_id",
        "syllable",
        "epoch_raw",
        "n_phrases",
        "mean_ms",
        "sd_ms",
        "variance_ms2",
    ]

    work["animal_id"] = work["animal_id"].astype(str).str.strip()
    work["syllable"] = work["syllable"].astype(str).str.strip()
    work["epoch"] = work["epoch_raw"].apply(
        lambda value: canonical_epoch(
            value,
            early=early_group,
            late=late_group,
            post=post_group,
        )
    )

    for column in ["n_phrases", "mean_ms", "sd_ms", "variance_ms2"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work[work["epoch"].isin(["Early Pre", "Late Pre", "Post"])].copy()
    work = work[work["n_phrases"].fillna(0) >= min_n_phrases].copy()
    work = work[
        np.isfinite(work["mean_ms"])
        & np.isfinite(work["sd_ms"])
        & np.isfinite(work["variance_ms2"])
        & (work["mean_ms"] > 0)
        & (work["sd_ms"] > 0)
        & (work["variance_ms2"] >= 0)
    ].copy()

    # Average duplicate rows within the same animal/syllable/epoch if present.
    work = (
        work.groupby(["animal_id", "syllable", "epoch"], as_index=False)
        .agg(
            n_phrases=("n_phrases", "sum"),
            mean_ms=("mean_ms", "mean"),
            sd_ms=("sd_ms", "mean"),
            variance_ms2=("variance_ms2", "mean"),
        )
    )

    work["sd_s"] = work["sd_ms"] / 1000.0
    work["cv"] = work["sd_ms"] / work["mean_ms"]

    work = work.merge(metadata, on="animal_id", how="left")
    missing_group = work["display_group"].isna()
    if missing_group.any():
        missing_animals = sorted(work.loc[missing_group, "animal_id"].unique())
        raise ValueError(
            "These animals were not found in the metadata workbook:\n"
            + "\n".join(f"  - {animal}" for animal in missing_animals)
        )

    return work


def pivot_epoch_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["n_phrases", "mean_ms", "sd_ms", "variance_ms2", "sd_s", "cv"]

    pieces = []
    for value_col in value_columns:
        pivot = long_df.pivot_table(
            index=[
                "animal_id",
                "syllable",
                "display_group",
                "display_group_original",
                "raw_hit_type",
            ],
            columns="epoch",
            values=value_col,
            aggfunc="mean",
        )
        pivot = pivot.rename(
            columns={
                "Early Pre": f"early_pre_{value_col}",
                "Late Pre": f"late_pre_{value_col}",
                "Post": f"post_{value_col}",
            }
        )
        pieces.append(pivot)

    wide = pd.concat(pieces, axis=1).reset_index()

    required = []
    for metric in value_columns:
        required.extend(
            [
                f"early_pre_{metric}",
                f"late_pre_{metric}",
                f"post_{metric}",
            ]
        )
    wide = wide.dropna(subset=required).copy()

    # Selection metric exactly follows pooled Late Pre/Post variance:
    # arithmetic mean of Late Pre and Post variance.
    wide["pooled_late_post_variance_ms2"] = wide[
        ["late_pre_variance_ms2", "post_variance_ms2"]
    ].mean(axis=1)

    return wide


def load_selected_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selected-pairs CSV not found:\n{path}")

    raw = pd.read_csv(path)
    animal_col = find_column(
        raw,
        None,
        ["animal_id", "Animal ID", "bird", "Bird ID"],
        "selected-pairs animal",
    )
    syllable_col = find_column(
        raw,
        None,
        ["syllable", "Syllable", "label", "syllable_label"],
        "selected-pairs syllable",
    )
    out = raw[[animal_col, syllable_col]].copy()
    out.columns = ["animal_id", "syllable"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["syllable"] = out["syllable"].astype(str).str.strip()
    return out.drop_duplicates().reset_index(drop=True)


def select_pairs(
    wide: pd.DataFrame,
    *,
    selected_pairs_csv: Path | None,
    top_fraction: float,
) -> tuple[pd.DataFrame, str]:
    if selected_pairs_csv is not None:
        pairs = load_selected_pairs(selected_pairs_csv)
        selected = wide.merge(
            pairs.assign(_selected_pair=True),
            on=["animal_id", "syllable"],
            how="inner",
        )
        source = f"Exact selected pairs from {selected_pairs_csv.name}"
        return selected, source

    if not (0 < top_fraction <= 1):
        raise ValueError("--top-fraction must be greater than 0 and at most 1.")

    parts = []
    for animal_id, group in wide.groupby("animal_id", dropna=False):
        values = group["pooled_late_post_variance_ms2"].dropna().to_numpy(float)
        if values.size == 0:
            continue

        threshold = np.nanpercentile(values, 100.0 * (1.0 - top_fraction))
        keep = group[
            group["pooled_late_post_variance_ms2"] >= threshold
        ].copy()
        keep["selection_threshold_variance_ms2"] = threshold
        keep["top_fraction"] = top_fraction
        parts.append(keep)

    if not parts:
        raise ValueError("No syllables were selected.")

    selected = pd.concat(parts, ignore_index=True)
    source = (
        "Recomputed pooled Late Pre/Post variance selection "
        f"(top fraction={top_fraction:.3f})"
    )
    return selected, source


# ---------------------------------------------------------------------------
# Animal-level metric construction
# ---------------------------------------------------------------------------

def _aggregate(series: pd.Series, stat: str) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values) if stat == "mean" else np.nanmedian(values))


def calculate_syllable_level_changes(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()

    # Raw interval changes.
    out["baseline_change_sd_s"] = (
        out["late_pre_sd_s"] - out["early_pre_sd_s"]
    )
    out["lesion_change_sd_s"] = (
        out["post_sd_s"] - out["late_pre_sd_s"]
    )
    out["did_sd_s"] = (
        out["lesion_change_sd_s"] - out["baseline_change_sd_s"]
    )

    out["baseline_change_cv"] = (
        out["late_pre_cv"] - out["early_pre_cv"]
    )
    out["lesion_change_cv"] = (
        out["post_cv"] - out["late_pre_cv"]
    )
    out["did_cv"] = (
        out["lesion_change_cv"] - out["baseline_change_cv"]
    )

    # Log2 proportional interval changes.
    out["baseline_log2_change_sd"] = np.log2(
        out["late_pre_sd_s"] / out["early_pre_sd_s"]
    )
    out["lesion_log2_change_sd"] = np.log2(
        out["post_sd_s"] / out["late_pre_sd_s"]
    )
    out["did_log2_sd"] = (
        out["lesion_log2_change_sd"] - out["baseline_log2_change_sd"]
    )

    out["baseline_log2_change_cv"] = np.log2(
        out["late_pre_cv"] / out["early_pre_cv"]
    )
    out["lesion_log2_change_cv"] = np.log2(
        out["post_cv"] / out["late_pre_cv"]
    )
    out["did_log2_cv"] = (
        out["lesion_log2_change_cv"] - out["baseline_log2_change_cv"]
    )

    return out.replace([np.inf, -np.inf], np.nan)


def calculate_animal_level(
    syllable_df: pd.DataFrame,
    *,
    within_animal_stat: str,
) -> pd.DataFrame:
    metric_columns = [
        # Epoch values
        "early_pre_sd_s",
        "late_pre_sd_s",
        "post_sd_s",
        "early_pre_cv",
        "late_pre_cv",
        "post_cv",
        # Raw interval changes
        "baseline_change_sd_s",
        "lesion_change_sd_s",
        "did_sd_s",
        "baseline_change_cv",
        "lesion_change_cv",
        "did_cv",
        # Log2 changes
        "baseline_log2_change_sd",
        "lesion_log2_change_sd",
        "did_log2_sd",
        "baseline_log2_change_cv",
        "lesion_log2_change_cv",
        "did_log2_cv",
    ]

    rows = []
    for animal_id, group in syllable_df.groupby("animal_id", dropna=False):
        first = group.iloc[0]
        row = {
            "animal_id": animal_id,
            "display_group": first["display_group"],
            "display_group_original": first["display_group_original"],
            "raw_hit_type": first["raw_hit_type"],
            "n_selected_syllables": int(group["syllable"].nunique()),
            "within_animal_stat": within_animal_stat,
        }
        for column in metric_columns:
            row[column] = _aggregate(group[column], within_animal_stat)

        # Phrase counts retained across all selected syllables.
        for epoch in ["early_pre", "late_pre", "post"]:
            row[f"total_{epoch}_phrases"] = float(
                np.nansum(group[f"{epoch}_n_phrases"].to_numpy(float))
            )

        rows.append(row)

    animal = pd.DataFrame(rows)

    # Make sure all plotted groups are represented canonically.
    animal["display_group"] = animal["display_group"].apply(pooled_group)
    order = {group: index for index, group in enumerate(PLOT_GROUPS)}
    animal["_group_order"] = animal["display_group"].map(order)
    animal = animal.sort_values(
        ["_group_order", "animal_id"],
        na_position="last",
    ).drop(columns="_group_order").reset_index(drop=True)

    return animal


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci_one_sample(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    stat: str,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    observed = summarize(values, stat)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(n_bootstrap, values.size),
        replace=True,
    )

    if stat == "mean":
        boot_stats = np.nanmean(draws, axis=1)
    else:
        boot_stats = np.nanmedian(draws, axis=1)

    low, high = np.nanpercentile(boot_stats, [2.5, 97.5])
    return observed, float(low), float(high)


def signflip_p_greater(
    values: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    stat: str,
) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)

    if n == 0:
        return np.nan, "none", 0

    observed = summarize(values, stat)

    # Exact enumeration is practical for the current group sizes.
    if n <= 16:
        permuted_stats = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            signed = values * np.asarray(signs)
            permuted_stats.append(summarize(signed, stat))
        permuted_stats = np.asarray(permuted_stats, dtype=float)
        p = (
            np.sum(permuted_stats >= observed - 1e-15) + 1
        ) / (len(permuted_stats) + 1)
        return float(p), "exact", int(len(permuted_stats))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        permuted = summarize(values * signs, stat)
        if permuted >= observed - 1e-15:
            count += 1

    p = (count + 1) / (n_permutations + 1)
    return float(p), "monte_carlo", int(n_permutations)


def bootstrap_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    stat: str,
) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan

    observed = summarize(a, stat) - summarize(b, stat)
    rng = np.random.default_rng(seed)

    draw_a = rng.choice(a, size=(n_bootstrap, a.size), replace=True)
    draw_b = rng.choice(b, size=(n_bootstrap, b.size), replace=True)

    if stat == "mean":
        boot_diff = np.nanmean(draw_a, axis=1) - np.nanmean(draw_b, axis=1)
    else:
        boot_diff = np.nanmedian(draw_a, axis=1) - np.nanmedian(draw_b, axis=1)

    low, high = np.nanpercentile(boot_diff, [2.5, 97.5])
    return observed, float(low), float(high)


def label_shuffle_p_greater(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    stat: str,
) -> tuple[float, str, int]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return np.nan, "none", 0

    observed = summarize(a, stat) - summarize(b, stat)
    pooled = np.concatenate([a, b])
    n_a = int(a.size)
    total = int(pooled.size)

    n_combinations = math.comb(total, n_a)

    # Exact label enumeration when practical.
    if n_combinations <= 100_000:
        count = 0
        total_tested = 0
        indices = np.arange(total)
        for a_indices_tuple in itertools.combinations(indices, n_a):
            a_indices = np.asarray(a_indices_tuple, dtype=int)
            mask = np.ones(total, dtype=bool)
            mask[a_indices] = False
            perm_a = pooled[a_indices]
            perm_b = pooled[mask]
            permuted = summarize(perm_a, stat) - summarize(perm_b, stat)
            if permuted >= observed - 1e-15:
                count += 1
            total_tested += 1
        p = (count + 1) / (total_tested + 1)
        return float(p), "exact", int(total_tested)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        permuted = summarize(perm_a, stat) - summarize(perm_b, stat)
        if permuted >= observed - 1e-15:
            count += 1

    p = (count + 1) / (n_permutations + 1)
    return float(p), "monte_carlo", int(n_permutations)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(p.shape, np.nan)

    finite_mask = np.isfinite(p)
    finite = p[finite_mask]
    m = int(finite.size)

    if m == 0:
        return adjusted

    order = np.argsort(finite)
    ranked = finite[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    restored = np.empty_like(q)
    restored[order] = q
    adjusted[finite_mask] = restored
    return adjusted


def calculate_within_group_stats(
    animal: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    group_stat: str,
) -> pd.DataFrame:
    """
    Calculate one-sample DID statistics for every expected lesion group.

    Groups with no finite values remain in the output with n_birds=0 and NaN
    estimates. This makes missing groups explicit without emitting empty-slice
    warnings or stopping the remaining analysis.
    """
    rows = []

    for metric_index, (metric_name, spec) in enumerate(METRIC_SPECS.items()):
        for group_index, group_name in enumerate(PLOT_GROUPS):
            subset = animal[animal["display_group"] == group_name].copy()

            did_values = pd.to_numeric(
                subset[spec["did_col"]],
                errors="coerce",
            ).to_numpy(float)
            baseline_values = pd.to_numeric(
                subset[spec["baseline_col"]],
                errors="coerce",
            ).to_numpy(float)
            lesion_values = pd.to_numeric(
                subset[spec["lesion_col"]],
                errors="coerce",
            ).to_numpy(float)

            observed, ci_low, ci_high = bootstrap_ci_one_sample(
                did_values,
                n_bootstrap=n_bootstrap,
                seed=seed + metric_index * 100 + group_index,
                stat=group_stat,
            )
            p_value, permutation_mode, n_tested = signflip_p_greater(
                did_values,
                n_permutations=n_permutations,
                seed=seed + 10_000 + metric_index * 100 + group_index,
                stat=group_stat,
            )

            rows.append(
                {
                    "metric": metric_name,
                    "metric_title": spec["title"],
                    "display_group": group_name,
                    "group_label": GROUP_LABELS[group_name],
                    "n_birds": int(np.isfinite(did_values).sum()),
                    "group_statistic": group_stat,
                    "median_baseline_change": summarize(
                        baseline_values,
                        "median",
                    ),
                    "median_lesion_change": summarize(
                        lesion_values,
                        "median",
                    ),
                    "observed_DID": observed,
                    "bootstrap_CI95_low": ci_low,
                    "bootstrap_CI95_high": ci_high,
                    "signflip_alternative": "DID > 0",
                    "signflip_p_raw": p_value,
                    "signflip_mode": permutation_mode,
                    "signflip_permutations_tested": n_tested,
                    "resampling_unit": "animal",
                }
            )

    out = pd.DataFrame(rows)
    out["signflip_p_fdr_bh"] = np.nan

    for metric_name, indices in out.groupby("metric").groups.items():
        out.loc[indices, "signflip_p_fdr_bh"] = bh_adjust(
            out.loc[indices, "signflip_p_raw"]
        )

    return out

def calculate_between_group_contrasts(
    animal: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    group_stat: str,
) -> pd.DataFrame:
    contrasts = [
        (POOLED_ML, LATERAL, "Pooled M+L > lateral only"),
        (POOLED_ML, SHAM, "Pooled M+L > sham"),
    ]

    rows = []
    for metric_index, (metric_name, spec) in enumerate(METRIC_SPECS.items()):
        for contrast_index, (group_a, group_b, label) in enumerate(contrasts):
            a = animal.loc[
                animal["display_group"] == group_a,
                spec["did_col"],
            ].to_numpy(float)
            b = animal.loc[
                animal["display_group"] == group_b,
                spec["did_col"],
            ].to_numpy(float)

            observed, ci_low, ci_high = bootstrap_difference_ci(
                a,
                b,
                n_bootstrap=n_bootstrap,
                seed=seed + 20_000 + metric_index * 100 + contrast_index,
                stat=group_stat,
            )
            p_value, mode, n_tested = label_shuffle_p_greater(
                a,
                b,
                n_permutations=n_permutations,
                seed=seed + 30_000 + metric_index * 100 + contrast_index,
                stat=group_stat,
            )

            rows.append(
                {
                    "metric": metric_name,
                    "metric_title": spec["title"],
                    "contrast": label,
                    "group_A": group_a,
                    "group_B": group_b,
                    "n_A_birds": int(np.isfinite(a).sum()),
                    "n_B_birds": int(np.isfinite(b).sum()),
                    "group_statistic": group_stat,
                    "observed_DID_difference_A_minus_B": observed,
                    "bootstrap_CI95_low": ci_low,
                    "bootstrap_CI95_high": ci_high,
                    "labelshuffle_alternative": "A > B",
                    "labelshuffle_p_raw": p_value,
                    "labelshuffle_mode": mode,
                    "labelshuffle_permutations_tested": n_tested,
                    "resampling_unit": "animal",
                }
            )

    out = pd.DataFrame(rows)
    out["labelshuffle_p_fdr_bh"] = np.nan

    for metric_name, indices in out.groupby("metric").groups.items():
        out.loc[indices, "labelshuffle_p_fdr_bh"] = bh_adjust(
            out.loc[indices, "labelshuffle_p_raw"]
        )

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"[OK] Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_interval_changes(
    animal: pd.DataFrame,
    within_stats: pd.DataFrame,
    *,
    metric_name: str,
    output_path: Path,
    seed: int,
    dpi: int,
    show: bool,
) -> bool:
    """
    Plot baseline and lesion-associated changes for groups with finite data.

    Returns True when a plot is written and False when no finite data exist.
    """
    spec = METRIC_SPECS[metric_name]
    rng = np.random.default_rng(seed)

    plot_data: dict[str, pd.DataFrame] = {}
    skipped_groups = []

    for group_name in PLOT_GROUPS:
        subset = animal[animal["display_group"] == group_name].copy()
        subset = subset.replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(
            subset=[
                spec["baseline_col"],
                spec["lesion_col"],
                spec["did_col"],
            ]
        )

        if subset.empty:
            skipped_groups.append(group_name)
        else:
            plot_data[group_name] = subset

    if skipped_groups:
        print(
            f"[WARN] {metric_name}: skipping groups with no finite "
            "Early Pre/Late Pre/Post values: "
            + ", ".join(GROUP_LABELS[group] for group in skipped_groups)
        )

    active_groups = [
        group for group in PLOT_GROUPS if group in plot_data
    ]
    if not active_groups:
        print(
            f"[WARN] {metric_name}: no groups had finite interval-change "
            f"values; plot not written: {output_path.name}"
        )
        return False

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    base_positions = {}
    x_tick_positions = []
    x_tick_labels = []

    all_y = np.concatenate(
        [
            subset[
                [spec["baseline_col"], spec["lesion_col"]]
            ].to_numpy(float).ravel()
            for subset in plot_data.values()
        ]
    )
    all_y = all_y[np.isfinite(all_y)]
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    y_span = y_max - y_min
    annotation_offset = 0.08 * y_span if y_span > 0 else 0.05

    for group_index, group_name in enumerate(active_groups):
        x_baseline = group_index * 3.0
        x_lesion = x_baseline + 1.0
        base_positions[group_name] = (x_baseline, x_lesion)
        x_tick_positions.extend([x_baseline, x_lesion])
        x_tick_labels.extend(["Baseline", "Post-lesion"])

        subset = plot_data[group_name]
        jitter = rng.uniform(-0.055, 0.055, size=len(subset))

        for j, (_, row) in enumerate(subset.iterrows()):
            x1 = x_baseline + jitter[j]
            x2 = x_lesion + jitter[j]
            y1 = float(row[spec["baseline_col"]])
            y2 = float(row[spec["lesion_col"]])

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=GROUP_COLORS[group_name],
                alpha=0.55,
                linewidth=1.1,
                zorder=1,
            )
            ax.scatter(
                [x1, x2],
                [y1, y2],
                color=GROUP_COLORS[group_name],
                s=38,
                alpha=0.85,
                zorder=2,
            )

        med_baseline = summarize(
            subset[spec["baseline_col"]].to_numpy(float),
            "median",
        )
        med_lesion = summarize(
            subset[spec["lesion_col"]].to_numpy(float),
            "median",
        )
        ax.scatter(
            [x_baseline, x_lesion],
            [med_baseline, med_lesion],
            marker="D",
            color="black",
            s=58,
            zorder=4,
        )

        group_stats = within_stats[
            (within_stats["metric"] == metric_name)
            & (within_stats["display_group"] == group_name)
        ]
        if not group_stats.empty:
            stats_row = group_stats.iloc[0]
            annotation = (
                f"{GROUP_LABELS[group_name]}\n"
                f"n={int(stats_row['n_birds'])}, "
                f"DID p={format_p(stats_row['signflip_p_raw'])}"
            )
            midpoint = (x_baseline + x_lesion) / 2
            group_ymax = float(
                np.max(
                    subset[
                        [spec["baseline_col"], spec["lesion_col"]]
                    ].to_numpy(float)
                )
            )
            ax.text(
                midpoint,
                group_ymax + annotation_offset,
                annotation,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=25, ha="right")
    ax.set_ylabel(spec["interval_ylabel"])
    ax.set_xlabel("Interval within each lesion group")
    ax.set_title(
        f"{spec['title']}: baseline change versus post-lesion change"
    )
    ax.grid(axis="y", alpha=0.22)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.55",
            label="Individual bird",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="black",
            label="Group median",
            markersize=6,
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    # Expand lower margin, then add group labels beneath each pair.
    fig.subplots_adjust(bottom=0.24)
    current_ymin, current_ymax = ax.get_ylim()
    group_label_y = current_ymin - 0.11 * (current_ymax - current_ymin)
    for group_name, (x_baseline, x_lesion) in base_positions.items():
        ax.text(
            (x_baseline + x_lesion) / 2,
            group_label_y,
            GROUP_LABELS[group_name],
            ha="center",
            va="top",
            fontweight="bold",
            clip_on=False,
        )

    save_figure(fig, output_path, dpi=dpi, show=show)
    return True

def plot_did_with_ci(
    animal: pd.DataFrame,
    within_stats: pd.DataFrame,
    *,
    metric_name: str,
    output_path: Path,
    seed: int,
    dpi: int,
    show: bool,
) -> bool:
    """
    Plot DID values and bootstrap CIs for groups with finite estimates.

    Returns True when a plot is written and False when no finite data exist.
    """
    spec = METRIC_SPECS[metric_name]
    rng = np.random.default_rng(seed)

    stats_for_metric = within_stats[
        within_stats["metric"] == metric_name
    ].copy()

    plot_rows = []
    skipped_groups = []

    for group_name in PLOT_GROUPS:
        values = animal.loc[
            animal["display_group"] == group_name,
            spec["did_col"],
        ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)

        stats_row = stats_for_metric[
            stats_for_metric["display_group"] == group_name
        ]

        if values.size == 0 or stats_row.empty:
            skipped_groups.append(group_name)
            continue

        stats_row = stats_row.iloc[0]
        estimate = float(stats_row["observed_DID"])
        low = float(stats_row["bootstrap_CI95_low"])
        high = float(stats_row["bootstrap_CI95_high"])

        if not all(np.isfinite([estimate, low, high])):
            skipped_groups.append(group_name)
            continue

        plot_rows.append(
            {
                "group": group_name,
                "values": values,
                "stats": stats_row,
            }
        )

    if skipped_groups:
        print(
            f"[WARN] {metric_name}: bootstrap-CI plot omits groups with "
            "no finite DID estimate: "
            + ", ".join(GROUP_LABELS[group] for group in skipped_groups)
        )

    if not plot_rows:
        print(
            f"[WARN] {metric_name}: no finite DID values; plot not written: "
            f"{output_path.name}"
        )
        return False

    fig, ax = plt.subplots(figsize=(7.8, 6.3))

    for group_index, plot_row in enumerate(plot_rows):
        group_name = plot_row["group"]
        values = plot_row["values"]
        stats_row = plot_row["stats"]

        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        x = np.full(len(values), group_index, dtype=float) + jitter

        ax.scatter(
            x,
            values,
            s=50,
            color=GROUP_COLORS[group_name],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

        estimate = float(stats_row["observed_DID"])
        low = float(stats_row["bootstrap_CI95_low"])
        high = float(stats_row["bootstrap_CI95_high"])

        ax.errorbar(
            group_index,
            estimate,
            yerr=np.array([[estimate - low], [high - estimate]]),
            fmt="D",
            markersize=7,
            markerfacecolor="black",
            markeredgecolor="black",
            ecolor="black",
            elinewidth=2,
            capsize=5,
            capthick=1.8,
            zorder=4,
        )

        ax.annotate(
            f"n={int(stats_row['n_birds'])}\n"
            f"p={format_p(stats_row['signflip_p_raw'])}",
            xy=(group_index, high),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    active_groups = [row["group"] for row in plot_rows]
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(active_groups)))
    ax.set_xticklabels(
        [GROUP_LABELS[group] for group in active_groups],
        rotation=18,
        ha="right",
    )
    ax.set_xlabel("Lesion group")
    ax.set_ylabel(spec["did_ylabel"])
    ax.set_title(
        f"{spec['title']}\n"
        "Post-lesion change minus baseline change"
    )
    ax.grid(axis="y", alpha=0.22)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="white",
            label="Individual bird",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="black",
            label="Group statistic",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="|",
            linestyle="-",
            label="95% bootstrap CI",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    save_figure(fig, output_path, dpi=dpi, show=show)
    return True

# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def write_summary(
    path: Path,
    *,
    selection_source: str,
    animal: pd.DataFrame,
    within_stats: pd.DataFrame,
    contrasts: pd.DataFrame,
    within_animal_stat: str,
    group_stat: str,
    n_bootstrap: int,
    n_permutations: int,
) -> None:
    lines = [
        "Early Pre / Late Pre / Post difference-in-differences analysis",
        "================================================================",
        "",
        f"Selection source: {selection_source}",
        f"Within-bird syllable summary: {within_animal_stat}",
        f"Group statistic: {group_stat}",
        f"Bootstrap resamples: {n_bootstrap}",
        f"Requested permutation iterations: {n_permutations}",
        "Resampling unit: animal",
        "",
        "Bird counts:",
    ]

    for group_name in PLOT_GROUPS:
        count = int(
            animal.loc[
                animal["display_group"] == group_name,
                "animal_id",
            ].nunique()
        )
        lines.append(f"  {GROUP_LABELS[group_name]}: {count}")

    for metric_name, spec in METRIC_SPECS.items():
        lines.extend(
            [
                "",
                spec["title"],
                "-" * len(spec["title"]),
                "Within-group DID > 0:",
            ]
        )

        subset = within_stats[within_stats["metric"] == metric_name]
        for _, row in subset.iterrows():
            lines.append(
                f"  {row['group_label']}: "
                f"{group_stat} DID={row['observed_DID']:+.6g}, "
                f"95% CI [{row['bootstrap_CI95_low']:+.6g}, "
                f"{row['bootstrap_CI95_high']:+.6g}], "
                f"sign-flip p={row['signflip_p_raw']:.6g}, "
                f"BH-FDR p={row['signflip_p_fdr_bh']:.6g}"
            )

        lines.append("Between-group DID contrasts:")
        subset_contrasts = contrasts[contrasts["metric"] == metric_name]
        for _, row in subset_contrasts.iterrows():
            lines.append(
                f"  {row['contrast']}: "
                f"difference={row['observed_DID_difference_A_minus_B']:+.6g}, "
                f"95% CI [{row['bootstrap_CI95_low']:+.6g}, "
                f"{row['bootstrap_CI95_high']:+.6g}], "
                f"label-shuffle p={row['labelshuffle_p_raw']:.6g}, "
                f"BH-FDR p={row['labelshuffle_p_fdr_bh']:.6g}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    plot_dir = ensure_dir(out_dir / "plots")

    metadata = load_metadata(
        args.metadata_excel.expanduser().resolve(),
        sheet_requested=args.metadata_sheet,
        animal_col_requested=args.metadata_animal_col,
        hit_col_requested=args.metadata_hit_type_col,
    )

    long_df = load_stats_long(
        args.stats_csv.expanduser().resolve(),
        metadata=metadata,
        early_group=args.early_group,
        late_group=args.late_group,
        post_group=args.post_group,
        min_n_phrases=args.min_n_phrases,
    )
    long_df.to_csv(out_dir / "three_epoch_long_filtered.csv", index=False)

    wide = pivot_epoch_metrics(long_df)
    wide.to_csv(out_dir / "three_epoch_animal_syllable_all_candidates.csv", index=False)

    selected_pairs_path = (
        args.selected_pairs_csv.expanduser().resolve()
        if args.selected_pairs_csv is not None
        else None
    )

    selected, selection_source = select_pairs(
        wide,
        selected_pairs_csv=selected_pairs_path,
        top_fraction=args.top_fraction,
    )

    if selected.empty:
        raise ValueError(
            "No selected animal-by-syllable pairs had complete Early Pre, "
            "Late Pre, and Post data."
        )

    selected = calculate_syllable_level_changes(selected)
    selected.to_csv(
        out_dir / "selected_animal_syllable_three_epoch_metrics.csv",
        index=False,
    )

    selected[
        [
            "animal_id",
            "syllable",
            "display_group",
            "display_group_original",
            "raw_hit_type",
        ]
    ].drop_duplicates().to_csv(
        out_dir / "selected_pairs_used.csv",
        index=False,
    )

    animal = calculate_animal_level(
        selected,
        within_animal_stat=args.within_animal_stat,
    )
    animal.to_csv(
        out_dir / "animal_level_difference_in_differences.csv",
        index=False,
    )

    # Record exactly how many birds have finite values for each metric and group.
    diagnostic_rows = []
    for metric_name, spec in METRIC_SPECS.items():
        for group_name in PLOT_GROUPS:
            subset = animal[animal["display_group"] == group_name]
            finite_mask = (
                np.isfinite(
                    pd.to_numeric(
                        subset[spec["baseline_col"]],
                        errors="coerce",
                    )
                )
                & np.isfinite(
                    pd.to_numeric(
                        subset[spec["lesion_col"]],
                        errors="coerce",
                    )
                )
                & np.isfinite(
                    pd.to_numeric(
                        subset[spec["did_col"]],
                        errors="coerce",
                    )
                )
            )
            diagnostic_rows.append(
                {
                    "metric": metric_name,
                    "display_group": group_name,
                    "group_label": GROUP_LABELS[group_name],
                    "n_animals_total_after_three_epoch_filter": int(
                        subset["animal_id"].nunique()
                    ),
                    "n_animals_with_finite_metric": int(finite_mask.sum()),
                }
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(
        out_dir / "retained_bird_counts_by_group_and_metric.csv",
        index=False,
    )

    print("\n[INFO] Retained birds with finite three-epoch metrics:")
    for metric_name in METRIC_SPECS:
        print(f"  {metric_name}:")
        metric_rows = diagnostics[diagnostics["metric"] == metric_name]
        for _, diagnostic_row in metric_rows.iterrows():
            print(
                f"    {diagnostic_row['group_label']}: "
                f"{int(diagnostic_row['n_animals_with_finite_metric'])}"
            )

    within_stats = calculate_within_group_stats(
        animal,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
        group_stat=args.group_stat,
    )
    within_stats.to_csv(
        out_dir / "within_group_difference_in_differences_stats.csv",
        index=False,
    )

    contrasts = calculate_between_group_contrasts(
        animal,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
        group_stat=args.group_stat,
    )
    contrasts.to_csv(
        out_dir / "between_group_difference_in_differences_contrasts.csv",
        index=False,
    )

    combined = pd.concat(
        [
            within_stats.assign(table_section="Within-group DID"),
            contrasts.assign(table_section="Between-group DID contrast"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(
        out_dir / "difference_in_differences_combined_for_copy_paste.csv",
        index=False,
    )

    for metric_index, metric_name in enumerate(METRIC_SPECS):
        spec = METRIC_SPECS[metric_name]

        plot_interval_changes(
            animal,
            within_stats,
            metric_name=metric_name,
            output_path=plot_dir / f"{spec['stem']}_baseline_vs_lesion_change.png",
            seed=args.seed + metric_index * 100,
            dpi=args.dpi,
            show=args.show,
        )

        plot_did_with_ci(
            animal,
            within_stats,
            metric_name=metric_name,
            output_path=plot_dir / f"{spec['stem']}_difference_in_differences_bootstrap_CI.png",
            seed=args.seed + 1_000 + metric_index * 100,
            dpi=args.dpi,
            show=args.show,
        )

    summary_path = out_dir / "difference_in_differences_summary.txt"
    write_summary(
        summary_path,
        selection_source=selection_source,
        animal=animal,
        within_stats=within_stats,
        contrasts=contrasts,
        within_animal_stat=args.within_animal_stat,
        group_stat=args.group_stat,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )

    print("\n[OK] Analysis complete.")
    print(f"[OK] Selection: {selection_source}")
    print(f"[OK] Animals analyzed: {animal['animal_id'].nunique()}")
    print(f"[OK] Selected animal-by-syllable pairs: {len(selected)}")
    print(f"[OK] Output directory: {out_dir}")
    print(f"[OK] Summary: {summary_path}")

    print("\nPrimary raw-SD DID results:")
    primary = within_stats[within_stats["metric"] == "sd_raw"]
    for _, row in primary.iterrows():
        print(
            f"  {row['group_label']}: "
            f"DID={row['observed_DID']:+.4f} s, "
            f"95% CI [{row['bootstrap_CI95_low']:+.4f}, "
            f"{row['bootstrap_CI95_high']:+.4f}], "
            f"p={row['signflip_p_raw']:.4g}"
        )


if __name__ == "__main__":
    main()
