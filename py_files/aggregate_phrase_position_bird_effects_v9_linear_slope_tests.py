#!/usr/bin/env python3
"""
Aggregate phrase-position acoustic/spectrogram analyses across birds.

Designed for outputs from:
    phrase_position_acoustic_crosscorr_v6_majority_vote_runs.py

Main questions:
1) Do syllables become less spectrogram-similar as repeated phrases continue?
2) Is that phrase-position effect stronger after lesion than before lesion?
3) In the post-lesion-only stutter tail, are syllables more acoustically different
   than in the shared pre/post phrase-position range?
4) Are these effects consistent across Medial + Lateral lesion birds?

Input examples:
    <root>/<ANIMAL>/<ANIMAL>_all_labels_phrase_position_segment_features.csv
or:
    <root>/<ANIMAL>/<ANIMAL>_label<LABEL>_phrase_position_segment_features.csv

Key outputs:
    phrase_position_label_slopes.csv
    phrase_position_label_effects.csv
  phrase_position_slope_against_zero_stats.csv
    phrase_position_tail_effects.csv
    phrase_position_pre_post_distribution_label_tests.csv
    phrase_position_pre_post_distribution_bird_summary.csv
    phrase_position_pre_post_distribution_bird_stats.csv
    phrase_position_bird_summary.csv
    phrase_position_bird_stats.csv

Figures:
    post_minus_pre_degradation_slope_by_bird_*.png
    pre_vs_post_degradation_slope_bird_paired_*.png
    baseline_vs_lesion_slope_change_*.png
    post_only_tail_effect_by_bird_*.png
    max_post_phrase_extent_by_bird_*.png
    pre_post_distribution_effect_by_bird_*.png
    pre_vs_post_distribution_bird_paired_*.png
    aggregate_binned_*_vs_elapsed_time.png
    aggregate_binned_*_vs_previous_repeats.png

Interpretation:
    For corr_to_phrase_early_template:
        lower correlation = more acoustic change/degradation.
        The script converts raw slope to degradation_slope = -1 * raw_slope.

    For distance_from_phrase_early_template:
        higher distance = more acoustic change/degradation.
        The script uses degradation_slope = raw_slope.

    post_minus_pre_degradation_slope > 0 means:
        post-lesion syllables lose similarity / gain distance more strongly with phrase position.

    tail_degradation_minus_shared > 0 means:
        the post-lesion-only stutter tail is more acoustically different than the shared post-lesion range.

    post_minus_pre_median_degradation_value > 0 means:
        post-lesion segment values are more degraded/changed than pre-lesion values.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PRE_POST_ORDER = ["pre", "post"]
PREDICTORS = ["elapsed_time_in_phrase_s", "n_previous_segments"]

PREDICTOR_LABELS = {
    "elapsed_time_in_phrase_s": "Elapsed time in repeated phrase (s)",
    "n_previous_segments": "Number of previous analyzed syllables in phrase",
}
PREDICTOR_SHORT = {
    "elapsed_time_in_phrase_s": "elapsed_time",
    "n_previous_segments": "previous_repeats",
}
PREDICTOR_UNITS = {
    "elapsed_time_in_phrase_s": "per second",
    "n_previous_segments": "per previous repeat",
}

FEATURE_LABELS = {
    "corr_to_phrase_early_template": "Spectrogram correlation to early-phrase template",
    "distance_from_phrase_early_template": "Spectrogram distance from early-phrase template (1 - corr)",
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative (kHz/s)",
    "mean_wiener_entropy": "Mean Wiener entropy",
    "q95_wiener_entropy": "95th percentile Wiener entropy",
}

ANIMAL_COL_CANDIDATES = ["Animal ID", "animal_id", "Animal", "animal", "Bird", "bird"]
LESION_HIT_TYPE_COL_CANDIDATES = [
    "Lesion hit type", "lesion_hit_type", "Hit type", "hit_type", "Lesion Type",
    "lesion type", "Treatment group", "treatment_group", "Group", "group",
]

HIGH_VARIANCE_ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal", "animal", "Bird", "bird",
    "animal", "bird_id", "AnimalID",
]
HIGH_VARIANCE_LABEL_COL_CANDIDATES = [
    "label", "Label", "syllable", "Syllable", "syllable_label", "Syllable label",
    "cluster", "Cluster", "cluster_id", "Cluster ID", "hdbscan_label", "HDBSCAN label",
    "syllable_cluster", "Syllable cluster", "chosen_label", "selected_label",
]
HIGH_VARIANCE_SELECT_COL_CANDIDATES = [
    "is_top_30", "is_top30", "top_30", "top30", "top30_phrase_variance",
    "high_variance", "highly_variable", "include", "selected", "keep",
]
HIGH_VARIANCE_RANK_COL_CANDIDATES = [
    "rank", "Rank", "variance_rank", "Variance rank", "phrase_variance_rank",
    "duration_variance_rank", "repeat_time_variance_rank", "rank_within_bird",
]
HIGH_VARIANCE_METRIC_COL_CANDIDATES = [
    "phrase_duration_variance", "Phrase duration variance",
    "phrase_duration_variance_ms2", "Variance_ms2", "variance_ms2",
    "PostMinusPreVar_ms2", "post_minus_pre_var_ms2",
    "PostMinusPreVariance_ms2", "post_minus_pre_variance_ms2",
    "PrePooledVariance_ms2", "pre_pooled_variance_ms2",
    "Std_ms", "std_ms", "IQR_ms", "iqr_ms",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(x).strip().lower()).strip()


def find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None




def normalize_animal_id(value) -> str:
    """Normalize bird/animal IDs for matching across CSVs."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label_id(value) -> str:
    """Normalize syllable/cluster labels for matching across CSVs.

    This handles common cases like integer labels stored as 3.0, or strings like
    "label3" / "USA5288_label3".
    """
    if pd.isna(value):
        return ""
    s = str(value).strip()
    # If the value is numeric-looking, convert 3.0 -> 3 while preserving labels like 3a.
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    # Pull the final label number/name from strings like USA5288_label3 or label3.
    m = re.search(r"(?:^|_)label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def parse_bool_like(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = normalize_text(value)
    return s in {"1", "true", "t", "yes", "y", "include", "included", "keep", "kept", "selected", "top", "top 30", "high", "high variance", "highly variable"}


def read_table_auto(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV/Excel tables used for optional high-variance label filtering."""
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            return pd.read_csv(p)
    return pd.read_csv(p)


def aggregate_metric(series: pd.Series, how: str) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return np.nan
    if how == "median":
        return float(vals.median())
    if how == "mean":
        return float(vals.mean())
    if how == "min":
        return float(vals.min())
    return float(vals.max())


def load_high_variance_label_filter(
    high_variance_labels_csv: str,
    animal_col: Optional[str] = None,
    label_col: Optional[str] = None,
    select_col: Optional[str] = None,
    metric_col: Optional[str] = None,
    rank_col: Optional[str] = None,
    top_fraction: float = 0.30,
    metric_direction: str = "largest",
    metric_agg: str = "max",
    rank_direction: str = "ascending",
    use_all_rows: bool = False,
) -> Tuple[set[tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """Return selected (animal_id, label) pairs for high repeat-time variability.

    This function is intentionally flexible because the upstream "top 30%" table may
    be either:
      1) already filtered to the high-variance labels, or
      2) a full phrase-duration stats table with a metric/rank column.

    Selection priority:
      A. if use_all_rows=True, treat every row as an already-filtered high-variance label list
      B. explicit/auto boolean include column, if present
      C. explicit/auto rank column, choosing the best top_fraction per bird
      D. explicit/auto metric column, choosing the largest/smallest top_fraction per bird
      E. all rows in the table, treated as an already-filtered high-variance list
    """
    if not (0 < float(top_fraction) <= 1):
        raise ValueError("--high-variance-top-fraction must be > 0 and <= 1")

    raw = read_table_auto(high_variance_labels_csv)
    if raw.empty:
        raise ValueError(f"High-variance labels table is empty: {high_variance_labels_csv}")

    animal_col = animal_col or find_first_existing(raw.columns, HIGH_VARIANCE_ANIMAL_COL_CANDIDATES)
    label_col = label_col or find_first_existing(raw.columns, HIGH_VARIANCE_LABEL_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(
            "Could not identify animal ID column in high-variance labels table. "
            f"Use --high-variance-animal-col. Columns: {list(raw.columns)}"
        )
    if label_col is None:
        raise ValueError(
            "Could not identify syllable/cluster label column in high-variance labels table. "
            f"Use --high-variance-label-col. Columns: {list(raw.columns)}"
        )

    work = raw.copy()
    work["animal_id"] = work[animal_col].map(normalize_animal_id)
    work["label"] = work[label_col].map(normalize_label_id)
    work = work[(work["animal_id"] != "") & (work["label"] != "")].copy()
    if work.empty:
        raise ValueError("No non-empty animal_id/label pairs found in high-variance labels table after normalization.")

    selection_source = "all_rows_in_csv_treated_as_already_filtered"
    metric_or_rank_used = ""

    if use_all_rows:
        selected = work[["animal_id", "label"]].drop_duplicates().copy()
        selection_source = "all_rows_in_csv_treated_as_already_filtered_forced"
    else:
        select_col = select_col or find_first_existing(work.columns, HIGH_VARIANCE_SELECT_COL_CANDIDATES)
        rank_col = rank_col or find_first_existing(work.columns, HIGH_VARIANCE_RANK_COL_CANDIDATES)
        metric_col = metric_col or find_first_existing(work.columns, HIGH_VARIANCE_METRIC_COL_CANDIDATES)

    if (not use_all_rows) and select_col is not None:
        selected = work[work[select_col].map(parse_bool_like)].copy()
        selection_source = f"boolean_select_col:{select_col}"
        metric_or_rank_used = select_col
    elif (not use_all_rows) and rank_col is not None:
        rank_work = work[["animal_id", "label", rank_col]].copy()
        rank_work["_rank"] = pd.to_numeric(rank_work[rank_col], errors="coerce")
        rank_work = rank_work[np.isfinite(rank_work["_rank"])].copy()
        if rank_work.empty:
            raise ValueError(f"Rank column {rank_col!r} was found, but it did not contain finite numeric values.")
        # If multiple rows exist per label, keep the best rank for that label.
        if rank_direction == "descending":
            label_scores = rank_work.groupby(["animal_id", "label"], as_index=False)["_rank"].max()
            ascending = False
        else:
            label_scores = rank_work.groupby(["animal_id", "label"], as_index=False)["_rank"].min()
            ascending = True
        selected_parts = []
        for animal, sb in label_scores.groupby("animal_id"):
            sb = sb.sort_values("_rank", ascending=ascending).copy()
            n_keep = max(1, int(math.ceil(sb.shape[0] * float(top_fraction))))
            selected_parts.append(sb.head(n_keep))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=["animal_id", "label"])
        selection_source = f"rank_col:{rank_col}; top_fraction:{top_fraction}; direction:{rank_direction}"
        metric_or_rank_used = rank_col
    elif (not use_all_rows) and metric_col is not None:
        metric_work = work[["animal_id", "label", metric_col]].copy()
        metric_work["_metric"] = pd.to_numeric(metric_work[metric_col], errors="coerce")
        metric_work = metric_work[np.isfinite(metric_work["_metric"])].copy()
        if metric_work.empty:
            raise ValueError(f"Metric column {metric_col!r} was found, but it did not contain finite numeric values.")
        label_scores = (
            metric_work.groupby(["animal_id", "label"])["_metric"]
            .apply(lambda x: aggregate_metric(x, metric_agg))
            .reset_index(name="_metric")
        )
        ascending = metric_direction == "smallest"
        selected_parts = []
        for animal, sb in label_scores.groupby("animal_id"):
            sb = sb.sort_values("_metric", ascending=ascending).copy()
            n_keep = max(1, int(math.ceil(sb.shape[0] * float(top_fraction))))
            selected_parts.append(sb.head(n_keep))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=["animal_id", "label"])
        selection_source = f"metric_col:{metric_col}; top_fraction:{top_fraction}; direction:{metric_direction}; agg:{metric_agg}"
        metric_or_rank_used = metric_col
    else:
        selected = work[["animal_id", "label"]].drop_duplicates().copy()

    selected = selected[["animal_id", "label"]].drop_duplicates().copy()
    if selected.empty:
        raise ValueError("High-variance label filter selected 0 animal_id/label pairs.")
    selected["selection_source"] = selection_source
    selected["metric_or_rank_col_used"] = metric_or_rank_used
    selected["source_table"] = str(Path(high_variance_labels_csv).expanduser())

    summary = (
        selected.groupby("animal_id").agg(n_high_variance_labels=("label", "nunique"))
        .reset_index()
        .sort_values("animal_id")
    )
    selected_pairs = set(zip(selected["animal_id"].astype(str), selected["label"].astype(str)))
    return selected_pairs, selected.sort_values(["animal_id", "label"]), summary


def apply_high_variance_label_filter(
    segment_df: pd.DataFrame,
    selected_pairs: set[tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter segment rows to selected high-variance animal x label pairs."""
    if segment_df.empty:
        return segment_df, pd.DataFrame()
    work = segment_df.copy()
    work["_animal_id_norm"] = work["animal_id"].map(normalize_animal_id)
    work["_label_norm"] = work["label"].map(normalize_label_id)
    work["_keep_high_variance_label"] = [
        (a, l) in selected_pairs for a, l in zip(work["_animal_id_norm"], work["_label_norm"])
    ]
    before = (
        work.groupby("animal_id").agg(
            n_segment_rows_before=("label", "size"),
            n_labels_before=("label", lambda x: pd.Series(x).map(normalize_label_id).nunique()),
        )
        .reset_index()
    )
    filtered = work[work["_keep_high_variance_label"]].copy()
    after = (
        filtered.groupby("animal_id").agg(
            n_segment_rows_after=("label", "size"),
            n_labels_after=("label", lambda x: pd.Series(x).map(normalize_label_id).nunique()),
            labels_kept=("label", lambda x: ",".join(sorted(pd.Series(x).map(normalize_label_id).unique(), key=str))),
        )
        .reset_index()
    )
    summary = before.merge(after, on="animal_id", how="left")
    for col in ["n_segment_rows_after", "n_labels_after"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["labels_kept"] = summary["labels_kept"].fillna("")
    summary["n_segment_rows_removed"] = summary["n_segment_rows_before"] - summary["n_segment_rows_after"]
    summary["n_labels_removed"] = summary["n_labels_before"] - summary["n_labels_after"]
    filtered = filtered.drop(columns=["_animal_id_norm", "_label_norm", "_keep_high_variance_label"])
    if filtered.empty:
        raise RuntimeError(
            "High-variance label filtering removed all segment rows. Check that animal and label columns match "
            "between the high-variance CSV and the phrase-position segment CSVs."
        )
    return filtered, summary

def is_medial_lateral_hit(hit_type) -> bool:
    s = normalize_text(hit_type)
    raw = str(hit_type).strip().lower().replace(" ", "")
    return (
        ("medial" in s and "lateral" in s)
        or ("m+l" in raw)
        or ("medial+lateral" in raw)
        or ("medialandlateral" in raw)
    )


def load_medial_lateral_animals(
    metadata_excel_path: str,
    sheet_name: str = "animal_hit_type_summary",
    animal_col: Optional[str] = None,
    hit_type_col: Optional[str] = None,
) -> Tuple[set[str], pd.DataFrame]:
    meta = pd.read_excel(Path(metadata_excel_path).expanduser(), sheet_name=sheet_name)
    animal_col = animal_col or find_first_existing(meta.columns, ANIMAL_COL_CANDIDATES)
    hit_type_col = hit_type_col or find_first_existing(meta.columns, LESION_HIT_TYPE_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(f"Could not identify animal ID column in sheet {sheet_name!r}. Columns: {list(meta.columns)}")
    if hit_type_col is None:
        raise ValueError(f"Could not identify lesion hit-type column in sheet {sheet_name!r}. Columns: {list(meta.columns)}")

    work = meta.copy()
    work[animal_col] = work[animal_col].astype(str).str.strip()
    label_ml = work[hit_type_col].map(is_medial_lateral_hit)

    # Fallback: if the sheet has separate medial/lateral hit columns, require both to be non-miss.
    medial_col = find_first_existing(work.columns, ["Medial Area X hit type", "Medial hit type", "medial_hit_type"])
    lateral_col = find_first_existing(work.columns, ["Lateral Area X hit type", "Lateral hit type", "lateral_hit_type"])
    parsed_ml = pd.Series(False, index=work.index)
    if medial_col is not None and lateral_col is not None:
        miss_like = {"", "miss", "nan", "none", "no", "n"}
        medial_hit = work[medial_col].map(lambda x: normalize_text(x) not in miss_like)
        lateral_hit = work[lateral_col].map(lambda x: normalize_text(x) not in miss_like)
        parsed_ml = medial_hit & lateral_hit

    work["_is_medial_lateral"] = label_ml | parsed_ml
    work["_medial_lateral_filter_source"] = np.where(
        label_ml, "lesion_hit_type_label",
        np.where(parsed_ml, "parsed_medial_lateral_hit_columns", "not_medial_lateral"),
    )
    animals = set(work.loc[work["_is_medial_lateral"], animal_col].dropna().astype(str).str.strip())
    return animals, work


def parse_animal_label_from_filename(path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Handles USA5288_label3_phrase_position_segment_features.csv
    m = re.search(r"(?P<animal>[^/\\]+)_label(?P<label>.+?)_phrase_position_segment_features\.csv$", path.name)
    if m:
        return m.group("animal"), m.group("label")
    # Handles USA5288_all_labels_phrase_position_segment_features.csv
    m = re.search(r"(?P<animal>[^/\\]+)_all_labels_phrase_position_segment_features\.csv$", path.name)
    if m:
        return m.group("animal"), None
    return None, None


def find_segment_feature_files(root: Path) -> List[Path]:
    # Prefer combined all-label files to avoid double counting if individual label files also exist.
    all_label_files = sorted(root.rglob("*_all_labels_phrase_position_segment_features.csv"))
    if all_label_files:
        return all_label_files
    return sorted(root.rglob("*_phrase_position_segment_features.csv"))


def read_segment_feature_tables(
    root: Path,
    allowed_animals: Optional[set[str]] = None,
    min_segments_per_table: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = find_segment_feature_files(root)
    if not files:
        raise FileNotFoundError(f"No phrase-position segment feature CSVs found under: {root}")

    frames = []
    used_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        if df.empty or df.shape[0] < min_segments_per_table:
            continue

        animal_from_file, label_from_file = parse_animal_label_from_filename(f)
        if "animal_id" not in df.columns:
            df["animal_id"] = animal_from_file or "unknown"
        if "label" not in df.columns and label_from_file is not None:
            df["label"] = label_from_file
        if "label" not in df.columns:
            print(f"[WARN] Skipping {f}; no label column and label not inferable from file name")
            continue

        df["animal_id"] = df["animal_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()

        if allowed_animals is not None:
            df = df[df["animal_id"].isin(allowed_animals)].copy()
            if df.empty:
                continue

        # Standardize aliases from older scripts.
        if "n_previous_segments" not in df.columns and "n_previous_repeats" in df.columns:
            df["n_previous_segments"] = df["n_previous_repeats"]
        if "pre_post" not in df.columns and "period" in df.columns:
            df["pre_post"] = np.where(df["period"].astype(str).str.contains("post"), "post", "pre")

        df["source_csv"] = str(f)
        frames.append(df)
        used_rows.append({
            "csv_path": str(f),
            "n_rows_used": int(df.shape[0]),
            "animals": ",".join(sorted(df["animal_id"].unique())),
            "labels": ",".join(sorted(df["label"].astype(str).unique())),
        })

    if not frames:
        raise RuntimeError("No usable segment-feature rows after filtering.")

    out = pd.concat(frames, ignore_index=True)
    out["pre_post"] = np.where(out["period"].astype(str).str.contains("post"), "post", "pre")
    return out, pd.DataFrame(used_rows)


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_nanmedian(x) -> float:
    arr = finite_array(x)
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr))


def safe_nanmean(x) -> float:
    arr = finite_array(x)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def linear_slope_and_stats(x, y, min_n: int = 5) -> Dict[str, float | str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    row: Dict[str, float | str] = {
        "n": n,
        "slope": np.nan,
        "intercept": np.nan,
        "r_value": np.nan,
        "p_value": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "fit_status": "ok",
    }
    if n < min_n:
        row["fit_status"] = "too_few_points"
        return row
    xv = x[valid]
    yv = y[valid]
    if np.unique(xv).size < 2:
        row["fit_status"] = "constant_x"
        return row
    if np.unique(yv).size < 2:
        row["fit_status"] = "constant_y"
        return row

    if SCIPY_AVAILABLE:
        try:
            lr = stats.linregress(xv, yv)
            sp = stats.spearmanr(xv, yv)
            row.update({
                "slope": float(lr.slope),
                "intercept": float(lr.intercept),
                "r_value": float(lr.rvalue),
                "p_value": float(lr.pvalue),
                "spearman_r": float(sp.correlation),
                "spearman_p": float(sp.pvalue),
            })
        except Exception as e:
            row["fit_status"] = f"fit_error:{e}"
    else:
        try:
            slope, intercept = np.polyfit(xv, yv, deg=1)
            row.update({"slope": float(slope), "intercept": float(intercept)})
        except Exception as e:
            row["fit_status"] = f"fit_error:{e}"
    return row


def degradation_multiplier(feature: str) -> float:
    # Converts raw slope to a slope where positive means "more acoustic degradation/change".
    if feature == "corr_to_phrase_early_template":
        return -1.0
    return 1.0


def degradation_value(feature: str, values) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if feature == "corr_to_phrase_early_template":
        return 1.0 - vals
    return vals


def compute_label_slopes(df: pd.DataFrame, features: List[str], min_n: int) -> pd.DataFrame:
    rows = []
    group_specs = []
    for p in PERIOD_ORDER:
        group_specs.append(("period", p))
    for pp in PRE_POST_ORDER:
        group_specs.append(("pre_post", pp))

    for (animal, label), sub_label in df.groupby(["animal_id", "label"], dropna=False):
        for feature in features:
            if feature not in sub_label.columns:
                print(f"[WARN] Missing feature column {feature}; skipping")
                continue
            mult = degradation_multiplier(feature)
            for group_col, group in group_specs:
                sub = sub_label[sub_label[group_col].astype(str) == group].copy()
                for predictor in PREDICTORS:
                    if predictor not in sub.columns:
                        continue
                    res = linear_slope_and_stats(sub[predictor].to_numpy(dtype=float), sub[feature].to_numpy(dtype=float), min_n=min_n)
                    raw_slope = float(res["slope"]) if np.isfinite(res["slope"]) else np.nan
                    rows.append({
                        "animal_id": animal,
                        "label": label,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS.get(feature, feature),
                        "group_col": group_col,
                        "group": group,
                        "predictor": predictor,
                        "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
                        "n": int(res["n"]),
                        "slope": raw_slope,
                        "degradation_slope": mult * raw_slope if np.isfinite(raw_slope) else np.nan,
                        "intercept": res["intercept"],
                        "r_value": res["r_value"],
                        "p_value": res["p_value"],
                        "spearman_r": res["spearman_r"],
                        "spearman_p": res["spearman_p"],
                        "fit_status": res["fit_status"],
                        "degradation_multiplier": mult,
                    })
    return pd.DataFrame(rows)


def get_slope_value(slopes_df: pd.DataFrame, animal: str, label: str, feature: str, predictor: str,
                    group_col: str, group: str, value_col: str = "degradation_slope") -> float:
    sub = slopes_df[
        (slopes_df["animal_id"].astype(str) == str(animal)) &
        (slopes_df["label"].astype(str) == str(label)) &
        (slopes_df["feature"] == feature) &
        (slopes_df["predictor"] == predictor) &
        (slopes_df["group_col"] == group_col) &
        (slopes_df["group"] == group)
    ]
    if sub.empty:
        return np.nan
    return float(sub[value_col].iloc[0])


def compute_label_effects(slopes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if slopes_df.empty:
        return pd.DataFrame(rows)
    key_df = slopes_df[["animal_id", "label", "feature", "predictor"]].drop_duplicates()
    for _, key in key_df.iterrows():
        animal = str(key["animal_id"])
        label = str(key["label"])
        feature = key["feature"]
        predictor = key["predictor"]
        pre = get_slope_value(slopes_df, animal, label, feature, predictor, "pre_post", "pre")
        post = get_slope_value(slopes_df, animal, label, feature, predictor, "pre_post", "post")
        ep = get_slope_value(slopes_df, animal, label, feature, predictor, "period", "early_pre")
        lp = get_slope_value(slopes_df, animal, label, feature, predictor, "period", "late_pre")
        epost = get_slope_value(slopes_df, animal, label, feature, predictor, "period", "early_post")
        lpost = get_slope_value(slopes_df, animal, label, feature, predictor, "period", "late_post")

        baseline_signed = lp - ep if np.isfinite(lp) and np.isfinite(ep) else np.nan
        lesion_signed = post - lp if np.isfinite(post) and np.isfinite(lp) else np.nan
        baseline_abs = abs(baseline_signed) if np.isfinite(baseline_signed) else np.nan
        lesion_abs = abs(lesion_signed) if np.isfinite(lesion_signed) else np.nan

        rows.append({
            "animal_id": animal,
            "label": label,
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
            "pre_degradation_slope": pre,
            "post_degradation_slope": post,
            "post_minus_pre_degradation_slope": post - pre if np.isfinite(post) and np.isfinite(pre) else np.nan,
            "early_pre_degradation_slope": ep,
            "late_pre_degradation_slope": lp,
            "early_post_degradation_slope": epost,
            "late_post_degradation_slope": lpost,
            "baseline_slope_change_signed": baseline_signed,
            "lesion_slope_change_signed": lesion_signed,
            "lesion_minus_baseline_slope_change_signed": lesion_signed - baseline_signed if np.isfinite(lesion_signed) and np.isfinite(baseline_signed) else np.nan,
            "baseline_slope_change_abs": baseline_abs,
            "lesion_slope_change_abs": lesion_abs,
            "lesion_minus_baseline_slope_change_abs": lesion_abs - baseline_abs if np.isfinite(lesion_abs) and np.isfinite(baseline_abs) else np.nan,
        })
    return pd.DataFrame(rows)


def compute_tail_effects(df: pd.DataFrame, features: List[str], min_tail_n: int = 5) -> pd.DataFrame:
    """Compute post-only stutter-tail summaries for each animal x label x feature x predictor.

    Shared post range is post rows with predictor <= max pre predictor.
    Tail range is post rows with predictor > max pre predictor.
    For correlation, values are converted to degradation = 1 - corr before comparing tail vs shared.
    For distance and acoustic features, the raw values are used.
    """
    rows = []
    for (animal, label), sub_label in df.groupby(["animal_id", "label"], dropna=False):
        pre = sub_label[sub_label["pre_post"] == "pre"]
        post = sub_label[sub_label["pre_post"] == "post"]
        if pre.empty or post.empty:
            continue
        for predictor in PREDICTORS:
            if predictor not in sub_label.columns:
                continue
            max_pre = np.nanmax(pd.to_numeric(pre[predictor], errors="coerce"))
            max_post = np.nanmax(pd.to_numeric(post[predictor], errors="coerce"))
            if not (np.isfinite(max_pre) and np.isfinite(max_post)):
                continue
            post_shared = post[pd.to_numeric(post[predictor], errors="coerce") <= max_pre].copy()
            post_tail = post[pd.to_numeric(post[predictor], errors="coerce") > max_pre].copy()
            tail_extent = max_post - max_pre
            for feature in features:
                if feature not in sub_label.columns:
                    continue
                shared_vals = degradation_value(feature, pd.to_numeric(post_shared[feature], errors="coerce"))
                tail_vals = degradation_value(feature, pd.to_numeric(post_tail[feature], errors="coerce"))
                pre_vals = degradation_value(feature, pd.to_numeric(pre[feature], errors="coerce"))
                post_vals = degradation_value(feature, pd.to_numeric(post[feature], errors="coerce"))
                rows.append({
                    "animal_id": animal,
                    "label": label,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
                    "max_pre_predictor": float(max_pre),
                    "max_post_predictor": float(max_post),
                    "post_only_tail_extent": float(tail_extent),
                    "n_pre": int(np.isfinite(pre_vals).sum()),
                    "n_post_shared": int(np.isfinite(shared_vals).sum()),
                    "n_post_tail": int(np.isfinite(tail_vals).sum()),
                    "has_tail": bool(np.isfinite(tail_extent) and tail_extent > 0 and np.isfinite(tail_vals).sum() >= min_tail_n),
                    "median_pre_degradation_value": safe_nanmedian(pre_vals),
                    "median_post_degradation_value": safe_nanmedian(post_vals),
                    "median_post_shared_degradation_value": safe_nanmedian(shared_vals),
                    "median_post_tail_degradation_value": safe_nanmedian(tail_vals),
                    "tail_minus_shared_degradation_value": (
                        safe_nanmedian(tail_vals) - safe_nanmedian(shared_vals)
                        if np.isfinite(safe_nanmedian(tail_vals)) and np.isfinite(safe_nanmedian(shared_vals)) else np.nan
                    ),
                    "tail_minus_pre_degradation_value": (
                        safe_nanmedian(tail_vals) - safe_nanmedian(pre_vals)
                        if np.isfinite(safe_nanmedian(tail_vals)) and np.isfinite(safe_nanmedian(pre_vals)) else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out


def wilcoxon_1sample(vals, alternative="greater") -> float:
    arr = finite_array(vals)
    if arr.size < 1 or not SCIPY_AVAILABLE:
        return np.nan
    try:
        # scipy wilcoxon errors if all zeros with zero_method="wilcox".
        if np.allclose(arr, 0):
            return 1.0
        return float(stats.wilcoxon(arr, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def sign_test(vals, alternative="greater") -> float:
    arr = finite_array(vals)
    arr = arr[~np.isclose(arr, 0)]
    if arr.size == 0 or not SCIPY_AVAILABLE:
        return np.nan
    n_pos = int(np.sum(arr > 0))
    n = int(arr.size)
    try:
        if alternative == "greater":
            return float(stats.binomtest(n_pos, n=n, p=0.5, alternative="greater").pvalue)
        elif alternative == "less":
            return float(stats.binomtest(n_pos, n=n, p=0.5, alternative="less").pvalue)
        return float(stats.binomtest(n_pos, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def mannwhitney_pre_post(pre_vals, post_vals, alternative="greater") -> Tuple[float, float]:
    """Mann-Whitney U test comparing post vs pre values.

    Values should already be converted to the analysis scale. In this script,
    that means corr_to_phrase_early_template is tested as degradation = 1 - corr,
    so alternative="greater" asks whether post-lesion degradation is higher.

    Returns:
        (p_value, u_statistic)
    """
    pre = finite_array(pre_vals)
    post = finite_array(post_vals)
    if pre.size < 1 or post.size < 1 or not SCIPY_AVAILABLE:
        return np.nan, np.nan
    try:
        res = stats.mannwhitneyu(post, pre, alternative=alternative)
        return float(res.pvalue), float(res.statistic)
    except Exception:
        return np.nan, np.nan


def cliffs_delta_post_vs_pre(pre_vals, post_vals) -> float:
    """Cliff's delta effect size; positive means post values tend to be higher than pre."""
    pre = finite_array(pre_vals)
    post = finite_array(post_vals)
    if pre.size < 1 or post.size < 1 or not SCIPY_AVAILABLE:
        return np.nan
    try:
        u = stats.mannwhitneyu(post, pre, alternative="two-sided").statistic
        return float((2.0 * u / (post.size * pre.size)) - 1.0)
    except Exception:
        return np.nan


def degradation_axis_label(feature: str) -> str:
    if feature == "corr_to_phrase_early_template":
        return "Degradation value (1 - spectrogram correlation)"
    if feature == "distance_from_phrase_early_template":
        return "Degradation value (spectrogram distance; 1 - corr)"
    return f"{FEATURE_LABELS.get(feature, feature)} (higher = larger value)"


def is_pre_normalized_feature(feature: str) -> bool:
    return str(feature).startswith("pre_norm__")


def wrap_axis_label(text: str, width: int = 30) -> str:
    parts = []
    for chunk in str(text).split("\n"):
        chunk = chunk.strip()
        if not chunk:
            parts.append("")
        else:
            parts.append("\n".join(textwrap.wrap(chunk, width=width, break_long_words=False)))
    return "\n".join(parts)


def distribution_ylabel(feature: str, mode: str) -> str:
    if is_pre_normalized_feature(feature):
        if mode == "paired":
            return "Bird median normalized value"
        if mode == "delta":
            return "Post - pre median normalized value"
        if mode == "tail":
            return "Post-only tail - shared post range\nmedian normalized value"
    else:
        if mode == "paired":
            return f"Bird median {degradation_axis_label(feature)}"
        if mode == "delta":
            return "Post - pre median degradation value"
        if mode == "tail":
            return "Post-only tail - shared post range\nmedian degradation value"
    return mode


def save_figure_with_margins(fig, out_png, left=0.18, right=0.98, top=0.88, bottom=0.14, wspace=None, hspace=None):
    kwargs = {"left": left, "right": right, "top": top, "bottom": bottom}
    if wspace is not None:
        kwargs["wspace"] = wspace
    if hspace is not None:
        kwargs["hspace"] = hspace
    fig.subplots_adjust(**kwargs)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")



def add_pre_baseline_normalized_features(
    df: pd.DataFrame,
    features: List[str],
    baseline_stat: str = "median",
    min_pre_n: int = 5,
    epsilon: float = 1e-9,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Add per-animal x label feature columns normalized to the pre-lesion baseline.

    For each animal x syllable label x feature, this function first converts the feature to
    the same analysis scale used elsewhere in the script:
      * corr_to_phrase_early_template -> degradation = 1 - corr
      * distance/entropy/pitch features -> raw value

    It then computes a pre-lesion baseline for that animal x label x feature and creates:
        normalized value = analysis value / pre-lesion baseline

    Therefore, for each usable animal x label, the pre-lesion median-normalized value is
    approximately 1 when baseline_stat="median". Post-lesion values > 1 mean the feature
    increased relative to that syllable's own pre-lesion baseline.

    Rows with too few pre-lesion values or near-zero baselines are left as NaN for that
    normalized feature so they do not produce unstable ratios.
    """
    if baseline_stat not in {"median", "mean"}:
        raise ValueError("baseline_stat must be 'median' or 'mean'")
    if df.empty:
        return df.copy(), [], pd.DataFrame()

    out = df.copy()
    norm_features: List[str] = []
    info_rows = []

    # Ensure pre_post exists and is standardized.
    if "pre_post" not in out.columns and "period" in out.columns:
        out["pre_post"] = np.where(out["period"].astype(str).str.contains("post"), "post", "pre")

    for feature in features:
        if feature not in out.columns:
            print(f"[WARN] Cannot pre-normalize missing feature column {feature}; skipping")
            continue

        norm_col = f"pre_norm__{safe_name(feature)}"
        norm_features.append(norm_col)
        FEATURE_LABELS[norm_col] = (
            f"Pre-normalized {degradation_axis_label(feature)} "
            f"(value / animal×label pre-lesion {baseline_stat})"
        )
        out[norm_col] = np.nan

        for (animal, label), idx in out.groupby(["animal_id", "label"], dropna=False).groups.items():
            idx = pd.Index(idx)
            group_raw = pd.to_numeric(out.loc[idx, feature], errors="coerce")
            group_vals = pd.Series(degradation_value(feature, group_raw), index=idx, dtype=float)
            pre_idx = idx[out.loc[idx, "pre_post"].astype(str).to_numpy() == "pre"]
            pre_vals = finite_array(group_vals.loc[pre_idx].to_numpy(dtype=float)) if len(pre_idx) else np.array([])
            n_pre = int(pre_vals.size)

            if baseline_stat == "mean":
                baseline = float(np.nanmean(pre_vals)) if n_pre > 0 else np.nan
            else:
                baseline = float(np.nanmedian(pre_vals)) if n_pre > 0 else np.nan

            usable = bool(n_pre >= int(min_pre_n) and np.isfinite(baseline) and abs(baseline) > float(epsilon))
            if usable:
                out.loc[idx, norm_col] = group_vals / baseline

            post_idx = idx[out.loc[idx, "pre_post"].astype(str).to_numpy() == "post"]
            post_vals = finite_array(group_vals.loc[post_idx].to_numpy(dtype=float)) if len(post_idx) else np.array([])
            post_med = float(np.nanmedian(post_vals)) if post_vals.size else np.nan
            pre_med = float(np.nanmedian(pre_vals)) if pre_vals.size else np.nan
            info_rows.append({
                "animal_id": animal,
                "label": label,
                "feature": feature,
                "normalized_feature": norm_col,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "baseline_stat": baseline_stat,
                "pre_baseline_value": baseline,
                "pre_baseline_epsilon": float(epsilon),
                "min_pre_n_required": int(min_pre_n),
                "n_pre_for_baseline": n_pre,
                "n_post_values": int(post_vals.size),
                "normalization_status": "ok" if usable else "too_few_pre_or_near_zero_baseline",
                "raw_median_pre_analysis_value": pre_med,
                "raw_median_post_analysis_value": post_med,
                "raw_post_over_pre_median_ratio": (
                    post_med / pre_med if np.isfinite(post_med) and np.isfinite(pre_med) and abs(pre_med) > float(epsilon) else np.nan
                ),
                "post_over_pre_baseline_ratio": (
                    post_med / baseline if usable and np.isfinite(post_med) else np.nan
                ),
                "post_percent_change_from_pre_baseline": (
                    100.0 * ((post_med / baseline) - 1.0) if usable and np.isfinite(post_med) else np.nan
                ),
            })

    info_df = pd.DataFrame(info_rows)
    return out, norm_features, info_df

def compute_pre_post_distribution_tests(df: pd.DataFrame, features: List[str], min_n: int = 5) -> pd.DataFrame:
    """Compare pre- vs post-lesion feature distributions for each animal x label.

    This is complementary to the slope analysis. The slope analysis asks whether
    the phrase-position trend changes after lesion. This distribution analysis
    asks whether the overall pre/post segment-value distributions differ.

    For corr_to_phrase_early_template, values are converted to degradation = 1 - corr
    before testing so positive post-minus-pre values always mean "more degradation/change
    after lesion". For distance/acoustic features, raw values are used.
    """
    rows = []
    if df.empty:
        return pd.DataFrame(rows)

    for (animal, label), sub_label in df.groupby(["animal_id", "label"], dropna=False):
        pre = sub_label[sub_label["pre_post"] == "pre"]
        post = sub_label[sub_label["pre_post"] == "post"]
        if pre.empty or post.empty:
            continue
        for feature in features:
            if feature not in sub_label.columns:
                continue
            pre_raw = pd.to_numeric(pre[feature], errors="coerce")
            post_raw = pd.to_numeric(post[feature], errors="coerce")
            pre_vals = degradation_value(feature, pre_raw)
            post_vals = degradation_value(feature, post_raw)
            n_pre = int(np.isfinite(pre_vals).sum())
            n_post = int(np.isfinite(post_vals).sum())
            pre_med = safe_nanmedian(pre_vals)
            post_med = safe_nanmedian(post_vals)
            pre_mean = safe_nanmean(pre_vals)
            post_mean = safe_nanmean(post_vals)
            ok = n_pre >= min_n and n_post >= min_n
            mw_gt_p, mw_u = mannwhitney_pre_post(pre_vals, post_vals, alternative="greater") if ok else (np.nan, np.nan)
            mw_two_p, _ = mannwhitney_pre_post(pre_vals, post_vals, alternative="two-sided") if ok else (np.nan, np.nan)
            mw_less_p, _ = mannwhitney_pre_post(pre_vals, post_vals, alternative="less") if ok else (np.nan, np.nan)
            rows.append({
                "animal_id": animal,
                "label": label,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "analysis_value_label": degradation_axis_label(feature),
                "n_pre": n_pre,
                "n_post": n_post,
                "min_n_required": int(min_n),
                "test_status": "ok" if ok else "too_few_values",
                "median_pre_degradation_value": pre_med,
                "median_post_degradation_value": post_med,
                "post_minus_pre_median_degradation_value": (
                    post_med - pre_med if np.isfinite(post_med) and np.isfinite(pre_med) else np.nan
                ),
                "mean_pre_degradation_value": pre_mean,
                "mean_post_degradation_value": post_mean,
                "post_minus_pre_mean_degradation_value": (
                    post_mean - pre_mean if np.isfinite(post_mean) and np.isfinite(pre_mean) else np.nan
                ),
                "mannwhitney_post_gt_pre_p": mw_gt_p,
                "mannwhitney_post_lt_pre_p": mw_less_p,
                "mannwhitney_twosided_p": mw_two_p,
                "mannwhitney_u_post_vs_pre": mw_u,
                "cliffs_delta_post_vs_pre": cliffs_delta_post_vs_pre(pre_vals, post_vals) if ok else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in [c for c in out.columns if c.endswith("_p")]:
            out[f"{col}_holm_within_label_tests"] = holm_adjust(out[col])
    return out


def summarize_pre_post_distributions(distribution_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pre/post distribution effects at the bird level and test across birds."""
    bird_rows = []
    stat_rows = []
    if distribution_df.empty:
        return pd.DataFrame(bird_rows), pd.DataFrame(stat_rows)

    ok_df = distribution_df[distribution_df["test_status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(bird_rows), pd.DataFrame(stat_rows)

    for feature, sub in ok_df.groupby("feature"):
        for animal, sb in sub.groupby("animal_id"):
            bird_rows.append({
                "animal_id": animal,
                "summary_type": "pre_post_distribution",
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "predictor": "none",
                "predictor_label": "none",
                "n_labels": int(sb["label"].nunique()),
                "median_pre_distribution_degradation_value": safe_nanmedian(sb["median_pre_degradation_value"]),
                "median_post_distribution_degradation_value": safe_nanmedian(sb["median_post_degradation_value"]),
                "median_post_minus_pre_distribution_degradation_value": safe_nanmedian(sb["post_minus_pre_median_degradation_value"]),
                "median_distribution_cliffs_delta_post_vs_pre": safe_nanmedian(sb["cliffs_delta_post_vs_pre"]),
                # Keep other summary columns present so this can concatenate with phrase_position_bird_summary.csv.
                "median_pre_degradation_slope": np.nan,
                "median_post_degradation_slope": np.nan,
                "median_post_minus_pre_degradation_slope": np.nan,
                "median_baseline_slope_change_abs": np.nan,
                "median_lesion_slope_change_abs": np.nan,
                "median_lesion_minus_baseline_slope_change_abs": np.nan,
                "median_tail_minus_shared_degradation_value": np.nan,
                "median_tail_minus_pre_degradation_value": np.nan,
                "median_max_pre_predictor": np.nan,
                "median_max_post_predictor": np.nan,
                "median_post_only_tail_extent": np.nan,
            })

        b = sub.groupby("animal_id").agg(
            post_minus_pre_distribution=("post_minus_pre_median_degradation_value", lambda x: safe_nanmedian(x)),
            pre_distribution=("median_pre_degradation_value", lambda x: safe_nanmedian(x)),
            post_distribution=("median_post_degradation_value", lambda x: safe_nanmedian(x)),
            cliffs_delta=("cliffs_delta_post_vs_pre", lambda x: safe_nanmedian(x)),
            n_labels=("label", "nunique"),
        ).reset_index()
        stat_rows.append({
            "summary_type": "pre_post_distribution",
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": "none",
            "predictor_label": "none",
            "n_birds": int(finite_array(b["post_minus_pre_distribution"]).size),
            "median_bird_pre_distribution_degradation_value": safe_nanmedian(b["pre_distribution"]),
            "median_bird_post_distribution_degradation_value": safe_nanmedian(b["post_distribution"]),
            "median_bird_post_minus_pre_distribution_degradation_value": safe_nanmedian(b["post_minus_pre_distribution"]),
            "wilcoxon_distribution_post_minus_pre_gt0_p": wilcoxon_1sample(b["post_minus_pre_distribution"], alternative="greater"),
            "wilcoxon_distribution_post_minus_pre_twosided_p": wilcoxon_1sample(b["post_minus_pre_distribution"], alternative="two-sided"),
            "sign_test_distribution_post_minus_pre_gt0_p": sign_test(b["post_minus_pre_distribution"], alternative="greater"),
            "median_bird_distribution_cliffs_delta_post_vs_pre": safe_nanmedian(b["cliffs_delta"]),
        })

    bird_df = pd.DataFrame(bird_rows)
    stat_df = add_holm_columns(pd.DataFrame(stat_rows))
    return bird_df, stat_df


def add_holm_columns(stat_df: pd.DataFrame) -> pd.DataFrame:
    """Add/refresh Holm-adjusted columns for every raw p-value column in a stats table."""
    if stat_df is None or stat_df.empty:
        return stat_df
    out = stat_df.copy()
    raw_p_cols = [c for c in out.columns if c.endswith("_p") and not c.endswith("_holm_p")]
    for col in raw_p_cols:
        out[f"{col}_holm"] = holm_adjust(out[col])
    return out


def summarize_birds(effects_df: pd.DataFrame, tail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bird_rows = []
    stat_rows = []

    # Bird-level slope summaries.
    if not effects_df.empty:
        for (feature, predictor), sub in effects_df.groupby(["feature", "predictor"]):
            for animal, sb in sub.groupby("animal_id"):
                bird_rows.append({
                    "animal_id": animal,
                    "summary_type": "slope_effect",
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
                    "n_labels": int(sb.shape[0]),
                    "median_pre_degradation_slope": safe_nanmedian(sb["pre_degradation_slope"]),
                    "median_post_degradation_slope": safe_nanmedian(sb["post_degradation_slope"]),
                    "median_post_minus_pre_degradation_slope": safe_nanmedian(sb["post_minus_pre_degradation_slope"]),
                    "median_baseline_slope_change_abs": safe_nanmedian(sb["baseline_slope_change_abs"]),
                    "median_lesion_slope_change_abs": safe_nanmedian(sb["lesion_slope_change_abs"]),
                    "median_lesion_minus_baseline_slope_change_abs": safe_nanmedian(sb["lesion_minus_baseline_slope_change_abs"]),
                    "median_tail_minus_shared_degradation_value": np.nan,
                    "median_tail_minus_pre_degradation_value": np.nan,
                    "median_max_pre_predictor": np.nan,
                    "median_max_post_predictor": np.nan,
                    "median_post_only_tail_extent": np.nan,
                })

    # Bird-level tail summaries.
    if not tail_df.empty:
        for (feature, predictor), sub in tail_df.groupby(["feature", "predictor"]):
            for animal, sb in sub.groupby("animal_id"):
                st = sb[sb["has_tail"]].copy()
                bird_rows.append({
                    "animal_id": animal,
                    "summary_type": "post_only_tail_effect",
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
                    "n_labels": int(sb.shape[0]),
                    "n_labels_with_tail": int(st.shape[0]),
                    "median_pre_degradation_slope": np.nan,
                    "median_post_degradation_slope": np.nan,
                    "median_post_minus_pre_degradation_slope": np.nan,
                    "median_baseline_slope_change_abs": np.nan,
                    "median_lesion_slope_change_abs": np.nan,
                    "median_lesion_minus_baseline_slope_change_abs": np.nan,
                    "median_tail_minus_shared_degradation_value": safe_nanmedian(st["tail_minus_shared_degradation_value"]),
                    "median_tail_minus_pre_degradation_value": safe_nanmedian(st["tail_minus_pre_degradation_value"]),
                    "median_max_pre_predictor": safe_nanmedian(sb["max_pre_predictor"]),
                    "median_max_post_predictor": safe_nanmedian(sb["max_post_predictor"]),
                    "median_post_only_tail_extent": safe_nanmedian(sb["post_only_tail_extent"]),
                })

    bird_df = pd.DataFrame(bird_rows)

    # Statistical tests across birds.
    for (feature, predictor), sub in effects_df.groupby(["feature", "predictor"]) if not effects_df.empty else []:
        # Median per bird prevents birds with many labels from dominating.
        b = sub.groupby("animal_id").agg(
            post_minus_pre=("post_minus_pre_degradation_slope", lambda x: safe_nanmedian(x)),
            lesion_minus_baseline=("lesion_minus_baseline_slope_change_abs", lambda x: safe_nanmedian(x)),
            pre_slope=("pre_degradation_slope", lambda x: safe_nanmedian(x)),
            post_slope=("post_degradation_slope", lambda x: safe_nanmedian(x)),
            n_labels=("label", "nunique"),
        ).reset_index()
        stat_rows.append({
            "summary_type": "slope_effect",
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
            "n_birds": int(finite_array(b["post_minus_pre"]).size),
            "median_bird_post_minus_pre_degradation_slope": safe_nanmedian(b["post_minus_pre"]),
            "wilcoxon_post_minus_pre_gt0_p": wilcoxon_1sample(b["post_minus_pre"], alternative="greater"),
            "wilcoxon_post_minus_pre_twosided_p": wilcoxon_1sample(b["post_minus_pre"], alternative="two-sided"),
            "sign_test_post_minus_pre_gt0_p": sign_test(b["post_minus_pre"], alternative="greater"),
            "median_bird_lesion_minus_baseline_slope_change_abs": safe_nanmedian(b["lesion_minus_baseline"]),
            "wilcoxon_lesion_minus_baseline_gt0_p": wilcoxon_1sample(b["lesion_minus_baseline"], alternative="greater"),
            "wilcoxon_lesion_minus_baseline_twosided_p": wilcoxon_1sample(b["lesion_minus_baseline"], alternative="two-sided"),
            "sign_test_lesion_minus_baseline_gt0_p": sign_test(b["lesion_minus_baseline"], alternative="greater"),
        })

    for (feature, predictor), sub in tail_df.groupby(["feature", "predictor"]) if not tail_df.empty else []:
        st = sub[sub["has_tail"]].copy()
        b = st.groupby("animal_id").agg(
            tail_minus_shared=("tail_minus_shared_degradation_value", lambda x: safe_nanmedian(x)),
            tail_minus_pre=("tail_minus_pre_degradation_value", lambda x: safe_nanmedian(x)),
            n_labels_with_tail=("label", "nunique"),
            max_post=("max_post_predictor", lambda x: safe_nanmedian(x)),
            max_pre=("max_pre_predictor", lambda x: safe_nanmedian(x)),
            tail_extent=("post_only_tail_extent", lambda x: safe_nanmedian(x)),
        ).reset_index()
        stat_rows.append({
            "summary_type": "post_only_tail_effect",
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
            "n_birds": int(finite_array(b["tail_minus_shared"]).size),
            "median_bird_tail_minus_shared_degradation_value": safe_nanmedian(b["tail_minus_shared"]),
            "wilcoxon_tail_minus_shared_gt0_p": wilcoxon_1sample(b["tail_minus_shared"], alternative="greater"),
            "wilcoxon_tail_minus_shared_twosided_p": wilcoxon_1sample(b["tail_minus_shared"], alternative="two-sided"),
            "sign_test_tail_minus_shared_gt0_p": sign_test(b["tail_minus_shared"], alternative="greater"),
            "median_bird_tail_minus_pre_degradation_value": safe_nanmedian(b["tail_minus_pre"]),
            "wilcoxon_tail_minus_pre_gt0_p": wilcoxon_1sample(b["tail_minus_pre"], alternative="greater"),
            "wilcoxon_tail_minus_pre_twosided_p": wilcoxon_1sample(b["tail_minus_pre"], alternative="two-sided"),
            "sign_test_tail_minus_pre_gt0_p": sign_test(b["tail_minus_pre"], alternative="greater"),
        })

    stat_df = add_holm_columns(pd.DataFrame(stat_rows))
    return bird_df, stat_df


def summarize_slopes_against_zero(slopes_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize whether phrase-position slopes are different from zero.

    Label-level slopes are first reduced to one median slope per bird for each
    feature x predictor x group. The bird medians are then tested against zero.

    For corr_to_phrase_early_template, compute_label_slopes already converted
    the raw slope to degradation_slope = -1 * raw slope. For all other features,
    degradation_slope is the raw feature slope. Therefore, positive values mean
    that the analysis value increases with phrase position.
    """
    bird_rows = []
    stat_rows = []
    if slopes_df is None or slopes_df.empty:
        return pd.DataFrame(bird_rows), pd.DataFrame(stat_rows)

    use = slopes_df.copy()
    use = use[use["fit_status"].astype(str) == "ok"].copy()
    use["degradation_slope"] = pd.to_numeric(use["degradation_slope"], errors="coerce")
    use = use[np.isfinite(use["degradation_slope"])].copy()
    if use.empty:
        return pd.DataFrame(bird_rows), pd.DataFrame(stat_rows)

    group_cols = ["feature", "predictor", "group_col", "group"]
    for (feature, predictor, group_col, group), sub in use.groupby(group_cols, dropna=False):
        # One value per bird, median across syllable labels.
        b = sub.groupby("animal_id").agg(
            median_slope=("degradation_slope", lambda x: safe_nanmedian(x)),
            n_labels=("label", "nunique"),
            n_label_slopes=("degradation_slope", lambda x: int(finite_array(x).size)),
        ).reset_index()
        for _, row in b.iterrows():
            bird_rows.append({
                "summary_type": "slope_against_zero",
                "animal_id": row["animal_id"],
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
                "group_col": group_col,
                "group": group,
                "n_labels": int(row["n_labels"]),
                "n_label_slopes": int(row["n_label_slopes"]),
                "median_slope": row["median_slope"],
            })

        vals = b["median_slope"].to_numpy(dtype=float)
        stat_rows.append({
            "summary_type": "slope_against_zero",
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "predictor_label": PREDICTOR_LABELS.get(predictor, predictor),
            "group_col": group_col,
            "group": group,
            "n_birds": int(finite_array(vals).size),
            "median_bird_slope": safe_nanmedian(vals),
            "wilcoxon_slope_gt0_p": wilcoxon_1sample(vals, alternative="greater"),
            "wilcoxon_slope_lt0_p": wilcoxon_1sample(vals, alternative="less"),
            "wilcoxon_slope_twosided_p": wilcoxon_1sample(vals, alternative="two-sided"),
            "sign_test_slope_gt0_p": sign_test(vals, alternative="greater"),
            "sign_test_slope_lt0_p": sign_test(vals, alternative="less"),
            "sign_test_slope_twosided_p": sign_test(vals, alternative="two-sided"),
        })

    return pd.DataFrame(bird_rows), add_holm_columns(pd.DataFrame(stat_rows))


def slope_zero_stat_subtitle(stat_df: pd.DataFrame, feature: str, predictor: str) -> str:
    """Return compact p-value text for slope > 0 tests in pre and post groups."""
    if stat_df is None or stat_df.empty:
        return ""
    sub = stat_df[
        (stat_df["summary_type"] == "slope_against_zero") &
        (stat_df["feature"] == feature) &
        (stat_df["predictor"] == predictor) &
        (stat_df["group_col"] == "pre_post") &
        (stat_df["group"].isin(["pre", "post"]))
    ].copy()
    if sub.empty:
        return ""
    pieces = []
    for group in ["pre", "post"]:
        r = sub[sub["group"] == group]
        if r.empty:
            continue
        r0 = r.iloc[0]
        med = r0.get("median_bird_slope")
        pieces.append(f"{group} slope>0 p={fmt_p(r0.get('wilcoxon_slope_gt0_p'))}, median={med:.3g}" if np.isfinite(med) else f"{group} slope>0 p={fmt_p(r0.get('wilcoxon_slope_gt0_p'))}")
    return "linear relation: " + "; ".join(pieces) if pieces else ""



def fmt_p(p) -> str:
    try:
        p = float(p)
    except Exception:
        return "NA"
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def stat_subtitle(stat_df: pd.DataFrame, feature: str, predictor: str, summary_type: str) -> str:
    if stat_df is None or stat_df.empty:
        return ""
    sub = stat_df[(stat_df["feature"] == feature) & (stat_df["predictor"] == predictor) & (stat_df["summary_type"] == summary_type)]
    if sub.empty:
        return ""
    r = sub.iloc[0]
    if summary_type == "slope_effect":
        return (
            f"bird-level Wilcoxon post>pre p={fmt_p(r.get('wilcoxon_post_minus_pre_gt0_p'))}; "
            f"lesion>baseline p={fmt_p(r.get('wilcoxon_lesion_minus_baseline_gt0_p'))}"
        )
    if summary_type == "pre_post_distribution":
        return (
            f"bird-level Wilcoxon post>pre p={fmt_p(r.get('wilcoxon_distribution_post_minus_pre_gt0_p'))}; "
            f"two-sided p={fmt_p(r.get('wilcoxon_distribution_post_minus_pre_twosided_p'))}"
        )
    return (
        f"bird-level Wilcoxon tail>shared p={fmt_p(r.get('wilcoxon_tail_minus_shared_gt0_p'))}; "
        f"tail>pre p={fmt_p(r.get('wilcoxon_tail_minus_pre_gt0_p'))}"
    )


def distribution_stat_subtitle(stat_df: pd.DataFrame, feature: str) -> str:
    return stat_subtitle(stat_df, feature, "none", "pre_post_distribution")


def add_pvalue_note(ax, text: str):
    if not text:
        return
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
    )


def add_zero_line(ax):
    ax.axhline(0, linestyle="--", linewidth=1.2, alpha=0.8)


def plot_post_minus_pre_by_bird(effects_df, bird_df, stat_df, feature, predictor, out_png, title_prefix):
    sub_l = effects_df[(effects_df["feature"] == feature) & (effects_df["predictor"] == predictor)].copy()
    sub_b = bird_df[(bird_df["summary_type"] == "slope_effect") & (bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    sub_b = sub_b.sort_values("animal_id")
    birds = sub_b["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for _, row in sub_l.iterrows():
        y = row["post_minus_pre_degradation_slope"]
        if not np.isfinite(y):
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature, predictor)) % 1000) / 1000.0 - 0.5) * 0.22
        ax.scatter(xmap[row["animal_id"]] + jitter, y, s=38, alpha=0.35)
    xs = np.arange(len(birds))
    ys = sub_b["median_post_minus_pre_degradation_slope"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=115, marker="D", edgecolors="black", label="Bird median")
    add_zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ax.set_ylabel(f"Post - pre degradation slope ({PREDICTOR_UNITS[predictor]})")
    ax.set_title(f"{title_prefix}\n{FEATURE_LABELS.get(feature, feature)}\n{stat_subtitle(stat_df, feature, predictor, 'slope_effect')}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_figure_with_margins(fig, out_png, left=0.20, right=0.98, top=0.86, bottom=0.16)


def plot_pre_vs_post_paired_birds(bird_df, stat_df, feature, predictor, out_png, title_prefix):
    sub = bird_df[(bird_df["summary_type"] == "slope_effect") & (bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("animal_id")
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    for _, row in sub.iterrows():
        y0 = row["median_pre_degradation_slope"]
        y1 = row["median_post_degradation_slope"]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        ax.plot([0, 1], [y0, y1], marker="o", alpha=0.8)
        ax.text(1.03, y1, row["animal_id"], fontsize=8, va="center")
    ax.set_xlim(-0.15, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(f"Bird median degradation slope ({PREDICTOR_UNITS[predictor]})")
    ax.set_title(f"{title_prefix}\nBird-level paired pre vs post\n{FEATURE_LABELS.get(feature, feature)}\n{stat_subtitle(stat_df, feature, predictor, 'slope_effect')}")
    add_zero_line(ax)
    ax.grid(axis="y", alpha=0.25)
    save_figure_with_margins(fig, out_png, left=0.20, right=0.98, top=0.86, bottom=0.16)


def plot_baseline_vs_lesion(effects_df, bird_df, stat_df, feature, predictor, out_png, title_prefix):
    sub_l = effects_df[(effects_df["feature"] == feature) & (effects_df["predictor"] == predictor)].copy()
    sub_b = bird_df[(bird_df["summary_type"] == "slope_effect") & (bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    birds = sorted(sub_l["animal_id"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    b2c = {b: cmap(i % 10) for i, b in enumerate(birds)}
    vals = []
    for _, row in sub_l.iterrows():
        x = row["baseline_slope_change_abs"]
        y = row["lesion_slope_change_abs"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        vals += [float(x), float(y)]
        ax.scatter(x, y, s=35, alpha=0.35, color=b2c[row["animal_id"]])
    for _, row in sub_b.iterrows():
        x = row["median_baseline_slope_change_abs"]
        y = row["median_lesion_slope_change_abs"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        vals += [float(x), float(y)]
        ax.scatter(x, y, s=120, marker="D", edgecolors="black", color=b2c.get(row["animal_id"], "gray"))
        ax.text(x, y, row["animal_id"], fontsize=8, ha="left", va="bottom")
    if not vals:
        plt.close(fig)
        return
    lo = 0.0
    hi = max(vals) * 1.08 if max(vals) > 0 else 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="equal change")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Baseline drift |late pre - early pre| ({PREDICTOR_UNITS[predictor]})")
    ax.set_ylabel(f"Lesion-associated shift |post - late pre| ({PREDICTOR_UNITS[predictor]})")
    ax.set_title(f"{title_prefix}\nBaseline drift vs lesion-associated shift\n{FEATURE_LABELS.get(feature, feature)}\n{stat_subtitle(stat_df, feature, predictor, 'slope_effect')}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    save_figure_with_margins(fig, out_png, left=0.20, right=0.98, top=0.86, bottom=0.16)


def plot_tail_effect_by_bird(tail_df, bird_df, stat_df, feature, predictor, out_png, title_prefix):
    sub_l = tail_df[(tail_df["feature"] == feature) & (tail_df["predictor"] == predictor) & (tail_df["has_tail"])].copy()
    sub_b = bird_df[(bird_df["summary_type"] == "post_only_tail_effect") & (bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    sub_b = sub_b.sort_values("animal_id")
    birds = sub_b["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for _, row in sub_l.iterrows():
        y = row["tail_minus_shared_degradation_value"]
        if not np.isfinite(y):
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature, predictor, "tail")) % 1000) / 1000.0 - 0.5) * 0.22
        ax.scatter(xmap[row["animal_id"]] + jitter, y, s=38, alpha=0.35)
    xs = np.arange(len(birds))
    ys = sub_b["median_tail_minus_shared_degradation_value"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=115, marker="D", edgecolors="black", label="Bird median")
    add_zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ax.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "tail"), width=28))
    ax.set_title(f"{title_prefix}\nPost-only stutter-tail effect\n{FEATURE_LABELS.get(feature, feature)}\n{stat_subtitle(stat_df, feature, predictor, 'post_only_tail_effect')}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_figure_with_margins(fig, out_png, left=0.21, right=0.98, top=0.84, bottom=0.16)


def plot_pre_post_distribution_effect_by_bird(distribution_df, distribution_bird_df, stat_df, feature, out_png, title_prefix):
    """Plot animal x label post-pre distribution shifts, with bird medians and bird-level p-value."""
    sub_l = distribution_df[(distribution_df["feature"] == feature) & (distribution_df["test_status"] == "ok")].copy()
    sub_b = distribution_bird_df[(distribution_bird_df["summary_type"] == "pre_post_distribution") & (distribution_bird_df["feature"] == feature)].copy()
    if sub_l.empty or sub_b.empty:
        return
    sub_b = sub_b.sort_values("animal_id")
    birds = sub_b["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for _, row in sub_l.iterrows():
        y = row["post_minus_pre_median_degradation_value"]
        if not np.isfinite(y):
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature, "distribution")) % 1000) / 1000.0 - 0.5) * 0.22
        ax.scatter(xmap[row["animal_id"]] + jitter, y, s=38, alpha=0.35)
    xs = np.arange(len(birds))
    ys = sub_b["median_post_minus_pre_distribution_degradation_value"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=115, marker="D", edgecolors="black", label="Bird median")
    add_zero_line(ax)
    ptxt = distribution_stat_subtitle(stat_df, feature)
    add_pvalue_note(ax, ptxt)
    ax.set_xticks(xs)
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ax.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "delta"), width=28))
    ax.set_title(f"{title_prefix}\nPre vs post lesion distributions by bird\n{degradation_axis_label(feature)}\n{ptxt}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_figure_with_margins(fig, out_png, left=0.21, right=0.98, top=0.84, bottom=0.16)


def plot_pre_vs_post_distribution_paired_birds(distribution_bird_df, stat_df, feature, out_png, title_prefix):
    """Plot paired bird-median pre and post distribution values."""
    sub = distribution_bird_df[(distribution_bird_df["summary_type"] == "pre_post_distribution") & (distribution_bird_df["feature"] == feature)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("animal_id")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for _, row in sub.iterrows():
        y0 = row["median_pre_distribution_degradation_value"]
        y1 = row["median_post_distribution_degradation_value"]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        ax.plot([0, 1], [y0, y1], marker="o", alpha=0.8)
        ax.text(1.03, y1, row["animal_id"], fontsize=8, va="center")
    ptxt = distribution_stat_subtitle(stat_df, feature)
    add_pvalue_note(ax, ptxt)
    ax.set_xlim(-0.15, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "paired"), width=28))
    ax.set_title(f"{title_prefix}\nBird-level paired pre vs post distributions\n{degradation_axis_label(feature)}\n{ptxt}")
    ax.grid(axis="y", alpha=0.25)
    save_figure_with_margins(fig, out_png, left=0.23, right=0.98, top=0.84, bottom=0.16)


def plot_normalized_feature_summary(
    distribution_df, distribution_bird_df, tail_df, bird_df, stat_df, feature, out_png, title_prefix
):
    """Create a 2x2 summary figure for a normalized feature.

    Panels:
      A. Bird-level paired pre vs post distributions
      B. Post-minus-pre distribution effect by bird
      C. Post-only tail effect by bird (elapsed time)
      D. Post-only tail effect by bird (previous repeats)
    """
    if not is_pre_normalized_feature(feature):
        return

    sub_dist = distribution_df[(distribution_df["feature"] == feature) & (distribution_df["test_status"] == "ok")].copy()
    sub_dist_b = distribution_bird_df[(distribution_bird_df["summary_type"] == "pre_post_distribution") & (distribution_bird_df["feature"] == feature)].copy()
    if sub_dist_b.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: paired bird-level pre vs post distributions.
    sub = sub_dist_b.sort_values("animal_id")
    for _, row in sub.iterrows():
        y0 = row["median_pre_distribution_degradation_value"]
        y1 = row["median_post_distribution_degradation_value"]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        axA.plot([0, 1], [y0, y1], marker="o", alpha=0.8)
        axA.text(1.03, y1, row["animal_id"], fontsize=8, va="center")
    ptxt = distribution_stat_subtitle(stat_df, feature)
    add_pvalue_note(axA, ptxt)
    axA.set_xlim(-0.15, 1.35)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["Pre", "Post"])
    axA.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "paired"), width=24))
    axA.set_title("A. Bird-level paired pre vs post")
    axA.grid(axis="y", alpha=0.25)

    # Panel B: post-pre distribution effect by bird.
    sub_b = sub_dist_b.sort_values("animal_id")
    birds = sub_b["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    for _, row in sub_dist.iterrows():
        y = row["post_minus_pre_median_degradation_value"]
        if not np.isfinite(y):
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature, "summary_distribution")) % 1000) / 1000.0 - 0.5) * 0.22
        axB.scatter(xmap[row["animal_id"]] + jitter, y, s=32, alpha=0.35)
    xs = np.arange(len(birds))
    ys = sub_b["median_post_minus_pre_distribution_degradation_value"].to_numpy(dtype=float)
    axB.scatter(xs, ys, s=100, marker="D", edgecolors="black", label="Bird median")
    add_zero_line(axB)
    add_pvalue_note(axB, ptxt)
    axB.set_xticks(xs)
    axB.set_xticklabels(birds, rotation=30, ha="right")
    axB.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "delta"), width=24))
    axB.set_title("B. Post - pre effect by bird")
    axB.grid(axis="y", alpha=0.25)
    axB.legend(frameon=True, fontsize=9)

    # Panels C and D: tail effect by bird for both predictors.
    def draw_tail_panel(ax, predictor, panel_letter):
        sub_l = tail_df[(tail_df["feature"] == feature) & (tail_df["predictor"] == predictor) & (tail_df["has_tail"])].copy()
        sub_bird = bird_df[(bird_df["summary_type"] == "post_only_tail_effect") & (bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
        if sub_l.empty or sub_bird.empty:
            ax.text(0.5, 0.5, "No usable tail data", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return
        sub_bird = sub_bird.sort_values("animal_id")
        birds_local = sub_bird["animal_id"].tolist()
        xmap_local = {b: i for i, b in enumerate(birds_local)}
        for _, row in sub_l.iterrows():
            y = row["tail_minus_shared_degradation_value"]
            if not np.isfinite(y):
                continue
            jitter = ((hash((row["animal_id"], str(row["label"]), feature, predictor, "summary_tail")) % 1000) / 1000.0 - 0.5) * 0.22
            ax.scatter(xmap_local[row["animal_id"]] + jitter, y, s=32, alpha=0.35)
        xs_local = np.arange(len(birds_local))
        ys_local = sub_bird["median_tail_minus_shared_degradation_value"].to_numpy(dtype=float)
        ax.scatter(xs_local, ys_local, s=100, marker="D", edgecolors="black", label="Bird median")
        add_zero_line(ax)
        add_pvalue_note(ax, stat_subtitle(stat_df, feature, predictor, "post_only_tail_effect"))
        ax.set_xticks(xs_local)
        ax.set_xticklabels(birds_local, rotation=30, ha="right")
        ax.set_ylabel(wrap_axis_label(distribution_ylabel(feature, "tail"), width=24))
        ax.set_title(f"{panel_letter}. Post-only tail effect ({PREDICTOR_LABELS[predictor]})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=True, fontsize=9)

    draw_tail_panel(axC, "elapsed_time_in_phrase_s", "C")
    draw_tail_panel(axD, "n_previous_segments", "D")

    fig.suptitle(
        f"{title_prefix}\nSummary of normalized pre/post and tail effects\n{FEATURE_LABELS.get(feature, feature)}",
        fontsize=16, y=0.985
    )
    save_figure_with_margins(fig, out_png, left=0.14, right=0.98, top=0.83, bottom=0.10, wspace=0.34, hspace=0.42)


def plot_max_extent_by_bird(tail_df, predictor, out_png, title_prefix):
    sub = tail_df[tail_df["predictor"] == predictor].copy()
    if sub.empty:
        return
    # One row per animal x label x predictor is enough; feature repeats duplicate extent rows.
    sub = sub.drop_duplicates(["animal_id", "label", "predictor"])
    bird = sub.groupby("animal_id").agg(
        median_max_pre=("max_pre_predictor", lambda x: safe_nanmedian(x)),
        median_max_post=("max_post_predictor", lambda x: safe_nanmedian(x)),
        median_tail_extent=("post_only_tail_extent", lambda x: safe_nanmedian(x)),
    ).reset_index().sort_values("animal_id")
    birds = bird["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for _, row in sub.iterrows():
        i = xmap[row["animal_id"]]
        ax.plot([i - 0.12, i + 0.12], [row["max_pre_predictor"], row["max_post_predictor"]], color="0.75", alpha=0.45)
        ax.scatter(i - 0.12, row["max_pre_predictor"], s=30, alpha=0.30, label="label pre max" if _ == sub.index[0] else None)
        ax.scatter(i + 0.12, row["max_post_predictor"], s=30, alpha=0.30, label="label post max" if _ == sub.index[0] else None)
    xs = np.arange(len(birds))
    ax.scatter(xs - 0.18, bird["median_max_pre"], s=110, marker="D", edgecolors="black", label="bird median pre max")
    ax.scatter(xs + 0.18, bird["median_max_post"], s=110, marker="D", edgecolors="black", label="bird median post max")
    ax.set_xticks(xs)
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ax.set_ylabel(PREDICTOR_LABELS[predictor])
    ax.set_title(f"{title_prefix}\nMaximum phrase extent by bird")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def binned_median(x: np.ndarray, y: np.ndarray, bins: np.ndarray, min_n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        vals = y[mask]
        if vals.size < min_n:
            continue
        rows.append({
            "x_mid": float((lo + hi) / 2),
            "n": int(vals.size),
            "median": float(np.nanmedian(vals)),
            "q25": float(np.nanpercentile(vals, 25)),
            "q75": float(np.nanpercentile(vals, 75)),
        })
    return pd.DataFrame(rows)


def plot_aggregate_binned_position(df, feature, predictor, out_png, title_prefix, n_bins=40, max_x=None, min_bin_n=10, stat_df=None, slope_zero_stat_df=None):
    if feature not in df.columns or predictor not in df.columns:
        return
    use = df.copy()
    use[predictor] = pd.to_numeric(use[predictor], errors="coerce")
    use[feature] = pd.to_numeric(use[feature], errors="coerce")
    valid = np.isfinite(use[predictor]) & np.isfinite(use[feature])
    if max_x is not None:
        valid &= use[predictor] <= max_x
    use = use[valid].copy()
    if use.empty:
        return
    xmax = np.nanmax(use[predictor]) if max_x is None else max_x
    xmin = np.nanmin(use[predictor])
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return
    bins = np.linspace(xmin, xmax, n_bins + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for pp in ["pre", "post"]:
        sub = use[use["pre_post"] == pp]
        b = binned_median(sub[predictor].to_numpy(float), sub[feature].to_numpy(float), bins, min_n=min_bin_n)
        if b.empty:
            continue
        ax.plot(b["x_mid"], b["median"], linewidth=2.2, marker="o", markersize=3.5, label=f"{pp} median")
        ax.fill_between(b["x_mid"], b["q25"], b["q75"], alpha=0.16, label=f"{pp} IQR")
    # Mark median max pre/post ranges across labels. This helps show the shared vs stutter-tail region.
    max_pre_by_label = use[use["pre_post"] == "pre"].groupby(["animal_id", "label"])[predictor].max()
    max_post_by_label = use[use["pre_post"] == "post"].groupby(["animal_id", "label"])[predictor].max()
    med_max_pre = safe_nanmedian(max_pre_by_label)
    med_max_post = safe_nanmedian(max_post_by_label)
    if np.isfinite(med_max_pre):
        ax.axvline(med_max_pre, linestyle="--", linewidth=1.2, label="median max pre range")
    if np.isfinite(med_max_post):
        ax.axvline(med_max_post, linestyle=":", linewidth=1.2, label="median max post range")
    ax.set_xlabel(PREDICTOR_LABELS[predictor])
    ax.set_ylabel(wrap_axis_label(FEATURE_LABELS.get(feature, feature), width=38))
    dist_txt = distribution_stat_subtitle(stat_df, feature) if stat_df is not None else ""
    slope_txt = slope_zero_stat_subtitle(slope_zero_stat_df, feature, predictor) if slope_zero_stat_df is not None else ""
    note_lines = [x for x in [dist_txt, slope_txt] if x]
    ptxt = "\n".join(note_lines)
    add_pvalue_note(ax, ptxt)
    title_lines = [title_prefix, "Aggregate binned median ± IQR", FEATURE_LABELS.get(feature, feature)]
    if ptxt:
        title_lines.append(ptxt)
    ax.set_title("\n".join(title_lines))
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8)
    save_figure_with_margins(fig, out_png, left=0.20, right=0.98, top=0.82, bottom=0.14)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate phrase-position acoustic/spectrogram effects across birds.")
    p.add_argument("--phrase-position-root", required=True,
                   help="Root folder containing outputs from phrase_position_acoustic_crosscorr_v6_majority_vote_runs.py")
    p.add_argument("--out-dir", default=None,
                   help="Output folder. Default: <phrase-position-root>/aggregate_bird_effects_v7_pre_normalized")
    p.add_argument("--metadata-excel-path", default=None,
                   help="Metadata Excel path for Medial + Lateral filtering")
    p.add_argument("--lesion-filter", choices=["medial_lateral", "all"], default="medial_lateral")
    p.add_argument("--lesion-metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--lesion-animal-col", default=None)
    p.add_argument("--lesion-hit-type-col", default=None)
    p.add_argument("--animal-ids", default=None,
                   help="Optional comma-separated animal IDs to include after lesion filtering")
    p.add_argument("--high-variance-labels-csv", default=None,
                   help=(
                       "Optional CSV/TSV/Excel table identifying highly variable repeat-time syllables. "
                       "When provided, the analysis keeps only matching animal x label pairs before calculating slopes, "
                       "tail effects, pre/post distribution tests, and plots."
                   ))
    p.add_argument("--high-variance-animal-col", default=None,
                   help="Animal/bird ID column in --high-variance-labels-csv. Auto-detected if omitted.")
    p.add_argument("--high-variance-label-col", default=None,
                   help="Syllable/cluster label column in --high-variance-labels-csv. Auto-detected if omitted.")
    p.add_argument("--high-variance-select-col", default=None,
                   help="Optional boolean/include column in the high-variance CSV. If omitted, common names are auto-detected.")
    p.add_argument("--high-variance-rank-col", default=None,
                   help="Optional rank column. If provided/found, keeps the top fraction within each bird by rank.")
    p.add_argument("--high-variance-rank-direction", choices=["ascending", "descending"], default="ascending",
                   help="Rank direction. Use ascending when rank 1 is most variable. Default: ascending.")
    p.add_argument("--high-variance-metric-col", default=None,
                   help="Optional variability metric column. If provided/found, keeps the top fraction within each bird by this metric.")
    p.add_argument("--high-variance-metric-direction", choices=["largest", "smallest"], default="largest",
                   help="Metric direction for selecting variable syllables. Default keeps largest values.")
    p.add_argument("--high-variance-metric-agg", choices=["max", "median", "mean", "min"], default="max",
                   help="How to collapse duplicate animal x label metric rows before ranking. Default: max.")
    p.add_argument("--high-variance-top-fraction", type=float, default=0.30,
                   help="Fraction of labels to keep within each bird when using a metric/rank column. Default: 0.30.")
    p.add_argument("--high-variance-use-all-rows", action="store_true",
                   help=(
                       "Use every animal x label row in --high-variance-labels-csv as the selected high-variance set. "
                       "Use this when the CSV is already restricted to the top 30%% variable repeat-time syllables."
                   ))
    p.add_argument("--features", default="corr_to_phrase_early_template,distance_from_phrase_early_template",
                   help="Comma-separated features to aggregate")
    p.add_argument("--min-values-per-group", type=int, default=5,
                   help="Minimum segment values required to fit a slope")
    p.add_argument("--min-values-per-distribution", type=int, default=None,
                   help="Minimum finite pre and post values per animal x label for pre/post distribution tests. Default: --min-values-per-group")
    p.add_argument("--skip-pre-normalized-analysis", action="store_true",
                   help="Skip the additional analysis of feature values normalized to each animal x label pre-lesion baseline.")
    p.add_argument("--pre-normalization-stat", choices=["median", "mean"], default="median",
                   help="Pre-lesion baseline statistic used for normalization. Default: median.")
    p.add_argument("--pre-normalization-min-pre-values", type=int, default=None,
                   help="Minimum pre-lesion values required to compute a stable normalization baseline. Default: --min-values-per-distribution if set, else --min-values-per-group.")
    p.add_argument("--pre-normalization-epsilon", type=float, default=1e-9,
                   help="Minimum absolute baseline allowed for ratio normalization. Smaller/zero baselines are set to NaN. Default: 1e-9.")
    p.add_argument("--min-tail-values", type=int, default=5,
                   help="Minimum post-tail values required for tail summaries")
    p.add_argument("--max-plot-elapsed-s", type=float, default=None,
                   help="Optional cap for aggregate binned elapsed-time plots. Omit for dynamic max.")
    p.add_argument("--max-plot-previous-repeats", type=float, default=None,
                   help="Optional cap for aggregate binned repeat-count plots. Omit for dynamic max.")
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--min-bin-n", type=int, default=20)
    p.add_argument("--title-prefix", default="Medial + Lateral birds")
    p.add_argument("--save-combined-segments", action="store_true",
                   help="Save the combined segment-level table. This can be large.")
    return p.parse_args()


def write_readme(out_dir: Path):
    (out_dir / "README_phrase_position_bird_effects_v9.txt").write_text(
        "Aggregate phrase-position bird-level analysis.\n\n"
        "Core idea:\n"
        "  Each animal x syllable label gets a pre and post phrase-position slope.\n"
        "  Then each bird is summarized by the median across its selected labels.\n"
        "  This prevents birds/labels with many segmented syllables from dominating.\n\n"
        "Optional high-variance label filtering:\n"
        "  If --high-variance-labels-csv was used, all calculations are restricted to the\n"
        "  selected high repeat-time variability animal x syllable labels before any slopes,\n"
        "  tail effects, distribution tests, or plots are computed.\n"
        "  Check high_variance_label_filter_used.csv and high_variance_label_filter_match_summary_by_bird.csv.\n\n"
        "Pre-baseline-normalized analysis:\n"
        "  Unless --skip-pre-normalized-analysis is used, the script also creates normalized_* outputs.\n"
        "  Each feature is converted to the usual analysis scale, then divided by that animal x label's\n"
        "  pre-lesion baseline. With the default median baseline, pre values are centered near 1.\n"
        "  Post values > 1 mean the feature increased relative to that syllable's own baseline.\n"
        "  Check pre_baseline_normalization_values.csv and normalized_phrase_position_* outputs.\n\n"
        "Main interpretation columns:\n"
        "  post_minus_pre_degradation_slope:\n"
        "    > 0 means stronger post-lesion acoustic drift with phrase position.\n"
        "  lesion_minus_baseline_slope_change_abs:\n"
        "    > 0 means lesion-associated slope change exceeds early-pre to late-pre drift.\n"
        "  tail_minus_shared_degradation_value:\n"
        "    > 0 means the post-only stutter tail is more acoustically different than the shared post range.\n\n"
        "For corr_to_phrase_early_template, degradation is converted to 1 - corr.\n"
        "For distance_from_phrase_early_template, degradation is the distance itself.\n\n"
        "Most important files:\n"
        "  high_variance_label_filter_used.csv, if high-variance filtering was requested\n"
        "  high_variance_label_filter_match_summary_by_bird.csv, if high-variance filtering was requested\n"
        "  phrase_position_label_effects.csv\n"
        "  phrase_position_tail_effects.csv\n"
        "  phrase_position_pre_post_distribution_label_tests.csv\n"
        "  phrase_position_pre_post_distribution_bird_summary.csv\n"
        "  phrase_position_pre_post_distribution_bird_stats.csv\n"
        "  phrase_position_bird_summary.csv\n"
        "  phrase_position_bird_stats.csv\n"
    )


def main():
    args = parse_args()
    root = Path(args.phrase_position_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "aggregate_bird_effects_v7_pre_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_animals = None
    if args.lesion_filter == "medial_lateral":
        if args.metadata_excel_path is None:
            raise ValueError("--metadata-excel-path is required when --lesion-filter medial_lateral")
        allowed_animals, lesion_meta = load_medial_lateral_animals(
            metadata_excel_path=args.metadata_excel_path,
            sheet_name=args.lesion_metadata_sheet,
            animal_col=args.lesion_animal_col,
            hit_type_col=args.lesion_hit_type_col,
        )
        print(f"[INFO] Medial + Lateral animals from metadata: {sorted(allowed_animals)}")
        lesion_meta.to_csv(out_dir / "lesion_metadata_with_medial_lateral_filter.csv", index=False)
        print(f"[SAVED] {out_dir / 'lesion_metadata_with_medial_lateral_filter.csv'}")

    if args.animal_ids:
        requested = {x.strip() for x in args.animal_ids.split(",") if x.strip()}
        allowed_animals = requested if allowed_animals is None else (allowed_animals & requested)
        print(f"[INFO] Requested animals after lesion filtering: {sorted(allowed_animals)}")

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    segment_df, used_files_df = read_segment_feature_tables(root, allowed_animals=allowed_animals)
    print(f"[INFO] Loaded {len(segment_df):,} segment rows from {len(used_files_df):,} CSV files before optional high-variance label filtering")

    if args.high_variance_labels_csv:
        selected_pairs, high_var_labels_df, high_var_source_summary_df = load_high_variance_label_filter(
            high_variance_labels_csv=args.high_variance_labels_csv,
            animal_col=args.high_variance_animal_col,
            label_col=args.high_variance_label_col,
            select_col=args.high_variance_select_col,
            metric_col=args.high_variance_metric_col,
            rank_col=args.high_variance_rank_col,
            top_fraction=args.high_variance_top_fraction,
            metric_direction=args.high_variance_metric_direction,
            metric_agg=args.high_variance_metric_agg,
            rank_direction=args.high_variance_rank_direction,
            use_all_rows=args.high_variance_use_all_rows,
        )
        high_var_labels_df.to_csv(out_dir / "high_variance_label_filter_used.csv", index=False)
        high_var_source_summary_df.to_csv(out_dir / "high_variance_label_filter_source_summary_by_bird.csv", index=False)
        print(f"[INFO] High-variance label filter selected {len(selected_pairs):,} animal x label pairs")
        print(f"[SAVED] {out_dir / 'high_variance_label_filter_used.csv'}")
        print(f"[SAVED] {out_dir / 'high_variance_label_filter_source_summary_by_bird.csv'}")

        rows_before = len(segment_df)
        labels_before = segment_df.groupby("animal_id")["label"].nunique().sum()
        segment_df, high_var_match_summary_df = apply_high_variance_label_filter(segment_df, selected_pairs)
        high_var_match_summary_df.to_csv(out_dir / "high_variance_label_filter_match_summary_by_bird.csv", index=False)
        print(f"[INFO] Kept {len(segment_df):,} of {rows_before:,} segment rows after high-variance label filtering")
        print(f"[INFO] Kept {segment_df.groupby('animal_id')['label'].nunique().sum():,} of {labels_before:,} bird-label combinations after filtering")
        print(f"[SAVED] {out_dir / 'high_variance_label_filter_match_summary_by_bird.csv'}")

    print(f"[INFO] Final analysis table has {len(segment_df):,} segment rows")
    used_files_df.to_csv(out_dir / "input_phrase_position_segment_files_used.csv", index=False)
    print(f"[SAVED] {out_dir / 'input_phrase_position_segment_files_used.csv'}")

    if args.save_combined_segments:
        segment_df.to_csv(out_dir / "combined_phrase_position_segment_features_used.csv", index=False)
        print(f"[SAVED] {out_dir / 'combined_phrase_position_segment_features_used.csv'}")

    distribution_min_n = args.min_values_per_distribution if args.min_values_per_distribution is not None else args.min_values_per_group
    normalization_min_pre_n = (
        args.pre_normalization_min_pre_values
        if args.pre_normalization_min_pre_values is not None
        else distribution_min_n
    )

    slopes_df = compute_label_slopes(segment_df, features=features, min_n=args.min_values_per_group)
    slopes_df.to_csv(out_dir / "phrase_position_label_slopes.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_label_slopes.csv'}")

    slope_zero_bird_df, slope_zero_stat_df = summarize_slopes_against_zero(slopes_df)
    slope_zero_bird_df.to_csv(out_dir / "phrase_position_slope_against_zero_bird_summary.csv", index=False)
    slope_zero_stat_df.to_csv(out_dir / "phrase_position_slope_against_zero_stats.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_slope_against_zero_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'phrase_position_slope_against_zero_stats.csv'}")

    effects_df = compute_label_effects(slopes_df)
    effects_df.to_csv(out_dir / "phrase_position_label_effects.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_label_effects.csv'}")

    tail_df = compute_tail_effects(segment_df, features=features, min_tail_n=args.min_tail_values)
    tail_df.to_csv(out_dir / "phrase_position_tail_effects.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_tail_effects.csv'}")

    distribution_df = compute_pre_post_distribution_tests(segment_df, features=features, min_n=distribution_min_n)
    distribution_df.to_csv(out_dir / "phrase_position_pre_post_distribution_label_tests.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_pre_post_distribution_label_tests.csv'}")

    distribution_bird_df, distribution_stat_df = summarize_pre_post_distributions(distribution_df)
    distribution_bird_df.to_csv(out_dir / "phrase_position_pre_post_distribution_bird_summary.csv", index=False)
    distribution_stat_df.to_csv(out_dir / "phrase_position_pre_post_distribution_bird_stats.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_pre_post_distribution_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'phrase_position_pre_post_distribution_bird_stats.csv'}")

    bird_df, stat_df = summarize_birds(effects_df, tail_df)
    if not distribution_bird_df.empty:
        bird_df = pd.concat([bird_df, distribution_bird_df], ignore_index=True, sort=False)
    if not slope_zero_bird_df.empty:
        bird_df = pd.concat([bird_df, slope_zero_bird_df], ignore_index=True, sort=False)
    stat_parts = [stat_df]
    if not distribution_stat_df.empty:
        stat_parts.append(distribution_stat_df)
    if not slope_zero_stat_df.empty:
        stat_parts.append(slope_zero_stat_df)
    stat_df = pd.concat(stat_parts, ignore_index=True, sort=False) if stat_parts else pd.DataFrame()
    if not stat_df.empty:
        stat_df = add_holm_columns(stat_df)
    bird_df.to_csv(out_dir / "phrase_position_bird_summary.csv", index=False)
    stat_df.to_csv(out_dir / "phrase_position_bird_stats.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'phrase_position_bird_stats.csv'}")

    # Optional parallel analysis after normalizing each animal x label to its own pre-lesion baseline.
    normalized_segment_df = pd.DataFrame()
    normalized_features: List[str] = []
    normalized_distribution_df = pd.DataFrame()
    normalized_distribution_bird_df = pd.DataFrame()
    normalized_bird_df = pd.DataFrame()
    normalized_stat_df = pd.DataFrame()
    normalized_tail_df = pd.DataFrame()
    normalized_effects_df = pd.DataFrame()
    normalized_slope_zero_bird_df = pd.DataFrame()
    normalized_slope_zero_stat_df = pd.DataFrame()

    if not args.skip_pre_normalized_analysis:
        normalized_segment_df, normalized_features, normalization_info_df = add_pre_baseline_normalized_features(
            segment_df,
            features=features,
            baseline_stat=args.pre_normalization_stat,
            min_pre_n=normalization_min_pre_n,
            epsilon=args.pre_normalization_epsilon,
        )
        normalization_info_df.to_csv(out_dir / "pre_baseline_normalization_values.csv", index=False)
        print(f"[INFO] Created {len(normalized_features):,} pre-baseline-normalized feature columns")
        print(f"[SAVED] {out_dir / 'pre_baseline_normalization_values.csv'}")

        if normalized_features:
            normalized_slopes_df = compute_label_slopes(
                normalized_segment_df, features=normalized_features, min_n=args.min_values_per_group
            )
            normalized_slopes_df.to_csv(out_dir / "normalized_phrase_position_label_slopes.csv", index=False)
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_label_slopes.csv'}")

            normalized_slope_zero_bird_df, normalized_slope_zero_stat_df = summarize_slopes_against_zero(normalized_slopes_df)
            normalized_slope_zero_bird_df.to_csv(
                out_dir / "normalized_phrase_position_slope_against_zero_bird_summary.csv", index=False
            )
            normalized_slope_zero_stat_df.to_csv(
                out_dir / "normalized_phrase_position_slope_against_zero_stats.csv", index=False
            )
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_slope_against_zero_bird_summary.csv'}")
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_slope_against_zero_stats.csv'}")

            normalized_effects_df = compute_label_effects(normalized_slopes_df)
            normalized_effects_df.to_csv(out_dir / "normalized_phrase_position_label_effects.csv", index=False)
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_label_effects.csv'}")

            normalized_tail_df = compute_tail_effects(
                normalized_segment_df, features=normalized_features, min_tail_n=args.min_tail_values
            )
            normalized_tail_df.to_csv(out_dir / "normalized_phrase_position_tail_effects.csv", index=False)
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_tail_effects.csv'}")

            normalized_distribution_df = compute_pre_post_distribution_tests(
                normalized_segment_df, features=normalized_features, min_n=distribution_min_n
            )
            normalized_distribution_df.to_csv(
                out_dir / "normalized_phrase_position_pre_post_distribution_label_tests.csv", index=False
            )
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_pre_post_distribution_label_tests.csv'}")

            normalized_distribution_bird_df, normalized_distribution_stat_df = summarize_pre_post_distributions(
                normalized_distribution_df
            )
            normalized_distribution_bird_df.to_csv(
                out_dir / "normalized_phrase_position_pre_post_distribution_bird_summary.csv", index=False
            )
            normalized_distribution_stat_df.to_csv(
                out_dir / "normalized_phrase_position_pre_post_distribution_bird_stats.csv", index=False
            )
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_pre_post_distribution_bird_summary.csv'}")
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_pre_post_distribution_bird_stats.csv'}")

            normalized_bird_df, normalized_stat_df = summarize_birds(normalized_effects_df, normalized_tail_df)
            if not normalized_distribution_bird_df.empty:
                normalized_bird_df = pd.concat([normalized_bird_df, normalized_distribution_bird_df], ignore_index=True, sort=False)
            if not normalized_slope_zero_bird_df.empty:
                normalized_bird_df = pd.concat([normalized_bird_df, normalized_slope_zero_bird_df], ignore_index=True, sort=False)
            normalized_stat_parts = [normalized_stat_df]
            if not normalized_distribution_stat_df.empty:
                normalized_stat_parts.append(normalized_distribution_stat_df)
            if not normalized_slope_zero_stat_df.empty:
                normalized_stat_parts.append(normalized_slope_zero_stat_df)
            normalized_stat_df = pd.concat(normalized_stat_parts, ignore_index=True, sort=False) if normalized_stat_parts else pd.DataFrame()
            if not normalized_stat_df.empty:
                normalized_stat_df = add_holm_columns(normalized_stat_df)
            normalized_bird_df.to_csv(out_dir / "normalized_phrase_position_bird_summary.csv", index=False)
            normalized_stat_df.to_csv(out_dir / "normalized_phrase_position_bird_stats.csv", index=False)
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_bird_summary.csv'}")
            print(f"[SAVED] {out_dir / 'normalized_phrase_position_bird_stats.csv'}")

    for feature in features:
        plot_pre_post_distribution_effect_by_bird(
            distribution_df, distribution_bird_df, stat_df, feature,
            out_dir / f"pre_post_distribution_effect_by_bird_{safe_name(feature)}.png",
            args.title_prefix,
        )
        plot_pre_vs_post_distribution_paired_birds(
            distribution_bird_df, stat_df, feature,
            out_dir / f"pre_vs_post_distribution_bird_paired_{safe_name(feature)}.png",
            args.title_prefix,
        )
        for predictor in PREDICTORS:
            pred_tag = PREDICTOR_SHORT[predictor]
            plot_post_minus_pre_by_bird(
                effects_df, bird_df, stat_df, feature, predictor,
                out_dir / f"post_minus_pre_degradation_slope_by_bird_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            plot_pre_vs_post_paired_birds(
                bird_df, stat_df, feature, predictor,
                out_dir / f"pre_vs_post_degradation_slope_bird_paired_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            plot_baseline_vs_lesion(
                effects_df, bird_df, stat_df, feature, predictor,
                out_dir / f"baseline_vs_lesion_slope_change_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            plot_tail_effect_by_bird(
                tail_df, bird_df, stat_df, feature, predictor,
                out_dir / f"post_only_tail_effect_by_bird_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            max_x = args.max_plot_elapsed_s if predictor == "elapsed_time_in_phrase_s" else args.max_plot_previous_repeats
            plot_aggregate_binned_position(
                segment_df, feature, predictor,
                out_dir / f"aggregate_binned_{safe_name(feature)}_vs_{pred_tag}.png",
                args.title_prefix,
                n_bins=args.n_bins,
                max_x=max_x,
                min_bin_n=args.min_bin_n,
                stat_df=stat_df,
                slope_zero_stat_df=slope_zero_stat_df,
            )

    if normalized_features:
        normalized_title_prefix = (
            f"{args.title_prefix}\nPre-normalized to each syllable's pre-lesion {args.pre_normalization_stat}"
        )
        for feature in normalized_features:
            plot_pre_post_distribution_effect_by_bird(
                normalized_distribution_df, normalized_distribution_bird_df, normalized_stat_df, feature,
                out_dir / f"normalized_pre_post_distribution_effect_by_bird_{safe_name(feature)}.png",
                normalized_title_prefix,
            )
            plot_pre_vs_post_distribution_paired_birds(
                normalized_distribution_bird_df, normalized_stat_df, feature,
                out_dir / f"normalized_pre_vs_post_distribution_bird_paired_{safe_name(feature)}.png",
                normalized_title_prefix,
            )
            for predictor in PREDICTORS:
                pred_tag = PREDICTOR_SHORT[predictor]
                plot_post_minus_pre_by_bird(
                    normalized_effects_df, normalized_bird_df, normalized_stat_df, feature, predictor,
                    out_dir / f"normalized_post_minus_pre_degradation_slope_by_bird_{safe_name(feature)}_{pred_tag}.png",
                    normalized_title_prefix,
                )
                plot_pre_vs_post_paired_birds(
                    normalized_bird_df, normalized_stat_df, feature, predictor,
                    out_dir / f"normalized_pre_vs_post_degradation_slope_bird_paired_{safe_name(feature)}_{pred_tag}.png",
                    normalized_title_prefix,
                )
                plot_baseline_vs_lesion(
                    normalized_effects_df, normalized_bird_df, normalized_stat_df, feature, predictor,
                    out_dir / f"normalized_baseline_vs_lesion_slope_change_{safe_name(feature)}_{pred_tag}.png",
                    normalized_title_prefix,
                )
                plot_tail_effect_by_bird(
                    normalized_tail_df, normalized_bird_df, normalized_stat_df, feature, predictor,
                    out_dir / f"normalized_post_only_tail_effect_by_bird_{safe_name(feature)}_{pred_tag}.png",
                    normalized_title_prefix,
                )
                max_x = args.max_plot_elapsed_s if predictor == "elapsed_time_in_phrase_s" else args.max_plot_previous_repeats
                plot_aggregate_binned_position(
                    normalized_segment_df, feature, predictor,
                    out_dir / f"normalized_aggregate_binned_{safe_name(feature)}_vs_{pred_tag}.png",
                    normalized_title_prefix,
                    n_bins=args.n_bins,
                    max_x=max_x,
                    min_bin_n=args.min_bin_n,
                    stat_df=normalized_stat_df,
                    slope_zero_stat_df=normalized_slope_zero_stat_df,
                )
            plot_normalized_feature_summary(
                normalized_distribution_df, normalized_distribution_bird_df, normalized_tail_df, normalized_bird_df, normalized_stat_df,
                feature,
                out_dir / f"normalized_summary_prepost_tail_{safe_name(feature)}.png",
                normalized_title_prefix,
            )

    for predictor in PREDICTORS:
        plot_max_extent_by_bird(
            tail_df, predictor,
            out_dir / f"max_post_phrase_extent_by_bird_{PREDICTOR_SHORT[predictor]}.png",
            args.title_prefix,
        )

    write_readme(out_dir)
    print(f"[SAVED] {out_dir / 'README_phrase_position_bird_effects_v9.txt'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
