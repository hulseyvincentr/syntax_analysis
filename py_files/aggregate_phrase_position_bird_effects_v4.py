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
    phrase_position_tail_effects.csv
    phrase_position_bird_summary.csv
    phrase_position_bird_stats.csv

Figures:
    post_minus_pre_degradation_slope_by_bird_*.png
    pre_vs_post_degradation_slope_bird_paired_*.png
    baseline_vs_lesion_slope_change_*.png
    post_only_tail_effect_by_bird_*.png
    max_post_phrase_extent_by_bird_*.png
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
"""

from __future__ import annotations

import argparse
import math
import re
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

    stat_df = pd.DataFrame(stat_rows)
    if not stat_df.empty:
        for col in [c for c in stat_df.columns if c.endswith("_p")]:
            stat_df[f"{col}_holm"] = holm_adjust(stat_df[col])
    return bird_df, stat_df


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
    return (
        f"bird-level Wilcoxon tail>shared p={fmt_p(r.get('wilcoxon_tail_minus_shared_gt0_p'))}; "
        f"tail>pre p={fmt_p(r.get('wilcoxon_tail_minus_pre_gt0_p'))}"
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
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


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
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


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
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


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
    ax.set_ylabel("Post-only tail - shared post range\nmedian degradation value")
    ax.set_title(f"{title_prefix}\nPost-only stutter-tail effect\n{FEATURE_LABELS.get(feature, feature)}\n{stat_subtitle(stat_df, feature, predictor, 'post_only_tail_effect')}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


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


def plot_aggregate_binned_position(df, feature, predictor, out_png, title_prefix, n_bins=40, max_x=None, min_bin_n=10):
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
    ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
    ax.set_title(f"{title_prefix}\nAggregate binned median ± IQR\n{FEATURE_LABELS.get(feature, feature)}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate phrase-position acoustic/spectrogram effects across birds.")
    p.add_argument("--phrase-position-root", required=True,
                   help="Root folder containing outputs from phrase_position_acoustic_crosscorr_v6_majority_vote_runs.py")
    p.add_argument("--out-dir", default=None,
                   help="Output folder. Default: <phrase-position-root>/aggregate_bird_effects_v4")
    p.add_argument("--metadata-excel-path", default=None,
                   help="Metadata Excel path for Medial + Lateral filtering")
    p.add_argument("--lesion-filter", choices=["medial_lateral", "all"], default="medial_lateral")
    p.add_argument("--lesion-metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--lesion-animal-col", default=None)
    p.add_argument("--lesion-hit-type-col", default=None)
    p.add_argument("--animal-ids", default=None,
                   help="Optional comma-separated animal IDs to include after lesion filtering")
    p.add_argument("--features", default="corr_to_phrase_early_template,distance_from_phrase_early_template",
                   help="Comma-separated features to aggregate")
    p.add_argument("--min-values-per-group", type=int, default=5,
                   help="Minimum segment values required to fit a slope")
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
    (out_dir / "README_phrase_position_bird_effects_v4.txt").write_text(
        "Aggregate phrase-position bird-level analysis.\n\n"
        "Core idea:\n"
        "  Each animal x syllable label gets a pre and post phrase-position slope.\n"
        "  Then each bird is summarized by the median across its selected labels.\n"
        "  This prevents birds/labels with many segmented syllables from dominating.\n\n"
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
        "  phrase_position_label_effects.csv\n"
        "  phrase_position_tail_effects.csv\n"
        "  phrase_position_bird_summary.csv\n"
        "  phrase_position_bird_stats.csv\n"
    )


def main():
    args = parse_args()
    root = Path(args.phrase_position_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "aggregate_bird_effects_v4"
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
    print(f"[INFO] Loaded {len(segment_df):,} segment rows from {len(used_files_df):,} CSV files")
    used_files_df.to_csv(out_dir / "input_phrase_position_segment_files_used.csv", index=False)
    print(f"[SAVED] {out_dir / 'input_phrase_position_segment_files_used.csv'}")

    if args.save_combined_segments:
        segment_df.to_csv(out_dir / "combined_phrase_position_segment_features_used.csv", index=False)
        print(f"[SAVED] {out_dir / 'combined_phrase_position_segment_features_used.csv'}")

    slopes_df = compute_label_slopes(segment_df, features=features, min_n=args.min_values_per_group)
    slopes_df.to_csv(out_dir / "phrase_position_label_slopes.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_label_slopes.csv'}")

    effects_df = compute_label_effects(slopes_df)
    effects_df.to_csv(out_dir / "phrase_position_label_effects.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_label_effects.csv'}")

    tail_df = compute_tail_effects(segment_df, features=features, min_tail_n=args.min_tail_values)
    tail_df.to_csv(out_dir / "phrase_position_tail_effects.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_tail_effects.csv'}")

    bird_df, stat_df = summarize_birds(effects_df, tail_df)
    bird_df.to_csv(out_dir / "phrase_position_bird_summary.csv", index=False)
    stat_df.to_csv(out_dir / "phrase_position_bird_stats.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'phrase_position_bird_stats.csv'}")

    for feature in features:
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
            )

    for predictor in PREDICTORS:
        plot_max_extent_by_bird(
            tail_df, predictor,
            out_dir / f"max_post_phrase_extent_by_bird_{PREDICTOR_SHORT[predictor]}.png",
            args.title_prefix,
        )

    write_readme(out_dir)
    print(f"[SAVED] {out_dir / 'README_phrase_position_bird_effects_v4.txt'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
