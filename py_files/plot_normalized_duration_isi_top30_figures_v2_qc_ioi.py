#!/usr/bin/env python3
"""
QC-aware normalized duration / gap / inter-onset interval analysis for repeated syllable phrases.

Designed for outputs from:
    phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py

This is a more conservative version of the normalized combined-figure script.
It implements the QC suggestions that are especially important for ISI/gap analyses:

1) Computes inter-onset interval (IOI): current syllable onset - previous syllable onset.
   This is often more stable than end-to-start ISI when adjacent segments nearly touch.

2) Excludes tiny-denominator animal x label baselines for normalized plots.
   By default, ISI baselines below 5 ms are excluded because dividing by near-zero
   pre-lesion gaps can create huge ratios.

3) Excludes tiny-denominator paired pre/post repeat bins.
   For paired post/pre effects, an animal x label x repeat index is excluded if the
   pre value is below the feature-specific minimum.

4) Requires a minimum number of animal x label units and birds per repeat index
   before drawing aggregate mean points.

5) Saves n-per-repeat-index QC tables and draws small n panels underneath figures.

6) Uses robust y-limits based on percentiles so a few extreme ratios do not dominate
   the visual scale. The underlying CSVs still contain the finite plotted values.

Main outputs in <out-dir>/figures:
    normalized_duration_gap_ioi_with_n.png
    post_pre_log2_ratio_duration_gap_ioi_with_n.png
    post_pre_normalized_difference_duration_gap_ioi_with_n.png
    post_pre_ratio_duration_gap_ioi_with_n.png
    normalized_time_panels_with_n.png
    post_pre_log2_ratio_time_panels_with_n.png

Key output CSVs in <out-dir>:
    combined_segments_used_with_ioi_and_qc.csv
    pre_baseline_qc_by_animal_label_feature.csv
    normalized_to_pre_mean_unit_values.csv
    normalized_to_pre_mean_repeat_summaries.csv
    paired_post_pre_by_animal_label_repeat_filtered.csv
    paired_effect_repeat_summaries.csv
    bin_counts_normalized.csv
    bin_counts_paired_effects.csv

Interpretation:
    Normalized-to-pre value:
        1.0 = equal to that animal x label's own pre-lesion mean.

    Post/pre ratio:
        1.0 = no change; >1 = longer post; <1 = shorter post.

    log2(post/pre):
        0 = no change; +1 = 2x longer post; -1 = half as long post.

    (post - pre)/(post + pre):
        0 = no change; positive = longer post; negative = shorter post.
        Bounded between -1 and +1 for positive values.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

PRE_COLOR = "#1f77b4"   # blue
POST_COLOR = "#d62728"  # red
EFFECT_COLOR = "#333333"
UNIT_COLOR = "#777777"
POINT_ALPHA = 0.22
POINT_SIZE = 14
LINE_WIDTH = 2.6
MARKER_SIZE = 5.2
SEM_ALPHA = 0.14
EPS = 1e-12

ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal", "animal", "Bird", "bird", "bird_id", "AnimalID"
]
LABEL_COL_CANDIDATES = [
    "label", "Label", "syllable", "Syllable", "syllable_label", "Syllable label",
    "cluster", "Cluster", "cluster_id", "Cluster ID", "hdbscan_label", "HDBSCAN label",
    "mapped_syllable_order", "syllable_cluster", "Syllable cluster", "chosen_label", "selected_label"
]

SEGMENT_REPEAT_FEATURES = ["segment_duration_ms", "preceding_isi_ms", "preceding_ioi_ms"]
TIME_FEATURES = ["cumulative_phrase_time_s", "phrase_duration_s"]

FEATURE_LABELS = {
    "segment_duration_ms": "Syllable duration",
    "preceding_isi_ms": "Preceding inter-syllabic interval",
    "preceding_ioi_ms": "Preceding inter-onset interval",
    "following_isi_ms": "Following inter-syllabic interval",
    "cumulative_phrase_time_s": "Cumulative phrase time",
    "cumulative_syllable_duration_s": "Cumulative syllable duration only",
    "phrase_duration_s": "Total repeated-label phrase duration",
}

FEATURE_UNITS = {
    "segment_duration_ms": "ms",
    "preceding_isi_ms": "ms",
    "preceding_ioi_ms": "ms",
    "following_isi_ms": "ms",
    "cumulative_phrase_time_s": "s",
    "cumulative_syllable_duration_s": "s",
    "phrase_duration_s": "s",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


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
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"(?:^|_)label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def infer_animal_from_path(path: Path) -> str:
    m = re.search(r"(?P<animal>[^/\\]+)_all_labels_duration_isi_segment_timing\.csv$", path.name)
    if m:
        return m.group("animal")
    m = re.search(r"(?P<animal>[^/\\]+)_label.+?_duration_isi_segment_timing\.csv$", path.name)
    if m:
        return m.group("animal")
    return path.parent.name


def finite_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def wilcoxon_against_null(values: np.ndarray, null_value: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or not SCIPY_AVAILABLE:
        return np.nan
    centered = values - float(null_value)
    if np.allclose(centered, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(centered, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def find_segment_timing_files(root: Path) -> list[Path]:
    """Find timing CSVs, preferring one all-label file per animal."""
    all_label_files = sorted(
        f for f in root.rglob("*_all_labels_duration_isi_segment_timing.csv")
        if not f.name.startswith("._")
    )
    if all_label_files:
        by_animal: dict[str, Path] = {}
        for f in all_label_files:
            animal = infer_animal_from_path(f)
            by_animal.setdefault(animal, f)
        return sorted(by_animal.values())

    return sorted(
        f for f in root.rglob("*_duration_isi_segment_timing.csv")
        if not f.name.startswith("._")
    )


def load_top30_pairs(path: str | None, animal_col: str | None = None, label_col: str | None = None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    table = pd.read_csv(Path(path).expanduser())
    if table.empty:
        raise ValueError(f"Top-30 label-pair CSV is empty: {path}")
    animal_col = animal_col or find_first_existing(table.columns, ANIMAL_COL_CANDIDATES)
    label_col = label_col or find_first_existing(table.columns, LABEL_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(f"Could not find animal ID column in {path}. Columns: {list(table.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column in {path}. Columns: {list(table.columns)}")
    pairs = set()
    for _, row in table.iterrows():
        animal = normalize_animal_id(row[animal_col])
        label = normalize_label_id(row[label_col])
        if animal and label:
            pairs.add((animal, label))
    if not pairs:
        raise ValueError(f"No usable animal x label pairs found in {path}")
    return pairs


def read_segments(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(args.timing_root).expanduser()
    files = find_segment_timing_files(root)
    if not files:
        raise FileNotFoundError(f"No *_duration_isi_segment_timing.csv files found under {root}")

    frames = []
    manifest_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[WARN] Could not read {f}: {exc}")
            continue
        if df.empty:
            continue
        if "animal_id" not in df.columns:
            df["animal_id"] = infer_animal_from_path(f)
        if "label" not in df.columns:
            print(f"[WARN] Skipping {f}; no label column")
            continue
        if "pre_post" not in df.columns:
            if "period" in df.columns:
                df["pre_post"] = np.where(
                    df["period"].astype(str).str.contains("post", case=False, na=False),
                    "post", "pre",
                )
            else:
                print(f"[WARN] Skipping {f}; no pre_post or period column")
                continue

        df["animal_id"] = df["animal_id"].map(normalize_animal_id)
        df["label"] = df["label"].map(normalize_label_id)
        df["pre_post"] = df["pre_post"].astype(str).str.lower().str.strip()
        df = df[df["pre_post"].isin(["pre", "post"])].copy()
        if df.empty:
            continue
        df["source_csv"] = str(f)
        frames.append(df)
        manifest_rows.append({
            "csv_path": str(f),
            "animal_id": ",".join(sorted(df["animal_id"].dropna().astype(str).unique())),
            "n_rows": int(df.shape[0]),
            "n_labels": int(df["label"].nunique()),
            "labels": ",".join(sorted(df["label"].dropna().astype(str).unique(), key=lambda x: (len(x), x))),
        })

    if not frames:
        raise RuntimeError("No usable timing rows found after reading files.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest_rows)


def filter_segments(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.animals:
        animals = {a.strip() for a in args.animals.split(",") if a.strip()}
        out = out[out["animal_id"].isin(animals)].copy()

    pairs = load_top30_pairs(args.top30_labels_csv, args.animal_col, args.label_col)
    if pairs is not None:
        before = out.shape[0]
        out = out[
            [(a, l) in pairs for a, l in zip(out["animal_id"].astype(str), out["label"].astype(str))]
        ].copy()
        print(f"[INFO] Top-30 pair filter kept {out.shape[0]} / {before} segment rows")
        if out.empty:
            raise RuntimeError("Top-30 filter removed all rows. Check animal/label columns and label formatting.")

    if args.max_repeat_index is not None and "repeat_index_in_phrase" in out.columns:
        out = out[finite_numeric(out["repeat_index_in_phrase"]) <= args.max_repeat_index].copy()
    return out


def compute_preceding_ioi_ms(segments: pd.DataFrame) -> pd.DataFrame:
    """Add preceding_ioi_ms = current onset - previous onset within each phrase.

    Uses segment_start_elapsed_s when available. If not available, falls back to:
        preceding_ioi_ms = previous segment_duration_ms + preceding_isi_ms
    """
    work = segments.copy()
    group_cols = ["animal_id", "label", "pre_post", "phrase_id"]
    for extra in ["period", "source_csv", "recording_key", "file_index", "file_indices_stitched"]:
        if extra in work.columns and extra not in group_cols:
            group_cols.append(extra)

    if "repeat_index_in_phrase" in work.columns:
        sort_cols = group_cols + ["repeat_index_in_phrase"]
    else:
        sort_cols = group_cols
    work = work.sort_values(sort_cols).copy()

    if "segment_start_elapsed_s" in work.columns:
        work["segment_start_elapsed_s"] = finite_numeric(work["segment_start_elapsed_s"])
        prev_start = work.groupby(group_cols, dropna=False)["segment_start_elapsed_s"].shift(1)
        ioi = (work["segment_start_elapsed_s"] - prev_start) * 1000.0
        work["preceding_ioi_ms"] = np.where(np.isfinite(ioi) & (ioi >= 0), ioi, np.nan)
    else:
        work["segment_duration_ms"] = finite_numeric(work["segment_duration_ms"])
        work["preceding_isi_ms"] = finite_numeric(work["preceding_isi_ms"])
        prev_duration = work.groupby(group_cols, dropna=False)["segment_duration_ms"].shift(1)
        ioi = prev_duration + work["preceding_isi_ms"]
        work["preceding_ioi_ms"] = np.where(np.isfinite(ioi) & (ioi >= 0), ioi, np.nan)

    if "repeat_index_in_phrase" in work.columns:
        first_repeat = finite_numeric(work["repeat_index_in_phrase"]) <= 1
        work.loc[first_repeat, "preceding_ioi_ms"] = np.nan
    return work


def build_phrase_table(segments: pd.DataFrame) -> pd.DataFrame:
    required = ["animal_id", "label", "pre_post", "phrase_id", "phrase_duration_s", "n_segments_in_phrase"]
    missing = [c for c in required if c not in segments.columns]
    if missing:
        raise ValueError(f"Cannot build phrase table; missing columns: {missing}")
    group_cols = ["animal_id", "label", "pre_post", "phrase_id"]
    for extra in ["period", "recording_key", "file_index", "file_indices_stitched", "source_csv"]:
        if extra in segments.columns:
            group_cols.append(extra)
    phrase = (
        segments.groupby(group_cols, as_index=False)
        .agg(
            phrase_duration_s=("phrase_duration_s", "first"),
            n_segments_in_phrase=("n_segments_in_phrase", "first"),
            n_segment_rows=("repeat_index_in_phrase", "size"),
        )
    )
    phrase["n_segments_in_phrase"] = finite_numeric(phrase["n_segments_in_phrase"]).astype("Int64")
    phrase["phrase_duration_s"] = finite_numeric(phrase["phrase_duration_s"])
    phrase = phrase[np.isfinite(phrase["phrase_duration_s"]) & phrase["n_segments_in_phrase"].notna()].copy()
    phrase["n_segments_in_phrase"] = phrase["n_segments_in_phrase"].astype(int)
    return phrase


def get_min_pre_value_for_feature(feature: str, args) -> float:
    if feature == "preceding_isi_ms":
        return float(args.min_pre_mean_ms_for_isi)
    if feature == "preceding_ioi_ms":
        return float(args.min_pre_mean_ms_for_ioi)
    if feature == "segment_duration_ms":
        return float(args.min_pre_mean_ms_for_duration)
    # seconds-based cumulative/phrase features usually have no tiny-denominator problem,
    # but keep a small positive minimum to avoid divide-by-zero.
    return float(args.min_pre_mean_s_for_time_features)


def compute_pre_baseline_qc(df: pd.DataFrame, features: list[str], args) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        work = df[["animal_id", "label", "pre_post", feature]].copy()
        work[feature] = finite_numeric(work[feature])
        pre = work[(work["pre_post"] == "pre") & np.isfinite(work[feature])].copy()
        if pre.empty:
            continue
        base = (
            pre.groupby(["animal_id", "label"], as_index=False)
            .agg(pre_baseline=(feature, "mean"), n_pre_baseline_values=(feature, "size"))
        )
        min_pre = get_min_pre_value_for_feature(feature, args)
        base["feature"] = feature
        base["feature_label"] = FEATURE_LABELS.get(feature, feature)
        base["min_required_pre_baseline"] = min_pre
        base["passes_pre_baseline_filter"] = np.isfinite(base["pre_baseline"]) & (base["pre_baseline"] >= min_pre)
        rows.append(base)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_within_label_pre_baseline(df: pd.DataFrame, y_col: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide each value by that animal x label's overall pre-lesion mean for this feature."""
    work = df.copy()
    work[y_col] = finite_numeric(work[y_col])
    pre = work[(work["pre_post"] == "pre") & np.isfinite(work[y_col])].copy()
    baseline = (
        pre.groupby(["animal_id", "label"], as_index=False)
        .agg(pre_baseline=(y_col, "mean"), n_pre_baseline_values=(y_col, "size"))
    )
    min_pre = get_min_pre_value_for_feature(y_col, args)
    baseline["passes_pre_baseline_filter"] = np.isfinite(baseline["pre_baseline"]) & (baseline["pre_baseline"] >= min_pre)
    baseline["min_required_pre_baseline"] = min_pre
    baseline["feature"] = y_col
    work = work.merge(baseline[["animal_id", "label", "pre_baseline", "n_pre_baseline_values", "passes_pre_baseline_filter"]], on=["animal_id", "label"], how="left")
    norm_col = f"{y_col}_normalized_to_pre_mean"
    ok = (
        np.isfinite(work[y_col]) &
        np.isfinite(work["pre_baseline"]) &
        (work["pre_baseline"] > EPS) &
        (work["passes_pre_baseline_filter"] == True)
    )
    work[norm_col] = np.nan
    work.loc[ok, norm_col] = work.loc[ok, y_col] / work.loc[ok, "pre_baseline"]
    return work, baseline


def make_unit_means(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_raw_values_per_unit_bin: int = 1,
) -> pd.DataFrame:
    """Return animal x label x pre/post x x-bin means."""
    cols = ["animal_id", "label", "pre_post", x_col, y_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for unit means: {missing}")
    work = df[cols].copy()
    work[x_col] = finite_numeric(work[x_col])
    work[y_col] = finite_numeric(work[y_col])
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])].copy()
    if work.empty:
        return pd.DataFrame()
    work[x_col] = work[x_col].astype(int)
    unit = (
        work.groupby(["animal_id", "label", "pre_post", x_col], as_index=False)
        .agg(unit_mean=(y_col, "mean"), unit_median=(y_col, "median"), n_raw_values=(y_col, "size"))
    )
    unit = unit[unit["n_raw_values"] >= int(min_raw_values_per_unit_bin)].copy()
    return unit


def summarize_by_x(
    unit: pd.DataFrame,
    x_col: str,
    value_col: str,
    group_col: Optional[str],
    min_units: int,
    min_animals: int,
    null_value: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    if unit.empty:
        return pd.DataFrame()
    groupers = [x_col] if group_col is None else [group_col, x_col]
    for key, sub in unit.groupby(groupers, dropna=False):
        if group_col is None:
            x = key[0] if isinstance(key, tuple) else key
            grp = None
        else:
            if isinstance(key, tuple):
                grp, x = key
            else:
                raise RuntimeError("Unexpected non-tuple grouped key when group_col is not None")
        vals = finite_numeric(sub[value_col]).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n_units = int(vals.size)
        n_animals = int(sub["animal_id"].nunique()) if "animal_id" in sub.columns else np.nan
        n_pairs = int(sub[["animal_id", "label"]].drop_duplicates().shape[0]) if {"animal_id", "label"}.issubset(sub.columns) else np.nan
        passes = (n_units >= int(min_units)) and (n_animals >= int(min_animals))
        row = {
            x_col: int(x),
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "median": float(np.median(vals)) if vals.size else np.nan,
            "sem": sem(vals),
            "n_units": n_units,
            "n_animals": n_animals,
            "n_animal_label_pairs": n_pairs,
            "passes_min_n_filter": bool(passes),
        }
        if group_col is not None:
            row[group_col] = grp
        if null_value is not None and vals.size:
            row["wilcoxon_p_vs_null"] = wilcoxon_against_null(vals, null_value)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = [x_col] if group_col is None else [group_col, x_col]
    return out.sort_values(sort_cols)


def paired_post_pre_by_repeat(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    args,
) -> pd.DataFrame:
    """Compute paired post/pre values within animal x label x repeat bin."""
    unit = make_unit_means(
        df,
        x_col=x_col,
        y_col=y_col,
        min_raw_values_per_unit_bin=args.min_raw_values_per_unit_bin,
    )
    if unit.empty:
        return pd.DataFrame()
    wide = unit.pivot_table(
        index=["animal_id", "label", x_col],
        columns="pre_post",
        values="unit_mean",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    if "pre" not in wide.columns or "post" not in wide.columns:
        return pd.DataFrame()
    wide = wide[np.isfinite(wide["pre"]) & np.isfinite(wide["post"])].copy()
    min_pre = get_min_pre_value_for_feature(y_col, args)
    wide["min_required_pre_value"] = min_pre
    wide["passes_pre_value_filter"] = wide["pre"] >= min_pre
    wide["feature"] = y_col
    wide["x_col"] = x_col
    wide = wide[(wide["passes_pre_value_filter"]) & (wide["post"] >= 0)].copy()
    if wide.empty:
        return pd.DataFrame()
    wide["post_minus_pre"] = wide["post"] - wide["pre"]
    wide["post_pre_ratio"] = wide["post"] / wide["pre"]
    wide["log2_post_pre_ratio"] = np.log2(wide["post_pre_ratio"])
    denom = wide["post"] + wide["pre"]
    wide["normalized_difference_post_minus_pre_over_sum"] = np.where(
        denom > EPS,
        (wide["post"] - wide["pre"]) / denom,
        np.nan,
    )
    wide["percent_change_post_vs_pre"] = 100.0 * (wide["post_pre_ratio"] - 1.0)
    return wide.sort_values([x_col, "animal_id", "label"])


def dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep_h = []
    keep_l = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            keep_h.append(h)
            keep_l.append(lab)
            seen.add(lab)
    if keep_h:
        ax.legend(keep_h, keep_l, fontsize=8, frameon=True)


def set_robust_ylim(ax, values, null_value: Optional[float], args, lower_bound: Optional[float] = None):
    if not args.robust_ylim:
        return
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return
    lo, hi = np.nanpercentile(vals, [args.robust_ylim_low, args.robust_ylim_high])
    if null_value is not None and np.isfinite(null_value):
        lo = min(lo, null_value)
        hi = max(hi, null_value)
    if lower_bound is not None:
        lo = max(lo, lower_bound)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def plot_n_panel(ax, count_df: pd.DataFrame, x_col: str, kind: str = "prepost"):
    if count_df.empty:
        ax.axis("off")
        return
    if kind == "prepost" and "pre_post" in count_df.columns:
        for group, color in [("pre", PRE_COLOR), ("post", POST_COLOR)]:
            sub = count_df[count_df["pre_post"] == group].copy()
            if sub.empty:
                continue
            ax.plot(sub[x_col], sub["n_units"], color=color, linewidth=1.8, marker="o", markersize=3, label=f"{group} n")
    else:
        ax.plot(count_df[x_col], count_df["n_units"], color=EFFECT_COLOR, linewidth=1.8, marker="o", markersize=3, label="paired n")
    ax.set_ylabel("n units", fontsize=8)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=8)
    dedupe_legend(ax)


def add_suptitle_with_counts(fig, segments: pd.DataFrame, title: str):
    n_animals = segments["animal_id"].nunique()
    n_labels = segments[["animal_id", "label"]].drop_duplicates().shape[0]
    n_segments = segments.shape[0]
    fig.suptitle(f"{title}\n{n_animals} birds, {n_labels} animal×label pairs, {n_segments:,} segmented syllables", fontsize=14)


def save_csv(df: pd.DataFrame, path: Path):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        print(f"[SAVED] {path}")


def normalized_unit_and_summary(df: pd.DataFrame, x_col: str, feature: str, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work, baseline = add_within_label_pre_baseline(df, feature, args)
    norm_col = f"{feature}_normalized_to_pre_mean"
    unit = make_unit_means(
        work,
        x_col=x_col,
        y_col=norm_col,
        min_raw_values_per_unit_bin=args.min_raw_values_per_unit_bin,
    )
    if unit.empty:
        return unit, pd.DataFrame(), baseline
    unit["feature"] = feature
    unit["x_col"] = x_col
    summary = summarize_by_x(
        unit,
        x_col=x_col,
        value_col="unit_mean",
        group_col="pre_post",
        min_units=args.min_units_per_bin,
        min_animals=args.min_animals_per_bin,
        null_value=1.0,
    )
    if not summary.empty:
        summary["feature"] = feature
        summary["x_col"] = x_col
    return unit, summary, baseline


def plot_normalized_panel(ax, n_ax, unit: pd.DataFrame, summary: pd.DataFrame, x_col: str, feature: str, xlabel: str, args, rng):
    values_for_ylim = []
    for group, color in [("pre", PRE_COLOR), ("post", POST_COLOR)]:
        sub = unit[unit["pre_post"] == group].copy() if not unit.empty else pd.DataFrame()
        if sub.empty:
            continue
        if args.show_unit_points:
            show = sub
            if args.max_unit_points_per_panel and show.shape[0] > args.max_unit_points_per_panel:
                show = show.sample(n=args.max_unit_points_per_panel, random_state=int(rng.integers(0, 2**32 - 1)))
            x = show[x_col].to_numpy(dtype=float)
            if args.jitter > 0:
                x = x + rng.uniform(-args.jitter, args.jitter, size=x.size)
            y = show["unit_mean"].to_numpy(dtype=float)
            values_for_ylim.extend(y[np.isfinite(y)].tolist())
            ax.scatter(x, y, s=args.point_size, alpha=args.point_alpha, color=color,
                       edgecolors="none", label=f"{group} animal×label means", rasterized=True)

        sline = summary[(summary["pre_post"] == group) & (summary["passes_min_n_filter"] == True)].copy() if not summary.empty else pd.DataFrame()
        if not sline.empty:
            xx = sline[x_col].to_numpy(dtype=float)
            yy = sline["mean"].to_numpy(dtype=float)
            values_for_ylim.extend(yy[np.isfinite(yy)].tolist())
            ax.plot(xx, yy, color=color, linewidth=args.line_width, marker="o", markersize=args.marker_size,
                    label=f"{group} mean")
            if args.show_sem_band:
                ee = sline["sem"].to_numpy(dtype=float)
                ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
                if np.any(ok):
                    ax.fill_between(xx[ok], yy[ok] - ee[ok], yy[ok] + ee[ok], color=color, alpha=SEM_ALPHA, linewidth=0)

    ax.axhline(1.0, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{FEATURE_LABELS.get(feature, feature)} / pre mean")
    ax.set_title(f"{FEATURE_LABELS.get(feature, feature)}")
    ax.grid(alpha=0.25)
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    set_robust_ylim(ax, values_for_ylim, 1.0, args, lower_bound=0.0)
    dedupe_legend(ax)

    counts = summary[["pre_post", x_col, "n_units", "n_animals", "passes_min_n_filter"]].copy() if not summary.empty else pd.DataFrame()
    plot_n_panel(n_ax, counts, x_col=x_col, kind="prepost")
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        n_ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    n_ax.set_xlabel(xlabel, fontsize=8)


def effect_unit_and_summary(paired: pd.DataFrame, x_col: str, feature: str, effect_col: str, null_value: float, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = paired[(paired["feature"] == feature) & (paired["x_col"] == x_col)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work[effect_col] = finite_numeric(work[effect_col])
    work = work[np.isfinite(work[effect_col])].copy()
    if work.empty:
        return work, pd.DataFrame()
    summary = summarize_by_x(
        work.rename(columns={effect_col: "effect_value"}),
        x_col=x_col,
        value_col="effect_value",
        group_col=None,
        min_units=args.min_units_per_bin,
        min_animals=args.min_animals_per_bin,
        null_value=null_value,
    )
    if not summary.empty:
        summary["feature"] = feature
        summary["x_col"] = x_col
        summary["effect_col"] = effect_col
    return work, summary


def plot_effect_panel(ax, n_ax, unit: pd.DataFrame, summary: pd.DataFrame, x_col: str, feature: str,
                      effect_col: str, ylabel: str, null_value: float, xlabel: str, args, rng):
    values_for_ylim = []
    if not unit.empty and args.show_unit_points:
        show = unit
        if args.max_unit_points_per_panel and show.shape[0] > args.max_unit_points_per_panel:
            show = show.sample(n=args.max_unit_points_per_panel, random_state=int(rng.integers(0, 2**32 - 1)))
        x = show[x_col].to_numpy(dtype=float)
        if args.jitter > 0:
            x = x + rng.uniform(-args.jitter, args.jitter, size=x.size)
        y = show[effect_col].to_numpy(dtype=float)
        values_for_ylim.extend(y[np.isfinite(y)].tolist())
        ax.scatter(x, y, s=args.point_size, alpha=args.point_alpha, color=UNIT_COLOR,
                   edgecolors="none", label="animal×label paired values", rasterized=True)

    sline = summary[summary["passes_min_n_filter"] == True].copy() if not summary.empty else pd.DataFrame()
    if not sline.empty:
        xx = sline[x_col].to_numpy(dtype=float)
        yy = sline["mean"].to_numpy(dtype=float)
        values_for_ylim.extend(yy[np.isfinite(yy)].tolist())
        ax.plot(xx, yy, color=EFFECT_COLOR, linewidth=args.line_width, marker="o", markersize=args.marker_size,
                label="mean paired effect")
        if args.show_sem_band:
            ee = sline["sem"].to_numpy(dtype=float)
            ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
            if np.any(ok):
                ax.fill_between(xx[ok], yy[ok] - ee[ok], yy[ok] + ee[ok], color=EFFECT_COLOR, alpha=SEM_ALPHA, linewidth=0)

    ax.axhline(null_value, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{FEATURE_LABELS.get(feature, feature)}")
    ax.grid(alpha=0.25)
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    lower_bound = 0.0 if effect_col == "post_pre_ratio" else None
    set_robust_ylim(ax, values_for_ylim, null_value, args, lower_bound=lower_bound)
    dedupe_legend(ax)

    counts = summary[[x_col, "n_units", "n_animals", "passes_min_n_filter"]].copy() if not summary.empty else pd.DataFrame()
    plot_n_panel(n_ax, counts, x_col=x_col, kind="paired")
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        n_ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    n_ax.set_xlabel(xlabel, fontsize=8)


def make_grid_with_n(ncols: int, figsize: tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, height_ratios=[4.0, 0.8], hspace=0.10, wspace=0.25)
    main_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    n_axes = [fig.add_subplot(gs[1, i], sharex=main_axes[i]) for i in range(ncols)]
    return fig, main_axes, n_axes


def make_normalized_feature_figure(specs, segments_for_counts, out_png: Path, title: str, args, rng, all_norm_units, all_norm_summaries):
    fig, axes, n_axes = make_grid_with_n(len(specs), figsize=(5.2 * len(specs), 6.3))
    for ax, n_ax, spec in zip(axes, n_axes, specs):
        df, x_col, feature, xlabel = spec
        unit = all_norm_units[(all_norm_units["feature"] == feature) & (all_norm_units["x_col"] == x_col)].copy()
        summary = all_norm_summaries[(all_norm_summaries["feature"] == feature) & (all_norm_summaries["x_col"] == x_col)].copy()
        plot_normalized_panel(ax, n_ax, unit, summary, x_col, feature, xlabel, args, rng)
    add_suptitle_with_counts(fig, segments_for_counts, title)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_effect_feature_figure(specs, paired_all, segments_for_counts, out_png: Path, title: str,
                               effect_col: str, ylabel: str, null_value: float, args, rng,
                               all_effect_summaries: list[pd.DataFrame]):
    fig, axes, n_axes = make_grid_with_n(len(specs), figsize=(5.2 * len(specs), 6.3))
    for ax, n_ax, spec in zip(axes, n_axes, specs):
        _df, x_col, feature, xlabel = spec
        unit, summary = effect_unit_and_summary(paired_all, x_col, feature, effect_col, null_value, args)
        if not summary.empty:
            all_effect_summaries.append(summary)
        plot_effect_panel(ax, n_ax, unit, summary, x_col, feature, effect_col, ylabel, null_value, xlabel, args, rng)
    add_suptitle_with_counts(fig, segments_for_counts, title)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_figures(segments: pd.DataFrame, out_dir: Path, args):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    segments = segments.copy()
    for c in [
        "repeat_index_in_phrase", "segment_duration_ms", "preceding_isi_ms", "preceding_ioi_ms",
        "cumulative_phrase_time_s", "phrase_duration_s", "n_segments_in_phrase"
    ]:
        if c in segments.columns:
            segments[c] = finite_numeric(segments[c])

    phrase = build_phrase_table(segments)
    if args.max_total_repeats is not None:
        phrase = phrase[phrase["n_segments_in_phrase"] <= args.max_total_repeats].copy()
    phrase_for_plot = phrase.rename(columns={"n_segments_in_phrase": "repeat_index_in_phrase"})

    repeat_specs = [
        (segments, "repeat_index_in_phrase", "segment_duration_ms", "Syllable repeat index within phrase"),
        (segments, "repeat_index_in_phrase", "preceding_isi_ms", "Syllable repeat index within phrase"),
        (segments, "repeat_index_in_phrase", "preceding_ioi_ms", "Syllable repeat index within phrase"),
    ]
    time_specs = [
        (segments, "repeat_index_in_phrase", "cumulative_phrase_time_s", "Syllable repeat index within phrase"),
        (phrase_for_plot, "repeat_index_in_phrase", "phrase_duration_s", "Total segmented repeats in phrase"),
    ]
    all_specs = repeat_specs + time_specs

    # Pre-baseline QC table.
    pre_baseline_qc = []
    norm_units = []
    norm_summaries = []
    for df, x_col, feature, _xlabel in all_specs:
        unit, summary, baseline = normalized_unit_and_summary(df, x_col, feature, args)
        if not unit.empty:
            norm_units.append(unit)
        if not summary.empty:
            norm_summaries.append(summary)
        if not baseline.empty:
            pre_baseline_qc.append(baseline)
    norm_units_all = pd.concat(norm_units, ignore_index=True) if norm_units else pd.DataFrame()
    norm_summaries_all = pd.concat(norm_summaries, ignore_index=True) if norm_summaries else pd.DataFrame()
    pre_baseline_qc_all = pd.concat(pre_baseline_qc, ignore_index=True) if pre_baseline_qc else pd.DataFrame()

    save_csv(pre_baseline_qc_all, out_dir / "pre_baseline_qc_by_animal_label_feature.csv")
    save_csv(norm_units_all, out_dir / "normalized_to_pre_mean_unit_values.csv")
    save_csv(norm_summaries_all, out_dir / "normalized_to_pre_mean_repeat_summaries.csv")
    if not norm_summaries_all.empty:
        save_csv(norm_summaries_all[["feature", "x_col", "pre_post", "repeat_index_in_phrase", "n_units", "n_animals", "n_animal_label_pairs", "passes_min_n_filter"]].copy(),
                 out_dir / "bin_counts_normalized.csv")

    # Paired post/pre tables.
    paired_tables = []
    for df, x_col, feature, _xlabel in all_specs:
        paired = paired_post_pre_by_repeat(df, x_col=x_col, y_col=feature, args=args)
        if not paired.empty:
            paired_tables.append(paired)
    paired_all = pd.concat(paired_tables, ignore_index=True) if paired_tables else pd.DataFrame()
    save_csv(paired_all, out_dir / "paired_post_pre_by_animal_label_repeat_filtered.csv")

    # Normalized pre/post figures with n panels.
    make_normalized_feature_figure(
        repeat_specs, segments, fig_dir / "normalized_duration_gap_ioi_with_n.png",
        "Normalized syllable duration, end-to-start ISI, and inter-onset interval",
        args, rng, norm_units_all, norm_summaries_all,
    )
    make_normalized_feature_figure(
        time_specs, segments, fig_dir / "normalized_time_panels_with_n.png",
        "Normalized cumulative phrase time and total repeated-label phrase duration",
        args, rng, norm_units_all, norm_summaries_all,
    )

    effect_defs = [
        ("log2_post_pre_ratio", "log2(post / pre)", 0.0, "post_pre_log2_ratio_duration_gap_ioi_with_n.png", "log2 post/pre ratio: duration, ISI, and IOI"),
        ("normalized_difference_post_minus_pre_over_sum", "(post - pre) / (post + pre)", 0.0, "post_pre_normalized_difference_duration_gap_ioi_with_n.png", "Normalized post-pre difference: duration, ISI, and IOI"),
        ("post_pre_ratio", "Post / pre ratio", 1.0, "post_pre_ratio_duration_gap_ioi_with_n.png", "Post/pre ratio: duration, ISI, and IOI"),
        ("percent_change_post_vs_pre", "% change post vs pre", 0.0, "post_pre_percent_change_duration_gap_ioi_with_n.png", "Percent change post vs pre: duration, ISI, and IOI"),
    ]
    effect_summaries = []
    for effect_col, ylabel, null_value, filename, title in effect_defs:
        make_effect_feature_figure(
            repeat_specs, paired_all, segments, fig_dir / filename, title,
            effect_col, ylabel, null_value, args, rng, effect_summaries,
        )

    # Also create paired time effect figures for the most useful metrics.
    for effect_col, ylabel, null_value, filename, title in [
        ("log2_post_pre_ratio", "log2(post / pre)", 0.0, "post_pre_log2_ratio_time_panels_with_n.png", "log2 post/pre ratio: cumulative time and phrase duration"),
        ("normalized_difference_post_minus_pre_over_sum", "(post - pre) / (post + pre)", 0.0, "post_pre_normalized_difference_time_panels_with_n.png", "Normalized post-pre difference: cumulative time and phrase duration"),
        ("post_pre_ratio", "Post / pre ratio", 1.0, "post_pre_ratio_time_panels_with_n.png", "Post/pre ratio: cumulative time and phrase duration"),
    ]:
        make_effect_feature_figure(
            time_specs, paired_all, segments, fig_dir / filename, title,
            effect_col, ylabel, null_value, args, rng, effect_summaries,
        )

    effect_summary_all = pd.concat(effect_summaries, ignore_index=True) if effect_summaries else pd.DataFrame()
    save_csv(effect_summary_all, out_dir / "paired_effect_repeat_summaries.csv")
    if not effect_summary_all.empty:
        cols = [c for c in ["feature", "x_col", "effect_col", "repeat_index_in_phrase", "n_units", "n_animals", "n_animal_label_pairs", "passes_min_n_filter"] if c in effect_summary_all.columns]
        save_csv(effect_summary_all[cols].copy(), out_dir / "bin_counts_paired_effects.csv")

    phrase.to_csv(out_dir / "combined_phrase_rows_used.csv", index=False)
    print(f"[SAVED] {out_dir / 'combined_phrase_rows_used.csv'}")


def main():
    parser = argparse.ArgumentParser(description="QC-aware normalized duration/ISI/IOI figures for top-30 phrase-duration-variance syllables.")
    parser.add_argument("--timing-root", required=True, help="Root containing per-bird timing output folders")
    parser.add_argument("--out-dir", required=True, help="Output directory for QC-aware normalized figures and summary CSVs")
    parser.add_argument("--top30-labels-csv", default=None, help="Optional animal x label CSV to filter to top-30 labels")
    parser.add_argument("--animal-col", default=None, help="Animal ID column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--label-col", default=None, help="Label column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--animals", default=None, help="Optional comma-separated animal IDs to include")
    parser.add_argument("--max-repeat-index", type=int, default=None, help="Optional maximum repeat index to plot for segment-level repeat-index panels")
    parser.add_argument("--max-total-repeats", type=int, default=None, help="Optional maximum total repeats to plot for phrase-duration panels")

    parser.add_argument("--min-raw-values-per-unit-bin", type=int, default=1,
                        help="Minimum raw segment/phrase rows within one animal x label x pre/post x repeat bin")
    parser.add_argument("--min-units-per-bin", type=int, default=5,
                        help="Minimum animal-label units needed to draw an aggregate mean point")
    parser.add_argument("--min-animals-per-bin", type=int, default=3,
                        help="Minimum birds needed to draw an aggregate mean point")

    parser.add_argument("--min-pre-mean-ms-for-isi", type=float, default=5.0,
                        help="Minimum pre-lesion mean end-to-start ISI in ms for normalization and paired ratios")
    parser.add_argument("--min-pre-mean-ms-for-ioi", type=float, default=10.0,
                        help="Minimum pre-lesion mean inter-onset interval in ms for normalization and paired ratios")
    parser.add_argument("--min-pre-mean-ms-for-duration", type=float, default=1.0,
                        help="Minimum pre-lesion mean syllable duration in ms for normalization and paired ratios")
    parser.add_argument("--min-pre-mean-s-for-time-features", type=float, default=0.001,
                        help="Minimum pre-lesion mean in seconds for cumulative/phrase time normalizations")

    parser.add_argument("--robust-ylim", action="store_true", default=True,
                        help="Use percentile-based robust y-limits for plots")
    parser.add_argument("--no-robust-ylim", dest="robust_ylim", action="store_false")
    parser.add_argument("--robust-ylim-low", type=float, default=1.0,
                        help="Lower percentile for robust y-limits")
    parser.add_argument("--robust-ylim-high", type=float, default=99.0,
                        help="Upper percentile for robust y-limits")

    parser.add_argument("--show-unit-points", action="store_true", default=True,
                        help="Show transparent animal x label points")
    parser.add_argument("--hide-unit-points", dest="show_unit_points", action="store_false")
    parser.add_argument("--max-unit-points-per-panel", type=int, default=25000,
                        help="Downsample transparent unit points per panel. Use 0 for no limit.")
    parser.add_argument("--jitter", type=float, default=0.06,
                        help="Small x-jitter for transparent points so vertical stacks are easier to see")
    parser.add_argument("--show-sem-band", action="store_true",
                        help="Add faint SEM bands around mean lines")

    parser.add_argument("--point-alpha", type=float, default=POINT_ALPHA)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    parser.add_argument("--line-width", type=float, default=LINE_WIDTH)
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()

    if args.max_unit_points_per_panel == 0:
        args.max_unit_points_per_panel = None
    if not (0 <= args.robust_ylim_low < args.robust_ylim_high <= 100):
        raise ValueError("Require 0 <= --robust-ylim-low < --robust-ylim-high <= 100")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, manifest = read_segments(args)
    segments = filter_segments(segments, args)
    required = [
        "repeat_index_in_phrase", "segment_duration_ms", "preceding_isi_ms",
        "cumulative_phrase_time_s", "phrase_duration_s", "n_segments_in_phrase", "phrase_id"
    ]
    missing = [c for c in required if c not in segments.columns]
    if missing:
        raise ValueError(f"Timing CSVs are missing required columns: {missing}")

    segments = compute_preceding_ioi_ms(segments)

    manifest.to_csv(out_dir / "combined_figure_input_files_used.csv", index=False)
    segments.to_csv(out_dir / "combined_segments_used_with_ioi_and_qc.csv", index=False)
    print(f"[SAVED] {out_dir / 'combined_segments_used_with_ioi_and_qc.csv'}")

    n_animals = segments["animal_id"].nunique()
    n_pairs = segments[["animal_id", "label"]].drop_duplicates().shape[0]
    print(f"[INFO] Combined data: {n_animals} birds, {n_pairs} animal x label pairs, {segments.shape[0]:,} segment rows")
    print(f"[INFO] ISI pre-denominator filter: >= {args.min_pre_mean_ms_for_isi} ms")
    print(f"[INFO] IOI pre-denominator filter: >= {args.min_pre_mean_ms_for_ioi} ms")
    print(f"[INFO] Aggregate bins require >= {args.min_units_per_bin} animal-label units and >= {args.min_animals_per_bin} birds")

    make_figures(segments, out_dir, args)
    print(f"[DONE] QC-aware normalized figures and summaries saved to {out_dir}")


if __name__ == "__main__":
    main()
