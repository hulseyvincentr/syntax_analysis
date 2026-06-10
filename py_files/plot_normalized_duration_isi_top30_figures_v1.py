#!/usr/bin/env python3
"""
Make normalized combined duration / ISI phrase-position figures across birds.

Designed for outputs from:
    phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py

Typical input structure after batch running:
    <timing-root>/<ANIMAL>/<ANIMAL>_all_labels_duration_isi_segment_timing.csv

Why this script exists:
    Raw syllable durations are hard to combine across syllables because some
    syllables are naturally short and others are naturally long. This script
    normalizes timing values within each animal x label before combining birds.

Main outputs:
    1) normalized_duration_and_isi_vs_repeat_index.png
       - plots pre/post trajectories after dividing by each animal x label's
         own pre-lesion mean.
       - y = 1 means equal to that syllable's own pre-lesion baseline.

    2) post_pre_ratio_duration_and_isi_vs_repeat_index.png
       - paired post/pre ratio at each repeat index.
       - y = 1 means post equals pre at that repeat index.
       - y > 1 means longer post-lesion.
       - y < 1 means shorter post-lesion.

    3) post_pre_log2_ratio_duration_and_isi_vs_repeat_index.png
       - log2(post/pre) at each repeat index.
       - y = 0 means post equals pre.
       - y = 1 means post is 2x pre.
       - y = -1 means post is 1/2 pre.

    4) post_pre_normalized_difference_duration_and_isi_vs_repeat_index.png
       - (post - pre) / (post + pre) at each repeat index.
       - y = 0 means post equals pre.
       - positive means post is longer.
       - negative means post is shorter.
       - bounded between -1 and +1 for positive durations/ISIs.

The paired post/pre plots first compute means within animal x label x repeat index
for pre and post separately, then pair those values. This avoids comparing unlike
syllables directly.

Plotting convention:
    - pre-lesion: blue
    - post-lesion: red
    - normalized-effect plots: black/gray summary with individual animal-label
      unit points in gray
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRE_COLOR = "#1f77b4"   # blue
POST_COLOR = "#d62728"  # red
EFFECT_COLOR = "#333333"
UNIT_COLOR = "#777777"
POINT_ALPHA = 0.22
POINT_SIZE = 16
LINE_WIDTH = 2.8
MARKER_SIZE = 5.5
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

DEFAULT_SEGMENT_FEATURES = ["segment_duration_ms", "preceding_isi_ms"]
DEFAULT_CUMULATIVE_FEATURES = ["cumulative_phrase_time_s"]
DEFAULT_PHRASE_FEATURES = ["phrase_duration_s"]

FEATURE_LABELS = {
    "segment_duration_ms": "Syllable duration",
    "preceding_isi_ms": "Preceding inter-syllabic interval",
    "following_isi_ms": "Following inter-syllabic interval",
    "cumulative_phrase_time_s": "Cumulative phrase time",
    "cumulative_syllable_duration_s": "Cumulative syllable duration only",
    "phrase_duration_s": "Total repeated-label phrase duration",
}

FEATURE_UNITS = {
    "segment_duration_ms": "ms",
    "preceding_isi_ms": "ms",
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


def finite_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


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


def build_phrase_table(segments: pd.DataFrame) -> pd.DataFrame:
    required = ["animal_id", "label", "pre_post", "phrase_id", "phrase_duration_s", "n_segments_in_phrase"]
    missing = [c for c in required if c not in segments.columns]
    if missing:
        raise ValueError(f"Cannot build phrase table; missing columns: {missing}")
    group_cols = ["animal_id", "label", "pre_post", "phrase_id"]
    for extra in ["period", "recording_key", "file_index", "file_indices_stitched"]:
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


def summarize_units(unit: pd.DataFrame, x_col: str, value_col: str, group_col: str = "pre_post", min_units: int = 3) -> pd.DataFrame:
    rows = []
    if unit.empty:
        return pd.DataFrame()
    for (grp, x), sub in unit.groupby([group_col, x_col], dropna=False):
        vals = finite_numeric(sub[value_col]).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < min_units:
            continue
        rows.append({
            group_col: grp,
            x_col: int(x),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "sem": sem(vals),
            "n_units": int(vals.size),
            "n_animals": int(sub["animal_id"].nunique()) if "animal_id" in sub.columns else np.nan,
            "n_labels": int(sub[["animal_id", "label"]].drop_duplicates().shape[0]) if {"animal_id", "label"}.issubset(sub.columns) else np.nan,
        })
    return pd.DataFrame(rows).sort_values([group_col, x_col]) if rows else pd.DataFrame()


def add_within_label_pre_baseline(df: pd.DataFrame, y_col: str, normalized_col: str) -> pd.DataFrame:
    """Divide each value by that animal x label's overall pre-lesion mean for this feature."""
    work = df.copy()
    work[y_col] = finite_numeric(work[y_col])
    pre = work[(work["pre_post"] == "pre") & np.isfinite(work[y_col])].copy()
    baseline = (
        pre.groupby(["animal_id", "label"], as_index=False)
        .agg(pre_baseline=(y_col, "mean"), n_pre_baseline_values=(y_col, "size"))
    )
    work = work.merge(baseline, on=["animal_id", "label"], how="left")
    ok = np.isfinite(work[y_col]) & np.isfinite(work["pre_baseline"]) & (work["pre_baseline"] > EPS)
    work[normalized_col] = np.nan
    work.loc[ok, normalized_col] = work.loc[ok, y_col] / work.loc[ok, "pre_baseline"]
    return work


def paired_post_pre_by_repeat(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_raw_values_per_unit_bin: int = 1,
) -> pd.DataFrame:
    """Compute paired post/pre values within animal x label x repeat bin."""
    unit = make_unit_means(df, x_col=x_col, y_col=y_col, min_raw_values_per_unit_bin=min_raw_values_per_unit_bin)
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
    wide = wide[(wide["pre"] > EPS) & (wide["post"] >= 0)].copy()
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


def group_style(group: str) -> tuple[str, str]:
    group = str(group).lower()
    if group == "post":
        return POST_COLOR, "post"
    return PRE_COLOR, "pre"


def plot_normalized_pre_post(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    feature: str,
    x_label: str,
    args,
    rng: np.random.Generator,
) -> pd.DataFrame:
    norm_col = f"{y_col}_normalized_to_pre_mean"
    work = add_within_label_pre_baseline(df, y_col=y_col, normalized_col=norm_col)
    unit = make_unit_means(
        work,
        x_col=x_col,
        y_col=norm_col,
        min_raw_values_per_unit_bin=args.min_raw_values_per_unit_bin,
    )
    summary = summarize_units(
        unit,
        x_col=x_col,
        value_col="unit_mean",
        group_col="pre_post",
        min_units=args.min_units_per_bin,
    )
    unit["feature"] = feature
    summary["feature"] = feature if not summary.empty else feature

    for group in ["pre", "post"]:
        sub = unit[unit["pre_post"] == group].copy()
        if sub.empty:
            continue
        color, label = group_style(group)
        if args.show_unit_points:
            show = sub
            if args.max_unit_points_per_group and show.shape[0] > args.max_unit_points_per_group:
                show = show.sample(n=args.max_unit_points_per_group, random_state=int(rng.integers(0, 2**32 - 1)))
            x = show[x_col].to_numpy(dtype=float)
            if args.jitter > 0:
                x = x + rng.uniform(-args.jitter, args.jitter, size=x.size)
            ax.scatter(
                x,
                show["unit_mean"].to_numpy(dtype=float),
                s=args.point_size,
                alpha=args.point_alpha,
                color=color,
                edgecolors="none",
                label=f"{label} animal×label means",
                rasterized=True,
            )
        sline = summary[summary["pre_post"] == group].copy() if not summary.empty else pd.DataFrame()
        if not sline.empty:
            xx = sline[x_col].to_numpy(dtype=float)
            yy = sline["mean"].to_numpy(dtype=float)
            ax.plot(
                xx, yy,
                color=color,
                linewidth=args.line_width,
                marker="o",
                markersize=args.marker_size,
                label=f"{label} mean of animal×label means",
            )
            if args.show_sem_band and "sem" in sline.columns:
                ee = sline["sem"].to_numpy(dtype=float)
                ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
                if np.any(ok):
                    ax.fill_between(xx[ok], yy[ok] - ee[ok], yy[ok] + ee[ok], color=color, alpha=SEM_ALPHA, linewidth=0)

    ax.axhline(1.0, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel(x_label)
    units = FEATURE_UNITS.get(feature, "")
    ax.set_ylabel(f"{FEATURE_LABELS.get(feature, feature)} / animal×label pre mean")
    ax.set_title(f"{FEATURE_LABELS.get(feature, feature)} normalized to each syllable's pre mean")
    ax.grid(alpha=0.25)
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    dedupe_legend(ax)
    return summary


def plot_effect_metric(
    ax,
    paired: pd.DataFrame,
    x_col: str,
    effect_col: str,
    feature: str,
    x_label: str,
    y_label: str,
    null_value: float,
    args,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if paired.empty:
        ax.text(0.5, 0.5, "No paired pre/post bins", ha="center", va="center", transform=ax.transAxes)
        return pd.DataFrame()

    work = paired.copy()
    work[effect_col] = finite_numeric(work[effect_col])
    work = work[np.isfinite(work[effect_col])].copy()
    if work.empty:
        ax.text(0.5, 0.5, "No finite paired pre/post values", ha="center", va="center", transform=ax.transAxes)
        return pd.DataFrame()

    if args.show_unit_points:
        show = work
        if args.max_unit_points_per_group and show.shape[0] > args.max_unit_points_per_group:
            show = show.sample(n=args.max_unit_points_per_group, random_state=int(rng.integers(0, 2**32 - 1)))
        x = show[x_col].to_numpy(dtype=float)
        if args.jitter > 0:
            x = x + rng.uniform(-args.jitter, args.jitter, size=x.size)
        ax.scatter(
            x,
            show[effect_col].to_numpy(dtype=float),
            s=args.point_size,
            alpha=args.point_alpha,
            color=UNIT_COLOR,
            edgecolors="none",
            label="animal×label paired values",
            rasterized=True,
        )

    summary = summarize_units(
        work.rename(columns={effect_col: "effect_value"}),
        x_col=x_col,
        value_col="effect_value",
        group_col="_dummy_group",
        min_units=args.min_units_per_bin,
    ) if False else None

    # summarize_units expects a group column; do it directly here for clarity.
    rows = []
    for x, sub in work.groupby(x_col):
        vals = sub[effect_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < args.min_units_per_bin:
            continue
        rows.append({
            x_col: int(x),
            "feature": feature,
            "effect_col": effect_col,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "sem": sem(vals),
            "n_units": int(vals.size),
            "n_animals": int(sub["animal_id"].nunique()),
            "n_labels": int(sub[["animal_id", "label"]].drop_duplicates().shape[0]),
        })
    summary_df = pd.DataFrame(rows).sort_values(x_col) if rows else pd.DataFrame()

    if not summary_df.empty:
        xx = summary_df[x_col].to_numpy(dtype=float)
        yy = summary_df["mean"].to_numpy(dtype=float)
        ax.plot(
            xx, yy,
            color=EFFECT_COLOR,
            linewidth=args.line_width,
            marker="o",
            markersize=args.marker_size,
            label="mean paired effect",
        )
        if args.show_sem_band:
            ee = summary_df["sem"].to_numpy(dtype=float)
            ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
            if np.any(ok):
                ax.fill_between(xx[ok], yy[ok] - ee[ok], yy[ok] + ee[ok], color=EFFECT_COLOR, alpha=SEM_ALPHA, linewidth=0)

    ax.axhline(null_value, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{FEATURE_LABELS.get(feature, feature)}: post vs pre")
    ax.grid(alpha=0.25)
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        ax.set_xlim(0.5, args.max_repeat_index + 0.5)
    dedupe_legend(ax)
    return summary_df


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


def add_suptitle_with_counts(fig, segments: pd.DataFrame, title: str):
    n_animals = segments["animal_id"].nunique()
    n_labels = segments[["animal_id", "label"]].drop_duplicates().shape[0]
    n_segments = segments.shape[0]
    fig.suptitle(f"{title}\n{n_animals} birds, {n_labels} animal×label pairs, {n_segments:,} segmented syllables", fontsize=14)


def save_summary(df: pd.DataFrame, path: Path):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        print(f"[SAVED] {path}")


def make_normalized_figures(segments: pd.DataFrame, out_dir: Path, args):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    # Ensure key numeric columns.
    segments = segments.copy()
    for c in ["repeat_index_in_phrase", "segment_duration_ms", "preceding_isi_ms", "cumulative_phrase_time_s"]:
        if c in segments.columns:
            segments[c] = finite_numeric(segments[c])

    phrase = build_phrase_table(segments)
    if args.max_total_repeats is not None:
        phrase = phrase[phrase["n_segments_in_phrase"] <= args.max_total_repeats].copy()
    phrase_for_plot = phrase.rename(columns={"n_segments_in_phrase": "repeat_index_in_phrase"})

    # A. Values normalized to each animal x label's own pre mean.
    normalized_summaries = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.2), squeeze=False)
    normalized_summaries.append(plot_normalized_pre_post(
        axes[0, 0], segments, "repeat_index_in_phrase", "segment_duration_ms", "segment_duration_ms",
        "Syllable repeat index within phrase", args, rng,
    ))
    normalized_summaries.append(plot_normalized_pre_post(
        axes[0, 1], segments, "repeat_index_in_phrase", "preceding_isi_ms", "preceding_isi_ms",
        "Syllable repeat index within phrase", args, rng,
    ))
    normalized_summaries.append(plot_normalized_pre_post(
        axes[1, 0], segments, "repeat_index_in_phrase", "cumulative_phrase_time_s", "cumulative_phrase_time_s",
        "Syllable repeat index within phrase", args, rng,
    ))
    normalized_summaries.append(plot_normalized_pre_post(
        axes[1, 1], phrase_for_plot, "repeat_index_in_phrase", "phrase_duration_s", "phrase_duration_s",
        "Total segmented repeats in phrase", args, rng,
    ))
    add_suptitle_with_counts(fig, segments, "Top-30% phrase-duration-variance timing normalized within each syllable")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_png = fig_dir / "normalized_duration_isi_time_four_panel.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    norm_all = pd.concat([x for x in normalized_summaries if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in normalized_summaries) else pd.DataFrame()
    save_summary(norm_all, out_dir / "normalized_to_pre_mean_repeat_summaries.csv")

    # B. Paired post/pre effect metrics by repeat index.
    paired_specs = [
        (segments, "repeat_index_in_phrase", "segment_duration_ms", "Syllable repeat index within phrase"),
        (segments, "repeat_index_in_phrase", "preceding_isi_ms", "Syllable repeat index within phrase"),
        (segments, "repeat_index_in_phrase", "cumulative_phrase_time_s", "Syllable repeat index within phrase"),
        (phrase_for_plot, "repeat_index_in_phrase", "phrase_duration_s", "Total segmented repeats in phrase"),
    ]
    paired_tables = []
    for source_df, x_col, feature, _ in paired_specs:
        paired = paired_post_pre_by_repeat(
            source_df,
            x_col=x_col,
            y_col=feature,
            min_raw_values_per_unit_bin=args.min_raw_values_per_unit_bin,
        )
        if not paired.empty:
            paired["feature"] = feature
            paired["x_col"] = x_col
            paired_tables.append(paired)
    paired_all = pd.concat(paired_tables, ignore_index=True) if paired_tables else pd.DataFrame()
    save_summary(paired_all, out_dir / "paired_post_pre_by_animal_label_repeat.csv")

    effect_plot_defs = [
        (
            "post_pre_ratio",
            "Post / pre ratio",
            1.0,
            "post_pre_ratio_duration_isi_time_four_panel.png",
            "Post/pre ratio by repeat index",
        ),
        (
            "log2_post_pre_ratio",
            "log2(post / pre)",
            0.0,
            "post_pre_log2_ratio_duration_isi_time_four_panel.png",
            "log2 post/pre ratio by repeat index",
        ),
        (
            "normalized_difference_post_minus_pre_over_sum",
            "(post - pre) / (post + pre)",
            0.0,
            "post_pre_normalized_difference_duration_isi_time_four_panel.png",
            "Normalized post-pre difference by repeat index",
        ),
        (
            "percent_change_post_vs_pre",
            "% change post vs pre",
            0.0,
            "post_pre_percent_change_duration_isi_time_four_panel.png",
            "Percent change post vs pre by repeat index",
        ),
    ]

    for effect_col, ylabel, null_value, filename, title in effect_plot_defs:
        summaries = []
        fig, axes = plt.subplots(2, 2, figsize=(14, 10.2), squeeze=False)
        for ax, (source_df, x_col, feature, xlabel) in zip(axes.ravel(), paired_specs):
            paired = paired_all[(paired_all["feature"] == feature) & (paired_all["x_col"] == x_col)].copy() if not paired_all.empty else pd.DataFrame()
            summaries.append(plot_effect_metric(
                ax,
                paired,
                x_col=x_col,
                effect_col=effect_col,
                feature=feature,
                x_label=xlabel,
                y_label=ylabel,
                null_value=null_value,
                args=args,
                rng=rng,
            ))
        add_suptitle_with_counts(fig, segments, title)
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        out_png = fig_dir / filename
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")
        summary_all = pd.concat([s for s in summaries if s is not None and not s.empty], ignore_index=True) if any(s is not None and not s.empty for s in summaries) else pd.DataFrame()
        save_summary(summary_all, out_dir / f"summary_{safe_name(effect_col)}.csv")

    # C. Compact two-panel versions focused only on duration and ISI.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    plot_normalized_pre_post(
        axes[0, 0], segments, "repeat_index_in_phrase", "segment_duration_ms", "segment_duration_ms",
        "Syllable repeat index within phrase", args, rng,
    )
    plot_normalized_pre_post(
        axes[0, 1], segments, "repeat_index_in_phrase", "preceding_isi_ms", "preceding_isi_ms",
        "Syllable repeat index within phrase", args, rng,
    )
    add_suptitle_with_counts(fig, segments, "Normalized syllable duration and ISI")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = fig_dir / "normalized_duration_and_isi_vs_repeat_index.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    for effect_col, ylabel, null_value, filename, title in [
        ("post_pre_ratio", "Post / pre ratio", 1.0, "post_pre_ratio_duration_and_isi_vs_repeat_index.png", "Post/pre ratio: duration and ISI"),
        ("log2_post_pre_ratio", "log2(post / pre)", 0.0, "post_pre_log2_ratio_duration_and_isi_vs_repeat_index.png", "log2 post/pre ratio: duration and ISI"),
        ("normalized_difference_post_minus_pre_over_sum", "(post - pre) / (post + pre)", 0.0, "post_pre_normalized_difference_duration_and_isi_vs_repeat_index.png", "Normalized post-pre difference: duration and ISI"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
        for ax, feature in zip(axes.ravel(), ["segment_duration_ms", "preceding_isi_ms"]):
            paired = paired_all[(paired_all["feature"] == feature) & (paired_all["x_col"] == "repeat_index_in_phrase")].copy() if not paired_all.empty else pd.DataFrame()
            plot_effect_metric(
                ax, paired, "repeat_index_in_phrase", effect_col, feature,
                "Syllable repeat index within phrase", ylabel, null_value, args, rng,
            )
        add_suptitle_with_counts(fig, segments, title)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_png = fig_dir / filename
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")

    phrase.to_csv(out_dir / "combined_phrase_rows_used.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Make normalized combined top-30 duration/ISI phrase-position figures across birds.")
    parser.add_argument("--timing-root", required=True, help="Root containing per-bird timing output folders")
    parser.add_argument("--out-dir", required=True, help="Output directory for normalized combined figures and summary CSVs")
    parser.add_argument("--top30-labels-csv", default=None, help="Optional animal x label CSV to filter to top-30 labels")
    parser.add_argument("--animal-col", default=None, help="Animal ID column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--label-col", default=None, help="Label column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--animals", default=None, help="Optional comma-separated animal IDs to include")
    parser.add_argument("--max-repeat-index", type=int, default=None, help="Optional maximum repeat index to plot for segment-level repeat-index panels")
    parser.add_argument("--max-total-repeats", type=int, default=None, help="Optional maximum total repeats to plot for phrase-duration panels")

    parser.add_argument("--min-raw-values-per-unit-bin", type=int, default=1,
                        help="Minimum raw segment/phrase rows within one animal x label x pre/post x repeat bin")
    parser.add_argument("--min-units-per-bin", type=int, default=3,
                        help="Minimum animal-label units needed to draw a mean point")
    parser.add_argument("--show-unit-points", action="store_true", default=True,
                        help="Show transparent animal x label points")
    parser.add_argument("--hide-unit-points", dest="show_unit_points", action="store_false")
    parser.add_argument("--max-unit-points-per-group", type=int, default=25000,
                        help="Downsample transparent unit points per group/panel. Use 0 for no limit.")
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

    if args.max_unit_points_per_group == 0:
        args.max_unit_points_per_group = None

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, manifest = read_segments(args)
    segments = filter_segments(segments, args)

    required = [
        "repeat_index_in_phrase", "segment_duration_ms", "preceding_isi_ms",
        "cumulative_phrase_time_s", "phrase_duration_s", "n_segments_in_phrase"
    ]
    missing = [c for c in required if c not in segments.columns]
    if missing:
        raise ValueError(f"Timing CSVs are missing required columns: {missing}")

    manifest.to_csv(out_dir / "combined_figure_input_files_used.csv", index=False)
    segments.to_csv(out_dir / "combined_segments_used.csv", index=False)

    n_animals = segments["animal_id"].nunique()
    n_pairs = segments[["animal_id", "label"]].drop_duplicates().shape[0]
    print(f"[INFO] Combined data: {n_animals} birds, {n_pairs} animal x label pairs, {segments.shape[0]:,} segment rows")
    make_normalized_figures(segments, out_dir, args)
    print(f"[DONE] Normalized combined figures and summaries saved to {out_dir}")


if __name__ == "__main__":
    main()
