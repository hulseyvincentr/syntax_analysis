#!/usr/bin/env python3
"""
Make combined duration / ISI phrase-position figures across birds.

Designed for outputs from:
    phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py

Typical input structure after batch running:
    <timing-root>/<ANIMAL>/<ANIMAL>_all_labels_duration_isi_segment_timing.csv

This script makes combined figures for the top 30% phrase-duration-variance
syllables, including:
    1) syllable duration vs repeat index
    2) preceding inter-syllabic interval (ISI) vs repeat index
    3) cumulative time spent in the repeated phrase vs repeat index
    4) total repeated-label phrase duration vs total segmented repeats

Plotting convention:
    - all pre-lesion data are blue
    - all post-lesion data are red
    - individual points are transparent
    - summary lines are the mean in the same color

Important aggregation note:
    By default, the mean line is not the simple mean of all segment rows.
    It first averages within each animal x label x repeat index, then averages
    those animal-label means. This reduces domination by birds/labels with many
    segmented syllables. Use --line-level row if you want raw row-level means.
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
POINT_ALPHA = 0.12
POINT_SIZE = 14
LINE_WIDTH = 2.8
MARKER_SIZE = 5.5
SEM_ALPHA = 0.12

ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal", "animal", "Bird", "bird", "bird_id", "AnimalID"
]
LABEL_COL_CANDIDATES = [
    "label", "Label", "syllable", "Syllable", "syllable_label", "Syllable label",
    "cluster", "Cluster", "cluster_id", "Cluster ID", "hdbscan_label", "HDBSCAN label",
    "mapped_syllable_order", "syllable_cluster", "Syllable cluster", "chosen_label", "selected_label"
]

FEATURE_LABELS = {
    "segment_duration_ms": "Syllable duration (ms)",
    "preceding_isi_ms": "Preceding inter-syllabic interval (ms)",
    "following_isi_ms": "Following inter-syllabic interval (ms)",
    "cumulative_phrase_time_s": "Cumulative phrase time (s)",
    "cumulative_syllable_duration_s": "Cumulative syllable duration only (s)",
    "phrase_duration_s": "Total repeated-label phrase duration (s)",
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

    # Fallback: use per-label files if all-label files do not exist.
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

    if args.max_repeat_index is not None:
        out = out[pd.to_numeric(out["repeat_index_in_phrase"], errors="coerce") <= args.max_repeat_index].copy()
    return out


def finite_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def make_line_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "pre_post",
    line_level: str = "animal_label",
    min_points_per_bin: int = 3,
) -> pd.DataFrame:
    """Summarize means for plot lines.

    line_level='animal_label' first averages within each animal x label x group x x-bin,
    then averages those unit means. This is the recommended default.
    line_level='row' averages raw rows directly.
    """
    work = df[["animal_id", "label", group_col, x_col, y_col]].copy()
    work[x_col] = finite_numeric(work[x_col])
    work[y_col] = finite_numeric(work[y_col])
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])].copy()
    if work.empty:
        return pd.DataFrame()

    # x values are repeat counts, so force integer-like values for grouping.
    work[x_col] = work[x_col].astype(int)

    if line_level == "row":
        rows = []
        for (grp, x), sub in work.groupby([group_col, x_col], dropna=False):
            vals = sub[y_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < min_points_per_bin:
                continue
            rows.append({
                group_col: grp,
                x_col: int(x),
                "mean": float(np.mean(vals)),
                "sem": sem(vals),
                "n_values": int(vals.size),
                "n_units": int(vals.size),
                "n_animals": int(sub["animal_id"].nunique()),
                "n_labels": int(sub[["animal_id", "label"]].drop_duplicates().shape[0]),
            })
        return pd.DataFrame(rows).sort_values([group_col, x_col]) if rows else pd.DataFrame()

    unit = (
        work.groupby(["animal_id", "label", group_col, x_col], as_index=False)
        .agg(unit_mean=(y_col, "mean"), n_values=(y_col, "size"))
    )
    rows = []
    for (grp, x), sub in unit.groupby([group_col, x_col], dropna=False):
        vals = sub["unit_mean"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < min_points_per_bin:
            continue
        rows.append({
            group_col: grp,
            x_col: int(x),
            "mean": float(np.mean(vals)),
            "sem": sem(vals),
            "n_units": int(vals.size),
            "n_values": int(sub["n_values"].sum()),
            "n_animals": int(sub["animal_id"].nunique()),
            "n_labels": int(sub[["animal_id", "label"]].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(rows).sort_values([group_col, x_col]) if rows else pd.DataFrame()


def sample_raw_points(sub: pd.DataFrame, max_points: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or sub.shape[0] <= max_points:
        return sub
    return sub.sample(n=max_points, random_state=int(rng.integers(0, 2**32 - 1)))


def group_style(group: str) -> tuple[str, str]:
    group = str(group).lower()
    if group == "post":
        return POST_COLOR, "post"
    return PRE_COLOR, "pre"


def plot_feature_vs_x(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    args,
    rng: np.random.Generator,
) -> pd.DataFrame:
    summary = make_line_summary(
        df,
        x_col=x_col,
        y_col=y_col,
        group_col="pre_post",
        line_level=args.line_level,
        min_points_per_bin=args.min_points_per_bin,
    )

    for group in ["pre", "post"]:
        sub = df[df["pre_post"] == group].copy()
        if sub.empty or x_col not in sub.columns or y_col not in sub.columns:
            continue
        sub[x_col] = finite_numeric(sub[x_col])
        sub[y_col] = finite_numeric(sub[y_col])
        sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])].copy()
        if sub.empty:
            continue
        if args.show_raw_points:
            show = sample_raw_points(sub, args.max_raw_points_per_group, rng)
            x = show[x_col].to_numpy(dtype=float)
            if args.jitter > 0:
                x = x + rng.uniform(-args.jitter, args.jitter, size=x.size)
            color, label = group_style(group)
            ax.scatter(
                x,
                show[y_col].to_numpy(dtype=float),
                s=args.point_size,
                alpha=args.point_alpha,
                color=color,
                edgecolors="none",
                label=f"{label} segments" if y_col != "phrase_duration_s" else f"{label} phrases",
                rasterized=True,
            )

        if not summary.empty:
            sline = summary[summary["pre_post"] == group].copy()
            if not sline.empty:
                color, label = group_style(group)
                xx = sline[x_col].to_numpy(dtype=float)
                yy = sline["mean"].to_numpy(dtype=float)
                ax.plot(
                    xx,
                    yy,
                    color=color,
                    linewidth=args.line_width,
                    marker="o",
                    markersize=args.marker_size,
                    label=f"{label} mean",
                )
                if args.show_sem_band and "sem" in sline.columns:
                    ee = sline["sem"].to_numpy(dtype=float)
                    ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
                    if np.any(ok):
                        ax.fill_between(xx[ok], yy[ok] - ee[ok], yy[ok] + ee[ok], color=color, alpha=SEM_ALPHA, linewidth=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if args.max_repeat_index is not None and x_col == "repeat_index_in_phrase":
        ax.set_xlim(0.5, args.max_repeat_index + 0.5)

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
    return summary


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


def add_suptitle_with_counts(fig, segments: pd.DataFrame, title: str):
    n_animals = segments["animal_id"].nunique()
    n_labels = segments[["animal_id", "label"]].drop_duplicates().shape[0]
    n_segments = segments.shape[0]
    fig.suptitle(f"{title}\n{n_animals} birds, {n_labels} animal×label pairs, {n_segments:,} segmented syllables", fontsize=14)


def make_combined_figures(segments: pd.DataFrame, out_dir: Path, args):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    # Duration + ISI side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    summary_duration = plot_feature_vs_x(
        axes[0, 0], segments,
        x_col="repeat_index_in_phrase",
        y_col="segment_duration_ms",
        x_label="Syllable repeat index within phrase",
        y_label="Syllable duration (ms)",
        title="Syllable duration vs repeat index",
        args=args,
        rng=rng,
    )
    summary_isi = plot_feature_vs_x(
        axes[0, 1], segments,
        x_col="repeat_index_in_phrase",
        y_col="preceding_isi_ms",
        x_label="Syllable repeat index within phrase",
        y_label="Preceding ISI (ms)",
        title="ISI before current syllable vs repeat index",
        args=args,
        rng=rng,
    )
    add_suptitle_with_counts(fig, segments, "Combined top-variance syllable timing by phrase position")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = fig_dir / "combined_duration_and_isi_vs_repeat_index.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    # Time spent repeating side-by-side.
    phrase = build_phrase_table(segments)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    summary_cum = plot_feature_vs_x(
        axes[0, 0], segments,
        x_col="repeat_index_in_phrase",
        y_col="cumulative_phrase_time_s",
        x_label="Syllable repeat index within phrase",
        y_label="Cumulative phrase time through repeat (s)",
        title="Time from phrase start to each repeat ending",
        args=args,
        rng=rng,
    )
    summary_phrase = plot_feature_vs_x(
        axes[0, 1], phrase.rename(columns={"n_segments_in_phrase": "repeat_index_in_phrase"}),
        x_col="repeat_index_in_phrase",
        y_col="phrase_duration_s",
        x_label="Total segmented repeats in phrase",
        y_label="Total repeated-label phrase duration (s)",
        title="Total time spent repeating syllable in each phrase",
        args=args,
        rng=rng,
    )
    add_suptitle_with_counts(fig, segments, "Combined time spent repeating high-variance syllables")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = fig_dir / "combined_time_spent_repeating_vs_repeat_index.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    # Four-panel version for easier manuscript/slide use.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.2), squeeze=False)
    plot_feature_vs_x(
        axes[0, 0], segments, "repeat_index_in_phrase", "segment_duration_ms",
        "Syllable repeat index within phrase", "Syllable duration (ms)",
        "A. Syllable duration", args, rng,
    )
    plot_feature_vs_x(
        axes[0, 1], segments, "repeat_index_in_phrase", "preceding_isi_ms",
        "Syllable repeat index within phrase", "Preceding ISI (ms)",
        "B. Intersyllabic interval", args, rng,
    )
    plot_feature_vs_x(
        axes[1, 0], segments, "repeat_index_in_phrase", "cumulative_phrase_time_s",
        "Syllable repeat index within phrase", "Cumulative phrase time (s)",
        "C. Cumulative phrase time", args, rng,
    )
    plot_feature_vs_x(
        axes[1, 1], phrase.rename(columns={"n_segments_in_phrase": "repeat_index_in_phrase"}),
        "repeat_index_in_phrase", "phrase_duration_s",
        "Total segmented repeats in phrase", "Total repeated-label phrase duration (s)",
        "D. Total time spent repeating", args, rng,
    )
    add_suptitle_with_counts(fig, segments, "Combined top-30% phrase-duration-variance syllable timing")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_png = fig_dir / "combined_duration_isi_time_four_panel.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    # Save plotting summaries used for the mean lines.
    summary_duration.to_csv(out_dir / "combined_repeat_summary_segment_duration_ms.csv", index=False)
    summary_isi.to_csv(out_dir / "combined_repeat_summary_preceding_isi_ms.csv", index=False)
    summary_cum.to_csv(out_dir / "combined_repeat_summary_cumulative_phrase_time_s.csv", index=False)
    summary_phrase.to_csv(out_dir / "combined_repeat_summary_phrase_duration_s_by_total_repeats.csv", index=False)
    phrase.to_csv(out_dir / "combined_phrase_rows_used.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Make combined top-30 duration/ISI phrase-position figures across birds.")
    parser.add_argument("--timing-root", required=True, help="Root containing per-bird timing output folders")
    parser.add_argument("--out-dir", required=True, help="Output directory for combined figures and summary CSVs")
    parser.add_argument("--top30-labels-csv", default=None, help="Optional animal x label CSV to filter to top-30 labels")
    parser.add_argument("--animal-col", default=None, help="Animal ID column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--label-col", default=None, help="Label column in --top30-labels-csv, if auto-detection fails")
    parser.add_argument("--animals", default=None, help="Optional comma-separated animal IDs to include")
    parser.add_argument("--max-repeat-index", type=int, default=None, help="Optional maximum repeat index to plot")

    parser.add_argument("--line-level", choices=["animal_label", "row"], default="animal_label",
                        help="Mean line aggregation. Recommended: animal_label.")
    parser.add_argument("--min-points-per-bin", type=int, default=3,
                        help="Minimum animal-label units, or rows with --line-level row, needed to draw a mean point")
    parser.add_argument("--show-raw-points", action="store_true", default=True)
    parser.add_argument("--hide-raw-points", dest="show_raw_points", action="store_false")
    parser.add_argument("--max-raw-points-per-group", type=int, default=25000,
                        help="Downsample raw transparent points per pre/post group per panel. Use 0 for no limit.")
    parser.add_argument("--jitter", type=float, default=0.06,
                        help="Small x-jitter for raw points so vertical stacks are easier to see")
    parser.add_argument("--show-sem-band", action="store_true",
                        help="Add faint SEM bands around mean lines")

    parser.add_argument("--point-alpha", type=float, default=POINT_ALPHA)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    parser.add_argument("--line-width", type=float, default=LINE_WIDTH)
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()

    if args.max_raw_points_per_group == 0:
        args.max_raw_points_per_group = None

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
    make_combined_figures(segments, out_dir, args)
    print(f"[DONE] Combined figures and summaries saved to {out_dir}")


if __name__ == "__main__":
    main()
