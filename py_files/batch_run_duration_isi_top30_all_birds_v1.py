#!/usr/bin/env python3
"""
Batch-run phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py for the
selected top-variance syllable labels in every bird.

Default behavior assumes --top30-labels-csv is already filtered to the top 30%
phrase-duration-variance syllables. It reads all animal x label pairs in that
CSV, finds each bird's NPZ as <npz-root>/<animal>/<animal>.npz, and runs the
timing script once per bird with --cluster-label label1,label2,...

If you instead provide a full phrase-duration variance table, use
--select-top-fraction-from-table with --metric-col or --rank-col.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal", "animal", "Bird", "bird", "bird_id", "AnimalID"
]
LABEL_COL_CANDIDATES = [
    "label", "Label", "syllable", "Syllable", "syllable_label", "Syllable label",
    "cluster", "Cluster", "cluster_id", "Cluster ID", "hdbscan_label", "HDBSCAN label",
    "mapped_syllable_order", "syllable_cluster", "Syllable cluster", "chosen_label", "selected_label"
]


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


def load_selected_pairs(args) -> pd.DataFrame:
    table = pd.read_csv(Path(args.top30_labels_csv).expanduser())
    if table.empty:
        raise ValueError(f"Top-30 labels CSV is empty: {args.top30_labels_csv}")

    animal_col = args.animal_col or find_first_existing(table.columns, ANIMAL_COL_CANDIDATES)
    label_col = args.label_col or find_first_existing(table.columns, LABEL_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(f"Could not find animal ID column. Columns: {list(table.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column. Columns: {list(table.columns)}")

    work = table.copy()
    work["animal_id"] = work[animal_col].map(normalize_animal_id)
    work["label"] = work[label_col].map(normalize_label_id)
    work = work[(work["animal_id"] != "") & (work["label"] != "")].copy()
    if work.empty:
        raise ValueError("No non-empty animal_id/label pairs after normalization.")

    if args.animals:
        allowed = {x.strip() for x in args.animals.split(",") if x.strip()}
        work = work[work["animal_id"].isin(allowed)].copy()

    if not args.select_top_fraction_from_table:
        selected = work[["animal_id", "label"]].drop_duplicates().copy()
        selected["selection_source"] = "all_rows_in_top30_csv_treated_as_already_filtered"
        return selected.sort_values(["animal_id", "label"])

    if not (0 < args.top_fraction <= 1):
        raise ValueError("--top-fraction must be > 0 and <= 1")

    if args.rank_col:
        if args.rank_col not in work.columns:
            raise ValueError(f"--rank-col {args.rank_col!r} not found. Columns: {list(work.columns)}")
        rank_work = work[["animal_id", "label", args.rank_col]].copy()
        rank_work["_rank"] = pd.to_numeric(rank_work[args.rank_col], errors="coerce")
        rank_work = rank_work[np.isfinite(rank_work["_rank"])].copy()
        if rank_work.empty:
            raise ValueError(f"Rank column {args.rank_col!r} has no finite numeric values.")
        if args.rank_direction == "descending":
            label_scores = rank_work.groupby(["animal_id", "label"], as_index=False)["_rank"].max()
            ascending = False
        else:
            label_scores = rank_work.groupby(["animal_id", "label"], as_index=False)["_rank"].min()
            ascending = True
        score_col = "_rank"
        source = f"top_fraction_from_rank_col:{args.rank_col};direction:{args.rank_direction}"
    elif args.metric_col:
        if args.metric_col not in work.columns:
            raise ValueError(f"--metric-col {args.metric_col!r} not found. Columns: {list(work.columns)}")
        metric_work = work[["animal_id", "label", args.metric_col]].copy()
        metric_work["_metric"] = pd.to_numeric(metric_work[args.metric_col], errors="coerce")
        metric_work = metric_work[np.isfinite(metric_work["_metric"])].copy()
        if metric_work.empty:
            raise ValueError(f"Metric column {args.metric_col!r} has no finite numeric values.")
        label_scores = (
            metric_work.groupby(["animal_id", "label"])["_metric"]
            .apply(lambda x: aggregate_metric(x, args.metric_agg))
            .reset_index(name="_metric")
        )
        ascending = args.metric_direction == "smallest"
        score_col = "_metric"
        source = f"top_fraction_from_metric_col:{args.metric_col};direction:{args.metric_direction};agg:{args.metric_agg}"
    else:
        raise ValueError("Use --metric-col or --rank-col with --select-top-fraction-from-table.")

    parts = []
    for animal, sb in label_scores.groupby("animal_id"):
        sb = sb.sort_values(score_col, ascending=ascending).copy()
        n_keep = max(1, int(math.ceil(sb.shape[0] * args.top_fraction)))
        parts.append(sb.head(n_keep))
    selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["animal_id", "label"])
    selected = selected[["animal_id", "label"]].drop_duplicates().copy()
    selected["selection_source"] = source + f";top_fraction:{args.top_fraction}"
    return selected.sort_values(["animal_id", "label"])


def find_npz_for_animal(npz_root: Path, animal: str, template: Optional[str] = None) -> Optional[Path]:
    if template:
        p = Path(template.format(npz_root=str(npz_root), animal=animal)).expanduser()
        return p if p.exists() else None

    candidates = [
        npz_root / animal / f"{animal}.npz",
        npz_root / animal / f"{animal}_outputs.npz",
        npz_root / f"{animal}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = sorted(npz_root.rglob(f"{animal}.npz"))
    if matches:
        return matches[0]
    matches = sorted(npz_root.rglob(f"*{animal}*.npz"))
    return matches[0] if matches else None


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def main():
    parser = argparse.ArgumentParser(description="Batch-run duration/ISI phrase-position analysis on top-variance labels for all birds.")
    parser.add_argument("--timing-script", required=True, help="Path to phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py")
    parser.add_argument("--npz-root", required=True, help="Root containing <animal>/<animal>.npz files")
    parser.add_argument("--metadata-excel-path", required=True)
    parser.add_argument("--top30-labels-csv", required=True, help="CSV listing selected top-30% animal x label pairs")
    parser.add_argument("--out-root", required=True, help="Output root; one subfolder per bird is created")
    parser.add_argument("--animal-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--animals", default=None, help="Optional comma-separated animal IDs to run")
    parser.add_argument("--npz-path-template", default=None, help="Optional template, e.g. '{npz_root}/{animal}/{animal}.npz'")

    parser.add_argument("--select-top-fraction-from-table", action="store_true", help="Use this only if the CSV is a full table, not already top-30 filtered.")
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--metric-col", default=None, help="Metric column to select largest/smallest top fraction within each bird")
    parser.add_argument("--metric-direction", default="largest", choices=["largest", "smallest"])
    parser.add_argument("--metric-agg", default="max", choices=["max", "median", "mean", "min"])
    parser.add_argument("--rank-col", default=None, help="Rank column to select best top fraction within each bird")
    parser.add_argument("--rank-direction", default="ascending", choices=["ascending", "descending"])

    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    # Common options forwarded to the timing script.
    parser.add_argument("--stitch-across-file-segments", action="store_true")
    parser.add_argument("--majority-vote-window-bins", type=int, default=7)
    parser.add_argument("--majority-vote-min-fraction", type=float, default=0.50)
    parser.add_argument("--max-gap-bins-to-merge", type=int, default=3)
    parser.add_argument("--min-label-run-bins", type=int, default=40)
    parser.add_argument("--bin-ms", type=float, default=2.7)
    parser.add_argument("--keep-edge-segments", action="store_true")
    parser.add_argument("--max-plot-repeat-index", type=int, default=None)
    parser.add_argument("--plot-group-col", default="pre_post", choices=["pre_post", "period"])

    args = parser.parse_args()

    timing_script = Path(args.timing_script).expanduser()
    if not timing_script.exists():
        raise FileNotFoundError(f"Timing script not found: {timing_script}")
    npz_root = Path(args.npz_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    selected = load_selected_pairs(args)
    selected.to_csv(out_root / "top30_label_pairs_used_for_batch.csv", index=False)
    summary = selected.groupby("animal_id").agg(n_labels=("label", "nunique"), labels=("label", lambda x: ",".join(sorted(map(str, set(x)), key=lambda v: (len(v), v))))).reset_index()
    summary.to_csv(out_root / "top30_label_pairs_used_for_batch_summary_by_bird.csv", index=False)
    print(f"[INFO] Birds to run: {summary.shape[0]}")
    print(summary.to_string(index=False))

    failures = []
    for _, row in summary.iterrows():
        animal = str(row["animal_id"])
        labels = str(row["labels"])
        npz_path = find_npz_for_animal(npz_root, animal, args.npz_path_template)
        if npz_path is None:
            msg = f"Could not find NPZ for {animal} under {npz_root}"
            print(f"[ERROR] {msg}")
            failures.append({"animal_id": animal, "error": msg})
            if not args.continue_on_error:
                raise FileNotFoundError(msg)
            continue

        bird_out = out_root / animal
        bird_out.mkdir(parents=True, exist_ok=True)
        combined_csv = bird_out / f"{animal}_all_labels_duration_isi_segment_timing.csv"
        if args.skip_existing and combined_csv.exists():
            print(f"[SKIP] {animal}: {combined_csv} already exists")
            continue

        cmd = [
            args.python_exe,
            str(timing_script),
            "--npz-path", str(npz_path),
            "--animal-id", animal,
            "--metadata-excel-path", str(Path(args.metadata_excel_path).expanduser()),
            "--cluster-label", labels,
            "--out-dir", str(bird_out),
            "--majority-vote-window-bins", str(args.majority_vote_window_bins),
            "--majority-vote-min-fraction", str(args.majority_vote_min_fraction),
            "--max-gap-bins-to-merge", str(args.max_gap_bins_to_merge),
            "--min-label-run-bins", str(args.min_label_run_bins),
            "--bin-ms", str(args.bin_ms),
            "--plot-group-col", args.plot_group_col,
        ]
        if args.stitch_across_file_segments:
            cmd.append("--stitch-across-file-segments")
        if args.keep_edge_segments:
            cmd.append("--keep-edge-segments")
        if args.max_plot_repeat_index is not None:
            cmd.extend(["--max-plot-repeat-index", str(args.max_plot_repeat_index)])

        print("\n[RUN]", shell_join(cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            msg = f"Timing script failed for {animal} with return code {exc.returncode}"
            print(f"[ERROR] {msg}")
            failures.append({"animal_id": animal, "error": msg})
            if not args.continue_on_error:
                raise

    if failures:
        pd.DataFrame(failures).to_csv(out_root / "batch_failures.csv", index=False)
        print(f"[DONE WITH FAILURES] Wrote {out_root / 'batch_failures.csv'}")
    else:
        print("[DONE] All requested birds completed.")


if __name__ == "__main__":
    main()
