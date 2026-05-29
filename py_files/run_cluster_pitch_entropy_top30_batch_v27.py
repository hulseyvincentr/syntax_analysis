#!/usr/bin/env python3
"""
run_cluster_pitch_entropy_top30_batch_v27.py

Batch wrapper for running the single-label pitch/entropy/segmentation pipeline
on the top-30% phrase-duration-variance syllables for one animal.

This script reads a CSV containing top-variance syllable/cluster labels,
filters it to a requested animal ID, extracts unique labels, and then calls
an existing single-label driver script (default:
cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py) once per label.

Example use:
    python run_cluster_pitch_entropy_top30_batch_v27.py \
      --driver-script cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py \
      --top30-csv /path/to/merged_top30_bc_rows.csv \
      --animal-id USA5288 \
      --npz-path /Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz \
      --metadata-excel-path /Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx \
      --out-root /Volumes/my_own_SSD/updated_AreaX_outputs/pitch_entropy_top30_USA5288_v27 \
      -- --spec-key s --label-key hdbscan_labels --feature-period all ...

All arguments after '--' are passed directly to the single-label driver script.
You can also omit the '--'; unknown arguments are passed through automatically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


ANIMAL_COL_CANDIDATES = [
    "animal_id", "Animal ID", "Animal_ID", "animal", "Animal", "bird", "Bird",
    "bird_id", "Bird ID", "subject", "Subject", "animalID", "AnimalID",
]

LABEL_COL_CANDIDATES = [
    "cluster_label", "Cluster label", "cluster", "Cluster",
    "syllable_label", "Syllable label", "syllable", "Syllable",
    "label", "Label", "hdbscan_label", "hdbscan_labels", "HDBSCAN label",
    "target_label", "Target label", "phrase_label", "Phrase label",
    "phrase_identity", "Phrase identity", "syllable_identity", "Syllable identity",
    "mapped_syllable_order", "mapped_syllable", "syllable_id", "Syllable ID",
]

TOP30_FLAG_CANDIDATES = [
    "top30", "top_30", "is_top30", "in_top30", "top30_flag", "top_30_flag",
    "top_30_percent", "top30_percent", "is_top_30_percent",
]


def _find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    # Exact match first.
    for cand in candidates:
        if cand in cols:
            return cand
    # Case/spacing-insensitive fallback.
    norm_to_col = {re.sub(r"[\s_\-]+", "", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s_\-]+", "", cand).lower()
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def _label_to_cli_string(value) -> Optional[str]:
    """Convert CSV label values into strings that match NPZ labels robustly."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Common cases: "17.0" should become "17".
        try:
            f = float(s)
            if np.isfinite(f) and f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return s
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
        if np.isfinite(f):
            return str(f)
    except Exception:
        return str(value)
    return None


def _truthy_series(series: pd.Series) -> pd.Series:
    """Interpret boolean-ish top30 columns robustly."""
    if series.dtype == bool:
        return series.fillna(False)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "t", "yes", "y", "1", "top", "top30", "top_30"])


def load_top30_labels(
        top30_csv: str,
        animal_id: str,
        label_col: Optional[str] = None,
        animal_col: Optional[str] = None,
        top30_flag_col: Optional[str] = None,
        max_labels: Optional[int] = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    df = pd.read_csv(top30_csv)
    original_n = len(df)

    chosen_animal_col = animal_col or _find_first_existing(df.columns, ANIMAL_COL_CANDIDATES)
    chosen_label_col = label_col or _find_first_existing(df.columns, LABEL_COL_CANDIDATES)
    chosen_top30_col = top30_flag_col or _find_first_existing(df.columns, TOP30_FLAG_CANDIDATES)

    metadata = {
        "top30_csv": top30_csv,
        "original_rows": original_n,
        "animal_col": chosen_animal_col,
        "label_col": chosen_label_col,
        "top30_flag_col": chosen_top30_col,
    }

    if chosen_label_col is None:
        raise ValueError(
            "Could not identify the syllable/cluster label column in the top30 CSV.\n"
            f"Columns found: {list(df.columns)}\n"
            "Please rerun with --label-col '<column name>'."
        )

    filtered = df.copy()
    if chosen_animal_col is not None:
        filtered = filtered[filtered[chosen_animal_col].astype(str).str.strip() == str(animal_id)]
        metadata["rows_after_animal_filter"] = len(filtered)
    else:
        print("[WARN] No animal ID column found in top30 CSV; using all rows.")
        metadata["rows_after_animal_filter"] = len(filtered)

    # If the CSV has a top30 boolean flag, apply it. If it is already a top30-only file,
    # there may be no such column, which is fine.
    if chosen_top30_col is not None:
        filtered = filtered[_truthy_series(filtered[chosen_top30_col])]
        metadata["rows_after_top30_filter"] = len(filtered)
    else:
        metadata["rows_after_top30_filter"] = len(filtered)

    labels = []
    seen = set()
    for v in filtered[chosen_label_col].tolist():
        s = _label_to_cli_string(v)
        if s is None or s in seen:
            continue
        labels.append(s)
        seen.add(s)

    if max_labels is not None:
        labels = labels[:int(max_labels)]

    if not labels:
        raise ValueError(
            f"No labels found for animal_id={animal_id!r} in {top30_csv}.\n"
            f"Detected animal column={chosen_animal_col!r}; label column={chosen_label_col!r}."
        )

    return labels, filtered, metadata


def remove_passthrough_arg(passthrough: list[str], arg_name: str) -> list[str]:
    """Remove an option and its value from passthrough if user accidentally includes it."""
    cleaned = []
    i = 0
    while i < len(passthrough):
        token = passthrough[i]
        if token == arg_name:
            i += 2
            continue
        if token.startswith(arg_name + "="):
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Run the pitch/entropy/segmentation pipeline for every top-30% phrase-duration-variance label for one animal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--driver-script", default="cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py",
                        help="Single-label driver script to call for each label. Use the newest working version if desired.")
    parser.add_argument("--top30-csv", required=True,
                        help="CSV containing top-30% phrase-duration-variance syllable/cluster labels.")
    parser.add_argument("--animal-id", required=True, help="Animal ID to filter top30 CSV and pass to driver, e.g. USA5288.")
    parser.add_argument("--npz-path", required=True, help="Path to this animal's NPZ file.")
    parser.add_argument("--metadata-excel-path", required=True, help="Metadata Excel path to pass to the single-label driver.")
    parser.add_argument("--out-root", required=True, help="Root output folder. A subfolder is created for each label.")
    parser.add_argument("--label-col", default=None, help="Column in top30 CSV containing cluster/syllable labels. Auto-detected if omitted.")
    parser.add_argument("--animal-col", default=None, help="Column in top30 CSV containing animal IDs. Auto-detected if omitted.")
    parser.add_argument("--top30-flag-col", default=None, help="Optional boolean-ish column marking top30 rows. Auto-detected if present.")
    parser.add_argument("--max-labels", type=int, default=None, help="Optional debugging limit on number of labels to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a label if its output folder already has a DONE marker.")
    parser.add_argument("--done-marker", default="_RUN_COMPLETE.txt", help="Marker file written to each label output folder on successful completion.")

    args, passthrough = parser.parse_known_args()

    # Remove separator if user used: wrapper args -- child args.
    passthrough = [x for x in passthrough if x != "--"]
    # Avoid duplicates/conflicts with values managed by this wrapper.
    for managed in ["--cluster-label", "--animal-id", "--npz-path", "--metadata-excel-path", "--out-dir"]:
        passthrough = remove_passthrough_arg(passthrough, managed)

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    driver_path = Path(args.driver_script).expanduser()
    if not driver_path.is_absolute():
        driver_path = Path.cwd() / driver_path
    if not driver_path.exists():
        raise FileNotFoundError(
            f"Could not find driver script: {driver_path}\n"
            "Put this batch wrapper in the same folder as the single-label script, or pass --driver-script with the full path."
        )

    labels, filtered_df, meta = load_top30_labels(
        top30_csv=args.top30_csv,
        animal_id=args.animal_id,
        label_col=args.label_col,
        animal_col=args.animal_col,
        top30_flag_col=args.top30_flag_col,
        max_labels=args.max_labels,
    )

    manifest_path = out_root / f"{args.animal_id}_top30_batch_manifest.csv"
    filtered_path = out_root / f"{args.animal_id}_top30_rows_used.csv"
    filtered_df.to_csv(filtered_path, index=False)

    print("[INFO] Top30 CSV:", args.top30_csv)
    print("[INFO] Detected columns:", meta)
    print(f"[INFO] Labels to run for {args.animal_id}: {labels}")
    print(f"[INFO] Saving filtered top30 rows to: {filtered_path}")
    print(f"[INFO] Saving manifest to: {manifest_path}")

    manifest_rows = []
    start_all = time.time()

    for run_idx, label in enumerate(labels, start=1):
        label_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))
        label_out_dir = out_root / f"{args.animal_id}_label{label_safe}"
        label_out_dir.mkdir(parents=True, exist_ok=True)
        done_path = label_out_dir / args.done_marker

        if args.skip_existing and done_path.exists():
            print(f"[SKIP] {run_idx}/{len(labels)} label {label}: done marker exists at {done_path}")
            manifest_rows.append({
                "animal_id": args.animal_id,
                "label": label,
                "out_dir": str(label_out_dir),
                "return_code": 0,
                "status": "skipped_existing",
                "elapsed_s": 0.0,
                "command": "",
            })
            continue

        cmd = [
            sys.executable,
            str(driver_path),
            "--npz-path", args.npz_path,
            "--cluster-label", str(label),
            "--animal-id", args.animal_id,
            "--metadata-excel-path", args.metadata_excel_path,
            "--out-dir", str(label_out_dir),
        ] + passthrough

        print("\n" + "=" * 90)
        print(f"[RUN] {run_idx}/{len(labels)}: {args.animal_id} label {label}")
        print("[CMD]", " ".join([repr(c) if " " in c else c for c in cmd]))

        if args.dry_run:
            manifest_rows.append({
                "animal_id": args.animal_id,
                "label": label,
                "out_dir": str(label_out_dir),
                "return_code": np.nan,
                "status": "dry_run",
                "elapsed_s": 0.0,
                "command": " ".join(cmd),
            })
            continue

        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "success" if proc.returncode == 0 else "failed"
        print(f"[DONE] label {label} return_code={proc.returncode} elapsed={elapsed:.1f} s")

        if proc.returncode == 0:
            done_path.write_text(
                f"Completed {args.animal_id} label {label}\n"
                f"Elapsed seconds: {elapsed:.3f}\n"
                f"Command: {' '.join(cmd)}\n"
            )

        manifest_rows.append({
            "animal_id": args.animal_id,
            "label": label,
            "out_dir": str(label_out_dir),
            "return_code": proc.returncode,
            "status": status,
            "elapsed_s": elapsed,
            "command": " ".join(cmd),
        })
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        if proc.returncode != 0:
            print("[ERROR] A label run failed. Stopping batch so the issue can be fixed before continuing.")
            print(f"[INFO] Partial manifest saved to: {manifest_path}")
            sys.exit(proc.returncode)

    total_elapsed = time.time() - start_all
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print("\n" + "=" * 90)
    print(f"[DONE] Batch complete. Ran {len(manifest_rows)} labels in {total_elapsed / 60.0:.1f} min")
    print(f"[SAVED] {manifest_path}")


if __name__ == "__main__":
    main()
