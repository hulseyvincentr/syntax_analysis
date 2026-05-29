#!/usr/bin/env python3
"""
run_cluster_pitch_entropy_high_phrase_variance_medial_lateral_batch_v29.py

Batch wrapper for running the single-label pitch / entropy / segmentation pipeline
on the highest phrase-duration-variance syllables for Medial+Lateral lesioned birds.

This version is designed for files like:
    usage_balanced_phrase_duration_stats.csv

Expected useful columns include:
    Animal ID
    Syllable
    Group
    Post_vs_Pre_Delta_Variance_ms2
    Post_vs_Pre_Variance_Ratio
    Variance_ms2
    N_phrases
    Pre_N_phrases

Default behavior:
    - Use rows where Group == "Post"
    - Keep only birds whose metadata lesion hit type contains BOTH "Medial" and "Lateral"
      (this includes Partial Medial and Lateral lesion and Complete Medial and Lateral lesion)
    - Rank syllables within each remaining bird by Post_vs_Pre_Delta_Variance_ms2, descending
    - Select the top 30% of syllables per bird
    - Run the single-label driver once per selected bird/syllable label

The wrapper can run one bird or all Medial+Lateral birds in the duration-stat file.
Use --animal-id USA5288 for one bird, or omit --animal-id for all matching birds.

Example:
    python run_cluster_pitch_entropy_high_phrase_variance_batch_v28.py \
      --driver-script cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py \
      --duration-stats-path /Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv \
      --npz-root /Volumes/my_own_SSD/updated_AreaX_outputs \
      --metadata-excel-path /Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx \
      --out-root /Volumes/my_own_SSD/updated_AreaX_outputs/pitch_entropy_high_phrase_variance_v28 \
      -- --spec-key s --label-key hdbscan_labels --feature-period all ...

All unknown arguments are passed directly to the single-label driver script.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd


ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal_ID", "animal", "Animal", "bird", "Bird",
    "bird_id", "Bird ID", "subject", "Subject", "animalID", "AnimalID",
]

LABEL_COL_CANDIDATES = [
    "Syllable", "syllable", "syllable_label", "Syllable label", "syllable_id", "Syllable ID",
    "cluster_label", "Cluster label", "cluster", "Cluster", "label", "Label",
    "hdbscan_label", "hdbscan_labels", "HDBSCAN label", "target_label", "Target label",
    "phrase_label", "Phrase label", "phrase_identity", "Phrase identity",
    "mapped_syllable_order", "mapped_syllable", "syllable_identity", "Syllable identity",
]

GROUP_COL_CANDIDATES = ["Group", "group", "period", "Period", "condition", "Condition", "epoch", "Epoch"]

LESION_HIT_TYPE_COL_CANDIDATES = [
    "Lesion hit type", "lesion_hit_type", "Hit type", "hit_type", "Lesion Type",
    "lesion type", "Treatment group", "treatment_group", "Group", "group",
]

DEFAULT_METRIC_CANDIDATES = [
    "Post_vs_Pre_Delta_Variance_ms2",
    "Post_vs_Pre_Variance_Ratio",
    "Variance_ms2",
]


PASSTHROUGH_MANAGED_ARGS = [
    "--cluster-label", "--animal-id", "--npz-path", "--metadata-excel-path", "--out-dir",
]


def _find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    norm_to_col = {re.sub(r"[\s_\-]+", "", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s_\-]+", "", cand).lower()
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def _label_to_cli_string(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _read_table(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(p, sheet_name=sheet_name or 0)
    return pd.read_csv(p)


def _normalize_text_for_filter(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(x).strip().lower())


def _is_medial_lateral_hit(hit_type) -> bool:
    """Return True for partial/complete Medial+Lateral lesion labels, but not lateral-only."""
    s = _normalize_text_for_filter(hit_type)
    compact = s.replace(" ", "")
    return (
        ("medial" in s and "lateral" in s)
        or ("m+l" in str(hit_type).strip().lower())
        or ("medial+lateral" in str(hit_type).strip().lower().replace(" ", ""))
    )


def load_medial_lateral_animals(
        metadata_excel_path: str,
        sheet_name: str = "animal_hit_type_summary",
        animal_col: Optional[str] = None,
        hit_type_col: Optional[str] = None,
        verbose: bool = True,
) -> tuple[set[str], pd.DataFrame]:
    """Read the metadata workbook and return Animal IDs with Medial+Lateral lesion hit types."""
    meta = pd.read_excel(Path(metadata_excel_path).expanduser(), sheet_name=sheet_name)
    animal_col = animal_col or _find_first_existing(meta.columns, ANIMAL_COL_CANDIDATES)
    hit_type_col = hit_type_col or _find_first_existing(meta.columns, LESION_HIT_TYPE_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(
            f"Could not auto-detect animal ID column in metadata sheet {sheet_name!r}.\n"
            f"Columns found: {list(meta.columns)}\n"
            "Rerun with --lesion-animal-col '<column name>'."
        )
    if hit_type_col is None:
        raise ValueError(
            f"Could not auto-detect lesion hit type column in metadata sheet {sheet_name!r}.\n"
            f"Columns found: {list(meta.columns)}\n"
            "Rerun with --lesion-hit-type-col '<column name>'."
        )

    work = meta.copy()
    work[animal_col] = work[animal_col].astype(str).str.strip()
    work["_is_medial_lateral"] = work[hit_type_col].map(_is_medial_lateral_hit)
    ml_animals = set(work.loc[work["_is_medial_lateral"], animal_col].dropna().astype(str).str.strip())

    if verbose:
        print(f"[INFO] Lesion metadata sheet: {sheet_name!r}")
        print(f"[INFO] Lesion metadata columns: animal_col={animal_col!r}, hit_type_col={hit_type_col!r}")
        print(f"[INFO] Medial+Lateral lesioned animals found ({len(ml_animals)}): {sorted(ml_animals)}")
        counts = work[hit_type_col].astype(str).value_counts(dropna=False)
        print("[INFO] Lesion hit type counts:")
        for hit, count in counts.items():
            print(f"       {hit}: {count}")

    return ml_animals, work


def _choose_metric_col(df: pd.DataFrame, metric_col: Optional[str]) -> str:
    if metric_col:
        if metric_col not in df.columns:
            raise ValueError(
                f"Requested --metric-col {metric_col!r} was not found.\n"
                f"Columns found: {list(df.columns)}"
            )
        return metric_col
    chosen = _find_first_existing(df.columns, DEFAULT_METRIC_CANDIDATES)
    if chosen is None:
        raise ValueError(
            "Could not auto-detect a ranking metric column.\n"
            f"Tried: {DEFAULT_METRIC_CANDIDATES}\n"
            f"Columns found: {list(df.columns)}\n"
            "Please rerun with --metric-col '<column name>'."
        )
    return chosen


def _normalize_group_name(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def remove_passthrough_arg(passthrough: list[str], arg_name: str) -> list[str]:
    """Remove an option and its value from passthrough if user accidentally includes it."""
    cleaned: list[str] = []
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


def select_high_variance_labels(
        df: pd.DataFrame,
        animal_col: str,
        label_col: str,
        metric_col: str,
        group_col: Optional[str] = None,
        selection_group: str = "Post",
        animal_ids: Optional[list[str]] = None,
        top_fraction: float = 0.30,
        top_n: Optional[int] = None,
        min_post_phrases: Optional[int] = None,
        min_pre_phrases: Optional[int] = None,
        require_metric_positive: bool = False,
        sort_descending: bool = True,
) -> pd.DataFrame:
    """Return selected rows with rank columns, one row per animal/syllable."""
    work = df.copy()

    work[animal_col] = work[animal_col].astype(str).str.strip()
    work["_label_cli"] = work[label_col].map(_label_to_cli_string)
    work["_metric_numeric"] = pd.to_numeric(work[metric_col], errors="coerce")

    if animal_ids:
        requested = {str(a).strip() for a in animal_ids}
        work = work[work[animal_col].isin(requested)]

    if group_col is not None and selection_group:
        target_group = _normalize_group_name(selection_group)
        work = work[work[group_col].map(_normalize_group_name) == target_group]

    if min_post_phrases is not None:
        if "N_phrases" not in work.columns:
            raise ValueError("--min-post-phrases was requested, but column 'N_phrases' was not found.")
        work = work[pd.to_numeric(work["N_phrases"], errors="coerce") >= int(min_post_phrases)]

    if min_pre_phrases is not None:
        if "Pre_N_phrases" not in work.columns:
            raise ValueError("--min-pre-phrases was requested, but column 'Pre_N_phrases' was not found.")
        work = work[pd.to_numeric(work["Pre_N_phrases"], errors="coerce") >= int(min_pre_phrases)]

    work = work[work["_label_cli"].notna() & work["_metric_numeric"].notna()].copy()
    if require_metric_positive:
        work = work[work["_metric_numeric"] > 0].copy()

    # If there are duplicated animal/syllable rows after filtering, keep the row
    # with the strongest metric for that label.
    work = work.sort_values([animal_col, "_label_cli", "_metric_numeric"], ascending=[True, True, not sort_descending])
    work = work.drop_duplicates(subset=[animal_col, "_label_cli"], keep="first")

    selected_rows = []
    for animal_id, sub in work.groupby(animal_col, sort=True):
        sub = sub.sort_values("_metric_numeric", ascending=not sort_descending).copy()
        n_available = len(sub)
        if n_available == 0:
            continue
        if top_n is not None:
            n_select = min(int(top_n), n_available)
        else:
            n_select = max(1, int(math.ceil(float(top_fraction) * n_available)))
        chosen = sub.head(n_select).copy()
        chosen["rank_within_animal"] = np.arange(1, len(chosen) + 1)
        chosen["n_available_syllables_for_animal"] = n_available
        chosen["n_selected_for_animal"] = n_select
        chosen["selection_metric_col"] = metric_col
        chosen["selection_metric_value"] = chosen["_metric_numeric"]
        selected_rows.append(chosen)

    if not selected_rows:
        raise ValueError(
            "No labels were selected. Check animal IDs, group filter, metric column, and phrase-count filters."
        )

    selected = pd.concat(selected_rows, ignore_index=True)
    return selected


def resolve_npz_path(animal_id: str, npz_root: Optional[str], npz_pattern: str, npz_path: Optional[str]) -> str:
    if npz_path:
        # Only valid for single-animal runs; caller checks that.
        return str(Path(npz_path).expanduser())
    if not npz_root:
        raise ValueError("For multi-animal runs, provide --npz-root, or provide --npz-path for a single --animal-id run.")
    root = Path(npz_root).expanduser()
    rel = npz_pattern.format(animal_id=animal_id)
    return str(root / rel)


def main():
    parser = argparse.ArgumentParser(
        description="Run pitch/entropy/segmentation analysis on high phrase-duration-variance syllables for Medial+Lateral lesioned birds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--driver-script", default="cluster_pitch_entropy_panels_v26_distribution_metrics_ecdf.py",
                        help="Single-label driver script to call for each selected label. Use the newest working version if desired.")
    parser.add_argument("--duration-stats-path", required=True,
                        help="CSV/XLSX like usage_balanced_phrase_duration_stats.csv containing per-bird phrase duration metrics.")
    parser.add_argument("--sheet-name", default=None,
                        help="Sheet name if --duration-stats-path is an Excel workbook. Ignored for CSV.")
    parser.add_argument("--animal-id", action="append", default=None,
                        help="Animal ID to run. Can be supplied multiple times. Omit to run all animals in the stats file.")
    parser.add_argument("--npz-path", default=None,
                        help="NPZ path for a single-animal run. For all-bird runs, use --npz-root instead.")
    parser.add_argument("--npz-root", default=None,
                        help="Root folder containing per-animal NPZ files.")
    parser.add_argument("--npz-pattern", default="{animal_id}/{animal_id}.npz",
                        help="Pattern under --npz-root for each animal NPZ. {animal_id} is replaced.")
    parser.add_argument("--metadata-excel-path", required=True,
                        help="Metadata Excel path to pass to the single-label driver and use for lesion-group filtering.")
    parser.add_argument("--lesion-filter", choices=["medial_lateral", "all"], default="medial_lateral",
                        help="Which birds to include. Default keeps only birds whose lesion hit type contains both Medial and Lateral.")
    parser.add_argument("--lesion-metadata-sheet", default="animal_hit_type_summary",
                        help="Metadata workbook sheet containing one row per animal and lesion hit type.")
    parser.add_argument("--lesion-animal-col", default=None,
                        help="Animal ID column in the lesion metadata sheet. Auto-detected if omitted.")
    parser.add_argument("--lesion-hit-type-col", default=None,
                        help="Lesion hit type column in the lesion metadata sheet. Auto-detected if omitted.")
    parser.add_argument("--out-root", required=True,
                        help="Root output folder. Subfolders are created per animal and label.")

    parser.add_argument("--animal-col", default=None, help="Animal ID column. Auto-detected if omitted.")
    parser.add_argument("--label-col", default=None, help="Syllable/cluster label column. Auto-detected if omitted.")
    parser.add_argument("--group-col", default=None, help="Group/period column. Auto-detected if omitted.")
    parser.add_argument("--selection-group", default="Post",
                        help="Group/period to use for ranking. For usage_balanced_phrase_duration_stats.csv, use 'Post'. Use empty string to disable group filtering.")
    parser.add_argument("--metric-col", default=None,
                        help="Metric column for ranking. Default auto-detects Post_vs_Pre_Delta_Variance_ms2, then ratio, then Variance_ms2.")
    parser.add_argument("--top-fraction", type=float, default=0.30,
                        help="Fraction of labels to select per bird when --top-n is not provided.")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Optional fixed number of labels to select per bird instead of --top-fraction.")
    parser.add_argument("--min-post-phrases", type=int, default=None,
                        help="Optional minimum Post N_phrases for a syllable to be eligible.")
    parser.add_argument("--min-pre-phrases", type=int, default=None,
                        help="Optional minimum Pre_N_phrases for a syllable to be eligible.")
    parser.add_argument("--require-metric-positive", action="store_true",
                        help="Only keep labels where the ranking metric is positive, useful for selecting increased post-lesion variance.")
    parser.add_argument("--sort-ascending", action="store_true",
                        help="Select the lowest metric values instead of the highest. Usually leave off.")

    parser.add_argument("--max-labels-per-animal", type=int, default=None,
                        help="Debugging limit after top-fraction selection; caps labels per animal.")
    parser.add_argument("--max-total-runs", type=int, default=None,
                        help="Debugging limit on total animal/label driver runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write selected-label files without running driver.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a label if its output folder already has a DONE marker.")
    parser.add_argument("--done-marker", default="_RUN_COMPLETE.txt", help="Marker file written to each label output folder on successful completion.")

    args, passthrough = parser.parse_known_args()
    passthrough = [x for x in passthrough if x != "--"]
    for managed in PASSTHROUGH_MANAGED_ARGS:
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

    if args.npz_path and (not args.animal_id or len(args.animal_id) != 1):
        raise ValueError("--npz-path can only be used with exactly one --animal-id. For all birds, use --npz-root.")

    df = _read_table(args.duration_stats_path, sheet_name=args.sheet_name)
    animal_col = args.animal_col or _find_first_existing(df.columns, ANIMAL_COL_CANDIDATES)
    label_col = args.label_col or _find_first_existing(df.columns, LABEL_COL_CANDIDATES)
    group_col = args.group_col or _find_first_existing(df.columns, GROUP_COL_CANDIDATES)
    metric_col = _choose_metric_col(df, args.metric_col)

    if animal_col is None:
        raise ValueError(f"Could not auto-detect animal column. Columns found: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not auto-detect label column. Columns found: {list(df.columns)}")
    if args.selection_group and group_col is None:
        raise ValueError(
            f"--selection-group {args.selection_group!r} was requested, but no group column was found.\n"
            f"Columns found: {list(df.columns)}\n"
            "Rerun with --group-col '<column name>' or --selection-group ''."
        )

    selected = select_high_variance_labels(
        df,
        animal_col=animal_col,
        label_col=label_col,
        metric_col=metric_col,
        group_col=group_col,
        selection_group=args.selection_group,
        animal_ids=animal_ids_for_selection,
        top_fraction=args.top_fraction,
        top_n=args.top_n,
        min_post_phrases=args.min_post_phrases,
        min_pre_phrases=args.min_pre_phrases,
        require_metric_positive=args.require_metric_positive,
        sort_descending=(not args.sort_ascending),
    )

    if args.max_labels_per_animal is not None:
        selected = selected[selected["rank_within_animal"] <= int(args.max_labels_per_animal)].copy()
    if args.max_total_runs is not None:
        selected = selected.head(int(args.max_total_runs)).copy()

    selected_path = out_root / "selected_high_phrase_duration_variance_labels.csv"
    selected.to_csv(selected_path, index=False)

    if lesion_meta_used is not None:
        lesion_meta_path = out_root / "lesion_metadata_with_medial_lateral_filter.csv"
        lesion_meta_used.to_csv(lesion_meta_path, index=False)
        print(f"[SAVED] {lesion_meta_path}")

    manifest_path = out_root / "high_phrase_duration_variance_batch_manifest.csv"
    print("[INFO] Duration stats path:", args.duration_stats_path)
    print("[INFO] Detected columns:")
    print(f"       animal_col={animal_col!r}")
    print(f"       label_col={label_col!r}")
    print(f"       group_col={group_col!r}")
    print(f"       metric_col={metric_col!r}")
    print(f"[INFO] Selection group: {args.selection_group!r}")
    print(f"[INFO] Lesion filter: {args.lesion_filter!r}")
    print(f"[INFO] Selected {len(selected)} total animal/label runs across {selected[animal_col].nunique()} animal(s).")
    print(f"[SAVED] {selected_path}")

    print("[INFO] Selected labels by animal:")
    for animal_id, sub in selected.groupby(animal_col, sort=True):
        labels = sub["_label_cli"].astype(str).tolist()
        print(f"  {animal_id}: {labels}")

    manifest_rows = []
    start_all = time.time()

    for run_idx, (_row_index, row) in enumerate(selected.iterrows(), start=1):
        animal_id = str(row[animal_col]).strip()
        label = str(row["_label_cli"]).strip()
        rank = int(row.get("rank_within_animal", run_idx))
        metric_value = row.get("selection_metric_value", np.nan)

        animal_safe = _safe_name(animal_id)
        label_safe = _safe_name(label)
        label_out_dir = out_root / animal_safe / f"{animal_safe}_label{label_safe}"
        label_out_dir.mkdir(parents=True, exist_ok=True)
        done_path = label_out_dir / args.done_marker

        if args.skip_existing and done_path.exists():
            print(f"[SKIP] {run_idx}/{len(selected)} {animal_id} label {label}: done marker exists at {done_path}")
            manifest_rows.append({
                "animal_id": animal_id,
                "label": label,
                "rank_within_animal": rank,
                "selection_metric_col": metric_col,
                "selection_metric_value": metric_value,
                "npz_path": "",
                "out_dir": str(label_out_dir),
                "return_code": 0,
                "status": "skipped_existing",
                "elapsed_s": 0.0,
                "command": "",
            })
            continue

        npz_path = resolve_npz_path(
            animal_id=animal_id,
            npz_root=args.npz_root,
            npz_pattern=args.npz_pattern,
            npz_path=args.npz_path,
        )
        if not Path(npz_path).exists():
            msg = f"NPZ file not found for {animal_id}: {npz_path}"
            print(f"[ERROR] {msg}")
            manifest_rows.append({
                "animal_id": animal_id,
                "label": label,
                "rank_within_animal": rank,
                "selection_metric_col": metric_col,
                "selection_metric_value": metric_value,
                "npz_path": npz_path,
                "out_dir": str(label_out_dir),
                "return_code": 2,
                "status": "missing_npz",
                "elapsed_s": 0.0,
                "command": "",
            })
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            if not args.dry_run:
                sys.exit(2)
            continue

        cmd = [
            sys.executable,
            str(driver_path),
            "--npz-path", npz_path,
            "--cluster-label", str(label),
            "--animal-id", animal_id,
            "--metadata-excel-path", args.metadata_excel_path,
            "--out-dir", str(label_out_dir),
        ] + passthrough

        print("\n" + "=" * 100)
        print(f"[RUN] {run_idx}/{len(selected)}: {animal_id} label {label} rank={rank} {metric_col}={metric_value}")
        print("[CMD]", " ".join([repr(c) if " " in c else c for c in cmd]))

        if args.dry_run:
            manifest_rows.append({
                "animal_id": animal_id,
                "label": label,
                "rank_within_animal": rank,
                "selection_metric_col": metric_col,
                "selection_metric_value": metric_value,
                "npz_path": npz_path,
                "out_dir": str(label_out_dir),
                "return_code": np.nan,
                "status": "dry_run",
                "elapsed_s": 0.0,
                "command": " ".join(cmd),
            })
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            continue

        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "success" if proc.returncode == 0 else "failed"
        print(f"[DONE] {animal_id} label {label} return_code={proc.returncode} elapsed={elapsed:.1f} s")

        if proc.returncode == 0:
            done_path.write_text(
                f"Completed {animal_id} label {label}\n"
                f"Rank within animal: {rank}\n"
                f"Selection metric: {metric_col}={metric_value}\n"
                f"Elapsed seconds: {elapsed:.3f}\n"
                f"Command: {' '.join(cmd)}\n"
            )

        manifest_rows.append({
            "animal_id": animal_id,
            "label": label,
            "rank_within_animal": rank,
            "selection_metric_col": metric_col,
            "selection_metric_value": metric_value,
            "npz_path": npz_path,
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
    print("\n" + "=" * 100)
    print(f"[DONE] Batch complete. Processed {len(manifest_rows)} animal/label runs in {total_elapsed / 60.0:.1f} min")
    print(f"[SAVED] {manifest_path}")


if __name__ == "__main__":
    main()