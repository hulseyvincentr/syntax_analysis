#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_batch_pre_post_variance.py

Batch runner for:
    umap_pre_post_early_late_cluster_variance_bc.py

What it does
------------
- Finds NPZ files in a root directory
- Dynamically imports + reloads the single-bird UMAP/variance/BC pipeline
  so edits are picked up in Spyder without stale imports
- Runs the single-bird pipeline on each NPZ
- Saves per-bird outputs in subfolders under the batch output directory
- Concatenates all per-bird summary CSVs into one batch summary CSV
- Saves a manifest CSV recording success/failure for each bird
- Warns if expected summary columns (including UMAP-space metrics) are missing

Primary batch output
--------------------
out_dir/
  <animal_id>/
    <animal_id>_cluster_variance_bc_summary.csv
    clusters/
      ... figures ...

  batch_cluster_variance_bc_summary.csv
  batch_run_manifest.csv
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import argparse
import traceback
import os
import importlib

import pandas as pd


# ----------------------------
# Expected summary columns
# ----------------------------

EXPECTED_SUMMARY_COLUMNS = [
    "animal_id",
    "cluster",
    "cluster_change",
    "pre_present_any",
    "post_present_any",

    "n_pre_total_raw",
    "n_post_total_raw",
    "n_pre_early_raw",
    "n_pre_late_raw",
    "n_post_early_raw",
    "n_post_late_raw",

    "bc_pre_early_vs_late",
    "bc_post_early_vs_late",
    "bc_pre_vs_post",

    "centroid_shift_raw",
    "pre_rms_radius_raw",
    "post_rms_radius_raw",
    "post_over_pre_rms_radius",
    "pre_trace_cov_raw",
    "post_trace_cov_raw",
    "post_over_pre_trace_cov",

    "centroid_shift_umap",
    "pre_rms_radius_umap",
    "post_rms_radius_umap",
    "post_over_pre_rms_radius_umap",

    "bc_pre_early_vs_late_equal_groups",
    "bc_post_early_vs_late_equal_groups",

    "centroid_shift_raw_equal_groups",
    "pre_rms_radius_raw_equal_groups",
    "post_rms_radius_raw_equal_groups",
    "post_over_pre_rms_radius_equal_groups",
    "pre_trace_cov_raw_equal_groups",
    "post_trace_cov_raw_equal_groups",
    "post_over_pre_trace_cov_equal_groups",

    "centroid_shift_umap_equal_groups",
    "pre_rms_radius_umap_equal_groups",
    "post_rms_radius_umap_equal_groups",
    "post_over_pre_rms_radius_umap_equal_groups",
]


# ----------------------------
# Helpers
# ----------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _find_npz_files(root_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted([p for p in root_dir.glob(pattern) if p.is_file()])


def _choose_default_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 1))


def _load_single_bird_module():
    """
    Dynamically import + reload the single-bird module so Spyder/IPython
    picks up edits without needing a kernel restart.
    """
    mod = importlib.import_module("umap_pre_post_early_late_cluster_variance_bc")
    mod = importlib.reload(mod)
    return mod


def _validate_summary_columns(df: pd.DataFrame, *, label: str) -> None:
    missing = [c for c in EXPECTED_SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        print(f"[warn] {label}: missing expected columns:")
        for c in missing:
            print(f"       - {c}")
    else:
        print(f"[ok] {label}: all expected columns present.")


def _run_one_npz(npz_path: Path, base_cfg_dict: Dict[str, Any], batch_out_dir: Path) -> Dict[str, Any]:
    animal_id = _infer_animal_id(npz_path)
    bird_out_dir = batch_out_dir / animal_id
    _safe_mkdir(bird_out_dir)

    try:
        mod = _load_single_bird_module()
        UMAPEarlyLateConfig = mod.UMAPEarlyLateConfig
        run_one_bird = mod.run_one_bird

        cfg_dict = dict(base_cfg_dict)
        cfg_dict["npz_path"] = Path(npz_path)
        cfg_dict["out_dir"] = bird_out_dir
        cfg_dict["out_prefix"] = animal_id

        cfg = UMAPEarlyLateConfig(**cfg_dict)
        summary_csv = run_one_bird(cfg)

        summary_path = Path(summary_csv)
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                _validate_summary_columns(df, label=f"{animal_id}")
            except Exception as e:
                print(f"[warn] {animal_id}: could not read summary CSV for validation: {e}")

        return {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "status": "ok",
            "summary_csv": str(summary_csv),
            "error": "",
        }

    except Exception as e:
        return {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "status": "failed",
            "summary_csv": "",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _collect_batch_summary(manifest_rows: List[Dict[str, Any]], out_csv: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for row in manifest_rows:
        if row.get("status") != "ok":
            continue

        summary_csv = row.get("summary_csv", "")
        if not summary_csv:
            continue

        p = Path(summary_csv)
        if not p.exists():
            continue

        try:
            df = pd.read_csv(p)
            if "animal_id" not in df.columns:
                df["animal_id"] = row.get("animal_id", "")
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Failed reading per-bird summary {p}: {e}")

    if len(dfs) == 0:
        batch_df = pd.DataFrame()
    else:
        batch_df = pd.concat(dfs, ignore_index=True)

    batch_df.to_csv(out_csv, index=False)

    if len(batch_df) > 0:
        _validate_summary_columns(batch_df, label="batch summary")

    return batch_df


# ----------------------------
# Main batch routine
# ----------------------------

def run_batch(
    *,
    base_cfg_dict: Dict[str, Any],
    root_dir: Path,
    recursive: bool,
    out_dir: Path,
    workers: int,
) -> Dict[str, Path]:
    _safe_mkdir(out_dir)

    npz_files = _find_npz_files(root_dir, recursive=recursive)
    print(f"[batch] Found {len(npz_files)} NPZ files under: {root_dir}")

    if len(npz_files) == 0:
        manifest_csv = out_dir / "batch_run_manifest.csv"
        batch_csv = out_dir / "batch_cluster_variance_bc_summary.csv"
        pd.DataFrame().to_csv(manifest_csv, index=False)
        pd.DataFrame().to_csv(batch_csv, index=False)
        print("[batch] No NPZ files found.")
        return {
            "manifest_csv": manifest_csv,
            "batch_summary_csv": batch_csv,
            "out_dir": out_dir,
        }

    manifest_rows: List[Dict[str, Any]] = []

    if workers <= 1:
        for npz_path in npz_files:
            print(f"[batch] Running: {npz_path.name}")
            res = _run_one_npz(npz_path, base_cfg_dict, out_dir)
            manifest_rows.append(res)
            print(f"[batch] {res['animal_id']}: {res['status']}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"[batch] Using workers={workers}")

        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for npz_path in npz_files:
                futures[ex.submit(_run_one_npz, npz_path, base_cfg_dict, out_dir)] = npz_path

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    npz_path = futures[fut]
                    res = {
                        "animal_id": _infer_animal_id(npz_path),
                        "npz_path": str(npz_path),
                        "status": "failed",
                        "summary_csv": "",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                manifest_rows.append(res)
                print(f"[batch] {res['animal_id']}: {res['status']}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "batch_run_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    batch_csv = out_dir / "batch_cluster_variance_bc_summary.csv"
    batch_df = _collect_batch_summary(manifest_rows, batch_csv)

    print(f"[batch] Saved manifest: {manifest_csv}")
    print(f"[batch] Saved batch summary: {batch_csv}")
    print(f"[batch] Successful birds: {(manifest_df['status'] == 'ok').sum() if len(manifest_df) else 0}")
    print(f"[batch] Total cluster rows: {len(batch_df)}")

    return {
        "manifest_csv": manifest_csv,
        "batch_summary_csv": batch_csv,
        "out_dir": out_dir,
    }


# ----------------------------
# CLI
# ----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_batch_pre_post_variance.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root-dir", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)
    p.add_argument("--plot-max-points-per-period", default=None, type=int)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"])
    p.add_argument("--seed", default=0, type=int, help="Use -1 for no seed / faster UMAP")

    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=100, type=int)
    p.add_argument("--overlap-density-bins", default=180, type=int)
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)

    p.add_argument("--workers", default=0, type=int, help="0 = auto")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = _choose_default_workers()

    # Store config as plain dict so it can be passed cleanly to worker processes
    base_cfg_dict = {
        "npz_path": Path("DUMMY_WILL_BE_REPLACED"),
        "metadata_xlsx": Path(args.metadata_xlsx),

        "array_key": args.array_key,
        "cluster_key": args.cluster_key,
        "file_key": args.file_key,

        "include_noise": bool(args.include_noise),
        "only_cluster_id": None,
        "vocalization_only": not bool(args.no_vocalization_only),

        "min_points_per_period": int(args.min_points_per_period),
        "max_points_per_period": (int(args.max_points_per_period) if args.max_points_per_period is not None else None),
        "max_points_per_period_for_plot": (
            int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None
        ),

        "exclude_treatment_day_from_post": bool(args.exclude_treatment_day_from_post),
        "early_late_split_method": str(args.early_late_split_method),
        "random_seed": int(args.seed),

        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": str(args.metric),

        "out_dir": None,
        "out_prefix": None,
        "dpi": int(args.dpi),
        "bc_bins": int(args.bc_bins),
        "overlap_density_bins": int(args.overlap_density_bins),
        "overlap_density_gamma": float(args.overlap_density_gamma),
    }

    paths = run_batch(
        base_cfg_dict=base_cfg_dict,
        root_dir=Path(args.root_dir),
        recursive=bool(args.recursive),
        out_dir=Path(args.out_dir),
        workers=workers,
    )

    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()