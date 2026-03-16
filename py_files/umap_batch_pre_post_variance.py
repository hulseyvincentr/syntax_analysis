#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_batch_pre_post_variance.py

%run /Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/umap_batch_pre_post_variance.py --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" --recursive --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/batch_umap_pre_post_variance" --array-key "predictions" --cluster-key "hdbscan_labels" --file-key "file_indices" --min-points-per-period 200 --max-points-per-period 20000 --plot-max-points-per-period 3000 --n-neighbors 30 --min-dist 0.1 --metric "euclidean" --bc-bins 100 --overlap-density-bins 120 --overlap-density-gamma 0.55 --seed -1

Batch runner for:
    umap_pre_post_early_late_cluster_variance_bc.py

This script:
  - finds all .npz files in a root directory
  - runs the updated single-bird metric code on each file
  - saves each bird's outputs into its own subfolder
  - combines all per-bird summary CSVs into:
        batch_cluster_variance_bc_summary.csv
  - writes:
        batch_run_status.csv

IMPORTANT
---------
This script does NOT duplicate the metric logic.
Instead, it imports the sibling script:
    umap_pre_post_early_late_cluster_variance_bc.py

So whatever new metrics you add there automatically propagate here.

Expected directory structure
----------------------------
py_files/
  umap_pre_post_early_late_cluster_variance_bc.py
  umap_batch_pre_post_variance.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import importlib.util
import sys

import pandas as pd


def _load_single_bird_module():
    """
    Load the sibling metric-generation script dynamically so the batch runner
    always uses the latest version of the single-bird code.
    """
    this_file = Path(__file__).resolve()
    single_bird_path = this_file.with_name("umap_pre_post_early_late_cluster_variance_bc.py")

    if not single_bird_path.exists():
        raise FileNotFoundError(
            f"Could not find sibling script:\n  {single_bird_path}\n"
            "Make sure umap_pre_post_early_late_cluster_variance_bc.py is in the same folder."
        )

    spec = importlib.util.spec_from_file_location(
        "umap_pre_post_early_late_cluster_variance_bc",
        str(single_bird_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {single_bird_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["umap_pre_post_early_late_cluster_variance_bc"] = module
    spec.loader.exec_module(module)
    return module


_SINGLE_BIRD = _load_single_bird_module()
UMAPEarlyLateConfig = _SINGLE_BIRD.UMAPEarlyLateConfig
run_one_bird = _SINGLE_BIRD.run_one_bird


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _find_npz_files(root_dir: Path, recursive: bool = True) -> List[Path]:
    pat = "**/*.npz" if recursive else "*.npz"
    return sorted([p for p in root_dir.glob(pat) if p.is_file()])


def run_batch_directory(
    *,
    root_dir: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    recursive: bool = True,
    common_cfg_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    root_dir = Path(root_dir)
    metadata_xlsx = Path(metadata_xlsx)
    out_dir = Path(out_dir)
    common_cfg_kwargs = dict(common_cfg_kwargs or {})

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not metadata_xlsx.exists():
        raise FileNotFoundError(f"metadata_xlsx not found: {metadata_xlsx}")

    _safe_mkdir(out_dir)

    npz_files = _find_npz_files(root_dir, recursive=recursive)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {root_dir}")

    print(f"[batch] found {len(npz_files)} npz files in {root_dir}")

    all_rows: List[pd.DataFrame] = []
    batch_status_rows: List[Dict[str, Any]] = []

    for i, npz_path in enumerate(npz_files, start=1):
        animal_id = _infer_animal_id(npz_path)
        bird_out_dir = out_dir / animal_id
        _safe_mkdir(bird_out_dir)

        print(f"\n[batch] ({i}/{len(npz_files)}) running {animal_id}: {npz_path.name}")

        try:
            cfg = UMAPEarlyLateConfig(
                npz_path=npz_path,
                metadata_xlsx=metadata_xlsx,
                out_dir=bird_out_dir,
                **common_cfg_kwargs,
            )

            summary_csv = Path(run_one_bird(cfg))

            if not summary_csv.exists():
                raise FileNotFoundError(
                    f"Expected per-bird summary CSV not created:\n  {summary_csv}"
                )

            df = pd.read_csv(summary_csv)

            # Only add animal_id if it is missing
            if "animal_id" not in df.columns:
                df.insert(0, "animal_id", animal_id)

            if "npz_path" not in df.columns:
                df.insert(1, "npz_path", str(npz_path))

            all_rows.append(df)

            batch_status_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "status": "ok",
                "summary_csv": str(summary_csv),
                "out_dir": str(bird_out_dir),
            })

        except Exception as e:
            print(f"[batch] ERROR for {animal_id}: {repr(e)}")
            batch_status_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "status": f"error: {repr(e)}",
                "summary_csv": "",
                "out_dir": str(bird_out_dir),
            })

    batch_status_csv = out_dir / "batch_run_status.csv"
    pd.DataFrame(batch_status_rows).to_csv(batch_status_csv, index=False)

    batch_summary_csv = out_dir / "batch_cluster_variance_bc_summary.csv"
    if len(all_rows) > 0:
        batch_summary = pd.concat(all_rows, axis=0, ignore_index=True)
        batch_summary.to_csv(batch_summary_csv, index=False)
    else:
        pd.DataFrame().to_csv(batch_summary_csv, index=False)

    print(f"\n[batch] saved status CSV: {batch_status_csv}")
    print(f"[batch] saved aggregate summary CSV: {batch_summary_csv}")
    return batch_summary_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_batch_pre_post_variance.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root-dir", required=True, type=str, help="Root directory containing multiple .npz files")
    p.add_argument("--recursive", action="store_true", help="Recursively search for .npz files")
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--cluster-id", default=None, type=int, help="If set, run only this cluster for every bird")
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

    p.add_argument("--out-prefix", default=None, type=str)
    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=100, type=int)
    p.add_argument("--overlap-density-bins", default=180, type=int)
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    common_cfg_kwargs = dict(
        array_key=args.array_key,
        cluster_key=args.cluster_key,
        file_key=args.file_key,

        include_noise=bool(args.include_noise),
        only_cluster_id=(int(args.cluster_id) if args.cluster_id is not None else None),
        vocalization_only=not bool(args.no_vocalization_only),

        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=(int(args.max_points_per_period) if args.max_points_per_period is not None else None),
        max_points_per_period_for_plot=(
            int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None
        ),

        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        early_late_split_method=str(args.early_late_split_method),
        random_seed=int(args.seed),

        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),

        out_prefix=args.out_prefix,
        dpi=int(args.dpi),
        bc_bins=int(args.bc_bins),
        overlap_density_bins=int(args.overlap_density_bins),
        overlap_density_gamma=float(args.overlap_density_gamma),
    )

    run_batch_directory(
        root_dir=Path(args.root_dir),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=Path(args.out_dir),
        recursive=bool(args.recursive),
        common_cfg_kwargs=common_cfg_kwargs,
    )


if __name__ == "__main__":
    main()