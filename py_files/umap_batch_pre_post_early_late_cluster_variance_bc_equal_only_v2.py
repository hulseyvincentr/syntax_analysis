#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_batch_pre_post_early_late_cluster_variance_bc_equal_only.py

Batch runner for:
    umap_pre_post_early_late_cluster_variance_bc_tuned_equal_only.py

This script iterates over one .npz file per bird, calls the single-bird
UMAP/BC pipeline, and aggregates the per-bird summary CSVs into one batch CSV.

Typical expected input layout:
    /Volumes/my_own_SSD/updated_AreaX_outputs/
        USA5288/
            USA5288.npz
        USA5337/
            USA5337.npz
        ...

Default behavior:
    - recursively finds .npz files
    - keeps only files where file stem matches parent folder name
        e.g. USA5288/USA5288.npz
      This avoids accidentally analyzing many small segment .npz files.
    - creates one output folder per bird under --out-dir
    - saves:
        batch_cluster_variance_bc_summary.csv
        batch_run_manifest.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import argparse
import importlib.util
import sys
import traceback

import pandas as pd


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_id_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def _load_single_bird_module(script_path: Path):
    script_path = Path(script_path).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find single-bird script: {script_path}")

    spec = importlib.util.spec_from_file_location("single_bird_umap_bc_module", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {script_path}")

    module = importlib.util.module_from_spec(spec)

    # Important for Python 3.11 dataclasses:
    # the module must be present in sys.modules before exec_module() runs.
    # Otherwise @dataclass can fail with:
    # AttributeError: 'NoneType' object has no attribute '__dict__'
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    missing = [
        name for name in ["UMAPEarlyLateConfig", "run_one_bird"]
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            f"Single-bird script is missing required public item(s): {missing}. "
            "Use the tuned/equal-only script that defines UMAPEarlyLateConfig and run_one_bird(cfg)."
        )

    return module


def _infer_animal_id(npz_path: Path) -> str:
    return Path(npz_path).stem.split("_")[0]


def find_npz_files(
    *,
    root_dir: Path,
    npz_glob: str,
    recursive: bool,
    require_parent_stem_match: bool,
    include_animal_ids: Optional[Sequence[str]],
    exclude_animal_ids: Optional[Sequence[str]],
    out_dir: Path,
) -> List[Path]:
    root_dir = Path(root_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    if recursive:
        candidates = sorted(root_dir.glob(f"**/{npz_glob}"))
    else:
        candidates = sorted(root_dir.glob(npz_glob))

    include_set = {x.strip() for x in include_animal_ids} if include_animal_ids else None
    exclude_set = {x.strip() for x in exclude_animal_ids} if exclude_animal_ids else set()

    out: List[Path] = []
    seen = set()

    for p in candidates:
        if not p.is_file():
            continue

        try:
            resolved = p.resolve()
        except Exception:
            resolved = p

        # Do not recursively re-analyze npz files in the output folder.
        try:
            resolved.relative_to(out_dir)
            continue
        except Exception:
            pass

        animal_id = _infer_animal_id(resolved)

        if require_parent_stem_match and resolved.stem != resolved.parent.name:
            continue

        if include_set is not None and animal_id not in include_set:
            continue

        if animal_id in exclude_set:
            continue

        if str(resolved) not in seen:
            out.append(resolved)
            seen.add(str(resolved))

    return out


def run_batch(
    *,
    single_bird_script: Path,
    root_dir: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    npz_glob: str,
    recursive: bool,
    require_parent_stem_match: bool,
    include_animal_ids: Optional[Sequence[str]],
    exclude_animal_ids: Optional[Sequence[str]],
    max_birds: Optional[int],
    skip_existing: bool,
    dry_run: bool,
    array_key: str,
    cluster_key: str,
    file_key: str,
    vocalization_key: str,
    file_map_key: str,
    include_noise: bool,
    only_cluster_id: Optional[int],
    vocalization_only: bool,
    min_points_per_period: int,
    max_points_per_period: Optional[int],
    plot_max_points_per_period: Optional[int],
    exclude_treatment_day_from_post: bool,
    early_late_split_method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    dpi: int,
    bc_bins: int,
    overlap_density_bins: int,
    overlap_density_gamma: float,
    metadata_sheet: Optional[str],
) -> None:
    root_dir = Path(root_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    metadata_xlsx = Path(metadata_xlsx).expanduser().resolve()
    _safe_mkdir(out_dir)

    module = _load_single_bird_module(single_bird_script)

    npz_files = find_npz_files(
        root_dir=root_dir,
        npz_glob=npz_glob,
        recursive=recursive,
        require_parent_stem_match=require_parent_stem_match,
        include_animal_ids=include_animal_ids,
        exclude_animal_ids=exclude_animal_ids,
        out_dir=out_dir,
    )

    if max_birds is not None and int(max_birds) > 0:
        npz_files = npz_files[: int(max_birds)]

    if not npz_files:
        raise FileNotFoundError(
            "No .npz files were found. "
            "Try checking --root-dir, using --no-require-parent-stem-match, "
            "or changing --npz-glob."
        )

    print("=" * 88)
    print("Batch UMAP/BC run")
    print("=" * 88)
    print(f"Single-bird script: {Path(single_bird_script).expanduser().resolve()}")
    print(f"Root dir:           {root_dir}")
    print(f"Metadata xlsx:      {metadata_xlsx}")
    print(f"Output dir:         {out_dir}")
    print(f"Birds found:        {len(npz_files)}")
    print("=" * 88)

    manifest_rows: List[Dict[str, Any]] = []
    summary_paths: List[Path] = []

    for i, npz_path in enumerate(npz_files, start=1):
        animal_id = _infer_animal_id(npz_path)
        bird_out_dir = out_dir / animal_id
        summary_csv = bird_out_dir / f"{animal_id}_cluster_variance_bc_summary.csv"

        print()
        print("-" * 88)
        print(f"[{i}/{len(npz_files)}] {animal_id}")
        print(f"NPZ: {npz_path}")
        print(f"OUT: {bird_out_dir}")
        print("-" * 88)

        if skip_existing and summary_csv.exists():
            print(f"[SKIP] Summary already exists: {summary_csv}")
            manifest_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "out_dir": str(bird_out_dir),
                "summary_csv": str(summary_csv),
                "status": "skipped_existing",
                "error": "",
            })
            summary_paths.append(summary_csv)
            continue

        if dry_run:
            print("[DRY RUN] Would run this bird.")
            manifest_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "out_dir": str(bird_out_dir),
                "summary_csv": str(summary_csv),
                "status": "dry_run",
                "error": "",
            })
            continue

        try:
            cfg = module.UMAPEarlyLateConfig(
                npz_path=npz_path,
                metadata_xlsx=metadata_xlsx,
                array_key=array_key,
                cluster_key=cluster_key,
                file_key=file_key,
                vocalization_key=vocalization_key,
                file_map_key=file_map_key,
                include_noise=bool(include_noise),
                only_cluster_id=only_cluster_id,
                vocalization_only=bool(vocalization_only),
                min_points_per_period=int(min_points_per_period),
                max_points_per_period=(int(max_points_per_period) if max_points_per_period is not None else None),
                max_points_per_period_for_plot=(
                    int(plot_max_points_per_period) if plot_max_points_per_period is not None else None
                ),
                exclude_treatment_day_from_post=bool(exclude_treatment_day_from_post),
                early_late_split_method=str(early_late_split_method),
                random_seed=int(seed),
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric=str(metric),
                out_dir=bird_out_dir,
                out_prefix=animal_id,
                dpi=int(dpi),
                bc_bins=int(bc_bins),
                overlap_density_bins=int(overlap_density_bins),
                overlap_density_gamma=float(overlap_density_gamma),
                metadata_sheet=metadata_sheet,
            )

            result = module.run_one_bird(cfg)
            result_path = Path(result)
            summary_paths.append(result_path)

            manifest_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "out_dir": str(bird_out_dir),
                "summary_csv": str(result_path),
                "status": "done",
                "error": "",
            })

        except Exception as e:
            print(f"[ERROR] {animal_id}: {e}")
            traceback.print_exc()
            manifest_rows.append({
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "out_dir": str(bird_out_dir),
                "summary_csv": str(summary_csv),
                "status": "error",
                "error": repr(e),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "batch_run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Aggregate successful per-bird summary CSVs.
    dfs: List[pd.DataFrame] = []
    for summary_path in summary_paths:
        try:
            if Path(summary_path).exists():
                df = pd.read_csv(summary_path)
                df["source_summary_csv"] = str(summary_path)
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read summary CSV {summary_path}: {e}")

    if dfs:
        batch_df = pd.concat(dfs, ignore_index=True)
        batch_summary_path = out_dir / "batch_cluster_variance_bc_summary.csv"
        batch_df.to_csv(batch_summary_path, index=False)
        print()
        print("=" * 88)
        print(f"[DONE] Saved batch summary: {batch_summary_path}")
        print(f"[DONE] Saved run manifest:  {manifest_path}")
        print("=" * 88)
    else:
        print()
        print("=" * 88)
        print(f"[DONE] Saved run manifest: {manifest_path}")
        print("[WARN] No per-bird summary CSVs were aggregated.")
        print("=" * 88)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the tuned/equal-only UMAP pre/post early/late BC script over each bird.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root-dir", required=True, help="Root folder containing per-bird folders with .npz files.")
    p.add_argument("--metadata-xlsx", required=True, help="Metadata Excel file with treatment dates.")
    p.add_argument("--out-dir", required=True, help="Batch output folder.")

    p.add_argument(
        "--single-bird-script",
        default="umap_pre_post_early_late_cluster_variance_bc_tuned_equal_only.py",
        help="Path to the single-bird script. If relative, resolved from current working directory.",
    )

    p.add_argument("--npz-glob", default="*.npz", help="Glob for .npz files. With --recursive, this is used as **/<glob>.")
    p.add_argument("--no-recursive", action="store_true", help="Only search directly inside --root-dir.")
    p.add_argument(
        "--no-require-parent-stem-match",
        action="store_true",
        help=(
            "By default, only analyze files where parent folder name matches file stem, "
            "such as USA5288/USA5288.npz. Use this flag to include all matching .npz files."
        ),
    )
    p.add_argument("--include-animal-ids", nargs="*", default=None, help="Optional list of animal IDs to include.")
    p.add_argument("--exclude-animal-ids", nargs="*", default=None, help="Optional list of animal IDs to exclude.")
    p.add_argument("--max-birds", default=None, type=int, help="Optional cap for testing.")
    p.add_argument("--skip-existing", action="store_true", help="Skip birds with an existing per-bird summary CSV.")
    p.add_argument("--dry-run", action="store_true", help="Print what would run without running UMAP.")

    # Single-bird forwarded parameters
    p.add_argument("--array-key", default="predictions")
    p.add_argument("--cluster-key", default="hdbscan_labels")
    p.add_argument("--file-key", default="file_indices")
    p.add_argument("--vocalization-key", default="vocalization")
    p.add_argument("--file-map-key", default="file_map")

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--only-cluster-id", default=None, type=int)
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=6000, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)
    p.add_argument("--plot-max-points-per-period", default=None, type=int)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"])

    p.add_argument("--seed", default=0, type=int, help="Use -1 for no fixed seed / faster UMAP.")
    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean")

    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=20, type=int)
    p.add_argument("--overlap-density-bins", default=20, type=int)
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)
    p.add_argument("--metadata-sheet", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    include_ids = _parse_id_list(args.include_animal_ids)
    exclude_ids = _parse_id_list(args.exclude_animal_ids)

    run_batch(
        single_bird_script=Path(args.single_bird_script).expanduser(),
        root_dir=Path(args.root_dir).expanduser(),
        metadata_xlsx=Path(args.metadata_xlsx).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        npz_glob=str(args.npz_glob),
        recursive=not bool(args.no_recursive),
        require_parent_stem_match=not bool(args.no_require_parent_stem_match),
        include_animal_ids=include_ids,
        exclude_animal_ids=exclude_ids,
        max_birds=args.max_birds,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
        array_key=str(args.array_key),
        cluster_key=str(args.cluster_key),
        file_key=str(args.file_key),
        vocalization_key=str(args.vocalization_key),
        file_map_key=str(args.file_map_key),
        include_noise=bool(args.include_noise),
        only_cluster_id=args.only_cluster_id,
        vocalization_only=not bool(args.no_vocalization_only),
        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=args.max_points_per_period,
        plot_max_points_per_period=args.plot_max_points_per_period,
        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        early_late_split_method=str(args.early_late_split_method),
        seed=int(args.seed),
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        dpi=int(args.dpi),
        bc_bins=int(args.bc_bins),
        overlap_density_bins=int(args.overlap_density_bins),
        overlap_density_gamma=float(args.overlap_density_gamma),
        metadata_sheet=args.metadata_sheet,
    )


if __name__ == "__main__":
    main()
