#!/usr/bin/env python3
"""Run the Figure 4 Bhattacharyya-coefficient pipeline from per-bird NPZ files.

This driver makes the analysis settings explicit and invokes the frozen analysis
scripts stored beside it. It runs the 20 analyzable birds used in the final
bird-level analysis by default. USA5505 is intentionally omitted because it has
no usable post-treatment recordings for the four-period comparison.

The raw-data rerun should be validated against the frozen accepted summaries
using validate_figure4_results.py. During manuscript preparation, a rerun of the
legacy NPZ inputs produced different per-cluster values for several previously
analyzed birds, so matching file names and command-line settings alone are not a
substitute for frozen inputs and output validation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_BIRDS = [
    "R08", "R09", "R10",
    "USA5271", "USA5283", "USA5288",
    "USA5325", "USA5326", "USA5336", "USA5337",
    "USA5347", "USA5371", "USA5443", "USA5468",
    "USA5483", "USA5494", "USA5499",
    "USA5506", "USA5509", "USA5510",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Run per-bird Figure 4 BC analyses and final batch plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root searched recursively for per-bird NPZ files.")
    p.add_argument("--metadata-excel", type=Path, required=True)
    p.add_argument("--phrase-csv", type=Path, required=True,
                   help="Phrase-duration statistics used to define the top 30% clusters.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Fresh output root for per-bird summaries and batch results.")
    p.add_argument("--npz-manifest", type=Path, default=None,
                   help="Optional CSV with columns animal_id,npz_path. Recommended when NPZ discovery is ambiguous.")
    p.add_argument("--birds", nargs="+", default=DEFAULT_BIRDS)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--wrapper-script", type=Path,
                   default=here / "bc_cluster_qc_and_summaries_v19_full_contiguous_majority_vote_smoothing.py")
    p.add_argument("--export-script", type=Path,
                   default=here / "export_equal_umap_cluster_spectrograms_v23_full_contiguous_majority_vote_smoothing_umap_title_spacing.py")
    p.add_argument("--spectrogram-script", type=Path,
                   default=here / "pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v7.py")
    p.add_argument("--batch-script", type=Path,
                   default=here / "bc_batch_lesion_group_summary_with_remaining_v8_dynamic_ylims_whisker_brackets_ML_combined_final.py")
    p.add_argument("--allow-existing-output", action="store_true")
    p.add_argument("--skip-batch", action="store_true")
    return p.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_manifest(path: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"animal_id", "npz_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"NPZ manifest must contain {sorted(required)}; found {reader.fieldnames}")
        for row in reader:
            animal = str(row["animal_id"]).strip()
            raw = str(row["npz_path"]).strip()
            if animal and raw:
                result[animal] = Path(raw).expanduser().resolve()
    return result


def discover_npz(data_root: Path, animal_id: str) -> Path:
    all_npz = list(data_root.rglob("*.npz"))
    exact = [p for p in all_npz if p.stem.lower() == animal_id.lower()]
    if len(exact) == 1:
        return exact[0].resolve()
    starts = [p for p in all_npz if p.stem.lower().startswith(animal_id.lower())]
    parent_exact = [p for p in starts if p.parent.name.lower() == animal_id.lower()]
    candidates = parent_exact or starts
    if len(candidates) != 1:
        preview = "\n".join(str(p) for p in candidates[:20]) or "  none"
        raise RuntimeError(
            f"Could not identify exactly one NPZ for {animal_id}. "
            f"Use --npz-manifest. Candidates:\n{preview}"
        )
    return candidates[0].resolve()


def build_command(args: argparse.Namespace, animal: str, npz_path: Path, out_root: Path) -> List[str]:
    return [
        args.python,
        str(args.wrapper_script),
        "--npz-path", str(npz_path),
        "--metadata-excel-path", str(args.metadata_excel),
        "--spectrogram-script", str(args.spectrogram_script),
        "--v8-script", str(args.export_script),
        "--phrase-csv", str(args.phrase_csv),
        "--out-dir", str(out_root),
        "--animal-id", animal,
        "--top-fraction", "0.30",
        "--post-group-name", "Post",
        "--top-min-n-phrases", "100",
        "--period-mode", "early_late_pre_post",
        "--treatment-day-assignment", "exclude",
        "--early-late-split-method", "file_median",
        "--bc-analysis-mode", "run_balanced_full_contiguous",
        "--min-runs-per-group", "20",
        "--max-runs-per-group", "200",
        "--min-full-run-duration-ms", "100",
        "--run-sample-mode", "random",
        "--apply-majority-vote-label-smoothing",
        "--majority-vote-window-bins", "200",
        "--spectrogram-source-mode", "expanded_full_runs",
        "--full-run-fixed-duration-s", "5.4",
        "--seconds-per-bin", "0.0027",
        "--umap-density-bins", "20",
        "--bc-grid-point-coverage", "0.99",
        "--min-balanced-duration-s", "2.0",
        "--seed", "0",
        "--dpi", "200",
    ]


def run_one(animal: str, command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> tuple[str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(shlex.join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(env), check=False)
    return animal, int(proc.returncode)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.metadata_excel = require_file(args.metadata_excel, "metadata Excel")
    args.phrase_csv = require_file(args.phrase_csv, "phrase-duration CSV")
    args.wrapper_script = require_file(args.wrapper_script, "BC wrapper script")
    args.export_script = require_file(args.export_script, "BC export script")
    args.spectrogram_script = require_file(args.spectrogram_script, "spectrogram helper")
    args.batch_script = require_file(args.batch_script, "batch script")
    args.output_root = args.output_root.expanduser().resolve()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output root already exists: {args.output_root}\n"
            "Use a new path or pass --allow-existing-output."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.npz_manifest.expanduser().resolve()) if args.npz_manifest else {}
    npz_by_bird: Dict[str, Path] = {}
    for bird in args.birds:
        path = manifest.get(bird) or discover_npz(args.data_root, bird)
        npz_by_bird[bird] = require_file(path, f"NPZ for {bird}")

    env = os.environ.copy()
    for key in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"

    logs = args.output_root / "_logs"
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_one,
                bird,
                build_command(args, bird, npz_by_bird[bird], args.output_root),
                logs / f"{bird}.log",
                env,
            ): bird
            for bird in args.birds
        }
        for future in as_completed(futures):
            bird, returncode = future.result()
            if returncode == 0:
                print(f"[OK] {bird}")
            else:
                failures.append(bird)
                print(f"[FAILED] {bird}; see {logs / f'{bird}.log'}")

    if failures:
        raise RuntimeError(f"Per-bird analyses failed: {', '.join(failures)}")

    if not args.skip_batch:
        batch_out = args.output_root / "_batch_figure4_selected_bins_median_final"
        command = [
            args.python, str(args.batch_script),
            "--bc-root", str(args.output_root),
            "--metadata-excel", str(args.metadata_excel),
            "--out-dir", str(batch_out),
            "--metadata-sheet", "animal_hit_type_summary",
            "--metadata-animal-col", "Animal ID",
            "--metadata-hit-type-col", "Lesion hit type",
            "--bc-method", "selected_bins",
            "--bird-aggregate", "median",
            "--dpi", "300",
        ]
        print("[RUN] " + shlex.join(command))
        subprocess.run(command, env=env, check=True)
        print(f"[DONE] Batch results: {batch_out}")


if __name__ == "__main__":
    main()
