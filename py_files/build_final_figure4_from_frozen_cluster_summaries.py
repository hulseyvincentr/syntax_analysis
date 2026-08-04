#!/usr/bin/env python3
"""Build the accepted final Figure 4 analysis from frozen cluster summaries.

The final manuscript comparison retains the original accepted summaries for the
17 previously analyzed birds and adds R08, R09, and R10 from the expanded
lateral-only analysis. This script assembles that input set transparently and
runs the final selected-bin, median-within-bird batch analysis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, List


ORIGINAL_BIRDS = [
    "USA5271", "USA5283", "USA5288", "USA5325", "USA5326",
    "USA5336", "USA5337", "USA5347", "USA5371", "USA5443",
    "USA5468", "USA5483", "USA5494", "USA5499", "USA5506",
    "USA5509", "USA5510",
]
ADDED_LATERAL_BIRDS = ["R08", "R09", "R10"]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Assemble frozen original summaries plus R08/R09/R10 and run final Figure 4 plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--original-summary-root", type=Path, required=True)
    p.add_argument("--added-lateral-summary-root", type=Path, required=True)
    p.add_argument("--metadata-excel", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--batch-script",
        type=Path,
        default=here / "bc_batch_lesion_group_summary_with_remaining_v8_dynamic_ylims_whisker_brackets_ML_combined_final.py",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def summary_path(root: Path, bird: str) -> Path:
    return root / bird / f"{bird}_cluster_bc_summary.csv"


def copy_summary(source_root: Path, output_root: Path, bird: str, source_label: str, rows: List[dict]) -> None:
    source = summary_path(source_root, bird)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty summary for {bird}: {source}")
    destination_dir = output_root / bird
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    rows.append({"animal_id": bird, "source_label": source_label, "source_csv": str(source)})


def main() -> None:
    args = parse_args()
    original = args.original_summary_root.expanduser().resolve()
    added = args.added_lateral_summary_root.expanduser().resolve()
    metadata = args.metadata_excel.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    batch_script = args.batch_script.expanduser().resolve()

    for path, label in [(original, "original summary root"), (added, "added-lateral summary root")]:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for path, label in [(metadata, "metadata Excel"), (batch_script, "batch script")]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest_rows: List[dict] = []
    for bird in ORIGINAL_BIRDS:
        copy_summary(original, output, bird, "original_accepted", manifest_rows)
    for bird in ADDED_LATERAL_BIRDS:
        copy_summary(added, output, bird, "added_lateral", manifest_rows)

    manifest = output / "frozen_summary_source_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["animal_id", "source_label", "source_csv"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    batch_out = output / "_batch_lesion_group_summaries_dynamic_ylims_ML_combined"
    command = [
        args.python, str(batch_script),
        "--bc-root", str(output),
        "--metadata-excel", str(metadata),
        "--out-dir", str(batch_out),
        "--metadata-sheet", "animal_hit_type_summary",
        "--metadata-animal-col", "Animal ID",
        "--metadata-hit-type-col", "Lesion hit type",
        "--bc-method", "selected_bins",
        "--bird-aggregate", "median",
        "--dpi", "300",
    ]
    subprocess.run(command, check=True)
    print(f"[DONE] Frozen-summary manifest: {manifest}")
    print(f"[DONE] Final batch output: {batch_out}")


if __name__ == "__main__":
    main()
