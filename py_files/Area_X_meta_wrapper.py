# Area_X_meta_wrapper.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for Area X analyses using a metadata Excel sheet.

- Reads an Excel with at least:
    * "Animal ID" (default column name; configurable)
    * "Treatment date" (default column name; configurable)
- For each unique Animal ID, searches a root directory recursively for:
    * a decoded annotations JSON matching pattern:  *_decoded_database.json
    * a song detection JSON     matching pattern:  *_song_detection.json
  (Both patterns are configurable.)
- Uses the Excel treatment date for that animal and calls
  Area_X_wrapper.run_area_x_meta_bundle(...) once per animal.

Example (Python):
    from pathlib import Path
    from Area_X_meta_wrapper import Area_X_meta_wrapper

    results = Area_X_meta_wrapper(
        metadata_excel=Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_lesion_metadata.xlsx"),
        json_root_dir=Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs"),
        #output_root="/path/to/figures_root",   # optional; per-animal subfolders will be made
        # (optional) override column names if your sheet uses different headers:
        animal_id_col="Animal ID",
        treatment_date_col="Treatment date",
        # (optional) dry run to just see matches without running analyses:
        dry_run=False,
        # (optional) any kwargs below are forwarded to run_area_x_meta_bundle(...)
        max_gap_between_song_segments=500,
        merged_song_gap_ms=500,
        include_other_bin=True,
        include_other_in_hist=True,
        show=True,
    )

CLI:

    python Area_X_meta_wrapper.py \
        --excel /path/to/metadata.xlsx \
        --root  /path/to/all_jsons_root \
        --out   /path/to/figures_root \
        --animal-col "Animal ID" \
        --date-col "Treatment date" \
        --dry-run 0

Outputs:
- A list of per-animal result objects (success or failure + paths)
- Optional CSV summary at <output_root>/AreaX_meta_runs.csv
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd

# Import the single-animal runner
import Area_X_wrapper as axw


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnimalRunResult:
    animal_id: str
    treatment_date: Optional[str]
    decoded_path: Optional[str]
    detection_path: Optional[str]
    output_dir: Optional[str]
    success: bool
    error: Optional[str] = None
    # Optional highlights you may care about:
    last_syllable_hist_path: Optional[str] = None
    last_syllable_pies_path: Optional[str] = None
    preceder_successor_panels: int = 0
    durations_prepost_path: Optional[str] = None
    durations_three_group_path: Optional[str] = None
    heatmap_v1_path: Optional[str] = None
    heatmap_vmax_path: Optional[str] = None
    tte_timecourse_path: Optional[str] = None
    tte_three_group_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm(s: Union[str, Path]) -> str:
    return str(Path(s)).lower()

def _parse_treatment_date(val: Any) -> Optional[str]:
    """
    Robustly parse a date-like value to ISO YYYY-MM-DD string. Returns None if parsing fails.
    """
    if val is None:
        return None
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return str(ts.normalize().date())
    except Exception:
        return None

def _find_one_by_animal(root: Path, glob_pattern: str, animal_id: str) -> Optional[Path]:
    """
    Find exactly one file under root whose path contains the animal_id (case-insensitive)
    and matches the glob pattern (searched recursively). If multiple, pick the most
    recently modified one.
    """
    animal_low = animal_id.lower()
    candidates = [p for p in root.rglob(glob_pattern) if animal_low in _norm(p)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer most recent mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def Area_X_meta_wrapper(
    *,
    metadata_excel: Union[str, Path],
    json_root_dir: Union[str, Path],
    output_root: Optional[Union[str, Path]] = None,

    # Excel column names (customize if needed)
    animal_id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",

    # File search patterns
    detection_glob: str = "*song_detection.json",
    decoded_glob: str = "*decoded_database.json",

    # Behavior
    dry_run: bool = False,
    write_summary_csv: bool = True,

    # Forwarded to axw.run_area_x_meta_bundle(...)
    **bundle_kwargs,
) -> List[AnimalRunResult]:
    """
    Batch-run Area X analyses for all animals in a metadata Excel.
    Returns a list of AnimalRunResult objects (one per animal).
    """
    metadata_excel = Path(metadata_excel)
    json_root_dir = Path(json_root_dir)

    if output_root is not None:
        output_root = _ensure_dir(output_root)

    # Load Excel
    df = pd.read_excel(metadata_excel)
    if animal_id_col not in df.columns:
        raise ValueError(f"Column '{animal_id_col}' not found in {metadata_excel}")
    if treatment_date_col not in df.columns:
        raise ValueError(f"Column '{treatment_date_col}' not found in {metadata_excel}")

    # Build a map: animal_id -> parsed treatment date (earliest if multiple)
    results: List[AnimalRunResult] = []
    grouped = df.groupby(animal_id_col, dropna=True)

    for animal, sub in grouped:
        if pd.isna(animal):
            continue
        animal_str = str(animal).strip()
        # Choose earliest non-null parsed date (warn if multiple)
        dates = [d for d in ( _parse_treatment_date(v) for v in sub[treatment_date_col].tolist() ) if d is not None]
        tdate = min(dates) if dates else None

        # Locate files
        decoded_path  = _find_one_by_animal(json_root_dir, decoded_glob,  animal_str)
        detection_path = _find_one_by_animal(json_root_dir, detection_glob, animal_str)

        # Decide output dir per animal
        if output_root is not None:
            outdir = _ensure_dir(Path(output_root) / animal_str)
        else:
            # default next to decoded file (or under root if missing)
            base = Path(decoded_path).parent if decoded_path else json_root_dir
            outdir = _ensure_dir(base / "figures" / animal_str)

        # Dry run: just report matches
        if dry_run:
            results.append(AnimalRunResult(
                animal_id=animal_str,
                treatment_date=tdate,
                decoded_path=None if decoded_path is None else str(decoded_path),
                detection_path=None if detection_path is None else str(detection_path),
                output_dir=str(outdir),
                success=(decoded_path is not None and detection_path is not None and tdate is not None),
                error=None if (decoded_path and detection_path and tdate) else
                      ("Missing decoded" if not decoded_path else
                       "Missing detection" if not detection_path else
                       "Missing/invalid treatment date"),
            ))
            continue

        # Validate inputs for run
        if decoded_path is None or detection_path is None or tdate is None:
            err = ("Missing decoded" if decoded_path is None else
                   "Missing detection" if detection_path is None else
                   "Missing/invalid treatment date")
            results.append(AnimalRunResult(
                animal_id=animal_str, treatment_date=tdate,
                decoded_path=None if decoded_path is None else str(decoded_path),
                detection_path=None if detection_path is None else str(detection_path),
                output_dir=str(outdir), success=False, error=err
            ))
            print(f"[WARN] Skipping {animal_str}: {err}")
            continue

        # Run the single-animal bundle
        try:
            print(f"[RUN] {animal_str} | date={tdate}")
            res = axw.run_area_x_meta_bundle(
                song_detection_json=detection_path,
                decoded_annotations_json=decoded_path,
                treatment_date=tdate,
                base_output_dir=outdir,
                **bundle_kwargs,
            )

            # Collect highlights
            results.append(AnimalRunResult(
                animal_id=animal_str,
                treatment_date=tdate,
                decoded_path=str(decoded_path),
                detection_path=str(detection_path),
                output_dir=str(outdir),
                success=True,
                error=None,
                last_syllable_hist_path=res.last_syllable_hist_path,
                last_syllable_pies_path=res.last_syllable_pies_path,
                preceder_successor_panels=len(res.per_target_outputs or {}),
                durations_prepost_path=res.duration_plots.pre_post_path if res.duration_plots else None,
                durations_three_group_path=res.duration_plots.three_group_path if res.duration_plots else None,
                heatmap_v1_path=res.heatmaps.path_v1 if res.heatmaps else None,
                heatmap_vmax_path=res.heatmaps.path_vmax if res.heatmaps else None,
                tte_timecourse_path=res.tte_plots.timecourse_path if res.tte_plots else None,
                tte_three_group_path=res.tte_plots.three_group_path if res.tte_plots else None,
            ))
        except Exception as e:
            results.append(AnimalRunResult(
                animal_id=animal_str,
                treatment_date=tdate,
                decoded_path=str(decoded_path),
                detection_path=str(detection_path),
                output_dir=str(outdir),
                success=False,
                error=str(e),
            ))
            print(f"[ERROR] {animal_str}: {e}")

    # Optional summary CSV
    if write_summary_csv and (output_root is not None):
        summary_csv = Path(output_root) / "AreaX_meta_runs.csv"
        df_sum = pd.DataFrame([asdict(r) for r in results])
        df_sum.to_csv(summary_csv, index=False)
        print(f"[OK] Wrote summary: {summary_csv}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch Area X meta wrapper: run per-animal analyses using a metadata Excel.")
    p.add_argument("--excel",   type=str, required=True, help="Path to metadata Excel file.")
    p.add_argument("--root",    type=str, required=True, help="Root directory that contains all decoded/detected JSONs.")
    p.add_argument("--out",     type=str, default=None,  help="Output root (per-animal subfolders will be created).")
    p.add_argument("--animal-col", type=str, default="Animal ID", help="Excel column name for animal IDs.")
    p.add_argument("--date-col",   type=str, default="Treatment date", help="Excel column name for treatment dates.")
    p.add_argument("--detect-glob", type=str, default="*song_detection.json", help="Glob to find detection JSONs.")
    p.add_argument("--decoded-glob", type=str, default="*decoded_database.json", help="Glob to find decoded JSONs.")
    p.add_argument("--dry-run", type=int, default=0, help="1 = list matches only, do not run analyses.")
    p.add_argument("--no-summary", action="store_true", help="Do not write a summary CSV.")

    # A small set of common forwarded options (you can add more as needed)
    p.add_argument("--ann-gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")
    p.add_argument("--merged-song-gap-ms", type=int, default=500)
    p.add_argument("--include-other", action="store_true")
    p.add_argument("--include-other-in-hist", action="store_true")
    p.add_argument("--no-show", action="store_true")

    args = p.parse_args()

    _ = Area_X_meta_wrapper(
        metadata_excel=args.excel,
        json_root_dir=args.root,
        output_root=args.out,
        animal_id_col=args.animal_col,
        treatment_date_col=args.date_col,
        detection_glob=args.detect_glob,
        decoded_glob=args.decoded_glob,
        dry_run=bool(args.dry_run),
        write_summary_csv=not args.no_summary,
        # forwarded run options
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        merged_song_gap_ms=args.merged_song_gap_ms,
        include_other_bin=args.include_other,
        include_other_in_hist=args.include_other_in_hist,
        show=not args.no_show,
    )
