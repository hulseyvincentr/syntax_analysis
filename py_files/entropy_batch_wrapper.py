#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_batch_wrapper.py

Batch wrapper that loops over bird folders inside a ROOT directory and, for each bird:
  1) merges split songs + appends annotations (via entropy_by_syllable.build_merged_annotations_df)
  2) resolves treatment date (either provided OR looked up from metadata Excel for that bird)
  3) computes per-syllable H_a pre vs post
  4) computes total transition entropy TE pre vs post
  5) saves:
       - per-bird plot (H_a pre vs post, colored by syllable; readable legend; usage panel)
       - per-bird CSV (H_a table)
  6) writes a batch summary CSV across birds

OUTPUT FOLDER LOCATION (as you requested):
  Created as a *sibling* of root_dir, NOT inside it:
    out_dir = root_dir.parent / "entropy_figures"

Expected file layout inside each bird folder:
  <BIRD>/<something>_decoded_database.json
  <BIRD>/<something>_song_detection.json

This script is intentionally conservative:
  - if a folder is missing inputs or treatment-date lookup fails, it logs and skips that bird.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

import entropy_by_syllable as ebs


# ──────────────────────────────────────────────────────────────────────────────
# File discovery helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pick_best_match(files: List[Path], prefer_token: str) -> Optional[Path]:
    """
    If multiple files match a pattern, prefer ones containing the folder name token.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    prefer = [p for p in files if prefer_token.lower() in p.name.lower()]
    if prefer:
        # if multiple still, pick shortest name (often the canonical one)
        prefer.sort(key=lambda p: (len(p.name), p.name))
        return prefer[0]
    files.sort(key=lambda p: (len(p.name), p.name))
    return files[0]


def find_inputs_for_bird_folder(bird_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns (decoded_json, song_detection_json) or (None, None) if not found.
    """
    token = bird_dir.name

    decoded_candidates = []
    decoded_candidates.extend(list(bird_dir.glob("*_decoded_database.json")))
    decoded_candidates.extend(list(bird_dir.glob("*decoded_database*.json")))

    detect_candidates = []
    detect_candidates.extend(list(bird_dir.glob("*_song_detection.json")))
    detect_candidates.extend(list(bird_dir.glob("*song_detection*.json")))

    decoded = _pick_best_match(decoded_candidates, token)
    detect = _pick_best_match(detect_candidates, token)
    return decoded, detect


# ──────────────────────────────────────────────────────────────────────────────
# Core per-bird run
# ──────────────────────────────────────────────────────────────────────────────
def run_entropy_for_one_bird(
    *,
    bird_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    out_dir: Union[str, Path],
    treatment_date: Optional[str] = None,          # if provided, overrides Excel lookup
    sheet_name: Optional[str] = "metadata_with_hit_type",
    include_treatment_day: bool = False,
    ignore_labels: Optional[Sequence[str]] = ("silence", "-"),
    log_base: float = 2.0,
    min_transitions: int = 5,

    # merge settings:
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # plotting settings:
    legend_fontsize: float = 7.0,
    legend_ncol: int = 2,
    usage_log_scale: bool = True,

    stats_n_perm: int = 20000,
    stats_seed: int = 0,
) -> Dict[str, object]:
    bird_dir = Path(bird_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoded, song_detection = find_inputs_for_bird_folder(bird_dir)
    if decoded is None or song_detection is None:
        raise FileNotFoundError(
            f"Missing inputs in {bird_dir}. "
            f"decoded={decoded}, song_detection={song_detection}"
        )

    # 1) merge
    merged_df = ebs.build_merged_annotations_df(
        decoded_database_json=decoded,
        song_detection_json=song_detection,
        max_gap_between_song_segments=max_gap_between_song_segments,
        segment_index_offset=segment_index_offset,
        merge_repeated_syllables=merge_repeated_syllables,
        repeat_gap_ms=repeat_gap_ms,
        repeat_gap_inclusive=repeat_gap_inclusive,
    )

    # 2) treatment date resolution (arg OR Excel)
    treatment_dt, animal_id_used, src = ebs.resolve_treatment_date(
        treatment_date_arg=treatment_date,
        metadata_xlsx=Path(metadata_xlsx),
        animal_id_arg=bird_dir.name,      # folder name is usually the animal_id
        merged_df=merged_df,
        sheet_name=sheet_name,
    )

    # 3) organize
    organized_df = ebs.build_organized_from_merged_df(
        merged_df,
        compute_durations=False,          # not needed for entropy
        add_recording_datetime=True,
        treatment_date=treatment_dt,
    ).organized_df

    # 4) compute H_a table
    ha_df = ebs.compute_pre_post_Ha_all_syllables(
        organized_df,
        treatment_date=treatment_dt,
        include_treatment_day=include_treatment_day,
        collapse_repeats=True,
        ignore_labels=list(ignore_labels) if ignore_labels else None,
        log_base=log_base,
        min_transitions=min_transitions,
    )
    if ha_df.empty:
        raise ValueError(f"No syllables passed min_transitions={min_transitions} for {animal_id_used}")

    # 5) compute total transition entropy TE (pre vs post)
    TE_pre, TE_post = ebs.compute_pre_post_total_transition_entropy(
        organized_df,
        treatment_date=treatment_dt,
        include_treatment_day=include_treatment_day,
        collapse_repeats=True,
        ignore_labels=list(ignore_labels) if ignore_labels else None,
        log_base=log_base,
    )

    # Save per-bird CSV
    ha_csv = out_dir / f"{animal_id_used}__Ha_all_syllables_pre_post.csv"
    ha_df.to_csv(ha_csv, index=False)

    # Plot (entropy + usage panel, colored by syllable)
    plot_path = out_dir / f"{animal_id_used}__Ha_all_syllables__boxplot_by_label__with_usage.png"
    title = (
        f"{animal_id_used} | Hₐ pre vs post (colored by syllable)\n"
        f"TE_pre={TE_pre.total_entropy:.3g} (n={TE_pre.n_total_transitions}) | "
        f"TE_post={TE_post.total_entropy:.3g} (n={TE_post.n_total_transitions}) | "
        f"treat={treatment_dt.date()}"
    )

    stats = ebs.plot_Ha_boxplot_pre_post_by_label(
        ha_df,
        out_path=plot_path,
        title=title,
        connect_pairs=True,
        annotate_stats=True,
        stats_n_perm=stats_n_perm,
        stats_seed=stats_seed,
        legend_fontsize=legend_fontsize,
        legend_ncol=legend_ncol,
        show_usage_panel=True,
        usage_log_scale=usage_log_scale,
    )

    # Return a compact summary row for the batch CSV
    summary: Dict[str, object] = {
        "animal_id": animal_id_used,
        "bird_folder": str(bird_dir),
        "decoded_json": str(decoded),
        "song_detection_json": str(song_detection),
        "treatment_date": str(pd.to_datetime(treatment_dt).date()),
        "treatment_date_source": src,
        "n_syllables_plotted": int(len(ha_df)),
        "min_transitions": int(min_transitions),

        "TE_pre": float(TE_pre.total_entropy) if pd.notna(TE_pre.total_entropy) else None,
        "TE_post": float(TE_post.total_entropy) if pd.notna(TE_post.total_entropy) else None,
        "TE_delta_post_minus_pre": (
            float(TE_post.total_entropy - TE_pre.total_entropy)
            if pd.notna(TE_pre.total_entropy) and pd.notna(TE_post.total_entropy)
            else None
        ),
        "TE_n_transitions_pre": int(TE_pre.n_total_transitions),
        "TE_n_transitions_post": int(TE_post.n_total_transitions),

        "plot_path": str(plot_path),
        "ha_csv_path": str(ha_csv),
    }

    # Pull a few test stats fields if present
    if isinstance(stats, dict):
        for k in [
            "n_paired_syllables",
            "signflip_perm_p_two_sided",
            "wilcoxon_p_two_sided",
            "ttest_rel_p_two_sided",
            "mean_diff_post_minus_pre",
            "median_diff_post_minus_pre",
        ]:
            if k in stats:
                summary[k] = stats[k]

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Batch wrapper (what you asked for)
# ──────────────────────────────────────────────────────────────────────────────
def run_entropy_batch(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    treatment_date: Optional[str] = None,          # optional global override
    sheet_name: Optional[str] = "metadata_with_hit_type",
    out_dir: Optional[Union[str, Path]] = None,    # defaults to sibling entropy_figures

    # shared compute settings:
    include_treatment_day: bool = False,
    ignore_labels: Optional[Sequence[str]] = ("silence", "-"),
    log_base: float = 2.0,
    min_transitions: int = 5,

    # merge settings:
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # plotting settings:
    legend_fontsize: float = 7.0,
    legend_ncol: int = 2,
    usage_log_scale: bool = True,

    # stats:
    stats_n_perm: int = 20000,
    stats_seed: int = 0,

    # behavior:
    skip_folders: Sequence[str] = ("figures", "phrase_duration_pre_post_grouped", "entropy_figures"),
) -> pd.DataFrame:
    """
    Loops over immediate subdirectories of root_dir, treating each as a bird folder.
    Saves per-bird plots + CSVs into out_dir.

    Returns a DataFrame summary and also writes:
      out_dir / "entropy_batch_summary.csv"
    """
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir.parent / "entropy_figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bird_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    bird_dirs = [p for p in bird_dirs if p.name not in set(skip_folders)]

    summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for bd in sorted(bird_dirs, key=lambda p: p.name):
        try:
            row = run_entropy_for_one_bird(
                bird_dir=bd,
                metadata_xlsx=metadata_xlsx,
                out_dir=out_dir,
                treatment_date=treatment_date,
                sheet_name=sheet_name,
                include_treatment_day=include_treatment_day,
                ignore_labels=ignore_labels,
                log_base=log_base,
                min_transitions=min_transitions,
                max_gap_between_song_segments=max_gap_between_song_segments,
                segment_index_offset=segment_index_offset,
                merge_repeated_syllables=merge_repeated_syllables,
                repeat_gap_ms=repeat_gap_ms,
                repeat_gap_inclusive=repeat_gap_inclusive,
                legend_fontsize=legend_fontsize,
                legend_ncol=legend_ncol,
                usage_log_scale=usage_log_scale,
                stats_n_perm=stats_n_perm,
                stats_seed=stats_seed,
            )
            summaries.append(row)
            print(f"✅ {bd.name}: wrote plot + CSV")
        except Exception as e:
            print(f"⚠️  {bd.name}: skipped ({e})")
            failures.append({"bird_folder": str(bd), "error": str(e)})

    summary_df = pd.DataFrame(summaries)
    summary_csv = out_dir / "entropy_batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_df.to_csv(out_dir / "entropy_batch_failures.csv", index=False)

    print(f"\nDone. Summary → {summary_csv}")
    if failures:
        print(f"Failures → {out_dir / 'entropy_batch_failures.csv'}")

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL CLI
# ──────────────────────────────────────────────────────────────────────────────
def _main():
    import argparse
    p = argparse.ArgumentParser(description="Batch per-bird H_a + TE plots (pre vs post) into sibling entropy_figures/")
    p.add_argument("--root", required=True, type=str, help="Root directory containing bird folders")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Path to metadata Excel with Treatment date")
    p.add_argument("--sheet-name", default="metadata_with_hit_type", type=str, help="Excel sheet name (optional)")
    p.add_argument("--treatment-date", default=None, type=str, help="Optional override YYYY-MM-DD")
    p.add_argument("--min-transitions", default=5, type=int, help="Min transitions (n_a) to include syllable")
    p.add_argument("--no-usage-log", action="store_true", help="Disable log scale on usage panel")
    args = p.parse_args()

    run_entropy_batch(
        root_dir=args.root,
        metadata_xlsx=args.metadata_xlsx,
        treatment_date=args.treatment_date,
        sheet_name=args.sheet_name,
        min_transitions=args.min_transitions,
        usage_log_scale=not args.no_usage_log,
    )


if __name__ == "__main__":
    _main()


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT SPYDER CONSOLE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import sys, importlib

# If needed, add your code directory:
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import entropy_batch_wrapper as ebw
importlib.reload(ebw)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")

# This writes to: root_dir.parent / "entropy_figures"
summary_df = ebw.run_entropy_batch(
    root_dir=root_dir,
    metadata_xlsx=metadata_xlsx,
    sheet_name="metadata_with_hit_type",
    min_transitions=5,
    legend_fontsize=7,
    legend_ncol=2,
    usage_log_scale=True,
)

summary_df.head()
"""
