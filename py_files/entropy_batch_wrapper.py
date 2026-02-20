#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_batch_wrapper.py

Batch wrapper that loops over bird folders inside a ROOT directory and, for each bird, runs
the "do everything" pipeline in entropy_by_syllable.make_entropy_by_syllable_outputs(...).

Per bird, it produces (depending on toggles):
  - H_a boxplot (colored by syllable) + optional usage panel
  - (optional) "neighbors hist" directory: top-K preceding + following syllables for each target
  - (optional) mean_remaining_to_end_pre_vs_post.png (+ CSV), WITH legend
  - (optional) variance-tier boxplots:
        Ha_high_variance_pre_post.png
        Ha_low_variance_pre_post.png
  - H_a table CSV and (if enabled) mean_remaining CSV

And writes:
  - entropy_batch_summary.csv
  - entropy_batch_failures.csv  (if any birds fail)

Expected file layout inside each bird folder:
  <BIRD>/<something>_decoded_database.json
  <BIRD>/<something>_song_detection.json

OUTPUT FOLDER LOCATION:
  By default created as a *sibling* of root_dir:
      out_dir = root_dir.parent / "entropy_figures"
  And each bird gets its own subfolder:
      out_dir / <animal_id> / ...

Notes
-----
• This wrapper is conservative: if a bird folder is missing inputs or treatment-date lookup fails,
  it logs and skips that bird.
• Optional seasonality control via max_days_pre/max_days_post is passed through to entropy_by_syllable.
• Optional minimum-days checks (min_days_pre/min_days_post) are enforced here (requires a quick
  merge+organize pass to count unique recording dates). If you don't need that guard, leave them None.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

import entropy_by_syllable as ebs


# ──────────────────────────────────────────────────────────────────────────────
# File discovery helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pick_best_match(files: List[Path], prefer_token: str) -> Optional[Path]:
    """If multiple files match a pattern, prefer ones containing the folder-name token."""
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    prefer = [p for p in files if prefer_token.lower() in p.name.lower()]
    if prefer:
        prefer.sort(key=lambda p: (len(p.name), p.name))
        return prefer[0]
    files.sort(key=lambda p: (len(p.name), p.name))
    return files[0]


def find_inputs_for_bird_folder(bird_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Returns (decoded_json, song_detection_json) or (None, None) if not found."""
    token = bird_dir.name

    decoded_candidates: List[Path] = []
    decoded_candidates.extend(list(bird_dir.glob("*_decoded_database.json")))
    decoded_candidates.extend(list(bird_dir.glob("*decoded_database*.json")))

    detect_candidates: List[Path] = []
    detect_candidates.extend(list(bird_dir.glob("*_song_detection.json")))
    detect_candidates.extend(list(bird_dir.glob("*song_detection*.json")))

    decoded = _pick_best_match(decoded_candidates, token)
    detect = _pick_best_match(detect_candidates, token)
    return decoded, detect


def _n_unique_days(df: pd.DataFrame) -> int:
    if df is None or df.empty or "Date" not in df.columns:
        return 0
    d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return int(d.dropna().nunique())


# ──────────────────────────────────────────────────────────────────────────────
# Core per-bird run
# ──────────────────────────────────────────────────────────────────────────────
def run_entropy_for_one_bird(
    *,
    bird_dir: Union[str, Path],
    metadata_xlsx: Optional[Union[str, Path]] = None,
    out_dir: Union[str, Path],

    # Treatment date controls
    treatment_date: Optional[str] = None,          # if provided, overrides Excel lookup
    sheet_name: Optional[str] = "metadata_with_hit_type",
    include_treatment_day: bool = False,

    # Entropy controls
    ignore_labels: Optional[Sequence[str]] = ("silence", "-"),
    log_base: float = 2.0,
    min_transitions: int = 1,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,

    # Optional: minimum number of unique recording days in the (windowed) pre/post sets
    # (useful to avoid noisy / underpowered birds)
    min_days_pre: Optional[int] = None,
    min_days_post: Optional[int] = None,

    # Merge settings (split-song aware)
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # Neighbor histograms (prev+next)
    make_neighbors_hist: bool = True,
    neighbors_top_k: int = 12,
    neighbors_targets: Optional[Sequence[str]] = None,

    # Remaining-to-end plot
    make_mean_remaining_to_end: bool = True,
    mean_remaining_min_occurrences: int = 10,
    mean_remaining_legend_max_labels: int = 60,

    # Variance-tier plots (top X% variance vs rest)
    make_variance_tiers: bool = False,
    variance_path: Optional[Union[str, Path]] = None,
    variance_sheet_name: Optional[str] = None,
    variance_top_frac: float = 0.30,
    variance_use_pre_only: bool = True,
    variance_agg: str = "median",
    variance_min_rows_per_syllable: int = 5,

    show: bool = False,
) -> Dict[str, object]:
    bird_dir = Path(bird_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoded, song_detection = find_inputs_for_bird_folder(bird_dir)
    if decoded is None or song_detection is None:
        raise FileNotFoundError(
            f"Missing inputs in {bird_dir}. decoded={decoded}, song_detection={song_detection}"
        )

    # ── Optional min-days guard ────────────────────────────────────────────────
    # We do a quick merge + organize + split just to count unique pre/post days.
    # Then we run the full pipeline (which will merge again). This is redundant
    # but keeps the wrapper independent from internal return details.
    n_days_pre = None
    n_days_post = None
    treatment_dt_str = treatment_date

    if (min_days_pre is not None) or (min_days_post is not None):
        if treatment_date is None and metadata_xlsx is None:
            raise ValueError("min_days_pre/post requires either treatment_date OR metadata_xlsx to resolve lesion date.")

        merged_df = ebs.build_merged_annotations_df(
            decoded_database_json=decoded,
            song_detection_json=song_detection,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=segment_index_offset,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
        )

        t_dt, animal_id_used, _src = ebs.resolve_treatment_date(
            treatment_date_arg=treatment_date,
            metadata_xlsx=Path(metadata_xlsx) if metadata_xlsx else None,
            animal_id_arg=bird_dir.name,
            merged_df=merged_df,
            sheet_name=sheet_name,
        )

        # Use a normalized string so the full pipeline doesn't need to re-read Excel.
        treatment_dt_str = str(pd.to_datetime(t_dt).date())

        organized_df = ebs.build_organized_from_merged_df(
            merged_df,
            compute_durations=False,
            add_recording_datetime=True,
            treatment_date=t_dt,
        ).organized_df

        pre_df_w, post_df_w = ebs.split_pre_post(
            organized_df,
            treatment_date=t_dt,
            include_treatment_day=include_treatment_day,
            max_days_pre=max_days_pre,
            max_days_post=max_days_post,
        )
        n_days_pre = _n_unique_days(pre_df_w)
        n_days_post = _n_unique_days(post_df_w)

        if (min_days_pre is not None) and (n_days_pre < int(min_days_pre)):
            raise ValueError(
                f"{animal_id_used}: only {n_days_pre} pre days in window; requires min_days_pre={int(min_days_pre)}"
            )
        if (min_days_post is not None) and (n_days_post < int(min_days_post)):
            raise ValueError(
                f"{animal_id_used}: only {n_days_post} post days in window; requires min_days_post={int(min_days_post)}"
            )

    # ── Run the full per-bird pipeline ─────────────────────────────────────────
    res = ebs.make_entropy_by_syllable_outputs(
        decoded_database_json=decoded,
        song_detection_json=song_detection,
        metadata_xlsx=metadata_xlsx,
        metadata_sheet_name=sheet_name,

        # if we already resolved treatment date for min-days, pass it directly
        treatment_date=treatment_dt_str,
        animal_id=bird_dir.name,

        out_dir=out_dir,

        include_treatment_day=include_treatment_day,
        collapse_repeats=True,
        ignore_labels=list(ignore_labels) if ignore_labels else None,

        log_base=log_base,
        min_transitions=min_transitions,
        max_days_pre=max_days_pre,
        max_days_post=max_days_post,

        make_neighbors_hist=make_neighbors_hist,
        neighbors_top_k=neighbors_top_k,
        neighbors_targets=neighbors_targets,

        make_mean_remaining_to_end=make_mean_remaining_to_end,
        mean_remaining_min_occurrences=mean_remaining_min_occurrences,
        mean_remaining_legend_max_labels=mean_remaining_legend_max_labels,

        make_variance_tiers=make_variance_tiers,
        variance_path=variance_path,
        variance_sheet_name=variance_sheet_name,
        variance_top_frac=variance_top_frac,
        variance_use_pre_only=variance_use_pre_only,
        variance_agg=variance_agg,
        variance_min_rows_per_syllable=variance_min_rows_per_syllable,

        show=show,
    )

    # ── Build a compact summary row ────────────────────────────────────────────
    pre_TE = res.get("pre_TE", None)
    post_TE = res.get("post_TE", None)

    def _get(obj, attr, default=None):
        try:
            return getattr(obj, attr)
        except Exception:
            return default

    # TE values
    TE_pre = _get(pre_TE, "total_entropy", None)
    TE_post = _get(post_TE, "total_entropy", None)
    TE_n_pre = _get(pre_TE, "n_total_transitions", None)
    TE_n_post = _get(post_TE, "n_total_transitions", None)

    # variance tier info
    var = res.get("variance_tiers", None)
    var_n_top = None
    var_paths = {}
    if isinstance(var, dict):
        var_n_top = var.get("n_top_syllables", None)
        var_paths = var.get("paths", {}) if isinstance(var.get("paths", {}), dict) else {}

    paths = res.get("paths", {}) if isinstance(res.get("paths", {}), dict) else {}

    # `ha_df` is returned; use its length as plotted syllables (after min_transitions filter)
    ha_df = res.get("ha_df", None)
    n_syllables = int(len(ha_df)) if hasattr(ha_df, "__len__") and ha_df is not None else None

    summary: Dict[str, object] = {
        "animal_id": res.get("animal_id", bird_dir.name),
        "bird_folder": str(bird_dir),
        "decoded_json": str(decoded),
        "song_detection_json": str(song_detection),

        "treatment_date": res.get("treatment_date", None),
        "treatment_date_source": res.get("treatment_date_source", None),

        "max_days_pre": None if max_days_pre is None else int(max_days_pre),
        "max_days_post": None if max_days_post is None else int(max_days_post),
        "min_transitions": int(min_transitions),

        "n_syllables_in_ha_df": n_syllables,

        "TE_pre": float(TE_pre) if TE_pre is not None and pd.notna(TE_pre) else None,
        "TE_post": float(TE_post) if TE_post is not None and pd.notna(TE_post) else None,
        "TE_delta_post_minus_pre": (
            float(TE_post - TE_pre)
            if (TE_pre is not None and TE_post is not None and pd.notna(TE_pre) and pd.notna(TE_post))
            else None
        ),
        "TE_n_transitions_pre": int(TE_n_pre) if TE_n_pre is not None else None,
        "TE_n_transitions_post": int(TE_n_post) if TE_n_post is not None else None,

        # Optional day counts (only computed if min_days_pre/post were requested)
        "min_days_pre_required": None if min_days_pre is None else int(min_days_pre),
        "min_days_post_required": None if min_days_post is None else int(min_days_post),
        "n_days_pre_in_window": None if n_days_pre is None else int(n_days_pre),
        "n_days_post_in_window": None if n_days_post is None else int(n_days_post),

        # Key outputs
        "out_dir": res.get("out_dir", str(out_dir)),
        "Ha_plot_by_label_with_usage": paths.get("Ha_boxplot_by_label_with_usage", None),
        "Ha_plot_connected": paths.get("Ha_boxplot_connected", None),
        "Ha_table_csv": paths.get("Ha_table_csv", None),

        "neighbors_hist_dir": paths.get("neighbors_hist_dir", None),
        "mean_remaining_to_end_plot": paths.get("mean_remaining_to_end", None),
        "mean_remaining_to_end_csv": paths.get("mean_remaining_table_csv", None),

        # Variance-tier outputs
        "variance_tiers_dir": paths.get("variance_tiers_dir", None),
        "variance_top_frac": float(variance_top_frac) if make_variance_tiers else None,
        "variance_n_top_syllables": int(var_n_top) if var_n_top is not None else None,
        "variance_plot_high": var_paths.get("high", None) if var_paths else None,
        "variance_plot_low": var_paths.get("low", None) if var_paths else None,
        "variance_summary_csv": var_paths.get("variance_summary", None) if var_paths else None,
    }

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Batch wrapper
# ──────────────────────────────────────────────────────────────────────────────
def run_entropy_batch(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Optional[Union[str, Path]] = None,
    sheet_name: Optional[str] = "metadata_with_hit_type",
    out_dir: Optional[Union[str, Path]] = None,    # defaults to sibling entropy_figures

    # global override (rare; typically None so per-bird Excel lookup is used)
    treatment_date: Optional[str] = None,

    # entropy controls
    include_treatment_day: bool = False,
    ignore_labels: Optional[Sequence[str]] = ("silence", "-"),
    log_base: float = 2.0,
    min_transitions: int = 1,
    max_days_pre: Optional[int] = None,
    max_days_post: Optional[int] = None,

    # min-days guard (optional)
    min_days_pre: Optional[int] = None,
    min_days_post: Optional[int] = None,

    # merge controls
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # neighbors
    make_neighbors_hist: bool = True,
    neighbors_top_k: int = 12,
    neighbors_targets: Optional[Sequence[str]] = None,

    # remaining-to-end
    make_mean_remaining_to_end: bool = True,
    mean_remaining_min_occurrences: int = 10,
    mean_remaining_legend_max_labels: int = 60,

    # variance tiers
    make_variance_tiers: bool = False,
    variance_path: Optional[Union[str, Path]] = None,
    variance_sheet_name: Optional[str] = None,
    variance_top_frac: float = 0.30,
    variance_use_pre_only: bool = True,
    variance_agg: str = "median",
    variance_min_rows_per_syllable: int = 5,

    show: bool = False,

    # behavior
    skip_folders: Sequence[str] = ("figures", "phrase_duration_pre_post_grouped", "entropy_figures"),
) -> pd.DataFrame:
    """
    Loops over immediate subdirectories of root_dir, treating each as a bird folder.
    Writes per-bird outputs into: out_dir/<animal_id>/

    Writes:
      out_dir / "entropy_batch_summary.csv"
      out_dir / "entropy_batch_failures.csv" (if any)
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
            decoded, detect = find_inputs_for_bird_folder(bd)
            if decoded is None or detect is None:
                raise FileNotFoundError(f"Missing decoded/detect JSONs in {bd}")

            bird_out = out_dir / bd.name
            bird_out.mkdir(parents=True, exist_ok=True)

            row = run_entropy_for_one_bird(
                bird_dir=bd,
                metadata_xlsx=metadata_xlsx,
                out_dir=bird_out,

                treatment_date=treatment_date,
                sheet_name=sheet_name,
                include_treatment_day=include_treatment_day,

                ignore_labels=ignore_labels,
                log_base=log_base,
                min_transitions=min_transitions,
                max_days_pre=max_days_pre,
                max_days_post=max_days_post,

                min_days_pre=min_days_pre,
                min_days_post=min_days_post,

                max_gap_between_song_segments=max_gap_between_song_segments,
                segment_index_offset=segment_index_offset,
                merge_repeated_syllables=merge_repeated_syllables,
                repeat_gap_ms=repeat_gap_ms,
                repeat_gap_inclusive=repeat_gap_inclusive,

                make_neighbors_hist=make_neighbors_hist,
                neighbors_top_k=neighbors_top_k,
                neighbors_targets=neighbors_targets,

                make_mean_remaining_to_end=make_mean_remaining_to_end,
                mean_remaining_min_occurrences=mean_remaining_min_occurrences,
                mean_remaining_legend_max_labels=mean_remaining_legend_max_labels,

                make_variance_tiers=make_variance_tiers,
                variance_path=variance_path,
                variance_sheet_name=variance_sheet_name,
                variance_top_frac=variance_top_frac,
                variance_use_pre_only=variance_use_pre_only,
                variance_agg=variance_agg,
                variance_min_rows_per_syllable=variance_min_rows_per_syllable,

                show=show,
            )

            summaries.append(row)
            print(f"✅ {bd.name}: completed")
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Batch per-bird H_a + TE + neighbor hist + remaining-to-end + variance-tier plots."
    )
    p.add_argument("--root", required=True, type=str, help="Root directory containing bird folders")
    p.add_argument("--metadata-xlsx", default=None, type=str, help="Path to metadata Excel with Treatment date")
    p.add_argument("--sheet-name", default="metadata_with_hit_type", type=str, help="Excel sheet name (optional)")
    p.add_argument("--treatment-date", default=None, type=str, help="Optional override YYYY-MM-DD")

    # Windowing / seasonality
    p.add_argument("--max-days-pre", default=None, type=int, help="Restrict PRE to within this many days before lesion")
    p.add_argument("--max-days-post", default=None, type=int, help="Restrict POST to within this many days after lesion")
    p.add_argument("--include-treatment-day", action="store_true", help="Include lesion day in both pre/post splits")

    # Minimum-days guard
    p.add_argument("--min-days-pre", default=None, type=int, help="Require at least this many PRE recording days (windowed)")
    p.add_argument("--min-days-post", default=None, type=int, help="Require at least this many POST recording days (windowed)")

    # Entropy controls
    p.add_argument("--min-transitions", default=1, type=int, help="Min transitions (n_a) to include a syllable")
    p.add_argument("--log-base", default=2.0, type=float, help="Log base for entropy (2 -> bits)")
    p.add_argument("--ignore-labels", default="silence,-", type=str, help="Comma-separated labels to ignore")

    # Neighbor hist
    p.add_argument("--no-neighbors-hist", action="store_true", help="Disable neighbor histograms")
    p.add_argument("--neighbors-top-k", default=12, type=int, help="Top-K neighbors to show for prev/next hists")

    # Remaining-to-end plot
    p.add_argument("--no-mean-remaining", action="store_true", help="Disable mean_remaining_to_end plot")
    p.add_argument("--mean-remaining-min-occ", default=10, type=int, help="Min occurrences per syllable to include")
    p.add_argument("--mean-remaining-legend-max", default=60, type=int, help="Max labels in legend before omitting")

    # Variance tiers
    p.add_argument("--variance-path", default=None, type=str, help="CSV or Excel containing per-syllable variance points")
    p.add_argument("--variance-sheet", default=None, type=str, help="Excel sheet name for variance table")
    p.add_argument("--variance-top-frac", default=0.30, type=float, help="Top fraction (e.g., 0.30 = top 30%)")
    p.add_argument("--variance-agg", default="median", type=str, help="How to aggregate variance per syllable (median/mean)")
    p.add_argument("--variance-min-rows", default=5, type=int, help="Min rows per syllable in variance table")

    p.add_argument("--show", action="store_true", help="Show figures interactively (slower; not recommended in batch)")

    return p


def main():
    args = _build_arg_parser().parse_args()

    ignore = [s.strip() for s in str(args.ignore_labels).split(",") if s.strip()]

    make_var = bool(args.variance_path)

    run_entropy_batch(
        root_dir=args.root,
        metadata_xlsx=args.metadata_xlsx,
        sheet_name=args.sheet_name,
        treatment_date=args.treatment_date,

        include_treatment_day=bool(args.include_treatment_day),

        ignore_labels=ignore,
        log_base=float(args.log_base),
        min_transitions=int(args.min_transitions),

        max_days_pre=args.max_days_pre,
        max_days_post=args.max_days_post,
        min_days_pre=args.min_days_pre,
        min_days_post=args.min_days_post,

        make_neighbors_hist=not args.no_neighbors_hist,
        neighbors_top_k=int(args.neighbors_top_k),

        make_mean_remaining_to_end=not args.no_mean_remaining,
        mean_remaining_min_occurrences=int(args.mean_remaining_min_occ),
        mean_remaining_legend_max_labels=int(args.mean_remaining_legend_max),

        make_variance_tiers=make_var,
        variance_path=args.variance_path,
        variance_sheet_name=args.variance_sheet,
        variance_top_frac=float(args.variance_top_frac),
        variance_agg=str(args.variance_agg),
        variance_min_rows_per_syllable=int(args.variance_min_rows),
        variance_use_pre_only=True,  # default behavior; edit if you want both pre+post

        show=bool(args.show),
    )


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT SPYDER CONSOLE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import entropy_batch_wrapper as ebw
importlib.reload(ebw)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")
variance_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/aggregate_variance_points_by_bird_and_syllable.csv")

summary_df = ebw.run_entropy_batch(
    root_dir=root_dir,
    metadata_xlsx=metadata_xlsx,
    sheet_name="metadata_with_hit_type",

    # seasonality window
    max_days_pre=40,
    max_days_post=40,

    # optional guard
    min_days_pre=5,
    min_days_post=5,

    min_transitions=50,

    make_neighbors_hist=True,
    neighbors_top_k=12,

    make_mean_remaining_to_end=True,
    mean_remaining_min_occurrences=10,
    mean_remaining_legend_max_labels=60,

    make_variance_tiers=True,
    variance_path=variance_csv,
    variance_top_frac=0.30,
    variance_agg="median",
    variance_min_rows_per_syllable=5,
)

print(summary_df.head())
"""
