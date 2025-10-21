# -*- coding: utf-8 -*-
# Area_X_analysis_wrapper.py
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]


def _ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_labels(labels: Optional[str]) -> Optional[List[str]]:
    """
    Accepts a comma-separated string like '0,1,2,3' → ['0','1','2','3'].
    Returns None if labels is falsy.
    """
    if not labels:
        return None
    return [s.strip() for s in labels.split(",") if s.strip()]


def _as_tuple_wh(pair: Sequence[float]) -> Tuple[float, float]:
    if len(pair) != 2:
        raise ValueError("--movie_figsize needs exactly two numbers, e.g., --movie_figsize 8 7")
    return float(pair[0]), float(pair[1])


# ──────────────────────────────────────────────────────────────────────────────
# Individual steps (each wrapped to be robust if a module is missing)
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_transitions_movie(
    decoded: PathLike,
    out_root: PathLike,
    restrict_to_labels: Optional[Sequence[str]] = None,
    treatment_date: Optional[str] = None,
    min_row_total: int = 0,
    fps: int = 2,
    figsize: Tuple[float, float] = (8, 7),
    enforce_consistent_order: bool = True,
    show_plots: bool = False,
    reload_modules: bool = True,
) -> Optional[int]:
    """
    Builds per-day first-order transition matrices (PNGs/CSVs) and exports a GIF movie.
    Returns the number of days processed (or None if the module isn't available).
    """
    try:
        import build_daily_first_order_transitions_movie as daily
        if reload_modules:
            importlib.reload(daily)
        from build_daily_first_order_transitions_movie import run_daily_first_order_transitions
    except Exception as e:
        print("[daily_transitions] Skipped: could not import build_daily_first_order_transitions_movie:", e)
        return None

    outdir = _ensure_dir(Path(out_root) / "daily_transitions")
    movie = outdir / "first_order_daily.gif"

    out = run_daily_first_order_transitions(
        decoded_database_json=str(decoded),
        only_song_present=False,
        restrict_to_labels=list(restrict_to_labels) if restrict_to_labels else None,
        min_row_total=min_row_total,
        output_dir=str(outdir),      # per-day PNG/CSV
        save_csv=False,
        save_png=True,
        show_plots=show_plots,       # keep False when exporting movies
        save_movie_path=str(movie),  # ".gif" recommended
        movie_fps=fps,
        movie_figsize=figsize,
        enforce_consistent_order=enforce_consistent_order,
        treatment_date=treatment_date,   # e.g., "2025-03-04" or None
    )

    try:
        n_days = len(out)
    except Exception:
        n_days = None

    print(f"[daily_transitions] Movie: {movie}")
    if n_days is not None:
        print(f"[daily_transitions] Exported daily matrices for {n_days} day(s).")
    return n_days


def run_syllable_heatmap(
    decoded: PathLike,
    out_root: PathLike,
    treatment_date: Optional[str] = None,
    show: bool = True,
) -> Optional[Path]:
    """
    Builds a daily log-scaled syllable count heatmap PNG. Saves to figures/heatmap/.
    Returns the save_path (or None if modules are missing).
    """
    try:
        from organized_decoded_serialTS_segments import build_organized_segments_with_durations
    except Exception as e:
        print("[syllable_heatmap] Skipped: could not import organized_decoded_serialTS_segments:", e)
        return None

    try:
        from syllable_heatmap import (
            build_daily_avg_count_table,
            plot_log_scaled_syllable_counts,
            _infer_animal_id,
        )
    except Exception as e:
        print("[syllable_heatmap] Skipped: could not import syllable_heatmap helpers:", e)
        return None

    decoded = Path(decoded)
    outdir = _ensure_dir(Path(out_root) / "heatmap")

    out = build_organized_segments_with_durations(
        decoded_database_json=decoded,
        only_song_present=True,
        compute_durations=False,
        add_recording_datetime=True,
    )

    count_table = build_daily_avg_count_table(
        out.organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=out.unique_syllable_labels,
    )

    animal_id = _infer_animal_id(out.organized_df, decoded) or "unknown_animal"
    save_path = outdir / f"{animal_id}_syllable_heatmap.png"

    fig, ax = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,  # optional string like "YYYY-MM-DD" or "YYYY.MM.DD"
        save_path=save_path,
        show=show,
    )

    print(f"[syllable_heatmap] Saved to: {save_path}")
    return save_path


def run_phrase_duration_groups(
    decoded: PathLike,
    out_root: PathLike,
    treatment_date: str,
    grouping_mode: str = "auto_balance",     # or "explicit"
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    only_song_present: bool = True,
    restrict_to_labels: Optional[Sequence[str]] = None,
    y_max_ms: int = 40_000,
    show_plots: bool = True,
) -> bool:
    """
    Runs grouped phrase-duration analysis. Saves figures into figures/phrase_duration/.
    Returns True if executed, False if module missing.
    """
    try:
        from phrase_duration_pre_vs_post_grouped import run_phrase_duration_pre_vs_post_grouped
    except Exception as e:
        print("[phrase_duration] Skipped: could not import phrase_duration_pre_vs_post_grouped:", e)
        return False

    outdir = _ensure_dir(Path(out_root) / "phrase_duration")

    kwargs = dict(
        decoded_database_json=str(decoded),
        output_dir=str(outdir),
        treatment_date=treatment_date,
        grouping_mode=grouping_mode,
        only_song_present=only_song_present,
        restrict_to_labels=list(restrict_to_labels) if restrict_to_labels else None,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
    )

    if grouping_mode == "explicit":
        kwargs.update(
            early_group_size=early_group_size,
            late_group_size=late_group_size,
            post_group_size=post_group_size,
        )

    _ = run_phrase_duration_pre_vs_post_grouped(**kwargs)
    print(f"[phrase_duration] Ran with grouping_mode='{grouping_mode}'. Output dir: {outdir}")
    return True


def run_tte_by_day(
    decoded: PathLike,
    out_root: PathLike,
    treatment_date: Optional[str] = None,
    treatment_in: str = "post",
    only_song_present: bool = True,
    show: bool = True,
    min_songs_per_day: int = 5,
) -> bool:
    """
    Computes TTE by day, builds a time-course plot and an aggregate pre/post plot if treatment_date given.
    Returns True if executed, False if module missing.
    """
    try:
        import tte_by_day
        importlib.reload(tte_by_day)
        from tte_by_day import TTE_by_day
    except Exception as e:
        print("[tte_by_day] Skipped: could not import tte_by_day:", e)
        return False

    fig_dir = _ensure_dir(Path(out_root) / "tte_by_day")

    out = TTE_by_day(
        decoded_database_json=str(decoded),
        creation_metadata_json=None,
        only_song_present=only_song_present,
        fig_dir=str(fig_dir),
        show=show,
        min_songs_per_day=min_songs_per_day,
        treatment_date=treatment_date,
        treatment_in=treatment_in,
    )

    try:
        print(f"[tte_by_day] p-value: {out.prepost_p_value}, method: {out.prepost_test}")
        print(f"[tte_by_day] Time-course: {out.figure_path_timecourse}")
        print(f"[tte_by_day] Pre/Post box: {out.figure_path_prepost_box}")
    except Exception:
        pass

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Composite runner
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WrapperConfig:
    decoded: Path
    output_root: Path
    treatment_date: Optional[str]

    # Feature toggles
    skip_daily: bool = False
    skip_heatmap: bool = False
    skip_phrase: bool = False
    skip_tte: bool = False

    # Shared
    restrict_to_labels: Optional[List[str]] = None
    show_plots: bool = True

    # Daily transitions
    min_row_total: int = 0
    movie_fps: int = 2
    movie_figsize: Tuple[float, float] = (8.0, 7.0)
    enforce_consistent_order: bool = True
    reload_modules: bool = True

    # Phrase duration
    grouping_mode: str = "auto_balance"   # "auto_balance" or "explicit"
    early_group_size: int = 100
    late_group_size: int = 100
    post_group_size: int = 100
    y_max_ms: int = 40_000

    # TTE by day
    min_songs_per_day: int = 5
    treatment_in: str = "post"            # include treatment day with 'post' (or 'pre')


def run_all(cfg: WrapperConfig) -> None:
    print("============================================================")
    print("Area X Analysis Wrapper")
    print("Decoded JSON:", cfg.decoded)
    print("Output root  :", cfg.output_root)
    print("Treatment date:", cfg.treatment_date)
    print("============================================================")

    if not cfg.skip_daily:
        run_daily_transitions_movie(
            decoded=cfg.decoded,
            out_root=cfg.output_root,
            restrict_to_labels=cfg.restrict_to_labels,
            treatment_date=cfg.treatment_date,
            min_row_total=cfg.min_row_total,
            fps=cfg.movie_fps,
            figsize=cfg.movie_figsize,
            enforce_consistent_order=cfg.enforce_consistent_order,
            show_plots=False if cfg.treatment_date else False,  # keep False for movies
            reload_modules=cfg.reload_modules,
        )

    if not cfg.skip_heatmap:
        run_syllable_heatmap(
            decoded=cfg.decoded,
            out_root=cfg.output_root,
            treatment_date=cfg.treatment_date,
            show=cfg.show_plots,
        )

    if not cfg.skip_phrase and cfg.treatment_date:
        run_phrase_duration_groups(
            decoded=cfg.decoded,
            out_root=cfg.output_root,
            treatment_date=cfg.treatment_date,
            grouping_mode=cfg.grouping_mode,
            early_group_size=cfg.early_group_size,
            late_group_size=cfg.late_group_size,
            post_group_size=cfg.post_group_size,
            only_song_present=True,
            restrict_to_labels=cfg.restrict_to_labels,
            y_max_ms=cfg.y_max_ms,
            show_plots=cfg.show_plots,
        )
    elif not cfg.skip_phrase:
        print("[phrase_duration] Skipped: requires --treatment_date.")

    if not cfg.skip_tte:
        run_tte_by_day(
            decoded=cfg.decoded,
            out_root=cfg.output_root,
            treatment_date=cfg.treatment_date,
            treatment_in=cfg.treatment_in,
            only_song_present=True,
            show=cfg.show_plots,
            min_songs_per_day=cfg.min_songs_per_day,
        )

    print("============================================================")
    print("All requested analyses finished.")
    print("Outputs saved under:", cfg.output_root.resolve())
    print("============================================================")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wrapper to run Area X analyses on a single decoded JSON."
    )
    p.add_argument("--decoded", type=str, required=True,
                   help="Path to *_decoded_database.json.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Root output directory (default: <decoded>/figures).")
    p.add_argument("--treatment_date", type=str, default=None,
                   help='Optional treatment date (e.g., "2025-05-22").')

    # Which steps to skip
    p.add_argument("--skip_daily", action="store_true", help="Skip daily transitions movie.")
    p.add_argument("--skip_heatmap", action="store_true", help="Skip syllable heatmap.")
    p.add_argument("--skip_phrase", action="store_true", help="Skip phrase-duration grouping.")
    p.add_argument("--skip_tte", action="store_true", help="Skip TTE-by-day analysis.")

    # Shared/labels
    p.add_argument("--restrict_to_labels", type=str, default=None,
                   help='Comma-separated labels like "0,1,2,3". Omit to use all.')
    p.add_argument("--show_plots", action="store_true", help="Show interactive plots where applicable.")

    # Daily transitions
    p.add_argument("--min_row_total", type=int, default=0)
    p.add_argument("--movie_fps", type=int, default=2)
    p.add_argument("--movie_figsize", type=float, nargs=2, default=[8.0, 7.0],
                   help="Movie figure size, e.g. --movie_figsize 8 7")
    p.add_argument("--no_reload", action="store_true",
                   help="Do not reload modules before running daily transitions.")

    # Phrase duration
    p.add_argument("--grouping_mode", type=str, choices=["auto_balance", "explicit"],
                   default="auto_balance", help="Grouping strategy.")
    p.add_argument("--early_group_size", type=int, default=100)
    p.add_argument("--late_group_size", type=int, default=100)
    p.add_argument("--post_group_size", type=int, default=100)
    p.add_argument("--y_max_ms", type=int, default=40000)

    # TTE by day
    p.add_argument("--min_songs_per_day", type=int, default=5)
    p.add_argument("--treatment_in", type=str, choices=["pre", "post"], default="post",
                   help="Whether to include treatment day in 'pre' or 'post'.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    decoded = Path(args.decoded).expanduser().resolve()
    if not decoded.exists():
        raise FileNotFoundError(f"Decoded JSON not found: {decoded}")

    if args.output_dir:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        out_root = Path(decoded).parent / "figures"
    _ensure_dir(out_root)

    cfg = WrapperConfig(
        decoded=decoded,
        output_root=out_root,
        treatment_date=args.treatment_date,

        skip_daily=args.skip_daily,
        skip_heatmap=args.skip_heatmap,
        skip_phrase=args.skip_phrase,
        skip_tte=args.skip_tte,

        restrict_to_labels=_parse_labels(args.restrict_to_labels),
        show_plots=args.show_plots,

        min_row_total=args.min_row_total,
        movie_fps=args.movie_fps,
        movie_figsize=_as_tuple_wh(args.movie_figsize),
        enforce_consistent_order=True,
        reload_modules=not args.no_reload,

        grouping_mode=args.grouping_mode,
        early_group_size=args.early_group_size,
        late_group_size=args.late_group_size,
        post_group_size=args.post_group_size,
        y_max_ms=args.y_max_ms,

        min_songs_per_day=args.min_songs_per_day,
        treatment_in=args.treatment_in,
    )

    run_all(cfg)


if __name__ == "__main__":
    main()


"""
from pathlib import Path
from Area_X_analysis_wrapper import run_all, WrapperConfig

decoded = Path("/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R09_RC11_Comp1_and_laptop_decoded_database.json")
outdir  = Path("/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R09_figures")
tdate   = "2025-05-23"                 # or None

cfg = WrapperConfig(
    decoded=decoded,
    output_root=outdir,
    treatment_date=tdate,

    # toggle steps (False = run; True = skip)
    skip_daily=False,
    skip_heatmap=False,
    skip_phrase=False,
    skip_tte=False,

    # labels (None = use all)
    restrict_to_labels=None,  # e.g. ['0','1','2','3']

    # daily transitions movie
    movie_fps=2,
    movie_figsize=(8, 7),
    min_row_total=0,

    # phrase duration
    grouping_mode="auto_balance",  # or "explicit"
    early_group_size=100,
    late_group_size=100,
    post_group_size=100,
    y_max_ms=40000,

    # TTE
    min_songs_per_day=5,
    treatment_in="post",

    # visuals
    show_plots=True,
)

run_all(cfg)

"""
