# -*- coding: utf-8 -*-
# daily_am_pm_first_order_transitions_movie.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v3 as iio  # GIF/APNG out-of-the-box; MP4 needs imageio-ffmpeg/ffmpeg

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import: prefer segments-aware, fallback to legacy durations organizer
# ──────────────────────────────────────────────────────────────────────────────
_USING_SEGMENTS = False
try:
    from organize_decoded_with_segments import (
        build_organized_segments_with_durations as _build_organized
    )
    _USING_SEGMENTS = True
except ImportError:
    from organize_decoded_with_durations import (
        build_organized_dataset_with_durations as _build_organized
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _try_int_for_sort(x):
    try:
        return int(x)
    except Exception:
        return x

def _as_str_labels(labels: Iterable) -> List[str]:
    return [str(l) for l in labels]

def _ensure_dir(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if not p:
        return None
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _pad_frames_to_same_size(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pad all frames to the same (max_h, max_w) with zeros (black).
    Accepts a list of HxWxC arrays with identical C.
    """
    if not frames:
        return frames
    max_h = max(fr.shape[0] for fr in frames)
    max_w = max(fr.shape[1] for fr in frames)
    c = frames[0].shape[2] if frames[0].ndim == 3 else 3

    out = []
    for fr in frames:
        h, w = fr.shape[:2]
        if h == max_h and w == max_w:
            out.append(fr)
            continue
        padded = np.zeros((max_h, max_w, c), dtype=fr.dtype)
        padded[:h, :w, ...] = fr
        out.append(padded)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# First-order transition matrices
# ──────────────────────────────────────────────────────────────────────────────
def build_first_order_transition_matrices(
    organized_df: pd.DataFrame,
    *,
    order_column: str = "syllable_order",
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, probs) row-stochastic first-order transitions."""
    if order_column not in organized_df.columns:
        raise KeyError(f"Expected column '{order_column}' in organized_df.")

    transition_counts = defaultdict(lambda: defaultdict(int))

    for order in organized_df[order_column]:
        if isinstance(order, list) and len(order) > 1:
            order_str = _as_str_labels(order)
            for i in range(len(order_str) - 1):
                transition_counts[order_str[i]][order_str[i + 1]] += 1

    all_labels = set(transition_counts.keys()) | {s for d in transition_counts.values() for s in d}
    all_labels = _as_str_labels(all_labels)

    if restrict_to_labels is not None:
        allowed = set(_as_str_labels(restrict_to_labels))
        all_labels = [l for l in all_labels if l in allowed]

    if not all_labels:
        return (pd.DataFrame(dtype=int), pd.DataFrame(dtype=float))

    sorted_labels = sorted(all_labels, key=_try_int_for_sort)
    counts = pd.DataFrame(0, index=sorted_labels, columns=sorted_labels, dtype=int)

    for curr_syll, nexts in transition_counts.items():
        if curr_syll not in counts.index:
            continue
        for nxt, c in nexts.items():
            if nxt in counts.columns:
                counts.loc[curr_syll, nxt] = c

    if min_row_total > 0 and not counts.empty:
        sparse = counts.sum(axis=1) < min_row_total
        counts.loc[sparse, :] = 0

    if counts.empty:
        return (counts, counts.astype(float))

    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0)
    return counts, probs


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers + frame rendering
# ──────────────────────────────────────────────────────────────────────────────
def render_single_heatmap_frame(
    probs: pd.DataFrame,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 7),
    fontsize: int = 8,
) -> np.ndarray:
    """Render a single heatmap to an RGB frame (backend-safe)."""
    if probs.empty:
        # small black frame to keep lists non-empty where expected
        return np.zeros((10, 10, 3), dtype=np.uint8)

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        probs, cmap="binary", vmin=0.0, vmax=1.0, cbar=False, square=True
    )
    ax.set_xlabel("Next Syllable"); ax.set_ylabel("Current Syllable")
    if title:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=fontsize)
    plt.tight_layout(pad=0.5)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def render_side_by_side_frame(
    left: Optional[pd.DataFrame],
    right: Optional[pd.DataFrame],
    *,
    left_title: str = "AM",
    right_title: str = "PM",
    supertitle: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
    fontsize: int = 8,
) -> np.ndarray:
    """Render two heatmaps side-by-side to one RGB frame. Empty halves show 'No data'."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, probs, ttl in [(axes[0], left, left_title), (axes[1], right, right_title)]:
        if probs is not None and not probs.empty:
            sns.heatmap(probs, cmap="binary", vmin=0.0, vmax=1.0, cbar=False, square=True, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=fontsize)
            ax.set_xlabel("Next Syllable"); ax.set_ylabel("Current Syllable")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=fontsize+2)
        ax.set_title(ttl)
    if supertitle:
        fig.suptitle(supertitle)
    plt.tight_layout(pad=0.8)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def _write_movie(path: Union[str, Path], frames: List[np.ndarray], fps: int) -> None:
    """Write GIF/APNG/MP4 with sensible defaults. Frames are padded to same size."""
    if not frames:
        return
    frames = _pad_frames_to_same_size(frames)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".gif", ".apng"):
        # duration = seconds per frame; loop=0 means infinite
        iio.imwrite(path, frames, duration=1 / max(fps, 1), loop=0)
    elif suffix in (".mp4", ".mov", ".mkv", ".webm"):
        # Requires ffmpeg backend (imageio-ffmpeg + ffmpeg on PATH)
        try:
            with iio.imopen(path, "w", plugin="pyav") as out:
                out.init_video_stream(fps=max(fps, 1))
                for fr in frames:
                    out.write_frame(fr)
        except Exception:
            # Fallback writer (may still require ffmpeg)
            iio.imwrite(path, frames, fps=max(fps, 1))
    else:
        raise ValueError(f"Unsupported movie extension: {suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# AM/PM runner + movies
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_am_pm_first_order_transitions_with_movies(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    only_song_present: bool = False,
    restrict_to_labels: Optional[Iterable[str]] = None,   # lock axes to this set/order if provided
    min_row_total: int = 0,
    # Outputs (PNG/CSV)
    output_dir: Optional[Union[str, Path]] = None,
    save_csv: bool = True,
    save_png: bool = True,
    show_plots: bool = False,
    # Movie outputs (set any subset you want)
    save_movie_both_path: Optional[Union[str, Path]] = None,   # AM+PM side by side (per-day frame)
    save_movie_am_path: Optional[Union[str, Path]] = None,     # AM-only sequential frames
    save_movie_pm_path: Optional[Union[str, Path]] = None,     # PM-only sequential frames
    save_movie_in_order_path: Optional[Union[str, Path]] = None,  # Day1 AM → Day1 PM → Day2 AM → ...
    movie_fps: int = 2,
    # Frame sizes
    single_figsize: Tuple[float, float] = (8, 7),
    pair_figsize: Tuple[float, float] = (14, 6),
    enforce_consistent_order: bool = True,
) -> Dict[str, Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]]:
    """
    Build AM/PM first-order matrices per day, save per-day outputs, and write
    four movies:
      • AM+PM paired frames per day  ( *_transition_matrix_am_pm_per_day.gif )
      • AM-only sequential frames    ( *_transition_matrix_am_only.gif )
      • PM-only sequential frames    ( *_transition_matrix_pm_only.gif )
      • In-order AM→PM sequence      ( *_transition_matrix_am_pm_in_order.gif )

    Returns
    -------
    results[date]['AM' or 'PM'] -> { 'probs': DF|None, 'figure_path': Path|None, ... }
    """
    # Build organized dataset (segments-aware or legacy)
    if _USING_SEGMENTS:
        org = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=False,
            add_recording_datetime=True,
        )
    else:
        org = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=False,
        )

    df = org.organized_df.copy()
    if "Date" not in df.columns or "Hour" not in df.columns:
        raise KeyError("Organized DataFrame must include 'Date' and 'Hour' columns.")

    df = df[df["Date"].notna()].copy()
    df["DateOnly"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["HourNum"] = pd.to_numeric(df["Hour"], errors="coerce")

    # Animal ID for titles & standardized filenames
    animal_id = None
    if "Animal ID" in df.columns:
        nn = df["Animal ID"].dropna()
        if not nn.empty:
            animal_id = str(nn.iloc[0])
    if not animal_id:
        animal_id = "unknown_animal"

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # If caller didn’t supply movie paths, build standardized ones using animal_id
    if out_dir:
        base = out_dir / f"{animal_id}_transition_matrix"
        if save_movie_both_path is None:
            save_movie_both_path = base.with_name(base.stem + "_am_pm_per_day.gif")
        if save_movie_am_path is None:
            save_movie_am_path = base.with_name(base.stem + "_am_only.gif")
        if save_movie_pm_path is None:
            save_movie_pm_path = base.with_name(base.stem + "_pm_only.gif")
        if save_movie_in_order_path is None:
            save_movie_in_order_path = base.with_name(base.stem + "_am_pm_in_order.gif")

    # First pass: compute AM/PM probs per day and gather union of labels
    union_labels = set(_as_str_labels(restrict_to_labels)) if restrict_to_labels else set()
    per_day_probs: Dict[str, Dict[str, pd.DataFrame]] = OrderedDict()

    for date_val, day_df in df.groupby("DateOnly", sort=True):
        date_str = pd.to_datetime(date_val).strftime("%Y.%m.%d")
        am_df = day_df[(day_df["HourNum"] >= 0) & (day_df["HourNum"] < 12)]
        pm_df = day_df[(day_df["HourNum"] >= 12) & (day_df["HourNum"] <= 23)]

        halves: Dict[str, pd.DataFrame] = {}
        for half_name, part_df in (("AM", am_df), ("PM", pm_df)):
            counts, probs = build_first_order_transition_matrices(
                part_df,
                order_column="syllable_order",
                restrict_to_labels=restrict_to_labels,
                min_row_total=min_row_total,
            )
            if not counts.empty and counts.values.sum() > 0:
                halves[half_name] = probs
                if enforce_consistent_order:
                    union_labels.update(probs.index.tolist())
                    union_labels.update(probs.columns.tolist())
        if halves:
            per_day_probs[date_str] = halves

    # Canonical axis order
    if enforce_consistent_order:
        if restrict_to_labels is not None:
            canonical = list(_as_str_labels(restrict_to_labels))
        else:
            canonical = sorted(list(union_labels), key=_try_int_for_sort)
    else:
        canonical = None

    # Second pass: save per-day outputs and build movie frames
    frames_both: List[np.ndarray] = []
    frames_am: List[np.ndarray] = []
    frames_pm: List[np.ndarray] = []
    frames_in_order: List[np.ndarray] = []

    results: Dict[str, Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]] = OrderedDict()

    for date_str, halves in per_day_probs.items():
        results[date_str] = {
            "AM": {"probs": None, "figure_path": None, "counts_csv": None, "probs_csv": None},
            "PM": {"probs": None, "figure_path": None, "counts_csv": None, "probs_csv": None},
        }

        # Reindex to canonical order
        probs_am = halves.get("AM")
        probs_pm = halves.get("PM")
        if canonical is not None:
            if probs_am is not None:
                probs_am = probs_am.reindex(index=canonical, columns=canonical, fill_value=0)
            if probs_pm is not None:
                probs_pm = probs_pm.reindex(index=canonical, columns=canonical, fill_value=0)

        # Titles
        base_title = f"{animal_id} {date_str}"
        title_am = f"{base_title} AM First-Order Transition Probabilities"
        title_pm = f"{base_title} PM First-Order Transition Probabilities"
        supertitle = f"{base_title}   AM (left) | PM (right)"
        if getattr(org, "treatment_date", None):
            title_am += f" (Treatment: {org.treatment_date})"
            title_pm += f" (Treatment: {org.treatment_date})"
            supertitle += f"  (Treatment: {org.treatment_date})"

        # Save CSV/PNG if requested
        if out_dir:
            if probs_am is not None and save_png:
                fp_am = out_dir / f"{date_str}_AM_first_order_probs.png"
                plt.figure(figsize=(9, 8))
                sns.heatmap(probs_am, cmap="binary", vmin=0, vmax=1, cbar_kws={"label": "Transition Probability"}, square=True)
                plt.title(title_am); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0, va="center"); plt.tight_layout()
                plt.savefig(fp_am, dpi=300, bbox_inches="tight"); plt.close()
                results[date_str]["AM"]["figure_path"] = fp_am
            if probs_pm is not None and save_png:
                fp_pm = out_dir / f"{date_str}_PM_first_order_probs.png"
                plt.figure(figsize=(9, 8))
                sns.heatmap(probs_pm, cmap="binary", vmin=0, vmax=1, cbar_kws={"label": "Transition Probability"}, square=True)
                plt.title(title_pm); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0, va="center"); plt.tight_layout()
                plt.savefig(fp_pm, dpi=300, bbox_inches="tight"); plt.close()
                results[date_str]["PM"]["figure_path"] = fp_pm
            if save_csv:
                if probs_am is not None:
                    csv_am = out_dir / f"{date_str}_AM_first_order_probs.csv"
                    probs_am.to_csv(csv_am); results[date_str]["AM"]["probs_csv"] = csv_am
                if probs_pm is not None:
                    csv_pm = out_dir / f"{date_str}_PM_first_order_probs.csv"
                    probs_pm.to_csv(csv_pm); results[date_str]["PM"]["probs_csv"] = csv_pm

        # Build frames for movies
        if probs_am is not None:
            frames_am.append(render_single_heatmap_frame(probs_am, title=title_am, figsize=single_figsize))
            results[date_str]["AM"]["probs"] = probs_am
        if probs_pm is not None:
            frames_pm.append(render_single_heatmap_frame(probs_pm, title=title_pm, figsize=single_figsize))
            results[date_str]["PM"]["probs"] = probs_pm

        # Paired frame (AM left, PM right). If a half is missing, it shows "No data".
        frame_both = render_side_by_side_frame(
            probs_am if probs_am is not None else None,
            probs_pm if probs_pm is not None else None,
            left_title="AM", right_title="PM", supertitle=supertitle, figsize=pair_figsize
        )
        frames_both.append(frame_both)

        # In-order sequence: AM then PM (include "No data" panels if missing so timeline is consistent)
        frames_in_order.append(
            render_single_heatmap_frame(probs_am if probs_am is not None else pd.DataFrame(), title=title_am, figsize=single_figsize)
        )
        frames_in_order.append(
            render_single_heatmap_frame(probs_pm if probs_pm is not None else pd.DataFrame(), title=title_pm, figsize=single_figsize)
        )

        # Optional live plots
        if show_plots:
            if probs_am is not None:
                plt.figure(figsize=(9, 8))
                sns.heatmap(probs_am, cmap="binary", vmin=0, vmax=1, cbar_kws={"label": "Transition Probability"}, square=True)
                plt.title(title_am); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0, va="center"); plt.tight_layout(); plt.show()
            if probs_pm is not None:
                plt.figure(figsize=(9, 8))
                sns.heatmap(probs_pm, cmap="binary", vmin=0, vmax=1, cbar_kws={"label": "Transition Probability"}, square=True)
                plt.title(title_pm); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0, va="center"); plt.tight_layout(); plt.show()

    # Write movies (filenames already standardized if out_dir was given)
    if save_movie_both_path and frames_both:
        _write_movie(save_movie_both_path, frames_both, movie_fps)
    if save_movie_am_path and frames_am:
        _write_movie(save_movie_am_path, frames_am, movie_fps)
    if save_movie_pm_path and frames_pm:
        _write_movie(save_movie_pm_path, frames_pm, movie_fps)
    if save_movie_in_order_path and frames_in_order:
        _write_movie(save_movie_in_order_path, frames_in_order, movie_fps)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/USA5288_decoded_database.json"
    meta    = "/path/to/USA5288_creation_data.json"

    _ = run_daily_am_pm_first_order_transitions_with_movies(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        restrict_to_labels=None,         # or your canonical ordered list, e.g. ['0','1',...]
        min_row_total=0,                 # e.g. 5 to suppress ultra-sparse rows
        output_dir="figures/daily_transitions_am_pm",
        save_csv=True,
        save_png=True,
        show_plots=False,
        # You can omit the next four to use the standardized animalID-based names:
        # save_movie_both_path="figures/daily_transitions_am_pm/USA5288_transition_matrix_am_pm_per_day.gif",
        # save_movie_am_path="figures/daily_transitions_am_pm/USA5288_transition_matrix_am_only.gif",
        # save_movie_pm_path="figures/daily_transitions_am_pm/USA5288_transition_matrix_pm_only.gif",
        # save_movie_in_order_path="figures/daily_transitions_am_pm/USA5288_transition_matrix_am_pm_in_order.gif",
        movie_fps=2,
        enforce_consistent_order=True,
    )



"""
from daily_am_pm_first_order_transitions_movie import run_daily_am_pm_first_order_transitions_with_movies

decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5323_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/USA5323_metadata.json"
_ = run_daily_am_pm_first_order_transitions_with_movies(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,
    restrict_to_labels=None,   # pass a list like ['0','1',...,'21'] to lock the axes order
    min_row_total=0,           # try 5 to zero ultra-sparse rows before normalization
    output_dir="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures/daily_transitions_am_pm",
    save_csv=False,
    save_png=False,
    show_plots=True,
    movie_fps=1,
    enforce_consistent_order=True,
)
"""
