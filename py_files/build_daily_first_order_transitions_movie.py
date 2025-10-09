# build_daily_first_order_transitions_movie.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For movie writing
import imageio.v3 as iio  # GIF works without ffmpeg; MP4 tries imageio-ffmpeg backend

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import: prefer Excel-serial segments-aware organizer (no metadata)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Your new organizer (derives datetime from Excel serial in filename)
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _build_organized
    )
    _ORGANIZER_NAME = "organized_decoded_serialTS_segments"
except ImportError as e:
    raise ImportError(
        "Could not import 'organized_decoded_serialTS_segments'. "
        "Please ensure it is available on PYTHONPATH."
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────
def _try_int_for_sort(x):
    try:
        return int(x)
    except Exception:
        return x

def _as_str_labels(labels: Iterable) -> List[str]:
    return [str(l) for l in labels]

def _infer_animal_id(df: pd.DataFrame, decoded_path: Union[str, Path] | None = None) -> Optional[str]:
    """Best-effort animal ID."""
    if "Animal ID" in df.columns:
        non_null = df["Animal ID"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])
    if decoded_path:
        stem = Path(decoded_path).stem
        # Common pattern: <ANIMAL>_...
        if "_" in stem:
            return stem.split("_")[0]
        return stem
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core transition-matrix builder
# ──────────────────────────────────────────────────────────────────────────────
def build_first_order_transition_matrices(
    organized_df: pd.DataFrame,
    *,
    order_column: str = "syllable_order",
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if order_column not in organized_df.columns:
        raise KeyError(f"Expected column '{order_column}' in organized_df.")

    transition_counts = defaultdict(lambda: defaultdict(int))

    for order in organized_df[order_column]:
        if isinstance(order, list) and len(order) > 1:
            order_str = _as_str_labels(order)
            for i in range(len(order_str) - 1):
                curr_syll = order_str[i]
                next_syll = order_str[i + 1]
                transition_counts[curr_syll][next_syll] += 1

    all_labels = set(transition_counts.keys()) | {
        s for d in transition_counts.values() for s in d
    }
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
        for next_syll, c in nexts.items():
            if next_syll in counts.columns:
                counts.loc[curr_syll, next_syll] = c

    if min_row_total > 0 and not counts.empty:
        bad_rows = counts.sum(axis=1) < min_row_total
        counts.loc[bad_rows, :] = 0

    if counts.empty:
        return (counts, counts.astype(float))

    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0)

    return counts, probs


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper (also used for frame rendering)
# ──────────────────────────────────────────────────────────────────────────────
def plot_transition_matrix(
    probs: pd.DataFrame,
    *,
    title: Optional[str] = None,
    xlabel: str = "Next Syllable",
    ylabel: str = "Current Syllable",
    figsize: Tuple[float, float] = (9, 8),
    show: bool = True,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> None:
    if probs.empty:
        return

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        probs,
        cmap="binary",           # white (low) → black (high)
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Transition Probability"},
        square=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    plt.tight_layout()

    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def render_transition_frame_array(
    probs: pd.DataFrame,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> np.ndarray:
    """
    Render a heatmap to an RGB array for movie frames (no on-disk file needed).
    Uses the RGBA buffer to be compatible across Matplotlib versions/backends.
    """
    if probs.empty:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Render off-screen
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        probs,
        cmap="binary",
        vmin=0.0,
        vmax=1.0,
        cbar=False,   # cleaner frames
        square=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=8)
    plt.tight_layout(pad=0.5)

    # Draw and extract RGBA buffer (backend-safe)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()  # drop alpha; ensure memory-contiguous
    plt.close(fig)
    return rgb


# ──────────────────────────────────────────────────────────────────────────────
# Daily runner + movie maker (no metadata JSON required)
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_first_order_transitions(
    decoded_database_json: Union[str, Path],
    *,
    only_song_present: bool = False,
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
    output_dir: Optional[Union[str, Path]] = None,  # save per-day PNG/CSVs
    save_csv: bool = True,
    save_png: bool = True,
    show_plots: bool = True,
    # ── movie options ──
    save_movie_path: Optional[Union[str, Path]] = None,  # ".gif" recommended; ".mp4" tries ffmpeg
    movie_fps: int = 2,
    movie_figsize: Tuple[float, float] = (8, 7),
    enforce_consistent_order: bool = True,  # keep same label order across days
    # ── optional decoration ──
    treatment_date: Optional[str] = None,   # if provided, appended in titles
) -> Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]:
    """
    For each recording date with data, build & plot a first-order transition matrix.
    Optionally writes a movie where each frame is a day.

    Uses 'organized_decoded_serialTS_segments.build_organized_segments_with_durations'
    (Excel-serial timestamps; no metadata needed).

    Returns
    -------
    dict[date_str] -> {
        'counts': None,
        'probs':  DataFrame,
        'figure_path': Optional[Path],
        'counts_csv': None,
        'probs_csv': Optional[Path],
    }
    """
    # Build organized dataset (Excel-serial based; add_recording_datetime=True)
    org = _build_organized(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=None,   # ignored by organizer
        only_song_present=only_song_present,
        compute_durations=False,
        add_recording_datetime=True,
    )
    df = org.organized_df.copy()

    if "Date" not in df.columns:
        raise KeyError("Organized DataFrame missing 'Date' column.")
    if "syllable_order" not in df.columns:
        raise KeyError("Organized DataFrame missing 'syllable_order' column.")

    df = df[df["Date"].notna()].copy()
    df["DateOnly"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Best-effort animal ID
    animal_id = _infer_animal_id(df, decoded_database_json)

    output_dir = Path(output_dir) if output_dir else None
    results: Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]] = OrderedDict()

    # First pass: compute per-day matrices and collect union of labels (for consistent ordering)
    union_labels = set(_as_str_labels(restrict_to_labels)) if restrict_to_labels else set()
    day_rows = list(df.groupby("DateOnly", sort=True))
    per_day_probs: Dict[str, pd.DataFrame] = OrderedDict()

    for date_val, day_df in day_rows:
        date_str = pd.to_datetime(date_val).strftime("%Y.%m.%d")
        _, probs = build_first_order_transition_matrices(
            day_df,
            order_column="syllable_order",
            restrict_to_labels=restrict_to_labels,
            min_row_total=min_row_total,
        )
        # Skip days with no transitions
        if probs.empty or (probs.values.sum() == 0):
            continue

        per_day_probs[date_str] = probs.copy()
        if enforce_consistent_order:
            union_labels.update(per_day_probs[date_str].index.tolist())
            union_labels.update(per_day_probs[date_str].columns.tolist())

    # Decide canonical order
    if enforce_consistent_order:
        if restrict_to_labels is not None:
            # Respect user-provided order if list/tuple
            canonical = list(_as_str_labels(restrict_to_labels))
        else:
            canonical = sorted(list(union_labels), key=_try_int_for_sort)
    else:
        canonical = None  # each day uses its own order

    # Second pass: save plots/CSVs and build frames
    frames: List[np.ndarray] = []
    for date_str, probs in per_day_probs.items():
        # Reindex to canonical order for consistent frames
        if canonical:
            probs = probs.reindex(index=canonical, columns=canonical, fill_value=0)

        # Paths
        fig_path = probs_csv = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{date_str}_first_order"
            if save_png:
                fig_path = output_dir / f"{stem}_probs.png"
            if save_csv:
                probs_csv = output_dir / f"{stem}_probs.csv"

        # Title
        title_bits = []
        if animal_id:
            title_bits.append(animal_id)
        title_bits.append(f"{date_str} First-Order Transition Probabilities")
        # Optional treatment date (prefer explicit argument; else organizer’s attribute if present)
        td = treatment_date if treatment_date is not None else getattr(org, "treatment_date", None)
        if td:
            title_bits.append(f"(Treatment: {td})")
        title = " ".join(title_bits)

        # Save CSV
        if probs_csv:
            probs.to_csv(probs_csv)

        # Plot PNG for the day
        if fig_path or show_plots:
            plot_transition_matrix(
                probs,
                title=title,
                save_fig_path=fig_path,
                show=show_plots,
            )

        # Render movie frame
        if save_movie_path:
            frame = render_transition_frame_array(
                probs, title=title, figsize=movie_figsize
            )
            frames.append(frame)

        # Store result
        results[date_str] = {
            "counts": None,     # counts not stored in this pass to save memory
            "probs": probs,
            "figure_path": fig_path,
            "counts_csv": None,    # not produced in this variant
            "probs_csv": probs_csv,
        }

    # Write movie if requested
    if save_movie_path and frames:
        save_movie_path = Path(save_movie_path)
        save_movie_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_movie_path.suffix.lower()

        if suffix in (".gif", ".apng"):
            # GIF/APNG – no ffmpeg required
            iio.imwrite(save_movie_path, frames, fps=movie_fps)
        elif suffix in (".mp4", ".mov", ".mkv", ".webm"):
            # Try mp4/etc via imageio-ffmpeg/pyav backend
            try:
                with iio.imopen(save_movie_path, "w", plugin="pyav") as out:
                    out.init_video_stream(fps=movie_fps)
                    for fr in frames:
                        out.write_frame(fr)
            except Exception:
                # Fallback: may still require ffmpeg installed
                iio.imwrite(save_movie_path, frames, fps=movie_fps)
        else:
            raise ValueError(
                f"Unsupported movie extension '{suffix}'. Use '.gif' (recommended) or '.mp4'."
            )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Example usage (Spyder/console friendly)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/USA5288_decoded_database.json"

    out = run_daily_first_order_transitions(
        decoded_database_json=decoded,
        only_song_present=False,
        restrict_to_labels=None,                # e.g., ['0','1','2'] to lock axes
        min_row_total=0,                        # e.g., 5 to suppress sparse rows
        output_dir="figures/daily_transitions", # per-day PNG/CSV
        save_csv=True,
        save_png=True,
        show_plots=False,                       # disable live popups when making movies
        save_movie_path="figures/daily_transitions/first_order_daily.gif",
        movie_fps=2,
        movie_figsize=(8, 7),
        enforce_consistent_order=True,
        treatment_date=None,                    # or "2025-03-04"
    )
    print(f"Exported daily matrices for {len(out)} day(s).")


"""
# In Spyder console

import importlib
import build_daily_first_order_transitions_movie as daily
importlib.reload(daily)
from build_daily_first_order_transitions_movie import run_daily_first_order_transitions

# Input JSON
decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5323_decoded_database.json"

# Outputs
outdir = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures/daily_transitions"
movie  = f"{outdir}/first_order_daily.gif"

# Run (treatment_date is optional; remove or change as needed)
out = run_daily_first_order_transitions(
    decoded_database_json=decoded,
    only_song_present=False,
    restrict_to_labels=None,        # or a fixed list to lock axis order: ['0','1','2', ...]
    min_row_total=0,
    output_dir=outdir,              # per-day PNG/CSV goes here
    save_csv=False,
    save_png=True,
    show_plots=False,               # keep False when exporting movies
    save_movie_path=movie,          # ".gif" recommended
    movie_fps=2,
    movie_figsize=(8, 7),
    enforce_consistent_order=True,
    treatment_date=None,            # e.g., "2025-03-04"
)

print(f"Exported daily matrices for {len(out)} day(s).")

"""
