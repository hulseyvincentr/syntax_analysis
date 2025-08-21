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

from organize_decoded_with_durations import build_organized_dataset_with_durations


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
# Daily runner + movie maker
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_first_order_transitions(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
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
) -> Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]:
    """
    For each recording date with data, build & plot a first-order transition matrix.
    Optionally writes a movie where each frame is a day.

    Returns
    -------
    dict[date_str] -> {
        'counts': DataFrame,
        'probs':  DataFrame,
        'figure_path': Optional[Path],
        'counts_csv': Optional[Path],
        'probs_csv': Optional[Path],
    }
    """
    org = build_organized_dataset_with_durations(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=False,
    )
    df = org.organized_df.copy()

    if "Date" not in df.columns:
        raise KeyError("Organized DataFrame missing 'Date' column.")
    df = df[df["Date"].notna()].copy()
    df["DateOnly"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Best-effort animal ID
    animal_id = None
    if "Animal ID" in df.columns:
        non_null = df["Animal ID"].dropna()
        if not non_null.empty:
            animal_id = str(non_null.iloc[0])

    output_dir = Path(output_dir) if output_dir else None
    results: Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]] = OrderedDict()

    # First pass: compute per-day matrices and collect union of labels (for consistent ordering)
    union_labels = set(_as_str_labels(restrict_to_labels)) if restrict_to_labels else set()

    day_rows = list(df.groupby("DateOnly", sort=True))
    per_day_probs: Dict[str, pd.DataFrame] = OrderedDict()

    for date_val, day_df in day_rows:
        date_str = pd.to_datetime(date_val).strftime("%Y.%m.%d")
        counts, probs = build_first_order_transition_matrices(
            day_df,
            order_column="syllable_order",
            restrict_to_labels=restrict_to_labels,
            min_row_total=min_row_total,
        )
        # Skip days with no transitions
        if counts.empty or (counts.values.sum() == 0):
            continue

        per_day_probs[date_str] = probs.copy()
        if enforce_consistent_order:
            union_labels.update(per_day_probs[date_str].index.tolist())
            union_labels.update(per_day_probs[date_str].columns.tolist())

    # Decide canonical order
    if enforce_consistent_order:
        if restrict_to_labels is not None:
            # respect user-provided order if it's a list/tuple
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
        fig_path = counts_csv = probs_csv = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{date_str}_first_order"
            if save_png:
                fig_path = output_dir / f"{stem}_probs.png"
            if save_csv:
                counts_csv = output_dir / f"{stem}_counts.csv"
                probs_csv  = output_dir / f"{stem}_probs.csv"

        # Titles
        title_bits = []
        if animal_id:
            title_bits.append(animal_id)
        title_bits.append(f"{date_str} First-Order Transition Probabilities")
        if org.treatment_date:
            title_bits.append(f"(Treatment: {org.treatment_date})")
        title = " ".join(title_bits)

        # Save CSVs (recompute counts from probs' row-sums is lossy; skip – only probs here)
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
            "counts": None,     # counts aren’t kept in this second pass to save memory
            "probs": probs,
            "figure_path": fig_path,
            "counts_csv": counts_csv,
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
            # Try mp4/etc via imageio-ffmpeg backend (requires ffmpeg)
            try:
                with iio.imopen(save_movie_path, "w", plugin="pyav") as out:
                    out.init_video_stream(fps=movie_fps)
                    for fr in frames:
                        out.write_frame(fr)
            except Exception:
                # Fallback: try legacy writer (may still require ffmpeg)
                iio.imwrite(save_movie_path, frames, fps=movie_fps)  # may raise if backend missing
        else:
            raise ValueError(
                f"Unsupported movie extension '{suffix}'. Use '.gif' (recommended) or '.mp4'."
            )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/USA5288_decoded_database.json"
    meta    = "/path/to/USA5288_creation_data.json"

    out = run_daily_first_order_transitions(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        only_song_present=False,
        restrict_to_labels=None,                # e.g., ['0','1','2'] for fixed subset
        min_row_total=0,                        # e.g., 5 to suppress sparse rows
        output_dir="figures/daily_transitions", # per-day PNG/CSV
        save_csv=True,
        save_png=True,
        show_plots=False,                       # disable live popups when making movies
        save_movie_path="figures/daily_transitions/first_order_daily.gif",
        movie_fps=2,
        movie_figsize=(8, 7),
        enforce_consistent_order=True,
    )
    print(f"Exported daily matrices for {len(out)} day(s).")

"""
from build_daily_first_order_transitions_movie import run_daily_first_order_transitions

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

_ = run_daily_first_order_transitions(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    only_song_present=False,
    restrict_to_labels=None,   # or your canonical list to fix label order
    min_row_total=0,
    output_dir="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures/daily_transitions",
    save_csv=False,
    save_png=True,
    show_plots=False,  # off when exporting movies
    save_movie_path="/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures/daily_transitions/first_order_daily.gif",
    movie_fps=2,
    enforce_consistent_order=True,
)


"""
