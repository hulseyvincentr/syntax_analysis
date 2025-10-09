# -*- coding: utf-8 -*-

# batch_first_order_transitions_movie.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict
from collections import defaultdict, OrderedDict
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For movie writing
import imageio.v3 as iio  # GIF/APNG work without ffmpeg; MP4 tries imageio-ffmpeg/pyav


# ──────────────────────────────────────────────────────────────────────────────
# Organizer import (preference order):
#   1) organize_decoded_serialTS_segments  ← Excel-serial timestamp in filename
#   2) organized_decoded_serialTS_segments ← alt name (typo/back-compat)
#   3) organize_decoded_with_segments
#   4) organize_decoded_with_durations
# ──────────────────────────────────────────────────────────────────────────────
_ORGANIZE_FUNC = None
_ORGANIZER_NAME = None

try:
    from organize_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _ORGANIZE_FUNC  # type: ignore
    )
    _ORGANIZER_NAME = "organize_decoded_serialTS_segments"
except Exception:
    try:
        from organized_decoded_serialTS_segments import (
            build_organized_segments_with_durations as _ORGANIZE_FUNC  # type: ignore
        )
        _ORGANIZER_NAME = "organized_decoded_serialTS_segments"
    except Exception:
        try:
            from organize_decoded_with_segments import (
                build_organized_segments_with_durations as _ORGANIZE_FUNC  # type: ignore
            )
            _ORGANIZER_NAME = "organize_decoded_with_segments"
        except Exception:
            from organize_decoded_with_durations import (
                build_organized_dataset_with_durations as _ORGANIZE_FUNC  # type: ignore
            )
            _ORGANIZER_NAME = "organize_decoded_with_durations"


def _call_organizer(decoded_database_json: Union[str, Path],
                    creation_metadata_json: Optional[Union[str, Path]] = None):
    """
    Call whichever organizer we imported, passing only the kwargs it supports.
    This prefers the Excel-serial organizer, which does NOT need metadata.
    """
    if _ORGANIZE_FUNC is None:
        raise ImportError("No organizer function could be imported.")

    sig = inspect.signature(_ORGANIZE_FUNC)
    kwargs = {}

    # common/possible parameters
    if "decoded_database_json" in sig.parameters:
        kwargs["decoded_database_json"] = decoded_database_json
    if "creation_metadata_json" in sig.parameters:
        kwargs["creation_metadata_json"] = creation_metadata_json
    if "only_song_present" in sig.parameters:
        kwargs["only_song_present"] = False     # filter explicitly below
    if "compute_durations" in sig.parameters:
        kwargs["compute_durations"] = False     # we don't need durations here
    if "add_recording_datetime" in sig.parameters:
        kwargs["add_recording_datetime"] = True # Excel-serial organizer uses this

    return _ORGANIZE_FUNC(**kwargs)


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

def _window_ranges(n: int, batch_size: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Produce [(start, end_exclusive), ...] windows over n items.
    Non-overlapping by default; set stride < batch_size to overlap.
    """
    if stride is None or stride <= 0:
        stride = batch_size
    starts = range(0, max(n, 0), stride)
    return [(s, min(s + batch_size, n)) for s in starts if s < n]


# ──────────────────────────────────────────────────────────────────────────────
# Core transition-matrix builder
# ──────────────────────────────────────────────────────────────────────────────
def build_first_order_transition_matrices(
    df: pd.DataFrame,
    *,
    order_column: str = "syllable_order",
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build first-order transition counts/probabilities from df[order_column],
    where each row is a list of labels (length >= 2 to contribute transitions).
    """
    if order_column not in df.columns:
        raise KeyError(f"Expected column '{order_column}' in DataFrame.")

    transition_counts = defaultdict(lambda: defaultdict(int))

    for order in df[order_column]:
        if isinstance(order, list) and len(order) > 1:
            order_str = _as_str_labels(order)
            for i in range(len(order_str) - 1):
                a = order_str[i]
                b = order_str[i + 1]
                transition_counts[a][b] += 1

    all_labels = set(transition_counts.keys()) | {s for d in transition_counts.values() for s in d}
    all_labels = _as_str_labels(all_labels)

    if restrict_to_labels is not None:
        allowed = set(_as_str_labels(restrict_to_labels))
        all_labels = [l for l in all_labels if l in allowed]

    if not all_labels:
        return (pd.DataFrame(dtype=int), pd.DataFrame(dtype=float))

    sorted_labels = sorted(all_labels, key=_try_int_for_sort)
    counts = pd.DataFrame(0, index=sorted_labels, columns=sorted_labels, dtype=int)

    for a, nexts in transition_counts.items():
        if a not in counts.index:
            continue
        for b, c in nexts.items():
            if b in counts.columns:
                counts.loc[a, b] = c

    if min_row_total > 0 and not counts.empty:
        low = counts.sum(axis=1) < min_row_total
        counts.loc[low, :] = 0

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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
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

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner + movie maker (now using the Excel-serial organizer by default)
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_first_order_transitions(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Optional[Union[str, Path]] = None,   # kept for back-compat; ignored by serialTS organizer
    *,
    batch_size: int = 100,                  # songs per batch
    stride: Optional[int] = None,           # None → non-overlapping; set < batch_size for overlap
    restrict_to_labels: Optional[Iterable[str]] = None,
    min_row_total: int = 0,
    output_dir: Optional[Union[str, Path]] = None,   # save per-batch PNG/CSV
    save_csv: bool = True,
    save_png: bool = True,
    show_plots: bool = True,
    # movie options
    save_movie_path: Optional[Union[str, Path]] = None,  # ".gif" recommended; ".mp4" tries ffmpeg/pyav
    movie_fps: int = 2,
    movie_figsize: Tuple[float, float] = (8, 7),
    enforce_consistent_order: bool = True,  # keep same label order across batches
) -> Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]]:
    """
    Filters to rows with song_present == True, then for each consecutive batch of
    `batch_size` songs, builds and exports a first-order transition matrix. Also
    assembles frames into a movie if `save_movie_path` is provided.

    Uses the Excel-serial organizer if available (no metadata required).
    """
    # Build organized dataset (prefers Excel-serial organizer)
    org = _call_organizer(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=creation_metadata_json,
    )
    df = org.organized_df.copy()

    # Filter to song_present == True if available
    if "song_present" in df.columns:
        df = df[df["song_present"] == True].copy()

    if df.empty:
        raise ValueError("No rows with song_present == True found.")

    # Best-effort animal ID
    animal_id = None
    if "Animal ID" in df.columns:
        non_null = df["Animal ID"].dropna()
        if not non_null.empty:
            animal_id = str(non_null.iloc[0])

    df = df.reset_index(drop=True)  # define song order

    # Decide canonical label order (optional)
    canonical: Optional[List[str]] = None
    if enforce_consistent_order and restrict_to_labels is not None:
        # Respect user-given order if it's a sequence
        canonical = list(_as_str_labels(restrict_to_labels))

    # Prepare output aggregations
    output_dir = Path(output_dir) if output_dir else None
    results: Dict[str, Dict[str, Optional[Union[pd.DataFrame, Path]]]] = OrderedDict()
    frames: List[np.ndarray] = []
    union_labels = set(_as_str_labels(restrict_to_labels)) if restrict_to_labels else set()

    # Create windows over the filtered rows (non-overlapping by default)
    windows = _window_ranges(len(df), batch_size=batch_size, stride=stride)

    # First pass: compute probs & collect union labels if needed
    per_batch_probs: List[Tuple[str, pd.DataFrame]] = []
    for b_idx, (i, j) in enumerate(windows, start=1):
        batch_df = df.iloc[i:j]
        # Skip very small batches that cannot produce transitions
        counts, probs = build_first_order_transition_matrices(
            batch_df,
            order_column="syllable_order",
            restrict_to_labels=restrict_to_labels,
            min_row_total=min_row_total,
        )
        if counts.empty or (counts.values.sum() == 0):
            continue

        label = f"Batch_{b_idx:03d} ({i+1}–{j})"  # 1-based indices in label
        per_batch_probs.append((label, probs.copy()))

        if enforce_consistent_order and restrict_to_labels is None:
            union_labels.update(probs.index.tolist())
            union_labels.update(probs.columns.tolist())

    # Decide canonical order if requested and not provided
    if enforce_consistent_order and canonical is None:
        canonical = sorted(list(union_labels), key=_try_int_for_sort) if union_labels else None

    # Second pass: save plots/CSVs and build frames
    for label, probs in per_batch_probs:
        # Reindex to canonical order for consistent frames
        if enforce_consistent_order and canonical:
            probs = probs.reindex(index=canonical, columns=canonical, fill_value=0)

        # Paths
        fig_path = probs_csv = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = label.replace(" ", "_").replace("(", "").replace(")", "").replace("–", "-")
            if save_png:
                fig_path = output_dir / f"{stem}_probs.png"
            if save_csv:
                probs_csv = output_dir / f"{stem}_probs.csv"

        # Title
        title_bits = []
        if animal_id:
            title_bits.append(animal_id)
        title_bits.append(f"{label} First-Order Transition Probabilities")
        if getattr(org, "treatment_date", None):
            title_bits.append(f"(Treatment: {org.treatment_date})")
        title = " ".join(title_bits)

        # Save CSV
        if probs_csv:
            probs.to_csv(probs_csv)

        # Plot PNG for the batch
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
        results[label] = {
            "probs": probs,
            "figure_path": fig_path,
            "probs_csv": probs_csv,
        }

    # Write movie if requested
    if save_movie_path and frames:
        save_movie_path = Path(save_movie_path)
        save_movie_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_movie_path.suffix.lower()

        if suffix in (".gif", ".apng"):
            iio.imwrite(save_movie_path, frames, fps=movie_fps)
        elif suffix in (".mp4", ".mov", ".mkv", ".webm"):
            # Try pyav first (imageio plugin); fallback to generic writer
            try:
                with iio.imopen(save_movie_path, "w", plugin="pyav") as out:
                    out.init_video_stream(fps=movie_fps)
                    for fr in frames:
                        out.write_frame(fr)
            except Exception:
                iio.imwrite(save_movie_path, frames, fps=movie_fps)
        else:
            raise ValueError(
                f"Unsupported movie extension '{suffix}'. Use '.gif' (recommended) or '.mp4'."
            )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoded = "/path/to/decoded_database.json"
    # For the Excel-serial organizer, metadata is not required:
    meta    = None

    out = run_batch_first_order_transitions(
        decoded_database_json=decoded,
        creation_metadata_json=meta,         # kept for back-compat; ignored by serialTS organizer
        batch_size=100,
        stride=None,
        restrict_to_labels=None,
        min_row_total=0,
        output_dir="figures/batch_transitions",
        save_csv=True,
        save_png=True,
        show_plots=False,
        save_movie_path="figures/batch_transitions/first_order_batches.gif",
        movie_fps=2,
        movie_figsize=(8, 7),
        enforce_consistent_order=True,
    )
    print(f"Organizer used: {_ORGANIZER_NAME}")
    print(f"Exported batch matrices for {len(out)} batch(es).")


"""
from batch_first_order_transitions_movie import run_batch_first_order_transitions

# Paths to your inputs
decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_RC6_Comp2_decoded_database.json"
# The Excel-serial organizer doesn't need metadata; set to None.
meta    = None
outdir  = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_figures/"

# (Optional) Fix the label set/order across batches:
# canonical_labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']

# Run the batch transition pipeline
results = run_batch_first_order_transitions(
    decoded_database_json=decoded,
    creation_metadata_json=meta,             # ignored by the Excel-serial organizer
    batch_size=100,                          # group songs into batches of 100
    stride=None,                             # None = non-overlapping; e.g., 50 for rolling windows
    restrict_to_labels=None,                 # or use canonical_labels to lock axes
    min_row_total=0,                         # filter out rows with very few transitions (e.g., 5)
    output_dir=outdir,                       # save per-batch PNGs/CSVs here
    save_csv=False,
    save_png=True,
    show_plots=False,                        # don't pop up each figure interactively
    save_movie_path=f"{outdir}/first_order_batches.gif",  # final movie
    movie_fps=2,                             # frames per second
    movie_figsize=(8, 7),
    enforce_consistent_order=True,           # keep label order fixed across batches
)

print(f"Exported transition matrices for {len(results)} batches.")

# Inspect one batch
for batch_label, info in results.items():
    print(batch_label)
    print("PNG:", info["figure_path"])
    print("CSV:", info["probs_csv"])
    print(info["probs"].head())  # peek at transition probability table
    break  # just show the first batch



"""
