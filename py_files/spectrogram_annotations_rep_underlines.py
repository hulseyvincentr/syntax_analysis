#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectrogram_annotations_rep_underlines.py

Plot one continuous spectrogram segment with:
  1) a colored annotation/label stripe below the spectrogram
  2) alternating horizontal underline markers for each repetition of a target label
  3) x-axis tick labels in seconds

This is intended for inspecting possible stuttered syllable repetitions while
preserving song context.

Example:
python spectrogram_annotations_rep_underlines.py \
  --npz-path /path/to/USA5288.npz \
  --output-path /path/to/output/USA5288_label3_example.png \
  --target-label 3 \
  --start-frame 300627 \
  --end-frame 303869 \
  --label-key hdbscan_labels \
  --spectrogram-key s \
  --seconds-per-bin 0.0027 \
  --x-tick-step-s 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# -----------------------------
# Basic helpers
# -----------------------------
def _find_contiguous_runs(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find contiguous runs of identical labels in a 1D label array.

    Returns
    -------
    starts : np.ndarray
        Start index of each run, inclusive.
    ends : np.ndarray
        End index of each run, exclusive.
    run_labels : np.ndarray
        Label identity for each run.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if labels.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=labels.dtype),
        )

    change_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, labels.size]
    run_labels = labels[starts]
    return starts, ends, run_labels


def _build_default_label_color_lut(labels: np.ndarray) -> Dict[Any, str]:
    """
    Build a simple label -> color dictionary using tab20/tab20b/tab20c.
    Noise label -1 is gray.
    """
    uniq = sorted(np.unique(labels), key=lambda x: str(x))
    cmap_list = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )

    lut: Dict[Any, str] = {}
    color_i = 0
    for lab in uniq:
        if str(lab) == "-1":
            lut[lab] = "#7f7f7f"
        else:
            lut[lab] = mcolors.to_hex(cmap_list[color_i % len(cmap_list)])
            color_i += 1
    return lut


def _load_label_color_lut_from_file(path: Optional[Path]) -> Optional[Dict[Any, str]]:
    """
    Optional simple color LUT loader.

    Accepts a JSON file like:
      {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}

    Keys are kept as strings, but plotting also tries string fallbacks.
    """
    if path is None:
        return None
    import json

    with open(path, "r") as f:
        raw = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def _get_color_for_label(label_color_lut: Dict[Any, str], lab: Any) -> str:
    """Get color for lab, with string/int fallbacks."""
    if lab in label_color_lut:
        return label_color_lut[lab]
    if str(lab) in label_color_lut:
        return label_color_lut[str(lab)]
    try:
        lab_int = int(lab)
        if lab_int in label_color_lut:
            return label_color_lut[lab_int]
        if str(lab_int) in label_color_lut:
            return label_color_lut[str(lab_int)]
    except Exception:
        pass
    return "#cccccc"


def _labels_equal(a: Any, b: Any) -> bool:
    """Robust label comparison across string/int label representations."""
    if a == b:
        return True
    if str(a) == str(b):
        return True
    try:
        return int(a) == int(b)
    except Exception:
        return False


def _orient_spectrogram_to_FxT(S: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Orient spectrogram as frequency x time, matching labels length."""
    S = np.asarray(S)
    labels = np.asarray(labels)

    if S.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, got shape {S.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {labels.shape}.")

    if S.shape[1] == labels.size:
        return S
    if S.shape[0] == labels.size:
        return S.T

    raise ValueError(
        f"Could not orient spectrogram. Spectrogram shape is {S.shape}, "
        f"but labels length is {labels.size}. One spectrogram dimension must match labels length."
    )


def _choose_key(arr: np.lib.npyio.NpzFile, requested_key: Optional[str], candidates: Sequence[str], kind: str) -> str:
    """Choose an NPZ key, preferring user-requested key then candidates."""
    if requested_key is not None:
        if requested_key not in arr.files:
            raise KeyError(
                f"Requested {kind} key '{requested_key}' was not found in NPZ. "
                f"Available keys: {arr.files}"
            )
        return requested_key

    for key in candidates:
        if key in arr.files:
            return key

    raise KeyError(
        f"Could not find a {kind} key. Tried {list(candidates)}. "
        f"Available keys: {arr.files}"
    )


def _slice_segment(S: np.ndarray, labels: np.ndarray, start_frame: Optional[int], end_frame: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Slice S and labels to requested time-bin range."""
    T = labels.size
    start = 0 if start_frame is None else int(start_frame)
    end = T if end_frame is None else int(end_frame)

    if start < 0:
        raise ValueError("start_frame must be >= 0.")
    if end <= start:
        raise ValueError("end_frame must be greater than start_frame.")
    if end > T:
        raise ValueError(f"end_frame {end} is beyond available time bins T={T}.")

    return S[:, start:end], labels[start:end], start, end


def _compute_intensity_scale(
    S: np.ndarray,
    contrast_percentiles: Optional[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute vmin/vmax for imshow."""
    if contrast_percentiles is None:
        return None, None
    lo, hi = contrast_percentiles
    vmin, vmax = np.nanpercentile(S, [lo, hi])
    return float(vmin), float(vmax)


# -----------------------------
# Main plotting function
# -----------------------------
def plot_spectrogram_with_annotations_and_rep_underlines(
    spectrogram: np.ndarray,
    labels: np.ndarray,
    target_label: Any,
    *,
    seconds_per_bin: float = 0.0027,
    x_tick_step_s: float = 0.5,
    cmap: str = "gray_r",
    label_color_lut: Optional[Dict[Any, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 6),
    show_label_numbers_in_stripe: bool = False,
    show_target_label_on_rep_axis: bool = True,
    rep_text_fontsize: int = 8,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot a spectrogram with annotation stripe, alternating underlines, and seconds x-axis.

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram array, shape (F, T) or (T, F).
    labels : np.ndarray
        1D label array of length T.
    target_label : int or str
        Label whose contiguous repetitions should be marked.
    seconds_per_bin : float
        Time represented by one spectrogram column. For Miranda's data: 0.0027 seconds.
    x_tick_step_s : float
        Spacing between x-axis ticks in seconds.
    cmap : str
        Colormap for spectrogram.
    label_color_lut : dict or None
        Optional dict mapping labels to colors. If None, a default LUT is generated.
    title : str or None
        Figure title.
    figsize : tuple
        Figure size in inches.
    show_label_numbers_in_stripe : bool
        If True, write label numbers inside sufficiently wide annotation blocks.
    show_target_label_on_rep_axis : bool
        If True, write the target label number below each rep label.
    rep_text_fontsize : int
        Font size for repetition labels.
    save_path : Path or None
        Output image path.
    show_plot : bool
        Whether to display interactively.
    vmin, vmax : float or None
        Optional spectrogram intensity scaling.
    """
    S = _orient_spectrogram_to_FxT(np.asarray(spectrogram), np.asarray(labels))
    labels = np.asarray(labels)

    if label_color_lut is None:
        label_color_lut = _build_default_label_color_lut(labels)

    starts, ends, run_labels = _find_contiguous_runs(labels)

    fig, (ax_spec, ax_lbl, ax_rep) = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [12, 1.0, 1.9], "hspace": 0.03},
    )

    # Spectrogram axis
    ax_spec.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax_spec.set_ylabel("Freq bins", fontsize=11)
    ax_spec.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if title is not None:
        ax_spec.set_title(title, fontsize=14, pad=10)

    # Annotation color stripe axis
    ax_lbl.set_ylim(0, 1)
    ax_lbl.set_yticks([])
    ax_lbl.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for side in ["top", "right", "left", "bottom"]:
        ax_lbl.spines[side].set_visible(False)

    for start, end, lab in zip(starts, ends, run_labels):
        color = _get_color_for_label(label_color_lut, lab)
        ax_lbl.add_patch(
            Rectangle(
                (start, 0),
                end - start,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
        )
        if show_label_numbers_in_stripe and (end - start) >= 25:
            ax_lbl.text(
                (start + end) / 2,
                0.5,
                str(lab),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    ax_lbl.set_xlim(0, S.shape[1])

    # Repetition underline + time axis
    ax_rep.set_ylim(0, 1)
    ax_rep.set_yticks([])
    for side in ["top", "right", "left"]:
        ax_rep.spines[side].set_visible(False)

    rep_count = 0
    for start, end, lab in zip(starts, ends, run_labels):
        if not _labels_equal(lab, target_label):
            continue

        rep_count += 1
        center = (start + end) / 2
        duration_ms = (end - start) * seconds_per_bin * 1000

        # Underline odd-numbered repetitions only:
        # rep1 yes, rep2 no, rep3 yes, rep4 no, etc.
        if rep_count % 2 == 1:
            ax_rep.hlines(
                y=0.30,
                xmin=start,
                xmax=end,
                color="black",
                linewidth=2.2,
            )

        ax_rep.text(
            center,
            0.80,
            f"rep {rep_count}",
            ha="center",
            va="center",
            fontsize=rep_text_fontsize,
        )

        if show_target_label_on_rep_axis:
            ax_rep.text(
                center,
                0.58,
                f"label {target_label}",
                ha="center",
                va="center",
                fontsize=rep_text_fontsize,
            )

        ax_rep.text(
            center,
            0.43,
            f"{duration_ms:.0f} ms",
            ha="center",
            va="center",
            fontsize=max(rep_text_fontsize - 1, 6),
        )

    if rep_count == 0:
        ax_rep.text(
            0.5,
            0.65,
            f"No contiguous runs of target label {target_label!r} found in this segment",
            transform=ax_rep.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    tick_step_bins = max(1, int(round(float(x_tick_step_s) / float(seconds_per_bin))))
    xticks = np.arange(0, S.shape[1] + 1, tick_step_bins)
    xtick_labels = [f"{x * seconds_per_bin:.1f}" for x in xticks]

    ax_rep.set_xticks(xticks)
    ax_rep.set_xticklabels(xtick_labels, fontsize=10)
    ax_rep.set_xlabel("Time (s)", fontsize=11)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"[SAVE] {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Command-line interface
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a continuous spectrogram segment with annotation stripe, alternating "
            "rep underlines for a target label, and time labels in seconds."
        )
    )
    parser.add_argument("--npz-path", required=True, help="Path to .npz file containing spectrogram and labels.")
    parser.add_argument("--output-path", required=True, help="Path where the PNG figure will be saved.")
    parser.add_argument("--target-label", required=True, help="Target syllable/cluster label to underline, e.g. 3.")

    parser.add_argument("--start-frame", type=int, default=None, help="Start time bin/frame, inclusive. Default: 0.")
    parser.add_argument("--end-frame", type=int, default=None, help="End time bin/frame, exclusive. Default: full recording.")

    parser.add_argument("--spectrogram-key", default=None, help="NPZ key for spectrogram. If omitted, tries common keys.")
    parser.add_argument("--label-key", default=None, help="NPZ key for labels. If omitted, tries common keys.")

    parser.add_argument("--seconds-per-bin", type=float, default=0.0027, help="Seconds per spectrogram column. Default: 0.0027.")
    parser.add_argument("--x-tick-step-s", type=float, default=0.5, help="X-axis tick spacing in seconds. Default: 0.5.")

    parser.add_argument("--cmap", default="gray_r", help="Spectrogram colormap. Default: gray_r.")
    parser.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"), help="Percentile contrast clipping. Default: 1 99.5.")
    parser.add_argument("--no-contrast-percentiles", action="store_true", help="Disable percentile contrast clipping.")

    parser.add_argument("--label-color-json", default=None, help="Optional JSON file mapping label strings to colors.")
    parser.add_argument("--show-label-numbers-in-stripe", action="store_true", help="Write label numbers in long enough annotation blocks.")
    parser.add_argument("--hide-target-label-on-rep-axis", action="store_true", help="Only show rep labels/durations, not 'label X'.")
    parser.add_argument("--rep-text-fontsize", type=int, default=8)

    parser.add_argument("--figure-width", type=float, default=18.0)
    parser.add_argument("--figure-height", type=float, default=6.0)
    parser.add_argument("--title", default=None, help="Optional custom title.")
    parser.add_argument("--show-plot", action="store_true", help="Show interactively in addition to saving.")

    args = parser.parse_args()

    npz_path = Path(args.npz_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    arr = np.load(npz_path, allow_pickle=True)

    spectrogram_key = _choose_key(
        arr,
        args.spectrogram_key,
        candidates=["s", "spectrogram", "original_spectogram", "original_spectrogram"],
        kind="spectrogram",
    )
    label_key = _choose_key(
        arr,
        args.label_key,
        candidates=["hdbscan_labels", "labels", "ground_truth_labels", "predicted_labels"],
        kind="label",
    )

    S_full = np.asarray(arr[spectrogram_key])
    labels_full = np.asarray(arr[label_key])
    S_full = _orient_spectrogram_to_FxT(S_full, labels_full)

    S_seg, labels_seg, start, end = _slice_segment(S_full, labels_full, args.start_frame, args.end_frame)

    contrast = None if args.no_contrast_percentiles else tuple(args.contrast_percentiles)
    vmin, vmax = _compute_intensity_scale(S_seg, contrast)

    label_color_lut = _load_label_color_lut_from_file(
        Path(args.label_color_json).expanduser() if args.label_color_json else None
    )

    title = args.title
    if title is None:
        title = (
            f"{npz_path.name} | {label_key} | target label {args.target_label} | "
            f"frames {start}:{end} | x-axis: seconds"
        )

    plot_spectrogram_with_annotations_and_rep_underlines(
        spectrogram=S_seg,
        labels=labels_seg,
        target_label=args.target_label,
        seconds_per_bin=args.seconds_per_bin,
        x_tick_step_s=args.x_tick_step_s,
        cmap=args.cmap,
        label_color_lut=label_color_lut,
        title=title,
        figsize=(args.figure_width, args.figure_height),
        show_label_numbers_in_stripe=args.show_label_numbers_in_stripe,
        show_target_label_on_rep_axis=not args.hide_target_label_on_rep_axis,
        rep_text_fontsize=args.rep_text_fontsize,
        save_path=output_path,
        show_plot=args.show_plot,
        vmin=vmin,
        vmax=vmax,
    )


if __name__ == "__main__":
    main()
