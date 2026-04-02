#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_sample_song_annotations.py

Create spectrogram figures with colored label bars from a TweetyBERT-style NPZ.

Features
--------
- Works as a standalone .py script (CLI) or as an importable module.
- Supports labels from:
    * hdbscan_labels
    * ground_truth_labels
    * predictions (uses argmax across classes)
- Optionally smooths labels with a sliding-window mode filter.
- Optionally loads a fixed label->color mapping from a JSON file.
  If no JSON is provided, colors are generated automatically.
- Handles spectrogram orientation automatically using the label length.
- Uses file_indices/file_map to choose example song segments when available.
- Can display figures and/or save them to disk.
- Can compare two full song segments on the same x-axis timescale.
- In comparison mode, it can make both:
    * a titled comparison figure with dedicated Song A / Song B title rows
    * a second clean comparison figure with only spectrograms and annotations
- In example mode, it can generate separate versions with x-axis units in:
    * frames
    * milliseconds
    * seconds

Main public functions
---------------------
- run_plot_label_blocks(...)
    Plot multiple example segments, similar to the original notebook/script.
- get_segment_records(...)
    Return segment metadata so you can choose specific segments to compare.
- compare_two_segments_same_timescale(...)
    Plot two full song segments with shared x-scale and optional shared contrast.
    Can also make a second clean figure with no Song A / Song B titles.
- compare_two_segments_by_file_name(...)
    Convenience wrapper that finds segments by file name, then compares them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# -----------------------------------------------------------------------------
# Label smoothing
# -----------------------------------------------------------------------------
def smooth_labels_sliding_mode(labels: np.ndarray, w: int) -> np.ndarray:
    """
    Smooth a 1D label sequence with a sliding-window mode filter.

    Ties are broken by earliest first occurrence inside the current window.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if w <= 1 or labels.size == 0:
        return labels.copy()

    n = labels.size
    half_left = (w - 1) // 2
    half_right = (w - 1) - half_left

    counts: defaultdict[int, int] = defaultdict(int)
    pos: defaultdict[int, deque] = defaultdict(deque)

    def add(j: int) -> None:
        lab = int(labels[j])
        counts[lab] += 1
        pos[lab].append(j)

    def remove(j: int) -> None:
        lab = int(labels[j])
        counts[lab] -= 1
        if pos[lab] and pos[lab][0] == j:
            pos[lab].popleft()
        else:
            try:
                pos[lab].remove(j)
            except ValueError:
                pass
        if counts[lab] <= 0:
            del counts[lab]
            del pos[lab]

    left = 0
    right = min(n - 1, half_right)
    for j in range(left, right + 1):
        add(j)

    out = np.empty_like(labels)
    for i in range(n):
        best_lab = None
        best_count = -1
        best_firstpos = 10**18

        for lab, c in counts.items():
            firstpos = pos[lab][0]
            if (c > best_count) or (c == best_count and firstpos < best_firstpos):
                best_lab = lab
                best_count = c
                best_firstpos = firstpos

        out[i] = best_lab

        new_left = max(0, (i + 1) - half_left)
        new_right = min(n - 1, (i + 1) + half_right)
        while left < new_left:
            remove(left)
            left += 1
        while right < new_right:
            right += 1
            add(right)

    return out


# -----------------------------------------------------------------------------
# Color utilities
# -----------------------------------------------------------------------------
def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]


def build_label_color_lut(
    labels: np.ndarray,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    label_universe: Optional[Sequence[int]] = None,
) -> Dict[int, str]:
    """
    Build a label->color dictionary.

    Priority:
    1. If fixed_label_colors_json is provided, load colors from JSON.
    2. Any missing labels are filled from the fallback palette.
    3. If no JSON is provided, generate all colors from the fallback palette.

    Noise label -1 defaults to gray unless overridden by the JSON.
    """
    palette = _get_tab60_palette()
    current_labels = {int(l) for l in np.unique(np.asarray(labels).astype(int))}

    if label_universe is None:
        all_labels = sorted(current_labels)
    else:
        all_labels = sorted({int(l) for l in label_universe} | current_labels)

    lut: Dict[int, str] = {-1: "#7f7f7f"}

    if fixed_label_colors_json is not None:
        json_path = Path(fixed_label_colors_json)
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            lut[int(key)] = str(value)

    non_noise = [lab for lab in all_labels if lab != -1]
    if len(non_noise) > len(palette):
        print(f"[WARN] {len(non_noise)} non-noise labels exceed {len(palette)} colors; colors will repeat.")

    for lab in non_noise:
        if lab not in lut:
            lut[lab] = palette[int(lab) % len(palette)]

    return lut


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def load_labels(data: np.lib.npyio.NpzFile, label_source: str) -> Tuple[np.ndarray, str]:
    """Load the requested label source from the NPZ."""
    if label_source == "predictions":
        if "predictions" not in data.files:
            raise KeyError(f"'predictions' not found. Keys: {data.files}")
        labels_full = np.argmax(data["predictions"], axis=1)
        used_label_key = "predictions (argmax)"
    else:
        if label_source not in data.files:
            raise KeyError(f"'{label_source}' not found. Keys: {data.files}")
        labels_full = data[label_source]
        used_label_key = label_source

    labels_full = np.asarray(labels_full)
    if labels_full.ndim != 1:
        raise ValueError(f"Expected a 1D label array for '{label_source}', got shape {labels_full.shape}")

    return labels_full.astype(int), used_label_key


def load_spectrogram_matching_labels(
    data: np.lib.npyio.NpzFile,
    labels_len: int,
) -> np.ndarray:
    """
    Load the spectrogram and orient it to shape (time, freq) to match labels_len.
    """
    if "s" in data.files:
        s = data["s"]
    elif "original_spectogram" in data.files:
        s = data["original_spectogram"]
    else:
        raise KeyError(f"No spectrogram key found. Keys: {data.files}")

    s = np.asarray(s)
    if s.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, got shape {s.shape}")

    if s.shape[0] == labels_len:
        return s
    if s.shape[1] == labels_len:
        return s.T

    raise ValueError(
        f"Could not align spectrogram shape {s.shape} with labels length {labels_len}."
    )


def load_file_map(data: np.lib.npyio.NpzFile):
    fm = data["file_map"] if "file_map" in data.files else None
    if fm is not None and hasattr(fm, "item"):
        try:
            fm = fm.item()
        except Exception:
            pass
    return fm


def file_name_from_id(fid: int, file_map) -> str:
    if isinstance(file_map, dict):
        value = file_map.get(int(fid), fid)
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return str(value[0])
        return str(value)
    return str(fid)


# -----------------------------------------------------------------------------
# Segment helpers
# -----------------------------------------------------------------------------
def build_segments(
    labels_len: int,
    data: np.lib.npyio.NpzFile,
    max_plot_frames: int,
    min_segment_frames: int,
) -> Tuple[List[Tuple[int, int]], List[int], List[Tuple[int, int, int, int]]]:
    """
    Build song/file segments using file_indices when available.

    Returns
    -------
    segments : list of (start, end)
    seg_ids : list of segment ids / file ids
    kept : filtered examples as (orig_i, s0, e0, fid)
    """
    if "file_indices" in data.files:
        file_ids = np.asarray(data["file_indices"])
        if file_ids.ndim != 1 or file_ids.size != labels_len:
            raise ValueError(
                f"file_indices must be 1D with length {labels_len}, got shape {file_ids.shape}"
            )
        change_pts = np.where(np.diff(file_ids) != 0)[0] + 1
        starts = np.r_[0, change_pts]
        ends = np.r_[change_pts, labels_len]
        segments = list(zip(starts.tolist(), ends.tolist()))
        seg_ids = [int(file_ids[s]) for s, _ in segments]
    else:
        chunk = max_plot_frames
        starts = np.arange(0, labels_len, chunk)
        segments = [(int(s), int(min(labels_len, s + chunk))) for s in starts]
        seg_ids = list(range(len(segments)))

    kept = [
        (i, s, e, seg_ids[i])
        for i, (s, e) in enumerate(segments)
        if (e - s) >= min_segment_frames
    ]

    if len(kept) == 0:
        raise RuntimeError(
            "No segments passed the min_segment_frames filter. Lower min_segment_frames."
        )

    return segments, seg_ids, kept


def choose_spread_examples(
    kept: Sequence[Tuple[int, int, int, int]],
    n_examples: int,
) -> List[Tuple[int, int, int, int]]:
    idxs = np.linspace(0, len(kept) - 1, min(n_examples, len(kept)), dtype=int)
    return [kept[i] for i in idxs]


def get_segment_records(
    npz_path: Union[str, Path],
    *,
    label_source: str = "hdbscan_labels",
    min_segment_frames: int = 800,
    max_plot_frames: int = 4000,
) -> List[Dict[str, Union[int, str]]]:
    """
    Return per-segment metadata so you can pick specific segments to compare.

    Returns a list of dicts with keys:
        filtered_index, all_index, start_frame, end_frame, n_frames,
        file_id, file_name
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    try:
        labels_full, _ = load_labels(data, label_source)
        file_map = load_file_map(data)
        _segments, _seg_ids, kept = build_segments(
            labels_len=labels_full.size,
            data=data,
            max_plot_frames=max_plot_frames,
            min_segment_frames=min_segment_frames,
        )

        records: List[Dict[str, Union[int, str]]] = []
        for filtered_index, (orig_i, s0, e0, fid) in enumerate(kept):
            records.append(
                {
                    "filtered_index": int(filtered_index),
                    "all_index": int(orig_i),
                    "start_frame": int(s0),
                    "end_frame": int(e0),
                    "n_frames": int(e0 - s0),
                    "file_id": int(fid),
                    "file_name": file_name_from_id(fid, file_map),
                }
            )
        return records
    finally:
        try:
            data.close()
        except Exception:
            pass


def _resolve_segment_from_records(
    records: Sequence[Dict[str, Union[int, str]]],
    segment_index: int,
    index_space: str,
) -> Dict[str, Union[int, str]]:
    if index_space == "filtered":
        matches = [r for r in records if int(r["filtered_index"]) == int(segment_index)]
    elif index_space == "all":
        matches = [r for r in records if int(r["all_index"]) == int(segment_index)]
    else:
        raise ValueError("index_space must be 'filtered' or 'all'")

    if len(matches) == 0:
        raise ValueError(f"No segment found for {index_space}_index={segment_index}")
    if len(matches) > 1:
        raise ValueError(f"Multiple segments found for {index_space}_index={segment_index}")
    return matches[0]


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _normalize_x_axis_unit(unit: str) -> str:
    unit = str(unit).strip().lower()
    if unit in {"frames", "frame", "bins", "timebins", "time_bins"}:
        return "frames"
    if unit in {"ms", "millisecond", "milliseconds"}:
        return "ms"
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return "seconds"
    raise ValueError("x_axis_unit must be one of: frames, ms, seconds")


def _parse_x_axis_units(text: Optional[str]) -> Optional[List[str]]:
    if text is None or text == "":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    out: List[str] = []
    for p in parts:
        u = _normalize_x_axis_unit(p)
        if u not in out:
            out.append(u)
    return out


def _get_x_axis_scale_and_label(
    x_axis_unit: str,
    frame_ms: Optional[float],
) -> Tuple[float, str]:
    unit = _normalize_x_axis_unit(x_axis_unit)

    if unit == "frames":
        return 1.0, "Time (frames)"

    if frame_ms is None:
        raise ValueError("frame_ms must be provided when x_axis_unit is 'ms' or 'seconds'")

    if unit == "ms":
        return float(frame_ms), "Time (ms)"

    return float(frame_ms) / 1000.0, "Time (s)"


def _run_boundaries(lab_view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    frames = lab_view.size
    if frames == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    change = np.where(lab_view[1:] != lab_view[:-1])[0] + 1
    run_starts = np.r_[0, change]
    run_ends = np.r_[change, frames]
    return run_starts, run_ends


def _style_spec_and_bar_axes(ax_spec, ax_bar, hide_bar_xlabels: bool = False) -> None:
    for ax in (ax_spec, ax_bar):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_spec.spines["bottom"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_spec.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if hide_bar_xlabels:
        ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def _draw_label_bar(
    ax,
    lab_view: np.ndarray,
    label_color_lut: Dict[int, str],
    label_edge_lw: float,
    *,
    x_scale: float = 1.0,
) -> None:
    run_starts, run_ends = _run_boundaries(lab_view)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    for rs, re in zip(run_starts, run_ends):
        lab = int(lab_view[rs])
        ax.add_patch(
            Rectangle(
                (rs * x_scale, 0),
                (re - rs) * x_scale,
                1,
                facecolor=label_color_lut.get(lab, "#7f7f7f"),
                edgecolor="white",
                linewidth=label_edge_lw,
            )
        )

    ax.set_xlim(0, lab_view.size * x_scale)


# -----------------------------------------------------------------------------
# Single-example plotting workflow
# -----------------------------------------------------------------------------
def plot_segment(
    *,
    seg_num: int,
    s0: int,
    e0: int,
    fid: int,
    spectrogram_tf: np.ndarray,
    labels_full: np.ndarray,
    used_label_key: str,
    file_map,
    label_color_lut: Dict[int, str],
    pad: int,
    max_plot_frames: int,
    smooth_w: int,
    label_bar_height: float,
    label_edge_lw: float,
    show_plots: bool,
    save_plots: bool,
    out_dir: Optional[Path],
    x_axis_unit: str = "frames",
    frame_ms: Optional[float] = None,
) -> Optional[Path]:
    """Plot one segment with a spectrogram and a colored label bar."""
    total_frames = labels_full.size

    view_start = max(0, s0 - pad)
    view_end = min(total_frames, e0 + pad)

    if (view_end - view_start) > max_plot_frames:
        mid = (s0 + e0) // 2
        half = max_plot_frames // 2
        view_start = max(0, mid - half)
        view_end = min(total_frames, view_start + max_plot_frames)

    s_view = spectrogram_tf[view_start:view_end].T
    lab_view = labels_full[view_start:view_end].copy()

    if smooth_w > 1:
        lab_view = smooth_labels_sliding_mode(lab_view, smooth_w)

    vmin, vmax = np.percentile(s_view, [5, 99])

    x_scale, x_label = _get_x_axis_scale_and_label(x_axis_unit, frame_ms)
    x_max = (view_end - view_start) * x_scale

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, label_bar_height], hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    fig.suptitle(f"Label blocks (smoothed w={smooth_w})", y=0.995)

    ax1.imshow(
        s_view,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        extent=(0, x_max, 0, s_view.shape[0]),
        interpolation="nearest",
    )
    ax1.set_xlim(0, x_max)
    ax1.set_ylabel("Freq bins")

    name = file_name_from_id(fid, file_map)
    ax1.set_title(
        f"Example {seg_num + 1} | {used_label_key} | {name} | "
        f"frames {view_start}:{view_end} | x-axis: {x_axis_unit}",
        pad=6,
    )

    _draw_label_bar(
        ax2,
        lab_view,
        label_color_lut,
        label_edge_lw,
        x_scale=x_scale,
    )
    ax2.set_xlabel(x_label)
    _style_spec_and_bar_axes(ax1, ax2, hide_bar_xlabels=False)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.92, hspace=0.0)

    save_path = None
    if save_plots and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"label_blocks_example_{seg_num + 1:02d}_{x_axis_unit}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def run_plot_label_blocks(
    npz_path: Union[str, Path],
    *,
    out_dir: Optional[Union[str, Path]] = None,
    n_examples: int = 10,
    label_source: str = "hdbscan_labels",
    smooth_w: int = 51,
    pad: int = 200,
    max_plot_frames: int = 4000,
    label_bar_height: float = 0.07,
    label_edge_lw: float = 0.6,
    min_segment_frames: int = 800,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    label_universe: Optional[Sequence[int]] = None,
    show_plots: bool = True,
    save_plots: bool = False,
    x_axis_units: Sequence[str] = ("frames", "ms", "seconds"),
    frame_ms: Optional[float] = None,
) -> List[Path]:
    """
    Run the label-block plotting workflow on one NPZ.

    Returns
    -------
    list of saved figure paths (empty if save_plots=False).
    """
    npz_path = Path(npz_path)
    if out_dir is not None:
        out_dir = Path(out_dir)

    normalized_units: List[str] = []
    for unit in x_axis_units:
        u = _normalize_x_axis_unit(unit)
        if u not in normalized_units:
            normalized_units.append(u)

    if any(u in {"ms", "seconds"} for u in normalized_units) and frame_ms is None:
        raise ValueError("frame_ms must be provided when x_axis_units includes 'ms' or 'seconds'")

    data = np.load(npz_path, allow_pickle=True)
    try:
        labels_full, used_label_key = load_labels(data, label_source)
        spectrogram_tf = load_spectrogram_matching_labels(data, labels_full.size)
        file_map = load_file_map(data)

        label_color_lut = build_label_color_lut(
            labels_full,
            fixed_label_colors_json=fixed_label_colors_json,
            label_universe=label_universe,
        )

        segments, _seg_ids, kept = build_segments(
            labels_len=labels_full.size,
            data=data,
            max_plot_frames=max_plot_frames,
            min_segment_frames=min_segment_frames,
        )
        chosen = choose_spread_examples(kept, n_examples)

        print(
            f"Found {len(segments)} total segments; using {len(chosen)} examples "
            f"(filtered by min_segment_frames={min_segment_frames})."
        )
        if fixed_label_colors_json is not None:
            print(f"Using fixed label colors from: {Path(fixed_label_colors_json)}")
        else:
            print("Using internally generated label colors.")
        print(f"Generating x-axis versions: {', '.join(normalized_units)}")

        saved_paths: List[Path] = []
        for k, (_orig_i, s0, e0, fid) in enumerate(chosen):
            for unit in normalized_units:
                saved = plot_segment(
                    seg_num=k,
                    s0=s0,
                    e0=e0,
                    fid=fid,
                    spectrogram_tf=spectrogram_tf,
                    labels_full=labels_full,
                    used_label_key=used_label_key,
                    file_map=file_map,
                    label_color_lut=label_color_lut,
                    pad=pad,
                    max_plot_frames=max_plot_frames,
                    smooth_w=smooth_w,
                    label_bar_height=label_bar_height,
                    label_edge_lw=label_edge_lw,
                    show_plots=show_plots,
                    save_plots=save_plots,
                    out_dir=out_dir,
                    x_axis_unit=unit,
                    frame_ms=frame_ms,
                )
                if saved is not None:
                    saved_paths.append(saved)

        return saved_paths
    finally:
        try:
            data.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Two-song comparison workflow
# -----------------------------------------------------------------------------
def _build_two_song_comparison_figure(
    *,
    s_view_a: np.ndarray,
    s_view_b: np.ndarray,
    lab_view_a: np.ndarray,
    lab_view_b: np.ndarray,
    fname_a: str,
    fname_b: str,
    s0_a: int,
    e0_a: int,
    s0_b: int,
    e0_b: int,
    view_start_a: int,
    view_end_a: int,
    view_start_b: int,
    view_end_b: int,
    used_label_key: str,
    smooth_w: int,
    label_color_lut: Dict[int, str],
    label_edge_lw: float,
    label_bar_height: float,
    shared_len: int,
    vmin_a: float,
    vmax_a: float,
    vmin_b: float,
    vmax_b: float,
    use_seconds: bool,
    frame_ms: Optional[float],
    show_song_titles: bool,
    tick_label_fontsize: int = 20,
    axis_label_fontsize: int = 24,
):
    """Build either the titled or clean two-song comparison figure."""
    if show_song_titles:
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(
            6,
            1,
            height_ratios=[0.14, 1.0, label_bar_height, 0.14, 1.0, label_bar_height],
            hspace=0.04,
        )

        ax_title_a = fig.add_subplot(gs[0, 0])
        ax_spec_a = fig.add_subplot(gs[1, 0])
        ax_bar_a = fig.add_subplot(gs[2, 0], sharex=ax_spec_a)
        ax_title_b = fig.add_subplot(gs[3, 0], sharex=ax_spec_a)
        ax_spec_b = fig.add_subplot(gs[4, 0], sharex=ax_spec_a)
        ax_bar_b = fig.add_subplot(gs[5, 0], sharex=ax_spec_a)

        for ax in (ax_title_a, ax_title_b):
            ax.set_axis_off()

        fig.suptitle(
            f"Two-song comparison | {used_label_key} | smoothed w={smooth_w}",
            y=0.992,
            fontsize=18,
        )

        title_a = (
            f"Song A | {fname_a} | full segment {s0_a}:{e0_a} | "
            f"view {view_start_a}:{view_end_a}"
        )
        title_b = (
            f"Song B | {fname_b} | full segment {s0_b}:{e0_b} | "
            f"view {view_start_b}:{view_end_b}"
        )

        ax_title_a.text(
            0.5,
            0.5,
            title_a,
            ha="center",
            va="center",
            fontsize=13,
            transform=ax_title_a.transAxes,
        )
        ax_title_b.text(
            0.5,
            0.5,
            title_b,
            ha="center",
            va="center",
            fontsize=13,
            transform=ax_title_b.transAxes,
        )

        top = 0.93
    else:
        fig = plt.figure(figsize=(14, 7.5))
        gs = fig.add_gridspec(
            4,
            1,
            height_ratios=[1.0, label_bar_height, 1.0, label_bar_height],
            hspace=0.04,
        )

        ax_spec_a = fig.add_subplot(gs[0, 0])
        ax_bar_a = fig.add_subplot(gs[1, 0], sharex=ax_spec_a)
        ax_spec_b = fig.add_subplot(gs[2, 0], sharex=ax_spec_a)
        ax_bar_b = fig.add_subplot(gs[3, 0], sharex=ax_spec_a)
        top = 0.97

    ax_spec_a.imshow(
        s_view_a,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        vmin=vmin_a,
        vmax=vmax_a,
        extent=(0, s_view_a.shape[1], 0, s_view_a.shape[0]),
        interpolation="nearest",
    )
    ax_spec_b.imshow(
        s_view_b,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        vmin=vmin_b,
        vmax=vmax_b,
        extent=(0, s_view_b.shape[1], 0, s_view_b.shape[0]),
        interpolation="nearest",
    )

    ax_spec_a.set_ylabel("Freq bins", fontsize=axis_label_fontsize)
    ax_spec_b.set_ylabel("Freq bins", fontsize=axis_label_fontsize)
    ax_spec_a.set_xlim(0, shared_len)
    ax_spec_b.set_xlim(0, shared_len)

    _draw_label_bar(ax_bar_a, lab_view_a, label_color_lut, label_edge_lw)
    _draw_label_bar(ax_bar_b, lab_view_b, label_color_lut, label_edge_lw)

    _style_spec_and_bar_axes(ax_spec_a, ax_bar_a, hide_bar_xlabels=True)
    _style_spec_and_bar_axes(ax_spec_b, ax_bar_b, hide_bar_xlabels=False)

    if use_seconds:
        if frame_ms is None:
            raise ValueError("frame_ms must be provided when use_seconds=True")
        ax_bar_b.set_xlabel("Time (s)", fontsize=axis_label_fontsize)
        ticks = ax_bar_b.get_xticks()
        labels = [f"{t * frame_ms / 1000.0:.2f}" for t in ticks]
        ax_bar_b.set_xticks(ticks)
        ax_bar_b.set_xticklabels(labels, fontsize=tick_label_fontsize)
    else:
        ax_bar_b.set_xlabel("Time (frames)", fontsize=axis_label_fontsize)

    ax_spec_a.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax_spec_b.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax_bar_b.tick_params(axis="both", labelsize=tick_label_fontsize)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=top)
    return fig


def compare_two_segments_same_timescale(
    npz_path: Union[str, Path],
    *,
    segment_index_a: int,
    segment_index_b: int,
    index_space: str = "filtered",
    out_dir: Optional[Union[str, Path]] = None,
    label_source: str = "hdbscan_labels",
    smooth_w: int = 51,
    pad: int = 0,
    min_segment_frames: int = 800,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    label_universe: Optional[Sequence[int]] = None,
    shared_contrast: bool = True,
    label_bar_height: float = 0.09,
    label_edge_lw: float = 0.6,
    use_seconds: bool = False,
    frame_ms: Optional[float] = None,
    tick_label_fontsize: int = 20,
    axis_label_fontsize: int = 24,
    show_plot: bool = True,
    save_plot: bool = False,
    make_clean_figure: bool = True,
    show_clean_figure: Optional[bool] = None,
    save_clean_figure: Optional[bool] = None,
) -> Optional[Path]:
    """
    Compare two full song segments on the same x-axis timescale.

    This function does not crop to max_plot_frames. It always shows the full
    selected segment, plus optional padding.

    It makes a titled comparison figure, and optionally a second clean figure
    with only the spectrograms and annotation bars (no Song A / Song B titles).

    Returns
    -------
    Optional[Path]
        Path to the titled comparison figure if save_plot=True, otherwise None.
        When save_clean_figure is enabled, the clean figure is also saved with
        a ``_clean`` suffix and its path is printed.
    """
    npz_path = Path(npz_path)
    if out_dir is not None:
        out_dir = Path(out_dir)

    if show_clean_figure is None:
        show_clean_figure = show_plot
    if save_clean_figure is None:
        save_clean_figure = save_plot

    data = np.load(npz_path, allow_pickle=True)
    try:
        labels_full, used_label_key = load_labels(data, label_source)
        spectrogram_tf = load_spectrogram_matching_labels(data, labels_full.size)
        _file_map = load_file_map(data)

        records = get_segment_records(
            npz_path=npz_path,
            label_source=label_source,
            min_segment_frames=min_segment_frames,
        )
        rec_a = _resolve_segment_from_records(records, segment_index_a, index_space)
        rec_b = _resolve_segment_from_records(records, segment_index_b, index_space)

        s0_a = int(rec_a["start_frame"])
        e0_a = int(rec_a["end_frame"])
        fname_a = str(rec_a["file_name"])

        s0_b = int(rec_b["start_frame"])
        e0_b = int(rec_b["end_frame"])
        fname_b = str(rec_b["file_name"])

        total_frames = labels_full.size

        view_start_a = max(0, s0_a - pad)
        view_end_a = min(total_frames, e0_a + pad)
        view_start_b = max(0, s0_b - pad)
        view_end_b = min(total_frames, e0_b + pad)

        s_view_a = spectrogram_tf[view_start_a:view_end_a].T
        s_view_b = spectrogram_tf[view_start_b:view_end_b].T

        lab_view_a = labels_full[view_start_a:view_end_a].copy()
        lab_view_b = labels_full[view_start_b:view_end_b].copy()

        if smooth_w > 1:
            lab_view_a = smooth_labels_sliding_mode(lab_view_a, smooth_w)
            lab_view_b = smooth_labels_sliding_mode(lab_view_b, smooth_w)

        label_color_lut = build_label_color_lut(
            np.concatenate([lab_view_a, lab_view_b]),
            fixed_label_colors_json=fixed_label_colors_json,
            label_universe=label_universe,
        )

        if shared_contrast:
            combined = np.concatenate([s_view_a.ravel(), s_view_b.ravel()])
            vmin, vmax = np.percentile(combined, [5, 99])
            vmin_a = vmin_b = vmin
            vmax_a = vmax_b = vmax
        else:
            vmin_a, vmax_a = np.percentile(s_view_a, [5, 99])
            vmin_b, vmax_b = np.percentile(s_view_b, [5, 99])

        len_a = view_end_a - view_start_a
        len_b = view_end_b - view_start_b
        shared_len = max(len_a, len_b)

        titled_fig = _build_two_song_comparison_figure(
            s_view_a=s_view_a,
            s_view_b=s_view_b,
            lab_view_a=lab_view_a,
            lab_view_b=lab_view_b,
            fname_a=fname_a,
            fname_b=fname_b,
            s0_a=s0_a,
            e0_a=e0_a,
            s0_b=s0_b,
            e0_b=e0_b,
            view_start_a=view_start_a,
            view_end_a=view_end_a,
            view_start_b=view_start_b,
            view_end_b=view_end_b,
            used_label_key=used_label_key,
            smooth_w=smooth_w,
            label_color_lut=label_color_lut,
            label_edge_lw=label_edge_lw,
            label_bar_height=label_bar_height,
            shared_len=shared_len,
            vmin_a=vmin_a,
            vmax_a=vmax_a,
            vmin_b=vmin_b,
            vmax_b=vmax_b,
            use_seconds=use_seconds,
            frame_ms=frame_ms,
            show_song_titles=True,
            tick_label_fontsize=tick_label_fontsize,
            axis_label_fontsize=axis_label_fontsize,
        )

        save_path = None
        clean_save_path = None
        stem_a = Path(fname_a).stem
        stem_b = Path(fname_b).stem

        if save_plot and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"compare_{stem_a}__vs__{stem_b}.png"
            titled_fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SAVE] {save_path}")

        if show_plot:
            titled_fig.show()
        else:
            plt.close(titled_fig)

        if make_clean_figure:
            clean_fig = _build_two_song_comparison_figure(
                s_view_a=s_view_a,
                s_view_b=s_view_b,
                lab_view_a=lab_view_a,
                lab_view_b=lab_view_b,
                fname_a=fname_a,
                fname_b=fname_b,
                s0_a=s0_a,
                e0_a=e0_a,
                s0_b=s0_b,
                e0_b=e0_b,
                view_start_a=view_start_a,
                view_end_a=view_end_a,
                view_start_b=view_start_b,
                view_end_b=view_end_b,
                used_label_key=used_label_key,
                smooth_w=smooth_w,
                label_color_lut=label_color_lut,
                label_edge_lw=label_edge_lw,
                label_bar_height=label_bar_height,
                shared_len=shared_len,
                vmin_a=vmin_a,
                vmax_a=vmax_a,
                vmin_b=vmin_b,
                vmax_b=vmax_b,
                use_seconds=use_seconds,
                frame_ms=frame_ms,
                show_song_titles=False,
                tick_label_fontsize=tick_label_fontsize,
                axis_label_fontsize=axis_label_fontsize,
            )

            if save_clean_figure and out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                clean_save_path = out_dir / f"compare_{stem_a}__vs__{stem_b}_clean.png"
                clean_fig.savefig(clean_save_path, dpi=300, bbox_inches="tight")
                print(f"[SAVE] {clean_save_path}")

            if show_clean_figure:
                clean_fig.show()
            else:
                plt.close(clean_fig)

        return save_path
    finally:
        try:
            data.close()
        except Exception:
            pass


def compare_two_segments_by_file_name(
    npz_path: Union[str, Path],
    *,
    file_name_a: str,
    file_name_b: str,
    label_source: str = "hdbscan_labels",
    min_segment_frames: int = 800,
    **kwargs,
) -> Optional[Path]:
    """
    Convenience wrapper: compare two segments by file name instead of index.
    """
    records = get_segment_records(
        npz_path=npz_path,
        label_source=label_source,
        min_segment_frames=min_segment_frames,
    )

    match_a = [r for r in records if str(r["file_name"]) == file_name_a]
    match_b = [r for r in records if str(r["file_name"]) == file_name_b]

    if len(match_a) != 1:
        raise ValueError(f"Expected exactly 1 match for file_name_a, got {len(match_a)}")
    if len(match_b) != 1:
        raise ValueError(f"Expected exactly 1 match for file_name_b, got {len(match_b)}")

    return compare_two_segments_same_timescale(
        npz_path=npz_path,
        segment_index_a=int(match_a[0]["filtered_index"]),
        segment_index_b=int(match_b[0]["filtered_index"]),
        index_space="filtered",
        label_source=label_source,
        min_segment_frames=min_segment_frames,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------
def _parse_label_universe(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text == "":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot spectrogram label blocks from a TweetyBERT-style NPZ.")
    p.add_argument("--npz-path", required=True, help="Path to the input .npz file")
    p.add_argument("--mode", default="examples", choices=["examples", "compare"], help="Run example plots or two-song comparison")
    p.add_argument("--out-dir", default=None, help="Directory to save figures")

    # Shared
    p.add_argument(
        "--label-source",
        default="hdbscan_labels",
        choices=["hdbscan_labels", "ground_truth_labels", "predictions"],
        help="Source of labels for the colored label bar",
    )
    p.add_argument("--smooth-w", type=int, default=51, help="Sliding-mode smoothing window; 1 disables smoothing")
    p.add_argument("--pad", type=int, default=200, help="Context around each segment in frames")
    p.add_argument("--min-segment-frames", type=int, default=800, help="Skip segments shorter than this many frames")
    p.add_argument(
        "--fixed-label-colors-json",
        default=None,
        help="Optional JSON file mapping labels to hex colors",
    )
    p.add_argument(
        "--label-universe",
        default=None,
        help="Optional comma-separated label universe, e.g. '0,1,2,3,4'",
    )
    p.add_argument(
        "--x-axis-units",
        default="frames,ms,seconds",
        help="Comma-separated x-axis units to generate for example plots: frames, ms, seconds",
    )
    p.add_argument(
        "--frame-ms",
        type=float,
        default=None,
        help="Milliseconds per frame/time bin when using ms or seconds axes",
    )
    p.add_argument("--show-plots", dest="show_plots", action="store_true", help="Display figures interactively")
    p.add_argument("--no-show-plots", dest="show_plots", action="store_false", help="Do not display figures")
    p.set_defaults(show_plots=True)

    # Example mode
    p.add_argument("--n-examples", type=int, default=10, help="Number of example figures to make")
    p.add_argument("--max-plot-frames", type=int, default=4000, help="Maximum number of frames shown per example figure")
    p.add_argument("--label-bar-height", type=float, default=0.07, help="Relative height of the colored label strip")
    p.add_argument("--label-edge-lw", type=float, default=0.6, help="Line width of label block edges")
    p.add_argument("--save-plots", action="store_true", help="Save figures to --out-dir")

    # Compare mode
    p.add_argument("--compare-segment-a", type=int, default=None, help="First segment index for compare mode")
    p.add_argument("--compare-segment-b", type=int, default=None, help="Second segment index for compare mode")
    p.add_argument(
        "--segment-index-space",
        default="filtered",
        choices=["filtered", "all"],
        help="Whether compare-segment indices refer to filtered or all segments",
    )
    p.add_argument("--shared-contrast", dest="shared_contrast", action="store_true", help="Use one contrast scale for both songs")
    p.add_argument("--separate-contrast", dest="shared_contrast", action="store_false", help="Scale each song separately")
    p.set_defaults(shared_contrast=True)
    p.add_argument("--use-seconds", action="store_true", help="Label x-axis in seconds instead of frames")
    p.add_argument("--tick-label-fontsize", type=int, default=20, help="Tick label fontsize for compare figures")
    p.add_argument("--axis-label-fontsize", type=int, default=24, help="Axis label fontsize for compare figures")
    p.add_argument("--save-plot", action="store_true", help="Save titled comparison figure to --out-dir")
    p.add_argument("--no-clean-figure", dest="make_clean_figure", action="store_false", help="Do not make the second clean comparison figure")
    p.add_argument("--make-clean-figure", dest="make_clean_figure", action="store_true", help="Make the second clean comparison figure")
    p.set_defaults(make_clean_figure=True)
    p.add_argument("--show-clean-figure", dest="show_clean_figure", action="store_true", help="Display the clean comparison figure")
    p.add_argument("--no-show-clean-figure", dest="show_clean_figure", action="store_false", help="Do not display the clean comparison figure")
    p.set_defaults(show_clean_figure=None)
    p.add_argument("--save-clean-figure", dest="save_clean_figure", action="store_true", help="Save the clean comparison figure to --out-dir")
    p.add_argument("--no-save-clean-figure", dest="save_clean_figure", action="store_false", help="Do not save the clean comparison figure")
    p.set_defaults(save_clean_figure=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    label_universe = _parse_label_universe(args.label_universe)
    x_axis_units = _parse_x_axis_units(args.x_axis_units)

    if args.mode == "examples":
        run_plot_label_blocks(
            npz_path=args.npz_path,
            out_dir=args.out_dir,
            n_examples=args.n_examples,
            label_source=args.label_source,
            smooth_w=args.smooth_w,
            pad=args.pad,
            max_plot_frames=args.max_plot_frames,
            label_bar_height=args.label_bar_height,
            label_edge_lw=args.label_edge_lw,
            min_segment_frames=args.min_segment_frames,
            fixed_label_colors_json=args.fixed_label_colors_json,
            label_universe=label_universe,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
            x_axis_units=x_axis_units if x_axis_units is not None else ("frames", "ms", "seconds"),
            frame_ms=args.frame_ms,
        )
        return

    if args.compare_segment_a is None or args.compare_segment_b is None:
        raise ValueError("In compare mode, --compare-segment-a and --compare-segment-b are required.")

    compare_two_segments_same_timescale(
        npz_path=args.npz_path,
        segment_index_a=args.compare_segment_a,
        segment_index_b=args.compare_segment_b,
        index_space=args.segment_index_space,
        out_dir=args.out_dir,
        label_source=args.label_source,
        smooth_w=args.smooth_w,
        pad=args.pad,
        min_segment_frames=args.min_segment_frames,
        fixed_label_colors_json=args.fixed_label_colors_json,
        label_universe=label_universe,
        shared_contrast=args.shared_contrast,
        label_bar_height=max(args.label_bar_height, 0.08),
        label_edge_lw=args.label_edge_lw,
        use_seconds=args.use_seconds,
        frame_ms=args.frame_ms,
        tick_label_fontsize=args.tick_label_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        show_plot=args.show_plots,
        save_plot=args.save_plot,
        make_clean_figure=args.make_clean_figure,
        show_clean_figure=args.show_clean_figure,
        save_clean_figure=args.save_clean_figure,
    )


if __name__ == "__main__":
    main()
    
    
    """
    import importlib
import npz_sample_song_annotations as nsa
importlib.reload(nsa)

nsa.compare_two_segments_same_timescale(
    npz_path="/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz",
    segment_index_a=31,
    segment_index_b=158,
    index_space="filtered",
    out_dir="/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/song_comparisons",
    label_source="hdbscan_labels",
    smooth_w=51,
    pad=0,
    min_segment_frames=800,
    shared_contrast=True,
    use_seconds=True,
    frame_ms=2.70,
    tick_label_fontsize=20,
    axis_label_fontsize=24,
    show_plot=True,
    save_plot=True,
    make_clean_figure=True,
    show_clean_figure=True,
    save_clean_figure=True,
)
    
    """