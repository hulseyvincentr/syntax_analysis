#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pre_post_syllable_sample_spectrograms.py

Compare stitched sample spectrograms for each HDBSCAN label before vs after a
lesion/treatment date stored in an Excel metadata sheet.

This long-row version plots pre-lesion samples as one wide row and post-lesion
samples as one wide row underneath, which is easier to inspect when using long
stretches such as 20,000 stitched bins. It also adds optional bout-underlines
under each row and x-axis labels in stitched seconds.

Expected NPZ keys
-----------------
Required:
    hdbscan_labels : shape (T,)
    s              : spectrogram, shape (F, T) or (T, F)

Usually required for pre/post splitting:
    file_indices   : shape (T,), maps each time bin to an entry in file_map
    file_map       : dict-like mapping file index -> filename/path/metadata tuple

Optional:
    embedding_outputs : shape (T, 2+) for an optional UMAP pre/post plot

Public function
---------------
plot_pre_post_spectrogram_samples_for_labels(...)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# =========================
# Colors for labels / UMAP
# =========================
def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]


def build_label_color_lut(all_labels: np.ndarray) -> Dict[int, str]:
    uniq = sorted(np.unique(all_labels.astype(int)))
    non_noise = [l for l in uniq if l != -1]
    palette = _get_tab60_palette()

    if len(non_noise) > len(palette):
        print(f"[WARN] {len(non_noise)} non-noise labels exceed 60 colors; colors will repeat.")

    lut: Dict[int, str] = {-1: "#7f7f7f"}
    for i, lab in enumerate(non_noise):
        lut[lab] = palette[i % len(palette)]
    return lut


# =========================
# Animal ID / date helpers
# =========================
def _infer_animal_id_from_path(p: Path) -> str:
    candidates: List[str] = []

    stem = p.stem.strip()
    if stem:
        candidates.append(stem.split("_")[0].strip())
        candidates.append(stem)

    parent_name = p.parent.name.strip()
    if parent_name:
        candidates.append(parent_name)
        candidates.append(parent_name.split("_")[0].strip())

    seen = set()
    for c in candidates:
        if c and c not in seen:
            return c
        seen.add(c)
    return p.stem


def _excel_serial_to_date(x: Union[int, float]) -> date:
    return (datetime(1899, 12, 30) + timedelta(days=float(x))).date()


def _parse_date_from_value(value: Any) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        if 20000 <= float(value) <= 60000:
            return _excel_serial_to_date(float(value))
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None

    if re.fullmatch(r"\d{5}(?:\.\d+)?", text):
        serial = float(text)
        if 20000 <= serial <= 60000:
            return _excel_serial_to_date(serial)

    serial_match = re.search(r"(?<!\d)([2345]\d{4})(?:\.\d+)?(?!\d)", text)
    if serial_match:
        serial = float(serial_match.group(0))
        if 20000 <= serial <= 60000:
            return _excel_serial_to_date(serial)

    patterns = [
        r"(?<!\d)(20\d{2})[-_./]?([01]\d)[-_./]?([0-3]\d)(?!\d)",
        r"(?<!\d)([01]\d)[-_./]?([0-3]\d)[-_./]?(20\d{2})(?!\d)",
        r"(?<!\d)([01]\d)([0-3]\d)(\d{2})(?!\d)",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        try:
            if pat.startswith(r"(?<!\d)(20"):
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif m.group(3).startswith("20"):
                mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                mo, d, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                y = 2000 + yy
            return date(y, mo, d)
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.date()

    return None


def _normalize_animal_id(x: Any) -> str:
    return str(x).strip()


def get_treatment_date_from_metadata(
    metadata_excel_path: Union[str, Path],
    animal_id: str,
    *,
    metadata_sheet: str = "metadata",
    animal_id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
) -> date:
    metadata_excel_path = Path(metadata_excel_path)
    df = pd.read_excel(metadata_excel_path, sheet_name=metadata_sheet)

    required = {animal_id_col, treatment_date_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Metadata sheet '{metadata_sheet}' is missing required column(s): {sorted(missing)}. "
            f"Columns found: {list(df.columns)}"
        )

    target = _normalize_animal_id(animal_id)
    animal_ids = df[animal_id_col].map(_normalize_animal_id)
    matching = df.loc[animal_ids == target, treatment_date_col].dropna()

    if matching.empty:
        available = sorted(animal_ids.dropna().unique())
        raise ValueError(
            f"No treatment date found for animal_id='{animal_id}' in sheet '{metadata_sheet}'. "
            f"First available IDs: {available[:20]}"
        )

    parsed_dates = []
    for val in matching:
        d = _parse_date_from_value(val)
        if d is not None:
            parsed_dates.append(d)

    if not parsed_dates:
        raise ValueError(f"Could not parse a treatment date for animal_id='{animal_id}'. Raw values: {list(matching)}")

    unique_dates = sorted(set(parsed_dates))
    if len(unique_dates) > 1:
        print(f"[WARN] Multiple treatment dates found for {animal_id}: {unique_dates}; using {unique_dates[0]}")

    return unique_dates[0]


# =========================
# NPZ / file-date helpers
# =========================
def _as_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _file_map_to_dict(file_map_obj: Any) -> Dict[Any, Any]:
    file_map_obj = _as_object(file_map_obj)
    if isinstance(file_map_obj, dict):
        return file_map_obj
    if isinstance(file_map_obj, (list, tuple, np.ndarray)):
        return {i: v for i, v in enumerate(file_map_obj)}
    raise TypeError(f"Unsupported file_map type: {type(file_map_obj)}")


def _entry_to_text(entry: Any) -> str:
    entry = _as_object(entry)
    if isinstance(entry, (str, Path)):
        return str(entry)
    if isinstance(entry, dict):
        return " ".join(_entry_to_text(v) for v in entry.values())
    if isinstance(entry, (list, tuple, np.ndarray)):
        return " ".join(_entry_to_text(v) for v in entry)
    return str(entry)


def _lookup_file_map_entry(file_map: Dict[Any, Any], file_index: Any) -> Any:
    candidates = [file_index]

    try:
        candidates.append(int(file_index))
    except Exception:
        pass

    candidates.append(str(file_index))

    for key in candidates:
        if key in file_map:
            return file_map[key]

    raise KeyError(f"file_index {file_index!r} not found in file_map. Example keys: {list(file_map.keys())[:10]}")


def get_timebin_recording_dates(
    arr: np.lib.npyio.NpzFile,
    *,
    expected_length: int,
    date_array_key: Optional[str] = None,
) -> np.ndarray:
    candidate_keys = []
    if date_array_key is not None:
        candidate_keys.append(date_array_key)
    candidate_keys.extend(["recording_dates", "recording_date", "date", "dates", "date_time", "datetime"])

    for key in candidate_keys:
        if key not in arr.files:
            continue
        raw = np.asarray(arr[key])
        if raw.shape[0] != expected_length:
            print(f"[WARN] Found arr['{key}'], but length {raw.shape[0]} does not match T={expected_length}; ignoring.")
            continue
        unique_raw = {}
        out = np.empty(expected_length, dtype=object)
        for i, val in enumerate(raw):
            text_key = str(val)
            if text_key not in unique_raw:
                unique_raw[text_key] = _parse_date_from_value(val)
            out[i] = unique_raw[text_key]
        if np.any(pd.notna(out)):
            print(f"[INFO] Using arr['{key}'] to split pre/post lesion.")
            return out

    if "file_indices" not in arr.files or "file_map" not in arr.files:
        raise ValueError(
            "Could not find time-bin dates. Expected either a per-timebin date array "
            "or both arr['file_indices'] and arr['file_map']."
        )

    file_indices = np.asarray(arr["file_indices"])
    if file_indices.shape[0] != expected_length:
        raise ValueError(f"arr['file_indices'] length {file_indices.shape[0]} does not match T={expected_length}.")

    file_map = _file_map_to_dict(arr["file_map"])
    unique_file_indices = np.unique(file_indices)
    file_index_to_date: Dict[Any, Optional[date]] = {}

    for file_idx in unique_file_indices:
        entry = _lookup_file_map_entry(file_map, file_idx)
        text = _entry_to_text(entry)
        file_index_to_date[file_idx] = _parse_date_from_value(text)

    missing = [idx for idx, d in file_index_to_date.items() if d is None]
    if missing:
        print(f"[WARN] Could not parse recording dates for {len(missing)} file_map entries.")
        for idx in missing[:5]:
            print(f"       file_index={idx!r}, entry={_entry_to_text(_lookup_file_map_entry(file_map, idx))[:160]}")

    parsed_dates = [d for d in file_index_to_date.values() if d is not None]
    if not parsed_dates:
        raise ValueError("Could not parse any recording dates from file_map entries.")

    out = np.empty(expected_length, dtype=object)
    for file_idx in unique_file_indices:
        out[file_indices == file_idx] = file_index_to_date[file_idx]

    print(
        f"[INFO] Parsed recording dates from file_map: "
        f"{min(parsed_dates)} to {max(parsed_dates)} across {len(set(parsed_dates))} unique dates."
    )
    return out


def make_pre_post_masks(
    timebin_dates: np.ndarray,
    treatment_date: date,
    *,
    treatment_day_assignment: str = "exclude",
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.array([d is not None for d in timebin_dates], dtype=bool)
    date_values = np.array([d if d is not None else date.min for d in timebin_dates], dtype=object)

    pre_mask = valid & (date_values < treatment_date)
    post_mask = valid & (date_values > treatment_date)

    if treatment_day_assignment == "pre":
        pre_mask = pre_mask | (valid & (date_values == treatment_date))
    elif treatment_day_assignment == "post":
        post_mask = post_mask | (valid & (date_values == treatment_date))
    elif treatment_day_assignment == "exclude":
        pass
    else:
        raise ValueError("treatment_day_assignment must be one of: 'exclude', 'pre', 'post'.")

    return pre_mask, post_mask


# =========================
# Plotting helpers
# =========================
def _orient_spectrogram_to_FxT(S: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if S.ndim != 2:
        raise ValueError("Expected a 2D spectrogram in arr['s'].")
    if S.shape[1] != labels.shape[0] and S.shape[0] == labels.shape[0]:
        S = S.T
    if S.shape[1] != labels.shape[0]:
        raise ValueError(
            f"After orientation, spectrogram time dimension {S.shape[1]} must match labels length {labels.shape[0]}."
        )
    return S


def _select_stitched_indices(
    idx: np.ndarray,
    *,
    sample_i: int,
    spectrogram_length: int,
    sample_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_mode == "first":
        start = sample_i * spectrogram_length
        end = start + spectrogram_length
        return idx[start:end]

    if sample_mode == "random":
        selected = rng.choice(idx, size=spectrogram_length, replace=False)
        return np.sort(selected)

    raise ValueError("sample_mode must be 'first' or 'random'.")


def _hide_spectrogram_ticks(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )



def _find_stitched_bout_spans(selected_indices: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find original contiguous bouts within a stitched sequence of selected label bins.

    Parameters
    ----------
    selected_indices : np.ndarray
        Original time-bin indices selected for a single label/epoch. These should be
        sorted in original recording order. In the plotted spectrogram, these bins are
        re-indexed to stitched x positions 0..N-1.

    Returns
    -------
    spans : list of tuples
        Each tuple is (stitched_start, stitched_end, original_start, original_end),
        where stitched_end is exclusive and original_end is inclusive.
    """
    selected_indices = np.asarray(selected_indices)
    if selected_indices.size == 0:
        return []

    # New bout whenever adjacent selected bins were not adjacent in the original array.
    # This separates repeated syllable bouts without drawing vertical lines over data.
    breaks = np.flatnonzero(np.diff(selected_indices) > 1) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, selected_indices.size]

    spans: List[Tuple[int, int, int, int]] = []
    for s, e in zip(starts, ends):
        spans.append((int(s), int(e), int(selected_indices[s]), int(selected_indices[e - 1])))
    return spans


def _format_time_label(seconds: float) -> str:
    """Readable tick labels for seconds."""
    if seconds >= 10:
        return f"{seconds:.0f}"
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{seconds:.0f}"
    return f"{seconds:.1f}"


def _set_stitched_time_axis(
    ax: plt.Axes,
    *,
    n_bins: int,
    seconds_per_bin: float,
    x_tick_step_s: float,
    xlabel: str = "Stitched label time (s)",
) -> None:
    """Set x-ticks in seconds for a stitched spectrogram axis."""
    if seconds_per_bin <= 0:
        raise ValueError("seconds_per_bin must be > 0.")
    if x_tick_step_s <= 0:
        raise ValueError("x_tick_step_s must be > 0.")

    step_bins = max(1, int(round(x_tick_step_s / seconds_per_bin)))
    xticks = np.arange(0, n_bins + 1, step_bins)
    ax.set_xticks(xticks)
    ax.set_xticklabels([_format_time_label(x * seconds_per_bin) for x in xticks], fontsize=10)
    ax.tick_params(axis="x", pad=0)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=1)


def _draw_bout_underline_axis(
    ax: plt.Axes,
    selected_indices: np.ndarray,
    *,
    seconds_per_bin: float,
    show_bout_underlines: bool = True,
    show_bout_labels: bool = False,
    min_bout_label_bins: int = 80,
    underline_odd_bouts_only: bool = True,
    epoch_label: str = "bouts",
) -> None:
    """
    Draw horizontal underlines for alternating bouts in a stitched label sequence.

    This avoids vertical lines over the spectrogram. Bout boundaries are inferred from
    gaps in the original time-bin indices: if two neighboring stitched bins were not
    adjacent in the original recording, they belong to different bouts.
    """
    spans = _find_stitched_bout_spans(selected_indices)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(epoch_label, rotation=0, labelpad=24, va="center", fontsize=10)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    # Keep the bottom spine only on the final axis where time labels are shown.
    # The caller can hide it for upper underline axes.
    ax.grid(False)

    if not show_bout_underlines:
        return

    for bout_i, (start, end, original_start, original_end) in enumerate(spans, start=1):
        draw_this_bout = True
        if underline_odd_bouts_only:
            draw_this_bout = (bout_i % 2 == 1)

        if draw_this_bout:
            line_y = 0.88 if not show_bout_labels else 0.72
            ax.hlines(
                y=line_y,
                xmin=start,
                xmax=end,
                color="black",
                linewidth=2.0,
            )

        if show_bout_labels and (end - start) >= min_bout_label_bins:
            center = (start + end) / 2
            duration_ms = (end - start) * seconds_per_bin * 1000
            ax.text(
                center,
                0.20,
                f"bout {bout_i}\n{duration_ms:.0f} ms",
                ha="center",
                va="center",
                fontsize=7,
            )


def plot_umap_pre_post_by_label(
    embedding: np.ndarray,
    labels: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    lut: Dict[int, str],
    outdir: Optional[Path],
    *,
    animal_id: str,
    treatment_date: date,
    show_plot: bool = True,
    max_points_per_epoch: int = 100_000,
    random_seed: int = 0,
) -> None:
    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0] or embedding.shape[1] < 2:
        print("[WARN] UMAP pre/post plot skipped: embedding must be shape (T, 2+) matching labels length.")
        return

    rng = np.random.default_rng(random_seed)

    def subsample(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if idx.size > max_points_per_epoch:
            idx = rng.choice(idx, size=max_points_per_epoch, replace=False)
            idx = np.sort(idx)
        return idx

    pre_idx = subsample(pre_mask)
    post_idx = subsample(post_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, idx, epoch in zip(axes, [pre_idx, post_idx], ["Pre-lesion", "Post-lesion"]):
        colors = [lut[int(lab)] for lab in labels[idx]]
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=colors,
            s=8,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_title(f"{epoch}\nN={idx.size:,} time bins")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(False)

    fig.suptitle(f"{animal_id} — HDBSCAN labels by epoch; lesion date = {treatment_date}", y=1.02)
    fig.tight_layout()

    if outdir is not None:
        save_path = outdir / "umap_hdbscan_labels_pre_vs_post.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_label_color_key_png(
    labels: np.ndarray,
    lut: Dict[int, str],
    outdir: Path,
    *,
    animal_id: Optional[str] = None,
) -> None:
    uniq = sorted(np.unique(labels.astype(int)))
    handles = [Patch(facecolor=lut[l], edgecolor="none", label=str(l)) for l in uniq]

    max_rows = 28
    ncols = max(1, int(np.ceil(len(handles) / max_rows)))
    nrows = int(np.ceil(len(handles) / ncols))

    fig_w = 2.8 + 1.6 * ncols
    fig_h = max(2.0, 0.33 * nrows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    title = f"{animal_id} — label color key" if animal_id is not None else "Label color key"

    ax.legend(
        handles=handles,
        title=title,
        ncol=ncols,
        loc="center",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.5,
    )

    save_path = outdir / "umap_hdbscan_labels_key.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[SAVE] {save_path}")


# =========================
# Main public function
# =========================
def plot_pre_post_spectrogram_samples_for_labels(
    npz_path: Union[str, Path],
    metadata_excel_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    animal_id: Optional[str] = None,
    metadata_sheet: str = "metadata",
    animal_id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    selected_labels: Optional[Sequence[int]] = None,
    skip_noise_label: bool = True,
    spectrogram_length: int = 20_000,
    num_sample_spectrograms: Optional[int] = None,
    treatment_day_assignment: str = "exclude",
    sample_mode: str = "first",
    random_seed: int = 0,
    cmap: str = "gray_r",
    show_colorbar: bool = False,
    show_plots: bool = True,
    save_figures: bool = True,
    shared_intensity_scale: bool = True,
    contrast_percentiles: Optional[Tuple[float, float]] = None,
    make_umap_pre_post_plot: bool = True,
    save_umap_label_key: bool = True,
    date_array_key: Optional[str] = None,
    figure_width: float = 22.0,
    row_height: float = 2.8,
    underline_row_height: float = 0.18,
    subplot_hspace: float = 0.005,
    seconds_per_bin: float = 0.0027,
    x_tick_step_s: float = 5.0,
    show_bout_underlines: bool = True,
    show_bout_labels: bool = False,
    min_bout_label_bins: int = 80,
) -> pd.DataFrame:
    npz_path = Path(npz_path)
    metadata_excel_path = Path(metadata_excel_path)
    animal_id = animal_id or _infer_animal_id_from_path(npz_path)

    outdir: Optional[Path] = None
    bout_lines_dir: Optional[Path] = None
    no_bout_lines_dir: Optional[Path] = None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        bout_lines_dir = outdir / "bout_lines"
        no_bout_lines_dir = outdir / "no_bout_lines"
        bout_lines_dir.mkdir(parents=True, exist_ok=True)
        no_bout_lines_dir.mkdir(parents=True, exist_ok=True)

    treatment_date = get_treatment_date_from_metadata(
        metadata_excel_path=metadata_excel_path,
        animal_id=animal_id,
        metadata_sheet=metadata_sheet,
        animal_id_col=animal_id_col,
        treatment_date_col=treatment_date_col,
    )

    arr = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(arr["hdbscan_labels"]).astype(int)
    S = _orient_spectrogram_to_FxT(np.asarray(arr["s"]), labels)

    timebin_dates = get_timebin_recording_dates(arr, expected_length=labels.shape[0], date_array_key=date_array_key)
    pre_mask, post_mask = make_pre_post_masks(
        timebin_dates,
        treatment_date,
        treatment_day_assignment=treatment_day_assignment,
    )

    print(f"[INFO] Animal ID: {animal_id}")
    print(f"[INFO] Lesion/treatment date: {treatment_date}")
    print(f"[INFO] Pre-lesion bins:  {int(pre_mask.sum()):,}")
    print(f"[INFO] Post-lesion bins: {int(post_mask.sum()):,}")

    if selected_labels is None:
        labels_to_process = [int(l) for l in np.unique(labels)]
        if skip_noise_label and -1 in labels_to_process:
            labels_to_process.remove(-1)
    else:
        labels_to_process = [int(l) for l in selected_labels]

    labels_to_process = sorted(labels_to_process)
    if not labels_to_process:
        print("[WARN] No labels selected to process.")
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    summary_rows: List[Dict[str, Any]] = []
    label_color_lut = build_label_color_lut(labels)

    for lbl in labels_to_process:
        pre_idx = np.flatnonzero((labels == lbl) & pre_mask)
        post_idx = np.flatnonzero((labels == lbl) & post_mask)

        max_pre_samples = pre_idx.size // spectrogram_length
        max_post_samples = post_idx.size // spectrogram_length
        max_paired_samples = min(max_pre_samples, max_post_samples)

        # If num_sample_spectrograms is None or <= 0, make all available paired samples.
        if num_sample_spectrograms is None or num_sample_spectrograms <= 0:
            actual_samples = max_paired_samples
        else:
            actual_samples = min(num_sample_spectrograms, max_paired_samples)

        if actual_samples == 0:
            print(
                f"[WARN] Label {lbl}: not enough bins for paired pre/post sample. "
                f"pre={pre_idx.size}, post={post_idx.size}, need={spectrogram_length} each."
            )
            summary_rows.append(
                {
                    "animal_id": animal_id,
                    "label": lbl,
                    "treatment_date": treatment_date.isoformat(),
                    "pre_bins": int(pre_idx.size),
                    "post_bins": int(post_idx.size),
                    "samples_made": 0,
                    "saved_path": None,
                    "note": "not enough paired pre/post bins",
                }
            )
            continue

        leftover_pre = int(pre_idx.size - actual_samples * spectrogram_length)
        leftover_post = int(post_idx.size - actual_samples * spectrogram_length)
        print(
            f"[INFO] {animal_id} — label {lbl}: making {actual_samples} paired pre/post sample(s); "
            f"pre bins={pre_idx.size}, post bins={post_idx.size}, "
            f"leftover pre={leftover_pre}, leftover post={leftover_post}."
        )

        for k in range(actual_samples):
            pre_sel = _select_stitched_indices(
                pre_idx,
                sample_i=k,
                spectrogram_length=spectrogram_length,
                sample_mode=sample_mode,
                rng=rng,
            )
            post_sel = _select_stitched_indices(
                post_idx,
                sample_i=k,
                spectrogram_length=spectrogram_length,
                sample_mode=sample_mode,
                rng=rng,
            )

            S_pre = S[:, pre_sel].astype(float)
            S_post = S[:, post_sel].astype(float)

            vmin = vmax = None
            if shared_intensity_scale:
                combined = np.concatenate([S_pre.ravel(), S_post.ravel()])
                if contrast_percentiles is None:
                    vmin = float(np.nanmin(combined))
                    vmax = float(np.nanmax(combined))
                else:
                    lo, hi = contrast_percentiles
                    vmin, vmax = np.nanpercentile(combined, [lo, hi])
                    vmin = float(vmin)
                    vmax = float(vmax)

            # Version 1: with bout underline rows
            fig_b, axes_b = plt.subplots(
                4,
                1,
                figsize=(figure_width, row_height * 2.0 + underline_row_height * 2.0 + 0.15),
                sharex=True,
                gridspec_kw={
                    "height_ratios": [row_height, underline_row_height, row_height, underline_row_height],
                    "hspace": subplot_hspace,
                },
            )
            ax_pre_b, ax_pre_bouts, ax_post_b, ax_post_bouts = axes_b

            im0_b = ax_pre_b.imshow(S_pre, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_pre_b.set_ylabel(
                f"Pre\n< {treatment_date}\n{spectrogram_length:,} bins",
                rotation=0,
                labelpad=42,
                va="center",
                fontsize=12,
            )
            _hide_spectrogram_ticks(ax_pre_b)

            _draw_bout_underline_axis(
                ax_pre_bouts,
                pre_sel,
                seconds_per_bin=seconds_per_bin,
                show_bout_underlines=show_bout_underlines,
                show_bout_labels=show_bout_labels,
                min_bout_label_bins=min_bout_label_bins,
                epoch_label="bouts",
            )
            ax_pre_bouts.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax_pre_bouts.spines["bottom"].set_visible(False)

            im1_b = ax_post_b.imshow(S_post, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_post_b.set_ylabel(
                f"Post\n> {treatment_date}\n{spectrogram_length:,} bins",
                rotation=0,
                labelpad=42,
                va="center",
                fontsize=12,
            )
            _hide_spectrogram_ticks(ax_post_b)

            _draw_bout_underline_axis(
                ax_post_bouts,
                post_sel,
                seconds_per_bin=seconds_per_bin,
                show_bout_underlines=show_bout_underlines,
                show_bout_labels=show_bout_labels,
                min_bout_label_bins=min_bout_label_bins,
                epoch_label="bouts",
            )
            _set_stitched_time_axis(
                ax_post_bouts,
                n_bins=spectrogram_length,
                seconds_per_bin=seconds_per_bin,
                x_tick_step_s=x_tick_step_s,
                xlabel="Stitched label time (s)",
            )

            if show_colorbar:
                fig_b.colorbar(im1_b, ax=[ax_pre_b, ax_post_b], shrink=0.75, label="Spectrogram intensity")

            fig_b.suptitle(
                f"{animal_id} — HDBSCAN label {lbl} — sample {k + 1}/{actual_samples}",
                y=0.995,
                fontsize=16,
            )
            fig_b.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.03)

            save_path_with_bouts = None
            if save_figures and outdir is not None:
                fname_b = f"label{lbl}_pre_vs_post_sample{k + 1}_N{spectrogram_length}_with_bouts.png"
                save_path_with_bouts = (bout_lines_dir or outdir) / fname_b
                fig_b.savefig(save_path_with_bouts, dpi=300, bbox_inches="tight", pad_inches=0.05)
                print(f"[SAVE] {save_path_with_bouts}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig_b)

            # Version 2: spectrogram-only (no bout rows)
            fig_nb, axes_nb = plt.subplots(
                2,
                1,
                figsize=(figure_width, row_height * 2.0 + 0.25),
                sharex=True,
                gridspec_kw={
                    "height_ratios": [row_height, row_height],
                    "hspace": max(0.04, subplot_hspace),
                },
            )
            ax_pre_nb, ax_post_nb = axes_nb

            im0_nb = ax_pre_nb.imshow(S_pre, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_pre_nb.set_ylabel(
                f"Pre\n< {treatment_date}\n{spectrogram_length:,} bins",
                rotation=0,
                labelpad=42,
                va="center",
                fontsize=12,
            )
            _hide_spectrogram_ticks(ax_pre_nb)

            im1_nb = ax_post_nb.imshow(S_post, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_post_nb.set_ylabel(
                f"Post\n> {treatment_date}\n{spectrogram_length:,} bins",
                rotation=0,
                labelpad=42,
                va="center",
                fontsize=12,
            )
            _hide_spectrogram_ticks(ax_post_nb)
            ax_post_nb.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            _set_stitched_time_axis(
                ax_post_nb,
                n_bins=spectrogram_length,
                seconds_per_bin=seconds_per_bin,
                x_tick_step_s=x_tick_step_s,
                xlabel="Stitched label time (s)",
            )

            if show_colorbar:
                fig_nb.colorbar(im1_nb, ax=[ax_pre_nb, ax_post_nb], shrink=0.75, label="Spectrogram intensity")

            fig_nb.suptitle(
                f"{animal_id} — HDBSCAN label {lbl} — sample {k + 1}/{actual_samples}",
                y=0.995,
                fontsize=16,
            )
            fig_nb.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.08)

            save_path_no_bouts = None
            if save_figures and outdir is not None:
                fname_nb = f"label{lbl}_pre_vs_post_sample{k + 1}_N{spectrogram_length}_no_bouts.png"
                save_path_no_bouts = (no_bout_lines_dir or outdir) / fname_nb
                fig_nb.savefig(save_path_no_bouts, dpi=300, bbox_inches="tight", pad_inches=0.05)
                print(f"[SAVE] {save_path_no_bouts}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig_nb)

            summary_rows.append(
                {
                    "animal_id": animal_id,
                    "label": lbl,
                    "treatment_date": treatment_date.isoformat(),
                    "pre_bins": int(pre_idx.size),
                    "post_bins": int(post_idx.size),
                    "samples_made": int(actual_samples),
                    "sample_number": int(k + 1),
                    "spectrogram_length_bins": int(spectrogram_length),
                    "seconds_per_bin": float(seconds_per_bin),
                    "x_tick_step_s": float(x_tick_step_s),
                    "pre_bouts_in_sample": int(len(_find_stitched_bout_spans(pre_sel))),
                    "post_bouts_in_sample": int(len(_find_stitched_bout_spans(post_sel))),
                    "sample_mode": sample_mode,
                    "saved_path_with_bouts": str(save_path_with_bouts) if save_path_with_bouts is not None else None,
                    "saved_path_no_bouts": str(save_path_no_bouts) if save_path_no_bouts is not None else None,
                    "note": "ok",
                }
            )

    summary = pd.DataFrame(summary_rows)

    if outdir is not None and not summary.empty:
        summary_path = outdir / "pre_post_spectrogram_sample_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[SAVE] {summary_path}")

    if save_umap_label_key and outdir is not None:
        save_label_color_key_png(labels, label_color_lut, outdir, animal_id=animal_id)

    if make_umap_pre_post_plot:
        if "embedding_outputs" not in arr.files:
            print("[WARN] 'embedding_outputs' not found in NPZ; skipping UMAP pre/post plot.")
        else:
            plot_umap_pre_post_by_label(
                embedding=np.asarray(arr["embedding_outputs"]),
                labels=labels,
                pre_mask=pre_mask,
                post_mask=post_mask,
                lut=label_color_lut,
                outdir=outdir,
                animal_id=animal_id,
                treatment_date=treatment_date,
                show_plot=show_plots,
                random_seed=random_seed,
            )

    return summary


def _parse_selected_labels(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    out: List[int] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out or None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Make long-row pre/post stitched spectrogram samples for HDBSCAN labels. "
            "Pre-lesion is plotted on the top row and post-lesion underneath."
        )
    )
    parser.add_argument("--npz-path", required=True, help="Path to one bird .npz file.")
    parser.add_argument("--metadata-excel-path", required=True, help="Path to metadata Excel workbook.")
    parser.add_argument("--output-dir", required=True, help="Directory where output figures/CSV will be saved.")
    parser.add_argument("--animal-id", default=None, help="Optional animal ID override if it cannot be inferred from the .npz path.")
    parser.add_argument("--selected-labels", nargs="*", default=None, help="Optional labels, e.g. --selected-labels 0 1 2 or --selected-labels 0,1,2")
    parser.add_argument("--include-noise", action="store_true", help="Include HDBSCAN noise label -1.")
    parser.add_argument("--spectrogram-length", type=int, default=20000, help="Stitched bins per pre/post row. Default: 20000.")
    parser.add_argument(
        "--num-sample-spectrograms",
        type=int,
        default=0,
        help=(
            "Number of paired samples per label. Set to 0 (default) to make all available full-length paired samples. "
            "Example: if a label has 20,000 stitched bins in both pre and post, and --spectrogram-length is 5000, "
            "the script will make 4 figures for that label."
        ),
    )
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first", help="How to select stitched bins within each pre/post label.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed used when sample-mode=random.")
    parser.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")
    parser.add_argument("--cmap", default="gray_r")
    parser.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"), help="Linear percentile clipping, e.g. 1 99.5. Use --no-contrast-percentiles to disable.")
    parser.add_argument("--no-contrast-percentiles", action="store_true", help="Use raw min/max scaling instead of percentile clipping.")
    parser.add_argument("--show-colorbar", action="store_true")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively. Usually leave off in Terminal.")
    parser.add_argument("--no-umap", action="store_true", help="Skip the optional UMAP pre/post plot.")
    parser.add_argument("--no-label-key", action="store_true", help="Skip the separate label color key PNG.")
    parser.add_argument("--figure-width", type=float, default=22.0, help="Figure width in inches.")
    parser.add_argument("--row-height", type=float, default=2.8, help="Height of each pre/post spectrogram row in inches.")
    parser.add_argument("--underline-row-height", type=float, default=0.18, help="Height of each bout-underline row in inches. Smaller values place the bout lines in a tighter row.")
    parser.add_argument("--subplot-hspace", type=float, default=0.005, help="Vertical spacing between stacked axes. Smaller values reduce white space between spectrograms and bout lines.")
    parser.add_argument("--seconds-per-bin", type=float, default=0.0027, help="Seconds per spectrogram column. Default: 0.0027 for 2.70 ms/bin.")
    parser.add_argument("--x-tick-step-s", type=float, default=5.0, help="Spacing between x-axis ticks in seconds. Default: 5 seconds.")
    parser.add_argument("--no-bout-underlines", action="store_true", help="Do not draw alternating horizontal bout underlines.")
    parser.add_argument("--show-bout-labels", action="store_true", help="Label long enough bouts with bout number and duration in ms.")
    parser.add_argument("--min-bout-label-bins", type=int, default=80, help="Minimum bout length in bins before printing a bout label when --show-bout-labels is used.")

    args = parser.parse_args()

    contrast = None if args.no_contrast_percentiles else tuple(args.contrast_percentiles)

    plot_pre_post_spectrogram_samples_for_labels(
        npz_path=args.npz_path,
        metadata_excel_path=args.metadata_excel_path,
        output_dir=args.output_dir,
        animal_id=args.animal_id,
        selected_labels=_parse_selected_labels(args.selected_labels),
        skip_noise_label=not args.include_noise,
        spectrogram_length=args.spectrogram_length,
        num_sample_spectrograms=args.num_sample_spectrograms,
        treatment_day_assignment=args.treatment_day_assignment,
        sample_mode=args.sample_mode,
        random_seed=args.random_seed,
        cmap=args.cmap,
        show_colorbar=args.show_colorbar,
        show_plots=args.show_plots,
        save_figures=True,
        shared_intensity_scale=True,
        contrast_percentiles=contrast,
        make_umap_pre_post_plot=not args.no_umap,
        save_umap_label_key=not args.no_label_key,
        figure_width=args.figure_width,
        row_height=args.row_height,
        underline_row_height=args.underline_row_height,
        subplot_hspace=args.subplot_hspace,
        seconds_per_bin=args.seconds_per_bin,
        x_tick_step_s=args.x_tick_step_s,
        show_bout_underlines=not args.no_bout_underlines,
        show_bout_labels=args.show_bout_labels,
        min_bout_label_bins=args.min_bout_label_bins,
    )
