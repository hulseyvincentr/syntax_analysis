#!/usr/bin/env python3
"""Plot post-lesion TweetyBERT UMAP points over a gray pre-lesion reference.

Designed to run directly in Spyder for USA5288. The defaults assume:

    NPZ:    ~/Desktop/USA5288.npz
    output: ~/Desktop/syntax_analysis/figures/

The pre- and post-lesion markers use exactly the same ``POINT_SIZE``. Pre-lesion
points are plotted first in gray; post-lesion points are then plotted above them
using the same tab20 + tab20b + tab20c label palette used by
``syllable_sample_spectrograms_noise_merged_v3.py``.

Important: this script uses the saved shared UMAP coordinates in
``embedding_outputs``. It does not fit separate pre- and post-lesion UMAPs.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration: these defaults are ready for USA5288
# =============================================================================

NPZ_PATH = Path.home() / "Desktop" / "USA5288.npz"
OUTPUT_DIR = Path.home() / "Desktop" / "syntax_analysis" / "figures"

# USA5288 lesion date from the project metadata.
TREATMENT_DATE = date(2024, 4, 9)

UMAP_KEY = "embedding_outputs"
LABEL_KEY = "hdbscan_labels"
FILE_INDEX_KEY = "file_indices"
FILE_MAP_KEY = "file_map"
VOCALIZATION_KEY = "vocalization"

VOCALIZATION_ONLY = True
EXCLUDE_TREATMENT_DAY = True

# This matches the noise-cleaning logic used for the Figure 2 UMAP. Noise is
# reassigned only within the same recording file, never across file boundaries.
MERGE_NOISE_WITH_NEAREST_LABEL = True
PLOT_REMAINING_NOISE = False

# Equalize the number of plotted pre- and post-lesion observations so density
# differences do not merely reflect unequal recording duration.
MATCH_PERIOD_POINT_COUNTS = True
MAX_POINTS_PER_PERIOD = 200_000
RANDOM_SEED = 42

# Marker area is identical for pre and post. Only color/opacity and draw order
# differ between periods.
POINT_SIZE = 5.0
PRE_COLOR = "#666666"
PRE_ALPHA = 0.50
POST_ALPHA = 0.95

FIGSIZE = (8.0, 7.0)
DPI = 600
SHOW_AXES = False
SHOW_PERIOD_LEGEND = True
SHOW_PLOT = True
SAVE_SVG = True


# =============================================================================
# NPZ and metadata helpers
# =============================================================================

def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_file_map(file_map_obj: Any) -> dict[int, str]:
    """Normalize common NPZ ``file_map`` encodings to {index: filename}."""
    if file_map_obj is None:
        return {}

    if isinstance(file_map_obj, np.ndarray):
        if file_map_obj.shape == ():
            return normalize_file_map(file_map_obj.item())
        return normalize_file_map(file_map_obj.tolist())

    if isinstance(file_map_obj, dict):
        mapping: dict[int, str] = {}
        for key, value in file_map_obj.items():
            try:
                mapping[int(key)] = _decode_text(value)
                continue
            except (TypeError, ValueError):
                pass

            # Also support the reverse encoding {filename: index}.
            try:
                mapping[int(value)] = _decode_text(key)
            except (TypeError, ValueError):
                continue
        return mapping

    if isinstance(file_map_obj, (list, tuple)):
        if file_map_obj and isinstance(file_map_obj[0], (list, tuple)):
            mapping = {}
            for item in file_map_obj:
                if len(item) < 2:
                    continue
                try:
                    mapping[int(item[0])] = _decode_text(item[1])
                except (TypeError, ValueError):
                    try:
                        mapping[int(item[1])] = _decode_text(item[0])
                    except (TypeError, ValueError):
                        pass
            return mapping
        return {int(i): _decode_text(value) for i, value in enumerate(file_map_obj)}

    return {}


def _datetime_from_excel_serial(
    serial: float,
    month: int | None = None,
    day: int | None = None,
    hour: int | None = None,
    minute: int | None = None,
    second: int | None = None,
) -> datetime:
    dt = datetime(1899, 12, 30) + timedelta(days=float(serial))
    if month is not None and day is not None:
        dt = datetime(
            dt.year,
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
        )
    return dt


def parse_recording_datetime(file_name: str) -> datetime | None:
    """Parse recording time from the filename formats used in this dataset."""
    text = Path(_decode_text(file_name)).name

    # Excel serial split across two underscore fields:
    # USA5505_45450_59646335_6_7_16_34_6_segment_0
    match = re.search(
        r"_(\d{5})_(\d{6,9})_(\d{1,2})_(\d{1,2})_"
        r"(\d{1,2})_(\d{1,2})_(\d{1,2})(?:_|$)",
        text,
    )
    if match:
        serial_day, serial_fraction, mo, dy, hh, mm, ss = match.groups()
        try:
            serial = float(f"{serial_day}.{serial_fraction}")
            return _datetime_from_excel_serial(
                serial, int(mo), int(dy), int(hh), int(mm), int(ss)
            )
        except (TypeError, ValueError, OverflowError):
            pass

    # Excel serial stored as one decimal field:
    # R08_45786.27152402_5_9_7_32_32_segment_0
    match = re.search(
        r"_(\d{5}(?:\.\d+)?)_(\d{1,2})_(\d{1,2})_"
        r"(\d{1,2})_(\d{1,2})_(\d{1,2})(?:_|$)",
        text,
    )
    if match:
        serial, mo, dy, hh, mm, ss = match.groups()
        try:
            return _datetime_from_excel_serial(
                float(serial), int(mo), int(dy), int(hh), int(mm), int(ss)
            )
        except (TypeError, ValueError, OverflowError):
            pass

    patterns = (
        # YYYY_MM_DD_HH_MM_SS or YYYY-MM-DD-HH-MM-SS
        (
            r"(?<!\d)(20\d{2})[-_](\d{1,2})[-_](\d{1,2})"
            r"[T _-]?(\d{1,2})[:_-](\d{1,2})[:_-](\d{1,2})(?!\d)",
            "%Y %m %d %H %M %S",
        ),
        # YYYYMMDD_HHMMSS
        (
            r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)",
            "%Y %m %d %H %M %S",
        ),
        # YYMMDD_HHMMSS
        (
            r"(?<!\d)(\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?!\d)",
            "%y %m %d %H %M %S",
        ),
        # Date only: YYYY_MM_DD or YYYY-MM-DD
        (
            r"(?<!\d)(20\d{2})[-_](\d{1,2})[-_](\d{1,2})(?!\d)",
            "%Y %m %d",
        ),
    )

    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return datetime.strptime(" ".join(match.groups()), fmt)
            except ValueError:
                continue
    return None


def ensure_umap_xy(array: np.ndarray) -> np.ndarray:
    xy = np.asarray(array)
    xy = np.squeeze(xy)
    if xy.ndim != 2:
        raise ValueError(f"{UMAP_KEY!r} must be 2D after squeeze; found {xy.shape}.")
    if xy.shape[1] == 2:
        return xy.astype(float, copy=False)
    if xy.shape[0] == 2:
        return xy.T.astype(float, copy=False)
    raise ValueError(f"{UMAP_KEY!r} must have two columns; found {xy.shape}.")


def merge_noise_within_files(labels: np.ndarray, file_indices: np.ndarray) -> np.ndarray:
    """Replace -1 with the nearest non-noise temporal neighbor in each file."""
    labels = np.asarray(labels, dtype=int).copy()
    file_indices = np.asarray(file_indices).reshape(-1)

    for file_index in np.unique(file_indices):
        positions = np.flatnonzero(file_indices == file_index)
        values = labels[positions]
        noise = values == -1
        if not np.any(noise) or np.all(noise):
            continue

        valid_positions = np.flatnonzero(~noise)
        left = np.full(len(values), -1, dtype=int)
        left[valid_positions] = valid_positions
        left = np.maximum.accumulate(left)

        right = np.full(len(values), len(values), dtype=int)
        right[valid_positions] = valid_positions
        right = np.minimum.accumulate(right[::-1])[::-1]

        noise_positions = np.flatnonzero(noise)
        left_idx = left[noise_positions]
        right_idx = right[noise_positions]
        has_left = left_idx >= 0
        has_right = right_idx < len(values)
        left_distance = noise_positions - left_idx
        right_distance = right_idx - noise_positions
        choose_left = has_left & (~has_right | (left_distance <= right_distance))

        replacements = values.copy()
        use_left = noise_positions[choose_left]
        use_right = noise_positions[~choose_left & has_right]
        replacements[use_left] = values[left[use_left]]
        replacements[use_right] = values[right[use_right]]
        labels[positions] = replacements

    return labels


# =============================================================================
# Stable cluster palette
# =============================================================================

def tab60_palette() -> list[str]:
    colors: list[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors.extend(mcolors.to_hex(cmap(i)) for i in range(cmap.N))
    return colors


def build_label_lut(all_labels: np.ndarray) -> dict[int, str]:
    labels = sorted(int(label) for label in np.unique(all_labels) if int(label) != -1)
    palette = tab60_palette()
    lut = {-1: "#7f7f7f"}
    for index, label in enumerate(labels):
        lut[label] = palette[index % len(palette)]
    return lut


# =============================================================================
# Plotting
# =============================================================================

def _subsample_equal_periods(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)

    if MATCH_PERIOD_POINT_COUNTS:
        target = min(len(pre_indices), len(post_indices))
    else:
        target = max(len(pre_indices), len(post_indices))

    if MAX_POINTS_PER_PERIOD is not None:
        target = min(target, int(MAX_POINTS_PER_PERIOD))

    def sample(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= target:
            return indices
        return np.sort(rng.choice(indices, size=target, replace=False))

    return sample(pre_indices), sample(post_indices)


def make_pre_post_umap(
    npz_path: Path = NPZ_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path | None]:
    npz_path = Path(npz_path).expanduser()
    output_dir = Path(output_dir).expanduser()

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Could not find {npz_path}. If the filename or location differs, "
            "edit NPZ_PATH near the top of this script."
        )

    with np.load(npz_path, allow_pickle=True) as data:
        required = (UMAP_KEY, LABEL_KEY, FILE_INDEX_KEY, FILE_MAP_KEY)
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing NPZ keys {missing}. Available keys: {data.files}")

        xy = ensure_umap_xy(data[UMAP_KEY])
        labels = np.asarray(data[LABEL_KEY]).reshape(-1).astype(int)
        file_indices = np.asarray(data[FILE_INDEX_KEY]).reshape(-1).astype(int)
        file_map = normalize_file_map(data[FILE_MAP_KEY])

        if VOCALIZATION_ONLY and VOCALIZATION_KEY in data.files:
            vocalization = np.asarray(data[VOCALIZATION_KEY]).reshape(-1).astype(bool)
        else:
            vocalization = np.ones(len(xy), dtype=bool)

    lengths = {
        UMAP_KEY: len(xy),
        LABEL_KEY: len(labels),
        FILE_INDEX_KEY: len(file_indices),
        VOCALIZATION_KEY: len(vocalization),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"NPZ arrays have different lengths: {lengths}")
    if not file_map:
        raise ValueError(f"Could not interpret {FILE_MAP_KEY!r} from {npz_path}.")

    if MERGE_NOISE_WITH_NEAREST_LABEL:
        labels = merge_noise_within_files(labels, file_indices)

    dates_by_file: dict[int, date] = {}
    unparsable: list[str] = []
    for file_index, file_name in file_map.items():
        recording_datetime = parse_recording_datetime(file_name)
        if recording_datetime is None:
            unparsable.append(file_name)
        else:
            dates_by_file[int(file_index)] = recording_datetime.date()

    point_dates = np.asarray(
        [dates_by_file.get(int(file_index)) for file_index in file_indices],
        dtype=object,
    )
    parsed_date_mask = np.asarray([value is not None for value in point_dates])
    if not np.any(parsed_date_mask):
        examples = list(file_map.values())[:8]
        raise ValueError(
            "Could not parse recording dates from file_map. Example filenames:\n"
            + "\n".join(f"  {name}" for name in examples)
        )

    finite_xy = np.isfinite(xy).all(axis=1)
    keep = vocalization & finite_xy & parsed_date_mask
    if not PLOT_REMAINING_NOISE:
        keep &= labels != -1

    pre_mask = keep & np.asarray(
        [value is not None and value < TREATMENT_DATE for value in point_dates]
    )
    if EXCLUDE_TREATMENT_DAY:
        post_mask = keep & np.asarray(
            [value is not None and value > TREATMENT_DATE for value in point_dates]
        )
    else:
        post_mask = keep & np.asarray(
            [value is not None and value >= TREATMENT_DATE for value in point_dates]
        )

    pre_indices = np.flatnonzero(pre_mask)
    post_indices = np.flatnonzero(post_mask)
    if len(pre_indices) == 0 or len(post_indices) == 0:
        parsed_dates = sorted(set(value for value in point_dates if value is not None))
        raise ValueError(
            f"Need both pre- and post-lesion points around {TREATMENT_DATE}. "
            f"Found pre={len(pre_indices):,}, post={len(post_indices):,}. "
            f"Parsed date range: {parsed_dates[:1]} to {parsed_dates[-1:]}."
        )

    original_pre_n = len(pre_indices)
    original_post_n = len(post_indices)
    pre_indices, post_indices = _subsample_equal_periods(pre_indices, post_indices)

    # Build the LUT from every retained label, not separately by period, so a
    # label always receives the same color.
    label_lut = build_label_lut(labels[keep])
    post_colors = [label_lut[int(label)] for label in labels[post_indices]]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Same marker size for both calls. Pre is drawn first and post second.
    ax.scatter(
        xy[pre_indices, 0],
        xy[pre_indices, 1],
        s=POINT_SIZE,
        c=PRE_COLOR,
        alpha=PRE_ALPHA,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        xy[post_indices, 0],
        xy[post_indices, 1],
        s=POINT_SIZE,
        c=post_colors,
        alpha=POST_ALPHA,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    ax.set_aspect("equal", adjustable="box")
    if SHOW_AXES:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    else:
        ax.set_axis_off()

    if SHOW_PERIOD_LEGEND:
        post_example_color = next(
            (label_lut[label] for label in sorted(label_lut) if label != -1),
            "#1f77b4",
        )
        handles = [
            mlines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6,
                markerfacecolor=PRE_COLOR, markeredgewidth=0, alpha=PRE_ALPHA,
                label="Pre-lesion (gray)",
            ),
            mlines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6,
                markerfacecolor=post_example_color, markeredgewidth=0,
                label="Post-lesion (colors denote cluster)",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout(pad=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "USA5288_pre_gray_post_cluster_colors_umap.png"
    svg_path = output_dir / "USA5288_pre_gray_post_cluster_colors_umap.svg"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if SAVE_SVG:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    else:
        svg_path = None

    print(f"Loaded: {npz_path}")
    print(f"Treatment date: {TREATMENT_DATE}")
    print(f"Parsed {len(dates_by_file):,} file dates; {len(unparsable):,} unparsed.")
    print(
        f"Eligible points before matched sampling: "
        f"pre={original_pre_n:,}, post={original_post_n:,}"
    )
    print(
        f"Plotted points: pre={len(pre_indices):,}, post={len(post_indices):,}; "
        f"marker size={POINT_SIZE} for both"
    )
    print(f"Saved PNG: {png_path}")
    if svg_path is not None:
        print(f"Saved SVG: {svg_path}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return png_path, svg_path


if __name__ == "__main__":
    make_pre_post_umap()
