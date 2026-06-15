#!/usr/bin/env python3
"""
phrase_position_acoustic_crosscorr_v19_expanded_context_qc.py

Analyze whether acoustic structure changes as a repeated syllable phrase progresses.

Version 12 keeps the version 11 bug fixes and adds a short-lag local acoustic-change-rate analysis. Instead of relying only on a fixed lag such as lag 10, it can compute all within-phrase spectrogram-pair distances for nearby syllables, default nearest neighbors only, then plot local acoustic change rate as a function of repeat number or elapsed phrase time.

For each target cluster label, this script:
1) finds contiguous phrase-label runs within each recording file
2) uses the same Gardner-style canary segmentation approach used in the pitch/entropy scripts
   to split each run into individual syllable renditions
3) computes per-syllable acoustic summaries:
      - mean / median / q95 pitch derivative
      - mean / median / q95 Wiener entropy
      - mean / median pitch
      - syllable duration
4) annotates each segmented syllable with phrase-position variables:
      - repeat_index_in_phrase
      - n_previous_segments
      - elapsed_time_in_phrase_s
      - fraction_through_phrase
      - phrase_duration_s
      - n_segments_in_phrase
5) makes plots and tables to test whether acoustic features drift later in long repeated phrases.
6) optionally computes spectrogram cross-correlation to an early-phrase template and tests whether similarity declines with repeated syllables.
7) also computes intra-phrase cross-correlation to the Nth previous syllable, default N=10.
8) computes short-lag local acoustic-change rates from all nearby pairs within each phrase, default nearest-neighbor pairs.

Primary outputs:
- <animal>_label<label>_phrase_position_segment_features.csv
- <animal>_label<label>_phrase_position_stats.csv
- <animal>_label<label>_early_late_window_summary.csv
- <animal>_label<label>_feature_vs_elapsed_time.png
- <animal>_label<label>_feature_vs_previous_repeats.png
- <animal>_label<label>_early_vs_late_windows.png
- <animal>_label<label>_slope_summary_by_period.png
- <animal>_label<label>_crosscorr_vs_elapsed_time.png
- <animal>_label<label>_crosscorr_vs_previous_repeats.png
- <animal>_label<label>_local_shortlag_pairs.csv
- <animal>_label<label>_local_shortlag_phrase_rate_summary.csv
- <animal>_label<label>_local_rate_vs_elapsed_time.png
- <animal>_label<label>_local_rate_vs_previous_repeats.png
- <animal>_label<label>_local_shortlag_pair_qc_selected.csv
- <animal>_label<label>_local_shortlag_pair_qc.pdf
- <animal>_label<label>_phrase_context_qc_selected_pairs.csv
- <animal>_label<label>_phrase_context_qc.pdf
- <animal>_label<label>_expanded_phrase_context_qc_selected_pairs.csv
- <animal>_label<label>_expanded_phrase_context_qc.pdf

Notes:
- By default, this script treats a "phrase" as a contiguous run of the target cluster label within one file_index.
- To recover long stutters split across chunked file_map entries such as _segment_0, _segment_1, use --stitch-across-file-segments.
- If your decoder briefly interrupts a phrase with small gaps/noise labels, use --phrase-label-smoothing target_majority_vote and/or --max-gap-bins-to-merge.
- By default, the first and last canary segments in each contiguous run are dropped, matching the previous
  QC-driven choice to avoid edge/truncation artifacts.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
    from scipy.signal import butter, filtfilt, find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    butter = filtfilt = find_peaks = None
    SCIPY_AVAILABLE = False

PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PERIOD_TITLES = {
    "early_pre": "Early pre-lesion",
    "late_pre": "Late pre-lesion",
    "early_post": "Early post-lesion",
    "late_post": "Late post-lesion",
}
PRE_POST_MAP = {
    "early_pre": "pre",
    "late_pre": "pre",
    "early_post": "post",
    "late_post": "post",
}
DEFAULT_FEATURES = [
    "mean_pitch_derivative_khz_per_s",
    "q95_pitch_derivative_khz_per_s",
    "mean_wiener_entropy",
    "q95_wiener_entropy",
]
FEATURE_LABELS = {
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative (kHz/s)",
    "median_pitch_derivative_khz_per_s": "Median pitch derivative (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative (kHz/s)",
    "mean_wiener_entropy": "Mean Wiener entropy",
    "median_wiener_entropy": "Median Wiener entropy",
    "q95_wiener_entropy": "95th percentile Wiener entropy",
    "mean_pitch_khz": "Mean pitch (kHz)",
    "median_pitch_khz": "Median pitch (kHz)",
    "segment_duration_s": "Segment duration (s)",
    "corr_to_phrase_early_template": "Spectrogram corr. to early-phrase template",
    "distance_from_phrase_early_template": "Spectrogram distance from early-phrase template (1 - corr)",
    "corr_to_lag10_previous_syllable": "Spectrogram corr. to syllable 10 repeats earlier",
    "distance_from_lag10_previous_syllable": "Spectrogram distance from syllable 10 repeats earlier (1 - corr)",
    "local_rate_spectral_displacement_per_repeat": "Local acoustic change rate, d per repeat",
    "local_rate_spectral_displacement_sq_per_repeat": "Local acoustic change rate, d² per repeat",
    "shortlag_pair_corr": "Short-lag pair spectrogram correlation",
    "shortlag_pair_spectral_displacement": "Short-lag spectral displacement d",
    "shortlag_pair_spectral_displacement_sq": "Short-lag spectral displacement d²",
}


def get_lag_crosscorr_feature_names(lags: Sequence[int]) -> List[str]:
    """Return correlation and distance feature-column names for requested lags."""
    out: List[str] = []
    for lag in lags:
        lag = int(lag)
        out.extend([
            f"corr_to_lag{lag}_previous_syllable",
            f"distance_from_lag{lag}_previous_syllable",
        ])
    return out


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        value = str(value)
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def feature_label(feature: str) -> str:
    """Human-readable axis/stat label, including dynamic lag-correlation columns."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    m = re.match(r"corr_to_lag(\d+)_previous_syllable$", str(feature))
    if m:
        lag = int(m.group(1))
        return f"Spectrogram corr. to syllable {lag} repeats earlier"
    m = re.match(r"distance_from_lag(\d+)_previous_syllable$", str(feature))
    if m:
        lag = int(m.group(1))
        return f"Spectrogram distance from syllable {lag} repeats earlier (1 - corr)"
    m = re.match(r"local_rate_spectral_displacement(?:_sq)?_per_repeat$", str(feature))
    if m:
        if "_sq_" in str(feature):
            return "Local acoustic change rate, d² per repeat"
        return "Local acoustic change rate, d per repeat"
    return str(feature)


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_values(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def normalize_label_value(label) -> str:
    """Return a stable string representation for cluster labels.

    This avoids silent mismatches such as NPZ labels becoming "17" while labels
    read from a CSV become "17.0" after pandas has treated the column as float.
    Integer-like numeric labels are normalized to integer strings; all other labels
    are stripped strings.
    """
    if isinstance(label, bytes):
        label = label.decode(errors="replace")
    if label is None:
        return ""
    try:
        if pd.isna(label):
            return ""
    except Exception:
        pass
    s = str(label).strip()
    if s == "":
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def normalize_label_array(labels: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels)
    flat = [normalize_label_value(x) for x in labels_arr.ravel()]
    return np.asarray(flat, dtype=str).reshape(labels_arr.shape)


def get_file_map_dict(file_map_obj):
    """Convert loaded NPZ file_map object to a plain dictionary."""
    if isinstance(file_map_obj, dict):
        return file_map_obj
    if isinstance(file_map_obj, np.ndarray):
        if file_map_obj.shape == ():
            obj = file_map_obj.item()
            if isinstance(obj, dict):
                return obj
            return dict(obj)
        if len(file_map_obj) == 1:
            obj = file_map_obj[0]
            if isinstance(obj, dict):
                return obj
            return dict(obj)
    try:
        return dict(file_map_obj)
    except Exception as exc:
        raise ValueError("Could not convert file_map to a dictionary.") from exc


def file_entry_to_name(entry) -> str:
    if isinstance(entry, (list, tuple, np.ndarray)):
        if len(entry) == 0:
            return ""
        entry = entry[0]
    return os.path.basename(str(entry))


def get_file_map_entry(file_map: Dict, fidx: int):
    """Fetch file_map entry robustly when keys are ints, numpy ints, or strings."""
    keys_to_try = []
    try:
        keys_to_try.append(int(fidx))
    except Exception:
        pass
    try:
        keys_to_try.append(np.int64(fidx))
    except Exception:
        pass
    keys_to_try.append(str(fidx))
    try:
        keys_to_try.append(str(int(fidx)))
    except Exception:
        pass

    seen = set()
    for key in keys_to_try:
        key_marker = (type(key), str(key))
        if key_marker in seen:
            continue
        seen.add(key_marker)
        try:
            if key in file_map:
                return file_map[key]
        except Exception:
            pass
        try:
            value = file_map.get(key, None)
            if value is not None:
                return value
        except Exception:
            pass
    raise KeyError(f"file index {fidx} not found in file_map with int or string keys")


def parse_datetime_from_filename(file_name) -> pd.Timestamp:
    """Try several filename datetime formats, including Excel serial dates."""
    base = os.path.basename(str(file_name))

    segmented_excel_match = re.search(
        r'_(?P<serial>\d{5}(?:\.\d+)?)_'
        r'(?P<m>\d{1,2})_(?P<d>\d{1,2})_'
        r'(?P<H>\d{1,2})_(?P<M>\d{1,2})_(?P<S>\d{1,2})'
        r'(?:_segment|\.)',
        base,
    )
    if segmented_excel_match:
        gd = segmented_excel_match.groupdict()
        serial = float(gd["serial"])
        serial_dt = pd.to_datetime(serial, unit="D", origin="1899-12-30")
        return pd.Timestamp(
            year=int(serial_dt.year),
            month=int(gd["m"]),
            day=int(gd["d"]),
            hour=int(gd["H"]),
            minute=int(gd["M"]),
            second=int(gd["S"]),
        )

    excel_match = re.search(r'(?<!\d)(?P<serial>\d{5}(?:\.\d+)?)(?!\d)', base)
    if excel_match:
        serial = float(excel_match.group("serial"))
        if 30000 <= serial <= 60000:
            return pd.to_datetime(serial, unit="D", origin="1899-12-30")

    patterns = [
        r'(?P<y>20\d{2})[_-](?P<m>\d{1,2})[_-](?P<d>\d{1,2})[_-](?P<H>\d{1,2})[_-](?P<M>\d{1,2})(?:[_-](?P<S>\d{1,2}))?',
        r'(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        r'(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        r'(?P<y>20\d{2})[_-](?P<m>\d{1,2})[_-](?P<d>\d{1,2})',
    ]
    for pat in patterns:
        match = re.search(pat, base)
        if match:
            gd = match.groupdict()
            y = int(gd["y"])
            if y < 100:
                y += 2000
            return pd.Timestamp(
                year=y,
                month=int(gd["m"]),
                day=int(gd["d"]),
                hour=int(gd.get("H") or 0),
                minute=int(gd.get("M") or 0),
                second=int(gd.get("S") or 0),
            )

    raise ValueError(
        f"Could not parse a datetime from filename: {file_name}\n"
        "Edit parse_datetime_from_filename() for your filename format."
    )


def load_treatment_date(metadata_excel_path, animal_id, metadata_sheet="metadata",
                        animal_id_col="Animal ID", treatment_date_col="Treatment date") -> pd.Timestamp:
    meta = pd.read_excel(metadata_excel_path, sheet_name=metadata_sheet)
    hits = meta[meta[animal_id_col].astype(str) == str(animal_id)]
    if hits.empty:
        raise ValueError(f"Animal ID {animal_id} not found in metadata sheet {metadata_sheet!r}.")
    td = pd.to_datetime(hits.iloc[0][treatment_date_col])
    if pd.isna(td):
        raise ValueError(f"Treatment date missing for animal {animal_id}.")
    return td.normalize()


def split_in_half(sorted_items: Sequence[int]) -> Tuple[List[int], List[int]]:
    items = list(sorted_items)
    n = len(items)
    if n == 0:
        return [], []
    if n == 1:
        return items, []
    mid = n // 2
    return items[:mid], items[mid:]


def split_by_file_period(unique_file_indices: Sequence[int], file_time_lookup: Dict[int, pd.Timestamp],
                         treatment_date: pd.Timestamp, include_treatment_day_in_post=False,
                         split_mode="file_half") -> Dict[str, List[int]]:
    pre_files = []
    post_files = []
    for fidx in unique_file_indices:
        dt = pd.Timestamp(file_time_lookup[int(fidx)]).normalize()
        if dt < treatment_date:
            pre_files.append(int(fidx))
        elif dt > treatment_date:
            post_files.append(int(fidx))
        elif include_treatment_day_in_post:
            post_files.append(int(fidx))

    pre_files = sorted(pre_files, key=lambda x: file_time_lookup[int(x)])
    post_files = sorted(post_files, key=lambda x: file_time_lookup[int(x)])

    if split_mode == "file_half":
        early_pre, late_pre = split_in_half(pre_files)
        early_post, late_post = split_in_half(post_files)
    elif split_mode == "file_median":
        def median_split(files):
            if len(files) == 0:
                return [], []
            times = np.asarray([pd.Timestamp(file_time_lookup[f]).value for f in files], dtype=np.int64)
            med = np.median(times)
            left = [f for f in files if pd.Timestamp(file_time_lookup[f]).value <= med]
            right = [f for f in files if pd.Timestamp(file_time_lookup[f]).value > med]
            if len(left) == 0 or len(right) == 0:
                return split_in_half(files)
            return left, right
        early_pre, late_pre = median_split(pre_files)
        early_post, late_post = median_split(post_files)
    else:
        raise ValueError("split_mode must be file_half or file_median")

    return {
        "early_pre": early_pre,
        "late_pre": late_pre,
        "early_post": early_post,
        "late_post": late_post,
    }


def to_linear_power(X: np.ndarray, spec_scale="linear") -> np.ndarray:
    eps = 1e-12
    X = np.asarray(X, dtype=float)
    if spec_scale == "linear":
        Y = X.copy()
        finite_rows = np.any(np.isfinite(Y), axis=1)
        if np.any(finite_rows):
            row_min = np.nanmin(Y[finite_rows, :], axis=1, keepdims=True)
            bad_rows = row_min[:, 0] <= 0
            finite_idx = np.flatnonzero(finite_rows)
            if np.any(bad_rows):
                Y[finite_idx[bad_rows], :] = Y[finite_idx[bad_rows], :] - row_min[bad_rows] + eps
        finite_vals = np.isfinite(Y)
        Y[finite_vals & (Y < eps)] = eps
        return Y
    if spec_scale == "log10":
        return np.power(10.0, X)
    if spec_scale == "loge":
        return np.exp(X)
    if spec_scale == "shift":
        Y = X.copy()
        finite_rows = np.any(np.isfinite(Y), axis=1)
        if np.any(finite_rows):
            row_min = np.nanmin(Y[finite_rows, :], axis=1, keepdims=True)
            finite_idx = np.flatnonzero(finite_rows)
            Y[finite_idx, :] = Y[finite_idx, :] - row_min + eps
        return Y
    raise ValueError("spec_scale must be linear, log10, loge, or shift")


def moving_average_nan(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    half = int(window) // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        chunk = x[lo:hi]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def compute_wiener_entropy(X_tx_f: np.ndarray, spec_scale="linear") -> np.ndarray:
    eps = 1e-12
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    out = np.full(power.shape[0], np.nan, dtype=float)
    for i in range(power.shape[0]):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        row = row[np.isfinite(row)]
        row = np.clip(row, eps, None)
        gm = np.exp(np.mean(np.log(row)))
        am = np.mean(row)
        out[i] = gm / (am + eps)
    return out


def compute_total_power(X_tx_f: np.ndarray, spec_scale="linear") -> np.ndarray:
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    out = np.full(power.shape[0], np.nan, dtype=float)
    for i in range(power.shape[0]):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        out[i] = np.nansum(row)
    return out


def compute_pitch_trace_and_derivative(
        X_tx_f: np.ndarray,
        bin_s: float,
        max_freq_khz=12.0,
        pitch_band_min_khz=0.8,
        pitch_band_max_khz=10.0,
        spec_scale="linear",
        smoothing_window=5,
        min_peak_fraction=0.0) -> Tuple[np.ndarray, np.ndarray]:
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    T, F = power.shape
    freqs_khz = np.linspace(0, max_freq_khz, F)
    band_mask = (freqs_khz >= pitch_band_min_khz) & (freqs_khz <= pitch_band_max_khz)
    if not np.any(band_mask):
        raise ValueError("Pitch band does not overlap frequency axis.")
    band_freqs = freqs_khz[band_mask]
    pitch_khz = np.full(T, np.nan, dtype=float)
    for i in range(T):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        bp = row[band_mask]
        if np.all(~np.isfinite(bp)):
            continue
        bp = np.nan_to_num(bp, nan=0.0)
        total = bp.sum()
        if total <= 0:
            continue
        peak_idx = int(np.argmax(bp))
        if min_peak_fraction > 0 and bp[peak_idx] / total < min_peak_fraction:
            continue
        pitch_khz[i] = band_freqs[peak_idx]
    pitch_khz = moving_average_nan(pitch_khz, int(smoothing_window))
    pitch_derivative = np.full(T, np.nan, dtype=float)
    for i in range(1, T):
        if np.isfinite(pitch_khz[i]) and np.isfinite(pitch_khz[i - 1]):
            pitch_derivative[i] = abs(pitch_khz[i] - pitch_khz[i - 1]) / bin_s
    return pitch_khz, pitch_derivative


def crop_frequency_band(X: np.ndarray, max_freq_khz=12.0, freq_min_khz=0.0, freq_max_khz=None) -> np.ndarray:
    """Crop a time x frequency spectrogram to a frequency band."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return X
    if freq_max_khz is None:
        freq_max_khz = max_freq_khz
    F = X.shape[1]
    freqs = np.linspace(0, max_freq_khz, F)
    mask = (freqs >= freq_min_khz) & (freqs <= freq_max_khz)
    if not np.any(mask):
        return X
    return X[:, mask]


def transform_spectrogram_for_corr(X: np.ndarray, spec_scale="linear", corr_transform="log_power") -> np.ndarray:
    """Transform spectrogram values before correlation."""
    X = np.asarray(X, dtype=float)
    if corr_transform == "raw":
        return X.copy()
    power = to_linear_power(X, spec_scale=spec_scale)
    if corr_transform == "linear_power":
        return power
    if corr_transform == "log_power":
        return np.log10(power + 1e-12)
    raise ValueError("corr_transform must be raw, linear_power, or log_power")


def time_normalize_spectrogram(X: np.ndarray, n_time=64) -> Optional[np.ndarray]:
    """Linearly time-normalize a time x frequency spectrogram."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    T, F = X.shape
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, int(n_time))
    out = np.full((int(n_time), F), np.nan, dtype=float)
    for f in range(F):
        y = X[:, f]
        good = np.isfinite(y)
        if np.sum(good) < 2:
            continue
        out[:, f] = np.interp(new, old[good], y[good])
    return out


def vectorize_for_corr(X_norm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Flatten, z-score-ish normalize, and return a unit vector for correlation."""
    if X_norm is None:
        return None
    v = np.asarray(X_norm, dtype=float).ravel()
    good = np.isfinite(v)
    if np.sum(good) < 5:
        return None
    fill = np.nanmean(v[good])
    v = np.where(good, v, fill)
    v = v - np.mean(v)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return v / norm


def corr_between_unit_vectors(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def corr_to_spectral_displacement(corr: float) -> float:
    """Convert correlation between unit-normalized spectrogram vectors to Euclidean distance.

    For unit vectors a and b, ||a-b||^2 = 2 * (1 - corr(a,b)).
    This gives a bounded, distance-like measure where 0 means identical vectors.
    """
    if not np.isfinite(corr):
        return np.nan
    c = float(np.clip(corr, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - c))))


def build_local_shortlag_pair_rows(
    seg_infos: List[Dict[str, object]],
    base_row: Dict[str, object],
    max_lag: int = 1,
    global_run_indices: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    """Build rows for all short-lag syllable pairs within one phrase.

    Each row is one pair (i, j) with 1 <= j-i <= max_lag. The row is placed at
    the midpoint between the two syllables in repeat/time coordinates. For
    max_lag=1, local_rate_spectral_displacement_per_repeat is simply the
    nearest-neighbor spectrogram displacement between consecutive renditions.

    If global_run_indices is provided, the output rows also include the global
    start/end bins for both syllables in the pair so QC figures can later pull
    the exact spectrogram snippets back out of the original NPZ.
    """
    rows: List[Dict[str, object]] = []
    max_lag = int(max_lag)
    global_run_indices = np.asarray(global_run_indices, dtype=int) if global_run_indices is not None else None
    if max_lag <= 0 or len(seg_infos) < 2:
        return rows

    def _global_bounds(local_start: int, local_end: int) -> Tuple[float, float]:
        if global_run_indices is None:
            return np.nan, np.nan
        if local_start < 0 or local_end <= local_start or local_start >= len(global_run_indices) or (local_end - 1) >= len(global_run_indices):
            return np.nan, np.nan
        return int(global_run_indices[local_start]), int(global_run_indices[local_end - 1]) + 1

    for i in range(len(seg_infos)):
        vi = seg_infos[i].get("corr_vec")
        if vi is None:
            continue
        j_stop = min(len(seg_infos), i + max_lag + 1)
        for j in range(i + 1, j_stop):
            vj = seg_infos[j].get("corr_vec")
            if vj is None:
                continue
            lag = int(j - i)
            corr = corr_between_unit_vectors(vi, vj)
            d = corr_to_spectral_displacement(corr)
            d2 = d * d if np.isfinite(d) else np.nan
            rep_i = int(seg_infos[i].get("seg_order", i + 1))
            rep_j = int(seg_infos[j].get("seg_order", j + 1))
            elapsed_i = float(seg_infos[i].get("elapsed_mid", np.nan))
            elapsed_j = float(seg_infos[j].get("elapsed_mid", np.nan))
            start_elapsed_i = float(seg_infos[i].get("elapsed_start", np.nan))
            end_elapsed_j = float(seg_infos[j].get("elapsed_end", np.nan))
            s0_i = int(seg_infos[i].get("s0", -1))
            s1_i = int(seg_infos[i].get("s1", -1))
            s0_j = int(seg_infos[j].get("s0", -1))
            s1_j = int(seg_infos[j].get("s1", -1))
            global_start_i, global_end_i = _global_bounds(s0_i, s1_i)
            global_start_j, global_end_j = _global_bounds(s0_j, s1_j)
            mid_repeat = 0.5 * (rep_i + rep_j)
            mid_elapsed = 0.5 * (elapsed_i + elapsed_j) if np.isfinite(elapsed_i) and np.isfinite(elapsed_j) else np.nan
            row = dict(base_row)
            row.update({
                "pair_start_repeat_index": rep_i,
                "pair_end_repeat_index": rep_j,
                "pair_mid_repeat_index": mid_repeat,
                "pair_mid_n_previous_segments": mid_repeat - 1.0,
                "pair_start_elapsed_time_s": elapsed_i,
                "pair_end_elapsed_time_s": elapsed_j,
                "pair_start_segment_start_elapsed_s": start_elapsed_i,
                "pair_end_segment_end_elapsed_s": end_elapsed_j,
                "pair_mid_elapsed_time_s": mid_elapsed,
                "pair_lag_repeats": lag,
                "pair_start_segment_local_start_bin": s0_i,
                "pair_start_segment_local_end_bin": s1_i,
                "pair_end_segment_local_start_bin": s0_j,
                "pair_end_segment_local_end_bin": s1_j,
                "pair_start_segment_global_start_bin": global_start_i,
                "pair_start_segment_global_end_bin": global_end_i,
                "pair_end_segment_global_start_bin": global_start_j,
                "pair_end_segment_global_end_bin": global_end_j,
                "shortlag_pair_corr": corr,
                "shortlag_pair_spectral_displacement": d,
                "shortlag_pair_spectral_displacement_sq": d2,
                "local_rate_spectral_displacement_per_repeat": d / lag if np.isfinite(d) and lag > 0 else np.nan,
                "local_rate_spectral_displacement_sq_per_repeat": d2 / lag if np.isfinite(d2) and lag > 0 else np.nan,
                # Aliases used by the existing plotting/statistics helpers.
                "elapsed_time_in_phrase_s": mid_elapsed,
                "n_previous_segments": mid_repeat - 1.0,
                "repeat_index_in_phrase": mid_repeat,
            })
            rows.append(row)
    return rows


def compute_local_phrase_rate_summary(pair_df: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    """Summarize short-lag acoustic change within each phrase.

    If max_lag > 1, the linear_slope columns estimate how displacement grows
    with pair lag within each phrase. If max_lag == 1, x is constant, so the
    regression status will be constant_x; in that case the median/mean local
    nearest-neighbor rate columns are the relevant summary.
    """
    if pair_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["animal_id", "label", "period", "pre_post", "phrase_id"]
    for group_key, sub in pair_df.groupby(group_cols, dropna=False):
        first = sub.iloc[0]
        x = pd.to_numeric(sub["pair_lag_repeats"], errors="coerce").to_numpy(dtype=float)
        for distance_col, rate_col, label_suffix in [
            (
                "shortlag_pair_spectral_displacement",
                "local_rate_spectral_displacement_per_repeat",
                "d",
            ),
            (
                "shortlag_pair_spectral_displacement_sq",
                "local_rate_spectral_displacement_sq_per_repeat",
                "d2",
            ),
        ]:
            y = pd.to_numeric(sub[distance_col], errors="coerce").to_numpy(dtype=float)
            r = pd.to_numeric(sub[rate_col], errors="coerce").to_numpy(dtype=float)
            fit = _safe_spearman_and_linear(x, y, min_n=min_pairs)
            valid_r = r[np.isfinite(r)]
            row = {
                "animal_id": first.get("animal_id", np.nan),
                "label": first.get("label", np.nan),
                "period": first.get("period", np.nan),
                "pre_post": first.get("pre_post", np.nan),
                "phrase_id": first.get("phrase_id", np.nan),
                "feature": rate_col,
                "feature_label": feature_label(rate_col),
                "distance_definition": label_suffix,
                "n_pairs": int(np.sum(np.isfinite(y))),
                "n_unique_lags": int(np.unique(x[np.isfinite(x)]).size) if np.any(np.isfinite(x)) else 0,
                "pair_lag_min": float(np.nanmin(x)) if np.any(np.isfinite(x)) else np.nan,
                "pair_lag_max": float(np.nanmax(x)) if np.any(np.isfinite(x)) else np.nan,
                "mean_local_rate_per_repeat": float(np.nanmean(valid_r)) if valid_r.size else np.nan,
                "median_local_rate_per_repeat": float(np.nanmedian(valid_r)) if valid_r.size else np.nan,
                "linear_slope_distance_per_lag": fit["linear_slope"],
                "linear_intercept_distance_vs_lag": fit["linear_intercept"],
                "linear_r_distance_vs_lag": fit["linear_r"],
                "linear_p_distance_vs_lag": fit["linear_p"],
                "spearman_r_distance_vs_lag": fit["spearman_r"],
                "spearman_p_distance_vs_lag": fit["spearman_p"],
                "fit_status": fit["fit_status"],
                "phrase_duration_s": first.get("phrase_duration_s", np.nan),
                "n_segments_in_phrase": first.get("n_segments_in_phrase", np.nan),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _select_evenly_spaced_indices(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, k)
    idx = np.round(idx).astype(int)
    # ensure unique and ordered, then backfill if rounding collided
    chosen = []
    seen = set()
    for ii in idx.tolist():
        if ii not in seen:
            chosen.append(ii)
            seen.add(ii)
    cand = 0
    while len(chosen) < k and cand < n:
        if cand not in seen:
            chosen.append(cand)
            seen.add(cand)
        cand += 1
    return sorted(chosen)


def select_local_pair_qc_rows(
    pair_df: pd.DataFrame,
    n_total: int = 8,
    sample_mode: str = "quantile_corr",
    pairs_per_period: int = 0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Choose a small QC set of local short-lag pairs for visualization."""
    if pair_df.empty or int(n_total) <= 0:
        return pd.DataFrame(columns=list(pair_df.columns) if not pair_df.empty else None)

    work = pair_df.copy()
    work["shortlag_pair_corr"] = pd.to_numeric(work.get("shortlag_pair_corr"), errors="coerce")
    work = work[np.isfinite(work["shortlag_pair_corr"])].copy()
    if work.empty:
        return work

    rng = np.random.default_rng(int(random_seed))

    def _sample_block(sub: pd.DataFrame, k: int) -> pd.DataFrame:
        if sub.empty or k <= 0:
            return sub.iloc[0:0].copy()
        if len(sub) <= k:
            return sub.copy()
        if sample_mode == "random":
            return sub.iloc[np.sort(rng.choice(len(sub), size=k, replace=False))].copy()
        sub_sorted = sub.sort_values("shortlag_pair_corr", ascending=True).reset_index(drop=True)
        if sample_mode == "lowest_corr":
            return sub_sorted.iloc[:k].copy()
        if sample_mode == "highest_corr":
            return sub_sorted.iloc[-k:].copy()
        # default: quantile-spaced across the correlation distribution
        pick_idx = _select_evenly_spaced_indices(len(sub_sorted), k)
        return sub_sorted.iloc[pick_idx].copy()

    picked_parts = []
    if int(pairs_per_period) > 0 and "period" in work.columns:
        for period in PERIOD_ORDER:
            sub = work[work["period"] == period].copy()
            if sub.empty:
                continue
            picked_parts.append(_sample_block(sub, int(pairs_per_period)))
        if picked_parts:
            work = pd.concat(picked_parts, ignore_index=False)
            work = work.loc[~work.index.duplicated(keep="first")].copy()
            if len(work) > int(n_total):
                work = _sample_block(work, int(n_total))
            return work.reset_index(drop=True)

    return _sample_block(work, int(n_total)).reset_index(drop=True)


def extract_global_segment_spectrogram(
    X: np.ndarray,
    start_bin,
    end_bin,
    pad_bins: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return a spectrogram segment plus padding metadata for QC plotting."""
    try:
        s0 = int(start_bin)
        s1 = int(end_bin)
        pad_bins = int(pad_bins)
    except Exception:
        return None, None, None, None, None
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or s1 <= s0 or s0 < 0 or s0 >= X.shape[0]:
        return None, None, None, None, None
    s1 = min(s1, X.shape[0])
    if s1 <= s0:
        return None, None, None, None, None
    pad_bins = max(0, pad_bins)
    disp0 = max(0, s0 - pad_bins)
    disp1 = min(X.shape[0], s1 + pad_bins)
    if disp1 <= disp0:
        return None, None, None, None, None
    return X[disp0:disp1, :], disp0, disp1, s0 - disp0, s1 - disp0


def spectrogram_for_qc_display(X_spec: Optional[np.ndarray], args) -> Optional[np.ndarray]:
    """Return a display-only spectrogram for QC figures.

    This deliberately does NOT reuse the vector/correlation normalization.
    Correlation values are still computed from the analysis transform; this
    function is only for making the figures look like interpretable
    spectrograms: smooth grayscale log-power images with dark energy on a
    light background.
    """
    if X_spec is None or np.asarray(X_spec).size == 0:
        return None
    freq_min = args.qc_plot_freq_min_khz if args.qc_plot_freq_min_khz is not None else args.corr_freq_min_khz
    freq_max = args.qc_plot_freq_max_khz if args.qc_plot_freq_max_khz is not None else args.corr_freq_max_khz
    Xc = crop_frequency_band(
        X_spec,
        max_freq_khz=args.max_freq_khz,
        freq_min_khz=freq_min if freq_min is not None else 0.0,
        freq_max_khz=freq_max,
    )
    Xc = np.asarray(Xc, dtype=float)
    if Xc.size == 0:
        return None

    mode = str(getattr(args, "qc_plot_transform", "log_power"))
    if mode == "raw":
        Y = Xc.copy()
    else:
        power = to_linear_power(Xc, spec_scale=args.spec_scale)
        power = np.asarray(power, dtype=float)
        power[~np.isfinite(power)] = np.nan
        power = np.maximum(power, 0.0)
        if mode == "linear_power":
            Y = power
        elif mode == "log_power":
            # Use a data-dependent small floor only for display so the
            # background does not become a huge negative log tail.
            finite_pos = power[np.isfinite(power) & (power > 0)]
            if finite_pos.size:
                floor = np.nanpercentile(finite_pos, max(0.0, min(50.0, float(getattr(args, "qc_log_floor_pct", 1.0)))))
                floor = max(float(floor), 1e-12)
            else:
                floor = 1e-12
            Y = np.log10(np.maximum(power, floor))
        else:
            raise ValueError("qc_plot_transform must be raw, linear_power, or log_power")

    # Optional light smoothing for display only. This helps the QC figures look
    # less pixelated without changing any analysis values.
    time_smooth = int(getattr(args, "qc_display_smooth_time_bins", 1))
    freq_smooth = int(getattr(args, "qc_display_smooth_freq_bins", 1))
    if time_smooth > 1 or freq_smooth > 1:
        Y = smooth_2d_for_display(Y, time_window=time_smooth, freq_window=freq_smooth)

    return Y


def smooth_2d_for_display(Y: np.ndarray, time_window: int = 1, freq_window: int = 1) -> np.ndarray:
    """Small separable moving-average smoother for display only."""
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()

    def _smooth_axis(arr: np.ndarray, window: int, axis: int) -> np.ndarray:
        window = int(window)
        if window <= 1:
            return arr
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=float) / float(window)
        arr2 = np.moveaxis(arr, axis, 0)
        sm = np.empty_like(arr2, dtype=float)
        for idx in np.ndindex(arr2.shape[1:]):
            y = arr2[(slice(None),) + idx]
            good = np.isfinite(y)
            if not np.any(good):
                sm[(slice(None),) + idx] = np.nan
                continue
            y_fill = np.where(good, y, np.nanmedian(y[good]))
            sm[(slice(None),) + idx] = np.convolve(y_fill, kernel, mode="same")
        return np.moveaxis(sm, 0, axis)

    out = _smooth_axis(out, time_window, axis=0)
    out = _smooth_axis(out, freq_window, axis=1)
    return out


def _transform_qc_spectrogram(X_seg: Optional[np.ndarray], args) -> Optional[np.ndarray]:
    return spectrogram_for_qc_display(X_seg, args)


def _plot_spectrogram_on_axis(
    ax,
    Y: Optional[np.ndarray],
    args,
    title: str,
    seg_start_rel_bin: Optional[int] = None,
    seg_end_rel_bin: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    x_max_s: Optional[float] = None,
) -> None:
    if Y is None or np.asarray(Y).size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    freq_min = args.qc_plot_freq_min_khz if args.qc_plot_freq_min_khz is not None else args.corr_freq_min_khz
    freq_max = args.qc_plot_freq_max_khz if args.qc_plot_freq_max_khz is not None else args.corr_freq_max_khz
    Y = np.asarray(Y, dtype=float)
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = robust_minmax(Y, lo=args.qc_contrast_low_pct, hi=args.qc_contrast_high_pct)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(Y), np.nanmax(Y)
    duration_s = Y.shape[0] * (args.bin_ms / 1000.0)
    extent = [0.0, duration_s, freq_min if freq_min is not None else 0.0, freq_max if freq_max is not None else args.max_freq_khz]
    ax.imshow(
        Y.T,
        origin="lower",
        aspect="auto",
        interpolation="none",
        resample=False,
        cmap=args.qc_plot_cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    # Mark the actual syllable boundaries within the padded display window.
    if seg_start_rel_bin is not None and seg_end_rel_bin is not None:
        x0 = float(seg_start_rel_bin) * (args.bin_ms / 1000.0)
        x1 = float(seg_end_rel_bin) * (args.bin_ms / 1000.0)
        ax.axvline(x0, color="red", linestyle="--", linewidth=args.qc_pair_boundary_lw, alpha=args.qc_pair_boundary_alpha)
        ax.axvline(x1, color="red", linestyle="--", linewidth=args.qc_pair_boundary_lw, alpha=args.qc_pair_boundary_alpha)
    if x_max_s is not None and np.isfinite(x_max_s) and x_max_s > 0:
        ax.set_xlim(0.0, float(x_max_s))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")


def _transform_full_phrase_spectrogram(X_phrase: Optional[np.ndarray], args) -> Optional[np.ndarray]:
    return spectrogram_for_qc_display(X_phrase, args)


def make_phrase_context_qc_outputs(
    qc_rows: pd.DataFrame,
    segment_df: pd.DataFrame,
    X: np.ndarray,
    out_dir: Path,
    prefix: str,
    args,
) -> None:
    """Show the full phrase spectrogram with all segmentation boundaries overlaid.

    Each page corresponds to one selected QC pair and highlights the two segments
    used in that pair, while also showing all segment boundaries in the phrase.
    """
    if qc_rows is None or qc_rows.empty or segment_df.empty:
        return
    required_seg_cols = [
        "phrase_id", "repeat_index_in_phrase", "segment_start_global_bin", "segment_end_global_bin",
        "phrase_global_start_bin", "phrase_global_end_bin", "period", "label"
    ]
    for col in required_seg_cols:
        if col not in segment_df.columns:
            print(f"[WARN] Skipping phrase-context QC because required segment column is missing: {col}")
            return

    out_csv = out_dir / f"{prefix}_phrase_context_qc_selected_pairs.csv"
    out_pdf = out_dir / f"{prefix}_phrase_context_qc.pdf"
    out_png_dir = out_dir / f"{prefix}_phrase_context_qc_pngs"
    ensure_dir(out_png_dir)
    qc_rows.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    freq_min = args.qc_plot_freq_min_khz if args.qc_plot_freq_min_khz is not None else args.corr_freq_min_khz
    freq_max = args.qc_plot_freq_max_khz if args.qc_plot_freq_max_khz is not None else args.corr_freq_max_khz

    # Choose one common x-axis span across all selected phrase-context examples so
    # phrase plots are easier to compare from page to page.
    if args.qc_global_phrase_xmax_s is not None:
        global_phrase_xmax_s = float(args.qc_global_phrase_xmax_s)
    elif getattr(args, "qc_global_phrase_time_axis", True):
        phrase_durations = []
        for _, row in qc_rows.iterrows():
            phrase_id = row.get("phrase_id", np.nan)
            period = row.get("period", "")
            label = row.get("label", "")
            sub = segment_df[(segment_df["phrase_id"] == phrase_id) & (segment_df["period"].astype(str) == str(period)) & (segment_df["label"].astype(str) == str(label))].copy()
            if sub.empty:
                continue
            first = sub.iloc[0]
            g0 = pd.to_numeric(first.get("phrase_global_start_bin"), errors="coerce")
            g1 = pd.to_numeric(first.get("phrase_global_end_bin"), errors="coerce")
            if np.isfinite(g0) and np.isfinite(g1) and g1 > g0:
                phrase_durations.append((float(g1) - float(g0)) * (args.bin_ms / 1000.0))
        global_phrase_xmax_s = max(phrase_durations) if phrase_durations else None
    else:
        global_phrase_xmax_s = None

    with PdfPages(out_pdf) as pdf:
        for _, row in qc_rows.iterrows():
            phrase_id = row.get("phrase_id", np.nan)
            period = row.get("period", "")
            label = row.get("label", "")
            sub = segment_df[(segment_df["phrase_id"] == phrase_id) & (segment_df["period"].astype(str) == str(period)) & (segment_df["label"].astype(str) == str(label))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("repeat_index_in_phrase").reset_index(drop=True)
            first = sub.iloc[0]
            g0 = int(first["phrase_global_start_bin"])
            g1 = int(first["phrase_global_end_bin"])
            X_phrase = extract_global_segment_spectrogram(X, g0, g1, pad_bins=0)[0]
            Y = _transform_full_phrase_spectrogram(X_phrase, args)
            if Y is None:
                continue

            y_arr = np.asarray(Y, dtype=float)
            valid = y_arr[np.isfinite(y_arr)]
            if valid.size:
                vmin, vmax = np.percentile(valid, [args.qc_contrast_low_pct, args.qc_contrast_high_pct])
            else:
                vmin = vmax = None

            phrase_duration_s = (g1 - g0) * (args.bin_ms / 1000.0)
            fig, ax = plt.subplots(1, 1, figsize=(13, 4.8), constrained_layout=True)
            ax.imshow(
                y_arr.T,
                origin="lower",
                aspect="auto",
                interpolation="none",
                resample=False,
                cmap=args.qc_plot_cmap,
                extent=[0.0, phrase_duration_s, freq_min if freq_min is not None else 0.0, freq_max if freq_max is not None else args.max_freq_khz],
                vmin=vmin,
                vmax=vmax,
            )

            rep_a = pd.to_numeric(row.get("pair_start_repeat_index"), errors="coerce")
            rep_b = pd.to_numeric(row.get("pair_end_repeat_index"), errors="coerce")
            highlight_repeats = set()
            if np.isfinite(rep_a): highlight_repeats.add(int(rep_a))
            if np.isfinite(rep_b): highlight_repeats.add(int(rep_b))

            # overlay all segment boundaries and labels
            y_top = (freq_max if freq_max is not None else args.max_freq_khz)
            for _, srow in sub.iterrows():
                s0 = pd.to_numeric(srow.get("segment_start_global_bin"), errors="coerce")
                s1 = pd.to_numeric(srow.get("segment_end_global_bin"), errors="coerce")
                rep = pd.to_numeric(srow.get("repeat_index_in_phrase"), errors="coerce")
                if not (np.isfinite(s0) and np.isfinite(s1) and np.isfinite(rep)):
                    continue
                x0 = (float(s0) - g0) * (args.bin_ms / 1000.0)
                x1 = (float(s1) - g0) * (args.bin_ms / 1000.0)
                rep_i = int(rep)
                if rep_i in highlight_repeats:
                    if args.qc_highlight_span_alpha > 0:
                        ax.axvspan(x0, x1, color='tab:red', alpha=args.qc_highlight_span_alpha, lw=0)
                    ax.axvline(x0, color='tab:red', linestyle='--', linewidth=args.qc_highlight_boundary_lw, alpha=args.qc_highlight_boundary_alpha)
                    ax.axvline(x1, color='tab:red', linestyle='--', linewidth=args.qc_highlight_boundary_lw, alpha=args.qc_highlight_boundary_alpha)
                    txt_color = 'tab:red'
                    fw = 'bold'
                else:
                    ax.axvline(x0, color='dodgerblue', linestyle='--', linewidth=args.qc_context_boundary_lw, alpha=args.qc_context_boundary_alpha)
                    ax.axvline(x1, color='dodgerblue', linestyle='--', linewidth=args.qc_context_boundary_lw, alpha=args.qc_context_boundary_alpha)
                    txt_color = 'dodgerblue'
                    fw = 'normal'
                xmid = 0.5 * (x0 + x1)
                ax.text(xmid, y_top - 0.12 * (y_top - (freq_min if freq_min is not None else 0.0)), str(rep_i),
                        ha='center', va='bottom', fontsize=8, color=txt_color, fontweight=fw,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=args.qc_label_box_alpha, pad=0.3))

            corr_val = pd.to_numeric(row.get("shortlag_pair_corr"), errors="coerce")
            d_val = pd.to_numeric(row.get("shortlag_pair_spectral_displacement"), errors="coerce")
            title = f"{row.get('animal_id', '')} label {label} | {period} | phrase {phrase_id}: full phrase context"
            subtitle_bits = []
            if np.isfinite(rep_a) and np.isfinite(rep_b):
                subtitle_bits.append(f"highlighted pair repeats {int(rep_a)} and {int(rep_b)}")
            if np.isfinite(corr_val):
                subtitle_bits.append(f"corr={corr_val:.3f}")
            if np.isfinite(d_val):
                subtitle_bits.append(f"d={d_val:.3f}")
            subtitle_bits.append(f"n segments={len(sub)}")
            ax.set_title(title + "\n" + " | ".join(subtitle_bits), fontsize=12)
            ax.set_xlabel("Time within phrase (s)")
            ax.set_ylabel("Frequency (kHz)")
            if global_phrase_xmax_s is not None and np.isfinite(global_phrase_xmax_s) and global_phrase_xmax_s > 0:
                ax.set_xlim(0.0, float(global_phrase_xmax_s))
            else:
                ax.set_xlim(0.0, phrase_duration_s)
            pdf.savefig(fig, dpi=args.qc_plot_dpi)
            png_path = out_png_dir / f"phrase_{int(phrase_id) if pd.notna(phrase_id) else 0}_pair_{int(row['qc_plot_index']) if 'qc_plot_index' in row else 0}.png"
            fig.savefig(png_path, dpi=args.qc_plot_dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"[SAVED] {png_path}")
    print(f"[SAVED] {out_pdf}")



def _get_segment_rows_for_qc_phrase(
    segment_df: pd.DataFrame,
    phrase_id,
    period,
    label,
) -> pd.DataFrame:
    """Robustly retrieve segment rows for one selected QC phrase."""
    if segment_df.empty:
        return segment_df.iloc[0:0].copy()
    work = segment_df.copy()
    mask = work["phrase_id"].astype(str) == str(phrase_id)
    if "period" in work.columns:
        mask &= work["period"].astype(str) == str(period)

    # Prefer normalized labels if present. This avoids 19 vs 19.0 mismatches.
    if "label_norm" in work.columns:
        lab = normalize_label_for_qc(label)
        mask &= work["label_norm"].astype(str).map(normalize_label_for_qc) == lab
    elif "label" in work.columns:
        lab = normalize_label_for_qc(label)
        mask &= work["label"].astype(str).map(normalize_label_for_qc) == lab

    out = work[mask].copy()
    if "repeat_index_in_phrase" in out.columns:
        out["repeat_index_in_phrase"] = pd.to_numeric(out["repeat_index_in_phrase"], errors="coerce")
        out = out.sort_values("repeat_index_in_phrase")
    return out.reset_index(drop=True)


def normalize_label_for_qc(x) -> str:
    """Normalize labels for plotting lookups, e.g. 19.0 -> 19."""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def _expanded_context_records(
    qc_rows: pd.DataFrame,
    segment_df: pd.DataFrame,
    X: np.ndarray,
    args,
) -> List[Dict[str, object]]:
    """Build metadata for expanded phrase-context rows."""
    records: List[Dict[str, object]] = []
    before_bins = int(round(max(0.0, float(args.qc_context_before_s)) / (args.bin_ms / 1000.0)))
    after_bins = int(round(max(0.0, float(args.qc_context_after_s)) / (args.bin_ms / 1000.0)))
    fixed_window_bins = None
    if args.qc_expanded_context_window_s is not None and float(args.qc_expanded_context_window_s) > 0:
        fixed_window_bins = int(round(float(args.qc_expanded_context_window_s) / (args.bin_ms / 1000.0)))

    for _, row in qc_rows.iterrows():
        phrase_id = row.get("phrase_id", np.nan)
        period = row.get("period", "")
        label = row.get("label", "")
        sub = _get_segment_rows_for_qc_phrase(segment_df, phrase_id=phrase_id, period=period, label=label)
        if sub.empty:
            continue

        first = sub.iloc[0]
        try:
            phrase_g0 = int(pd.to_numeric(first.get("phrase_global_start_bin"), errors="coerce"))
            phrase_g1 = int(pd.to_numeric(first.get("phrase_global_end_bin"), errors="coerce"))
        except Exception:
            continue
        if phrase_g1 <= phrase_g0:
            continue

        if fixed_window_bins is not None:
            center = int(round(0.5 * (phrase_g0 + phrase_g1)))
            context_g0 = center - fixed_window_bins // 2
            context_g1 = context_g0 + fixed_window_bins
        else:
            context_g0 = phrase_g0 - before_bins
            context_g1 = phrase_g1 + after_bins

        context_g0 = max(0, int(context_g0))
        context_g1 = min(int(X.shape[0]), int(context_g1))
        if context_g1 <= context_g0:
            continue

        records.append({
            "qc_plot_index": row.get("qc_plot_index", len(records) + 1),
            "animal_id": row.get("animal_id", ""),
            "label": label,
            "period": period,
            "phrase_id": phrase_id,
            "sub": sub,
            "phrase_g0": phrase_g0,
            "phrase_g1": phrase_g1,
            "context_g0": context_g0,
            "context_g1": context_g1,
            "duration_s": (context_g1 - context_g0) * (args.bin_ms / 1000.0),
            "pair_start_repeat_index": row.get("pair_start_repeat_index", np.nan),
            "pair_end_repeat_index": row.get("pair_end_repeat_index", np.nan),
            "shortlag_pair_corr": row.get("shortlag_pair_corr", np.nan),
            "shortlag_pair_spectral_displacement": row.get("shortlag_pair_spectral_displacement", np.nan),
        })
    return records


def _draw_segment_boundaries_on_expanded_context(
    ax,
    record: Dict[str, object],
    args,
    freq_min: float,
    freq_max: float,
) -> None:
    """Overlay target-phrase segment boundaries on an expanded context row."""
    sub = record["sub"]
    context_g0 = int(record["context_g0"])
    rep_a = pd.to_numeric(record.get("pair_start_repeat_index"), errors="coerce")
    rep_b = pd.to_numeric(record.get("pair_end_repeat_index"), errors="coerce")
    highlight_repeats = set()
    if np.isfinite(rep_a):
        highlight_repeats.add(int(rep_a))
    if np.isfinite(rep_b):
        highlight_repeats.add(int(rep_b))

    y_top = float(freq_max)
    y_bottom = float(freq_min)
    label_y = y_top - 0.12 * (y_top - y_bottom)

    for _, srow in sub.iterrows():
        s0 = pd.to_numeric(srow.get("segment_start_global_bin"), errors="coerce")
        s1 = pd.to_numeric(srow.get("segment_end_global_bin"), errors="coerce")
        rep = pd.to_numeric(srow.get("repeat_index_in_phrase"), errors="coerce")
        if not (np.isfinite(s0) and np.isfinite(s1) and np.isfinite(rep)):
            continue
        x0 = (float(s0) - context_g0) * (args.bin_ms / 1000.0)
        x1 = (float(s1) - context_g0) * (args.bin_ms / 1000.0)
        rep_i = int(rep)

        if rep_i in highlight_repeats:
            if args.qc_highlight_span_alpha > 0:
                ax.axvspan(x0, x1, color="tab:red", alpha=args.qc_highlight_span_alpha, lw=0)
            ax.axvline(
                x0,
                color="tab:red",
                linestyle="--",
                linewidth=args.qc_highlight_boundary_lw,
                alpha=args.qc_highlight_boundary_alpha,
            )
            ax.axvline(
                x1,
                color="tab:red",
                linestyle="--",
                linewidth=args.qc_highlight_boundary_lw,
                alpha=args.qc_highlight_boundary_alpha,
            )
            txt_color = "tab:red"
            fw = "bold"
        elif bool(args.qc_expanded_show_all_boundaries):
            ax.axvline(
                x0,
                color="dodgerblue",
                linestyle="--",
                linewidth=args.qc_context_boundary_lw,
                alpha=args.qc_context_boundary_alpha,
            )
            ax.axvline(
                x1,
                color="dodgerblue",
                linestyle="--",
                linewidth=args.qc_context_boundary_lw,
                alpha=args.qc_context_boundary_alpha,
            )
            txt_color = "dodgerblue"
            fw = "normal"
        else:
            continue

        xmid = 0.5 * (x0 + x1)
        ax.text(
            xmid,
            label_y,
            str(rep_i),
            ha="center",
            va="bottom",
            fontsize=args.qc_expanded_label_fontsize,
            color=txt_color,
            fontweight=fw,
            bbox=dict(facecolor="white", edgecolor="none", alpha=args.qc_label_box_alpha, pad=0.25),
        )


def make_expanded_phrase_context_qc_outputs(
    qc_rows: pd.DataFrame,
    segment_df: pd.DataFrame,
    X: np.ndarray,
    out_dir: Path,
    prefix: str,
    args,
) -> None:
    """Make row-style expanded context spectrograms similar to expanded-full-run QC.

    Unlike the ordinary phrase-context QC, these pages extract additional time
    before and after each selected phrase, then stack multiple examples as long
    horizontal rows. This is intended to visually resemble the expanded full-run
    spectrogram examples used in the BC QC pipeline while still highlighting the
    phrase/pair used for local-rate QC.
    """
    if not bool(getattr(args, "qc_expanded_context", True)):
        return
    if qc_rows is None or qc_rows.empty or segment_df.empty:
        return

    records = _expanded_context_records(qc_rows, segment_df, X, args)
    if not records:
        print("[WARN] No expanded phrase-context records could be built.")
        return

    out_pdf = out_dir / f"{prefix}_expanded_phrase_context_qc.pdf"
    out_png_dir = out_dir / f"{prefix}_expanded_phrase_context_qc_pngs"
    out_csv = out_dir / f"{prefix}_expanded_phrase_context_qc_selected_pairs.csv"
    ensure_dir(out_png_dir)

    # Save a lightweight table without embedded dataframes.
    rows = []
    for r in records:
        rows.append({
            "qc_plot_index": r["qc_plot_index"],
            "animal_id": r["animal_id"],
            "label": r["label"],
            "period": r["period"],
            "phrase_id": r["phrase_id"],
            "phrase_global_start_bin": r["phrase_g0"],
            "phrase_global_end_bin": r["phrase_g1"],
            "expanded_context_start_bin": r["context_g0"],
            "expanded_context_end_bin": r["context_g1"],
            "expanded_context_duration_s": r["duration_s"],
            "pair_start_repeat_index": r["pair_start_repeat_index"],
            "pair_end_repeat_index": r["pair_end_repeat_index"],
            "shortlag_pair_corr": r["shortlag_pair_corr"],
            "shortlag_pair_spectral_displacement": r["shortlag_pair_spectral_displacement"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    freq_min = args.qc_plot_freq_min_khz if args.qc_plot_freq_min_khz is not None else args.corr_freq_min_khz
    freq_max = args.qc_plot_freq_max_khz if args.qc_plot_freq_max_khz is not None else args.corr_freq_max_khz
    if freq_min is None:
        freq_min = 0.0
    if freq_max is None:
        freq_max = args.max_freq_khz

    if args.qc_expanded_context_xmax_s is not None:
        x_max_s = float(args.qc_expanded_context_xmax_s)
    elif bool(getattr(args, "qc_expanded_context_global_time_axis", True)):
        x_max_s = max(float(r["duration_s"]) for r in records)
    else:
        x_max_s = None

    rows_per_page = max(1, int(args.qc_expanded_context_rows_per_page))
    n_pages = int(np.ceil(len(records) / rows_per_page))

    with PdfPages(out_pdf) as pdf:
        for page_i in range(n_pages):
            page_records = records[page_i * rows_per_page : (page_i + 1) * rows_per_page]
            nrows = len(page_records)
            fig_h = max(2.0, float(args.qc_expanded_context_row_height) * nrows + 1.0)
            fig, axes = plt.subplots(
                nrows,
                1,
                figsize=(float(args.qc_expanded_context_fig_width), fig_h),
                squeeze=False,
                constrained_layout=True,
            )
            axes = axes.ravel()

            for ax, rec in zip(axes, page_records):
                context_g0 = int(rec["context_g0"])
                context_g1 = int(rec["context_g1"])
                X_context = X[context_g0:context_g1, :]
                Y = spectrogram_for_qc_display(X_context, args)
                if Y is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                y_arr = np.asarray(Y, dtype=float)
                valid = y_arr[np.isfinite(y_arr)]
                if valid.size:
                    vmin, vmax = np.percentile(valid, [args.qc_contrast_low_pct, args.qc_contrast_high_pct])
                else:
                    vmin = vmax = None

                dur_s = y_arr.shape[0] * (args.bin_ms / 1000.0)
                ax.imshow(
                    y_arr.T,
                    origin="lower",
                    aspect="auto",
                    interpolation="none",
                    resample=False,
                    cmap=args.qc_plot_cmap,
                    extent=[0.0, dur_s, float(freq_min), float(freq_max)],
                    vmin=vmin,
                    vmax=vmax,
                )

                _draw_segment_boundaries_on_expanded_context(ax, rec, args, float(freq_min), float(freq_max))

                # Mark the full target phrase span lightly, independent of the selected pair.
                phrase_x0 = (int(rec["phrase_g0"]) - context_g0) * (args.bin_ms / 1000.0)
                phrase_x1 = (int(rec["phrase_g1"]) - context_g0) * (args.bin_ms / 1000.0)
                if args.qc_phrase_span_alpha > 0:
                    ax.axvspan(phrase_x0, phrase_x1, color="gray", alpha=args.qc_phrase_span_alpha, lw=0)

                corr = pd.to_numeric(rec.get("shortlag_pair_corr"), errors="coerce")
                dval = pd.to_numeric(rec.get("shortlag_pair_spectral_displacement"), errors="coerce")
                rep_a = pd.to_numeric(rec.get("pair_start_repeat_index"), errors="coerce")
                rep_b = pd.to_numeric(rec.get("pair_end_repeat_index"), errors="coerce")
                bits = [
                    f"{rec.get('period', '')}",
                    f"phrase {rec.get('phrase_id', '')}",
                    f"expanded context {float(rec['duration_s']):.2f} s",
                ]
                if np.isfinite(rep_a) and np.isfinite(rep_b):
                    bits.append(f"pair repeats {int(rep_a)}-{int(rep_b)}")
                if np.isfinite(corr):
                    bits.append(f"corr={corr:.3f}")
                if np.isfinite(dval):
                    bits.append(f"d={dval:.3f}")
                ax.set_ylabel("Freq. (kHz)")
                ax.text(
                    -0.012,
                    0.5,
                    " | ".join(bits),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
                if x_max_s is not None and np.isfinite(x_max_s) and x_max_s > 0:
                    ax.set_xlim(0, x_max_s)
                else:
                    ax.set_xlim(0, dur_s)

            axes[-1].set_xlabel("Time in expanded context (s)")
            fig.suptitle(
                f"{prefix}: expanded phrase-context QC ({page_i + 1}/{n_pages})",
                fontsize=12,
                y=1.02,
            )
            pdf.savefig(fig, dpi=args.qc_plot_dpi)
            png_path = out_png_dir / f"expanded_context_page_{page_i + 1:03d}.png"
            fig.savefig(png_path, dpi=args.qc_plot_dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVED] {png_path}")
    print(f"[SAVED] {out_pdf}")


def make_local_pair_qc_outputs(pair_df: pd.DataFrame, segment_df: pd.DataFrame, X: np.ndarray, out_dir: Path, prefix: str, args) -> None:
    """Generate a small QC set of paired-syllable spectrogram figures."""
    n_qc = int(getattr(args, "qc_local_pair_count", 0))
    if pair_df.empty or n_qc <= 0:
        return
    required_cols = [
        "pair_start_segment_global_start_bin",
        "pair_start_segment_global_end_bin",
        "pair_end_segment_global_start_bin",
        "pair_end_segment_global_end_bin",
        "shortlag_pair_corr",
    ]
    for col in required_cols:
        if col not in pair_df.columns:
            print(f"[WARN] Skipping local pair QC plots because required column is missing: {col}")
            return
    qc_rows = select_local_pair_qc_rows(
        pair_df,
        n_total=n_qc,
        sample_mode=getattr(args, "qc_local_pair_sample_mode", "quantile_corr"),
        pairs_per_period=int(getattr(args, "qc_local_pairs_per_period", 0)),
        random_seed=int(getattr(args, "qc_random_seed", 0)),
    )
    if qc_rows.empty:
        print("[WARN] No local pair QC rows selected.")
        return

    qc_csv = out_dir / f"{prefix}_local_shortlag_pair_qc_selected.csv"
    qc_pdf = out_dir / f"{prefix}_local_shortlag_pair_qc.pdf"
    qc_png_dir = out_dir / f"{prefix}_local_shortlag_pair_qc_pngs"
    ensure_dir(qc_png_dir)
    qc_rows = qc_rows.reset_index(drop=True).copy()
    qc_rows.insert(0, "qc_plot_index", np.arange(1, len(qc_rows) + 1))
    qc_rows.to_csv(qc_csv, index=False)
    print(f"[SAVED] {qc_csv}")

    # Also save full phrase-context figures with all segmentation boundaries overlaid.
    make_phrase_context_qc_outputs(qc_rows, segment_df, X, out_dir=out_dir, prefix=prefix, args=args)
    # Also save larger, row-style context panels resembling expanded full-run QC spectrograms.
    make_expanded_phrase_context_qc_outputs(qc_rows, segment_df, X, out_dir=out_dir, prefix=prefix, args=args)

    pad_bins = int(round(max(0.0, float(args.qc_pad_ms)) / float(args.bin_ms)))

    # Choose one common x-axis span across all isolated pair-QC figures so pages are
    # easier to compare across different phrases/contexts.
    if args.qc_global_pair_xmax_s is not None:
        global_pair_xmax_s = float(args.qc_global_pair_xmax_s)
    elif getattr(args, "qc_global_pair_time_axis", True):
        pair_durations = []
        for _, row in qc_rows.iterrows():
            vals = []
            for prefix_col in ["pair_start_segment_global", "pair_end_segment_global"]:
                s0 = pd.to_numeric(row.get(f"{prefix_col}_start_bin"), errors="coerce")
                s1 = pd.to_numeric(row.get(f"{prefix_col}_end_bin"), errors="coerce")
                if np.isfinite(s0) and np.isfinite(s1) and s1 > s0:
                    dur_s = (float(s1) - float(s0) + 2.0 * pad_bins) * (args.bin_ms / 1000.0)
                    vals.append(dur_s)
            if vals:
                pair_durations.append(max(vals))
        global_pair_xmax_s = max(pair_durations) if pair_durations else None
    else:
        global_pair_xmax_s = None

    with PdfPages(qc_pdf) as pdf:
        for _, row in qc_rows.iterrows():
            X_a, disp0_a, disp1_a, rel0_a, rel1_a = extract_global_segment_spectrogram(
                X,
                row.get("pair_start_segment_global_start_bin"),
                row.get("pair_start_segment_global_end_bin"),
                pad_bins=pad_bins,
            )
            X_b, disp0_b, disp1_b, rel0_b, rel1_b = extract_global_segment_spectrogram(
                X,
                row.get("pair_end_segment_global_start_bin"),
                row.get("pair_end_segment_global_end_bin"),
                pad_bins=pad_bins,
            )
            Y_a = _transform_qc_spectrogram(X_a, args)
            Y_b = _transform_qc_spectrogram(X_b, args)

            # Use shared contrast across the pair so the two panels are easier to compare.
            if Y_a is not None and Y_b is not None and getattr(args, "qc_pair_shared_contrast", True):
                combo = np.concatenate([np.asarray(Y_a, dtype=float).ravel(), np.asarray(Y_b, dtype=float).ravel()])
                combo = combo[np.isfinite(combo)]
                if combo.size:
                    vmin, vmax = np.percentile(combo, [args.qc_contrast_low_pct, args.qc_contrast_high_pct])
                else:
                    vmin, vmax = None, None
            else:
                vmin = vmax = None

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), constrained_layout=True)
            title_a = f"Repeat {int(row.get('pair_start_repeat_index', np.nan))}" if np.isfinite(pd.to_numeric(row.get('pair_start_repeat_index'), errors='coerce')) else "First syllable"
            title_b = f"Repeat {int(row.get('pair_end_repeat_index', np.nan))}" if np.isfinite(pd.to_numeric(row.get('pair_end_repeat_index'), errors='coerce')) else "Second syllable"
            x_max_pair = None
            if global_pair_xmax_s is not None and np.isfinite(global_pair_xmax_s) and global_pair_xmax_s > 0:
                x_max_pair = float(global_pair_xmax_s)
            elif getattr(args, "qc_match_pair_time_axis", True):
                dur_a = (np.asarray(Y_a).shape[0] * (args.bin_ms / 1000.0)) if Y_a is not None else np.nan
                dur_b = (np.asarray(Y_b).shape[0] * (args.bin_ms / 1000.0)) if Y_b is not None else np.nan
                vals = [x for x in [dur_a, dur_b] if np.isfinite(x)]
                if vals:
                    x_max_pair = max(vals)
            _plot_spectrogram_on_axis(axes[0], Y_a, args, title=title_a, seg_start_rel_bin=rel0_a, seg_end_rel_bin=rel1_a, vmin=vmin, vmax=vmax, x_max_s=x_max_pair)
            _plot_spectrogram_on_axis(axes[1], Y_b, args, title=title_b, seg_start_rel_bin=rel0_b, seg_end_rel_bin=rel1_b, vmin=vmin, vmax=vmax, x_max_s=x_max_pair)

            corr_val = pd.to_numeric(row.get("shortlag_pair_corr"), errors="coerce")
            d_val = pd.to_numeric(row.get("shortlag_pair_spectral_displacement"), errors="coerce")
            lag_val = pd.to_numeric(row.get("pair_lag_repeats"), errors="coerce")
            phrase_id = row.get("phrase_id", np.nan)
            period = row.get("period", "")
            label = row.get("label", "")
            animal_id = row.get('animal_id', '')
            subtitle = f"{animal_id} label {label} | {period} | phrase {phrase_id}"
            if np.isfinite(lag_val):
                subtitle += f" | lag={int(lag_val)}"
            if np.isfinite(corr_val):
                subtitle += f" | corr={corr_val:.3f}"
            if np.isfinite(d_val):
                subtitle += f" | d={d_val:.3f}"
            dur_a_core = pd.to_numeric(row.get("pair_start_segment_global_end_bin"), errors="coerce") - pd.to_numeric(row.get("pair_start_segment_global_start_bin"), errors="coerce")
            dur_b_core = pd.to_numeric(row.get("pair_end_segment_global_end_bin"), errors="coerce") - pd.to_numeric(row.get("pair_end_segment_global_start_bin"), errors="coerce")
            if np.isfinite(dur_a_core) and np.isfinite(dur_b_core):
                subtitle += f" | dur={dur_a_core * args.bin_ms:.0f}/{dur_b_core * args.bin_ms:.0f} ms"
            if pad_bins > 0:
                subtitle += f" | ±{args.qc_pad_ms:.0f} ms context"
            fig.suptitle(subtitle, fontsize=12)
            pdf.savefig(fig, dpi=args.qc_plot_dpi)
            png_path = qc_png_dir / f"pair_{int(row['qc_plot_index']):03d}_phrase{int(phrase_id) if pd.notna(phrase_id) else 0}_lag{int(lag_val) if np.isfinite(lag_val) else 0}.png"
            fig.savefig(png_path, dpi=args.qc_plot_dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVED] {png_path}")
    print(f"[SAVED] {qc_pdf}")


def prepare_segment_for_crosscorr(X_seg: np.ndarray, args) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return time-normalized spectrogram and unit-vector representation for one segment."""
    Xc = crop_frequency_band(
        X_seg,
        max_freq_khz=args.max_freq_khz,
        freq_min_khz=args.corr_freq_min_khz,
        freq_max_khz=args.corr_freq_max_khz,
    )
    Y = transform_spectrogram_for_corr(
        Xc,
        spec_scale=args.spec_scale,
        corr_transform=args.corr_transform,
    )
    Yn = time_normalize_spectrogram(Y, n_time=args.corr_time_bins)
    vec = vectorize_for_corr(Yn)
    return Yn, vec


def build_corr_template(norm_mats: List[np.ndarray], min_template_segments=2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    usable = [m for m in norm_mats if m is not None and np.any(np.isfinite(m))]
    if len(usable) < int(min_template_segments):
        return None, None
    template = np.nanmean(np.stack(usable, axis=0), axis=0)
    return template, vectorize_for_corr(template)


def choose_phrase_template_indices(seg_infos: List[Dict[str, object]], args) -> List[int]:
    """Choose early segments used to build an early-phrase template."""
    if len(seg_infos) == 0:
        return []
    mode = args.phrase_template_mode
    if mode == "first_n":
        return list(range(min(int(args.phrase_template_first_n), len(seg_infos))))
    if mode == "first_seconds":
        idx = []
        for i, info in enumerate(seg_infos):
            if float(info.get("elapsed_mid", np.nan)) <= float(args.phrase_template_first_s):
                idx.append(i)
        if len(idx) < int(args.min_phrase_template_segments):
            idx = list(range(min(int(args.phrase_template_first_n), len(seg_infos))))
        return idx
    raise ValueError("phrase_template_mode must be first_n or first_seconds")


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def fill_small_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or mask.size == 0:
        return mask
    false_runs = finite_runs(~mask)
    for s, e in false_runs:
        if s == 0 or e == len(mask):
            continue
        if (e - s) <= max_gap:
            mask[s:e] = True
    return mask


def centered_fraction_true(mask: np.ndarray, window_bins: int) -> np.ndarray:
    """Centered rolling fraction of True values. Edges use shorter denominators."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    if n == 0:
        return np.asarray([], dtype=float)
    w = int(window_bins)
    if w <= 1:
        return mask.astype(float)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float)
    numer = np.convolve(mask.astype(float), kernel, mode="same")
    denom = np.convolve(np.ones(n, dtype=float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = numer / denom
    return frac


def make_target_phrase_mask(
    local_labels: np.ndarray,
    target_label: str,
    smoothing: str = "none",
    majority_vote_window_bins: int = 1,
    majority_vote_threshold: float = 0.5,
    majority_vote_keep_raw_target_bins: bool = False,
) -> np.ndarray:
    """Return target-label mask for phrase-run detection.

    smoothing="target_majority_vote" performs a target-vs-other majority vote:
    a bin is included in the target phrase if at least `majority_vote_threshold`
    of the centered window is the target label. This is useful when a true
    repeated phrase is interrupted by brief decoder label glitches.
    """
    local_labels = normalize_label_array(local_labels)
    target_label = normalize_label_value(target_label)
    raw_mask = local_labels == target_label
    if smoothing in (None, "none"):
        return raw_mask
    if smoothing != "target_majority_vote":
        raise ValueError("phrase_label_smoothing must be 'none' or 'target_majority_vote'.")

    frac = centered_fraction_true(raw_mask, majority_vote_window_bins)
    smoothed = frac >= float(majority_vote_threshold)
    if majority_vote_keep_raw_target_bins:
        smoothed = smoothed | raw_mask
    return smoothed


def interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = x.copy()
    good = np.isfinite(out)
    if not np.any(good):
        return out
    if np.sum(good) == 1:
        out[~good] = out[good][0]
        return out
    idx = np.arange(len(out))
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def canary_dominant_freq_with_spectrum(sig: np.ndarray, fs: float, band=(2.0, 20.0)):
    sig = interp_nan_1d(np.asarray(sig, dtype=float))
    if len(sig) < 4 or not np.any(np.isfinite(sig)):
        return np.nan, np.array([], dtype=float), np.array([], dtype=float)
    sig = sig - np.nanmean(sig)
    if not np.isfinite(np.nanstd(sig)) or np.nanstd(sig) <= 0:
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        return np.nan, freqs, np.zeros_like(freqs)
    win = np.hanning(len(sig))
    if np.all(win == 0):
        win = np.ones(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan, freqs, spec
    valid = np.flatnonzero(mask)
    f0 = float(freqs[valid[np.argmax(spec[mask])]])
    return f0, freqs, spec


def canary_bandpass_power(power: np.ndarray, fs: float, f0: float, frac=0.5, pad_periods=3.0) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("Canary segmentation requires scipy. Install with: conda install scipy")
    power = interp_nan_1d(np.asarray(power, dtype=float))
    if len(power) < 6 or not np.isfinite(f0) or f0 <= 0:
        return np.full_like(power, np.nan, dtype=float)
    lo = max(f0 * (1.0 - frac), 0.1)
    hi = min(f0 * (1.0 + frac), fs / 2.0 * 0.99)
    if hi <= lo:
        return np.full_like(power, np.nan, dtype=float)
    b, a = butter(2, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    npad = min(int(pad_periods * fs / f0), len(power) - 1)
    if npad < 1:
        return filtfilt(b, a, power, padtype=None)
    padded = np.pad(power, npad, mode="reflect")
    filt = filtfilt(b, a, padded, padtype=None)
    return filt[npad: npad + len(power)]


def segment_power_by_canary_method(
        pitch_khz: np.ndarray,
        total_power: np.ndarray,
        bin_s: float,
        rate_band=(2.0, 20.0),
        frac=0.5,
        edge_guard_periods=0.5,
        pad_periods=3.0,
        snap=True,
        snap_periods=0.5,
        min_points=12,
        drop_edge_segments=True,
        min_segment_bins=3) -> Dict[str, object]:
    if not SCIPY_AVAILABLE:
        raise ImportError("Canary segmentation requires scipy. Install with: conda install scipy")
    pitch_khz = np.asarray(pitch_khz, dtype=float)
    total_power = np.asarray(total_power, dtype=float)
    n = len(pitch_khz)
    fs = 1.0 / bin_s

    bandpassed_power = np.full(n, np.nan, dtype=float)
    boundary_mask = np.zeros(n, dtype=bool)
    presnap_mask = np.zeros(n, dtype=bool)
    segment_ids = np.full(n, -1, dtype=int)
    segments = []
    diagnostics = []
    valid_for_chunks = np.isfinite(pitch_khz) & np.isfinite(total_power)
    chunks = finite_runs(valid_for_chunks)
    global_segment_id = 0

    for chunk_start, chunk_end in chunks:
        if (chunk_end - chunk_start) < min_points:
            continue
        pitch_chunk = interp_nan_1d(pitch_khz[chunk_start:chunk_end])
        power_chunk = interp_nan_1d(total_power[chunk_start:chunk_end])
        if np.sum(np.isfinite(pitch_chunk)) < min_points or np.sum(np.isfinite(power_chunk)) < min_points:
            continue
        f0, freqs, spec = canary_dominant_freq_with_spectrum(pitch_chunk, fs=fs, band=rate_band)
        if not np.isfinite(f0) or f0 <= 0:
            continue
        filt = canary_bandpass_power(power_chunk, fs=fs, f0=f0, frac=frac, pad_periods=pad_periods)
        bandpassed_power[chunk_start:chunk_end] = filt
        if not np.any(np.isfinite(filt)):
            continue
        min_dist = max(1, int(0.6 * fs / f0))
        presnap_rel, _ = find_peaks(-filt, distance=min_dist)
        guard = int(edge_guard_periods * fs / f0)
        presnap_rel = presnap_rel[(presnap_rel >= guard) & (presnap_rel < len(filt) - guard)]
        presnap_abs = presnap_rel + chunk_start
        presnap_mask[presnap_abs] = True
        if snap and len(presnap_rel) > 0:
            half = max(1, int(snap_periods * fs / f0))
            snapped_rel = []
            for bi in presnap_rel:
                a = max(0, int(bi) - half)
                b = min(len(power_chunk), int(bi) + half + 1)
                if b <= a:
                    continue
                snapped_rel.append(a + int(np.nanargmin(power_chunk[a:b])))
            boundary_rel = np.unique(np.asarray(snapped_rel, dtype=int))
        else:
            boundary_rel = np.unique(presnap_rel.astype(int))
        boundary_rel = boundary_rel[(boundary_rel > 0) & (boundary_rel < len(power_chunk) - 1)]
        boundary_abs = boundary_rel + chunk_start
        boundary_mask[boundary_abs] = True

        segment_pairs = []
        for k in range(len(boundary_abs) - 1):
            seg_start = int(boundary_abs[k])
            seg_end = int(boundary_abs[k + 1])
            if seg_end <= seg_start or (seg_end - seg_start) < min_segment_bins:
                continue
            segment_pairs.append((seg_start, seg_end))

        if drop_edge_segments and len(segment_pairs) > 2:
            kept_pairs = segment_pairs[1:-1]
        else:
            kept_pairs = segment_pairs

        for seg_start, seg_end in kept_pairs:
            segment_ids[seg_start:seg_end] = global_segment_id
            segments.append({
                "segment_id": global_segment_id,
                "chunk_start_bin": int(chunk_start),
                "chunk_end_bin": int(chunk_end),
                "segment_start_bin": seg_start,
                "segment_end_bin": seg_end,
                "segment_start_time_s": seg_start * bin_s,
                "segment_end_time_s": seg_end * bin_s,
                "segment_duration_s": (seg_end - seg_start) * bin_s,
                "dominant_pitch_modulation_freq_hz": f0,
                "bandpass_frac": frac,
                "min_distance_bins": int(min_dist),
                "n_boundary_minima_in_chunk": int(len(boundary_abs)),
                "segmentation_method": "canary_power_bandpass_snap" if snap else "canary_power_bandpass_no_snap",
                "dropped_edge_segments": bool(drop_edge_segments and len(segment_pairs) > 2),
                "n_raw_segment_pairs_in_chunk": int(len(segment_pairs)),
            })
            global_segment_id += 1
        diagnostics.append({
            "chunk_start_bin": int(chunk_start),
            "chunk_end_bin": int(chunk_end),
            "f0_hz": f0,
            "period_ms": 1000.0 / f0,
            "n_boundaries": int(len(boundary_abs)),
        })

    return {
        "bandpassed_power": bandpassed_power,
        "presnap_mask": presnap_mask,
        "boundary_mask": boundary_mask,
        "segment_ids": segment_ids,
        "segments": segments,
        "diagnostics": diagnostics,
    }


def robust_minmax(arr: np.ndarray, lo=2, hi=98) -> Tuple[float, float]:
    x = clean_values(arr)
    if x.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a, b = np.nanmin(x), np.nanmax(x)
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


def summarize_segment_feature(values: np.ndarray) -> Dict[str, float]:
    values = clean_values(values)
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "q75": np.nan, "q90": np.nan, "q95": np.nan}
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "q75": float(np.nanpercentile(values, 75)),
        "q90": float(np.nanpercentile(values, 90)),
        "q95": float(np.nanpercentile(values, 95)),
    }


def build_file_time_lookup(file_map: Dict, file_indices_unique: np.ndarray) -> Dict[int, pd.Timestamp]:
    lookup = {}
    for fidx in file_indices_unique:
        fidx = int(fidx)
        entry = get_file_map_entry(file_map, fidx)
        fname = file_entry_to_name(entry)
        lookup[fidx] = parse_datetime_from_filename(fname)
    return lookup


def recording_key_and_segment_from_filename(file_name) -> Tuple[str, Optional[int]]:
    """Return a recording-level key and optional segment number.

    Many file_map entries look like:
        USA5288_45382.42553504_3_31_11_49_13_segment_0.npz

    The older phrase finder treated each file_index independently. If a long
    stutter spans segment_0, segment_1, ..., that artificially caps the phrase
    duration at one chunk. This helper lets us stitch consecutive file_index
    chunks belonging to the same original recording.
    """
    base = os.path.basename(str(file_name))
    stem = base
    for ext in [".npz", ".wav", ".WAV", ".npy"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]

    m = re.search(r"(?P<key>.+?)[_-]segment[_-]?(?P<seg>\d+)$", stem)
    if m:
        return m.group("key"), int(m.group("seg"))

    return stem, None


def ordered_unique_ints(values: Sequence[int]) -> List[int]:
    seen = set()
    out = []
    for v in values:
        iv = int(v)
        if iv not in seen:
            out.append(iv)
            seen.add(iv)
    return out


def find_phrase_runs_for_label(labels: np.ndarray, file_indices: np.ndarray, target_label: str,
                               period_files: Iterable[int], min_label_run_bins: int,
                               max_gap_bins_to_merge: int,
                               file_map: Optional[Dict] = None,
                               stitch_across_file_segments: bool = False,
                               phrase_label_smoothing: str = "none",
                               majority_vote_window_bins: int = 1,
                               majority_vote_threshold: float = 0.5,
                               majority_vote_keep_raw_target_bins: bool = False) -> List[Dict[str, int]]:
    """Find contiguous phrase-label runs.

    Default behavior is conservative: runs are found separately within each
    file_index. With stitch_across_file_segments=True, file_map entries sharing
    the same recording key (same filename after removing _segment_N) are
    concatenated in segment-number order before finding runs. This allows long
    stuttered phrases split across chunked file_index entries to be analyzed
    as one phrase.
    """
    runs = []
    period_files = set(int(x) for x in period_files)

    if not stitch_across_file_segments:
        for fidx in sorted(period_files):
            file_mask = file_indices == int(fidx)
            idx = np.flatnonzero(file_mask)
            if idx.size == 0:
                continue
            local_labels = labels[idx]
            target_mask = make_target_phrase_mask(
                local_labels,
                target_label=str(target_label),
                smoothing=phrase_label_smoothing,
                majority_vote_window_bins=majority_vote_window_bins,
                majority_vote_threshold=majority_vote_threshold,
                majority_vote_keep_raw_target_bins=majority_vote_keep_raw_target_bins,
            )
            target_mask = fill_small_false_gaps(target_mask, max_gap_bins_to_merge)
            for local_start, local_end in finite_runs(target_mask):
                run_len = local_end - local_start
                if run_len < min_label_run_bins:
                    continue
                run_indices = idx[local_start:local_end]
                global_start = int(run_indices[0])
                global_end = int(run_indices[-1]) + 1
                runs.append({
                    "file_index": int(fidx),
                    "file_indices_stitched": str(int(fidx)),
                    "n_files_stitched": 1,
                    "recording_key": (
                        file_entry_to_name(get_file_map_entry(file_map, int(fidx)))
                        if file_map is not None else str(fidx)
                    ),
                    "global_start_bin": global_start,
                    "global_end_bin": global_end,
                    "local_start_bin": int(local_start),
                    "local_end_bin": int(local_end),
                    "n_bins": int(len(run_indices)),
                    "phrase_label_smoothing": phrase_label_smoothing,
                    "majority_vote_window_bins": int(majority_vote_window_bins),
                    "majority_vote_threshold": float(majority_vote_threshold),
                    "max_gap_bins_to_merge": int(max_gap_bins_to_merge),
                    "global_indices": run_indices.astype(int),
                })
        return runs

    if file_map is None:
        raise ValueError("stitch_across_file_segments=True requires file_map.")

    groups: Dict[str, List[Tuple[int, int]]] = {}
    for fidx in sorted(period_files):
        try:
            fname = file_entry_to_name(get_file_map_entry(file_map, int(fidx)))
        except Exception:
            fname = str(fidx)
        key, seg_num = recording_key_and_segment_from_filename(fname)
        sort_seg = int(seg_num) if seg_num is not None else int(fidx)
        groups.setdefault(key, []).append((sort_seg, int(fidx)))

    for key, entries in sorted(groups.items(), key=lambda kv: kv[0]):
        entries = sorted(entries, key=lambda x: (x[0], x[1]))
        idx_pieces = []
        fidx_by_local_piece = []
        for seg_num, fidx in entries:
            idx = np.flatnonzero(file_indices == int(fidx))
            if idx.size == 0:
                continue
            idx_pieces.append(idx.astype(int))
            fidx_by_local_piece.extend([int(fidx)] * int(idx.size))
        if not idx_pieces:
            continue

        concat_idx = np.concatenate(idx_pieces).astype(int)
        concat_fidx = np.asarray(fidx_by_local_piece, dtype=int)
        local_labels = labels[concat_idx]
        target_mask = make_target_phrase_mask(
            local_labels,
            target_label=str(target_label),
            smoothing=phrase_label_smoothing,
            majority_vote_window_bins=majority_vote_window_bins,
            majority_vote_threshold=majority_vote_threshold,
            majority_vote_keep_raw_target_bins=majority_vote_keep_raw_target_bins,
        )
        target_mask = fill_small_false_gaps(target_mask, max_gap_bins_to_merge)

        for local_start, local_end in finite_runs(target_mask):
            run_len = local_end - local_start
            if run_len < min_label_run_bins:
                continue
            run_indices = concat_idx[local_start:local_end]
            run_file_indices = ordered_unique_ints(concat_fidx[local_start:local_end])
            if len(run_indices) == 0:
                continue
            global_start = int(run_indices[0])
            global_end = int(run_indices[-1]) + 1
            runs.append({
                "file_index": int(run_file_indices[0]),
                "file_indices_stitched": ",".join(str(x) for x in run_file_indices),
                "n_files_stitched": int(len(run_file_indices)),
                "recording_key": str(key),
                "global_start_bin": global_start,
                "global_end_bin": global_end,
                "local_start_bin": int(local_start),
                "local_end_bin": int(local_end),
                "n_bins": int(len(run_indices)),
                "phrase_label_smoothing": phrase_label_smoothing,
                "majority_vote_window_bins": int(majority_vote_window_bins),
                "majority_vote_threshold": float(majority_vote_threshold),
                "max_gap_bins_to_merge": int(max_gap_bins_to_merge),
                "global_indices": run_indices.astype(int),
            })

    return runs


def analyze_label_phrase_position(args, cluster_label: str, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_label = normalize_label_value(cluster_label)
    ensure_dir(out_dir)
    print(f"[INFO] Loading NPZ for label {cluster_label}...")
    data = np.load(args.npz_path, allow_pickle=True)
    X = np.asarray(data[args.spec_key], dtype=float)
    labels = normalize_label_array(data[args.label_key])
    file_indices = np.asarray(data[args.file_index_key]).astype(int)

    if X.shape[0] != labels.shape[0]:
        if X.ndim == 2 and X.shape[1] == labels.shape[0]:
            print("[INFO] Spectrogram appears transposed; transposing to time x frequency.")
            X = X.T
        else:
            raise ValueError(f"Spectrogram first dimension {X.shape[0]} does not match labels length {labels.shape[0]}")

    file_map = get_file_map_dict(data[args.file_map_key])
    unique_files = np.unique(file_indices)
    print("[INFO] Parsing file timestamps...")
    file_time_lookup = build_file_time_lookup(file_map, unique_files)
    treatment_date = load_treatment_date(
        args.metadata_excel_path,
        args.animal_id,
        metadata_sheet=args.metadata_sheet,
        animal_id_col=args.animal_id_col,
        treatment_date_col=args.treatment_date_col,
    )
    print(f"[INFO] Treatment date: {treatment_date.date()}")

    period_files = split_by_file_period(
        unique_files,
        file_time_lookup,
        treatment_date,
        include_treatment_day_in_post=args.include_treatment_day_in_post,
        split_mode=args.split_mode,
    )

    bin_s = args.bin_ms / 1000.0
    if args.phrase_label_smoothing != "none":
        print(
            f"[INFO] Phrase-run detection uses {args.phrase_label_smoothing}: "
            f"window={args.majority_vote_window_bins} bins "
            f"({args.majority_vote_window_bins * bin_s * 1000.0:.1f} ms), "
            f"threshold={args.majority_vote_threshold}, "
            f"keep_raw={args.majority_vote_keep_raw_target_bins}; "
            f"then bridges gaps <= {args.max_gap_bins_to_merge} bins."
        )
    all_rows = []
    phrase_counter = 0
    diagnostics_rows = []
    all_local_pair_rows = []

    for period in PERIOD_ORDER:
        runs = find_phrase_runs_for_label(
            labels,
            file_indices,
            str(cluster_label),
            period_files[period],
            min_label_run_bins=args.min_label_run_bins,
            max_gap_bins_to_merge=args.max_gap_bins_to_merge,
            file_map=file_map,
            stitch_across_file_segments=args.stitch_across_file_segments,
            phrase_label_smoothing=args.phrase_label_smoothing,
            majority_vote_window_bins=args.majority_vote_window_bins,
            majority_vote_threshold=args.majority_vote_threshold,
            majority_vote_keep_raw_target_bins=args.majority_vote_keep_raw_target_bins,
        )
        if runs:
            max_run_bins = max(int(r.get("n_bins", 0)) for r in runs)
            max_run_files = max(int(r.get("n_files_stitched", 1)) for r in runs)
            print(
                f"[INFO] {period}: {len(runs)} phrase-label runs for label {cluster_label}; "
                f"max duration={max_run_bins * bin_s:.2f}s; "
                f"max run bins={max_run_bins}; "
                f"max stitched files={max_run_files}"
            )
        else:
            print(f"[INFO] {period}: 0 phrase-label runs for label {cluster_label}")
        for run_idx, run in enumerate(runs):
            phrase_counter += 1
            run_indices = np.asarray(
                run.get("global_indices", np.arange(int(run["global_start_bin"]), int(run["global_end_bin"]))),
                dtype=int,
            )
            if run_indices.size == 0:
                continue
            g0 = int(run_indices[0])
            g1 = int(run_indices[-1]) + 1
            X_run = X[run_indices, :]
            if X_run.shape[0] < args.min_label_run_bins:
                continue

            pitch_khz, pitch_deriv = compute_pitch_trace_and_derivative(
                X_run,
                bin_s=bin_s,
                max_freq_khz=args.max_freq_khz,
                pitch_band_min_khz=args.pitch_band_min_khz,
                pitch_band_max_khz=args.pitch_band_max_khz,
                spec_scale=args.spec_scale,
                smoothing_window=args.pitch_smoothing_window,
                min_peak_fraction=args.pitch_min_peak_fraction,
            )
            wiener = compute_wiener_entropy(X_run, spec_scale=args.spec_scale)
            total_power = compute_total_power(X_run, spec_scale=args.spec_scale)

            seg = segment_power_by_canary_method(
                pitch_khz=pitch_khz,
                total_power=total_power,
                bin_s=bin_s,
                rate_band=(args.canary_rate_min_hz, args.canary_rate_max_hz),
                frac=args.canary_bandpass_frac,
                edge_guard_periods=args.canary_edge_guard_periods,
                pad_periods=args.canary_pad_periods,
                snap=not args.no_canary_snap,
                snap_periods=args.canary_snap_periods,
                min_points=args.segmentation_min_points,
                drop_edge_segments=not args.keep_edge_segments,
                min_segment_bins=args.min_segment_bins,
            )
            segments = seg["segments"]
            n_segments = len(segments)
            phrase_n_bins = int(len(run_indices))
            phrase_duration_s = phrase_n_bins * bin_s
            diagnostics_rows.append({
                "animal_id": args.animal_id,
                "label": cluster_label,
                "period": period,
                "phrase_id": phrase_counter,
                "file_index": run["file_index"],
                "file_indices_stitched": run.get("file_indices_stitched", str(run["file_index"])),
                "n_files_stitched": int(run.get("n_files_stitched", 1)),
                "recording_key": run.get("recording_key", ""),
                "global_start_bin": g0,
                "global_end_bin": g1,
                "phrase_duration_s": phrase_duration_s,
                "n_bins": phrase_n_bins,
                "n_segments": n_segments,
                "n_diagnostics_chunks": len(seg.get("diagnostics", [])),
                "phrase_label_smoothing": run.get("phrase_label_smoothing", args.phrase_label_smoothing),
                "majority_vote_window_bins": run.get("majority_vote_window_bins", args.majority_vote_window_bins),
                "majority_vote_threshold": run.get("majority_vote_threshold", args.majority_vote_threshold),
                "max_gap_bins_to_merge": run.get("max_gap_bins_to_merge", args.max_gap_bins_to_merge),
            })
            if n_segments == 0:
                continue

            # Prepare per-segment cross-correlation vectors, then build an early-phrase template
            # from the first few segmented syllables in this repeated phrase.
            seg_infos = []
            for seg_order_tmp, rec_tmp in enumerate(segments, start=1):
                s0_tmp = int(rec_tmp["segment_start_bin"])
                s1_tmp = int(rec_tmp["segment_end_bin"])
                if s1_tmp <= s0_tmp:
                    continue
                elapsed_start_tmp = s0_tmp * bin_s
                elapsed_end_tmp = s1_tmp * bin_s
                elapsed_mid_tmp = 0.5 * (elapsed_start_tmp + elapsed_end_tmp)
                X_seg_tmp = X_run[s0_tmp:s1_tmp, :]
                X_norm_tmp, corr_vec_tmp = prepare_segment_for_crosscorr(X_seg_tmp, args)
                seg_infos.append({
                    "rec": rec_tmp,
                    "seg_order": seg_order_tmp,
                    "s0": s0_tmp,
                    "s1": s1_tmp,
                    "elapsed_start": elapsed_start_tmp,
                    "elapsed_end": elapsed_end_tmp,
                    "elapsed_mid": elapsed_mid_tmp,
                    "X_norm": X_norm_tmp,
                    "corr_vec": corr_vec_tmp,
                    "corr_usable": corr_vec_tmp is not None,
                })

            template_indices = choose_phrase_template_indices(seg_infos, args)
            template_mats = [seg_infos[i]["X_norm"] for i in template_indices if i < len(seg_infos)]
            phrase_template, phrase_template_vec = build_corr_template(
                template_mats,
                min_template_segments=args.min_phrase_template_segments,
            )
            template_n_segments = len([m for m in template_mats if m is not None])
            template_max_elapsed_s = np.nan
            if len(template_indices) > 0:
                template_max_elapsed_s = np.nanmax([seg_infos[i]["elapsed_end"] for i in template_indices if i < len(seg_infos)])
            template_index_set = set(int(i) for i in template_indices if i < len(seg_infos))
            leave_one_out_template_vecs: Dict[int, Optional[np.ndarray]] = {}
            if args.leave_one_out_template_corr and len(template_index_set) > 0:
                for ii in template_index_set:
                    loo_mats = [
                        seg_infos[j]["X_norm"]
                        for j in template_index_set
                        if j != ii and seg_infos[j].get("X_norm") is not None
                    ]
                    _, loo_vec = build_corr_template(
                        loo_mats,
                        min_template_segments=args.min_phrase_template_segments,
                    )
                    leave_one_out_template_vecs[ii] = loo_vec

            if not getattr(args, "no_local_rate", False):
                base_pair_row = {
                    "animal_id": args.animal_id,
                    "label": str(cluster_label),
                    "period": period,
                    "pre_post": PRE_POST_MAP[period],
                    "phrase_id": phrase_counter,
                    "file_index": int(run["file_index"]),
                    "file_indices_stitched": run.get("file_indices_stitched", str(run["file_index"])),
                    "n_files_stitched": int(run.get("n_files_stitched", 1)),
                    "recording_key": run.get("recording_key", ""),
                    "file_datetime": file_time_lookup[int(run["file_index"])],
                    "phrase_global_start_bin": g0,
                    "phrase_global_end_bin": g1,
                    "phrase_duration_s": phrase_duration_s,
                    "phrase_n_bins": phrase_n_bins,
                    "n_segments_in_phrase": n_segments,
                    "local_rate_max_lag": int(args.local_rate_max_lag),
                    "corr_time_bins": int(args.corr_time_bins),
                    "corr_freq_min_khz": args.corr_freq_min_khz,
                    "corr_freq_max_khz": args.corr_freq_max_khz,
                    "corr_transform": args.corr_transform,
                }
                all_local_pair_rows.extend(
                    build_local_shortlag_pair_rows(
                        seg_infos,
                        base_row=base_pair_row,
                        max_lag=args.local_rate_max_lag,
                        global_run_indices=run_indices,
                    )
                )

            for info in seg_infos:
                rec = info["rec"]
                seg_order = int(info["seg_order"])
                s0 = int(info["s0"])
                s1 = int(info["s1"])
                if s1 <= s0:
                    continue
                sl = slice(s0, s1)
                pd_stats = summarize_segment_feature(pitch_deriv[sl])
                we_stats = summarize_segment_feature(wiener[sl])
                pitch_stats = summarize_segment_feature(pitch_khz[sl])
                elapsed_start = float(info["elapsed_start"])
                elapsed_end = float(info["elapsed_end"])
                elapsed_mid = float(info["elapsed_mid"])
                seg_zero_based_idx = seg_order - 1
                template_vec_for_this_segment = phrase_template_vec
                if args.leave_one_out_template_corr and seg_zero_based_idx in leave_one_out_template_vecs:
                    template_vec_for_this_segment = leave_one_out_template_vecs.get(seg_zero_based_idx)
                corr_to_template = corr_between_unit_vectors(info.get("corr_vec"), template_vec_for_this_segment)
                distance_from_template = 1.0 - corr_to_template if np.isfinite(corr_to_template) else np.nan
                lag_corr_values = {}
                for lag in getattr(args, "lag_crosscorr_steps", [10]):
                    lag = int(lag)
                    # seg_order is 1-based; list index is 0-based.
                    lag_idx = seg_order - lag - 1
                    prev_vec = seg_infos[lag_idx].get("corr_vec") if 0 <= lag_idx < len(seg_infos) else None
                    lag_corr = corr_between_unit_vectors(info.get("corr_vec"), prev_vec)
                    lag_corr_values[f"corr_to_lag{lag}_previous_syllable"] = lag_corr
                    lag_corr_values[f"distance_from_lag{lag}_previous_syllable"] = (
                        1.0 - lag_corr if np.isfinite(lag_corr) else np.nan
                    )
                    lag_corr_values[f"lag{lag}_reference_repeat_index_in_phrase"] = (
                        lag_idx + 1 if prev_vec is not None else np.nan
                    )
                all_rows.append({
                    "animal_id": args.animal_id,
                    "label": str(cluster_label),
                    "period": period,
                    "pre_post": PRE_POST_MAP[period],
                    "phrase_id": phrase_counter,
                    "file_index": int(run["file_index"]),
                    "file_indices_stitched": run.get("file_indices_stitched", str(run["file_index"])),
                    "n_files_stitched": int(run.get("n_files_stitched", 1)),
                    "recording_key": run.get("recording_key", ""),
                    "phrase_label_smoothing": run.get("phrase_label_smoothing", args.phrase_label_smoothing),
                    "majority_vote_window_bins": run.get("majority_vote_window_bins", args.majority_vote_window_bins),
                    "majority_vote_threshold": run.get("majority_vote_threshold", args.majority_vote_threshold),
                    "max_gap_bins_to_merge": run.get("max_gap_bins_to_merge", args.max_gap_bins_to_merge),
                    "file_datetime": file_time_lookup[int(run["file_index"])],
                    "phrase_global_start_bin": g0,
                    "phrase_global_end_bin": g1,
                    "phrase_duration_s": phrase_duration_s,
                    "phrase_n_bins": phrase_n_bins,
                    "n_segments_in_phrase": n_segments,
                    "repeat_index_in_phrase": seg_order,
                    "n_previous_segments": seg_order - 1,
                    "elapsed_time_in_phrase_s": elapsed_mid,
                    "segment_start_elapsed_s": elapsed_start,
                    "segment_end_elapsed_s": elapsed_end,
                    "fraction_through_phrase": elapsed_mid / phrase_duration_s if phrase_duration_s > 0 else np.nan,
                    "segment_start_global_bin": int(run_indices[s0]) if s0 < len(run_indices) else np.nan,
                    "segment_end_global_bin": int(run_indices[s1 - 1]) + 1 if (s1 - 1) < len(run_indices) and s1 > 0 else np.nan,
                    "segment_duration_s": (s1 - s0) * bin_s,
                    "mean_pitch_derivative_khz_per_s": pd_stats["mean"],
                    "median_pitch_derivative_khz_per_s": pd_stats["median"],
                    "q75_pitch_derivative_khz_per_s": pd_stats["q75"],
                    "q90_pitch_derivative_khz_per_s": pd_stats["q90"],
                    "q95_pitch_derivative_khz_per_s": pd_stats["q95"],
                    "mean_wiener_entropy": we_stats["mean"],
                    "median_wiener_entropy": we_stats["median"],
                    "q75_wiener_entropy": we_stats["q75"],
                    "q90_wiener_entropy": we_stats["q90"],
                    "q95_wiener_entropy": we_stats["q95"],
                    "mean_pitch_khz": pitch_stats["mean"],
                    "median_pitch_khz": pitch_stats["median"],
                    "corr_usable": bool(info.get("corr_usable", False)),
                    "phrase_template_mode": args.phrase_template_mode,
                    "phrase_template_n_segments": int(template_n_segments),
                    "phrase_template_max_elapsed_s": template_max_elapsed_s,
                    "leave_one_out_template_corr": bool(args.leave_one_out_template_corr),
                    "corr_to_phrase_early_template": corr_to_template,
                    "distance_from_phrase_early_template": distance_from_template,
                    **lag_corr_values,
                    "dominant_pitch_modulation_freq_hz": rec.get("dominant_pitch_modulation_freq_hz", np.nan),
                })

    segment_df = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(diagnostics_rows)
    local_pair_df = pd.DataFrame(all_local_pair_rows)
    prefix = f"{args.animal_id}_label{str(cluster_label).replace('.', 'p')}"
    seg_csv = out_dir / f"{prefix}_phrase_position_segment_features.csv"
    diag_csv = out_dir / f"{prefix}_phrase_position_phrase_diagnostics.csv"
    local_pair_csv = out_dir / f"{prefix}_local_shortlag_pairs.csv"
    segment_df.to_csv(seg_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)
    local_pair_df.to_csv(local_pair_csv, index=False)
    print(f"[SAVED] {seg_csv}")
    print(f"[SAVED] {diag_csv}")
    print(f"[SAVED] {local_pair_csv}")
    if not diag_df.empty:
        max_duration = pd.to_numeric(diag_df.get("phrase_duration_s"), errors="coerce").max()
        max_segments = pd.to_numeric(diag_df.get("n_segments"), errors="coerce").max()
        max_files = pd.to_numeric(diag_df.get("n_files_stitched", pd.Series([1])), errors="coerce").max()
        print(
            f"[INFO] Label {cluster_label} phrase diagnostics: "
            f"max phrase duration={max_duration:.2f}s; "
            f"max segmented syllables={max_segments:.0f}; "
            f"max stitched files={max_files:.0f}"
        )

    if segment_df.empty:
        print(f"[WARN] No segmented syllables found for label {cluster_label}; skipping plots.")
        return segment_df, pd.DataFrame(), pd.DataFrame(), local_pair_df, pd.DataFrame(), pd.DataFrame()

    features = unique_preserve_order([x.strip() for x in args.features.split(",") if x.strip()])
    stats_df = compute_position_stats(segment_df, features=features, min_n=args.min_values_per_group)
    stats_csv = out_dir / f"{prefix}_phrase_position_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[SAVED] {stats_csv}")

    early_late_df = compute_early_late_window_summary(
        segment_df,
        features=features,
        early_window_s=args.early_window_s,
        late_window_start_s=args.late_window_start_s,
        late_window_duration_s=args.late_window_duration_s,
        min_segments_per_window=args.min_segments_per_window,
    )
    early_late_csv = out_dir / f"{prefix}_early_late_window_summary.csv"
    early_late_df.to_csv(early_late_csv, index=False)
    print(f"[SAVED] {early_late_csv}")

    make_feature_vs_elapsed_time_plot(
        segment_df,
        features=features,
        out_png=out_dir / f"{prefix}_feature_vs_elapsed_time.png",
        title=f"{args.animal_id} label {cluster_label}: acoustic features vs elapsed phrase time",
        max_x=args.max_plot_elapsed_s,
    )
    make_feature_vs_previous_repeats_plot(
        segment_df,
        features=features,
        out_png=out_dir / f"{prefix}_feature_vs_previous_repeats.png",
        title=f"{args.animal_id} label {cluster_label}: acoustic features vs previous repeats",
        max_repeats=args.max_plot_previous_repeats,
    )

    corr_features = unique_preserve_order([
        "corr_to_phrase_early_template",
        "distance_from_phrase_early_template",
        *get_lag_crosscorr_feature_names(args.lag_crosscorr_steps),
    ])
    corr_features = [f for f in corr_features if f in segment_df.columns]
    corr_stats_df = compute_position_stats(segment_df, features=corr_features, min_n=args.min_values_per_group)
    corr_stats_csv = out_dir / f"{prefix}_phrase_position_crosscorr_stats.csv"
    corr_stats_df.to_csv(corr_stats_csv, index=False)
    print(f"[SAVED] {corr_stats_csv}")
    make_feature_vs_elapsed_time_plot(
        segment_df,
        features=corr_features,
        out_png=out_dir / f"{prefix}_crosscorr_vs_elapsed_time.png",
        title=f"{args.animal_id} label {cluster_label}: spectrogram similarity vs elapsed phrase time",
        max_x=args.max_plot_elapsed_s,
    )
    make_feature_vs_previous_repeats_plot(
        segment_df,
        features=corr_features,
        out_png=out_dir / f"{prefix}_crosscorr_vs_previous_repeats.png",
        title=f"{args.animal_id} label {cluster_label}: spectrogram similarity vs previous repeats",
        max_repeats=args.max_plot_previous_repeats,
    )
    local_rate_stats_df = pd.DataFrame()
    local_phrase_rate_df = pd.DataFrame()
    if not getattr(args, "no_local_rate", False) and not local_pair_df.empty:
        local_rate_features = [
            "local_rate_spectral_displacement_per_repeat",
            "local_rate_spectral_displacement_sq_per_repeat",
        ]
        local_rate_features = [f for f in local_rate_features if f in local_pair_df.columns]
        local_rate_stats_df = compute_position_stats(
            local_pair_df,
            features=local_rate_features,
            min_n=args.min_values_per_group,
        )
        local_rate_stats_csv = out_dir / f"{prefix}_local_rate_position_stats.csv"
        local_rate_stats_df.to_csv(local_rate_stats_csv, index=False)
        print(f"[SAVED] {local_rate_stats_csv}")
        local_phrase_rate_df = compute_local_phrase_rate_summary(
            local_pair_df,
            min_pairs=args.min_local_rate_pairs_per_phrase,
        )
        local_phrase_rate_csv = out_dir / f"{prefix}_local_shortlag_phrase_rate_summary.csv"
        local_phrase_rate_df.to_csv(local_phrase_rate_csv, index=False)
        print(f"[SAVED] {local_phrase_rate_csv}")
        make_feature_vs_elapsed_time_plot(
            local_pair_df,
            features=local_rate_features,
            out_png=out_dir / f"{prefix}_local_rate_vs_elapsed_time.png",
            title=f"{args.animal_id} label {cluster_label}: local acoustic change rate vs elapsed phrase time",
            max_x=args.max_plot_elapsed_s,
        )
        make_feature_vs_previous_repeats_plot(
            local_pair_df,
            features=local_rate_features,
            out_png=out_dir / f"{prefix}_local_rate_vs_previous_repeats.png",
            title=f"{args.animal_id} label {cluster_label}: local acoustic change rate vs previous repeats",
            max_repeats=args.max_plot_previous_repeats,
        )
        make_slope_summary_plot(
            local_rate_stats_df,
            features=local_rate_features,
            out_png=out_dir / f"{prefix}_local_rate_slope_summary_by_period.png",
            title=f"{args.animal_id} label {cluster_label}: slope of local acoustic change rate vs phrase time",
        )
        make_local_pair_qc_outputs(
            local_pair_df,
            segment_df,
            X,
            out_dir=out_dir,
            prefix=prefix,
            args=args,
        )

    make_early_late_window_plot(
        early_late_df,
        features=features,
        out_png=out_dir / f"{prefix}_early_vs_late_windows.png",
        title=f"{args.animal_id} label {cluster_label}: early vs late within repeated phrases",
    )
    make_slope_summary_plot(
        stats_df,
        features=features,
        out_png=out_dir / f"{prefix}_slope_summary_by_period.png",
        title=f"{args.animal_id} label {cluster_label}: slope of acoustic feature vs phrase time",
    )

    # Return acoustic, cross-correlation, and local-rate stats together for the combined all-label table.
    stats_to_combine = [x for x in [stats_df, corr_stats_df, local_rate_stats_df] if not x.empty]
    combined_stats_for_return = pd.concat(stats_to_combine, ignore_index=True) if stats_to_combine else pd.DataFrame()
    return segment_df, combined_stats_for_return, early_late_df, local_pair_df, local_rate_stats_df, local_phrase_rate_df


def _safe_spearman_and_linear(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> Dict[str, float]:
    """Safely compute Spearman correlation and linear regression.

    Returns NaNs if there are too few points, if x is constant, or if y is constant.
    This prevents crashes like:
        ValueError: Cannot calculate a linear regression if all x values are identical
    """
    out = {
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "linear_slope": np.nan,
        "linear_intercept": np.nan,
        "linear_r": np.nan,
        "linear_p": np.nan,
        "n_valid": 0,
        "n_unique_x": 0,
        "n_unique_y": 0,
        "fit_status": "not_run",
    }
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    out["n_valid"] = int(xv.size)
    if xv.size > 0:
        out["n_unique_x"] = int(np.unique(xv).size)
        out["n_unique_y"] = int(np.unique(yv).size)

    if xv.size < min_n:
        out["fit_status"] = "too_few_valid_points"
        return out
    if np.unique(xv).size < 2:
        out["fit_status"] = "constant_x"
        return out
    if np.unique(yv).size < 2:
        out["fit_status"] = "constant_y"
        return out

    if SCIPY_AVAILABLE:
        try:
            sp = stats.spearmanr(xv, yv)
            out["spearman_r"] = float(sp.correlation) if np.isfinite(sp.correlation) else np.nan
            out["spearman_p"] = float(sp.pvalue) if np.isfinite(sp.pvalue) else np.nan
        except Exception:
            pass
        try:
            lr = stats.linregress(xv, yv)
            out["linear_slope"] = float(lr.slope)
            out["linear_intercept"] = float(lr.intercept)
            out["linear_r"] = float(lr.rvalue)
            out["linear_p"] = float(lr.pvalue)
            out["fit_status"] = "ok"
        except Exception as exc:
            out["fit_status"] = f"linregress_failed:{type(exc).__name__}"
    else:
        try:
            slope, intercept = np.polyfit(xv, yv, deg=1)
            out["linear_slope"] = float(slope)
            out["linear_intercept"] = float(intercept)
            out["fit_status"] = "ok_polyfit_only"
        except Exception as exc:
            out["fit_status"] = f"polyfit_failed:{type(exc).__name__}"
    return out


def compute_position_stats(df: pd.DataFrame, features: List[str], min_n=5) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        for group_col, group_values in [
            ("period", PERIOD_ORDER),
            ("pre_post", ["pre", "post"]),
        ]:
            for group in group_values:
                sub = df[df[group_col] == group].copy()
                x_time = sub["elapsed_time_in_phrase_s"].to_numpy(dtype=float)
                x_rep = sub["n_previous_segments"].to_numpy(dtype=float)
                y = sub[feature].to_numpy(dtype=float)

                time_fit = _safe_spearman_and_linear(x_time, y, min_n=min_n)
                rep_fit = _safe_spearman_and_linear(x_rep, y, min_n=min_n)

                row = {
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "group_col": group_col,
                    "group": group,
                    "n": int(np.sum(np.isfinite(y))),
                    "n_time_valid": int(time_fit["n_valid"]),
                    "n_time_unique_x": int(time_fit["n_unique_x"]),
                    "n_time_unique_y": int(time_fit["n_unique_y"]),
                    "elapsed_time_fit_status": time_fit["fit_status"],
                    "n_repeat_valid": int(rep_fit["n_valid"]),
                    "n_repeat_unique_x": int(rep_fit["n_unique_x"]),
                    "n_repeat_unique_y": int(rep_fit["n_unique_y"]),
                    "previous_repeats_fit_status": rep_fit["fit_status"],
                    "spearman_r_elapsed_time": time_fit["spearman_r"],
                    "spearman_p_elapsed_time": time_fit["spearman_p"],
                    "linear_slope_per_s": time_fit["linear_slope"],
                    "linear_intercept_elapsed_time": time_fit["linear_intercept"],
                    "linear_r_elapsed_time": time_fit["linear_r"],
                    "linear_p_elapsed_time": time_fit["linear_p"],
                    "spearman_r_previous_repeats": rep_fit["spearman_r"],
                    "spearman_p_previous_repeats": rep_fit["spearman_p"],
                    "linear_slope_per_repeat": rep_fit["linear_slope"],
                    "linear_intercept_previous_repeats": rep_fit["linear_intercept"],
                    "linear_r_previous_repeats": rep_fit["linear_r"],
                    "linear_p_previous_repeats": rep_fit["linear_p"],
                }
                rows.append(row)
    return pd.DataFrame(rows)


def compute_early_late_window_summary(df: pd.DataFrame, features: List[str], early_window_s=1.0,
                                      late_window_start_s=10.0, late_window_duration_s=1.0,
                                      min_segments_per_window=1) -> pd.DataFrame:
    rows = []
    late_end = late_window_start_s + late_window_duration_s
    for (period, phrase_id), sub in df.groupby(["period", "phrase_id"]):
        for feature in features:
            if feature not in sub.columns:
                continue
            early = sub[(sub["elapsed_time_in_phrase_s"] >= 0) & (sub["elapsed_time_in_phrase_s"] <= early_window_s)]
            late = sub[(sub["elapsed_time_in_phrase_s"] >= late_window_start_s) & (sub["elapsed_time_in_phrase_s"] <= late_end)]
            ev = clean_values(early[feature].to_numpy(dtype=float)) if len(early) else np.array([])
            lv = clean_values(late[feature].to_numpy(dtype=float)) if len(late) else np.array([])
            if ev.size < min_segments_per_window or lv.size < min_segments_per_window:
                continue
            first = sub.iloc[0]
            rows.append({
                "animal_id": first["animal_id"],
                "label": first["label"],
                "period": period,
                "pre_post": PRE_POST_MAP[period],
                "phrase_id": phrase_id,
                "feature": feature,
                "feature_label": feature_label(feature),
                "early_window_s": early_window_s,
                "late_window_start_s": late_window_start_s,
                "late_window_end_s": late_end,
                "n_early_segments": int(ev.size),
                "n_late_segments": int(lv.size),
                "early_median": float(np.nanmedian(ev)),
                "late_median": float(np.nanmedian(lv)),
                "late_minus_early": float(np.nanmedian(lv) - np.nanmedian(ev)),
                "phrase_duration_s": float(first["phrase_duration_s"]),
                "n_segments_in_phrase": int(first["n_segments_in_phrase"]),
            })
    return pd.DataFrame(rows)


def binned_median_line(x: np.ndarray, y: np.ndarray, n_bins=12, x_min=None, x_max=None):
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    if x_min is None:
        x_min = np.nanmin(x)
    if x_max is None:
        x_max = np.nanmax(x)
    if x_max <= x_min:
        return np.array([]), np.array([]), np.array([])
    bins = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (x >= bins[i]) & (x < bins[i + 1] if i < n_bins - 1 else x <= bins[i + 1])
        counts[i] = int(np.sum(m))
        if counts[i] > 0:
            med[i] = np.nanmedian(y[m])
    good = np.isfinite(med)
    return centers[good], med[good], counts[good]



def get_pre_post_x_ranges(df: pd.DataFrame, x_col: str, max_x: Optional[float] = None) -> Dict[str, float]:
    """Return plotting ranges for pre/post x-values, optionally clipped to max_x.

    This is used to show the shared pre/post range and the post-lesion-only tail.
    The tail is the range beyond the maximum pre-lesion x-value where only post-lesion
    stuttered repetitions may exist.
    """
    ranges = {
        "x_min": np.nan,
        "x_max": np.nan,
        "pre_min": np.nan,
        "pre_max": np.nan,
        "post_min": np.nan,
        "post_max": np.nan,
    }
    if x_col not in df.columns or "pre_post" not in df.columns:
        return ranges

    work = df[[x_col, "pre_post"]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work = work[np.isfinite(work[x_col])]
    if max_x is not None:
        work = work[work[x_col] <= max_x]
    if work.empty:
        return ranges

    ranges["x_min"] = float(np.nanmin(work[x_col]))
    ranges["x_max"] = float(np.nanmax(work[x_col]))

    pre = work.loc[work["pre_post"] == "pre", x_col].to_numpy(dtype=float)
    post = work.loc[work["pre_post"] == "post", x_col].to_numpy(dtype=float)
    if pre.size:
        ranges["pre_min"] = float(np.nanmin(pre))
        ranges["pre_max"] = float(np.nanmax(pre))
    if post.size:
        ranges["post_min"] = float(np.nanmin(post))
        ranges["post_max"] = float(np.nanmax(post))

    # For the biological stuttering plot, prefer showing through max post if present.
    if post.size:
        ranges["x_max"] = ranges["post_max"]
    if max_x is not None and np.isfinite(ranges["x_max"]):
        ranges["x_max"] = min(ranges["x_max"], float(max_x))
    return ranges


def add_pre_post_tail_annotations(ax, ranges: Dict[str, float], x_label_units: str = "") -> None:
    """Add max-pre marker and post-lesion-only shaded tail to an axis."""
    pre_max = ranges.get("pre_max", np.nan)
    post_max = ranges.get("post_max", np.nan)
    x_min = ranges.get("x_min", np.nan)
    x_max = ranges.get("x_max", np.nan)

    if not (np.isfinite(pre_max) and np.isfinite(post_max) and np.isfinite(x_min) and np.isfinite(x_max)):
        return
    if post_max <= pre_max:
        # No post-only extended range for this feature/label.
        return

    # Limit shading to the actually displayed axis range.
    shade_start = max(pre_max, x_min)
    shade_end = min(post_max, x_max)
    if shade_end <= shade_start:
        return

    ax.axvline(pre_max, linestyle="--", linewidth=1.5, alpha=0.7, label="max pre range")
    ax.axvspan(shade_start, shade_end, alpha=0.08, label="post-only stutter range")
    ax.axvline(shade_end, linestyle=":", linewidth=1.2, alpha=0.45, label="max post range")

    # Add a small text annotation near the top of the plot. Use axes coordinates for y.
    try:
        label = "post-only\nstutter range"
        ax.text(
            0.985,
            0.94,
            label,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.75,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=2),
        )
    except Exception:
        pass


def deduplicate_legend(ax, fontsize: int = 8) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab in seen or lab == "":
            continue
        seen.add(lab)
        new_handles.append(h)
        new_labels.append(lab)
    if new_handles:
        ax.legend(new_handles, new_labels, frameon=True, fontsize=fontsize)


def make_feature_vs_elapsed_time_plot(df: pd.DataFrame, features: List[str], out_png: Path, title: str,
                                      max_x: Optional[float] = None):
    """Plot features versus elapsed phrase time.

    The x-axis is allowed to extend into the post-lesion-only range, so long
    stuttered post-lesion phrases are visible. The pre-lesion binned median line
    stops where pre-lesion data stop; it is not extrapolated into the tail.
    """
    n = len(features)
    if n == 0:
        return
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)

    ranges = get_pre_post_x_ranges(df, "elapsed_time_in_phrase_s", max_x=max_x)
    shared_xlim = None
    if np.isfinite(ranges.get("x_min", np.nan)) and np.isfinite(ranges.get("x_max", np.nan)):
        pad = 0.03 * max(1e-9, ranges["x_max"] - ranges["x_min"])
        shared_xlim = (max(0.0, ranges["x_min"] - pad), ranges["x_max"] + pad)

    for ax, feature in zip(axes.ravel(), features):
        if feature not in df.columns:
            ax.axis("off")
            continue
        for group in ["pre", "post"]:
            sub = df[df["pre_post"] == group]
            x = sub["elapsed_time_in_phrase_s"].to_numpy(dtype=float)
            y = sub[feature].to_numpy(dtype=float)
            if max_x is not None:
                keep = x <= max_x
                x = x[keep]
                y = y[keep]
            ax.scatter(x, y, s=10, alpha=0.16, label=f"{group} segments")
            # IMPORTANT: bins are computed only over each group's observed range.
            # This makes the pre line stop where pre data stop instead of extrapolating.
            cx, my, counts = binned_median_line(x, y, n_bins=12)
            if len(cx):
                ax.plot(cx, my, linewidth=2.5, marker="o", label=f"{group} binned median")
        add_pre_post_tail_annotations(ax, ranges, x_label_units="s")
        if shared_xlim is not None:
            ax.set_xlim(*shared_xlim)
        ax.set_xlabel("Elapsed time in repeated phrase (s)")
        ax.set_ylabel(feature_label(feature))
        ax.set_title(feature_label(feature))
        ax.grid(alpha=0.25)
        deduplicate_legend(ax, fontsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_feature_vs_previous_repeats_plot(df: pd.DataFrame, features: List[str], out_png: Path, title: str,
                                          max_repeats: Optional[int] = None):
    """Plot features versus previous repeat number.

    The x-axis is allowed to extend into repeat numbers that occur only post-lesion.
    The pre-lesion median-by-repeat line stops where pre-lesion data stop.
    """
    n = len(features)
    if n == 0:
        return
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)

    ranges = get_pre_post_x_ranges(df, "n_previous_segments", max_x=max_repeats)
    shared_xlim = None
    if np.isfinite(ranges.get("x_min", np.nan)) and np.isfinite(ranges.get("x_max", np.nan)):
        pad = 0.03 * max(1.0, ranges["x_max"] - ranges["x_min"])
        shared_xlim = (max(-0.5, ranges["x_min"] - pad), ranges["x_max"] + pad)

    for ax, feature in zip(axes.ravel(), features):
        if feature not in df.columns:
            ax.axis("off")
            continue
        for group in ["pre", "post"]:
            sub = df[df["pre_post"] == group]
            x = sub["n_previous_segments"].to_numpy(dtype=float)
            y = sub[feature].to_numpy(dtype=float)
            if max_repeats is not None:
                keep = x <= max_repeats
                x = x[keep]
                y = y[keep]
            ax.scatter(x, y, s=10, alpha=0.16, label=f"{group} segments")
            # Median per integer repeat count; computed only where that group has data.
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                xx = x[valid].astype(int)
                yy = y[valid]
                centers, med = [], []
                for k in sorted(np.unique(xx)):
                    if max_repeats is not None and k > max_repeats:
                        continue
                    vals = yy[xx == k]
                    if vals.size >= 2:
                        centers.append(k)
                        med.append(np.nanmedian(vals))
                if len(centers):
                    ax.plot(centers, med, linewidth=2.5, marker="o", label=f"{group} median by repeat")
        add_pre_post_tail_annotations(ax, ranges, x_label_units="repeats")
        if shared_xlim is not None:
            ax.set_xlim(*shared_xlim)
        ax.set_xlabel("Number of previous analyzed syllables in phrase")
        ax.set_ylabel(feature_label(feature))
        ax.set_title(feature_label(feature))
        ax.grid(alpha=0.25)
        deduplicate_legend(ax, fontsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_early_late_window_plot(early_late_df: pd.DataFrame, features: List[str], out_png: Path, title: str):
    if early_late_df.empty:
        print("[WARN] No early/late window rows; skipping early_vs_late plot.")
        return
    n = len(features)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        sub = early_late_df[early_late_df["feature"] == feature]
        if sub.empty:
            ax.axis("off")
            continue
        x_positions = {"pre": [0, 1], "post": [3, 4]}
        labels = ["pre early", "pre late", "post early", "post late"]
        for group in ["pre", "post"]:
            g = sub[sub["pre_post"] == group]
            xs = x_positions[group]
            for _, row in g.iterrows():
                ax.plot(xs, [row["early_median"], row["late_median"]], alpha=0.18, linewidth=1)
                ax.scatter(xs, [row["early_median"], row["late_median"]], s=12, alpha=0.22)
            if len(g):
                ax.scatter(xs, [np.nanmedian(g["early_median"]), np.nanmedian(g["late_median"])], s=80, marker="D", label=f"{group} median")
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(feature_label(feature))
        ax.set_title(feature_label(feature))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=True, fontsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_slope_summary_plot(stats_df: pd.DataFrame, features: List[str], out_png: Path, title: str):
    if stats_df.empty:
        return
    sub = stats_df[(stats_df["group_col"] == "period") & (stats_df["feature"].isin(features))].copy()
    if sub.empty:
        return
    n = len(features)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        s = sub[sub["feature"] == feature].set_index("group").reindex(PERIOD_ORDER).reset_index()
        ax.axhline(0, linestyle="--", linewidth=1.2)
        ax.scatter(np.arange(len(PERIOD_ORDER)), s["linear_slope_per_s"].to_numpy(dtype=float), s=80)
        ax.set_xticks(np.arange(len(PERIOD_ORDER)))
        ax.set_xticklabels([PERIOD_TITLES[p] for p in PERIOD_ORDER], rotation=25, ha="right")
        ax.set_ylabel("Linear slope per second into phrase")
        ax.set_title(feature_label(feature))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def parse_label_list(args) -> List[str]:
    labels = []
    if args.cluster_label is not None:
        labels.extend([normalize_label_value(x) for x in str(args.cluster_label).split(",") if normalize_label_value(x)])
    if args.labels_csv is not None:
        lab_col = args.labels_csv_label_col
        df = pd.read_csv(args.labels_csv)
        if args.labels_csv_animal_col in df.columns:
            df = df[df[args.labels_csv_animal_col].astype(str) == str(args.animal_id)]
        if lab_col not in df.columns:
            # Try common alternatives.
            for candidate in ["label", "syllable_label", "cluster_label", "Syllable", "Syllable label", "mapped_syllable_order"]:
                if candidate in df.columns:
                    lab_col = candidate
                    break
        if lab_col not in df.columns:
            raise ValueError(f"Could not find label column {args.labels_csv_label_col!r} in {args.labels_csv}")
        labels.extend([normalize_label_value(x) for x in df[lab_col].dropna().tolist() if normalize_label_value(x)])
    labels = sorted(set(labels), key=lambda x: (len(str(x)), str(x)))
    if len(labels) == 0:
        raise ValueError("Provide --cluster-label or --labels-csv.")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Analyze acoustic features as a function of position within repeated syllable phrases.")
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--cluster-label", default=None, help="Single label or comma-separated labels, e.g. 17 or 3,17,19")
    parser.add_argument("--labels-csv", default=None, help="Optional CSV containing labels to analyze for this animal.")
    parser.add_argument("--labels-csv-label-col", default="label")
    parser.add_argument("--labels-csv-animal-col", default="animal_id")
    parser.add_argument("--animal-id", required=True)
    parser.add_argument("--metadata-excel-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--spec-key", default="s")
    parser.add_argument("--label-key", default="hdbscan_labels")
    parser.add_argument("--file-index-key", default="file_indices")
    parser.add_argument("--file-map-key", default="file_map")
    parser.add_argument("--metadata-sheet", default="metadata")
    parser.add_argument("--animal-id-col", default="Animal ID")
    parser.add_argument("--treatment-date-col", default="Treatment date")
    parser.add_argument("--bin-ms", type=float, default=2.7)
    parser.add_argument("--max-freq-khz", type=float, default=12.0)
    parser.add_argument("--spec-scale", default="linear", choices=["linear", "log10", "loge", "shift"])
    parser.add_argument("--split-mode", default="file_half", choices=["file_half", "file_median"])
    parser.add_argument("--include-treatment-day-in-post", action="store_true")
    parser.add_argument("--min-label-run-bins", type=int, default=40,
                        help="Minimum target-label phrase-run length in time bins. Default 40 bins (~108 ms at 2.7 ms/bin).")
    parser.add_argument("--max-gap-bins-to-merge", type=int, default=0,
                        help="After optional majority-vote smoothing, merge target-label runs across small non-target gaps within a file or stitched recording. Default 0.")
    parser.add_argument("--phrase-label-smoothing", default="none", choices=["none", "target_majority_vote"],
                        help=("How to smooth labels before phrase-run detection. "
                              "Use target_majority_vote to apply a target-vs-other majority vote so brief decoder label interruptions do not split long stutters."))
    parser.add_argument("--majority-vote-window-bins", type=int, default=31,
                        help="Centered window size, in time bins, for --phrase-label-smoothing target_majority_vote. Even values are rounded up to odd. Default 31 bins.")
    parser.add_argument("--majority-vote-threshold", type=float, default=0.5,
                        help="Minimum fraction of target-label bins in the centered window required to call the center bin part of the phrase. Default 0.5.")
    parser.add_argument("--majority-vote-keep-raw-target-bins", action="store_true",
                        help="After target majority vote, OR the smoothed mask with the original target-label mask. This preserves raw target bins but may keep more short islands.")
    parser.add_argument(
        "--stitch-across-file-segments",
        action="store_true",
        help=(
            "Treat consecutive file_index chunks from the same original recording "
            "(_segment_N filenames) as one continuous recording before finding phrase-label runs. "
            "Use this when long stuttered phrases were split across chunked NPZ segments."
        ),
    )
    parser.add_argument("--min-segment-bins", type=int, default=3)
    parser.add_argument("--pitch-band-min-khz", type=float, default=0.8)
    parser.add_argument("--pitch-band-max-khz", type=float, default=10.0)
    parser.add_argument("--pitch-min-peak-fraction", type=float, default=0.0)
    parser.add_argument("--pitch-smoothing-window", type=int, default=5)
    parser.add_argument("--canary-rate-min-hz", type=float, default=2.0)
    parser.add_argument("--canary-rate-max-hz", type=float, default=20.0)
    parser.add_argument("--canary-bandpass-frac", type=float, default=0.5)
    parser.add_argument("--canary-edge-guard-periods", type=float, default=0.5)
    parser.add_argument("--canary-pad-periods", type=float, default=3.0)
    parser.add_argument("--canary-snap-periods", type=float, default=0.5)
    parser.add_argument("--no-canary-snap", action="store_true")
    parser.add_argument("--segmentation-min-points", type=int, default=12)
    parser.add_argument("--keep-edge-segments", action="store_true", help="Do not drop first/last segment in each run.")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--min-values-per-group", type=int, default=5)
    parser.add_argument("--early-window-s", type=float, default=1.0)
    parser.add_argument("--late-window-start-s", type=float, default=10.0)
    parser.add_argument("--late-window-duration-s", type=float, default=1.0)
    parser.add_argument("--min-segments-per-window", type=int, default=1)
    parser.add_argument("--max-plot-elapsed-s", type=float, default=None,
                        help="Optional x-axis maximum for elapsed-time plots. If omitted, plot through the max post-lesion elapsed time so post-only stutter tails are visible.")
    parser.add_argument("--max-plot-previous-repeats", type=int, default=None,
                        help="Optional x-axis maximum for previous-repeat plots. If omitted, plot through the max post-lesion repeat count so post-only stutter tails are visible.")
    # Spectrogram cross-correlation to an early-phrase template.
    parser.add_argument("--corr-time-bins", type=int, default=64,
                        help="Number of time bins to time-normalize each segmented syllable before spectrogram correlation.")
    parser.add_argument("--corr-freq-min-khz", type=float, default=0.0,
                        help="Lower frequency bound used for spectrogram correlation.")
    parser.add_argument("--corr-freq-max-khz", type=float, default=None,
                        help="Upper frequency bound used for spectrogram correlation. Default: max_freq_khz.")
    parser.add_argument("--corr-transform", default="log_power", choices=["raw", "linear_power", "log_power"],
                        help="Transform used before spectrogram correlation. Default: log_power.")
    parser.add_argument("--phrase-template-mode", default="first_n", choices=["first_n", "first_seconds"],
                        help="How to choose the early-phrase segments used as the phrase template.")
    parser.add_argument("--phrase-template-first-n", type=int, default=3,
                        help="Number of early segmented syllables used for the phrase template when mode=first_n, and fallback for first_seconds.")
    parser.add_argument("--phrase-template-first-s", type=float, default=1.0,
                        help="Elapsed phrase time window used for the phrase template when mode=first_seconds.")
    parser.add_argument("--min-phrase-template-segments", type=int, default=2,
                        help="Minimum number of early phrase segments needed to build a template.")
    parser.add_argument("--leave-one-out-template-corr", action="store_true",
                        help=("For segments that are part of the early-template set, compute corr_to_phrase_early_template "
                              "against a template rebuilt without that same segment. This avoids self-correlation inflation "
                              "for the earliest syllables. Default keeps the historical behavior."))
    parser.add_argument("--lag-crosscorr-steps", default="10",
                        help=("Comma-separated previous-syllable lags for intra-phrase cross-correlation. "
                              "Default 10 means compare each syllable to the syllable 10 occurrences earlier "
                              "within the same repeated phrase."))
    parser.add_argument("--no-local-rate", action="store_true",
                        help=("Disable the short-lag local acoustic-change-rate analysis. By default this analysis "
                              "uses nearest-neighbor syllable pairs so short phrases are not censored by a fixed lag."))
    parser.add_argument("--local-rate-max-lag", type=int, default=1,
                        help=("Maximum within-phrase syllable lag used for local acoustic-change-rate pairs. "
                              "Default 1 uses only nearest-neighbor pairs. Values like 2 or 3 include all pairs "
                              "within that many repeats and also allow phrase-level regression of distance vs lag."))
    parser.add_argument("--min-local-rate-pairs-per-phrase", type=int, default=3,
                        help="Minimum short-lag pairs per phrase for phrase-level local-rate summaries. Default 3.")
    parser.add_argument("--qc-local-pair-count", type=int, default=8,
                        help=("Number of local short-lag pair QC spectrogram examples to save per label. "
                              "Set 0 to disable. Default 8."))
    parser.add_argument("--qc-local-pair-sample-mode", default="quantile_corr",
                        choices=["quantile_corr", "lowest_corr", "highest_corr", "random"],
                        help=("How to choose local short-lag pair QC examples. Default quantile_corr "
                              "spreads selections across the correlation distribution."))
    parser.add_argument("--qc-local-pairs-per-period", type=int, default=0,
                        help=("Optional number of QC pairs to sample separately from each period "
                              "(early_pre, late_pre, early_post, late_post). Default 0 ignores period stratification."))
    parser.add_argument("--qc-random-seed", type=int, default=0,
                        help="Random seed used for QC pair sampling when needed. Default 0.")
    parser.add_argument("--qc-plot-freq-min-khz", type=float, default=None,
                        help="Optional lower frequency bound for QC spectrogram displays. Default uses --corr-freq-min-khz.")
    parser.add_argument("--qc-plot-freq-max-khz", type=float, default=None,
                        help="Optional upper frequency bound for QC spectrogram displays. Default uses --corr-freq-max-khz.")
    parser.add_argument("--qc-plot-transform", default="log_power", choices=["raw", "linear_power", "log_power"],
                        help=("Display-only transform used for QC spectrograms. This does not change the correlation analysis. "
                              "Default log_power for a natural spectrogram appearance."))
    parser.add_argument("--qc-plot-cmap", default="Greys",
                        help="Matplotlib colormap used for QC spectrogram displays. Default Greys so high-amplitude regions are black and low-amplitude regions are white.")
    parser.add_argument("--qc-plot-dpi", type=int, default=250,
                        help="DPI used when saving QC PDF pages and PNGs. Default 250 for sharper rendering.")
    parser.add_argument("--qc-pad-ms", type=float, default=20.0,
                        help="Amount of time context to show before and after each QC syllable, in ms. Default 20 ms.")
    parser.add_argument("--qc-pair-shared-contrast", action="store_true", default=True,
                        help="Use one shared grayscale contrast range for both spectrograms in a QC pair. Default on.")
    parser.add_argument("--no-qc-pair-shared-contrast", dest="qc_pair_shared_contrast", action="store_false",
                        help="Disable shared contrast scaling across the two spectrograms in a QC pair.")
    parser.add_argument("--qc-match-pair-time-axis", action="store_true", default=True,
                        help="Use the same x-axis limit for both spectrograms in a QC pair so durations are visually comparable. Default on.")
    parser.add_argument("--no-qc-match-pair-time-axis", dest="qc_match_pair_time_axis", action="store_false",
                        help="Allow each QC spectrogram in a pair to use its own x-axis span.")
    parser.add_argument("--qc-global-pair-time-axis", action="store_true", default=True,
                        help="Use one common x-axis maximum across all isolated pair-QC figures within a label. Default on.")
    parser.add_argument("--no-qc-global-pair-time-axis", dest="qc_global_pair_time_axis", action="store_false",
                        help="Disable the shared x-axis maximum across all isolated pair-QC figures.")
    parser.add_argument("--qc-global-pair-xmax-s", type=float, default=None,
                        help="Optional manual x-axis maximum in seconds for all isolated pair-QC figures. Default: auto from the selected QC set.")
    parser.add_argument("--qc-global-phrase-time-axis", action="store_true", default=True,
                        help="Use one common x-axis maximum across all phrase-context QC figures within a label. Default on.")
    parser.add_argument("--no-qc-global-phrase-time-axis", dest="qc_global_phrase_time_axis", action="store_false",
                        help="Disable the shared x-axis maximum across all phrase-context QC figures.")
    parser.add_argument("--qc-global-phrase-xmax-s", type=float, default=None,
                        help="Optional manual x-axis maximum in seconds for all phrase-context QC figures. Default: auto from the selected QC phrases.")
    parser.add_argument("--qc-expanded-context", action="store_true", default=True,
                        help="Save row-style expanded phrase-context QC spectrograms with extra time before/after each selected phrase. Default on.")
    parser.add_argument("--no-qc-expanded-context", dest="qc_expanded_context", action="store_false",
                        help="Disable row-style expanded phrase-context QC spectrograms.")
    parser.add_argument("--qc-context-before-s", type=float, default=1.0,
                        help="Seconds of extra context to show before the target phrase in expanded context QC. Default 1.0.")
    parser.add_argument("--qc-context-after-s", type=float, default=1.0,
                        help="Seconds of extra context to show after the target phrase in expanded context QC. Default 1.0.")
    parser.add_argument("--qc-expanded-context-window-s", type=float, default=None,
                        help="Optional fixed total window length in seconds for expanded context QC, centered on the target phrase. Overrides before/after.")
    parser.add_argument("--qc-expanded-context-global-time-axis", action="store_true", default=True,
                        help="Use one common x-axis span across expanded context QC rows/pages. Default on.")
    parser.add_argument("--no-qc-expanded-context-global-time-axis", dest="qc_expanded_context_global_time_axis", action="store_false",
                        help="Allow each expanded context QC row to use its own x-axis span.")
    parser.add_argument("--qc-expanded-context-xmax-s", type=float, default=None,
                        help="Optional manual x-axis maximum in seconds for expanded context QC rows. Example: 5.4 for 2000 bins at 2.7 ms/bin.")
    parser.add_argument("--qc-expanded-context-rows-per-page", type=int, default=5,
                        help="Number of expanded context rows per PDF page. Default 5, similar to expanded full-run QC examples.")
    parser.add_argument("--qc-expanded-context-row-height", type=float, default=1.55,
                        help="Height in inches per expanded context row. Default 1.55.")
    parser.add_argument("--qc-expanded-context-fig-width", type=float, default=16.0,
                        help="Figure width in inches for expanded context QC pages. Default 16.")
    parser.add_argument("--qc-expanded-show-all-boundaries", action="store_true", default=True,
                        help="Overlay all target-syllable segment boundaries in expanded context QC. Default on.")
    parser.add_argument("--no-qc-expanded-show-all-boundaries", dest="qc_expanded_show_all_boundaries", action="store_false",
                        help="Only show highlighted pair boundaries in expanded context QC.")
    parser.add_argument("--qc-expanded-label-fontsize", type=float, default=7,
                        help="Repeat-number label font size for expanded context QC. Default 7.")
    parser.add_argument("--qc-phrase-span-alpha", type=float, default=0.0,
                        help="Optional gray shading alpha for the full target phrase span in expanded context QC. Default 0.")
    parser.add_argument("--qc-contrast-low-pct", type=float, default=5.0,
                        help="Lower percentile used for QC grayscale contrast scaling. Default 5 for softer background suppression.")
    parser.add_argument("--qc-contrast-high-pct", type=float, default=99.8,
                        help="Upper percentile used for QC grayscale contrast scaling. Default 99.8 for smoother high-amplitude detail.")
    parser.add_argument("--qc-log-floor-pct", type=float, default=1.0,
                        help="Positive-power percentile used as the display floor before log10 for QC spectrograms. Default 1.")
    parser.add_argument("--qc-display-smooth-time-bins", type=int, default=1,
                        help="Optional moving-average smoothing window in time bins for QC display only. Default 1 disables smoothing.")
    parser.add_argument("--qc-display-smooth-freq-bins", type=int, default=1,
                        help="Optional moving-average smoothing window in frequency bins for QC display only. Default 1 disables smoothing.")
    parser.add_argument("--qc-pair-boundary-alpha", type=float, default=0.55,
                        help="Alpha for red syllable boundary lines in isolated pair QC plots. Default 0.55.")
    parser.add_argument("--qc-pair-boundary-lw", type=float, default=0.9,
                        help="Line width for red syllable boundary lines in isolated pair QC plots. Default 0.9.")
    parser.add_argument("--qc-context-boundary-alpha", type=float, default=0.35,
                        help="Alpha for non-highlighted segment boundaries in phrase-context QC plots. Default 0.35.")
    parser.add_argument("--qc-context-boundary-lw", type=float, default=0.7,
                        help="Line width for non-highlighted segment boundaries in phrase-context QC plots. Default 0.7.")
    parser.add_argument("--qc-highlight-span-alpha", type=float, default=0.06,
                        help="Alpha for the highlighted pair span in phrase-context QC plots. Set 0 to remove filled highlight. Default 0.06.")
    parser.add_argument("--qc-highlight-boundary-alpha", type=float, default=0.65,
                        help="Alpha for highlighted pair boundary lines in phrase-context QC plots. Default 0.65.")
    parser.add_argument("--qc-highlight-boundary-lw", type=float, default=1.0,
                        help="Line width for highlighted pair boundary lines in phrase-context QC plots. Default 1.0.")
    parser.add_argument("--qc-label-box-alpha", type=float, default=0.35,
                        help="Alpha for white boxes behind repeat-number labels in phrase-context QC plots. Default 0.35.")
    args = parser.parse_args()
    args.lag_crosscorr_steps = [int(x.strip()) for x in str(args.lag_crosscorr_steps).split(",") if x.strip()]
    if len(args.lag_crosscorr_steps) == 0:
        args.lag_crosscorr_steps = [10]
    if any(x <= 0 for x in args.lag_crosscorr_steps):
        raise ValueError("--lag-crosscorr-steps must contain positive integers, e.g. 10 or 5,10")
    args.lag_crosscorr_steps = sorted(set(args.lag_crosscorr_steps))
    if int(args.local_rate_max_lag) <= 0:
        raise ValueError("--local-rate-max-lag must be a positive integer, e.g. 1, 2, or 3")
    if int(args.min_local_rate_pairs_per_phrase) <= 0:
        raise ValueError("--min-local-rate-pairs-per-phrase must be positive")

    if not SCIPY_AVAILABLE:
        raise ImportError("This script requires scipy for canary segmentation and statistics. Install with: conda install scipy")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    labels = parse_label_list(args)
    print(f"[INFO] Labels to analyze for {args.animal_id}: {', '.join(labels)}")

    all_segments = []
    all_stats = []
    all_early_late = []
    all_local_pairs = []
    all_local_rate_stats = []
    all_local_phrase_rates = []
    failed_labels = []
    for lab in labels:
        label_out = out_dir / f"{args.animal_id}_label{str(lab).replace('.', 'p')}"
        try:
            seg_df, stats_df, early_late_df, local_pair_df, local_rate_stats_df, local_phrase_rate_df = analyze_label_phrase_position(args, lab, label_out)
        except Exception as exc:
            print(f"[ERROR] Label {lab} failed for {args.animal_id}: {type(exc).__name__}: {exc}")
            failed_labels.append({
                "animal_id": args.animal_id,
                "label": lab,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            continue
        if not seg_df.empty:
            all_segments.append(seg_df)
        if not stats_df.empty:
            all_stats.append(stats_df)
        if not early_late_df.empty:
            all_early_late.append(early_late_df)
        if not local_pair_df.empty:
            all_local_pairs.append(local_pair_df)
        if not local_rate_stats_df.empty:
            all_local_rate_stats.append(local_rate_stats_df)
        if not local_phrase_rate_df.empty:
            all_local_phrase_rates.append(local_phrase_rate_df)

    if all_segments:
        combined_segments = pd.concat(all_segments, ignore_index=True)
        combined_segments.to_csv(out_dir / f"{args.animal_id}_all_labels_phrase_position_segment_features.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_phrase_position_segment_features.csv'}")
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        combined_stats.to_csv(out_dir / f"{args.animal_id}_all_labels_phrase_position_stats.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_phrase_position_stats.csv'}")
    if all_early_late:
        combined_early_late = pd.concat(all_early_late, ignore_index=True)
        combined_early_late.to_csv(out_dir / f"{args.animal_id}_all_labels_early_late_window_summary.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_early_late_window_summary.csv'}")
    if all_local_pairs:
        combined_local_pairs = pd.concat(all_local_pairs, ignore_index=True)
        combined_local_pairs.to_csv(out_dir / f"{args.animal_id}_all_labels_local_shortlag_pairs.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_local_shortlag_pairs.csv'}")
    if all_local_rate_stats:
        combined_local_rate_stats = pd.concat(all_local_rate_stats, ignore_index=True)
        combined_local_rate_stats.to_csv(out_dir / f"{args.animal_id}_all_labels_local_rate_position_stats.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_local_rate_position_stats.csv'}")
    if all_local_phrase_rates:
        combined_local_phrase_rates = pd.concat(all_local_phrase_rates, ignore_index=True)
        combined_local_phrase_rates.to_csv(out_dir / f"{args.animal_id}_all_labels_local_shortlag_phrase_rate_summary.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_local_shortlag_phrase_rate_summary.csv'}")

    if failed_labels:
        failed_df = pd.DataFrame(failed_labels)
        failed_csv = out_dir / f"{args.animal_id}_failed_labels.csv"
        failed_df.to_csv(failed_csv, index=False)
        print(f"[SAVED] {failed_csv}")

    readme = out_dir / "README_phrase_position_acoustic_analysis.txt"
    readme.write_text(
        "Phrase-position acoustic analysis outputs.\n\n"
        "Interpretation:\n"
        "- feature_vs_elapsed_time: tests whether acoustic features change as seconds elapse within a repeated phrase.\n"
        "- feature_vs_previous_repeats: tests whether features change as more syllables have preceded the current syllable.\n"
        "- early_vs_late_windows: compares each long phrase's early window to a later window.\n"
        "- slope_summary_by_period: shows linear slopes of feature vs elapsed phrase time for early/late pre/post.\n"
        "- local_shortlag_pairs: one row per nearby syllable pair. The default nearest-neighbor version avoids the lag-10 censoring problem because early syllables contribute through forward pairs.\n"
        "- local_rate_vs_elapsed_time / previous_repeats: plots d or d² per repeat against the pair midpoint to test whether acoustic change accelerates across repetitions.\n\n"
        "Caveat: phrase_id is defined as a contiguous target-label run. With --stitch-across-file-segments, consecutive _segment_N chunks from the same original recording are stitched before phrase detection. If decoder labels briefly interrupt a phrase, try --max-gap-bins-to-merge.\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE]")


if __name__ == "__main__":
    main()
