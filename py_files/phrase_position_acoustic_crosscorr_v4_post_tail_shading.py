#!/usr/bin/env python3
"""
phrase_position_acoustic_crosscorr_v4_post_tail_shading.py

Analyze whether acoustic structure changes as a repeated syllable phrase progresses.

Version 4 keeps the v3 fixes and adds post-lesion-only tail shading/markers to phrase-position plots. The pre line stops where pre data stop, while the post line can continue into stutter-induced longer durations/repeat counts.

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

Notes:
- This script treats a "phrase" as a contiguous run of the target cluster label within one file.
- If your decoder briefly interrupts a phrase with small gaps/noise labels, use --max-gap-bins-to-merge.
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
}


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_values(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def normalize_label_array(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).astype(str)


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
        entry = file_map.get(fidx, file_map.get(str(fidx), None))
        if entry is None:
            raise KeyError(f"file index {fidx} not found in file_map")
        fname = file_entry_to_name(entry)
        lookup[fidx] = parse_datetime_from_filename(fname)
    return lookup


def find_phrase_runs_for_label(labels: np.ndarray, file_indices: np.ndarray, target_label: str,
                               period_files: Iterable[int], min_label_run_bins: int,
                               max_gap_bins_to_merge: int) -> List[Dict[str, int]]:
    runs = []
    period_files = set(int(x) for x in period_files)
    for fidx in sorted(period_files):
        file_mask = file_indices == int(fidx)
        idx = np.flatnonzero(file_mask)
        if idx.size == 0:
            continue
        local_labels = labels[idx]
        target_mask = local_labels == str(target_label)
        target_mask = fill_small_false_gaps(target_mask, max_gap_bins_to_merge)
        for local_start, local_end in finite_runs(target_mask):
            run_len = local_end - local_start
            if run_len < min_label_run_bins:
                continue
            global_start = int(idx[local_start])
            global_end = int(idx[local_end - 1]) + 1
            runs.append({
                "file_index": int(fidx),
                "global_start_bin": global_start,
                "global_end_bin": global_end,
                "local_start_bin": int(local_start),
                "local_end_bin": int(local_end),
                "n_bins": int(global_end - global_start),
            })
    return runs


def analyze_label_phrase_position(args, cluster_label: str, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    all_rows = []
    phrase_counter = 0
    diagnostics_rows = []

    for period in PERIOD_ORDER:
        runs = find_phrase_runs_for_label(
            labels,
            file_indices,
            str(cluster_label),
            period_files[period],
            min_label_run_bins=args.min_label_run_bins,
            max_gap_bins_to_merge=args.max_gap_bins_to_merge,
        )
        print(f"[INFO] {period}: {len(runs)} phrase-label runs for label {cluster_label}")
        for run_idx, run in enumerate(runs):
            phrase_counter += 1
            g0 = int(run["global_start_bin"])
            g1 = int(run["global_end_bin"])
            X_run = X[g0:g1, :]
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
            phrase_duration_s = (g1 - g0) * bin_s
            diagnostics_rows.append({
                "animal_id": args.animal_id,
                "label": cluster_label,
                "period": period,
                "phrase_id": phrase_counter,
                "file_index": run["file_index"],
                "global_start_bin": g0,
                "global_end_bin": g1,
                "phrase_duration_s": phrase_duration_s,
                "n_bins": g1 - g0,
                "n_segments": n_segments,
                "n_diagnostics_chunks": len(seg.get("diagnostics", [])),
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
                corr_to_template = corr_between_unit_vectors(info.get("corr_vec"), phrase_template_vec)
                distance_from_template = 1.0 - corr_to_template if np.isfinite(corr_to_template) else np.nan
                all_rows.append({
                    "animal_id": args.animal_id,
                    "label": str(cluster_label),
                    "period": period,
                    "pre_post": PRE_POST_MAP[period],
                    "phrase_id": phrase_counter,
                    "file_index": int(run["file_index"]),
                    "file_datetime": file_time_lookup[int(run["file_index"])],
                    "phrase_global_start_bin": g0,
                    "phrase_global_end_bin": g1,
                    "phrase_duration_s": phrase_duration_s,
                    "phrase_n_bins": g1 - g0,
                    "n_segments_in_phrase": n_segments,
                    "repeat_index_in_phrase": seg_order,
                    "n_previous_segments": seg_order - 1,
                    "elapsed_time_in_phrase_s": elapsed_mid,
                    "segment_start_elapsed_s": elapsed_start,
                    "segment_end_elapsed_s": elapsed_end,
                    "fraction_through_phrase": elapsed_mid / phrase_duration_s if phrase_duration_s > 0 else np.nan,
                    "segment_start_global_bin": g0 + s0,
                    "segment_end_global_bin": g0 + s1,
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
                    "corr_to_phrase_early_template": corr_to_template,
                    "distance_from_phrase_early_template": distance_from_template,
                    "dominant_pitch_modulation_freq_hz": rec.get("dominant_pitch_modulation_freq_hz", np.nan),
                })

    segment_df = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(diagnostics_rows)
    prefix = f"{args.animal_id}_label{str(cluster_label).replace('.', 'p')}"
    seg_csv = out_dir / f"{prefix}_phrase_position_segment_features.csv"
    diag_csv = out_dir / f"{prefix}_phrase_position_phrase_diagnostics.csv"
    segment_df.to_csv(seg_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)
    print(f"[SAVED] {seg_csv}")
    print(f"[SAVED] {diag_csv}")

    if segment_df.empty:
        print(f"[WARN] No segmented syllables found for label {cluster_label}; skipping plots.")
        return segment_df, pd.DataFrame(), pd.DataFrame()

    features = [x.strip() for x in args.features.split(",") if x.strip()]
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

    corr_features = ["corr_to_phrase_early_template", "distance_from_phrase_early_template"]
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

    # Return acoustic and cross-correlation stats together for the combined all-label table.
    combined_stats_for_return = pd.concat([stats_df, corr_stats_df], ignore_index=True) if not corr_stats_df.empty else stats_df
    return segment_df, combined_stats_for_return, early_late_df


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
                    "feature_label": FEATURE_LABELS.get(feature, feature),
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
                "feature_label": FEATURE_LABELS.get(feature, feature),
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
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        ax.set_title(FEATURE_LABELS.get(feature, feature))
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
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        ax.set_title(FEATURE_LABELS.get(feature, feature))
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
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        ax.set_title(FEATURE_LABELS.get(feature, feature))
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
        ax.set_title(FEATURE_LABELS.get(feature, feature))
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
        labels.extend([x.strip() for x in str(args.cluster_label).split(",") if x.strip()])
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
        labels.extend(df[lab_col].dropna().astype(str).tolist())
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
                        help="Merge target-label runs across small non-target gaps within a file. Default 0.")
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
    args = parser.parse_args()

    if not SCIPY_AVAILABLE:
        raise ImportError("This script requires scipy for canary segmentation and statistics. Install with: conda install scipy")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    labels = parse_label_list(args)
    print(f"[INFO] Labels to analyze for {args.animal_id}: {', '.join(labels)}")

    all_segments = []
    all_stats = []
    all_early_late = []
    failed_labels = []
    for lab in labels:
        label_out = out_dir / f"{args.animal_id}_label{str(lab).replace('.', 'p')}"
        try:
            seg_df, stats_df, early_late_df = analyze_label_phrase_position(args, lab, label_out)
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
        "- slope_summary_by_period: shows linear slopes of feature vs elapsed phrase time for early/late pre/post.\n\n"
        "Caveat: phrase_id is defined as a contiguous target-label run within a file. If decoder labels briefly interrupt a phrase, try --max-gap-bins-to-merge.\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE]")


if __name__ == "__main__":
    main()
