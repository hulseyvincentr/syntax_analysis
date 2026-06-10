#!/usr/bin/env python3
"""
Analyze syllable duration and inter-syllable interval (ISI) within repeated syllable phrases.

This is a lighter timing-focused companion to the phrase-position acoustic/cross-correlation
scripts. It keeps the same basic processing ideas:

1) load a TweetyBERT / HDBSCAN-style NPZ with spectrogram, labels, file indices, and file_map
2) optionally smooth the decoder labels with a centered majority vote, applied within each file
3) find contiguous phrase-label runs for each target syllable label
4) optionally stitch chunked file_map entries such as _segment_0, _segment_1, ... before finding runs
5) segment each repeated-label run using the canary power-minima method
6) measure timing for each segmented syllable rendition:
      - syllable duration
      - preceding inter-syllable interval (ISI; gap from previous segment end to current start)
      - following ISI
      - cumulative phrase time through each repeat
      - cumulative syllable/vocalization time through each repeat
7) save CSVs and make plots:
      - duration and ISI vs repeat index in the phrase
      - time spent repeating the syllable in the phrase

Important interpretation:
- repeat_index_in_phrase starts at 1 for the first segmented syllable in a phrase.
- preceding_isi_s is NaN for repeat 1 because there is no previous segmented syllable.
- cumulative_phrase_time_s includes syllable durations plus gaps/ISIs from phrase start to that repeat's end.
- cumulative_syllable_duration_s includes only summed segmented syllable durations up to that repeat.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter
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
                               stitch_across_file_segments: bool = False) -> List[Dict[str, int]]:
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
            target_mask = local_labels == str(target_label)
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
                    "recording_key": file_entry_to_name(file_map[int(fidx)]) if file_map is not None and int(fidx) in file_map else str(fidx),
                    "global_start_bin": global_start,
                    "global_end_bin": global_end,
                    "local_start_bin": int(local_start),
                    "local_end_bin": int(local_end),
                    "n_bins": int(len(run_indices)),
                    "global_indices": run_indices.astype(int),
                })
        return runs

    if file_map is None:
        raise ValueError("stitch_across_file_segments=True requires file_map.")

    groups: Dict[str, List[Tuple[int, int]]] = {}
    for fidx in sorted(period_files):
        try:
            fname = file_entry_to_name(file_map[int(fidx)])
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
        target_mask = local_labels == str(target_label)
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
                "global_indices": run_indices.astype(int),
            })

    return runs


TIMING_FEATURE_LABELS = {
    "segment_duration_s": "Syllable duration (s)",
    "segment_duration_ms": "Syllable duration (ms)",
    "preceding_isi_s": "Preceding inter-syllable interval (s)",
    "preceding_isi_ms": "Preceding inter-syllable interval (ms)",
    "following_isi_s": "Following inter-syllable interval (s)",
    "following_isi_ms": "Following inter-syllable interval (ms)",
    "cumulative_phrase_time_s": "Cumulative phrase time through this repeat (s)",
    "cumulative_syllable_duration_s": "Cumulative syllable duration through this repeat (s)",
    "phrase_duration_s": "Total time spent in repeated-label phrase (s)",
}


def parse_label_set(text: str) -> set[str]:
    if text is None:
        return set()
    out = set()
    for item in str(text).split(','):
        s = item.strip()
        if s:
            out.add(s)
    return out


def majority_vote_smooth_labels(
    labels: np.ndarray,
    file_indices: np.ndarray,
    window_bins: int = 7,
    min_fraction: float = 0.50,
    ignore_labels: Optional[set[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Centered majority-vote smoothing of decoder labels within each file_index.

    Ties keep the original label. If ignore_labels is provided, those labels are ignored
    when choosing the majority label, unless the whole window would be ignored.
    """
    labels = np.asarray(labels).astype(str)
    file_indices = np.asarray(file_indices).astype(int)
    ignore_labels = set(ignore_labels or [])

    if window_bins is None or int(window_bins) <= 1:
        summary = pd.DataFrame([{
            "majority_vote_enabled": False,
            "window_bins": int(window_bins or 1),
            "min_fraction": float(min_fraction),
            "n_bins": int(labels.size),
            "n_bins_changed": 0,
            "fraction_bins_changed": 0.0,
        }])
        return labels.copy(), summary

    window_bins = int(window_bins)
    if window_bins % 2 == 0:
        window_bins += 1
        print(f"[INFO] Majority-vote window was even; using next odd value: {window_bins}")
    half = window_bins // 2
    min_fraction = float(min_fraction)

    out = labels.copy()
    change_rows = []

    for fidx in np.unique(file_indices):
        idx = np.flatnonzero(file_indices == int(fidx))
        if idx.size == 0:
            continue
        local = labels[idx]
        for i in range(local.size):
            lo = max(0, i - half)
            hi = min(local.size, i + half + 1)
            vals = list(local[lo:hi])
            usable = [v for v in vals if v not in ignore_labels]
            if len(usable) == 0:
                usable = vals
            counts = Counter(usable)
            if len(counts) == 0:
                continue
            most_common = counts.most_common()
            top_label, top_count = most_common[0]
            # Tie: keep the original label.
            if len(most_common) > 1 and most_common[1][1] == top_count:
                continue
            if top_count / max(1, len(usable)) >= min_fraction:
                out[idx[i]] = str(top_label)

    changed = labels != out
    for old, new in sorted(set(zip(labels[changed], out[changed])), key=lambda x: (x[0], x[1])):
        mask = changed & (labels == old) & (out == new)
        change_rows.append({
            "majority_vote_enabled": True,
            "window_bins": int(window_bins),
            "min_fraction": min_fraction,
            "original_label": old,
            "smoothed_label": new,
            "n_bins_changed": int(np.sum(mask)),
        })
    if not change_rows:
        change_rows.append({
            "majority_vote_enabled": True,
            "window_bins": int(window_bins),
            "min_fraction": min_fraction,
            "original_label": "",
            "smoothed_label": "",
            "n_bins_changed": 0,
        })
    summary = pd.DataFrame(change_rows)
    summary["n_bins_total"] = int(labels.size)
    summary["n_bins_changed_total"] = int(np.sum(changed))
    summary["fraction_bins_changed_total"] = float(np.mean(changed)) if labels.size else np.nan
    return out, summary


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


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
            for candidate in ["label", "syllable_label", "cluster_label", "Syllable", "Syllable label", "mapped_syllable_order", "hdbscan_label"]:
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


def _safe_spearman_and_linear(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> Dict[str, float | str]:
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
    if xv.size:
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


def compute_timing_stats(df: pd.DataFrame, features: List[str], min_n: int = 5) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        for group_col, groups in [("period", PERIOD_ORDER), ("pre_post", ["pre", "post"]), ("label", sorted(df["label"].astype(str).unique()))]:
            for group in groups:
                sub = df[df[group_col].astype(str) == str(group)].copy()
                if sub.empty:
                    continue
                x = sub["repeat_index_in_phrase"].to_numpy(dtype=float)
                y = sub[feature].to_numpy(dtype=float)
                fit = _safe_spearman_and_linear(x, y, min_n=min_n)
                rows.append({
                    "feature": feature,
                    "feature_label": TIMING_FEATURE_LABELS.get(feature, feature),
                    "group_col": group_col,
                    "group": group,
                    "n_valid": int(fit["n_valid"]),
                    "n_unique_repeat_index": int(fit["n_unique_x"]),
                    "n_unique_y": int(fit["n_unique_y"]),
                    "fit_status": fit["fit_status"],
                    "spearman_r_vs_repeat_index": fit["spearman_r"],
                    "spearman_p_vs_repeat_index": fit["spearman_p"],
                    "linear_slope_per_repeat": fit["linear_slope"],
                    "linear_intercept": fit["linear_intercept"],
                    "linear_r": fit["linear_r"],
                    "linear_p": fit["linear_p"],
                    "median_value": float(np.nanmedian(y[np.isfinite(y)])) if np.any(np.isfinite(y)) else np.nan,
                    "mean_value": float(np.nanmean(y[np.isfinite(y)])) if np.any(np.isfinite(y)) else np.nan,
                })
    return pd.DataFrame(rows)


def _median_iqr_by_repeat(sub: pd.DataFrame, feature: str, min_n: int) -> pd.DataFrame:
    rows = []
    for rep, srep in sub.groupby("repeat_index_in_phrase"):
        vals = pd.to_numeric(srep[feature], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            continue
        rows.append({
            "repeat_index_in_phrase": int(rep),
            "n": int(vals.size),
            "median": float(np.nanmedian(vals)),
            "q25": float(np.nanpercentile(vals, 25)),
            "q75": float(np.nanpercentile(vals, 75)),
        })
    return pd.DataFrame(rows).sort_values("repeat_index_in_phrase") if rows else pd.DataFrame()


def _plot_feature_vs_repeat(ax, df: pd.DataFrame, feature: str, group_col: str, group_order: List[str],
                            y_label: str, min_bin_n: int, max_repeat: Optional[int] = None):
    use = df.copy()
    if max_repeat is not None:
        use = use[pd.to_numeric(use["repeat_index_in_phrase"], errors="coerce") <= max_repeat].copy()
    for group in group_order:
        sub = use[use[group_col].astype(str) == str(group)].copy()
        if sub.empty or feature not in sub.columns:
            continue
        x = pd.to_numeric(sub["repeat_index_in_phrase"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[feature], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            ax.scatter(x[valid], y[valid], s=12, alpha=0.18, label=f"{group} segments")
        b = _median_iqr_by_repeat(sub, feature, min_n=min_bin_n)
        if not b.empty:
            xx = b["repeat_index_in_phrase"].to_numpy(dtype=float)
            med = b["median"].to_numpy(dtype=float)
            q25 = b["q25"].to_numpy(dtype=float)
            q75 = b["q75"].to_numpy(dtype=float)
            ax.plot(xx, med, marker="o", linewidth=2.5, label=f"{group} median")
            ax.fill_between(xx, q25, q75, alpha=0.15, label=f"{group} IQR")
    ax.set_xlabel("Syllable repeat index within phrase")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep_h = []
    keep_l = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            keep_h.append(h)
            keep_l.append(lab)
            seen.add(lab)
    if keep_h:
        ax.legend(keep_h, keep_l, frameon=True, fontsize=8)


def make_duration_isi_vs_repeat_plot(df: pd.DataFrame, out_png: Path, title: str,
                                     group_col: str = "pre_post", min_bin_n: int = 3,
                                     max_repeat: Optional[int] = None):
    if df.empty:
        return
    if group_col == "period":
        group_order = [g for g in PERIOD_ORDER if g in set(df["period"].astype(str))]
    else:
        group_order = [g for g in ["pre", "post"] if g in set(df["pre_post"].astype(str))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    _plot_feature_vs_repeat(
        axes[0, 0], df, "segment_duration_ms", group_col, group_order,
        "Syllable duration (ms)", min_bin_n=min_bin_n, max_repeat=max_repeat,
    )
    axes[0, 0].set_title("Syllable duration vs repeat index")
    _plot_feature_vs_repeat(
        axes[0, 1], df, "preceding_isi_ms", group_col, group_order,
        "Preceding inter-syllable interval (ms)", min_bin_n=min_bin_n, max_repeat=max_repeat,
    )
    axes[0, 1].set_title("ISI before current syllable vs repeat index")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_time_spent_repeating_plot(df: pd.DataFrame, out_png: Path, title: str,
                                   group_col: str = "pre_post", min_bin_n: int = 3,
                                   max_repeat: Optional[int] = None):
    if df.empty:
        return
    if group_col == "period":
        group_order = [g for g in PERIOD_ORDER if g in set(df["period"].astype(str))]
    else:
        group_order = [g for g in ["pre", "post"] if g in set(df["pre_post"].astype(str))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), squeeze=False)
    _plot_feature_vs_repeat(
        axes[0, 0], df, "cumulative_phrase_time_s", group_col, group_order,
        "Cumulative phrase time through repeat (s)", min_bin_n=min_bin_n, max_repeat=max_repeat,
    )
    axes[0, 0].set_title("Time from phrase start to each repeat ending")

    # One row per phrase for total phrase duration vs total number of repeats.
    phrase_df = (
        df.groupby(["animal_id", "label", "period", "pre_post", "phrase_id"], as_index=False)
        .agg(
            phrase_duration_s=("phrase_duration_s", "first"),
            n_segments_in_phrase=("n_segments_in_phrase", "first"),
        )
    )
    for group in group_order:
        sub = phrase_df[phrase_df[group_col].astype(str) == str(group)]
        if sub.empty:
            continue
        x = pd.to_numeric(sub["n_segments_in_phrase"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["phrase_duration_s"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            axes[0, 1].scatter(x[valid], y[valid], s=18, alpha=0.25, label=f"{group} phrases")
        b = _median_iqr_by_repeat(
            sub.rename(columns={"n_segments_in_phrase": "repeat_index_in_phrase"}),
            "phrase_duration_s",
            min_n=min_bin_n,
        )
        if not b.empty:
            xx = b["repeat_index_in_phrase"].to_numpy(dtype=float)
            med = b["median"].to_numpy(dtype=float)
            axes[0, 1].plot(xx, med, marker="o", linewidth=2.5, label=f"{group} median")
    axes[0, 1].set_xlabel("Total segmented repeats in phrase")
    axes[0, 1].set_ylabel("Total repeated-label phrase duration (s)")
    axes[0, 1].set_title("Total time spent repeating syllable in each phrase")
    axes[0, 1].grid(alpha=0.25)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, frameon=True, fontsize=8)

    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def analyze_label_timing(args, cluster_label: str, out_dir: Path, loaded=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    if loaded is None:
        print(f"[INFO] Loading NPZ for label {cluster_label}...")
        data = np.load(args.npz_path, allow_pickle=True)
        X = np.asarray(data[args.spec_key], dtype=float)
        labels_raw = normalize_label_array(data[args.label_key])
        file_indices = np.asarray(data[args.file_index_key]).astype(int)
        file_map = get_file_map_dict(data[args.file_map_key])
    else:
        X, labels_raw, file_indices, file_map = loaded

    if X.shape[0] != labels_raw.shape[0]:
        if X.ndim == 2 and X.shape[1] == labels_raw.shape[0]:
            print("[INFO] Spectrogram appears transposed; transposing to time x frequency.")
            X = X.T
        else:
            raise ValueError(f"Spectrogram first dimension {X.shape[0]} does not match labels length {labels_raw.shape[0]}")

    ignore_labels = parse_label_set(args.majority_vote_ignore_labels)
    if args.disable_majority_vote:
        labels = labels_raw.copy()
        mv_summary = pd.DataFrame([{
            "majority_vote_enabled": False,
            "window_bins": 1,
            "min_fraction": np.nan,
            "n_bins_total": int(labels_raw.size),
            "n_bins_changed_total": 0,
            "fraction_bins_changed_total": 0.0,
        }])
    else:
        labels, mv_summary = majority_vote_smooth_labels(
            labels_raw,
            file_indices,
            window_bins=args.majority_vote_window_bins,
            min_fraction=args.majority_vote_min_fraction,
            ignore_labels=ignore_labels,
        )

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
    phrase_rows = []
    phrase_counter = 0

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
        )
        if runs:
            max_run_bins = max(int(r.get("n_bins", 0)) for r in runs)
            max_run_files = max(int(r.get("n_files_stitched", 1)) for r in runs)
            print(
                f"[INFO] {period}: {len(runs)} phrase-label runs for label {cluster_label}; "
                f"max run duration={max_run_bins * bin_s:.2f}s; max stitched files={max_run_files}"
            )
        else:
            print(f"[INFO] {period}: 0 phrase-label runs for label {cluster_label}")

        for run in runs:
            phrase_counter += 1
            run_indices = np.asarray(run.get("global_indices", np.arange(int(run["global_start_bin"]), int(run["global_end_bin"]))), dtype=int)
            if run_indices.size == 0:
                continue
            g0 = int(run_indices[0])
            g1 = int(run_indices[-1]) + 1
            X_run = X[run_indices, :]
            if X_run.shape[0] < args.min_label_run_bins:
                continue

            # We still compute pitch because the canary segmentation method uses the pitch/power rhythm.
            pitch_khz, _pitch_deriv = compute_pitch_trace_and_derivative(
                X_run,
                bin_s=bin_s,
                max_freq_khz=args.max_freq_khz,
                pitch_band_min_khz=args.pitch_band_min_khz,
                pitch_band_max_khz=args.pitch_band_max_khz,
                spec_scale=args.spec_scale,
                smoothing_window=args.pitch_smoothing_window,
                min_peak_fraction=args.pitch_min_peak_fraction,
            )
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
            segments = sorted(seg["segments"], key=lambda r: int(r["segment_start_bin"]))
            n_segments = len(segments)
            phrase_n_bins = int(run_indices.size)
            phrase_duration_s = phrase_n_bins * bin_s

            phrase_rows.append({
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
                "phrase_n_bins": phrase_n_bins,
                "phrase_duration_s": phrase_duration_s,
                "n_segments_in_phrase": n_segments,
                "n_segmentation_diagnostic_chunks": len(seg.get("diagnostics", [])),
            })
            if n_segments == 0:
                continue

            cumulative_syllable_duration_s = 0.0
            for idx_seg, rec in enumerate(segments):
                s0 = int(rec["segment_start_bin"])
                s1 = int(rec["segment_end_bin"])
                if s1 <= s0:
                    continue
                prev_rec = segments[idx_seg - 1] if idx_seg > 0 else None
                next_rec = segments[idx_seg + 1] if idx_seg < (len(segments) - 1) else None
                segment_duration_s = (s1 - s0) * bin_s
                cumulative_syllable_duration_s += segment_duration_s
                segment_start_elapsed_s = s0 * bin_s
                segment_end_elapsed_s = s1 * bin_s
                segment_mid_elapsed_s = 0.5 * (segment_start_elapsed_s + segment_end_elapsed_s)

                preceding_isi_s = np.nan
                if prev_rec is not None:
                    preceding_isi_s = max(0.0, (s0 - int(prev_rec["segment_end_bin"])) * bin_s)
                following_isi_s = np.nan
                if next_rec is not None:
                    following_isi_s = max(0.0, (int(next_rec["segment_start_bin"]) - s1) * bin_s)

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
                    "file_datetime": file_time_lookup[int(run["file_index"])],
                    "phrase_global_start_bin": g0,
                    "phrase_global_end_bin": g1,
                    "phrase_n_bins": phrase_n_bins,
                    "phrase_duration_s": phrase_duration_s,
                    "n_segments_in_phrase": n_segments,
                    "repeat_index_in_phrase": idx_seg + 1,
                    "n_previous_segments": idx_seg,
                    "segment_start_elapsed_s": segment_start_elapsed_s,
                    "segment_end_elapsed_s": segment_end_elapsed_s,
                    "elapsed_time_in_phrase_s": segment_mid_elapsed_s,
                    "cumulative_phrase_time_s": segment_end_elapsed_s,
                    "cumulative_syllable_duration_s": cumulative_syllable_duration_s,
                    "fraction_through_phrase": segment_mid_elapsed_s / phrase_duration_s if phrase_duration_s > 0 else np.nan,
                    "segment_start_global_bin": int(run_indices[s0]) if s0 < len(run_indices) else np.nan,
                    "segment_end_global_bin": int(run_indices[s1 - 1]) + 1 if (s1 - 1) < len(run_indices) and s1 > 0 else np.nan,
                    "segment_duration_s": segment_duration_s,
                    "segment_duration_ms": segment_duration_s * 1000.0,
                    "preceding_isi_s": preceding_isi_s,
                    "preceding_isi_ms": preceding_isi_s * 1000.0 if np.isfinite(preceding_isi_s) else np.nan,
                    "following_isi_s": following_isi_s,
                    "following_isi_ms": following_isi_s * 1000.0 if np.isfinite(following_isi_s) else np.nan,
                    "dominant_pitch_modulation_freq_hz": rec.get("dominant_pitch_modulation_freq_hz", np.nan),
                    "segmentation_method": rec.get("segmentation_method", ""),
                    "dropped_edge_segments": bool(rec.get("dropped_edge_segments", False)),
                    "majority_vote_window_bins": 1 if args.disable_majority_vote else int(args.majority_vote_window_bins),
                    "majority_vote_min_fraction": np.nan if args.disable_majority_vote else float(args.majority_vote_min_fraction),
                })

    segment_df = pd.DataFrame(all_rows)
    phrase_df = pd.DataFrame(phrase_rows)
    prefix = f"{args.animal_id}_label{safe_name(cluster_label)}"

    seg_csv = out_dir / f"{prefix}_duration_isi_segment_timing.csv"
    phrase_csv = out_dir / f"{prefix}_duration_isi_phrase_summary.csv"
    mv_csv = out_dir / f"{prefix}_majority_vote_summary.csv"
    stats_csv = out_dir / f"{prefix}_duration_isi_stats.csv"

    segment_df.to_csv(seg_csv, index=False)
    phrase_df.to_csv(phrase_csv, index=False)
    mv_summary.to_csv(mv_csv, index=False)
    print(f"[SAVED] {seg_csv}")
    print(f"[SAVED] {phrase_csv}")
    print(f"[SAVED] {mv_csv}")

    if segment_df.empty:
        print(f"[WARN] No segmented syllables found for label {cluster_label}; skipping plots and stats.")
        return segment_df, phrase_df, pd.DataFrame()

    timing_features = ["segment_duration_ms", "preceding_isi_ms", "following_isi_ms", "cumulative_phrase_time_s", "cumulative_syllable_duration_s"]
    stats_df = compute_timing_stats(segment_df, timing_features, min_n=args.min_values_per_group)
    stats_df.to_csv(stats_csv, index=False)
    print(f"[SAVED] {stats_csv}")

    make_duration_isi_vs_repeat_plot(
        segment_df,
        out_png=out_dir / f"{prefix}_duration_and_isi_vs_repeat_index.png",
        title=f"{args.animal_id} label {cluster_label}: syllable duration and ISI vs repeat index",
        group_col=args.plot_group_col,
        min_bin_n=args.min_repeat_bin_n,
        max_repeat=args.max_plot_repeat_index,
    )
    make_time_spent_repeating_plot(
        segment_df,
        out_png=out_dir / f"{prefix}_time_spent_repeating_vs_repeat_index.png",
        title=f"{args.animal_id} label {cluster_label}: time spent repeating syllable in phrase",
        group_col=args.plot_group_col,
        min_bin_n=args.min_repeat_bin_n,
        max_repeat=args.max_plot_repeat_index,
    )
    return segment_df, phrase_df, stats_df


def main():
    parser = argparse.ArgumentParser(description="Measure syllable duration and inter-syllable intervals across repeat position in repeated syllable phrases.")
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
                        help="Merge target-label runs across small non-target gaps within a file or stitched recording. Default 0.")
    parser.add_argument("--stitch-across-file-segments", action="store_true",
                        help="Treat consecutive file_index chunks from the same original recording (_segment_N filenames) as one continuous recording before finding phrase-label runs.")

    # Majority vote smoothing.
    parser.add_argument("--disable-majority-vote", action="store_true", help="Turn off majority-vote label smoothing.")
    parser.add_argument("--majority-vote-window-bins", type=int, default=7,
                        help="Centered window size, in decoder time bins, for majority-vote smoothing. Even values are rounded up to odd. Default 7.")
    parser.add_argument("--majority-vote-min-fraction", type=float, default=0.50,
                        help="Minimum fraction of usable bins in the window required to replace the center label. Default 0.50.")
    parser.add_argument("--majority-vote-ignore-labels", default="-1,nan,None",
                        help="Comma-separated labels to ignore while choosing the majority label, e.g. -1,nan,None. Default: -1,nan,None.")

    # Canary segmentation settings.
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

    # Plot / stats settings.
    parser.add_argument("--plot-group-col", default="pre_post", choices=["pre_post", "period"],
                        help="How to group plotted points/median lines. Default: pre_post.")
    parser.add_argument("--min-repeat-bin-n", type=int, default=3,
                        help="Minimum number of segments required to draw a median/IQR at a repeat index. Default 3.")
    parser.add_argument("--min-values-per-group", type=int, default=5,
                        help="Minimum finite values required for slope/correlation stats. Default 5.")
    parser.add_argument("--max-plot-repeat-index", type=int, default=None,
                        help="Optional cap on repeat index shown in plots.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    labels_to_run = parse_label_list(args)
    print(f"[INFO] Labels to analyze: {labels_to_run}")

    # Load once and pass into each label analysis to avoid re-reading the NPZ repeatedly.
    data = np.load(args.npz_path, allow_pickle=True)
    X = np.asarray(data[args.spec_key], dtype=float)
    labels_raw = normalize_label_array(data[args.label_key])
    file_indices = np.asarray(data[args.file_index_key]).astype(int)
    file_map = get_file_map_dict(data[args.file_map_key])
    loaded = (X, labels_raw, file_indices, file_map)

    all_segments = []
    all_phrases = []
    all_stats = []
    for lab in labels_to_run:
        seg_df, phrase_df, stats_df = analyze_label_timing(args, lab, out_dir=out_dir, loaded=loaded)
        if not seg_df.empty:
            all_segments.append(seg_df)
        if not phrase_df.empty:
            all_phrases.append(phrase_df)
        if not stats_df.empty:
            all_stats.append(stats_df.assign(label=str(lab), animal_id=str(args.animal_id)))

    if all_segments:
        combined_segments = pd.concat(all_segments, ignore_index=True)
        combined_segments.to_csv(out_dir / f"{args.animal_id}_all_labels_duration_isi_segment_timing.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_duration_isi_segment_timing.csv'}")
        make_duration_isi_vs_repeat_plot(
            combined_segments,
            out_png=out_dir / f"{args.animal_id}_all_labels_duration_and_isi_vs_repeat_index.png",
            title=f"{args.animal_id} all labels: syllable duration and ISI vs repeat index",
            group_col=args.plot_group_col,
            min_bin_n=args.min_repeat_bin_n,
            max_repeat=args.max_plot_repeat_index,
        )
        make_time_spent_repeating_plot(
            combined_segments,
            out_png=out_dir / f"{args.animal_id}_all_labels_time_spent_repeating_vs_repeat_index.png",
            title=f"{args.animal_id} all labels: time spent repeating syllable in phrase",
            group_col=args.plot_group_col,
            min_bin_n=args.min_repeat_bin_n,
            max_repeat=args.max_plot_repeat_index,
        )
    if all_phrases:
        combined_phrases = pd.concat(all_phrases, ignore_index=True)
        combined_phrases.to_csv(out_dir / f"{args.animal_id}_all_labels_duration_isi_phrase_summary.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_duration_isi_phrase_summary.csv'}")
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        combined_stats.to_csv(out_dir / f"{args.animal_id}_all_labels_duration_isi_stats.csv", index=False)
        print(f"[SAVED] {out_dir / f'{args.animal_id}_all_labels_duration_isi_stats.csv'}")


if __name__ == "__main__":
    main()
