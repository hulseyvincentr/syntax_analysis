#!/usr/bin/env python3
"""
cluster_pitch_entropy_panels_v16_hist_density_fixed.py
For one HDBSCAN / phrase cluster label, generate:
1) full-run context spectrograms
2) stitched spectrogram panels of only that label for early_pre / late_pre / early_post / late_post
3) spectrograms with separate feature panels for combined pitch + pitch derivative, Wiener entropy, and total power
   for early_pre, late_pre, early_post, and late_post by default
4) CSV of feature time series for the stitched selected rows
5) optional pitch-contour FFT segmentation CSVs, visual segmentation guides, segmented-syllable QC figures, segmented-bin histograms, and overlaid density plots
Assumptions:
- NPZ contains:
    - spectrogram array: shape (N_timebins, N_freqbins), default key='s'
    - cluster labels: shape (N_timebins,), default key='hdbscan_labels'
    - file indices per timebin: shape (N_timebins,), default key='file_indices'
    - file_map: dict-like object mapping file index -> file name/path, default key='file_map'
- Treatment date is looked up from the metadata Excel sheet
- Pre/post split is based on file date relative to treatment date
- Early/late split is done within pre and within post using file order (default file_half)
Notes:
- "Pitch" here is estimated from the dominant spectral peak in a configurable frequency band.
  Therefore "pitch derivative" is an approximate spectral-peak derivative, not a Sound Analysis Pro clone.
- Wiener entropy is computed as spectral flatness: geometric mean / arithmetic mean.
"""
import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PERIOD_TITLES = {
    "early_pre": "Early pre-lesion",
    "late_pre": "Late pre-lesion",
    "early_post": "Early post-lesion",
    "late_post": "Late post-lesion",
}
PERIOD_COLORS = {
    "early_pre": "#4C78A8",
    "late_pre": "#59A14F",
    "early_post": "#E15759",
    "late_post": "#B07AA1",
}
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
def normalize_label_array(labels):
    return np.asarray(labels).astype(str)
def get_file_map_dict(file_map_obj):
    """Convert loaded NPZ file_map object to a plain dict."""
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
def file_entry_to_name(entry):
    """file_map value might be a string, list, tuple, or np.ndarray."""
    if isinstance(entry, (list, tuple, np.ndarray)):
        if len(entry) == 0:
            return ""
        entry = entry[0]
    return os.path.basename(str(entry))
def parse_datetime_from_filename(file_name):
    """
    Try several filename datetime formats.
    Examples supported:
    - USA5443_2024_04_01_12_34_56.wav
    - USA5443_20240401_123456.wav
    - USA5443_240401_123456.wav
    - 2024-04-01_12-34-56
    - 2024_04_01
    - USA5288_45382.42553504_3_31_11_49_13_segment_0.npz
    For filenames like:
        USA5288_45382.42553504_3_31_11_49_13_segment_0.npz
    the number 45382.42553504 is treated as an Excel serial date.
    The year/date are inferred from that serial date, and the trailing
    month/day/hour/minute/second tokens are used for the clock time when present.
    """
    base = os.path.basename(str(file_name))
    # ------------------------------------------------------------------
    # Format used by segmented files, e.g.:
    # USA5288_45382.42553504_3_31_11_49_13_segment_0.npz
    # Excel serial 45382 -> 2024-03-31; trailing tokens give M_D_H_M_S.
    # ------------------------------------------------------------------
    segmented_excel_match = re.search(
        r'_(?P<serial>\d{5}(?:\.\d+)?)_'
        r'(?P<m>\d{1,2})_(?P<d>\d{1,2})_'
        r'(?P<H>\d{1,2})_(?P<M>\d{1,2})_(?P<S>\d{1,2})'
        r'(?:_segment|\.)',
        base
    )
    if segmented_excel_match:
        gd = segmented_excel_match.groupdict()
        serial = float(gd["serial"])
        serial_dt = pd.to_datetime(serial, unit="D", origin="1899-12-30")
        # Use the year from the Excel serial date, but use the explicit
        # month/day/time encoded later in the filename. This preserves the
        # recording clock time in names like ..._3_31_11_49_13_segment_0.
        return pd.Timestamp(
            year=int(serial_dt.year),
            month=int(gd["m"]),
            day=int(gd["d"]),
            hour=int(gd["H"]),
            minute=int(gd["M"]),
            second=int(gd["S"]),
        )
    # Excel serial date/time somewhere in the filename, e.g. 45382.42553504.
    # This is a fallback for files that do not include a later M_D_H_M_S pattern.
    excel_match = re.search(r'(?<!\d)(?P<serial>\d{5}(?:\.\d+)?)(?!\d)', base)
    if excel_match:
        serial = float(excel_match.group("serial"))
        # Reasonable Excel serial range for modern recording dates.
        if 30000 <= serial <= 60000:
            return pd.to_datetime(serial, unit="D", origin="1899-12-30")
    patterns = [
        # 2024_04_01_12_34_56 or 2024-04-01-12-34-56
        r'(?P<y>20\d{2})[_-](?P<m>\d{1,2})[_-](?P<d>\d{1,2})[_-](?P<H>\d{1,2})[_-](?P<M>\d{1,2})(?:[_-](?P<S>\d{1,2}))?',
        # 20240401_123456
        r'(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        # 240401_123456
        r'(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        # date only: 2024_04_01 or 2024-04-01
        r'(?P<y>20\d{2})[_-](?P<m>\d{1,2})[_-](?P<d>\d{1,2})',
    ]
    for pat in patterns:
        match = re.search(pat, base)
        if match:
            gd = match.groupdict()
            y = int(gd["y"])
            if y < 100:
                y += 2000
            m = int(gd["m"])
            d = int(gd["d"])
            H = int(gd.get("H") or 0)
            M = int(gd.get("M") or 0)
            S = int(gd.get("S") or 0)
            return pd.Timestamp(year=y, month=m, day=d, hour=H, minute=M, second=S)
    raise ValueError(
        f"Could not parse a datetime from filename:\n{file_name}\n"
        "Please edit parse_datetime_from_filename() for your filename format."
    )
def load_treatment_date(metadata_excel_path, animal_id,
                        metadata_sheet="metadata",
                        animal_id_col="Animal ID",
                        treatment_date_col="Treatment date"):
    meta = pd.read_excel(metadata_excel_path, sheet_name=metadata_sheet)
    hits = meta[meta[animal_id_col].astype(str) == str(animal_id)]
    if hits.empty:
        raise ValueError(
            f"Animal ID {animal_id} not found in metadata sheet '{metadata_sheet}'."
        )
    td = pd.to_datetime(hits.iloc[0][treatment_date_col])
    if pd.isna(td):
        raise ValueError(f"Treatment date missing for animal {animal_id}.")
    return td.normalize()
def split_in_half(sorted_items):
    n = len(sorted_items)
    if n == 0:
        return [], []
    mid = n // 2
    if n == 1:
        return sorted_items, []
    return sorted_items[:mid], sorted_items[mid:]
def split_by_file_period(unique_file_indices, file_time_lookup, treatment_date,
                         include_treatment_day_in_post=False,
                         split_mode="file_half"):
    """Split file indices into early_pre, late_pre, early_post, late_post."""
    pre_files = []
    post_files = []
    for fidx in unique_file_indices:
        dt = pd.Timestamp(file_time_lookup[fidx]).normalize()
        if dt < treatment_date:
            pre_files.append(fidx)
        elif dt > treatment_date:
            post_files.append(fidx)
        elif include_treatment_day_in_post:
            post_files.append(fidx)
    pre_files = sorted(pre_files, key=lambda x: file_time_lookup[x])
    post_files = sorted(post_files, key=lambda x: file_time_lookup[x])
    if split_mode == "file_half":
        early_pre, late_pre = split_in_half(pre_files)
        early_post, late_post = split_in_half(post_files)
    elif split_mode == "file_median":
        def median_split(files):
            if len(files) == 0:
                return [], []
            times = np.array([pd.Timestamp(file_time_lookup[f]).value for f in files], dtype=np.int64)
            med = np.median(times)
            left = [f for f in files if pd.Timestamp(file_time_lookup[f]).value <= med]
            right = [f for f in files if pd.Timestamp(file_time_lookup[f]).value > med]
            if len(left) == 0 or len(right) == 0:
                return split_in_half(files)
            return left, right
        early_pre, late_pre = median_split(pre_files)
        early_post, late_post = median_split(post_files)
    else:
        raise ValueError("split_mode must be 'file_half' or 'file_median'")
    return {
        "early_pre": early_pre,
        "late_pre": late_pre,
        "early_post": early_post,
        "late_post": late_post,
    }
def find_true_runs(mask):
    """Return list of (start, end) pairs, end-exclusive."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, ends))
def find_label_runs(label_array):
    """
    Return contiguous runs of identical labels.
    Returns a list of (start, end, label), where start/end are relative
    to the passed label_array and end is exclusive.
    """
    label_array = np.asarray(label_array).astype(str)
    if len(label_array) == 0:
        return []
    change_points = np.where(label_array[1:] != label_array[:-1])[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(label_array)]))
    runs = []
    for run_start, run_end in zip(starts, ends):
        runs.append((int(run_start), int(run_end), str(label_array[run_start])))
    return runs
def choose_runs_evenly(runs, max_runs):
    if len(runs) <= max_runs:
        return runs
    idx = np.linspace(0, len(runs) - 1, max_runs).round().astype(int)
    idx = np.unique(idx)
    return [runs[i] for i in idx]
def context_bounds(start, end, n_total, context_bins):
    lo = max(0, start - context_bins)
    hi = min(n_total, end + context_bins)
    return lo, hi
def build_stitched_rows(runs, max_rows=5, max_bins_per_row=2000, separator_bins=2):
    """
    Input runs: list of (start, end)
    Output: list of row segment lists, e.g.
      [
        [(s1,e1),(s2,e2),...],
        [(s3,e3),...],
      ]
    """
    rows = []
    current = []
    current_len = 0
    for (start, end) in runs:
        run_len = end - start
        needed = run_len if len(current) == 0 else (separator_bins + run_len)
        if len(current) > 0 and current_len + needed > max_bins_per_row:
            rows.append(current)
            if len(rows) >= max_rows:
                return rows
            current = [(start, end)]
            current_len = run_len
        else:
            current.append((start, end))
            current_len += needed
    if len(current) > 0 and len(rows) < max_rows:
        rows.append(current)
    return rows
def stitch_segments(spectrogram_tx_f, segment_list, separator_bins=2):
    """
    Returns:
      X: shape (T, F)
      source_bin_idx: length T, np.nan for separator columns
      is_separator: bool length T
      segment_ids: length T, integer segment id or -1 for separators
    """
    blocks = []
    source_bins = []
    separators = []
    segment_ids = []
    for seg_id, (start, end) in enumerate(segment_list):
        if seg_id > 0 and separator_bins > 0:
            sep_block = np.full((separator_bins, spectrogram_tx_f.shape[1]), np.nan, dtype=float)
            blocks.append(sep_block)
            source_bins.extend([np.nan] * separator_bins)
            separators.extend([True] * separator_bins)
            segment_ids.extend([-1] * separator_bins)
        block = spectrogram_tx_f[start:end, :]
        blocks.append(block)
        source_bins.extend(list(range(start, end)))
        separators.extend([False] * (end - start))
        segment_ids.extend([seg_id] * (end - start))
    if len(blocks) == 0:
        return None, None, None, None
    X = np.vstack(blocks)
    return X, np.array(source_bins, dtype=float), np.array(separators, dtype=bool), np.array(segment_ids, dtype=int)
def to_linear_power(X, spec_scale="linear"):
    """Convert spectrogram values to strictly positive power-like values.

    Rows that are entirely NaN, such as stitched separator columns, are
    preserved as NaN so they plot as gaps instead of zeros. This also avoids
    all-NaN RuntimeWarnings from np.nanmin.
    """
    eps = 1e-12
    X = np.asarray(X, dtype=float)

    if spec_scale == "linear":
        Y = X.copy()
        finite_rows = np.any(np.isfinite(Y), axis=1)
        row_min = np.zeros((Y.shape[0], 1), dtype=float)
        if np.any(finite_rows):
            row_min[finite_rows] = np.nanmin(Y[finite_rows, :], axis=1, keepdims=True)

        bad_rows = finite_rows & (row_min[:, 0] <= 0)
        if np.any(bad_rows):
            Y[bad_rows, :] = Y[bad_rows, :] - row_min[bad_rows, :] + eps

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
        row_min = np.zeros((Y.shape[0], 1), dtype=float)
        if np.any(finite_rows):
            row_min[finite_rows] = np.nanmin(Y[finite_rows, :], axis=1, keepdims=True)
            Y[finite_rows, :] = Y[finite_rows, :] - row_min[finite_rows, :] + eps
        return Y

    raise ValueError("spec_scale must be one of: linear, log10, loge, shift")

def moving_average_nan(x, window):
    if window <= 1:
        return x.copy()
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - window // 2)
        hi = min(len(x), i + window // 2 + 1)
        chunk = x[lo:hi]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out
def compute_wiener_entropy(X_tx_f, spec_scale="linear"):
    """Wiener entropy / spectral flatness per time bin. Returns values in approximately [0,1]."""
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

def compute_total_power(X_tx_f, spec_scale="linear"):
    """Total power per time bin.

    This is the sum of power across frequency bins for each time bin.
    It is computed for every finite row, with no sound/silence thresholding.
    Entirely NaN rows, such as stitched separators, remain NaN.
    """
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    out = np.full(power.shape[0], np.nan, dtype=float)

    for i in range(power.shape[0]):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        out[i] = np.nansum(row)

    return out

def compute_pitch_trace_and_derivative(
        X_tx_f,
        bin_s,
        max_freq_khz=12.0,
        pitch_band_min_khz=0.8,
        pitch_band_max_khz=10.0,
        spec_scale="linear",
        smoothing_window=5,
        min_peak_fraction=0.0):
    """
    Estimate pitch as the dominant spectral peak in a frequency band,
    using a permissive peak-selection rule, then compute absolute first derivative in kHz/s.
    """
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    T, F = power.shape
    freqs_khz = np.linspace(0, max_freq_khz, F)
    band_mask = (freqs_khz >= pitch_band_min_khz) & (freqs_khz <= pitch_band_max_khz)
    if not np.any(band_mask):
        raise ValueError("Pitch band does not overlap the frequency axis.")
    band_freqs = freqs_khz[band_mask]
    pitch_khz = np.full(T, np.nan, dtype=float)
    for i in range(T):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        band_power = row[band_mask]
        if np.all(~np.isfinite(band_power)):
            continue
        band_power = np.nan_to_num(band_power, nan=0.0)
        total = band_power.sum()
        if total <= 0:
            continue
        peak_idx = np.argmax(band_power)
        peak_val = band_power[peak_idx]
        peak_fraction = peak_val / total
        if min_peak_fraction > 0 and peak_fraction < min_peak_fraction:
            continue
        pitch_khz[i] = band_freqs[peak_idx]
    pitch_khz = moving_average_nan(pitch_khz, smoothing_window)
    pitch_derivative = np.full(T, np.nan, dtype=float)
    for i in range(1, T):
        if np.isfinite(pitch_khz[i]) and np.isfinite(pitch_khz[i - 1]):
            pitch_derivative[i] = abs(pitch_khz[i] - pitch_khz[i - 1]) / bin_s
    return pitch_khz, pitch_derivative

def finite_runs(mask):
    """Return contiguous True runs from a boolean mask as (start, end), end-exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, ends))


def interp_nan_1d(x):
    """Linearly interpolate NaNs in a 1D array. Returns all-NaN if no finite values."""
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


def lowpass_filter_fft_1d(x, bin_s, cutoff_hz):
    """
    Low-pass filter a 1D trace using an FFT.

    NaNs should already be interpolated before this function is called.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x.copy()
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        return x.copy()

    mean_val = np.nanmean(x)
    centered = x - mean_val
    freqs = np.fft.rfftfreq(len(centered), d=bin_s)
    spectrum = np.fft.rfft(centered)
    spectrum[freqs > cutoff_hz] = 0
    filtered = np.fft.irfft(spectrum, n=len(centered)) + mean_val
    return filtered


def estimate_dominant_modulation_frequency_hz(
        pitch_trace,
        bin_s,
        min_freq_hz=1.0,
        max_freq_hz=80.0):
    """
    Estimate the dominant modulation frequency in the pitch contour.

    This uses the FFT of a mean-subtracted, windowed pitch trace and ignores DC.
    The returned value is in Hz, i.e. cycles per second in the pitch contour.
    """
    x = interp_nan_1d(pitch_trace)
    good = np.isfinite(x)
    if np.sum(good) < 4:
        return np.nan

    x = x - np.nanmean(x)
    if np.nanstd(x) <= 0:
        return np.nan

    # Windowing reduces edge artifacts in the FFT.
    window = np.hanning(len(x))
    if np.all(window == 0):
        window = np.ones(len(x))
    xw = x * window

    freqs = np.fft.rfftfreq(len(xw), d=bin_s)
    amp = np.abs(np.fft.rfft(xw))

    valid = (
        np.isfinite(freqs) &
        (freqs >= min_freq_hz) &
        (freqs <= max_freq_hz)
    )
    if not np.any(valid):
        return np.nan

    valid_idx = np.flatnonzero(valid)
    peak_idx = valid_idx[np.argmax(amp[valid])]
    return float(freqs[peak_idx])


def find_local_minima_with_distance(y, min_distance_bins=3):
    """
    Find local minima with a simple minimum-spacing rule.

    This avoids adding scipy as a dependency. Candidate minima are local valleys.
    If multiple candidates are too close, the deeper valley is retained.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)

    candidates = np.where((y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    if len(candidates) == 0:
        return candidates.astype(int)

    min_distance_bins = max(1, int(min_distance_bins))

    # Greedy retain: process deepest minima first, then sort by time.
    order = candidates[np.argsort(y[candidates])]
    kept = []
    for c in order:
        if all(abs(c - k) >= min_distance_bins for k in kept):
            kept.append(int(c))
    kept = np.array(sorted(kept), dtype=int)
    return kept


def segment_pitch_contour_by_fft(
        pitch_khz,
        bin_s,
        is_separator=None,
        min_mod_freq_hz=1.0,
        max_mod_freq_hz=80.0,
        cutoff_multiplier=1.0,
        min_distance_cycles=0.50,
        min_points=12):
    """
    Segment a pitch contour by:
      1) estimating the dominant modulation frequency with the FFT,
      2) low-pass filtering the pitch contour at that frequency,
      3) placing candidate syllable boundaries at minima of the filtered trace.

    Returns
    -------
    filtered_pitch : ndarray
        Low-pass filtered pitch, with NaNs outside analyzed chunks.
    boundary_mask : ndarray of bool
        True at minima-based candidate boundaries.
    segment_ids : ndarray of int
        Per-bin segment IDs, -1 outside valid segments.
    segments : list of dict
        Segment metadata with start/end bins and dominant modulation frequency.
    """
    pitch_khz = np.asarray(pitch_khz, dtype=float)
    n = len(pitch_khz)

    if is_separator is None:
        is_separator = np.zeros(n, dtype=bool)
    is_separator = np.asarray(is_separator, dtype=bool)

    filtered_pitch = np.full(n, np.nan, dtype=float)
    boundary_mask = np.zeros(n, dtype=bool)
    segment_ids = np.full(n, -1, dtype=int)
    segments = []

    valid_for_chunks = (~is_separator) & np.isfinite(pitch_khz)
    chunks = finite_runs(valid_for_chunks)

    global_segment_id = 0

    for chunk_start, chunk_end in chunks:
        x_raw = pitch_khz[chunk_start:chunk_end]
        if len(x_raw) < min_points:
            continue

        x = interp_nan_1d(x_raw)
        dom_freq_hz = estimate_dominant_modulation_frequency_hz(
            x,
            bin_s=bin_s,
            min_freq_hz=min_mod_freq_hz,
            max_freq_hz=max_mod_freq_hz,
        )

        if np.isfinite(dom_freq_hz) and dom_freq_hz > 0:
            cutoff_hz = dom_freq_hz * cutoff_multiplier
        else:
            cutoff_hz = np.nan

        x_filt = lowpass_filter_fft_1d(x, bin_s=bin_s, cutoff_hz=cutoff_hz)
        filtered_pitch[chunk_start:chunk_end] = x_filt

        if np.isfinite(dom_freq_hz) and dom_freq_hz > 0:
            period_bins = 1.0 / (dom_freq_hz * bin_s)
            min_distance_bins = max(2, int(round(min_distance_cycles * period_bins)))
        else:
            min_distance_bins = max(2, int(round(0.050 / bin_s)))  # fallback: ~50 ms

        minima_rel = find_local_minima_with_distance(
            x_filt,
            min_distance_bins=min_distance_bins
        )

        # Avoid edge minima; endpoints are used as segment starts/ends.
        minima_rel = minima_rel[(minima_rel > 0) & (minima_rel < len(x_filt) - 1)]
        minima_abs = minima_rel + chunk_start
        boundary_mask[minima_abs] = True

        # Segments are intervals between minima, including chunk edges.
        edges = np.concatenate([[chunk_start], minima_abs, [chunk_end]])
        edges = np.unique(edges.astype(int))
        edges.sort()

        for seg_i in range(len(edges) - 1):
            seg_start = int(edges[seg_i])
            seg_end = int(edges[seg_i + 1])
            if seg_end <= seg_start:
                continue
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
                "dominant_pitch_modulation_freq_hz": dom_freq_hz,
                "lowpass_cutoff_hz": cutoff_hz,
                "min_distance_bins": int(min_distance_bins),
                "n_boundary_minima_in_chunk": int(len(minima_abs)),
            })
            global_segment_id += 1

    return filtered_pitch, boundary_mask, segment_ids, segments


def robust_minmax(x, lower_pct=5, upper_pct=95):
    """Normalize to [0,1] using robust percentiles."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return out
    lo = np.nanpercentile(x[good], lower_pct)
    hi = np.nanpercentile(x[good], upper_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(x[good])
        hi = np.nanmax(x[good])
        if hi <= lo:
            out[good] = 0.5
            return out
    out[good] = np.clip((x[good] - lo) / (hi - lo), 0, 1)
    return out
def imshow_spectrogram(ax, X_tx_f, bin_s, max_freq_khz=12.0, cmap="gray_r"):
    """Display X shape: (T, F)."""
    T, _F = X_tx_f.shape
    extent = [0, T * bin_s, 0, max_freq_khz]
    finite = np.isfinite(X_tx_f)
    if np.any(finite):
        vmin = np.nanpercentile(X_tx_f[finite], 5)
        vmax = np.nanpercentile(X_tx_f[finite], 99.5)
        if vmax <= vmin:
            vmax = np.nanmax(X_tx_f[finite])
            vmin = np.nanmin(X_tx_f[finite])
    else:
        vmin, vmax = 0, 1
    ax.imshow(
        X_tx_f.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(0, max_freq_khz)
def choose_items_evenly(items, max_items):
    """Return up to max_items sampled evenly across the list."""
    if len(items) <= max_items:
        return items
    idx = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    idx = np.unique(idx)
    return [items[i] for i in idx]


def plot_segment_qc_figure(
        segment_qc_entries,
        out_png,
        animal_id,
        target_label,
        period_name,
        bin_s=0.0027,
        max_freq_khz=12.0,
        max_segments=60,
        target_panel_duration_s=5.0,
        max_panels=8,
        separator_bins=2):
    """Plot stitched segmented-syllable QC panels totaling about target_panel_duration_s per panel."""
    if len(segment_qc_entries) == 0:
        print(f"[WARN] No segmented syllable QC entries available for {period_name}; skipping QC figure.")
        return

    entries = choose_items_evenly(segment_qc_entries, max_segments)

    # Build panels of stitched segmented syllables, targeting ~5 seconds total per panel.
    panels = []
    current_panel = []
    current_dur = 0.0
    for ent in entries:
        dur = ent["X"].shape[0] * bin_s
        if current_panel and (current_dur + dur > target_panel_duration_s):
            panels.append(current_panel)
            current_panel = [ent]
            current_dur = dur
        else:
            current_panel.append(ent)
            current_dur += dur
    if current_panel:
        panels.append(current_panel)

    panels = choose_items_evenly(panels, max_panels)
    n_panels = len(panels)

    def stitch_panel(panel_entries):
        X_parts = []
        pitch_parts = []
        filt_parts = []
        deriv_parts = []
        wiener_parts = []
        power_parts = []
        boundaries = []
        segment_spans = []
        t_cursor = 0

        for idx, ent in enumerate(panel_entries):
            X = ent["X"]
            pitch = ent["pitch_khz"]
            filt = ent["filtered_pitch_khz"]
            deriv = ent["pitch_deriv_khz_per_s"]
            wiener = ent["wiener"]
            power = ent["total_power_norm"]

            if idx > 0 and separator_bins > 0:
                X_parts.append(np.full((separator_bins, X.shape[1]), np.nan, dtype=float))
                pitch_parts.append(np.full(separator_bins, np.nan, dtype=float))
                filt_parts.append(np.full(separator_bins, np.nan, dtype=float))
                deriv_parts.append(np.full(separator_bins, np.nan, dtype=float))
                wiener_parts.append(np.full(separator_bins, np.nan, dtype=float))
                power_parts.append(np.full(separator_bins, np.nan, dtype=float))
                boundaries.append(t_cursor)
                t_cursor += separator_bins

            seg_start = t_cursor
            seg_end = seg_start + X.shape[0]
            segment_spans.append({
                "start": seg_start,
                "end": seg_end,
                "segment_id": ent["segment_id"],
            })

            X_parts.append(X)
            pitch_parts.append(pitch)
            filt_parts.append(filt)
            deriv_parts.append(deriv)
            wiener_parts.append(wiener)
            power_parts.append(power)
            t_cursor = seg_end

        return {
            "X": np.vstack(X_parts),
            "pitch": np.concatenate(pitch_parts),
            "filt": np.concatenate(filt_parts),
            "deriv": np.concatenate(deriv_parts),
            "wiener": np.concatenate(wiener_parts),
            "power": np.concatenate(power_parts),
            "boundaries": boundaries,
            "segment_spans": segment_spans,
            "n_segments": len(panel_entries),
            "duration_s": t_cursor * bin_s,
        }

    stitched_panels = [stitch_panel(panel) for panel in panels]

    height_ratios = []
    for _ in range(n_panels):
        height_ratios.extend([3.0, 1.25, 1.05])

    fig, axes = plt.subplots(
        n_panels * 3,
        1,
        figsize=(18, max(4.8 * n_panels, 7)),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios}
    )
    axes = axes.ravel()
    pretty_period = PERIOD_TITLES.get(period_name, period_name.replace("_", " "))

    for i, pan in enumerate(stitched_panels):
        spec_ax = axes[3 * i]
        pitch_ax = axes[3 * i + 1]
        entropy_ax = axes[3 * i + 2]

        X = pan["X"]
        pitch_khz = pan["pitch"]
        filtered_pitch_khz = pan["filt"]
        pitch_deriv_khz_per_s = pan["deriv"]
        wiener = pan["wiener"]
        total_power_norm = pan["power"]
        t = np.arange(X.shape[0]) * bin_s

        imshow_spectrogram(spec_ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)
        spec_ax.set_ylabel("Freq (kHz)")
        spec_ax.tick_params(axis="x", labelbottom=False)
        spec_ax.text(
            -0.01, 0.5,
            f"panel {i + 1}\n{pan['n_segments']} segs\n{pan['duration_s']:.2f} s",
            transform=spec_ax.transAxes,
            ha="right", va="center", fontsize=8.5
        )

        # Mark separators and annotate segment ids.
        for b in pan["boundaries"]:
            x_b = b * bin_s
            spec_ax.axvline(x_b, color="white", lw=0.9, alpha=0.75, linestyle="--")
            pitch_ax.axvline(x_b, color="0.65", lw=0.8, alpha=0.9, linestyle="--")
            entropy_ax.axvline(x_b, color="0.65", lw=0.8, alpha=0.9, linestyle="--")
        for span in pan["segment_spans"]:
            x_mid = ((span["start"] + span["end"]) / 2.0) * bin_s
            spec_ax.text(
                x_mid, max_freq_khz - 0.35, str(span["segment_id"]),
                ha="center", va="top", fontsize=7, color="#0072B2",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15)
            )

        pitch_ax.plot(t, pitch_khz, color="#C77700", lw=2.1, alpha=1.0, label="Pitch (kHz)")
        if np.any(np.isfinite(filtered_pitch_khz)):
            pitch_ax.plot(t, filtered_pitch_khz, color="#0072B2", lw=1.8, alpha=0.95, label="Low-pass pitch")
        pitch_ax.axhline(0, color="black", lw=1.0, alpha=0.7, linestyle="--")
        pitch_ax.set_ylabel("Pitch\n(kHz)")
        pitch_ax.tick_params(axis="both", labelsize=8)
        pitch_ax.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 0)
        pitch_ax.tick_params(axis="x", labelbottom=False)

        pitch2_ax = pitch_ax.twinx()
        pitch2_ax.plot(t, pitch_deriv_khz_per_s, color="#007A4D", lw=2.0, alpha=1.0, label="Pitch derivative (kHz/s)")
        pitch2_ax.axhline(0, color="#007A4D", lw=1.0, alpha=0.75, linestyle=":")
        pitch2_ax.set_ylabel("Pitch deriv.\n(kHz/s)")
        pitch2_ax.tick_params(axis="y", labelsize=8)

        h1, l1 = pitch_ax.get_legend_handles_labels()
        h2, l2 = pitch2_ax.get_legend_handles_labels()
        pitch_ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=8)

        entropy_ax.plot(t, wiener, color="#B83280", lw=2.0, alpha=1.0, label="Wiener entropy")
        entropy_ax.axhline(0, color="#B83280", lw=1.0, alpha=0.65, linestyle=":")
        entropy_ax.set_ylabel("Wiener\nentropy")
        entropy_ax.set_ylim(-0.02, 1.02)
        entropy_ax.tick_params(axis="both", labelsize=8)
        entropy_ax.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 0)

        power_ax = entropy_ax.twinx()
        power_ax.plot(t, total_power_norm, color="#333333", lw=1.5, alpha=0.85, label="Total power (norm)")
        power_ax.set_ylabel("Total power\n(norm)")
        power_ax.set_ylim(-0.02, 1.02)
        power_ax.tick_params(axis="y", labelsize=8)

        h1, l1 = entropy_ax.get_legend_handles_labels()
        h2, l2 = power_ax.get_legend_handles_labels()
        entropy_ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=8)

        if i == 0:
            spec_ax.set_title(
                f"{animal_id} — label {target_label} — {pretty_period} stitched segmented syllable QC\n"
                f"Each panel stitches multiple segmented syllables (~{target_panel_duration_s:.1f} s total) with pitch, pitch derivative, Wiener entropy, and total power.",
                fontsize=13
            )

        if i < n_panels - 1:
            entropy_ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Time across stitched segmented syllables (s)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")



def plot_full_run_contexts(
        s_tx_f,
        labels,
        runs_by_period,
        out_png,
        target_label,
        animal_id,
        bin_s=0.0027,
        max_freq_khz=12.0,
        context_bins=300,
        max_runs_per_period=5,
        min_label_run_bins=8):
    """Full run contexts figure: multiple rows, grouped by period."""
    selected_by_period = {
        p: choose_runs_evenly(runs_by_period.get(p, []), max_runs_per_period)
        for p in PERIOD_ORDER
    }
    total_rows = sum(len(v) for v in selected_by_period.values())
    if total_rows == 0:
        print(f"[WARN] No runs found for label {target_label}; skipping full run contexts.")
        return
    fig, axes = plt.subplots(total_rows, 1, figsize=(18, max(2.2 * total_rows, 5)), squeeze=False)
    axes = axes.ravel()
    row_idx = 0
    for period in PERIOD_ORDER:
        runs = selected_by_period[period]
        color = PERIOD_COLORS[period]
        for i, (start, end) in enumerate(runs):
            ax = axes[row_idx]
            lo, hi = context_bounds(start, end, s_tx_f.shape[0], context_bins)
            X = s_tx_f[lo:hi, :]
            context_labels = labels[lo:hi]
            label_runs = find_label_runs(context_labels)
            imshow_spectrogram(ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)
            ax.set_ylabel("Freq (kHz)")
            x0 = (start - lo) * bin_s
            width = (end - start) * bin_s
            rect = Rectangle((x0, 0), width, max_freq_khz, fill=False, edgecolor=color, linewidth=2.0)
            ax.add_patch(rect)
            # Highlight every occurrence of the target label inside the context window,
            # and make the target label text stand out for intuition.
            for run_start, run_end, run_label in label_runs:
                if str(run_label) == str(target_label):
                    x_run = run_start * bin_s
                    w_run = (run_end - run_start) * bin_s
                    ax.add_patch(Rectangle(
                        (x_run, 0), w_run, max_freq_khz,
                        facecolor=color, edgecolor="none", alpha=0.12, zorder=1
                    ))
            # Annotate contiguous syllable / cluster labels across the context window.
            # Very short runs are skipped to avoid overcrowding the figure.
            for run_start, run_end, run_label in label_runs:
                if (run_end - run_start) < min_label_run_bins:
                    continue
                x_mid = ((run_start + run_end) / 2.0) * bin_s
                is_target = (str(run_label) == str(target_label))
                ax.text(
                    x_mid,
                    max_freq_khz - 0.35,
                    str(run_label),
                    ha="center",
                    va="top",
                    fontsize=7.2 if is_target else 7,
                    fontweight="bold" if is_target else "normal",
                    color=(color if is_target else "black"),
                    bbox=dict(facecolor="white", edgecolor=(color if is_target else "none"),
                              alpha=0.75 if is_target else 0.60, pad=0.25 if is_target else 0.2),
                )
            ax.text(
                -0.01, 0.5,
                f"{PERIOD_TITLES[period]}\nrun {i + 1}\n{end - start} bins",
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=9
            )
            if row_idx == 0:
                ax.set_title(
                    f"{animal_id} — label {target_label} — full run contexts\n"
                    f"Colored outline = selected target run; translucent shading and bold labels = all target-label occurrences; text = contiguous labels; ±{context_bins} context bins",
                    fontsize=13
                )
            if row_idx < total_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Context time (s)")
            row_idx += 1
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
def plot_stitched_selected_bins_by_period(
        s_tx_f,
        rows_by_period,
        out_png,
        target_label,
        animal_id,
        bin_s=0.0027,
        max_freq_khz=12.0,
        separator_bins=2):
    """Make a stitched selected-bins figure with multiple example rows per period."""
    plot_rows = []
    for period in PERIOD_ORDER:
        seg_rows = rows_by_period.get(period, [])
        if len(seg_rows) == 0:
            plot_rows.append((period, None, None))
        else:
            for row_num, seg_list in enumerate(seg_rows, start=1):
                plot_rows.append((period, row_num, seg_list))
    total_rows = len(plot_rows)
    fig, axes = plt.subplots(total_rows, 1, figsize=(18, max(2.2 * total_rows, 8)), squeeze=False)
    axes = axes.ravel()
    for ax, (period, row_num, seg_list) in zip(axes, plot_rows):
        if seg_list is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{PERIOD_TITLES[period]}: no runs found", ha="center", va="center", fontsize=11)
            continue
        X, _source_bins, _is_sep, _seg_ids = stitch_segments(s_tx_f, seg_list, separator_bins=separator_bins)
        imshow_spectrogram(ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)
        ax.set_ylabel("Freq (kHz)")
        left_label = PERIOD_TITLES[period] if row_num == 1 else f"{PERIOD_TITLES[period]}\nrow {row_num}"
        ax.text(
            -0.01, 0.5,
            left_label,
            transform=ax.transAxes,
            ha="right", va="center", fontsize=9
        )
    axes[0].set_title(
        f"{animal_id} — label {target_label} — stitched spectrograms of only this label\n"
        f"Multiple example rows for early pre / late pre / early post / late post",
        fontsize=13
    )
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
def plot_feature_overlay_rows(
        s_tx_f,
        row_segments,
        out_png,
        out_csv,
        out_segment_csv,
        out_segment_qc_png,
        animal_id,
        target_label,
        period_name="early_pre",
        bin_s=0.0027,
        max_freq_khz=12.0,
        spec_scale="linear",
        pitch_band_min_khz=0.8,
        pitch_band_max_khz=10.0,
        pitch_min_peak_fraction=0.0,
        pitch_smoothing_window=5,
        pitch_segmentation_enabled=True,
        segmentation_min_mod_freq_hz=1.0,
        segmentation_max_mod_freq_hz=80.0,
        segmentation_cutoff_multiplier=1.0,
        segmentation_min_distance_cycles=0.50,
        segmentation_min_points=12,
        segment_qc_max_segments=60,
        segment_qc_pad_bins=0,
        segment_qc_target_duration_s=5.0,
        segment_qc_max_panels=8,
        separator_bins=2):
    """Plot stitched selected rows with spectrograms, feature panels, and optional pitch-FFT segmentation.

    For each row, the figure shows:
      1) spectrogram only, with no time-series overlays
      2) pitch + absolute pitch-derivative, with optional low-pass filtered pitch
      3) Wiener entropy + normalized total power

    If pitch segmentation is enabled, candidate syllable boundaries are estimated by:
      FFT(pitch contour) -> dominant modulation frequency -> low-pass pitch at that frequency
      -> minima of the filtered pitch trace.

    Gaps in the feature traces indicate NaN values, while dashed horizontal
    baselines mark true zero where applicable.
    """
    if len(row_segments) == 0:
        print(f"[WARN] No stitched rows available for {period_name}; skipping feature overlay.")
        return {
            "segmented_wiener_values": np.array([], dtype=float),
            "segmented_pitch_derivative_values": np.array([], dtype=float),
        }

    n_rows = len(row_segments)
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([4.0, 1.25, 1.05])

    fig, axes = plt.subplots(
        n_rows * 3,
        1,
        figsize=(18, max(5.6 * n_rows, 7)),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios}
    )
    axes = axes.ravel()
    all_csv_rows = []
    all_segment_rows = []
    all_segment_qc_entries = []
    segmented_wiener_values = []
    segmented_pitch_derivative_values = []
    pretty_period = PERIOD_TITLES.get(period_name, period_name.replace("_", " "))

    for row_idx, seg_list in enumerate(row_segments):
        X, source_bins, is_sep, seg_ids = stitch_segments(s_tx_f, seg_list, separator_bins=separator_bins)
        spec_ax = axes[3 * row_idx]
        pitchderiv_ax = axes[3 * row_idx + 1]
        entropy_ax = axes[3 * row_idx + 2]

        imshow_spectrogram(spec_ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)

        wiener = compute_wiener_entropy(X, spec_scale=spec_scale)
        total_power = compute_total_power(X, spec_scale=spec_scale)
        pitch_khz, pitch_deriv_khz_per_s = compute_pitch_trace_and_derivative(
            X,
            bin_s=bin_s,
            max_freq_khz=max_freq_khz,
            pitch_band_min_khz=pitch_band_min_khz,
            pitch_band_max_khz=pitch_band_max_khz,
            spec_scale=spec_scale,
            smoothing_window=pitch_smoothing_window,
            min_peak_fraction=pitch_min_peak_fraction
        )

        wiener[is_sep] = np.nan
        total_power[is_sep] = np.nan
        pitch_khz[is_sep] = np.nan
        pitch_deriv_khz_per_s[is_sep] = np.nan

        if pitch_segmentation_enabled:
            filtered_pitch_khz, boundary_mask, pitch_segment_ids, segment_records = segment_pitch_contour_by_fft(
                pitch_khz,
                bin_s=bin_s,
                is_separator=is_sep,
                min_mod_freq_hz=segmentation_min_mod_freq_hz,
                max_mod_freq_hz=segmentation_max_mod_freq_hz,
                cutoff_multiplier=segmentation_cutoff_multiplier,
                min_distance_cycles=segmentation_min_distance_cycles,
                min_points=segmentation_min_points,
            )
        else:
            filtered_pitch_khz = np.full_like(pitch_khz, np.nan, dtype=float)
            boundary_mask = np.zeros_like(is_sep, dtype=bool)
            pitch_segment_ids = np.full(len(pitch_khz), -1, dtype=int)
            segment_records = []

        wiener_norm = robust_minmax(wiener)
        total_power_norm = robust_minmax(total_power)
        pitch_norm = robust_minmax(pitch_khz)
        pitch_deriv_norm = robust_minmax(pitch_deriv_khz_per_s)
        filtered_pitch_norm = robust_minmax(filtered_pitch_khz)
        t = np.arange(X.shape[0]) * bin_s

        # Mark stitched separators on all row panels.
        sep_idx = np.where(is_sep)[0]
        if len(sep_idx) > 0:
            transition_idx = [sep_idx[0]]
            for i in range(1, len(sep_idx)):
                if sep_idx[i] != sep_idx[i - 1] + 1:
                    transition_idx.append(sep_idx[i])
            for si in transition_idx:
                x_sep = si * bin_s
                spec_ax.axvline(x_sep, color="white", lw=0.8, alpha=0.6, linestyle="--")
                pitchderiv_ax.axvline(x_sep, color="0.65", lw=0.8, alpha=0.9, linestyle="--")
                entropy_ax.axvline(x_sep, color="0.65", lw=0.8, alpha=0.9, linestyle="--")

        # Mark pitch-segmentation minima/boundaries.
        boundary_bins = np.flatnonzero(boundary_mask)
        for b in boundary_bins:
            x_b = b * bin_s
            spec_ax.axvline(x_b, color="#0072B2", lw=1.0, alpha=0.65, linestyle=":")
            pitchderiv_ax.axvline(x_b, color="#0072B2", lw=1.0, alpha=0.75, linestyle=":")
            entropy_ax.axvline(x_b, color="#0072B2", lw=1.0, alpha=0.55, linestyle=":")

        # Spectrogram panel only: no time-series overlays.
        spec_ax.set_ylabel("Freq (kHz)")
        spec_ax.text(
            -0.01, 0.5,
            f"row {row_idx + 1}",
            transform=spec_ax.transAxes,
            ha="right", va="center", fontsize=9
        )
        spec_ax.tick_params(axis="x", labelbottom=False)

        # Combined pitch + derivative panel.
        pitchderiv_ax.plot(
            t, pitch_khz,
            color="#C77700",
            lw=2.0,
            alpha=1.0,
            label="Pitch (kHz)",
        )
        if pitch_segmentation_enabled:
            pitchderiv_ax.plot(
                t, filtered_pitch_khz,
                color="#0072B2",
                lw=1.8,
                alpha=0.95,
                label="Low-pass pitch",
            )

        pitchderiv_ax.axhline(0, color="black", lw=1.0, alpha=0.75, linestyle="--")
        pitchderiv_ax.set_ylabel("Pitch\n(kHz)")
        pitchderiv_ax.tick_params(axis="both", labelsize=8)
        pitchderiv_ax.set_xlim(t[0], t[-1] if len(t) > 0 else 0)
        pitchderiv_ax.tick_params(axis="x", labelbottom=False)

        deriv2_ax = pitchderiv_ax.twinx()
        deriv2_ax.plot(
            t, pitch_deriv_khz_per_s,
            color="#007A4D",
            lw=2.0,
            alpha=1.0,
            label="Pitch derivative (kHz/s)",
        )
        deriv2_ax.axhline(0, color="#007A4D", lw=1.0, alpha=0.75, linestyle=":")
        deriv2_ax.set_ylabel("Pitch deriv.\n(kHz/s)")
        deriv2_ax.tick_params(axis="y", labelsize=8)

        handles1, labels1 = pitchderiv_ax.get_legend_handles_labels()
        handles2, labels2 = deriv2_ax.get_legend_handles_labels()
        pitchderiv_ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True, fontsize=8)

        # Separate Wiener entropy panel, with normalized total power for reference.
        entropy_ax.plot(
            t, wiener,
            color="#B83280",
            lw=2.0,
            alpha=1.0,
            label="Wiener entropy",
        )
        entropy_ax.axhline(0, color="#B83280", lw=1.0, alpha=0.65, linestyle=":")
        entropy_ax.set_ylabel("Wiener\nentropy")
        entropy_ax.set_ylim(-0.02, 1.02)
        entropy_ax.tick_params(axis="both", labelsize=8)
        entropy_ax.set_xlim(t[0], t[-1] if len(t) > 0 else 0)

        power_ax = entropy_ax.twinx()
        power_ax.plot(
            t, total_power_norm,
            color="#333333",
            lw=1.6,
            alpha=0.85,
            label="Total power (norm)",
        )
        power_ax.set_ylabel("Total power\n(norm)")
        power_ax.set_ylim(-0.02, 1.02)
        power_ax.tick_params(axis="y", labelsize=8)

        handles1, labels1 = entropy_ax.get_legend_handles_labels()
        handles2, labels2 = power_ax.get_legend_handles_labels()
        entropy_ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True, fontsize=8)

        if row_idx == 0:
            seg_text = (
                " Blue dotted lines = pitch-contour minima used as candidate syllable boundaries."
                if pitch_segmentation_enabled else ""
            )
            spec_ax.set_title(
                f"{animal_id} — label {target_label} — {period_name}_expanded_full_runs\n"
                f"Spectrograms shown without overlays; feature panels below show pitch, pitch derivative, Wiener entropy, and total power. Gaps = NaN.{seg_text}",
                fontsize=13
            )

        # Per-bin feature CSV.
        for i in range(X.shape[0]):
            all_csv_rows.append({
                "animal_id": animal_id,
                "label": target_label,
                "period": period_name,
                "row": row_idx + 1,
                "bin_in_row": i,
                "time_s": i * bin_s,
                "source_bin_index": source_bins[i] if np.isfinite(source_bins[i]) else np.nan,
                "is_separator": bool(is_sep[i]),
                "segment_id_within_row": int(seg_ids[i]),
                "pitch_fft_segment_id": int(pitch_segment_ids[i]),
                "is_pitch_fft_boundary": bool(boundary_mask[i]),
                "wiener_entropy": wiener[i],
                "wiener_entropy_normalized": wiener_norm[i],
                "total_power": total_power[i],
                "total_power_normalized": total_power_norm[i],
                "pitch_khz": pitch_khz[i],
                "pitch_normalized": pitch_norm[i],
                "lowpass_filtered_pitch_khz": filtered_pitch_khz[i],
                "lowpass_filtered_pitch_normalized": filtered_pitch_norm[i],
                "pitch_derivative_khz_per_s": pitch_deriv_khz_per_s[i],
                "pitch_derivative_hz_per_s": pitch_deriv_khz_per_s[i] * 1000.0 if np.isfinite(pitch_deriv_khz_per_s[i]) else np.nan,
                "pitch_derivative_normalized": pitch_deriv_norm[i],
            })

        # Segment summary CSV and segmented-syllable QC entries.
        for rec in segment_records:
            seg_start = int(rec["segment_start_bin"])
            seg_end = int(rec["segment_end_bin"])
            if seg_end <= seg_start:
                continue
            sl = slice(seg_start, seg_end)
            all_segment_rows.append({
                "animal_id": animal_id,
                "label": target_label,
                "period": period_name,
                "row": row_idx + 1,
                "pitch_fft_segment_id": int(rec["segment_id"]),
                "segment_start_bin": seg_start,
                "segment_end_bin": seg_end,
                "segment_start_time_s": rec["segment_start_time_s"],
                "segment_end_time_s": rec["segment_end_time_s"],
                "segment_duration_s": rec["segment_duration_s"],
                "source_start_bin_index": source_bins[seg_start] if np.isfinite(source_bins[seg_start]) else np.nan,
                "source_end_bin_index": source_bins[seg_end - 1] if np.isfinite(source_bins[seg_end - 1]) else np.nan,
                "dominant_pitch_modulation_freq_hz": rec["dominant_pitch_modulation_freq_hz"],
                "lowpass_cutoff_hz": rec["lowpass_cutoff_hz"],
                "min_distance_bins": rec["min_distance_bins"],
                "n_boundary_minima_in_chunk": rec["n_boundary_minima_in_chunk"],
                "n_bins": seg_end - seg_start,
                "mean_pitch_khz": np.nanmean(pitch_khz[sl]) if np.any(np.isfinite(pitch_khz[sl])) else np.nan,
                "median_pitch_khz": np.nanmedian(pitch_khz[sl]) if np.any(np.isfinite(pitch_khz[sl])) else np.nan,
                "mean_lowpass_pitch_khz": np.nanmean(filtered_pitch_khz[sl]) if np.any(np.isfinite(filtered_pitch_khz[sl])) else np.nan,
                "mean_pitch_derivative_khz_per_s": np.nanmean(pitch_deriv_khz_per_s[sl]) if np.any(np.isfinite(pitch_deriv_khz_per_s[sl])) else np.nan,
                "mean_wiener_entropy": np.nanmean(wiener[sl]) if np.any(np.isfinite(wiener[sl])) else np.nan,
                "mean_total_power": np.nanmean(total_power[sl]) if np.any(np.isfinite(total_power[sl])) else np.nan,
                "mean_total_power_normalized": np.nanmean(total_power_norm[sl]) if np.any(np.isfinite(total_power_norm[sl])) else np.nan,
            })

            seg_w = wiener[sl]
            seg_pd = pitch_deriv_khz_per_s[sl]
            if np.any(np.isfinite(seg_w)):
                segmented_wiener_values.extend(seg_w[np.isfinite(seg_w)].tolist())
            if np.any(np.isfinite(seg_pd)):
                segmented_pitch_derivative_values.extend(seg_pd[np.isfinite(seg_pd)].tolist())

            qc_start = max(0, seg_start - int(segment_qc_pad_bins))
            qc_end = min(len(pitch_khz), seg_end + int(segment_qc_pad_bins))
            qc_slice = slice(qc_start, qc_end)
            all_segment_qc_entries.append({
                "row": row_idx + 1,
                "segment_id": int(rec["segment_id"]),
                "duration_s": rec["segment_duration_s"],
                "X": X[qc_slice, :].copy(),
                "pitch_khz": pitch_khz[qc_slice].copy(),
                "filtered_pitch_khz": filtered_pitch_khz[qc_slice].copy(),
                "pitch_deriv_khz_per_s": pitch_deriv_khz_per_s[qc_slice].copy(),
                "wiener": wiener[qc_slice].copy(),
                "total_power_norm": total_power_norm[qc_slice].copy(),
            })

        if row_idx < n_rows - 1:
            entropy_ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    df = pd.DataFrame(all_csv_rows)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    seg_df = pd.DataFrame(all_segment_rows)
    seg_df.to_csv(out_segment_csv, index=False)
    print(f"[SAVED] {out_segment_csv}")

    if pitch_segmentation_enabled and len(all_segment_qc_entries) > 0:
        plot_segment_qc_figure(
            all_segment_qc_entries,
            out_png=out_segment_qc_png,
            animal_id=animal_id,
            target_label=target_label,
            period_name=period_name,
            bin_s=bin_s,
            max_freq_khz=max_freq_khz,
            max_segments=segment_qc_max_segments,
            target_panel_duration_s=segment_qc_target_duration_s,
            max_panels=segment_qc_max_panels,
            separator_bins=separator_bins,
        )

    return {
        "segmented_wiener_values": np.asarray(segmented_wiener_values, dtype=float),
        "segmented_pitch_derivative_values": np.asarray(segmented_pitch_derivative_values, dtype=float),
    }

def plot_segmented_bin_histograms(values_by_period, out_png, animal_id, target_label, value_name, x_label, bins=60, log_counts=False):
    """Plot 2x2 histograms of segmented-bin values, one panel per period."""
    periods_with_data = [p for p in PERIOD_ORDER if len(values_by_period.get(p, [])) > 0]
    if len(periods_with_data) == 0:
        print(f"[WARN] No segmented-bin values available for {value_name}; skipping histogram figure.")
        return

    finite_values = []
    for p in PERIOD_ORDER:
        arr = np.asarray(values_by_period.get(p, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            finite_values.append(arr)

    if len(finite_values) == 0:
        print(f"[WARN] All segmented-bin values were NaN for {value_name}; skipping histogram figure.")
        return

    all_vals = np.concatenate(finite_values)
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo -= 0.5
            hi += 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    axes = axes.ravel()
    for ax, period in zip(axes, PERIOD_ORDER):
        arr = np.asarray(values_by_period.get(period, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.text(0.5, 0.5, "No segmented bins", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(PERIOD_TITLES[period])
            ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            continue

        ax.hist(arr, bins=bin_edges, alpha=0.85, edgecolor="black")
        if log_counts:
            ax.set_yscale("log")
        ax.set_title(f"{PERIOD_TITLES[period]}\nN bins = {arr.size:,}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.axvline(np.nanmedian(arr), color="red", linestyle="--", linewidth=1.2, alpha=0.9, label=f"median = {np.nanmedian(arr):.3g}")
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        f"{animal_id} — label {target_label} — segmented-bin histogram: {value_name}",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def _simple_gaussian_kde(x_grid, values):
    """Lightweight Gaussian KDE using NumPy only."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    if arr.size == 1:
        bw = max(1e-3, 0.05 * (np.abs(arr[0]) + 1.0))
    else:
        std = np.nanstd(arr, ddof=1)
        if not np.isfinite(std) or std <= 0:
            std = max(1e-3, 0.05 * (np.nanmax(arr) - np.nanmin(arr) + 1.0))
        bw = 1.06 * std * (arr.size ** (-1.0 / 5.0))
        bw = max(float(bw), 1e-3)
    diffs = (x_grid[:, None] - arr[None, :]) / bw
    kern = np.exp(-0.5 * diffs**2) / (bw * np.sqrt(2.0 * np.pi))
    return np.nanmean(kern, axis=1)


def plot_overlaid_density(values_by_period, out_png, animal_id, target_label, value_name, x_label, n_points=400):
    """Plot one overlaid density figure with all four conditions together."""
    finite_values = []
    for p in PERIOD_ORDER:
        arr = np.asarray(values_by_period.get(p, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            finite_values.append(arr)

    if len(finite_values) == 0:
        print(f"[WARN] No segmented-bin values available for overlaid density plot: {value_name}; skipping.")
        return

    all_vals = np.concatenate(finite_values)
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo -= 0.5
            hi += 0.5
    x_grid = np.linspace(lo, hi, n_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False
    for period in PERIOD_ORDER:
        arr = np.asarray(values_by_period.get(period, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        dens = _simple_gaussian_kde(x_grid, arr)
        color = PERIOD_COLORS.get(period, None)
        ax.plot(x_grid, dens, lw=2.0, alpha=0.95, label=f"{PERIOD_TITLES[period]} (N={arr.size:,})", color=color)
        ax.axvline(np.nanmedian(arr), linestyle='--', lw=1.1, alpha=0.7, color=color)
        any_plotted = True

    if not any_plotted:
        print(f"[WARN] Density plot had no plottable data for {value_name}; skipping.")
        plt.close(fig)
        return

    ax.set_title(f"{animal_id} — label {target_label} — overlaid density: {value_name}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate cluster spectrogram / pitch-derivative / Wiener-entropy panels.")
    parser.add_argument("--npz-path", required=True, help="Path to bird .npz file")
    parser.add_argument("--cluster-label", required=True, help="Target cluster label, e.g. 11")
    parser.add_argument("--animal-id", required=True, help="Animal ID, e.g. USA5443")
    parser.add_argument("--metadata-excel-path", required=True, help="Metadata Excel with treatment dates")
    parser.add_argument("--out-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--spec-key", default="s", help="Spectrogram key in NPZ")
    parser.add_argument("--label-key", default="hdbscan_labels", help="Cluster labels key in NPZ")
    parser.add_argument("--file-index-key", default="file_indices", help="File index key in NPZ")
    parser.add_argument("--file-map-key", default="file_map", help="File map key in NPZ")
    parser.add_argument("--metadata-sheet", default="metadata", help="Sheet name in Excel")
    parser.add_argument("--animal-id-col", default="Animal ID", help="Animal ID column in metadata")
    parser.add_argument("--treatment-date-col", default="Treatment date", help="Treatment date column in metadata")
    parser.add_argument("--bin-ms", type=float, default=2.7, help="Milliseconds per time bin")
    parser.add_argument("--max-freq-khz", type=float, default=12.0, help="Top of frequency axis in kHz")
    parser.add_argument("--spec-scale", choices=["linear", "log10", "loge", "shift"], default="linear",
                        help="How to interpret spectrogram values for feature computations")
    parser.add_argument("--include-treatment-day-in-post", action="store_true",
                        help="If set, treatment-day files are counted as post")
    parser.add_argument("--split-mode", choices=["file_half", "file_median"], default="file_half",
                        help="How to split early vs late within pre and post")
    parser.add_argument("--context-bins", type=int, default=500, help="Context bins on each side of a run")
    parser.add_argument("--max-context-runs-per-period", type=int, default=10, help="Rows per period in context figure")
    parser.add_argument(
        "--min-label-run-bins",
        type=int,
        default=8,
        help="Minimum contiguous label-run length to annotate in the full-run context figure"
    )
    parser.add_argument("--period-max-bins", type=int, default=3500, help="Max bins per stitched row")
    parser.add_argument("--feature-rows", type=int, default=8, help="Rows for feature overlay figure")
    parser.add_argument("--selected-bins-rows-per-period", type=int, default=3,
                        help="How many stitched selected-bin rows to show per period in the selected-bins-by-period figure.")
    parser.add_argument("--feature-period", default="all", choices=PERIOD_ORDER + ["all"],
                        help="Which period to use for feature-panel figures. Use 'all' to generate early_pre, late_pre, early_post, and late_post.")
    parser.add_argument("--separator-bins", type=int, default=2, help="Blank bins inserted between stitched runs")
    parser.add_argument("--pitch-band-min-khz", type=float, default=0.8, help="Lower frequency bound for pitch estimation")
    parser.add_argument("--pitch-band-max-khz", type=float, default=10.0, help="Upper frequency bound for pitch estimation")
    parser.add_argument("--pitch-min-peak-fraction", type=float, default=0.0,
                        help="Minimum fraction of in-band power that must be contained in the dominant peak to accept a pitch estimate. Use 0 for a permissive max-frequency estimate.")
    parser.add_argument("--pitch-smoothing-window", type=int, default=5,
                        help="Moving-average smoothing window for the pitch trace (in time bins).")
    parser.add_argument("--disable-pitch-fft-segmentation", action="store_true",
                        help="Disable the experimental pitch-contour FFT segmentation.")
    parser.add_argument("--segmentation-min-mod-freq-hz", type=float, default=1.0,
                        help="Minimum pitch-contour modulation frequency to consider in the FFT segmentation.")
    parser.add_argument("--segmentation-max-mod-freq-hz", type=float, default=80.0,
                        help="Maximum pitch-contour modulation frequency to consider in the FFT segmentation.")
    parser.add_argument("--segmentation-cutoff-multiplier", type=float, default=1.0,
                        help="Low-pass cutoff is dominant modulation frequency times this multiplier.")
    parser.add_argument("--segmentation-min-distance-cycles", type=float, default=0.50,
                        help="Minimum spacing between segmentation minima, in cycles of the dominant modulation period.")
    parser.add_argument("--segmentation-min-points", type=int, default=12,
                        help="Minimum number of valid pitch bins required to segment a pitch-contour chunk.")
    parser.add_argument("--segment-qc-max-segments", type=int, default=60,
                        help="Maximum number of segmented syllables to sample before stitching them into QC panels.")
    parser.add_argument("--segment-qc-pad-bins", type=int, default=0,
                        help="Optional number of bins of padding to include before and after each segmented syllable in the QC figure.")
    parser.add_argument("--segment-qc-target-duration-s", type=float, default=5.0,
                        help="Target total duration of stitched segmented syllables per QC panel, in seconds.")
    parser.add_argument("--segment-qc-max-panels", type=int, default=8,
                        help="Maximum number of stitched segmented-syllable QC panels to show per period.")
    args = parser.parse_args()
    ensure_dir(args.out_dir)
    bin_s = args.bin_ms / 1000.0
    target_label = str(args.cluster_label)
    print("[INFO] Loading NPZ...")
    data = np.load(args.npz_path, allow_pickle=True)
    s_tx_f = np.asarray(data[args.spec_key], dtype=float)
    labels = normalize_label_array(data[args.label_key])
    file_indices = np.asarray(data[args.file_index_key]).astype(int)
    file_map = get_file_map_dict(data[args.file_map_key])
    if s_tx_f.shape[0] != len(labels) or s_tx_f.shape[0] != len(file_indices):
        raise ValueError(
            "Shape mismatch:\n"
            f"spectrogram rows = {s_tx_f.shape[0]}\n"
            f"labels length   = {len(labels)}\n"
            f"file_indices len= {len(file_indices)}"
        )
    print("[INFO] Parsing file timestamps...")
    unique_file_indices = sorted(np.unique(file_indices))
    file_time_lookup = {}
    for fidx in unique_file_indices:
        entry = file_map.get(int(fidx), file_map.get(str(fidx), None))
        if entry is None:
            raise KeyError(f"file_map missing entry for file index {fidx}")
        fname = file_entry_to_name(entry)
        file_time_lookup[int(fidx)] = parse_datetime_from_filename(fname)
    print("[INFO] Loading treatment date...")
    treatment_date = load_treatment_date(
        args.metadata_excel_path,
        animal_id=args.animal_id,
        metadata_sheet=args.metadata_sheet,
        animal_id_col=args.animal_id_col,
        treatment_date_col=args.treatment_date_col,
    )
    print(f"[INFO] Treatment date: {treatment_date.date()}")
    print("[INFO] Splitting files into early/late pre/post...")
    files_by_period = split_by_file_period(
        unique_file_indices=unique_file_indices,
        file_time_lookup=file_time_lookup,
        treatment_date=treatment_date,
        include_treatment_day_in_post=args.include_treatment_day_in_post,
        split_mode=args.split_mode,
    )
    period_masks = {}
    for period, fidx_list in files_by_period.items():
        period_masks[period] = np.isin(file_indices, fidx_list)
    print("[INFO] Finding runs...")
    runs_by_period = {}
    label_mask = (labels == target_label)
    for period in PERIOD_ORDER:
        runs_by_period[period] = find_true_runs(label_mask & period_masks[period])
        print(f"  {period}: {len(runs_by_period[period])} runs")
    out_context_png = os.path.join(args.out_dir, f"{args.animal_id}_label{target_label}_full_run_contexts.png")
    plot_full_run_contexts(
        s_tx_f=s_tx_f,
        labels=labels,
        runs_by_period=runs_by_period,
        out_png=out_context_png,
        target_label=target_label,
        animal_id=args.animal_id,
        bin_s=bin_s,
        max_freq_khz=args.max_freq_khz,
        context_bins=args.context_bins,
        max_runs_per_period=args.max_context_runs_per_period,
        min_label_run_bins=args.min_label_run_bins,
    )
    # Also generate a separate full-run-contexts figure for each period.
    for period in PERIOD_ORDER:
        one_period_runs = {p: (runs_by_period[p] if p == period else []) for p in PERIOD_ORDER}
        out_context_period_png = os.path.join(
            args.out_dir,
            f"{args.animal_id}_label{target_label}_{period}_full_run_contexts.png"
        )
        plot_full_run_contexts(
            s_tx_f=s_tx_f,
            labels=labels,
            runs_by_period=one_period_runs,
            out_png=out_context_period_png,
            target_label=target_label,
            animal_id=args.animal_id,
            bin_s=bin_s,
            max_freq_khz=args.max_freq_khz,
            context_bins=args.context_bins,
            max_runs_per_period=args.max_context_runs_per_period,
            min_label_run_bins=args.min_label_run_bins,
        )
    print("[INFO] Building stitched rows by period...")
    rows_by_period = {}
    for period in PERIOD_ORDER:
        rows_by_period[period] = build_stitched_rows(
            runs_by_period[period],
            max_rows=args.selected_bins_rows_per_period,
            max_bins_per_row=args.period_max_bins,
            separator_bins=args.separator_bins
        )
    out_period_png = os.path.join(args.out_dir, f"{args.animal_id}_label{target_label}_selected_bins_by_period.png")
    plot_stitched_selected_bins_by_period(
        s_tx_f=s_tx_f,
        rows_by_period=rows_by_period,
        out_png=out_period_png,
        target_label=target_label,
        animal_id=args.animal_id,
        bin_s=bin_s,
        max_freq_khz=args.max_freq_khz,
        separator_bins=args.separator_bins,
    )
    if args.feature_period == "all":
        feature_periods_to_plot = PERIOD_ORDER
    else:
        feature_periods_to_plot = [args.feature_period]

    histogram_wiener_by_period = {p: np.array([], dtype=float) for p in PERIOD_ORDER}
    histogram_pitch_deriv_by_period = {p: np.array([], dtype=float) for p in PERIOD_ORDER}

    for feature_period in feature_periods_to_plot:
        print(f"[INFO] Building feature overlay rows for {feature_period}...")
        feature_rows = build_stitched_rows(
            runs_by_period[feature_period],
            max_rows=args.feature_rows,
            max_bins_per_row=args.period_max_bins,
            separator_bins=args.separator_bins
        )
        out_feature_png = os.path.join(
            args.out_dir,
            f"{args.animal_id}_label{target_label}_{feature_period}_pitch_derivative_wiener_entropy_overlay.png"
        )
        out_feature_csv = os.path.join(
            args.out_dir,
            f"{args.animal_id}_label{target_label}_{feature_period}_pitch_derivative_wiener_entropy.csv"
        )
        out_segment_csv = os.path.join(
            args.out_dir,
            f"{args.animal_id}_label{target_label}_{feature_period}_pitch_fft_segments.csv"
        )
        out_segment_qc_png = os.path.join(
            args.out_dir,
            f"{args.animal_id}_label{target_label}_{feature_period}_segmented_syllable_qc.png"
        )
        feature_result = plot_feature_overlay_rows(
            s_tx_f=s_tx_f,
            row_segments=feature_rows,
            out_png=out_feature_png,
            out_csv=out_feature_csv,
            out_segment_csv=out_segment_csv,
            out_segment_qc_png=out_segment_qc_png,
            animal_id=args.animal_id,
            target_label=target_label,
            period_name=feature_period,
            bin_s=bin_s,
            max_freq_khz=args.max_freq_khz,
            spec_scale=args.spec_scale,
            pitch_band_min_khz=args.pitch_band_min_khz,
            pitch_band_max_khz=args.pitch_band_max_khz,
            pitch_min_peak_fraction=args.pitch_min_peak_fraction,
            pitch_smoothing_window=args.pitch_smoothing_window,
            pitch_segmentation_enabled=(not args.disable_pitch_fft_segmentation),
            segmentation_min_mod_freq_hz=args.segmentation_min_mod_freq_hz,
            segmentation_max_mod_freq_hz=args.segmentation_max_mod_freq_hz,
            segmentation_cutoff_multiplier=args.segmentation_cutoff_multiplier,
            segmentation_min_distance_cycles=args.segmentation_min_distance_cycles,
            segmentation_min_points=args.segmentation_min_points,
            segment_qc_max_segments=args.segment_qc_max_segments,
            segment_qc_pad_bins=args.segment_qc_pad_bins,
            segment_qc_target_duration_s=args.segment_qc_target_duration_s,
            segment_qc_max_panels=args.segment_qc_max_panels,
            separator_bins=args.separator_bins,
        )
        if feature_result is not None:
            histogram_wiener_by_period[feature_period] = feature_result.get("segmented_wiener_values", np.array([], dtype=float))
            histogram_pitch_deriv_by_period[feature_period] = feature_result.get("segmented_pitch_derivative_values", np.array([], dtype=float))

    out_wiener_hist_png = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_segmented_bin_wiener_entropy_histograms.png"
    )
    plot_segmented_bin_histograms(
        histogram_wiener_by_period,
        out_png=out_wiener_hist_png,
        animal_id=args.animal_id,
        target_label=target_label,
        value_name="Wiener entropy",
        x_label="Wiener entropy",
        bins=60,
        log_counts=False,
    )

    out_pitch_deriv_hist_png = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_segmented_bin_pitch_derivative_histograms.png"
    )
    plot_segmented_bin_histograms(
        histogram_pitch_deriv_by_period,
        out_png=out_pitch_deriv_hist_png,
        animal_id=args.animal_id,
        target_label=target_label,
        value_name="Pitch derivative",
        x_label="Pitch derivative (kHz/s)",
        bins=60,
        log_counts=False,
    )

    out_wiener_density_png = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_segmented_bin_wiener_entropy_density_overlay.png"
    )
    plot_overlaid_density(
        histogram_wiener_by_period,
        out_png=out_wiener_density_png,
        animal_id=args.animal_id,
        target_label=target_label,
        value_name="Wiener entropy",
        x_label="Wiener entropy",
        n_points=400,
    )

    out_pitch_deriv_density_png = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_segmented_bin_pitch_derivative_density_overlay.png"
    )
    plot_overlaid_density(
        histogram_pitch_deriv_by_period,
        out_png=out_pitch_deriv_density_png,
        animal_id=args.animal_id,
        target_label=target_label,
        value_name="Pitch derivative",
        x_label="Pitch derivative (kHz/s)",
        n_points=400,
    )

    print("[DONE]")
if __name__ == "__main__":
    main()