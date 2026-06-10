#!/usr/bin/env python3
"""
cluster_spectrogram_crosscorr_canary_segmentation_v1.py

For one bird and one HDBSCAN/phrase cluster label, use the same Gardner-style
canary segmentation approach from the pitch/entropy analysis, then calculate
spectrogram cross-correlation similarity for segmented syllables from that same
cluster.

Main outputs:
1) Per-segment CSV with template-correlation values.
2) Pre/post ECDF and overlay-histogram plots of correlation to a late-pre template.
3) Pre/post ECDF and overlay-histogram plots of within-condition pairwise correlations.
4) Period mean-template QC figure.
5) Example high/low post-lesion similarity figure.

Interpretation:
- Higher spectrogram correlation = more similar acoustic structure.
- If post-lesion distributions shift lower than pre-lesion, post syllables are
  less similar/stereotyped relative to the pre-lesion template or within-condition.

Assumptions:
- NPZ contains a spectrogram matrix (default key: s) with shape (time_bins, freq_bins)
- labels and file indices are one value per time bin.
- file_map maps file index -> original filename/path.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
PERIOD_COLORS = {
    "early_pre": "#4C78A8",
    "late_pre": "#59A14F",
    "early_post": "#E15759",
    "late_post": "#B07AA1",
    "combined_pre": "#4C78A8",
    "combined_post": "#E45756",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_label_array(labels):
    return np.asarray(labels).astype(str)


def get_file_map_dict(file_map_obj):
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
    if isinstance(entry, (list, tuple, np.ndarray)):
        if len(entry) == 0:
            return ""
        entry = entry[0]
    return os.path.basename(str(entry))


def parse_datetime_from_filename(file_name):
    base = os.path.basename(str(file_name))

    # Example: USA5288_45382.42553504_3_31_11_49_13_segment_0.npz
    segmented_excel_match = re.search(
        r'_(?P<serial>\d{5}(?:\.\d+)?)_'
        r'(?P<m>\d{1,2})_(?P<d>\d{1,2})_'
        r'(?P<H>\d{1,2})_(?P<M>\d{1,2})_(?P<S>\d{1,2})'
        r'(?:_segment|\.)',
        base,
    )
    if segmented_excel_match:
        gd = segmented_excel_match.groupdict()
        serial_dt = pd.to_datetime(float(gd["serial"]), unit="D", origin="1899-12-30")
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

    raise ValueError(f"Could not parse datetime from filename: {file_name}")


def load_treatment_date(metadata_excel_path, animal_id, metadata_sheet="metadata",
                        animal_id_col="Animal ID", treatment_date_col="Treatment date"):
    meta = pd.read_excel(metadata_excel_path, sheet_name=metadata_sheet)
    hits = meta[meta[animal_id_col].astype(str) == str(animal_id)]
    if hits.empty:
        raise ValueError(f"Animal ID {animal_id} not found in metadata sheet '{metadata_sheet}'.")
    td = pd.to_datetime(hits.iloc[0][treatment_date_col])
    if pd.isna(td):
        raise ValueError(f"Treatment date missing for animal {animal_id}.")
    return td.normalize()


def split_in_half(sorted_items):
    n = len(sorted_items)
    if n == 0:
        return [], []
    if n == 1:
        return sorted_items, []
    mid = n // 2
    return sorted_items[:mid], sorted_items[mid:]


def split_by_file_period(unique_file_indices, file_time_lookup, treatment_date,
                         include_treatment_day_in_post=False, split_mode="file_half"):
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
        raise ValueError("split_mode must be file_half or file_median")

    return {
        "early_pre": early_pre,
        "late_pre": late_pre,
        "early_post": early_post,
        "late_post": late_post,
    }


def find_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def finite_runs(mask):
    return find_true_runs(mask)


def to_linear_power(X, spec_scale="linear"):
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
    raise ValueError("spec_scale must be linear, log10, loge, or shift")


def moving_average_nan(x, window):
    if window <= 1:
        return np.asarray(x, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - window // 2)
        hi = min(len(x), i + window // 2 + 1)
        chunk = x[lo:hi]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def compute_total_power(X_tx_f, spec_scale="linear"):
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    out = np.full(power.shape[0], np.nan, dtype=float)
    for i in range(power.shape[0]):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        out[i] = np.nansum(row)
    return out


def compute_pitch_trace(X_tx_f, bin_s, max_freq_khz=12.0,
                        pitch_band_min_khz=0.8, pitch_band_max_khz=10.0,
                        spec_scale="linear", smoothing_window=5,
                        min_peak_fraction=0.0):
    power = to_linear_power(X_tx_f, spec_scale=spec_scale)
    T, F = power.shape
    freqs_khz = np.linspace(0, max_freq_khz, F)
    band_mask = (freqs_khz >= pitch_band_min_khz) & (freqs_khz <= pitch_band_max_khz)
    if not np.any(band_mask):
        raise ValueError("Pitch band does not overlap frequency axis")
    band_freqs = freqs_khz[band_mask]
    pitch = np.full(T, np.nan, dtype=float)
    for i in range(T):
        row = power[i]
        if np.all(~np.isfinite(row)):
            continue
        bp = row[band_mask]
        if np.all(~np.isfinite(bp)):
            continue
        bp = np.nan_to_num(bp, nan=0.0)
        total = np.sum(bp)
        if total <= 0:
            continue
        idx = int(np.argmax(bp))
        peak_fraction = bp[idx] / total
        if min_peak_fraction > 0 and peak_fraction < min_peak_fraction:
            continue
        pitch[i] = band_freqs[idx]
    return moving_average_nan(pitch, smoothing_window)


def interp_nan_1d(x):
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


def canary_dominant_freq_with_spectrum(sig, fs, band=(2.0, 20.0)):
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


def canary_bandpass_power(power, fs, f0, frac=0.5, pad_periods=3.0):
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


def segment_power_by_canary_method(pitch_khz, total_power, bin_s, rate_band=(2.0, 20.0),
                                   frac=0.5, edge_guard_periods=0.5, pad_periods=3.0,
                                   snap=True, snap_periods=0.5, min_points=12,
                                   drop_edge_segments=True):
    if not SCIPY_AVAILABLE:
        raise ImportError("Canary segmentation requires scipy. Install with: conda install scipy")

    pitch_khz = np.asarray(pitch_khz, dtype=float)
    total_power = np.asarray(total_power, dtype=float)
    n = len(pitch_khz)
    fs = 1.0 / bin_s

    bandpassed_power = np.full(n, np.nan, dtype=float)
    boundary_mask = np.zeros(n, dtype=bool)
    presnap_mask = np.zeros(n, dtype=bool)
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
            if seg_end <= seg_start or (seg_end - seg_start) < 3:
                continue
            segment_pairs.append((seg_start, seg_end))

        if drop_edge_segments and len(segment_pairs) > 2:
            kept_pairs = segment_pairs[1:-1]
        else:
            kept_pairs = segment_pairs

        for seg_start, seg_end in kept_pairs:
            segments.append({
                "segment_id": global_segment_id,
                "segment_start_bin": int(seg_start),
                "segment_end_bin": int(seg_end),
                "segment_duration_s": (seg_end - seg_start) * bin_s,
                "dominant_pitch_modulation_freq_hz": f0,
                "min_distance_bins": int(min_dist),
                "n_boundary_minima_in_chunk": int(len(boundary_abs)),
                "segmentation_method": "canary_power_bandpass_snap" if snap else "canary_power_bandpass_no_snap",
                "dropped_edge_segments": bool(drop_edge_segments and len(segment_pairs) > 2),
            })
            global_segment_id += 1

        diagnostics.append({
            "chunk_start_bin": int(chunk_start),
            "chunk_end_bin": int(chunk_end),
            "f0_hz": f0,
            "period_ms": 1000.0 / f0,
            "freqs": freqs,
            "spec": spec,
            "presnap_idx": presnap_abs.astype(int),
            "boundary_idx": boundary_abs.astype(int),
        })

    return {
        "bandpassed_power": bandpassed_power,
        "boundary_mask": boundary_mask,
        "presnap_mask": presnap_mask,
        "segments": segments,
        "diagnostics": diagnostics,
    }


def _time_normalize_trace(y, n_points=101):
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return None
    good = np.isfinite(y)
    if np.sum(good) < 2:
        return None
    x_old = np.linspace(0.0, 100.0, y.size)
    x_new = np.linspace(0.0, 100.0, int(n_points))
    return np.interp(x_new, x_old[good], y[good])


def compute_pitch_contour_inlier_mask(entries, n_points=101, filter_sd=2.0, max_outside_fraction=0.05):
    traces = []
    valid_indices = []
    for i, ent in enumerate(entries):
        trace = ent.get("pitch_khz")
        norm = _time_normalize_trace(trace, n_points=n_points)
        if norm is None:
            continue
        traces.append(norm)
        valid_indices.append(i)
    mask = np.zeros(len(entries), dtype=bool)
    if len(traces) == 0:
        return mask, {"n_total_entries": len(entries), "n_valid_contours": 0, "n_kept": 0}
    M = np.vstack(traces)
    ref_mean = np.nanmean(M, axis=0)
    ref_sd = np.nanstd(M, axis=0)
    ref_sd_safe = np.where(np.isfinite(ref_sd) & (ref_sd > 1e-9), ref_sd, 1e-9)
    lower = ref_mean - filter_sd * ref_sd_safe
    upper = ref_mean + filter_sd * ref_sd_safe
    outside = (M < lower[None, :]) | (M > upper[None, :])
    outside_fraction = np.nanmean(outside, axis=1)
    keep_valid = outside_fraction <= float(max_outside_fraction)
    if np.sum(keep_valid) < 3:
        keep_valid = np.ones(M.shape[0], dtype=bool)
    for idx, keep in zip(valid_indices, keep_valid):
        mask[idx] = bool(keep)
    return mask, {
        "n_total_entries": len(entries),
        "n_valid_contours": len(traces),
        "n_kept": int(np.sum(mask)),
        "filter_sd": filter_sd,
        "max_outside_fraction": max_outside_fraction,
    }


def crop_frequency_band(X, max_freq_khz=12.0, freq_min_khz=0.0, freq_max_khz=None):
    X = np.asarray(X, dtype=float)
    if freq_max_khz is None:
        freq_max_khz = max_freq_khz
    F = X.shape[1]
    freqs = np.linspace(0, max_freq_khz, F)
    mask = (freqs >= freq_min_khz) & (freqs <= freq_max_khz)
    if not np.any(mask):
        return X
    return X[:, mask]


def transform_spectrogram_for_corr(X, spec_scale="linear", corr_transform="log_power"):
    X = np.asarray(X, dtype=float)
    if corr_transform == "raw":
        Y = X.copy()
    else:
        power = to_linear_power(X, spec_scale=spec_scale)
        if corr_transform == "linear_power":
            Y = power
        elif corr_transform == "log_power":
            Y = np.log10(power + 1e-12)
        else:
            raise ValueError("corr_transform must be raw, linear_power, or log_power")
    return Y


def time_normalize_spectrogram(X, n_time=64):
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


def vectorize_for_corr(X_norm):
    v = np.asarray(X_norm, dtype=float).ravel()
    good = np.isfinite(v)
    if np.sum(good) < 5:
        return None
    med = np.nanmean(v[good])
    v = np.where(good, v, med)
    v = v - np.mean(v)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return v / norm


def corr_between_unit_vectors(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def imshow_spectrogram(ax, X_tx_f, bin_s=None, max_freq_khz=12.0, cmap="gray_r", title=None):
    X = np.asarray(X_tx_f, dtype=float)
    T, _F = X.shape
    extent = [0, T if bin_s is None else T * bin_s, 0, max_freq_khz]
    finite = np.isfinite(X)
    if np.any(finite):
        vmin = np.nanpercentile(X[finite], 5)
        vmax = np.nanpercentile(X[finite], 99.5)
        if vmax <= vmin:
            vmin, vmax = np.nanmin(X[finite]), np.nanmax(X[finite])
    else:
        vmin, vmax = 0, 1
    ax.imshow(X.T, aspect="auto", origin="lower", interpolation="nearest",
              extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(0, max_freq_khz)
    if title:
        ax.set_title(title)


def segment_cluster_periods(s_tx_f, labels, file_indices, files_by_period, target_label, args):
    """Return dict period -> list of segment entries using canary segmentation within each target-label run."""
    out = {p: [] for p in PERIOD_ORDER}
    label_mask = labels == str(target_label)

    for period in PERIOD_ORDER:
        period_file_set = set(int(x) for x in files_by_period[period])
        period_mask = np.array([int(f) in period_file_set for f in file_indices], dtype=bool)
        target_mask = label_mask & period_mask
        runs = find_true_runs(target_mask)
        print(f"  {period}: {len(runs)} target-label runs")

        segment_counter = 0
        for run_i, (start, end) in enumerate(runs):
            X_run = s_tx_f[start:end, :]
            if X_run.shape[0] < args.segmentation_min_points:
                continue
            pitch = compute_pitch_trace(
                X_run,
                bin_s=args.bin_ms / 1000.0,
                max_freq_khz=args.max_freq_khz,
                pitch_band_min_khz=args.pitch_band_min_khz,
                pitch_band_max_khz=args.pitch_band_max_khz,
                spec_scale=args.spec_scale,
                smoothing_window=args.pitch_smoothing_window,
                min_peak_fraction=args.pitch_min_peak_fraction,
            )
            power = compute_total_power(X_run, spec_scale=args.spec_scale)
            seg_result = segment_power_by_canary_method(
                pitch,
                power,
                bin_s=args.bin_ms / 1000.0,
                rate_band=(args.canary_rate_min_hz, args.canary_rate_max_hz),
                frac=args.canary_bandpass_frac,
                edge_guard_periods=args.canary_edge_guard_periods,
                pad_periods=args.canary_pad_periods,
                snap=(not args.canary_no_snap),
                snap_periods=args.canary_snap_periods,
                min_points=args.segmentation_min_points,
                drop_edge_segments=(not args.canary_keep_edge_segments),
            )
            for rec in seg_result["segments"]:
                seg_start = int(rec["segment_start_bin"])
                seg_end = int(rec["segment_end_bin"])
                if (seg_end - seg_start) < args.min_segment_bins:
                    continue
                if args.max_segment_bins is not None and (seg_end - seg_start) > args.max_segment_bins:
                    continue
                X_seg = X_run[seg_start:seg_end, :]
                pitch_seg = pitch[seg_start:seg_end]
                out[period].append({
                    "animal_id": args.animal_id,
                    "label": str(target_label),
                    "period": period,
                    "run_index": run_i,
                    "segment_id_in_period": segment_counter,
                    "global_start_bin": int(start + seg_start),
                    "global_end_bin": int(start + seg_end),
                    "local_run_start_bin": int(seg_start),
                    "local_run_end_bin": int(seg_end),
                    "duration_s": float((seg_end - seg_start) * args.bin_ms / 1000.0),
                    "dominant_pitch_modulation_freq_hz": rec.get("dominant_pitch_modulation_freq_hz", np.nan),
                    "X": X_seg,
                    "pitch_khz": pitch_seg,
                    "pitch_contour_inlier": True,
                })
                segment_counter += 1

        print(f"    segmented syllables: {len(out[period])}")
    return out


def apply_pitch_contour_filter_by_period(segments_by_period, n_points=101, filter_sd=2.0, max_outside_fraction=0.05):
    for period, entries in segments_by_period.items():
        if len(entries) == 0:
            continue
        mask, info = compute_pitch_contour_inlier_mask(entries, n_points=n_points,
                                                       filter_sd=filter_sd,
                                                       max_outside_fraction=max_outside_fraction)
        for ent, keep in zip(entries, mask):
            ent["pitch_contour_inlier"] = bool(keep)
        print(f"[INFO] {period}: pitch-contour filter kept {info['n_kept']} / {info['n_total_entries']} segmented syllables")
    return segments_by_period


def prepare_segment_vectors(segments_by_period, args):
    """Add normalized spectrogram image and unit vector to each segment entry."""
    for period, entries in segments_by_period.items():
        for ent in entries:
            X = ent["X"]
            X = crop_frequency_band(X, max_freq_khz=args.max_freq_khz,
                                    freq_min_khz=args.corr_freq_min_khz,
                                    freq_max_khz=args.corr_freq_max_khz)
            Y = transform_spectrogram_for_corr(X, spec_scale=args.spec_scale,
                                               corr_transform=args.corr_transform)
            Y_norm = time_normalize_spectrogram(Y, n_time=args.corr_time_bins)
            if Y_norm is None:
                ent["corr_usable"] = False
                ent["X_norm"] = None
                ent["corr_vector"] = None
                continue
            v = vectorize_for_corr(Y_norm)
            ent["corr_usable"] = v is not None
            ent["X_norm"] = Y_norm
            ent["corr_vector"] = v
    return segments_by_period


def select_entries(segments_by_period, period, inliers_only=True):
    entries = segments_by_period.get(period, [])
    out = []
    for ent in entries:
        if inliers_only and not bool(ent.get("pitch_contour_inlier", True)):
            continue
        if not ent.get("corr_usable", False):
            continue
        out.append(ent)
    return out


def combined_entries(segments_by_period, periods, inliers_only=True):
    out = []
    for p in periods:
        out.extend(select_entries(segments_by_period, p, inliers_only=inliers_only))
    return out


def build_template(entries):
    if len(entries) == 0:
        return None, None
    mats = [ent["X_norm"] for ent in entries if ent.get("X_norm") is not None]
    if len(mats) == 0:
        return None, None
    template = np.nanmean(np.stack(mats, axis=0), axis=0)
    vec = vectorize_for_corr(template)
    return template, vec


def sample_entries_evenly(entries, max_n):
    if max_n is None or len(entries) <= max_n:
        return entries
    idx = np.linspace(0, len(entries) - 1, max_n).round().astype(int)
    idx = np.unique(idx)
    return [entries[i] for i in idx]


def pairwise_correlations(entries, max_segments=300, max_pairs=50000, rng=None):
    entries = sample_entries_evenly(entries, max_segments)
    vecs = [ent["corr_vector"] for ent in entries if ent.get("corr_vector") is not None]
    if len(vecs) < 2:
        return np.array([], dtype=float)
    V = np.vstack(vecs)
    C = V @ V.T
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size > max_pairs:
        if rng is None:
            rng = np.random.default_rng(0)
        vals = rng.choice(vals, size=max_pairs, replace=False)
    return vals.astype(float)


def clean_values(x):
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def compute_stats(pre, post, measure_name):
    pre = clean_values(pre)
    post = clean_values(post)
    row = {
        "measure": measure_name,
        "n_pre": int(pre.size),
        "n_post": int(post.size),
        "mean_pre": float(np.nanmean(pre)) if pre.size else np.nan,
        "mean_post": float(np.nanmean(post)) if post.size else np.nan,
        "median_pre": float(np.nanmedian(pre)) if pre.size else np.nan,
        "median_post": float(np.nanmedian(post)) if post.size else np.nan,
        "delta_median_post_minus_pre": float(np.nanmedian(post) - np.nanmedian(pre)) if pre.size and post.size else np.nan,
        "q10_pre": float(np.nanpercentile(pre, 10)) if pre.size else np.nan,
        "q10_post": float(np.nanpercentile(post, 10)) if post.size else np.nan,
        "q90_pre": float(np.nanpercentile(pre, 90)) if pre.size else np.nan,
        "q90_post": float(np.nanpercentile(post, 90)) if post.size else np.nan,
        "ks_statistic": np.nan,
        "ks_p_value": np.nan,
        "mannwhitney_p_value": np.nan,
        "wasserstein_distance": np.nan,
        "scipy_available": SCIPY_AVAILABLE,
    }
    if SCIPY_AVAILABLE and pre.size > 0 and post.size > 0:
        ks = stats.ks_2samp(pre, post, alternative="two-sided", mode="auto")
        mw = stats.mannwhitneyu(pre, post, alternative="two-sided", method="auto")
        row["ks_statistic"] = float(ks.statistic)
        row["ks_p_value"] = float(ks.pvalue)
        row["mannwhitney_p_value"] = float(mw.pvalue)
        row["wasserstein_distance"] = float(stats.wasserstein_distance(pre, post))
    return row


def ecdf_xy(values):
    x = np.sort(clean_values(values))
    if x.size == 0:
        return x, np.array([], dtype=float)
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def plot_ecdf(pre, post, out_png, title, xlabel):
    pre_x, pre_y = ecdf_xy(pre)
    post_x, post_y = ecdf_xy(post)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if pre_x.size:
        ax.step(pre_x, pre_y, where="post", lw=2.2, color=PERIOD_COLORS["combined_pre"], label=f"Pre-lesion (N={pre_x.size:,})")
    if post_x.size:
        ax.step(post_x, post_y, where="post", lw=2.2, color=PERIOD_COLORS["combined_post"], label=f"Post-lesion (N={post_x.size:,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_overlay_hist(pre, post, out_png, title, xlabel, bins=80):
    pre = clean_values(pre)
    post = clean_values(post)
    if pre.size == 0 and post.size == 0:
        return
    allv = np.concatenate([pre, post]) if pre.size and post.size else (pre if pre.size else post)
    lo, hi = np.nanpercentile(allv, [0, 100])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1, 1
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if pre.size:
        ax.hist(pre, bins=edges, density=True, alpha=0.45, color=PERIOD_COLORS["combined_pre"], label=f"Pre-lesion (N={pre.size:,})")
        ax.axvline(np.nanmedian(pre), color=PERIOD_COLORS["combined_pre"], ls="--", lw=1.3)
    if post.size:
        ax.hist(post, bins=edges, density=True, alpha=0.45, color=PERIOD_COLORS["combined_post"], label=f"Post-lesion (N={post.size:,})")
        ax.axvline(np.nanmedian(post), color=PERIOD_COLORS["combined_post"], ls="--", lw=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_period_templates(templates_by_period, out_png, animal_id, target_label, max_freq_khz=12.0):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    axes = axes.ravel()
    finite_vals = []
    for template in templates_by_period.values():
        if template is not None:
            finite_vals.append(template[np.isfinite(template)])
    if finite_vals:
        allv = np.concatenate(finite_vals)
        vmin = np.nanpercentile(allv, 5)
        vmax = np.nanpercentile(allv, 99.5)
    else:
        vmin, vmax = 0, 1
    for ax, period in zip(axes, PERIOD_ORDER):
        X = templates_by_period.get(period)
        if X is None:
            ax.text(0.5, 0.5, "No template", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(PERIOD_TITLES[period])
            continue
        ax.imshow(X.T, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[0, 100, 0, max_freq_khz], cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.set_title(PERIOD_TITLES[period])
        ax.set_xlabel("Normalized syllable time (%)")
        ax.set_ylabel("Frequency (kHz)")
    fig.suptitle(f"{animal_id} — label {target_label} — mean normalized spectrogram templates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_high_low_examples(template, post_entries, out_png, animal_id, target_label, max_freq_khz=12.0, n_each=3):
    if template is None or len(post_entries) == 0:
        return
    usable = [e for e in post_entries if np.isfinite(e.get("corr_late_pre_template", np.nan))]
    if len(usable) == 0:
        return
    usable = sorted(usable, key=lambda e: e["corr_late_pre_template"])
    chosen = usable[:n_each] + usable[-n_each:]
    titles = [f"low corr={e['corr_late_pre_template']:.2f}" for e in usable[:n_each]] + [f"high corr={e['corr_late_pre_template']:.2f}" for e in usable[-n_each:]]
    fig, axes = plt.subplots(2, max(n_each, 3) + 1, figsize=(15, 6), squeeze=False)
    # first column template spanning rows visually repeated
    for r in range(2):
        ax = axes[r, 0]
        ax.imshow(template.T, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[0, 100, 0, max_freq_khz], cmap="gray_r")
        ax.set_title("late-pre template" if r == 0 else "")
        ax.set_xlabel("time (%)")
        ax.set_ylabel("Freq (kHz)")
    for i, ent in enumerate(chosen):
        r = 0 if i < n_each else 1
        c = (i % n_each) + 1
        ax = axes[r, c]
        X = ent["X_norm"]
        ax.imshow(X.T, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[0, 100, 0, max_freq_khz], cmap="gray_r")
        ax.set_title(titles[i])
        ax.set_xlabel("time (%)")
        ax.set_ylabel("Freq (kHz)")
    fig.suptitle(f"{animal_id} — label {target_label} — post-lesion examples vs late-pre template", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def write_segments_csv(segments_by_period, out_csv):
    rows = []
    for period, entries in segments_by_period.items():
        for e in entries:
            rows.append({
                "animal_id": e["animal_id"],
                "label": e["label"],
                "period": period,
                "run_index": e["run_index"],
                "segment_id_in_period": e["segment_id_in_period"],
                "global_start_bin": e["global_start_bin"],
                "global_end_bin": e["global_end_bin"],
                "duration_s": e["duration_s"],
                "dominant_pitch_modulation_freq_hz": e.get("dominant_pitch_modulation_freq_hz", np.nan),
                "pitch_contour_inlier": bool(e.get("pitch_contour_inlier", True)),
                "corr_usable": bool(e.get("corr_usable", False)),
                "corr_late_pre_template": e.get("corr_late_pre_template", np.nan),
                "corr_combined_pre_template": e.get("corr_combined_pre_template", np.nan),
                "corr_own_period_template": e.get("corr_own_period_template", np.nan),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Segment a target cluster and compare spectrogram cross-correlations pre vs post lesion.")
    p.add_argument("--npz-path", required=True)
    p.add_argument("--cluster-label", required=True)
    p.add_argument("--animal-id", required=True)
    p.add_argument("--metadata-excel-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--spec-key", default="s")
    p.add_argument("--label-key", default="hdbscan_labels")
    p.add_argument("--file-index-key", default="file_indices")
    p.add_argument("--file-map-key", default="file_map")
    p.add_argument("--metadata-sheet", default="metadata")
    p.add_argument("--animal-id-col", default="Animal ID")
    p.add_argument("--treatment-date-col", default="Treatment date")
    p.add_argument("--bin-ms", type=float, default=2.7)
    p.add_argument("--max-freq-khz", type=float, default=12.0)
    p.add_argument("--spec-scale", default="linear", choices=["linear", "log10", "loge", "shift"])
    p.add_argument("--include-treatment-day-in-post", action="store_true")
    p.add_argument("--split-mode", default="file_half", choices=["file_half", "file_median"])

    # Pitch/canary segmentation args, matched to previous script defaults.
    p.add_argument("--pitch-band-min-khz", type=float, default=0.8)
    p.add_argument("--pitch-band-max-khz", type=float, default=10.0)
    p.add_argument("--pitch-min-peak-fraction", type=float, default=0.0)
    p.add_argument("--pitch-smoothing-window", type=int, default=5)
    p.add_argument("--canary-rate-min-hz", type=float, default=2.0)
    p.add_argument("--canary-rate-max-hz", type=float, default=20.0)
    p.add_argument("--canary-bandpass-frac", type=float, default=0.5)
    p.add_argument("--canary-edge-guard-periods", type=float, default=0.5)
    p.add_argument("--canary-pad-periods", type=float, default=3.0)
    p.add_argument("--canary-no-snap", action="store_true")
    p.add_argument("--canary-snap-periods", type=float, default=0.5)
    p.add_argument("--canary-keep-edge-segments", action="store_true")
    p.add_argument("--segmentation-min-points", type=int, default=12)
    p.add_argument("--min-segment-bins", type=int, default=3)
    p.add_argument("--max-segment-bins", type=int, default=None)

    # Pitch contour QC filter.
    p.add_argument("--apply-pitch-contour-filter", action="store_true", default=True)
    p.add_argument("--no-pitch-contour-filter", dest="apply_pitch_contour_filter", action="store_false")
    p.add_argument("--pitch-contour-qc-points", type=int, default=101)
    p.add_argument("--pitch-contour-filter-sd", type=float, default=2.0)
    p.add_argument("--pitch-contour-max-outside-fraction", type=float, default=0.05)

    # Cross-correlation args.
    p.add_argument("--corr-time-bins", type=int, default=64,
                   help="Number of time bins for time-normalized segmented syllable spectrograms.")
    p.add_argument("--corr-freq-min-khz", type=float, default=0.0)
    p.add_argument("--corr-freq-max-khz", type=float, default=None)
    p.add_argument("--corr-transform", default="log_power", choices=["raw", "linear_power", "log_power"])
    p.add_argument("--pairwise-max-segments-per-group", type=int, default=300)
    p.add_argument("--pairwise-max-pairs", type=int, default=50000)
    p.add_argument("--hist-bins", type=int, default=80)
    p.add_argument("--random-seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not SCIPY_AVAILABLE:
        raise ImportError("This script requires scipy for canary segmentation. Install with: conda install scipy")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    bin_s = args.bin_ms / 1000.0

    print("[INFO] Loading NPZ...")
    data = np.load(args.npz_path, allow_pickle=True)
    s_tx_f = np.asarray(data[args.spec_key], dtype=float)
    labels = normalize_label_array(data[args.label_key])
    file_indices = np.asarray(data[args.file_index_key]).astype(int)
    file_map = get_file_map_dict(data[args.file_map_key])

    if s_tx_f.shape[0] != len(labels) or s_tx_f.shape[0] != len(file_indices):
        raise ValueError(
            "Shape mismatch: spectrogram rows, labels, and file_indices must match.\n"
            f"spectrogram rows={s_tx_f.shape[0]}, labels={len(labels)}, file_indices={len(file_indices)}"
        )

    print("[INFO] Parsing file timestamps...")
    unique_file_indices = np.unique(file_indices)
    file_time_lookup = {}
    for fidx in unique_file_indices:
        fname = file_entry_to_name(file_map[int(fidx)])
        file_time_lookup[int(fidx)] = parse_datetime_from_filename(fname)

    print("[INFO] Loading treatment date...")
    treatment_date = load_treatment_date(
        args.metadata_excel_path,
        args.animal_id,
        metadata_sheet=args.metadata_sheet,
        animal_id_col=args.animal_id_col,
        treatment_date_col=args.treatment_date_col,
    )
    print(f"[INFO] Treatment date: {treatment_date.date()}")

    print("[INFO] Splitting files into early/late pre/post...")
    files_by_period = split_by_file_period(
        unique_file_indices,
        file_time_lookup,
        treatment_date,
        include_treatment_day_in_post=args.include_treatment_day_in_post,
        split_mode=args.split_mode,
    )
    for period in PERIOD_ORDER:
        print(f"  {period}: {len(files_by_period[period])} files")

    print("[INFO] Segmenting target cluster with canary method...")
    segments_by_period = segment_cluster_periods(
        s_tx_f=s_tx_f,
        labels=labels,
        file_indices=file_indices,
        files_by_period=files_by_period,
        target_label=str(args.cluster_label),
        args=args,
    )

    if args.apply_pitch_contour_filter:
        segments_by_period = apply_pitch_contour_filter_by_period(
            segments_by_period,
            n_points=args.pitch_contour_qc_points,
            filter_sd=args.pitch_contour_filter_sd,
            max_outside_fraction=args.pitch_contour_max_outside_fraction,
        )

    print("[INFO] Preparing normalized spectrogram vectors...")
    segments_by_period = prepare_segment_vectors(segments_by_period, args)

    inliers_only = args.apply_pitch_contour_filter
    period_entries = {p: select_entries(segments_by_period, p, inliers_only=inliers_only) for p in PERIOD_ORDER}
    combined_pre = combined_entries(segments_by_period, ["early_pre", "late_pre"], inliers_only=inliers_only)
    combined_post = combined_entries(segments_by_period, ["early_post", "late_post"], inliers_only=inliers_only)

    for p in PERIOD_ORDER:
        print(f"[INFO] {p}: {len(period_entries[p])} usable segmented syllables for correlation")
    print(f"[INFO] combined_pre: {len(combined_pre)}; combined_post: {len(combined_post)}")

    # Templates.
    period_templates = {}
    period_template_vecs = {}
    for p in PERIOD_ORDER:
        template, vec = build_template(period_entries[p])
        period_templates[p] = template
        period_template_vecs[p] = vec

    late_pre_template, late_pre_vec = build_template(period_entries["late_pre"])
    combined_pre_template, combined_pre_vec = build_template(combined_pre)
    if late_pre_vec is None:
        print("[WARN] No late-pre template available; falling back to combined-pre template for late-pre-template correlations.")
        late_pre_template, late_pre_vec = combined_pre_template, combined_pre_vec

    # Add template correlations to each segment.
    for period in PERIOD_ORDER:
        for ent in period_entries[period]:
            ent["corr_late_pre_template"] = corr_between_unit_vectors(ent["corr_vector"], late_pre_vec)
            ent["corr_combined_pre_template"] = corr_between_unit_vectors(ent["corr_vector"], combined_pre_vec)
            ent["corr_own_period_template"] = corr_between_unit_vectors(ent["corr_vector"], period_template_vecs.get(period))

    # Also copy corr values to non-selected entries where possible so the CSV records all segments.
    for period in PERIOD_ORDER:
        for ent in segments_by_period[period]:
            if not ent.get("corr_usable", False):
                continue
            ent["corr_late_pre_template"] = corr_between_unit_vectors(ent["corr_vector"], late_pre_vec)
            ent["corr_combined_pre_template"] = corr_between_unit_vectors(ent["corr_vector"], combined_pre_vec)
            ent["corr_own_period_template"] = corr_between_unit_vectors(ent["corr_vector"], period_template_vecs.get(period))

    segment_csv = out_dir / f"{args.animal_id}_label{args.cluster_label}_spectrogram_crosscorr_segments.csv"
    seg_df = write_segments_csv(segments_by_period, segment_csv)

    # Template-correlation distributions.
    pre_template_corr = np.array([e["corr_late_pre_template"] for e in combined_pre if np.isfinite(e.get("corr_late_pre_template", np.nan))], dtype=float)
    post_template_corr = np.array([e["corr_late_pre_template"] for e in combined_post if np.isfinite(e.get("corr_late_pre_template", np.nan))], dtype=float)
    template_stats = compute_stats(pre_template_corr, post_template_corr, "correlation_to_late_pre_template")
    pd.DataFrame([template_stats]).to_csv(out_dir / f"{args.animal_id}_label{args.cluster_label}_template_correlation_stats.csv", index=False)
    print(f"[SAVED] {out_dir / f'{args.animal_id}_label{args.cluster_label}_template_correlation_stats.csv'}")

    plot_ecdf(
        pre_template_corr,
        post_template_corr,
        out_dir / f"{args.animal_id}_label{args.cluster_label}_template_corr_to_late_pre_ecdf_pre_post.png",
        title=(f"{args.animal_id} — label {args.cluster_label} — correlation to late-pre template\n"
               f"Δmedian post-pre = {template_stats['delta_median_post_minus_pre']:.3g}"),
        xlabel="Spectrogram correlation to late-pre template",
    )
    plot_overlay_hist(
        pre_template_corr,
        post_template_corr,
        out_dir / f"{args.animal_id}_label{args.cluster_label}_template_corr_to_late_pre_overlay_histogram.png",
        title=(f"{args.animal_id} — label {args.cluster_label} — correlation to late-pre template\n"
               f"MW p={template_stats['mannwhitney_p_value']:.3g}; KS p={template_stats['ks_p_value']:.3g}"),
        xlabel="Spectrogram correlation to late-pre template",
        bins=args.hist_bins,
    )

    # Within-condition pairwise correlations.
    rng = np.random.default_rng(args.random_seed)
    pairwise = {}
    for name, entries in {
        "early_pre": period_entries["early_pre"],
        "late_pre": period_entries["late_pre"],
        "early_post": period_entries["early_post"],
        "late_post": period_entries["late_post"],
        "combined_pre": combined_pre,
        "combined_post": combined_post,
    }.items():
        pairwise[name] = pairwise_correlations(
            entries,
            max_segments=args.pairwise_max_segments_per_group,
            max_pairs=args.pairwise_max_pairs,
            rng=rng,
        )
        print(f"[INFO] {name}: {pairwise[name].size} pairwise correlations")

    pairwise_stats = compute_stats(pairwise["combined_pre"], pairwise["combined_post"], "within_condition_pairwise_correlation")
    pd.DataFrame([pairwise_stats]).to_csv(out_dir / f"{args.animal_id}_label{args.cluster_label}_pairwise_correlation_stats.csv", index=False)
    print(f"[SAVED] {out_dir / f'{args.animal_id}_label{args.cluster_label}_pairwise_correlation_stats.csv'}")

    pairwise_rows = []
    for name, vals in pairwise.items():
        for v in vals:
            pairwise_rows.append({"group": name, "pairwise_spectrogram_correlation": float(v)})
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(out_dir / f"{args.animal_id}_label{args.cluster_label}_pairwise_correlation_values.csv", index=False)
    print(f"[SAVED] {out_dir / f'{args.animal_id}_label{args.cluster_label}_pairwise_correlation_values.csv'}")

    plot_ecdf(
        pairwise["combined_pre"],
        pairwise["combined_post"],
        out_dir / f"{args.animal_id}_label{args.cluster_label}_within_condition_pairwise_corr_ecdf_pre_post.png",
        title=(f"{args.animal_id} — label {args.cluster_label} — within-condition pairwise spectrogram correlation\n"
               f"Δmedian post-pre = {pairwise_stats['delta_median_post_minus_pre']:.3g}"),
        xlabel="Within-condition pairwise spectrogram correlation",
    )
    plot_overlay_hist(
        pairwise["combined_pre"],
        pairwise["combined_post"],
        out_dir / f"{args.animal_id}_label{args.cluster_label}_within_condition_pairwise_corr_overlay_histogram.png",
        title=(f"{args.animal_id} — label {args.cluster_label} — within-condition pairwise spectrogram correlation\n"
               f"MW p={pairwise_stats['mannwhitney_p_value']:.3g}; KS p={pairwise_stats['ks_p_value']:.3g}"),
        xlabel="Within-condition pairwise spectrogram correlation",
        bins=args.hist_bins,
    )

    # QC figures.
    plot_period_templates(
        period_templates,
        out_dir / f"{args.animal_id}_label{args.cluster_label}_period_mean_spectrogram_templates.png",
        animal_id=args.animal_id,
        target_label=args.cluster_label,
        max_freq_khz=(args.corr_freq_max_khz if args.corr_freq_max_khz is not None else args.max_freq_khz),
    )
    plot_high_low_examples(
        late_pre_template,
        combined_post,
        out_dir / f"{args.animal_id}_label{args.cluster_label}_post_examples_low_high_template_corr.png",
        animal_id=args.animal_id,
        target_label=args.cluster_label,
        max_freq_khz=(args.corr_freq_max_khz if args.corr_freq_max_khz is not None else args.max_freq_khz),
        n_each=3,
    )

    # Small README.
    readme = out_dir / "README_spectrogram_crosscorr.txt"
    readme.write_text(
        "Spectrogram cross-correlation analysis using canary segmentation.\n\n"
        "Key files:\n"
        f"- {segment_csv.name}: one row per segmented syllable with template-correlation values.\n"
        f"- {args.animal_id}_label{args.cluster_label}_template_correlation_stats.csv: pre/post stats for correlation to late-pre template.\n"
        f"- {args.animal_id}_label{args.cluster_label}_pairwise_correlation_stats.csv: pre/post stats for within-condition pairwise correlation.\n\n"
        "Main interpretation:\n"
        "- Higher correlation means more similar spectrogram structure.\n"
        "- Lower post-lesion template correlations suggest post-lesion syllables are less similar to the late-pre template.\n"
        "- Lower post-lesion within-condition pairwise correlations suggest post-lesion syllables are less internally stereotyped.\n"
        "- Segment/bin-level p-values are exploratory; summarize effects at syllable and bird level for paper-level inference.\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE]")


if __name__ == "__main__":
    main()
