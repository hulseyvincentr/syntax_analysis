#!/usr/bin/env python3
"""
cluster_pitch_entropy_panels.py

For one HDBSCAN / phrase cluster label, generate:
1) full-run context spectrograms
2) stitched spectrogram panels of only that label for early_pre / late_pre / early_post / late_post
3) spectrograms with overlaid Wiener entropy + pitch derivative traces
4) CSV of feature time series for the stitched selected rows

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
    """
    base = os.path.basename(str(file_name))

    patterns = [
        # 2024_04_01_12_34_56 or 2024-04-01-12-34-56
        r'(?P<y>20\d{2})[_-](?P<m>\d{2})[_-](?P<d>\d{2})[_-](?P<H>\d{2})[_-](?P<M>\d{2})(?:[_-](?P<S>\d{2}))?',
        # 20240401_123456
        r'(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        # 240401_123456
        r'(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
        # date only: 2024_04_01 or 2024-04-01
        r'(?P<y>20\d{2})[_-](?P<m>\d{2})[_-](?P<d>\d{2})',
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
    """Convert spectrogram values to strictly positive power-like values."""
    eps = 1e-12
    X = np.asarray(X, dtype=float)

    if spec_scale == "linear":
        Y = X.copy()
        row_min = np.nanmin(Y, axis=1, keepdims=True)
        row_min[~np.isfinite(row_min)] = 0
        bad = row_min <= 0
        Y[bad[:, 0], :] = Y[bad[:, 0], :] - row_min[bad[:, 0], :] + eps
        Y[Y < eps] = eps
        return Y

    if spec_scale == "log10":
        return np.power(10.0, X)

    if spec_scale == "loge":
        return np.exp(X)

    if spec_scale == "shift":
        Y = X.copy()
        row_min = np.nanmin(Y, axis=1, keepdims=True)
        row_min[~np.isfinite(row_min)] = 0
        Y = Y - row_min + eps
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


def compute_pitch_trace_and_derivative(
        X_tx_f,
        bin_s,
        max_freq_khz=12.0,
        pitch_band_min_khz=0.8,
        pitch_band_max_khz=10.0,
        spec_scale="linear",
        smoothing_window=5,
        min_peak_fraction=0.08):
    """
    Estimate pitch as the dominant spectral peak in a frequency band,
    then compute absolute first derivative in kHz/s.
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

        if peak_fraction < min_peak_fraction:
            continue

        pitch_khz[i] = band_freqs[peak_idx]

    pitch_khz = moving_average_nan(pitch_khz, smoothing_window)

    pitch_derivative = np.full(T, np.nan, dtype=float)
    for i in range(1, T):
        if np.isfinite(pitch_khz[i]) and np.isfinite(pitch_khz[i - 1]):
            pitch_derivative[i] = abs(pitch_khz[i] - pitch_khz[i - 1]) / bin_s

    return pitch_khz, pitch_derivative


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


def plot_full_run_contexts(
        s_tx_f,
        runs_by_period,
        out_png,
        target_label,
        animal_id,
        bin_s=0.0027,
        max_freq_khz=12.0,
        context_bins=300,
        max_runs_per_period=5):
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

            imshow_spectrogram(ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)
            ax.set_ylabel("Freq (kHz)")

            x0 = (start - lo) * bin_s
            width = (end - start) * bin_s
            rect = Rectangle((x0, 0), width, max_freq_khz, fill=False, edgecolor=color, linewidth=2.0)
            ax.add_patch(rect)

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
                    f"Colored box = target label run, ±{context_bins} context bins",
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
    """Make a 4-row figure with one stitched row per period."""
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), squeeze=False)
    axes = axes.ravel()

    for ax, period in zip(axes, PERIOD_ORDER):
        seg_rows = rows_by_period.get(period, [])
        if len(seg_rows) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{PERIOD_TITLES[period]}: no runs found", ha="center", va="center", fontsize=11)
            continue

        X, _source_bins, _is_sep, _seg_ids = stitch_segments(s_tx_f, seg_rows[0], separator_bins=separator_bins)
        imshow_spectrogram(ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)
        ax.set_ylabel("Freq (kHz)")
        ax.text(
            -0.01, 0.5,
            f"{PERIOD_TITLES[period]}\nstitched selected bins\n{len(seg_rows[0])} run(s)",
            transform=ax.transAxes,
            ha="right", va="center", fontsize=9
        )

    axes[0].set_title(
        f"{animal_id} — label {target_label} — stitched spectrograms of only this label\n"
        f"One row each for early pre / late pre / early post / late post",
        fontsize=13
    )
    axes[-1].set_xlabel("Stitched selected time bins (s)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_feature_overlay_rows(
        s_tx_f,
        row_segments,
        out_png,
        out_csv,
        animal_id,
        target_label,
        period_name="early_pre",
        bin_s=0.0027,
        max_freq_khz=12.0,
        spec_scale="linear",
        pitch_band_min_khz=0.8,
        pitch_band_max_khz=10.0,
        separator_bins=2):
    """Plot stitched selected rows with overlaid Wiener entropy and pitch derivative."""
    if len(row_segments) == 0:
        print(f"[WARN] No stitched rows available for {period_name}; skipping feature overlay.")
        return

    fig, axes = plt.subplots(len(row_segments), 1, figsize=(18, max(3 * len(row_segments), 5)), squeeze=False)
    axes = axes.ravel()
    all_csv_rows = []

    for row_idx, seg_list in enumerate(row_segments):
        X, source_bins, is_sep, seg_ids = stitch_segments(s_tx_f, seg_list, separator_bins=separator_bins)
        ax = axes[row_idx]
        imshow_spectrogram(ax, X, bin_s=bin_s, max_freq_khz=max_freq_khz)

        wiener = compute_wiener_entropy(X, spec_scale=spec_scale)
        pitch_khz, pitch_deriv_khz_per_s = compute_pitch_trace_and_derivative(
            X,
            bin_s=bin_s,
            max_freq_khz=max_freq_khz,
            pitch_band_min_khz=pitch_band_min_khz,
            pitch_band_max_khz=pitch_band_max_khz,
            spec_scale=spec_scale,
            smoothing_window=5,
            min_peak_fraction=0.08
        )

        wiener[is_sep] = np.nan
        pitch_khz[is_sep] = np.nan
        pitch_deriv_khz_per_s[is_sep] = np.nan

        wiener_norm = robust_minmax(wiener)
        pitch_deriv_norm = robust_minmax(pitch_deriv_khz_per_s)
        t = np.arange(X.shape[0]) * bin_s

        ax2 = ax.twinx()
        ax2.plot(t, wiener_norm, color="#CC79A7", lw=1.8, label="Wiener entropy")
        ax2.plot(t, pitch_deriv_norm, color="#009E73", lw=1.8, label="Pitch derivative")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Normalized feature value")
        ax2.tick_params(axis="y", labelsize=8)

        sep_idx = np.where(is_sep)[0]
        if len(sep_idx) > 0:
            transition_idx = [sep_idx[0]]
            for i in range(1, len(sep_idx)):
                if sep_idx[i] != sep_idx[i - 1] + 1:
                    transition_idx.append(sep_idx[i])
            for si in transition_idx:
                ax.axvline(si * bin_s, color="white", lw=0.8, alpha=0.6, linestyle="--")

        ax.set_ylabel("Freq (kHz)")
        ax.text(
            -0.01, 0.5,
            f"{period_name}\nrow {row_idx + 1}\n{X.shape[0]} bins",
            transform=ax.transAxes,
            ha="right", va="center", fontsize=9
        )

        if row_idx == 0:
            ax.set_title(
                f"{animal_id} — label {target_label} — {period_name}_expanded_full_runs\n"
                f"Spectrogram with overlaid Wiener entropy and pitch derivative",
                fontsize=13
            )
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc="upper right", frameon=True)

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
                "wiener_entropy": wiener[i],
                "pitch_khz": pitch_khz[i],
                "pitch_derivative_khz_per_s": pitch_deriv_khz_per_s[i],
                "pitch_derivative_hz_per_s": pitch_deriv_khz_per_s[i] * 1000.0 if np.isfinite(pitch_deriv_khz_per_s[i]) else np.nan,
            })

    axes[-1].set_xlabel("Stitched selected time bins (s)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    df = pd.DataFrame(all_csv_rows)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")


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

    parser.add_argument("--context-bins", type=int, default=300, help="Context bins on each side of a run")
    parser.add_argument("--max-context-runs-per-period", type=int, default=5, help="Rows per period in context figure")
    parser.add_argument("--period-max-bins", type=int, default=2000, help="Max bins per stitched row")
    parser.add_argument("--feature-rows", type=int, default=5, help="Rows for feature overlay figure")
    parser.add_argument("--feature-period", default="early_pre", choices=PERIOD_ORDER,
                        help="Which period to use for the feature overlay figure")
    parser.add_argument("--separator-bins", type=int, default=2, help="Blank bins inserted between stitched runs")

    parser.add_argument("--pitch-band-min-khz", type=float, default=0.8, help="Lower frequency bound for pitch estimation")
    parser.add_argument("--pitch-band-max-khz", type=float, default=10.0, help="Upper frequency bound for pitch estimation")

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
        runs_by_period=runs_by_period,
        out_png=out_context_png,
        target_label=target_label,
        animal_id=args.animal_id,
        bin_s=bin_s,
        max_freq_khz=args.max_freq_khz,
        context_bins=args.context_bins,
        max_runs_per_period=args.max_context_runs_per_period,
    )

    print("[INFO] Building stitched rows by period...")
    rows_by_period = {}
    for period in PERIOD_ORDER:
        rows_by_period[period] = build_stitched_rows(
            runs_by_period[period],
            max_rows=1,
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

    print(f"[INFO] Building feature overlay rows for {args.feature_period}...")
    feature_rows = build_stitched_rows(
        runs_by_period[args.feature_period],
        max_rows=args.feature_rows,
        max_bins_per_row=args.period_max_bins,
        separator_bins=args.separator_bins
    )

    out_feature_png = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_{args.feature_period}_pitch_derivative_wiener_entropy_overlay.png"
    )
    out_feature_csv = os.path.join(
        args.out_dir,
        f"{args.animal_id}_label{target_label}_{args.feature_period}_pitch_derivative_wiener_entropy.csv"
    )

    plot_feature_overlay_rows(
        s_tx_f=s_tx_f,
        row_segments=feature_rows,
        out_png=out_feature_png,
        out_csv=out_feature_csv,
        animal_id=args.animal_id,
        target_label=target_label,
        period_name=args.feature_period,
        bin_s=bin_s,
        max_freq_khz=args.max_freq_khz,
        spec_scale=args.spec_scale,
        pitch_band_min_khz=args.pitch_band_min_khz,
        pitch_band_max_khz=args.pitch_band_max_khz,
        separator_bins=args.separator_bins,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
