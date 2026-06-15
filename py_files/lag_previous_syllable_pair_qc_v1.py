#!/usr/bin/env python3
"""
lag_previous_syllable_pair_qc_v1.py

Companion QC script for phrase_position_acoustic_crosscorr_v11_bugfixes.py.

Goal
----
Make visual QC figures showing pairs of syllable spectrograms that are N repeats
apart within the same repeated phrase, along with the saved and/or recomputed
cross-correlation value.

Typical use case
----------------
After running phrase_position_acoustic_crosscorr_v11_bugfixes.py, you should have
one segment-level CSV such as:

    USA5288_label17_phrase_position_segment_features.csv

This QC script reads that CSV plus the original NPZ file. For each eligible row,
it finds the earlier segment in the same phrase whose repeat index is lag steps
back, extracts both spectrogram snippets using the saved global bin boundaries,
and plots them side-by-side.

Outputs
-------
- selected_lag<lag>_pairs.csv
- lag<lag>_pair_qc.pdf        multipage PDF, one pair per page
- individual_pngs/*.png       optional individual pair PNGs

Example
-------
python lag_previous_syllable_pair_qc_v1.py \
  --npz-path /Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz \
  --segment-csv /Volumes/my_own_SSD/updated_AreaX_outputs/.../USA5288_label17_phrase_position_segment_features.csv \
  --out-dir /Volumes/my_own_SSD/updated_AreaX_outputs/.../USA5288_label17_lag10_pair_QC \
  --lag 10 \
  --n-pairs-total 24 \
  --sample-mode quantile_corr \
  --plot-freq-min-khz 0.8 \
  --plot-freq-max-khz 10.0 \
  --recompute-corr
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


PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"all", "none"}:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_float(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def clean_label_string(x) -> str:
    """Normalize labels so 17, 17.0, np.int64(17), and '17.0' all match as '17'."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s == "":
        return s
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_spectrogram_from_npz(npz_path: str, spec_key: str, max_needed_bin: Optional[int] = None) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if spec_key not in data:
        raise KeyError(f"spec_key {spec_key!r} not found in NPZ. Available keys: {list(data.keys())}")
    X = np.asarray(data[spec_key], dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Spectrogram array must be 2D, got shape {X.shape}")

    # Most TweetyBERT NPZs store spectrogram as time x frequency. If not, infer from
    # the largest segment end bin in the CSV.
    if max_needed_bin is not None and max_needed_bin > 0:
        if X.shape[0] < max_needed_bin <= X.shape[1]:
            print("[INFO] Spectrogram appears transposed relative to CSV global bins; transposing.")
            X = X.T
    return X


def to_linear_power(X: np.ndarray, spec_scale: str = "linear") -> np.ndarray:
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


def crop_frequency_band(
    X: np.ndarray,
    max_freq_khz: float = 12.0,
    freq_min_khz: float = 0.0,
    freq_max_khz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a time x frequency spectrogram and return cropped matrix plus freq axis."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {X.shape}")
    if freq_max_khz is None:
        freq_max_khz = max_freq_khz
    F = X.shape[1]
    freqs = np.linspace(0, float(max_freq_khz), F)
    mask = (freqs >= float(freq_min_khz)) & (freqs <= float(freq_max_khz))
    if not np.any(mask):
        print("[WARN] Requested frequency crop has no overlap with axis; using full frequency range.")
        return X, freqs
    return X[:, mask], freqs[mask]


def transform_spectrogram_for_corr(X: np.ndarray, spec_scale: str = "linear", corr_transform: str = "log_power") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if corr_transform == "raw":
        return X.copy()
    power = to_linear_power(X, spec_scale=spec_scale)
    if corr_transform == "linear_power":
        return power
    if corr_transform == "log_power":
        return np.log10(power + 1e-12)
    raise ValueError("corr_transform must be raw, linear_power, or log_power")


def time_normalize_spectrogram(X: np.ndarray, n_time: int = 64) -> Optional[np.ndarray]:
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


def prepare_corr_vector(X_seg: np.ndarray, args) -> Optional[np.ndarray]:
    X_crop, _ = crop_frequency_band(
        X_seg,
        max_freq_khz=args.max_freq_khz,
        freq_min_khz=args.corr_freq_min_khz,
        freq_max_khz=args.corr_freq_max_khz,
    )
    Y = transform_spectrogram_for_corr(
        X_crop,
        spec_scale=args.spec_scale,
        corr_transform=args.corr_transform,
    )
    Yn = time_normalize_spectrogram(Y, n_time=args.corr_time_bins)
    return vectorize_for_corr(Yn)


def corr_between_segments(X_ref: np.ndarray, X_cur: np.ndarray, args) -> float:
    ref_vec = prepare_corr_vector(X_ref, args)
    cur_vec = prepare_corr_vector(X_cur, args)
    if ref_vec is None or cur_vec is None:
        return np.nan
    return float(np.dot(ref_vec, cur_vec))


def extract_segment(X: np.ndarray, row: pd.Series) -> np.ndarray:
    s0 = int(row["segment_start_global_bin"])
    s1 = int(row["segment_end_global_bin"])
    if s1 <= s0:
        raise ValueError(f"Bad segment bounds: start={s0}, end={s1}")
    if s0 < 0 or s1 > X.shape[0]:
        raise IndexError(f"Segment bounds [{s0}, {s1}) exceed spectrogram length {X.shape[0]}")
    return X[s0:s1, :]


def build_reference_lookup(df: pd.DataFrame, lag: int) -> Dict[Tuple[str, str, str, int, int], int]:
    """Map (animal, label, period, phrase_id, repeat_index) -> dataframe index."""
    lookup: Dict[Tuple[str, str, str, int, int], int] = {}
    for idx, row in df.iterrows():
        try:
            key = (
                str(row.get("animal_id", "")),
                clean_label_string(row.get("label", "")),
                str(row.get("period", "")),
                int(row["phrase_id"]),
                int(row["repeat_index_in_phrase"]),
            )
        except Exception:
            continue
        lookup[key] = idx
    return lookup


def prepare_pair_candidates(df: pd.DataFrame, lag: int, corr_col: str, args) -> pd.DataFrame:
    required = [
        "phrase_id",
        "repeat_index_in_phrase",
        "segment_start_global_bin",
        "segment_end_global_bin",
        "period",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Segment CSV is missing required columns: {missing}")

    work = df.copy()
    if "animal_id" not in work.columns:
        work["animal_id"] = ""
    if "label" not in work.columns:
        work["label"] = ""
    work["label"] = work["label"].map(clean_label_string)

    if args.animal_id is not None:
        work = work[work["animal_id"].astype(str) == str(args.animal_id)]
    if args.label is not None:
        want = clean_label_string(args.label)
        work = work[work["label"].map(clean_label_string) == want]

    periods = parse_comma_list(args.periods)
    if periods is not None:
        work = work[work["period"].astype(str).isin(periods)]

    pre_posts = parse_comma_list(args.pre_post)
    if pre_posts is not None and "pre_post" in work.columns:
        work = work[work["pre_post"].astype(str).isin(pre_posts)]

    numeric_cols = [
        "phrase_id",
        "repeat_index_in_phrase",
        "segment_start_global_bin",
        "segment_end_global_bin",
        "phrase_duration_s",
        "elapsed_time_in_phrase_s",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if corr_col in work.columns:
        work[corr_col] = pd.to_numeric(work[corr_col], errors="coerce")
    else:
        work[corr_col] = np.nan

    work = work.dropna(subset=["phrase_id", "repeat_index_in_phrase", "segment_start_global_bin", "segment_end_global_bin"])

    # Build the reference lookup BEFORE filtering current rows to repeat_index > lag.
    # Otherwise valid current segments such as repeat 11 would fail to find their
    # reference repeat 1 in a lag-10 comparison.
    reference_pool = work.copy()

    current_pool = work[work["repeat_index_in_phrase"] > int(lag)].copy()
    if args.min_repeat_index is not None:
        current_pool = current_pool[current_pool["repeat_index_in_phrase"] >= int(args.min_repeat_index)]
    if args.max_repeat_index is not None:
        current_pool = current_pool[current_pool["repeat_index_in_phrase"] <= int(args.max_repeat_index)]
    if args.min_phrase_duration_s is not None and "phrase_duration_s" in current_pool.columns:
        current_pool = current_pool[current_pool["phrase_duration_s"] >= float(args.min_phrase_duration_s)]
    if args.min_corr is not None:
        current_pool = current_pool[current_pool[corr_col] >= float(args.min_corr)]
    if args.max_corr is not None:
        current_pool = current_pool[current_pool[corr_col] <= float(args.max_corr)]

    # Use lookup from the animal/label/period-filtered reference pool, while applying
    # repeat/correlation filters only to the current segment rows.
    lookup = build_reference_lookup(reference_pool, lag=lag)
    pair_rows = []
    for cur_idx, cur in current_pool.iterrows():
        try:
            ref_repeat = int(cur["repeat_index_in_phrase"]) - int(lag)
            key = (
                str(cur.get("animal_id", "")),
                clean_label_string(cur.get("label", "")),
                str(cur.get("period", "")),
                int(cur["phrase_id"]),
                ref_repeat,
            )
        except Exception:
            continue
        ref_idx = lookup.get(key)
        if ref_idx is None:
            continue
        ref = work.loc[ref_idx]
        row = {
            "current_df_index": cur_idx,
            "reference_df_index": ref_idx,
            "lag": int(lag),
            "animal_id": cur.get("animal_id", ""),
            "label": clean_label_string(cur.get("label", "")),
            "period": cur.get("period", ""),
            "pre_post": cur.get("pre_post", ""),
            "phrase_id": int(cur["phrase_id"]),
            "reference_repeat_index": int(ref["repeat_index_in_phrase"]),
            "current_repeat_index": int(cur["repeat_index_in_phrase"]),
            "reference_elapsed_time_s": safe_float(ref.get("elapsed_time_in_phrase_s", np.nan)),
            "current_elapsed_time_s": safe_float(cur.get("elapsed_time_in_phrase_s", np.nan)),
            "phrase_duration_s": safe_float(cur.get("phrase_duration_s", np.nan)),
            "n_segments_in_phrase": safe_float(cur.get("n_segments_in_phrase", np.nan)),
            "saved_corr": safe_float(cur.get(corr_col, np.nan)),
            "reference_start_global_bin": int(ref["segment_start_global_bin"]),
            "reference_end_global_bin": int(ref["segment_end_global_bin"]),
            "current_start_global_bin": int(cur["segment_start_global_bin"]),
            "current_end_global_bin": int(cur["segment_end_global_bin"]),
            "file_index": cur.get("file_index", np.nan),
            "recording_key": cur.get("recording_key", ""),
        }
        pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs

    # Apply finite-corr filter only if we are not recomputing. If recomputing, allow NaN
    # saved values so the QC can still be made from segment boundaries.
    if not args.recompute_corr:
        pairs = pairs[np.isfinite(pairs["saved_corr"])]
    return pairs.reset_index(drop=True)


def choose_evenly_spaced_indices(n_items: int, n_choose: int) -> np.ndarray:
    if n_items <= 0 or n_choose <= 0:
        return np.array([], dtype=int)
    if n_choose >= n_items:
        return np.arange(n_items, dtype=int)
    return np.unique(np.round(np.linspace(0, n_items - 1, n_choose)).astype(int))


def sample_pairs(pairs: pd.DataFrame, args) -> pd.DataFrame:
    if pairs.empty:
        return pairs

    rng = np.random.default_rng(args.random_seed)

    def sample_one_group(g: pd.DataFrame, n: int) -> pd.DataFrame:
        if g.empty or n <= 0:
            return g.iloc[0:0]
        mode = args.sample_mode
        if mode == "random":
            take = rng.choice(g.index.to_numpy(), size=min(n, len(g)), replace=False)
            return g.loc[take]
        if mode == "lowest_corr":
            return g.sort_values("plot_corr", ascending=True).head(n)
        if mode == "highest_corr":
            return g.sort_values("plot_corr", ascending=False).head(n)
        if mode == "first":
            return g.sort_values(["period", "phrase_id", "current_repeat_index"]).head(n)
        if mode == "quantile_corr":
            sorted_g = g.sort_values("plot_corr", ascending=True)
            idx = choose_evenly_spaced_indices(len(sorted_g), n)
            return sorted_g.iloc[idx]
        raise ValueError("sample_mode must be random, quantile_corr, lowest_corr, highest_corr, or first")

    if args.pairs_per_period is not None:
        pieces = []
        period_order = [p for p in PERIOD_ORDER if p in set(pairs["period"].astype(str))]
        extra_periods = sorted(set(pairs["period"].astype(str)) - set(period_order))
        for period in period_order + extra_periods:
            g = pairs[pairs["period"].astype(str) == period]
            pieces.append(sample_one_group(g, int(args.pairs_per_period)))
        out = pd.concat(pieces, ignore_index=True) if pieces else pairs.iloc[0:0]
    else:
        out = sample_one_group(pairs, int(args.n_pairs_total))

    # Stable, readable order in output PDF.
    if not out.empty:
        out = out.sort_values(["period", "phrase_id", "current_repeat_index", "plot_corr"], na_position="last")
    return out.reset_index(drop=True)


def robust_vmin_vmax(mats: Sequence[np.ndarray], lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    vals = []
    for m in mats:
        a = np.asarray(m, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    vmin, vmax = np.percentile(x, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def spectrogram_for_plot(X_seg: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray]:
    X_crop, freqs = crop_frequency_band(
        X_seg,
        max_freq_khz=args.max_freq_khz,
        freq_min_khz=args.plot_freq_min_khz,
        freq_max_khz=args.plot_freq_max_khz,
    )
    if args.plot_transform == "raw":
        Y = X_crop
    elif args.plot_transform == "linear_power":
        Y = to_linear_power(X_crop, spec_scale=args.spec_scale)
    elif args.plot_transform == "log_power":
        Y = np.log10(to_linear_power(X_crop, spec_scale=args.spec_scale) + 1e-12)
    else:
        raise ValueError("plot_transform must be raw, linear_power, or log_power")
    return Y, freqs


def plot_pair_page(
    X: np.ndarray,
    pair: pd.Series,
    args,
    out_png: Optional[Path] = None,
):
    ref_s0 = int(pair["reference_start_global_bin"])
    ref_s1 = int(pair["reference_end_global_bin"])
    cur_s0 = int(pair["current_start_global_bin"])
    cur_s1 = int(pair["current_end_global_bin"])

    X_ref = X[ref_s0:ref_s1, :]
    X_cur = X[cur_s0:cur_s1, :]
    Y_ref, freqs_ref = spectrogram_for_plot(X_ref, args)
    Y_cur, freqs_cur = spectrogram_for_plot(X_cur, args)
    vmin, vmax = robust_vmin_vmax([Y_ref, Y_cur], lo=args.plot_vmin_percentile, hi=args.plot_vmax_percentile)

    ref_dur_ms = (ref_s1 - ref_s0) * float(args.bin_ms)
    cur_dur_ms = (cur_s1 - cur_s0) * float(args.bin_ms)
    ymin = float(np.nanmin(freqs_ref)) if len(freqs_ref) else args.plot_freq_min_khz
    ymax = float(np.nanmax(freqs_ref)) if len(freqs_ref) else args.plot_freq_max_khz

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, Y, dur_ms, repeat_idx, elapsed_s, label in [
        (axes[0], Y_ref, ref_dur_ms, pair["reference_repeat_index"], pair["reference_elapsed_time_s"], "Reference"),
        (axes[1], Y_cur, cur_dur_ms, pair["current_repeat_index"], pair["current_elapsed_time_s"], "Current"),
    ]:
        ax.imshow(
            Y.T,
            aspect="auto",
            origin="lower",
            extent=[0.0, dur_ms, ymin, ymax],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{label}: repeat {int(repeat_idx)}\nelapsed={safe_float(elapsed_s):.2f}s, dur={dur_ms:.1f} ms")
        ax.set_xlabel("Time within syllable (ms)")
        ax.set_ylabel("Frequency (kHz)")

    saved_corr = safe_float(pair.get("saved_corr", np.nan))
    recomputed_corr = safe_float(pair.get("recomputed_corr", np.nan))
    corr_bits = []
    if np.isfinite(saved_corr):
        corr_bits.append(f"saved r={saved_corr:.3f}")
    if np.isfinite(recomputed_corr):
        corr_bits.append(f"recomputed r={recomputed_corr:.3f}")
    corr_text = "; ".join(corr_bits) if corr_bits else "corr=NaN"

    title_prefix = f"{args.title_prefix}: " if args.title_prefix else ""
    fig.suptitle(
        f"{title_prefix}lag {int(pair['lag'])} pair | "
        f"animal={pair.get('animal_id', '')}, label={pair.get('label', '')}, "
        f"{pair.get('period', '')}, phrase={int(pair['phrase_id'])} | {corr_text}",
        fontsize=11,
    )

    # Add small metadata footer.
    footer = (
        f"recording={pair.get('recording_key', '')} | "
        f"ref bins=[{ref_s0},{ref_s1}) current bins=[{cur_s0},{cur_s1}) | "
        f"phrase duration={safe_float(pair.get('phrase_duration_s', np.nan)):.2f}s"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8)

    if out_png is not None:
        ensure_dir(out_png.parent)
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    return fig


def make_safe_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    if len(text) > max_len:
        text = text[:max_len]
    return text or "pair"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="QC sample pairs of lag-apart syllable spectrograms and show cross-correlation values."
    )
    parser.add_argument("--npz-path", required=True, help="Original NPZ used by the phrase-position analysis.")
    parser.add_argument("--segment-csv", required=True, help="*_phrase_position_segment_features.csv from v10/v11 analysis.")
    parser.add_argument("--out-dir", required=True, help="Output directory for QC figures and selected-pairs CSV.")
    parser.add_argument("--lag", type=int, default=10, help="Number of repeats back to compare. Default: 10.")
    parser.add_argument("--corr-column", default=None, help="Saved correlation column. Default: corr_to_lag<lag>_previous_syllable.")

    parser.add_argument("--spec-key", default="s", help="NPZ spectrogram key. Default: s.")
    parser.add_argument("--bin-ms", type=float, default=2.7, help="Spectrogram bin size in ms. Default: 2.7.")
    parser.add_argument("--max-freq-khz", type=float, default=12.0, help="Maximum frequency represented by spectrogram bins.")
    parser.add_argument("--spec-scale", choices=["linear", "log10", "loge", "shift"], default="linear")

    parser.add_argument("--animal-id", default=None, help="Optional animal filter if CSV contains multiple animals.")
    parser.add_argument("--label", default=None, help="Optional label filter if CSV contains multiple labels.")
    parser.add_argument("--periods", default="all", help="Comma-separated periods to include, or all. Default: all.")
    parser.add_argument("--pre-post", default="all", help="Comma-separated pre_post groups to include, or all. Default: all.")
    parser.add_argument("--min-repeat-index", type=int, default=None)
    parser.add_argument("--max-repeat-index", type=int, default=None)
    parser.add_argument("--min-phrase-duration-s", type=float, default=None)
    parser.add_argument("--min-corr", type=float, default=None, help="Filter by saved corr, if available.")
    parser.add_argument("--max-corr", type=float, default=None, help="Filter by saved corr, if available.")

    parser.add_argument("--n-pairs-total", type=int, default=24, help="Total pairs to plot if --pairs-per-period is not set.")
    parser.add_argument("--pairs-per-period", type=int, default=None, help="Sample this many pairs within each period.")
    parser.add_argument(
        "--sample-mode",
        choices=["quantile_corr", "random", "lowest_corr", "highest_corr", "first"],
        default="quantile_corr",
        help="How to choose pairs. quantile_corr gives low/mid/high examples across the corr range.",
    )
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument("--recompute-corr", action="store_true", help="Recompute correlation from the NPZ snippets and save it next to saved_corr.")
    parser.add_argument("--corr-freq-min-khz", type=float, default=0.8)
    parser.add_argument("--corr-freq-max-khz", type=float, default=10.0)
    parser.add_argument("--corr-time-bins", type=int, default=64)
    parser.add_argument("--corr-transform", choices=["raw", "linear_power", "log_power"], default="log_power")

    parser.add_argument("--plot-freq-min-khz", type=float, default=0.8)
    parser.add_argument("--plot-freq-max-khz", type=float, default=10.0)
    parser.add_argument("--plot-transform", choices=["raw", "linear_power", "log_power"], default="log_power")
    parser.add_argument("--plot-vmin-percentile", type=float, default=2.0)
    parser.add_argument("--plot-vmax-percentile", type=float, default=98.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--skip-individual-pngs", action="store_true")

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    png_dir = out_dir / "individual_pngs"
    if not args.skip_individual_pngs:
        ensure_dir(png_dir)

    corr_col = args.corr_column or f"corr_to_lag{int(args.lag)}_previous_syllable"

    print(f"[INFO] Reading segment CSV: {args.segment_csv}")
    df = pd.read_csv(args.segment_csv)

    # Max bin needed can be inferred before loading NPZ to catch transposed spectrograms.
    max_needed_bin = None
    for col in ["segment_end_global_bin"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if np.any(np.isfinite(vals)):
                max_needed_bin = int(np.nanmax(vals))

    print(f"[INFO] Loading spectrogram from NPZ: {args.npz_path}")
    X = load_spectrogram_from_npz(args.npz_path, args.spec_key, max_needed_bin=max_needed_bin)
    print(f"[INFO] Spectrogram shape: {X.shape} [time x frequency]")

    print(f"[INFO] Preparing lag-{args.lag} pair candidates using column {corr_col!r}")
    pairs = prepare_pair_candidates(df, lag=int(args.lag), corr_col=corr_col, args=args)
    print(f"[INFO] Candidate pairs after filters: {len(pairs)}")
    if pairs.empty:
        print("[WARN] No eligible pairs found. Check lag, label, periods, and segment CSV columns.")
        return 0

    # Recompute correlations when requested. This also creates plot_corr used for sampling.
    recomputed = []
    if args.recompute_corr:
        print("[INFO] Recomputing pair correlations from NPZ snippets...")
        for _, pair in pairs.iterrows():
            try:
                X_ref = X[int(pair["reference_start_global_bin"]): int(pair["reference_end_global_bin"]), :]
                X_cur = X[int(pair["current_start_global_bin"]): int(pair["current_end_global_bin"]), :]
                recomputed.append(corr_between_segments(X_ref, X_cur, args))
            except Exception:
                recomputed.append(np.nan)
        pairs["recomputed_corr"] = recomputed
    else:
        pairs["recomputed_corr"] = np.nan

    # Plot corr is recomputed if available, otherwise saved. If both are NaN, those rows can still
    # be shown only under first/random; for quantile/highest/lowest they are removed.
    pairs["plot_corr"] = pairs["recomputed_corr"]
    pairs.loc[~np.isfinite(pairs["plot_corr"]), "plot_corr"] = pairs.loc[~np.isfinite(pairs["plot_corr"]), "saved_corr"]
    if args.sample_mode in {"quantile_corr", "lowest_corr", "highest_corr"}:
        before = len(pairs)
        pairs = pairs[np.isfinite(pairs["plot_corr"])].reset_index(drop=True)
        if len(pairs) < before:
            print(f"[INFO] Removed {before - len(pairs)} pairs without finite saved/recomputed corr for {args.sample_mode} sampling.")
    if pairs.empty:
        print("[WARN] No pairs have finite correlation values after recomputation/filtering.")
        return 0

    selected = sample_pairs(pairs, args)
    print(f"[INFO] Selected pairs for QC figures: {len(selected)}")

    selected_csv = out_dir / f"selected_lag{int(args.lag)}_pairs.csv"
    selected.to_csv(selected_csv, index=False)
    print(f"[SAVED] {selected_csv}")

    pdf_path = out_dir / f"lag{int(args.lag)}_pair_qc.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, pair in selected.iterrows():
            animal = make_safe_filename(pair.get("animal_id", "animal"))
            label = make_safe_filename(pair.get("label", "label"))
            period = make_safe_filename(pair.get("period", "period"))
            phrase = int(pair["phrase_id"])
            cur_rep = int(pair["current_repeat_index"])
            corr_val = safe_float(pair.get("plot_corr", np.nan))
            corr_tag = "nan" if not np.isfinite(corr_val) else f"{corr_val:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
            stem = f"{i+1:03d}_{animal}_label{label}_{period}_phrase{phrase}_rep{cur_rep}_lag{int(args.lag)}_r{corr_tag}"
            out_png = None if args.skip_individual_pngs else png_dir / f"{stem}.png"
            try:
                fig = plot_pair_page(X, pair, args, out_png=out_png)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                print(f"[WARN] Failed to plot pair {i}: {type(exc).__name__}: {exc}")
    print(f"[SAVED] {pdf_path}")

    if not args.skip_individual_pngs:
        print(f"[SAVED] individual PNGs in {png_dir}")

    # Helpful quick summary.
    summary = selected.copy()
    for col in ["saved_corr", "recomputed_corr", "plot_corr"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    print("[INFO] Selected correlation summary:")
    print(summary[["period", "phrase_id", "reference_repeat_index", "current_repeat_index", "saved_corr", "recomputed_corr", "plot_corr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
