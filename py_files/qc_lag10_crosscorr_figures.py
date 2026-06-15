#!/usr/bin/env python3
"""
QC plots for lag-N intra-phrase spectrogram cross-correlation analyses.

Designed for outputs from phrase_position_acoustic_crosscorr_v10_lag10_previous.py,
for example:
    <root>/<ANIMAL>/<ANIMAL>_all_labels_phrase_position_segment_features.csv

Main QC questions:
1) Is the aggregate pre-lesion drop in lag-10 correlation present within birds?
2) Which bird x syllable labels drive the apparent drop?
3) Are late pre bins supported by enough segments / phrases / labels?
4) Can we inspect representative current-vs-lag10 syllable pairs?

The script generates:
    qc_lag10_per_bird_curves_*.png
    qc_lag10_bin_contributions_*.png
    qc_lag10_bird_label_slopes_*.png
    qc_lag10_selected_pair_examples_index.csv
    qc_lag10_pair_examples_*.png

The pair-example panels always show pair metadata and correlation values. If the CSVs
contain detectable segment boundary columns and --npz-root is provided, the script
will also try to draw actual spectrograms for the reference/current syllables.
If those columns are not available, the figure will clearly say that spectrograms
could not be reconstructed from the existing CSV columns.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

FEATURE_LABELS = {
    "corr_to_lag10_previous_syllable": "Spectrogram correlation to syllable 10 repeats earlier",
    "distance_from_lag10_previous_syllable": "Spectrogram distance from syllable 10 repeats earlier",
    "corr_to_phrase_early_template": "Spectrogram correlation to early-phrase template",
    "distance_from_phrase_early_template": "Spectrogram distance from early-phrase template",
    "mean_wiener_entropy": "Mean Wiener entropy",
    "q95_wiener_entropy": "95th percentile Wiener entropy",
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative (kHz/s)",
}

PREDICTOR_LABELS = {
    "n_previous_segments": "Number of previous analyzed syllables in phrase",
    "elapsed_time_in_phrase_s": "Elapsed time in repeated phrase (s)",
}
PREDICTOR_SHORT = {
    "n_previous_segments": "previous_repeats",
    "elapsed_time_in_phrase_s": "elapsed_time",
}

PHRASE_COL_CANDIDATES = [
    "phrase_id", "phrase_uid", "phrase_key", "phrase_run_id", "phrase_index",
    "phrase_label_run_id", "run_id", "bout_id", "bout_index",
]
REPEAT_COL_CANDIDATES = [
    "repeat_index_in_phrase", "segment_index_in_phrase", "segment_order_in_phrase",
    "syllable_index_in_phrase", "seg_order", "segment_order",
]
START_COL_CANDIDATES = [
    "segment_start_bin", "segment_onset_bin", "onset_bin", "start_bin",
    "segment_start_idx", "start_idx", "start_frame", "onset_frame",
]
END_COL_CANDIDATES = [
    "segment_end_bin", "segment_offset_bin", "offset_bin", "end_bin",
    "segment_end_idx", "end_idx", "end_frame", "offset_frame",
]
FILE_INDEX_COL_CANDIDATES = [
    "file_index", "file_idx", "file_indices", "file_indices_stitched", "source_file_index",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
        key = re.sub(r"[^a-z0-9]+", "_", str(cand).lower()).strip("_")
        if key in norm:
            return norm[key]
    return None


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_median(x) -> float:
    arr = finite_array(x)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def read_segment_tables(root: Path, animals: Optional[set[str]] = None) -> pd.DataFrame:
    # Prefer combined all-label files to avoid double counting label-level CSVs.
    files = sorted(root.rglob("*_all_labels_phrase_position_segment_features.csv"))
    if not files:
        files = sorted(root.rglob("*_phrase_position_segment_features.csv"))
    if not files:
        raise FileNotFoundError(f"No phrase-position segment feature CSVs found under {root}")

    frames = []
    used = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        if df.empty:
            continue
        if "animal_id" not in df.columns:
            m = re.match(r"([^_/]+)_", f.name)
            df["animal_id"] = m.group(1) if m else f.parent.name
        if "label" not in df.columns:
            m = re.search(r"_label([^_]+)_", f.name)
            if m:
                df["label"] = m.group(1)
        if "label" not in df.columns:
            print(f"[WARN] Skipping {f}; no label column")
            continue
        df["animal_id"] = df["animal_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        if animals is not None:
            df = df[df["animal_id"].isin(animals)].copy()
            if df.empty:
                continue
        if "pre_post" not in df.columns and "period" in df.columns:
            df["pre_post"] = np.where(df["period"].astype(str).str.contains("post", case=False, na=False), "post", "pre")
        if "pre_post" in df.columns:
            df["pre_post"] = np.where(df["pre_post"].astype(str).str.contains("post", case=False, na=False), "post", "pre")
        df["source_csv"] = str(f)
        frames.append(df)
        used.append({"csv_path": str(f), "n_rows": int(df.shape[0]), "animal_id": ",".join(sorted(df["animal_id"].unique()))})
    if not frames:
        raise RuntimeError("No usable segment rows loaded.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[INFO] Loaded {len(out):,} rows from {len(frames):,} CSVs")
    return out


def add_repeat_and_phrase_columns(df: pd.DataFrame, lag: int) -> Tuple[pd.DataFrame, Optional[str], str]:
    out = df.copy()
    repeat_col = find_first_existing(out.columns, REPEAT_COL_CANDIDATES)
    if repeat_col is None:
        if "n_previous_segments" in out.columns:
            out["_qc_repeat_index"] = pd.to_numeric(out["n_previous_segments"], errors="coerce") + 1
            repeat_col = "_qc_repeat_index"
        else:
            out["_qc_repeat_index"] = np.nan
            repeat_col = "_qc_repeat_index"
    out["_qc_repeat_index"] = pd.to_numeric(out[repeat_col], errors="coerce")

    phrase_col = find_first_existing(out.columns, PHRASE_COL_CANDIDATES)
    if phrase_col is None:
        # Fallback: if no explicit phrase ID is available, use a conservative combination of columns.
        # This is enough for aggregate QC, but not reliable for identifying exact spectrogram pairs.
        possible = ["animal_id", "label", "period", "source_csv"]
        existing = [c for c in possible if c in out.columns]
        out["_qc_phrase_fallback"] = out[existing].astype(str).agg("|".join, axis=1)
        phrase_col = "_qc_phrase_fallback"
        print("[WARN] No explicit phrase ID column found; exact lag-pair matching may be limited.")
    out["_qc_phrase_id"] = out[phrase_col].astype(str)

    ref_col = f"lag{lag}_reference_repeat_index_in_phrase"
    if ref_col not in out.columns:
        out[ref_col] = out["_qc_repeat_index"] - lag
    out["_qc_lag_reference_repeat_index"] = pd.to_numeric(out[ref_col], errors="coerce")
    return out, phrase_col, repeat_col


def binned_summary(df: pd.DataFrame, feature: str, predictor: str, bins: np.ndarray, min_bin_n: int) -> pd.DataFrame:
    rows = []
    use = df.copy()
    use[predictor] = pd.to_numeric(use[predictor], errors="coerce")
    use[feature] = pd.to_numeric(use[feature], errors="coerce")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (use[predictor] >= lo) & (use[predictor] < hi) & np.isfinite(use[feature])
        sub = use[mask]
        vals = finite_array(sub[feature])
        if vals.size < min_bin_n:
            continue
        rows.append({
            "bin": i, "x_mid": float((lo + hi) / 2), "x_lo": float(lo), "x_hi": float(hi),
            "n_segments": int(vals.size),
            "n_phrases": int(sub["_qc_phrase_id"].nunique()) if "_qc_phrase_id" in sub.columns else np.nan,
            "n_bird_labels": int(sub[["animal_id", "label"]].drop_duplicates().shape[0]),
            "median": float(np.nanmedian(vals)),
            "q25": float(np.nanpercentile(vals, 25)),
            "q75": float(np.nanpercentile(vals, 75)),
        })
    return pd.DataFrame(rows)


def make_bins(df: pd.DataFrame, predictor: str, n_bins: int, max_x: Optional[float]) -> np.ndarray:
    x = pd.to_numeric(df[predictor], errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        raise ValueError(f"No finite values for predictor {predictor}")
    xmin = float(np.nanmin(x))
    xmax = float(max_x) if max_x is not None else float(np.nanmax(x))
    if predictor == "n_previous_segments":
        xmin = max(0.0, math.floor(xmin))
    if xmax <= xmin:
        xmax = xmin + 1.0
    return np.linspace(xmin, xmax, n_bins + 1)


def plot_per_bird_curves(df: pd.DataFrame, feature: str, predictor: str, out_png: Path, n_bins: int, min_bin_n: int, max_x: Optional[float], title_prefix: str):
    use = df[np.isfinite(pd.to_numeric(df[feature], errors="coerce")) & np.isfinite(pd.to_numeric(df[predictor], errors="coerce"))].copy()
    if use.empty:
        return
    bins = make_bins(use, predictor, n_bins, max_x)
    birds = sorted(use["animal_id"].unique())
    n = len(birds)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False, sharey=True)
    for ax, bird in zip(axes.ravel(), birds):
        sb = use[use["animal_id"] == bird]
        for pp in ["pre", "post"]:
            b = binned_summary(sb[sb["pre_post"] == pp], feature, predictor, bins, min_bin_n)
            if b.empty:
                continue
            ax.plot(b["x_mid"], b["median"], marker="o", linewidth=2, markersize=3.5, label=f"{pp} median")
            ax.fill_between(b["x_mid"], b["q25"], b["q75"], alpha=0.18, label=f"{pp} IQR")
        ax.set_title(bird)
        ax.grid(alpha=0.25)
        ax.set_xlabel(PREDICTOR_LABELS.get(predictor, predictor))
    for ax in axes[:, 0]:
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
    for ax in axes.ravel()[len(birds):]:
        ax.set_axis_off()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.suptitle(f"{title_prefix}\nPer-bird lag correlation QC\n{FEATURE_LABELS.get(feature, feature)}", fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.22, hspace=0.34)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_bin_contributions(df: pd.DataFrame, feature: str, predictor: str, out_png: Path, n_bins: int, min_bin_n: int, max_x: Optional[float], title_prefix: str):
    use = df[np.isfinite(pd.to_numeric(df[feature], errors="coerce")) & np.isfinite(pd.to_numeric(df[predictor], errors="coerce"))].copy()
    if use.empty:
        return
    bins = make_bins(use, predictor, n_bins, max_x)
    summaries = []
    for pp in ["pre", "post"]:
        b = binned_summary(use[use["pre_post"] == pp], feature, predictor, bins, min_bin_n=1)
        if not b.empty:
            b["pre_post"] = pp
            summaries.append(b)
    if not summaries:
        return
    out = pd.concat(summaries, ignore_index=True)
    out_csv = out_png.with_suffix(".csv")
    out.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.8), sharex=True)
    metrics = [("n_segments", "Number of lag-pair segments"), ("n_phrases", "Number of phrases"), ("n_bird_labels", "Number of bird×labels")]
    for ax, (metric, label) in zip(axes, metrics):
        for pp in ["pre", "post"]:
            sub = out[out["pre_post"] == pp]
            if sub.empty:
                continue
            ax.plot(sub["x_mid"], sub[metric], marker="o", linewidth=2, markersize=3.5, label=pp)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
    axes[-1].set_xlabel(PREDICTOR_LABELS.get(predictor, predictor))
    fig.suptitle(f"{title_prefix}\nQC: bin contributions for {FEATURE_LABELS.get(feature, feature)}", fontsize=15)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.09, hspace=0.16)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def linear_slope(x, y, min_n: int) -> Dict[str, float | str]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    n = int(valid.sum())
    row = {"n": n, "slope": np.nan, "intercept": np.nan, "r_value": np.nan, "p_value": np.nan, "status": "ok"}
    if n < min_n:
        row["status"] = "too_few_points"
        return row
    xv = xv[valid]
    yv = yv[valid]
    if np.unique(xv).size < 2:
        row["status"] = "constant_x"
        return row
    if SCIPY_AVAILABLE:
        try:
            lr = stats.linregress(xv, yv)
            row.update({"slope": float(lr.slope), "intercept": float(lr.intercept), "r_value": float(lr.rvalue), "p_value": float(lr.pvalue)})
        except Exception as e:
            row["status"] = f"fit_error:{e}"
    else:
        try:
            slope, intercept = np.polyfit(xv, yv, deg=1)
            row.update({"slope": float(slope), "intercept": float(intercept)})
        except Exception as e:
            row["status"] = f"fit_error:{e}"
    return row


def compute_slopes(df: pd.DataFrame, feature: str, predictor: str, min_n: int) -> pd.DataFrame:
    rows = []
    use = df.copy()
    use[feature] = pd.to_numeric(use[feature], errors="coerce")
    use[predictor] = pd.to_numeric(use[predictor], errors="coerce")
    for (animal, label, pp), sub in use.groupby(["animal_id", "label", "pre_post"], dropna=False):
        res = linear_slope(sub[predictor], sub[feature], min_n=min_n)
        rows.append({
            "animal_id": animal, "label": label, "pre_post": pp,
            "feature": feature, "predictor": predictor,
            **res,
        })
    return pd.DataFrame(rows)


def plot_bird_label_slopes(df: pd.DataFrame, feature: str, predictor: str, out_png: Path, min_n: int, title_prefix: str):
    slopes = compute_slopes(df, feature, predictor, min_n=min_n)
    out_csv = out_png.with_suffix(".csv")
    slopes.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    ok = slopes[np.isfinite(slopes["slope"])].copy()
    if ok.empty:
        return
    birds = sorted(ok["animal_id"].unique())
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    offsets = {"pre": -0.16, "post": 0.16}
    for _, row in ok.iterrows():
        pp = row["pre_post"]
        if pp not in offsets:
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), pp, feature, predictor)) % 1000) / 1000.0 - 0.5) * 0.12
        ax.scatter(xmap[row["animal_id"]] + offsets[pp] + jitter, row["slope"], s=35, alpha=0.35, label=pp if pp not in ax.get_legend_handles_labels()[1] else None)
    med = ok.groupby(["animal_id", "pre_post"])["slope"].median().reset_index()
    for _, row in med.iterrows():
        pp = row["pre_post"]
        if pp not in offsets:
            continue
        ax.scatter(xmap[row["animal_id"]] + offsets[pp], row["slope"], s=110, marker="D", edgecolors="black")
    ax.axhline(0, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(np.arange(len(birds)))
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ax.set_ylabel(f"Bird×label linear slope\n{FEATURE_LABELS.get(feature, feature)} vs {PREDICTOR_LABELS.get(predictor, predictor)}")
    ax.set_title(f"{title_prefix}\nQC: bird×label slopes for lag correlation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.16)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def parse_first_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value)
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def find_npz(npz_root: Optional[Path], animal: str) -> Optional[Path]:
    if npz_root is None:
        return None
    candidates = [npz_root / animal / f"{animal}.npz", npz_root / f"{animal}.npz"]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(npz_root.rglob(f"{animal}.npz"))
    return hits[0] if hits else None


def get_spectrogram_array(npz_path: Path) -> Optional[np.ndarray]:
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    for key in ["s", "spectrogram", "spectrograms", "S"]:
        if key in z.files:
            arr = z[key]
            if isinstance(arr, np.ndarray):
                return arr
    return None


def extract_spectrogram(row: pd.Series, spec: Optional[np.ndarray], start_col: Optional[str], end_col: Optional[str], file_col: Optional[str]) -> Optional[np.ndarray]:
    if spec is None or start_col is None or end_col is None:
        return None
    try:
        start = int(float(row[start_col]))
        end = int(float(row[end_col]))
    except Exception:
        return None
    if end <= start:
        return None
    # 3D spectrograms: try file index as first dimension.
    if spec.ndim == 3:
        fi = parse_first_int(row[file_col]) if file_col else None
        if fi is None or fi < 0 or fi >= spec.shape[0]:
            return None
        arr = spec[fi]
    elif spec.ndim == 2:
        arr = spec
    else:
        return None
    try:
        # Common shapes: freq x time OR time x freq. Use whichever axis can accommodate end.
        if end <= arr.shape[1]:
            seg = arr[:, start:end]
        elif end <= arr.shape[0]:
            seg = arr[start:end, :].T
        else:
            return None
    except Exception:
        return None
    if seg.size == 0:
        return None
    # Downsample if huge for plotting speed.
    if seg.shape[1] > 400:
        idx = np.linspace(0, seg.shape[1] - 1, 400).astype(int)
        seg = seg[:, idx]
    return np.asarray(seg, dtype=float)


def plot_pair_examples_for_label(
    df: pd.DataFrame,
    animal: str,
    label: str,
    feature: str,
    out_png: Path,
    npz_root: Optional[Path],
    lag: int,
    n_each: int,
    title_prefix: str,
) -> pd.DataFrame:
    sub = df[(df["animal_id"].astype(str) == str(animal)) & (df["label"].astype(str) == str(label))].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub[np.isfinite(sub[feature])].copy()
    if sub.empty:
        return pd.DataFrame()

    selected_parts = []
    for pp in ["pre", "post"]:
        spp = sub[sub["pre_post"] == pp].copy()
        if spp.empty:
            continue
        spp = spp.sort_values(feature)
        low = spp.head(n_each)
        high = spp.tail(n_each)
        mid_idx = np.argsort(np.abs(spp[feature] - np.nanmedian(spp[feature]))).to_numpy()[:n_each]
        mid = spp.iloc[mid_idx]
        for group_name, part in [("low_corr", low), ("median_corr", mid), ("high_corr", high)]:
            p = part.copy()
            p["qc_rank_group"] = group_name
            selected_parts.append(p)
    if not selected_parts:
        return pd.DataFrame()
    sel = pd.concat(selected_parts, ignore_index=False).drop_duplicates()
    # Keep a compact number of rows; prioritizes pre then post, low/mid/high.
    sel = sel.head(max(1, n_each * 6)).copy()

    # Build lookup for exact reference row.
    group_cols = ["animal_id", "label", "pre_post", "_qc_phrase_id"]
    ref_lookup = {}
    for idx, row in sub.iterrows():
        key = tuple(row[c] for c in group_cols) + (int(row["_qc_repeat_index"]) if np.isfinite(row["_qc_repeat_index"]) else None,)
        ref_lookup[key] = idx

    npz_path = find_npz(npz_root, animal)
    spec = get_spectrogram_array(npz_path) if npz_path is not None else None
    start_col = find_first_existing(sub.columns, START_COL_CANDIDATES)
    end_col = find_first_existing(sub.columns, END_COL_CANDIDATES)
    file_col = find_first_existing(sub.columns, FILE_INDEX_COL_CANDIDATES)
    can_draw_specs = spec is not None and start_col is not None and end_col is not None

    nrows = len(sel)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.5, max(2.0 * nrows, 5.5)), squeeze=False)
    out_rows = []
    for r, (idx, row) in enumerate(sel.iterrows()):
        ref_repeat = row.get("_qc_lag_reference_repeat_index", np.nan)
        ref_idx = None
        if np.isfinite(ref_repeat):
            key = tuple(row[c] for c in group_cols) + (int(ref_repeat),)
            ref_idx = ref_lookup.get(key)
        ref_row = sub.loc[ref_idx] if ref_idx is not None else None

        current_seg = extract_spectrogram(row, spec, start_col, end_col, file_col) if can_draw_specs else None
        ref_seg = extract_spectrogram(ref_row, spec, start_col, end_col, file_col) if (can_draw_specs and ref_row is not None) else None

        for ax in axes[r, :]:
            ax.set_xticks([])
            ax.set_yticks([])
        if ref_seg is not None:
            axes[r, 0].imshow(ref_seg, aspect="auto", origin="lower", interpolation="nearest")
        else:
            axes[r, 0].text(0.5, 0.5, "Reference\nspectrogram\nnot reconstructed", ha="center", va="center", fontsize=9)
        if current_seg is not None:
            axes[r, 1].imshow(current_seg, aspect="auto", origin="lower", interpolation="nearest")
        else:
            axes[r, 1].text(0.5, 0.5, "Current\nspectrogram\nnot reconstructed", ha="center", va="center", fontsize=9)
        axes[r, 0].set_ylabel(f"{row['pre_post']}\n{row.get('qc_rank_group', '')}", rotation=0, labelpad=35, va="center")
        axes[r, 0].set_title("10 repeats earlier" if r == 0 else "")
        axes[r, 1].set_title("current syllable" if r == 0 else "")
        txt = (
            f"animal={animal}\nlabel={label}\nperiod={row.get('period', row.get('pre_post', ''))}\n"
            f"current repeat={row.get('_qc_repeat_index', np.nan):.0f}\n"
            f"reference repeat={ref_repeat:.0f}\n"
            f"{feature}={row.get(feature, np.nan):.3f}\n"
            f"phrase={str(row.get('_qc_phrase_id', 'NA'))[:35]}"
        )
        if not can_draw_specs:
            txt += "\n\nSpectrogram note:\nNo detectable NPZ/boundary columns.\nPair metadata is still valid."
        axes[r, 2].text(0.02, 0.98, txt, ha="left", va="top", fontsize=9)
        axes[r, 2].set_axis_off()
        out_rows.append({
            "animal_id": animal, "label": label, "pre_post": row.get("pre_post"), "period": row.get("period"),
            "qc_rank_group": row.get("qc_rank_group"), "current_row_index": int(idx),
            "reference_row_index": int(ref_idx) if ref_idx is not None else np.nan,
            "current_repeat_index": row.get("_qc_repeat_index"), "reference_repeat_index": ref_repeat,
            feature: row.get(feature), "phrase_id": row.get("_qc_phrase_id"),
            "spectrogram_reconstructed": bool(current_seg is not None and ref_seg is not None),
            "npz_path": str(npz_path) if npz_path else "",
            "start_col": start_col or "", "end_col": end_col or "", "file_col": file_col or "",
        })
    fig.suptitle(f"{title_prefix}\nLag-{lag} pair examples: {animal} label {label}\n{FEATURE_LABELS.get(feature, feature)}", fontsize=14)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.04, wspace=0.12, hspace=0.22)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    return pd.DataFrame(out_rows)


def choose_sample_labels(slopes: pd.DataFrame, n: int, prefer_period: str = "pre") -> pd.DataFrame:
    ok = slopes[(slopes["pre_post"] == prefer_period) & np.isfinite(slopes["slope"])].copy()
    if ok.empty:
        ok = slopes[np.isfinite(slopes["slope"])].copy()
    # For correlation features, strongest suspicious drop = most negative slope.
    ok = ok.sort_values("slope", ascending=True)
    return ok[["animal_id", "label", "slope", "n", "pre_post"]].drop_duplicates(["animal_id", "label"]).head(n)


def main():
    p = argparse.ArgumentParser(description="Create QC plots for lag-10 intra-phrase cross-correlation outputs.")
    p.add_argument("--phrase-position-root", required=True, help="Root containing *_all_labels_phrase_position_segment_features.csv files")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: <root>/lag10_qc_figures")
    p.add_argument("--npz-root", default=None, help="Optional NPZ root for reconstructing spectrogram pair panels")
    p.add_argument("--animals", default=None, help="Optional comma-separated animals to include")
    p.add_argument("--feature", default="corr_to_lag10_previous_syllable", help="Lag feature to QC")
    p.add_argument("--lag", type=int, default=10)
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--min-bin-n", type=int, default=10)
    p.add_argument("--min-slope-n", type=int, default=10)
    p.add_argument("--max-plot-previous-repeats", type=float, default=None)
    p.add_argument("--max-plot-elapsed-s", type=float, default=None)
    p.add_argument("--n-sample-labels", type=int, default=4, help="Number of bird-labels to use for pair example panels")
    p.add_argument("--n-pairs-each", type=int, default=2, help="For pair examples, number each from low/median/high correlation per condition")
    p.add_argument("--sample-animal-labels", default=None, help="Optional manual list like USA5288:3,USA5443:17")
    p.add_argument("--title-prefix", default="Medial + Lateral birds: top 30% variable repeat-time syllables")
    args = p.parse_args()

    root = Path(args.phrase_position_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "lag10_qc_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_root = Path(args.npz_root).expanduser().resolve() if args.npz_root else None
    animals = {x.strip() for x in args.animals.split(",") if x.strip()} if args.animals else None

    df = read_segment_tables(root, animals=animals)
    if args.feature not in df.columns:
        raise ValueError(f"Feature {args.feature!r} not found. Available lag columns: {[c for c in df.columns if 'lag' in c.lower()]}")
    if "pre_post" not in df.columns:
        raise ValueError("Could not find or infer pre_post column.")
    df, phrase_col, repeat_col = add_repeat_and_phrase_columns(df, lag=args.lag)

    # Keep only rows with usable lag feature values.
    df[args.feature] = pd.to_numeric(df[args.feature], errors="coerce")
    usable = df[np.isfinite(df[args.feature])].copy()
    print(f"[INFO] Usable {args.feature}: {len(usable):,} / {len(df):,} rows")
    print(f"[INFO] Phrase column used: {phrase_col}; repeat column used: {repeat_col}")

    # Save a compact summary table.
    summary = usable.groupby(["animal_id", "label", "pre_post"]).agg(
        n_rows=(args.feature, "size"),
        median_feature=(args.feature, "median"),
        max_repeat=("_qc_repeat_index", "max"),
        n_phrases=("_qc_phrase_id", "nunique"),
    ).reset_index()
    summary.to_csv(out_dir / "qc_lag10_usable_rows_by_bird_label.csv", index=False)
    print(f"[SAVED] {out_dir / 'qc_lag10_usable_rows_by_bird_label.csv'}")

    predictors = ["n_previous_segments", "elapsed_time_in_phrase_s"]
    all_slopes = []
    for predictor in predictors:
        if predictor not in usable.columns:
            continue
        max_x = args.max_plot_previous_repeats if predictor == "n_previous_segments" else args.max_plot_elapsed_s
        pred_tag = PREDICTOR_SHORT.get(predictor, safe_name(predictor))
        plot_per_bird_curves(
            usable, args.feature, predictor,
            out_dir / f"qc_lag10_per_bird_curves_{safe_name(args.feature)}_vs_{pred_tag}.png",
            n_bins=args.n_bins, min_bin_n=args.min_bin_n, max_x=max_x, title_prefix=args.title_prefix,
        )
        plot_bin_contributions(
            usable, args.feature, predictor,
            out_dir / f"qc_lag10_bin_contributions_{safe_name(args.feature)}_vs_{pred_tag}.png",
            n_bins=args.n_bins, min_bin_n=args.min_bin_n, max_x=max_x, title_prefix=args.title_prefix,
        )
        plot_bird_label_slopes(
            usable, args.feature, predictor,
            out_dir / f"qc_lag10_bird_label_slopes_{safe_name(args.feature)}_vs_{pred_tag}.png",
            min_n=args.min_slope_n, title_prefix=args.title_prefix,
        )
        slopes = compute_slopes(usable, args.feature, predictor, min_n=args.min_slope_n)
        all_slopes.append(slopes)

    slopes_all = pd.concat(all_slopes, ignore_index=True) if all_slopes else pd.DataFrame()
    slopes_all.to_csv(out_dir / "qc_lag10_all_bird_label_slopes.csv", index=False)
    print(f"[SAVED] {out_dir / 'qc_lag10_all_bird_label_slopes.csv'}")

    # Select sample labels for pair-example panels. Prefer slope vs previous repeats.
    sample_pairs = []
    if args.sample_animal_labels:
        for item in args.sample_animal_labels.split(","):
            if ":" not in item:
                continue
            animal, label = item.split(":", 1)
            sample_pairs.append({"animal_id": animal.strip(), "label": label.strip()})
    else:
        if not slopes_all.empty:
            slope_prev = slopes_all[slopes_all["predictor"] == "n_previous_segments"].copy()
            chosen = choose_sample_labels(slope_prev, n=args.n_sample_labels, prefer_period="pre")
            sample_pairs = chosen[["animal_id", "label"]].to_dict("records")
        else:
            sample_pairs = usable[["animal_id", "label"]].drop_duplicates().head(args.n_sample_labels).to_dict("records")

    pair_rows = []
    for item in sample_pairs:
        animal = str(item["animal_id"])
        label = str(item["label"])
        out_png = out_dir / f"qc_lag10_pair_examples_{safe_name(args.feature)}_{safe_name(animal)}_label{safe_name(label)}.png"
        rows = plot_pair_examples_for_label(
            usable, animal, label, args.feature, out_png, npz_root=npz_root, lag=args.lag,
            n_each=args.n_pairs_each, title_prefix=args.title_prefix,
        )
        if not rows.empty:
            pair_rows.append(rows)
    if pair_rows:
        pair_index = pd.concat(pair_rows, ignore_index=True)
        pair_index.to_csv(out_dir / "qc_lag10_selected_pair_examples_index.csv", index=False)
        print(f"[SAVED] {out_dir / 'qc_lag10_selected_pair_examples_index.csv'}")

    # Diagnostic note for spectrogram reconstruction.
    note = (
        "Lag-10 QC outputs generated.\n\n"
        f"Feature: {args.feature}\n"
        f"Rows loaded: {len(df):,}\n"
        f"Rows with finite feature: {len(usable):,}\n"
        f"Phrase column used: {phrase_col}\n"
        f"Repeat column used: {repeat_col}\n\n"
        "If pair-example panels say spectrograms were not reconstructed, the segment CSVs did not contain\n"
        "detectable start/end bin columns for extracting syllable spectrograms from the NPZ. The pair\n"
        "metadata and correlation ranks are still useful for identifying examples to inspect manually.\n"
    )
    (out_dir / "README_lag10_qc.txt").write_text(note)
    print(f"[SAVED] {out_dir / 'README_lag10_qc.txt'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
