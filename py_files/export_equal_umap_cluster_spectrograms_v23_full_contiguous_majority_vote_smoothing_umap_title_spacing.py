#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_equal_umap_cluster_spectrograms_v3_top30_flat.py

Companion utility for UMAP/HDBSCAN time-bin analyses.

For each selected HDBSCAN/UMAP cluster in one bird, this script:
  1) builds equal-sized time-bin datasets across periods/groups,
  2) saves a UMAP scatterplot using exactly those selected time bins, and
  3) saves matching spectrogram PNGs.

Important v8 change:
  The default UMAP/Bhattacharyya analysis is now run-balanced and
  phrase-normalized. Instead of comparing sparse random time bins, this mode:
    1) finds full contiguous HDBSCAN phrase-label runs,
    2) selects the same number of full runs from each group, and
    3) samples the same normalized phase positions from every selected run.

  This makes every phrase-label bout contribute equally and prevents long
  post-lesion stutter bouts from dominating the BC simply because they contain
  many more frames. A frame-weighted/equal-time-bin mode is still available with
  --bc-analysis-mode frame_weighted.

Typical NPZ keys expected:
  - embedding_outputs: (N, 2) UMAP coordinates
  - hdbscan_labels:    (N,) cluster IDs
  - s:                 spectrogram/time-bin matrix
  - file_indices:      (N,) file identity per time bin, optional but recommended
  - file_map:          mapping used by your existing pre/post spectrogram script
  - vocalization:      (N,) optional mask for vocalization bins

This script imports your existing pre/post spectrogram helper script for:
  - treatment-date lookup
  - time-bin date parsing from file_map
  - pre/post masks
  - spectrogram orientation

Output layout:
  /Volumes/my_own_SSD/updated_AreaX_outputs/equal_umap_cluster_spectrograms_exports/
      USA5443/
          USA5443_label02_UMAP_equal_timebins.png
          USA5443_label02_selected_equal_timebins.csv
          USA5443_label02_early_pre_spectrogram_N6000_rowbins2000.png
          USA5443_label02_late_pre_spectrogram_N6000_rowbins2000.png
          ...

If --phrase-csv, --top30-csv, or --input-csv is provided, only the top
phrase-duration-variance labels are exported. This keeps the animal folder
focused on the top variance clusters only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import argparse
import importlib.util
import math
import re
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "equal_umap_cluster_spectrograms_v23_full_contiguous_majority_vote_smoothing_umap_title_spacing"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_token(x: Any) -> str:
    s = str(x).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _cluster_token(cluster_id: int) -> str:
    cluster_id = int(cluster_id)
    if cluster_id < 0:
        return "label_noise"
    return f"label{cluster_id:02d}"


def _parse_list_arg(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def _fill_noise_labels_decoder_style(labels: np.ndarray, noise_label: int = -1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Replace noise labels using the decoder data-prep convention.

    This matches the behavior of the decoder prep snippet described by the user:
    each noise label is replaced in-place by the closest non-noise label found
    scanning left/right, with left taking priority when available. Because the
    operation is performed left-to-right and mutates the label vector, internal
    noise gaps are effectively forward-filled from the previous non-noise label,
    while leading noise is assigned to the first later non-noise label.

    If the entire array is noise, labels are returned unchanged.
    """
    original = np.asarray(labels).astype(int)
    filled = original.copy()
    noise_label = int(noise_label)
    noise_mask = filled == noise_label
    n_noise_before = int(noise_mask.sum())

    info: Dict[str, Any] = {
        "noise_label": noise_label,
        "n_noise_labels_before_fill": n_noise_before,
        "n_noise_labels_after_fill": n_noise_before,
        "n_noise_labels_replaced": 0,
        "decoder_style_noise_fill_applied": False,
        "decoder_style_noise_fill_all_noise_warning": False,
    }
    if n_noise_before == 0:
        return filled, info

    nonnoise_idx = np.flatnonzero(filled != noise_label)
    if nonnoise_idx.size == 0:
        info["decoder_style_noise_fill_all_noise_warning"] = True
        return filled, info

    # Leading noise becomes the first available non-noise label.
    first_nonnoise = int(nonnoise_idx[0])
    if first_nonnoise > 0:
        filled[:first_nonnoise] = int(filled[first_nonnoise])

    # Because this is left-to-right in-place, every later noise bin adopts the
    # immediately preceding filled label. This is equivalent to the decoder loop
    # after the first non-noise label has been encountered.
    for i in np.flatnonzero(filled == noise_label):
        if i > 0 and filled[i - 1] != noise_label:
            filled[i] = int(filled[i - 1])
        else:
            # Defensive fallback for unusual cases; mirrors rightward search.
            r = i + 1
            while r < filled.shape[0] and filled[r] == noise_label:
                r += 1
            if r < filled.shape[0]:
                filled[i] = int(filled[r])

    n_noise_after = int((filled == noise_label).sum())
    info.update({
        "n_noise_labels_after_fill": n_noise_after,
        "n_noise_labels_replaced": int(n_noise_before - n_noise_after),
        "decoder_style_noise_fill_applied": True,
    })
    return filled, info




def _rle_label_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return contiguous runs as (start, end_exclusive, label)."""
    x = np.asarray(labels).astype(int)
    if x.size == 0:
        return []
    runs: List[Tuple[int, int, int]] = []
    start = 0
    prev = int(x[0])
    for i in range(1, x.size):
        lab = int(x[i])
        if lab != prev:
            runs.append((start, i, prev))
            start = i
            prev = lab
    runs.append((start, x.size, prev))
    return runs


def _smooth_short_label_interruptions_global(
    labels: np.ndarray,
    *,
    max_interruption_bins: int,
    max_iterations: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Globally bridge short A-B-A label interruptions.

    If a short run of label B is surrounded on both sides by the same label A,
    and len(B) <= max_interruption_bins, replace B with A. This is applied
    globally before cluster/run selection, so all labels use the same cleaned
    sequence. The operation can be iterated because bridging one short gap can
    merge neighboring A runs and expose another short A-B-A gap.

    Example with max_interruption_bins >= 1:
        17 17 17 12 17 -> 17 17 17 17 17
    """
    original = np.asarray(labels).astype(int)
    smoothed = original.copy()
    max_interruption_bins = int(max_interruption_bins)
    max_iterations = max(1, int(max_iterations))

    info: Dict[str, Any] = {
        "global_short_interruption_smoothing_applied": False,
        "global_short_interruption_max_bins": max_interruption_bins,
        "global_short_interruption_iterations_requested": max_iterations,
        "global_short_interruption_iterations_used": 0,
        "global_short_interruption_runs_replaced": 0,
        "global_short_interruption_bins_replaced": 0,
        "global_short_interruption_examples": "",
        "majority_vote_label_smoothing_requested": bool(apply_majority_vote_label_smoothing),
        "majority_vote_label_smoothing_applied": False,
        "majority_vote_window_bins": int(majority_vote_window_bins),
        "majority_vote_window_ms": float(majority_vote_window_bins) * float(seconds_per_bin) * 1000.0,
        "majority_vote_bins_changed": 0,
        "majority_vote_sequences_smoothed": 0,
        "majority_vote_by_sequence": bool(majority_vote_by_file and file_indices_for_smoothing is not None),
    }

    if max_interruption_bins <= 0 or smoothed.size < 3:
        return smoothed, info

    examples: List[str] = []
    total_runs_replaced = 0
    total_bins_replaced = 0

    for iteration in range(max_iterations):
        runs = _rle_label_runs(smoothed)
        changed_this_iter = False
        for j in range(1, len(runs) - 1):
            left_start, left_end, left_lab = runs[j - 1]
            gap_start, gap_end, gap_lab = runs[j]
            right_start, right_end, right_lab = runs[j + 1]
            gap_len = int(gap_end - gap_start)

            if left_lab == right_lab and gap_lab != left_lab and gap_len <= max_interruption_bins:
                smoothed[gap_start:gap_end] = int(left_lab)
                total_runs_replaced += 1
                total_bins_replaced += gap_len
                changed_this_iter = True
                if len(examples) < 20:
                    examples.append(f"{left_lab}-{gap_lab}-{right_lab}@{gap_start}:{gap_end}({gap_len})")

        info["global_short_interruption_iterations_used"] = iteration + 1
        if not changed_this_iter:
            break

    info.update({
        "global_short_interruption_smoothing_applied": bool(total_runs_replaced > 0),
        "global_short_interruption_runs_replaced": int(total_runs_replaced),
        "global_short_interruption_bins_replaced": int(total_bins_replaced),
        "global_short_interruption_examples": ";".join(examples),
    })
    return smoothed, info



def _majority_vote_1d(labels: np.ndarray, window_size: int) -> np.ndarray:
    """Apply TweetyBERT-style temporal majority-vote smoothing to one sequence.

    For each position, use a dynamically clipped sliding window and assign the
    most frequent label. If multiple labels tie, choose the label that appears
    first in that window.
    """
    x = np.asarray(labels).astype(int)
    n = int(x.size)
    window_size = int(window_size)
    if n == 0 or window_size <= 1:
        return x.copy()

    # Use exactly window_size bins away from edges when possible.
    half_left = window_size // 2
    half_right = window_size - half_left

    out = np.empty_like(x)
    counts: Dict[int, int] = {}
    start = 0
    end = 0

    def _add(v: int) -> None:
        v = int(v)
        counts[v] = counts.get(v, 0) + 1

    def _remove(v: int) -> None:
        v = int(v)
        c = counts.get(v, 0) - 1
        if c <= 0:
            counts.pop(v, None)
        else:
            counts[v] = c

    for i in range(n):
        desired_start = max(0, i - half_left)
        desired_end = min(n, i + half_right)

        while start < desired_start:
            _remove(int(x[start]))
            start += 1
        while end < desired_end:
            _add(int(x[end]))
            end += 1

        if not counts:
            out[i] = int(x[i])
            continue

        max_count = max(counts.values())
        candidates = {lab for lab, c in counts.items() if c == max_count}
        if len(candidates) == 1:
            out[i] = next(iter(candidates))
        else:
            # Tie-break exactly as described: pick the tied label that appears
            # first in the current window.
            chosen = int(x[start])
            for lab in x[start:end]:
                lab = int(lab)
                if lab in candidates:
                    chosen = lab
                    break
            out[i] = chosen

    return out


def _apply_majority_vote_label_smoothing(
    labels: np.ndarray,
    *,
    window_bins: int,
    sequence_ids: Optional[np.ndarray] = None,
    smooth_by_sequence: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply global temporal majority-vote smoothing, optionally within files.

    This matches the smoothing described for TweetyBERT annotations: each bin is
    reassigned to the most frequent label in a centered sliding window, with
    dynamic clipping at sequence boundaries and first-in-window tie breaking.

    If sequence_ids are supplied and smooth_by_sequence=True, smoothing is applied
    independently to contiguous blocks with the same sequence/file ID so labels
    are not smoothed across recording boundaries.
    """
    original = np.asarray(labels).astype(int)
    smoothed = original.copy()
    window_bins = int(window_bins)

    info: Dict[str, Any] = {
        "majority_vote_label_smoothing_requested": bool(window_bins > 1),
        "majority_vote_label_smoothing_applied": False,
        "majority_vote_window_bins": int(window_bins),
        "majority_vote_bins_changed": 0,
        "majority_vote_sequences_smoothed": 0,
        "majority_vote_by_sequence": bool(smooth_by_sequence and sequence_ids is not None),
    }

    if window_bins <= 1 or original.size == 0:
        return smoothed, info

    if bool(smooth_by_sequence) and sequence_ids is not None:
        seq = np.asarray(sequence_ids)
        if seq.shape[0] != original.shape[0]:
            raise ValueError(
                f"sequence_ids length {seq.shape[0]} does not match labels length {original.shape[0]}"
            )
        start = 0
        n_seq = 0
        for i in range(1, original.size + 1):
            if i == original.size or seq[i] != seq[start]:
                smoothed[start:i] = _majority_vote_1d(original[start:i], window_bins)
                n_seq += 1
                start = i
        info["majority_vote_sequences_smoothed"] = int(n_seq)
    else:
        smoothed = _majority_vote_1d(original, window_bins)
        info["majority_vote_sequences_smoothed"] = 1

    n_changed = int(np.sum(smoothed != original))
    info.update({
        "majority_vote_label_smoothing_applied": bool(n_changed > 0),
        "majority_vote_bins_changed": n_changed,
    })
    return smoothed, info


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Pick the first matching column by case-insensitive name."""
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return None


def _label_key(x: Any) -> str:
    """Normalize labels like 3.0 -> '3' so CSV labels match integer HDBSCAN labels."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def _label_to_int(label: Any) -> Optional[int]:
    key = _label_key(label)
    if key == "":
        return None
    try:
        return int(key)
    except Exception:
        return None


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    text = s.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"])


def read_top_variance_cluster_ids(
    input_csv: Path,
    *,
    animal_id: str,
    top_fraction: float = 0.30,
    post_group_name: str = "Post",
    min_n_phrases: int = 100,
    animal_col: Optional[str] = None,
    label_col: Optional[str] = None,
    group_col: Optional[str] = None,
    nphrases_col: Optional[str] = None,
    phrase_variance_col: Optional[str] = None,
    rank_col: Optional[str] = None,
) -> List[int]:
    """
    Read either a previous top-variance table or the original phrase-duration table,
    then return the selected top-variance cluster IDs for one bird.

    Supported inputs:
      1) a table with is_top_phrase_variance already computed
      2) usage_balanced_phrase_duration_stats.csv, where this function filters to
         the post-lesion group and selects the top_fraction highest-variance labels
         within the requested animal.
    """
    input_csv = Path(input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Top-variance/phrase CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if animal_col is None:
        animal_col = _pick_column(df, ["animal_id", "Animal ID", "animal id", "bird", "Bird ID"])
    if label_col is None:
        label_col = _pick_column(df, ["syllable_key", "label", "cluster", "cluster_id", "HDBSCAN label", "Syllable", "syllable"])
    if group_col is None:
        group_col = _pick_column(df, ["Group", "group", "period", "phrase_group"])
    if nphrases_col is None:
        nphrases_col = _pick_column(df, ["N_phrases", "n_phrases", "phrase_n_phrases", "nphrases", "count"])
    if phrase_variance_col is None:
        phrase_variance_col = _pick_column(df, ["phrase_variance_ms2", "Variance_ms2", "variance_ms2", "Post_Variance_ms2"])
    if rank_col is None:
        rank_col = _pick_column(df, ["phrase_variance_rank_within_bird", "rank", "variance_rank"])

    if animal_col is None:
        raise KeyError(f"Could not infer animal ID column in {input_csv}. Columns: {list(df.columns)}")
    if label_col is None:
        raise KeyError(f"Could not infer label/cluster column in {input_csv}. Columns: {list(df.columns)}")
    if phrase_variance_col is None:
        raise KeyError(
            f"Could not infer phrase-variance column in {input_csv}. "
            f"Expected something like Variance_ms2 or phrase_variance_ms2. Columns: {list(df.columns)}"
        )

    out = df.copy()
    out["animal_id_for_filter"] = out[animal_col].astype(str).str.strip()
    out = out[out["animal_id_for_filter"] == str(animal_id).strip()].copy()
    if out.empty:
        raise ValueError(f"No rows in {input_csv} matched animal_id={animal_id!r}.")

    out["cluster_id_for_filter"] = out[label_col].map(_label_to_int)
    out = out[out["cluster_id_for_filter"].notna()].copy()
    out["cluster_id_for_filter"] = out["cluster_id_for_filter"].astype(int)
    out["_sort_phrase_variance"] = pd.to_numeric(out[phrase_variance_col], errors="coerce")

    if nphrases_col is not None and nphrases_col in out.columns:
        out["_n_phrases_for_filter"] = pd.to_numeric(out[nphrases_col], errors="coerce")
    else:
        out["_n_phrases_for_filter"] = np.nan

    # CASE 1: previous top-variance output table.
    if "is_top_phrase_variance" in out.columns:
        out = out[_as_bool_series(out["is_top_phrase_variance"])].copy()
        if rank_col is not None and rank_col in out.columns:
            out["_sort_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
        else:
            out["_sort_rank"] = np.nan
        out = (
            out.sort_values(
                ["_sort_rank", "_sort_phrase_variance", "cluster_id_for_filter"],
                ascending=[True, False, True],
                na_position="last",
            )
            .drop_duplicates(subset=["cluster_id_for_filter"], keep="first")
            .copy()
        )
    # CASE 2: original phrase-duration stats table.
    else:
        if group_col is not None and group_col in out.columns:
            out = out[
                out[group_col].astype(str).str.strip().str.lower()
                == str(post_group_name).strip().lower()
            ].copy()
            if out.empty:
                raise ValueError(
                    f"No rows matched post group {post_group_name!r} in column {group_col!r} "
                    f"for animal {animal_id}."
                )
        else:
            print(
                "[WARN] No Group column found and no is_top_phrase_variance column found. "
                "Using all rows and selecting top fraction within the bird."
            )

        out = out[np.isfinite(out["_sort_phrase_variance"])].copy()
        if nphrases_col is not None and nphrases_col in out.columns:
            out = out[np.isfinite(out["_n_phrases_for_filter"])].copy()
            out = out[out["_n_phrases_for_filter"] >= int(min_n_phrases)].copy()

        if out.empty:
            raise ValueError(
                "No phrase-duration rows survived filtering. Try lowering --top-min-n-phrases "
                "or check --post-group-name."
            )

        # Collapse duplicate animal/label rows by keeping the max post variance row.
        out = (
            out.sort_values(
                ["cluster_id_for_filter", "_sort_phrase_variance", "_n_phrases_for_filter"],
                ascending=[True, False, False],
                na_position="last",
            )
            .drop_duplicates(subset=["cluster_id_for_filter"], keep="first")
            .copy()
        )

        if not (0 < float(top_fraction) <= 1):
            raise ValueError("--top-fraction must be > 0 and <= 1")

        out = out.sort_values("_sort_phrase_variance", ascending=False).copy()
        n_total = len(out)
        n_top = max(1, int(math.ceil(n_total * float(top_fraction))))
        out["phrase_variance_rank_within_bird"] = np.arange(1, n_total + 1)
        out = out.head(n_top).copy()

    cluster_ids = sorted(int(x) for x in out["cluster_id_for_filter"].dropna().unique())
    return cluster_ids


def _load_module_from_path(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find spectrogram helper script: {path}")

    spec = importlib.util.spec_from_file_location("spectrogram_helper_for_umap_cluster_export", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper script from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "get_treatment_date_from_metadata",
        "get_timebin_recording_dates",
        "make_pre_post_masks",
        "_orient_spectrogram_to_FxT",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Spectrogram helper script {path} is missing expected function(s): {missing}"
        )
    return module


def _hide_spectrogram_ticks(ax) -> None:
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def _get_embedding_xy(arr: Mapping[str, Any], embedding_key: str) -> np.ndarray:
    if embedding_key not in arr:
        raise KeyError(
            f"Embedding key '{embedding_key}' was not found in the NPZ. "
            f"Available keys: {list(arr.keys())}"
        )
    xy = np.asarray(arr[embedding_key])
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"Expected '{embedding_key}' to have shape (N, 2), but got {xy.shape}. "
            "This helper expects already-computed UMAP coordinates, such as embedding_outputs."
        )
    return xy.astype(float)


def _get_optional_array(arr: Mapping[str, Any], key: str, expected_len: int) -> Optional[np.ndarray]:
    if key not in arr:
        return None
    x = np.asarray(arr[key])
    if x.shape[0] != expected_len:
        print(f"[WARN] Array '{key}' has length {x.shape[0]}, expected {expected_len}; ignoring it.")
        return None
    return x


# -----------------------------------------------------------------------------
# Period/group construction
# -----------------------------------------------------------------------------

def _ordered_unique_files_for_mask(mask: np.ndarray, file_indices: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([], dtype=file_indices.dtype)

    file_vals = file_indices[idx]
    first_pos: Dict[Any, int] = {}
    for i, f in zip(idx, file_vals):
        if f not in first_pos:
            first_pos[f] = int(i)
    ordered = sorted(first_pos.keys(), key=lambda f: first_pos[f])
    return np.asarray(ordered, dtype=file_indices.dtype)


def _split_mask_into_early_late(
    base_mask: np.ndarray,
    *,
    file_indices: Optional[np.ndarray],
    split_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    base_mask = np.asarray(base_mask, dtype=bool)
    early = np.zeros_like(base_mask, dtype=bool)
    late = np.zeros_like(base_mask, dtype=bool)

    idx = np.flatnonzero(base_mask)
    if idx.size == 0:
        return early, late

    if file_indices is None or split_method == "timebin_half":
        half = idx.size // 2
        early[idx[:half]] = True
        late[idx[half:]] = True
        return early, late

    files = _ordered_unique_files_for_mask(base_mask, file_indices)
    if files.size <= 1:
        # Fallback: split time bins if there is only one file represented.
        half = idx.size // 2
        early[idx[:half]] = True
        late[idx[half:]] = True
        return early, late

    half = files.size // 2
    early_files = set(files[:half].tolist())
    late_files = set(files[half:].tolist())

    early = base_mask & np.isin(file_indices, list(early_files))
    late = base_mask & np.isin(file_indices, list(late_files))
    return early, late


def build_period_masks(
    *,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    file_indices: Optional[np.ndarray],
    period_mode: str,
    early_late_split_method: str,
) -> Dict[str, np.ndarray]:
    """Return named period masks used for equal-sized UMAP/spectrogram exports."""
    pre_mask = np.asarray(pre_mask, dtype=bool)
    post_mask = np.asarray(post_mask, dtype=bool)

    if period_mode == "pre_post":
        return {
            "pre": pre_mask,
            "post": post_mask,
        }

    if period_mode == "early_late_pre_post":
        early_pre, late_pre = _split_mask_into_early_late(
            pre_mask,
            file_indices=file_indices,
            split_method=early_late_split_method,
        )
        early_post, late_post = _split_mask_into_early_late(
            post_mask,
            file_indices=file_indices,
            split_method=early_late_split_method,
        )
        return {
            "early_pre": early_pre,
            "late_pre": late_pre,
            "early_post": early_post,
            "late_post": late_post,
        }

    raise ValueError(f"Unknown period_mode: {period_mode}")


# -----------------------------------------------------------------------------
# Equal-size time-bin selection
# -----------------------------------------------------------------------------

def _choose_indices(
    idx: np.ndarray,
    *,
    n: int,
    sample_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose n indices from an already time-ordered index vector.

    Modes:
      first
        Take the first n available bins.

      random
        Randomly sample n available bins.

      time_balanced
        Deterministically sample evenly spaced bins across the available
        cluster-labeled time bins for that group. This keeps the same number
        of bins per group but spreads them across the full available pool.

      time_balanced_random_offset
        Same idea as time_balanced, but uses a random offset within the
        sampling stride so repeated seeds/draws can sample a different evenly
        spaced grid.
    """
    idx = np.asarray(idx, dtype=int)
    if n <= 0:
        return np.array([], dtype=int)
    if idx.size < n:
        raise ValueError(f"Cannot choose {n} indices from only {idx.size} available indices.")

    mode = str(sample_mode).strip().lower()

    if mode == "first":
        return np.asarray(idx[:n], dtype=int)

    if mode == "random":
        return np.sort(rng.choice(idx, size=n, replace=False)).astype(int)

    if mode == "time_balanced":
        # Evenly spaced positions through the available cluster-time axis.
        # If n == idx.size, this returns every bin.
        pos = np.floor((np.arange(n, dtype=float) + 0.5) * (idx.size / float(n))).astype(int)
        pos = np.clip(pos, 0, idx.size - 1)
        return np.asarray(idx[pos], dtype=int)

    if mode == "time_balanced_random_offset":
        # Systematic sample with a random phase/offset.
        stride = idx.size / float(n)
        offset = rng.uniform(0.0, stride)
        pos = np.floor(offset + np.arange(n, dtype=float) * stride).astype(int)
        pos = np.clip(pos, 0, idx.size - 1)
        return np.asarray(idx[pos], dtype=int)

    raise ValueError(
        "sample_mode must be one of: 'first', 'random', "
        "'time_balanced', or 'time_balanced_random_offset'."
    )


def select_equal_timebins_for_cluster(
    *,
    labels: np.ndarray,
    cluster_id: int,
    period_masks: Mapping[str, np.ndarray],
    min_points_per_group: int,
    max_points_per_group: Optional[int],
    sample_mode: str,
    rng: np.random.Generator,
    vocalization_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, int], int]:
    """
    Select equal-sized time-bin arrays for one cluster.

    Returns:
        selected_by_group: dict of group name -> selected time-bin indices, or None if skipped
        available_counts: dict of group name -> available time-bin count before equalization
        n_equal: number selected per group
    """
    labels = np.asarray(labels).astype(int)
    cluster_mask = labels == int(cluster_id)
    if vocalization_mask is not None:
        cluster_mask = cluster_mask & np.asarray(vocalization_mask, dtype=bool)

    raw_by_group: Dict[str, np.ndarray] = {}
    available_counts: Dict[str, int] = {}
    for group_name, group_mask in period_masks.items():
        idx = np.flatnonzero(cluster_mask & np.asarray(group_mask, dtype=bool))
        raw_by_group[group_name] = idx.astype(int)
        available_counts[group_name] = int(idx.size)

    if not raw_by_group:
        return None, available_counts, 0

    n_equal = min(available_counts.values()) if available_counts else 0
    if max_points_per_group is not None and int(max_points_per_group) > 0:
        n_equal = min(n_equal, int(max_points_per_group))

    if n_equal < int(min_points_per_group):
        return None, available_counts, int(n_equal)

    selected_by_group = {
        group_name: _choose_indices(idx, n=n_equal, sample_mode=sample_mode, rng=rng)
        for group_name, idx in raw_by_group.items()
    }
    return selected_by_group, available_counts, int(n_equal)


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def _chunk_indices(idx: np.ndarray, bins_per_row: int, max_rows: Optional[int] = None) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    bins_per_row = max(1, int(bins_per_row))
    chunks = [idx[start:start + bins_per_row] for start in range(0, idx.size, bins_per_row)]
    if max_rows is not None and int(max_rows) > 0:
        chunks = chunks[: int(max_rows)]
    return [np.asarray(ch, dtype=int) for ch in chunks if ch.size > 0]


def _contiguous_segments_from_sorted_indices(idx: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open contiguous segments [start, end) from sorted integer indices."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []
    idx = np.sort(idx)
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, breaks)
    return [(int(ch[0]), int(ch[-1]) + 1) for ch in chunks if ch.size > 0]


def build_label_run_lookup(labels: np.ndarray, cluster_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a lookup from time-bin index to contiguous HDBSCAN label-run ID.

    Returns:
        run_id_for_bin: length-N array, -1 outside this cluster.
        run_starts: half-open run starts.
        run_ends: half-open run ends.
    """
    labels = np.asarray(labels).astype(int)
    mask = labels == int(cluster_id)
    true_idx = np.flatnonzero(mask)
    run_id_for_bin = np.full(labels.shape[0], -1, dtype=np.int64)
    if true_idx.size == 0:
        return run_id_for_bin, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    run_segments = _contiguous_segments_from_sorted_indices(true_idx)
    run_starts = np.asarray([s for s, _ in run_segments], dtype=np.int64)
    run_ends = np.asarray([e for _, e in run_segments], dtype=np.int64)
    for run_id, (s, e) in enumerate(run_segments):
        run_id_for_bin[s:e] = int(run_id)
    return run_id_for_bin, run_starts, run_ends


def summarize_selected_bins_within_full_runs(
    *,
    selected_indices: np.ndarray,
    run_id_for_bin: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    seconds_per_bin: float,
    group_name: str,
    cluster_id: int,
) -> pd.DataFrame:
    """Summarize which full HDBSCAN runs overlap the selected equal UMAP time bins."""
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.size == 0:
        return pd.DataFrame()

    valid = selected_indices[(selected_indices >= 0) & (selected_indices < run_id_for_bin.shape[0])]
    selected_run_ids = run_id_for_bin[valid]
    selected_run_ids = selected_run_ids[selected_run_ids >= 0]
    selected_run_ids = np.unique(selected_run_ids).astype(int)

    selected_set = set(int(x) for x in valid.tolist())
    rows: List[Dict[str, Any]] = []
    for run_id in selected_run_ids:
        s = int(run_starts[run_id])
        e = int(run_ends[run_id])
        selected_in_run = np.asarray([x for x in range(s, e) if x in selected_set], dtype=int)
        selected_segments = _contiguous_segments_from_sorted_indices(selected_in_run)
        run_n_bins = int(e - s)
        selected_n_bins = int(selected_in_run.size)
        coverage = float(selected_n_bins / run_n_bins) if run_n_bins > 0 else np.nan
        rows.append({
            "cluster_id": int(cluster_id),
            "group": group_name,
            "run_id": int(run_id),
            "run_start_bin": s,
            "run_end_bin_exclusive": e,
            "run_n_bins": run_n_bins,
            "run_duration_ms": run_n_bins * float(seconds_per_bin) * 1000.0,
            "selected_n_bins_in_run": selected_n_bins,
            "selected_duration_ms_in_run": selected_n_bins * float(seconds_per_bin) * 1000.0,
            "coverage_fraction": coverage,
            "n_selected_segments_in_run": int(len(selected_segments)),
            "selected_segment_starts": ";".join(str(a) for a, _ in selected_segments),
            "selected_segment_ends_exclusive": ";".join(str(b) for _, b in selected_segments),
        })
    return pd.DataFrame(rows)


def expand_selected_bins_to_full_runs(
    *,
    selected_indices: np.ndarray,
    run_id_for_bin: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    max_total_bins: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Expand selected equal UMAP time bins to complete contiguous label runs.

    The returned index vector preserves each full run internally, then stitches
    full runs together in time order. If max_total_bins is set, it truncates the
    concatenated full-run bins to avoid enormous PNGs.
    """
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.size == 0:
        return np.array([], dtype=int), []
    valid = selected_indices[(selected_indices >= 0) & (selected_indices < run_id_for_bin.shape[0])]
    run_ids = run_id_for_bin[valid]
    run_ids = sorted(int(x) for x in np.unique(run_ids) if int(x) >= 0)
    arrays: List[np.ndarray] = []
    total = 0
    for run_id in run_ids:
        s = int(run_starts[run_id])
        e = int(run_ends[run_id])
        if e <= s:
            continue
        arr = np.arange(s, e, dtype=int)
        if max_total_bins is not None and int(max_total_bins) > 0:
            remaining = int(max_total_bins) - total
            if remaining <= 0:
                break
            arr = arr[:remaining]
        arrays.append(arr)
        total += int(arr.size)
        if max_total_bins is not None and int(max_total_bins) > 0 and total >= int(max_total_bins):
            break
    if not arrays:
        return np.array([], dtype=int), run_ids
    return np.concatenate(arrays).astype(int), run_ids


# -----------------------------------------------------------------------------
# Run-balanced, phrase-normalized BC selection and visualization
# -----------------------------------------------------------------------------

def _candidate_run_ids_for_group(
    *,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    group_mask: np.ndarray,
    min_run_bins: int,
    phase_bins_per_run: int,
    min_group_fraction: float,
    vocalization_mask: Optional[np.ndarray] = None,
    allow_repeat_phase_bins: bool = False,
) -> List[int]:
    """Return full contiguous label-run IDs that are eligible for one group.

    A run is eligible if most of its bins belong to the requested period/group.
    This avoids selecting a run that only grazes a group boundary. By default,
    runs also need enough bins to sample the requested phrase phases without
    reusing the same bin.
    """
    group_mask = np.asarray(group_mask, dtype=bool)
    min_run_bins = max(1, int(min_run_bins))
    phase_bins_per_run = max(1, int(phase_bins_per_run))
    min_group_fraction = float(min_group_fraction)

    out: List[int] = []
    for run_id, (s0, e0) in enumerate(zip(run_starts, run_ends)):
        s = int(s0)
        e = int(e0)
        n = int(e - s)
        if n <= 0:
            continue
        if n < min_run_bins:
            continue
        if (not allow_repeat_phase_bins) and n < phase_bins_per_run:
            continue
        group_fraction = float(np.mean(group_mask[s:e])) if n > 0 else 0.0
        if group_fraction < min_group_fraction:
            continue
        if vocalization_mask is not None:
            voc = np.asarray(vocalization_mask, dtype=bool)
            if float(np.mean(voc[s:e])) < min_group_fraction:
                continue
        out.append(int(run_id))
    return out


def _choose_run_ids(
    run_ids: Sequence[int],
    *,
    n: int,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> List[int]:
    run_ids = [int(x) for x in run_ids]
    n = int(n)
    if n <= 0:
        return []
    if len(run_ids) < n:
        raise ValueError(f"Cannot choose {n} runs from only {len(run_ids)} candidate runs.")
    mode = str(mode).lower()
    if mode == "first":
        return sorted(run_ids, key=lambda r: int(run_starts[r]))[:n]
    if mode == "longest":
        return sorted(run_ids, key=lambda r: int(run_ends[r] - run_starts[r]), reverse=True)[:n]
    if mode == "random":
        return sorted([int(x) for x in rng.choice(np.asarray(run_ids, dtype=int), size=n, replace=False)], key=lambda r: int(run_starts[r]))
    raise ValueError("run_sample_mode must be one of: random, first, longest")


def _sample_phrase_phase_bins_for_run(
    *,
    run_start: int,
    run_end: int,
    phase_bins_per_run: int,
    allow_repeat_phase_bins: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample fixed normalized phase positions from one full phrase-label run.

    Uses phase-bin centers, so for 25 phase bins the first sample is around 2%
    of the run and the last is around 98% of the run. This avoids overfocusing
    on boundaries while still spanning the entire phrase-label bout.
    """
    s = int(run_start)
    e = int(run_end)
    n = int(e - s)
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    phase_bins_per_run = max(1, int(phase_bins_per_run))
    if (not allow_repeat_phase_bins) and n < phase_bins_per_run:
        return np.array([], dtype=int), np.array([], dtype=float)
    phase_fracs = (np.arange(phase_bins_per_run, dtype=float) + 0.5) / float(phase_bins_per_run)
    rel = np.floor(phase_fracs * n).astype(int)
    rel = np.clip(rel, 0, n - 1)
    bins = s + rel
    return bins.astype(int), phase_fracs.astype(float)


def select_run_balanced_phrase_normalized_bins_for_cluster(
    *,
    labels: np.ndarray,
    cluster_id: int,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    period_masks: Mapping[str, np.ndarray],
    min_runs_per_group: int,
    max_runs_per_group: Optional[int],
    phase_bins_per_run: int,
    min_full_run_duration_ms: float,
    seconds_per_bin: float,
    min_run_group_fraction: float,
    run_sample_mode: str,
    rng: np.random.Generator,
    vocalization_mask: Optional[np.ndarray] = None,
    allow_repeat_phase_bins: bool = False,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, int], int, Dict[str, np.ndarray], pd.DataFrame]:
    """Select equal numbers of full phrase-label runs and normalized phase bins.

    Returns:
      selected_by_group: group -> flattened time-bin indices, one fixed phase grid per run
      available_run_counts: group -> candidate full-run count before equalization
      n_equal_runs: selected full runs per group
      phase_indices_by_group: group -> array shape (n_equal_runs, phase_bins_per_run)
      selection_df: one row per selected phase-bin sample
    """
    min_run_bins = max(1, int(math.ceil(float(min_full_run_duration_ms) / (float(seconds_per_bin) * 1000.0))))
    phase_bins_per_run = max(1, int(phase_bins_per_run))

    candidate_by_group: Dict[str, List[int]] = {}
    available_run_counts: Dict[str, int] = {}
    for group_name, group_mask in period_masks.items():
        candidates = _candidate_run_ids_for_group(
            run_starts=run_starts,
            run_ends=run_ends,
            group_mask=np.asarray(group_mask, dtype=bool),
            min_run_bins=min_run_bins,
            phase_bins_per_run=phase_bins_per_run,
            min_group_fraction=float(min_run_group_fraction),
            vocalization_mask=vocalization_mask,
            allow_repeat_phase_bins=bool(allow_repeat_phase_bins),
        )
        candidate_by_group[group_name] = candidates
        available_run_counts[group_name] = int(len(candidates))

    if not candidate_by_group:
        return None, available_run_counts, 0, {}, pd.DataFrame()

    n_equal_runs = min(available_run_counts.values()) if available_run_counts else 0
    if max_runs_per_group is not None and int(max_runs_per_group) > 0:
        n_equal_runs = min(n_equal_runs, int(max_runs_per_group))
    if n_equal_runs < int(min_runs_per_group):
        return None, available_run_counts, int(n_equal_runs), {}, pd.DataFrame()

    selected_by_group: Dict[str, np.ndarray] = {}
    phase_indices_by_group: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for group_name, candidates in candidate_by_group.items():
        chosen_runs = _choose_run_ids(
            candidates,
            n=int(n_equal_runs),
            run_starts=run_starts,
            run_ends=run_ends,
            mode=str(run_sample_mode),
            rng=rng,
        )
        matrices: List[np.ndarray] = []
        for run_order, run_id in enumerate(chosen_runs):
            s = int(run_starts[run_id])
            e = int(run_ends[run_id])
            bins, phase_fracs = _sample_phrase_phase_bins_for_run(
                run_start=s,
                run_end=e,
                phase_bins_per_run=phase_bins_per_run,
                allow_repeat_phase_bins=bool(allow_repeat_phase_bins),
            )
            if bins.size != phase_bins_per_run:
                continue
            matrices.append(bins)
            for phase_i, (timebin_index, phase_fraction) in enumerate(zip(bins, phase_fracs)):
                rows.append({
                    "cluster_id": int(cluster_id),
                    "group": group_name,
                    "run_order_within_group": int(run_order),
                    "run_id": int(run_id),
                    "run_start_bin": s,
                    "run_end_bin_exclusive": e,
                    "run_n_bins": int(e - s),
                    "run_duration_ms": float(e - s) * float(seconds_per_bin) * 1000.0,
                    "phase_bin_index": int(phase_i),
                    "phrase_phase_fraction": float(phase_fraction),
                    "timebin_index": int(timebin_index),
                })
        if not matrices:
            return None, available_run_counts, 0, {}, pd.DataFrame()
        mat = np.vstack(matrices).astype(int)
        phase_indices_by_group[group_name] = mat
        selected_by_group[group_name] = mat.ravel().astype(int)

    return selected_by_group, available_run_counts, int(n_equal_runs), phase_indices_by_group, pd.DataFrame(rows)


def select_run_balanced_full_contiguous_bins_for_cluster(
    *,
    labels: np.ndarray,
    cluster_id: int,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    period_masks: Mapping[str, np.ndarray],
    min_runs_per_group: int,
    max_runs_per_group: Optional[int],
    min_full_run_duration_ms: float,
    seconds_per_bin: float,
    min_run_group_fraction: float,
    run_sample_mode: str,
    rng: np.random.Generator,
    vocalization_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, int], int, pd.DataFrame, Dict[str, int]]:
    """Select equal numbers of full contiguous phrase-label runs per group.

    Unlike run_balanced_phrase_normalized, this mode does NOT sample a fixed
    number of phase bins from each run. Instead, after selecting matched full
    runs, every time bin inside each selected contiguous run is used for UMAP/BC.

    Returns:
      selected_by_group: group -> all time-bin indices from all selected runs
      available_run_counts: group -> candidate full-run count before equalization
      n_equal_runs: selected full runs per group
      selection_df: one row per selected full-run time bin, with run boundaries
      selected_bin_counts: group -> number of selected bins after expansion
    """
    min_run_bins = max(1, int(math.ceil(float(min_full_run_duration_ms) / (float(seconds_per_bin) * 1000.0))))

    candidate_by_group: Dict[str, List[int]] = {}
    available_run_counts: Dict[str, int] = {}
    for group_name, group_mask in period_masks.items():
        # phase_bins_per_run=1 avoids imposing phrase-phase sampling constraints.
        candidates = _candidate_run_ids_for_group(
            run_starts=run_starts,
            run_ends=run_ends,
            group_mask=np.asarray(group_mask, dtype=bool),
            min_run_bins=min_run_bins,
            phase_bins_per_run=1,
            min_group_fraction=float(min_run_group_fraction),
            vocalization_mask=vocalization_mask,
            allow_repeat_phase_bins=True,
        )
        candidate_by_group[group_name] = candidates
        available_run_counts[group_name] = int(len(candidates))

    if not candidate_by_group:
        return None, available_run_counts, 0, pd.DataFrame(), {}

    n_equal_runs = min(available_run_counts.values()) if available_run_counts else 0
    if max_runs_per_group is not None and int(max_runs_per_group) > 0:
        n_equal_runs = min(n_equal_runs, int(max_runs_per_group))
    if n_equal_runs < int(min_runs_per_group):
        return None, available_run_counts, int(n_equal_runs), pd.DataFrame(), {}

    selected_by_group: Dict[str, np.ndarray] = {}
    selected_bin_counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    for group_name, candidates in candidate_by_group.items():
        chosen_runs = _choose_run_ids(
            candidates,
            n=int(n_equal_runs),
            run_starts=run_starts,
            run_ends=run_ends,
            mode=str(run_sample_mode),
            rng=rng,
        )
        chunks: List[np.ndarray] = []
        for run_order, run_id in enumerate(chosen_runs):
            s = int(run_starts[run_id])
            e = int(run_ends[run_id])
            if e <= s:
                continue
            bins = np.arange(s, e, dtype=int)
            chunks.append(bins)
            run_n_bins = int(e - s)
            for offset, timebin_index in enumerate(bins):
                rows.append({
                    "cluster_id": int(cluster_id),
                    "group": group_name,
                    "run_order_within_group": int(run_order),
                    "run_id": int(run_id),
                    "run_start_bin": s,
                    "run_end_bin_exclusive": e,
                    "run_n_bins": run_n_bins,
                    "run_duration_ms": float(run_n_bins) * float(seconds_per_bin) * 1000.0,
                    "within_run_bin_index": int(offset),
                    "phrase_phase_fraction": float((offset + 0.5) / run_n_bins) if run_n_bins > 0 else np.nan,
                    "timebin_index": int(timebin_index),
                    "selection_mode": "run_balanced_full_contiguous",
                })
        if not chunks:
            return None, available_run_counts, 0, pd.DataFrame(), {}
        idx = np.concatenate(chunks).astype(int)
        selected_by_group[group_name] = idx
        selected_bin_counts[group_name] = int(idx.size)

    return selected_by_group, available_run_counts, int(n_equal_runs), pd.DataFrame(rows), selected_bin_counts


def save_phrase_normalized_selected_bins_csv(
    *,
    selection_df: pd.DataFrame,
    out_csv: Path,
    file_indices: Optional[np.ndarray],
    timebin_dates: Optional[np.ndarray],
) -> None:
    df = selection_df.copy()
    if not df.empty:
        tb = df["timebin_index"].astype(int).to_numpy()
        if file_indices is not None:
            df["file_index"] = file_indices[tb]
        if timebin_dates is not None:
            df["recording_date"] = [str(timebin_dates[i]) for i in tb]
    pd.DataFrame(df).to_csv(out_csv, index=False)


def compute_phrase_normalized_bc_by_phase(
    *,
    embedding_xy: np.ndarray,
    phase_indices_by_group: Mapping[str, np.ndarray],
    density_bins: int,
) -> pd.DataFrame:
    """Compute phase-by-phase BC curves for run-balanced phrase-normalized samples."""
    if not phase_indices_by_group:
        return pd.DataFrame()
    xy = np.asarray(embedding_xy, dtype=float)
    all_selected = np.concatenate([np.asarray(v).ravel() for v in phase_indices_by_group.values() if np.asarray(v).size > 0])
    if all_selected.size == 0:
        return pd.DataFrame()

    # Use selected phrase-normalized points to define the density grid. This focuses
    # the phase-normalized BC on the cluster region rather than wasting bins on
    # far-away UMAP space.
    sx = xy[all_selected, 0]
    sy = xy[all_selected, 1]
    x_min, x_max = float(np.nanmin(sx)), float(np.nanmax(sx))
    y_min, y_max = float(np.nanmin(sy)), float(np.nanmax(sy))
    pad_x = max(1e-6, (x_max - x_min) * 0.05)
    pad_y = max(1e-6, (y_max - y_min) * 0.05)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, int(density_bins) + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, int(density_bins) + 1)

    first_mat = next(iter(phase_indices_by_group.values()))
    n_phase = int(np.asarray(first_mat).shape[1])

    comparisons: List[Tuple[str, np.ndarray, np.ndarray]] = []
    keys = set(phase_indices_by_group.keys())
    if {"early_pre", "late_pre"}.issubset(keys):
        comparisons.append(("pre_early_vs_late", np.asarray(phase_indices_by_group["early_pre"]), np.asarray(phase_indices_by_group["late_pre"])))
    if {"early_post", "late_post"}.issubset(keys):
        comparisons.append(("post_early_vs_late", np.asarray(phase_indices_by_group["early_post"]), np.asarray(phase_indices_by_group["late_post"])))
    if {"early_pre", "late_pre", "early_post", "late_post"}.issubset(keys):
        pre_mat = np.vstack([np.asarray(phase_indices_by_group["early_pre"]), np.asarray(phase_indices_by_group["late_pre"])])
        post_mat = np.vstack([np.asarray(phase_indices_by_group["early_post"]), np.asarray(phase_indices_by_group["late_post"])])
        comparisons.append(("pre_vs_post", pre_mat, post_mat))
    if len(comparisons) == 0 and len(phase_indices_by_group) >= 2:
        names = list(phase_indices_by_group.keys())[:2]
        comparisons.append((f"{names[0]}_vs_{names[1]}", np.asarray(phase_indices_by_group[names[0]]), np.asarray(phase_indices_by_group[names[1]])))

    rows: List[Dict[str, Any]] = []
    for comp_name, mat_a, mat_b in comparisons:
        for phase_i in range(n_phase):
            idx_a = mat_a[:, phase_i].astype(int)
            idx_b = mat_b[:, phase_i].astype(int)
            ha = _compute_hist_density(xy, idx_a, x_edges=x_edges, y_edges=y_edges)
            hb = _compute_hist_density(xy, idx_b, x_edges=x_edges, y_edges=y_edges)
            rows.append({
                "comparison": comp_name,
                "phase_bin_index": int(phase_i),
                "phrase_phase_fraction": float((phase_i + 0.5) / n_phase),
                "n_a_runs": int(mat_a.shape[0]),
                "n_b_runs": int(mat_b.shape[0]),
                "bc": _hist_bc(ha, hb),
                "density_bins": int(density_bins),
            })
    return pd.DataFrame(rows)


def plot_phrase_normalized_bc_by_phase(
    *,
    phase_bc_df: pd.DataFrame,
    animal_id: str,
    cluster_id: int,
    out_png: Path,
    dpi: int,
) -> None:
    if phase_bc_df is None or phase_bc_df.empty:
        raise ValueError("No phrase-normalized BC rows to plot.")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for comp_name, sub in phase_bc_df.groupby("comparison", sort=False):
        sub = sub.sort_values("phrase_phase_fraction")
        ax.plot(sub["phrase_phase_fraction"], sub["bc"], marker="o", linewidth=1.5, label=str(comp_name))
    ax.set_xlabel("Normalized phrase-label run phase (0=start, 1=end)")
    ax.set_ylabel("Bhattacharyya coefficient")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{animal_id} label {cluster_id}: phrase-normalized BC by run phase")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_umap_phrase_phase_colored(
    *,
    embedding_xy: np.ndarray,
    selection_df: pd.DataFrame,
    animal_id: str,
    cluster_id: int,
    out_png: Path,
    dpi: int,
    max_points_per_panel: int = 25000,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Plot phrase-normalized selected UMAP points colored by within-run phase."""
    if selection_df is None or selection_df.empty:
        raise ValueError("No phrase-normalized selection rows to plot.")
    if rng is None:
        rng = np.random.default_rng(0)
    xy = np.asarray(embedding_xy, dtype=float)
    groups = list(dict.fromkeys(selection_df["group"].astype(str).tolist()))
    n = len(groups)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.8 * nrows), squeeze=False)
    last_sc = None
    for ax, group_name in zip(axes.ravel(), groups):
        sub = selection_df[selection_df["group"].astype(str) == group_name].copy()
        if len(sub) > int(max_points_per_panel):
            sub = sub.sample(n=int(max_points_per_panel), random_state=int(rng.integers(0, 2**31 - 1)))
        idx = sub["timebin_index"].astype(int).to_numpy()
        phase = sub["phrase_phase_fraction"].astype(float).to_numpy()
        last_sc = ax.scatter(xy[idx, 0], xy[idx, 1], c=phase, s=5, alpha=0.75, linewidths=0, vmin=0, vmax=1)
        n_runs = sub["run_id"].nunique() if "run_id" in sub.columns else np.nan
        ax.set_title(f"{group_name}\n{int(n_runs):,} runs; {len(sub):,} phase-normalized points")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("Normalized phrase phase")
    fig.suptitle(f"{animal_id} label {cluster_id}: UMAP points colored by phrase phase", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _choose_run_rows_for_context_examples(
    run_df: pd.DataFrame,
    *,
    max_examples: int,
    mode: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if run_df is None or run_df.empty or int(max_examples) <= 0:
        return pd.DataFrame()
    df = run_df.copy()
    if mode == "longest":
        return df.sort_values("run_n_bins", ascending=False).head(int(max_examples)).copy()
    if mode == "shortest":
        return df.sort_values("run_n_bins", ascending=True).head(int(max_examples)).copy()
    if mode == "random":
        n = min(int(max_examples), len(df))
        return df.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    # mixed: include long, short, and random examples.
    max_examples = int(max_examples)
    n_long = max(1, max_examples // 3)
    n_short = max(1, max_examples // 3)
    long_df = df.sort_values("run_n_bins", ascending=False).head(n_long)
    short_df = df.sort_values("run_n_bins", ascending=True).head(n_short)
    used = set(long_df.index.tolist()) | set(short_df.index.tolist())
    remaining = df.loc[[i for i in df.index if i not in used]]
    n_remaining = max(0, max_examples - len(long_df) - len(short_df))
    if n_remaining > 0 and len(remaining) > 0:
        rand_df = remaining.sample(n=min(n_remaining, len(remaining)), random_state=int(rng.integers(0, 2**31 - 1)))
        return pd.concat([long_df, short_df, rand_df], axis=0).drop_duplicates().head(max_examples).copy()
    return pd.concat([long_df, short_df], axis=0).drop_duplicates().head(max_examples).copy()


def _parse_semicolon_ints(x: Any) -> List[int]:
    if pd.isna(x):
        return []
    text = str(x).strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except Exception:
                pass
    return out


def _fixed_duration_panel(
    *,
    S_FxT: np.ndarray,
    context_start: int,
    panel_n_bins: int,
) -> np.ndarray:
    """Return a fixed-width spectrogram panel padded with NaN to avoid stretching.

    Shorter available contexts occupy only the left portion of the panel, so all
    rows can share the same x-axis without visually stretching the spectrogram.
    """
    panel = np.full((S_FxT.shape[0], int(panel_n_bins)), np.nan, dtype=float)
    context_start = int(context_start)
    end = min(S_FxT.shape[1], context_start + int(panel_n_bins))
    n = max(0, end - context_start)
    if n > 0:
        panel[:, :n] = S_FxT[:, context_start:end].astype(float)
    return panel


def plot_representative_full_run_contexts(
    *,
    animal_id: str,
    cluster_id: int,
    group_name: str,
    run_df: pd.DataFrame,
    S_FxT: np.ndarray,
    out_png: Path,
    seconds_per_bin: float,
    fixed_panel_duration_s: float,
    context_bins: int,
    max_context_bins: int,
    max_examples: int,
    example_mode: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    rng: np.random.Generator,
    labels: Optional[np.ndarray] = None,
    show_all_labels_track: bool = False,
) -> None:
    """Plot representative original-context full label runs.

    Blue shading shows the complete HDBSCAN phrase-label run. Red shading shows
    the equal UMAP-selected bins within that full run. When
    show_all_labels_track=True, a compact annotation strip is added at the top of
    each panel showing every contiguous label segment in the displayed context.

    Alignment note:
    The spectrogram image and the optional label strip are both rendered from the
    exact same displayed bin window and use the same bin-edge x coordinates. This
    avoids subtle offsets caused by mixing fixed requested durations with the
    actual number of displayed bins.
    """
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    chosen = _choose_run_rows_for_context_examples(
        run_df,
        max_examples=int(max_examples),
        mode=str(example_mode),
        rng=rng,
    )
    if chosen.empty:
        raise ValueError(f"No full runs available for representative context plot: {group_name}")

    n_rows = len(chosen)
    fig_h = max(2.0, 1.25 * n_rows + 0.9)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, fig_h), sharex=False)
    if n_rows == 1:
        axes = [axes]

    N = S_FxT.shape[1]
    panel_n_bins = max(1, int(round(float(fixed_panel_duration_s) / float(seconds_per_bin))))
    last_display_duration_s = float(panel_n_bins) * float(seconds_per_bin)

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        run_start = int(row["run_start_bin"])
        run_end = int(row["run_end_bin_exclusive"])
        context_start = max(0, run_start - int(context_bins))

        if int(max_context_bins) > 0:
            panel_n_bins_this_row = min(panel_n_bins, int(max_context_bins))
        else:
            panel_n_bins_this_row = panel_n_bins

        panel = _fixed_duration_panel(
            S_FxT=S_FxT,
            context_start=context_start,
            panel_n_bins=panel_n_bins_this_row,
        )

        display_n_bins = int(panel.shape[1])
        display_duration_s = float(display_n_bins) * float(seconds_per_bin)
        last_display_duration_s = display_duration_s
        x_edges_s = np.arange(display_n_bins + 1, dtype=float) * float(seconds_per_bin)

        context_end = min(N, context_start + display_n_bins)
        actual_n_bins = max(0, context_end - context_start)

        display_labels = None
        if labels is not None:
            display_labels = np.asarray(labels[context_start:context_end]).astype(int)
            if int(display_labels.size) != int(actual_n_bins):
                raise ValueError(
                    f"Label/spectrogram alignment mismatch for {animal_id} label {cluster_id} {group_name}: "
                    f"expected {actual_n_bins} displayed label bins, got {display_labels.size}."
                )

        ax.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[float(x_edges_s[0]), float(x_edges_s[-1]), 0, panel.shape[0]],
            interpolation="nearest",
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_xlim(0.0, float(display_duration_s))

        def _clip_abs_bin_interval_to_panel(abs_start: int, abs_end: int) -> Optional[Tuple[float, float]]:
            s_abs = max(int(abs_start), context_start)
            e_abs = min(int(abs_end), context_end)
            if e_abs <= s_abs:
                return None
            rel0 = max(0, s_abs - context_start)
            rel1 = min(display_n_bins, e_abs - context_start)
            if rel1 <= rel0:
                return None
            return float(x_edges_s[rel0]), float(x_edges_s[rel1])

        blue_span = _clip_abs_bin_interval_to_panel(run_start, run_end)
        if blue_span is not None:
            blue_x0, blue_x1 = blue_span
            ax.axvspan(blue_x0, blue_x1, color="tab:blue", alpha=0.08, zorder=2)
            y0, y1 = ax.get_ylim()
            ax.add_patch(
                plt.Rectangle(
                    (blue_x0, y0),
                    blue_x1 - blue_x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="tab:blue",
                    linewidth=1.2,
                    zorder=6,
                )
            )
            ax.hlines(
                y0 + 0.04 * (y1 - y0),
                blue_x0,
                blue_x1,
                colors="tab:blue",
                linewidth=2.6,
                zorder=7,
            )

        seg_starts = _parse_semicolon_ints(row.get("selected_segment_starts", ""))
        seg_ends = _parse_semicolon_ints(row.get("selected_segment_ends_exclusive", ""))
        for s, e in zip(seg_starts, seg_ends):
            red_span = _clip_abs_bin_interval_to_panel(int(s), int(e))
            if red_span is not None:
                red_x0, red_x1 = red_span
                ax.axvspan(red_x0, red_x1, color="tab:red", alpha=0.10, zorder=4)
                ax.vlines(
                    [red_x0, red_x1],
                    ymin=ax.get_ylim()[0],
                    ymax=ax.get_ylim()[1],
                    colors="tab:red",
                    linewidth=0.45,
                    alpha=0.75,
                    zorder=8,
                )

        if bool(show_all_labels_track) and display_labels is not None and display_labels.size > 0:
            y0, y1 = ax.get_ylim()
            track_h = max(2.0, 0.13 * (y1 - y0))
            track_y0 = y1 - track_h
            ax.add_patch(
                plt.Rectangle(
                    (0.0, track_y0),
                    float(display_duration_s),
                    track_h,
                    facecolor=(1, 1, 1, 0.45),
                    edgecolor="none",
                    zorder=9,
                )
            )

            rel_seg_start = 0
            prev = int(display_labels[0])
            for rel_idx in range(1, int(display_labels.size) + 1):
                boundary = (rel_idx == int(display_labels.size)) or (int(display_labels[rel_idx]) != prev)
                if not boundary:
                    continue
                x0 = float(x_edges_s[rel_seg_start])
                x1 = float(x_edges_s[rel_idx])
                lab = int(prev)
                if x1 > x0:
                    ax.add_patch(
                        plt.Rectangle(
                            (x0, track_y0),
                            max(1e-6, x1 - x0),
                            track_h,
                            facecolor=_label_track_color(lab),
                            edgecolor="none",
                            alpha=0.35,
                            zorder=10,
                        )
                    )
                    ax.vlines(x0, track_y0, y1, colors=(0, 0, 0, 0.18), linewidth=0.35, zorder=11)
                    dur_s = max(0.0, x1 - x0)
                    if dur_s >= max(0.06, 6.0 * float(seconds_per_bin)):
                        txt = str(lab)
                        fs = 8 if dur_s >= 0.12 else 7
                        ax.text(
                            0.5 * (x0 + x1),
                            track_y0 + 0.5 * track_h,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=fs,
                            color="black",
                            zorder=12,
                            bbox=dict(
                                boxstyle="round,pad=0.12",
                                facecolor=(1, 1, 1, 0.55),
                                edgecolor="none",
                            ),
                        )
                if rel_idx < int(display_labels.size):
                    rel_seg_start = rel_idx
                    prev = int(display_labels[rel_idx])

            ax.hlines(track_y0, 0.0, float(display_duration_s), colors=(0, 0, 0, 0.35), linewidth=0.5, zorder=12)
            ax.text(
                float(display_duration_s) - 0.01 * float(display_duration_s),
                track_y0 + 0.5 * track_h,
                "all labels",
                ha="right",
                va="center",
                fontsize=7,
                color="black",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=(1, 1, 1, 0.50), edgecolor="none"),
            )

        ax.set_ylabel(
            f"run {int(row['run_id'])}\n{int(row['run_n_bins'])} bins\ncoverage={float(row['coverage_fraction']):.2f}",
            rotation=0,
            labelpad=58,
            va="center",
            fontsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Original recording context (s)")
    subtitle = (
        "Representative full HDBSCAN phrase-label runs; blue outline/bar = full run, red marks = equal UMAP-selected bins"
    )
    if bool(show_all_labels_track):
        subtitle += "; top strip = contiguous labels in displayed context"
    subtitle += f"; fixed x-axis 0–{float(last_display_duration_s):.1f} s"
    fig.suptitle(
        f"{animal_id} — label {int(cluster_id):02d} — {group_name}\n{subtitle}",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.12)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)






def _label_track_color(label: int):
    """Deterministic color for a syllable/cluster label in context plots."""
    if int(label) == -1:
        return (0.65, 0.65, 0.65, 1.0)
    cmap = plt.get_cmap("tab20")
    return cmap(int(label) % 20)


def _contiguous_label_segments(labels: np.ndarray, start_bin: int, end_bin_exclusive: int):
    """Return contiguous (start, end, label) segments in labels[start:end]."""
    start_bin = int(start_bin)
    end_bin_exclusive = int(end_bin_exclusive)
    if end_bin_exclusive <= start_bin:
        return []
    x = np.asarray(labels[start_bin:end_bin_exclusive]).astype(int)
    if x.size == 0:
        return []
    segs = []
    seg_start = start_bin
    prev = int(x[0])
    for offset, lab in enumerate(x[1:], start=1):
        lab = int(lab)
        if lab != prev:
            segs.append((seg_start, start_bin + offset, prev))
            seg_start = start_bin + offset
            prev = lab
    segs.append((seg_start, end_bin_exclusive, prev))
    return segs

def _compute_shared_intensity_limits(
    S_FxT: np.ndarray,
    selected_by_group: Mapping[str, np.ndarray],
    contrast_percentiles: Optional[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    arrays: List[np.ndarray] = []
    for idx in selected_by_group.values():
        idx = np.asarray(idx, dtype=int)
        if idx.size > 0:
            arrays.append(S_FxT[:, idx].astype(float).ravel())
    if not arrays:
        return None, None

    combined = np.concatenate(arrays)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return None, None

    if contrast_percentiles is None:
        return float(np.nanmin(combined)), float(np.nanmax(combined))

    lo, hi = contrast_percentiles
    vmin, vmax = np.nanpercentile(combined, [float(lo), float(hi)])
    return float(vmin), float(vmax)


def _compute_hist_density(
    xy: np.ndarray,
    idx: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a normalized 2D UMAP density histogram.

    If weights is supplied, it must have one value per selected time bin. This is
    useful for run-weighted full-contiguous-run BC: every bin in a run can be
    assigned weight 1/run_length, so every selected run contributes equal total
    mass even though long runs contain more frames.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != idx.shape[0]:
            raise ValueError(f"weights length {w.shape[0]} does not match idx length {idx.shape[0]}")
        good = np.isfinite(w) & (w >= 0)
        idx = idx[good]
        w = w[good]
        if idx.size == 0:
            return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    H, _, _ = np.histogram2d(xy[idx, 0], xy[idx, 1], bins=[x_edges, y_edges], weights=w)
    H = H.T.astype(float)
    total = H.sum()
    if total > 0:
        H /= total
    return H


def _density_to_rgb(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Map two normalized density maps to magenta/green overlap on black.
    Overlap appears white when both densities are present.
    """
    if pa.size == 0:
        return np.zeros((1, 1, 3), dtype=float)
    max_a = float(np.nanmax(pa)) if np.size(pa) else 0.0
    max_b = float(np.nanmax(pb)) if np.size(pb) else 0.0
    a = pa / max_a if max_a > 0 else np.zeros_like(pa, dtype=float)
    b = pb / max_b if max_b > 0 else np.zeros_like(pb, dtype=float)
    rgb = np.zeros(pa.shape + (3,), dtype=float)
    # A -> magenta, B -> green. Equal occupancy blends to white.
    rgb[..., 0] = np.clip(a, 0, 1)
    rgb[..., 1] = np.clip(b, 0, 1)
    rgb[..., 2] = np.clip(a, 0, 1)
    return np.clip(rgb, 0, 1)


def _hist_bc(pa: np.ndarray, pb: np.ndarray) -> float:
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    sa = pa.sum()
    sb = pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa = pa / sa
    pb = pb / sb
    return float(np.sum(np.sqrt(pa * pb)))



def _make_umap_density_edges(
    xy: np.ndarray,
    idx_arrays: Sequence[np.ndarray],
    *,
    density_bins: int,
    pad_fraction: float = 0.06,
    point_coverage: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float], Dict[str, float]]:
    """Build shared 2D histogram edges for UMAP BC calculations.

    By default, the grid spans all supplied selected points. If point_coverage is
    supplied as a value between 0 and 1, the grid is instead built from the
    central fraction of selected points closest to the robust median center.
    Points outside the grid are ignored by np.histogram2d and each histogram is
    renormalized. This is useful when a few outliers make the density grid mostly
    empty and visually/quantitatively dilute the cluster region.

    Example: point_coverage=0.99 keeps the nearest 99% of selected points for
    defining the grid and excludes the farthest 1% from the histogram support.
    """
    xy = np.asarray(xy, dtype=float)
    arrays = [np.asarray(idx, dtype=int).ravel() for idx in idx_arrays if np.asarray(idx).size > 0]
    if not arrays:
        raise ValueError("No selected UMAP indices were provided for density grid construction.")

    selected_all = np.concatenate(arrays).astype(int)
    selected_all = selected_all[(selected_all >= 0) & (selected_all < xy.shape[0])]
    if selected_all.size == 0:
        raise ValueError("No valid selected UMAP indices were provided for density grid construction.")

    pts = xy[selected_all]
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite]
    if pts.shape[0] == 0:
        raise ValueError("Selected UMAP points are all non-finite.")

    requested_coverage = np.nan if point_coverage is None else float(point_coverage)
    used_crop = False
    core = pts

    if point_coverage is not None and float(point_coverage) < 1.0:
        cov = float(point_coverage)
        if not (0.0 < cov <= 1.0):
            raise ValueError(f"point_coverage must be in (0, 1], got {point_coverage!r}")
        if pts.shape[0] >= 10:
            center = np.nanmedian(pts, axis=0)
            q25 = np.nanpercentile(pts, 25, axis=0)
            q75 = np.nanpercentile(pts, 75, axis=0)
            scale = q75 - q25
            fallback_scale = np.nanstd(pts, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback_scale)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
            robust_dist = np.sqrt(np.sum(((pts - center) / scale) ** 2, axis=1))
            cutoff = float(np.nanquantile(robust_dist, cov))
            keep = np.isfinite(robust_dist) & (robust_dist <= cutoff)
            if int(np.sum(keep)) >= 4:
                core = pts[keep]
                used_crop = True

    x_min = float(np.nanmin(core[:, 0]))
    x_max = float(np.nanmax(core[:, 0]))
    y_min = float(np.nanmin(core[:, 1]))
    y_max = float(np.nanmax(core[:, 1]))

    if not np.isfinite(x_min + x_max + y_min + y_max):
        raise ValueError("Could not construct finite UMAP density grid bounds.")

    # Avoid zero-width grids for very small/tight clusters.
    if abs(x_max - x_min) < 1e-12:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5

    pad_x = max(1e-6, (x_max - x_min) * float(pad_fraction))
    pad_y = max(1e-6, (y_max - y_min) * float(pad_fraction))
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, int(density_bins) + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, int(density_bins) + 1)

    in_grid = (
        (pts[:, 0] >= x_edges[0]) & (pts[:, 0] <= x_edges[-1]) &
        (pts[:, 1] >= y_edges[0]) & (pts[:, 1] <= y_edges[-1])
    )
    info = {
        "requested_point_coverage": requested_coverage,
        "actual_point_coverage": float(np.mean(in_grid)) if pts.shape[0] else np.nan,
        "n_points_for_grid": float(pts.shape[0]),
        "n_points_inside_grid": float(np.sum(in_grid)),
        "used_crop": float(bool(used_crop)),
    }
    return x_edges, y_edges, (float(x_edges[0]), float(x_edges[-1])), (float(y_edges[0]), float(y_edges[-1])), info

def build_full_contiguous_run_indices_from_phrase_selection_df(
    selection_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Expand the selected run-balanced phrase samples back to complete runs.

    This supports two sensitivity-analysis variants:
      1) full-contiguous-run BC, selected runs only: use all bins from each
         selected run with equal per-bin weights.
      2) run-weighted full-contiguous-run BC: use all bins from each selected
         run but weight each bin by 1/run_length so each run contributes equal
         total mass.

    The input should be the phrase-normalized selected-bin table produced by
    save_phrase_normalized_selected_bins_csv, which includes group, run_id,
    run_start_bin, and run_end_bin_exclusive.
    """
    if selection_df is None or selection_df.empty:
        return {}, {}, {}
    required = {"group", "run_id", "run_start_bin", "run_end_bin_exclusive"}
    missing = sorted(required.difference(selection_df.columns))
    if missing:
        raise KeyError(f"Cannot expand selected runs; missing columns: {missing}")

    idx_by_group: Dict[str, List[np.ndarray]] = {}
    weights_by_group: Dict[str, List[np.ndarray]] = {}
    n_runs_by_group: Dict[str, int] = {}

    run_df = (
        selection_df[["group", "run_id", "run_start_bin", "run_end_bin_exclusive"]]
        .drop_duplicates()
        .sort_values(["group", "run_start_bin", "run_end_bin_exclusive", "run_id"])
        .copy()
    )
    for group_name, sub in run_df.groupby("group", sort=False):
        group_key = str(group_name)
        idx_chunks: List[np.ndarray] = []
        weight_chunks: List[np.ndarray] = []
        for _, row in sub.iterrows():
            s = int(row["run_start_bin"])
            e = int(row["run_end_bin_exclusive"])
            if e <= s:
                continue
            bins = np.arange(s, e, dtype=int)
            idx_chunks.append(bins)
            # Equal total mass per run. Histograms are normalized afterward, so
            # the absolute scale is unimportant; the relative per-bin weights are.
            weight_chunks.append(np.full(bins.shape[0], 1.0 / float(bins.shape[0]), dtype=float))
        if idx_chunks:
            idx_by_group[group_key] = idx_chunks
            weights_by_group[group_key] = weight_chunks
            n_runs_by_group[group_key] = int(len(idx_chunks))

    idx_out = {g: np.concatenate(chunks).astype(int) for g, chunks in idx_by_group.items()}
    weights_out = {g: np.concatenate(chunks).astype(float) for g, chunks in weights_by_group.items()}
    return idx_out, weights_out, n_runs_by_group


def plot_umap_cluster_selected_timebins(
    *,
    embedding_xy: np.ndarray,
    labels: np.ndarray,
    selected_by_group: Mapping[str, np.ndarray],
    animal_id: str,
    cluster_id: int,
    out_png: Path,
    dpi: int,
    density_bins: int = 20,
    scatter_point_size: float = 6.0,
    analysis_label: str = "equal selected time-bin UMAPs",
    n_each_text: Optional[str] = None,
    weights_by_group: Optional[Mapping[str, np.ndarray]] = None,
    bc_grid_point_coverage: Optional[float] = None,
) -> None:
    """Save one BC-style UMAP summary plot for one cluster using exactly the selected time bins.
    For early_late_pre_post, this matches the BC-style format: four scatter panels and three density-overlap panels.
    Overlap density uses 2D histograms with `density_bins` bins per axis; white regions indicate overlap.
    """
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    labels = np.asarray(labels).astype(int)
    xy = np.asarray(embedding_xy, dtype=float)

    groups_present = [(name, np.asarray(idx, dtype=int)) for name, idx in selected_by_group.items() if len(idx) > 0]
    group_names = [name for name, _ in groups_present]
    if not groups_present:
        raise ValueError("No selected groups to plot on UMAP.")

    # Use the selected points for this cluster to define the plotting / density
    # coordinate frame. Earlier versions used the whole-bird UMAP extent for the
    # density histogram, which made the density panels look zoomed out relative
    # to the scatter panels and could waste most density bins on empty space.
    x_edges, y_edges, plot_xlim, plot_ylim, grid_info = _make_umap_density_edges(
        xy,
        [idx for _, idx in groups_present if np.asarray(idx).size > 0],
        density_bins=int(density_bins),
        pad_fraction=0.06,
        point_coverage=bc_grid_point_coverage,
    )
    grid_suffix = ""
    if bc_grid_point_coverage is not None and float(bc_grid_point_coverage) < 1.0:
        grid_suffix = f" | grid central coverage={float(bc_grid_point_coverage):.3f}; actual kept={grid_info['actual_point_coverage']:.3f}"

    colors = {
        "pre": "#8a2be2",
        "post": "#8a2be2",
        "early": "#8a2be2",
        "late": "#66cc66",
        "default": "#4c78a8",
    }

    def _group_color(name: str) -> str:
        n = str(name).lower()
        if "late" in n:
            return colors["late"]
        if "pre" in n or "post" in n or "early" in n:
            return colors["early"]
        return colors["default"]

    # Preferred BC-style layout for four groups.
    if set(group_names) == {"early_pre", "late_pre", "early_post", "late_post"}:
        order = ["early_pre", "late_pre", "early_post", "late_post"]
        idx_by = {k: np.asarray(selected_by_group[k], dtype=int) for k in order}
        counts = {k: int(len(v)) for k, v in idx_by.items()}
        pre_early_date = f"N={counts['early_pre']:,}"
        pre_late_date = f"N={counts['late_pre']:,}"
        post_early_date = f"N={counts['early_post']:,}"
        post_late_date = f"N={counts['late_post']:,}"

        weight_by = weights_by_group or {}

        def _weights_for(key: str) -> Optional[np.ndarray]:
            if key not in weight_by:
                return None
            return np.asarray(weight_by[key], dtype=float)

        def _hist_for(key: str) -> np.ndarray:
            return _compute_hist_density(
                xy,
                idx_by[key],
                x_edges=x_edges,
                y_edges=y_edges,
                weights=_weights_for(key),
            )

        def _concat_weights(keys: Sequence[str]) -> Optional[np.ndarray]:
            if not all(k in weight_by for k in keys):
                return None
            return np.concatenate([np.asarray(weight_by[k], dtype=float) for k in keys])

        hist_pre = _hist_for('early_pre')
        hist_lpre = _hist_for('late_pre')
        hist_epost = _hist_for('early_post')
        hist_lpost = _hist_for('late_post')
        bc_pre = _hist_bc(hist_pre, hist_lpre)
        bc_post = _hist_bc(hist_epost, hist_lpost)
        hist_pre_all = _compute_hist_density(
            xy,
            np.concatenate([idx_by['early_pre'], idx_by['late_pre']]),
            x_edges=x_edges,
            y_edges=y_edges,
            weights=_concat_weights(['early_pre', 'late_pre']),
        )
        hist_post_all = _compute_hist_density(
            xy,
            np.concatenate([idx_by['early_post'], idx_by['late_post']]),
            x_edges=x_edges,
            y_edges=y_edges,
            weights=_concat_weights(['early_post', 'late_post']),
        )
        bc_prepost = _hist_bc(hist_pre_all, hist_post_all)

        fig = plt.figure(figsize=(16.5, 9.4))
        gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 0.9, 1.1], wspace=0.12, hspace=0.30)
        ax_ep = fig.add_subplot(gs[0, 0])
        ax_lp = fig.add_subplot(gs[0, 1], sharex=ax_ep, sharey=ax_ep)
        ax_pre_ov = fig.add_subplot(gs[0, 2])
        ax_eo = fig.add_subplot(gs[1, 0], sharex=ax_ep, sharey=ax_ep)
        ax_lo = fig.add_subplot(gs[1, 1], sharex=ax_ep, sharey=ax_ep)
        ax_post_ov = fig.add_subplot(gs[1, 2])
        ax_prepost_ov = fig.add_subplot(gs[:, 3])

        scatter_specs = [
            (ax_ep, 'early_pre', 'Pre-lesion early', _group_color('early_pre')),
            (ax_lp, 'late_pre', 'Pre-lesion late', _group_color('late_pre')),
            (ax_eo, 'early_post', 'Post-lesion early', _group_color('early_post')),
            (ax_lo, 'late_post', 'Post-lesion late', _group_color('late_post')),
        ]
        for ax, key, title, color in scatter_specs:
            idx = idx_by[key]
            ax.scatter(xy[idx, 0], xy[idx, 1], s=scatter_point_size, c=color, alpha=0.70, linewidths=0)
            ax.set_title(f"{title}", color=color, fontsize=13, pad=10)
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        def _draw_overlap(ax, hist_a, hist_b, title_text):
            ax.imshow(
                _density_to_rgb(hist_a, hist_b),
                origin='lower',
                extent=[plot_xlim[0], plot_xlim[1], plot_ylim[0], plot_ylim[1]],
                aspect='equal',
                interpolation='nearest',
            )
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(title_text, fontsize=12, pad=10)
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        _draw_overlap(ax_pre_ov, hist_pre, hist_lpre, f"Pre early vs late overlap\nBC={bc_pre:.3f}")
        _draw_overlap(ax_post_ov, hist_epost, hist_lpost, f"Post early vs late overlap\nBC={bc_post:.3f}")
        _draw_overlap(
            ax_prepost_ov,
            hist_pre_all,
            hist_post_all,
            f"Pre vs post overlap\npre N={len(idx_by['early_pre']) + len(idx_by['late_pre']):,} | "
            f"post N={len(idx_by['early_post']) + len(idx_by['late_post']):,}\nBC={bc_prepost:.3f}",
        )

        if n_each_text is None:
            n_each_text = f"n_each={min(counts.values()):,}"
        fig.suptitle(
            f"{animal_id} cluster {cluster_id}: {analysis_label}\n"
            f"{n_each_text} | pre BC={bc_pre:.3f} | post BC={bc_post:.3f} | pre/post BC={bc_prepost:.3f} | density bins={int(density_bins)}{grid_suffix}",
            y=0.985, fontsize=15
        )
        # Reserve extra room above the panel grid so the figure-level title
        # does not overlap with the subpanel titles.
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(out_png, dpi=int(dpi), bbox_inches='tight')
        plt.close(fig)
        return

    # Simpler fallback for pre_post or other group sets.
    ncols = len(groups_present) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 5.2))
    if ncols == 1:
        axes = [axes]
    for ax, (group_name, idx) in zip(axes[:-1], groups_present):
        color = _group_color(group_name)
        ax.scatter(xy[idx, 0], xy[idx, 1], s=scatter_point_size, c=color, alpha=0.70, linewidths=0)
        ax.set_title(f"{group_name}", color=color)
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if len(groups_present) >= 2:
        g1, i1 = groups_present[0]
        g2, i2 = groups_present[1]
        h1 = _compute_hist_density(xy, i1, x_edges=x_edges, y_edges=y_edges)
        h2 = _compute_hist_density(xy, i2, x_edges=x_edges, y_edges=y_edges)
        bc = _hist_bc(h1, h2)
        ax = axes[-1]
        ax.imshow(_density_to_rgb(h1, h2), origin='lower', extent=[plot_xlim[0], plot_xlim[1], plot_ylim[0], plot_ylim[1]], aspect='equal', interpolation='nearest')
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{g1} vs {g2} overlap\nBC={bc:.3f} | bins={int(density_bins)}{grid_suffix}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{animal_id} — HDBSCAN label {cluster_id} — {analysis_label}", y=0.985, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_png, dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)


def plot_spectrogram_rows_for_timebins(
    *,
    S_FxT: np.ndarray,
    selected_indices: np.ndarray,
    animal_id: str,
    cluster_id: int,
    group_name: str,
    out_png: Path,
    bins_per_row: int,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    subplot_hspace: float,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    max_rows_per_spectrogram: Optional[int] = None,
    show_row_ylabels: bool = True,
) -> None:
    """
    Save one multi-row spectrogram figure for a group of selected time bins.

    Each row is up to bins_per_row selected time bins. The last row may be shorter;
    it is shown only over its true duration while the x-axis remains fixed to the
    full bins_per_row duration.
    """
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)

    selected_indices = np.asarray(selected_indices, dtype=int)
    chunks = _chunk_indices(selected_indices, bins_per_row=bins_per_row, max_rows=max_rows_per_spectrogram)
    if not chunks:
        raise ValueError(f"No selected indices to plot for {group_name}.")

    n_rows = len(chunks)
    max_time_s = int(bins_per_row) * float(seconds_per_bin)

    fig_h = max(2.0, float(row_height) * n_rows + 0.7)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(float(figure_width), fig_h),
        sharex=True,
        gridspec_kw={"hspace": max(0.04, float(subplot_hspace))},
    )
    if n_rows == 1:
        axes = [axes]

    for row_i, (ax, chunk) in enumerate(zip(axes, chunks), start=1):
        S_panel = S_FxT[:, chunk].astype(float)
        actual_bins = int(S_panel.shape[1])
        actual_time_s = actual_bins * float(seconds_per_bin)

        ax.imshow(
            S_panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, actual_time_s, 0, S_panel.shape[0]],
        )
        ax.set_xlim(0, max_time_s)
        ax.set_facecolor("white")
        _hide_spectrogram_ticks(ax)
        ax.set_ylabel(
            f"{group_name}\nrow {row_i}\n{actual_bins:,} bins",
            rotation=0,
            labelpad=48,
            va="center",
            fontsize=10,
        )

    xticks = np.arange(0, max_time_s + 1e-9, float(x_tick_step_s))
    axes[-1].tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{x:g}" for x in xticks])
    axes[-1].set_xlabel("Stitched selected time bins (s)")

    shown_bins = int(sum(ch.size for ch in chunks))
    truncated_note = "" if shown_bins == selected_indices.size else f" | showing {shown_bins:,}/{selected_indices.size:,} bins"
    fig.suptitle(
        f"{animal_id} — HDBSCAN label {cluster_id} — {group_name}\n"
        f"Exact selected UMAP time bins{truncated_note}; {n_rows} row(s), up to {int(bins_per_row):,} bins each",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.08)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_selected_timebins_csv(
    *,
    selected_by_group: Mapping[str, np.ndarray],
    out_csv: Path,
    cluster_id: int,
    file_indices: Optional[np.ndarray],
    timebin_dates: Optional[np.ndarray],
) -> None:
    rows: List[Dict[str, Any]] = []
    for group_name, idx in selected_by_group.items():
        idx = np.asarray(idx, dtype=int)
        for within_group_order, timebin_index in enumerate(idx):
            row: Dict[str, Any] = {
                "cluster_id": int(cluster_id),
                "group": group_name,
                "within_group_order": int(within_group_order),
                "timebin_index": int(timebin_index),
            }
            if file_indices is not None:
                row["file_index"] = file_indices[timebin_index]
            if timebin_dates is not None:
                row["recording_date"] = str(timebin_dates[timebin_index])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Main export function
# -----------------------------------------------------------------------------

def export_equal_umap_cluster_spectrograms_for_one_bird(
    *,
    npz_path: Path,
    metadata_excel_path: Path,
    spectrogram_script: Path,
    out_dir: Path,
    animal_id: Optional[str] = None,
    embedding_key: str = "embedding_outputs",
    cluster_key: str = "hdbscan_labels",
    spectrogram_key: str = "s",
    file_key: str = "file_indices",
    vocalization_key: str = "vocalization",
    include_noise: bool = False,
    only_cluster_ids: Optional[Sequence[int]] = None,
    vocalization_only: bool = True,
    period_mode: str = "pre_post",
    treatment_day_assignment: str = "exclude",
    early_late_split_method: str = "file_median",
    bc_analysis_mode: str = "run_balanced_phrase_normalized",
    min_points_per_group: int = 6000,
    max_points_per_group: Optional[int] = 6000,
    sample_mode: str = "random",
    min_runs_per_group: int = 20,
    max_runs_per_group: Optional[int] = 200,
    phase_bins_per_run: int = 25,
    min_full_run_duration_ms: float = 100.0,
    min_run_group_fraction: float = 0.80,
    run_sample_mode: str = "random",
    allow_repeat_phase_bins: bool = False,
    save_phase_normalized_bc: bool = True,
    save_full_run_bc_variants: bool = False,
    seed: int = 0,
    bins_per_spectrogram_row: int = 2000,
    max_rows_per_spectrogram: Optional[int] = None,
    spectrogram_source_mode: str = "expanded_full_runs",
    save_representative_full_run_contexts: bool = True,
    max_expanded_full_run_bins: Optional[int] = 10000,
    full_run_context_bins: int = 80,
    full_run_max_context_bins: int = 1500,
    full_run_fixed_duration_s: float = 3.0,
    full_run_max_examples: int = 12,
    full_run_example_mode: str = "mixed",
    seconds_per_bin: float = 0.0027,
    x_tick_step_s: float = 1.0,
    figure_width: float = 24.0,
    row_height: float = 2.2,
    subplot_hspace: float = 0.005,
    cmap: str = "gray_r",
    contrast_percentiles: Optional[Tuple[float, float]] = (1, 99.5),
    dpi: int = 200,
    umap_density_bins: int = 20,
    bc_grid_point_coverage: Optional[float] = None,
    fill_noise_labels_from_nearest_nonnoise: bool = False,
    noise_label: int = -1,
    smooth_short_label_interruptions: bool = False,
    max_label_interruption_ms: float = 50.0,
    smoothing_max_iterations: int = 3,
    apply_majority_vote_label_smoothing: bool = False,
    majority_vote_window_bins: int = 200,
    majority_vote_by_file: bool = True,
    dry_run: bool = False,
    top_variance_csv: Optional[Path] = None,
    top_fraction: float = 0.30,
    post_group_name: str = "Post",
    top_min_n_phrases: int = 100,
) -> Path:
    """
    Generate one UMAP plot and matching spectrogram figures for each cluster.

    By default, the UMAP/BC plot is run-balanced and phrase-normalized:
    each group contributes the same number of full phrase-label runs, and each
    run contributes the same number of normalized phase samples. This avoids
    frame-count dominance by long post-lesion stutter bouts.

    Set bc_analysis_mode='frame_weighted' to reproduce the older equal-time-bin
    BC behavior. Spectrograms can either show exact selected bins or expand
    them to the full contiguous HDBSCAN phrase-label runs that contain them.
    """
    npz_path = Path(npz_path).expanduser().resolve()
    metadata_excel_path = Path(metadata_excel_path).expanduser().resolve()
    spectrogram_script = Path(spectrogram_script).expanduser().resolve()

    # Treat --out-dir as an OUTPUT ROOT, not as an animal-specific folder.
    # This creates the requested flat structure:
    #   <out_dir>/<animal_id>/USA5443_label02_UMAP_equal_timebins.png
    #   <out_dir>/<animal_id>/USA5443_label02_early_pre_spectrogram_....png
    out_root = Path(out_dir).expanduser().resolve()
    _safe_mkdir(out_root)

    helper = _load_module_from_path(spectrogram_script)
    arr = np.load(npz_path, allow_pickle=True)

    if animal_id is None:
        animal_id = npz_path.stem.split("_")[0]
    animal_id = str(animal_id)
    animal_out_dir = out_root / _clean_token(animal_id)
    _safe_mkdir(animal_out_dir)

    labels_raw = np.asarray(arr[cluster_key]).astype(int)
    file_indices_for_smoothing = _get_optional_array(arr, file_key, labels_raw.shape[0])
    max_interruption_bins_for_info = (
        int(round(float(max_label_interruption_ms) / float(seconds_per_bin)))
        if float(seconds_per_bin) > 0
        else 0
    )
    label_cleanup_info: Dict[str, Any] = {
        "noise_label": int(noise_label),
        "n_noise_labels_before_fill": int(np.sum(labels_raw == int(noise_label))),
        "n_noise_labels_after_fill": int(np.sum(labels_raw == int(noise_label))),
        "n_noise_labels_replaced": 0,
        "decoder_style_noise_fill_applied": False,
        "decoder_style_noise_fill_all_noise_warning": False,
        "global_short_interruption_smoothing_requested": bool(smooth_short_label_interruptions),
        "global_short_interruption_max_ms": float(max_label_interruption_ms),
        "global_short_interruption_max_bins": int(max_interruption_bins_for_info),
        "global_short_interruption_smoothing_applied": False,
        "global_short_interruption_iterations_requested": int(smoothing_max_iterations),
        "global_short_interruption_iterations_used": 0,
        "global_short_interruption_runs_replaced": 0,
        "global_short_interruption_bins_replaced": 0,
        "global_short_interruption_examples": "",
    }
    if bool(fill_noise_labels_from_nearest_nonnoise):
        labels, noise_info = _fill_noise_labels_decoder_style(labels_raw, noise_label=int(noise_label))
        label_cleanup_info.update(noise_info)
    else:
        labels = labels_raw.copy()

    if bool(smooth_short_label_interruptions):
        max_interruption_bins = int(max_interruption_bins_for_info)
        labels, smooth_info = _smooth_short_label_interruptions_global(
            labels,
            max_interruption_bins=max_interruption_bins,
            max_iterations=int(smoothing_max_iterations),
        )
        label_cleanup_info.update({
            "global_short_interruption_smoothing_requested": True,
            "global_short_interruption_max_ms": float(max_label_interruption_ms),
        })
        label_cleanup_info.update(smooth_info)

    if bool(apply_majority_vote_label_smoothing):
        labels, majority_info = _apply_majority_vote_label_smoothing(
            labels,
            window_bins=int(majority_vote_window_bins),
            sequence_ids=file_indices_for_smoothing,
            smooth_by_sequence=bool(majority_vote_by_file),
        )
        label_cleanup_info.update({
            "majority_vote_label_smoothing_requested": True,
            "majority_vote_window_bins": int(majority_vote_window_bins),
            "majority_vote_window_ms": float(majority_vote_window_bins) * float(seconds_per_bin) * 1000.0,
        })
        label_cleanup_info.update(majority_info)
    embedding_xy = _get_embedding_xy(arr, embedding_key)
    if embedding_xy.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Embedding length {embedding_xy.shape[0]} does not match label length {labels.shape[0]}."
        )

    S_FxT = helper._orient_spectrogram_to_FxT(np.asarray(arr[spectrogram_key]), labels)
    if S_FxT.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Oriented spectrogram has {S_FxT.shape[1]} time bins, but labels have {labels.shape[0]}."
        )

    file_indices = file_indices_for_smoothing if file_indices_for_smoothing is not None else _get_optional_array(arr, file_key, labels.shape[0])
    vocalization = _get_optional_array(arr, vocalization_key, labels.shape[0])
    vocalization_mask = None
    if vocalization_only and vocalization is not None:
        vocalization_mask = np.asarray(vocalization).astype(bool)

    treatment_date = helper.get_treatment_date_from_metadata(
        metadata_excel_path=metadata_excel_path,
        animal_id=animal_id,
    )
    timebin_dates = helper.get_timebin_recording_dates(arr, expected_length=labels.shape[0], date_array_key=None)
    pre_mask, post_mask = helper.make_pre_post_masks(
        timebin_dates,
        treatment_date,
        treatment_day_assignment=treatment_day_assignment,
    )
    period_masks = build_period_masks(
        pre_mask=pre_mask,
        post_mask=post_mask,
        file_indices=file_indices,
        period_mode=period_mode,
        early_late_split_method=early_late_split_method,
    )

    if only_cluster_ids is not None:
        cluster_ids = [int(x) for x in only_cluster_ids]
    else:
        cluster_ids = sorted(int(x) for x in np.unique(labels))
        if not include_noise:
            cluster_ids = [x for x in cluster_ids if x >= 0]

    top_variance_cluster_ids: Optional[List[int]] = None
    if top_variance_csv is not None:
        top_variance_cluster_ids = read_top_variance_cluster_ids(
            top_variance_csv,
            animal_id=animal_id,
            top_fraction=float(top_fraction),
            post_group_name=str(post_group_name),
            min_n_phrases=int(top_min_n_phrases),
        )
        top_set = set(int(x) for x in top_variance_cluster_ids)
        before_n = len(cluster_ids)
        cluster_ids = [x for x in cluster_ids if int(x) in top_set]
        print(
            f"[INFO] Top-variance CSV: {Path(top_variance_csv).expanduser().resolve()}"
        )
        print(
            f"[INFO] Top-variance labels for {animal_id}: "
            f"{', '.join(_cluster_token(x) for x in top_variance_cluster_ids)}"
        )
        print(
            f"[INFO] Filtering clusters by top variance: {before_n} available -> {len(cluster_ids)} to evaluate"
        )

    print("=" * 92)
    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print(f"[INFO] Animal: {animal_id}")
    print(f"[INFO] NPZ: {npz_path}")
    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Animal output: {animal_out_dir}")
    print("[INFO] Output layout: flat files directly in the animal folder; no per-label subfolders")
    print("[INFO] Filename order within each label: 01 early_pre, 02 late_pre, 03 early_post, 04 late_post, 05 UMAP, 99 selected CSV")
    print("[INFO] UMAP scatter panels omit axis-label text and omit per-panel N from subplot titles.")
    print(f"[INFO] BC analysis mode: {bc_analysis_mode}")
    if bool(fill_noise_labels_from_nearest_nonnoise):
        print(
            f"[INFO] Decoder-style noise fill applied to labels: "
            f"noise_label={int(noise_label)}, "
            f"before={label_cleanup_info.get('n_noise_labels_before_fill', 0):,}, "
            f"after={label_cleanup_info.get('n_noise_labels_after_fill', 0):,}, "
            f"replaced={label_cleanup_info.get('n_noise_labels_replaced', 0):,}"
        )
        if bool(label_cleanup_info.get('decoder_style_noise_fill_all_noise_warning', False)):
            print("[WARN] All labels were noise; decoder-style fill could not replace -1 labels.")
    else:
        print(
            f"[INFO] Decoder-style noise fill is OFF; "
            f"noise_label={int(noise_label)}, n_noise_labels={label_cleanup_info.get('n_noise_labels_before_fill', 0):,}"
        )
    if bool(smooth_short_label_interruptions):
        print(
            f"[INFO] Global short-interruption smoothing is ON: "
            f"max_gap={float(max_label_interruption_ms):.1f} ms "
            f"({label_cleanup_info.get('global_short_interruption_max_bins', 0):,} bins), "
            f"iterations={label_cleanup_info.get('global_short_interruption_iterations_used', 0):,}, "
            f"runs bridged={label_cleanup_info.get('global_short_interruption_runs_replaced', 0):,}, "
            f"bins replaced={label_cleanup_info.get('global_short_interruption_bins_replaced', 0):,}"
        )
    else:
        print("[INFO] Global short-interruption smoothing is OFF.")
    if bool(apply_majority_vote_label_smoothing):
        print(
            f"[INFO] Majority-vote temporal smoothing is ON: "
            f"window={int(majority_vote_window_bins):,} bins "
            f"({label_cleanup_info.get('majority_vote_window_ms', 0):.1f} ms), "
            f"by_file={bool(label_cleanup_info.get('majority_vote_by_sequence', False))}, "
            f"sequences={label_cleanup_info.get('majority_vote_sequences_smoothed', 0):,}, "
            f"bins changed={label_cleanup_info.get('majority_vote_bins_changed', 0):,}"
        )
    else:
        print("[INFO] Majority-vote temporal smoothing is OFF.")

    if str(bc_analysis_mode) == "run_balanced_phrase_normalized":
        print(
            f"[INFO] Run-balanced/phrase-normalized settings: min_runs/group={min_runs_per_group}, "
            f"max_runs/group={max_runs_per_group}, phase_bins/run={phase_bins_per_run}, "
            f"min_full_run_duration_ms={min_full_run_duration_ms}"
        )
        print("[INFO] UMAP/BC uses equal numbers of full phrase-label runs and equal normalized phase samples per run.")
        if bool(save_full_run_bc_variants):
            print("[INFO] Also saving full-contiguous-run and run-weighted full-contiguous-run BC sensitivity plots.")
    elif str(bc_analysis_mode) == "run_balanced_full_contiguous":
        print(
            f"[INFO] Run-balanced/full-contiguous settings: min_runs/group={min_runs_per_group}, "
            f"max_runs/group={max_runs_per_group}, min_full_run_duration_ms={min_full_run_duration_ms}"
        )
        print("[INFO] UMAP/BC uses equal numbers of full phrase-label runs per group and ALL bins within those selected runs.")
        print("[INFO] No within-run phase subsampling is applied in this mode; red QC marks should tile selected blue runs.")
    else:
        print("[INFO] UMAP/BC uses frame-weighted equal selected time bins. Use --sample-mode time_balanced to sample evenly across the available cluster-time axis.")
    print(f"[INFO] Spectrogram source mode: {spectrogram_source_mode}")
    print("[INFO] Expanded spectrograms restore full phrase-label runs for visualization/QC.")
    print(f"[INFO] Treatment date: {treatment_date}")
    print(f"[INFO] Period mode: {period_mode}")
    print(f"[INFO] Groups: {list(period_masks.keys())}")
    print(f"[INFO] Clusters to evaluate: {len(cluster_ids)}")
    print("=" * 92)

    if not cluster_ids:
        raise ValueError(
            "No clusters are left to evaluate after filtering. Check --phrase-csv/--top30-csv, "
            "--top-fraction, --top-min-n-phrases, and --animal-id."
        )

    rng = np.random.default_rng(int(seed))
    summary_rows: List[Dict[str, Any]] = []

    for cluster_id in cluster_ids:
        cluster_name = _cluster_token(cluster_id)
        cluster_out_dir = animal_out_dir
        run_id_for_bin, run_starts, run_ends = build_label_run_lookup(labels, cluster_id)

        # Make a cluster-specific RNG while remaining reproducible across runs.
        cluster_seed = int(rng.integers(0, 2**31 - 1))
        cluster_rng = np.random.default_rng(cluster_seed)

        mode_bc = str(bc_analysis_mode).strip().lower()
        phase_indices_by_group: Dict[str, np.ndarray] = {}
        phrase_selection_df = pd.DataFrame()
        n_equal_runs = 0

        if mode_bc == "frame_weighted":
            selected_by_group, available_counts, n_equal = select_equal_timebins_for_cluster(
                labels=labels,
                cluster_id=cluster_id,
                period_masks=period_masks,
                min_points_per_group=int(min_points_per_group),
                max_points_per_group=max_points_per_group,
                sample_mode=sample_mode,
                rng=cluster_rng,
                vocalization_mask=vocalization_mask,
            )
            row_base: Dict[str, Any] = {
                "animal_id": animal_id,
                "cluster_id": int(cluster_id),
                "cluster_token": cluster_name,
                "bc_analysis_mode": mode_bc,
                "status": "skipped" if selected_by_group is None else "done",
                "n_equal_selected_per_group": int(n_equal),
                "n_equal_runs_per_group": np.nan,
                "phase_bins_per_run": np.nan,
                "min_points_per_group": int(min_points_per_group),
                "max_points_per_group": max_points_per_group,
                "sample_mode": sample_mode,
                "cluster_seed": cluster_seed,
            }
            for group_name, count in available_counts.items():
                row_base[f"available_{group_name}_bins"] = int(count)
            if selected_by_group is None:
                print(
                    f"[SKIP] {animal_id} {cluster_name}: frame-weighted n_equal={n_equal:,} < "
                    f"min_points_per_group={int(min_points_per_group):,}; counts={available_counts}"
                )
                summary_rows.append(row_base)
                continue
            print(
                f"[RUN] {animal_id} {cluster_name}: frame-weighted selected N={n_equal:,} bins/group; "
                f"available bins={available_counts}"
            )
            umap_analysis_label = "frame-weighted equal time-bin UMAPs"
            umap_n_each_text = f"n_each={int(n_equal):,} time bins"
        elif mode_bc == "run_balanced_phrase_normalized":
            selected_by_group, available_counts, n_equal_runs, phase_indices_by_group, phrase_selection_df = select_run_balanced_phrase_normalized_bins_for_cluster(
                labels=labels,
                cluster_id=cluster_id,
                run_starts=run_starts,
                run_ends=run_ends,
                period_masks=period_masks,
                min_runs_per_group=int(min_runs_per_group),
                max_runs_per_group=max_runs_per_group,
                phase_bins_per_run=int(phase_bins_per_run),
                min_full_run_duration_ms=float(min_full_run_duration_ms),
                seconds_per_bin=float(seconds_per_bin),
                min_run_group_fraction=float(min_run_group_fraction),
                run_sample_mode=str(run_sample_mode),
                rng=cluster_rng,
                vocalization_mask=vocalization_mask,
                allow_repeat_phase_bins=bool(allow_repeat_phase_bins),
            )
            n_equal_points = int(n_equal_runs) * int(phase_bins_per_run)
            row_base = {
                "animal_id": animal_id,
                "cluster_id": int(cluster_id),
                "cluster_token": cluster_name,
                "bc_analysis_mode": mode_bc,
                "status": "skipped" if selected_by_group is None else "done",
                "n_equal_selected_per_group": int(n_equal_points),
                "n_equal_runs_per_group": int(n_equal_runs),
                "phase_bins_per_run": int(phase_bins_per_run),
                "min_runs_per_group": int(min_runs_per_group),
                "max_runs_per_group": max_runs_per_group,
                "min_full_run_duration_ms": float(min_full_run_duration_ms),
                "run_sample_mode": str(run_sample_mode),
                "cluster_seed": cluster_seed,
            }
            for group_name, count in available_counts.items():
                row_base[f"available_{group_name}_full_runs"] = int(count)
            if selected_by_group is None:
                print(
                    f"[SKIP] {animal_id} {cluster_name}: run-balanced n_equal_runs={n_equal_runs:,} < "
                    f"min_runs_per_group={int(min_runs_per_group):,}; candidate full-run counts={available_counts}"
                )
                summary_rows.append(row_base)
                continue
            print(
                f"[RUN] {animal_id} {cluster_name}: run-balanced selected {n_equal_runs:,} full runs/group "
                f"x {int(phase_bins_per_run):,} phase bins/run = {n_equal_points:,} points/group; "
                f"candidate full runs={available_counts}"
            )
            umap_analysis_label = "run-balanced phrase-normalized UMAPs"
            umap_n_each_text = f"n_runs/group={int(n_equal_runs):,}; phase bins/run={int(phase_bins_per_run):,}; points/group={n_equal_points:,}"
        elif mode_bc == "run_balanced_full_contiguous":
            selected_by_group, available_counts, n_equal_runs, phrase_selection_df, selected_bin_counts = select_run_balanced_full_contiguous_bins_for_cluster(
                labels=labels,
                cluster_id=cluster_id,
                run_starts=run_starts,
                run_ends=run_ends,
                period_masks=period_masks,
                min_runs_per_group=int(min_runs_per_group),
                max_runs_per_group=max_runs_per_group,
                min_full_run_duration_ms=float(min_full_run_duration_ms),
                seconds_per_bin=float(seconds_per_bin),
                min_run_group_fraction=float(min_run_group_fraction),
                run_sample_mode=str(run_sample_mode),
                rng=cluster_rng,
                vocalization_mask=vocalization_mask,
            )
            n_min_selected_bins = int(min(selected_bin_counts.values())) if selected_bin_counts else 0
            row_base = {
                "animal_id": animal_id,
                "cluster_id": int(cluster_id),
                "cluster_token": cluster_name,
                "bc_analysis_mode": mode_bc,
                "status": "skipped" if selected_by_group is None else "done",
                "n_equal_selected_per_group": int(n_min_selected_bins),
                "n_equal_runs_per_group": int(n_equal_runs),
                "phase_bins_per_run": np.nan,
                "min_runs_per_group": int(min_runs_per_group),
                "max_runs_per_group": max_runs_per_group,
                "min_full_run_duration_ms": float(min_full_run_duration_ms),
                "run_sample_mode": str(run_sample_mode),
                "cluster_seed": cluster_seed,
                "selected_bin_count_summary": ";".join(f"{k}:{v}" for k, v in sorted(selected_bin_counts.items())),
            }
            for group_name, count in available_counts.items():
                row_base[f"available_{group_name}_full_runs"] = int(count)
            for group_name, count in selected_bin_counts.items():
                row_base[f"selected_{group_name}_full_contiguous_bins"] = int(count)
            if selected_by_group is None:
                print(
                    f"[SKIP] {animal_id} {cluster_name}: full-contiguous n_equal_runs={n_equal_runs:,} < "
                    f"min_runs_per_group={int(min_runs_per_group):,}; candidate full-run counts={available_counts}"
                )
                summary_rows.append(row_base)
                continue
            counts_text = "; ".join(f"{k}: {v:,} bins" for k, v in sorted(selected_bin_counts.items()))
            print(
                f"[RUN] {animal_id} {cluster_name}: run-balanced full-contiguous selected {n_equal_runs:,} full runs/group; "
                f"selected bins after expansion=({counts_text}); candidate full runs={available_counts}"
            )
            umap_analysis_label = "run-balanced full-contiguous-run UMAPs"
            umap_n_each_text = f"n_runs/group={int(n_equal_runs):,}; all bins in selected runs; {counts_text}"
        else:
            raise ValueError("bc_analysis_mode must be 'frame_weighted', 'run_balanced_phrase_normalized', or 'run_balanced_full_contiguous'")
        row_base.update(label_cleanup_info)
        cleanup_modes: List[str] = []
        if bool(fill_noise_labels_from_nearest_nonnoise):
            cleanup_modes.append("decoder_style_nearest_nonnoise")
        if bool(smooth_short_label_interruptions):
            cleanup_modes.append(f"global_short_interruptions_le_{float(max_label_interruption_ms):g}ms")
        if bool(apply_majority_vote_label_smoothing):
            cleanup_modes.append(f"majority_vote_w{int(majority_vote_window_bins)}bins")
        row_base["label_cleanup_mode"] = "+".join(cleanup_modes) if cleanup_modes else "none"

        if dry_run:
            row_base["status"] = "dry_run"
            summary_rows.append(row_base)
            continue

        vmin, vmax = _compute_shared_intensity_limits(
            S_FxT,
            selected_by_group,
            contrast_percentiles=contrast_percentiles,
        )

        ordered_prefix = {
            "early_pre": "01",
            "late_pre": "02",
            "early_post": "03",
            "late_post": "04",
        }
        if mode_bc == "run_balanced_phrase_normalized":
            umap_png = cluster_out_dir / f"{animal_id}_{cluster_name}_05_UMAP_runbalanced_phrase_normalized.png"
            selected_csv = cluster_out_dir / f"{animal_id}_{cluster_name}_99_runbalanced_phrase_normalized_selected_bins.csv"
            save_phrase_normalized_selected_bins_csv(
                selection_df=phrase_selection_df,
                out_csv=selected_csv,
                file_indices=file_indices,
                timebin_dates=timebin_dates,
            )
        elif mode_bc == "run_balanced_full_contiguous":
            umap_png = cluster_out_dir / f"{animal_id}_{cluster_name}_05_UMAP_runbalanced_full_contiguous.png"
            selected_csv = cluster_out_dir / f"{animal_id}_{cluster_name}_99_runbalanced_full_contiguous_selected_bins.csv"
            save_phrase_normalized_selected_bins_csv(
                selection_df=phrase_selection_df,
                out_csv=selected_csv,
                file_indices=file_indices,
                timebin_dates=timebin_dates,
            )
        else:
            umap_png = cluster_out_dir / f"{animal_id}_{cluster_name}_05_UMAP_frame_weighted_equal_timebins.png"
            selected_csv = cluster_out_dir / f"{animal_id}_{cluster_name}_99_frame_weighted_selected_equal_timebins.csv"
            save_selected_timebins_csv(
                selected_by_group=selected_by_group,
                out_csv=selected_csv,
                cluster_id=cluster_id,
                file_indices=file_indices,
                timebin_dates=timebin_dates,
            )

        plot_umap_cluster_selected_timebins(
            embedding_xy=embedding_xy,
            labels=labels,
            selected_by_group=selected_by_group,
            animal_id=animal_id,
            cluster_id=cluster_id,
            out_png=umap_png,
            dpi=int(dpi),
            density_bins=int(umap_density_bins),
            analysis_label=umap_analysis_label,
            n_each_text=umap_n_each_text,
            bc_grid_point_coverage=bc_grid_point_coverage,
        )
        print(f"[SAVE] {umap_png}")
        print(f"[SAVE] {selected_csv}")

        phase_bc_csv = ""
        phase_bc_png = ""
        phase_colored_umap_png = ""
        full_contiguous_umap_png = ""
        run_weighted_full_contiguous_umap_png = ""
        if mode_bc == "run_balanced_phrase_normalized" and bool(save_phase_normalized_bc):
            try:
                phase_bc_df = compute_phrase_normalized_bc_by_phase(
                    embedding_xy=embedding_xy,
                    phase_indices_by_group=phase_indices_by_group,
                    density_bins=int(umap_density_bins),
                )
                phase_bc_csv_path = cluster_out_dir / f"{animal_id}_{cluster_name}_06_phrase_normalized_BC_by_phase.csv"
                phase_bc_png_path = cluster_out_dir / f"{animal_id}_{cluster_name}_06_phrase_normalized_BC_by_phase.png"
                phase_bc_df.to_csv(phase_bc_csv_path, index=False)
                if not phase_bc_df.empty:
                    plot_phrase_normalized_bc_by_phase(
                        phase_bc_df=phase_bc_df,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        out_png=phase_bc_png_path,
                        dpi=int(dpi),
                    )
                phase_bc_csv = str(phase_bc_csv_path)
                phase_bc_png = str(phase_bc_png_path)
                print(f"[SAVE] {phase_bc_csv_path}")
                print(f"[SAVE] {phase_bc_png_path}")
            except Exception as e:
                print(f"[ERROR] Could not save phrase-normalized BC-by-phase outputs for {animal_id} {cluster_name}: {e}")
                traceback.print_exc()
            try:
                phase_umap_png_path = cluster_out_dir / f"{animal_id}_{cluster_name}_07_UMAP_phrase_phase_colored.png"
                plot_umap_phrase_phase_colored(
                    embedding_xy=embedding_xy,
                    selection_df=phrase_selection_df,
                    animal_id=animal_id,
                    cluster_id=cluster_id,
                    out_png=phase_umap_png_path,
                    dpi=int(dpi),
                    rng=cluster_rng,
                )
                phase_colored_umap_png = str(phase_umap_png_path)
                print(f"[SAVE] {phase_umap_png_path}")
            except Exception as e:
                print(f"[ERROR] Could not save phase-colored UMAP for {animal_id} {cluster_name}: {e}")
                traceback.print_exc()

        if mode_bc in {"run_balanced_phrase_normalized", "run_balanced_full_contiguous"} and bool(save_full_run_bc_variants):
            try:
                full_idx_by_group, run_weights_by_group, n_runs_full_by_group = build_full_contiguous_run_indices_from_phrase_selection_df(
                    phrase_selection_df
                )
                if full_idx_by_group:
                    full_counts = {k: int(len(v)) for k, v in full_idx_by_group.items()}
                    full_runs_text = "; ".join(
                        f"{k}: {n_runs_full_by_group.get(k, 0):,} runs, {full_counts.get(k, 0):,} bins"
                        for k in sorted(full_idx_by_group.keys())
                    )

                    full_contiguous_path = cluster_out_dir / f"{animal_id}_{cluster_name}_05b_UMAP_full_contiguous_selected_runs.png"
                    plot_umap_cluster_selected_timebins(
                        embedding_xy=embedding_xy,
                        labels=labels,
                        selected_by_group=full_idx_by_group,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        out_png=full_contiguous_path,
                        dpi=int(dpi),
                        density_bins=int(umap_density_bins),
                        analysis_label="full-contiguous-run UMAPs, selected runs only",
                        n_each_text=full_runs_text,
                        bc_grid_point_coverage=bc_grid_point_coverage,
                    )
                    full_contiguous_umap_png = str(full_contiguous_path)
                    print(f"[SAVE] {full_contiguous_path}")

                    run_weighted_path = cluster_out_dir / f"{animal_id}_{cluster_name}_05c_UMAP_run_weighted_full_contiguous_selected_runs.png"
                    plot_umap_cluster_selected_timebins(
                        embedding_xy=embedding_xy,
                        labels=labels,
                        selected_by_group=full_idx_by_group,
                        weights_by_group=run_weights_by_group,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        out_png=run_weighted_path,
                        dpi=int(dpi),
                        density_bins=int(umap_density_bins),
                        analysis_label="run-weighted full-contiguous-run UMAPs, selected runs only",
                        n_each_text=full_runs_text + "; each run weight=1",
                        bc_grid_point_coverage=bc_grid_point_coverage,
                    )
                    run_weighted_full_contiguous_umap_png = str(run_weighted_path)
                    print(f"[SAVE] {run_weighted_path}")
            except Exception as e:
                print(f"[ERROR] Could not save full-run BC sensitivity plots for {animal_id} {cluster_name}: {e}")
                traceback.print_exc()

        spectrogram_paths: List[str] = []
        full_run_csv_paths: List[str] = []
        full_run_context_paths: List[str] = []
        mode = str(spectrogram_source_mode).strip().lower()
        valid_modes = {"selected_timebins", "expanded_full_runs", "both"}
        if mode not in valid_modes:
            raise ValueError(f"spectrogram_source_mode must be one of {sorted(valid_modes)}, got {spectrogram_source_mode!r}")

        expanded_cap = None
        if max_expanded_full_run_bins is not None and int(max_expanded_full_run_bins) > 0:
            expanded_cap = int(max_expanded_full_run_bins)

        for group_name, selected_idx in selected_by_group.items():
            safe_group = _clean_token(group_name)
            order_prefix = ordered_prefix.get(group_name, "50")

            # Summarize how exact equal-selected UMAP bins map back onto complete
            # contiguous HDBSCAN phrase-label runs. This is the key QC for avoiding
            # misleading chopped selected-bin spectrograms.
            full_run_df = summarize_selected_bins_within_full_runs(
                selected_indices=selected_idx,
                run_id_for_bin=run_id_for_bin,
                run_starts=run_starts,
                run_ends=run_ends,
                seconds_per_bin=float(seconds_per_bin),
                group_name=group_name,
                cluster_id=cluster_id,
            )
            full_run_csv = cluster_out_dir / (
                f"{animal_id}_{cluster_name}_{order_prefix}_{safe_group}_full_runs_from_equal_timebins.csv"
            )
            full_run_df.to_csv(full_run_csv, index=False)
            full_run_csv_paths.append(str(full_run_csv))
            print(f"[SAVE] {full_run_csv}")

            if mode in {"selected_timebins", "both"}:
                spec_png = cluster_out_dir / (
                    f"{animal_id}_{cluster_name}_{order_prefix}_{safe_group}_selected_timebins_spectrogram_"
                    f"N{int(len(selected_idx))}_rowbins{int(bins_per_spectrogram_row)}.png"
                )
                try:
                    plot_spectrogram_rows_for_timebins(
                        S_FxT=S_FxT,
                        selected_indices=selected_idx,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        group_name=group_name,
                        out_png=spec_png,
                        bins_per_row=int(bins_per_spectrogram_row),
                        seconds_per_bin=float(seconds_per_bin),
                        x_tick_step_s=float(x_tick_step_s),
                        figure_width=float(figure_width),
                        row_height=float(row_height),
                        subplot_hspace=float(subplot_hspace),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        max_rows_per_spectrogram=max_rows_per_spectrogram,
                        show_row_ylabels=True,
                    )
                    spectrogram_paths.append(str(spec_png))
                    print(f"[SAVE] {spec_png}")
                except Exception as e:
                    print(f"[ERROR] Could not save selected-timebin spectrogram for {animal_id} {cluster_name} {group_name}: {e}")
                    traceback.print_exc()

            if mode in {"expanded_full_runs", "both"}:
                expanded_idx, expanded_run_ids = expand_selected_bins_to_full_runs(
                    selected_indices=selected_idx,
                    run_id_for_bin=run_id_for_bin,
                    run_starts=run_starts,
                    run_ends=run_ends,
                    max_total_bins=expanded_cap,
                )
                spec_png = cluster_out_dir / (
                    f"{animal_id}_{cluster_name}_{order_prefix}_{safe_group}_expanded_full_runs_spectrogram_"
                    f"N{int(len(expanded_idx))}_from{len(expanded_run_ids)}runs_rowbins{int(bins_per_spectrogram_row)}.png"
                )
                try:
                    plot_spectrogram_rows_for_timebins(
                        S_FxT=S_FxT,
                        selected_indices=expanded_idx,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        group_name=f"{group_name} expanded full runs",
                        out_png=spec_png,
                        bins_per_row=int(bins_per_spectrogram_row),
                        seconds_per_bin=float(seconds_per_bin),
                        x_tick_step_s=float(x_tick_step_s),
                        figure_width=float(figure_width),
                        row_height=float(row_height),
                        subplot_hspace=float(subplot_hspace),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        max_rows_per_spectrogram=max_rows_per_spectrogram,
                        show_row_ylabels=False,
                    )
                    spectrogram_paths.append(str(spec_png))
                    print(f"[SAVE] {spec_png}")
                except Exception as e:
                    print(f"[ERROR] Could not save expanded full-run spectrogram for {animal_id} {cluster_name} {group_name}: {e}")
                    traceback.print_exc()

            if bool(save_representative_full_run_contexts) and full_run_df is not None and not full_run_df.empty:
                context_png = cluster_out_dir / (
                    f"{animal_id}_{cluster_name}_{order_prefix}_{safe_group}_representative_full_run_contexts.png"
                )
                try:
                    plot_representative_full_run_contexts(
                        S_FxT=S_FxT,
                        run_df=full_run_df,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        group_name=group_name,
                        out_png=context_png,
                        seconds_per_bin=float(seconds_per_bin),
                        context_bins=int(full_run_context_bins),
                        max_context_bins=int(full_run_max_context_bins),
                        fixed_panel_duration_s=float(full_run_fixed_duration_s),
                        max_examples=int(full_run_max_examples),
                        example_mode=str(full_run_example_mode),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        rng=cluster_rng,
                        labels=labels,
                        show_all_labels_track=False,
                    )
                    full_run_context_paths.append(str(context_png))
                    print(f"[SAVE] {context_png}")

                    context_png_all_labels = cluster_out_dir / (
                        f"{animal_id}_{cluster_name}_{order_prefix}_{safe_group}_representative_full_run_contexts_all_labels.png"
                    )
                    plot_representative_full_run_contexts(
                        S_FxT=S_FxT,
                        run_df=full_run_df,
                        animal_id=animal_id,
                        cluster_id=cluster_id,
                        group_name=group_name,
                        out_png=context_png_all_labels,
                        seconds_per_bin=float(seconds_per_bin),
                        context_bins=int(full_run_context_bins),
                        max_context_bins=int(full_run_max_context_bins),
                        fixed_panel_duration_s=float(full_run_fixed_duration_s),
                        max_examples=int(full_run_max_examples),
                        example_mode=str(full_run_example_mode),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        rng=cluster_rng,
                        labels=labels,
                        show_all_labels_track=True,
                    )
                    full_run_context_paths.append(str(context_png_all_labels))
                    print(f"[SAVE] {context_png_all_labels}")
                except Exception as e:
                    print(f"[ERROR] Could not save representative full-run contexts for {animal_id} {cluster_name} {group_name}: {e}")
                    traceback.print_exc()

        row_base.update({
            "umap_png": str(umap_png),
            "selected_timebins_csv": str(selected_csv),
            "phase_normalized_bc_by_phase_csv": phase_bc_csv,
            "phase_normalized_bc_by_phase_png": phase_bc_png,
            "phase_colored_umap_png": phase_colored_umap_png,
            "full_contiguous_selected_runs_umap_png": full_contiguous_umap_png,
            "run_weighted_full_contiguous_selected_runs_umap_png": run_weighted_full_contiguous_umap_png,
            "full_run_mapping_csvs": ";".join(full_run_csv_paths),
            "spectrogram_pngs": ";".join(spectrogram_paths),
            "representative_full_run_context_pngs": ";".join(full_run_context_paths),
            "spectrogram_source_mode": mode,
        })
        summary_rows.append(row_base)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = animal_out_dir / f"{animal_id}_equal_umap_cluster_spectrogram_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    n_done = int((summary_df.get("status", pd.Series(dtype=str)) == "done").sum()) if not summary_df.empty else 0
    n_skipped = int((summary_df.get("status", pd.Series(dtype=str)) == "skipped").sum()) if not summary_df.empty else 0
    print("=" * 92)
    print(f"[DONE] Saved summary: {summary_csv}")
    print(f"[DONE] Clusters with figures: {n_done}; skipped by equal-size/minimum-bin criteria: {n_skipped}")
    print(f"[DONE] Figures are under: {animal_out_dir}")
    print("=" * 92)
    return summary_csv


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "For one bird, export equal-sized UMAP cluster plots and matching "
            "spectrogram panels from the exact selected time bins."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True, help="Path to one bird's .npz file.")
    p.add_argument("--metadata-excel-path", required=True, help="Path to metadata Excel file with treatment dates.")
    p.add_argument("--spectrogram-script", required=True, help="Path to your pre/post spectrogram helper script.")
    p.add_argument(
        "--out-dir",
        default="/Volumes/my_own_SSD/updated_AreaX_outputs/equal_umap_cluster_spectrograms_exports",
        help=(
            "Output ROOT folder. The script will create an animal-ID subfolder inside this root, "
            "and save all UMAP/spectrogram images directly inside that animal folder."
        ),
    )
    p.add_argument("--animal-id", default=None, help="Optional animal ID. If omitted, inferred from NPZ filename.")
    p.add_argument("--input-csv", default=None, help="Optional top-variance or phrase-duration CSV used to restrict outputs to top-variance labels.")
    p.add_argument("--top30-csv", default=None, help="Backward-compatible alias for --input-csv when using a precomputed top-variance CSV.")
    p.add_argument("--phrase-csv", default=None, help="Backward-compatible alias for --input-csv when using usage_balanced_phrase_duration_stats.csv.")
    p.add_argument("--top-fraction", type=float, default=0.30, help="When using --phrase-csv, select this top fraction by post variance within the bird.")
    p.add_argument("--post-group-name", default="Post", help="When using --phrase-csv, group name used for post-lesion rows.")
    p.add_argument("--top-min-n-phrases", type=int, default=100, help="When using --phrase-csv, minimum N_phrases for ranking top-variance labels.")

    p.add_argument("--embedding-key", default="embedding_outputs")
    p.add_argument("--cluster-key", default="hdbscan_labels")
    p.add_argument("--spectrogram-key", default="s")
    p.add_argument("--file-key", default="file_indices")
    p.add_argument("--vocalization-key", default="vocalization")
    p.add_argument("--include-noise", action="store_true", help="Include cluster -1/noise.")
    p.add_argument(
        "--fill-noise-labels-from-nearest-nonnoise",
        action="store_true",
        help=(
            "Optional decoder-style label cleanup before selecting clusters/runs: replace -1/noise "
            "labels with the nearest non-noise label using the same left-to-right, in-place "
            "logic as decoder data prep. This changes label runs used for BC/QC but does not "
            "change the UMAP coordinates."
        ),
    )
    p.add_argument("--noise-label", type=int, default=-1, help="Noise label to fill when --fill-noise-labels-from-nearest-nonnoise is used.")
    p.add_argument(
        "--smooth-short-label-interruptions",
        action="store_true",
        help=(
            "Optional global label smoothing before BC/run selection: bridge short A-B-A "
            "interruptions by replacing the middle B run with A when it is shorter than "
            "--max-label-interruption-ms. Example: 17 17 12 17 -> 17 17 17 17."
        ),
    )
    p.add_argument("--max-label-interruption-ms", type=float, default=50.0, help="Maximum duration of the middle interruption run to bridge when --smooth-short-label-interruptions is used.")
    p.add_argument("--smoothing-max-iterations", type=int, default=3, help="Maximum number of global smoothing passes; multiple passes can bridge chains of short interruptions.")
    p.add_argument(
        "--apply-majority-vote-label-smoothing",
        action="store_true",
        help=(
            "Apply TweetyBERT-style temporal majority-vote smoothing before BC/run selection: "
            "each bin is replaced by the most frequent label in a centered sliding window, "
            "with first-in-window tie breaking."
        ),
    )
    p.add_argument("--majority-vote-window-bins", type=int, default=200, help="Window size in time bins for --apply-majority-vote-label-smoothing. 200 bins is ~540 ms at 2.7 ms/bin.")
    p.add_argument("--majority-vote-across-file-boundaries", action="store_true", help="If set, apply majority voting across the entire concatenated label array instead of independently within file_indices segments.")
    p.add_argument("--only-cluster-ids", nargs="*", default=None, help="Optional cluster IDs to export, e.g. --only-cluster-ids 0 3 11")
    p.add_argument("--no-vocalization-only", action="store_true", help="Do not restrict selected bins to vocalization==True.")

    p.add_argument(
        "--period-mode",
        choices=["pre_post", "early_late_pre_post"],
        default="pre_post",
        help="Groups to equalize across for each cluster.",
    )
    p.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")
    p.add_argument(
        "--early-late-split-method",
        choices=["file_median", "file_half", "timebin_half"],
        default="file_median",
        help="How to split pre/post periods when --period-mode early_late_pre_post is used.",
    )

    p.add_argument(
        "--bc-analysis-mode",
        choices=["run_balanced_phrase_normalized", "run_balanced_full_contiguous", "frame_weighted"],
        default="run_balanced_phrase_normalized",
        help=(
            "BC/UMAP sampling mode. run_balanced_phrase_normalized selects equal numbers "
            "of full phrase-label runs per group and equal normalized phase samples per run. "
            "run_balanced_full_contiguous selects equal numbers of full runs and uses every bin in those runs; frame_weighted reproduces the older equal-time-bin behavior."
        ),
    )
    p.add_argument("--min-runs-per-group", type=int, default=20, help="Minimum full phrase-label runs required in every group for run-balanced BC.")
    p.add_argument("--max-runs-per-group", type=int, default=200, help="Cap full phrase-label runs per group for run-balanced BC. Use 0 for no cap.")
    p.add_argument("--phase-bins-per-run", type=int, default=25, help="Number of normalized phase samples drawn from each selected full run.")
    p.add_argument("--min-full-run-duration-ms", type=float, default=100.0, help="Minimum full phrase-label run duration included in run-balanced BC.")
    p.add_argument("--min-run-group-fraction", type=float, default=0.80, help="Minimum fraction of a full run that must fall in a period/group for run-balanced BC.")
    p.add_argument("--run-sample-mode", choices=["random", "first", "longest"], default="random", help="How to choose full runs when more than max-runs-per-group are available.")
    p.add_argument("--allow-repeat-phase-bins", action="store_true", help="Allow repeated time-bin indices when phase-normalizing runs shorter than phase-bins-per-run.")
    p.add_argument("--no-phase-normalized-bc", action="store_true", help="Skip phase-by-phase BC curve and phase-colored UMAP outputs.")
    p.add_argument("--save-full-run-bc-variants", action="store_true", help="For run-balanced mode, also save UMAP/BC plots using full contiguous selected runs, with and without run weighting.")

    p.add_argument("--min-points-per-group", type=int, default=6000, help="Frame-weighted mode only: minimum equalized bins required in every group for a cluster to generate figures.")
    p.add_argument(
        "--max-points-per-group",
        type=int,
        default=6000,
        help="Cap selected bins per group. Use 0 for no cap.",
    )
    p.add_argument("--sample-mode", choices=["random", "first", "time_balanced", "time_balanced_random_offset"], default="random", help="Frame-weighted mode only: how to select equal time bins from each group.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--bins-per-spectrogram-row", type=int, default=2000)
    p.add_argument("--max-rows-per-spectrogram", type=int, default=None, help="Optional cap to avoid huge PNGs.")
    p.add_argument(
        "--spectrogram-source-mode",
        choices=["selected_timebins", "expanded_full_runs", "both"],
        default="expanded_full_runs",
        help=(
            "Which time bins to use for spectrogram visualization. "
            "UMAP/BC always uses exact equal selected time bins. "
            "expanded_full_runs expands selected bins to their complete contiguous HDBSCAN phrase-label runs."
        ),
    )
    p.add_argument("--max-expanded-full-run-bins", type=int, default=10000, help="Cap expanded full-run spectrogram bins per group; use 0 for no cap.")
    p.add_argument("--no-representative-full-run-contexts", action="store_true", help="Skip representative original-context full-run spectrogram figures.")
    p.add_argument("--full-run-context-bins", type=int, default=80, help="Context bins on each side for representative full-run context plots.")
    p.add_argument("--full-run-max-context-bins", type=int, default=1500, help="Legacy cap on available context bins copied into a fixed-width panel; use 0 for no cap.")
    p.add_argument("--full-run-fixed-duration-s", type=float, default=3.0, help="Fixed x-axis duration in seconds for representative full-run context plots. Shorter examples occupy only part of the panel instead of being stretched.")
    p.add_argument("--full-run-max-examples", type=int, default=12, help="Maximum representative full-run context examples per group.")
    p.add_argument("--full-run-example-mode", choices=["mixed", "longest", "shortest", "random"], default="mixed")
    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--x-tick-step-s", type=float, default=1.0)
    p.add_argument("--figure-width", type=float, default=24.0)
    p.add_argument("--row-height", type=float, default=2.2)
    p.add_argument("--subplot-hspace", type=float, default=0.005)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--umap-density-bins", type=int, default=20, help="Number of 2D histogram bins per axis for the BC-style UMAP overlap density plots.")
    p.add_argument("--bc-grid-point-coverage", type=float, default=None, help="Optional central fraction of selected UMAP points used to define the BC density grid, e.g. 0.99 to crop distant outliers. Omit or use 1.0 to include all selected points.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    only_cluster_ids = None
    if args.only_cluster_ids:
        only_cluster_ids = [int(x) for x in _parse_list_arg(args.only_cluster_ids)]

    max_points = int(args.max_points_per_group)
    if max_points <= 0:
        max_points_per_group = None
    else:
        max_points_per_group = max_points

    max_runs = int(args.max_runs_per_group)
    if max_runs <= 0:
        max_runs_per_group = None
    else:
        max_runs_per_group = max_runs

    contrast = None if bool(args.no_contrast_percentiles) else tuple(float(x) for x in args.contrast_percentiles)
    input_csv_arg = args.input_csv or args.top30_csv or args.phrase_csv
    top_variance_csv = Path(input_csv_arg).expanduser() if input_csv_arg else None

    export_equal_umap_cluster_spectrograms_for_one_bird(
        npz_path=Path(args.npz_path).expanduser(),
        metadata_excel_path=Path(args.metadata_excel_path).expanduser(),
        spectrogram_script=Path(args.spectrogram_script).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        animal_id=args.animal_id,
        embedding_key=str(args.embedding_key),
        cluster_key=str(args.cluster_key),
        spectrogram_key=str(args.spectrogram_key),
        file_key=str(args.file_key),
        vocalization_key=str(args.vocalization_key),
        include_noise=bool(args.include_noise),
        only_cluster_ids=only_cluster_ids,
        vocalization_only=not bool(args.no_vocalization_only),
        period_mode=str(args.period_mode),
        treatment_day_assignment=str(args.treatment_day_assignment),
        early_late_split_method=str(args.early_late_split_method),
        bc_analysis_mode=str(args.bc_analysis_mode),
        min_points_per_group=int(args.min_points_per_group),
        max_points_per_group=max_points_per_group,
        sample_mode=str(args.sample_mode),
        min_runs_per_group=int(args.min_runs_per_group),
        max_runs_per_group=max_runs_per_group,
        phase_bins_per_run=int(args.phase_bins_per_run),
        min_full_run_duration_ms=float(args.min_full_run_duration_ms),
        min_run_group_fraction=float(args.min_run_group_fraction),
        run_sample_mode=str(args.run_sample_mode),
        allow_repeat_phase_bins=bool(args.allow_repeat_phase_bins),
        save_phase_normalized_bc=not bool(args.no_phase_normalized_bc),
        save_full_run_bc_variants=bool(args.save_full_run_bc_variants),
        seed=int(args.seed),
        bins_per_spectrogram_row=int(args.bins_per_spectrogram_row),
        max_rows_per_spectrogram=args.max_rows_per_spectrogram,
        spectrogram_source_mode=str(args.spectrogram_source_mode),
        save_representative_full_run_contexts=not bool(args.no_representative_full_run_contexts),
        max_expanded_full_run_bins=(None if int(args.max_expanded_full_run_bins) <= 0 else int(args.max_expanded_full_run_bins)),
        full_run_context_bins=int(args.full_run_context_bins),
        full_run_max_context_bins=int(args.full_run_max_context_bins),
        full_run_fixed_duration_s=float(args.full_run_fixed_duration_s),
        full_run_max_examples=int(args.full_run_max_examples),
        full_run_example_mode=str(args.full_run_example_mode),
        seconds_per_bin=float(args.seconds_per_bin),
        x_tick_step_s=float(args.x_tick_step_s),
        figure_width=float(args.figure_width),
        row_height=float(args.row_height),
        subplot_hspace=float(args.subplot_hspace),
        cmap=str(args.cmap),
        contrast_percentiles=contrast,
        dpi=int(args.dpi),
        umap_density_bins=int(args.umap_density_bins),
        bc_grid_point_coverage=(None if args.bc_grid_point_coverage is None or float(args.bc_grid_point_coverage) >= 1.0 else float(args.bc_grid_point_coverage)),
        fill_noise_labels_from_nearest_nonnoise=bool(args.fill_noise_labels_from_nearest_nonnoise),
        noise_label=int(args.noise_label),
        smooth_short_label_interruptions=bool(args.smooth_short_label_interruptions),
        max_label_interruption_ms=float(args.max_label_interruption_ms),
        smoothing_max_iterations=int(args.smoothing_max_iterations),
        apply_majority_vote_label_smoothing=bool(args.apply_majority_vote_label_smoothing),
        majority_vote_window_bins=int(args.majority_vote_window_bins),
        majority_vote_by_file=not bool(args.majority_vote_across_file_boundaries),
        dry_run=bool(args.dry_run),
        top_variance_csv=top_variance_csv,
        top_fraction=float(args.top_fraction),
        post_group_name=str(args.post_group_name),
        top_min_n_phrases=int(args.top_min_n_phrases),
    )


if __name__ == "__main__":
    main()
