#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_equal_umap_cluster_spectrograms_v3_top30_flat.py

Companion utility for UMAP/HDBSCAN time-bin analyses.

For each selected HDBSCAN/UMAP cluster in one bird, this script:
  1) builds equal-sized time-bin datasets across periods/groups,
  2) saves a UMAP scatterplot using exactly those selected time bins, and
  3) saves matching spectrogram PNGs.

Important v7 change:
  The UMAP/Bhattacharyya calculations still use exactly the equal-sized
  selected time bins. However, spectrogram visualization can now expand those
  selected bins back to the full contiguous HDBSCAN phrase-label runs that
  contain them. This avoids the misleading chopped-looking spectrograms that
  happen when sparse random time bins from long stuttered runs are stitched
  together.

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

SCRIPT_VERSION = "equal_umap_cluster_spectrograms_v7_full_run_spectrograms"


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
    idx = np.asarray(idx, dtype=int)
    if n <= 0:
        return np.array([], dtype=int)
    if idx.size < n:
        raise ValueError(f"Cannot choose {n} indices from only {idx.size} available indices.")

    if sample_mode == "first":
        return np.asarray(idx[:n], dtype=int)
    if sample_mode == "random":
        return np.sort(rng.choice(idx, size=n, replace=False)).astype(int)
    raise ValueError("sample_mode must be 'first' or 'random'.")


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


def plot_representative_full_run_contexts(
    *,
    S_FxT: np.ndarray,
    run_df: pd.DataFrame,
    animal_id: str,
    cluster_id: int,
    group_name: str,
    out_png: Path,
    seconds_per_bin: float,
    context_bins: int,
    max_context_bins: int,
    max_examples: int,
    example_mode: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    rng: np.random.Generator,
) -> None:
    """Plot representative original-context full label runs.

    Blue shading shows the complete HDBSCAN phrase-label run. Red shading shows
    the equal UMAP-selected bins within that full run. This is the sanity-check
    companion to the exact-selected/expanded stitched spectrograms.
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
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        run_start = int(row["run_start_bin"])
        run_end = int(row["run_end_bin_exclusive"])
        context_start = max(0, run_start - int(context_bins))
        context_end = min(N, run_end + int(context_bins))

        if int(max_context_bins) > 0 and (context_end - context_start) > int(max_context_bins):
            # Prefer to show the beginning of the run plus context. This keeps long
            # stutter bouts inspectable without creating enormous rows.
            context_end = min(N, context_start + int(max_context_bins))

        panel = S_FxT[:, context_start:context_end].astype(float)
        duration_s = panel.shape[1] * float(seconds_per_bin)
        ax.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, duration_s, 0, panel.shape[0]],
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_xlim(0, duration_s)

        # Blue: full HDBSCAN run.
        blue_start = max(run_start, context_start)
        blue_end = min(run_end, context_end)
        if blue_end > blue_start:
            ax.axvspan(
                (blue_start - context_start) * float(seconds_per_bin),
                (blue_end - context_start) * float(seconds_per_bin),
                color="tab:blue",
                alpha=0.16,
            )

        # Red: selected time-bin segments within the full run.
        seg_starts = _parse_semicolon_ints(row.get("selected_segment_starts", ""))
        seg_ends = _parse_semicolon_ints(row.get("selected_segment_ends_exclusive", ""))
        for s, e in zip(seg_starts, seg_ends):
            s2 = max(int(s), context_start)
            e2 = min(int(e), context_end)
            if e2 > s2:
                ax.axvspan(
                    (s2 - context_start) * float(seconds_per_bin),
                    (e2 - context_start) * float(seconds_per_bin),
                    color="tab:red",
                    alpha=0.28,
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

    axes[-1].set_xlabel("Original recording context around full phrase-label run (s)")
    fig.suptitle(
        f"{animal_id} — label {int(cluster_id):02d} — {group_name}\n"
        "Representative full HDBSCAN phrase-label runs; blue = full run, red = equal UMAP-selected bins",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.12)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


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
) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    H, _, _ = np.histogram2d(xy[idx, 0], xy[idx, 1], bins=[x_edges, y_edges])
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

    x_min = float(np.nanmin(xy[:, 0]))
    x_max = float(np.nanmax(xy[:, 0]))
    y_min = float(np.nanmin(xy[:, 1]))
    y_max = float(np.nanmax(xy[:, 1]))
    pad_x = max(1e-6, (x_max - x_min) * 0.04)
    pad_y = max(1e-6, (y_max - y_min) * 0.04)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, int(density_bins) + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, int(density_bins) + 1)

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

        hist_pre = _compute_hist_density(xy, idx_by['early_pre'], x_edges=x_edges, y_edges=y_edges)
        hist_lpre = _compute_hist_density(xy, idx_by['late_pre'], x_edges=x_edges, y_edges=y_edges)
        hist_epost = _compute_hist_density(xy, idx_by['early_post'], x_edges=x_edges, y_edges=y_edges)
        hist_lpost = _compute_hist_density(xy, idx_by['late_post'], x_edges=x_edges, y_edges=y_edges)
        bc_pre = _hist_bc(hist_pre, hist_lpre)
        bc_post = _hist_bc(hist_epost, hist_lpost)
        hist_pre_all = _compute_hist_density(xy, np.concatenate([idx_by['early_pre'], idx_by['late_pre']]), x_edges=x_edges, y_edges=y_edges)
        hist_post_all = _compute_hist_density(xy, np.concatenate([idx_by['early_post'], idx_by['late_post']]), x_edges=x_edges, y_edges=y_edges)
        bc_prepost = _hist_bc(hist_pre_all, hist_post_all)

        fig = plt.figure(figsize=(16.5, 8.2))
        gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 0.9, 1.1], wspace=0.10, hspace=0.18)
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
            ax.set_title(f"{title}", color=color, fontsize=13, pad=4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax_pre_ov.imshow(_density_to_rgb(hist_pre, hist_lpre), origin='lower', aspect='auto')
        ax_pre_ov.set_title(f"Pre early vs late overlap\nBC={bc_pre:.3f}", fontsize=12, pad=4)
        ax_post_ov.imshow(_density_to_rgb(hist_epost, hist_lpost), origin='lower', aspect='auto')
        ax_post_ov.set_title(f"Post early vs late overlap\nBC={bc_post:.3f}", fontsize=12, pad=4)
        ax_prepost_ov.imshow(_density_to_rgb(hist_pre_all, hist_post_all), origin='lower', aspect='auto')
        ax_prepost_ov.set_title(
            f"Pre vs post overlap\npre N={len(idx_by['early_pre']) + len(idx_by['late_pre']):,} | post N={len(idx_by['early_post']) + len(idx_by['late_post']):,}\nBC={bc_prepost:.3f}",
            fontsize=12, pad=4
        )
        for ax in [ax_pre_ov, ax_post_ov, ax_prepost_ov]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('black')
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig.suptitle(
            f"{animal_id} cluster {cluster_id} early/late UMAPs (equal groups only)\n"
            f"n_each={min(counts.values()):,} | pre BC={bc_pre:.3f} | post BC={bc_post:.3f} | pre/post BC={bc_prepost:.3f} | density bins={int(density_bins)}",
            y=0.985, fontsize=18
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if len(groups_present) >= 2:
        g1, i1 = groups_present[0]
        g2, i2 = groups_present[1]
        h1 = _compute_hist_density(xy, i1, x_edges=x_edges, y_edges=y_edges)
        h2 = _compute_hist_density(xy, i2, x_edges=x_edges, y_edges=y_edges)
        bc = _hist_bc(h1, h2)
        ax = axes[-1]
        ax.imshow(_density_to_rgb(h1, h2), origin='lower', aspect='auto')
        ax.set_title(f"{g1} vs {g2} overlap\nBC={bc:.3f} | bins={int(density_bins)}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{animal_id} — HDBSCAN label {cluster_id} — equal selected time bins", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    min_points_per_group: int = 6000,
    max_points_per_group: Optional[int] = 6000,
    sample_mode: str = "random",
    seed: int = 0,
    bins_per_spectrogram_row: int = 2000,
    max_rows_per_spectrogram: Optional[int] = None,
    spectrogram_source_mode: str = "expanded_full_runs",
    save_representative_full_run_contexts: bool = True,
    max_expanded_full_run_bins: Optional[int] = 10000,
    full_run_context_bins: int = 80,
    full_run_max_context_bins: int = 1500,
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
    dry_run: bool = False,
    top_variance_csv: Optional[Path] = None,
    top_fraction: float = 0.30,
    post_group_name: str = "Post",
    top_min_n_phrases: int = 100,
) -> Path:
    """
    Generate one UMAP plot and matching spectrogram figures for each cluster.

    The UMAP/BC plot uses the same equal-sized selected time-bin indices.
    The selected indices are saved to CSV for traceability. Spectrograms can
    either show those exact selected bins or expand them to the full contiguous
    HDBSCAN phrase-label runs that contain them.
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

    labels = np.asarray(arr[cluster_key]).astype(int)
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

    file_indices = _get_optional_array(arr, file_key, labels.shape[0])
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
    print(f"[INFO] Spectrogram source mode: {spectrogram_source_mode}")
    print("[INFO] UMAP/BC still uses exact equal selected time bins; expanded spectrograms restore full phrase-label runs.")
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
            "status": "skipped" if selected_by_group is None else "done",
            "n_equal_selected_per_group": int(n_equal),
            "min_points_per_group": int(min_points_per_group),
            "max_points_per_group": max_points_per_group,
            "sample_mode": sample_mode,
            "cluster_seed": cluster_seed,
        }
        for group_name, count in available_counts.items():
            row_base[f"available_{group_name}_bins"] = int(count)

        if selected_by_group is None:
            print(
                f"[SKIP] {animal_id} {cluster_name}: n_equal={n_equal:,} < "
                f"min_points_per_group={int(min_points_per_group):,}; counts={available_counts}"
            )
            summary_rows.append(row_base)
            continue

        print(
            f"[RUN] {animal_id} {cluster_name}: selected N={n_equal:,} per group; "
            f"available={available_counts}"
        )
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
        umap_png = cluster_out_dir / f"{animal_id}_{cluster_name}_05_UMAP_equal_timebins.png"
        selected_csv = cluster_out_dir / f"{animal_id}_{cluster_name}_99_selected_equal_timebins.csv"
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
        )
        print(f"[SAVE] {umap_png}")
        print(f"[SAVE] {selected_csv}")

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
                        max_examples=int(full_run_max_examples),
                        example_mode=str(full_run_example_mode),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        rng=cluster_rng,
                    )
                    full_run_context_paths.append(str(context_png))
                    print(f"[SAVE] {context_png}")
                except Exception as e:
                    print(f"[ERROR] Could not save representative full-run contexts for {animal_id} {cluster_name} {group_name}: {e}")
                    traceback.print_exc()

        row_base.update({
            "umap_png": str(umap_png),
            "selected_timebins_csv": str(selected_csv),
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

    p.add_argument("--min-points-per-group", type=int, default=6000, help="Minimum equalized bins required in every group for a cluster to generate figures. Lower this to generate more clusters.")
    p.add_argument(
        "--max-points-per-group",
        type=int,
        default=6000,
        help="Cap selected bins per group. Use 0 for no cap.",
    )
    p.add_argument("--sample-mode", choices=["random", "first"], default="random")
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
    p.add_argument("--full-run-max-context-bins", type=int, default=1500, help="Maximum total bins per context example row; use 0 for no cap.")
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
        min_points_per_group=int(args.min_points_per_group),
        max_points_per_group=max_points_per_group,
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
        bins_per_spectrogram_row=int(args.bins_per_spectrogram_row),
        max_rows_per_spectrogram=args.max_rows_per_spectrogram,
        spectrogram_source_mode=str(args.spectrogram_source_mode),
        save_representative_full_run_contexts=not bool(args.no_representative_full_run_contexts),
        max_expanded_full_run_bins=(None if int(args.max_expanded_full_run_bins) <= 0 else int(args.max_expanded_full_run_bins)),
        full_run_context_bins=int(args.full_run_context_bins),
        full_run_max_context_bins=int(args.full_run_max_context_bins),
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
        dry_run=bool(args.dry_run),
        top_variance_csv=top_variance_csv,
        top_fraction=float(args.top_fraction),
        post_group_name=str(args.post_group_name),
        top_min_n_phrases=int(args.top_min_n_phrases),
    )


if __name__ == "__main__":
    main()
