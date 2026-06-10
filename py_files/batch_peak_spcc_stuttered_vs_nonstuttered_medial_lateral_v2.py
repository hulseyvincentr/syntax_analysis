#!/usr/bin/env python3
"""
batch_peak_spcc_stuttered_vs_nonstuttered_medial_lateral.py

Batch analysis for literature-style peak spectrographic cross-correlation (SPCC)
of stuttered vs non-stuttered syllables in Medial + Lateral Area X lesion birds.

What this script does
---------------------
For each selected bird:
1) Uses phrase-duration variance change to choose:
   - stuttered labels: labels with the largest post-pre phrase-duration variance increase
   - non-stuttered labels: labels with near-zero/small phrase-duration variance change
2) Uses your existing canary segmentation code from
   cluster_spectrogram_crosscorr_canary_segmentation_v1.py to segment same-label syllable
   renditions into syllable-like units.
3) Computes peak spectrographic cross-correlation (SPCC) between same-label syllables by
   sliding one spectrogram against the other in time and taking the maximum Pearson
   correlation across shifts.
4) Summarizes each bird x label with:
   - median pre-pre SPCC: pre-lesion within-condition stereotypy
   - median post-post SPCC: post-lesion within-condition stereotypy
   - median pre-post SPCC: similarity between pre and post syllables
   - stereotypy_delta_post_minus_pre = post-post median - pre-pre median
   - identity_shift_delta_prepost_minus_prepre = pre-post median - pre-pre median
5) Tests at the bird level:
   - Are stuttered syllable deltas < 0?
   - Are non-stuttered syllable deltas < 0?
   - Are stuttered deltas more negative than non-stuttered deltas?

Why bird-level stats?
---------------------
Pairwise correlations are useful descriptive values, but they are not independent:
the same syllable renditions are reused in many pairs. This script therefore computes
one summary per bird x syllable, then averages within bird before group-level tests.

Required input
--------------
- metadata Excel with Animal ID, Treatment date, and lesion hit-type columns
- phrase-duration summary CSV with post-pre variance delta per animal x syllable
- NPZ root with one .npz per bird
- your existing single-label segmentation/crosscorr script, used as a function library

Example
-------
cd "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files"

python batch_peak_spcc_stuttered_vs_nonstuttered_medial_lateral.py \
  --crosscorr-script "./cluster_spectrogram_crosscorr_canary_segmentation_v1.py" \
  --metadata-excel-path "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --metadata-sheet "metadata_with_hit_type" \
  --hit-type-cols "Lesion hit type" \
  --phrase-duration-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv" \
  --variance-delta-col "Post_vs_Pre_Delta_Variance_ms2" \
  --npz-root "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/medial_lateral_peak_spcc_stuttered_vs_nonstuttered" \
  --top-n-stuttered 3 \
  --top-n-nonstuttered 3 \
  --min-pre-phrases 20 \
  --min-post-phrases 20 \
  --max-shift-ms 30 \
  --corr-freq-min-khz 0.8 \
  --corr-freq-max-khz 10.0 \
  --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False


# -----------------------------
# General helpers
# -----------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_col(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def label_to_str(x: object) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)) and np.isfinite(x) and float(x).is_integer():
        return str(int(x))
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def safe_name(s: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def find_column(df: pd.DataFrame, requested: Optional[str], candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    if requested:
        if requested in cols:
            return requested
        requested_norm = norm_col(requested)
        for c in cols:
            if norm_col(c) == requested_norm:
                return c
        if required:
            raise ValueError(f"Requested column not found: {requested!r}. Available columns: {cols}")
        return None

    norm_to_col = {norm_col(c): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        nc = norm_col(cand)
        if nc in norm_to_col:
            return norm_to_col[nc]

    if required:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}. Available columns: {cols}")
    return None


def is_post_value(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    return s in {"post", "post lesion", "postlesion", "after", "after lesion"} or s.startswith("post")


def truthy_hit_value(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"y", "yes", "true", "t", "1", "hit", "complete", "partial"}


def load_python_module(path: str | Path, module_name: str = "crosscorr_segmenter"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find Python script to import: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------
# Metadata and syllable selection
# -----------------------------


def auto_hit_type_columns(meta: pd.DataFrame) -> list[str]:
    out = []
    for c in meta.columns:
        n = norm_col(c)
        if ("hittype" in n) or (("hit" in n) and ("type" in n)) or (("lesion" in n) and ("type" in n)):
            out.append(c)
    return out


def read_metadata_animals(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    meta = pd.read_excel(args.metadata_excel_path, sheet_name=args.metadata_sheet)
    animal_col = find_column(meta, args.animal_id_col, ["Animal ID", "animal_id", "AnimalID"])

    if args.hit_type_cols:
        hit_type_cols = [find_column(meta, c, [], required=True) for c in args.hit_type_cols]
    else:
        hit_type_cols = auto_hit_type_columns(meta)

    selected_mask = pd.Series(False, index=meta.index)
    hit_regex = re.compile(args.hit_type_regex, flags=re.IGNORECASE)

    if hit_type_cols:
        hit_text = meta[hit_type_cols].astype(str).agg(" | ".join, axis=1)
        selected_mask = hit_text.str.contains(hit_regex, na=False)
        print(f"[INFO] Using hit-type columns for lesion filter: {hit_type_cols}")
    else:
        print("[WARN] No hit-type columns found. Trying medial+lateral hit flag fallback.")

    medial_flag_col = find_column(
        meta,
        args.medial_hit_flag_col,
        ["Medial Area X hit?", "Medial hit?", "Medial hit", "Medial Area X hit"],
        required=False,
    )
    lateral_flag_col = find_column(
        meta,
        args.lateral_hit_flag_col,
        ["Lateral Area X hit?", "Lateral hit?", "Lateral hit", "Lateral Area X hit"],
        required=False,
    )
    if args.also_accept_both_hit_flags and medial_flag_col and lateral_flag_col:
        flag_mask = meta[medial_flag_col].map(truthy_hit_value) & meta[lateral_flag_col].map(truthy_hit_value)
        selected_mask = selected_mask | flag_mask
        print(f"[INFO] Also accepting rows with both hit flags: {medial_flag_col!r}, {lateral_flag_col!r}")

    if args.visible_only:
        visible_col = find_column(
            meta,
            args.visible_col,
            ["Area X visible in histology?", "Area X visible", "Area X visible in histology"],
            required=False,
        )
        if visible_col:
            selected_mask = selected_mask & meta[visible_col].map(truthy_hit_value)
            print(f"[INFO] Applying visible-only filter using column: {visible_col!r}")
        else:
            print("[WARN] --visible-only was requested, but no visible-histology column was found.")

    selected = meta.loc[selected_mask].copy()
    selected[animal_col] = selected[animal_col].astype(str)
    selected_animals = sorted(selected[animal_col].dropna().astype(str).unique().tolist())
    if not selected_animals:
        raise ValueError(
            "No animals matched the lesion filter. Check --metadata-sheet, --hit-type-cols, "
            "--hit-type-regex, or try omitting --visible-only."
        )
    print(f"[INFO] Selected {len(selected_animals)} animal(s): {', '.join(selected_animals)}")
    selected = selected.rename(columns={animal_col: "animal_id"})
    return selected, selected_animals


def find_period_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    return find_column(df, requested, ["Group", "group", "Period", "period", "Epoch", "epoch"], required=False)


def find_variance_delta_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    candidates = [
        "Post_vs_Pre_Delta_Variance_ms2",
        "PostMinusPreVar_ms2",
        "PostMinusPreVariance_ms2",
        "PostMinusPreVariance",
        "PostMinusPreVar",
        "delta_variance_ms2",
        "DeltaVariance_ms2",
        "post_minus_pre_variance_ms2",
        "post_minus_pre_var_ms2",
    ]
    col = find_column(df, requested, candidates, required=False)
    if col:
        return col
    for c in df.columns:
        n = norm_col(c)
        if "post" in n and "pre" in n and "var" in n:
            return c
    return None


def compute_variance_delta_if_needed(df: pd.DataFrame, animal_col: str, label_col: str, period_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    variance_col = find_column(
        out,
        None,
        ["Variance_ms2", "Variance_ms^2", "Variance", "Var_ms2", "variance_ms2"],
        required=False,
    )
    prepooled_col = find_column(
        out,
        None,
        ["Pre_Variance_ms2", "PrePooledVariance_ms2", "PrePooledVar_ms2", "PrePooledVariance"],
        required=False,
    )
    if variance_col and prepooled_col:
        out["PostMinusPreVar_ms2_auto"] = pd.to_numeric(out[variance_col], errors="coerce") - pd.to_numeric(out[prepooled_col], errors="coerce")
        return out

    if not (variance_col and period_col):
        raise ValueError(
            "Could not find a variance-delta column and could not compute one. "
            "Expected a post-pre variance delta column or period rows with Variance_ms2."
        )

    tmp = out[[animal_col, label_col, period_col, variance_col]].copy()
    tmp["period_norm"] = tmp[period_col].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False).str.replace("-", " ", regex=False)
    tmp[variance_col] = pd.to_numeric(tmp[variance_col], errors="coerce")
    piv = tmp.pivot_table(index=[animal_col, label_col], columns="period_norm", values=variance_col, aggfunc="mean")
    post_candidates = [c for c in piv.columns if str(c).startswith("post")]
    pre_candidates = [c for c in ["late pre", "pre", "early pre"] if c in piv.columns]
    if not post_candidates or not pre_candidates:
        raise ValueError(f"Could not compute variance delta from periods: {list(piv.columns)}")
    delta = (piv[post_candidates[0]] - piv[pre_candidates[0]]).rename("PostMinusPreVar_ms2_auto").reset_index()
    out = out.merge(delta, on=[animal_col, label_col], how="left")
    print(f"[INFO] Computed variance delta as {post_candidates[0]!r} - {pre_candidates[0]!r}.")
    return out


def read_phrase_table_for_label_selection(args: argparse.Namespace, selected_animals: list[str]) -> pd.DataFrame:
    phrase = pd.read_csv(args.phrase_duration_csv)
    animal_col = find_column(phrase, args.phrase_animal_id_col, ["Animal ID", "animal_id", "AnimalID", "animal"])
    label_col = find_column(phrase, args.label_col, ["Syllable", "syllable", "label", "Label", "cluster_label", "Cluster label", "syllable_label"])
    period_col = find_period_col(phrase, args.period_col)
    delta_col = find_variance_delta_column(phrase, args.variance_delta_col)

    phrase[animal_col] = phrase[animal_col].astype(str)
    phrase[label_col] = phrase[label_col].map(label_to_str)

    if delta_col is None:
        phrase = compute_variance_delta_if_needed(phrase, animal_col, label_col, period_col)
        delta_col = "PostMinusPreVar_ms2_auto"

    # Restrict to Post rows when period rows exist. In your phrase-duration CSV, the post rows
    # contain the post-pre delta and post N_phrases values.
    if period_col:
        post_like = phrase[period_col].map(is_post_value)
        if post_like.any():
            phrase = phrase.loc[post_like].copy()

    pre_n_col = find_column(phrase, args.pre_n_col, ["Pre_N_phrases", "Pre N phrases", "N_pre", "pre_n", "PreN"], required=False)
    post_n_col = find_column(phrase, args.post_n_col, ["Post_N_phrases", "Post N phrases", "N_post", "post_n", "N_phrases"], required=False)

    phrase[delta_col] = pd.to_numeric(phrase[delta_col], errors="coerce")
    phrase = phrase[phrase[animal_col].isin(selected_animals)].copy()
    phrase = phrase[np.isfinite(phrase[delta_col])].copy()

    if pre_n_col:
        phrase[pre_n_col] = pd.to_numeric(phrase[pre_n_col], errors="coerce")
        phrase = phrase[phrase[pre_n_col] >= args.min_pre_phrases].copy()
    if post_n_col:
        phrase[post_n_col] = pd.to_numeric(phrase[post_n_col], errors="coerce")
        phrase = phrase[phrase[post_n_col] >= args.min_post_phrases].copy()

    if phrase.empty:
        raise ValueError("No phrase-duration rows remained after animal/sample-size filtering.")

    keep_cols = [animal_col, label_col, delta_col]
    if pre_n_col:
        keep_cols.append(pre_n_col)
    if post_n_col and post_n_col not in keep_cols:
        keep_cols.append(post_n_col)

    # One row per animal x label. If duplicates remain, keep the largest variance delta.
    collapsed = phrase[keep_cols].copy()
    rename = {animal_col: "animal_id", label_col: "cluster_label", delta_col: "phrase_variance_delta_ms2"}
    if pre_n_col:
        rename[pre_n_col] = "pre_n_phrases"
    if post_n_col:
        rename[post_n_col] = "post_n_phrases"
    collapsed = collapsed.rename(columns=rename)
    collapsed = (
        collapsed.groupby(["animal_id", "cluster_label"], as_index=False)
        .agg({
            "phrase_variance_delta_ms2": "max",
            **({"pre_n_phrases": "max"} if "pre_n_phrases" in collapsed.columns else {}),
            **({"post_n_phrases": "max"} if "post_n_phrases" in collapsed.columns else {}),
        })
    )
    return collapsed


def select_stuttered_and_nonstuttered_labels(args: argparse.Namespace, selected_animals: list[str]) -> pd.DataFrame:
    labels = read_phrase_table_for_label_selection(args, selected_animals)
    pieces = []
    for animal, sub0 in labels.groupby("animal_id", sort=True):
        sub = sub0.copy()
        sub["cluster_label"] = sub["cluster_label"].map(label_to_str)

        # Stuttered: largest positive variance increases.
        stut_candidates = sub.copy()
        if not args.allow_nonpositive_stuttered:
            stut_candidates = stut_candidates[stut_candidates["phrase_variance_delta_ms2"] > args.min_stutter_variance_delta]
        stut = stut_candidates.sort_values("phrase_variance_delta_ms2", ascending=False).head(args.top_n_stuttered).copy()
        stut["stutter_status"] = "stuttered"
        stut["selection_rank_within_bird"] = np.arange(1, len(stut) + 1)

        # Non-stuttered: matched number of labels with near-zero or low variance change,
        # excluding selected stuttered labels.
        stut_labels = set(stut["cluster_label"].astype(str))
        non = sub[~sub["cluster_label"].astype(str).isin(stut_labels)].copy()
        if args.nonstutter_max_variance_delta is not None:
            non = non[non["phrase_variance_delta_ms2"] <= args.nonstutter_max_variance_delta].copy()

        if args.nonstutter_mode == "near_zero":
            non["nonstutter_sort_key"] = non["phrase_variance_delta_ms2"].abs()
            non = non.sort_values(["nonstutter_sort_key", "phrase_variance_delta_ms2"], ascending=[True, True])
        elif args.nonstutter_mode == "lowest":
            non = non.sort_values("phrase_variance_delta_ms2", ascending=True)
        elif args.nonstutter_mode == "lowest_positive":
            non = non[non["phrase_variance_delta_ms2"] >= 0].sort_values("phrase_variance_delta_ms2", ascending=True)
        else:
            raise ValueError("--nonstutter-mode must be near_zero, lowest, or lowest_positive")

        n_non = args.top_n_nonstuttered if args.top_n_nonstuttered is not None else args.top_n_stuttered
        non = non.head(n_non).copy()
        non["stutter_status"] = "nonstuttered"
        non["selection_rank_within_bird"] = np.arange(1, len(non) + 1)
        non = non.drop(columns=["nonstutter_sort_key"], errors="ignore")

        if len(stut) < args.top_n_stuttered:
            print(f"[WARN] {animal}: selected only {len(stut)} stuttered labels.")
        if len(non) < n_non:
            print(f"[WARN] {animal}: selected only {len(non)} non-stuttered labels.")

        pieces.extend([stut, non])

    if not pieces:
        raise ValueError("Could not select any stuttered/non-stuttered labels.")
    plan = pd.concat(pieces, ignore_index=True)
    plan = plan.sort_values(["animal_id", "stutter_status", "selection_rank_within_bird"]).reset_index(drop=True)
    print(f"[INFO] Planned {len(plan)} bird/label analyses.")
    return plan


# -----------------------------
# NPZ loading and segmentation
# -----------------------------


def find_npz_for_animal(npz_root: str | Path, animal_id: str, template: str) -> Path:
    root = Path(npz_root)
    if not root.exists():
        raise FileNotFoundError(f"NPZ root does not exist: {root}")
    pattern = template.format(animal_id=animal_id)
    candidates = sorted(root.rglob(pattern))
    if not candidates:
        candidates = sorted(root.rglob(f"*{animal_id}*.npz"))
    candidates = [p for p in candidates if p.is_file() and not p.name.startswith("._")]
    if not candidates:
        raise FileNotFoundError(f"Could not find an NPZ for {animal_id} under {root}")
    exact = [p for p in candidates if p.name == f"{animal_id}.npz"]
    if exact:
        candidates = exact
    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
    if len(candidates) > 1:
        print(f"[WARN] Multiple NPZ files found for {animal_id}. Using: {candidates[0]}")
    return candidates[0]


def load_and_segment_one_label(cc, args: argparse.Namespace, npz_path: Path, animal_id: str, label: str):
    data = np.load(npz_path, allow_pickle=True)
    s_tx_f = np.asarray(data[args.spec_key], dtype=float)
    labels = cc.normalize_label_array(data[args.label_key])
    file_indices = np.asarray(data[args.file_index_key]).astype(int)
    file_map = cc.get_file_map_dict(data[args.file_map_key])

    if s_tx_f.shape[0] != len(labels) or s_tx_f.shape[0] != len(file_indices):
        raise ValueError(
            "Shape mismatch: spectrogram rows, labels, and file_indices must match. "
            f"spectrogram rows={s_tx_f.shape[0]}, labels={len(labels)}, file_indices={len(file_indices)}"
        )

    unique_file_indices = np.unique(file_indices)
    file_time_lookup = {}
    for fidx in unique_file_indices:
        fname = cc.file_entry_to_name(file_map[int(fidx)])
        file_time_lookup[int(fidx)] = cc.parse_datetime_from_filename(fname)

    treatment_date = cc.load_treatment_date(
        args.metadata_excel_path,
        animal_id,
        metadata_sheet=args.metadata_sheet,
        animal_id_col=args.animal_id_col,
        treatment_date_col=args.treatment_date_col,
    )

    files_by_period = cc.split_by_file_period(
        unique_file_indices,
        file_time_lookup,
        treatment_date,
        include_treatment_day_in_post=args.include_treatment_day_in_post,
        split_mode=args.split_mode,
    )

    print(f"[INFO] Segmenting {animal_id} label {label} with canary method...")

    # The imported single-label script stores animal_id on its argparse Namespace
    # and uses args.animal_id inside segment_cluster_periods(). Because this batch
    # script loops across birds, set it immediately before each segmentation call.
    # Without this, jobs fail with:
    #   'Namespace' object has no attribute 'animal_id'
    args.animal_id = animal_id

    segments_by_period = cc.segment_cluster_periods(
        s_tx_f=s_tx_f,
        labels=labels,
        file_indices=file_indices,
        files_by_period=files_by_period,
        target_label=str(label),
        args=args,
    )

    if args.apply_pitch_contour_filter:
        segments_by_period = cc.apply_pitch_contour_filter_by_period(
            segments_by_period,
            n_points=args.pitch_contour_qc_points,
            filter_sd=args.pitch_contour_filter_sd,
            max_outside_fraction=args.pitch_contour_max_outside_fraction,
        )

    return segments_by_period


# -----------------------------
# Peak SPCC calculation
# -----------------------------


def prepare_spcc_matrices(cc, segments_by_period: dict, args: argparse.Namespace) -> dict:
    """Add a raw, non-time-normalized spectrogram matrix for peak SPCC."""
    for period, entries in segments_by_period.items():
        for ent in entries:
            X = ent.get("X")
            if X is None:
                ent["spcc_usable"] = False
                ent["spcc_matrix"] = None
                continue
            X = cc.crop_frequency_band(
                X,
                max_freq_khz=args.max_freq_khz,
                freq_min_khz=args.corr_freq_min_khz,
                freq_max_khz=args.corr_freq_max_khz,
            )
            Y = cc.transform_spectrogram_for_corr(
                X,
                spec_scale=args.spec_scale,
                corr_transform=args.corr_transform,
            )
            usable = (
                Y.ndim == 2
                and Y.shape[0] >= args.min_segment_bins
                and Y.shape[0] >= args.min_spcc_time_bins
                and Y.shape[1] >= args.min_spcc_freq_bins
                and np.isfinite(Y).sum() >= args.min_finite_pixels
            )
            ent["spcc_matrix"] = Y if usable else None
            ent["spcc_usable"] = bool(usable)
    return segments_by_period


def select_spcc_entries(segments_by_period: dict, periods: list[str], inliers_only: bool = True) -> list[dict]:
    out = []
    for period in periods:
        for ent in segments_by_period.get(period, []):
            if inliers_only and not bool(ent.get("pitch_contour_inlier", True)):
                continue
            if not ent.get("spcc_usable", False):
                continue
            out.append(ent)
    return out


def pearson_corr_flat(a: np.ndarray, b: np.ndarray, min_pixels: int) -> float:
    va = np.asarray(a, dtype=float).ravel()
    vb = np.asarray(b, dtype=float).ravel()
    good = np.isfinite(va) & np.isfinite(vb)
    if int(np.sum(good)) < min_pixels:
        return np.nan
    va = va[good]
    vb = vb[good]
    va = va - np.mean(va)
    vb = vb - np.mean(vb)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.dot(va, vb) / denom)


def peak_spcc(A: np.ndarray, B: np.ndarray, max_shift_bins: int, min_overlap_bins: int, min_overlap_fraction: float, min_pixels: int) -> float:
    """
    Peak spectrographic cross-correlation across small time shifts.

    A and B are time x frequency matrices. For each integer time shift, overlapping
    time rows are flattened and Pearson-correlated. The maximum valid correlation
    across shifts is returned.
    """
    if A is None or B is None:
        return np.nan
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        return np.nan
    TA, TB = A.shape[0], B.shape[0]
    if TA < 2 or TB < 2:
        return np.nan

    min_overlap = max(int(min_overlap_bins), int(math.ceil(float(min_overlap_fraction) * min(TA, TB))))
    best = np.nan

    for shift in range(-int(max_shift_bins), int(max_shift_bins) + 1):
        if shift >= 0:
            a0 = shift
            b0 = 0
            n = min(TA - a0, TB)
        else:
            a0 = 0
            b0 = -shift
            n = min(TA, TB - b0)
        if n < min_overlap:
            continue
        corr = pearson_corr_flat(A[a0:a0 + n, :], B[b0:b0 + n, :], min_pixels=min_pixels)
        if np.isfinite(corr) and (not np.isfinite(best) or corr > best):
            best = corr
    return float(best) if np.isfinite(best) else np.nan


def sample_entries_evenly(entries: list[dict], max_n: Optional[int]) -> list[dict]:
    if max_n is None or len(entries) <= max_n:
        return list(entries)
    idx = np.linspace(0, len(entries) - 1, int(max_n)).round().astype(int)
    idx = np.unique(idx)
    return [entries[int(i)] for i in idx]


def sampled_pair_indices(n_a: int, n_b: Optional[int], within: bool, max_pairs: int, rng: np.random.Generator):
    if within:
        all_i, all_j = np.triu_indices(n_a, k=1)
    else:
        all_i, all_j = np.meshgrid(np.arange(n_a), np.arange(n_b), indexing="ij")
        all_i = all_i.ravel()
        all_j = all_j.ravel()
    n_pairs = len(all_i)
    if n_pairs == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if max_pairs is not None and n_pairs > max_pairs:
        keep = rng.choice(n_pairs, size=int(max_pairs), replace=False)
        all_i = all_i[keep]
        all_j = all_j[keep]
    return all_i.astype(int), all_j.astype(int)


def compute_spcc_values(entries_a: list[dict], entries_b: Optional[list[dict]], args: argparse.Namespace, rng: np.random.Generator, within: bool) -> np.ndarray:
    if within:
        entries = sample_entries_evenly(entries_a, args.pairwise_max_segments_per_group)
        if len(entries) < 2:
            return np.array([], dtype=float)
        ixs, jxs = sampled_pair_indices(len(entries), None, within=True, max_pairs=args.pairwise_max_pairs, rng=rng)
        vals = []
        for i, j in zip(ixs, jxs):
            vals.append(peak_spcc(
                entries[i]["spcc_matrix"],
                entries[j]["spcc_matrix"],
                max_shift_bins=args.max_shift_bins,
                min_overlap_bins=args.min_overlap_bins,
                min_overlap_fraction=args.min_overlap_fraction,
                min_pixels=args.min_finite_pixels,
            ))
    else:
        entries_a = sample_entries_evenly(entries_a, args.cross_condition_max_segments_per_group)
        entries_b = sample_entries_evenly(entries_b or [], args.cross_condition_max_segments_per_group)
        if len(entries_a) == 0 or len(entries_b) == 0:
            return np.array([], dtype=float)
        ixs, jxs = sampled_pair_indices(len(entries_a), len(entries_b), within=False, max_pairs=args.cross_condition_max_pairs, rng=rng)
        vals = []
        for i, j in zip(ixs, jxs):
            vals.append(peak_spcc(
                entries_a[i]["spcc_matrix"],
                entries_b[j]["spcc_matrix"],
                max_shift_bins=args.max_shift_bins,
                min_overlap_bins=args.min_overlap_bins,
                min_overlap_fraction=args.min_overlap_fraction,
                min_pixels=args.min_finite_pixels,
            ))
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def summarize_values(vals: np.ndarray, prefix: str) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q90": np.nan,
        }
    return {
        f"{prefix}_n": int(vals.size),
        f"{prefix}_mean": float(np.nanmean(vals)),
        f"{prefix}_median": float(np.nanmedian(vals)),
        f"{prefix}_q10": float(np.nanpercentile(vals, 10)),
        f"{prefix}_q90": float(np.nanpercentile(vals, 90)),
    }


def ecdf_xy(vals: np.ndarray):
    x = np.sort(np.asarray(vals, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x, np.array([], dtype=float)
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def plot_one_label_spcc(prepre: np.ndarray, postpost: np.ndarray, prepost: np.ndarray, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for vals, label in [
        (prepre, "pre-pre within"),
        (postpost, "post-post within"),
        (prepost, "pre-post cross"),
    ]:
        x, y = ecdf_xy(vals)
        if x.size:
            ax.step(x, y, where="post", lw=2, label=f"{label} (N={x.size:,})")
    ax.set_xlabel("Peak spectrographic cross-correlation")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Run one label and batch
# -----------------------------


def run_one_label(cc, args: argparse.Namespace, rec: pd.Series, out_dir: Path) -> dict:
    animal_id = str(rec["animal_id"])
    label = label_to_str(rec["cluster_label"])
    status = str(rec["stutter_status"])
    rng = np.random.default_rng(args.random_seed + abs(hash((animal_id, label))) % 1_000_000)

    npz_path = find_npz_for_animal(args.npz_root, animal_id, args.npz_glob_template)
    label_dir = ensure_dir(out_dir / animal_id / status / f"label_{safe_name(label)}")
    summary_path = label_dir / f"{animal_id}_label{label}_peak_spcc_summary.csv"

    base = {
        "animal_id": animal_id,
        "cluster_label": label,
        "stutter_status": status,
        "selection_rank_within_bird": rec.get("selection_rank_within_bird", np.nan),
        "phrase_variance_delta_ms2": rec.get("phrase_variance_delta_ms2", np.nan),
        "pre_n_phrases_from_duration_csv": rec.get("pre_n_phrases", np.nan),
        "post_n_phrases_from_duration_csv": rec.get("post_n_phrases", np.nan),
        "npz_path": str(npz_path),
        "out_dir": str(label_dir),
    }

    if args.skip_existing and summary_path.exists():
        prev = pd.read_csv(summary_path)
        if not prev.empty:
            row = prev.iloc[0].to_dict()
            row["run_status"] = "skipped_existing"
            return row

    if args.dry_run:
        base["run_status"] = "dry_run"
        return base

    try:
        segments_by_period = load_and_segment_one_label(cc, args, npz_path, animal_id, label)
        segments_by_period = prepare_spcc_matrices(cc, segments_by_period, args)

        inliers_only = args.apply_pitch_contour_filter
        pre_entries = select_spcc_entries(segments_by_period, ["early_pre", "late_pre"], inliers_only=inliers_only)
        post_entries = select_spcc_entries(segments_by_period, ["early_post", "late_post"], inliers_only=inliers_only)

        period_counts = {
            f"n_segments_{period}": len(select_spcc_entries(segments_by_period, [period], inliers_only=inliers_only))
            for period in ["early_pre", "late_pre", "early_post", "late_post"]
        }
        base.update(period_counts)
        base["n_segments_combined_pre"] = len(pre_entries)
        base["n_segments_combined_post"] = len(post_entries)

        prepre = compute_spcc_values(pre_entries, None, args, rng, within=True)
        postpost = compute_spcc_values(post_entries, None, args, rng, within=True)
        prepost = compute_spcc_values(pre_entries, post_entries, args, rng, within=False)

        # Save pair values.
        pair_rows = []
        for group_name, vals in [
            ("pre_pre_within", prepre),
            ("post_post_within", postpost),
            ("pre_post_cross", prepost),
        ]:
            for v in vals:
                pair_rows.append({
                    "animal_id": animal_id,
                    "cluster_label": label,
                    "stutter_status": status,
                    "pair_group": group_name,
                    "peak_spcc": float(v),
                })
        pd.DataFrame(pair_rows).to_csv(label_dir / f"{animal_id}_label{label}_peak_spcc_pair_values.csv", index=False)

        base.update(summarize_values(prepre, "prepre_spcc"))
        base.update(summarize_values(postpost, "postpost_spcc"))
        base.update(summarize_values(prepost, "prepost_spcc"))
        base["stereotypy_delta_post_minus_pre"] = base["postpost_spcc_median"] - base["prepre_spcc_median"]
        base["identity_shift_delta_prepost_minus_prepre"] = base["prepost_spcc_median"] - base["prepre_spcc_median"]
        base["postpost_minus_prepost"] = base["postpost_spcc_median"] - base["prepost_spcc_median"]
        base["max_shift_bins"] = int(args.max_shift_bins)
        base["max_shift_ms"] = float(args.max_shift_bins * args.bin_ms)
        base["corr_transform"] = args.corr_transform
        base["corr_freq_min_khz"] = args.corr_freq_min_khz
        base["corr_freq_max_khz"] = args.corr_freq_max_khz
        base["run_status"] = "completed"

        pd.DataFrame([base]).to_csv(summary_path, index=False)
        plot_one_label_spcc(
            prepre,
            postpost,
            prepost,
            label_dir / f"{animal_id}_label{label}_peak_spcc_ecdf.png",
            title=f"{animal_id} label {label} ({status})\npeak SPCC distributions",
        )
        print(f"[DONE] {animal_id} label {label} ({status})")
        return base

    except Exception as exc:
        base["run_status"] = "failed"
        base["error"] = str(exc)
        print(f"[ERROR] {animal_id} label {label} ({status}): {exc}")
        if args.stop_on_error:
            raise
        return base


# -----------------------------
# Group-level stats and plots
# -----------------------------


def wilcoxon_less(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not SCIPY_AVAILABLE or vals.size < 3 or np.allclose(vals, 0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(vals, alternative="less")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def make_group_stats(summary: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "stereotypy_delta_post_minus_pre",
        "identity_shift_delta_prepost_minus_prepre",
    ]
    ok = summary[summary["run_status"].isin(["completed", "skipped_existing"])].copy()
    for m in metrics:
        ok[m] = pd.to_numeric(ok[m], errors="coerce")

    bird_level = (
        ok.groupby(["animal_id", "stutter_status"], as_index=False)[metrics]
        .mean()
    )
    bird_level.to_csv(out_dir / "bird_level_stuttered_vs_nonstuttered_peak_spcc.csv", index=False)

    rows = []
    for metric in metrics:
        for status in ["stuttered", "nonstuttered"]:
            vals = bird_level.loc[bird_level["stutter_status"] == status, metric].dropna().values
            w, p = wilcoxon_less(vals)
            rows.append({
                "comparison": f"{status}: delta < 0",
                "metric": metric,
                "n_birds": int(vals.size),
                "mean_delta": float(np.nanmean(vals)) if vals.size else np.nan,
                "median_delta": float(np.nanmedian(vals)) if vals.size else np.nan,
                "wilcoxon_statistic": w,
                "wilcoxon_p_one_sided_less": p,
            })

        wide = bird_level.pivot(index="animal_id", columns="stutter_status", values=metric)
        if "stuttered" in wide.columns and "nonstuttered" in wide.columns:
            wide = wide.dropna(subset=["stuttered", "nonstuttered"])
            diff = (wide["stuttered"] - wide["nonstuttered"]).values
            w, p = wilcoxon_less(diff)
            rows.append({
                "comparison": "stuttered - nonstuttered < 0",
                "metric": metric,
                "n_birds": int(np.isfinite(diff).sum()),
                "mean_delta": float(np.nanmean(diff)) if diff.size else np.nan,
                "median_delta": float(np.nanmedian(diff)) if diff.size else np.nan,
                "wilcoxon_statistic": w,
                "wilcoxon_p_one_sided_less": p,
            })

    tests = pd.DataFrame(rows)
    tests.to_csv(out_dir / "group_level_peak_spcc_wilcoxon_tests.csv", index=False)
    return bird_level, tests


def plot_paired_bird_delta(bird_level: pd.DataFrame, metric: str, out_png: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    xmap = {"nonstuttered": 0, "stuttered": 1}
    for animal, sub in bird_level.groupby("animal_id"):
        sub = sub[sub["stutter_status"].isin(["nonstuttered", "stuttered"])].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        if sub[metric].notna().sum() == 0:
            continue
        xs = [xmap[s] for s in sub["stutter_status"]]
        ys = sub[metric].values
        if len(xs) == 2:
            ax.plot(xs, ys, alpha=0.45, lw=1)
        ax.scatter(xs, ys, s=45)
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                ax.text(x + 0.03, y, animal, fontsize=8, va="center")
    ax.axhline(0, ls="--", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-stuttered", "Stuttered"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")

    # If there are real finite values, zoom the y-axis around them instead of
    # leaving matplotlib's empty/default 0-1 axis.
    yvals = pd.to_numeric(bird_level[metric], errors="coerce").dropna().values
    if yvals.size > 0:
        ymin = min(float(np.nanmin(yvals)), 0.0)
        ymax = max(float(np.nanmax(yvals)), 0.0)
        pad = max(0.02, 0.12 * (ymax - ymin if ymax > ymin else 0.1))
        ax.set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_vs_delta(summary: pd.DataFrame, metric: str, out_png: Path, title: str, ylabel: str):
    ok = summary[summary["run_status"].isin(["completed", "skipped_existing"])].copy()
    ok[metric] = pd.to_numeric(ok[metric], errors="coerce")
    ok["phrase_variance_delta_ms2"] = pd.to_numeric(ok["phrase_variance_delta_ms2"], errors="coerce")
    ok = ok[np.isfinite(ok[metric]) & np.isfinite(ok["phrase_variance_delta_ms2"]) & (ok["phrase_variance_delta_ms2"] > 0)]
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for status, sub in ok.groupby("stutter_status"):
        ax.scatter(sub["phrase_variance_delta_ms2"], sub[metric], s=55, label=status, alpha=0.8)
        for _, r in sub.iterrows():
            ax.text(r["phrase_variance_delta_ms2"], r[metric], f"{r['animal_id']}:{r['cluster_label']}", fontsize=8)
    ax.set_xscale("log")
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel("Post - pre phrase-duration variance change (ms²)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(summary: pd.DataFrame, bird_level: pd.DataFrame, out_dir: Path):
    plot_paired_bird_delta(
        bird_level,
        metric="stereotypy_delta_post_minus_pre",
        out_png=out_dir / "bird_level_stereotypy_delta_peak_spcc_stuttered_vs_nonstuttered.png",
        title="Within-condition stereotypy change by bird",
        ylabel="Δ median peak SPCC: post-post - pre-pre",
    )
    plot_paired_bird_delta(
        bird_level,
        metric="identity_shift_delta_prepost_minus_prepre",
        out_png=out_dir / "bird_level_identity_shift_peak_spcc_stuttered_vs_nonstuttered.png",
        title="Pre/post identity shift by bird",
        ylabel="Δ median peak SPCC: pre-post - pre-pre",
    )
    plot_variance_vs_delta(
        summary,
        metric="stereotypy_delta_post_minus_pre",
        out_png=out_dir / "phrase_variance_delta_vs_stereotypy_delta_peak_spcc.png",
        title="Do high variance-increase syllables lose within-condition stereotypy?",
        ylabel="Δ median peak SPCC: post-post - pre-pre",
    )
    plot_variance_vs_delta(
        summary,
        metric="identity_shift_delta_prepost_minus_prepre",
        out_png=out_dir / "phrase_variance_delta_vs_identity_shift_peak_spcc.png",
        title="Do high variance-increase syllables shift away from pre-lesion form?",
        ylabel="Δ median peak SPCC: pre-post - pre-pre",
    )


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare peak SPCC for stuttered vs non-stuttered syllables in Medial + Lateral lesion birds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core paths.
    p.add_argument("--crosscorr-script", "--segmentation-script", dest="crosscorr_script", required=True,
                   help="Path to cluster_spectrogram_crosscorr_canary_segmentation_v1.py. Used as a segmentation function library.")
    p.add_argument("--metadata-excel-path", required=True)
    p.add_argument("--phrase-duration-csv", required=True)
    p.add_argument("--npz-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--npz-glob-template", default="{animal_id}.npz")

    # Metadata selection.
    p.add_argument("--metadata-sheet", default="metadata_with_hit_type")
    p.add_argument("--animal-id-col", default="Animal ID")
    p.add_argument("--treatment-date-col", default="Treatment date")
    p.add_argument("--hit-type-cols", nargs="*", default=None,
                   help="Metadata column(s) used to select Medial + Lateral birds, e.g. 'Lesion hit type'.")
    p.add_argument("--hit-type-regex", default=r"medial.*lateral|lateral.*medial",
                   help="Regex applied to hit-type text. Default selects visible medial+lateral labels, not generic 'large lesion' labels.")
    p.add_argument("--also-accept-both-hit-flags", action="store_true",
                   help="Also include rows where both medial and lateral hit flag columns are truthy.")
    p.add_argument("--medial-hit-flag-col", default=None)
    p.add_argument("--lateral-hit-flag-col", default=None)
    p.add_argument("--visible-only", action="store_true",
                   help="Require Area X visible in histology column to be truthy.")
    p.add_argument("--visible-col", default=None)

    # Phrase duration / label selection.
    p.add_argument("--phrase-animal-id-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--period-col", default=None)
    p.add_argument("--variance-delta-col", default=None)
    p.add_argument("--pre-n-col", default=None)
    p.add_argument("--post-n-col", default=None)
    p.add_argument("--top-n-stuttered", type=int, default=3)
    p.add_argument("--top-n-nonstuttered", type=int, default=3)
    p.add_argument("--min-pre-phrases", type=int, default=20)
    p.add_argument("--min-post-phrases", type=int, default=20)
    p.add_argument("--min-stutter-variance-delta", type=float, default=0.0)
    p.add_argument("--allow-nonpositive-stuttered", action="store_true")
    p.add_argument("--nonstutter-mode", default="near_zero", choices=["near_zero", "lowest", "lowest_positive"],
                   help="How to choose non-stuttered labels after excluding stuttered labels.")
    p.add_argument("--nonstutter-max-variance-delta", type=float, default=None,
                   help="Optional maximum allowed variance delta for non-stuttered labels.")

    # NPZ keys.
    p.add_argument("--spec-key", default="s")
    p.add_argument("--label-key", default="hdbscan_labels")
    p.add_argument("--file-index-key", default="file_indices")
    p.add_argument("--file-map-key", default="file_map")

    # Timing/splitting.
    p.add_argument("--bin-ms", type=float, default=2.7)
    p.add_argument("--include-treatment-day-in-post", action="store_true")
    p.add_argument("--split-mode", default="file_half", choices=["file_half", "file_median"])

    # Canary segmentation args. These mirror the existing single-label script.
    p.add_argument("--max-freq-khz", type=float, default=12.0)
    p.add_argument("--spec-scale", default="linear", choices=["linear", "log10", "loge", "shift"])
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

    # Pitch contour QC.
    p.add_argument("--apply-pitch-contour-filter", action="store_true", default=True)
    p.add_argument("--no-pitch-contour-filter", dest="apply_pitch_contour_filter", action="store_false")
    p.add_argument("--pitch-contour-qc-points", type=int, default=101)
    p.add_argument("--pitch-contour-filter-sd", type=float, default=2.0)
    p.add_argument("--pitch-contour-max-outside-fraction", type=float, default=0.05)

    # SPCC parameters.
    p.add_argument("--corr-freq-min-khz", type=float, default=0.8)
    p.add_argument("--corr-freq-max-khz", type=float, default=10.0)
    p.add_argument("--corr-transform", default="log_power", choices=["raw", "linear_power", "log_power"])
    p.add_argument("--max-shift-ms", type=float, default=30.0,
                   help="Maximum temporal shift allowed for peak SPCC, in ms.")
    p.add_argument("--max-shift-bins", type=int, default=None,
                   help="Override --max-shift-ms with an exact number of bins.")
    p.add_argument("--min-overlap-bins", type=int, default=5)
    p.add_argument("--min-overlap-fraction", type=float, default=0.80,
                   help="Minimum overlap as a fraction of the shorter segment.")
    p.add_argument("--min-spcc-time-bins", type=int, default=5)
    p.add_argument("--min-spcc-freq-bins", type=int, default=5)
    p.add_argument("--min-finite-pixels", type=int, default=25)
    p.add_argument("--pairwise-max-segments-per-group", type=int, default=300)
    p.add_argument("--pairwise-max-pairs", type=int, default=50000)
    p.add_argument("--cross-condition-max-segments-per-group", type=int, default=250)
    p.add_argument("--cross-condition-max-pairs", type=int, default=50000)

    # Run behavior.
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)

    args = p.parse_args()
    if args.max_shift_bins is None:
        args.max_shift_bins = max(0, int(round(args.max_shift_ms / args.bin_ms)))
    return args


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    cc = load_python_module(args.crosscorr_script)
    if not getattr(cc, "SCIPY_AVAILABLE", False):
        raise ImportError("The imported segmentation script reports that scipy is unavailable. Install scipy first.")

    selected_meta, selected_animals = read_metadata_animals(args)
    selected_meta.to_csv(out_dir / "selected_medial_lateral_animals_for_peak_spcc.csv", index=False)

    plan = select_stuttered_and_nonstuttered_labels(args, selected_animals)
    plan.to_csv(out_dir / "stuttered_and_nonstuttered_syllables_to_analyze.csv", index=False)

    run_rows = []
    for _, rec in plan.iterrows():
        print("\n" + "=" * 80)
        print(f"[JOB] {rec['animal_id']} label {rec['cluster_label']} ({rec['stutter_status']})")
        row = run_one_label(cc, args, rec, out_dir)
        run_rows.append(row)

    summary = pd.DataFrame(run_rows)
    summary.to_csv(out_dir / "all_birds_stuttered_vs_nonstuttered_peak_spcc_summary.csv", index=False)

    if not args.dry_run:
        bird_level, tests = make_group_stats(summary, out_dir)
        make_plots(summary, bird_level, out_dir)
        print("\n[STATS]")
        print(tests.to_string(index=False))

    readme = out_dir / "README_peak_spcc_stuttered_vs_nonstuttered.txt"
    readme.write_text(
        "Peak SPCC stuttered vs non-stuttered analysis\n"
        "=============================================\n\n"
        "Key output files:\n"
        "- selected_medial_lateral_animals_for_peak_spcc.csv\n"
        "- stuttered_and_nonstuttered_syllables_to_analyze.csv\n"
        "- all_birds_stuttered_vs_nonstuttered_peak_spcc_summary.csv\n"
        "- bird_level_stuttered_vs_nonstuttered_peak_spcc.csv\n"
        "- group_level_peak_spcc_wilcoxon_tests.csv\n\n"
        "Important columns in the summary:\n"
        "- prepre_spcc_median: median peak SPCC among pre-pre pairs\n"
        "- postpost_spcc_median: median peak SPCC among post-post pairs\n"
        "- prepost_spcc_median: median peak SPCC among pre-post pairs\n"
        "- stereotypy_delta_post_minus_pre = postpost_spcc_median - prepre_spcc_median\n"
        "  Negative values mean post-lesion renditions are less internally similar than pre-lesion renditions.\n"
        "- identity_shift_delta_prepost_minus_prepre = prepost_spcc_median - prepre_spcc_median\n"
        "  Negative values mean post-lesion renditions are less similar to pre-lesion renditions than pre renditions are to each other.\n\n"
        "Statistics are run on bird-level averages, not individual pairwise correlations.\n"
    )

    print("\n[DONE]")
    print(f"[SAVED] {out_dir / 'stuttered_and_nonstuttered_syllables_to_analyze.csv'}")
    print(f"[SAVED] {out_dir / 'all_birds_stuttered_vs_nonstuttered_peak_spcc_summary.csv'}")


if __name__ == "__main__":
    main()
