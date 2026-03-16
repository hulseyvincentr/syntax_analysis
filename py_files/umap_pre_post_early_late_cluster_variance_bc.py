#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_pre_post_early_late_cluster_variance_bc.py
%run /Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/umap_pre_post_early_late_cluster_variance_bc.py --npz-path "/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443.npz" --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/umap_pre_post_early_late_cluster_variance_bc" --array-key "predictions" --cluster-key "hdbscan_labels" --file-key "file_indices" --min-points-per-period 200 --max-points-per-period 20000 --plot-max-points-per-period 5000 --n-neighbors 30 --min-dist 0.1 --metric "euclidean" --bc-bins 100 --seed -1

Version A + cluster variance / mean-shift summary.

For ONE bird NPZ:
  1) Load high-dimensional features (default: predictions) + HDBSCAN clusters + file_indices + file_map
  2) Split time bins into pre- vs post-treatment using the treatment date from metadata XLSX
  3) Within PRE, split files into early vs late
  4) Within POST, split files into early vs late
  5) For each cluster:
       - fit ONE UMAP on ALL available points in that cluster
       - select balanced subsets AFTER UMAP fitting for:
           * pre early vs pre late
           * post early vs post late
           * total pre vs total post
       - compute Bhattacharyya coefficients in that shared UMAP space:
           * bc_pre_early_vs_late
           * bc_post_early_vs_late
           * bc_pre_vs_post
       - compute pre/post centroid shift and spread change in the ORIGINAL latent space:
           * centroid_shift_raw
           * pre_rms_radius_raw
           * post_rms_radius_raw
           * post_over_pre_rms_radius
           * pre_trace_cov_raw
           * post_trace_cov_raw
           * post_over_pre_trace_cov
       - mark whether the cluster:
           * appeared after treatment
           * disappeared after treatment
           * is present both pre and post
           * is present but too sparse for some BCs
       - make one 2x3 figure:
           row 1 = pre early / pre late / pre overlap
           row 2 = post early / post late / post overlap

Outputs:
  out_dir/
    clusters/
      <prefix>_cluster<id>_pre_post_early_late_umap.png
    <prefix>_cluster_variance_bc_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import argparse
import datetime as _dt

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_umap():
    try:
        import umap.umap_ as umap  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: umap-learn.\n"
            "Install in your conda env with ONE of:\n"
            "  conda install -c conda-forge umap-learn\n"
            "  pip install umap-learn\n"
            f"\nOriginal import error: {repr(e)}"
        )
    return umap


@dataclass(frozen=True)
class UMAPEarlyLateConfig:
    npz_path: Path
    metadata_xlsx: Path

    metadata_sheet_treatment: str = "metadata"
    treatment_date_col: str = "Treatment date"
    animal_id_col_candidates: Tuple[str, ...] = ("Animal ID", "animal_id", "animal id", "Bird", "bird id")

    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"
    file_map_key: str = "file_map"
    vocalization_key: str = "vocalization"
    vocalization_only: bool = True

    include_noise: bool = False
    only_cluster_id: Optional[int] = None

    min_points_per_period: int = 200
    max_points_per_period: Optional[int] = None
    max_points_per_period_for_plot: Optional[int] = None

    exclude_treatment_day_from_post: bool = False
    early_late_split_method: str = "file_median"  # file_median | file_half
    random_seed: int = 0

    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"

    out_dir: Optional[Path] = None
    out_prefix: Optional[str] = None
    dpi: int = 200
    bc_bins: int = 100


_EXCEL_ORIGIN = _dt.datetime(1899, 12, 30)


def _unwrap_file_map_value(v: Any) -> str:
    x = v
    for _ in range(4):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        else:
            break
    return str(x)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _parse_excel_serial_days(token: str) -> Optional[_dt.datetime]:
    try:
        x = float(token)
    except Exception:
        return None
    if not (10000.0 <= x <= 80000.0):
        return None
    try:
        return _EXCEL_ORIGIN + _dt.timedelta(days=float(x))
    except Exception:
        return None


def parse_datetime_from_filename(file_path: str | Path, *, year_default: int = 2024) -> Optional[_dt.datetime]:
    base = Path(file_path).name
    parts = base.split("_")

    if len(parts) >= 2:
        dt = _parse_excel_serial_days(parts[1])
        if dt is not None:
            return dt

    year = year_default
    for p in parts:
        if p.isdigit() and len(p) == 4:
            y = int(p)
            if 1990 <= y <= 2100:
                year = y
                break

    try:
        month, day, hour, minute, second = map(int, parts[2:7])
        return _dt.datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def _build_file_datetime_map(file_map_obj: Any, *, year_default: int) -> Dict[int, _dt.datetime]:
    fm = file_map_obj
    if isinstance(fm, np.ndarray) and fm.shape == ():
        try:
            fm = fm.item()
        except Exception:
            pass

    if isinstance(fm, dict):
        mapping = {int(k): v for k, v in fm.items()}
    elif isinstance(fm, (list, tuple)):
        mapping = {int(i): v for i, v in enumerate(fm)}
    else:
        raise ValueError(f"Unexpected file_map structure: {type(fm)}")

    dt_map: Dict[int, _dt.datetime] = {}
    for idx, v in mapping.items():
        fp = _unwrap_file_map_value(v)
        dt = parse_datetime_from_filename(fp, year_default=year_default)
        if dt is not None:
            dt_map[int(idx)] = dt
    return dt_map


def _split_pre_post_masks(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    treatment_date: _dt.datetime,
    *,
    exclude_treatment_day_from_post: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    fi = file_indices.astype(int)
    max_idx = int(fi.max()) if fi.size else 0

    file_dt_arr = np.array([None] * (max_idx + 1), dtype=object)
    for idx, dt in file_dt_map.items():
        if 0 <= idx <= max_idx:
            file_dt_arr[idx] = dt

    point_dt = file_dt_arr[fi]

    t0 = _dt.datetime.combine(treatment_date.date(), _dt.time(0, 0, 0))
    t1 = _dt.datetime.combine(treatment_date.date(), _dt.time(23, 59, 59, 999999))

    pre = np.array([(d is not None and d < t0) for d in point_dt], dtype=bool)
    if exclude_treatment_day_from_post:
        post = np.array([(d is not None and d > t1) for d in point_dt], dtype=bool)
    else:
        post = np.array([(d is not None and d >= t0) for d in point_dt], dtype=bool)

    return pre, post


def _split_early_late_masks_by_file(
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
    period_mask: np.ndarray,
    *,
    method: str = "file_median",
) -> Tuple[np.ndarray, np.ndarray]:
    fi = file_indices.astype(int)
    files = np.unique(fi[period_mask])

    file_dts: List[Tuple[int, _dt.datetime]] = []
    for f in files:
        dt = file_dt_map.get(int(f), None)
        if dt is not None:
            file_dts.append((int(f), dt))

    if len(file_dts) < 2:
        return period_mask.copy(), np.zeros_like(period_mask, dtype=bool)

    file_dts.sort(key=lambda t: t[1])

    method = (method or "file_median").lower()
    if method == "file_half":
        mid = len(file_dts) // 2
        early_files = {f for f, _ in file_dts[:mid]}
        late_files = {f for f, _ in file_dts[mid:]}
    else:
        ts = np.array([dt.timestamp() for _, dt in file_dts], dtype=float)
        med = float(np.median(ts))
        early_files = {f for (f, dt) in file_dts if dt.timestamp() <= med}
        late_files = {f for (f, dt) in file_dts if dt.timestamp() > med}
        if len(late_files) == 0:
            mid = len(file_dts) // 2
            early_files = {f for f, _ in file_dts[:mid]}
            late_files = {f for f, _ in file_dts[mid:]}

    early_mask = period_mask & np.isin(fi, np.fromiter(early_files, dtype=int))
    late_mask = period_mask & np.isin(fi, np.fromiter(late_files, dtype=int))
    return early_mask, late_mask


def _treatment_dt_from_any(value: Any) -> Optional[_dt.datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime.combine(value, _dt.time(0, 0, 0))
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
    except Exception:
        pass
    try:
        return _dt.datetime.fromisoformat(str(value))
    except Exception:
        return None


def load_treatment_date_for_animal(
    metadata_xlsx: Path,
    animal_id: str,
    *,
    sheet_name: str,
    treatment_date_col: str,
    animal_id_col_candidates: Tuple[str, ...],
) -> _dt.datetime:
    df = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)

    animal_col = None
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in animal_id_col_candidates:
        key = cand.strip().lower()
        if key in lower_cols:
            animal_col = lower_cols[key]
            break
    if animal_col is None:
        animal_col = df.columns[0]

    treat_col = None
    for c in df.columns:
        if treatment_date_col.strip().lower() == str(c).strip().lower():
            treat_col = c
            break
    if treat_col is None:
        for c in df.columns:
            if "treatment" in str(c).lower() and "date" in str(c).lower():
                treat_col = c
                break
    if treat_col is None:
        raise KeyError(
            f"Could not find treatment date column like {treatment_date_col!r} in sheet {sheet_name!r}"
        )

    sub = df[df[animal_col].astype(str).str.strip() == str(animal_id).strip()]
    if len(sub) == 0:
        raise KeyError(f"Animal ID {animal_id!r} not found in metadata sheet {sheet_name!r}")

    td_series = sub[treat_col].dropna()
    if len(td_series) == 0:
        raise ValueError(f"Treatment date missing for animal {animal_id!r} in metadata sheet {sheet_name!r}")

    td = _treatment_dt_from_any(td_series.iloc[0])
    if td is None:
        raise ValueError(f"Could not parse treatment date value for animal {animal_id!r}: {td_series.iloc[0]!r}")

    return td


def _balanced_subsample_pair(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    rng: np.random.Generator,
    cap: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    n_a_raw = int(idx_a.size)
    n_b_raw = int(idx_b.size)
    n_target = min(n_a_raw, n_b_raw)
    if cap is not None:
        n_target = min(n_target, int(cap))

    if n_target <= 0:
        return idx_a[:0], idx_b[:0], 0, n_a_raw, n_b_raw

    if idx_a.size > n_target:
        idx_a = rng.choice(idx_a, size=n_target, replace=False)
    if idx_b.size > n_target:
        idx_b = rng.choice(idx_b, size=n_target, replace=False)

    return idx_a, idx_b, int(n_target), n_a_raw, n_b_raw


def _format_date_range(dts: List[_dt.datetime]) -> str:
    if not dts:
        return "(no dates)"
    return f"{min(dts).date().isoformat()} to {max(dts).date().isoformat()}"


def _dates_for_indices(
    point_indices: np.ndarray,
    file_indices: np.ndarray,
    file_dt_map: Dict[int, _dt.datetime],
) -> List[_dt.datetime]:
    dts: List[_dt.datetime] = []
    for fi in file_indices[point_indices]:
        dt = file_dt_map.get(int(fi))
        if dt is not None:
            dts.append(dt)
    return dts


def bhattacharyya_coefficient_2d_hist(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    *,
    bins: int = 100,
    padding_frac: float = 0.02,
) -> float:
    if xy_a.size == 0 or xy_b.size == 0:
        return float("nan")

    xy_all = np.vstack([xy_a, xy_b])
    mins = np.min(xy_all, axis=0)
    maxs = np.max(xy_all, axis=0)
    spans = (maxs - mins) * float(padding_frac) + 1e-9

    x_range = (float(mins[0] - spans[0]), float(maxs[0] + spans[0]))
    y_range = (float(mins[1] - spans[1]), float(maxs[1] + spans[1]))

    Ha, _, _ = np.histogram2d(xy_a[:, 0], xy_a[:, 1], bins=bins, range=[x_range, y_range])
    Hb, _, _ = np.histogram2d(xy_b[:, 0], xy_b[:, 1], bins=bins, range=[x_range, y_range])

    Pa = Ha.astype(float)
    Pb = Hb.astype(float)

    sa = Pa.sum()
    sb = Pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")

    Pa /= sa
    Pb /= sb

    bc = float(np.sum(np.sqrt(Pa * Pb)))
    return float(np.clip(bc, 0.0, 1.0))


def _make_umap_reducer(umap, cfg: UMAPEarlyLateConfig):
    return umap.UMAP(
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        random_state=(None if int(cfg.random_seed) < 0 else int(cfg.random_seed)),
    )


def _style_overlap_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def _plot_row(
    axes_row,
    *,
    xy: np.ndarray,
    is_late: np.ndarray,
    period_name: str,
    early_range: str,
    late_range: str,
    n_early: int,
    n_late: int,
    bc: float,
    early_color: str,
    late_color: str,
) -> None:
    early_xy = xy[~is_late]
    late_xy = xy[is_late]

    ax = axes_row[0]
    ax.set_title(f"{period_name} early\n{early_range}\nN={n_early}")
    ax.scatter(early_xy[:, 0], early_xy[:, 1], s=8, alpha=0.45, c=early_color, edgecolors="none")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(False)

    ax = axes_row[1]
    ax.set_title(f"{period_name} late\n{late_range}\nN={n_late}")
    ax.scatter(late_xy[:, 0], late_xy[:, 1], s=8, alpha=0.45, c=late_color, edgecolors="none")
    ax.set_xlabel("UMAP Dimension 1")
    ax.grid(False)

    ax = axes_row[2]
    ax.set_title(f"{period_name} overlap\nBC={bc:.3f}")
    _style_overlap_axis(ax)
    ax.set_xlabel("UMAP Dimension 1")
    ax.scatter(early_xy[:, 0], early_xy[:, 1], s=8, alpha=0.45, c=early_color, edgecolors="none")
    ax.scatter(late_xy[:, 0], late_xy[:, 1], s=8, alpha=0.45, c=late_color, edgecolors="none")
    ax.grid(False)


def _mark_row_skipped(axes_row, message: str, *, ylabel: str) -> None:
    for j, ax in enumerate(axes_row):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _centroid_and_spread_metrics(Xg: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      mu: centroid vector
      rms_radius: sqrt(mean(||x-mu||^2))
      trace_cov: trace of covariance matrix
    """
    if Xg.ndim != 2 or Xg.shape[0] == 0:
        return np.array([], dtype=float), float("nan"), float("nan")

    mu = np.mean(Xg, axis=0)
    centered = Xg - mu
    sq_dists = np.sum(centered * centered, axis=1)
    rms_radius = float(np.sqrt(np.mean(sq_dists))) if sq_dists.size > 0 else float("nan")

    if Xg.shape[0] >= 2:
        trace_cov = float(np.trace(np.cov(Xg, rowvar=False, ddof=1)))
    else:
        trace_cov = float("nan")

    return mu, rms_radius, trace_cov


def _cluster_change_label(pre_present_any: bool, post_present_any: bool) -> str:
    if pre_present_any and post_present_any:
        return "present_pre_and_post"
    if pre_present_any and not post_present_any:
        return "disappeared_after_treatment"
    if post_present_any and not pre_present_any:
        return "appeared_after_treatment"
    return "absent_both"


def run_one_bird(cfg: UMAPEarlyLateConfig) -> Path:
    umap = _import_umap()

    npz_path = Path(cfg.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    animal_id = _infer_animal_id(npz_path)
    treatment_dt = load_treatment_date_for_animal(
        metadata_xlsx=Path(cfg.metadata_xlsx),
        animal_id=animal_id,
        sheet_name=cfg.metadata_sheet_treatment,
        treatment_date_col=cfg.treatment_date_col,
        animal_id_col_candidates=cfg.animal_id_col_candidates,
    )

    data = np.load(npz_path, allow_pickle=True)
    needed = [cfg.array_key, cfg.cluster_key, cfg.file_key, cfg.file_map_key]
    if cfg.vocalization_only:
        needed.append(cfg.vocalization_key)
    missing = [k for k in needed if k not in data]
    if missing:
        raise KeyError(f"NPZ missing keys {missing}. Present keys: {list(data.keys())}")

    X = np.asarray(data[cfg.array_key])
    clusters = np.asarray(data[cfg.cluster_key]).astype(int)
    file_indices = np.asarray(data[cfg.file_key]).astype(int)
    file_map_obj = data[cfg.file_map_key]

    if X.ndim != 2:
        raise ValueError(f"{cfg.array_key} must be 2D. Got shape {X.shape}")
    if not (len(X) == len(clusters) == len(file_indices)):
        raise ValueError("Length mismatch among X, clusters, file_indices")

    base_mask = np.ones(len(X), dtype=bool)
    if cfg.vocalization_only and cfg.vocalization_key in data:
        voc = np.asarray(data[cfg.vocalization_key])
        try:
            base_mask &= (voc.astype(int) == 1)
        except Exception:
            pass

    file_dt_map = _build_file_datetime_map(file_map_obj, year_default=int(treatment_dt.year))

    pre_mask, post_mask = _split_pre_post_masks(
        file_indices=file_indices,
        file_dt_map=file_dt_map,
        treatment_date=treatment_dt,
        exclude_treatment_day_from_post=cfg.exclude_treatment_day_from_post,
    )
    pre_mask &= base_mask
    post_mask &= base_mask

    pre_early_mask, pre_late_mask = _split_early_late_masks_by_file(
        file_indices, file_dt_map, pre_mask, method=cfg.early_late_split_method
    )
    post_early_mask, post_late_mask = _split_early_late_masks_by_file(
        file_indices, file_dt_map, post_mask, method=cfg.early_late_split_method
    )

    out_dir = Path(cfg.out_dir) if cfg.out_dir is not None else (npz_path.parent / f"umap_pre_post_early_late_{animal_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.out_prefix if cfg.out_prefix is not None else f"{animal_id}_{cfg.array_key}"

    unique_labels = np.unique(clusters)
    if not cfg.include_noise:
        unique_labels = unique_labels[unique_labels != -1]
    if cfg.only_cluster_id is not None:
        if int(cfg.only_cluster_id) not in set(map(int, unique_labels)):
            raise ValueError(
                f"--cluster-id {cfg.only_cluster_id} not found in labels (after include_noise={cfg.include_noise})."
            )
        unique_labels = np.array([int(cfg.only_cluster_id)], dtype=int)

    rng = np.random.default_rng() if int(cfg.random_seed) < 0 else np.random.default_rng(int(cfg.random_seed))
    min_need = int(cfg.min_points_per_period)

    pre_early_color = "#7B2CBF"
    pre_late_color = "#C77DFF"
    post_early_color = "#1B9E4B"
    post_late_color = "#A8E6A3"

    summary_rows: List[Dict[str, Any]] = []

    for lab in unique_labels:
        lab = int(lab)

        # All points for this cluster (used to fit ONE shared UMAP)
        idx_cluster_all = np.where((clusters == lab) & base_mask)[0]
        if idx_cluster_all.size == 0:
            continue

        X_cluster = X[idx_cluster_all]
        reducer = _make_umap_reducer(umap, cfg)
        xy_cluster = reducer.fit_transform(X_cluster)

        # Map original point index -> cluster-local UMAP row
        pos_map = {int(idx): i for i, idx in enumerate(idx_cluster_all)}

        # Raw group indices
        idx_pre_early_all = np.where(pre_early_mask & (clusters == lab))[0]
        idx_pre_late_all = np.where(pre_late_mask & (clusters == lab))[0]
        idx_post_early_all = np.where(post_early_mask & (clusters == lab))[0]
        idx_post_late_all = np.where(post_late_mask & (clusters == lab))[0]
        idx_pre_all = np.where(pre_mask & (clusters == lab))[0]
        idx_post_all = np.where(post_mask & (clusters == lab))[0]

        n_pre_early_raw = int(idx_pre_early_all.size)
        n_pre_late_raw = int(idx_pre_late_all.size)
        n_post_early_raw = int(idx_post_early_all.size)
        n_post_late_raw = int(idx_post_late_all.size)
        n_pre_total_raw = int(idx_pre_all.size)
        n_post_total_raw = int(idx_post_all.size)

        pre_present_any = n_pre_total_raw > 0
        post_present_any = n_post_total_raw > 0
        cluster_change = _cluster_change_label(pre_present_any, post_present_any)

        # Raw latent-space metrics for pre vs post
        centroid_shift_raw = float("nan")
        pre_rms_radius_raw = float("nan")
        post_rms_radius_raw = float("nan")
        post_over_pre_rms_radius = float("nan")
        pre_trace_cov_raw = float("nan")
        post_trace_cov_raw = float("nan")
        post_over_pre_trace_cov = float("nan")

        if pre_present_any:
            mu_pre_raw, pre_rms_radius_raw, pre_trace_cov_raw = _centroid_and_spread_metrics(X[idx_pre_all])
        else:
            mu_pre_raw = np.array([], dtype=float)

        if post_present_any:
            mu_post_raw, post_rms_radius_raw, post_trace_cov_raw = _centroid_and_spread_metrics(X[idx_post_all])
        else:
            mu_post_raw = np.array([], dtype=float)

        if pre_present_any and post_present_any and mu_pre_raw.size > 0 and mu_post_raw.size > 0:
            centroid_shift_raw = float(np.linalg.norm(mu_post_raw - mu_pre_raw))
            post_over_pre_rms_radius = _safe_ratio(post_rms_radius_raw, pre_rms_radius_raw)
            post_over_pre_trace_cov = _safe_ratio(post_trace_cov_raw, pre_trace_cov_raw)

        # PRE early/late BC after UMAP fit
        pre_bc_status = "ok"
        bc_pre = float("nan")
        idx_pre_early_plot = np.array([], dtype=int)
        idx_pre_late_plot = np.array([], dtype=int)
        n_pre_used = 0
        n_pre_plot = 0

        if n_pre_early_raw < min_need or n_pre_late_raw < min_need:
            pre_bc_status = "skipped_small"
        else:
            idx_pre_early_bal, idx_pre_late_bal, n_pre_used, _, _ = _balanced_subsample_pair(
                idx_pre_early_all, idx_pre_late_all, rng=rng, cap=cfg.max_points_per_period
            )
            if n_pre_used < min_need:
                pre_bc_status = "skipped_after_balance"
            else:
                idx_pre_early_plot = idx_pre_early_bal
                idx_pre_late_plot = idx_pre_late_bal
                n_pre_plot = int(n_pre_used)
                if cfg.max_points_per_period_for_plot is not None:
                    idx_pre_early_plot, idx_pre_late_plot, n_pre_plot, _, _ = _balanced_subsample_pair(
                        idx_pre_early_bal, idx_pre_late_bal, rng=rng, cap=cfg.max_points_per_period_for_plot
                    )

                xy_pre_early = xy_cluster[[pos_map[int(i)] for i in idx_pre_early_plot]]
                xy_pre_late = xy_cluster[[pos_map[int(i)] for i in idx_pre_late_plot]]
                bc_pre = bhattacharyya_coefficient_2d_hist(
                    xy_pre_early, xy_pre_late, bins=int(cfg.bc_bins)
                )

        # POST early/late BC after UMAP fit
        post_bc_status = "ok"
        bc_post = float("nan")
        idx_post_early_plot = np.array([], dtype=int)
        idx_post_late_plot = np.array([], dtype=int)
        n_post_used = 0
        n_post_plot = 0

        if n_post_early_raw < min_need or n_post_late_raw < min_need:
            post_bc_status = "skipped_small"
        else:
            idx_post_early_bal, idx_post_late_bal, n_post_used, _, _ = _balanced_subsample_pair(
                idx_post_early_all, idx_post_late_all, rng=rng, cap=cfg.max_points_per_period
            )
            if n_post_used < min_need:
                post_bc_status = "skipped_after_balance"
            else:
                idx_post_early_plot = idx_post_early_bal
                idx_post_late_plot = idx_post_late_bal
                n_post_plot = int(n_post_used)
                if cfg.max_points_per_period_for_plot is not None:
                    idx_post_early_plot, idx_post_late_plot, n_post_plot, _, _ = _balanced_subsample_pair(
                        idx_post_early_bal, idx_post_late_bal, rng=rng, cap=cfg.max_points_per_period_for_plot
                    )

                xy_post_early = xy_cluster[[pos_map[int(i)] for i in idx_post_early_plot]]
                xy_post_late = xy_cluster[[pos_map[int(i)] for i in idx_post_late_plot]]
                bc_post = bhattacharyya_coefficient_2d_hist(
                    xy_post_early, xy_post_late, bins=int(cfg.bc_bins)
                )

        # Direct balanced pre vs post BC after UMAP fit
        pre_post_bc_status = "ok"
        bc_pre_vs_post = float("nan")
        n_pre_post_used = 0

        if n_pre_total_raw < min_need or n_post_total_raw < min_need:
            pre_post_bc_status = "skipped_small"
        else:
            idx_pre_bal, idx_post_bal, n_pre_post_used, _, _ = _balanced_subsample_pair(
                idx_pre_all, idx_post_all, rng=rng, cap=cfg.max_points_per_period
            )
            if n_pre_post_used < min_need:
                pre_post_bc_status = "skipped_after_balance"
            else:
                xy_pre_bal = xy_cluster[[pos_map[int(i)] for i in idx_pre_bal]]
                xy_post_bal = xy_cluster[[pos_map[int(i)] for i in idx_post_bal]]
                bc_pre_vs_post = bhattacharyya_coefficient_2d_hist(
                    xy_pre_bal, xy_post_bal, bins=int(cfg.bc_bins)
                )

        # Make figure if either row is usable
        if pre_bc_status == "ok" or post_bc_status == "ok":
            fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), sharex=True, sharey=True)

            # PRE row
            if pre_bc_status == "ok":
                xy_pre_early = xy_cluster[[pos_map[int(i)] for i in idx_pre_early_plot]]
                xy_pre_late = xy_cluster[[pos_map[int(i)] for i in idx_pre_late_plot]]
                xy_pre_plot = np.vstack([xy_pre_early, xy_pre_late])
                is_pre_late_plot = np.array(
                    [False] * len(xy_pre_early) + [True] * len(xy_pre_late),
                    dtype=bool,
                )

                _plot_row(
                    axes[0],
                    xy=xy_pre_plot,
                    is_late=is_pre_late_plot,
                    period_name="Pre-lesion",
                    early_range=_format_date_range(_dates_for_indices(idx_pre_early_all, file_indices, file_dt_map)),
                    late_range=_format_date_range(_dates_for_indices(idx_pre_late_all, file_indices, file_dt_map)),
                    n_early=int(idx_pre_early_plot.size),
                    n_late=int(idx_pre_late_plot.size),
                    bc=float(bc_pre),
                    early_color=pre_early_color,
                    late_color=pre_late_color,
                )
            else:
                _mark_row_skipped(axes[0], f"Pre-lesion {pre_bc_status}", ylabel="UMAP Dimension 2")

            # POST row
            if post_bc_status == "ok":
                xy_post_early = xy_cluster[[pos_map[int(i)] for i in idx_post_early_plot]]
                xy_post_late = xy_cluster[[pos_map[int(i)] for i in idx_post_late_plot]]
                xy_post_plot = np.vstack([xy_post_early, xy_post_late])
                is_post_late_plot = np.array(
                    [False] * len(xy_post_early) + [True] * len(xy_post_late),
                    dtype=bool,
                )

                _plot_row(
                    axes[1],
                    xy=xy_post_plot,
                    is_late=is_post_late_plot,
                    period_name="Post-lesion",
                    early_range=_format_date_range(_dates_for_indices(idx_post_early_all, file_indices, file_dt_map)),
                    late_range=_format_date_range(_dates_for_indices(idx_post_late_all, file_indices, file_dt_map)),
                    n_early=int(idx_post_early_plot.size),
                    n_late=int(idx_post_late_plot.size),
                    bc=float(bc_post),
                    early_color=post_early_color,
                    late_color=post_late_color,
                )
            else:
                _mark_row_skipped(axes[1], f"Post-lesion {post_bc_status}", ylabel="UMAP Dimension 2")

            fig.suptitle(
                f"{animal_id} cluster {lab} early/late UMAPs (Version A: fit on all cluster points)\n"
                f"change={cluster_change} | pre BC={bc_pre if np.isfinite(bc_pre) else np.nan:.3f} | "
                f"post BC={bc_post if np.isfinite(bc_post) else np.nan:.3f} | "
                f"pre/post BC={bc_pre_vs_post if np.isfinite(bc_pre_vs_post) else np.nan:.3f}\n"
                f"centroid shift(raw)={centroid_shift_raw if np.isfinite(centroid_shift_raw) else np.nan:.3g} | "
                f"post/pre rms radius={post_over_pre_rms_radius if np.isfinite(post_over_pre_rms_radius) else np.nan:.3g} | "
                f"post/pre trace(cov)={post_over_pre_trace_cov if np.isfinite(post_over_pre_trace_cov) else np.nan:.3g} | "
                f"treatment={treatment_dt.date().isoformat()}",
                y=1.03,
            )

            fig.tight_layout()
            out_png = clusters_dir / f"{prefix}_cluster{lab}_pre_post_early_late_umap.png"
            fig.savefig(out_png, dpi=int(cfg.dpi), bbox_inches="tight")
            plt.close(fig)
        else:
            out_png = Path("")

        summary_rows.append({
            "cluster": lab,
            "fit_mode": "versionA_fit_all_cluster_points_then_balance_subsets",

            "cluster_change": cluster_change,
            "pre_present_any": bool(pre_present_any),
            "post_present_any": bool(post_present_any),

            "n_pre_total_raw": n_pre_total_raw,
            "n_post_total_raw": n_post_total_raw,
            "n_pre_early_raw": n_pre_early_raw,
            "n_pre_late_raw": n_pre_late_raw,
            "n_post_early_raw": n_post_early_raw,
            "n_post_late_raw": n_post_late_raw,

            "pre_bc_status": pre_bc_status,
            "post_bc_status": post_bc_status,
            "pre_post_bc_status": pre_post_bc_status,

            "n_pre_used": int(n_pre_used),
            "n_post_used": int(n_post_used),
            "n_pre_post_used": int(n_pre_post_used),
            "n_pre_plot_per_period": int(n_pre_plot),
            "n_post_plot_per_period": int(n_post_plot),

            "bc_pre_early_vs_late": float(bc_pre),
            "bc_post_early_vs_late": float(bc_post),
            "bc_pre_vs_post": float(bc_pre_vs_post),

            "centroid_shift_raw": float(centroid_shift_raw),
            "pre_rms_radius_raw": float(pre_rms_radius_raw),
            "post_rms_radius_raw": float(post_rms_radius_raw),
            "post_over_pre_rms_radius": float(post_over_pre_rms_radius),
            "pre_trace_cov_raw": float(pre_trace_cov_raw),
            "post_trace_cov_raw": float(post_trace_cov_raw),
            "post_over_pre_trace_cov": float(post_over_pre_trace_cov),

            "out_png": str(out_png),
        })

        print(
            f"[cluster {lab}] change={cluster_change} | "
            f"n_pre_total={n_pre_total_raw} n_post_total={n_post_total_raw} | "
            f"pre_BC={bc_pre if np.isfinite(bc_pre) else np.nan:.3f} "
            f"post_BC={bc_post if np.isfinite(bc_post) else np.nan:.3f} "
            f"pre_post_BC={bc_pre_vs_post if np.isfinite(bc_pre_vs_post) else np.nan:.3f} | "
            f"centroid_shift_raw={centroid_shift_raw if np.isfinite(centroid_shift_raw) else np.nan:.3g} | "
            f"post/pre_rms={post_over_pre_rms_radius if np.isfinite(post_over_pre_rms_radius) else np.nan:.3g}"
        )

    summary_csv = out_dir / f"{prefix}_cluster_variance_bc_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")
    return summary_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_pre_post_early_late_cluster_variance_bc.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", default=None, type=str)

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--cluster-id", default=None, type=int, help="If set, run only this cluster")
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)
    p.add_argument("--plot-max-points-per-period", default=None, type=int)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"])
    p.add_argument("--seed", default=0, type=int, help="Use -1 for no seed / faster parallel UMAP")

    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--out-prefix", default=None, type=str)
    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=100, type=int)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = UMAPEarlyLateConfig(
        npz_path=Path(args.npz_path),
        metadata_xlsx=Path(args.metadata_xlsx),

        array_key=args.array_key,
        cluster_key=args.cluster_key,
        file_key=args.file_key,

        include_noise=bool(args.include_noise),
        only_cluster_id=(int(args.cluster_id) if args.cluster_id is not None else None),
        vocalization_only=not bool(args.no_vocalization_only),

        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=(int(args.max_points_per_period) if args.max_points_per_period is not None else None),
        max_points_per_period_for_plot=(
            int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None
        ),

        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        early_late_split_method=str(args.early_late_split_method),
        random_seed=int(args.seed),

        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),

        out_dir=(Path(args.out_dir) if args.out_dir else None),
        out_prefix=args.out_prefix,
        dpi=int(args.dpi),
        bc_bins=int(args.bc_bins),
    )

    run_one_bird(cfg)


if __name__ == "__main__":
    main()