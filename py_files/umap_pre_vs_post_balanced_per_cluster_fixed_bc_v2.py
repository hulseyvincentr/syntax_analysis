#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_pre_vs_post_balanced_per_cluster.py

For ONE bird NPZ:
  - Load high-dim features (default: predictions) + HDBSCAN clusters + file_indices + file_map
  - Split points into pre vs post relative to treatment date from metadata XLSX
  - For each cluster (or one selected cluster):
      * take equal sample sizes in pre and post (balanced without replacement)
      * fit one UMAP on the combined (pre+post) balanced set
      * plot:
          - BEFORE (pre)
          - AFTER (post)
          - OVERLAP (pre vs post)

Outputs:
  out_dir/
    clusters/
      <prefix>_cluster<id>_before_after_overlap.png
    <prefix>_umap_pre_post_cluster_summary.csv
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


# -----------------------------
# Optional dependency: umap-learn
# -----------------------------
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


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class UMAPPrePostConfig:
    npz_path: Path
    metadata_xlsx: Path

    # metadata sheets/cols
    metadata_sheet_treatment: str = "metadata"
    treatment_date_col: str = "Treatment date"
    animal_id_col_candidates: Tuple[str, ...] = ("Animal ID", "animal_id", "animal id", "Bird", "bird id")

    # NPZ keys
    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"
    file_map_key: str = "file_map"
    vocalization_key: str = "vocalization"
    vocalization_only: bool = True

    include_noise: bool = False
    only_cluster_id: Optional[int] = None  # if set, run only this cluster

    # Sampling requirements (used for UMAP fitting)
    min_points_per_period: int = 200   # require at least this many pre AND post in the cluster before plotting
    max_points_per_period: Optional[int] = None  # cap for speed/memory; balanced size <= this

    # Plotting downsample (independent of fitting). None = plot all balanced points.
    max_points_per_period_for_plot: Optional[int] = None

    exclude_treatment_day_from_post: bool = False

    random_seed: int = 0

    # UMAP params
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"

    # Output
    out_dir: Optional[Path] = None
    out_prefix: Optional[str] = None
    dpi: int = 200

    # Bhattacharyya coefficient histogram bins (in UMAP space)
    bc_bins: int = 100


# -----------------------------
# Helpers
# -----------------------------
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
    """Parse a datetime from a recording filename referenced in file_map."""
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

    mapping: Dict[int, Any]
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
    n_target = min(n_a_raw, n_b_raw)  # always balance to the smaller period
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
    dmin = min(dts).date().isoformat()
    dmax = max(dts).date().isoformat()
    return f"{dmin} to {dmax}"


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
    eps: float = 1e-12,
) -> float:
    """Bhattacharyya coefficient BC in [0,1] using a 2D histogram estimate.

    BC = sum_{i} sqrt(p_i * q_i) where p and q are normalized histograms.
    1 means identical distributions (under this discretization); 0 means no overlap.
    """
    if xy_a.size == 0 or xy_b.size == 0:
        return float("nan")

    xy_all = np.vstack([xy_a, xy_b])
    xmin, ymin = np.min(xy_all, axis=0)
    xmax, ymax = np.max(xy_all, axis=0)

    dx = (xmax - xmin) * float(padding_frac) + 1e-9
    dy = (ymax - ymin) * float(padding_frac) + 1e-9
    x_range = (xmin - dx, xmax + dx)
    y_range = (ymin - dy, ymax + dy)

    Ha, _, _ = np.histogram2d(xy_a[:, 0], xy_a[:, 1], bins=bins, range=[x_range, y_range])
    Hb, _, _ = np.histogram2d(xy_b[:, 0], xy_b[:, 1], bins=bins, range=[x_range, y_range])

    Pa = Ha.astype(float)
    Pb = Hb.astype(float)

    sa = Pa.sum()
    sb = Pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")

    Pa = Pa / sa
    Pb = Pb / sb

    bc = float(np.sum(np.sqrt((Pa + eps) * (Pb + eps))))
    # cap to [0,1] for numerical drift
    if bc < 0:
        bc = 0.0
    if bc > 1:
        bc = 1.0
    return bc


# -----------------------------
# Plotting: BEFORE / AFTER / OVERLAP
# -----------------------------
def _plot_before_after_overlap(
    *,
    xy: np.ndarray,
    is_post: np.ndarray,
    animal_id: str,
    cluster_id: int,
    treatment_date: _dt.datetime,
    before_range: str,
    after_range: str,
    n_before: int,
    n_after: int,
    bc: float,
    out_png: Path,
    dpi: int,
) -> None:
    # Colors to mimic your example
    before_color = "#3b1aa8"  # deep purple
    after_color = "#0a7a2a"   # green
    overlap_before_color = "#ff00ff"  # magenta
    overlap_after_color = "#00ff00"   # bright green

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)

    fig.suptitle(f"{animal_id} cluster {cluster_id} (treatment {treatment_date.date().isoformat()})", y=1.02)

    # BEFORE
    ax = axes[0]
    ax.set_title(f"Before: {before_range}\nN={n_before}")
    pre_xy = xy[~is_post]
    ax.scatter(pre_xy[:, 0], pre_xy[:, 1], s=8, alpha=0.45, c=before_color, edgecolors="none")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(False)

    # AFTER
    ax = axes[1]
    ax.set_title(f"After: {after_range}\nN={n_after}")
    post_xy = xy[is_post]
    ax.scatter(post_xy[:, 0], post_xy[:, 1], s=8, alpha=0.45, c=after_color, edgecolors="none")
    ax.set_xlabel("UMAP Dimension 1")
    ax.grid(False)

    # OVERLAP (black background)
    ax = axes[2]
    ax.set_title(f"Overlap\nBC={bc:.3f}")
    ax.set_facecolor("black")
    # make tick labels visible on black bg
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("UMAP Dimension 1")

    ax.scatter(pre_xy[:, 0], pre_xy[:, 1], s=8, alpha=0.45, c=overlap_before_color, edgecolors="none")
    ax.scatter(post_xy[:, 0], post_xy[:, 1], s=8, alpha=0.45, c=overlap_after_color, edgecolors="none")
    ax.grid(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main per-bird runner
# -----------------------------
def run_one_bird(cfg: UMAPPrePostConfig) -> Path:
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

    N = int(X.shape[0])

    base_mask = np.ones(N, dtype=bool)
    if cfg.vocalization_only and cfg.vocalization_key in data:
        voc = np.asarray(data[cfg.vocalization_key])
        try:
            base_mask &= (voc.astype(int) == 1)
        except Exception:
            pass

    year_default = int(treatment_dt.year)
    file_dt_map = _build_file_datetime_map(file_map_obj, year_default=year_default)
    pre_mask, post_mask = _split_pre_post_masks(
        file_indices=file_indices,
        file_dt_map=file_dt_map,
        treatment_date=treatment_dt,
        exclude_treatment_day_from_post=cfg.exclude_treatment_day_from_post,
    )
    pre_mask &= base_mask
    post_mask &= base_mask

    out_dir = cfg.out_dir if cfg.out_dir is not None else (npz_path.parent / f"umap_pre_post_{animal_id}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.out_prefix if cfg.out_prefix is not None else f"{animal_id}_{cfg.array_key}"
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = np.unique(clusters)
    if not cfg.include_noise:
        unique_labels = unique_labels[unique_labels != -1]

    if cfg.only_cluster_id is not None:
        if int(cfg.only_cluster_id) not in set(map(int, unique_labels)):
            raise ValueError(
                f"--cluster-id {cfg.only_cluster_id} not found in labels (after include_noise={cfg.include_noise})."
            )
        unique_labels = np.array([int(cfg.only_cluster_id)], dtype=int)

    rng = (np.random.default_rng() if int(cfg.random_seed) < 0 else np.random.default_rng(int(cfg.random_seed)))
    min_need = int(cfg.min_points_per_period)

    summary_rows: List[Dict[str, Any]] = []

    for lab in unique_labels:
        lab = int(lab)

        idx_pre_all = np.where(pre_mask & (clusters == lab))[0]
        idx_post_all = np.where(post_mask & (clusters == lab))[0]

        n_pre_raw = int(idx_pre_all.size)
        n_post_raw = int(idx_post_all.size)

        if n_pre_raw < min_need or n_post_raw < min_need:
            summary_rows.append({
                "cluster": lab,
                "status": "skipped_small",
                "n_pre_raw": n_pre_raw,
                "n_post_raw": n_post_raw,
                "n_used": 0,
                "out_png": "",
            })
            continue

        # Balanced subsampling for fitting
        idx_pre_fit, idx_post_fit, n_target, _, _ = _balanced_subsample_pair(
            idx_pre_all, idx_post_all, rng=rng, cap=cfg.max_points_per_period
        )

        if n_target < min_need:
            summary_rows.append({
                "cluster": lab,
                "status": "skipped_after_balance",
                "n_pre_raw": n_pre_raw,
                "n_post_raw": n_post_raw,
                "n_used": n_target,
                "out_png": "",
            })
            continue

        # Optional additional downsample for plotting (still balanced)
        idx_pre_plot = idx_pre_fit
        idx_post_plot = idx_post_fit
        if cfg.max_points_per_period_for_plot is not None:
            idx_pre_plot, idx_post_plot, n_plot, _, _ = _balanced_subsample_pair(
                idx_pre_fit, idx_post_fit, rng=rng, cap=cfg.max_points_per_period_for_plot
            )
        else:
            n_plot = int(n_target)

        # Build combined matrix for UMAP fitting (use FIT indices, not plot)
        X_pre_fit = X[idx_pre_fit]
        X_post_fit = X[idx_post_fit]
        X_pair_fit = np.vstack([X_pre_fit, X_post_fit])

        is_post_fit = np.zeros(X_pair_fit.shape[0], dtype=bool)
        is_post_fit[X_pre_fit.shape[0]:] = True

        reducer = umap.UMAP(
            n_neighbors=int(cfg.n_neighbors),
            min_dist=float(cfg.min_dist),
            metric=str(cfg.metric),
            random_state=(None if int(cfg.random_seed) < 0 else int(cfg.random_seed)),
        )
        xy_fit = reducer.fit_transform(X_pair_fit)

        # Quantify overlap between pre/post in the fitted UMAP space
        bc = bhattacharyya_coefficient_2d_hist(
            xy_fit[~is_post_fit],
            xy_fit[is_post_fit],
            bins=getattr(cfg, 'bc_bins', 100),
        )


        # Now pull the subset of points we want to PLOT from the fitted embedding
        # Map original fit indices -> positions in X_pair_fit
        # first block: idx_pre_fit, second block: idx_post_fit
        pre_pos = {int(idx): i for i, idx in enumerate(idx_pre_fit)}
        post_pos = {int(idx): (i + idx_pre_fit.size) for i, idx in enumerate(idx_post_fit)}

        plot_positions: List[int] = []
        plot_is_post: List[bool] = []

        for idx in idx_pre_plot:
            plot_positions.append(pre_pos[int(idx)])
            plot_is_post.append(False)
        for idx in idx_post_plot:
            plot_positions.append(post_pos[int(idx)])
            plot_is_post.append(True)

        plot_positions_arr = np.asarray(plot_positions, dtype=int)
        plot_is_post_arr = np.asarray(plot_is_post, dtype=bool)
        xy_plot = xy_fit[plot_positions_arr]

        # Date ranges (use ALL points in cluster, not just sampled)
        before_range = _format_date_range(_dates_for_indices(idx_pre_all, file_indices, file_dt_map))
        after_range = _format_date_range(_dates_for_indices(idx_post_all, file_indices, file_dt_map))

        out_png = clusters_dir / f"{prefix}_cluster{lab}_before_after_overlap.png"
        _plot_before_after_overlap(
            xy=xy_plot,
            is_post=plot_is_post_arr,
            animal_id=animal_id,
            cluster_id=lab,
            treatment_date=treatment_dt,
            before_range=before_range,
            after_range=after_range,
            n_before=int(idx_pre_plot.size),
            n_after=int(idx_post_plot.size),
            out_png=out_png,
            dpi=int(cfg.dpi),
            bc=float(bc),
        )

        summary_rows.append({
            "cluster": lab,
            "status": "ok",
            "n_pre_raw": n_pre_raw,
            "n_post_raw": n_post_raw,
            "n_used": int(n_target),
            "n_plot_per_period": int(n_plot),
            "bhattacharyya_bc": float(bc),
            "out_png": str(out_png),
        })

        print(f"[UMAP] cluster {lab}: pre_raw={n_pre_raw} post_raw={n_post_raw} used={n_target} plot_per_period={n_plot} -> {out_png.name}")

    summary_csv = out_dir / f"{prefix}_umap_pre_post_cluster_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")
    return summary_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_pre_vs_post_balanced_per_cluster.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True, type=str, help="Path to ONE bird .npz (e.g., .../USA5288.npz).")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Excel metadata with treatment dates.")
    p.add_argument("--out-dir", default=None, type=str, help="Output directory (default: alongside NPZ).")

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--cluster-id", default=None, type=int, help="If set, run ONLY this cluster (one syllable).")
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)

    p.add_argument("--plot-max-points-per-period", default=None, type=int,
                   help="Optional: downsample plotted points per period (pre/post) after fitting.")

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")

    p.add_argument("--seed", default=0, type=int, help="Random seed (use -1 for no seed / faster parallel UMAP)")

    # UMAP params
    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--out-prefix", default=None, type=str)
    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=100, type=int, help="Bins for 2D histogram Bhattacharyya overlap in UMAP space")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = UMAPPrePostConfig(
        npz_path=Path(args.npz_path),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=(Path(args.out_dir) if args.out_dir else None),

        array_key=args.array_key,
        cluster_key=args.cluster_key,
        file_key=args.file_key,

        include_noise=bool(args.include_noise),
        only_cluster_id=(int(args.cluster_id) if args.cluster_id is not None else None),
        vocalization_only=not bool(args.no_vocalization_only),

        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=(int(args.max_points_per_period) if args.max_points_per_period is not None else None),

        max_points_per_period_for_plot=(int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None),

        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),

        random_seed=int(args.seed),

        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),

        out_prefix=args.out_prefix,
        dpi=int(args.dpi),
        bc_bins=int(args.bc_bins),
    )

    run_one_bird(cfg)


if __name__ == "__main__":
    main()
