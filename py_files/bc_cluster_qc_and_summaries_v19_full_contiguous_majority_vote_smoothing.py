#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_cluster_qc_and_summaries_v9.py

Wrapper around export_equal_umap_cluster_spectrograms_v8_run_balanced_phrase_normalized.py
that adds:

1) per-cluster Bhattacharyya-coefficient summaries for one bird,
2) bird-level boxplots across all qualifying clusters,
3) bird-level boxplots across high-variance phrase-duration clusters only,
4) fixed-timescale representative spectrogram QC panels (e.g. all rows 0->2 s).

Typical usage:
  python bc_cluster_qc_and_summaries_v9.py \
    --npz-path /Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443.npz \
    --metadata-excel-path /Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx \
    --spectrogram-script /Users/.../pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v7.py \
    --phrase-csv /Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv \
    --out-dir /Volumes/my_own_SSD/updated_AreaX_outputs/bc_cluster_qc_outputs \
    --animal-id USA5443 \
    --period-mode early_late_pre_post \
    --bc-analysis-mode run_balanced_phrase_normalized \
    --min-balanced-duration-s 2.0 \
    --phase-bins-per-run 25 \
    --min-runs-per-group 20 \
    --max-runs-per-group 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import argparse
import importlib.util
import math
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SCRIPT_VERSION = "bc_cluster_qc_and_summaries_v12_full_run_bc_variants"
DEFAULT_V8_SCRIPT = "export_equal_umap_cluster_spectrograms_v17_full_contiguous_main_decoder_noise_fill.py"


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return float("nan"), "n.s."
    try:
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception:
        p = float("nan")
    return p, _stars_from_p(p)


def _compute_weighted_hist_density(
    xy: np.ndarray,
    idx: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != idx.shape[0]:
            raise ValueError(f"weights length {w.shape[0]} does not match idx length {idx.shape[0]}")
        keep = np.isfinite(w) & (w >= 0)
        idx = idx[keep]
        w = w[keep]
        if idx.size == 0:
            return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    H, _, _ = np.histogram2d(xy[idx, 0], xy[idx, 1], bins=[x_edges, y_edges], weights=w)
    H = H.T.astype(float)
    total = H.sum()
    if total > 0:
        H /= total
    return H


def _hist_bc_local(pa: np.ndarray, pb: np.ndarray) -> float:
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    sa = pa.sum()
    sb = pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa = pa / sa
    pb = pb / sb
    return float(np.sum(np.sqrt(pa * pb)))



def _make_umap_density_edges_local(
    xy: np.ndarray,
    idx_arrays: Sequence[np.ndarray],
    *,
    density_bins: int,
    pad_fraction: float = 0.04,
    point_coverage: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build shared 2D histogram edges for BC from selected cluster points.

    If point_coverage is None or >=1, the grid spans all selected points. If it
    is between 0 and 1, the grid is built from the central fraction of selected
    points closest to the robust median center. Points outside the grid are
    dropped by np.histogram2d and each histogram is renormalized.
    """
    xy = np.asarray(xy, dtype=float)
    arrays = [np.asarray(idx, dtype=int).ravel() for idx in idx_arrays if np.asarray(idx).size > 0]
    if not arrays:
        raise ValueError("No selected indices available for BC grid.")
    selected_all = np.concatenate(arrays).astype(int)
    selected_all = selected_all[(selected_all >= 0) & (selected_all < xy.shape[0])]
    if selected_all.size == 0:
        raise ValueError("No valid selected indices available for BC grid.")

    pts = xy[selected_all]
    pts = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])]
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
            fallback = np.nanstd(pts, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
            scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
            dist = np.sqrt(np.sum(((pts - center) / scale) ** 2, axis=1))
            cutoff = float(np.nanquantile(dist, cov))
            keep = np.isfinite(dist) & (dist <= cutoff)
            if int(np.sum(keep)) >= 4:
                core = pts[keep]
                used_crop = True

    x_min = float(np.nanmin(core[:, 0]))
    x_max = float(np.nanmax(core[:, 0]))
    y_min = float(np.nanmin(core[:, 1]))
    y_max = float(np.nanmax(core[:, 1]))
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
    return x_edges, y_edges, info

def _compute_cluster_bcs(
    *,
    embedding_xy: np.ndarray,
    selected_by_group: Mapping[str, np.ndarray],
    density_bins: int,
    weights_by_group: Optional[Mapping[str, np.ndarray]] = None,
    bc_grid_point_coverage: Optional[float] = None,
) -> Dict[str, float]:
    """Compute BC values for early/late pre, early/late post, and pre/post.

    weights_by_group is optional. If supplied, each group must have one weight per
    time-bin index. This is used for run-weighted full-contiguous-run BC.
    """
    xy = np.asarray(embedding_xy, dtype=float)

    out = {
        "bc_pre": np.nan,
        "bc_post": np.nan,
        "bc_prepost": np.nan,
        "bc_grid_requested_point_coverage": np.nan if bc_grid_point_coverage is None else float(bc_grid_point_coverage),
        "bc_grid_actual_point_coverage": np.nan,
        "bc_grid_n_points_for_grid": np.nan,
        "bc_grid_n_points_inside_grid": np.nan,
    }
    keys = set(selected_by_group.keys())
    if not {"early_pre", "late_pre", "early_post", "late_post"}.issubset(keys):
        return out

    x_edges, y_edges, grid_info = _make_umap_density_edges_local(
        xy,
        [selected_by_group[k] for k in ["early_pre", "late_pre", "early_post", "late_post"]],
        density_bins=int(density_bins),
        pad_fraction=0.04,
        point_coverage=bc_grid_point_coverage,
    )
    out["bc_grid_actual_point_coverage"] = grid_info.get("actual_point_coverage", np.nan)
    out["bc_grid_n_points_for_grid"] = grid_info.get("n_points_for_grid", np.nan)
    out["bc_grid_n_points_inside_grid"] = grid_info.get("n_points_inside_grid", np.nan)

    weight_by = weights_by_group or {}

    def _idx(key: str) -> np.ndarray:
        return np.asarray(selected_by_group[key], dtype=int)

    def _weights(key: str) -> Optional[np.ndarray]:
        if key not in weight_by:
            return None
        return np.asarray(weight_by[key], dtype=float)

    def _hist(key: str) -> np.ndarray:
        return _compute_weighted_hist_density(
            xy,
            _idx(key),
            x_edges=x_edges,
            y_edges=y_edges,
            weights=_weights(key),
        )

    def _concat_weights(keys: Sequence[str]) -> Optional[np.ndarray]:
        if not all(k in weight_by for k in keys):
            return None
        return np.concatenate([np.asarray(weight_by[k], dtype=float) for k in keys])

    ep = _idx("early_pre")
    lp = _idx("late_pre")
    eo = _idx("early_post")
    lo = _idx("late_post")

    hist_ep = _hist("early_pre")
    hist_lp = _hist("late_pre")
    hist_eo = _hist("early_post")
    hist_lo = _hist("late_post")
    out["bc_pre"] = _hist_bc_local(hist_ep, hist_lp)
    out["bc_post"] = _hist_bc_local(hist_eo, hist_lo)

    hist_pre_all = _compute_weighted_hist_density(
        xy,
        np.concatenate([ep, lp]),
        x_edges=x_edges,
        y_edges=y_edges,
        weights=_concat_weights(["early_pre", "late_pre"]),
    )
    hist_post_all = _compute_weighted_hist_density(
        xy,
        np.concatenate([eo, lo]),
        x_edges=x_edges,
        y_edges=y_edges,
        weights=_concat_weights(["early_post", "late_post"]),
    )
    out["bc_prepost"] = _hist_bc_local(hist_pre_all, hist_post_all)
    return out


def _load_selected_bins(selected_csv: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(selected_csv)
    if "group" not in df.columns or "timebin_index" not in df.columns:
        raise KeyError(f"Expected columns 'group' and 'timebin_index' in {selected_csv}")
    out: Dict[str, np.ndarray] = {}
    for group_name, sub in df.groupby("group", sort=False):
        out[str(group_name)] = sub["timebin_index"].astype(int).to_numpy()
    return out


def _load_full_contiguous_selected_runs_from_selected_csv(
    selected_csv: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Load Analysis 3/4 data from a run-balanced selected-bin CSV.

    Returns:
      full_idx_by_group: group -> all contiguous bins from each selected run.
      run_weights_by_group: group -> per-bin weights, 1/run_length for each bin.
      n_runs_by_group: group -> number of selected full runs.

    This requires the phrase-normalized selected CSV, which contains group,
    run_id, run_start_bin, and run_end_bin_exclusive. Frame-weighted selected
    CSVs do not have enough run-boundary information for these analyses.
    """
    df = pd.read_csv(selected_csv)
    required = {"group", "run_id", "run_start_bin", "run_end_bin_exclusive"}
    if not required.issubset(df.columns):
        return {}, {}, {}

    run_df = (
        df[["group", "run_id", "run_start_bin", "run_end_bin_exclusive"]]
        .drop_duplicates()
        .sort_values(["group", "run_start_bin", "run_end_bin_exclusive", "run_id"])
        .copy()
    )

    full_idx_by_group: Dict[str, List[np.ndarray]] = {}
    run_weights_by_group: Dict[str, List[np.ndarray]] = {}
    n_runs_by_group: Dict[str, int] = {}

    for group_name, sub in run_df.groupby("group", sort=False):
        g = str(group_name)
        full_idx_by_group[g] = []
        run_weights_by_group[g] = []
        for _, r in sub.iterrows():
            s = int(r["run_start_bin"])
            e = int(r["run_end_bin_exclusive"])
            if e <= s:
                continue
            bins = np.arange(s, e, dtype=int)
            full_idx_by_group[g].append(bins)
            run_weights_by_group[g].append(np.full(len(bins), 1.0 / float(len(bins)), dtype=float))
        n_runs_by_group[g] = int(len(full_idx_by_group[g]))

    full_idx_out = {g: np.concatenate(chunks).astype(int) for g, chunks in full_idx_by_group.items() if chunks}
    weights_out = {g: np.concatenate(chunks).astype(float) for g, chunks in run_weights_by_group.items() if chunks}
    return full_idx_out, weights_out, n_runs_by_group


def _add_prefixed_bcs(row: Dict[str, Any], prefix: str, bcs: Mapping[str, float]) -> None:
    row[f"bc_pre_{prefix}"] = float(bcs.get("bc_pre", np.nan))
    row[f"bc_post_{prefix}"] = float(bcs.get("bc_post", np.nan))
    row[f"bc_prepost_{prefix}"] = float(bcs.get("bc_prepost", np.nan))


def _method_subset_for_boxplot(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pre_col = f"bc_pre_{prefix}"
    post_col = f"bc_post_{prefix}"
    prepost_col = f"bc_prepost_{prefix}"
    if pre_col not in df.columns or post_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["bc_pre"] = pd.to_numeric(out[pre_col], errors="coerce")
    out["bc_post"] = pd.to_numeric(out[post_col], errors="coerce")
    out["bc_prepost"] = pd.to_numeric(out[prepost_col], errors="coerce") if prepost_col in out.columns else np.nan
    out = out[np.isfinite(out["bc_pre"]) & np.isfinite(out["bc_post"])].copy()
    return out


def plot_bc_boxplot_for_cluster_set(
    *,
    bc_df: pd.DataFrame,
    title: str,
    out_png: Path,
    subtitle: str = "",
    dpi: int = 200,
) -> None:
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    if bc_df.empty:
        raise ValueError("No rows available for boxplot.")

    pre = pd.to_numeric(bc_df["bc_pre"], errors="coerce").to_numpy(dtype=float)
    post = pd.to_numeric(bc_df["bc_post"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) == 0:
        raise ValueError("No finite pre/post BC values available for boxplot.")

    p, stars = _paired_wilcoxon(pre, post)

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    bp = ax.boxplot(
        [pre, post],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=0),
        boxprops=dict(facecolor="#f2f2f2", edgecolor="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.6),
        capprops=dict(color="black", linewidth=1.6),
    )
    # Colored median lines to mimic prior style.
    med_pre = float(np.nanmedian(pre))
    med_post = float(np.nanmedian(post))
    ax.hlines(med_pre, 1 - 0.275, 1 + 0.275, colors="red", linewidth=2.0, zorder=4)
    ax.hlines(med_post, 2 - 0.275, 2 + 0.275, colors="#5e2ca5", linewidth=2.0, zorder=4)

    # light paired lines
    for a, b in zip(pre, post):
        ax.plot([1, 2], [a, b], color="0.78", linewidth=0.9, zorder=1)
    ax.scatter(np.repeat(1.0, len(pre)), pre, s=18, color="red", alpha=0.55, zorder=3)
    ax.scatter(np.repeat(2.0, len(post)), post, s=18, color="#5e2ca5", alpha=0.55, zorder=3)

    y_top = max(np.nanmax(pre), np.nanmax(post))
    y_bottom = min(np.nanmin(pre), np.nanmin(post))
    yr = max(0.02, y_top - y_bottom)
    yb = y_top + 0.08 * yr
    ax.plot([1, 1, 2, 2], [yb - 0.02*yr, yb, yb, yb - 0.02*yr], color="black", linewidth=1.6)
    ax.text(1.5, yb + 0.015*yr, stars, ha="center", va="bottom", fontsize=18)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient\n(early vs. late, equal selected time bins)", fontsize=15)
    main = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(main, fontsize=18, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(0.35, 2.65)
    ax.set_ylim(max(0.0, y_bottom - 0.08*yr), yb + 0.12*yr)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)



def _fixed_duration_panel(
    *,
    S_FxT: np.ndarray,
    context_start: int,
    panel_n_bins: int,
) -> np.ndarray:
    panel = np.full((S_FxT.shape[0], int(panel_n_bins)), np.nan, dtype=float)
    context_start = int(context_start)
    end = min(S_FxT.shape[1], context_start + int(panel_n_bins))
    n = max(0, end - context_start)
    if n > 0:
        panel[:, :n] = S_FxT[:, context_start:end].astype(float)
    return panel


def plot_fixed_timescale_full_run_contexts(
    *,
    S_FxT: np.ndarray,
    run_df: pd.DataFrame,
    animal_id: str,
    cluster_id: int,
    group_name: str,
    out_png: Path,
    seconds_per_bin: float,
    fixed_panel_duration_s: float,
    pre_context_bins: int,
    max_examples: int,
    example_mode: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    rng: np.random.Generator,
) -> None:
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    chosen = v8._choose_run_rows_for_context_examples(
        run_df,
        max_examples=int(max_examples),
        mode=str(example_mode),
        rng=rng,
    )
    if chosen.empty:
        raise ValueError(f"No rows available for fixed-timescale context plot: {group_name}")

    panel_n_bins = max(1, int(round(float(fixed_panel_duration_s) / float(seconds_per_bin))))
    n_rows = len(chosen)
    fig_h = max(2.0, 1.28 * n_rows + 0.9)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        run_start = int(row["run_start_bin"])
        run_end = int(row["run_end_bin_exclusive"])
        context_start = max(0, run_start - int(pre_context_bins))
        panel = _fixed_duration_panel(
            S_FxT=S_FxT,
            context_start=context_start,
            panel_n_bins=panel_n_bins,
        )
        ax.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, float(fixed_panel_duration_s), 0, panel.shape[0]],
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_xlim(0, float(fixed_panel_duration_s))

        # Blue = full run, red = selected equal-UMAP bins, positioned relative to context_start.
        #
        # Earlier versions drew blue first as a transparent fill and then red on top.
        # When selected-bin coverage is high, the red fill hides the blue fill. Here
        # the full run is shown redundantly as:
        #   1) a faint blue span,
        #   2) a dark-blue outline rectangle, and
        #   3) a thick blue bar at the bottom of the panel.
        # The selected bins are drawn as red vertical marks, so they do not obscure
        # the full-run extent.
        blue_start = max(run_start, context_start)
        blue_end = min(run_end, context_start + panel_n_bins)
        blue_x0 = (blue_start - context_start) * float(seconds_per_bin)
        blue_x1 = (blue_end - context_start) * float(seconds_per_bin)
        if blue_end > blue_start:
            ax.axvspan(
                blue_x0,
                blue_x1,
                color="tab:blue",
                alpha=0.08,
                zorder=2,
            )
            y0, y1 = ax.get_ylim()
            ax.add_patch(
                plt.Rectangle(
                    (blue_x0, y0),
                    blue_x1 - blue_x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="tab:blue",
                    linewidth=1.4,
                    zorder=6,
                )
            )
            ax.hlines(
                y0 + 0.04 * (y1 - y0),
                blue_x0,
                blue_x1,
                colors="tab:blue",
                linewidth=3.0,
                zorder=7,
            )

        seg_starts = v8._parse_semicolon_ints(row.get("selected_segment_starts", ""))
        seg_ends = v8._parse_semicolon_ints(row.get("selected_segment_ends_exclusive", ""))
        for s, e in zip(seg_starts, seg_ends):
            s2 = max(int(s), context_start)
            e2 = min(int(e), context_start + panel_n_bins)
            if e2 > s2:
                red_x0 = (s2 - context_start) * float(seconds_per_bin)
                red_x1 = (e2 - context_start) * float(seconds_per_bin)
                # If a segment covers many consecutive bins, use a very light fill
                # plus boundary lines; if it is only one/few bins, the boundary lines
                # still make it visible without hiding the blue run.
                ax.axvspan(
                    red_x0,
                    red_x1,
                    color="tab:red",
                    alpha=0.10,
                    zorder=4,
                )
                ax.vlines(
                    [red_x0, red_x1],
                    ymin=ax.get_ylim()[0],
                    ymax=ax.get_ylim()[1],
                    colors="tab:red",
                    linewidth=0.45,
                    alpha=0.75,
                    zorder=8,
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
    fig.suptitle(
        f"{animal_id} — label {int(cluster_id):02d} — {group_name}\n"
        f"Representative full HDBSCAN phrase-label runs; blue outline/bar = full run, red marks = equal UMAP-selected bins; fixed x-axis 0–{float(fixed_panel_duration_s):.1f} s",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.12)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def run_pipeline(
    *,
    npz_path: Path,
    metadata_excel_path: Path,
    spectrogram_script: Path,
    out_dir: Path,
    animal_id: Optional[str],
    phrase_csv: Optional[Path],
    top_fraction: float,
    post_group_name: str,
    top_min_n_phrases: int,
    include_noise: bool,
    fill_noise_labels_from_nearest_nonnoise: bool,
    noise_label: int,
    smooth_short_label_interruptions: bool,
    max_label_interruption_ms: float,
    smoothing_max_iterations: int,
    apply_majority_vote_label_smoothing: bool,
    majority_vote_window_bins: int,
    majority_vote_by_file: bool,
    only_cluster_ids: Optional[Sequence[int]],
    vocalization_only: bool,
    period_mode: str,
    treatment_day_assignment: str,
    early_late_split_method: str,
    bc_analysis_mode: str,
    min_points_per_group: int,
    max_points_per_group: Optional[int],
    sample_mode: str,
    min_runs_per_group: int,
    max_runs_per_group: Optional[int],
    phase_bins_per_run: int,
    min_full_run_duration_ms: float,
    min_run_group_fraction: float,
    run_sample_mode: str,
    allow_repeat_phase_bins: bool,
    save_phase_normalized_bc: bool,
    save_full_run_bc_variants: bool,
    seed: int,
    bins_per_spectrogram_row: int,
    max_rows_per_spectrogram: Optional[int],
    spectrogram_source_mode: str,
    save_representative_full_run_contexts: bool,
    max_expanded_full_run_bins: Optional[int],
    full_run_context_bins: int,
    full_run_max_context_bins: int,
    full_run_max_examples: int,
    full_run_example_mode: str,
    full_run_fixed_duration_s: float,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    subplot_hspace: float,
    cmap: str,
    contrast_percentiles: Optional[Tuple[float, float]],
    dpi: int,
    umap_density_bins: int,
    bc_grid_point_coverage: Optional[float],
    dry_run: bool,
    min_balanced_duration_s: float,
    fixed_panel_duration_s: float,
    fixed_panel_pre_context_bins: int,
    fixed_panel_max_examples: int,
) -> Path:
    summary_csv = v8.export_equal_umap_cluster_spectrograms_for_one_bird(
        npz_path=npz_path,
        metadata_excel_path=metadata_excel_path,
        spectrogram_script=spectrogram_script,
        out_dir=out_dir,
        animal_id=animal_id,
        include_noise=include_noise,
        fill_noise_labels_from_nearest_nonnoise=fill_noise_labels_from_nearest_nonnoise,
        noise_label=noise_label,
        smooth_short_label_interruptions=smooth_short_label_interruptions,
        max_label_interruption_ms=max_label_interruption_ms,
        smoothing_max_iterations=smoothing_max_iterations,
        apply_majority_vote_label_smoothing=apply_majority_vote_label_smoothing,
        majority_vote_window_bins=majority_vote_window_bins,
        majority_vote_by_file=majority_vote_by_file,
        only_cluster_ids=only_cluster_ids,
        vocalization_only=vocalization_only,
        period_mode=period_mode,
        treatment_day_assignment=treatment_day_assignment,
        early_late_split_method=early_late_split_method,
        bc_analysis_mode=bc_analysis_mode,
        min_points_per_group=min_points_per_group,
        max_points_per_group=max_points_per_group,
        sample_mode=sample_mode,
        min_runs_per_group=min_runs_per_group,
        max_runs_per_group=max_runs_per_group,
        phase_bins_per_run=phase_bins_per_run,
        min_full_run_duration_ms=min_full_run_duration_ms,
        min_run_group_fraction=min_run_group_fraction,
        run_sample_mode=run_sample_mode,
        allow_repeat_phase_bins=allow_repeat_phase_bins,
        save_phase_normalized_bc=save_phase_normalized_bc,
        save_full_run_bc_variants=save_full_run_bc_variants,
        seed=seed,
        bins_per_spectrogram_row=bins_per_spectrogram_row,
        max_rows_per_spectrogram=max_rows_per_spectrogram,
        spectrogram_source_mode=spectrogram_source_mode,
        save_representative_full_run_contexts=save_representative_full_run_contexts,
        max_expanded_full_run_bins=max_expanded_full_run_bins,
        full_run_context_bins=full_run_context_bins,
        full_run_max_context_bins=full_run_max_context_bins,
        full_run_max_examples=full_run_max_examples,
        full_run_example_mode=full_run_example_mode,
        full_run_fixed_duration_s=full_run_fixed_duration_s,
        seconds_per_bin=seconds_per_bin,
        x_tick_step_s=x_tick_step_s,
        figure_width=figure_width,
        row_height=row_height,
        subplot_hspace=subplot_hspace,
        cmap=cmap,
        contrast_percentiles=contrast_percentiles,
        dpi=dpi,
        umap_density_bins=umap_density_bins,
        bc_grid_point_coverage=bc_grid_point_coverage,
        dry_run=dry_run,
        top_variance_csv=None,  # evaluate all clusters first
        top_fraction=top_fraction,
        post_group_name=post_group_name,
        top_min_n_phrases=top_min_n_phrases,
    )
    if dry_run:
        return Path(summary_csv)

    summary_df = pd.read_csv(summary_csv)
    animal_id_final = str(summary_df["animal_id"].dropna().astype(str).iloc[0]) if not summary_df.empty else (animal_id or npz_path.stem.split("_")[0])
    animal_out_dir = Path(out_dir).expanduser().resolve() / animal_id_final
    _safe_mkdir(animal_out_dir)

    arr = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(arr["hdbscan_labels"]).astype(int)
    embedding_xy = v8._get_embedding_xy(arr, "embedding_outputs")
    helper = _load_module_from_path(spectrogram_script, "spec_helper_v9")
    S_FxT = helper._orient_spectrogram_to_FxT(np.asarray(arr["s"]), labels)

    # Determine high-variance subset if available.
    high_variance_ids: Optional[set[int]] = None
    if phrase_csv is not None and Path(phrase_csv).exists():
        try:
            hv = v8.read_top_variance_cluster_ids(
                Path(phrase_csv),
                animal_id=animal_id_final,
                top_fraction=float(top_fraction),
                post_group_name=str(post_group_name),
                min_n_phrases=int(top_min_n_phrases),
            )
            high_variance_ids = set(int(x) for x in hv)
        except Exception as e:
            print(f"[WARN] Could not determine high-variance clusters from {phrase_csv}: {e}")
            traceback.print_exc()
            high_variance_ids = None

    cluster_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    done_df = summary_df[summary_df.get("status", pd.Series(dtype=str)).astype(str) == "done"].copy()
    for _, row in done_df.iterrows():
        try:
            cluster_id = int(row["cluster_id"])
            selected_csv = Path(str(row["selected_timebins_csv"]))
            if not selected_csv.exists():
                print(f"[WARN] Missing selected CSV for cluster {cluster_id}: {selected_csv}")
                continue
            selected_by_group = _load_selected_bins(selected_csv)

            # Existing/default calculation: BC using the exact selected bins.
            bc_vals = _compute_cluster_bcs(
                embedding_xy=embedding_xy,
                selected_by_group=selected_by_group,
                density_bins=int(umap_density_bins),
                bc_grid_point_coverage=bc_grid_point_coverage,
            )

            # Analysis 3: full-contiguous-run BC, selected runs only.
            # Analysis 4: run-weighted full-contiguous-run BC, selected runs only.
            full_idx_by_group, run_weights_by_group, n_full_runs_by_group = _load_full_contiguous_selected_runs_from_selected_csv(selected_csv)
            if full_idx_by_group:
                bc_full_contiguous = _compute_cluster_bcs(
                    embedding_xy=embedding_xy,
                    selected_by_group=full_idx_by_group,
                    density_bins=int(umap_density_bins),
                    bc_grid_point_coverage=bc_grid_point_coverage,
                )
                bc_run_weighted = _compute_cluster_bcs(
                    embedding_xy=embedding_xy,
                    selected_by_group=full_idx_by_group,
                    weights_by_group=run_weights_by_group,
                    density_bins=int(umap_density_bins),
                    bc_grid_point_coverage=bc_grid_point_coverage,
                )
            else:
                bc_full_contiguous = {"bc_pre": np.nan, "bc_post": np.nan, "bc_prepost": np.nan}
                bc_run_weighted = {"bc_pre": np.nan, "bc_post": np.nan, "bc_prepost": np.nan}

            n_equal = int(pd.to_numeric(row.get("n_equal_selected_per_group", np.nan), errors="coerce"))
            balanced_duration_s = float(n_equal) * float(seconds_per_bin)
            pass_min_duration = bool(balanced_duration_s >= float(min_balanced_duration_s))
            is_high_variance = bool(high_variance_ids is not None and int(cluster_id) in high_variance_ids)
            cluster_row = {
                "animal_id": animal_id_final,
                "cluster_id": int(cluster_id),
                "cluster_token": row.get("cluster_token", f"label{int(cluster_id):02d}"),
                "status": row.get("status", "done"),
                "n_equal_selected_per_group": n_equal,
                "n_equal_runs_per_group": pd.to_numeric(row.get("n_equal_runs_per_group", np.nan), errors="coerce"),
                "phase_bins_per_run": pd.to_numeric(row.get("phase_bins_per_run", np.nan), errors="coerce"),
                "balanced_duration_s_per_group": balanced_duration_s,
                "passes_min_balanced_duration": pass_min_duration,
                "is_high_variance_cluster": is_high_variance,

                # Backward-compatible columns: exact selected-bin BC.
                "bc_pre": bc_vals["bc_pre"],
                "bc_post": bc_vals["bc_post"],
                "bc_prepost": bc_vals["bc_prepost"],

                "has_full_contiguous_run_bc_variants": bool(full_idx_by_group),
                "n_full_contiguous_runs_by_group": ";".join(f"{k}:{v}" for k, v in sorted(n_full_runs_by_group.items())),
                "n_full_contiguous_bins_by_group": ";".join(f"{k}:{len(v)}" for k, v in sorted(full_idx_by_group.items())),
                "umap_png": row.get("umap_png", ""),
                "full_contiguous_selected_runs_umap_png": row.get("full_contiguous_selected_runs_umap_png", ""),
                "run_weighted_full_contiguous_selected_runs_umap_png": row.get("run_weighted_full_contiguous_selected_runs_umap_png", ""),
                "selected_timebins_csv": str(selected_csv),
                "full_run_mapping_csvs": row.get("full_run_mapping_csvs", ""),
                "representative_full_run_context_pngs": row.get("representative_full_run_context_pngs", ""),
            }
            _add_prefixed_bcs(cluster_row, "selected_bins", bc_vals)
            _add_prefixed_bcs(cluster_row, "full_contiguous_selected_runs", bc_full_contiguous)
            _add_prefixed_bcs(cluster_row, "run_weighted_full_contiguous", bc_run_weighted)
            cluster_rows.append(cluster_row)

            # Create fixed-timescale QC spectrograms for each group-specific run summary CSV.
            csvs_text = str(row.get("full_run_mapping_csvs", "")).strip()
            group_csvs = [Path(x) for x in csvs_text.split(";") if str(x).strip()]
            vmin, vmax = v8._compute_shared_intensity_limits(S_FxT, selected_by_group, contrast_percentiles)
            for run_csv in group_csvs:
                if not run_csv.exists():
                    continue
                run_df = pd.read_csv(run_csv)
                if run_df.empty:
                    continue
                group_name = str(run_df.get("group_name", pd.Series([run_csv.stem])).iloc[0]) if "group_name" in run_df.columns else run_csv.stem
                out_png = run_csv.with_name(run_csv.stem.replace("_full_runs_from_equal_timebins", f"_fixed_{fixed_panel_duration_s:.1f}s_full_run_contexts") + ".png")
                try:
                    plot_fixed_timescale_full_run_contexts(
                        S_FxT=S_FxT,
                        run_df=run_df,
                        animal_id=animal_id_final,
                        cluster_id=cluster_id,
                        group_name=group_name,
                        out_png=out_png,
                        seconds_per_bin=float(seconds_per_bin),
                        fixed_panel_duration_s=float(fixed_panel_duration_s),
                        pre_context_bins=int(fixed_panel_pre_context_bins),
                        max_examples=int(fixed_panel_max_examples),
                        example_mode=str(full_run_example_mode),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                    )
                    print(f"[SAVE] {out_png}")
                except Exception as e:
                    print(f"[WARN] Could not save fixed-timescale QC spectrogram for {run_csv}: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"[WARN] Could not process cluster summary row: {e}")
            traceback.print_exc()

    bc_df = pd.DataFrame(cluster_rows)
    bc_csv = animal_out_dir / f"{animal_id_final}_cluster_bc_summary.csv"
    bc_df.to_csv(bc_csv, index=False)
    print(f"[SAVE] {bc_csv}")

    # Aggregate summaries / boxplots.
    qual_df = bc_df[bc_df["passes_min_balanced_duration"] == True].copy()
    if not qual_df.empty:
        all_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_all_clusters_min{min_balanced_duration_s:.1f}s.png"
        subtitle = f"all qualifying clusters | minimum balanced duration per group = {float(min_balanced_duration_s):.1f} s | N clusters = {len(qual_df)}"
        plot_bc_boxplot_for_cluster_set(
            bc_df=qual_df,
            title=f"{animal_id_final}: BC across all clusters",
            subtitle=subtitle,
            out_png=all_boxplot,
            dpi=int(dpi),
        )
        print(f"[SAVE] {all_boxplot}")

        stats_rows = [{
            "set_name": "all_clusters",
            "bc_method": "selected_bins",
            "n_clusters": int(len(qual_df)),
            "mean_bc_pre": float(np.nanmean(qual_df["bc_pre"])),
            "mean_bc_post": float(np.nanmean(qual_df["bc_post"])),
            "mean_bc_prepost": float(np.nanmean(qual_df["bc_prepost"])),
            "median_bc_pre": float(np.nanmedian(qual_df["bc_pre"])),
            "median_bc_post": float(np.nanmedian(qual_df["bc_post"])),
            "median_bc_prepost": float(np.nanmedian(qual_df["bc_prepost"])),
        }]

        bc_method_defs = [
            ("full_contiguous_selected_runs", "full-contiguous selected runs"),
            ("run_weighted_full_contiguous", "run-weighted full-contiguous selected runs"),
        ]
        for prefix, label in bc_method_defs:
            method_df = _method_subset_for_boxplot(qual_df, prefix)
            if method_df.empty:
                continue
            method_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_all_clusters_{prefix}_min{min_balanced_duration_s:.1f}s.png"
            method_subtitle = (
                f"{label} | all qualifying clusters | minimum balanced duration per group = "
                f"{float(min_balanced_duration_s):.1f} s | N clusters = {len(method_df)}"
            )
            plot_bc_boxplot_for_cluster_set(
                bc_df=method_df,
                title=f"{animal_id_final}: BC across all clusters",
                subtitle=method_subtitle,
                out_png=method_boxplot,
                dpi=int(dpi),
            )
            print(f"[SAVE] {method_boxplot}")
            stats_rows.append({
                "set_name": "all_clusters",
                "bc_method": prefix,
                "n_clusters": int(len(method_df)),
                "mean_bc_pre": float(np.nanmean(method_df["bc_pre"])),
                "mean_bc_post": float(np.nanmean(method_df["bc_post"])),
                "mean_bc_prepost": float(np.nanmean(method_df["bc_prepost"])),
                "median_bc_pre": float(np.nanmedian(method_df["bc_pre"])),
                "median_bc_post": float(np.nanmedian(method_df["bc_post"])),
                "median_bc_prepost": float(np.nanmedian(method_df["bc_prepost"])),
            })

        if "is_high_variance_cluster" in qual_df.columns and qual_df["is_high_variance_cluster"].any():
            hv_df = qual_df[qual_df["is_high_variance_cluster"] == True].copy()
            if not hv_df.empty:
                hv_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_high_variance_clusters_min{min_balanced_duration_s:.1f}s.png"
                hv_subtitle = (
                    f"high-variance phrase-duration clusters only | minimum balanced duration per group = {float(min_balanced_duration_s):.1f} s | "
                    f"N clusters = {len(hv_df)}"
                )
                plot_bc_boxplot_for_cluster_set(
                    bc_df=hv_df,
                    title=f"{animal_id_final}: BC across high-variance clusters",
                    subtitle=hv_subtitle,
                    out_png=hv_boxplot,
                    dpi=int(dpi),
                )
                print(f"[SAVE] {hv_boxplot}")
                stats_rows.append({
                    "set_name": "high_variance_clusters",
                    "bc_method": "selected_bins",
                    "n_clusters": int(len(hv_df)),
                    "mean_bc_pre": float(np.nanmean(hv_df["bc_pre"])),
                    "mean_bc_post": float(np.nanmean(hv_df["bc_post"])),
                    "mean_bc_prepost": float(np.nanmean(hv_df["bc_prepost"])),
                    "median_bc_pre": float(np.nanmedian(hv_df["bc_pre"])),
                    "median_bc_post": float(np.nanmedian(hv_df["bc_post"])),
                    "median_bc_prepost": float(np.nanmedian(hv_df["bc_prepost"])),
                })

        # Remaining/non-top-variance clusters: all qualifying clusters that were NOT
        # in the top phrase-duration-variance set. This is the complement of
        # high_variance_clusters within qual_df.
        if "is_high_variance_cluster" in qual_df.columns:
            rem_df = qual_df[qual_df["is_high_variance_cluster"] == False].copy()
            if not rem_df.empty:
                rem_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_remaining_non_high_variance_clusters_min{min_balanced_duration_s:.1f}s.png"
                rem_subtitle = (
                    f"remaining non-top-variance clusters only | minimum balanced duration per group = {float(min_balanced_duration_s):.1f} s | "
                    f"N clusters = {len(rem_df)}"
                )
                plot_bc_boxplot_for_cluster_set(
                    bc_df=rem_df,
                    title=f"{animal_id_final}: BC across remaining non-top-variance clusters",
                    subtitle=rem_subtitle,
                    out_png=rem_boxplot,
                    dpi=int(dpi),
                )
                print(f"[SAVE] {rem_boxplot}")
                stats_rows.append({
                    "set_name": "remaining_non_high_variance_clusters",
                    "bc_method": "selected_bins",
                    "n_clusters": int(len(rem_df)),
                    "mean_bc_pre": float(np.nanmean(rem_df["bc_pre"])),
                    "mean_bc_post": float(np.nanmean(rem_df["bc_post"])),
                    "mean_bc_prepost": float(np.nanmean(rem_df["bc_prepost"])),
                    "median_bc_pre": float(np.nanmedian(rem_df["bc_pre"])),
                    "median_bc_post": float(np.nanmedian(rem_df["bc_post"])),
                    "median_bc_prepost": float(np.nanmedian(rem_df["bc_prepost"])),
                })

                for prefix, label in bc_method_defs:
                    rem_method_df = _method_subset_for_boxplot(rem_df, prefix)
                    if rem_method_df.empty:
                        continue
                    rem_method_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_remaining_non_high_variance_clusters_{prefix}_min{min_balanced_duration_s:.1f}s.png"
                    rem_method_subtitle = (
                        f"{label} | remaining non-top-variance clusters only | minimum balanced duration per group = "
                        f"{float(min_balanced_duration_s):.1f} s | N clusters = {len(rem_method_df)}"
                    )
                    plot_bc_boxplot_for_cluster_set(
                        bc_df=rem_method_df,
                        title=f"{animal_id_final}: BC across remaining non-top-variance clusters",
                        subtitle=rem_method_subtitle,
                        out_png=rem_method_boxplot,
                        dpi=int(dpi),
                    )
                    print(f"[SAVE] {rem_method_boxplot}")
                    stats_rows.append({
                        "set_name": "remaining_non_high_variance_clusters",
                        "bc_method": prefix,
                        "n_clusters": int(len(rem_method_df)),
                        "mean_bc_pre": float(np.nanmean(rem_method_df["bc_pre"])),
                        "mean_bc_post": float(np.nanmean(rem_method_df["bc_post"])),
                        "mean_bc_prepost": float(np.nanmean(rem_method_df["bc_prepost"])),
                        "median_bc_pre": float(np.nanmedian(rem_method_df["bc_pre"])),
                        "median_bc_post": float(np.nanmedian(rem_method_df["bc_post"])),
                        "median_bc_prepost": float(np.nanmedian(rem_method_df["bc_prepost"])),
                    })

        stats_csv = animal_out_dir / f"{animal_id_final}_BC_aggregate_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
        print(f"[SAVE] {stats_csv}")
    else:
        print("[WARN] No clusters passed the minimum balanced-duration threshold; no aggregate boxplots were created.")

    return bc_csv



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one-bird cluster BC/QC exports, then create bird-level BC summaries, "
            "boxplots, and fixed-timescale representative spectrogram QC panels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True)
    p.add_argument("--metadata-excel-path", required=True)
    p.add_argument("--spectrogram-script", required=True)
    p.add_argument("--v8-script", default=DEFAULT_V8_SCRIPT, help="Path to the v8 run-balanced phrase-normalized export script.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--animal-id", default=None)
    p.add_argument("--phrase-csv", default=None, help="Optional phrase-duration CSV used to define high-variance clusters.")
    p.add_argument("--top-fraction", type=float, default=0.30)
    p.add_argument("--post-group-name", default="Post")
    p.add_argument("--top-min-n-phrases", type=int, default=100)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument(
        "--fill-noise-labels-from-nearest-nonnoise",
        action="store_true",
        help=(
            "Optional decoder-style label cleanup before BC selection: replace -1/noise "
            "labels with nearest non-noise labels using the same in-place logic as decoder data prep."
        ),
    )
    p.add_argument("--noise-label", type=int, default=-1)
    p.add_argument(
        "--smooth-short-label-interruptions",
        action="store_true",
        help=(
            "Optional global label smoothing before BC/run selection: bridge short A-B-A "
            "interruptions by replacing the middle B run with A when it is shorter than "
            "--max-label-interruption-ms. Example: 17 17 12 17 -> 17 17 17 17."
        ),
    )
    p.add_argument("--max-label-interruption-ms", type=float, default=50.0)
    p.add_argument("--smoothing-max-iterations", type=int, default=3)
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
    p.add_argument("--majority-vote-across-file-boundaries", action="store_true", help="If set, apply majority voting across the whole concatenated label array instead of independently within file_indices segments.")
    p.add_argument("--only-cluster-ids", nargs="*", default=None)
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--period-mode", choices=["pre_post", "early_late_pre_post"], default="early_late_pre_post")
    p.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")
    p.add_argument("--early-late-split-method", choices=["file_median", "file_half", "timebin_half"], default="file_median")

    p.add_argument("--bc-analysis-mode", choices=["run_balanced_phrase_normalized", "run_balanced_full_contiguous", "frame_weighted"], default="run_balanced_phrase_normalized")
    p.add_argument("--min-runs-per-group", type=int, default=20)
    p.add_argument("--max-runs-per-group", type=int, default=200)
    p.add_argument("--phase-bins-per-run", type=int, default=25)
    p.add_argument("--min-full-run-duration-ms", type=float, default=100.0)
    p.add_argument("--min-run-group-fraction", type=float, default=0.80)
    p.add_argument("--run-sample-mode", choices=["random", "first", "longest"], default="random")
    p.add_argument("--allow-repeat-phase-bins", action="store_true")
    p.add_argument("--no-phase-normalized-bc", action="store_true")
    p.add_argument("--no-full-run-bc-variant-plots", action="store_true", help="Do not ask the export helper to save full-contiguous/run-weighted UMAP overlap plots. Summary CSV still computes these BC variants when run-boundary info is available.")

    p.add_argument("--min-points-per-group", type=int, default=6000)
    p.add_argument("--max-points-per-group", type=int, default=6000)
    p.add_argument("--sample-mode", choices=["random", "first", "time_balanced", "time_balanced_random_offset"], default="random", help="Frame-weighted mode only: random, first, or evenly spaced time-balanced sampling.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--bins-per-spectrogram-row", type=int, default=2000)
    p.add_argument("--max-rows-per-spectrogram", type=int, default=None)
    p.add_argument("--spectrogram-source-mode", choices=["selected_timebins", "expanded_full_runs", "both"], default="expanded_full_runs")
    p.add_argument("--max-expanded-full-run-bins", type=int, default=10000)
    p.add_argument("--no-representative-full-run-contexts", action="store_true")
    p.add_argument("--full-run-context-bins", type=int, default=80)
    p.add_argument("--full-run-max-context-bins", type=int, default=1500)
    p.add_argument("--full-run-max-examples", type=int, default=12)
    p.add_argument("--full-run-example-mode", choices=["mixed", "longest", "shortest", "random"], default="mixed")
    p.add_argument("--full-run-fixed-duration-s", type=float, default=5.4, help="Fixed x-axis duration in seconds for representative_full_run_contexts plots written by the export helper. Set to 5.4 to match the default 2000-bin expanded full-run spectrogram rows at 2.7 ms/bin.")
    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--x-tick-step-s", type=float, default=1.0)
    p.add_argument("--figure-width", type=float, default=24.0)
    p.add_argument("--row-height", type=float, default=2.2)
    p.add_argument("--subplot-hspace", type=float, default=0.005)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--umap-density-bins", type=int, default=20)
    p.add_argument("--bc-grid-point-coverage", type=float, default=None, help="Optional central fraction of selected UMAP points used to define the BC density grid, e.g. 0.99 to crop distant outliers. Omit or use 1.0 to include all selected points.")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--min-balanced-duration-s", type=float, default=2.0, help="Minimum balanced duration per group required for a cluster to contribute to aggregate BC summaries.")
    p.add_argument("--fixed-panel-duration-s", type=float, default=2.0, help="Fixed x-axis duration for representative QC spectrogram rows.")
    p.add_argument("--fixed-panel-pre-context-bins", type=int, default=40, help="How much pre-run context to show at the start of each fixed-duration representative QC row.")
    p.add_argument("--fixed-panel-max-examples", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global v8
    v8 = _load_module_from_path(Path(args.v8_script).expanduser().resolve(), "export_equal_umap_cluster_spectrograms_v8")

    only_cluster_ids = None
    if args.only_cluster_ids:
        only_cluster_ids = [int(x) for x in _parse_list_arg(args.only_cluster_ids)]

    max_points_per_group = None if int(args.max_points_per_group) <= 0 else int(args.max_points_per_group)
    max_runs_per_group = None if int(args.max_runs_per_group) <= 0 else int(args.max_runs_per_group)
    max_expanded_full_run_bins = None if int(args.max_expanded_full_run_bins) <= 0 else int(args.max_expanded_full_run_bins)
    contrast = None if bool(args.no_contrast_percentiles) else tuple(float(x) for x in args.contrast_percentiles)

    run_pipeline(
        npz_path=Path(args.npz_path).expanduser().resolve(),
        metadata_excel_path=Path(args.metadata_excel_path).expanduser().resolve(),
        spectrogram_script=Path(args.spectrogram_script).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        animal_id=args.animal_id,
        phrase_csv=(Path(args.phrase_csv).expanduser().resolve() if args.phrase_csv else None),
        top_fraction=float(args.top_fraction),
        post_group_name=str(args.post_group_name),
        top_min_n_phrases=int(args.top_min_n_phrases),
        include_noise=bool(args.include_noise),
        fill_noise_labels_from_nearest_nonnoise=bool(args.fill_noise_labels_from_nearest_nonnoise),
        noise_label=int(args.noise_label),
        smooth_short_label_interruptions=bool(args.smooth_short_label_interruptions),
        max_label_interruption_ms=float(args.max_label_interruption_ms),
        smoothing_max_iterations=int(args.smoothing_max_iterations),
        apply_majority_vote_label_smoothing=bool(args.apply_majority_vote_label_smoothing),
        majority_vote_window_bins=int(args.majority_vote_window_bins),
        majority_vote_by_file=not bool(args.majority_vote_across_file_boundaries),
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
        save_full_run_bc_variants=not bool(args.no_full_run_bc_variant_plots),
        seed=int(args.seed),
        bins_per_spectrogram_row=int(args.bins_per_spectrogram_row),
        max_rows_per_spectrogram=args.max_rows_per_spectrogram,
        spectrogram_source_mode=str(args.spectrogram_source_mode),
        save_representative_full_run_contexts=not bool(args.no_representative_full_run_contexts),
        max_expanded_full_run_bins=max_expanded_full_run_bins,
        full_run_context_bins=int(args.full_run_context_bins),
        full_run_max_context_bins=int(args.full_run_max_context_bins),
        full_run_max_examples=int(args.full_run_max_examples),
        full_run_example_mode=str(args.full_run_example_mode),
        full_run_fixed_duration_s=float(args.full_run_fixed_duration_s),
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
        dry_run=bool(args.dry_run),
        min_balanced_duration_s=float(args.min_balanced_duration_s),
        fixed_panel_duration_s=float(args.fixed_panel_duration_s),
        fixed_panel_pre_context_bins=int(args.fixed_panel_pre_context_bins),
        fixed_panel_max_examples=int(args.fixed_panel_max_examples),
    )


if __name__ == "__main__":
    main()
