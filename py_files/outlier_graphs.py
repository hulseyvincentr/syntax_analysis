# -*- coding: utf-8 -*-
"""
outlier_graphs.py

Pre vs Post variance scatterplots with outlier selection + Area X metadata coloring
(using organize_metadata_excel.build_areax_metadata + *_final_volumes.json attachments).

Key features
------------
1) Histology lesion-% sourcing uses:
       organize_metadata_excel.build_areax_metadata(..., volumes_dir=histology_volumes_dir)

2) Continuous/discrete coloring matches phrase_and_metadata_continuous_plotting.py logic:
   - If Area X visible AND lesion% finite -> continuous colormap + colorbar
   - Else -> discrete category colors (sham/red, large lesion not visible/black, miss+unknown/lightgray)

3) Identity-line agreement stats (y=x) computed separately for SHAM vs NMA:
   - Pearson r (on log10 variances if log_scale=True)
   - Lin's Concordance Correlation Coefficient (CCC) (agreement with y=x)
   - MAE and RMSE deviation from identity (log10(post/pre) if log_scale else post-pre)
   Printed to terminal and optionally displayed ON THE RIGHT, UNDER THE LEGEND.

4) Additional plots:
   - Ratio vs lesion volume: y = post_variance / pre_variance, x = lesion_volume
     SHAM points plotted ON TOP in red.

Notes
-----
- "Correlation to identity line" is best captured by Lin's CCC (agreement, not just correlation).
- This script tries to auto-detect lesion volume columns if you do not specify them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------------------------------------------------------
# Import your metadata builder
# -----------------------------------------------------------------------------
try:
    from organize_metadata_excel import build_areax_metadata
except Exception:
    build_areax_metadata = None


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _as_float(x: Any) -> float:
    """Best-effort float conversion (supports numeric strings, strips % and commas)."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return float("nan")
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return float("nan")
        s = s.replace("%", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return float("nan")
    return float("nan")


def _is_visible_from_meta(raw: Any) -> bool:
    """Match phrase_and_metadata_continuous_plotting.py visibility logic."""
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if s == "":
        return False
    if "not visible" in s:
        return False
    if s.startswith("n"):
        return False
    if s.startswith("y"):
        return True
    if s == "visible" or " visible" in s:
        return True
    return False


def _normalize_flag(val: Any) -> Optional[bool]:
    """Return True/False for Y/N-like values; None if unknown/blank."""
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s:
        return None
    if s in {"Y", "YES", "TRUE", "T", "1"}:
        return True
    if s in {"N", "NO", "FALSE", "F", "0"}:
        return False
    return None


def _canonicalize_special_category(s: Any) -> str:
    """
    Canonicalize special categories that can appear in metadata.
    Standard keys:
      - "sham saline injection"
      - "large lesion Area X not visible"
      - "miss"
      - "unknown"
    """
    if s is None:
        return "unknown"
    ss = str(s).strip()
    if ss == "":
        return "unknown"

    low = ss.lower()

    if ("saline" in low and "sham" in low) or ("sham" in low):
        return "sham saline injection"

    if ("large" in low) and ("lesion" in low) and ("not visible" in low):
        return "large lesion Area X not visible"

    if low.strip() == "miss" or "miss" in low:
        return "miss"

    if low.strip() == "unknown":
        return "unknown"

    return ss


def _hit_bool_from_hit_type(val: Any) -> Optional[bool]:
    """
    Interpret build_areax_metadata's per-region hit-type strings:
      - "bilateral" / "unilateral_L" / "unilateral_R" => True
      - "miss" => False
      - "unknown" / sham / large lesion not visible => None (not informative for region hit)
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    low = s.lower()

    # explicitly non-informative for this purpose
    if ("sham" in low) or ("saline" in low and "sham" in low):
        return None
    if ("large" in low) and ("lesion" in low) and ("not visible" in low):
        return None
    if low.strip() == "unknown":
        return None

    if "bilateral" in low:
        return True
    if "unilateral_l" in low or "unilateral_r" in low:
        return True
    if low.strip() == "miss":
        return False

    return None


def _infer_med_lat_hit_from_injections(entry: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Fallback: infer whether there is ANY Medial hit and ANY Lateral hit
    by scanning entry["injections"] for Y/N flags.

    Returns (medial_hit, lateral_hit) each in {True, False, None}.
    """
    injections = entry.get("injections", []) or []
    if not injections:
        return None, None

    med_any = False
    lat_any = False
    med_has_true = False
    med_has_false = False
    lat_has_true = False
    lat_has_false = False

    for inj in injections:
        m = _normalize_flag(inj.get("Medial Area X hit?", None))
        l = _normalize_flag(inj.get("Lateral Area X hit?", None))

        if m is not None:
            med_any = True
            med_has_true = med_has_true or bool(m)
            med_has_false = med_has_false or (not bool(m))

        if l is not None:
            lat_any = True
            lat_has_true = lat_has_true or bool(l)
            lat_has_false = lat_has_false or (not bool(l))

    def _collapse(any_flag: bool, has_true: bool, has_false: bool) -> Optional[bool]:
        if not any_flag:
            return None
        if has_true:
            return True
        if has_false:
            return False
        return None

    return _collapse(med_any, med_has_true, med_has_false), _collapse(lat_any, lat_has_true, lat_has_false)


def _get_lesion_pct_value(
    entry: Dict[str, Any],
    *,
    left_col: str,
    right_col: str,
    lesion_pct_mode: str = "avg",
) -> float:
    l = _as_float(entry.get(left_col, np.nan))
    r = _as_float(entry.get(right_col, np.nan))
    mode = (lesion_pct_mode or "avg").lower()

    if mode == "left":
        return l
    if mode == "right":
        return r

    vals = [v for v in (l, r) if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _pretty_axes(ax: plt.Axes) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def _apply_scales(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_scale: bool,
) -> Tuple[float, float]:
    """
    Set x/y scales and choose symmetric limits based on data.
    Returns (lo, hi) for y=x line / limits.
    """
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo = float(np.nanmin(np.concatenate([x, y]))) / 1.3
        hi = float(np.nanmax(np.concatenate([x, y]))) * 1.3
        if not np.isfinite(lo) or lo <= 0:
            lo = 1e-3
    else:
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo = lo - pad
        hi = hi + pad
        lo = max(0.0, lo)

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        lo, hi = (1e-3, 1.0) if log_scale else (0.0, 1.0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return lo, hi


def _build_subtitle_lines(
    *,
    pre_group: str,
    post_group: str,
    rank_on: str,
    min_n_phrases: int,
    log_scale: bool,
    include_visibility_note: bool = False,
) -> Tuple[str, str]:
    """Returns two subtitle lines to reduce whitespace / overly long titles."""
    line2 = f"pre={pre_group}, post={post_group}, rank_on={rank_on}, min_N_phrases={min_n_phrases}"
    line2 += ", log–log" if log_scale else ", linear"

    if include_visibility_note:
        line1 = "Area X visible: colored by % lesion (colorbar); not visible or missing %: discrete hit-type colors"
    else:
        line1 = line2
        line2 = ""

    return line1, line2


def _classify_point_category(
    entry: Dict[str, Any],
    *,
    area_visible_col: str,
    treatment_type_col: str = "Treatment type",
) -> Tuple[str, Optional[bool], Optional[bool], bool]:
    """
    Decide the DISCRETE category label for plotting,
    and return (medial_hit_bool, lateral_hit_bool, is_visible).

    Categories:
      - "sham saline injection"
      - "large lesion Area X not visible"
      - "Area X visible (Medial+Lateral hit)"
      - "Area X visible (Medial only)"
      - "Area X visible (Lateral only)"
      - "miss"
      - "unknown"
    """
    is_vis = _is_visible_from_meta(entry.get(area_visible_col, None))

    ttype = _canonicalize_special_category(entry.get(treatment_type_col, None))

    med_type = entry.get("Medial Area X hit type", None)
    lat_type = entry.get("Lateral Area X hit type", None)
    med_type_can = _canonicalize_special_category(med_type)
    lat_type_can = _canonicalize_special_category(lat_type)

    if (ttype == "sham saline injection") or (med_type_can == "sham saline injection") or (lat_type_can == "sham saline injection"):
        return "sham saline injection", None, None, is_vis

    if (med_type_can == "large lesion Area X not visible") or (lat_type_can == "large lesion Area X not visible"):
        return "large lesion Area X not visible", None, None, is_vis

    med_hit = _hit_bool_from_hit_type(med_type)
    lat_hit = _hit_bool_from_hit_type(lat_type)

    if (med_hit is None) and (lat_hit is None):
        med_hit, lat_hit = _infer_med_lat_hit_from_injections(entry)

    if is_vis:
        if (med_hit is False) and (lat_hit is False):
            return "miss", med_hit, lat_hit, is_vis
        if (med_hit is True) and (lat_hit is True):
            return "Area X visible (Medial+Lateral hit)", med_hit, lat_hit, is_vis
        if (med_hit is True) and (lat_hit is not True):
            return "Area X visible (Medial only)", med_hit, lat_hit, is_vis
        if (lat_hit is True) and (med_hit is not True):
            return "Area X visible (Lateral only)", med_hit, lat_hit, is_vis
        return "unknown", med_hit, lat_hit, is_vis

    # Not visible
    if (med_hit is False) and (lat_hit is False):
        return "miss", med_hit, lat_hit, is_vis
    return "unknown", med_hit, lat_hit, is_vis


# -----------------------------------------------------------------------------
# Identity-line agreement stats
# -----------------------------------------------------------------------------
def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC):
    measures agreement with identity line (slope 1, intercept 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")

    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x, ddof=1)) if x.size > 1 else float("nan")
    vy = float(np.var(y, ddof=1)) if y.size > 1 else float("nan")
    if not (np.isfinite(vx) and np.isfinite(vy)) or vx <= 0 or vy <= 0:
        return float("nan")

    r = _pearson_r(x, y)
    if not np.isfinite(r):
        return float("nan")

    ccc = (2.0 * r * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2)
    return float(ccc)


def _identity_agreement_stats(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    log_scale: bool,
) -> Dict[str, float]:
    """
    Stats describing agreement with y=x.
    If log_scale, we compute deltas in log10 space: delta = log10(post/pre).
    Otherwise, delta = post - pre.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post)
    pre = pre[m]
    post = post[m]

    out: Dict[str, float] = {"n": float(pre.size)}

    if pre.size < 2:
        out.update({"pearson_r": float("nan"), "ccc": float("nan"), "mae_delta": float("nan"), "rmse_delta": float("nan")})
        return out

    if log_scale:
        good = (pre > 0) & (post > 0)
        pre2 = pre[good]
        post2 = post[good]
        if pre2.size < 2:
            out.update({"pearson_r": float("nan"), "ccc": float("nan"), "mae_delta": float("nan"), "rmse_delta": float("nan")})
            return out
        x = np.log10(pre2)
        y = np.log10(post2)
        delta = np.log10(post2 / pre2)  # = y - x
    else:
        x = pre
        y = post
        delta = post - pre

    out["pearson_r"] = _pearson_r(x, y)
    out["ccc"] = _lin_ccc(x, y)
    out["mae_delta"] = float(np.mean(np.abs(delta))) if delta.size else float("nan")
    out["rmse_delta"] = float(np.sqrt(np.mean(delta ** 2))) if delta.size else float("nan")
    return out


def _format_identity_stats_block(
    label: str,
    stats: Dict[str, float],
    *,
    log_scale: bool,
) -> str:
    n = int(stats.get("n", 0))
    r = stats.get("pearson_r", float("nan"))
    ccc = stats.get("ccc", float("nan"))
    mae = stats.get("mae_delta", float("nan"))
    rmse = stats.get("rmse_delta", float("nan"))
    if log_scale:
        return (
            f"{label} (n={n})\n"
            f"Pearson r={r:.3f}   CCC={ccc:.3f}\n"
            f"MAE |log10(post/pre)|={mae:.3f}\n"
            f"RMSE log10(post/pre)={rmse:.3f}"
        )
    else:
        return (
            f"{label} (n={n})\n"
            f"Pearson r={r:.3f}   CCC={ccc:.3f}\n"
            f"MAE |post-pre|={mae:.3g}\n"
            f"RMSE post-pre={rmse:.3g}"
        )


def _place_stats_under_legend_axcoords(
    fig: plt.Figure,
    ax: plt.Axes,
    legend_obj: plt.Legend,
    stats_text: str,
    *,
    pad_axes_y: float = 0.02,
    fontsize: float = 10.0,
) -> None:
    """
    Put stats text UNDER a legend, using AXES coordinates.
    Works even when the legend is outside the axes (bbox_to_anchor with x>1),
    because we compute legend bbox after draw and place the text directly below.
    """
    if not stats_text or legend_obj is None:
        return

    # Need renderer to get legend's bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = legend_obj.get_window_extent(renderer=renderer)

    # Convert bbox bottom-left into axes coords
    bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
    x0 = float(bbox_axes.x0)
    y0 = float(bbox_axes.y0)

    ax.text(
        x0,
        y0 - pad_axes_y,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
        clip_on=False,
    )


def _place_stats_under_legend_figcoords(
    fig: plt.Figure,
    legend_obj: plt.Legend,
    stats_text: str,
    *,
    pad_fig_y: float = 0.01,
    fontsize: float = 10.0,
) -> None:
    """
    Put stats text UNDER a FIGURE legend (fig.legend), using figure coordinates.
    """
    if not stats_text or legend_obj is None:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = legend_obj.get_window_extent(renderer=renderer)
    bbox_fig = bbox_disp.transformed(fig.transFigure.inverted())
    x0 = float(bbox_fig.x0)
    y0 = float(bbox_fig.y0)

    fig.text(
        x0,
        y0 - pad_fig_y,
        stats_text,
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )


# -----------------------------------------------------------------------------
# Treatment-group + lesion volume helpers
# -----------------------------------------------------------------------------
def _infer_treatment_group(
    entry: Dict[str, Any],
    *,
    treatment_type_col: str = "Treatment type",
    nma_token: str = "nma",
) -> str:
    """Returns one of: {"sham", "nma", "other"}."""
    t_raw = entry.get(treatment_type_col, None)
    t_can = _canonicalize_special_category(t_raw)
    if t_can == "sham saline injection":
        return "sham"

    low = str(t_raw).strip().lower() if t_raw is not None else ""
    if nma_token and (nma_token.lower() in low):
        return "nma"

    # some pipelines store NMA in other fields; try a few common ones
    for k in ["Treatment", "Treatment Type", "Treatment_type", "Lesion type", "Lesion", "Drug"]:
        v = entry.get(k, None)
        if v is None:
            continue
        if nma_token.lower() in str(v).lower():
            return "nma"

    return "other"


def _auto_find_lesion_volume_cols(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Best-effort guess of lesion volume columns from a metadata entry dict.
    Returns (left_col, right_col, total_col).
    """
    keys = list(entry.keys())
    cand = []
    for k in keys:
        lk = str(k).lower()
        if ("volume" in lk) and ("lesion" in lk):
            cand.append(k)

    if not cand:
        return None, None, None

    def score(k: str) -> int:
        lk = k.lower()
        s = 0
        if "area x" in lk or "areax" in lk:
            s += 5
        if lk.startswith("l_") or "left" in lk:
            s += 2
        if lk.startswith("r_") or "right" in lk:
            s += 2
        if "mm3" in lk or "mm^3" in lk:
            s += 1
        return s

    cand_sorted = sorted(cand, key=lambda k: score(str(k)), reverse=True)

    left = None
    right = None
    total = None

    for k in cand_sorted:
        lk = str(k).lower()
        if left is None and (lk.startswith("l_") or "left" in lk):
            left = k
        elif right is None and (lk.startswith("r_") or "right" in lk):
            right = k

    if left is None and right is None:
        total = cand_sorted[0]
    return left, right, total


def _get_lesion_volume_value(
    entry: Dict[str, Any],
    *,
    left_col: Optional[str],
    right_col: Optional[str],
    total_col: Optional[str],
    mode: str = "sum",
) -> float:
    """
    Returns lesion volume as float.
    - If total_col exists and is numeric, use it (unless mode is left/right).
    - Otherwise use left/right columns and mode:
        mode="sum" (default): L + R (or whichever exists)
        mode="avg": mean of finite L/R
        mode="left"/"right": take that side
    """
    mode = (mode or "sum").lower()

    if total_col and (mode not in {"left", "right"}):
        vtot = _as_float(entry.get(total_col, np.nan))
        if np.isfinite(vtot):
            return float(vtot)

    l = _as_float(entry.get(left_col, np.nan)) if left_col else float("nan")
    r = _as_float(entry.get(right_col, np.nan)) if right_col else float("nan")

    if mode == "left":
        return float(l)
    if mode == "right":
        return float(r)
    if mode == "avg":
        vals = [v for v in (l, r) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    vals = [v for v in (l, r) if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.sum(vals))


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------
def plot_pre_post_variance_scatter(
    *,
    csv_path: Union[str, Path],
    out_dir: Union[str, Path],
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    variance_col: str = "Variance_ms2",
    group_col: str = "Group",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    nphrases_col: str = "N_phrases",
    min_n_phrases: int = 0,
    # outlier selection
    top_percentile: Optional[float] = 90,
    rank_on: str = "post",  # 'pre'|'post'|'max'
    # metadata coloring
    metadata_excel: Optional[Union[str, Path]] = None,
    meta_sheet_name: Union[int, str] = 0,
    area_visible_col: str = "Area X visible in histology?",
    treatment_type_col: str = "Treatment type",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    lesion_pct_mode: str = "avg",
    histology_volumes_dir: Optional[Union[str, Path]] = None,
    make_continuous_lesion_pct_plot: bool = True,
    # identity-line stats
    compute_identity_stats: bool = True,
    annotate_identity_stats: bool = True,
    nma_token: str = "nma",
    # ratio vs lesion volume
    make_ratio_vs_volume_plots: bool = True,
    left_lesion_vol_col: Optional[str] = None,
    right_lesion_vol_col: Optional[str] = None,
    total_lesion_vol_col: Optional[str] = None,
    lesion_vol_mode: str = "sum",   # "sum" | "avg" | "left" | "right"
    ratio_logy: bool = False,
    volume_logx: bool = False,
    # plot appearance
    cmap_name: str = "Purples",
    log_scale: bool = True,
    alpha: float = 0.85,
    marker_size: float = 30.0,
    discrete_color_map: Optional[Dict[str, Any]] = None,
    show: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Build pre vs post variance scatterplots across all birds.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {animal_col, group_col, syllable_col, variance_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    df[animal_col] = df[animal_col].astype(str)
    df[group_col] = df[group_col].astype(str)

    # Aggregate within (animal, syllable, group)
    agg_cols = [animal_col, syllable_col, group_col]
    if nphrases_col in df.columns:
        g = df.groupby(agg_cols, dropna=False, as_index=False).agg(
            **{
                variance_col: (variance_col, "mean"),
                nphrases_col: (nphrases_col, "mean"),
            }
        )
    else:
        g = df.groupby(agg_cols, dropna=False, as_index=False).agg(
            **{variance_col: (variance_col, "mean")}
        )

    pre = g[g[group_col] == pre_group][
        [animal_col, syllable_col, variance_col]
        + ([nphrases_col] if nphrases_col in g.columns else [])
    ].copy()
    post = g[g[group_col] == post_group][
        [animal_col, syllable_col, variance_col]
        + ([nphrases_col] if nphrases_col in g.columns else [])
    ].copy()

    if pre.empty or post.empty:
        raise ValueError(
            f"Need both pre_group={pre_group!r} and post_group={post_group!r} rows in the CSV."
        )

    merged = pd.merge(
        pre,
        post,
        on=[animal_col, syllable_col],
        how="inner",
        suffixes=("_pre", "_post"),
    )

    if merged.empty:
        raise ValueError("No overlapping (animal, syllable) pairs between pre and post groups.")

    merged.rename(
        columns={
            f"{variance_col}_pre": "pre_variance",
            f"{variance_col}_post": "post_variance",
        },
        inplace=True,
    )

    # N_phrases filter
    if min_n_phrases > 0 and nphrases_col in merged.columns:
        npre = merged.get(f"{nphrases_col}_pre", np.nan).astype(float)
        npost = merged.get(f"{nphrases_col}_post", np.nan).astype(float)
        keep_n = (npre >= min_n_phrases) & (npost >= min_n_phrases)
        merged = merged.loc[keep_n].copy()

    # finite / log constraints
    merged["pre_variance"] = merged["pre_variance"].astype(float)
    merged["post_variance"] = merged["post_variance"].astype(float)
    finite = np.isfinite(merged["pre_variance"]) & np.isfinite(merged["post_variance"])
    merged = merged.loc[finite].copy()

    if log_scale:
        merged = merged.loc[
            (merged["pre_variance"] > 0.0) & (merged["post_variance"] > 0.0)
        ].copy()

    if merged.empty:
        raise ValueError("No valid finite pre/post variance pairs after filtering.")

    # --- Top-percentile (within-animal) filter ---
    if top_percentile is not None:
        tp = float(top_percentile)
        if not (0.0 < tp < 100.0):
            raise ValueError("top_percentile must be between 0 and 100 (exclusive).")

        r = (rank_on or "post").lower()
        if r not in {"pre", "post", "max"}:
            raise ValueError("rank_on must be one of: 'pre', 'post', 'max'.")

        if r == "pre":
            merged["_score"] = merged["pre_variance"].astype(float)
        elif r == "post":
            merged["_score"] = merged["post_variance"].astype(float)
        else:
            merged["_score"] = np.maximum(
                merged["pre_variance"].astype(float),
                merged["post_variance"].astype(float),
            )

        keep_mask = pd.Series(False, index=merged.index)
        for _, sub in merged.groupby(animal_col, sort=False):
            vals = sub["_score"].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            cutoff = float(np.nanpercentile(vals, tp))
            keep_mask.loc[sub.index] = sub["_score"].astype(float) >= cutoff

        merged = merged.loc[keep_mask].copy()
        merged.drop(columns=["_score"], inplace=True, errors="ignore")

    if merged.empty:
        raise ValueError(
            "No data left after top-percentile filtering. "
            "Try lowering top_percentile or min_n_phrases."
        )

    # --- Load metadata using build_areax_metadata ---
    meta_dict: Dict[str, Dict[str, Any]] = {}
    if metadata_excel is not None:
        if build_areax_metadata is None:
            raise RuntimeError(
                "Could not import build_areax_metadata from organize_metadata_excel.py. "
                "Ensure organize_metadata_excel.py is on your PYTHONPATH."
            )

        meta_dict = build_areax_metadata(
            Path(metadata_excel),
            sheet_name=meta_sheet_name,
            volumes_dir=(Path(histology_volumes_dir) if histology_volumes_dir is not None else None),
        )

        if verbose:
            n_animals = len(meta_dict)
            n_with_L = sum(
                np.isfinite(_as_float(v.get(left_lesion_pct_col, np.nan)))
                for v in meta_dict.values()
            )
            n_with_R = sum(
                np.isfinite(_as_float(v.get(right_lesion_pct_col, np.nan)))
                for v in meta_dict.values()
            )
            print(f"[INFO] Loaded metadata for {n_animals} animals.")
            print(f"[INFO] Animals with finite {left_lesion_pct_col}: {n_with_L}")
            print(f"[INFO] Animals with finite {right_lesion_pct_col}: {n_with_R}")

    # Discrete colors
    cmap = plt.get_cmap(cmap_name)
    visible_both = cmap(0.80)
    visible_med_only = cmap(0.62)
    visible_lat_only = cmap(0.48)

    if discrete_color_map is None:
        discrete_color_map = {
            "Area X visible (Medial+Lateral hit)": visible_both,
            "Area X visible (Medial only)": visible_med_only,
            "Area X visible (Lateral only)": visible_lat_only,
            "sham saline injection": "red",
            "large lesion Area X not visible": "black",
            "miss": "lightgray",
            "unknown": "lightgray",
        }

    # Attach per-point metadata fields
    vis_list: List[bool] = []
    lesion_list: List[float] = []
    cat_list: List[str] = []
    med_hit_list: List[Optional[bool]] = []
    lat_hit_list: List[Optional[bool]] = []
    treat_group_list: List[str] = []
    lesion_vol_list: List[float] = []

    # Auto-detect lesion volume columns if none provided
    auto_left_vol = None
    auto_right_vol = None
    auto_total_vol = None
    if metadata_excel is not None and (left_lesion_vol_col is None and right_lesion_vol_col is None and total_lesion_vol_col is None):
        for _aid, entry in meta_dict.items():
            if not isinstance(entry, dict) or not entry:
                continue
            lcol, rcol, tcol = _auto_find_lesion_volume_cols(entry)
            if lcol or rcol or tcol:
                auto_left_vol, auto_right_vol, auto_total_vol = lcol, rcol, tcol
                break
        if verbose:
            print(f"[INFO] Auto-detected lesion volume cols: L={auto_left_vol}, R={auto_right_vol}, total={auto_total_vol}")

    for _, row in merged.iterrows():
        aid = str(row[animal_col])
        entry = meta_dict.get(aid, {})

        if entry:
            cat, med_hit, lat_hit, is_vis = _classify_point_category(
                entry,
                area_visible_col=area_visible_col,
                treatment_type_col=treatment_type_col,
            )
            pct = _get_lesion_pct_value(
                entry,
                left_col=left_lesion_pct_col,
                right_col=right_lesion_pct_col,
                lesion_pct_mode=lesion_pct_mode,
            )
            tg = _infer_treatment_group(entry, treatment_type_col=treatment_type_col, nma_token=nma_token)

            lvol_col = left_lesion_vol_col or auto_left_vol
            rvol_col = right_lesion_vol_col or auto_right_vol
            tvol_col = total_lesion_vol_col or auto_total_vol
            vol = _get_lesion_volume_value(entry, left_col=lvol_col, right_col=rvol_col, total_col=tvol_col, mode=lesion_vol_mode)
        else:
            cat, med_hit, lat_hit, is_vis = ("unknown", None, None, False)
            pct = float("nan")
            tg = "other"
            vol = float("nan")

        vis_list.append(bool(is_vis))
        lesion_list.append(float(pct) if np.isfinite(pct) else float("nan"))
        cat_list.append(str(cat))
        med_hit_list.append(med_hit)
        lat_hit_list.append(lat_hit)
        treat_group_list.append(str(tg))
        lesion_vol_list.append(float(vol) if np.isfinite(vol) else float("nan"))

    merged["is_visible"] = vis_list
    merged["lesion_pct"] = lesion_list
    merged["hit_category"] = cat_list
    merged["medial_hit_any"] = med_hit_list
    merged["lateral_hit_any"] = lat_hit_list
    merged["treatment_group"] = treat_group_list  # sham | nma | other
    merged["lesion_volume"] = lesion_vol_list     # may be NaN if not found

    # Save table
    tp_tag = f"topP{int(top_percentile)}" if top_percentile is not None else "all"
    table_path = out_dir / (
        f"pre_post_variance_table__pre_{pre_group.replace(' ','_')}"
        f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}.csv"
    )
    merged.to_csv(table_path, index=False)

    # Identity stats (sham vs nma)
    identity_stats: Dict[str, Dict[str, float]] = {}
    stats_text_block: str = ""
    if compute_identity_stats:
        sham_mask = merged["treatment_group"].astype(str).str.lower().eq("sham")
        nma_mask = merged["treatment_group"].astype(str).str.lower().eq("nma")

        x_sham = merged.loc[sham_mask, "pre_variance"].to_numpy(dtype=float)
        y_sham = merged.loc[sham_mask, "post_variance"].to_numpy(dtype=float)
        x_nma = merged.loc[nma_mask, "pre_variance"].to_numpy(dtype=float)
        y_nma = merged.loc[nma_mask, "post_variance"].to_numpy(dtype=float)

        identity_stats["sham"] = _identity_agreement_stats(x_sham, y_sham, log_scale=log_scale)
        identity_stats["nma"] = _identity_agreement_stats(x_nma, y_nma, log_scale=log_scale)

        sham_txt = _format_identity_stats_block("SHAM", identity_stats["sham"], log_scale=log_scale)
        nma_txt = _format_identity_stats_block("NMA", identity_stats["nma"], log_scale=log_scale)
        stats_text_block = sham_txt + "\n\n" + nma_txt

        print("\n[IDENTITY LINE AGREEMENT: y=x]")
        print(sham_txt)
        print()
        print(nma_txt)
        print()

    # Title core
    title_core = (
        f"Pre vs Post top {int(top_percentile)}% variance"
        if top_percentile is not None
        else "Pre vs Post variance"
    )

    # -------------------------------------------------------------------------
    # Plot 1: Discrete
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12.2, 6.8))
    x = merged["pre_variance"].to_numpy(dtype=float)
    y = merged["post_variance"].to_numpy(dtype=float)

    handles1: List[Any] = []

    if metadata_excel is None:
        ax1.scatter(x, y, s=marker_size, alpha=alpha, edgecolors="none")
        sub = f"pre={pre_group}, post={post_group}, rank_on={rank_on}, min_N_phrases={min_n_phrases}"
        sub += ", log–log" if log_scale else ", linear"
        ax1.set_title(f"{title_core}\n({sub})")
    else:
        cats = merged["hit_category"].astype(str)

        plot_order = [
            "Area X visible (Medial+Lateral hit)",
            "Area X visible (Medial only)",
            "Area X visible (Lateral only)",
            "large lesion Area X not visible",
            "sham saline injection",
            "miss",
            "unknown",
        ]

        for cat in plot_order:
            m = (cats == cat).to_numpy(dtype=bool)
            if not np.any(m):
                continue
            col = discrete_color_map.get(cat, "gray")

            a = alpha
            if cat in {"miss"}:
                a = 0.55
            if cat in {"unknown"}:
                a = 0.35

            ax1.scatter(
                x[m],
                y[m],
                s=marker_size,
                alpha=a,
                color=col,
                edgecolors="none",
            )

            handles1.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="none",
                    markersize=9,
                    markerfacecolor=col,
                    markeredgecolor="none",
                    label=cat,
                )
            )

        m_other = ~np.isin(cats.to_numpy(), np.array(plot_order, dtype=object))
        if np.any(m_other):
            ax1.scatter(x[m_other], y[m_other], s=marker_size, alpha=0.25, color="gray", edgecolors="none")
            handles1.append(
                mlines.Line2D([], [], marker="o", linestyle="none", markersize=9,
                             markerfacecolor="gray", markeredgecolor="none", label="other")
            )

        sub = f"pre={pre_group}, post={post_group}, rank_on={rank_on}, min_N_phrases={min_n_phrases}"
        sub += ", log–log" if log_scale else ", linear"
        ax1.set_title(f"{title_core}\n({sub})")

    lo, hi = _apply_scales(ax1, x, y, log_scale=log_scale)
    ax1.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1.5)
    handles1.append(mlines.Line2D([], [], color="red", linestyle="--", label="y=x"))

    ax1.set_xlabel(f"{pre_group} variance (ms$^2$)")
    ax1.set_ylabel(f"{post_group} variance (ms$^2$)")
    _pretty_axes(ax1)

    legend1 = None
    if handles1:
        # Put legend on the right (outside); stats will be placed underneath
        legend1 = ax1.legend(
            handles=handles1,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.80),
            frameon=True,
            framealpha=0.9,
            fontsize=10,
            borderaxespad=0.0,
        )

    if compute_identity_stats and annotate_identity_stats and stats_text_block and legend1 is not None:
        _place_stats_under_legend_axcoords(
            fig1,
            ax1,
            legend1,
            stats_text_block,
            pad_axes_y=0.02,
            fontsize=10.0,
        )

    fig1.tight_layout()
    fig_path_discrete = out_dir / (
        f"pre_post_variance_scatter__pre_{pre_group.replace(' ','_')}"
        f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}__discrete.png"
    )
    fig1.savefig(fig_path_discrete, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Continuous lesion % shading (with discrete overlays)
    # -------------------------------------------------------------------------
    fig_path_cont: Optional[Path] = None

    if (metadata_excel is not None) and make_continuous_lesion_pct_plot:
        x_vis: List[float] = []
        y_vis: List[float] = []
        c_vis: List[float] = []
        discrete_points: Dict[str, Tuple[List[float], List[float]]] = {}

        for _, row in merged.iterrows():
            aid = str(row[animal_col])
            entry = meta_dict.get(aid, {})

            x_val = float(row["pre_variance"])
            y_val = float(row["post_variance"])

            if entry:
                is_vis = _is_visible_from_meta(entry.get(area_visible_col, None))
                pct = _get_lesion_pct_value(
                    entry,
                    left_col=left_lesion_pct_col,
                    right_col=right_lesion_pct_col,
                    lesion_pct_mode=lesion_pct_mode,
                )
                cat, _, _, _ = _classify_point_category(
                    entry, area_visible_col=area_visible_col, treatment_type_col=treatment_type_col
                )
            else:
                is_vis = False
                pct = float("nan")
                cat = "unknown"

            # Continuous only for visible + finite lesion%
            if is_vis and np.isfinite(pct):
                x_vis.append(x_val)
                y_vis.append(y_val)
                c_vis.append(float(pct))
            else:
                discrete_points.setdefault(cat, ([], []))
                discrete_points[cat][0].append(x_val)
                discrete_points[cat][1].append(y_val)

        if x_vis or discrete_points:
            fig2, ax2 = plt.subplots(figsize=(12.8, 6.8))

            sc = None
            legend_handles: List[Any] = []

            if x_vis:
                x_arr_vis = np.array(x_vis, dtype=float)
                y_arr_vis = np.array(y_vis, dtype=float)
                c_arr = np.array(c_vis, dtype=float)

                if c_arr.size > 0 and np.any(np.isfinite(c_arr)):
                    cmax = float(np.nanmax(c_arr))
                    vmin, vmax = (0.0, cmax) if (np.isfinite(cmax) and cmax > 0) else (0.0, 1.0)
                else:
                    vmin, vmax = 0.0, 1.0

                sc = ax2.scatter(
                    x_arr_vis,
                    y_arr_vis,
                    c=c_arr,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    s=marker_size,
                    alpha=alpha,
                    edgecolors="none",
                )

                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        marker="o",
                        linestyle="none",
                        markersize=9,
                        markerfacecolor="none",
                        markeredgecolor="blue",
                        markeredgewidth=1.2,
                        label="Area X visible (see colorbar)",
                    )
                )

            # Discrete overlays
            overlay_order = [
                "Area X visible (Medial+Lateral hit)",
                "Area X visible (Medial only)",
                "Area X visible (Lateral only)",
                "large lesion Area X not visible",
                "sham saline injection",
                "miss",
                "unknown",
            ]
            overlay_order += [k for k in discrete_points.keys() if k not in overlay_order]

            for cat in overlay_order:
                if cat not in discrete_points:
                    continue
                xs, ys = discrete_points[cat]
                color = discrete_color_map.get(cat, "gray")

                # make sham solid (no outline)
                if cat == "sham saline injection":
                    edgecolors = "none"
                    linewidths = 0.0
                    legend_edgecolor = "none"
                    legend_edgewidth = 0.0
                else:
                    edgecolors = "black"
                    linewidths = 0.4
                    legend_edgecolor = "black"
                    legend_edgewidth = 0.8

                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)

                ax2.scatter(
                    xs_arr,
                    ys_arr,
                    s=marker_size,
                    alpha=0.9 if cat not in {"unknown"} else 0.35,
                    color=color,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )

                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        marker="o",
                        linestyle="none",
                        markersize=9,
                        markerfacecolor=color,
                        markeredgecolor=legend_edgecolor,
                        markeredgewidth=legend_edgewidth,
                        label=cat,
                    )
                )

            # limits on all points
            all_x_arrays: List[np.ndarray] = []
            all_y_arrays: List[np.ndarray] = []
            if x_vis:
                all_x_arrays.append(np.array(x_vis, dtype=float))
                all_y_arrays.append(np.array(y_vis, dtype=float))
            for xs, ys in discrete_points.values():
                all_x_arrays.append(np.array(xs, dtype=float))
                all_y_arrays.append(np.array(ys, dtype=float))

            x_all_full = np.concatenate(all_x_arrays) if all_x_arrays else np.array([], dtype=float)
            y_all_full = np.concatenate(all_y_arrays) if all_y_arrays else np.array([], dtype=float)

            lo2, hi2 = _apply_scales(ax2, x_all_full, y_all_full, log_scale=log_scale)
            ax2.plot([lo2, hi2], [lo2, hi2], linestyle="--", color="red", linewidth=1.5)

            # Title (wrapped)
            line1, line2 = _build_subtitle_lines(
                pre_group=pre_group,
                post_group=post_group,
                rank_on=rank_on,
                min_n_phrases=min_n_phrases,
                log_scale=log_scale,
                include_visibility_note=True,
            )
            if line2:
                ax2.set_title(f"{title_core}\n({line1})\n({line2})")
            else:
                ax2.set_title(f"{title_core}\n({line1})")

            ax2.set_xlabel(f"{pre_group} variance (ms$^2$)")
            ax2.set_ylabel(f"{post_group} variance (ms$^2$)")
            _pretty_axes(ax2)

            # Colorbar + legend placement
            if sc is not None:
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="3.5%", pad=0.10)
                cbar = fig2.colorbar(sc, cax=cax)
                cbar.set_label("% of Area X lesioned", fontsize=11)
                x_anchor = 1.18  # legend further right than colorbar
            else:
                x_anchor = 1.05

            legend_handles.append(mlines.Line2D([], [], color="red", linestyle="--", label="y=x"))

            legend2 = None
            if legend_handles:
                legend2 = fig2.legend(
                    handles=legend_handles,
                    loc="upper left",
                    bbox_to_anchor=(x_anchor, 0.80),
                    borderaxespad=0.0,
                    frameon=True,
                    facecolor="white",
                    framealpha=0.85,
                    fontsize=10,
                )

            if compute_identity_stats and annotate_identity_stats and stats_text_block and legend2 is not None:
                _place_stats_under_legend_figcoords(
                    fig2,
                    legend2,
                    stats_text_block,
                    pad_fig_y=0.01,
                    fontsize=10.0,
                )

            fig2.tight_layout()
            fig_path_cont = out_dir / (
                f"pre_post_variance_scatter__pre_{pre_group.replace(' ','_')}"
                f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}__continuous.png"
            )
            fig2.savefig(fig_path_cont, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig2)

        else:
            if verbose:
                print(
                    "[INFO] Continuous lesion-% plot skipped: no points available.\n"
                    "       Check metadata_excel and histology_volumes_dir."
                )

    # -------------------------------------------------------------------------
    # Plot 3: Ratio (post/pre) vs lesion volume
    #   - SHAM points plotted LAST (on top) in RED
    # -------------------------------------------------------------------------
    fig_path_ratio: Optional[Path] = None

    if (metadata_excel is not None) and make_ratio_vs_volume_plots:
        vol = merged["lesion_volume"].to_numpy(dtype=float)
        pre_v = merged["pre_variance"].to_numpy(dtype=float)
        post_v = merged["post_variance"].to_numpy(dtype=float)

        good = np.isfinite(vol) & np.isfinite(pre_v) & np.isfinite(post_v) & (pre_v > 0)
        vol2 = vol[good]
        ratio = (post_v[good] / pre_v[good]).astype(float)
        tg = merged.loc[good, "treatment_group"].astype(str).str.lower().to_numpy()

        if vol2.size > 0:
            fig3, ax3 = plt.subplots(figsize=(9.6, 6.4))

            color_map = {"sham": "red", "nma": "blue", "other": "gray"}

            # Plot order matters: draw "other" first, then NMA, then SHAM last so SHAM is on top.
            plot_order = [
                ("other", 0.25, 1),  # (group, alpha, zorder)
                ("nma",   0.75, 2),
                ("sham",  0.90, 3),
            ]

            for group_name, a, z in plot_order:
                m = (tg == group_name)
                if not np.any(m):
                    continue

                ax3.scatter(
                    vol2[m],
                    ratio[m],
                    s=marker_size,
                    alpha=a,
                    color=color_map.get(group_name, "gray"),
                    edgecolors="none",
                    label=group_name,
                    zorder=z,
                )

            ax3.axhline(1.0, linestyle="--", color="red", linewidth=1.5, label="ratio=1 (y=x)")

            if volume_logx:
                ax3.set_xscale("log")
            if ratio_logy:
                ax3.set_yscale("log")

            ax3.set_xlabel("Lesion volume (metadata units)")
            ax3.set_ylabel(f"{post_group}/{pre_group} variance ratio")
            ax3.set_title(f"Variance ratio vs lesion volume\n(pre={pre_group}, post={post_group}, {tp_tag}_{rank_on})")
            _pretty_axes(ax3)
            ax3.legend(loc="best", frameon=True, framealpha=0.9)

            fig3.tight_layout()
            fig_path_ratio = out_dir / (
                f"variance_ratio_vs_lesion_volume__pre_{pre_group.replace(' ','_')}"
                f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}.png"
            )
            fig3.savefig(fig_path_ratio, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig3)
        else:
            if verbose:
                print("[INFO] Ratio-vs-volume plot skipped: no finite lesion_volume values found.")

    return {
        "table": merged,
        "table_path": str(table_path),
        "figure_path_discrete": str(fig_path_discrete),
        "figure_path_continuous": (str(fig_path_cont) if fig_path_cont is not None else None),
        "figure_path_ratio_vs_volume": (str(fig_path_ratio) if fig_path_ratio is not None else None),
        "identity_stats": identity_stats,
        "n_points": int(len(merged)),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description="Pre vs Post variance scatterplots with outlier filtering and Area X metadata coloring."
    )
    p.add_argument("--csv", dest="csv_path", required=True, type=str, help="Path to compiled phrase-duration stats CSV.")
    p.add_argument("--out_dir", dest="out_dir", required=True, type=str, help="Directory to save outputs.")
    p.add_argument("--pre_group", default="Late Pre", type=str)
    p.add_argument("--post_group", default="Post", type=str)
    p.add_argument("--variance_col", default="Variance_ms2", type=str)
    p.add_argument("--top_percentile", default=90.0, type=float,
                   help="Within-animal cutoff percentile (e.g., 90 keeps top 10%). Use 0 to disable.")
    p.add_argument("--rank_on", default="post", choices=["pre", "post", "max"], type=str)
    p.add_argument("--min_n_phrases", default=0, type=int)
    p.add_argument("--metadata_excel", default=None, type=str)
    p.add_argument("--meta_sheet_name", default="0", type=str)
    p.add_argument("--histology_volumes_dir", default=None, type=str,
                   help="Directory containing '*_final_volumes.json' files.")
    p.add_argument("--lesion_pct_mode", default="avg", choices=["left", "right", "avg"], type=str)
    p.add_argument("--no_continuous", action="store_true", help="Disable continuous lesion-% plot.")
    p.add_argument("--linear", action="store_true", help="Use linear axes (default is log-log).")
    p.add_argument("--cmap", default="Purples", type=str, help="Colormap for continuous lesion%% shading.")
    p.add_argument("--show", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # identity stats
    p.add_argument("--no_identity_stats", action="store_true", help="Disable identity-line agreement stats.")
    p.add_argument("--no_identity_annot", action="store_true", help="Do not annotate identity stats on plots.")
    p.add_argument("--nma_token", default="nma", type=str, help="Token used to identify NMA treatments in metadata strings.")

    # ratio vs volume
    p.add_argument("--no_ratio_vs_volume", action="store_true", help="Disable ratio-vs-lesion-volume plot.")
    p.add_argument("--left_lesion_vol_col", default=None, type=str)
    p.add_argument("--right_lesion_vol_col", default=None, type=str)
    p.add_argument("--total_lesion_vol_col", default=None, type=str)
    p.add_argument("--lesion_vol_mode", default="sum", choices=["sum", "avg", "left", "right"], type=str)
    p.add_argument("--ratio_logy", action="store_true", help="Log-scale y-axis for ratio plot.")
    p.add_argument("--volume_logx", action="store_true", help="Log-scale x-axis for volume plot.")

    return p


def _parse_sheet_name(x: str) -> Union[int, str]:
    xs = str(x).strip()
    if xs.isdigit():
        return int(xs)
    return xs


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    tp = float(args.top_percentile)
    tp_val: Optional[float] = None if tp <= 0 else tp

    res = plot_pre_post_variance_scatter(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        pre_group=args.pre_group,
        post_group=args.post_group,
        variance_col=args.variance_col,
        top_percentile=tp_val,
        rank_on=args.rank_on,
        min_n_phrases=args.min_n_phrases,
        metadata_excel=args.metadata_excel,
        meta_sheet_name=_parse_sheet_name(args.meta_sheet_name),
        histology_volumes_dir=args.histology_volumes_dir,
        lesion_pct_mode=args.lesion_pct_mode,
        make_continuous_lesion_pct_plot=(not args.no_continuous),
        cmap_name=args.cmap,
        log_scale=(not args.linear),
        show=args.show,
        verbose=args.verbose,
        compute_identity_stats=(not args.no_identity_stats),
        annotate_identity_stats=(not args.no_identity_annot),
        nma_token=args.nma_token,
        make_ratio_vs_volume_plots=(not args.no_ratio_vs_volume),
        left_lesion_vol_col=args.left_lesion_vol_col,
        right_lesion_vol_col=args.right_lesion_vol_col,
        total_lesion_vol_col=args.total_lesion_vol_col,
        lesion_vol_mode=args.lesion_vol_mode,
        ratio_logy=args.ratio_logy,
        volume_logx=args.volume_logx,
    )

    print("Discrete plot:", res["figure_path_discrete"])
    print("Continuous plot:", res["figure_path_continuous"])
    print("Ratio vs volume plot:", res["figure_path_ratio_vs_volume"])
    print("Table:", res["table_path"])


"""
Spyder console example (FULL)

This example:
  1) imports the updated outlier_graphs.py
  2) runs the pre-vs-post variance scatter (discrete + continuous lesion-% plot)
  3) computes/prints identity-line agreement stats separately for SHAM vs NMA
     (Pearson r + Lin's CCC + MAE/RMSE deviation from y=x)
  4) makes the ratio-vs-lesion-volume plot (post/pre vs lesion volume)

Adjust ONLY the paths (and optionally the lesion volume column names).


from pathlib import Path
import sys, importlib

# Folder that contains outlier_graphs.py (adjust if needed)
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import outlier_graphs as og
importlib.reload(og)

# -----------------------
# Inputs
# -----------------------
compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")
histology_dir = Path("/Volumes/my_own_SSD/histology_files/lesion_quantification_csvs_jsons")

# Output directory
out_dir = compiled_csv.parent / "outlier_variance_scatter"
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# Run
# -----------------------
res = og.plot_pre_post_variance_scatter(
    csv_path=compiled_csv,
    out_dir=out_dir,

    # Groups / columns
    pre_group="Late Pre",
    post_group="Post",
    variance_col="Variance_ms2",
    group_col="Group",
    animal_col="Animal ID",
    syllable_col="Syllable",
    nphrases_col="N_phrases",

    # Outlier selection (within each animal)
    top_percentile=70,         # keep top 10% highest-variance syllables per bird
    rank_on="post",            # rank by post variance (alternatives: "pre", "max")
    min_n_phrases=50,          # set 0 to disable

    # Metadata + histology (for coloring + lesion% and volume)
    metadata_excel=metadata_xlsx,
    meta_sheet_name=0,         # sheet index or name
    histology_volumes_dir=histology_dir,   # contains *_final_volumes.json
    area_visible_col="Area X visible in histology?",
    treatment_type_col="Treatment type",
    left_lesion_pct_col="L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col="R_Percent_of_Area_X_Lesioned_pct",
    lesion_pct_mode="avg",     # "left" | "right" | "avg"

    # -------- NEW: identity-line stats (SHAM vs NMA) --------
    compute_identity_stats=True,     # prints stats to terminal
    annotate_identity_stats=True,    # also writes stats onto plots
    nma_token="nma",                # token used to identify NMA in metadata strings

    # Plots
    make_continuous_lesion_pct_plot=True,
    cmap_name="Purples",
    log_scale=True,                 # log-log axes on the pre/post scatter
    show=True,
    verbose=True,

    # -------- NEW: ratio vs lesion volume plot --------
    make_ratio_vs_volume_plots=True,

    # If auto-detect fails, set these explicitly to your metadata keys:
    # left_lesion_vol_col="L_Area_X_Lesion_Volume_mm3",
    # right_lesion_vol_col="R_Area_X_Lesion_Volume_mm3",
    # total_lesion_vol_col="Total_Area_X_Lesion_Volume_mm3",
    lesion_vol_mode="sum",          # "sum" | "avg" | "left" | "right"

    # Optional axis scales for ratio-vs-volume plot
    ratio_logy=False,               # set True if ratios span orders of magnitude
    volume_logx=False,              # set True if volumes span orders of magnitude
)

# -----------------------
# Outputs
# -----------------------
print("\n=== Saved outputs ===")
print("Saved table:", res["table_path"])
print("Discrete figure:", res["figure_path_discrete"])
print("Continuous figure:", res["figure_path_continuous"])
print("Ratio vs volume figure:", res["figure_path_ratio_vs_volume"])
print("N points plotted:", res["n_points"])

print("\n=== Identity-line agreement stats (returned dict) ===")
print(res["identity_stats"])

"""