# -*- coding: utf-8 -*-
"""
outlier_graphs.py

Pre vs Post variance scatterplots with outlier selection + Area X metadata coloring
(using organize_metadata_excel.build_areax_metadata + *_final_volumes.json attachments).

Added (recent updates)
----------------------
A) Identity-line agreement stats (SHAM vs NMA): Pearson r, Lin's CCC, MAE/RMSE of
   delta relative to y=x (on log10 scale if log-log).

B) Permutation test on mean log-ratios between groups:
      log_ratio = log10(post/pre)
   - One-tailed: NMA > SHAM
   - Two-tailed: |NMA - SHAM| large in either direction

C) NEW: Tail/outlier metrics on log10(variance) per bird (using ALL pre/post pairs,
   i.e., BEFORE top_percentile filtering), then compare NMA vs SHAM.
   This supports the “probability of outliers increases post” hypothesis.

   Per-bird (computed on log10 variance separately for pre and post):
     - Quantiles: Q25, Q50, Q75, Q90, Q95, Q99
     - IQR = Q75 - Q25
     - Upper-tail spreads: (Q95-Q50), (Q90-Q50), (Q95-Q75), (Q90-Q75)
     - Tail-heaviness proxy: (Q95-Q75)/(Q75-Q50) with divide-by-zero guard

   Outlier “fences” computed from PRE only (per bird) on log10(pre_variance):
     - Mild:   lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
     - Extreme lower = Q1 - 3.0*IQR, upper = Q3 + 3.0*IQR

   Then counts/probabilities computed in PRE and POST relative to those PRE fences:
     - p_mild_high_pre/post, p_extreme_high_pre/post (and low-side too)
     - RD = p_post - p_pre
     - RR = (p_post+cc)/(p_pre+cc), logRR = ln(RR)

   Group comparisons (NMA vs SHAM) are done on per-bird deltas (post-pre):
     - permutation tests (one-tailed NMA > SHAM and two-tailed)
     - printed to terminal and appended to the right-side annotation block

D) Ratio vs lesion % plot: y=post/pre (variance ratio), x=% Area X lesioned
   includes SHAM (red) + NMA (blue). SHAM missing % is plotted at x=0 (linear x).
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
      - "unknown" / sham / large lesion not visible => None
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    low = s.lower()

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
    """Fallback: infer Medial/Lateral hit by scanning entry["injections"] for Y/N flags."""
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
    """Set x/y scales and choose symmetric limits based on data."""
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
    """Decide the DISCRETE category label for plotting."""
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
    """Lin's Concordance Correlation Coefficient (CCC): agreement with identity line."""
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
    If log_scale, delta = log10(post/pre). Otherwise delta = post - pre.
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
        delta = np.log10(post2 / pre2)
    else:
        x = pre
        y = post
        delta = post - pre

    out["pearson_r"] = _pearson_r(x, y)
    out["ccc"] = _lin_ccc(x, y)
    out["mae_delta"] = float(np.mean(np.abs(delta))) if delta.size else float("nan")
    out["rmse_delta"] = float(np.sqrt(np.mean(delta ** 2))) if delta.size else float("nan")
    return out


def _format_identity_stats_block(label: str, stats: Dict[str, float], *, log_scale: bool) -> str:
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
    """Place stats text UNDER an axes legend (even if legend is outside axes)."""
    if not stats_text or legend_obj is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = legend_obj.get_window_extent(renderer=renderer)
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
    """Place stats text UNDER a figure legend (fig.legend), using figure coords."""
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
# Permutation tests
# -----------------------------------------------------------------------------
def permutation_test_one_tailed(
    group1: Union[np.ndarray, List[float]],
    group2: Union[np.ndarray, List[float]],
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    One-tailed permutation test on mean difference (group2 - group1): P(perm_diff >= observed_diff).

    group1: control (sham)
    group2: experimental (NMA)
    """
    rng = np.random.default_rng(seed)

    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if g1.size < 1 or g2.size < 1:
        return float("nan"), float("nan")

    observed_diff = float(np.mean(g2) - np.mean(g1))
    combined = np.concatenate([g1, g2])
    n1 = g1.size

    count_extreme = 0
    for _ in range(int(n_permutations)):
        rng.shuffle(combined)
        perm_diff = float(np.mean(combined[n1:]) - np.mean(combined[:n1]))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / float(n_permutations)
    return observed_diff, float(p_value)


def permutation_test_two_tailed(
    group1: Union[np.ndarray, List[float]],
    group2: Union[np.ndarray, List[float]],
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Two-tailed permutation test on mean difference (group2 - group1):
      p = P(|perm_diff| >= |observed_diff|).
    """
    rng = np.random.default_rng(seed)

    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if g1.size < 1 or g2.size < 1:
        return float("nan"), float("nan")

    observed_diff = float(np.mean(g2) - np.mean(g1))
    combined = np.concatenate([g1, g2])
    n1 = g1.size

    count_extreme = 0
    abs_obs = abs(observed_diff)
    for _ in range(int(n_permutations)):
        rng.shuffle(combined)
        perm_diff = float(np.mean(combined[n1:]) - np.mean(combined[:n1]))
        if abs(perm_diff) >= abs_obs:
            count_extreme += 1

    p_value = count_extreme / float(n_permutations)
    return observed_diff, float(p_value)


def _compute_log_ratios(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Compute log10(post/pre) safely (drops nonfinite/<=0)."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
    pre = pre[m]
    post = post[m]
    if pre.size == 0:
        return np.array([], dtype=float)
    return np.log10(post / pre)


def _format_permtest_block(
    *,
    mean_sham: float,
    mean_nma: float,
    observed_diff: float,
    p_one: float,
    p_two: float,
    n_sham: int,
    n_nma: int,
) -> str:
    if not (np.isfinite(observed_diff) and np.isfinite(p_one) and np.isfinite(p_two)):
        return "Permutation test (log-ratio): insufficient data"
    return (
        "Permutation test on mean log10(post/pre)\n"
        "One-tailed: NMA > SHAM; Two-tailed: |diff| large\n"
        f"SHAM mean={mean_sham:.4f} (n={n_sham})\n"
        f"NMA  mean={mean_nma:.4f} (n={n_nma})\n"
        f"Observed diff (NMA-SHAM)={observed_diff:.4f}\n"
        f"p(one)={p_one:.4f}   p(two)={p_two:.4f}"
    )


# -----------------------------------------------------------------------------
# Tail / outlier metrics on log10 variance (per-bird)
# -----------------------------------------------------------------------------
def _tail_metrics_from_values(
    x: np.ndarray,
    *,
    q_list: Tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
) -> Dict[str, float]:
    """
    Compute quantiles + tail spreads on a 1D array x (expects finite values).
    Returns NaNs if insufficient values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[str, float] = {"n": float(x.size)}
    if x.size < 2:
        # still allow median/quantiles with 1 point, but spreads are unstable; keep simple.
        for q in q_list:
            out[f"q{int(round(q*100))}"] = float(np.nan) if x.size == 0 else float(np.nanpercentile(x, q * 100.0))
        out.update(
            {
                "iqr": float("nan"),
                "spread_q95_q50": float("nan"),
                "spread_q90_q50": float("nan"),
                "spread_q95_q75": float("nan"),
                "spread_q90_q75": float("nan"),
                "tail_heaviness": float("nan"),
            }
        )
        return out

    # quantiles
    qvals = {q: float(np.nanpercentile(x, q * 100.0)) for q in q_list}
    for q, v in qvals.items():
        out[f"q{int(round(q*100))}"] = float(v)

    # convenient names (guard if missing)
    q25 = qvals.get(0.25, float("nan"))
    q50 = qvals.get(0.50, float("nan"))
    q75 = qvals.get(0.75, float("nan"))
    q90 = qvals.get(0.90, float("nan"))
    q95 = qvals.get(0.95, float("nan"))

    iqr = q75 - q25 if (np.isfinite(q75) and np.isfinite(q25)) else float("nan")
    out["iqr"] = float(iqr)

    out["spread_q95_q50"] = float(q95 - q50) if (np.isfinite(q95) and np.isfinite(q50)) else float("nan")
    out["spread_q90_q50"] = float(q90 - q50) if (np.isfinite(q90) and np.isfinite(q50)) else float("nan")
    out["spread_q95_q75"] = float(q95 - q75) if (np.isfinite(q95) and np.isfinite(q75)) else float("nan")
    out["spread_q90_q75"] = float(q90 - q75) if (np.isfinite(q90) and np.isfinite(q75)) else float("nan")

    denom = (q75 - q50) if (np.isfinite(q75) and np.isfinite(q50)) else float("nan")
    numer = (q95 - q75) if (np.isfinite(q95) and np.isfinite(q75)) else float("nan")
    if np.isfinite(denom) and abs(denom) > 1e-12 and np.isfinite(numer):
        out["tail_heaviness"] = float(numer / denom)
    else:
        out["tail_heaviness"] = float("nan")

    return out


def _tukey_fences_from_pre(
    x_pre: np.ndarray,
) -> Dict[str, float]:
    """
    Tukey fences computed on x_pre (expects finite values, typically log10(pre_variance)).
    Returns NaNs if insufficient.
    """
    x_pre = np.asarray(x_pre, dtype=float)
    x_pre = x_pre[np.isfinite(x_pre)]
    out: Dict[str, float] = {"n_pre_for_fence": float(x_pre.size)}
    if x_pre.size < 2:
        out.update(
            {
                "q1": float("nan"),
                "q3": float("nan"),
                "iqr": float("nan"),
                "lower_mild": float("nan"),
                "upper_mild": float("nan"),
                "lower_extreme": float("nan"),
                "upper_extreme": float("nan"),
            }
        )
        return out

    q1 = float(np.nanpercentile(x_pre, 25.0))
    q3 = float(np.nanpercentile(x_pre, 75.0))
    iqr = float(q3 - q1)

    out["q1"] = q1
    out["q3"] = q3
    out["iqr"] = iqr

    if np.isfinite(iqr):
        out["lower_mild"] = float(q1 - 1.5 * iqr)
        out["upper_mild"] = float(q3 + 1.5 * iqr)
        out["lower_extreme"] = float(q1 - 3.0 * iqr)
        out["upper_extreme"] = float(q3 + 3.0 * iqr)
    else:
        out["lower_mild"] = float("nan")
        out["upper_mild"] = float("nan")
        out["lower_extreme"] = float("nan")
        out["upper_extreme"] = float("nan")

    return out


def _per_bird_tail_and_outlier_probabilities(
    df_pairs: pd.DataFrame,
    *,
    animal_col: str = "Animal ID",
    treatment_group_col: str = "treatment_group",
    pre_col: str = "pre_variance",
    post_col: str = "post_variance",
    min_syllables_per_bird: int = 5,
    rr_cc: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute per-bird tail metrics on log10 variance (pre & post) + deltas,
    and outlier probabilities based on PRE fences applied to both pre and post.

    df_pairs should contain ALL valid pre/post pairs (i.e., BEFORE top_percentile filtering).
    """
    rows: List[Dict[str, Any]] = []

    for aid, sub in df_pairs.groupby(animal_col, sort=False):
        tg = str(sub[treatment_group_col].iloc[0]).lower() if treatment_group_col in sub.columns else "other"

        pre = sub[pre_col].to_numpy(dtype=float)
        post = sub[post_col].to_numpy(dtype=float)

        good = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
        pre = pre[good]
        post = post[good]

        n = int(pre.size)
        if n < max(1, int(min_syllables_per_bird)):
            # keep row but mostly NaN; still useful for debugging
            rows.append(
                {
                    animal_col: str(aid),
                    "treatment_group": tg,
                    "n_syllables": n,
                }
            )
            continue

        log_pre = np.log10(pre)
        log_post = np.log10(post)

        # Tail metrics pre/post
        tm_pre = _tail_metrics_from_values(log_pre)
        tm_post = _tail_metrics_from_values(log_post)

        # Deltas post - pre for a few key quantities
        def _get(tm: Dict[str, float], k: str) -> float:
            v = tm.get(k, float("nan"))
            return float(v) if np.isfinite(v) else float("nan")

        delta_cols = {}
        for k in [
            "q50",
            "q90",
            "q95",
            "q99",
            "iqr",
            "spread_q95_q50",
            "spread_q90_q50",
            "spread_q95_q75",
            "spread_q90_q75",
            "tail_heaviness",
        ]:
            delta_cols[f"delta_{k}"] = _get(tm_post, k) - _get(tm_pre, k)

        # Tukey fences from PRE
        fences = _tukey_fences_from_pre(log_pre)
        upper_mild = fences.get("upper_mild", float("nan"))
        upper_ext = fences.get("upper_extreme", float("nan"))
        lower_mild = fences.get("lower_mild", float("nan"))
        lower_ext = fences.get("lower_extreme", float("nan"))

        # Outlier counts/probabilities (relative to PRE fences)
        def _count_prob(x: np.ndarray, thr: float, side: str) -> Tuple[int, float]:
            if not np.isfinite(thr):
                return 0, float("nan")
            if side == "high":
                c = int(np.sum(x > thr))
            else:
                c = int(np.sum(x < thr))
            p = c / float(x.size) if x.size > 0 else float("nan")
            return c, float(p)

        n_mild_high_pre, p_mild_high_pre = _count_prob(log_pre, upper_mild, "high")
        n_mild_high_post, p_mild_high_post = _count_prob(log_post, upper_mild, "high")
        n_ext_high_pre, p_ext_high_pre = _count_prob(log_pre, upper_ext, "high")
        n_ext_high_post, p_ext_high_post = _count_prob(log_post, upper_ext, "high")

        n_mild_low_pre, p_mild_low_pre = _count_prob(log_pre, lower_mild, "low")
        n_mild_low_post, p_mild_low_post = _count_prob(log_post, lower_mild, "low")
        n_ext_low_pre, p_ext_low_pre = _count_prob(log_pre, lower_ext, "low")
        n_ext_low_post, p_ext_low_post = _count_prob(log_post, lower_ext, "low")

        # RD and RR (high-side is what you likely care about for “fatter upper tail”)
        def _rd(p_post: float, p_pre: float) -> float:
            if not (np.isfinite(p_post) and np.isfinite(p_pre)):
                return float("nan")
            return float(p_post - p_pre)

        def _rr(p_post: float, p_pre: float, cc: float) -> Tuple[float, float]:
            if not (np.isfinite(p_post) and np.isfinite(p_pre)):
                return float("nan"), float("nan")
            rr = (p_post + cc) / (p_pre + cc)
            logrr = np.log(rr) if (rr > 0) else float("nan")
            return float(rr), float(logrr)

        rd_mild_high = _rd(p_mild_high_post, p_mild_high_pre)
        rd_ext_high = _rd(p_ext_high_post, p_ext_high_pre)
        rr_mild_high, logrr_mild_high = _rr(p_mild_high_post, p_mild_high_pre, rr_cc)
        rr_ext_high, logrr_ext_high = _rr(p_ext_high_post, p_ext_high_pre, rr_cc)

        rows.append(
            {
                animal_col: str(aid),
                "treatment_group": tg,
                "n_syllables": n,
                # fences
                "pre_fence_q1": float(fences.get("q1", float("nan"))),
                "pre_fence_q3": float(fences.get("q3", float("nan"))),
                "pre_fence_iqr": float(fences.get("iqr", float("nan"))),
                "pre_upper_mild": float(upper_mild),
                "pre_upper_extreme": float(upper_ext),
                "pre_lower_mild": float(lower_mild),
                "pre_lower_extreme": float(lower_ext),
                # outlier probs
                "p_mild_high_pre": float(p_mild_high_pre),
                "p_mild_high_post": float(p_mild_high_post),
                "p_extreme_high_pre": float(p_ext_high_pre),
                "p_extreme_high_post": float(p_ext_high_post),
                "p_mild_low_pre": float(p_mild_low_pre),
                "p_mild_low_post": float(p_mild_low_post),
                "p_extreme_low_pre": float(p_ext_low_pre),
                "p_extreme_low_post": float(p_ext_low_post),
                "rd_mild_high": float(rd_mild_high),
                "rd_extreme_high": float(rd_ext_high),
                "rr_mild_high": float(rr_mild_high),
                "logrr_mild_high": float(logrr_mild_high),
                "rr_extreme_high": float(rr_ext_high),
                "logrr_extreme_high": float(logrr_ext_high),
                # tail metrics pre/post
                "pre_q25": float(tm_pre.get("q25", float("nan"))),
                "pre_q50": float(tm_pre.get("q50", float("nan"))),
                "pre_q75": float(tm_pre.get("q75", float("nan"))),
                "pre_q90": float(tm_pre.get("q90", float("nan"))),
                "pre_q95": float(tm_pre.get("q95", float("nan"))),
                "pre_q99": float(tm_pre.get("q99", float("nan"))),
                "pre_iqr": float(tm_pre.get("iqr", float("nan"))),
                "pre_spread_q95_q50": float(tm_pre.get("spread_q95_q50", float("nan"))),
                "pre_spread_q90_q50": float(tm_pre.get("spread_q90_q50", float("nan"))),
                "pre_spread_q95_q75": float(tm_pre.get("spread_q95_q75", float("nan"))),
                "pre_spread_q90_q75": float(tm_pre.get("spread_q90_q75", float("nan"))),
                "pre_tail_heaviness": float(tm_pre.get("tail_heaviness", float("nan"))),
                "post_q25": float(tm_post.get("q25", float("nan"))),
                "post_q50": float(tm_post.get("q50", float("nan"))),
                "post_q75": float(tm_post.get("q75", float("nan"))),
                "post_q90": float(tm_post.get("q90", float("nan"))),
                "post_q95": float(tm_post.get("q95", float("nan"))),
                "post_q99": float(tm_post.get("q99", float("nan"))),
                "post_iqr": float(tm_post.get("iqr", float("nan"))),
                "post_spread_q95_q50": float(tm_post.get("spread_q95_q50", float("nan"))),
                "post_spread_q90_q50": float(tm_post.get("spread_q90_q50", float("nan"))),
                "post_spread_q95_q75": float(tm_post.get("spread_q95_q75", float("nan"))),
                "post_spread_q90_q75": float(tm_post.get("spread_q90_q75", float("nan"))),
                "post_tail_heaviness": float(tm_post.get("tail_heaviness", float("nan"))),
                # deltas
                **delta_cols,
                # raw counts (handy)
                "n_mild_high_pre": int(n_mild_high_pre),
                "n_mild_high_post": int(n_mild_high_post),
                "n_extreme_high_pre": int(n_ext_high_pre),
                "n_extreme_high_post": int(n_ext_high_post),
                "n_mild_low_pre": int(n_mild_low_pre),
                "n_mild_low_post": int(n_mild_low_post),
                "n_extreme_low_pre": int(n_ext_low_pre),
                "n_extreme_low_post": int(n_ext_low_post),
            }
        )

    return pd.DataFrame(rows)


def _format_tailtest_block(
    *,
    metric_name: str,
    sham_vals: np.ndarray,
    nma_vals: np.ndarray,
    obs_diff: float,
    p_one: float,
    p_two: float,
) -> str:
    sham_vals = np.asarray(sham_vals, dtype=float)
    nma_vals = np.asarray(nma_vals, dtype=float)
    sham_vals = sham_vals[np.isfinite(sham_vals)]
    nma_vals = nma_vals[np.isfinite(nma_vals)]
    if sham_vals.size == 0 or nma_vals.size == 0:
        return f"{metric_name}: insufficient data"

    return (
        f"{metric_name}\n"
        f"SHAM mean={np.mean(sham_vals):.4f} (n={sham_vals.size})\n"
        f"NMA  mean={np.mean(nma_vals):.4f} (n={nma_vals.size})\n"
        f"Observed diff (NMA-SHAM)={obs_diff:.4f}\n"
        f"p(one)={p_one:.4f}   p(two)={p_two:.4f}"
    )


# -----------------------------------------------------------------------------
# Treatment-group helper
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

    for k in ["Treatment", "Treatment Type", "Treatment_type", "Lesion type", "Lesion", "Drug"]:
        v = entry.get(k, None)
        if v is None:
            continue
        if nma_token.lower() in str(v).lower():
            return "nma"

    return "other"


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
    # outlier selection (for plotting + log-ratio tests)
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
    # permutation test on log-ratios (NMA vs SHAM) using OUTLIER-SELECTED points
    do_perm_test: bool = True,
    perm_n_permutations: int = 10000,
    perm_seed: int = 0,
    annotate_perm_test: bool = True,
    # NEW: per-bird tail/outlier metrics using ALL points (before top_percentile)
    do_tail_metrics: bool = True,
    tail_min_syllables_per_bird: int = 5,
    tail_rr_cc: float = 1e-6,
    tail_perm_n: int = 20000,
    tail_perm_seed: int = 1,
    annotate_tail_metrics: bool = True,
    # ratio vs lesion percent
    make_ratio_vs_volume_plots: bool = True,
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

    Returns dict with:
      - table (outlier-selected)
      - table_all_pairs (before top_percentile)
      - per_bird_tail_metrics (if computed)
      - figure paths, stats dicts, test results
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

    merged_base = pd.merge(
        pre,
        post,
        on=[animal_col, syllable_col],
        how="inner",
        suffixes=("_pre", "_post"),
    )

    if merged_base.empty:
        raise ValueError("No overlapping (animal, syllable) pairs between pre and post groups.")

    merged_base.rename(
        columns={
            f"{variance_col}_pre": "pre_variance",
            f"{variance_col}_post": "post_variance",
        },
        inplace=True,
    )

    # N_phrases filter
    if min_n_phrases > 0 and nphrases_col in merged_base.columns:
        npre = merged_base.get(f"{nphrases_col}_pre", np.nan).astype(float)
        npost = merged_base.get(f"{nphrases_col}_post", np.nan).astype(float)
        keep_n = (npre >= min_n_phrases) & (npost >= min_n_phrases)
        merged_base = merged_base.loc[keep_n].copy()

    # finite / log constraints
    merged_base["pre_variance"] = merged_base["pre_variance"].astype(float)
    merged_base["post_variance"] = merged_base["post_variance"].astype(float)
    finite = np.isfinite(merged_base["pre_variance"]) & np.isfinite(merged_base["post_variance"])
    merged_base = merged_base.loc[finite].copy()

    if log_scale:
        merged_base = merged_base.loc[(merged_base["pre_variance"] > 0.0) & (merged_base["post_variance"] > 0.0)].copy()

    if merged_base.empty:
        raise ValueError("No valid finite pre/post variance pairs after filtering.")

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
            n_with_L = sum(np.isfinite(_as_float(v.get(left_lesion_pct_col, np.nan))) for v in meta_dict.values())
            n_with_R = sum(np.isfinite(_as_float(v.get(right_lesion_pct_col, np.nan))) for v in meta_dict.values())
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

    # Attach per-point metadata fields to merged_base
    vis_list: List[bool] = []
    lesion_list: List[float] = []
    cat_list: List[str] = []
    med_hit_list: List[Optional[bool]] = []
    lat_hit_list: List[Optional[bool]] = []
    treat_group_list: List[str] = []

    # Also build animal->treatment group mapping (for per-bird metrics)
    animal_to_treat: Dict[str, str] = {}

    for aid in merged_base[animal_col].astype(str).unique():
        entry = meta_dict.get(str(aid), {})
        if entry:
            tg = _infer_treatment_group(entry, treatment_type_col=treatment_type_col, nma_token=nma_token)
        else:
            tg = "other"
        animal_to_treat[str(aid)] = str(tg)

    for _, row in merged_base.iterrows():
        aid = str(row[animal_col])
        entry = meta_dict.get(aid, {})

        if entry:
            cat, med_hit, lat_hit, is_vis = _classify_point_category(
                entry, area_visible_col=area_visible_col, treatment_type_col=treatment_type_col
            )
            pct = _get_lesion_pct_value(
                entry, left_col=left_lesion_pct_col, right_col=right_lesion_pct_col, lesion_pct_mode=lesion_pct_mode
            )
            tg = animal_to_treat.get(aid, "other")
        else:
            cat, med_hit, lat_hit, is_vis = ("unknown", None, None, False)
            pct = float("nan")
            tg = "other"

        vis_list.append(bool(is_vis))
        lesion_list.append(float(pct) if np.isfinite(pct) else float("nan"))
        cat_list.append(str(cat))
        med_hit_list.append(med_hit)
        lat_hit_list.append(lat_hit)
        treat_group_list.append(str(tg))

    merged_base["is_visible"] = vis_list
    merged_base["lesion_pct"] = lesion_list  # % Area X lesioned
    merged_base["hit_category"] = cat_list
    merged_base["medial_hit_any"] = med_hit_list
    merged_base["lateral_hit_any"] = lat_hit_list
    merged_base["treatment_group"] = treat_group_list  # sham | nma | other

    # Keep a copy of ALL pairs (before outlier selection) for tail/outlier-prob metrics
    merged_all_pairs = merged_base.copy()

    # --- Top-percentile (within-animal) filter (for plotting + log-ratio tests) ---
    merged = merged_base.copy()
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
            merged["_score"] = np.maximum(merged["pre_variance"].astype(float), merged["post_variance"].astype(float))

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
        raise ValueError("No data left after top-percentile filtering. Try lowering top_percentile or min_n_phrases.")

    # Save outlier-selected table
    tp_tag = f"topP{int(top_percentile)}" if top_percentile is not None else "all"
    table_path = out_dir / (
        f"pre_post_variance_table__pre_{pre_group.replace(' ','_')}"
        f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}.csv"
    )
    merged.to_csv(table_path, index=False)

    # Save ALL-pairs table (before top-percentile)
    table_all_path = out_dir / (
        f"pre_post_variance_table_ALLPAIRS__pre_{pre_group.replace(' ','_')}"
        f"__post_{post_group.replace(' ','_')}.csv"
    )
    merged_all_pairs.to_csv(table_all_path, index=False)

    # Identity stats + Permutation test blocks (for annotation)
    identity_stats: Dict[str, Dict[str, float]] = {}
    permtest_results: Dict[str, float] = {}
    tail_results: Dict[str, Any] = {}
    per_bird_tail_df: Optional[pd.DataFrame] = None
    stats_blocks: List[str] = []

    # Build OUTLIER-SELECTED group arrays for identity/log-ratio tests
    sham_mask = merged["treatment_group"].astype(str).str.lower().eq("sham")
    nma_mask = merged["treatment_group"].astype(str).str.lower().eq("nma")

    pre_sham = merged.loc[sham_mask, "pre_variance"].to_numpy(dtype=float)
    post_sham = merged.loc[sham_mask, "post_variance"].to_numpy(dtype=float)
    pre_nma = merged.loc[nma_mask, "pre_variance"].to_numpy(dtype=float)
    post_nma = merged.loc[nma_mask, "post_variance"].to_numpy(dtype=float)

    if compute_identity_stats:
        identity_stats["sham"] = _identity_agreement_stats(pre_sham, post_sham, log_scale=log_scale)
        identity_stats["nma"] = _identity_agreement_stats(pre_nma, post_nma, log_scale=log_scale)

        sham_txt = _format_identity_stats_block("SHAM", identity_stats["sham"], log_scale=log_scale)
        nma_txt = _format_identity_stats_block("NMA", identity_stats["nma"], log_scale=log_scale)

        print("\n[IDENTITY LINE AGREEMENT: y=x]")
        print(sham_txt)
        print()
        print(nma_txt)
        print()

        stats_blocks.append(sham_txt)
        stats_blocks.append(nma_txt)

    # Permutation tests on log-ratios using OUTLIER-SELECTED points
    if do_perm_test:
        log_ratio_sham = _compute_log_ratios(pre_sham, post_sham)
        log_ratio_nma = _compute_log_ratios(pre_nma, post_nma)

        obs_diff, p_one = permutation_test_one_tailed(
            log_ratio_sham,
            log_ratio_nma,
            n_permutations=int(perm_n_permutations),
            seed=int(perm_seed),
        )
        _, p_two = permutation_test_two_tailed(
            log_ratio_sham,
            log_ratio_nma,
            n_permutations=int(perm_n_permutations),
            seed=int(perm_seed),
        )

        mean_sham = float(np.mean(log_ratio_sham)) if log_ratio_sham.size else float("nan")
        mean_nma = float(np.mean(log_ratio_nma)) if log_ratio_nma.size else float("nan")

        permtest_results = {
            "mean_log_ratio_sham": mean_sham,
            "mean_log_ratio_nma": mean_nma,
            "observed_diff_nma_minus_sham": float(obs_diff),
            "p_value_one_tailed_nma_gt_sham": float(p_one),
            "p_value_two_tailed": float(p_two),
            "n_sham": int(log_ratio_sham.size),
            "n_nma": int(log_ratio_nma.size),
            "n_permutations": int(perm_n_permutations),
            "seed": int(perm_seed),
        }

        print("[PERMUTATION TEST: log10(post/pre) mean difference]")
        print(f"Sham mean log-ratio: {mean_sham:.4f}")
        print(f"NMA mean log-ratio:  {mean_nma:.4f}")
        print(f"Observed difference (NMA - sham): {obs_diff:.4f}")
        print(f"One-tailed p-value (NMA>sham): {p_one:.4f}")
        print(f"Two-tailed p-value: {p_two:.4f}")
        if np.isfinite(p_one) and p_one < 0.05:
            print("Result (one-tailed): NMA > sham is significant at alpha=0.05")
        else:
            print("Result (one-tailed): not significant at alpha=0.05")
        print()

        if annotate_perm_test:
            stats_blocks.append(
                _format_permtest_block(
                    mean_sham=mean_sham,
                    mean_nma=mean_nma,
                    observed_diff=obs_diff,
                    p_one=p_one,
                    p_two=p_two,
                    n_sham=int(log_ratio_sham.size),
                    n_nma=int(log_ratio_nma.size),
                )
            )

    # NEW: Tail/outlier-probability metrics (per bird) using ALL pairs
    if do_tail_metrics:
        if metadata_excel is None:
            if verbose:
                print("[INFO] Tail/outlier metrics skipped: metadata_excel is None (need sham vs nma labels).")
        else:
            per_bird_tail_df = _per_bird_tail_and_outlier_probabilities(
                merged_all_pairs,
                animal_col=animal_col,
                treatment_group_col="treatment_group",
                pre_col="pre_variance",
                post_col="post_variance",
                min_syllables_per_bird=int(tail_min_syllables_per_bird),
                rr_cc=float(tail_rr_cc),
            )

            per_bird_path = out_dir / (
                f"per_bird_tail_outlier_metrics__pre_{pre_group.replace(' ','_')}"
                f"__post_{post_group.replace(' ','_')}.csv"
            )
            per_bird_tail_df.to_csv(per_bird_path, index=False)

            # Compare NMA vs SHAM on selected per-bird deltas
            df_tb = per_bird_tail_df.copy()
            df_tb["treatment_group"] = df_tb["treatment_group"].astype(str).str.lower()

            sham_birds = df_tb[df_tb["treatment_group"].eq("sham")].copy()
            nma_birds = df_tb[df_tb["treatment_group"].eq("nma")].copy()

            # choose a few key endpoints for reporting/testing
            metrics_to_test = [
                ("delta_q95", "ΔQ95 (post-pre) of log10(variance)"),
                ("delta_q90", "ΔQ90 (post-pre) of log10(variance)"),
                ("delta_iqr", "ΔIQR (post-pre) of log10(variance)"),
                ("delta_spread_q95_q50", "Δ(Q95-Q50) (post-pre) on log10(variance)"),
                ("rd_extreme_high", "Δp_extreme_high (post-pre; pre-fence)"),
            ]

            print("[TAIL/OUTLIER METRICS: per-bird; comparing NMA vs SHAM]")
            tail_results = {"per_bird_csv": str(per_bird_path), "tests": {}}

            tail_blocks: List[str] = []
            for col, label in metrics_to_test:
                if col not in df_tb.columns:
                    continue
                sham_vals = sham_birds[col].to_numpy(dtype=float)
                nma_vals = nma_birds[col].to_numpy(dtype=float)

                obs, p1 = permutation_test_one_tailed(
                    sham_vals, nma_vals,
                    n_permutations=int(tail_perm_n),
                    seed=int(tail_perm_seed),
                )
                _, p2 = permutation_test_two_tailed(
                    sham_vals, nma_vals,
                    n_permutations=int(tail_perm_n),
                    seed=int(tail_perm_seed),
                )

                tail_results["tests"][col] = {
                    "label": label,
                    "observed_diff_nma_minus_sham": float(obs),
                    "p_one_tailed_nma_gt_sham": float(p1),
                    "p_two_tailed": float(p2),
                    "n_sham_birds": int(np.sum(np.isfinite(sham_vals))),
                    "n_nma_birds": int(np.sum(np.isfinite(nma_vals))),
                    "n_permutations": int(tail_perm_n),
                    "seed": int(tail_perm_seed),
                }

                print(f"\n{label}")
                print(f"  SHAM mean={np.nanmean(sham_vals):.4f} (n={np.sum(np.isfinite(sham_vals))})")
                print(f"  NMA  mean={np.nanmean(nma_vals):.4f} (n={np.sum(np.isfinite(nma_vals))})")
                print(f"  Observed diff (NMA-SHAM)={obs:.4f}")
                print(f"  p(one)={p1:.4f}   p(two)={p2:.4f}")

                if annotate_tail_metrics:
                    tail_blocks.append(
                        _format_tailtest_block(
                            metric_name=f"Tail/outlier test: {label}",
                            sham_vals=sham_vals,
                            nma_vals=nma_vals,
                            obs_diff=obs,
                            p_one=p1,
                            p_two=p2,
                        )
                    )

            print("\n[Saved per-bird tail/outlier metrics CSV]")
            print(per_bird_path)
            print()

            if annotate_tail_metrics and tail_blocks:
                # keep this short-ish on the plot: include just the last 2 blocks
                # (usually Δ(Q95-Q50) and Δp_extreme_high), but you can change if you prefer
                blocks_to_show = tail_blocks[-2:] if len(tail_blocks) > 2 else tail_blocks
                stats_blocks.append("\n\n".join(blocks_to_show))

    stats_text_block = "\n\n".join([b for b in stats_blocks if b.strip()])

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

            ax1.scatter(x[m], y[m], s=marker_size, alpha=a, color=col, edgecolors="none")

            handles1.append(
                mlines.Line2D([], [], marker="o", linestyle="none", markersize=9,
                             markerfacecolor=col, markeredgecolor="none", label=cat)
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
        legend1 = ax1.legend(
            handles=handles1,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.80),
            frameon=True,
            framealpha=0.9,
            fontsize=10,
            borderaxespad=0.0,
        )

    if annotate_identity_stats and stats_text_block and legend1 is not None:
        _place_stats_under_legend_axcoords(fig1, ax1, legend1, stats_text_block, pad_axes_y=0.02, fontsize=10.0)

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
                    entry, left_col=left_lesion_pct_col, right_col=right_lesion_pct_col, lesion_pct_mode=lesion_pct_mode
                )
                cat, _, _, _ = _classify_point_category(
                    entry, area_visible_col=area_visible_col, treatment_type_col=treatment_type_col
                )
            else:
                is_vis = False
                pct = float("nan")
                cat = "unknown"

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
                    x_arr_vis, y_arr_vis, c=c_arr, cmap=cmap_name,
                    vmin=vmin, vmax=vmax, s=marker_size, alpha=alpha, edgecolors="none"
                )

                legend_handles.append(
                    mlines.Line2D([], [], marker="o", linestyle="none", markersize=9,
                                 markerfacecolor="none", markeredgecolor="blue",
                                 markeredgewidth=1.2, label="Area X visible (see colorbar)")
                )

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

                ax2.scatter(xs_arr, ys_arr, s=marker_size,
                           alpha=0.9 if cat not in {"unknown"} else 0.35,
                           color=color, edgecolors=edgecolors, linewidths=linewidths)

                legend_handles.append(
                    mlines.Line2D([], [], marker="o", linestyle="none", markersize=9,
                                 markerfacecolor=color, markeredgecolor=legend_edgecolor,
                                 markeredgewidth=legend_edgewidth, label=cat)
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

            line1, line2 = _build_subtitle_lines(
                pre_group=pre_group, post_group=post_group, rank_on=rank_on,
                min_n_phrases=min_n_phrases, log_scale=log_scale, include_visibility_note=True
            )
            if line2:
                ax2.set_title(f"{title_core}\n({line1})\n({line2})")
            else:
                ax2.set_title(f"{title_core}\n({line1})")

            ax2.set_xlabel(f"{pre_group} variance (ms$^2$)")
            ax2.set_ylabel(f"{post_group} variance (ms$^2$)")
            _pretty_axes(ax2)

            if sc is not None:
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="3.5%", pad=0.10)
                cbar = fig2.colorbar(sc, cax=cax)
                cbar.set_label("% of Area X lesioned", fontsize=11)
                x_anchor = 1.18
            else:
                x_anchor = 1.05

            legend_handles.append(mlines.Line2D([], [], color="red", linestyle="--", label="y=x"))

            legend2 = None
            if legend_handles:
                legend2 = fig2.legend(
                    handles=legend_handles, loc="upper left", bbox_to_anchor=(x_anchor, 0.80),
                    borderaxespad=0.0, frameon=True, facecolor="white", framealpha=0.85, fontsize=10
                )

            if annotate_identity_stats and stats_text_block and legend2 is not None:
                _place_stats_under_legend_figcoords(fig2, legend2, stats_text_block, pad_fig_y=0.01, fontsize=10.0)

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
                print("[INFO] Continuous lesion-% plot skipped: no points available.")

    # -------------------------------------------------------------------------
    # Plot 3: Ratio (post/pre) vs lesion PERCENT (lesion_pct)
    #   Includes SHAM even if lesion_pct missing by filling (linear: 0.0; logx: eps)
    # -------------------------------------------------------------------------
    fig_path_ratio: Optional[Path] = None

    if (metadata_excel is not None) and make_ratio_vs_volume_plots:
        pct_raw = merged["lesion_pct"].to_numpy(dtype=float)
        pre_v = merged["pre_variance"].to_numpy(dtype=float)
        post_v = merged["post_variance"].to_numpy(dtype=float)
        tg_all = merged["treatment_group"].astype(str).str.lower().to_numpy()

        pct = pct_raw.copy()
        sham_missing = (tg_all == "sham") & (~np.isfinite(pct))
        if np.any(sham_missing):
            if volume_logx:
                pos = pct[np.isfinite(pct) & (pct > 0)]
                eps = (float(np.nanmin(pos)) / 10.0) if pos.size else 1e-6
                pct[sham_missing] = eps
            else:
                pct[sham_missing] = 0.0

        if verbose:
            def _count(mask): return int(np.sum(mask))
            print("[INFO] Ratio-vs-% plot points by treatment_group (outlier-selected):",
                  {"sham": _count(tg_all == "sham"), "nma": _count(tg_all == "nma"), "other": _count(tg_all == "other")})
            print("[INFO] Ratio-vs-% sham missing lesion_pct (filled):", int(np.sum(sham_missing)))
            print("[INFO] Ratio-vs-% finite lesion_pct points:", int(np.sum(np.isfinite(pct))))

        good = np.isfinite(pct) & np.isfinite(pre_v) & np.isfinite(post_v) & (pre_v > 0) & (post_v > 0)
        x_pct = pct[good]
        ratio = (post_v[good] / pre_v[good]).astype(float)
        tg = tg_all[good]

        if x_pct.size > 0:
            fig3, ax3 = plt.subplots(figsize=(9.6, 6.4))

            color_map = {"sham": "red", "nma": "blue", "other": "gray"}
            plot_order = [("other", 0.25, 1), ("nma", 0.75, 2), ("sham", 0.90, 3)]

            for group_name, a, z in plot_order:
                m = (tg == group_name)
                if not np.any(m):
                    continue
                ax3.scatter(
                    x_pct[m], ratio[m], s=marker_size, alpha=a,
                    color=color_map.get(group_name, "gray"),
                    edgecolors="none", label=group_name, zorder=z
                )

            ax3.axhline(1.0, linestyle="--", color="red", linewidth=1.5, label="ratio=1 (y=x)")

            if volume_logx:
                ax3.set_xscale("log")
            if ratio_logy:
                ax3.set_yscale("log")

            ax3.set_xlabel("% of Area X lesioned")
            ax3.set_ylabel(f"{post_group}/{pre_group} variance ratio")
            ax3.set_title(f"Variance ratio vs % Area X lesioned\n(pre={pre_group}, post={post_group}, {tp_tag}_{rank_on})")
            _pretty_axes(ax3)
            ax3.legend(loc="upper right", frameon=True, framealpha=0.9)

            fig3.tight_layout()
            fig_path_ratio = out_dir / (
                f"variance_ratio_vs_lesion_pct__pre_{pre_group.replace(' ','_')}"
                f"__post_{post_group.replace(' ','_')}__{tp_tag}_{rank_on}.png"
            )
            fig3.savefig(fig_path_ratio, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig3)
        else:
            if verbose:
                print("[INFO] Ratio-vs-% plot skipped: no valid points after filtering.")

    return {
        "table": merged,
        "table_path": str(table_path),
        "table_all_pairs": merged_all_pairs,
        "table_all_pairs_path": str(table_all_path),
        "per_bird_tail_metrics": per_bird_tail_df,
        "tail_results": tail_results,
        "figure_path_discrete": str(fig_path_discrete),
        "figure_path_continuous": (str(fig_path_cont) if fig_path_cont is not None else None),
        "figure_path_ratio_vs_lesion_pct": (str(fig_path_ratio) if fig_path_ratio is not None else None),
        "identity_stats": identity_stats,
        "permtest_results": permtest_results,
        "n_points": int(len(merged)),
        "n_points_all_pairs": int(len(merged_all_pairs)),
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
    p.add_argument("--no_identity_annot", action="store_true", help="Do not annotate stats on plots.")
    p.add_argument("--nma_token", default="nma", type=str, help="Token used to identify NMA treatments in metadata strings.")

    # log-ratio permutation tests
    p.add_argument("--no_perm_test", action="store_true", help="Disable permutation test on outlier-selected log-ratios.")
    p.add_argument("--perm_n", default=10000, type=int, help="Number of permutations (log-ratio test).")
    p.add_argument("--perm_seed", default=0, type=int, help="Permutation RNG seed (log-ratio test).")
    p.add_argument("--no_perm_annot", action="store_true", help="Do not include permutation test block on plots.")

    # NEW: tail/outlier metrics
    p.add_argument("--no_tail_metrics", action="store_true", help="Disable per-bird tail/outlier metrics (ALL pairs).")
    p.add_argument("--tail_min_syll", default=5, type=int, help="Min syllables per bird for tail metrics.")
    p.add_argument("--tail_rr_cc", default=1e-6, type=float, help="Continuity correction for RR.")
    p.add_argument("--tail_perm_n", default=20000, type=int, help="Permutations for tail metric group tests.")
    p.add_argument("--tail_perm_seed", default=1, type=int, help="Seed for tail metric group tests.")
    p.add_argument("--no_tail_annot", action="store_true", help="Do not include tail/outlier test blocks on plots.")

    # ratio vs lesion percent
    p.add_argument("--no_ratio_vs_pct", action="store_true", help="Disable ratio-vs-lesion-% plot.")
    p.add_argument("--ratio_logy", action="store_true", help="Log-scale y-axis for ratio plot.")
    p.add_argument("--pct_logx", action="store_true", help="Log-scale x-axis for % plot (usually leave off).")

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
        do_perm_test=(not args.no_perm_test),
        perm_n_permutations=args.perm_n,
        perm_seed=args.perm_seed,
        annotate_perm_test=(not args.no_perm_annot),
        do_tail_metrics=(not args.no_tail_metrics),
        tail_min_syllables_per_bird=args.tail_min_syll,
        tail_rr_cc=args.tail_rr_cc,
        tail_perm_n=args.tail_perm_n,
        tail_perm_seed=args.tail_perm_seed,
        annotate_tail_metrics=(not args.no_tail_annot),
        make_ratio_vs_volume_plots=(not args.no_ratio_vs_pct),
        ratio_logy=args.ratio_logy,
        volume_logx=args.pct_logx,
    )

    print("Discrete plot:", res["figure_path_discrete"])
    print("Continuous plot:", res["figure_path_continuous"])
    print("Ratio vs lesion % plot:", res["figure_path_ratio_vs_lesion_pct"])
    print("Outlier-selected table:", res["table_path"])
    print("ALL-pairs table:", res["table_all_pairs_path"])
    if res.get("permtest_results"):
        print("Log-ratio permutation test results:", res["permtest_results"])
    if res.get("tail_results"):
        print("Tail/outlier metrics results:", res["tail_results"])


"""
SPYDER CONSOLE EXAMPLE USAGE
---------------------------

from pathlib import Path
import sys, importlib

# Folder that contains outlier_graphs.py (adjust if needed)
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import outlier_graphs as og
importlib.reload(og)

# Inputs
compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")
histology_dir = Path("/Volumes/my_own_SSD/histology_files/lesion_quantification_csvs_jsons")  # contains *_final_volumes.json

# Output directory
out_dir = compiled_csv.parent / "outlier_variance_scatter"
out_dir.mkdir(parents=True, exist_ok=True)

res = og.plot_pre_post_variance_scatter(
    csv_path=compiled_csv,
    out_dir=out_dir,
    pre_group="Late Pre",
    post_group="Post",
    variance_col="Variance_ms2",

    # OUTLIER SELECTION (for plotting + log-ratio permutation test)
    top_percentile=70,        # keeps top 30% highest-variance syllables per bird
    rank_on="post",

    # filters
    min_n_phrases=10,

    # metadata + histology
    metadata_excel=metadata_xlsx,
    meta_sheet_name=0,
    histology_volumes_dir=histology_dir,
    lesion_pct_mode="avg",

    # plots
    make_continuous_lesion_pct_plot=True,
    log_scale=True,

    # identity stats + log-ratio tests (outlier-selected)
    compute_identity_stats=True,
    annotate_identity_stats=True,
    do_perm_test=True,
    perm_n_permutations=10000,
    perm_seed=0,

    # NEW: tail/outlier metrics using ALL pairs (before top_percentile)
    do_tail_metrics=True,
    tail_min_syllables_per_bird=5,
    tail_perm_n=20000,
    tail_perm_seed=1,
    annotate_tail_metrics=True,

    # ratio vs % lesioned
    make_ratio_vs_volume_plots=True,
    ratio_logy=False,
    volume_logx=False,

    show=True,
    verbose=True,
)

print("Saved outlier-selected table:", res["table_path"])
print("Saved ALL-pairs table:", res["table_all_pairs_path"])
print("Discrete figure:", res["figure_path_discrete"])
print("Continuous figure:", res["figure_path_continuous"])
print("Ratio vs lesion % figure:", res["figure_path_ratio_vs_lesion_pct"])
print("Per-bird tail metrics df shape:", None if res["per_bird_tail_metrics"] is None else res["per_bird_tail_metrics"].shape)
"""
