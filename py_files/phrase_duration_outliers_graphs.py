# -*- coding: utf-8 -*-
"""
phrase_duration_outliers_graphs.py

Visualize differences between outliers for multiple tail metrics (logvar, cv),
in unweighted and weighted_by_N_phrases modes.

This module expects an INPUT DIRECTORY that contains (some or all) of these CSVs
(from your outlier/tail-metrics pipeline):

Required for most plots:
  - per_bird_group_tail_metrics_logvar.csv
  - per_bird_group_tail_metrics_cv.csv

Optional (enables extra plots if present):
  - logvar_long_flagged_unweighted.csv
  - logvar_long_flagged_weighted.csv
  - cv_long_flagged_unweighted.csv
  - cv_long_flagged_weighted.csv
  - per_bird_prepost_tail_summary.csv
  - per_bird_prepost_distribution_tests.csv
  - pooled_quantile_regression_coeffs.csv
  - per_bird_evt_gpd_summary.csv

Outputs:
  By default, saves figures into:
      <input_dir>/outlier_figures/

Public function:
  make_outlier_comparison_figures(input_dir, output_dir=None, ...)

CLI usage:
  python phrase_duration_outliers_graphs.py --input_dir /path/to/outlier_csvs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = [
    "make_outlier_comparison_figures",
]


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────
def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read CSV: {path.name} ({type(e).__name__}: {e})")
        return None


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _norm_group_labels(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("-", " ", regex=False)
    )


def _get_treat_color_map(treat_classes: Sequence[str]) -> Dict[str, int]:
    # Map each treat_class to a color-cycle index (matplotlib default colors)
    uniq = [x for x in pd.unique(pd.Series(list(treat_classes)).dropna())]
    uniq = [str(x) for x in uniq]
    uniq_sorted = sorted(uniq)
    return {tc: i for i, tc in enumerate(uniq_sorted)}


def _apply_treat_class_mapping(
    df: pd.DataFrame,
    treat_map: Dict[str, str],
    id_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["treat_class"] = out[id_col].map(treat_map).fillna("unknown")
    return out


def _default_metrics_to_compare() -> List[str]:
    # These should exist in per_bird_group_tail_metrics_*.csv
    return [
        "frac_high_outliers_1p5",
        "frac_high_outliers_3",
        "frac_modz_abs_gt_3p5",
        "tail_heaviness_95_75_over_75_50",
        "upper_tail_spread_95_50",
    ]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ──────────────────────────────────────────────────────────────────────────────
# Core aggregation: Post minus pooled-Pre deltas per bird
# ──────────────────────────────────────────────────────────────────────────────
def _pooled_pre_from_group_metrics(
    gm: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    weighting_col: str,
    measure_name: str,
    metric_col: str,
    pre_groups: Sequence[str],
    post_group: str,
) -> pd.DataFrame:
    """
    Compute pooled pre metric (Early+Late combined) and Post metric, plus delta = Post - Pre.

    Note: This *approximates* pooled Pre by a weighted average across Early/Late:
      - unweighted: weights = n_rows
      - weighted_by_N_phrases: weights = weight_sum

    Returns columns:
      [id_col, weighting, measure, pre_value, post_value, delta, pre_weight, post_weight]
    """
    required = {id_col, group_col, weighting_col, metric_col, "n_rows", "weight_sum"}
    missing = [c for c in required if c not in gm.columns]
    if missing:
        raise KeyError(f"Group metrics missing required columns: {missing}")

    df = gm.copy()
    df[group_col] = _norm_group_labels(df[group_col])
    df[weighting_col] = df[weighting_col].astype(str).str.strip()

    df[metric_col] = _safe_numeric(df[metric_col])
    df["n_rows"] = _safe_numeric(df["n_rows"])
    df["weight_sum"] = _safe_numeric(df["weight_sum"])

    rows = []
    for (aid, weighting), sub in df.groupby([id_col, weighting_col], sort=False):
        pre = sub[sub[group_col].isin(list(pre_groups))].copy()
        post = sub[sub[group_col] == post_group].copy()

        if pre.empty or post.empty:
            continue

        if str(weighting) == "weighted_by_N_phrases":
            wcol = "weight_sum"
        else:
            # unweighted
            wcol = "n_rows"

        w_pre = _safe_numeric(pre[wcol]).fillna(0.0).values
        x_pre = _safe_numeric(pre[metric_col]).values

        # Weighted average across Early/Late groups
        m = np.isfinite(x_pre) & np.isfinite(w_pre) & (w_pre > 0)
        if np.any(m):
            pre_value = float(np.sum(x_pre[m] * w_pre[m]) / np.sum(w_pre[m]))
            pre_weight = float(np.sum(w_pre[m]))
        else:
            # fallback: simple mean of whatever exists
            pre_value = float(np.nanmean(x_pre)) if np.isfinite(x_pre).any() else np.nan
            pre_weight = float(np.nansum(w_pre)) if np.isfinite(w_pre).any() else np.nan

        # Post value: if multiple rows exist, take weighted average (same logic)
        w_post = _safe_numeric(post[wcol]).fillna(0.0).values
        x_post = _safe_numeric(post[metric_col]).values
        m2 = np.isfinite(x_post) & np.isfinite(w_post) & (w_post > 0)
        if np.any(m2):
            post_value = float(np.sum(x_post[m2] * w_post[m2]) / np.sum(w_post[m2]))
            post_weight = float(np.sum(w_post[m2]))
        else:
            post_value = float(np.nanmean(x_post)) if np.isfinite(x_post).any() else np.nan
            post_weight = float(np.nansum(w_post)) if np.isfinite(w_post).any() else np.nan

        delta = post_value - pre_value if (np.isfinite(post_value) and np.isfinite(pre_value)) else np.nan

        rows.append({
            id_col: aid,
            "weighting": str(weighting),
            "measure": measure_name,
            "metric": metric_col,
            "pre_value": pre_value,
            "post_value": post_value,
            "delta_post_minus_pre": delta,
            "pre_weight": pre_weight,
            "post_weight": post_weight,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Long flagged tables aggregation (optional)
# ──────────────────────────────────────────────────────────────────────────────
def _summarize_long_flags(
    long_df: pd.DataFrame,
    *,
    id_col: str,
    group_col: str,
    weighting_name: str,
    prefix: str,  # "logvar" or "cv"
) -> pd.DataFrame:
    """
    From *_long_flagged_*.csv, compute per-(Animal, Group) fractions and counts.

    Outputs columns:
      [id_col, Group, weighting, measure, n_rows,
       frac_high_outlier_1p5, frac_high_outlier_3, frac_modz_abs_gt_3p5,
       n_high_outlier_1p5, n_high_outlier_3, n_modz_abs_gt_3p5]
    """
    df = long_df.copy()
    if id_col not in df.columns:
        # try a common fallback
        if "animal_id" in df.columns:
            df = df.rename(columns={"animal_id": id_col})
        else:
            raise KeyError(f"Long flagged table is missing id_col '{id_col}'")

    if group_col not in df.columns:
        raise KeyError(f"Long flagged table is missing group_col '{group_col}'")

    df[group_col] = _norm_group_labels(df[group_col])

    c_hi_1p5 = f"{prefix}_is_high_outlier_1p5"
    c_hi_3 = f"{prefix}_is_high_outlier_3"
    c_modz = f"{prefix}_modz_abs_gt_3p5"
    for c in [c_hi_1p5, c_hi_3, c_modz]:
        if c not in df.columns:
            raise KeyError(f"Long flagged table missing expected flag column: '{c}'")

    # booleans
    df[c_hi_1p5] = df[c_hi_1p5].astype(bool)
    df[c_hi_3] = df[c_hi_3].astype(bool)
    df[c_modz] = df[c_modz].astype(bool)

    out_rows = []
    for (aid, grp), sub in df.groupby([id_col, group_col], sort=False):
        n = int(sub.shape[0])
        if n <= 0:
            continue
        out_rows.append({
            id_col: aid,
            group_col: grp,
            "weighting": weighting_name,
            "measure": prefix,
            "n_rows": float(n),
            "frac_high_outlier_1p5": float(sub[c_hi_1p5].mean()),
            "frac_high_outlier_3": float(sub[c_hi_3].mean()),
            "frac_modz_abs_gt_3p5": float(sub[c_modz].mean()),
            "n_high_outlier_1p5": float(sub[c_hi_1p5].sum()),
            "n_high_outlier_3": float(sub[c_hi_3].sum()),
            "n_modz_abs_gt_3p5": float(sub[c_modz].sum()),
        })
    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save_or_show(fig: plt.Figure, out_path: Path, *, show: bool, save: bool, dpi: int = 200) -> None:
    if save:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_group_boxplots(
    gm: pd.DataFrame,
    *,
    out_path: Path,
    id_col: str,
    group_col: str,
    measure: str,
    weighting: str,
    metric: str,
    show: bool,
    save: bool,
    dpi: int,
) -> None:
    """
    Boxplot across birds for each Group, for one metric.
    """
    df = gm.copy()
    df[group_col] = _norm_group_labels(df[group_col])
    df = df[(df["measure"] == measure) & (df["weighting"] == weighting)].copy()

    if df.empty or metric not in df.columns:
        return

    groups = ["Early Pre", "Late Pre", "Post"]
    data = []
    labels = []
    for g in groups:
        vals = _safe_numeric(df.loc[df[group_col] == g, metric]).values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(g)

    if not data:
        return

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(f"{metric} across birds ({measure}, {weighting})")
    ax.set_ylabel(metric)
    ax.set_xlabel("Group")
    ax.grid(True, alpha=0.25)

    _save_or_show(fig, out_path, show=show, save=save, dpi=dpi)


def _plot_delta_scatter_by_weighting(
    deltas: pd.DataFrame,
    *,
    out_path: Path,
    id_col: str,
    metric: str,
    treat_map: Optional[Dict[str, str]],
    show: bool,
    save: bool,
    dpi: int,
) -> None:
    """
    Scatter of delta_post_minus_pre for logvar vs cv, grouped by weighting.
    Produces a single figure with two x-categories: unweighted, weighted_by_N_phrases.
    """
    df = deltas[deltas["metric"] == metric].copy()
    if df.empty:
        return

    if treat_map is not None:
        df = _apply_treat_class_mapping(df, treat_map, id_col=id_col)
    else:
        df["treat_class"] = "unknown"

    weightings = ["unweighted", "weighted_by_N_phrases"]
    measures = ["logvar", "cv"]
    markers = {"logvar": "o", "cv": "s"}

    # build color mapping from treat_class
    color_map = _get_treat_color_map(df["treat_class"].astype(str).tolist())
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)

    # x positions
    x_base = {w: i for i, w in enumerate(weightings)}
    # small offsets for measures
    x_off = {"logvar": -0.12, "cv": +0.12}

    for meas in measures:
        dsub = df[df["measure"] == meas]
        if dsub.empty:
            continue

        for tc, sub_tc in dsub.groupby("treat_class", sort=False):
            x = []
            y = []
            for _, r in sub_tc.iterrows():
                w = str(r["weighting"])
                if w not in x_base:
                    continue
                xv = x_base[w] + x_off[meas] + np.random.uniform(-0.04, 0.04)
                x.append(xv)
                y.append(float(r["delta_post_minus_pre"]))
            if not x:
                continue

            # pick a default-cycle color index
            c = None
            if cycle is not None:
                c = cycle[color_map.get(str(tc), 0) % len(cycle)]

            ax.scatter(
                x, y,
                marker=markers.get(meas, "o"),
                label=f"{meas} | {tc}",
                alpha=0.85,
                c=c,
                edgecolors="none",
            )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks([x_base[w] for w in weightings])
    ax.set_xticklabels(weightings, rotation=0)
    ax.set_ylabel("Post − pooled Pre")
    ax.set_title(f"Delta(Post − Pre): {metric} (logvar vs cv; colored by treat_class if available)")
    ax.grid(True, alpha=0.25)

    # deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        new_h.append(h)
        new_l.append(l)
    if new_h:
        ax.legend(new_h, new_l, fontsize=9, frameon=True, loc="best")

    _save_or_show(fig, out_path, show=show, save=save, dpi=dpi)


def _plot_logvar_vs_cv_delta_scatter(
    deltas: pd.DataFrame,
    *,
    out_path: Path,
    id_col: str,
    metric: str,
    weighting: str,
    treat_map: Optional[Dict[str, str]],
    show: bool,
    save: bool,
    dpi: int,
) -> None:
    """
    For a given weighting + metric, scatter:
      x = delta(logvar)
      y = delta(cv)
    per bird.
    """
    df = deltas[deltas["metric"] == metric].copy()
    df = df[df["weighting"] == weighting].copy()
    if df.empty:
        return

    # pivot to wide: rows=bird, cols=measure
    wide = df.pivot_table(index=id_col, columns="measure", values="delta_post_minus_pre", aggfunc="mean")
    if wide.empty:
        return

    x = wide.get("logvar", pd.Series(index=wide.index, dtype=float)).astype(float)
    y = wide.get("cv", pd.Series(index=wide.index, dtype=float)).astype(float)

    m = np.isfinite(x.values) & np.isfinite(y.values)
    x = x[m]
    y = y[m]
    birds = x.index.astype(str).tolist()
    if len(birds) == 0:
        return

    # colors by treat_class if possible
    if treat_map is not None:
        tc = [treat_map.get(b, "unknown") for b in birds]
    else:
        tc = ["unknown"] * len(birds)

    color_map = _get_treat_color_map(tc)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    fig = plt.figure(figsize=(6.8, 6.0))
    ax = fig.add_subplot(111)

    for tclass in sorted(set(tc)):
        idx = [i for i, t in enumerate(tc) if t == tclass]
        if not idx:
            continue
        xs = [float(x.iloc[i]) for i in idx]
        ys = [float(y.iloc[i]) for i in idx]
        c = None
        if cycle is not None:
            c = cycle[color_map.get(str(tclass), 0) % len(cycle)]
        ax.scatter(xs, ys, label=str(tclass), alpha=0.85, c=c, edgecolors="none")

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(f"Δ Post−Pre (logvar) for {metric}")
    ax.set_ylabel(f"Δ Post−Pre (cv) for {metric}")
    ax.set_title(f"logvar vs cv deltas ({metric}, {weighting})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, frameon=True, loc="best")

    _save_or_show(fig, out_path, show=show, save=save, dpi=dpi)


def _plot_prepost_risk_diff(
    prepost: pd.DataFrame,
    *,
    out_path: Path,
    id_col: str,
    weighting: str,
    treat_class_col: str = "treat_class",
    show: bool,
    save: bool,
    dpi: int,
) -> None:
    """
    Scatter of per-bird risk_diff for logvar vs cv using per_bird_prepost_tail_summary.csv.
    """
    if prepost is None or prepost.empty:
        return
    if id_col not in prepost.columns or "weighting" not in prepost.columns:
        return

    df = prepost.copy()
    df = df[df["weighting"].astype(str) == str(weighting)].copy()
    if df.empty:
        return

    # these should exist per your spec
    if "logvar_risk_diff" not in df.columns or "cv_risk_diff" not in df.columns:
        return

    df["logvar_risk_diff"] = _safe_numeric(df["logvar_risk_diff"])
    df["cv_risk_diff"] = _safe_numeric(df["cv_risk_diff"])

    m = np.isfinite(df["logvar_risk_diff"].values) & np.isfinite(df["cv_risk_diff"].values)
    df = df.loc[m].copy()
    if df.empty:
        return

    if treat_class_col not in df.columns:
        df[treat_class_col] = "unknown"

    color_map = _get_treat_color_map(df[treat_class_col].astype(str).tolist())
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    fig = plt.figure(figsize=(6.8, 6.0))
    ax = fig.add_subplot(111)

    for tc, sub in df.groupby(treat_class_col, sort=False):
        xs = sub["logvar_risk_diff"].astype(float).values
        ys = sub["cv_risk_diff"].astype(float).values
        c = None
        if cycle is not None:
            c = cycle[color_map.get(str(tc), 0) % len(cycle)]
        ax.scatter(xs, ys, label=str(tc), alpha=0.85, c=c, edgecolors="none")

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("risk_diff (logvar): p_post - p_pre at t=Pre Q95")
    ax.set_ylabel("risk_diff (cv): p_post - p_pre at t=Pre Q95")
    ax.set_title(f"Per-bird tail risk_diff (weighting={weighting})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, frameon=True, loc="best")

    _save_or_show(fig, out_path, show=show, save=save, dpi=dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Public: main figure generator
# ──────────────────────────────────────────────────────────────────────────────
def make_outlier_comparison_figures(
    input_dir: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    weighting_col: str = "weighting",
    pre_groups: Sequence[str] = ("Early Pre", "Late Pre"),
    post_group: str = "Post",
    metrics: Optional[Sequence[str]] = None,
    show_plots: bool = True,
    save_plots: bool = True,
    dpi: int = 200,
) -> Dict[str, Path]:
    """
    Reads outlier/tail CSVs from `input_dir` and saves comparison figures.

    Returns:
      dict mapping figure_key -> file path
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "outlier_figures"
    outdir = _ensure_dir(Path(output_dir))

    if metrics is None:
        metrics = _default_metrics_to_compare()

    # Load group metrics (required for most plots)
    gm_logvar_path = input_dir / "per_bird_group_tail_metrics_logvar.csv"
    gm_cv_path = input_dir / "per_bird_group_tail_metrics_cv.csv"

    gm_logvar = _read_csv_if_exists(gm_logvar_path)
    gm_cv = _read_csv_if_exists(gm_cv_path)

    if gm_logvar is None and gm_cv is None:
        raise FileNotFoundError(
            f"Could not find group tail metrics CSVs in: {input_dir}\n"
            f"Expected at least one of:\n"
            f"  - {gm_logvar_path.name}\n"
            f"  - {gm_cv_path.name}"
        )

    # Normalize + attach measure label
    gm_frames = []
    if gm_logvar is not None:
        g = gm_logvar.copy()
        g[group_col] = _norm_group_labels(g[group_col]) if group_col in g.columns else g.get(group_col, "")
        g["measure"] = "logvar"
        gm_frames.append(g)
    if gm_cv is not None:
        g = gm_cv.copy()
        g[group_col] = _norm_group_labels(g[group_col]) if group_col in g.columns else g.get(group_col, "")
        g["measure"] = "cv"
        gm_frames.append(g)

    gm_all = pd.concat(gm_frames, ignore_index=True)
    # basic normalization
    if id_col not in gm_all.columns:
        # allow a few common alternatives
        for alt in ["animal_id", "AnimalID", "bird", "Bird"]:
            if alt in gm_all.columns:
                gm_all = gm_all.rename(columns={alt: id_col})
                break
    if id_col not in gm_all.columns:
        raise KeyError(f"Group metrics tables must include '{id_col}' column (or a recognizable alternative).")

    gm_all[id_col] = gm_all[id_col].astype(str).str.strip()
    gm_all[group_col] = _norm_group_labels(gm_all[group_col]) if group_col in gm_all.columns else ""

    if weighting_col not in gm_all.columns:
        # If missing, assume unweighted (older runs)
        gm_all[weighting_col] = "unweighted"
    gm_all[weighting_col] = gm_all[weighting_col].astype(str).str.strip()

    # Optional: treat_class mapping from prepost tail summary
    prepost_path = input_dir / "per_bird_prepost_tail_summary.csv"
    prepost = _read_csv_if_exists(prepost_path)
    treat_map: Optional[Dict[str, str]] = None
    if prepost is not None and not prepost.empty and id_col in prepost.columns:
        if "treat_class" in prepost.columns:
            tmp = prepost[[id_col, "treat_class"]].copy()
            tmp[id_col] = tmp[id_col].astype(str).str.strip()
            tmp["treat_class"] = tmp["treat_class"].astype(str).str.strip()
            treat_map = dict(tmp.dropna().groupby(id_col)["treat_class"].first())

    # Build a delta table for each metric across measures and weightings
    delta_frames = []
    for meas in ["logvar", "cv"]:
        sub = gm_all[gm_all["measure"] == meas].copy()
        if sub.empty:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                print(f"[WARN] Missing metric column '{metric}' in {meas} group metrics; skipping it.")
                continue
            d = _pooled_pre_from_group_metrics(
                sub,
                id_col=id_col,
                group_col=group_col,
                weighting_col=weighting_col,
                measure_name=meas,
                metric_col=metric,
                pre_groups=pre_groups,
                post_group=post_group,
            )
            delta_frames.append(d)

    deltas = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()

    fig_paths: Dict[str, Path] = {}

    # ── Plot A: For each metric, delta scatter by weighting with logvar vs cv ──
    if not deltas.empty:
        for metric in metrics:
            out_path = outdir / f"delta_post_minus_pre__{metric}.png"
            _plot_delta_scatter_by_weighting(
                deltas,
                out_path=out_path,
                id_col=id_col,
                metric=metric,
                treat_map=treat_map,
                show=show_plots,
                save=save_plots,
                dpi=dpi,
            )
            if out_path.is_file():
                fig_paths[f"delta_post_minus_pre__{metric}"] = out_path

        # ── Plot B: logvar vs cv delta scatter for a few key metrics ──
        for metric in ["frac_high_outliers_1p5", "frac_modz_abs_gt_3p5"]:
            if metric not in metrics:
                continue
            for weighting in ["unweighted", "weighted_by_N_phrases"]:
                out_path = outdir / f"logvar_vs_cv_delta__{metric}__{weighting}.png"
                _plot_logvar_vs_cv_delta_scatter(
                    deltas,
                    out_path=out_path,
                    id_col=id_col,
                    metric=metric,
                    weighting=weighting,
                    treat_map=treat_map,
                    show=show_plots,
                    save=save_plots,
                    dpi=dpi,
                )
                if out_path.is_file():
                    fig_paths[f"logvar_vs_cv_delta__{metric}__{weighting}"] = out_path

    # ── Plot C: group boxplots across birds (per measure/weighting/metric) ──
    for meas in ["logvar", "cv"]:
        for weighting in ["unweighted", "weighted_by_N_phrases"]:
            for metric in metrics:
                if metric not in gm_all.columns:
                    continue
                out_path = outdir / f"group_boxplot__{meas}__{weighting}__{metric}.png"
                _plot_group_boxplots(
                    gm_all,
                    out_path=out_path,
                    id_col=id_col,
                    group_col=group_col,
                    measure=meas,
                    weighting=weighting,
                    metric=metric,
                    show=show_plots,
                    save=save_plots,
                    dpi=dpi,
                )
                if out_path.is_file():
                    fig_paths[f"group_boxplot__{meas}__{weighting}__{metric}"] = out_path

    # ── Plot D (optional): per-bird risk_diff (tail summary file) ──
    if prepost is not None and not prepost.empty:
        for weighting in ["unweighted", "weighted_by_N_phrases"]:
            out_path = outdir / f"prepost_tail_risk_diff__{weighting}.png"
            _plot_prepost_risk_diff(
                prepost,
                out_path=out_path,
                id_col=id_col,
                weighting=weighting,
                show=show_plots,
                save=save_plots,
                dpi=dpi,
            )
            if out_path.is_file():
                fig_paths[f"prepost_tail_risk_diff__{weighting}"] = out_path

    # ── Optional: summarize long flagged tables if they exist (no extra plots by default) ──
    # (This is mostly useful if you want to extend plotting later.)
    long_files = [
        ("logvar_long_flagged_unweighted.csv", "unweighted", "logvar"),
        ("logvar_long_flagged_weighted.csv", "weighted_by_N_phrases", "logvar"),
        ("cv_long_flagged_unweighted.csv", "unweighted", "cv"),
        ("cv_long_flagged_weighted.csv", "weighted_by_N_phrases", "cv"),
    ]
    long_summaries = []
    for fname, weighting, prefix in long_files:
        df_long = _read_csv_if_exists(input_dir / fname)
        if df_long is None or df_long.empty:
            continue
        try:
            s = _summarize_long_flags(
                df_long,
                id_col=id_col,
                group_col=group_col,
                weighting_name=weighting,
                prefix=prefix,
            )
            long_summaries.append(s)
        except Exception as e:
            print(f"[WARN] Could not summarize {fname}: {type(e).__name__}: {e}")

    if long_summaries:
        long_summary_df = pd.concat(long_summaries, ignore_index=True)
        out_path = outdir / "long_flagged__per_bird_group_summary.csv"
        long_summary_df.to_csv(out_path, index=False)
        fig_paths["long_flagged__per_bird_group_summary_csv"] = out_path

    print(f"[DONE] Saved {len(fig_paths)} outputs in: {outdir}")
    return fig_paths


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Make outlier comparison figures from tail/outlier CSVs.")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing the outlier CSVs.")
    ap.add_argument("--output_dir", type=str, default=None, help="Directory to save figures (default: input_dir/outlier_figures).")
    ap.add_argument("--id_col", type=str, default="Animal ID", help="Animal ID column name.")
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    ap.add_argument("--no_save", action="store_true", help="Do not save plots.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_outlier_comparison_figures(
        args.input_dir,
        output_dir=args.output_dir,
        id_col=args.id_col,
        show_plots=bool(args.show),
        save_plots=not bool(args.no_save),
        dpi=int(args.dpi),
    )
    
"""
from pathlib import Path
import importlib

# 1) Point this to the folder that contains phrase_duration_outliers_graphs.py
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
import sys
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import phrase_duration_outliers_graphs as pog
importlib.reload(pog)

# 2) Point this to the directory that contains the outlier CSVs:
#    per_bird_group_tail_metrics_logvar.csv
#    per_bird_group_tail_metrics_cv.csv
#    (and optionally: per_bird_prepost_tail_summary.csv, *_long_flagged_*.csv, etc.)
input_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")

# 3) Choose where to save the figures (optional)
#    If you leave this as None, it saves to: input_dir / "outlier_figures"
output_dir = input_dir / "outlier_figures"

# 4) Run figure generation
fig_paths = pog.make_outlier_comparison_figures(
    input_dir=input_dir,
    output_dir=output_dir,
    id_col="Animal ID",
    group_col="Group",
    weighting_col="weighting",
    pre_groups=("Early Pre", "Late Pre"),
    post_group="Post",
    metrics=[
        "frac_high_outliers_1p5",
        "frac_high_outliers_3",
        "frac_modz_abs_gt_3p5",
        "tail_heaviness_95_75_over_75_50",
        "upper_tail_spread_95_50",
    ],
    show_plots=True,    # set False if you only want saved files
    save_plots=False,
    dpi=200,
)

print("Saved outputs:")
for k, p in fig_paths.items():
    print(" ", k, "->", p)


"""
