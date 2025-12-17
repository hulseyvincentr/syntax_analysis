#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phrase_duration_stats.py

Make Pre vs Post boxplots for log-variance of phrase durations and run
stats separately for:

  1) Bilateral NMA lesion injections (experimental)
  2) Bilateral saline sham injections (control)

For each bird we use:

    Pre_logvar  = mean log10(variance) in Late Pre
    Post_logvar = mean log10(variance) in Post

and also:

    Pre_cv  = mean coefficient of variation (σ/μ) in Late Pre
    Post_cv = mean coefficient of variation (σ/μ) in Post

We then compute per-bird Post-Pre differences and:

  - Paired t-tests within each group (for both log-variance and CV boxplots)
  - Mann–Whitney U on |Post-Pre| (log-variance) comparing NMA vs sham
  - Levene’s test on the variance of Δ log-variance between groups
  - Between-groups tests on signed Δ log-variance (NMA vs sham):
        * Welch two-sample t-test
        * Mann–Whitney U (two-sided)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Optional SciPy for stats
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------
def _load_phrase_stats(
    compiled_stats_path: Union[str, Path],
    compiled_format: Optional[str] = None,
) -> pd.DataFrame:
    """Load compiled phrase-duration stats into a DataFrame."""
    path = Path(compiled_stats_path)

    if compiled_format is None:
        suf = path.suffix.lower()
        if suf == ".csv":
            compiled_format = "csv"
        elif suf == ".json":
            compiled_format = "json"
        elif suf == ".npz":
            compiled_format = "npz"
        else:
            raise ValueError(
                f"Cannot infer compiled_format from suffix {path.suffix!r}."
            )

    if compiled_format == "csv":
        df = pd.read_csv(path)
    elif compiled_format == "json":
        df = pd.read_json(path)
    elif compiled_format == "npz":
        arr = np.load(path, allow_pickle=True)
        if "phrase_stats" not in arr:
            raise KeyError("NPZ must contain an array named 'phrase_stats'.")
        df = pd.DataFrame(arr["phrase_stats"])
    else:
        raise ValueError(f"Unsupported compiled_format={compiled_format!r}")

    return df


def _load_metadata(
    metadata_excel_path: Union[str, Path],
    sheet_name: Union[int, str] = "metadata",
    id_col: str = "Animal ID",
) -> pd.DataFrame:
    """
    Load metadata directly from Excel.

    Expected columns (at minimum):
        'Animal ID', 'Treatment type'

    Returns a DataFrame with unique Animal IDs.
    """
    metadata_excel_path = Path(metadata_excel_path)
    meta_df = pd.read_excel(metadata_excel_path, sheet_name=sheet_name)

    if id_col not in meta_df.columns:
        raise KeyError(f"Column {id_col!r} not found in metadata Excel.")
    if "Treatment type" not in meta_df.columns:
        raise KeyError("Expected 'Treatment type' column in metadata Excel.")

    # Keep only what we need and ensure uniqueness on Animal ID
    meta_df = meta_df[[id_col, "Treatment type"]].drop_duplicates(subset=id_col)

    return meta_df


# ---------------------------------------------------------------------
# Core computation: per-bird log-variance for Late Pre and Post
# ---------------------------------------------------------------------
def _compute_per_bird_log_variance(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    var_col: str = "Variance_ms2",
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    log_mode: str = "log10",
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by Animal ID with columns:

        'Pre_logvar'   (Late Pre)
        'Post_logvar'  (Post)

    Values are the mean log-variance across syllables for each bird/group.
    """
    if var_col not in phrase_df.columns:
        raise KeyError(f"{var_col!r} column not found in phrase stats DataFrame.")
    if group_col not in phrase_df.columns or id_col not in phrase_df.columns:
        raise KeyError(
            f"Missing required columns for grouping ({id_col!r}, {group_col!r})."
        )

    df = phrase_df.copy()

    # Keep only the groups we care about
    df = df[df[group_col].isin([pre_group, post_group])].copy()
    df = df.dropna(subset=[var_col])

    # Log-transform variance
    if log_mode.lower() == "log10":
        df["log_variance"] = np.log10(df[var_col].astype(float).clip(lower=1e-12))
        log_label = "log₁₀ variance (ms²)"
    elif log_mode.lower() in {"ln", "log"}:
        df["log_variance"] = np.log(df[var_col].astype(float).clip(lower=1e-12))
        log_label = "ln variance (ms²)"
    else:
        raise ValueError(f"Unsupported log_mode={log_mode!r}; use 'log10' or 'ln'.")

    # Average log-variance across syllables for each bird and group
    grp = (
        df.groupby([id_col, group_col])["log_variance"]
        .mean()
        .rename("log_variance_mean")
    )

    wide = grp.unstack(group_col)

    # Keep only birds that have both Pre and Post values
    if pre_group not in wide.columns or post_group not in wide.columns:
        raise RuntimeError(
            f"Expected groups {pre_group!r} and {post_group!r} "
            f"in column {group_col!r}."
        )

    wide = wide[[pre_group, post_group]].dropna(how="any")

    wide = wide.rename(columns={pre_group: "Pre_logvar", post_group: "Post_logvar"})
    wide.index.name = id_col
    wide.attrs["log_label"] = log_label  # stash y-axis label

    return wide


# ---------------------------------------------------------------------
# Core computation: per-bird coefficient of variation (CV)
# ---------------------------------------------------------------------
def _compute_per_bird_cv(
    phrase_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    pre_group: str = "Late Pre",
    post_group: str = "Post",
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by Animal ID with columns:

        'Pre_cv'   (Late Pre)
        'Post_cv'  (Post)

    Where CV per syllable = sqrt(variance_ms2) / mean_ms,
    and we average CV across syllables for each bird/group.
    """
    missing_cols = [c for c in [mean_col, var_col, group_col, id_col] if c not in phrase_df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns for CV computation: {missing_cols}")

    df = phrase_df.copy()
    df = df[df[group_col].isin([pre_group, post_group])].copy()
    df = df.dropna(subset=[mean_col, var_col])

    means = df[mean_col].astype(float)
    vars_ = df[var_col].astype(float).clip(lower=0.0)

    # Avoid division by zero: set CV to NaN where mean == 0
    means_safe = means.replace(0.0, np.nan)
    df["cv"] = np.sqrt(vars_) / means_safe

    # Drop rows where CV is NaN (mean=0 or missing)
    df = df.dropna(subset=["cv"])

    grp = df.groupby([id_col, group_col])["cv"].mean().rename("cv_mean")
    wide = grp.unstack(group_col)

    if pre_group not in wide.columns or post_group not in wide.columns:
        raise RuntimeError(
            f"Expected groups {pre_group!r} and {post_group!r} for CV in column {group_col!r}."
        )

    wide = wide[[pre_group, post_group]].dropna(how="any")
    wide = wide.rename(columns={pre_group: "Pre_cv", post_group: "Post_cv"})
    wide.index.name = id_col

    return wide


# ---------------------------------------------------------------------
# Treatment type classification
# ---------------------------------------------------------------------
def _classify_treatment_type(raw: Any) -> str:
    """
    Collapse free-text treatment descriptions into:
        'nma'   -> bilateral NMA lesion injections (experimental)
        'sham'  -> saline sham injections (control)
        'other' -> anything else
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "other"
    s = str(raw).strip().lower()
    if "sham" in s or "saline" in s:
        return "sham"
    if "nma" in s:
        return "nma"
    return "other"


# ---------------------------------------------------------------------
# Plotting: two-panel figure with boxplots & paired lines (log-variance)
# ---------------------------------------------------------------------
def _make_two_panel_boxplot_figure_logvar(
    wide_logvar: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    log_label: str = "log₁₀ variance (ms²)",
    output_dir: Union[str, Path],
    file_prefix: str = "phrase_log_variance_pre_post",
    show_plots: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[Path]]:
    """
    Build a figure with two panels (NMA lesion vs sham saline),
    run paired t-tests, and return (p_nma, p_sham, fig_path).

    Each panel gets its own legend to the RIGHT of that panel.
    The y-axis is fixed to [4.0, 6.5] in log10(ms²) units.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError(
            "SciPy is required for t-tests but could not be imported. "
            "Install scipy or modify the code to use a different stats package."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify each bird by treatment type
    meta_df = meta_df.copy()
    meta_df["treat_class"] = meta_df["Treatment type"].map(_classify_treatment_type)

    # Merge treatment class into wide_logvar
    wide = wide_logvar.reset_index()
    wide = wide.merge(
        meta_df[[id_col, "Treatment type", "treat_class"]],
        on=id_col,
        how="left",
    )
    wide = wide.set_index(id_col)

    # Split into NMA and sham sets
    nma_df = wide[wide["treat_class"] == "nma"][["Pre_logvar", "Post_logvar"]]
    sham_df = wide[wide["treat_class"] == "sham"][["Pre_logvar", "Post_logvar"]]

    # Prepare a global marker map so each bird has a unique marker
    all_ids: List[str] = list(wide.index.astype(str))
    all_ids_sorted = sorted(set(all_ids))
    marker_list = ["o", "s", "D", "v", "^", "<", ">", "P", "X", "*", "p", "h", "H", "8"]
    marker_map: Dict[str, str] = {}
    for i, bid in enumerate(all_ids_sorted):
        marker_map[bid] = marker_list[i % len(marker_list)]

    # Bigger, wider figure so everything is readable
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 6),
        sharey=True,
        constrained_layout=False,
    )

    y_lower, y_upper = 4.0, 6.5
    y_range = y_upper - y_lower

    def _paired_boxplot(
        ax: plt.Axes,
        group_df: pd.DataFrame,
        group_label: str,
        color_pre: str,
        color_post: str,
    ) -> Optional[float]:
        """Draw one panel and return the p-value (or None if not enough birds)."""
        if group_df.empty:
            ax.set_visible(False)
            print(f"[INFO] No birds found for group {group_label!r}.")
            return None

        group_df = group_df.dropna(subset=["Pre_logvar", "Post_logvar"])
        if group_df.empty:
            ax.set_visible(False)
            print(f"[INFO] All log-variance values NaN for group {group_label!r}.")
            return None

        # Paired t-test
        pre_vals = group_df["Pre_logvar"].values
        post_vals = group_df["Post_logvar"].values

        if len(pre_vals) < 2:
            t_stat, p_val = np.nan, np.nan
            print(
                f"{group_label}: fewer than 2 birds (n={len(pre_vals)}). "
                "Skipping formal t-test."
            )
        else:
            t_stat, p_val = _scipy_stats.ttest_rel(
                pre_vals, post_vals, nan_policy="omit"
            )
            print(
                f"{group_label}: paired t-test (Pre vs Post, log-variance) "
                f"t = {t_stat:.3f}, p = {p_val:.4e}, n = {len(pre_vals)}"
            )

        # Positions for Pre and Post
        x_pre, x_post = 0, 1
        positions = [x_pre, x_post]

        # Boxplots
        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )

        box_colors = [color_pre, color_post]
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.3)
            patch.set_edgecolor("black")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1.0)

        # Overlay individual birds & connecting lines
        bird_ids = group_df.index.astype(str)
        for bid, pre, post in zip(bird_ids, pre_vals, post_vals):
            m = marker_map.get(bid, "o")
            ax.plot(
                [x_pre, x_post],
                [pre, post],
                color="lightgray",
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                x_pre,
                pre,
                color=color_pre,
                edgecolors="black",
                marker=m,
                s=60,
                zorder=2,
            )
            ax.scatter(
                x_post,
                post,
                color=color_post,
                edgecolors="black",
                marker=m,
                s=60,
                zorder=2,
            )

        # X-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels(["Pre lesion", "Post lesion"])
        ax.set_ylabel(log_label)
        ax.set_title(f"{group_label} (n = {len(group_df)} birds)")

        # Fixed y-limits: log10 variance from 4.0 to 6.5
        ax.set_ylim(y_lower, y_upper)

        # Significance stars (relative to y-range)
        if not np.isfinite(p_val):
            sig = "n/a"
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        bracket_y = y_upper - 0.25 * y_range
        text_y = y_upper - 0.12 * y_range

        ax.plot(
            [x_pre, x_pre, x_post, x_post],
            [bracket_y, bracket_y + 0.08 * y_range,
             bracket_y + 0.08 * y_range, bracket_y],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (x_pre + x_post),
            text_y,
            sig,
            ha="center",
            va="bottom",
            fontsize=13,
        )

        # Per-panel legend (only birds in this group), to the right of the axis
        unique_birds = sorted(set(bird_ids))
        legend_handles: List[mlines.Line2D] = []
        for bid in unique_birds:
            m = marker_map.get(bid, "o")
            handle = mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker=m,
                markersize=8,
                markerfacecolor="gray",
                markeredgecolor="black",
                label=bid,
            )
            legend_handles.append(handle)

        ax.legend(
            handles=legend_handles,
            title="Bird",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            ncol=1,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            framealpha=0.9,
        )

        return p_val

    # Left: NMA lesions
    p_nma = _paired_boxplot(
        axes[0],
        nma_df,
        "Bilateral NMA lesion injections",
        color_pre="#6baed6",  # blue-ish
        color_post="#fb6a4a",  # red-ish
    )

    # Right: sham saline
    p_sham = _paired_boxplot(
        axes[1],
        sham_df,
        "Bilateral saline sham injection",
        color_pre="#6baed6",
        color_post="#fb6a4a",
    )

    # Layout: leave some space on the right for legends
    plt.subplots_adjust(
        left=0.07,
        right=0.90,
        top=0.92,
        bottom=0.12,
        wspace=0.35,
    )

    fig_path = output_dir / f"{file_prefix}_NMA_vs_sham_log_variance.png"
    fig.savefig(fig_path, dpi=600)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f"[PLOT] Saved log-variance boxplot figure: {fig_path}")
    return p_nma, p_sham, fig_path


# ---------------------------------------------------------------------
# Plotting: two-panel figure with boxplots & paired lines (CV)
# ---------------------------------------------------------------------
def _make_two_panel_boxplot_figure_cv(
    wide_cv: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    id_col: str = "Animal ID",
    y_label: str = "Coefficient of variation (σ / μ)",
    output_dir: Union[str, Path],
    file_prefix: str = "phrase_cv_pre_post",
    show_plots: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[Path]]:
    """
    Build a figure with two panels (NMA lesion vs sham saline),
    run paired t-tests on CV, and return (p_nma, p_sham, fig_path).

    Y-limits for BOTH panels are determined from the min and max CV
    values of the NMA lesion group (Pre + Post), with a small padding.
    Each panel has its bird legend to the RIGHT of that panel.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError(
            "SciPy is required for t-tests but could not be imported. "
            "Install scipy or modify the code to use a different stats package."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify each bird by treatment type
    meta_df = meta_df.copy()
    meta_df["treat_class"] = meta_df["Treatment type"].map(_classify_treatment_type)

    # Merge treatment class into wide_cv
    wide = wide_cv.reset_index()
    wide = wide.merge(
        meta_df[[id_col, "Treatment type", "treat_class"]],
        on=id_col,
        how="left",
    )
    wide = wide.set_index(id_col)

    # Split into NMA and sham sets
    nma_df = wide[wide["treat_class"] == "nma"][["Pre_cv", "Post_cv"]]
    sham_df = wide[wide["treat_class"] == "sham"][["Pre_cv", "Post_cv"]]

    # Prepare a global marker map so each bird has a unique marker
    all_ids: List[str] = list(wide.index.astype(str))
    all_ids_sorted = sorted(set(all_ids))
    marker_list = ["o", "s", "D", "v", "^", "<", ">", "P", "X", "*", "p", "h", "H", "8"]
    marker_map: Dict[str, str] = {}
    for i, bid in enumerate(all_ids_sorted):
        marker_map[bid] = marker_list[i % len(marker_list)]

    # --- Global y-limits based ONLY on NMA CV values ---
    if nma_df.empty:
        # If no NMA birds, fall back to all data
        all_vals = np.concatenate(
            [wide_cv["Pre_cv"].values, wide_cv["Post_cv"].values]
        )
    else:
        all_vals = np.concatenate(
            [nma_df["Pre_cv"].values, nma_df["Post_cv"].values]
        )

    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        y_min_global, y_max_global = 0.0, 1.0
    else:
        y_min_global = float(np.nanmin(all_vals))
        y_max_global = float(np.nanmax(all_vals))
        if y_min_global == y_max_global:
            y_min_global -= 0.1 * abs(y_min_global) if y_min_global != 0 else 0.1
            y_max_global += 0.1 * abs(y_max_global) if y_max_global != 0 else 0.1

    y_range_global = y_max_global - y_min_global
    pad = 0.08 * y_range_global
    y_lower = y_min_global - pad
    y_upper = y_max_global + 1.6 * pad

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 6),
        sharey=True,
        constrained_layout=False,
    )

    def _paired_boxplot(
        ax: plt.Axes,
        group_df: pd.DataFrame,
        group_label: str,
        color_pre: str,
        color_post: str,
    ) -> Optional[float]:
        if group_df.empty:
            ax.set_visible(False)
            print(f"[INFO] No birds found for CV in group {group_label!r}.")
            return None

        group_df = group_df.dropna(subset=["Pre_cv", "Post_cv"])
        if group_df.empty:
            ax.set_visible(False)
            print(f"[INFO] All CV values NaN for group {group_label!r}.")
            return None

        pre_vals = group_df["Pre_cv"].values
        post_vals = group_df["Post_cv"].values

        if len(pre_vals) < 2:
            t_stat, p_val = np.nan, np.nan
            print(
                f"{group_label} (CV): fewer than 2 birds (n={len(pre_vals)}). "
                "Skipping formal t-test."
            )
        else:
            t_stat, p_val = _scipy_stats.ttest_rel(
                pre_vals, post_vals, nan_policy="omit"
            )
            print(
                f"{group_label}: paired t-test (Pre vs Post, CV) "
                f"t = {t_stat:.3f}, p = {p_val:.4e}, n = {len(pre_vals)}"
            )

        x_pre, x_post = 0, 1
        positions = [x_pre, x_post]

        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        box_colors = [color_pre, color_post]
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.3)
            patch.set_edgecolor("black")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1.0)

        bird_ids = group_df.index.astype(str)
        for bid, pre, post in zip(bird_ids, pre_vals, post_vals):
            m = marker_map.get(bid, "o")
            ax.plot(
                [x_pre, x_post],
                [pre, post],
                color="lightgray",
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                x_pre,
                pre,
                color=color_pre,
                edgecolors="black",
                marker=m,
                s=60,
                zorder=2,
            )
            ax.scatter(
                x_post,
                post,
                color=color_post,
                edgecolors="black",
                marker=m,
                s=60,
                zorder=2,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(["Pre lesion", "Post lesion"])
        ax.set_ylabel(y_label)
        ax.set_title(f"{group_label} (n = {len(group_df)} birds)")

        ax.set_ylim(y_lower, y_upper)

        # Significance stars
        if not np.isfinite(p_val):
            sig = "n/a"
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        bracket_y = y_upper - 0.25 * y_range_global
        text_y = y_upper - 0.12 * y_range_global

        ax.plot(
            [x_pre, x_pre, x_post, x_post],
            [bracket_y, bracket_y + 0.08 * y_range_global,
             bracket_y + 0.08 * y_range_global, bracket_y],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (x_pre + x_post),
            text_y,
            sig,
            ha="center",
            va="bottom",
            fontsize=13,
        )

        # Per-panel legend to the right of that axis
        unique_birds = sorted(set(bird_ids))
        legend_handles: List[mlines.Line2D] = []
        for bid in unique_birds:
            m = marker_map.get(bid, "o")
            handle = mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker=m,
                markersize=8,
                markerfacecolor="gray",
                markeredgecolor="black",
                label=bid,
            )
            legend_handles.append(handle)

        ax.legend(
            handles=legend_handles,
            title="Bird",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            ncol=1,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            framealpha=0.9,
        )

        return p_val

    p_nma = _paired_boxplot(
        axes[0],
        nma_df,
        "Bilateral NMA lesion injections",
        color_pre="#6baed6",
        color_post="#fb6a4a",
    )
    p_sham = _paired_boxplot(
        axes[1],
        sham_df,
        "Bilateral saline sham injection",
        color_pre="#6baed6",
        color_post="#fb6a4a",
    )

    plt.subplots_adjust(
        left=0.07,
        right=0.90,
        top=0.92,
        bottom=0.12,
        wspace=0.35,
    )

    fig_path = output_dir / f"{file_prefix}_NMA_vs_sham_cv.png"
    fig.savefig(fig_path, dpi=600)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print(f"[PLOT] Saved CV boxplot figure: {fig_path}")
    return p_nma, p_sham, fig_path


# ---------------------------------------------------------------------
# Public wrapper (includes MWU & Levene on log-variance)
# ---------------------------------------------------------------------
def run_phrase_variance_stats(
    compiled_stats_path: Union[str, Path],
    metadata_excel_path: Union[str, Path],
    *,
    compiled_format: Optional[str] = None,
    metadata_sheet_name: Union[int, str] = "metadata",
    output_dir: Optional[Union[str, Path]] = None,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    mean_col: str = "Mean_ms",
    var_col: str = "Variance_ms2",
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    log_mode: str = "log10",
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    High-level helper to:
      1) Load compiled phrase stats and metadata
      2) Compute per-bird log-variance (Late Pre vs Post)
      3) Compute per-bird CV (Late Pre vs Post)
      4) Make two-panel NMA vs sham figures for both log-variance and CV
      5) Run paired t-tests within groups for both measures
      6) Run Mann–Whitney U and Levene on log-variance differences (Post-Pre)
      7) Run between-groups tests on signed Δ log-variance (NMA vs sham)
    """
    compiled_stats_path = Path(compiled_stats_path)

    # Output directory
    if output_dir is None:
        output_dir = compiled_stats_path.parent / "phrase_duration_variance_stats"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phrase_df = _load_phrase_stats(compiled_stats_path, compiled_format=compiled_format)
    meta_df = _load_metadata(
        metadata_excel_path,
        sheet_name=metadata_sheet_name,
        id_col=id_col,
    )

    # --- Log-variance per bird ---
    wide_logvar = _compute_per_bird_log_variance(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        var_col=var_col,
        pre_group=pre_group,
        post_group=post_group,
        log_mode=log_mode,
    )
    log_label = wide_logvar.attrs.get("log_label", "log₁₀ variance (ms²)")

    # --- CV per bird ---
    wide_cv = _compute_per_bird_cv(
        phrase_df,
        id_col=id_col,
        group_col=group_col,
        mean_col=mean_col,
        var_col=var_col,
        pre_group=pre_group,
        post_group=post_group,
    )

    # --------- MWU, Levene, and between-groups tests on log-variance Δ ----------
    mw_p_abs = np.nan
    lev_p = np.nan
    t_p_signed = np.nan
    mw_p_signed = np.nan

    if _HAVE_SCIPY:
        meta_for_stats = meta_df.copy()
        meta_for_stats["treat_class"] = meta_for_stats["Treatment type"].map(
            _classify_treatment_type
        )

        merged = wide_logvar.reset_index().merge(
            meta_for_stats[[id_col, "treat_class"]],
            on=id_col,
            how="left",
        )

        sham_stats = merged[merged["treat_class"] == "sham"][
            ["Pre_logvar", "Post_logvar", id_col]
        ].dropna()
        nma_stats = merged[merged["treat_class"] == "nma"][
            ["Pre_logvar", "Post_logvar", id_col]
        ].dropna()

        if (not sham_stats.empty) and (not nma_stats.empty):
            ctrl_diffs = (
                sham_stats["Post_logvar"].values - sham_stats["Pre_logvar"].values
            )
            exp_diffs = (
                nma_stats["Post_logvar"].values - nma_stats["Pre_logvar"].values
            )

            ctrl_abs_dev = np.abs(ctrl_diffs)
            exp_abs_dev = np.abs(exp_diffs)

            # Test 1: Mann–Whitney U on absolute deviation
            mw_u_abs, mw_p_abs = _scipy_stats.mannwhitneyu(
                exp_abs_dev, ctrl_abs_dev, alternative="greater"
            )

            print("\n--- TEST 1: |Δ log10 variance| (Post-Pre) ---")
            print(
                f"Mean |Δ log10 var| (Control; sham): {np.mean(ctrl_abs_dev):.4f}"
            )
            print(
                f"Mean |Δ log10 var| (Experimental; NMA): {np.mean(exp_abs_dev):.4f}"
            )
            print(f"Mann-Whitney U p-value (exp > ctrl): {mw_p_abs:.5f}")
            if mw_p_abs < 0.05:
                print(
                    "RESULT: Significant. NMA lesions cause larger deviations from baseline."
                )

            # Test 2: Levene's test on differences
            lev_stat, lev_p = _scipy_stats.levene(ctrl_diffs, exp_diffs)

            print("\n--- TEST 2: Levene on Δ log10 variance (Post-Pre) ---")
            print(f"Variance of Δ (Control; sham): {np.var(ctrl_diffs):.4f}")
            print(f"Variance of Δ (Experimental; NMA): {np.var(exp_diffs):.4f}")
            print(f"Levene p-value: {lev_p:.5f}")
            if lev_p < 0.05:
                print(
                    "RESULT: Significant. NMA lesions increase variability of the change."
                )

            # Test 3: Between-groups tests on signed Δ log10 variance
            t_stat_signed, t_p_signed = _scipy_stats.ttest_ind(
                exp_diffs, ctrl_diffs, equal_var=False
            )
            mw_u_signed, mw_p_signed = _scipy_stats.mannwhitneyu(
                exp_diffs, ctrl_diffs, alternative="two-sided"
            )

            print("\n--- TEST 3: Between-groups on signed Δ log10 variance (Post-Pre) ---")
            print(
                f"Mean Δ log10 var (Control; sham): {np.mean(ctrl_diffs):.4f}"
            )
            print(
                f"Mean Δ log10 var (Experimental; NMA): {np.mean(exp_diffs):.4f}"
            )
            print(
                f"Welch t-test p-value (two-sided): {t_p_signed:.5f}"
            )
            print(
                f"Mann-Whitney p-value (two-sided): {mw_p_signed:.5f}"
            )

        else:
            print(
                "\n[INFO] Not enough birds in one or both groups for MWU/Levene/Δ tests "
                "on log-variance."
            )
    else:
        print(
            "\n[WARN] SciPy not available; skipping Mann–Whitney, Levene, and Δ tests."
        )

    # --- Figures + paired t-tests ---
    p_nma_log, p_sham_log, fig_path_log = _make_two_panel_boxplot_figure_logvar(
        wide_logvar,
        meta_df,
        id_col=id_col,
        log_label=log_label,
        output_dir=output_dir,
        show_plots=show_plots,
    )

    p_nma_cv, p_sham_cv, fig_path_cv = _make_two_panel_boxplot_figure_cv(
        wide_cv,
        meta_df,
        id_col=id_col,
        y_label="Coefficient of variation (σ / μ)",
        output_dir=output_dir,
        show_plots=show_plots,
    )

    return {
        "wide_logvar": wide_logvar,
        "wide_cv": wide_cv,
        "metadata": meta_df,
        "p_nma": p_nma_log,
        "p_sham": p_sham_log,
        "p_nma_cv": p_nma_cv,
        "p_sham_cv": p_sham_cv,
        "p_mw_abs_dev": mw_p_abs,
        "p_levene_diffs": lev_p,
        "p_t_signed_delta": t_p_signed,
        "p_mw_signed_delta": mw_p_signed,
        "figure_path": fig_path_log,
        "figure_path_cv": fig_path_cv,
    }


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Pre vs Post log-variance & CV boxplots and stats for "
            "Bilateral NMA lesions vs saline sham injections."
        )
    )
    parser.add_argument(
        "compiled_stats_path",
        type=str,
        help="Path to compiled phrase-duration stats file (CSV / JSON / NPZ).",
    )
    parser.add_argument(
        "metadata_excel_path",
        type=str,
        help="Path to Area X lesion metadata Excel file.",
    )
    parser.add_argument(
        "--compiled_format",
        type=str,
        default=None,
        choices=["csv", "json", "npz"],
        help="File format of compiled stats (inferred from suffix if omitted).",
    )
    parser.add_argument(
        "--metadata_sheet_name",
        type=str,
        default="metadata",
        help="Sheet name or index for metadata Excel (default: 'metadata').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save figures "
            "(default: 'phrase_duration_variance_stats' next to compiled stats)."
        ),
    )
    parser.add_argument(
        "--log_mode",
        type=str,
        default="log10",
        choices=["log10", "ln", "log"],
        help="Which log to use for variance (default: log10).",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, show the figures interactively in addition to saving.",
    )

    args = parser.parse_args()

    # Allow sheet name to be "0" or an actual string
    sheet_name: Union[int, str]
    try:
        sheet_name = int(args.metadata_sheet_name)
    except ValueError:
        sheet_name = args.metadata_sheet_name

    out_dir = Path(args.output_dir) if args.output_dir is not None else None

    res = run_phrase_variance_stats(
        compiled_stats_path=args.compiled_stats_path,
        metadata_excel_path=args.metadata_excel_path,
        compiled_format=args.compiled_format,
        metadata_sheet_name=sheet_name,
        output_dir=out_dir,
        log_mode=args.log_mode,
        show_plots=args.show_plots,)
    
    print("\n--- Paired t-test results (log-variance) ---")
    print(f"NMA lesions log-var t-test p-value: {res['p_nma']}")
    print(f"Sham saline log-var t-test p-value: {res['p_sham']}")
    print("\n--- Paired t-test results (CV) ---")
    print(f"NMA lesions CV t-test p-value: {res['p_nma_cv']}")
    print(f"Sham saline CV t-test p-value: {res['p_sham_cv']}")
    print("\n--- MWU / Levene / Δ tests on log-variance ---")
    print(f"Mann-Whitney |Δ| p-value (NMA > sham): {res['p_mw_abs_dev']}")
    print(f"Levene Δ variance p-value: {res['p_levene_diffs']}")
    print(f"Welch t-test signed Δ p-value (two-sided): {res['p_t_signed_delta']}")
    print(f"Mann-Whitney signed Δ p-value (two-sided): {res['p_mw_signed_delta']}")
    print(f"\nLog-variance figure saved at: {res['figure_path']}")
    print(f"CV figure saved at: {res['figure_path_cv']}")

"""
from pathlib import Path
import importlib
import phrase_duration_stats as pds
importlib.reload(pds)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
excel_path   = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")

res = pds.run_phrase_variance_stats(
    compiled_stats_path=compiled_csv,
    metadata_excel_path=excel_path,
    output_dir=compiled_csv.parent / "phrase_duration_variance_stats",
    show_plots=True,
)

print("NMA log-var p:", res["p_nma"])
print("Sham log-var p:", res["p_sham"])
print("NMA CV p:", res["p_nma_cv"])
print("Sham CV p:", res["p_sham_cv"])
print("Log-var fig:", res["figure_path"])
print("CV fig:", res["figure_path_cv"])


"""