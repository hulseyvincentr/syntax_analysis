#!/usr/bin/env python3
"""
Cleaned Panel D phrase-duration variability time-course plots.

This script reads the daily top-variance phrase-duration table and makes clearer
time-course figures designed to show whether variability remains elevated after
lesion.

Main outputs:
  1. panel_D_clean_sd_timecourse_raw_ms.png
     - y-axis = phrase-duration standard deviation in ms
     - faint lines = individual animal-level trajectories
     - thick line = group rolling median
     - shaded region = rolling IQR

  2. panel_D_clean_sd_timecourse_delta_from_late_pre.png
     - y-axis = daily SD minus each animal's own Late Pre baseline SD
     - useful for showing persistence relative to baseline

  3. panel_D_clean_sd_period_summary_boxplot.png
     - Late Pre vs Early Post vs Late Post summary
     - each point = one animal

The rolling summaries are computed separately on the pre-lesion side and
post-lesion side, so the smoothing window does not blur across lesion day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────────
# Display groups and colors
# ──────────────────────────────────────────────────────────────────────────────
SHAM_GROUP = "sham saline injection"
LATERAL_ONLY_GROUP = "Lateral lesion only"
PARTIAL_ML_GROUP = "Partial Medial and Lateral lesion"
COMPLETE_ML_GROUP = "Complete Medial and Lateral lesion"
POOLED_ML_GROUP = "Complete and partial medial and lateral lesion"

DEFAULT_COLORS = {
    COMPLETE_ML_GROUP: "#3F007D",
    PARTIAL_ML_GROUP: "#7A4FB7",
    LATERAL_ONLY_GROUP: "#A88BD9",
    SHAM_GROUP: "#1B9E77",
    POOLED_ML_GROUP: "#7A4FB7",
    "unknown": "#4D4D4D",
}

GROUP_ALIASES = {
    COMPLETE_ML_GROUP: [
        "complete medial and lateral lesion",
        "complete medial+lateral lesion",
        "large lesion area x not visible",
        "area x not visible",
        "complete m+l",
        "complete ml",
    ],
    PARTIAL_ML_GROUP: [
        "partial medial and lateral lesion",
        "partial medial+lateral lesion",
        "area x visible (medial+lateral hit)",
        "medial+lateral hit",
        "m+l hit",
        "partial m+l",
        "partial ml",
    ],
    LATERAL_ONLY_GROUP: [
        "lateral lesion only",
        "lateral hit only",
        "area x visible (single hit)",
        "single hit",
        "lateral only",
    ],
    SHAM_GROUP: [
        "sham saline injection",
        "sham",
        "saline",
    ],
}


def _norm_label(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def canonical_group(raw: Any) -> str:
    """Map raw metadata / lesion labels to a display group."""
    s = _norm_label(raw)
    if not s:
        return "unknown"

    for display, aliases in GROUP_ALIASES.items():
        if s == _norm_label(display):
            return display
        if any(s == _norm_label(a) for a in aliases):
            return display

    # Fallback fuzzy matching.
    if "sham" in s or "saline" in s:
        return SHAM_GROUP
    if "lateral" in s and "medial" in s:
        if "complete" in s or "not visible" in s or "large" in s:
            return COMPLETE_ML_GROUP
        return PARTIAL_ML_GROUP
    if "lateral" in s:
        return LATERAL_ONLY_GROUP

    return str(raw)


def pool_display_group(group: Any) -> str:
    g = str(group)
    if g in {COMPLETE_ML_GROUP, PARTIAL_ML_GROUP}:
        return POOLED_ML_GROUP
    return g


def color_for_group(group: str) -> str:
    return DEFAULT_COLORS.get(group, "#4D4D4D")


def pretty_group_label(group: str) -> str:
    if group == SHAM_GROUP:
        return "sham saline\ninjection"
    if group == LATERAL_ONLY_GROUP:
        return "Lateral lesion\nonly"
    if group == POOLED_ML_GROUP:
        return "Complete and partial\nmedial and lateral lesion"
    if group == PARTIAL_ML_GROUP:
        return "Partial medial and\nlateral lesion"
    if group == COMPLETE_ML_GROUP:
        return "Complete medial and\nlateral lesion"
    return str(group)


GROUP_ORDER_POOLED = [SHAM_GROUP, LATERAL_ONLY_GROUP, POOLED_ML_GROUP]
GROUP_ORDER_SEPARATE = [SHAM_GROUP, LATERAL_ONLY_GROUP, PARTIAL_ML_GROUP, COMPLETE_ML_GROUP]


# ──────────────────────────────────────────────────────────────────────────────
# Input loading
# ──────────────────────────────────────────────────────────────────────────────
def load_hit_type_map(
    metadata_excel: Optional[Union[str, Path]],
    *,
    sheet_name: Union[str, int] = "animal_hit_type_summary",
    animal_col: str = "Animal ID",
    hit_type_col: str = "Lesion hit type",
) -> Dict[str, str]:
    if metadata_excel is None:
        return {}

    metadata_excel = Path(metadata_excel)
    if not metadata_excel.exists():
        raise FileNotFoundError(f"Could not find metadata Excel file: {metadata_excel}")

    try:
        meta = pd.read_excel(metadata_excel, sheet_name=sheet_name)
    except ValueError:
        # Helpful fallback if the requested sheet name is not present.
        meta = pd.read_excel(metadata_excel, sheet_name=0)

    if animal_col not in meta.columns:
        raise ValueError(
            f"Metadata file is missing animal column {animal_col!r}. "
            f"Found columns: {list(meta.columns)}"
        )

    if hit_type_col not in meta.columns:
        # Try a few likely alternatives.
        candidates = [
            "Lesion hit type",
            "lesion hit type",
            "lesion hit type grouping",
            "Hit type",
            "hit_type",
            "Treatment type",
        ]
        found = next((c for c in candidates if c in meta.columns), None)
        if found is None:
            raise ValueError(
                f"Metadata file is missing hit-type column {hit_type_col!r}. "
                f"Found columns: {list(meta.columns)}"
            )
        hit_type_col = found

    sub = meta[[animal_col, hit_type_col]].dropna(subset=[animal_col]).copy()
    sub[animal_col] = sub[animal_col].astype(str)
    return dict(zip(sub[animal_col], sub[hit_type_col]))


def load_daily_table(
    daily_csv: Union[str, Path],
    *,
    metadata_excel: Optional[Union[str, Path]] = None,
    meta_sheet_name: Union[str, int] = "animal_hit_type_summary",
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: str = "Lesion hit type",
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    pool_medial_lateral: bool = True,
) -> pd.DataFrame:
    daily_csv = Path(daily_csv)
    if not daily_csv.exists():
        raise FileNotFoundError(f"Could not find daily CSV: {daily_csv}")

    df = pd.read_csv(daily_csv)
    required = ["animal_id", "syllable", "relative_day", "Variance (ms^2)", "lesion_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Daily CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["animal_id"] = df["animal_id"].astype(str)
    df["syllable"] = df["syllable"].astype(str)
    df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    df["Variance (ms^2)"] = pd.to_numeric(df["Variance (ms^2)"], errors="coerce")
    df = df.dropna(subset=["relative_day", "Variance (ms^2)"])
    df = df[df["Variance (ms^2)"] >= 0].copy()

    hit_map = load_hit_type_map(
        metadata_excel,
        sheet_name=meta_sheet_name,
        animal_col=meta_animal_col,
        hit_type_col=meta_hit_type_col,
    ) if metadata_excel is not None else {}

    df["raw_hit_type"] = df["animal_id"].map(hit_map)
    df["display_group"] = [
        canonical_group(raw if pd.notna(raw) and str(raw).strip() else lesion_group)
        for raw, lesion_group in zip(df["raw_hit_type"], df["lesion_group"])
    ]
    if pool_medial_lateral:
        df["display_group"] = df["display_group"].map(pool_display_group)

    df["sd_ms"] = np.sqrt(df["Variance (ms^2)"].astype(float))

    if x_min is not None:
        df = df[df["relative_day"] >= int(x_min)]
    if x_max is not None:
        df = df[df["relative_day"] <= int(x_max)]

    return df.sort_values(["display_group", "animal_id", "syllable", "relative_day"]).reset_index(drop=True)


def make_daily_unit_table(
    df: pd.DataFrame,
    *,
    unit_level: str = "animal",
    within_unit_stat: str = "median",
) -> pd.DataFrame:
    """Collapse data to one row per unit per day.

    unit_level="animal" is recommended for manuscript figures because each line
    is one bird per day after collapsing across selected syllables.
    """
    unit_level = unit_level.lower()
    within_unit_stat = within_unit_stat.lower()

    if unit_level == "animal":
        group_cols = ["display_group", "animal_id", "relative_day"]
        unit_col = "animal_id"
    elif unit_level == "animal_syllable":
        df = df.copy()
        df["unit_id"] = df["animal_id"].astype(str) + "::" + df["syllable"].astype(str)
        group_cols = ["display_group", "animal_id", "syllable", "unit_id", "relative_day"]
        unit_col = "unit_id"
    else:
        raise ValueError("--unit-level must be 'animal' or 'animal_syllable'")

    if within_unit_stat == "median":
        out = df.groupby(group_cols, dropna=False)["sd_ms"].median().reset_index()
    elif within_unit_stat == "mean":
        out = df.groupby(group_cols, dropna=False)["sd_ms"].mean().reset_index()
    else:
        raise ValueError("--within-unit-stat must be 'median' or 'mean'")

    if unit_level == "animal":
        out["unit_id"] = out["animal_id"].astype(str)

    return out.sort_values(["display_group", "unit_id", "relative_day"]).reset_index(drop=True)


def add_baseline_columns(
    unit_df: pd.DataFrame,
    *,
    baseline_start: int = -7,
    baseline_end: int = -1,
) -> pd.DataFrame:
    out = unit_df.copy()

    baseline_rows = out[
        (out["relative_day"] >= baseline_start)
        & (out["relative_day"] <= baseline_end)
    ].copy()

    # Fallback to all pre-lesion days for units lacking baseline-window data.
    baseline = baseline_rows.groupby("unit_id")["sd_ms"].median()
    all_pre = out[out["relative_day"] < 0].groupby("unit_id")["sd_ms"].median()
    baseline = baseline.combine_first(all_pre)

    out["baseline_sd_ms"] = out["unit_id"].map(baseline)
    out["sd_delta_from_late_pre_ms"] = out["sd_ms"] - out["baseline_sd_ms"]
    out["sd_ratio_to_late_pre"] = out["sd_ms"] / out["baseline_sd_ms"]
    out.loc[~np.isfinite(out["sd_ratio_to_late_pre"]), "sd_ratio_to_late_pre"] = np.nan
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Rolling summaries
# ──────────────────────────────────────────────────────────────────────────────
def rolling_window_summary(
    unit_df: pd.DataFrame,
    *,
    value_col: str,
    window_days: int = 7,
    summary_stat: str = "median",
    no_cross_lesion_smoothing: bool = True,
) -> pd.DataFrame:
    """Compute group rolling median/mean and IQR by relative day.

    For each group and day, this collects unit-level daily values within a
    centered rolling day window. If no_cross_lesion_smoothing=True, pre-lesion
    days only borrow from pre-lesion days and post-lesion days only borrow from
    post-lesion days.
    """
    if value_col not in unit_df.columns:
        raise ValueError(f"Missing value column {value_col!r}")

    half = max(int(window_days) // 2, 0)
    rows = []

    summary_stat = summary_stat.lower()
    group_order = GROUP_ORDER_POOLED if POOLED_ML_GROUP in set(unit_df["display_group"]) else GROUP_ORDER_SEPARATE
    groups = [g for g in group_order if g in set(unit_df["display_group"])]
    groups += [g for g in sorted(set(unit_df["display_group"])) if g not in groups]

    for group in groups:
        gdf = unit_df[unit_df["display_group"] == group].copy()
        days = sorted(gdf["relative_day"].dropna().astype(int).unique())

        for day in days:
            lo, hi = day - half, day + half
            window = gdf[(gdf["relative_day"] >= lo) & (gdf["relative_day"] <= hi)].copy()

            if no_cross_lesion_smoothing:
                if day < 0:
                    window = window[window["relative_day"] < 0]
                elif day > 0:
                    window = window[window["relative_day"] > 0]
                else:
                    window = window[window["relative_day"] == 0]

            vals = window[value_col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            center_vals = gdf[gdf["relative_day"] == day][value_col].dropna().to_numpy(dtype=float)
            rows.append({
                "display_group": group,
                "relative_day": day,
                "value_col": value_col,
                "window_days": window_days,
                "summary": float(np.nanmedian(vals) if summary_stat == "median" else np.nanmean(vals)),
                "q25": float(np.nanpercentile(vals, 25)),
                "q75": float(np.nanpercentile(vals, 75)),
                "mean": float(np.nanmean(vals)),
                "median": float(np.nanmedian(vals)),
                "n_values_window": int(len(vals)),
                "n_units_window": int(window["unit_id"].nunique()),
                "n_units_day": int(gdf.loc[gdf["relative_day"] == day, "unit_id"].nunique()),
                "day_raw_median": float(np.nanmedian(center_vals)) if len(center_vals) else np.nan,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def pretty_axes(ax: plt.Axes, tick_label_fontsize: float = 12) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def set_common_ylim_from_percentile(
    axes: Sequence[plt.Axes],
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    include_zero: bool = False,
    pad_frac: float = 0.08,
) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    lo = float(np.nanpercentile(vals, lower_percentile))
    hi = float(np.nanpercentile(vals, upper_percentile))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span

    for ax in axes:
        ax.set_ylim(lo, hi)


def plot_clean_timecourse_panels(
    unit_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    value_col: str,
    y_label: str,
    title: Optional[str] = None,
    group_order: Optional[Sequence[str]] = None,
    x_min: int = -30,
    x_max: int = 30,
    common_ylim: bool = True,
    y_lower_percentile: float = 1.0,
    y_upper_percentile: float = 99.0,
    no_percentile_ylim: bool = False,
    include_zero_in_ylim: bool = False,
    raw_alpha: float = 0.12,
    raw_linewidth: float = 0.8,
    smooth_linewidth: float = 3.0,
    iqr_alpha: float = 0.22,
    figsize: Tuple[float, float] = (14, 8.6),
    dpi: int = 300,
    show: bool = False,
) -> Path:
    out_path = Path(out_path)

    if group_order is None:
        group_order = GROUP_ORDER_POOLED if POOLED_ML_GROUP in set(unit_df["display_group"]) else GROUP_ORDER_SEPARATE
    group_order = [g for g in group_order if g in set(unit_df["display_group"])]

    fig, axes = plt.subplots(len(group_order), 1, figsize=figsize, sharex=True, sharey=common_ylim)
    if len(group_order) == 1:
        axes = [axes]

    for ax, group in zip(axes, group_order):
        color = color_for_group(group)
        g = unit_df[unit_df["display_group"] == group].copy()

        # Faint raw unit-level trajectories.
        for _, u in g.groupby("unit_id", sort=False):
            u = u.sort_values("relative_day")
            ax.plot(
                u["relative_day"],
                u[value_col],
                color=color,
                alpha=raw_alpha,
                linewidth=raw_linewidth,
                marker="o",
                markersize=1.8,
                markeredgewidth=0,
                zorder=1,
            )

        s = smooth_df[(smooth_df["display_group"] == group) & (smooth_df["value_col"] == value_col)].copy()
        if not s.empty:
            s = s.sort_values("relative_day")
            ax.fill_between(
                s["relative_day"].to_numpy(dtype=float),
                s["q25"].to_numpy(dtype=float),
                s["q75"].to_numpy(dtype=float),
                color=color,
                alpha=iqr_alpha,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                s["relative_day"],
                s["summary"],
                color=color,
                linewidth=smooth_linewidth,
                zorder=3,
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        if include_zero_in_ylim or value_col.endswith("delta_from_late_pre_ms"):
            ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.65)

        ax.set_title(pretty_group_label(group).replace("\n", " "), fontsize=14, pad=4)
        ax.set_xlim(x_min, x_max)
        pretty_axes(ax, tick_label_fontsize=12)

    if title:
        fig.suptitle(title, fontsize=16, y=0.99)

    fig.supylabel(y_label, fontsize=16, x=0.055)
    axes[-1].set_xlabel("Days relative to lesion", fontsize=15)

    if common_ylim:
        if no_percentile_ylim:
            vals = unit_df[value_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                if include_zero_in_ylim:
                    lo, hi = min(lo, 0.0), max(hi, 0.0)
                span = max(hi - lo, 1.0)
                for ax in axes:
                    ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
        else:
            set_common_ylim_from_percentile(
                axes,
                unit_df[value_col].to_numpy(dtype=float),
                lower_percentile=y_lower_percentile,
                upper_percentile=y_upper_percentile,
                include_zero=include_zero_in_ylim or value_col.endswith("delta_from_late_pre_ms"),
            )

    # Legend describing raw vs summary.
    handles = [
        Line2D([0], [0], color="black", alpha=raw_alpha, linewidth=raw_linewidth, marker="o", markersize=2, label="animal daily values"),
        Line2D([0], [0], color="black", linewidth=smooth_linewidth, label="rolling median"),
        Line2D([0], [0], color="black", alpha=iqr_alpha, linewidth=8, label="rolling IQR"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="lesion day"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.005, 1.0))

    top_rect = 0.94 if title else 0.98
    fig.tight_layout(rect=[0.055, 0.02, 0.86, top_rect])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def make_period_summary(
    unit_df: pd.DataFrame,
    *,
    value_col: str = "sd_delta_from_late_pre_ms",
    late_pre_start: int = -7,
    late_pre_end: int = -1,
    early_post_start: int = 1,
    early_post_end: int = 7,
    late_post_start: int = 8,
    late_post_end: int = 30,
) -> pd.DataFrame:
    periods = [
        ("Late Pre", late_pre_start, late_pre_end),
        ("Early Post", early_post_start, early_post_end),
        ("Late Post", late_post_start, late_post_end),
    ]
    rows = []
    for (group, unit_id), u in unit_df.groupby(["display_group", "unit_id"], dropna=False):
        for label, start, end in periods:
            vals = u.loc[(u["relative_day"] >= start) & (u["relative_day"] <= end), value_col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append({
                "display_group": group,
                "unit_id": unit_id,
                "period": label,
                "period_start": start,
                "period_end": end,
                "value_col": value_col,
                "median_value": float(np.nanmedian(vals)),
                "mean_value": float(np.nanmean(vals)),
                "n_days": int(len(vals)),
            })
    return pd.DataFrame(rows)


def plot_period_summary_boxplot(
    period_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    value_col: str = "median_value",
    y_label: str = "Δ SD from Late Pre baseline (ms)",
    title: Optional[str] = "Persistence of phrase-duration variability",
    group_order: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (12, 5.8),
    dpi: int = 300,
    show: bool = False,
) -> Path:
    out_path = Path(out_path)
    if group_order is None:
        group_order = GROUP_ORDER_POOLED if POOLED_ML_GROUP in set(period_df["display_group"]) else GROUP_ORDER_SEPARATE
    group_order = [g for g in group_order if g in set(period_df["display_group"])]

    periods = ["Late Pre", "Early Post", "Late Post"]
    offsets = {"Late Pre": -0.24, "Early Post": 0.0, "Late Post": 0.24}
    width = 0.19

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(123)

    xticks = []
    xticklabels = []
    all_vals = []

    for gi, group in enumerate(group_order):
        base = gi * 1.3
        color = color_for_group(group)
        xticks.append(base)
        xticklabels.append(pretty_group_label(group))

        for period in periods:
            vals = period_df.loc[
                (period_df["display_group"] == group) & (period_df["period"] == period),
                value_col,
            ].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            all_vals.extend(vals.tolist())
            pos = base + offsets[period]
            bp = ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True, showfliers=False)

            # Period intensity: Late Pre white, post periods filled.
            for patch in bp["boxes"]:
                patch.set_edgecolor(color)
                patch.set_linewidth(1.4)
                if period == "Late Pre":
                    patch.set_facecolor("white")
                elif period == "Early Post":
                    patch.set_facecolor(color)
                    patch.set_alpha(0.30)
                else:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)

            for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
                item.set_color(color)
                item.set_linewidth(1.25)

            jitter = rng.uniform(-0.035, 0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=26,
                facecolors="white" if period == "Late Pre" else color,
                edgecolors=color,
                linewidths=0.8,
                alpha=0.85,
                zorder=3,
            )

    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_ylabel(y_label, fontsize=14)
    if title:
        ax.set_title(title, fontsize=15, pad=8)
    pretty_axes(ax, tick_label_fontsize=11)

    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Late Pre"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="black", markeredgecolor="black", alpha=0.30, markersize=8, label="Early Post"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="black", markeredgecolor="black", alpha=0.55, markersize=8, label="Late Post"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make cleaned Panel D SD time-course plots from daily phrase-duration variance CSV."
    )
    parser.add_argument("--daily-csv", required=True, help="Daily phrase-duration variance CSV.")
    parser.add_argument("--metadata-excel", default=None, help="Area X lesion metadata Excel file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--meta-sheet-name", default="animal_hit_type_summary")
    parser.add_argument("--meta-animal-col", default="Animal ID")
    parser.add_argument("--meta-hit-type-col", default="Lesion hit type")
    parser.add_argument("--x-min", type=int, default=-30)
    parser.add_argument("--x-max", type=int, default=30)

    parser.add_argument("--unit-level", choices=["animal", "animal_syllable"], default="animal",
                        help="Use animal for manuscript-style biological units; animal_syllable for more raw spaghetti.")
    parser.add_argument("--within-unit-stat", choices=["median", "mean"], default="median",
                        help="How to collapse syllables within each animal/day when unit-level=animal.")
    parser.add_argument("--no-pool-medial-lateral", action="store_true",
                        help="Keep partial and complete medial/lateral groups separate instead of pooling them.")

    parser.add_argument("--smooth-window-days", type=int, default=7,
                        help="Centered rolling window size in days. Use 5 or 7. Default: 7.")
    parser.add_argument("--smooth-stat", choices=["median", "mean"], default="median",
                        help="Group summary statistic for the thick rolling line.")
    parser.add_argument("--allow-cross-lesion-smoothing", action="store_true",
                        help="If set, smoothing can borrow data across lesion day. Default prevents this.")

    parser.add_argument("--baseline-start", type=int, default=-7)
    parser.add_argument("--baseline-end", type=int, default=-1)
    parser.add_argument("--early-post-start", type=int, default=1)
    parser.add_argument("--early-post-end", type=int, default=7)
    parser.add_argument("--late-post-start", type=int, default=8)
    parser.add_argument("--late-post-end", type=int, default=30)

    parser.add_argument("--y-lower-percentile", type=float, default=1.0)
    parser.add_argument("--y-upper-percentile", type=float, default=99.0)
    parser.add_argument("--no-percentile-y-limit", action="store_true",
                        help="Use full min/max y-limits instead of percentile-based limits.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_daily_table(
        args.daily_csv,
        metadata_excel=args.metadata_excel,
        meta_sheet_name=args.meta_sheet_name,
        meta_animal_col=args.meta_animal_col,
        meta_hit_type_col=args.meta_hit_type_col,
        x_min=args.x_min,
        x_max=args.x_max,
        pool_medial_lateral=not args.no_pool_medial_lateral,
    )
    raw_df.to_csv(out_dir / "panel_D_clean_input_with_sd.csv", index=False)

    unit_df = make_daily_unit_table(
        raw_df,
        unit_level=args.unit_level,
        within_unit_stat=args.within_unit_stat,
    )
    unit_df = add_baseline_columns(
        unit_df,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )
    unit_df.to_csv(out_dir / "panel_D_clean_unit_daily_sd.csv", index=False)

    group_order = GROUP_ORDER_SEPARATE if args.no_pool_medial_lateral else GROUP_ORDER_POOLED

    # Raw SD in milliseconds, with rolling median/IQR.
    smooth_raw = rolling_window_summary(
        unit_df,
        value_col="sd_ms",
        window_days=args.smooth_window_days,
        summary_stat=args.smooth_stat,
        no_cross_lesion_smoothing=not args.allow_cross_lesion_smoothing,
    )
    smooth_raw.to_csv(out_dir / "panel_D_clean_rolling_sd_ms_summary.csv", index=False)

    plot_clean_timecourse_panels(
        unit_df,
        smooth_raw,
        out_dir / "panel_D_clean_sd_timecourse_raw_ms.png",
        value_col="sd_ms",
        y_label="Standard deviation of phrase durations (ms)",
        title=f"Phrase-duration SD over time ({args.smooth_window_days}-day rolling {args.smooth_stat})",
        group_order=group_order,
        x_min=args.x_min,
        x_max=args.x_max,
        y_lower_percentile=args.y_lower_percentile,
        y_upper_percentile=args.y_upper_percentile,
        no_percentile_ylim=args.no_percentile_y_limit,
        include_zero_in_ylim=False,
        dpi=args.dpi,
        show=args.show,
    )

    # Baseline-normalized delta SD in milliseconds.
    smooth_delta = rolling_window_summary(
        unit_df,
        value_col="sd_delta_from_late_pre_ms",
        window_days=args.smooth_window_days,
        summary_stat=args.smooth_stat,
        no_cross_lesion_smoothing=not args.allow_cross_lesion_smoothing,
    )
    smooth_delta.to_csv(out_dir / "panel_D_clean_rolling_delta_sd_summary.csv", index=False)

    plot_clean_timecourse_panels(
        unit_df,
        smooth_delta,
        out_dir / "panel_D_clean_sd_timecourse_delta_from_late_pre.png",
        value_col="sd_delta_from_late_pre_ms",
        y_label=f"Δ SD from Late Pre baseline (ms)",
        title=f"Baseline-normalized phrase-duration SD ({args.smooth_window_days}-day rolling {args.smooth_stat})",
        group_order=group_order,
        x_min=args.x_min,
        x_max=args.x_max,
        y_lower_percentile=args.y_lower_percentile,
        y_upper_percentile=args.y_upper_percentile,
        no_percentile_ylim=args.no_percentile_y_limit,
        include_zero_in_ylim=True,
        dpi=args.dpi,
        show=args.show,
    )

    # Persistence summary: Late Pre vs Early Post vs Late Post.
    period_df = make_period_summary(
        unit_df,
        value_col="sd_delta_from_late_pre_ms",
        late_pre_start=args.baseline_start,
        late_pre_end=args.baseline_end,
        early_post_start=args.early_post_start,
        early_post_end=args.early_post_end,
        late_post_start=args.late_post_start,
        late_post_end=args.late_post_end,
    )
    period_df.to_csv(out_dir / "panel_D_clean_period_summary_delta_sd.csv", index=False)

    plot_period_summary_boxplot(
        period_df,
        out_dir / "panel_D_clean_sd_period_summary_boxplot.png",
        value_col="median_value",
        y_label="Median Δ SD from Late Pre baseline (ms)",
        title="Persistence of phrase-duration variability",
        group_order=group_order,
        dpi=args.dpi,
        show=args.show,
    )

    print("[DONE] Saved cleaned Panel D outputs to:")
    print(out_dir)
    print("[KEY OUTPUTS]")
    print(out_dir / "panel_D_clean_sd_timecourse_raw_ms.png")
    print(out_dir / "panel_D_clean_sd_timecourse_delta_from_late_pre.png")
    print(out_dir / "panel_D_clean_sd_period_summary_boxplot.png")


if __name__ == "__main__":
    main()
