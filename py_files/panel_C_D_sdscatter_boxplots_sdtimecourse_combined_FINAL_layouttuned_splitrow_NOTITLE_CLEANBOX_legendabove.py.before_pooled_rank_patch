#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panel_C_D_variance_plots_with_json_colors_tight_axes_sd_stats_cb_palette_boxplots.py

Make both figure components from the same lesion-group color JSON.

Panel C
-------
Pre/post variance scatterplot:
    x = Late Pre variance
    y = Post variance
    each point = one animal × syllable

Additional Panel C SD plot
--------------------------
Pre/post standard deviation scatterplot:
    x = sqrt(Late Pre variance)
    y = sqrt(Post variance)
    each point = one animal × syllable

Panel D
-------
Variance-over-time plot:
    x = days relative to lesion
    y = variance of phrase durations
    each line = one animal × syllable trajectory

Edits in this version
---------------------
1. Panel C scatterplots do not include an internal legend by default, because a
   separate legend file is already saved.
2. Panel C axes are tightened to the actual min/max of the plotted data rather
   than rounded to broad powers of 10.
3. A separate Late Pre vs Post standard deviation scatterplot is saved in
   addition to the variance scatterplot.
4. Panel C statistics are saved automatically when --scatter-csv is provided:
      - animal × syllable change metrics
      - animal-level change metrics
      - paired pre/post tests within each lesion group
      - one-tailed Welch's t-tests comparing each lesion group's change metric
        against sham controls
5. Additional Panel C boxplots are saved automatically:
      - Late Pre vs Post variance distributions
      - Late Pre vs Post standard deviation distributions
      - change in variance versus sham controls
      - change in standard deviation versus sham controls

Statistical design
------------------
By default, significance tests are run at the animal level to avoid treating
multiple syllables from the same bird as independent biological replicates.
For each animal, the script averages the selected animal × syllable rows and
then tests pre/post changes across animals.

Default tests:
    1. Within-group paired t-tests:
          post metric > pre metric
       for variance and standard deviation.

    2. Welch's t-tests versus sham controls:
          lesion-group change metric > sham change metric
       for variance_delta, sd_delta, variance_log2_ratio, and sd_log2_ratio.

The pooled medial+lateral group is created automatically by combining:
    Complete Medial and Lateral lesion
    Partial Medial and Lateral lesion

Use --stats-level animal_syllable only if you intentionally want each
animal × syllable row treated as an independent unit.

Expected Panel C input CSV
--------------------------
Usually usage_balanced_phrase_duration_stats.csv, with columns:
    Animal ID
    Syllable
    Group                 e.g., Early Pre, Late Pre, Post
    N_phrases             optional but recommended
    Variance_ms2

Expected Panel D input CSV
--------------------------
Usually batch_aligned_phrase_duration_variance_top30.csv, with columns:
    animal_id
    syllable
    relative_day
    Variance (ms^2)
    lesion_group

Optional metadata Excel
-----------------------
Use metadata to map each animal to lesion hit type. This is especially useful
for splitting the combined medial/lateral panel into:
    Partial Medial and Lateral lesion
    Complete Medial and Lateral lesion

Example terminal usage
----------------------
python panel_C_D_variance_plots_with_json_colors_tight_axes_sd_stats_cb_palette_boxplots.py \
  --scatter-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv" \
  --daily-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/top30_phrase_variance_BC_only_v4/batch_aligned_phrase_duration_variance_top30.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --color-json "/Volumes/my_own_SSD/updated_AreaX_outputs/areax_lesion_group_colors_v1.json" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/panel_C_D_variance_outputs" \
  --scatter-top-percentile 70 \
  --rank-on post \
  --y-scale symlog \
  --x-min -30 \
  --x-max 30 \
  --no-show
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - used only if scipy is unavailable
    scipy_stats = None

PathLike = Union[str, Path]


# ──────────────────────────────────────────────────────────────────────────────
# Default colors, used only if no JSON file is supplied.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_COLOR_CONFIG: Dict[str, Any] = {
    "display_order": [
        "Complete Medial and Lateral lesion",
        "Partial Medial and Lateral lesion",
        "Lateral lesion only",
        "sham saline injection",
    ],
    "groups": {
        "Complete Medial and Lateral lesion": {
            "color": "#3F007D",
            "aliases": [
                "large lesion Area X not visible",
                "Area X not visible",
                "complete medial and lateral lesion",
            ],
        },
        "Partial Medial and Lateral lesion": {
            "color": "#7A4FB7",
            "aliases": [
                "Area X visible (medial+lateral hit)",
                "medial+lateral hit",
                "m+l hit",
            ],
        },
        "Lateral lesion only": {
            "color": "#A88BD9",
            "aliases": [
                "lateral hit only",
                "Area X visible (single hit)",
                "single hit",
            ],
        },
        "sham saline injection": {
            "color": "#1B9E77",
            "aliases": ["sham saline injection", "sham", "saline"],
        },
        "unknown": {
            "color": "#4D4D4D",
            "aliases": ["unknown", "nan", "none", ""],
        },
    },
    "panel_d_panel_order": [
        {"title": "sham saline injection", "display_groups": ["sham saline injection"]},
        {"title": "Lateral lesion only", "display_groups": ["Lateral lesion only"]},
        {
            "title": "Medial and Lateral lesions",
            "display_groups": [
                "Partial Medial and Lateral lesion",
                "Complete Medial and Lateral lesion",
            ],
        },
    ],
}


SHAM_GROUP = "sham saline injection"
COMPLETE_ML_GROUP = "Complete Medial and Lateral lesion"
PARTIAL_ML_GROUP = "Partial Medial and Lateral lesion"
LATERAL_ONLY_GROUP = "Lateral lesion only"
POOLED_ML_GROUP = "Complete and partial medial and lateral lesion"
ML_GROUPS = [COMPLETE_ML_GROUP, PARTIAL_ML_GROUP]


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_sheet_name(sheet_name: Union[str, int]) -> Union[str, int]:
    if isinstance(sheet_name, str) and sheet_name.strip().isdigit():
        return int(sheet_name.strip())
    return sheet_name


def _standardize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def _lower(x: Any) -> str:
    return _standardize_text(x).lower()


def _load_json_config(color_json: Optional[PathLike]) -> Dict[str, Any]:
    if color_json is None:
        return DEFAULT_COLOR_CONFIG

    p = Path(color_json)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "groups" not in cfg:
        raise ValueError(f"Color JSON must contain a top-level 'groups' dictionary: {p}")
    if "display_order" not in cfg:
        cfg["display_order"] = list(cfg["groups"].keys())
    if "panel_d_panel_order" not in cfg:
        cfg["panel_d_panel_order"] = DEFAULT_COLOR_CONFIG["panel_d_panel_order"]
    return cfg


def _build_alias_lookup(cfg: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for display_name, info in cfg.get("groups", {}).items():
        lookup[_lower(display_name)] = display_name
        for alias in info.get("aliases", []):
            lookup[_lower(alias)] = display_name
    return lookup


def _color_map_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for display_name, info in cfg.get("groups", {}).items():
        out[display_name] = str(info.get("color", "#4D4D4D"))
    return out


def _canonical_display_group(raw: Any, cfg: Dict[str, Any], *, fallback: str = "unknown") -> str:
    """Map raw metadata / lesion group strings to the display labels in the JSON."""
    s = _lower(raw)
    if s == "":
        return fallback

    alias_lookup = _build_alias_lookup(cfg)
    if s in alias_lookup:
        return alias_lookup[s]

    # Heuristic fallbacks for older labels.
    if "sham" in s or ("saline" in s and "lesion" not in s):
        return "sham saline injection"

    if "large" in s and "lesion" in s and "not visible" in s:
        return "Complete Medial and Lateral lesion"
    if "area x" in s and "not visible" in s:
        return "Complete Medial and Lateral lesion"
    if "complete" in s and "medial" in s and "lateral" in s:
        return "Complete Medial and Lateral lesion"

    if "single" in s or "lateral only" in s or "lateral hit only" in s:
        return "Lateral lesion only"

    if "medial" in s and "lateral" in s:
        return "Partial Medial and Lateral lesion"
    if "m+l" in s or "combined large lesion and m+l" in s:
        # If we only know that this was in the combined M+L group, but do not
        # have metadata to split partial vs complete, use the partial label as
        # the conservative fallback. Provide metadata to split it correctly.
        return "Partial Medial and Lateral lesion"

    return fallback


def _read_excel_best_sheet(
    metadata_excel: PathLike,
    preferred_sheet: Union[int, str],
    *,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    p = Path(metadata_excel)
    preferred_sheet = _safe_sheet_name(preferred_sheet)

    try:
        df = pd.read_excel(p, sheet_name=preferred_sheet)
        if all(c in df.columns for c in required_cols):
            return df
    except Exception:
        pass

    xls = pd.ExcelFile(p)
    for sheet in xls.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet)
        if all(c in df.columns for c in required_cols):
            print(f"[INFO] Using metadata sheet {sheet!r} because it has {list(required_cols)}")
            return df

    raise ValueError(
        f"Could not find a metadata sheet in {p} with required columns: {list(required_cols)}"
    )


def _load_hit_type_map(
    metadata_excel: Optional[PathLike],
    *,
    sheet_name: Union[int, str],
    animal_col: str = "Animal ID",
    hit_type_col: Optional[str] = "Lesion hit type",
) -> Dict[str, str]:
    if metadata_excel is None:
        return {}

    p = Path(metadata_excel)
    if not p.exists():
        raise FileNotFoundError(f"metadata_excel not found: {p}")

    required = [animal_col]
    df = _read_excel_best_sheet(p, sheet_name, required_cols=required)

    if hit_type_col is None or hit_type_col not in df.columns:
        candidates = [
            "Lesion hit type",
            "Hit type",
            "Hit Type",
            "hit_type",
            "Category",
            "category",
        ]
        hit_type_col = next((c for c in candidates if c in df.columns), None)

    if hit_type_col is None or hit_type_col not in df.columns:
        raise ValueError(
            f"Could not find hit-type column in metadata. Available columns: {list(df.columns)}"
        )

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        aid = row.get(animal_col)
        hit = row.get(hit_type_col)
        if pd.isna(aid) or pd.isna(hit):
            continue
        out[str(aid).strip()] = str(hit).strip()
    return out


def _pretty_axes(ax: plt.Axes, tick_label_fontsize: float = 14) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def _tight_positive_limits(values: np.ndarray, *, padding_frac: float = 0.0) -> tuple[float, float]:
    """Return tight positive limits for log-scaled data."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        raise ValueError("Cannot make log-scaled scatterplot because no positive values were found.")

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    if vmin == vmax:
        vmin *= 0.95
        vmax *= 1.05

    if padding_frac > 0:
        # Padding is multiplicative in log space so the margin is visually
        # symmetric on log-scaled axes.
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)
        span = max(log_max - log_min, 1e-9)
        log_min -= padding_frac * span
        log_max += padding_frac * span
        vmin = float(10 ** log_min)
        vmax = float(10 ** log_max)

    return vmin, vmax


# ──────────────────────────────────────────────────────────────────────────────
# Panel C: pre/post variance and SD scatterplots
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_scatter_table(
    csv_path: PathLike,
    *,
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    variance_col: str = "Variance_ms2",
    group_col: str = "Group",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    nphrases_col: str = "N_phrases",
    min_n_phrases: int = 5,
    top_percentile: Optional[float] = 70.0,
    rank_on: str = "post",
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = "metadata_with_hit_type",
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = "Lesion hit type",
    color_config: Dict[str, Any] = DEFAULT_COLOR_CONFIG,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [animal_col, syllable_col, group_col, variance_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Scatter CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for pre_group={pre_group!r} and post_group={post_group!r}."
        )

    df[variance_col] = pd.to_numeric(df[variance_col], errors="coerce")
    if nphrases_col in df.columns:
        df[nphrases_col] = pd.to_numeric(df[nphrases_col], errors="coerce")
        df = df[df[nphrases_col].fillna(0) >= min_n_phrases].copy()

    # Average if there are repeated rows for the same animal × syllable × group.
    long = (
        df.groupby([animal_col, syllable_col, group_col], dropna=False)[variance_col]
        .mean()
        .reset_index()
    )

    wide = long.pivot_table(
        index=[animal_col, syllable_col],
        columns=group_col,
        values=variance_col,
        aggfunc="mean",
    ).reset_index()

    if pre_group not in wide.columns or post_group not in wide.columns:
        raise ValueError(
            f"After pivoting, could not find both {pre_group!r} and {post_group!r} columns."
        )

    wide = wide.rename(columns={pre_group: "pre_variance", post_group: "post_variance"})
    wide["pre_variance"] = pd.to_numeric(wide["pre_variance"], errors="coerce")
    wide["post_variance"] = pd.to_numeric(wide["post_variance"], errors="coerce")
    wide = wide.dropna(subset=["pre_variance", "post_variance"])

    # Log-scaled scatterplots require positive values.
    wide = wide[(wide["pre_variance"] > 0) & (wide["post_variance"] > 0)].copy()

    # Add standard deviation columns for the separate SD scatterplot.
    wide["pre_sd"] = np.sqrt(wide["pre_variance"])
    wide["post_sd"] = np.sqrt(wide["post_variance"])

    # Add per-animal × syllable change metrics for downstream statistics.
    # Delta is Post - Late Pre. Log2 ratio is log2(Post / Late Pre).
    wide["variance_delta"] = wide["post_variance"] - wide["pre_variance"]
    wide["variance_ratio"] = wide["post_variance"] / wide["pre_variance"]
    wide["variance_log2_ratio"] = np.log2(wide["variance_ratio"])
    wide["sd_delta"] = wide["post_sd"] - wide["pre_sd"]
    wide["sd_ratio"] = wide["post_sd"] / wide["pre_sd"]
    wide["sd_log2_ratio"] = np.log2(wide["sd_ratio"])

    if top_percentile is not None:
        top_percentile = float(top_percentile)
        rank_on = str(rank_on).lower()
        if rank_on == "pre":
            wide["rank_metric"] = wide["pre_variance"]
        elif rank_on == "post":
            wide["rank_metric"] = wide["post_variance"]
        elif rank_on == "max":
            wide["rank_metric"] = wide[["pre_variance", "post_variance"]].max(axis=1)
        else:
            raise ValueError("rank_on must be one of: pre, post, max")

        keep_parts = []
        for _, g in wide.groupby(animal_col, dropna=False):
            vals = g["rank_metric"].dropna()
            if vals.empty:
                continue
            threshold = np.nanpercentile(vals.to_numpy(dtype=float), top_percentile)
            keep_parts.append(g[g["rank_metric"] >= threshold].copy())
        wide = pd.concat(keep_parts, ignore_index=True) if keep_parts else wide.iloc[0:0].copy()

    hit_type_map = (
        _load_hit_type_map(
            metadata_excel,
            sheet_name=meta_sheet_name,
            animal_col=meta_animal_col,
            hit_type_col=meta_hit_type_col,
        )
        if metadata_excel is not None
        else {}
    )

    wide["raw_hit_type"] = wide[animal_col].astype(str).map(hit_type_map)
    wide["display_group"] = wide["raw_hit_type"].apply(
        lambda x: _canonical_display_group(x, color_config, fallback="unknown")
    )

    return wide.reset_index(drop=True)


def plot_panel_c_scatter(
    scatter_df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    x_col: str = "pre_variance",
    y_col: str = "post_variance",
    x_label: str = "Late Pre variance (ms²)",
    y_label: str = "Post variance (ms²)",
    figsize: tuple[float, float] = (7.2, 6.2),
    marker_size: float = 30.0,
    alpha: float = 0.78,
    diagonal_color: str = "red",
    axis_padding_frac: float = 0.045,
    x_label_fontsize: float = 18,
    y_label_fontsize: float = 18,
    tick_label_fontsize: float = 14,
    show_legend: bool = False,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """Plot a log-log Late Pre vs Post scatterplot.

    The same function is used for variance and SD by changing x_col/y_col and
    the axis labels.
    """
    out_path = Path(out_path)
    color_map = _color_map_from_config(color_config)
    display_order = list(color_config.get("display_order", color_map.keys()))

    if x_col not in scatter_df.columns or y_col not in scatter_df.columns:
        raise ValueError(
            f"scatter_df must contain {x_col!r} and {y_col!r}. Found: {list(scatter_df.columns)}"
        )

    finite = scatter_df[[x_col, y_col, "display_group"]].replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite[(finite[x_col] > 0) & (finite[y_col] > 0)].copy()
    if finite.empty:
        raise ValueError(f"No finite positive rows to plot using {x_col!r} and {y_col!r}.")

    effective_padding = max(float(axis_padding_frac), 0.045)
    x_min, x_max = _tight_positive_limits(finite[x_col].to_numpy(dtype=float), padding_frac=effective_padding)
    y_min, y_max = _tight_positive_limits(finite[y_col].to_numpy(dtype=float), padding_frac=effective_padding)

    fig, ax = plt.subplots(figsize=figsize)

    # Identity line. It spans the full visible range and is clipped to the axes.
    line_min = min(x_min, y_min)
    line_max = max(x_max, y_max)
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color=diagonal_color,
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        zorder=0,
    )

    # Plot least prominent first and most prominent last, while keeping same marker size.
    present_groups = set(finite["display_group"].astype(str))
    plot_order = [g for g in display_order[::-1] if g in present_groups]
    if "unknown" in present_groups and "unknown" not in plot_order:
        plot_order.insert(0, "unknown")

    for group in plot_order:
        g = finite[finite["display_group"].astype(str) == group]
        if g.empty:
            continue
        ax.scatter(
            g[x_col],
            g[y_col],
            s=marker_size,
            c=color_map.get(group, "#4D4D4D"),
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            label=group,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label, fontsize=x_label_fontsize)
    ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    if show_legend:
        legend_groups = [g for g in display_order if g in present_groups]
        if "unknown" in present_groups:
            legend_groups.append("unknown")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color_map.get(group, "#4D4D4D"),
                markeredgecolor="none",
                markersize=7,
                label=group,
                alpha=alpha,
            )
            for group in legend_groups
        ]
        ax.legend(handles=handles, frameon=False, fontsize=10.5, loc="lower right")

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.98)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Panel C statistics
# ──────────────────────────────────────────────────────────────────────────────
def _require_scipy_for_stats() -> None:
    if scipy_stats is None:
        raise ImportError(
            "scipy is required for statistical tests. Install it with `pip install scipy` "
            "or `conda install scipy`."
        )


def _finite_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _p_from_two_sided_t(t_stat: float, p_two_sided: float, alternative: str) -> float:
    """Convert a two-sided t-test p-value to the requested one-sided direction."""
    if not np.isfinite(t_stat) or not np.isfinite(p_two_sided):
        return np.nan

    alternative = str(alternative).lower()
    if alternative == "two-sided":
        return float(p_two_sided)
    if alternative == "greater":
        return float(p_two_sided / 2.0) if t_stat >= 0 else float(1.0 - p_two_sided / 2.0)
    if alternative == "less":
        return float(p_two_sided / 2.0) if t_stat <= 0 else float(1.0 - p_two_sided / 2.0)
    raise ValueError("alternative must be one of: greater, less, two-sided")


def _welch_df(x: np.ndarray, y: np.ndarray) -> float:
    """Welch-Satterthwaite approximate degrees of freedom."""
    x = _finite_1d(x)
    y = _finite_1d(y)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    if vx == 0 and vy == 0:
        return np.nan
    sx = vx / nx
    sy = vy / ny
    denom = (sx * sx) / (nx - 1) + (sy * sy) / (ny - 1)
    if denom <= 0:
        return np.nan
    return float((sx + sy) ** 2 / denom)


def _safe_mean(x: np.ndarray) -> float:
    x = _finite_1d(x)
    return float(np.mean(x)) if x.size else np.nan


def _safe_sd(x: np.ndarray) -> float:
    x = _finite_1d(x)
    return float(np.std(x, ddof=1)) if x.size >= 2 else np.nan


def _safe_sem(x: np.ndarray) -> float:
    x = _finite_1d(x)
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size >= 2 else np.nan


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaN values remain NaN."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    idx = np.where(finite)[0]
    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = ranked.size

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out[idx[order]] = adjusted
    return out


def _add_pooled_ml_rows(df: pd.DataFrame, *, group_col: str = "display_group") -> pd.DataFrame:
    """Append duplicate rows representing complete+partial medial/lateral pooled."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r} for pooled medial/lateral rows.")
    pooled = df[df[group_col].isin(ML_GROUPS)].copy()
    if pooled.empty:
        return df.copy()
    pooled[group_col] = POOLED_ML_GROUP
    return pd.concat([df.copy(), pooled], ignore_index=True)


def _ordered_stats_groups(groups_present: Iterable[str]) -> list[str]:
    preferred = [
        SHAM_GROUP,
        LATERAL_ONLY_GROUP,
        PARTIAL_ML_GROUP,
        COMPLETE_ML_GROUP,
        POOLED_ML_GROUP,
        "unknown",
    ]
    present = list(dict.fromkeys(str(g) for g in groups_present if str(g) != "nan"))
    ordered = [g for g in preferred if g in present]
    ordered += [g for g in present if g not in ordered]
    return ordered


def _build_animal_level_change_table(
    scatter_df: pd.DataFrame,
    *,
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
) -> pd.DataFrame:
    """Average selected animal × syllable rows within each animal for stats."""
    required = [
        animal_col,
        syllable_col,
        "display_group",
        "pre_variance",
        "post_variance",
        "pre_sd",
        "post_sd",
        "variance_delta",
        "variance_ratio",
        "variance_log2_ratio",
        "sd_delta",
        "sd_ratio",
        "sd_log2_ratio",
    ]
    missing = [c for c in required if c not in scatter_df.columns]
    if missing:
        raise ValueError(f"Cannot build animal-level stats table; missing columns: {missing}")

    def _first_nonempty(x: pd.Series) -> str:
        vals = [v for v in x.astype(str).tolist() if v and v != "nan"]
        return vals[0] if vals else "unknown"

    agg = (
        scatter_df.groupby([animal_col], dropna=False)
        .agg(
            display_group=("display_group", _first_nonempty),
            n_syllables=(syllable_col, "nunique"),
            pre_variance=("pre_variance", "mean"),
            post_variance=("post_variance", "mean"),
            pre_sd=("pre_sd", "mean"),
            post_sd=("post_sd", "mean"),
            variance_delta=("variance_delta", "mean"),
            variance_ratio=("variance_ratio", "mean"),
            variance_log2_ratio=("variance_log2_ratio", "mean"),
            sd_delta=("sd_delta", "mean"),
            sd_ratio=("sd_ratio", "mean"),
            sd_log2_ratio=("sd_log2_ratio", "mean"),
        )
        .reset_index()
    )
    return agg


def _paired_ttest_summary(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    alternative: str = "greater",
) -> dict[str, float]:
    """Paired t-test summary for post vs pre."""
    _require_scipy_for_stats()
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    n = int(pre.size)

    if n < 2:
        return {
            "n_units": n,
            "pre_mean": _safe_mean(pre),
            "post_mean": _safe_mean(post),
            "mean_change_post_minus_pre": _safe_mean(post - pre) if n else np.nan,
            "sd_change_post_minus_pre": _safe_sd(post - pre) if n else np.nan,
            "sem_change_post_minus_pre": _safe_sem(post - pre) if n else np.nan,
            "t_stat": np.nan,
            "df": np.nan,
            "p_two_sided": np.nan,
            "p_value": np.nan,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = scipy_stats.ttest_rel(post, pre, nan_policy="omit")
    t_stat = float(res.statistic)
    p_two = float(res.pvalue)
    p_val = _p_from_two_sided_t(t_stat, p_two, alternative)
    change = post - pre

    return {
        "n_units": n,
        "pre_mean": _safe_mean(pre),
        "post_mean": _safe_mean(post),
        "mean_change_post_minus_pre": _safe_mean(change),
        "sd_change_post_minus_pre": _safe_sd(change),
        "sem_change_post_minus_pre": _safe_sem(change),
        "t_stat": t_stat,
        "df": float(n - 1),
        "p_two_sided": p_two,
        "p_value": p_val,
    }


def _welch_ttest_summary(
    group_values: np.ndarray,
    sham_values: np.ndarray,
    *,
    alternative: str = "greater",
) -> dict[str, float]:
    """Welch t-test summary for group change metric vs sham change metric."""
    _require_scipy_for_stats()
    x = _finite_1d(group_values)
    y = _finite_1d(sham_values)
    nx = int(x.size)
    ny = int(y.size)

    base = {
        "n_group": nx,
        "n_sham": ny,
        "group_mean": _safe_mean(x),
        "sham_mean": _safe_mean(y),
        "mean_difference_group_minus_sham": _safe_mean(x) - _safe_mean(y),
        "group_sd": _safe_sd(x),
        "sham_sd": _safe_sd(y),
        "t_stat": np.nan,
        "df": np.nan,
        "p_two_sided": np.nan,
        "p_value": np.nan,
    }

    if nx < 2 or ny < 2:
        return base

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = scipy_stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    t_stat = float(res.statistic)
    p_two = float(res.pvalue)
    p_val = _p_from_two_sided_t(t_stat, p_two, alternative)
    base.update(
        {
            "t_stat": t_stat,
            "df": _welch_df(x, y),
            "p_two_sided": p_two,
            "p_value": p_val,
        }
    )
    return base


def run_panel_c_stats(
    scatter_df: pd.DataFrame,
    out_dir: PathLike,
    *,
    stats_level: str = "animal",
    alternative: str = "greater",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    change_metrics: Sequence[str] = (
        "variance_delta",
        "sd_delta",
        "variance_log2_ratio",
        "sd_log2_ratio",
    ),
    base_metrics: Sequence[str] = ("variance", "sd"),
) -> Dict[str, Path]:
    """Save Panel C change metrics and statistical tests.

    Parameters
    ----------
    scatter_df:
        Output of _prepare_scatter_table(). Each row is one animal × syllable.
    stats_level:
        "animal" averages selected syllables within each bird before testing
        and is the default to avoid pseudoreplication.
        "animal_syllable" treats every animal × syllable row as independent.
    alternative:
        Direction for one-tailed p-values. The default "greater" tests:
            paired tests: post > pre
            Welch tests: lesion group change > sham change
        Use "two-sided" for two-sided p-values.
    """
    out_dir = _ensure_dir(Path(out_dir))
    alternative = str(alternative).lower()
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("stats alternative must be one of: greater, less, two-sided")
    stats_level = str(stats_level).lower()
    if stats_level not in {"animal", "animal_syllable"}:
        raise ValueError("stats_level must be one of: animal, animal_syllable")

    outputs: Dict[str, Path] = {}

    # Save the per-animal × syllable metric table that all plots and stats are derived from.
    animal_syllable_path = out_dir / "panel_C_change_metrics_animal_syllable.csv"
    scatter_df.to_csv(animal_syllable_path, index=False)
    outputs["panel_C_change_metrics_animal_syllable"] = animal_syllable_path

    # Default biological replicate table: one row per animal.
    animal_df = _build_animal_level_change_table(
        scatter_df,
        animal_col=animal_col,
        syllable_col=syllable_col,
    )
    animal_path = out_dir / "panel_C_change_metrics_animal_level.csv"
    animal_df.to_csv(animal_path, index=False)
    outputs["panel_C_change_metrics_animal_level"] = animal_path

    if stats_level == "animal":
        stats_df = animal_df.copy()
        stats_unit_col = animal_col
        stats_table_path = animal_path
    else:
        stats_df = scatter_df.copy()
        stats_unit_col = f"{animal_col} × {syllable_col}"
        stats_table_path = animal_syllable_path

    stats_df_with_pooled = _add_pooled_ml_rows(stats_df, group_col="display_group")
    groups = _ordered_stats_groups(stats_df_with_pooled["display_group"].dropna().astype(str).unique())

    # 1) Within-group paired pre/post tests.
    paired_rows: list[dict[str, Any]] = []
    for metric in base_metrics:
        metric = str(metric).lower()
        if metric not in {"variance", "sd"}:
            raise ValueError("base_metrics must contain only 'variance' and/or 'sd'")
        pre_col = f"pre_{metric}"
        post_col = f"post_{metric}"
        if pre_col not in stats_df_with_pooled.columns or post_col not in stats_df_with_pooled.columns:
            continue

        for group in groups:
            if group == "unknown":
                # Keep unknown in metric tables, but skip formal group tests.
                continue
            g = stats_df_with_pooled[stats_df_with_pooled["display_group"].astype(str) == group]
            summary = _paired_ttest_summary(g[pre_col].to_numpy(), g[post_col].to_numpy(), alternative=alternative)
            paired_rows.append(
                {
                    "test_family": "within_group_pre_post",
                    "method": "paired t-test",
                    "stats_level": stats_level,
                    "stats_unit": stats_unit_col,
                    "group": group,
                    "base_metric": metric,
                    "pre_col": pre_col,
                    "post_col": post_col,
                    "alternative": "post > pre" if alternative == "greater" else ("post < pre" if alternative == "less" else "two-sided"),
                    "source_table": str(stats_table_path),
                    **summary,
                }
            )

    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df["p_bh_fdr"] = _bh_fdr(paired_df["p_value"].to_numpy(dtype=float))
    paired_path = out_dir / f"panel_C_within_group_pre_post_paired_tests_{stats_level}_level.csv"
    paired_df.to_csv(paired_path, index=False)
    outputs["panel_C_within_group_pre_post_tests"] = paired_path

    # 2) Welch tests of lesion-group change metrics relative to sham controls.
    welch_rows: list[dict[str, Any]] = []
    if SHAM_GROUP in set(stats_df_with_pooled["display_group"].astype(str)):
        sham_df = stats_df_with_pooled[stats_df_with_pooled["display_group"].astype(str) == SHAM_GROUP]
        comparison_groups = [
            LATERAL_ONLY_GROUP,
            PARTIAL_ML_GROUP,
            COMPLETE_ML_GROUP,
            POOLED_ML_GROUP,
        ]
        comparison_groups += [
            g
            for g in groups
            if g not in comparison_groups and g not in {SHAM_GROUP, "unknown"}
        ]

        for metric in change_metrics:
            metric = str(metric)
            if metric not in stats_df_with_pooled.columns:
                print(f"[WARN] Skipping missing change metric column: {metric}")
                continue
            sham_values = sham_df[metric].to_numpy(dtype=float)
            for group in comparison_groups:
                g = stats_df_with_pooled[stats_df_with_pooled["display_group"].astype(str) == group]
                if g.empty:
                    continue
                summary = _welch_ttest_summary(g[metric].to_numpy(dtype=float), sham_values, alternative=alternative)
                welch_rows.append(
                    {
                        "test_family": "change_metric_vs_sham",
                        "method": "Welch independent-samples t-test",
                        "stats_level": stats_level,
                        "stats_unit": stats_unit_col,
                        "group": group,
                        "reference_group": SHAM_GROUP,
                        "change_metric": metric,
                        "alternative": f"{group} > {SHAM_GROUP}" if alternative == "greater" else (f"{group} < {SHAM_GROUP}" if alternative == "less" else "two-sided"),
                        "source_table": str(stats_table_path),
                        **summary,
                    }
                )
    else:
        print(f"[WARN] Could not run Welch tests versus sham because {SHAM_GROUP!r} is absent.")

    welch_df = pd.DataFrame(welch_rows)
    if not welch_df.empty:
        welch_df["p_bh_fdr"] = _bh_fdr(welch_df["p_value"].to_numpy(dtype=float))
    welch_path = out_dir / f"panel_C_change_metric_vs_sham_welch_tests_{stats_level}_level.csv"
    welch_df.to_csv(welch_path, index=False)
    outputs["panel_C_change_metric_vs_sham_welch_tests"] = welch_path

    # 3) Human-readable plain-text summary.
    summary_path = out_dir / f"panel_C_stats_summary_{stats_level}_level.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Panel C statistics summary\n")
        f.write("==========================\n\n")
        f.write(f"Stats level: {stats_level}\n")
        f.write(f"Stats unit: {stats_unit_col}\n")
        f.write(f"Alternative: {alternative}\n")
        f.write(f"Source table used for tests: {stats_table_path}\n\n")
        f.write("Important design note:\n")
        f.write(
            "The default animal-level tests first average the selected syllables within each animal, "
            "then use animals as the statistical units. This avoids treating multiple syllables from "
            "the same bird as independent biological replicates.\n\n"
        )

        f.write("Within-group pre/post paired tests\n")
        f.write("----------------------------------\n")
        if paired_df.empty:
            f.write("No paired tests were run.\n")
        else:
            for _, row in paired_df.iterrows():
                f.write(
                    f"{row['group']} | {row['base_metric']}: "
                    f"n={int(row['n_units']) if pd.notna(row['n_units']) else 'NA'}, "
                    f"mean pre={row['pre_mean']:.4g}, mean post={row['post_mean']:.4g}, "
                    f"mean change={row['mean_change_post_minus_pre']:.4g}, "
                    f"t={row['t_stat']:.3g}, df={row['df']:.3g}, "
                    f"p={_format_p(float(row['p_value']))}, "
                    f"BH-FDR p={_format_p(float(row['p_bh_fdr']))}\n"
                )

        f.write("\nWelch tests of change metric versus sham\n")
        f.write("----------------------------------------\n")
        if welch_df.empty:
            f.write("No Welch tests were run.\n")
        else:
            for _, row in welch_df.iterrows():
                f.write(
                    f"{row['group']} vs {row['reference_group']} | {row['change_metric']}: "
                    f"n_group={int(row['n_group']) if pd.notna(row['n_group']) else 'NA'}, "
                    f"n_sham={int(row['n_sham']) if pd.notna(row['n_sham']) else 'NA'}, "
                    f"group mean={row['group_mean']:.4g}, sham mean={row['sham_mean']:.4g}, "
                    f"difference={row['mean_difference_group_minus_sham']:.4g}, "
                    f"t={row['t_stat']:.3g}, df={row['df']:.3g}, "
                    f"p={_format_p(float(row['p_value']))}, "
                    f"BH-FDR p={_format_p(float(row['p_bh_fdr']))}\n"
                )

    outputs["panel_C_stats_summary"] = summary_path
    return outputs



# ──────────────────────────────────────────────────────────────────────────────
# Panel C boxplots
# ──────────────────────────────────────────────────────────────────────────────
def _boxplot_group_color(group: str, color_config: Dict[str, Any]) -> str:
    """Return a plotting color for a lesion group, including pooled M+L."""
    color_map = _color_map_from_config(color_config)
    if group == POOLED_ML_GROUP:
        return color_map.get(PARTIAL_ML_GROUP, "#7A4FB7")
    return color_map.get(group, "#4D4D4D")


def _collapse_ml_groups_for_boxplots(df: pd.DataFrame, *, group_col: str = "display_group") -> pd.DataFrame:
    """Pool partial + complete medial/lateral lesions into one combined group."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r} for pooled boxplot groups.")
    out = df.copy()
    out[group_col] = out[group_col].replace({PARTIAL_ML_GROUP: POOLED_ML_GROUP, COMPLETE_ML_GROUP: POOLED_ML_GROUP})
    return out


def _format_group_label_horizontal(group: str) -> str:
    """Wrap long lesion-group labels for horizontal x tick display."""
    mapping = {
        SHAM_GROUP: "sham saline\ninjection",
        LATERAL_ONLY_GROUP: "Lateral lesion\nonly",
        PARTIAL_ML_GROUP: "Partial Medial and\nLateral lesion",
        COMPLETE_ML_GROUP: "Complete Medial and\nLateral lesion",
        POOLED_ML_GROUP: "Complete and partial\nmedial and lateral lesion",
    }
    return mapping.get(str(group), str(group).replace(" ", "\n", 1))


def _ordered_boxplot_groups(groups_present: Iterable[str], *, include_pooled_ml: bool = True) -> list[str]:
    preferred = [
        SHAM_GROUP,
        LATERAL_ONLY_GROUP,
        POOLED_ML_GROUP,
    ]
    if not include_pooled_ml:
        preferred = [SHAM_GROUP, LATERAL_ONLY_GROUP, PARTIAL_ML_GROUP, COMPLETE_ML_GROUP]
    present = list(dict.fromkeys(str(g) for g in groups_present if str(g) != "nan"))
    ordered = [g for g in preferred if g in present]
    ordered += [g for g in present if g not in ordered and g != "unknown"]
    if "unknown" in present:
        ordered.append("unknown")
    return ordered


def _apply_optional_y_scale(
    ax: plt.Axes,
    *,
    y_scale: str,
    values: Optional[np.ndarray] = None,
    symlog_linthresh: Optional[float] = None,
) -> None:
    y_scale = str(y_scale).lower()
    if y_scale == "linear":
        return
    if y_scale == "log":
        ax.set_yscale("log")
        return
    if y_scale == "symlog":
        if symlog_linthresh is None:
            vals = _finite_1d(values if values is not None else [])
            if vals.size:
                symlog_linthresh = max(np.nanpercentile(np.abs(vals), 10), 1e-9)
            else:
                symlog_linthresh = 1.0
        ax.set_yscale("symlog", linthresh=float(symlog_linthresh))
        return
    raise ValueError("y_scale must be one of: linear, log, symlog")


def plot_pre_post_distribution_boxplot(
    df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    pre_col: str,
    post_col: str,
    y_label: str,
    title: Optional[str] = None,
    group_col: str = "display_group",
    unit_col: str = "Animal ID",
    y_scale: str = "log",
    paired_stats_df: Optional[pd.DataFrame] = None,
    base_metric: Optional[str] = None,
    pool_ml_groups: bool = True,
    figsize: tuple[float, float] = (10.8, 6.6),
    box_width: float = 0.28,
    point_alpha: float = 0.80,
    point_size: float = 32.0,
    line_alpha: float = 0.22,
    show_points: bool = True,
    show_paired_lines: bool = True,
    tick_label_fontsize: float = 14,
    axis_label_fontsize: float = 18,
    title_fontsize: float = 18,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """Plot paired Late Pre and Post distributions for each lesion group.

    By default this is intended for the animal-level summary table, so each
    point is one animal. Open symbols/boxes are Late Pre, filled symbols/boxes
    are Post. Thin paired lines connect the same unit where possible when
    show_paired_lines=True. Individual points are overlaid when show_points=True.

    When pool_ml_groups=True, partial and complete medial/lateral lesion birds
    are combined into one pooled medial+lateral group for all boxplots.
    """
    out_path = Path(out_path)
    required = [group_col, pre_col, post_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for pre/post boxplot: {missing}")

    work = df.copy()
    if pool_ml_groups:
        work = _collapse_ml_groups_for_boxplots(work, group_col=group_col)
    work[pre_col] = pd.to_numeric(work[pre_col], errors="coerce")
    work[post_col] = pd.to_numeric(work[post_col], errors="coerce")
    work = work.dropna(subset=[group_col, pre_col, post_col])
    if work.empty:
        raise ValueError(f"No rows available for pre/post boxplot using {pre_col} and {post_col}.")

    display_order = _ordered_boxplot_groups(work[group_col].astype(str).unique(), include_pooled_ml=pool_ml_groups)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    xticks: list[float] = []
    xticklabels: list[str] = []
    all_values: list[float] = []
    group_positions: dict[str, tuple[float, float, float]] = {}

    for i, group in enumerate(display_order):
        g = work[work[group_col].astype(str) == group].copy()
        if g.empty:
            continue

        color = _boxplot_group_color(group, color_config)
        center = i * 1.55
        pre_pos = center - 0.20
        post_pos = center + 0.20
        pre_vals = g[pre_col].dropna().to_numpy(dtype=float)
        post_vals = g[post_col].dropna().to_numpy(dtype=float)
        all_values.extend(pre_vals.tolist())
        all_values.extend(post_vals.tolist())

        paired = g.dropna(subset=[pre_col, post_col])
        if show_paired_lines:
            for _, row in paired.iterrows():
                ax.plot([pre_pos, post_pos], [row[pre_col], row[post_col]], color=color, alpha=line_alpha, linewidth=0.8, zorder=1)

        if len(pre_vals):
            bp_pre = ax.boxplot([pre_vals], positions=[pre_pos], widths=box_width, patch_artist=True, showfliers=False)
            for patch in bp_pre["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor(color)
                patch.set_linewidth(1.6)
            for item in bp_pre["whiskers"] + bp_pre["caps"] + bp_pre["medians"]:
                item.set_color(color)
                item.set_linewidth(1.4)
            if show_points:
                jitter = rng.uniform(-0.045, 0.045, size=len(pre_vals))
                ax.scatter(np.full(len(pre_vals), pre_pos) + jitter, pre_vals, s=point_size, facecolors="white", edgecolors=color, alpha=point_alpha, linewidths=1.1, zorder=3)

        if len(post_vals):
            bp_post = ax.boxplot([post_vals], positions=[post_pos], widths=box_width, patch_artist=True, showfliers=False)
            for patch in bp_post["boxes"]:
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(0.45)
                patch.set_linewidth(1.6)
            for item in bp_post["whiskers"] + bp_post["caps"] + bp_post["medians"]:
                item.set_color(color)
                item.set_linewidth(1.4)
            if show_points:
                jitter = rng.uniform(-0.045, 0.045, size=len(post_vals))
                ax.scatter(np.full(len(post_vals), post_pos) + jitter, post_vals, s=point_size, facecolors=color, edgecolors="none", alpha=point_alpha, zorder=3)

        xticks.append(center)
        xticklabels.append(_format_group_label_horizontal(group))
        group_positions[str(group)] = (pre_pos, post_pos, center)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=tick_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=-2)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    # Title intentionally omitted for manuscript panel assembly.
    _apply_optional_y_scale(ax, y_scale=y_scale, values=np.asarray(all_values, dtype=float))
    _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Late Pre"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="black", markeredgecolor="black", alpha=0.45, markersize=8, label="Post"),
    ]
    # Put the Late Pre/Post legend fully above the axes so it does not
    # overlap either the data or the significance brackets.
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=13,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.24),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    if paired_stats_df is not None and not paired_stats_df.empty and base_metric is not None:
        for i, group in enumerate(display_order):
            row = paired_stats_df[(paired_stats_df.get("group", pd.Series(dtype=str)).astype(str) == group) & (paired_stats_df.get("base_metric", pd.Series(dtype=str)).astype(str) == str(base_metric))]
            if row.empty:
                continue
            p_value = row.iloc[0].get("p_value", np.nan)
            stars = _significance_stars(p_value)
            pre_pos, post_pos, _ = group_positions.get(str(group), (i * 1.55 - 0.20, i * 1.55 + 0.20, i * 1.55))
            sig_label = stars if stars else "n.s."
            _draw_sig_bracket_axesfrac(ax, pre_pos, post_pos, y_frac=0.955, h_frac=0.018, label=sig_label)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.24, top=0.76)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _significance_stars(p_value: float) -> str:
    """Return significance stars for a p-value."""
    try:
        p = float(p_value)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_plot_value(value: float) -> str:
    """Compact numeric formatter for medians printed under boxplots."""
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    av = abs(v)
    if av >= 1e6:
        return f"{v:.2g}"
    if av >= 1e4:
        return f"{v:.3g}"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2g}"





def _draw_sig_bracket_axesfrac(
    ax: plt.Axes,
    x1: float,
    x2: float,
    *,
    y_frac: float,
    h_frac: float = 0.020,
    label: str = "*",
    linewidth: float = 1.1,
    fontsize: float = 13.5,
) -> None:
    """Draw a comparison bracket inside the axes using axis-fraction y coordinates."""
    trans = ax.get_xaxis_transform()
    ax.plot(
        [x1, x1, x2, x2],
        [y_frac, y_frac + h_frac, y_frac + h_frac, y_frac],
        transform=trans,
        color="black",
        linewidth=linewidth,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2.0,
        y_frac + h_frac + 0.004,
        label,
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
    )

def plot_change_metric_boxplot_vs_sham(
    df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    metric_col: str,
    y_label: str,
    stats_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    group_col: str = "display_group",
    include_pooled_ml: bool = True,
    y_scale: str = "symlog",
    figsize: tuple[float, float] = (9.3, 6.0),
    box_width: float = 0.42,
    point_alpha: float = 0.82,
    point_size: float = 34.0,
    tick_label_fontsize: float = 14,
    axis_label_fontsize: float = 18,
    title_fontsize: float = 18,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """Plot Post-Late Pre change metrics and annotate Welch tests versus sham.

    For all boxplots, the partial and complete medial/lateral lesion groups are
    pooled into one combined medial+lateral group.
    """
    out_path = Path(out_path)
    required = [group_col, metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for change-metric boxplot: {missing}")

    plot_df = df.copy()
    plot_df[group_col] = plot_df[group_col].astype(str)
    if include_pooled_ml:
        plot_df = _collapse_ml_groups_for_boxplots(plot_df, group_col=group_col)
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[group_col, metric_col])
    if plot_df.empty:
        raise ValueError(f"No rows available for change-metric boxplot using {metric_col}.")

    plot_order = _ordered_boxplot_groups(plot_df[group_col].astype(str).unique(), include_pooled_ml=include_pooled_ml)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(1)
    all_values: list[float] = []

    for i, group in enumerate(plot_order):
        g = plot_df[plot_df[group_col].astype(str) == group].copy()
        vals = g[metric_col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        all_values.extend(vals.tolist())
        color = _boxplot_group_color(group, color_config)

        bp = ax.boxplot([vals], positions=[i], widths=box_width, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.45)
            patch.set_linewidth(1.6)
        for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
            item.set_color(color)
            item.set_linewidth(1.4)

        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=point_size, facecolors=color, edgecolors="none", alpha=point_alpha, zorder=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(len(plot_order)))
    ax.set_xticklabels([_format_group_label_horizontal(g) for g in plot_order], rotation=0, ha="center", fontsize=tick_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=8)
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=16)
    _apply_optional_y_scale(ax, y_scale=y_scale, values=np.asarray(all_values, dtype=float))
    _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    if stats_df is not None and not stats_df.empty:
        # Use the exact rows from the saved Welch stats table. That table already
        # contains a correctly computed pooled medial+lateral row. Do NOT relabel
        # partial/complete rows here, or the annotation can accidentally display
        # the partial-only p-value on the pooled boxplot.
        work_stats = stats_df.copy()
        for i, group in enumerate(plot_order):
            if group == SHAM_GROUP:
                n_sham_only = int((plot_df[group_col].astype(str) == group).sum())
                sham_vals_for_median = plot_df.loc[plot_df[group_col].astype(str) == group, metric_col].to_numpy(dtype=float)
                sham_median = np.nanmedian(sham_vals_for_median) if len(sham_vals_for_median) else np.nan
                ax.text(
                    i,
                    -0.30,
                    f"median Δ={_format_plot_value(sham_median)}\nreference\nn={n_sham_only}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9.0,
                    clip_on=False,
                )
                continue
            row = work_stats[
                (work_stats.get("group", pd.Series(dtype=str)).astype(str) == group)
                & (work_stats.get("change_metric", pd.Series(dtype=str)).astype(str) == metric_col)
            ]
            if row.empty:
                continue
            p_value = row.iloc[0].get("p_value", np.nan)
            t_stat = row.iloc[0].get("t_stat", np.nan)
            n_group = row.iloc[0].get("n_group", np.nan)
            n_sham = row.iloc[0].get("n_sham", np.nan)
            p_txt = _format_p(float(p_value)) if pd.notna(p_value) else "NA"
            t_txt = f"{float(t_stat):.2g}" if pd.notna(t_stat) else "NA"
            try:
                n_txt = f"n={int(n_group)} vs {int(n_sham)}"
            except Exception:
                n_txt = "n=NA"
            stars = _significance_stars(p_value)
            group_vals_for_median = plot_df.loc[plot_df[group_col].astype(str) == str(group), metric_col].to_numpy(dtype=float)
            group_median = np.nanmedian(group_vals_for_median) if len(group_vals_for_median) else np.nan
            ax.text(
                i,
                -0.34,
                f"median Δ={_format_plot_value(group_median)}\nt={t_txt}\np={p_txt}\n{n_txt}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11.0,
                clip_on=False,
            )
            sig_label = stars if stars else "n.s."
            y_frac = 0.93 if i == 1 else 0.975
            _draw_sig_bracket_axesfrac(ax, 0.0, float(i), y_frac=y_frac, h_frac=0.018, label=sig_label)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.50, top=0.88)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def make_panel_c_boxplots(
    scatter_df: pd.DataFrame,
    out_dir: PathLike,
    *,
    color_config: Dict[str, Any],
    stats_level: str = "animal",
    stats_paired_df: Optional[pd.DataFrame] = None,
    stats_welch_df: Optional[pd.DataFrame] = None,
    boxplot_tick_label_fontsize: float = 14,
    boxplot_axis_label_fontsize: float = 18,
    boxplot_title_fontsize: float = 18,
    prepost_boxplot_fig_width: float = 8.2,
    prepost_boxplot_fig_height: float = 4.8,
    delta_boxplot_fig_width: float = 8.2,
    delta_boxplot_fig_height: float = 5.0,
    show: bool = False,
    dpi: int = 300,
) -> Dict[str, Path]:
    """Save Panel C distribution and change-metric boxplots.

    All boxplots pool the partial and complete medial/lateral lesion groups into
    one combined medial+lateral lesion group.
    """
    out_dir = _ensure_dir(Path(out_dir))
    outputs: Dict[str, Path] = {}

    animal_df = _build_animal_level_change_table(scatter_df)
    animal_boxplot_path = out_dir / "panel_C_boxplot_animal_level_summary.csv"
    animal_df.to_csv(animal_boxplot_path, index=False)
    outputs["panel_C_boxplot_animal_level_summary"] = animal_boxplot_path

    stats_level = str(stats_level).lower()
    plot_df = scatter_df.copy() if stats_level == "animal_syllable" else animal_df.copy()
    pooled_plot_df = _collapse_ml_groups_for_boxplots(plot_df, group_col="display_group")
    pooled_summary_path = out_dir / "panel_C_boxplot_pooledML_summary.csv"
    pooled_plot_df.to_csv(pooled_summary_path, index=False)
    outputs["panel_C_boxplot_pooledML_summary"] = pooled_summary_path

    outputs["panel_C_variance_pre_post_boxplot"] = plot_pre_post_distribution_boxplot(
        plot_df,
        out_dir / "panel_C_variance_pre_post_boxplot_pooledML.png",
        color_config=color_config,
        pre_col="pre_variance",
        post_col="post_variance",
        y_label="Variance of Phrase Durations (ms²)",
        title=None,
        y_scale="log",
        paired_stats_df=stats_paired_df,
        base_metric="variance",
        pool_ml_groups=True,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(prepost_boxplot_fig_width, prepost_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )
    outputs["panel_C_sd_pre_post_boxplot"] = plot_pre_post_distribution_boxplot(
        plot_df,
        out_dir / "panel_C_sd_pre_post_boxplot_pooledML.png",
        color_config=color_config,
        pre_col="pre_sd",
        post_col="post_sd",
        y_label="Standard Deviation of\nPhrase Durations (ms)",
        title=None,
        y_scale="log",
        paired_stats_df=stats_paired_df,
        base_metric="sd",
        pool_ml_groups=True,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(prepost_boxplot_fig_width, prepost_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )

    outputs["panel_C_variance_pre_post_boxplot_boxesonly"] = plot_pre_post_distribution_boxplot(
        plot_df,
        out_dir / "panel_C_variance_pre_post_boxplot_pooledML_boxesonly.png",
        color_config=color_config,
        pre_col="pre_variance",
        post_col="post_variance",
        y_label="Variance of Phrase Durations (ms²)",
        title=None,
        y_scale="log",
        paired_stats_df=stats_paired_df,
        base_metric="variance",
        pool_ml_groups=True,
        show_points=False,
        show_paired_lines=False,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(prepost_boxplot_fig_width, prepost_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )
    outputs["panel_C_sd_pre_post_boxplot_boxesonly"] = plot_pre_post_distribution_boxplot(
        plot_df,
        out_dir / "panel_C_sd_pre_post_boxplot_pooledML_boxesonly.png",
        color_config=color_config,
        pre_col="pre_sd",
        post_col="post_sd",
        y_label="Standard Deviation of\nPhrase Durations (ms)",
        title=None,
        y_scale="log",
        paired_stats_df=stats_paired_df,
        base_metric="sd",
        pool_ml_groups=True,
        show_points=False,
        show_paired_lines=False,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(prepost_boxplot_fig_width, prepost_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )

    outputs["panel_C_variance_delta_vs_sham_boxplot"] = plot_change_metric_boxplot_vs_sham(
        plot_df,
        out_dir / "panel_C_variance_delta_vs_sham_boxplot_pooledML.png",
        color_config=color_config,
        metric_col="variance_delta",
        y_label="Δ Variance (Post − Late Pre, ms²)",
        title="Change in variance by lesion group",
        stats_df=stats_welch_df,
        y_scale="symlog",
        include_pooled_ml=True,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(delta_boxplot_fig_width, delta_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )
    outputs["panel_C_sd_delta_vs_sham_boxplot"] = plot_change_metric_boxplot_vs_sham(
        plot_df,
        out_dir / "panel_C_sd_delta_vs_sham_boxplot_pooledML.png",
        color_config=color_config,
        metric_col="sd_delta",
        y_label="Δ Standard Deviation\n(Post − Late Pre, ms)",
        title="Change in standard deviation by lesion group",
        stats_df=stats_welch_df,
        y_scale="symlog",
        include_pooled_ml=True,
        tick_label_fontsize=boxplot_tick_label_fontsize,
        axis_label_fontsize=boxplot_axis_label_fontsize,
        title_fontsize=boxplot_title_fontsize,
        figsize=(delta_boxplot_fig_width, delta_boxplot_fig_height),
        show=show,
        dpi=dpi,
    )

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Panel D: variance over days
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_daily_table(
    csv_path: PathLike,
    *,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = "animal_hit_type_summary",
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = "Lesion hit type",
    color_config: Dict[str, Any] = DEFAULT_COLOR_CONFIG,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["animal_id", "syllable", "relative_day", "Variance (ms^2)", "lesion_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Daily CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["animal_id"] = df["animal_id"].astype(str)
    df["syllable"] = df["syllable"].astype(str)
    df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    df["Variance (ms^2)"] = pd.to_numeric(df["Variance (ms^2)"], errors="coerce")
    df = df.dropna(subset=["relative_day", "Variance (ms^2)"])

    hit_type_map = (
        _load_hit_type_map(
            metadata_excel,
            sheet_name=meta_sheet_name,
            animal_col=meta_animal_col,
            hit_type_col=meta_hit_type_col,
        )
        if metadata_excel is not None
        else {}
    )

    df["raw_hit_type"] = df["animal_id"].map(hit_type_map)

    # Prefer raw metadata. If missing, fall back to lesion_group from the CSV.
    df["display_group"] = [
        _canonical_display_group(
            raw if pd.notna(raw) else lesion_group,
            color_config,
            fallback=_canonical_display_group(lesion_group, color_config, fallback="unknown"),
        )
        for raw, lesion_group in zip(df["raw_hit_type"], df["lesion_group"])
    ]

    if x_min is not None:
        df = df[df["relative_day"] >= int(x_min)]
    if x_max is not None:
        df = df[df["relative_day"] <= int(x_max)]

    return df.sort_values(["display_group", "animal_id", "syllable", "relative_day"]).reset_index(drop=True)


def _apply_y_scale(ax: plt.Axes, y_scale: str, symlog_linthresh: float) -> None:
    y_scale = y_scale.lower()
    if y_scale == "linear":
        return
    if y_scale == "log":
        ax.set_yscale("log")
        return
    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh)
        return
    raise ValueError("y_scale must be one of: linear, log, symlog")


def _set_common_ylim(axes: Iterable[plt.Axes], df: pd.DataFrame, y_scale: str) -> None:
    vals = pd.to_numeric(df["Variance (ms^2)"], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    if y_scale == "log":
        vals = vals[vals > 0]
        if vals.size == 0:
            return
        ymin, ymax = float(vals.min()) * 0.95, float(vals.max()) * 1.05
    elif y_scale == "symlog":
        ymin, ymax = -1e4, 1e9
    else:
        ymin, ymax = 0.0, float(vals.max()) * 1.05

    for ax in axes:
        ax.set_ylim(ymin, ymax)


def _plot_daily_trajectories(
    ax: plt.Axes,
    df_subset: pd.DataFrame,
    *,
    color_config: Dict[str, Any],
    line_alpha: float = 0.45,
    line_width: float = 1.15,
    marker_size: float = 2.4,
) -> None:
    color_map = _color_map_from_config(color_config)
    for (_, _, display_group), g in df_subset.groupby(["animal_id", "syllable", "display_group"], sort=False):
        g = g.sort_values("relative_day")
        ax.plot(
            g["relative_day"],
            g["Variance (ms^2)"],
            color=color_map.get(display_group, "#4D4D4D"),
            alpha=line_alpha,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markeredgewidth=0,
            solid_capstyle="round",
        )


def _legend_handles(color_config: Dict[str, Any], *, lw: float = 2.8, markersize: float = 6.0) -> list[Line2D]:
    color_map = _color_map_from_config(color_config)
    order = list(color_config.get("display_order", color_map.keys()))
    return [
        Line2D(
            [0],
            [0],
            color=color_map.get(group, "#4D4D4D"),
            lw=lw,
            marker="o",
            markersize=markersize,
            label=group,
        )
        for group in order
        if group in color_map and group != "unknown"
    ]


def plot_panel_d_variance_over_days(
    daily_df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    y_scale: str = "symlog",
    symlog_linthresh: float = 10000.0,
    figsize: tuple[float, float] = (16, 9),
    line_alpha: float = 0.45,
    line_width: float = 1.15,
    marker_size: float = 2.4,
    x_label_fontsize: float = 20,
    y_label_fontsize: float = 20,
    tick_label_fontsize: float = 15,
    common_panel_ylim: bool = True,
    show_panel_titles: bool = True,
    figure_title: Optional[str] = None,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    out_path = Path(out_path)
    panel_defs = list(color_config.get("panel_d_panel_order", DEFAULT_COLOR_CONFIG["panel_d_panel_order"]))
    n_panels = len(panel_defs)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, sharey=common_panel_ylim)
    if n_panels == 1:
        axes = [axes]

    if figure_title:
        fig.suptitle(figure_title, fontsize=18, y=0.98)

    for ax, panel_def in zip(axes, panel_defs):
        title = panel_def.get("title", "")
        groups = set(panel_def.get("display_groups", []))
        g = daily_df[daily_df["display_group"].isin(groups)].copy()
        if not g.empty:
            _plot_daily_trajectories(
                ax,
                g,
                color_config=color_config,
                line_alpha=line_alpha,
                line_width=line_width,
                marker_size=marker_size,
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.1, alpha=0.8)
        if show_panel_titles:
            ax.set_title(title, fontsize=15, pad=4)
        _apply_y_scale(ax, y_scale, symlog_linthresh)
        _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    if common_panel_ylim:
        _set_common_ylim(axes, daily_df, y_scale=y_scale.lower())

    fig.supylabel("Variance of Phrase Durations (ms²)", fontsize=y_label_fontsize, x=0.065)
    axes[-1].set_xlabel("Days relative to lesion", fontsize=x_label_fontsize)

    top_rect = 0.96 if figure_title else 1.0
    fig.tight_layout(rect=[0.055, 0, 1, top_rect])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def save_panel_legend(
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    figsize: tuple[float, float] = (8, 2.4),
    fontsize: float = 13,
    dpi: int = 300,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(
        handles=_legend_handles(color_config, lw=0, markersize=9),
        frameon=False,
        loc="center",
        ncol=1,
        fontsize=fontsize,
        handletextpad=0.8,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Panel D alternative: cleaned SD timecourse
# ──────────────────────────────────────────────────────────────────────────────
def _panel_d_group_order_from_config(color_config: Dict[str, Any]) -> list[str]:
    panel_defs = list(color_config.get("panel_d_panel_order", DEFAULT_COLOR_CONFIG["panel_d_panel_order"]))
    ordered: list[str] = []
    seen: set[str] = set()
    for panel_def in panel_defs:
        groups = panel_def.get("display_groups", [])
        if len(groups) > 1 and set(groups) == {COMPLETE_ML_GROUP, PARTIAL_ML_GROUP}:
            g = POOLED_ML_GROUP
            if g not in seen:
                ordered.append(g)
                seen.add(g)
        else:
            for g in groups:
                if g not in seen:
                    ordered.append(g)
                    seen.add(g)
    return ordered


def _pretty_group_label_timecourse(group: str) -> str:
    if group == SHAM_GROUP:
        return "sham saline injection"
    if group == LATERAL_ONLY_GROUP:
        return "Lateral lesion only"
    if group == POOLED_ML_GROUP:
        return "Complete and partial medial and lateral lesion"
    if group == PARTIAL_ML_GROUP:
        return "Partial medial and lateral lesion"
    if group == COMPLETE_ML_GROUP:
        return "Complete medial and lateral lesion"
    return str(group)


def _make_daily_unit_table_for_sd_timecourse(
    df: pd.DataFrame,
    *,
    pool_ml_groups: bool = True,
    unit_level: str = "animal",
    within_unit_stat: str = "median",
) -> pd.DataFrame:
    work = df.copy()
    work["sd_ms"] = np.sqrt(np.clip(pd.to_numeric(work["Variance (ms^2)"], errors="coerce").to_numpy(dtype=float), 0, None))
    if pool_ml_groups:
        work["display_group"] = work["display_group"].replace({COMPLETE_ML_GROUP: POOLED_ML_GROUP, PARTIAL_ML_GROUP: POOLED_ML_GROUP})
    unit_level = unit_level.lower()
    within_unit_stat = within_unit_stat.lower()
    if unit_level == "animal":
        group_cols = ["display_group", "animal_id", "relative_day"]
    elif unit_level == "animal_syllable":
        work["unit_id"] = work["animal_id"].astype(str) + "::" + work["syllable"].astype(str)
        group_cols = ["display_group", "animal_id", "syllable", "unit_id", "relative_day"]
    else:
        raise ValueError("unit_level must be 'animal' or 'animal_syllable'")

    if within_unit_stat == "median":
        out = work.groupby(group_cols, dropna=False)["sd_ms"].median().reset_index()
    elif within_unit_stat == "mean":
        out = work.groupby(group_cols, dropna=False)["sd_ms"].mean().reset_index()
    else:
        raise ValueError("within_unit_stat must be 'median' or 'mean'")

    if unit_level == "animal":
        out["unit_id"] = out["animal_id"].astype(str)
    return out.sort_values(["display_group", "unit_id", "relative_day"]).reset_index(drop=True)


def _rolling_window_summary_for_sd_timecourse(
    unit_df: pd.DataFrame,
    *,
    value_col: str = "sd_ms",
    window_days: int = 7,
    summary_stat: str = "median",
    group_order: Optional[Sequence[str]] = None,
    no_cross_lesion_smoothing: bool = True,
) -> pd.DataFrame:
    if value_col not in unit_df.columns:
        raise ValueError(f"Missing value column {value_col!r}")
    half = max(int(window_days) // 2, 0)
    summary_stat = str(summary_stat).lower()
    rows = []
    groups = [g for g in (group_order or _panel_d_group_order_from_config(DEFAULT_COLOR_CONFIG)) if g in set(unit_df["display_group"])]
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
            rows.append({
                "display_group": group,
                "relative_day": day,
                "value_col": value_col,
                "window_days": int(window_days),
                "summary": float(np.nanmedian(vals) if summary_stat == "median" else np.nanmean(vals)),
                "q25": float(np.nanpercentile(vals, 25)),
                "q75": float(np.nanpercentile(vals, 75)),
                "n_values_window": int(len(vals)),
                "n_units_window": int(window["unit_id"].nunique()),
            })
    return pd.DataFrame(rows)


def _set_common_ylim_from_percentile_timecourse(
    axes: Sequence[plt.Axes],
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    pad_frac: float = 0.08,
) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    lo = float(np.nanpercentile(vals, lower_percentile))
    hi = float(np.nanpercentile(vals, upper_percentile))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    for ax in axes:
        ax.set_ylim(lo, hi)


def plot_panel_d_sd_timecourse(
    unit_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    value_col: str = "sd_ms",
    title: Optional[str] = None,
    y_label: str = "Standard deviation\nof phrase durations (ms)",
    group_order: Optional[Sequence[str]] = None,
    x_min: int = -30,
    x_max: int = 30,
    common_ylim: bool = True,
    y_lower_percentile: float = 1.0,
    y_upper_percentile: float = 99.0,
    show_raw_traces: bool = True,
    raw_alpha: float = 0.12,
    raw_linewidth: float = 0.8,
    smooth_linewidth: float = 3.0,
    iqr_alpha: float = 0.22,
    tick_label_fontsize: float = 12.5,
    panel_title_fontsize: float = 14.0,
    title_fontsize: float = 16.5,
    axis_label_fontsize: float = 17.0,
    legend_fontsize: float = 13.5,
    y_label_x: float = 0.045,
    figsize: tuple[float, float] = (14.0, 8.6),
    dpi: int = 300,
    show: bool = False,
) -> Path:
    out_path = Path(out_path)
    if group_order is None:
        group_order = _panel_d_group_order_from_config(color_config)
    group_order = [g for g in group_order if g in set(unit_df["display_group"])]

    fig, axes = plt.subplots(len(group_order), 1, figsize=figsize, sharex=True, sharey=common_ylim)
    if len(group_order) == 1:
        axes = [axes]

    color_map = _color_map_from_config(color_config)
    for ax, group in zip(axes, group_order):
        color = color_map.get(group, "#4D4D4D")
        g = unit_df[unit_df["display_group"] == group].copy()
        if show_raw_traces:
            for _, u in g.groupby("unit_id", sort=False):
                u = u.sort_values("relative_day")
                ax.plot(
                    u["relative_day"], u[value_col], color=color, alpha=raw_alpha, linewidth=raw_linewidth,
                    marker="o", markersize=1.8, markeredgewidth=0, zorder=1
                )
        s = smooth_df[(smooth_df["display_group"] == group) & (smooth_df["value_col"] == value_col)].copy()
        if not s.empty:
            s = s.sort_values("relative_day")
            ax.fill_between(s["relative_day"].to_numpy(dtype=float), s["q25"].to_numpy(dtype=float), s["q75"].to_numpy(dtype=float), color=color, alpha=iqr_alpha, linewidth=0, zorder=2)
            ax.plot(s["relative_day"], s["summary"], color=color, linewidth=smooth_linewidth, zorder=3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_title(_pretty_group_label_timecourse(group), fontsize=panel_title_fontsize, pad=6)
        ax.set_xlim(x_min, x_max)
        _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    if common_ylim:
        _set_common_ylim_from_percentile_timecourse(axes, unit_df[value_col].to_numpy(dtype=float), lower_percentile=y_lower_percentile, upper_percentile=y_upper_percentile)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.98)
    fig.supylabel(y_label, fontsize=axis_label_fontsize, x=y_label_x)
    axes[-1].set_xlabel("Days relative to lesion", fontsize=axis_label_fontsize)

    handles = []
    if show_raw_traces:
        handles.append(Line2D([0], [0], color="black", alpha=raw_alpha, linewidth=raw_linewidth, marker="o", markersize=2, label="animal daily values"))
    handles.extend([
        Line2D([0], [0], color="black", linewidth=smooth_linewidth, label="rolling median"),
        Line2D([0], [0], color="black", alpha=iqr_alpha, linewidth=8, label="rolling IQR"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="lesion day"),
    ])
    fig.legend(handles=handles, frameon=False, fontsize=legend_fontsize, loc="upper right", bbox_to_anchor=(0.985, 0.83), bbox_transform=fig.transFigure)
    fig.tight_layout(rect=[0.045, 0.03, 0.985, 0.93], h_pad=0.65)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_panel_d_sd_timecourse_overlay(
    unit_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    out_path: PathLike,
    *,
    color_config: Dict[str, Any],
    value_col: str = "sd_ms",
    title: Optional[str] = None,
    y_label: str = "Standard deviation\nof phrase durations (ms)",
    group_order: Optional[Sequence[str]] = None,
    x_min: int = -30,
    x_max: int = 30,
    y_lower_percentile: float = 1.0,
    y_upper_percentile: float = 99.0,
    show_raw_traces: bool = True,
    raw_alpha: float = 0.08,
    raw_linewidth: float = 0.75,
    smooth_linewidth: float = 3.0,
    iqr_alpha: float = 0.14,
    tick_label_fontsize: float = 13.5,
    title_fontsize: float = 16.5,
    axis_label_fontsize: float = 18.5,
    legend_fontsize: float = 15.0,
    figsize: tuple[float, float] = (8.8, 4.6),
    dpi: int = 300,
    show: bool = False,
) -> Path:
    """Plot all lesion hit-type SD timecourses on one shared axis."""
    out_path = Path(out_path)
    if group_order is None:
        group_order = _panel_d_group_order_from_config(color_config)
    group_order = [g for g in group_order if g in set(unit_df["display_group"])]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    color_map = _color_map_from_config(color_config)

    for group in group_order:
        color = color_map.get(group, "#4D4D4D")
        g = unit_df[unit_df["display_group"] == group].copy()

        if show_raw_traces:
            for _, u in g.groupby("unit_id", sort=False):
                u = u.sort_values("relative_day")
                ax.plot(
                    u["relative_day"],
                    u[value_col],
                    color=color,
                    alpha=raw_alpha,
                    linewidth=raw_linewidth,
                    marker="o",
                    markersize=1.6,
                    markeredgewidth=0,
                    zorder=1,
                )

        s = smooth_df[(smooth_df["display_group"] == group) & (smooth_df["value_col"] == value_col)].copy()
        if not s.empty:
            s = s.sort_values("relative_day")
            x = s["relative_day"].to_numpy(dtype=float)
            q25 = s["q25"].to_numpy(dtype=float)
            q75 = s["q75"].to_numpy(dtype=float)
            summary = s["summary"].to_numpy(dtype=float)
            ax.fill_between(x, q25, q75, color=color, alpha=iqr_alpha, linewidth=0, zorder=2)
            ax.plot(x, summary, color=color, linewidth=smooth_linewidth, zorder=3)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.85, zorder=4)
    ax.set_xlim(x_min, x_max)
    _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)
    _set_common_ylim_from_percentile_timecourse(
        [ax],
        unit_df[value_col].to_numpy(dtype=float),
        lower_percentile=y_lower_percentile,
        upper_percentile=y_upper_percentile,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=-2)
    ax.set_xlabel("Days relative to lesion", fontsize=axis_label_fontsize)

    handles = []
    if show_raw_traces:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                alpha=min(0.40, raw_alpha * 2.5),
                linewidth=1.0,
                marker="o",
                markersize=2.2,
                label="animal daily values",
            )
        )
    for group in group_order:
        color = color_map.get(group, "#4D4D4D")
        handles.append(Line2D([0], [0], color=color, linewidth=smooth_linewidth, label=_pretty_group_label_timecourse(group)))
    handles.extend(
        [
            Patch(facecolor="#808080", edgecolor="none", alpha=min(0.35, iqr_alpha * 1.8), label="rolling IQR"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="lesion day"),
        ]
    )
    ax.legend(handles=handles, frameon=False, fontsize=legend_fontsize, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path




# ──────────────────────────────────────────────────────────────────────────────
# One-call wrapper
# ──────────────────────────────────────────────────────────────────────────────
def make_panel_c_and_d_variance_plots(
    *,
    scatter_csv: Optional[PathLike],
    daily_csv: Optional[PathLike],
    out_dir: PathLike,
    color_json: Optional[PathLike] = None,
    metadata_excel: Optional[PathLike] = None,
    scatter_meta_sheet_name: Union[int, str] = "metadata_with_hit_type",
    daily_meta_sheet_name: Union[int, str] = "animal_hit_type_summary",
    meta_animal_col: str = "Animal ID",
    scatter_meta_hit_type_col: Optional[str] = "Lesion hit type",
    daily_meta_hit_type_col: Optional[str] = "Lesion hit type",
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    scatter_top_percentile: Optional[float] = 70.0,
    rank_on: str = "post",
    min_n_phrases: int = 5,
    scatter_axis_padding_frac: float = 0.0,
    run_stats: bool = True,
    stats_level: str = "animal",
    stats_alternative: str = "greater",
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_scale: str = "symlog",
    smooth_window_days: int = 7,
    smooth_stat: str = "median",
    unit_level: str = "animal",
    within_unit_stat: str = "median",
    timecourse_fig_width: float = 14.0,
    timecourse_fig_height: float = 8.6,
    timecourse_legend_fontsize: float = 13.5,
    timecourse_y_label_x: float = 0.045,
    overlay_fig_width: float = 8.2,
    overlay_fig_height: float = 4.2,
    overlay_tick_fontsize: float = 13.5,
    overlay_axis_label_fontsize: float = 18.5,
    overlay_title_fontsize: float = 16.5,
    boxplot_tick_label_fontsize: float = 14,
    boxplot_axis_label_fontsize: float = 18,
    boxplot_title_fontsize: float = 18,
    prepost_boxplot_fig_width: float = 8.2,
    prepost_boxplot_fig_height: float = 4.8,
    delta_boxplot_fig_width: float = 8.2,
    delta_boxplot_fig_height: float = 5.0,
    show: bool = False,
    dpi: int = 300,
) -> Dict[str, Path]:
    out_dir = _ensure_dir(Path(out_dir))
    color_config = _load_json_config(color_json)

    outputs: Dict[str, Path] = {}

    if scatter_csv is not None:
        scatter_df = _prepare_scatter_table(
            scatter_csv,
            pre_group=pre_group,
            post_group=post_group,
            min_n_phrases=min_n_phrases,
            top_percentile=scatter_top_percentile,
            rank_on=rank_on,
            metadata_excel=metadata_excel,
            meta_sheet_name=scatter_meta_sheet_name,
            meta_animal_col=meta_animal_col,
            meta_hit_type_col=scatter_meta_hit_type_col,
            color_config=color_config,
        )

        # Main scatter table. This includes both variance and SD columns.
        table_path = out_dir / "panel_C_late_pre_vs_post_variance_and_sd_scatter_table.csv"
        scatter_df.to_csv(table_path, index=False)
        outputs["panel_C_variance_and_sd_table"] = table_path

        # Backward-compatible table name for variance-only workflows.
        variance_table_path = out_dir / "panel_C_late_pre_vs_post_variance_scatter_table.csv"
        scatter_df.to_csv(variance_table_path, index=False)
        outputs["panel_C_variance_table"] = variance_table_path

        # Variance scatterplot, with no internal legend and tight axes.
        outputs["panel_C_variance_scatter"] = plot_panel_c_scatter(
            scatter_df,
            out_dir / "panel_C_late_pre_vs_post_variance_scatter.png",
            color_config=color_config,
            x_col="pre_variance",
            y_col="post_variance",
            x_label="Late Pre variance (ms²)",
            y_label="Post variance (ms²)",
            axis_padding_frac=scatter_axis_padding_frac,
            show_legend=False,
            show=show,
            dpi=dpi,
        )

        # Standard deviation scatterplot, with no internal legend and tight axes.
        outputs["panel_C_sd_scatter"] = plot_panel_c_scatter(
            scatter_df,
            out_dir / "panel_C_late_pre_vs_post_sd_scatter.png",
            color_config=color_config,
            x_col="pre_sd",
            y_col="post_sd",
            x_label="Late Pre standard deviation (ms)",
            y_label="Post standard deviation (ms)",
            axis_padding_frac=scatter_axis_padding_frac,
            show_legend=False,
            show=show,
            dpi=dpi,
        )

        stats_paired_df: Optional[pd.DataFrame] = None
        stats_welch_df: Optional[pd.DataFrame] = None
        if run_stats:
            stats_outputs = run_panel_c_stats(
                scatter_df,
                out_dir,
                stats_level=stats_level,
                alternative=stats_alternative,
            )
            outputs.update(stats_outputs)

            paired_path = stats_outputs.get("panel_C_within_group_pre_post_tests")
            if paired_path is not None and Path(paired_path).exists():
                try:
                    stats_paired_df = pd.read_csv(paired_path)
                except Exception as exc:
                    print(f"[WARN] Could not read paired stats table for boxplot annotations: {exc}")

            welch_path = stats_outputs.get("panel_C_change_metric_vs_sham_welch_tests")
            if welch_path is not None and Path(welch_path).exists():
                try:
                    stats_welch_df = pd.read_csv(welch_path)
                except Exception as exc:
                    print(f"[WARN] Could not read Welch stats table for boxplot annotations: {exc}")

        boxplot_outputs = make_panel_c_boxplots(
            scatter_df,
            out_dir,
            color_config=color_config,
            stats_level=stats_level,
            stats_paired_df=stats_paired_df,
            stats_welch_df=stats_welch_df,
            boxplot_tick_label_fontsize=boxplot_tick_label_fontsize,
            boxplot_axis_label_fontsize=boxplot_axis_label_fontsize,
            boxplot_title_fontsize=boxplot_title_fontsize,
            prepost_boxplot_fig_width=prepost_boxplot_fig_width,
            prepost_boxplot_fig_height=prepost_boxplot_fig_height,
            delta_boxplot_fig_width=delta_boxplot_fig_width,
            delta_boxplot_fig_height=delta_boxplot_fig_height,
            show=show,
            dpi=dpi,
        )
        outputs.update(boxplot_outputs)

    if daily_csv is not None:
        daily_df = _prepare_daily_table(
            daily_csv,
            x_min=x_min,
            x_max=x_max,
            metadata_excel=metadata_excel,
            meta_sheet_name=daily_meta_sheet_name,
            meta_animal_col=meta_animal_col,
            meta_hit_type_col=daily_meta_hit_type_col,
            color_config=color_config,
        )
        daily_table_path = out_dir / "panel_D_sd_timecourse_source_table.csv"
        daily_df.to_csv(daily_table_path, index=False)
        outputs["panel_D_table"] = daily_table_path

        unit_df = _make_daily_unit_table_for_sd_timecourse(
            daily_df,
            pool_ml_groups=True,
            unit_level=unit_level,
            within_unit_stat=within_unit_stat,
        )
        unit_table_path = out_dir / "panel_D_sd_timecourse_unit_table.csv"
        unit_df.to_csv(unit_table_path, index=False)
        outputs["panel_D_unit_table"] = unit_table_path

        group_order = _panel_d_group_order_from_config(color_config)
        smooth_df = _rolling_window_summary_for_sd_timecourse(
            unit_df,
            value_col="sd_ms",
            window_days=smooth_window_days,
            summary_stat=smooth_stat,
            group_order=group_order,
        )
        smooth_path = out_dir / "panel_D_sd_timecourse_rolling_summary.csv"
        smooth_df.to_csv(smooth_path, index=False)
        outputs["panel_D_rolling_summary"] = smooth_path

        timecourse_title = f"Phrase-duration SD over time ({smooth_window_days}-day rolling {smooth_stat})"
        outputs["panel_D_sd_timecourse"] = plot_panel_d_sd_timecourse(
            unit_df,
            smooth_df,
            out_dir / "panel_D_sd_timecourse.png",
            color_config=color_config,
            value_col="sd_ms",
            title=None,
            y_label="Standard deviation\nof phrase durations (ms)",
            group_order=group_order,
            x_min=-30 if x_min is None else int(x_min),
            x_max=30 if x_max is None else int(x_max),
            show_raw_traces=True,
            legend_fontsize=timecourse_legend_fontsize,
            y_label_x=timecourse_y_label_x,
            figsize=(timecourse_fig_width, timecourse_fig_height),
            dpi=dpi,
            show=show,
        )
        outputs["panel_D_sd_timecourse_summary_only"] = plot_panel_d_sd_timecourse(
            unit_df,
            smooth_df,
            out_dir / "panel_D_sd_timecourse_summary_only.png",
            color_config=color_config,
            value_col="sd_ms",
            title=None,
            y_label="Standard deviation\nof phrase durations (ms)",
            group_order=group_order,
            x_min=-30 if x_min is None else int(x_min),
            x_max=30 if x_max is None else int(x_max),
            show_raw_traces=False,
            legend_fontsize=timecourse_legend_fontsize,
            y_label_x=timecourse_y_label_x,
            figsize=(timecourse_fig_width, timecourse_fig_height),
            dpi=dpi,
            show=False,
        )
        overlay_size = (overlay_fig_width, overlay_fig_height)
        outputs["panel_D_sd_timecourse_overlay"] = plot_panel_d_sd_timecourse_overlay(
            unit_df,
            smooth_df,
            out_dir / "panel_D_sd_timecourse_overlay.png",
            color_config=color_config,
            value_col="sd_ms",
            title=None,
            y_label="Standard deviation\nof phrase durations (ms)",
            group_order=group_order,
            x_min=-30 if x_min is None else int(x_min),
            x_max=30 if x_max is None else int(x_max),
            show_raw_traces=True,
            tick_label_fontsize=overlay_tick_fontsize,
            axis_label_fontsize=overlay_axis_label_fontsize,
            title_fontsize=overlay_title_fontsize,
            legend_fontsize=timecourse_legend_fontsize,
            figsize=overlay_size,
            dpi=dpi,
            show=False,
        )
        outputs["panel_D_sd_timecourse_overlay_summary_only"] = plot_panel_d_sd_timecourse_overlay(
            unit_df,
            smooth_df,
            out_dir / "panel_D_sd_timecourse_overlay_summary_only.png",
            color_config=color_config,
            value_col="sd_ms",
            title=None,
            y_label="Standard deviation\nof phrase durations (ms)",
            group_order=group_order,
            x_min=-30 if x_min is None else int(x_min),
            x_max=30 if x_max is None else int(x_max),
            show_raw_traces=False,
            tick_label_fontsize=overlay_tick_fontsize,
            axis_label_fontsize=overlay_axis_label_fontsize,
            title_fontsize=overlay_title_fontsize,
            legend_fontsize=timecourse_legend_fontsize,
            figsize=overlay_size,
            dpi=dpi,
            show=False,
        )
        outputs["legend"] = save_panel_legend(
            out_dir / "panel_C_D_lesion_group_legend.png",
            color_config=color_config,
            dpi=dpi,
        )

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Make Panel C pre/post scatterplots, boxplots, statistics, plus a cleaned Panel D "
            "phrase-duration SD timecourse using one lesion-group color palette."
        )
    )

    p.add_argument(
        "--scatter-csv",
        type=str,
        default=None,
        help="Path to usage_balanced_phrase_duration_stats.csv or similar. Omit to skip Panel C.",
    )
    p.add_argument(
        "--daily-csv",
        type=str,
        default=None,
        help="Path to batch_aligned_phrase_duration_variance_top30.csv or similar. Omit to skip Panel D.",
    )
    p.add_argument("--out-dir", type=str, required=True, help="Directory where outputs will be saved.")
    p.add_argument("--color-json", type=str, default=None, help="Path to lesion-group color JSON.")
    p.add_argument("--metadata-excel", type=str, default=None, help="Metadata Excel with animal IDs and lesion hit type labels.")

    p.add_argument("--scatter-meta-sheet-name", type=str, default="metadata_with_hit_type")
    p.add_argument("--daily-meta-sheet-name", type=str, default="animal_hit_type_summary")
    p.add_argument("--meta-animal-col", type=str, default="Animal ID")
    p.add_argument("--scatter-meta-hit-type-col", type=str, default="Lesion hit type")
    p.add_argument("--daily-meta-hit-type-col", type=str, default="Lesion hit type")

    p.add_argument("--pre-group", type=str, default="Late Pre")
    p.add_argument("--post-group", type=str, default="Post")
    p.add_argument(
        "--scatter-top-percentile",
        type=float,
        default=70.0,
        help="Percentile threshold within each animal. Use 70 for top 30%%. Use -1 to disable.",
    )
    p.add_argument("--rank-on", choices=["pre", "post", "max"], default="post")
    p.add_argument("--min-n-phrases", type=int, default=5)
    p.add_argument(
        "--scatter-axis-padding-frac",
        type=float,
        default=0.0,
        help=(
            "Optional fractional padding in log10 space for Panel C scatter axes. "
            "Default 0.0 uses exact plotted data min/max. Try 0.03 if points look clipped."
        ),
    )

    p.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip Panel C statistical tests. By default stats are run when --scatter-csv is provided.",
    )
    p.add_argument(
        "--stats-level",
        choices=["animal", "animal_syllable"],
        default="animal",
        help=(
            "Statistical unit for Panel C tests. Default animal averages selected syllables within "
            "each bird before testing. animal_syllable treats each animal × syllable as independent."
        ),
    )
    p.add_argument(
        "--stats-alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help=(
            "Alternative hypothesis. Default greater tests post > pre for paired tests and "
            "lesion-group change > sham change for Welch tests."
        ),
    )

    p.add_argument("--x-min", type=int, default=None)
    p.add_argument("--x-max", type=int, default=None)
    p.add_argument("--y-scale", choices=["linear", "log", "symlog"], default="symlog")

    p.add_argument("--smooth-window-days", type=int, default=7, help="Centered rolling window size in days for the SD timecourse.")
    p.add_argument("--smooth-stat", choices=["median", "mean"], default="median", help="Rolling summary statistic for the SD timecourse.")
    p.add_argument("--unit-level", choices=["animal", "animal_syllable"], default="animal", help="Unit to collapse the daily timecourse before smoothing.")
    p.add_argument("--within-unit-stat", choices=["median", "mean"], default="median", help="Statistic used to collapse multiple syllables within each unit per day.")
    p.add_argument("--timecourse-fig-width", type=float, default=14.0, help="Width in inches for the SD timecourse panels.")
    p.add_argument("--timecourse-fig-height", type=float, default=8.6, help="Height in inches for the SD timecourse panels.")
    p.add_argument("--legend-fontsize", type=float, default=13.5, help="Legend font size for the SD timecourse panels.")
    p.add_argument("--timecourse-ylabel-x", type=float, default=0.045, help="Figure-coordinate x-position for the shared y-label on the SD timecourse panels. Increase slightly to move the label closer to the axes.")
    p.add_argument("--overlay-fig-width", type=float, default=8.2, help="Width in inches for the single-axis SD timecourse overlay figure.")
    p.add_argument("--overlay-fig-height", type=float, default=4.2, help="Height in inches for the single-axis SD timecourse overlay figure.")
    p.add_argument("--overlay-tick-fontsize", type=float, default=13.5, help="Tick-label font size for the single-axis SD timecourse overlay figure.")
    p.add_argument("--overlay-axis-label-fontsize", type=float, default=18.5, help="Axis-label font size for the single-axis SD timecourse overlay figure.")
    p.add_argument("--overlay-title-fontsize", type=float, default=16.5, help="Title font size for the single-axis SD timecourse overlay figure.")
    p.add_argument("--boxplot-tick-fontsize", type=float, default=16, help="Tick-label font size for Panel C boxplots.")
    p.add_argument("--boxplot-axis-label-fontsize", type=float, default=20, help="Axis-label font size for Panel C boxplots.")
    p.add_argument("--boxplot-title-fontsize", type=float, default=20, help="Title font size for Panel C boxplots.")
    p.add_argument("--prepost-boxplot-fig-width", type=float, default=8.2, help="Width in inches for the Late Pre vs Post boxplots.")
    p.add_argument("--prepost-boxplot-fig-height", type=float, default=4.8, help="Height in inches for the Late Pre vs Post boxplots.")
    p.add_argument("--delta-boxplot-fig-width", type=float, default=8.2, help="Width in inches for the change-vs-sham boxplots.")
    p.add_argument("--delta-boxplot-fig-height", type=float, default=5.0, help="Height in inches for the change-vs-sham boxplots.")

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true", help="Alias to be explicit in Terminal examples; overrides --show.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    scatter_top = (
        None
        if args.scatter_top_percentile is not None and args.scatter_top_percentile < 0
        else args.scatter_top_percentile
    )
    show = bool(args.show and not args.no_show)

    out = make_panel_c_and_d_variance_plots(
        scatter_csv=args.scatter_csv,
        daily_csv=args.daily_csv,
        out_dir=args.out_dir,
        color_json=args.color_json,
        metadata_excel=args.metadata_excel,
        scatter_meta_sheet_name=args.scatter_meta_sheet_name,
        daily_meta_sheet_name=args.daily_meta_sheet_name,
        meta_animal_col=args.meta_animal_col,
        scatter_meta_hit_type_col=args.scatter_meta_hit_type_col,
        daily_meta_hit_type_col=args.daily_meta_hit_type_col,
        pre_group=args.pre_group,
        post_group=args.post_group,
        scatter_top_percentile=scatter_top,
        rank_on=args.rank_on,
        min_n_phrases=args.min_n_phrases,
        scatter_axis_padding_frac=args.scatter_axis_padding_frac,
        run_stats=not args.skip_stats,
        stats_level=args.stats_level,
        stats_alternative=args.stats_alternative,
        x_min=args.x_min,
        x_max=args.x_max,
        y_scale=args.y_scale,
        smooth_window_days=args.smooth_window_days,
        smooth_stat=args.smooth_stat,
        unit_level=args.unit_level,
        within_unit_stat=args.within_unit_stat,
        timecourse_fig_width=args.timecourse_fig_width,
        timecourse_fig_height=args.timecourse_fig_height,
        timecourse_legend_fontsize=args.legend_fontsize,
        timecourse_y_label_x=args.timecourse_ylabel_x,
        overlay_fig_width=args.overlay_fig_width,
        overlay_fig_height=args.overlay_fig_height,
        overlay_tick_fontsize=args.overlay_tick_fontsize,
        overlay_axis_label_fontsize=args.overlay_axis_label_fontsize,
        overlay_title_fontsize=args.overlay_title_fontsize,
        boxplot_tick_label_fontsize=args.boxplot_tick_fontsize,
        boxplot_axis_label_fontsize=args.boxplot_axis_label_fontsize,
        boxplot_title_fontsize=args.boxplot_title_fontsize,
        prepost_boxplot_fig_width=args.prepost_boxplot_fig_width,
        prepost_boxplot_fig_height=args.prepost_boxplot_fig_height,
        delta_boxplot_fig_width=args.delta_boxplot_fig_width,
        delta_boxplot_fig_height=args.delta_boxplot_fig_height,
        show=show,
        dpi=args.dpi,
    )

    print("[OK] Wrote:")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
