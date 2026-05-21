#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panel_C_D_variance_plots_with_json_colors.py

Make both figure components from the same lesion-group color JSON:

Panel C
-------
Pre/post variance scatterplot:
    x = Late Pre variance
    y = Post variance
    each point = one animal × syllable

Panel D
-------
Variance-over-time plot:
    x = days relative to lesion
    y = variance of phrase durations
    each line = one animal × syllable trajectory

The key design choice in this version is that Panel C points are all the same
size and marker shape. Groups differ by color/darkness only. The same colors
are then reused for Panel D lines so the panels are visually consistent.

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
python panel_C_D_variance_plots_with_json_colors.py \
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
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
            "color": "#2B004F",
            "aliases": ["large lesion Area X not visible", "Area X not visible", "complete medial and lateral lesion"],
        },
        "Partial Medial and Lateral lesion": {
            "color": "#6F4BAE",
            "aliases": ["Area X visible (medial+lateral hit)", "medial+lateral hit", "m+l hit"],
        },
        "Lateral lesion only": {
            "color": "#9A8FBF",
            "aliases": ["lateral hit only", "Area X visible (single hit)", "single hit"],
        },
        "sham saline injection": {
            "color": "#707070",
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


# ──────────────────────────────────────────────────────────────────────────────
# Panel C: pre/post variance scatter
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
    wide = wide[(wide["pre_variance"] > 0) & (wide["post_variance"] > 0)].copy()

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
        for aid, g in wide.groupby(animal_col, dropna=False):
            vals = g["rank_metric"].dropna()
            if vals.empty:
                continue
            threshold = np.nanpercentile(vals.to_numpy(dtype=float), top_percentile)
            keep_parts.append(g[g["rank_metric"] >= threshold].copy())
        wide = pd.concat(keep_parts, ignore_index=True) if keep_parts else wide.iloc[0:0].copy()

    hit_type_map = _load_hit_type_map(
        metadata_excel,
        sheet_name=meta_sheet_name,
        animal_col=meta_animal_col,
        hit_type_col=meta_hit_type_col,
    ) if metadata_excel is not None else {}

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
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    figsize: tuple[float, float] = (7.2, 6.2),
    marker_size: float = 30.0,
    alpha: float = 0.78,
    diagonal_color: str = "red",
    x_label_fontsize: float = 18,
    y_label_fontsize: float = 18,
    tick_label_fontsize: float = 14,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    out_path = Path(out_path)
    color_map = _color_map_from_config(color_config)
    display_order = list(color_config.get("display_order", color_map.keys()))

    fig, ax = plt.subplots(figsize=figsize)

    finite = scatter_df[["pre_variance", "post_variance"]].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("No finite pre/post variance rows to plot for Panel C.")

    xy_min = float(np.nanmin(finite.to_numpy(dtype=float)))
    xy_max = float(np.nanmax(finite.to_numpy(dtype=float)))
    axis_min = 10 ** math.floor(math.log10(max(xy_min * 0.7, 1e-12)))
    axis_max = 10 ** math.ceil(math.log10(xy_max * 1.3))

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=diagonal_color,
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        zorder=0,
    )

    # Plot least prominent first and most prominent last, while keeping same marker size.
    plot_order = [g for g in display_order[::-1] if g in set(scatter_df["display_group"].astype(str))]
    if "unknown" in set(scatter_df["display_group"].astype(str)) and "unknown" not in plot_order:
        plot_order.insert(0, "unknown")

    for group in plot_order:
        g = scatter_df[scatter_df["display_group"].astype(str) == group]
        if g.empty:
            continue
        ax.scatter(
            g["pre_variance"],
            g["post_variance"],
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
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("Late Pre variance (ms²)", fontsize=x_label_fontsize)
    ax.set_ylabel("Post variance (ms²)", fontsize=y_label_fontsize)
    _pretty_axes(ax, tick_label_fontsize=tick_label_fontsize)

    legend_groups = [g for g in display_order if g in set(scatter_df["display_group"].astype(str))]
    if "unknown" in set(scatter_df["display_group"].astype(str)):
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


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

    hit_type_map = _load_hit_type_map(
        metadata_excel,
        sheet_name=meta_sheet_name,
        animal_col=meta_animal_col,
        hit_type_col=meta_hit_type_col,
    ) if metadata_excel is not None else {}

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
            [0], [0],
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_scale: str = "symlog",
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
        table_path = out_dir / "panel_C_late_pre_vs_post_variance_scatter_table.csv"
        scatter_df.to_csv(table_path, index=False)
        outputs["panel_C_table"] = table_path

        outputs["panel_C_scatter"] = plot_panel_c_scatter(
            scatter_df,
            out_dir / "panel_C_late_pre_vs_post_variance_scatter.png",
            color_config=color_config,
            show=show,
            dpi=dpi,
        )

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
        daily_table_path = out_dir / "panel_D_variance_over_days_table.csv"
        daily_df.to_csv(daily_table_path, index=False)
        outputs["panel_D_table"] = daily_table_path

        outputs["panel_D_panels"] = plot_panel_d_variance_over_days(
            daily_df,
            out_dir / "panel_D_variance_over_days_panels.png",
            color_config=color_config,
            y_scale=y_scale,
            show_panel_titles=True,
            show=show,
            dpi=dpi,
        )
        outputs["panel_D_panels_no_titles"] = plot_panel_d_variance_over_days(
            daily_df,
            out_dir / "panel_D_variance_over_days_panels_no_titles.png",
            color_config=color_config,
            y_scale=y_scale,
            show_panel_titles=False,
            show=False,
            dpi=dpi,
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
        description="Make Panel C pre/post variance scatter and Panel D variance-over-days plots using one JSON color scheme."
    )

    p.add_argument("--scatter-csv", type=str, default=None, help="Path to usage_balanced_phrase_duration_stats.csv or similar. Omit to skip Panel C.")
    p.add_argument("--daily-csv", type=str, default=None, help="Path to batch_aligned_phrase_duration_variance_top30.csv or similar. Omit to skip Panel D.")
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
    p.add_argument("--scatter-top-percentile", type=float, default=70.0, help="Percentile threshold within each animal. Use 70 for top 30%%. Use -1 to disable.")
    p.add_argument("--rank-on", choices=["pre", "post", "max"], default="post")
    p.add_argument("--min-n-phrases", type=int, default=5)

    p.add_argument("--x-min", type=int, default=None)
    p.add_argument("--x-max", type=int, default=None)
    p.add_argument("--y-scale", choices=["linear", "log", "symlog"], default="symlog")

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true", help="Alias to be explicit in Terminal examples; overrides --show.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    scatter_top = None if args.scatter_top_percentile is not None and args.scatter_top_percentile < 0 else args.scatter_top_percentile
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
        x_min=args.x_min,
        x_max=args.x_max,
        y_scale=args.y_scale,
        show=show,
        dpi=args.dpi,
    )

    print("[OK] Wrote:")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
