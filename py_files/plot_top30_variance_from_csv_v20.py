#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_top30_variance_from_csv_v20.py

Make lesion-type grouped "top 30% variance syllables" plots directly from the
cached batch CSV produced by the phrase-duration batch pipeline.

This version adds:
- an additional panel figure with no subplot titles
- a separate larger PNG legend for the lesion color-coding
- optional filtering to exclude phrase/day rows with low occurrence counts
- automatic extra log-scale output versions for easier interpretation of low positive variances

Expected CSV columns
--------------------
- animal_id
- syllable
- relative_day
- Variance (ms^2)
- lesion_group

Optional per-row occurrence counts
----------------------------
If your CSV includes a count column (for example occurrences or n), you can
optionally filter out rows with fewer than a chosen number of occurrences per
day before plotting. This is useful for suppressing low-variance rows that may
reflect phrases that were rarely or not meaningfully sung on a given day.

Optional metadata Excel
-----------------------
If you provide the metadata Excel, the script will further split the combined
panel/group into:
- Partial Medial and Lateral lesion        -> dark purple
- Complete Medial and Lateral lesion       -> black

while keeping both in the SAME panel labeled:
    "combined large lesion and M+L hits"

Outputs
-------
- batch_aligned_phrase_duration_variance_top30_by_lesion_type_combined.png
- batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels.png
- batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels_no_titles.png
- batch_aligned_phrase_duration_variance_top30_by_lesion_type_legend_large.png
- optional *_min_occ_<N>.png filtered plot versions when a minimum occurrence threshold is provided
- optional *_log_y.png companion versions with a standard log y-axis
- optional *_linear_y.png companion versions with a standard linear y-axis
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LESION_GROUP_ORDER = [
    "sham saline injection",
    "lateral hit only",
    "combined large lesion and M+L hits",
]

LESION_GROUP_TITLE_COLORS = {
    "sham saline injection": "#d62728",
    "lateral hit only": "#b9a7ff",
    "combined large lesion and M+L hits": "#6a3d9a",
}

DISPLAY_GROUP_COLORS: Dict[str, str] = {
    "sham saline injection": "#d62728",
    "lateral hit only": "#c4b5fd",
    "Area X visible (medial+lateral hit)": "#6a3d9a",
    "large lesion Area X not visible": "#000000",
}

DISPLAY_GROUP_ORDER = [
    "sham saline injection",
    "lateral hit only",
    "Area X visible (medial+lateral hit)",
    "large lesion Area X not visible",
]

LEGEND_DISPLAY_GROUP_ORDER = [
    "large lesion Area X not visible",
    "Area X visible (medial+lateral hit)",
    "lateral hit only",
    "sham saline injection",
]

LEGEND_LABEL_MAP = {
    "Area X visible (medial+lateral hit)": "Partial Medial and Lateral lesion",
    "lateral hit only": "Lateral lesion only",
    "large lesion Area X not visible": "Complete Medial and Lateral lesion",
    "sham saline injection": "Sham saline injection",
}

GROUP_LINE_ALPHA = {
    "sham saline injection": 0.24,
    "lateral hit only": 0.38,
    "Area X visible (medial+lateral hit)": 0.45,
    "large lesion Area X not visible": 0.52,
}
GROUP_LINE_WIDTH = {
    "sham saline injection": 1.15,
    "lateral hit only": 1.55,
    "Area X visible (medial+lateral hit)": 1.85,
    "large lesion Area X not visible": 1.95,
}
GROUP_MARKER_SIZE = {
    "sham saline injection": 2.7,
    "lateral hit only": 3.1,
    "Area X visible (medial+lateral hit)": 3.3,
    "large lesion Area X not visible": 3.4,
}


def _normalize_lesion_group(x: object) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()

    sham_aliases = {"sham saline injection", "sham", "saline"}
    lateral_aliases = {
        "lateral hit only",
        "area x visible (single hit)",
        "single hit",
        "single lateral hit",
    }
    combined_aliases = {
        "combined large lesion and m+l hits",
        "area x visible (medial+lateral hit)",
        "medial+lateral hit",
        "medial lateral hit, combined with large lesion",
        "large lesion area x not visible",
        "area x visible and large lesion area x not visible",
        "combined (visible ml + not visible)",
        "m+l hit",
        "medial+lateral",
        "medial lateral",
    }

    if s in sham_aliases:
        return "sham saline injection"
    if s in lateral_aliases:
        return "lateral hit only"
    if s in combined_aliases:
        return "combined large lesion and M+L hits"
    return str(x)


def _normalize_raw_hit_type(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in {"sham saline injection", "sham", "saline"}:
        return "sham saline injection"
    if s in {
        "area x visible (single hit)",
        "single hit",
        "single lateral hit",
        "lateral hit only",
    }:
        return "lateral hit only"
    if s in {
        "area x visible (medial+lateral hit)",
        "medial+lateral hit",
        "m+l hit",
        "medial+lateral",
        "medial lateral",
    }:
        return "Area X visible (medial+lateral hit)"
    if s in {
        "large lesion area x not visible",
        "area x not visible",
        "large lesion, area x not visible",
    }:
        return "large lesion Area X not visible"
    return str(x)


def _display_group_from_row(lesion_group: str, raw_hit_type: Optional[str]) -> str:
    if lesion_group == "combined large lesion and M+L hits":
        raw_norm = _normalize_raw_hit_type(raw_hit_type)
        if raw_norm in {"Area X visible (medial+lateral hit)", "large lesion Area X not visible"}:
            return raw_norm
        return "Area X visible (medial+lateral hit)"
    return lesion_group


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"animal_id", "syllable", "relative_day", "Variance (ms^2)", "lesion_group"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))


def _load_hit_type_map(
    metadata_excel: Optional[Path],
    *,
    hit_type_sheet_name: str = "animal_hit_type_summary",
    hit_type_id_col: str = "Animal ID",
    hit_type_col: str = "Lesion hit type",
) -> Dict[str, str]:
    if metadata_excel is None:
        return {}

    df = pd.read_excel(metadata_excel, sheet_name=hit_type_sheet_name)
    if hit_type_id_col not in df.columns:
        raise ValueError(f"Column {hit_type_id_col!r} not found in sheet {hit_type_sheet_name!r}.")
    if hit_type_col not in df.columns:
        raise ValueError(f"Column {hit_type_col!r} not found in sheet {hit_type_sheet_name!r}.")

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        aid = row.get(hit_type_id_col)
        hit = row.get(hit_type_col)
        if pd.isna(aid) or pd.isna(hit):
            continue
        out[str(aid).strip()] = str(hit)
    return out


def _canonicalize_col_name(name: object) -> str:
    s = str(name).strip().lower()
    for ch in " ()[]{}-/.":
        s = s.replace(ch, "")
    return s


def _resolve_occurrence_count_col(df: pd.DataFrame, requested_col: Optional[str]) -> str:
    if requested_col is not None:
        if requested_col not in df.columns:
            raise ValueError(
                f"Requested occurrence count column {requested_col!r} was not found in the CSV. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return requested_col

    candidate_map = {
        _canonicalize_col_name(col): col
        for col in df.columns
    }
    candidate_names = [
        "occurrences",
        "occurrence_count",
        "occurrencecounts",
        "count",
        "counts",
        "n",
        "noccurrences",
        "numoccurrences",
        "numberofoccurrences",
        "phrasecount",
        "dailycount",
        "nfiles",
    ]
    for cand in candidate_names:
        if cand in candidate_map:
            return candidate_map[cand]

    raise ValueError(
        "A minimum occurrence filter was requested, but no occurrence-count column could be found automatically. "
        "Please provide the column explicitly with --occurrence-count-col. Available columns: "
        + ", ".join(map(str, df.columns))
    )


def _prepare_df(
    csv_path: Path,
    x_min: Optional[int],
    x_max: Optional[int],
    *,
    metadata_excel: Optional[Path] = None,
    hit_type_sheet_name: str = "animal_hit_type_summary",
    hit_type_id_col: str = "Animal ID",
    hit_type_col: str = "Lesion hit type",
    min_occurrences_per_day: Optional[int] = None,
    occurrence_count_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    df = df.copy()
    df["animal_id"] = df["animal_id"].astype(str)
    df["syllable"] = df["syllable"].astype(str)
    df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    df["Variance (ms^2)"] = pd.to_numeric(df["Variance (ms^2)"], errors="coerce")
    df["lesion_group"] = df["lesion_group"].map(_normalize_lesion_group)

    if min_occurrences_per_day is not None:
        resolved_occurrence_count_col = _resolve_occurrence_count_col(df, occurrence_count_col)
        df[resolved_occurrence_count_col] = pd.to_numeric(df[resolved_occurrence_count_col], errors="coerce")
        df = df[df[resolved_occurrence_count_col] >= int(min_occurrences_per_day)]

    df = df.dropna(subset=["relative_day", "Variance (ms^2)"])

    hit_type_map = _load_hit_type_map(
        metadata_excel,
        hit_type_sheet_name=hit_type_sheet_name,
        hit_type_id_col=hit_type_id_col,
        hit_type_col=hit_type_col,
    )
    df["raw_hit_type"] = df["animal_id"].map(hit_type_map)
    df["display_group"] = [
        _display_group_from_row(lg, ht)
        for lg, ht in zip(df["lesion_group"], df["raw_hit_type"])
    ]

    if x_min is not None:
        df = df[df["relative_day"] >= int(x_min)]
    if x_max is not None:
        df = df[df["relative_day"] <= int(x_max)]

    return df.sort_values(["lesion_group", "display_group", "animal_id", "syllable", "relative_day"]).reset_index(drop=True)


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
    raise ValueError(f"Unsupported y_scale: {y_scale}")


def _apply_requested_symlog_ylim(ax: plt.Axes, y_scale: str) -> None:
    if y_scale.lower() == "symlog":
        ax.set_ylim(-1e4, 1e9)


def _set_common_ylim(axes: Iterable[plt.Axes], df: pd.DataFrame, y_scale: str) -> None:
    vals = pd.to_numeric(df["Variance (ms^2)"], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    if y_scale.lower() == "log":
        vals = vals[vals > 0]
        if vals.size == 0:
            return
        ymin = float(vals.min()) * 0.95
        ymax = float(vals.max()) * 1.05
    elif y_scale.lower() == "symlog":
        ymin = -1e4
        ymax = 1e9
    else:
        ymin = 0.0
        ymax = float(vals.max())
        ymax *= 1.05 if ymax > 0 else 1.0

    for ax in axes:
        ax.set_ylim(ymin, ymax)


def _plot_trajectory_group(
    ax: plt.Axes,
    df_subset: pd.DataFrame,
    *,
    use_group_specific_style: bool = True,
    base_line_alpha: float = 0.18,
    base_line_width: float = 1.0,
    base_marker_size: float = 2.8,
) -> None:
    group_cols = ["animal_id", "syllable", "lesion_group", "display_group"]
    for (_, _, _, display_group), g in df_subset.groupby(group_cols, sort=False):
        color = DISPLAY_GROUP_COLORS.get(display_group, "0.5")
        g = g.sort_values("relative_day")

        if use_group_specific_style:
            line_alpha = GROUP_LINE_ALPHA.get(display_group, base_line_alpha)
            line_width = GROUP_LINE_WIDTH.get(display_group, base_line_width)
            marker_size = GROUP_MARKER_SIZE.get(display_group, base_marker_size)
        else:
            line_alpha = base_line_alpha
            line_width = base_line_width
            marker_size = base_marker_size

        ax.plot(
            g["relative_day"],
            g["Variance (ms^2)"],
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markeredgewidth=0,
            solid_capstyle="round",
        )


def _legend_handles(
    order: Optional[Iterable[str]] = None,
    *,
    label_map: Optional[Dict[str, str]] = None,
    lw: float = 3.0,
    markersize: float = 5.5,
) -> list[Line2D]:
    order = list(order) if order is not None else list(DISPLAY_GROUP_ORDER)
    if label_map is None:
        label_map = {group: group for group in order}
    return [
        Line2D(
            [0], [0],
            color=DISPLAY_GROUP_COLORS[group],
            lw=lw,
            marker="o",
            markersize=markersize,
            label=label_map.get(group, group),
        )
        for group in order
    ]


def _ylabel_text(multiline: bool = True) -> str:
    return "Variance of\nPhrase Durations\n(ms²)" if multiline else "Variance of Phrase Durations (ms²)"


def plot_combined(
    df: pd.DataFrame,
    out_path: Path,
    *,
    y_scale: str = "linear",
    symlog_linthresh: float = 10000.0,
    figsize: tuple[float, float] = (16, 10),
    line_alpha: float = 0.18,
    line_width: float = 1.0,
    marker_size: float = 2.8,
    x_label_fontsize: float = 24,
    y_label_fontsize: float = 24,
    tick_label_fontsize: float = 20,
    show_plot: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    split_y_label_lines: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    _plot_trajectory_group(
        ax,
        df,
        use_group_specific_style=True,
        base_line_alpha=line_alpha,
        base_line_width=line_width,
        base_marker_size=marker_size,
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1.4)
    ax.set_title("Top 30% variance syllables — aligned to lesion day, colored by lesion hit type", fontsize=18)
    ax.set_xlabel("Days relative to lesion", fontsize=x_label_fontsize)
    ax.set_ylabel(_ylabel_text(split_y_label_lines), fontsize=y_label_fontsize, labelpad=16)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.legend(
        handles=_legend_handles(order=LEGEND_DISPLAY_GROUP_ORDER, label_map=LEGEND_LABEL_MAP),
        title="Lesion hit type",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )

    _apply_y_scale(ax, y_scale=y_scale, symlog_linthresh=symlog_linthresh)
    _apply_requested_symlog_ylim(ax, y_scale)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0.08, 0, 0.82, 1])
    fig.savefig(out_path, dpi=dpi, transparent=transparent)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_panels(
    df: pd.DataFrame,
    out_path: Path,
    *,
    y_scale: str = "linear",
    symlog_linthresh: float = 10000.0,
    figsize: tuple[float, float] = (16, 10),
    line_alpha: float = 0.18,
    line_width: float = 1.0,
    marker_size: float = 2.8,
    x_label_fontsize: float = 24,
    y_label_fontsize: float = 24,
    tick_label_fontsize: float = 20,
    common_panel_ylim: bool = True,
    show_plot: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    split_y_label_lines: bool = False,
    show_panel_titles: bool = True,
    figure_title: Optional[str] = "Top 30% variance syllables — aligned to lesion day by lesion hit type",
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=common_panel_ylim)
    if figure_title:
        fig.suptitle(figure_title, fontsize=20, y=0.98)

    for ax, lesion_group in zip(axes, LESION_GROUP_ORDER):
        g = df[df["lesion_group"] == lesion_group].copy()
        if not g.empty:
            _plot_trajectory_group(
                ax,
                g,
                use_group_specific_style=True,
                base_line_alpha=line_alpha,
                base_line_width=line_width,
                base_marker_size=marker_size,
            )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.3)
        if show_panel_titles:
            ax.set_title(
                lesion_group,
                fontsize=16,
                color=LESION_GROUP_TITLE_COLORS.get(lesion_group, "black"),
                pad=4,
            )
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        _apply_y_scale(ax, y_scale=y_scale, symlog_linthresh=symlog_linthresh)
        _apply_requested_symlog_ylim(ax, y_scale)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if common_panel_ylim:
        _set_common_ylim(axes, df, y_scale=y_scale)

    fig.supylabel(
        "Variance of Phrase Durations (ms²)" if not split_y_label_lines else _ylabel_text(True),
        fontsize=y_label_fontsize,
        x=0.075,
        y=0.5,
    )
    axes[-1].set_xlabel("Days relative to lesion", fontsize=x_label_fontsize)
    fig.subplots_adjust(left=0.11)

    top_rect = 0.96 if figure_title else 1.0
    fig.tight_layout(rect=[0.06, 0, 1, top_rect])
    fig.savefig(out_path, dpi=dpi, transparent=transparent)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def save_large_legend_png(
    out_path: Path,
    *,
    figsize: tuple[float, float] = (18, 4.2),
    dpi: int = 300,
    transparent: bool = False,
    font_size: float = 24,
    markersize: float = 20,
    frameon: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    handles = _legend_handles(
        order=LEGEND_DISPLAY_GROUP_ORDER,
        label_map=LEGEND_LABEL_MAP,
        lw=0,
        markersize=markersize,
    )

    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=frameon,
        fontsize=font_size,
        handletextpad=1.0,
        borderpad=0.6,
        labelspacing=0.7,
        fancybox=False,
        edgecolor=None,
    )
    if frameon and legend.get_frame() is not None:
        legend.get_frame().set_facecolor("white")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi, transparent=transparent, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _suffix_with_log(base_suffix: str) -> str:
    return f"{base_suffix}_log_y" if base_suffix else "_log_y"


def _suffix_with_linear(base_suffix: str) -> str:
    return f"{base_suffix}_linear_y" if base_suffix else "_linear_y"


def _save_linear_scale_companion_plots(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    base_suffix: str,
    panels_figsize: tuple[float, float],
    combined_figsize: tuple[float, float],
    line_alpha: float,
    line_width: float,
    marker_size: float,
    x_label_fontsize: float,
    y_label_fontsize: float,
    tick_label_fontsize: float,
    common_panel_ylim: bool,
    dpi: int,
    transparent: bool,
    split_y_label_lines: bool,
    make_titleless_panel_version: bool,
    titleless_figure_title: Optional[str],
) -> dict[str, Path]:
    linear_suffix = _suffix_with_linear(base_suffix)
    outputs: dict[str, Path] = {}

    combined_linear_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_combined{linear_suffix}.png"
    plot_combined(
        df,
        combined_linear_path,
        y_scale="linear",
        figsize=combined_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        show_plot=False,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
    )
    outputs["combined_linear_path"] = combined_linear_path

    panels_linear_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels{linear_suffix}.png"
    plot_panels(
        df,
        panels_linear_path,
        y_scale="linear",
        figsize=panels_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        common_panel_ylim=common_panel_ylim,
        show_plot=False,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
        show_panel_titles=True,
        figure_title="Top 30% variance syllables — aligned to lesion day by lesion hit type (linear y-axis)",
    )
    outputs["panels_linear_path"] = panels_linear_path

    if make_titleless_panel_version:
        panels_no_titles_linear_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels_no_titles{linear_suffix}.png"
        plot_panels(
            df,
            panels_no_titles_linear_path,
            y_scale="linear",
            figsize=panels_figsize,
            line_alpha=line_alpha,
            line_width=line_width,
            marker_size=marker_size,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            common_panel_ylim=common_panel_ylim,
            show_plot=False,
            dpi=dpi,
            transparent=transparent,
            split_y_label_lines=split_y_label_lines,
            show_panel_titles=False,
            figure_title=titleless_figure_title,
        )
        outputs["panels_no_titles_linear_path"] = panels_no_titles_linear_path

    return outputs


def _save_log_scale_companion_plots(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    base_suffix: str,
    panels_figsize: tuple[float, float],
    combined_figsize: tuple[float, float],
    line_alpha: float,
    line_width: float,
    marker_size: float,
    x_label_fontsize: float,
    y_label_fontsize: float,
    tick_label_fontsize: float,
    common_panel_ylim: bool,
    dpi: int,
    transparent: bool,
    split_y_label_lines: bool,
    make_titleless_panel_version: bool,
    titleless_figure_title: Optional[str],
) -> dict[str, Path]:
    log_suffix = _suffix_with_log(base_suffix)
    outputs: dict[str, Path] = {}

    combined_log_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_combined{log_suffix}.png"
    plot_combined(
        df,
        combined_log_path,
        y_scale="log",
        figsize=combined_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        show_plot=False,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
    )
    outputs["combined_log_path"] = combined_log_path

    panels_log_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels{log_suffix}.png"
    plot_panels(
        df,
        panels_log_path,
        y_scale="log",
        figsize=panels_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        common_panel_ylim=common_panel_ylim,
        show_plot=False,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
        show_panel_titles=True,
        figure_title="Top 30% variance syllables — aligned to lesion day by lesion hit type (log y-axis)",
    )
    outputs["panels_log_path"] = panels_log_path

    if make_titleless_panel_version:
        panels_no_titles_log_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels_no_titles{log_suffix}.png"
        plot_panels(
            df,
            panels_no_titles_log_path,
            y_scale="log",
            figsize=panels_figsize,
            line_alpha=line_alpha,
            line_width=line_width,
            marker_size=marker_size,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            common_panel_ylim=common_panel_ylim,
            show_plot=False,
            dpi=dpi,
            transparent=transparent,
            split_y_label_lines=split_y_label_lines,
            show_panel_titles=False,
            figure_title=titleless_figure_title,
        )
        outputs["panels_no_titles_log_path"] = panels_no_titles_log_path

    return outputs


def make_top30_plots_from_csv(
    csv_path: Path | str,
    out_dir: Path | str,
    *,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    metadata_excel: Optional[Path | str] = None,
    hit_type_sheet_name: str = "animal_hit_type_summary",
    hit_type_id_col: str = "Animal ID",
    hit_type_col: str = "Lesion hit type",
    y_scale: str = "linear",
    symlog_linthresh: float = 10000.0,
    combined_figsize: tuple[float, float] = (16, 10),
    panels_figsize: tuple[float, float] = (16, 10),
    legend_figsize: tuple[float, float] = (18, 4.2),
    line_alpha: float = 0.18,
    line_width: float = 1.0,
    marker_size: float = 2.8,
    x_label_fontsize: float = 24,
    y_label_fontsize: float = 24,
    tick_label_fontsize: float = 20,
    common_panel_ylim: bool = True,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    split_y_label_lines: bool = False,
    make_titleless_panel_version: bool = True,
    titleless_figure_title: Optional[str] = None,
    legend_font_size: float = 24,
    legend_marker_size: float = 20,
    min_occurrences_per_day: Optional[int] = None,
    occurrence_count_col: Optional[str] = None,
    also_save_log_versions: bool = True,
    also_save_linear_versions: bool = True,
) -> dict[str, Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_excel = Path(metadata_excel) if metadata_excel is not None else None

    df = _prepare_df(
        csv_path,
        x_min=x_min,
        x_max=x_max,
        metadata_excel=metadata_excel,
        hit_type_sheet_name=hit_type_sheet_name,
        hit_type_id_col=hit_type_id_col,
        hit_type_col=hit_type_col,
        min_occurrences_per_day=min_occurrences_per_day,
        occurrence_count_col=occurrence_count_col,
    )
    if df.empty:
        if min_occurrences_per_day is None:
            raise ValueError("No rows left to plot after loading/filtering the CSV.")
        raise ValueError(
            f"No rows left to plot after filtering the CSV with min_occurrences_per_day={int(min_occurrences_per_day)}."
        )

    suffix = "" if min_occurrences_per_day is None else f"_min_occ_{int(min_occurrences_per_day)}"
    combined_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_combined{suffix}.png"
    panels_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels{suffix}.png"
    panels_no_titles_path = out_dir / f"batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels_no_titles{suffix}.png"
    legend_large_path = out_dir / "batch_aligned_phrase_duration_variance_top30_by_lesion_type_legend_large.png"

    plot_combined(
        df,
        combined_path,
        y_scale=y_scale,
        symlog_linthresh=symlog_linthresh,
        figsize=combined_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        show_plot=show_plots,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
    )

    plot_panels(
        df,
        panels_path,
        y_scale=y_scale,
        symlog_linthresh=symlog_linthresh,
        figsize=panels_figsize,
        line_alpha=line_alpha,
        line_width=line_width,
        marker_size=marker_size,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        common_panel_ylim=common_panel_ylim,
        show_plot=show_plots,
        dpi=dpi,
        transparent=transparent,
        split_y_label_lines=split_y_label_lines,
        show_panel_titles=True,
        figure_title="Top 30% variance syllables — aligned to lesion day by lesion hit type",
    )

    outputs: dict[str, Path] = {
        "combined_path": combined_path,
        "panels_path": panels_path,
    }

    if make_titleless_panel_version:
        plot_panels(
            df,
            panels_no_titles_path,
            y_scale=y_scale,
            symlog_linthresh=symlog_linthresh,
            figsize=panels_figsize,
            line_alpha=line_alpha,
            line_width=line_width,
            marker_size=marker_size,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            common_panel_ylim=common_panel_ylim,
            show_plot=False,
            dpi=dpi,
            transparent=transparent,
            split_y_label_lines=split_y_label_lines,
            show_panel_titles=False,
            figure_title=titleless_figure_title,
        )
        outputs["panels_no_titles_path"] = panels_no_titles_path

    save_large_legend_png(
        legend_large_path,
        figsize=legend_figsize,
        dpi=dpi,
        transparent=transparent,
        font_size=legend_font_size,
        markersize=legend_marker_size,
        frameon=False,
    )
    outputs["legend_large_path"] = legend_large_path

    if also_save_log_versions and y_scale.lower() != "log":
        outputs.update(
            _save_log_scale_companion_plots(
                df,
                out_dir,
                base_suffix=suffix,
                panels_figsize=panels_figsize,
                combined_figsize=combined_figsize,
                line_alpha=line_alpha,
                line_width=line_width,
                marker_size=marker_size,
                x_label_fontsize=x_label_fontsize,
                y_label_fontsize=y_label_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                common_panel_ylim=common_panel_ylim,
                dpi=dpi,
                transparent=transparent,
                split_y_label_lines=split_y_label_lines,
                make_titleless_panel_version=make_titleless_panel_version,
                titleless_figure_title=titleless_figure_title,
            )
        )

    if also_save_linear_versions and y_scale.lower() != "linear":
        outputs.update(
            _save_linear_scale_companion_plots(
                df,
                out_dir,
                base_suffix=suffix,
                panels_figsize=panels_figsize,
                combined_figsize=combined_figsize,
                line_alpha=line_alpha,
                line_width=line_width,
                marker_size=marker_size,
                x_label_fontsize=x_label_fontsize,
                y_label_fontsize=y_label_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                common_panel_ylim=common_panel_ylim,
                dpi=dpi,
                transparent=transparent,
                split_y_label_lines=split_y_label_lines,
                make_titleless_panel_version=make_titleless_panel_version,
                titleless_figure_title=titleless_figure_title,
            )
        )

    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot top-30% aligned variance figures directly from the cached batch CSV.")
    p.add_argument("--csv-path", type=str, required=True, help="Path to batch_aligned_phrase_duration_variance_top30.csv")
    p.add_argument("--out-dir", type=str, required=True, help="Directory where PNG outputs will be saved")
    p.add_argument("--x-min", type=int, default=None, help="Optional minimum relative day to plot")
    p.add_argument("--x-max", type=int, default=None, help="Optional maximum relative day to plot")
    p.add_argument("--metadata-excel", type=str, default=None, help="Optional metadata Excel to split combined panel into partial M+L lesion vs complete M+L lesion black lines")
    p.add_argument("--hit-type-sheet-name", type=str, default="animal_hit_type_summary")
    p.add_argument("--hit-type-id-col", type=str, default="Animal ID")
    p.add_argument("--hit-type-col", type=str, default="Lesion hit type")
    p.add_argument("--y-scale", type=str, default="linear", choices=["linear", "log", "symlog"])
    p.add_argument("--symlog-linthresh", type=float, default=10000.0)
    p.add_argument("--line-alpha", type=float, default=0.18)
    p.add_argument("--line-width", type=float, default=1.0)
    p.add_argument("--marker-size", type=float, default=2.8)
    p.add_argument("--x-label-fontsize", type=float, default=18)
    p.add_argument("--y-label-fontsize", type=float, default=18)
    p.add_argument("--tick-label-fontsize", type=float, default=12)
    p.add_argument("--legend-font-size", type=float, default=24)
    p.add_argument("--legend-marker-size", type=float, default=20)
    p.add_argument("--min-occurrences-per-day", type=int, default=None, help="Optional minimum number of phrase occurrences required for a syllable/day row to be plotted")
    p.add_argument("--occurrence-count-col", type=str, default=None, help="Optional CSV column name containing per-day occurrence counts; if omitted, the script tries to detect it automatically")
    p.add_argument("--no-log-version", action="store_true", help="Do not save the extra companion plots with a standard log y-axis")
    p.add_argument("--no-linear-version", action="store_true", help="Do not save the extra companion plots with a standard linear y-axis")
    p.add_argument("--split-y-label-lines", action="store_true", help="Split the y-axis label across multiple lines")
    p.add_argument("--no-common-panel-ylim", action="store_true", help="Allow each lesion-group panel to use its own y-range")
    p.add_argument("--no-titleless-panel-version", action="store_true", help="Do not save the additional panel figure without subplot titles")
    p.add_argument("--titleless-figure-title", type=str, default=None, help="Optional overall title for the no-subplot-title panel figure")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--show", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    out = make_top30_plots_from_csv(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        x_min=args.x_min,
        x_max=args.x_max,
        metadata_excel=args.metadata_excel,
        hit_type_sheet_name=args.hit_type_sheet_name,
        hit_type_id_col=args.hit_type_id_col,
        hit_type_col=args.hit_type_col,
        y_scale=args.y_scale,
        symlog_linthresh=args.symlog_linthresh,
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        marker_size=args.marker_size,
        x_label_fontsize=args.x_label_fontsize,
        y_label_fontsize=args.y_label_fontsize,
        tick_label_fontsize=args.tick_label_fontsize,
        legend_font_size=args.legend_font_size,
        legend_marker_size=args.legend_marker_size,
        min_occurrences_per_day=args.min_occurrences_per_day,
        occurrence_count_col=args.occurrence_count_col,
        also_save_log_versions=not args.no_log_version,
        also_save_linear_versions=not args.no_linear_version,
        split_y_label_lines=args.split_y_label_lines,
        common_panel_ylim=not args.no_common_panel_ylim,
        make_titleless_panel_version=not args.no_titleless_panel_version,
        titleless_figure_title=args.titleless_figure_title,
        show_plots=args.show,
        dpi=args.dpi,
        transparent=args.transparent,
    )
    print("[OK] Wrote:")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
