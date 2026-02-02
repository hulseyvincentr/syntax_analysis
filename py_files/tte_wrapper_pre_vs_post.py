#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tte_wrapper_pre_vs_post.py

Wrapper around tte_by_day.TTE_by_day to:

1) Run TTE_by_day on all birds found under a root directory.
2) Collect daily aggregated TTE per bird.
3) Build:
    - Multi-bird TTE vs days-from-lesion BROKEN x-axis plot (colored by bird)
    - Multi-bird TTE vs days-from-lesion BROKEN x-axis plot (colored by treatment group: Sham vs NMA)
    - Grouped pre vs post paired boxplots (one subplot per treatment type).

Key updates (this edit):
- Remove numeric labels over points (n_songs labels OFF).
- Use dot markers instead of 'x'.
- Move legends to the RIGHT of the figure (outside the data area).
- Add a second plot colored by Sham vs NMA lesion.

Outputs
-------
TTEWrapperResult with:
- multi_bird_timecourse_path:                per-bird colors plot
- multi_bird_timecourse_by_group_path:       treatment-group colors plot
- prepost_boxplot_path:                      grouped pre/post boxplots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for cross-bird paired stats
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Core per-bird TTE function
from tte_by_day import (
    TTE_by_day,
    TTEByDayResult,
    _extract_id_from_text,
    _parse_treatment_date,
)

# Optional metadata helper (not required; we can also read Excel directly)
try:
    from organize_metadata_excel import build_areax_metadata  # type: ignore
except Exception:  # pragma: no cover
    build_areax_metadata = None  # type: ignore


# ---------------------------------------------------------------------
# Dataclasses for returning structured results
# ---------------------------------------------------------------------
@dataclass
class BirdDailyTTEResult:
    animal_id: str
    decoded_path: Path
    song_detection_path: Optional[Path]
    treatment_date: pd.Timestamp
    treatment_type: Optional[str]
    tte_result: TTEByDayResult
    daily_df: pd.DataFrame  # results_df + animal_id + days_from_lesion + treatment_type + treatment_group


@dataclass
class TTEWrapperResult:
    per_bird: List[BirdDailyTTEResult]
    combined_daily_df: pd.DataFrame
    prepost_summary_df: pd.DataFrame
    multi_bird_timecourse_path: Optional[Path]                 # colored by bird
    multi_bird_timecourse_by_group_path: Optional[Path]        # colored by Sham vs NMA
    prepost_boxplot_path: Optional[Path]
    prepost_p_value: Optional[float]
    group_p_values: Dict[str, Optional[float]]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _load_treatment_info_from_metadata(
    metadata_excel: Union[str, Path]
) -> Tuple[Dict[str, Union[str, pd.Timestamp]], Dict[str, Optional[str]]]:
    """
    Build two dicts from the metadata Excel:
        bird_to_date: {animal_id -> treatment_date}
        bird_to_type: {animal_id -> treatment_type}

    Tries build_areax_metadata if available; else reads Excel with pandas.
    """
    metadata_excel = Path(metadata_excel)

    def _from_df(dfm: pd.DataFrame) -> Tuple[Dict[str, Union[str, pd.Timestamp]], Dict[str, Optional[str]]]:
        id_col = "Animal ID" if "Animal ID" in dfm.columns else dfm.columns[0]

        # date col
        tcol = None
        for cand in ["Treatment date", "Treatment_date", "treatment_date"]:
            if cand in dfm.columns:
                tcol = cand
                break
        if tcol is None:
            raise KeyError("Could not find a 'Treatment date' column in metadata Excel.")

        # type col (optional)
        type_col = None
        for cand in ["Treatment type", "Treatment_type", "treatment_type"]:
            if cand in dfm.columns:
                type_col = cand
                break

        bird_to_date: Dict[str, Union[str, pd.Timestamp]] = {}
        bird_to_type: Dict[str, Optional[str]] = {}

        for aid, sub in dfm.groupby(id_col):
            aid_str = str(aid)

            tvals = sub[tcol].dropna()
            if len(tvals) > 0:
                bird_to_date[aid_str] = tvals.iloc[0]

            if type_col is not None:
                type_vals = sub[type_col].dropna()
                bird_to_type[aid_str] = type_vals.iloc[0] if len(type_vals) > 0 else None
            else:
                bird_to_type[aid_str] = None

        return bird_to_date, bird_to_type

    if build_areax_metadata is not None:
        try:
            meta = build_areax_metadata(metadata_excel)  # type: ignore
            if isinstance(meta, dict):
                bird_to_date: Dict[str, Union[str, pd.Timestamp]] = {}
                bird_to_type: Dict[str, Optional[str]] = {}
                for aid, info in meta.items():
                    if not isinstance(info, dict):
                        continue
                    aid_str = str(aid)

                    date_val = None
                    for key in ["Treatment date", "Treatment_date", "treatment_date"]:
                        if key in info and pd.notna(info[key]):
                            date_val = info[key]
                            break
                    if date_val is not None:
                        bird_to_date[aid_str] = date_val

                    type_val = None
                    for key in ["Treatment type", "Treatment_type", "treatment_type"]:
                        if key in info and pd.notna(info[key]):
                            type_val = info[key]
                            break
                    bird_to_type[aid_str] = type_val
                return bird_to_date, bird_to_type
            else:
                dfm = pd.DataFrame(meta)
                return _from_df(dfm)
        except Exception:
            pass

    dfm = pd.read_excel(metadata_excel)
    return _from_df(dfm)


def _find_detection_for_decoded(decoded_path: Path) -> Optional[Path]:
    """Try to find matching *_song_detection.json for a *_decoded_database.json."""
    parent = decoded_path.parent
    name = decoded_path.name

    if name.endswith("_decoded_database.json"):
        stem = name[: -len("_decoded_database.json")]
        candidate = parent / f"{stem}_song_detection.json"
        if candidate.exists():
            return candidate

    candidates = list(parent.glob("*song_detection*.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_animal_id_from_path(decoded_path: Path) -> str:
    """Infer an animal_id from decoded path using tte_by_day's regex helper."""
    text = decoded_path.stem
    mid = _extract_id_from_text(text)
    if mid:
        return mid
    return text.split("_")[0]


def _compute_pre_post_means(
    daily_df: pd.DataFrame, treatment_date: pd.Timestamp
) -> Tuple[Optional[float], Optional[float]]:
    """Bird-level mean of daily agg_TTE pre (< tx) and post (>= tx)."""
    tx = pd.Timestamp(treatment_date)
    pre_vals = daily_df.loc[daily_df["day"] < tx, "agg_TTE"]
    post_vals = daily_df.loc[daily_df["day"] >= tx, "agg_TTE"]
    pre_mean = float(pre_vals.mean()) if len(pre_vals) > 0 else None
    post_mean = float(post_vals.mean()) if len(post_vals) > 0 else None
    return pre_mean, post_mean


def _treatment_group_from_type(treatment_type: Optional[str]) -> str:
    """
    Collapse free-form 'Treatment type' into:
      - 'Sham' (saline/sham/control)
      - 'NMA'  (NMA lesion)
      - 'Unknown' otherwise
    """
    if treatment_type is None or (isinstance(treatment_type, float) and pd.isna(treatment_type)):
        return "Unknown"
    s = str(treatment_type).lower()
    if ("sham" in s) or ("saline" in s) or ("control" in s):
        return "Sham"
    if "nma" in s:
        return "NMA"
    return "Unknown"


# -----------------------------
# Broken x-axis plotting helpers
# -----------------------------
def _infer_x_segments_from_days(
    days_from_lesion: Union[pd.Series, np.ndarray, List[int]],
    *,
    gap_threshold: int = 3,
    force_zero_split: bool = True,
) -> List[Tuple[int, int]]:
    """Split x into segments when gaps exceed gap_threshold; also split at 0 (pre/post)."""
    xs = []
    for v in list(days_from_lesion):
        if pd.isna(v):
            continue
        try:
            xs.append(int(v))
        except Exception:
            continue
    if not xs:
        return []
    xs = sorted(set(xs))

    has_neg = any(x < 0 for x in xs)
    has_pos = any(x > 0 for x in xs)

    segments: List[Tuple[int, int]] = []
    start = xs[0]
    prev = xs[0]

    for x in xs[1:]:
        if force_zero_split and has_neg and has_pos and (prev < 0) and (x > 0):
            segments.append((start, 0))
            start = x
        elif (x - prev) > gap_threshold:
            segments.append((start, prev))
            start = x
        prev = x

    segments.append((start, prev))

    if force_zero_split and has_neg and has_pos:
        fixed: List[Tuple[int, int]] = []
        for a, b in segments:
            if a < 0 and b > 0:
                fixed.append((a, 0))
                fixed.append((1, b))
            else:
                fixed.append((a, b))
        segments = fixed

    return segments


def _add_x_break_marks(ax_left, ax_right, *, size: float = 0.015, lw: float = 1.5):
    """Draw diagonal '//' marks between adjacent axes."""
    kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False, linewidth=lw)
    ax_left.plot((1 - size, 1 + size), (-size, +size), **kwargs)
    ax_left.plot((1 - size, 1 + size), (-2 * size, 0), **kwargs)

    kwargs = dict(transform=ax_right.transAxes, color="k", clip_on=False, linewidth=lw)
    ax_right.plot((-size, +size), (-size, +size), **kwargs)
    ax_right.plot((-size, +size), (-2 * size, 0), **kwargs)


def _plot_all_birds_broken_x(
    combined_daily: pd.DataFrame,
    *,
    fig_dir: Path,
    out_name: str,
    title: str,
    color_mode: str = "bird",  # "bird" or "treatment_group"
    ycol: str = "agg_TTE",
    xcol: str = "days_from_lesion",
    gap_threshold: int = 3,
    marker: str = "o",         # dot marker
    markersize: float = 4.5,
    linewidth: float = 1.8,
    show: bool = True,
) -> Optional[Path]:
    """
    Broken-x multi-bird plot:
      - color_mode="bird": each bird gets its own color, legend lists birds
      - color_mode="treatment_group": birds colored by Sham vs NMA (legend lists groups)
    """
    if combined_daily.empty:
        return None

    segments = _infer_x_segments_from_days(combined_daily[xcol], gap_threshold=gap_threshold, force_zero_split=True)
    if not segments:
        return None

    # widths proportional to span
    widths = []
    for a, b in segments:
        span = max(1, abs(b - a) + 1)
        widths.append(max(1.2, min(10.0, float(span))))

    fig, axes = plt.subplots(
        1,
        len(segments),
        figsize=(max(10, 1.2 * sum(widths)), 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.05},
    )
    if len(segments) == 1:
        axes = [axes]

    yvals = pd.to_numeric(combined_daily[ycol], errors="coerce").dropna()
    if len(yvals) == 0:
        return None
    y_min = float(yvals.min())
    y_max = float(yvals.max())
    pad = 0.12 * (y_max - y_min) if y_max > y_min else 0.1
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad

    # axis formatting
    for si, ((xmin, xmax), ax) in enumerate(zip(segments, axes)):
        if xmin == xmax:
            ax.set_xlim(xmin - 0.5, xmax + 0.5)
            ax.set_xticks([xmin])
            ax.set_xticklabels([str(xmin)])
        else:
            ax.set_xlim(xmin, xmax)
            xticks = list(range(xmin, xmax + 1))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks])

        ax.set_ylim(y_lo, y_hi)
        ax.grid(axis="y", linewidth=1.0, alpha=0.35)
        ax.grid(axis="x", visible=False)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(top=False, right=False)

        if si > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False, left=False)

    # breaks
    for i in range(len(axes) - 1):
        axes[i].spines["right"].set_visible(False)
        axes[i + 1].spines["left"].set_visible(False)
        _add_x_break_marks(axes[i], axes[i + 1], size=0.015, lw=1.5)

    # color maps
    birds = sorted(combined_daily["animal_id"].astype(str).unique().tolist())

    if color_mode == "bird":
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bird_color = {bid: color_cycle[i % len(color_cycle)] for i, bid in enumerate(birds)}
        legend_handles = []
        legend_labels = []
    else:
        # group coloring
        group_color = {"NMA": "tab:blue", "Sham": "tab:orange", "Unknown": "0.5"}
        legend_handles = {}
        legend_labels = []

    # plot per bird per segment (don’t connect across breaks)
    for bid in birds:
        dfb = combined_daily[combined_daily["animal_id"].astype(str) == bid].copy()
        dfb = dfb.sort_values(xcol)

        if color_mode == "bird":
            col = bird_color[bid]
            label_for_legend = bid
        else:
            grp = _treatment_group_from_type(dfb["treatment_type"].iloc[0] if "treatment_type" in dfb.columns else None)
            col = group_color.get(grp, "0.5")
            label_for_legend = grp

        first_segment_for_this_line = True

        for (xmin, xmax), ax in zip(segments, axes):
            seg = dfb[(dfb[xcol].astype(int) >= xmin) & (dfb[xcol].astype(int) <= xmax)].copy()
            if seg.empty:
                continue

            x = seg[xcol].astype(int).to_numpy()
            y = pd.to_numeric(seg[ycol], errors="coerce").to_numpy()

            ln = ax.plot(
                x,
                y,
                marker=marker,
                linestyle="-",
                linewidth=linewidth,
                markersize=markersize,
                color=col,
                label=None,
            )[0]

            # collect legend handles once
            if first_segment_for_this_line:
                if color_mode == "bird":
                    legend_handles.append(ln)
                    legend_labels.append(label_for_legend)
                else:
                    # one handle per group
                    if label_for_legend not in legend_handles:
                        legend_handles[label_for_legend] = ln
                first_segment_for_this_line = False

    # lesion line at 0 + labels
    for (xmin, xmax), ax in zip(segments, axes):
        if xmin <= 0 <= xmax or xmax == 0:
            ax.axvline(0, linestyle="--", linewidth=1.5, color="0.6")
            ax.text(0, y_hi, "pre lesion", rotation=90, va="top", ha="right", fontsize=10, color="k")
            ax.text(0, y_hi, "\npost lesion", rotation=90, va="top", ha="left", fontsize=10, color="k")
            break

    # Titles/labels
    axes[0].set_ylabel("Transition Entropy")
    fig.supxlabel("Days post lesion")
    fig.suptitle(title, y=1.02)

    # Legend OUTSIDE to the right
    if color_mode == "bird":
        handles = legend_handles
        labels = legend_labels
    else:
        # preserve order
        order = ["NMA", "Sham", "Unknown"]
        handles = [legend_handles[g] for g in order if g in legend_handles]
        labels = [g for g in order if g in legend_handles]

    # leave space on the right for legend
    fig.tight_layout(rect=[0, 0, 0.84, 1.0])
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=True, title=None)

    out_path = fig_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------
def tte_wrapper_pre_vs_post(
    decoded_root: Union[str, Path],
    *,
    metadata_excel: Optional[Union[str, Path]] = None,
    bird_treatment_dates: Optional[Dict[str, Union[str, pd.Timestamp]]] = None,
    fig_dir: Optional[Union[str, Path]] = None,
    min_songs_per_day: int = 1,
    show: bool = True,
    # merge / detection knobs (forwarded to TTE_by_day)
    ann_gap_ms: int = 500,
    seg_offset: int = 0,
    merge_repeats: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    dur_merge_gap_ms: int = 500,
    treatment_in: str = "post",
    # broken x-axis behavior
    x_gap_threshold: int = 3,
) -> TTEWrapperResult:
    """
    Run TTE_by_day for all birds under decoded_root, then build:
      1) broken-x all-birds plot colored by BIRD
      2) broken-x all-birds plot colored by TREATMENT GROUP (Sham vs NMA)
      3) grouped pre/post bird-level paired boxplots (unchanged)
    """
    decoded_root = Path(decoded_root)

    # ---------- Collect treatment dates + types ----------
    bird_to_tdate: Dict[str, Union[str, pd.Timestamp]] = {}
    bird_to_ttype: Dict[str, Optional[str]] = {}

    if metadata_excel is not None:
        try:
            md_dates, md_types = _load_treatment_info_from_metadata(metadata_excel)
            bird_to_tdate.update(md_dates)
            bird_to_ttype.update(md_types)
        except Exception as e:  # pragma: no cover
            print(f"WARNING: failed to load metadata from {metadata_excel}: {e}")

    if bird_treatment_dates:
        bird_to_tdate.update(bird_treatment_dates)

    if not bird_to_tdate:
        print(
            "WARNING: No treatment dates found; you must supply either "
            "metadata_excel or bird_treatment_dates. Wrapper may skip all birds."
        )

    # ---------- Figure directory ----------
    if fig_dir is None:
        fig_dir = decoded_root.parent / "tte_wrapper_figures"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Discover per-bird decoded JSONs ----------
    decoded_files = sorted(decoded_root.rglob("*_decoded_database.json"))
    decoded_files = [p for p in decoded_files if fig_dir not in p.parents]

    if not decoded_files:
        print(f"No *_decoded_database.json files found under {decoded_root}.")
        empty = pd.DataFrame()
        return TTEWrapperResult(
            per_bird=[],
            combined_daily_df=empty,
            prepost_summary_df=empty,
            multi_bird_timecourse_path=None,
            multi_bird_timecourse_by_group_path=None,
            prepost_boxplot_path=None,
            prepost_p_value=None,
            group_p_values={},
        )

    per_bird_results: List[BirdDailyTTEResult] = []

    for decoded_path in decoded_files:
        animal_id = _infer_animal_id_from_path(decoded_path)
        t_raw = bird_to_tdate.get(animal_id)
        if t_raw is None:
            print(f"Skipping {decoded_path} (animal_id={animal_id}): no treatment date found.")
            continue

        t_parsed = _parse_treatment_date(t_raw)
        if t_parsed is None:
            print(f"Skipping {decoded_path} (animal_id={animal_id}): could not parse treatment date {t_raw!r}.")
            continue

        t_type = bird_to_ttype.get(animal_id)
        det_path = _find_detection_for_decoded(decoded_path)

        # Per-bird figure directory (per-bird outputs from TTE_by_day)
        bird_fig_dir = fig_dir / animal_id
        bird_fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running TTE_by_day for {animal_id}...")

        def _call_tte(song_det_path: Optional[Path]) -> TTEByDayResult:
            return TTE_by_day(
                decoded_database_json=decoded_path,
                song_detection_json=song_det_path,
                max_gap_between_song_segments=ann_gap_ms,
                segment_index_offset=seg_offset,
                merge_repeated_syllables=merge_repeats,
                repeat_gap_ms=repeat_gap_ms,
                repeat_gap_inclusive=repeat_gap_inclusive,
                merged_song_gap_ms=(None if dur_merge_gap_ms == 0 else dur_merge_gap_ms),
                fig_dir=bird_fig_dir,
                show=False,  # don’t pop per-bird figures
                min_songs_per_day=min_songs_per_day,
                treatment_date=t_parsed,
                treatment_in=treatment_in,
            )

        try:
            tte_res = _call_tte(det_path)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(
                f"  -> ERROR reading song_detection_json for {animal_id} ({det_path}): {e}. "
                "Retrying WITHOUT song_detection_json."
            )
            try:
                tte_res = _call_tte(None)
            except Exception as e2:
                print(f"  -> Failed again without song_detection_json for {animal_id}: {e2}. Skipping.")
                continue
        except Exception as e:
            print(f"  -> Unexpected error in TTE_by_day for {animal_id}: {e}. Skipping.")
            continue

        if tte_res.results_df.empty:
            print(f"  -> No valid days for {animal_id}; skipping.")
            continue

        daily_df = tte_res.results_df.copy()
        tx = pd.Timestamp(t_parsed)
        daily_df["animal_id"] = animal_id
        daily_df["days_from_lesion"] = (daily_df["day"] - tx).dt.days
        daily_df["treatment_type"] = t_type
        daily_df["treatment_group"] = _treatment_group_from_type(t_type)

        per_bird_results.append(
            BirdDailyTTEResult(
                animal_id=animal_id,
                decoded_path=decoded_path,
                song_detection_path=det_path,
                treatment_date=tx,
                treatment_type=t_type,
                tte_result=tte_res,
                daily_df=daily_df,
            )
        )

    if not per_bird_results:
        print("No birds produced usable TTE_by_day results.")
        empty = pd.DataFrame()
        return TTEWrapperResult(
            per_bird=[],
            combined_daily_df=empty,
            prepost_summary_df=empty,
            multi_bird_timecourse_path=None,
            multi_bird_timecourse_by_group_path=None,
            prepost_boxplot_path=None,
            prepost_p_value=None,
            group_p_values={},
        )

    # -----------------------------------------------------------------
    # Combine daily data across birds
    # -----------------------------------------------------------------
    combined_daily = pd.concat([b.daily_df for b in per_bird_results], ignore_index=True)

    # -----------------------------------------------------------------
    # Plot A: Multi-bird broken-x, colored by bird (DOT markers, no labels)
    # -----------------------------------------------------------------
    multi_bird_path = _plot_all_birds_broken_x(
        combined_daily,
        fig_dir=fig_dir,
        out_name="TTE_all_birds_brokenx_colored_by_bird.png",
        title="Transition Entropy by day (all birds)",
        color_mode="bird",
        gap_threshold=x_gap_threshold,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        show=show,
    )

    # -----------------------------------------------------------------
    # Plot B: Multi-bird broken-x, colored by treatment group (Sham vs NMA)
    # -----------------------------------------------------------------
    multi_bird_group_path = _plot_all_birds_broken_x(
        combined_daily,
        fig_dir=fig_dir,
        out_name="TTE_all_birds_brokenx_colored_by_group.png",
        title="Transition Entropy by day (all birds) — colored by Sham vs NMA",
        color_mode="treatment_group",
        gap_threshold=x_gap_threshold,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        show=show,
    )

    # -----------------------------------------------------------------
    # Build pre vs post summary per bird
    # -----------------------------------------------------------------
    rows = []
    for b in per_bird_results:
        pre_mean, post_mean = _compute_pre_post_means(b.daily_df, b.treatment_date)
        if pre_mean is None or post_mean is None:
            print(f"Skipping {b.animal_id} in pre/post summary: missing pre or post days.")
            continue
        rows.append(
            {
                "animal_id": b.animal_id,
                "pre_mean_TTE": pre_mean,
                "post_mean_TTE": post_mean,
                "treatment_type": b.treatment_type,
            }
        )
    prepost_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["animal_id", "pre_mean_TTE", "post_mean_TTE", "treatment_type"]
    )

    # -----------------------------------------------------------------
    # Plot 2: Grouped Pre vs Post paired boxplots (bird-level means)
    # -----------------------------------------------------------------
    prepost_path: Optional[Path] = None
    p_value: Optional[float] = None
    group_p_values: Dict[str, Optional[float]] = {}

    if not prepost_df.empty:
        all_pre_vals = prepost_df["pre_mean_TTE"].astype(float).to_list()
        all_post_vals = prepost_df["post_mean_TTE"].astype(float).to_list()
        if _HAVE_SCIPY and len(all_pre_vals) >= 2:
            try:
                _, p_value = _scipy_stats.wilcoxon(all_pre_vals, all_post_vals)
            except Exception:
                p_value = None

        group_series = prepost_df["treatment_type"].fillna("Unknown")
        groups = sorted(group_series.unique())
        n_groups = len(groups)

        fig2, axes = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 5), sharey=True)
        if n_groups == 1:
            axes = [axes]

        marker_cycle = ["o", "P", "s", "D", "^", "v", "<", ">", "X", "*", "h", "H"]

        def _p_to_stars(p: float) -> str:
            return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))

        for ax, grp in zip(axes, groups):
            gdf = prepost_df[group_series == grp]
            if gdf.empty:
                ax.set_visible(False)
                group_p_values[grp] = None
                continue

            pre_vals = gdf["pre_mean_TTE"].astype(float).to_list()
            post_vals = gdf["post_mean_TTE"].astype(float).to_list()
            bird_ids = gdf["animal_id"].astype(str).to_list()

            x_pre, x_post = 1.0, 2.0
            handles = []

            for i, (bid, pre, post) in enumerate(zip(bird_ids, pre_vals, post_vals)):
                marker = marker_cycle[i % len(marker_cycle)]
                ax.plot([x_pre, x_post], [pre, post], color="0.7", linewidth=1.5, zorder=1)
                ax.scatter([x_pre], [pre], marker=marker, color="tab:blue", edgecolors="none", zorder=2)
                ax.scatter([x_post], [post], marker=marker, color="tab:red", edgecolors="none", zorder=2)
                handles.append(plt.Line2D([0], [0], marker=marker, linestyle="none", color="tab:blue", label=bid))

            bp = ax.boxplot(
                [pre_vals, post_vals],
                positions=[x_pre, x_post],
                widths=0.35,
                showfliers=False,
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("tab:blue"); bp["boxes"][0].set_alpha(0.2)
            bp["boxes"][1].set_facecolor("tab:red");  bp["boxes"][1].set_alpha(0.2)

            ax.set_xticks([x_pre, x_post])
            ax.set_xticklabels(["Pre lesion", "Post lesion"])
            ax.set_ylabel("Total Transition Entropy" if ax is axes[0] else "")
            ax.set_title(f"{grp} (n = {len(bird_ids)} birds)")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(top=False, right=False)

            ax.legend(handles=handles, title="Bird", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

            if _HAVE_SCIPY and len(pre_vals) >= 2:
                try:
                    _, g_p = _scipy_stats.wilcoxon(pre_vals, post_vals)
                except Exception:
                    g_p = None
            else:
                g_p = None
            group_p_values[grp] = g_p

            if g_p is not None:
                y_max = max(max(pre_vals), max(post_vals))
                y_min = min(min(pre_vals), min(post_vals))
                height = y_max + 0.08 * (y_max - y_min)
                h = 0.03 * (y_max - y_min)
                ax.plot([x_pre, x_pre, x_post, x_post], [height, height + h, height + h, height], color="k", linewidth=1.5)
                ax.text((x_pre + x_post) / 2.0, height + h * 1.2, _p_to_stars(g_p), ha="center", va="bottom")

        fig2.suptitle(f"Total Transition Entropy (n = {len(prepost_df)} birds)", y=1.02)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        prepost_path = fig_dir / "TTE_pre_vs_post_birds_boxplot.png"
        fig2.savefig(prepost_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig2)

    return TTEWrapperResult(
        per_bird=per_bird_results,
        combined_daily_df=combined_daily,
        prepost_summary_df=prepost_df,
        multi_bird_timecourse_path=(Path(multi_bird_path) if multi_bird_path else None),
        multi_bird_timecourse_by_group_path=(Path(multi_bird_group_path) if multi_bird_group_path else None),
        prepost_boxplot_path=prepost_path,
        prepost_p_value=p_value,
        group_p_values=group_p_values,
    )


"""
Example usage (Spyder console)
------------------------------

from pathlib import Path
import importlib
import tte_wrapper_pre_vs_post as tw

importlib.reload(tw)

decoded_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_excel = decoded_root / "Area_X_lesion_metadata.xlsx"
fig_root = decoded_root.parent / "tte_summary_figs"

res = tw.tte_wrapper_pre_vs_post(
    decoded_root=decoded_root,
    metadata_excel=metadata_excel,
    fig_dir=fig_root,
    min_songs_per_day=5,
    show=True,
    x_gap_threshold=3,
)

print("Bird-colored:", res.multi_bird_timecourse_path)
print("Group-colored:", res.multi_bird_timecourse_by_group_path)
print("Pre/Post:", res.prepost_boxplot_path)
"""
