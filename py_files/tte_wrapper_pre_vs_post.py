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
    - Multi-bird TTE vs days-from-lesion BROKEN x-axis plot (colored by lesion hit-type buckets)
    - Mean±SD timecourse by hit-type bucket (3 lines with shaded SD band):
         * sham saline injection
         * Area X visible (single hit)
         * COMBINED (visible ML + not visible)

4) Pre vs Post paired boxplots (bird-level means) using DAILY agg_TTE (equal weight per date):
    - overall (all birds)
    - three separate figures for the 3 hit-type buckets above

5) NEW: Pre vs Post paired boxplots (bird-level means) using SONG-BALANCED windows
   (equal number of songs pre and post *within each bird*):
    - overall (all birds)
    - three separate figures for the 3 hit-type buckets above

Notes
-----
- Metadata Excel may store treatment date/type and lesion hit type in DIFFERENT sheets.
  This wrapper reads ALL sheets and merges values by Animal ID.
- Requires updated tte_by_day.py that returns `per_song_df` with a `file_TTE` column,
  and exports `select_song_balanced_pre_post`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for paired stats
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Core per-bird TTE function + helpers
from tte_by_day import (
    TTE_by_day,
    TTEByDayResult,
    _extract_id_from_text,
    _parse_treatment_date,
    select_song_balanced_pre_post,  # NEW (Feb 2026)
)

# Optional metadata helper (not required; we also read Excel directly)
try:
    from organize_metadata_excel import build_areax_metadata  # type: ignore
except Exception:  # pragma: no cover
    build_areax_metadata = None  # type: ignore


# ---------------------------------------------------------------------
# Hit-type bucket definitions (match your outlier_graphs naming)
# ---------------------------------------------------------------------
HIT_SHAM = "sham saline injection"
HIT_VISIBLE_SINGLE = "Area X visible (single hit)"
HIT_VISIBLE_ML = "Area X visible (medial+lateral hit)"
HIT_NOT_VISIBLE = "large lesion Area X not visible"
COMBINED_LABEL = "COMBINED (visible ML + not visible)"
OTHER_LABEL = "Other/Unknown"


def _canonical_hit_type(raw: Optional[str]) -> Optional[str]:
    """
    Map messy/variant hit-type strings to canonical labels above.
    Returns None if unknown.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    sl = s.lower()

    # sham
    if ("sham" in sl) or ("saline" in sl) or ("control" in sl):
        return HIT_SHAM

    # not visible / large lesion
    if ("not visible" in sl) or ("large lesion" in sl):
        return HIT_NOT_VISIBLE

    # visible single
    if ("visible" in sl) and ("single" in sl):
        return HIT_VISIBLE_SINGLE

    # visible medial+lateral (accept a few variants)
    if ("visible" in sl) and (
        ("medial" in sl and "lateral" in sl)
        or ("medial+lateral" in sl)
        or ("medial + lateral" in sl)
        or ("ml" in sl and "visible" in sl)
    ):
        return HIT_VISIBLE_ML

    # miss / unknown
    if ("miss" in sl) or ("unknown" in sl):
        return None

    return None


def _hit_type_subset_label(canon: Optional[str]) -> Optional[str]:
    """
    Return one of the three requested buckets:
      - HIT_SHAM
      - HIT_VISIBLE_SINGLE
      - COMBINED_LABEL (HIT_VISIBLE_ML + HIT_NOT_VISIBLE)
    or None if canon should be excluded.
    """
    if canon is None:
        return None
    if canon == HIT_SHAM:
        return HIT_SHAM
    if canon == HIT_VISIBLE_SINGLE:
        return HIT_VISIBLE_SINGLE
    if canon in (HIT_VISIBLE_ML, HIT_NOT_VISIBLE):
        return COMBINED_LABEL
    return None


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
    lesion_hit_type: Optional[str]

    tte_result: TTEByDayResult

    daily_df: pd.DataFrame  # results_df + extras: animal_id, days_from_lesion, treatment_type, treatment_group, lesion_hit_type

    # NEW (song-balanced summary info; computed in wrapper)
    song_balanced_n_eff: Optional[int] = None
    song_balanced_pre_mean: Optional[float] = None
    song_balanced_post_mean: Optional[float] = None


@dataclass
class TTEWrapperResult:
    per_bird: List[BirdDailyTTEResult]
    combined_daily_df: pd.DataFrame

    # Daily-based pre/post summary (bird-level means over DAYS)
    prepost_summary_df: pd.DataFrame

    # NEW: Song-balanced pre/post summary (bird-level means over selected songs)
    prepost_song_balanced_summary_df: pd.DataFrame

    # Timecourse plots
    multi_bird_timecourse_path: Optional[Path]                 # colored by bird
    multi_bird_timecourse_by_group_path: Optional[Path]        # colored by Sham vs NMA
    multi_bird_timecourse_by_hit_type_path: Optional[Path]     # colored by hit-type bucket
    hit_bucket_mean_sd_timecourse_path: Optional[Path]         # mean±SD lines for 3 buckets

    # Daily-based pre/post plots
    prepost_boxplot_path_all: Optional[Path]                   # all birds combined (daily-based)
    prepost_boxplot_paths_by_hit_bucket: Dict[str, Optional[Path]]
    prepost_p_value_all: Optional[float]
    prepost_p_values_by_hit_bucket: Dict[str, Optional[float]]

    # NEW: Song-balanced pre/post plots
    prepost_song_balanced_boxplot_path_all: Optional[Path]
    prepost_song_balanced_boxplot_paths_by_hit_bucket: Dict[str, Optional[Path]]
    prepost_song_balanced_p_value_all: Optional[float]
    prepost_song_balanced_p_values_by_hit_bucket: Dict[str, Optional[float]]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
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


def _load_metadata_from_excel_all_sheets(
    metadata_excel: Union[str, Path]
) -> Tuple[Dict[str, Union[str, pd.Timestamp]], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Read ALL sheets in metadata_excel and build:
        bird_to_date:     {animal_id -> treatment_date}
        bird_to_type:     {animal_id -> treatment_type}
        bird_to_hit_type: {animal_id -> lesion_hit_type}

    Works even if these columns live on different sheets.
    """
    metadata_excel = Path(metadata_excel)

    bird_to_date: Dict[str, Union[str, pd.Timestamp]] = {}
    bird_to_type: Dict[str, Optional[str]] = {}
    bird_to_hit: Dict[str, Optional[str]] = {}

    def _extract_from_df(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        # ID column
        id_col = None
        for cand in ["Animal ID", "AnimalID", "Animal Id", "Animal_ID", "animal_id"]:
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]

        # treatment date col
        tcol = None
        for cand in ["Treatment date", "Treatment_date", "treatment_date", "Tx date", "Surgery date", "Treatment Date"]:
            if cand in df.columns:
                tcol = cand
                break

        # treatment type col
        type_col = None
        for cand in ["Treatment type", "Treatment_type", "treatment_type", "Tx type", "Treatment Type"]:
            if cand in df.columns:
                type_col = cand
                break

        # hit type col
        hit_col = None
        for cand in [
            "Lesion hit type",
            "Hit type",
            "animal_hit_type",
            "Animal hit type",
            "Lesion_hit_type",
            "hit_type",
            "Hit Type",
        ]:
            if cand in df.columns:
                hit_col = cand
                break

        # group and take first non-null per bird
        for aid, sub in df.groupby(id_col):
            aid_str = str(aid).strip()
            if not aid_str:
                continue

            if tcol is not None and aid_str not in bird_to_date:
                tvals = sub[tcol].dropna()
                if len(tvals) > 0:
                    bird_to_date[aid_str] = tvals.iloc[0]

            if type_col is not None and (aid_str not in bird_to_type or bird_to_type.get(aid_str) is None):
                tvals = sub[type_col].dropna()
                if len(tvals) > 0:
                    bird_to_type[aid_str] = tvals.iloc[0]

            if hit_col is not None and (aid_str not in bird_to_hit or bird_to_hit.get(aid_str) is None):
                hvals = sub[hit_col].dropna()
                if len(hvals) > 0:
                    bird_to_hit[aid_str] = hvals.iloc[0]

    # First pass: build_areax_metadata if available
    if build_areax_metadata is not None:
        try:
            meta = build_areax_metadata(metadata_excel)  # type: ignore
            if isinstance(meta, dict):
                for aid, info in meta.items():
                    if not isinstance(info, dict):
                        continue
                    aid_str = str(aid).strip()
                    if not aid_str:
                        continue

                    for key in ["Treatment date", "Treatment_date", "treatment_date"]:
                        if key in info and pd.notna(info[key]) and aid_str not in bird_to_date:
                            bird_to_date[aid_str] = info[key]
                            break

                    for key in ["Treatment type", "Treatment_type", "treatment_type"]:
                        if key in info and pd.notna(info[key]):
                            bird_to_type[aid_str] = info[key]
                            break

                    for key in ["Lesion hit type", "Hit type", "animal_hit_type", "hit_type"]:
                        if key in info and pd.notna(info[key]):
                            bird_to_hit[aid_str] = info[key]
                            break
        except Exception:
            pass

    # Second pass: parse ALL sheets
    xls = pd.ExcelFile(metadata_excel)
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(metadata_excel, sheet_name=sheet)
            _extract_from_df(df)
        except Exception:
            continue

    for aid in list(bird_to_date.keys()):
        bird_to_type.setdefault(aid, None)
        bird_to_hit.setdefault(aid, None)

    return bird_to_date, bird_to_type, bird_to_hit


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


def _compute_pre_post_means_daily(
    daily_df: pd.DataFrame, treatment_date: pd.Timestamp
) -> Tuple[Optional[float], Optional[float]]:
    """Bird-level mean of daily agg_TTE pre (< tx) and post (>= tx)."""
    tx = pd.Timestamp(treatment_date)
    pre_vals = daily_df.loc[daily_df["day"] < tx, "agg_TTE"]
    post_vals = daily_df.loc[daily_df["day"] >= tx, "agg_TTE"]
    pre_mean = float(pre_vals.mean()) if len(pre_vals) > 0 else None
    post_mean = float(post_vals.mean()) if len(post_vals) > 0 else None
    return pre_mean, post_mean


def _compute_pre_post_means_song_balanced(
    tte_res: TTEByDayResult,
    *,
    treatment_date: pd.Timestamp,
    n_songs: int,
    treatment_in: str,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Bird-level mean of per-song file_TTE for:
      - last n_eff pre songs
      - first n_eff post songs
    n_eff is the effective number of songs possible for that bird.
    """
    per_song_df = getattr(tte_res, "per_song_df", None)
    if per_song_df is None or not isinstance(per_song_df, pd.DataFrame) or per_song_df.empty:
        return None, None, 0

    try:
        pre_sel, post_sel, n_eff = select_song_balanced_pre_post(
            per_song_df,
            treatment_date=treatment_date,
            n_songs=int(n_songs),
            treatment_in=str(treatment_in),
        )
    except Exception:
        return None, None, 0

    if n_eff <= 0 or pre_sel.empty or post_sel.empty:
        return None, None, 0

    pre_mean = float(pd.to_numeric(pre_sel["file_TTE"], errors="coerce").dropna().mean())
    post_mean = float(pd.to_numeric(post_sel["file_TTE"], errors="coerce").dropna().mean())
    if not np.isfinite(pre_mean) or not np.isfinite(post_mean):
        return None, None, n_eff
    return pre_mean, post_mean, n_eff


def _wilcoxon_p(pre_vals: List[float], post_vals: List[float]) -> Optional[float]:
    if not _HAVE_SCIPY:
        return None
    if len(pre_vals) < 2:
        return None
    try:
        _, p = _scipy_stats.wilcoxon(pre_vals, post_vals)
        return float(p)
    except Exception:
        return None


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
    xs: List[int] = []
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
    color_mode: str = "bird",  # "bird" | "treatment_group" | "hit_type"
    ycol: str = "agg_TTE",
    xcol: str = "days_from_lesion",
    gap_threshold: int = 3,
    marker: str = "o",
    markersize: float = 4.5,
    linewidth: float = 1.8,
    show: bool = True,
) -> Optional[Path]:
    """
    Broken-x multi-bird plot:
      - color_mode="bird": each bird gets its own color, legend lists birds
      - color_mode="treatment_group": birds colored by Sham vs NMA (legend lists groups)
      - color_mode="hit_type": birds colored by hit-type BUCKET (legend lists 3 buckets + Other/Unknown)
    """
    if combined_daily.empty:
        return None

    segments = _infer_x_segments_from_days(combined_daily[xcol], gap_threshold=gap_threshold, force_zero_split=True)
    if not segments:
        return None

    widths: List[float] = []
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

    for i in range(len(axes) - 1):
        axes[i].spines["right"].set_visible(False)
        axes[i + 1].spines["left"].set_visible(False)
        _add_x_break_marks(axes[i], axes[i + 1], size=0.015, lw=1.5)

    birds = sorted(combined_daily["animal_id"].astype(str).unique().tolist())

    if color_mode == "bird":
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bird_color = {bid: color_cycle[i % len(color_cycle)] for i, bid in enumerate(birds)}
        legend_handles: Union[List[object], Dict[str, object]] = []
        legend_labels: List[str] = []

    elif color_mode == "treatment_group":
        group_color = {"NMA": "tab:blue", "Sham": "tab:orange", "Unknown": "0.5"}
        legend_handles = {}  # one handle per group
        legend_labels = []

    elif color_mode == "hit_type":
        hit_color = {
            HIT_SHAM: "tab:red",
            HIT_VISIBLE_SINGLE: "tab:purple",
            COMBINED_LABEL: "tab:blue",
            OTHER_LABEL: "0.5",
        }
        legend_handles = {}  # one handle per bucket
        legend_labels = []

    else:
        raise ValueError(f"Unknown color_mode: {color_mode!r}")

    for bid in birds:
        dfb = combined_daily[combined_daily["animal_id"].astype(str) == bid].copy()
        dfb = dfb.sort_values(xcol)

        if color_mode == "bird":
            col = bird_color[bid]
            label_for_legend = bid

        elif color_mode == "treatment_group":
            grp = _treatment_group_from_type(
                dfb["treatment_type"].iloc[0] if "treatment_type" in dfb.columns else None
            )
            col = group_color.get(grp, "0.5")
            label_for_legend = grp

        else:  # hit_type
            raw_hit = dfb["lesion_hit_type"].iloc[0] if "lesion_hit_type" in dfb.columns else None
            canon = _canonical_hit_type(None if pd.isna(raw_hit) else str(raw_hit))
            bucket = _hit_type_subset_label(canon)
            label_for_legend = bucket if bucket is not None else OTHER_LABEL
            col = hit_color.get(label_for_legend, "0.5")

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

            if first_segment_for_this_line:
                if color_mode == "bird":
                    legend_handles.append(ln)  # type: ignore
                    legend_labels.append(label_for_legend)
                else:
                    if label_for_legend not in legend_handles:  # type: ignore
                        legend_handles[label_for_legend] = ln  # type: ignore
                first_segment_for_this_line = False

    # mark lesion boundary
    for (xmin, xmax), ax in zip(segments, axes):
        if xmin <= 0 <= xmax or xmax == 0:
            ax.axvline(0, linestyle="--", linewidth=1.5, color="0.6")
            ax.text(0, y_hi, "pre lesion", rotation=90, va="top", ha="right", fontsize=10, color="k")
            ax.text(0, y_hi, "\npost lesion", rotation=90, va="top", ha="left", fontsize=10, color="k")
            break

    axes[0].set_ylabel("Transition Entropy")
    fig.supxlabel("Days from lesion")
    fig.suptitle(title, y=1.02)

    if color_mode == "bird":
        handles = legend_handles  # type: ignore
        labels = legend_labels
    elif color_mode == "treatment_group":
        order = ["NMA", "Sham", "Unknown"]
        handles = [legend_handles[g] for g in order if g in legend_handles]  # type: ignore
        labels = [g for g in order if g in legend_handles]  # type: ignore
    else:  # hit_type
        order = [HIT_SHAM, HIT_VISIBLE_SINGLE, COMBINED_LABEL, OTHER_LABEL]
        handles = [legend_handles[g] for g in order if g in legend_handles]  # type: ignore
        labels = [g for g in order if g in legend_handles]  # type: ignore

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
# Mean ± SD timecourse by hit-type bucket (3 lines, shaded SD)
# ---------------------------------------------------------------------
def _split_on_gaps(xs: np.ndarray, ys: np.ndarray, sds: np.ndarray, *, gap_threshold: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split arrays into segments when x gaps exceed gap_threshold."""
    if xs.size == 0:
        return []
    segs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, xs.size):
        if int(xs[i] - xs[i - 1]) > int(gap_threshold):
            segs.append((xs[start:i], ys[start:i], sds[start:i]))
            start = i
    segs.append((xs[start:], ys[start:], sds[start:]))
    return segs


def _plot_hit_bucket_mean_sd_timecourse(
    combined_daily: pd.DataFrame,
    *,
    fig_dir: Path,
    out_name: str,
    title: str,
    xcol: str = "days_from_lesion",
    ycol: str = "agg_TTE",
    gap_threshold: int = 3,
    show: bool = True,
) -> Optional[Path]:
    """
    Plot 3 lines (mean across birds per day) with shaded ±1 SD across birds:
      - sham saline injection
      - Area X visible (single hit)
      - COMBINED (visible ML + not visible)
    """
    if combined_daily.empty:
        return None

    df = combined_daily.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])

    # assign bucket
    df["hit_bucket"] = df["lesion_hit_type"].apply(
        lambda x: _hit_type_subset_label(_canonical_hit_type(None if pd.isna(x) else str(x)))
    )
    df = df[df["hit_bucket"].isin([HIT_SHAM, HIT_VISIBLE_SINGLE, COMBINED_LABEL])].copy()
    if df.empty:
        return None

    # aggregate across birds per (bucket, day)
    g = (
        df.groupby(["hit_bucket", xcol])[ycol]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values([xcol])
    )
    if g.empty:
        return None

    # use default cycle (no hard-coded colors)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    bucket_order = [HIT_SHAM, HIT_VISIBLE_SINGLE, COMBINED_LABEL]
    bucket_color = {b: cycle[i % len(cycle)] for i, b in enumerate(bucket_order)}

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for b in bucket_order:
        gb = g[g["hit_bucket"] == b].copy()
        if gb.empty:
            continue

        xs = gb[xcol].astype(int).to_numpy()
        ys = gb["mean"].astype(float).to_numpy()
        sds = gb["std"].astype(float).to_numpy()
        sds = np.where(np.isfinite(sds), sds, 0.0)  # if n=1 -> std NaN -> show no band

        segs = _split_on_gaps(xs, ys, sds, gap_threshold=gap_threshold)

        for j, (xseg, yseg, sdseg) in enumerate(segs):
            if xseg.size == 0:
                continue
            label = b if j == 0 else None
            line = ax.plot(xseg, yseg, marker="o", linewidth=2.0, label=label, color=bucket_color[b])[0]
            ax.fill_between(
                xseg,
                yseg - sdseg,
                yseg + sdseg,
                alpha=0.2,
                color=line.get_color(),
                linewidth=0,
            )

    ax.axvline(0, linestyle="--", linewidth=1.5, color="0.6")
    ax.set_xlabel("Days from lesion")
    ax.set_ylabel("Mean daily TTE (agg_TTE)")
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)

    fig.tight_layout(rect=[0, 0, 0.82, 1.0])
    fig.legend(loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=True, title=None)

    out_path = fig_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Pre/Post paired boxplot helper (generic columns)
# ---------------------------------------------------------------------
def _plot_prepost_paired_box(
    df: pd.DataFrame,
    *,
    fig_dir: Path,
    out_name: str,
    title: str,
    pre_col: str,
    post_col: str,
    ylabel: str,
    show: bool,
    p_text_prefix: str = "Wilcoxon",
) -> Tuple[Optional[Path], Optional[float]]:
    """
    df must have columns: animal_id, <pre_col>, <post_col>
    Returns (path, wilcoxon_p).
    """
    if df.empty:
        return None, None

    pre_vals = pd.to_numeric(df[pre_col], errors="coerce").to_list()
    post_vals = pd.to_numeric(df[post_col], errors="coerce").to_list()
    bird_ids = df["animal_id"].astype(str).to_list()

    # filter finite pairs
    pairs = [(bid, pre, post) for bid, pre, post in zip(bird_ids, pre_vals, post_vals) if np.isfinite(pre) and np.isfinite(post)]
    if len(pairs) == 0:
        return None, None

    bird_ids = [p[0] for p in pairs]
    pre_vals_f = [float(p[1]) for p in pairs]
    post_vals_f = [float(p[2]) for p in pairs]

    p_val = _wilcoxon_p(pre_vals_f, post_vals_f)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    marker_cycle = ["o", "P", "s", "D", "^", "v", "<", ">", "X", "*", "h", "H"]
    for i, (bid, pre, post) in enumerate(zip(bird_ids, pre_vals_f, post_vals_f)):
        mk = marker_cycle[i % len(marker_cycle)]
        ax.plot([1, 2], [pre, post], color="0.7", linewidth=1.5, zorder=1)
        ax.scatter([1], [pre], marker=mk, edgecolors="none", zorder=2)
        ax.scatter([2], [post], marker=mk, edgecolors="none", zorder=2)

    bp = ax.boxplot(
        [pre_vals_f, post_vals_f],
        positions=[1, 2],
        widths=0.35,
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre lesion", "Post lesion"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)

    handles = [
        plt.Line2D([0], [0], marker=marker_cycle[i % len(marker_cycle)], linestyle="none", label=bid)
        for i, bid in enumerate(bird_ids)
    ]
    fig.tight_layout(rect=[0, 0, 0.80, 1.0])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=True, title="Bird")

    if p_val is not None:
        ax.text(
            0.02, 0.02,
            f"{p_text_prefix} p={p_val:.3g}",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=10,
            color="k",
        )

    out_path = fig_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path, p_val


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
    # broken x-axis / gap behavior
    x_gap_threshold: int = 3,
    # NEW: song-balanced selection
    n_songs_balanced: int = 100,
) -> TTEWrapperResult:
    """
    Run TTE_by_day for all birds under decoded_root, then build:

      Timecourse figures:
        1) broken-x all-birds plot colored by BIRD
        2) broken-x all-birds plot colored by TREATMENT GROUP (Sham vs NMA)
        3) broken-x all-birds plot colored by HIT-TYPE bucket
        4) mean±SD timecourse plot (3 hit-type buckets)

      Pre/Post (daily-based):
        5) paired pre/post plots (bird-level mean of DAILY agg_TTE):
             - overall (all birds)
             - 3 requested hit-type bucket figures

      Pre/Post (song-balanced):
        6) paired pre/post plots (bird-level mean of per-song file_TTE for last N pre and first N post):
             - overall (all birds)
             - 3 requested hit-type bucket figures
    """
    decoded_root = Path(decoded_root)

    # ---------- Collect treatment dates + types + hit types ----------
    bird_to_tdate: Dict[str, Union[str, pd.Timestamp]] = {}
    bird_to_ttype: Dict[str, Optional[str]] = {}
    bird_to_hit: Dict[str, Optional[str]] = {}

    if metadata_excel is not None:
        try:
            md_dates, md_types, md_hits = _load_metadata_from_excel_all_sheets(metadata_excel)
            bird_to_tdate.update(md_dates)
            bird_to_ttype.update(md_types)
            bird_to_hit.update(md_hits)
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
            prepost_song_balanced_summary_df=empty,
            multi_bird_timecourse_path=None,
            multi_bird_timecourse_by_group_path=None,
            multi_bird_timecourse_by_hit_type_path=None,
            hit_bucket_mean_sd_timecourse_path=None,
            prepost_boxplot_path_all=None,
            prepost_boxplot_paths_by_hit_bucket={},
            prepost_p_value_all=None,
            prepost_p_values_by_hit_bucket={},
            prepost_song_balanced_boxplot_path_all=None,
            prepost_song_balanced_boxplot_paths_by_hit_bucket={},
            prepost_song_balanced_p_value_all=None,
            prepost_song_balanced_p_values_by_hit_bucket={},
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
        hit_raw = bird_to_hit.get(animal_id)
        det_path = _find_detection_for_decoded(decoded_path)

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
        daily_df["lesion_hit_type"] = hit_raw  # keep raw; canonicalize only when needed

        # NEW: song-balanced per-bird summary (mean of file_TTE over last/first N songs)
        sb_pre_mean, sb_post_mean, sb_n_eff = _compute_pre_post_means_song_balanced(
            tte_res,
            treatment_date=tx,
            n_songs=int(n_songs_balanced),
            treatment_in=str(treatment_in),
        )

        per_bird_results.append(
            BirdDailyTTEResult(
                animal_id=animal_id,
                decoded_path=decoded_path,
                song_detection_path=det_path,
                treatment_date=tx,
                treatment_type=t_type,
                lesion_hit_type=hit_raw,
                tte_result=tte_res,
                daily_df=daily_df,
                song_balanced_n_eff=(sb_n_eff if sb_n_eff > 0 else None),
                song_balanced_pre_mean=sb_pre_mean,
                song_balanced_post_mean=sb_post_mean,
            )
        )

    if not per_bird_results:
        print("No birds produced usable TTE_by_day results.")
        empty = pd.DataFrame()
        return TTEWrapperResult(
            per_bird=[],
            combined_daily_df=empty,
            prepost_summary_df=empty,
            prepost_song_balanced_summary_df=empty,
            multi_bird_timecourse_path=None,
            multi_bird_timecourse_by_group_path=None,
            multi_bird_timecourse_by_hit_type_path=None,
            hit_bucket_mean_sd_timecourse_path=None,
            prepost_boxplot_path_all=None,
            prepost_boxplot_paths_by_hit_bucket={},
            prepost_p_value_all=None,
            prepost_p_values_by_hit_bucket={},
            prepost_song_balanced_boxplot_path_all=None,
            prepost_song_balanced_boxplot_paths_by_hit_bucket={},
            prepost_song_balanced_p_value_all=None,
            prepost_song_balanced_p_values_by_hit_bucket={},
        )

    combined_daily = pd.concat([b.daily_df for b in per_bird_results], ignore_index=True)

    # Plot A: colored by bird
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

    # Plot B: colored by Sham vs NMA
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

    # Plot C: colored by hit-type bucket (3 buckets + Other)
    multi_bird_hit_path = _plot_all_birds_broken_x(
        combined_daily,
        fig_dir=fig_dir,
        out_name="TTE_all_birds_brokenx_colored_by_hit_type.png",
        title="Transition Entropy by day (all birds) — colored by lesion hit type",
        color_mode="hit_type",
        gap_threshold=x_gap_threshold,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        show=show,
    )

    # Plot D: mean ± SD timecourse for the 3 buckets
    mean_sd_path = _plot_hit_bucket_mean_sd_timecourse(
        combined_daily,
        fig_dir=fig_dir,
        out_name="TTE_hitBuckets_mean_with_SD_band.png",
        title="Mean daily TTE by lesion hit type (±1 SD across birds)",
        gap_threshold=x_gap_threshold,
        show=show,
    )

    # -----------------------------------------------------------------
    # Pre vs post summary per bird (DAILY-based: bird-level mean over days)
    # -----------------------------------------------------------------
    rows_daily = []
    for b in per_bird_results:
        pre_mean, post_mean = _compute_pre_post_means_daily(b.daily_df, b.treatment_date)
        if pre_mean is None or post_mean is None:
            print(f"Skipping {b.animal_id} in DAILY pre/post summary: missing pre or post days.")
            continue
        rows_daily.append(
            {
                "animal_id": b.animal_id,
                "pre_mean_TTE": pre_mean,
                "post_mean_TTE": post_mean,
                "treatment_type": b.treatment_type,
                "treatment_group": _treatment_group_from_type(b.treatment_type),
                "lesion_hit_type": b.lesion_hit_type,
            }
        )

    prepost_df_daily = pd.DataFrame(rows_daily) if rows_daily else pd.DataFrame(
        columns=["animal_id", "pre_mean_TTE", "post_mean_TTE", "treatment_type", "treatment_group", "lesion_hit_type"]
    )

    # -----------------------------------------------------------------
    # NEW: Pre vs post summary per bird (SONG-balanced: bird-level mean over selected songs)
    # -----------------------------------------------------------------
    rows_song = []
    for b in per_bird_results:
        if b.song_balanced_pre_mean is None or b.song_balanced_post_mean is None or not b.song_balanced_n_eff:
            print(f"Skipping {b.animal_id} in SONG-balanced pre/post summary: insufficient songs or missing per_song_df.")
            continue
        rows_song.append(
            {
                "animal_id": b.animal_id,
                "pre_mean_song_TTE": float(b.song_balanced_pre_mean),
                "post_mean_song_TTE": float(b.song_balanced_post_mean),
                "n_songs_used_per_side": int(b.song_balanced_n_eff),
                "n_songs_requested": int(n_songs_balanced),
                "treatment_type": b.treatment_type,
                "treatment_group": _treatment_group_from_type(b.treatment_type),
                "lesion_hit_type": b.lesion_hit_type,
            }
        )

    prepost_df_song = pd.DataFrame(rows_song) if rows_song else pd.DataFrame(
        columns=[
            "animal_id",
            "pre_mean_song_TTE",
            "post_mean_song_TTE",
            "n_songs_used_per_side",
            "n_songs_requested",
            "treatment_type",
            "treatment_group",
            "lesion_hit_type",
        ]
    )

    # -----------------------------------------------------------------
    # Pre/Post plots: DAILY-based
    # -----------------------------------------------------------------
    prepost_all_path: Optional[Path] = None
    prepost_all_p: Optional[float] = None
    if not prepost_df_daily.empty:
        prepost_all_path, prepost_all_p = _plot_prepost_paired_box(
            prepost_df_daily,
            fig_dir=fig_dir,
            out_name="TTE_pre_vs_post_ALL_birds_boxplot_DAILY.png",
            title=f"Total Transition Entropy — Pre vs Post (daily means; n={len(prepost_df_daily)} birds)",
            pre_col="pre_mean_TTE",
            post_col="post_mean_TTE",
            ylabel="Total Transition Entropy (agg_TTE; mean across days)",
            show=show,
            p_text_prefix="Wilcoxon",
        )

    bucket_paths_daily: Dict[str, Optional[Path]] = {}
    bucket_ps_daily: Dict[str, Optional[float]] = {}

    if not prepost_df_daily.empty:
        canon = prepost_df_daily["lesion_hit_type"].apply(lambda x: _canonical_hit_type(None if pd.isna(x) else str(x)))
        bucket = canon.apply(_hit_type_subset_label)
        tmp = prepost_df_daily.copy()
        tmp["hit_bucket"] = bucket

        for label, out_name in [
            (HIT_SHAM, "TTE_pre_vs_post_hitBucket_SHAM_DAILY.png"),
            (HIT_VISIBLE_SINGLE, "TTE_pre_vs_post_hitBucket_VISIBLE_SINGLE_DAILY.png"),
            (COMBINED_LABEL, "TTE_pre_vs_post_hitBucket_COMBINED_DAILY.png"),
        ]:
            sub = tmp[tmp["hit_bucket"] == label].copy()
            if sub.empty:
                bucket_paths_daily[label] = None
                bucket_ps_daily[label] = None
                continue

            pth, pv = _plot_prepost_paired_box(
                sub,
                fig_dir=fig_dir,
                out_name=out_name,
                title=f"Total Transition Entropy — Pre vs Post (daily means; {label}; n={len(sub)} birds)",
                pre_col="pre_mean_TTE",
                post_col="post_mean_TTE",
                ylabel="Total Transition Entropy (agg_TTE; mean across days)",
                show=show,
                p_text_prefix="Wilcoxon",
            )
            bucket_paths_daily[label] = (Path(pth) if pth else None)
            bucket_ps_daily[label] = pv

    # -----------------------------------------------------------------
    # NEW: Pre/Post plots: SONG-balanced
    # -----------------------------------------------------------------
    prepost_song_all_path: Optional[Path] = None
    prepost_song_all_p: Optional[float] = None
    if not prepost_df_song.empty:
        used = prepost_df_song["n_songs_used_per_side"].astype(int)
        used_summary = f"requested {n_songs_balanced}; used median={int(used.median())}, min={int(used.min())}, max={int(used.max())}"
        prepost_song_all_path, prepost_song_all_p = _plot_prepost_paired_box(
            prepost_df_song,
            fig_dir=fig_dir,
            out_name="TTE_pre_vs_post_ALL_birds_boxplot_SONG_BALANCED.png",
            title=f"Total Transition Entropy — Pre vs Post (song-balanced; {used_summary}; n={len(prepost_df_song)} birds)",
            pre_col="pre_mean_song_TTE",
            post_col="post_mean_song_TTE",
            ylabel="Per-song Total Transition Entropy (file_TTE; mean of selected songs)",
            show=show,
            p_text_prefix="Wilcoxon",
        )

    bucket_paths_song: Dict[str, Optional[Path]] = {}
    bucket_ps_song: Dict[str, Optional[float]] = {}

    if not prepost_df_song.empty:
        canon = prepost_df_song["lesion_hit_type"].apply(lambda x: _canonical_hit_type(None if pd.isna(x) else str(x)))
        bucket = canon.apply(_hit_type_subset_label)
        tmp = prepost_df_song.copy()
        tmp["hit_bucket"] = bucket

        for label, out_name in [
            (HIT_SHAM, "TTE_pre_vs_post_hitBucket_SHAM_SONG_BALANCED.png"),
            (HIT_VISIBLE_SINGLE, "TTE_pre_vs_post_hitBucket_VISIBLE_SINGLE_SONG_BALANCED.png"),
            (COMBINED_LABEL, "TTE_pre_vs_post_hitBucket_COMBINED_SONG_BALANCED.png"),
        ]:
            sub = tmp[tmp["hit_bucket"] == label].copy()
            if sub.empty:
                bucket_paths_song[label] = None
                bucket_ps_song[label] = None
                continue

            used = sub["n_songs_used_per_side"].astype(int)
            used_summary = f"requested {n_songs_balanced}; used median={int(used.median())}, min={int(used.min())}, max={int(used.max())}"
            pth, pv = _plot_prepost_paired_box(
                sub,
                fig_dir=fig_dir,
                out_name=out_name,
                title=f"Total Transition Entropy — Pre vs Post (song-balanced; {used_summary}; {label}; n={len(sub)} birds)",
                pre_col="pre_mean_song_TTE",
                post_col="post_mean_song_TTE",
                ylabel="Per-song Total Transition Entropy (file_TTE; mean of selected songs)",
                show=show,
                p_text_prefix="Wilcoxon",
            )
            bucket_paths_song[label] = (Path(pth) if pth else None)
            bucket_ps_song[label] = pv

    return TTEWrapperResult(
        per_bird=per_bird_results,
        combined_daily_df=combined_daily,
        prepost_summary_df=prepost_df_daily,
        prepost_song_balanced_summary_df=prepost_df_song,
        multi_bird_timecourse_path=(Path(multi_bird_path) if multi_bird_path else None),
        multi_bird_timecourse_by_group_path=(Path(multi_bird_group_path) if multi_bird_group_path else None),
        multi_bird_timecourse_by_hit_type_path=(Path(multi_bird_hit_path) if multi_bird_hit_path else None),
        hit_bucket_mean_sd_timecourse_path=(Path(mean_sd_path) if mean_sd_path else None),
        prepost_boxplot_path_all=prepost_all_path,
        prepost_boxplot_paths_by_hit_bucket=bucket_paths_daily,
        prepost_p_value_all=prepost_all_p,
        prepost_p_values_by_hit_bucket=bucket_ps_daily,
        prepost_song_balanced_boxplot_path_all=(Path(prepost_song_all_path) if prepost_song_all_path else None),
        prepost_song_balanced_boxplot_paths_by_hit_bucket=bucket_paths_song,
        prepost_song_balanced_p_value_all=prepost_song_all_p,
        prepost_song_balanced_p_values_by_hit_bucket=bucket_ps_song,
    )


"""
Example usage (Spyder console)
------------------------------

from pathlib import Path
import importlib
import tte_wrapper_pre_vs_post as tw

importlib.reload(tw)

decoded_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_excel = decoded_root / "Area_X_lesion_metadata_with_hit_types.xlsx"
fig_root = decoded_root.parent / "tte_summary_figs_hit_types"

res = tw.tte_wrapper_pre_vs_post(
    decoded_root=decoded_root,
    metadata_excel=metadata_excel,
    fig_dir=fig_root,
    min_songs_per_day=5,
    n_songs_balanced=1000,   # NEW: last 100 pre, first 100 post (per bird, if available)
    show=True,
    x_gap_threshold=3,
)

print("Bird-colored:", res.multi_bird_timecourse_path)
print("Group-colored:", res.multi_bird_timecourse_by_group_path)
print("Hit-type-colored:", res.multi_bird_timecourse_by_hit_type_path)
print("Hit-bucket mean±SD:", res.hit_bucket_mean_sd_timecourse_path)

print("DAILY pre/post (all):", res.prepost_boxplot_path_all)
print("DAILY pre/post (by bucket):", res.prepost_boxplot_paths_by_hit_bucket)

print("SONG-balanced pre/post (all):", res.prepost_song_balanced_boxplot_path_all)
print("SONG-balanced pre/post (by bucket):", res.prepost_song_balanced_boxplot_paths_by_hit_bucket)

# Inspect summary tables:
print(res.prepost_summary_df.head())
print(res.prepost_song_balanced_summary_df.head())
"""
