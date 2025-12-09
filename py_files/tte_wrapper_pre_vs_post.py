#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tte_wrapper_pre_vs_post.py

Wrapper around tte_by_day.TTE_by_day to:

1) Run TTE_by_day on all birds found under a root directory.
2) Collect daily aggregated TTE per bird.
3) Build:
    - A multi-bird TTE vs days-from-lesion time-course plot.
    - Grouped pre vs post paired boxplots (one subplot per treatment type).

Assumptions
-----------
- Each bird has at least one file matching "*_decoded_database.json"
  somewhere under `decoded_root` (searched recursively).

- A treatment date must be available per bird, either via:
    * metadata_excel with columns "Animal ID", "Treatment date",
      and "Treatment type", or
    * an explicit bird_treatment_dates dict: {"USA5283": "2024-03-05", ...}

- A song-detection JSON is optional but recommended. The wrapper searches
  for (in order):
    * "<stem>_song_detection.json" in the same folder as the decoded JSON
      (where stem is everything before "_decoded_database.json").
    * If not found, any "*song_detection*.json" in the same folder if there
      is exactly one candidate.

The per-bird daily TTE is taken from TTE_by_day.results_df["agg_TTE"].
For the cross-bird pre/post plot, each bird contributes:
    pre_mean  = mean(agg_TTE on days < lesion date)
    post_mean = mean(agg_TTE on days >= lesion date),

Plots are stratified by the metadata "Treatment type" (e.g.,
"Bilateral NMA lesion injections" vs "Bilateral saline sham injection").

Outputs
-------
TTEWrapperResult with:
- per_bird:           list of per-bird daily results
- combined_daily_df:  daily TTE across all birds
- prepost_summary_df: one row per bird with pre/post means + treatment_type
- multi_bird_timecourse_path: path to time-course PNG
- prepost_boxplot_path:       path to grouped boxplot PNG
- prepost_p_value:    overall Wilcoxon p (all birds pooled)
- group_p_values:     dict {treatment_type -> Wilcoxon p for that group}
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
    daily_df: pd.DataFrame  # results_df with extra columns: animal_id, days_from_lesion, treatment_type


@dataclass
class TTEWrapperResult:
    per_bird: List[BirdDailyTTEResult]
    combined_daily_df: pd.DataFrame  # all birds, one row per day
    prepost_summary_df: pd.DataFrame  # one row per bird (pre_mean, post_mean, treatment_type)
    multi_bird_timecourse_path: Optional[Path]
    prepost_boxplot_path: Optional[Path]
    prepost_p_value: Optional[float]             # overall (all birds) paired p-value
    group_p_values: Dict[str, Optional[float]]   # per-treatment-type p-values


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

    This first tries organize_metadata_excel.build_areax_metadata if available.
    If that fails, it falls back to reading the Excel directly with pandas.
    """
    metadata_excel = Path(metadata_excel)

    def _from_df(dfm: pd.DataFrame) -> Tuple[Dict[str, Union[str, pd.Timestamp]], Dict[str, Optional[str]]]:
        if "Animal ID" in dfm.columns:
            id_col = "Animal ID"
        else:
            id_col = dfm.columns[0]

        # Find date column
        tcol = None
        for cand in ["Treatment date", "Treatment_date", "treatment_date"]:
            if cand in dfm.columns:
                tcol = cand
                break
        if tcol is None:
            raise KeyError("Could not find a 'Treatment date' column in metadata Excel.")

        # Find treatment type column (optional but strongly expected)
        type_col = None
        for cand in ["Treatment type", "Treatment_type", "treatment_type"]:
            if cand in dfm.columns:
                type_col = cand
                break

        bird_to_date: Dict[str, Union[str, pd.Timestamp]] = {}
        bird_to_type: Dict[str, Optional[str]] = {}

        # We may have multiple rows per animal (one per injection); just take the first non-NA
        for aid, sub in dfm.groupby(id_col):
            aid_str = str(aid)
            # treatment date
            tvals = sub[tcol].dropna()
            if len(tvals) > 0:
                bird_to_date[aid_str] = tvals.iloc[0]
            # treatment type
            if type_col is not None:
                type_vals = sub[type_col].dropna()
                bird_to_type[aid_str] = type_vals.iloc[0] if len(type_vals) > 0 else None
            else:
                bird_to_type[aid_str] = None

        return bird_to_date, bird_to_type

    # Try using build_areax_metadata if available
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
                    # date
                    date_val = None
                    for key in ["Treatment date", "Treatment_date", "treatment_date"]:
                        if key in info and pd.notna(info[key]):
                            date_val = info[key]
                            break
                    if date_val is not None:
                        bird_to_date[aid_str] = date_val
                    # type
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
            # Fall through to direct Excel read
            pass

    # Fallback: read Excel directly
    dfm = pd.read_excel(metadata_excel)
    return _from_df(dfm)


def _find_detection_for_decoded(decoded_path: Path) -> Optional[Path]:
    """
    Try to find the matching *_song_detection.json for a given *_decoded_database.json.
    """
    parent = decoded_path.parent
    name = decoded_path.name

    # 1) Direct string replacement
    if name.endswith("_decoded_database.json"):
        stem = name[: -len("_decoded_database.json")]
        candidate = parent / f"{stem}_song_detection.json"
        if candidate.exists():
            return candidate

    # 2) Any "*song_detection*.json" in the same folder, if unique
    candidates = list(parent.glob("*song_detection*.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_animal_id_from_path(decoded_path: Path) -> str:
    """
    Infer an animal_id from the decoded JSON path, using the same logic
    as tte_by_day._extract_id_from_text where possible.
    """
    text = decoded_path.stem
    mid = _extract_id_from_text(text)
    if mid:
        return mid
    # fallback: first chunk before "_"
    return text.split("_")[0]


def _compute_pre_post_means(
    daily_df: pd.DataFrame, treatment_date: pd.Timestamp
) -> Tuple[Optional[float], Optional[float]]:
    """
    Given one bird's daily_df, compute:
        pre_mean  = mean(agg_TTE on days < treatment_date)
        post_mean = mean(agg_TTE on days >= treatment_date)
    Returns (pre_mean, post_mean), each possibly None if no days in that side.
    """
    tx = pd.Timestamp(treatment_date)
    pre_mask = daily_df["day"] < tx
    post_mask = daily_df["day"] >= tx

    pre_vals = daily_df.loc[pre_mask, "agg_TTE"]
    post_vals = daily_df.loc[post_mask, "agg_TTE"]

    pre_mean = float(pre_vals.mean()) if len(pre_vals) > 0 else None
    post_mean = float(post_vals.mean()) if len(post_vals) > 0 else None
    return pre_mean, post_mean


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
    label_n_songs: bool = False,
    # merge / detection knobs (forwarded to TTE_by_day)
    ann_gap_ms: int = 500,
    seg_offset: int = 0,
    merge_repeats: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    dur_merge_gap_ms: int = 500,
    treatment_in: str = "post",
) -> TTEWrapperResult:
    """
    Run TTE_by_day for all birds under decoded_root, then build:
        1) Multi-bird TTE time-course figure.
        2) Grouped bird-level pre vs post paired boxplots.

    Parameters
    ----------
    decoded_root : str or Path
        Root directory containing *_decoded_database.json files
        (searched recursively with rglob).

    metadata_excel : str or Path, optional
        Path to Area X metadata Excel. Used to populate:
            - bird -> treatment date mapping
            - bird -> treatment type mapping

    bird_treatment_dates : dict, optional
        Explicit mapping {animal_id: treatment_date}. Entries here override
        metadata-derived dates if both are provided.

    fig_dir : str or Path, optional
        Directory to write the multi-bird figures. If None, a folder
        "tte_wrapper_figures" is created as a *sibling* of decoded_root,
        i.e., decoded_root.parent / "tte_wrapper_figures". This keeps
        figures out of the data tree.

    min_songs_per_day : int, default 1
        Forwarded to TTE_by_day.

    show : bool, default True
        Whether to show the combined figures (per-bird figures are
        controlled by TTE_by_day(show=...)).

    label_n_songs : bool, default False
        If True and 'n_songs' is present in TTE_by_day output, label each
        point in the time-course plot with the number of songs that day.
        Default False (no labels) for a cleaner plot.

    ann_gap_ms, seg_offset, merge_repeats, repeat_gap_ms,
    repeat_gap_inclusive, dur_merge_gap_ms, treatment_in :
        Forwarded to TTE_by_day for each bird.

    Returns
    -------
    TTEWrapperResult
        Holds per-bird results, combined daily_df, combined pre/post
        summary, plot paths, and p-values.
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
        # Explicit mapping overrides anything from metadata (for dates only)
        bird_to_tdate.update(bird_treatment_dates)

    if not bird_to_tdate:
        print(
            "WARNING: No treatment dates found; you must supply either "
            "metadata_excel or bird_treatment_dates. Wrapper may skip all birds."
        )

    # ---------- Figure directory ----------
    if fig_dir is None:
        # Place figures OUTSIDE decoded_root by default
        fig_dir = decoded_root.parent / "tte_wrapper_figures"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Discover per-bird decoded JSONs ----------
    decoded_files = sorted(decoded_root.rglob("*_decoded_database.json"))

    # Safety: exclude any decoded files that might live inside fig_dir
    decoded_files = [p for p in decoded_files if fig_dir not in p.parents]

    if not decoded_files:
        print(f"No *_decoded_database.json files found under {decoded_root}.")
        empty = pd.DataFrame()
        return TTEWrapperResult(
            per_bird=[],
            combined_daily_df=empty,
            prepost_summary_df=empty,
            multi_bird_timecourse_path=None,
            prepost_boxplot_path=None,
            prepost_p_value=None,
            group_p_values={},
        )

    per_bird_results: List[BirdDailyTTEResult] = []

    for decoded_path in decoded_files:
        animal_id = _infer_animal_id_from_path(decoded_path)
        t_raw = bird_to_tdate.get(animal_id)
        if t_raw is None:
            print(
                f"Skipping {decoded_path} (animal_id={animal_id}): "
                f"no treatment date found in metadata / bird_treatment_dates."
            )
            continue

        t_parsed = _parse_treatment_date(t_raw)
        if t_parsed is None:
            print(
                f"Skipping {decoded_path} (animal_id={animal_id}): "
                f"could not parse treatment date {t_raw!r}."
            )
            continue

        t_type = bird_to_ttype.get(animal_id)

        det_path = _find_detection_for_decoded(decoded_path)

        # Per-bird figure directory
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
                merged_song_gap_ms=(
                    None if dur_merge_gap_ms == 0 else dur_merge_gap_ms
                ),
                fig_dir=bird_fig_dir,
                show=False,  # per-bird plots: don't pop up unless you want to change this
                min_songs_per_day=min_songs_per_day,
                treatment_date=t_parsed,
                treatment_in=treatment_in,
            )

        try:
            tte_res = _call_tte(det_path)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(
                f"  -> ERROR reading song_detection_json for {animal_id} "
                f"({det_path}): {e}. "
                "Retrying TTE_by_day WITHOUT song_detection_json "
                "(no split-song merging)."
            )
            try:
                tte_res = _call_tte(None)
            except Exception as e2:
                print(
                    f"  -> Failed again without song_detection_json for {animal_id}: {e2}. "
                    "Skipping this bird."
                )
                continue
        except Exception as e:
            print(
                f"  -> Unexpected error in TTE_by_day for {animal_id}: {e}. Skipping."
            )
            continue

        if tte_res.results_df.empty:
            print(f"  -> No valid days for {animal_id}; skipping.")
            continue

        daily_df = tte_res.results_df.copy()
        tx = pd.Timestamp(t_parsed)
        daily_df["animal_id"] = animal_id
        daily_df["days_from_lesion"] = (daily_df["day"] - tx).dt.days
        daily_df["treatment_type"] = t_type

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
            prepost_boxplot_path=None,
            prepost_p_value=None,
            group_p_values={},
        )

    # -----------------------------------------------------------------
    # Combine daily data across birds
    # -----------------------------------------------------------------
    combined_daily = pd.concat([b.daily_df for b in per_bird_results], ignore_index=True)

    # -----------------------------------------------------------------
    # Plot 1: Multi-bird TTE time-course (days-from-lesion)
    # -----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    birds = sorted({b.animal_id for b in per_bird_results})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {bid: color_cycle[i % len(color_cycle)] for i, bid in enumerate(birds)}

    for bid in birds:
        dfb = combined_daily[combined_daily["animal_id"] == bid].copy()
        dfb = dfb.sort_values("days_from_lesion")

        ax1.plot(
            dfb["days_from_lesion"],
            dfb["agg_TTE"],
            marker="o",
            linestyle="-",
            linewidth=2,
            color=color_map[bid],
            label=bid,
        )
        # Optional: label each point with n_songs (default is OFF)
        if label_n_songs and "n_songs" in dfb.columns:
            for x, y, n in zip(dfb["days_from_lesion"], dfb["agg_TTE"], dfb["n_songs"]):
                ax1.text(x, y, str(int(n)), fontsize=8, ha="center", va="bottom")

    ax1.axvline(0, linestyle="--", color="k", linewidth=1.5)
    ax1.set_xlabel("Days relative to lesion (0 = lesion day)")
    ax1.set_ylabel("Total Transition Entropy")
    ax1.set_title("Total Transition Entropy by day (all birds)")
    ax1.legend(title="Bird", bbox_to_anchor=(1.02, 1), loc="upper left")

    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.tick_params(top=False, right=False)

    fig1.tight_layout()
    multi_bird_path = fig_dir / "TTE_multi_bird_timecourse.png"
    fig1.savefig(multi_bird_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # -----------------------------------------------------------------
    # Build pre vs post summary per bird
    # -----------------------------------------------------------------
    rows = []
    for b in per_bird_results:
        pre_mean, post_mean = _compute_pre_post_means(b.daily_df, b.treatment_date)
        if pre_mean is None or post_mean is None:
            print(
                f"Skipping {b.animal_id} in pre/post summary: "
                f"missing pre or post days."
            )
            continue
        rows.append(
            {
                "animal_id": b.animal_id,
                "pre_mean_TTE": pre_mean,
                "post_mean_TTE": post_mean,
                "treatment_type": b.treatment_type,
            }
        )

    if rows:
        prepost_df = pd.DataFrame(rows)
    else:
        prepost_df = pd.DataFrame(
            columns=["animal_id", "pre_mean_TTE", "post_mean_TTE", "treatment_type"]
        )

    # -----------------------------------------------------------------
    # Plot 2: Grouped Pre vs Post paired boxplots (bird-level means)
    # -----------------------------------------------------------------
    prepost_path: Optional[Path] = None
    p_value: Optional[float] = None  # overall (all birds) paired test
    group_p_values: Dict[str, Optional[float]] = {}

    if not prepost_df.empty:
        # Overall paired stat across ALL birds (regardless of group)
        all_pre_vals = prepost_df["pre_mean_TTE"].astype(float).to_list()
        all_post_vals = prepost_df["post_mean_TTE"].astype(float).to_list()
        if _HAVE_SCIPY and len(all_pre_vals) >= 2:
            try:
                _, p_value = _scipy_stats.wilcoxon(all_pre_vals, all_post_vals)
            except Exception:
                p_value = None

        # Grouping by treatment_type (fall back to a single "All" group if missing)
        if "treatment_type" not in prepost_df.columns:
            prepost_df["treatment_type"] = "All"
        group_series = prepost_df["treatment_type"].fillna("Unknown")
        groups = sorted(group_series.unique())
        n_groups = len(groups)

        fig2, axes = plt.subplots(
            1, n_groups, figsize=(5.5 * n_groups, 5), sharey=True
        )
        if n_groups == 1:
            axes = [axes]

        marker_cycle = ["o", "P", "s", "D", "^", "v", "<", ">", "X", "*", "h", "H"]

        def _p_to_stars(p: float) -> str:
            return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))

        for ax, grp in zip(axes, groups):
            gmask = group_series == grp
            gdf = prepost_df[gmask]
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
                ax.plot(
                    [x_pre, x_post],
                    [pre, post],
                    color="0.7",
                    linewidth=1.5,
                    zorder=1,
                )
                ax.scatter(
                    [x_pre],
                    [pre],
                    marker=marker,
                    color="tab:blue",
                    edgecolors="none",
                    zorder=2,
                )
                ax.scatter(
                    [x_post],
                    [post],
                    marker=marker,
                    color="tab:red",
                    edgecolors="none",
                    zorder=2,
                )
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker=marker,
                        linestyle="none",
                        color="tab:blue",
                        label=bid,
                    )
                )

            # Boxplots for pre/post distributions in this group
            bp = ax.boxplot(
                [pre_vals, post_vals],
                positions=[x_pre, x_post],
                widths=0.35,
                showfliers=False,
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("tab:blue")
            bp["boxes"][0].set_alpha(0.2)
            bp["boxes"][1].set_facecolor("tab:red")
            bp["boxes"][1].set_alpha(0.2)

            ax.set_xticks([x_pre, x_post])
            ax.set_xticklabels(["Pre lesion", "Post lesion"])
            if ax is axes[0]:
                ax.set_ylabel("Total Transition Entropy")
            else:
                ax.set_ylabel("")
            ax.set_title(f"{grp} (n = {len(bird_ids)} birds)")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(top=False, right=False)

            # Legend for birds in this group
            ax.legend(
                handles=handles,
                title="Bird",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
            )

            # Group-specific paired Wilcoxon, if possible
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
                ax.plot(
                    [x_pre, x_pre, x_post, x_post],
                    [height, height + h, height + h, height],
                    color="k",
                    linewidth=1.5,
                )
                label = _p_to_stars(g_p)
                ax.text(
                    (x_pre + x_post) / 2.0,
                    height + h * 1.2,
                    label,
                    ha="center",
                    va="bottom",
                )

        fig2.suptitle(
            f"Total Transition Entropy (n = {len(prepost_df)} birds)", y=1.02
        )
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
        multi_bird_timecourse_path=multi_bird_path,
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

# Put summary figs OUTSIDE decoded_root:
fig_root = decoded_root.parent / "tte_summary_figs"

res = tw.tte_wrapper_pre_vs_post(
    decoded_root=decoded_root,
    metadata_excel=metadata_excel,
    fig_dir=fig_root,
    min_songs_per_day=5,
    show=True,
)

print(res.combined_daily_df.head())
print(res.prepost_summary_df)
print("Timecourse fig:", res.multi_bird_timecourse_path)
print("Pre/Post fig:", res.prepost_boxplot_path)
print("Overall pre/post Wilcoxon p-value (all birds):", res.prepost_p_value)
print("Per-group p-values:", res.group_p_values)
"""
