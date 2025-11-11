#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Area_X_meta_wrapper.py
"""
Wrapper that:
  • Merges decoded annotations ONCE
  • Merges detected songs (durations) ONCE
then runs FOUR analyses using those merged tables:
  1) Last-syllable histogram + pies  (balanced Early/Late/Post)
  2) Per-target PRECEDER & SUCCESSOR combined panels (balanced groups)
  3) Song-duration comparisons (Pre vs Post; Early-Pre vs Late-Pre vs Post with stats)
  4) Daily syllable-usage heatmaps (linear scale), two variants:
        A) vmax = 1.0 (fixed)
        B) vmax = observed max (data-driven)

Depends on modules in your PYTHONPATH / same folder:
  - merge_annotations_from_split_songs.py      (build_decoded_with_split_labels)
  - merge_potential_split_up_song.py           (build_detected_and_merged_songs)
  - last_syllable_plots.py
  - pre_and_post_syllable_plots.py
  - song_duration_comparison.py
  - syllable_heatmap_linear.py                 (build_daily_avg_count_table, plot_linear_scaled_syllable_counts)

Typical usage (Spyder):

    from pathlib import Path
    import importlib
    import Area_X_meta_wrapper as axmw
    importlib.reload(axmw)

    detect  = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
    decoded = Path("/Volumes/my_own_ssd/2025_areax_lesion/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json")
    tdate   = "2025-05-22"

    out = axmw.run_area_x_meta_bundle(
        song_detection_json=detect,
        decoded_annotations_json=decoded,
        treatment_date=tdate,
        base_output_dir=decoded.parent / "figures",
        # merge knobs (kept consistent across analyses)
        max_gap_between_song_segments=500,
        segment_index_offset=0,
        merge_repeated_syllables=True,
        repeat_gap_ms=10.0,
        repeat_gap_inclusive=False,
        merged_song_gap_ms=500,
        # last-syllable
        top_k_last_labels=15,
        # preceder/successor
        target_labels=None,
        top_k_targets=8,
        top_k_preceders=12,
        top_k_successors=12,
        include_start_as_preceder=False,
        include_end_as_successor=False,
        include_other_bin=True,
        include_other_in_hist=True,
        other_label="Other",
        # heatmaps
        heatmap_cmap="Greys",
        nearest_match=True,
        max_days_off=1,
        # show figures interactively?
        show=True,
    )

    print("Last-syllable hist:", out.last_syllable_hist_path)
    print("Last-syllable pies:", out.last_syllable_pies_path)
    print("Targets used:", out.targets_used)
    print("Per-target panels:", out.per_target_outputs)
    print("Durations pre/post:", out.duration_plots.pre_post_path)
    print("Durations three-group:", out.duration_plots.three_group_path)
    print("Three-group stats:", out.duration_plots.three_group_stats)
    print("Heatmap v1.0:", out.heatmaps.path_v1)
    print("Heatmap vmax=max:", out.heatmaps.path_vmax)
    print("Heatmap vmax_max value:", out.heatmaps.vmax_max)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Core builders (merge once) ---
from merge_annotations_from_split_songs import build_decoded_with_split_labels
from merge_potential_split_up_song import build_detected_and_merged_songs

# --- Analysis modules ---
import last_syllable_plots as lsp
import pre_and_post_syllable_plots as pps
import song_duration_comparison as sdc
import syllable_heatmap_linear as shlin  # provides build_daily_avg_count_table & plot_linear_scaled_syllable_counts


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _as_timestamp(d) -> pd.Timestamp:
    return pd.to_datetime(str(d)).normalize()

def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    """
    Collect unique syllable labels from a dict column and return them in
    numeric order (0,1,2,...) when possible; otherwise lexicographic.
    """
    labs: List[str] = []
    for d in df[dict_col].dropna():
        if isinstance(d, dict):
            labs.extend(list(map(str, d.keys())))
    labs = list(dict.fromkeys(labs))  # preserve first-seen order
    # numeric sort when all labels are integers
    try:
        as_int = [int(x) for x in labs]
        order = np.argsort(as_int)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses for outputs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DurationPlotsOut:
    pre_post_path: Optional[str]
    three_group_path: Optional[str]
    three_group_stats: Dict[str, Any]

@dataclass
class HeatmapOut:
    path_v1: Optional[str]      # vmax=1.0
    path_vmax: Optional[str]    # vmax=max observed
    vmax_max: float

@dataclass
class AreaXMetaResult:
    # Last-syllable results
    last_syllable_hist_path: Optional[str]
    last_syllable_pies_path: Optional[str]
    last_syllable_label_order: List[str]
    last_syllable_counts: Dict[str, int]

    # Preceder/successor results
    targets_used: List[str]
    per_target_outputs: Dict[str, Optional[str]]  # label -> path

    # Duration results
    duration_plots: DurationPlotsOut

    # Heatmaps
    heatmaps: HeatmapOut

    # Shared/merged tables
    merged_annotations_df: pd.DataFrame
    merged_detected_df: pd.DataFrame

    # Balanced splits (from annotations table, for convenience)
    early_df: pd.DataFrame
    late_df: pd.DataFrame
    post_df: pd.DataFrame


# ──────────────────────────────────────────────────────────────────────────────
# Main runner (merge-once → reuse)
# ──────────────────────────────────────────────────────────────────────────────

def run_area_x_meta_bundle(
    *,
    song_detection_json: Union[str, Path],
    decoded_annotations_json: Union[str, Path],
    treatment_date: Union[str, pd.Timestamp],
    base_output_dir: Optional[Union[str, Path]] = None,

    # ---- annotations merge (for syllable analyses) ----
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # ---- detection merge (for duration analyses) ----
    merged_song_gap_ms: Optional[int] = 500,   # None/0 to skip merging

    # ---- last-syllable knobs ----
    top_k_last_labels: int = 15,
    last_hist_filename: str = "last_syllable_hist.png",
    last_pies_filename: str = "last_syllable_pies.png",

    # ---- preceder/successor knobs ----
    target_labels: Optional[List[str]] = None,  # None → auto from balanced pools
    top_k_targets: int = 8,
    top_k_preceders: int = 12,
    top_k_successors: int = 12,
    include_start_as_preceder: bool = False,
    include_end_as_successor: bool = False,
    start_token: str = "<START>",
    end_token: str = "<END>",
    include_other_bin: bool = True,
    include_other_in_hist: bool = True,
    other_label: str = "Other",

    # ---- heatmap knobs ----
    heatmap_cmap: str = "Greys",
    nearest_match: bool = True,
    max_days_off: int = 1,
    heatmap_v1_filename: str = "syllable_heatmap_linear_vmax1.00.png",
    heatmap_vmax_filename: str = "syllable_heatmap_linear_vmaxMAX.png",

    # ---- display ----
    show: bool = True,
) -> AreaXMetaResult:

    song_detection_json = Path(song_detection_json)
    decoded_annotations_json = Path(decoded_annotations_json)
    tdate = _as_timestamp(treatment_date)

    # Base outdir default
    if base_output_dir is None:
        base_output_dir = decoded_annotations_json.parent
    base_output_dir = Path(base_output_dir)
    last_dir = base_output_dir / "last_syllable"
    pps_dir  = base_output_dir / "preceder_successor_panels"
    dur_dir  = base_output_dir / "durations"
    hm_dir   = base_output_dir / "heatmaps"
    for d in (last_dir, pps_dir, dur_dir, hm_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── MERGE ONCE: decoded annotations (for syllable analyses & heatmaps) ──
    ann = build_decoded_with_split_labels(
        decoded_database_json=decoded_annotations_json,
        song_detection_json=song_detection_json,
        only_song_present=True,
        compute_durations=True,
        add_recording_datetime=True,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=max_gap_between_song_segments,
        segment_index_offset=segment_index_offset,
        merge_repeated_syllables=merge_repeated_syllables,
        repeat_gap_ms=repeat_gap_ms,
        repeat_gap_inclusive=repeat_gap_inclusive,
    )
    merged_annotations_df = ann.annotations_appended_df.copy()

    # ── MERGE ONCE: detected songs (for duration analyses) ──
    if merged_song_gap_ms and int(merged_song_gap_ms) > 0:
        det = build_detected_and_merged_songs(
            song_detection_json,
            songs_only=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=int(merged_song_gap_ms),
        )
        merged_detected_df = det["detected_merged_songs"].copy()
        _dur_title_suffix = f"(Merged gap < {int(merged_song_gap_ms)} ms)"
    else:
        merged_detected_df = sdc._prepare_table(song_detection_json, merge_gap_ms=None, songs_only=True)
        _dur_title_suffix = "(Unmerged segments)"

    # ──────────────────────────────────────────────────────────
    # 1) LAST-SYLLABLE PLOTS (using lsp helpers on merged_annotations_df)
    # ──────────────────────────────────────────────────────────
    if merged_annotations_df.empty:
        last_hist_path = None
        last_pies_path = None
        last_label_order: List[str] = []
        last_counts = {"balanced_n": 0, "early_n": 0, "late_n": 0, "post_n": 0}
        early_df = merged_annotations_df.copy()
        late_df  = merged_annotations_df.copy()
        post_df  = merged_annotations_df.copy()
    else:
        dfA = merged_annotations_df.copy()
        if "Recording DateTime" not in dfA.columns and "recording_datetime" in dfA.columns:
            dfA = dfA.rename(columns={"recording_datetime": "Recording DateTime"})

        early_df, late_df, post_df, n_bal = lsp._split_pre_post_balanced(dfA, tdate)

        last_early = lsp._extract_last_syllables(early_df)
        last_late  = lsp._extract_last_syllables(late_df)
        last_post  = lsp._extract_last_syllables(post_df)

        counts_by_group, label_order = lsp._counts_by_group(last_early, last_late, last_post, top_k=top_k_last_labels)
        n_by_group = {
            "Early Pre": int(last_early.shape[0]),
            "Late Pre":  int(last_late.shape[0]),
            "Post":      int(last_post.shape[0]),
        }

        title_base = f"Last syllables per song (balanced groups; Treatment {tdate.date()})"
        hist_title = title_base + f"\n(Top {top_k_last_labels} labels)"
        pies_title = "Last syllables: Early-Pre / Late-Pre / Post"

        last_hist_path = (last_dir / last_hist_filename)
        last_pies_path = (last_dir / last_pies_filename)

        _ = lsp._plot_histogram(
            counts_by_group, label_order, hist_title,
            n_by_group=n_by_group, output_path=last_hist_path, show=show
        )
        _ = lsp._plot_pies(
            counts_by_group, label_order, pies_title,
            n_by_group=n_by_group, output_path=last_pies_path, show=show
        )

        last_label_order = list(label_order)
        last_counts = {"balanced_n": int(n_bal),
                       "early_n": n_by_group["Early Pre"], "late_n": n_by_group["Late Pre"], "post_n": n_by_group["Post"]}

        last_hist_path = str(last_hist_path)
        last_pies_path = str(last_pies_path)

    # ──────────────────────────────────────────────────────────
    # 2) PRECEDER + SUCCESSOR PANELS (pps helpers on merged_annotations_df)
    # ──────────────────────────────────────────────────────────
    targets_used: List[str] = []
    per_target_outputs: Dict[str, Optional[str]] = {}

    if not merged_annotations_df.empty:
        dfB = merged_annotations_df.copy()
        if "Recording DateTime" not in dfB.columns and "recording_datetime" in dfB.columns:
            dfB = dfB.rename(columns={"recording_datetime": "Recording DateTime"})
        e_df, l_df, p_df, _ = pps._balanced_split_by_treatment(dfB, tdate)

        # choose targets
        if target_labels is None:
            cat_df = pd.concat([e_df, l_df, p_df], ignore_index=True)
            targets = pps._select_target_labels_auto(cat_df, top_k_targets=top_k_targets)
        else:
            targets = [str(t) for t in target_labels]
        targets_used = list(targets)

        groups = ["Early Pre", "Late Pre", "Post"]
        title_suffix = f"(balanced Early-Pre / Late-Pre / Post; Treatment {tdate.date()})"

        for tgt in targets:
            # preceders
            e_pre_raw = pps._preceder_counts_for_target(e_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            l_pre_raw = pps._preceder_counts_for_target(l_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            p_pre_raw = pps._preceder_counts_for_target(p_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            order_pre = pps._unified_order(top_k_preceders, e_pre_raw, l_pre_raw, p_pre_raw)

            # successors
            e_suc_raw = pps._successor_counts_for_target(e_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            l_suc_raw = pps._successor_counts_for_target(l_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            p_suc_raw = pps._successor_counts_for_target(p_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            order_suc = pps._unified_order(top_k_successors, e_suc_raw, l_suc_raw, p_suc_raw)

            # optional "Other"
            order_pre_hist = pps._order_with_optional_other(order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                                                            include_other=include_other_in_hist, other_label=other_label)
            order_pre_pie  = pps._order_with_optional_other(order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                                                            include_other=True, other_label=other_label)
            order_suc_hist = pps._order_with_optional_other(order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                                                            include_other=include_other_in_hist, other_label=other_label)
            order_suc_pie  = pps._order_with_optional_other(order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                                                            include_other=True, other_label=other_label)

            def _prep_counts(raw_s: pd.Series, order_for_hist: List[str], order_for_pie: List[str]):
                s_hist = pps._apply_other_bin(raw_s, order_for_hist, include_other=(include_other_in_hist), other_label=other_label)
                s_pie  = pps._apply_other_bin(raw_s, order_for_pie,  include_other=True,                  other_label=other_label)
                return s_hist, s_pie

            e_pre_hist, e_pre_pie = _prep_counts(e_pre_raw, order_pre_hist, order_pre_pie)
            l_pre_hist, l_pre_pie = _prep_counts(l_pre_raw, order_pre_hist, order_pre_pie)
            p_pre_hist, p_pre_pie = _prep_counts(p_pre_raw, order_pre_hist, order_pre_pie)

            e_suc_hist, e_suc_pie = _prep_counts(e_suc_raw, order_suc_hist, order_suc_pie)
            l_suc_hist, l_suc_pie = _prep_counts(l_suc_raw, order_suc_hist, order_suc_pie)
            p_suc_hist, p_suc_pie = _prep_counts(p_suc_raw, order_suc_hist, order_suc_pie)

            counts_pre_by_group_pie  = {"Early Pre": e_pre_pie,  "Late Pre": l_pre_pie,  "Post": p_pre_pie}
            counts_suc_by_group_pie  = {"Early Pre": e_suc_pie,  "Late Pre": l_suc_pie,  "Post": p_suc_pie}

            n_pre_by_group = {g: int(counts_pre_by_group_pie[g].sum()) for g in groups}
            n_suc_by_group = {g: int(counts_suc_by_group_pie[g].sum()) for g in groups}

            # pick histogram orders actually present
            def _pick_order(d: Dict[str, pd.Series]) -> List[str]:
                for g in groups:
                    if len(d[g].index) > 0:
                        return list(d[g].index)
                return []
            order_pre_used_hist = _pick_order({"Early Pre": e_pre_hist, "Late Pre": l_pre_hist, "Post": p_pre_hist})
            order_suc_used_hist = _pick_order({"Early Pre": e_suc_hist, "Late Pre": l_suc_hist, "Post": p_suc_hist})

            out_path = pps_dir / f"preceder_successor_target_{tgt}.png"
            _ = pps._plot_pre_suc_panel(
                target_label=str(tgt),
                title_suffix=title_suffix,
                counts_pre_by_group=counts_pre_by_group_pie,
                order_pre=order_pre_used_hist,
                n_pre_by_group=n_pre_by_group,
                counts_suc_by_group=counts_suc_by_group_pie,
                order_suc=order_suc_used_hist,
                n_suc_by_group=n_suc_by_group,
                include_other_in_hist=include_other_in_hist,
                other_label=other_label,
                output_path=out_path, show=show,
            )
            per_target_outputs[str(tgt)] = str(out_path)

    # ──────────────────────────────────────────────────────────
    # 3) SONG-DURATION PLOTS (reuse merged_detected_df; use sdc public plotters)
    # ──────────────────────────────────────────────────────────
    pre_df, post_df = sdc._split_pre_post(merged_detected_df, treatment_date=tdate)
    pre_s  = sdc._durations_seconds(pre_df)
    post_s = sdc._durations_seconds(post_df)

    dur_prepost_path = dur_dir / "song_duration_pre_post.png"
    _plot1 = sdc.plot_pre_post_boxplot(
        pre_s, post_s,
        treatment_date=tdate,
        merge_gap_ms=merged_song_gap_ms,
        output_path=dur_prepost_path,
        show=show,
        title=f"Song durations Pre vs Post (Treatment: {tdate.date()})\n{_dur_title_suffix}",
    )

    e3, l3, p3, _n = sdc._split_three_groups_balanced(pre_df, post_df)
    eS, lS, pS = sdc._durations_seconds(e3), sdc._durations_seconds(l3), sdc._durations_seconds(p3)

    dur_three_path = dur_dir / "song_duration_three_group.png"
    _plot2 = sdc.plot_three_group_boxplot(
        eS, lS, pS,
        treatment_date=tdate,
        merge_gap_ms=merged_song_gap_ms,
        output_path=dur_three_path,
        show=show,
        title=f"Song durations: Early Pre / Late Pre / Post (Treatment: {tdate.date()})\n{_dur_title_suffix}",
        do_stats=True,
    )

    duration_out = DurationPlotsOut(
        pre_post_path=str(dur_prepost_path),
        three_group_path=str(dur_three_path),
        three_group_stats=_plot2.get("stats", {}),
    )

    # ──────────────────────────────────────────────────────────
    # 4) HEATMAPS (reuse merged_annotations_df; no re-merge)
    # ──────────────────────────────────────────────────────────
    if merged_annotations_df.empty:
        hm_v1_path = None
        hm_vmax_path = None
        vmax_max = float("nan")
    else:
        dfH = merged_annotations_df.copy()

        # Ensure a simple calendar Date column
        if "Date" not in dfH.columns:
            dt_col = "Recording DateTime" if "Recording DateTime" in dfH.columns else "recording_datetime"
            if dt_col not in dfH.columns:
                raise ValueError("No datetime column found to derive 'Date' (expected 'Recording DateTime' or 'recording_datetime').")
            dfH["Date"] = pd.to_datetime(dfH[dt_col]).dt.date

        # Build label list and daily avg count table
        label_col = "syllable_onsets_offsets_ms_dict"
        syllable_labels = _collect_unique_labels_sorted(dfH, label_col)
        count_table = shlin.build_daily_avg_count_table(
            dfH,
            label_column=label_col,
            date_column="Date",
            syllable_labels=syllable_labels,
        )

        # File names (animal id helper from your plotting module)
        animal_id = (shlin._infer_animal_id(dfH, decoded_annotations_json) or decoded_annotations_json.stem or "unknown_animal")
        hm_v1_path   = hm_dir / f"{animal_id}_{heatmap_v1_filename}"
        hm_vmax_path = hm_dir / f"{animal_id}_{heatmap_vmax_filename}"

        # vmax for Plot B
        raw_max = float(np.nanmax(count_table.to_numpy())) if count_table.size else 1.0
        vmax_max = raw_max if np.isfinite(raw_max) and raw_max > 0 else 1.0

        # Plot A: fixed vmax=1.0
        _ = shlin.plot_linear_scaled_syllable_counts(
            count_table,
            animal_id=animal_id,
            treatment_date=tdate,
            save_path=hm_v1_path,
            show=show,
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=1.0,
            nearest_match=nearest_match,
            max_days_off=max_days_off,
        )

        # Plot B: vmax = observed maximum
        _ = shlin.plot_linear_scaled_syllable_counts(
            count_table,
            animal_id=animal_id,
            treatment_date=tdate,
            save_path=hm_vmax_path,
            show=show,
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=vmax_max,
            nearest_match=nearest_match,
            max_days_off=max_days_off,
        )

        hm_v1_path = str(hm_v1_path)
        hm_vmax_path = str(hm_vmax_path)

    heatmaps_out = HeatmapOut(path_v1=hm_v1_path, path_vmax=hm_vmax_path, vmax_max=float(vmax_max))

    # Build final result
    return AreaXMetaResult(
        last_syllable_hist_path=last_hist_path,
        last_syllable_pies_path=last_pies_path,
        last_syllable_label_order=last_label_order,
        last_syllable_counts=last_counts,
        targets_used=targets_used,
        per_target_outputs=per_target_outputs,
        duration_plots=duration_out,
        heatmaps=heatmaps_out,
        merged_annotations_df=merged_annotations_df,
        merged_detected_df=merged_detected_df,
        early_df=early_df,
        late_df=late_df,
        post_df=post_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Area X meta wrapper: merge once, run last-syllable, preceder/successor, duration, and heatmap plots.")
    p.add_argument("--song-detection", "-d", type=str, required=True, help="Path to *_song_detection.json")
    p.add_argument("--annotations", "-a", type=str, required=True, help="Path to *_decoded_database.json")
    p.add_argument("--treatment-date", "-t", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--base-out", type=str, default=None, help="Base output directory (default: annotations.parent)")

    # annotations merge
    p.add_argument("--ann-gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")

    # detection merge (durations)
    p.add_argument("--merged-song-gap-ms", type=int, default=500, help="Set 0/None to disable merging durations.")

    # last-syllable
    p.add_argument("--top-k-last", type=int, default=15)
    p.add_argument("--last-hist-name", type=str, default="last_syllable_hist.png")
    p.add_argument("--last-pies-name", type=str, default="last_syllable_pies.png")

    # preceder/successor
    p.add_argument("--targets", nargs="*", default=None, help="Explicit target labels; omit to auto-select.")
    p.add_argument("--top-k-targets", type=int, default=8)
    p.add_argument("--top-k-preceders", type=int, default=12)
    p.add_argument("--top-k-successors", type=int, default=12)
    p.add_argument("--include-start", action="store_true")
    p.add_argument("--include-end", action="store_true")
    p.add_argument("--include-other", action="store_true")
    p.add_argument("--include-other-in-hist", action="store_true")
    p.add_argument("--other-label", type=str, default="Other")

    # heatmaps
    p.add_argument("--heatmap-cmap", type=str, default="Greys")
    p.add_argument("--nearest-match", action="store_true")
    p.add_argument("--max-days-off", type=int, default=1)
    p.add_argument("--heatmap-v1-name", type=str, default="syllable_heatmap_linear_vmax1.00.png")
    p.add_argument("--heatmap-vmax-name", type=str, default="syllable_heatmap_linear_vmaxMAX.png")

    p.add_argument("--no-show", action="store_true")

    args = p.parse_args()

    res = run_area_x_meta_bundle(
        song_detection_json=args.song_detection,
        decoded_annotations_json=args.annotations,
        treatment_date=args.treatment_date,
        base_output_dir=args.base_out,
        # annotations merge
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        # detection merge
        merged_song_gap_ms=args.merged_song_gap_ms,
        # last-syllable
        top_k_last_labels=args.top_k_last,
        last_hist_filename=args.last_hist_name,
        last_pies_filename=args.last_pies_name,
        # preceder/successor
        target_labels=args.targets,
        top_k_targets=args.top_k_targets,
        top_k_preceders=args.top_k_preceders,
        top_k_successors=args.top_k_successors,
        include_start_as_preceder=args.include_start,
        include_end_as_successor=args.include_end,
        include_other_bin=args.include_other,
        include_other_in_hist=args.include_other_in_hist,
        other_label=args.other_label,
        # heatmaps
        heatmap_cmap=args.heatmap_cmap,
        nearest_match=args.nearest_match,
        max_days_off=args.max_days_off,
        heatmap_v1_filename=args.heatmap_v1_name,
        heatmap_vmax_filename=args.heatmap_vmax_name,
        # display
        show=not args.no_show,
    )

    print("[OK] Last-syllable hist:", res.last_syllable_hist_path)
    print("[OK] Last-syllable pies:", res.last_syllable_pies_path)
    print("[OK] Targets used:", res.targets_used)
    print("[OK] Per-target panels:", len(res.per_target_outputs))
    print("[OK] Durations Pre/Post:", res.duration_plots.pre_post_path)
    print("[OK] Durations Three-Group:", res.duration_plots.three_group_path)
    print("[OK] Three-Group Stats:", res.duration_plots.three_group_stats)
    print("[OK] Heatmap v1.0:", res.heatmaps.path_v1)
    print("[OK] Heatmap vmax=max:", res.heatmaps.path_vmax)
    print("[OK] Heatmap vmax_max:", res.heatmaps.vmax_max)


"""
from pathlib import Path
import importlib
import Area_X_meta_wrapper as axmw
importlib.reload(axmw)

detect  = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
decoded = Path("/Volumes/my_own_ssd/2025_areax_lesion/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json")
tdate   = "2025-05-22"

res = axmw.run_area_x_meta_bundle(
    song_detection_json=detect,
    decoded_annotations_json=decoded,
    treatment_date=tdate,
    base_output_dir=decoded.parent / "figures",
    max_gap_between_song_segments=500,
    segment_index_offset=0,
    merge_repeated_syllables=True,
    repeat_gap_ms=10.0,
    repeat_gap_inclusive=False,
    merged_song_gap_ms=500,
    top_k_last_labels=15,
    target_labels=None,
    top_k_targets=8,
    top_k_preceders=12,
    top_k_successors=12,
    include_start_as_preceder=False,
    include_end_as_successor=False,
    include_other_bin=True,
    include_other_in_hist=True,
    other_label="Other",
    heatmap_cmap="Greys",
    nearest_match=True,
    max_days_off=1,
    show=True,
)

print("Last-syllable hist:", res.last_syllable_hist_path)
print("Last-syllable pies:", res.last_syllable_pies_path)
print("Durations pre/post:", res.duration_plots.pre_post_path)
print("Durations three-group:", res.duration_plots.three_group_path)
print("Heatmap v1.0:", res.heatmaps.path_v1)
print("Heatmap vmax=max:", res.heatmaps.path_vmax)
print("vmax_max used:", res.heatmaps.vmax_max)

"""