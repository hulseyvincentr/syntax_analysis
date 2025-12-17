# Area_X_wrapper.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper that:
  • Merges decoded annotations ONCE
  • Merges detected songs (durations) ONCE
then runs analyses using those merged tables.

NEW (batch mode):
  • Optionally pass a JSON ROOT directory containing subfolders per animal_id
  • Provide an Excel metadata file to look up treatment dates
  • Iterate across all animal folders and run the full bundle per bird
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Iterable

import numpy as np
import pandas as pd

# --- Core builders (merge once) ---
from merge_annotations_from_split_songs import build_decoded_with_split_labels
from merge_potential_split_up_song import build_detected_and_merged_songs

# --- Analysis modules ---
import last_syllable_plots as lsp
import pre_and_post_syllable_plots as pps
import song_duration_comparison as sdc
import syllable_heatmap_linear as shlin
import tte_by_day as tte
import phrase_duration_pre_vs_post_grouped as pdpg

# Optional: daily phrase duration plots
try:
    import phrase_duration_over_days as gpday
    _HAS_GPDAY = True
except Exception:
    _HAS_GPDAY = False


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _as_timestamp(d) -> pd.Timestamp:
    """Normalize to midnight Timestamp."""
    return pd.to_datetime(str(d)).normalize()


def _excel_serial_to_timestamp(x: Any) -> Optional[pd.Timestamp]:
    """
    Convert Excel serial date (days since 1899-12-30) to Timestamp.
    Returns None if not convertible.
    """
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # Excel serials are often floats/ints like 45424.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        # Heuristic: Excel serials for modern dates are typically > 20000
        if float(x) > 1000:
            try:
                return pd.to_datetime(float(x), unit="D", origin="1899-12-30").normalize()
            except Exception:
                return None
    return None


def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    """
    Collect unique syllable labels from a dict column and return them in
    numeric order (0,1,2,...) when possible; otherwise lexicographic.
    """
    labs: List[str] = []
    if dict_col in df.columns:
        for d in df[dict_col].dropna():
            if isinstance(d, dict):
                labs.extend(list(map(str, d.keys())))
    labs = list(dict.fromkeys(labs))  # preserve first-seen order
    try:
        as_int = [int(x) for x in labs]
        order = np.argsort(as_int)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)


def _iter_animal_dirs(json_root_dir: Path) -> List[Path]:
    """Immediate subdirectories only (each should be an animal_id folder)."""
    if not json_root_dir.exists():
        raise FileNotFoundError(f"json_root_dir does not exist: {json_root_dir}")
    return sorted([p for p in json_root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _pick_single_match(paths: List[Path], kind: str, animal_dir: Path) -> Path:
    if len(paths) == 0:
        raise FileNotFoundError(f"Could not find {kind} JSON in: {animal_dir}")
    if len(paths) == 1:
        return paths[0]
    # If multiple, prefer ones that contain the folder name as prefix
    folder_id = animal_dir.name
    pref = [p for p in paths if p.name.startswith(folder_id + "_")]
    if len(pref) == 1:
        return pref[0]
    # Otherwise pick shortest filename to reduce “backup copies” odds
    return sorted(paths, key=lambda p: len(p.name))[0]


def _find_required_jsons(animal_dir: Path) -> Tuple[Path, Path]:
    """
    Find (decoded_database_json, song_detection_json) inside an animal folder.
    Robust to naming variations.
    """
    # Common patterns
    decoded_candidates = list(animal_dir.glob("*decoded_database*.json")) + list(animal_dir.glob("*decoded*database*.json"))
    detect_candidates  = list(animal_dir.glob("*song_detection*.json")) + list(animal_dir.glob("*song*detect*.json"))

    # Fallback: any json containing keywords
    if not decoded_candidates:
        decoded_candidates = [p for p in animal_dir.glob("*.json") if ("decoded" in p.name.lower() and "database" in p.name.lower())]
    if not detect_candidates:
        detect_candidates = [p for p in animal_dir.glob("*.json") if ("song" in p.name.lower() and "detect" in p.name.lower())]

    decoded_json = _pick_single_match(decoded_candidates, "decoded_database", animal_dir)
    detect_json  = _pick_single_match(detect_candidates, "song_detection", animal_dir)
    return decoded_json, detect_json


def _normalize_colname(s: str) -> str:
    return "".join(str(s).strip().lower().split())


def _auto_find_metadata_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to find (animal_id_col, treatment_date_col) from common names.
    Raises ValueError if not found.
    """
    cols = list(df.columns)
    norm = {_normalize_colname(c): c for c in cols}

    # Animal ID candidates
    animal_keys = [
        "animalid", "animal_id", "birdid", "bird_id", "id", "subjectid", "subject_id"
    ]
    # Treatment date candidates
    date_keys = [
        "treatmentdate", "treatment_date", "surgerydate", "surgery_date", "lesiondate", "lesion_date", "injdate", "inj_date"
    ]

    def _find(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in norm:
                return norm[k]
        # fuzzy contains
        for nk, orig in norm.items():
            for k in keys:
                if k in nk:
                    return orig
        return None

    id_col = _find(animal_keys)
    date_col = _find(date_keys)

    if id_col is None or date_col is None:
        raise ValueError(
            "Could not auto-detect required metadata columns.\n"
            f"Columns found: {cols}\n"
            "Please pass id_col=... and treatment_date_col=... explicitly."
        )
    return id_col, date_col


def load_treatment_dates_from_excel(
    metadata_excel: Union[str, Path],
    *,
    sheet_name: Union[str, int] = 0,
    id_col: Optional[str] = None,
    treatment_date_col: Optional[str] = None,
) -> Dict[str, pd.Timestamp]:
    """
    Read metadata Excel and return mapping: animal_id -> treatment_date (Timestamp normalized).
    Handles strings, datetimes, and Excel serial date numbers.
    """
    metadata_excel = Path(metadata_excel)
    if not metadata_excel.exists():
        raise FileNotFoundError(f"metadata_excel does not exist: {metadata_excel}")

    df = pd.read_excel(metadata_excel, sheet_name=sheet_name)

    if id_col is None or treatment_date_col is None:
        auto_id_col, auto_date_col = _auto_find_metadata_columns(df)
        id_col = id_col or auto_id_col
        treatment_date_col = treatment_date_col or auto_date_col

    out: Dict[str, pd.Timestamp] = {}
    for _, row in df.iterrows():
        aid = row.get(id_col, None)
        tval = row.get(treatment_date_col, None)
        if aid is None or (isinstance(aid, float) and np.isnan(aid)):
            continue
        aid_str = str(aid).strip()

        # Convert treatment date value
        ts = None
        ts = _excel_serial_to_timestamp(tval)
        if ts is None:
            try:
                ts = pd.to_datetime(tval).normalize()
            except Exception:
                ts = None

        if ts is None or pd.isna(ts):
            continue
        out[aid_str] = ts

    return out


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
class TTEPlotsOut:
    timecourse_path: Optional[str]
    three_group_path: Optional[str]         # clean (no stats panel)
    three_group_stats_path: Optional[str]   # right-hand stats panel
    prepost_p_value: Optional[float]
    prepost_test: Optional[str]

@dataclass
class PhraseDurationPlotsOut:
    paths: Dict[str, Optional[str]]
    syllable_labels: List[str]
    y_limits: Tuple[float, float]

@dataclass
class DailyPhraseDurationOut:
    output_dir: Optional[str]
    num_plots: int
    sample_paths: List[str]

@dataclass
class AreaXMetaResult:
    last_syllable_hist_path: Optional[str]
    last_syllable_pies_path: Optional[str]
    last_syllable_label_order: List[str]
    last_syllable_counts: Dict[str, int]

    targets_used: List[str]
    per_target_outputs: Dict[str, Optional[str]]  # label -> path

    duration_plots: DurationPlotsOut
    heatmaps: HeatmapOut

    daily_phrase_duration: DailyPhraseDurationOut
    phrase_duration_plots: PhraseDurationPlotsOut

    tte_plots: TTEPlotsOut

    merged_annotations_df: pd.DataFrame
    merged_detected_df: pd.DataFrame

    early_df: pd.DataFrame
    late_df: pd.DataFrame
    post_df: pd.DataFrame


@dataclass
class AreaXBatchResult:
    """
    Results of running across a json_root_dir.
    - results: animal_id -> AreaXMetaResult (only for successes)
    - summary_df: one row per attempted animal with status + key paths
    - errors: animal_id -> error string (only for failures)
    """
    results: Dict[str, AreaXMetaResult]
    summary_df: pd.DataFrame
    errors: Dict[str, str]


# ──────────────────────────────────────────────────────────────────────────────
# Main runner (single animal): merge-once → reuse
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

    # ---- TTE knobs ----
    tte_min_songs_per_day: int = 1,
    tte_treatment_in: str = "post",  # "post", "pre", or "exclude"

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
    pd_dir   = base_output_dir / "phrase_durations"
    pd_daily_dir = base_output_dir / "phrase_duration_daily"
    tte_dir  = base_output_dir / "tte"
    for d in (last_dir, pps_dir, dur_dir, hm_dir, pd_dir, pd_daily_dir, tte_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── MERGE ONCE: decoded annotations ──
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

    # ── MERGE ONCE: detected songs (durations) ──
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

    # Infer animal ID once for consistent naming across analyses
    _animal_id = (shlin._infer_animal_id(merged_annotations_df, decoded_annotations_json)
                  or decoded_annotations_json.stem or "unknown_animal")

    # ──────────────────────────────────────────────────────────
    # 1) LAST-SYLLABLE PLOTS
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

        last_hist_path_p = (last_dir / last_hist_filename)
        last_pies_path_p = (last_dir / last_pies_filename)

        _ = lsp._plot_histogram(
            counts_by_group, label_order, hist_title,
            n_by_group=n_by_group, output_path=last_hist_path_p, show=show
        )
        _ = lsp._plot_pies(
            counts_by_group, label_order, pies_title,
            n_by_group=n_by_group, output_path=last_pies_path_p, show=show
        )

        last_label_order = list(label_order)
        last_counts = {"balanced_n": int(n_bal),
                       "early_n": n_by_group["Early Pre"], "late_n": n_by_group["Late Pre"], "post_n": n_by_group["Post"]}

        last_hist_path = str(last_hist_path_p)
        last_pies_path = str(last_pies_path_p)

    # ──────────────────────────────────────────────────────────
    # 2) PRECEDER + SUCCESSOR PANELS
    # ──────────────────────────────────────────────────────────
    targets_used: List[str] = []
    per_target_outputs: Dict[str, Optional[str]] = {}

    if not merged_annotations_df.empty:
        dfB = merged_annotations_df.copy()
        if "Recording DateTime" not in dfB.columns and "recording_datetime" in dfB.columns:
            dfB = dfB.rename(columns={"recording_datetime": "Recording DateTime"})
        e_df, l_df, p_df, _ = pps._balanced_split_by_treatment(dfB, tdate)

        if target_labels is None:
            cat_df = pd.concat([e_df, l_df, p_df], ignore_index=True)
            targets = pps._select_target_labels_auto(cat_df, top_k_targets=top_k_targets)
        else:
            targets = [str(t) for t in target_labels]
        targets_used = list(targets)

        title_suffix = f"(balanced Early-Pre / Late-Pre / Post; Treatment {tdate.date()})"

        for tgt in targets:
            e_pre_raw = pps._preceder_counts_for_target(e_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            l_pre_raw = pps._preceder_counts_for_target(l_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            p_pre_raw = pps._preceder_counts_for_target(p_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
            order_pre = pps._unified_order(top_k_preceders, e_pre_raw, l_pre_raw, p_pre_raw)

            e_suc_raw = pps._successor_counts_for_target(e_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            l_suc_raw = pps._successor_counts_for_target(l_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            p_suc_raw = pps._successor_counts_for_target(p_df, tgt, include_end=include_end_as_successor, end_token=end_token)
            order_suc = pps._unified_order(top_k_successors, e_suc_raw, l_suc_raw, p_suc_raw)

            order_pre_hist = pps._order_with_optional_other(
                order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                include_other=include_other_in_hist, other_label=other_label
            )
            order_pre_pie = pps._order_with_optional_other(
                order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                include_other=True, other_label=other_label
            )
            order_suc_hist = pps._order_with_optional_other(
                order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                include_other=include_other_in_hist, other_label=other_label
            )
            order_suc_pie = pps._order_with_optional_other(
                order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                include_other=True, other_label=other_label
            )

            def _prep_counts(raw_s: pd.Series, order_for_hist: List[str], order_for_pie: List[str]):
                s_hist = pps._apply_other_bin(raw_s, order_for_hist, include_other=include_other_in_hist, other_label=other_label)
                s_pie  = pps._apply_other_bin(raw_s, order_for_pie,  include_other=True,              other_label=other_label)
                return s_hist, s_pie

            e_pre_hist, e_pre_pie = _prep_counts(e_pre_raw, order_pre_hist, order_pre_pie)
            l_pre_hist, l_pre_pie = _prep_counts(l_pre_raw, order_pre_hist, order_pre_pie)
            p_pre_hist, p_pre_pie = _prep_counts(p_pre_raw, order_pre_hist, order_pre_pie)

            e_suc_hist, e_suc_pie = _prep_counts(e_suc_raw, order_suc_hist, order_suc_pie)
            l_suc_hist, l_suc_pie = _prep_counts(l_suc_raw, order_suc_hist, order_suc_pie)
            p_suc_hist, p_suc_pie = _prep_counts(p_suc_raw, order_suc_hist, order_suc_pie)

            n_pre_by_group = {g: int(v.sum()) for g, v in {"Early Pre": e_pre_pie, "Late Pre": l_pre_pie, "Post": p_pre_pie}.items()}
            n_suc_by_group = {g: int(v.sum()) for g, v in {"Early Pre": e_suc_pie, "Late Pre": l_suc_pie, "Post": p_suc_pie}.items()}

            def _pick_order(d: Dict[str, pd.Series]) -> List[str]:
                for g in ["Early Pre", "Late Pre", "Post"]:
                    if len(d[g].index) > 0:
                        return list(d[g].index)
                return []

            order_pre_used_hist = _pick_order({"Early Pre": e_pre_hist, "Late Pre": l_pre_hist, "Post": p_pre_hist})
            order_suc_used_hist = _pick_order({"Early Pre": e_suc_hist, "Late Pre": l_suc_hist, "Post": p_suc_hist})

            out_path = pps_dir / f"preceder_successor_target_{tgt}.png"
            _ = pps._plot_pre_suc_panel(
                target_label=str(tgt),
                title_suffix=title_suffix,
                counts_pre_by_group={"Early Pre": e_pre_pie, "Late Pre": l_pre_pie, "Post": p_pre_pie},
                order_pre=order_pre_used_hist,
                n_pre_by_group=n_pre_by_group,
                counts_suc_by_group={"Early Pre": e_suc_pie, "Late Pre": l_suc_pie, "Post": p_suc_pie},
                order_suc=order_suc_used_hist,
                n_suc_by_group=n_suc_by_group,
                include_other_in_hist=include_other_in_hist,
                other_label=other_label,
                output_path=out_path, show=show,
            )
            per_target_outputs[str(tgt)] = str(out_path)

    # ──────────────────────────────────────────────────────────
    # 3) SONG-DURATION PLOTS
    # ──────────────────────────────────────────────────────────
    pre_df, post_df = sdc._split_pre_post(merged_detected_df, treatment_date=tdate)
    pre_s  = sdc._durations_seconds(pre_df)
    post_s = sdc._durations_seconds(post_df)

    dur_prepost_path = dur_dir / "song_duration_pre_post.png"
    _ = sdc.plot_pre_post_boxplot(
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
    # 4) HEATMAPS
    # ──────────────────────────────────────────────────────────
    if merged_annotations_df.empty:
        hm_v1_path = None
        hm_vmax_path = None
        vmax_max = float("nan")
    else:
        dfH = merged_annotations_df.copy()

        if "Date" not in dfH.columns:
            dt_col = "Recording DateTime" if "Recording DateTime" in dfH.columns else "recording_datetime"
            if dt_col not in dfH.columns:
                raise ValueError("No datetime column found to derive 'Date' (expected 'Recording DateTime' or 'recording_datetime').")
            dfH["Date"] = pd.to_datetime(dfH[dt_col]).dt.date

        label_col = "syllable_onsets_offsets_ms_dict"
        syllable_labels = _collect_unique_labels_sorted(dfH, label_col)
        count_table = shlin.build_daily_avg_count_table(
            dfH,
            label_column=label_col,
            date_column="Date",
            syllable_labels=syllable_labels,
        )

        hm_v1_path_p   = hm_dir / f"{_animal_id}_{heatmap_v1_filename}"
        hm_vmax_path_p = hm_dir / f"{_animal_id}_{heatmap_vmax_filename}"

        raw_max = float(np.nanmax(count_table.to_numpy())) if count_table.size else 1.0
        vmax_max = raw_max if np.isfinite(raw_max) and raw_max > 0 else 1.0

        _ = shlin.plot_linear_scaled_syllable_counts(
            count_table,
            animal_id=_animal_id,
            treatment_date=tdate,
            save_path=hm_v1_path_p,
            show=show,
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=1.0,
            nearest_match=nearest_match,
            max_days_off=max_days_off,
        )

        _ = shlin.plot_linear_scaled_syllable_counts(
            count_table,
            animal_id=_animal_id,
            treatment_date=tdate,
            save_path=hm_vmax_path_p,
            show=show,
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=vmax_max,
            nearest_match=nearest_match,
            max_days_off=max_days_off,
        )

        hm_v1_path = str(hm_v1_path_p)
        hm_vmax_path = str(hm_vmax_path_p)

    heatmaps_out = HeatmapOut(path_v1=hm_v1_path, path_vmax=hm_vmax_path, vmax_max=float(vmax_max))

    # ──────────────────────────────────────────────────────────
    # 5) PHRASE-DURATION (A): daily per-syllable, across days
    # ──────────────────────────────────────────────────────────
    if _HAS_GPDAY and not merged_annotations_df.empty:
        gpday.graph_phrase_duration_over_days(
            premerged_annotations_df=merged_annotations_df,
            premerged_annotations_path=decoded_annotations_json,
            save_output_to_this_file_path=pd_daily_dir,
            y_max_ms=25000,
            xtick_every=1,
            show_plots=show,
            treatment_date=tdate,
            syllables_subset=None,
        )
        made = sorted([str(p) for p in pd_daily_dir.glob(f"{_animal_id}_syllable_*_phrase_duration_plot.png")])
        daily_out = DailyPhraseDurationOut(
            output_dir=str(pd_daily_dir),
            num_plots=len(made),
            sample_paths=made[:5],
        )
    else:
        daily_out = DailyPhraseDurationOut(output_dir=None, num_plots=0, sample_paths=[])

    # ──────────────────────────────────────────────────────────
    # 6) PHRASE-DURATION (B): aggregate Early/Late/Post (with stats)
    # ──────────────────────────────────────────────────────────
    if merged_annotations_df.empty:
        pd_out = PhraseDurationPlotsOut(paths={}, syllable_labels=[], y_limits=(0.0, 1.0))
    else:
        label_col = "syllable_onsets_offsets_ms_dict"
        pd_labels = _collect_unique_labels_sorted(merged_annotations_df, label_col)
        pd_res = pdpg.run_phrase_duration_pre_vs_post_grouped(
            premerged_annotations_df=merged_annotations_df,
            output_dir=pd_dir,
            treatment_date=tdate,
            grouping_mode="auto_balance",
            restrict_to_labels=[str(x) for x in pd_labels] if pd_labels else None,
            y_max_ms=None,
            show_plots=show,
            animal_id_override=_animal_id,
        )
        pd_paths = {
            "early_pre_path": (None if pd_res.early_pre_path is None else str(pd_res.early_pre_path)),
            "late_pre_path": (None if pd_res.late_pre_path is None else str(pd_res.late_pre_path)),
            "post_path": (None if pd_res.post_path is None else str(pd_res.post_path)),
            "aggregate_three_box_path": (None if pd_res.aggregate_three_box_path is None else str(pd_res.aggregate_three_box_path)),
            "aggregate_three_box_auto_ylim_path": (None if pd_res.aggregate_three_box_auto_ylim_path is None else str(pd_res.aggregate_three_box_auto_ylim_path)),
            "aggregate_three_box_with_syll_points_path": (None if pd_res.aggregate_three_box_with_syll_points_path is None else str(pd_res.aggregate_three_box_with_syll_points_path)),
            "per_syllable_variance_boxscatter_path": (None if pd_res.per_syllable_variance_boxscatter_path is None else str(pd_res.per_syllable_variance_boxscatter_path)),
            "aggregate_three_box_variance_with_syll_points_path": (None if pd_res.aggregate_three_box_variance_with_syll_points_path is None else str(pd_res.aggregate_three_box_variance_with_syll_points_path)),
            "aggregate_three_box_variance_with_stats_path": (None if pd_res.aggregate_three_box_variance_with_stats_path is None else str(pd_res.aggregate_three_box_variance_with_stats_path)),
        }
        pd_out = PhraseDurationPlotsOut(
            paths=pd_paths,
            syllable_labels=[str(x) for x in (pd_res.syllable_labels or [])],
            y_limits=tuple(pd_res.y_limits) if pd_res.y_limits else (0.0, 1.0),
        )

    # ──────────────────────────────────────────────────────────
    # 7) TTE
    # ──────────────────────────────────────────────────────────
    tte_res = tte.TTE_by_day(
        decoded_database_json=decoded_annotations_json,
        premerged_annotations_df=merged_annotations_df,
        premerged_durations_df=merged_detected_df,
        animal_id_override=_animal_id,
        fig_dir=tte_dir,
        show=show,
        min_songs_per_day=tte_min_songs_per_day,
        treatment_date=tdate,
        treatment_in=tte_treatment_in,
        creation_metadata_json=None,
        only_song_present=True,
        compute_durations=False,
        organize_builder=None,
    )

    tte_out = TTEPlotsOut(
        timecourse_path=(None if tte_res.figure_path_timecourse is None else str(tte_res.figure_path_timecourse)),
        three_group_path=(None if tte_res.figure_path_prepost_box_plain is None else str(tte_res.figure_path_prepost_box_plain)),
        three_group_stats_path=(None if tte_res.figure_path_prepost_box_stats is None else str(tte_res.figure_path_prepost_box_stats)),
        prepost_p_value=tte_res.prepost_p_value,
        prepost_test=tte_res.prepost_test,
    )

    return AreaXMetaResult(
        last_syllable_hist_path=last_hist_path,
        last_syllable_pies_path=last_pies_path,
        last_syllable_label_order=last_label_order,
        last_syllable_counts=last_counts,
        targets_used=targets_used,
        per_target_outputs=per_target_outputs,
        duration_plots=duration_out,
        heatmaps=heatmaps_out,
        daily_phrase_duration=daily_out,
        phrase_duration_plots=pd_out,
        tte_plots=tte_out,
        merged_annotations_df=merged_annotations_df,
        merged_detected_df=merged_detected_df,
        early_df=early_df,
        late_df=late_df,
        post_df=post_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Batch runner (root directory mode)
# ──────────────────────────────────────────────────────────────────────────────

def run_area_x_meta_bundle_root(
    *,
    json_root_dir: Union[str, Path],
    metadata_excel: Union[str, Path],
    base_output_root: Optional[Union[str, Path]] = None,
    animals: Optional[Iterable[str]] = None,
    sheet_name: Union[str, int] = 0,
    id_col: Optional[str] = None,
    treatment_date_col: Optional[str] = None,
    skip_missing_treatment_date: bool = True,
    show: bool = False,
    **kwargs: Any,
) -> AreaXBatchResult:
    """
    Iterate over subfolders in json_root_dir (each folder name is animal_id),
    look up treatment date from metadata_excel, find required JSONs, and run
    run_area_x_meta_bundle() for each animal.

    Parameters
    ----------
    json_root_dir : directory containing subfolders per animal_id.
    metadata_excel : path to Area_X_lesion_metadata.xlsx (or similar).
    base_output_root : if provided, outputs go to base_output_root/animal_id.
                       if None, outputs go to animal_dir/'figures'.
    animals : optional whitelist of animal IDs (folder names) to run.
    show : show plots during batch run (default False).
    kwargs : forwarded to run_area_x_meta_bundle (merge knobs, plot knobs, etc.)

    Returns
    -------
    AreaXBatchResult
    """
    json_root_dir = Path(json_root_dir)
    metadata_excel = Path(metadata_excel)

    treat_map = load_treatment_dates_from_excel(
        metadata_excel,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
    )

    animal_dirs = _iter_animal_dirs(json_root_dir)
    if animals is not None:
        wanted = set(str(a).strip() for a in animals)
        animal_dirs = [d for d in animal_dirs if d.name in wanted]

    results: Dict[str, AreaXMetaResult] = {}
    errors: Dict[str, str] = []
    rows: List[Dict[str, Any]] = []
    error_map: Dict[str, str] = {}

    base_output_root_p = Path(base_output_root) if base_output_root is not None else None
    if base_output_root_p is not None:
        base_output_root_p.mkdir(parents=True, exist_ok=True)

    for animal_dir in animal_dirs:
        animal_id = animal_dir.name
        row: Dict[str, Any] = {"animal_id": animal_id, "animal_dir": str(animal_dir)}

        try:
            if animal_id not in treat_map:
                msg = f"No treatment date found for {animal_id} in {metadata_excel.name}"
                if skip_missing_treatment_date:
                    row.update({"status": "SKIP", "reason": msg})
                    rows.append(row)
                    continue
                raise KeyError(msg)

            tdate = treat_map[animal_id]
            decoded_json, detect_json = _find_required_jsons(animal_dir)

            if base_output_root_p is None:
                out_dir = animal_dir / "figures"
            else:
                out_dir = base_output_root_p / animal_id
            out_dir.mkdir(parents=True, exist_ok=True)

            res = run_area_x_meta_bundle(
                song_detection_json=detect_json,
                decoded_annotations_json=decoded_json,
                treatment_date=tdate,
                base_output_dir=out_dir,
                show=show,
                **kwargs,
            )
            results[animal_id] = res

            row.update({
                "status": "OK",
                "treatment_date": str(tdate.date()),
                "decoded_json": str(decoded_json),
                "song_detection_json": str(detect_json),
                "output_dir": str(out_dir),
                "tte_timecourse": res.tte_plots.timecourse_path,
                "dur_prepost": res.duration_plots.pre_post_path,
                "heatmap_v1": res.heatmaps.path_v1,
                "phrase_daily_dir": res.daily_phrase_duration.output_dir,
            })
            rows.append(row)

        except Exception as e:
            error_map[animal_id] = repr(e)
            row.update({"status": "ERROR", "reason": repr(e)})
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    return AreaXBatchResult(results=results, summary_df=summary_df, errors=error_map)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Area X meta wrapper. Single-animal mode OR batch mode (json root dir + metadata Excel)."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json-root", type=str, default=None,
                      help="Batch mode: root directory containing subfolders per animal_id.")
    mode.add_argument("--annotations", "-a", type=str, default=None,
                      help="Single mode: Path to *_decoded_database.json")

    # Batch mode args
    p.add_argument("--metadata-excel", type=str, default=None,
                   help="Batch mode: path to metadata Excel to look up treatment dates.")
    p.add_argument("--batch-out", type=str, default=None,
                   help="Optional: output root; results go into batch_out/animal_id. Default: animal_dir/figures.")
    p.add_argument("--animals", nargs="*", default=None,
                   help="Optional: restrict batch run to these animal IDs (folder names).")
    p.add_argument("--sheet-name", type=str, default="0",
                   help="Excel sheet name or index (default 0).")
    p.add_argument("--id-col", type=str, default=None,
                   help="Optional: explicit Animal ID column name in metadata Excel.")
    p.add_argument("--treatment-date-col", type=str, default=None,
                   help="Optional: explicit Treatment date column name in metadata Excel.")
    p.add_argument("--no-skip-missing-treatment", action="store_true",
                   help="If set, missing treatment dates become ERROR instead of SKIP.")

    # Single mode args
    p.add_argument("--song-detection", "-d", type=str, default=None,
                   help="Single mode: Path to *_song_detection.json")
    p.add_argument("--treatment-date", "-t", type=str, default=None,
                   help="Single mode: YYYY-MM-DD")
    p.add_argument("--base-out", type=str, default=None,
                   help="Single mode: base output directory (default: annotations.parent)")

    # Common merge + plot knobs (forwarded)
    p.add_argument("--ann-gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true",
                   help="If set, merge repeated syllables (function default is True; pass this in batch if desired).")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")

    p.add_argument("--merged-song-gap-ms", type=int, default=500,
                   help="Gap threshold (ms) to merge detected song segments; 0 disables merging.")

    p.add_argument("--top-k-last", type=int, default=15)
    p.add_argument("--top-k-targets", type=int, default=8)
    p.add_argument("--top-k-preceders", type=int, default=12)
    p.add_argument("--top-k-successors", type=int, default=12)

    p.add_argument("--include-start", action="store_true")
    p.add_argument("--include-end", action="store_true")
    p.add_argument("--include-other", action="store_true")
    p.add_argument("--include-other-in-hist", action="store_true")
    p.add_argument("--other-label", type=str, default="Other")

    p.add_argument("--heatmap-cmap", type=str, default="Greys")
    p.add_argument("--nearest-match", action="store_true")
    p.add_argument("--max-days-off", type=int, default=1)

    p.add_argument("--tte-min-songs-per-day", type=int, default=1)
    p.add_argument("--tte-treatment-in", type=str, default="post", choices=["post", "pre", "exclude"])

    p.add_argument("--show", action="store_true", help="Show plots during run (default off).")

    args = p.parse_args()

    # Convert sheet-name string to int if possible
    sheet_name: Union[int, str]
    try:
        sheet_name = int(args.sheet_name)
    except Exception:
        sheet_name = args.sheet_name

    common_kwargs = dict(
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,  # NOTE: store_true default False
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        merged_song_gap_ms=args.merged_song_gap_ms,
        top_k_last_labels=args.top_k_last,
        top_k_targets=args.top_k_targets,
        top_k_preceders=args.top_k_preceders,
        top_k_successors=args.top_k_successors,
        include_start_as_preceder=args.include_start,
        include_end_as_successor=args.include_end,
        include_other_bin=args.include_other,
        include_other_in_hist=args.include_other_in_hist,
        other_label=args.other_label,
        heatmap_cmap=args.heatmap_cmap,
        nearest_match=args.nearest_match,
        max_days_off=args.max_days_off,
        tte_min_songs_per_day=args.tte_min_songs_per_day,
        tte_treatment_in=args.tte_treatment_in,
    )

    if args.json_root is not None:
        # Batch mode
        if args.metadata_excel is None:
            raise SystemExit("Batch mode requires --metadata-excel")

        batch = run_area_x_meta_bundle_root(
            json_root_dir=args.json_root,
            metadata_excel=args.metadata_excel,
            base_output_root=args.batch_out,
            animals=args.animals,
            sheet_name=sheet_name,
            id_col=args.id_col,
            treatment_date_col=args.treatment_date_col,
            skip_missing_treatment_date=(not args.no_skip_missing_treatment),
            show=args.show,
            **common_kwargs,
        )

        print(batch.summary_df.to_string(index=False))
        print(f"\n[OK] Finished batch. Successes={len(batch.results)} Errors={len(batch.errors)}")

    else:
        # Single mode
        if args.song_detection is None or args.treatment_date is None:
            raise SystemExit("Single mode requires --song-detection and --treatment-date")

        res = run_area_x_meta_bundle(
            song_detection_json=args.song_detection,
            decoded_annotations_json=args.annotations,
            treatment_date=args.treatment_date,
            base_output_dir=args.base_out,
            show=args.show,
            **common_kwargs,
        )

        print("[OK] Last-syllable hist:", res.last_syllable_hist_path)
        print("[OK] Last-syllable pies:", res.last_syllable_pies_path)
        print("[OK] Targets used:", res.targets_used)
        print("[OK] Per-target panels:", len(res.per_target_outputs))
        print("[OK] Durations Pre/Post:", res.duration_plots.pre_post_path)
        print("[OK] Durations Three-Group:", res.duration_plots.three_group_path)
        print("[OK] Heatmap v1.0:", res.heatmaps.path_v1)
        print("[OK] Heatmap vmax=max:", res.heatmaps.path_vmax)
        print("[OK] Daily phrase-duration dir:", res.daily_phrase_duration.output_dir)
        print("[OK] Daily phrase-duration count:", res.daily_phrase_duration.num_plots)
        print("[OK] TTE time-course:", res.tte_plots.timecourse_path)
        print("[OK] TTE 3-group (clean):", res.tte_plots.three_group_path)
        print("[OK] TTE 3-group (stats):", res.tte_plots.three_group_stats_path)
        print("[OK] TTE pre/post p-value:", res.tte_plots.prepost_p_value, res.tte_plots.prepost_test)


"""
from pathlib import Path
import importlib
import Area_X_wrapper as axw
importlib.reload(axw)

root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
meta = root / "Area_X_lesion_metadata.xlsx"

batch = axw.run_area_x_meta_bundle_root(
    json_root_dir=root,
    metadata_excel=meta,
    base_output_root=root / "figures_batch",   # optional; otherwise each animal gets animal_dir/figures
    show=False,
    # you can pass any run_area_x_meta_bundle kwargs here:
    max_gap_between_song_segments=500,
    merge_repeated_syllables=True,
    merged_song_gap_ms=500,
    tte_min_songs_per_day=5,
)

batch.summary_df.head()
"""