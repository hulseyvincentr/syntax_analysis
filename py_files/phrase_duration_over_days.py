# phrase_duration_over_days.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List, Tuple

import math
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: merged-builder import (used if you pass song_detection_json, or if
#           you want to run on the merged annotations directly inside this file)
# ──────────────────────────────────────────────────────────────────────────────
_HAS_MERGE_BUILDER = False
try:
    from merge_annotations_from_split_songs import build_decoded_with_split_labels  # type: ignore
    _HAS_MERGE_BUILDER = True
except Exception:
    _HAS_MERGE_BUILDER = False

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import: prefer Excel-serial, segments-aware builder (no metadata)
# Fallbacks keep legacy modules working.
# ──────────────────────────────────────────────────────────────────────────────
_USING_SERIAL_SEGMENTS = False
try:
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _build_organized,
        OrganizedDataset,
    )
    _USING_SERIAL_SEGMENTS = True
except ImportError:
    try:
        from organize_decoded_with_segments import (
            build_organized_segments_with_durations as _build_organized,
            OrganizedDataset,
        )
    except ImportError:
        from organize_decoded_with_durations import (
            build_organized_dataset_with_durations as _build_organized,
            OrganizedDataset,
        )

# ————————————————————————————————————————————————————————————————
# Seaborn back-compat (older seaborn used `scale`, newer uses `density_norm`)
# ————————————————————————————————————————————————————————————————
def _violin_with_backcompat(*, data, x, y, order=None, color="lightgray", inner="quartile"):
    try:
        return sns.violinplot(
            data=data, x=x, y=y, order=order, inner=inner, density_norm="width", color=color
        )
    except TypeError:
        return sns.violinplot(
            data=data, x=x, y=y, order=order, inner=inner, scale="width", color=color
        )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for merged-annotations tables (dict → durations per label)
# ──────────────────────────────────────────────────────────────────────────────
def _maybe_parse_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(obj)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None

def _row_ms_per_bin(row: pd.Series) -> float | None:
    for k in ["time_bin_ms", "timebin_ms", "bin_ms", "ms_per_bin"]:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return None

def _is_timebin_col(colname: str) -> bool:
    c = (colname or "").lower()
    return "timebin" in c or c.endswith("_bins") or "bin" in c

def _extract_durations_from_spans(spans, *, ms_per_bin: float | None, treat_as_bins: bool) -> List[float]:
    """
    Accepts:
      • [[on, off], ...] or [on, off]
      • [{"onset_ms":..., "offset_ms":...}, ...]
      • [{"onset_bin":..., "offset_bin":...}, ...]
    Converts bins→ms if treat_as_bins and ms_per_bin is provided.
    """
    out: List[float] = []
    if spans is None:
        return out

    if isinstance(spans, (list, tuple)) and len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
        spans = [spans]
    if isinstance(spans, dict):
        spans = [spans]
    if not isinstance(spans, (list, tuple)):
        return out

    for item in spans:
        on = off = None
        using_bins = False

        if isinstance(item, dict):
            if "onset_ms" in item or "on" in item:
                on = item.get("onset_ms", item.get("on"))
                off = item.get("offset_ms", item.get("off"))
            elif "onset_bin" in item or "on_bin" in item:
                on = item.get("onset_bin", item.get("on_bin"))
                off = item.get("offset_bin", item.get("off_bin"))
                using_bins = True
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            on, off = item[:2]
            using_bins = False

        try:
            dur = float(off) - float(on)
            if dur < 0:
                continue
            if treat_as_bins or using_bins:
                if ms_per_bin:
                    dur *= float(ms_per_bin)
                else:
                    continue
            out.append(dur)
        except Exception:
            continue
    return out

def _find_best_spans_column(df: pd.DataFrame) -> str | None:
    # Prefer ms dicts, then timebins, then any *_dict column with values.
    for c in [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_ms",
        "onsets_offsets_ms_dict",
    ]:
        if c in df.columns and df[c].notna().any():
            return c
    for c in [
        "syllable_onsets_offsets_timebins",
        "syllable_onsets_offsets_timebins_dict",
    ]:
        if c in df.columns and df[c].notna().any():
            return c
    for c in df.columns:
        lc = str(c).lower()
        if lc.endswith("_dict") and df[c].notna().any():
            return c
    return None

def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    labs: List[str] = []
    for d in df[dict_col].dropna():
        dd = _maybe_parse_dict(d) if not isinstance(d, dict) else d
        if isinstance(dd, dict):
            labs.extend(list(map(str, dd.keys())))
    labs = list(dict.fromkeys(labs))
    try:
        ints = [int(x) for x in labs]
        order = np.argsort(ints)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)

def _build_durations_columns_from_dicts(df: pd.DataFrame, labels: Sequence[str]) -> Tuple[pd.DataFrame, str | None]:
    col = _find_best_spans_column(df)
    if col is None:
        return df.copy(), None

    def _per_song(row: pd.Series) -> Dict[str, List[float]]:
        raw = row.get(col, None)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return {}
        d = _maybe_parse_dict(raw) if not isinstance(raw, dict) else raw
        if not isinstance(d, dict):
            return {}
        mpb = _row_ms_per_bin(row)
        treat_as_bins = _is_timebin_col(col)
        out: Dict[str, List[float]] = {}
        for lbl, spans in d.items():
            vals = _extract_durations_from_spans(spans, ms_per_bin=mpb, treat_as_bins=treat_as_bins)
            if vals:
                out[str(lbl)] = vals
        return out

    per_song = df.apply(_per_song, axis=1)
    out = df.copy()
    for lbl in [str(x) for x in labels]:
        out[f"syllable_{lbl}_durations"] = per_song.apply(lambda d: d.get(lbl, []))
    return out, col

def _choose_dt_series(df: pd.DataFrame) -> pd.Series:
    # Try explicit datetime
    for c in ["Recording DateTime", "recording_datetime"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                return dt

    # Try separate 'Date' and 'Time'
    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str).str.replace(".", "-", regex=False) + " " + df["Time"].astype(str),
            errors="coerce",
        )
        if dt.notna().any():
            return dt

    # Try just 'Date'
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        if dt.notna().any():
            return dt

    return pd.Series([pd.NaT] * len(df), index=df.index)

def _infer_animal_id(df: pd.DataFrame, fallback_path: Optional[Path]) -> str:
    for col in ["animal_id", "Animal", "Animal ID"]:
        if col in df.columns and pd.notna(df[col]).any():
            val = str(df[col].dropna().iloc[0]).strip()
            if val:
                return val
    if fallback_path is not None:
        stem = fallback_path.stem
        if stem:
            return stem.split("_")[0] or "unknown_animal"
    return "unknown_animal"

def _coerce_treatment_date(user_treatment_date: Optional[str | pd.Timestamp], fallback_str: Optional[str]) -> Optional[pd.Timestamp]:
    if user_treatment_date is not None:
        if isinstance(user_treatment_date, pd.Timestamp):
            return user_treatment_date.normalize()
        dt = pd.to_datetime(user_treatment_date, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()

    if fallback_str:
        dt = pd.to_datetime(fallback_str, format="%Y.%m.%d", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(fallback_str, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()
    return None

def _drop_tz_if_present(dt: pd.Series) -> pd.Series:
    """
    Robustly remove timezone info if dt is tz-aware (prevents categorical mismatches
    that can lead to "blank" plots).
    """
    try:
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            return dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt

# ──────────────────────────────────────────────────────────────────────────────
# Core plotter for one syllable (aligned day axis with gaps preserved)
# FIX: Use string date categories to prevent dtype/category mismatches that can
#      silently produce empty violins/strip plots.
# ──────────────────────────────────────────────────────────────────────────────
def _plot_one_syllable(
    exploded: pd.DataFrame,
    col: str,
    full_date_range: pd.DatetimeIndex,
    out_path: Path,
    *,
    y_max_ms: int = 25_000,
    jitter: float | bool = True,
    point_size: int = 5,
    point_alpha: float = 0.7,
    figsize: tuple[int, int] = (20, 11),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    treatment_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
):
    exploded = exploded.copy()

    # Robust date handling (avoid tz + categorical dtype mismatches)
    dt = pd.to_datetime(exploded["Date"], errors="coerce")
    dt = _drop_tz_if_present(dt)
    exploded["DateDay"] = dt.dt.normalize()

    exploded[col] = pd.to_numeric(exploded[col], errors="coerce")
    exploded = exploded.dropna(subset=[col, "DateDay"])
    if exploded.empty:
        return

    # Convert both the order and the data to matching strings (key fix)
    date_order = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in full_date_range]
    exploded["DateStr"] = exploded["DateDay"].dt.strftime("%Y-%m-%d")
    exploded["DateCat"] = pd.Categorical(exploded["DateStr"], categories=date_order, ordered=True)

    sns.set(style="white")
    plt.figure(figsize=figsize)

    _violin_with_backcompat(
        data=exploded,
        x="DateCat",
        y=col,
        order=date_order,
        color="lightgray",
        inner="quartile",
    )
    sns.stripplot(
        data=exploded,
        x="DateCat",
        y=col,
        order=date_order,
        jitter=jitter,
        size=point_size,
        color="#2E4845",
        alpha=point_alpha,
    )

    ax = plt.gca()
    plt.ylim(0, y_max_ms)
    plt.xlabel("Recording Date", fontsize=font_size_labels)
    plt.ylabel("Phrase Duration (ms)", fontsize=font_size_labels)

    tick_idx = list(range(0, len(date_order), max(1, int(xtick_every))))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_order[i] for i in tick_idx], rotation=90, fontsize=xtick_fontsize)

    if treatment_date_dt is not None and not pd.isna(treatment_date_dt):
        t_str = pd.Timestamp(treatment_date_dt).strftime("%Y-%m-%d")
        if t_str in date_order:
            t_idx = date_order.index(t_str)
            plt.axvline(x=t_idx, color="red", linestyle="--", label="Treatment Date")
            plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent))
    if show_plots:
        plt.show()
    else:
        plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def graph_phrase_duration_over_days(
    decoded_database_json: str | Path | None = None,
    creation_metadata_json: str | Path | None = None,
    save_output_to_this_file_path: str | Path = ".",
    *,
    # NEW: run directly on the already-merged annotations table (wrapper-friendly)
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,  # used only to infer animal id

    # NEW: optionally do the merge *inside* this function if both JSONs are provided
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    only_song_present: bool = False,
    y_max_ms: int = 25_000,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[int, int] = (20, 11),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    show_plots: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
    treatment_date: Optional[str | pd.Timestamp] = None,
) -> "OrganizedDataset | None":
    """
    Make per-syllable violin+strip plots across calendar days.

    Modes
    -----
    1) Premerged path (recommended when calling from wrappers):
       - Pass `premerged_annotations_df` (DataFrame returned by build_decoded_with_split_labels().annotations_appended_df).
    2) Internal merge:
       - Provide BOTH `decoded_database_json` (decoded DB) AND `song_detection_json` (detection JSON)
         to run the same merged builder inside this function.
    3) Organizer fallback (legacy behavior):
       - Provide `decoded_database_json` (and optionally `creation_metadata_json`) and we’ll use the
         segments-aware organizer to compute durations columns.

    Returns
    -------
    OrganizedDataset | None
        - In modes (2) and (1), returns None (plots only), to keep dependencies light.
        - In mode (3), returns the organizer’s `OrganizedDataset` like before.
    """
    save_dir = Path(save_output_to_this_file_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # MODE 1/2: Use merged annotations (premerged or merge here)
    # ──────────────────────────────────────────────────────────
    if premerged_annotations_df is not None or (decoded_database_json and song_detection_json):
        if premerged_annotations_df is not None:
            df_merged = premerged_annotations_df.copy()
            merged_path_hint = Path(premerged_annotations_path) if premerged_annotations_path else None
        else:
            if not _HAS_MERGE_BUILDER:
                raise RuntimeError(
                    "merge_annotations_from_split_songs.build_decoded_with_split_labels is not available; "
                    "install/put it on PYTHONPATH to use internal merging."
                )
            decoded_database_json = Path(decoded_database_json)  # type: ignore[arg-type]
            song_detection_json  = Path(song_detection_json)     # type: ignore[arg-type]

            ann = build_decoded_with_split_labels(
                decoded_database_json=decoded_database_json,
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
            df_merged = ann.annotations_appended_df.copy()
            merged_path_hint = Path(decoded_database_json)

        # Prepare datetime + Date column
        dt = _choose_dt_series(df_merged)
        dt = _drop_tz_if_present(pd.to_datetime(dt, errors="coerce"))
        df_merged = df_merged.assign(_dt=dt)
        df_merged = df_merged.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
        if df_merged.empty:
            print("[WARN] Merged annotations dataframe has no valid datetime rows; nothing to plot.")
            return None
        df_merged["Date"] = df_merged["_dt"].dt.date

        # Choose labels from the best spans dict column and build durations columns
        spans_col = _find_best_spans_column(df_merged)
        if spans_col is None:
            print("[WARN] No *_dict spans column found in merged annotations; nothing to plot.")
            return None

        labels = _collect_unique_labels_sorted(df_merged, spans_col)
        if syllables_subset is not None:
            keep = set(map(str, syllables_subset))
            labels = [str(l) for l in labels if str(l) in keep]
        if not labels:
            print("[WARN] No syllables to plot after filtering.")
            return None

        df_merged, _ = _build_durations_columns_from_dicts(df_merged, labels)

        # Full date range (preserve gaps)
        earliest = pd.to_datetime(df_merged["_dt"].min()).normalize()
        latest   = pd.to_datetime(df_merged["_dt"].max()).normalize()
        full_date_range = pd.date_range(start=earliest, end=latest, freq="D")

        # Treatment date: you can pass string/Timestamp; merged tables may also contain "treatment_date"
        fallback_treat = None
        if "treatment_date" in df_merged.columns and pd.notna(df_merged["treatment_date"]).any():
            try:
                fallback_treat = str(df_merged["treatment_date"].dropna().iloc[0])
            except Exception:
                fallback_treat = None
        t_dt = _coerce_treatment_date(treatment_date, fallback_treat)

        # Animal ID for filenames
        animal_id = _infer_animal_id(df_merged, merged_path_hint)

        # Plot each syllable
        for lbl in labels:
            col = f"syllable_{lbl}_durations"
            if col not in df_merged.columns:
                continue
            exploded = df_merged[["Date", col]].explode(col)
            if exploded[col].dropna().empty:
                continue

            out_path = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_plot.png"
            _plot_one_syllable(
                exploded=exploded,
                col=col,
                full_date_range=full_date_range,
                out_path=out_path,
                y_max_ms=y_max_ms,
                jitter=jitter,
                point_size=point_size,
                point_alpha=point_alpha,
                figsize=figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=t_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

        return None  # merged mode → no OrganizedDataset

    # ──────────────────────────────────────────────────────────
    # MODE 3: Organizer fallback (original behavior)
    # ──────────────────────────────────────────────────────────
    out = _build_organized(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=None if _USING_SERIAL_SEGMENTS else creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=True,
        add_recording_datetime=True if _USING_SERIAL_SEGMENTS else False,
    )

    df = out.organized_df.copy()

    # Organizer Date can be string/float; normalize into datetime
    df["Date"] = pd.to_datetime(df.get("Date", pd.Series([pd.NaT] * len(df))), errors="coerce")
    df["Date"] = _drop_tz_if_present(df["Date"])

    if not df["Date"].notna().any():
        return out

    earliest_date = df["Date"].min().normalize()
    latest_date = df["Date"].max().normalize()
    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq="D")

    t_dt = _coerce_treatment_date(treatment_date, getattr(out, "treatment_date", None))

    labels = out.unique_syllable_labels
    if syllables_subset is not None:
        keep = set(map(str, syllables_subset))
        labels = [lab for lab in labels if str(lab) in keep]
    if not labels:
        return out

    try:
        animal_id = str(df["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"

    # Use python date objects for the plotting helper (it will convert safely)
    df["_DateOnly"] = df["Date"].dt.date

    for syllable_label in labels:
        col = f"syllable_{syllable_label}_durations"
        if col not in df.columns:
            continue
        exploded = df[["_DateOnly", col]].rename(columns={"_DateOnly": "Date"}).explode(col)
        if exploded[col].dropna().empty:
            continue
        out_path = Path(save_dir) / f"{animal_id}_syllable_{syllable_label}_phrase_duration_plot.png"
        _plot_one_syllable(
            exploded=exploded,
            col=col,
            full_date_range=full_date_range,
            out_path=out_path,
            y_max_ms=y_max_ms,
            jitter=jitter,
            point_size=point_size,
            point_alpha=point_alpha,
            figsize=figsize,
            font_size_labels=font_size_labels,
            xtick_fontsize=xtick_fontsize,
            xtick_every=xtick_every,
            treatment_date_dt=t_dt,
            show_plots=show_plots,
            dpi=dpi,
            transparent=transparent,
        )

    return out


# ——————————————————————————————————————————
# Optional CLI
# ——————————————————————————————————————————
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot per-syllable phrase duration distributions across calendar days "
                    "(works with organizer OR merged annotations)."
    )
    ap.add_argument("--decoded", type=str, default=None, help="Path to *_decoded_database.json")
    ap.add_argument("--creation-meta", type=str, default=None, help="Path to *_creation_data.json (unused with serial organizer)")
    ap.add_argument("--save-dir", type=str, required=True, help="Directory to save PNGs")

    # Merged-mode (internal) options
    ap.add_argument("--song-detection", type=str, default=None, help="Path to *_song_detection.json (enables internal merge)")
    ap.add_argument("--ann-gap-ms", type=int, default=500)
    ap.add_argument("--seg-offset", type=int, default=0)
    ap.add_argument("--merge-repeats", action="store_true")
    ap.add_argument("--repeat-gap-ms", type=float, default=10.0)
    ap.add_argument("--repeat-gap-inclusive", action="store_true")

    ap.add_argument("--only-song-present", action="store_true")
    ap.add_argument("--y-max-ms", type=int, default=25000)
    ap.add_argument("--xtick-every", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--syllables", type=str, nargs="*", default=None, help="Subset of syllable labels to plot")
    ap.add_argument("--treatment-date", type=str, default=None, help="e.g., '2025-03-04' or '2025.03.04'")

    args = ap.parse_args()

    graph_phrase_duration_over_days(
        decoded_database_json=args.decoded,
        creation_metadata_json=args.creation_meta,
        save_output_to_this_file_path=args.save_dir,
        song_detection_json=args.song_detection,
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        only_song_present=args.only_song_present,
        y_max_ms=args.y_max_ms,
        xtick_every=args.xtick_every,
        dpi=args.dpi,
        transparent=args.transparent,
        show_plots=args.show,
        syllables_subset=args.syllables,
        treatment_date=args.treatment_date,
    )

"""
from pathlib import Path
import sys, importlib

# Point to your code folder if needed
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import merge_annotations_from_split_songs as mas
importlib.reload(mas)

# Paths to one bird's JSONs
detect  = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443_song_detection.json")
decoded = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443_decoded_database.json")

# Build merged annotations
res = mas.build_decoded_with_split_labels(
    decoded_database_json=decoded,
    song_detection_json=detect,
    only_song_present=True,
    compute_durations=True,
    add_recording_datetime=True,
    songs_only=True,
    flatten_spec_params=True,
    max_gap_between_song_segments=500,
    segment_index_offset=0,
    merge_repeated_syllables=True,
    repeat_gap_ms=10.0,
    repeat_gap_inclusive=False,
)

merged_df = res.annotations_appended_df

# Print 2–3 lines + key columns (adjust column list if yours differs)
cols = [c for c in ["file_name", "Segment", "was_merged", "merged_n_parts",
                   "Recording DateTime", "Date", "Hour", "Minute", "Second"]
        if c in merged_df.columns]

print("Merged DF shape:", merged_df.shape)
print(merged_df[cols].head(3).to_string(index=False))

"""