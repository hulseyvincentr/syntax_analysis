from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List, Tuple

import math
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# ──────────────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GREEN_VIOLIN_COLOR = "#BFD8D2"
DEFAULT_GREEN_POINT_COLOR = "#2E4845"
DEFAULT_NOISE_COLOR = "#7f7f7f"


def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]


def _build_default_label_colors(labels: Sequence[str]) -> Dict[str, str]:
    palette = _get_tab60_palette()
    out: Dict[str, str] = {"-1": DEFAULT_NOISE_COLOR}
    for lbl in labels:
        s = str(lbl)
        if s == "-1":
            out[s] = DEFAULT_NOISE_COLOR
            continue
        try:
            idx = int(s) % len(palette)
        except Exception:
            idx = abs(hash(s)) % len(palette)
        out[s] = palette[idx]
    return out


def _load_label_colors(
    labels: Sequence[str],
    fixed_label_colors_json: Optional[Union[str, Path]],
) -> Dict[str, str]:
    fallback = _build_default_label_colors(labels)
    if fixed_label_colors_json is None:
        return fallback

    p = Path(fixed_label_colors_json)
    with open(p, "r") as f:
        raw = json.load(f)

    loaded = {str(k): str(v) for k, v in raw.items()}
    merged = fallback.copy()
    merged.update(loaded)
    return merged


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
    for c in ["Recording DateTime", "recording_datetime"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                return dt

    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str).str.replace(".", "-", regex=False) + " " + df["Time"].astype(str),
            errors="coerce",
        )
        if dt.notna().any():
            return dt

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
    violin_color: str = "lightgray",
    point_color: str = DEFAULT_GREEN_POINT_COLOR,
    y_max_ms: int = 25_000,
    jitter: float | bool = True,
    point_size: int = 5,
    point_alpha: float = 0.7,
    violin_alpha: float = 0.45,
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

    dt = pd.to_datetime(exploded["Date"], errors="coerce")
    dt = _drop_tz_if_present(dt)
    exploded["DateDay"] = dt.dt.normalize()

    exploded[col] = pd.to_numeric(exploded[col], errors="coerce")
    exploded = exploded.dropna(subset=[col, "DateDay"])
    if exploded.empty:
        return

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
        color=violin_color,
        inner="quartile",
    )

    ax = plt.gca()
    for coll in getattr(ax, "collections", []):
        try:
            coll.set_alpha(violin_alpha)
        except Exception:
            pass

    sns.stripplot(
        data=exploded,
        x="DateCat",
        y=col,
        order=date_order,
        jitter=jitter,
        size=point_size,
        color=point_color,
        alpha=point_alpha,
    )

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
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,
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
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_green_plots: bool = True,
    make_colored_plots: bool = True,
    green_violin_color: str = DEFAULT_GREEN_VIOLIN_COLOR,
    green_point_color: str = DEFAULT_GREEN_POINT_COLOR,
    violin_alpha: float = 0.45,
):
    """
    Make per-syllable violin+strip plots across calendar days.

    Produces up to two plot sets for each syllable:
      1) a green monochrome version
      2) a syllable-colored version using a fixed JSON mapping if provided,
         otherwise a stable internally generated label-color mapping.

    Returns
    -------
    OrganizedDataset | None
        - In merged modes, returns None (plots only).
        - In organizer fallback mode, returns the organizer’s OrganizedDataset.
    """
    save_dir = Path(save_output_to_this_file_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not make_green_plots and not make_colored_plots:
        raise ValueError("At least one of make_green_plots or make_colored_plots must be True.")

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
            song_detection_json = Path(song_detection_json)      # type: ignore[arg-type]

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

        dt = _choose_dt_series(df_merged)
        dt = _drop_tz_if_present(pd.to_datetime(dt, errors="coerce"))
        df_merged = df_merged.assign(_dt=dt)
        df_merged = df_merged.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
        if df_merged.empty:
            print("[WARN] Merged annotations dataframe has no valid datetime rows; nothing to plot.")
            return None
        df_merged["Date"] = df_merged["_dt"].dt.date

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

        label_colors = _load_label_colors(labels, fixed_label_colors_json)
        df_merged, _ = _build_durations_columns_from_dicts(df_merged, labels)

        earliest = pd.to_datetime(df_merged["_dt"].min()).normalize()
        latest = pd.to_datetime(df_merged["_dt"].max()).normalize()
        full_date_range = pd.date_range(start=earliest, end=latest, freq="D")

        fallback_treat = None
        if "treatment_date" in df_merged.columns and pd.notna(df_merged["treatment_date"]).any():
            try:
                fallback_treat = str(df_merged["treatment_date"].dropna().iloc[0])
            except Exception:
                fallback_treat = None
        t_dt = _coerce_treatment_date(treatment_date, fallback_treat)

        animal_id = _infer_animal_id(df_merged, merged_path_hint)

        for lbl in labels:
            col = f"syllable_{lbl}_durations"
            if col not in df_merged.columns:
                continue
            exploded = df_merged[["Date", col]].explode(col)
            if exploded[col].dropna().empty:
                continue

            if make_green_plots:
                out_path_green = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_plot_green.png"
                _plot_one_syllable(
                    exploded=exploded,
                    col=col,
                    full_date_range=full_date_range,
                    out_path=out_path_green,
                    violin_color=green_violin_color,
                    point_color=green_point_color,
                    y_max_ms=y_max_ms,
                    jitter=jitter,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    violin_alpha=violin_alpha,
                    figsize=figsize,
                    font_size_labels=font_size_labels,
                    xtick_fontsize=xtick_fontsize,
                    xtick_every=xtick_every,
                    treatment_date_dt=t_dt,
                    show_plots=show_plots,
                    dpi=dpi,
                    transparent=transparent,
                )

            if make_colored_plots:
                syll_color = label_colors.get(str(lbl), DEFAULT_GREEN_POINT_COLOR)
                out_path_col = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_plot_colored.png"
                _plot_one_syllable(
                    exploded=exploded,
                    col=col,
                    full_date_range=full_date_range,
                    out_path=out_path_col,
                    violin_color=syll_color,
                    point_color=syll_color,
                    y_max_ms=y_max_ms,
                    jitter=jitter,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    violin_alpha=violin_alpha,
                    figsize=figsize,
                    font_size_labels=font_size_labels,
                    xtick_fontsize=xtick_fontsize,
                    xtick_every=xtick_every,
                    treatment_date_dt=t_dt,
                    show_plots=show_plots,
                    dpi=dpi,
                    transparent=transparent,
                )

        return None

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

    label_colors = _load_label_colors([str(l) for l in labels], fixed_label_colors_json)

    try:
        animal_id = str(df["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"

    df["_DateOnly"] = df["Date"].dt.date

    for syllable_label in labels:
        col = f"syllable_{syllable_label}_durations"
        if col not in df.columns:
            continue
        exploded = df[["_DateOnly", col]].rename(columns={"_DateOnly": "Date"}).explode(col)
        if exploded[col].dropna().empty:
            continue

        if make_green_plots:
            out_path_green = Path(save_dir) / f"{animal_id}_syllable_{syllable_label}_phrase_duration_plot_green.png"
            _plot_one_syllable(
                exploded=exploded,
                col=col,
                full_date_range=full_date_range,
                out_path=out_path_green,
                violin_color=green_violin_color,
                point_color=green_point_color,
                y_max_ms=y_max_ms,
                jitter=jitter,
                point_size=point_size,
                point_alpha=point_alpha,
                violin_alpha=violin_alpha,
                figsize=figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=t_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

        if make_colored_plots:
            syll_color = label_colors.get(str(syllable_label), DEFAULT_GREEN_POINT_COLOR)
            out_path_col = Path(save_dir) / f"{animal_id}_syllable_{syllable_label}_phrase_duration_plot_colored.png"
            _plot_one_syllable(
                exploded=exploded,
                col=col,
                full_date_range=full_date_range,
                out_path=out_path_col,
                violin_color=syll_color,
                point_color=syll_color,
                y_max_ms=y_max_ms,
                jitter=jitter,
                point_size=point_size,
                point_alpha=point_alpha,
                violin_alpha=violin_alpha,
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Plot per-syllable phrase duration distributions across calendar days "
            "(works with organizer OR merged annotations), with optional green and/or colored plot sets."
        )
    )
    ap.add_argument("--decoded", type=str, default=None, help="Path to *_decoded_database.json")
    ap.add_argument("--creation-meta", type=str, default=None, help="Path to *_creation_data.json (unused with serial organizer)")
    ap.add_argument("--save-dir", type=str, required=True, help="Directory to save PNGs")

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
    ap.add_argument("--fixed-label-colors-json", type=str, default=None, help="Optional JSON mapping label -> hex color")
    ap.add_argument("--no-green-plots", action="store_true", help="Disable green monochrome plots")
    ap.add_argument("--no-colored-plots", action="store_true", help="Disable syllable-colored plots")

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
        fixed_label_colors_json=args.fixed_label_colors_json,
        make_green_plots=not args.no_green_plots,
        make_colored_plots=not args.no_colored_plots,
    )
