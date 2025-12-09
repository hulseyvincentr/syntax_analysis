# syllable_heatmap_linear.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports for JSON-based pipelines
try:
    # Full merge (detection + merged songs)
    from merge_annotations_from_split_songs import build_decoded_with_split_labels
except Exception:  # pragma: no cover - keep module importable even if missing
    build_decoded_with_split_labels = None  # type: ignore[assignment]

try:
    # Organizer that builds per-segment annotations from decoded JSON
    from organized_decoded_serialTS_segments import build_organized_segments_with_durations
except Exception:  # pragma: no cover
    build_organized_segments_with_durations = None  # type: ignore[assignment]

__all__ = [
    "_infer_animal_id",
    "build_daily_avg_count_table",
    "build_daily_avg_count_table_from_decoded",
    "plot_linear_scaled_syllable_counts",
    "plot_linear_scaled_syllable_counts_pair",
    "batch_plot_daily_syllable_counts",
]

# ---------- helpers ----------

def _infer_animal_id(df: pd.DataFrame, fallback: Union[str, Path, None] = None) -> Optional[str]:
    """Try to read an animal/bird ID from common columns; else use filename stem."""
    for c in ["Animal ID", "animal_id", "animal", "bird_id", "Animal", "ID"]:
        if c in df.columns and df[c].notna().any():
            s = str(df[c].dropna().iloc[0]).strip()
            if s:
                return s
    if fallback is not None:
        try:
            return Path(str(fallback)).stem
        except Exception:
            pass
    return None


def _sorted_labels_numeric_first(labels: List[str | int]) -> List[str]:
    """Sort labels numerically when possible, then lexicographically."""
    def _key(x):
        s = str(x)
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [str(x) for x in sorted(labels, key=_key)]


# ---------- table builder from an existing DataFrame ----------

def build_daily_avg_count_table(
    df: pd.DataFrame,
    *,
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
    syllable_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a table with rows = dates, cols = labels (numeric order),
    values = average count per song on that date.

    Expects `label_column` dicts: {label: [[on,off], ...], ...}.

    Parameters
    ----------
    df : pd.DataFrame
        One row per song/segment, with a column holding syllable interval dicts
        and a date column.
    label_column : str, default "syllable_onsets_offsets_ms_dict"
        Column containing dicts {label: [[on, off], ...], ...}.
    date_column : str, default "Date"
        Column indicating recording date (string, datetime, or date-like).
    syllable_labels : list[str] or None
        Optional explicit list of syllable labels to include. If None, all labels
        observed in `label_column` are used.

    Returns
    -------
    pd.DataFrame
        Index: dates
        Columns: syllable labels (sorted numeric-first)
        Values: average count per song on that date.
    """
    d = df.copy()
    if date_column not in d.columns:
        raise KeyError(f"'{date_column}' not in dataframe.")
    d[date_column] = pd.to_datetime(d[date_column]).dt.date

    # Determine label list if not provided → include ALL labels found
    if syllable_labels is None:
        labs: List[str] = []
        for v in d.get(label_column, pd.Series([], dtype=object)).dropna():
            if isinstance(v, dict):
                labs.extend(list(v.keys()))
        syllable_labels = list(dict.fromkeys(map(str, labs)))  # preserve discovery order
    syllable_labels = _sorted_labels_numeric_first(syllable_labels)

    # Create full grid (keeps zeros where a label is unused on a date)
    dates = sorted(pd.unique(d[date_column]))
    table = pd.DataFrame(
        0.0, index=pd.Index(dates, name="Date"),
        columns=[str(x) for x in syllable_labels], dtype=float
    )

    for date, g in d.groupby(date_column):
        n_songs = int(len(g))
        if n_songs == 0:
            continue
        counts = dict.fromkeys(table.columns, 0.0)
        for v in g.get(label_column, pd.Series([], dtype=object)).dropna():
            if not isinstance(v, dict):
                continue
            for lab, intervals in v.items():
                try:
                    counts[str(lab)] += float(len(intervals))
                except Exception:
                    # Ignore malformed entries
                    pass
        for lab in counts:
            table.at[date, lab] = counts[lab] / max(1, n_songs)  # avg per song
    return table


# ---------- table builder directly from decoded JSON (+ optional merge) ----------

def build_daily_avg_count_table_from_decoded(
    *,
    decoded_database_json: Union[str, Path],
    song_detection_json: Optional[Union[str, Path]] = None,
    merge_split_songs: bool = False,
    # Organizer options (for both merged and unmerged paths)
    only_song_present: bool = True,
    compute_durations: bool = True,
    add_recording_datetime: bool = True,
    # Detection / merge options (used when merge_split_songs=True)
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = False,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    # Table options
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
    syllable_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper to go from decoded JSON (and optional song detection JSON)
    to a daily average syllable-count table.

    If `merge_split_songs` is False:
      - Uses `organized_decoded_serialTS_segments.build_organized_segments_with_durations`
        to build a per-segment annotations dataframe.
      - Builds the daily table from `org.organized_df`.

    If `merge_split_songs` is True:
      - Uses `merge_annotations_from_split_songs.build_decoded_with_split_labels`
        to:
          * Run detection and find split-up songs.
          * Build per-segment annotations.
          * Append annotations across segments for split songs.
      - Builds the daily table from the merged annotations dataframe
        (`res.annotations_appended_df`).

    Returns
    -------
    pd.DataFrame
        Daily average count table, suitable for `plot_linear_scaled_syllable_counts`.
    """
    decoded_database_json = Path(decoded_database_json)

    if merge_split_songs:
        # ---- Use the full detection+merge pipeline --------------------------
        if build_decoded_with_split_labels is None:
            raise ImportError(
                "merge_split_songs=True but 'build_decoded_with_split_labels' "
                "could not be imported from 'merge_annotations_from_split_songs'. "
                "Make sure that module is on your PYTHONPATH."
            )
        if song_detection_json is None:
            raise ValueError(
                "song_detection_json is required when merge_split_songs=True."
            )

        song_detection_json = Path(song_detection_json)
        res = build_decoded_with_split_labels(
            decoded_database_json=decoded_database_json,
            song_detection_json=song_detection_json,
            only_song_present=only_song_present,
            compute_durations=compute_durations,
            add_recording_datetime=add_recording_datetime,
            songs_only=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=segment_index_offset,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
        )
        df_for_table = res.annotations_appended_df.copy()

    else:
        # ---- Plain per-segment organizer (no merge) -------------------------
        if build_organized_segments_with_durations is None:
            raise ImportError(
                "merge_split_songs=False but 'build_organized_segments_with_durations' "
                "could not be imported from 'organized_decoded_serialTS_segments'. "
                "Make sure that module is on your PYTHONPATH."
            )

        org = build_organized_segments_with_durations(
            decoded_database_json=decoded_database_json,
            only_song_present=only_song_present,
            compute_durations=compute_durations,
            add_recording_datetime=add_recording_datetime,
        )
        df_for_table = org.organized_df.copy()

    # Safety: ensure a date column exists if we're going to use it
    if date_column not in df_for_table.columns:
        raise KeyError(
            f"Expected a '{date_column}' column in the annotations dataframe "
            f"produced from {decoded_database_json}, but it was not found."
        )

    return build_daily_avg_count_table(
        df_for_table,
        label_column=label_column,
        date_column=date_column,
        syllable_labels=syllable_labels,
    )


# ---------- plotter (labels on Y, dates on X) ----------

def plot_linear_scaled_syllable_counts(
    count_table: pd.DataFrame,
    *,
    animal_id: str,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: str = "Greys",
    vmin: float = 0.0,
    vmax: float = 1.0,
    nearest_match: bool = True,
    max_days_off: int = 1,
):
    """
    Heatmap with DATES on X-axis and SYLLABLE LABELS on Y-axis (numeric order).
    Adds a vertical dashed red line at the (nearest) treatment date.

    Parameters
    ----------
    count_table : pd.DataFrame
        Daily average counts returned by `build_daily_avg_count_table` or
        `build_daily_avg_count_table_from_decoded`.
    animal_id : str
        Used in the plot title.
    treatment_date : str or pd.Timestamp or None
        Optional treatment date. If provided, the nearest date (within
        `max_days_off` days) gets a dashed red vertical line.
    save_path : str or Path or None
        If provided, the figure is saved here.
    show : bool, default True
        If True, call plt.show(). If False, the figure is left open for the
        caller to manage (useful for wrappers).
    cmap : str, default "Greys"
        Matplotlib colormap name.
    vmin, vmax : float
        Color scale limits for the (linear) counts.
    nearest_match : bool, default True
        If True, use the nearest date within `max_days_off` as the treatment day.
        If False, only draw the line if an exact date match exists.
    max_days_off : int, default 1
        Maximum allowed offset in days between `treatment_date` and a recording
        date when `nearest_match=True`.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if count_table.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    # Ensure numeric label ordering (columns) and chronological dates (rows)
    cols_sorted = _sorted_labels_numeric_first(list(count_table.columns))
    ct = count_table.reindex(columns=cols_sorted)
    idx = pd.to_datetime(ct.index)
    ct = ct.iloc[np.argsort(idx.values)]

    # Matrix: rows=dates, cols=labels → transpose for labels on Y
    M = ct.to_numpy().T
    dates = [pd.to_datetime(str(d)).date().isoformat() for d in ct.index]
    labels = list(ct.columns)

    fig, ax = plt.subplots(
        figsize=(max(8, 0.28 * len(dates) + 4), max(4, 0.30 * len(labels) + 2))
    )
    im = ax.imshow(
        M,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # ticks
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels(dates, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Recording date")
    ax.set_ylabel("Syllable label")
    ax.set_title(f"{animal_id}: average syllable count per song (linear scale)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Avg count / song")

    # treatment date line (vertical)
    if treatment_date is not None and len(dates):
        t = pd.to_datetime(str(treatment_date)).date()
        date_objs = [pd.to_datetime(d).date() for d in ct.index]
        diffs = [abs((d - t).days) for d in date_objs]
        j = int(np.argmin(diffs))
        if nearest_match or date_objs[j] == t:
            if diffs[j] <= int(max_days_off):
                ax.vlines(
                    j,
                    -0.5,
                    len(labels) - 0.5,
                    colors="red",
                    linestyles="--",
                    linewidth=1.5,
                )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# ---------- make both fixed-scale and auto-scale plots for a bird ----------

def plot_linear_scaled_syllable_counts_pair(
    count_table: pd.DataFrame,
    *,
    animal_id: str,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    save_path_fixed: Optional[Union[str, Path]] = None,
    save_path_auto: Optional[Union[str, Path]] = None,
    cmap: str = "Greys",
    fixed_vmin: float = 0.0,
    fixed_vmax: float = 1.0,
    nearest_match: bool = True,
    max_days_off: int = 1,
    show: bool = True,
):
    """
    Convenience wrapper: generate TWO heatmaps for the same bird:

      1) Fixed/global color scale [fixed_vmin, fixed_vmax]
      2) Auto-scaled colorbar based on this bird's local min/max

    Returns
    -------
    ((fig_fixed, ax_fixed), (fig_auto, ax_auto))
    """
    # 1) Fixed/global scale
    fig_fixed, ax_fixed = plot_linear_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        save_path=save_path_fixed,
        show=False,  # we'll control showing at the end
        cmap=cmap,
        vmin=fixed_vmin,
        vmax=fixed_vmax,
        nearest_match=nearest_match,
        max_days_off=max_days_off,
    )

    # 2) Auto-scale based on this bird's data
    data = count_table.to_numpy()
    finite_mask = np.isfinite(data)
    if finite_mask.any():
        local_vmin = float(data[finite_mask].min())
        local_vmax = float(data[finite_mask].max())
        # Avoid degenerate colorbars (vmin == vmax)
        if local_vmin == local_vmax:
            local_vmin = local_vmin - 0.5
            local_vmax = local_vmax + 0.5
    else:
        # Fall back to the fixed scale if everything is NaN/inf
        local_vmin, local_vmax = fixed_vmin, fixed_vmax

    fig_auto, ax_auto = plot_linear_scaled_syllable_counts(
        count_table,
        animal_id=f"{animal_id} (auto-scaled)",
        treatment_date=treatment_date,
        save_path=save_path_auto,
        show=False,
        cmap=cmap,
        vmin=local_vmin,
        vmax=local_vmax,
        nearest_match=nearest_match,
        max_days_off=max_days_off,
    )

    if show:
        plt.show()

    return (fig_fixed, ax_fixed), (fig_auto, ax_auto)


# ---------- batch helper: iterate over many birds in a root directory ----------

def batch_plot_daily_syllable_counts(
    root_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    merge_split_songs: bool = False,
    use_pair_plots: bool = True,
    # Patterns for filenames inside each bird directory
    decoded_pattern: str = "{animal_id}_decoded_database.json",
    song_detection_pattern: str = "{animal_id}_song_detection.json",
    # Plot options
    cmap: str = "Greys",
    fixed_vmin: float = 0.0,
    fixed_vmax: float = 1.0,
    nearest_match: bool = True,
    max_days_off: int = 1,
    show: bool = False,
    # Optional per-bird treatment dates: {"USA5325": "2024-03-05", ...}
    treatment_dates: Optional[Dict[str, Union[str, pd.Timestamp]]] = None,
    # Organizer / merge options (forwarded)
    only_song_present: bool = True,
    compute_durations: bool = True,
    add_recording_datetime: bool = True,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = False,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
) -> Dict[str, Dict[str, Optional[Union[str, pd.DataFrame]]]]:
    """
    Iterate over many birds in a root directory, build daily syllable-count tables,
    and save heatmaps either into each bird's 'figures' subdirectory (default) or
    into a shared output directory if `output_dir` is provided.

    Directory layout (example)
    --------------------------
    root_dir/
        USA5325/
            USA5325_decoded_database.json
            USA5325_song_detection.json
            figures/
        USA5326/
            USA5326_decoded_database.json
            ...

    For each bird directory:
      1) Try to find the decoded JSON using `decoded_pattern`.
      2) Optionally find the song_detection JSON using `song_detection_pattern`
         and run the merge pipeline if `merge_split_songs=True`.
      3) Build the daily average count table.
      4) Save either:
         - a fixed-scale plot, or
         - both fixed-scale and auto-scale plots (if use_pair_plots=True)
         into:
             - <bird_dir>/figures/ (if output_dir is None), or
             - output_dir (if provided).

    Filenames follow the pattern:
        animal_id_scaling_used_syllable_usage.png

    e.g.:
        USA5325_fixed_syllable_usage.png
        USA5325_auto_syllable_usage.png

    Returns
    -------
    dict
        Mapping animal_id → {
            "count_table": pd.DataFrame or None,
            "fixed_path": str or None,
            "auto_path": str or None,
        }
    """
    root_dir = Path(root_dir)
    results: Dict[str, Dict[str, Optional[Union[str, pd.DataFrame]]]] = {}

    # Normalize output_dir once, if provided
    output_dir_path: Optional[Path] = None
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    for bird_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        animal_id = bird_dir.name

        decoded_path = bird_dir / decoded_pattern.format(animal_id=animal_id)
        if not decoded_path.is_file():
            print(f"[batch] {animal_id}: decoded database not found at {decoded_path}, skipping.")
            results[animal_id] = {"count_table": None, "fixed_path": None, "auto_path": None}
            continue

        # Decide on detection JSON for this bird
        song_det_path: Optional[Path] = None
        if merge_split_songs:
            candidate = bird_dir / song_detection_pattern.format(animal_id=animal_id)
            if candidate.is_file():
                song_det_path = candidate
            else:
                print(
                    f"[batch] {animal_id}: merge_split_songs=True but no song_detection JSON at "
                    f"{candidate}; falling back to unmerged per-segment data."
                )

        # Build the daily table
        try:
            count_table = build_daily_avg_count_table_from_decoded(
                decoded_database_json=decoded_path,
                song_detection_json=song_det_path,
                merge_split_songs=merge_split_songs and (song_det_path is not None),
                only_song_present=only_song_present,
                compute_durations=compute_durations,
                add_recording_datetime=add_recording_datetime,
                max_gap_between_song_segments=max_gap_between_song_segments,
                segment_index_offset=segment_index_offset,
                merge_repeated_syllables=merge_repeated_syllables,
                repeat_gap_ms=repeat_gap_ms,
                repeat_gap_inclusive=repeat_gap_inclusive,
                label_column=label_column,
                date_column=date_column,
            )
        except Exception as exc:
            print(f"[batch] {animal_id}: error building daily table: {exc}")
            results[animal_id] = {"count_table": None, "fixed_path": None, "auto_path": None}
            continue

        if count_table.empty:
            print(f"[batch] {animal_id}: daily count table is empty; skipping plots.")
            results[animal_id] = {"count_table": count_table, "fixed_path": None, "auto_path": None}
            continue

        # Decide where to save figures
        if output_dir_path is not None:
            fig_dir = output_dir_path
        else:
            fig_dir = bird_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

        # New filename convention: animal_id_scaling_used_syllable_usage
        fixed_path = fig_dir / f"{animal_id}_fixed_syllable_usage.png"
        auto_path = fig_dir / f"{animal_id}_auto_syllable_usage.png"

        treatment_date = None
        if treatment_dates is not None:
            treatment_date = treatment_dates.get(animal_id)

        if use_pair_plots:
            try:
                plot_linear_scaled_syllable_counts_pair(
                    count_table,
                    animal_id=animal_id,
                    treatment_date=treatment_date,
                    save_path_fixed=fixed_path,
                    save_path_auto=auto_path,
                    cmap=cmap,
                    fixed_vmin=fixed_vmin,
                    fixed_vmax=fixed_vmax,
                    nearest_match=nearest_match,
                    max_days_off=max_days_off,
                    show=show,
                )
            except Exception as exc:
                print(f"[batch] {animal_id}: error plotting pair of heatmaps: {exc}")
                results[animal_id] = {
                    "count_table": count_table,
                    "fixed_path": None,
                    "auto_path": None,
                }
                continue
        else:
            try:
                plot_linear_scaled_syllable_counts(
                    count_table,
                    animal_id=animal_id,
                    treatment_date=treatment_date,
                    save_path=fixed_path,
                    show=show,
                    cmap=cmap,
                    vmin=fixed_vmin,
                    vmax=fixed_vmax,
                    nearest_match=nearest_match,
                    max_days_off=max_days_off,
                )
                auto_path = None
            except Exception as exc:
                print(f"[batch] {animal_id}: error plotting fixed-scale heatmap: {exc}")
                results[animal_id] = {
                    "count_table": count_table,
                    "fixed_path": None,
                    "auto_path": None,
                }
                continue

        results[animal_id] = {
            "count_table": count_table,
            "fixed_path": str(fixed_path),
            "auto_path": str(auto_path) if use_pair_plots else None,
        }

    return results


"""
from pathlib import Path
import pandas as pd
import importlib
import syllable_heatmap_linear as shl

importlib.reload(shl)

root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
out_dir = root / "syllable_usage_heatmaps"

# Build treatment_dates dict from Excel (same as you had)
meta_path = root / "Area_X_lesion_metadata.xlsx"
meta_df = pd.read_excel(meta_path, sheet_name=0)
meta_df["Treatment date"] = pd.to_datetime(meta_df["Treatment date"], errors="coerce")
meta_clean = meta_df.dropna(subset=["Animal ID", "Treatment date"])
treatment_dates = (
    meta_clean
    .groupby("Animal ID")["Treatment date"]
    .first()
    .to_dict()
)

results = shl.batch_plot_daily_syllable_counts(
    root_dir=root,
    output_dir=out_dir,           # <--- NEW: all figures go here
    merge_split_songs=True,
    use_pair_plots=True,
    fixed_vmin=0.0,
    fixed_vmax=5.0,
    show=False,
    treatment_dates=treatment_dates,
)


"""