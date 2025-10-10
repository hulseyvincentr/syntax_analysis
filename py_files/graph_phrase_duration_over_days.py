# graph_phrase_duration_over_days.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        # previous segments-aware module (needs metadata)
        from organize_decoded_with_segments import (
            build_organized_segments_with_durations as _build_organized,
            OrganizedDataset,
        )
    except ImportError:
        # legacy durations-only organizer (needs metadata)
        from organize_decoded_with_durations import (
            build_organized_dataset_with_durations as _build_organized,
            OrganizedDataset,
        )

# ————————————————————————————————————————————————————————————————
# Compatibility wrapper: seaborn 0.13+ uses density_norm; older uses scale
# ————————————————————————————————————————————————————————————————
def _violin_with_backcompat(
    *, data, x, y, order=None, color="lightgray", inner="quartile"
):
    try:
        return sns.violinplot(
            data=data, x=x, y=y, order=order,
            inner=inner, density_norm="width", color=color
        )
    except TypeError:
        return sns.violinplot(
            data=data, x=x, y=y, order=order,
            inner=inner, scale="width", color=color
        )


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
    xtick_every: int = 1,  # label every Nth day
    treatment_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
):
    """Render a violin+strip plot for one syllable with perfectly aligned x-axis."""
    # Normalize to midnight and create ordered categorical covering the entire window
    exploded = exploded.copy()
    exploded["DateDay"] = pd.to_datetime(exploded["Date"]).dt.normalize()
    exploded[col] = pd.to_numeric(exploded[col], errors="coerce")
    exploded = exploded.dropna(subset=[col])
    if exploded.empty:
        return

    exploded["DateCat"] = pd.Categorical(
        exploded["DateDay"],
        categories=full_date_range,   # includes days with no data → preserves gaps
        ordered=True
    )

    sns.set(style="white")  # no gridlines
    plt.figure(figsize=figsize)

    # Single calls with fixed 'order' ensure category positions match tick indices
    _violin_with_backcompat(
        data=exploded, x="DateCat", y=col, order=full_date_range,
        color="lightgray", inner="quartile"
    )
    sns.stripplot(
        data=exploded, x="DateCat", y=col, order=full_date_range,
        jitter=jitter, size=point_size, color="#2E4845", alpha=point_alpha
    )

    # Axes cosmetics
    ax = plt.gca()
    plt.ylim(0, y_max_ms)
    plt.xlabel("Recording Date", fontsize=font_size_labels)
    plt.ylabel("Phrase Duration (ms)", fontsize=font_size_labels)

    # Ticks = integer positions; labels = dates at that spacing
    tick_idx = list(range(0, len(full_date_range), max(1, int(xtick_every))))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [full_date_range[i].strftime("%Y-%m-%d") for i in tick_idx],
        rotation=90, fontsize=xtick_fontsize
    )

    # Treatment date (optional): draw at its categorical index
    if treatment_date_dt is not None and not pd.isna(treatment_date_dt):
        t_idx = full_date_range.get_indexer([treatment_date_dt.normalize()])[0]
        if t_idx >= 0:
            plt.axvline(x=t_idx, color="red", linestyle="--", label="Treatment Date")
            plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=300, transparent=False)
    if show_plots:
        plt.show()
    else:
        plt.close()


def _coerce_treatment_date(
    user_treatment_date: Optional[str | pd.Timestamp],
    fallback_str: Optional[str],
) -> Optional[pd.Timestamp]:
    """
    Try to parse a user-supplied treatment_date first; otherwise fall back to a string like 'YYYY.MM.DD'.
    Accepts:
      • 'YYYY-MM-DD' or 'YYYY.MM.DD' (or anything pandas can parse)
      • pandas.Timestamp
      • None → returns None
    """
    if user_treatment_date is not None:
        if isinstance(user_treatment_date, pd.Timestamp):
            return user_treatment_date.normalize()
        # try very permissive parsing
        dt = pd.to_datetime(user_treatment_date, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()

    if fallback_str:
        # Try strict dotted format first, then general parsing
        dt = pd.to_datetime(fallback_str, format="%Y.%m.%d", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(fallback_str, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()

    return None


def graph_phrase_duration_over_days(
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path | None,  # ignored when using Excel-serial organizer
    save_output_to_this_file_path: str | Path,
    *,
    only_song_present: bool = False,
    y_max_ms: int = 25_000,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    dpi: int = 300,                        # kept for signature compatibility
    transparent: bool = True,              # kept for signature compatibility
    figsize: tuple[int, int] = (20, 11),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,                  # label every Nth day to reduce clutter
    show_plots: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
    treatment_date: Optional[str | pd.Timestamp] = None,   # <<< NEW
) -> OrganizedDataset:
    """
    Build the organized dataset (with syllable_<label>_durations) and
    save a violin+strip plot per syllable across all calendar days from
    the earliest to latest recording date (missing days are shown as gaps).

    Notes
    -----
    • When using `organized_decoded_serialTS_segments`, `creation_metadata_json` is ignored.
    • Recording DateTime is derived from the Excel serial embedded in filenames.
    • `treatment_date`: string ('YYYY-MM-DD' or 'YYYY.MM.DD') or pandas.Timestamp. If provided,
      it overrides any `out.treatment_date`. A red dashed line is drawn at that date.
    """
    save_dir = Path(save_output_to_this_file_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build organized dataset (segments-aware or legacy). We need durations → compute_durations=True.
    if _USING_SERIAL_SEGMENTS:
        out = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=None,          # explicitly ignored by the serial organizer
            only_song_present=only_song_present,
            compute_durations=True,
            add_recording_datetime=True,          # derive time from Excel serial
        )
    else:
        out = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=True,
        )

    df = out.organized_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if not df["Date"].notna().any():
        return out  # no valid dates → nothing to plot

    earliest_date = df["Date"].min().normalize()
    latest_date = df["Date"].max().normalize()
    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq="D")

    # Treatment date: prefer user-provided value; otherwise fall back to out.treatment_date
    treatment_date_dt = _coerce_treatment_date(treatment_date, out.treatment_date)

    # Choose which syllables to render
    labels = out.unique_syllable_labels
    if syllables_subset is not None:
        labels = [lab for lab in labels if lab in set(syllables_subset)]
    if not labels:
        return out

    # For filenames
    try:
        animal_id = str(df["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"

    # Plot each syllable
    for syllable_label in labels:
        col = f"syllable_{syllable_label}_durations"
        if col not in df.columns:
            continue

        exploded = df[["Date", col]].explode(col)
        # If everything is NA/empty for this syllable, skip cleanly
        if exploded[col].dropna().empty:
            continue

        out_path = save_dir / f"{animal_id}_syllable_{syllable_label}_phrase_duration_plot.png"
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
            treatment_date_dt=treatment_date_dt,   # <<< pass into plotter
            show_plots=show_plots,
        )

    return out


# ——————————————————————————————————————————
# Optional CLI
# ——————————————————————————————————————————
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plot per-syllable phrase duration distributions across calendar days.")
    ap.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    ap.add_argument(
        "creation_metadata_json",
        nargs="?",
        default=None,
        help="Path to *_creation_data.json (ignored when using organized_decoded_serialTS_segments.py)"
    )
    ap.add_argument("save_dir", type=str, help="Directory to save PNGs")
    ap.add_argument("--only-song-present", action="store_true", help="Filter to rows with song_present == True")
    ap.add_argument("--y-max-ms", type=int, default=25000, help="Y-axis upper limit in ms")
    ap.add_argument("--xtick-every", type=int, default=1, help="Label every Nth day on x-axis")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    ap.add_argument("--syllables", type=str, nargs="*", default=None, help="Subset of syllable labels to plot")
    ap.add_argument(
        "--treatment-date",
        type=str,
        default=None,
        help="Date of treatment (e.g., '2024-07-15' or '2024.07.15'); draws red dashed line."
    )
    args = ap.parse_args()

    graph_phrase_duration_over_days(
        decoded_database_json=args.decoded_database_json,
        creation_metadata_json=args.creation_metadata_json,
        save_output_to_this_file_path=args.save_dir,
        only_song_present=args.only_song_present,
        y_max_ms=args.y_max_ms,
        xtick_every=args.xtick_every,
        show_plots=args.show,
        syllables_subset=args.syllables,
        treatment_date=args.treatment_date,   # <<< NEW
    )


"""
Example usage:

from graph_phrase_duration_over_days import graph_phrase_duration_over_days

decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_RC6_Comp2_decoded_database.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08"

# Option A: provide a treatment_date explicitly (preferred)
out = graph_phrase_duration_over_days(
    decoded_database_json=decoded,
    creation_metadata_json=None,   # ignored when using organized_decoded_serialTS_segments
    save_output_to_this_file_path=outdir,
    only_song_present=True,
    y_max_ms=25000,
    show_plots=True,
    treatment_date="2025-03-04",   # or "2024.07.15"
)

# Option B: omit treatment_date to fall back to out.treatment_date if present
# out = graph_phrase_duration_over_days(decoded, None, outdir)
"""
