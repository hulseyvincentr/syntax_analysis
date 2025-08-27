# graph_AM_vs_PM_phrase_duration_over_days.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import: prefer segments-aware, fallback to legacy durations organizer
# ──────────────────────────────────────────────────────────────────────────────
_USING_SEGMENTS = False
try:
    from organize_decoded_with_segments import (
        build_organized_segments_with_durations as _build_organized,
        OrganizedDataset,
    )
    _USING_SEGMENTS = True
except ImportError:
    from organize_decoded_with_durations import (
        build_organized_dataset_with_durations as _build_organized,
        OrganizedDataset,
    )


# ————————————————————————————————————————————————————————————————
# Compatibility wrapper: seaborn 0.13+ uses density_norm; older uses scale
# ————————————————————————————————————————————————————————————————
def _violin_with_backcompat(*, data, x, y, order=None, color="lightgray", inner="quartile"):
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


def _plot_one_syllable_am_pm(
    exploded: pd.DataFrame,
    col: str,
    full_date_range: pd.DatetimeIndex,
    out_path: Path,
    *,
    y_max_ms: int = 25_000,
    jitter: float | bool = True,
    point_size: int = 5,
    point_alpha: float = 0.7,
    figsize: tuple[int, int] = (22, 11),
    font_size_labels: int = 26,
    xtick_fontsize: int = 8,
    xtick_every_days: int = 1,   # label every Nth day (both AM & PM for that day)
    surgery_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
):
    """
    Render a violin+strip plot for one syllable with two categories per day: AM and PM.
    """
    # Prepare long-form with AM/PM classification
    df = exploded.copy()
    df["DateDay"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with valid date, hour, and duration
    df = df.dropna(subset=["DateDay", "Hour", col])
    if df.empty:
        return

    df["AMPM"] = pd.Series(["AM" if h < 12 else "PM" for h in df["Hour"]], index=df.index)

    # Build the ordered categorical spanning every (day, AM), (day, PM)
    full_cats = []
    for d in full_date_range:
        full_cats.append(f"{d.strftime('%Y-%m-%d')} AM")
        full_cats.append(f"{d.strftime('%Y-%m-%d')} PM")

    df["DateAMPM"] = df["DateDay"].dt.strftime("%Y-%m-%d") + " " + df["AMPM"]
    df["DateAMPM"] = pd.Categorical(df["DateAMPM"], categories=full_cats, ordered=True)

    # If everything is outside the range/categories, skip
    if df["DateAMPM"].isna().all():
        return

    sns.set(style="white")
    plt.figure(figsize=figsize)

    # One call per layer with fixed order → perfect alignment (including empty categories)
    _violin_with_backcompat(
        data=df, x="DateAMPM", y=col, order=full_cats,
        color="lightgray", inner="quartile"
    )
    sns.stripplot(
        data=df, x="DateAMPM", y=col, order=full_cats,
        jitter=jitter, size=point_size, color="#2E4845", alpha=point_alpha
    )

    # Axes cosmetics
    ax = plt.gca()
    plt.ylim(0, y_max_ms)
    plt.xlabel("Recording Date (AM / PM)", fontsize=font_size_labels)
    plt.ylabel("Phrase Duration (ms)", fontsize=font_size_labels)

    # X-tick labeling: label both AM & PM but only every Nth *day* to reduce clutter
    # Day i corresponds to indices (2*i) -> AM, (2*i+1) -> PM
    day_step = max(1, int(xtick_every_days))
    tick_idx = []
    tick_labels = []
    for i, d in enumerate(full_date_range):
        if i % day_step == 0:
            tick_idx.extend([2 * i, 2 * i + 1])
            tick_labels.extend([f"{d:%Y-%m-%d} AM", f"{d:%Y-%m-%d} PM"])

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=xtick_fontsize)

    # Surgery date line (optional). Place it between AM and PM of the surgery day (after AM).
    if surgery_date_dt is not None and not pd.isna(surgery_date_dt):
        sday = surgery_date_dt.normalize()
        day_idx = full_date_range.get_indexer([sday])[0]
        if day_idx >= 0:
            x_at_boundary = 2 * day_idx + 1  # between AM (2*i) and PM (2*i+1)
            plt.axvline(x=x_at_boundary, color="red", linestyle="--", label="Surgery Date")
            plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=300, transparent=False)
    if show_plots:
        plt.show()
    else:
        plt.close()


def graph_AM_vs_PM_phrase_duration_over_days(
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path,
    save_output_to_this_file_path: str | Path,
    *,
    only_song_present: bool = False,
    y_max_ms: int = 25_000,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    figsize: tuple[int, int] = (22, 11),
    font_size_labels: int = 26,
    xtick_fontsize: int = 8,
    xtick_every_days: int = 1,            # show AM/PM labels every Nth day
    show_plots: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
) -> OrganizedDataset:
    """
    Build dataset (with syllable_<label>_durations) and save one figure per syllable
    showing **two** distributions per day: AM (00:00–11:59) and PM (12:00–23:59).

    Returns the OrganizedDataset for further analysis.
    """
    save_dir = Path(save_output_to_this_file_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build organized dataset (segments-aware or legacy). We need durations → compute_durations=True.
    if _USING_SEGMENTS:
        out = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=True,
            add_recording_datetime=True,
        )
    else:
        out = _build_organized(
            decoded_database_json=decoded_database_json,
            creation_metadata_json=creation_metadata_json,
            only_song_present=only_song_present,
            compute_durations=True,
        )

    df = out.organized_df.copy()
    # Validate needed columns
    for need in ["Date", "Hour"]:
        if need not in df.columns:
            # Nothing to plot if these are missing
            return out

    # Ensure Date is datetime and build the full date window
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if not df["Date"].notna().any():
        return out

    earliest_date = df["Date"].min().normalize()
    latest_date = df["Date"].max().normalize()
    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq="D")

    # Optional surgery date
    surgery_date_dt = (
        pd.to_datetime(out.treatment_date, format="%Y.%m.%d", errors="coerce")
        if out.treatment_date else None
    )

    # Which syllables to plot?
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

        exploded = df[["Date", "Hour", col]].explode(col)
        if exploded[col].dropna().empty:
            continue

        out_path = save_dir / f"{animal_id}_syllable_{syllable_label}_AMPM_phrase_duration_plot.png"
        _plot_one_syllable_am_pm(
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
            xtick_every_days=xtick_every_days,
            surgery_date_dt=surgery_date_dt,
            show_plots=show_plots,
        )

    return out


# ——————————————————————————————————————————
# Optional CLI
# ——————————————————————————————————————————
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot per-syllable phrase durations with AM and PM distributions per day."
    )
    ap.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    ap.add_argument("creation_metadata_json", type=str, help="Path to *_creation_data.json")
    ap.add_argument("save_dir", type=str, help="Directory to save PNGs")
    ap.add_argument("--only-song-present", action="store_true", help="Filter to rows with song_present == True")
    ap.add_argument("--y-max-ms", type=int, default=25000, help="Y-axis upper limit in ms")
    ap.add_argument("--xtick-every-days", type=int, default=1, help="Label AM/PM for every Nth day")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    ap.add_argument("--syllables", type=str, nargs="*", default=None, help="Subset of syllable labels to plot")
    args = ap.parse_args()

    graph_AM_vs_PM_phrase_duration_over_days(
        decoded_database_json=args.decoded_database_json,
        creation_metadata_json=args.creation_metadata_json,
        save_output_to_this_file_path=args.save_dir,
        only_song_present=args.only_song_present,
        y_max_ms=args.y_max_ms,
        xtick_every_days=args.xtick_every_days,
        show_plots=args.show,
        syllables_subset=args.syllables,
    )


"""
from graph_AM_vs_PM_phrase_duration_over_days import graph_AM_vs_PM_phrase_duration_over_days

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"
outdir  = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures"

out = graph_AM_vs_PM_phrase_duration_over_days(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    save_output_to_this_file_path=outdir,
    only_song_present=False,   # set to True if you want to drop rows where song_present == False
    y_max_ms=2500,
    show_plots=True,           # True = display each plot interactively, False = just save PNGs
)
"""
