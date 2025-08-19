# syllable_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ["plot_log_scaled_syllable_counts", "build_daily_avg_count_table"]


# ───────────────────────────────────────────────────────────────────────────────
# Heatmap (log10-scaled) of daily average syllable counts per song
# ───────────────────────────────────────────────────────────────────────────────
def plot_log_scaled_syllable_counts(
    count_table: pd.DataFrame,
    animal_id: Optional[str] = None,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    *,
    # Visual/behavior knobs
    pseudocount: float = 1e-3,
    figsize: Tuple[int, int] = (16, 6),
    cmap: str = "Greys_r",                  # reversed so low/zero ≈ white
    cbar_label: str = "Log10 Avg Count per Song",
    sort_dates: bool = True,                # ensure chronological columns
    date_format: str = "%Y-%m-%d",
    show: bool = True,                      # show the figure
    save_path: Optional[Union[str, Path]] = None,  # if provided, save here
    # Treatment-line behavior
    mark_treatment: bool = True,
    nearest_match: bool = True,             # if False, require exact date match
    max_days_off: int = 1,                  # tolerance for nearest match (days)
    line_position: str = "center",          # "center" or "boundary"
    line_kwargs: Optional[dict] = None,     # e.g., {"color": "red", "linestyle": "--", "linewidth": 2}
):
    """
    Plot a log10-scaled heatmap of per-syllable average counts per day.

    Parameters
    ----------
    count_table
        DataFrame: rows = syllable labels, columns = dates, values = avg count per song.
    animal_id
        Optional; used in the plot title.
    treatment_date
        Optional; pd.Timestamp or parseable string.
    pseudocount
        Value added before log10 to avoid log(0); default 1e-3 -> zeros map to -3.
    figsize
        Figure size (width, height).
    cmap
        Matplotlib colormap. Default 'Greys_r' makes low values light/white.
    cbar_label
        Label for the colorbar.
    sort_dates
        Sort columns chronologically by their datetime interpretation.
    date_format
        Format used for x-axis tick labels.
    show
        If True, calls plt.show(). Set False in larger pipelines.
    save_path
        If provided, saves the figure to this path.
    mark_treatment
        If True and treatment_date is provided, draw a vertical dashed line.
    nearest_match
        If True, mark the nearest date column within 'max_days_off' days;
        if False, only mark if exact column match exists.
    max_days_off
        Integer day tolerance for nearest-match behavior.
    line_position
        'center' draws through the center of the matched column;
        'boundary' draws on the left boundary of that column.
    line_kwargs
        Dict of kwargs passed to ax.axvline (color, linestyle, linewidth, etc.).

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and Axes.
    """
    if count_table is None or count_table.empty:
        raise ValueError("count_table is empty or None.")

    # Optionally sort columns chronologically
    columns = count_table.columns
    if sort_dates:
        try:
            columns = sorted(columns, key=pd.to_datetime)
            count_table = count_table.reindex(columns=columns)
        except Exception:
            # If parsing fails, proceed with original order
            pass

    # Log transform with pseudocount
    log_scaled_table = np.log10(count_table + float(pseudocount))

    # Build figure/axes
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        log_scaled_table,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    # Prepare and set x-ticks explicitly (centers of cells)
    date_idx = pd.to_datetime(log_scaled_table.columns)
    xticks = np.arange(len(date_idx)) + 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime(date_format) for d in date_idx], rotation=45, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Syllable Label")

    if animal_id:
        ax.set_title(f"{animal_id} Daily Avg Occurrence of Each Syllable (Log₁₀ Scale)")

    # Treatment date marker
    if mark_treatment and treatment_date is not None:
        try:
            td = pd.to_datetime(treatment_date)
            if len(date_idx) > 0 and not pd.isna(td):
                if nearest_match:
                    diffs = np.abs((date_idx - td).days)
                    j = int(np.argmin(diffs))
                    ok = diffs[j] <= int(max_days_off)
                else:
                    matches = np.where(date_idx == td)[0]
                    j = int(matches[0]) if len(matches) else -1
                    ok = j >= 0

                if ok:
                    x = j + 0.5 if line_position == "center" else j
                    default_line = {"color": "red", "linestyle": "--", "linewidth": 2}
                    if line_kwargs:
                        default_line.update(line_kwargs)
                    ax.axvline(x=x, **default_line)
        except Exception:
            # Fail silently if parsing/marking goes wrong
            pass

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


# ───────────────────────────────────────────────────────────────────────────────
# Build a count_table (rows=labels, cols=dates) from organized_df
# ───────────────────────────────────────────────────────────────────────────────
def build_daily_avg_count_table(
    organized_df: pd.DataFrame,
    *,
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
    syllable_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Construct a table where each cell is the **average count per song** of a syllable on a day.

    Parameters
    ----------
    organized_df
        DataFrame produced by build_organized_dataset(); must contain a dict column of
        {label -> list of [onset_ms, offset_ms] ...} and a datetime 'Date' column.
    label_column
        Column holding the dict mapping syllable labels to intervals.
    date_column
        Column with pandas datetime per row (will be normalized to midnight).
    syllable_labels
        Optional explicit label list; defaults to union of labels in the dataset (sorted).

    Returns
    -------
    pd.DataFrame
        Index = syllable labels (sorted), Columns = unique dates (Timestamp), Values = mean counts per song.
    """
    if label_column not in organized_df.columns or date_column not in organized_df.columns:
        raise KeyError(f"organized_df must contain '{label_column}' and '{date_column}' columns.")

    df = organized_df.copy()
    # Normalize dates to day (midnight) to avoid dup columns from different times
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_column])

    # Collect label universe if not given
    if syllable_labels is None:
        labels_set: set[str] = set()
        for v in df[label_column]:
            if isinstance(v, dict) and v:
                labels_set.update(v.keys())
        syllable_labels = sorted(labels_set)

    # Prepare empty result table
    unique_days = sorted(df[date_column].dropna().unique())
    count_table = pd.DataFrame(0.0, index=syllable_labels, columns=unique_days)

    # Group by day and compute mean counts per file
    for day, g in df.groupby(date_column):
        # Build a list of Series (one per file) with counts per label
        per_file = []
        for v in g[label_column]:
            if isinstance(v, dict) and v:
                counts = {lab: len(v.get(lab, [])) for lab in syllable_labels}
            else:
                counts = {lab: 0 for lab in syllable_labels}
            per_file.append(pd.Series(counts, dtype=float))

        if len(per_file) > 0:
            mean_counts = pd.concat(per_file, axis=1).fillna(0).mean(axis=1)
            count_table.loc[:, day] = mean_counts

    return count_table


# ───────────────────────────────────────────────────────────────────────────────
# Example usage (CLI-friendly): build organized data, compute count_table, plot
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    # We import your organizer from the sibling module you created earlier.
    try:
        from organize_decoded_dataset import build_organized_dataset
    except ImportError as e:
        raise SystemExit(
            "Could not import build_organized_dataset. Make sure organize_decoded_dataset.py "
            "is on your PYTHONPATH or in the same folder."
        ) from e

    p = argparse.ArgumentParser(
        description="Build daily avg syllable-count heatmap (log10) from decoded dataset."
    )
    p.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    p.add_argument("creation_metadata_json", type=str, help="Path to *_creation_data.json")
    p.add_argument("--save", type=str, default="", help="Optional path to save the figure (e.g., figures/heatmap.png)")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    p.add_argument("--nearest-line", action="store_true", help="Mark nearest date to treatment (within 1 day)")
    args = p.parse_args()

    # 1) Organize dataset
    out = build_organized_dataset(
        decoded_database_json=args.decoded_database_json,
        creation_metadata_json=args.creation_metadata_json,
        verbose=True,
    )
    organized_df = out.organized_df

    # 2) Build the count_table (avg counts per song per day)
    count_table = build_daily_avg_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=out.unique_syllable_labels,  # ensure stable label order
    )

    # 3) Pull animal_id (if consistent across files)
    animal_id = None
    if "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]

    # 4) Read treatment_date from metadata JSON (if present)
    treatment_date = None
    try:
        with open(args.creation_metadata_json, "r") as f:
            meta = json.load(f)
        if isinstance(meta.get("treatment_date", None), str):
            treatment_date = meta["treatment_date"]
    except Exception:
        pass

    # 5) Plot
    _, _ = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        show=not args.no_show,
        save_path=(args.save or None),
        sort_dates=True,
        nearest_match=args.nearest_line,
        max_days_off=1,
        cmap="Greys",
    )

"""
from organize_decoded_dataset import build_organized_dataset
from syllable_heatmap import build_daily_avg_count_table, plot_log_scaled_syllable_counts

decoded = "/path/to/USA5288_decoded_database.json"
meta    = "/path/to/USA5288_creation_data.json"

out = build_organized_dataset(decoded, meta, verbose=True)
df  = out.organized_d
"""