# syllable_heatmap_linear.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "_infer_animal_id",
    "build_daily_avg_count_table",
    "plot_linear_scaled_syllable_counts",
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

# ---------- table builder ----------

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
                    pass
        for lab in counts:
            table.at[date, lab] = counts[lab] / max(1, n_songs)  # avg per song
    return table

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
    """
    if count_table.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show: plt.show()
        else: plt.close(fig)
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
        figsize=(max(8, 0.28*len(dates)+4), max(4, 0.30*len(labels)+2))
    )
    im = ax.imshow(M, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

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
                ax.vlines(j, -0.5, len(labels)-0.5,
                          colors="red", linestyles="--", linewidth=1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax
