# am_vs_pm_syllable_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "build_am_pm_count_table",
    "plot_am_pm_syllable_heatmap",
    "run_am_pm_syllable_heatmap",
]

# ───────────────────────────────────────────────────────────────────────────────
# Build AM/PM table (rows=labels, columns=YYYY-MM-DD AM/PM)
# ───────────────────────────────────────────────────────────────────────────────
def build_am_pm_count_table(
    organized_df: pd.DataFrame,
    *,
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
    hour_col: str = "Hour",
    minute_col: str = "Minute",
    second_col: str = "Second",
    noon: str = "12:00:00",
    afternoon_end: str = "23:19:59",   # noon → 11:19:59 PM
    syllable_labels: Optional[List[str]] = None,
    normalize: str = "proportion",      # "proportion" (column sums to 1) or "per_song"
) -> pd.DataFrame:
    need = {label_column, date_column, hour_col, minute_col, second_col}
    missing = [c for c in need if c not in organized_df.columns]
    if missing:
        raise KeyError(f"organized_df missing required columns: {missing}")

    df = organized_df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_column])

    for c in (hour_col, minute_col, second_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(lower=0)

    df["TimeOfDay"] = (
        pd.to_timedelta(df[hour_col], unit="h")
        + pd.to_timedelta(df[minute_col], unit="m")
        + pd.to_timedelta(df[second_col], unit="s")
    )

    noon_td = pd.to_timedelta(noon)
    aft_end_td = pd.to_timedelta(afternoon_end)
    df["Period"] = np.where(
        df["TimeOfDay"] < noon_td, "AM",
        np.where(df["TimeOfDay"] <= aft_end_td, "PM", None),
    )
    df = df[df["Period"].notna()].copy()

    if syllable_labels is None:
        labels_set: set[str] = set()
        for v in df[label_column]:
            if isinstance(v, dict) and v:
                labels_set.update(v.keys())
        syllable_labels = sorted(labels_set)

    unique_days = sorted(df[date_column].unique())
    columns: List[str] = []
    for d in unique_days:
        ds = pd.to_datetime(d).strftime("%Y-%m-%d")
        columns.extend([f"{ds} AM", f"{ds} PM"])

    result = pd.DataFrame(0.0, index=syllable_labels, columns=columns)

    def _row_counts(v: dict) -> pd.Series:
        if isinstance(v, dict) and v:
            return pd.Series({lab: float(len(v.get(lab, []))) for lab in syllable_labels}, dtype=float)
        return pd.Series({lab: 0.0 for lab in syllable_labels}, dtype=float)

    for day, g_day in df.groupby(date_column):
        day_str = pd.to_datetime(day).strftime("%Y-%m-%d")
        for period, g in g_day.groupby("Period"):
            if period not in ("AM", "PM"):
                continue
            per_file = [_row_counts(v) for v in g[label_column]]
            if not per_file:
                continue
            mean_counts = pd.concat(per_file, axis=1).fillna(0).mean(axis=1)
            if normalize.lower() == "proportion":
                s = float(mean_counts.sum())
                if s > 0:
                    mean_counts = mean_counts / s
            result.loc[:, f"{day_str} {period}"] = mean_counts

    return result

# ───────────────────────────────────────────────────────────────────────────────
# Heatmap (low values white, high values dark)
# ───────────────────────────────────────────────────────────────────────────────
def plot_am_pm_syllable_heatmap(
    table: pd.DataFrame,
    animal_id: Optional[str] = None,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    *,
    period: Optional[str] = None,       # NEW: "AM", "PM", or None (combined)
    figsize: Tuple[int, int] = (18, 6),
    cmap: str = "Greys",                # default white→black (low→high)
    value_label: str = "Proportion per Period",
    log_scale: bool = False,
    pseudocount: float = 1e-6,          # used only if log_scale
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    line_position: str = "boundary",    # "boundary" or "center"
) -> tuple[plt.Figure, plt.Axes]:
    if table is None or table.empty:
        raise ValueError("Input table is empty.")

    # Sort columns by date and AM(0)/PM(1)
    def _key(col: str) -> tuple[pd.Timestamp, int]:
        parts = col.split()
        dt = pd.to_datetime(parts[0])
        ap = 0 if (len(parts) > 1 and parts[1] == "AM") else 1
        return (dt, ap)

    cols_sorted = sorted(table.columns, key=_key)
    table = table.reindex(columns=cols_sorted)

    # Optional AM/PM filter
    if period is not None:
        period = period.upper()
        if period not in ("AM", "PM"):
            raise ValueError("period must be 'AM', 'PM', or None.")
        keep = [c for c in table.columns if c.endswith(f" {period}")]
        table = table.loc[:, keep]
        if table.empty:
            raise ValueError(f"No columns found for period='{period}'.")

    plot_data = table.copy()
    cbar_label = value_label
    if log_scale:
        plot_data = np.log10(plot_data + float(pseudocount))
        cbar_label = f"log10({value_label} + {pseudocount:g})"

    # Mask NaN/inf so true missing data shows as white
    finite_vals = plot_data.values[np.isfinite(plot_data.values)]
    mask = ~np.isfinite(plot_data.values)

    # Determine color limits
    if vmin is None or vmax is None:
        if log_scale:
            if finite_vals.size:
                vmin = float(np.nanpercentile(finite_vals, 1)) if vmin is None else vmin
                vmax = float(np.nanpercentile(finite_vals, 99)) if vmax is None else vmax
            else:
                vmin = vmin if vmin is not None else -6.0
                vmax = vmax if vmax is not None else 0.0
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_data, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
        cbar_kws={"label": cbar_label}, ax=ax,
    )

    ax.set_xticks(np.arange(len(plot_data.columns)) + 0.5)
    ax.set_xticklabels(plot_data.columns, rotation=45, ha="right")
    if period is None:
        ax.set_xlabel("Date (AM / PM)")
    else:
        ax.set_xlabel(f"Date ({period} only)")
    ax.set_ylabel("Syllable Label")

    title_scale = "Log" if log_scale else "Normalized"
    if animal_id:
        if period is None:
            ax.set_title(f"{animal_id} AM vs PM Syllable Occurrence ({title_scale} Scale)")
        else:
            ax.set_title(f"{animal_id} {period} Syllable Occurrence ({title_scale} Scale)")

    # Treatment line (if the date exists among filtered columns)
    if treatment_date:
        try:
            td = pd.to_datetime(treatment_date).strftime("%Y-%m-%d")
            idxs = [i for i, c in enumerate(plot_data.columns) if c.startswith(td)]
            if idxs:
                i0 = idxs[0]
                x = (i0 if line_position == "boundary" else i0 + 0.5)
                ax.axvline(x=x, color="red", linestyle="--", linewidth=2)
        except Exception:
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
# Convenience wrapper + CLI
# ───────────────────────────────────────────────────────────────────────────────
def run_am_pm_syllable_heatmap(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    normalize: str = "proportion",
    log_scale: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: str = "Greys",
    verbose: bool = True,
) -> Dict[str, Any]:
    try:
        from organize_decoded_dataset import build_organized_dataset
    except ImportError as e:
        raise SystemExit(
            "Could not import build_organized_dataset. Ensure organize_decoded_dataset.py is importable."
        ) from e

    decoded = Path(decoded_database_json)
    meta = Path(creation_metadata_json)
    if not decoded.exists():
        raise FileNotFoundError(f"Decoded database JSON not found: {decoded}")
    if not meta.exists():
        raise FileNotFoundError(f"Creation metadata JSON not found: {meta}")

    out = build_organized_dataset(decoded, meta, verbose=verbose)
    organized_df = out.organized_df

    # Try to read treatment date from metadata JSON
    treatment_date = None
    try:
        with meta.open("r") as f:
            treatment_date = json.load(f).get("treatment_date", None)
    except Exception:
        pass

    # Build AM/PM table
    table = build_am_pm_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=out.unique_syllable_labels,
        normalize=normalize,
    )

    # Single animal id (if exactly one)
    animal_id = None
    if "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]

    # Resolve save paths
    save_combined = save_am = save_pm = None
    if save_path:
        save_path = Path(save_path)
        parent = save_path.parent
        stem = save_path.stem
        ext = save_path.suffix or ".png"
        save_combined = parent / f"{stem}_combined{ext}"
        save_am = parent / f"{stem}_AM{ext}"
        save_pm = parent / f"{stem}_PM{ext}"

    # Plot combined
    fig_all, ax_all = plot_am_pm_syllable_heatmap(
        table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        cmap=cmap,
        value_label=("Avg Count per Song" if normalize == "per_song" else "Proportion per Period"),
        log_scale=log_scale,
        show=show,
        save_path=save_combined,
        line_position="boundary",
        period=None,
    )

    # Plot AM-only (skip gracefully if no AM columns)
    fig_am = ax_am = None
    try:
        fig_am, ax_am = plot_am_pm_syllable_heatmap(
            table,
            animal_id=animal_id,
            treatment_date=treatment_date,
            cmap=cmap,
            value_label=("Avg Count per Song" if normalize == "per_song" else "Proportion per Period"),
            log_scale=log_scale,
            show=show,
            save_path=save_am,
            line_position="boundary",
            period="AM",
        )
    except ValueError as e:
        if verbose:
            print(f"[WARN] AM-only plot skipped: {e}")

    # Plot PM-only (skip gracefully if no PM columns)
    fig_pm = ax_pm = None
    try:
        fig_pm, ax_pm = plot_am_pm_syllable_heatmap(
            table,
            animal_id=animal_id,
            treatment_date=treatment_date,
            cmap=cmap,
            value_label=("Avg Count per Song" if normalize == "per_song" else "Proportion per Period"),
            log_scale=log_scale,
            show=show,
            save_path=save_pm,
            line_position="boundary",
            period="PM",
        )
    except ValueError as e:
        if verbose:
            print(f"[WARN] PM-only plot skipped: {e}")

    return {
        "organized_df": organized_df,
        "table": table,
        "fig_combined": fig_all,
        "ax_combined": ax_all,
        "fig_am": fig_am,
        "ax_am": ax_am,
        "fig_pm": fig_pm,
        "ax_pm": ax_pm,
        "animal_id": animal_id,
        "treatment_date": treatment_date,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AM vs PM syllable heatmap from decoded dataset.")
    p.add_argument("decoded_database_json", type=str)
    p.add_argument("creation_metadata_json", type=str)
    p.add_argument("--normalize", type=str, default="proportion", choices=["proportion", "per_song"])
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    _ = run_am_pm_syllable_heatmap(
        a.decoded_database_json, a.creation_metadata_json,
        normalize=a.normalize, log_scale=a.log_scale,
        save_path=(a.save or None), show=not a.no_show,
    )



"""
import os, sys, importlib
from am_vs_pm_syllable_heatmap import run_am_pm_syllable_heatmap
decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
created = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

res = run_am_pm_syllable_heatmap(
    decoded, created,
    normalize="proportion",           # default; each AM/PM column sums to 1
    log_scale=True,                  # set True if you use per_song and want log
    save_path="figures/USA5288_am_pm.png",
    cmap="Greys",
    show=True,
)

"""
