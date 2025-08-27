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
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _numeric_aware_key(x):
    """
    Sort key that prefers numeric ordering when possible.
    Treats '01', '1', and 1 as the same numeric value for order,
    falling back to case-insensitive string order. Tie-break by
    original string to keep ordering stable if both '01' and '1' exist.
    """
    s = str(x)
    try:
        return (0, int(s), s)
    except Exception:
        return (1, s.lower(), s)


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
    sort_labels_numerically: bool = True,  # enforce 0,1,2,... row order
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

    # Collect / normalize labels
    if syllable_labels is None:
        labels_set: set[str] = set()
        for v in df[label_column]:
            if isinstance(v, dict) and v:
                labels_set.update(map(str, v.keys()))
        syllable_labels = list(labels_set)
    else:
        syllable_labels = list(map(str, syllable_labels))

    if sort_labels_numerically:
        syllable_labels = sorted(syllable_labels, key=_numeric_aware_key)

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
    period: Optional[str] = None,       # "AM", "PM", or None (combined)
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
    sort_rows_numerically: bool = True, # ensure 0,1,2,... row order
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

    # Optional numeric row order
    if sort_rows_numerically:
        new_row_order = sorted([str(x) for x in table.index], key=_numeric_aware_key)
        table = table.reindex(index=new_row_order)

    plot_data = table.copy()
    if log_scale:
        plot_data = np.log10(plot_data + float(pseudocount))

    # Determine color limits
    finite_vals = plot_data.values[np.isfinite(plot_data.values)]
    mask = ~np.isfinite(plot_data.values)

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

    # Proper mathtext label so log10 renders correctly
    cbar_label = (r"$\log_{10}$ " + value_label) if log_scale else value_label

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_data, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
        cbar_kws={"label": cbar_label}, ax=ax,
    )

    ax.set_xticks(np.arange(len(plot_data.columns)) + 0.5)
    ax.set_xticklabels(plot_data.columns, rotation=90, ha="right")
    if period is None:
        ax.set_xlabel("Date (AM / PM)")
    else:
        ax.set_xlabel(f"Date ({period} only)")
    ax.set_ylabel("Syllable Label")

    # Title with mathtext for log base when needed
    if animal_id:
        scale_txt = r"$\log_{10}$" if log_scale else "Normalized"
        if period is None:
            ax.set_title(f"{animal_id} AM vs PM Syllable Occurrence ({scale_txt} Scale)")
        else:
            ax.set_title(f"{animal_id} {period} Syllable Occurrence ({scale_txt} Scale)")

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
    save_path: Optional[Union[str, Path]] = None,   # directory OR a file path (we'll use its parent)
    show: bool = True,
    cmap: str = "Greys",
    verbose: bool = True,
) -> Dict[str, Any]:
    # ── Organizer import (segments-aware first, fallback to legacy)
    _USING_SEGMENTS = False
    try:
        from organize_decoded_with_segments import (
            build_organized_segments_with_durations as _build_organized
        )
        _USING_SEGMENTS = True
    except ImportError:
        try:
            from organize_decoded_dataset import (
                build_organized_dataset as _build_organized
            )
        except ImportError as e:
            raise SystemExit(
                "Could not import an organizer. Ensure either "
                "organize_decoded_with_segments.py or organize_decoded_dataset.py "
                "is importable."
            ) from e

    decoded = Path(decoded_database_json)
    meta = Path(creation_metadata_json)
    if not decoded.exists():
        raise FileNotFoundError(f"Decoded database JSON not found: {decoded}")
    if not meta.exists():
        raise FileNotFoundError(f"Creation metadata JSON not found: {meta}")

    # Build the organized dataset using the available organizer
    if _USING_SEGMENTS:
        out = _build_organized(
            decoded_database_json=decoded,
            creation_metadata_json=meta,
            only_song_present=False,
            compute_durations=False,       # durations not needed for this heatmap
            add_recording_datetime=True,   # harmless; handy for other tasks
        )
    else:
        out = _build_organized(decoded, meta, verbose=verbose)

    organized_df = out.organized_df

    # Treatment date: prefer organizer’s attribute, fallback to raw metadata
    treatment_date = getattr(out, "treatment_date", None)
    if not treatment_date:
        try:
            with meta.open("r") as f:
                treatment_date = json.load(f).get("treatment_date", None)
        except Exception:
            treatment_date = None

    # Build AM/PM table
    table = build_am_pm_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=getattr(out, "unique_syllable_labels", None),
        normalize=normalize,
        sort_labels_numerically=True,
    )

    # Single animal id (if exactly one)
    animal_id = None
    if "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]
    if not animal_id:
        animal_id = "unknown_animal"

    # Resolve output directory and filenames:
    # - If save_path is a file path, use its parent directory.
    # - If save_path is a directory (or None), use it directly (or CWD).
    outdir = Path(save_path) if save_path else Path.cwd()
    if outdir.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)

    base = f"{animal_id}_syllable_heatmap"
    save_combined = outdir / f"{base}_am_pm_combined.png"
    save_am = outdir / f"{base}_am.png"}
    save_pm = outdir / f"{base}_pm.png"}

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
        sort_rows_numerically=True,
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
            sort_rows_numerically=True,
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
            sort_rows_numerically=True,
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
        "combined_png": str(save_combined),
        "am_png": str(save_am),
        "pm_png": str(save_pm),
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AM vs PM syllable heatmap from decoded dataset.")
    p.add_argument("decoded_database_json", type=str)
    p.add_argument("creation_metadata_json", type=str)
    p.add_argument("--normalize", type=str, default="proportion", choices=["proportion", "per_song"])
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--save", type=str, default="", help="Directory or file path; filenames are auto-set to animalID_syllable_heatmap_*.png")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    _ = run_am_pm_syllable_heatmap(
        a.decoded_database_json, a.creation_metadata_json,
        normalize=a.normalize, log_scale=a.log_scale,
        save_path=(a.save or None), show=not a.no_show,
    )
    
    """
    from am_vs_pm_syllable_heatmap import run_am_pm_syllable_heatmap
    from pathlib import Path
    
    # Input JSONs
    decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5323_decoded_database.json"
    meta    = "/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/USA5323_metadata.json"
    
    # Output directory (all three PNGs will be saved here: combined, AM-only, PM-only)
    outdir = Path("/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run
    res = run_am_pm_syllable_heatmap(
        decoded_database_json=decoded,
        creation_metadata_json=meta,
        normalize="proportion",    # "proportion" (default) or "per_song"
        log_scale=True,            # log10 scale, helpful if some values are very low
        save_path=outdir,          # directory or full file path; filenames auto-set
        show=True,                 # True = display plots interactively, False = just save PNGs
        cmap="Greys",              # white→black, low→high
    )
    
    # Inspect outputs
    print("Animal ID:", res["animal_id"])
    print("Treatment date:", res["treatment_date"])
    print("Combined PNG:", res["combined_png"])
    print("AM PNG:", res["am_png"])
    print("PM PNG:", res["pm_png"])
    
    # Example: look at the table backing the heatmap
    print(res["table"].head())

    
    """
