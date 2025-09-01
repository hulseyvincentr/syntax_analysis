# batch_syllable_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = [
    "build_batches_of_count_table",
    "plot_batch_syllable_heatmap",
    "run_batch_syllable_heatmap",
]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _numeric_aware_key(x):
    """
    Numeric-aware sort key: '01','1',1 → same order, else fallback to string.
    """
    s = str(x)
    try:
        return (0, int(s), s)
    except Exception:
        return (1, s.lower(), s)


# ───────────────────────────────────────────────────────────────────────────────
# Build table (rows=labels, columns=Batch_001, Batch_002, ...)
# ───────────────────────────────────────────────────────────────────────────────
def build_batches_of_count_table(
    organized_df: pd.DataFrame,
    *,
    label_column: str = "syllable_onsets_offsets_ms_dict",
    syllable_labels: Optional[List[str]] = None,
    normalize: str = "proportion",      # "proportion" (column sums to 1) or "per_song"
    sort_labels_numerically: bool = True,
    batch_size: int = 100,
    start_at_index: int = 0,
) -> pd.DataFrame:
    """
    Build a table of syllable usage, grouped in consecutive batches of N songs.
    Only rows with song_present == True are included.

    Each row in `organized_df` is assumed to be one song/recording.
    The `label_column` should contain dicts mapping syllable label -> list of events
    (the list length is treated as the count).
    """
    if label_column not in organized_df.columns:
        raise KeyError(f"organized_df missing required column: {label_column}")

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    # Filter to only song_present == True
    if "song_present" in organized_df.columns:
        df = organized_df[organized_df["song_present"] == True].copy()
    else:
        df = organized_df.copy()

    df = df.reset_index(drop=True)

    # Optionally skip initial rows
    if start_at_index > 0:
        df = df.iloc[start_at_index:].reset_index(drop=True)

    # Determine label set
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

    # Assign batch numbers from row order (after filtering)
    df["Batch"] = (df.index // batch_size) + 1

    # Prepare result table
    unique_batches = df["Batch"].unique()
    unique_batches.sort()
    columns = [f"Batch_{i:03d}" for i in unique_batches]
    result = pd.DataFrame(0.0, index=syllable_labels, columns=columns)

    # Helper: count labels in one row
    def _row_counts(v: dict) -> pd.Series:
        if isinstance(v, dict) and v:
            return pd.Series(
                {lab: float(len(v.get(lab, []))) for lab in syllable_labels},
                dtype=float,
            )
        return pd.Series({lab: 0.0 for lab in syllable_labels}, dtype=float)

    # Aggregate per batch
    for b, g in df.groupby("Batch"):
        per_file = [_row_counts(v) for v in g[label_column]]
        if not per_file:
            continue
        mean_counts = pd.concat(per_file, axis=1).fillna(0).mean(axis=1)

        if normalize.lower() == "proportion":
            s = float(mean_counts.sum())
            if s > 0:
                mean_counts = mean_counts / s

        result.loc[:, f"Batch_{b:03d}"] = mean_counts

    return result


# ───────────────────────────────────────────────────────────────────────────────
# Heatmap (low values white, high values dark) for batch columns
# ───────────────────────────────────────────────────────────────────────────────
def plot_batch_syllable_heatmap(
    table: pd.DataFrame,
    animal_id: Optional[str] = None,
    *,
    batch_size: Optional[int] = None,   # include in title
    figsize: Tuple[int, int] = (18, 6),
    cmap: str = "Greys",
    value_label: str = "Proportion per Batch",
    log_scale: bool = False,
    pseudocount: float = 1e-6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    sort_rows_numerically: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap from a batch-based syllable table."""
    if table is None or table.empty:
        raise ValueError("Input table is empty.")

    # Sort columns numerically by batch number
    def _batch_key(col: str) -> int:
        try:
            if col.startswith("Batch_"):
                return int(col.split("_")[1])
        except Exception:
            pass
        return int(10**9)

    cols_sorted = sorted(table.columns, key=_batch_key)
    table = table.reindex(columns=cols_sorted)

    # Optional numeric row order
    if sort_rows_numerically:
        new_row_order = sorted([str(x) for x in table.index], key=_numeric_aware_key)
        table = table.reindex(index=new_row_order)

    plot_data = table.copy()
    if log_scale:
        plot_data = np.log10(plot_data + float(pseudocount))

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

    cbar_label = (r"$\log_{10}$ " + value_label) if log_scale else value_label

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_data, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
        cbar_kws={"label": cbar_label}, ax=ax,
    )

    ax.set_xticks(np.arange(len(plot_data.columns)) + 0.5)
    ax.set_xticklabels(plot_data.columns, rotation=90, ha="right")
    ax.set_xlabel("Consecutive Batches (song_present=True)")
    ax.set_ylabel("Syllable Label")

    if animal_id:
        scale_txt = r"$\log_{10}$" if log_scale else "Normalized"
        if batch_size:
            ax.set_title(f"{animal_id} Syllable Occurrence by {batch_size}-Song Batches ({scale_txt} Scale)")
        else:
            ax.set_title(f"{animal_id} Syllable Occurrence by Batches ({scale_txt} Scale)")

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# ───────────────────────────────────────────────────────────────────────────────
# Wrapper
# ───────────────────────────────────────────────────────────────────────────────
def run_batch_syllable_heatmap(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    normalize: str = "proportion",
    log_scale: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: str = "Greys",
    batch_size: int = 100,
    start_at_index: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Wrapper: organize dataset, build batch-based table (song_present only), plot heatmap."""
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

    # Build the organized dataset
    if _USING_SEGMENTS:
        out = _build_organized(
            decoded_database_json=decoded,
            creation_metadata_json=meta,
            only_song_present=False,
            compute_durations=False,
            add_recording_datetime=True,
        )
    else:
        out = _build_organized(decoded, meta, verbose=verbose)

    organized_df = out.organized_df

    # Build batch-based table using only song_present == True
    table = build_batches_of_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        syllable_labels=getattr(out, "unique_syllable_labels", None),
        normalize=normalize,
        sort_labels_numerically=True,
        batch_size=batch_size,
        start_at_index=start_at_index,
    )

    # Single animal id (if exactly one)
    animal_id = None
    if "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]
    if not animal_id:
        animal_id = "unknown_animal"

    outdir = Path(save_path) if save_path else Path.cwd()
    if outdir.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)

    save_png = outdir / f"{animal_id}_syllable_heatmap_batches.png"

    fig, ax = plot_batch_syllable_heatmap(
        table,
        animal_id=animal_id,
        cmap=cmap,
        value_label=("Avg Count per Song" if normalize == "per_song" else "Proportion per Batch"),
        log_scale=log_scale,
        show=show,
        save_path=save_png,
        sort_rows_numerically=True,
        batch_size=batch_size,  # pass through for title
    )

    return {
        "organized_df": organized_df,
        "table": table,
        "fig": fig,
        "ax": ax,
        "animal_id": animal_id,
        "png": str(save_png),
        "batch_size": batch_size,
        "start_at_index": start_at_index,
        "normalize": normalize,
        "log_scale": log_scale,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Batch (every N songs) syllable heatmap from decoded dataset.")
    p.add_argument("decoded_database_json", type=str)
    p.add_argument("creation_metadata_json", type=str)
    p.add_argument("--normalize", type=str, default="proportion", choices=["proportion", "per_song"])
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--save", type=str, default="", help="Directory or file path; filename auto-set to <animal>_syllable_heatmap_batches.png")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--batch-size", type=int, default=100, help="Songs per batch (default: 100)")
    p.add_argument("--start-at-index", type=int, default=0, help="Row offset before forming batches")
    a = p.parse_args()

    _ = run_batch_syllable_heatmap(
        a.decoded_database_json, a.creation_metadata_json,
        normalize=a.normalize, log_scale=a.log_scale,
        save_path=(a.save or None), show=not a.no_show,
        batch_size=a.batch_size, start_at_index=a.start_at_index,
    )

    """
    # Example (Python session):
from batch_syllable_heatmap import run_batch_syllable_heatmap

decoded = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5497/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5497_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5497/USA5497_metadata.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5497/batch_figures"
    
res = run_batch_syllable_heatmap(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    normalize="proportion",
    log_scale=True,
    save_path=outdir,
    show=True,
    cmap="Greys",
    batch_size=100,
    start_at_index=0,
)

    print("Animal ID:", res["animal_id"])
    print("PNG:", res["png"])
    print(res["table"].head())
        """

