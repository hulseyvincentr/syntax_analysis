# syllable_heatmap_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import json
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import preference order:
#   1) serialTS segments (Excel-serial timestamps; no metadata required)
#   2) segments-aware organizer
#   3) legacy organizer
#   4) durations organizer
# ──────────────────────────────────────────────────────────────────────────────
_ORG_MODE = "serialTS"  # "serialTS" | "segments" | "legacy" | "durations"

try:
    # Preferred: Excel-serial-based organizer (ignores metadata)
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _build_organized,
    )
    _ORG_MODE = "serialTS"
except ImportError:
    try:
        from organize_decoded_with_segments import (
            build_organized_segments_with_durations as _build_organized,
        )
        _ORG_MODE = "segments"
    except ImportError:
        try:
            from organize_decoded_dataset import build_organized_dataset as _build_organized  # type: ignore
            _ORG_MODE = "legacy"
        except ImportError:
            from organize_decoded_with_durations import (  # type: ignore
                build_organized_dataset_with_durations as _build_organized
            )
            _ORG_MODE = "durations"

from syllable_heatmap import (
    build_daily_avg_count_table,
    plot_log_scaled_syllable_counts,
)


def _coerce_timestamp_or_str(v: Optional[Union[str, pd.Timestamp]]) -> Optional[Union[str, pd.Timestamp]]:
    """Accept strings like 'YYYY-MM-DD'/'YYYY.MM.DD' or pd.Timestamp; return parsed Timestamp or original string."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v
    if isinstance(v, str) and v.strip():
        s = v.strip()
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(s, errors="raise")
        except Exception:
            return s
    return None


def run_daily_syllable_heatmap(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Optional[Union[str, Path]] = None,  # optional to support serialTS
    *,
    # NEW: directory to save the figure; filename is auto-named by animal_id
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    nearest_match: bool = True,
    max_days_off: int = 1,
    sort_dates: bool = True,
    pseudocount: float = 1e-3,
    cmap: str = "Greys",
    date_format: str = "%Y-%m-%d",
    # dataframe options
    label_column: str = "syllable_onsets_offsets_ms_dict",
    date_column: str = "Date",
    # overrides
    animal_id_override: Optional[str] = None,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    # misc
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Organize decoded dataset, compute per-day avg syllable counts, and plot a log10-scaled heatmap.

    Parameters
    ----------
    decoded_database_json : str|Path
    creation_metadata_json : Optional[str|Path]
        Optional for serialTS; required by some fallback organizers.
    output_dir : Optional[str|Path]
        Directory where the figure will be saved. The filename is auto-named as
        '<animal_id>_syllable_heatmap.png'. If animal_id cannot be resolved, falls
        back to '<decoded_stem>_syllable_heatmap.png'. If None, the figure is not saved.
    show : bool
        Whether to display the plot window.

    Returns
    -------
    dict with keys:
      - organized_df (pd.DataFrame)
      - count_table (pd.DataFrame)
      - fig, ax (Matplotlib objects)
      - animal_id (str|None)
      - treatment_date (str|pd.Timestamp|None)
      - save_path (Path|None)
    """
    decoded = Path(decoded_database_json)
    if not decoded.exists():
        raise FileNotFoundError(f"Decoded database JSON not found: {decoded}")

    meta: Optional[Path] = Path(creation_metadata_json) if creation_metadata_json else None
    if _ORG_MODE in {"segments", "legacy", "durations"} and (meta is None or not meta.exists()):
        raise FileNotFoundError(
            "Creation metadata JSON is required for this organizer mode "
            f"({_ORG_MODE}) but was missing/not found."
        )

    # 1) Organize dataset
    if _ORG_MODE in {"serialTS", "segments"}:
        out = _build_organized(
            decoded_database_json=decoded,
            creation_metadata_json=meta,   # serialTS safely ignores this
            only_song_present=False,
            compute_durations=False,       # counts only
            add_recording_datetime=True,   # ensure Date/Hour populated
        )
    elif _ORG_MODE == "durations":
        out = _build_organized(
            decoded_database_json=decoded,
            creation_metadata_json=meta,
            only_song_present=False,
            compute_durations=False,
        )
    else:  # legacy
        out = _build_organized(decoded, meta, verbose=verbose)

    organized_df = out.organized_df

    # 2) Build daily avg count table (mean counts per file per day)
    count_table = build_daily_avg_count_table(
        organized_df,
        label_column=label_column,
        date_column=date_column,
        syllable_labels=getattr(out, "unique_syllable_labels", None),
    )

    # 3) Animal ID (if consistent) or override
    animal_id = animal_id_override
    if animal_id is None and "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]

    # 4) Treatment date comes ONLY from the argument
    treatment_date = _coerce_timestamp_or_str(treatment_date)

    # 5) Determine save_path from output_dir + auto filename
    save_path: Optional[Path] = None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Prefer the resolved animal_id; otherwise fall back to the decoded file's stem
        base = animal_id if (animal_id and animal_id.strip()) else decoded.stem
        # Sanitize base a bit (spaces -> underscores)
        base = base.replace(" ", "_")
        save_path = outdir / f"{base}_syllable_heatmap.png"

    # 6) Plot heatmap
    fig, ax = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        save_path=(str(save_path) if save_path else None),
        show=show,
        nearest_match=nearest_match,
        max_days_off=max_days_off,
        sort_dates=sort_dates,
        pseudocount=pseudocount,
        cmap=cmap,
        date_format=date_format,
    )

    return {
        "organized_df": organized_df,
        "count_table": count_table,
        "fig": fig,
        "ax": ax,
        "animal_id": animal_id,
        "treatment_date": treatment_date,
        "save_path": save_path,
    }


# Optional CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize decoded dataset and plot log10-scaled syllable-count heatmap."
    )
    parser.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    parser.add_argument("--meta", type=str, default="", help="Optional path to *_creation_data.json (required for some organizers)")
    parser.add_argument("--outdir", type=str, default="", help="Folder to save the figure; filename auto-named by animal_id")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--exact-line", action="store_true", help="Require exact date for treatment line (no nearest match)")
    parser.add_argument("--max-days-off", type=int, default=1, help="Tolerance (days) for nearest-match line")
    parser.add_argument("--treatment-date", type=str, default="", help='Treatment date, e.g. "2025-03-04" or "2025.03.04"')
    args = parser.parse_args()

    td = args.treatment_date if args.treatment_date else None
    outdir = args.outdir or None

    _ = run_daily_syllable_heatmap(
        decoded_database_json=args.decoded_database_json,
        creation_metadata_json=(args.meta or None),
        output_dir=outdir,
        show=not args.no_show,
        nearest_match=not args.exact_line,
        max_days_off=args.max_days_off,
        treatment_date=td,
    )

"""
from syllable_heatmap_wrapper import run_daily_syllable_heatmap

decoded = "/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis/USA5507_RC5/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5507_RC5_Comp2_decoded_database.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis/USA5507_RC5/figures"  # folder only

res = run_daily_syllable_heatmap(
    decoded_database_json=decoded,
    creation_metadata_json=None,       # not needed with serialTS organizer
    output_dir=outdir,                 # auto-saves as <animal_id>_syllable_heatmap.png
    treatment_date="2025-02-04",       # optional
    nearest_match=True,
    max_days_off=1,
    show=True,
)

print("Saved to:", res["save_path"])  # e.g., /.../figures/USA5323_syllable_heatmap.png


"""