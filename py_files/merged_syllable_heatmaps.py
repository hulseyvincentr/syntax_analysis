# merged_syllable_heatmaps.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd

from merge_annotations_from_split_songs import build_decoded_with_split_labels
import syllable_heatmap_linear as shlin  # library above

# ---------- helpers ----------

def _collect_unique_labels(df: pd.DataFrame, dict_col: str = "syllable_onsets_offsets_ms_dict") -> List[str]:
    labs: List[str] = []
    for d in df.get(dict_col, pd.Series([], dtype=object)).dropna():
        if isinstance(d, dict):
            labs.extend(list(d.keys()))
    # keep discovery order
    return list(dict.fromkeys(str(x) for x in labs))

def _sorted_labels_numeric_first(labels: List[str | int]) -> List[str]:
    def _key(x):
        s = str(x)
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [str(x) for x in sorted(labels, key=_key)]

# ---------- main ----------

def run_merged_syllable_heatmaps(
    *,
    song_detection_json: Union[str, Path],
    decoded_annotations_json: Union[str, Path],
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    # merge knobs
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    # plotting knobs
    cmap: str = "Greys",
    show: bool = True,
) -> Dict[str, Any]:
    """
    Merge split songs + coalesce repeats, build daily avg table (ALL labels, numeric order),
    then render two heatmaps: vmax=1.0 and vmax=max observed.
    """
    song_detection_json = Path(song_detection_json)
    decoded_annotations_json = Path(decoded_annotations_json)

    if output_dir is None:
        output_dir = decoded_annotations_json.parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Merge once (songs + repeats)
    res = build_decoded_with_split_labels(
        decoded_database_json=decoded_annotations_json,
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
    merged_df = res.annotations_appended_df.copy()
    if merged_df.empty:
        return {"merged_df": merged_df, "count_table": pd.DataFrame(),
                "save_path_1": None, "save_path_2": None, "vmax_max": 1.0,
                "note": "No data after merge."}

    # Ensure calendar Date column
    if "Date" not in merged_df.columns:
        dt_col = "Recording DateTime" if "Recording DateTime" in merged_df.columns else "recording_datetime"
        if dt_col not in merged_df.columns:
            raise ValueError("No datetime column found (expected 'Recording DateTime' or 'recording_datetime').")
        merged_df["Date"] = pd.to_datetime(merged_df[dt_col]).dt.date

    # 2) Build daily avg table with ALL labels, sorted numerically
    labs = _collect_unique_labels(merged_df, "syllable_onsets_offsets_ms_dict")
    syllable_labels = _sorted_labels_numeric_first(labs)
    count_table = shlin.build_daily_avg_count_table(
        merged_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=syllable_labels,
    )

    # Filenames/ID
    animal_id = shlin._infer_animal_id(merged_df, decoded_annotations_json) or decoded_annotations_json.stem or "unknown_animal"
    save_path_1 = output_dir / f"{animal_id}_syllable_heatmap_linear_vmax1.00.png"
    save_path_2 = output_dir / f"{animal_id}_syllable_heatmap_linear_vmaxMAX.png"

    # 3) vmax for Plot 2
    raw_max = float(np.nanmax(count_table.to_numpy())) if count_table.size else 1.0
    vmax_max = raw_max if np.isfinite(raw_max) and raw_max > 0 else 1.0

    # 4) Plots (labels on Y, dates on X)
    shlin.plot_linear_scaled_syllable_counts(
        count_table, animal_id=animal_id, treatment_date=treatment_date,
        save_path=save_path_1, show=show, cmap=cmap, vmin=0.0, vmax=1.0,
        nearest_match=True, max_days_off=1,
    )
    shlin.plot_linear_scaled_syllable_counts(
        count_table, animal_id=animal_id, treatment_date=treatment_date,
        save_path=save_path_2, show=show, cmap=cmap, vmin=0.0, vmax=vmax_max,
        nearest_match=True, max_days_off=1,
    )

    return {
        "merged_df": merged_df,
        "count_table": count_table,
        "save_path_1": str(save_path_1),
        "save_path_2": str(save_path_2),
        "vmax_max": vmax_max,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Merge songs+phrases then plot syllable-usage heatmaps.")
    p.add_argument("--detect", required=True, type=str)
    p.add_argument("--decoded", required=True, type=str)
    p.add_argument("--tdate", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--gap", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")
    p.add_argument("--cmap", type=str, default="Greys")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    out = run_merged_syllable_heatmaps(
        song_detection_json=args.detect,
        decoded_annotations_json=args.decoded,
        treatment_date=args.tdate,
        output_dir=args.outdir,
        max_gap_between_song_segments=args.gap,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        cmap=args.cmap,
        show=not args.no_show,
    )
    print("Saved 1:", out["save_path_1"])
    print("Saved 2:", out["save_path_2"])
    print("vmax for Plot 2:", out["vmax_max"])


"""
from pathlib import Path
import sys, importlib

# Ensure the folder is on sys.path
sys.path.append("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")

# Reload the clean library and verify it exports the needed functions
import syllable_heatmap_linear as shlin
importlib.reload(shlin)
print("Loaded from:", shlin.__file__)
print("Exports present:",
      hasattr(shlin, "build_daily_avg_count_table"),
      hasattr(shlin, "plot_linear_scaled_syllable_counts"),
      hasattr(shlin, "_infer_animal_id"))

# Runner
import merged_syllable_heatmaps as msh
importlib.reload(msh)

detect  = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
decoded = Path("/Volumes/my_own_ssd/2025_areax_lesion/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R08_RC6_Comp2_decoded_database.json")
tdate   = "2025-05-22"   # or None

res = msh.run_merged_syllable_heatmaps(
    song_detection_json=detect,
    decoded_annotations_json=decoded,
    treatment_date=tdate,
    output_dir=decoded.parent / "figures",
    max_gap_between_song_segments=500,
    segment_index_offset=0,
    merge_repeated_syllables=True,
    repeat_gap_ms=10.0,
    repeat_gap_inclusive=False,
    cmap="Greys",
    show=True,
)

print("Saved:", res["save_path_1"])
print("Saved:", res["save_path_2"])
print("Max avg count per song (vmax for Plot 2):", res["vmax_max"])

"""