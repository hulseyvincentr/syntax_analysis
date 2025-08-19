# syllable_heatmap_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import json
import pandas as pd

from organize_decoded_dataset import build_organized_dataset
from syllable_heatmap import (
    build_daily_avg_count_table,
    plot_log_scaled_syllable_counts,
)


def run_daily_syllable_heatmap(
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    *,
    # plotting options (passed through to plot_log_scaled_syllable_counts)
    save_path: Optional[Union[str, Path]] = None,
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
    # overrides (optional)
    animal_id_override: Optional[str] = None,
    treatment_date_override: Optional[Union[str, pd.Timestamp]] = None,
    # misc
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end wrapper: organize decoded dataset, compute per-day avg syllable counts,
    and plot a log10-scaled heatmap.

    Returns a dict with:
      - organized_df: pd.DataFrame
      - count_table:  pd.DataFrame (rows=labels, cols=dates)
      - fig, ax:      Matplotlib objects
      - animal_id:    str | None
      - treatment_date: str | pd.Timestamp | None
    """
    decoded = Path(decoded_database_json)
    meta = Path(creation_metadata_json)
    if not decoded.exists():
        raise FileNotFoundError(f"Decoded database JSON not found: {decoded}")
    if not meta.exists():
        raise FileNotFoundError(f"Creation metadata JSON not found: {meta}")

    # 1) Organize dataset
    out = build_organized_dataset(decoded, meta, verbose=verbose)
    organized_df = out.organized_df

    # 2) Build daily avg count table (mean counts per file per day)
    count_table = build_daily_avg_count_table(
        organized_df,
        label_column=label_column,
        date_column=date_column,
        syllable_labels=out.unique_syllable_labels,  # stable order
    )

    # 3) Animal ID (if consistent) or override
    animal_id = animal_id_override
    if animal_id is None and "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            animal_id = ids[0]

    # 4) Treatment date from metadata or override
    treatment_date = treatment_date_override
    if treatment_date is None:
        try:
            with meta.open("r") as f:
                meta_json = json.load(f)
            treatment_date = meta_json.get("treatment_date", None)
        except Exception:
            treatment_date = None

    # 5) Plot heatmap
    fig, ax = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        save_path=save_path,
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
    }


# Optional CLI so you can also run this file directly if you want.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize decoded dataset and plot log10-scaled syllable-count heatmap."
    )
    parser.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    parser.add_argument("creation_metadata_json", type=str, help="Path to *_creation_data.json")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the figure")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--exact-line", action="store_true", help="Require exact date for treatment line (no nearest match)")
    parser.add_argument("--max-days-off", type=int, default=1, help="Tolerance (days) for nearest-match line")
    args = parser.parse_args()

    _ = run_daily_syllable_heatmap(
        args.decoded_database_json,
        args.creation_metadata_json,
        save_path=(args.save or None),
        show=not args.no_show,
        nearest_match=not args.exact_line,
        max_days_off=args.max_days_off,
    )

"""
from syllable_heatmap_wrapper import run_daily_syllable_heatmap

from pathlib import Path

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
created = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"

# Pick a filename (not just a folder)
out_dir = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data"
save_path = str(Path(out_dir) / "USA5288_syllable_heatmap.png")

res = run_daily_syllable_heatmap(
    decoded_database_json=decoded,
    creation_metadata_json=created,
    save_path=save_path,       # keyword arg + full file path
    show=True,
    nearest_match=True,
    max_days_off=1,
)

"""