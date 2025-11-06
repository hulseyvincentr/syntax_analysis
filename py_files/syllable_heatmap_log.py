# syllable_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_log_scaled_syllable_counts", "build_daily_avg_count_table"]

# ───────────────────────────────────────────────────────────────────────────────
# Organizer import preference order (Excel-serial builder first)
# ───────────────────────────────────────────────────────────────────────────────
_ORG_MODE = "serialTS"

try:
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _build_organized,
    )
    _ORG_MODE = "serialTS"
except ImportError:
    try:
        from organize_decoded_serialTS_segments import (
            build_organized_segments_with_durations as _build_organized,
        )
        _ORG_MODE = "serialTS"
    except ImportError:
        try:
            from organize_decoded_with_segments import (
                build_organized_segments_with_durations as _build_organized,  # type: ignore
            )
            _ORG_MODE = "segments"
        except ImportError:
            try:
                from organize_decoded_dataset import (
                    build_organized_dataset as _build_organized,  # type: ignore
                )
                _ORG_MODE = "legacy"
            except ImportError:
                from organize_decoded_with_durations import (  # type: ignore
                    build_organized_dataset_with_durations as _build_organized,
                )
                _ORG_MODE = "durations"


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _numeric_aware_key(x):
    s = str(x)
    try:
        return (0, int(s), s)
    except Exception:
        return (1, s.lower(), s)


def _infer_animal_id(organized_df: pd.DataFrame, decoded_path: Union[str, Path] | None = None) -> Optional[str]:
    """
    Best-effort animal_id inference:
      1) Single unique value in 'Animal ID' column (string) → return it
      2) Regex on decoded filename: r'(USA\\d+)'  → return first match
      3) Token before '_decoded_database' in filename → return it
      4) Fallback: None
    """
    # 1) From dataframe
    if "Animal ID" in organized_df.columns:
        ids = [x for x in organized_df["Animal ID"].dropna().unique().tolist() if isinstance(x, str)]
        if len(ids) == 1:
            return ids[0]

    # 2/3) From decoded path
    if decoded_path is not None:
        name = Path(decoded_path).name
        # Prefer explicit USA#### pattern
        m = re.search(r"(USA\d+)", name, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        # Token immediately before '_decoded_database'
        m2 = re.search(r"([^_/]+)_decoded_database", name)
        if m2:
            return m2.group(1)

    return None


# ───────────────────────────────────────────────────────────────────────────────
# Heatmap (log10-scaled) of daily average syllable counts per song
# ───────────────────────────────────────────────────────────────────────────────
def plot_log_scaled_syllable_counts(
    count_table: pd.DataFrame,
    animal_id: Optional[str] = None,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    *,
    pseudocount: float = 1e-3,
    figsize: Tuple[int, int] = (16, 6),
    cmap: str = "Greys",
    cbar_label: str = r"$\log_{10}$ Avg Count per Song",
    sort_dates: bool = True,
    date_format: str = "%Y-%m-%d",
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    mark_treatment: bool = True,
    nearest_match: bool = True,
    max_days_off: int = 1,
    line_position: str = "center",
    line_kwargs: Optional[dict] = None,
    sort_rows_numerically: bool = True,
):
    if count_table is None or count_table.empty:
        raise ValueError("count_table is empty or None.")

    if sort_rows_numerically:
        new_row_order = sorted([str(x) for x in count_table.index], key=_numeric_aware_key)
        count_table = count_table.reindex(index=new_row_order)

    columns = count_table.columns
    if sort_dates:
        try:
            columns = sorted(columns, key=pd.to_datetime)
            count_table = count_table.reindex(columns=columns)
        except Exception:
            pass

    log_scaled_table = np.log10(count_table + float(pseudocount))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        log_scaled_table,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    date_idx = pd.to_datetime(log_scaled_table.columns)
    xticks = np.arange(len(date_idx)) + 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime(date_format) for d in date_idx], rotation=90, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Syllable Label")

    title_id = animal_id or "unknown_animal"
    ax.set_title(f"{title_id} Daily Avg Occurrence of Each Syllable ($\\log_{{10}}$ Scale)")

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
    sort_labels_numerically: bool = True,
) -> pd.DataFrame:
    if label_column not in organized_df.columns or date_column not in organized_df.columns:
        raise KeyError(f"organized_df must contain '{label_column}' and '{date_column}' columns.")

    df = organized_df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_column])

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

    unique_days = sorted(df[date_column].dropna().unique())
    count_table = pd.DataFrame(0.0, index=syllable_labels, columns=unique_days)

    for day, g in df.groupby(date_column):
        per_file = []
        for v in g[label_column]:
            if isinstance(v, dict) and v:
                counts = {lab: len(v.get(lab, [])) for lab in syllable_labels}
            else:
                counts = {lab: 0 for lab in syllable_labels}
            per_file.append(pd.Series(counts, dtype=float))

        if per_file:
            mean_counts = pd.concat(per_file, axis=1).fillna(0).mean(axis=1)
            count_table.loc[:, day] = mean_counts

    return count_table


# ───────────────────────────────────────────────────────────────────────────────
# Example CLI (uses Excel-serial organizer; accepts --output-dir and auto filename)
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Build daily avg syllable-count heatmap (log10) from decoded dataset (Excel-serial timestamps)."
    )
    p.add_argument("decoded_database_json", type=str, help="Path to *_decoded_database.json")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to save the figure. Default: the decoded JSON's parent directory.")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    p.add_argument("--nearest-line", action="store_true", help="Mark nearest date to treatment (within 1 day)")
    p.add_argument("--treatment-date", type=str, default=None, help="Optional treatment date (e.g., 2025-03-04)")
    p.add_argument("--only-song-present", action="store_true", help="Filter to rows with song_present == True")
    p.add_argument("--pseudocount", type=float, default=1e-3, help="Pseudocount added before log10")
    args = p.parse_args()

    decoded_path = Path(args.decoded_database_json)

    # Build organized outputs using the preferred Excel-serial builder
    if _ORG_MODE in {"serialTS", "segments"}:
        out = _build_organized(
            decoded_database_json=args.decoded_database_json,
            creation_metadata_json=None,
            only_song_present=args.only_song_present,
            compute_durations=False,
            add_recording_datetime=True,
        )
    elif _ORG_MODE == "durations":
        out = _build_organized(
            decoded_database_json=args.decoded_database_json,
            creation_metadata_json=None,
            only_song_present=args.only_song_present,
            compute_durations=False,
        )
    else:
        out = _build_organized(args.decoded_database_json, None)

    organized_df = out.organized_df

    count_table = build_daily_avg_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=getattr(out, "unique_syllable_labels", None),
        sort_labels_numerically=True,
    )

    # Infer animal_id and construct auto filename
    animal_id = _infer_animal_id(organized_df, decoded_path) or "unknown_animal"

    output_dir = Path(args.output_dir) if args.output_dir else decoded_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{animal_id}_syllable_heatmap.png"

    _ = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=args.treatment_date,
        show=not args.no_show,
        save_path=save_path,
        sort_dates=True,
        nearest_match=args.nearest_line,
        max_days_off=1,
        cmap="Greys",
        sort_rows_numerically=True,
        pseudocount=args.pseudocount,
    )



"""
from organized_decoded_serialTS_segments import build_organized_segments_with_durations
from syllable_heatmap_log import build_daily_avg_count_table, plot_log_scaled_syllable_counts, _infer_animal_id
from pathlib import Path

decoded = Path("/Users/mirandahulsey-vincent/Desktop/USA1234_VALIDATION_Data/USA1234_decoded_database.json")

# where to save the figure
outdir = decoded.parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    only_song_present=True,
    compute_durations=False,
    add_recording_datetime=True,
)

count_table = build_daily_avg_count_table(
    out.organized_df,
    label_column="syllable_onsets_offsets_ms_dict",
    date_column="Date",
    syllable_labels=out.unique_syllable_labels,
)

animal_id = _infer_animal_id(out.organized_df, decoded) or "unknown_animal"
save_path = outdir / f"{animal_id}_syllable_heatmap.png"

fig, ax = plot_log_scaled_syllable_counts(
    count_table,
    animal_id=animal_id,
    treatment_date="2025-05-22",  # optional
    save_path=save_path,
    show=True,
)

print("Saved to:", save_path)



"""
