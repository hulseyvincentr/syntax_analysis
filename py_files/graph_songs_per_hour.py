# graph_songs_per_hour.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Use the Excel-serial-based organizer
from organized_decoded_serialTS_segments import build_organized_segments_with_durations


def _best_datetime_series(df: pd.DataFrame) -> pd.Series:
    """
    Prefer 'Recording DateTime'; if NaT, fall back to per-row 'creation_date' (ISO) if present.
    (The serial-based builder already supplies 'Recording DateTime' from the Excel serial.)
    """
    dt = df.get("Recording DateTime", pd.Series([pd.NaT] * len(df), index=df.index)).copy()
    if "creation_date" in df.columns:
        need = dt.isna()
        if need.any():
            cd = pd.to_datetime(df.loc[need, "creation_date"], errors="coerce")
            dt.loc[need] = cd
    return dt


def _collect_offhours_songs(
    df: pd.DataFrame,
    animal_id: str | None,
    start_hour: int = 6,
    end_hour: int = 18,
):
    """
    Build a list of off-hours rows (hour < start_hour or hour >= end_hour).
    If df already has a '__dt__' column, use it directly; otherwise compute a best-effort dt.
    """
    # 1) Use provided __dt__ if present, else compute it
    if "__dt__" in df.columns:
        dt = pd.to_datetime(df["__dt__"], errors="coerce")
    else:
        dt = _best_datetime_series(df)

    valid = df.copy()
    valid["__dt__"] = dt
    valid = valid.dropna(subset=["__dt__"])

    # Compute hour from the trusted datetime column
    valid["__hour__"] = valid["__dt__"].dt.hour

    # 2) Off-hours mask
    mask = (valid["__hour__"] < start_hour) | (valid["__hour__"] >= end_hour)
    off = valid.loc[mask].copy()

    # 3) Build JSON-able rows
    out_rows = []
    for _, row in off.iterrows():
        rdt = row["__dt__"]
        hour_val = None if pd.isna(rdt) else int(pd.to_datetime(rdt).hour)
        out_rows.append({
            "animal_id": animal_id or row.get("Animal ID"),
            "file_name": row.get("file_name"),
            "segment": None if pd.isna(row.get("Segment")) else int(row.get("Segment")),
            "date_str": None if pd.isna(rdt) else pd.to_datetime(rdt).strftime("%Y.%m.%d"),
            "hour": hour_val,
            "recording_datetime_iso": None if pd.isna(rdt) else pd.to_datetime(rdt).isoformat(),
            "file_stem": row.get("File Stem"),
            "syllables_present": row.get("syllables_present", []),
        })
    return out_rows


def songs_per_hour(
    decoded_database_json: str | Path,
    *,
    only_song_present: bool = True,
    show: bool = True,
    export_offhours_json_path: str | Path | None = None,
    offhours_start_hour: int = 6,
    offhours_end_hour: int = 18,
):
    """
    Plot number of songs per hour (one line per day). Only requires the decoded JSON.
    Optionally export off-hours segments to JSON.

    Parameters
    ----------
    decoded_database_json : str | Path
        Path to the decoded database JSON.
    only_song_present : bool
        Keep only rows where 'song_present' is True.
    show : bool
        Show the plot.
    export_offhours_json_path : str | Path | None
        If provided, writes the off-hours segments JSON to this path.
    offhours_start_hour, offhours_end_hour : int
        Define daytime window [start_hour, end_hour); outside counts as off-hours.

    Returns
    -------
    grouped : pd.DataFrame
        Long-form counts of songs per day and hour (Date, Hour, song_count).
    pivoted : pd.DataFrame
        Wide format with Hour as index and each column = a day (0–23 rows).
    offhours_list : list[dict]
        Off-hours segments (also written to JSON if export_offhours_json_path is provided).
    organized_df : pd.DataFrame
        The full organized dataframe from the builder (for debugging/inspection).
    """
    out = build_organized_segments_with_durations(
        decoded_database_json=decoded_database_json,
        only_song_present=only_song_present,
        compute_durations=True,
    )
    organized_df = out.organized_df  # return this for inspection

    # Robust datetime: prefer Recording DateTime; fall back to per-row creation_date (if any)
    dt = _best_datetime_series(organized_df)
    valid = organized_df.copy()
    valid["__dt__"] = dt
    valid = valid.dropna(subset=["__dt__"])

    if valid.empty:
        print("No rows with a valid datetime. "
              "Check that 'Recording DateTime' exists per row.")
        return (
            pd.DataFrame(columns=["Date", "Hour", "song_count"]),
            pd.DataFrame(),  # pivoted
            [],              # offhours
            organized_df,    # still return for debugging
        )

    valid["Date"] = valid["__dt__"].dt.date
    valid["Hour"] = valid["__dt__"].dt.hour

    # Animal ID inference
    animal_id = None
    if "Animal ID" in valid.columns:
        ids = valid["Animal ID"].dropna().unique()
        if len(ids) >= 1:
            animal_id = str(ids[0])

    # Group → counts
    grouped = (
        valid.groupby(["Date", "Hour"])
        .size()
        .reset_index(name="song_count")
    )

    # Ensure all hours 0–23 appear
    all_hours = pd.Index(range(24), name="Hour")
    pivoted = grouped.pivot(index="Hour", columns="Date", values="song_count")
    pivoted = pivoted.reindex(all_hours).fillna(0)

    # Plot
    if show:
        plt.figure(figsize=(12, 6))
        plotted_any = False
        for day in pivoted.columns:
            y = pivoted[day]
            if y.notna().any():
                plt.plot(pivoted.index, y, marker="o", label=str(day))
                plotted_any = True
        plt.xticks(range(0, 24))
        plt.xlim(0, 23)
        plt.ylim(0, 150)
        plt.xlabel("Hour of Day (0–23)")
        plt.ylabel("Number of Songs")
        title = "Songs per Hour (one line per day)"
        if animal_id:
            title += f" — {animal_id}"
        plt.title(title)
        if plotted_any:
            plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # Off-hours (uses the same __dt__ that powered the plot)
    offhours_list = _collect_offhours_songs(
        df=valid,
        animal_id=animal_id,
        start_hour=offhours_start_hour,
        end_hour=offhours_end_hour,
    )

    # Optional export
    if export_offhours_json_path is not None:
        export_path = Path(export_offhours_json_path).parent / f"{animal_id}_offhours_segments.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "animal_id": animal_id,
            "daytime_window": {
                "start_hour_inclusive": offhours_start_hour,
                "end_hour_exclusive": offhours_end_hour
            },
            "n_offhours_segments": len(offhours_list),
            "offhours_segments": offhours_list
        }
        with export_path.open("w") as f:
            json.dump(payload, f, indent=2)

    return grouped, pivoted, offhours_list, organized_df


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
"""
from pathlib import Path
from graph_songs_per_hour import songs_per_hour

decoded = Path("/Users/mirandahulsey-vincent/Desktop/AFP_lesion_QC_detection/USA5510/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5510_RC3_Comp2_decoded_database.json")

# Output dir = same folder as decoded JSON
outdir = decoded.parent

# Call the function (note: 4 return values)
long_counts, wide_counts, offhours, organized_df = songs_per_hour(
    decoded_database_json=decoded,
    only_song_present=True,
    show=True,
    export_offhours_json_path=outdir / "offhours_segments.json",
    offhours_start_hour=4,
    offhours_end_hour=19,
)

print(f"Off-hours segments: {len(offhours)}")
print(offhours[:3])  # peek first few

# Optional: quick peek to debug datetime fields
cols = ["file_name","Date","Hour","Minute","Second","Recording DateTime"]
print(organized_df[[c for c in cols if c in organized_df.columns]].head())
"""
