#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPZ_colorcode_groups_UMAP_serialTS.py

UMAP overlap plotting for TweetyBERT NPZ outputs.

This version:
- Uses ONLY the Excel-style serial timestamp embedded in each filename
  to infer the recording date, e.g.:

      'USA5443_45424.54633884_5_12_15_10_33_segment_0.npz'
                 ^^^^^^^^^^^^^
                 Excel-style serial (days since 1899-12-30)

- The JSON file is OPTIONAL and, if provided, is used ONLY for:
    * Animal ID (if present in the path)
    * Treatment date (for figure title)
    * Treatment type (for figure title)

Per-file dates NEVER come from the JSON.

Public API
----------
- get_min_max_recording_dates(npz_path)
    -> (min_date, max_date, unique_dates)

- run_umap_overlap_from_npz(
      npz_path,
      json_path=None,
      date_range_before=(start, end),
      date_range_after=(start, end),
      ...
  )

Typical Spyder usage
--------------------
from pathlib import Path
import importlib
import NPZ_colorcode_groups_UMAP_serialTS as umapmod

importlib.reload(umapmod)

npz_path = Path("/Volumes/my_own_SSD/USA5443_updated/USA5443.npz")

# 1) Get min/max recording dates from NPZ
min_date, max_date, unique_dates = umapmod.get_min_max_recording_dates(npz_path)
print("Earliest date:", min_date)
print("Latest date:  ", max_date)

# 2) Define treatment date
treatment_date_str = "2024-04-30"

# Before = all data strictly before treatment
date_range_before = (str(min_date), "2024-04-29")
# After  = all data on/after treatment
date_range_after  = ("2024-04-30", str(max_date))

# 3) Plot
umapmod.run_umap_overlap_from_npz(
    npz_path=npz_path,
    json_path=None,  # or a JSON if you want a nicer title
    date_range_before=date_range_before,
    date_range_after=date_range_after,
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Union, Tuple
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "get_min_max_recording_dates",
    "run_umap_overlap_from_npz",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def excel_serial_to_datetime(serial: float) -> np.datetime64:
    """
    Convert an Excel-style serial date (days since 1899-12-30)
    to numpy.datetime64.

    Parameters
    ----------
    serial : float
        Excel serial (including fractional day).

    Returns
    -------
    np.datetime64
        Corresponding datetime.
    """
    dt = pd.to_datetime(serial, unit="D", origin="1899-12-30")
    return np.datetime64(dt)


def _parse_animal_and_treatment_from_json(
    json_path: Optional[Union[str, Path]]
) -> Tuple[str, str, str]:
    """
    Extract animal_id, treatment_date, treatment_type from a JSON path.

    Parameters
    ----------
    json_path : str or Path or None
        Path to the creation_data JSON.

    Returns
    -------
    animal_id : str
    treatment_date : str
    treatment_type : str

    Notes
    -----
    If json_path is None or invalid, returns "Unknown ..." defaults.
    This function is ONLY for title metadata and does NOT affect
    per-file dates (which come from the filename serial).
    """
    animal_id = "Unknown ID"
    treatment_date = "Unknown treatment date"
    treatment_type = "Unknown treatment type"

    if json_path is None:
        return animal_id, treatment_date, treatment_type

    json_path = Path(json_path)
    if not json_path.is_file():
        return animal_id, treatment_date, treatment_type

    # Animal ID from path (e.g., "USA5443")
    match = re.search(r"(USA\d{4})", str(json_path))
    if match:
        animal_id = match.group(1)

    # Metadata from JSON
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        treatment_date = meta.get("treatment_date", treatment_date)
        treatment_type = meta.get("treatment_type", treatment_type)
    except Exception as exc:
        print(f"[WARN] Could not read JSON metadata from {json_path}: {exc}")

    return animal_id, treatment_date, treatment_type


# ---------------------------------------------------------------------
# Public helper: find earliest/latest recording dates from NPZ
# ---------------------------------------------------------------------
def get_min_max_recording_dates(npz_path: Union[str, Path]):
    """
    Inspect a TweetyBERT NPZ and return:
        - earliest recording date (numpy.datetime64[D])
        - latest recording date  (numpy.datetime64[D])
        - sorted unique dates    (numpy.ndarray of datetime64[D])

    Dates are derived from the Excel-style serial in the filename
    (file_map), e.g. 'USA5443_45424.54633884_5_12_15_10_33_segment_0.npz'.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    try:
        file_map_dictionary = data["file_map"].item()
        dates = []

        for file_index, filename_tuple in file_map_dictionary.items():
            filename = filename_tuple[0]
            parts = filename.split("_")
            if len(parts) < 2:
                continue

            serial_str = parts[1]
            try:
                serial = float(serial_str)
            except ValueError:
                continue

            dt64 = excel_serial_to_datetime(serial)
            date_only = dt64.astype("datetime64[D]")
            dates.append(date_only)

        if not dates:
            raise RuntimeError("No valid dates could be parsed from file_map filenames.")

        dates_arr = np.array(dates, dtype="datetime64[D]")
        min_date = dates_arr.min()
        max_date = dates_arr.max()
        unique_dates = np.sort(np.unique(dates_arr))

        return min_date, max_date, unique_dates

    finally:
        try:
            data.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Main public plotting function
# ---------------------------------------------------------------------
def run_umap_overlap_from_npz(
    npz_path: Union[str, Path],
    json_path: Optional[Union[str, Path]] = None,
    date_range_before: Optional[Sequence[str]] = None,
    date_range_after: Optional[Sequence[str]] = None,
    bins: int = 300,
    brightness_factor: float = 6.0,
) -> None:
    """
    Make before/after UMAP overlap plots from a TweetyBERT NPZ.

    Parameters
    ----------
    npz_path : str or Path
        Path to USA####.npz file (TweetyBERT output).
    json_path : str or Path or None, optional
        OPTIONAL path to creation_data JSON (for treatment metadata).
        If None or invalid, animal_id/treatment fields in the title
        will fall back to "Unknown".
    date_range_before : sequence of 2 str
        Start and end date (inclusive) for the "before" dataset,
        e.g. ("2025-02-12", "2025-02-21").
    date_range_after : sequence of 2 str
        Start and end date (inclusive) for the "after" dataset,
        e.g. ("2025-02-22", "2025-03-04").
    bins : int, default 300
        Number of bins for the 2D histograms.
    brightness_factor : float, default 6.0
        Scaling factor for RGB overlap intensity.

    Notes
    -----
    - Recording dates are derived ONLY from the Excel-style serial
      embedded in each filename in file_map:
          'USA5443_45424.54633884_5_12_15_10_33_segment_0.npz'
                   ^^^^^^^^^^^^^
                   serial (days since 1899-12-30)
    - JSON is NOT used for any per-file dates; it's only for title metadata.
    """
    npz_path = Path(npz_path)

    if date_range_before is None or date_range_after is None:
        raise ValueError("Please provide both date_range_before and date_range_after.")

    # Extract metadata for title (animal ID, treatment date/type)
    animal_id, treatment_date, treatment_type = _parse_animal_and_treatment_from_json(
        json_path
    )

    print(f"Using NPZ:   {npz_path}")
    if json_path is not None:
        print(f"Using JSON:  {json_path}")
    else:
        print("No JSON provided; title will use default 'Unknown' metadata.")
    print(f"Animal ID:   {animal_id}")
    print(f"Treatment:   {treatment_type} on {treatment_date}")
    print(f"Date ranges: BEFORE {date_range_before}, AFTER {date_range_after}")

    # -----------------------------------------------------------------
    # Load NPZ lazily (better for huge arrays)
    # -----------------------------------------------------------------
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    try:
        file_map_dictionary = data["file_map"].item()
        embedding_outputs = data["embedding_outputs"]  # (N, 2)
        file_indices = data["file_indices"]            # (N,)

        # -----------------------------------------------------------------
        # Build per-file DataFrame using Excel serial from filename
        # -----------------------------------------------------------------
        rows = []
        for file_index, filename_tuple in file_map_dictionary.items():
            filename = filename_tuple[0]
            parts = filename.split("_")
            if len(parts) < 2:
                print(f"[WARN] Unexpected filename for file_index {file_index}: {filename!r}")
                continue

            serial_str = parts[1]  # e.g. "45424.54633884"
            try:
                serial = float(serial_str)
            except ValueError:
                print(f"[WARN] Could not parse serial from {filename!r}")
                continue

            dt64 = excel_serial_to_datetime(serial)
            date_only = dt64.astype("datetime64[D]")
            rows.append((file_index, filename, serial, dt64, date_only))

        if not rows:
            raise RuntimeError("No valid filenames could be parsed for serial timestamps.")

        file_map_df = pd.DataFrame(
            rows, columns=["file_index", "Filename", "serial", "DateTime", "Date"]
        )

        print("\nExample rows from file_map_df (from filename serial):")
        print(file_map_df.head())

        # -----------------------------------------------------------------
        # Build one-row-per-UMAP-point DataFrame: file_index, X, Y, Date
        # -----------------------------------------------------------------
        coords_df = pd.DataFrame(
            {
                "file_index": file_indices,
                "X": embedding_outputs[:, 0],
                "Y": embedding_outputs[:, 1],
            }
        )

        final_df = coords_df.merge(
            file_map_df[["file_index", "DateTime", "Date"]],
            on="file_index",
            how="left",
        )

        # Ensure Date is proper datetime64[ns]
        final_df["Date"] = pd.to_datetime(
            final_df["Date"].values.astype("datetime64[D]")
        )

        print("\nFirst 10 rows of final_df (Date, X, Y):")
        print(final_df[["Date", "X", "Y"]].head(10))

        unique_dates = np.sort(final_df["Date"].dropna().unique())
        print("\nUnique dates in the dataset (sorted):")
        for d in unique_dates:
            print(d)

        # -----------------------------------------------------------------
        # Filter by user-defined date ranges
        # -----------------------------------------------------------------
        drb0, drb1 = pd.to_datetime(date_range_before[0]), pd.to_datetime(
            date_range_before[1]
        )
        dra0, dra1 = pd.to_datetime(date_range_after[0]), pd.to_datetime(
            date_range_after[1]
        )

        before_treatment = final_df[
            (final_df["Date"] >= drb0) & (final_df["Date"] <= drb1)
        ]
        after_treatment = final_df[
            (final_df["Date"] >= dra0) & (final_df["Date"] <= dra1)
        ]

        num_before = len(before_treatment)
        num_after = len(after_treatment)
        print(f"\nNumber of UMAP points BEFORE: {num_before}")
        print(f"Number of UMAP points AFTER:  {num_after}")

        # -----------------------------------------------------------------
        # Build heatmaps and overlap image
        # -----------------------------------------------------------------
        if num_before == 0 or num_after == 0:
            print("[WARN] One of the date ranges has zero points; plots may be empty.")

        heatmap_before, xedges, yedges = np.histogram2d(
            before_treatment["X"], before_treatment["Y"], bins=bins
        )
        heatmap_after, _, _ = np.histogram2d(
            after_treatment["X"], after_treatment["Y"], bins=[xedges, yedges]
        )

        # Normalize each heatmap, guarding against division by zero
        if heatmap_before.max() > 0:
            heatmap_before = heatmap_before / heatmap_before.max()
        if heatmap_after.max() > 0:
            heatmap_after = heatmap_after / heatmap_after.max()

        # RGB overlap image: purple (before) vs green (after)
        rgb_image = np.zeros((heatmap_before.shape[0], heatmap_before.shape[1], 3))
        rgb_image[..., 0] = np.clip(
            heatmap_before.T * brightness_factor, 0, 1
        )  # red (part of purple)
        rgb_image[..., 1] = np.clip(
            heatmap_after.T * brightness_factor, 0, 1
        )  # green
        rgb_image[..., 2] = np.clip(
            heatmap_before.T * brightness_factor, 0, 1
        )  # blue (purple)

        # -----------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------
        plt.figure(figsize=(18, 6))

        # Before
        plt.subplot(1, 3, 1)
        plt.imshow(
            heatmap_before.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="Purples",
            vmax=0.1,
        )
        plt.title(
            f"Before: {date_range_before[0]} to {date_range_before[1]}\nN={num_before}",
            fontsize=14,
        )
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

        # After
        plt.subplot(1, 3, 2)
        plt.imshow(
            heatmap_after.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="Greens",
            vmax=0.1,
        )
        plt.title(
            f"After: {date_range_after[0]} to {date_range_after[1]}\nN={num_after}",
            fontsize=14,
        )
        plt.xlabel("UMAP Dimension 1")

        # Overlap
        plt.subplot(1, 3, 3)
        plt.imshow(
            rgb_image,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
        )
        plt.title("Overlap", fontsize=14)
        plt.xlabel("UMAP Dimension 1")

        plt.suptitle(f"{animal_id} {treatment_type} on {treatment_date}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    finally:
        # Make sure we close the NPZ file handle
        try:
            data.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Optional: simple example when run as a script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # EDIT THIS PATH IF YOU WANT TO RUN THIS FILE DIRECTLY
    example_npz = Path("/path/to/USA5443.npz")  # <- update

    if example_npz.is_file():
        # 1) Get min/max recording dates
        min_date, max_date, unique_dates = get_min_max_recording_dates(example_npz)
        print("Earliest date:", min_date)
        print("Latest date:  ", max_date)

        # 2) Treatment date for this example
        treatment_date_str = "2024-04-30"

        # Before = all data strictly before treatment
        date_range_before = (str(min_date), "2024-04-29")
        # After  = all data on/after treatment
        date_range_after = ("2024-04-30", str(max_date))

        run_umap_overlap_from_npz(
            npz_path=example_npz,
            json_path=None,  # optional; only for title metadata
            date_range_before=date_range_before,
            date_range_after=date_range_after,
        )
    else:
        print(
            "Edit 'example_npz' in the __main__ block to run this file directly,\n"
            "or import get_min_max_recording_dates and run_umap_overlap_from_npz "
            "from Spyder instead."
        )


"""
from pathlib import Path
import importlib
import NPZ_colorcode_groups_UMAP_serialTS as umapmod  # use your actual filename

importlib.reload(umapmod)

# 1) Point to your NPZ
npz_path = Path("/Volumes/my_own_SSD/USA5443_updated/USA5443.npz")

# 2) Get earliest and latest dates from the NPZ
min_date, max_date, unique_dates = umapmod.get_min_max_recording_dates(npz_path)
print("Earliest date:", min_date)
print("Latest date:  ", max_date)

# 3) Treatment date
treatment_date_str = "2024-04-30"

# Before = all data strictly before treatment
date_range_before = (str(min_date), "2024-04-29")

# After = all data on/after treatment
date_range_after  = ("2024-04-30", str(max_date))

print("Using date_range_before:", date_range_before)
print("Using date_range_after: ", date_range_after)

# 4) Run the UMAP overlap plot
umapmod.run_umap_overlap_from_npz(
    npz_path=npz_path,
    json_path=None,  # or a real JSON path if you want a nicer title
    date_range_before=date_range_before,
    date_range_after=date_range_after,
)


"""