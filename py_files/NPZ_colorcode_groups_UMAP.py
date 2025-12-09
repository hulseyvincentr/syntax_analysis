#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPZ_colorcode_groups_UMAP.py

UMAP overlap plotting for TweetyBERT NPZ outputs.

Features
--------
- Uses ONLY the Excel-style serial timestamp embedded in each filename
  to infer the recording date, e.g.:

      'USA5443_45424.54633884_5_12_15_10_33_segment_0.npz'
                ^^^^^^^^^^^^^
                Excel serial (days since 1899-12-30)

- animal_id is inferred from the NPZ filename or parent folder
  (e.g., "USA5443", "ZEBRA01", "CanaryA", etc.).

- Treatment date & treatment type are looked up automatically from a
  metadata Excel sheet (e.g., Area_X_lesion_metadata.xlsx) with columns:
      "Animal ID", "Treatment date", "Treatment type"

- You can:
    * run on a SINGLE .npz file, OR
    * point to a ROOT DIRECTORY that contains subfolders per bird:
          root/
            USA5288/USA5288.npz
            USA5337/USA5337.npz
            ...

Public API
----------
- get_min_max_recording_dates(npz_path)
    -> (min_date, max_date, unique_dates)

- run_umap_overlap_from_npz(
      npz_path,
      date_range_before=(start, end),
      date_range_after=(start, end),
      treatment_date=None,
      treatment_type=None,
      ...
  )

- load_treatment_metadata_from_excel(
      metadata_excel_path,
      sheet_name=0,
      id_col="Animal ID",
      treatment_date_col="Treatment date",
      treatment_type_col="Treatment type",
  )

- run_umap_for_path(
      npz_or_root,
      metadata_excel,
      ...
  )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Union, Dict, Tuple
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "excel_serial_to_datetime",
    "is_valid_npz",
    "get_min_max_recording_dates",
    "run_umap_overlap_from_npz",
    "load_treatment_metadata_from_excel",
    "run_umap_for_path",
]

# ---------------------------------------------------------------------
# Basic helpers
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


def is_valid_npz(path: Union[str, Path]) -> bool:
    """
    Return True if `path` looks like a valid .npz (zip) file.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.namelist()) > 0
    except Exception:
        return False


def _infer_animal_id_from_path(
    npz_path: Path,
    metadata_keys: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Infer an animal_id from an NPZ path.

    Strategy
    --------
    1. Take the file stem (without .npz), e.g.
           'BIRD123_45424.5463_segment_0' -> 'BIRD123'
       i.e. everything before the first underscore.
    2. Also consider the full stem and the parent folder name as candidates.
    3. If metadata_keys are provided, return the first candidate that
       appears in metadata_keys.
    4. Otherwise, return the first non-empty candidate.

    This allows IDs like 'USA5443', 'ZEBRA01', 'CanaryA', etc.,
    as long as they match the 'Animal ID' entries in the metadata Excel.
    """
    candidates: list[str] = []

    # File stem: e.g. 'BIRD123_45424.5463_segment_0' or 'BIRD123'
    stem = npz_path.stem
    if stem:
        parts = stem.split("_")
        # First token is typically the bird ID
        candidates.append(parts[0].strip())
        # Also consider the full stem
        candidates.append(stem.strip())

    # Parent folder name (e.g. 'BIRD123')
    parent_name = npz_path.parent.name.strip()
    if parent_name:
        candidates.append(parent_name)
        parent_parts = parent_name.split("_")
        if parent_parts:
            candidates.append(parent_parts[0].strip())

    # De-duplicate while preserving order
    seen = set()
    unique_candidates: list[str] = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        unique_candidates.append(c)

    # If we know the valid IDs (metadata keys), prefer one of those
    if metadata_keys is not None:
        key_set = set(str(k) for k in metadata_keys)
        for c in unique_candidates:
            if c in key_set:
                return c

    # Fallback: just return the first candidate if any
    return unique_candidates[0] if unique_candidates else None


# ---------------------------------------------------------------------
# Public helper: find earliest/latest recording dates from NPZ
# ---------------------------------------------------------------------
def get_min_max_recording_dates(npz_path: Union[str, Path]) -> Tuple[np.datetime64, np.datetime64, np.ndarray]:
    """
    Inspect a TweetyBERT NPZ and return:
        - earliest recording date (numpy.datetime64[D])
        - latest recording date  (numpy.datetime64[D])
        - sorted unique dates    (numpy.ndarray of datetime64[D])

    Dates are derived from the Excel-style serial in the filename
    (file_map), e.g. 'USA5443_45424.54633884_5_12_15_10_33_segment_0.npz'.
    """
    npz_path = Path(npz_path)

    if not is_valid_npz(npz_path):
        raise ValueError(
            f"{npz_path} is not a valid .npz (zip) file. "
            "It may be corrupted or incomplete."
        )

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
# Main plotting function for a single NPZ
# ---------------------------------------------------------------------
def run_umap_overlap_from_npz(
    npz_path: Union[str, Path],
    date_range_before: Sequence[str],
    date_range_after: Sequence[str],
    treatment_date: Optional[str] = None,
    treatment_type: Optional[str] = None,
    bins: int = 300,
    brightness_factor: float = 6.0,
) -> None:
    """
    Make before/after UMAP overlap plots from a TweetyBERT NPZ.

    Parameters
    ----------
    npz_path : str or Path
        Path to USA####.npz file (TweetyBERT output). The filename
        is used to infer animal_id (e.g. "USA5443", "ZEBRA01").
    date_range_before : sequence of 2 str
        Start and end date (inclusive) for the "before" dataset,
        e.g. ("2024-04-01", "2024-04-29").
    date_range_after : sequence of 2 str
        Start and end date (inclusive) for the "after" dataset,
        e.g. ("2024-04-30", "2024-05-10").
    treatment_date : str, optional
        String to display in the title, e.g. "2024-04-30".
        Does NOT affect the date filtering logic.
    treatment_type : str, optional
        String to display in the title, e.g. "bilateral NMA lesion".
        Does NOT affect the date filtering logic.
    bins : int, default 300
        Number of bins for the 2D histograms.
    brightness_factor : float, default 6.0
        Scaling factor for RGB overlap intensity.
    """
    npz_path = Path(npz_path)

    if not is_valid_npz(npz_path):
        raise ValueError(
            f"{npz_path} is not a valid .npz (zip) file. "
            "It may be corrupted or incomplete."
        )

    if date_range_before is None or date_range_after is None:
        raise ValueError("Please provide both date_range_before and date_range_after.")

    # Infer animal_id from NPZ path (generic, not just USA####)
    animal_id = _infer_animal_id_from_path(npz_path) or "Unknown ID"

    title_treatment_date = treatment_date or "Unknown treatment date"
    title_treatment_type = treatment_type or "Unknown treatment type"

    print("=" * 80)
    print(f"NPZ:        {npz_path}")
    print(f"Animal ID:  {animal_id}")
    print(f"Treatment:  {title_treatment_type} on {title_treatment_date}")
    print(f"Date ranges BEFORE: {date_range_before}, AFTER: {date_range_after}")

    # -----------------------------------------------------------------
    # Load NPZ lazily
    # -----------------------------------------------------------------
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    try:
        file_map_dictionary = data["file_map"].item()
        embedding_outputs = data["embedding_outputs"]  # shape (N, 2)
        file_indices = data["file_indices"]            # shape (N,)

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

        plt.suptitle(
            f"{animal_id} {title_treatment_type} on {title_treatment_date}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    finally:
        try:
            data.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Metadata loading from Excel
# ---------------------------------------------------------------------
def _normalize_date_to_str(value) -> Optional[str]:
    """Convert Excel / string date to 'YYYY-MM-DD' or None."""
    if pd.isna(value):
        return None
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def load_treatment_metadata_from_excel(
    metadata_excel_path: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    treatment_type_col: str = "Treatment type",
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Load treatment date and treatment type per animal from an Excel sheet.

    Parameters
    ----------
    metadata_excel_path : str or Path
        Path to Area_X_lesion_metadata.xlsx (or similar).
    sheet_name : int or str, default 0
        Sheet index or name.
    id_col : str, default "Animal ID"
        Column with bird IDs, e.g. "USA5288".
    treatment_date_col : str, default "Treatment date"
        Column with the surgery/treatment date.
    treatment_type_col : str, default "Treatment type"
        Column with description, e.g. "Bilateral NMA lesion injections".

    Returns
    -------
    metadata : dict
        Mapping from animal_id -> {"treatment_date": ..., "treatment_type": ...}
        where treatment_date is a 'YYYY-MM-DD' string (or None).
    """
    metadata_excel_path = Path(metadata_excel_path)
    df = pd.read_excel(metadata_excel_path, sheet_name=sheet_name)

    for col in (id_col, treatment_date_col, treatment_type_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in metadata Excel {metadata_excel_path}")

    # Collapse multiple rows per animal (e.g., multiple injections)
    grouped = (
        df[[id_col, treatment_date_col, treatment_type_col]]
        .dropna(subset=[id_col])
        .groupby(id_col, as_index=False)
        .first()
    )

    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    for _, row in grouped.iterrows():
        animal_id = str(row[id_col]).strip()
        tdate_str = _normalize_date_to_str(row[treatment_date_col])
        ttype = str(row[treatment_type_col]).strip() if not pd.isna(row[treatment_type_col]) else "Unknown treatment type"
        metadata[animal_id] = {
            "treatment_date": tdate_str,
            "treatment_type": ttype,
        }

    print(f"\nLoaded metadata for {len(metadata)} animals from {metadata_excel_path}")
    return metadata


# ---------------------------------------------------------------------
# High-level wrapper: run on one NPZ or a root directory
# ---------------------------------------------------------------------
def _compute_default_date_ranges(
    min_date: np.datetime64,
    max_date: np.datetime64,
    treatment_date_str: str,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Compute default 'before' and 'after' date ranges.

    Policy:
    - BEFORE: from min_date up to the day BEFORE treatment.
    - AFTER:  from treatment_date THROUGH max_date.

    If treatment_date lies outside [min_date, max_date], we fall back to:
    - treatment before min_date  -> all data in AFTER
    - treatment after max_date   -> all data in BEFORE
    """
    min_ts = pd.to_datetime(str(min_date))
    max_ts = pd.to_datetime(str(max_date))
    treat_ts = pd.to_datetime(treatment_date_str)

    # treatment before recordings -> all AFTER
    if treat_ts < min_ts:
        date_range_before = (min_ts.strftime("%Y-%m-%d"), min_ts.strftime("%Y-%m-%d"))
        date_range_after = (min_ts.strftime("%Y-%m-%d"), max_ts.strftime("%Y-%m-%d"))
        return date_range_before, date_range_after

    # treatment after recordings -> all BEFORE
    if treat_ts > max_ts:
        date_range_before = (min_ts.strftime("%Y-%m-%d"), max_ts.strftime("%Y-%m-%d"))
        date_range_after = (max_ts.strftime("%Y-%m-%d"), max_ts.strftime("%Y-%m-%d"))
        return date_range_before, date_range_after

    # Normal case
    before_end = treat_ts - pd.Timedelta(days=1)
    if before_end < min_ts:
        before_end = min_ts

    date_range_before = (
        min_ts.strftime("%Y-%m-%d"),
        before_end.strftime("%Y-%m-%d"),
    )
    date_range_after = (
        treat_ts.strftime("%Y-%m-%d"),
        max_ts.strftime("%Y-%m-%d"),
    )
    return date_range_before, date_range_after


def _find_npz_files(npz_or_root: Union[str, Path]) -> Sequence[Path]:
    """
    Given either a single .npz path or a root directory, return a list
    of .npz files to process.

    Directory layout supported:
        root/
          USA5288/USA5288.npz
          USA5337/USA5337.npz
          ...
    plus any .npz that might live directly in `root`.
    """
    p = Path(npz_or_root)

    if p.is_file():
        if p.suffix.lower() == ".npz":
            return [p]
        else:
            raise ValueError(f"File {p} does not have .npz extension.")

    if not p.is_dir():
        raise FileNotFoundError(f"{p} is neither a .npz file nor a directory.")

    npz_paths: list[Path] = []

    # 1) Any .npz directly inside root
    npz_paths.extend(sorted(p.glob("*.npz")))

    # 2) One level down: assume each subdir is a bird folder
    for sub in sorted(p.iterdir()):
        if not sub.is_dir():
            continue
        # Most common pattern: USA5337/USA5337.npz (or any ID/ID.npz)
        candidate = sub / f"{sub.name}.npz"
        if candidate.is_file():
            npz_paths.append(candidate)
        else:
            # Fallback: any .npz inside the subdir
            npz_paths.extend(sorted(sub.glob("*.npz")))

    # Deduplicate
    unique: list[Path] = []
    seen = set()
    for path in npz_paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def run_umap_for_path(
    npz_or_root: Union[str, Path],
    metadata_excel: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    treatment_type_col: str = "Treatment type",
    bins: int = 300,
    brightness_factor: float = 6.0,
) -> None:
    """
    High-level wrapper.

    Run UMAP before/after overlap plots for either:
      - a single NPZ file, OR
      - a directory containing subdirectories per bird, each with a .npz.

    Treatment date and treatment type are automatically loaded from
    `metadata_excel` based on Animal ID.

    Parameters
    ----------
    npz_or_root : str or Path
        Path to a single .npz file OR to a root directory with bird folders.
    metadata_excel : str or Path
        Path to Area_X_lesion_metadata.xlsx (or similar).
    sheet_name : int or str, default 0
        Sheet in the Excel file.
    id_col, treatment_date_col, treatment_type_col : str
        Column names in the Excel file.
    bins, brightness_factor
        Passed through to run_umap_overlap_from_npz.
    """
    npz_or_root = Path(npz_or_root)
    metadata_excel = Path(metadata_excel)

    # Load metadata for all birds
    metadata = load_treatment_metadata_from_excel(
        metadata_excel_path=metadata_excel,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        treatment_type_col=treatment_type_col,
    )

    npz_paths = _find_npz_files(npz_or_root)
    if not npz_paths:
        print(f"No .npz files found under {npz_or_root}")
        return

    print(f"\nFound {len(npz_paths)} NPZ file(s) to process.\n")

    metadata_keys = metadata.keys()

    for npz_path in npz_paths:
        npz_path = Path(npz_path)

        # Infer animal_id generically from filename / parent folder,
        # using metadata keys to choose the right candidate.
        animal_id = _infer_animal_id_from_path(npz_path, metadata_keys=metadata_keys)

        if animal_id is None:
            print(f"[WARN] Could not infer animal_id from {npz_path}. Skipping.")
            continue

        if animal_id not in metadata:
            print(f"[WARN] No metadata found for inferred animal_id {animal_id!r} in Excel. Skipping.")
            continue

        meta = metadata[animal_id]
        treatment_date_str = meta["treatment_date"]
        treatment_type_str = meta["treatment_type"]

        if treatment_date_str is None:
            print(f"[WARN] No treatment date for {animal_id}. Skipping.")
            continue

        try:
            min_date, max_date, _ = get_min_max_recording_dates(npz_path)
        except Exception as e:
            print(f"[ERROR] Could not read {npz_path}: {e}")
            continue

        date_range_before, date_range_after = _compute_default_date_ranges(
            min_date=min_date,
            max_date=max_date,
            treatment_date_str=treatment_date_str,
        )

        run_umap_overlap_from_npz(
            npz_path=npz_path,
            date_range_before=date_range_before,
            date_range_after=date_range_after,
            treatment_date=treatment_date_str,
            treatment_type=treatment_type_str,
            bins=bins,
            brightness_factor=brightness_factor,
        )


# ---------------------------------------------------------------------
# Optional example when run as a standalone script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # EDIT THESE PATHS if you want to run this file directly
    example_root = Path("/path/to/updated_AreaX_outputs")  # can be a single .npz or a directory
    metadata_excel = example_root / "Area_X_lesion_metadata.xlsx"

    if metadata_excel.is_file():
        run_umap_for_path(
            npz_or_root=example_root,
            metadata_excel=metadata_excel,
        )
    else:
        print(
            "Please edit the __main__ block in NPZ_colorcode_groups_UMAP.py\n"
            "to point to your NPZ file or root directory and metadata Excel."
        )


"""
Example Spyder usage
--------------------

from pathlib import Path
import importlib
import NPZ_colorcode_groups_UMAP as umapmod

importlib.reload(umapmod)

root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_excel = root / "Area_X_lesion_metadata.xlsx"

umapmod.run_umap_for_path(
    npz_or_root=root,
    metadata_excel=metadata_excel,
)
"""
