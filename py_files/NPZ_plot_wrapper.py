#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ_plot_wrapper.py

Wrapper to run BOTH:

1) UMAP overlap plots (before vs after treatment) using
   NPZ_colorcode_groups_UMAP.run_umap_overlap_from_npz

2) Syllable repertoire / sample spectrograms using
   syllable_sample_spectrograms.plot_spectrogram_samples_for_labels

on either:
    - a single NPZ file, or
    - a root directory containing per-bird NPZs.

Expected metadata Excel columns:
    - "Animal ID"
    - "Treatment date"   (date or string parsable by pandas)
    - "Treatment type"   (string)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, List, Union, Tuple
import re

import numpy as np
import pandas as pd

import NPZ_colorcode_groups_UMAP as umapmod
import syllable_sample_spectrograms as sss


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iter_npz_files(npz_or_root: Path) -> List[Path]:
    """
    Return a list of NPZ files.

    - If npz_or_root is a file and ends with .npz → [that file]
    - If it's a directory → all *.npz under it (recursively)
    """
    npz_or_root = Path(npz_or_root)
    if npz_or_root.is_file() and npz_or_root.suffix == ".npz":
        return [npz_or_root]
    if npz_or_root.is_dir():
        return sorted(npz_or_root.rglob("*.npz"))
    raise FileNotFoundError(f"{npz_or_root} is neither a .npz file nor a directory.")


def _infer_animal_id_from_path(p: Path) -> str:
    """
    Try to infer animal ID from an NPZ path.

    Looks for patterns like 'USA5337' in the filename first;
    if not found, falls back to the stem.
    """
    name = p.name
    m = re.search(r"(USA\d{4})", name)
    if m:
        return m.group(1)

    # As a fallback, search entire path string
    m2 = re.search(r"(USA\d{4})", p.as_posix())
    if m2:
        return m2.group(1)

    # Last fallback: just use the stem of the filename
    return p.stem


def _load_metadata_dict(
    excel_path: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    treatment_type_col: str = "Treatment type",
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Load metadata Excel and build a dict:

        meta[animal_id] = {
            "treatment_date": "YYYY-MM-DD" or None,
            "treatment_type": str or None,
        }
    """
    excel_path = Path(excel_path)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Normalize column names a bit (strip whitespace)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    if id_col not in df.columns:
        raise KeyError(f"ID column '{id_col}' not found in {excel_path}")

    if treatment_date_col not in df.columns:
        raise KeyError(f"Treatment date column '{treatment_date_col}' not found in {excel_path}")

    if treatment_type_col not in df.columns:
        raise KeyError(f"Treatment type column '{treatment_type_col}' not found in {excel_path}")

    # Coerce dates
    df[treatment_date_col] = pd.to_datetime(df[treatment_date_col], errors="coerce")

    meta: Dict[str, Dict[str, Optional[str]]] = {}

    for _, row in df.iterrows():
        raw_id = row[id_col]
        if pd.isna(raw_id):
            continue
        animal_id = str(raw_id).strip()
        if not animal_id:
            continue

        tdate = row[treatment_date_col]
        if pd.isna(tdate):
            tdate_str = None
        else:
            tdate_str = pd.to_datetime(tdate).strftime("%Y-%m-%d")

        ttype = row[treatment_type_col]
        if pd.isna(ttype):
            ttype_str = None
        else:
            ttype_str = str(ttype).strip()

        meta[animal_id] = {
            "treatment_date": tdate_str,
            "treatment_type": ttype_str,
        }

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Public wrapper
# ──────────────────────────────────────────────────────────────────────────────

def NPZ_plot_wrapper(
    npz_or_root: Union[str, Path],
    metadata_excel: Union[str, Path],
    *,
    # Where to save repertoire / spectrogram plots; if None, each NPZ's parent dir is used
    repertoire_root: Optional[Union[str, Path]] = None,
    # Excel metadata settings
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    treatment_type_col: str = "Treatment type",
    # UMAP overlap settings
    bins: int = 300,
    brightness_factor: float = 6.0,
    # Spectrogram / repertoire settings
    selected_labels: Optional[Sequence[int]] = None,  # None → all labels (except noise if skip_noise_label)
    skip_noise_label: bool = True,
    spectrogram_length: int = 1000,
    num_sample_spectrograms: int = 1,
    cmap: str = "gray_r",
    show_colorbar: bool = False,
    show_plots: bool = True,
    save_sample_spectrograms: bool = True,
    make_umap_plot: bool = True,
    show_umap_legend: bool = True,  # currently only used inside syllable_sample_spectrograms
) -> None:
    """
    Run UMAP overlap + syllable sample spectrogram plots on one NPZ or a tree of NPZs.

    Parameters
    ----------
    npz_or_root : str or Path
        Single TweetyBERT NPZ file or a root directory containing per-bird NPZs.
    metadata_excel : str or Path
        Path to Area_X_lesion_metadata.xlsx (or equivalent).
    repertoire_root : str or Path, optional
        Root where per-bird repertoire folders will be created.
        If None, each NPZ's parent directory is used.
    Other parameters are forwarded to the underlying NPZ_colorcode_groups_UMAP
    and syllable_sample_spectrograms functions (where compatible).
    """
    npz_or_root = Path(npz_or_root)
    metadata_excel = Path(metadata_excel)

    if repertoire_root is not None:
        repertoire_root = Path(repertoire_root)

    # Load metadata once
    meta = _load_metadata_dict(
        metadata_excel,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        treatment_type_col=treatment_type_col,
    )

    # Collect NPZ files
    all_npz_files = _iter_npz_files(npz_or_root)
    if not all_npz_files:
        print(f"[WARN] No .npz files found under {npz_or_root}")
        return

    print(f"[INFO] Found {len(all_npz_files)} NPZ file(s) under {npz_or_root}")

    # Process each NPZ
    for npz_path in all_npz_files:
        print("\n" + "=" * 80)
        print(f"[INFO] Processing NPZ: {npz_path}")
        animal_id = _infer_animal_id_from_path(npz_path)
        print(f"[INFO] Inferred animal ID: {animal_id}")

        # Decide save directory for this bird (for spectrogram + label UMAP)
        if repertoire_root is not None:
            bird_outdir = repertoire_root / f"{animal_id}_repertoire"
        else:
            bird_outdir = npz_path.parent / f"{animal_id}_repertoire"

        bird_outdir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output directory: {bird_outdir}")

        # Look up metadata
        meta_row = meta.get(animal_id)
        if meta_row is None:
            print(f"[WARN] No metadata row found for '{animal_id}' in {metadata_excel}.")
            treatment_date_str = None
            treatment_type_str = "Unknown treatment"
        else:
            treatment_date_str = meta_row.get("treatment_date")
            treatment_type_str = meta_row.get("treatment_type") or "Unknown treatment"

        # ────────────────────────────────────────────────────────────────
        # 1) Spectrogram samples + (optional) basic label-colored UMAP
        # ────────────────────────────────────────────────────────────────
        try:
            sss.plot_spectrogram_samples_for_labels(
                npz_path=npz_path,
                output_dir=bird_outdir,
                selected_labels=selected_labels,
                skip_noise_label=skip_noise_label,
                spectrogram_length=spectrogram_length,
                num_sample_spectrograms=num_sample_spectrograms,
                cmap=cmap,
                show_colorbar=show_colorbar,
                show_plots=show_plots,
                save_sample_spectrograms=save_sample_spectrograms,
                # We let NPZ_colorcode_groups_UMAP handle the "before vs after" UMAP
                make_umap_plot=True,
                show_umap_legend=True,
            )
        except Exception as e:
            print(f"[ERROR] Spectrogram sampling failed for {npz_path}: {e}")

        # ────────────────────────────────────────────────────────────────
        # 2) UMAP overlap (before vs after treatment)
        # ────────────────────────────────────────────────────────────────
        if not make_umap_plot:
            continue

        if treatment_date_str is None:
            print(f"[WARN] No treatment date for '{animal_id}'; skipping UMAP overlap plot.")
            continue

        try:
            # Get earliest / latest recording dates from NPZ
            min_date, max_date, unique_dates = umapmod.get_min_max_recording_dates(npz_path)
        except Exception as e:
            print(f"[ERROR] get_min_max_recording_dates failed for {npz_path}: {e}")
            continue

        # Convert treatment date to numpy datetime64 for comparison
        try:
            treat_dt = np.datetime64(treatment_date_str)
        except Exception as e:
            print(f"[WARN] Could not parse treatment date '{treatment_date_str}' for '{animal_id}': {e}")
            continue

        # Ensure treatment date is strictly between min_date and max_date
        if not (min_date < treat_dt < max_date):
            print(
                f"[WARN] Treatment date {treat_dt} is not strictly between "
                f"earliest {min_date} and latest {max_date} for '{animal_id}'; "
                f"skipping UMAP overlap plot."
            )
            continue

        # Build date ranges (string form)
        date_range_before: Tuple[str, str] = (str(min_date), treatment_date_str)
        date_range_after: Tuple[str, str] = (treatment_date_str, str(max_date))

        print(f"[INFO] Using date_range_before = {date_range_before}")
        print(f"[INFO] Using date_range_after  = {date_range_after}")
        print(f"[INFO] Treatment type: {treatment_type_str}")

        try:
            # NOTE: only pass parameters that actually exist in your current
            # run_umap_overlap_from_npz signature
            umapmod.run_umap_overlap_from_npz(
                npz_path=npz_path,
                date_range_before=date_range_before,
                date_range_after=date_range_after,
                treatment_date=treatment_date_str,
                treatment_type=treatment_type_str,
                bins=bins,
                brightness_factor=brightness_factor,
            )
        except Exception as e:
            print(f"[ERROR] UMAP overlap plotting failed for {npz_path}: {e}")


if __name__ == "__main__":
    # Minimal local test stub (edit paths if you want to run this file directly)
    example_root = Path("/path/to/updated_AreaX_outputs")
    example_metadata = example_root / "Area_X_lesion_metadata.xlsx"
    print("[INFO] NPZ_plot_wrapper.py loaded as a script. Edit __main__ if you want to test directly.")
    # NPZ_plot_wrapper(example_root, example_metadata)
