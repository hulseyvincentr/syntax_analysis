# phrase_and_metadata_plotting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Tuple

import json
import numpy as np
import pandas as pd

from phrase_duration_pre_vs_post_grouped import (
    run_batch_phrase_duration_from_excel,
    GroupedPlotsResult,
)

# Try to import your existing Excel organizer
try:
    # Adjust this import if your function lives in a different module
    from organize_metadata_excel import build_areax_metadata, load_metadata_with_schema
except Exception as e:
    organize_metadata_excel = None
    _METADATA_IMPORT_ERR = e


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PhraseAndMetadataResult:
    # Big concatenated phrase-duration stats across birds
    phrase_duration_stats: pd.DataFrame
    # Organized Excel metadata (from organize_metadata_excel)
    organized_metadata: pd.DataFrame
    # Per-animal plotting results (only populated when rebuilt, not when loading)
    per_animal_results: Dict[str, GroupedPlotsResult]
    # Info about compiled phrase_duration_stats file (if used)
    compiled_stats_path: Optional[Path] = None
    compiled_stats_format: Optional[str] = None  # "json" or "npz"
    loaded_from_compiled: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for saving / loading phrase_duration_stats only
# ──────────────────────────────────────────────────────────────────────────────
def _infer_format_from_suffix(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".json":
        return "json"
    if suf == ".npz":
        return "npz"
    # Default to JSON if unknown suffix
    return "json"


def _save_phrase_stats(
    phrase_duration_stats: pd.DataFrame,
    outpath: Path,
    fmt: str = "json",
) -> None:
    """
    Save phrase_duration_stats to disk.

    fmt = "json":
        outpath is a JSON file with records orientation.
    fmt = "npz":
        outpath is a NumPy .npz file with arrays:
            phrase_data, phrase_columns
    """
    fmt = fmt.lower()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        # records-orientation is easy to re-load
        phrase_duration_stats.to_json(outpath, orient="records", indent=2)
    elif fmt == "npz":
        np.savez_compressed(
            outpath,
            phrase_data=phrase_duration_stats.to_numpy(),
            phrase_columns=np.array(phrase_duration_stats.columns, dtype=object),
        )
    else:
        raise ValueError("compiled_stats_format must be 'json' or 'npz'.")


def _load_phrase_stats(
    compiled_path: Union[str, Path],
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load phrase_duration_stats from a compiled file.
    If fmt is None, infer from suffix (.json or .npz).
    """
    compiled_path = Path(compiled_path)
    if fmt is None:
        fmt = _infer_format_from_suffix(compiled_path)

    fmt = fmt.lower()

    if fmt == "json":
        df = pd.read_json(compiled_path, orient="records")
    elif fmt == "npz":
        arr = np.load(compiled_path, allow_pickle=True)
        df = pd.DataFrame(
            arr["phrase_data"],
            columns=[str(c) for c in arr["phrase_columns"]],
        )
    else:
        raise ValueError("compiled_format must be 'json' or 'npz'.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Metadata organizer wrapper
# ──────────────────────────────────────────────────────────────────────────────
def _build_organized_metadata(
    excel_path: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
) -> pd.DataFrame:
    """
    Call your existing organize_metadata_excel() helper to build a rich
    metadata structure, then turn it into a DataFrame if needed.
    """
    if organize_metadata_excel is None:
        raise ImportError(
            "organize_metadata_excel could not be imported.\n"
            "Make sure you have a module 'organize_metadata_excel.py' "
            "with a function organize_metadata_excel(), or adjust the "
            "import in phrase_and_metadata_plotting.py."
        )

    excel_path = Path(excel_path)

    # NOTE: Adjust arguments here if your organize_metadata_excel signature differs.
    organized = organize_metadata_excel(excel_path=excel_path, sheet_name=sheet_name)

    # Already a DataFrame?
    if isinstance(organized, pd.DataFrame):
        return organized

    # Common case: dict-of-dicts keyed by animal_id
    if isinstance(organized, dict):
        try:
            df = pd.DataFrame.from_dict(organized, orient="index").reset_index()
            # You can rename "index" to something like "Animal ID" if you want:
            # df = df.rename(columns={"index": "Animal ID"})
            return df
        except Exception:
            # Fallback: generic frame
            return pd.DataFrame(organized)

    # Last-resort fallback
    return pd.DataFrame(organized)


# ──────────────────────────────────────────────────────────────────────────────
# Core builder: run pre_vs_post grouped for all birds and stack stats
# ──────────────────────────────────────────────────────────────────────────────
def _build_phrase_stats_from_scratch(
    *,
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, GroupedPlotsResult]]:
    """
    Use run_batch_phrase_duration_from_excel to run phrase_duration_pre_vs_post_grouped
    for each bird, then concatenate all res.phrase_duration_stats_df into a single
    phrase_duration_stats DataFrame (with an id_col identifying the bird).
    """
    excel_path = Path(excel_path)
    json_root = Path(json_root)

    batch_results: Dict[str, GroupedPlotsResult] = run_batch_phrase_duration_from_excel(
        excel_path=excel_path,
        json_root=json_root,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        grouping_mode=grouping_mode,
        early_group_size=early_group_size,
        late_group_size=late_group_size,
        post_group_size=post_group_size,
        restrict_to_labels=restrict_to_labels,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
    )

    frames = []
    for animal_id, res in batch_results.items():
        stats = res.phrase_duration_stats_df.copy()
        if stats.empty:
            continue
        # Track which bird this row comes from
        stats[id_col] = animal_id
        frames.append(stats)

    if frames:
        phrase_duration_stats = pd.concat(frames, ignore_index=True)
    else:
        phrase_duration_stats = pd.DataFrame(
            columns=[
                "Group",
                "Syllable",
                "N_phrases",
                "Mean_ms",
                "SEM_ms",
                "Median_ms",
                "Std_ms",
                "Variance_ms2",
                id_col,
            ]
        )

    return phrase_duration_stats, batch_results


# ──────────────────────────────────────────────────────────────────────────────
# Main "build from scratch" API
# ──────────────────────────────────────────────────────────────────────────────
def run_phrase_durations_from_metadata(
    *,
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",    # or "explicit"
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,
    # optional: where to save compiled phrase_duration_stats
    compiled_stats_path: Optional[Union[str, Path]] = None,
    compiled_format: str = "json",          # "json" or "npz"
) -> PhraseAndMetadataResult:
    """
    Build everything from scratch:

      • organized_metadata: via organize_metadata_excel(excel_path, ...)
      • phrase_duration_stats: via run_batch_phrase_duration_from_excel + stacking
      • per_animal_results: full GroupedPlotsResult objects per bird

    Optionally saves phrase_duration_stats to compiled_stats_path in the chosen
    compiled_format ("json" or "npz").
    """
    # 1) Organized metadata from Excel (keeps full Excel info)
    organized_metadata = _build_organized_metadata(
        excel_path=excel_path,
        sheet_name=sheet_name,
    )

    # 2) Phrase-duration stats across birds
    phrase_duration_stats, per_animal_results = _build_phrase_stats_from_scratch(
        excel_path=excel_path,
        json_root=json_root,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        grouping_mode=grouping_mode,
        early_group_size=early_group_size,
        late_group_size=late_group_size,
        post_group_size=post_group_size,
        restrict_to_labels=restrict_to_labels,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
    )

    compiled_path: Optional[Path] = None
    fmt_norm: Optional[str] = None

    if compiled_stats_path is not None:
        compiled_path = Path(compiled_stats_path)
        fmt_norm = compiled_format.lower()
        _save_phrase_stats(
            phrase_duration_stats=phrase_duration_stats,
            outpath=compiled_path,
            fmt=fmt_norm,
        )
        print(f"[INFO] Saved compiled phrase_duration_stats to: {compiled_path} ({fmt_norm})")

    return PhraseAndMetadataResult(
        phrase_duration_stats=phrase_duration_stats,
        organized_metadata=organized_metadata,
        per_animal_results=per_animal_results,
        compiled_stats_path=compiled_path,
        compiled_stats_format=fmt_norm,
        loaded_from_compiled=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entrypoint: load compiled stats OR build from scratch
# ──────────────────────────────────────────────────────────────────────────────
def phrase_and_metadata_plotting(
    *,
    # Option A: load from compiled phrase_duration_stats file
    compiled_stats_path: Optional[Union[str, Path]] = None,
    compiled_format: Optional[str] = None,   # if None, inferred from suffix

    # Option B: build from scratch (also used if compiled file missing)
    excel_path: Optional[Union[str, Path]] = None,
    json_root: Optional[Union[str, Path]] = None,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,
) -> PhraseAndMetadataResult:
    """
    High-level entrypoint.

    Mode 1: If compiled_stats_path is provided and exists:
        - Load phrase_duration_stats from that file.
        - Build organized_metadata from Excel (if excel_path is provided),
          otherwise return an empty metadata DataFrame.
        - per_animal_results is left empty (we don't reconstruct plots from file).

    Mode 2: Otherwise:
        - Require excel_path and json_root.
        - Run full phrase_duration_pre_vs_post_grouped over all birds
          via run_phrase_durations_from_metadata().
        - Optionally save phrase_duration_stats to compiled_stats_path.
    """
    # Try load-from-compiled mode first
    if compiled_stats_path is not None:
        cpath = Path(compiled_stats_path)
        if cpath.exists():
            fmt = compiled_format or _infer_format_from_suffix(cpath)
            phrase_stats = _load_phrase_stats(cpath, fmt)

            if excel_path is not None:
                organized_metadata = _build_organized_metadata(
                    excel_path=excel_path,
                    sheet_name=sheet_name,
                )
            else:
                organized_metadata = pd.DataFrame()

            print(f"[INFO] Loaded compiled phrase_duration_stats from: {cpath} ({fmt})")
            return PhraseAndMetadataResult(
                phrase_duration_stats=phrase_stats,
                organized_metadata=organized_metadata,
                per_animal_results={},
                compiled_stats_path=cpath,
                compiled_stats_format=fmt.lower(),
                loaded_from_compiled=True,
            )
        else:
            print(f"[INFO] compiled_stats_path does not exist yet; will build: {cpath}")

    # Build-from-scratch mode
    if excel_path is None or json_root is None:
        raise ValueError(
            "If compiled_stats_path is not provided or does not exist, "
            "you must supply excel_path and json_root."
        )

    # Decide what format to use when saving if compiled_stats_path is given
    if compiled_stats_path is not None:
        cpath = Path(compiled_stats_path)
        fmt = compiled_format or _infer_format_from_suffix(cpath)
    else:
        cpath = None
        fmt = "json"

    return run_phrase_durations_from_metadata(
        excel_path=excel_path,
        json_root=json_root,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        grouping_mode=grouping_mode,
        early_group_size=early_group_size,
        late_group_size=late_group_size,
        post_group_size=post_group_size,
        restrict_to_labels=restrict_to_labels,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
        compiled_stats_path=cpath,
        compiled_format=fmt,
    )


"""
OPTION 1: 
from pathlib import Path
import importlib
import phrase_and_metadata_plotting as pmp
importlib.reload(pmp)

excel_path = Path("/Users/mirandahulsey-vincent/Desktop/Area_X_lesion_metadata.xlsx")
json_root  = Path("//Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_lesion_metadata.xlsx")
compiled   = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/compiled_phrase_duration_stats.json")

res = pmp.phrase_and_metadata_plotting(
    excel_path=excel_path,
    json_root=json_root,
    compiled_stats_path=compiled,  # will be CREATED here
    compiled_format="json",        # or "npz"
    sheet_name=0,
    id_col="Animal ID",
    treatment_date_col="Treatment date",
    grouping_mode="auto_balance",
    restrict_to_labels=[str(i) for i in range(26)],
    y_max_ms=40000,
    show_plots=False,
)

phrase_df = res.phrase_duration_stats       # big phrase-duration table
meta_df   = res.organized_metadata          # from organize_metadata_excel

OPTION 2:
from pathlib import Path
import importlib
import phrase_and_metadata_plotting as pmp
importlib.reload(pmp)

excel_path = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_lesion_metadata.xlsx")
compiled   = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/compiled_phrase_duration_stats.json")

res = pmp.phrase_and_metadata_plotting(
    compiled_stats_path=compiled,  # file already exists
    compiled_format="json",        # optional, inferred from suffix
    excel_path=excel_path,         # so we can still build organized_metadata
    sheet_name=0,
)

phrase_df = res.phrase_duration_stats
meta_df   = res.organized_metadata



"""