# phrase_and_metadata_plotting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phrase_duration_pre_vs_post_grouped import (
    GroupedPlotsResult,
)

# Helper that builds big_df across birds
from phrase_duration_birds_stats_df import (
    BirdsPhraseDurationStats,
    build_birds_phrase_duration_stats_df,
)

# Try to import your existing Excel organizer (Area X–specific)
try:
    from organize_metadata_excel import build_areax_metadata

    def organize_metadata_excel(
        excel_path: Union[str, Path],
        *,
        sheet_name: Union[int, str] = 0,
    ) -> Dict[str, Dict[str, object]]:
        """
        Wrapper that lets this module call a unified name organize_metadata_excel.

        Internally uses build_areax_metadata(excel_file=..., sheet_name=...).
        """
        return build_areax_metadata(excel_file=excel_path, sheet_name=sheet_name)

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
    compiled_stats_format: Optional[str] = None  # "json", "npz", or "csv"
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
    if suf == ".csv":
        return "csv"
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
    fmt = "csv":
        outpath is a CSV file (index=False).
    """
    fmt = fmt.lower()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        phrase_duration_stats.to_json(outpath, orient="records", indent=2)
    elif fmt == "npz":
        np.savez_compressed(
            outpath,
            phrase_data=phrase_duration_stats.to_numpy(),
            phrase_columns=np.array(phrase_duration_stats.columns, dtype=object),
        )
    elif fmt == "csv":
        phrase_duration_stats.to_csv(outpath, index=False)
    else:
        raise ValueError("compiled_stats_format must be 'json', 'npz', or 'csv'.")


def _load_phrase_stats(
    compiled_path: Union[str, Path],
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load phrase_duration_stats from a compiled file.
    If fmt is None, infer from suffix (.json, .npz, or .csv).
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
    elif fmt == "csv":
        df = pd.read_csv(compiled_path)
    else:
        raise ValueError("compiled_format must be 'json', 'npz', or 'csv'.")

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
            "organize_metadata_excel/build_areax_metadata could not be imported.\n"
            "Make sure you have a module 'organize_metadata_excel.py' "
            "with a function build_areax_metadata(), or adjust the "
            "import/wrapper in phrase_and_metadata_plotting.py."
        )

    excel_path = Path(excel_path)

    organized = organize_metadata_excel(excel_path=excel_path, sheet_name=sheet_name)

    # Already a DataFrame?
    if isinstance(organized, pd.DataFrame):
        return organized

    # Common case: dict-of-dicts keyed by animal_id
    if isinstance(organized, dict):
        try:
            df = pd.DataFrame.from_dict(organized, orient="index").reset_index()
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
    Use build_birds_phrase_duration_stats_df (which wraps
    run_batch_phrase_duration_from_excel / phrase_duration_pre_vs_post_grouped)
    for each bird, then return:

      - phrase_duration_stats: big concatenated DataFrame
      - per_animal_results: dict[animal_id, GroupedPlotsResult]
    """
    stats_result: BirdsPhraseDurationStats = build_birds_phrase_duration_stats_df(
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

    return stats_result.phrase_duration_stats_df, stats_result.per_animal_results


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
    compiled_format: str = "json",          # "json", "npz", or "csv"
) -> PhraseAndMetadataResult:
    """
    Build everything from scratch:

      • organized_metadata: via organize_metadata_excel(excel_path, ...)
      • phrase_duration_stats: via build_birds_phrase_duration_stats_df + stacking
      • per_animal_results: full GroupedPlotsResult objects per bird

    Optionally saves phrase_duration_stats to compiled_stats_path in the chosen
    compiled_format ("json", "npz", or "csv").
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
        - Load phrase_duration_stats from that file (csv/json/npz).
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


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_long_color_cycle(n: int):
    """
    Build a long color cycle by concatenating tab20, tab20b, and tab20c.
    Handles up to ~60 distinct colors; repeats if n is larger.
    """
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        colors.extend(list(cmap.colors))

    if n <= len(colors):
        return colors[:n]

    reps = int(np.ceil(n / len(colors)))
    colors = (colors * reps)[:n]
    return colors


def _pretty_axes_basic(ax, x_rotation: int = 0):
    """Simple styling: remove top/right spines, set ticks, optional x tick rotation."""
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    if x_rotation:
        for lab in ax.get_xticklabels():
            lab.set_rotation(x_rotation)
            lab.set_horizontalalignment("right")


# ──────────────────────────────────────────────────────────────────────────────
# Combined plot + filtered subsets (mean / median / variance)
# plus additional sets colored by Medial / Lateral Area X hit type
# ──────────────────────────────────────────────────────────────────────────────
def plot_compiled_phrase_stats_by_syllable(
    phrase_stats: Optional[pd.DataFrame] = None,
    *,
    compiled_stats_path: Optional[Union[str, Path]] = None,
    compiled_format: Optional[str] = None,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syllable_col: str = "Syllable",
    mean_col: str = "Mean_ms",
    sem_col: str = "SEM_ms",
    group_order: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: str = "phrase_duration_lines",
    show_plots: bool = True,
    # metadata for anatomy-based coloring
    metadata_excel_path: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Union[int, str] = 0,
    medial_hit_col: str = "Medial Area X hit?",
) -> None:
    """
    Make:
      • One combined figure of all syllables (colored by animal_id)
      • Three filtered figures:
          - post mean > pre mean + 1 SD      (Y-axis = Mean_ms)
          - post median > max(pre medians)   (Y-axis = Median_ms)
          - post variance > pre pooled var   (Y-axis = Variance_ms2)

    For EACH filtered subset we additionally make:
      • a version colored by Medial Area X hit?   (Y / N / unknown)
      • a version colored by Medial Area X hit type
      • a version colored by Lateral Area X hit type

    Medial / lateral hit *types* are read from build_areax_metadata(), which
    adds the fields:
        "Medial Area X hit type"
        "Lateral Area X hit type"
    """
    # --- Load phrase_stats if not provided ---
    if phrase_stats is None:
        if compiled_stats_path is None:
            raise ValueError(
                "Either phrase_stats must be provided or compiled_stats_path "
                "must be given."
            )
        cpath = Path(compiled_stats_path)
        if not cpath.exists():
            raise FileNotFoundError(f"compiled_stats_path does not exist: {cpath}")
        fmt = compiled_format or _infer_format_from_suffix(cpath)
        phrase_stats = _load_phrase_stats(cpath, fmt)

    df_base = phrase_stats.copy()

    # Normalize group labels: "Early-Pre", "Early Pre " -> "Early Pre"
    df_base[group_col] = (
        df_base[group_col]
        .astype(str)
        .str.strip()
        .str.replace("-", " ", regex=False)
    )

    # Ensure required columns exist for plotting
    required_cols = {id_col, group_col, syllable_col, mean_col, sem_col}
    missing = required_cols - set(df_base.columns)
    if missing:
        raise KeyError(
            f"phrase_stats is missing required columns: {sorted(missing)}"
        )

    PRE_GROUPS = {"Early Pre", "Late Pre"}
    POST_GROUP = "Post"

    # Decide group order for x-axis
    if group_order is None:
        groups = list(df_base[group_col].unique())
        default_order = ["Early Pre", "Late Pre", "Post"]
        ordered = [g for g in default_order if g in groups]
        remaining = [g for g in sorted(groups) if g not in ordered]
        group_order = ordered + remaining
    else:
        group_order = [
            str(g).strip().replace("-", " ")
            for g in group_order
        ]

    # Map each group to a numeric x-position
    group_to_x = {g: i for i, g in enumerate(group_order)}

    # Color cycle based on number of animals (for animal-colored plots)
    animal_ids = sorted(df_base[id_col].dropna().unique(), key=str)
    colors = _make_long_color_cycle(len(animal_ids))

    # Create output dir if needed
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = None

    # ------------------------------------------------------------------
    # Build metadata-based mappings:
    #   - medial_hit_map       : raw "Medial Area X hit?" per animal
    #   - medial_type_map      : "Medial Area X hit type" per animal
    #   - lateral_type_map     : "Lateral Area X hit type" per animal
    # ------------------------------------------------------------------
    medial_hit_map: Dict[str, object] = {}
    medial_type_map: Dict[str, str] = {}
    lateral_type_map: Dict[str, str] = {}

    if metadata_excel_path is not None:
        # 1) Directly from the Excel sheet: "Medial Area X hit?"
        try:
            meta_df = pd.read_excel(metadata_excel_path, sheet_name=metadata_sheet_name)
            if id_col in meta_df.columns and medial_hit_col in meta_df.columns:
                grouped = (
                    meta_df.groupby(id_col)[medial_hit_col]
                    .apply(lambda s: next((v for v in s if pd.notna(v)), np.nan))
                    .reset_index()
                )
                for _, row in grouped.iterrows():
                    medial_hit_map[str(row[id_col])] = row[medial_hit_col]
            else:
                print(
                    f"[WARN] metadata Excel missing '{id_col}' or '{medial_hit_col}' "
                    "columns; Medial-hit Y/N coloring disabled."
                )
        except Exception as e:
            print(f"[WARN] Could not read metadata Excel for Medial-hit Y/N: {e}")

        # 2) Use build_areax_metadata to get hit *types*
        try:
            from organize_metadata_excel import build_areax_metadata as _bam

            meta_dict = _bam(excel_file=metadata_excel_path,
                             sheet_name=metadata_sheet_name)
            for aid, entry in meta_dict.items():
                akey = str(aid)
                med_type = entry.get("Medial Area X hit type", "unknown")
                lat_type = entry.get("Lateral Area X hit type", "unknown")

                med_type = str(med_type).strip() or "unknown"
                lat_type = str(lat_type).strip() or "unknown"

                medial_type_map[akey] = med_type
                lateral_type_map[akey] = lat_type
        except Exception as e:
            print(
                "[WARN] Could not derive Medial/Lateral Area X hit types from "
                f"organize_metadata_excel: {e}"
            )
            medial_type_map.clear()
            lateral_type_map.clear()

    def _is_medial_hit(animal: str) -> Optional[bool]:
        """Return True/False/None based on Medial Area X hit? mapping."""
        if not medial_hit_map:
            return None
        val = medial_hit_map.get(str(animal))
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().upper()
        if not s:
            return None
        if s in {"Y", "YES", "TRUE", "1"}:
            return True
        if s in {"N", "NO", "FALSE", "0"}:
            return False
        return None

    # Precompute category maps for all animals we might plot
    medial_hit_cat_map: Dict[str, str] = {}
    for animal in animal_ids:
        hit = _is_medial_hit(animal)
        if hit is True:
            cat = "Y"
        elif hit is False:
            cat = "N"
        else:
            cat = "Unknown"
        medial_hit_cat_map[str(animal)] = cat

    medial_type_cat_map: Dict[str, str] = {}
    lateral_type_cat_map: Dict[str, str] = {}
    for animal in animal_ids:
        akey = str(animal)
        med_t = medial_type_map.get(akey, "unknown") or "unknown"
        lat_t = lateral_type_map.get(akey, "unknown") or "unknown"
        medial_type_cat_map[akey] = str(med_t)
        lateral_type_cat_map[akey] = str(lat_t)

    # ------------------------------------------------------------------
    # Inner helper: draw ONE combined figure (colored by animal_id)
    # ------------------------------------------------------------------
    def _plot_one_combined(
        df_plot: pd.DataFrame,
        title: str,
        filename_suffix: str,
        *,
        legend_mode: str = "animal",  # "animal" or "animal_syllables"
        y_col: str = mean_col,
        yerr_col: Optional[str] = sem_col,
        y_label: str = "Mean phrase duration (ms)",
    ) -> None:
        """
        legend_mode:
          - "animal"          → legend entry per animal_id
          - "animal_syllables"→ legend entry per animal_id with syllables listed
        """
        syllables = sorted(df_plot[syllable_col].dropna().unique(), key=str)
        if not syllables:
            print(f"[WARN] No syllables found for subset '{filename_suffix}'; skipping.")
            return

        fig, ax = plt.subplots(figsize=(8, 5.5))

        if legend_mode == "animal_syllables":
            animal_to_syllables: Dict[str, Sequence[object]] = (
                df_plot.groupby(id_col)[syllable_col]
                .apply(lambda s: sorted(pd.unique(s)))
                .to_dict()
            )
        else:
            animal_to_syllables = {}

        seen_animals = set()
        legend_handles = []
        legend_labels = []

        for a_idx, animal in enumerate(animal_ids):
            color = colors[a_idx]
            a_all = df_plot[df_plot[id_col] == animal]
            if a_all.empty:
                continue

            for syll in syllables:
                a_df = a_all[a_all[syllable_col] == syll]
                if a_df.empty:
                    continue

                x_vals: list[float] = []
                y_vals: list[float] = []
                y_errs: list[float] = []

                for g in group_order:
                    rows = a_df[a_df[group_col] == g]
                    if rows.empty:
                        continue
                    x_vals.append(group_to_x[g])
                    y_vals.append(rows[y_col].iloc[0])
                    if yerr_col is not None and yerr_col in rows:
                        y_errs.append(rows[yerr_col].iloc[0])

                if not x_vals:
                    continue

                if yerr_col is None or not y_errs:
                    (line,) = ax.plot(
                        x_vals,
                        y_vals,
                        "o-",
                        linewidth=1.0,
                        markersize=3.5,
                        color=color,
                        alpha=0.35,
                    )
                    handle = line
                else:
                    container = ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_errs,
                        fmt="o-",
                        linewidth=1.0,
                        markersize=3.5,
                        capsize=2.5,
                        color=color,
                        alpha=0.35,
                    )
                    handle = container

                if animal not in seen_animals:
                    seen_animals.add(animal)
                    if legend_mode == "animal_syllables":
                        syll_list = animal_to_syllables.get(animal, [])
                        if syll_list:
                            syll_text = ", ".join(str(s) for s in syll_list)
                            label = f"{animal} (syll {syll_text})"
                        else:
                            label = str(animal)
                    else:
                        label = str(animal)

                    legend_handles.append(handle)
                    legend_labels.append(label)

        ax.set_xticks(list(group_to_x.values()))
        ax.set_xticklabels(list(group_to_x.keys()))
        ax.set_xlabel("Group")
        ax.set_ylabel(y_label)
        ax.set_title(title)

        _pretty_axes_basic(ax, x_rotation=0)

        if legend_handles:
            legend_title = (
                f"{id_col} (syllables)" if legend_mode == "animal_syllables" else id_col
            )
            ax.legend(
                legend_handles,
                legend_labels,
                title=legend_title,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=8,
            )

        fig.tight_layout(rect=[0.0, 0.0, 0.8, 0.95])

        if outdir is not None:
            fname = f"{file_prefix}{filename_suffix}.png"
            fig.savefig(outdir / fname, dpi=300)
            print(f"[PLOT] Saved combined figure: {outdir / fname}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Generic helper: draw ONE figure colored by a per-animal category
    # ------------------------------------------------------------------
    def _plot_one_by_category(
        df_plot: pd.DataFrame,
        title: str,
        filename_suffix: str,
        *,
        y_col: str,
        yerr_col: Optional[str],
        y_label: str,
        category_map: Dict[str, str],
        legend_title: str,
        category_colors: Dict[str, str],
        category_labels: Optional[Dict[str, str]] = None,
        category_order: Optional[Sequence[str]] = None,
    ) -> None:
        """
        category_map:    animal_id -> category string (e.g. 'bilateral', 'miss')
        category_colors: category  -> matplotlib color
        category_labels: category  -> legend label (optional)
        """
        if not category_map:
            print(
                f"[INFO] No category_map for '{legend_title}'; "
                f"skipping plot '{filename_suffix}'."
            )
            return

        syllables = sorted(df_plot[syllable_col].dropna().unique(), key=str)
        if not syllables:
            print(f"[WARN] No syllables found for subset '{filename_suffix}'; skipping.")
            return

        fig, ax = plt.subplots(figsize=(8, 5.5))
        legend_handles: Dict[str, object] = {}

        for animal in animal_ids:
            a_all = df_plot[df_plot[id_col] == animal]
            if a_all.empty:
                continue

            cat = category_map.get(str(animal), "unknown")
            if cat is None or (isinstance(cat, float) and np.isnan(cat)):
                cat = "unknown"
            cat = str(cat)

            color = category_colors.get(cat, category_colors.get("unknown", "gray"))

            for syll in syllables:
                a_df = a_all[a_all[syllable_col] == syll]
                if a_df.empty:
                    continue

                x_vals: list[float] = []
                y_vals: list[float] = []
                y_errs: list[float] = []

                for g in group_order:
                    rows = a_df[a_df[group_col] == g]
                    if rows.empty:
                        continue
                    x_vals.append(group_to_x[g])
                    y_vals.append(rows[y_col].iloc[0])
                    if yerr_col is not None and yerr_col in rows:
                        y_errs.append(rows[yerr_col].iloc[0])

                if not x_vals:
                    continue

                if yerr_col is None or not y_errs:
                    (line,) = ax.plot(
                        x_vals,
                        y_vals,
                        "o-",
                        linewidth=1.0,
                        markersize=3.5,
                        color=color,
                        alpha=0.35,
                    )
                    handle = line
                else:
                    container = ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_errs,
                        fmt="o-",
                        linewidth=1.0,
                        markersize=3.5,
                        capsize=2.5,
                        color=color,
                        alpha=0.35,
                    )
                    handle = container

                if cat not in legend_handles:
                    legend_handles[cat] = handle

        ax.set_xticks(list(group_to_x.values()))
        ax.set_xticklabels(list(group_to_x.keys()))
        ax.set_xlabel("Group")
        ax.set_ylabel(y_label)
        ax.set_title(title)

        _pretty_axes_basic(ax, x_rotation=0)

        if legend_handles:
            # Determine which categories to show in legend
            if category_order is None:
                cats_in_plot = list(legend_handles.keys())
            else:
                # Start with requested order, then append any others we saw
                cats_in_plot = [c for c in category_order if c in legend_handles]
                for c in legend_handles.keys():
                    if c not in cats_in_plot:
                        cats_in_plot.append(c)

            handles = [legend_handles[c] for c in cats_in_plot]
            labels = []
            for c in cats_in_plot:
                if category_labels and c in category_labels:
                    labels.append(category_labels[c])
                else:
                    labels.append(str(c))

            ax.legend(
                handles,
                labels,
                title=legend_title,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=8,
            )

        fig.tight_layout(rect=[0.0, 0.0, 0.8, 0.95])

        if outdir is not None:
            fname = f"{file_prefix}{filename_suffix}.png"
            fig.savefig(outdir / fname, dpi=300)
            print(f"[PLOT] Saved category-colored figure: {outdir / fname}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # 1) Plot ALL syllables: one legend entry per animal
    # ------------------------------------------------------------------
    _plot_one_combined(
        df_plot=df_base,
        title="All syllables: mean ± SEM by group and animal",
        filename_suffix="_all_syllables_combined",
        legend_mode="animal",
        y_col=mean_col,
        yerr_col=sem_col,
        y_label="Mean phrase duration (ms)",
    )

    # Common mask for Post rows
    post_mask = (df_base[group_col] == POST_GROUP)
    key_cols = [id_col, syllable_col]

    # Convenience category configs
    yN_colors = {"Y": "red", "N": "black", "Unknown": "gray", "unknown": "gray"}
    yN_labels = {
        "Y": "Medial Area X hit = Y",
        "N": "Medial Area X hit = N",
        "Unknown": "Medial Area X hit = unknown",
        "unknown": "Medial Area X hit = unknown",
    }
    yN_order = ["Y", "N", "Unknown"]

    # UPDATED: include 'sham' and keep palette consistent for Medial + Lateral
    type_colors = {
        "bilateral": "red",
        "unilateral_L": "orange",
        "unilateral_R": "purple",
        "sham": "green",     # sham controls
        "miss": "black",
        "unknown": "gray",
    }

    type_order = [
        "bilateral",
        "unilateral_L",
        "unilateral_R",
        "sham",
        "miss",
        "unknown",
    ]

    medial_type_labels = {
        "bilateral": "bilateral Medial Area X hit",
        "unilateral_L": "unilateral Medial hit (L)",
        "unilateral_R": "unilateral Medial hit (R)",
        "sham": "bilateral saline sham (Medial)",
        "miss": "no Medial Area X hit",
        "unknown": "Medial Area X type unknown",
    }
    lateral_type_labels = {
        "bilateral": "bilateral Lateral Area X hit",
        "unilateral_L": "unilateral Lateral hit (L)",
        "unilateral_R": "unilateral Lateral hit (R)",
        "sham": "bilateral saline sham (Lateral)",
        "miss": "no Lateral Area X hit",
        "unknown": "Lateral Area X type unknown",
    }

    # ------------------------------------------------------------------
    # 2) Filter by MEAN: post mean > pre mean + 1 SD
    # ------------------------------------------------------------------
    mean_metric_cols = {"Post_Mean_Above_Pre_Mean_Plus_1SD"}
    if mean_metric_cols.issubset(df_base.columns):
        interesting_post_mean = df_base[
            post_mask & df_base["Post_Mean_Above_Pre_Mean_Plus_1SD"]
        ]
        if not interesting_post_mean.empty:
            keys_mean = interesting_post_mean[key_cols].drop_duplicates()
            df_filtered_mean = df_base.merge(keys_mean, on=key_cols, how="inner")

            # Animal-colored + syllables in legend
            _plot_one_combined(
                df_plot=df_filtered_mean,
                title="Filtered syllables: post mean > pre mean + 1 SD",
                filename_suffix="_filtered_mean_gt_pre_plus1SD",
                legend_mode="animal_syllables",
                y_col=mean_col,
                yerr_col=sem_col,
                y_label="Mean phrase duration (ms)",
            )

            # Colored by Medial Area X hit? (Y/N)
            _plot_one_by_category(
                df_plot=df_filtered_mean,
                title=(
                    "Filtered syllables: post mean > pre mean + 1 SD\n"
                    "colored by Medial Area X hit?"
                ),
                filename_suffix="_filtered_mean_gt_pre_plus1SD_MhitYN",
                y_col=mean_col,
                yerr_col=sem_col,
                y_label="Mean phrase duration (ms)",
                category_map=medial_hit_cat_map,
                legend_title="Medial Area X hit?",
                category_colors=yN_colors,
                category_labels=yN_labels,
                category_order=yN_order,
            )

            # Colored by Medial Area X hit type
            _plot_one_by_category(
                df_plot=df_filtered_mean,
                title=(
                    "Filtered syllables: post mean > pre mean + 1 SD\n"
                    "colored by Medial Area X hit type"
                ),
                filename_suffix="_filtered_mean_gt_pre_plus1SD_Mtype",
                y_col=mean_col,
                yerr_col=sem_col,
                y_label="Mean phrase duration (ms)",
                category_map=medial_type_cat_map,
                legend_title="Medial Area X hit type",
                category_colors=type_colors,
                category_labels=medial_type_labels,
                category_order=type_order,
            )

            # Colored by Lateral Area X hit type
            _plot_one_by_category(
                df_plot=df_filtered_mean,
                title=(
                    "Filtered syllables: post mean > pre mean + 1 SD\n"
                    "colored by Lateral Area X hit type"
                ),
                filename_suffix="_filtered_mean_gt_pre_plus1SD_Ltype",
                y_col=mean_col,
                yerr_col=sem_col,
                y_label="Mean phrase duration (ms)",
                category_map=lateral_type_cat_map,
                legend_title="Lateral Area X hit type",
                category_colors=type_colors,
                category_labels=lateral_type_labels,
                category_order=type_order,
            )
        else:
            print("[INFO] No syllables met the mean > pre+1SD criterion.")
    else:
        print(
            "[INFO] Column 'Post_Mean_Above_Pre_Mean_Plus_1SD' not found; "
            "skipping mean-based filtered plot."
        )

    # ------------------------------------------------------------------
    # 3) Filter by MEDIAN: post median > max(Early-Pre median, Late-Pre median)
    # ------------------------------------------------------------------
    if "Median_ms" in df_base.columns:
        df_pre = df_base[df_base[group_col].isin(PRE_GROUPS)]
        df_post = df_base[df_base[group_col] == POST_GROUP]

        if not df_pre.empty and not df_post.empty:
            pre_medians = (
                df_pre
                .groupby(key_cols)["Median_ms"]
                .max()                     # bigger of Early & Late
                .reset_index()
                .rename(columns={"Median_ms": "Pre_Median_max_ms"})
            )
            post_medians = (
                df_post[key_cols + ["Median_ms"]]
                .rename(columns={"Median_ms": "Post_Median_ms"})
            )

            merged = pd.merge(post_medians, pre_medians, on=key_cols, how="inner")
            interesting_med = merged[
                merged["Post_Median_ms"] > merged["Pre_Median_max_ms"]
            ]

            if not interesting_med.empty:
                keys_med = interesting_med[key_cols].drop_duplicates()
                df_filtered_med = df_base.merge(keys_med, on=key_cols, how="inner")

                _plot_one_combined(
                    df_plot=df_filtered_med,
                    title="Filtered syllables: post median > max(Early, Late pre medians)",
                    filename_suffix="_filtered_median_gt_pre",
                    legend_mode="animal_syllables",
                    y_col="Median_ms",
                    yerr_col=None,
                    y_label="Median phrase duration (ms)",
                )

                _plot_one_by_category(
                    df_plot=df_filtered_med,
                    title=(
                        "Filtered syllables: post median > max(Early, Late pre medians)\n"
                        "colored by Medial Area X hit?"
                    ),
                    filename_suffix="_filtered_median_gt_pre_MhitYN",
                    y_col="Median_ms",
                    yerr_col=None,
                    y_label="Median phrase duration (ms)",
                    category_map=medial_hit_cat_map,
                    legend_title="Medial Area X hit?",
                    category_colors=yN_colors,
                    category_labels=yN_labels,
                    category_order=yN_order,
                )

                _plot_one_by_category(
                    df_plot=df_filtered_med,
                    title=(
                        "Filtered syllables: post median > max(Early, Late pre medians)\n"
                        "colored by Medial Area X hit type"
                    ),
                    filename_suffix="_filtered_median_gt_pre_Mtype",
                    y_col="Median_ms",
                    yerr_col=None,
                    y_label="Median phrase duration (ms)",
                    category_map=medial_type_cat_map,
                    legend_title="Medial Area X hit type",
                    category_colors=type_colors,
                    category_labels=medial_type_labels,
                    category_order=type_order,
                )

                _plot_one_by_category(
                    df_plot=df_filtered_med,
                    title=(
                        "Filtered syllables: post median > max(Early, Late pre medians)\n"
                        "colored by Lateral Area X hit type"
                    ),
                    filename_suffix="_filtered_median_gt_pre_Ltype",
                    y_col="Median_ms",
                    yerr_col=None,
                    y_label="Median phrase duration (ms)",
                    category_map=lateral_type_cat_map,
                    legend_title="Lateral Area X hit type",
                    category_colors=type_colors,
                    category_labels=lateral_type_labels,
                    category_order=type_order,
                )
            else:
                print("[INFO] No syllables met the median > pre criterion.")
        else:
            print(
                "[INFO] Missing pre or post rows when computing median filter; "
                "skipping median-based filtered plot."
            )
    else:
        print(
            "[INFO] Column 'Median_ms' not found; skipping median-based filtered plot."
        )

    # ------------------------------------------------------------------
    # 4) Filter by VARIANCE: post variance > pre pooled variance
    # ------------------------------------------------------------------
    var_metric_cols = {"Pre_Variance_ms2", "Variance_ms2"}
    if var_metric_cols.issubset(df_base.columns):
        df_post_var = df_base[post_mask & df_base["Pre_Variance_ms2"].notna()]
        interesting_var = df_post_var[
            df_post_var["Variance_ms2"] > df_post_var["Pre_Variance_ms2"]
        ]

        if not interesting_var.empty:
            keys_var = interesting_var[key_cols].drop_duplicates()
            df_filtered_var = df_base.merge(keys_var, on=key_cols, how="inner")

            _plot_one_combined(
                df_plot=df_filtered_var,
                title="Filtered syllables: post variance > pre pooled variance",
                filename_suffix="_filtered_variance_gt_pre",
                legend_mode="animal_syllables",
                y_col="Variance_ms2",
                yerr_col=None,
                y_label="Phrase duration variance (ms$^2$)",
            )

            _plot_one_by_category(
                df_plot=df_filtered_var,
                title=(
                    "Filtered syllables: post variance > pre pooled variance\n"
                    "colored by Medial Area X hit?"
                ),
                filename_suffix="_filtered_variance_gt_pre_MhitYN",
                y_col="Variance_ms2",
                yerr_col=None,
                y_label="Phrase duration variance (ms$^2$)",
                category_map=medial_hit_cat_map,
                legend_title="Medial Area X hit?",
                category_colors=yN_colors,
                category_labels=yN_labels,
                category_order=yN_order,
            )

            _plot_one_by_category(
                df_plot=df_filtered_var,
                title=(
                    "Filtered syllables: post variance > pre pooled variance\n"
                    "colored by Medial Area X hit type"
                ),
                filename_suffix="_filtered_variance_gt_pre_Mtype",
                y_col="Variance_ms2",
                yerr_col=None,
                y_label="Phrase duration variance (ms$^2$)",
                category_map=medial_type_cat_map,
                legend_title="Medial Area X hit type",
                category_colors=type_colors,
                category_labels=medial_type_labels,
                category_order=type_order,
            )

            _plot_one_by_category(
                df_plot=df_filtered_var,
                title=(
                    "Filtered syllables: post variance > pre pooled variance\n"
                    "colored by Lateral Area X hit type"
                ),
                filename_suffix="_filtered_variance_gt_pre_Ltype",
                y_col="Variance_ms2",
                yerr_col=None,
                y_label="Phrase duration variance (ms$^2$)",
                category_map=lateral_type_cat_map,
                legend_title="Lateral Area X hit type",
                category_colors=type_colors,
                category_labels=lateral_type_labels,
                category_order=type_order,
            )
        else:
            print("[INFO] No syllables met the variance > pre criterion.")
    else:
        print(
            "[INFO] Columns 'Pre_Variance_ms2' / 'Variance_ms2' not found; "
            "skipping variance-based filtered plot."
        )



"""
Example: plotting from a compiled CSV
-------------------------------------

from pathlib import Path
import importlib
import phrase_and_metadata_plotting as pmp
importlib.reload(pmp)

compiled_csv = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/compiled_phrase_duration_stats_with_prepost_metrics.csv")
excel_path = Path("/Users/mirandahulsey-vincent/Desktop/Area_X_lesion_metadata.xlsx")

pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",              # or omit; inferred from .csv suffix
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    mean_col="Mean_ms",
    sem_col="SEM_ms",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots",
    file_prefix="AreaX_phrase_durations",
    show_plots=True,
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    medial_hit_col="Medial Area X hit?",
)

"""
