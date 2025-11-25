# phrase_and_metadata_plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Try to import your existing Excel organizer (Area X–specific)
try:
    from organize_metadata_excel import build_areax_metadata
except Exception:
    build_areax_metadata = None


"""
Plotting utilities for compiled phrase-duration stats.

This module no longer computes the big concatenated DataFrame (`big_df`)
itself. You must generate a compiled phrase-duration stats file
(CSV/JSON/NPZ) ahead of time, for example:

  - Using `phrase_duration_balanced_syllable_usage.run_balanced_syllable_usage_from_metadata_excel`
    to produce a `usage_balanced_phrase_duration_stats.csv`, or
  - Using your previous pipeline that builds
    `compiled_phrase_duration_stats_with_prepost_metrics.csv`.

Then call `plot_compiled_phrase_stats_by_syllable(...)` on that compiled file.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Plotting: compiled phrase-duration stats (thresholded + N_phrases filtering)
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
    median_col: str = "Median_ms",
    var_col: str = "Variance_ms2",
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: str = "phrase_duration",
    show_plots: bool = False,
    # --- coloring style for line plots & category-based scatters ---
    color_by: str = "animal",  # "animal" or "metadata"
    metadata_excel_path: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Union[int, str] = 0,
    metadata_color_col: Optional[str] = None,
    category_color_map: Optional[Dict[str, str]] = None,
    # --- lesion percentage info (from metadata + *_final_volumes.json) ---
    metadata_volumes_dir: Optional[Union[str, Path]] = None,
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    # If set to "left", "right", or "avg", then for animals whose metadata
    # category is "bilateral", "unilateral_L", or "unilateral_R", their color
    # is a *shade* within the usual color family (Reds / Blues / Purples)
    # proportional to the % Area X lesioned (0–100). Sham / miss / large-lesion /
    # unknown still use fixed colors from category_color_map.
    lesion_color_mode: Optional[str] = None,
    # --- filtering thresholds (k·SD or k·IQR style) ---
    mean_sd_k: float = 1.0,
    median_iqr_k: float = 1.0,
    variance_iqr_k: float = 1.0,
    # --- minimum sample size per (animal, syllable) across groups ---
    min_phrases: int = 0,
    n_phrases_col: Optional[str] = "N_phrases",
) -> pd.DataFrame:
    """
    Plot per-syllable phrase-duration metrics across Early Pre / Late Pre / Post
    for all birds, using a compiled stats DataFrame.

    You must generate the compiled stats file beforehand (e.g. using your
    "balanced by song" or "balanced by syllable usage" pipeline) so that each
    row corresponds to (Animal ID, Group, Syllable).

    For each metric (mean/median/variance), this function produces:

      • Line plots vs group for:
          - All syllables
          - Threshold-filtered syllables:
                mean:      Post mean > Pre mean + k·SD
                median:    Post median > Pre median + k·IQR
                variance:  Post variance > Pre variance + k·IQR
          - N_phrases-filtered syllables:
                min N_phrases across groups ≥ min_phrases

      • Late Pre vs Post scatter plots for:
          - All syllables (one scatter per metric)
          - N_phrases-filtered syllables (if min_phrases > 0 and an
            N_phrases column is available)

      • Additional variance scatters with zoomed x-axis for:
          - All syllables
          - N_phrases-filtered syllables (if available)

    If `color_by="metadata"` and you provide both `metadata_excel_path`
    and `metadata_color_col`, then the color legend is based on that
    metadata category (e.g. "Medial Area X hit type", "Lateral Area X hit type",
    "total_inj_volume", etc.), using `category_color_map` for explicit colors
    (with auto-colors for any remaining categories).

    If in addition you provide:

        - `metadata_volumes_dir` (directory with *_final_volumes.json),
        - per-animal lesion-percentage keys `left_lesion_pct_col`,
          `right_lesion_pct_col`,
        - and set `lesion_color_mode` to "left", "right", or "avg",

    then for the categories "bilateral", "unilateral_L", and "unilateral_R",
    the color for each animal is modulated by the % Area X lesioned
    (in that hemisphere or the average of both), using:

        bilateral     → Reds colormap
        unilateral_L  → Blues colormap
        unilateral_R  → Purples colormap

    while the other categories (e.g. sham saline injection, miss,
    large lesion Area X not visible, unknown) keep their fixed colors
    from `category_color_map`.

    On runs where `lesion_color_mode` is active, plot titles include the
    note: "bilateral/unilateral colored by % lesion".
    """
    # ------------------------------------------------------------------
    # 0. Load / validate DataFrame
    # ------------------------------------------------------------------
    if phrase_stats is not None:
        df = phrase_stats.copy()
        compiled_stats_path = None  # not used for I/O here
    else:
        if compiled_stats_path is None:
            raise ValueError(
                "Either `phrase_stats` or `compiled_stats_path` must be provided."
            )
        compiled_stats_path = Path(compiled_stats_path)

        if compiled_format is None:
            suf = compiled_stats_path.suffix.lower()
            if suf == ".csv":
                compiled_format = "csv"
            elif suf == ".json":
                compiled_format = "json"
            elif suf == ".npz":
                compiled_format = "npz"
            else:
                raise ValueError(
                    f"Cannot infer compiled_format from suffix {compiled_stats_path.suffix!r}."
                )

        if compiled_format == "csv":
            df = pd.read_csv(compiled_stats_path)
        elif compiled_format == "json":
            df = pd.read_json(compiled_stats_path)
        elif compiled_format == "npz":
            arr = np.load(compiled_stats_path, allow_pickle=True)
            if "phrase_stats" not in arr:
                raise KeyError("NPZ must contain an array named 'phrase_stats'.")
            df = pd.DataFrame(arr["phrase_stats"])
        else:
            raise ValueError(f"Unsupported compiled_format={compiled_format!r}")

    required_cols = {id_col, group_col, syllable_col, mean_col, median_col, var_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Compiled stats DataFrame is missing columns: {missing}")

    # If we don't have SEM for mean, that's OK (we'll just skip errorbars).
    have_sem = sem_col in df.columns

    # ------------------------------------------------------------------
    # 0b. Build N_phrases filter keys (but do NOT globally filter df)
    # ------------------------------------------------------------------
    keys_min_phrases: Optional[set[Tuple[str, Any]]] = None
    if min_phrases is not None and min_phrases > 0:
        # Determine which N_phrases column to use
        n_col: Optional[str] = None
        candidate_cols: list[str] = []

        if n_phrases_col is not None:
            candidate_cols.append(n_phrases_col)
        # Also consider common fallbacks
        for c in ["N_phrases", "n_phrases"]:
            if c not in candidate_cols:
                candidate_cols.append(c)

        for c in candidate_cols:
            if c in df.columns:
                n_col = c
                break

        if n_col is not None:
            # Compute min N across groups for each (Animal, Syllable)
            grp = df.groupby([id_col, syllable_col])[n_col].min()
            keys_min_phrases = set(
                (str(a), s) for (a, s), val in grp.items() if val >= min_phrases
            )
            if not keys_min_phrases:
                print(
                    f"[INFO] No (Animal, Syllable) pairs with {n_col} >= {min_phrases}. "
                    "N_phrases-filtered plots will be empty."
                )
        else:
            print(
                "[INFO] `min_phrases` specified but no N_phrases/n_phrases column "
                "found; skipping N_phrases-based filtering."
            )

    have_nphrases_filter = (
        min_phrases is not None and min_phrases > 0 and keys_min_phrases is not None
    )

    # ------------------------------------------------------------------
    # 1. Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        if compiled_stats_path is None:
            output_dir = Path(".")
        else:
            output_dir = compiled_stats_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Build metadata dict (for categories and lesion percentages)
    # ------------------------------------------------------------------
    animal_ids = sorted(df[id_col].dropna().unique().tolist())

    meta_dict: Optional[Dict[str, Dict[str, Any]]] = None
    if metadata_excel_path is not None and build_areax_metadata is not None:
        try:
            # Newer signature with volumes_dir
            meta_dict = build_areax_metadata(
                metadata_excel_path,
                sheet_name=metadata_sheet_name,
                volumes_dir=metadata_volumes_dir,
            )
        except TypeError:
            # Fallback to older signature if volumes_dir is not accepted
            meta_dict = build_areax_metadata(
                metadata_excel_path,
                sheet_name=metadata_sheet_name,
            )

    # Default legend title
    legend_title = "Animal ID"

    # id_to_category maps each animal to a category string used for color & legend
    id_to_category: Dict[str, str] = {str(a): str(a) for a in animal_ids}

    # category_to_color maps category → color
    category_to_color: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 2a. Lesion-percentage maps (used only if lesion_color_mode is set)
    # ------------------------------------------------------------------
    id_to_left_pct: Dict[str, float] = {}
    id_to_right_pct: Dict[str, float] = {}
    id_to_avg_pct: Dict[str, float] = {}
    have_lesion_pct = False

    if meta_dict is not None:
        for a in animal_ids:
            a_str = str(a)
            entry = meta_dict.get(a_str, {})

            def _to_float(x: Any) -> float:
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            l_raw = entry.get(left_lesion_pct_col, None)
            r_raw = entry.get(right_lesion_pct_col, None)
            l = _to_float(l_raw)
            r = _to_float(r_raw)

            if np.isfinite(l) and np.isfinite(r):
                avg = 0.5 * (l + r)
            elif np.isfinite(l):
                avg = l
            elif np.isfinite(r):
                avg = r
            else:
                avg = float("nan")

            if np.isfinite(l) or np.isfinite(r):
                have_lesion_pct = True

            id_to_left_pct[a_str] = l
            id_to_right_pct[a_str] = r
            id_to_avg_pct[a_str] = avg

    # ------------------------------------------------------------------
    # 2b. Build category / color mapping (animal-based or metadata-based)
    # ------------------------------------------------------------------
    if color_by.lower() == "metadata" and build_areax_metadata is None:
        print(
            "[WARN] color_by='metadata' requested, but "
            "organize_metadata_excel.build_areax_metadata is not importable. "
            "Falling back to color_by='animal'."
        )
        color_by = "animal"

    if color_by.lower() == "animal":
        # One color per animal
        legend_title = "Animal ID"
        cmap = plt.get_cmap("tab20")
        for idx, a in enumerate(animal_ids):
            category = str(a)
            id_to_category[str(a)] = category
            category_to_color[category] = cmap(idx % cmap.N)

    elif color_by.lower() == "metadata":
        if meta_dict is None or metadata_color_col is None:
            print(
                "[WARN] color_by='metadata' requested but metadata not available "
                "or `metadata_color_col` not provided. Falling back to color_by='animal'."
            )
            color_by = "animal"
            # Recurse with animal coloring and same other arguments
            return plot_compiled_phrase_stats_by_syllable(
                df,
                compiled_stats_path=None,
                compiled_format=None,
                id_col=id_col,
                group_col=group_col,
                syllable_col=syllable_col,
                mean_col=mean_col,
                sem_col=sem_col,
                median_col=median_col,
                var_col=var_col,
                output_dir=output_dir,
                file_prefix=file_prefix,
                show_plots=show_plots,
                color_by="animal",
                metadata_excel_path=metadata_excel_path,
                metadata_sheet_name=metadata_sheet_name,
                metadata_color_col=None,
                category_color_map=category_color_map,
                metadata_volumes_dir=metadata_volumes_dir,
                left_lesion_pct_col=left_lesion_pct_col,
                right_lesion_pct_col=right_lesion_pct_col,
                lesion_color_mode=lesion_color_mode,
                mean_sd_k=mean_sd_k,
                median_iqr_k=median_iqr_k,
                variance_iqr_k=variance_iqr_k,
                min_phrases=min_phrases,
                n_phrases_col=n_phrases_col,
            )

        # Map each animal to the requested metadata category
        categories = set()
        for a in animal_ids:
            a_str = str(a)
            entry2: Dict[str, Any] = meta_dict.get(a_str, {})
            cat_val = entry2.get(metadata_color_col, "unknown")
            if cat_val is None or str(cat_val).strip() == "":
                cat_val = "unknown"
            cat_str = str(cat_val)
            id_to_category[a_str] = cat_str
            categories.add(cat_str)

        # Legend title becomes the metadata column name, e.g. "Medial Area X hit type"
        legend_title = metadata_color_col or "Metadata category"

        # Seed with user-specified colors (e.g., bilateral, unilateral_L, miss, sham, unknown)
        if category_color_map is not None:
            for cat, col in category_color_map.items():
                category_to_color[str(cat)] = col

        # For any remaining categories, assign automatic colors
        cmap = plt.get_cmap("tab20")
        for idx, cat in enumerate(sorted(categories)):
            if cat not in category_to_color:
                category_to_color[cat] = cmap(idx % cmap.N)

        # Ensure some sensible defaults
        if "miss" in categories and "miss" not in category_to_color:
            category_to_color["miss"] = "black"
        if "unknown" in categories and "unknown" not in category_to_color:
            category_to_color["unknown"] = "gray"

    else:
        raise ValueError(f"Unsupported color_by={color_by!r}. Use 'animal' or 'metadata'.")

    # ------------------------------------------------------------------
    # 2c. Helper to pick the actual color for each animal
    # ------------------------------------------------------------------
    lesion_color_mode = (
        lesion_color_mode.lower()
        if isinstance(lesion_color_mode, str)
        else None
    )
    if lesion_color_mode not in {None, "left", "right", "avg"}:
        print(
            f"[WARN] lesion_color_mode={lesion_color_mode!r} not in "
            "{None, 'left', 'right', 'avg'}; ignoring."
        )
        lesion_color_mode = None

    # Note that should appear on titles when lesion-based shading is active
    lesion_color_note: Optional[str] = None
    if lesion_color_mode is not None:
        lesion_color_note = "bilateral/unilateral colored by % lesion"

    # Colormaps for bilateral / unilateral_L / unilateral_R
    lesion_cmaps = {
        "bilateral": plt.get_cmap("Reds"),
        "unilateral_L": plt.get_cmap("Blues"),
        "unilateral_R": plt.get_cmap("Purples"),
    }
    lesion_categories = set(lesion_cmaps.keys())

    def _get_animal_color(a_str: str, a_cat: str) -> Any:
        """
        Base color from category_to_color, possibly modulated by lesion %
        for bilateral / unilateral_L / unilateral_R if lesion_color_mode
        is set and we have usable lesion data.
        """
        base_col = category_to_color.get(a_cat, "black")

        if (
            lesion_color_mode is None
            or not have_lesion_pct
            or a_cat not in lesion_categories
        ):
            return base_col

        if lesion_color_mode == "left":
            val = id_to_left_pct.get(a_str, float("nan"))
        elif lesion_color_mode == "right":
            val = id_to_right_pct.get(a_str, float("nan"))
        elif lesion_color_mode == "avg":
            val = id_to_avg_pct.get(a_str, float("nan"))
        else:
            return base_col

        if not np.isfinite(val):
            return base_col

        # Assume val is in percent 0–100; map to [0.2, 0.9] within the colormap
        v_norm = np.clip(val / 100.0, 0.0, 1.0)
        v_norm = 0.2 + 0.7 * v_norm

        cmap2 = lesion_cmaps.get(a_cat)
        if cmap2 is None:
            return base_col

        return cmap2(v_norm)

    # ------------------------------------------------------------------
    # 3. Helper to pretty up axes
    # ------------------------------------------------------------------
    def _pretty_axes(ax: plt.Axes, x_rotation: int = 0) -> None:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        if x_rotation:
            for lab in ax.get_xticklabels():
                lab.set_rotation(x_rotation)
                lab.set_horizontalalignment("right")

    # ------------------------------------------------------------------
    # 4. Threshold filters (using Pre_* columns from compiled CSV)
    # ------------------------------------------------------------------
    PRE_MEAN_COL        = "Pre_Mean_ms"
    PRE_VAR_COL         = "Pre_Variance_ms2"
    PRE_MEDIAN_COL      = "Pre_Median_ms"
    PRE_MEDIAN_IQR_COL  = "Pre_Median_IQR_ms"
    PRE_VAR_IQR_COL     = "Pre_Variance_IQR_ms2"

    have_pre_mean       = PRE_MEAN_COL in df.columns
    have_pre_var        = PRE_VAR_COL in df.columns
    have_pre_median     = PRE_MEDIAN_COL in df.columns
    have_pre_median_iqr = PRE_MEDIAN_IQR_COL in df.columns
    have_pre_var_iqr    = PRE_VAR_IQR_COL in df.columns

    # group labels we expect; we preserve whatever unique ordering exists,
    # but try to put Early Pre / Late Pre / Post in a sensible order
    raw_groups = list(df[group_col].dropna().unique())
    group_order: list[str] = []
    for g in ["Early Pre", "Late Pre", "Post"]:
        if g in raw_groups:
            group_order.append(g)
    # Add any other group labels at the end
    for g in raw_groups:
        if g not in group_order:
            group_order.append(g)

    # --- Mean filter: Post mean > Pre mean + k·SD ---
    if have_pre_mean and have_pre_var:
        post_rows = df[group_col].eq("Post")
        post = df.loc[post_rows, [id_col, syllable_col, mean_col, PRE_MEAN_COL, PRE_VAR_COL]].copy()
        # SD = sqrt(variance)
        post["Pre_SD_ms"] = np.sqrt(post[PRE_VAR_COL].clip(lower=0.0))
        post["Mean_threshold"] = post[PRE_MEAN_COL] + mean_sd_k * post["Pre_SD_ms"]
        mean_mask = post[mean_col] > post["Mean_threshold"]
        keys_mean_filtered = set(
            (str(a), s)
            for a, s in zip(post.loc[mean_mask, id_col], post.loc[mean_mask, syllable_col])
        )
    else:
        print(
            "[INFO] Pre mean or variance columns not found; using simple "
            "Post mean > Pre mean for filtering."
        )
        if have_pre_mean:
            post_rows = df[group_col].eq("Post") & df[PRE_MEAN_COL].notna()
            post = df.loc[post_rows, [id_col, syllable_col, mean_col, PRE_MEAN_COL]].copy()
            mean_mask = post[mean_col] > post[PRE_MEAN_COL]
            keys_mean_filtered = set(
                (str(a), s)
                for a, s in zip(post.loc[mean_mask, id_col], post.loc[mean_mask, syllable_col])
            )
        else:
            keys_mean_filtered = set()

    # --- Median filter: Post median > Pre median + k·IQR_median (if available) ---
    if have_pre_median and have_pre_median_iqr:
        post_rows = df[group_col].eq("Post")
        post = df.loc[post_rows, [id_col, syllable_col, median_col,
                                  PRE_MEDIAN_COL, PRE_MEDIAN_IQR_COL]].copy()
        post["Median_threshold"] = post[PRE_MEDIAN_COL] + median_iqr_k * post[PRE_MEDIAN_IQR_COL]
        median_mask = post[median_col] > post["Median_threshold"]
        keys_median_filtered = set(
            (str(a), s)
            for a, s in zip(post.loc[median_mask, id_col], post.loc[median_mask, syllable_col])
        )
    elif have_pre_median:
        print(
            "[INFO] No Pre_Median_IQR_ms column found; using simple "
            "Post median > Pre median for filtering."
        )
        post_rows = df[group_col].eq("Post")
        post = df.loc[post_rows, [id_col, syllable_col, median_col, PRE_MEDIAN_COL]].copy()
        median_mask = post[median_col] > post[PRE_MEDIAN_COL]
        keys_median_filtered = set(
            (str(a), s)
            for a, s in zip(post.loc[median_mask, id_col], post.loc[median_mask, syllable_col])
        )
    else:
        keys_median_filtered = set()

    # --- Variance filter: Post var > Pre var + k·IQR_var (if available) ---
    if have_pre_var and have_pre_var_iqr:
        post_rows = df[group_col].eq("Post")
        post = df.loc[post_rows, [id_col, syllable_col, var_col,
                                  PRE_VAR_COL, PRE_VAR_IQR_COL]].copy()
        post["Var_threshold"] = post[PRE_VAR_COL] + variance_iqr_k * post[PRE_VAR_IQR_COL]
        var_mask = post[var_col] > post["Var_threshold"]
        keys_var_filtered = set(
            (str(a), s)
            for a, s in zip(post.loc[var_mask, id_col], post.loc[var_mask, syllable_col])
        )
    elif have_pre_var:
        print(
            "[INFO] No Pre_Variance_IQR_ms2 column found; using simple "
            "Post variance > Pre variance for filtering."
        )
        post_rows = df[group_col].eq("Post")
        post = df.loc[post_rows, [id_col, syllable_col, var_col, PRE_VAR_COL]].copy()
        var_mask = post[var_col] > post[PRE_VAR_COL]
        keys_var_filtered = set(
            (str(a), s)
            for a, s in zip(post.loc[var_mask, id_col], post.loc[var_mask, syllable_col])
        )
    else:
        keys_var_filtered = set()

    # ------------------------------------------------------------------
    # 5. Core line-plot helper (group vs metric)
    # ------------------------------------------------------------------
    def _build_full_title(base: str,
                          threshold_text: Optional[str],
                          lesion_note: Optional[str]) -> str:
        """Combine base title, threshold text, and lesion-color note."""
        parts = []
        if threshold_text:
            parts.append(threshold_text)
        if lesion_note:
            parts.append(lesion_note)
        if parts:
            return f"{base}\n(" + "; ".join(parts) + ")"
        return base

    def _plot_metric_lines(
        df_src: pd.DataFrame,
        *,
        title: str,
        filename_suffix: str,
        y_col: str,
        yerr_col: Optional[str],
        y_label: str,
        filter_keys: Optional[set[Tuple[str, Any]]] = None,
        legend_title_override: Optional[str] = None,
        threshold_text: Optional[str] = None,
    ) -> None:
        """Plot lines for one metric (mean / median / variance)."""

        if df_src.empty:
            print(f"[INFO] Source DataFrame empty for {filename_suffix!r}. Skipping.")
            return

        if filter_keys is not None:
            # Keep only rows where (Animal ID, Syllable) is in filter_keys
            mask = [
                (str(a), s) in filter_keys
                for a, s in zip(df_src[id_col], df_src[syllable_col])
            ]
            df_plot = df_src.loc[mask].copy()
        else:
            df_plot = df_src.copy()

        if df_plot.empty:
            print(f"[INFO] No data to plot for {filename_suffix!r}. Skipping.")
            return

        x_positions = np.arange(len(group_order))

        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        cats_present = set()

        # One line per (animal, syllable) pair
        for animal in sorted(df_plot[id_col].unique()):
            a_str = str(animal)
            a_cat = id_to_category.get(a_str, a_str)
            color = _get_animal_color(a_str, a_cat)
            cats_present.add(a_cat)

            sdf = df_plot[df_plot[id_col] == animal]

            for syll in sorted(sdf[syllable_col].unique()):
                sub = sdf[sdf[syllable_col] == syll].copy()
                if sub.empty:
                    continue

                # Preserve group order on x-axis
                ys = []
                yerrs = []
                for g in group_order:
                    row = sub[sub[group_col] == g]
                    if row.empty:
                        ys.append(np.nan)
                        if yerr_col:
                            yerrs.append(np.nan)
                        continue
                    ys.append(float(row.iloc[0][y_col]))
                    if yerr_col:
                        yerrs.append(float(row.iloc[0][yerr_col]))
                ys = np.array(ys, dtype=float)

                if yerr_col:
                    yerrs = np.array(yerrs, dtype=float)
                else:
                    yerrs = None

                if np.all(np.isnan(ys)):
                    continue

                ax.errorbar(
                    x_positions,
                    ys,
                    yerr=yerrs,
                    fmt="-o",
                    lw=1.0,
                    ms=3.0,
                    color=color,
                    alpha=0.7,
                )

        # X-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_order)
        ax.set_ylabel(y_label)

        # Title, optionally with threshold text and lesion-color note
        full_title = _build_full_title(title, threshold_text, lesion_color_note)
        ax.set_title(full_title)

        _pretty_axes(ax)

        # Build a clean legend with one entry per *category*.
        # (Note: with lesion-based shading, the legend still shows the base
        # family color for that category.)
        legend_cats = sorted(cats_present)
        legend_handles = [
            mlines.Line2D([], [], color=category_to_color.get(cat, "black"),
                          marker="o", linestyle="-", label=cat)
            for cat in legend_cats
        ]

        if legend_handles:
            lt = legend_title_override if legend_title_override is not None else legend_title
            ax.legend(
                handles=legend_handles,
                title=lt,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fname = f"{file_prefix}{filename_suffix}.png"
        outpath = output_dir / fname
        fig.tight_layout()
        fig.savefig(outpath, dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print(f"[PLOT] Saved figure: {outpath}")

    # ------------------------------------------------------------------
    # 6. Late Pre vs Post scatter helper with y=x line (category colors)
    # ------------------------------------------------------------------
    def _plot_latepre_vs_post_scatter(
        df_src: pd.DataFrame,
        *,
        title: str,
        filename_suffix: str,
        metric_col: str,
        x_label: str,
        y_label: str,
        filter_keys: Optional[set[Tuple[str, Any]]] = None,
        legend_title_override: Optional[str] = None,
        threshold_text: Optional[str] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Scatter plot of Late Pre vs Post metric values
        (one point per Animal × Syllable), plus dashed y=x reference line.

        Returns (xlim, ylim) used in the plot.
        """

        late = df_src[df_src[group_col] == "Late Pre"].copy()
        post = df_src[df_src[group_col] == "Post"].copy()

        if late.empty or post.empty:
            print(f"[INFO] Need both 'Late Pre' and 'Post' rows for {filename_suffix!r}. Skipping.")
            return ((0.0, 1.0), (0.0, 1.0))

        merged = pd.merge(
            late[[id_col, syllable_col, metric_col]],
            post[[id_col, syllable_col, metric_col]],
            on=[id_col, syllable_col],
            suffixes=("_LatePre", "_Post"),
            how="inner",
        )

        if filter_keys is not None:
            mask = [
                (str(a), s) in filter_keys
                for a, s in zip(merged[id_col], merged[syllable_col])
            ]
            merged = merged.loc[mask].copy()

        if merged.empty:
            print(f"[INFO] No data after filtering for {filename_suffix!r}. Skipping.")
            return ((0.0, 1.0), (0.0, 1.0))

        # Drop rows with non-finite values
        merged = merged[
            np.isfinite(merged[f"{metric_col}_LatePre"]) &
            np.isfinite(merged[f"{metric_col}_Post"])
        ]

        if merged.empty:
            print(f"[INFO] No finite Late Pre/Post pairs for {filename_suffix!r}. Skipping.")
            return ((0.0, 1.0), (0.0, 1.0))

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        cats_present = set()

        for animal in sorted(merged[id_col].unique()):
            a_str = str(animal)
            a_cat = id_to_category.get(a_str, a_str)
            color = _get_animal_color(a_str, a_cat)
            cats_present.add(a_cat)

            sub = merged[merged[id_col] == animal]
            x_vals = sub[f"{metric_col}_LatePre"].values.astype(float)
            y_vals = sub[f"{metric_col}_Post"].values.astype(float)

            ax.scatter(
                x_vals,
                y_vals,
                s=25.0,
                alpha=0.7,
                color=color,
                edgecolors="none",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Title + optional threshold text + lesion-color note
        full_title = _build_full_title(title, threshold_text, lesion_color_note)
        ax.set_title(full_title)

        # y = x dashed red line
        x_all = merged[f"{metric_col}_LatePre"].values.astype(float)
        y_all = merged[f"{metric_col}_Post"].values.astype(float)
        xmin = np.nanmin(x_all)
        xmax = np.nanmax(x_all)
        ymin = np.nanmin(y_all)
        ymax = np.nanmax(y_all)

        lower = min(xmin, ymin)
        upper = max(xmax, ymax)
        if not np.isfinite(lower) or not np.isfinite(upper):
            lower, upper = 0.0, 1.0
        pad = 0.05 * (upper - lower) if upper > lower else 0.1

        line_min = max(0.0, lower - pad)  # keep origin visible for durations
        line_max = upper + pad

        ax.plot(
            [line_min, line_max],
            [line_min, line_max],
            linestyle="--",
            color="red",
            linewidth=1.5,
            label="y=x",
        )

        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)

        _pretty_axes(ax)

        # Legend: categories + y=x line
        legend_cats = sorted(cats_present)
        category_handles = [
            mlines.Line2D([], [], color=category_to_color.get(cat, "black"),
                          marker="o", linestyle="none", label=cat)
            for cat in legend_cats
        ]
        line_handle = mlines.Line2D(
            [], [], color="red", linestyle="--", label="y=x"
        )

        handles = category_handles + [line_handle]
        if handles:
            lt = legend_title_override if legend_title_override is not None else legend_title
            ax.legend(
                handles=handles,
                title=lt,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fname = f"{file_prefix}{filename_suffix}.png"
        outpath = output_dir / fname
        fig.tight_layout()
        fig.savefig(outpath, dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print(f"[PLOT] Saved Late Pre vs Post scatter figure: {outpath}")

        return ax.get_xlim(), ax.get_ylim()

    # ------------------------------------------------------------------
    # 6b. Zoomed-variance scatter helper (x from 0 to 1.05×max LatePre; keep y-lims)
    # ------------------------------------------------------------------
    def _plot_latepre_vs_post_scatter_zoom_x_from_zero(
        df_src: pd.DataFrame,
        *,
        title: str,
        filename_suffix: str,
        metric_col: str,
        x_label: str,
        y_label: str,
        filter_keys: Optional[set[Tuple[str, Any]]] = None,
        legend_title_override: Optional[str] = None,
        base_ylim: Optional[Tuple[float, float]] = None,
        threshold_text: Optional[str] = None,
    ) -> None:
        """
        Same as _plot_latepre_vs_post_scatter, but:
          - x-axis runs 0 → 1.05×max(Late Pre metric)
          - y-axis limits are fixed to base_ylim (from the non-zoomed plot).
        """

        late = df_src[df_src[group_col] == "Late Pre"].copy()
        post = df_src[df_src[group_col] == "Post"].copy()

        if late.empty or post.empty:
            print(f"[INFO] Need both 'Late Pre' and 'Post' rows for {filename_suffix!r}. Skipping.")
            return

        merged = pd.merge(
            late[[id_col, syllable_col, metric_col]],
            post[[id_col, syllable_col, metric_col]],
            on=[id_col, syllable_col],
            suffixes=("_LatePre", "_Post"),
            how="inner",
        )

        if filter_keys is not None:
            mask = [
                (str(a), s) in filter_keys
                for a, s in zip(merged[id_col], merged[syllable_col])
            ]
            merged = merged.loc[mask].copy()

        if merged.empty:
            print(f"[INFO] No data after filtering for {filename_suffix!r}. Skipping.")
            return

        merged = merged[
            np.isfinite(merged[f"{metric_col}_LatePre"]) &
            np.isfinite(merged[f"{metric_col}_Post"])
        ]
        if merged.empty:
            print(f"[INFO] No finite Late Pre/Post pairs for {filename_suffix!r}. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        cats_present = set()

        for animal in sorted(merged[id_col].unique()):
            a_str = str(animal)
            a_cat = id_to_category.get(a_str, a_str)
            color = _get_animal_color(a_str, a_cat)
            cats_present.add(a_cat)

            sub = merged[merged[id_col] == animal]
            x_vals = sub[f"{metric_col}_LatePre"].values.astype(float)
            y_vals = sub[f"{metric_col}_Post"].values.astype(float)

            ax.scatter(
                x_vals,
                y_vals,
                s=25.0,
                alpha=0.7,
                color=color,
                edgecolors="none",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        full_title = _build_full_title(title, threshold_text, lesion_color_note)
        ax.set_title(full_title)

        # ranges
        x_all = merged[f"{metric_col}_LatePre"].values.astype(float)
        y_all = merged[f"{metric_col}_Post"].values.astype(float)

        # y-limits from base plot
        if base_ylim is None:
            ymin = np.nanmin(y_all)
            ymax = np.nanmax(y_all)
            lower = min(0.0, ymin)
            upper = max(ymax, 1.0)
            pad = 0.05 * (upper - lower) if upper > lower else 0.1
            y_min_use = max(0.0, lower - pad)
            y_max_use = upper + pad
        else:
            y_min_use, y_max_use = base_ylim

        # x-limits: 0 → 1.05×max LatePre
        xmax_lp = np.nanmax(x_all)
        if not np.isfinite(xmax_lp) or xmax_lp <= 0:
            xmax_lp = y_max_use
        x_min_use = 0.0
        x_max_use = xmax_lp * 1.05

        # y=x line within visible window
        line_end = min(x_max_use, y_max_use)
        ax.plot(
            [0.0, line_end],
            [0.0, line_end],
            linestyle="--",
            color="red",
            linewidth=1.5,
            label="y=x",
        )

        ax.set_xlim(x_min_use, x_max_use)
        ax.set_ylim(y_min_use, y_max_use)

        _pretty_axes(ax)

        legend_cats = sorted(cats_present)
        category_handles = [
            mlines.Line2D([], [], color=category_to_color.get(cat, "black"),
                          marker="o", linestyle="none", label=cat)
            for cat in legend_cats
        ]
        line_handle = mlines.Line2D(
            [], [], color="red", linestyle="--", label="y=x"
        )
        handles = category_handles + [line_handle]
        if handles:
            lt = legend_title_override if legend_title_override is not None else legend_title
            ax.legend(
                handles=handles,
                title=lt,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fname = f"{file_prefix}{filename_suffix}.png"
        outpath = output_dir / fname
        fig.tight_layout()
        fig.savefig(outpath, dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print(f"[PLOT] Saved zoomed Late Pre vs Post scatter figure: {outpath}")

    # ------------------------------------------------------------------
    # 7. Line plots vs group
    # ------------------------------------------------------------------
    legend_title_override = legend_title  # usually "Animal ID" or metadata_color_col

    # --- Text labels for filters ---
    threshold_text_mean = None
    if have_pre_mean and have_pre_var:
        threshold_text_mean = f"Post mean > Pre mean + {mean_sd_k:.2g}×SD"
    elif have_pre_mean:
        threshold_text_mean = "Post mean > Pre mean"

    threshold_text_median = None
    if have_pre_median and have_pre_median_iqr:
        threshold_text_median = f"Post median > Pre median + {median_iqr_k:.2g}×IQR"
    elif have_pre_median:
        threshold_text_median = "Post median > Pre median"

    threshold_text_var = None
    if have_pre_var and have_pre_var_iqr:
        threshold_text_var = f"Post variance > Pre variance + {variance_iqr_k:.2g}×IQR"
    elif have_pre_var:
        threshold_text_var = "Post variance > Pre variance"

    nphr_text = None
    if have_nphrases_filter:
        nphr_text = f"N_phrases ≥ {min_phrases}"

    # 7.1 Mean (all syllables)
    _plot_metric_lines(
        df,
        title="All syllables: mean phrase duration by group",
        filename_suffix="_mean_all",
        y_col=mean_col,
        yerr_col=sem_col if have_sem else None,
        y_label="Phrase duration mean (ms)",
        filter_keys=None,
        legend_title_override=legend_title_override,
    )

    # 7.2 Mean (threshold-filtered: Post mean > Pre mean + k·SD)
    _plot_metric_lines(
        df,
        title="Threshold-filtered syllables: mean phrase duration by group",
        filename_suffix="_mean_filtered_post_gt_pre",
        y_col=mean_col,
        yerr_col=sem_col if have_sem else None,
        y_label="Phrase duration mean (ms)",
        filter_keys=keys_mean_filtered if keys_mean_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_mean,
    )

    # 7.3 Mean (N_phrases-filtered only)
    if have_nphrases_filter:
        _plot_metric_lines(
            df,
            title="N_phrases-filtered syllables: mean phrase duration by group",
            filename_suffix="_mean_filtered_by_nphrases",
            y_col=mean_col,
            yerr_col=sem_col if have_sem else None,
            y_label="Phrase duration mean (ms)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # 7.4 Median (all syllables)
    _plot_metric_lines(
        df,
        title="All syllables: median phrase duration by group",
        filename_suffix="_median_all",
        y_col=median_col,
        yerr_col=None,
        y_label="Phrase duration median (ms)",
        filter_keys=None,
        legend_title_override=legend_title_override,
    )

    # 7.5 Median (threshold-filtered)
    _plot_metric_lines(
        df,
        title="Threshold-filtered syllables: median phrase duration by group",
        filename_suffix="_median_filtered_post_gt_pre",
        y_col=median_col,
        yerr_col=None,
        y_label="Phrase duration median (ms)",
        filter_keys=keys_median_filtered if keys_median_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_median,
    )

    # 7.6 Median (N_phrases-filtered only)
    if have_nphrases_filter:
        _plot_metric_lines(
            df,
            title="N_phrases-filtered syllables: median phrase duration by group",
            filename_suffix="_median_filtered_by_nphrases",
            y_col=median_col,
            yerr_col=None,
            y_label="Phrase duration median (ms)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # 7.7 Variance (all syllables)
    _plot_metric_lines(
        df,
        title="All syllables: variance of phrase duration by group",
        filename_suffix="_variance_all",
        y_col=var_col,
        yerr_col=None,
        y_label="Phrase duration variance (ms$^2$)",
        filter_keys=None,
        legend_title_override=legend_title_override,
    )

    # 7.8 Variance (threshold-filtered)
    _plot_metric_lines(
        df,
        title="Threshold-filtered syllables: variance of phrase duration by group",
        filename_suffix="_variance_filtered_post_gt_pre",
        y_col=var_col,
        yerr_col=None,
        y_label="Phrase duration variance (ms$^2$)",
        filter_keys=keys_var_filtered if keys_var_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_var,
    )

    # 7.9 Variance (N_phrases-filtered only)
    if have_nphrases_filter:
        _plot_metric_lines(
            df,
            title="N_phrases-filtered syllables: variance of phrase duration by group",
            filename_suffix="_variance_filtered_by_nphrases",
            y_col=var_col,
            yerr_col=None,
            y_label="Phrase duration variance (ms$^2$)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # ------------------------------------------------------------------
    # 8. Late Pre vs Post scatter plots
    #    - All syllables (one scatter per metric)
    #    - N_phrases-filtered syllables (if available)
    # ------------------------------------------------------------------
    # 8.1 Mean – all syllables
    _plot_latepre_vs_post_scatter(
        df,
        title="All syllables: Late Pre vs Post mean phrase duration",
        filename_suffix="_mean_LatePre_vs_Post_scatter_all",
        metric_col=mean_col,
        x_label="Late Pre mean (ms)",
        y_label="Post lesion mean (ms)",
        filter_keys=None,
        legend_title_override=legend_title_override,
        threshold_text=None,
    )

    # 8.2 Mean – N_phrases-filtered syllables only
    if have_nphrases_filter:
        _plot_latepre_vs_post_scatter(
            df,
            title="N_phrases-filtered syllables: Late Pre vs Post mean phrase duration",
            filename_suffix="_mean_LatePre_vs_Post_scatter_nphrases",
            metric_col=mean_col,
            x_label="Late Pre mean (ms)",
            y_label="Post lesion mean (ms)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # 8.3 Median – all syllables
    _plot_latepre_vs_post_scatter(
        df,
        title="All syllables: Late Pre vs Post median phrase duration",
        filename_suffix="_median_LatePre_vs_Post_scatter_all",
        metric_col=median_col,
        x_label="Late Pre median (ms)",
        y_label="Post lesion median (ms)",
        filter_keys=None,
        legend_title_override=legend_title_override,
        threshold_text=None,
    )

    # 8.4 Median – N_phrases-filtered syllables only
    if have_nphrases_filter:
        _plot_latepre_vs_post_scatter(
            df,
            title="N_phrases-filtered syllables: Late Pre vs Post median phrase duration",
            filename_suffix="_median_LatePre_vs_Post_scatter_nphrases",
            metric_col=median_col,
            x_label="Late Pre median (ms)",
            y_label="Post lesion median (ms)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # 8.5 Variance – all syllables (original full-range)
    _, var_all_ylim = _plot_latepre_vs_post_scatter(
        df,
        title="All syllables: Late Pre vs Post variance of phrase duration",
        filename_suffix="_variance_LatePre_vs_Post_scatter_all",
        metric_col=var_col,
        x_label="Late Pre variance (ms$^2$)",
        y_label="Post lesion variance (ms$^2$)",
        filter_keys=None,
        legend_title_override=legend_title_override,
        threshold_text=None,
    )

    # 8.6 Variance – N_phrases-filtered syllables (original full-range)
    var_nphr_ylim = None
    if have_nphrases_filter:
        _, var_nphr_ylim = _plot_latepre_vs_post_scatter(
            df,
            title="N_phrases-filtered syllables: Late Pre vs Post variance of phrase duration",
            filename_suffix="_variance_LatePre_vs_Post_scatter_nphrases",
            metric_col=var_col,
            x_label="Late Pre variance (ms$^2$)",
            y_label="Post lesion variance (ms$^2$)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            threshold_text=nphr_text,
        )

    # 8.7 Variance – all syllables, zoomed x from 0 → 1.05×max LatePre, keep y-lims
    _plot_latepre_vs_post_scatter_zoom_x_from_zero(
        df,
        title="All syllables: Late Pre vs Post variance (zoomed x)",
        filename_suffix="_variance_LatePre_vs_Post_scatter_all_zoomx",
        metric_col=var_col,
        x_label="Late Pre variance (ms$^2$)",
        y_label="Post lesion variance (ms$^2$)",
        filter_keys=None,
        legend_title_override=legend_title_override,
        base_ylim=var_all_ylim,
        threshold_text=None,
    )

    # 8.8 Variance – N_phrases-filtered syllables, zoomed x, keep y-lims
    if have_nphrases_filter and var_nphr_ylim is not None:
        _plot_latepre_vs_post_scatter_zoom_x_from_zero(
            df,
            title="N_phrases-filtered syllables: Late Pre vs Post variance (zoomed x)",
            filename_suffix="_variance_LatePre_vs_Post_scatter_nphrases_zoomx",
            metric_col=var_col,
            x_label="Late Pre variance (ms$^2$)",
            y_label="Post lesion variance (ms$^2$)",
            filter_keys=keys_min_phrases,
            legend_title_override=legend_title_override,
            base_ylim=var_nphr_ylim,
            threshold_text=nphr_text,
        )

    return df


"""
Example usage (Spyder console)

from pathlib import Path
import importlib
import phrase_and_metadata_plotting as pmp
importlib.reload(pmp)

compiled_csv = Path(
    "/Volumes/my_own_ssd/2024_2025_Area_X_analysis/compiled_phrase_duration_stats_with_prepost_metrics.csv"
)
excel_path = Path("/Volumes/my_own_ssd/2024_2025_Area_X_analysis/Area_X_lesion_metadata.xlsx")
histology_volumes_dir = Path(
    "/Volumes/my_own_ssd/2024_2025_Area_X_analysis/lesion_quantification_csvs_jsons"
)

# 1) Colored by animal
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots",
    file_prefix="AreaX_phrase_durations",
    show_plots=True,
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 2) Colored by **Medial Area X hit type** (categorical only)
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_Medial",
    file_prefix="AreaX_phrase_durations_by_medial_type",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="Medial Area X hit type",
    metadata_volumes_dir=histology_volumes_dir,
    category_color_map={
        "bilateral": "red",
        "large lesion Area x not visible": "darkorange",
        "large lesion, Area X not visible": "darkorange",
        "unilateral_L": "blue",
        "unilateral_R": "purple",
        "sham saline injection": "black",
        "miss": "black",
        "unknown": "yellow",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 3) Colored by **Lateral Area X hit type** (categorical only)
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_Lateral",
    file_prefix="AreaX_phrase_durations_by_lateral_type",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="Lateral Area X hit type",
    metadata_volumes_dir=histology_volumes_dir,
    category_color_map={
        "bilateral": "red",
        "large lesion Area x not visible": "darkorange",
        "large lesion, Area X not visible": "darkorange",
        "unilateral_L": "blue",
        "unilateral_R": "purple",
        "sham saline injection": "black",
        "miss": "black",
        "unknown": "yellow",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 4) Colored by **total injection volume** (nL) as a metadata category
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_total_volume",
    file_prefix="AreaX_phrase_durations_by_total_inj_volume",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="total_inj_volume",
    metadata_volumes_dir=histology_volumes_dir,
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 5) SECOND SET, Medial hit-type plots:
#    bilateral / unilateral_L / unilateral_R shaded by **left hemisphere % lesion**.
#    Titles will include "(bilateral/unilateral colored by % lesion)".
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_Medial_Lpct",
    file_prefix="AreaX_phrase_durations_medial_shaded_by_L_pct",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="Medial Area X hit type",
    metadata_volumes_dir=histology_volumes_dir,
    lesion_color_mode="left",  # use left hemisphere lesion % for shading
    category_color_map={
        "bilateral": "red",     # base family color
        "large lesion Area x not visible": "darkorange",
        "large lesion, Area X not visible": "darkorange",
        "unilateral_L": "blue",
        "unilateral_R": "purple",
        "sham saline injection": "black",
        "miss": "black",
        "unknown": "yellow",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 6) SECOND SET, Lateral hit-type plots:
#    bilateral / unilateral_L / unilateral_R shaded by **right hemisphere % lesion**.
#    Titles will include "(bilateral/unilateral colored by % lesion)".
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_Lateral_Rpct",
    file_prefix="AreaX_phrase_durations_lateral_shaded_by_R_pct",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="Lateral Area X hit type",
    metadata_volumes_dir=histology_volumes_dir,
    lesion_color_mode="right",  # use right hemisphere lesion % for shading
    category_color_map={
        "bilateral": "red",
        "large lesion Area x not visible": "darkorange",
        "large lesion, Area X not visible": "darkorange",
        "unilateral_L": "blue",
        "unilateral_R": "purple",
        "sham saline injection": "black",
        "miss": "black",
        "unknown": "yellow",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)

# 7) OPTIONAL: Medial plots shaded by **average % lesion** (L/R avg).
#    Titles will include "(bilateral/unilateral colored by % lesion)".
pmp.plot_compiled_phrase_stats_by_syllable(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",
    id_col="Animal ID",
    group_col="Group",
    syllable_col="Syllable",
    output_dir=compiled_csv.parent / "phrase_duration_line_plots_Medial_avgPct",
    file_prefix="AreaX_phrase_durations_medial_shaded_by_avg_pct",
    show_plots=True,
    color_by="metadata",
    metadata_excel_path=excel_path,
    metadata_sheet_name=0,
    metadata_color_col="Medial Area X hit type",
    metadata_volumes_dir=histology_volumes_dir,
    lesion_color_mode="avg",  # use average(L,R) lesion % for shading
    category_color_map={
        "bilateral": "red",
        "large lesion Area x not visible": "darkorange",
        "large lesion, Area X not visible": "darkorange",
        "unilateral_L": "blue",
        "unilateral_R": "purple",
        "sham saline injection": "black",
        "miss": "black",
        "unknown": "yellow",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
    min_phrases=50,
)
"""
