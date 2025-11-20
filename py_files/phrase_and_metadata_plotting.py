# phrase_and_metadata_plotting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Tuple, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
except Exception:
    build_areax_metadata = None


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhraseAndMetadataResult:
    """Container for outputs of run_phrase_durations_from_metadata."""

    birds_stats: BirdsPhraseDurationStats
    compiled_stats_df: pd.DataFrame
    compiled_stats_path: Path


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: run phrase-duration analysis from Excel + JSON root
# ──────────────────────────────────────────────────────────────────────────────


def run_phrase_durations_from_metadata(
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",
    restrict_to_labels: Optional[Sequence[str]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = False,
    compiled_filename: str = "compiled_phrase_duration_stats_with_prepost_metrics.csv",
) -> PhraseAndMetadataResult:
    """
    High-level wrapper:

    1. Uses your metadata Excel file + JSON root to build phrase-duration stats
       for all birds via `build_birds_phrase_duration_stats_df`.
    2. Saves the compiled stats DataFrame to a CSV under:
          <json_root>/TweetyBERT_outputs/<compiled_filename>
    3. Returns a small dataclass with the stats and output path.

    This does *not* change the internal logic of how per-bird stats are
    computed; that still lives in `phrase_duration_birds_stats_df.py`.
    """
    excel_path = Path(excel_path)
    json_root = Path(json_root)

    birds_stats = build_birds_phrase_duration_stats_df(
        excel_path=excel_path,
        json_root=json_root,
        sheet_name=sheet_name,
        id_col=id_col,
        treatment_date_col=treatment_date_col,
        grouping_mode=grouping_mode,
        restrict_to_labels=restrict_to_labels,
        y_max_ms=y_max_ms,
        show_plots=show_plots,
    )

    big_df = birds_stats.phrase_duration_stats_df.copy()

    # Where to save the compiled stats
    out_dir = json_root / "TweetyBERT_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = out_dir / compiled_filename

    big_df.to_csv(compiled_path, index=False)
    print(f"Saved big_df with metrics to: {compiled_path}")

    return PhraseAndMetadataResult(
        birds_stats=birds_stats,
        compiled_stats_df=big_df,
        compiled_stats_path=compiled_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotting: compiled phrase-duration stats (thresholded by baseline variability)
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
    # --- coloring style ---
    color_by: str = "animal",  # "animal" or "metadata"
    metadata_excel_path: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Union[int, str] = 0,
    metadata_color_col: Optional[str] = None,
    category_color_map: Optional[Dict[str, str]] = None,
    # --- filtering thresholds (k·SD or k·IQR style) ---
    mean_sd_k: float = 1.0,
    median_iqr_k: float = 1.0,
    variance_iqr_k: float = 1.0,
) -> pd.DataFrame:
    """
    Plot per-syllable phrase-duration metrics across Early Pre / Late Pre / Post
    for all birds, using a compiled stats DataFrame.

    It produces:
      - Line plots of mean/median/variance vs group (all syllables + filtered)
      - Scatter plots of Late Pre vs Post for mean/median/variance
        (one point per Animal×Syllable), with a y=x dashed red line.
    """
    # ------------------------------------------------------------------
    # 0. Load / validate DataFrame
    # ------------------------------------------------------------------
    if phrase_stats is not None:
        df = phrase_stats.copy()
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
    # 2. Build color / category maps
    # ------------------------------------------------------------------
    animal_ids = sorted(df[id_col].dropna().unique().tolist())

    # Default legend title
    legend_title = "Animal ID"

    # id_to_category maps each animal to a category string used for color & legend
    id_to_category: Dict[str, str] = {str(a): str(a) for a in animal_ids}

    # category_to_color maps category → color
    category_to_color: Dict[str, str] = {}

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
        if metadata_excel_path is None or metadata_color_col is None:
            print(
                "[WARN] color_by='metadata' requested but "
                "`metadata_excel_path` or `metadata_color_col` not provided. "
                "Falling back to color_by='animal'."
            )
            color_by = "animal"
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
                mean_sd_k=mean_sd_k,
                median_iqr_k=median_iqr_k,
                variance_iqr_k=variance_iqr_k,
            )

        # Use Area X metadata to get e.g. "Medial Area X hit type" / "Lateral ..."
        meta_dict = build_areax_metadata(metadata_excel_path, sheet_name=metadata_sheet_name)

        # Map each animal to the requested metadata category
        categories = set()
        for a in animal_ids:
            a_str = str(a)
            entry: Dict[str, Any] = meta_dict.get(a_str, {})
            cat_val = entry.get(metadata_color_col, "unknown")
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

        # Ensure "miss" / "unknown" have sensible defaults
        if "miss" in categories and "miss" not in category_to_color:
            category_to_color["miss"] = "black"
        if "unknown" in categories and "unknown" not in category_to_color:
            category_to_color["unknown"] = "gray"

    else:
        raise ValueError(f"Unsupported color_by={color_by!r}. Use 'animal' or 'metadata'.")

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
    group_order = []
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
            color = category_to_color.get(a_cat, "black")
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

        # Title, optionally with threshold text appended
        full_title = title
        if threshold_text:
            full_title = f"{title}\n({threshold_text})"
        ax.set_title(full_title)

        _pretty_axes(ax)

        # Build a clean legend with one entry per category
        legend_cats = sorted(cats_present)
        legend_handles = [
            mlines.Line2D([], [], color=category_to_color.get(cat, "black"), marker="o",
                          linestyle="-", label=cat)
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
    # 6. New helper: Late Pre vs Post scatter plots with y=x line
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
    ) -> None:
        """
        Scatter plot of Late Pre vs Post metric values
        (one point per Animal × Syllable), plus dashed y=x reference line.
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

        # Drop rows with non-finite values
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
            color = category_to_color.get(a_cat, "black")
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

        # Title + optional threshold text
        full_title = title
        if threshold_text:
            full_title = f"{title}\n({threshold_text})"
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

    # ------------------------------------------------------------------
    # 7. Line plots vs group (same as before)
    # ------------------------------------------------------------------
    legend_title_override = legend_title  # usually "Animal ID" or metadata_color_col

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

    # 7.2 Mean (filtered: Post mean > Pre mean + k·SD)
    threshold_text_mean = None
    if have_pre_mean and have_pre_var:
        threshold_text_mean = f"Post mean > Pre mean + {mean_sd_k:.2g}×SD"
    elif have_pre_mean:
        threshold_text_mean = "Post mean > Pre mean"

    _plot_metric_lines(
        df,
        title="Filtered syllables: mean phrase duration by group",
        filename_suffix="_mean_filtered_post_gt_pre",
        y_col=mean_col,
        yerr_col=sem_col if have_sem else None,
        y_label="Phrase duration mean (ms)",
        filter_keys=keys_mean_filtered if keys_mean_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_mean,
    )

    # 7.3 Median (all syllables)
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

    # 7.4 Median (filtered)
    threshold_text_median = None
    if have_pre_median and have_pre_median_iqr:
        threshold_text_median = f"Post median > Pre median + {median_iqr_k:.2g}×IQR"
    elif have_pre_median:
        threshold_text_median = "Post median > Pre median"

    _plot_metric_lines(
        df,
        title="Filtered syllables: median phrase duration by group",
        filename_suffix="_median_filtered_post_gt_pre",
        y_col=median_col,
        yerr_col=None,
        y_label="Phrase duration median (ms)",
        filter_keys=keys_median_filtered if keys_median_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_median,
    )

    # 7.5 Variance (all syllables)
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

    # 7.6 Variance (filtered)
    threshold_text_var = None
    if have_pre_var and have_pre_var_iqr:
        threshold_text_var = f"Post variance > Pre variance + {variance_iqr_k:.2g}×IQR"
    elif have_pre_var:
        threshold_text_var = "Post variance > Pre variance"

    _plot_metric_lines(
        df,
        title="Filtered syllables: variance of phrase duration by group",
        filename_suffix="_variance_filtered_post_gt_pre",
        y_col=var_col,
        yerr_col=None,
        y_label="Phrase duration variance (ms$^2$)",
        filter_keys=keys_var_filtered if keys_var_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_var,
    )

    # ------------------------------------------------------------------
    # 8. Late Pre vs Post scatter plots (filtered syllables)
    # ------------------------------------------------------------------
    # 8.1 Mean
    _plot_latepre_vs_post_scatter(
        df,
        title="Filtered syllables: Late Pre vs Post mean phrase duration",
        filename_suffix="_mean_LatePre_vs_Post_scatter",
        metric_col=mean_col,
        x_label="Late Pre mean (ms)",
        y_label="Post lesion mean (ms)",
        filter_keys=keys_mean_filtered if keys_mean_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_mean,
    )

    # 8.2 Median
    _plot_latepre_vs_post_scatter(
        df,
        title="Filtered syllables: Late Pre vs Post median phrase duration",
        filename_suffix="_median_LatePre_vs_Post_scatter",
        metric_col=median_col,
        x_label="Late Pre median (ms)",
        y_label="Post lesion median (ms)",
        filter_keys=keys_median_filtered if keys_median_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_median,
    )

    # 8.3 Variance
    _plot_latepre_vs_post_scatter(
        df,
        title="Filtered syllables: Late Pre vs Post variance of phrase duration",
        filename_suffix="_variance_LatePre_vs_Post_scatter",
        metric_col=var_col,
        x_label="Late Pre variance (ms$^2$)",
        y_label="Post lesion variance (ms$^2$)",
        filter_keys=keys_var_filtered if keys_var_filtered else None,
        legend_title_override=legend_title_override,
        threshold_text=threshold_text_var,
    )

    return df


"""
from pathlib import Path
import importlib
import phrase_and_metadata_plotting as pmp
importlib.reload(pmp)

compiled_csv = Path(
    "/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/compiled_phrase_duration_stats_with_prepost_metrics.csv"
)
excel_path = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_lesion_metadata.xlsx")

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
)

# 2) Colored by **Medial Area X hit type**
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
    category_color_map={
        "bilateral": "red",
        "unilateral_L": "orange",
        "unilateral_R": "purple",
        "sham": "green",
        "miss": "black",
        "unknown": "gray",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
)

# 3) Colored by **Lateral Area X hit type**
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
    category_color_map={
        "bilateral": "red",
        "unilateral_L": "orange",
        "unilateral_R": "purple",
        "sham": "green",
        "miss": "black",
        "unknown": "gray",
    },
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
)

# 4) Colored by **total injection volume** (nL)
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
    mean_sd_k=1.0,
    median_iqr_k=1.0,
    variance_iqr_k=1.0,
)
"""
