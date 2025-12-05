# phrase_and_metadata_continuous_plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# We rely on your existing Excel organizer
try:
    from organize_metadata_excel import build_areax_metadata
except Exception:
    build_areax_metadata = None


def plot_continuous_hit_type_scatter(
    phrase_stats: Optional[pd.DataFrame] = None,
    *,
    compiled_stats_path: Optional[Union[str, Path]] = None,
    compiled_format: Optional[str] = None,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syllable_col: str = "Syllable",
    mean_col: str = "Mean_ms",
    median_col: str = "Median_ms",
    var_col: str = "Variance_ms2",
    n_phrases_col: str = "N_phrases",
    min_phrases: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: str = "AreaX_phrase_durations_continuous",
    # metadata / histology
    metadata_excel_path: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Union[int, str] = 0,
    metadata_volumes_dir: Optional[Union[str, Path]] = None,
    area_visible_col: str = "Area X visible in histology?",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    lesion_pct_mode: str = "avg",  # 'left', 'right', 'avg'
    # discrete colors for non-visible animals/categories
    discrete_color_map: Optional[Dict[str, str]] = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Generate Late Pre vs Post scatterplots for phrase-duration metrics
    (mean / median / variance) with continuous coloring by % Area X lesioned
    *only* for animals whose Area X is visible in histology.

    For each of:
        - "Medial Area X hit type"
        - "Lateral Area X hit type"

    this function builds three scatterplots (mean / median / variance).
    Points from animals with Area X visible are colored by a continuous
    colormap (Blues/Purples) using the lesion percentage; points from animals
    where Area X was not visible (or lesion percentage is unavailable) are
    plotted in discrete colors by hit-type category (e.g. sham vs
    "large lesion Area x not visible").
    """
    # ------------------------------------------------------------------
    # 0. Load / validate DataFrame
    # ------------------------------------------------------------------
    if phrase_stats is not None:
        df = phrase_stats.copy()
        compiled_stats_path = None
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

    # ------------------------------------------------------------------
    # 1. N_phrases filter keys (optional)
    # ------------------------------------------------------------------
    keys_min_phrases: Optional[set[Tuple[str, Any]]] = None
    if min_phrases is not None and min_phrases > 0 and n_phrases_col in df.columns:
        grp = df.groupby([id_col, syllable_col])[n_phrases_col].min()
        keys_min_phrases = set(
            (str(a), s) for (a, s), val in grp.items() if val >= min_phrases
        )
        if not keys_min_phrases:
            print(
                f"[INFO] No (Animal, Syllable) pairs with {n_phrases_col} >= {min_phrases}. "
                "Plots will be empty."
            )
    elif min_phrases is not None and min_phrases > 0:
        print(
            f"[INFO] min_phrases={min_phrases} requested but column {n_phrases_col!r} "
            "not found; skipping N_phrases-based filtering."
        )

    # ------------------------------------------------------------------
    # 2. Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        if compiled_stats_path is not None:
            output_dir = compiled_stats_path.parent
        else:
            output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load metadata (Area X visibility & lesion percentages)
    # ------------------------------------------------------------------
    if metadata_excel_path is None or build_areax_metadata is None:
        raise RuntimeError(
            "This plotting function requires `metadata_excel_path` and "
            "a working `build_areax_metadata` import from organize_metadata_excel."
        )

    metadata_excel_path = Path(metadata_excel_path)

    try:
        meta_dict: Dict[str, Dict[str, Any]] = build_areax_metadata(
            metadata_excel_path,
            sheet_name=metadata_sheet_name,
            volumes_dir=metadata_volumes_dir,
        )
    except TypeError:
        # Older signature without volumes_dir
        meta_dict = build_areax_metadata(
            metadata_excel_path,
            sheet_name=metadata_sheet_name,
        )

    # Default discrete colors for non-visible / unquantified animals
    if discrete_color_map is None:
        discrete_color_map = {
            # canonical key for any "large lesion ... not visible" variants
            "large lesion Area x not visible": "red",
            "sham saline injection": "gold",
            "miss": "black",
            "unknown": "lightgray",
        }

    # ------------------------------------------------------------------
    # 4. Helpers for metadata
    # ------------------------------------------------------------------
    lesion_pct_mode = (
        lesion_pct_mode.lower()
        if isinstance(lesion_pct_mode, str)
        else "avg"
    )
    if lesion_pct_mode not in {"left", "right", "avg"}:
        print(
            f"[WARN] lesion_pct_mode={lesion_pct_mode!r} not in "
            "{'left','right','avg'}; using 'avg'."
        )
        lesion_pct_mode = "avg"

    def _is_visible(entry: Dict[str, Any]) -> bool:
        """
        Decide whether Area X is visible in histology.

        • Any string starting with 'n' or containing 'not visible' → False
        • Any string starting with 'y' or containing ' visible' (but not 'not visible') → True
        """
        raw = entry.get(area_visible_col, "")
        if raw is None:
            return False
        s = str(raw).strip().lower()

        # Explicit negative cases first
        if "not visible" in s:
            return False
        if s.startswith("n"):
            return False

        # Positive cases
        if s.startswith("y"):
            return True
        if " visible" in s or s == "visible":
            return True

        return False

    def _get_lesion_pct(entry: Dict[str, Any]) -> float:
        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return float("nan")

        l = _to_float(entry.get(left_lesion_pct_col, None))
        r = _to_float(entry.get(right_lesion_pct_col, None))

        if lesion_pct_mode == "left":
            return l
        elif lesion_pct_mode == "right":
            return r
        else:  # "avg"
            vals = [v for v in (l, r) if np.isfinite(v)]
            if not vals:
                return float("nan")
            return float(np.mean(vals))

    def _canonical_hit_type(cat_raw: Any) -> str:
        """
        Normalize hit-type strings so that variations of
        'large lesion Area X not visible' all map to a canonical key,
        and sham/miss/unknown are collapsed sensibly.
        """
        if cat_raw is None:
            return "unknown"
        s = str(cat_raw).strip()
        if s == "":
            return "unknown"

        low = s.lower()

        if "large" in low and "lesion" in low and "visible" in low:
            # Covers e.g. 'large lesion Area x not visible',
            # 'large lesion, Area X not visible', etc.
            return "large lesion Area x not visible"
        if "sham" in low:
            return "sham saline injection"
        if "miss" in low:
            return "miss"

        return s  # fall back to original label

    def _pretty_axes(ax: plt.Axes) -> None:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

    def _build_title(base: str) -> str:
        notes = [
            "Area X visible: colored by % lesion (colorbar)",
            "Area X not visible or % lesion missing: discrete hit-type colors",
        ]
        if min_phrases is not None and min_phrases > 0 and n_phrases_col in df.columns:
            notes.append(f"N_phrases ≥ {min_phrases}")
        return base + "\n(" + "; ".join(notes) + ")"

    # ------------------------------------------------------------------
    # 5. Core scatter helper for one hemisphere & one metric
    # ------------------------------------------------------------------
    def _scatter_for_metric(
        hemi_label: str,
        hemi_hit_type_col: str,
        metric_col: str,
        metric_name: str,
        x_label: str,
        y_label: str,
        cmap_name: str = "Blues",
    ) -> None:
        """
        Build Late Pre vs Post scatter for a given hemisphere and metric.
        """
        # Prepare Late Pre / Post pairs
        late = df[df[group_col] == "Late Pre"][[id_col, syllable_col, metric_col]]
        post = df[df[group_col] == "Post"][[id_col, syllable_col, metric_col]]

        if late.empty or post.empty:
            print(
                f"[INFO] Need both 'Late Pre' and 'Post' rows for {metric_name} "
                f"({hemi_label}). Skipping."
            )
            return

        merged = pd.merge(
            late,
            post,
            on=[id_col, syllable_col],
            suffixes=("_LatePre", "_Post"),
            how="inner",
        )

        if merged.empty:
            print(
                f"[INFO] No overlapping Late Pre/Post rows for {metric_name} "
                f"({hemi_label}). Skipping."
            )
            return

        # Optional N_phrases filter
        if keys_min_phrases is not None:
            mask = [
                (str(a), s) in keys_min_phrases
                for a, s in zip(merged[id_col], merged[syllable_col])
            ]
            merged = merged.loc[mask].copy()

        if merged.empty:
            print(
                f"[INFO] No data after N_phrases filtering for {metric_name} "
                f"({hemi_label}). Skipping."
            )
            return

        # Drop non-finite metric pairs
        x_all = merged[f"{metric_col}_LatePre"].values.astype(float)
        y_all = merged[f"{metric_col}_Post"].values.astype(float)
        finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
        merged = merged.loc[finite_mask].copy()

        if merged.empty:
            print(
                f"[INFO] No finite Late Pre/Post pairs for {metric_name} "
                f"({hemi_label}). Skipping."
            )
            return

        # Split into continuous (visible & quantified) vs discrete (non-visible or NaN)
        x_vis: list[float] = []
        y_vis: list[float] = []
        c_vis: list[float] = []
        discrete_points: Dict[str, Tuple[list[float], list[float]]] = {}

        for _, row in merged.iterrows():
            a_str = str(row[id_col])
            entry = meta_dict.get(a_str, {})

            x_val = float(row[f"{metric_col}_LatePre"])
            y_val = float(row[f"{metric_col}_Post"])

            visible = _is_visible(entry)
            pct = _get_lesion_pct(entry)

            if visible and np.isfinite(pct):
                x_vis.append(x_val)
                y_vis.append(y_val)
                c_vis.append(pct)
            else:
                raw_cat = entry.get(hemi_hit_type_col, "unknown")
                cat = _canonical_hit_type(raw_cat)
                if cat not in discrete_points:
                    discrete_points[cat] = ([], [])
                discrete_points[cat][0].append(x_val)
                discrete_points[cat][1].append(y_val)

        if not x_vis and not discrete_points:
            print(
                f"[INFO] No data to plot for {metric_name} ({hemi_label}) "
                "(no visible or discrete points). Skipping."
            )
            return

        fig, ax = plt.subplots(figsize=(7.0, 6.0))

        # 1) Continuous points (Area X visible with lesion %)
        sc = None
        if x_vis:
            sc = ax.scatter(
                x_vis,
                y_vis,
                c=c_vis,
                cmap=cmap_name,
                vmin=0.0,
                vmax=100.0,
                s=28.0,
                alpha=0.85,
                edgecolors="none",
            )

        # 2) Discrete points
        legend_handles: list[Any] = []

        # Handle for continuous points (legend) – uses a clear marker
        if sc is not None:
            handle_cont = mlines.Line2D(
                [], [],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor="blue",
                markeredgewidth=1.2,
                label="Area X visible (see colorbar)",
            )
            legend_handles.append(handle_cont)

        for cat, (xs, ys) in discrete_points.items():
            color = discrete_color_map.get(cat, "gray")
            ax.scatter(
                xs,
                ys,
                s=32.0,
                alpha=0.9,
                color=color,
                edgecolors="black",
                linewidths=0.4,
            )
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="none",
                    markersize=9,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    label=cat,
                )
            )

        # Axes labels / title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        base_title = f"{hemi_label}: Late Pre vs Post {metric_name} phrase duration"
        ax.set_title(_build_title(base_title))

        # y=x reference line and limits
        all_x_arrays = []
        all_y_arrays = []

        if x_vis:
            all_x_arrays.append(np.array(x_vis))
            all_y_arrays.append(np.array(y_vis))
        for xs, ys in discrete_points.values():
            all_x_arrays.append(np.array(xs))
            all_y_arrays.append(np.array(ys))

        x_all = np.concatenate(all_x_arrays)
        y_all = np.concatenate(all_y_arrays)

        xmin = np.nanmin(x_all)
        xmax = np.nanmax(x_all)
        ymin = np.nanmin(y_all)
        ymax = np.nanmax(y_all)

        lower = min(xmin, ymin)
        upper = max(xmax, ymax)
        if not np.isfinite(lower) or not np.isfinite(upper):
            lower, upper = 0.0, 1.0
        pad = 0.05 * (upper - lower) if upper > lower else 0.1

        line_min = max(0.0, lower - pad)
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

        # Legend (add y=x handle)
        line_handle = mlines.Line2D(
            [], [], color="red", linestyle="--", label="y=x"
        )
        legend_handles.append(line_handle)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                borderaxespad=0.0,
                frameon=True,
                facecolor="white",
                framealpha=0.85,
                fontsize=10,
            )

        # Colorbar for continuous points (if any)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("% of Area X lesioned", fontsize=11)

        fname = f"{file_prefix}_{hemi_label.replace(' ', '')}_{metric_name}_LatePre_vs_Post_continuous.png"
        outpath = output_dir / fname
        fig.tight_layout()
        fig.savefig(outpath, dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print(f"[PLOT] Saved continuous scatter: {outpath}")

    # ------------------------------------------------------------------
    # 6. Generate plots for Medial and Lateral hit types
    # ------------------------------------------------------------------
    metrics = [
        ("mean", mean_col, "Late Pre mean (ms)", "Post lesion mean (ms)"),
        ("median", median_col, "Late Pre median (ms)", "Post lesion median (ms)"),
        ("variance", var_col, "Late Pre variance (ms$^2$)", "Post lesion variance (ms$^2$)"),
    ]

    medial_label = "Medial Area X hit type"
    lateral_label = "Lateral Area X hit type"

    for (metric_name, col, xlab, ylab) in metrics:
        _scatter_for_metric(
            hemi_label="Medial Area X",
            hemi_hit_type_col=medial_label,
            metric_col=col,
            metric_name=metric_name,
            x_label=xlab,
            y_label=ylab,
            cmap_name="Blues",
        )
        _scatter_for_metric(
            hemi_label="Lateral Area X",
            hemi_hit_type_col=lateral_label,
            metric_col=col,
            metric_name=metric_name,
            x_label=xlab,
            y_label=ylab,
            cmap_name="Purples",
        )

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate Late Pre vs Post scatterplots for phrase-duration metrics "
            "with continuous % lesion coloring for Area X visible animals and "
            "discrete colors for non-visible / unquantified cases."
        )
    )
    parser.add_argument(
        "compiled_stats_path",
        type=str,
        help="Path to compiled phrase-duration stats file (CSV / JSON / NPZ).",
    )
    parser.add_argument(
        "metadata_excel_path",
        type=str,
        help="Path to Area X lesion metadata Excel file.",
    )
    parser.add_argument(
        "--metadata_volumes_dir",
        type=str,
        default=None,
        help="Directory with *_final_volumes.json lesion volume files (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: parent of compiled_stats_path).",
    )
    parser.add_argument(
        "--min_phrases",
        type=int,
        default=50,
        help="Minimum N_phrases across groups for an (animal, syllable) to be plotted.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, display plots interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--lesion_pct_mode",
        type=str,
        choices=["left", "right", "avg"],
        default="avg",
        help="Which lesion percentage to use for continuous coloring.",
    )

    args = parser.parse_args()

    compiled_path = Path(args.compiled_stats_path)
    excel_path = Path(args.metadata_excel_path)
    out_dir = Path(args.output_dir) if args.output_dir is not None else None
    volumes_dir = (
        Path(args.metadata_volumes_dir)
        if args.metadata_volumes_dir is not None
        else None
    )

    plot_continuous_hit_type_scatter(
        compiled_stats_path=compiled_path,
        compiled_format=None,
        metadata_excel_path=excel_path,
        metadata_volumes_dir=volumes_dir,
        output_dir=out_dir,
        min_phrases=args.min_phrases,
        show_plots=args.show_plots,
        lesion_pct_mode=args.lesion_pct_mode,
    )

"""
from pathlib import Path
import importlib
import phrase_and_metadata_continuous_plotting as pmcp
importlib.reload(pmcp)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
excel_path   = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")
histology_volumes_dir      = Path("/Volumes/my_own_SSD/histology_files/lesion_quantification_csvs_jsons")

pmcp.plot_continuous_hit_type_scatter(
    compiled_stats_path=compiled_csv,
    compiled_format="csv",                    # file is a CSV
    metadata_excel_path=excel_path,
    metadata_volumes_dir=histology_volumes_dir,
    output_dir=compiled_csv.parent / "phrase_duration_continuous_scatter",
    file_prefix="AreaX_continuous",          # prefix for all PNGs
    min_phrases=50,                          # only plot syllables with ≥50 phrases
    lesion_pct_mode="avg",                   # color by average(L,R) % lesion
    show_plots=True,                         # also pop up the figures in Spyder
)


"""