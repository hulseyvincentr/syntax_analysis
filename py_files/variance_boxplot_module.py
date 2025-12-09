from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from organize_metadata_excel import build_areax_metadata
except Exception:
    build_areax_metadata = None


def plot_variance_boxplots(
    phrase_stats: Optional[pd.DataFrame] = None,
    *,
    compiled_stats_path: Optional[Union[str, Path]] = None,
    compiled_format: Optional[str] = None,
    id_col: str = "Animal ID",
    group_col: str = "Group",
    syllable_col: str = "Syllable",
    var_col: str = "Variance_ms2",
    n_phrases_col: str = "N_phrases",
    min_phrases: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: str = "variance_boxplot",
    # metadata / histology
    metadata_excel_path: Optional[Union[str, Path]] = None,
    metadata_sheet_name: Union[int, str] = 0,
    metadata_volumes_dir: Optional[Union[str, Path]] = None,
    medial_hit_type_col: str = "Medial Area X hit type",
    lateral_hit_type_col: str = "Lateral Area X hit type",
    show_plots: bool = False,
) -> None:
    """
    Generate box plots comparing log(variance) between Late Pre and Post
    for two groups: 1) NMA lesion, 2) sham saline injection.
    
    Creates separate plots for Medial and Lateral Area X.
    """
    # ------------------------------------------------------------------
    # 1. Load / validate DataFrame
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

    required_cols = {id_col, group_col, syllable_col, var_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Compiled stats DataFrame is missing columns: {missing}")

    # ------------------------------------------------------------------
    # 2. N_phrases filter
    # ------------------------------------------------------------------
    if min_phrases > 0 and n_phrases_col in df.columns:
        grp = df.groupby([id_col, syllable_col])[n_phrases_col].min()
        keys_min = set(
            (str(a), s) for (a, s), val in grp.items() if val >= min_phrases
        )
        if keys_min:
            mask = [
                (str(row[id_col]), row[syllable_col]) in keys_min
                for _, row in df.iterrows()
            ]
            df = df.loc[mask].copy()

    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        if compiled_stats_path is not None:
            output_dir = compiled_stats_path.parent
        else:
            output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Load metadata
    # ------------------------------------------------------------------
    if metadata_excel_path is None or build_areax_metadata is None:
        raise RuntimeError(
            "This plotting function requires `metadata_excel_path` and "
            "a working `build_areax_metadata` import."
        )

    metadata_excel_path = Path(metadata_excel_path)

    try:
        meta_dict: Dict[str, Dict[str, Any]] = build_areax_metadata(
            metadata_excel_path,
            sheet_name=metadata_sheet_name,
            volumes_dir=metadata_volumes_dir,
        )
    except TypeError:
        meta_dict = build_areax_metadata(
            metadata_excel_path,
            sheet_name=metadata_sheet_name,
        )

    # ------------------------------------------------------------------
    # 5. Helper to categorize hit type
    # ------------------------------------------------------------------
    def _get_lesion_category(entry: Dict[str, Any], hit_type_col: str) -> str:
        """Returns 'NMA lesion', 'sham', or 'other'"""
        raw = entry.get(hit_type_col, "")
        if raw is None or pd.isna(raw):
            return "other"
        s = str(raw).strip().lower()
        
        # More flexible matching
        if "sham" in s:
            return "sham"
        if "nma" in s:
            return "NMA lesion"
        # Also check for variations
        if "lesion" in s and "visible" not in s and "large" not in s:
            return "NMA lesion"
        return "other"

    # ------------------------------------------------------------------
    # 6. Create box plots for each hemisphere
    # ------------------------------------------------------------------
    def _create_boxplot(hemi_label: str, hit_type_col: str):
        """Create box plot for one hemisphere"""
        
        # Separate Late Pre and Post data
        late_pre = df[df[group_col] == "Late Pre"].copy()
        post = df[df[group_col] == "Post"].copy()
        
        if late_pre.empty or post.empty:
            print(f"[INFO] Missing Late Pre or Post data for {hemi_label}. Skipping.")
            return
        
        # Collect data for each group
        data_dict = {
            "NMA lesion": {"Late Pre": [], "Post": []},
            "sham": {"Late Pre": [], "Post": []}
        }
        
        # Process Late Pre
        for _, row in late_pre.iterrows():
            animal_id = str(row[id_col])
            variance = row[var_col]
            
            if not np.isfinite(variance) or variance <= 0:
                continue
                
            entry = meta_dict.get(animal_id, {})
            category = _get_lesion_category(entry, hit_type_col)
            
            if category in data_dict:
                data_dict[category]["Late Pre"].append(np.log10(variance))
        
        # Process Post
        for _, row in post.iterrows():
            animal_id = str(row[id_col])
            variance = row[var_col]
            
            if not np.isfinite(variance) or variance <= 0:
                continue
                
            entry = meta_dict.get(animal_id, {})
            category = _get_lesion_category(entry, hit_type_col)
            
            if category in data_dict:
                data_dict[category]["Post"].append(np.log10(variance))
        
        # Check if we have data
        has_data = False
        for cat in data_dict:
            if data_dict[cat]["Late Pre"] or data_dict[cat]["Post"]:
                has_data = True
                break
        
        if not has_data:
            print(f"[INFO] No data for {hemi_label}. Skipping.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for box plot
        plot_data = []
        labels = []
        colors = []
        
        color_map = {
            "NMA lesion": "#3498db",  # Blue
            "sham": "#e67e22"  # Orange
        }
        
        for category in ["NMA lesion", "sham"]:
            if data_dict[category]["Late Pre"]:
                plot_data.append(data_dict[category]["Late Pre"])
                labels.append(f"{category}\nLate Pre")
                colors.append(color_map[category])
            else:
                plot_data.append([])
                labels.append(f"{category}\nLate Pre")
                colors.append(color_map[category])
            
            if data_dict[category]["Post"]:
                plot_data.append(data_dict[category]["Post"])
                labels.append(f"{category}\nPost")
                colors.append(color_map[category])
            else:
                plot_data.append([])
                labels.append(f"{category}\nPost")
                colors.append(color_map[category])
        
        # Create box plot
        positions = [1, 2, 4, 5]
        bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                       patch_artist=True, showfliers=True,
                       medianprops=dict(color='black', linewidth=2),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Log₁₀(Variance) [log₁₀(ms²)]", fontsize=12)
        ax.set_title(f"{hemi_label}: Variance Comparison (Late Pre vs Post)\n" +
                    f"NMA Lesion vs Sham Saline Injection (min_phrases={min_phrases})",
                    fontsize=13, fontweight='bold')
        
        # Add gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add sample sizes as text
        y_pos = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for i, (pos, data) in enumerate(zip(positions, plot_data)):
            n = len(data)
            ax.text(pos, y_pos, f'n={n}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Save figure
        hemi_clean = hemi_label.replace(" ", "_")
        fname = f"{file_prefix}_{hemi_clean}.png"
        outpath = output_dir / fname
        
        fig.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        print(f"[PLOT] Saved box plot: {outpath}")
        
        # Print summary statistics
        print(f"\n{hemi_label} Summary:")
        for category in ["NMA lesion", "sham"]:
            for timepoint in ["Late Pre", "Post"]:
                data = data_dict[category][timepoint]
                if data:
                    print(f"  {category} - {timepoint}: n={len(data)}, "
                          f"median={np.median(data):.2f}, "
                          f"mean={np.mean(data):.2f}")
    
    # Create plots for both hemispheres
    _create_boxplot("Medial Area X", medial_hit_type_col)
    _create_boxplot("Lateral Area X", lateral_hit_type_col)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate box plots comparing log(variance) for NMA lesion vs sham groups"
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
        help="Directory with lesion volume files (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--min_phrases",
        type=int,
        default=50,
        help="Minimum N_phrases for filtering.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively.",
    )

    args = parser.parse_args()

    plot_variance_boxplots(
        compiled_stats_path=Path(args.compiled_stats_path),
        metadata_excel_path=Path(args.metadata_excel_path),
        metadata_volumes_dir=Path(args.metadata_volumes_dir) if args.metadata_volumes_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        min_phrases=args.min_phrases,
        show_plots=args.show_plots,
    )