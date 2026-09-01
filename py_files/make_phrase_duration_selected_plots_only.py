#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import py_compile

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py")
DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_selected_plots_only.py")

if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}.\n"
        "Run this patcher from the py_files directory."
    )

text = SRC.read_text(encoding="utf-8")

# Fix argparse literal percent if needed.
text = text.replace("place at 50% for", "place at 50%% for")

# Enforce surgery-day exclusion.
text = text.replace(
    'pre, post = df[df["_dt"] < t_date].copy(), df[df["_dt"] >= t_date].copy()',
    'pre, post = df[df["_dt"] < t_date].copy(), df[df["_dt"] > t_date].copy()',
)

# Stop writing the compact significance CSV.
sig_pattern = re.compile(
    r'    compact_significance_df = '
    r'_build_compact_significance_df\(compact_group_values, labels\)\n'
    r'    compact_significance_path = output_dir / '
    r'f"\{animal_id\}_compact_grouped_phrase_duration_within_syllable_significance\.csv"\n'
    r'    compact_significance_df\.to_csv\(compact_significance_path, index=False\)\n'
)
sig_replacement = (
    '    compact_significance_df = '
    '_build_compact_significance_df(compact_group_values, labels)\n'
    '    compact_significance_path = None\n'
)
text, n_sig = sig_pattern.subn(sig_replacement, text, count=1)
if n_sig != 1:
    raise RuntimeError(
        "Could not find the compact-significance CSV block "
        f"(matches={n_sig})."
    )

# Replace the plot-generation block with only the four requested plots.
plot_pattern = re.compile(
    r'    ep = lp = po = None\n'
    r'.*?'
    r'(?=    agg = None\n)',
    re.DOTALL,
)

plot_replacement = r'''    # SELECTED PLOTS ONLY
    ep = lp = po = None
    epc = lpc = poc = None

    compact_grouped = None
    compact_grouped_no_sig = None
    compact_grouped_no_title_no_xlabels = None
    compact_grouped_large_xticks = None
    compact_grouped_sorted_by_post_variance = None
    compact_grouped_sorted_by_post_variance_group_colors = None

    combined = None
    combined_colored = None

    # Always resolve label colors for the requested plots.
    if label_color_map is None:
        label_color_map = _resolve_label_color_map(
            labels,
            fixed_label_colors_json=fixed_label_colors_json,
        )
        if fixed_label_colors_json is not None:
            print(f"[INFO] Using fixed label colors from: {fixed_label_colors_json}")
        else:
            print("[INFO] Using auto-generated label colors.")

    if len(early_pre):
        epc = _make_plot(
            early_pre,
            "Early Pre-Treatment",
            len(early_pre),
            "early_pre",
            label_color_map_for_plot=label_color_map,
        )

    if len(late_pre):
        lpc = _make_plot(
            late_pre,
            "Late Pre-Treatment",
            len(late_pre),
            "late_pre",
            label_color_map_for_plot=label_color_map,
        )

    if len(post_g):
        poc = _make_plot(
            post_g,
            "Post-Treatment",
            len(post_g),
            "post",
            label_color_map_for_plot=label_color_map,
        )

    if len(early_pre) or len(late_pre) or len(post_g):
        compact_grouped_no_title_no_xlabels = (
            _make_compact_grouped_syllable_points_plot(
                show_significance=False,
                include_title=False,
                show_x_label=False,
                show_xtick_labels=False,
                file_suffix="_no_title_no_xlabels",
            )
        )

    if make_box_plots:
        print(
            "[INFO] Ignoring --make-box-plots: selected-plots-only mode "
            "generates only the four requested PNGs."
        )
    make_box_plots = False

'''

text, n_plot = plot_pattern.subn(plot_replacement, text, count=1)
if n_plot != 1:
    raise RuntimeError(
        "Could not identify the v18 plot-generation block "
        f"(matches={n_plot})."
    )

text = text.replace(
    "phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py",
    "phrase_duration_pre_vs_post_grouped_with_label_colors_selected_plots_only.py",
)

DST.write_text(text, encoding="utf-8")
py_compile.compile(str(DST), doraise=True)

print(f"[OK] Wrote and syntax-checked:\n  {DST}")
print("\nThis version generates only:")
print("  1. early-pre label-colored violin")
print("  2. late-pre label-colored violin")
print("  3. post label-colored violin")
print("  4. compact no-significance/no-title/no-xlabels comparison")
