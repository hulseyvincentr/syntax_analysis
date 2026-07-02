#!/usr/bin/env python3
"""
Create phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py
from phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py.

v17 changes relative to v16:
- Keeps the tight-artboard export idea, but makes the 1x3 bird strips more readable.
- Does NOT force SVG output; use PNG by default, with optional PDF only if requested.
- Removes the shared x-axis label from each bird strip to prevent overlap with syllable tick labels.
- Thins x tick labels automatically for birds with many syllable labels while still plotting all syllables.
- Uses larger margins and more readable default physical size.
"""
from __future__ import annotations

from pathlib import Path
import re

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py")
if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}. Run the v16 patcher first, then run this from your py_files directory."
    )

DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py")

text = SRC.read_text(encoding="utf-8")
text = text.replace("updated_v16", "updated_v17")
text = text.replace("# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py", "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py")

# Make v17 defaults more readable if user omits size arguments.
text = text.replace("figure_height_cm: float = 3.1", "figure_height_cm: float = 3.2")
text = text.replace('type=float, default=3.1,\n                   help="Height of each combined 1x3 bird strip in centimeters."',
                    'type=float, default=3.2,\n                   help="Height of each combined 1x3 bird strip in centimeters."')
text = text.replace("save_pad_inches: float = 0.015", "save_pad_inches: float = 0.02")
text = text.replace('type=float, default=0.015,\n                   help="Padding used with bbox_inches=\'tight\' when saving figures."',
                    'type=float, default=0.02,\n                   help="Padding used with bbox_inches=\'tight\' when saving figures."')

# Replace the combined grouped 1x3 plot function with a more readable version.
pattern = re.compile(
    r"    def _make_combined_grouped_plot\(label_color_map_for_plot=None\):\n"
    r".*?"
    r"        return outp\n",
    re.DOTALL,
)

replacement = r'''    def _make_combined_grouped_plot(label_color_map_for_plot=None):
        """Make one readable 1x3 bird-level figure: Early pre, Late pre, Post.

        v17 is designed for placing multiple bird strips on a manuscript artboard:
        - still tight enough for Illustrator layout
        - larger than v16's ultra-tight export
        - no shared x-label inside each strip by default
        - automatic thinning of x tick labels when many syllables are present
        """
        fig_w = _cm_to_in(figure_width_cm)
        fig_h = _cm_to_in(figure_height_cm)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(fig_w, fig_h),
            sharey=True,
            constrained_layout=False,
        )

        plot_defs = [
            (early_pre, "Early pre-lesion", len(early_pre)),
            (late_pre, "Late pre-lesion", len(late_pre)),
            (post_g, "Post-lesion", len(post_g)),
        ]

        # Smaller than the original manuscript figures, but readable at final artboard size.
        strip_label_fs = max(7, int(round(LABEL_FS * 0.36)))
        strip_tick_fs = max(6, int(round(TICK_FS * 0.32)))
        strip_title_fs = max(7, int(round(TITLE_FS * 0.36)))

        def _thin_xtick_labels(ax, n_labels: int):
            """Show only every Nth x tick label while keeping all syllables plotted."""
            if n_labels <= 12:
                step = 1
            elif n_labels <= 20:
                step = 2
            elif n_labels <= 32:
                step = 3
            else:
                step = 4

            ticklabels = ax.get_xticklabels()
            last_idx = len(ticklabels) - 1
            for j, lab in enumerate(ticklabels):
                lab.set_fontsize(strip_tick_fs)
                lab.set_rotation(0)
                lab.set_ha("center")
                # Keep first and last label visible, plus every step-th label.
                if j not in (0, last_idx) and (j % step != 0):
                    lab.set_visible(False)

        for i, (ax, (ds, title, n)) in enumerate(zip(axes, plot_defs)):
            if ds is None or len(ds) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No phrase durations",
                    ha="center",
                    va="center",
                    fontsize=strip_title_fs,
                )
                ax.set_title(title if show_condition_titles else "", fontsize=strip_title_fs, pad=1)
                _pretty_axes(ax, 0)
                if i != 0:
                    ax.tick_params(axis="y", left=False, labelleft=False)
                continue

            tall = _explode_for_plot(ds, labels)
            _violin_plus_strip(
                ax,
                tall,
                (y_min, y_max),
                labels,
                label_color_map=label_color_map_for_plot,
            )

            ax.set_title(title if show_condition_titles else "", fontsize=strip_title_fs, pad=1)
            ax.set_xlabel("")

            if i == 0:
                ax.set_ylabel("Phrase Duration (s)", fontsize=strip_label_fs, labelpad=1.5)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            # Interior axis formatting.
            ax.margins(x=0.006, y=0.015)
            ax.tick_params(axis="x", labelsize=strip_tick_fs, pad=1, length=2, width=0.6)
            ax.tick_params(axis="y", labelsize=strip_tick_fs, pad=1, length=2, width=0.6)
            ax.xaxis.label.set_size(strip_label_fs)
            ax.yaxis.label.set_size(strip_label_fs)
            _thin_xtick_labels(ax, len(labels))

            # Avoid extra white x-padding outside first/last syllable positions.
            xticks = ax.get_xticks()
            if len(xticks):
                ax.set_xlim(float(np.nanmin(xticks)) - 0.45, float(np.nanmax(xticks)) + 0.45)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(0.7)

        # Do NOT add fig.supxlabel here. It overlaps in small multi-bird layouts.
        # Add a single "Syllable Label" label in Illustrator for each section or the full figure.
        fig.subplots_adjust(
            left=0.065,
            right=0.996,
            bottom=0.115,
            top=0.900 if show_condition_titles else 0.985,
            wspace=0.065,
        )

        outname = f"{animal_id}_combined_grouped_phrase_durations"
        if label_color_map_for_plot is not None:
            outname += "_label_colors"
        outp = output_dir / f"{outname}.png"

        _save_figure_tight(
            fig,
            outp,
            dpi=600,
            pad_inches=save_pad_inches,
            save_svg=save_svg,
            save_pdf=save_pdf,
            facecolor="white",
        )
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        return outp
'''

text, n = pattern.subn(replacement, text, count=1)
if n != 1:
    raise RuntimeError(f"Could not replace _make_combined_grouped_plot; replacements={n}")

DST.write_text(text, encoding="utf-8")
print(f"[OK] Wrote {DST}")
print("[INFO] v17 defaults: PNG only unless --save-pdf or --save-svg is explicitly provided.")
