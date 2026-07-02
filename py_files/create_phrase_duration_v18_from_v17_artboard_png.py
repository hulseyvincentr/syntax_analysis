#!/usr/bin/env python3
"""
Create phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py
from phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py.

Why this patcher exists:
- The earlier v18 patcher expected to patch v15 directly and could not find
  the local _violin_plus_strip helper in your current copy.
- This version patches the v17 script that already exists in your py_files folder.

v18 goals:
- Revert toward the original visible violin-plot style.
- Keep points small enough that they do not hide the violins.
- Export PNG only by default; no SVGs are written by this v18 combined plot path.
- Export each 1x3 bird strip at 17.4 cm wide, so placing at 50% in Illustrator
  gives 8.7 cm, exactly half of a 17.4 cm artboard.
"""
from __future__ import annotations

from pathlib import Path
import re

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py")
DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py")

if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}. Run this from py_files after creating v17."
    )

text = SRC.read_text(encoding="utf-8")
text = text.replace("updated_v17", "updated_v18")
text = text.replace(
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v17.py",
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py",
)

# Defaults: full artboard width, taller strip, PNG-friendly padding.
text = text.replace("figure_width_cm: float = 16.4", "figure_width_cm: float = 17.4")
text = text.replace("figure_height_cm: float = 3.2", "figure_height_cm: float = 4.8")
text = text.replace("save_pad_inches: float = 0.02", "save_pad_inches: float = 0.03")
text = text.replace('type=float, default=16.4,\n                   help="Width of each combined 1x3 bird strip in centimeters."',
                    'type=float, default=17.4,\n                   help="Width of each combined 1x3 bird strip in cm. 17.4 matches the artboard width; place at 50% for 8.7 cm."')
text = text.replace('type=float, default=3.2,\n                   help="Height of each combined 1x3 bird strip in centimeters."',
                    'type=float, default=4.8,\n                   help="Height of each combined 1x3 bird strip in cm."')
text = text.replace('type=float, default=0.02,\n                   help="Padding used with bbox_inches=\'tight\' when saving figures."',
                    'type=float, default=0.03,\n                   help="Padding used with bbox_inches=\'tight\' when saving PNGs."')

# Add additional params to the main single-animal function if not already present.
old = """    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
) -> GroupedPlotsResult:
"""
new = """    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
    png_dpi: int = 300,
    point_size: float = 1.2,
    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
) -> GroupedPlotsResult:
"""
if old not in text:
    raise RuntimeError("Could not patch run_phrase_duration_pre_vs_post_grouped signature. Expected v17 structure.")
text = text.replace(old, new, 1)

# Add additional params to the batch function signature.
idx = text.find("def run_batch_phrase_duration_from_excel")
old = """    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
) -> Dict[str, GroupedPlotsResult]:
"""
new = """    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
    png_dpi: int = 300,
    point_size: float = 1.2,
    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
) -> Dict[str, GroupedPlotsResult]:
"""
loc = text.find(old, idx)
if loc == -1:
    raise RuntimeError("Could not patch batch function signature. Expected v17 structure.")
text = text[:loc] + new + text[loc + len(old):]

# Pass params from batch to single-animal function.
old = """            save_pad_inches=save_pad_inches,
            save_svg=save_svg,
            save_pdf=save_pdf,
            show_condition_titles=show_condition_titles,
        )
"""
new = """            save_pad_inches=save_pad_inches,
            save_svg=save_svg,
            save_pdf=save_pdf,
            show_condition_titles=show_condition_titles,
            png_dpi=png_dpi,
            point_size=point_size,
            point_alpha=point_alpha,
            violin_alpha=violin_alpha,
            violin_width=violin_width,
        )
"""
if old not in text:
    raise RuntimeError("Could not patch batch call to single-animal function.")
text = text.replace(old, new, 1)

# Add CLI args after --show-condition-titles.
old = """    p.add_argument("--show-condition-titles", dest="show_condition_titles", action="store_true",
                   help="Show Early pre/Late pre/Post titles above the 3 internal panels. Default: hide to save space.")
    p.add_argument("--animal-id", type=str, default=None)
"""
new = """    p.add_argument("--show-condition-titles", dest="show_condition_titles", action="store_true",
                   help="Show Early pre/Late pre/Post titles above the 3 internal panels. Default: hide to save space.")
    p.add_argument("--png-dpi", dest="png_dpi", type=int, default=300,
                   help="PNG resolution. 300 is usually enough for Illustrator placement; use 600 only if needed.")
    p.add_argument("--point-size", dest="point_size", type=float, default=1.2,
                   help="Overlay point size. Smaller values make violin bodies visible.")
    p.add_argument("--point-alpha", dest="point_alpha", type=float, default=0.40,
                   help="Overlay point transparency.")
    p.add_argument("--violin-alpha", dest="violin_alpha", type=float, default=0.72,
                   help="Violin body transparency.")
    p.add_argument("--violin-width", dest="violin_width", type=float, default=0.82,
                   help="Violin body width.")
    p.add_argument("--animal-id", type=str, default=None)
"""
if old not in text:
    raise RuntimeError("Could not add v18 CLI arguments. Expected v17 CLI structure.")
text = text.replace(old, new, 1)

# Pass CLI args through main batch and single-animal calls.
old = """            save_pad_inches=args.save_pad_inches,
            save_svg=args.save_svg,
            save_pdf=args.save_pdf,
            show_condition_titles=args.show_condition_titles,
        )
"""
new = """            save_pad_inches=args.save_pad_inches,
            save_svg=args.save_svg,
            save_pdf=args.save_pdf,
            show_condition_titles=args.show_condition_titles,
            png_dpi=args.png_dpi,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            violin_alpha=args.violin_alpha,
            violin_width=args.violin_width,
        )
"""
count = text.count(old)
if count < 1:
    raise RuntimeError("Could not pass v18 CLI args through main calls.")
text = text.replace(old, new)

# Replace the combined grouped plot function only. This avoids relying on the
# exact internal _violin_plus_strip implementation.
pattern = re.compile(
    r"    def _make_combined_grouped_plot\(label_color_map_for_plot=None\):\n"
    r".*?"
    r"        return outp\n",
    re.DOTALL,
)
replacement = r'''    def _make_combined_grouped_plot(label_color_map_for_plot=None):
        """Make one readable 1x3 bird-level figure: Early pre, Late pre, Post.

        v18 exports at a convenient full-artboard width (17.4 cm by default)
        so the PNG can be placed at 50% in Illustrator for an 8.7 cm panel.
        The plot uses the original violin helper, then makes point overlays
        smaller and violin bodies more visible.
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

        # Font sizes chosen for a 17.4 cm export that may be scaled to 50%.
        strip_label_fs = max(9, int(round(LABEL_FS * 0.50)))
        strip_tick_fs = max(8, int(round(TICK_FS * 0.48)))
        strip_title_fs = max(9, int(round(TITLE_FS * 0.45)))

        def _restore_visible_violin_style(ax):
            """Make violins visible and shrink dense point overlays."""
            # Import locally so this remains robust to top-of-file imports.
            from matplotlib.collections import PathCollection, PolyCollection

            for coll in ax.collections:
                if isinstance(coll, PolyCollection):
                    # Violin bodies are PolyCollections.
                    coll.set_alpha(float(violin_alpha))
                    coll.set_edgecolor("#666666")
                    coll.set_linewidth(0.55)
                    # Keep facecolors as produced by standard vs label-colored plot.
                elif isinstance(coll, PathCollection):
                    # Strip/scatter points are PathCollections. set_sizes expects area.
                    coll.set_sizes([float(point_size) ** 2])
                    coll.set_alpha(float(point_alpha))
                    try:
                        coll.set_edgecolor("none")
                    except Exception:
                        pass

        def _thin_xtick_labels(ax, n_labels: int):
            # Keep all syllables plotted, but thin labels if needed.
            if n_labels <= 24:
                step = 1
            elif n_labels <= 32:
                step = 2
            else:
                step = 3
            ticklabels = ax.get_xticklabels()
            last_idx = len(ticklabels) - 1
            for j, lab in enumerate(ticklabels):
                lab.set_fontsize(strip_tick_fs)
                lab.set_rotation(0)
                lab.set_ha("center")
                if step > 1 and j not in (0, last_idx) and (j % step != 0):
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
                ax.set_title(title if show_condition_titles else "", fontsize=strip_title_fs, pad=2)
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
            _restore_visible_violin_style(ax)

            ax.set_title(title if show_condition_titles else "", fontsize=strip_title_fs, pad=3)
            ax.set_xlabel("")

            if i == 0:
                ax.set_ylabel("Phrase Duration (s)", fontsize=strip_label_fs, labelpad=2)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            ax.margins(x=0.012, y=0.02)
            ax.tick_params(axis="x", labelsize=strip_tick_fs, pad=2, length=2.2, width=0.65)
            ax.tick_params(axis="y", labelsize=strip_tick_fs, pad=2, length=2.2, width=0.65)
            ax.xaxis.label.set_size(strip_label_fs)
            ax.yaxis.label.set_size(strip_label_fs)
            _thin_xtick_labels(ax, len(labels))

            xticks = ax.get_xticks()
            if len(xticks):
                ax.set_xlim(float(np.nanmin(xticks)) - 0.55, float(np.nanmax(xticks)) + 0.55)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(0.75)

        # Do not include a shared xlabel in each bird strip; add that once in Illustrator.
        fig.subplots_adjust(
            left=0.060,
            right=0.996,
            bottom=0.145,
            top=0.895 if show_condition_titles else 0.985,
            wspace=0.065,
        )

        outname = f"{animal_id}_combined_grouped_phrase_durations"
        if label_color_map_for_plot is not None:
            outname += "_label_colors"
        outp = output_dir / f"{outname}.png"

        # PNG only by default. SVG is intentionally not written here.
        _save_figure_tight(
            fig,
            outp,
            dpi=png_dpi,
            pad_inches=save_pad_inches,
            save_svg=False,
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
    raise RuntimeError(f"Could not replace _make_combined_grouped_plot; replacements={n}. Expected exactly 1.")

DST.write_text(text, encoding="utf-8")
print(f"[OK] Wrote {DST}")
print("[INFO] v18 is based on v17, exports PNG only by default, and uses visible violins with smaller points.")
print("[INFO] Default width is 17.4 cm; place at 50% in Illustrator for 8.7 cm half-artboard panels.")
