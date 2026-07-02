#!/usr/bin/env python3
"""
Create phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py
from phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py.

v16 changes:
- Tight 1x3 combined grouped plot export for Illustrator/artboard assembly.
- Adds figure size in cm, tight bbox/padding, SVG/PDF export.
- Keeps each bird's own x-axis syllable labels by default.
- Allows --outdir to work in batch mode as a separate output root.
"""
from __future__ import annotations

from pathlib import Path
import re

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py")
DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py")

if not SRC.exists():
    raise FileNotFoundError(f"Could not find {SRC}. Run this from your py_files directory.")

text = SRC.read_text(encoding="utf-8")
text = text.replace("# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v8.py", "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v16.py")
text = text.replace("updated_v15", "updated_v16")

# ---------------------------------------------------------------------
# Add cm-to-inch and tight-save helpers after font constants.
# ---------------------------------------------------------------------
helper_marker = "TITLE_FS = 24\nLABEL_FS = 22\nTICK_FS = 18\n"
helper_insert = """TITLE_FS = 24
LABEL_FS = 22
TICK_FS = 18

# Illustrator/artboard-friendly export helpers
# JNeurosci-style artboard mentioned by user: 17.4 cm wide x 18 cm high.
def _cm_to_in(cm: float) -> float:
    return float(cm) / 2.54


def _save_figure_tight(
    fig,
    out_png: Union[str, Path],
    *,
    dpi: int = 600,
    pad_inches: float = 0.015,
    save_svg: bool = False,
    save_pdf: bool = False,
    facecolor: str = "white",
):
    out_png = Path(out_png)
    save_kwargs = dict(
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor=facecolor,
        transparent=False,
    )
    fig.savefig(out_png, **save_kwargs)
    if save_svg:
        fig.savefig(
            out_png.with_suffix(".svg"),
            bbox_inches="tight",
            pad_inches=float(pad_inches),
            facecolor=facecolor,
            transparent=False,
        )
    if save_pdf:
        fig.savefig(
            out_png.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=float(pad_inches),
            facecolor=facecolor,
            transparent=False,
        )

"""
if helper_marker not in text:
    raise RuntimeError("Could not find font constant block to insert helpers.")
text = text.replace(helper_marker, helper_insert, 1)

# ---------------------------------------------------------------------
# Extend run_phrase_duration_pre_vs_post_grouped signature.
# ---------------------------------------------------------------------
old = """    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_colored_violin_plots: bool = False,
    make_box_plots: bool = False,
) -> GroupedPlotsResult:
"""
new = """    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_colored_violin_plots: bool = False,
    make_box_plots: bool = False,
    figure_width_cm: float = 16.4,
    figure_height_cm: float = 3.1,
    save_pad_inches: float = 0.015,
    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
) -> GroupedPlotsResult:
"""
if old not in text:
    raise RuntimeError("Could not patch run_phrase_duration_pre_vs_post_grouped signature.")
text = text.replace(old, new, 1)

# ---------------------------------------------------------------------
# Replace combined grouped 1x3 plot function with tight version.
# ---------------------------------------------------------------------
pattern = re.compile(
    r"    def _make_combined_grouped_plot\(label_color_map_for_plot=None\):\n"
    r".*?"
    r"        return outp\n",
    re.DOTALL,
)
replacement = r'''    def _make_combined_grouped_plot(label_color_map_for_plot=None):
        """Make one tight 1x3 bird-level figure: Early pre, Late pre, Post.

        This version is designed for Illustrator/artboard assembly:
        small margins, user-specified physical size in cm, tight saved bbox,
        and optional SVG/PDF export.
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

        # Scale fonts down for small strip-style panels.
        small_label_fs = max(7, int(round(LABEL_FS * 0.48)))
        small_tick_fs = max(6, int(round(TICK_FS * 0.48)))
        small_title_fs = max(7, int(round(TITLE_FS * 0.42)))

        for i, (ax, (ds, title, n)) in enumerate(zip(axes, plot_defs)):
            if ds is None or len(ds) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No phrase durations",
                    ha="center",
                    va="center",
                    fontsize=small_title_fs,
                )
                ax.set_title(title if show_condition_titles else "", fontsize=small_title_fs, pad=1)
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

            ax.set_title(title if show_condition_titles else "", fontsize=small_title_fs, pad=1)
            ax.set_xlabel("")

            if i == 0:
                ax.set_ylabel("Phrase Duration (s)", fontsize=small_label_fs, labelpad=2)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            # Tighten interior axis spacing and make tick labels workable at final size.
            ax.margins(x=0.005, y=0.01)
            ax.tick_params(axis="x", labelsize=small_tick_fs, pad=1, length=2)
            ax.tick_params(axis="y", labelsize=small_tick_fs, pad=1, length=2)
            ax.xaxis.label.set_size(small_label_fs)
            ax.yaxis.label.set_size(small_label_fs)

            # Avoid extra white x-padding outside the first/last syllable labels.
            xticks = ax.get_xticks()
            if len(xticks):
                ax.set_xlim(float(np.nanmin(xticks)) - 0.5, float(np.nanmax(xticks)) + 0.5)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        # One shared x label for the whole 1x3 strip.
        fig.supxlabel("Syllable Label", fontsize=small_label_fs, y=0.025)

        # These are intentionally compact; bbox_inches="tight" does final trimming.
        fig.subplots_adjust(
            left=0.050,
            right=0.998,
            bottom=0.205,
            top=0.925 if show_condition_titles else 0.985,
            wspace=0.035,
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

# ---------------------------------------------------------------------
# Extend batch wrapper signature and output handling.
# ---------------------------------------------------------------------
old = """def run_batch_phrase_duration_from_excel(
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    *,
"""
new = """def run_batch_phrase_duration_from_excel(
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    *,
    output_root: Optional[Union[str, Path]] = None,
"""
if old not in text:
    raise RuntimeError("Could not patch batch wrapper signature start.")
text = text.replace(old, new, 1)

old = """    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_colored_violin_plots: bool = False,
    make_box_plots: bool = False,
) -> Dict[str, GroupedPlotsResult]:
"""
new = """    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_colored_violin_plots: bool = False,
    make_box_plots: bool = False,
    figure_width_cm: float = 16.4,
    figure_height_cm: float = 3.1,
    save_pad_inches: float = 0.015,
    save_svg: bool = False,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
) -> Dict[str, GroupedPlotsResult]:
"""
# This signature block appears twice? Replace only the first remaining batch wrapper occurrence.
idx = text.find(old, text.find("def run_batch_phrase_duration_from_excel"))
if idx == -1:
    raise RuntimeError("Could not patch batch wrapper keyword signature.")
text = text[:idx] + new + text[idx + len(old):]

old = """    excel_path = Path(excel_path)
    json_root = Path(json_root)
"""
new = """    excel_path = Path(excel_path)
    json_root = Path(json_root)
    output_root = Path(output_root) if output_root is not None else None
"""
if old not in text:
    raise RuntimeError("Could not patch output_root conversion.")
text = text.replace(old, new, 1)

old = """        outdir = decoded_path.parent / "figures" / "phrase_durations"
        outdir.mkdir(parents=True, exist_ok=True)
"""
new = """        if output_root is None:
            outdir = decoded_path.parent / "figures" / "phrase_durations"
        else:
            outdir = output_root / animal_id / "figures" / "phrase_durations"
        outdir.mkdir(parents=True, exist_ok=True)
"""
if old not in text:
    raise RuntimeError("Could not patch batch outdir handling.")
text = text.replace(old, new, 1)

old = """            fixed_label_colors_json=fixed_label_colors_json,
            make_colored_violin_plots=make_colored_violin_plots,
            make_box_plots=make_box_plots,
        )
"""
new = """            fixed_label_colors_json=fixed_label_colors_json,
            make_colored_violin_plots=make_colored_violin_plots,
            make_box_plots=make_box_plots,
            figure_width_cm=figure_width_cm,
            figure_height_cm=figure_height_cm,
            save_pad_inches=save_pad_inches,
            save_svg=save_svg,
            save_pdf=save_pdf,
            show_condition_titles=show_condition_titles,
        )
"""
if old not in text:
    raise RuntimeError("Could not patch batch call to run_phrase_duration_pre_vs_post_grouped.")
text = text.replace(old, new, 1)

# ---------------------------------------------------------------------
# Add CLI arguments after --y_max_ms.
# ---------------------------------------------------------------------
old = """    p.add_argument("--y_max_ms", type=float, default=None)
    p.add_argument("--animal-id", type=str, default=None)
"""
new = """    p.add_argument("--y_max_ms", type=float, default=None)
    p.add_argument("--figure-width-cm", dest="figure_width_cm", type=float, default=16.4,
                   help="Width of each combined 1x3 bird strip in centimeters.")
    p.add_argument("--figure-height-cm", dest="figure_height_cm", type=float, default=3.1,
                   help="Height of each combined 1x3 bird strip in centimeters.")
    p.add_argument("--save-pad-inches", dest="save_pad_inches", type=float, default=0.015,
                   help="Padding used with bbox_inches='tight' when saving figures.")
    p.add_argument("--save-svg", dest="save_svg", action="store_true",
                   help="Also save SVG copies of combined 1x3 figures for Illustrator.")
    p.add_argument("--save-pdf", dest="save_pdf", action="store_true",
                   help="Also save PDF copies of combined 1x3 figures for Illustrator.")
    p.add_argument("--show-condition-titles", dest="show_condition_titles", action="store_true",
                   help="Show Early pre/Late pre/Post titles above the 3 internal panels. Default: hide to save space.")
    p.add_argument("--animal-id", type=str, default=None)
"""
if old not in text:
    raise RuntimeError("Could not patch CLI arguments.")
text = text.replace(old, new, 1)

# Batch main call: add output_root and tight export args.
old = """        results = run_batch_phrase_duration_from_excel(
            excel_path=args.metadata_excel,
            json_root=args.json_root,
            sheet_name=metadata_sheet,
"""
new = """        results = run_batch_phrase_duration_from_excel(
            excel_path=args.metadata_excel,
            json_root=args.json_root,
            output_root=args.outdir,
            sheet_name=metadata_sheet,
"""
if old not in text:
    raise RuntimeError("Could not patch batch main call start.")
text = text.replace(old, new, 1)

old = """            fixed_label_colors_json=args.fixed_label_colors_json,
            make_colored_violin_plots=args.make_colored_violins,
            make_box_plots=args.make_box_plots,
        )
"""
new = """            fixed_label_colors_json=args.fixed_label_colors_json,
            make_colored_violin_plots=args.make_colored_violins,
            make_box_plots=args.make_box_plots,
            figure_width_cm=args.figure_width_cm,
            figure_height_cm=args.figure_height_cm,
            save_pad_inches=args.save_pad_inches,
            save_svg=args.save_svg,
            save_pdf=args.save_pdf,
            show_condition_titles=args.show_condition_titles,
        )
"""
# Replace both batch-main and single-main occurrences now.
count = text.count(old)
if count < 1:
    raise RuntimeError("Could not patch main run calls with tight export args.")
text = text.replace(old, new)

DST.write_text(text, encoding="utf-8")
print(f"[OK] Wrote {DST}")
