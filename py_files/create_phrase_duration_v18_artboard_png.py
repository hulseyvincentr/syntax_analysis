#!/usr/bin/env python3
"""
Create phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py
from phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py.

v18 goal:
- Revert to a visible violin + small point overlay style.
- Export PNGs only by default; no SVGs.
- Use an Illustrator-friendly physical width: 17.4 cm by default, matching the paper artboard width.
  You can place at 50% in Illustrator for an 8.7 cm half-artboard panel.
- Keep each bird's own syllable labels; do not force common x-axis labels across birds.
- Keep y-limits shared when --y_max_ms is provided.
- Make --outdir work in batch mode as a separate output root.

Run from:
/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files
"""
from __future__ import annotations

from pathlib import Path
import re

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py")
DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py")

if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}. Run this patcher from your py_files directory, "
        "where phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py is stored."
    )

text = SRC.read_text(encoding="utf-8")
text = text.replace("updated_v15", "updated_v18")
text = text.replace(
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v8.py",
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py",
)
text = text.replace(
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v15.py",
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py",
)

# ---------------------------------------------------------------------
# Add cm-to-inch and PNG-only tight-save helpers after font constants.
# ---------------------------------------------------------------------
helper_marker = "TITLE_FS = 24\nLABEL_FS = 22\nTICK_FS = 18\n"
helper_insert = r'''TITLE_FS = 24
LABEL_FS = 22
TICK_FS = 18

# Illustrator/artboard-friendly export helpers.
# User artboard: 17.4 cm wide x 18 cm high.
def _cm_to_in(cm: float) -> float:
    return float(cm) / 2.54


def _save_figure_png_tight(
    fig,
    out_png: Union[str, Path],
    *,
    dpi: int = 300,
    pad_inches: float = 0.03,
    save_pdf: bool = False,
    facecolor: str = "white",
):
    """Save PNG with minimal extra whitespace; optionally also save PDF.

    SVG export is intentionally omitted in v18 because those were difficult to open/use.
    """
    out_png = Path(out_png)
    save_kwargs = dict(
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor=facecolor,
        transparent=False,
    )
    fig.savefig(out_png, **save_kwargs)
    if save_pdf:
        fig.savefig(
            out_png.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=float(pad_inches),
            facecolor=facecolor,
            transparent=False,
        )

'''
if helper_marker not in text:
    raise RuntimeError("Could not find font constant block to insert helpers.")
text = text.replace(helper_marker, helper_insert, 1)

# ---------------------------------------------------------------------
# Replace violin/strip helper with a robust version that keeps visible violins
# and uses small points, rather than the oversized-dot look.
# ---------------------------------------------------------------------
violin_pattern = re.compile(
    r"def _violin_plus_strip\(.*?\n(?=def _pretty_axes\()",
    re.DOTALL,
)
violin_replacement = r'''def _violin_plus_strip(
    ax,
    tall: pd.DataFrame,
    y_lim: Tuple[float, float],
    label_order: List[str],
    label_color_map: Optional[Dict[str, str]] = None,
    *,
    point_size: float = 1.2,
    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
):
    """Violin + small jittered points for phrase-duration distributions.

    v18 intentionally restores visible violin bodies and makes the overlaid points
    small enough that they do not obscure the distributions.
    This function is written defensively so it works with either the original
    exploded-table column names or nearby variants.
    """
    if tall is None or len(tall) == 0:
        lo, hi = y_lim
        hi_s = hi / 1000.0 if hi is not None and hi > 100 else hi
        ax.set_ylim(0, hi_s if hi_s is not None else 1)
        return

    plot_df = tall.copy()

    # Infer label and duration columns robustly.
    label_candidates = [
        "Syllable Label", "syllable_label", "Syllable", "syllable",
        "Label", "label", "HDBSCAN Label", "hdbscan_label",
    ]
    duration_candidates = [
        "Phrase Duration (s)", "phrase_duration_s", "duration_s", "Duration_s",
        "Phrase Duration (ms)", "phrase_duration_ms", "duration_ms", "Duration_ms",
        "duration", "Duration", "phrase_duration", "Phrase Duration",
    ]

    label_col = next((c for c in label_candidates if c in plot_df.columns), None)
    if label_col is None:
        # Fallback: first column containing label/syllable.
        label_col = next((c for c in plot_df.columns if "label" in str(c).lower() or "syll" in str(c).lower()), None)
    if label_col is None:
        raise ValueError(f"Could not infer syllable-label column from columns: {list(plot_df.columns)}")

    y_col = next((c for c in duration_candidates if c in plot_df.columns), None)
    if y_col is None:
        # Fallback: first numeric-looking column containing duration.
        y_col = next((c for c in plot_df.columns if "duration" in str(c).lower()), None)
    if y_col is None:
        raise ValueError(f"Could not infer duration column from columns: {list(plot_df.columns)}")

    plot_df["_plot_label_str"] = plot_df[label_col].astype(str)
    raw_y = pd.to_numeric(plot_df[y_col], errors="coerce")

    y0, y1 = y_lim
    # Convert to seconds if input appears to be milliseconds.
    raw_max = float(np.nanmax(raw_y.values)) if np.isfinite(raw_y).any() else 0.0
    use_ms = (y1 is not None and y1 > 100) or raw_max > 100
    plot_df["_plot_duration_s"] = raw_y / 1000.0 if use_ms else raw_y
    plot_df = plot_df[np.isfinite(plot_df["_plot_duration_s"])].copy()

    order = [str(x) for x in label_order]
    if not order:
        order = sorted(plot_df["_plot_label_str"].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    # Palette: label-colored version uses provided colors; standard version uses gray violins.
    if label_color_map_for_plot := label_color_map:
        palette = {str(k): v for k, v in label_color_map_for_plot.items()}
        # Ensure all labels have a fallback color.
        for lab in order:
            palette.setdefault(str(lab), "#D0D0D0")
        sns.violinplot(
            data=plot_df,
            x="_plot_label_str",
            y="_plot_duration_s",
            order=order,
            ax=ax,
            palette=palette,
            inner=None,
            cut=0,
            width=float(violin_width),
            linewidth=0.55,
            saturation=0.85,
        )
    else:
        sns.violinplot(
            data=plot_df,
            x="_plot_label_str",
            y="_plot_duration_s",
            order=order,
            ax=ax,
            color="#D0D0D0",
            inner=None,
            cut=0,
            width=float(violin_width),
            linewidth=0.55,
            saturation=1.0,
        )

    # Make violin bodies visible but not too heavy.
    for coll in ax.collections:
        if coll.__class__.__name__.lower().endswith("polycollection"):
            coll.set_alpha(float(violin_alpha))
            coll.set_edgecolor("#666666")
            coll.set_linewidth(0.55)

    # Small points overlaid on top of visible violins.
    sns.stripplot(
        data=plot_df,
        x="_plot_label_str",
        y="_plot_duration_s",
        order=order,
        ax=ax,
        color="#2E4A46",
        size=float(point_size),
        alpha=float(point_alpha),
        jitter=0.16,
        linewidth=0,
        zorder=3,
    )

    y0_s = y0 / 1000.0 if y0 is not None and y0 > 100 else y0
    y1_s = y1 / 1000.0 if y1 is not None and y1 > 100 else y1
    if y1_s is not None:
        ax.set_ylim(y0_s if y0_s is not None else 0, y1_s)
    ax.set_xlabel("Syllable Label")
    ax.set_ylabel("Phrase Duration (s)")

'''
text, n = violin_pattern.subn(violin_replacement, text, count=1)
if n != 1:
    raise RuntimeError(f"Could not replace _violin_plus_strip; replacements={n}. Expected exactly 1.")

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
    figure_width_cm: float = 17.4,
    figure_height_cm: float = 4.8,
    png_dpi: int = 300,
    save_pad_inches: float = 0.03,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
    add_shared_xlabel: bool = False,
    point_size: float = 1.2,
    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
) -> GroupedPlotsResult:
"""
if old not in text:
    raise RuntimeError("Could not patch run_phrase_duration_pre_vs_post_grouped signature.")
text = text.replace(old, new, 1)

# ---------------------------------------------------------------------
# Replace combined grouped 1x3 plot with artboard-friendly PNG version.
# ---------------------------------------------------------------------
combined_pattern = re.compile(
    r"    def _make_combined_grouped_plot\(label_color_map_for_plot=None\):\n"
    r".*?"
    r"        return outp\n",
    re.DOTALL,
)
combined_replacement = r'''    def _make_combined_grouped_plot(label_color_map_for_plot=None):
        """Make one readable 1x3 bird-level figure: Early pre, Late pre, Post.

        v18 is designed for Illustrator assembly:
        - 17.4 cm wide by default, matching the artboard width
        - place at 50% in Illustrator for a clean half-artboard-width panel
        - visible violins with small points
        - PNG export only by default
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

        # Sized for export at full artboard width and optional 50% scaling in Illustrator.
        strip_label_fs = max(10, int(round(LABEL_FS * 0.55)))
        strip_tick_fs = max(8, int(round(TICK_FS * 0.52)))
        strip_title_fs = max(10, int(round(TITLE_FS * 0.50)))

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
                point_size=point_size,
                point_alpha=point_alpha,
                violin_alpha=violin_alpha,
                violin_width=violin_width,
            )

            ax.set_title(title if show_condition_titles else "", fontsize=strip_title_fs, pad=2)
            ax.set_xlabel("")

            if i == 0:
                ax.set_ylabel("Phrase Duration (s)", fontsize=strip_label_fs, labelpad=2)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            # Keep all syllables plotted with each bird's own labels.
            ax.margins(x=0.012, y=0.02)
            ax.tick_params(axis="x", labelsize=strip_tick_fs, pad=1.5, length=2.2, width=0.65)
            ax.tick_params(axis="y", labelsize=strip_tick_fs, pad=1.5, length=2.2, width=0.65)

            # For very label-rich birds, slightly thin tick-label text only; all violins remain plotted.
            xticklabels = ax.get_xticklabels()
            if len(xticklabels) > 32:
                step = 2
                last_idx = len(xticklabels) - 1
                for j, lab in enumerate(xticklabels):
                    if j not in (0, last_idx) and (j % step != 0):
                        lab.set_visible(False)

            xticks = ax.get_xticks()
            if len(xticks):
                ax.set_xlim(float(np.nanmin(xticks)) - 0.55, float(np.nanmax(xticks)) + 0.55)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(0.75)

        if add_shared_xlabel:
            fig.supxlabel("Syllable Label", fontsize=strip_label_fs, y=0.035)
            bottom = 0.20
        else:
            bottom = 0.13

        fig.subplots_adjust(
            left=0.070,
            right=0.996,
            bottom=bottom,
            top=0.88 if show_condition_titles else 0.965,
            wspace=0.075,
        )

        outname = f"{animal_id}_combined_grouped_phrase_durations"
        if label_color_map_for_plot is not None:
            outname += "_label_colors"
        outp = output_dir / f"{outname}.png"

        _save_figure_png_tight(
            fig,
            outp,
            dpi=png_dpi,
            pad_inches=save_pad_inches,
            save_pdf=save_pdf,
            facecolor="white",
        )
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        return outp
'''
text, n = combined_pattern.subn(combined_replacement, text, count=1)
if n != 1:
    raise RuntimeError(f"Could not replace _make_combined_grouped_plot; replacements={n}. Expected exactly 1.")

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
    figure_width_cm: float = 17.4,
    figure_height_cm: float = 4.8,
    png_dpi: int = 300,
    save_pad_inches: float = 0.03,
    save_pdf: bool = False,
    show_condition_titles: bool = False,
    add_shared_xlabel: bool = False,
    point_size: float = 1.2,
    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
) -> Dict[str, GroupedPlotsResult]:
"""
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
            png_dpi=png_dpi,
            save_pad_inches=save_pad_inches,
            save_pdf=save_pdf,
            show_condition_titles=show_condition_titles,
            add_shared_xlabel=add_shared_xlabel,
            point_size=point_size,
            point_alpha=point_alpha,
            violin_alpha=violin_alpha,
            violin_width=violin_width,
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
    p.add_argument("--figure-width-cm", dest="figure_width_cm", type=float, default=17.4,
                   help="Width of each combined 1x3 bird strip in cm. 17.4 matches the artboard width; place at 50% for 8.7 cm.")
    p.add_argument("--figure-height-cm", dest="figure_height_cm", type=float, default=4.8,
                   help="Height of each combined 1x3 bird strip in cm.")
    p.add_argument("--png-dpi", dest="png_dpi", type=int, default=300,
                   help="PNG resolution. Use 300 for manageable files or 600 for very high resolution.")
    p.add_argument("--save-pad-inches", dest="save_pad_inches", type=float, default=0.03,
                   help="Padding used with bbox_inches='tight' when saving PNGs.")
    p.add_argument("--save-pdf", dest="save_pdf", action="store_true",
                   help="Also save PDF copies. SVG output is intentionally not generated in v18.")
    p.add_argument("--show-condition-titles", dest="show_condition_titles", action="store_true",
                   help="Show Early pre/Late pre/Post titles above the three internal panels.")
    p.add_argument("--add-shared-xlabel", dest="add_shared_xlabel", action="store_true",
                   help="Add one shared 'Syllable Label' xlabel under the 1x3 strip. Usually omit and add text in Illustrator.")
    p.add_argument("--point-size", dest="point_size", type=float, default=1.2,
                   help="Overlay point size for stripplot dots.")
    p.add_argument("--point-alpha", dest="point_alpha", type=float, default=0.40,
                   help="Overlay point transparency.")
    p.add_argument("--violin-alpha", dest="violin_alpha", type=float, default=0.72,
                   help="Violin body transparency.")
    p.add_argument("--violin-width", dest="violin_width", type=float, default=0.82,
                   help="Violin body width.")
    p.add_argument("--animal-id", type=str, default=None)
"""
if old not in text:
    raise RuntimeError("Could not patch CLI arguments after --y_max_ms.")
text = text.replace(old, new, 1)

# Batch main call: add output_root and artboard export args.
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
            png_dpi=args.png_dpi,
            save_pad_inches=args.save_pad_inches,
            save_pdf=args.save_pdf,
            show_condition_titles=args.show_condition_titles,
            add_shared_xlabel=args.add_shared_xlabel,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            violin_alpha=args.violin_alpha,
            violin_width=args.violin_width,
        )
"""
count = text.count(old)
if count < 1:
    raise RuntimeError("Could not patch main run call(s) with artboard export args.")
text = text.replace(old, new)

DST.write_text(text, encoding="utf-8")
print(f"[OK] Wrote {DST}")
print("[INFO] v18 exports PNG by default; no SVGs are generated.")
print("[INFO] Default width is 17.4 cm; place at 50% in Illustrator for 8.7 cm half-artboard panels.")
