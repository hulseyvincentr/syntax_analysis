#!/usr/bin/env python3
"""
Create phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v19.py
from phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py.

v19 goals:
- Keep the readable v18 violin style.
- Keep each bird's own syllable labels on the x-axis.
- Show every syllable label instead of thinning/hiding labels.
- Make x tick labels small enough to fit better on each 1x3 bird strip.
- Continue exporting PNG only by default.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py")
DST = Path("phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v19.py")

if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}. Run this from py_files after creating v18."
    )

text = SRC.read_text(encoding="utf-8")
text = text.replace("updated_v18", "updated_v19")
text = text.replace(
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v18.py",
    "# phrase_duration_pre_vs_post_grouped_with_label_colors_updated_v19.py",
)

# Add x tick labelsize to both the single-animal and batch function signatures.
old_sig_tail = """    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
) ->"""
new_sig_tail = """    point_alpha: float = 0.40,
    violin_alpha: float = 0.72,
    violin_width: float = 0.82,
    x_tick_labelsize: float = 6.0,
) ->"""
count = text.count(old_sig_tail)
if count < 2:
    raise RuntimeError(f"Could not patch function signatures; found {count} matching signature tails, expected at least 2.")
text = text.replace(old_sig_tail, new_sig_tail, 2)

# Pass x_tick_labelsize from batch to the single-animal function.
old_batch_call = """            point_alpha=point_alpha,
            violin_alpha=violin_alpha,
            violin_width=violin_width,
        )
"""
new_batch_call = """            point_alpha=point_alpha,
            violin_alpha=violin_alpha,
            violin_width=violin_width,
            x_tick_labelsize=x_tick_labelsize,
        )
"""
if old_batch_call not in text:
    raise RuntimeError("Could not patch batch call to pass x_tick_labelsize.")
text = text.replace(old_batch_call, new_batch_call, 1)

# Add command-line argument after --violin-width.
old_cli = """    p.add_argument("--violin-width", dest="violin_width", type=float, default=0.82,
                   help="Violin body width.")
    p.add_argument("--animal-id", type=str, default=None)
"""
new_cli = """    p.add_argument("--violin-width", dest="violin_width", type=float, default=0.82,
                   help="Violin body width.")
    p.add_argument("--x-tick-labelsize", dest="x_tick_labelsize", type=float, default=6.0,
                   help="Font size for syllable-label x tick labels. Use 5-7 if labels overlap.")
    p.add_argument("--animal-id", type=str, default=None)
"""
if old_cli not in text:
    raise RuntimeError("Could not add --x-tick-labelsize CLI argument. Expected v18 CLI block not found.")
text = text.replace(old_cli, new_cli, 1)

# Pass CLI arg through both batch and single-animal main calls.
old_main_call_tail = """            point_alpha=args.point_alpha,
            violin_alpha=args.violin_alpha,
            violin_width=args.violin_width,
        )
"""
new_main_call_tail = """            point_alpha=args.point_alpha,
            violin_alpha=args.violin_alpha,
            violin_width=args.violin_width,
            x_tick_labelsize=args.x_tick_labelsize,
        )
"""
main_count = text.count(old_main_call_tail)
if main_count < 1:
    raise RuntimeError("Could not pass --x-tick-labelsize through main calls.")
text = text.replace(old_main_call_tail, new_main_call_tail)

# Use separate x and y tick font sizes.
old_tick_defaults = """        strip_label_fs = max(9, int(round(LABEL_FS * 0.50)))
        strip_tick_fs = max(8, int(round(TICK_FS * 0.48)))
        strip_title_fs = max(9, int(round(TITLE_FS * 0.45)))
"""
new_tick_defaults = """        strip_label_fs = max(9, int(round(LABEL_FS * 0.50)))
        strip_y_tick_fs = max(8, int(round(TICK_FS * 0.48)))
        strip_x_tick_fs = float(x_tick_labelsize)
        strip_title_fs = max(9, int(round(TITLE_FS * 0.45)))
"""
if old_tick_defaults not in text:
    raise RuntimeError("Could not patch tick font size defaults.")
text = text.replace(old_tick_defaults, new_tick_defaults, 1)

# Replace the old label-thinning helper with a helper that shows all labels.
old_thin_func = """        def _thin_xtick_labels(ax, n_labels: int):
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
"""
new_thin_func = """        def _format_xtick_labels(ax, n_labels: int):
            # Keep every syllable label visible. Use a small font so the labels fit
            # when the 17.4 cm export is placed at 50% on the Illustrator artboard.
            for lab in ax.get_xticklabels():
                lab.set_fontsize(strip_x_tick_fs)
                lab.set_rotation(0)
                lab.set_ha("center")
                lab.set_visible(True)
"""
if old_thin_func not in text:
    raise RuntimeError("Could not replace _thin_xtick_labels helper. Expected v18 helper not found.")
text = text.replace(old_thin_func, new_thin_func, 1)

text = text.replace("_thin_xtick_labels(ax, len(labels))", "_format_xtick_labels(ax, len(labels))")
text = text.replace(
    "ax.tick_params(axis=\"x\", labelsize=strip_tick_fs, pad=2, length=2.2, width=0.65)",
    "ax.tick_params(axis=\"x\", labelsize=strip_x_tick_fs, pad=1, length=2.0, width=0.65)",
)
text = text.replace(
    "ax.tick_params(axis=\"y\", labelsize=strip_tick_fs, pad=2, length=2.2, width=0.65)",
    "ax.tick_params(axis=\"y\", labelsize=strip_y_tick_fs, pad=2, length=2.2, width=0.65)",
)

# Make a tiny bit more bottom room for all labels while keeping the figure tight.
text = text.replace(
    "            bottom=0.145,",
    "            bottom=0.160,",
    1,
)

DST.write_text(text, encoding="utf-8")
print(f"[OK] Wrote {DST}")
print("[INFO] v19 keeps all x-axis syllable labels visible with a small configurable font.")
print("[INFO] Try --x-tick-labelsize 6 first; use 5 if a high-label-count bird still overlaps.")
