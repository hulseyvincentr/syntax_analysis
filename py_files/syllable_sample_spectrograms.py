#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate stitched spectrogram samples for one or many HDBSCAN labels (linear scaling only).

- selected_labels = None  → process all labels found (optionally skip -1 as noise)
- selected_labels = [0, 3, 15] → process only that subset

For each label, we build `num_sample_spectrograms` spectrograms, each made from
exactly `spectrogram_length` timebins whose label == that label. The bins are
stitched in temporal order and are non-overlapping across samples for a label.

All sample spectrogram images are saved directly into `output_dir` (if provided)
**only** when `save_sample_spectrograms=True`.

Additionally, this script can plot a UMAP scatter colored by HDBSCAN labels
(if `embedding_outputs` is present in the NPZ). Colors use a 60-color qualitative
palette by concatenating tab20, tab20b, and tab20c. If you have >60 labels,
colors will repeat.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ============
# User inputs
# ============
f = "/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/USA5505/USA5505.npz"
output_dir = "/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/Area_X_figures/USA5505_repertoire"

# Process ALL labels: set to None
# Process a subset: provide a list, e.g. [0, 3, 15]
selected_labels: Optional[List[int]] = None

skip_noise_label = True      # If True, skip label == -1 when selected_labels is None
spectrogram_length = 1000    # timebins per sample
num_sample_spectrograms = 1  # samples to generate per label (non-overlapping)
cmap = "gray_r"              # black & white
show_colorbar = False        # add a colorbar for each image
show_plots = True            # show figures on screen
save_sample_spectrograms = True  # True → save sample spectrogram PNGs; False → don't save

# UMAP plot toggles (does not affect sample spectrogram saving)
make_umap_plot = True        # set False to skip making the UMAP plot
show_umap_legend = True      # legend with one entry per label

# ==============
# Load & orient
# ==============
arr = np.load(f, allow_pickle=True)
labels = arr["hdbscan_labels"]      # shape (T,)
S = arr["s"]                        # shape (F, T) or (T, F)

# Try to load UMAP embedding (optional)
embedding = arr["embedding_outputs"] if "embedding_outputs" in arr.files else None

if S.ndim != 2:
    raise ValueError("Expected a 2D spectrogram in arr['s'].")

# Ensure S has shape (F, T)
if S.shape[1] != labels.shape[0] and S.shape[0] == labels.shape[0]:
    S = S.T

F, T = S.shape
if T != labels.shape[0]:
    raise ValueError("After orientation, spectrogram time (T) must match labels length.")

# =========================
# Decide which labels to run
# =========================
unique_labels = np.unique(labels)
if selected_labels is None:
    labels_to_process = [int(l) for l in unique_labels]
    if skip_noise_label and (-1 in labels_to_process):
        labels_to_process.remove(-1)
else:
    labels_to_process = [int(l) for l in selected_labels]

labels_to_process = sorted(labels_to_process)
if not labels_to_process:
    raise ValueError("No labels selected to process.")

# Prepare save directory if requested
saved_paths = []
outdir = None
if output_dir:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

# =========================
# 60-color qualitative palette (tab20 + tab20b + tab20c)
# =========================
def _get_tab60_palette() -> List[str]:
    tab20  = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    palette = [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]  # 60 colors
    return palette

def build_label_color_lut(all_labels: np.ndarray) -> Dict[int, str]:
    """
    Map each label -> hex color.
    - Noise (-1) is fixed to gray.
    - Non-noise labels use a 60-color palette (repeats if >60 labels).
    """
    uniq = sorted(np.unique(all_labels.astype(int)))
    non_noise = [l for l in uniq if l != -1]
    palette = _get_tab60_palette()

    if len(non_noise) > len(palette):
        print(f"[WARN] {len(non_noise)} non-noise labels exceed 60 colors; colors will repeat.")

    lut: Dict[int, str] = {-1: "#7f7f7f"}  # gray for noise
    for i, lab in enumerate(non_noise):
        lut[lab] = palette[i % len(palette)]
    return lut

label_color_lut = build_label_color_lut(labels)

# =========================
# UMAP plot (optional)
# =========================
def plot_umap_colored_by_labels(embedding: np.ndarray,
                                labels: np.ndarray,
                                lut: Dict[int, str],
                                outdir: Optional[Path],
                                show_plot: bool,
                                add_legend: bool):
    """Scatter UMAP colored by HDBSCAN labels. Saves to output_dir if provided."""
    # Use first two columns in case embedding has >2 dims
    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0] or embedding.shape[1] < 2:
        print("[WARN] UMAP plot skipped: embedding must be shape (T, 2+) matching labels length.")
        return
    colors = [lut[int(lab)] for lab in labels]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(embedding[:, 0], embedding[:, 1],
               c=colors, s=10, alpha=0.7, linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Embedding (HDBSCAN labels)")
    ax.grid(True, alpha=0.3)

    # --- Legend with ~14 rows (so it doesn't stretch the plot vertically) ---
    if add_legend:
        uniq = sorted(np.unique(labels.astype(int)))
        handles = [Patch(facecolor=lut[l], label=str(l)) for l in uniq]

        max_rows = 14                     # target rows per column
        ncols = int(np.ceil(len(handles) / max_rows)) or 1

        # Place legend to the right of the axes; do not expand plot height
        leg = ax.legend(handles=handles,
                        title="Label",
                        ncol=ncols,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),   # outside axes on the right
                        borderaxespad=0.0,
                        frameon=False,
                        fontsize=9,
                        title_fontsize=10,
                        handlelength=1.0,
                        handletextpad=0.4,
                        columnspacing=1.0,
                        labelspacing=0.4)

        # Optional: ensure there's a bit of right margin so the legend isn't clipped
        fig = ax.get_figure()
        fig.subplots_adjust(right=0.80)   # tweak if your legend is wider/narrower


    plt.tight_layout()

    if outdir is not None:
        save_path = outdir / "umap_hdbscan_labels.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# =========================
# Spectrogram generation (linear scaling)
# =========================
def generate_and_plot_for_label(lbl: int):
    """Generate up to N non-overlapping stitched spectrograms for a given label."""
    idx = np.flatnonzero(labels == lbl)  # sorted indices for this label
    if idx.size == 0:
        print(f"[WARN] No timebins for label {lbl}; skipping.")
        return

    max_full_samples = idx.size // spectrogram_length
    actual_samples = min(num_sample_spectrograms, max_full_samples)
    if actual_samples == 0:
        print(f"[WARN] Label {lbl}: only {idx.size} bins; need {spectrogram_length} for one sample. Skipping.")
        return

    print(f"[INFO] Label {lbl}: making {actual_samples} sample(s) "
          f"of {spectrogram_length} bins each (bins available: {idx.size}).")

    for k in range(actual_samples):
        start = k * spectrogram_length
        end   = start + spectrogram_length
        selected_idx = idx[start:end]          # non-overlapping slice
        S_sel = S[:, selected_idx].astype(float)  # linear units

        # Plot (linear)
        plt.figure(figsize=(10, 4))
        im = plt.imshow(S_sel, origin="lower", aspect="auto", cmap=cmap)
        if show_colorbar:
            plt.colorbar(im, label="Spectrogram (linear units)")

        ax = plt.gca()
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)

        plt.title(f"Label {lbl} — sample {k+1}/{actual_samples} "
                  f"({spectrogram_length} bins stitched)")
        plt.tight_layout()

        # Save directly to output_dir when requested (no per-label subfolders)
        if save_sample_spectrograms and outdir is not None:
            fname = f"label{lbl}_sample{k+1}_N{spectrogram_length}.png"
            save_path = outdir / fname
            plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
            saved_paths.append(str(save_path))
            print(f"[SAVE] {save_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

# ===========
# Run
# ===========
for lbl in labels_to_process:
    generate_and_plot_for_label(lbl)

# UMAP plot at the end (once)
if make_umap_plot:
    if embedding is None:
        print("[WARN] 'embedding_outputs' not found in NPZ; skipping UMAP plot.")
    else:
        plot_umap_colored_by_labels(embedding, labels, label_color_lut,
                                    outdir, show_plots, show_umap_legend)

# Optional: print all saved sample spectrogram files at the end
if saved_paths:
    print("\nAll saved sample spectrogram files:")
    for p in saved_paths:
        print("  ", p)
