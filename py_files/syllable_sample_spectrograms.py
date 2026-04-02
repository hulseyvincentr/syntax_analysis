#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
syllable_sample_spectrograms.py

Generate stitched spectrogram samples for one or many HDBSCAN labels (linear
scaling only), and optionally plot a UMAP scatter colored by HDBSCAN labels.

Public function
---------------
plot_spectrogram_samples_for_labels(
    npz_path,
    output_dir,
    selected_labels=None,
    skip_noise_label=True,
    spectrogram_length=1000,
    num_sample_spectrograms=1,
    cmap="gray_r",
    show_colorbar=False,
    show_plots=True,
    save_sample_spectrograms=True,
    make_umap_plot=True,
    show_umap_legend=True,
    save_umap_label_key=True,
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# =========================
# 60-color qualitative palette (tab20 + tab20b + tab20c)
# =========================
def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    palette = [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]
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

    lut: Dict[int, str] = {-1: "#7f7f7f"}
    for i, lab in enumerate(non_noise):
        lut[lab] = palette[i % len(palette)]
    return lut


# =========================
# Helper to infer animal ID
# =========================
def _infer_animal_id_from_path(p: Path) -> str:
    """
    Infer an animal ID from an NPZ path without assuming a 'USA####' pattern.

    Strategy
    --------
    1. Use everything before the first underscore in the file stem.
       Example: 'CanaryA_45424.5463_segment_0.npz' -> 'CanaryA'
    2. Also consider the full stem.
    3. Also consider the parent folder name.
    4. Return the first non-empty unique candidate.
    """
    candidates: List[str] = []

    stem = p.stem.strip()
    if stem:
        parts = stem.split("_")
        candidates.append(parts[0].strip())
        candidates.append(stem)

    parent_name = p.parent.name.strip()
    if parent_name:
        candidates.append(parent_name)
        parent_parts = parent_name.split("_")
        if parent_parts:
            candidates.append(parent_parts[0].strip())

    seen = set()
    unique_candidates: List[str] = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        unique_candidates.append(c)

    return unique_candidates[0] if unique_candidates else p.stem


# =========================
# Separate label-color key
# =========================
def save_label_color_key_png(
    labels: np.ndarray,
    lut: Dict[int, str],
    outdir: Path,
    *,
    animal_id: Optional[str] = None,
) -> None:
    """Save a separate PNG showing label -> color mapping."""
    uniq = sorted(np.unique(labels.astype(int)))
    handles = [Patch(facecolor=lut[l], edgecolor="none", label=str(l)) for l in uniq]

    max_rows = 28
    ncols = max(1, int(np.ceil(len(handles) / max_rows)))
    nrows = int(np.ceil(len(handles) / ncols))

    fig_w = 2.8 + 1.6 * ncols
    fig_h = max(2.0, 0.33 * nrows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if animal_id is not None:
        title = f"{animal_id} — label color key"
    else:
        title = "Label color key"

    ax.legend(
        handles=handles,
        title=title,
        ncol=ncols,
        loc="center",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.5,
    )

    save_path = outdir / "umap_hdbscan_labels_key.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[SAVE] {save_path}")


# =========================
# UMAP plot (optional)
# =========================
def plot_umap_colored_by_labels(
    embedding: np.ndarray,
    labels: np.ndarray,
    lut: Dict[int, str],
    outdir: Optional[Path],
    show_plot: bool,
    add_legend: bool,
    *,
    animal_id: Optional[str] = None,
) -> None:
    """Scatter UMAP colored by HDBSCAN labels. Saves to output_dir if provided."""
    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0] or embedding.shape[1] < 2:
        print("[WARN] UMAP plot skipped: embedding must be shape (T, 2+) matching labels length.")
        return

    colors = [lut[int(lab)] for lab in labels]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if animal_id is not None:
        ax.set_title(f"{animal_id} — UMAP embedding (HDBSCAN labels)")
    else:
        ax.set_title("UMAP embedding (HDBSCAN labels)")

    ax.grid(False)

    if add_legend:
        uniq = sorted(np.unique(labels.astype(int)))
        handles = [Patch(facecolor=lut[l], edgecolor="none", label=str(l)) for l in uniq]

        max_rows = 14
        ncols = int(np.ceil(len(handles) / max_rows)) or 1

        ax.legend(
            handles=handles,
            title="Label",
            ncol=ncols,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=1.0,
            labelspacing=0.4,
        )

        fig.subplots_adjust(right=0.80)

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
# Main public function
# =========================
def plot_spectrogram_samples_for_labels(
    npz_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    selected_labels: Optional[Sequence[int]] = None,
    skip_noise_label: bool = True,
    spectrogram_length: int = 1000,
    num_sample_spectrograms: int = 1,
    cmap: str = "gray_r",
    show_colorbar: bool = False,
    show_plots: bool = True,
    save_sample_spectrograms: bool = True,
    make_umap_plot: bool = True,
    show_umap_legend: bool = True,
    save_umap_label_key: bool = True,
) -> None:
    """
    Generate stitched spectrogram samples for HDBSCAN labels and optionally a
    UMAP plot colored by labels. Also optionally save a separate PNG key for
    label-to-color mapping.
    """
    npz_path = Path(npz_path)
    animal_id = _infer_animal_id_from_path(npz_path)
    arr = np.load(npz_path, allow_pickle=True)

    labels = arr["hdbscan_labels"]
    S = arr["s"]

    embedding = arr["embedding_outputs"] if "embedding_outputs" in arr.files else None

    if S.ndim != 2:
        raise ValueError("Expected a 2D spectrogram in arr['s'].")

    if S.shape[1] != labels.shape[0] and S.shape[0] == labels.shape[0]:
        S = S.T

    F, T = S.shape
    if T != labels.shape[0]:
        raise ValueError("After orientation, spectrogram time (T) must match labels length.")

    unique_labels = np.unique(labels)
    if selected_labels is None:
        labels_to_process = [int(l) for l in unique_labels]
        if skip_noise_label and (-1 in labels_to_process):
            labels_to_process.remove(-1)
    else:
        labels_to_process = [int(l) for l in selected_labels]

    labels_to_process = sorted(labels_to_process)
    if not labels_to_process:
        print("[WARN] No labels selected to process.")
        return

    saved_paths: List[str] = []
    outdir: Optional[Path] = None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    label_color_lut = build_label_color_lut(labels)

    def generate_and_plot_for_label(lbl: int) -> None:
        """Generate up to N non-overlapping stitched spectrograms for a given label."""
        idx = np.flatnonzero(labels == lbl)
        if idx.size == 0:
            print(f"[WARN] No timebins for label {lbl}; skipping.")
            return

        max_full_samples = idx.size // spectrogram_length
        actual_samples = min(num_sample_spectrograms, max_full_samples)
        if actual_samples == 0:
            print(
                f"[WARN] Label {lbl}: only {idx.size} bins; "
                f"need {spectrogram_length} for one sample. Skipping."
            )
            return

        print(
            f"[INFO] {animal_id} — label {lbl}: making {actual_samples} sample(s) "
            f"of {spectrogram_length} bins each (bins available: {idx.size})."
        )

        for k in range(actual_samples):
            start = k * spectrogram_length
            end = start + spectrogram_length
            selected_idx = idx[start:end]
            S_sel = S[:, selected_idx].astype(float)

            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(S_sel, origin="lower", aspect="auto", cmap=cmap)
            if show_colorbar:
                fig.colorbar(im, ax=ax, label="Spectrogram (linear units)")

            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

            ax.set_title(
                f"{animal_id} — label {lbl} — sample {k+1}/{actual_samples} "
                f"({spectrogram_length} bins stitched)"
            )
            fig.tight_layout()

            if save_sample_spectrograms and outdir is not None:
                fname = f"label{lbl}_sample{k+1}_N{spectrogram_length}.png"
                save_path = outdir / fname
                fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
                saved_paths.append(str(save_path))
                print(f"[SAVE] {save_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    for lbl in labels_to_process:
        generate_and_plot_for_label(lbl)

    if save_umap_label_key:
        if outdir is None:
            print("[WARN] output_dir is None; cannot save separate label-color key PNG.")
        else:
            save_label_color_key_png(
                labels=labels,
                lut=label_color_lut,
                outdir=outdir,
                animal_id=animal_id,
            )

    if make_umap_plot:
        if embedding is None:
            print("[WARN] 'embedding_outputs' not found in NPZ; skipping UMAP plot.")
        else:
            plot_umap_colored_by_labels(
                embedding=embedding,
                labels=labels,
                lut=label_color_lut,
                outdir=outdir,
                show_plot=show_plots,
                add_legend=show_umap_legend,
                animal_id=animal_id,
            )

    if saved_paths:
        print("\nAll saved sample spectrogram files:")
        for p in saved_paths:
            print("  ", p)


if __name__ == "__main__":
    example_npz = Path("/path/to/some_bird.npz")
    example_out = Path("/path/to/output_dir")
    print("[INFO] This is just a self-test block. Edit paths if you want to run this file directly.")
    # plot_spectrogram_samples_for_labels(example_npz, example_out)

    """
    from pathlib import Path
    import importlib
    import syllable_sample_spectrograms as sss

    importlib.reload(sss)

    npz_path = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz")
    output_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288_repertoire")

    sss.plot_spectrogram_samples_for_labels(
        npz_path=npz_path,
        output_dir=output_dir,
        selected_labels=None,
        skip_noise_label=True,
        spectrogram_length=1000,
        num_sample_spectrograms=1,
        cmap="gray_r",
        show_colorbar=False,
        show_plots=True,
        save_sample_spectrograms=True,
        make_umap_plot=True,
        show_umap_legend=True,
        save_umap_label_key=True,
    )
    """