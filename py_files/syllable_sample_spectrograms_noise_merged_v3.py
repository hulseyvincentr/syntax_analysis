#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
syllable_sample_spectrograms_noise_merged.py

Generate stitched spectrogram samples for one or many HDBSCAN labels and make
UMAP scatter plots colored by HDBSCAN labels.

Updates in this version
-----------------------
1. HDBSCAN label -1 is displayed as "Noise" in legends and titles.
2. The original UMAP is still saved with noise retained.
3. A second UMAP can be saved in which noise bins (-1) are reassigned to the
   closest non-noise syllable label to the left or right in time. This mirrors
   the decoder-prep logic described by your colleague.

Notes
-----
- This script does not compute UMAP or HDBSCAN. It expects them to already be
  saved in the NPZ as:
      arr["embedding_outputs"]
      arr["hdbscan_labels"]
- The spectrogram is expected at arr["s"].
"""

from __future__ import annotations

import argparse
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
    labels = np.asarray(all_labels).astype(int)
    uniq = sorted(np.unique(labels))
    non_noise = [lab for lab in uniq if lab != -1]
    palette = _get_tab60_palette()

    if len(non_noise) > len(palette):
        print(f"[WARN] {len(non_noise)} non-noise labels exceed 60 colors; colors will repeat.")

    lut: Dict[int, str] = {-1: "#7f7f7f"}
    for i, lab in enumerate(non_noise):
        lut[int(lab)] = palette[i % len(palette)]
    return lut


def label_display_name(label: int) -> str:
    """
    Display HDBSCAN noise label -1 as 'Noise' instead of '-1'.
    """
    label = int(label)
    if label == -1:
        return "Noise"
    return str(label)


def label_file_name(label: int) -> str:
    """
    File-safe label name used in saved spectrogram names.
    """
    label = int(label)
    if label == -1:
        return "noise"
    return f"label{label}"


def merge_noise_with_nearest_non_noise(labels: np.ndarray) -> np.ndarray:
    """
    Replace HDBSCAN noise labels (-1) with the nearest non-noise label
    to the left or right in time.

    This mirrors the decoder-prep logic described by your colleague:
    each -1 is reassigned to the closest neighboring non--1 syllable.

    Tie rule
    --------
    If the nearest left and right non-noise labels are equally close, use the
    left label. This matches the left-first search in the decoder snippet.

    Edge case
    ---------
    If all labels are -1, there is no non-noise syllable to assign, so the
    original labels are returned unchanged.
    """
    labels = np.asarray(labels).astype(int)
    cleaned = labels.copy()

    noise_mask = labels == -1
    if not np.any(noise_mask):
        print("[INFO] No -1/noise labels found; noise-merged labels are unchanged.")
        return cleaned

    non_noise_idx = np.flatnonzero(~noise_mask)
    if non_noise_idx.size == 0:
        print("[WARN] All labels are -1/noise; cannot merge noise with nearby syllables.")
        return cleaned

    n = labels.size

    # nearest non-noise index to the left of each position
    left_nearest = np.full(n, -1, dtype=int)
    left_nearest[non_noise_idx] = non_noise_idx
    left_nearest = np.maximum.accumulate(left_nearest)

    # nearest non-noise index to the right of each position
    right_nearest = np.full(n, n, dtype=int)
    right_nearest[non_noise_idx] = non_noise_idx
    right_nearest = np.minimum.accumulate(right_nearest[::-1])[::-1]

    noise_idx = np.flatnonzero(noise_mask)
    left_idx = left_nearest[noise_idx]
    right_idx = right_nearest[noise_idx]

    has_left = left_idx >= 0
    has_right = right_idx < n

    left_dist = noise_idx - left_idx
    right_dist = right_idx - noise_idx

    # Use left on ties, matching the left-first behavior in the decoder code.
    choose_left = has_left & (~has_right | (left_dist <= right_dist))
    choose_right = has_right & ~choose_left

    cleaned[noise_idx[choose_left]] = labels[left_idx[choose_left]]
    cleaned[noise_idx[choose_right]] = labels[right_idx[choose_right]]

    n_merged = int(np.sum(noise_mask & (cleaned != -1)))
    n_remaining = int(np.sum(cleaned == -1))
    print(
        f"[INFO] Merged {n_merged} noise bins with nearest non-noise syllable. "
        f"Remaining noise bins: {n_remaining}."
    )

    return cleaned


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
    save_name: str = "umap_hdbscan_labels_key.png",
    title_suffix: str = "label color key",
) -> None:
    """Save a separate PNG showing label -> color mapping.

    This version intentionally has no title so it can be dropped directly into a
    manuscript figure. Labels are arranged in two columns with larger fonts.
    """
    labels = np.asarray(labels).astype(int)
    uniq = sorted(np.unique(labels))
    handles = [
        Patch(facecolor=lut.get(int(lab), "#7f7f7f"), edgecolor="none", label=label_display_name(int(lab)))
        for lab in uniq
    ]

    ncols = 2 if len(handles) > 1 else 1
    nrows = int(np.ceil(len(handles) / ncols))

    fig_w = 4.6 if ncols == 2 else 2.8
    fig_h = max(2.0, 0.48 * nrows + 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    ax.legend(
        handles=handles,
        ncol=ncols,
        loc="center",
        frameon=False,
        fontsize=14,
        handlelength=1.3,
        handletextpad=0.6,
        columnspacing=1.8,
        labelspacing=0.8,
        borderaxespad=0.0,
    )

    save_path = outdir / save_name
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
    save_name: str = "umap_hdbscan_labels.png",
    title_suffix: str = "UMAP embedding (HDBSCAN labels)",
    legend_title: str = "Syllable Label",
    point_size: float = 10,
    point_alpha: float = 0.7,
) -> None:
    """Scatter UMAP colored by labels. Saves to output_dir if provided.

    This version is formatted for figure panels: no title, no axis labels,
    no tick labels, and no legend drawn on the UMAP itself.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).astype(int)

    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0] or embedding.shape[1] < 2:
        print("[WARN] UMAP plot skipped: embedding must be shape (T, 2+) matching labels length.")
        return

    colors = [lut.get(int(lab), "#7f7f7f") for lab in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=point_alpha,
        linewidths=0,
    )

    # Clean manuscript-ready styling.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.tight_layout()

    if outdir is not None:
        save_path = outdir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
    make_sample_spectrograms: bool = True,
    save_sample_spectrograms: bool = True,
    make_umap_plot: bool = True,
    show_umap_legend: bool = True,
    save_umap_label_key: bool = True,
    make_umap_noise_merged_plot: bool = True,
    save_umap_noise_merged_key: bool = True,
    point_size: float = 10,
    point_alpha: float = 0.7,
) -> None:
    """
    Generate stitched spectrogram samples for HDBSCAN labels and/or
    save two UMAP figures:

    1. Raw HDBSCAN labels, with -1 displayed as Noise.
    2. Noise-merged labels, where -1 bins are reassigned to the closest
       non-noise syllable in time.
    """
    npz_path = Path(npz_path)
    animal_id = _infer_animal_id_from_path(npz_path)
    arr = np.load(npz_path, allow_pickle=True)

    try:
        labels = np.asarray(arr["hdbscan_labels"]).astype(int)
        S = np.asarray(arr["s"])
        embedding = arr["embedding_outputs"] if "embedding_outputs" in arr.files else None

        if S.ndim != 2:
            raise ValueError("Expected a 2D spectrogram in arr['s'].")

        # Orient spectrogram so it is frequency x time.
        if S.shape[1] != labels.shape[0] and S.shape[0] == labels.shape[0]:
            S = S.T

        _F, T = S.shape
        if T != labels.shape[0]:
            raise ValueError("After orientation, spectrogram time (T) must match labels length.")

        unique_labels = np.unique(labels)
        if selected_labels is None:
            labels_to_process = [int(lab) for lab in unique_labels]
            if skip_noise_label and (-1 in labels_to_process):
                labels_to_process.remove(-1)
        else:
            labels_to_process = [int(lab) for lab in selected_labels]

        labels_to_process = sorted(labels_to_process)
        if make_sample_spectrograms and not labels_to_process:
            print("[WARN] No labels selected for sample spectrograms.")

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
                print(f"[WARN] No timebins for {label_display_name(lbl)}; skipping.")
                return

            max_full_samples = idx.size // spectrogram_length
            actual_samples = min(num_sample_spectrograms, max_full_samples)
            if actual_samples == 0:
                print(
                    f"[WARN] {label_display_name(lbl)}: only {idx.size} bins; "
                    f"need {spectrogram_length} for one sample. Skipping."
                )
                return

            print(
                f"[INFO] {animal_id} — {label_display_name(lbl)}: making {actual_samples} sample(s) "
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
                    f"{animal_id} — {label_display_name(lbl)} — sample {k + 1}/{actual_samples} "
                    f"({spectrogram_length} bins stitched)"
                )
                fig.tight_layout()

                if save_sample_spectrograms and outdir is not None:
                    fname = f"{label_file_name(lbl)}_sample{k + 1}_N{spectrogram_length}.png"
                    save_path = outdir / fname
                    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
                    saved_paths.append(str(save_path))
                    print(f"[SAVE] {save_path}")

                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)

        if make_sample_spectrograms:
            for lbl in labels_to_process:
                generate_and_plot_for_label(lbl)
        else:
            print("[INFO] Skipping stitched sample spectrogram generation.")

        # ---------------------------------------------------------
        # Original HDBSCAN UMAP: keep -1 as Noise
        # ---------------------------------------------------------
        if save_umap_label_key:
            if outdir is None:
                print("[WARN] output_dir is None; cannot save separate label-color key PNG.")
            else:
                save_label_color_key_png(
                    labels=labels,
                    lut=label_color_lut,
                    outdir=outdir,
                    animal_id=animal_id,
                    save_name="umap_hdbscan_labels_key.png",
                    title_suffix="label color key",
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
                    save_name="umap_hdbscan_labels.png",
                    title_suffix="UMAP embedding (HDBSCAN labels; noise retained)",
                    legend_title="Syllable Label",
                    point_size=point_size,
                    point_alpha=point_alpha,
                )

        # ---------------------------------------------------------
        # Noise-merged UMAP: replace -1 with nearest non-noise label
        # ---------------------------------------------------------
        if make_umap_noise_merged_plot:
            if embedding is None:
                print("[WARN] 'embedding_outputs' not found in NPZ; skipping noise-merged UMAP plot.")
            else:
                labels_noise_merged = merge_noise_with_nearest_non_noise(labels)

                if save_umap_noise_merged_key:
                    if outdir is None:
                        print("[WARN] output_dir is None; cannot save noise-merged label-color key PNG.")
                    else:
                        save_label_color_key_png(
                            labels=labels_noise_merged,
                            lut=label_color_lut,
                            outdir=outdir,
                            animal_id=animal_id,
                            save_name="umap_hdbscan_labels_noise_merged_key.png",
                            title_suffix="noise-merged label color key",
                        )

                plot_umap_colored_by_labels(
                    embedding=embedding,
                    labels=labels_noise_merged,
                    lut=label_color_lut,
                    outdir=outdir,
                    show_plot=show_plots,
                    add_legend=show_umap_legend,
                    animal_id=animal_id,
                    save_name="umap_hdbscan_labels_noise_merged.png",
                    title_suffix="UMAP embedding (noise merged with nearest syllable)",
                    legend_title="Syllable Label",
                    point_size=point_size,
                    point_alpha=point_alpha,
                )

        if saved_paths:
            print("\nAll saved sample spectrogram files:")
            for p in saved_paths:
                print("  ", p)

    finally:
        try:
            arr.close()
        except Exception:
            pass


# =========================
# Command-line interface
# =========================
def _parse_selected_labels(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip() != ""]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stitched spectrogram samples by HDBSCAN label and UMAP plots "
            "with -1 displayed as Noise. Optionally saves a second UMAP with noise "
            "merged into the nearest non-noise syllable label."
        )
    )

    parser.add_argument("--npz-path", required=True, help="Path to input .npz file")
    parser.add_argument("--output-dir", required=True, help="Directory where figures will be saved")

    parser.add_argument(
        "--selected-labels",
        default=None,
        help="Optional comma-separated labels to process, e.g. '2,3,17,19'. Use '-1' for Noise.",
    )
    parser.add_argument(
        "--include-noise-label",
        action="store_true",
        help="Include Noise/-1 when making stitched sample spectrograms. Default skips Noise.",
    )
    parser.add_argument("--spectrogram-length", type=int, default=1000, help="Number of bins per stitched sample")
    parser.add_argument("--num-sample-spectrograms", type=int, default=1, help="Number of samples per label")
    parser.add_argument("--cmap", default="gray_r", help="Matplotlib colormap for sample spectrograms")
    parser.add_argument("--show-colorbar", action="store_true", help="Show colorbar on sample spectrograms")

    parser.add_argument("--show-plots", dest="show_plots", action="store_true", help="Display plots interactively")
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false", help="Do not display plots")
    parser.set_defaults(show_plots=True)

    parser.add_argument(
        "--make-sample-spectrograms",
        dest="make_sample_spectrograms",
        action="store_true",
        help="Generate stitched sample spectrogram figures.",
    )
    parser.add_argument(
        "--no-sample-spectrograms",
        dest="make_sample_spectrograms",
        action="store_false",
        help="Skip stitched sample spectrogram generation and only make requested UMAP/key outputs.",
    )
    parser.set_defaults(make_sample_spectrograms=True)

    parser.add_argument(
        "--save-sample-spectrograms",
        dest="save_sample_spectrograms",
        action="store_true",
        help="Save stitched sample spectrograms",
    )
    parser.add_argument(
        "--no-save-sample-spectrograms",
        dest="save_sample_spectrograms",
        action="store_false",
        help="Do not save stitched sample spectrograms",
    )
    parser.set_defaults(save_sample_spectrograms=True)

    parser.add_argument("--make-umap-plot", dest="make_umap_plot", action="store_true")
    parser.add_argument("--no-umap-plot", dest="make_umap_plot", action="store_false")
    parser.set_defaults(make_umap_plot=True)

    parser.add_argument("--show-umap-legend", dest="show_umap_legend", action="store_true")
    parser.add_argument("--no-umap-legend", dest="show_umap_legend", action="store_false")
    parser.set_defaults(show_umap_legend=True)

    parser.add_argument("--save-umap-label-key", dest="save_umap_label_key", action="store_true")
    parser.add_argument("--no-umap-label-key", dest="save_umap_label_key", action="store_false")
    parser.set_defaults(save_umap_label_key=True)

    parser.add_argument("--make-noise-merged-umap", dest="make_umap_noise_merged_plot", action="store_true")
    parser.add_argument("--no-noise-merged-umap", dest="make_umap_noise_merged_plot", action="store_false")
    parser.set_defaults(make_umap_noise_merged_plot=True)

    parser.add_argument("--save-noise-merged-key", dest="save_umap_noise_merged_key", action="store_true")
    parser.add_argument("--no-noise-merged-key", dest="save_umap_noise_merged_key", action="store_false")
    parser.set_defaults(save_umap_noise_merged_key=True)

    parser.add_argument("--point-size", type=float, default=10, help="Point size for UMAP scatter")
    parser.add_argument("--point-alpha", type=float, default=0.7, help="Point alpha for UMAP scatter")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    selected_labels = _parse_selected_labels(args.selected_labels)

    plot_spectrogram_samples_for_labels(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        selected_labels=selected_labels,
        skip_noise_label=not args.include_noise_label,
        spectrogram_length=args.spectrogram_length,
        num_sample_spectrograms=args.num_sample_spectrograms,
        cmap=args.cmap,
        show_colorbar=args.show_colorbar,
        show_plots=args.show_plots,
        make_sample_spectrograms=args.make_sample_spectrograms,
        save_sample_spectrograms=args.save_sample_spectrograms,
        make_umap_plot=args.make_umap_plot,
        show_umap_legend=args.show_umap_legend,
        save_umap_label_key=args.save_umap_label_key,
        make_umap_noise_merged_plot=args.make_umap_noise_merged_plot,
        save_umap_noise_merged_key=args.save_umap_noise_merged_key,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
    )


if __name__ == "__main__":
    main()
