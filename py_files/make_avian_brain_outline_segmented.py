#!/usr/bin/env python3
"""
make_avian_brain_outline_segmented.py

Generate a clean black-and-white avian brain outline schematic with added
segmentation in the thalamus/diencephalon and brainstem/hindbrain regions.

Outputs both PNG and SVG so the figure can be edited later in Illustrator,
Inkscape, PowerPoint, or other vector graphics software.

Example:
    python make_avian_brain_outline_segmented.py \
        --out-dir ./avian_brain_outline_outputs \
        --basename avian_brain_segmented_bw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

# Use a non-interactive backend so the script works from Terminal/SSH.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


def add_bezier_path(
    ax,
    pts,
    lw: float = 2.0,
    color: str = "black",
    closed: bool = False,
):
    """
    Draw a cubic Bezier path.

    pts format:
        [start,
         c1, c2, end,
         c1, c2, end,
         ...]

    Each segment after the start uses 3 points: control1, control2, endpoint.
    """
    if len(pts) < 4 or (len(pts) - 1) % 3 != 0:
        raise ValueError(
            "Bezier point list must be [start, c1, c2, end, c1, c2, end, ...]."
        )

    verts = [pts[0]]
    codes = [MplPath.MOVETO]

    for i in range(1, len(pts), 3):
        c1, c2, end = pts[i], pts[i + 1], pts[i + 2]
        verts.extend([c1, c2, end])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    if closed:
        verts.append((0, 0))  # ignored by CLOSEPOLY
        codes.append(MplPath.CLOSEPOLY)

    patch = PathPatch(
        MplPath(verts, codes),
        facecolor="none",
        edgecolor=color,
        lw=lw,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)
    return patch


def draw_cerebellum_lobe(
    ax,
    hub,
    theta_deg: float,
    length: float = 1.55,
    width: float = 0.60,
    lw: float = 2.0,
    color: str = "black",
):
    """Draw one rounded cerebellar lobe radiating from a hub point."""
    hx, hy = hub
    theta = np.deg2rad(theta_deg)

    # Unit vector pointing out from the hub.
    ux, uy = np.cos(theta), np.sin(theta)
    # Perpendicular vector for lobe width.
    px, py = -uy, ux

    start = (hx, hy)

    # One side of the lobe.
    c1 = (
        hx + 0.40 * length * ux + 0.50 * width * px,
        hy + 0.40 * length * uy + 0.50 * width * py,
    )
    c2 = (
        hx + 0.85 * length * ux + 0.55 * width * px,
        hy + 0.85 * length * uy + 0.55 * width * py,
    )
    tip = (hx + length * ux, hy + length * uy)

    # Return along the other side.
    c3 = (
        hx + 0.85 * length * ux - 0.55 * width * px,
        hy + 0.85 * length * uy - 0.55 * width * py,
    )
    c4 = (
        hx + 0.40 * length * ux - 0.50 * width * px,
        hy + 0.40 * length * uy - 0.50 * width * py,
    )
    end = (hx, hy)

    pts = [start, c1, c2, tip, c3, c4, end]
    add_bezier_path(ax, pts, lw=lw, color=color)


def make_segmented_avian_brain(
    out_dir: str | Path = ".",
    basename: str = "avian_brain_segmented_bw",
    line_width: float = 2.0,
    line_color: str = "black",
    dpi: int = 300,
    transparent: bool = False,
    figsize=(10, 7.5),
):
    """Generate the brain outline and save PNG + SVG."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none" if transparent else "white")

    # ==========================================================
    # 1) Cerebrum / telencephalon: outer outline only on top
    # ==========================================================
    cerebrum = [
        (3.35, 4.15),
        (3.55, 5.75), (4.65, 6.95), (6.20, 7.25),
        (7.65, 7.45), (9.45, 6.65), (10.20, 5.10),
        (10.75, 3.95), (10.95, 2.55), (10.25, 1.95),
        (9.55, 1.35), (7.75, 1.55), (6.15, 1.52),
        (4.95, 1.50), (3.95, 1.65), (3.35, 2.75),
        (3.00, 3.35), (3.00, 3.85), (3.35, 4.15),
    ]
    add_bezier_path(ax, cerebrum, lw=line_width, color=line_color)

    # Ventral/anterior underside contour under cerebrum.
    ventral_border = [
        (3.35, 2.75),
        (4.05, 2.25), (4.75, 2.15), (5.40, 2.35),
        (6.10, 2.55), (6.90, 2.95), (7.80, 2.85),
        (8.75, 2.78), (9.55, 2.35), (10.20, 1.95),
    ]
    add_bezier_path(ax, ventral_border, lw=line_width, color=line_color)

    # ==========================================================
    # 2) Cerebellum: uniform segmented lobes
    # ==========================================================
    cerebellum_hub = (2.45, 3.15)
    for angle in [102, 126, 150, 178, 208, 236]:
        draw_cerebellum_lobe(
            ax,
            hub=cerebellum_hub,
            theta_deg=angle,
            length=1.60,
            width=0.62,
            lw=line_width,
            color=line_color,
        )

    # Small connector from cerebellum toward hindbrain.
    add_bezier_path(
        ax,
        [
            (2.15, 2.20),
            (2.10, 2.55), (2.20, 2.95), (2.48, 3.15),
            (2.70, 3.28), (2.92, 3.10), (3.08, 2.80),
        ],
        lw=line_width,
        color=line_color,
    )

    # ==========================================================
    # 3) Thalamus / diencephalon: added segmentation
    # ==========================================================
    dorsal_thalamus = [
        (3.25, 2.55),
        (3.45, 3.10), (3.90, 3.35), (4.45, 3.20),
        (4.95, 3.05), (5.30, 2.65), (5.35, 2.15),
        (5.38, 1.78), (5.10, 1.40), (4.60, 1.30),
        (4.10, 1.22), (3.60, 1.40), (3.30, 1.75),
        (3.05, 2.02), (3.05, 2.28), (3.25, 2.55),
    ]
    add_bezier_path(ax, dorsal_thalamus, lw=line_width, color=line_color)

    ventral_thalamus = [
        (3.55, 1.55),
        (3.95, 1.15), (4.40, 0.98), (4.95, 1.03),
        (5.40, 1.07), (5.82, 1.28), (5.95, 1.62),
        (6.05, 1.92), (5.92, 2.18), (5.62, 2.30),
        (5.20, 2.42), (4.72, 2.30), (4.20, 2.05),
        (3.80, 1.88), (3.58, 1.75), (3.55, 1.55),
    ]
    add_bezier_path(ax, ventral_thalamus, lw=line_width, color=line_color)

    posterior_thalamus = [
        (5.22, 2.75),
        (5.45, 3.02), (5.78, 3.12), (6.10, 3.00),
        (6.35, 2.90), (6.52, 2.68), (6.48, 2.42),
        (6.42, 2.12), (6.18, 1.95), (5.88, 1.92),
        (5.58, 1.92), (5.32, 2.08), (5.20, 2.30),
        (5.10, 2.48), (5.10, 2.62), (5.22, 2.75),
    ]
    add_bezier_path(ax, posterior_thalamus, lw=line_width, color=line_color)

    # Internal thalamic subdivision lines.
    add_bezier_path(
        ax,
        [(3.45, 2.45), (3.90, 2.20), (4.55, 2.05), (5.15, 2.10)],
        lw=line_width,
        color=line_color,
    )
    add_bezier_path(
        ax,
        [(4.25, 3.10), (4.65, 2.85), (5.05, 2.70), (5.40, 2.35)],
        lw=line_width,
        color=line_color,
    )

    # ==========================================================
    # 4) Midbrain + hindbrain / brainstem segmentation
    # ==========================================================
    midbrain = [
        (2.72, 2.25),
        (2.92, 2.65), (3.18, 2.88), (3.55, 2.85),
        (3.88, 2.82), (4.02, 2.55), (4.00, 2.18),
        (3.98, 1.88), (3.78, 1.62), (3.48, 1.58),
        (3.15, 1.55), (2.88, 1.75), (2.75, 1.98),
        (2.66, 2.08), (2.66, 2.18), (2.72, 2.25),
    ]
    add_bezier_path(ax, midbrain, lw=line_width, color=line_color)

    pons = [
        (2.15, 1.72),
        (2.35, 2.08), (2.70, 2.18), (3.00, 2.02),
        (3.32, 1.86), (3.45, 1.52), (3.42, 1.18),
        (3.35, 0.88), (3.10, 0.62), (2.75, 0.55),
        (2.42, 0.48), (2.05, 0.60), (1.82, 0.82),
        (1.62, 1.02), (1.60, 1.30), (1.74, 1.52),
        (1.84, 1.65), (1.98, 1.72), (2.15, 1.72),
    ]
    add_bezier_path(ax, pons, lw=line_width, color=line_color)

    medulla = [
        (1.30, 1.38),
        (1.48, 1.72), (1.78, 1.84), (2.00, 1.72),
        (2.28, 1.58), (2.38, 1.28), (2.34, 0.95),
        (2.28, 0.72), (2.02, 0.48), (1.72, 0.40),
        (1.42, 0.32), (1.10, 0.40), (0.92, 0.58),
        (0.72, 0.80), (0.72, 1.10), (0.90, 1.25),
        (1.02, 1.35), (1.15, 1.40), (1.30, 1.38),
    ]
    add_bezier_path(ax, medulla, lw=line_width, color=line_color)

    spinal_cord = [
        (0.92, 0.95),
        (0.55, 0.85), (0.25, 0.58), (0.08, 0.18),
        (0.02, -0.02), (0.12, -0.16), (0.34, -0.20),
        (0.62, -0.28), (1.05, -0.10), (1.48, 0.10),
        (1.82, 0.26), (2.12, 0.36), (2.42, 0.34),
    ]
    add_bezier_path(ax, spinal_cord, lw=line_width, color=line_color)

    # Internal brainstem segmentation lines.
    add_bezier_path(
        ax,
        [(1.74, 1.52), (1.50, 1.38), (1.28, 1.25), (1.05, 1.05)],
        lw=line_width,
        color=line_color,
    )
    add_bezier_path(
        ax,
        [(2.35, 1.55), (2.10, 1.40), (1.88, 1.22), (1.66, 0.92)],
        lw=line_width,
        color=line_color,
    )

    # Boundary between hindbrain/cerebellum and cerebrum base.
    add_bezier_path(
        ax,
        [(2.95, 3.78), (2.82, 3.48), (2.72, 3.10), (2.70, 2.72)],
        lw=line_width,
        color=line_color,
    )

    # ==========================================================
    # Final formatting
    # ==========================================================
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 11.2)
    ax.set_ylim(-0.45, 7.8)
    ax.axis("off")
    plt.tight_layout(pad=0)

    png_path = out_dir / f"{basename}.png"
    svg_path = out_dir / f"{basename}.svg"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
    )
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    return png_path, svg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a black-and-white segmented avian brain outline schematic."
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output folder. Default: current directory.",
    )
    parser.add_argument(
        "--basename",
        default="avian_brain_segmented_bw",
        help="Base filename for PNG/SVG outputs. Default: avian_brain_segmented_bw",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width for outlines. Default: 2.0",
    )
    parser.add_argument(
        "--line-color",
        default="black",
        help="Line color. Default: black. Example: '#777777' for gray.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save with a transparent background instead of white.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_segmented_avian_brain(
        out_dir=args.out_dir,
        basename=args.basename,
        line_width=args.line_width,
        line_color=args.line_color,
        dpi=args.dpi,
        transparent=args.transparent,
    )
