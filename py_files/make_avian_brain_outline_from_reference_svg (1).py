#!/usr/bin/env python3
"""
make_avian_brain_outline_from_reference_svg.py

Create a black-and-white avian brain outline/segmentation figure from the
Birdbrain-style SVG reference, while removing labels, arrows, song-system
nucleus labels, and color fills.

Why this version works better than the hand-drawn coordinate version:
    It uses the actual anatomical region paths from the SVG reference, then
    converts them into a clean black-and-white outline. This keeps the
    telencephalon, cerebellum, thalamus, and brainstem proportions much closer
    to the reference image.

Recommended input:
    The original vector SVG file for the reference image, not a screenshot PNG.

Outputs:
    - editable SVG
    - PNG preview, if cairosvg is installed

Example:
    python make_avian_brain_outline_from_reference_svg.py \
      --input-svg ~/Downloads/Birdbrain.svg \
      --out-dir ~/Desktop/avian_brain_outline_outputs \
      --basename avian_brain_reference_bw \
      --line-width 3 \
      --png-width 1200

Optional dependency for PNG export:
    pip install cairosvg
"""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Set

SVG_NS = "http://www.w3.org/2000/svg"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
DC_NS = "http://purl.org/dc/elements/1.1/"
CC_NS = "http://web.resource.org/cc/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

ET.register_namespace("", SVG_NS)
ET.register_namespace("sodipodi", SODIPODI_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("cc", CC_NS)
ET.register_namespace("rdf", RDF_NS)

# These IDs correspond to anatomical region paths in the Birdbrain.svg-style
# reference used here. The script removes the labeled song nuclei and arrows.
#
# Keeping path3299 adds an extra outer/ventral outline from the source. I leave
# it off by default because it can create a double line along the top/edge when
# converted to black-and-white.
DEFAULT_ANATOMY_PATH_IDS: Set[str] = {
    "path3297",  # lower/brainstem-associated region
    "path3293",  # lower/brainstem-associated region
    "path3279",  # cerebellum gray region
    "path3277",  # cerebellar folia / subdivisions
    "path3213",  # small forebrain/anterior region
    "path4330",  # large telencephalon/pallium region
    "path4332",  # thalamus / hindbrain / ventral region
}

EXTRA_OUTLINE_PATH_IDS: Set[str] = {
    "path3299",
}


def style_string_to_dict(style: str) -> Dict[str, str]:
    """Convert an SVG style string into a dictionary."""
    out: Dict[str, str] = {}
    for part in style.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def style_dict_to_string(style_dict: Dict[str, str]) -> str:
    """Convert a dictionary into an SVG style string."""
    return ";".join(f"{key}:{value}" for key, value in style_dict.items())


def set_black_white_region_style(elem: ET.Element, line_width: float) -> None:
    """Make an anatomical region white-filled with a black outline."""
    style = style_string_to_dict(elem.attrib.get("style", ""))
    style.update(
        {
            "fill": "#ffffff",
            "fill-opacity": "1",
            "stroke": "#000000",
            "stroke-width": str(line_width),
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "stroke-opacity": "1",
        }
    )
    elem.attrib["style"] = style_dict_to_string(style)


def strip_to_anatomy_only(
    root: ET.Element,
    keep_path_ids: Set[str],
    line_width: float,
) -> None:
    """Remove labels/arrows/unwanted paths and restyle anatomy paths."""
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.split("}")[-1]
            elem_id = child.attrib.get("id", "")

            if tag == "path":
                if elem_id not in keep_path_ids:
                    parent.remove(child)
                else:
                    set_black_white_region_style(child, line_width=line_width)

            # Remove all text labels, including HVC/RA/LMAN/Area X/DLM labels.
            elif tag == "text":
                parent.remove(child)


def add_white_background(root: ET.Element) -> None:
    """Add an explicit white background rectangle for PNG previews."""
    rect = ET.Element(
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": "100%",
            "height": "100%",
            "fill": "white",
            "stroke": "none",
            "id": "background_white",
        },
    )
    root.insert(0, rect)
    root.attrib["style"] = "background-color:#ffffff"


def convert_svg_to_bw_outline(
    input_svg: Path,
    out_dir: Path,
    basename: str,
    line_width: float = 3.0,
    include_extra_outline: bool = False,
    export_png: bool = True,
    png_width: int = 1200,
) -> tuple[Path, Path | None]:
    """Create a black-and-white anatomy-only SVG and optionally a PNG."""
    input_svg = input_svg.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_svg.exists():
        raise FileNotFoundError(f"Could not find input SVG: {input_svg}")

    tree = ET.parse(input_svg)
    root = tree.getroot()

    keep_path_ids = set(DEFAULT_ANATOMY_PATH_IDS)
    if include_extra_outline:
        keep_path_ids |= EXTRA_OUTLINE_PATH_IDS

    strip_to_anatomy_only(root, keep_path_ids=keep_path_ids, line_width=line_width)
    add_white_background(root)

    svg_path = out_dir / f"{basename}.svg"
    tree.write(svg_path, encoding="UTF-8", xml_declaration=True)

    png_path: Path | None = None
    if export_png:
        png_path = out_dir / f"{basename}.png"
        try:
            import cairosvg  # type: ignore

            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=png_width,
                background_color="white",
            )
        except Exception as exc:  # pragma: no cover - helpful runtime message
            png_path = None
            print(
                "[WARN] SVG was saved, but PNG export failed. "
                "Install CairoSVG with `pip install cairosvg` if you want PNG export.\n"
                f"       Error: {exc}",
                file=sys.stderr,
            )

    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a black-and-white anatomy-only avian brain outline from "
            "the Birdbrain-style SVG reference."
        )
    )
    parser.add_argument(
        "--input-svg",
        required=True,
        help="Path to the original Birdbrain-style SVG reference file.",
    )
    parser.add_argument(
        "--out-dir",
        default="./avian_brain_outline_outputs",
        help="Directory where output SVG/PNG files will be saved.",
    )
    parser.add_argument(
        "--basename",
        default="avian_brain_reference_bw",
        help="Output base filename without extension.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=3.0,
        help="Black outline width in SVG source units. Try 2, 3, or 4.",
    )
    parser.add_argument(
        "--png-width",
        type=int,
        default=1200,
        help="Width of exported PNG preview in pixels.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only save SVG; do not try to export a PNG preview.",
    )
    parser.add_argument(
        "--include-extra-outline",
        action="store_true",
        help=(
            "Include an extra source outline path. This may add more detail, "
            "but can also create a double-line effect."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svg_path, png_path = convert_svg_to_bw_outline(
        input_svg=Path(args.input_svg),
        out_dir=Path(args.out_dir),
        basename=args.basename,
        line_width=args.line_width,
        include_extra_outline=args.include_extra_outline,
        export_png=not args.no_png,
        png_width=args.png_width,
    )

    print(f"Saved SVG: {svg_path}")
    if png_path is not None:
        print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
