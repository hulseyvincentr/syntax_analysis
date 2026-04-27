"""
brightfield_purple_preserve.py

A gentler brightfield / Nissl histology enhancement script designed to preserve
a purple stain appearance rather than pushing the image toward black-and-white.

Why this version?
-----------------
The earlier script increased contrast too aggressively, which made tissue look
very dark and desaturated. This version:

1. Enhances only the lightness channel (LAB color space)
2. Uses gentler contrast enhancement
3. Blends the enhanced lightness back with the original lightness
   so the image does not get overly black
4. Slightly boosts saturation
5. Optionally nudges color slightly toward purple (magenta + blue)

Recommended default behavior:
- gentle contrast
- slightly brighter output
- more stain color preserved

Example terminal usage
----------------------
cd ~/Documents/allPythonCode/syntax_analysis/py_files

python brightfield_purple_preserve.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg

A bit more purple:
python brightfield_purple_preserve.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --saturation-boost 1.25 \
    --magenta-shift 4 \
    --blue-shift -4

If still too dark, make it gentler and brighter:
python brightfield_purple_preserve.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --contrast-strength 0.35 \
    --gamma 0.85 \
    --clahe-clip-limit 0.6 \
    --blend-original 0.35 \
    --saturation-boost 1.30 \
    --magenta-shift 5 \
    --blue-shift -5
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sigmoid_contrast(x, gain=3.0, cutoff=0.45):
    """
    Gentle S-shaped contrast transform for values in [0, 1].
    """
    y = 1.0 / (1.0 + np.exp(-gain * (x - cutoff)))

    y_min = 1.0 / (1.0 + np.exp(-gain * (0.0 - cutoff)))
    y_max = 1.0 / (1.0 + np.exp(-gain * (1.0 - cutoff)))

    y = (y - y_min) / (y_max - y_min)
    return np.clip(y, 0, 1)


def apply_gamma(x, gamma=0.9):
    """
    Gamma correction for normalized [0, 1] image.
    gamma < 1 brightens
    gamma > 1 darkens
    """
    x = np.clip(x, 0, 1)
    return np.power(x, gamma)


def boost_saturation(rgb_image, saturation_boost=1.20):
    """
    Slight saturation increase to preserve stain color.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_boost
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def gentle_lab_color_shift(rgb_image, magenta_shift=3, blue_shift=-3):
    """
    Nudge color slightly toward purple in LAB space.

    LAB:
    - A channel: green <-> magenta/red
      increasing A pushes toward magenta/red
    - B channel: blue <-> yellow
      decreasing B pushes toward blue
    """
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1] += magenta_shift
    lab[..., 2] += blue_shift
    lab[..., 1] = np.clip(lab[..., 1], 0, 255)
    lab[..., 2] = np.clip(lab[..., 2], 0, 255)
    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_brightfield_purple_preserve(
    image_path,
    output_path=None,
    contrast_strength=0.40,
    sigmoid_gain=3.0,
    sigmoid_cutoff=0.45,
    gamma=0.90,
    low_percentile=1.0,
    high_percentile=99.5,
    clahe_clip_limit=0.8,
    clahe_grid_size=8,
    blend_original=0.35,
    saturation_boost=1.20,
    magenta_shift=3,
    blue_shift=-3,
    show=True,
):
    """
    Enhance a low-contrast brightfield image while preserving a purple stain.

    Parameters
    ----------
    contrast_strength : float
        0 to 1. How much of the contrast-adjusted lightness to apply.
        Lower = gentler.
    sigmoid_gain : float
        Nonlinear contrast strength.
    sigmoid_cutoff : float
        Midpoint of sigmoid. Lower = brighter.
    gamma : float
        gamma < 1 brightens the result.
    clahe_clip_limit : float
        local contrast strength. Lower = gentler.
    blend_original : float
        0 to 1. How much of the ORIGINAL lightness to retain.
        Higher = more faithful to original, less black.
    saturation_boost : float
        1.0 = no change. >1 increases color.
    magenta_shift : int
        Positive values increase magenta.
    blue_shift : int
        Negative values increase blue.
    """
    image_path = Path(image_path)

    if output_path is None:
        output_path = image_path.with_name(image_path.stem + "_purple_preserved.png")
    else:
        output_path = Path(output_path)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_orig, A_orig, B_orig = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_grid_size, clahe_grid_size),
    )
    L_clahe = clahe.apply(L_orig)

    L_float = L_clahe.astype(np.float32) / 255.0
    low = np.percentile(L_float, low_percentile)
    high = np.percentile(L_float, high_percentile)

    if high <= low:
        low = 0.0
        high = 1.0

    L_stretched = (L_float - low) / (high - low)
    L_stretched = np.clip(L_stretched, 0, 1)

    L_sigmoid = sigmoid_contrast(
        L_stretched,
        gain=sigmoid_gain,
        cutoff=sigmoid_cutoff,
    )

    # Blend the enhanced lightness with the pre-enhanced lightness
    # so the tissue doesn't become too harsh or black.
    L_orig_float = L_orig.astype(np.float32) / 255.0
    L_mixed = ((1.0 - contrast_strength) * L_orig_float) + (contrast_strength * L_sigmoid)

    # Additional blending back toward original to preserve the original look.
    L_mixed = ((1.0 - blend_original) * L_mixed) + (blend_original * L_orig_float)

    # Brighten slightly so it does not look overly dark.
    L_final = apply_gamma(L_mixed, gamma=gamma)
    L_final = np.clip(L_final, 0, 1)
    L_final_uint8 = (L_final * 255).astype(np.uint8)

    lab_enhanced = cv2.merge([L_final_uint8, A_orig, B_orig])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Gently restore some stain color.
    rgb_enhanced = boost_saturation(
        rgb_enhanced,
        saturation_boost=saturation_boost,
    )
    rgb_enhanced = gentle_lab_color_shift(
        rgb_enhanced,
        magenta_shift=magenta_shift,
        blue_shift=blue_shift,
    )

    out_bgr = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out_bgr)
    print(f"Saved enhanced image to: {output_path}")

    if show:
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(rgb_enhanced)
        plt.title("Purple-preserved enhancement")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return rgb_enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Enhance a brightfield image while preserving purple stain appearance."
    )

    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Default saves next to original image.",
    )
    parser.add_argument(
        "--contrast-strength",
        type=float,
        default=0.40,
        help="0-1. Lower = gentler enhancement. Default: 0.40",
    )
    parser.add_argument(
        "--sigmoid-gain",
        type=float,
        default=3.0,
        help="Contrast curve strength. Default: 3.0",
    )
    parser.add_argument(
        "--sigmoid-cutoff",
        type=float,
        default=0.45,
        help="Lower = brighter overall. Default: 0.45",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        help="gamma < 1 brightens. Default: 0.90",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=1.0,
        help="Low percentile for stretch. Default: 1.0",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.5,
        help="High percentile for stretch. Default: 99.5",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=0.8,
        help="Lower = gentler local contrast. Default: 0.8",
    )
    parser.add_argument(
        "--clahe-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size. Default: 8",
    )
    parser.add_argument(
        "--blend-original",
        type=float,
        default=0.35,
        help="0-1. Higher = more original look, less black. Default: 0.35",
    )
    parser.add_argument(
        "--saturation-boost",
        type=float,
        default=1.20,
        help="1.0 = no change. >1 increases color. Default: 1.20",
    )
    parser.add_argument(
        "--magenta-shift",
        type=float,
        default=3,
        help="Positive values add magenta. Default: 3",
    )
    parser.add_argument(
        "--blue-shift",
        type=float,
        default=-3,
        help="Negative values add blue. Default: -3",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the comparison figure.",
    )

    args = parser.parse_args()

    enhance_brightfield_purple_preserve(
        image_path=args.image_path,
        output_path=args.output_path,
        contrast_strength=args.contrast_strength,
        sigmoid_gain=args.sigmoid_gain,
        sigmoid_cutoff=args.sigmoid_cutoff,
        gamma=args.gamma,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_grid_size=args.clahe_grid_size,
        blend_original=args.blend_original,
        saturation_boost=args.saturation_boost,
        magenta_shift=args.magenta_shift,
        blue_shift=args.blue_shift,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
