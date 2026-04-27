"""
nonlinear_brightfield_contrast.py

Enhance low-contrast brightfield/Nissl histology images while preserving
more of the original stain color.

Features
--------
- Nonlinear sigmoid contrast enhancement
- Optional CLAHE local contrast enhancement
- Optional saturation boost to keep the image more purple / less grayscale
- Command-line arguments so you can tune settings from Mac Terminal

Example Mac terminal usage
--------------------------
cd ~/Documents/allPythonCode/syntax_analysis/py_files

python nonlinear_brightfield_contrast.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg

Stronger contrast:
python nonlinear_brightfield_contrast.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --sigmoid-gain 10 \
    --sigmoid-cutoff 0.52 \
    --clahe-clip-limit 1.5 \
    --saturation-boost 1.20

More purple / gentler:
python nonlinear_brightfield_contrast.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --sigmoid-gain 6 \
    --sigmoid-cutoff 0.50 \
    --clahe-clip-limit 1.0 \
    --saturation-boost 1.35
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sigmoid_contrast(x, gain=6, cutoff=0.50):
    """
    Nonlinear S-shaped contrast transform.

    Parameters
    ----------
    x : ndarray
        Image values normalized between 0 and 1.
    gain : float
        Higher values produce stronger contrast.
    cutoff : float
        Midpoint of the curve. Higher values darken the image overall;
        lower values lighten it overall.
    """
    y = 1 / (1 + np.exp(-gain * (x - cutoff)))

    y_min = 1 / (1 + np.exp(-gain * (0 - cutoff)))
    y_max = 1 / (1 + np.exp(-gain * (1 - cutoff)))
    y = (y - y_min) / (y_max - y_min)

    return np.clip(y, 0, 1)


def boost_saturation(rgb_image, saturation_boost=1.25):
    """
    Increase image saturation so the purple stain looks more natural.

    Parameters
    ----------
    rgb_image : ndarray
        RGB image.
    saturation_boost : float
        1.0 = no change, >1.0 increases saturation.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation_boost
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def enhance_brightfield_nonlinear(
    image_path,
    output_path=None,
    sigmoid_gain=6,
    sigmoid_cutoff=0.50,
    low_percentile=1,
    high_percentile=99.5,
    clahe_clip_limit=1.0,
    clahe_tile_grid_size=(8, 8),
    saturation_boost=1.25,
    show=True,
):
    """
    Enhance low-contrast brightfield histology image while preserving more
    of the original purple stain appearance.

    Steps
    -----
    1. Load the image.
    2. Convert to LAB color space.
    3. Apply CLAHE to the lightness channel.
    4. Apply percentile stretch.
    5. Apply nonlinear sigmoid contrast enhancement.
    6. Recombine with original color channels.
    7. Optionally boost saturation slightly.
    8. Save the enhanced image as a NEW file.

    Returns
    -------
    rgb_enhanced : ndarray
        Enhanced RGB image.
    """
    image_path = Path(image_path)

    if output_path is None:
        output_path = image_path.with_name(image_path.stem + "_nonlinear_contrast_purple.png")
    else:
        output_path = Path(output_path)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert to LAB so we can mainly alter brightness while preserving color.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Local nonlinear contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    L_clahe = clahe.apply(L)

    # Normalize to 0-1.
    L_float = L_clahe.astype(np.float32) / 255.0

    # Contrast stretch while ignoring extreme outlier pixels.
    low = np.percentile(L_float, low_percentile)
    high = np.percentile(L_float, high_percentile)

    if high <= low:
        raise ValueError(
            "Percentile stretch failed because high_percentile <= low_percentile. "
            "Try low_percentile=0 and high_percentile=100."
        )

    L_stretched = (L_float - low) / (high - low)
    L_stretched = np.clip(L_stretched, 0, 1)

    # Nonlinear sigmoid contrast enhancement.
    L_enhanced = sigmoid_contrast(
        L_stretched,
        gain=sigmoid_gain,
        cutoff=sigmoid_cutoff,
    )

    L_enhanced_uint8 = (L_enhanced * 255).astype(np.uint8)

    # Recombine with original color channels.
    lab_enhanced = cv2.merge([L_enhanced_uint8, A, B])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Restore some saturation so it stays more purple.
    rgb_enhanced = boost_saturation(
        rgb_enhanced,
        saturation_boost=saturation_boost,
    )

    # Save as a new file.
    bgr_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_enhanced)
    print(f"Saved enhanced image to: {output_path}")

    if show:
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(rgb_enhanced)
        plt.title("Enhanced")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return rgb_enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Enhance a brightfield histology image using nonlinear contrast."
    )

    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Default saves a new file next to the original image.",
    )
    parser.add_argument(
        "--sigmoid-gain",
        type=float,
        default=6,
        help="Contrast strength. Larger values = stronger contrast. Default: 6",
    )
    parser.add_argument(
        "--sigmoid-cutoff",
        type=float,
        default=0.50,
        help="Midpoint of the sigmoid. Higher = darker overall. Default: 0.50",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=1,
        help="Low percentile for contrast stretching. Default: 1",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.5,
        help="High percentile for contrast stretching. Default: 99.5",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=1.0,
        help="CLAHE local contrast strength. Higher can amplify noise. Default: 1.0",
    )
    parser.add_argument(
        "--clahe-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size. Default: 8 (used as 8x8)",
    )
    parser.add_argument(
        "--saturation-boost",
        type=float,
        default=1.25,
        help="Color saturation multiplier. 1.0 = no change, 1.25 = slightly more purple. Default: 1.25",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the side-by-side figure.",
    )

    args = parser.parse_args()

    enhance_brightfield_nonlinear(
        image_path=args.image_path,
        output_path=args.output_path,
        sigmoid_gain=args.sigmoid_gain,
        sigmoid_cutoff=args.sigmoid_cutoff,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=(args.clahe_grid_size, args.clahe_grid_size),
        saturation_boost=args.saturation_boost,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
