"""
nonlinear_brightfield_contrast.py

Enhance low-contrast brightfield/Nissl histology images by making darker
spots darker and lighter spots lighter using nonlinear contrast enhancement.

Recommended for Spyder/Jupyter:
    from nonlinear_brightfield_contrast import enhance_brightfield_nonlinear

    image_path = "/path/to/your/image.jpg"

    enhanced = enhance_brightfield_nonlinear(
        image_path=image_path,
        sigmoid_gain=10,
        sigmoid_cutoff=0.52,
        clahe_clip_limit=1.5,
        show=True
    )

Mac terminal example:
    python nonlinear_brightfield_contrast.py "/path/to/image.jpg"

Dependencies:
    conda install -c conda-forge opencv matplotlib numpy
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sigmoid_contrast(x, gain=8, cutoff=0.5):
    """
    Apply an S-shaped nonlinear contrast transform.

    Parameters
    ----------
    x : ndarray
        Image values normalized between 0 and 1.
    gain : float
        Higher values make the contrast stronger.
    cutoff : float
        Midpoint of the curve. Higher values darken the image overall;
        lower values lighten the image overall.
    """
    y = 1 / (1 + np.exp(-gain * (x - cutoff)))

    # Rescale output so the result still spans 0 to 1.
    y_min = 1 / (1 + np.exp(-gain * (0 - cutoff)))
    y_max = 1 / (1 + np.exp(-gain * (1 - cutoff)))
    y = (y - y_min) / (y_max - y_min)

    return np.clip(y, 0, 1)


def enhance_brightfield_nonlinear(
    image_path,
    output_path=None,
    sigmoid_gain=10,
    sigmoid_cutoff=0.52,
    low_percentile=1,
    high_percentile=99.5,
    clahe_clip_limit=1.5,
    clahe_tile_grid_size=(8, 8),
    show=True,
):
    """
    Enhance low-contrast brightfield histology images.

    This function:
    1. Loads the image.
    2. Converts the image to LAB color space.
    3. Enhances only the L/lightness channel.
    4. Applies CLAHE for local nonlinear contrast.
    5. Applies percentile stretching.
    6. Applies sigmoid nonlinear contrast enhancement.
    7. Saves the enhanced image.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.
    output_path : str, Path, or None
        Path to save the enhanced image. If None, saves next to the original
        with "_nonlinear_contrast.png" appended.
    sigmoid_gain : float
        Contrast strength. Try 6 for gentle, 10 for moderate, 14 for strong.
    sigmoid_cutoff : float
        Midpoint of the sigmoid. Try 0.45 to lighten, 0.50 neutral,
        0.55 to darken.
    low_percentile, high_percentile : float
        Percentile limits for contrast stretching. These help ignore extreme
        dust or edge artifacts.
    clahe_clip_limit : float
        Local contrast strength. Try 1.0 to 3.0. Higher can amplify noise.
    clahe_tile_grid_size : tuple
        Size of the local regions used for CLAHE.
    show : bool
        If True, show original and enhanced images side by side.

    Returns
    -------
    rgb_enhanced : ndarray
        Enhanced RGB image.
    """

    image_path = Path(image_path)

    if output_path is None:
        output_path = image_path.with_name(image_path.stem + "_nonlinear_contrast.png")
    else:
        output_path = Path(output_path)

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # LAB lets us enhance brightness/contrast while mostly preserving stain color.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # CLAHE is nonlinear/local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    L_clahe = clahe.apply(L)

    # Convert to 0-1.
    L_float = L_clahe.astype(np.float32) / 255.0

    # Percentile stretch ignores extreme outlier pixels.
    low = np.percentile(L_float, low_percentile)
    high = np.percentile(L_float, high_percentile)

    if high <= low:
        raise ValueError(
            "Percentile stretch failed because high percentile <= low percentile. "
            "Try low_percentile=0 and high_percentile=100."
        )

    L_stretched = (L_float - low) / (high - low)
    L_stretched = np.clip(L_stretched, 0, 1)

    # Nonlinear S-shaped contrast boost.
    L_enhanced = sigmoid_contrast(
        L_stretched,
        gain=sigmoid_gain,
        cutoff=sigmoid_cutoff,
    )

    L_enhanced_uint8 = (L_enhanced * 255).astype(np.uint8)

    lab_enhanced = cv2.merge([L_enhanced_uint8, A, B])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

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
        plt.title("Nonlinear contrast enhanced")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return rgb_enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Apply nonlinear contrast enhancement to a brightfield histology image."
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Default saves next to input image.",
    )
    parser.add_argument(
        "--sigmoid-gain",
        type=float,
        default=10,
        help="Contrast strength. Try 6 gentle, 10 moderate, 14 strong.",
    )
    parser.add_argument(
        "--sigmoid-cutoff",
        type=float,
        default=0.52,
        help="Midpoint. Lower lightens, higher darkens.",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=1.5,
        help="Local contrast strength. Try 1.0 to 3.0.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the comparison figure.",
    )

    args = parser.parse_args()

    enhance_brightfield_nonlinear(
        image_path=args.image_path,
        output_path=args.output_path,
        sigmoid_gain=args.sigmoid_gain,
        sigmoid_cutoff=args.sigmoid_cutoff,
        clahe_clip_limit=args.clahe_clip_limit,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
