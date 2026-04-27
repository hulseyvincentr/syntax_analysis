"""
brightfield_purple_preserve_bgwhiten.py

Enhance brightfield / Nissl histology images while preserving a purple stain
appearance and optionally whitening the bright background selectively.

Key features
------------
- Gentle contrast enhancement on the lightness channel
- Purple-preserving color handling
- Optional saturation boost
- Optional magenta / blue shift to keep the Nissl stain purple
- NEW: selective background whitening, aimed mostly at the bright background
  rather than the darker tissue

Example terminal usage
----------------------
cd ~/Documents/allPythonCode/syntax_analysis/py_files

# Basic usage
python brightfield_purple_preserve_bgwhiten.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg

# Whiter background, gentle contrast, preserve purple
python brightfield_purple_preserve_bgwhiten.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --contrast-strength 0.45 \
    --sigmoid-cutoff 0.42 \
    --gamma 0.82 \
    --clahe-clip-limit 0.9 \
    --blend-original 0.25 \
    --saturation-boost 1.25 \
    --magenta-shift 4 \
    --blue-shift -4 \
    --background-whiten 0.18

# Stronger background whitening
python brightfield_purple_preserve_bgwhiten.py \
    ~/Desktop/USA5483_042925.01_Nissl_5x_sect81.1_upper.jpg \
    --background-whiten 0.25 \
    --background-threshold 0.72 \
    --background-softness 0.08
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
    Gamma correction for normalized [0, 1] values.
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
    Increasing A adds magenta.
    Decreasing B adds blue.
    """
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1] += magenta_shift
    lab[..., 2] += blue_shift
    lab[..., 1] = np.clip(lab[..., 1], 0, 255)
    lab[..., 2] = np.clip(lab[..., 2], 0, 255)
    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def selective_background_whiten(
    rgb_image,
    whiten_strength=0.15,
    threshold=0.75,
    softness=0.10,
):
    """
    Selectively whiten brighter parts of the image.

    This is designed to affect the light background much more than the darker tissue.

    Parameters
    ----------
    rgb_image : ndarray
        RGB image.
    whiten_strength : float
        0 means no whitening. Typical useful range is 0.05 to 0.30.
    threshold : float
        Brightness threshold in [0, 1]. Pixels brighter than this are whitened more.
        Lower threshold affects more of the image.
    softness : float
        Controls how gradually whitening transitions around the threshold.
        Larger values = softer transition.
    """
    if whiten_strength <= 0:
        return rgb_image

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0

    # Smooth mask: near 0 for dark pixels, near 1 for bright pixels.
    mask = 1.0 / (1.0 + np.exp(-(L - threshold) / max(softness, 1e-6)))

    # Push bright regions toward white.
    # Lightness goes toward 1.0.
    L_new = L + whiten_strength * mask * (1.0 - L)

    # Reduce chroma slightly in the bright background so it looks whiter.
    A = lab[..., 1]
    B = lab[..., 2]
    neutral = 128.0
    chroma_reduce = whiten_strength * mask * 0.5
    A_new = A + chroma_reduce * (neutral - A)
    B_new = B + chroma_reduce * (neutral - B)

    lab[..., 0] = np.clip(L_new * 255.0, 0, 255)
    lab[..., 1] = np.clip(A_new, 0, 255)
    lab[..., 2] = np.clip(B_new, 0, 255)

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
    background_whiten=0.0,
    background_threshold=0.75,
    background_softness=0.10,
    show=True,
):
    """
    Enhance a low-contrast brightfield image while preserving a purple stain.

    Parameters
    ----------
    contrast_strength : float
        0 to 1. Lower = gentler.
    sigmoid_gain : float
        Nonlinear contrast strength.
    sigmoid_cutoff : float
        Lower = brighter.
    gamma : float
        gamma < 1 brightens.
    clahe_clip_limit : float
        Lower = gentler local contrast.
    blend_original : float
        0 to 1. Higher = more faithful to original, less black.
    saturation_boost : float
        >1 increases saturation.
    magenta_shift : float
        Positive adds magenta.
    blue_shift : float
        Negative adds blue.
    background_whiten : float
        0 = no selective whitening. Higher values whiten bright background more.
    background_threshold : float
        Brightness threshold for background whitening.
    background_softness : float
        Transition softness for background whitening.
    """
    image_path = Path(image_path)

    if output_path is None:
        output_path = image_path.with_name(image_path.stem + "_purple_preserved_bgwhiten.png")
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

    # Mix enhanced lightness with original to avoid overly harsh / black output.
    L_orig_float = L_orig.astype(np.float32) / 255.0
    L_mixed = ((1.0 - contrast_strength) * L_orig_float) + (contrast_strength * L_sigmoid)
    L_mixed = ((1.0 - blend_original) * L_mixed) + (blend_original * L_orig_float)

    # Brighten slightly if gamma < 1
    L_final = apply_gamma(L_mixed, gamma=gamma)
    L_final = np.clip(L_final, 0, 1)
    L_final_uint8 = (L_final * 255).astype(np.uint8)

    lab_enhanced = cv2.merge([L_final_uint8, A_orig, B_orig])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Preserve / nudge stain color.
    rgb_enhanced = boost_saturation(
        rgb_enhanced,
        saturation_boost=saturation_boost,
    )
    rgb_enhanced = gentle_lab_color_shift(
        rgb_enhanced,
        magenta_shift=magenta_shift,
        blue_shift=blue_shift,
    )

    # NEW: selectively whiten the bright background.
    rgb_enhanced = selective_background_whiten(
        rgb_enhanced,
        whiten_strength=background_whiten,
        threshold=background_threshold,
        softness=background_softness,
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
        help="Optional output path. Default saves next to the original image.",
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
        "--background-whiten",
        type=float,
        default=0.0,
        help="0 = no selective whitening. Try 0.10 to 0.25. Default: 0.0",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.75,
        help="Brightness threshold for selective background whitening. Lower affects more pixels. Default: 0.75",
    )
    parser.add_argument(
        "--background-softness",
        type=float,
        default=0.10,
        help="Softness of the transition for background whitening. Default: 0.10",
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
        background_whiten=args.background_whiten,
        background_threshold=args.background_threshold,
        background_softness=args.background_softness,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
