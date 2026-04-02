#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def try_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None


def reduce_to_2d(X, method="umap", random_state=42):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array for dimensionality reduction, got shape {X.shape}")

    if X.shape[1] == 2:
        return X

    method = method.lower()

    if method == "umap":
        umap_mod = try_import_umap()
        if umap_mod is not None:
            reducer = umap_mod.UMAP(
                n_components=2,
                random_state=random_state,
            )
            return reducer.fit_transform(X)

        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
        return reducer.fit_transform(X)

    raise ValueError(f"Unsupported method: {method}")


def load_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def get_optional_array(data, key):
    return data[key] if key in data.files else None


def get_file_map(data):
    if "file_map" not in data.files:
        return None
    try:
        return data["file_map"].item()
    except Exception:
        return data["file_map"]


def apply_mask_to_arrays(mask, arrays_dict):
    out = {}
    for key, value in arrays_dict.items():
        if value is None:
            out[key] = None
            continue
        arr = np.asarray(value)
        if arr.shape[0] == len(mask):
            out[key] = arr[mask]
        else:
            out[key] = value
    return out


def choose_labels(data, label_key):
    if label_key is None:
        if "ground_truth_labels" in data.files:
            label_key = "ground_truth_labels"
        elif "hdbscan_labels" in data.files:
            label_key = "hdbscan_labels"

    labels = data[label_key] if label_key is not None and label_key in data.files else None
    return label_key, labels


def scatter_panel(ax, coords, labels=None, title="", point_size=2):
    coords = np.asarray(coords)

    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=point_size)
    else:
        labels = np.asarray(labels)
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=point_size)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the saved TweetyBERT UMAP and a 2D reduction of a decoder-related array."
    )
    parser.add_argument("--npz-path", required=True, help="Path to TweetyBERT NPZ file")
    parser.add_argument(
        "--umap-key",
        default="embedding_outputs",
        help="Key for the already-saved 2D UMAP coordinates"
    )
    parser.add_argument(
        "--decoder-key",
        default="predictions",
        help="Key for the decoder-related or latent-space array to reduce to 2D"
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="Optional label key for coloring points, e.g. ground_truth_labels or hdbscan_labels"
    )
    parser.add_argument(
        "--method",
        default="umap",
        choices=["umap", "pca", "tsne"],
        help="Method used to reduce the decoder array to 2D"
    )
    parser.add_argument(
        "--vocal-only",
        action="store_true",
        help="If set, keep only time bins where vocalization == 1"
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Scatter point size"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of points to plot and reduce"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional path to save the figure, e.g. /path/to/umap_vs_decoder.png"
    )
    args = parser.parse_args()

    data = load_npz(args.npz_path)

    print("Available keys:")
    print(data.files)
    print()

    if args.umap_key not in data.files:
        raise KeyError(
            f'UMAP key "{args.umap_key}" was not found in the NPZ. '
            f"Available keys: {data.files}"
        )

    if args.decoder_key not in data.files:
        raise KeyError(
            f'Decoder key "{args.decoder_key}" was not found in the NPZ. '
            f"Available keys: {data.files}"
        )

    umap_coords = data[args.umap_key]
    decoder_array = data[args.decoder_key]
    label_key, labels = choose_labels(data, args.label_key)
    vocalization = get_optional_array(data, "vocalization")
    file_indices = get_optional_array(data, "file_indices")
    file_map = get_file_map(data)

    print(f"Using UMAP key: {args.umap_key} with shape {np.shape(umap_coords)}")
    print(f"Using decoder key: {args.decoder_key} with shape {np.shape(decoder_array)}")
    if label_key is not None and labels is not None:
        print(f"Using label key: {label_key} with shape {np.shape(labels)}")
    else:
        print("No label key found or provided; plotting without label colors.")

    arrays = {
        "umap_coords": np.asarray(umap_coords),
        "decoder_array": np.asarray(decoder_array),
        "labels": None if labels is None else np.asarray(labels),
        "vocalization": None if vocalization is None else np.asarray(vocalization),
        "file_indices": None if file_indices is None else np.asarray(file_indices),
    }

    if args.vocal_only:
        if arrays["vocalization"] is None:
            raise ValueError("Requested --vocal-only, but no vocalization array exists in the NPZ.")
        mask = arrays["vocalization"].astype(bool)
        arrays = apply_mask_to_arrays(mask, arrays)
        print(f"Applied vocal-only mask. Kept {int(mask.sum())} of {len(mask)} points.")

    n_points = arrays["umap_coords"].shape[0]

    if args.max_points is not None and args.max_points < n_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(n_points, size=args.max_points, replace=False))
        arrays = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[0] == n_points else v)
                  for k, v in arrays.items()}
        print(f"Subsampled to {args.max_points} points.")

    decoder_2d = reduce_to_2d(arrays["decoder_array"], method=args.method, random_state=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter_panel(
        axes[0],
        arrays["umap_coords"],
        labels=arrays["labels"],
        title=f"Saved UMAP: {args.umap_key}",
        point_size=args.point_size,
    )

    scatter_panel(
        axes[1],
        decoder_2d,
        labels=arrays["labels"],
        title=f"{args.decoder_key} reduced with {args.method.upper()}",
        point_size=args.point_size,
    )

    plt.tight_layout()

    if args.out_path:
        out_dir = os.path.dirname(args.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {args.out_path}")

    plt.show()

    if arrays["file_indices"] is not None and file_map is not None:
        print()
        print("First 5 source files:")
        for i in range(min(5, len(arrays["file_indices"]))):
            idx = int(arrays["file_indices"][i])
            try:
                file_name = file_map.get(idx, "unknown")
            except Exception:
                file_name = "unknown"
            print(i, "->", file_name)


if __name__ == "__main__":
    main()
