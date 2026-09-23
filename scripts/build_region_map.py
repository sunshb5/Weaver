#!/usr/bin/env python3
"""Rebuild the fixed RP-MoE region map from training-period climatology."""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIM = PROJECT_ROOT / "assets/statistics/clim.npz"
DEFAULT_MAP = PROJECT_ROOT / "assets/region_maps/region_labels_era5_agglomerative_patch2.npy"
DEFAULT_BIAS = PROJECT_ROOT / "assets/region_maps/region_to_expert_prior_init_era5_agglomerative_patch2_climv2.npy"


def load_patch_features(path: Path, height: int, width: int, patch_size: int):
    """Return per-patch temporal-climatology mean/std descriptors."""
    with np.load(path) as archive:
        keys = sorted(archive.files)
        if len(keys) != 71:
            raise ValueError(f"expected 71 climatology fields, found {len(keys)}")
        patch_h, patch_w = height // patch_size, width // patch_size
        means, stds = [], []
        for key in keys:
            field = np.asarray(archive[key], dtype=np.float64).squeeze()
            if field.shape != (height, width):
                raise ValueError(f"{key}: expected {(height, width)}, got {field.shape}")
            patches = field.reshape(patch_h, patch_size, patch_w, patch_size)
            means.append(patches.mean(axis=(1, 3)).ravel())
            stds.append(patches.std(axis=(1, 3)).ravel())
    return np.stack(means + stds, axis=1)


def grid_connectivity(height: int, width: int):
    """Four-neighbour patch connectivity with periodic longitude."""
    rows, cols = [], []
    for row in range(height):
        for col in range(width):
            index = row * width + col
            neighbours = [
                (row, (col - 1) % width),
                (row, (col + 1) % width),
            ]
            if row:
                neighbours.append((row - 1, col))
            if row + 1 < height:
                neighbours.append((row + 1, col))
            for neighbour_row, neighbour_col in neighbours:
                rows.append(index)
                cols.append(neighbour_row * width + neighbour_col)
    values = np.ones(len(rows), dtype=np.uint8)
    return sparse.csr_matrix((values, (rows, cols)), shape=(height * width,) * 2)


def build_region_artifacts(features, patch_h: int, patch_w: int, regions: int):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    standardized = (features - mean) / np.maximum(std, 1e-8)

    clustering = AgglomerativeClustering(
        n_clusters=regions,
        linkage="ward",
        connectivity=grid_connectivity(patch_h, patch_w),
        compute_full_tree=True,
    )
    labels = clustering.fit_predict(standardized)

    # Give stable north-to-south region IDs; longitude remains periodic.
    latitude = np.linspace(90.0, -90.0, patch_h).repeat(patch_w)
    north_to_south = np.argsort(
        [-latitude[labels == region].mean() for region in range(regions)]
    )
    remap = np.empty(regions, dtype=np.int64)
    remap[north_to_south] = np.arange(regions)
    labels = remap[labels]

    region_means = np.stack(
        [standardized[labels == region].mean(axis=0) for region in range(regions)]
    )
    normalized = region_means / np.maximum(
        np.linalg.norm(region_means, axis=1, keepdims=True), 1e-8
    )
    region_bias = normalized @ normalized.T
    return labels.astype(np.int32), region_bias.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clim", type=Path, default=DEFAULT_CLIM)
    parser.add_argument("--output-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-bias", type=Path, default=DEFAULT_BIAS)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--regions", type=int, default=9)
    args = parser.parse_args()

    if args.height % args.patch_size or args.width % args.patch_size:
        raise ValueError("height and width must be divisible by patch-size")
    features = load_patch_features(
        args.clim, args.height, args.width, args.patch_size
    )
    labels, bias = build_region_artifacts(
        features,
        args.height // args.patch_size,
        args.width // args.patch_size,
        args.regions,
    )
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_bias.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_map, labels)
    np.save(args.output_bias, bias)
    print(f"region map: {args.output_map} shape={labels.shape}")
    print(f"region counts: {np.bincount(labels, minlength=args.regions).tolist()}")
    print(f"region bias: {args.output_bias} shape={bias.shape}")


if __name__ == "__main__":
    main()
