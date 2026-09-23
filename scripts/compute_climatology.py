#!/usr/bin/env python3
"""Compute a multi-year spatial climatology from training HDF5 files."""

import argparse
import os
import sys
import time

import h5py
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weaver.data.iterative_dataset import get_data_given_path
from weaver.data_preprocessing.variables import ALL_CHANNELS


def compute_climatology(data_dir: str, variables: list, max_files: int = None):
    """Return ``{variable: [H, W] mean field}`` for the training split."""
    file_paths = sorted(glob(os.path.join(data_dir, "*.h5")))
    if max_files:
        file_paths = file_paths[:max_files]

    n_files = len(file_paths)
    print(f"Found {n_files} training files in {data_dir}")
    if n_files == 0:
        raise FileNotFoundError(f"no HDF5 files found below {data_dir}")

    # Read the first file to determine the grid shape.
    print(f"Reading first file to get shape: {os.path.basename(file_paths[0])}")
    sample = get_data_given_path(file_paths[0], variables)  # [V, H, W]
    V, H, W = sample.shape
    print(f"  Variables: {V}, Grid: {H} x {W}")

    # Accumulate in float64 to reduce numerical error.
    accum = {var: np.zeros((H, W), dtype=np.float64) for var in variables}

    t_start = time.time()
    for i, fpath in enumerate(file_paths):
        data = get_data_given_path(fpath, variables)  # [V, H, W]
        data = data.astype(np.float64)
        for vi, var in enumerate(variables):
            accum[var] += data[vi]

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_files - i - 1)
            print(f"  Processed {i+1}/{n_files} files | elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)

    # Convert sums to means.
    clim_dict = {}
    for var in variables:
        clim_dict[var] = (accum[var] / n_files).astype(np.float32)

    elapsed = time.time() - t_start
    print(f"Done. {n_files} files processed in {elapsed:.0f}s ({elapsed/n_files:.2f}s per file)")

    return clim_dict


def main():
    parser = argparse.ArgumentParser(
        description="Compute climatology (multi-year mean) from training data"
    )
    parser.add_argument("--data-dir", default="data/h5/train",
                        help="Path to training data directory")
    parser.add_argument("--output", default="data/h5/clim.npz",
                        help="Output npz file path")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files (for quick test)")
    parser.add_argument("--variables", nargs="*", default=None,
                        help="Variables to include (default: the canonical 71 fields)")
    parser.add_argument("--variables-file", default=None,
                        help="Optional .npz archive whose keys define the variable order")
    args = parser.parse_args()

    # HDF5 and statistics commonly live in different directories in the
    # release.  Do not assume a normalize_mean.npz beside the HDF5 files.
    if args.variables is None:
        if args.variables_file:
            with np.load(args.variables_file) as archive:
                variables = list(archive.files)
            print(f"Loaded {len(variables)} variables from {args.variables_file}")
        else:
            variables = list(ALL_CHANNELS)
            print(f"Using canonical release variable order ({len(variables)} fields)")
    else:
        variables = args.variables

    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(f"Training data directory not found: {args.data_dir}")

    clim_dict = compute_climatology(args.data_dir, variables, args.max_files)

    # Save the compressed archive.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez_compressed(args.output, **clim_dict)
    print(f"Climatology saved to: {args.output}")

    # Print a short summary for verification.
    print("\nClimatology summary (spatial mean of each variable):")
    for var in variables[:10]:
        val = clim_dict[var].mean()
        print(f"  {var:<35} {val:.4f}")
    if len(variables) > 10:
        print(f"  ... and {len(variables) - 10} more variables")


if __name__ == "__main__":
    main()
