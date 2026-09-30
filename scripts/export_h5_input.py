#!/usr/bin/env python3
"""Export one physical-unit model input from a WEAVER HDF5 timestep."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/weaver.yaml")
    args = parser.parse_args()

    with args.config.open() as handle:
        variables = yaml.safe_load(handle)["data"]["variables"]
    with h5py.File(args.h5, "r") as handle:
        if "input" not in handle:
            raise KeyError(f"{args.h5} has no 'input' group")
        missing = [name for name in variables if name not in handle["input"]]
        if missing:
            raise KeyError(f"{args.h5} is missing variables: {missing}")
        array = np.stack(
            [np.asarray(handle["input"][name], dtype=np.float32) for name in variables]
        )
    if array.shape != (len(variables), 128, 256):
        raise ValueError(f"expected {(len(variables), 128, 256)}, got {array.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    print(f"saved {array.shape} physical input to {args.output}")


if __name__ == "__main__":
    main()
