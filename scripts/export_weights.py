#!/usr/bin/env python3
"""Export a compact inference checkpoint from a Lightning training checkpoint."""

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Full Lightning training checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="New weights-only checkpoint (must differ from input)")
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("refusing to overwrite the training checkpoint in place")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("input is not a Lightning checkpoint with state_dict")
    exported = {"state_dict": checkpoint["state_dict"]}
    if "hyper_parameters" in checkpoint:
        exported["hyper_parameters"] = checkpoint["hyper_parameters"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(exported, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if set(verified) - {"state_dict", "hyper_parameters"}:
            raise RuntimeError(f"unexpected exported keys: {sorted(verified)}")
        if set(verified["state_dict"]) != set(checkpoint["state_dict"]):
            raise RuntimeError("exported state_dict keys differ from the input")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    verified = torch.load(destination, map_location="cpu", weights_only=True)
    if set(verified) - {"state_dict", "hyper_parameters"}:
        raise RuntimeError(f"unexpected exported keys: {sorted(verified)}")
    if set(verified["state_dict"]) != set(checkpoint["state_dict"]):
        raise RuntimeError("exported state_dict keys differ from the input")
    print(f"exported {len(verified['state_dict'])} tensors to {destination}")
    print(f"size: {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
