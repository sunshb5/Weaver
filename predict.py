#!/usr/bin/env python3
"""Generate paper-protocol autoregressive forecasts in physical units."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    load_model,
    resolve_config_path,
)
from weaver.utils.data_utils import CONSTANTS


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_INTERVALS = (6, 12, 24)


def load_statistics(directory: Path, variables, device):
    def ordered_archive(name):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"required statistics file not found: {path}")
        with np.load(path) as archive:
            missing = [variable for variable in variables if variable not in archive]
            if missing:
                raise KeyError(f"{path} is missing variables: {missing}")
            values = np.concatenate(
                [np.asarray(archive[variable]).reshape(-1) for variable in variables]
            ).astype(np.float32)
        if values.shape != (len(variables),):
            raise ValueError(f"{path}: expected one scalar per variable, got {values.shape}")
        return torch.from_numpy(values).view(1, -1, 1, 1).to(device)

    mean = ordered_archive("normalize_mean.npz")
    std = ordered_archive("normalize_std.npz")
    if torch.any(std <= 0):
        raise ValueError("normalize_std.npz contains a non-positive value")
    diff_std = {
        interval: ordered_archive(f"normalize_diff_std_{interval}.npz")
        for interval in BASE_INTERVALS
    }
    return mean, std, diff_std


def load_physical_input(path: Path, variable_count: int, device):
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3:
        array = array[None]
    expected = (variable_count, 128, 256)
    if array.ndim != 4 or tuple(array.shape[1:]) != expected:
        raise ValueError(
            f"input must have shape {expected} or (B, {expected[0]}, 128, 256); "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("input contains NaN or infinite values")
    return torch.from_numpy(array.astype(np.float32, copy=False)).to(device)


def rollout(model, initial_physical, variables, interval, steps, mean, std, diff_std):
    current = initial_physical.clone()
    interval_tensor = torch.full(
        (current.shape[0],), interval / 10.0, device=current.device, dtype=current.dtype
    )
    constant_indices = [
        index for index, variable in enumerate(variables) if variable in CONSTANTS
    ]
    for _ in range(steps):
        normalized = (current - mean) / std
        normalized_increment = model(normalized, variables, interval_tensor)
        if constant_indices:
            normalized_increment[:, constant_indices] = 0.0
        current = current + normalized_increment.float() * diff_std[interval]
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Physical-unit .npy field with the config variable order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lead-time", type=int, required=True,
                        help="Forecast horizon in hours")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--statistics-dir", type=Path, default=None,
        help="Statistics directory (default: assets/statistics beside --config)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA float16 autocast")
    parser.add_argument("--metadata", type=Path,
                        help="Optional JSON metadata output (default: OUTPUT.json)")
    args = parser.parse_args()

    if args.lead_time <= 0 or args.lead_time % 6:
        raise ValueError("lead-time must be a positive multiple of 6 hours")
    intervals = [
        interval for interval in BASE_INTERVALS if args.lead_time % interval == 0
    ]
    if not intervals:
        raise ValueError(f"no supported base interval divides {args.lead_time}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")

    device = torch.device(args.device)
    config = load_config(args.config)
    variables = config["data"]["variables"]
    model, loaded_variables = load_model(args.config, args.checkpoint, device)
    if list(loaded_variables) != list(variables):
        raise RuntimeError("model/config variable ordering mismatch")
    statistics_dir = args.statistics_dir or resolve_config_path(
        "assets/statistics", args.config
    )
    mean, std, diff_std = load_statistics(statistics_dir, variables, device)
    initial = load_physical_input(args.input, len(variables), device)

    forecasts = []
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and not args.no_amp,
        ):
            for interval in intervals:
                forecasts.append(
                    rollout(
                        model, initial, variables, interval,
                        args.lead_time // interval, mean, std, diff_std,
                    )
                )
    forecast = torch.stack(forecasts).mean(dim=0).float().cpu().numpy()
    if not np.isfinite(forecast).all():
        raise FloatingPointError("forecast contains NaN or infinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, forecast)

    metadata_path = args.metadata or Path(str(args.output) + ".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "lead_time_hours": args.lead_time,
        "base_intervals_hours": intervals,
        "shape": list(forecast.shape),
        "dtype": str(forecast.dtype),
        "variables": list(variables),
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"forecast saved to: {args.output}")
    print(f"metadata saved to: {metadata_path}")
    print(f"shape={forecast.shape}, intervals={intervals}, lead={args.lead_time}h")
    print(f"range=[{forecast.min():.6g}, {forecast.max():.6g}]")


if __name__ == "__main__":
    main()
