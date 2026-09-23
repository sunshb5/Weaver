#!/usr/bin/env python3
"""Run one-step inference with any WEAVER checkpoint/config pair."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from weaver.utils.data_utils import CONSTANTS
from weaver.utils.release_paths import default_checkpoint_path, default_config_path, release_root


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
RELEASE_ROOT = release_root()
DEFAULT_CONFIG = default_config_path()
DEFAULT_CHECKPOINT = default_checkpoint_path()


def resolve_config_path(value, config_path):
    """Resolve a config-relative path with a project-root compatibility fallback.

    The release YAML historically stores ``assets/...`` relative to the
    repository root, while a self-contained custom config may keep assets
    beside the YAML. Prefer the latter when it exists and retain the former
    for the bundled config.
    """

    path = Path(value)
    if path.is_absolute():
        return path
    config_dir = Path(config_path).resolve().parent
    beside_config = config_dir / path
    if beside_config.exists():
        return beside_config
    project_path = RELEASE_ROOT / path
    if project_path.exists():
        return project_path
    if config_dir.name == "configs":
        return project_path
    return beside_config


def load_config(path):
    with path.open() as handle:
        config = yaml.safe_load(handle)

    def resolve(value, root):
        if isinstance(value, dict):
            return {key: resolve(item, root) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, root) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            resolved = root
            for key in value[2:-1].split("."):
                resolved = resolved[key]
            return resolved
        return value

    return resolve(config, config)


def load_model(config_path, checkpoint_path, device):
    config = load_config(config_path)
    net_config = config["model"]["net"]
    for key in ("region_map_path", "region_prior_init_path"):
        value = net_config["init_args"].get(key)
        if value and not Path(value).is_absolute():
            net_config["init_args"][key] = str(resolve_config_path(value, config_path))
    module_name, class_name = net_config["class_path"].rsplit(".", 1)
    net_class = getattr(importlib.import_module(module_name), class_name)
    net = net_class(**net_config["init_args"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    net_state_dict = {}
    for name, value in state_dict.items():
        if name.startswith("net."):
            normalized_name = name[4:]
        elif name.startswith("model.net."):
            normalized_name = name[10:]
        else:
            normalized_name = name
        if normalized_name in net_state_dict:
            raise RuntimeError(
                f"checkpoint contains duplicate parameters after prefix normalization: "
                f"{normalized_name!r}"
            )
        net_state_dict[normalized_name] = value

    missing, unexpected = net.load_state_dict(net_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return net.to(device).eval(), config["data"]["variables"]


def load_input(path, variables, device):
    if path is None:
        return torch.randn(1, len(variables), 128, 256, device=device, dtype=torch.float32)
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3:
        array = array[None, ...]
    expected = (len(variables), 128, 256)
    if array.ndim != 4 or tuple(array.shape[1:]) != expected:
        raise ValueError(f"input must have shape (B, {expected[0]}, 128, 256), got {array.shape}")
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, help="Normalized input .npy, shape (71,128,256) or (B,71,128,256).")
    parser.add_argument("--output", type=Path, default=Path("prediction.npy"))
    parser.add_argument("--interval", type=int, choices=(6, 12, 24), default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU run")
    device = torch.device(args.device)
    model, variables = load_model(args.config, args.checkpoint, device)
    model_input = load_input(args.input, variables, device)
    interval = torch.full((model_input.shape[0],), args.interval / 10.0, device=device, dtype=torch.float32)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(model_input, variables, interval)

    for index, variable in enumerate(variables):
        if variable in CONSTANTS:
            prediction[:, index] = 0.0

    prediction = prediction.float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction)
    print(f"input shape: {tuple(model_input.shape)}")
    print(f"output shape: {prediction.shape}")
    print(f"output saved to: {args.output}")
    print(f"prediction range: [{prediction.min():.6g}, {prediction.max():.6g}]")


if __name__ == "__main__":
    main()
