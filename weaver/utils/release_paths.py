"""Locate configuration, assets, and weights in the project checkout."""

from __future__ import annotations

from pathlib import Path


def release_root() -> Path:
    """Return the repository root containing the bundled config and assets."""

    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return release_root() / "configs/weaver.yaml"


def default_checkpoint_path() -> Path:
    return release_root() / "checkpoints/weaver_weights.ckpt"
