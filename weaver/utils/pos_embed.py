"""Two-dimensional sinusoidal and checkpoint interpolation utilities."""

import numpy as np
import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """Generate a 2D sinusoidal embedding for a rectangular grid."""
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    # Build the embedding from the grid.
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        # Prefix an all-zero CLS token when requested.
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Build a 2D embedding by concatenating two 1D embeddings."""
    assert embed_dim % 2 == 0

    # Encode the two coordinate axes separately.
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Concatenate the axis embeddings.
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate a 1D sinusoidal embedding for flattened positions."""
    assert embed_dim % 2 == 0
    # Use the standard Transformer frequency schedule.
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # Concatenate sine and cosine components.
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    """Interpolate checkpoint position embeddings to a new image size."""
    if "net.dual_stream_embedding.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.dual_stream_embedding.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size

        # Infer the original grid from the 2:1 longitude/latitude ratio.
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)

        # Convert image size to patch-grid size.
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)

        if orig_size[0] != new_size[0]:
            print("Interpolate PEs from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            # Reshape for 2D interpolation.
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(
                0, 3, 1, 2
            )
            # Interpolate with bicubic sampling.
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode="bicubic", align_corners=False
            )
            # Restore the flattened token shape.
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.dual_stream_embedding.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    """Truncate channel embeddings when reducing the variable count."""
    if "net.channel_embed" in checkpoint_model:
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1]
        # Truncate only when the new length is smaller.
        if new_len <= old_len:
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[:, :new_len]
