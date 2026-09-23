"""Paper-facing WEAVER model.

This module implements the paper's dual-stream Transformer: Cross-Coupled
Attention (CCA), a variable-to-spatial bridge, spatial attention with a
Regional Prior Mixture-of-Experts (RP-MoE), and a spatial-to-variable bridge.

Variable-first attention with climate-region-routed MoE.

Each block applies CCA, variable-to-spatial projection, regional-prior MoE,
and spatial-to-variable projection. The release uses nine top-3 experts.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .backbone import WeaverBackbone, VariableToSpatialBridge

__all__ = ["Weaver", "VariableToSpatialBridge"]


class Weaver(WeaverBackbone):
    """WEAVER CCA + RP-MoE model.

    The forward pass follows the paper's variable-to-spatial bridge, spatial
    RP-MoE block, and spatial-to-variable bridge order.
    """

    def __init__(
        self,
        in_img_size, variables,
        patch_size=2, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0,
        use_gated_mlp=False,
        d_ch=128, channel_num_heads=4, use_activation_checkpointing=False,
        channel_k=None, pool_k=4,
        routed_num_experts=9, selected_experts=3,
        prior_scale_init=1.0, num_regions=9,
        region_map_path=None, region_prior_init_path=None,
        use_mlp_gate=False, use_weight_separation=False,
        use_gating_conv=False, noisy_gating=True,
    ):
        super().__init__(
            in_img_size=in_img_size, variables=variables,
            patch_size=patch_size, hidden_size=hidden_size, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            use_gated_mlp=use_gated_mlp,
            d_ch=d_ch, channel_num_heads=channel_num_heads,
            use_activation_checkpointing=use_activation_checkpointing,
            channel_k=channel_k,
            use_channel_branch=True,
            use_regional_prior_moe=True,
            routed_num_experts=routed_num_experts,
            selected_experts=selected_experts,
            prior_scale_init=prior_scale_init, num_regions=num_regions,
            region_map_path=region_map_path, region_prior_init_path=region_prior_init_path,
            use_mlp_gate=use_mlp_gate, use_weight_separation=use_weight_separation,
            use_gating_conv=use_gating_conv, noisy_gating=noisy_gating,
        )
        self.depth = depth
        self.pool_k = pool_k
        # The bridge uses K learnable queries.
        if pool_k is not None and pool_k > 1:
            self.variable_to_spatial = VariableToSpatialBridge(
                d_ch, hidden_size, channel_num_heads, K=pool_k
            )
        # Zero-initialized bridge gate and per-variable output scales.
        self.gate_pool = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.per_var_scale = nn.Parameter(torch.ones(len(variables), 1))

    def _dual_stage(self, z_agg, X, c, region_labels, i):
        """CCA variable stream → variable-to-spatial bridge → RP-MoE → bridge."""
        if self.use_activation_checkpointing:
            X, z_agg = checkpoint(self._cca_variable_stage, X, z_agg, c, i, use_reentrant=False)
        else:
            X, z_agg = self._cca_variable_stage(X, z_agg, c, i)
        z_agg = self.spatial_blocks[i](z_agg, c, region_labels)
        if i < self.depth - 1:                                    # Feed the updated spatial stream.
            X = X + self.spatial_to_variable(z_agg).unsqueeze(2) * self.per_var_scale
        return z_agg, X

    def _cca_variable_stage(self, X, z_agg, c, i):
        """CCA variable block followed by the variable-to-spatial bridge."""
        X = self.cca_blocks[i](X, c)
        z_agg = z_agg + self.gate_pool * self.variable_to_spatial(X)
        return X, z_agg

    def forward(self, x, variables, time_interval):
        z_agg, X = self.dual_stream_embedding(x, variables)
        z_agg = self.embedding_norm(z_agg)
        c = self.time_embedding(time_interval)
        B = z_agg.shape[0]
        region_labels = self.region_labels.unsqueeze(0).expand(B, -1)
        for i in range(len(self.spatial_blocks)):
            z_agg, X = self._dual_stage(z_agg, X, c, region_labels, i)
        x = self.forecast_head(z_agg, c)
        return self.unpatchify(x)
