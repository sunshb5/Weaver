"""
WEAVER backbone components.




"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp
from torch.utils.checkpoint import checkpoint
from .patch_embedding import DualStreamPatchEmbedding
from .rp_moe import SoftRegionalPriorMoE

try:
    from xformers.ops import memory_efficient_attention as _xformers_attention
except (ImportError, OSError):
    _xformers_attention = None

_xformers_operator_unsupported = False


def unbind(tensor, dim):
    """Keep the xFormers-style helper without requiring xFormers to import."""
    return torch.unbind(tensor, dim=dim)


def memory_efficient_attention(query, key, value, attn_bias=None):
    """Use xFormers when supported, otherwise fall back to PyTorch SDPA.

    WEAVER stores attention tensors as (batch, tokens, heads, head_dim), while
    PyTorch SDPA expects (batch, heads, tokens, head_dim).
    """
    global _xformers_operator_unsupported
    if _xformers_attention is not None and not _xformers_operator_unsupported:
        try:
            return _xformers_attention(query, key, value, attn_bias=attn_bias)
        except NotImplementedError:
            # A frequent release-time case is a CPU-only xFormers wheel or a
            # wheel built for a different CUDA architecture.  Remember the
            # failure so every layer does not repeat operator dispatch.
            _xformers_operator_unsupported = True

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_bias
    )
    return output.transpose(1, 2)


class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)
        self.act_fn = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.drop(x)
        x = self.down_proj(x)
        return x


def modulate(x, shift, scale):
    
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    
    def __init__(self, hidden_size):
        
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        
        
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  
        self.scale = head_dim**-0.5  

        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        q, k, v = unbind(qkv, 2)

        
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])  

        x = self.proj(x)       
        x = self.proj_drop(x)  # Dropout
        return x


class Block(nn.Module):
    """AdaLN-conditioned transformer block."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 use_gated_mlp=False, **block_kwargs):
        
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                                act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)  
        )

    def forward(self, x, c):
        
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossCoupledAttentionBlock(nn.Module):
    
    def __init__(self, d_ch, num_heads, hidden_size, mlp_ratio=4.0, **block_kwargs):
        
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_ch, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(d_ch, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(d_ch, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_ch * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=d_ch, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * d_ch, bias=True)
        )

    def forward(self, X, c):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        B, L, C, D = X.shape

        
        
        shift_msa = shift_msa.unsqueeze(1).unsqueeze(2)   # (B,1,1,D)
        scale_msa = scale_msa.unsqueeze(1).unsqueeze(2)
        shift_mlp = shift_mlp.unsqueeze(1).unsqueeze(2)
        scale_mlp = scale_mlp.unsqueeze(1).unsqueeze(2)
        gate_msa = gate_msa.unsqueeze(1).unsqueeze(2)
        gate_mlp = gate_mlp.unsqueeze(1).unsqueeze(2)

        
        h = self.norm1(X) * (1 + scale_msa) + shift_msa  # (B, L, C, D)
        h = h.reshape(B * L, C, D)  
        h = self.attn(h)  
        h = h.reshape(B, L, C, D)
        X = X + gate_msa * h

        
        h = self.norm2(X) * (1 + scale_mlp) + shift_mlp  # (B, L, C, D)
        h = h.reshape(B * L * C, D)
        h = self.mlp(h)
        h = h.reshape(B, L, C, D)
        X = X + gate_mlp * h

        return X


class CrossAttn(nn.Module):
    """Multi-head cross-attention layer."""
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        """q: (B, Nq, D), kv: (B, Nk, D) → (B, Nq, D)"""
        B, Nq, D = q.shape
        _, Nk, _ = kv.shape
        H, hd = self.num_heads, self.head_dim
        q = self.q(q).reshape(B, Nq, H, hd)
        k = self.k(kv).reshape(B, Nk, H, hd)
        v = self.v(kv).reshape(B, Nk, H, hd)
        out = memory_efficient_attention(q, k, v)
        out = out.reshape(B, Nq, D)
        return self.proj(out)


class CrossCoupledAttentionGatherBlock(nn.Module):
    
    def __init__(self, d_ch, num_heads, hidden_size, K=9, mlp_ratio=4.0):
        super().__init__()
        self.K = K
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        self.region_query = nn.Parameter(torch.zeros(K, d_ch))
        self.gather_norm = nn.LayerNorm(d_ch, elementwise_affine=False)
        self.gather = CrossAttn(d_ch, num_heads)
        
        self.norm1 = nn.LayerNorm(d_ch, elementwise_affine=False)
        self.var_attn = MemEffAttention(d_ch, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(d_ch, elementwise_affine=False)
        self.mlp = Mlp(in_features=d_ch, hidden_features=int(d_ch * mlp_ratio),
                       act_layer=approx_gelu, drop=0)
        
        self.dist_norm_q = nn.LayerNorm(d_ch, elementwise_affine=False)
        self.dist_norm_v = nn.LayerNorm(d_ch, elementwise_affine=False)
        self.distribute = CrossAttn(d_ch, num_heads)
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * d_ch))
        self.out_gate = nn.Parameter(torch.zeros(1, 1, 1, d_ch))

    def forward(self, X, c):
        
        B, L, C, D = X.shape
        K = self.K
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        def _exp(p):
            
            return p.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

        
        X_c = X.permute(0, 2, 1, 3).reshape(B * C, L, D)   
        q = self.region_query.unsqueeze(0).expand(B * C, K, D)  # (B·C, K, D)
        V = self.gather(q, self.gather_norm(X_c))           # (B·C, K, D)
        V = V.reshape(B, C, K, D).permute(0, 2, 1, 3).reshape(B * K, C, D)  # (B·K, C, D)

        
        V = V + _exp(gate_msa).unsqueeze(1) * self.var_attn(
                modulate(self.norm1(V), _exp(shift_msa), _exp(scale_msa)))
        V = V + _exp(gate_mlp).unsqueeze(1) * self.mlp(
                modulate(self.norm2(V), _exp(shift_mlp), _exp(scale_mlp)))

        
        V = V.reshape(B, K, C, D).permute(0, 2, 1, 3).reshape(B * C, K, D)  # (B·C, K, D)
        out = self.distribute(self.dist_norm_q(X_c), self.dist_norm_v(V))   # (B·C, L, D)
        out = out.reshape(B, C, L, D).permute(0, 2, 1, 3)   # (B, L, C, D)
        X = X + self.out_gate * out
        return X


class VariableToSpatialBridge(nn.Module):
    
    def __init__(self, d_ch, hidden_size, num_heads, K=1):
        
        super().__init__()
        self.K = K
        self.queries = nn.Parameter(torch.randn(K, d_ch) * 0.02)
        self.agg = CrossAttn(d_ch, num_heads)  
        self.proj = nn.Linear(d_ch, hidden_size)
        if K != 1:
            self.proj = nn.Linear(K * d_ch, hidden_size)

    def forward(self, X):
        
        B, L, C, D = X.shape
        X = X.reshape(B * L, C, D)
        q = self.queries.unsqueeze(0).expand(B * L, self.K, D)
        out = self.agg(q, X)  
        out = out.reshape(B, L, self.K * D)
        out = self.proj(out)  # (B, L, hidden_size)
        return out


class RegionalPriorMoEBlock(nn.Module):
    """AdaLN-Zero block with regional-prior MoE routing.

    Data-driven top-k gating is combined with a learnable region-to-expert prior.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 routed_num_experts=8, num_regions=33,
                 selected_experts=2, prior_scale_init=1.0,
                 use_mlp_gate=False, use_weight_separation=False,
                 use_gating_conv=False, img_size=None,
                 noisy_gating=True, region_prior_init=None,
                 use_region_prior=True,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.moe = SoftRegionalPriorMoE(
            input_size=hidden_size, output_size=hidden_size,
            hidden_size=mlp_hidden_dim,
            routed_num_experts=routed_num_experts,
            num_regions=num_regions,
            k=selected_experts, prior_scale_init=prior_scale_init,
            use_mlp_gate=use_mlp_gate,
            use_weight_separation=use_weight_separation,
            use_gating_conv=use_gating_conv,
            img_size=img_size,
            noisy_gating=noisy_gating,
            region_prior_init=region_prior_init,
            use_region_prior=use_region_prior)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, region_labels):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        moe_out = self.moe(x_mod, region_labels)
        x = x + gate_mlp.unsqueeze(1) * moe_out
        return x


class FinalLayer(nn.Module):
    
    def __init__(self, hidden_size, patch_size, out_channels):
        
        super().__init__()
        self.norm_final = nn.Identity()  
        
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        
        
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        x = modulate(self.norm_final(x), shift, scale)
        
        x = self.linear(x)
        return x


class WeaverBackbone(nn.Module):
    
    def __init__(self,
        in_img_size,        
        variables,          
        patch_size=2,       
        hidden_size=1024,   
        depth=24,           
        num_heads=16,       
        mlp_ratio=4.0,      
        use_gated_mlp=False,
        d_ch=128,           
        channel_num_heads=4,
        use_activation_checkpointing=False,  
        channel_k=None,     
        use_channel_branch=True,   
        use_regional_prior_moe=False, 
        routed_num_experts=9,
        selected_experts=3,
        prior_scale_init=1.0,
        num_regions=9,
        region_map_path=None,
        region_prior_init_path=None,
        use_mlp_gate=False,
        use_weight_separation=False,
        use_gating_conv=False,
        noisy_gating=True,
        use_region_prior=True,
    ):
        super().__init__()

        
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        self.use_regional_prior_moe = use_regional_prior_moe
        self.use_channel_branch = use_channel_branch
        self.use_region_prior = use_region_prior
        self.region_prior_init = None
        if region_prior_init_path is not None:
            self.region_prior_init = torch.from_numpy(
                np.load(region_prior_init_path, allow_pickle=False)).float()

        
        
        
        self.dual_stream_embedding = DualStreamPatchEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
            d_ch=d_ch,
            use_channel_branch=use_channel_branch,
        )
        self.embedding_norm = nn.LayerNorm(hidden_size)  

        
        
        self.time_embedding = TimestepEmbedder(hidden_size)

        
        
        h_patches = in_img_size[0] // patch_size
        w_patches = in_img_size[1] // patch_size
        if use_regional_prior_moe:
            self.spatial_blocks = nn.ModuleList([
                RegionalPriorMoEBlock(hidden_size, num_heads, mlp_ratio,
                                   routed_num_experts=routed_num_experts,
                                   num_regions=num_regions,
                                   selected_experts=selected_experts,
                                   prior_scale_init=prior_scale_init,
                                   use_mlp_gate=use_mlp_gate,
                                   use_weight_separation=use_weight_separation,
                                   use_gating_conv=use_gating_conv,
                                   img_size=(h_patches, w_patches),
                                   noisy_gating=noisy_gating,
                                   region_prior_init=getattr(self, "region_prior_init", None),
                                   use_region_prior=use_region_prior)
                for _ in range(depth)
            ])
        else:
            self.spatial_blocks = nn.ModuleList([
                Block(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                      use_gated_mlp=use_gated_mlp) for _ in range(depth)
            ])

        
        if use_regional_prior_moe and region_map_path is not None:
            self.register_buffer('region_labels',
                torch.from_numpy(np.load(region_map_path, allow_pickle=False)).long())
        else:
            self.register_buffer('region_labels', None)

        
        
        
        
        # Spatial-to-variable bridge (hidden_size -> d_ch).
        # Variable-to-spatial bridge (K * d_ch -> hidden_size).
        
        if use_channel_branch:
            self.spatial_to_variable = nn.Linear(hidden_size, d_ch)
            if channel_k is not None and channel_k > 0:
                self.cca_blocks = nn.ModuleList([
                    CrossCoupledAttentionGatherBlock(d_ch, channel_num_heads, hidden_size,
                                       K=channel_k, mlp_ratio=mlp_ratio)
                    for _ in range(depth)
                ])
            else:
                self.cca_blocks = nn.ModuleList([
                    CrossCoupledAttentionBlock(d_ch, channel_num_heads, hidden_size, mlp_ratio=mlp_ratio)
                    for _ in range(depth)
                ])
            self.variable_to_spatial = VariableToSpatialBridge(
                d_ch, hidden_size, channel_num_heads
            )
            self.use_activation_checkpointing = use_activation_checkpointing
        else:
            self.use_activation_checkpointing = False

        
        
        self.forecast_head = FinalLayer(hidden_size, patch_size, len(variables))

        
        self.initialize_weights()

    def initialize_weights(self):
        
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        
        trunc_normal_(self.time_embedding.mlp.weight, std=0.02)

        
        
        for block in self.spatial_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        
        if hasattr(self, "cca_blocks"):
            for block in self.cca_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                
                if hasattr(block, "region_query"):
                    trunc_normal_(block.region_query, std=0.02)
                if hasattr(block, "out_gate"):
                    nn.init.constant_(block.out_gate, 0)

        
        nn.init.constant_(self.forecast_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.forecast_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.forecast_head.linear.weight, 0)
        nn.init.constant_(self.forecast_head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        
        p = self.patch_size
        v = len(self.variables)
        
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]  

        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        
        
        x = torch.einsum("nhwpqv->nvhpwq", x)
        
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def _dual_stage(self, z_agg, X, c, i):
        
        
        z_agg = self.spatial_blocks[i](z_agg, c)
        
        
        
        
        if self.use_activation_checkpointing:
            X, z_agg = checkpoint(self._cca_stage, X, z_agg, c, i, use_reentrant=False)
        else:
            X, z_agg = self._cca_stage(X, z_agg, c, i)
        return z_agg, X

    def _cca_stage(self, X, z_agg, c, i):
        
        X = X + self.spatial_to_variable(z_agg).unsqueeze(2)
        
        X = self.cca_blocks[i](X, c)
        
        z_agg = z_agg + self.variable_to_spatial(X)
        return X, z_agg

    def forward(self, x, variables, time_interval):
        """

          1. DualStreamPatchEmbedding -> (z_agg, X)

        Args:

        Returns:
        """
        
        if self.use_regional_prior_moe and not self.use_channel_branch:
            z_agg = self.dual_stream_embedding(x, variables)          # (B, L, D)
            z_agg = self.embedding_norm(z_agg)
            c = self.time_embedding(time_interval)
            B = z_agg.shape[0]
            region_labels = self.region_labels.unsqueeze(0).expand(B, -1)
            for block in self.spatial_blocks:
                z_agg = block(z_agg, c, region_labels)
            x = self.forecast_head(z_agg, c)
            return self.unpatchify(x)

        
        
        
        z_agg, X = self.dual_stream_embedding(x, variables)
        z_agg = self.embedding_norm(z_agg)  

        
        time_interval_emb = self.time_embedding(time_interval)

        
        for i in range(len(self.spatial_blocks)):
            z_agg, X = self._dual_stage(z_agg, X, time_interval_emb, i)

        
        x = self.forecast_head(z_agg, time_interval_emb)
        
        x = self.unpatchify(x)

        return x
    
# variables = [
#     "2m_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "mean_sea_level_pressure",
#     "geopotential_50",
#     "geopotential_100",
#     "geopotential_150",
#     "geopotential_200",
#     "geopotential_250",
#     "geopotential_300",
#     "geopotential_400",
#     "geopotential_500",
#     "geopotential_600",
#     "geopotential_700",
#     "geopotential_850",
#     "geopotential_925",
#     "geopotential_1000",
#     "u_component_of_wind_50",
#     "u_component_of_wind_100",
#     "u_component_of_wind_150",
#     "u_component_of_wind_200",
#     "u_component_of_wind_250",
#     "u_component_of_wind_300",
#     "u_component_of_wind_400",
#     "u_component_of_wind_500",
#     "u_component_of_wind_600",
#     "u_component_of_wind_700",
#     "u_component_of_wind_850",
#     "u_component_of_wind_925",
#     "u_component_of_wind_1000",
#     "v_component_of_wind_50",
#     "v_component_of_wind_100",
#     "v_component_of_wind_150",
#     "v_component_of_wind_200",
#     "v_component_of_wind_250",
#     "v_component_of_wind_300",
#     "v_component_of_wind_400",
#     "v_component_of_wind_500",
#     "v_component_of_wind_600",
#     "v_component_of_wind_700",
#     "v_component_of_wind_850",
#     "v_component_of_wind_925",
#     "v_component_of_wind_1000",
#     "temperature_50",
#     "temperature_100",
#     "temperature_150",
#     "temperature_200",
#     "temperature_250",
#     "temperature_300",
#     "temperature_400",
#     "temperature_500",
#     "temperature_600",
#     "temperature_700",
#     "temperature_850",
#     "temperature_925",
#     "temperature_1000",
#     "specific_humidity_50",
#     "specific_humidity_100",
#     "specific_humidity_150",
#     "specific_humidity_200",
#     "specific_humidity_250",
#     "specific_humidity_300",
#     "specific_humidity_400",
#     "specific_humidity_500",
#     "specific_humidity_600",
#     "specific_humidity_700",
#     "specific_humidity_850",
#     "specific_humidity_925",
#     "specific_humidity_1000",
# ]
# from torch.utils.flop_counter import FlopCounterMode
# import numpy as np
# from weaver.utils.metrics import lat_weighted_mse
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# import torch.distributed as dist
# import os

# # # Mock environment variables for single-node distributed training
# # os.environ['RANK'] = '0'
# # os.environ['WORLD_SIZE'] = '1'
# # os.environ['MASTER_ADDR'] = 'localhost'
# # os.environ['MASTER_PORT'] = '12355'

# # dist.init_process_group(backend='nccl')

# device = 'cuda'
# patch_size = 8

# model = ViTAdaLN(
#     in_img_size=(721,1440),
#     variables=variables,
#     patch_size=patch_size,
#     embed_norm=True,
#     hidden_size=1024,
#     depth=24,
#     num_heads=16,
#     mlp_ratio=4.0,
# ).to(device).half()
# # model = FSDP(
# #     model, 
# #     sharding_strategy="SHARD_GRAD_OP",
# #     # activation_checkpointing_policy={Block, ClimaXEmbedding},
# #     # auto_wrap_policy=Block
# # )

# x = torch.randn((1, 69, 721, 1440)).to(device, dtype=torch.half)
# y = torch.rand_like(x)
# pad_size = patch_size - 721 % patch_size
# padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
# lat = np.random.randn(721)
# time_interval = torch.tensor([6]).to(dtype=x.dtype).to(device)

# flop_counter = FlopCounterMode(model, depth=2)
# with flop_counter:
#     # lat_weighted_mse(model(padded_x, variables, time_interval)[:, :, pad_size:], y, variables, lat)["w_mse_aggregate"].backward()
#     model(padded_x, variables, time_interval)
