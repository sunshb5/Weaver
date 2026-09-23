

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_

from weaver.utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class DualStreamPatchEmbedding(nn.Module):
    
    def __init__(
        self,
        variables,      
        img_size,       
        patch_size=2,   
        embed_dim=1024, 
        num_heads=16,   
        d_ch=128,       
        use_channel_branch=True,  
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables
        self.use_channel_branch = use_channel_branch

        
        
        
        
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        
        
        if use_channel_branch:
            self.token_embeds_128 = nn.ModuleList(
                [PatchEmbed(None, patch_size, 1, d_ch) for i in range(len(variables))]
            )
        
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        
        
        
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        
        
        
        
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)


        
        
        if use_channel_branch:
            self.channel_embed_128, _ = self.create_var_embedding(d_ch)

        
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        
        if use_channel_branch:
            self.pos_embed_128 = nn.Parameter(torch.zeros(1, self.num_patches, d_ch), requires_grad=True)

        
        self.initialize_weights()

    def initialize_weights(self):
        
        
        
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),  
            int(self.img_size[1] / self.patch_size),  
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        
        
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.variables)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))
        
        if self.use_channel_branch:
            channel_embed_128 = get_1d_sincos_pos_embed_from_grid(self.channel_embed_128.shape[-1], np.arange(len(self.variables)))
            self.channel_embed_128.data.copy_(torch.from_numpy(channel_embed_128).float().unsqueeze(0))

        
        if self.use_channel_branch:
            pos_embed_128 = get_2d_sincos_pos_embed(
                self.pos_embed_128.shape[-1],
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                cls_token=False,
            )
            self.pos_embed_128.data.copy_(torch.from_numpy(pos_embed_128).float().unsqueeze(0))

        
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        
        if self.use_channel_branch:
            for i in range(len(self.token_embeds_128)):
                w = self.token_embeds_128[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        
        b, _, l, _ = x.shape
        
        x = torch.einsum("bvld->blvd", x)
        
        x = x.flatten(0, 1)

        
        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        
        
        x, _ = self.channel_agg(var_query, x, x)
        x = x.squeeze()  # (B*L, D)

        
        x = x.unflatten(dim=0, sizes=(b, l))
        return x

    def forward(self, x: torch.Tensor, variables):
        
        
        if isinstance(variables, list):
            variables = tuple(variables)

        
        x_orig = x

        
        
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            
            
            embed_variable = self.token_embeds[id](x[:, i : i + 1])
            embeds.append(embed_variable)
        
        x = torch.stack(embeds, dim=1)

        
        
        var_embed = self.get_var_emb(self.channel_embed, variables)
        
        x = x + var_embed.unsqueeze(2)

        
        
        x = x + self.pos_embed.unsqueeze(1)

        
        per_channel = x  # (B, V, L, D)
        
        z_agg = self.aggregate_variables(per_channel)  # (B, L, D)

        
        
        if self.use_channel_branch:
            embeds_128 = []
            for i in range(len(var_ids)):
                id = var_ids[i]
                embed_128 = self.token_embeds_128[id](x_orig[:, i : i + 1])
                embeds_128.append(embed_128)
            X = torch.stack(embeds_128, dim=1)  # (B, V, L, d_ch)
            
            var_embed_128 = self.get_var_emb(self.channel_embed_128, variables)
            X = X + var_embed_128.unsqueeze(2)
            X = X + self.pos_embed_128.unsqueeze(1)
            
            X = torch.einsum("bvld->blvd", X)
            return z_agg, X
        return z_agg

    def get_per_channel(self, x: torch.Tensor, variables):
        
        if isinstance(variables, list):
            variables = tuple(variables)
        var_ids = self.get_var_ids(variables, x.device)

        
        embeds = []
        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1])
            embeds.append(embed_variable)
        per_ch = torch.stack(embeds, dim=1)  # (B, V, L, D)

        
        var_embed = self.get_var_emb(self.channel_embed, variables)
        per_ch = per_ch + var_embed.unsqueeze(2)
        per_ch = per_ch + self.pos_embed.unsqueeze(1)

        
        per_ch = torch.einsum("bvld->blvd", per_ch)
        return per_ch
