import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType
from functools import partial
from typing import Optional

from allatom_design.model.atom_denoiser.denoisers.dit_utils import (
    DiTBlock, FinalLayer, MultiHeadRMSNorm)

from openfold.model.primitives import Linear

class ResidueTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        DiT for unconditional backbone diffusion
        """
        super().__init__()

        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.n_aatype = cfg.n_aatype

        # Model parameters
        self.num_heads = cfg.num_heads
        self.condition_on_seq = cfg.condition_on_seq
        self.edge_attn_bias = cfg.edge_attn_bias

        # QK-normalization from SD3
        self.qk_normlayer = None
        if cfg.qk_rmsnorm:
            self.qk_normlayer = partial(MultiHeadRMSNorm, heads=cfg.num_heads)

        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_size, cfg.num_heads,
                     mlp_dropout=cfg.mlp_dropout, mlp_ratio=cfg.mlp_ratio,
                     inf=cfg.inf,
                     rotary_emb=cfg.rotary_emb,
                     qk_norm=cfg.qk_rmsnorm, norm_layer=self.qk_normlayer,
                     ) for _ in range(cfg.depth)
        ])

        self.embed_seq = nn.Linear(self.n_aatype, cfg.hidden_size)
        self.embed_seq = nn.Linear(self.n_aatype, cfg.hidden_size)
        self.embed_node = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")  # "glorot" should match DiT Patchify init
        self.final_layer = FinalLayer(cfg.hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self,
                node_embeddings: TensorType["b n h ", float],
                edge_embeddings: TensorType["b n k h", float],
                edge_index: TensorType["b n k", int],
                aatype_noised: Optional[TensorType["b n", int]],
                seq_mask: TensorType["b n", float],
                ) -> TensorType["b n 4 3", float]:  
        
        #embed mpnn conditioning to hidden dim of DiT
        node_embeddings = self.embed_node(node_embeddings)

        # Concatenate one-hot sequence conditioning
        if self.condition_on_seq:
            aatype_oh = F.one_hot(aatype_noised, num_classes=self.n_aatype).float()
            x = node_embeddings + self.embed_seq(aatype_oh)

        # Conditioning 
        c = node_embeddings

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.num_heads)
        attn_bias = None

        if self.edge_attn_bias:
            B, N, _ = edge_index.shape
            attn_bias = torch.zeros((B, N, N), device = x.device)
            proj_edge_embedding = self.proj_edge(edge_embeddings)
            attn_bias.scatter_(2, edge_index, proj_edge_embedding)

        for block in self.blocks:
            x = block(x, c, residx=None, attn_mask=attn_mask, attn_bias=attn_bias, per_token_conditioning = True)

        # Final layer
        x = self.final_layer(x, c, per_token_conditioning = True)
        x = x * seq_mask[..., None]  # zero out padding positions

        return x