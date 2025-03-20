from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType

from allatom_design.data.data import build_struct_pair_feat
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.timestep_embedders import \
    TimestepEmbedder
from openfold.model.primitives import Linear


class PairRepBuilder(nn.Module):
    """
    Builds AF2-like input pair representation, adapted from Openfold code.

    Similar to Proteina pair representation: (Geffner et al. https://openreview.net/pdf?id=TVQLu34bdw)
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        # Relative positional encodings
        self.rel_pos_embedder = RelativePositionalEncoding(cfg.rel_pos)

        # Embed pairwise distances
        self.distogram_cfg = cfg.distogram
        pair_in_channels = cfg.distogram.no_bins + 1  # +1 for the mask
        if self.use_self_conditioning:
            pair_in_channels *= 2
        self.pair_embedder = Linear(pair_in_channels, cfg.c_z)

        # AdaLN-conditioning
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.adaln = AF3_AdaLN(cfg.c_z, cfg.hidden_size)


    def forward(self,
                x: TensorType["b n 3", float],
                x_self_cond: Optional[TensorType["b n 3", float]],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                t: TensorType["b", float],
                ) -> TensorType["b n n k", float]:

        # Initialize pair representation with relative positional encodings
        z = self.rel_pos_embedder(residue_index)

        # Embed pairwise distances
        dgram_batch = {"pseudo_beta_mask": seq_mask, "pseudo_beta": x}
        pair_dists = build_struct_pair_feat(dgram_batch, **self.distogram_cfg)

        if self.use_self_conditioning:
            if x_self_cond is None:
                pair_dists_sc = torch.zeros_like(pair_dists)
            else:
                dgram_batch = {"pseudo_beta_mask": seq_mask, "pseudo_beta": x_self_cond}
                pair_dists_sc = build_struct_pair_feat(dgram_batch, **self.distogram_cfg)
            pair_dists = torch.cat([pair_dists, pair_dists_sc], dim=-1)

        z = z + self.pair_embedder(pair_dists)

        # Add in time conditioning with AdaLN
        c = self.t_embedder(t)
        c = rearrange(c, "b c -> b 1 1 c").expand(-1, z.shape[1], z.shape[2], -1)
        z = self.adaln(z, c)

        # Mask out padding
        mask_2d = seq_mask[:, None] * seq_mask[:, :, None]
        z = z * mask_2d[..., None]
        return z


class RelativePositionalEncoding(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.c_z = cfg.c_z
        self.relpos_k = cfg.relpos_k

        # RPE
        self.n_bins = 2 * self.relpos_k + 1
        self.linear_relpos = Linear(self.n_bins, self.c_z)


    def relpos(self, ri: TensorType["b n", int]) -> TensorType["b n n c_z", float]:
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)


    def forward(self, ri: TensorType["b n", int]) -> TensorType["b n n c_z", float]:
        z = self.relpos(ri.float())  # [b, n, n, c_z]
        return z






class AF3_AdaLN(nn.Module):
    """
    Algorithm 26
    Adapted from lucidrains: https://github.com/lucidrains/alphafold3-pytorch/blob/c42884e017e05599adea226b22fa12ccd0663ca5/alphafold3_pytorch/alphafold3.py#L625
    """
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)
        self.norm_cond = nn.LayerNorm(dim_cond, bias = False)

        self.to_gamma = nn.Sequential(
            Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = Linear(dim_cond, dim, bias=False)

    def forward(self,
                x: TensorType["b n h", float],
                cond: TensorType["b n h_cond", float]) -> TensorType["b n h", float]:

        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta
