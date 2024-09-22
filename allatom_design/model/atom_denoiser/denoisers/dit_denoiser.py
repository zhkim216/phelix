# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp
from torch.jit import Final
from torchtyping import TensorType

import allatom_design.data.conditioning_labels as cl
import allatom_design.data.data as data
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.ad_interpolants.edm_ca_interpolant import \
    EDM_CA
from allatom_design.model.atom_denoiser.denoisers.denoiser import BaseAtomDenoiser
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from openfold.model.primitives import Linear
from collections import defaultdict
from tqdm import tqdm


class DiTDenoiser(BaseAtomDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float]]):
        """
        Backbone diffusion with DiT
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = EDM_CA(cfg.interpolant, sigma_data=sigma_data)

        # Set up DiT model
        self.num_atoms_in = cfg.num_atoms_in
        self.use_self_conditioning = cfg.use_self_conditioning
        self.condition_on_seq = cfg.condition_on_seq

        # Input and output channels
        self.c = self.num_atoms_in * 3  # 3 xyz coordinates per atom
        self.in_channels = self.c * 2 if self.use_self_conditioning else self.c  # 2x for self-conditioning
        if self.condition_on_seq:
            # +n_aatype for seq conditioning
            self.in_channels = self.in_channels + cfg.n_aatype

        self.out_channels = self.c
        self.n_aatype = cfg.n_aatype

        # Model parameters
        self.num_heads = cfg.num_heads
        self.pos_encoding = cfg.pos_encoding

        self.x_embedder = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")  # "glorot" should match DiT Patchify init

        # Positional encodings
        assert self.pos_encoding in ["absolute", "absolute_residx", "rotary", "rotary_residx"]
        if self.pos_encoding in ["absolute", "absolute_residx"]:
            self.pos_embed = posemb_sincos_1d

        self.rotary_emb = None
        if self.pos_encoding in ["rotary", "rotary_residx"]:
            dim = cfg.hidden_size // cfg.num_heads
            use_residx = (self.pos_encoding == "rotary_residx")
            self.rotary_emb = rope.RotaryEmbedding(dim=dim, use_residx=use_residx, cache_if_possible=False)

        # Time embeddings
        n_time_embedders = 2
        self.t_embedders = nn.ModuleList([TimestepEmbedder(cfg.hidden_size) for _ in range(n_time_embedders)])

        # Conditioning
        self.cond_label_to_dropout_p = getattr(cfg, "cond_label_to_dropout_p", {})
        self.cond_labels = [k for k, v in self.cond_label_to_dropout_p.items() if v is not None]
        self.cond_embedders = nn.ModuleDict({
            label: LabelEmbedder(num_classes=cl.COND_NUM_CLASSES[label],
                                 hidden_size=cfg.hidden_size,
                                 dropout_prob=self.cond_label_to_dropout_p[label]) for label in self.cond_labels
        })

        # QK-normalization from SD3
        self.qk_normlayer = None
        if cfg.qk_rmsnorm:
            self.qk_normlayer = partial(MultiHeadRMSNorm, heads=cfg.num_heads)

        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_size, cfg.num_heads,
                     mlp_dropout=cfg.mlp_dropout, mlp_ratio=cfg.mlp_ratio,
                     inf=cfg.inf,
                     rotary_emb=self.rotary_emb,
                     qk_norm=cfg.qk_rmsnorm, norm_layer=self.qk_normlayer,
                     ) for _ in range(cfg.depth)
        ])
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

        # Initialize timestep embedding MLP:
        for t_embedder in self.t_embedders:
            nn.init.normal_(t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(t_embedder.mlp[2].weight, std=0.02)

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
                x_noised: TensorType["b n a 3", float],
                aatype_noised: Optional[TensorType["b n", int]],
                t: TensorType["b", float],  # timestep of inputs
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            residue_index=residue_index,
            seq_mask=seq_mask,
            cond_labels_in=cond_labels_in,
            aux_inputs=aux_inputs,
            is_sampling=is_sampling
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                           residue_index: TensorType["b n", int],
                           seq_mask: TensorType["b n", float],
                           cond_labels_in: Dict[str, TensorType["b", int]] = {},
                           aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                           is_sampling: bool = False
                           ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred of backbone
                                      Dict[str, TensorType["b ..."]]]:
        B, N = seq_mask.shape
        diffusion_aux = defaultdict(lambda: None)

        if not is_sampling:
            ### TRAINING ###

            # Get ground truth backbone coordinates
            x_bb_gt = aux_inputs["x"][..., rc.bb_idxs, :]

            if self.cfg.interpolant.name == "edm_ca":
                # Center N, C, and O on CA
                x_bb_gt[..., rc.nco_idxs, :] = x_bb_gt[..., rc.nco_idxs, :] - x_bb_gt[..., 1:2, :]

            # Repeat inputs for batch multiplier  # TODO: randomly augment these too
            M = self.cfg.training_batch_size_mult
            x_bb_gt_batched = repeat(x_bb_gt, "b n a x -> (m b) n a x", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)
            residue_index_batched = repeat(residue_index, "b n -> (m b) n", m=M, b=B)
            cond_labels_in_batched = {label: repeat(cond_labels_in[label], "b -> (m b)", m=M, b=B) for label in cond_labels_in}

            # Evaluate at specific timesteps (for validation)
            t_bb = None
            if aux_inputs["t_ca"] is not None and aux_inputs["t_nco"] is not None:
                t_bb = (torch.full((M * B, ), aux_inputs["t_ca"], device=x_bb_gt.device),
                        torch.full((M * B, ), aux_inputs["t_nco"], device=x_bb_gt.device))

            # Noise the ground truth backbone
            interpolant_out = self.interpolant({"x": x_bb_gt_batched, "aatype": None}, t=t_bb)
            xt_bb_batched = interpolant_out["x_noised"]
            t_batched = interpolant_out["t"]
            diffusion_aux["loss_weight_t"] = interpolant_out["loss_weight_t"]

            # Run denoising DiT
            denoiser_fn = self.dit
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, None, t_batched,
                                                           seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                           cond_labels_in=cond_labels_in_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched)

            x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, None, t_batched,
                                                   seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                   cond_labels_in=cond_labels_in_batched)


            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = x1_bb_batched
            diffusion_aux["bb_target"] = x_bb_gt_batched

        else:
            ### SAMPLING ###

            # Sample backbone from prior
            A = len(rc.bb_idxs)
            x0_bb = self.interpolant.sample_prior((B, N, A, 3), seq_mask.device)

            # Store trajectory
            xt_bb_traj, x1_bb_traj = [], []

            # Extract sampling parameters
            S = aux_inputs["num_steps"]
            timesteps = aux_inputs["timesteps"]
            churn_cfg = aux_inputs["churn_cfg"]
            noise_schedule = aux_inputs["noise_schedule"]
            ## extract overrides
            xt_bb_override = aux_inputs["xt_override"][..., rc.bb_idxs, :]
            xt_bb_override_mask = aux_inputs["xt_override_mask"][..., rc.bb_idxs, :]
            aatype_override = aux_inputs["aatype_override"]  # currently unused
            aatype_override_mask = aux_inputs["aatype_override_mask"]  # currently unused

            # Run integration steps
            denoiser_fn = partial(self.dit, aatype_noised=None,
                                  residue_index=residue_index, seq_mask=seq_mask,
                                  cond_labels_in=cond_labels_in)

            xt_bb = x0_bb
            for i in tqdm(range(S), leave=False, desc="Sampling..."):
                t = tuple(ts[:, i] for ts in timesteps) if len(timesteps) > 1 else timesteps[0][:, i]
                t_next = tuple(ts[:, i + 1] for ts in timesteps) if len(timesteps) > 1 else timesteps[0][:, i + 1]

                xt_bb, t = self.interpolant.churn(xt_bb, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

                xt_bb = xt_bb * (1 - xt_bb_override_mask[i]) + xt_bb_override[i] * xt_bb_override_mask[i]  # override xt for inputs
                xt_bb, aux_preds = self.interpolant.euler_step(denoiser_fn,
                                                               xt_bb,
                                                               t=t, t_next=t_next,
                                                               noise_schedule=noise_schedule,
                                                               cfg_cfg=None,
                                                               aux_inputs=aux_inputs)
                xt_bb = xt_bb * (1 - xt_bb_override_mask[i + 1]) + xt_bb_override[i + 1] * xt_bb_override_mask[i + 1]  # override xt for outputs  # TODO: should we override self-cond input too?

                if self.use_self_conditioning:
                    # Apply self-conditioning
                    denoiser_fn = partial(denoiser_fn, x_self_cond=aux_preds["x1_pred"])

                # Save current state
                xt_bb_traj.append(xt_bb.cpu())

                # Save current x1 prediction
                x1_bb_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            x1_bb = xt_bb
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=0)
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=0)

            if self.cfg.interpolant.name == "edm_ca":
                # Undo centering of N, C, and O on CA
                x1_bb[..., rc.nco_idxs, :] = x1_bb[..., rc.nco_idxs, :] + x1_bb[..., 1:2, :]
                diffusion_aux["xt_bb_traj"][..., rc.nco_idxs, :] = diffusion_aux["xt_bb_traj"][..., rc.nco_idxs, :] + diffusion_aux["xt_bb_traj"][..., 1:2, :]
                diffusion_aux["x1_bb_traj"][..., rc.nco_idxs, :] = diffusion_aux["x1_bb_traj"][..., rc.nco_idxs, :] + diffusion_aux["x1_bb_traj"][..., 1:2, :]

        return x1_bb, diffusion_aux


    def dit(self,
            x_noised: TensorType["b n 4 3", float],
            aatype_noised: Optional[TensorType["b n", int]],
            t: Tuple[TensorType["b", float]],  # can also be tuple [t_ca, t_nco]
            residue_index: TensorType["b n", int],
            seq_mask: TensorType["b n", float],
            x_self_cond: Optional[TensorType["b n 4 3", float]] = None,
            cond_labels_in: Dict[str, TensorType["b", int]] = {}
            ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred of backbone
                       Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        # Preconditioning
        precondition_in, precondition_out = self.interpolant.setup_preconditioning(x_noised, x_self_cond, t)
        x_noised, x_self_cond, t = precondition_in()  # input preconditioning

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x_noised)
            x_noised = torch.cat([x_noised, x_self_cond], dim=-1)

        x = rearrange(x_noised, "b n a x -> b n (a x)")

        # Concatenate one-hot sequence conditioning
        if self.condition_on_seq:
            aatype_oh = F.one_hot(aatype_noised, num_classes=self.n_aatype).float()
            x = torch.cat([x, aatype_oh], dim=-1)

        # Begin DiT forward pass
        x = self.x_embedder(x)

        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        # Conditioning
        if not isinstance(t, (tuple, list)):
            # make unimodal timestep into a tuple for convenience
            t = (t, )

        t = sum([self.t_embedders[i](t[i]) for i in range(len(t))])

        c = t
        for label_name in self.cond_labels:
            if label_name not in cond_labels_in:
                if self.cond_embedders[label_name].has_unconditional_token:
                    # if label is not in batch input, and the label supports unconditional tokens, default to unconditional generation
                    token_id = cl.COND_NUM_CLASSES[label_name]  # last token is the unconditional token
                else:
                    # otherwise, we provide a default token ID
                    token_id = cl.DEFAULT_TOKEN_ID[label_name]
                B = x_noised.shape[0]
                labels_in = torch.full((B, ), token_id, dtype=torch.long).to(x_noised.device)
            else:
                labels_in = cond_labels_in[label_name]

            # embed the label
            c = c + self.cond_embedders[label_name](labels_in, self.training)

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None)

        # Final layer
        x = self.final_layer(x, c)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x = precondition_out(x)  # output preconditioning

        return x, aux_preds


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_dropout: float, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self,
                x,
                c: Union[
                    TensorType["b h", float],  # per-sequence conditioning
                    TensorType["b n h", float]  # per-token conditioning
                    ],
                residx: TensorType["b n", float],
                attn_mask: TensorType["b n n", float],
                attn_bias: Optional[TensorType["b n n", float]],
                per_token_conditioning: bool = False  # whether c is per-token or per-sequence
                ):
        if not per_token_conditioning:
            assert c.dim() == 2, "Per-sequence conditioning requires shape [B, H] for c"
            c = c.unsqueeze(1)
        assert c.dim() == 3

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), residx=residx, attn_mask=attn_mask, attn_bias=attn_bias)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self,
                x,
                c: Union[
                    TensorType["b h", float],  # per-sequence conditioning
                    TensorType["b n h", float]  # per-token conditioning
                    ],
                per_token_conditioning: bool = False  # whether c is per-token or per-sequence
        ):
        if not per_token_conditioning:
            assert c.dim() == 2, "Per-sequence conditioning requires shape [B, H] for c"
            c = c.unsqueeze(1)
        assert c.dim() == 3

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    """
    Adapated from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py to deal with attention masking.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            inf: float = 1e9,
            rotary_emb: Optional[rope.RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.inf = inf  # for masked attention
        self.rotary_emb = rotary_emb


    def forward(self,
                x: torch.Tensor,
                residx: TensorType["b n", float],
                attn_mask: TensorType["b h n n", float],
                attn_bias: Optional[TensorType["b h n n", float]]
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, residx)
            k = self.rotary_emb.rotate_queries_or_keys(k, residx)

        if attn_bias is None:
            attn_bias = torch.zeros_like(attn_mask)
        attn_bias = torch.where(attn_mask.bool(), attn_bias, -self.inf)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# rmsnorm
# https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


def modulate(x: TensorType["b n h"], shift: TensorType["b n h"], scale: TensorType["b n h"]):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.has_unconditional_token = use_cfg_embedding  # used externally for default conditioning token settings


    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
