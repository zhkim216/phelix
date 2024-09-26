# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple, Union

import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_ca_interpolant import \
    EDM_CA
from allatom_design.model.atom_denoiser.denoisers.denoiser import \
    BaseAtomDenoiser
from allatom_design.model.atom_denoiser.denoisers.dit_utils import (
    DiTBlock, FinalLayer, LabelEmbedder, MultiHeadRMSNorm)
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from openfold.model.primitives import Linear


class ESMDiTDenoiser(BaseAtomDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float]]):
        """
        Backbone diffusion with DiT, conditioned on ESM sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.interpolant = EDM_CA(cfg.interpolant, sigma_data=sigma_data)

        # Set up ESM
        self.esm_wrapper = ESMWrapper(cfg.esm)

        # Set up DiT model
        self.dit = ESMConditionedDiT(cfg.dit, self.interpolant)


    def forward(self,
                x_noised: TensorType["b n a 3", float],
                aatype_noised: TensorType["b n", int],
                t: TensorType["b", float],  # timestep of inputs
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                mlm_mask: TensorType["b n", float],  # MLM mask for the input sequence
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        h_S = self.esm_wrapper(aatype_noised, seq_mask, residue_index, mlm_mask)

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            h_S=h_S,
            aatype_noised=aatype_noised,
            residue_index=residue_index,
            seq_mask=seq_mask,
            cond_labels_in=cond_labels_in,
            aux_inputs=aux_inputs,
            is_sampling=is_sampling
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                           h_S: TensorType["b n h", float],  # sequence embeddings for diffusion conditioning
                           aatype_noised: TensorType["b n", int],
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
            h_S_batched = repeat(h_S, "b n h -> (m b) n h", m=M, b=B)
            aatype_noised_batched = repeat(aatype_noised, "b n -> (m b) n", m=M, b=B)
            residue_index_batched = repeat(residue_index, "b n -> (m b) n", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)
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
                    x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                           h_S_batched, aatype_noised_batched, t_batched,
                                                           seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                           cond_labels_in=cond_labels_in_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched)

            x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                   h_S_batched, aatype_noised_batched, t_batched,
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

            # Run integration steps
            denoiser_fn = partial(self.dit, h_S=h_S, aatype_noised=aatype_noised,
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
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=1)  # [B S N A 3]
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=1)  # [B S N A 3]

            if self.cfg.interpolant.name == "edm_ca":
                # Undo centering of N, C, and O on CA
                x1_bb[..., rc.nco_idxs, :] = x1_bb[..., rc.nco_idxs, :] + x1_bb[..., 1:2, :]
                diffusion_aux["xt_bb_traj"][..., rc.nco_idxs, :] = diffusion_aux["xt_bb_traj"][..., rc.nco_idxs, :] + diffusion_aux["xt_bb_traj"][..., 1:2, :]
                diffusion_aux["x1_bb_traj"][..., rc.nco_idxs, :] = diffusion_aux["x1_bb_traj"][..., rc.nco_idxs, :] + diffusion_aux["x1_bb_traj"][..., 1:2, :]

        return x1_bb, diffusion_aux


class ESMConditionedDiT(nn.Module):
    def __init__(self, cfg: DictConfig, interpolant: ADInterpolant):
        """
        DiT for backbone diffusion conditioned on ESM sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = interpolant

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
                x_noised: TensorType["b n 4 3", float],
                h_S: TensorType["b n h", float],
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

        ### Begin DiT forward pass ###
        x = self.x_embedder(x)

        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        ### Conditioning ###

        # time conditioning
        if not isinstance(t, (tuple, list)):
            # make unimodal timestep into a tuple for convenience
            t = (t, )

        t = sum([self.t_embedders[i](t[i]) for i in range(len(t))])

        c = t

        # label conditioning
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

        # sequence embedding conditioning
        c = c.unsqueeze(1) + h_S

        ### Blocks ###
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None, per_token_conditioning=True)

        ### Final layer ###
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x = precondition_out(x)  # output preconditioning

        return x, aux_preds


class ESMWrapper(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Wrapper around ESM model to return sequence embeddings. Code is based on:
        https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py
        """
        super().__init__()

        self.cfg = cfg
        c_s = cfg.c_s


        self.esm, self.esm_dict = esm_registry.get(cfg.esm_type)()
        self.esm.requires_grad_(False)
        # self.esm.half()  # we train with bf16, so we shouldn't need to half the model

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMWrapper._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        self.embedding = nn.Embedding(self.cfg.n_tokens_embed, c_s, padding_idx=0)
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )


    def forward(
        self,
        aatype_noised: TensorType["b n", int],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        mlm_mask: TensorType["b n", float],
    ):
        """Runs a forward pass given input tokens.

        Args:
            aatype_noised: Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            seq_mask: Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residue_index: Residue indices of amino acids.
            mlm_mask: MLM mask on the input sequence, 1 denotes unmasked and 0 denotes masked
        """
        esm_aatype = self._af2_idx_to_esm_idx(aatype_noised, seq_mask)
        esm_aatype = self._mask_inputs_to_esm(esm_aatype, seq_mask, mlm_mask)

        # ESM2 doesn't use RoPE on residx?
        # Also, they seem to prepend BOS and append EOS tokens to every example regardless of cropping -- discrepancy with their supplement
        # https://github.com/facebookresearch/esm/issues/299
        # The pretrained ESM model seems to NaN out when neither BOS/EOS are provided
        esm_s, _ = self._compute_language_model_representations(esm_aatype)
        esm_s = esm_s.detach()

        # preprocess ESM sequence embedding
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)
        s_s_0 += self.embedding(aatype_noised)

        return s_s_0


    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = None
        return esm_s, esm_z

    def _mask_inputs_to_esm(self,
                            esmaa: TensorType["b n", int],
                            seq_mask: TensorType["b n", float],
                            mlm_mask: TensorType["b n", float]):
        """
        Mask nonpad positions where mlm_mask is 0.
        """
        new_esmaa = esmaa.clone()
        new_esmaa[(mlm_mask == 0) & (seq_mask == 1)] = self.esm_dict.mask_idx
        return new_esmaa

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]


    @staticmethod
    def _af2_to_esm(d: esm.Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in rc.restypes_with_x
        ]
        return torch.tensor(esm_reorder)


esm_registry = {
    "esm2_8M": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}