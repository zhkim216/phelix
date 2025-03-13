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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.model.atom_denoiser.denoisers.denoiser import \
    BaseAtomDenoiser
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from allatom_design.model.atom_denoiser.denoisers.u_dit_utils import (
    UDiTBlock, FinalLayer, UDiTLevel, LabelEmbedder, MultiHeadRMSNorm)
from openfold.model.primitives import Linear


class UDiTDenoiser(BaseAtomDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float]]):
        """
        Backbone diffusion with U-DiT
        """
        super().__init__()

        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.interpolant = EDM(cfg.interpolant, sigma_data=sigma_data)

        # Set up DiT
        self.u_dit = UDiT(cfg.u_dit, self.interpolant)

        # Autoguidance
        self.use_autoguidance = cfg.autoguidance.enabled
        if self.use_autoguidance:
            self.autoguidance_train_p = 1 / cfg.autoguidance.subsample_train_iter_mult
            self.guiding_model = UDiT(OmegaConf.merge(cfg.u_dit, cfg.autoguidance.u_dit), self.interpolant)  # override with autoguidance config


    def forward(self,
                x_motif: TensorType["b n 33 3", float],
                motif_mask: TensorType["b n 37 3", float],
                aatype_scaffold: TensorType["b n", int],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            aatype_scaffold=aatype_scaffold,
            residue_index=residue_index,
            seq_mask=seq_mask,
            cond_labels_in=cond_labels_in,
            aux_inputs=aux_inputs,
            is_sampling=is_sampling
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                           aatype_scaffold: TensorType["b n", int],
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

            # Repeat inputs for batch multiplier  # TODO: randomly augment these too
            M = self.cfg.training_batch_size_mult
            x_bb_gt_batched = repeat(x_bb_gt, "b n a x -> (m b) n a x", m=M, b=B)
            aatype_scaffold_batched = repeat(aatype_scaffold, "b n -> (m b) n", m=M, b=B)
            residue_index_batched = repeat(residue_index, "b n -> (m b) n", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)
            cond_labels_in_batched = {label: repeat(cond_labels_in[label], "b -> (m b)", m=M, b=B) for label in cond_labels_in}

            # Evaluate at specific timesteps (for validation)
            t_bb = None
            if aux_inputs["t_bb"] is not None:
                t_bb = torch.full((M * B, ), aux_inputs["t_bb"], device=x_bb_gt.device)

            # Noise the ground truth backbone
            interpolant_out = self.interpolant({"x": x_bb_gt_batched, "aatype": None}, t=t_bb)
            xt_bb_batched = interpolant_out["x_noised"]
            x_bb_target_batched = interpolant_out["x_target"]
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Run denoising U-DiT
            denoiser_fn = self.u_dit
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, aatype_scaffold_batched, t_batched,
                                                           seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                           cond_labels_in=cond_labels_in_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched)

            x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, aatype_scaffold_batched, t_batched,
                                                   seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                   cond_labels_in=cond_labels_in_batched)

            # Train autoguidance model
            diffusion_aux["autoguidance_aux"] = None
            if self.use_autoguidance and (np.random.uniform() < self.autoguidance_train_p):
                ### If memory spikes due to running the autoguidance model,
                ### consider activation checkpointing, separate optimization steps, alternating head predictions, or just training the models separately.
                denoiser_fn = self.guiding_model
                if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                    with torch.no_grad():
                        x1_bb_batched_guide, _ = denoiser_fn(xt_bb_batched, aatype_scaffold_batched, t_batched,
                                                             seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                             cond_labels_in=cond_labels_in_batched)
                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched_guide)

                x1_bb_batched_guide, _ = denoiser_fn(xt_bb_batched, aatype_scaffold_batched, t_batched,
                                                     seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                     cond_labels_in=cond_labels_in_batched)

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "bb_pred": x1_bb_batched_guide,
                    "bb_target": x_bb_target_batched,  # diffusion target; for edm this is just the ground truth coordinates
                    "loss_weight_t": loss_weight_t_batched
                }

            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = x1_bb_batched
            diffusion_aux["bb_target"] = x_bb_target_batched  # diffusion target; for edm this is just the ground truth coordinates
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched

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
            autoguidance_cfg = aux_inputs["autoguidance_cfg"]

            ## extract overrides
            xt_bb_override = aux_inputs["xt_override"][..., rc.bb_idxs, :]
            xt_bb_override_mask = aux_inputs["xt_override_mask"][..., rc.bb_idxs, :]

            # If a backbone input is provided, run partial diffusion instead
            if aux_inputs.get("x_bb_in", None) is not None:
                xt_bb = aux_inputs["x_bb_in"]
                S = aux_inputs["num_steps_partial"]

                timesteps = timesteps[:, -(S + 1):]  # truncate timesteps to the partial diffusion range
                xt_bb = self.interpolant.noise_x(xt_bb, timesteps[:, 0])  # noise samples to first of the truncated timesteps

                xt_bb_override = xt_bb_override[-(S + 1):]
                xt_bb_override_mask = xt_bb_override_mask[-(S + 1):]
            else:
                xt_bb = x0_bb

            # Apply autoguidance
            use_autoguidance = (autoguidance_cfg is not None) and (autoguidance_cfg["use_autoguidance"])
            if use_autoguidance:
                assert self.use_autoguidance, "Model must be trained with autoguidance to use it."
                autoguidance_cfg["autoguidance_fn"] = partial(self.guiding_model, aatype_scaffold=aatype_scaffold,
                                                              residue_index=residue_index, seq_mask=seq_mask,
                                                              cond_labels_in=cond_labels_in)

            # Run integration steps
            denoiser_fn = partial(self.u_dit, aatype_scaffold=aatype_scaffold,
                                  residue_index=residue_index, seq_mask=seq_mask,
                                  cond_labels_in=cond_labels_in)

            for i in tqdm(range(S), leave=False, desc="Sampling..."):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]

                xt_bb, t = self.interpolant.churn(xt_bb, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

                xt_bb = xt_bb * (1 - xt_bb_override_mask[i]) + xt_bb_override[i] * xt_bb_override_mask[i]  # override xt for inputs
                xt_bb, aux_preds = self.interpolant.euler_step(denoiser_fn,
                                                               xt_bb,
                                                               t=t, t_next=t_next,
                                                               noise_schedule=noise_schedule,
                                                               cfg_cfg=None,
                                                               autoguidance_cfg=autoguidance_cfg,
                                                               aux_inputs=aux_inputs)
                xt_bb = xt_bb * (1 - xt_bb_override_mask[i + 1]) + xt_bb_override[i + 1] * xt_bb_override_mask[i + 1]  # override xt for outputs  # TODO: should we override self-cond input too?

                if self.use_self_conditioning:
                    # Apply self-conditioning
                    denoiser_fn = partial(denoiser_fn, x_self_cond=aux_preds["x1_pred"])

                    if use_autoguidance:
                        # self-conditioning for autoguidance
                        autoguidance_cfg["autoguidance_fn"] = partial(autoguidance_cfg["autoguidance_fn"],
                                                                      x_self_cond=aux_preds["x1_pred_ag"])

                # Save current state
                xt_bb_traj.append(xt_bb.cpu())

                # Save current x1 prediction
                x1_bb_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            x1_bb = xt_bb
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=1)  # [B S N A 3]
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=1)  # [B S N A 3]

        return x1_bb, diffusion_aux


class UDiT(nn.Module):
    def __init__(self, cfg: DictConfig, interpolant: ADInterpolant):
        """
        U-DiT for unconditional backbone diffusion
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = interpolant

        # Set up U-DiT model
        self.num_atoms_in = cfg.num_atoms_in
        self.use_self_conditioning = cfg.use_self_conditioning
        self.condition_on_seq = cfg.condition_on_seq

        # Input and output channels
        self.channels = [int(cfg.base_channels * cfg.channel_mult ** i) for i in range(cfg.num_downsamples + 1)]
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

        self.x_embedder = Linear(self.in_channels, self.channels[0], bias=True, init="glorot")  # "glorot" should match DiT Patchify init

        # Positional encodings
        assert self.pos_encoding in ["absolute", "absolute_residx", "rotary", "rotary_residx"]

        self.use_rotary_emb = False
        self.use_residx = False

        if self.pos_encoding in ["absolute", "absolute_residx"]:
            self.pos_embed = posemb_sincos_1d
        elif self.pos_encoding in ["rotary", "rotary_residx"]:
            self.use_rotary_emb = True
            self.use_residx = (self.pos_encoding == "rotary_residx")

        # Time embeddings
        self.t_embedder = TimestepEmbedder(self.channels[-1])

        # Conditioning
        self.cond_label_to_dropout_p = getattr(cfg, "cond_label_to_dropout_p", {})
        self.cond_labels = [k for k, v in self.cond_label_to_dropout_p.items() if v is not None]
        self.cond_embedders = nn.ModuleDict({
            label: LabelEmbedder(num_classes=cl.COND_NUM_CLASSES[label],
                                 hidden_size=self.channels[-1],
                                 dropout_prob=self.cond_label_to_dropout_p[label]) for label in self.cond_labels
        })

        # QK-normalization from SD3
        self.qk_normlayer = None
        if cfg.qk_rmsnorm:
            self.qk_normlayer = partial(MultiHeadRMSNorm, heads=cfg.num_heads)

        # U-DiT Blocks
        def create_udit_level(level_idx: int):
            if level_idx == cfg.num_downsamples:
                return nn.ModuleList([
                    UDiTBlock(self.channels[-1],
                              embedding_size=self.channels[-1],
                              num_heads=cfg.num_heads,
                              use_rotary_emb=self.use_rotary_emb,
                              use_residx=self.use_residx,
                              mlp_dropout=cfg.mlp_dropout,
                              mlp_ratio=cfg.mlp_ratio,
                              inf=cfg.inf,
                              qk_norm=cfg.qk_rmsnorm,
                              norm_layer=self.qk_normlayer
                              ) for _ in range(cfg.num_mid_blocks)
                ])

            hidden_size = self.channels[level_idx]
            next_hidden_size = self.channels[level_idx + 1]

            down_block = nn.ModuleList([
                UDiTBlock(hidden_size,
                          embedding_size=self.channels[-1],
                          num_heads=cfg.num_heads,
                          use_rotary_emb=self.use_rotary_emb,
                          use_residx=self.use_residx,
                          mlp_dropout=cfg.mlp_dropout,
                          mlp_ratio=cfg.mlp_ratio,
                          inf=cfg.inf,
                          qk_norm=cfg.qk_rmsnorm,
                          norm_layer=self.qk_normlayer
                          ) for _ in range(cfg.num_updown_blocks)
            ])

            up_block = nn.ModuleList([
                UDiTBlock(hidden_size,
                          embedding_size=self.channels[-1],
                          num_heads=cfg.num_heads,
                          use_rotary_emb=self.use_rotary_emb,
                          use_residx=self.use_residx,
                          mlp_dropout=cfg.mlp_dropout,
                          mlp_ratio=cfg.mlp_ratio,
                          inf=cfg.inf,
                          qk_norm=cfg.qk_rmsnorm,
                          norm_layer=self.qk_normlayer
                          ) for _ in range(cfg.num_updown_blocks)
            ])

            middle_block = create_udit_level(level_idx + 1)
            return UDiTLevel(hidden_size, next_hidden_size, down_block, middle_block, up_block)

        self.u_dit = create_udit_level(0)
        self.dit = nn.ModuleList([
            UDiTBlock(self.channels[0],
                      embedding_size=self.channels[-1],
                      num_heads=cfg.num_heads,
                      use_rotary_emb=self.use_rotary_emb,
                      use_residx=self.use_residx,
                      mlp_dropout=cfg.mlp_dropout,
                      mlp_ratio=cfg.mlp_ratio,
                      inf=cfg.inf,
                      qk_norm=cfg.qk_rmsnorm,
                      norm_layer=self.qk_normlayer
                      ) for _ in range(cfg.num_dit_blocks)
        ])
        self.final_layer = FinalLayer(self.channels[0], self.channels[-1], self.out_channels)
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
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        def _adaln_init(module):
            if isinstance(module, UDiTBlock):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)
        self.apply(_adaln_init)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)



    def forward(self,
                x_noised: TensorType["b n 4 3", float],
                aatype_scaffold: Optional[TensorType["b n", int]],
                t: TensorType["b", float],
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
            aatype_oh = F.one_hot(aatype_scaffold, num_classes=self.n_aatype).float()
            x = torch.cat([x, aatype_oh], dim=-1)

        # Begin U-DiT forward pass
        x = self.x_embedder(x)

        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        # Conditioning
        t = self.t_embedder(t)

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
        x = self.u_dit(x, c, residx=residue_index.float(), seq_mask=seq_mask)
        for block in self.dit:
            x = block(x, c, residx=residue_index.float(), seq_mask=seq_mask, attn_bias=None)

        # Final layer
        x = self.final_layer(x, c)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3).float()
        x = precondition_out(x)  # output preconditioning

        return x, aux_preds
