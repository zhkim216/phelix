# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import copy
from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from timm.models.vision_transformer import Mlp
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (apply_random_augmentation,
                                      build_struct_pair_feat, cat_bb_scn,
                                      center_random_augmentation)
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.model.atom_denoiser.denoisers.denoiser import \
    BaseAtomDenoiser
from allatom_design.model.atom_denoiser.denoisers.dit_utils import (
    Attention, DiTBlock, FinalLayer, LabelEmbedder, MultiHeadRMSNorm)
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from openfold.model.dropout import DropoutColumnwise, DropoutRowwise
from openfold.model.heads import DistogramHead
from openfold.model.pair_transition import PairTransition
from openfold.model.primitives import LayerNorm, Linear
from openfold.model.triangular_attention import (TriangleAttentionEndingNode,
                                                 TriangleAttentionStartingNode)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)


class TriangleDiTDenoiser(BaseAtomDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float]]):
        """
        Backbone diffusion with DiT
        """
        super().__init__()

        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.interpolant = EDM(cfg.interpolant, sigma_data=sigma_data)

        # Set up DiT
        self.dit = PairStackDiT(cfg.dit, self.interpolant)

        # Autoguidance
        self.use_autoguidance = cfg.autoguidance.enabled
        if self.use_autoguidance:
            self.autoguidance_train_p = 1 / cfg.autoguidance.subsample_train_iter_mult
            self.guiding_model = PairStackDiT(OmegaConf.merge(cfg.dit, cfg.autoguidance.dit), self.interpolant)  # override with autoguidance config


    def forward(self,
                xt_scn: TensorType["b n 33 3", float],
                aatype_noised: TensorType["b n", int],
                t_sd: TensorType["b", float],  # timestep of sequence design inputs
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                mlm_mask: TensorType["b n", float],  # MLM mask for the input sequence
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
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

            # Repeat inputs for batch multiplier  # TODO: randomly augment these too
            M = self.cfg.training_batch_size_mult
            x_bb_gt_batched = repeat(x_bb_gt, "b n a x -> (m b) n a x", m=M, b=B)
            aatype_noised_batched = repeat(aatype_noised, "b n -> (m b) n", m=M, b=B)
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
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Run denoising DiT
            denoiser_fn = self.dit
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, aatype_noised_batched, t_batched,
                                                           seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                           cond_labels_in=cond_labels_in_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched)

            x1_bb_batched, aux_preds = denoiser_fn(xt_bb_batched, aatype_noised_batched, t_batched,
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
                        x1_bb_batched_guide, _ = denoiser_fn(xt_bb_batched, aatype_noised_batched, t_batched,
                                                             seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                             cond_labels_in=cond_labels_in_batched)
                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_self_cond=x1_bb_batched_guide)

                x1_bb_batched_guide, _ = denoiser_fn(xt_bb_batched, aatype_noised_batched, t_batched,
                                                     seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                     cond_labels_in=cond_labels_in_batched)

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "bb_pred": x1_bb_batched_guide,
                    "bb_target": x_bb_gt_batched,
                    "loss_weight_t": loss_weight_t_batched
                }

            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = x1_bb_batched
            diffusion_aux["bb_target"] = x_bb_gt_batched
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
                autoguidance_cfg["autoguidance_fn"] = partial(self.guiding_model, aatype_noised=aatype_noised,
                                                              residue_index=residue_index, seq_mask=seq_mask,
                                                              cond_labels_in=cond_labels_in)

            # Run integration steps
            denoiser_fn = partial(self.dit, aatype_noised=aatype_noised,
                                  residue_index=residue_index, seq_mask=seq_mask,
                                  cond_labels_in=cond_labels_in)

            for i in tqdm(range(S), leave=False, desc="Sampling..."):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]

                xt_bb, t = self.interpolant.churn(xt_bb, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling
                xt_bb = xt_bb * (1 - xt_bb_override_mask[i]) + xt_bb_override[i] * xt_bb_override_mask[i]  # override xt for inputs

                # Apply self-conditioning
                if self.use_self_conditioning and i > 0:
                    denoiser_fn = partial(denoiser_fn, x_self_cond=aux_preds["x1_pred"])

                    # self-conditioning for autoguidance
                    if use_autoguidance:
                        autoguidance_cfg["autoguidance_fn"] = partial(autoguidance_cfg["autoguidance_fn"],
                                                                      x_self_cond=aux_preds["x1_pred_ag"])


                xt_bb, aux_preds = self.interpolant.euler_step(denoiser_fn,
                                                               xt_bb,
                                                               t=t, t_next=t_next,
                                                               noise_schedule=noise_schedule,
                                                               cfg_cfg=None,
                                                               autoguidance_cfg=autoguidance_cfg,
                                                               aux_inputs=aux_inputs)
                xt_bb = xt_bb * (1 - xt_bb_override_mask[i + 1]) + xt_bb_override[i + 1] * xt_bb_override_mask[i + 1]  # override xt for outputs  # TODO: should we override self-cond input too?

                # Save current state
                xt_bb_traj.append(xt_bb.cpu())

                # Save current x1 prediction
                x1_bb_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            x1_bb = xt_bb
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=1)  # [B S N A 3]
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=1)  # [B S N A 3]

        return x1_bb, diffusion_aux


    def get_likelihoods(self,
                        num_steps: int,
                        x_bb: TensorType["b n 4 3"],
                        aatype: TensorType["b n", int],
                        seq_mask: TensorType["b n", float],
                        atom_mask: TensorType["b n 4", float],
                        residue_index: TensorType["b n", int],
                        cond_labels_in: Dict[str, TensorType["b", int]] = {},):
        denoiser_fn = partial(self.dit,
                              aatype_noised=aatype,
                              seq_mask=seq_mask,
                              residue_index=residue_index,
                              cond_labels_in=cond_labels_in)
        x1_mask = atom_mask[..., None].expand_as(x_bb)
        x1_mask = x1_mask * rearrange(seq_mask, "b n -> b n 1 1")
        likelihood_aux = self.interpolant.get_likelihoods(denoiser_fn, x_bb, x1_mask, num_steps)
        return likelihood_aux


class PairStackDiT(nn.Module):
    def __init__(self, cfg: DictConfig, interpolant: ADInterpolant):
        """
        DiT with a pair stack applied to the self-conditioning track.
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = interpolant

        # Set up DiT model
        self.num_atoms_in = cfg.num_atoms_in
        self.use_self_conditioning = cfg.use_self_conditioning
        assert self.use_self_conditioning, "PairStackDiT requires self-conditioning."
        self.condition_on_seq = cfg.condition_on_seq

        # Input and output channels
        self.c = self.num_atoms_in * 3  # 3 xyz coordinates per atom
        self.in_channels = self.c * 2 if self.use_self_conditioning else self.c  # 2x for self-conditioning
        if self.condition_on_seq:
            # +n_aatype for seq conditioning
            self.in_channels = self.in_channels + cfg.n_aatype

        self.out_channels = self.c
        self.n_aatype = cfg.n_aatype

        # Self-conditioning pair stack
        self.skip_pair_when_no_self_cond = cfg.skip_pair_when_no_self_cond
        self.self_cond_pair_stack = SelfCondPairStack(cfg.self_cond_pair_stack)
        self.pair_bias_embedder = nn.Sequential(LayerNorm(cfg.channels_2d),
                                                Linear(cfg.channels_2d, cfg.num_heads, init="normal", bias=False))


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
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)

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
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

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
                aatype_noised: Optional[TensorType["b n", int]],
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
            has_self_cond = x_self_cond is not None
            x_self_cond = x_self_cond if has_self_cond else torch.zeros_like(x_noised)
            x_noised = torch.cat([x_noised, x_self_cond], dim=-1)

            if not has_self_cond and self.skip_pair_when_no_self_cond:
                B, N, N_heads = *x_noised.shape[:2], self.num_heads
                pair_bias = torch.zeros((B, N, N, N_heads), device=x_noised.device)
            else:
                pair_bias = self.pair_bias_embedder(self.self_cond_pair_stack(x_self_cond * self.interpolant.sigma_data, residue_index, seq_mask))

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
        t = self.t_embedder(t)
        c = t

        for label_name in self.cond_labels:
            if label_name not in cond_labels_in:
                # default to placeholder token
                B = x_noised.shape[0]
                labels_in = torch.full((B, ), cl.PLACEHOLDER_TOKEN_ID, dtype=torch.long, device=x_noised.device)
            else:
                labels_in = cond_labels_in[label_name]

            # convert placeholder tokens to labels
            if self.cond_embedders[label_name].has_unconditional_token:
                # if the label supports unconditional tokens, default to unconditional generation
                token_id = cl.COND_NUM_CLASSES[label_name]  # last token is the unconditional token
            else:
                # otherwise, we provide a default token ID
                token_id = cl.DEFAULT_TOKEN_ID[label_name]
            labels_in = torch.where(labels_in == cl.PLACEHOLDER_TOKEN_ID, token_id, labels_in)

            # embed the label
            c = c + self.cond_embedders[label_name](labels_in, self.training)

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            attn_bias = rearrange(pair_bias, "b i j h -> b h i j")
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=attn_bias)

        # Final layer
        x = self.final_layer(x, c)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x = precondition_out(x)  # output preconditioning

        return x, aux_preds


class SelfCondPairStack(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Based on AF2/AF3 Evoformer/Pairformer pair stack. Currently not implemented exactly the same as the original.
        Input should be self-conditioning input and residue index.
        """
        super().__init__()

        self.cfg = cfg
        self.rel_pos_embedder = RelativePositionalEncoding(cfg.relative_positional_encoding)
        self.structure_embedder = StructureEmbedder(cfg.structure_embedder)

        # Blocks
        self.pair_block = PairStackBlock(cfg.pair_stack_block)


    def forward(self,
                x_self_cond: TensorType["b n 4 3", float],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                ) -> Tuple[TensorType["b n c_in", float], TensorType["b n n c_z", float]]:
        # Handle masking
        mask_2d = seq_mask[:, :, None] * seq_mask[:, None, :]

        # Begin forward pass
        z = self.rel_pos_embedder(residue_index)
        z = z + self.structure_embedder(x_self_cond, seq_mask)
        z = self.pair_block(z, z_mask=mask_2d)

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


class StructureEmbedder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # 2D embedder
        self.pair_embedder = Linear(cfg.embedder_2d.n_bins + 1,  # +1 for the mask
                                    cfg.embedder_2d.c_z,
                                    init="relu")  # TemplatePairEmbedder: Despite there being no relu nearby, the source uses that initializer
        self.pair_stack = PairStack(cfg.embedder_2d.pair_stack)


    def forward(self,
                x_noised: TensorType["b n 4 3", float],
                seq_mask: TensorType["b n", float]) -> TensorType["b n n c_z", float]:
        # Build 2D embeddings
        dgram_batch = {"pseudo_beta_mask": seq_mask, "pseudo_beta": x_noised[..., 1, :]}

        z = build_struct_pair_feat(dgram_batch, self.cfg.distogram.min_bin, self.cfg.distogram.max_bin, self.cfg.distogram.no_bins)
        z = self.pair_embedder(z)

        mask_2d = seq_mask[..., None] * seq_mask[..., None, :]
        z = self.pair_stack(z, z_mask=mask_2d)
        return z


class PairStack(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        # self.blocks_per_ckpt = cfg.blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(self.cfg.n_blocks):
            self.blocks.append(PairStackBlock(self.cfg.pair_stack_block))

        self.layer_norm = LayerNorm(self.cfg.c_z, eps=cfg.eps)


    def forward(self,
                z: TensorType["b n n c_z", float],
                z_mask: TensorType["b n n", float]
                ) -> TensorType["b n n c_z", float]:
        """
        Run the pair stack with gradient checkpointing. Without gradient checkpointing, equivalent to:
        for block in self.blocks:
            z = block(z, z_mask)
        """
        # Run blocks with gradient checkpointing
        # blocks = [partial(block, z_mask=z_mask) for block in self.blocks]
        # z, = checkpoint_blocks(blocks=blocks,
        #                        args=(z, ),
        #                        blocks_per_ckpt=self.blocks_per_ckpt if self.training else None)
        for block in self.blocks:
            z = block(z, z_mask)

        return self.layer_norm(z)



class PairStackBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.use_tri_mul = cfg.use_tri_mul
        self.tri_mul_first = cfg.tri_mul_first

        self.dropout_row = DropoutRowwise(cfg.dropout_rate)
        self.dropout_col = DropoutColumnwise(cfg.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(cfg.c_z, cfg.c_hidden_tri_att, cfg.n_heads, inf=cfg.inf)
        self.tri_att_end = TriangleAttentionEndingNode(cfg.c_z, cfg.c_hidden_tri_att, cfg.n_heads, inf=cfg.inf)

        if self.use_tri_mul:
            self.tri_mul_out = TriangleMultiplicationOutgoing(cfg.c_z, cfg.c_hidden_tri_mul)
            self.tri_mul_in = TriangleMultiplicationIncoming(cfg.c_z, cfg.c_hidden_tri_mul)

        self.pair_transition = PairTransition(cfg.c_z, cfg.pair_transition_n)


    def forward(self,
                z: TensorType["b n n c_z", float],
                z_mask: TensorType["b n n", float],
                ) -> TensorType["b n n c_z", float]:
        if not self.use_tri_mul:
            z = self.tri_att_start_end(z, z_mask)
        else:
            if self.tri_mul_first:
                z = self.tri_mul_out_in(z, z_mask)
                z = self.tri_att_start_end(z, z_mask)
            else:
                z = self.tri_att_start_end(z, z_mask)
                z = self.tri_mul_out_in(z, z_mask)

        z = z + self.pair_transition(z, mask=z_mask)

        return z


    def tri_mul_out_in(self,
                       z: TensorType["b n n c_z", float],
                       z_mask: TensorType["b n n", float]
                       ) -> TensorType["b n n c_z", float]:
        tmu_update = self.tri_mul_out(z, mask=z_mask)
        z = z + self.dropout_row(tmu_update)
        del tmu_update

        tmu_update = self.tri_mul_in(z, mask=z_mask)
        z = z + self.dropout_row(tmu_update)
        del tmu_update

        return z


    def tri_att_start_end(self,
                          z: TensorType["b n n c_z", float],
                          z_mask: TensorType["b n n", float],
                          ) -> TensorType["b n n c_z", float]:
        ta_update = self.tri_att_start(z, mask=z_mask)
        z = z + self.dropout_row(ta_update)
        del ta_update

        ta_update = self.tri_att_end(z, mask=z_mask)
        z = z + self.dropout_col(ta_update)
        del ta_update

        return z
