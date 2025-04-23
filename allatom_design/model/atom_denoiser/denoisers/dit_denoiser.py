from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import const
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.interpolants.ad_interpolants.sd3_rf_interpolant import \
    SD3_RF
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.dit_utils import (
    DiTBlock, FinalLayer, MultiHeadRMSNorm, MMDiTBlock)
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.motif_embedders import \
    MotifEmbedder
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.timestep_embedders import \
    TimestepEmbedder
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import FAMPNN
from openfold.model.primitives import Linear


class DiTDenoiser(nn.Module):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float]]):
        """
        Backbone diffusion with DiT
        """
        super().__init__()

        self.cfg = cfg
        self.task = cfg.get("task", "backbone")
        self.use_self_conditioning = cfg.use_self_conditioning

        # Set up scaffolding module
        self.motif_embedder = None
        if self.task == "scaffold":
            self.motif_embedder = MotifEmbedder(**cfg.motif_embedder)

        # Set up DiT
        self.interpolant = get_interpolant(cfg.interpolant, sigma_data)
        self.dit = DiT(cfg.dit, self.interpolant)

        # Autoguidance
        self.use_autoguidance = cfg.autoguidance.enabled
        if self.use_autoguidance:
            self.autoguidance_train_p = 1 / cfg.autoguidance.subsample_train_iter_mult
            self.guiding_model = DiT(OmegaConf.merge(cfg.dit, cfg.autoguidance.dit), self.interpolant)  # override with autoguidance config


    def setup(self):
        pass


    def forward(self,
                motif_inputs: dict[str, TensorType["b n ..."]],
                diffusion_inputs: dict[str, TensorType["b ..."]],
                is_sampling: bool = False,
                diffusion_params: dict[str, Any] | None = None,  # required only for sampling
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        if self.motif_embedder is not None:
            motif_inputs["motif_embed_1d"] = self.motif_embedder(motif_inputs)

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            diffusion_inputs=diffusion_inputs,
            motif_inputs=motif_inputs,
            is_sampling=is_sampling,
            diffusion_params=diffusion_params
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                           diffusion_inputs: dict[str, TensorType["b ..."]],
                           motif_inputs: dict[str, TensorType["b ..."]],
                           is_sampling: bool,
                           diffusion_params: dict[str, Any] | None,
                           ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred of backbone
                                      Dict[str, TensorType["b ..."]]]:
        B, N = diffusion_inputs["seq_mask"].shape
        diffusion_aux = defaultdict(lambda: None)

        if not is_sampling:
            ### TRAINING ###
            # Get ground truth backbone coordinates
            diffusion_inputs["x_bb"] = diffusion_inputs["x"][..., const.prot_bb_atom14_idxs, :]

            # Repeat inputs for batch multiplier  # TODO: randomly augment these too
            M = self.cfg.training_batch_size_mult
            diffusion_inputs_batched = {k: v[None].expand(M, *v.shape) if v is not None else None for k, v in diffusion_inputs.items()}
            diffusion_inputs_batched = {k: v.reshape(M * B, *v.shape[2:]) if v is not None else None for k, v in diffusion_inputs_batched.items()}

            # repeat conditioning inputs
            motif_inputs_batched = {k: v[None].expand(M, *v.shape) if v is not None else None for k, v in motif_inputs.items()}
            motif_inputs_batched = {k: v.reshape(M * B, *v.shape[2:]) if v is not None else None for k, v in motif_inputs_batched.items()}

            # Noise the ground truth backbone
            interpolant_out = self.interpolant({"x": diffusion_inputs_batched["x_bb"], "aatype": None}, t=diffusion_inputs_batched["t_bb"])
            x_bb_target_batched = interpolant_out["x_target"]
            xt_bb_batched = interpolant_out["x_noised"]
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Run denoising DiT
            denoiser_fn = self.dit
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    denoiser_pred_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                                   t_batched,
                                                                   seq_mask=diffusion_inputs_batched["seq_mask"],
                                                                   residue_index=diffusion_inputs_batched["residue_index"],
                                                                   motif_inputs=motif_inputs_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched, xt_bb_batched, t_batched))

            denoiser_pred_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                           t_batched,
                                                           seq_mask=diffusion_inputs_batched["seq_mask"],
                                                           residue_index=diffusion_inputs_batched["residue_index"],
                                                           motif_inputs=motif_inputs_batched)

            # Train autoguidance model
            diffusion_aux["autoguidance_aux"] = None
            if self.use_autoguidance and (np.random.uniform() < self.autoguidance_train_p):
                ### If memory spikes due to running the autoguidance model,
                ### consider activation checkpointing, separate optimization steps, alternating head predictions, or just training the models separately.
                denoiser_fn = self.guiding_model
                if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                    with torch.no_grad():
                        denoiser_pred_batched_guide, _ = denoiser_fn(xt_bb_batched,
                                                                     t_batched,
                                                                     seq_mask=diffusion_inputs_batched["seq_mask"],
                                                                     residue_index=diffusion_inputs_batched["residue_index"],
                                                                     motif_inputs=motif_inputs_batched)
                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched_guide, xt_bb_batched, t_batched))

                denoiser_pred_batched_guide, _ = denoiser_fn(xt_bb_batched,
                                                             t_batched,
                                                             seq_mask=diffusion_inputs_batched["seq_mask"],
                                                             residue_index=diffusion_inputs_batched["residue_index"],
                                                             motif_inputs=motif_inputs_batched)

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "bb_pred": denoiser_pred_batched_guide,
                    "bb_target": x_bb_target_batched,  # diffusion target; for edm this is just the ground truth coordinates
                    "loss_weight_t": loss_weight_t_batched,
                    "atom_mask": diffusion_inputs_batched["atom_mask"],
                    "motif_inputs_batched": motif_inputs_batched,
                    "diffusion_inputs_batched": diffusion_inputs_batched
                }

            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = denoiser_pred_batched
            diffusion_aux["bb_target"] = x_bb_target_batched  # diffusion target; for edm this is just the ground truth coordinates
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched
            diffusion_aux["atom_mask"] = diffusion_inputs_batched["atom_mask"]
            diffusion_aux["motif_inputs_batched"] = motif_inputs_batched
            diffusion_aux["diffusion_inputs_batched"] = diffusion_inputs_batched

        else:
            ### SAMPLING ###

            # Sample backbone from prior
            A = len(const.prot_bb_atoms)
            x0_bb = self.interpolant.sample_prior((B, N, A, 3), diffusion_inputs["seq_mask"].device)

            # Store trajectory
            xt_bb_traj, x1_bb_traj = [], []

            # Extract sampling parameters
            S = diffusion_params["num_steps"]
            timesteps = diffusion_params["timesteps"]
            noise_schedule = diffusion_params["noise_schedule"]
            churn_cfg = diffusion_params["churn_cfg"]
            autoguidance_cfg = diffusion_params["autoguidance_cfg"]

            # Apply autoguidance
            use_autoguidance = (autoguidance_cfg is not None) and (autoguidance_cfg["use_autoguidance"])
            if use_autoguidance:
                assert self.use_autoguidance, "Model must be trained with autoguidance to use it."
                autoguidance_cfg["autoguidance_fn"] = partial(self.guiding_model,
                                                              residue_index=diffusion_inputs["residue_index"],
                                                              seq_mask=diffusion_inputs["seq_mask"],
                                                              motif_inputs=motif_inputs)

            # Run integration steps
            denoiser_fn = partial(self.dit,
                                  residue_index=diffusion_inputs["residue_index"],
                                  seq_mask=diffusion_inputs["seq_mask"],
                                  motif_inputs=motif_inputs)

            xt_bb = x0_bb
            for i in tqdm(range(S), leave=False, desc="Sampling..."):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]

                xt_bb, t = self.interpolant.churn(xt_bb, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

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
                                                               autoguidance_cfg=autoguidance_cfg,)

                # Save current state
                xt_bb_traj.append(xt_bb.cpu())

                # Save current x1 prediction
                x1_bb_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            x1_bb = xt_bb
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=1)  # [B S N A 3]
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=1)  # [B S N A 3]

        return x1_bb, diffusion_aux


class DiT(nn.Module):
    def __init__(self, cfg: DictConfig, interpolant: ADInterpolant):
        """
        DiT for unconditional backbone diffusion
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = interpolant

        # Set up DiT model
        self.num_atoms_in = cfg.num_atoms_in
        self.use_self_conditioning = cfg.use_self_conditioning

        # Input and output channels
        self.c = self.num_atoms_in * 3  # 3 xyz coordinates per atom
        self.in_channels = self.c * 2 if self.use_self_conditioning else self.c  # 2x for self-conditioning

        self.out_channels = self.c
        self.n_aatype = cfg.n_aatype

        # Use motif conditioning
        self.use_motif_conditioning = cfg.get("task", "backbone") == "scaffold"

        # Model parameters
        self.num_heads = cfg.num_heads
        self.pos_encoding = cfg.pos_encoding

        self.x_embedder = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")  # "glorot" should match DiT Patchify init

        # Positional encodings
        assert self.pos_encoding in ["absolute", "absolute_residx", "rotary", "rotary_residx", "af2"]
        if self.pos_encoding in ["absolute", "absolute_residx"]:
            self.pos_embed = posemb_sincos_1d

        self.rotary_emb = None
        if self.pos_encoding in ["rotary", "rotary_residx"]:
            dim = cfg.hidden_size // cfg.num_heads
            use_residx = (self.pos_encoding == "rotary_residx")
            self.rotary_emb = rope.RotaryEmbedding(dim=dim, use_residx=use_residx, cache_if_possible=False)

        # Time embeddings
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)

        # QK-normalization from SD3
        self.qk_normlayer = None
        if cfg.qk_rmsnorm:
            self.qk_normlayer = partial(MultiHeadRMSNorm, heads=cfg.num_heads)

        # Blocks
        if self.use_motif_conditioning:
            # Multi-modal DiT block for separate weights for backbone and motif
            block = MMDiTBlock
        else:
            block = DiTBlock

        self.blocks = nn.ModuleList([
            block(cfg.hidden_size, cfg.num_heads,
                  mlp_dropout=cfg.mlp_dropout, mlp_ratio=cfg.mlp_ratio,
                  inf=cfg.inf,
                  rotary_emb=self.rotary_emb,
                  qk_norm=cfg.qk_rmsnorm, norm_layer=self.qk_normlayer) for _ in range(cfg.depth)
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
        self.blocks.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if isinstance(block, MMDiTBlock):
                for i in range(block.n_modalities):
                    nn.init.constant_(block.adaLN_modulations[i][-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulations[i][-1].bias, 0)
            else:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self,
                x_noised: TensorType["b n 4 3", float],
                t: TensorType["b", float],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                motif_inputs: dict[str, TensorType["b ..."]],
                x_self_cond: Optional[TensorType["b n 4 3", float]] = None,
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

        # Begin DiT forward pass
        x = self.x_embedder(x)

        # Motif conditioning
        residx_mask = None
        token_modality_idxs = None
        if self.use_motif_conditioning:
            # For now, concatenate motif as extra tokens
            N = x.shape[1]
            token_modality_idxs = torch.tensor([0, seq_mask.shape[1]], device=x.device)  # starting index for backbone and motif tokens

            x = torch.cat([x, motif_inputs["motif_embed_1d"]], dim=1)
            seq_mask = torch.cat([seq_mask, motif_inputs["token_pad_mask"]], dim=1)
            residx_mask = torch.cat([torch.ones_like(residue_index), motif_inputs["residx_mask"]], dim=1)  # for preventing RoPE from being applied to motif tokens
            residue_index = torch.cat([residue_index, motif_inputs["residue_index"]], dim=1)
            residue_index = residue_index * residx_mask  # mask out residx

        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        # Conditioning
        c = self.t_embedder(t)
        c = c.unsqueeze(1).expand((-1, x.shape[1], -1))  # expand to sequence length

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None, per_token_conditioning=True,
                      rope_mask=residx_mask, token_modality_idxs=token_modality_idxs)

        # Remove motif conditioning tokens
        if self.use_motif_conditioning:
            x = x[:, :N, :]
            c = c[:, :N, :]
            seq_mask = seq_mask[:, :N]

        # Final layer
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3).float()  # ensure we're not in bf16
        x = precondition_out(x)  # output preconditioning

        return x, aux_preds


def get_interpolant(cfg: DictConfig,
                    sigma_data: TensorType[(), float]
                    ) -> ADInterpolant:
    """
    Get the interpolant specified in the config.
    """
    if cfg.name == "edm":
        return EDM(cfg, sigma_data)
    elif cfg.name == "sd3_rf":
        return SD3_RF(cfg)
    else:
        raise ValueError(f"Unknown interpolant: {cfg.name}")
