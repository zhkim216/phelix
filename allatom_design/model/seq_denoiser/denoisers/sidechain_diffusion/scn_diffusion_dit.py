from collections import defaultdict
from functools import partial
from typing import Dict, Final, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data.data import cat_bb_scn
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import (
    DiTBlock, FinalLayer, MultiHeadRMSNorm)
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.sidechain_confidence import \
    SidechainConfidenceModule
from openfold.model.primitives import Linear
from allatom_design.data import life


class SidechainDiffusionModule(nn.Module):
    def __init__(self, cfg: DictConfig, scn_sigma_data: TensorType[(), float]):
        """
        Sidechain denoising module. For now, basically just a small DiT.
        """
        super().__init__()
        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.future_unmasking_schedule = getattr(cfg, "future_unmasking_schedule", None)
        self.scn_interpolant = EDM(cfg.interpolant, sigma_data=scn_sigma_data)

        # Set up DiT model
        self.dit = SidechainDiT(cfg.dit, self.scn_interpolant)

        # Autoguidance
        self.use_autoguidance = cfg.autoguidance.enabled
        if self.use_autoguidance:
            self.autoguidance_train_p = 1 / cfg.autoguidance.subsample_train_iter_mult
            self.guiding_model = SidechainDiT(OmegaConf.merge(cfg.dit, cfg.autoguidance.dit), self.scn_interpolant)  # override with autoguidance config

        # Confidence module
        self.use_confidence_module = cfg.confidence_module.enabled
        if self.use_confidence_module:
            self.confidence_module_train_p = 1 / cfg.confidence_module.subsample_train_iter_mult
            self.confidence_module = SidechainConfidenceModule(cfg.confidence_module)


    def sidechain_diffusion(self,
                            h_V: TensorType["b n h", float],
                            h_ESV: TensorType["b n h", float],
                            aatype: TensorType["b n", int],
                            x_bb: TensorType["b n a_bb 3", float],
                            seq_mask: TensorType["b n", float],
                            residue_index: TensorType["b n", int],
                            chain_index: TensorType["b n", int],
                            aux_inputs: Optional[Dict],
                            is_sampling: bool,
                            ) -> Tuple[TensorType["b n a 3", float],
                                       Dict[str, TensorType["b ...", float]]]:
        B, N, _ = h_V.shape
        diffusion_aux = defaultdict(lambda: None)

        if not is_sampling:
            # === Training === #
            # Teacher forcing: use ground truth aatype
            aatype = aux_inputs["aatype"]

            # Get ground truth sidechains for diffusion
            x_scn_gt = aux_inputs["x"][..., rc.non_bb_idxs, :]

            # Center sidechains on CA
            x_scn_gt = x_scn_gt - aux_inputs["x"][..., 1:2, :]
            scn_missing_atom_mask = aux_inputs["missing_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are missing
            x_scn_gt = torch.where(scn_missing_atom_mask[..., None].bool(), 0, x_scn_gt)  # fill missing atoms with zeroes
            scn_ghost_atom_mask = aux_inputs["ghost_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are not in the residue type
            x_scn_gt = torch.where(scn_ghost_atom_mask[..., None].bool(), 0, x_scn_gt)  # fill ghost atoms with zeroes

            # Repeat inputs for batch multiplier
            M = self.cfg.training_batch_size_mult
            x_scn_gt_batched = repeat(x_scn_gt, "b n a x -> (m b) n a x", m=M, b=B)
            h_V_batched = repeat(h_V, "b n h -> (m b) n h", m=M, b=B)
            aatype_batched = repeat(aatype, "b n -> (m b) n", m=M, b=B)
            x_bb_batched = repeat(x_bb, "b n a x -> (m b) n a x", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)
            mlm_mask_batched = repeat(aux_inputs["seq_mlm_mask"], "b n -> (m b) n", m=M, b=B)
            residue_index_batched = repeat(residue_index, "b n -> (m b) n", m=M, b=B)
            chain_index_batched = repeat(chain_index, "b n -> (m b) n", m=M, b=B)

            # Evaluate at specific timesteps (for validation)
            t_sd_batched = None
            if aux_inputs["t_scd"] is not None:
                t_sd_batched = torch.full((M * B, ), aux_inputs["t_scd"], device=x_scn_gt_batched.device)

            # Noise the ground truth sidechains
            interpolant_out = self.scn_interpolant({"x": x_scn_gt_batched, "aatype": aatype_batched}, t=t_sd_batched)
            xt_scn_batched = interpolant_out["x_noised"]
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Randomly unmask future residues to pack for training
            scd_mlm_mask_batched = self.unmask_future_residues(mlm_mask_batched, seq_mask_batched)
            x_scn_gt_batched = x_scn_gt_batched * rearrange(scd_mlm_mask_batched, "(m b) n -> (m b) n 1 1", m=M)

            # Run small denoising DiT
            denoiser_fn = self.dit
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_scn_batched, aux_preds = denoiser_fn(xt_scn_batched, aatype_batched, t_batched, h_V_batched, x_bb_batched,
                                                            seq_mask=seq_mask_batched, scd_mlm_mask=scd_mlm_mask_batched,
                                                            residue_index=residue_index_batched, chain_index=chain_index_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=x1_scn_batched)

            x1_scn_batched, aux_preds = denoiser_fn(xt_scn_batched, aatype_batched, t_batched, h_V_batched, x_bb_batched,
                                                    seq_mask=seq_mask_batched, scd_mlm_mask=scd_mlm_mask_batched,
                                                    residue_index=residue_index_batched, chain_index=chain_index_batched)

            # Train autoguidance model
            diffusion_aux["autoguidance_aux"] = None
            if self.use_autoguidance and (np.random.uniform() < self.autoguidance_train_p):
                ### If memory spikes due to running the autoguidance model,
                ### consider activation checkpointing, separate optimization steps, alternating head predictions, or just training the models separately.
                denoiser_fn = self.guiding_model
                if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                    with torch.no_grad():
                        x1_scn_batched_guide, _ = denoiser_fn(xt_scn_batched, aatype_batched, t_batched,
                                                              h_V_batched.detach(), x_bb_batched,
                                                              seq_mask=seq_mask_batched, scd_mlm_mask=scd_mlm_mask_batched,
                                                              residue_index=residue_index_batched, chain_index=chain_index_batched)

                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_scn_self_cond=x1_scn_batched_guide)

                x1_scn_batched_guide, _ = denoiser_fn(xt_scn_batched, aatype_batched, t_batched,
                                                      h_V_batched.detach(), x_bb_batched,
                                                      seq_mask=seq_mask_batched, scd_mlm_mask=scd_mlm_mask_batched,
                                                      residue_index=residue_index_batched, chain_index=chain_index_batched)

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "scn_pred": x1_scn_batched_guide,
                    "scn_target": x_scn_gt_batched,
                    "loss_weight_t": loss_weight_t_batched,
                    "scd_mlm_mask": scd_mlm_mask_batched,
                }

            # Train confidence module
            diffusion_aux["confidence_aux"] = None
            if self.use_confidence_module and (np.random.uniform() < self.confidence_module_train_p):
                # Use unbatched inputs
                conf_cfg = self.cfg.confidence_module

                with torch.no_grad():
                    self.eval()

                    # Create sidechain diffusion inputs

                    # create timesteps
                    B = h_V.shape[0]
                    t_scd = sampling_utils.get_timesteps_from_schedule(**conf_cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time
                    t_scd = t_scd[None].expand(B, -1).to(h_V.device)  # expand to batch size

                    # create noise schedule
                    noise_schedule = NoiseSchedule(conf_cfg.scn_diffusion.noise_schedule)

                    # create churn config
                    churn_cfg = dict(conf_cfg.scn_diffusion.churn_cfg)
                    scd_inputs = {"num_steps": conf_cfg.scn_diffusion.num_steps,
                                  "timesteps": t_scd,
                                  "noise_schedule": noise_schedule,
                                  "churn_cfg": churn_cfg,
                                  "autoguidance_cfg": dict(conf_cfg.scn_diffusion.autoguidance_cfg),
                                  }

                    # Randomly choose some residues to unmask for rollout
                    scd_mlm_mask_rollout = self.unmask_future_residues(aux_inputs["seq_mlm_mask"], seq_mask)
                    x_scn_gt_rollout = x_scn_gt * rearrange(scd_mlm_mask_rollout, "b n -> b n 1 1")

                    # Run diffusion mini rollout
                    rollout_aux_inputs = {"seq_mlm_mask": scd_mlm_mask_rollout,
                                          "scd": scd_inputs}
                    x1_scn_rollout, _ = self.sidechain_diffusion(h_V, h_ESV, aatype, x_bb,
                                                                 seq_mask, residue_index, chain_index,
                                                                 aux_inputs=rollout_aux_inputs,
                                                                 is_sampling=True)

                    x1_scn_rollout = x1_scn_rollout - x_bb[..., 1:2, :]  # center sidechains on input backbone to sidechain diffusion  # TODO: fix this

                    self.train()

                psce_logits = self.confidence_module(x1_scn_rollout.detach(),
                                               h_V.detach(),
                                               h_ESV.detach(),
                                               aatype.detach(),
                                               x_bb.detach(),
                                               seq_mask.detach(),
                                               residue_index.detach(),
                                               chain_index.detach(),
                                               scd_mlm_mask=scd_mlm_mask_rollout.detach(),
                                               )
                diffusion_aux["confidence_aux"] = {
                    "psce_logits": psce_logits,
                    "sce_bins_cfg": self.confidence_module.sce_bins_cfg,
                    "scn_pred_rollout": x1_scn_rollout,
                    "scn_target": x_scn_gt_rollout,
                    "scd_mlm_mask": scd_mlm_mask_rollout,
                }

            # Outputs
            x1_scn = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["scn_pred"] = x1_scn_batched
            diffusion_aux["scn_target"] = x_scn_gt_batched
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched
            diffusion_aux["scd_mlm_mask"] = scd_mlm_mask_batched

        else:
            # === Sampling === #

            # Sample sidechains from prior
            A = len(rc.non_bb_idxs)
            x0_scn = self.scn_interpolant.sample_prior((B, N, A, 3), h_V.device)

            # Extract sampling parameters
            scd_aux_inputs = aux_inputs["scd"]
            S_scd = scd_aux_inputs["num_steps"]
            timesteps = scd_aux_inputs["timesteps"]
            churn_cfg = scd_aux_inputs["churn_cfg"]
            noise_schedule = scd_aux_inputs["noise_schedule"]
            autoguidance_cfg = scd_aux_inputs["autoguidance_cfg"]
            return_scn_diffusion_aux = scd_aux_inputs.get("return_scn_diffusion_aux", False)
            aatype = scd_aux_inputs.get("aatype_override", aatype)  # use aatype_override for sidechain diffusion instead

            # Only pack residues that are unmasked
            scd_mlm_mask = aux_inputs["seq_mlm_mask"]

            # Apply autoguidance
            use_autoguidance = (autoguidance_cfg is not None) and (autoguidance_cfg["use_autoguidance"])
            if use_autoguidance:
                assert self.use_autoguidance, "Model must be trained with autoguidance to use it."
                autoguidance_cfg["autoguidance_fn"] = partial(self.guiding_model, aatype=aatype, x_bb=x_bb,
                                                              h_V=h_V, seq_mask=seq_mask,
                                                              residue_index=residue_index, chain_index=chain_index,)


            denoiser_fn = partial(self.dit, aatype=aatype, x_bb=x_bb,
                                  h_V=h_V, seq_mask=seq_mask, scd_mlm_mask=scd_mlm_mask,
                                  residue_index=residue_index, chain_index=chain_index)
            # Run integration steps
            # Store trajectory
            xt_scn_traj, x1_scn_traj = [], []

            xt_scn = x0_scn
            for i in range(S_scd):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]

                xt_scn, t = self.scn_interpolant.churn(xt_scn, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

                xt_scn, aux_preds = self.scn_interpolant.euler_step(denoiser_fn,
                                                                    xt_scn,
                                                                    t=t, t_next=t_next,
                                                                    noise_schedule=noise_schedule,
                                                                    autoguidance_cfg=autoguidance_cfg,
                                                                    cfg_cfg=None)

                if self.use_self_conditioning:
                    # Apply self-conditioning
                    denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux_preds["x1_pred"])

                    if use_autoguidance:
                        autoguidance_cfg["autoguidance_fn"] = partial(autoguidance_cfg["autoguidance_fn"],
                                                                      x_scn_self_cond=aux_preds["x1_pred_ag"])

                if return_scn_diffusion_aux:
                    # Save current state
                    xt_scn_traj.append(xt_scn.cpu())

                    # Save current x1 prediction
                    x1_scn_traj.append(aux_preds["x1_pred"].cpu())

            # Compute confidence
            if self.use_confidence_module:
                # TODO: uncenter by CA
                psce_logits = self.confidence_module(xt_scn,
                                                     h_V,
                                                     h_ESV,
                                                     aatype,
                                                     x_bb,
                                                     seq_mask,
                                                     residue_index,
                                                     chain_index,
                                                     scd_mlm_mask=scd_mlm_mask)
                psce = self.confidence_module.compute_psce(psce_logits)
                diffusion_aux["psce"] = psce
            else:
                diffusion_aux["psce"] = torch.zeros((B, N, A), device=xt_scn.device)

            # Finalize outputs
            x1_scn = xt_scn + x_bb[..., 1:2, :]  # undo centering of sidechain coordinates on CA
            diffusion_aux["scn_pred"] = x1_scn

            # Finalize trajectory outputs
            if return_scn_diffusion_aux:
                diffusion_aux["xt_scn_traj"] = torch.stack(xt_scn_traj, dim=1) + x_bb[:, None, :, 1:2, :].cpu()  # (B, S_scd, N, A, 3), undo centering
                diffusion_aux["x1_scn_traj"] = torch.stack(x1_scn_traj, dim=1) + x_bb[:, None, :, 1:2, :].cpu()  # (B, S_scd, N, A, 3), undo centering

        return x1_scn, diffusion_aux


    def get_likelihoods(self,
                        num_steps: int,
                        x1_scn: TensorType["b n a_scn 3"],  # not centered on CA
                        h_V: TensorType["b n h", float],
                        aatype: TensorType["b n", int],
                        x_bb: TensorType["b n a_bb 3", float],
                        seq_mask: TensorType["b n", float],
                        residue_index: TensorType["b n", int],
                        chain_index: TensorType["b n", int],
                        aux_inputs: Optional[Dict]):
        x1_scn = x1_scn - x_bb[:, :, 1:2, :]  # center sidechain coordinates on CA
        scn_atom_mask = aux_inputs["atom_mask"][..., rc.non_bb_idxs, None].expand_as(x1_scn)
        x1_scn = torch.where(scn_atom_mask.bool(), x1_scn, 0)  # re-fill missing / ghost atoms with zeroes

        # DEBUG
        x1_scn = x1_scn * 0  # hack: for some reason, zeroing out sidechain atoms gives us better (inverse) correlations

        # Extract sampling parameters
        scd_aux_inputs = aux_inputs["scd"]
        S_scd = num_steps

        # Only pack residues that are unmasked
        scd_mlm_mask = aux_inputs["seq_mlm_mask"]

        denoiser_fn = partial(self.dit, aatype=aatype, x_bb=x_bb,
                              h_V=h_V, seq_mask=seq_mask, scd_mlm_mask=scd_mlm_mask,
                              residue_index=residue_index, chain_index=chain_index)

        x1_mask = scn_atom_mask * rearrange(scd_mlm_mask, "b n -> b n 1 1")

        likelihood_aux = self.scn_interpolant.get_likelihoods(denoiser_fn, x1_scn, x1_mask, S_scd)

        # Preprocess trajectory output
        x_bb_traj = x_bb[:, None].expand(-1, S_scd, -1, -1, -1).cpu()
        likelihood_aux["likelihood_xt_traj"] = likelihood_aux["likelihood_xt_traj"] + x_bb_traj[..., 1:2, :]  # undo centering of sidechain coordinates on CA
        likelihood_aux["likelihood_xt_traj"] = cat_bb_scn(x_bb_traj, likelihood_aux["likelihood_xt_traj"])  # put scn coords back into full structure
        return likelihood_aux


    def unmask_future_residues(self,
                               mlm_mask: TensorType["b n", float],
                               seq_mask: TensorType["b n", float],
                               ) -> TensorType["b n", float]:
        """
        For training, randomly unmask future residues (these are residues that are currently masked by MLM mask). If schedule is None, unmask all residues.
        """
        B = mlm_mask.shape[0]
        if self.future_unmasking_schedule is None:
            # Unmask all residues
            scd_mlm_mask = torch.ones_like(mlm_mask, device=mlm_mask.device, dtype=torch.bool)
        elif self.future_unmasking_schedule == "uniform":
            # Unmask probability is uniform
            p = torch.rand(B, device=mlm_mask.device)  # choose unmasking probability
            scd_mlm_mask = (torch.rand(mlm_mask.shape, device=mlm_mask.device) < p[:, None]) | mlm_mask.bool()  # unmask some currently masked residues; 0 for masked residues

        scd_mlm_mask = scd_mlm_mask.float() * seq_mask  # mask out padding
        return scd_mlm_mask


class SidechainDiT(nn.Module):
    def __init__(self, cfg: DictConfig, scn_interpolant: ADInterpolant):
        """
        DiT for backbone diffusion conditioned on MPNN sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.scn_interpolant = scn_interpolant

        # Set up DiT model
        self.use_self_conditioning = cfg.use_self_conditioning
        self.in_channels = cfg.num_atoms_in * 3  # 37 * 3; input all atoms
        self.in_channels += cfg.n_aatype  # concatenate one-hot encoded amino acid type
        self.out_channels = len(rc.non_bb_idxs) * 3  # 33 * 3; output all sidechain atoms

        self.n_aatype = cfg.n_aatype

        if self.use_self_conditioning:
            self.in_channels += self.out_channels  # concatenate input with output from previous timestep

        # Positional encodings
        self.pos_encoding = cfg.pos_encoding
        assert self.pos_encoding in ["absolute", "absolute_residx", "rotary", "rotary_residx"]
        if self.pos_encoding in ["absolute", "absolute_residx"]:
            self.pos_embed = posemb_sincos_1d

        self.rotary_emb = None
        if self.pos_encoding in ["rotary", "rotary_residx"]:
            dim = cfg.hidden_size // cfg.num_heads
            use_residx = (self.pos_encoding == "rotary_residx")
            self.rotary_emb = rope.RotaryEmbedding(dim=dim, use_residx=use_residx, cache_if_possible=False)

        self.timestep_embedder = TimestepEmbedder(cfg.hidden_size)
        self.x_embedder = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")

        # input feature embedder: embed reference positions
        # self.f_embedder = Linear(cfg.num_atoms_in * 3, cfg.hidden_size)

        # node embedding conditioning
        self.h_V_embedder = Linear(cfg.c_h_V, cfg.hidden_size)

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
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

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
                x_scn: TensorType["b n a_scn 3", float],  # noisy sidechain atoms
                aatype: TensorType["b n", float],  # aatype to condition on (predicted during inference; GT during training)
                t: TensorType["b n", float],  # timestep
                h_V: TensorType["b n h", float],  # conditioning latent
                x_bb: TensorType["b n a_bb 3", float],  # denoised backbone atoms
                seq_mask: TensorType["b n", float],
                scd_mlm_mask: TensorType["b n", float],  # for masking future aatypes from being packed
                residue_index: TensorType["b n", float],
                chain_index: TensorType["b n", float],
                x_scn_self_cond: Optional[TensorType["b n a_scn 3", float]] = None,  # self-conditioning input
                ) -> Tuple[TensorType["b n a 3", float], Dict[str, TensorType["b ..."]]]:
        """
        TODO: use chain index
        """

        aux_preds = {}

        # Only pack residues that are not masked
        aatype = torch.where(scd_mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])  # TODO: replace with MASK
        aatype = aatype * seq_mask.long()  # set pad residues back to 0
        x_scn = x_scn * rearrange(scd_mlm_mask, "b n -> b n 1 1")  # mask out sidechain coords of future aatypes

        # Preconditioning
        precondition_in, precondition_out = self.scn_interpolant.setup_preconditioning(x_scn, x_scn_self_cond, t)
        x_scn, x_scn_self_cond, t = precondition_in()  # input preconditioning

        # Concatenate denoised backbone atoms and noised sidechain atoms
        x = cat_bb_scn(x_bb, x_scn)
        x = rearrange(x, "b n a x -> b n (a x)")

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_scn_self_cond is None:
                x_scn_self_cond = torch.zeros_like(x_scn)
            x_scn_self_cond = rearrange(x_scn_self_cond, "b n a x -> b n (a x)")
            x = torch.cat([x, x_scn_self_cond], dim=-1)

        # Concatenate one-hot sequence conditioning
        aatype_oh = F.one_hot(aatype, num_classes=self.n_aatype).float()  # aatype is ground truth during training
        x = torch.cat([x, aatype_oh], dim=-1)

        # Begin DiT forward pass
        x = self.x_embedder(x)

        # # Embed reference positions
        # ref_pos = life.RESTYPE_REF_POS_ATOM37.to(aatype.device)[aatype.long()] * rearrange(scd_mlm_mask, "b n -> b n 1 1")
        # ref_pos = rearrange(ref_pos, "b n a x -> b n (a x)")
        # x = x + self.f_embedder(ref_pos)

        # Positional encodings
        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        # Conditioning
        c = self.timestep_embedder(t).unsqueeze(1)

        # add conditioning from h_V
        h_V = self.h_V_embedder(h_V)
        c = c + h_V

        # Blocks
        attn_mask = rearrange(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b 1 i j")
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None, per_token_conditioning=True)

        # Final output
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x_scn = precondition_out(x)  # output preconditioning on sidechains

        # Re-mask sidechain atoms of masked residues
        x_scn = x_scn * rearrange(scd_mlm_mask, "b n -> b n 1 1")

        return x_scn, aux_preds

