from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import life
from allatom_design.data.data import (cat_bb_scn, get_rc_tensor,
                                      transform_sidechain_frame)
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import \
    FinalLayer
from allatom_design.model.atom_denoiser.denoisers.dit_utils import \
    DenoisingMLPBlock
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.sidechain_confidence import \
    SidechainConfidenceModule
from openfold.model.primitives import Linear


class SidechainDiffusionModule(nn.Module):
    def __init__(self, cfg: DictConfig, scn_sigma_data: TensorType[(), float]):
        """
        Sidechain denoising module. For now, basically just a small DiT.
        """
        super().__init__()
        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.scn_interpolant = EDM(cfg.interpolant, sigma_data=scn_sigma_data)

        # Set up denoising model
        self.scn_denoiser = SidechainMLP(cfg.scn_denoiser, self.scn_interpolant)

        # Confidence module
        self.use_confidence_module = cfg.confidence_module.enabled
        if self.use_confidence_module:
            self.confidence_module_train_p = 1 / cfg.confidence_module.subsample_train_iter_mult
            self.confidence_module = SidechainConfidenceModule(cfg.confidence_module)


    def sidechain_diffusion(self,
                            mpnn_feature_dict: Dict[str, TensorType["b ..."]],
                            aatype: TensorType["b n", int],
                            seq_mask: TensorType["b n", float],
                            residue_index: TensorType["b n", int],
                            chain_index: TensorType["b n", int],
                            aux_inputs: Optional[Dict],
                            is_sampling: bool,
                            ) -> Tuple[TensorType["b n a 3", float],
                                       Dict[str, TensorType["b ...", float]]]:
        h_V = mpnn_feature_dict["h_V"]
        B, N, _ = h_V.shape
        diffusion_aux = defaultdict(lambda: None)

        if not is_sampling:
            # === Training === #
            # Teacher forcing: use ground truth aatype
            aatype = aux_inputs["aatype"]

            # Get ground truth sidechains for diffusion
            x_scn_gt = aux_inputs["x"][..., rc.non_bb_idxs, :]

            # Transform sidechains from ground truth to local frame
            x_scn_local_gt, _ = transform_sidechain_frame(x_scn_gt,
                                                          aux_inputs["x"][..., rc.bb_idxs, :],
                                                          aux_inputs["atom_mask"][..., rc.non_bb_idxs],
                                                          aux_inputs["atom_mask"][..., rc.bb_idxs],
                                                          to_local=True)

            # Repeat inputs for batch multiplier
            M = self.cfg.training_batch_size_mult
            x_scn_local_gt_batched = repeat(x_scn_local_gt, "b n a x -> (m b) n a x", m=M, b=B)
            h_V_batched = repeat(h_V, "b n h -> (m b) n h", m=M, b=B)
            aatype_batched = repeat(aatype, "b n -> (m b) n", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)

            # Evaluate at specific timesteps (for validation)
            t_sd_batched = None
            if aux_inputs["t_scd"] is not None:
                t_sd_batched = torch.full((M * B, ), aux_inputs["t_scd"], device=x_scn_local_gt_batched.device)

            # Noise the ground truth local sidechains
            interpolant_out = self.scn_interpolant({"x": x_scn_local_gt_batched, "aatype": aatype_batched}, t=t_sd_batched)
            xt_scn_local_batched = interpolant_out["x_noised"]
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Run small denoising MLP
            denoiser_fn = self.scn_denoiser
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_scn_local_batched, aux_preds = denoiser_fn(xt_scn_local_batched, aatype_batched, t_batched, h_V_batched, seq_mask=seq_mask_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=x1_scn_local_batched)

            x1_scn_local_batched, aux_preds = denoiser_fn(xt_scn_local_batched, aatype_batched, t_batched, h_V_batched, seq_mask=seq_mask_batched)

            # Train confidence module
            diffusion_aux["confidence_aux"] = None
            if self.use_confidence_module and (np.random.uniform() < self.confidence_module_train_p):
                # Run mini rollout
                with torch.no_grad():
                    self.eval()
                    # use unbatched inputs
                    x1_scn_local_rollout = self.mini_rollout(h_V, aatype, seq_mask)  # in local frmae
                    self.train()

                # Run confidence module
                mpnn_feature_dict_in = {k: v.detach() for k, v in mpnn_feature_dict.items()}
                psce_logits, psce = self.confidence_module(x1_scn_local_rollout.detach(),
                                                           mpnn_feature_dict_in,
                                                           aatype.detach(),
                                                           seq_mask.detach(),
                                                           residue_index.detach(),
                                                           chain_index.detach())
                diffusion_aux["confidence_aux"] = {
                    "psce_logits": psce_logits,
                    "psce": psce,
                    "sce_bins_cfg": self.confidence_module.sce_bins_cfg,
                    "scn_pred_rollout": x1_scn_local_rollout,  # compute loss in local frame
                    "scn_target": x_scn_local_gt,
                }

            # Outputs
            x1_scn = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["scn_pred"] = x1_scn_local_batched
            diffusion_aux["scn_target"] = x_scn_local_gt_batched
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched

        else:
            # === Sampling === #

            # Sample sidechains from prior
            A = len(rc.non_bb_idxs)
            x0_scn_local = self.scn_interpolant.sample_prior((B, N, A, 3), h_V.device)

            # Extract sampling parameters
            scd_aux_inputs = aux_inputs["scd"]
            S_scd = scd_aux_inputs["num_steps"]
            timesteps = scd_aux_inputs["timesteps"]
            churn_cfg = scd_aux_inputs["churn_cfg"]
            noise_schedule = scd_aux_inputs["noise_schedule"]
            return_scn_diffusion_aux = scd_aux_inputs.get("return_scn_diffusion_aux", False)
            aatype_override_mask = scd_aux_inputs["aatype_override_mask"]
            aatype = torch.where(aatype_override_mask.bool(), scd_aux_inputs["aatype_override"], aatype)

            denoiser_fn = partial(self.scn_denoiser, aatype=aatype,
                                  h_V=h_V, seq_mask=seq_mask)
            # Run integration steps
            # Store trajectory
            xt_scn_traj, x1_scn_traj = [], []

            xt_scn_local = x0_scn_local
            for i in range(S_scd):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]

                xt_scn_local, t = self.scn_interpolant.churn(xt_scn_local, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

                xt_scn_local, aux_preds = self.scn_interpolant.euler_step(denoiser_fn,
                                                                          xt_scn_local,
                                                                          t=t, t_next=t_next,
                                                                          noise_schedule=noise_schedule,
                                                                          autoguidance_cfg=None,
                                                                          cfg_cfg=None)

                if self.use_self_conditioning:
                    # Apply self-conditioning
                    denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux_preds["x1_pred"])

                if return_scn_diffusion_aux:
                    # Save current state
                    xt_scn_traj.append(xt_scn_local.cpu())

                    # Save current x1 prediction
                    x1_scn_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            # Compute confidence using local scn coordinates
            if self.use_confidence_module:
                _, psce = self.confidence_module(xt_scn_local,
                                                 mpnn_feature_dict,
                                                 aatype,
                                                 seq_mask,
                                                 residue_index,
                                                 chain_index)
                diffusion_aux["psce"] = psce
            else:
                diffusion_aux["psce"] = torch.zeros((B, N, A), device=xt_scn_local.device)

            # Transform denoised sidechains back to global coordinates
            x_bb = mpnn_feature_dict["X"][..., rc.atom14_bb_idxs, :]
            atom_mask = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)  # assume all atoms are present  # TODO: we should pass in true atom mask here to account for missing backbone
            atom_mask_bb, atom_mask_scn = atom_mask[..., rc.bb_idxs], atom_mask[..., rc.non_bb_idxs]
            x1_scn, _ = transform_sidechain_frame(xt_scn_local, x_bb,
                                                  atom_mask_scn, atom_mask_bb, to_local=False)
            diffusion_aux["scn_pred"] = x1_scn

            # Finalize trajectory outputs
            if return_scn_diffusion_aux:
                xt_scn_traj = torch.stack(xt_scn_traj, dim=1)  # (B, S_scd, N, A, 3)
                xt_scn_traj, _ = transform_sidechain_frame(xt_scn_traj, x_bb[:, None].cpu(),
                                                           atom_mask_scn[:, None].cpu(), atom_mask_bb[:, None].cpu(),
                                                           to_local=False)
                diffusion_aux["xt_scn_traj"] = xt_scn_traj

                x1_scn_traj = torch.stack(x1_scn_traj, dim=1)  # (B, S_scd, N, A, 3)
                x1_scn_traj, _ = transform_sidechain_frame(x1_scn_traj, x_bb[:, None].cpu(),
                                                           atom_mask_scn[:, None].cpu(), atom_mask_bb[:, None].cpu(),
                                                           to_local=False)
                diffusion_aux["x1_scn_traj"] = x1_scn_traj

        return x1_scn, diffusion_aux


    def mini_rollout(self,
                     h_V: TensorType["b n h", float],
                     aatype: TensorType["b n", int],
                     seq_mask: TensorType["b n", float]) -> TensorType["b n 33 3", float]:
        B, N, _ = h_V.shape
        A = len(rc.non_bb_idxs)

        # Create sidechain diffusion inputs
        conf_cfg = self.cfg.confidence_module

        # create timesteps
        S_scd = conf_cfg.scn_diffusion.num_steps
        timesteps = sampling_utils.get_timesteps_from_schedule(**conf_cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time
        timesteps = timesteps[None].expand(B, -1).to(h_V.device)  # expand to batch size

        # create noise schedule
        noise_schedule = NoiseSchedule(conf_cfg.scn_diffusion.noise_schedule)

        # create churn config
        churn_cfg = dict(conf_cfg.scn_diffusion.churn_cfg)

        # Sample sidechains from prior
        x0_scn_local = self.scn_interpolant.sample_prior((B, N, A, 3), h_V.device)

        # Run integration steps
        denoiser_fn = partial(self.scn_denoiser, aatype=aatype,
                              h_V=h_V, seq_mask=seq_mask)

        xt_scn_local = x0_scn_local
        for i in range(S_scd):
            t = timesteps[:, i]
            t_next = timesteps[:, i + 1]
            xt_scn_local, t = self.scn_interpolant.churn(xt_scn_local, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling
            xt_scn_local, aux_preds = self.scn_interpolant.euler_step(denoiser_fn,
                                                                        xt_scn_local,
                                                                        t=t, t_next=t_next,
                                                                        noise_schedule=noise_schedule,
                                                                        autoguidance_cfg=None,
                                                                        cfg_cfg=None)
            if self.use_self_conditioning:
                # Apply self-conditioning
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux_preds["x1_pred"])

        x1_scn_local = xt_scn_local

        # Return sidechains in local frame
        return x1_scn_local


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

        denoiser_fn = partial(self.scn_denoiser, aatype=aatype, x_bb=x_bb,
                              h_V=h_V, seq_mask=seq_mask, scd_mlm_mask=scd_mlm_mask,
                              residue_index=residue_index, chain_index=chain_index)

        x1_mask = scn_atom_mask * rearrange(scd_mlm_mask, "b n -> b n 1 1")

        likelihood_aux = self.scn_interpolant.get_likelihoods(denoiser_fn, x1_scn, x1_mask, S_scd)

        # Preprocess trajectory output
        x_bb_traj = x_bb[:, None].expand(-1, S_scd, -1, -1, -1).cpu()
        likelihood_aux["likelihood_xt_traj"] = likelihood_aux["likelihood_xt_traj"] + x_bb_traj[..., 1:2, :]  # undo centering of sidechain coordinates on CA
        likelihood_aux["likelihood_xt_traj"] = cat_bb_scn(x_bb_traj, likelihood_aux["likelihood_xt_traj"])  # put scn coords back into full structure
        return likelihood_aux


class SidechainMLP(nn.Module):
    def __init__(self, cfg: DictConfig, scn_interpolant: ADInterpolant):
        """
        MLP for per-token sidechain diffusion conditioned on MPNN sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.scn_interpolant = scn_interpolant

        # Set up MLP model
        self.use_self_conditioning = cfg.use_self_conditioning
        self.in_channels = len(rc.non_bb_idxs) * 3  # 33 * 3; input sidechain atoms
        self.in_channels += cfg.n_aatype  # concatenate one-hot encoded amino acid type

        self.out_channels = len(rc.non_bb_idxs) * 3  # 33 * 3; output all sidechain atoms
        if self.use_self_conditioning:
            self.in_channels += self.out_channels  # concatenate input with output from previous timestep

        self.n_aatype = cfg.n_aatype

        # Conditioning
        self.timestep_embedder = TimestepEmbedder(cfg.hidden_size)

        # input feature embedder: embed reference positions
        self.f_embedder = Linear(cfg.num_atoms_in * 3, cfg.hidden_size)

        # node embedding conditioning
        self.h_V_embedder = nn.Linear(cfg.c_h_V, cfg.hidden_size)

        # Blocks
        self.x_embedder = Linear(self.in_channels, cfg.hidden_size, bias=True, init="glorot")

        # Blocks
        self.blocks = nn.ModuleList([
            DenoisingMLPBlock(cfg.hidden_size,
                              mlp_dropout=cfg.mlp_dropout,
                              mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.depth)
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
                seq_mask: TensorType["b n", float],
                x_scn_self_cond: Optional[TensorType["b n a_scn 3", float]] = None,  # self-conditioning input
                ) -> Tuple[TensorType["b n a 3", float], Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        # Preconditioning
        precondition_in, precondition_out = self.scn_interpolant.setup_preconditioning(x_scn, x_scn_self_cond, t)
        x_scn, x_scn_self_cond, t = precondition_in()  # input preconditioning

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_scn_self_cond is None:
                x_scn_self_cond = torch.zeros_like(x_scn)
            x_scn = torch.cat([x_scn, x_scn_self_cond], dim=-1)

        x = rearrange(x_scn, "b n a x -> b n (a x)")

        # Concatenate one-hot sequence conditioning
        aatype_oh = F.one_hot(aatype, num_classes=self.n_aatype).float()  # aatype is ground truth during training
        x = torch.cat([x, aatype_oh], dim=-1)

        # Begin MLP forward pass
        x = self.x_embedder(x)

        # Embed reference positions
        ref_pos = life.RESTYPE_REF_POS_ATOM37.to(aatype.device)[aatype.long()]
        ref_pos = rearrange(ref_pos, "b n a x -> b n (a x)")
        x = x + self.f_embedder(ref_pos)

        # Conditioning
        # embed timestep
        c = self.timestep_embedder(t).unsqueeze(1)

        # add conditioning from h_V
        h_V = self.h_V_embedder(h_V)
        c = c + h_V
        x = x + h_V

        # MLP blocks
        for block in self.blocks:
            x = block(x, c)

        # Final output
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x_scn = precondition_out(x)  # output preconditioning on sidechains

        return x_scn, aux_preds
