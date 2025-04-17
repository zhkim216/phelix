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

import allatom_design.data.conditioning_labels as cl
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.checkpoint_utils import repair_state_dict
from allatom_design.data import const
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.interpolants.ad_interpolants.sd3_rf_interpolant import \
    SD3_RF
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.dit_utils import (
    DiTBlock, FinalLayer, LabelEmbedder, MultiHeadRMSNorm)
from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.pair_rep_utils import (
    PairRepBuilder, RelativePositionalEncoding)
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
        self.use_self_conditioning = cfg.use_self_conditioning

        # Set up scaffolding module
        self.use_scaffold_module = cfg.get("scaffold_module", {}).get("enabled", False)
        if self.use_scaffold_module:
            self.fampnn = FAMPNN(cfg.scaffold_module.fampnn)

        # Set up DiT
        self.interpolant = get_interpolant(cfg.interpolant, sigma_data)
        self.dit = DiT(cfg.dit, self.interpolant)

        # Autoguidance
        self.use_autoguidance = cfg.autoguidance.enabled
        if self.use_autoguidance:
            self.autoguidance_train_p = 1 / cfg.autoguidance.subsample_train_iter_mult
            self.guiding_model = DiT(OmegaConf.merge(cfg.dit, cfg.autoguidance.dit), self.interpolant)  # override with autoguidance config


    def setup(self):
        if self.use_scaffold_module and self.cfg.scaffold_module.pretrained_weights_path is not None:
            # Load in pretrained fampnn weights
            if not self.cfg.scaffold_module.get("ablate_pretrained_weights", False):
                state_dict = torch.load(self.cfg.scaffold_module.pretrained_weights_path, map_location="cpu")["state_dict"]
                state_dict = repair_state_dict(state_dict)
                state_dict = {k.replace("model.denoiser.seq_design_module.", ""): v for k, v in state_dict.items() if k.startswith("model.denoiser.seq_design_module.")}
                self.fampnn.load_state_dict(state_dict)

            # set to eval mode and freeze weights
            if self.cfg.scaffold_module.get("freeze", True):
                self.fampnn.eval()
                self.fampnn.requires_grad_(False)


    def forward(self,
                motif_inputs: dict[str, TensorType["b n ..."]],
                diffusion_inputs: dict[str, TensorType["b ..."]],
                is_sampling: bool = False,
                diffusion_params: dict[str, Any] | None = None,  # required only for sampling
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        conditioning_inputs = {"h_s": None}
        if self.use_scaffold_module:
            conditioning_inputs["h_s"] = self.embed_motif(motif_inputs)

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            diffusion_inputs=diffusion_inputs,
            conditioning_inputs=conditioning_inputs,
            is_sampling=is_sampling,
            diffusion_params=diffusion_params
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                        #    x_motif: TensorType["b n 37 3", float],
                        #    motif_mask: TensorType["b n 37", float],
                        #    aatype_motif: TensorType["b n", int],
                        #    residue_index: TensorType["b n", int],
                        #    seq_mask: TensorType["b n", float],
                        #    h_s: TensorType["b n h"],
                        #    pair_bias: TensorType["b h n n", float] | None,
                           diffusion_inputs: dict[str, TensorType["b ..."]],
                           conditioning_inputs: dict[str, TensorType["b ..."]],
                           is_sampling: bool,
                           diffusion_params: dict[str, Any] | None,
                           ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred of backbone
                                      Dict[str, TensorType["b ..."]]]:
        B, N = diffusion_inputs["seq_mask"].shape
        diffusion_aux = defaultdict(lambda: None)

        if not is_sampling:
            ### TRAINING ###
            # Get ground truth backbone coordinates
            diffusion_inputs["x_bb"] = diffusion_inputs["x"][..., const.bb_idxs, :]

            # Repeat inputs for batch multiplier  # TODO: randomly augment these too
            M = self.cfg.training_batch_size_mult
            diffusion_inputs_batched = {k: v[None].expand(M, *v.shape) if v is not None else None for k, v in diffusion_inputs.items()}
            diffusion_inputs_batched = {k: v.reshape(M * B, *v.shape[2:]) if v is not None else None for k, v in diffusion_inputs_batched.items()}

            # repeat conditioning inputs
            conditioning_inputs_batched = {k: v[None].expand(M, *v.shape) if v is not None else None for k, v in conditioning_inputs.items()}
            conditioning_inputs_batched = {k: v.reshape(M * B, *v.shape[2:]) if v is not None else None for k, v in conditioning_inputs_batched.items()}

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
                                                                   residue_index=diffusion_inputs_batched["residue_index"])
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched, xt_bb_batched, t_batched))

            denoiser_pred_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                           t_batched,
                                                           seq_mask=diffusion_inputs_batched["seq_mask"],
                                                           residue_index=diffusion_inputs_batched["residue_index"])

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
                                                                     residue_index=diffusion_inputs_batched["residue_index"])
                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched_guide, xt_bb_batched, t_batched))

                denoiser_pred_batched_guide, _ = denoiser_fn(xt_bb_batched,
                                                             t_batched,
                                                             seq_mask=diffusion_inputs_batched["seq_mask"],
                                                             residue_index=diffusion_inputs_batched["residue_index"])

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "bb_pred": denoiser_pred_batched_guide,
                    "bb_target": x_bb_target_batched,  # diffusion target; for edm this is just the ground truth coordinates
                    "loss_weight_t": loss_weight_t_batched,
                    "atom_mask": diffusion_inputs_batched["atom_mask"]
                }

            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = denoiser_pred_batched
            diffusion_aux["bb_target"] = x_bb_target_batched  # diffusion target; for edm this is just the ground truth coordinates
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched
            diffusion_aux["atom_mask"] = diffusion_inputs_batched["atom_mask"]

        else:
            ### SAMPLING ###

            # Sample backbone from prior
            A = len(rc.bb_idxs)
            x0_bb = self.interpolant.sample_prior((B, N, A, 3), seq_mask.device)

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
                                                              x_motif=x_motif,
                                                              motif_mask=motif_mask,
                                                              aatype_motif=aatype_motif,
                                                              h_s=h_s,
                                                              pair_bias=pair_bias,
                                                              residue_index=residue_index, seq_mask=seq_mask,
                                                              cond_labels_in=cond_labels_in)

            # Run integration steps
            denoiser_fn = partial(self.dit,
                                  x_motif=x_motif,
                                  motif_mask=motif_mask,
                                  aatype_motif=aatype_motif,
                                  h_s=h_s,
                                  pair_bias=pair_bias,
                                  residue_index=residue_index, seq_mask=seq_mask,
                                  cond_labels_in=cond_labels_in)

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
                                                               autoguidance_cfg=autoguidance_cfg,
                                                               diffusion_inputs=diffusion_inputs)

                # Save current state
                xt_bb_traj.append(xt_bb.cpu())

                # Save current x1 prediction
                x1_bb_traj.append(aux_preds["x1_pred"].cpu())

            # Finalize outputs
            x1_bb = xt_bb
            diffusion_aux["xt_bb_traj"] = torch.stack(xt_bb_traj, dim=1)  # [B S N A 3]
            diffusion_aux["x1_bb_traj"] = torch.stack(x1_bb_traj, dim=1)  # [B S N A 3]

        return x1_bb, diffusion_aux


    @torch.compiler.disable
    def run_scaffold_module(self, packed_inputs: TensorType["b m ..."]) -> TensorType["b m h", float]:
        """
        Runs the scaffold module on packed motif inputs. Returns embedding of motif residues.
        """
        # Run motif embedding module on packed inputs
        _, mpnn_feature_dict = self.fampnn(
            denoised_coords=packed_inputs["x_motif"],
            aatype_noised=packed_inputs["aatype_motif"],
            seq_mask=packed_inputs["seq_mask"],
            atom_mask_noised=packed_inputs["motif_mask"],
            residue_index=packed_inputs["residue_index"],
            chain_encoding=packed_inputs["chain_encoding"]
        )
        h_V = mpnn_feature_dict["h_V"]  # [b m h]
        return h_V


    def embed_motif(self,
                    x_motif: TensorType["b n 37 3", float],
                    motif_mask: TensorType["b n 37", float],
                    aatype_motif: TensorType["b n", int],
                    seq_mask: TensorType["b n", float],
                    residue_index: TensorType["b n", int]) -> TensorType["b n h", float]:
        """
        Embeds motif residues and returns per-residue embeddings with embeddings for each motif residue, and 0 otherwise.

        First, packs motifs into a compact format of shape [b m], where m is the max number of motif residues in the batch.
        Then, runs the scaffold module on the packed motifs and scatters the result back to the original size.
        """
        B, N, A, _ = x_motif.shape

        # Get size of packed motifs
        motif_residue_mask = motif_mask.any(dim=-1)  # [b n]
        M = motif_residue_mask.sum(dim=-1).max()  # we pad to the max motif length M in this batch
        if M == 0:
            # if no motif residues, return zero embedding
            h_V = torch.zeros((B, N, self.fampnn.hidden_dim), device=x_motif.device)
            return h_V

        ### Pack motif residues into a compact format ###
        # construct motif indices, which are indices of the motif residues for each batch element (zero-padded)
        row_idx = torch.arange(B, device=x_motif.device)[:, None].expand(-1, N)
        col_idx = (motif_residue_mask.cumsum(dim=-1) * motif_residue_mask) - 1  # get where motif residues are, and -1 to get zero-indexed col idxs
        mask_idx = torch.arange(N, device=x_motif.device)[None].expand(B, -1)    # index into motif_residue_mask

        motif_indices = torch.zeros((B, M), device=x_motif.device, dtype=torch.long)
        motif_indices[row_idx[motif_residue_mask], col_idx[motif_residue_mask]] = mask_idx[motif_residue_mask]
        motif_pad_mask = torch.zeros_like(motif_indices).float()  # denotes padding of motif_indices
        motif_pad_mask[row_idx[motif_residue_mask], col_idx[motif_residue_mask]] = 1.0

        # use motif indices to gather into packed inputs
        packed_inputs = {"x_motif": x_motif, "motif_mask": motif_mask, "aatype_motif": aatype_motif, "seq_mask": seq_mask, "residue_index": residue_index}
        for k, v in packed_inputs.items():
            data_shape = v.shape[2:]
            gather_idxs = (motif_indices + (torch.arange(B, device=x_motif.device)[:, None] * N)).view(-1)  # get flat indices of motif residues
            gather_idxs = gather_idxs.view(-1, *((1,) * len(data_shape))).expand(-1, *data_shape)
            packed_v = v.view(-1, *data_shape).gather(0, gather_idxs).view(B, M, *data_shape)
            packed_v = (packed_v * motif_pad_mask.view(B, M, *((1,) * len(data_shape)))).type(v.dtype)
            packed_inputs[k] = packed_v
        packed_inputs["chain_encoding"] = torch.zeros_like(packed_inputs["residue_index"])  # TODO: add chain index to backbone diffusion

        # Run scaffold module on packed inputs and scatter back to original size
        h_V = self.run_scaffold_module(packed_inputs).float()
        h_s = torch.zeros((B, N, h_V.shape[-1]), device=h_V.device)  # [b n h]
        row_idx = torch.arange(B, device=h_V.device)[:, None].expand(-1, M)
        mask = motif_pad_mask.bool()
        h_s[row_idx[mask], motif_indices[mask]] = h_V[mask]

        return h_s


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

        self.scaffolding = cfg.scaffolding  # scaffold conditioning config

        # Input and output channels
        self.c = self.num_atoms_in * 3  # 3 xyz coordinates per atom
        self.in_channels = self.c * 2 if self.use_self_conditioning else self.c  # 2x for self-conditioning

        self.out_channels = self.c
        self.n_aatype = cfg.n_aatype

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
        self.blocks.apply(_basic_init)

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
                t: TensorType["b", float],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
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

        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "absolute_residx":
            x = x + self.pos_embed(x, residue_index=residue_index.float())

        # Conditioning
        t = self.t_embedder(t)
        c = t

        c = c.unsqueeze(1).expand((-1, x.shape[1], -1))  # expand to sequence length
        # if self.scaffolding.use_h_s:
        #     h_s = self.h_s_embedder(h_s)
        #     c = c + h_s

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None, per_token_conditioning=True)

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
