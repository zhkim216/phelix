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
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import EDM
from allatom_design.interpolants.ad_interpolants.sd3_rf_interpolant import \
    SD3_RF
from allatom_design.model.atom_denoiser.denoisers.denoiser import \
    BaseAtomDenoiser
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


class DiTDenoiser(BaseAtomDenoiser):
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

        # AF2-based positional encoding
        self.pos_encoding = cfg.get("pos_encoding", None)
        if self.pos_encoding == "af2":
            self.pos_embed = RelativePositionalEncoding(cfg.rel_pos_encoding)
            self.pair_bias_embedder = nn.Sequential(nn.LayerNorm(cfg.rel_pos_encoding.c_z),
                                                    Linear(cfg.rel_pos_encoding.c_z, cfg.dit.num_heads, init="normal", bias=False))

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
            self.fampnn.load_state_dict(torch.load(self.cfg.scaffold_module.pretrained_weights_path))


    def forward(self,
                x_motif: TensorType["b n 37 3", float],
                motif_mask: TensorType["b n 37", float],
                aatype_motif: TensorType["b n", int],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        if self.use_scaffold_module:
            h_s = self.process_motif(x_motif, motif_mask, aatype_motif, seq_mask, residue_index)
        else:
            h_s = None

        if self.pos_encoding == "af2":
            z = self.pos_embed(residue_index)
            pair_bias = rearrange(self.pair_bias_embedder(z), "b i j h -> b h i j")
        else:
            pair_bias = None

        x1_pred, bb_diffusion_aux = self.backbone_diffusion(
            x_motif=x_motif,
            motif_mask=motif_mask,
            aatype_motif=aatype_motif,
            residue_index=residue_index,
            seq_mask=seq_mask,
            h_s=h_s,
            pair_bias=pair_bias,
            cond_labels_in=cond_labels_in,
            aux_inputs=aux_inputs,
            is_sampling=is_sampling
        )

        aux_preds["bb_diffusion_aux"] = bb_diffusion_aux

        return x1_pred, aux_preds


    def backbone_diffusion(self,
                           x_motif: TensorType["b n 37 3", float],
                           motif_mask: TensorType["b n 37", float],
                           aatype_motif: TensorType["b n", int],
                           residue_index: TensorType["b n", int],
                           seq_mask: TensorType["b n", float],
                           h_s: TensorType["b n h"],
                           pair_bias: TensorType["b h n n", float] | None,
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
            residue_index_batched = repeat(residue_index, "b n -> (m b) n", m=M, b=B)
            seq_mask_batched = repeat(seq_mask, "b n -> (m b) n", m=M, b=B)
            cond_labels_in_batched = {label: repeat(cond_labels_in[label], "b -> (m b)", m=M, b=B) for label in cond_labels_in}

            # repeat scaffolding inputs
            x_scaffold_batched = repeat(x_motif, "b n a x -> (m b) n a x", m=M, b=B)
            scaffold_mask_batched = repeat(motif_mask, "b n a -> (m b) n a", m=M, b=B)
            aatype_scaffold_batched = repeat(aatype_motif, "b n -> (m b) n", m=M, b=B)
            h_s_batched = repeat(h_s, "b n h -> (m b) n h", m=M, b=B) if h_s is not None else None
            pair_bias_batched = repeat(pair_bias, "b h i j -> (m b) h i j", m=M, b=B) if pair_bias is not None else None  # TODO: expand instead to save memory?

            # Evaluate at specific timesteps (for validation)
            t_bb = None
            if aux_inputs["t_bb"] is not None:
                t_bb = torch.full((M * B, ), aux_inputs["t_bb"], device=x_bb_gt.device)

            # Noise the ground truth backbone
            interpolant_out = self.interpolant({"x": x_bb_gt_batched, "aatype": None}, t=t_bb)
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
                                                           x_scaffold_batched,
                                                           scaffold_mask_batched,
                                                           aatype_scaffold_batched,
                                                           h_s_batched,
                                                           pair_bias_batched,
                                                           t_batched,
                                                           seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                           cond_labels_in=cond_labels_in_batched)
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched, xt_bb_batched, t_batched))

            denoiser_pred_batched, aux_preds = denoiser_fn(xt_bb_batched,
                                                   x_scaffold_batched,
                                                   scaffold_mask_batched,
                                                   aatype_scaffold_batched,
                                                   h_s_batched,
                                                   pair_bias_batched,
                                                   t_batched,
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
                        denoiser_pred_batched_guide, _ = denoiser_fn(xt_bb_batched,
                                                             x_scaffold_batched,
                                                             scaffold_mask_batched,
                                                             aatype_scaffold_batched,
                                                             h_s_batched,
                                                             pair_bias_batched,
                                                             t_batched,
                                                             seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                             cond_labels_in=cond_labels_in_batched)
                    torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                    denoiser_fn = partial(denoiser_fn, x_self_cond=self.interpolant.get_x1_pred(denoiser_pred_batched_guide, xt_bb_batched, t_batched))

                denoiser_pred_batched_guide, _ = denoiser_fn(xt_bb_batched,
                                                     x_scaffold_batched,
                                                     scaffold_mask_batched,
                                                     aatype_scaffold_batched,
                                                     h_s_batched,
                                                     pair_bias_batched,
                                                     t_batched,
                                                     seq_mask=seq_mask_batched, residue_index=residue_index_batched,
                                                     cond_labels_in=cond_labels_in_batched)

                # add to autoguidance outputs
                diffusion_aux["autoguidance_aux"] = {
                    "bb_pred": denoiser_pred_batched_guide,
                    "bb_target": x_bb_target_batched,  # diffusion target; for edm this is just the ground truth coordinates
                    "loss_weight_t": loss_weight_t_batched
                }

            # Outputs
            x1_bb = None  # during training, we return the batched version in diffusion_aux

            # Cache intermediates for computing loss
            diffusion_aux["bb_pred"] = denoiser_pred_batched
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
            diffusion_inputs = aux_inputs["diffusion_inputs"]
            S = diffusion_inputs["num_steps"]
            timesteps = diffusion_inputs["timesteps"]
            noise_schedule = diffusion_inputs["noise_schedule"]
            churn_cfg = diffusion_inputs["churn_cfg"]
            autoguidance_cfg = diffusion_inputs["autoguidance_cfg"]

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
                                                               aux_inputs=aux_inputs)

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
    def process_motif(self, x_motif, motif_mask,
                         aatype_motif, seq_mask, residue_index):
        B, N, A, _ = x_motif.shape
        # Find residues where motif_mask is nonzero in any atom
        has_scaffold = motif_mask.any(dim=(-1))  # [b n]

        # Get indices of residues with scaffold information
        scaffold_indices = [torch.where(has_scaffold[bi])[0] for bi in range(B)]
        max_scaffolds = max(len(indices) for indices in scaffold_indices)

        if max_scaffolds > 0:  # Only process if we have any scaffold positions
            # Pack inputs to FAMPNN
            packed_x = []
            packed_scaffold_mask = []  # renamed from packed_mask for clarity
            packed_seq_mask = []
            packed_aatype = []
            packed_residue_index = []

            packed_bis = []  # keep track of batch indices for packed inputs

            for bi, indices in enumerate(scaffold_indices):
                if len(indices) > 0:
                    packed_bis.append(bi)

                    packed_x.append(x_motif[bi, indices])
                    packed_scaffold_mask.append(motif_mask[bi, indices])
                    packed_seq_mask.append(seq_mask[bi, indices])
                    packed_aatype.append(aatype_motif[bi, indices])
                    packed_residue_index.append(residue_index[bi, indices])

            # Pad to max length
            packed_x = torch.nn.utils.rnn.pad_sequence(packed_x, batch_first=True)  # [b' m 37 3]
            packed_scaffold_mask = torch.nn.utils.rnn.pad_sequence(packed_scaffold_mask, batch_first=True)  # [b' m 37]
            packed_seq_mask = torch.nn.utils.rnn.pad_sequence(packed_seq_mask, batch_first=True)  # [b' m]
            packed_aatype = torch.nn.utils.rnn.pad_sequence(packed_aatype, batch_first=True)  # [b' m]
            packed_residue_index = torch.nn.utils.rnn.pad_sequence(packed_residue_index, batch_first=True)  # [b' m]

            # Create chain encoding (all zeros for now)
            packed_chain_encoding = torch.zeros_like(packed_residue_index)

            # Run scaffold module on packed inputs
            _, mpnn_feature_dict = self.fampnn(
                denoised_coords=packed_x,
                aatype_noised=packed_aatype,
                seq_mask=packed_seq_mask,
                atom_mask_noised=packed_scaffold_mask,
                residue_index=packed_residue_index,
                chain_encoding=packed_chain_encoding
            )

            # Unpack outputs back to original size
            hV = mpnn_feature_dict["h_V"]  # [b' m h]
            h_s = torch.zeros((B, N, hV.shape[-1]), device=hV.device)  # [b n h]

            # Scatter back to original size. j is the row in hV, while bi is the original batch index.
            for j, bi in enumerate(packed_bis):
                idx = scaffold_indices[bi]
                h_s[bi, idx] = hV[j, :len(idx)]

        else:
            # If no scaffold positions, set h_s to zero and skip FAMPNN
            h_s = torch.zeros((B, N, self.fampnn.hidden_dim), device=x_motif.device)

        return h_s


class DiT(nn.Module):
    def __init__(self, cfg: DictConfig, interpolant: ADInterpolant):
        """
        DiT for unconditional backbone diffusion
        """
        super().__init__()

        self.cfg = cfg
        self.interpolant = interpolant

        # Set up pair representation builder
        self.use_pair_repr = cfg.get("pair_rep", {}).get("enabled", False)
        if self.use_pair_repr:
            self.pair_rep_builder = PairRepBuilder(cfg.pair_rep)

        # Set up DiT model
        self.num_atoms_in = cfg.num_atoms_in
        self.use_self_conditioning = cfg.use_self_conditioning

        self.scaffolding = cfg.scaffolding  # scaffold conditioning config

        # Input and output channels
        self.c = self.num_atoms_in * 3  # 3 xyz coordinates per atom
        self.in_channels = self.c * 2 if self.use_self_conditioning else self.c  # 2x for self-conditioning

        if self.scaffolding.use_concat:
            # +condition on motif + one-hot sequence conditioning
            self.in_channels = self.in_channels + rc.atom_type_num * 3
            self.in_channels = self.in_channels + cfg.n_aatype

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

        # Conditioning
        self.cond_label_to_dropout_p = getattr(cfg, "cond_label_to_dropout_p", {})
        self.cond_labels = [k for k, v in self.cond_label_to_dropout_p.items() if v is not None]
        self.cond_embedders = nn.ModuleDict({
            label: LabelEmbedder(num_classes=cl.COND_NUM_CLASSES[label],
                                 hidden_size=cfg.hidden_size,
                                 dropout_prob=self.cond_label_to_dropout_p[label]) for label in self.cond_labels
        })

        # Scaffold conditioning via per-residue embeddings
        if self.scaffolding.use_h_s:
            self.h_s_embedder = Linear(self.scaffolding.h_s, cfg.hidden_size)

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
        if self.use_pair_repr:
            self.to_pair_biases = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(cfg.pair_rep.c_z),
                              Linear(cfg.pair_rep.c_z, cfg.num_heads, init="normal", bias=False))  # openfold initialization
                for _ in range(cfg.depth)
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
                x_motif: TensorType["b n 37 3", float],
                motif_mask: TensorType["b n 37", float],
                aatype_motif: Optional[TensorType["b n", int]],
                h_s: Optional[TensorType["b n h"]],
                pair_bias: Optional[TensorType["b h n n", float]],
                t: TensorType["b", float],
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                x_self_cond: Optional[TensorType["b n 4 3", float]] = None,
                cond_labels_in: Dict[str, TensorType["b", int]] = {}
                ) -> Tuple[TensorType["b n 4 3", float],  # x1 pred of backbone
                           Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        if self.use_pair_repr:
            # Keep original x_noised and x_self_cond before preconditioning
            x_noised_orig = x_noised
            x_self_cond_orig = x_self_cond

        # Preconditioning
        precondition_in, precondition_out = self.interpolant.setup_preconditioning(x_noised, x_self_cond, t)
        x_noised, x_self_cond, t = precondition_in()  # input preconditioning

        # Construct pair representation
        if self.use_pair_repr:
            # Get pair representation
            z = self.pair_rep_builder(x_noised_orig[..., rc.atom_order["CA"], :],
                                      x_self_cond_orig[..., rc.atom_order["CA"], :] if x_self_cond_orig is not None else None,
                                      residue_index, seq_mask, t)

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x_noised)
            x_noised = torch.cat([x_noised, x_self_cond], dim=-1)

        x = rearrange(x_noised, "b n a x -> b n (a x)")

        # Concatenate scaffold conditioning
        if self.scaffolding.use_concat:
            x_motif = x_motif * motif_mask[..., None]  # zero out non-motif positions
            x_motif = rearrange(x_motif, "b n a x -> b n (a x)")
            x = torch.cat([x, x_motif], dim=-1)

            # also concatenate one-hot sequence conditioning
            aatype_oh = F.one_hot(aatype_motif, num_classes=self.n_aatype).float()
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

        # Scaffold conditioning
        c = c.unsqueeze(1).expand((-1, x.shape[1], -1))  # expand to sequence length
        if self.scaffolding.use_h_s:
            h_s = self.h_s_embedder(h_s)
            c = c + h_s

        # Blocks
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for bi, block in enumerate(self.blocks):
            attn_bias = pair_bias
            if self.use_pair_repr:
                to_pair_bias_fn = self.to_pair_biases[bi]
                attn_bias = rearrange(to_pair_bias_fn(z), "b i j h -> b h i j")
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=attn_bias, per_token_conditioning=True)

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
