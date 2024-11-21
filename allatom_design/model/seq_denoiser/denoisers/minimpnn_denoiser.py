from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope
from allatom_design.data.data import cat_bb_scn
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import \
    FinalLayer
from allatom_design.model.atom_denoiser.denoisers.dit_utils import (
    DiffusionMLPBlock, DiTBlock, MultiHeadRMSNorm)
from allatom_design.model.atom_denoiser.denoisers.pos_embed.sin_cos import \
    posemb_sincos_1d
from allatom_design.model.atom_denoiser.denoisers.timestep_embedders import \
    TimestepEmbedder
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.fampnn import \
    FaMPNN
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.scn_diffusion_dit import \
    SidechainDiffusionModule


class MiniMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task
        self.use_scn_diffusion = self.task in ["allatom_seq_des", 'scn_pack']

        # Sequence design model: MiniMPNN
        self.seq_design_module = FaMPNN(cfg.minimpnn)

        # Sidechain diffusion head: DiT
        if self.use_scn_diffusion:
            # Backbone encoder: DiT
            # self.backbone_encoder = BackboneEncoderDiT(cfg.backbone_encoder)
            self.scn_diffusion_module = SidechainDiffusionModule(cfg.scn_diffusion_module, self.scn_sigma_data)


    def forward(self,
                x_noised: TensorType["b n a 3", float],
                aatype_noised: TensorType["b n", int],
                t: TensorType["b", float],  # possibly a tuple (t_seq, t_scn)
                residue_index: TensorType["b n", int],
                chain_encoding: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                seq_self_cond: Optional[TensorType["b n k", float]] = None,  # k = n_aatype, logits
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n a 3", float],  # x1 pred
                           TensorType["b n", int],  # aatype pred
                           Dict[str, TensorType["b ..."]]  # aux_preds
                           ]:

        # 1. Sequence design
        if self.task in ['scn_pack']:
            seq_mlm_mask = torch.full_like(residue_index, 1.0)
        elif self.task in ['allatom_seq_des', 'seq_des']:
            seq_mlm_mask = aux_inputs['seq_mlm_mask']
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        seq_logits, node_embs, edge_embs, x_bb = self.seq_design_module(
            x_noised,
            aatype_noised,
            None, #no seq self cond
            seq_mask,
            residue_index,
            chain_encoding,
            seq_mlm_mask)

        aatype_pred, seq_probs = self.sample_aatype(seq_logits, aux_inputs, is_sampling)

        # If sampling, update mlm mask for sidechain diffusion
        if is_sampling:
            seq_mlm_mask = aux_inputs["mask_update_fn"](seq_mlm_mask, seq_probs=seq_probs)
            aux_inputs["seq_mlm_mask"] = seq_mlm_mask

        # Outputs
        aux_preds = {
            "seq_logits": seq_logits,
            "seq_probs": seq_probs,
            'seq_mask': seq_mask,
            'seq_mlm_mask': aux_inputs["seq_mlm_mask"],
            'scn_mlm_mask': aux_inputs.get('scn_mlm_mask', None)
        }

        # 2. Sidechain diffusion
        x1_pred = None
        if self.use_scn_diffusion:
            # node_embs = self.backbone_encoder(node_embs, x_bb,
            #                                   seq_mask, residue_index, chain_encoding)
            x1_scn_pred, scn_diffusion_aux = self.scn_diffusion_module.sidechain_diffusion(
                node_embs,
                edge_embs,
                aatype_pred,
                x_bb,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_encoding,
                aux_inputs=aux_inputs,
                is_sampling=is_sampling
            )

            aux_preds['scn_diffusion_aux'] = scn_diffusion_aux

            if is_sampling:
                # store the predicted sidechain coordinates with known backbone
                x1_pred = cat_bb_scn(x_bb, x1_scn_pred)


        return x1_pred, aatype_pred, aux_preds


    def sample_aatype(self,
                      seq_logits: TensorType["b n k", float],
                      aux_inputs: Dict[str, Any],
                      is_sampling: bool,
                      ) -> Tuple[TensorType["b n", int], TensorType["b n k", float]]:
        """
        Sample aatype from seq logits
        If training, just take argmax (this will be teacher-forced to the ground truth aatype during sidechain diffusion)
        If sampling, sample from (possibly temperature-scaled) logits

        Returns:
        - aatype_pred: Tensor["b n", int]
        - seq_probs: Tensor["b n k", float]
        """
        if not is_sampling:
            return seq_logits.argmax(dim=-1), F.softmax(seq_logits, dim=-1)

        tau = aux_inputs.get("temperature", 1.0)
        B, N = seq_logits.shape[:2]
        if tau == 0.0:
            aatype_pred = seq_logits.argmax(dim=-1)
            seq_probs = F.softmax(seq_logits, dim=-1)  # don't scale for confidence sampling
        else:
            scaled_logits = seq_logits / tau
            scaled_logits[..., rc.restype_order_with_x["X"]] = -1e9  # do not sample mask/unknowns
            seq_probs = F.softmax(scaled_logits, dim=-1)
            aatype_pred = torch.multinomial(seq_probs.view(B * N, -1), num_samples=1).view(B, N)
        return aatype_pred, seq_probs


    def get_sidechain_likelihoods(self,
                                  num_steps: int,
                                  x: TensorType["b n a 3", float],
                                  aatype: TensorType["b n", int],
                                  residue_index: TensorType["b n", int],
                                  chain_index: TensorType["b n", int],
                                  seq_mask: TensorType["b n", float],
                                  cond_labels_in: Dict[str, TensorType["b", int]] = {},
                                  aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                                  ):
        # 1. MiniMPNN on ground truth aatype for sequence embeddings
        _, h_V, h_ESV, _ = self.seq_design_module(x, aatype, None, seq_mask, residue_index, chain_index, aux_inputs["seq_mlm_mask"])

        # 2. Get sidechain likelihoods
        x1_scn, x_bb = x[..., rc.non_bb_idxs, :], x[..., rc.bb_idxs, :]
        likelihood_aux = self.scn_diffusion_module.get_likelihoods(num_steps, x1_scn, h_V, aatype, x_bb, seq_mask, residue_index, chain_index, aux_inputs)

        return likelihood_aux


class BackboneEncoderDiT(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        DiT to encode backbone coordinates.
        """
        super().__init__()

        self.cfg = cfg

        # Set up DiT-based backbone encoder
        self.in_channels = len(rc.bb_idxs) * 3
        self.out_channels = cfg.hidden_size

        # QK-normalization from SD3
        self.qk_normlayer = None
        if cfg.qk_rmsnorm:
            self.qk_normlayer = partial(MultiHeadRMSNorm, heads=cfg.num_heads)

        # DiT positional encoding
        self.pos_encoding = cfg.pos_encoding
        self.rotary_emb = None
        assert self.pos_encoding in ["rotary", "rotary_residx"]
        dim = cfg.hidden_size // cfg.num_heads
        use_residx = (self.pos_encoding == "rotary_residx")
        self.rotary_emb = rope.RotaryEmbedding(dim=dim, use_residx=use_residx, cache_if_possible=False)

        # DiT blocks
        self.bb_embedder = nn.Linear(self.in_channels, self.out_channels)
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_size, cfg.num_heads,
                     mlp_dropout=cfg.mlp_dropout, mlp_ratio=cfg.mlp_ratio,
                     inf=cfg.inf,
                     rotary_emb=self.rotary_emb,
                     qk_norm=cfg.qk_rmsnorm, norm_layer=self.qk_normlayer,
                     ) for _ in range(cfg.depth)
        ])

        # node embedding conditioning
        self.h_V_embedder = nn.Linear(cfg.c_h_V, cfg.hidden_size)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


    def forward(self,
                h_V: TensorType["b n h_mpnn", float],  # conditioning latent
                x_bb: TensorType["b n a_bb 3", float],  # backbone atoms
                seq_mask: TensorType["b n", float],
                residue_index: TensorType["b n", float],
                chain_index: TensorType["b n", float],
                ) -> Tuple[TensorType["b n h", float]]:
        """
        TODO: use chain index?
        """
        x_bb = rearrange(x_bb, "b n a x -> b n (a x)")
        x = self.bb_embedder(x_bb)

        # Use DiT to encode backbone coordinates within node embeddings
        c = self.h_V_embedder(h_V)
        attn_mask = repeat(seq_mask[:, :, None] * seq_mask[:, None, :], "b i j -> b h i j", h=self.cfg.num_heads)
        for block in self.blocks:
            x = block(x, c, residx=residue_index.float(), attn_mask=attn_mask, attn_bias=None, per_token_conditioning=True)

        c = x + c
        return c
