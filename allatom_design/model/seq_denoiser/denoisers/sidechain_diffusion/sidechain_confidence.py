
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import atom37_to_atom14, cat_bb_scn
from allatom_design.model.seq_denoiser.denoisers.seq_design.fampnn import (
    DecLayer, EncLayer, ProteinFeatures, SidechainProteinFeatures, gather_nodes, cat_neighbors_nodes)
from openfold.model.primitives import Linear
import torch


class SidechainConfidenceModule(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Sidechain confidence module that predicts the confidence of each sidechain atom as Predicted Sidechain Error (PSCE).
        """
        super().__init__()
        self.cfg = cfg
        self.structure_encoder = ConfidenceEncoder(cfg.structure_encoder)

        # Final MLP to predict PSCE
        self.sce_bins_cfg = cfg.sce_bins
        self.n_bins = self.sce_bins_cfg.n_bins
        self.mlp = nn.Sequential(
            Linear(cfg.c_h_V, cfg.hidden_size, bias=False, init="relu"),
            nn.SiLU(),
            Linear(cfg.hidden_size, len(rc.non_bb_idxs) * self.n_bins, bias=False, init="final")  # 33 sidechain atoms * n_bins
        )


    def forward(self,
                x1_scn_pred: TensorType["b n 33 3", float],  # scn pred output, absolute coordinates
                h_V: TensorType["b n h", float],
                h_ESV: TensorType["b n k h", float],
                aatype: TensorType["b n", int],
                x_bb: TensorType["b n 4 3", float],  # already noised
                seq_mask: TensorType["b n", float],
                residue_index: TensorType["b n", int],
                chain_index: TensorType["b n", int],
                scd_mlm_mask: TensorType["b n", float],
                ) -> TensorType["b n 33 n_bins", float]:
        X = cat_bb_scn(x_bb, x1_scn_pred)

        # Structure encoder
        h_V = self.structure_encoder(h_V, h_ESV, X, aatype, seq_mask, residue_index, chain_index)

        # MLP on node embeddings for PSCE prediction
        psce_logits = self.mlp(h_V)
        psce_logits = rearrange(psce_logits, "b n (a k) -> b n a k", k=self.n_bins)

        psce_logits = psce_logits * seq_mask[..., None, None]  # zero out padding positions
        return psce_logits


    # def compute_psce(self, psce_logits: TensorType["b n 33 n_bins", float]):


class ConfidenceEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.n_aatype = cfg.n_aatype
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.decoder_in = self.hidden_dim * 4
        self.use_ESV_in = cfg.use_ESV_in

        # Structure encoder
        self.sidechain_features = SidechainProteinFeatures(autoregressive=False,
                                                           node_features=self.node_features,
                                                           edge_features=self.edge_features,
                                                           top_k=self.k_neighbors,
                                                           augment_eps=0.0)
        self.features = ProteinFeatures(self.node_features, self.edge_features, top_k=self.k_neighbors, augment_eps=0.0)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_s = nn.Embedding(self.n_aatype, self.hidden_dim)
        self.W_e2 = nn.Linear(self.edge_features, self.hidden_dim, bias=True)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(self.hidden_dim, self.decoder_in, dropout=cfg.dropout_p)
            for _ in range(self.num_decoder_layers)
        ])

        # Initialize MPNN weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self,
                h_V_in: TensorType["b n h", float],
                h_ESV_in: TensorType["b n k 4h", float],
                X: TensorType["b n 37 3", float],
                S: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                residue_index: TensorType["b n", int],
                chain_index: TensorType["b n", int],
                ) -> TensorType["b n h", float]:
        X, atom14_mask = atom37_to_atom14(S, X)

        # Extract edge embeddings from rollout
        E, E_idx, X = self.features(X, seq_mask, residue_index, chain_index)  # TODO: make sure E_idx is the same
        h_E = self.W_e(E)
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # extract sidechain features
        E2, _ = self.sidechain_features(X, residue_index, chain_index, E_idx, atom14_mask)
        h_E2 = self.W_e2(E2)
        h_ES = torch.cat([h_ES, h_E2], dim = -1)

        # Make input node and edge embeddings
        h_V = h_V_in
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        if self.use_ESV_in:
            h_ESV = h_ESV + h_ESV_in

        mask = rearrange(seq_mask, "b n -> b n 1 1").expand_as(h_ESV)
        for layer in self.decoder_layers:
            h_ESV = mask * h_ESV  # mask out padding positions
            h_V, h_ESV = layer(h_V, h_ESV, seq_mask, E_idx)

        return h_V
