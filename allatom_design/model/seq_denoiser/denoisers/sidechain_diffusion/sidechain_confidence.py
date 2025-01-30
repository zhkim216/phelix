
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import atom37_to_atom14, cat_bb_scn
from allatom_design.model.seq_denoiser.denoisers.seq_design.fampnn import (
    DecLayer, EncLayer, ProteinFeatures, SidechainProteinFeatures,
    cat_neighbors_nodes, gather_nodes)
from openfold.model.primitives import Linear
import torch.nn.functional as F


class SidechainConfidenceModule(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Sidechain confidence module that predicts the confidence of each sidechain atom as Predicted Sidechain Error (PSCE).
        """
        super().__init__()
        self.cfg = cfg

        # Encode input structure
        self.structure_encoder = ConfidenceEncoder(cfg.structure_encoder)

        # Embed aatype into node embeddings
        self.aatype_embedder = Linear(cfg.n_aatype, cfg.c_h_V, bias=False)

        # Embed local sidechain coords into node embeddings
        self.sidechain_encoder = Linear(len(rc.non_bb_idxs) * 3, cfg.c_h_V, bias=False)

        # Final MLP to predict PSCE
        self.sce_bins_cfg = cfg.sce_bins
        self.n_bins = self.sce_bins_cfg.n_bins
        self.mlp = nn.Sequential(
            Linear(cfg.c_h_V, cfg.hidden_size, bias=False, init="relu"),
            nn.SiLU(),
            Linear(cfg.hidden_size, len(rc.non_bb_idxs) * self.n_bins, bias=False, init="final")  # 33 sidechain atoms * n_bins
        )


    def forward(self,
                x1_scn_local_pred: TensorType["b n 33 3", float],  # scn pred output, local coordinates
                mpnn_feature_dict: Dict[str, TensorType["b ..."]],
                aatype: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                residue_index: TensorType["b n", int],
                chain_index: TensorType["b n", int],
                ) -> TensorType["b n 33 n_bins", float]:
        # Structure encoder without using predicted sidechain coordinates
        h_V = self.structure_encoder(mpnn_feature_dict, seq_mask, residue_index, chain_index)

        # Embed aatype into node embeddings
        h_V = h_V + self.aatype_embedder(F.one_hot(aatype, num_classes=self.cfg.n_aatype).float())

        # Embed local sidechain coordinates into node embeddings
        h_V = h_V + self.sidechain_encoder(rearrange(x1_scn_local_pred, "b n a x -> b n (a x)"))

        # MLP on node embeddings for PSCE prediction
        psce_logits = self.mlp(h_V)
        psce_logits = rearrange(psce_logits, "b n (a k) -> b n a k", k=self.n_bins)

        psce_logits = psce_logits * seq_mask[..., None, None]  # zero out padding positions

        psce = self.compute_psce(psce_logits)  # compute per-atom PSCE as expectation

        return psce_logits, psce


    def compute_psce(self, psce_logits: TensorType["b n 33 n_bins", float]) -> TensorType["b n 33", float]:
        """
        Compute per-atom PSCE from logits as an expectation across the binned values.
        """
        # Compute bin centers
        lower = torch.linspace(self.cfg.sce_bins.min_bin,
                               self.cfg.sce_bins.max_bin,
                               self.n_bins,
                               device=psce_logits.device)
        step = lower[1] - lower[0]  # assume bins are equally spaced
        bin_centers = lower + step / 2

        # Compute expectation
        psce_probs = torch.nn.functional.softmax(psce_logits, dim=-1)
        psce = torch.sum(psce_probs * bin_centers, dim=-1)

        return psce


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

        # Structure encoder
        # TODO: remove this, this is not used in the current implementation
        self.sidechain_features = SidechainProteinFeatures(autoregressive=False,
                                                           node_features=self.node_features,
                                                           edge_features=self.edge_features,
                                                           top_k=self.k_neighbors)
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
                mpnn_feature_dict: Dict[str, TensorType["b ..."]],
                seq_mask: TensorType["b n", float],
                residue_index: TensorType["b n", int],
                chain_index: TensorType["b n", int],
                ) -> TensorType["b n h", float]:

        # Initialize with input node and edge embeddings from MPNN
        h_V = mpnn_feature_dict["h_V"]
        h_ESV = mpnn_feature_dict["h_ESV"]
        E_idx = mpnn_feature_dict["E_idx"]

        mask = rearrange(seq_mask, "b n -> b n 1 1").expand_as(h_ESV)
        for layer in self.decoder_layers:
            h_ESV = mask * h_ESV  # mask out padding positions
            h_V, h_ESV = layer(h_V, h_ESV, seq_mask, E_idx)

        return h_V
