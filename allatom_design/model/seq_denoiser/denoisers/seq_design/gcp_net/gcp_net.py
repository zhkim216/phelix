# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from typeguard import typechecked
from omegaconf import DictConfig
from allatom_design.data import residue_constants as rc
from allatom_design.model.seq_denoiser.denoisers.seq_design.gcp_net.gcp_modules import GCPEmbedding, ScalarVector, GCP2, GCPInteractions, GCPInputFeaturizer
from allatom_design.model.seq_denoiser.denoisers.seq_design.gcp_net.gcp_utils import centralize, localize

class GCPNet(nn.Module):
    """LightningModule for computational protein design (CPD) using GCPNet.

    This LightningModule organizes the PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR schedulers (configure_optimizers)
        - End of model training (on_fit_end)
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        # feature dimensionalities
        self.node_dims = ScalarVector(cfg.h_hidden_dim, cfg.chi_hidden_dim)
        self.edge_dims = ScalarVector(cfg.e_hidden_dim, cfg.xi_hidden_dim)
        self.input_featurizer = GCPInputFeaturizer(num_positional_embeddings=cfg.num_positional_embeddings)
        self.norm_x_diff = cfg.module.norm_x_diff
        # PyTorch modules #

        # input embeddings
        self.gcp_embedding = GCPEmbedding(
            cfg.edge_input_dims,
            cfg.node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=rc.atom_type_num,
            cfg=cfg.module,
            pre_norm=False
        )

        # message-passing encoder layers
        self.encoder_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=cfg.module,
                layer_cfg=cfg.layer,
                dropout=cfg.dropout,
            ) for _ in range(cfg.num_encoder_layers)
        )

        # GCP to coalesce scalar and vector-valued node features into scalar node features
        invariant_node_projection_dim = self.node_dims[0]
        self.invariant_node_projection = GCP2(
            self.node_dims,
            (invariant_node_projection_dim, 0),
            cfg.module
        )

        self.embed_mpnn_node = nn.Linear(cfg.h_hidden_dim, cfg.h_hidden_dim)
        self.embed_atom_type = nn.Linear(rc.atom_type_num, cfg.h_hidden_dim)
        self.embed_mpnn_edge = nn.Linear(cfg.e_hidden_dim, cfg.e_hidden_dim)

    @typechecked
    def forward(self, coords, seq, mpnn_E_idx, mpnn_node_features, mpnn_edge_features, padding_mask, atom14_mask):

        batch = self.input_featurizer(coords, seq, mpnn_E_idx, padding_mask, atom14_mask)

        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask
        )

        # craft complete local frames corresponding to each edge
        batch.f_ij = localize(
            batch.x,
            batch.edge_index,
            norm_x_diff=self.norm_x_diff,
            node_mask=batch.mask
        )

        # embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # sum scalar node features with embedded mpnn node edge features
        h += self.embed_mpnn_node(mpnn_node_features.flatten(0,1))

        # sum scalar node features with atom embedding
        h += self.embed_atom_type(batch.atom_type)

        # sum scalar edge features with embedded mpnn edge features
        e += self.embed_mpnn_edge(mpnn_edge_features.flatten(0,2))

        # encode graph features using a series of geometric message-passing layers
        for layer in self.encoder_layers:
            (h, chi) = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_mask=batch.mask
            )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # summarize intermediate node representations as final predictions
        out = self.invariant_node_projection(
            (batch.h, batch.chi),
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=batch.mask
        )  # e.g., GCP((h, chi)) -> h'


        return out
