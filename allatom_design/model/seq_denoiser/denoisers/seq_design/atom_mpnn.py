import math
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from boltz.model.modules.utils import LinearNoBias
from omegaconf import DictConfig
from torch.nn import functional as F
from torch_cluster import knn_graph
from torchtyping import TensorType

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data.data import batched_gather
from allatom_design.data.datasets.boltz_sd_dataset import \
    crop_batch_to_protein_only
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes, gather_edges, gather_nodes)
from chroma.layers.structure import diffusion

# https://github.com/pyg-team/pytorch_geometric/issues/8747
knn_graph = torch.compiler.disable(knn_graph)
from chroma.data.protein import Protein


class AtomMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors

        self.token_features = TokenFeatures(cfg.token_features)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=False)
        self.W_s = nn.Linear(len(const.tokens), self.hidden_dim, bias=False)
        self.decoder_in = self.hidden_dim * 3  # concat of h_E, h_S, h_V

        self.dropout = nn.Dropout(cfg.dropout_p)

        # Polymer backbone diffusion
        self.use_polymer_diffusion = cfg.get("use_polymer_diffusion", False)
        if self.use_polymer_diffusion:
            self.noise_perturb = diffusion.DiffusionChainCov(
                noise_schedule=cfg.polymer_diffusion.noise_schedule,
                beta_min=cfg.polymer_diffusion.noise_beta_range[0],
                beta_max=cfg.polymer_diffusion.noise_beta_range[1],
                log_snr_range=cfg.polymer_diffusion.noise_log_snr_range,
                covariance_model=cfg.polymer_diffusion.noise_covariance_model,
                complex_scaling=cfg.polymer_diffusion.noise_complex_scaling,
            )
            self.time_features = diffusion.NoiseTimeEmbedding(
                dim_embedding=self.node_features,
                noise_schedule=self.noise_perturb.noise_schedule,
            )
            self.t_max = cfg.polymer_diffusion.get("t_max", 1.0)
            self.canonicalize = cfg.polymer_diffusion.get("canonicalize", False)

        # Atom-level encoder
        if cfg.use_atom_encoder:
            self.atom_encoder = AtomGraphEncoder(cfg.atom_encoder)
        else:
            self.atom_encoder = None

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(self.hidden_dim, self.hidden_dim*2, dropout=cfg.dropout_p)
            for _ in range(self.num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(self.hidden_dim, self.decoder_in, dropout=cfg.dropout_p)
            for _ in range(self.num_decoder_layers)
        ])

        # Potts decoder
        self.use_potts = cfg.potts.use_potts
        self.use_msa_potts = cfg.potts.get("use_msa_potts", False)
        if self.use_potts:
            self.k_neighbors_potts = cfg.potts.get("k_neighbors_potts", None)
            self.max_dist_potts = cfg.potts.get("max_dist_potts", None)
            self.parameterization = cfg.potts.parameterization
            self.num_factors = cfg.potts.num_factors

            potts_init = partial(potts.GraphPotts,
                dim_nodes=self.node_features,
                dim_edges=self.decoder_in,
                num_states=len(const.tokens),
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )
            self.decoder_S_potts = potts_init()

            if self.use_msa_potts:
                self.msa_potts = potts_init()

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, len(const.tokens), bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, batch: dict[str, TensorType["b ..."]], is_sampling: bool):
        # If provided, add noise to input coordinates
        batch["coords"] = self._add_noise(batch)

        # Apply polymer backbone diffusion
        if self.use_polymer_diffusion:
            batch = self.polymer_noise_perturb(batch, is_sampling)

        # Get token-level features
        if self.atom_encoder is not None:
            h_V = self.atom_encoder(batch)
        else:
            B, N, C = batch["res_type"].shape
            h_V = torch.zeros((B, N, self.node_features), device=batch["res_type"].device)

        # Embed time step
        if self.use_polymer_diffusion:
            h_V = h_V + self.time_features(batch["t"])

        # Concatenate residue-level features to h_V
        ## first, mask out residues using gap token
        B, N, C = batch["res_type"].shape
        masked = F.one_hot(torch.full((B, N), const.token_ids["-"], device=batch["res_type"].device), num_classes=C).float()
        res_type = torch.where(batch["seq_cond_mask"].unsqueeze(-1).bool(), batch["res_type"], masked)
        h_S = self.W_s(res_type)

        # Build graph and get edge features
        h_E, E_idx, D_neighbors = self.token_features(batch)

        # Pass through encoder layers
        h_V = h_V + h_S
        h_E = self.W_e(h_E)
        token_mask = batch["token_exists_mask"]
        token_mask_2d = gather_nodes(token_mask.unsqueeze(-1), E_idx).squeeze(-1)
        token_mask_2d = token_mask.unsqueeze(-1) * token_mask_2d
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, token_mask, token_mask_2d)

        # Pass through decoder layers
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        for layer in self.decoder_layers:
            h_V, h_ESV = layer(h_V, h_ESV, token_mask, E_idx)

        # Potts model
        if self.use_potts:
            if self.max_dist_potts is not None:
                token_mask_2d = token_mask_2d * (D_neighbors <= self.max_dist_potts)  # mask out edges that are too far away

            if self.k_neighbors_potts is not None:
                # truncate to k_neighbors_potts
                h_ESV = h_ESV[:, :, :self.k_neighbors_potts]
                E_idx = E_idx[:, :, :self.k_neighbors_potts]
                token_mask_2d = token_mask_2d[:, :, :self.k_neighbors_potts]

            h, J = self.decoder_S_potts(h_V, h_ESV, E_idx, token_mask, token_mask_2d)
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": token_mask,
                "mask_ij": token_mask_2d,
            }

            if self.use_msa_potts:
                h_msa, J_msa = self.msa_potts(h_V, h_ESV, E_idx, token_mask, token_mask_2d)
                potts_decoder_aux["h_msa"] = h_msa
                potts_decoder_aux["J_msa"] = J_msa

        logits = self.W_out(h_V)

        # Output features
        mpnn_feature_dict = {"h_V": h_V, "h_ESV": h_ESV, "E_idx": E_idx}
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux

        return logits, mpnn_feature_dict


    def _add_noise(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n_atoms 3", float]:
        """
        If provided, add noise to input coordinates
        """
        if batch["noise"] is None:
            return batch["coords"]

        # TODO: implement support for noise labels / per-residue noise
        if batch["noise_labels"] is not None:
            raise NotImplementedError("Per-residue noise not yet implemented for AtomMPNN")

        # Add noise to input coordinates
        noised_coords = batch["coords"] + batch["noise"]
        return noised_coords


    @torch.compiler.disable
    def polymer_noise_perturb(self, batch: dict[str, TensorType["b ..."]], is_sampling: bool) -> dict[str, TensorType["b ..."]]:
        """
        Apply polymer backbone diffusion noise to the input coordinates.
        """
        if not is_sampling and self.training:
            # Sample timestep
            t = self.noise_perturb.sample_t(batch["token_pad_mask"])
            t = t * self.t_max
        else:
            # Use provided timestep or zero if not provided
            t = batch.get("t", batch["token_pad_mask"].new_zeros(batch["token_pad_mask"].shape[0]))

        # Get protein coordinates and apply noise to the backbone atoms
        protein_batch, protein_token_mask = crop_batch_to_protein_only(batch, return_crop_mask=True)
        X_prot = get_tokenwise_coords(protein_batch)
        X_prot_bb = X_prot[..., const.prot_bb_atom14_idxs, :]
        if self.canonicalize:
            X = X_prot_bb
            C = torch.where(protein_batch["token_exists_mask"].bool(), protein_batch["asym_id"] + 1, -(protein_batch["asym_id"] + 1))
            S = torch.zeros_like(protein_batch["token_exists_mask"])
            Xs, Cs, Ss = [], [], []
            for i in range(X.shape[0]):
                protein = Protein.from_XCS(X[i:i+1], C[i:i+1], S[i:i+1])
                protein.canonicalize()
                X_i, C_i, S_i = protein.to_XCS()
                Xs.append(X_i)
                C_i = C_i.squeeze(0)[protein_batch["token_pad_mask"][i].bool()]
                batch["token_exists_mask"][i, protein_token_mask[i]] = (C_i > 0).float()   # update token_exists_mask to reflect canonicalization
            X = torch.cat(Xs, dim=0)
            X_prot_bb = X

        if not is_sampling and self.training:
            # Only apply noise if not sampling and training
            X_prot_bb = self.noise_perturb(X_prot_bb, protein_batch["token_exists_mask"], t)

        X_prot[..., const.prot_bb_atom14_idxs, :] = X_prot_bb  # update protein coordinates with possibly canonicalized and noised backbone atoms

        # Scatter back to atom coordinates
        X_all = get_tokenwise_coords(batch)
        for i in range(X_all.shape[0]):
            X_all[i, protein_token_mask[i]] = X_prot[i][protein_batch["token_pad_mask"][i].bool()]
        batch["coords"] = get_atomwise_coords(batch, X_all)
        batch["t"] = t
        return batch


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        """
        super().__init__()
        self.cfg = cfg

        # Parameters
        self.ca_only = cfg.get("ca_only", True)  # backwards compatibility
        self.k_neighbors = cfg.k_neighbors
        self.num_rbf = cfg.num_rbf
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.edge_n_channel = cfg.edge_n_channel
        self.use_multichain_encoding = cfg.get("use_multichain_encoding", False)

        # Layers
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        num_pairwise_dists = 1 if self.ca_only else const.max_num_atoms ** 2
        edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists + 1
        self.edge_embedding = nn.Linear(edge_in, self.edge_n_channel, bias=False)
        self.norm_edges = nn.LayerNorm(self.edge_n_channel)


    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        X = self._get_token_coords(batch)
        D_neighbors, E_idx = self._dist(X, batch["token_exists_mask"].float())

        # Get RBF features
        if self.ca_only:
            RBF_all = self._rbf(D_neighbors)
        else:
            X_all, tokenwise_atom_cond_mask = get_tokenwise_coords(batch)
            X_all = torch.where(tokenwise_atom_cond_mask.unsqueeze(-1).bool(), X_all, X[..., None, :])  # replace all masked atoms with center atom for the residue

            RBF_all = []
            for i in range(X_all.shape[-2]):
                for j in range(X_all.shape[-2]):
                    RBF_all.append(self._get_rbf(X_all[..., i, :], X_all[..., j, :], E_idx))
            RBF_all = torch.cat(RBF_all, dim=-1)

        # Positional encodings
        residue_index = batch["residue_index"]
        offset = residue_index[:,:,None] - residue_index[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0]  # [B, L, K]

        chain_labels = torch.zeros_like(batch["asym_id"])
        if self.use_multichain_encoding:
            # only use multichain encoding if the model has been trained with it TODO: need to also handle residue index
            chain_labels = batch["asym_id"]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)

        # AF3 token_bond feature
        token_bonds = batch["token_bonds"]

        # (JH): fix to remove polymer-polymer bonds
        token_bonds_mask = (batch["mol_type"] == const.chain_type_ids["NONPOLYMER"]) # [B, L]
        token_bonds_mask = (token_bonds_mask[:,:,None] | token_bonds_mask[:,None,:])[..., None] # [B, L, L, 1]
        token_bonds = token_bonds * token_bonds_mask

        token_bonds = gather_edges(token_bonds, E_idx)

        # Concatenate edge features and embed
        E = torch.cat((E_positional, RBF_all, token_bonds), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx, D_neighbors


    def _get_token_coords(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n 3", float]:
        """
        Get token-level coordinates as an average over all known, resolved atoms in the token.
        """
        B, N, _ = batch["coords"].shape
        _, center_idx = torch.max(batch["token_to_center_atom"], dim=-1)
        X = batch["coords"][torch.arange(B).unsqueeze(-1), center_idx]  # get center atom for each token
        X = X * batch["token_exists_mask"].unsqueeze(-1)  # mask out padding and unresolved atoms
        return X


    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, sorted=True, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden + num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden * 2 + num_in, num_hidden, bias=True) # nh * 2 for vi AND vj
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_in, bias=True) # num_in is hidden dim of edges h_E
        self.norm3 = nn.LayerNorm(num_in)
        self.dropout3 = nn.Dropout(dropout)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, E_idx = None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        #edge updates
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, is_last_layer=False):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        if not is_last_layer:
            # only initialize if not last layer to avoid unused parameters
            self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
            self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.norm3 = nn.LayerNorm(num_hidden)


    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if not self.is_last_layer:
            # Edge updates
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class AtomGraphEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Similar to AF3 atom-attention encoder, but using structure to build a KNN graph rather than using
        sequence-local attention.
        """
        super(AtomGraphEncoder, self).__init__()
        self.cfg = cfg

        self.k_atom_neighbors = cfg.k_atom_neighbors
        self.atom_feature_dim = cfg.atom_feature_dim
        self.token_n_channel = cfg.token_n_channel
        self.atom_s = cfg.atom_s
        self.atom_z = cfg.atom_z
        self.dropout_p = cfg.dropout_p
        self.n_layers = cfg.n_layers

        # Embed 1D features
        self.embed_atom_features = LinearNoBias(self.atom_feature_dim, self.atom_s)

        # Embed 2D features
        # Reference position embeddings
        self.embed_ref_dist = LinearNoBias(1, self.atom_z)  # unlike AF3, we embed the distance, not the direction offset. TODO: switch to RBF?
        self.embed_inv_ref_dist = LinearNoBias(1, self.atom_z)  # 1 / (1 + dist**2)
        self.embed_v_mask = LinearNoBias(1, self.atom_z)  # embed mask for within-conformer edges

        # Coordinate embeddings
        self.embed_dist = LinearNoBias(1, self.atom_z)  # TODO: embed RBF?
        self.embed_inv_dist = LinearNoBias(1, self.atom_z)  # 1 / (1 + dist**2)
        self.embed_edge_mask = LinearNoBias(1, self.atom_z)  # embed mask for edges

        # Edge embedding MLP
        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(self.atom_z, self.atom_z),
            nn.ReLU(),
            LinearNoBias(self.atom_z, self.atom_z),
            nn.ReLU(),
            LinearNoBias(self.atom_z, self.atom_z),
        )

        # Graph encoding layers
        self.layers = nn.ModuleList([
            AtomGraphEncoderLayer(self.atom_s, self.atom_z,
                                  dropout=self.dropout_p,
                                  is_last_layer=(i == self.n_layers - 1))
            for i in range(self.n_layers)
        ])

        # Aggregation to token-level features
        self.atom_to_token_trans = nn.Sequential(LinearNoBias(self.atom_s, self.token_n_channel),
                                                 nn.ReLU())



    def forward(self, batch: dict[str, TensorType["b ..."]]):
        B, N, _ = batch["ref_pos"].shape
        K = self.k_atom_neighbors

        # Embed 1D features
        atom_cond_mask = batch["atom_cond_mask"].float()
        ref_charge = batch["ref_charge"] * atom_cond_mask
        ref_element = batch["ref_element"] * atom_cond_mask.unsqueeze(-1)
        ref_atom_name_chars = batch["ref_atom_name_chars"].reshape(B, N, 4 * 64) * atom_cond_mask.unsqueeze(-1)

        atom_feats = torch.cat(
            [
                # atom_ref_pos,  # not invariant
                atom_cond_mask.unsqueeze(-1),  # not needed?
                ref_charge.unsqueeze(-1),
                ref_element,
                ref_atom_name_chars,
            ],
            dim=-1,
        ).to(batch["coords"].dtype)
        c = self.embed_atom_features(atom_feats)

        # Build KNN graph on atoms we're conditioning on
        X = batch["coords"] * atom_cond_mask.unsqueeze(-1)
        E_idx = knn_neighbors_batched(X, atom_cond_mask, k=K)
        mask_2d = gather_nodes(atom_cond_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_2d = mask_2d * atom_cond_mask.unsqueeze(-1)

        # Embed offsets between reference positions
        ## get valid mask (within-conformer edges)
        atom_ref_pos = batch["ref_pos"] * batch["atomwise_seq_cond_mask"].unsqueeze(-1)  # if residue type is masked, set the whole ref_pos to 0 to avoid leakage
        ref_space_uid = batch["ref_space_uid"] * atom_cond_mask

        uid_neighbors = gather_nodes(ref_space_uid.unsqueeze(-1), E_idx).squeeze(-1)
        v_mask = (ref_space_uid.view(B, N, 1) == uid_neighbors) * mask_2d  # [B, N, K]
        v_mask = v_mask.unsqueeze(-1)  # [B, N, K, 1]

        ## embed distances between reference positions
        ref_pos_neighbors = gather_nodes(atom_ref_pos, E_idx)
        d_ref = (atom_ref_pos.view(B, N, 1, 3) - ref_pos_neighbors).norm(dim=-1, keepdim=True)
        inv_d_ref = 1 / (1 + d_ref**2)

        p = self.embed_ref_dist(d_ref) * v_mask  # [B, N, K, atom_n_channel]
        p = p + self.embed_inv_ref_dist(inv_d_ref) * v_mask
        p = p + self.embed_v_mask(v_mask) * v_mask

        # Embed distances between real positions
        coords_neighbors = gather_nodes(X, E_idx)
        d = (X.view(B, N, 1, 3) - coords_neighbors).norm(dim=-1, keepdim=True)
        inv_d = 1 / (1 + d**2)
        edge_mask = mask_2d.unsqueeze(-1)

        p = p + self.embed_dist(d) * edge_mask
        p = p + self.embed_inv_dist(inv_d) * edge_mask
        p = p + self.embed_edge_mask(edge_mask) * edge_mask
        p = p + self.p_mlp(p)  # embed 2D features

        # Run graph encoding layers
        q = c
        for layer in self.layers:
            q, p = layer(q, p, E_idx, mask_V=atom_cond_mask, mask_attend=mask_2d)

        # Aggregate to token-level features
        q_to_a = self.atom_to_token_trans(q) * atom_cond_mask.unsqueeze(-1)
        cond_atom_to_token = batch["atom_to_token"] * atom_cond_mask.unsqueeze(-1)  # cond_atom_to_token ensures that we don't average over unresolved or masked atoms
        atom_to_token_mean = cond_atom_to_token / (
            cond_atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        with torch.autocast(device_type="cuda", enabled=False):
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a


class AtomGraphEncoderLayer(nn.Module):
    def __init__(self, atom_s, atom_z, dropout=0.1, scale=30, is_last_layer=False):
        super(AtomGraphEncoderLayer, self).__init__()
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.scale = scale
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(atom_s)
        self.norm2 = nn.LayerNorm(atom_s)

        self.W1 = nn.Linear(2 * atom_s + atom_z, atom_s, bias=True)
        self.W2 = nn.Linear(atom_s, atom_s, bias=True)
        self.W3 = nn.Linear(atom_s, atom_s, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(atom_s, atom_s * 4)

        if not is_last_layer:
            # only initialize edge updates if not last layer to avoid unused parameters
            self.W11 = nn.Linear(2 * atom_s + atom_z, atom_z, bias=True)
            self.W12 = nn.Linear(atom_z, atom_z, bias=True)
            self.W13 = nn.Linear(atom_z, atom_z, bias=True)
            self.norm3 = nn.LayerNorm(atom_z)


    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if not self.is_last_layer:
            # Edge updates
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E



def knn_neighbors_batched(
    coords: TensorType["b n 3"],
    atom_mask: TensorType["b n"],
    k: int
) -> TensorType["b n k"]:
    """
    Returns a [B, N, k] tensor of neighbor indices per atom, ignoring padded and non-existent atoms
    and cutting off ties so each atom has at most k neighbors.
    Neighbors are "arbitrary among ties" because knn_graph may return more than k
    if distances tie. We keep only the first k that appear in its output.
    Positions for padded or non-existent atoms (or if <k neighbors) are filled with 0.
    """
    device = coords.device
    B, N, _ = coords.shape

    # Flatten batch dimension
    # shape = [B*N, 3], [B*N], etc.
    coords_flat = coords.view(B*N, 3)
    mask_flat = atom_mask.view(B*N)

    # Identify which flattened indices are valid (unmasked)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]  # 1D indices into [B*N]
    coords_valid = coords_flat[valid_idx]            # [M, 3], M <= B*N

    # For each valid flattened index, figure out (batch_idx, atom_idx_within_batch)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N).flatten()  # which batch each atom belongs to
    atom_indices  = torch.arange(N, device=device).unsqueeze(0).expand(B, N).flatten()  # which atom index within that batch
    valid_batch_idx = batch_indices[valid_idx]  # [M]
    valid_atom_idx  = atom_indices[valid_idx]   # [M]

    # Run knn_graph on the valid points (no cross-batch edges, thanks to batch=...)
    # shape of edge_index is [2, E], row=src, col=dst each in [0..M-1].
    edge_index = knn_graph(x=coords_valid, k=k, batch=valid_batch_idx, loop=False)
    dst, src = edge_index

    ###
    # We might get more than k edges per node if there are distance ties.
    # We'll keep only the first k edges per source node.
    ###

    # First, we group edges by their source node and count them up
    src_batch = valid_batch_idx[src]
    src_atom = valid_atom_idx[src]
    src_key = src_batch * N + src_atom  # [E], denotes the source node for each edge, where the source node is indexed by [B*N]
    E = src_key.size(0)

    ## sort the edges by source node to ensure that edges for the same source node are consecutive
    src_key_sorted, sorted_idx = torch.sort(src_key, stable=True)  # stable=True to preserve original edge order
    dst_sorted = dst[sorted_idx]  # align the destination nodes with the sorted source nodes

    ## count how many edges belong to each source node
    counts = torch.bincount(src_key_sorted, minlength=B*N)  # [B*N], index i represents the number of edges for source node i

    # Next, we filter out the edges that are not the first k edges per source node
    cumsum_counts = counts.cumsum(dim=0)
    start_offset = cumsum_counts[src_key_sorted] - counts[src_key_sorted]  # [E], represents the index of the first edge for the source node that this edge belongs to
    local_idx = torch.arange(E, device=device) - start_offset  # [E], represents the index of this edge in the edge list for the source node

    keep_edge_mask = local_idx < k  # keep only the first k edges per source node
    src_key_final = src_key_sorted[keep_edge_mask]  # [E], represents the source node for each edge
    dst_final = dst_sorted[keep_edge_mask]  # [E], represents the destination node for each edge
    dst_atom_final = valid_atom_idx[dst_final]  # [E], represents the atom index of the destination node within the batch
    local_idx_final = local_idx[keep_edge_mask]  # [E], represents the index of this edge in the edge list for the source node capped at k

    # Build up neighbors, with -1 for padded or non-existent atoms
    neighbors = torch.full((B, N, k), -1, device=device, dtype=torch.long)

    ## get the destination node indices within the batch for each edge
    neighbors_flat = neighbors.view(B * N * k)  # flatten for scatter
    scatter_index = src_key_final * k + local_idx_final  # scatter_index denotes the position in neighbors_flat to store the destination node index
    neighbors_flat.scatter_(0, scatter_index, dst_atom_final)  # scatter the destination node indices to the appropriate positions in neighbors_flat

    neighbors = neighbors_flat.view(B, N, k)
    neighbors = neighbors.clamp(min=0)  # clamp padding and non-existent atoms to 0
    return neighbors


def get_tokenwise_coords(batch: dict[str, TensorType["b ..."]]) -> tuple[TensorType["b n_tokens 23 3", float], TensorType["b n_tokens 23"]]:
    """
    Get token-level coordinates (padded to max_num_atoms per token). Batched version of pad_atom_feats_to_tokenwise for just coords.
    """
    B = batch["coords"].shape[0]
    device = batch["coords"].device

    # Build padded atom idxs
    n_atoms_per_token = batch["atom_to_token"].sum(dim=-2)
    atom_idxs = torch.cat([torch.zeros((B, 1), device=device), n_atoms_per_token.cumsum(dim=-1)[:, :-1]], dim=-1).long()
    padded_atom_idxs = atom_idxs[..., None] + torch.arange(const.max_num_atoms, device=device)[None, None]
    pad_mask = torch.arange(const.max_num_atoms, device=device)[None, None, :] < n_atoms_per_token[..., None]
    padded_atom_idxs = padded_atom_idxs * pad_mask  # mask out ghost atoms

    # Gather coords  # TODO: check this carefully
    B, N, _ = padded_atom_idxs.shape
    X_all = batched_gather(batch["coords"], padded_atom_idxs, dim=1, no_batch_dims=1) * pad_mask.view(B, N, const.max_num_atoms, 1)
    tokenwise_atom_cond_mask = batched_gather(batch["atom_cond_mask"], padded_atom_idxs, dim=1, no_batch_dims=1) * pad_mask.view(B, N, const.max_num_atoms)

    X_all = X_all * tokenwise_atom_cond_mask.unsqueeze(-1)  # zero out masked atoms
    return X_all, tokenwise_atom_cond_mask


def get_atomwise_coords(
    batch: dict[str, TensorType["b ..."]],
    tokenwise_coords: TensorType["b n_tokens 23 3", float],
) -> TensorType["b n_atoms 3", float]:
    """
    Inverse of get_tokenwise_coords. Given tokenwise coords [B, n_tokens, max_num_atoms, 3],
    reconstruct atomwise coords [B, n_atoms, 3].
    """
    B = batch["coords"].shape[0]
    device = batch["coords"].device

    x = batch["atomwise_token_idx"] * tokenwise_coords.shape[-2]  # flattened atomwise token indices
    is_start = torch.ones_like(x, dtype=torch.bool)
    is_start[:, 1:] = x[:, 1:] != x[:, :-1]
    pos = torch.arange(x.shape[-1], device=x.device).unsqueeze(0).expand(B, x.shape[-1])
    start_pos = torch.where(is_start, pos, torch.full_like(pos, -1))
    first_pos = torch.cummax(start_pos, dim=1).values
    local_idx = pos - first_pos
    gather_idx = x + local_idx
    gather_idx = gather_idx

    new_coords = batched_gather(tokenwise_coords.view(B, -1, 3), gather_idx, dim=1, no_batch_dims=1)
    return new_coords
