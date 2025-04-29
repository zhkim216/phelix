import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from boltz.model.modules.utils import LinearNoBias
from omegaconf import DictConfig
from torch_cluster import knn_graph
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
import allatom_design.data.const as const
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes, gather_edges, gather_nodes)

# https://github.com/pyg-team/pytorch_geometric/issues/8747
knn_graph = torch.compiler.disable(knn_graph)


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
        self.W_s = nn.Linear(self.node_features + len(const.tokens), self.hidden_dim, bias=False)
        self.decoder_in = self.hidden_dim * 2  # concat of h_V and h_E

        self.dropout = nn.Dropout(cfg.dropout_p)

        # Atom-level encoder
        self.atom_encoder = AtomGraphEncoder(cfg.atom_encoder)

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
        if self.use_potts:
            self.parameterization = cfg.potts.parameterization
            self.num_factors = cfg.potts.num_factors
            self.decoder_S_potts = potts.GraphPotts(
                dim_nodes=self.node_features,
                dim_edges=self.decoder_in,
                # num_states=self.n_aatype,
                num_states=len(const.tokens),
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, len(const.tokens), bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, batch: dict[str, TensorType["b ..."]]):
        # Get token-level features
        h_V = self.atom_encoder(batch)

        # Concatenate residue-level features to h_V
        h_V = torch.cat([h_V, batch["res_type"]], dim=-1)

        # Build graph and get edge features
        h_E, E_idx, token_mask = self.token_features(batch)

        # Pass through encoder layers
        h_V, h_E = self.W_s(h_V), self.W_e(h_E)
        token_mask_2d = gather_nodes(token_mask.unsqueeze(-1), E_idx).squeeze(-1)
        token_mask_2d = token_mask.unsqueeze(-1) * token_mask_2d
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, token_mask, token_mask_2d)

        # Pass through decoder layers
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        for layer in self.decoder_layers:
            h_V, h_EV = layer(h_V, h_EV, token_mask, E_idx)

        # Potts model
        if self.use_potts:
            h, J = self.decoder_S_potts(h_V, h_EV, E_idx, token_mask, token_mask_2d)
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": token_mask,
                "mask_ij": token_mask_2d,
            }

        logits = self.W_out(h_V)

        # Output features
        mpnn_feature_dict = {"h_V": h_V, "h_EV": h_EV, "E_idx": E_idx}
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux

        return logits, mpnn_feature_dict


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        """
        super().__init__()
        self.cfg = cfg

        # Parameters
        self.k_neighbors = cfg.k_neighbors
        self.num_rbf = cfg.num_rbf
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.edge_n_channel = cfg.edge_n_channel

        # Layers
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        edge_in = self.num_positional_embeddings + self.num_rbf
        self.edge_embedding = nn.Linear(edge_in, self.edge_n_channel, bias=False)
        self.norm_edges = nn.LayerNorm(self.edge_n_channel)


    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        X, token_mask = self._get_token_coords(batch)
        D_neighbors, E_idx = self._dist(X, token_mask)

        RBF_all = self._rbf(D_neighbors)

        # Positional encodings
        offset = batch["residue_index"][:,:,None] - batch["residue_index"][:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0]  # [B, L, K]

        chain_labels = batch["asym_id"]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx, token_mask


    def _get_token_coords(self, batch: dict[str, TensorType["b ..."]]) -> tuple[TensorType["b n 3", float],
                                                                                TensorType["b n", float]]:
        """
        Get token-level coordinates as an average over all known, resolved atoms in the token.
        """
        # mask out padding and unresolved atoms just in case
        atom_mask = batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        X = batch["coords"] * atom_mask.unsqueeze(-1)

        # normalize by the number of resolved atoms in the token
        resolved_atom_to_token = batch["resolved_atom_to_token"]
        atom_to_token_mean = resolved_atom_to_token / (
            resolved_atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )

        with torch.autocast(device_type="cuda", enabled=False):
            # Average over all known, resolved atoms in the token
            X = torch.bmm(atom_to_token_mean.transpose(1, 2), X)  # [B, N, 3]

        return X


    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, largest=False)
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
        self.norm3 = nn.LayerNorm(num_hidden)

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
        self.atom_n_channel = cfg.atom_n_channel
        self.token_n_channel = cfg.token_n_channel
        self.dropout_p = cfg.dropout_p
        self.n_layers = cfg.n_layers

        # Embed 1D features
        self.embed_atom_features = LinearNoBias(self.atom_feature_dim, self.atom_n_channel)

        # Embed 2D features
        # Reference position embeddings
        self.embed_ref_dist = LinearNoBias(1, self.atom_n_channel)  # unlike AF3, we embed the distance, not the direction offset. TODO: switch to RBF?
        self.embed_inv_ref_dist = LinearNoBias(1, self.atom_n_channel)  # 1 / (1 + dist**2)
        self.embed_v_mask = LinearNoBias(1, self.atom_n_channel)  # embed mask for within-conformer edges

        # Coordinate embeddings
        self.embed_dist = LinearNoBias(1, self.atom_n_channel)  # TODO: embed RBF?
        self.embed_inv_dist = LinearNoBias(1, self.atom_n_channel)  # 1 / (1 + dist**2)
        self.embed_edge_mask = LinearNoBias(1, self.atom_n_channel)  # embed mask for edges

        # Edge embedding MLP
        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(self.atom_n_channel, self.atom_n_channel),
            nn.ReLU(),
            LinearNoBias(self.atom_n_channel, self.atom_n_channel),
            nn.ReLU(),
            LinearNoBias(self.atom_n_channel, self.atom_n_channel),
        )

        # Graph encoding layers
        self.layers = nn.ModuleList([
            EncLayer(self.atom_n_channel, self.atom_n_channel * 2, dropout=self.dropout_p,
                     is_last_layer=(i == self.n_layers - 1))
            for i in range(self.n_layers)
        ])

        # Aggregation to token-level features
        self.atom_to_token_trans = nn.Sequential(LinearNoBias(self.atom_n_channel, self.token_n_channel),
                                                 nn.ReLU())



    def forward(self, batch: dict[str, TensorType["b ..."]]):
        B, N, _ = batch["ref_pos"].shape
        K = self.k_atom_neighbors

        # Embed 1D features
        atom_mask = batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        atom_ref_pos = batch["ref_pos"]
        ref_space_uid = batch["ref_space_uid"]
        atom_feats = torch.cat(
            [
                # atom_ref_pos,  # not invariant
                atom_mask.unsqueeze(-1),  # not needed?
                batch["ref_charge"].unsqueeze(-1),
                batch["ref_element"],
                batch["ref_atom_name_chars"].reshape(B, N, 4 * 64),
            ],
            dim=-1,
        ).to(batch["coords"].dtype)
        c = self.embed_atom_features(atom_feats)

        # Build KNN graph
        E_idx = knn_neighbors_batched(batch["coords"], atom_mask, k=K)
        mask_2d = gather_nodes(atom_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_2d = mask_2d * atom_mask.unsqueeze(-1)

        # Embed offsets between reference positions
        ## get valid mask (within-conformer edges)
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
        coords_neighbors = gather_nodes(batch["coords"], E_idx)
        d = (batch["coords"].view(B, N, 1, 3) - coords_neighbors).norm(dim=-1, keepdim=True)
        inv_d = 1 / (1 + d**2)
        edge_mask = mask_2d.unsqueeze(-1)

        p = p + self.embed_dist(d) * edge_mask
        p = p + self.embed_inv_dist(inv_d) * edge_mask
        p = p + self.embed_edge_mask(edge_mask) * edge_mask
        p = p + self.p_mlp(p)  # embed 2D features

        # Run graph encoding layers
        q = c
        for layer in self.layers:
            q, p = layer(q, p, E_idx, mask_V=atom_mask, mask_attend=mask_2d)

        # Aggregate to token-level features
        q_to_a = self.atom_to_token_trans(q) * atom_mask.unsqueeze(-1)
        resolved_atom_to_token = batch["resolved_atom_to_token"]  # resolved_atom_to_token ensures that we don't average over unresolved atoms
        atom_to_token_mean = resolved_atom_to_token / (
            resolved_atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        with torch.autocast(device_type="cuda", enabled=False):
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a


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


def test_knn_neighbors_batched():
    # 1) Single-batch, no padding, distinct points
    coords_1 = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    ]])  # shape = (1, 4, 3)
    mask_1 = torch.ones((1, 4), dtype=torch.bool)  # all valid
    k_1 = 2
    neighbors_1 = knn_neighbors_batched(coords_1, mask_1, k_1)
    print("Test 1 (distinct points, single batch):")
    print("coords_1.shape =", coords_1.shape)
    print("neighbors_1.shape =", neighbors_1.shape)
    print("neighbors_1 =", neighbors_1, "\n")

    # 2) Single-batch, repeated points to test ties
    coords_2 = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # identical to first
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # identical to third
        [1.0, 0.0, 0.0],  # identical to third
    ]])
    mask_2 = torch.ones((1, 5), dtype=torch.bool)
    k_2 = 2
    neighbors_2 = knn_neighbors_batched(coords_2, mask_2, k_2)
    print("Test 2 (ties, single batch):")
    print("coords_2.shape =", coords_2.shape)
    print("neighbors_2.shape =", neighbors_2.shape)
    print("neighbors_2 =", neighbors_2, "\n")

    # 3) Single-batch with padding
    # Suppose we have 6 "slots" but only 4 valid points
    coords_3 = torch.tensor([[
        [0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0],  # non-existent
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [9.9, 9.9, 9.9],  # padded
        [9.9, 9.9, 9.9],  # padded
    ]])
    mask_3 = torch.tensor([[1, 0, 1, 1, 0, 0]], dtype=torch.bool)
    k_3 = 4
    neighbors_3 = knn_neighbors_batched(coords_3, mask_3, k_3)
    print("Test 3 (single batch with padding):")
    print("coords_3.shape =", coords_3.shape)
    print("neighbors_3.shape =", neighbors_3.shape)
    print("neighbors_3 =", neighbors_3, "\n")

    # 4) Multi-batch with some padding in each
    coords_4 = torch.tensor([
        [
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [9.9, 9.9, 9.9], # padded
        ],
        [
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [13.0, 13.0, 13.0],  # repeated
        ]
    ])
    mask_4 = torch.tensor([
        [1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ], dtype=torch.bool)
    k_4 = 2
    neighbors_4 = knn_neighbors_batched(coords_4, mask_4, k_4)
    print("Test 4 (multi-batch, partial padding):")
    print("coords_4.shape =", coords_4.shape)
    print("neighbors_4.shape =", neighbors_4.shape)
    print("neighbors_4 =", neighbors_4, "\n")
