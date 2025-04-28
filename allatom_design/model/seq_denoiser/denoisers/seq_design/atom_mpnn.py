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
from allatom_design.data.data import atom37_to_atom14
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes, gather_edges, gather_nodes)

# https://github.com/pyg-team/pytorch_geometric/issues/8747
knn_graph = torch.compiler.disable(knn_graph)


class AtomMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.n_aatype = cfg.n_aatype
        self.seq_emb_dim = cfg.n_channel
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.ablate_noise_labels = getattr(cfg, "ablate_noise_labels", False)
        self.init_hV_with_hS = getattr(cfg, "init_hV_with_hS", False)

        self.features = ProteinFeatures(self.node_features, self.edge_features, top_k=self.k_neighbors)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_s = nn.Embedding(self.n_aatype, self.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout_p)
        self.decoder_in = self.hidden_dim * 3
        self.return_embedding = cfg.return_embedding

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
                dim_edges=self.edge_features * 4,
                num_states=self.n_aatype,
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_aatype, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, batch: dict[str, TensorType["b ..."]]):
        # Get token-level features
        h_V = self.atom_encoder(batch)
        h_V = torch.cat([h_V, batch["res_type"]], dim=-1)

        # Build graph and get edge features
        E, E_idx = self.token_features(batch)


        # E, E_idx = self.edge_features(batch)

        # Concatenate residue-level features to h_V

        # Pass through encoder layers





        # B, N, _, _ = denoised_coords.shape
        # S = aatype_noised

        # #prepare inputs for protein mpnn
        # X, atom14_mask = atom37_to_atom14(aatype_noised, denoised_coords, atom37_mask=atom_mask_noised)
        # X = torch.where(atom14_mask[..., None].bool(), X, X[..., 1:2, :])  # replace missing/ghost/masked atoms with CA

        # # Add noise to input coordinates
        # if noise is not None:
        #     X = X + noise
        # if noise_labels is None:
        #     # assume 0 noise if not provided
        #     noise_labels = torch.zeros_like(seq_mask)
        # elif isinstance(noise_labels, float):
        #     # assume constant noise if provided as float
        #     noise_labels = torch.ones_like(seq_mask) * noise_labels

        # Prepare node and edge embeddings
        E, E_idx, X = self.features(X, seq_mask, residue_index, chain_encoding)

        h_S = self.W_s(aatype_noised)  # [B, N, H]
        if h_S_init is not None:
            # add h_S_init to h_S
            h_S = h_S + h_S_init

        if self.init_hV_with_hS:
            # Initialize h_V with sequence embeddings
            h_V = h_S
        else:
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)

        h_E = self.W_e(E)

        if not self.ablate_noise_labels:
            # add noise label to last dimension of h_V
            h_V[..., -1] = h_V[..., -1] + noise_labels

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(seq_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = seq_mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, seq_mask, mask_attend)

        #keep copy of node embeddings from encoder
        h_V_enc = h_V.clone()

        # mask is all 1s
        mask_size = E_idx.shape[1]
        order_mask_backward = torch.ones(S.shape[0], mask_size, mask_size, device=E_idx.device)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = seq_mask.view([seq_mask.size(0), seq_mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        # Concatenate sequence embeddings to edge embeddings
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # edge embedding of encoder gets zeros for sequence added -> hidden dim = [Enc Embedding + Seq 0s ][128*2]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros((B, N, self.hidden_dim), device = h_S.device), h_E, E_idx)

        if self.model_type in ['sidechain']:

            # Add empty hidden dim of 128 to end of h_EXV to later sum with added sidechain distance information
            h_EX_encoder = cat_neighbors_nodes(torch.zeros((B, N, self.hidden_dim), device = h_S.device), h_EX_encoder, E_idx)

            # Extract sidechain features and concatenate to edge embeddings
            E2, _ = self.sidechain_features(X, residue_index, chain_encoding, E_idx, atom14_mask)

            #128 -> 128
            h_E2 = self.W_e2(E2)

            #concatenate sidechain information to Edge and Seq embeddings, Hidden dim is [128 Enc Edge, 128 Seq, 128 SC] = [128 * 3]
            h_ES = torch.cat([h_ES, h_E2], dim = -1)

        #concat h_V to edge embedding
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        #concat h_V_j to h_E_ij
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)

        for layer in self.decoder_layers:
            #encoder representation added to masked decoder representation
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V, h_ESV = layer(h_V, h_ESV, seq_mask, E_idx)

        #keep copy of node embeddings from encoder
        h_V_dec = h_V.clone()

        if self.use_gvp:
            padding_mask = (seq_mask != 1)
            h_V_flattened = self.vector_encoder(X, aatype_noised, E_idx, h_V, h_ESV, padding_mask, atom14_mask)
            h_V = h_V_flattened.reshape(B, N, -1)

        # Potts model
        if self.use_potts:
            mask_ij = mask_bw.squeeze(-1)
            h, J = self.decoder_S_potts(h_V, h_ESV, E_idx, seq_mask, mask_ij)
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": seq_mask,
                "mask_ij": mask_ij,
            }

        logits = self.W_out(h_V)

        h_V_out = None
        if self.return_embedding == 'encoder':
            h_V_out = h_V_enc
        elif self.return_embedding == 'decoder':
            h_V_out = h_V_dec
        elif self.return_embedding == 'last':
            h_V_out = h_V
        else:
            raise ValueError(f'Incorrect return embedding type specified: {self.return_embedding}, must be one of: encoder, decoder, gnn, or last!')

        mpnn_feature_dict = {"h_V": h_V_out, "h_ESV": h_ESV, "X": X, "atom14_mask": atom14_mask, "E_idx": E_idx, "S": S, "noise_labels": noise_labels}
        if return_encoder_embeds:
            mpnn_feature_dict["h_V_enc"] = h_V_enc

        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux

        return logits, mpnn_feature_dict


class TokenEdgeFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        """
        super().__init__()
        self.cfg = cfg




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
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)


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
            EncLayer(self.atom_n_channel, self.atom_n_channel * 2, dropout=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        # Aggregation to token-level features
        self.atom_to_token_trans = nn.Sequential(LinearNoBias(self.atom_n_channel, self.token_n_channel),
                                                 nn.ReLU())



    def forward(self, batch: dict[str, TensorType["b ..."]]):
        B, N, _ = batch["ref_pos"].shape
        K = self.k_atom_neighbors

        # Embed 1D features
        atom_pad_mask = batch["atom_pad_mask"]
        atom_ref_pos = batch["ref_pos"]
        ref_space_uid = batch["ref_space_uid"]
        atom_feats = torch.cat(
            [
                # atom_ref_pos,  # not invariant
                batch["atom_pad_mask"].unsqueeze(-1),  # not needed?
                batch["ref_charge"].unsqueeze(-1),
                batch["ref_element"],
                batch["ref_atom_name_chars"].reshape(B, N, 4 * 64),
            ],
            dim=-1,
        ).to(batch["coords"].dtype)
        c = self.embed_atom_features(atom_feats)

        # Build KNN graph
        E_idx = knn_neighbors_batched(batch["coords"], atom_pad_mask, k=K)
        mask_2d = gather_nodes(atom_pad_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_2d = mask_2d * atom_pad_mask.unsqueeze(-1)

        # Embed offsets between reference positions
        ## get valid mask (within-conformer edges)
        uid_neighbors = gather_nodes(ref_space_uid.unsqueeze(-1), E_idx).squeeze(-1)
        v_mask = (ref_space_uid.view(B, N, 1) == uid_neighbors) * mask_2d  # [B, N, K]
        v_mask = v_mask.unsqueeze(-1)  # [B, N, K, 1]

        ## embed distances between reference positions
        ref_pos_neighbors = gather_nodes(atom_ref_pos, E_idx)
        d_ref = (atom_ref_pos.view(B, N, 1, 3) - ref_pos_neighbors).norm(dim=-1, keepdim=True)
        inv_d_ref = 1 / (1 + d_ref**2)

        p = self.embed_ref_dist(d_ref) * v_mask
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
            q, p = layer(q, p, E_idx, mask_V=atom_pad_mask, mask_attend=mask_2d)

        # Aggregate to token-level features
        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = batch["atom_to_token"].float()
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a


def knn_neighbors_batched(
    coords: TensorType["b n 3"],
    atom_pad_mask: TensorType["b n"],
    k: int
) -> TensorType["b n k"]:
    """
    Returns a [B, N, k] tensor of neighbor indices per atom, ignoring padded atoms
    and cutting off ties so each atom has at most k neighbors.
    Neighbors are "arbitrary among ties" because knn_graph may return more than k
    if distances tie. We keep only the first k that appear in its output.
    Positions for padded atoms (or if <k neighbors) are filled with 0.
    """
    device = coords.device
    B, N, _ = coords.shape

    # Flatten batch dimension
    # shape = [B*N, 3], [B*N], etc.
    coords_flat = coords.view(B*N, 3)
    mask_flat = atom_pad_mask.view(B*N)

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

    # Build up neighbors, with -1 for padded atoms
    neighbors = torch.full((B, N, k), -1, device=device, dtype=torch.long)

    ## get the destination node indices within the batch for each edge
    neighbors_flat = neighbors.view(B * N * k)  # flatten for scatter
    scatter_index = src_key_final * k + local_idx_final  # scatter_index denotes the position in neighbors_flat to store the destination node index
    neighbors_flat.scatter_(0, scatter_index, dst_atom_final)  # scatter the destination node indices to the appropriate positions in neighbors_flat

    neighbors = neighbors_flat.view(B, N, k)
    neighbors = neighbors.clamp(min=0)  # clamp padding to 0
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
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [9.9, 9.9, 9.9],  # padded
        [9.9, 9.9, 9.9],  # padded
    ]])
    mask_3 = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
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
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ], dtype=torch.bool)
    k_4 = 2
    neighbors_4 = knn_neighbors_batched(coords_4, mask_4, k_4)
    print("Test 4 (multi-batch, partial padding):")
    print("coords_4.shape =", coords_4.shape)
    print("neighbors_4.shape =", neighbors_4.shape)
    print("neighbors_4 =", neighbors_4, "\n")
