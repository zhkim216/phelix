import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import atom37_to_atom14
from allatom_design.model.seq_denoiser.denoisers.seq_design.gvp.gvp_modules import \
    GVPEncoder
from allatom_design.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes, gather_edges, gather_nodes)


class FAMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.n_aatype = cfg.n_aatype
        self.seq_emb_dim = cfg.n_channel
        self.model_type = cfg.model_type
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors

        # Backbone noise options
        self.augment_eps = cfg.augment_eps
        self.per_residue_eps = cfg.get("per_residue_eps", False)
        self.max_eps = getattr(cfg, "max_eps", None)
        self.ablate_noise_labels = getattr(cfg, "ablate_noise_labels", False)

        self.features = ProteinFeatures(self.node_features, self.edge_features, top_k=self.k_neighbors,
                                        per_residue_eps=self.per_residue_eps, augment_eps=self.augment_eps, max_eps=self.max_eps)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_s = nn.Embedding(self.n_aatype, self.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout_p)
        self.decoder_in = self.hidden_dim * 3
        self.return_embedding = cfg.return_embedding
        self.use_gvp = cfg.use_gvp

        if self.model_type not in ['sidechain', 'baseline']:
            raise ValueError(f'Incorrect model type specified: {self.model_type}, must be one of: sidechain, or baseline!')

        if self.model_type in ['sidechain']:
            self.decoder_in += self.hidden_dim
            self.sidechain_features = SidechainProteinFeatures(node_features=self.node_features,
                                                               edge_features=self.edge_features,
                                                               top_k=self.k_neighbors)
            self.W_e2 = nn.Linear(self.edge_features, self.hidden_dim, bias=True)

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

        # GVP embeds vector features with scalar features
        if self.use_gvp:
            self.vector_encoder = GVPEncoder(cfg.gvp)

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_aatype, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(
        self,
        denoised_coords: TensorType["b n a x", float],
        aatype_noised: TensorType["b n", int], #will have UNK tokens where masking occurs
        seq_mask: TensorType["b n", float],
        atom_mask_noised: TensorType["b n a", float],  # denotes missing, ghost, and masked atoms
        residue_index: TensorType["b n", int],
        chain_encoding: TensorType["b n", int],
        noise_labels: Optional[Union[float, TensorType["b n"]]] = None,
    ):

        B, N, _, _ = denoised_coords.shape
        S = aatype_noised

        #prepare inputs for protein mpnn
        X, atom14_mask = atom37_to_atom14(aatype_noised, denoised_coords, atom37_mask=atom_mask_noised)
        X = torch.where(atom14_mask[..., None].bool(), X, X[..., 1:2, :])  # replace missing/ghost/masked atoms with CA

        # Prepare node and edge embeddings
        E, E_idx, X, noise_labels = self.features(X, seq_mask, residue_index, chain_encoding, noise_labels)

        #h_V is size [B,N,H]
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        if self.per_residue_eps and (self.augment_eps > 0) and not self.ablate_noise_labels:
            # add noise label to 128th dimension of h_V
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
        h_S = self.W_s(aatype_noised)
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

        mpnn_feature_dict = {"h_V": h_V_out, "h_ESV": h_ESV, "X": X, "atom14_mask": atom14_mask, "E_idx": E_idx, "S": S}
        return logits, mpnn_feature_dict


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, per_residue_eps=False, augment_eps=0., max_eps=None, num_chain_embeddings=16,):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.per_residue_eps = per_residue_eps
        self.max_eps = max_eps
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
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

    def forward(self, X, mask, residue_idx, chain_labels,
                noise_labels: Optional[Union[float, TensorType["b n"]]]):
        if self.per_residue_eps:
            # per-residue noise, based on pseudocode from Cho et al.
            if self.training and self.augment_eps > 0:
                # Training: randomly sample noise labels
                r = torch.randn_like(X)  # random vector for each atom (this might differ from Cho et al.)
                n = r / torch.norm(r, dim=-1, keepdim=True)
                s = truncated_half_normal_like(mask, self.augment_eps, self.max_eps) # per-residue noise label
                noise = n * rearrange(s, "b n -> b n 1 1")
                noise_labels = torch.abs(s)  # DISCREPANCY: noise labels should be positive
                X = X + noise
            elif (noise_labels is None) or (self.augment_eps == 0):
                # Inference: assume 0 noise if not provided
                noise_labels = torch.zeros_like(mask)
            elif isinstance(noise_labels, float):
                # Inference: constant noise label for every residue
                noise_labels = torch.ones_like(mask) * noise_labels
        elif self.training and self.augment_eps > 0:
            # training: add randomly sampled noise to input
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        N = X[:,:,0,:]
        Ca = X[:,:,1,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx, X, noise_labels

class SidechainProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30):
        """ Extract protein features """
        super(SidechainProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.zero_ghost_atoms = False

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        _, edge_in = 6, num_positional_embeddings + num_rbf * 4 * 10
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps) #2d mask makes it so only points which are both unmasked have distances
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
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

    def forward(self, X, residue_idx, chain_labels, E_idx, atom_mask):
        max_atoms = X.shape[-2] #14
        N = X[:,:,0,:]
        Ca = X[:,:,1,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        RBF_all = []

        for bb_atom in [Ca, N, C, O]:
            for non_bb_atom_pos in range(rc.num_bb_atoms, max_atoms):
                non_bb_atom_mask = atom_mask[:,:,non_bb_atom_pos]
                non_bb_atom_mask_neighbors = torch.gather(non_bb_atom_mask[...,None].expand(-1,-1,self.top_k), 1, E_idx)
                rbf = self._get_rbf(bb_atom, X[:, :, non_bb_atom_pos, :],  E_idx)

                #insert 0 for rbf where destination atom does not exist, if specified
                if self.zero_ghost_atoms:
                    rbf = torch.where(non_bb_atom_mask_neighbors[...,None].expand(-1,-1,-1,self.num_rbf) == 1, rbf, 0)
                RBF_all.append(rbf)

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)


        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


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


def truncated_half_normal_like(x: TensorType["...", float],
                               std: float, max_val: Optional[float]) -> TensorType["...", float]:
    if max_val is None:
        # return half-normal with no truncation
        return torch.abs(torch.randn_like(x) * std)
    u = torch.rand_like(x)
    truncated_factor = torch.erf(torch.tensor(max_val / (math.sqrt(2) * std)))
    u_scaled = u * truncated_factor
    samples = std * math.sqrt(2) * torch.erfinv(u_scaled)
    return samples
