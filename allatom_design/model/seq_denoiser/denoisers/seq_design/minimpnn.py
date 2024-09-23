import itertools
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc


class MiniMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.autoregressive = cfg.autoregressive
        self.n_aatype = cfg.n_aatype
        self.seq_emb_dim = cfg.n_channel
        self.use_self_conditioning_seq = cfg.use_self_conditioning_seq
        self.use_time_cond = cfg.use_time_cond
        assert not cfg.use_self_conditioning_seq and not cfg.use_time_cond, "Not implemented yet"

        self.model_type = cfg.model_type
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.augment_eps = cfg.augment_eps
        self.no_aatype_pred = cfg.no_aatype_pred

        if self.model_type == "protein_mpnn":
            self.features = ProteinFeatures(
                self.node_features, self.edge_features, top_k=self.k_neighbors, augment_eps=self.augment_eps
            )
        else:
            print("Choose --model_type flag from currently available models")
            sys.exit()

        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)

        # Input sequence embeddings
        self.in_channels = self.n_aatype
        if self.use_self_conditioning_seq:
            self.in_channels = self.in_channels + self.n_aatype  # concatenate previous prediction

        self.W_s = nn.Linear(self.in_channels, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout_p)

        # MiniMPNN: time conditioning
        time_cond_dim = None
        if self.use_time_cond:
            time_cond_dim = cfg.n_channel * cfg.noise_cond_mult
            self.noise_block = NoiseConditioningBlock(cfg.n_channel, time_cond_dim)
            self.time_block = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_cond_dim, self.hidden_dim)
            )

        # Encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [
                EncLayer(self.hidden_dim, self.hidden_dim * 2, dropout=cfg.dropout_p, time_cond_dim=time_cond_dim)
                for _ in range(self.num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList(
            [
                DecLayer(self.hidden_dim, self.hidden_dim * 3, dropout=cfg.dropout_p, time_cond_dim=time_cond_dim)
                for _ in range(self.num_decoder_layers)
            ]
        )

        # Output layers
        if not self.no_aatype_pred:
            self.W_out = nn.Linear(self.hidden_dim, self.n_aatype, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(
        self,
        denoised_coords: TensorType["b n a x", float],
        aatype_noised: TensorType["b n", int],
        seq_self_cond: Optional[TensorType["b n k", float]],  # logits
        t_bb: TensorType["b", float],  # backbone time
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
    ):
        # use backbone atoms only as X
        denoised_coords = denoised_coords[..., rc.bb_idxs, :]

        # use one-hot aatype as S
        aatype_oh_noised = F.one_hot(aatype_noised, self.n_aatype).float()

        # condition on backbone time
        time_cond = None
        if self.use_time_cond:
            time_cond = self.noise_block(t_bb)

        feature_dict = {
            "X": denoised_coords,
            "S": aatype_oh_noised,
            "S_self_cond": seq_self_cond,
            "time_cond": time_cond,
            "mask": seq_mask,
            "chain_mask": seq_mask,  # TODO: double check this
            "R_idx": residue_index,
            "chain_labels": seq_mask,  # TODO: add chain index here?
            "randn": None,
            }

        ### Encoder ###
        h_V, h_E, E_idx = self.encode(feature_dict)

        ### Decoder ###
        S = feature_dict["S"]
        S_self_cond = feature_dict["S_self_cond"]

        # Concatenate self-conditioning
        if self.use_self_conditioning_seq:
            if S_self_cond is None:
                S_self_cond = torch.zeros_like(S)
            else:
                # One-hot encode the argmax prediction
                S_self_cond = F.one_hot(S_self_cond.argmax(dim=-1), self.n_aatype)
            S = torch.cat([S, S_self_cond], dim=-1)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        mask_size = E_idx.shape[1]
        if self.autoregressive:
            decoding_order = torch.argsort((seq_mask+0.0001)*(torch.abs(torch.randn(seq_mask.shape, device=seq_mask.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=seq_mask.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        else:
            order_mask_backward = torch.ones(S.shape[0], mask_size, mask_size, device=E_idx.device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = seq_mask.view([seq_mask.size(0), seq_mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked seq.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, seq_mask, time_cond=time_cond)

        if self.no_aatype_pred:
            return None, h_V

        logits = self.W_out(h_V)
        return logits, h_V


    def encode(self, feature_dict):
        seq_mask = feature_dict["mask"]
        time_cond = feature_dict["time_cond"]
        device = seq_mask.device

        # Prepare node and edge embeddings
        E, E_idx = self.features(feature_dict)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)

        # Time conditioning
        if self.use_time_cond:
            time_cond_nodes = self.time_block(time_cond)
            h_V += time_cond_nodes  # time_cond is [b, 1, c]

        # Embed edge features
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(seq_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = seq_mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, seq_mask, mask_attend, time_cond)

        return h_V, h_E, E_idx


class NoiseConditioningBlock(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super().__init__()
        self.block = nn.Sequential(
            Noise_Embedding(n_in_channel),
            nn.Linear(n_in_channel, n_out_channel),
            nn.SiLU(),
            nn.Linear(n_out_channel, n_out_channel),
            Rearrange("b d -> b 1 d"),
        )

    def forward(self, noise_level):
        return self.block(noise_level)


class Noise_Embedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class ProteinFeatures(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
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


class DecLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, time_cond_dim=None):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.use_time_cond = False
        if time_cond_dim is not None:
            self.use_time_cond = True
            self.time_block = nn.Sequential(
                Rearrange('b 1 d -> b 1 1 d'),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2))

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None, time_cond=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.act(self.W2(self.act(self.W1(h_EV))))
        if self.use_time_cond:
            scale, shift = self.time_block(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W3(h_message)

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
        return h_V


class EncLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, time_cond_dim=None):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)
        self.norm3 = torch.nn.LayerNorm(num_hidden)

        self.use_time_cond = False
        if time_cond_dim is not None:
            self.use_time_cond = True
            self.time_block1 = nn.Sequential(
                Rearrange('b 1 d -> b 1 1 d'),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2))
            self.time_block2 = nn.Sequential(
                Rearrange('b 1 d -> b 1 1 d'),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2))


        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, time_cond=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.act(self.W2(self.act(self.W1(h_EV))))
        if self.use_time_cond:
            # Time-conditioning
            scale, shift = self.time_block1(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W3(h_message)

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
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.act(self.W12(self.act(self.W11(h_EV))))
        if self.use_time_cond:
            # Time-conditioning
            scale, shift = self.time_block2(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W13(h_message)

        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
