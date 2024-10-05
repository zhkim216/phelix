import itertools
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc

from allatom_design.data.data import (
    atom37_to_atom14,
    unpack,
    pack,
    get_rc_tensor,
    extract_ids_topk,
    gather_pos_enc
)

from .graph_transformer import GraphTransformer
from .esm_model import ESMWrapper

class FaMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.autoregressive = cfg.autoregressive
        self.n_aatype = cfg.n_aatype
        self.seq_emb_dim = cfg.n_channel
        self.use_self_conditioning_seq = cfg.use_self_conditioning_seq
        self.model_type = cfg.model_type
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.augment_eps = cfg.augment_eps
        self.no_aatype_pred = getattr(cfg, "no_aatype_pred", False)
        self.features = ProteinFeatures(self.node_features, self.edge_features, top_k=self.k_neighbors, augment_eps=self.augment_eps)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_s = nn.Embedding(self.n_aatype, self.hidden_dim)        
        self.dropout = nn.Dropout(cfg.dropout_p)
        self.decoder_in = self.hidden_dim * 3
        self.aggregation = cfg.aggregation
        self.return_embedding = cfg.return_embedding
        self.max_nn = max(cfg.graph_transformer.nns)
        self.pos_enc_size = rc.r_max * 4 + rc.s_max * 2 + 5
        self.dim_pos_enc = cfg.graph_transformer.dim_pos_enc
        self.num_heads = cfg.graph_transformer.num_heads
        self.pos_enc = cfg.graph_transformer.pos_enc
        self.attn_bias = cfg.graph_transformer.attn_bias
        self.use_esm = cfg.use_esm

        if self.model_type not in ['graph_transformer', 'sidechain', 'baseline']:
            raise ValueError(f'Incorrect model type specified: {self.model_type}, must be one of: graph_transformer, sidechain, or baseline!')
        
        if self.autoregressive and self.model_type in ['graph_transformer']:
            raise ValueError(f'Autoregressive training not implemented for model type: {self.model_type}')

        if self.model_type in ['graph_transformer', 'sidechain']:
            self.decoder_in += self.hidden_dim
            self.sidechain_features = SidechainProteinFeatures(autoregressive = self.autoregressive,
                                                              node_features = self.node_features, 
                                                              edge_features = self.edge_features, 
                                                              top_k=self.k_neighbors, 
                                                              augment_eps=self.augment_eps,)
            self.W_e2 = nn.Linear(self.edge_features, self.hidden_dim, bias=True)

        if self.model_type in ['graph_transformer']:
            self.atom_decoder_in = self.decoder_in + self.hidden_dim
            self.gt = GraphTransformer(cfg.graph_transformer)
            self.embed_pos = nn.Linear(self.pos_enc_size, self.dim_pos_enc, bias=True)
            self.proj_attn_bias = nn.Linear(self.pos_enc_size, self.num_heads, bias=True)

            self.atom_decoder_layers = nn.ModuleList([
                DecLayer(self.hidden_dim, self.atom_decoder_in, dropout=cfg.dropout_p)
                for _ in range(self.num_decoder_layers)
            ])

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

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_aatype, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_gt_inputs(
        self,
        X, 
        atom14_mask, 
        aatype_noised,
        seq_mask, 
        chain_encoding
    ):
        #get one hot encoding of atom identities
        atom_indices = get_rc_tensor(rc.RESTYPE_TO_ATOM37_IDX, aatype_noised[seq_mask == 1].flatten())
        atom_indices_packed = atom_indices[atom_indices != -1].flatten()
        q = F.one_hot(atom_indices_packed, num_classes=len(rc.atom_types)).float()
        tot_num_atoms = len(atom_indices_packed)

        #packing X and getting packed mask
        atoms14_mask_no_pad = atom14_mask * seq_mask[:,:, None]
        atom14_mask_packed_no_pad = (atom14_mask[seq_mask == 1] == 1)
        chain_encoding_packed_no_pad = chain_encoding[seq_mask == 1].flatten()

        unmasked_packed_X = pack(
                unpacked_rep = X[seq_mask == 1, ...],
                tgt_shape = (-1, 3),
                mask = atom14_mask_packed_no_pad
            )

        num_atoms_per_example = atoms14_mask_no_pad.sum(dim =(1,2)).long()
        num_residues_per_example = seq_mask.sum(dim = -1).long()
        num_atoms_per_residue = atom14_mask_packed_no_pad.sum(dim = -1).long().to(X.device)
        tot_num_residues = len(num_atoms_per_residue)

        batch_atom_start_idx = torch.cat((torch.tensor([0], device=q.device), num_atoms_per_example[:-1])).cumsum(dim=0)
        batch_residue_start_idx = torch.cat((torch.tensor([0], device=q.device), num_residues_per_example[:-1])).cumsum(dim=0)

        ids_topk = torch.zeros((tot_num_atoms, self.max_nn), dtype=torch.long, device=q.device)
        positional_enc_topk = torch.zeros((tot_num_atoms, self.max_nn, 137), dtype=q.dtype, device=q.device) if (self.pos_enc or self.attn_bias) else None

        # Process each batch example
        for atom_start_idx, residue_start_idx, na, nr in zip(batch_atom_start_idx, batch_residue_start_idx, num_atoms_per_example, num_residues_per_example):
            # Extract packed_X and compute ids_topk
            start_a, end_a = int(atom_start_idx), int(atom_start_idx + na)
            start_r, end_r = int(residue_start_idx), int(residue_start_idx + nr)

            packed_X_i = unmasked_packed_X[start_a: end_a, :]
            ids_topk_i = extract_ids_topk(packed_X_i, num_nn = self.max_nn)

            if (self.pos_enc or self.attn_bias):
                num_atoms_per_residue_i = num_atoms_per_residue[start_r: end_r]
                chain_encoding_i = chain_encoding_packed_no_pad[start_r: end_r]

                atom_residue_idx_i = torch.repeat_interleave(
                    torch.arange(nr).to(q.device),
                    num_atoms_per_residue_i
                )

                atom_chain_enc_i = torch.repeat_interleave(
                    chain_encoding_i,
                    num_atoms_per_residue_i
                )  

                same_res = torch.eq(atom_residue_idx_i.unsqueeze(0), atom_residue_idx_i.unsqueeze(1))
                same_chain = torch.eq(atom_chain_enc_i.unsqueeze(0), atom_chain_enc_i.unsqueeze(1))
                atom_idx_i = torch.arange(na).to(q.device)

                d_atom = torch.clamp(atom_idx_i.unsqueeze(0) - atom_idx_i.unsqueeze(1) + rc.r_max, min = 0, max = 2 * rc.r_max)
                d_atom[~(same_chain|same_res)] = 2 * rc.r_max + 1
                rel_atom_enc = F.one_hot(d_atom, num_classes = 2 * rc.r_max + 2)

                d_chain = torch.clamp(atom_chain_enc_i.unsqueeze(0) - atom_chain_enc_i.unsqueeze(1) + rc.s_max, min = 0, max = 2 * rc.s_max)
                rel_chain_enc = F.one_hot(d_chain, num_classes = 2 * rc.s_max + 1)

                d_res = torch.clamp(atom_residue_idx_i.unsqueeze(0) - atom_residue_idx_i.unsqueeze(1) + rc.r_max, min = 0, max = 2 * rc.r_max)
                d_res[~same_chain] = 2 * rc.r_max + 1
                rel_res_enc = F.one_hot(d_res, num_classes = 2 * rc.r_max + 2)

                positional_enc_i = torch.cat([rel_res_enc, rel_atom_enc, rel_chain_enc], dim = -1)

                positional_enc_topk_i = gather_pos_enc(ids_topk_i, positional_enc_i)
                positional_enc_topk[start_a: end_a, :, :] = positional_enc_topk_i

            # fill ids_topk and positional_enc_topk for entire batch with current example
            ids_topk[start_a: end_a, :] = ids_topk_i + start_a + 1
            
        return q, ids_topk, unmasked_packed_X, num_atoms_per_residue, positional_enc_topk, tot_num_atoms

    def aggregate(
        self, 
        h_A, 
        num_atoms_per_residue
    ):

        # Calculate the total number of residues
        num_residues = num_atoms_per_residue.size(0)

        # Generate residue indices that map each atom to its corresponding residue
        residue_indices = (
            torch.arange(num_residues, device=h_A.device)
            .repeat_interleave(num_atoms_per_residue)
            .unsqueeze(-1)
            .expand_as(h_A)
        )

        # Initialize the aggregated residue tensor
        h_R = torch.zeros(num_residues, self.hidden_dim, dtype=h_A.dtype, device=h_A.device)

        # Aggregate atom features to residue-level using scatter_reduce
        h_R.scatter_reduce(src=h_A, dim=0, index=residue_indices, reduce=self.aggregation)

        return h_R


    def forward(
        self,
        denoised_coords: TensorType["b n a x", float],
        aatype_noised: TensorType["b n", int], #will have UNK tokens where masking occurs
        seq_self_cond: Optional[TensorType["b n k", float]],  # logits
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        chain_encoding: TensorType["b n", int],
        mlm_mask: TensorType["b n", bool],
    ):

        B, N, _, _ = denoised_coords.shape
        S = aatype_noised

        #prepare inputs for protein mpnn
        X, atom14_mask = atom37_to_atom14(aatype_noised, denoised_coords)

        # Prepare node and edge embeddings
        E, E_idx, X = self.features(X, seq_mask, residue_index, chain_encoding)

        #save noised version of X_bb with augment eps for sidechain packing
        X_bb = X[:,:,rc.atom14_bb_idxs, :]

        #h_V is size [B,N,H]
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(seq_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = seq_mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, seq_mask, mask_attend)

        #keep copy of node embeddings from encoder
        h_V_enc = h_V.clone()
        
        # Concatenate self-conditioning
        if self.use_self_conditioning_seq:
            if seq_self_cond is None:
                S_self_cond = torch.zeros_like(S)
            else:
                # One-hot encode the argmax prediction
                S_self_cond = F.one_hot(seq_self_cond.argmax(dim=-1), self.n_aatype)
            S = torch.cat([S, seq_self_cond], dim=-1)

        mask_size = E_idx.shape[1]

        #implementation of causal mask if training autoregressively, otherwise mask is fully true
        if self.autoregressive:
            decoding_order = torch.argsort((seq_mask+0.0001)*(torch.abs(torch.randn(seq_mask.shape, device=seq_mask.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=seq_mask.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        else:
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

        if self.model_type in ['graph_transformer','sidechain']:

            # Add empty hidden dim of 128 to end of h_EXV to later sum with added sidechain distance information
            h_EX_encoder = cat_neighbors_nodes(torch.zeros((B, N, self.hidden_dim), device = h_S.device), h_EX_encoder, E_idx)
        
            # Extract sidechain features and concatenate to edge embeddings
            E2, _ = self.sidechain_features(X, seq_mask, residue_index, chain_encoding, E_idx, atom14_mask)
            
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
    
        if self.model_type in ['graph_transformer']:
            
            # Add empty hidden dim of 128 to end of h_EXV to later sum with added atom information
            h_EX_encoder = cat_neighbors_nodes(torch.zeros((B, N, self.hidden_dim), device = h_S.device), h_EX_encoder, E_idx)

            #get graph transformer inputs
            q, ids_topk, unmasked_packed_X, num_atoms_per_residue, positional_enc, num_atoms = self.get_gt_inputs(X, atom14_mask, aatype_noised, seq_mask, chain_encoding)
            
            #embed full atomic positional encoding into edge embedding and attention bias
            p_A = self.embed_pos(positional_enc).squeeze(-1) if self.pos_enc else None
            attn_bias = self.proj_attn_bias(positional_enc).view(num_atoms, self.max_nn, self.num_heads).permute(0,2,1) if self.attn_bias else None
            
            #graph transformer forward pass, garbage collect inputs
            h_A = self.gt(unmasked_packed_X, ids_topk, q, p_A, attn_bias)
            del q, ids_topk, unmasked_packed_X

            #aggergate atom embeddings into residue embeddings
            h_R = self.aggregate(h_A, num_atoms_per_residue)
            del h_A

            #unpack residue embeddings, (B N, ...) -> (B, N, ...)
            h_R = unpack(
                packed_rep = h_R,
                tgt_shape = (B, N, self.hidden_dim),
                mask = (seq_mask == 1)
            )
            
            #concatenate residue embedding to sequence embedding
            h_ESVR = cat_neighbors_nodes(h_R, h_ESV, E_idx)

            for layer in self.atom_decoder_layers:
                h_V, h_ESVR = layer(h_V, h_ESVR, seq_mask, E_idx)

        logits = self.W_out(h_V)

        if self.no_aatype_pred:
            logits = None 
        
        if self.return_embedding == 'encoder':
            return logits, h_V_enc, X_bb
        elif self.return_embedding == 'decoder':
            return logits, h_V_dec, X_bb
        elif self.return_embedding == 'last':
            return logits, h_V, X_bb
        else:
            raise ValueError(f'Incorrect return embedding type specified: {self.return_embedding}, must be one of: encoder, decoder, or last!')

    def sample(self, X, S_true, chain_encoding_all, residue_idx, mask, temperature=1.0, chain_mask = None, chain_M_pos=None):
        device = X.device

        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = mask #TODO: update for multi-chain sampling
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(torch.randn(chain_mask.shape, device=chain_mask.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits, dim=-1)
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        return S, all_probs

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


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
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

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
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
        return E, E_idx, X

class SidechainProteinFeatures(nn.Module):
    def __init__(self, autoregressive, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.,):
        """ Extract protein features """
        super(SidechainProteinFeatures, self).__init__()
        self.autoregressive = autoregressive
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 4 * 10
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

    def _get_rbf(self, A, B, E_idx, atom_mask):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)

        #if the model is autoregressive, we cannot a lot a residue to see it's own sidechains!
        if self.autoregressive:
            atom_indices = torch.arange(E_idx.shape[1]).view(1, E_idx.shape[1], 1).expand(E_idx.shape).to(E_idx.device)
            self_edges = E_idx == atom_indices
            RBF_A_B[self_edges] = 0.

        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels, E_idx, atom_mask):
        max_atoms = X.shape[-2] #14
        
        N = X[:,:,0,:]
        Ca = X[:,:,1,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        RBF_all = []        
        for bb_atom in [Ca, N, C, O]:
            for non_bb_atom_pos in range(rc.num_bb_atoms, max_atoms):
                    non_bb_atom_mask = atom_mask[:,:,non_bb_atom_pos]
                    RBF_all.append(self._get_rbf(bb_atom, X[:, :, non_bb_atom_pos, :],  E_idx, non_bb_atom_mask))

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
