# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Portions of this file were adapted from the open source code for the following
# two papers:
#
#   Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative
#   models for graph-based protein design. Advances in Neural Information
#   Processing Systems, 32.
#
#   Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2020).
#   Learning from Protein Structure with Geometric Vector Perceptrons. In
#   International Conference on Learning Representations.
#
# MIT License
# 
# Copyright (c) 2020 Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael Townshend, Ron Dror
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ================================================================
# The below license applies to the portions of the code (parts of 
# src/datasets.py and src/models.py) adapted from Ingraham, et al.
# ================================================================
# 
# MIT License
# 
# Copyright (c) 2019 John Ingraham, Vikas Garg, Regina Barzilay, Tommi Jaakkola
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allatom_design.model.seq_denoiser.denoisers.seq_design.gvp.gvp_modules import GVP, LayerNorm
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (  
    get_rc_tensor,
    orientations,
    dihedrals,
    sidechains,
    positional_embeddings,
    normalize,
    rbf,
    nan_to_num,
    dist
)

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features


class GVPGraphEmbedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.top_k_neighbors = cfg.top_k_neighbors
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.remove_edges_without_coords = cfg.remove_edges_without_coords
        node_input_dim = (6, 15) #changed from (7,3) because we no longer condition on coords mask, and we add seq + 20
        edge_input_dim = (32, 56) #changed from (7,3) because we no longer condition on coords_mask i, coords_mask j
        node_hidden_dim = (cfg.node_hidden_dim_scalar,
                cfg.node_hidden_dim_vector)
        edge_hidden_dim = (cfg.edge_hidden_dim_scalar,
                cfg.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )

        self.embed_mpnn_node = nn.Linear(cfg.node_hidden_dim_scalar, cfg.node_hidden_dim_scalar)
        self.embed_mpnn_edge = nn.Linear(cfg.edge_hidden_dim_scalar, cfg.edge_hidden_dim_scalar)
        self.zero_ghost_atoms = False

    def forward(self, coords, seq, mpnn_E_idx, mpnn_node_embedding, mpnn_edge_embedding, padding_mask, atom14_mask):
        with torch.no_grad():
            node_features = self.get_node_features(coords, padding_mask, atom14_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, padding_mask, mpnn_E_idx, atom14_mask)

        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)
        
        node_embeddings = (
            node_embeddings_scalar + self.embed_mpnn_node(mpnn_node_embedding),
            node_embeddings_vector
        )

        edge_embeddings = (
            edge_embeddings[0] + self.embed_mpnn_edge(mpnn_edge_embedding.flatten(1,2)),
            edge_embeddings[1]
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)

        return node_embeddings, edge_embeddings, edge_index

    def get_atom_type(self, seq, atom14_mask):
        atom_indices = get_rc_tensor(rc.RESTYPE_TO_ATOM37_IDX, seq)
        atom_indices = torch.where(atom_indices == -1, 0, atom_indices) #temporaily set ghost atom idx to 0
        atom_indices_one_hot = F.one_hot(atom_indices, num_classes=rc.atom_type_num).float()
        atom_indices_one_hot *= atom14_mask[..., None].expand_as(atom_indices_one_hot)
        atom_types_summed = torch.sum(atom_indices_one_hot, dim = -2)
        return atom_types_summed
    
    def get_node_features(self, coords, padding_mask, atom14_mask):
        # scalar features
        node_scalar_features = dihedrals(coords)

        # vector features
        X_ca = coords[:, :, 1]
        ca_orientations = orientations(X_ca)
        fa_orientations = self.intra_residue_orientations(coords, atom14_mask)

        #for residues w/out CB, overwrite with pseudo CB
        cb_orientations = sidechains(coords)
        no_cb_mask = atom14_mask[:, :, 4] == 0 #use atom14 mask to find positions with no cb
        no_cb_mask = torch.where(padding_mask, False, no_cb_mask) #exclude padded positions from getting pseudo cb
        fa_orientations[:,:,3][no_cb_mask, :] = cb_orientations[no_cb_mask]

        node_vector_features = torch.cat([ca_orientations, fa_orientations], dim=-2)
        return node_scalar_features, node_vector_features

    def intra_residue_orientations(self, coords, atom14_mask):
        X_ca = coords[:, :, 1]
        vectors = []
        atom_positions = [0,2,3,4,5,6,7,8,9,10,11,12,13]

        for atom_pos in atom_positions:
            atom_pos_mask = atom14_mask[:, :, atom_pos][:,:,None].expand(-1, -1, 3)
            intra_residue_vector = normalize(X_ca - coords[:, :, atom_pos])

            #set unit vector for missing atoms to 0
            if self.zero_ghost_atoms:
                intra_residue_vector = torch.where(atom_pos_mask == 1, intra_residue_vector, 0)
            vectors.append(intra_residue_vector)

        return torch.stack(vectors, dim=2)
    
    def get_edge_features(self, coords, padding_mask, E_idx, atom14_mask):
        X_ca = coords[:, :, 1]

        # Get distances to the top k neighbors, using E_idx from ProteinMPNN
        E_dist, E_residue_mask = dist(X_ca, E_idx, padding_mask)

        # Flatten the graph to be batch size 1 for torch_geometric package 
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF 
        pos_embeddings = positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)
        
        # Calculate relative orientation 
        E_vectors = self.get_edge_vectors(coords, E_idx, edge_index, B, L, k, atom14_mask)

        # Normalize and remove nans 
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        edge_index[:, ~E_residue_mask] = -1

        return (edge_s, edge_v), edge_index.transpose(0, 1) 
    
    def get_edge_vectors(self, coords, E_idx, edge_index, B, L, k, atom14_mask):
        max_atoms = coords.shape[-2] #14
        X_n = coords[:, :, 0]
        X_ca = coords[:, :, 1]
        X_c = coords[:, :, 2]
        X_o = coords[:, :, 3]
        
        vectors = []
        
        for bb_atom in [X_ca, X_n, X_c, X_o]:
            for atom_pos in range(max_atoms):
                atom_mask_pos = atom14_mask[:,:,atom_pos]
                atom_mask_neighbors = torch.gather(atom_mask_pos[...,None].expand(-1,-1,k), 1, E_idx)
                relative_orientation_vector = normalize(self.get_relative_orientation(bb_atom, coords[:,:,atom_pos], edge_index, B, L, k))

                #insert 0 for unit vectors where destination atom does not exist, if specified
                if self.zero_ghost_atoms:
                    relative_orientation_vector = torch.where(atom_mask_neighbors[...,None].expand(-1,-1,-1,3) == 1, relative_orientation_vector, 0).flatten(1, 2)
                vectors.append(relative_orientation_vector)
        
        return torch.stack(vectors, dim=2)

    def get_relative_orientation(self, X, Y, edge_index, B, L, k):
        X_src = X.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            Y,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )

        return X_src - X_dest

##UTILS

def flatten_graph(node_embeddings, edge_embeddings, edge_index):
    """
    Flattens the graph into a batch size one (with disconnected subgraphs for
    each example) to be compatible with pytorch-geometric package.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch size x edges x edge_embed_dim
                - vector: shape batch size x edges x edge_embed_dim x 3
        edge_index: shape batch_size x 2 (source node and target node) x edges
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        edge_embeddings: edge embeddings of in tuple form (scalar, vector)
                - scalar: shape batch total_edges x edge_embed_dim
                - vector: shape batch total_edges x edge_embed_dim x 3
        edge_index: shape 2 x total_edges
    """
    x_s, x_v = node_embeddings
    e_s, e_v = edge_embeddings
    batch_size, N = x_s.shape[0], x_s.shape[1]
    node_embeddings = (torch.flatten(x_s, 0, 1), torch.flatten(x_v, 0, 1))
    edge_embeddings = (torch.flatten(e_s, 0, 1), torch.flatten(e_v, 0, 1))

    edge_mask = torch.any(edge_index != -1, dim=1)
    # Re-number the nodes by adding batch_idx * N to each batch
    edge_index = edge_index + (torch.arange(batch_size, device=edge_index.device) *
            N).unsqueeze(-1).unsqueeze(-1)
    edge_index = edge_index.permute(1, 0, 2).flatten(1, 2)
    edge_mask = edge_mask.flatten()
    edge_index = edge_index[:, edge_mask] 
    edge_embeddings = (
        edge_embeddings[0][edge_mask, :],
        edge_embeddings[1][edge_mask, :]
    )
    return node_embeddings, edge_embeddings, edge_index 


def unflatten_graph(node_embeddings, batch_size):
    """
    Unflattens node embeddings.
    Args:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch total_nodes x node_embed_dim
                - vector: shape batch total_nodes x node_embed_dim x 3
        batch_size: int
    Returns:
        node_embeddings: node embeddings in tuple form (scalar, vector)
                - scalar: shape batch size x nodes x node_embed_dim
                - vector: shape batch size x nodes x node_embed_dim x 3
    """
    x_s, x_v = node_embeddings
    x_s = x_s.reshape(batch_size, -1, x_s.shape[1])
    x_v = x_v.reshape(batch_size, -1, x_v.shape[1], x_v.shape[2])
    return (x_s, x_v)
