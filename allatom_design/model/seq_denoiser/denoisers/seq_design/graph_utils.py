# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small graph tensor helpers used by the sequence-design Potts layer.

Adapted from Chroma's graph utilities so sequence design no longer depends on
the root-level ``chroma`` package.
"""

from typing import Tuple

import torch


def collect_neighbors(node_h: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Collect neighbor node features as edge features."""
    num_batch, num_nodes, num_neighbors = edge_idx.shape
    num_features = node_h.shape[2]

    idx_flat = edge_idx.reshape([num_batch, num_nodes * num_neighbors, 1])
    idx_flat = idx_flat.expand(-1, -1, num_features)
    neighbor_h = torch.gather(node_h, 1, idx_flat)
    return neighbor_h.reshape((num_batch, num_nodes, num_neighbors, num_features))


def collect_edges_transpose(
    edge_h: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_ij: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect edge embeddings for reversed edges at each directed edge."""
    num_batch, num_residues, num_k, num_features = list(edge_h.size())

    ij_to_ji, mask_ji = transpose_edge_idx(edge_idx, mask_ij)

    edge_h_flat = edge_h.reshape(num_batch, num_residues * num_k, -1)
    ij_to_ji = ij_to_ji.unsqueeze(-1).expand(-1, -1, num_features)
    edge_h_transpose = torch.gather(edge_h_flat, 1, ij_to_ji)
    edge_h_transpose = edge_h_transpose.reshape(
        num_batch,
        num_residues,
        num_k,
        num_features,
    )
    edge_h_transpose = mask_ji.unsqueeze(-1) * edge_h_transpose
    return edge_h_transpose, mask_ji


def transpose_edge_idx(
    edge_idx: torch.LongTensor,
    mask_ij: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Map each directed edge index to its reverse edge index when present."""
    num_batch, num_residues, num_k = list(edge_idx.size())

    edge_idx_flat = edge_idx.reshape([num_batch, num_residues * num_k, 1]).expand(
        -1,
        -1,
        num_k,
    )
    edge_idx_neighbors = torch.gather(edge_idx, 1, edge_idx_flat)
    edge_idx_neighbors = edge_idx_neighbors.reshape(
        [num_batch, num_residues, num_k, num_k]
    )

    residue_i = torch.arange(num_residues, device=edge_idx.device).reshape(
        (1, -1, 1, 1)
    )
    edge_idx_match = (edge_idx_neighbors == residue_i).type(torch.float32)
    return_mask, return_idx = torch.max(edge_idx_match, -1)

    ij_to_ji = edge_idx * num_k + return_idx
    ij_to_ji = ij_to_ji.reshape(num_batch, -1)

    mask_ji = torch.gather(mask_ij.reshape(num_batch, -1), -1, ij_to_ji)
    mask_ji = mask_ji.reshape(num_batch, num_residues, num_k)
    mask_ji = mask_ij * return_mask * mask_ji
    return ij_to_ji, mask_ji
