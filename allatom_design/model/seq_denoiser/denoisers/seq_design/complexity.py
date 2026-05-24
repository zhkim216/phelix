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

"""Local sequence-complexity regularization helpers.

Adapted from Chroma's complexity utilities so AtomMPNN sampling no longer
depends on the root-level ``chroma`` package.
"""

import numpy as np
import torch
import torch.nn.functional as F

from allatom_design.model.seq_denoiser.denoisers.seq_design.graph_utils import (
    collect_neighbors,
)

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def compositions(S: torch.Tensor, C: torch.LongTensor, w: int = 30):
    """Compute local residue compositions in same-chain sequence windows."""
    q = len(AA20)
    mask_i = (C > 0).float()
    if len(S.shape) == 2:
        S = F.one_hot(S, q)

    s_onehot = mask_i[..., None] * S
    kx = torch.arange(w, device=S.device) - w // 2
    edge_idx = (
        torch.arange(S.shape[1], device=S.device)[None, :, None] + kx[None, None, :]
    )
    edge_idx = edge_idx.expand(S.shape[0], -1, -1)

    mask_ij = (edge_idx > 0) & (edge_idx < S.shape[1])
    edge_idx = edge_idx.clamp(min=0, max=S.shape[1] - 1)
    c_i = C[..., None]
    c_j = collect_neighbors(c_i, edge_idx)[..., 0]
    mask_ij = (mask_ij & c_j.eq(c_i) & (c_i > 0) & (c_j > 0)).float()

    s_j = mask_ij[..., None] * collect_neighbors(s_onehot, edge_idx)
    n = s_j.sum(2)

    num_n = n.sum(-1, keepdims=True)
    p = n / (num_n + 1e-5)
    mask_i = ((num_n[..., 0] > 0) & (C > 0)).float()
    mask_ij = mask_i[..., None] * mask_ij
    return p, n, edge_idx, mask_i, mask_ij


def complexity_lcp(
    S: torch.LongTensor,
    C: torch.LongTensor,
    w: int = 30,
    entropy_min: float = 2.32,
    method: str = "naive",
    differentiable=True,
    eps: float = 1e-5,
    min_coverage=0.9,
) -> torch.Tensor:
    """Compute the Local Composition Perplexity regularization term."""
    del eps

    if S.shape[1] < w:
        w = S.shape[1]

    _, n, edge_idx, mask_i, mask_ij = compositions(S, C, w)

    mask_coverage = n.sum(-1) > int(min_coverage * w)

    h = estimate_entropy(n, method=method)
    u = mask_coverage * (torch.exp(h) - np.exp(entropy_min)).clamp(max=0).square()

    if differentiable and len(S.shape) == 3:
        n_neighbors = collect_neighbors(n, edge_idx)
        mask_coverage_j = collect_neighbors(mask_coverage[..., None], edge_idx)
        n_ij = (n_neighbors - S[:, :, None, :])[..., None, :] + torch.eye(
            n.shape[-1],
            device=n.device,
        )[None, None, None, ...]
        n_ij = n_ij.clamp(min=0)
        h_ij = estimate_entropy(n_ij, method=method)
        u_ij = (torch.exp(h_ij) - np.exp(entropy_min)).clamp(max=0).square()

        u_ij = mask_ij[..., None] * mask_coverage_j * u_ij
        u_differentiable = (u_ij.detach() * S[:, :, None, :]).sum([-1, -2])
        u = u.detach() + u_differentiable - u_differentiable.detach()

    return (mask_i * u).sum(1)


def estimate_entropy(
    N: torch.Tensor,
    method: str = "chao-shen",
    eps: float = 1e-11,
) -> torch.Tensor:
    """Estimate entropy from count tensors."""
    N = N.float()
    n_total = N.sum(-1, keepdims=True)
    p = N / (n_total + eps)

    if method == "chao-shen":
        singletons = N.long().eq(1).sum(-1, keepdims=True).float()
        c = 1.0 - singletons / (n_total + eps)
        p_adjust = c * p
        p_inclusion = (1.0 - (1.0 - p_adjust) ** n_total).clamp(min=eps)
        h = -(p_adjust * torch.log(p_adjust.clamp(min=eps)) / p_inclusion).sum(-1)
    elif method == "miller-maddow":
        bins = (N > 0).float().sum(-1)
        bias = (bins - 1) / (2 * n_total[..., 0] + eps)
        h = -(p * torch.log(p + eps)).sum(-1) + bias
    elif method == "laplace":
        N = N.float() + 1 / N.shape[-1]
        n_total = N.sum(-1, keepdims=True)
        p = N / (n_total + eps)
        h = -(p * torch.log(p)).sum(-1)
    else:
        h = -(p * torch.log(p + eps)).sum(-1)
    return h
