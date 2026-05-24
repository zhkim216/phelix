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

"""Layers for building Potts models.

This module contains layers for parameterizing Potts models from
graph embeddings.

Adapted from Chroma by Richard Shuai.
"""

from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from tqdm.auto import tqdm

from allatom_design.model.seq_denoiser.denoisers.seq_design import \
    graph_utils as graph


class GraphPotts(nn.Module):
    """Conditional Random Field (conditional Potts model) layer on a graph.

    Arguments:
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        num_states (int): Size of the vocabulary.
        parameterization (str): Parameterization choice in
            `{'linear', 'factor', 'factor_scale', 'score', 'score_zsum', 'score_scale'}`,
            or any of those suffixed with `_beta`, which will add in a globally
            learnable temperature scaling parameter.
        symmetric_J (bool): If True enforce symmetry of Potts model i.e.
            `J_ij(s_i, s_j) = J_ji(s_j, s_i)`.
        init_scale (float): Scale factor for the weights and couplings at
            initialization.
        dropout (float): Probability of per-dimension dropout on `[0,1]`.
        num_factors (int): Number of factors to use for the `factor`
            parameterization mode.
        beta_init (float): Initial temperature scaling factor for parameterizations
            with the `_beta` suffix.

    Inputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Outputs:
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        num_states: int,
        parameterization: str = "score",
        symmetric_J: bool = True,
        init_scale: float = 0.1,
        dropout: float = 0.0,
        num_factors: Optional[int] = None,
        beta_init: float = 10.0,
    ):
        super(GraphPotts, self).__init__()
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_states = num_states


        # Beta parameterization support temperature learning
        self.scale_beta = False
        if parameterization.endswith("_beta"):
            parameterization = parameterization.split("_beta")[0]
            self.scale_beta = True
            self.log_beta = nn.Parameter(np.log(beta_init) * torch.ones(1))

        self.init_scale = init_scale
        self.parameterization = parameterization
        self.symmetric_J = symmetric_J
        if self.parameterization == "linear":
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
        elif self.parameterization == "factor":
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
        elif self.parameterization == "score":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h_bg = nn.Linear(self.dim_nodes, 1)
            self.W_J_bg = nn.Linear(self.dim_edges, 1)
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(
                self.dim_edges, self.num_states * num_factors, bias=True
            )
            self.W_J_right = nn.Linear(
                self.dim_edges, self.num_states * num_factors, bias=True
            )
        elif self.parameterization == "score_zsum":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(
                self.dim_edges, self.num_states * num_factors, bias=True
            )
            self.W_J_right = nn.Linear(
                self.dim_edges, self.num_states * num_factors, bias=True
            )
        elif self.parameterization == "score_scale":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.W_h_bg = nn.Linear(self.dim_nodes, 1)
            self.W_J_bg = nn.Linear(self.dim_edges, 1)
            self.W_h_log_scale = nn.Linear(self.dim_nodes, 1)
            self.W_J_log_scale = nn.Linear(self.dim_edges, 1)
            self.W_h = nn.Linear(self.dim_nodes, self.num_states)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states * num_factors)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states * num_factors)
        elif self.parameterization == "factor_scale":
            # factor parameterization + per-token/per-edge learnable log-scale (no background).
            self.W_h_log_scale = nn.Linear(self.dim_nodes, 1)
            self.W_J_log_scale = nn.Linear(self.dim_edges, 1)
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
        else:
            print(f"Unknown potts parameterization: {parameterization}")
            raise NotImplementedError
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
    ):
        #! (JH) 260131Note
        # edge_idx: E_idx between only protein tokens in protein chains,
        # mask_i: protein_residue_node_mask, mask_ij: protein_residue_edge_mask_2d
        mask_J = _mask_J(edge_idx, mask_i, mask_ij)

        if self.parameterization == "linear":
            # Compute site params (h) from node embeddings
            # Compute coupling params (J) from edge embeddings
            scale = torch.exp(self.log_scale)
            h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
            J = scale * mask_J.unsqueeze(-1) * self.W_J(edge_h)
            J = J.view(list(edge_h.size())[:3] + ([self.num_states] * 2))
        elif self.parameterization == "factor":
            scale = torch.exp(self.log_scale)
            h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
            mask_J = scale * mask_J.unsqueeze(-1)
            shape_J = list(edge_h.size())[:3] + ([self.num_states] * 2)
            J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J)
            J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J)
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)
            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )
        elif self.parameterization == "score":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            scale = torch.exp(self.log_scale)
            mask_h = scale * mask_i.unsqueeze(-1)
            mask_J = scale * mask_J.unsqueeze(-1)
            h = mask_h * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J * self.W_J_left(edge_h)).view(
                shape_J_prefix + [self.num_states, self.num_factors]
            )
            J_right = (mask_J * self.W_J_right(edge_h)).view(
                shape_J_prefix + [self.num_factors, self.num_states]
            )
            J = torch.matmul(J_left, J_right)

            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )

            # Background components
            h = h + mask_h * self.W_h_bg(node_h)
            J = J + (mask_J * self.W_J_bg(edge_h)).unsqueeze(-1)
        elif self.parameterization == "score_zsum":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            scale = torch.exp(self.log_scale)
            mask_h_scale = scale * mask_i.unsqueeze(-1)
            mask_J_scale = scale * mask_J.unsqueeze(-1)
            h = mask_h_scale * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J_scale * self.W_J_left(edge_h)).view(
                shape_J_prefix + [self.num_states, self.num_factors]
            )
            J_right = (mask_J_scale * self.W_J_right(edge_h)).view(
                shape_J_prefix + [self.num_factors, self.num_states]
            )
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)

            # Zero-sum gauge
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )

            # Subtract off J background average
            mask_J = mask_J.view(list(mask_J.size()) + [1, 1])
            J_i_avg = J.sum(dim=[1, 2], keepdim=True) / mask_J.sum([1, 2], keepdim=True)
            J = mask_J * (J - J_i_avg)
        elif self.parameterization == "score_scale":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            mask_h = mask_i.unsqueeze(-1)
            mask_J = mask_J.unsqueeze(-1)
            h = mask_h * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J * self.W_J_left(edge_h)).view(
                shape_J_prefix + [self.num_states, self.num_factors]
            )
            J_right = (mask_J * self.W_J_right(edge_h)).view(
                shape_J_prefix + [self.num_factors, self.num_states]
            )
            J = torch.matmul(J_left, J_right)

            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )

            # Background components
            log_scale = np.log(self.init_scale)
            h_scale = torch.exp(self.W_h_log_scale(node_h) + log_scale)
            J_scale = torch.exp(self.W_J_log_scale(edge_h) + 2 * log_scale).unsqueeze(
                -1
            )
            h_bg = mask_h * self.W_h_bg(node_h)
            J_bg = (mask_J * self.W_J_bg(edge_h)).unsqueeze(-1)
            h = h_scale * (h + h_bg)
            J = J_scale * (J + J_bg)
        elif self.parameterization == "factor_scale":
            # factor-style bilinear J and unary h with per-token/per-edge
            # learnable log-scale (no background). The 2x offset on the J
            # log-scale preserves the scale^1 vs scale^2 asymmetry of the
            # plain `factor` mode at initialization.
            mask_h_raw = mask_i.unsqueeze(-1)
            mask_J_raw = mask_J.unsqueeze(-1)
            h = mask_h_raw * self.W_h(node_h)
            shape_J = list(edge_h.size())[:3] + ([self.num_states] * 2)
            J_left = (mask_J_raw * self.W_J_left(edge_h)).view(shape_J)
            J_right = (mask_J_raw * self.W_J_right(edge_h)).view(shape_J)
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)

            # Zero-sum gauge (matches factor)
            h = h - h.mean(-1, keepdim=True)
            J = (
                J
                - J.mean(-1, keepdim=True)
                - J.mean(-2, keepdim=True)
                + J.mean(dim=[-1, -2], keepdim=True)
            )

            # Per-token / per-edge learnable log-scale with init_scale offset.
            log_scale_offset = np.log(self.init_scale)
            h_scale = torch.exp(self.W_h_log_scale(node_h) + log_scale_offset)
            J_scale = torch.exp(
                self.W_J_log_scale(edge_h) + 2 * log_scale_offset
            ).unsqueeze(-1)
            h = h_scale * h
            J = J_scale * J

        if self.symmetric_J:
            J = self._symmetrize_J(J, edge_idx, mask_ij)

        if self.scale_beta:
            beta = torch.exp(self.log_beta)
            h = beta * h
            J = beta * J
        return h, J

    def _symmetrize_J_serial(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices, serial version."""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Symmetrization based on raw indexing - extremely slow; for debugging
        import time

        _start = time.time()
        J_symm = torch.zeros_like(J)
        for b in range(J.size(0)):
            for i in range(J.size(1)):
                for k_i in range(J.size(2)):
                    for k_j in range(J.size(2)):
                        j = edge_idx[b, i, k_i]
                        if edge_idx[b, j, k_j] == i:
                            J_symm[b, i, k_i, :, :] = (
                                J[b, i, k_i, :, :]
                                + J[b, j, k_j, :, :].transpose(-1, -2)
                            ) / 2.0
        speed = J.size(0) * J.size(1) / (time.time() - _start)
        print(f"symmetrized at {speed} residue/s")
        return J_symm

    def _symmetrize_J(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices via adding J_ij + J_ji^T"""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Flatten and gather J_ji matrices using transpose indexing
        J_flat = J.view(num_batch, num_residues, num_k, -1)
        J_flat_transpose, mask_ji = graph.collect_edges_transpose(
            J_flat, edge_idx, mask_ij
        )
        J_transpose = J_flat_transpose.view(
            num_batch, num_residues, num_k, num_states, num_states
        )
        # Transpose J_ji matrices to symmetrize as (J_ij + J_ji^T)/2
        J_transpose = J_transpose.transpose(-2, -1)
        mask_ji = (0.5 * mask_ji).view(num_batch, num_residues, num_k, 1, 1)
        J_symm = mask_ji * (J + J_transpose)
        return J_symm

    def energy(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts model energy from sequence.

        Inputs:
            S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            U (torch.Tensor): Potts total energies with shape `(num_batch)`.
                Lower energies are more favorable.
        """
        # Gather J [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
        S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
        S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, self.num_states, -1)
        J_ij = torch.gather(J, -1, S_j).squeeze(-1)

        # Sum out J contributions
        J_i = J_ij.sum(2) / 2.0
        r_i = h + J_i

        U_i = torch.gather(r_i, 2, S.unsqueeze(-1))
        U = U_i.sum([1, 2])
        return U

    def pseudolikelihood(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts pseudolikelihood from sequence

        Inputs:
            S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            log_probs (torch.Tensor): Potts log-pseudolihoods with shape
                `(num_batch, num_nodes, num_states)`.
        """
        return pseudolikelihood(S, h, J, edge_idx)


    def sample(
        self,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        S: Optional[torch.LongTensor] = None,
        mask_sample: Optional[torch.Tensor] = None,
        num_sweeps: int = 100,
        temperature: float = 0.1,
        temperature_init: float = 1.0,
        penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
        differentiable_penalty: bool = True,
        rejection_step: bool = False,
        proposal: Literal["dlmc", "chromatic"] = "dlmc",
        verbose: bool = False,
        edge_idx_coloring: Optional[torch.LongTensor] = None,
        mask_ij_coloring: Optional[torch.Tensor] = None,
        symmetry_order: Optional[int] = None,
        h_uncond: Optional[torch.Tensor] = None,
        J_uncond: Optional[torch.Tensor] = None,
        edge_idx_uncond: Optional[torch.LongTensor] = None,
        gamma: float = 1.0,
        gamma_schedule_cfg: Optional[dict] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample from Potts model with Chromatic Gibbs sampling.

        Args:
            h: Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J: Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
            mask_ij (torch.Tensor): Edge mask with shape
                `(num_batch, num_nodes, num_neighbors)`.
            S (torch.LongTensor, optional): Sequence for initialization with
                shape `(num_batch, num_nodes)`.
            mask_sample (torch.Tensor, optional): Binary sampling mask indicating
                positions which are free to change with shape
                `(num_batch, num_nodes)` or which tokens are acceptable at each position
                with shape `(num_batch, num_nodes, alphabet)`.
            num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
                i.e. the depth of sampling as measured by the number of times
                every position has had an opportunity to update.
            temperature (float): Final sampling temperature.
            temperature_init (float): Initial sampling temperature, which will
                be linearly interpolated to `temperature` over the course of
                the burn in phase.
            penalty_func (Callable, optional): An optional penalty function which
                takes a sequence `S` and outputes a `(num_batch)` shaped tensor
                of energy adjustments, for example as regularization.
            differentiable_penalty (bool): If True, gradients of penalty function
                will be used to adjust the proposals.
            rejection_step (bool): If True, perform a Metropolis-Hastings
                rejection step.
            proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            verbose (bool): If True print verbose output during sampling.
            edge_idx_coloring (torch.LongerTensor, optional): Alternative
                graph dependency structure that can be provided for the
                Chromatic Gibbs algorithm when it performs initial graph
                coloring. Has shape
                    `(num_batch, num_nodes, num_neighbors_coloring)`.
            mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
                structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.
            symmetry_order (int, optional): Optional integer argument to enable
                symmetric sequence decoding under `symmetry_order`-order symmetry.
                The first `(num_nodes // symmetry_order)` states will be free to
                move, and all consecutively tiled sets of states will be locked
                to these during decoding. Internally this is accomplished by
                summing the parameters Potts model under a symmetry constraint
                into this reduced sized system and then back imputing at the end.

        Returns:
            S (torch.LongTensor): Sampled sequences with
                shape `(num_batch, num_nodes)`.
            U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower
                is more favorable.
        """
        B, N, _ = h.shape

        if symmetry_order is not None:
            if h_uncond is not None:
                raise NotImplementedError(
                    "symmetry_order and classifier-free guidance cannot be combined."
                )
            h, J, edge_idx, mask_i, mask_ij = fold_symmetry(
                symmetry_order, h, J, edge_idx, mask_i, mask_ij
            )
            S = S[:, : (N // symmetry_order)]
            if mask_sample is not None:
                mask_sample = mask_sample[:, : (N // symmetry_order)]

        S_sample, U_sample = sample_potts(
            h,
            J,
            edge_idx,
            mask_i,
            mask_ij,
            S=S,
            mask_sample=mask_sample,
            num_sweeps=num_sweeps,
            temperature=temperature,
            temperature_init=temperature_init,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
            rejection_step=rejection_step,
            proposal=proposal,
            verbose=verbose,
            edge_idx_coloring=edge_idx_coloring,
            mask_ij_coloring=mask_ij_coloring,
            h_uncond=h_uncond,
            J_uncond=J_uncond,
            edge_idx_uncond=edge_idx_uncond,
            gamma=gamma,
            gamma_schedule_cfg=gamma_schedule_cfg,
        )

        if symmetry_order is not None:
            assert N % symmetry_order == 0
            S_sample = (
                S_sample[:, None, :].expand([-1, symmetry_order, -1]).reshape([B, N])
            )
        return S_sample, U_sample


def compute_potts_energy(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    return_per_res: bool = False,
):
    """Compute Potts model energies from sequence.

    Args:
        S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        return_per_res (bool): If True, additionally return the per-residue
            contributions ``U_per_res`` of shape `(num_batch, num_nodes)` that
            sum along the node axis to ``U``.

    Returns:
        U (torch.Tensor): Potts total energies with shape `(num_batch)`.
            Lower energies are more favorable.
        U_i (torch.Tensor): Potts local conditional energies with shape
            `(num_batch, num_nodes, num_states)`.
        U_per_res (torch.Tensor, optional): Per-residue contributions with
            shape `(num_batch, num_nodes)`. Only returned when
            ``return_per_res=True``.
    """
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx) # S: [b, n] / S_j: [b, n, k, 1]
    # S_j: neighbor's state
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
    # S_j: [b, n, k, num_states, 1]. but the second last dimension is just copied num_states times
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    # J: [b, n, k, num_states, num_states]
    # J_ij: Along the last axis, select only the column indicated by S_j at each position i,
    # yielding a tensor of shape (B, N, K, Q, 1) -> (B, N, K, Q)

    # Sum out J contributions to yield local conditionals
    J_i = J_ij.sum(2) # sum over neighbors, J_i: [b, n, num_states]
    U_i = h + J_i # U_i: [b, n, num_states]

    # Per-residue contribution: h_i(S_i) + 0.5 * sum_j J_{ij}(S_i, S_j).
    # The 0.5 corrects for double counting of each edge across the two endpoints.
    U_per_res = (
        torch.gather(U_i, -1, S[..., None]).squeeze(-1)
        - 0.5 * torch.gather(J_i, -1, S[..., None]).squeeze(-1)
    )  # [b, n]
    U = U_per_res.sum(-1)  # [b]
    if return_per_res:
        return U, U_i, U_per_res
    return U, U_i


def fold_symmetry(
    symmetry_order: int,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    normalize=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold Potts model symmetrically.

    Args:
        symmetry_order (int): The order of symmetry by which to fold the Potts
            model such that the first `(num_nodes // symmetry_order)` states
            represent the entire system and all fields and couplings to and
            among other copies of this base system are collected together in
            single reduced Potts model.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        normalize (bool): If True (default), aggregate the Potts model as an average
            energy across asymmetric units instead of as a sum.

    Returns:
        h_fold (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes_folded, num_states)`, where
            `num_nodes_folded =  num_nodes // symmetry_order`.
        J_fold (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes_folded, num_neighbors, num_states, num_states)`.
        edge_idx_fold (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes_folded, num_neighbors)`.
        mask_i_fold (torch.Tensor): Node mask with shape `(num_batch, num_nodes_folded)`.
        mask_ij_fold (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes_folded, num_neighbors)`.

    """
    B, N, K, Q, _ = J.shape
    device = h.device

    N_asymmetric = N // symmetry_order
    # Fold edges by densifying the assymetric unit and averaging
    edge_idx_au = torch.remainder(edge_idx, N_asymmetric).clamp(max=N_asymmetric - 1)

    def _pairwise_fold(_T):
        # Fold-sum along neighbor dimension
        shape = list(_T.shape)
        shape[2] = N_asymmetric
        _T_au_expand = torch.zeros(shape, device=device).float()
        extra_dims = len(_T.shape) - len(edge_idx_au.shape)
        edge_idx_au_expand = edge_idx_au.reshape(
            list(edge_idx_au.shape) + [1] * extra_dims
        ).expand([-1, -1, -1] + [Q] * extra_dims)
        _T_au_expand.scatter_add_(2, edge_idx_au_expand, _T.float())

        # Fold-mean along self dimension
        shape_out = [shape[0], -1, N_asymmetric, N_asymmetric] + shape[3:]
        _T_au = _T_au_expand.reshape(shape_out).sum(1)
        return _T_au

    J_fold = _pairwise_fold(J)
    mask_ij_fold = (_pairwise_fold(mask_ij) > 0).float()
    edge_idx_fold = (
        torch.arange(N_asymmetric, device=device)
        .long()[None, None, :]
        .expand(mask_ij_fold.shape)
    )

    # Drop unused edges
    K_fold = mask_ij_fold.sum(2).max().item()
    _, sort_ix = torch.sort(mask_ij_fold, dim=2, descending=True)
    sort_ix_J = sort_ix[..., None, None].expand(list(sort_ix.shape) + [Q, Q])
    edge_idx_fold = torch.gather(edge_idx_fold, 2, sort_ix)
    mask_ij_fold = torch.gather(mask_ij_fold, 2, sort_ix)
    J_fold = torch.gather(J_fold, 2, sort_ix_J)

    # Fold-mean along self dimension
    h_fold = h.reshape([B, -1, N_asymmetric, Q]).sum(1)
    mask_i_fold = (mask_i.reshape([B, -1, N_asymmetric]).sum(1) > 0).float()
    if normalize:
        h_fold = h_fold / symmetry_order
        J_fold = J_fold / symmetry_order
    return h_fold, J_fold, edge_idx_fold, mask_i_fold, mask_ij_fold


@torch.no_grad()
def _color_graph(edge_idx, mask_ij, max_iter=100):
    """Stochastic graph coloring."""
    # Randomly assign initial colors
    B, N, K = edge_idx.shape
    # By Brooks we only need K + 1, but one extra color aids convergence
    num_colors = K + 2
    S = torch.randint(0, num_colors, (B, N), device=edge_idx.device)

    # Ignore self-attachement
    ix = torch.arange(edge_idx.shape[1], device=edge_idx.device)[None, ..., None]
    mask_ij = (mask_ij * torch.ne(edge_idx, ix).float())[..., None]

    # Iteratively replace clashing sites with an available color
    i = 0
    total_clashes = 1
    while total_clashes > 0 and i < max_iter:
        # Tabulate available colors in neighborhood
        O_i = F.one_hot(S, num_colors).float()
        N_i = (mask_ij * graph.collect_neighbors(O_i, edge_idx)).sum(2)
        clashes = (O_i * N_i).sum(-1)
        N_i = torch.where(N_i > 0, -float("inf") * torch.ones_like(N_i), N_i)

        # Resample from this distribution where clashing
        S_new = torch.distributions.categorical.Categorical(logits=N_i).sample()
        S = torch.where(clashes > 0, S_new, S)
        i += 1
        total_clashes = clashes.sum().item()
    return S


@torch.no_grad()
def _build_gamma_schedule(num_iterations: int, schedule_cfg: dict) -> List[float]:
    """Return γ at every DLMC step for a time-dependent guidance schedule.

    Convention: t = 1 - k / max(1, num_iterations - 1), so k=0 → t=1 (initial
    / fully masked) and k=num_iterations-1 → t=0 (final). Schedules from
    Rojas et al., ICLR 2026, Table 2.

    Supported types:
        - "constant":       γ(t) = gamma_max
        - "ramp_up":        γ(t) = min(gamma_max, gamma_max * (1 - t) / (1 - tau))
        - "right_interval": γ(t) = gamma_max if t >= tau else 0.0
    """
    N = int(num_iterations)
    if N <= 0:
        return []
    g_max = float(schedule_cfg["gamma_max"])
    stype = schedule_cfg["type"]
    denom_k = max(1, N - 1)

    out: List[float] = []
    for k in range(N):
        t = 1.0 - k / denom_k
        if stype == "constant":
            g = g_max
        elif stype == "ramp_up":
            tau = float(schedule_cfg["tau"])
            denom = max(1e-12, 1.0 - tau)
            g = min(g_max, g_max * (1.0 - t) / denom)
        elif stype == "right_interval":
            tau = float(schedule_cfg["tau"])
            g = g_max if t >= tau else 0.0
        else:
            raise ValueError(f"Unknown gamma schedule type: {stype!r}")
        out.append(float(g))
    return out


def sample_potts(
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    S: Optional[torch.LongTensor] = None,
    mask_sample: Optional[torch.Tensor] = None,
    num_sweeps: int = 100,
    temperature: float = 1.0,
    temperature_init: float = 1.0,
    annealing_fraction: float = 0.8,
    penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    differentiable_penalty: bool = True,
    rejection_step: bool = False,
    proposal: Literal["dlmc", "chromatic"] = "dlmc",
    verbose: bool = True,
    return_trajectory: bool = False,
    thin_sweeps: int = 3,
    edge_idx_coloring: Optional[torch.LongTensor] = None,
    mask_ij_coloring: Optional[torch.Tensor] = None,
    h_uncond: Optional[torch.Tensor] = None,
    J_uncond: Optional[torch.Tensor] = None,
    edge_idx_uncond: Optional[torch.LongTensor] = None,
    gamma: float = 1.0,
    gamma_schedule_cfg: Optional[dict] = None,
) -> Union[
    tuple[torch.LongTensor, torch.Tensor],
    tuple[torch.LongTensor, torch.Tensor, list[torch.LongTensor], list[torch.Tensor]],
]:
    """Sample from Potts model with Chromatic Gibbs sampling.

    Args:
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        S (torch.LongTensor, optional): Sequence for initialization with
            shape `(num_batch, num_nodes)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating
            positions which are free to change with shape
            `(num_batch, num_nodes)` or which tokens are acceptable at each position
            with shape `(num_batch, num_nodes, alphabet)`.
        num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
            i.e. the depth of sampling as measured by the number of times
            every position has had an opportunity to update.
        temperature (float): Final sampling temperature.
        temperature_init (float): Initial sampling temperature, which will
            be linearly interpolated to `temperature` over the course of
            the burn in phase.
        annealing_fraction (float): Fraction of the total sampling run during
            which temperature annealing occurs.
        penalty_func (Callable, optional): An optional penalty function which
            takes a sequence `S` and outputes a `(num_batch)` shaped tensor
            of energy adjustments, for example as regularization.
        differentiable_penalty (bool): If True, gradients of penalty function
            will be used to adjust the proposals.
        rejection_step (bool): If True, perform a Metropolis-Hastings
            rejection step.
        proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
        verbose (bool): If True print verbose output during sampling.
        return_trajectory (bool): If True, also output the sampling trajectories
            of `S` and `U`.
        thin_sweeps (int): When returning trajectories, only save every `thin_sweeps`
            state to reduce memory usage.
        edge_idx_coloring (torch.LongerTensor, optional): Alternative
            graph dependency structure that can be provided for the
            Chromatic Gibbs algorithm when it performs initial graph
            coloring. Has shape
                `(num_batch, num_nodes, num_neighbors_coloring)`.
        mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
            structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.

    Returns:
        S (torch.LongTensor): Sampled sequences with
            shape `(num_batch, num_nodes)`.
        U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower is more
            favorable.atb
        S_trajectory (list[torch.LongTensor]): List of sampled sequences through
            time each with shape `(num_batch, num_nodes)`.
        U_trajectory (list[torch.Tensor]): List of sampled energies through time
            each with shape `(num_batch)`.
    """
    # Initialize masked proposals and mask h
    mask_S, mask_mutatable, S = init_sampling_masks(-h, mask_sample, S) # mask_mutatable is mask_S_1D
    h_numerical_zero = h.max() + 1e3 * max(1.0, temperature) # Prohibit sampling tokens where mask_S > 0
    h = torch.where(mask_S > 0, h, h_numerical_zero * torch.ones_like(h))

    # Classifier-free-style guidance: if an uncond branch is provided, we
    # sample from a mix of the cond and uncond DLMC proposals at every sweep.
    use_guidance = h_uncond is not None
    if use_guidance:
        if proposal != "dlmc":
            raise NotImplementedError(
                "Potts guidance is only supported with the DLMC proposal; got "
                f"proposal={proposal!r}."
            )
        assert J_uncond is not None and edge_idx_uncond is not None, (
            "h_uncond was provided but J_uncond / edge_idx_uncond are missing."
        )
        # Apply a per-branch numerical-zero floor. Reusing cond's floor
        # (h.max() + 1e3*T) under-suppresses banned tokens whenever
        # h_uncond.max() > h.max(), which in turn widens the cond/uncond
        # disagreement that CFG at gamma > 1 amplifies. Computing the
        # uncond floor from h_uncond.max() keeps banned = numerically
        # zero in both branches regardless of their relative scale.
        h_numerical_zero_uncond = h_uncond.max() + 1e3 * max(1.0, temperature)
        h_uncond = torch.where(
            mask_S > 0,
            h_uncond,
            h_numerical_zero_uncond * torch.ones_like(h_uncond),
        )

    # Block update schedule
    if proposal == "chromatic":
        if edge_idx_coloring is None:
            edge_idx_coloring = edge_idx
        if mask_ij_coloring is None:
            mask_ij_coloring = mask_ij
        schedule = _color_graph(edge_idx_coloring, mask_ij_coloring)
        num_colors = schedule.max() + 1
        num_iterations = num_colors * num_sweeps
    else:
        num_iterations = num_sweeps

    num_iterations_annealing = int(annealing_fraction * num_iterations)
    temperatures = np.linspace(
        temperature_init, temperature, num_iterations_annealing
    ).tolist() + [temperature] * (num_iterations - num_iterations_annealing)

    if use_guidance:
        def _energy_proposal(_S, _T, _gamma):
            return _potts_proposal_dlmc_guidance_energy(
                _S,
                h,
                J,
                edge_idx,
                h_uncond,
                J_uncond,
                edge_idx_uncond,
                gamma=_gamma,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
    elif proposal == "chromatic":
        def _energy_proposal(_S, _T, _gamma=None):
            return _potts_proposal_gibbs(
                _S,
                h,
                J,
                edge_idx,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
    elif proposal == "dlmc":
        def _energy_proposal(_S, _T, _gamma=None):
            return _potts_proposal_dlmc(
                _S,
                h,
                J,
                edge_idx,
                T=_T,
                penalty_func=penalty_func,
                differentiable_penalty=differentiable_penalty,
            )
    else:
        raise NotImplementedError

    # Per-step γ trajectory. When no schedule is supplied (or schedule is the
    # constant type), this collapses to [gamma] * num_iterations and behavior
    # is bit-for-bit identical to the legacy constant-γ path.
    if (
        use_guidance
        and gamma_schedule_cfg is not None
        and gamma_schedule_cfg.get("type", "constant") != "constant"
    ):
        gammas_per_step = _build_gamma_schedule(num_iterations, gamma_schedule_cfg)
    else:
        gammas_per_step = [float(gamma)] * num_iterations

    cumulative_sweeps = 0
    if return_trajectory:
        S_trajectory = []
        U_trajectory = []
    for i, T_i in enumerate(tqdm(temperatures, desc="Potts Sampling", leave=False)):
        g_i = gammas_per_step[i]
        # Cycle through Gibbs updates random sites to the update with fixed prob
        if proposal == "chromatic":
            mask_update = schedule.eq(i % num_colors)
        else:
            mask_update = torch.ones_like(S) > 0
        if mask_mutatable is not None:
            mask_update = mask_update * (mask_mutatable > 0)

        # Compute current energy and local conditionals
        U, logp = _energy_proposal(S, T_i, g_i)

        # Propose
        S_new = torch.distributions.categorical.Categorical(logits=logp).sample()
        S_new = torch.where(mask_update, S_new, S)
        #* As padded positions only have 1 at index 0, they will be always alanine anyway

        # Metropolis-Hastings adjusment
        if rejection_step:

            def _flux(_U, _logp, _S):
                logp_transition = torch.gather(_logp, -1, _S[..., None])
                _logp_ij = (mask_update.float() * logp_transition[..., 0]).sum(1)
                flux = -_U / T_i + _logp_ij
                return flux

            U_new, logp_new = _energy_proposal(S_new, T_i, g_i)

            _flux_backward = _flux(U_new, logp_new, S)
            _flux_forward = _flux(U, logp, S_new)
            acc_ratio = torch.exp((_flux_backward - _flux_forward)).clamp(max=1.0)
            if verbose:  # and i % 100 == 0:
                print(
                    f"{(U_new - U).mean().item():0.2f}"
                    f"\t{(_flux_backward - _flux_forward).mean().item():0.2f}"
                    f"\t{acc_ratio.mean().item():0.2f}"
                )
            u = torch.bernoulli(acc_ratio)[..., None]
            S = torch.where(u > 0, S_new, S)
            cumulative_sweeps += (u * mask_update).sum(1).mean().item() / S.shape[1]
        else:
            S = S_new
            cumulative_sweeps += (mask_update).float().sum(1).mean().item() / S.shape[1]

        if return_trajectory and i % (thin_sweeps) == 0:
            S_trajectory.append(S)
            U_trajectory.append(U)

        if use_guidance:
            # Keep the reported U consistent with the mixed distribution
            # we are actually sampling from (penalty-free; raw physical Potts
            # energies on both branches, mixed with the terminal γ).
            g_final = gammas_per_step[-1] if gammas_per_step else float(gamma)
            U_cond_final, _ = compute_potts_energy(S, h, J, edge_idx)
            U_uncond_final, _ = compute_potts_energy(S, h_uncond, J_uncond, edge_idx_uncond)
            U = g_final * U_cond_final + (1.0 - g_final) * U_uncond_final
        else:
            U, _ = compute_potts_energy(S, h, J, edge_idx)

    if verbose:
        print(f"Effective number of sweeps: {cumulative_sweeps}")
    if return_trajectory:
        return S, U, S_trajectory, U_trajectory
    else:
        return S, U


def init_sampling_masks(
    logits_init: torch.Tensor,
    mask_sample: Optional[torch.Tensor] = None,
    S: Optional[torch.LongTensor] = None,
    ban_S: Optional[list[int]] = None,
    pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Parse sampling masks and an initial sequence.

    Args:
        logits_init (torch.Tensor): Logits for sequence initialization with shape
            `(num_batch, num_nodes, alphabet)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating which
            positions are free to change with shape `(num_batch, num_nodes)` or which
            tokens are valid at each position with shape
            `(num_batch, num_nodes, alphabet)`. In the latter case, `mask_sample` will
            take priority over `S` except for positions in which `mask_sample` is
            all zero.
        S (torch.LongTensor optional): Initial sequence with shape
            `(num_batch, num_nodes)`.
        ban_S (list of int, optional): Optional list of alphabet indices to ban from
            all positions during sampling.
        pos_restrict_aatype (tuple of torch.Tensor, optional): Tuple of two tensors with shape
            `(num_batch, num_nodes)` and `(num_batch, num_nodes, alphabet)`, respectively,
            indicating which positions are restricted to certain aatypes and which aatypes
            are allowed at each position.

    Returns:
        mask_sample (torch.Tensor): Finalized position specific mask with shape
            `(num_batch, num_nodes, alphabet)`.
        S (torch.Tensor): Self-consistent initial `S` with shape
            `(num_batch, num_nodes)`.
    """

    if S is None and mask_sample is not None:
        raise Exception("To use masked sampling, please provide an initial S")

    if mask_sample is None:
        mask_S = torch.ones_like(logits_init)
    elif mask_sample.dim() == 2:
        # Position-restricted sampling
        # Used for generating initial mask for potts sampling
        # mask_sample: mask for positions that are free to sample
        # mask_sample: (B, N), logits_init: (B, N, const.AF3_ENCODING.n_tokens)
        # S: (B, N). Initial sequence.
        mask_sample_expand = mask_sample[..., None].expand(logits_init.shape) # (B, N, const.AF3_ENCODING.n_tokens)
        # mask_sample is 1 for positions that are free to sample, 0 for positions that are not free to sample
        O_init = F.one_hot(S, logits_init.shape[-1]).float() # (B, N, const.AF3_ENCODING.n_tokens)
        mask_S = mask_sample_expand + (1 - mask_sample_expand) * O_init
        # Since O_init is 0 for padded positions, mask_S[b, padded_positions, 0] = 1
        # mask_S is 0 for non-padded positions, and 1 for padded positions.
    elif mask_sample.dim() == 3:
        O_init = F.one_hot(S, logits_init.shape[-1]).float()
        # Mutation-restricted sampling
        mask_zero = (mask_sample.sum(-1, keepdim=True) == 0).float()
        # for padded_positions, mask_sample[b, padded_positions, 0] = 1
        # So mask_zero is 0 for padded positions
        mask_S = ((mask_zero * O_init + mask_sample) > 0).float()
        # And thus mask_S[b, padded_positions, 0] = 1
    else:
        raise NotImplementedError

    # Handle aatype restrictions
    if ban_S is not None:
        # ban certain aatypes
        mask_S[:, :, ban_S] = 0.0
        # ban_S = {"X"} + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)
        # (251109) mask_S is 0.0 for all non-protein tokens for now.

    if pos_restrict_aatype is not None:
        # restrict to certain aatypes at certain positions
        restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype  # (B, N), (B, N, K)
        mask_S[restrict_pos_mask.bool()] = allowed_aatype_mask[restrict_pos_mask.bool()]

    mask_S_1D = (mask_S.sum(-1) > 1).float()  # check where we can sample
    # For initial mask generation,
    # padded positions are 0, as mask_S.sum(-1) = 1 for padded positions
    # For the second mask generation, also the same

    logits_init_masked = 1000 * mask_S + logits_init
    #! 1000 where we can sample, 0 where we can't (or don't want to) sample
    S_init = torch.distributions.categorical.Categorical(logits=logits_init_masked).sample()
    S = torch.where(mask_S_1D.bool(), S_init, S)  # where we can sample, set S to S_init
    S = torch.where(mask_S.sum(-1) == 1, mask_S.argmax(-1), S)  # where there is only one possible aatype, set S to the aatype
    # This is why [b, padded_positions, 0] = 1 for S.
    return mask_S, mask_S_1D, S


def _potts_proposal_gibbs(
    S, h, J, edge_idx, T=1.0, penalty_func=None, differentiable_penalty=True
):
    U, U_i = compute_potts_energy(S, h, J, edge_idx)

    if penalty_func is not None:
        if differentiable_penalty:
            with torch.enable_grad():
                S_onehot = F.one_hot(S, h.shape[0 - 1]).float()
                S_onehot.requires_grad = True
                U_penalty = penalty_func(S_onehot)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [S_onehot])[
                    0
                ].detach()
                U_penalty = U_penalty.detach()
            U_i = U_i + 0.5 * U_i_adjustment
        else:
            U_penalty = penalty_func(S_onehot)
        U = U + U_penalty

    logp_i = F.log_softmax(-U_i / T, dim=-1)
    return U, logp_i


def _potts_proposal_dlmc(
    S,
    h,
    J,
    edge_idx,
    T=1.0,
    penalty_func=None,
    differentiable_penalty=True,
    dt=0.1,
    autoscale=True,
    balancing_func="sigmoid",
):
    # Compute energy gap
    U, U_i = compute_potts_energy(S, h, J, edge_idx)
    # print(U)
    U_i = U_i
    if penalty_func is not None:
        O = F.one_hot(S, h.shape[0 - 1]).float()
        if differentiable_penalty:
            with torch.enable_grad():
                O.requires_grad = True
                U_penalty = penalty_func(O)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
                U_penalty = U_penalty.detach()
                U_i_adjustment = U_i_adjustment - torch.gather(
                    U_i_adjustment, -1, S[..., None]
                )
                # Base-off the values by subtracting the U_i_adjustment of the current state
            U_i_mutate = U_i - torch.gather(U_i, -1, S[..., None])
            # Base-off, but it's not used anywhere, why?

            U_i = U_i + U_i_adjustment
        else:
            U_penalty = penalty_func(O)
        U = U + U_penalty

    # Compute local equilibrium distribution
    logP_j = F.log_softmax(-U_i / T, dim=-1)

    # Compute transition log probabilities
    O = F.one_hot(S, h.shape[0 - 1]).float()
    logP_i = torch.gather(logP_j, -1, S[..., None])
    # log probability of the current state
    if balancing_func == "sqrt":
        log_Q_ij = 0.5 * (logP_j - logP_i)
    elif balancing_func == "sigmoid":
        log_Q_ij = F.logsigmoid(logP_j - logP_i)
    else:
        raise NotImplementedError

    rate = torch.exp(log_Q_ij - logP_j)

    # Compute transition probability
    logP_ij = logP_j + (-(-dt * rate).expm1()).log()
    p_flip = ((1.0 - O) * logP_ij.exp()).sum(-1, keepdim=True)

    # DEBUG:
    # flux = ((1. - O) * torch.exp(log_Q_ij)).mean([1,2], keepdim=True)
    # print(f" ->Flux is {flux.item():0.2f}, FlipProb is {p_flip.mean():0.2f}")

    logP_ii = (1.0 - p_flip).clamp(1e-5).log()
    logP_ij = (1.0 - O) * logP_ij + O * logP_ii
    return U, logP_ij


def _potts_proposal_dlmc_guidance_energy(
    S,
    h_cond,
    J_cond,
    edge_idx_cond,
    h_uncond,
    J_uncond,
    edge_idx_uncond,
    gamma=1.0,
    T=1.0,
    penalty_func=None,
    differentiable_penalty=True,
    dt=0.1,
    balancing_func="sigmoid",
):
    """Energy-space CFG: build h_mix, J_mix once, reuse `_potts_proposal_dlmc`.

    Exploits the linearity of the Potts energy in `(h, J)` at fixed `edge_idx`::

        U(S; γ·h_cond + (1-γ)·h_uncond, γ·J_cond + (1-γ)·J_uncond)
            = γ·U_cond(S) + (1-γ)·U_uncond(S)

    so running the standard DLMC proposal on the linearly-mixed parameters
    samples from the Boltzmann distribution of `U_guided = γ·U_cond + (1-γ)·U_uncond`.
    Requires `edge_idx_cond == edge_idx_uncond`, which holds whenever cond and
    uncond differ only in `atom_cond_mask` (same atom positions ⇒ same kNN
    graph). This is asserted at runtime; any future regression where the graph
    depends on the conditioning will surface immediately.

    Complexity penalties are handled by the inner `_potts_proposal_dlmc` on the
    mixed parameters — since the penalty depends only on `S`, this is equivalent
    to applying it once and sharing it across both branches.
    """
    assert h_cond.shape == h_uncond.shape, (
        f"cond/uncond h shape mismatch: {tuple(h_cond.shape)} vs {tuple(h_uncond.shape)}"
    )
    assert J_cond.shape == J_uncond.shape, (
        f"cond/uncond J shape mismatch: {tuple(J_cond.shape)} vs {tuple(J_uncond.shape)}"
    )
    assert edge_idx_cond.shape == edge_idx_uncond.shape, (
        f"cond/uncond edge_idx shape mismatch: "
        f"{tuple(edge_idx_cond.shape)} vs {tuple(edge_idx_uncond.shape)}"
    )
    assert torch.equal(edge_idx_cond, edge_idx_uncond), (
        "edge_idx mismatch between cond/uncond Potts branches — energy-space "
        "guidance requires the two branches to share the same neighbor graph."
    )

    h_mix = gamma * h_cond + (1.0 - gamma) * h_uncond
    J_mix = gamma * J_cond + (1.0 - gamma) * J_uncond

    return _potts_proposal_dlmc(
        S,
        h_mix,
        J_mix,
        edge_idx_cond,
        T=T,
        penalty_func=penalty_func,
        differentiable_penalty=differentiable_penalty,
        dt=dt,
        balancing_func=balancing_func,
    )


def _mask_J(edge_idx, mask_i, mask_ij):
    # Remove self edges
    device = edge_idx.device
    ii = torch.arange(edge_idx.shape[1]).view((1, -1, 1)).to(device)
    not_self = torch.ne(edge_idx, ii).type(torch.float32)

    # Remove missing edges
    self_present = mask_i.unsqueeze(-1)
    neighbor_present = graph.collect_neighbors(self_present, edge_idx)
    neighbor_present = neighbor_present.squeeze(-1)

    mask_J = not_self * self_present * neighbor_present
    if mask_ij is not None:
        mask_J = mask_ij * mask_J
    return mask_J


def pseudolikelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
) -> torch.Tensor:
    """Compute Potts pseudolikelihood log-probabilities from a sequence.

    Module-level mirror of :meth:`GraphPotts.pseudolikelihood` so loss code can
    call it without a GraphPotts instance (same pattern as
    ``log_composite_likelihood``).

    Inputs:
        S (torch.LongTensor): Sequence with shape ``(num_batch, num_nodes)``.
        h (torch.Tensor): Potts fields with shape
            ``(num_batch, num_nodes, num_states)``.
        J (torch.Tensor): Potts couplings with shape
            ``(num_batch, num_nodes, num_neighbors, num_states, num_states)``.
        edge_idx (torch.LongTensor): Edge indices with shape
            ``(num_batch, num_nodes, num_neighbors)``.

    Outputs:
        log_probs (torch.Tensor): Per-site conditional log-probabilities with
            shape ``(num_batch, num_nodes, num_states)``.
    """
    num_states = J.shape[-1]
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    J_i = J_ij.sum(2)
    logits = h + J_i
    return F.log_softmax(-logits, dim=-1)


def log_pseudolikelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Potts per-site pseudolikelihood at the true residue.

    Sibling of :func:`log_composite_likelihood`. Returns the masked log-prob
    tensor and site mask so the loss module aggregates identically in both
    cases (no one-hot target arithmetic in the caller).

    Inputs:
        S (torch.LongTensor): Sequence with shape ``(num_batch, num_nodes)``.
        h (torch.Tensor): Potts fields with shape
            ``(num_batch, num_nodes, num_states)``.
        J (torch.Tensor): Potts couplings with shape
            ``(num_batch, num_nodes, num_neighbors, num_states, num_states)``.
        edge_idx (torch.LongTensor): Edge indices with shape
            ``(num_batch, num_nodes, num_neighbors)``.
        mask_i (torch.Tensor): Node mask with shape ``(num_batch, num_nodes)``.
        smoothing_alpha (float): Label smoothing probability on ``(0, 1)``.

    Outputs:
        logp_i (torch.Tensor): ``log p(S_i | S_{N(i)})`` at the true ``S_i``,
            masked by ``mask_i``, with shape ``(num_batch, num_nodes)``.
        mask_i (torch.Tensor): Site mask (returned for symmetry with
            ``log_composite_likelihood``).
    """
    num_states = J.shape[-1]

    # Full per-site conditional log-prob: logp[b,i,q] = log p(S_i = q | S_{N(i)})
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)
    J_i = J_ij.sum(2)
    logp = F.log_softmax(-(h + J_i), dim=-1)

    # Score the true residue at each site.
    logp_i = torch.gather(logp, -1, S.unsqueeze(-1)).squeeze(-1)

    # Optional label smoothing — per-site analog of log_composite_likelihood's
    # pair-level scheme. num_bins = num_states (not num_states**2).
    if smoothing_alpha > 0.0:
        prob_no_smooth = 1.0 - smoothing_alpha
        prob_background = (1.0 - prob_no_smooth) / float(num_states - 1)
        # Corrects for double-counting of the foreground bin inside logp.sum(-1).
        p_foreground = prob_no_smooth - prob_background
        logp_i = p_foreground * logp_i + prob_background * logp.sum(-1)

    logp_i = mask_i * logp_i
    return logp_i, mask_i


def log_composite_likelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Potts pairwise composite likelihoods from sequence.

    Inputs:
        S (torch.LongTensor): Sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        smoothing_alpha (float): Label smoothing probability on `(0,1)`.

    Outputs:
        logp_ij (torch.Tensor): Potts pairwise composite likelihoods evaluated
            for the current sequence with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_p_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    num_batch, num_residues, num_k, num_states, _ = list(J.size())

    # Gather J clamped at j
    # [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    # (B,i,j,A_i)
    J_clamp_j = torch.gather(J, -1, S_j).squeeze(-1)

    # Gather J clamped at i
    S_i = S.view(num_batch, num_residues, 1, 1, 1)
    S_i = S_i.expand(-1, -1, num_k, num_states, num_states)
    # (B,i,j,1,A_j)
    J_clamp_i = torch.gather(J, -2, S_i)

    # Compute background per-site contributions that sum out J
    # (B,i,j,A_i) => (B,i,A_i)
    r_i = h + J_clamp_j.sum(2)
    r_j = graph.collect_neighbors(r_i, edge_idx)

    # Remove J_ij from the i contributions
    # (B,i,A_i) => (B,i,:,A_i,:)
    r_i = r_i.view([num_batch, num_residues, 1, num_states, 1])
    r_i_minus_ij = r_i - J_clamp_j.unsqueeze(-1)

    # Remove J_ji from the j contributions
    # (B,j,A_j) => (B,:,j,:,A_j)
    r_j = r_j.view([num_batch, num_residues, num_k, 1, num_states])
    r_j_minus_ji = r_j - J_clamp_i

    # Composite likelihood (B,i,j,A_i,A_j)
    logits_ij = r_i_minus_ij + r_j_minus_ji + J
    logits_ij = logits_ij.view([num_batch, num_residues, num_k, -1])
    logp = F.log_softmax(-logits_ij, dim=-1)
    logp = logp.view([num_batch, num_residues, num_k, num_states, num_states])

    # Score the current sequence under
    # (B,i,j,A_i,A_j) => (B,i,j,A_i) => (B,i,j)
    logp_j = torch.gather(logp, -1, S_j).squeeze(-1)
    S_i = S.view(num_batch, num_residues, 1, 1).expand(-1, -1, num_k, -1)
    logp_ij = torch.gather(logp_j, -1, S_i).squeeze(-1)

    # Optional label smoothing (scaled assuming per-token smoothing )
    if smoothing_alpha > 0.0:
        # Foreground probability
        num_bins = num_states ** 2
        prob_no_smooth = (1.0 - smoothing_alpha) ** 2
        prob_background = (1.0 - prob_no_smooth) / float(num_bins - 1)
        # The second term corrects for double counting in background sum
        p_foreground = prob_no_smooth - prob_background
        logp_ij = p_foreground * logp_ij + prob_background * logp.sum([-2, -1])

    mask_p_ij = _mask_J(edge_idx, mask_i, mask_ij)
    logp_ij = mask_p_ij * logp_ij
    return logp_ij, mask_p_ij
