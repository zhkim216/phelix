from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from scipy import stats
from torchtyping import TensorType
from allatom_design.data import residue_constants as rc


class ScaffoldManager(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Handles providing scaffolding inputs for the AtomDenoiser.
        """
        super().__init__()
        self.cfg = cfg

        self.contiguous_p = cfg.contiguous_p
        self.discontiguous_p = cfg.discontiguous_p
        self.max_span_len = cfg.max_span_len
        self.max_discontiguous_res = cfg.max_discontiguous_res
        self.dist_threshold = cfg.dist_threshold

        # Define conditioning types and their probabilities
        self.conditioning_types = ["contiguous", "discontiguous", "none"]
        self.conditioning_probs = torch.tensor([
            self.contiguous_p,
            self.discontiguous_p,
            1.0 - (self.contiguous_p + self.discontiguous_p)
        ])


    @torch.compiler.disable
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                ) -> Dict[str, Any]:
        atom_mask = batch["atom_mask"]
        seq_mask = batch["seq_mask"]
        x = batch["x"]

        scaffold_mask = torch.zeros_like(atom_mask)    # 1 for unmasked, 0 for masked
        seq_lens = seq_mask.sum(dim=-1).long()
        device = scaffold_mask.device

        for i, seq_len in enumerate(seq_lens):
            seq_len = seq_len.item()

            # Choose a conditioning type
            conditioning_type = self.conditioning_types[torch.multinomial(self.conditioning_probs, 1).item()]
            if conditioning_type == "contiguous":
                # Scaffold a sequence-contiguous span
                span_len = torch.randint(1, min(self.max_span_len, seq_len) + 1, (1,), device=device).item()
                start = torch.randint(0, seq_len - span_len + 1, (1,), device=device).item()
                scaffold_mask[i, start:start + span_len] = 1
            elif conditioning_type == "discontiguous":
                # Scaffold based on spatial proximity
                # we select neighbors by CA distances
                ca_coords = x[i, :, rc.atom_order["CA"]]
                ca_dist = torch.cdist(ca_coords, ca_coords)

                # select random residue
                random_residue_idx = torch.randint(0, seq_len, (1,), device=device).item()
                ca_dist_i = ca_dist[random_residue_idx] + 1e5 * (1 - atom_mask[i, :, rc.atom_order["CA"]])  # mask out non-existing atoms
                close_mask = ca_dist_i <= self.dist_threshold
                n_neighbors = close_mask.sum().int()

                if n_neighbors <= 1:
                    # If we have 1 or 0 neighbors, fall back to just using the selected residue
                    scaffold_mask[i, random_residue_idx] = 1
                else:
                    # Pick random number of neighbors
                    n_to_select = torch.randint(2, min(self.max_discontiguous_res, n_neighbors) + 1, (1,), device=device).item()

                    # Get indices of neighbors (including the original residue)
                    neighbor_indices = torch.where(close_mask)[0]
                    selected_indices = neighbor_indices[torch.randperm(len(neighbor_indices))[:n_to_select]]
                    scaffold_mask[i, selected_indices] = 1

        # Only condition on backbone atoms  # TODO: support sidechain atoms
        scaffold_mask[:, rc.non_bb_idxs] = 0
        aatype_in = torch.full_like(batch["residue_index"], fill_value=rc.restype_order_with_x["X"])  # TODO: fix for sequence conditioning

        scaffold_mask = scaffold_mask * atom_mask  # unmask only existing atoms
        x_scaffold = x * scaffold_mask[..., None]
        return {"x_scaffold": x_scaffold, "scaffold_mask": scaffold_mask, "aatype_in": aatype_in}
