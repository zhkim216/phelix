from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (apply_random_augmentation,
                                      center_random_augmentation)


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
        self.conditioning_types = ["contiguous", "discontiguous", "unconditional"]
        self.conditioning_probs = torch.tensor([
            self.contiguous_p,
            self.discontiguous_p,
            1.0 - (self.contiguous_p + self.discontiguous_p)
        ])

        self.translation_scale = cfg.get("translation_scale", 1.0)
        self.se3_augment = cfg.get("se3_augment", True)

        self.cond_type_override = None  # only use this conditioning type; for inference and evaluation


    @torch.compiler.disable
    def forward(self, example: Dict[str, TensorType["..."]],) -> Dict[str, Any]:
        atom_mask = example["atom_mask"]
        seq_mask = example["seq_mask"]
        x = example["x"]

        motif_mask = torch.zeros_like(atom_mask)    # 1 for unmasked, 0 for masked
        seq_len = seq_mask.sum().long().item()
        device = motif_mask.device

        if self.cond_type_override is None:
            # choose a conditioning type (during training)
            conditioning_type = self.conditioning_types[torch.multinomial(self.conditioning_probs, 1).item()]
        else:
            # use the override conditioning type (during inference)
            assert not self.training, "Override conditioning type should not be set during training"
            conditioning_type = self.cond_type_override

        assert conditioning_type in self.conditioning_types, f"Unknown conditioning type: {conditioning_type}"

        if conditioning_type == "contiguous":
            # Scaffold a sequence-contiguous span
            span_len = torch.randint(1, min(self.max_span_len, seq_len) + 1, (1,), device=device).item()
            start = torch.randint(0, seq_len - span_len + 1, (1,), device=device).item()
            motif_mask[start:start + span_len] = 1
        elif conditioning_type == "discontiguous":
            # Scaffold based on spatial proximity
            # we select neighbors by CA distances
            ca_coords = x[:, rc.atom_order["CA"]]
            ca_dist = torch.cdist(ca_coords, ca_coords)

            # select random residue
            random_residue_idx = torch.randint(0, seq_len, (1,), device=device).item()
            ca_dist_i = ca_dist[random_residue_idx] + 1e5 * (1 - atom_mask[:, rc.atom_order["CA"]])  # mask out non-existing atoms
            close_mask = ca_dist_i <= self.dist_threshold
            n_neighbors = close_mask.sum().int()

            if n_neighbors <= 1:
                # If we have 1 or 0 neighbors, fall back to just using the selected residue
                motif_mask[random_residue_idx] = 1
            else:
                # Pick random number of neighbors
                n_to_select = torch.randint(2, min(self.max_discontiguous_res, n_neighbors) + 1, (1,), device=device).item()

                # Get indices of neighbors (including the original residue)
                neighbor_indices = torch.where(close_mask)[0]
                selected_indices = neighbor_indices[torch.randperm(len(neighbor_indices))[:n_to_select]]
                motif_mask[selected_indices] = 1

        # Only condition on backbone atoms  # TODO: support sidechain atoms
        motif_mask[:, rc.non_bb_idxs] = 0
        aatype_motif = torch.full_like(example["residue_index"], fill_value=rc.restype_order_with_x["X"])  # TODO: fix for sequence conditioning

        motif_mask = motif_mask * atom_mask  # unmask only existing atoms
        x_motif = x * motif_mask[..., None]

        # Re-center on CA of motif residues
        x_recentered = x
        if self.se3_augment and (motif_mask[..., 1:2].any()):  # only center if there are any scaffolding residues
            x_motif, transforms = center_random_augmentation(x_motif, seq_mask, motif_mask,
                                                             translation_scale=self.translation_scale,
                                                             return_transforms=True)
            x_recentered = apply_random_augmentation(x, transforms, seq_mask, atom_mask)

        return {"x_motif": x_motif, "motif_mask": motif_mask, "aatype_motif": aatype_motif, "x_recentered": x_recentered}


    def set_conditioning_type(self, conditioning_type: str) -> None:
        """
        Set the conditioning type for the scaffold manager.
        """
        assert conditioning_type in self.conditioning_types, f"Unknown conditioning type: {conditioning_type}"
        self.cond_type_override = conditioning_type
        print(f"ScaffoldManager: set conditioning type to {conditioning_type}")


def get_scaffold_manager(cfg: Optional[DictConfig]) -> Optional[ScaffoldManager]:
    """
    Get the scaffold manager specified in the config.
    """
    if (cfg is None) or (cfg.name == "unconditional"):
        return None
    elif cfg.name == "scaffold_manager":
        return ScaffoldManager(cfg)
    else:
        raise ValueError(f"Unknown scaffold manager: {cfg.name}")
