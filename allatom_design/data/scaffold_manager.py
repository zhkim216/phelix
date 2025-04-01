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

        # Sidechain options
        self.sidechain_dropout_p = cfg.get("sidechain_dropout_p", 1.0)  # 1.0 means backbone-only conditioning

        self.translation_scale = cfg.get("translation_scale", 1.0)
        self.se3_augment = cfg.get("se3_augment", True)

        # Override conditioning type
        self.cond_type_override = None  # only use this conditioning type; for inference and evaluation
        self.motif_type = "random"  # ["backbone", "allatom", "random"]


    @torch.compiler.disable
    def forward(self, example: Dict[str, TensorType["..."]],) -> Dict[str, Any]:
        atom_mask = example["atom_mask"]
        seq_mask = example["seq_mask"]
        x = example["x"]
        aatype = example["aatype"]

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

        motif_mask = motif_mask * atom_mask  # unmask only existing atoms

        # Sidechain / aatype conditioning
        scn_mask = self.get_motif_scn_mask(motif_mask)

        # Apply motif masks
        motif_mask[:, rc.non_bb_idxs] = motif_mask[:, rc.non_bb_idxs] * scn_mask[:, None]  # mask out some sidechains
        x_motif = x * motif_mask[..., None]

        # where we keep the sidechain, use the original aatype
        aatype_motif = torch.where(scn_mask.bool(), aatype, torch.full_like(example["residue_index"], fill_value=rc.restype_order_with_x["X"]))

        # Re-center on CA of motif residues
        x_recentered = x
        if self.se3_augment and (motif_mask[..., 1:2].any()):  # only center if there are any scaffolding residues
            x_motif, transforms = center_random_augmentation(x_motif, seq_mask, motif_mask,
                                                             translation_scale=self.translation_scale,
                                                             return_transforms=True)
            x_recentered = apply_random_augmentation(x, transforms, seq_mask, atom_mask)

        return {"x_motif": x_motif, "motif_mask": motif_mask, "aatype_motif": aatype_motif, "x_recentered": x_recentered}


    def get_motif_scn_mask(self, motif_mask: TensorType["n a", float]) -> TensorType["n", float]:
        """
        Get mask denoting which sidechains within the motif to condition on. 1 if we should condition on the sidechain, 0 otherwise.
        """
        N, _ = motif_mask.shape
        motif_residue_mask = motif_mask.any(dim=-1)

        if self.motif_type == "random":
            # Randomly mask out sidechains
            if torch.rand(1) < self.sidechain_dropout_p:
                # Only condition on backbone atoms
                scn_mask = torch.zeros(N, device=motif_mask.device)
            else:
                # Condition on sidechain atoms and their aatypes
                p = torch.rand(1)  # for this example, choose a random probability to keep sidechains of each sidechain
                scn_mask = torch.rand(N, device=motif_mask.device) < p
                scn_mask = scn_mask * motif_residue_mask  # subset sidechain mask to current motif residues

        else:
            assert not self.training, "Motif type should only be set during inference"
            if self.motif_type == "backbone":
                # use only backbone for motif conditioning
                scn_mask = torch.zeros(N, device=motif_mask.device) * motif_residue_mask
            elif self.motif_type == "allatom":
                # use all sidechains for motif conditioning
                scn_mask = torch.ones(N, device=motif_mask.device) * motif_residue_mask
            else:
                raise ValueError(f"Unknown motif type: {self.motif_type}")

        return scn_mask.float()


    def set_conditioning_type(self, conditioning_type: str) -> None:
        """
        Set the conditioning type for the scaffold manager.
        """
        assert conditioning_type in self.conditioning_types, f"Unknown conditioning type: {conditioning_type}"
        self.cond_type_override = conditioning_type
        print(f"ScaffoldManager: set conditioning type to {conditioning_type}")


    def set_motif_type(self, motif_type: str) -> None:
        """
        Set the motif type for the scaffold manager.
        """
        assert motif_type in ["backbone", "allatom", "random"], f"Unknown motif type: {motif_type}"
        self.motif_type = motif_type
        print(f"ScaffoldManager: set motif type to {motif_type}")


    def reset(self) -> None:
        """
        Reset the scaffold manager to its original state.
        """
        self.cond_type_override = None
        self.motif_type = "random"


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
