from typing import Optional

import torch
import torch.nn as nn
from allatom_design.data.types import Tokenized
from omegaconf import DictConfig
from allatom_design.data import const
from torchtyping import TensorType


class MotifSelector():
    def __init__(self, cfg: DictConfig):
        """
        Handles selecting motifs for the AtomDenoiser.
        """
        super().__init__()
        self.cfg = cfg

        # Parse motif token selection probabilities
        self.motif_type, self.motif_probs = [], []
        for motif_type, motif_prob in cfg.p.items():
            self.motif_type.append(motif_type)
            self.motif_probs.append(motif_prob)

        # Parse restype mask probabilities
        self.restype_mask_type, self.restype_mask_probs = [], []
        for restype_mask_type, restype_mask_prob in cfg.restype_mask_p.items():
            self.restype_mask_type.append(restype_mask_type)
            self.restype_mask_probs.append(restype_mask_prob)

        # Parse motif atom selection probabilities
        self.motif_atom_type, self.motif_atom_probs = [], []
        for motif_atom_type, motif_atom_prob in cfg.atom_p.items():
            self.motif_atom_type.append(motif_atom_type)
            self.motif_atom_probs.append(motif_atom_prob)


    def select_motif_tokens(self, tokenized: Tokenized) -> TensorType["n", float]:
        """
        Selects a motif from tokenized data.
        """
        motif_type = self.motif_type[torch.multinomial(torch.tensor(self.motif_probs), 1).item()]

        if motif_type == "unconditional":
            return select_unconditional(tokenized)
        elif motif_type == "protein_contiguous":
            return select_protein_contiguous_motif(tokenized, **self.cfg.protein_contiguous)
        elif motif_type == "protein_discontiguous":
            return select_protein_discontiguous_motif(tokenized, **self.cfg.protein_discontiguous)
        else:
            raise ValueError(f"Unknown motif selector: {motif_type}")


    def create_restype_mask(self, tokenized: Tokenized) -> TensorType["n", float]:
        """
        Create a mask denoting which restypes to mask out. 0 if we should mask, 1 if we should keep.

        Non-polymer restypes are always kept (1).
        """
        restype_mask_type = self.restype_mask_type[torch.multinomial(torch.tensor(self.restype_mask_probs), 1).item()]

        if restype_mask_type == "all":
            # Mask out all restypes
            restype_mask = torch.zeros(len(tokenized.tokens))
        elif restype_mask_type == "uniform":
            # Mask out with probability keep_p ~ U[0, 1]
            keep_p = torch.rand(1).item()
            restype_mask = (torch.rand(len(tokenized.tokens)) < keep_p).float()
        else:
            raise ValueError(f"Unknown restype mask type: {restype_mask_type}")

        # Non-polymer restypes are always kept
        nonpolymer_mask = tokenized.tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        restype_mask[nonpolymer_mask] = 1

        return restype_mask


    def create_residx_mask(self, tokenized: Tokenized) -> TensorType["n", float]:
        pass


    def select_motif_atoms(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Given features subsetted to motif tokens, select atoms for the motif.
        """
        motif_atom_type = self.motif_atom_type[torch.multinomial(torch.tensor(self.motif_atom_probs), 1).item()]

        if motif_atom_type == "all":
            return select_all_atoms(feats)
        elif motif_atom_type == "protein_sidechain":
            return select_protein_sidechain_atoms(feats)
        elif motif_atom_type == "protein_backbone":
            return select_protein_backbone_atoms(feats)
        else:
            raise ValueError(f"Unknown motif atom selector: {motif_atom_type}")


##################### Token-level motif selection #####################

def select_unconditional(tokenized: Tokenized) -> TensorType["n", float]:
    """
    Selects a motif for the AtomDenoiser.
    """
    return torch.zeros(len(tokenized.tokens))


def select_protein_contiguous_motif(tokenized: Tokenized,
                                    max_span_len: int) -> TensorType["n", float]:
    """
    Selects a contiguous protein motif for the AtomDenoiser.

    TODO: might want to select the motif span from protein residues only
    """
    # Select from known protein residues
    protein_token_mask = tokenized.tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
    known_residue_mask = tokenized.tokens["res_type"] != const.token_ids[const.unk_token["PROTEIN"]]
    protein_token_mask = protein_token_mask * known_residue_mask

    # Select contiguous span
    seq_len = len(tokenized.tokens)
    span_len = torch.randint(1, min(max_span_len, seq_len) + 1, (1,)).item()
    start = torch.randint(0, seq_len - span_len + 1, (1,)).item()

    motif_token_mask = torch.zeros(seq_len)
    motif_token_mask[start:start + span_len] = 1

    return motif_token_mask * protein_token_mask


def select_protein_discontiguous_motif(tokenized: Tokenized,
                                       max_discontiguous_res: int,
                                       dist_threshold: float) -> TensorType["n", float]:
    """
    Selects a discontiguous protein motif for the AtomDenoiser.

    TODO: might want to select the motif among protein residues only
    """
    # Select from known protein residues
    protein_token_mask = tokenized.tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
    known_residue_mask = tokenized.tokens["res_type"] != const.token_ids[const.unk_token["PROTEIN"]]
    protein_token_mask = protein_token_mask * known_residue_mask

    # Select discontiguous residues
    seq_len = len(tokenized.tokens)
    motif_token_mask = torch.zeros(seq_len)

    ## we select residues by center coords (CA for proteins, C1' for nucleic acids)
    center_coords = torch.tensor(tokenized.tokens["center_coords"])
    dist = torch.cdist(center_coords, center_coords)

    ## select random residue
    random_residue_idx = torch.randint(0, seq_len, (1,)).item()
    dist_i = dist[random_residue_idx] + 1e5 * (1 - tokenized.tokens["resolved_mask"])  # mask out non-existing atoms
    close_mask = dist_i <= dist_threshold
    n_neighbors = close_mask.sum().int()

    if n_neighbors <= 1:
        # If we have 1 or 0 neighbors, fall back to just using the selected residue
        motif_token_mask[random_residue_idx] = 1
    else:
        # Pick random number of neighbors
        n_to_select = torch.randint(2, min(max_discontiguous_res, n_neighbors) + 1, (1,)).item()

        # Get indices of neighbors (including the original residue)
        neighbor_indices = torch.where(close_mask)[0]
        selected_indices = neighbor_indices[torch.randperm(len(neighbor_indices))[:n_to_select]]
        motif_token_mask[selected_indices] = 1

    motif_token_mask = motif_token_mask * protein_token_mask  # unmask only existing atoms

    return motif_token_mask


##################### Atom-level motif selection #####################
def select_all_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects all atoms for the AtomDenoiser.
    """
    return feats["atom_resolved_mask"]


def select_protein_sidechain_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects protein sidechain atoms for the AtomDenoiser.
    """
    return feats["prot_scn_atom_mask"] * feats["atom_resolved_mask"]


def select_protein_backbone_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects protein backbone atoms for the AtomDenoiser.
    """
    return feats["prot_bb_atom_mask"] * feats["atom_resolved_mask"]


def get_motif_selector(cfg: Optional[DictConfig]) -> Optional[MotifSelector]:
    """
    Get the motif selector specified in the config.
    """
    if (cfg is None) or (cfg.name == "unconditional"):
        return None
    elif cfg.name == "motif_selector":
        return MotifSelector(cfg)
    else:
        raise ValueError(f"Unknown motif selector: {cfg.name}")
