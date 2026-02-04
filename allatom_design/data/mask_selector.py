from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType

from allatom_design.data import const


class MaskSelector:
    def __init__(self, cfg: DictConfig):
        """
        Handles selecting masks for training the sequence design model.
        """
        super().__init__()
        self.cfg = cfg

        self.restype_masking_schedule = cfg.restype_masking_schedule
        self.restype_masking_cfg = OmegaConf.to_container(cfg.restype_masking_cfg[self.restype_masking_schedule], resolve=True)  # to dict to avoid dataloader issues?
        self.atom_masking_schedule = cfg.atom_masking_schedule
        self.atom_masking_cfg = OmegaConf.to_container(cfg.atom_masking_cfg[self.atom_masking_schedule], resolve=True)  
        self.scn_context_ratio = cfg.scn_context_ratio
        

    def sample_seq_cond_mask(self,
                             batch: dict[str, TensorType["b ..."]],
                             t: TensorType["b", float] | None = None
                             ) -> TensorType["b n_tokens", float]:
        """
        Create a mask denoting which restypes to mask out.
        0 if we should mask, 1 if we should keep. Non-protein restypes are always kept (1).
        """
        B, N = batch["token_pad_mask"].shape
        device = batch["token_pad_mask"].device

        if t is None:
            # Sample timestep
            t = self._sample_t(B, device=device, schedule=self.restype_masking_schedule, cfg=self.restype_masking_cfg)

        # Create mask based on timestep
        seq_cond_mask = torch.rand(B, N, device=device) < rearrange(t, "b -> b 1")

        # Non-protein and non-standard restypes are always kept
        standard_aa_prot_token_mask = batch["token_is_prot_std_aa"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        
        seq_cond_mask = torch.where(~standard_aa_prot_token_mask.bool(),
                                    torch.ones_like(seq_cond_mask),
                                    seq_cond_mask)

        seq_cond_mask = seq_cond_mask * batch["token_pad_mask"] * batch["token_resolved_mask"]  # mask out padding, non-resolved entries
        
        
        return seq_cond_mask


    def sample_atom_cond_mask(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n_atoms", float]:
        """
        Create a mask denoting which atoms to mask out.
        0 if we should mask, 1 if we should keep.
        """
        B, _ = batch["atom_to_token_map"].shape
        device = batch["atom_resolved_mask"].device

        atom_cond_mask = batch["atom_resolved_mask"].clone()  # [n_atoms]
                
        # # Mask out the sidechain of a token with probability p in U[0, 1]  # TODO: try different masking schemes
        # tok_keep_scn_p = self._sample_t(B, device=device, schedule=self.atom_masking_schedule, cfg=self.atom_masking_cfg)
        # tok_keep_scn_mask = torch.rand_like(batch["seq_cond_mask"]) < rearrange(tok_keep_scn_p, "b -> b 1")
        
        # Following LigandMPNN, sample sidechains only scn_context_ratio
        standard_aa_prot_token_mask = batch["token_is_prot_std_aa"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        standard_aa_prot_atom_mask = batch["atom_is_prot_std_aa"] * batch["atom_resolved_mask"] * batch["atom_pad_mask"]
        target_count = (standard_aa_prot_token_mask.sum(dim=-1) * self.scn_context_ratio).long()
        
        random_priority = torch.where(
            standard_aa_prot_token_mask.bool(),
            torch.rand_like(standard_aa_prot_token_mask.float()),
            torch.full_like(standard_aa_prot_token_mask.float(), fill_value=-float("inf"))
        )
        rank = random_priority.argsort(dim=-1, descending=True).argsort(dim=-1)        
        tok_keep_scn_mask = standard_aa_prot_token_mask * (rank < target_count.unsqueeze(-1)).float()                              
        atomwise_tok_keep_scn_mask = tok_keep_scn_mask.gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        
        standard_aa_prot_bb_atom_mask = standard_aa_prot_atom_mask * batch["prot_bb_atom_mask"]
        standard_aa_prot_scn_wo_cb_atom_mask = standard_aa_prot_atom_mask * batch["prot_scn_wo_cb_atom_mask"] 
        #! (JH) Following LigandMPNN, we mask out the CB atoms of the sidechain, as we use pseudo CB coordinates in the ligand & interaction module
        
        # Select sidechain atoms or backbone atoms from the standard amino acids in protein chains
        prot_atom_mask = torch.where(atomwise_tok_keep_scn_mask.bool(),
                                     standard_aa_prot_scn_wo_cb_atom_mask,
                                     standard_aa_prot_bb_atom_mask)
        prot_atom_mask = prot_atom_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
                
        # Keep all atoms in non-protein chains or atoms in non-standard residues or covalent modifications in protein chains
        atom_cond_mask = torch.where(standard_aa_prot_atom_mask.bool(),
                                     prot_atom_mask,
                                     atom_cond_mask)
                    
        atom_cond_mask = atom_cond_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        
        return atom_cond_mask    
            
    def _sample_t(
        self, 
        B: int = None, 
        device: torch.device = None,
        schedule: str = None,
        cfg: dict = None,
    ) -> TensorType["b", float]:
        """
        Sample a timestep from the masking schedule.
        t = probability of keeping the restype unmasked
        
        Args:
            B: batch size
            device: torch device
            schedule: masking schedule name (defaults to restype_masking_schedule)
            cfg: masking config dict (defaults to restype_masking_cfg)
        """
        # Use defaults if not provided (backward compatible)
        if schedule is None:
            schedule = self.restype_masking_schedule
        if cfg is None:
            cfg = self.restype_masking_cfg
            
        if schedule == "constant_t":
            t = torch.ones(B, device=device) * cfg["t"]
        elif schedule.startswith("uniform"):
            # sample time from uniform distribution
            t_min, t_max = cfg["t_min"], cfg["t_max"]
            t = torch.rand(B, device=device) * (t_max - t_min) + t_min

            # apply transformation to t
            if schedule == "uniform_t":
                t = t
            elif schedule == "uniform_squared_t":
                t = t ** 2
            elif schedule == "uniform_cubed_t":
                t = t ** 3
            elif schedule == "uniform_cosine_t":
                t = 1 - torch.cos(t * np.pi / 2)
            elif schedule == "uniform_sqrt_t":
                t = t ** 0.5
            elif schedule == "uniform_cbrt_t":
                t = t ** (1/3)
        else:
            raise ValueError(f"Unknown masking schedule: {schedule}")

        return t


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


def get_mask_selector(cfg: Optional[DictConfig]) -> Optional[MaskSelector]:
    """
    Get the mask selector specified in the config.
    """
    if cfg.name == "mask_selector":
        return MaskSelector(cfg)
    else:
        raise ValueError(f"Unknown mask selector: {cfg.name}")
