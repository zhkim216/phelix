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
        self.pseudo_ligand_backbone_mask_radius = cfg.get("pseudo_ligand_backbone_mask_radius", 0)  # JH Changed 260415


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


    # JH Changed 260415 — Pseudo-ligand sidechain conditioning via Y track
    def sample_atom_cond_mask(
        self, batch: dict[str, TensorType["b ..."]]
    ) -> tuple[TensorType["b n_atoms", float], TensorType["b n_tokens", float], TensorType["b n_tokens", float]]:
        """
        Create atom-level conditioning mask and pseudo-ligand token masks.

        Returns:
            atom_cond_mask:          [B, n_atoms]  — 1 = keep atom, 0 = mask
            tok_keep_scn_mask:       [B, n_tokens] — 1 = pseudo-ligand position (sidechain visible, removed from protein graph)
            expanded_backbone_mask:  [B, n_tokens] — 1 = backbone-masked neighbor of pseudo-ligand (also removed from protein graph)
        """
        B, _ = batch["atom_to_token_map"].shape
        device = batch["atom_resolved_mask"].device

        atom_cond_mask = batch["atom_resolved_mask"].clone()

        standard_aa_prot_token_mask = batch["token_is_prot_std_aa"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        standard_aa_prot_atom_mask = batch["atom_is_prot_std_aa"] * batch["atom_resolved_mask"] * batch["atom_pad_mask"]

        # --- Token eligibility: must have at least one can_be_pseudo_ligand atom ---
        can_be_pl = batch.get("can_be_pseudo_ligand", torch.zeros_like(batch["atom_resolved_mask"]))
        token_eligible = torch.zeros(B, batch["token_pad_mask"].shape[1], device=device)
        token_eligible.scatter_reduce_(1, batch["atom_to_token_map"], can_be_pl, reduce="amax", include_self=False)
        token_eligible = standard_aa_prot_token_mask * (token_eligible > 0).float()

        # --- Select scn_context_ratio fraction from eligible tokens ---
        target_count = (token_eligible.sum(dim=-1) * self.scn_context_ratio).long()
        random_priority = torch.where(
            token_eligible.bool(),
            torch.rand_like(token_eligible),
            torch.full_like(token_eligible, -float("inf"))
        )
        rank = random_priority.argsort(dim=-1, descending=True).argsort(dim=-1)
        tok_keep_scn_mask = token_eligible * (rank < target_count.unsqueeze(-1)).float()

        # --- Expand to ±radius sequential neighbors for backbone masking ---
        expanded_backbone_mask = self._expand_backbone_mask(tok_keep_scn_mask, batch, self.pseudo_ligand_backbone_mask_radius)

        # --- Build atom_cond_mask ---
        atomwise_tok_keep_scn_mask = tok_keep_scn_mask.gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        atomwise_expanded_bb_mask = expanded_backbone_mask.gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        standard_aa_prot_bb_atom_mask = standard_aa_prot_atom_mask * batch["prot_bb_atom_mask"]
        standard_aa_prot_scn_wo_cb_atom_mask = standard_aa_prot_atom_mask * batch["prot_scn_wo_cb_atom_mask"]

        # Pseudo-ligand tokens: sidechain excl CB visible, backbone hidden
        # Expanded neighbor tokens: ALL atoms hidden (prevent backbone leaking into ligand context)
        # Normal tokens: backbone visible, sidechain hidden
        prot_atom_mask = torch.where(
            atomwise_tok_keep_scn_mask.bool(),
            standard_aa_prot_scn_wo_cb_atom_mask,                # pseudo-ligand: sidechain only
            torch.where(
                atomwise_expanded_bb_mask.bool(),
                torch.zeros_like(standard_aa_prot_bb_atom_mask), # neighbor: nothing
                standard_aa_prot_bb_atom_mask                    # normal: backbone
            )
        )
        prot_atom_mask = prot_atom_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        atom_cond_mask = torch.where(standard_aa_prot_atom_mask.bool(), prot_atom_mask, atom_cond_mask)
        atom_cond_mask = atom_cond_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        return atom_cond_mask, tok_keep_scn_mask, expanded_backbone_mask

    @staticmethod
    def _expand_backbone_mask(
        tok_keep_scn_mask: TensorType["b n", float],
        batch: dict[str, TensorType["b ..."]],
        radius: int,
    ) -> TensorType["b n", float]:
        """Expand pseudo-ligand mask to ±radius sequential neighbors on the same chain.

        Returns a mask that is 1 for neighbor positions (NOT including the pseudo-ligand
        itself), 0 elsewhere.
        """  # JH Changed 260415
        if radius == 0:
            return torch.zeros_like(tok_keep_scn_mask)

        residue_index = batch["residue_index"]  # [B, N]
        asym_id = batch["asym_id"]              # [B, N]

        # Pairwise same-chain check and residue distance
        same_chain = (asym_id.unsqueeze(2) == asym_id.unsqueeze(1))           # [B, N, N]
        res_dist = (residue_index.unsqueeze(2) - residue_index.unsqueeze(1)).abs()  # [B, N, N]
        within_radius = same_chain & (res_dist <= radius) & (res_dist > 0)    # [B, N, N], exclude self

        # For each position, check if ANY pseudo-ligand is within radius on the same chain
        expanded = torch.einsum("bnm,bm->bn", within_radius.float(), tok_keep_scn_mask)
        expanded = (expanded > 0).float()

        # Exclude positions that are already pseudo-ligand themselves
        expanded = expanded * (1 - tok_keep_scn_mask)

        return expanded
            
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
