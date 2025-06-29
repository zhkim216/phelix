import copy
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data.pdb_utils import *
from allatom_design.model.seq_denoiser.denoisers.atom_mpnn_denoiser import \
    AtomMPNNDenoiser
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser


class SeqDenoiser(nn.Module):
    """
    Sequence denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task

        # Data scaling parameters
        # scale sidechains separately from the backbone
        self.register_buffer("bb_std", torch.tensor(1.0))
        self.register_buffer("bb_mean", torch.tensor(0.0))

        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = (self.bb_std, self.scn_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)

        # Mask selector
        self.mask_selector = cfg.mask_selector


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                t: TensorType["b", float] | None = None
                ) -> dict[str, TensorType["b ..."]]:
        outputs = {}

        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)

        with torch.no_grad():
            # Sample sequence and atom conditioning masks
            batch["seq_cond_mask"] = self.mask_selector.sample_seq_cond_mask(batch, t)  # 1 if we should condition on the restype, 0 otherwise
            batch["atom_cond_mask"] = self.mask_selector.sample_atom_cond_mask(batch)  # 1 if we should condition on the atom, 0 otherwise

            # Ensure the conditioning masks only contain non-pad, resolved entries
            batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_pad_mask"] * batch["token_resolved_mask"]
            batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        # Denoise sequence
        _, aux_preds = self.denoiser(batch)

        # Additional outputs for computing loss
        outputs.update(aux_preds)

        return outputs


    def set_scale_factors(self,
                          scale_factors: Dict[str, Tuple[float, float]]):
        bb_mean, bb_std = scale_factors["bb"]
        self.bb_mean.data = torch.tensor(bb_mean)
        self.bb_std.data = torch.tensor(bb_std)
        print(f"Setting bb_mean: {bb_mean}, bb_std: {bb_std}")

        scn_mean, scn_std = scale_factors["scn"]
        self.scn_mean.data = torch.tensor(scn_mean)
        self.scn_std.data = torch.tensor(scn_std)
        print(f"Setting scn_mean: {scn_mean}, scn_std: {scn_std}")


    def sample(self,
               batch: dict[str, TensorType["b ..."]],
               sampling_inputs: dict[str, Any]):

        # Handle inference noise labels
        batch["noise_labels"] = sampling_inputs.get("noise_labels", None)
        batch["noise"] = None

        if batch["noise_labels"] is not None:
            raise NotImplementedError("Noise labels are not implemented yet")

        if sampling_inputs["add_noise"]:
            raise NotImplementedError("Adding noise is not implemented yet")

        if sampling_inputs.get("t", None) is not None:
            batch["t"] = torch.full((batch["token_pad_mask"].shape[0],), fill_value=sampling_inputs["t"], device=batch["token_pad_mask"].device)

        # Choose sampling method
        if sampling_inputs["use_potts_sampling"]:
            output_feats, aux = self.denoiser.potts_sample(batch, sampling_inputs)

        return output_feats, aux


def get_denoiser(cfg: DictConfig,
                 sigma_data: TensorType[(), float]
                 ) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "fampnn" or cfg.name == "minimpnn":  # backwards compatibility
        from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
            FAMPNNDenoiser
        return FAMPNNDenoiser(cfg, sigma_data)
    elif cfg.name == "atom_mpnn":
        return AtomMPNNDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


def truncated_half_normal_like(x: TensorType["...", float],
                               std: float, max_val: Optional[float]) -> TensorType["...", float]:
    if max_val is None:
        # return half-normal with no truncation
        return torch.abs(torch.randn_like(x) * std)
    u = torch.rand_like(x)
    truncated_factor = torch.erf(torch.tensor(max_val / (math.sqrt(2) * std)))
    u_scaled = u * truncated_factor
    samples = std * math.sqrt(2) * torch.erfinv(u_scaled)
    return samples
