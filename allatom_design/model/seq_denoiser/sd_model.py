import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.data.pdb_utils import *
from allatom_design.interpolants.sd_interpolants.sd_interpolant import \
    SDInterpolant
from allatom_design.interpolants.sd_interpolants.mar_interpolant import \
    MAR
from allatom_design.model.seq_denoiser.denoisers.denoiser import BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.minimpnn_denoiser import MiniMPNNDenoiser


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
        self.register_buffer("bb_mean", torch.tensor(0.0))
        self.register_buffer("bb_std", torch.tensor(1.0))

        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = (self.bb_std, self.scn_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)
        self.interpolant = get_interpolant(cfg.interpolant)


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,  # (t_bb, t_scn) if multimodal
                ) -> Dict[str, TensorType["b ..."]]:
        """
        batch should contain:
        - x: TensorType["b n a 3", float]
        - residue_index: TensorType["b n", int]
        - seq_mask: TensorType["b n", float]
        - cond_labels_in: Dict[str, TensorType["b", int]]
        """
        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)
        outputs = {}

        # Apply interpolant to noise the inputs
        interpolant_out = self.interpolant(batch, t)
        batch["x_noised"] = interpolant_out["x_noised"]
        batch["aatype_noised"] = interpolant_out["aatype_noised"]

        # During training, keep track of certain additional features
        aux_inputs = {
            "x": batch["x"],  # ground truth coordinates
            "aatype": batch["aatype"],  # ground truth aatype
            "ghost_atom_mask": batch["ghost_atom_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "t_sd": batch.get("t_scn_diff", None),  # scalar; fix t_sd (sidechain diffusion time) if provided, usually for eval
        }

        # Denoise coords
        _, _, aux_preds = self.denoiser(batch["x_noised"], batch["aatype_noised"], None,
                                           batch["residue_index"], batch["seq_mask"],
                                           cond_labels_in=batch["cond_labels_in"],
                                           aux_inputs=aux_inputs)

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


def get_denoiser(cfg: DictConfig,
                 sigma_data: TensorType[(), float]  # can also be a tuple of sigmas for ca, nco
                 ) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "minimpnn":
        return MiniMPNNDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


def get_interpolant(cfg: DictConfig) -> SDInterpolant:
    """
    Get the interpolant specified in the config.
    """
    if cfg.name == "mar":
        return MAR(cfg)
    else:
        raise ValueError(f"Unknown interpolant: {cfg.name}")
