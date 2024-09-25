import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.pdb_utils import *
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.interpolants.sd_interpolants.mar_interpolant import MAR
from allatom_design.interpolants.sd_interpolants.sd_interpolant import SDInterpolant
from allatom_design.model.atom_denoiser.denoisers.denoiser import \
    BaseAtomDenoiser
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import \
    DiTDenoiser
from allatom_design.model.atom_denoiser.denoisers.esm_dit_denoiser import \
    ESMDiTDenoiser


class AtomDenoiser(nn.Module):
    """
    Atom denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.task = cfg.task

        # Data scaling parameters
        # Scale CA separately from rest of the backbone
        self.register_buffer("ca_mean", torch.tensor(0.0))
        self.register_buffer("ca_std", torch.tensor(1.0))

        self.register_buffer("nco_mean", torch.tensor(0.0))
        self.register_buffer("nco_std", torch.tensor(1.0))

        self.sigma_data = (self.ca_std, self.nco_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)
        self.sd_interpolant = get_interpolant(getattr(cfg, "sd_interpolant", None))


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

        # Mask out sequence and sidechains with sd_interpolant
        if self.sd_interpolant is None:
            # don't noise
            batch["x_noised"] = batch["x"]
            batch["aatype_noised"] = batch["aatype"]
            batch["mlm_mask"] = torch.ones_like(batch["seq_mask"]) * batch["seq_mask"]
        else:
            interpolant_out = self.sd_interpolant(batch, t)
            batch["x_noised"] = interpolant_out["x_noised"]
            batch["aatype_noised"] = interpolant_out["aatype_noised"]
            batch["mlm_mask"] = interpolant_out["mlm_mask"]

        # During training, keep track of certain additional features
        aux_inputs = {
            "x": batch["x"],  # ground truth coordinates
            # "add_bos": batch["add_bos"],  # per-example flag of whether to add BOS token
            # "add_eos": batch["add_eos"],  # per-example flag of whether to add EOS token
            "t_ca": batch.get("t_ca", None),  # scalar; fix t_ca if provided, usually for eval
            "t_nco": batch.get("t_nco", None),  # scalar; fix t_nco if provided, usually for eval
        }

        # Denoise coords
        _, aux_preds = self.denoiser(batch["x_noised"], batch["aatype_noised"], None,
                                     batch["residue_index"], batch["seq_mask"], batch["mlm_mask"],
                                     cond_labels_in=batch["cond_labels_in"],
                                     aux_inputs=aux_inputs)

        # Additional outputs for computing loss
        outputs.update(aux_preds)

        return outputs


    def set_scale_factors(self,
                          scale_factors: Dict[str, Tuple[float, float]]):
        ca_mean, ca_std = scale_factors["ca"]
        self.ca_mean.data = torch.tensor(ca_mean)
        self.ca_std.data = torch.tensor(ca_std)
        print(f"Setting ca_mean: {ca_mean}, ca_std: {ca_std}")

        nco_mean, nco_std = scale_factors["nco"]
        self.nco_mean.data = torch.tensor(nco_mean)
        self.nco_std.data = torch.tensor(nco_std)
        print(f"Setting nco_mean: {nco_mean}, nco_std: {nco_std}")


    def sample(self,
               lengths: TensorType["b", int],
               residue_index: TensorType["b n", int],
               timesteps: Tuple[TensorType["b s+1", float]],  # tuple of timesteps for (t_ca, t_nco)
               xt_override: Optional[TensorType["s+1 b n a 3", float]] = None,
               xt_override_mask: Optional[TensorType["s+1 b n a 3", float]] = None,
               aatype_override: Optional[TensorType["s+1 b n", int]] = None,
               aatype_override_mask: Optional[TensorType["s+1 b n", int]] = None,
               cond_labels: Dict[str, TensorType["b", int]] = {},
               noise_schedule: Tuple[Optional[NoiseSchedule]] = None,  # noise schedule for (t_ca, t_nco)
               churn_cfg: Tuple[Optional[Dict[str, float]]] = None, # churn config for (t_ca, t_nco)
               ) -> Tuple[TensorType["b n 4 3", float],
                          Dict[str, torch.Tensor]]:
        """
        Sample from the model.

        Returns the final denoised coords and auxiliary outputs.

        aux includes:
        - seq_mask: TensorType["b n", float]
        - x1_traj: TensorType["s b n a 3", float], s=num_steps
        - xt_traj: TensorType["s b n a 3", float], s=num_steps

        Sampling parameters:
        - xt_override: override coords at each step with this tensor where xt_override_mask is 1.
        - churn_cfg contains:
            - s_churn: controls overall amount of stochasticity to add in sampling
            - s_noise: std of noise to add with churn
        - cond_labels: dictionary mapping from conditioning label to token ID for each batch element
        """
        B, N = residue_index.shape

        aux = {}  # keep track of auxiliary outputs

        # Create seq mask
        ranges = torch.arange(N, device=residue_index.device).expand(B, N)
        seq_mask = (ranges < lengths[:, None]).float()
        aux["seq_mask"] = seq_mask.cpu()

        # Initialize sequence prior (all masked)
        aatype_noised = torch.full_like(residue_index, fill_value=self.restype_order_with_x["X"])
        mlm_mask = torch.zeros_like(seq_mask)
        aux["mlm_mask"] = mlm_mask

        # Make unimodal timesteps into a tuple for consistency
        if not isinstance(timesteps, (tuple, list)):
            timesteps = (timesteps,)

        num_steps = timesteps[0].shape[-1] - 1

        # Handle xt overrides
        if xt_override is None:
            # dummy values
            xt_override = torch.zeros(1, device=residue_index.device).expand(num_steps + 1, B, N, rc.atom_type_num, 3)
            xt_override_mask = torch.zeros(1, device=residue_index.device).expand(num_steps + 1, B, N, rc.atom_type_num, 3)

        # Handle aatype overrides
        if aatype_override is None:
            # dummy values
            aatype_override = torch.full((), fill_value=rc.restype_order_with_x["X"], device=residue_index.device).expand(num_steps + 1, B, N)
            aatype_override_mask = torch.zeros(1, device=residue_index.device, dtype=torch.long).expand(num_steps + 1, B, N)

        # Construct denoiser inputs
        aux_inputs = {
            "num_steps": num_steps,
            "timesteps": timesteps,
            "churn_cfg": churn_cfg,
            "noise_schedule": noise_schedule,
            # overrides
            "xt_override": xt_override,
            "xt_override_mask": xt_override_mask,
            "aatype_override": aatype_override,
            "aatype_override_mask": aatype_override_mask,
        }
        x1_bb, aux_preds = self.denoiser(x_noised=None, aatype_noised=aatype_noised, t=None, residue_index=residue_index,
                                         seq_mask=seq_mask, mlm_mask=mlm_mask, cond_labels_in=cond_labels, aux_inputs=aux_inputs, is_sampling=True)

        aux.update(aux_preds["bb_diffusion_aux"])
        return x1_bb, aux


    @staticmethod
    def save_samples_to_pdb(samples: Dict[str, TensorType["b ..."]],
                            filenames: List[str]
                            ) -> None:
        """
        Save samples from the denoiser to PDB files.

        Samples should contain the following keys:
        - x_bb_denoised: Tensor["b n 4 3", float]
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        """
        B, N, _, _= samples["x_bb_denoised"].shape
        residue_index = samples["residue_index"]
        seq_mask = samples["seq_mask"]

        x_denoised = samples["x_bb_denoised"]

        aatype = torch.full_like(residue_index, fill_value=rc.restype_order["G"], dtype=torch.long)  # force aatype to glycine
        final_atom37_positions = torch.zeros((B, N, 37, 3), device=aatype.device)
        final_atom37_positions[:, :, rc.bb_idxs, :] = x_denoised
        atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=aatype.device)[aatype] * seq_mask[..., None]

        feats = {
            "aatype": aatype,
            "atom_positions": final_atom37_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": torch.zeros_like(residue_index),  # TODO: support multiple chains
            "b_factors": torch.ones_like(atom_mask, dtype=torch.float32),
        }

        feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}  # move to cpu
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")


    @staticmethod
    def save_trajs_to_pdb(traj_aux: Dict[str, Any],
                          residue_index: TensorType["b n", int],
                          chain_index: TensorType["b n", int],
                          save_traj_mask: List[bool],
                          save_traj_steps: List[int],
                          x_traj_key: str,
                          filenames: List[str],
                          traj_conect: bool,
                          align_models_to_idx: Optional[int] = None):
        """
        Save trajectories from the denoiser to PDB files.

        Args:
        - traj_aux: auxiliary output from sampling trajectory
        - residue_index
        - chain_index
        - save_traj_mask: list of bools indicating which main trajectories to save
        - save_traj_steps: list of indices indicating which steps along the main trajectory to save
        - x_traj_key: key in traj_aux for the denoised atom positions
            - "x1_bb_traj" gives the x1 prediction along the backbone diffusion trajectory
            - "xt_bb_traj" gives the current state along the backbone diffusion trajectory
        - filenames: list of filenames to save the trajectories to
        - traj_conect: whether to include CONECT records in the PDB files
        - align_models_to_idx: if not None, align all models to the model at this index
        """
        B = traj_aux["seq_mask"].shape[0]
        device = traj_aux["seq_mask"].device
        for i in range(B):
            if save_traj_mask[i]:
                if x_traj_key in ["x1_bb_traj", "xt_bb_traj"]:
                    # Save x1 or xt traj
                    aatype_i = torch.full_like(traj_aux["seq_mask"][i], fill_value=rc.restype_order["G"], dtype=torch.long)  # force aatype to glycine
                    aatype_traj = aatype_i.unsqueeze(0).expand(len(save_traj_steps), -1)
                    atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, N, A]
                    x_bb_traj = traj_aux[x_traj_key][i, save_traj_steps]
                else:
                    assert False, f"Unknown x_traj_key: {x_traj_key}"

                # Put backbone positions into atom37 format
                S, N, _, X = x_bb_traj.shape
                x_traj = torch.zeros((S, N, rc.atom_type_num, 3), device=device)
                x_traj[:, :, rc.bb_idxs, :] = x_bb_traj

                traj_feats = {
                    "aatype": aatype_traj,
                    "atom_positions": x_traj,
                    "atom_mask": atom_mask,
                    "residue_index": residue_index[i].unsqueeze(0).expand(aatype_traj.shape[0], -1),
                    "chain_index": chain_index[i].unsqueeze(0).expand(aatype_traj.shape[0], -1),
                    "b_factors": None
                }
                traj_feats = {k: v.cpu() if v is not None else v for k, v  in traj_feats.items()}
                write_to_pdb_frames(**traj_feats, filename=filenames[i], mode="aa", conect=traj_conect, align_models_to_idx=align_models_to_idx)


def get_denoiser(cfg: DictConfig,
                 sigma_data: TensorType[(), float]  # can also be a tuple of sigmas for ca, nco
                 ) -> BaseAtomDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "dit":
        return DiTDenoiser(cfg, sigma_data)
    elif cfg.name == "esm_dit":
        return ESMDiTDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


def get_interpolant(cfg: Optional[DictConfig]) -> SDInterpolant:
    """
    Get the interpolant specified in the config.
    """
    if cfg is None:
        return None
    elif cfg.name == "mar":
        return MAR(cfg)
    else:
        raise ValueError(f"Unknown interpolant: {cfg.name}")
