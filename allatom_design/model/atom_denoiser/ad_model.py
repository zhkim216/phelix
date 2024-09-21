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
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_ca_interpolant import \
    EDM_CA
from allatom_design.model.atom_denoiser.denoisers.denoiser import Denoiser
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import \
    DiTDenoiser


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

        # Noised inputs (for now we don't noise)
        batch["x_noised"] = batch["x"]
        batch["aatype_noised"] = batch["aatype"]

        # During training, keep track of certain additional features
        aux_inputs = {
            "x": batch["x"],  # ground truth coordinates
            "t_ca": batch.get("t_ca", None),  # scalar; fix t_ca if provided, usually for eval
            "t_nco": batch.get("t_nco", None),  # scalar; fix t_nco if provided, usually for eval
        }

        # Denoise coords
        x1_pred, aux_preds = self.denoiser(batch["x_noised"], batch["aatype_noised"], None,
                                           batch["residue_index"], batch["seq_mask"],
                                           cond_labels_in=batch["cond_labels_in"],
                                           aux_inputs=aux_inputs)

        outputs["x1_pred"] = x1_pred

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
            #    timesteps: TensorType["b s+1", float],  # can also be 2-tuple [b s] for multimodal, in order of (t_bb, t_scn)
            # CHANGE TO override_timesteps?
               res_decoding_order_mode: str,
               xt_override: Optional[TensorType["s+1 b n a 3", float]] = None,
               xt_override_mask: Optional[TensorType["s+1 b n a 3", float]] = None,
               aatype_override: Optional[TensorType["s+1 b n", int]] = None,  # for fixed-sequence sampling, e.g. in sidechain packing
               aatype_override_mask: Optional[TensorType["s+1 b n", int]] = None,
               cond_labels: Dict[str, TensorType["b", int]] = {},
               churn_cfg: Optional[Dict[str, float]] = None,
               scn_diff_aux_inputs: Dict[str, Any] = {},  # sampling config for scn diffusion head (num_steps)
               bb_diff_aux_inputs: Dict[str, Any] = {}  # sampling config for bb diffusion head (num_steps)
               ) -> Tuple[TensorType["b n a 3", float],
                          Dict[str, torch.Tensor]]:
        """
        Sample from the model.

        Returns the final denoised coords and auxiliary outputs.

        aux includes:
        - seq_mask: TensorType["b n", float]
        - x1_traj: TensorType["s b n a 3", float], s=num_steps
        - xt_traj: TensorType["s b n a 3", float], s=num_steps
        - seq_logits_traj: TensorType["s b n a", float], s=num_steps
        - pred_aatype_traj: TensorType["s b n", int], s=num_steps


        Sampling parameters:
        - xt_override: override coords at each step with this tensor where xt_override_mask is 1.
        - churn_cfg contains:
            - s_churn: controls overall amount of stochasticity to add in sampling
            - s_noise: std of noise to add with churn
        - cond_labels: dictionary mapping from conditioning label to token ID for each batch element
        """
        B, N, A = *residue_index.shape, self.cfg.num_atoms_in

        aux_inputs = {}  # construct auxiliary inputs for steps
        aux = {}  # keep track of auxiliary outputs

        # Create sequence mask
        ranges = torch.arange(N, device=residue_index.device).expand(B, N)
        seq_mask = (ranges < lengths[:, None]).float()
        aux["seq_mask"] = seq_mask.cpu()

        # Sample priors
        x0 = self.interpolant.sample_prior((B, N, A, 3), device=residue_index.device) * rearrange(seq_mask, "b n -> b n 1 1")
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()  # TODO: make seq prior use MASK rather than UNK

        # Get residue decoding order
        res_decoding_order = sampling_utils.get_res_decoding_order(mode=res_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps)
        aux_inputs["res_decoding_order"] = res_decoding_order

        aux_inputs["lengths"] = lengths
        mlm_mask = torch.zeros_like(seq_mask).float()  # start with all masked tokens
        aux_inputs["mlm_mask"] = (mlm_mask.clone(), mlm_mask.clone())  # multimodal MLM mask

        # Initialize trajectories
        xt_traj, aatype_t_traj = [], []  # keep track of denoised coords and aatype trajectory
        x1_traj, aatype_pred_traj = [], []  # keep track of x1 and aatype prediction trajectory
        seq_logits_traj = []  # keep track of sequence logits trajectory
        bb_diffusion_aux_all, scn_diffusion_aux_all = [], []  # keep track of all diffusion trajectories if "return_traj" is True in diff_aux_inputs

        # Make unimodal timesteps into a tuple for consistency
        if not isinstance(timesteps, (tuple, list)):
            timesteps = (timesteps,)

        num_steps = timesteps[0].shape[-1] - 1

        # Handle xt overrides
        if xt_override is None:
            # dummy values
            xt_override = torch.zeros((num_steps + 1, B, N, A, 3), device=residue_index.device)
            xt_override_mask = torch.zeros((num_steps + 1, B, N, A, 3), device=residue_index.device)  # don't override anything

        # Handle aatype overrides
        if aatype_override is None:
            # dummy values
            aatype_override = torch.full((num_steps + 1, B, N), fill_value=rc.restype_order_with_x["X"], device=residue_index.device)
            aatype_override_mask = torch.zeros((num_steps + 1, B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Provide sidechain diffusion and backbone diffusion config
        aux_inputs.update(scn_diff_aux_inputs)
        aux_inputs.update(bb_diff_aux_inputs)

        # Run integration steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              cond_labels_in=cond_labels,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        aux_preds = None
        for i in tqdm(range(num_steps), leave=False, desc="Sampling..."):
            # get current and next timesteps, squeezing if unimodal
            t = tuple(ts[:, i] for ts in timesteps) if len(timesteps) > 1 else timesteps[0][:, i]
            t_next = tuple(ts[:, i + 1] for ts in timesteps) if len(timesteps) > 1 else timesteps[0][:, i + 1]

            xt, t = self.interpolant.churn(xt, t, churn_cfg=churn_cfg)  # Karras et al. stochastic sampling

            xt = xt * (1 - xt_override_mask[i]) + xt_override[i] * xt_override_mask[i]  # override xt for inputs  # TODO: can we move this into denoiser or wrap this somewhere?
            aatype_t = aatype_t * (1 - aatype_override_mask[i]) + aatype_override[i] * aatype_override_mask[i]  # override aatype for inputs
            xt, aatype_t, aux_preds = self.interpolant.euler_step(denoiser_fn,
                                                                  xt, aatype_t,
                                                                  t=t, t_next=t_next,
                                                                  cfg_cfg=None,
                                                                  noise_schedule=None,
                                                                  aux_inputs=aux_inputs
                                                                  )

            xt = xt * (1 - xt_override_mask[i + 1]) + xt_override[i + 1] * xt_override_mask[i + 1]  # override xt for outputs  # TODO: should we override self-cond input too?
            aatype_t = aatype_t * (1 - aatype_override_mask[i + 1]) + aatype_override[i + 1] * aatype_override_mask[i + 1]  # override aatype for outputs

            # Save current denoised coords and aatype
            xt_traj.append(xt.cpu())
            aatype_t_traj.append(aatype_t.cpu())

            # Save current auxiliary outputs
            seq_logits_traj.append(aux_preds["seq_logits"].cpu())

            # Save current x1 prediction
            x1_traj.append(aux_preds["x1_pred"].cpu())
            aatype_pred_traj.append(aux_preds["aatype_pred"].cpu())

            # Save backbone and sidechain diffusion outputs
            if bb_diff_aux_inputs.get("return_traj", False):
                bb_diffusion_aux_all.append(aux_preds["bb_diffusion_aux"])

            if scn_diff_aux_inputs.get("return_traj", False):
                scn_diffusion_aux_all.append(aux_preds["scn_diffusion_aux"])

        aux["x1_traj"] = torch.stack(x1_traj, dim=0)
        aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=0)
        aux["xt_traj"] = torch.stack(xt_traj, dim=0)
        aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=0)
        aux["seq_logits_traj"] = torch.stack(seq_logits_traj, dim=0)
        aux["pred_aatype"] = aatype_t_traj[-1]

        aux["bb_diffusion_aux"], aux["scn_di∏ffusion_aux"] = None, None
        if bb_diff_aux_inputs.get("return_traj", False):
            aux["bb_diffusion_aux"] = {k: torch.stack([d[k] for d in bb_diffusion_aux_all], dim=0) for k in bb_diffusion_aux_all[0]}
        if scn_diff_aux_inputs.get("return_traj", False):
            aux["scn_diffusion_aux"] = {k: torch.stack([d[k] for d in scn_diffusion_aux_all], dim=0) for k in scn_diffusion_aux_all[0]}

        return xt, aux


def get_denoiser(cfg: DictConfig,
                 sigma_data: TensorType[(), float]  # can also be a tuple of sigmas for ca, nco
                 ) -> Denoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "dit":
        return DiTDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


