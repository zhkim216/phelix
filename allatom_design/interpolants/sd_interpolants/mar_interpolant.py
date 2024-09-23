from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.sd_interpolants.sd_interpolant import SDInterpolant


class MAR(SDInterpolant):
    def __init__(self, cfg: DictConfig):
        """
        Interpolant for masked autoregressive diffusion on sequence and sidechains.
        """
        super().__init__()
        self.cfg = cfg

        # Training noise distribution
        self.training_noise_schedule = cfg.training_noise_schedule
        assert self.training_noise_schedule in ["uniform_t", "constant_t", "uniform_squared_t"], f"Unknown timestep schedule: {self.timestep_schedule}"

        self.training_noise_cfg = cfg.training_noise_cfg[self.training_noise_schedule]


    @torch.compiler.disable
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                x0: Optional[TensorType["b n a 3"]] = None,
                ) -> Dict[str, Any]:
        x1 = batch["x"]

        # Sample time steps if not provided
        if t is None:
            t = self.sample_timestep(x1.shape[0], device=x1.device)

        # Get noisy samples
        xt, aatype_noised, mlm_mask = self.noise_samples(x1, batch["aatype"], t, batch["seq_mask"])

        # Construct outputs
        outputs = {}
        outputs["t"] = t  # [b]
        outputs["x_noised"] = xt  # [b n a 3]
        outputs["aatype_noised"] = aatype_noised  # [b n]
        outputs["mlm_mask"] = mlm_mask  # [b n]

        return outputs


    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        """
        Sample a batch of b timesteps from the noise schedule.

        - uniform_t: sample time from uniform distribution
        - constant_t: sample time from constant distribution
        """
        if self.training_noise_schedule == "uniform_t":
            t_min, t_max = self.training_noise_cfg.t_min, self.training_noise_cfg.t_max
            t = torch.rand(n, device=device) * (t_max - t_min) + t_min
        elif self.training_noise_schedule == "constant_t":
            t = torch.ones(n, device=device) * self.training_noise_cfg.t
        elif self.training_noise_schedule == "uniform_squared_t":
            t_min, t_max = self.training_noise_cfg.t_min, self.training_noise_cfg.t_max
            t = torch.rand(n, device=device) * (t_max - t_min) + t_min
            t = t ** 2

        return t


    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        """
        Sample n samples from the prior.
        """
        return torch.zeros(*shape, device=device)


    def noise_samples(self,
                      x: TensorType["b n a 3"],
                      aatype: TensorType["b n", int],
                      t: TensorType["b"],
                      seq_mask: TensorType["b n"],
                      ) -> Tuple[TensorType["b n a 3", float],
                                 TensorType["b n", int],
                                 TensorType["b n", int]]:
        """
        Add noise to x and aatype.

        For MAR, we keep each residue with probability t.
        When masking residues, we zero out the coordinates and set aatype to X.
        """
        B, N, _, _ = x.shape
        mlm_mask = torch.rand(B, N, device=x.device) < rearrange(t, "b -> b 1")  # 1 if we keep the residue, 0 if we mask it
        mlm_mask = (mlm_mask * seq_mask).float()  # mask out residues that are not in the sequence

        x_noised = x.clone()
        x_noised[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(mlm_mask, "b n -> b n 1 1").float()
        aatype_noised = torch.where(mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])  # TODO: replace with MASK
        return x_noised, aatype_noised, mlm_mask


    def denoising_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   aatype_t: TensorType["b n", int],
                   t: TensorType["b", float],
                   t_next: TensorType["b", float],
                   aatype_decoding_order: TensorType["b n", int],
                   aux_inputs: Optional[Dict[str, Any]] = None
                   ) -> Tuple[TensorType["b n a 3", float],  # xt_next
                              TensorType["b n", int],  # aatype_t_next
                              Dict[str, TensorType["b ..."]]  # aux preds
                              ]:
        x1_pred, aatype_pred, aux_preds = f(xt, aatype_t, t=t)

        # "euler" step for discrete space
        ## Unmask a certain number of residues in the sequence
        K_prev = torch.ceil(t * aux_inputs["lengths"]).long()[..., None]  # number of residues to be unmasked at the current time step
        K = torch.ceil(t_next * aux_inputs["lengths"]).long()[..., None]  # number of residues to be unmasked at the next time step
        residues_to_unmask = (K_prev <= aatype_decoding_order) & (aatype_decoding_order < K)
        aatype_t_next = torch.where(residues_to_unmask, aatype_pred, aatype_t)

        ## Unmask residues with x1_pred
        unmasked_residues = (aatype_decoding_order < K)
        xt_next = torch.where(unmasked_residues[..., None, None], x1_pred, xt)
        aux_inputs["mlm_mask"] = torch.where(unmasked_residues, torch.ones_like(aatype_t_next), torch.zeros_like(aatype_t_next))

        # Add to auxiliary outputs
        aux_preds["x1_pred"] = x1_pred
        aux_preds["aatype_pred"] = aatype_pred

        return xt_next, aatype_t_next, aux_preds
