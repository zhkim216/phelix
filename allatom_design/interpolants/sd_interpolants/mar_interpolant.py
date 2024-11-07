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

        # Additional training tasks
        self.full_noise_p = getattr(cfg, "full_noise_p", 0.0)  # randomly set full_noise_p timesteps to full noise
        self.drop_scn_p = getattr(cfg, "drop_scn_p", 0.0)  # randomly set all sidechain atoms to zero


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
        outputs["seq_mlm_mask"] = mlm_mask  # [b n]

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

        if self.full_noise_p > 0:
            # randomly set full_noise_p timesteps to full noise
            t = torch.where(torch.rand(n, device=device) < self.full_noise_p, torch.zeros(n, device=device), t)

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
        Add noise to x and aatype. Return x_noised, aatype_noised, and mlm_mask.

        For MAR, we keep each residue with probability t.
        When masking residues, we zero out the coordinates and set aatype to X.
        """
        B, N, _, _ = x.shape
        mlm_mask = torch.rand(B, N, device=x.device) < rearrange(t, "b -> b 1")  # 1 if we keep the residue, 0 if we mask it
        mlm_mask = (mlm_mask * seq_mask).float()  # mask out residues that are not in the sequence
        x_noised = x.clone()

        # Mask sidechains based on mlm_mask
        x_noised[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(mlm_mask, "b n -> b n 1 1").float()

        # Mask sequence based on mlm_mask
        aatype_noised = torch.where(mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])  # TODO: replace with MASK
        aatype_noised = aatype_noised * seq_mask  # set pad residues back to 0
        aatype_noised = aatype_noised.long()

        # Occasionally zero out all sidechain atoms
        scn_mask = torch.ones(B, device=x.device)
        if self.drop_scn_p > 0:
            scn_mask = (torch.rand(B, device=x.device) > self.drop_scn_p).float()
            x_noised[..., rc.non_bb_idxs, :] = x_noised[..., rc.non_bb_idxs, :] * rearrange(scn_mask, "b -> b 1 1 1")

        return x_noised, aatype_noised, mlm_mask

    def corrector_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   aatype_t: TensorType["b n", int],
                   K: TensorType["b", int],
                   unmasked_prev: TensorType["b n", int],
                   t: TensorType["b", float],
                   aux_inputs: Dict[str, TensorType["b ..."]] # aux_inputs
                   ) -> Dict[str, TensorType["b ..."]]: #aux preds

        # Get tokens to remask
        b, n = aatype_t.shape
        mlm_mask = aux_inputs['seq_mlm_mask'].clone()
        unmasked_positions = torch.nonzero(mlm_mask, as_tuple=False)
        unmasked_by_example = [unmasked_positions[unmasked_positions[:, 0] == i, 1] for i in range(b)]

        # Randomly select K positions for each batch
        selected_positions = torch.cat([
            batch_indices[torch.randperm(len(batch_indices))[:K[i]]]
            for i, batch_indices in enumerate(unmasked_by_example)])

        # Create indices for remasking
        batch_indices = torch.arange(b, device=xt.device).repeat_interleave(K)

        # Noise sequence by setting selected positions to 0 in mlm_mask
        mlm_mask[batch_indices, selected_positions] = 0
        aux_inputs['seq_mlm_mask'] = mlm_mask
        unmasked_prev = aux_inputs['seq_mlm_mask'].clone()
        aatype_t_noised = torch.where(mlm_mask.bool(), aatype_t, rc.restype_order_with_x["X"])

        # Noise sidechain
        xt_noised = xt.clone()
        xt_noised[..., rc.non_bb_idxs, :] = xt_noised[..., rc.non_bb_idxs, :] * rearrange(mlm_mask, "b n -> b n 1 1").float()

        # Run sequence denoiser
        x1_pred, aatype_pred, aux_preds = f(xt_noised, aatype_t_noised, t=t)

        # Add to auxiliary outputs
        aux_preds["x1_pred"] = x1_pred
        aux_preds["aatype_pred"] = aatype_pred

        return x1_pred, aatype_pred, aux_preds, unmasked_prev


    def remask_K(self,
                    xt: TensorType["b n a 3", float],
                    aatype_t: TensorType["b n", int],
                    mlm_mask: TensorType["b n"],
                    K: TensorType["b", int]) -> Tuple[TensorType["b n a 3", float],
                                                      TensorType["b n", int],
                                                      TensorType["b n", float]
                                                      ]:
        """
        Mask out K residues chosen from unmasked residues in mlm_mask.
        """
        # Get tokens to remask
        B, N = aatype_t.shape
        unmasked_positions = torch.nonzero(mlm_mask, as_tuple=False)
        unmasked_by_example = [unmasked_positions[unmasked_positions[:, 0] == i, 1] for i in range(B)]

        # Randomly select K positions for each batch
        selected_positions = torch.cat([
            batch_indices[torch.randperm(len(batch_indices))[:K[i]]]
            for i, batch_indices in enumerate(unmasked_by_example)])

        # Create indices for remasking
        batch_indices = torch.arange(B, device=xt.device).repeat_interleave(K)

        # Noise sequence by setting selected positions to 0 in mlm_mask
        mlm_mask = mlm_mask.clone()
        mlm_mask[batch_indices, selected_positions] = 0
        aatype_t_noised = torch.where(mlm_mask.bool(), aatype_t, rc.restype_order_with_x["X"])

        # Noise sidechain
        xt_noised = xt.clone()
        xt_noised[..., rc.non_bb_idxs, :] = xt_noised[..., rc.non_bb_idxs, :] * rearrange(mlm_mask, "b n -> b n 1 1").float()

        return xt_noised, aatype_t_noised, mlm_mask
