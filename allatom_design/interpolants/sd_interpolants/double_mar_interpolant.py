from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.interpolants.sd_interpolants.sd_interpolant import SDInterpolant
from allatom_design.interpolants.sd_interpolants.mar_interpolant import MAR
from allatom_design.interpolants.ad_interpolants.sampling_schedule import NoiseSchedule

class DOUBLE_MAR(SDInterpolant):
    def __init__(self, cfg: DictConfig):

        """
        Multi-modal interpolant to define separate noise schedules on sequence and sidechain coordinates.

        Sequence: Masked autoregressive diffusion (MAR)

        Sidechain: Masked autoregressive diffusion (MAR)

        Time should be passed in as a tuple of two tensors for separate time steps on sidechain and backbone (t_sd, t_scn).
        """
        super().__init__()
        self.cfg = cfg
        self.seq_interpolant = MAR(cfg.seq)
        self.scn_interpolant = MAR(cfg.scn)


    @torch.compiler.disable
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Tuple[TensorType["b", float], TensorType["b", float]] = None,  # (t_bb, t_scn)
                ) -> Dict[str, Any]:
        x1 = batch["x"]

        # Sample time steps if not provided
        if t is None:
            t = self.sample_timestep(x1.shape[0], x1.device)

        # Get noisy samples
        xt, aatype_noised, seq_mlm_mask, scn_mlm_mask = self.noise_samples(x1, batch["aatype"], t)

        # Construct outputs
        outputs = {}
        outputs["t"] = t  # [b]
        outputs["x_noised"] = xt  # [b n a 3]
        outputs["x_target"] = x1  # we directly predict clean coordinates [b n a 3]
        outputs["aatype_noised"] = aatype_noised  # [b n]
        outputs["loss_weight_t"] = self.get_loss_weight(t)  # ([b], [b])
        outputs["seq_mlm_mask"] = seq_mlm_mask  # [b n], 1 where unmasked, 0 where masked
        outputs["scn_mlm_mask"] = scn_mlm_mask  # [b n], 1 where unmasked, 0 where masked

        return outputs


    def sample_timestep(self, n: int, device: torch.device) -> Tuple[TensorType["b"], TensorType["b"]]:
        """
        Sample a batch of b timesteps from the noise schedule.
        """
        t_seq = self.seq_interpolant.sample_timestep(n, device)
        t_scn = self.scn_interpolant.sample_timestep(n, device)
        return t_seq, t_scn


    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3", float]:
        """
        Sample n samples from the prior.

        For backbone, initialize from sigma_bb at t=0. For sidechain, initialize to all zeroes.
        """
        x0 = torch.zeros(*shape, device=device)
        return x0


    def noise_samples(self,
                      x: TensorType["b n a 3"],
                      aatype: TensorType["b n", int],
                      t: TensorType["b"]) -> Tuple[TensorType["b n a 3", float],
                                                   TensorType["b n", int],
                                                   TensorType["b n", int]]:
        """
        Add noise to the samples according to the time steps for bb and scn.
        """
        #parse multiple times 
        t_seq, t_scn = t
        B, N, _, _ = x.shape

        #mlm mask for sequence
        seq_mlm_mask = torch.rand(B, N, device=x.device) < rearrange(t_seq, "b -> b 1")  # 1 if we keep the residue, 0 if we mask it
        seq_mlm_mask = seq_mlm_mask.long()

        #mlm mask for sidechain mask will be union'd with sequence mask
        scn_mlm_mask = torch.rand(B, N, device=x.device) < rearrange(t_scn, "b -> b 1")  # 1 if we keep the residue, 0 if we mask it
        scn_mlm_mask = scn_mlm_mask.long()
        scn_mlm_mask = scn_mlm_mask * seq_mlm_mask 

        #mask sidechain coordinates accoring to scn_mlm_mask
        x_noised = x.clone()
        x_noised[..., rc.non_bb_idxs, :] = x[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()

        #mask sequence according to seq_mask
        aatype_noised = torch.where(seq_mlm_mask.bool(), aatype, rc.restype_order_with_x["X"])# TODO: replace with MASK

        return x_noised, aatype_noised, seq_mlm_mask, scn_mlm_mask


    def churn(self,
              xt: TensorType["b n a 3", float],
              t: Tuple[TensorType["b", float], TensorType["b", float]],  # (t_bb, t_scn)
              churn_cfg: Optional[DictConfig]) -> Tuple[TensorType["b n a 3", float], TensorType["b", float]]:
        """
        Add churn to current time step based on EDM stochatic sampler.

        churn_cfg should have separate configs for scn and bb under .bb and .scn.
        """
        return xt, t  # no churn for MAR


    def euler_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   aatype_t: TensorType["b n", int],
                   t: TensorType["b", float],
                   t_next: TensorType["b", float],
                   aux_inputs: Optional[Dict[str, Any]] = None,
                   ) -> Tuple[TensorType["b n a 3", float],  # xt_next
                              TensorType["b n", int],  # aatype_t_next
                              Dict[str, TensorType["b ..."]]  # aux preds
                              ]:
        # "euler" step for discrete space
        ## Unmask a certain number of residues in the sequence
        x1_pred, aatype_pred, aux_preds = f(xt, aatype_t, t=t)

        unmasked_prev = aux_inputs['unmasked_prev']
        K_prev = torch.sum(unmasked_prev, dim=-1).unsqueeze(-1) # number of residues cumulatively unmasked at the last time step
        K = torch.ceil(t_next * aux_inputs["lengths"]).long()[..., None]  # number of residues to be unmasked at the next time step

        seq_mask = aux_preds['seq_mask']
        confidence, _ = torch.max(aux_preds['seq_probs'], dim = -1)

        #padded tokens sent to end of order
        confidence = torch.where(seq_mask == 0, -1e6, confidence)

        #previously unmasked tokens sent to beginning of order
        confidence = torch.where(unmasked_prev, 1e6, confidence)

        #set decoding order based on confidence
        confidence_decoding_order = torch.argsort(torch.argsort(confidence, dim = -1, descending = True))
        residues_to_unmask = (confidence_decoding_order >= K_prev) & (confidence_decoding_order < K) 
        aatype_t_next = torch.where(residues_to_unmask, aatype_pred, aatype_t)

        ## Repack sidechains of all unmasked residues
        unmasked_residues = residues_to_unmask + unmasked_prev
        xt_next = torch.where(unmasked_residues[..., None, None], x1_pred, xt)

        # Add to auxiliary outputs
        aux_preds["x1_pred"] = x1_pred
        aux_preds["aatype_pred"] = aatype_pred
        aux_preds["unmasked_residues"] = unmasked_residues

        return xt_next, aatype_t_next, aux_preds


    def get_loss_weight(self,
                        t: Tuple[TensorType["b", float], TensorType["b", float]],
                        ) -> Tuple[TensorType["b", float], TensorType["b", float]]:
        """
        Compute the weight of the loss at time t.
        """
        t_seq, t_scn = t
        loss_weight_seq = torch.ones_like(t_seq)  # we deal with loss on masked sidechains using sd_mlm_mask
        loss_weight_scn = torch.ones_like(t_scn)  # we deal with loss on masked residues using scn_mlm_mask
        return loss_weight_seq, loss_weight_scn

    def setup_preconditioning(x_noised: TensorType["b n a 3", float],
                              x_self_cond: Optional[TensorType["b n a 3", float]],
                              t: Tuple[TensorType["b", float], TensorType["b", float]]) -> Tuple[Callable, Callable]:
        """
        Returns
        - a function that preconditions the input to the denoiser
        - a function that preconditions the output of the denoiser
        """
        raise NotImplementedError("Preconditioning not implemented for MAR.")