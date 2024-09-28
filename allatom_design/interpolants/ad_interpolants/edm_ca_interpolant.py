from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from scipy import stats
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_ca_nco
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.edm_interpolant import \
    EDM
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule


class EDM_CA(ADInterpolant):
    def __init__(self, cfg: DictConfig, sigma_data: Tuple[TensorType[(), float]]):
        """
        EDM from Karras et al.

        Unlike Karras et al., time steps go from 0 (pure noise) to 1 (clean data) for consistency with other interpolants.

        Run EDM on CA atoms separately from EDM on the rest of the backbone atoms.
        - Non-CA backbone atoms will be centered at CA.
        """
        super().__init__()
        self.cfg = cfg

        ca_sigma_data, nco_sigma_data = sigma_data  # nco_sigma_data does not include CA
        self.ca_interpolant = EDM(cfg.ca, ca_sigma_data)
        self.nco_interpolant = EDM(cfg.nco, nco_sigma_data)


    @torch.compiler.disable
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                ) -> Dict[str, Any]:
        x1 = batch["x"]

        # Sample time steps if not provided
        if t is None:
            t = self.sample_timestep(x1.shape[0], device=x1.device)

        xt = self.noise_x(x1, t)

        # Construct outputs
        outputs = {}
        outputs["t"] = t  # [b]
        outputs["x_noised"] = xt
        outputs["x_target"] = x1  # we directly predict x1
        outputs["aatype_noised"] = batch["aatype"]  # we do not noise aatype
        outputs["loss_weight_t"] = self.get_loss_weight(t)  # Tuple[[b], [b]]

        return outputs


    def sigma(self, t: Tuple[TensorType["b", float]]) -> Tuple[TensorType["b", float]]:
        """
        Convert time step to noise level.
        """
        t_ca, t_nco = t
        sigma_ca = self.ca_interpolant.sigma(t_ca)
        sigma_nco = self.nco_interpolant.sigma(t_nco)
        return sigma_ca, sigma_nco


    def sigma_inv(self, sigma: Tuple[TensorType["b", float]]) -> Tuple[TensorType["b", float]]:
        """
        Convert noise level to time step.
        """
        sigma_ca, sigma_nco = sigma
        t_ca = self.ca_interpolant.sigma_inv(sigma_ca)
        t_nco = self.nco_interpolant.sigma_inv(sigma_nco)

        return t_ca, t_nco


    def sample_timestep(self, n: int, device: torch.device) -> Tuple[TensorType["b"]]:
        """
        Sample a batch of b timesteps from the noise schedule.
        """
        t_ca = self.ca_interpolant.sample_timestep(n, device)
        t_nco = self.nco_interpolant.sample_timestep(n, device)

        return t_ca, t_nco


    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        """
        Sample n samples from the prior.
        """
        x0 = torch.zeros(*shape, device=device)
        x0_ca = self.ca_interpolant.sample_prior(x0[..., 1:2, :].shape, device)
        x0_nco = self.nco_interpolant.sample_prior(x0[..., rc.nco_idxs, :].shape, device)
        x0 = cat_ca_nco(x0_ca, x0_nco)
        return x0


    def noise_x(self, x: TensorType["b n a 3"], t: Tuple[TensorType["b"]]) -> TensorType["b n a 3"]:
        """
        Add noise to x.
        """
        sigma_ca, sigma_nco = self.sigma(t)
        x_ca = x[..., 1:2, :] + torch.randn_like(x[..., 1:2, :]) * rearrange(sigma_ca, "b -> b 1 1 1")
        x_nco = x[..., rc.nco_idxs, :] + torch.randn_like(x[..., rc.nco_idxs, :]) * rearrange(sigma_nco, "b -> b 1 1 1")
        return cat_ca_nco(x_ca, x_nco)


    def churn(self,
              xt: TensorType["b n a 3", float],
              t: Tuple[TensorType["b", float]],  # t_ca, t_nco
              churn_cfg: Tuple[Optional[Dict[str, float]]]  # ca, nco
              ) -> Tuple[TensorType["b n a 3", float], TensorType["b", float]]:
        """
        Add churn to current time step based on EDM stochatic sampler.
        """
        churn_cfg_ca, churn_cfg_nco = churn_cfg

        xt_ca, xt_nco = xt[..., 1:2, :], xt[..., rc.nco_idxs, :]
        t_ca, t_nco = t

        if churn_cfg_ca is None or churn_cfg_ca["s_churn"] == 0:
            xt_hat_ca = xt_ca
            t_hat_ca = t_ca
        else:
            s_t_min_ca = churn_cfg_ca["s_t_min"] * self.ca_interpolant.sigma_data
            s_t_max_ca = churn_cfg_ca["s_t_max"] * self.ca_interpolant.sigma_data
            sigma_ca = self.ca_interpolant.sigma(t_ca)
            churn_mask_ca = (s_t_min_ca <= sigma_ca) & (sigma_ca <= s_t_max_ca)
            gamma_i_ca = (churn_cfg_ca["s_churn"] / churn_cfg_ca["num_steps"]) * churn_mask_ca.float()
            sigma_hat_ca = sigma_ca + gamma_i_ca * sigma_ca

            eps_i = torch.randn_like(xt_ca) * churn_cfg_ca["s_noise"]
            xt_hat_ca = xt_ca + eps_i * rearrange((sigma_hat_ca ** 2 - sigma_ca ** 2).sqrt(), "b -> b 1 1 1")
            t_hat_ca = self.ca_interpolant.sigma_inv(sigma_hat_ca)


        if churn_cfg_nco is None or churn_cfg_nco["s_churn"] == 0:
            xt_hat_nco = xt_nco
            t_hat_nco = t_nco
        else:
            s_t_min_nco = churn_cfg_nco["s_t_min"] * self.nco_interpolant.sigma_data
            s_t_max_nco = churn_cfg_nco["s_t_max"] * self.nco_interpolant.sigma_data
            sigma_nco = self.nco_interpolant.sigma(t_nco)
            churn_mask_nco = (s_t_min_nco <= sigma_nco) & (sigma_nco <= s_t_max_nco)
            gamma_i_nco = (churn_cfg_nco["s_churn"] / churn_cfg_nco["num_steps"]) * churn_mask_nco.float()
            sigma_hat_nco = sigma_nco + gamma_i_nco * sigma_nco

            eps_i = torch.randn_like(xt_nco) * churn_cfg_nco["s_noise"]
            xt_hat_nco = xt_nco + eps_i * rearrange((sigma_hat_nco ** 2 - sigma_nco ** 2).sqrt(), "b -> b 1 1 1")
            t_hat_nco = self.nco_interpolant.sigma_inv(sigma_hat_nco)

        xt_hat = cat_ca_nco(xt_hat_ca, xt_hat_nco)
        t_hat = (t_hat_ca, t_hat_nco)
        return xt_hat, t_hat


    def euler_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   t: TensorType["b", float],
                   t_next: TensorType["b", float],
                   noise_schedule: Tuple[Optional[NoiseSchedule]],
                   cfg_cfg: Optional[DictConfig],  # classifier-free guidance config
                   aux_inputs: Optional[Dict[str, Any]] = None
                   ) -> Tuple[TensorType["b n a 3", float],  # xt_next
                              Dict[str, TensorType["b ..."]]  # aux preds
                              ]:
        """
        Take an Euler step using the function f.

        f is the forward function of the denoiser trained with this interpolant.
        - It should take in the current state and the current time.
        """
        x1_pred, aux_preds = f(xt, t=t)
        aux_preds["x1_pred"] = x1_pred  # save x1_pred before any guidance modifications

        # Handle guidance
        autoguidance_cfg = aux_inputs.get("autoguidance_cfg", None)
        if (autoguidance_cfg is not None) and (autoguidance_cfg["use_autoguidance"]):
            # TODO: move this to each edm_interpolant
            f_autoguidance = autoguidance_cfg["autoguidance_fn"]
            x1_pred_ag, _ = f_autoguidance(xt, t=t)

            w = autoguidance_cfg["w"]
            # x1_pred = w * x1_pred + (1 - w) * x1_pred_ag
            x1_pred += (w - 1) * (aux_preds["x1_pred"] - x1_pred_ag)
            aux_preds["x1_pred_ag"] = x1_pred_ag

        sc_guidance_cfg = aux_inputs.get("sc_guidance_cfg", None)
        if (sc_guidance_cfg is not None) and (sc_guidance_cfg["use_sc_guidance"]):
            f_sc_guidance = sc_guidance_cfg["sc_guidance_fn"]
            x1_pred_sg, _ = f_sc_guidance(xt, t=t)

            w = sc_guidance_cfg["w"]
            x1_pred += (w - 1) * (aux_preds["x1_pred"] - x1_pred_sg)
            aux_preds["x1_pred_sg"] = x1_pred_sg


        sigma_ca, sigma_nco = self.sigma(t)
        score_ca = (xt[..., 1:2, :] - x1_pred[..., 1:2, :]) / rearrange(sigma_ca, "b -> b 1 1 1")
        score_nco = (xt[..., rc.nco_idxs, :] - x1_pred[..., rc.nco_idxs, :]) / rearrange(sigma_nco, "b -> b 1 1 1")

        if cfg_cfg is not None:
            raise NotImplementedError("Classifier-free guidance is not implemented yet.")

        # Handle noise schedules
        noise_schedule_ca, noise_schedule_nco = noise_schedule
        if noise_schedule_ca is not None:
            score_ca = noise_schedule_ca.scale_vf(score_ca, t)
        if noise_schedule_nco is not None:
            score_nco = noise_schedule_nco.scale_vf(score_nco, t)

        # take the step
        sigma_next_ca, sigma_next_nco = self.sigma(t_next)
        dscore_ca = rearrange(sigma_next_ca - sigma_ca, "b -> b 1 1 1") * score_ca
        dscore_scn = rearrange(sigma_next_nco - sigma_nco, "b -> b 1 1 1") * score_nco
        dscore = cat_ca_nco(dscore_ca, dscore_scn)
        xt_next = xt + dscore
        return xt_next, aux_preds


    def get_loss_weight(self, t: Tuple[TensorType["b"]]) -> Tuple[TensorType["b"]]:
        """
        Compute the weight of the loss at time t.
        """
        t_ca, t_nco = t
        return self.ca_interpolant.get_loss_weight(t_ca), self.nco_interpolant.get_loss_weight(t_nco)


    def setup_preconditioning(self,
                              x_noised: TensorType["b n a 3", float],
                              x_self_cond: Optional[TensorType["b n a 3", float]],
                              t: TensorType["b", float]) -> Tuple[Callable, Callable]:
        """
        Set up preconditioning input and output functions.
        """
        t_ca, t_nco = t
        x_noised_ca, x_noised_nco = x_noised[..., 1:2, :], x_noised[..., rc.nco_idxs, :]
        x_self_cond_ca, x_self_cond_nco = None, None
        if x_self_cond is not None:
            x_self_cond_ca, x_self_cond_nco = x_self_cond[..., 1:2, :], x_self_cond[..., rc.nco_idxs, :]
        precondition_in_ca, precondition_out_ca = self.ca_interpolant.setup_preconditioning(x_noised_ca, x_self_cond_ca, t_ca)
        precondition_in_nco, precondition_out_nco = self.nco_interpolant.setup_preconditioning(x_noised_nco, x_self_cond_nco, t_nco)

        # precondition CA and NCO separately
        def precondition_in():
            pc_x_noised_ca, pc_x_self_cond_ca, c_noise_ca = precondition_in_ca()
            pc_x_noised_nco, pc_x_self_cond_nco, c_noise_nco = precondition_in_nco()
            pc_x_noised = cat_ca_nco(pc_x_noised_ca, pc_x_noised_nco)
            pc_x_self_cond = None
            if pc_x_self_cond_ca is not None and pc_x_self_cond_nco is not None:
                pc_x_self_cond = cat_ca_nco(pc_x_self_cond_ca, pc_x_self_cond_nco)
            c_noise = (c_noise_ca, c_noise_nco)
            return pc_x_noised, pc_x_self_cond, c_noise

        precondition_out = lambda x: cat_ca_nco(precondition_out_ca(x[..., 1:2, :]), precondition_out_nco(x[..., rc.nco_idxs, :]))
        return precondition_in, precondition_out
