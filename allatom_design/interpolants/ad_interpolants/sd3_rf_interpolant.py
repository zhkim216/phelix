from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from allatom_design.interpolants.ad_interpolants.ad_interpolant import \
    ADInterpolant
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule



class SD3_RF(ADInterpolant):
    def __init__(self, cfg: DictConfig):
        """
        Stable Diffusion 3 Rectified Flow.

        Unlike in SD3, time goes from 0 to 1 to maintain consistency with the other interpolants.
        """
        super().__init__()
        self.cfg = cfg

        self.training_t_schedule = cfg.training_t_schedule
        assert self.training_t_schedule in ["logit_normal", "mode", "proteina"], f"Unknown timestep schedule: {self.training_t_schedule}"

        self.training_t_schedule_cfg = cfg.training_t_schedule_cfg[self.training_t_schedule]


    @torch.compiler.disable
    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                ) -> dict[str, Any]:
        x1 = batch["x"]
        x0 = self.sample_prior(x1.shape, device=x1.device)

        if t is None:
            t = self.sample_timestep(x1.shape[0], device=x1.device)
        xt = self.get_conditional_flow(x0, x1, t)
        target = self.get_conditional_vf(x0, x1, t)
        loss_weight_t = self.get_loss_weight(t)

        # Construct outputs
        outputs = {}
        outputs["t"]  = t
        outputs["x_noised"] = xt
        outputs["x_target"] = target  # we predict x1 - x0
        outputs["aatype_noised"] = batch["aatype"]  # we do not noise aatype
        outputs["loss_weight_t"]  = loss_weight_t
        return outputs


    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        """
        Sample a batch of b timesteps from the timestep schedule.
        """
        if self.training_t_schedule == "logit_normal":
            location, scale = self.training_t_schedule_cfg.location, self.training_t_schedule_cfg.scale
            u = torch.randn(n, device=device) * scale + location
            t = torch.sigmoid(u)
        elif self.training_t_schedule == "mode":
            s = self.training_t_schedule_cfg.s
            u = torch.rand(n, device=device)
            t = 1 - u - s * (torch.cos(torch.pi/2 * u) ** 2 - 1 + u)
        elif self.training_t_schedule == "proteina":
            # Adapated from Proteina; mixture of Uniform and Beta
            alpha = self.training_t_schedule_cfg.alpha
            beta = self.training_t_schedule_cfg.beta
            dist = torch.distributions.beta.Beta(alpha, beta)
            samples_beta = dist.sample((n, )).to(device)
            samples_uniform = torch.rand(n, device=device)
            u = torch.rand(n, device=device)
            return torch.where(u < 0.02, samples_uniform, samples_beta)
        return t


    def sample_prior(self, shape: Tuple, device: torch.device):
        """
        Sample n samples from the prior.
        - N(0, 1)
        """
        return torch.randn(*shape, device=device)


    def get_conditional_flow(self,
                             x: TensorType["b n a 3"],
                             x1: TensorType["b n a 3"],
                             t: TensorType["b"]) -> TensorType["b n a 3"]:
        """
        Compute the conditional flow at time t given x1.
        """
        t = rearrange(t, "b -> b 1 1 1")
        return (1 - t) * x + t * x1


    def get_conditional_vf(self,
                           x0: TensorType["b n a 3"],
                           x1: TensorType["b n a 3"],
                           t: TensorType["b"]) -> TensorType["b n a 3"]:
        """
        Compute the conditional vector field at time t as a function of x0.
        """
        return x1 - x0


    def churn(self,
              xt: TensorType["b n a 3", float],
              t: TensorType["b", float],
              churn_cfg: Optional[DictConfig]) -> tuple[TensorType["b n a 3", float], TensorType["b", float]]:
        """
        Add churn to current time step based on EDM stochatic sampler.
        """
        if churn_cfg is None:
            return xt, t
        # TODO: check this math, I didn't put enough thought into this
        # the main idea is that in flow matching, s(t) = t , and sigma(t) = 1 - t / t
        sigma_inv = lambda sigma: 1 / (1 + sigma)
        sigma = lambda t: (1 - t) / t.clamp(1e-5)
        churn_mask = (churn_cfg["s_t_min"] <= t) & (t <= churn_cfg["s_t_max"])
        gamma_i = (churn_cfg["s_churn"] / churn_cfg["num_steps"]) * churn_mask.float()
        t_hat = sigma_inv(sigma(t) + gamma_i * sigma(t))
        eps_i = torch.randn_like(xt) * churn_cfg["s_noise"]

        scaling = rearrange((t_hat / t.clamp(1e-5)), "b -> b 1 1 1")
        std = rearrange((sigma(t_hat) ** 2 - sigma(t) ** 2).clip(min=0).sqrt(), "b -> b 1 1 1")
        xt_hat = scaling * xt + std * rearrange(t_hat, "b -> b 1 1 1") * eps_i
        return xt_hat, t_hat


    def euler_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   t: TensorType["b", float],
                   t_next: TensorType["b", float],
                   noise_schedule: Optional[NoiseSchedule],
                   cfg_cfg: Optional[DictConfig],  # classifier-free guidance config
                   autoguidance_cfg: Optional[DictConfig],  # autoguidance config
                   aux_inputs: Optional[dict[str, Any]] = None
                   ) -> tuple[TensorType["b n a 3", float],  # xt_next
                              dict[str, TensorType["b ..."]]  # aux preds
                              ]:
        """
        Take an Euler step using the function f.

        f is the forward function of the denoiser trained with this interpolant.
        - It should take in the current state and the current time.

        Returns:
        - xt_next (the prediction for the next time step)
        - x1_pred (the current timestep's prediction of x1)
        """
        if cfg_cfg is not None:
            raise NotImplementedError("Classifier-free guidance not implemented yet for SD3 RF interpolant.")

        dt = rearrange(t_next - t, "b -> b 1 1 1")
        vf, aux_preds = f(xt, t=t)

        aux_preds["x1_pred"] = self.get_x1_pred(vf, xt, t)  # save x1 pred before any guidance modifications

        if noise_schedule is not None:
            vf = noise_schedule.scale_vf(vf, t)

        xt_next = xt + dt * vf
        return xt_next, aux_preds


    def get_loss_weight(self, t: TensorType["b"]) -> TensorType["b"]:
        """
        Compute the weight of the loss at time t.
        """
        return torch.ones_like(t)


    def get_x1_pred(self,
                    denoiser_pred: TensorType["b n a 3", float],
                    xt: TensorType["b n a 3", float],
                    t: TensorType["b", float]
                    ) -> TensorType["b n a 3", float]:
        """
        Given a prediction from the denoiser, return the prediction of x1 at time t.
        """
        return xt + (1 - rearrange(t, "b -> b 1 1 1")) * denoiser_pred



    def setup_preconditioning(self,
                              x_noised: TensorType["b n a 3", float],
                              x_self_cond: Optional[TensorType["b n a 3", float]],
                              t: TensorType["b", float]) -> tuple[Callable, Callable]:
        """
        Set up preconditioning input and output functions.
        """

        def precondition_in() -> tuple[TensorType["b n a 3", float],  # x_noised
                                       TensorType["b n a 3", float],  # x_self_cond
                                       TensorType["b", float]  # c_noise
                                       ]:
            """
            """
            return x_noised, x_self_cond, t

        def precondition_out(denoiser_pred: Union[TensorType["b n a 3", float],
                                                  TensorType["m b n a 3", float]]
                             ) -> TensorType["b n a 3", float]:
            """
            """
            return denoiser_pred


        return precondition_in, precondition_out
