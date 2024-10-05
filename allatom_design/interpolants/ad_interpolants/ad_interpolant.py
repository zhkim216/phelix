from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule


class ADInterpolant(nn.Module, ABC):
    """
    Generic interpolant on atoms / coordinates.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                ) -> Dict[str, Any]:
        """
        Noises the inputs and returns:
        - t: the sampled timestep [b]
        - x_noised: the noised coordinates (xt) [b n a 3]
        - x_target: the coordinates to predict (x1) [b n a 3]
        - aatype_noised: the noised aatype [b n]
        - loss_weight_t: the timestep-based loss weights for each element of this batch [b]
        """

        pass


    @abstractmethod
    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        pass


    @abstractmethod
    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        pass


    @abstractmethod
    def churn(self,
              xt: TensorType["b n a 3", float],
              t: TensorType["b", float],
              churn_cfg: Optional[DictConfig]) -> Tuple[TensorType["b n a 3", float], TensorType["b", float]]:
        """
        Add churn to current time step based on EDM stochatic sampler.
        """
        pass


    @abstractmethod
    def euler_step(self,
                   f: Callable,
                   xt: TensorType["b n a 3", float],
                   aatype_t: TensorType["b n", int],  # aatype at time t
                   t: TensorType["b", float],
                   t_next: TensorType["b", float],
                   noise_schedule: Optional[NoiseSchedule],
                   cfg_cfg: Optional[DictConfig],  # classifier-free guidance config
                   aux_inputs: Optional[Dict[str, Any]] = None
                   ) -> Tuple[TensorType["b n a 3", float],  # xt_next
                              TensorType["b n", int],  # aatype_t_next
                              Dict[str, TensorType["b ..."]]  # aux preds
                              ]:
        """
        Take an Euler step using the function f.

        f is the forward function of the denoiser trained with this interpolant.
        - It should take in the current state and the current time.

        """
        pass


    @abstractmethod
    def setup_preconditioning(x_noised: TensorType["b n a 3", float],
                              x_self_cond: Optional[TensorType["b n a 3", float]],
                              t: Tuple[TensorType["b", float], TensorType["b", float]]) -> Tuple[Callable, Callable]:
        """
        Returns
        - a function that preconditions the input to the denoiser
        - a function that preconditions the output of the denoiser
        """
        pass