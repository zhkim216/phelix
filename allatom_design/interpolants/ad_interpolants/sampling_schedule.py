from typing import Tuple, Union

import torch
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc


class NoiseSchedule():
    def __init__(self, cfg: DictConfig):
        """
        Noise schedule to use at sampling time.

        A noise schedule config should have the following attributes:
            name: str  # name of the noise schedule
            kwarg1: Any  # additional arguments for the noise schedule
            kwarg2: Any  # additional arguments for the noise schedule

        Examples:

        Linear schedule: (1-t) * x0 + t * x1
            name: "linear"

        Expoential schedule: x0 * exp(-c * t) + x1 * (1 - exp(-c * t))
            name: "exponential"
            c: float  # exponential decay constant

        """
        super().__init__()
        self.cfg = cfg


    def scale_vf(self,
                 vf: TensorType["b n a 3"],
                 t: Union[TensorType["b", float],
                          Tuple[TensorType["b", float], TensorType["b", float]]]
                 ) -> TensorType["b n a 3"]:
        """
        Scales vector field according to the noise schedule.
        """
        # apply noise schedule
        if self.cfg.name == "linear":
            return vf
        elif self.cfg.name == "exponential":
            c = self.cfg.c
            return vf * c * torch.exp(-c * t)
        elif self.cfg.name == "step_scale":
            c = self.cfg.c
            return vf * c
        else:
            raise NotImplementedError(f"Unknown noise schedule: {self.cfg.name}")