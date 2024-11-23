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
                 t: TensorType["b", float],
                 ) -> TensorType["b n a 3"]:
        """
        Scales vector field according to the noise schedule.
        """
        # apply noise schedule
        if self.cfg.name == "linear":
            return vf
        elif self.cfg.name == "exponential":
            c = self.cfg.c
            return vf * c * torch.exp(-c * rearrange(t, "b -> b 1 1 1"))
        elif self.cfg.name == "step_scale":
            c = self.cfg.c
            return vf * c
        elif self.cfg.name == "step_scale_interval":
            c1 = self.cfg.c1
            c2 = self.cfg.c2
            t_c = self.cfg.t_c
            c_t = torch.where((0 <= t) & (t <= t_c), c1, c2)
            c_t = rearrange(c_t, "b -> b 1 1 1")
            return vf * c_t
        else:
            raise NotImplementedError(f"Unknown noise schedule: {self.cfg.name}")