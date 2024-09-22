from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule


class SDInterpolant(nn.Module, ABC):
    """
    Generic interpolant for masking sequence / sidechains.
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
        - aatype_noised: the noised aatype [b n]
        - mlm_mask: 1 if the residue is kept, 0 if it is masked [b n]
        """
        pass


    @abstractmethod
    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        pass


    @abstractmethod
    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        pass
