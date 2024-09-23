from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType
from omegaconf import DictConfig


def get_decoding_order(mode: str,
                       seq_mask: TensorType["b n", float],
                       **kwargs) -> TensorType["b n", int]:
    """
    Get the order in which residues should be decoded, from 0 to N-1.
    Padding tokens are decoded last.

    mode:
    - "random": decode residues in random order
    - "autoregressive": decode residues in autoregressive order
    - "random_spans": decode spans in random order.
        - kwargs["timesteps"]: TensorType["b s+1", float] proportion of residues to be unmasked at each timestep

    """
    B, N = seq_mask.shape

    if mode == "random":
        res_decoding_order = torch.where(seq_mask.bool(), torch.rand_like(seq_mask), 1.0e6)  # decode padded positions last
        res_decoding_order = res_decoding_order.argsort(dim=-1)
    elif mode == "autoregressive":
        res_decoding_order = torch.arange(N, device=seq_mask.device).expand(B, N)
        res_decoding_order = torch.where(seq_mask.bool(), res_decoding_order, 1.0e6)  # decode padded positions last
    elif mode == "random_spans":
        timesteps = kwargs["timesteps"]
        lengths = seq_mask.sum(dim=-1).long()
        res_decoding_order = torch.where(seq_mask.bool(), torch.zeros_like(seq_mask), 1.0e6)
        for i in range(B):
            N_unmasked = (lengths[i] * timesteps[i]).ceil().long()  # number of unmasked residues at each timestep
            chunk_sizes = N_unmasked[1:] - N_unmasked[:-1]  # number of residues to unmask at each timestep
            indices = torch.arange(lengths[i], device=seq_mask.device)
            chunks = torch.split(indices, chunk_sizes.tolist())
            chunks = [chunks[i] for i in torch.randperm(len(chunks))]
            res_decoding_order[i, :lengths[i]] = torch.cat(chunks)
    elif mode == "random_bidirectional":
        indices = get_random_bidirectional(seq_mask)
        res_decoding_order = torch.argsort(indices, dim=-1)
    else:
        raise NotImplementedError(f"residue decoding order mode {mode} not implemented")

    return res_decoding_order.long()


def get_random_bidirectional(seq_mask: TensorType["b n", float]) -> TensorType["b n", int]:
    """
    Start from a random position and decode residues randomly in both directions.
    """
    B, N = seq_mask.shape
    p = torch.rand((B, ), device=seq_mask.device)  # choose percentage of residues to generate to the left; uniform
    p = torch.stack([p, 1 - p], dim=1)

    partition_flags = torch.multinomial(p, num_samples=N, replacement=True) + 1  # 1=generate to the left, 2=generate to the right
    partition_flags = partition_flags * seq_mask.long()  # mask out residues that are not in the sequence
    partition_flags[:, 0] = 0  # the first residue is not a shift

    start_idx = (partition_flags == 1).sum(dim=1)  # we start based on the number of residues to the left
    counts = torch.where(partition_flags == 1, torch.cumsum(-1 * (partition_flags == 1), dim=1), partition_flags)
    counts = torch.where(partition_flags == 2, torch.cumsum(partition_flags == 2, dim=1), counts)

    indices = (counts + start_idx[..., None]) * seq_mask.long() # add shifts to start index
    return indices


def get_timestep_schedule(mode: str,
                          num_steps: int,
                          t_start: float,
                          t_end: float
                          ) -> TensorType["S+1", float]:
    """
    Get timestep schedule for sampling. Essentially warps the time schedule to be non-linear.

    """
    S = num_steps
    timesteps = torch.linspace(t_start, t_end, S + 1)
    if mode == "linear":
        pass
    elif mode == "square":
        timesteps = timesteps ** 2
    elif mode == "cubic":
        timesteps = timesteps ** 3
    elif mode == "sqrt":
        timesteps = timesteps ** 0.5
    elif mode == "cosine":
        # TODO: double check that this is what MaskGIT means by cosine
        timesteps = 1 - torch.cos(timesteps * np.pi / 2)
    elif mode == "last_only":
        timesteps = torch.zeros_like(timesteps)
        timesteps[-1] = 1.0
    else:
        raise NotImplementedError(f"timestep schedule mode {mode} not implemented")

    return timesteps
