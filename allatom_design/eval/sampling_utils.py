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

    if mode == "autoregressive":
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
    else:
        res_decoding_order = torch.where(seq_mask.bool(), torch.rand_like(seq_mask), 1.0e6)  # decode padded positions last
        res_decoding_order = res_decoding_order.argsort(dim=-1)

    return res_decoding_order.long()

def get_confidence_decoding_order(mode: str,
                                  seq_probs: TensorType["b n", float],
                                  seq_mask: TensorType["b n", float],
                                  unmasked_prev: TensorType["b n", int]) -> TensorType["b n", int]:
    """
    Use sequence probabilities to decide a confidence based sampling order
    """
    if mode == 'greedy':
        confidence, _ = torch.max(seq_probs, dim = -1)
        confidence = torch.where(seq_mask == 0, -1e6, confidence) #padded tokens sent to end of order
        confidence = torch.where(unmasked_prev == 1, 1e6, confidence) #previously unmasked tokens sent to beginning of order
        confidence_decoding_order = torch.argsort(torch.argsort(confidence, dim = -1, descending = True)) #update decoding order based on confidence
    else:
        raise ValueError(f'Confidence mode {mode} has not been implemented yet!')

    return confidence_decoding_order


def update_mlm_mask(mlm_mask: TensorType["b n", float],
                    aatype_decoding_order: TensorType["b n", int],
                    aatype_decoding_order_mode: str,
                    K: TensorType["b", int],
                    seq_mask: TensorType["b n", float],
                    seq_probs: TensorType["b n k", float],
                    ) -> TensorType["b n", float]:
    """
    Update mlm_mask so that K total residues are unmasked.
    """
    mlm_mask_prev = mlm_mask.clone()
    if aatype_decoding_order_mode in ['greedy']:
        aatype_decoding_order = get_confidence_decoding_order(mode=aatype_decoding_order_mode,
                                                              seq_probs=seq_probs,
                                                              seq_mask=seq_mask,
                                                              unmasked_prev=mlm_mask_prev)

    ## using decoding order to decide positions to unmask
    residues_to_unmask = (~mlm_mask_prev.bool()) & (aatype_decoding_order < K[:,None])
    mlm_mask = residues_to_unmask + mlm_mask_prev
    return mlm_mask


def unmask(xt,
           aatype_t,
           x1_pred,
           aatype_pred,
           mlm_mask_prev,
           mlm_mask) -> Tuple[TensorType["b n a 3", float],
                              TensorType["b n", int]]:
    """
    Update aatype pred and x1 based on newly unmasked residues.
    """
    residues_to_unmask = mlm_mask - mlm_mask_prev

    ## Unmask residues
    aatype_t = torch.where(residues_to_unmask.bool(), aatype_pred, aatype_t)

    ## Pack sidechains of residues we're unmasking, if using an aasd model
    if x1_pred is not None:
        xt = torch.where(residues_to_unmask[..., None, None].bool(), x1_pred, xt)

    return xt, aatype_t


def get_timesteps_from_schedule(mode: str,
                                num_steps: int,
                                t_start: float,
                                t_end: float
                                ) -> TensorType["S+1", float]:
    """
    Get timesteps from timestep schedule for sampling. Essentially warps the time schedule to be non-linear.
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
    elif mode == "cbrt":
        timesteps = timesteps ** (1.0 / 3.0)
    elif mode == "cosine":
        timesteps = 1 - torch.cos(timesteps * np.pi / 2)
    elif mode == "last_only":
        timesteps = torch.zeros_like(timesteps)
        timesteps[-1] = 1.0
    elif mode == "first_only":
        timesteps = torch.ones_like(timesteps)
        timesteps[0] = 0.0
    else:
        raise NotImplementedError(f"timestep schedule mode {mode} not implemented")

    return timesteps
