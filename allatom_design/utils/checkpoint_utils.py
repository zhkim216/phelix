from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def get_cfg_from_ckpt(
    ckpt_path: str,
    return_as_dict: bool = False,
) -> tuple[DictConfig | dict[str, Any], dict[str, Any]]:
    """
    Load the config directly from the cfg arg passed into the model during training.

    Also returns the model checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(cfg_dict)

    if return_as_dict:
        return cfg_dict, ckpt
    return cfg, ckpt


def repair_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Repair the state dict to avoid issues with loading the model checkpoint due to torch.compile().

    https://github.com/pytorch/pytorch/issues/101107
    """
    return {src_key.replace("_orig_mod.", ""): value for src_key, value in state_dict.items()}
