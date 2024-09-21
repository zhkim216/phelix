from pathlib import Path
from typing import Dict, Union, Tuple, Any

import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict



def get_cfg_from_ckpt(ckpt_path: str,
                      return_as_dict: bool = False) -> Tuple[Union[DictConfig, Dict],
                                                             Dict[str, Any]]:
    """
    Load the config directly from the cfg arg passed into the model during training.

    Also returns the model checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(cfg_dict)

    if return_as_dict:
        return cfg_dict, ckpt
    return cfg, ckpt


def resume_ckpt_cfg(current_cfg: DictConfig) -> DictConfig:
    """
    Handles logic for obtaining a config for resuming training from a checkpoint.

    resume_opts.use_current_cfg: If True, the current config is used instead of the checkpoint config
    resume_opts.overrides: a dict of overrides to apply to the cfg

    Parameters:
    - current_cfg (DictConfig): The active configuration.

    Returns:
    - DictConfig: The updated configuration for resuming training
    """
    resume_opts = current_cfg.resume

    lit_model_cfg, lit_model_ckpt = get_cfg_from_ckpt(resume_opts.ckpt_path)

    # use the current cfg instead of the checkpoint cfg
    cfg = current_cfg if resume_opts.use_current_cfg else lit_model_cfg

    # set a temporary exp_name if not provided to avoid overwriting the resumed experiment
    with open_dict(resume_opts):
        resume_opts.overrides.exp_name = resume_opts.overrides.get("exp_name", f"{lit_model_cfg.exp_name}_resumed")

    with open_dict(cfg):
        # retain resume info in new cfg
        cfg.resume = resume_opts

        if ("train" in resume_opts.overrides) and (resume_opts.overrides.train.get("compile_model") is False):
            # get rid of _orig_mod. prefix in saved compiled models
            lit_model_ckpt["state_dict"] = repair_state_dict(lit_model_ckpt["state_dict"])

        # if trying to override optimizer, throw an error
        if "optim" in resume_opts.overrides:
            raise ValueError("Cannot override optimizer when resuming training, we probably have to switch away from pure Lightning to do this better...")

    # apply specific overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.create(resume_opts.overrides))

    return cfg, lit_model_ckpt


def repair_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair the state dict to avoid issues with loading the model checkpoint due to torch.compile().

    https://github.com/pytorch/pytorch/issues/101107

    Parameters:
    - state_dict (Dict[str, Any]): The model state dict.

    Returns:
    - Dict[str, Any]: The repaired state dict.
    """
    pairings = [
        (src_key, src_key.replace("_orig_mod.", ""))
        for src_key in state_dict.keys()
    ]
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = state_dict[src_key]

    return out_state_dict
