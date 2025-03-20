"""
Utils for sampling from backbone generation models.
"""
from typing import Any

from omegaconf import DictConfig, OmegaConf
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


def get_bb_gen_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a backbone generation model.
    """
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ad_ckpt).eval()
    bb_gen_model = {"model": lit_ad_model.model,
                    "sampling_cfg": OmegaConf.load(cfg.sampling_cfg),
                    "device": device}

    return bb_gen_model
