import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_bb_scn
from allatom_design.data.pdb_utils import *
from allatom_design.eval import sampling_utils
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule


class AllAtomModel():
    """
    All-atom model. Composed of a pretrained atom denoiser and a pretrained sequence denoiser.
    """
    def __init__(self, lit_ad_model: LitAtomDenoiser, lit_sd_model: LitSeqDenoiser):
        super().__init__()

        self.lit_ad_model = lit_ad_model
        self.lit_sd_model = lit_sd_model

        self.ad_model = lit_ad_model.model
        self.sd_model = lit_sd_model.model


    def sample(self,
               lengths: TensorType["b", int],
               residue_index: TensorType["b n", int],
               ad_sampling_inputs: Dict[str, Any],
               sd_sampling_inputs: Dict[str, Any],
               cond_labels: Dict[str, TensorType["b", int]] = {},
               ) -> Tuple[TensorType["b n a 3", float],
                          TensorType["b n", int],
                          Dict[str, torch.Tensor]]:
        """
        Draw samples from the allatom model.
        """
        B, N = residue_index.shape

        aux = {}  # keep track of auxiliary outputs

        # Create seq mask
        ranges = torch.arange(N, device=residue_index.device).expand(B, N)
        seq_mask = (ranges < lengths[:, None]).float()
        aux["seq_mask"] = seq_mask.cpu()

        # Initialize initial coordinates (all 0s)
        xt = torch.zeros((B, N, rc.atom_type_num, 3), device=residue_index.device)

        # Initialize sequence prior (all masked)
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"])
        mlm_mask = torch.zeros_like(seq_mask)
        aux["mlm_mask"] = mlm_mask

        # Sample backbone
        x1_bb, aux_preds_bb = self.sample_backbone(
            xt_scn=None,  # once we get partial diffusion working, feed in x0
            aatype_noised=aatype_noised,
            residue_index=residue_index,
            seq_mask=seq_mask,
            mlm_mask=mlm_mask,
            cond_labels=cond_labels,
            **ad_sampling_inputs,
        )

        # Sample sequence
        xt[..., rc.bb_idxs, :] = x1_bb
        x1, aatype_pred, aux_preds_seq = self.sample_seq(
            x=xt,
            seq_mask=seq_mask,
            residue_index=residue_index,
            cond_labels=cond_labels,
            **sd_sampling_inputs
        )

        return x1, aatype_pred, aux


    def sample_backbone(
        self,
        xt_scn: TensorType["b n 33 3", float],  # TODO: handle partial diffusion somewhere.
        aatype_noised: TensorType["b n", int],
        residue_index: TensorType["b n", int],
        seq_mask: TensorType["b n", float],
        mlm_mask: TensorType["b n", float],
        timesteps: Tuple[TensorType["b s+1", float]],  # tuple of timesteps for (t_ca, t_nco)
        xt_override: Optional[TensorType["s+1 b n a 3", float]] = None,
        xt_override_mask: Optional[TensorType["s+1 b n a 3", float]] = None,
        cond_labels: Dict[str, TensorType["b", int]] = {},
        noise_schedule: Tuple[Optional[NoiseSchedule]] = None,  # noise schedule for (t_ca, t_nco)
        churn_cfg: Tuple[Optional[Dict[str, float]]] = None, # churn config for (t_ca, t_nco)
        ):
        """
        Run diffusion (or partial diffusion) to generate backbone, conditioned on noisy sequence.
        """
        B, N = residue_index.shape
        S_bb = timesteps[0].shape[-1] - 1

        # Handle xt overrides
        if xt_override is None:
            # dummy values
            xt_override = torch.zeros(1, device=residue_index.device).expand(S_bb + 1, B, N, rc.atom_type_num, 3)
            xt_override_mask = torch.zeros(1, device=residue_index.device).expand(S_bb + 1, B, N, rc.atom_type_num, 3)


        # Construct atom denoiser inputs
        aux_inputs_bb = {
            "num_steps": S_bb,
            "timesteps": timesteps,
            "churn_cfg": churn_cfg,
            "noise_schedule": noise_schedule,
            # overrides
            "xt_override": xt_override,
            "xt_override_mask": xt_override_mask,
        }

        # Run atom denoiser
        x1_bb, aux_preds_bb = self.ad_model.denoiser(xt_scn=xt_scn, aatype_noised=aatype_noised, t=None,
                                                     residue_index=residue_index, seq_mask=seq_mask, mlm_mask=mlm_mask,
                                                     cond_labels_in=cond_labels, aux_inputs=aux_inputs_bb, is_sampling=True)

        return x1_bb, aux_preds_bb


    def sample_seq(self,
                   x: TensorType["b n a 3", float],
                   seq_mask: TensorType["b n", float],
                   residue_index: TensorType["b n", int],
                   timesteps: Tuple[TensorType["b s+1", float]],
                   aatype_decoding_order_mode: str,
                   cond_labels: Dict[str, TensorType["b", int]],
                   aatype_override: Optional[TensorType["s+1 b n", int]] = None,  # for fixed-sequence sampling, e.g. in sidechain packing
                   aatype_override_mask: Optional[TensorType["s+1 b n", int]] = None,
                   scd_inputs: Dict[str, Any] = {},  # sidechain diffusion inputs
                   ):
        B, N, A, _ = x.shape
        S_seq = timesteps[0].shape[-1] - 1

        aux, aux_inputs_seq = {}, {}

        # Set up backbone input
        x0 = x.clone()
        x0[..., rc.non_bb_idxs, :] = 0.0  # zero out non-backbone atoms

        # Handle default overrides
        # TODO: handle xt overrides, especially important for conditioning on known sequence/sidechain atoms
        if aatype_override is None:
            # dummy values
            aatype_override = torch.full((S_seq + 1, B, N), fill_value=rc.restype_order_with_x["X"], device=residue_index.device)
            aatype_override_mask = torch.zeros((S_seq + 1, B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Add sidechain diffusion inputs
        aux_inputs_seq["scd"] = scd_inputs

        # Sample aatype prior
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()  # TODO: make seq prior use MASK rather than UNK

        # Get residue decoding order
        aatype_decoding_order = sampling_utils.get_decoding_order(mode=aatype_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps)
        aux_inputs_seq["lengths"] = seq_mask.sum(dim=-1)

        # Run denoising steps
        seq_denoiser_fn = partial(self.sd_model.denoiser,
                                  residue_index=residue_index,
                                  seq_mask=seq_mask,
                                  cond_labels_in=cond_labels,
                                  aux_inputs=aux_inputs_seq,
                                  is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        for i in tqdm(range(S_seq), leave=False, desc="Sampling..."):
            # get current and next timesteps
            t = timesteps[:, i]
            t_next = timesteps[:, i + 1]

            aatype_t = aatype_t * (1 - aatype_override_mask[i]) + aatype_override[i] * aatype_override_mask[i]  # override aatype for inputs
            xt, aatype_t, aux_preds = self.sd_model.interpolant.denoising_step(seq_denoiser_fn,
                                                                               xt, aatype_t,
                                                                               t=t, t_next=t_next,
                                                                               aatype_decoding_order=aatype_decoding_order,
                                                                               aux_inputs=aux_inputs_seq)
            aatype_t = aatype_t * (1 - aatype_override_mask[i + 1]) + aatype_override[i + 1] * aatype_override_mask[i + 1]  # override aatype for outputs  # TODO: should we override self-cond input too?

            if getattr(self.sd_model.denoiser, "use_self_conditioning_seq", False):
                # Apply sequence self-conditioning
                seq_denoiser_fn = partial(seq_denoiser_fn, seq_self_cond=aux_preds["seq_logits"])


        return xt, aatype_t, aux


    @staticmethod
    def save_samples_to_pdb(samples: Dict[str, TensorType["b ..."]],
                            filenames: List[str],
                            ) -> None:
        """
        Save samples from the allatom denoiser to PDB files.
        Samples should contain the following keys:
        - x_denoised: Tensor["b n a 3", float]
        - aatype_denoised: Tensor["b n", int]
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        - pred_aatype: Tensor["b n", int]

        Args:
        - bb_only_samples: whether the samples come from a backbone-only model
        """
        final_atom37_positions = samples["x_denoised"]
        aatype = samples["aatype_denoised"]
        seq_mask = samples["seq_mask"]
        residue_index = samples["residue_index"]

        # Create atom mask, including backbone atoms even for unknown aatype
        atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=aatype.device)[aatype] * seq_mask[..., None]

        feats = {
            "aatype": aatype,
            "atom_positions": final_atom37_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": torch.zeros_like(residue_index),  # TODO: support multiple chains
            "b_factors": torch.ones_like(atom_mask, dtype=torch.float32),
        }

        feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}  # move to cpu
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")
