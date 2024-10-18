import copy
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_bb_scn, stack_aux_traj
from allatom_design.data.pdb_utils import *
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


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
               chain_index: TensorType["b n", int],
               timesteps: Tuple[TensorType["S+1", float]],  # joint timesteps (T_ad, T_sd); same for all batch elements
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

        # Initialize sequence / sidechain prior (all masked, time t=0)
        xt_scn = torch.zeros(B, N, len(rc.non_bb_idxs), 3, device=residue_index.device)
        aatype_t = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"])
        mlm_mask = torch.zeros_like(seq_mask)
        aux["mlm_mask"] = mlm_mask

        # Initialize trajectories
        xt_traj, aatype_t_traj, aatype_pred_traj = [], [], []
        aux_preds_ad_traj, aux_preds_sd_traj = [], []

        S = timesteps[0].shape[0] - 1  # number of sampling steps
        for i in tqdm(range(S), desc="Sampling", leave=False):
            T = tuple(ts[i] for ts in timesteps)
            T_next = tuple(ts[i+1] for ts in timesteps)

            # Sample backbone
            T_sd = T[1].expand(B)
            x1_bb, aux_preds_ad = self.sample_backbone(
                xt_scn=xt_scn,
                aatype_noised=aatype_t,
                t_sd=T_sd,
                residue_index=residue_index,
                seq_mask=seq_mask,
                mlm_mask=mlm_mask,
                cond_labels=cond_labels,
                **ad_sampling_inputs,
            )

            # Sample sequence
            xt = cat_bb_scn(x1_bb, xt_scn)
            x1, aatype_pred, aux_preds_sd = self.sample_seq(
                x=xt,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_index,
                cond_labels=cond_labels,
                **sd_sampling_inputs
            )

            # Set up partial diffusion for next backbone diffusion
            T_next_ad, T_next_sd = T_next
            S_bb = ad_sampling_inputs["timesteps"].shape[1] - 1  # number of backbone sampling steps
            partial_diff_inputs = {"x_bb_in": x1_bb, "num_steps_partial": math.ceil(((1 - T_next_ad) * S_bb).item())}
            ad_sampling_inputs["partial_diff_inputs"] = partial_diff_inputs

            # Noise sequence back to time T_next_sd
            T_next_sd = T_next_sd.expand(B)
            xt, aatype_t, mlm_mask = self.sd_model.interpolant.noise_samples(x1, aatype_pred, T_next_sd, seq_mask)
            xt_scn = xt[..., rc.non_bb_idxs, :]

            # Save auxiliary outputs
            xt_traj.append(cat_bb_scn(x1_bb, xt_scn).cpu())  # TODO: rename this to x1 traj?
            aatype_t_traj.append(aatype_t.cpu())
            aatype_pred_traj.append(aatype_pred.cpu())
            aux_preds_ad_traj.append(aux_preds_ad)
            aux_preds_sd_traj.append(aux_preds_sd)

        # preprocess trajectories
        aux["xt_traj"] = torch.stack(xt_traj, dim=1)  # [B, S, N, A, X]
        aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=1)  # [B, S, N]
        aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=1)  # [B, S, N]

        # preprocess aux trajectories
        # aux["aux_preds_ad_traj"] = stack_aux_traj(aux_preds_ad_traj)  # values are [B, S, S_bb, N, A, X]
        # aux["aux_preds_sd_traj"] = stack_aux_traj(aux_preds_sd_traj)  # values are [B, S, S_seq, ...]

        return x1, aatype_pred, aux


    def sample_backbone(
        self,
        xt_scn: TensorType["b n 33 3", float],  # TODO: handle partial diffusion somewhere.
        aatype_noised: TensorType["b n", int],
        t_sd: TensorType["b n", float],  # timestep of sequence design inputs
        residue_index: TensorType["b n", int],
        seq_mask: TensorType["b n", float],
        mlm_mask: TensorType["b n", float],
        timesteps: TensorType["b s+1", float],
        xt_override: Optional[TensorType["s+1 b n a 3", float]] = None,
        xt_override_mask: Optional[TensorType["s+1 b n a 3", float]] = None,
        cond_labels: Dict[str, TensorType["b", int]] = {},
        noise_schedule: Optional[NoiseSchedule] = None,
        churn_cfg: Optional[Dict[str, float]] = None,
        autoguidance_cfg: Optional[Dict[str, Any]] = None,  # autoguidance config
        partial_diff_inputs: Dict[str, Any] = {},
        ):
        """
        Run diffusion (or partial diffusion) to generate backbone, conditioned on noisy sequence.
        """
        B, N = residue_index.shape
        S_bb = timesteps.shape[-1] - 1

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
            "autoguidance_cfg": autoguidance_cfg,
            # overrides
            "xt_override": xt_override,
            "xt_override_mask": xt_override_mask,
            "x_bb_in": partial_diff_inputs.get("x_bb_in", None),
            "num_steps_partial": partial_diff_inputs.get("num_steps_partial", None),
        }

        # Run atom denoiser
        x1_bb, aux_preds_bb = self.ad_model.denoiser(xt_scn=xt_scn, aatype_noised=aatype_noised, t_sd=t_sd,
                                                     residue_index=residue_index, seq_mask=seq_mask, mlm_mask=mlm_mask,
                                                     cond_labels_in=cond_labels, aux_inputs=aux_inputs_bb, is_sampling=True)
        aux = aux_preds_bb["bb_diffusion_aux"]
        return x1_bb, aux


    def sample_seq(self,
                   x: TensorType["b n a 3", float],
                   seq_mask: TensorType["b n", float],
                   residue_index: TensorType["b n", int],
                   chain_index: TensorType["b n", int],
                   timesteps: Tuple[TensorType["b s+1", float]],
                   aatype_decoding_order_mode: str,
                   num_corrector_steps: int,
                   corrector_step_ratio: TensorType["1", float],
                   cond_labels: Dict[str, TensorType["b", int]],
                   aatype_override: Optional[TensorType["s+1 b n", int]] = None,  # for fixed-sequence sampling, e.g. in sidechain packing
                   aatype_override_mask: Optional[TensorType["s+1 b n", int]] = None,
                   scd_inputs: Dict[str, Any] = {},  # sidechain diffusion inputs
                   ):
        """
        scd_inputs should contain the following keys:
        - num_steps: int
        - timesteps: TensorType["b S_scd+1", float]
        - churn_cfg: Dict[str, Any]
        - noise_schedule: Dict[str, Any]
        - autoguidance_cfg: Optional[Dict[str, Any]]  # for autoguidance, None if not used
        """
        aux, aux_inputs = {}, {}
        S_seq = timesteps.shape[1] - 1
        B, N, A, _ = x.shape

        # Set up backbone input
        x0 = x.clone()
        x0[..., rc.non_bb_idxs, :] = 0.0  # zero out sidechain atoms

        # Handle default overrides
        # TODO: handle xt overrides, especially important for conditioning on known sequence/sidechain atoms? or maybe we want to do this directly in aatype/x input
        if aatype_override is None:
            # dummy values
            aatype_override = torch.full((S_seq + 1, B, N), fill_value=rc.restype_order_with_x["X"], device=residue_index.device)
            aatype_override_mask = torch.zeros((S_seq + 1, B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        # Sample aatype prior
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()  # TODO: make seq prior use MASK rather than UNK

        # Get residue decoding order
        aatype_decoding_order = sampling_utils.get_decoding_order(mode=aatype_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps)
        aux_inputs["lengths"] = seq_mask.sum(dim=-1)
        aux_inputs["seq_mlm_mask"] = torch.zeros_like(seq_mask).float()  # start with all masked tokens

        # Initialize trajectories
        xt_traj = []
        aatype_t_traj, aatype_pred_traj = [], []
        seq_logits_traj = []
        scn_diffusion_aux_traj = []

        # Run denoising steps
        seq_denoiser_fn = partial(self.sd_model.denoiser,
                              residue_index=residue_index,
                              chain_encoding=chain_index,
                              seq_mask=seq_mask,
                              cond_labels_in=cond_labels,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        unmasked_prev = torch.zeros_like(seq_mask, dtype=torch.bool)

        # Run unmasking steps
        unmasking_fn = partial(sampling_utils.unmask,
                               aatype_decoding_order=aatype_decoding_order,
                               aatype_decoding_order_mode=aatype_decoding_order_mode,
                               seq_mask=seq_mask,
                               aux_inputs=aux_inputs)

        timesteps_K = torch.ceil(timesteps * aux_inputs["lengths"][:,None]).long()
        for i in tqdm(range(S_seq), leave=False, desc="Sampling..."):
            # get current and next timesteps
            t, t_next = timesteps[:, i], timesteps[:, i + 1]

            # get next K residues to unmask
            K_next = timesteps_K[:, i + 1]

            # override aatype for inputs
            aatype_t = aatype_t * (1 - aatype_override_mask[i]) + aatype_override[i] * aatype_override_mask[i]

            # Run sequence denoiser
            x1_pred, aatype_pred, aux_preds = seq_denoiser_fn(xt, aatype_t, t=t)

            # Unmask according to timestep and decoding order
            xt, aatype_t, unmasked_prev = unmasking_fn(xt, aatype_t, x1_pred,
                                                       aatype_pred, aux_preds,
                                                       unmasked_prev, K_next)
            if i > 1:
                for j in range(num_corrector_steps):
                    # corrector step where we mask and denoise equally
                    K_corrector = torch.ceil(K_next * corrector_step_ratio).long()
                    x1_pred, aatype_pred, aux_preds, unmasked_prev = self.interpolant.corrector_step(seq_denoiser_fn,
                                                                                      xt, aatype_t, K_corrector,
                                                                                      unmasked_prev,
                                                                                      t=t, aux_inputs=aux_inputs)
                    # Unmask according to timestep and decoding order
                    xt, aatype_t, unmasked_prev = unmasking_fn(xt, aatype_t, x1_pred,
                                                              aatype_pred, aux_preds,
                                                              unmasked_prev, K_corrector)

            aatype_t = aatype_t * (1 - aatype_override_mask[i + 1]) + aatype_override[i + 1] * aatype_override_mask[i + 1]  # override aatype for outputs  # TODO: should we override self-cond input too?

            if getattr(self.sd_model.denoiser, "use_self_conditioning_seq", False):
                # Apply sequence self-conditioning
                seq_denoiser_fn = partial(seq_denoiser_fn, seq_self_cond=aux_preds["seq_logits"])

            # Save trajectory outputs
            xt_traj.append(xt.cpu())
            aatype_t_traj.append(aatype_t.cpu())
            aatype_pred_traj.append(aatype_pred.cpu())
            seq_logits_traj.append(aux_preds["seq_logits"].cpu())

            if scd_inputs.get("return_scn_diffusion_aux", False):
                scn_diffusion_aux_traj.append({k: v.cpu() for k, v in aux_preds["scn_diffusion_aux"].items()})

        aux["xt_traj"] = torch.stack(xt_traj, dim=1)
        aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=1)
        aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=1)
        aux["seq_logits_traj"] = torch.stack(seq_logits_traj, dim=1)
        aux["scn_diffusion_aux_traj"] = scn_diffusion_aux_traj
        aux["seq_mask"] = seq_mask

        # preprocess diffusion aux traj
        if scd_inputs.get("return_scn_diffusion_aux", False):
            aux["scn_diffusion_aux_traj"] = stack_aux_traj(scn_diffusion_aux_traj, dim=1)  # values are shape (B, S, S_scd, N, A, 3)
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


    @staticmethod
    def save_trajs_to_pdb(traj_aux: Dict[str, Any],
                          residue_index: TensorType["b n", int],
                          chain_index: TensorType["b n", int],
                          save_traj_mask: List[bool],
                          save_traj_steps: List[int],
                          save_ad_traj_steps: List[int],
                          save_sd_traj_steps: List[int],
                          save_scd_traj_steps: List[int],
                          x_traj_key: str,
                          filenames: List[str],
                          traj_conect: bool,
                          align_models_to_idx: Optional[int] = None,):
        """
        Save trajectories from the denoiser to PDB files.

        Args:
        - traj_aux: auxiliary output from sampling trajectory
        - residue_index
        - chain_index
        - save_traj_mask: list of bools indicating which main trajectories to save
        - save_traj_steps: list of indices indicating which steps along the main trajectory to save
        - x_traj_key: key in traj_aux for the denoised atom positions
            - "x1_traj" gives the x1 prediction for each timestep
            - "xt_traj" gives the current state along the trajectory
        - filenames: list of filenames to save the trajectories to
        - traj_conect: whether to include CONECT records in the PDB files
        """

        B = traj_aux["seq_mask"].shape[0]
        device = traj_aux["seq_mask"].device
        for i in range(B):
            if save_traj_mask[i]:
                if x_traj_key == "xt_traj":
                    # === Save xt with aatype_t for each step in joint trajectory === #
                    x_traj = traj_aux["xt_traj"][i, save_traj_steps]  # [S, N, A, 3]
                    aatype_traj = traj_aux["aatype_t_traj"][i, save_traj_steps]  # [S, N]
                    atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, N, A]



                # if x_traj_key == "full_xt_traj":
                #     # === Assemble full xt traj from backbone and sequence diffusion === #
                #     xt_bb_traj = traj_aux["aux_preds_ad_traj"]["xt_bb_traj"][i, save_traj_steps, save_ad_traj_steps]  # [S, S_bb, N, 4, 3]
                #     xt_scn_traj = traj_aux["aux_preds_sd_traj"]["xt_scn_traj"][i, save_traj_steps, save_sd_traj_steps, save_scd_traj_steps]  # [S, S_sd, S_scd, N, 33, 3]

                #     # make sidechain diffusion reflect the most recent step of backbone diffusion
                #     xt_scn_traj = torch.cat([xt_bb_traj[:, :-1, None], xt_scn_traj], dim=-2)  # [S, S_sd, S_scd, N, 37, 3]

                #     # assemble full xt traj by interleaving bb and scn
                #     S, S_bb, N, _, _ = xt_bb_traj.shape
                #     _, S_sd, S_scd, _, _, _ = xt_scn_traj.shape
                #     xt_traj = torch.zeros((S, S_bb + S_sd + S_scd, N, rc.atom_type_num, 3), device=device)
                #     xt_traj[:, :S_bb] = xt_bb_traj
                #     xt_traj[:, S_bb:] = rearrange(xt_scn_traj, "s s_sd s_scd n a x -> s (s_sd s_scd) n a x")

                #     # flatten
                #     xt_traj = rearrange(xt_traj, "s s_all n a x -> (s s_all) n a x")

                #     # aatype traj



                # if aatype_traj_key in ["aatype_pred_traj", "aatype_t_traj"]:
                #     # Save aatype_pred or aatype_t traj
                #     aatype_traj = traj_aux[aatype_traj_key][i, save_traj_steps]
                #     atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, N, A]
                #     x_traj = traj_aux[x_traj_key][i, save_traj_steps]

                # elif x_traj_key in ["x1_scn_traj", "xt_scn_traj"]:
                #     # Save sidechain diffusion traj
                #     B, S, S_scd, N, A, _ = traj_aux["scn_diffusion_aux_traj"][x_traj_key].shape

                #     # index with both save_traj_steps and save_diff_traj_steps
                #     grid_S, grid_S_scd = torch.meshgrid(torch.tensor(save_traj_steps), torch.tensor(save_diff_traj_steps), indexing='ij')

                #     # get aatype and atom mask
                #     aatype_traj = traj_aux["aatype_t_traj"].unsqueeze(2).expand(-1, -1, S_scd, -1)  # expand along diffusion steps dim, [B, S, S_scd, N, A]
                #     aatype_traj = aatype_traj[i, grid_S, grid_S_scd]  # [S, S_scd, N]
                #     atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, S_scd, N, A]

                #     # construct full atom positions from sidechain diffusion aux
                #     x_scn_traj = traj_aux["scn_diffusion_aux_traj"][x_traj_key][i, grid_S, grid_S_scd]  # [S, S_scd, N, A, X]
                #     x_bb_traj = rearrange(traj_aux["xt_traj"][i, -1][..., rc.bb_idxs, :], "n a x -> 1 1 n a x").expand(x_scn_traj.shape[0], x_scn_traj.shape[1], -1, -1, -1)
                #     x_traj = cat_bb_scn(x_bb_traj, x_scn_traj)  # [S, S_scd, N, A, X]

                #     # flatten steps
                #     x_traj = rearrange(x_traj, "s S_scd n a x -> (s S_scd) n a x")
                #     aatype_traj = rearrange(aatype_traj, "s S_scd n -> (s S_scd) n")
                #     atom_mask = rearrange(atom_mask, "s S_scd n a -> (s S_scd) n a")

                # else:
                #     assert False, f"Unknown x_traj_key: {x_traj_key}"

                traj_feats = {
                    "aatype": aatype_traj,
                    "atom_positions": x_traj,
                    "atom_mask": atom_mask,
                    "residue_index": residue_index[i].unsqueeze(0).expand(aatype_traj.shape[0], -1),
                    "chain_index": chain_index[i].unsqueeze(0).expand(aatype_traj.shape[0], -1),
                    "b_factors": None
                }
                traj_feats = {k: v.cpu() if v is not None else v for k, v  in traj_feats.items()}
                write_to_pdb_frames(**traj_feats, filename=filenames[i], mode="aa", conect=traj_conect, align_models_to_idx=align_models_to_idx)

