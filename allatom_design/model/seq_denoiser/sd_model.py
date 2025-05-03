import copy
import math
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_bb_scn, get_rc_tensor, stack_aux_traj
from allatom_design.data.pdb_utils import *
from allatom_design.eval.eval_utils import sampling_utils
from allatom_design.model.seq_denoiser.denoisers.atom_mpnn_denoiser import \
    AtomMPNNDenoiser
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser
from chroma.layers import complexity


class SeqDenoiser(nn.Module):
    """
    Sequence denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task

        # Data scaling parameters
        # scale sidechains separately from the backbone
        self.register_buffer("bb_std", torch.tensor(1.0))
        self.register_buffer("bb_mean", torch.tensor(0.0))

        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = (self.bb_std, self.scn_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)

        # Mask selector
        self.mask_selector = cfg.mask_selector


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                t: TensorType["b", float] | None = None
                ) -> dict[str, TensorType["b ..."]]:
        outputs = {}

        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)

        # Sample sequence and atom conditioning masks
        batch["seq_cond_mask"] = self.mask_selector.sample_seq_cond_mask(batch, t)  # 1 if we should condition on the restype, 0 otherwise
        batch["atom_cond_mask"] = self.mask_selector.sample_atom_cond_mask(batch)  # 1 if we should condition on the atom, 0 otherwise

        # Ensure the conditioning masks only contain non-pad, resolved entries
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_pad_mask"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        # Denoise sequence
        _, aux_preds = self.denoiser(batch)

        # Additional outputs for computing loss
        outputs.update(aux_preds)

        return outputs


    def set_scale_factors(self,
                          scale_factors: Dict[str, Tuple[float, float]]):
        bb_mean, bb_std = scale_factors["bb"]
        self.bb_mean.data = torch.tensor(bb_mean)
        self.bb_std.data = torch.tensor(bb_std)
        print(f"Setting bb_mean: {bb_mean}, bb_std: {bb_std}")

        scn_mean, scn_std = scale_factors["scn"]
        self.scn_mean.data = torch.tensor(scn_mean)
        self.scn_std.data = torch.tensor(scn_std)
        print(f"Setting scn_mean: {scn_mean}, scn_std: {scn_std}")


    def sample(self,
               batch: dict[str, TensorType["b ..."]],
               sampling_inputs: dict[str, Any]):

        # Handle inference noise labels
        batch["noise_labels"] = sampling_inputs.get("noise_labels", None)
        batch["noise"] = None

        if batch["noise_labels"] is not None:
            raise NotImplementedError("Noise labels are not implemented yet")

        if sampling_inputs["add_noise"]:
            raise NotImplementedError("Adding noise is not implemented yet")

        # Choose sampling method
        if sampling_inputs["use_potts_sampling"]:
            res_type_pred = self.denoiser.potts_sample(batch, sampling_inputs)
        else:
            raise NotImplementedError("Only Potts sampling is currently implemented")
        return res_type_pred


    # def sample(self,
    #            x: TensorType["b n a 3", float],
    #            aatype: TensorType["b n", int],
    #            seq_mask: TensorType["b n", float],
    #            missing_atom_mask: TensorType["b n a 3", float],  # 1 where atoms are missing
    #            residue_index: TensorType["b n", int],
    #            chain_index:  TensorType["b n", int],
    #            timesteps: TensorType["b s+1", float],  # timesteps for t_seq
    #            temperature: float,  # 0.0 for argmax / greedy sampling
    #            aatype_decoding_order_mode: str,
    #            seq_only: bool = False,  # only sample sequence
    #            repack_last: bool = False,  # repack last step after sampling the sequence
    #            psce_threshold: Optional[float] = None,  # during design, only keep sidechains with psce below threshold; None to keep all
    #            scn_override_mask: Optional[TensorType["b n", int]] = None,
    #            aatype_override_mask: Optional[TensorType["b n", int]] = None,
    #            pos_restrict_aatype: Optional[Tuple[TensorType["b n", float],
    #                                                TensorType["b n k", int]]] = None,  # restrict aatype sampling at certain positions
    #            omit_aas: Optional[List[str]] = None,  # omit certain amino acids from sampling, e.g. ["C", "G"]
    #            noise_labels: Optional[Union[float, TensorType["b n"]]] = None,  # per-residue noise label
    #            add_noise: bool = False,
    #            use_potts_sampling: bool = False,
    #            potts_sampling_cfg: Dict[str, Any] = {},
    #            ):
    #     """
    #     scd_inputs should contain the following keys:
    #     - num_steps: int
    #     - timesteps: TensorType["b S_scd+1", float]
    #     - churn_cfg: Dict[str, Any]
    #     - noise_schedule: Dict[str, Any]
    #     """
    #     aux, aux_inputs = {}, {}
    #     S = timesteps.shape[1] - 1
    #     B, N, A, _ = x.shape

    #     # Handle default overrides
    #     if aatype_override_mask is None:
    #         aatype_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

    #     if scn_override_mask is None:
    #         scn_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

    #     # Handle aatype restrictions
    #     aux_inputs["omit_aas"] = omit_aas
    #     aux_inputs["pos_restrict_aatype"] = pos_restrict_aatype

    #     # Add in noise label
    #     aux_inputs["noise_labels"] = noise_labels

    #     # Add in noise to the input if requested
    #     if add_noise:
    #         assert noise_labels is not None, "Need noise labels to know how much noise to add to backbone for add_noise option"
    #         if type(noise_labels) is float:
    #             noise_labels = torch.full((B, N), fill_value=noise_labels, device=seq_mask.device)  # assume constant noise label
    #         noise = torch.randn((B, N, 14, 3), device=seq_mask.device) * rearrange(noise_labels, "b n -> b n 1 1")  # random noise for each atom
    #         aux_inputs["noise"] = noise

    #     # Set up structure input dependent on structure mask
    #     x0 = x.clone()
    #     x0[:,:,rc.non_bb_idxs,:] =  x0[:,:,rc.non_bb_idxs,:] * scn_override_mask[:,:,None,None]

    #     # Sample aatype prior dependency on aatype mask
    #     aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
    #     aatype_noised = torch.where(aatype_override_mask == 1, aatype, aatype_noised)

    #     # Override aatype in sidechain diffusion as well
    #     scd_inputs["aatype_override"] = aatype_noised
    #     scd_inputs["aatype_override_mask"] = aatype_override_mask

    #     # Add sidechain diffusion inputs
    #     aux_inputs["scd"] = scd_inputs

    #     # Get residue decoding order
    #     seq_mlm_mask = torch.zeros_like(seq_mask).float() + aatype_override_mask # start with all masked tokens, other than partial seq
    #     scn_mlm_mask = torch.zeros_like(seq_mask).float() + scn_override_mask # start with all masked tokens, other than partial scn
    #     aatype_decoding_order = sampling_utils.get_decoding_order(mode=aatype_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps, mlm_mask_prev=seq_mlm_mask)
    #     aux_inputs["lengths"] = seq_mask.sum(dim=-1)
    #     aux_inputs["temperature"] = temperature

    #     # Initialize trajectories
    #     xt_traj = []
    #     aatype_t_traj, aatype_pred_traj = [], []
    #     psce_t_traj = []
    #     seq_logits_traj = []
    #     scn_diffusion_aux_traj = []

    #     # Set up function for updating mlm mask
    #     mask_update_fn = partial(sampling_utils.update_mlm_mask,
    #                              aatype_decoding_order=aatype_decoding_order,
    #                              aatype_decoding_order_mode=aatype_decoding_order_mode,
    #                              seq_mask=seq_mask)

    #     # Run denoising steps
    #     denoiser_fn = partial(self.denoiser,
    #                           residue_index=residue_index,
    #                           seq_mask=seq_mask,
    #                           missing_atom_mask=missing_atom_mask,
    #                           chain_encoding=chain_index,
    #                           aux_inputs=aux_inputs,
    #                           is_sampling=True)

    #     xt = x0
    #     aatype_t = aatype_noised
    #     seq_probs_t = torch.zeros((B, N, len(rc.restypes_with_x)), device=x.device)  # keep track of unscaled sequence probabilities as we decode
    #     psce_t = torch.zeros((B, N, len(rc.non_bb_idxs)), device=x.device)  # keep track of sidechain confidence as we decode

    #     # Handle differences in provided sequence and sidechain masks
    #     if torch.any((aatype_override_mask - scn_override_mask) < 0):
    #         raise ValueError('Sidechain cannot be fixed at any positions where sequence is not fixed')

    #     if torch.any((aatype_override_mask - scn_override_mask) > 0) and not seq_only:
    #         # If we have more sequence than sidechains, pack all sidechains to catch up to aatype_override_mask
    #         xt, _, aux_preds_pack = self.sidechain_pack(xt, aatype_t, seq_mask, missing_atom_mask, residue_index, chain_index, scn_override_mask, aatype_override_mask, scd_inputs)
    #         psce_t = aux_preds_pack["psce"]  # reflect confidence in packed sidechains
    #         scn_mlm_mask = seq_mlm_mask.clone()

    #         if psce_threshold is not None:
    #             # Re-mask sidechains with low confidence, but only if we are not at the last step

    #             # get mask based on per-residue confidence
    #             atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_t)[..., rc.non_bb_idxs]
    #             psce_t_per_res = (psce_t * atom_mask_scn).sum(dim=-1) / atom_mask_scn.sum(dim=-1).clamp(min=1)  # average confidence per residue
    #             print(f"Number before thresholding: {scn_mlm_mask.sum(dim=-1)[0]}")
    #             scn_mlm_mask = scn_mlm_mask * (psce_t_per_res <= psce_threshold).float()
    #             print(f"Number after thresholding: {scn_mlm_mask.sum(dim=-1)[0]}")

    #             # apply mask
    #             xt[..., rc.non_bb_idxs, :] = xt[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()
    #             psce_t = psce_t * rearrange(scn_mlm_mask, "b n -> b n 1")

    #     # Get timesteps based on the number of unmasked residues
    #     # assert (seq_mlm_mask == scn_mlm_mask).all(), "Expecting sequence and sidechain masks to be the same before sampling starts."
    #     num_partial = seq_mlm_mask.sum(dim=-1).long()
    #     timesteps_K = torch.ceil(timesteps * (aux_inputs["lengths"][:, None] - num_partial[:,None])).long()  # timestep schedule is defined relative to masked residues
    #     timesteps_K += num_partial[:,None]

    #     if use_potts_sampling:
    #         assert self.denoiser.seq_design_module.use_potts, "Denoiser must be trained with Potts decoder to use Potts sampling"
    #         return self.potts_sample(
    #             potts_sampling_cfg=potts_sampling_cfg,
    #             denoiser_fn=denoiser_fn,
    #             xt=xt,
    #             aatype_t=aatype_t,
    #             seq_mask=seq_mask,
    #             seq_mlm_mask=seq_mlm_mask,
    #             scn_mlm_mask=scn_mlm_mask,
    #             omit_aas=omit_aas,
    #         )

    #     for i in tqdm(range(S), leave=False, desc="Sampling..."):
    #         # get next K residues to unmask
    #         K_next = timesteps_K[:, i + 1]

    #         # Run sequence denoiser
    #         x1_pred, aatype_pred, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)

    #         # Update mask
    #         seq_mlm_mask_prev, scn_mlm_mask_prev = seq_mlm_mask.clone(), scn_mlm_mask.clone()
    #         seq_mlm_mask = mask_update_fn(seq_mlm_mask,
    #                                       K=K_next, aatype_pred=aatype_pred,
    #                                       scaled_seq_probs=aux_preds["scaled_seq_probs"],
    #                                       psce=aux_preds["scn_diffusion_aux"]["psce"])
    #         scn_mlm_mask = seq_mlm_mask.clone() if not seq_only else scn_override_mask  # default to user-provided sidechains if seq_only

    #         # Unmask sequence, sidechains, and sidechain confidence
    #         aatype_t = sampling_utils.unmask(aatype_t, aatype_pred, seq_mlm_mask_prev, seq_mlm_mask)
    #         xt = sampling_utils.unmask(xt, x1_pred, scn_mlm_mask_prev, scn_mlm_mask)
    #         seq_probs_t = sampling_utils.unmask(seq_probs_t, aux_preds["seq_probs"], seq_mlm_mask_prev, seq_mlm_mask)
    #         psce_t = sampling_utils.unmask(psce_t, aux_preds["scn_diffusion_aux"]["psce"], scn_mlm_mask_prev, scn_mlm_mask)

    #         if (psce_threshold is not None) and (i != S - 1):
    #             # Re-mask sidechains with low confidence, but only if we are not at the last step

    #             # get mask based on per-residue confidence
    #             atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_t)[..., rc.non_bb_idxs]
    #             psce_t_per_res = (psce_t * atom_mask_scn).sum(dim=-1) / atom_mask_scn.sum(dim=-1).clamp(min=1)  # average confidence per residue
    #             scn_mlm_mask = scn_mlm_mask * (psce_t_per_res <= psce_threshold).float()

    #             # apply mask
    #             xt[..., rc.non_bb_idxs, :] = xt[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()
    #             psce_t = psce_t * rearrange(scn_mlm_mask, "b n -> b n 1")

    #         # Save trajectory outputs
    #         xt_traj.append(xt.cpu())
    #         aatype_t_traj.append(aatype_t.cpu())
    #         psce_t_traj.append(psce_t.cpu())
    #         aatype_pred_traj.append(aatype_pred.cpu())
    #         seq_logits_traj.append(aux_preds["seq_logits"].cpu())

    #         if scd_inputs.get("return_scn_diffusion_aux", False):
    #             scn_diffusion_aux_traj.append({k: aux_preds["scn_diffusion_aux"][k].cpu() for k in ["xt_scn_traj", "x1_scn_traj"]})

    #     if repack_last:
    #         # Repack the structure after sampling the sequence (ignoring the provided sidechains)
    #         xt, _, aux_preds_pack = self.sidechain_pack(xt, aatype_t, seq_mask, missing_atom_mask, residue_index, chain_index,
    #                                                     scn_override_mask,  # start from the provided sidechains
    #                                                     seq_mlm_mask,  # pack to the known sequence
    #                                                     scd_inputs)
    #         psce_t = aux_preds_pack["psce"]
    #         scn_mlm_mask = seq_mlm_mask.clone()

    #     aux["xt_traj"] = torch.stack(xt_traj, dim=1)
    #     aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=1)
    #     aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=1)
    #     aux["seq_probs"] = seq_probs_t
    #     aux["psce"] = psce_t
    #     aux["psce_t_traj"] = torch.stack(psce_t_traj, dim=1)
    #     aux["seq_logits_traj"] = torch.stack(seq_logits_traj, dim=1)
    #     aux["scn_diffusion_aux_traj"] = scn_diffusion_aux_traj
    #     aux["seq_mask"] = seq_mask

    #     # preprocess diffusion aux traj
    #     if scd_inputs.get("return_scn_diffusion_aux", False):
    #         aux["scn_diffusion_aux_traj"] = stack_aux_traj(scn_diffusion_aux_traj, dim=1)  # values are shape (B, S, S_scd, N, A, 3)

    #     return xt, aatype_t, aux


    def potts_sample(self,
                     potts_sampling_cfg: Dict[str, Any],
                     denoiser_fn: Callable,
                     xt: TensorType["b n a 3", float],
                     aatype_t: TensorType["b n", int],
                     seq_mask: TensorType["b n", float],
                     seq_mlm_mask: TensorType["b n", float],
                     scn_mlm_mask: TensorType["b n", float],
                     omit_aas: Optional[List[str]] = None,
                     ):
        if seq_mlm_mask.sum() != 0 and scn_mlm_mask.sum() != 0:
            raise NotImplementedError("For now, sequence and sidechain must be fully masked for Potts sampling")

        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]

        B, N = aatype_t.shape
        logits_init = torch.zeros((B, N, len(rc.restypes_with_x)), device=aatype_t.device).float()

        # Handle banned amino acids
        ban_S = {"X"}
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_S = [rc.restype_order_with_x[aa] for aa in ban_S]

        # Initialize random sequence and sampling masks
        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init, mask_sample=(1 - seq_mlm_mask), S=aatype_t, ban_S=ban_S
        )

        # Complexity regularization
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        symmetry_order = None
        if regularization == "LCP":
            # C_complexity = (
            #     C
            #     if symmetry_order is None
            #     else C[:, : C.shape[1] // symmetry_order]
            # )
            C_complexity = seq_mask.clone()  # TODO: is C for multi-chain?
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)
            # edge_idx_coloring, mask_ij_coloring = complexity.graph_lcp(C, edge_idx, mask_ij)

        _, _, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)
        potts_decoder_aux = aux_preds["potts_decoder_aux"]

        S_sample, _ = self.denoiser.seq_design_module.decoder_S_potts.sample(
            potts_decoder_aux["h"],
            potts_decoder_aux["J"],
            potts_decoder_aux["edge_idx"],
            potts_decoder_aux["mask_i"],
            potts_decoder_aux["mask_ij"],
            S=S_init,
            mask_sample=mask_sample,
            temperature=potts_temperature,
            num_sweeps=potts_sweeps,
            penalty_func=penalty_func,
            proposal=potts_proposal,
            rejection_step=(potts_proposal == "chromatic"),
            verbose=False,
            edge_idx_coloring=edge_idx_coloring,
            mask_ij_coloring=mask_ij_coloring,
        )
        aux = defaultdict(lambda: None)  # TODO: temporary
        xt = xt.clone()  # TODO: temporary
        return xt, S_sample, aux


    @staticmethod
    def save_samples_to_pdb(samples: Dict[str, TensorType["b ..."]],
                            filenames: List[str],
                            ) -> None:
        """
        Save samples from the denoiser to PDB files. Handles post-processing of denoiser outputs.
        Samples should contain the following keys:
        - x_denoised: Tensor["b n a 3", float]
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        - pred_aatype: Tensor["b n", int]
        - psce: Tensor["b n 33", float]

        Args:
        - bb_only_samples: whether the samples come from a backbone-only model
        """
        final_atom37_positions = samples["x_denoised"]
        residue_index = samples["residue_index"]
        seq_mask = samples["seq_mask"]
        aatype = samples["pred_aatype"]
        chain_index = samples["chain_index"]

        # Create atom mask, including backbone atoms even for unknown aatype
        atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=aatype.device)[aatype] * seq_mask[..., None]
        atom_mask = atom_mask * (1 - samples["missing_atom_mask"])  # mask out missing atoms

        # Set b-factors to predicted Sidechain Error (PSCE)
        b_factors = torch.zeros_like(atom_mask, dtype=torch.float32).cpu()
        if samples.get("psce") is not None:
            b_factors[..., rc.non_bb_idxs] = samples["psce"].cpu()

        feats = {
            "aatype": aatype,
            "atom_positions": final_atom37_positions,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
            "chain_index": chain_index,
            "b_factors":b_factors
        }

        feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}  # move to cpu
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")


    @staticmethod
    def save_trajs_to_pdb(traj_aux: Dict[str, Any],
                          residue_index: TensorType["b n", int],
                          chain_index: TensorType["b n", int],
                          save_traj_mask: List[bool],
                          save_traj_steps: List[int],
                          save_diff_traj_steps: List[int],
                          x_traj_key: str,
                          aatype_traj_key: str,
                          filenames: List[str],
                          traj_conect: bool,
                          align_models_to_idx: Optional[int] = None,):
        """
        Save trajectories from the denoiser to PDB files. Handles post-processing of denoiser outputs.

        Args:
        - traj_aux: auxiliary output from sampling trajectory
        - residue_index
        - chain_index
        - save_traj_mask: list of bools indicating which main trajectories to save
        - save_traj_steps: list of indices indicating which steps along the main trajectory to save
        - save_diff_traj_steps: for each step along the main traj we're saving, list of indices indicating which steps along each diffusion trajectory to save
        - x_traj_key: key in traj_aux for the denoised atom positions
            - "x1_traj" gives the x1 prediction for each timestep
            - "xt_traj" gives the current state along the trajectory
            - "x1_scn_traj" gives the x1 prediction along the sidechain diffusion trajectory
            - "xt_scn_traj" gives the current state along the sidechain diffusion trajectory
        - aatype_traj_key: key in traj_aux for the predicted aatype
            - "aatype_pred_traj" gives the prediction of noiseless aatype for each timestep
            - "aatype_t_traj" gives the current state along the trajectory
        - filenames: list of filenames to save the trajectories to
        - traj_conect: whether to include CONECT records in the PDB files
        """

        B = traj_aux["seq_mask"].shape[0]
        device = traj_aux["seq_mask"].device
        for i in range(B):
            if save_traj_mask[i]:
                if aatype_traj_key in ["aatype_pred_traj", "aatype_t_traj"]:
                    # Save aatype_pred or aatype_t traj
                    aatype_traj = traj_aux[aatype_traj_key][i, save_traj_steps]
                    atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, N, A]
                    x_traj = traj_aux[x_traj_key][i, save_traj_steps]

                elif x_traj_key in ["x1_scn_traj", "xt_scn_traj"]:
                    # Save sidechain diffusion traj
                    B, S, S_scd, N, A, _ = traj_aux["scn_diffusion_aux_traj"][x_traj_key].shape

                    # index with both save_traj_steps and save_diff_traj_steps
                    grid_S, grid_S_scd = torch.meshgrid(torch.tensor(save_traj_steps), torch.tensor(save_diff_traj_steps), indexing='ij')

                    # get aatype and atom mask
                    aatype_traj = traj_aux["aatype_t_traj"].unsqueeze(2).expand(-1, -1, S_scd, -1)  # expand along diffusion steps dim, [B, S, S_scd, N, A]
                    aatype_traj = aatype_traj[i, grid_S, grid_S_scd]  # [S, S_scd, N]
                    atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, S_scd, N, A]

                    # construct full atom positions from sidechain diffusion aux
                    x_scn_traj = traj_aux["scn_diffusion_aux_traj"][x_traj_key][i, grid_S, grid_S_scd]  # [S, S_scd, N, A, X]
                    x_bb_traj = rearrange(traj_aux["xt_traj"][i, -1][..., rc.bb_idxs, :], "n a x -> 1 1 n a x").expand(x_scn_traj.shape[0], x_scn_traj.shape[1], -1, -1, -1)
                    x_traj = cat_bb_scn(x_bb_traj, x_scn_traj)  # [S, S_scd, N, A, X]

                    # flatten steps
                    x_traj = rearrange(x_traj, "s S_scd n a x -> (s S_scd) n a x")
                    aatype_traj = rearrange(aatype_traj, "s S_scd n -> (s S_scd) n")
                    atom_mask = rearrange(atom_mask, "s S_scd n a -> (s S_scd) n a")

                else:
                    assert False, f"Unknown x_traj_key: {x_traj_key}"

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


def get_denoiser(cfg: DictConfig,
                 sigma_data: TensorType[(), float]
                 ) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "fampnn" or cfg.name == "minimpnn":  # backwards compatibility
        return FAMPNNDenoiser(cfg, sigma_data)
    elif cfg.name == "atom_mpnn":
        return AtomMPNNDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


def truncated_half_normal_like(x: TensorType["...", float],
                               std: float, max_val: Optional[float]) -> TensorType["...", float]:
    if max_val is None:
        # return half-normal with no truncation
        return torch.abs(torch.randn_like(x) * std)
    u = torch.rand_like(x)
    truncated_factor = torch.erf(torch.tensor(max_val / (math.sqrt(2) * std)))
    u_scaled = u * truncated_factor
    samples = std * math.sqrt(2) * torch.erfinv(u_scaled)
    return samples
