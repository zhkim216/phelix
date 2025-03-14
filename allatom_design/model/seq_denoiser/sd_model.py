import copy
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_bb_scn, get_rc_tensor, stack_aux_traj
from allatom_design.data.pdb_utils import *
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.sd_interpolants.mar_interpolant import MAR
from allatom_design.interpolants.sd_interpolants.sd_interpolant import \
    SDInterpolant
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser


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
        self.interpolant = get_interpolant(cfg.interpolant)

        self.drop_residx_p = cfg.get("drop_residx_p", 0.0)

        # Backbone noise options
        denoiser_cfg = cfg.denoiser
        fampnn_cfg = denoiser_cfg.get("fampnn", denoiser_cfg.get("minimpnn", None))  # default to FAMPNN / MiniMPNN settings for backwards compatibility
        self.augment_eps = denoiser_cfg.get("augment_eps", fampnn_cfg.get("augment_eps", None))
        self.per_residue_eps = denoiser_cfg.get("per_residue_eps", fampnn_cfg.get("per_residue_eps", False))
        self.max_eps = denoiser_cfg.get("max_eps", fampnn_cfg.get("max_eps", None))


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,
                aux_inputs_override: Optional[Dict[str, TensorType["b ..."]]] = None,  # for providing overrides to aux_inputs
                skip_interpolant: bool = False,  # for dpo-finetuning, interpolant is applied to the batch outside of the model
                ) -> Dict[str, TensorType["b ..."]]:
        """
        batch should contain:
        - x: TensorType["b n a 3", float]
        - residue_index: TensorType["b n", int]
        - seq_mask: TensorType["b n", float]
        """
        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)
        outputs = {}

        ## Apply interpolant to mask the inputs ##
        if not skip_interpolant:
            interpolant_out = self.interpolant(batch, t)
            batch["x_noised"] = interpolant_out["x_noised"]
            batch["aatype_noised"] = interpolant_out["aatype_noised"]
            batch["seq_mlm_mask"] = interpolant_out["seq_mlm_mask"]  # 1 for unmasked aatype
            batch["scn_mlm_mask"] = interpolant_out["scn_mlm_mask"]  # 1 for unmasked sidechains

        ## Get random backbone noise ##
        noise, noise_labels = self.get_random_noise(batch["seq_mask"])

        ## Randomly drop out residue index ##
        drop_residx = None
        if self.training:
            # at train time, randomly drop out all residue indices for each batch element
            residue_index = batch["residue_index"]
            drop_residx = torch.rand(residue_index.shape[0], device=residue_index.device) < self.drop_residx_p  # [B]

        # During training, keep track of certain additional features
        aux_inputs = {
            "x": batch["x"],  # ground truth coordinates
            "aatype": batch["aatype"],  # ground truth aatype
            "atom_mask": batch["atom_mask"],  # ground truth atom mask; includes missing, ghost, and pad atoms
            "t_scd": batch.get("t_scd", None),  # scalar; fix t_scd (sidechain diffusion time) if provided, usually for eval
            "seq_mlm_mask": batch["seq_mlm_mask"],
            "scn_mlm_mask": batch["scn_mlm_mask"],
            "noise": noise,
            "noise_labels": noise_labels,
            "drop_residx": drop_residx,
        }
        aux_inputs.update(aux_inputs_override or {})  # override aux_inputs if provided

        # Denoise coords
        _, _, aux_preds = self.denoiser(batch["x_noised"], batch["aatype_noised"],
                                        batch["residue_index"], batch['chain_index'],
                                        batch["seq_mask"], batch["missing_atom_mask"],
                                        batch["scn_mlm_mask"],
                                        aux_inputs=aux_inputs)

        # Additional outputs for computing loss
        outputs.update(aux_preds)

        return outputs

    def score(self,
              x: TensorType["b n 37 3", float],
              aatype: TensorType["b n", int],
              seq_mask: TensorType["b n", float],
              missing_atom_mask: TensorType["b n 37", float],  # 1 where atoms are missing
              scn_mlm_mask: TensorType["b n", float],  # 0 for masked sidechains
              residue_index: TensorType["b n", int],
              chain_index: TensorType["b n", int],
              return_embeddings: bool = False,
        ) -> TensorType["b n"]:
        atom_mask_noised = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)  # 0 for ghost atoms; X only has backbone atoms
        atom_mask_noised = atom_mask_noised * seq_mask.unsqueeze(-1)  # mask out padding
        atom_mask_noised = atom_mask_noised * (1 - missing_atom_mask)  # mask out missing atoms
        atom_mask_noised[..., rc.non_bb_idxs] = atom_mask_noised[..., rc.non_bb_idxs] * scn_mlm_mask.unsqueeze(-1)  # mask out masked sidechain atoms

        # Run denoiser and get logits
        seq_logits, mpnn_feature_dict = self.denoiser.seq_design_module(x,
                                                                        aatype,
                                                                        seq_mask,
                                                                        atom_mask_noised,
                                                                        residue_index,
                                                                        chain_index,
                                                                        return_encoder_embeds=return_embeddings)
        log_probs = F.log_softmax(seq_logits, dim=-1)
        if return_embeddings:
            return log_probs, mpnn_feature_dict
        return log_probs


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


    def get_random_noise(self, seq_mask: TensorType["b n", float]) -> Tuple[TensorType["b n 14 3", float], TensorType["b n", float]]:
        ## Choose random backbone noise ##
        B, N = seq_mask.shape

        if self.per_residue_eps:
            # per-residue noise. Unlike Cho et al., we sample noise stds from a uniform distribution and apply different noise to each atom in a residue
            if self.training and self.augment_eps > 0:
                # training: randomly sample noise labels
                noise_labels = torch.rand_like(seq_mask, device=seq_mask.device) * self.augment_eps  # sample std for each residue from uniform distribution
                noise = torch.randn((B, N, 14, 3), device=seq_mask.device) * rearrange(noise_labels, "b n -> b n 1 1")  # random noise for each atom
            else:
                # eval: assume no noise
                noise, noise_labels = None, None

        else:
            # global noise, similar to ProteinMPNN
            noise_labels = None
            if self.training and self.augment_eps > 0:
                # training: add randomly sampled noise to input
                noise = self.augment_eps * torch.randn((B, N, 14, 3), device=seq_mask.device)
                noise_labels = None
            else:
                # eval: assume no noise
                noise, noise_labels = None, None

        return noise, noise_labels


    def sidechain_pack(self,
                       x: TensorType["b n a 3", float],
                       aatype: TensorType["b n", int],
                       seq_mask: TensorType["b n", float],
                       missing_atom_mask: TensorType["b n 37", float],  # 1 where atoms are missing
                       residue_index: TensorType["b n", int],
                       chain_index: TensorType["b n", int],
                       scn_override_mask: Optional[TensorType["b n", int]] = None,
                       aatype_override_mask: Optional[TensorType["b n", int]] = None,
                       scd_inputs: Dict[str, Any] = {}):
        """
        Given backbone and sequence, denoise sidechain atoms (sidechain packing).

        Also supports packing partial sequence with partial sidechains through aatype_override_mask and scn_override_mask.


        scd_inputs should contain the following keys:
        - num_steps: int
        - timesteps: TensorType["b S_scd+1", float]
        - churn_cfg: Dict[str, Any]
        - noise_schedule: Dict[str, Any]
        """
        aux, aux_inputs = {}, {}
        B, N, A, _ = x.shape

        # Override aatype with the input aatype during sequence denoising
        if aatype_override_mask is None:
            # if not provided, assume full sequence
            aatype_override_mask = seq_mask.clone()

        # Set sidechain to fully masked
        if scn_override_mask is None:
            # if not provided, assume no sidechains
            scn_override_mask = torch.zeros_like(seq_mask)

        # Set up structure input dependent on structure mask
        x0 = x.clone()
        x0[:,:,rc.non_bb_idxs,:] =  x0[:,:,rc.non_bb_idxs,:] * scn_override_mask[:,:,None,None]

        # Sample aatype prior dependency on aatype mask
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
        aatype_noised = torch.where(aatype_override_mask == 1, aatype, aatype_noised)

        # Override aatype in sidechain diffusion as well
        scd_inputs["aatype_override"] = aatype_noised
        scd_inputs["aatype_override_mask"] = aatype_override_mask

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        seq_mlm_mask = torch.zeros_like(seq_mask).float() + aatype_override_mask # start with all masked tokens, other than partial seq
        scn_mlm_mask = torch.zeros_like(seq_mask).float() + scn_override_mask # start with all masked tokens, other than partial scn
        assert torch.all((seq_mlm_mask - scn_mlm_mask) >= 0), "Unmasked sidechains should be a subset of unmasked sequence"

        # Run denoising steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              missing_atom_mask=missing_atom_mask,
                              chain_encoding=chain_index,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        psce_t = torch.zeros((B, N, len(rc.non_bb_idxs)), device=x.device)

        # Run sequence denoiser to get packed sidechains
        x1_pred, _, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)

        # Unmask sidechains and sidechain confidence to match seq_mlm_mask
        xt = sampling_utils.unmask(xt, x1_pred, scn_mlm_mask, seq_mlm_mask)
        psce_t = sampling_utils.unmask(psce_t, aux_preds["scn_diffusion_aux"]["psce"], scn_mlm_mask, seq_mlm_mask)

        aux["psce"] = psce_t
        aux["seq_mask"] = seq_mask

        return xt, aatype_t, aux


    def sample(self,
               x: TensorType["b n a 3", float],
               aatype: TensorType["b n", int],
               seq_mask: TensorType["b n", float],
               missing_atom_mask: TensorType["b n a 3", float],  # 1 where atoms are missing
               residue_index: TensorType["b n", int],
               chain_index:  TensorType["b n", int],
               timesteps: TensorType["b s+1", float],  # timesteps for t_seq
               temperature: float,  # 0.0 for argmax / greedy sampling
               aatype_decoding_order_mode: str,
               seq_only: bool = False,  # only sample sequence
               repack_last: bool = False,  # repack last step after sampling the sequence
               psce_threshold: Optional[float] = None,  # during design, only keep sidechains with psce below threshold; None to keep all
               scn_override_mask: Optional[TensorType["b n", int]] = None,
               aatype_override_mask: Optional[TensorType["b n", int]] = None,
               restrict_pos_aatype: Optional[Tuple[TensorType["b n", float],
                                                   TensorType["b n k", int]]] = None,  # restrict aatype sampling at certain positions
               omit_aas: Optional[List[str]] = None,  # omit certain amino acids from sampling, e.g. ["C", "G"]
               noise_labels: Optional[Union[float, TensorType["b n"]]] = None,  # per-residue noise label
               add_noise: bool = False,
               scd_inputs: Dict[str, Any] = {},  # sidechain diffusion inputs
               ):
        """
        scd_inputs should contain the following keys:
        - num_steps: int
        - timesteps: TensorType["b S_scd+1", float]
        - churn_cfg: Dict[str, Any]
        - noise_schedule: Dict[str, Any]
        """
        aux, aux_inputs = {}, {}
        S = timesteps.shape[1] - 1
        B, N, A, _ = x.shape

        # Handle default overrides
        if aatype_override_mask is None:
            aatype_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        if scn_override_mask is None:
            scn_override_mask = torch.zeros((B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Handle aatype restrictions
        aux_inputs["omit_aas"] = omit_aas
        aux_inputs["restrict_pos_aatype"] = restrict_pos_aatype

        # Add in noise label
        aux_inputs["noise_labels"] = noise_labels

        # Add in noise to the input if requested
        if add_noise:
            assert noise_labels is not None and self.per_residue_eps
            if type(noise_labels) is float:
                noise_labels = torch.full((B, N), fill_value=noise_labels, device=seq_mask.device)  # assume constant noise label
            noise = torch.randn((B, N, 14, 3), device=seq_mask.device) * rearrange(noise_labels, "b n -> b n 1 1")  # random noise for each atom
            aux_inputs["noise"] = noise


        # Set up structure input dependent on structure mask
        x0 = x.clone()
        x0[:,:,rc.non_bb_idxs,:] =  x0[:,:,rc.non_bb_idxs,:] * scn_override_mask[:,:,None,None]

        # Sample aatype prior dependency on aatype mask
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
        aatype_noised = torch.where(aatype_override_mask == 1, aatype, aatype_noised)

        # Override aatype in sidechain diffusion as well
        scd_inputs["aatype_override"] = aatype_noised
        scd_inputs["aatype_override_mask"] = aatype_override_mask

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        # Get residue decoding order
        seq_mlm_mask = torch.zeros_like(seq_mask).float() + aatype_override_mask # start with all masked tokens, other than partial seq
        scn_mlm_mask = torch.zeros_like(seq_mask).float() + scn_override_mask # start with all masked tokens, other than partial scn
        aatype_decoding_order = sampling_utils.get_decoding_order(mode=aatype_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps, mlm_mask_prev=seq_mlm_mask)
        aux_inputs["lengths"] = seq_mask.sum(dim=-1)
        aux_inputs["temperature"] = temperature

        # Initialize trajectories
        xt_traj = []
        aatype_t_traj, aatype_pred_traj = [], []
        psce_t_traj = []
        seq_logits_traj = []
        scn_diffusion_aux_traj = []

        # Set up function for updating mlm mask
        mask_update_fn = partial(sampling_utils.update_mlm_mask,
                                 aatype_decoding_order=aatype_decoding_order,
                                 aatype_decoding_order_mode=aatype_decoding_order_mode,
                                 seq_mask=seq_mask)

        # Run denoising steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              missing_atom_mask=missing_atom_mask,
                              chain_encoding=chain_index,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        seq_probs_t = torch.zeros((B, N, len(rc.restypes_with_x)), device=x.device)  # keep track of unscaled sequence probabilities as we decode
        psce_t = torch.zeros((B, N, len(rc.non_bb_idxs)), device=x.device)  # keep track of sidechain confidence as we decode

        # Handle differences in provided sequence and sidechain masks
        if torch.any((aatype_override_mask - scn_override_mask) < 0):
            raise ValueError('Sidechain cannot be fixed at any positions where sequence is not fixed')

        if torch.any((aatype_override_mask - scn_override_mask) > 0) and not seq_only:
            # If we have more sequence than sidechains, pack all sidechains to catch up to aatype_override_mask
            xt, _, aux_preds_pack = self.sidechain_pack(xt, aatype_t, seq_mask, missing_atom_mask, residue_index, chain_index, scn_override_mask, aatype_override_mask, scd_inputs)
            psce_t = aux_preds_pack["psce"]  # reflect confidence in packed sidechains
            scn_mlm_mask = seq_mlm_mask.clone()

            if psce_threshold is not None:
                # Re-mask sidechains with low confidence, but only if we are not at the last step

                # get mask based on per-residue confidence
                atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_t)[..., rc.non_bb_idxs]
                psce_t_per_res = (psce_t * atom_mask_scn).sum(dim=-1) / atom_mask_scn.sum(dim=-1).clamp(min=1)  # average confidence per residue
                print(f"Number before thresholding: {scn_mlm_mask.sum(dim=-1)[0]}")
                scn_mlm_mask = scn_mlm_mask * (psce_t_per_res <= psce_threshold).float()
                print(f"Number after thresholding: {scn_mlm_mask.sum(dim=-1)[0]}")

                # apply mask
                xt[..., rc.non_bb_idxs, :] = xt[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()
                psce_t = psce_t * rearrange(scn_mlm_mask, "b n -> b n 1")

        # Get timesteps based on the number of unmasked residues
        # assert (seq_mlm_mask == scn_mlm_mask).all(), "Expecting sequence and sidechain masks to be the same before sampling starts."
        num_partial = seq_mlm_mask.sum(dim=-1).long()
        timesteps_K = torch.ceil(timesteps * (aux_inputs["lengths"][:, None] - num_partial[:,None])).long()  # timestep schedule is defined relative to masked residues
        timesteps_K += num_partial[:,None]

        for i in tqdm(range(S), leave=False, desc="Sampling..."):
            # get next K residues to unmask
            K_next = timesteps_K[:, i + 1]

            # Run sequence denoiser
            x1_pred, aatype_pred, aux_preds = denoiser_fn(xt, aatype_t, scn_mlm_mask=scn_mlm_mask)

            # Update mask
            seq_mlm_mask_prev, scn_mlm_mask_prev = seq_mlm_mask.clone(), scn_mlm_mask.clone()
            seq_mlm_mask = mask_update_fn(seq_mlm_mask,
                                          K=K_next, aatype_pred=aatype_pred,
                                          scaled_seq_probs=aux_preds["scaled_seq_probs"],
                                          psce=aux_preds["scn_diffusion_aux"]["psce"])
            scn_mlm_mask = seq_mlm_mask.clone() if not seq_only else scn_override_mask  # default to user-provided sidechains if seq_only

            # Unmask sequence, sidechains, and sidechain confidence
            aatype_t = sampling_utils.unmask(aatype_t, aatype_pred, seq_mlm_mask_prev, seq_mlm_mask)
            xt = sampling_utils.unmask(xt, x1_pred, scn_mlm_mask_prev, scn_mlm_mask)
            seq_probs_t = sampling_utils.unmask(seq_probs_t, aux_preds["seq_probs"], seq_mlm_mask_prev, seq_mlm_mask)
            psce_t = sampling_utils.unmask(psce_t, aux_preds["scn_diffusion_aux"]["psce"], scn_mlm_mask_prev, scn_mlm_mask)

            if (psce_threshold is not None) and (i != S - 1):
                # Re-mask sidechains with low confidence, but only if we are not at the last step

                # get mask based on per-residue confidence
                atom_mask_scn = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_t)[..., rc.non_bb_idxs]
                psce_t_per_res = (psce_t * atom_mask_scn).sum(dim=-1) / atom_mask_scn.sum(dim=-1).clamp(min=1)  # average confidence per residue
                scn_mlm_mask = scn_mlm_mask * (psce_t_per_res <= psce_threshold).float()

                # apply mask
                xt[..., rc.non_bb_idxs, :] = xt[..., rc.non_bb_idxs, :] * rearrange(scn_mlm_mask, "b n -> b n 1 1").float()
                psce_t = psce_t * rearrange(scn_mlm_mask, "b n -> b n 1")

            # Save trajectory outputs
            xt_traj.append(xt.cpu())
            aatype_t_traj.append(aatype_t.cpu())
            psce_t_traj.append(psce_t.cpu())
            aatype_pred_traj.append(aatype_pred.cpu())
            seq_logits_traj.append(aux_preds["seq_logits"].cpu())

            if scd_inputs.get("return_scn_diffusion_aux", False):
                scn_diffusion_aux_traj.append({k: aux_preds["scn_diffusion_aux"][k].cpu() for k in ["xt_scn_traj", "x1_scn_traj"]})

        if repack_last:
            # Repack the structure after sampling the sequence (ignoring the provided sidechains)
            xt, _, aux_preds_pack = self.sidechain_pack(xt, aatype_t, seq_mask, missing_atom_mask, residue_index, chain_index,
                                                        scn_override_mask,  # start from the provided sidechains
                                                        seq_mlm_mask,  # pack to the known sequence
                                                        scd_inputs)
            psce_t = aux_preds_pack["psce"]
            scn_mlm_mask = seq_mlm_mask.clone()

        aux["xt_traj"] = torch.stack(xt_traj, dim=1)
        aux["aatype_t_traj"] = torch.stack(aatype_t_traj, dim=1)
        aux["aatype_pred_traj"] = torch.stack(aatype_pred_traj, dim=1)
        aux["seq_probs"] = seq_probs_t
        aux["psce"] = psce_t
        aux["psce_t_traj"] = torch.stack(psce_t_traj, dim=1)
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
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")


def get_interpolant(cfg: DictConfig) -> SDInterpolant:
    """
    Get the interpolant specified in the config.
    """
    if cfg.name == "mar":
        return MAR(cfg)
    else:
        raise ValueError(f"Unknown interpolant: {cfg.name}")


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
