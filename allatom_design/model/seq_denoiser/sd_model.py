import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import cat_bb_scn, stack_aux_traj
from allatom_design.data.pdb_utils import *
from allatom_design.eval import sampling_utils
from allatom_design.interpolants.sd_interpolants.mar_interpolant import MAR
from allatom_design.interpolants.sd_interpolants.sd_interpolant import \
    SDInterpolant
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.minimpnn_denoiser import \
    MiniMPNNDenoiser


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
        self.register_buffer("bb_mean", torch.tensor(0.0))
        self.register_buffer("bb_std", torch.tensor(1.0))

        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = (self.bb_std, self.scn_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)
        self.interpolant = get_interpolant(cfg.interpolant)


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                t: Optional[TensorType["b", float]] = None,  # (t_bb, t_scn) if multimodal
                ) -> Dict[str, TensorType["b ..."]]:
        """
        batch should contain:
        - x: TensorType["b n a 3", float]
        - residue_index: TensorType["b n", int]
        - seq_mask: TensorType["b n", float]
        - cond_labels_in: Dict[str, TensorType["b", int]]
        """
        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)
        outputs = {}

        # Apply interpolant to noise the inputs
        interpolant_out = self.interpolant(batch, t)
        batch["x_noised"] = interpolant_out["x_noised"]
        batch["aatype_noised"] = interpolant_out["aatype_noised"]

        # During training, keep track of certain additional features
        aux_inputs = {
            "x": batch["x"],  # ground truth coordinates
            "aatype": batch["aatype"],  # ground truth aatype
            "ghost_atom_mask": batch["ghost_atom_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "t_scd": batch.get("t_scd", None),  # scalar; fix t_scd (sidechain diffusion time) if provided, usually for eval
        }

        # Denoise coords
        _, _, aux_preds = self.denoiser(batch["x_noised"], batch["aatype_noised"], None,
                                        batch["residue_index"], batch["seq_mask"],
                                        cond_labels_in=batch["cond_labels_in"],
                                        aux_inputs=aux_inputs)

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


    def sidechain_pack(self,
                       x: TensorType["b n a 3", float],
                       aatype: TensorType["b n", int],
                       seq_mask: TensorType["b n", float],
                       residue_index: TensorType["b n", int],
                       scd_inputs: Dict[str, Any],
                       **sampling_kwargs) -> Tuple[TensorType["b n", int],
                                                   Dict[str, torch.Tensor]]:
        """
        Given backbone and sequence, denoise sidechain atoms (sidechain packing).
        """
        # Fix sequence time to 1 with a single pass
        t_seq = torch.tensor([1.0, 1.0]).to(x.device)[None].expand(x.shape[0], -1)  # [B, 2]
        timesteps = t_seq
        target_dims = (t_seq.shape[1], *aatype.shape)

        # Override aatype with the input aatype during sequence denoising
        aatype_override = aatype.unsqueeze(0).expand(*target_dims)
        aatype_override_mask = torch.ones_like(aatype)
        aatype_override_mask = aatype_override_mask.unsqueeze(0).expand(*target_dims).long()  # view not clone to save a bit of memory

        return self.sample(x, seq_mask, residue_index, timesteps,
                           aatype_decoding_order_mode="random",  # does not matter for sidechain packing
                           aatype_override=aatype_override, aatype_override_mask=aatype_override_mask,
                           scd_inputs=scd_inputs,
                           **sampling_kwargs)


    def sample(self,
               x: TensorType["b n a 3", float],
               seq_mask: TensorType["b n", float],
               residue_index: TensorType["b n", int],
               timesteps: TensorType["b s+1", float],  # timesteps for t_seq
               aatype_decoding_order_mode: str,
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
        S = timesteps.shape[1] - 1
        B, N, A, _ = x.shape

        # Set up backbone input
        x0 = x.clone()
        x0[..., rc.non_bb_idxs, :] = 0.0  # zero out sidechain atoms

        # Handle default overrides
        # TODO: handle xt overrides, especially important for conditioning on known sequence/sidechain atoms? or maybe we want to do this directly in aatype/x input
        if aatype_override is None:
            # dummy values
            aatype_override = torch.full((S + 1, B, N), fill_value=rc.restype_order_with_x["X"], device=residue_index.device)
            aatype_override_mask = torch.zeros((S + 1, B, N), device=residue_index.device, dtype=torch.long)  # don't override anything

        # Add sidechain diffusion inputs
        aux_inputs["scd"] = scd_inputs

        # Sample aatype prior
        aatype_noised = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()  # TODO: make seq prior use MASK rather than UNK

        # Get residue decoding order
        aatype_decoding_order = sampling_utils.get_decoding_order(mode=aatype_decoding_order_mode, seq_mask=seq_mask, timesteps=timesteps)
        aux_inputs["lengths"] = seq_mask.sum(dim=-1)

        # Initialize trajectories
        xt_traj = []
        aatype_t_traj, aatype_pred_traj = [], []
        seq_logits_traj = []
        scn_diffusion_aux_traj = []

        # Run denoising steps
        denoiser_fn = partial(self.denoiser,
                              residue_index=residue_index,
                              seq_mask=seq_mask,
                              cond_labels_in=cond_labels,
                              aux_inputs=aux_inputs,
                              is_sampling=True)

        xt = x0
        aatype_t = aatype_noised
        for i in tqdm(range(S), leave=False, desc="Sampling..."):
            # get current and next timesteps
            t = timesteps[:, i]
            t_next = timesteps[:, i + 1]

            aatype_t = aatype_t * (1 - aatype_override_mask[i]) + aatype_override[i] * aatype_override_mask[i]  # override aatype for inputs
            xt, aatype_t, aux_preds = self.interpolant.denoising_step(denoiser_fn,
                                                                      xt, aatype_t,
                                                                      t=t, t_next=t_next,
                                                                      aatype_decoding_order=aatype_decoding_order,
                                                                      aux_inputs=aux_inputs)
            aatype_t = aatype_t * (1 - aatype_override_mask[i + 1]) + aatype_override[i + 1] * aatype_override_mask[i + 1]  # override aatype for outputs  # TODO: should we override self-cond input too?

            if getattr(self.denoiser, "use_self_conditioning_seq", False):
                # Apply sequence self-conditioning
                denoiser_fn = partial(denoiser_fn, seq_self_cond=aux_preds["seq_logits"])

            # Save trajectory outputs
            xt_traj.append(xt.cpu())
            aatype_t_traj.append(aatype_t.cpu())
            aatype_pred_traj.append(aux_preds["aatype_pred"].cpu())
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
        Save samples from the denoiser to PDB files. Handles post-processing of denoiser outputs.
        Samples should contain the following keys:
        - x_denoised: Tensor["b n a 3", float]
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        - pred_aatype: Tensor["b n", int]

        Args:
        - bb_only_samples: whether the samples come from a backbone-only model
        """
        final_atom37_positions = samples["x_denoised"]
        residue_index = samples["residue_index"]
        seq_mask = samples["seq_mask"]
        aatype = samples["pred_aatype"]

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
                 sigma_data: TensorType[(), float]  # can also be a tuple of sigmas for ca, nco
                 ) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "minimpnn":
        return MiniMPNNDenoiser(cfg, sigma_data)
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
