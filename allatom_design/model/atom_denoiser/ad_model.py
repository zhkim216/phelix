import copy
from typing import Any, Tuple

import torch.nn as nn
from omegaconf import DictConfig

from allatom_design.data.pdb_utils import *
from allatom_design.model.atom_denoiser.denoisers.dit_denoiser import \
    DiTDenoiser


class AtomDenoiser(nn.Module):
    """
    Atom denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.task = cfg.task

        # Data scaling parameters
        self.register_buffer("bb_std", torch.tensor(1.0))
        self.sigma_data = self.bb_std

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: Dict[str, TensorType["b ..."]],
                ) -> Dict[str, TensorType["b ..."]]:
        outputs = {}
        # Deepcopy batch to avoid modifying original batch
        batch = copy.deepcopy(batch)
        diffusion_inputs = batch["diffusion_inputs"]

        # During training, keep track of certain additional features
        x = batch["diffusion_inputs"]["x"]
        B = x.shape[0]
        t_bb = batch.get("t_bb", None)  # scalar; fix t_bb if provided, usually for eval
        diffusion_inputs["t_bb"] = torch.full((B, ), t_bb, device=x.device) if t_bb is not None else None

        # Denoise coords
        _, aux_preds = self.denoiser(batch["motif_inputs"],
                                     diffusion_inputs=diffusion_inputs)

        # Additional outputs for computing loss
        outputs.update(aux_preds)


        return outputs


    def set_sigma_data(self, sigma_data: float):
        self.bb_std.data = torch.tensor(sigma_data)
        print(f"Setting backbone sigma_data: {sigma_data}")


    def sample(self,
               lengths: TensorType["b", int],
               residue_index: TensorType["b n", int],
               diffusion_params: Dict[str, Any],
               scaffold_inputs: Optional[Dict[str, torch.Tensor]] = None,
               cond_labels: Dict[str, TensorType["b", int]] = {},
               ) -> Tuple[TensorType["b n 4 3", float], Dict[str, torch.Tensor]]:
        """
        Sample from the model.

        Returns the final denoised coords and auxiliary outputs.
        """
        B, N = residue_index.shape

        aux = {}  # keep track of auxiliary outputs

        # Create seq mask
        ranges = torch.arange(N, device=residue_index.device).expand(B, N)
        seq_mask = (ranges < lengths[:, None]).float()
        aux["seq_mask"] = seq_mask.cpu()

        # Initialize motif inputs
        if scaffold_inputs is None:
            x_motif = torch.zeros((B, N, rc.atom_type_num, 3), device=residue_index.device)
            motif_mask = torch.zeros((B, N, rc.atom_type_num), device=residue_index.device)
            aatype_motif = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"])
        else:
            x_motif = scaffold_inputs["x_motif"]
            motif_mask = scaffold_inputs["motif_mask"]
            aatype_motif = scaffold_inputs["aatype_motif"]

        x1_bb, aux_preds = self.denoiser(x_motif=x_motif,
                                         motif_mask=motif_mask,
                                         aatype_motif=aatype_motif,
                                         residue_index=residue_index,
                                         seq_mask=seq_mask,
                                         cond_labels_in=cond_labels,
                                         diffusion_inputs=diffusion_inputs,
                                         is_sampling=True)

        aux.update(aux_preds["bb_diffusion_aux"])
        return x1_bb, aux


    @staticmethod
    def save_samples_to_pdb(samples: Dict[str, TensorType["b ..."]],
                            filenames: List[str]
                            ) -> None:
        """
        Save samples from the denoiser to PDB files.

        Samples should contain the following keys:
        - x_bb: Tensor["b n 4 3", float]`
        - seq_mask: Tensor["b n", float]
        - residue_index: Tensor["b n", int]
        """
        B, N, _, _= samples["x_bb"].shape
        residue_index = samples["residue_index"]
        seq_mask = samples["seq_mask"]

        x_bb = samples["x_bb"]

        aatype = torch.full_like(residue_index, fill_value=rc.restype_order["G"], dtype=torch.long)  # force aatype to glycine
        final_atom37_positions = torch.zeros((B, N, 37, 3), device=aatype.device)
        final_atom37_positions[:, :, rc.bb_idxs, :] = x_bb
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
                          x_traj_key: str,
                          filenames: List[str],
                          traj_conect: bool,
                          align_models_to_idx: Optional[int] = None):
        """
        Save trajectories from the denoiser to PDB files.

        Args:
        - traj_aux: auxiliary output from sampling trajectory
        - residue_index
        - chain_index
        - save_traj_mask: list of bools indicating which main trajectories to save
        - save_traj_steps: list of indices indicating which steps along the main trajectory to save
        - x_traj_key: key in traj_aux for the denoised atom positions
            - "x1_bb_traj" gives the x1 prediction along the backbone diffusion trajectory
            - "xt_bb_traj" gives the current state along the backbone diffusion trajectory
        - filenames: list of filenames to save the trajectories to
        - traj_conect: whether to include CONECT records in the PDB files
        - align_models_to_idx: if not None, align all models to the model at this index
        """
        B = traj_aux["seq_mask"].shape[0]
        device = traj_aux["seq_mask"].device
        for i in range(B):
            if save_traj_mask[i]:
                if x_traj_key in ["x1_bb_traj", "xt_bb_traj"]:
                    # Save x1 or xt traj
                    aatype_i = torch.full_like(traj_aux["seq_mask"][i], fill_value=rc.restype_order["G"], dtype=torch.long)  # force aatype to glycine
                    aatype_traj = aatype_i.unsqueeze(0).expand(len(save_traj_steps), -1)
                    atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)[aatype_traj] * traj_aux["seq_mask"][i, :, None]  # [S, N, A]
                    x_bb_traj = traj_aux[x_traj_key][i, save_traj_steps]
                else:
                    assert False, f"Unknown x_traj_key: {x_traj_key}"

                # Put backbone positions into atom37 format
                S, N, _, X = x_bb_traj.shape
                x_traj = torch.zeros((S, N, rc.atom_type_num, 3), device=device)
                x_traj[:, :, rc.bb_idxs, :] = x_bb_traj

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
                 ) -> nn.Module:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "dit":
        return DiTDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")
