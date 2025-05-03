import logging

import torch
import torch.nn as nn
from einops import repeat, rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from allatom_design.data import const


class ADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task

        # Parse loss_weights
        self.loss_weights = {}

        for k in cfg.loss_weights.keys():
            if isinstance(cfg.loss_weights[k], DictConfig):
                for k2 in cfg.loss_weights[k].keys():
                    self.loss_weights[f"{k}/{k2}"] = cfg.loss_weights[k][k2]
            else:
                self.loss_weights[k] = cfg.loss_weights[k]

        # Define losses based on task
        self.loss_keys = {"bb/mse_loss"}

        if self.task == "scaffold":
            self.loss_keys.add("motif/mse_loss")
            self.loss_keys.add("motif_proximity_bb/mse_loss")

        # Handle autoguidance loss and loss weights
        self.loss_keys = self.loss_keys.union({"autoguidance/bb/mse_loss"})
        self.loss_weights["autoguidance/bb/mse_loss"] = self.loss_weights["bb/mse_loss"]


    def forward(self, outputs, batch, return_aux: bool = False):
        """
        Compute losses for the atom denoiser.

        Expects outputs to contain a "bb_diffusion_aux" key with the following structure:
        - x1_pred: (m*b, n, a, 3) x1 prediction
        - x_target: (m*b, n, a, 3) target
        - loss_weight_t: (m*b) loss weight for each time step.

        m denotes batch size multiplier, b denotes batch size.
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss

        bb_diff_outputs = outputs["bb_diffusion_aux"]
        loss_weight_bb = bb_diff_outputs["loss_weight_t"]

        bb_pred = bb_diff_outputs["bb_pred"]
        bb_target = bb_diff_outputs["bb_target"]
        bb_mask = bb_diff_outputs["atom_mask"][..., const.prot_bb_atom14_idxs, None].expand_as(bb_pred)

        aux["bb/mse_loss"] = masked_mse(bb_pred,
                                        bb_target,
                                        weights=bb_mask)
        aux_monitor["bb/unweighted_mse_loss"] = aux["bb/mse_loss"].mean().detach().clone()
        aux["bb/mse_loss"] = aux["bb/mse_loss"] * loss_weight_bb  # apply time step loss weight

        # Compute motif losses
        if self.task == "scaffold":
            motif_inputs = bb_diff_outputs["motif_inputs_batched"]

            # Compute a loss only on motif tokens
            ## for now, assume diffusion token residue index is same as motif token residue index  # TODO: find a better way to track this that will generalize to multi-chain
            motif_residx = torch.where(motif_inputs["token_pad_mask"].bool(),   # set pad residx to -9999
                                       motif_inputs["residue_index"], -9999)
            diffusion_motif_mask = (bb_diff_outputs["diffusion_inputs_batched"]["residue_index"].unsqueeze(-1) == motif_residx.unsqueeze(1)).any(dim=-1)  # look for matching residx

            motif_bb_pred = bb_pred * diffusion_motif_mask[..., None, None]
            motif_bb_target = bb_target * diffusion_motif_mask[..., None, None]
            motif_bb_mask = bb_mask * diffusion_motif_mask[..., None, None]

            aux["motif/mse_loss"] = masked_mse(motif_bb_pred,
                                               motif_bb_target,
                                               weights=motif_bb_mask)

            aux_monitor["motif/unweighted_mse_loss"] = aux["motif/mse_loss"].mean().detach().clone()
            aux["motif/mse_loss"] = aux["motif/mse_loss"] * loss_weight_bb  # apply time step loss weight

            # Compute a distance-weighted backbone MSE loss (tokens upweighted by 1 / (1 + d^2) where d is the distance to closest motif coordinate)
            aux["motif_proximity_bb/mse_loss"] = motif_proximity_weighted_mse(bb_pred,
                                                                              bb_target,
                                                                              bb_mask,
                                                                              motif_inputs["motif_coords"],
                                                                              motif_inputs["motif_atom_mask"])
            # aux_monitor["motif_proximity_bb/unweighted_mse_loss"] = aux["motif_proximity_bb/mse_loss"].mean().detach().clone()  # no need to log unweighted loss
            aux["motif_proximity_bb/mse_loss"] = aux["motif_proximity_bb/mse_loss"] * loss_weight_bb  # apply time step loss weight

        # Compute loss for autoguidance model
        if bb_diff_outputs.get("autoguidance_aux") is not None:
            guidance_outputs = bb_diff_outputs["autoguidance_aux"]
            loss_weight_bb_ag = guidance_outputs["loss_weight_t"]

            bb_pred_ag = guidance_outputs["bb_pred"]
            bb_target_ag = guidance_outputs["bb_target"]
            bb_mask_ag = guidance_outputs["atom_mask"][..., const.prot_bb_atom14_idxs, None].expand_as(bb_pred_ag)

            aux["autoguidance/bb/mse_loss"] = masked_mse(bb_pred_ag,
                                                         bb_target_ag,
                                                         mask=bb_mask_ag)
            aux["autoguidance/bb/mse_loss"] = aux["autoguidance/bb/mse_loss"] * loss_weight_bb_ag  # apply time step loss weight


        # Aggregate losses
        total_loss = 0
        for loss_name, loss in aux.items():
            aux[loss_name] = loss.mean().detach().clone()

            # Average over batch
            loss = loss.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Loss {loss_name} is NaN or Inf, skipping...")
                loss = loss.new_tensor(0., requires_grad=True)

            if loss_name in self.loss_keys:
                # Only allow losses that are in the loss_keys to contribute to the total loss
                total_loss += loss * self.loss_weights[loss_name]  # apply manual per-loss loss weighting

        aux["total_loss"] = total_loss

        # Monitor other metrics
        aux.update(aux_monitor)

        if return_aux:
            return total_loss, aux
        return total_loss


def masked_mse(x: TensorType["b ...", float],
               y: TensorType["b ...", float],
               weights: TensorType["b ...", float]) -> TensorType["b", float]:
    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * weights
    mse = mse.sum(data_dims) / weights.sum(data_dims).clamp(min=1e-6)
    return mse


def motif_proximity_weighted_mse(bb_pred: TensorType["b n_tokens 4 3", float],
                                 bb_target: TensorType["b n_tokens 4 3", float],
                                 bb_mask: TensorType["b n_tokens 4 3", float],
                                 motif_coords: TensorType["b n_atoms 3", float],
                                 motif_atom_mask: TensorType["b n_atoms", bool]) -> TensorType["b", float]:
    """
    Compute a loss on backbone tokens with weights 1 / (1 + d^2) where d is the distance to closest motif coordinates.
    """
    # Compute distance of ground truth CA to motif coords
    bb_target_ca = bb_target[..., const.prot_bb_atoms.index("CA"), :]
    dists = torch.cdist(bb_target_ca, motif_coords)  # [b, n_tokens, n_atoms]

    # Compute weights for each token
    proximity_kernel = 1 / (1 + dists ** 2)
    proximity_kernel = proximity_kernel * motif_atom_mask.unsqueeze(1)

    # max over all atoms in the motif to get weight from closest atom
    token_weights = proximity_kernel.max(dim=-1).values  # [b, n_tokens]

    # handle token coordinate masking (both padding and missing atoms)
    token_weights = token_weights[..., None, None] * bb_mask  # [b n_tokens 4 3]
    return masked_mse(bb_pred,
                      bb_target,
                      weights=token_weights)
