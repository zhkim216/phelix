import logging

import torch
import torch.nn as nn
from einops import repeat
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc


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
            self.loss_keys.add("scaffold/mse_loss")

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
        M = bb_pred.shape[0] // batch["x_mask"].shape[0]  # diffusion batch multiplier
        bb_mask = repeat(batch["x_mask"][..., rc.bb_idxs, :], "b n a x -> (m b) n a x", m=M)

        aux["bb/mse_loss"] = masked_mse(bb_pred,
                                        bb_target,
                                        mask=bb_mask)
        aux_monitor["bb/unweighted_mse_loss"] = aux["bb/mse_loss"].mean().detach().clone()
        aux["bb/mse_loss"] = aux["bb/mse_loss"] * loss_weight_bb  # apply time step loss weight

        # Compute loss for autoguidance model
        if bb_diff_outputs.get("autoguidance_aux") is not None:
            guidance_outputs = bb_diff_outputs["autoguidance_aux"]
            loss_weight_bb_ag = guidance_outputs["loss_weight_t"]

            bb_pred_ag = guidance_outputs["bb_pred"]
            bb_target_ag = guidance_outputs["bb_target"]
            M = bb_pred_ag.shape[0] // batch["x_mask"].shape[0]  # diffusion batch multiplier
            bb_mask_ag = repeat(batch["x_mask"][..., rc.bb_idxs, :], "b n a x -> (m b) n a x", m=M)

            aux["autoguidance/bb/mse_loss"] = masked_mse(bb_pred_ag,
                                                         bb_target_ag,
                                                         mask=bb_mask_ag)
            aux["autoguidance/bb/mse_loss"] = aux["autoguidance/bb/mse_loss"] * loss_weight_bb_ag  # apply time step loss weight

        # Compute scaffold loss
        if self.task == "scaffold":
            scaffold_aux = outputs["scaffold_aux"]
            bb_scaffold_mask = scaffold_aux["scaffold_mask"][..., rc.bb_idxs]
            bb_scaffold_mask = repeat(bb_scaffold_mask, "b n a -> (m b) n a", m=M)
            bb_mask = bb_mask * bb_scaffold_mask[..., None]

            aux["scaffold/mse_loss"] = masked_mse(bb_pred,
                                                  bb_target,
                                                  mask=bb_mask)
            test = masked_mse(bb_pred,
                              scaffold_aux["x_scaffold"][..., rc.bb_idxs, :],
                              mask=bb_mask
                              )
            aux_monitor["scaffold/unweighted_mse_loss"] = aux["scaffold/mse_loss"].mean().detach().clone()
            aux["scaffold/mse_loss"] = aux["scaffold/mse_loss"] * loss_weight_bb  # apply time step loss weight

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
               mask: TensorType["b ...", float]
               ) -> TensorType["b", float]:

    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * mask
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=1e-6)
    return mse
