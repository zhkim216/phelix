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
        self.loss_keys = {"bb_ca/mse_loss", "bb_nco/mse_loss"}


    def forward(self, outputs, batch, return_aux: bool = False):
        """
        Compute losses for the atom denoiser.

        Expects outputs to contain a "bb_diffusion_aux" key with the following structure:
        - x1_pred: (m*b, n, a, 3) x1 prediction
        - x_target: (m*b, n, a, 3) target
        - loss_weight_t: (m*b) or Tuple[(m*b), (m*b)] loss weight for each time step. If tuple: (loss_weight_ca, loss_weight_nco)

        m denotes batch size multiplier, b denotes batch size.
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss

        # Compute loss for CA and NCO separately
        bb_diff_outputs = outputs["bb_diffusion_aux"]
        loss_weight_ca, loss_weight_nco = bb_diff_outputs["loss_weight_t"]

        bb_pred = bb_diff_outputs["bb_pred"]
        bb_target = bb_diff_outputs["bb_target"]
        M = bb_pred.shape[0] // batch["x_mask"].shape[0]  # diffusion batch multiplier
        bb_mask = repeat(batch["x_mask"][..., rc.bb_idxs, :], "b n a x -> (m b) n a x", m=M)

        # CA
        aux["bb_ca/mse_loss"] = masked_mse(bb_pred[..., 1:2, :],
                                           bb_target[..., 1:2, :],
                                           mask=bb_mask[..., 1:2, :])
        aux_monitor["bb_ca/unweighted_mse_loss"] = aux["bb_ca/mse_loss"].mean().detach().clone()
        aux["bb_ca/mse_loss"] = aux["bb_ca/mse_loss"] * loss_weight_ca  # apply time step loss weight

        # NCO
        aux["bb_nco/mse_loss"] = masked_mse(bb_pred[..., rc.nco_idxs, :],
                                            bb_target[..., rc.nco_idxs, :],
                                            mask=bb_mask[..., rc.nco_idxs, :])
        aux_monitor["bb_nco/unweighted_mse_loss"] = aux["bb_nco/mse_loss"].mean().detach().clone()
        aux["bb_nco/mse_loss"] = aux["bb_nco/mse_loss"] * loss_weight_nco  # apply time step loss weight

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
