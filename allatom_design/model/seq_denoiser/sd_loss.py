import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc


class SDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task
        self.use_scn_diffusion_loss = self.task in ["scn_pack", "allatom_seq_des"]
        self.use_seq_pred = self.task in ["seq_des", "allatom_seq_des"]

        # Parse loss_weights
        self.loss_weights = {}

        for k in cfg.loss_weights.keys():
            if isinstance(cfg.loss_weights[k], DictConfig):
                for k2 in cfg.loss_weights[k].keys():
                    self.loss_weights[f"{k}/{k2}"] = cfg.loss_weights[k][k2]
            else:
                self.loss_weights[k] = cfg.loss_weights[k]

        # Define losses based on task
        if self.task == "seq_des":
            self.loss_keys = {"seq_loss"}
        elif self.task == "scn_pack":
            self.loss_keys = {"scn/mse_loss"}
        elif self.task == "allatom_seq_des":
            self.loss_keys = {"seq_loss", "scn/mse_loss"}
        else:
            raise ValueError(f"Unrecognized task: {self.task}")


    def forward(self, outputs, batch, eval_pack = True, eval_seq = True, eval_total = True, return_aux: bool = False):
        """
        Compute losses for the atom denoiser.
        If computing seq_loss, expects outputs to contain:
        - seq_logits: (b, n, k) sequence logits

        If computing scn/mse_loss, expects outputs to contain a "scn_diffusion_aux" key with the following structure:
        - x1_pred: (m*b, n, 33, 3) x1 prediction
        - x_target: (m*b, n, 33, 3) target

        m denotes batch size multiplier, b denotes batch size.
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss
        if self.use_seq_pred and eval_seq:
            # compute sequence loss from sequence design module
            seq_lengths = batch["seq_mask"].sum(-1).long()
            
            # compute sequence loss from sequence design module
            seq_loss_mask = batch['seq_mask'] * (1 - outputs["seq_mlm_mask"])
            aux["seq_loss"] = masked_cross_entropy(outputs["seq_logits"], batch["aatype"], seq_loss_mask,
                                                   seq_loss_cfg=self.cfg.seq_loss)
            aux_monitor["seq_acc"] = masked_seq_accuracy(outputs["seq_logits"], batch["aatype"], seq_loss_mask).mean().detach().clone()

        if self.use_scn_diffusion_loss and eval_pack:
            # We use sidechain diffusion auxiliary outputs to compute loss
            scn_diff_outputs = outputs["scn_diffusion_aux"]

            ## handle batch multiplier dimension
            scn_pred = scn_diff_outputs["scn_pred"]
            scn_target = scn_diff_outputs["scn_target"]
            M = scn_pred.shape[0] // batch["x_mask"].shape[0]  # diffusion batch multiplier
            mask = repeat(batch["x_mask"][..., rc.non_bb_idxs, :], "b n a x -> (m b) n a x", m=M)
            mask = mask * repeat(1 - outputs["scn_mlm_mask"], "b n -> (m b) n 1 1", m=M)  # only compute loss on masked sidechain positions

            ## loss weight based on EDM loss
            loss_weight_scn = scn_diff_outputs["loss_weight_t"]

            # Compute sidechain MSE loss
            aux["scn/mse_loss"] = masked_mse(scn_pred,
                                             scn_target,
                                             mask=mask)
            aux_monitor["scn/unweighted_mse_loss"] = aux["scn/mse_loss"].mean().detach().clone()
            aux["scn/mse_loss"] = aux["scn/mse_loss"] * loss_weight_scn  # apply time step loss weight


        # Aggregate losses
        total_loss = 0
        for loss_name, loss in aux.items():
            aux[loss_name] = loss.mean().detach().clone()

            # Average over batch
            loss = loss.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Loss {loss_name} is NaN or Inf, skipping...")
                loss = loss.new_tensor(0., requires_grad=True)

            if eval_total:
                if loss_name in self.loss_keys:
                    # Only allow losses that are in the loss_keys to contribute to the total loss
                    total_loss += loss * self.loss_weights[loss_name]  # apply manual per-loss loss weighting
            
        if eval_total:
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


def masked_mse(x: TensorType["b ...", float],
               y: TensorType["b ...", float],
               mask: TensorType["b ...", float]
               ) -> TensorType["b", float]:

    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * mask
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=1e-6)
    return mse


def masked_cross_entropy(logits: TensorType["b n k", float],
                         target: TensorType["b n", int],
                         mask: TensorType["b n", float],
                         seq_loss_cfg: DictConfig,
                         ) -> TensorType["b", float]:
    """
    Compute cross entropy loss with masking.

    seq_loss_cfg has the following keys:
    - label_smoothing: float, label smoothing factor
    - n_aatype: int, number of amino acid types
    - per_token_avg: bool, whether to average loss per token (false will divide by fixed_size)
    """
    target_oh = F.one_hot(target, num_classes=logits.shape[-1]).float()

    # Unpack seq_loss_cfg
    label_smoothing = seq_loss_cfg.label_smoothing
    n_aatype = seq_loss_cfg.n_aatype
    per_token_avg = seq_loss_cfg.per_token_avg

    # Label smoothing
    target_oh = target_oh + label_smoothing / n_aatype
    target_oh = target_oh / target_oh.sum(dim=-1, keepdim=True)

    # Compute cross entropy loss
    logprobs = F.log_softmax(logits, dim=-1)
    cel = -(logprobs * target_oh).sum(dim=-1)

    if per_token_avg:
        # average loss per token
        loss = (cel * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    else:
        # divide by constant N to get loss on roughly the same scale as per_token_avg
        N = mask.shape[1]
        loss = (cel * mask).sum(dim=-1) / N

    return loss


def masked_seq_accuracy(logits: TensorType["b n k", float],
                        target: TensorType["b n", int],
                        mask: TensorType["b n", float]
                        ) -> TensorType["b", float]:
    """
    Compute sequence accuracy with masking.
    """
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float()
    return (correct * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
