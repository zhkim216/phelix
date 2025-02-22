import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torchtyping import TensorType

from allatom_design.data import residue_constants as rc
from allatom_design.model.seq_denoiser.sd_loss import (masked_cross_entropy,
                                                       masked_mse,
                                                       masked_seq_accuracy,
                                                       psce_loss)


class RLSDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Parse loss_weights
        self.loss_weights = {}

        def parse_weights(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    parse_weights(v, f"{prefix}{k}/" if prefix else f"{k}/")
                else:
                    self.loss_weights[f"{prefix}{k}"] = v

        loss_weights = OmegaConf.to_container(cfg.loss_weights, resolve=True)
        parse_weights(loss_weights)

        self.loss_keys = {"winner/seq_loss", "winner/scn/mse_loss", "winner/psce_loss",
                          "loser/seq_loss", "loser/scn/mse_loss", "loser/psce_loss",
                          "dpo/seq_loss"}


    def forward(self, outputs, outputs_ref, batch, eval_pack = True, eval_seq = True, eval_total = True, return_aux: bool = False):
        """
        Compute DPO losses based on current model and reference model.
        If computing seq_loss, expects outputs to contain:
        - seq_logits: (b, n, k) sequence logits

        If computing scn/mse_loss, expects outputs to contain a "scn_diffusion_aux" key with the following structure:
        - x1_pred: (m*b, n, 33, 3) x1 prediction
        - x_target: (m*b, n, 33, 3) target

        m denotes batch size multiplier, b denotes batch size.
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss

        # Compute loss separately for winners and losers
        for mode in ["winner", "loser"]:
            batch_i = {k: v[::2] if mode == "winner" else v[1::2] for k, v in batch.items() if k not in ["pdb_key", "cond_labels_in", "t_scd"]}
            outputs_i = {k: v[::2] if mode == "winner" else v[1::2] for k, v in outputs.items() if k not in ["scn_diffusion_aux"]}

            if eval_seq:
                # compute sequence loss from sequence design module
                # compute sequence loss on masked tokens
                seq_loss_mask = batch_i['seq_mask'] * (1 - outputs_i["seq_mlm_mask"])

                #mask unk tokens from loss calculation
                seq_loss_mask = seq_loss_mask * (1 - batch_i['seq_unk_mask'])

                aux[f"{mode}/seq_loss"] = masked_cross_entropy(outputs_i["seq_logits"], batch_i["aatype"], seq_loss_mask,
                                                    seq_loss_cfg=self.cfg.seq_loss)
                aux_monitor[f"{mode}/seq_acc"] = masked_seq_accuracy(outputs_i["seq_logits"], batch_i["aatype"], seq_loss_mask).mean().detach().clone()

            if eval_pack:
                # We use sidechain diffusion auxiliary outputs to compute loss
                scn_diff_outputs = outputs["scn_diffusion_aux"]
                scn_diff_outputs_i = {k: v[::2] if mode == "winner" else v[1::2] for k, v in scn_diff_outputs.items() if k not in ["confidence_aux"]}

                ## handle batch multiplier dimension
                scn_pred = scn_diff_outputs_i["scn_pred"]
                scn_target = scn_diff_outputs_i["scn_target"]
                M = scn_pred.shape[0] // batch_i["x_mask"].shape[0]  # diffusion batch multiplier
                mask = repeat(batch_i["x_mask"][..., rc.non_bb_idxs, :], "b n a x -> (m b) n a x", m=M)

                ## mask out loss wherever the backbone frame doesn't exist
                mask = mask * scn_diff_outputs_i["bb_frames_exists"][..., None, None]

                ## loss weight based on EDM loss
                loss_weight_scn = scn_diff_outputs_i["loss_weight_t"]

                # Compute sidechain MSE loss
                aux[f"{mode}/scn/mse_loss"] = masked_mse(scn_pred,
                                                scn_target,
                                                mask=mask,
                                                per_token_avg=self.cfg.mse_loss.per_token_avg,)
                aux_monitor[f"{mode}/scn/unweighted_mse_loss"] = aux[f"{mode}/scn/mse_loss"].mean().detach().clone()
                aux[f"{mode}/scn/mse_loss"] = aux[f"{mode}/scn/mse_loss"] * loss_weight_scn  # apply time step loss weight

                # Compute loss for confidence model
                if scn_diff_outputs.get("confidence_aux") is not None:
                    confidence_outputs = scn_diff_outputs["confidence_aux"]
                    confidence_outputs_i = {k: v[::2] if mode == "winner" else v[1::2] for k, v in confidence_outputs.items() if k not in ["sce_bins_cfg"]}

                    psce_logits = confidence_outputs_i["psce_logits"]
                    scn_pred_rollout = confidence_outputs_i["scn_pred_rollout"]
                    scn_target = confidence_outputs_i["scn_target"]
                    scn_atom_mask = batch_i["atom_mask"][..., rc.non_bb_idxs]

                    # Construct residue-level mask
                    new_scn_mask = (1 - outputs_i["scn_mlm_mask"])  # only compute confidence loss over masked sidechains
                    new_scn_mask = new_scn_mask * (1 - batch_i["seq_unk_mask"])  # mask out true unk tokens
                    new_scn_mask = new_scn_mask * batch_i["seq_mask"]  # mask out padding
                    new_scn_mask = new_scn_mask * confidence_outputs_i["bb_frames_exists"]  # mask out residues with missing backbone frames

                    # Compute PSCE confidence loss
                    psce_mask = rearrange(new_scn_mask, "b n -> b n 1") * scn_atom_mask  # mask out ghost and missing sidechain atoms
                    aux[f"{mode}/psce_loss"] = psce_loss(psce_logits, scn_pred_rollout, scn_target, psce_mask,
                                                self.cfg.inf,
                                                **confidence_outputs["sce_bins_cfg"])

                    # monitor rollout sidechain RMSD (averaged across residues with masked sidechains)
                    msd_per_res = (psce_mask[..., None] * (scn_target - scn_pred_rollout)).pow(2).sum(dim=(-1, -2)) / psce_mask.sum(dim=-1).clamp(min=1)
                    rmsd_per_res = msd_per_res.sqrt()
                    rmsd = (rmsd_per_res * new_scn_mask).sum(dim=-1) / new_scn_mask.sum(dim=-1).clamp(min=1)
                    aux_monitor[f"{mode}/rollout/scn_rmsd"] = rmsd.mean().detach().clone()

                    # monitor per-atom sce vs psce correlation
                    psce = confidence_outputs_i["psce"]
                    sce = torch.norm(scn_pred_rollout - scn_target, dim=-1)
                    rho = spearmanr(psce[psce_mask.bool()].detach().cpu(), sce[psce_mask.bool()].detach().cpu())[0]
                    pearson_r = pearsonr(psce[psce_mask.bool()].detach().cpu(), sce[psce_mask.bool()].detach().cpu())[0]
                    aux_monitor[f"{mode}/rollout/sce_vs_psce_rho"] = rho
                    aux_monitor[f"{mode}/rollout/sce_vs_psce_pearson_r"] = pearson_r

        # Compute DPO losses
        dpo_seq_loss = compute_dpo_loss(outputs["seq_logits"], outputs_ref["seq_logits"], batch["aatype"], batch["seq_mask"], self.cfg.dpo)
        aux["dpo/seq_loss"] = dpo_seq_loss

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


def compute_dpo_loss(logits_policy: TensorType["b n k", float],
                     logits_ref: TensorType["b n k", float],
                     target: TensorType["b n", int],
                     seq_mask: TensorType["b n", float],
                     dpo_loss_cfg: DictConfig,
                     ) -> TensorType["b", float]:
    target_oh = F.one_hot(target, num_classes=logits_policy.shape[-1]).float()

    # Compute log-likelihood for current policy and reference
    logprobs_policy = F.log_softmax(logits_policy, dim=-1)
    ll = (logprobs_policy * target_oh).sum(-1)
    ll = (ll * seq_mask).sum(-1)  # sum over sequence length

    logprobs_ref = F.log_softmax(logits_ref, dim=-1)
    ll_ref = (logprobs_ref * target_oh).sum(-1)
    ll_ref = (ll_ref * seq_mask).sum(-1)  # sum over sequence length

    # Separate log-likelihoods by winner and loser of each pair
    ll_w, ll_l = ll[::2], ll[1::2]
    ll_ref_w, ll_ref_l = ll_ref[::2], ll_ref[1::2]

    # Compute DPO loss
    beta = dpo_loss_cfg.beta
    dpo_loss = -torch.log(nn.functional.sigmoid(beta * (ll_w - ll_ref_w - ll_l + ll_ref_l)))
    return dpo_loss
