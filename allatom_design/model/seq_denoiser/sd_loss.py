import logging
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torchtyping import TensorType

import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data import const

logger = logging.getLogger(__name__)


class SDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task
        self.use_seq_pred = self.task in ["seq_des", "lc_seq_des"]
        self.main_seq_loss_pocket_only = bool(cfg.get("main_seq_loss_pocket_only", False))
        self.main_potts_loss_pocket_only = bool(cfg.get("main_potts_loss_pocket_only", False))

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
            self.loss_keys = {"seq_loss", "potts_composite_loss", "potts_pseudolikelihood_loss"}
        elif self.task == "lc_seq_des":
            self.loss_keys = {"seq_loss", "potts_composite_loss", "potts_pseudolikelihood_loss"}
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        # Accumulator for non-standard AA token violations detected during
        # seq_loss computation. Kept as a non-persistent buffer so that it
        # moves with the module's device and is NOT saved to checkpoints
        # (preserves state_dict compatibility with existing checkpoints).
        self.register_buffer(
            "_nonstd_aa_violation_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )


    def forward(self, outputs, batch, eval_seq = True, eval_total = True, return_aux: bool = False):
        """
        Compute losses for the sequence denoiser
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss

        if self.use_seq_pred and eval_seq:
            # compute sequence loss from sequence design module
            target_restype = batch["restype"].argmax(dim=-1)
            seq_loss_mask = outputs["protein_residue_node_mask"] * (1 - outputs["seq_cond_mask"])  # compute loss only on masked tokens. protein_residue_node_mask is already for standard AA only.                        
            main_seq_loss_mask = seq_loss_mask
            pocket_mask = batch.get("token_is_ligand_pocket", None) if self.task == "lc_seq_des" else None

            if self.task == "lc_seq_des" and (self.main_seq_loss_pocket_only or self.main_potts_loss_pocket_only) and pocket_mask is None:
                logger.warning("token_is_ligand_pocket is missing; falling back to full seq/potts main loss mask")

            if self.main_seq_loss_pocket_only and self.task == "lc_seq_des" and pocket_mask is not None:
                main_seq_loss_mask = main_seq_loss_mask * pocket_mask

            # Accumulate non-standard AA violation count as a tensor so we
            # avoid the per-step GPU->CPU sync that ``.item()`` would force.
            # The accumulator is inspected (and zeroed) at epoch end by the
            # LightningModule.
            violation = (
                (~outputs["protein_residue_node_mask"].bool()) & main_seq_loss_mask.bool()
            ).sum()
            self._nonstd_aa_violation_count = (
                self._nonstd_aa_violation_count + violation.detach()
            )

            aux["seq_loss"] = masked_cross_entropy(outputs["seq_logits"], target_restype, main_seq_loss_mask,
                                                   seq_loss_cfg=self.cfg.seq_loss)
            
            if torch.isnan(outputs["seq_logits"]).any():
                logger.warning(f"seq_logits contains NaN!")
            
            aux_monitor["seq_acc"] = masked_seq_accuracy(outputs["seq_logits"], target_restype, main_seq_loss_mask).mean().detach().clone()
            
            if self.task == "lc_seq_des": 
                if pocket_mask is None:
                    ligand_pocket_seq_loss_mask = torch.zeros_like(seq_loss_mask)
                else:
                    ligand_pocket_seq_loss_mask = seq_loss_mask * pocket_mask
                                            
                # Select only samples that have pocket residues holding ligands
                has_close_ligands = ligand_pocket_seq_loss_mask.sum(dim=-1) > 0     
                                                            
                if has_close_ligands.any():
                    ligand_pocket_seq_loss = masked_cross_entropy(outputs["seq_logits"], target_restype, ligand_pocket_seq_loss_mask, seq_loss_cfg=self.cfg.seq_loss)
                    ligand_pocket_seq_loss = ligand_pocket_seq_loss[has_close_ligands]
                    aux_monitor["ligand_pocket_seq_loss"] = ligand_pocket_seq_loss.mean().detach().clone()
                         
                    ligand_pocket_seq_acc = masked_seq_accuracy(outputs["seq_logits"], target_restype, ligand_pocket_seq_loss_mask)
                    ligand_pocket_seq_acc = ligand_pocket_seq_acc[has_close_ligands]                
                    aux_monitor["ligand_pocket_seq_acc"] = ligand_pocket_seq_acc.mean().detach().clone()                

            else: 
                ligand_pocket_seq_loss_mask = None

            if outputs.get("potts_decoder_aux") is not None:
                potts_decoder_aux = outputs["potts_decoder_aux"]
                main_potts_seq_loss_mask = None
                if self.main_potts_loss_pocket_only and self.task == "lc_seq_des" and pocket_mask is not None:
                    main_potts_seq_loss_mask = ligand_pocket_seq_loss_mask

                # Composite and pseudolikelihood Potts losses are independently
                # toggleable via loss_weights. Gating mirrors the same ``weight > 0``
                # pattern for both so default configs (composite=1.0, pseudolikelihood=0.0)
                # remain bit-identical while a pseudolikelihood-only run skips the
                # composite path entirely.
                if self.loss_weights.get("potts_composite_loss", 0.0) > 0.0:
                    potts_loss, ligand_pocket_potts_loss = potts_composite_loss(S = target_restype,
                                                                       potts_decoder_aux = potts_decoder_aux,
                                                                       label_smoothing = self.cfg.potts.label_smoothing,
                                                                       per_token_avg = self.cfg.potts.per_token_avg,
                                                                       main_seq_loss_mask = main_potts_seq_loss_mask,
                                                                       compute_ligand_pocket_loss = self.task == "lc_seq_des",
                                                                       ligand_pocket_seq_loss_mask = ligand_pocket_seq_loss_mask,
                                                                       )
                    aux["potts_composite_loss"] = potts_loss
                    if ligand_pocket_potts_loss is not None:
                        aux_monitor["ligand_pocket_potts_composite_loss"] = ligand_pocket_potts_loss.mean().detach().clone()

                if self.loss_weights.get("potts_pseudolikelihood_loss", 0.0) > 0.0:
                    pl_loss, pl_ligand_pocket_loss = potts_pseudolikelihood_loss(
                        S=target_restype,
                        potts_decoder_aux=potts_decoder_aux,
                        label_smoothing=self.cfg.potts.label_smoothing,
                        per_token_avg=self.cfg.potts.per_token_avg,
                        main_seq_loss_mask=main_potts_seq_loss_mask,
                        compute_ligand_pocket_loss=self.task == "lc_seq_des",
                        ligand_pocket_seq_loss_mask=ligand_pocket_seq_loss_mask,
                    )
                    aux["potts_pseudolikelihood_loss"] = pl_loss
                    if pl_ligand_pocket_loss is not None:
                        aux_monitor["ligand_pocket_potts_pseudolikelihood_loss"] = pl_ligand_pocket_loss.mean().detach().clone()

        # Aggregate losses
        total_loss = 0.0        
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


def masked_cross_entropy(logits: TensorType["b n c", float],
                         target: TensorType["b n", int],
                         mask: TensorType["b n", float],
                         seq_loss_cfg: DictConfig,
                         ) -> TensorType["b", float]:
    """
    Compute cross entropy loss with masking.

    seq_loss_cfg has the following keys:
    - label_smoothing: float, label smoothing factor
    - per_token_avg: bool, whether to average loss per token (false will divide by fixed_size)
    """
    n_classes = const.AF3_ENCODING.n_tokens
    target_oh = F.one_hot(target, num_classes=n_classes).float()

    # Unpack seq_loss_cfg
    label_smoothing = seq_loss_cfg.label_smoothing
    per_token_avg = seq_loss_cfg.per_token_avg

    # Label smoothing
    target_oh = target_oh + label_smoothing / n_classes
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


def potts_composite_loss(S: TensorType["b n", int] = None,
                         potts_decoder_aux: dict[str, TensorType["b ...", float]] = None,
                         label_smoothing: float = 0.1,
                         per_token_avg: bool = False,
                         main_seq_loss_mask: TensorType["b n", float] = None,
                         compute_ligand_pocket_loss: bool = False,
                         ligand_pocket_seq_loss_mask: TensorType["b n", float] = None,
                         ) -> TensorType["b", float]:

    # Log composite likelihood
    logp_ij, mask_p_ij = potts.log_composite_likelihood(
        S,
        potts_decoder_aux["h"],
        potts_decoder_aux["J"],
        potts_decoder_aux["edge_idx"],
        potts_decoder_aux["mask_i"],
        potts_decoder_aux["mask_ij"],
        smoothing_alpha=label_smoothing,
    )

    # Map into approximate local likelihoods
    logp_i = (
        potts_decoder_aux["mask_i"]
        * torch.sum(mask_p_ij * logp_ij, dim=-1)
        / (2.0 * torch.sum(mask_p_ij, dim=-1) + 1e-3)
    )

    # Get loss per sample
    mask = potts_decoder_aux["mask_i"]
    if main_seq_loss_mask is not None:
        mask = mask * main_seq_loss_mask
    if per_token_avg:
        # average loss per token
        loss = (-logp_i * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    else:
        # divide by constant N to get loss on roughly the same scale as per_token_avg
        N = mask.shape[1]
        loss = -(logp_i * mask).sum(dim=-1) / N
    
    if not compute_ligand_pocket_loss:        
        return loss, None        
    else:
        if ligand_pocket_seq_loss_mask is None:
            return loss, None
        ligand_pocket_loss = (-logp_i * ligand_pocket_seq_loss_mask).sum(dim=-1) / ligand_pocket_seq_loss_mask.sum(dim=-1).clamp(min=1e-8)
        has_close_ligands = ligand_pocket_seq_loss_mask.sum(dim=-1) > 0         
        if has_close_ligands.any():
            ligand_pocket_loss = ligand_pocket_loss[has_close_ligands]
        else:
            ligand_pocket_loss = torch.tensor(0.0, device=loss.device)
        return loss, ligand_pocket_loss


def potts_pseudolikelihood_loss(S: TensorType["b n", int] = None,
                                potts_decoder_aux: dict[str, TensorType["b ...", float]] = None,
                                label_smoothing: float = 0.1,
                                per_token_avg: bool = False,
                                main_seq_loss_mask: TensorType["b n", float] = None,
                                compute_ligand_pocket_loss: bool = False,
                                ligand_pocket_seq_loss_mask: TensorType["b n", float] = None,
                                ) -> tuple[TensorType["b", float], TensorType["b", float] | None]:
    """Potts pseudolikelihood NLL loss (direct-regression form).

    Mirrors ``potts_composite_loss``: ``potts.log_pseudolikelihood`` returns a
    masked per-site log-prob at the true residue; here we just negate + average
    over unmasked positions, with no one-hot target arithmetic.
    """
    # ``log_pseudolikelihood`` masks logp_i internally and returns the node mask
    # unchanged; reuse it here to avoid a redundant ``potts_decoder_aux`` lookup.
    logp_i, mask = potts.log_pseudolikelihood(
        S,
        potts_decoder_aux["h"],
        potts_decoder_aux["J"],
        potts_decoder_aux["edge_idx"],
        potts_decoder_aux["mask_i"],
        smoothing_alpha=label_smoothing,
    )

    if main_seq_loss_mask is not None:
        mask = mask * main_seq_loss_mask

    if per_token_avg:
        loss = -(logp_i * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    else:
        N = mask.shape[1]
        loss = -(logp_i * mask).sum(dim=-1) / N

    if not compute_ligand_pocket_loss or ligand_pocket_seq_loss_mask is None:
        return loss, None

    ligand_pocket_loss = -(logp_i * ligand_pocket_seq_loss_mask).sum(dim=-1) \
                         / ligand_pocket_seq_loss_mask.sum(dim=-1).clamp(min=1e-8)
    has_close_ligands = ligand_pocket_seq_loss_mask.sum(dim=-1) > 0
    if has_close_ligands.any():
        ligand_pocket_loss = ligand_pocket_loss[has_close_ligands]
    else:
        ligand_pocket_loss = torch.tensor(0.0, device=loss.device)
    return loss, ligand_pocket_loss
