import itertools
from collections import defaultdict
from typing import Any, Dict

import lightning as L
import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torchtyping import TensorType

from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.model.phema import PowerFunctionEMA
from allatom_design.model.seq_denoiser.rl_sd_loss import RLSDLoss
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.eval.folding_utils import run_esmfold_batched


class LitRLSeqDenoiser(L.LightningModule):
    def __init__(self, cfg: DictConfig, base_cfg: DictConfig):
        """
        base_cfg: The config for the base pretrained model
        """
        super().__init__()
        self.cfg = cfg
        self.base_cfg = base_cfg
        self.struct_pred_cfg = cfg.struct_pred_cfg
        self.model = SeqDenoiser(base_cfg.model)

        if cfg.train.compile_model:
            print(f"Using torch.compile to optimize model performance...")
            self.model = torch.compile(self.model)

        # Initialize EMA tracker after torch.compile
        self.ema_tracker = PowerFunctionEMA(self.model)

        # Set up loss
        self.loss = RLSDLoss(cfg.loss)
        self.save_hyperparameters()


    def load_base_model(self, base_model: SeqDenoiser):
        print("Loading base model...")
        if not hasattr(base_model, "_orig_mod") and self.cfg.train.compile_model:
            # Model is not compiled, so compile it
            base_model = torch.compile(base_model)
        if hasattr(base_model, "_orig_mod") and not self.cfg.train.compile_model:
            # Model is compiled, so uncompile it
            base_model = base_model._orig_mod

        self.model = base_model


    def setup(self, stage: str):
        if stage == "fit":
            # At start of training, load in ESMFold
            self.esmfold = get_struct_pred_model(self.struct_pred_cfg, device=self.device)


    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)

    def on_train_start(self):
        # Initialize EMA trackers at the start of training
        self.ema_tracker.reset()


    def run_policy(self, batch):
        pass


    def get_reward(self, ):
        esm_preds = run_esmfold_batched(sequences_list=sequences_list,
                                        residue_index_list=residue_index_list,
                                        chain_index_list=chain_index_list,
                                        model=self.esmfold["esmfold"],
                                        tokenizer=self.esmfold["tokenizer"],
                                        max_tokens_per_batch=self.struct_pred_cfg.esmfold.max_tokens_per_batch,
                                        )
        # stack all outputs since they are the same length for a given PDB
        esm_preds = {k: torch.stack(v, dim=0) for k, v in esm_preds.items()}

        # Write to pdb file
        feats = {
            "aatype": esm_preds["aatype"],
            "atom_positions": esm_preds["pred_coords"],
            "atom_mask": esm_preds["atom_mask"],
            "residue_index": esm_preds["residue_index"],
            "chain_index": torch.zeros_like(esm_preds["residue_index"]),
            "b_factors": None,
        }


        # Compute structure metrics
        B, _, _, _ = struct_preds["pred_coords"].shape
        metrics, pred_coords_ca_aligned = eval_metrics.compute_structure_metrics(
            struct_preds["pred_coords"],
            sampled_pdb_feats["all_atom_positions"][None].expand(B, -1, -1, -1),
            sampled_pdb_feats["all_atom_mask"][None].expand(B, -1, -1),
            metrics_to_compute=["sc_ca_rmsd", "sc_aa_rmsd"],
            temp_dir=temp_dir,
        )


    def training_step(self, batch: Dict[str, TensorType["b ..."]], batch_idx: int):
        outputs = self(batch)
        loss, aux = self.loss(outputs, batch, return_aux=True)

        # Logging
        self._log(batch, outputs, aux, batch_idx, phase="train")

        return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Update EMA tracker
            self.ema_tracker.update(t=self.trainer.global_step)


    def validation_step(self, batch: Dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        phase_suffix = ""

        outputs = self(batch)
        _, aux = self.loss(outputs, batch, return_aux=True)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix)

        # eval seq design over discrete sequence noise
        aux_t = defaultdict(list)
        for eval_t in self.cfg.eval.eval_timesteps:
            B = batch["seq_mask"].shape[0]
            t_seq = torch.full((B, ), fill_value=eval_t).to(self.device)
            outputs = self(batch, t=t_seq)
            _, aux = self.loss(outputs, batch, eval_total = False, return_aux=True)
            aux = {k: v for k, v in aux.items() if "seq" in k}  # trim aux to sequence metrics
            self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=f"_t{eval_t}")

            # aggregate across timesteps
            for k, v in aux.items():
                aux_t[k].append(v)

        # average across timesteps and log
        aux_t = {k: torch.stack(v).mean().item() for k, v in aux_t.items()}
        self._log(batch, None, aux_t, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix="_avg_t")

        # eval sidechain packing over edm sidechain noise, fully unmasked sequence
        B = batch["seq_mask"].shape[0]
        t_seq = torch.full((B, ), fill_value=0).to(self.device)

        for t_scd in self.cfg.eval.eval_timesteps:
            batch["t_scd"] = t_scd
            outputs = self(batch, t=t_seq)
            _, aux = self.loss(outputs, batch, eval_seq = False, eval_total = False, return_aux=True)
            aux = {k: v for k, v in aux.items() if "scn/" in k}  # trim aux to sidechain diffusion metrics
            aux = {k: v for k, v in aux.items() if "unweighted" not in k}  # trim out unweighted loss
            self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix="/scn_diff",
                        key_suffix=f"_t_scd{t_scd}")


    def _log(self,
             batch: Dict[str, TensorType["b ..."]],
             outputs: Dict[str, TensorType["b ..."]],
             aux: Dict[str, float],
             batch_idx: int,
             phase: str,
             phase_suffix: str = "",
             key_suffix: str = ""):
        """
        phase_suffix: used to differentiate between different phases of validation (e.g. different fixed sizes), should include a leading "/"
        key_suffix: adds a suffix to the key
        """
        bs = len(batch["pdb_key"])

        log_dict = {}
        for k, v in aux.items():
            log_dict[f"{phase}{phase_suffix}/{k}{key_suffix}"] = v

        self.log_dict(log_dict, on_step=(phase == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                      add_dataloader_idx=False, batch_size=bs)


    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        if optim_cfg.optimizer == "adamw":
            optimizer = AdamW(list(self.model.parameters()) + list(self.loss.parameters()),
                            lr=optim_cfg.adamw.lr, eps=1.0e-15)
            scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1, total_iters=optim_cfg.adamw.warmup_steps)
        elif optim_cfg.optimizer == "noam":
            optimizer = Adam(list(self.model.parameters()) + list(self.loss.parameters()),
                             lr=0, betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamLR(optimizer,
                               model_size=128,  # hardcoded to MPNN hidden dim
                               factor=optim_cfg.noam.factor,
                               warmup=optim_cfg.noam.warmup_steps)
        elif optim_cfg.optimizer == "adam_inv_sqrt":
            optimizer = Adam(list(self.model.parameters()) + list(self.loss.parameters()),
                             lr=0, betas=(0.9, 0.99), eps=1e-9)
            scheduler = InverseSqrtLR(optimizer,
                                      ref_lr=optim_cfg.adam_inv_sqrt.ref_lr,
                                      ref_steps=optim_cfg.adam_inv_sqrt.ref_steps,
                                      warmup_steps=optim_cfg.adam_inv_sqrt.warmup_steps)
        else:
            raise ValueError(f"Unknown optimizer: {optim_cfg.optimizer}")

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                    }
                }


    def on_before_optimizer_step(self, optimizer):
        # Compute the 1-norm and 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        for norm_type in [1, 2]:
            grad_norms = grad_norm(self.model, norm_type=norm_type)

            total_norm_key = f"grad_{float(norm_type)}_norm_total"
            if total_norm_key in grad_norms:
                total_norm = grad_norms[total_norm_key]
                self.log_dict({f"total_l{norm_type}_grad_norm": total_norm})


class NoamLR(LRScheduler):
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        rate = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return [rate for _ in self.base_lrs]


class InverseSqrtLR(LRScheduler):
    def __init__(self, optimizer, ref_lr: float, ref_steps: int, warmup_steps: int, last_epoch=-1):
        self.ref_lr = ref_lr
        self.ref_steps = ref_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        lr = self.ref_lr
        if self.ref_steps > 0:
            lr /= np.sqrt(max(step / self.ref_steps, 1))
        if self.warmup_steps > 0:
            lr *= min(step / self.warmup_steps, 1)

        return [lr for _ in self.base_lrs]
