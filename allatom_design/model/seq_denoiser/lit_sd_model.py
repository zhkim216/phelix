from collections import defaultdict
from typing import Dict

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from torchtyping import TensorType

from allatom_design.model.ema.phema import PowerFunctionEMA
from allatom_design.model.lr_schedule import InverseSqrtLR, NoamLR
from allatom_design.model.seq_denoiser.sd_loss import SDLoss
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


class LitSeqDenoiser(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = SeqDenoiser(cfg.model)

        if cfg.train.compile_model:
            print(f"Using torch.compile to optimize model performance...")
            self.model = torch.compile(self.model)

        self.use_phema = cfg.model.get("ema", {}).get("use_phema", True)
        if self.use_phema:
            # Use EDM2 post-hoc EMA
            self.ema_tracker = PowerFunctionEMA(self.model)

        # Set up loss
        self.loss = SDLoss(cfg.loss)
        self.save_hyperparameters()
        
        # For tracking lp metrics across epoch
        self.lp_metrics = {"loss": [], "acc": []}
        self.val_lp_metrics = {"loss": [], "acc": []}


    def setup(self, stage: str):
        if stage == "fit":
            # At start of training, load in pretrained modules if needed
            if self.cfg.resume.ckpt_path is None:
                self.model.setup()


    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)


    def on_train_start(self):
        # Initialize EMA trackers at the start of training (if using phema)
        if self.use_phema:
            self.ema_tracker.reset()
            
    def on_train_epoch_start(self):
        # Reset lp metrics at the start of each epoch
        self.lp_metrics = {"loss": [], "acc": []}
        
    def on_validation_epoch_start(self):
        # Reset validation lp metrics at the start of each epoch
        self.val_lp_metrics = {"loss": [], "acc": []}
        self.val_lp_metrics_t = {t: {"loss": [], "acc": []} for t in self.cfg.eval.eval_timesteps}
        self.val_lp_metrics_avg_t = {"loss": [], "acc": []}
        # Aggregators for distributed-safe validation logging
        self._val_metric_sums: dict[str, torch.Tensor] = {}
        self._val_metric_counts: dict[str, int] = {}
                
    def training_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int):
        outputs = self(batch)
        loss, aux = self.loss(outputs, batch, return_aux=True)

        # Logging
        self._log(batch, outputs, aux, batch_idx, phase="train")
        
        # Collect lp metrics if available (only for samples with ligands)
        if "lp_seq_loss" in aux:
            self.lp_metrics["loss"].append(aux["lp_seq_loss"])
            self.lp_metrics["acc"].append(aux["lp_seq_acc"])

        return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            if self.use_phema:
                # Update EMA tracker
                self.ema_tracker.update(t=self.trainer.global_step)


    def validation_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        phase_suffix = ""

        outputs = self(batch)
        _, aux = self.loss(outputs, batch, return_aux=True)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix)
        
        if "lp_seq_loss" in aux:
            self.val_lp_metrics["loss"].append(aux["lp_seq_loss"])
            self.val_lp_metrics["acc"].append(aux["lp_seq_acc"])

        # eval seq design over discrete sequence noise
        if self.model.task in ["seq_des", "lc_seq_des"]:
            aux_t = defaultdict(list)            
            lp_loss_t_vals, lp_acc_t_vals = [], []
            
            for eval_t in self.cfg.eval.eval_timesteps:
                B = batch["token_pad_mask"].shape[0]
                t_seq = torch.full((B, ), fill_value=eval_t).to(self.device)
                outputs = self(batch, t=t_seq)
                _, aux = self.loss(outputs, batch, eval_total = False, return_aux=True)
                aux = {k: v for k, v in aux.items() if ("seq" in k) or ("potts" in k)}  # trim aux to sequence metrics
                self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=f"_t{eval_t}")

                # Collect validation lp metrics for this timestep if available
                if "lp_seq_loss" in aux:
                    self.val_lp_metrics_t[eval_t]["loss"].append(aux["lp_seq_loss"])
                    self.val_lp_metrics_t[eval_t]["acc"].append(aux["lp_seq_acc"])
                    lp_loss_t_vals.append(aux["lp_seq_loss"])
                    lp_acc_t_vals.append(aux["lp_seq_acc"])

                # aggregate across timesteps
                for k, v in aux.items():
                    aux_t[k].append(v)

            if lp_loss_t_vals:
                self.val_lp_metrics_avg_t["loss"].append(torch.stack(lp_loss_t_vals).mean())
                self.val_lp_metrics_avg_t["acc"].append(torch.stack(lp_acc_t_vals).mean())

            # average across timesteps and log
            aux_t = {k: torch.stack(v).mean().item() for k, v in aux_t.items()}            
            self._log(batch, None, aux_t, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix="_avg_t")


    def _log(self,
             batch: dict[str, TensorType["b ..."]],
             outputs: dict[str, TensorType["b ..."]],
             aux: dict[str, float],
             batch_idx: int,
             phase: str,
             phase_suffix: str = "",
             key_suffix: str = ""):
        """
        phase_suffix: used to differentiate between different phases of validation (e.g. different fixed sizes), should include a leading "/"
        key_suffix: adds a suffix to the key
        """
        bs = len(batch["example_id"])

        log_dict = {}
        for k, v in aux.items():
            log_dict[f"{phase}{phase_suffix}/{k}{key_suffix}"] = v

        if phase == "train":
            trainer = getattr(self, "trainer", None)
            has_logger = bool(getattr(trainer, "logger", None))
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=has_logger, sync_dist=True,
                          add_dataloader_idx=False, batch_size=bs)
            return

        # Validation/Test phase: accumulate locally for distributed-safe epoch-end reduction
        for key, val in log_dict.items():
            val_tensor = torch.as_tensor(val, dtype=torch.float32, device=self.device)
            sum_contribution = (val_tensor * float(bs)).detach()
            if key in self._val_metric_sums:
                self._val_metric_sums[key] = self._val_metric_sums[key] + sum_contribution
                self._val_metric_counts[key] = self._val_metric_counts[key] + bs
            else:
                self._val_metric_sums[key] = sum_contribution
                self._val_metric_counts[key] = bs


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
        trainer = getattr(self, "trainer", None)
        has_logger = bool(getattr(trainer, "logger", None))
        for norm_type in [1, 2]:
            grad_norms = grad_norm(self.model, norm_type=norm_type)

            total_norm_key = f"grad_{float(norm_type)}_norm_total"
            if total_norm_key in grad_norms:
                total_norm = grad_norms[total_norm_key]
                self.log_dict({f"total_l{norm_type}_grad_norm": total_norm}, logger=has_logger)
    
    def on_train_epoch_end(self):
        # Log average lp metrics for the epoch (only from batches that had ligands)
        if self.lp_metrics["loss"]:
            avg_lp_loss = torch.stack(self.lp_metrics["loss"]).mean()
            avg_lp_acc = torch.stack(self.lp_metrics["acc"]).mean()
            
            self.log("train/lp_seq_loss_epoch", avg_lp_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train/lp_seq_acc_epoch", avg_lp_acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
    def on_validation_epoch_end(self):
        # Distributed-safe reduction of validation metrics accumulated in _log
        local_keys = list(self._val_metric_sums.keys()) if hasattr(self, "_val_metric_sums") else []

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_key_lists: list[list[str]] = [None for _ in range(world_size)]  # type: ignore
            dist.all_gather_object(gathered_key_lists, local_keys)

            union_keys = sorted({k for kl in gathered_key_lists for k in kl})

            for key in union_keys:
                local_sum = self._val_metric_sums.get(key, torch.tensor(0.0, device=self.device, dtype=torch.float32))
                local_cnt = torch.tensor(float(self._val_metric_counts.get(key, 0)), device=self.device, dtype=torch.float32)

                sum_tensor = local_sum.clone()
                cnt_tensor = local_cnt.clone()
                dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)

                if self.trainer.is_global_zero and cnt_tensor.item() > 0:
                    avg = (sum_tensor / cnt_tensor).item()
                    self.log(key, avg, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        else:
            # Single-process fallback
            for key, sum_tensor in self._val_metric_sums.items():
                cnt = float(self._val_metric_counts.get(key, 0))
                if cnt > 0:
                    avg = (sum_tensor / cnt).item()
                    self.log(key, avg, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)