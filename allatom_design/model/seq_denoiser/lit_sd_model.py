from collections import defaultdict
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from torchtyping import TensorType

import logging

from allatom_design.model.ema.phema import PowerFunctionEMA
from allatom_design.model.lr_schedule import InverseSqrtLR, NoamLR
from allatom_design.model.seq_denoiser.sd_loss import SDLoss
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser

logger = logging.getLogger(__name__)


class LitSeqDenoiser(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = SeqDenoiser(cfg.model)

        if cfg.train.compile.compile_model:
            print(f"Using torch.compile to optimize model performance...")            
            self.model = torch.compile(self.model,
                                        backend=cfg.train.compile.compile_backend,
                                        mode=cfg.train.compile.mode,
                                        fullgraph=cfg.train.compile.fullgraph,
                                        dynamic=cfg.train.compile.dynamic)

        self.use_phema = cfg.model.get("ema", {}).get("use_phema", True)
        if self.use_phema:
            # Use EDM2 post-hoc EMA
            self.ema_tracker = PowerFunctionEMA(self.model)

        # Set up loss
        self.loss = SDLoss(cfg.loss)
        self.save_hyperparameters()
        

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
                
    @staticmethod
    def _pop_non_tensor_fields(batch: dict) -> dict:
        """Remove non-tensor entries from ``batch`` and return them.

        torch.compile's dynamo guards on Python-level structures (e.g. the
        contents of ``batch["example_id"]`` string lists) and will recompile
        on every step if those values change, eventually hitting the
        recompile limit and falling back to eager. Stripping them before
        the forward avoids that.
        """
        meta_fields = {k: batch[k] for k in list(batch) if not isinstance(batch[k], torch.Tensor)}
        for k in meta_fields:
            del batch[k]
        return meta_fields

    def training_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int):
        meta_fields = self._pop_non_tensor_fields(batch)
        outputs = self(batch)
        batch.update(meta_fields)

        loss, aux = self.loss(outputs, batch, return_aux=True)

        # Logging
        self._log(batch, outputs, aux, batch_idx, phase="train")

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            if self.use_phema:
                # Update EMA tracker
                self.ema_tracker.update(t=self.trainer.global_step)

    def on_train_epoch_end(self):
        # Report non-standard AA token violations accumulated by SDLoss
        # over the epoch (single device->host sync, once per epoch).
        count = int(self._nonstd_aa_violation_tensor().item())
        if count > 0:
            logger.warning(
                f"seq_loss was computed over {count} non-standard AA tokens this epoch"
            )
        self._nonstd_aa_violation_tensor().zero_()

    def _nonstd_aa_violation_tensor(self) -> torch.Tensor:
        """Return the SDLoss violation accumulator, regardless of compile wrapping."""
        return self.loss._nonstd_aa_violation_count


    def validation_step(self, batch: dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        phase_suffix = ""

        # Strip non-tensor fields once for the whole step; self._log needs
        # them back (e.g. batch["example_id"] for batch size), so restore
        # before any logging call.
        meta_fields = self._pop_non_tensor_fields(batch)

        outputs = self(batch)
        _, aux = self.loss(outputs, batch, return_aux=True)
        batch.update(meta_fields)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix)

        # eval seq design over discrete sequence noise

        aux_t = defaultdict(list)

        for eval_t in self.cfg.eval.eval_timesteps:
            B = batch["token_pad_mask"].shape[0]
            t_seq = torch.full((B, ), fill_value=eval_t).to(self.device)

            meta_fields = self._pop_non_tensor_fields(batch)
            outputs = self(batch, t=t_seq)
            _, aux = self.loss(outputs, batch, eval_total = False, return_aux=True)
            batch.update(meta_fields)

            aux = {k: v for k, v in aux.items() if ("seq" in k) or ("potts" in k)}  # trim aux to sequence metrics
            self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=f"_t{eval_t}")

            # aggregate across timesteps
            for k, v in aux.items():
                aux_t[k].append(v)

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
                
        self.log_dict(
            log_dict,
            on_step=(phase == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=bs,
        )


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
    
