import itertools
import copy
from typing import Any, Dict

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from torchtyping import TensorType

from allatom_design.model.atom_denoiser.ad_loss import ADLoss
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.lr_schedule import InverseSqrtLR, NoamLR
from allatom_design.model.phema import PowerFunctionEMA


class LitAtomDenoiser(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AtomDenoiser(cfg.model)

        if cfg.train.compile_model:
            print(f"Using torch.compile to optimize model performance...")
            self.model = torch.compile(self.model)

        self.use_phema = cfg.model.ema.use_phema
        self.ema_decay = cfg.model.ema.ema_decay
        if self.use_phema:
            # Use EDM2 post-hoc EMA
            self.ema_tracker = PowerFunctionEMA(self.model)
        else:
            # Use vanilla EMA
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.requires_grad_(False)

        # Set up loss
        self.loss = ADLoss(cfg.loss)
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


    def training_step(self, batch: Dict[str, TensorType["b ..."]], batch_idx: int):
        outputs = self(batch)
        loss, aux = self.loss(outputs, batch, return_aux=True)

        # Logging
        self._log(batch, outputs, aux, batch_idx, phase="train")

        return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update whichever EMA we're using
        if self.use_phema:
            self.ema_tracker.update(t=self.trainer.global_step)
        else:
            self.update_ema()


    def validation_step(self, batch: Dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        if dataloader_idx == 0:
            phase_suffix = ""
        elif dataloader_idx == 1:
            phase_suffix = "2"

        # Use the appropriate model based on EMA
        if self.use_phema:
            # evaluate with current model
            outputs = self(batch)
        else:
            # evaluate with EMA model
            outputs = self.model_ema(batch)

        _, aux = self.loss(outputs, batch, return_aux=True)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix)

        B = batch["seq_mask"].shape[0]

        # Log metrics per timestep
        ts = list(self.cfg.eval.eval_timesteps_bb)

        for t in ts:
            batch["t_bb"] = t
            if self.use_phema:
                outputs = self(batch)
            else:
                outputs = self.model_ema(batch)

            _, aux = self.loss(outputs, batch, return_aux=True)
            aux = {k: v for k, v in aux.items() if "total" not in k}  # trim out total loss
            self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=f"_tbb{t}")


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


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        If not self.use_phema, store weight of EMA model in checkpoint.
        """
        if not self.use_phema:
            checkpoint["state_dict"] = self.model_ema.state_dict()


    def update_ema(self) -> None:
        """
        If not self.use_phema, performs vanilla EMA update based on self.ema_decay.
        """
        with torch.no_grad():
            for p_ema, p in zip(self.model_ema.parameters(), self.model.parameters()):
                p_ema.copy_(p_ema * self.ema_decay + p * (1.0 - self.ema_decay))
