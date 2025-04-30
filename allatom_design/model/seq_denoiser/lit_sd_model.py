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
from torch.optim.lr_scheduler import LinearLR
from torchtyping import TensorType

from allatom_design.model.lr_schedule import InverseSqrtLR, NoamLR
from allatom_design.model.ema.phema import PowerFunctionEMA
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
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            if self.use_phema:
                # Update EMA tracker
                self.ema_tracker.update(t=self.trainer.global_step)


    def validation_step(self, batch: Dict[str, TensorType["b ..."]], batch_idx: int, dataloader_idx: int = 0):
        # Lightning automatically disables grads + sets model to eval mode
        phase_suffix = ""

        outputs = self(batch)
        _, aux = self.loss(outputs, batch, return_aux=True)
        self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix)

        # eval seq design over discrete sequence noise
        if self.model.task in ["seq_des"]:
            aux_t = defaultdict(list)
            for eval_t in self.cfg.eval.eval_timesteps:
                B = batch["token_pad_mask"].shape[0]
                t_seq = torch.full((B, ), fill_value=eval_t).to(self.device)
                outputs = self(batch, t=t_seq)
                _, aux = self.loss(outputs, batch, eval_total = False, return_aux=True)
                aux = {k: v for k, v in aux.items() if ("seq" in k) or ("potts" in k)}  # trim aux to sequence metrics
                self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix=f"_t{eval_t}")

                # aggregate across timesteps
                for k, v in aux.items():
                    aux_t[k].append(v)

            # average across timesteps and log
            aux_t = {k: torch.stack(v).mean().item() for k, v in aux_t.items()}
            self._log(batch, None, aux_t, batch_idx, phase="val", phase_suffix=phase_suffix, key_suffix="_avg_t")

        # # eval sidechain packing over edm sidechain noise, fully unmasked sequence
        # if self.model.task in ["allatom_seq_des", "scn_pack"]:
        #     B = batch["seq_mask"].shape[0]
        #     t_seq = torch.full((B, ), fill_value=0).to(self.device)

        #     for t_scd in self.cfg.eval.eval_timesteps:
        #         batch["t_scd"] = t_scd
        #         outputs = self(batch, t=t_seq)
        #         _, aux = self.loss(outputs, batch, eval_seq = False, eval_total = False, return_aux=True)
        #         aux = {k: v for k, v in aux.items() if "scn/" in k}  # trim aux to sidechain diffusion metrics
        #         aux = {k: v for k, v in aux.items() if "unweighted" not in k}  # trim out unweighted loss
        #         self._log(batch, outputs, aux, batch_idx, phase="val", phase_suffix="/scn_diff",
        #                     key_suffix=f"_t_scd{t_scd}")

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
