import os
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import hydra
import lightning as L
import torch
import wandb
import yaml
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import allatom_design.data.datasets.ad_dataset as ad_dataset
from allatom_design.checkpoint_utils import EMATrackerCheckpoint
from allatom_design.data import residue_constants as rc
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="configs/seq_denoiser", config_name="seq_denoiser", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for training an sequence denoiser model.
    """
    assert cfg.resume.ckpt_path is None, "Resuming checkpoints not supported yet, should be None"

    # Update config and resolve
    update_config(cfg)  # Conditionally update certain config values
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create wandb dir
    wandb_dir = str(Path(cfg.out_dir, cfg.project))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

    # Set wandb cache directory
    wandb_cache_dir = str(Path(cfg.out_dir, cfg.project, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up dataloaders
    init_dataloader = partial(get_dataloader, num_workers=cfg.num_workers, cuda=cfg.cuda)

    _, train_dataloader = init_dataloader(phase="train", data_cfg=cfg.data, batch_size=cfg.train.batch_size)
    _, val_dataloader = init_dataloader(phase="eval", data_cfg=cfg.data, batch_size=cfg.train.batch_size)
    val_dataloaders = [val_dataloader]

    # Init wandb
    local_rank = os.environ.get("LOCAL_RANK", None)
    print(f"Local rank: {local_rank}")

    if cfg.no_wandb:
        log_dir = Path(cfg.out_dir, cfg.project, "debug")
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        logger = False  # disables logging
    else:
        if local_rank is None:
            # If none, then we are either on node rank 0 or not using DDP
            wandb.init(
                project=cfg.project,
                entity=cfg.wandb_id,
                name=cfg.exp_name,
                group=cfg.group,
                config=cfg_dict,
                dir=wandb_dir,
            )
            os.environ["WANDB_RUN_NAME"] = wandb.run.name

        wandb_run_name = os.environ["WANDB_RUN_NAME"]
        log_dir = Path(cfg.out_dir, cfg.project, wandb_run_name)  # base log dir

        # path for run outputs
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)

        logger = WandbLogger(
            name=cfg.exp_name,
            project=cfg.project,
            entity=cfg.wandb_id,
            experiment=wandb.run if local_rank is None else DummyExperiment(),
            save_dir=results_dir,
        )
        print(f"Wandb run name: {wandb_run_name}")

    # Set up logging
    ckpt_dir = Path(log_dir, "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Preserve configs
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    with open(Path(log_dir, "config_unresolved.yaml"), "w") as f:
        # also preserve unresolved config
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=False), f)

    # Set up model
    lit_model = LitSeqDenoiser(cfg)

    if not cfg.no_wandb:
        logger.watch(lit_model.model, log="all", log_freq=cfg.logging.wandb_watch_freq)

    # Define callbacks
    callbacks = []
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                 save_top_k=-1,
                                                 monitor="epoch",
                                                 mode="max",
                                                 every_n_epochs=cfg.checkpointing.save_latest_every_n_epochs,
                                                 filename="sd-epoch{epoch:02d}",
                                                 auto_insert_metric_name=False
                                                 )
    val_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=cfg.checkpointing.save_top_k,
                                              monitor="val/total_loss",
                                              mode="min",
                                              filename="sd-epoch{epoch:02d}-val_loss{val/total_loss:.4f}",
                                              auto_insert_metric_name=False  # needed since metric has / in name
                                              )

    ema_checkpoint = EMATrackerCheckpoint(save_dir=f"{ckpt_dir}/ema_tracker",
                                          save_freq_epochs=cfg.checkpointing.save_ema_every_n_epochs)

    callbacks += [latest_checkpoint_callback, val_checkpoint_callback, ema_checkpoint]

    train_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                save_top_k=cfg.checkpointing.save_top_k,
                                                monitor="train/total_loss_epoch",
                                                mode="min",
                                                filename="sd-epoch{epoch:02d}-train_loss{train/total_loss:.4f}",
                                                auto_insert_metric_name=False  # needed since metric has / in name
                                                )
    callbacks.append(train_checkpoint_callback)

    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Compute scale factors for sigma data
    scale_factors = ad_dataset.compute_scale_factors(train_dataloader, n_examples=1000)
    lit_model.model.set_scale_factors(scale_factors)  # set scale factors in model

    # Train
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.logging.log_dir,
                        log_every_n_steps=cfg.logging.log_every_n_steps,
                        callbacks=callbacks,
                        **cfg.trainer
                        )
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders)


def get_dataloader(phase: str,
                   data_cfg: DictConfig,
                   batch_size: int,
                   num_workers: int,
                   cuda: bool) -> Tuple[ADDataset, DataLoader]:
    dataset = ADDataset(phase=phase, **data_cfg)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=cuda,
                            shuffle=(phase == "train"),
                            drop_last=True
                            )
    return dataset, dataloader



def update_config(cfg: DictConfig) -> None:
    """
    Applies conditional changes to the config.
    """
    # Numerical stability for mixed precision training
    if str(cfg.trainer.precision) in ["16", "16-true", "16-mixed"]:
        cfg.model.inf = 1e4
        cfg.model.eps = 1e-4

    if getattr(cfg.denoiser, "autoguidance", None) and cfg.denoiser.autoguidance.enabled:
        # Autoguidance model parameters are not always used
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"


if __name__ == "__main__":
    main()
