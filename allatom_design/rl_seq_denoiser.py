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

import allatom_design.data.datasets.rl_sd_dataset as rl_sd_dataset
from allatom_design.checkpoint_utils import (EMATrackerCheckpoint,
                                             get_cfg_from_ckpt)
from allatom_design.data.datasets.rl_sd_dataset import RLSDDataset, contrastive_collate_fn
from allatom_design.model.seq_denoiser.lit_rl_sd_model import LitRLSeqDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="configs/seq_denoiser", config_name="rl_seq_denoiser", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for finetuning a sequence denoiser model with RL.
    """
    # Load from checkpoint
    print(f"Finetuning from checkpoint: {cfg.ckpt_path}")
    lit_base_model = LitSeqDenoiser.load_from_checkpoint(cfg.ckpt_path)
    base_cfg = lit_base_model.cfg

    # Resolve config
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

    _, train_dataloader = init_dataloader(pdb_path=cfg.train_pdb_path, data_cfg=cfg.data, batch_size=cfg.train.batch_size, shuffle=True)
    _, val_dataloader = init_dataloader(pdb_path=cfg.val_pdb_path, data_cfg=cfg.data, batch_size=cfg.train.batch_size, shuffle=False)
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
    lit_model = LitRLSeqDenoiser(cfg, base_cfg)
    lit_model.load_base_model(lit_base_model.model)

    if not cfg.no_wandb:
        logger.watch(lit_model.model, log="all", log_freq=cfg.logging.wandb_watch_freq)

    # Define callbacks
    callbacks = []
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                 save_top_k=-1,
                                                 every_n_train_steps=cfg.checkpointing.save_latest_every_n_steps,
                                                 filename="sd-step{step}-epoch{epoch:02d}",
                                                 auto_insert_metric_name=False
                                                 )
    val_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=cfg.checkpointing.save_top_k,
                                              monitor="val/total_loss",
                                              mode="min",
                                              filename="sd-epoch{epoch:02d}-step{step}-val_loss{val/total_loss:.4f}",
                                              auto_insert_metric_name=False  # needed since metric has / in name
                                              )
    ema_checkpoint = EMATrackerCheckpoint(save_dir=f"{ckpt_dir}/ema_tracker",
                                          save_freq_steps=cfg.checkpointing.save_ema_every_n_steps)

    callbacks += [latest_checkpoint_callback, val_checkpoint_callback, ema_checkpoint]

    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Train
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.logging.log_dir,
                        log_every_n_steps=cfg.logging.log_every_n_steps,
                        callbacks=callbacks,
                        **cfg.trainer
                        )
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders)


def get_dataloader(pdb_path: str,
                   data_cfg: DictConfig,
                   batch_size: int,
                   num_workers: int,
                   cuda: bool,
                   shuffle: bool) -> Tuple[RLSDDataset, DataLoader]:
    dataset = RLSDDataset(pdb_path, **data_cfg)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=cuda,
                            shuffle=shuffle,
                            drop_last=True,
                            collate_fn=contrastive_collate_fn
                            )
    return dataset, dataloader


if __name__ == "__main__":
    main()
