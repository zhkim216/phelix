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
from allatom_design.checkpoint_utils import (EMATrackerCheckpoint,
                                             resume_ckpt_cfg)
from allatom_design.data import residue_constants as rc
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.data.datasets.multi_dataset import MultiDataset
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="configs/atom_denoiser", config_name="atom_denoiser", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for training an atom denoiser model.
    """
    # If resuming from checkpoint, get config
    if cfg.resume.ckpt_path:
        print(f"Resuming from checkpoint: {cfg.resume.ckpt_path}")
        cfg, safe_ckpt_to_resume = resume_ckpt_cfg(cfg)

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
    init_dataloader = partial(get_dataloader, data_cfg=cfg.data, num_workers=cfg.num_workers, cuda=cfg.cuda, batch_size=cfg.train.batch_size)
    _, train_dataloader = init_dataloader(phase="train")
    _, val_dataloader = init_dataloader(phase="eval")

    _, train_dataloader = init_dataloader(phase="train", data_cfg=cfg.data, batch_size=cfg.train.batch_size)
    _, val_dataloader = init_dataloader(phase="eval", data_cfg=cfg.data, batch_size=cfg.train.batch_size)
    val_dataloaders = [val_dataloader]

    if cfg.data.run_eval2:
        _, val2_dataloader = init_dataloader(phase="eval2", data_cfg=cfg.data, batch_size=cfg.train.batch_size)
        val_dataloaders.append(val2_dataloader)

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
    resumed_ckpt_path = None
    if cfg.resume.ckpt_path:
        resumed_ckpt_path = f"{ckpt_dir}/orig_resumed.ckpt"
        if local_rank is None:
            # save the original checkpoint to resume from (also handles overrides to torch.compile)
            torch.save(safe_ckpt_to_resume, resumed_ckpt_path)
        lit_model = LitAtomDenoiser.load_from_checkpoint(resumed_ckpt_path, cfg=cfg)
    else:
        lit_model = LitAtomDenoiser(cfg)

    if not cfg.no_wandb:
        logger.watch(lit_model.model, log="all", log_freq=cfg.logging.wandb_watch_freq)

    # Define callbacks
    callbacks = []
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                 save_top_k=-1,
                                                 every_n_train_steps=cfg.checkpointing.save_latest_every_n_steps,
                                                 filename="ad-step{step}-epoch{epoch:02d}",
                                                 auto_insert_metric_name=False
                                                 )
    val_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=cfg.checkpointing.save_top_k,
                                              monitor="val/total_loss",
                                              mode="min",
                                              filename="ad-epoch{epoch:02d}-step{step}-val_loss{val/total_loss:.4f}",
                                              auto_insert_metric_name=False  # needed since metric has / in name
                                              )
    ema_checkpoint = EMATrackerCheckpoint(save_dir=f"{ckpt_dir}/ema_tracker",
                                          save_freq_steps=cfg.checkpointing.save_ema_every_n_steps)

    callbacks += [latest_checkpoint_callback, val_checkpoint_callback, ema_checkpoint]

    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Compute scale factors for sigma data
    scale_factors = ad_dataset.compute_scale_factors(train_dataloader, n_examples=1000)

    # override sigma_data if specified for consistent loss scaling
    bb_sigma_data_override, scn_sigma_data_override = cfg.model.override_sigma_data
    if bb_sigma_data_override is not None:
        print(f"Overriding bb sigma data with {bb_sigma_data_override}")
        bb_mean, bb_std = scale_factors["bb"]
        bb_std = bb_sigma_data_override
        scale_factors["bb"] = (bb_mean, bb_std)
    if scn_sigma_data_override is not None:
        print(f"Overriding scn sigma data with {scn_sigma_data_override}")
        scn_mean, scn_std = scale_factors["scn"]
        scn_std = scn_sigma_data_override
        scale_factors["scn"] = (scn_mean, scn_std)

    # set scale factors in model
    lit_model.model.set_scale_factors(scale_factors)

    # Train
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.logging.log_dir,
                        log_every_n_steps=cfg.logging.log_every_n_steps,
                        callbacks=callbacks,
                        **cfg.trainer
                        )
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders, ckpt_path=resumed_ckpt_path)


def get_dataloader(phase: str,
                   data_cfg: DictConfig,
                   batch_size: int,
                   num_workers: int,
                   cuda: bool) -> Tuple[ADDataset, DataLoader]:
    num_datasets = len(data_cfg.pdb_paths)
    if data_cfg.designability_csvs is None:
        data_cfg.designability_csvs = [None] * num_datasets

    datasets = [ADDataset(pdb_path=data_cfg.pdb_paths[i],
                          designability_csv=data_cfg.designability_csvs[i],
                          phase=phase, **data_cfg) for i in range(num_datasets)]
    if phase == "train":
        dataset = MultiDataset(datasets, data_cfg.dataset_weights, primary_dset_idx=0)
    elif phase in ["eval", "eval2"]:
        # only use the primary dataset for validation
        dataset = datasets[0]
    else:
        raise ValueError(f"Invalid phase: {phase}")

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
