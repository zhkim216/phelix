import os
from pathlib import Path

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

import allatom_design.data.datasets.ad_dataset as ad_dataset
from allatom_design.checkpoint_utils import (EMATrackerCheckpoint,
                                             resume_ckpt_cfg)
from allatom_design.data import residue_constants as rc
from allatom_design.data.datasets.ad_dataset import LitADDataModule
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.ema import EMA, EMAModelCheckpoint


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

    # Create wandb dir and set wandb cache directory
    wandb_dir = str(Path(cfg.out_dir, cfg.project))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)
    wandb_cache_dir = str(Path(cfg.out_dir, cfg.project, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up LightningDataModule
    datamodule = LitADDataModule(
        data_cfg=cfg.data,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.num_workers,
        cuda=cfg.cuda,
    )

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
                                                 auto_insert_metric_name=False)

    epoch_latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                       monitor="epoch",
                                                       mode="max",
                                                       save_top_k=1,
                                                       every_n_epochs=cfg.checkpointing.save_for_resuming_every_n_epochs,
                                                       filename="ad-epoch{epoch:02d}",
                                                       auto_insert_metric_name=False)

    callbacks += [latest_checkpoint_callback, epoch_latest_checkpoint_callback]

    if cfg.model.ema.use_phema:
        # For post-hoc EMA, we save snapshots to an ema_tracker directory so we can reconstruct the EMA profile afterwards
        ema_checkpoint = EMATrackerCheckpoint(save_dir=f"{ckpt_dir}/ema_tracker",
                                              save_freq_steps=cfg.checkpointing.save_phema_every_n_steps)
        callbacks.append(ema_checkpoint)
    else:
        # Otherwise, we directly save the EMA model to checkpoints/ema
        # EMA callback to average model weights
        ema_decay = cfg.model.ema.ema_decay
        ema_callback = EMA(decay=ema_decay)
        callbacks.append(ema_callback)

        # Save EMA model under checkpoints/ema
        ema_ckpt_dir = f"{ckpt_dir}/ema"
        Path(ema_ckpt_dir).mkdir(parents=True, exist_ok=True)
        latest_ema_checkpoint_callback = EMAModelCheckpoint(
            dirpath=ema_ckpt_dir,
            save_top_k=-1,
            every_n_train_steps=cfg.checkpointing.save_latest_every_n_steps,
            filename="ad-step{step}-epoch{epoch:02d}-ema" + str(ema_decay),
            auto_insert_metric_name=False
        )
        callbacks.append(latest_ema_checkpoint_callback)

    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Set sigma data in model
    lit_model.model.set_sigma_data(cfg.model.sigma_data)

    # Train
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.logging.log_dir,
                        log_every_n_steps=cfg.logging.log_every_n_steps,
                        callbacks=callbacks,
                        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),  # number of devices per node
                        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),  # number of nodes
                        **cfg.trainer
                        )
    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=resumed_ckpt_path)



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

    if cfg.data.cluster_sample:
        # if we're using cluster sampling, we want to reload the dataloader every epoch
        cfg.trainer.reload_dataloaders_every_n_epochs = 1
    else:
        # don't reload dataloaders every epoch
        cfg.trainer.reload_dataloaders_every_n_epochs = 0


if __name__ == "__main__":
    main()
