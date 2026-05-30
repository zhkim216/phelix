import os
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
import yaml
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from atomworks.ml.samplers import set_sampler_epoch
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from allatom_design.callback.unusedparamdetector import UnusedParamDetector
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from allatom_design.model.ema.ema import EMA, EMAModelCheckpoint, EMATrackerCheckpoint
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.utils.checkpoint_utils import repair_state_dict


def build_sd_datamodule(data_cfg: DictConfig) -> L.LightningDataModule:
    dataset_impl = data_cfg.get("dataset_impl", "atomworks_sd")
    if dataset_impl == "atomworks_sd":
        from allatom_design.data.datasets.atomworks_sd_dataset import AtomworksSDDataModule

        return AtomworksSDDataModule(data_cfg)
    if dataset_impl == "mg_proto":
        from allatom_design.data.datasets.atomworks_sd_dataset_mg_proto import (
            AtomworksSDMGProtoDataModule,
        )

        return AtomworksSDMGProtoDataModule(data_cfg)
    if dataset_impl == "proto":
        from allatom_design.data.datasets.atomworks_sd_dataset_proto import (
            AtomworksSDProtoDataModule,
        )

        return AtomworksSDProtoDataModule(data_cfg)
    raise ValueError(
        f"Unknown data.dataset_impl={dataset_impl!r}. "
        "Supported values: 'atomworks_sd', 'mg_proto', 'proto'."
    )


@hydra.main(config_path="configs_local/seq_denoiser", config_name="mg_proto_no_filter", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for training an sequence denoiser model.
    """
    # Get resume checkpoint path if provided
    resume_ckpt_path = cfg.resume.ckpt_path
    resume_run_name = None  # Will be set if resuming
    if resume_ckpt_path is not None:
        assert Path(resume_ckpt_path).exists(), f"Resume checkpoint not found: {resume_ckpt_path}"
        print(f"Will resume training from checkpoint: {resume_ckpt_path}")

        # Extract run name from checkpoint path (e.g., lilac-sound-509 from .../lilac-sound-509/checkpoints/...)
        resume_run_name = Path(resume_ckpt_path).parent.parent.name
        print(f"Will resume with run name: {resume_run_name}")

        # Handle config loading from checkpoint directory
        cfg = load_resume_config(cfg, resume_ckpt_path)

    # Update config and resolve
    update_config(cfg)  # Conditionally update certain config values
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create wandb dir and set wandb cache directory
    wandb_dir = str(Path(cfg.out_dir, cfg.wandb.project))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)
    wandb_cache_dir = str(Path(cfg.out_dir, cfg.wandb.project, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.train.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up LightningDataModule
    datamodule = build_sd_datamodule(cfg.data)

    # Init wandb only on node rank 0
    local_rank = os.environ.get("LOCAL_RANK", None)
    print(f"Local rank: {local_rank}")

    if cfg.wandb.no_wandb:
        log_dir = Path(cfg.out_dir, cfg.wandb.project, "debug")
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        logger = False  # disables logging
    else:
        if local_rank is None:
            # If none, then we are either on node rank 0 or not using DDP
            # Use resume_run_name if resuming, otherwise use exp_name (wandb will auto-generate if None)
            run_name = resume_run_name if resume_run_name else cfg.exp_name
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.wandb_id,
                name=run_name,
                group=cfg.wandb.group,
                config=cfg_dict,
                dir=wandb_dir,
            )
            os.environ["WANDB_RUN_NAME"] = wandb.run.name

        wandb_run_name = os.environ["WANDB_RUN_NAME"]

        # If resuming, use the original run directory instead of creating a new one
        if resume_run_name:
            log_dir = Path(cfg.out_dir, cfg.wandb.project, resume_run_name)
        else:
            log_dir = Path(cfg.out_dir, cfg.wandb.project, wandb_run_name)  # base log dir

        # path for run outputs
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)

        logger = WandbLogger(
            name=cfg.exp_name,
            project=cfg.wandb.project,
            entity=cfg.wandb.wandb_id,
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

    # Set up model
    lit_model = LitSeqDenoiser(cfg)

    # Load only model weights from the pretrained checkpoint
    if cfg.finetuning.enabled:
        print(f"Loading model weights from {cfg.finetuning.ckpt_path}")
        checkpoint = torch.load(cfg.finetuning.ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        if not cfg.train.compile.compile_model:
            state_dict = repair_state_dict(state_dict)
        lit_model.load_state_dict(state_dict, strict=True)

    if not cfg.wandb.no_wandb and cfg.logging.get("wandb_watch_enabled", False):
        logger.watch(lit_model.model,
                     log=cfg.logging.get("wandb_watch_mode", "gradients"),
                     log_freq=cfg.logging.wandb_watch_freq)

    # Define callbacks
    callbacks = []
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                 save_top_k=-1,
                                                 every_n_train_steps=cfg.checkpointing.save_latest_every_n_steps,
                                                 filename="sd-step{step}-epoch{epoch:02d}",
                                                 auto_insert_metric_name=False)

    epoch_latest_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                       monitor="epoch",
                                                       mode="max",
                                                       save_top_k=1,
                                                       every_n_epochs=cfg.checkpointing.save_for_resuming_every_n_epochs,
                                                       filename="sd-epoch{epoch:02d}",
                                                       auto_insert_metric_name=False)

    callbacks += [latest_checkpoint_callback, epoch_latest_checkpoint_callback]

    sampler_epoch_callback = SamplerEpochCallback()
    callbacks.append(sampler_epoch_callback)

    # EMA callbacks
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
            filename="sd-step{step}-epoch{epoch:02d}-ema" + str(ema_decay),
            auto_insert_metric_name=False
        )
        callbacks.append(latest_ema_checkpoint_callback)

    if cfg.train.debug:
        unused_param_detector = UnusedParamDetector()
        callbacks.append(unused_param_detector)

    # log learning rate
    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Set sigma data in model
    bb_std, scn_std = cfg.model.sigma_data
    bb_mean, scn_mean = 0.0, 0.0  # unused; for backwards compatibility
    scale_factors = {"bb": (bb_mean, bb_std), "scn": (scn_mean, scn_std),}
    lit_model.model.set_scale_factors(scale_factors)

    # Train
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.logging.log_dir,
                        log_every_n_steps=cfg.logging.log_every_n_steps,
                        callbacks=callbacks,
                        **cfg.trainer
                        )
    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=resume_ckpt_path)

class SamplerEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if hasattr(dm, "_train_sampler"):
            set_sampler_epoch(dm._train_sampler, trainer.current_epoch)

def load_resume_config(cfg: DictConfig, resume_ckpt_path: str) -> DictConfig:
    """
    Load config from checkpoint directory if use_current_cfg is False,
    and apply any overrides specified in resume.overrides.

    Args:
        cfg: Current config from hydra
        resume_ckpt_path: Path to the checkpoint file

    Returns:
        Updated config (either current or loaded from checkpoint with overrides)
    """
    use_current_cfg = cfg.resume.get("use_current_cfg", True)
    overrides = cfg.resume.get("overrides", {})

    if not use_current_cfg:
        # Load config from checkpoint directory
        # Checkpoint path: /path/to/run/checkpoints/sd-step100-epoch03.ckpt
        # Config path: /path/to/run/config.yaml
        ckpt_path = Path(resume_ckpt_path)
        run_dir = ckpt_path.parent.parent  # Go up from checkpoints/ to run directory
        saved_config_path = run_dir / "config.yaml"

        if not saved_config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {saved_config_path}. "
                f"Cannot resume with use_current_cfg=False. "
                f"Set use_current_cfg=True to use current config instead."
            )

        print(f"Loading config from checkpoint directory: {saved_config_path}")
        with open(saved_config_path, "r") as f:
            saved_cfg_dict = yaml.safe_load(f)

        # Convert to OmegaConf
        cfg = OmegaConf.create(saved_cfg_dict)

        # Restore resume settings from current config (so we keep the ckpt_path etc.)
        cfg.resume = OmegaConf.create({
            "ckpt_path": resume_ckpt_path,
            "use_current_cfg": False,
            "overrides": overrides
        })

    # Apply overrides
    if overrides:
        print(f"Applying config overrides: {overrides}")
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value, merge=True)

    return cfg


def update_config(cfg: DictConfig) -> None:
    """
    Applies conditional changes to the config.
    """
    # Numerical stability for mixed precision training
    if str(cfg.trainer.precision) in ["16", "16-true", "16-mixed"]:
        cfg.model.inf = 1e4
        cfg.model.eps = 1e-4


if __name__ == "__main__":
    main()
