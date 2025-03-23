from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.bb_gen_utils import (
    get_bb_gen_model, run_bb_uncond_sampling)
from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_training_checkpoints, wandb_setup)
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="eval_bb_gen_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating backbone generation during training.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in MPNN + structure prediction model for self-consistency evals
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Get checkpoints from denoiser training run
    ad_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "atom_denoiser",
                                                 cfg.eval_every_n_ckpts,
                                                 cfg.start_step, cfg.end_step)

    # Get lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
    lengths_to_sample = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample

    # Evaluate each checkpoint
    pbar = tqdm(ad_ckpts, desc="Evaluating checkpoints...")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))  # extract step and epoch from checkpoint name
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)

        # Load in backbone generation model
        cfg.bb_gen_cfg.ckpt_path = ad_ckpt
        bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)

        # We set the seed each checkpoint
        L.seed_everything(cfg.seed)

        # Run backbone sampling
        sampled_pdb_paths = run_bb_uncond_sampling(model=bb_gen_model["model"],
                                                   cfg=bb_gen_model["sampling_cfg"],
                                                   device=device,
                                                   lengths=lengths_to_sample,
                                                   out_dir=log_dir_i)

        ### CALCULATE STRUCTURE METRICS ###
        per_pdb_info, sample_metrics = eval_metrics.compute_per_pdb_info(sampled_pdb_paths, seq_des_model, struct_pred_model, device,
                                                                         out_dir=log_dir_i, temp_dir=f"{log_dir_i}/tmp", nntm_dataset=cfg.nntm_dataset)

        # Save per-pdb info
        torch.save(per_pdb_info, f"{log_dir_i}/per_pdb_info.pt")

        # === Calculate a scalar for each metric to log === #
        metrics = {}
        metrics.update({f"mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
        metrics.update({f"median/{k}": np.median(v) for k, v in sample_metrics.items()})

        ### Compute metrics that require all samples ##
        if cfg.compute_diversity_metrics:
            diversity_metrics = eval_metrics.run_diversity_eval(sampled_pdb_paths, per_pdb_info, cfg.diversity_eval, log_dir_i)
            metrics.update(diversity_metrics)

        # Log aggregated metrics to wandb
        metrics = {f"bb_gen/{k}": v for k, v in metrics.items()}
        torch.save(metrics, f"{log_dir_i}/metrics.pt")
        if not cfg.wandb.no_wandb:
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch
            wandb.log(metrics, step=global_step)

    # Finish wandb logging
    if not cfg.wandb.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
