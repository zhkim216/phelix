import os
from collections import defaultdict
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
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                 get_training_checkpoints,
                                                 wandb_setup)
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model, run_fampnn
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="eval_seq_des_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the self-consistency of a sequence denoiser model during training.
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

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Load in structure prediction model for co-design self-consistency evals
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Get checkpoints from denoiser training run
    sd_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "seq_denoiser",
                                                 cfg.eval_every_n_ckpts,
                                                 cfg.start_step, cfg.end_step)

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints for self-consistency...")
    for sd_ckpt in pbar:
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))  # extract step and epoch from checkpoint name
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)

        # Load in sequence design model
        cfg.seq_des_cfg.fampnn.fampnn_ckpt = sd_ckpt
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

        # We set the seed each checkpoint
        L.seed_everything(cfg.seed)

        # Run FAMPNN
        _, aux = run_fampnn(seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"],
                            pdb_paths=pdb_files, device=device, out_dir=log_dir_i)
        sampled_pdbs = aux["out_pdbs"]

        # Run self-consistency evaluation
        out_metrics = defaultdict(list)

        sc_info = eval_metrics.run_self_consistency_eval(
            sampled_pdbs,
            None,
            struct_pred_model,
            device,
            out_dir=log_dir_i,
            temp_dir=f"{log_dir_i}/tmp"
        )

        # Aggregate results
        sc_metrics = defaultdict(list)
        for pdb in sampled_pdbs:
            if "sc_metrics" in sc_info[pdb]:
                for k, v in sc_info[pdb]["sc_metrics"].items():
                    sc_metrics[f"{k}"].append(v.item())
            else:
                print(f"No self-consistency metrics for {pdb}, skipping...")

        # Update metrics
        out_metrics = {f"seq_des/mean/{k}": np.mean(v) for k, v in sc_metrics.items()}
        out_metrics.update({f"seq_des/median/{k}": np.median(v) for k, v in sc_metrics.items()})

        # Log metrics to wandb
        if not cfg.wandb.no_wandb:
            out_metrics["trainer/global_step"] = global_step
            out_metrics["trainer/epoch"] = epoch

            wandb.log(out_metrics, step=global_step)


if __name__ == "__main__":
    main()
