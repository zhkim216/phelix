import os
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, get_training_checkpoints, process_pdb_files, wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          run_seq_des)


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

    # Load in PDB files to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)

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
        cfg.seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

        # We set the seed each checkpoint
        L.seed_everything(cfg.seed)

        # Run sequence design model
        _, aux = run_seq_des(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                             struct_file_paths=processed_struct_files, device=device, out_dir=log_dir_i)
        sampled_pdbs = aux["out_pdbs"]

        # Run self-consistency evaluation
        out_metrics = defaultdict(list)

        id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
            sampled_pdbs,
            struct_pred_model,
            cfg.pdb_processing_cfg,
            out_dir=log_dir_i)

        # Save metrics as CSV
        metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
        metrics_df.to_csv(f"{log_dir_i}/sc_metrics.csv", index=False)

        # Aggregate results
        sc_metrics = defaultdict(list)
        for record_id, metrics in id_to_metrics.items():
            for k, v in metrics.items():
                sc_metrics[f"{k}"].append(v)

        # Update metrics
        out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
        out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

        # Log metrics to wandb
        if not cfg.wandb.no_wandb:
            out_metrics["trainer/global_step"] = global_step
            out_metrics["trainer/epoch"] = epoch

            wandb.log(out_metrics, step=global_step)


if __name__ == "__main__":
    main()
