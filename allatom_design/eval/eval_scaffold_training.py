import pickle
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
from allatom_design.eval.eval_utils.bb_gen_utils import (
    get_bb_gen_model, run_motif_cond_type_sampling)
from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, get_training_checkpoints, wandb_setup, process_pdb_files)
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="eval_scaffold_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating scaffold-based generation.
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

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Process PDB files into .npz structure format
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)

    # Get checkpoints from denoiser training run
    ad_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "atom_denoiser",
                                                 cfg.eval_every_n_ckpts,
                                                 cfg.start_step, cfg.end_step)

    ### Sample from each checkpoint ###
    motif_cond_type_cfgs = cfg.motif_conditioning_type_cfgs

    pbar = tqdm(ad_ckpts, desc=f"Sampling on {len(processed_struct_files)} PDB(s) with {len(motif_cond_type_cfgs)} motif conditioning types and {len(ad_ckpts)} checkpoint(s)...")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))  # extract step and epoch from checkpoint name
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Load in backbone generation model
        cfg.bb_gen_cfg.ckpt_path = ad_ckpt
        bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)

        # Evaluate separately for each scaffold conditioning type
        for motif_cond_type_cfg in motif_cond_type_cfgs:
            L.seed_everything(cfg.seed)  # reset seed for each checkpoint and conditioning type

            # create output directory for this epoch and conditioning type
            log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}/{motif_cond_type_cfg['name']}"
            Path(log_dir_i).mkdir(parents=True, exist_ok=True)

            # Process PDBs in batches
            sampled_pdbs, motif_info = run_motif_cond_type_sampling(model=bb_gen_model["model"],
                                                                    data_cfg=bb_gen_model["data_cfg"],
                                                                    cfg=bb_gen_model["sampling_cfg"],
                                                                    motif_cond_type_cfg=motif_cond_type_cfg,
                                                                    device=device,
                                                                    struct_file_paths=processed_struct_files,
                                                                    out_dir=log_dir_i)

            # === CALCULATE STRUCTURE METRICS ===
            per_pdb_info, sample_metrics = eval_metrics.compute_per_pdb_info(sampled_pdbs, seq_des_model, struct_pred_model, device,
                                                                             out_dir=log_dir_i, temp_dir=f"{log_dir_i}/tmp",
                                                                             sc_kwargs={"metrics_to_compute": ["sc_ca_rmsd", "sc_ca_tm", "motif_bb_rmsd"],
                                                                                        "motif_info": motif_info},
                                                                             nntm_dataset=cfg.nntm_dataset)

            # get RMSD between input motif and sampled structure (as opposed to the predicted structure)
            for pdb in sampled_pdbs:
                master_df = motif_info[pdb]["master_df"]
                sample_metrics["sampled_motif_ca_rmsd"].append(master_df.iloc[0]["rmsd"] if master_df is not None else np.nan)

            # Save per-pdb info
            torch.save(per_pdb_info, f"{log_dir_i}/per_pdb_info.pt")

            # === Calculate a scalar for each metric to log === #
            metrics = {}

            # mean and median of all metrics
            metrics.update({f"mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
            metrics.update({f"median/{k}": np.median(v) for k, v in sample_metrics.items()})

            # for motif_bb_rmsd, calculate the number of success below 1 RMSD
            motif_rmsd_key = f"{cfg.seq_des_cfg.model_name}_motif_bb_rmsd_best"
            metrics[f"success_count/motif_bb_rmsd"] = np.sum(np.array(sample_metrics[motif_rmsd_key]) < 1.0)
            metrics[f"success_rate/motif_bb_rmsd"] = np.mean(np.array(sample_metrics[motif_rmsd_key]) < 1.0)

            # Log metrics to wandb
            metrics = {f"{motif_cond_type_cfg['name']}/{k}": v for k, v in metrics.items()}
            torch.save(metrics, f"{log_dir_i}/metrics.pt")
            if not cfg.wandb.no_wandb:
                metrics["trainer/global_step"] = global_step
                metrics["trainer/epoch"] = epoch
                wandb.log(metrics)

    if not cfg.wandb.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
