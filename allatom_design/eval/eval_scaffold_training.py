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
    get_bb_gen_model, run_sm_sampling)
from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, get_training_checkpoints, wandb_setup)
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

    # Get checkpoints from denoiser training run
    ad_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "atom_denoiser",
                                                 cfg.eval_every_n_ckpts,
                                                 cfg.start_step, cfg.end_step)

    ### Sample from each checkpoint ###
    pbar = tqdm(ad_ckpts, desc=f"Sampling on {len(pdb_files)} PDB(s) with {len(ad_ckpts)} checkpoint(s)...")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))  # extract step and epoch from checkpoint name
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Load in backbone generation model
        cfg.bb_gen_cfg.ckpt_path = ad_ckpt
        bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)
        sm = bb_gen_model["scaffold_manager"]

        # Evaluate separately for each scaffold conditioning type
        for scaffold_conditioning_type in cfg.scaffold_conditioning_types:
            L.seed_everything(cfg.seed)  # reset seed for each checkpoint and conditioning type
            sm.set_conditioning_type(scaffold_conditioning_type)  # set the conditioning type for the scaffold manager

            # create output directory for this epoch and conditioning type
            log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}/{scaffold_conditioning_type}"
            Path(log_dir_i).mkdir(parents=True, exist_ok=True)
            motif_pdbs_dir_i = f"{log_dir_i}/motif_pdbs"
            Path(motif_pdbs_dir_i).mkdir(parents=True, exist_ok=True)
            centered_gt_pdbs_dir_i = f"{log_dir_i}/centered_gt_pdbs"
            Path(centered_gt_pdbs_dir_i).mkdir(parents=True, exist_ok=True)
            sampled_pdbs_dir_i = f"{log_dir_i}/sampled_pdbs"
            Path(sampled_pdbs_dir_i).mkdir(parents=True, exist_ok=True)
            saved_metrics_dir_i = f"{log_dir_i}/metrics"
            Path(saved_metrics_dir_i).mkdir(parents=True, exist_ok=True)

            # Process PDBs in batches
            sampled_pdbs, motif_info = run_sm_sampling(model=bb_gen_model["model"],
                                                        sm=sm,
                                                        cfg=cfg.bb_gen_cfg.sampling_cfg,
                                                        device=device,
                                                        pdb_paths=pdb_files,
                                                        out_dir=log_dir_i)

            # === CALCULATE STRUCTURE METRICS ===
            all_metrics = defaultdict(dict)

            # Secondary structure
            ss_info = eval_metrics.compute_secondary_structure_content(sampled_pdbs)
            for pdb, v in ss_info.items():
                all_metrics[pdb]["ss_info"] = v

            # MPNN + structure prediction self-consistency
            sc_info = eval_metrics.run_self_consistency_eval(sampled_pdbs,
                                                             seq_des_model,
                                                             struct_pred_model,
                                                             device,
                                                             out_dir=log_dir_i,
                                                             temp_dir=f"{log_dir_i}/tmp",
                                                             metrics_to_compute=["sc_ca_rmsd", "sc_ca_tm", "motif_bb_rmsd"],
                                                             motif_info=motif_info
                                                             )
            for pdb, v in sc_info.items():
                all_metrics[pdb]["sc_info"] = v

            # nnTM
            if cfg.nntm_dataset is not None:
                nntm_info = eval_metrics.run_nntm_eval(sampled_pdbs, dataset=cfg.nntm_dataset, out_dir=log_dir_i)
                for pdb, v in nntm_info.items():
                    all_metrics[pdb]["nntm_info"] = v

            # get RMSD between input motif and sampled structure
            for pdb in sampled_pdbs:
                all_metrics[pdb]["sampled_motif_bb_rmsd"] = eval_metrics.compute_motif_bb_rmsd(pdb, motif_info[pdb]["x_motif"], motif_info[pdb]["motif_mask"])

            # Save per-sample metrics as pt
            torch.save(all_metrics, f"{saved_metrics_dir_i}/step_{global_step}_all_metrics.pt")

            # Aggregate metrics
            sample_metrics = defaultdict(list)
            for pdb in sampled_pdbs:
                # secondary structure metrics
                for k, v in ss_info[pdb].items():
                    sample_metrics[f"{k}"].append(v)

                # self-consistency metrics
                for k, v in sc_info[pdb]["sc_metrics"].items():
                    best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                    sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_best"].append(best_sc_metric.item())

                    if len(v) > 1:
                        # only report mean if we run multiple sequences per sample
                        mean_sc_metric = torch.mean(v)
                        sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_mean"].append(mean_sc_metric.item())

                # nnTM metrics
                if cfg.nntm_dataset is not None:
                    sample_metrics["nntm"].append(all_metrics[pdb]["nntm_info"])

                # RMSD between input motif and sampled structure
                sample_metrics["sampled_motif_bb_rmsd"].append(all_metrics[pdb]["sampled_motif_bb_rmsd"])

            # === Calculate metrics to log === #
            metrics = {}

            # mean and median of all metrics
            metrics.update({f"scaffold/mean/{scaffold_conditioning_type}/{k}": np.mean(v) for k, v in sample_metrics.items()})
            metrics.update({f"scaffold/median/{scaffold_conditioning_type}/{k}": np.median(v) for k, v in sample_metrics.items()})

            # for motif_bb_rmsd, calculate the number of success below 1 RMSD
            motif_rmsd_key = f"{cfg.seq_des_cfg.model_name}_motif_bb_rmsd_best"
            metrics[f"scaffold/success_count/{scaffold_conditioning_type}/motif_bb_rmsd"] = np.sum(np.array(sample_metrics[motif_rmsd_key]) < 1.0)
            metrics[f"scaffold/success_rate/{scaffold_conditioning_type}/motif_bb_rmsd"] = np.mean(np.array(sample_metrics[motif_rmsd_key]) < 1.0)


            if not cfg.wandb.no_wandb:
                metrics["trainer/global_step"] = global_step
                metrics["trainer/epoch"] = epoch
                wandb.log(metrics)

    if not cfg.wandb.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
