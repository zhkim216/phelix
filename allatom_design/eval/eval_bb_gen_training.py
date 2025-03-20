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

from allatom_design.data.data import load_feats_from_pdb
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
        sampled_pdb_paths = run_bb_uncond_sampling(bb_gen_model["model"],
                                                   bb_gen_model["sampling_cfg"],
                                                   lengths_to_sample,
                                                   device,
                                                   out_dir=log_dir_i)

        ### CALCULATE STRUCTURE METRICS ###
        all_metrics = defaultdict(dict)
        pdbs = sampled_pdb_paths

        # Get secondary structure info
        ss_info = eval_metrics.compute_secondary_structure_content(pdbs)
        for pdb, v in ss_info.items():
            all_metrics[pdb]["ss_info"] = v

        # Run MPNN + structure prediction self-consistency evals
        sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                         seq_des_model,
                                                         struct_pred_model,
                                                         device,
                                                         out_dir=log_dir_i,
                                                         temp_dir=f"{log_dir_i}/tmp")
        for pdb, v in sc_info.items():
            all_metrics[pdb]["sc_info"] = v

        # Run nnTM evaluation
        if cfg.nntm_dataset is not None:
            nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=log_dir_i)

            for pdb, v in nntm_info.items():
                all_metrics[pdb]["nntm_info"] = v


        ### SAVE METRICS ###
        # Save all metrics to pickle file
        with open(f"{log_dir_i}/all_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)

        # Aggregate per-pdb metrics
        sample_metrics = defaultdict(list)
        for pdb in pdbs:
            # secondary structure metrics
            for k, v in ss_info[pdb].items():
                sample_metrics[f"{k}"].append(v)

            # self-consistency metrics
            for k, v in sc_info[pdb]["sc_metrics"].items():
                # take mean and best across MPNN sequences
                best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_best"].append(best_sc_metric.item())

                if len(v) > 1:
                    # only report mean if we run multiple sequences per sample
                    sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_mean"].append(mean_sc_metric.item())
                    mean_sc_metric = torch.mean(v)

            # nntm metrics
            if cfg.nntm_dataset is not None:
                sample_metrics["nntm"].append(nntm_info[pdb])

        ### Compute metrics that require all samples ##
        if cfg.compute_diversity_metrics:
            # === Calculate mean pairwise TM score === #
            coords = [load_feats_from_pdb(pdb)["all_atom_positions"] for pdb in pdbs]
            sample_metrics["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(coords,
                                                                                    temp_dir=f"{log_dir_i}/tmp",
                                                                                    subsample_pairs=cfg.pairwise_tm_subsample)

            # === Run clustering analysis === #
            for sctm_cutoff in cfg.clustering.sctm_cutoffs:
                # Cluster only on designable samples (scTM > sctm_cutoff)
                designable_pdbs = [pdb for pdb in pdbs if (all_metrics[pdb]["sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff).any()]
                sample_metrics[f"{cfg.seq_des_cfg.model_name}_sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

                cluster_out_dir = Path(f"{log_dir_i}/clustering/sctm{sctm_cutoff}")
                sample_metrics[f"{cfg.seq_des_cfg.model_name}_sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(designable_pdbs, cluster_out_dir, f"{log_dir_i}/tmp",
                                                                                              **cfg.clustering.foldseek_opts)

        # === Calculate metrics to log === #
        metrics = {}
        metrics.update({f"bb_gen/mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
        metrics.update({f"bb_gen/median/{k}": np.median(v) for k, v in sample_metrics.items()})

        # Log metrics to wandb
        if not cfg.wandb.no_wandb:
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
