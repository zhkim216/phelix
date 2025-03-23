import os
import pickle
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

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.bb_gen_utils import (
    get_bb_gen_model, run_bb_uncond_sampling)
from allatom_design.eval.eval_utils.eval_setup_utils import wandb_setup
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../../configs/eval/sampling", config_name="bb_unconditional", version_base="1.3.2")
def main(cfg: DictConfig):
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

    # Load in atom denoiser
    bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)
    sampling_cfg = bb_gen_model["sampling_cfg"]

    # Define the range of lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size).repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample

    if cfg.save_traj.enabled:
        save_traj_inputs = {
            "save_traj_mask": np.tile(np.arange(cfg.n_samples_per_length) < cfg.save_traj.n_traj_per_length, len(lengths_to_sample)),  # for each protein, True if we should save the trajectory
            "save_traj_steps": np.linspace(0, sampling_cfg.num_steps - 1, cfg.save_traj.limit_traj_steps, dtype=int),  # get the which diffusion timesteps we'll save along the trajectory
            "traj_conect": cfg.save_traj.traj_conect,
            "align_traj_to_last_step": cfg.save_traj.align_traj_to_last_step
        }
    else:
        save_traj_inputs = None

    # === Sample structures === #
    print(f"Drawing {cfg.n_samples_per_length} samples each of lengths {start} to {end} with step size {cfg.length_step_size}")

    # Run unconditional sampling
    sampled_pdb_paths = run_bb_uncond_sampling(model=bb_gen_model["model"],
                                               cfg=sampling_cfg,
                                               device=device,
                                               lengths=lengths_to_sample,
                                               out_dir=log_dir,
                                               save_traj_inputs=save_traj_inputs)

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)

    # === Get lengths and bins of sampled structures === #
    lengths = lengths_to_sample
    pdbs = sampled_pdb_paths
    bins = [int(length / cfg.length_bin_size) * cfg.length_bin_size for length in lengths]  # bins are defined by their starting length
    for pdb, length, bin in zip(pdbs, lengths, bins):
        all_metrics[pdb]["length"] = length
        all_metrics[pdb]["bin"] = bin

    # === Load MPNN and structure prediction models === #
    if cfg.sc.run_mpnn_sc:
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # === Get secondary structure info === #
    ss_info = eval_metrics.compute_secondary_structure_content(pdbs)
    for pdb, v in ss_info.items():
        all_metrics[pdb]["ss_info"] = v

    # === Run self-consistency evaluations === #
    sc_pdbs, sc_bins = pdbs, bins
    if cfg.sc.bin_range is not None:
        # run self-consistency only on samples within the specified bin range
        sc_subset = list(zip(*[(pdb, b) for pdb, b in zip(pdbs, bins) if cfg.sc.bin_range[0] <= b <= cfg.sc.bin_range[1]]))
        sc_pdbs, sc_bins = sc_subset if len(sc_subset) > 0 else ([], [])

    if cfg.sc.max_samples_per_bin is not None and len(sc_pdbs) > 0:
        # randomly sample to limit the number of samples per bin
        df = pd.DataFrame({"pdb": sc_pdbs, "bin": sc_bins})
        sc_pdbs = df.groupby("bin")["pdb"].apply(lambda x: x.sample(n=min(cfg.sc.max_samples_per_bin, len(x)))).tolist()

    if cfg.sc.run_mpnn_sc:
        sc_info = eval_metrics.run_self_consistency_eval(sc_pdbs,
                                                         seq_des_model,
                                                         struct_pred_model,
                                                         device,
                                                         out_dir=log_dir,
                                                         temp_dir=f"{log_dir}/tmp")
        for pdb, v in sc_info.items():
            all_metrics[pdb]["sc_info"] = v

    # === Run nnTM evaluation === #
    if cfg.nntm_dataset is not None:
        nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=log_dir)
        for pdb, v in nntm_info.items():
            all_metrics[pdb]["nntm_info"] = v

    # === Aggregate metrics by length === #
    bin_to_metrics = {}
    for bin in set(bins):
        pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if b == bin]

        metrics_b = defaultdict(list)
        for pdb in pdbs_b:
            # Secondary structure metrics
            for k, v in all_metrics[pdb]["ss_info"].items():
                metrics_b[f"{k}"].append(v)

            # MPNN self-consistency metrics
            if "sc_info" in all_metrics[pdb]:
                for k, v in all_metrics[pdb]["sc_info"]["sc_metrics"].items():
                    # take mean and best across MPNN sequences
                    best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                    metrics_b[f"{cfg.seq_des_cfg.model_name}_{k}_best"].append(best_sc_metric.item())

                    if len(v) > 1:
                        # only log mean if there are multiple MPNN sequences per backbone
                        mean_sc_metric = torch.mean(v)
                        metrics_b[f"{cfg.seq_des_cfg.model_name}_{k}_mean"].append(mean_sc_metric.item())

            # nnTM metrics
            if "nntm_info" in all_metrics[pdb]:
                metrics_b["nntm"].append(all_metrics[pdb]["nntm_info"])

        # Average metrics across samples
        metrics_b_mean = {f"mean/{k}": np.mean(v) for k, v in metrics_b.items()}
        metrics_b_median = {f"median/{k}": np.median(v) for k, v in metrics_b.items()}

        bin_to_metrics[bin] = metrics_b_mean
        bin_to_metrics[bin].update(metrics_b_median)

    # === Calculate mean pairwise TM score by length === #
    for bin in set(bins):
        pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if b == bin]
        coords_b = [load_feats_from_pdb(pdb)["all_atom_positions"] for pdb in pdbs_b]
        bin_to_metrics[bin]["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(coords_b,
                                                                                    temp_dir=f"{log_dir}/tmp",
                                                                                    subsample_pairs=cfg.pairwise_tm_subsample)

    # === Run clustering analysis === #
    for sctm_cutoff in cfg.clustering.sctm_cutoffs:
        # Cluster by length bin
        for bin in set(bins):
            pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if (b == bin) and ("sc_info" in all_metrics[pdb])]
            if len(pdbs_b) == 0:
                # skip if we don't have self-consistency info for any samples in this bin
                continue

            # Cluster only on designable samples (scTM > sctm_cutoff)
            designable_pdbs = [pdb for pdb in pdbs_b if (all_metrics[pdb]["sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff).any()]
            bin_to_metrics[bin][f"{cfg.seq_des_cfg.model_name}_sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

            cluster_out_dir = Path(f"{log_dir}/clustering/bin{bin}_sctm{sctm_cutoff}")
            bin_to_metrics[bin][f"{cfg.seq_des_cfg.model_name}_sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(
                designable_pdbs,
                cluster_out_dir,
                f"{log_dir}/tmp",
                **cfg.clustering.foldseek_opts
            )


    # === Compute KL(p||q) for secondary structure distributions === #
    if cfg.ss_kld.dssp_csv is not None:
        dssp_df = pd.read_csv(cfg.ss_kld.dssp_csv)
        dssp_df["% Helix"] = dssp_df["% Helix"] * 100
        dssp_df["% Strand"] = dssp_df["% Strand"] * 100
        for bin in set(bins):
            pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if b == bin]
            dssp_df_b = dssp_df[(bin <= dssp_df["length"]) & (dssp_df["length"] <= bin + cfg.length_bin_size)]
            ss_info_df_b = pd.DataFrame([all_metrics[pdb]["ss_info"] for pdb in pdbs_b], index=pdbs_b)

            p_alpha, p_beta = dssp_df_b["% Helix"].tolist(), dssp_df_b["% Strand"].tolist()
            q_alpha, q_beta = ss_info_df_b["pct_alpha"].tolist(), ss_info_df_b["pct_beta"].tolist()
            bin_to_metrics[bin]["ss_kld"] = eval_metrics.compute_ss_kl(p_alpha, p_beta, q_alpha, q_beta,
                                                                    bin_size=cfg.ss_kld.bin_size, pseudocount=cfg.ss_kld.pseudocount)


    ### SAVE METRICS ###
    # Save metrics to pickle file
    with open(f"{log_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    with open(f"{log_dir}/L_to_metrics.pkl", "wb") as f:
        pickle.dump(bin_to_metrics, f)

    # Set up wandb logging
    if not cfg.wandb.no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(log_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(log_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.wandb_id,
            name=cfg.exp_name,
            group=cfg.wandb.group,
            config=cfg_dict,
            dir=wandb_dir,
        )

        # Log metrics
        for bin in sorted(bin_to_metrics.keys()):
            metrics_b = bin_to_metrics[bin]
            metrics_b["length_bin"] = bin
            wandb.log(metrics_b, step=bin)

        wandb.finish()


if __name__ == "__main__":
    main()
