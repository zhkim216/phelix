import glob
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.fampnn_utils import get_seq_des_model
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_bb_gen_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating backbone generation.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create wandb dir
    wandb_dir = str(Path(cfg.out_dir))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

    # Set wandb cache directory
    wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up logging
    if cfg.no_wandb:
        log_dir = Path(cfg.out_dir, "debug")
    else:
        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )
        log_dir = Path(cfg.out_dir, wandb.run.name)  # base log dir

    # Set up out directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)

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
    ema_ckpt_dir = f"{cfg.denoiser_train_dir}/checkpoints/ema"
    if Path(ema_ckpt_dir).exists():
        # Use EMA checkpoints if they exist
        print(f"Using EMA checkpoints from {ema_ckpt_dir}")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)-ema(\d+\.\d+)\.ckpt$")  # match checkpoints of the form ad-step{step}-epoch{epoch}-ema{decay_rate}.ckpt
        ad_ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]
    else:
        print(f"Using non-EMA checkpoints from {cfg.denoiser_train_dir}/checkpoints")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)\.ckpt$")  # Only match checkpoints of the form ad-step{step}-epoch{epoch}.ckpt
        ad_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    pbar = tqdm(ad_ckpts, desc="Evaluating checkpoints")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Skip if global_step is before start_step
        if (cfg.start_step is not None) and (global_step < cfg.start_step):
            continue

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)
        sampled_pdbs_dir_i = f"{log_dir_i}/sampled_pdbs"
        Path(sampled_pdbs_dir_i).mkdir(parents=True, exist_ok=True)
        saved_metrics_dir_i = f"{log_dir_i}/metrics"
        Path(saved_metrics_dir_i).mkdir(parents=True, exist_ok=True)

        # Load denoiser model
        lit_ad_model = LitAtomDenoiser.load_from_checkpoint(ad_ckpt).eval()

        ### BEGIN EVAL ###
        # Define the range of lengths to sample
        start, end = cfg.length_range
        lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
        all_lengths = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample

        # Sample backbones
        pdbs = []
        L.seed_everything(cfg.seed)  # reset seed for each checkpoint
        for i in range(0, len(all_lengths), cfg.batch_size):
            # Choose lengths and residue index
            lengths = torch.tensor(all_lengths[i:i + cfg.batch_size], dtype=torch.long).to(lit_ad_model.device)
            B = lengths.shape[0]
            residue_index = torch.arange(lengths.max(), dtype=torch.long).to(lit_ad_model.device)
            residue_index = residue_index[None].expand(B, -1)

            # Create timesteps for backbone
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
            t_bb = t_bb[None].expand(B, -1).to(lit_ad_model.device)
            timesteps = t_bb

            # Create noise schedule for backbone
            noise_schedule = NoiseSchedule(cfg.noise_schedule)

            # Create churn config for backbone
            churn_cfg = dict(cfg.churn_cfg)

            cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, lit_ad_model.device)
            x_bb_denoised, aux = lit_ad_model.model.sample(lengths,
                                                            residue_index=residue_index,
                                                            timesteps=timesteps,
                                                            cond_labels=cond_labels_in,
                                                            noise_schedule=noise_schedule,
                                                            churn_cfg=churn_cfg,
                                                            autoguidance_cfg=dict(cfg.autoguidance_cfg),
                                                            )

            samples = {"x_bb_denoised": x_bb_denoised,
                        "seq_mask": aux["seq_mask"],
                        "residue_index": residue_index}
            samples = {k: v.cpu() if v is not None else v for k, v  in samples.items()}

            # Save samples
            filenames = [f"{sampled_pdbs_dir_i}/step_{global_step}_sample_{i+j}_L{l.item()}.pdb" for j, l in enumerate(lengths)]

            AtomDenoiser.save_samples_to_pdb(samples, filenames)
            pdbs.extend(filenames)

        ### CALCULATE STRUCTURE METRICS ###
        all_metrics = defaultdict(dict)
        pdbs = natsorted(pdbs)

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
        with open(f"{saved_metrics_dir_i}/epoch_{epoch}_all_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)

        # Aggregate per-pdb metrics
        sample_metrics = defaultdict(list)
        for pdb in pdbs:
            # secondary structure metrics
            for k, v in ss_info[pdb].items():
                sample_metrics[f"{k}"].append(v)

            # MPNN self-consistency metrics
            for k, v in sc_info[pdb]["sc_metrics"].items():
                # take mean and best across MPNN sequences
                mean_sc_metric = torch.mean(v)
                best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))

                sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_mean"].append(mean_sc_metric.item())
                sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_best"].append(best_sc_metric.item())

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
                sample_metrics[f"sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

                cluster_out_dir = Path(f"{log_dir_i}/clustering/sctm{sctm_cutoff}")
                sample_metrics[f"sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(designable_pdbs, cluster_out_dir, f"{log_dir_i}/tmp",
                                                                                              **cfg.clustering.foldseek_opts)

        # === Calculate mean metrics === #
        metrics = {f"bb_gen/S{cfg.num_steps}/{k}": np.mean(v) for k, v in sample_metrics.items()}

        # Log metrics to wandb
        if not cfg.no_wandb:
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
