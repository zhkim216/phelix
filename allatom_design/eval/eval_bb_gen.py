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
from transformers import AutoTokenizer, EsmForProteinFolding

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_bb_gen", version_base="1.3.2")
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
    sampled_pdbs_dir = f"{log_dir}/sampled_pdbs"
    Path(sampled_pdbs_dir).mkdir(parents=True, exist_ok=True)
    saved_metrics_dir = f"{log_dir}/metrics"
    Path(saved_metrics_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Load in MPNN + ESMFold for co-design self-consistency evals
    mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
    mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)  # override base mpnn config with mpnn.overrides
    mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
    esmfold.esm = esmfold.esm.half()
    esmfold = esmfold.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"ld-epoch\d+\.ckpt$")  # only consider ckpts of form ld-epochXXXX.ckpt
    ad_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    pbar = tqdm(ad_ckpts, desc="Evaluating checkpoints")
    for ad_ckpt in pbar:
        # Skip if epoch is before start_epoch
        epoch = int(Path(ad_ckpt).stem.replace("ld-epoch", ""))
        pbar.set_postfix_str(f"Epoch: {epoch}")
        if (cfg.start_epoch is not None) and (epoch < cfg.start_epoch):
            continue

        # Load denoiser model and dataset
        lit_ad_model = LitAtomDenoiser.load_from_checkpoint(ad_ckpt).eval()
        with open_dict(lit_ad_model.cfg.data):
            lit_ad_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

        ### BEGIN EVAL ###
        # Define the range of lengths to sample
        start, end = cfg.length_range
        lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
        all_lengths = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample

        # Sample backbones
        for S in cfg.num_steps_list:
            pdbs = []

            for i in range(0, len(all_lengths), cfg.batch_size):
                # Choose lengths and residue index
                lengths = torch.tensor(all_lengths[i:i + cfg.batch_size], dtype=torch.long).to(lit_ad_model.device)
                B = lengths.shape[0]
                residue_index = torch.arange(lengths.max(), dtype=torch.long).to(lit_ad_model.device)
                residue_index = residue_index[None].expand(B, -1)

                # Create timesteps, separating timesteps for CA and NCO
                cfg.timestep_schedule.num_steps = S  # set num_steps for this iteration
                t_ca = sampling_utils.get_timestep_schedule(**cfg.timestep_schedule.ca)
                t_ca = t_ca[None].expand(B, -1).to(lit_ad_model.device)
                t_nco = sampling_utils.get_timestep_schedule(**cfg.timestep_schedule.nco)
                t_nco = t_nco[None].expand(B, -1).to(lit_ad_model.device)
                timesteps = (t_ca, t_nco)

                # Create noise schedules for CA and NCO
                noise_schedule = (NoiseSchedule(cfg.ca_diffusion.noise_schedule),
                                  NoiseSchedule(cfg.nco_diffusion.noise_schedule))

                # Create churn configs for CA and NCO
                churn_cfg = (dict(cfg.ca_diffusion.churn_cfg), dict(cfg.nco_diffusion.churn_cfg))

                cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, lit_ad_model.device)
                x_bb_denoised, aux = lit_ad_model.model.sample(lengths,
                                                            residue_index=residue_index,
                                                            timesteps=timesteps,
                                                            cond_labels=cond_labels_in,
                                                            noise_schedule=noise_schedule,
                                                            churn_cfg=churn_cfg)

                samples = {"x_bb_denoised": x_bb_denoised,
                            "seq_mask": aux["seq_mask"],
                            "residue_index": residue_index}
                samples = {k: v.cpu() if v is not None else v for k, v  in samples.items()}

                # Save samples
                filenames = [f"{sampled_pdbs_dir}/epoch_{epoch}_S{S}_sample_{i+j}_len_{l.item()}.pdb" for j, l in enumerate(lengths)]
                AtomDenoiser.save_samples_to_pdb(samples, filenames)
                pdbs.extend(filenames)

            ### CALCULATE STRUCTURE METRICS ###
            all_metrics = defaultdict(dict)
            pdbs = natsorted(pdbs)

            # Get secondary structure info
            ss_info = eval_metrics.compute_secondary_structure_content(pdbs)
            for pdb, v in ss_info.items():
                all_metrics[pdb]["ss_info"] = v

            # Run MPNN + ESMFold self-consistency evals
            mpnn_sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                                  mpnn_model, mpnn_cfg,
                                                                  esmfold, tokenizer,
                                                                  device,
                                                                  out_dir=log_dir)
            for pdb, v in mpnn_sc_info.items():
                all_metrics[pdb]["mpnn_sc_info"] = v

            # Run nnTM evaluation
            if cfg.nntm_dataset is not None:
                nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=cfg.out_dir)

                for pdb, v in nntm_info.items():
                    all_metrics[pdb]["nntm_info"] = v

            ### SAVE METRICS ###
            # Save all metrics to pickle file
            with open(f"{saved_metrics_dir}/epoch_{epoch}_S{S}_all_metrics.pkl", "wb") as f:
                pickle.dump(all_metrics, f)

            # Aggregate metrics to log
            sample_metrics = defaultdict(list)
            for pdb in pdbs:
                # secondary structure metrics
                for k, v in ss_info[pdb].items():
                    sample_metrics[f"{k}"].append(v)

                # MPNN self-consistency metrics
                for k, v in mpnn_sc_info[pdb]["sc_metrics"].items():
                    # take mean and best across MPNN sequences
                    mean_sc_metric = torch.mean(v)
                    best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                    sample_metrics[f"mpnn_{k}_mean"].append(mean_sc_metric.item())
                    sample_metrics[f"mpnn_{k}_best"].append(best_sc_metric.item())

                # nntm metrics
                if cfg.nntm_dataset is not None:
                    sample_metrics["nntm"].append(nntm_info[pdb])

            metrics = {f"bb_gen/S{S}/{k}": np.mean(v) for k, v in sample_metrics.items()}

            # Log metrics to wandb
            if not cfg.no_wandb:
                # Get global step
                global_step = torch.load(ad_ckpt, map_location="cpu")["global_step"]
                metrics["trainer/global_step"] = global_step

                wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
