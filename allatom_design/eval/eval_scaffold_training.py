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
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import (get_length_from_pdb, load_feats_from_pdb,
                                      pad_to_max_len)
from allatom_design.data.datasets.scaffold_manager import get_scaffold_manager
from allatom_design.data.datasets.sd_dataset import process_single_pdb
from allatom_design.data.pdb_utils import *
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.fampnn_utils import get_seq_des_model
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_scaffold_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating scaffold-based generation.
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        log_dir = Path(cfg.out_dir, wandb.run.name)

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
        print(f"Using EMA checkpoints from {ema_ckpt_dir}")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)-ema(\d+\.\d+)\.ckpt$")
        ad_ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]
    else:
        print(f"Using non-EMA checkpoints from {cfg.denoiser_train_dir}/checkpoints")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)\.ckpt$")
        ad_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    # Gather PDBs to sample; apply length filtering / subsampling
    if cfg.pdb_key_list is not None:
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        pdb_files = natsorted(list(Path(cfg.pdb_dir).glob(f"*{cfg.pdb_key_ext}")))
        pdb_files = [str(x) for x in pdb_files]

    print(f"Found {len(pdb_files)} PDB(s) to scaffold from in {cfg.pdb_dir}")

    if cfg.subset_length_range is not None:
        min_length, max_length = cfg.subset_length_range
        results = Parallel(n_jobs=-1)(delayed(get_length_from_pdb)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        # Filter PDBs based on length
        pdb_files = [pdb for pdb in pdb_files if min_length <= pdb_to_length[pdb] <= max_length]
        print(f"Filtered to {len(pdb_files)} PDB(s) within length range {min_length} to {max_length}")

    if cfg.n_subsample is not None:
        # Randomly subsample PDBs
        np.random.seed(cfg.seed)
        np.random.shuffle(pdb_files)
        pdb_files = pdb_files[:cfg.n_subsample]

    # Sample from each checkpoint
    pbar = tqdm(ad_ckpts, desc=f"Sampling on {len(pdb_files)} PDB(s) with {len(ad_ckpts)} checkpoint(s)...")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        if (cfg.start_step is not None) and (global_step < cfg.start_step):
            continue

        # Load denoiser model
        lit_ad_model = LitAtomDenoiser.load_from_checkpoint(ad_ckpt).eval().to(device)

        # Create scaffold manager
        sm = get_scaffold_manager(lit_ad_model.cfg.scaffold_manager).eval()

        # Prepare sampling schedule
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
        noise_schedule = NoiseSchedule(cfg.noise_schedule)
        churn_cfg = dict(cfg.churn_cfg)

        # Evaluate separately for each scaffold conditioning type
        for scaffold_conditioning_type in cfg.scaffold_conditioning_types:
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
            L.seed_everything(cfg.seed)  # reset seed for each checkpoint and conditioning type
            sampled_pdbs = []
            motif_info = {}  # map from pdb path to motif mask and coordinates

            sm.set_conditioning_type(scaffold_conditioning_type)  # set the conditioning type for the scaffold manager
            for i in range(0, len(pdb_files), cfg.batch_size):
                pdb_batch = pdb_files[i : i + cfg.batch_size]
                B = len(pdb_batch)

                # Load/prepare data
                batch_list = []
                pdb_names = []
                for pdb_path in pdb_batch:
                    data = load_feats_from_pdb(pdb_path)
                    single = process_single_pdb(data, sm)
                    batch_list.append(single)
                    pdb_names.append(Path(pdb_path).stem)

                # Pad each record, then stack
                max_len = max(b["x"].shape[0] for b in batch_list)
                model_input_keys = ["x", "seq_mask", "atom_mask", "missing_atom_mask", "residue_index",
                                    "x_motif", "motif_mask", "aatype_scaffold"]
                batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len) for b in batch_list]
                batch_dict = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}

                # Move to device
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

                # Save motifs
                motif_samples = {
                    "aatype": batch_dict["aatype_scaffold"],
                    "atom_positions": batch_dict["x_motif"],
                    "atom_mask": batch_dict["motif_mask"],
                    "residue_index": batch_dict["residue_index"],
                    "chain_index": torch.zeros_like(batch_dict["residue_index"]),
                    "b_factors": torch.ones_like(batch_dict["motif_mask"], dtype=torch.float32),
                }
                feats_cpu = {k: v.cpu() for k, v in motif_samples.items()}
                motif_filenames = []
                for j, name in enumerate(pdb_names):
                    motif_filenames.append(f"{motif_pdbs_dir_i}/motif_{name}.pdb")
                write_batched_to_pdb(**feats_cpu, filenames=motif_filenames, mode="aa")

                # Save centered
                samples_centered = {
                    "x_bb_denoised": batch_dict["x"][..., rc.bb_idxs, :].cpu(),  # just backbone coords
                    "seq_mask": batch_dict["seq_mask"].cpu(),
                    "residue_index": batch_dict["residue_index"].cpu(),
                }
                centered_filenames = []
                for j, name in enumerate(pdb_names):
                    centered_filenames.append(f"{centered_gt_pdbs_dir_i}/centered_{name}.pdb")
                AtomDenoiser.save_samples_to_pdb(samples_centered, centered_filenames)

                # Build scaffold inputs
                scaffold_inputs = {
                    "x_motif": batch_dict["x_motif"],
                    "motif_mask": batch_dict["motif_mask"],
                    "aatype_scaffold": batch_dict["aatype_scaffold"],
                }

                # Timesteps
                timesteps = t_bb[None].expand(B, -1).to(device)

                # Conditional labels if needed
                cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device=device)

                # Sample
                x_bb_denoised, aux = lit_ad_model.model.sample(
                    lengths=batch_dict["seq_mask"].sum(dim=-1),  # not actually used for scaffolding
                    residue_index=batch_dict["residue_index"],
                    timesteps=timesteps,
                    cond_labels=cond_labels_in,
                    noise_schedule=noise_schedule,
                    churn_cfg=churn_cfg,
                    autoguidance_cfg=dict(cfg.autoguidance_cfg),
                    scaffold_inputs=scaffold_inputs,
                )

                # Save final structures
                samples_final = {
                    "x_bb_denoised": x_bb_denoised.cpu(),
                    "seq_mask": batch_dict["seq_mask"].cpu(),
                    "residue_index": batch_dict["residue_index"].cpu(),
                }
                out_filenames = []
                for j, name in enumerate(pdb_names):
                    pdb_path = f"{sampled_pdbs_dir_i}/sample_{name}_{i+j}.pdb"
                    out_filenames.append(pdb_path)

                    # add motif info
                    motif_info[pdb_path] = {"motif_mask": batch_dict["motif_mask"][j].cpu(),
                                            "x_motif": batch_dict["x_motif"][j].cpu()}
                    length_j = batch_dict["seq_mask"][j].sum().long().item()
                    motif_info[pdb_path] = {k: v[:length_j].clone() for k, v in motif_info[pdb_path].items()}  # get rid of padding

                AtomDenoiser.save_samples_to_pdb(samples_final, out_filenames)
                sampled_pdbs.extend(out_filenames)


            # === CALCULATE STRUCTURE METRICS ===
            all_metrics = defaultdict(dict)
            sampled_pdbs = natsorted(sampled_pdbs)

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
                                                             override_metrics_to_compute=["sc_ca_rmsd", "sc_ca_tm", "motif_bb_rmsd"],
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

            # Save per-sample metrics
            with open(f"{saved_metrics_dir_i}/step_{global_step}_all_metrics.pkl", "wb") as f:
                pickle.dump(all_metrics, f)

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


            if not cfg.no_wandb:
                metrics["trainer/global_step"] = global_step
                metrics["trainer/epoch"] = epoch
                wandb.log(metrics)

        del lit_ad_model

    if not cfg.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
