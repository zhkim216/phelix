import glob
import os
import pickle
import re
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import (load_feats_from_pdb, pad_to_max_len,
                                      process_single_pdb)
from allatom_design.data.datasets.ad_dataset import get_scaffold_manager
from allatom_design.data.pdb_utils import *
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.eval.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.data.datasets.scaffold_manager import get_scaffold_manager


@hydra.main(config_path="../../configs/eval/sampling", config_name="bb_scaffold", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Load atom denoiser
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ad_ckpt).eval()
    device = lit_ad_model.device

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.ad_ckpt).parent.parent
        model_name = Path(cfg.ad_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/draw_samples/{model_name}/{cfg.exp_name}"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "samples")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Load PDBs
    if cfg.pdb_key_list is not None:
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdbs = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        pdbs = natsorted(list(Path(cfg.pdb_dir).glob(f"*{cfg.pdb_key_ext}")))
        pdbs = [str(x) for x in pdbs]

    if len(pdbs) == 0:
        raise ValueError(f"No PDB files found under {cfg.pdb_dir} with extension {cfg.pdb_key_ext}")

    print(f"Loaded {len(pdbs)} PDB files.")

    # Define scaffold manager
    sm = get_scaffold_manager(lit_ad_model.cfg.scaffold_manager)

    # Timesteps, noise schedule, churn config
    t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.noise_schedule)
    churn_cfg = dict(cfg.churn_cfg)

    # Process in batches
    pbar = tqdm(total=len(pdbs))
    for i in range(0, len(pdbs), cfg.batch_size):
        pdb_batch_files = pdbs[i : i + cfg.batch_size]
        B = len(pdb_batch_files)

        # Load and collate the data
        batch_list = []
        for pdb_file in pdb_batch_files:
            data = load_feats_from_pdb(pdb_file)
            single = process_single_pdb(data, sm)
            batch_list.append(single)

        pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

        # Create a batch dictionary from batch_list by stacking
        model_input_keys = ["x", "seq_mask", "atom_mask", "missing_atom_mask", "residue_index", "x_scaffold", "scaffold_mask", "aatype_scaffold"]
        max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
        batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
        batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

        # Move to device
        batch = {k: batch[k].to(device) for k in model_input_keys}

        # Save scaffolding motifs
        scaffold_res_mask = batch["scaffold_mask"].any(dim=-1)  # shape [B, N]

        motif_samples = {"aatype": batch["aatype_scaffold"],
                         "atom_positions": batch["x_scaffold"],
                         "atom_mask": batch["scaffold_mask"],
                         "residue_index": batch["residue_index"],
                         "chain_index": torch.zeros_like(batch["residue_index"]),
                         "b_factors": torch.ones_like(batch["scaffold_mask"], dtype=torch.float32)
                         }
        feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in motif_samples.items()}  # move to cpu
        motif_filenames = [f"{sample_out_dir}/motif_{pdb_stem}.pdb" for pdb_stem in pdb_names]
        write_batched_to_pdb(**feats, filenames=motif_filenames, mode="aa")

        # Save centered examples
        samples = {
            "x_bb_denoised": batch["x"][..., rc.bb_idxs, :].cpu(),
            "seq_mask": batch["seq_mask"].cpu(),
            "residue_index": batch["residue_index"].cpu(),
        }
        centered_filenames = [f"{sample_out_dir}/centered_{pdb_stem}.pdb" for pdb_stem in pdb_names]
        AtomDenoiser.save_samples_to_pdb(samples, centered_filenames)

        # Build scaffold inputs
        scaffold_inputs = {
            "x_scaffold": batch["x_scaffold"],
            "scaffold_mask": batch["scaffold_mask"],
            "aatype_scaffold": batch["aatype_scaffold"],
        }

        # Build timesteps for sampling
        timesteps = t_bb[None].expand(B, -1).to(device)

        # cond_labels_in if needed
        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)  # or None if unconditional

        # Sample
        x_bb_denoised, aux = lit_ad_model.model.sample(
            lengths=batch["seq_mask"].sum(dim=-1),  # not used for motif scaffolding, the model can parse shapes
            residue_index=batch["residue_index"].to(device),
            timesteps=timesteps,
            cond_labels=cond_labels_in,
            noise_schedule=noise_schedule,
            churn_cfg=churn_cfg,
            autoguidance_cfg=dict(cfg.autoguidance_cfg),
            scaffold_inputs=scaffold_inputs,
        )

        # Move to CPU
        samples = {
            "x_bb_denoised": x_bb_denoised.cpu(),
            "seq_mask": batch["seq_mask"].cpu(),
            "residue_index": batch["residue_index"].cpu(),
        }

        # Save final sampled structures
        out_filenames = []
        for j, pdb_file in enumerate(pdb_batch_files):
            pdb_stem = Path(pdb_file).stem
            out_filenames.append(f"{sample_out_dir}/sample_{pdb_stem}_{i + j}.pdb")

        AtomDenoiser.save_samples_to_pdb(samples, out_filenames)

        pbar.update(B)

    pbar.close()

    del lit_ad_model  # free up memory

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)
    pdbs = natsorted(glob.glob(f"{sample_out_dir}/*.pdb"))

    # === Get lengths and bins of sampled structures === #
    lengths = [int(re.search(r"len(\d+)", pdb).groups()[-1]) for pdb in pdbs]
    bins = [int(length / cfg.length_bin_size) * cfg.length_bin_size for length in lengths]  # bins are defined by their starting length
    for pdb, length, bin in zip(pdbs, lengths, bins):
        all_metrics[pdb]["length"] = length
        all_metrics[pdb]["bin"] = bin

    # === Load MPNN and structure prediction models === #
    if cfg.sc.run_mpnn_sc:
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)
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
        mpnn_sc_info = eval_metrics.run_self_consistency_eval(sc_pdbs,
                                                              mpnn_model, mpnn_cfg,
                                                              struct_pred_model,
                                                              device,
                                                              out_dir=cfg.out_dir,
                                                              temp_dir=f"{cfg.out_dir}/tmp")
        for pdb, v in mpnn_sc_info.items():
            all_metrics[pdb]["mpnn_sc_info"] = v


    # === Run nnTM evaluation === #
    if cfg.nntm_dataset is not None:
        nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=cfg.out_dir)
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
            if "mpnn_sc_info" in all_metrics[pdb]:
                mpnn_sc_info = all_metrics[pdb]["mpnn_sc_info"]
                for k, v in mpnn_sc_info["sc_metrics"].items():
                    # take mean and best across MPNN sequences
                    best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                    metrics_b[f"mpnn_{k}_best"].append(best_sc_metric.item())

                    if len(v) > 1:
                        # only log mean if there are multiple MPNN sequences per backbone
                        mean_sc_metric = torch.mean(v)
                        metrics_b[f"mpnn_{k}_mean"].append(mean_sc_metric.item())

            # nnTM metrics
            if "nntm_info" in all_metrics[pdb]:
                metrics_b["nntm"].append(all_metrics[pdb]["nntm_info"])

        # Average metrics across samples
        metrics_b_avg = {}
        for k, v in metrics_b.items():
            metrics_b_avg[k] = np.mean(v)

            # For MPNN metrics, get SE from bootstrapping
            if "mpnn" in k:
                metrics_b_avg[f"{k}_se"] = eval_metrics.bootstrap_se(v, n_samples=1000)

        bin_to_metrics[bin] = metrics_b_avg

    # === Calculate mean pairwise TM score by length === #
    for bin in set(bins):
        pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if b == bin]
        coords_b = [load_feats_from_pdb(pdb, chain_residx_gap=None)["all_atom_positions"] for pdb in pdbs_b]
        bin_to_metrics[bin]["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(coords_b,
                                                                                    temp_dir=f"{cfg.out_dir}/tmp",
                                                                                    subsample_pairs=cfg.pairwise_tm_subsample)

    # === Run clustering analysis === #
    for sctm_cutoff in cfg.clustering.sctm_cutoffs:
        # Cluster by length bin
        for bin in set(bins):
            pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if (b == bin) and ("mpnn_sc_info" in all_metrics[pdb])]
            if len(pdbs_b) == 0:
                # skip if we don't have self-consistency info for any samples in this bin
                continue

            # Cluster only on designable samples (scTM > sctm_cutoff)
            designable_pdbs = [pdb for pdb in pdbs_b if all_metrics[pdb]["mpnn_sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff]
            bin_to_metrics[bin][f"sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

            cluster_out_dir = Path(f"{cfg.out_dir}/clustering/bin{bin}_sctm{sctm_cutoff}")
            bin_to_metrics[bin][f"sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(designable_pdbs, cluster_out_dir, f"{cfg.out_dir}/tmp",
                                                                                                **cfg.clustering.foldseek_opts)


    # === Compute KL(p||q) for secondary structure distributions === #
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
    with open(f"{cfg.out_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    with open(f"{cfg.out_dir}/L_to_metrics.pkl", "wb") as f:
        pickle.dump(bin_to_metrics, f)

    # Set up wandb logging
    if not cfg.no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(cfg.out_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
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
