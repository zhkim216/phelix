import fcntl
import glob
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (get_length_from_pdb, load_feats_from_pdb,
                                      pad_to_max_len)
from allatom_design.data.datasets.sd_dataset import process_single_pdb
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_seq_des_scaffolding", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating self-consistency on a set of PDBs given:
    1. backbone-only
    2. sequence of motif
    3. sequence + sidechain of motif
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directories
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.checkpoint_path).parent.parent
        model_name = Path(cfg.checkpoint_path).stem
        cfg.out_dir = f"{model_run_dir}/eval_sc_dpo/{model_name}/{cfg.exp_name}"

    out_dir = cfg.out_dir  # base output directory
    sample_out_dir = f"{out_dir}/samples"
    fasta_out_dir = f"{out_dir}/fastas"  # directory for sequences in FASTA format
    sample_pkl_dir = f"{out_dir}/sample_pkls"  # directory for pkls containing various sample info

    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(fasta_out_dir).mkdir(parents=True, exist_ok=True)
    Path(sample_pkl_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Device setup
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Load in structure prediction model
    pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions (if running folding)
    Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
    self_consistency_path = f"{out_dir}/self_consistency_metrics.csv"

    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Setup sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,  # will be filled per batch
        "noise_schedule": noise_schedule,
        "churn_cfg": churn_cfg,
        "return_scn_diffusion_aux": False
    }

    ### Load in PDB files ###
    if cfg.pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        # Get all PDBs with .pdb extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*.pdb")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

    # If specified, pre-sort by length (descending)
    if (cfg.presort_by_length) or (cfg.subset_length_range is not None):
        results = Parallel(n_jobs=-1)(
            delayed(get_length_from_pdb)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths")
        )
        pdb_to_length = dict(results)

        if cfg.subset_length_range is not None:
            min_len, max_len = cfg.subset_length_range
            pdb_files = [f for f in pdb_files if min_len <= pdb_to_length[f] <= max_len]

        if cfg.presort_by_length:
            # sort by length, longest first
            pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)

    if cfg.num_pdbs is not None:
        # Use isolated random state for consistent subsampling
        rng = np.random.RandomState(cfg.seed)

        # subsample, ensuring order is preserved
        chosen_indices = sorted(rng.choice(len(pdb_files), cfg.num_pdbs, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    ### SAMPLING ###
    print(f"Sampling with num denoising steps S={cfg.num_steps} on {len(pdb_files)} PDBs")

    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Accumulate self-consistency metrics across batches
    sc_metrics_dfs = []

    # Process PDBs in batches
    pdb_files_repeated = np.repeat(pdb_files, cfg.num_seqs_per_pdb)
    pbar = tqdm(total=len(pdb_files_repeated))

    for i in range(0, len(pdb_files_repeated), cfg.batch_size):
        pdb_batch_files = pdb_files_repeated[i : i + cfg.batch_size]
        B = len(pdb_batch_files)

        # Load and process all PDBs in this batch
        batch_list = []
        batch_chain_id_mapping = []
        for pdb_file in pdb_batch_files:
            data = load_feats_from_pdb(pdb_file)
            single = process_single_pdb(data)
            batch_list.append(single)

            # store chain ID mapping if needed (not used in this snippet)
            batch_chain_id_mapping.append(data["chain_id_mapping"])

        pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

        # Create a batch dictionary from batch_list by stacking
        model_input_keys = [
            "x", "aatype", "seq_mask", "missing_atom_mask",
            "residue_index", "chain_index", "interface_residue_mask"
        ]
        max_len = max(b["x"].shape[0] for b in batch_list)
        batch_list = [
            pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)
            for b in batch_list
        ]
        batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}
        batch = {k: batch[k].to(device) for k in model_input_keys}

        # Prepare scd_inputs for this batch
        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

        # Prepare sampling timesteps
        timesteps = t_seq[None].expand(B, -1).to(device)

        # Run sampling
        x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
            batch["x"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            cond_labels={},  # TODO: remove cond labels
            timesteps=timesteps,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            seq_only=cfg.seq_only,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            noise_labels=cfg.noise_labels,
            aatype_override_mask=None,
            scn_override_mask=None,
            scd_inputs=scd_inputs,
        )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
            # save other useful info
            "original_aatype": batch["aatype"],
            "interface_residue_mask": batch["interface_residue_mask"],
        }

        # Save outputs: PDB, FASTA, PKL
        pdb_keys = [f"{pdb_name}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
        pdbs = [f"{sample_out_dir}/{pdb_key}.pdb" for pdb_key in pdb_keys]
        fastas = [f"{fasta_out_dir}/{pdb_key}.fasta" for pdb_key in pdb_keys]
        pred_seqs = []

        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        for j, pdb_file in enumerate(pdb_batch_files):
            seq_mask_i = samples["seq_mask"][j].cpu()
            pred_aatype_i = samples["pred_aatype"][j].cpu()
            pred_aatype_i = pred_aatype_i[seq_mask_i.bool()]
            pred_seq_i = "".join(rc.restypes_with_x[a] for a in pred_aatype_i)
            pred_seqs.append(pred_seq_i)

            with open(fastas[j], "w") as f:
                f.write(f">{pdb_keys[j]}\n{pred_seq_i}\n")

        # Save samples as PKL
        for j in range(B):
            sample_j = {k: v[j].cpu().numpy() for k, v in samples.items()}
            seq_mask_j = sample_j["seq_mask"]
            sample_j = {k: v[seq_mask_j.astype(bool)] for k, v in sample_j.items()}
            with open(f"{sample_pkl_dir}/{pdb_keys[j]}.pkl", "wb") as f:
                pickle.dump(sample_j, f)

        # Run self-consistency evaluations with ESMFold and AF2
        sc_info = eval_metrics.run_self_consistency_eval(
            pdbs,
            None,  # no MPNN model to use sequence from PDB
            struct_pred_model,
            device,
            out_dir=pred_out_dir,
            temp_dir=f"{pred_out_dir}/tmp",
            metrics_to_compute=["sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm"]
        )


        # Aggregate results
        sc_metrics = defaultdict(list)
        for j, pdb_path in enumerate(pdbs):
            sc_metrics["pdb_name"].append(Path(pdb_path).stem)
            sc_metrics["pdb_key"].append(pdb_names[j])
            sc_metrics["pred_seq"].append(pred_seqs[j])

            for k, v in sc_info[pdb_path]["sc_metrics"].items():
                sc_metrics[k].append(v.item())

            # Add plddt info
            sc_metrics["avg_ca_plddt"].append(sc_info[pdb_path]["struct_preds"]["avg_ca_plddt"].item())

        out_df_batch = pd.DataFrame(sc_metrics)

        sc_metrics_dfs.append(out_df_batch)

        pbar.update(B)

    pbar.close()

    # After the entire loop, concatenate all batch-level DataFrames and write once
    out_sc_df = pd.concat(sc_metrics_dfs, ignore_index=True)
    out_sc_df.to_csv(self_consistency_path, index=False)

    # Get mean and median of self-consistency metrics


    # Wandb logging
    if not cfg.no_wandb:
        wandb_dir = str(Path(cfg.out_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

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

        metrics = {}
        for model in ["esmfold", "af2"]:
            for sc_metric in ["sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm", "avg_ca_plddt"]:
                metrics[f"{model}/{sc_metric}_mean"] = out_sc_df[f"{model}_{sc_metric}"].mean()
                metrics[f"{model}/{sc_metric}_median"] = out_sc_df[f"{model}_{sc_metric}"].median()

        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
