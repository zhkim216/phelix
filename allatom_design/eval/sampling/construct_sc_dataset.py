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
import yaml
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (load_feats_from_pdb, pad_to_max_len,
                                      process_single_pdb)
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="construct_sc_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for all PDBs in a directory.
    For each temperature, we produce num_seqs_per_pdb sequences per PDB, then run self-consistency evaluation.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directories
    out_dir = cfg.out_dir
    sample_out_dir = f"{out_dir}/samples"
    fasta_out_dir = f"{out_dir}/fastas"
    sample_pkl_dir = f"{out_dir}/sample_pkls"
    pred_out_dir = f"{out_dir}/preds"
    sc_info_dir = f"{out_dir}/sc_info"

    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(fasta_out_dir).mkdir(parents=True, exist_ok=True)
    Path(sample_pkl_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
    Path(sc_info_dir).mkdir(parents=True, exist_ok=True)


    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Device setup
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Load structure prediction model for self-consistency eval
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)
    self_consistency_path = f"{out_dir}/self_consistency_metrics.csv"

    # Setup sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,
        "noise_schedule": noise_schedule,
        "churn_cfg": churn_cfg,
        "return_scn_diffusion_aux": False
    }

    # Read in PDB files
    if cfg.pdb_key_list is not None:
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*.pdb")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

    # Parallelization
    if cfg.array_id is not None:
        # Determine chunk size
        array_id = cfg.array_id
        num_arrays = cfg.num_arrays
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # If needed, calculate lengths of each PDB
    if cfg.presort_by_length or (cfg.subset_length_range is not None):
        # determine lengths
        results = Parallel(n_jobs=-1)(delayed(get_length)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        if cfg.subset_length_range is not None:
            # filter by length
            min_len, max_len = cfg.subset_length_range
            pdb_files = [f for f in pdb_files if min_len <= pdb_to_length[f] <= max_len]

        if cfg.presort_by_length:
            # sort by length, longest first
            pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)


    ### SAMPLING ###
    print(f"Sampling with num denoising steps S={cfg.num_steps} on {len(pdb_files)} PDBs (array_id={cfg.array_id})")

    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    sc_metrics = defaultdict(list)
    for temperature, num_seqs_per_pdb in zip(cfg.temperature_list, cfg.num_seqs_per_pdb_list):
        # Process PDBs in batches of size B
        pdb_files_repeated = np.repeat(pdb_files, num_seqs_per_pdb)

        pbar = tqdm(total=len(pdb_files_repeated))
        for i in range(0, len(pdb_files_repeated), cfg.batch_size):
            pdb_batch_files = pdb_files_repeated[i:i+cfg.batch_size]
            B = len(pdb_batch_files)

            # Load and process all PDBs in this batch
            batch_list = []
            for pdb_file in pdb_batch_files:
                data = load_feats_from_pdb(pdb_file)
                single = process_single_pdb(data)
                batch_list.append(single)

            pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

            # Create a batch dictionary from batch_list by stacking
            model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
            max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
            batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
            batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

            # Move to device
            batch = {k: batch[k].to(device) for k in model_input_keys}

            # Prepare scd_inputs for this batch
            scd_inputs = dict(scd_inputs_template)
            scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

            # Prepare sampling timesteps
            timesteps = t_seq[None].expand(B, -1).to(device)

            cond_labels_in = {
                "crop_aug": torch.Tensor([cl.DEFAULT_TOKEN_ID['crop_aug']]*B).to(device),
                "dataset_source": torch.Tensor([cl.DEFAULT_TOKEN_ID['dataset_source']]*B).to(device),
            }

            # Run sampling
            x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
                batch["x"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                missing_atom_mask=batch["missing_atom_mask"],
                residue_index=batch["residue_index"],
                chain_index=batch["chain_index"],
                cond_labels=cond_labels_in,
                timesteps=timesteps,
                aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                seq_only=cfg.seq_only,
                temperature=temperature,
                repack_last=cfg.repack_last,
                psce_threshold=cfg.psce_threshold,
                noise_labels=cfg.noise_labels,
                aatype_override_mask=None,  # no conditioning
                scn_override_mask=None,  # no conditioning
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

            # Save outputs
            # Save to PDB
            pdb_keys = [f"{pdb_name}_sample{(i+j) % num_seqs_per_pdb}_T{temperature}" for j, pdb_name in enumerate(pdb_names)]
            pdbs = [f"{sample_out_dir}/{pdb_key}.pdb" for pdb_key in pdb_keys]
            fastas = [f"{fasta_out_dir}/{pdb_key}.fasta" for pdb_key in pdb_keys]
            pred_seqs = []
            SeqDenoiser.save_samples_to_pdb(samples, pdbs)

            for j, pdb_file in enumerate(pdb_batch_files):
                # Extract the sequence
                seq_mask_i = samples["seq_mask"][j].cpu()
                pred_aatype_i = samples["pred_aatype"][j].cpu()
                pred_aatype_i = pred_aatype_i[seq_mask_i.bool()]
                pred_seq_i = "".join(rc.restypes_with_x[a] for a in pred_aatype_i)
                pred_seqs.append(pred_seq_i)

                # Save fasta
                fasta_out = fastas[j]
                with open(fasta_out, "w") as f:
                    f.write(f">{pdb_keys[j]}\n{pred_seq_i}\n")

            # Save samples as pkl
            for j in range(B):
                sample_j = {k: v[j].cpu().numpy() for k, v in samples.items()}
                # crop to the actual sequence length
                seq_mask_j = sample_j["seq_mask"]
                sample_j = {k: v[seq_mask_j.astype(bool)] for k, v in sample_j.items()}
                with open(f"{sample_pkl_dir}/{pdb_keys[j]}.pkl", "wb") as f:
                    pickle.dump(sample_j, f)

            # Run self-consistency evaluation
            sc_info = eval_metrics.run_self_consistency_eval(
                pdbs,
                None, None,  # no MPNN model for co-design eval
                struct_pred_model,
                device,
                out_dir=pred_out_dir,
                eval_codesign=True,
                temp_dir=f"{pred_out_dir}/tmp",
                override_metrics_to_compute=["sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm"]
            )

            # Dump self-consistency info to pickle
            for k, v in sc_info.items():
                sample_name = Path(k).stem
                with open(f"{sc_info_dir}/{sample_name}.pkl", "wb") as f:
                    pickle.dump(v, f)

            # Aggregate results
            for j, pdb in enumerate(pdbs):
                sc_metrics["pdb_name"].append(Path(pdb).stem)
                sc_metrics["pdb_key"].append(pdb_names[j])
                sc_metrics["temperature"].append(temperature)
                sc_metrics["pred_seq"].append(pred_seqs[j])
                for k, v in sc_info[pdb]["sc_metrics"].items():
                    sc_metrics[f"{k}"].append(v.item())
                sc_metrics["avg_plddt"].append(sc_info[pdb]["struct_preds"]["avg_plddt"].item())

            pbar.update(B)

    # Safely append to CSV using a file lock
    out_df = pd.DataFrame(sc_metrics)
    with open(self_consistency_path, "a+") as f:
        # Acquire exclusive lock
        fcntl.flock(f, fcntl.LOCK_EX)

        # Check if file is empty
        f.seek(0, os.SEEK_END)
        file_empty = (f.tell() == 0)

        # Write DataFrame
        out_df.to_csv(f, index=False, header=file_empty)

        # Release lock
        fcntl.flock(f, fcntl.LOCK_UN)

    pbar.close()


def get_length(pdb_file: str) -> Tuple[str, int]:
    data = load_feats_from_pdb(pdb_file)
    return pdb_file, len(data["aatype"])


if __name__ == "__main__":
    main()
