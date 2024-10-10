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

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.allatom_denoiser.allatom_model import AllAtomModel
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="draw_allatom_samples", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Construct allatom model
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ad_ckpt).eval()
    device = lit_ad_model.device
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.sd_ckpt).eval()
    allatom_model = AllAtomModel(lit_ad_model, lit_sd_model)

    # Create out dirs in atom denoiser directory and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.ad_ckpt).parent.parent
        model_name = Path(cfg.ad_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/draw_allatom_samples/{model_name}/{cfg.exp_name}"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "final_samples")
    intermediate_out_dir = Path(cfg.out_dir, "intermediate_samples")
    traj_out_dir = Path(cfg.out_dir, "traj")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(intermediate_out_dir).mkdir(parents=True, exist_ok=True)
    Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    # Define the range of lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
    all_lengths = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample
    save_traj_mask = np.tile(np.arange(cfg.n_samples_per_length) < cfg.n_traj_per_length, len(lengths_to_sample))  # get mask of the trajectories we'll save
    save_traj_steps = np.linspace(0, cfg.joint.num_steps - 1, cfg.limit_traj_steps, dtype=int)  # get the steps of the trajectories we'll save
    print(f"Drawing {cfg.n_samples_per_length} samples each of lengths {start} to {end} with step size {cfg.length_step_size}")

    ### SAMPLE ###
    pbar = tqdm(total=len(all_lengths))

    for i in range(0, len(all_lengths), cfg.batch_size):
        # Choose lengths and residue index
        lengths = torch.tensor(all_lengths[i:i + cfg.batch_size], dtype=torch.long).to(device)
        B = lengths.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long).to(device)
        residue_index = residue_index[None].expand(B, -1)
        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

        # === Handle all-atom model joint timesteps === #
        T_ad = sampling_utils.get_timesteps_from_schedule(**cfg.joint.ad.timestep_schedule).to(device)
        T_sd = sampling_utils.get_timesteps_from_schedule(**cfg.joint.sd.timestep_schedule).to(device)
        timesteps = (T_ad, T_sd)

        # === Handle atom denoiser inputs === #
        # Create timesteps for backbone
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.ad.timestep_schedule)
        t_bb = t_bb[None].expand(B, -1).to(device)

        ad_sampling_inputs = {
            "timesteps": t_bb,
            "noise_schedule": NoiseSchedule(cfg.ad.noise_schedule),
            "churn_cfg": dict(cfg.ad.churn_cfg),
            "autoguidance_cfg": dict(cfg.ad.autoguidance_cfg),
        }

        # === Handle sequence denoiser inputs === #
        sd_sampling_inputs = {}

        # Define sequence denoising timesteps
        t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.sd.seq.timestep_schedule)
        t_seq = t_seq[None].expand(B, -1).to(device)
        sd_sampling_inputs["timesteps"] = t_seq
        sd_sampling_inputs["aatype_decoding_order_mode"] = cfg.sd.seq.aatype_decoding_order_mode

        # Set up sidechain diffusion inputs
        t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.sd.scd.timestep_schedule)
        t_scd = t_scd[None].expand(B, -1).to(device)
        noise_schedule = NoiseSchedule(cfg.sd.scd.noise_schedule)
        churn_cfg = dict(cfg.sd.scd.churn_cfg)
        scd_inputs = {"num_steps": cfg.sd.scd.num_steps,
                      "timesteps": t_scd,
                      "noise_schedule": noise_schedule,
                      "churn_cfg": churn_cfg,
                      "autoguidance_cfg": dict(cfg.sd.scd.autoguidance_cfg),
                      "return_scn_diffusion_aux": False}
        sd_sampling_inputs["scd_inputs"] = scd_inputs

        # === Sample from allatom model === #
        x_denoised, aatype_denoised, aux = allatom_model.sample(lengths=lengths,
                                                                residue_index=residue_index,
                                                                timesteps=timesteps,
                                                                ad_sampling_inputs=ad_sampling_inputs,
                                                                sd_sampling_inputs=sd_sampling_inputs,
                                                                cond_labels=cond_labels_in)


        samples = {"x_denoised": x_denoised,
                   "aatype_denoised": aatype_denoised,
                   "seq_mask": aux["seq_mask"],
                   "residue_index": residue_index}
        samples = {k: v.cpu() if v is not None else v for k, v  in samples.items()}

        # Save samples
        filenames = [f"{sample_out_dir}/sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)]
        AllAtomModel.save_samples_to_pdb(samples, filenames)

        # Save samples for each intermediate step of the trajectory
        for step in range(cfg.joint.num_steps - 1):
            samples_step = {"x_denoised": aux["xt_traj"][:, step],
                         "aatype_denoised": aux["aatype_pred_traj"][:, step],
                         "seq_mask": aux["seq_mask"],
                         "residue_index": residue_index}
            samples_step = {k: v.cpu() if v is not None else v for k, v in samples_step.items()}
            T_ad, T_sd = timesteps[0][step], timesteps[1][step]
            step_suffix = f"Tad{T_ad:.2f}_Tsd{T_sd:.2f}"
            filenames = [f"{intermediate_out_dir}/sample_len{lengths[j]}_{i + j}_{step_suffix}.pdb" for j in range(B)]
            AllAtomModel.save_samples_to_pdb(samples_step, filenames)

        # Write trajectories to file
        align_models_to_idx = None
        if cfg.align_traj_to_last_step:
            # align all predictions along the trajectory to the last step
            align_models_to_idx = cfg.limit_traj_steps - 1

        save_trajs_fn = partial(AllAtomModel.save_trajs_to_pdb, aux, residue_index=residue_index, chain_index=torch.zeros_like(residue_index),
                                save_traj_mask=save_traj_mask, save_traj_steps=save_traj_steps,
                                save_ad_traj_steps=None, save_sd_traj_steps=None, save_scd_traj_steps=None,
                                traj_conect=cfg.traj_conect, align_models_to_idx=align_models_to_idx)
        # save xt traj
        save_trajs_fn(x_traj_key="xt_traj", filenames=[f"{traj_out_dir}/xt_traj_sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)])
        pbar.update(B)

    pbar.close()

    # free up memory; we don't need denoisers anymore
    del lit_ad_model
    del lit_sd_model

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)
    pdbs = natsorted(glob.glob(f"{sample_out_dir}/*.pdb"))

    # === Get lengths and bins of sampled structures === #
    lengths = [int(re.search(r"len(\d+)", pdb).groups()[-1]) for pdb in pdbs]
    bins = [int(length / cfg.length_bin_size) * cfg.length_bin_size for length in lengths]  # bins are defined by their starting length
    for pdb, length, bin in zip(pdbs, lengths, bins):
        all_metrics[pdb]["length"] = length
        all_metrics[pdb]["bin"] = bin

    # === Load structure prediction model === #
    if cfg.sc.run_codes_sc:
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

    if cfg.sc.run_codes_sc:
        codes_sc_info = eval_metrics.run_self_consistency_eval(sc_pdbs,
                                                               None, None,  # no MPNN model for co-design eval
                                                               struct_pred_model,
                                                               device,
                                                               out_dir=cfg.out_dir,
                                                               eval_codesign=True,
                                                               temp_dir=f"{cfg.out_dir}/tmp")

        for pdb, v in codes_sc_info.items():
            all_metrics[pdb]["codes_sc_info"] = v

    if cfg.sc.run_codes_sc_intermediate:
        for step in tqdm(range(cfg.joint.num_steps - 1), desc="Running intermediate step self-consistency evaluations"):
            T_ad, T_sd = timesteps[0][step], timesteps[1][step]
            step_suffix = f"Tad{T_ad:.2f}_Tsd{T_sd:.2f}"
            sc_pdbs_intermediate = [f"{intermediate_out_dir}/{Path(pdb).stem}_{step_suffix}.pdb" for pdb in sc_pdbs]
            codes_sc_info_intermediate = eval_metrics.run_self_consistency_eval(sc_pdbs_intermediate,
                                                                                None, None,  # no MPNN model for co-design eval
                                                                                struct_pred_model,
                                                                                device,
                                                                                out_dir=cfg.out_dir,
                                                                                eval_codesign=True,
                                                                                temp_dir=f"{cfg.out_dir}/tmp")

            for pdb in codes_sc_info:
                pdb_intermediate = f"{intermediate_out_dir}/{Path(pdb).stem}_{step_suffix}.pdb"
                v = codes_sc_info_intermediate[pdb_intermediate]
                all_metrics[pdb][f"{step_suffix}/codes_sc_info"] = v


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

            # Co-design self-consistency metrics
            if "codes_sc_info" in all_metrics[pdb]:
                codes_sc_info = all_metrics[pdb]["codes_sc_info"]
                for k, v in codes_sc_info["sc_metrics"].items():
                    # take best across co-designed sequences
                    # TODO: make it possible to do multiple co-design sequences per backbone / multiple trajectories
                    best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                    metrics_b[f"codes_{k}_best"].append(best_sc_metric.item())

            if cfg.sc.run_codes_sc_intermediate:
                for step in range(cfg.joint.num_steps - 1):
                    T_ad, T_sd = timesteps[0][step], timesteps[1][step]
                    step_suffix = f"Tad{T_ad:.2f}_Tsd{T_sd:.2f}"
                    if f"{step_suffix}/codes_sc_info" in all_metrics[pdb]:
                        codes_sc_info = all_metrics[pdb][f"{step_suffix}/codes_sc_info"]
                        for k, v in codes_sc_info["sc_metrics"].items():
                            best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
                            metrics_b[f"{step_suffix}/codes_{k}_best"].append(best_sc_metric.item())

            # nnTM metrics
            if "nntm_info" in all_metrics[pdb]:
                metrics_b["nntm"].append(all_metrics[pdb]["nntm_info"])

        # Average metrics across samples
        metrics_b_avg = {}
        for k, v in metrics_b.items():
            metrics_b_avg[k] = np.mean(v)

            # For codes metrics, get SE from bootstrapping
            if "codes" in k:
                metrics_b_avg[f"{k}_se"] = eval_metrics.bootstrap_se(v, n_samples=1000)

        bin_to_metrics[bin] = metrics_b_avg

    # === Calculate mean pairwise TM score by length === #
    for bin in set(bins):
        pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if b == bin]
        coords_b = [load_feats_from_pdb(pdb, chain_residx_gap=None)["all_atom_positions"] for pdb in pdbs_b]
        bin_to_metrics[bin]["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(coords_b,
                                                                                    temp_dir=f"{cfg.out_dir}/tmp",
                                                                                    subsample_pairs=cfg.pairwise_tm_subsample)

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
