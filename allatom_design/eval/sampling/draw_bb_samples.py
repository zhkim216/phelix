import glob
import pickle
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
import yaml
from allatom_design.eval.folding_utils import get_struct_pred_model
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="draw_bb_samples", version_base="1.3.2")
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
    traj_out_dir = Path(cfg.out_dir, "traj")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    # Define the range of lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
    all_lengths = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample
    save_traj_mask = np.tile(np.arange(cfg.n_samples_per_length) < cfg.n_traj_per_length, len(lengths_to_sample))  # get mask of the trajectories we'll save
    save_traj_steps = np.linspace(0, cfg.num_steps - 1, cfg.limit_traj_steps, dtype=int)  # get the steps of the trajectories we'll save
    print(f"Drawing {cfg.n_samples_per_length} samples each of lengths {start} to {end} with step size {cfg.length_step_size}")

    # Override s_max
    if cfg.ca.s_max_override is not None:
        lit_ad_model.model.denoiser.interpolant.ca_interpolant.set_s_max(cfg.ca.s_max_override)

    if cfg.nco.s_max_override is not None:
        lit_ad_model.model.denoiser.interpolant.nco_interpolant.set_s_max(cfg.nco.s_max_override)

    ### SAMPLE ###
    pbar = tqdm(total=len(all_lengths))

    for i in range(0, len(all_lengths), cfg.batch_size):
        # Choose lengths and residue index
        lengths = torch.tensor(all_lengths[i:i + cfg.batch_size], dtype=torch.long).to(lit_ad_model.device)
        B = lengths.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long).to(lit_ad_model.device)
        residue_index = residue_index[None].expand(B, -1)

        # Create timesteps, separating timesteps for CA and NCO
        t_ca = sampling_utils.get_timesteps_from_schedule(**cfg.ca.timestep_schedule)
        t_ca = t_ca[None].expand(B, -1).to(lit_ad_model.device)
        t_nco = sampling_utils.get_timesteps_from_schedule(**cfg.nco.timestep_schedule)
        t_nco = t_nco[None].expand(B, -1).to(lit_ad_model.device)
        timesteps = (t_ca, t_nco)

        # Create noise schedules for CA and NCO
        noise_schedule = (NoiseSchedule(cfg.ca.noise_schedule),
                          NoiseSchedule(cfg.nco.noise_schedule))

        # Create churn configs for CA and NCO
        churn_cfg = (dict(cfg.ca.churn_cfg), dict(cfg.nco.churn_cfg))

        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, lit_ad_model.device)
        x_bb_denoised, aux = lit_ad_model.model.sample(lengths,
                                                       residue_index=residue_index,
                                                       timesteps=timesteps,
                                                       cond_labels=cond_labels_in,
                                                       noise_schedule=noise_schedule,
                                                       churn_cfg=churn_cfg,
                                                       autoguidance_cfg=dict(cfg.autoguidance_cfg)
                                                       )

        samples = {"x_bb_denoised": x_bb_denoised,
                   "seq_mask": aux["seq_mask"],
                   "residue_index": residue_index}
        samples = {k: v.cpu() if v is not None else v for k, v  in samples.items()}

        # Save samples
        filenames = [f"{sample_out_dir}/sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)]
        AtomDenoiser.save_samples_to_pdb(samples, filenames)

        # Write trajectories to file
        align_models_to_idx = None
        if cfg.align_traj_to_last_step:
            # align all predictions along the trajectory to the last step
            align_models_to_idx = cfg.limit_traj_steps - 1

        save_trajs_fn = partial(AtomDenoiser.save_trajs_to_pdb, aux, residue_index=residue_index, chain_index=torch.zeros_like(residue_index),
                                save_traj_mask=save_traj_mask, save_traj_steps=save_traj_steps,
                                traj_conect=cfg.traj_conect, align_models_to_idx=align_models_to_idx)
        # save x1_bb traj
        save_trajs_fn(x_traj_key="x1_bb_traj", filenames=[f"{traj_out_dir}/x1_traj_sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)])

        # save xt_bb traj
        save_trajs_fn(x_traj_key="xt_bb_traj", filenames=[f"{traj_out_dir}/xt_traj_sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)])

        pbar.update(B)
    pbar.close()

    del lit_ad_model  # free up memory; we don't need denoiser anymore

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)
    pdbs = natsorted(glob.glob(f"{sample_out_dir}/*.pdb"))

    # Load in MPNN and struct pred models
    if cfg.run_mpnn_sc:
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    if (cfg.run_mpnn_sc or cfg.run_codes_sc):
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Get secondary structure info
    ss_info = eval_metrics.compute_secondary_structure_content(pdbs)
    for pdb, v in ss_info.items():
        all_metrics[pdb]["ss_info"] = v

    # Run MPNN self-consistency evaluation
    if cfg.run_mpnn_sc:
        mpnn_sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                              mpnn_model, mpnn_cfg,
                                                              struct_pred_model,
                                                              device,
                                                              out_dir=cfg.out_dir)
        for pdb, v in mpnn_sc_info.items():
            all_metrics[pdb]["mpnn_sc_info"] = v

    # Run co-design self-consistency evaluation
    if cfg.run_codes_sc:
        codes_sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                               None, None,  # no MPNN model for co-design eval
                                                               struct_pred_model,
                                                               device,
                                                               out_dir=cfg.out_dir,
                                                               eval_codesign=True)
        for pdb, v in codes_sc_info.items():
            all_metrics[pdb]["codes_sc_info"] = v

    # Run nnTM evaluation
    if cfg.nntm_dataset is not None:
        nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=cfg.out_dir)
        for pdb, v in nntm_info.items():
            all_metrics[pdb]["nntm_info"] = v

    ### SAVE METRICS ###
    # Save all metrics to pickle file
    with open(f"{cfg.out_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    # Save certain metrics to a csv file
    metrics_df = defaultdict(list)
    for pdb in pdbs:
        if cfg.run_mpnn_sc:
            mpnn_sc_info = all_metrics[pdb]["mpnn_sc_info"]
            num_seqs = len(mpnn_sc_info["mpnn_preds"]["mpnn_seqs"])
        else:
            num_seqs = 1

        ss_info = all_metrics[pdb]["ss_info"]
        for i in range(num_seqs):
            metrics_df["pdb"].append(pdb)
            metrics_df["seq_idx"].append(i)

            # add secondary structure metrics of original sample
            metrics_df["pct_alpha"].append(all_metrics[pdb]["ss_info"]["pct_alpha"])
            metrics_df["pct_beta"].append(all_metrics[pdb]["ss_info"]["pct_beta"])

            # add self-consistency metrics
            if cfg.run_mpnn_sc:
                metrics_df["mpnn_seq"].append(mpnn_sc_info["mpnn_preds"]["mpnn_seqs"][i])
                metrics_df["mpnn_sc_ca_rmsd"].append(mpnn_sc_info["sc_metrics"]["sc_ca_rmsd"][i].item())
                metrics_df["mpnn_sc_ca_tm"].append(mpnn_sc_info["sc_metrics"]["sc_ca_tm"][i].item())
                metrics_df["mpnn_sc_avg_plddt"].append(mpnn_sc_info["struct_preds"]["avg_plddt"][i].item())

            # add co-design self-consistency metrics (same for each MPNN sequence since we calculate these on the original sample)
            if cfg.run_codes_sc:
                codes_sc_info = all_metrics[pdb]["codes_sc_info"]
                metrics_df["codes_seq"].append(codes_sc_info["sample_seq"])
                metrics_df["codes_sc_ca_rmsd"].append(codes_sc_info["sc_metrics"]["sc_ca_rmsd"].squeeze().item())
                metrics_df["codes_sc_aa_rmsd"].append(codes_sc_info["sc_metrics"]["sc_aa_rmsd"].squeeze().item())
                metrics_df["codes_sc_ca_tm"].append(codes_sc_info["sc_metrics"]["sc_ca_tm"].squeeze().item())
                metrics_df["codes_sc_aa_tm"].append(codes_sc_info["sc_metrics"]["sc_aa_tm"].squeeze().item())
                metrics_df["codes_sc_avg_plddt"].append(codes_sc_info["struct_preds"]["avg_plddt"].squeeze().item())

            # add nntm metrics
            if cfg.nntm_dataset is not None:
                metrics_df["nntm"].append(all_metrics[pdb]["nntm_info"])

    metrics_df = pd.DataFrame(metrics_df)

    # extract length of samples if it's convenient
    if cfg.run_mpnn_sc:
        metrics_df["len"] = metrics_df["mpnn_seq"].str.len()
    elif cfg.run_codes_sc:
        metrics_df["len"] = metrics_df["codes_seq"].str.len()
    metrics_df.to_csv(f"{cfg.out_dir}/metrics.csv", index=False)

    ### PLOT METRICS ###
    plot_out_dir = f"{cfg.out_dir}/plots"
    Path(plot_out_dir).mkdir(parents=True, exist_ok=True)

    if cfg.run_mpnn_sc:
        plot_sc_ca_rmsd_vs_len(metrics_df, plot_out_dir)
        plot_sc_ca_tm_vs_len(metrics_df, plot_out_dir)
        if cfg.nntm_dataset is not None:
            plot_sc_ca_rmsd_vs_nntm(metrics_df, plot_out_dir)

    if cfg.run_codes_sc:
        plot_sc_ca_rmsd_vs_len(metrics_df, plot_out_dir, plot_codesign=True)
        plot_sc_ca_tm_vs_len(metrics_df, plot_out_dir, plot_codesign=True)
        if cfg.nntm_dataset is not None:
            plot_sc_ca_rmsd_vs_nntm(metrics_df, plot_out_dir, plot_codesign=True)


def plot_sc_ca_rmsd_vs_len(metrics_df: pd.DataFrame,
                           out_dir: str,
                           plot_codesign: bool = False):
    """
    Plot RMSD vs protein length similar to Protpardelle figures.
    """
    if plot_codesign:
        rmsd_key = "codes_sc_ca_rmsd"
        plddt_key = "codes_sc_avg_plddt"
        out_file = f"{out_dir}/codesign_sc_ca_rmsd_vs_len.png"
    else:
        rmsd_key = "mpnn_sc_ca_rmsd"
        plddt_key = "mpnn_sc_avg_plddt"
        out_file = f"{out_dir}/mpnn_sc_ca_rmsd_vs_len.png"

    # First, get best rows among MPNN sequences by scRMSD (or any row for the codesign model)
    metrics_df_best = metrics_df.iloc[metrics_df[rmsd_key].groupby(metrics_df["pdb"]).idxmin()]

    rmsd = metrics_df_best[rmsd_key]
    lengths = metrics_df_best["len"]

    # Plot proportion with RMSD < 2
    proportions = metrics_df_best.groupby("len")[rmsd_key].apply(lambda x: (x < 2).mean()).sort_index()
    proportions = proportions.rolling(window=11, center=True, min_periods=1).mean()

    # Plot scRMSD vs protein length
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.set_xlabel('protein length')
    sc = ax1.scatter(lengths, rmsd, c=metrics_df_best[plddt_key], s=2, vmin=0.2, vmax=1.0)
    fig.colorbar(sc, label='pLDDT', pad=0.2)
    ax1.axhline(y=2, color='r', linestyle='--', lw=1)
    ax1.set_ylabel('scRMSD')
    ax1.set_ylim(0, 25)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'% samples with scRMSD < 2')
    ax2.yaxis.label.set_color('tab:orange')
    ax2.plot(proportions.index, proportions, color='tab:orange')
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_sc_ca_tm_vs_len(metrics_df: pd.DataFrame,
                         out_dir: str,
                         plot_codesign: bool = False):
    """
    Plot TM score vs protein length
    """
    if plot_codesign:
        rmsd_key = "codes_sc_ca_rmsd"
        tm_key = "codes_sc_ca_tm"
        plddt_key = "codes_sc_avg_plddt"
        out_file = f"{out_dir}/codesign_sc_ca_tm_vs_len.png"
    else:
        rmsd_key = "mpnn_sc_ca_rmsd"
        tm_key = "mpnn_sc_ca_tm"
        plddt_key = "mpnn_sc_avg_plddt"
        out_file = f"{out_dir}/mpnn_sc_ca_tm_vs_len.png"

    # First, get best rows among MPNN sequences by scRMSD (or any row for the codesign model)
    metrics_df_best = metrics_df.iloc[metrics_df[rmsd_key].groupby(metrics_df["pdb"]).idxmin()]

    tm = metrics_df_best[tm_key]
    lengths = metrics_df_best["len"]

    # Plot proportion with TM > 0.5
    proportions = metrics_df_best.groupby("len")[tm_key].apply(lambda x: (x > 0.5).mean()).sort_index()
    proportions = proportions.rolling(window=11, center=True, min_periods=1).mean()

    # Plot scTM vs protein length
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.set_xlabel('protein length')
    sc = ax1.scatter(lengths, tm, c=metrics_df_best[plddt_key], s=2, vmin=0.2, vmax=1.0)
    fig.colorbar(sc, label='pLDDT', pad=0.2)
    ax1.axhline(y=0.5, color='r', linestyle='--', lw=1)
    ax1.set_ylabel('scTM')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'% samples with scTM > 0.5')
    ax2.yaxis.label.set_color('tab:orange')
    ax2.plot(proportions.index, proportions, color='tab:orange')
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_sc_ca_rmsd_vs_nntm(metrics_df: pd.DataFrame,
                            out_dir: str,
                            plot_codesign: bool = False):
    """
    Plot scRMSD vs nntm
    """
    if plot_codesign:
        rmsd_key = "codes_sc_ca_rmsd"
        out_file = f"{out_dir}/codesign_sc_ca_rmsd_vs_nntm.png"
    else:
        rmsd_key = "mpnn_sc_ca_rmsd"
        out_file = f"{out_dir}/mpnn_sc_ca_rmsd_vs_nntm.png"

    # First, get best rows among MPNN sequences by scRMSD (or any row for the codesign model)
    metrics_df_best = metrics_df.iloc[metrics_df[rmsd_key].groupby(metrics_df["pdb"]).idxmin()]

    rmsd = metrics_df_best[rmsd_key]
    nntm = metrics_df_best["nntm"]

    # Plot scRMSD vs nntm
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.set_xlabel('nntm')
    sc = ax1.scatter(nntm, rmsd, c=metrics_df_best["len"], s=1, cmap="plasma")
    fig.colorbar(sc, label='len', pad=0.2)
    ax1.set_ylabel('scRMSD')
    ax1.set_ylim(0, 15)
    ax1.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
