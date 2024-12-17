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
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import pad_to_max_len, trim_to_max_len
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.data.pdb_utils import write_batched_to_pdb
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="sidechain_pack", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Latent denoiser
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.sd_ckpt).eval()
    device = lit_sd_model.device

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.sd_ckpt).parent.parent
        model_name = Path(cfg.sd_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/sidechain_pack/{model_name}/{cfg.exp_name}"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "samples")
    traj_out_dir = Path(cfg.out_dir, "traj")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    # Load model config
    with open_dict(lit_sd_model.cfg.data):
        lit_sd_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

    # Load dataset based on model config
    if lit_sd_model.cfg.data.overfit > 0:
        # For debugging with overfitting models, we sidechain pack on the training set
        dataset = ADDataset(phase="train", **lit_sd_model.cfg.data)
    else:
        dataset = ADDataset(phase="eval", evaluation_mode = True, **lit_sd_model.cfg.data)

    dataset.subset_to_length_range(cfg.subset_length_range[0], cfg.subset_length_range[1])  # only eval on proteins within this length range
    num_pdbs = cfg.num_pdbs if cfg.num_pdbs is not None else len(dataset)
    val_dataloader = DataLoader(dataset, batch_size=num_pdbs, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    # Define some random examples to sample
    examples = next(iter(val_dataloader))
    example_indices = np.repeat(np.arange(num_pdbs), cfg.num_samples_per_pdb)
    save_traj_indices = set(np.random.choice(len(example_indices), cfg.n_traj, replace=False))  # get some random indices to save trajectories for
    save_sd_traj_steps = np.linspace(0, cfg.scn_diffusion.num_steps - 1, cfg.limit_diff_traj_steps, dtype=int)  # get the steps of the trajectories we'll save for scn diffusion

    # Create sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    # create noise schedule
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

    # create churn config
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                 "timesteps": None,  # filled in based on batch size
                 "noise_schedule": noise_schedule,
                 "churn_cfg": churn_cfg,
                 "return_scn_diffusion_aux": cfg.limit_diff_traj_steps > 0
                 }

    ### SAMPLE ###
    sample_info = defaultdict(list)

    pbar = tqdm(range(0, len(example_indices), cfg.batch_size))
    for bi in pbar:
        idxs = example_indices[bi:bi + cfg.batch_size]
        batch_i = ADDataset.index_into_batch(examples, idxs)
        batch_i = trim_to_max_len(batch_i)

        x, aatype = batch_i["x"].to(device), batch_i["aatype"].to(device)
        scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)
        seq_mask, missing_atom_mask = batch_i["seq_mask"].to(device), batch_i["missing_atom_mask"].to(device)
        residue_index, chain_index = batch_i["residue_index"].to(device), batch_i["chain_index"].to(device)
        cond_labels_in = {"crop_aug": batch_i["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

        x_in = x.clone()
        x_in[..., rc.non_bb_idxs, :] = 0  # shouldn't be necessary, but just to be safe
        x_denoised, aatype_denoised, aux = lit_sd_model.model.sidechain_pack(
            x_in,
            aatype,
            seq_mask=seq_mask,
            missing_atom_mask=missing_atom_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            cond_labels=cond_labels_in,
            scd_inputs=scd_inputs,
        )

        samples = {"x_denoised": x_denoised,
                   "seq_mask": seq_mask,
                   "missing_atom_mask": missing_atom_mask,
                   "residue_index": residue_index,
                   "chain_index": chain_index,
                   "pred_aatype": aatype_denoised,
                   "psce": aux["psce"],
                   }

        # Store sample info
        seq_mask, aatype = seq_mask.cpu(), aatype.cpu()
        core_mask, surface_mask = eval_metrics.get_core_surface_mask(x.cpu(), batch_i["atom_mask"].cpu())
        sample_info_i = {"pdb_key": batch_i["pdb_key"], "seq_mask": seq_mask, "aatype": aatype, "core_mask": core_mask, "surface_mask": surface_mask, "psce": aux["psce"]}


        # Sidechain RMSD per residue
        atom_mask = batch_i["atom_mask"]
        scn_info, ca_aligned_coords1 = eval_metrics.compute_structure_metrics(x.cpu(), x_denoised.cpu(),
                                                                              atom_mask, aatype=aatype,
                                                                              metrics_to_compute=["scn_rmsd_per_pos",
                                                                                                #   "scn_rmsd_per_pos_ligandmpnn",
                                                                                                  "chi_metrics_per_pos",
                                                                                                  "sce"])
        for k, v in scn_info.items():
            sample_info_i[k] = v

        # Pad sample_info for this batch back to max length
        sample_info_i = pad_to_max_len(sample_info_i, max_len=dataset.fixed_size)

        # Append sample info for this batch
        for k, v in sample_info_i.items():
            sample_info[k].append(v)

        # Save samples
        samples = {k: v.detach().cpu() for k, v in samples.items()}
        pdb_keys = batch_i["pdb_key"]
        filenames = [f"{sample_out_dir}/{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)]
        SeqDenoiser.save_samples_to_pdb(samples, filenames)

        # Save CA-aligned ground truth samples
        ca_aligned_gt_dir = Path(cfg.out_dir, "ca_aligned_gt")
        ca_aligned_gt_dir.mkdir(parents=True, exist_ok=True)
        feats = {
            "aatype": batch_i["aatype"],
            "atom_positions": ca_aligned_coords1,
            "atom_mask": atom_mask,
            "residue_index": batch_i["residue_index"],
            "chain_index": batch_i["chain_index"],
            "b_factors": None,
        }
        filenames = [f"{ca_aligned_gt_dir}/gt_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)]
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")

        # Write trajectories to file
        save_traj_mask = [bi + i in save_traj_indices for i in range(batch_i["x"].shape[0])]  # which among the batch to save
        save_traj_steps = [0]   # only 1 seq design step in inverse folding to save (the first index)

        save_trajs_fn = partial(SeqDenoiser.save_trajs_to_pdb, aux,
                                residue_index=batch_i["residue_index"],
                                chain_index=batch_i["chain_index"],
                                save_traj_mask=save_traj_mask,
                                save_traj_steps=save_traj_steps,
                                save_diff_traj_steps=save_sd_traj_steps,
                                traj_conect=cfg.traj_conect)

        # save x1_scn traj
        save_trajs_fn(x_traj_key="x1_scn_traj", aatype_traj_key=None,  # uses aatype_t traj
                      filenames=[f"{traj_out_dir}/x1_scn_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])

        # save xt_scn traj
        save_trajs_fn(x_traj_key="xt_scn_traj", aatype_traj_key=None,  # uses aatype_t traj
                      filenames=[f"{traj_out_dir}/xt_scn_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])

        # save likelihood traj
        # SeqDenoiser.save_sidechain_likelihood_traj(likelihood_aux, aatype, seq_mask, batch_i["residue_index"], batch_i["chain_index"],
        #                                            save_traj_mask=save_traj_mask, save_diff_traj_steps=save_sd_traj_steps,
        #                                            filenames=[f"{traj_out_dir}/likelihood_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)],
        #                                            traj_conect=cfg.traj_conect)


    sample_info = {k: torch.cat(v, dim=0) if k != "pdb_key" else v for k, v in sample_info.items()}  # concatenate all samples as final output

    del lit_sd_model  # free up memory; we don't need denoiser anymore

    # Save metrics
    with open(f"{cfg.out_dir}/sample_info.pkl", "wb") as f:
        pickle.dump(sample_info, f)


    ### Compute sidechain metrics ###
    scn_metrics = {}
    seq_mask = sample_info["seq_mask"]

    # Average RMSD per protein over proteins
    scn_rmsd_avg = sample_info["scn_rmsd_per_pos"].sum(dim=-1) / seq_mask.sum(dim=-1)
    scn_metrics["scn_rmsd_avg"] = scn_rmsd_avg.mean().item()
    print(f"Average RMSD per protein: {scn_metrics['scn_rmsd_avg']:.3f}")

    # Average RMSD over all residues
    scn_rmsd_avg_all = (sample_info["scn_rmsd_per_pos"] * seq_mask).sum() / seq_mask.sum()
    scn_metrics["scn_rmsd_avg_all"] = scn_rmsd_avg_all.item()

    # Average RMSD over all core and surface residues
    for key in ["core", "surface"]:
        mask = sample_info[f"{key}_mask"]
        scn_rmsd_avg = (sample_info["scn_rmsd_per_pos"][mask] * seq_mask[mask]).sum() / seq_mask[mask].sum()
        scn_metrics[f"scn_rmsd_avg_{key}"] = scn_rmsd_avg.item()

    # Get average RMSD per residue
    for aa_idx, aa in enumerate(rc.restypes_with_x):
        aatype_mask = sample_info["aatype"] == aa_idx
        rmsd_i = sample_info["scn_rmsd_per_pos"][aatype_mask]
        rmsd_avg_i = (rmsd_i * seq_mask[aatype_mask]).sum() / seq_mask[aatype_mask].sum()

        print(f"Average RMSD for {aa}: {rmsd_avg_i:.3f} Å")
        scn_metrics[f"scn_rmsd_avg_{aa}"] = rmsd_avg_i.item()

    print(f"Average RMSD for all residues: {scn_metrics['scn_rmsd_avg_all']:.3f} Å")
    print(f"Average RMSD for core residues: {scn_metrics['scn_rmsd_avg_core']:.3f} Å")
    print(f"Average RMSD for surface residues: {scn_metrics['scn_rmsd_avg_surface']:.3f} Å")

    # Plot average sidechain RMSD per residue
    rmsd_avg_aas = [(aa, scn_metrics[f"scn_rmsd_avg_{aa}"]) for aa in rc.restypes_with_x]
    rmsd_avg_aas = sorted(rmsd_avg_aas, key=lambda x: x[1])

    plt.figure(figsize=(12, 6))
    plt.plot([aa for aa, _ in rmsd_avg_aas], [rmsd for _, rmsd in rmsd_avg_aas], marker="o", linestyle="--")
    plt.xlabel("Residue")
    plt.ylabel("Average sidechain RMSD (Å)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/scn_rmsd_per_res.png")
    plt.close()

    # Get average chi metrics per chi angle
    chi_mask = sample_info["chi_mask"]  # [B, N, 4]
    chi_mae_avg = (sample_info["chi_mae_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    chi_acc_avg = (sample_info["chi_acc_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    for ci in range(4):
        scn_metrics[f"chi{ci+1}_mae_avg"] = chi_mae_avg[ci].item()
        scn_metrics[f"chi{ci+1}_acc_avg"] = chi_acc_avg[ci].item()


    # Save metrics as csv with pandas
    metrics_df = pd.DataFrame(scn_metrics, index=[0])
    metrics_df.to_csv(f"{cfg.out_dir}/scn_metrics.csv", index=False)

    # plot_rmsd_vs_npa(sample_info, cfg.out_dir)
    # plot_rmsd_vs_npa_per_residue(sample_info, cfg.out_dir)
    # plot_per_protein_rmsd_vs_npa(sample_info, cfg.out_dir)


def plot_rmsd_vs_npa(sample_info, out_dir: str):
    scn_rmsd_per_pos = sample_info["scn_rmsd_per_pos"]
    npa = sample_info["npa"]
    res_num_atoms = sample_info["res_num_atoms"]

    valid_mask = (res_num_atoms > 0) & torch.isfinite(npa)
    valid_rmsd = scn_rmsd_per_pos[valid_mask]
    valid_npa = npa[valid_mask]
    rho, _ = spearmanr(valid_npa.cpu().numpy(), valid_rmsd.cpu().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(valid_npa.cpu(), valid_rmsd.cpu(), alpha=0.5)
    plt.xlabel('Nats per Atom (npa)')
    plt.ylabel('RMSD per Position')
    plt.title(f'Scatter Plot of RMSD vs Nats per Atom\nSpearman r = {rho:.3f}')
    plt.grid(True)
    plt.xlim([0, 50])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scn_rmsd_vs_npa.png")
    plt.close()

def plot_rmsd_vs_npa_per_residue(sample_info, out_dir: str):
    scn_rmsd_per_pos = sample_info["scn_rmsd_per_pos"]
    npa = sample_info["npa"]
    res_num_atoms = sample_info["res_num_atoms"]
    aatype = sample_info["aatype"]
    seq_mask = sample_info["seq_mask"]

    valid_mask = (res_num_atoms > 0) & torch.isfinite(npa) & (seq_mask > 0)
    valid_rmsd = scn_rmsd_per_pos[valid_mask]
    valid_npa = npa[valid_mask]
    valid_aatype = aatype[valid_mask]

    residue_types = rc.restypes_with_x
    output_dir = Path(f"{out_dir}/scn_rmsd_vs_npa_per_residue")
    output_dir.mkdir(parents=True, exist_ok=True)

    for aa_idx, aa in enumerate(residue_types):
        residue_mask = (valid_aatype == aa_idx)
        rmsd_i = valid_rmsd[residue_mask]
        npa_i = valid_npa[residue_mask]
        if len(rmsd_i) == 0:
            continue
        rho, _ = spearmanr(npa_i.cpu().numpy(), rmsd_i.cpu().numpy())
        plt.figure(figsize=(8, 6))
        plt.scatter(npa_i.cpu(), rmsd_i.cpu(), alpha=0.5)
        plt.title(f'Residue: {aa}\nSpearman r = {rho:.3f}')
        plt.xlabel('Nats per Atom (npa)')
        plt.ylabel('RMSD per Position')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'scn_rmsd_vs_npa_{aa}.png')
        plt.close()


def plot_per_protein_rmsd_vs_npa(sample_info, out_dir: str):
    scn_rmsd_per_pos = sample_info["scn_rmsd_per_pos"]
    npa = sample_info["npa"]
    res_num_atoms = sample_info["res_num_atoms"]
    seq_mask = sample_info["seq_mask"]

    # valid_mask = (res_num_atoms > 0) & torch.isfinite(npa) & (seq_mask > 0)
    valid_mask = (res_num_atoms > 1) & torch.isfinite(npa) & (seq_mask > 0)  # excludes alanine
    num_valid_residues = valid_mask.sum(dim=1)
    valid_protein_mask = num_valid_residues > 0

    avg_rmsd_per_protein = torch.zeros_like(num_valid_residues, dtype=torch.float)
    avg_npa_per_protein = torch.zeros_like(num_valid_residues, dtype=torch.float)

    sum_rmsd_per_protein = (scn_rmsd_per_pos * valid_mask.float()).sum(dim=1)
    sum_npa_per_protein = (torch.where(valid_mask.bool(), npa, 0)).sum(dim=1)

    avg_rmsd_per_protein[valid_protein_mask] = sum_rmsd_per_protein[valid_protein_mask] / num_valid_residues[valid_protein_mask]
    avg_npa_per_protein[valid_protein_mask] = sum_npa_per_protein[valid_protein_mask] / num_valid_residues[valid_protein_mask]

    rho, _ = spearmanr(avg_npa_per_protein[valid_protein_mask].cpu().numpy(), avg_rmsd_per_protein[valid_protein_mask].cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(avg_npa_per_protein[valid_protein_mask].cpu(), avg_rmsd_per_protein[valid_protein_mask].cpu(), alpha=0.6, edgecolors="none", s=75, c="forestgreen")
    plt.xlabel('Likelihood of corrupted sidechains per protein (nats per atom)')
    plt.ylabel('Sidechain RMSD')
    plt.title(f'Sidechain RMSD vs. corruption likelihood \nSpearman $\\rho$ = {rho:.3f}')
    plt.grid(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/per_protein_rmsd_vs_npa.pdf", dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    main()
