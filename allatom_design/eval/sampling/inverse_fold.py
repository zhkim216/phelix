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
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.data.pdb_utils import write_to_pdb_frames
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="inverse_fold", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Load in seq denoiser
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.sd_ckpt).eval()
    device = lit_sd_model.device

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.sd_ckpt).parent.parent
        model_name = Path(cfg.sd_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/inverse_fold/{model_name}/{cfg.exp_name}"

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
        dataset = ADDataset(phase="train", **lit_sd_model.cfg.data)
    else:
        dataset = ADDataset(phase="eval", **lit_sd_model.cfg.data)
    val_dataloader = DataLoader(dataset, batch_size=cfg.num_pdbs, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=False)
    dataset.subset_to_length_range(cfg.subset_length_range[0], cfg.subset_length_range[1])  # only eval on proteins within this length range

    # Define some random examples to sample
    S = cfg.timestep_schedule.num_steps
    examples = next(iter(val_dataloader))
    example_indices = np.repeat(np.arange(cfg.num_pdbs), cfg.num_seqs_per_pdb)
    save_traj_indices = set(np.random.choice(len(example_indices), cfg.n_traj, replace=False))  # get some random indices to save trajectories for
    save_traj_steps = np.linspace(0, S - 1, cfg.limit_traj_steps, dtype=int)  # get the steps of the trajectories we'll save
    save_sd_traj_steps = np.linspace(0, cfg.scn_diffusion.num_steps - 1, cfg.limit_diff_traj_steps, dtype=int)  # get the steps of the trajectories we'll save for scn diffusion

    # Define sequence denoising timesteps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    # create sidechain diffusion noise schedule
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

    # create sidechain diffusion churn config
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                 "timesteps": None,  # filled in based on batch size
                 "noise_schedule": noise_schedule,
                 "churn_cfg": churn_cfg,
                 "autoguidance_cfg": dict(cfg.scn_diffusion.autoguidance_cfg),
                 "return_scn_diffusion_aux": cfg.limit_diff_traj_steps > 0
                 }

    ### SAMPLE ###
    seq_rec_metrics = defaultdict(list)
    pbar = tqdm(range(0, len(example_indices), cfg.batch_size))
    for bi in pbar:
        idxs = example_indices[bi:bi + cfg.batch_size]
        batch_i = ADDataset.index_into_batch(examples, idxs)
        x, seq_mask, residue_index = batch_i["x"].to(device), batch_i["seq_mask"].to(device), batch_i["residue_index"].to(device)
        timesteps = t_seq[None].expand(x.shape[0], -1).to(device)
        scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)

        # Define conditioning labels when we inverse fold
        cond_labels_in = {"crop_aug": batch_i["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

        x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
            x,
            seq_mask=seq_mask,
            residue_index=residue_index,
            timesteps=timesteps,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            cond_labels=cond_labels_in,
            scd_inputs=scd_inputs,
        )
        samples = {"x_denoised": x_denoised,
                   "seq_mask": seq_mask,
                   "residue_index": residue_index,
                   "pred_aatype": aatype_denoised,
                   "aatype_pred_traj": aux["aatype_pred_traj"],
                   "aatype_t_traj": aux["aatype_t_traj"],
                   }

        # Update info for sequence recovery eval
        seq_rec_metrics["pdb"] += batch_i["pdb_key"]

        seq_mask = seq_mask.cpu()
        for i in range(batch_i["x"].shape[0]):
            # Ground truth seqs
            gt_aatype = batch_i["aatype"][i][seq_mask[i].bool()]
            gt_seq = "".join([rc.restypes_with_x[i] for i in gt_aatype])
            seq_rec_metrics["gt_seq"].append(gt_seq)

            # Predicted seqs
            pred_aatype = samples["pred_aatype"][i][seq_mask[i].bool()]
            pred_seq = "".join([rc.restypes_with_x[i] for i in pred_aatype])
            seq_rec_metrics["pred_seq"].append(pred_seq)

            # Sequence accuracy
            seq_rec_metrics["seq_acc"].append(np.mean([np.array(list(pred_seq)) == np.array(list(gt_seq))]))

        # Save samples
        pdb_keys = batch_i["pdb_key"]
        filenames = [f"{sample_out_dir}/{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)]
        SeqDenoiser.save_samples_to_pdb(samples, filenames)

        # Write trajectories to file
        save_traj_mask = [bi + i in save_traj_indices for i in range(batch_i["x"].shape[0])]

        save_trajs_fn = partial(SeqDenoiser.save_trajs_to_pdb, aux,
                                residue_index=residue_index,
                                chain_index=batch_i["chain_index"],
                                save_traj_mask=save_traj_mask,
                                save_traj_steps=save_traj_steps,
                                save_diff_traj_steps=save_sd_traj_steps,
                                traj_conect=cfg.traj_conect)

        # save aatype pred traj
        save_trajs_fn(x_traj_key="xt_traj", aatype_traj_key="aatype_pred_traj",
                      filenames=[f"{traj_out_dir}/aatype_pred_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])

        # save aatype t traj
        save_trajs_fn(x_traj_key="xt_traj", aatype_traj_key="aatype_t_traj",
                      filenames=[f"{traj_out_dir}/aatype_t_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])

        # save x1_scn traj
        save_trajs_fn(x_traj_key="x1_scn_traj", aatype_traj_key=None,  # uses aatype_t traj
                      filenames=[f"{traj_out_dir}/x1_scn_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])

        # save xt_scn traj
        save_trajs_fn(x_traj_key="xt_scn_traj", aatype_traj_key=None,  # uses aatype_t traj
                      filenames=[f"{traj_out_dir}/xt_scn_traj_{pdb_key}_{bi + i}.pdb" for i, pdb_key in enumerate(pdb_keys)])


    del lit_sd_model  # free up memory; we don't need denoiser anymore

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)
    pdbs = natsorted(glob.glob(f"{sample_out_dir}/*.pdb"))

    # Run co-design self-consistency evaluation
    if cfg.run_codes_sc:
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

        codes_sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                               None, None,  # no MPNN model for co-design eval
                                                               struct_pred_model,
                                                               device,
                                                               out_dir=cfg.out_dir,
                                                               eval_codesign=True,
                                                               temp_dir=f"{cfg.out_dir}/tmp")

        for pdb, v in codes_sc_info.items():
            all_metrics[pdb]["codes_sc_info"] = v

    ### SAVE METRICS ###
    # Save all metrics to pickle file
    with open(f"{cfg.out_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    # Save certain metrics to a csv file
    metrics_df = defaultdict(list)
    for pdb in pdbs:
        num_seqs = 1  # TODO: fix this
        for i in range(num_seqs):
            metrics_df["pdb"].append(pdb)
            metrics_df["seq_idx"].append(i)

            # add co-design self-consistency metrics (same for each MPNN sequence since we calculate these on the original sample)
            if cfg.run_codes_sc:
                codes_sc_info = all_metrics[pdb]["codes_sc_info"]
                metrics_df["codes_seq"].append(codes_sc_info["sample_seq"])
                metrics_df["codes_sc_ca_rmsd"].append(codes_sc_info["sc_metrics"]["sc_ca_rmsd"].squeeze().item())
                metrics_df["codes_sc_aa_rmsd"].append(codes_sc_info["sc_metrics"]["sc_aa_rmsd"].squeeze().item())
                metrics_df["codes_sc_ca_tm"].append(codes_sc_info["sc_metrics"]["sc_ca_tm"].squeeze().item())
                metrics_df["codes_sc_avg_plddt"].append(codes_sc_info["struct_preds"]["avg_plddt"].squeeze().item())

    metrics_df = pd.DataFrame(metrics_df)


if __name__ == "__main__":
    main()
