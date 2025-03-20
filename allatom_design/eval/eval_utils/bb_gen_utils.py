"""
Utils for sampling from backbone generation models.
"""
from functools import partial
from pathlib import Path
from typing import Any

import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import load_feats_from_pdb, pad_to_max_len
from allatom_design.data.datasets.ad_dataset import (get_scaffold_manager,
                                                     process_single_pdb_ad)
from allatom_design.data.pdb_utils import write_batched_to_pdb
from allatom_design.data.scaffold_manager import ScaffoldManager
from allatom_design.eval.eval_utils import sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


def get_bb_gen_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a backbone generation model.
    """
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    sampling_cfg = OmegaConf.load(cfg.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, cfg.overrides)
    bb_gen_model = {"model": lit_ad_model.model,
                    "scaffold_manager": get_scaffold_manager(lit_ad_model.cfg.scaffold_manager),
                    "sampling_cfg": sampling_cfg,
                    "device": device}

    return bb_gen_model


def run_bb_uncond_sampling(model: AtomDenoiser,
                           cfg: DictConfig,
                           lengths: list[int],
                           device: str,
                           out_dir: str,
                           save_traj_inputs: dict[str, Any] | None = None) -> list[str]:
    """
    Run unconditional sampling from a backbone generation model.
    """
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    if save_traj_inputs is not None:
        # Save diffusion trajectories
        traj_out_dir = Path(out_dir, "traj")
        Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    sampled_pdb_paths = []
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    for i in tqdm(range(0, len(lengths), cfg.batch_size)):
        lengths_batch = lengths[i:i + cfg.batch_size]
        B = lengths_batch.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long, device=device)  # assume residue index is 0 to max length
        residue_index = residue_index[None].expand(B, -1)

        # Set up backbone diffusion inputs
        diffusion_inputs = {}
        diffusion_inputs["num_steps"] = cfg.num_steps
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
        diffusion_inputs["timesteps"] = t_bb[None].expand(B, -1).to(device)
        diffusion_inputs["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
        diffusion_inputs["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
        diffusion_inputs["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

        # Create conditioning labels
        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

        # Sample backbones
        x_bb_denoised, aux = model.sample(lengths=lengths_batch,
                                          residue_index=residue_index,
                                          diffusion_inputs=diffusion_inputs,
                                          cond_labels=cond_labels_in)
        samples = {"x_bb": x_bb_denoised,
                   "seq_mask": aux["seq_mask"],
                   "residue_index": residue_index}
        samples = {k: v.cpu() if v is not None else v for k, v in samples.items()}

        # Save samples
        filenames = [f"{sample_out_dir}/sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)]
        AtomDenoiser.save_samples_to_pdb(samples, filenames)
        sampled_pdb_paths.extend(filenames)

        if save_traj_inputs is not None:
            # Save trajectories
            save_trajs_fn = partial(AtomDenoiser.save_trajs_to_pdb, aux, residue_index=residue_index, chain_index=torch.zeros_like(residue_index),
                                    save_traj_mask=save_traj_inputs["save_traj_mask"], save_traj_steps=save_traj_inputs["save_traj_steps"],
                                    traj_conect=save_traj_inputs["traj_conect"], align_models_to_idx=save_traj_inputs["align_traj_to_last_step"])
            save_trajs_fn(x_traj_key="x1_bb_traj", filenames=[f"{traj_out_dir}/x1_traj_sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)])
            save_trajs_fn(x_traj_key="xt_bb_traj", filenames=[f"{traj_out_dir}/xt_traj_sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)])


    return sampled_pdb_paths


def get_bb_batch(pdb_batch_files: list[str],
                 sm: ScaffoldManager | None,
                 device: str,
                 parallel_pool: Parallel | None) -> tuple[dict[str, TensorType["b n ..."]],
                                                        list[str]]:
    """
    Get a batch of backbone generation model inputs from a list of PDB files.
    """
    if parallel_pool is None:
        # Load PDBs sequentially
        batch_data = [load_feats_from_pdb(pdb_file) for pdb_file in pdb_batch_files]
    else:
        # Load PDBs in parallel
        batch_data = parallel_pool(delayed(load_feats_from_pdb)(pdb_file) for pdb_file in pdb_batch_files)

    # Load and process all PDBs in this batch
    batch_list = []
    for data in batch_data:
        single = process_single_pdb_ad(data, sm, convert_types=True)
        batch_list.append(single)

    pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]  # includes extension

    # Create a batch dictionary from batch_list by stacking
    model_input_keys = ["x", "seq_mask", "atom_mask", "missing_atom_mask", "residue_index", "x_motif", "motif_mask", "aatype_motif"]
    max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
    batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len) for b in batch_list]  # pad each batch to max length
    batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

    # Move to device
    batch = {k: batch[k].to(device) for k in model_input_keys}

    return batch, pdb_names



def run_bb_scaffold_sampling(model: AtomDenoiser,
                             sm: ScaffoldManager,
                             cfg: DictConfig,
                             pdb_paths: list[str],
                             device: str,
                             out_dir: str) -> list[str]:
    """
    Run scaffold sampling from a backbone generation model. Uses the input PDBs to extract motifs.
    """
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")  # stores generated samples
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    motif_out_dir = Path(out_dir, "motifs")  # stores motifs
    Path(motif_out_dir).mkdir(parents=True, exist_ok=True)
    centered_gt_out_dir = Path(out_dir, "centered_gt")  # stores centered ground truth examples from which motifs were drawn
    Path(centered_gt_out_dir).mkdir(parents=True, exist_ok=True)

    sampled_pdb_paths = []
    with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
        for i in range(0, len(pdb_paths), cfg.batch_size):
            pdb_batch_files = pdb_paths[i:i + cfg.batch_size]
            B = len(pdb_batch_files)
            batch, pdb_names = get_bb_batch(pdb_batch_files, sm, device, parallel_pool)

            ### Save motifs as PDBs ###
            # Save motifs
            motif_samples = {"aatype": batch["aatype_motif"],
                             "atom_positions": batch["x_motif"],
                             "atom_mask": batch["motif_mask"],
                             "residue_index": batch["residue_index"],
                             "chain_index": torch.zeros_like(batch["residue_index"]),  # TODO: fix this
                             "b_factors": torch.ones_like(batch["motif_mask"], dtype=torch.float32)
                             }
            feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in motif_samples.items()}  # move to cpu
            motif_filenames = [f"{motif_out_dir}/motif_{pdb_stem}.pdb" for pdb_stem in pdb_names]
            write_batched_to_pdb(**feats, filenames=motif_filenames, mode="aa")

            write_batched_to_pdb(**feats, filenames=motif_filenames, mode="aa")

            # Save centered examples from which motifs were drawn
            samples = {
                "x_bb": batch["x"][..., rc.bb_idxs, :].cpu(),
                "seq_mask": batch["seq_mask"].cpu(),
                "residue_index": batch["residue_index"].cpu(),
            }
            centered_filenames = [f"{centered_gt_out_dir}/centered_{Path(pdb_names[j]).stem}.pdb" for j in range(B)]
            AtomDenoiser.save_samples_to_pdb(samples, centered_filenames)

            ### Sample backbones ###
            # Set up backbone diffusion inputs
            diffusion_inputs = {}
            diffusion_inputs["num_steps"] = cfg.num_steps
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
            diffusion_inputs["timesteps"] = t_bb[None].expand(B, -1).to(device)
            diffusion_inputs["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
            diffusion_inputs["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
            diffusion_inputs["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

            # Create conditioning labels
            cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

            # Build scaffold inputs
            scaffold_inputs = {
                "x_motif": batch["x_motif"],
                "motif_mask": batch["motif_mask"],
                "aatype_motif": batch["aatype_motif"],
            }

            # Sample backbones
            x_bb_denoised, _ = model.sample(lengths=batch["seq_mask"].sum(dim=-1),
                                            residue_index=batch["residue_index"],  # this uses the true residue index from PDBs
                                            diffusion_inputs=diffusion_inputs,
                                            scaffold_inputs=scaffold_inputs,
                                            cond_labels=cond_labels_in)
            samples = {
                "x_bb": x_bb_denoised.cpu(),
                "seq_mask": batch["seq_mask"].cpu(),
                "residue_index": batch["residue_index"].cpu(),
            }

            # Save samples
            filenames = [f"{sample_out_dir}/sample_{Path(pdb_names[j]).stem}_{i + j}.pdb" for j in range(B)]
            AtomDenoiser.save_samples_to_pdb(samples, filenames)
            sampled_pdb_paths.extend(filenames)

    return sampled_pdb_paths
