"""
Utils for sampling from backbone generation models.
"""
import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data import residue_constants as rc
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.data import (atom_apply_random_augmentation,
                                      atom_center_random_augmentation,
                                      center_random_augmentation, to)
from allatom_design.data.datasets.boltz_ad_dataset import (
    ad_collator, featurize_diffusion_inputs, featurize_motif_inputs, add_tokenwise_atom_feats)
from allatom_design.data.pdb_utils import write_batched_to_pdb
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.types import Structure, Tokenized
from allatom_design.data.write.mmcif import (write_batched_structures_to_mmcif,
                                             write_ad_feats_to_mmcif)
from allatom_design.eval.eval_utils import eval_metrics, sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


def get_bb_gen_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a backbone generation model.
    """
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    model_cfg, _ = get_cfg_from_ckpt(cfg.ckpt_path)
    data_cfg = hydra.utils.instantiate(model_cfg.data)
    sampling_cfg = OmegaConf.load(cfg.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.to_container(cfg.overrides, resolve=True))
    bb_gen_model = {"model": lit_ad_model.model,
                    "data_cfg": data_cfg,
                    "sampling_cfg": sampling_cfg,
                    "device": device}

    return bb_gen_model


def run_bb_uncond_sampling(model: AtomDenoiser,
                           cfg: DictConfig,
                           device: str,
                           lengths: list[int],
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
    pbar = tqdm(total=len(lengths), desc="Sampling backbones")
    for i in range(0, len(lengths), cfg.batch_size):
        lengths_batch = lengths[i:i + cfg.batch_size]
        B = lengths_batch.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long, device=device)  # assume residue index is 0 to max length
        residue_index = residue_index[None].expand(B, -1)

        # Set up backbone diffusion inputs
        diffusion_params = {}
        diffusion_params["num_steps"] = cfg.num_steps
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
        diffusion_params["timesteps"] = t_bb[None].expand(B, -1).to(device)
        diffusion_params["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
        diffusion_params["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
        diffusion_params["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

        # Create seq mask from lengths
        seq_mask = (residue_index < lengths_batch[:, None]).float()

        # Construct diffusion inputs
        diffusion_inputs = {
            "seq_mask": seq_mask,
            "residue_index": residue_index,
        }

        # Sample backbones
        x_bb_denoised, aux = model.sample(diffusion_inputs=diffusion_inputs,
                                          diffusion_params=diffusion_params,
                                          motif_inputs=None)

        samples = {"x_bb": x_bb_denoised,
                   "seq_mask": seq_mask,
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

        pbar.update(B)
    pbar.close()
    return sampled_pdb_paths


def run_bb_partial_diffusion(model: AtomDenoiser,
                             data_cfg: DictConfig,
                             cfg: DictConfig,  # sampling config
                             device: str,
                             struct_file_paths: list[str],
                             n_samples_per_pdb: int,
                             out_dir: str) -> list[str]:
    """
    Run partial diffusion on a set of structures.
    """
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Load in input PDB
    sampled_pdb_paths = []
    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    struct_file_paths = np.repeat(struct_file_paths, n_samples_per_pdb).tolist()
    with parallel_context as parallel_pool:
        pbar = tqdm(total=len(struct_file_paths), desc="Running partial diffusion")
        for i in range(0, len(struct_file_paths), cfg.batch_size):
            struct_file_batch_paths = struct_file_paths[i:i + cfg.batch_size]
            B = len(struct_file_batch_paths)

            # Get batch of inputs
            unconditional_motif_cfg = {"name": "unconditional", "motif_type": "unconditional"}  # TODO: find a better way to handle unconditional sampling from scaffolding models / creation of empty motifs
            batch, input_structures = get_bb_batch(struct_file_batch_paths, data_cfg, unconditional_motif_cfg, device, parallel_pool)

            # Set up backbone diffusion inputs
            diffusion_params = {}
            diffusion_params["use_partial_diffusion"] = True
            diffusion_params["num_steps"] = cfg.num_steps
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
            diffusion_params["timesteps"] = t_bb[None].expand(B, -1).to(device)
            diffusion_params["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
            diffusion_params["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
            diffusion_params["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

            # Sample backbones
            x_bb_denoised, _ = model.sample(diffusion_inputs=batch["diffusion_inputs"],  # uses true auth seq ID from PDBs
                                            diffusion_params=diffusion_params,
                                            motif_inputs=batch["motif_inputs"])
            samples = {"x_bb": x_bb_denoised,
                       "seq_mask": batch["diffusion_inputs"]["seq_mask"],
                       "residue_index": batch["diffusion_inputs"]["residue_index"]}
            samples = {k: v.cpu() if v is not None else v for k, v in samples.items()}

            # Save samples
            filenames = [f"{sample_out_dir}/sample_{batch['pdb_key'][j]}_{(i+j) % n_samples_per_pdb}.pdb" for j in range(B)]
            AtomDenoiser.save_samples_to_pdb(samples, filenames)
            sampled_pdb_paths.extend(filenames)

            pbar.update(B)
        pbar.close()

    return sampled_pdb_paths


def get_bb_batch(struct_file_batch_paths: list[str],
                 data_cfg: DictConfig,
                 motif_cond_type_cfg: DictConfig | None,
                 device: str,
                 parallel_pool: Parallel | None,
                 ) -> tuple[dict[str, Any], list[Structure]]:
    """
    Get a batch of backbone generation model inputs from a list of PDB files.
    """
    if parallel_pool is None:
        # Load PDBs sequentially
        # batch_examples = [get_bb_example(struct_file_path, data_cfg, motif_cond_type_cfg, device) for struct_file_path in struct_file_batch_paths]
        batch_examples, input_structures = zip(*[get_bb_example(struct_file_path, data_cfg, motif_cond_type_cfg) for struct_file_path in struct_file_batch_paths])
    else:
        # Load PDBs in parallel
        batch_examples, input_structures = zip(*parallel_pool(delayed(get_bb_example)(struct_file_path, data_cfg, motif_cond_type_cfg) for struct_file_path in struct_file_batch_paths))

    # Collate examples
    batch = ad_collator(batch_examples)
    batch = to(batch, device)  # move to device

    return batch, input_structures


def get_bb_example(struct_file_path: str,
                   data_cfg: DictConfig,
                   motif_cond_type_cfg: DictConfig | None = None) -> tuple[dict[str, Any], Structure]:
    """
    Get a backbone generation model input from a PDB file.
    """
    example = {}

    input_data = load_input(struct_file_path)

    # Tokenize structure (no cropping applied)
    tokenized = data_cfg["tokenizer"].tokenize(input_data)
    tokenized = add_tokenwise_atom_feats(tokenized, data_cfg["featurizer"])

    # Featurize diffusion inputs
    example["diffusion_inputs"] = featurize_diffusion_inputs(tokenized, use_auth_as_residx=data_cfg["use_auth_as_residx"], max_tokens=None)

    # Featurize motif
    example["motif_inputs"] = featurize_motif_inputs(tokenized, data_cfg["use_auth_as_residx"], data_cfg["motif_selector"], data_cfg["motif_cropper"], data_cfg["motif_featurizer"],
                                                     motif_data_kwargs=data_cfg["motif_feats"],
                                                     motif_cond_type_cfg=motif_cond_type_cfg)

    # Center motif atoms and apply random augmentation
    if example["motif_inputs"]["token_pad_mask"].sum() > 0:
        # Center on motif atoms
        centered_motif_coords, transforms = atom_center_random_augmentation(example["motif_inputs"]["motif_coords"],
                                                                            example["motif_inputs"]["motif_atom_mask"],
                                                                            apply_random_augmentation=data_cfg["se3_augment_cfg"]["enabled"],
                                                                            translation_scale=data_cfg["se3_augment_cfg"]["translation_scale"],
                                                                            return_transforms=True)
        example["motif_inputs"]["motif_coords"] = centered_motif_coords

        # Apply transformation to input structure
        input_data.structure.atoms["coords"] = atom_apply_random_augmentation(torch.tensor(input_data.structure.atoms["coords"]),
                                                                              torch.tensor(input_data.structure.atoms["is_present"]),
                                                                              transforms)



    example["pdb_key"] = Path(struct_file_path).stem
    return example, input_data.structure



def run_backbone_scaffolding(model: AtomDenoiser,
                             cfg: DictConfig,
                             device: str,
                             motif_info_df: pd.DataFrame,
                             out_dir: str) -> list[str]:
    """
    Run scaffold sampling from a backbone generation model. Uses the motif info dataframe to extract motifs.
    Motif info dataframe must have the following columns:
        - pdb_path: path to the PDB file
        - length: length of the motif
        - contigs: semicolon-separated string of contig indices, e.g. "10;B1-17;40;A1-7;10"
            where the length of the contig is specified by the single numeric value, and the motif contig is specified by the chain letter and residue indices
        - N: number of samples to generate
    """
    raise NotImplementedError("This function needs to be updated")
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")  # stores generated samples
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    motif_out_dir = Path(out_dir, "motifs")  # stores motifs
    Path(motif_out_dir).mkdir(parents=True, exist_ok=True)
    centered_gt_out_dir = Path(out_dir, "centered_gt")  # stores centered ground truth examples from which motifs were drawn
    Path(centered_gt_out_dir).mkdir(parents=True, exist_ok=True)

    sampled_pdb_paths = []
    for motif_idx in tqdm(range(0, len(motif_info_df)), desc="Scaffolding backbones"):
        motif_info = motif_info_df.iloc[motif_idx]
        pdb_path = motif_info["pdb_path"]
        length = motif_info["length"]
        contigs_str = motif_info["contigs"]
        num_to_sample = motif_info["N"]

        # Load in input PDB
        example, pdb_name, chain_id_mapping = get_bb_example(pdb_path, sm=None, device=device)
        pdb_stem = Path(pdb_name).stem

        # Parse contigs string
        contigs = parse_contigs_str(contigs_str, chain_id_mapping, example["residue_index"], example["chain_index"])

        # Build input batch
        pbar = tqdm(total=num_to_sample, desc=f"Sampling backbones for motif {Path(pdb_path).stem}", leave=False)
        for i in range(0, num_to_sample, cfg.batch_size):
            B = min(cfg.batch_size, num_to_sample - i)

            # Build scaffold inputs by parsing contigs string and extracting motif residues
            scaffold_inputs = [build_scaffold_inputs_from_contigs(example["x"], example["atom_mask"], length, contigs) for _ in range(B)]
            batch_scaffold_inputs = {k: torch.stack([v[k] for v in scaffold_inputs], dim=0) for k in scaffold_inputs[0].keys()}

            ### Save motifs as PDBs ###
            # Save motifs
            motif_samples = {"aatype": batch_scaffold_inputs["aatype_motif"],
                             "atom_positions": batch_scaffold_inputs["x_motif"],
                             "atom_mask": batch_scaffold_inputs["motif_mask"],
                             "residue_index": torch.arange(length, device=device).expand(B, -1),
                             "chain_index": torch.zeros((B, length), device=device),
                             "b_factors": torch.ones_like(batch_scaffold_inputs["motif_mask"], dtype=torch.float32)
                             }
            feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in motif_samples.items()}  # move to cpu
            motif_filenames = [f"{motif_out_dir}/motif{motif_idx}_{pdb_stem}_{i + j}.pdb" for j in range(B)]
            write_batched_to_pdb(**feats, filenames=motif_filenames, mode="aa")

            # Set up backbone diffusion inputs
            diffusion_params = {}
            diffusion_params["num_steps"] = cfg.num_steps
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
            diffusion_params["timesteps"] = t_bb[None].expand(B, -1).to(device)
            diffusion_params["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
            diffusion_params["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
            diffusion_params["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

            # Create conditioning labels
            cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

            # Sample backbones
            x_bb_denoised, _ = model.sample(lengths=torch.ones(B, device=device) * length,
                                            residue_index=torch.arange(length, device=device).expand(B, -1),
                                            diffusion_params=diffusion_params,
                                            scaffold_inputs=batch_scaffold_inputs,
                                            cond_labels=cond_labels_in)

            # Save samples
            samples = {
                "x_bb": x_bb_denoised.cpu(),
                "seq_mask": torch.ones((B, length)),
                "residue_index": torch.arange(length).expand(B, -1),
            }
            filenames = [f"{sample_out_dir}/sample{motif_idx}_{pdb_stem}_{i + j}.pdb" for j in range(B)]
            AtomDenoiser.save_samples_to_pdb(samples, filenames)
            sampled_pdb_paths.extend(filenames)

            pbar.update(B)
        pbar.close()

    return sampled_pdb_paths


def build_scaffold_inputs_from_contigs(x: TensorType["n a 3", float],
                                       atom_mask: TensorType["n a", float],
                                       length: int,
                                       contigs: list[tuple[str, list[int]]]) -> dict[str, torch.Tensor]:
    """
    Build scaffold inputs from contigs
    """
    N, A, _ = x.shape
    x_motif = torch.zeros((length, A, 3), device=x.device)
    motif_mask = torch.zeros((length, A), device=x.device)

    # Fill in motif indices
    current_contig_start = 0
    for segment_type, segment_info in contigs:
        if segment_type == "motif_indices":
            motif_indices = segment_info
            x_motif[current_contig_start:current_contig_start + len(motif_indices)] = x[motif_indices]
            motif_mask[current_contig_start:current_contig_start + len(motif_indices)] = 1
            current_contig_start += len(motif_indices)
        elif segment_type == "contig":
            contig_length = segment_info
            motif_mask[current_contig_start:current_contig_start + contig_length] = 0
            current_contig_start += contig_length

    # Only condition on backbone atoms  # TODO: support sidechain atoms
    motif_mask[:, rc.non_bb_idxs] = 0
    x_motif = x_motif * motif_mask[..., None]
    aatype_motif = torch.full((length,), fill_value=rc.restype_order_with_x["X"], device=x.device)  # TODO: fix for sequence conditioning

    # Re-center on CA of motif residues
    seq_mask = torch.ones(length, device=x.device)
    if (motif_mask[..., 1:2].any()):  # only center if there are any scaffolding residues
        x_motif, transforms = center_random_augmentation(x_motif, seq_mask, motif_mask,
                                                         translation_scale=0.0,
                                                         return_transforms=True)

    return {"x_motif": x_motif, "motif_mask": motif_mask, "aatype_motif": aatype_motif}



def parse_contigs_str(contigs_str: str,
                      chain_id_mapping: dict[str, int],
                      residue_index: TensorType["n", int],
                      chain_index: TensorType["n", int]) -> list[
                          tuple[str, int | list[int]]
                        ]:
    """
    Parse out motif segments and single-value contigs from a semicolon-separated string of the form
    "10;B1-17;40;A1-7;10", preserving the order as:
      [
        ("contig", 10),
        ("motif_indices", [absolute_idx1, absolute_idx2, ...]),
        ("contig", 40),
        ("motif_indices", [absolute_idx3, absolute_idx4, ...]),
        ...
      ]

    Each motif segment is of the form "<chain_id><residue_index_start>-<residue_index_end>" (e.g. "B1-17").
    Each contig is a single numeric value (e.g. "10") and is stored as an int in the output.

    Args:
        contigs_str (str): The semicolon-separated string containing either motif segments or single-value contigs.
        chain_id_mapping (dict[str, int]): Mapping of chain letter to chain index.
        residue_index (TensorType["n", int]): Tensor of residue indices.
        chain_index (TensorType["n", int]): Tensor of chain indices.

    Returns:
        list[tuple[str, list[int] | int]]: A list of tuples where each tuple is either:
            ("motif_indices", [absolute_indices]) or ("contig", int_value   ).
    """
    parsed_segments = []

    contigs_str = contigs_str.strip()
    if not contigs_str:
        return parsed_segments

    segments = [seg.strip() for seg in contigs_str.split(";") if seg.strip()]

    for seg in segments:
        # Identify if segment is a motif (starts with chain letter) or a single-value contig (starts with a digit)
        if re.match(r"^[A-Za-z]", seg):
            # Parse a motif segment like "B1-17"
            match = re.match(r"^([A-Za-z])(\d+)-(\d+)$", seg)
            if not match:
                raise ValueError(f"Invalid motif segment format: {seg}")

            chain_letter, start_str, end_str = match.groups()
            start_res = int(start_str)
            end_res = int(end_str)

            if chain_letter not in chain_id_mapping:
                raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

            chain_i = chain_id_mapping[chain_letter]
            range_mask = (chain_index == chain_i) & (residue_index >= start_res) & (residue_index <= end_res)
            matching_indices = torch.where(range_mask)[0]

            found_residues = set(residue_index[matching_indices].tolist())
            for r in range(start_res, end_res + 1):
                if r not in found_residues:
                    print(f"Warning: Requested motif position {chain_letter}{r} not found in structure.")

            parsed_segments.append(("motif_indices", matching_indices.tolist()))
        else:
            # Parse a single-value contig like "10"
            try:
                contig_val = int(seg)
            except ValueError:
                raise ValueError(f"Invalid contig format (must be numeric): {seg}")
            parsed_segments.append(("contig", contig_val))

    return parsed_segments


def run_motif_cond_type_sampling(model: AtomDenoiser,
                                 data_cfg: DictConfig,
                                 motif_cond_type_cfg: DictConfig,
                                 cfg: DictConfig,  # sampling config
                                 device: str,
                                 struct_file_paths: list[str],
                                 out_dir: str) -> tuple[list[str], dict[str, Any]]:
    """
    Run motif-conditioned sampling from a backbone generation model based on the motif conditioning type config.
    Uses the input PDBs to extract motifs, calling the motif selector in the data_cfg to pick motifs.
    """
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")  # stores generated samples
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    motif_out_dir = Path(out_dir, "motifs")  # stores motifs
    Path(motif_out_dir).mkdir(parents=True, exist_ok=True)
    centered_gt_out_dir = Path(out_dir, "centered_gt")  # stores centered ground truth examples from which motifs were drawn
    Path(centered_gt_out_dir).mkdir(parents=True, exist_ok=True)
    master_search_dir = Path(out_dir, "master_search")  # stores master search results
    Path(master_search_dir).mkdir(parents=True, exist_ok=True)

    # Set motif selection type
    data_cfg["motif_selector"].set_motif_cond_type_cfg(motif_cond_type_cfg)

    sampled_pdb_paths = []
    motif_info = {}  # map from pdb path to motif mask and coordinates
    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    with parallel_context as parallel_pool:
        pbar = tqdm(total=len(struct_file_paths), desc="Sampling backbones")
        for i in range(0, len(struct_file_paths), cfg.batch_size):
            struct_file_batch_paths = struct_file_paths[i:i + cfg.batch_size]
            B = len(struct_file_batch_paths)

            batch, input_structures = get_bb_batch(struct_file_batch_paths, data_cfg, motif_cond_type_cfg, device, parallel_pool)

            ### Save motifs as PDBs ###
            # Save motifs
            if motif_cond_type_cfg["motif_type"] != "unconditional":
                # TODO: make sure write_ad_feats_to_mmcif can support empty motifs
                motif_feats_out = batch["motif_inputs"]
                motif_feats_out["coords"] = batch["motif_inputs"]["motif_coords"]
                batch_motif_paths = [f"{motif_out_dir}/motif_{batch['pdb_key'][j]}_{i + j}.cif" for j in range(B)]
                write_ad_feats_to_mmcif(motif_feats_out, filenames=batch_motif_paths)

            # Save centered examples from which motifs were drawn
            batch_centered_paths = [f"{centered_gt_out_dir}/centered_{batch['pdb_key'][j]}_{i + j}.cif" for j in range(B)]
            write_batched_structures_to_mmcif(input_structures, batch_centered_paths)

            ### Sample backbones ###
            # Set up backbone diffusion params
            diffusion_params = {}
            diffusion_params["num_steps"] = cfg.num_steps
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
            diffusion_params["timesteps"] = t_bb[None].expand(B, -1).to(device)
            diffusion_params["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
            diffusion_params["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
            diffusion_params["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

            # Sample backbones
            x_bb_denoised, _ = model.sample(diffusion_inputs=batch["diffusion_inputs"],  # uses true auth seq ID from PDBs
                                            diffusion_params=diffusion_params,
                                            motif_inputs=batch["motif_inputs"])
            samples = {
                "x_bb": x_bb_denoised.cpu(),
                "seq_mask": batch["diffusion_inputs"]["seq_mask"].cpu(),
                "residue_index": batch["diffusion_inputs"]["residue_index"].cpu() + 1,  # 1-indexed for saving to PDB
            }

            # Save samples
            batch_sampled_paths = [f"{sample_out_dir}/sample_{batch['pdb_key'][j]}_{i + j}.pdb" for j in range(B)]
            AtomDenoiser.save_samples_to_pdb(samples, batch_sampled_paths)
            sampled_pdb_paths.extend(batch_sampled_paths)

            if motif_cond_type_cfg["motif_type"] != "unconditional":
                # Get motif indices within sampled structures
                master_dfs = []
                for j, pdb_path in enumerate(batch_sampled_paths):
                    master_df = eval_metrics.motif_master_search(batch_motif_paths[j], batch_sampled_paths[j], f"{master_search_dir}/temp")
                    master_df.to_csv(f"{master_search_dir}/master_hits_{batch['pdb_key'][j]}_{i + j}.tsv", sep="\t", index=False)
                    master_dfs.append(master_df)
            else:
                master_dfs = [None] * B

            # Add motif info
            for j, pdb_path in enumerate(batch_sampled_paths):
                # add motif and centered ground truth paths
                motif_info[pdb_path] = {"motif_path": batch_motif_paths[j],
                                        "centered_path": batch_centered_paths[j],
                                        "master_df": master_dfs[j]}

            pbar.update(B)
        pbar.close()

    return sampled_pdb_paths, motif_info