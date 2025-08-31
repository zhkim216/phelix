"""
Utils for sampling from sequence design models.
"""
import copy
import itertools
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gemmi
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.io.parser import parse as aw_parse
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import atom_center_random_augmentation, to
from allatom_design.data.datasets.atomworks_sd_dataset import sd_collator
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    load_input,
    mmcif_to_pdb,
)
from allatom_design.data.transform.preprocess import preprocess_transform
from allatom_design.data.transform.sd_featurizer import sd_featurizer
from allatom_design.data.types import Structure
from allatom_design.data.write.mmcif import (
    batch_write_feats_to_mmcif,
    write_feats_to_mmcif,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design.potts import (
    compute_potts_energy,
)
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


def get_seq_des_model(cfg: DictConfig, device: str) -> Dict[str, Any]:
    """
    Load in a sequence design model. Similar to get_struct_pred_model()
    Example config:

    seq_des_cfg:
    # MPNN args
    model_name: "atom_mpnn"  # ["proteinmpnn", "atom_mpnn"]
    proteinmpnn:
        mpnn_cfg: allatom_design/configs/seq_des/proteinmpnn.yaml
        mpnn_params_dir: /media/scratch/huang_lab/allatom_design/model_params/mpnn
        overrides:
        # num seqs per structure will be batch_size * number_of_batches
        batch_size: 1
        number_of_batches: 1
        verbose: false
    atom_mpnn:
        # Atom MPNN args
        atom_mpnn_cfg: allatom_design/configs/seq_des/atom_mpnn_inference.yaml
        atom_mpnn_ckpt:
    """
    model_name = cfg.model_name
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.atom_mpnn.ckpt_path).eval()
    model_cfg, _ = get_cfg_from_ckpt(cfg.atom_mpnn.ckpt_path)
    data_cfg = hydra.utils.instantiate(model_cfg.data)
    sampling_cfg = OmegaConf.load(cfg.atom_mpnn.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.to_container(cfg.atom_mpnn.overrides, resolve=True))
    seq_des_model["model"] = lit_sd_model.model
    seq_des_model["data_cfg"] = data_cfg
    seq_des_model["sampling_cfg"] = sampling_cfg

    return seq_des_model


def run_seq_des(model: SeqDenoiser,
                data_cfg: DictConfig,
                cfg: DictConfig,  # sampling config
                pdb_paths: list[str],
                device: str,
                pos_constraint_df: Optional[pd.DataFrame] = None,  # optional df for specifying fixed positions for a given pdb name (including extensions)
                out_dir: Optional[str] = None,
                ) -> tuple[dict[str, dict[str, torch.Tensor]],
                           dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux will be a dictionary with the following keys:
        - "out_pdbs": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
    """
    # Set up output directory
    outputs = defaultdict(list)

    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype", "use_label_asym_id"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # columns in input df must be a subset of valid columns
            raise ValueError(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")  # set index to pdb name
        pos_constraint_df.index = pos_constraint_df.index.str.lower()  # convert to lowercase

        # set empty string to NaN for easier parsing
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids
    if cfg.verbose and cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {cfg.omit_aas}")

    # Process PDBs in batches of size B
    pbar = tqdm(total=len(pdb_paths), desc=f"Sampling {len(pdb_paths)} PDBs, {cfg.num_seqs_per_pdb} sequences per PDB...")

    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    with parallel_context as parallel_pool:
        for i in range(0, len(pdb_paths), cfg.batch_size):
            batch_pdb_paths = pdb_paths[i:i+cfg.batch_size]
            B = len(batch_pdb_paths)
            batch = get_sd_batch(batch_pdb_paths, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)

            # Initialize seq_cond and atom_cond masks
            batch = initialize_sampling_masks(batch)

            # Parse fixed positions
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=cfg.verbose)

            # Restrict aatype sampling at certain positions
            sampling_inputs = OmegaConf.to_container(cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(batch, pos_constraint_df, verbose=cfg.verbose)

            if cfg["use_protein_only"]:
                # subset to standard protein-only features; useful for ablations to only condition on protein
                batch["token_exists_override"] = (batch["mol_type"] == const.chain_type_ids["PROTEIN"]) & batch["is_standard"]

            # Run sampling
            output_feats, aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs to cif files
            if out_dir is not None:
                for si, feats_si in enumerate(output_feats):
                    if cfg["save_protein_only"]:
                        # crop to protein-only features; useful for ablations to only fold with protein sequence
                        feats_si = crop_batch_to_protein_only(feats_si)

                    sample_stems = [f"{Path(pdb_file).stem}_sample{si}" for pdb_file in batch_struct_files]
                    batch_out_files = [f"{sample_out_dir}/{sample_stem}.cif" for sample_stem in sample_stems]  # output PDBs
                    batch_write_feats_to_mmcif(feats_si, input_structs=input_structs, filenames=batch_out_files)
                    outputs["out_pdbs"].extend(batch_out_files)
                    outputs["pdb_keys"].extend(output_feats[si]["pdb_key"])

                    # get sampled sequences as a string, with ":" to separate chains
                    for bi in range(output_feats[si]["asym_id"].shape[0]):
                        chain_seqs = []
                        chain_input_seqs = []
                        for chain_id in feats_si["asym_id"][bi].unique():
                            chain_mask = (feats_si["asym_id"][bi] == chain_id).squeeze(0)
                            chain_mask = chain_mask * feats_si["token_pad_mask"][bi].bool().squeeze(0)
                            # temporary: don't save non-protein tokens
                            chain_mask = chain_mask & (feats_si["mol_type"][bi] == const.chain_type_ids["PROTEIN"])
                            if not chain_mask.any():
                                continue

                            # store sampled sequence as a string
                            chain_res_type = feats_si["res_type"][bi].squeeze(0).argmax(dim=-1)[chain_mask]
                            chain_seq = [const.prot_token_to_letter[const.tokens[x]] for x in chain_res_type]
                            chain_seqs.append("".join(chain_seq))

                            # store input sequence as a string
                            chain_input_res_type = aux["input_res_type"][si][bi].squeeze(0).argmax(dim=-1)[chain_mask]
                            chain_input_seq = [const.prot_token_to_letter[const.tokens[x]] for x in chain_input_res_type]
                            chain_input_seqs.append("".join(chain_input_seq))

                        outputs["seqs"].append(":".join(chain_seqs))  # store sampled sequences for each sample
                        outputs["input_seqs"].append(":".join(chain_input_seqs))  # store input sequences for each sample

                # If specified, save potts parameters
                if cfg.get("save_potts_params", False):
                    potts_params_dir = f"{out_dir}/potts_params"
                    Path(potts_params_dir).mkdir(parents=True, exist_ok=True)
                    sample_stems = [f"{Path(pdb_file).stem}_sample{si}" for pdb_file in batch_struct_files]
                    for i, sample_stem in enumerate(sample_stems):
                        potts_params = {k: v[i] for k, v in aux["potts_decoder_aux"].items()}
                        torch.save(potts_params, f"{potts_params_dir}/{sample_stem}.pt")

            pbar.update(B)
    pbar.close()

    return outputs


def score_sequences_ensemble(model: SeqDenoiser,
                             data_cfg: DictConfig,
                             cfg: DictConfig,  # sampling config
                             pdb_to_processed_conformers: dict[str, list[str]],  # maps from a given pdb name to its processed conformer structure files
                             pdb_to_sequences: dict[str, list[str]],  # maps from a given pdb name to its sequences
                             device: str,
                             out_dir: Optional[str] = None,
                             ) -> dict[str, Any]:
    """
    Score sequences using Potts parameters computed from input backbones.
    """
    # Set up output directory
    outputs = {}

    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    with parallel_context as parallel_pool:
        for pdb_name, struct_files in tqdm(pdb_to_processed_conformers.items(), desc=f"Scoring {len(pdb_to_processed_conformers)} PDBs..."):
            # Get batch of conformers
            batch = get_sd_batch(struct_files, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)

            # Initialize seq_cond and atom_cond masks
            batch = initialize_sampling_masks(batch)

            # Get potts parameters for each conformer
            sampling_inputs = OmegaConf.to_container(cfg, resolve=True)
            potts_decoder_aux, batch = model.score_samples(batch, sampling_inputs=sampling_inputs)

            # Get sequences
            S = []
            for seq in pdb_to_sequences[pdb_name]:
                S.append(torch.tensor([const.token_ids[const.prot_letter_to_token[aa]] for aa in seq], device=device))
            S = torch.stack(S, dim=0)

            # Score sequences for each conformer
            n_conformers = potts_decoder_aux["h"].shape[0]
            U = torch.zeros(n_conformers, S.shape[0], device=device)
            for ci in range(n_conformers):
                potts_decoder_aux_ci = {k: v[ci] for k, v in potts_decoder_aux.items()}
                potts_decoder_aux_ci = {k: v.expand(S.shape[0], *(v.ndim * (-1, ))) for k, v in potts_decoder_aux_ci.items()}  # expand to match S
                U_ci, _ = compute_potts_energy(S, potts_decoder_aux_ci["h"], potts_decoder_aux_ci["J"], potts_decoder_aux_ci["edge_idx"])
                U[ci] = U_ci

            # Take minimum over conformers to get final energy
            U = U.min(dim=0)[0]

            # Store results
            outputs[pdb_name] = U.cpu()

    return outputs


def run_seq_des_ensemble(model: SeqDenoiser,
                         data_cfg: DictConfig,
                         cfg: DictConfig,  # sampling config
                         pdb_to_processed_conformers: dict[str, list[str]],  # maps from a given pdb name to its processed conformer structure files
                         device: str,
                         pos_constraint_df: Optional[pd.DataFrame] = None,  # optional df for specifying fixed positions for a given pdb name (including extensions)
                         use_primary_res_type: bool = True,  # if True, use res_type from primary structure, otherwise use res_type from conformer struct file
                         out_dir: Optional[str] = None,
                         ) -> dict[str, Any]:
    """
    Given a list of processed structure files, run sequence design on them.
    """
    # Set up output directory
    outputs = defaultdict(list)

    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # columns in input df must be a subset of valid columns
            raise ValueError(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")  # set index to pdb name
        pos_constraint_df.index = pos_constraint_df.index.str.lower()  # convert to lowercase

        # set empty string to NaN for easier parsing
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids
    if cfg.verbose and cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {cfg.omit_aas}")

    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    with parallel_context as parallel_pool:
        for pdb_name, struct_files in tqdm(pdb_to_processed_conformers.items(), desc=f"Sampling {len(pdb_to_processed_conformers)} PDBs, {cfg.num_seqs_per_pdb} sequences per PDB..."):
            # Flatten struct_files and create tied_sampling_ids
            batch = get_sd_batch(struct_files, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)
            batch["tied_sampling_ids"] = torch.zeros(len(struct_files), device=device, dtype=torch.long)  # tie all samples together

            # Use res_type from primary structure
            if use_primary_res_type:
                batch["res_type"] = batch["res_type"][0:1].expand(len(struct_files), *((batch["res_type"].ndim - 1) * (-1, )))  # use res_type from primary structure

            # Initialize seq_cond and atom_cond masks
            batch = initialize_sampling_masks(batch)

            # Parse fixed positions
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=cfg.verbose)

            # Restrict aatype sampling at certain positions
            sampling_inputs = OmegaConf.to_container(cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(batch, pos_constraint_df, verbose=cfg.verbose)

            if cfg["use_protein_only"]:
                # subset to standard protein-only features; useful for ablations to only condition on protein
                batch["token_exists_override"] = (batch["mol_type"] == const.chain_type_ids["PROTEIN"]) & batch["is_standard"]

            # Run sampling
            output_feats, aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Format outputs
            for si in range(len(output_feats)):  # iterate over number of sampled sequences per PDB
                feats_si = output_feats[si]
                U_si = aux["U"][si].item()

                if cfg["save_protein_only"]:
                    # crop to protein-only features; useful for ablations to only fold with protein sequence
                    feats_si = crop_batch_to_protein_only(feats_si)

                if out_dir is not None:
                    # ave outputs to cif files
                    out_file = f"{sample_out_dir}/{pdb_name}_sample{si}.cif"
                    batch_write_feats_to_mmcif(feats_si, input_structs=input_structs[0:1], filenames=[out_file])

                outputs["out_pdbs"].append(out_file)  # store output PDB paths
                outputs["n_conformers"].append(len(input_structs))  # store number of conformers for each PDB (some may have been skipped due to parsing issues)
                outputs["U"].append(U_si)  # store energies for each sample

                # get sampled sequences as a string, with ":" to separate chains
                chain_seqs = []
                chain_input_seqs = []
                for chain_id in feats_si["asym_id"].unique():
                    # store sampled sequence as a string
                    chain_mask = (feats_si["asym_id"] == chain_id).squeeze(0)
                    chain_res_type = feats_si["res_type"].squeeze(0).argmax(dim=-1)[chain_mask]
                    chain_seq = [const.prot_token_to_letter[const.tokens[x]] for x in chain_res_type]
                    chain_seqs.append("".join(chain_seq))

                    # store input sequence as a string
                    chain_input_res_type = aux["input_res_type"][si].squeeze(0).argmax(dim=-1)[chain_mask]
                    chain_input_seq = [const.prot_token_to_letter[const.tokens[x]] for x in chain_input_res_type]
                    chain_input_seqs.append("".join(chain_input_seq))

                outputs["seqs"].append(":".join(chain_seqs))  # store sampled sequences for each sample
                outputs["input_seqs"].append(":".join(chain_input_seqs))  # store input sequences for each sample

            # If specified, save potts parameters
            if cfg.get("save_potts_params", False):
                potts_params_dir = f"{out_dir}/potts_params"
                Path(potts_params_dir).mkdir(parents=True, exist_ok=True)
                potts_params = {k: v[0] for k, v in aux["potts_decoder_aux"].items()}
                torch.save(potts_params, f"{potts_params_dir}/{pdb_name}.pt")

    return outputs


def get_sd_batch(pdb_paths: list[str], device: str,
                 data_cfg: DictConfig,
                 parallel_pool: Parallel | None) -> tuple[dict[str, TensorType["b n ..."]],
                                                          list[str],
                                                          list[Dict[str, int]]]:
    if parallel_pool is None:
        # Load PDBs sequentially
        batch_examples = [get_sd_example(pdb_path, data_cfg) for pdb_path in pdb_paths]
    else:
        # Load PDBs in parallel
        batch_examples = parallel_pool(delayed(get_sd_example)(pdb_path, data_cfg) for pdb_path in pdb_paths)

    # Collate examples
    batch = sd_collator(batch_examples)
    batch = to(batch, device)  # move to device

    return batch


def get_sd_example(pdb_path: str, data_cfg: DictConfig) -> tuple[dict[str, TensorType["b n ..."]],
                                                                         dict[str, Any]]:
    """
    Given a structure file path, return a batch of features and the input structure.
    """
    transformation_id = "1"  # keep only the first assembly
    input_data = aw_parse(pdb_path,
                          extra_fields="all",
                          hydrogen_policy="remove",
                          build_assembly=[transformation_id],
                          add_missing_atoms=False,  # True overrides extra_fields
                          )
    atom_array_from_cif = input_data["assemblies"][transformation_id][0] # (1, num_atoms) -> (num_atoms)

    # Run the pipeline on the CIF data
    pipeline = preprocess_transform()
    cif_out = pipeline(
        data={
            "example_id": Path(pdb_path).stem,
            "atom_array": atom_array_from_cif,
            "chain_info": input_data["chain_info"],
        }
    )

    featurizer = sd_featurizer()

    example = featurizer(cif_out)

    return example



def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]]) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """
    # Initialize sequence mask: always condition on non-protein or non-standard residues
    standard_prot_mask = batch['is_protein']
    batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
    batch["seq_cond_mask"] = torch.where(standard_prot_mask, torch.zeros_like(batch["seq_cond_mask"]), batch["token_resolved_mask"])

    # Initialize atom mask: condition on backbone atoms, non-protein atoms, and non-standard residues
    batch["atom_cond_mask"] = batch["prot_bb_atom_mask"]  # condition on backbone atoms

    ## condition on non-protein atoms and non-standard residues
    atomwise_standard_prot_mask = torch.gather(standard_prot_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
    batch["atom_cond_mask"] = torch.where(atomwise_standard_prot_mask.bool(), batch["atom_cond_mask"], batch["atom_resolved_mask"])

    return batch


def parse_fixed_pos_info(batch: dict[str, TensorType["b ..."]],
                        #  input_structs: list[Structure],
                         pos_constraint_df: pd.DataFrame | None,
                         verbose: bool = False) -> dict[str, torch.Tensor]:

    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for the aatype and sidechain overrides.
    - possibly overridden "aatype"


    The pos_constraint_df should have the following format:
    index: PDB name (including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    seq_cond_mask, atom_cond_mask = batch["seq_cond_mask"].clone(), batch["atom_cond_mask"].clone()

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        return batch

    for i, example_id in enumerate(batch["example_id"]):
        if verbose:
            print(f"\n======================== {example_id} ========================")

        if example_id not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {example_id}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[example_id]
        fixed_pos_seq, fixed_pos_scn = row.get("fixed_pos_seq", np.nan), row.get("fixed_pos_scn", np.nan)  # get fixed positions for this PDB
        use_label_asym_id = row.get("use_label_asym_id", False)

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        #* make chain_to_asym_id mapping directly from batch[i]
        batch_i = batch[i]

        # batch_i should have these features:
        # ['example_id', 'atom_array', 'residue_index', 'token_index', 'asym_id', 'entity_id',
        #   'sym_id', 'restype', 'is_protein', 'is_rna', 'is_dna', 'is_ligand', 'is_atomized',
        #     'atom_to_token_map', 'token_bonds', 'coords', 'token_resolved_mask',
        #       'atom_resolved_mask', 'token_pad_mask', 'atom_pad_mask', 'prot_bb_atom_mask',
        #         'prot_scn_atom_mask', 'token_to_center_atom']

        if use_label_asym_id:
            chain_to_asym_id = {c["name"]: c["asym_id"] for c in input_struct.chains}  # we use label_asym_name as the chain name for fixing positions
            asym_id_to_chain = {c["asym_id"]: c["name"] for c in input_struct.chains}
        else:
            chain_to_asym_id = {c["auth_asym_name"]: c["asym_id"] for c in input_struct.chains}  # we use auth_asym_name as the chain name for fixing positions, not the label_asym_name
            asym_id_to_chain = {c["asym_id"]: c["auth_asym_name"] for c in input_struct.chains}

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(fixed_pos_override_seq, chain_to_asym_id, example["auth_seq_id"], example["asym_id"])
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                batch["res_type"][i, abs_pos_i] = F.one_hot(torch.tensor(const.token_ids[const.prot_letter_to_token[aa]], device=batch["res_type"].device), num_classes=len(const.tokens))

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, chain_to_asym_id, example["auth_seq_id"], example["asym_id"])
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            # if verbose:
            #     print("Fixed sequence:")
            #     visualize_sequences(example, seq_cond_mask[i], input_struct, asym_id_to_chain)
        else:
            if verbose:
                print(f"{example_id}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, chain_to_asym_id, example["auth_seq_id"], example["asym_id"])
            scn_atom_mask = torch.isin(example["atomwise_token_idx"], torch.tensor(abs_fixed_pos_scn, device=example["atomwise_token_idx"].device))
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            # ensure that we're not fixing sidechains when we override the PDB sequence
            scn_cond_num_atoms = scn_atom_mask.float() @ example["atom_to_token"].float()  # number of atoms we're conditioning on at fixed_pos_scn
            if not pd.isna(fixed_pos_override_seq):
                assert (scn_cond_num_atoms[override_abs_pos] == 0).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            # if verbose:
            #     print("Fixed sidechains:")
            #     visualize_sequences(example, scn_cond_num_atoms > 0, input_struct, asym_id_to_chain)
        else:
            if verbose:
                print(f"{example_id}: No fixed sidechain positions specified.")

    # Update batch
    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def parse_pos_restrict_aatype_info(batch: Dict[str, TensorType["b ..."]],
                                   input_structs: list[Structure],
                                   pos_constraint_df: Optional[pd.DataFrame],
                                   verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """
    Given a pos_constraint_df containing position restrictions for each PDB, return:
    - a mask indicating which positions have restricted amino acid sampling
    - a mask indicating which amino acids are allowed at each position

    The pos_constraint_df should have the following format:
    index: PDB name (including extension)
    columns: ["pos_restrict_aatype"]
    where each entry is a comma-separated string of positions in the format "A1:AVG,B10:ILMV", or None.
    """
    B, N = batch["token_pad_mask"].shape
    K = len(const.tokens)

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["token_pad_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["token_pad_mask"].device)

    if verbose:
        print("\n Position-wise amino acid restrictions:")

    for i, pdb_key in enumerate(batch["pdb_key"]):
        if pdb_key not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_key}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_key]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_key}: No position-wise amino acid restrictions specified.")
            continue

        # Set up example
        example = {k: v[i] for k, v in batch.items()}
        input_struct = input_structs[i]
        chain_to_asym_id = {c["auth_asym_name"]: c["asym_id"] for c in input_struct.chains}  # we use auth_asym_name as the chain name for fixing positions, not the label_asym_name
        asym_id_to_chain = {c["asym_id"]: c["auth_asym_name"] for c in input_struct.chains}

        if verbose:
            print(f"{pdb_key}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(
            pos_restrict_aatype,
            chain_to_asym_id,
            example["auth_seq_id"],
            example["asym_id"]
        )

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in const.prot_letter_to_token:
                    allowed_aatype_mask[i, pos_idx, const.token_ids[const.prot_letter_to_token[aa]]] = 1.0
                else:
                    print(f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_key} at position {pdb_pos[abs_pos.index(pos_idx)]}")

        if verbose:
            # Print a summary of the restrictions
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f"  Position {pos_str}: Restricted to {allowed_aa}")
            print("\n========================\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_fixed_pos_str(fixed_pos_str: str,
                        chain_id_mapping: Dict[str, int],
                        residue_index: TensorType["n", int],
                        chain_index: TensorType["n", int]) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A", "B1", "C10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A,B1,C10-25").
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        List[int]: List of absolute indices to set to 1 in the masks.
    """
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match_with_residues = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        # Match pattern for just a chain ID, e.g., "A"
        match_chain_only = re.match(r"([A-Za-z])$", pos)

        if match_with_residues:
            chain_letter = match_with_residues.group(1)
            start_residue = int(match_with_residues.group(2))
            end_residue = int(match_with_residues.group(3)) if match_with_residues.group(3) else start_residue

            if chain_letter not in chain_id_mapping:
                raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

            # For the given chain, create a mask for all residues in the desired range
            chain_i = chain_id_mapping[chain_letter]
            range_mask = (chain_index == chain_i) & (residue_index >= start_residue) & (residue_index <= end_residue)
            matching_indices = torch.where(range_mask)[0]

            # Check that each residue in the requested range; warn if not found
            found_residues = residue_index[matching_indices].tolist()
            found_residues_set = set(found_residues)

            for r in range(start_residue, end_residue + 1):
                if r not in found_residues_set:
                    print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

            # Extend our fixed indices with whatever we did find
            fixed_indices.extend(matching_indices.tolist())
        elif match_chain_only:
            chain_letter = match_chain_only.group(1)

            if chain_letter not in chain_id_mapping:
                raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

            # For the given chain, create a mask for all residues
            chain_i = chain_id_mapping[chain_letter]
            chain_mask = (chain_index == chain_i)
            matching_indices = torch.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
        else:
            raise ValueError(f"Invalid position format: {pos}")

    return fixed_indices

def parse_fixed_pos_override_seq_str(override_str: str,
                                 chain_id_mapping: dict[str, int],
                                 residue_index: TensorType["n", int],
                                 chain_index: TensorType["n", int]
                                 ) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and override amino acids (e.g., ["A", "L"]).

    Args:
        override_str (str): Comma-separated string of position overrides
                           in the format "<chain+residue>:<desired aatype>"
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        tuple: (pdb_pos, abs_pos, override_aatypes) - lists with corresponding entries
    """
    if not override_str or override_str.strip() == "":
        return [], [], []

    pdb_pos = []
    override_aatypes = []

    # Split by comma and process each override
    overrides = [o.strip() for o in override_str.split(",") if o.strip()]

    for override in overrides:
        # Split by colon to get position and override aatype
        parts = override.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'A26:A'")

        pos, aatype = parts[0].strip(), parts[1].strip()

        if len(aatype) != 1 or aatype not in rc.restypes_with_x:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), chain_id_mapping, residue_index, chain_index)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(pos_restrict_str: str,
                                  chain_id_mapping: dict[str, int],  # maps chain letter to chain index
                                  residue_index: TensorType["n", int],
                                  chain_index: TensorType["n", int]) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and allowed aatypes (e.g., ["AVG", "VG"]).

    Args:
        pos_restrict_str (str): Comma-separated string of position restrictions
                               in the format "<chain+residue>:<allowed aatypes>"
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        tuple: (pdb_pos, abs_pos, allowed_aatypes) - lists with corresponding entries
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    # Split by comma and process each restriction
    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        # Split by colon to get position and allowed aatypes
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), chain_id_mapping, residue_index, chain_index)

    return pdb_pos, abs_pos, allowed_aatypes


def visualize_sequences(example: dict[str, torch.Tensor],
                        cond_mask: TensorType["n", int],
                        input_struct: Structure,
                        asym_id_to_chain: Dict[str, str],
                        ) -> str:
    """
    Visualize the conditioning sequence for a given batch of residues.
    """
    # Construct chain map (map chain_id to index of chain in the input structure)
    chain_map = {c["asym_id"]: i for i, c in enumerate(input_struct.chains)}

    # first, get full sequence for each chain
    sequences = {}
    for chain_id in example["asym_id"].unique().tolist():
        chain_mask = example["asym_id"] == chain_id
        mol_type = example["mol_type"][chain_mask].unique().tolist()[0]
        chain_cond_mask = cond_mask[chain_mask]
        if mol_type != const.chain_type_ids["NONPOLYMER"]:
            # Extract sequence from the features
            # Get the unpadded sequence for this chain
            res_type = example["res_type"][chain_mask].argmax(dim=-1)
            res_type = res_type[example["token_pad_mask"][chain_mask].bool()].tolist()
            sequence = [const.tokens[res_type[ri]] for ri in range(len(res_type))]
            sequences[chain_id] = "".join([x if chain_cond_mask[j] else "-" for j, x in enumerate(gemmi.one_letter_code(sequence))])
        else:
            # Extract sequence from the input structure, since non-polymer chains are never redesigned
            chain_i = input_struct.chains[chain_map[chain_id]]
            res_start = chain_i["res_idx"]
            res_end = chain_i["res_idx"] + chain_i["res_num"]
            sequence = input_struct.residues[res_start:res_end]["name"].tolist()
            sequence = "".join([f"<{x}>" for x in sequence])  # <> to denote CCD code, not 1-letter
            sequences[chain_id] = sequence

    # print sequences for each chain
    for chain_id, seq in sequences.items():
        print(f"Chain {asym_id_to_chain[chain_id]}: {seq}")