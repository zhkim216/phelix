"""
Utils for sampling from sequence design models.
"""
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import atomworks.enums as aw_enums
import atomworks.io.utils.sequence as aw_sequence
import biotite.structure as struc
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.ml.utils.token import apply_token_wise, get_token_starts
from biotite.structure import AtomArray
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import to
from allatom_design.data.datasets.atomworks_sd_dataset import sd_collator
from allatom_design.data.transform.preprocess import preprocess_transform
from allatom_design.data.transform.sd_featurizer import sd_featurizer
from allatom_design.data.write.mmcif import batch_write_feats_to_mmcif
from allatom_design.model.seq_denoiser.denoisers.seq_design.potts import \
    compute_potts_energy
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


def get_seq_des_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a sequence design model. Similar to get_struct_pred_model()
    Example config:

    seq_des_cfg:
        # MPNN args
        model_name: "atom_mpnn"  # ["atom_mpnn"]
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
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype", "use_label_asym_name"]
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
            batch = get_sd_batch(batch_pdb_paths, data_cfg=data_cfg, device=device, parallel_pool=parallel_pool)

            # Initialize seq_cond and atom_cond masks
            batch = initialize_sampling_masks(batch)

            # Parse fixed positions
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=cfg.verbose)

            # Restrict aatype sampling at certain positions
            sampling_inputs = OmegaConf.to_container(cfg, resolve=True)
            # sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(batch, pos_constraint_df, verbose=cfg.verbose)  # TODO: re-implement

            # Run sampling
            id_to_atom_arrays, aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs.
            if out_dir is not None:
                for example_id, atom_arrays in id_to_atom_arrays.items():
                    # Save output atom arrays to cif files.
                    sample_stems = [f"{example_id}_sample{si}" for si in range(len(atom_arrays))]
                    batch_out_files = [f"{sample_out_dir}/{sample_stem}.cif" for sample_stem in sample_stems]  # output PDBs

                    for si in range(len(atom_arrays)):
                        atom_array = atom_arrays[si]
                        with open(batch_out_files[si], "w") as f:
                            f.write(to_cif_string(atom_array, include_entity_poly=True, include_nan_coords=False))
                        outputs["example_ids"].append(example_id)
                    outputs["out_pdbs"].extend(batch_out_files)

                    # Get sampled sequences as a string, with ":" to separate chains.
                    for si in range(len(atom_arrays)):
                        chain_seqs = []
                        prot_atom_array = atom_arrays[si][struc.filter_amino_acids(atom_arrays[si])]
                        prot_1to3_fn = np.vectorize(lambda x: aw_sequence.get_1_from_3_letter_code(x, aw_enums.ChainType.POLYPEPTIDE_L))
                        for asym_id in np.unique(prot_atom_array.pn_unit_iid):
                            asym_mask = prot_atom_array.pn_unit_iid == asym_id
                            chain_atom_array = prot_atom_array[asym_mask]
                            _, resnames = struc.get_residues(chain_atom_array)
                            chain_seq = "".join(prot_1to3_fn(resnames))
                            chain_seqs.append(chain_seq)
                        outputs["seqs"].append(":".join(chain_seqs))

                    # If specified, save potts parameters
                    if cfg.get("save_potts_params", False):
                        potts_params_dir = f"{out_dir}/potts_params"
                        Path(potts_params_dir).mkdir(parents=True, exist_ok=True)
                        sample_stems = [f"{example_id}_sample{si}" for example_id in batch["example_id"] for si in range(len(atom_arrays))]
                        for i, sample_stem in enumerate(sample_stems):
                            potts_params = {k: v[i] for k, v in aux["potts_decoder_aux"].items()}
                            torch.save(potts_params, f"{potts_params_dir}/{sample_stem}.pt")

            pbar.update(B)
    pbar.close()

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


def get_sd_batch(pdb_paths: list[str],
                 *,
                 data_cfg: DictConfig,
                 device: str,
                 parallel_pool: Parallel | None) -> dict[str, Any]:
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


def get_sd_example(pdb_path: str, data_cfg: DictConfig) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.
    """
    # BACKWARDS COMPATIBILITY  TODO: remove this once we've retrained the models
    if "cif_parser_args" not in data_cfg:
        data_cfg.cif_parser_args = {"add_missing_atoms": True, "remove_waters": True, "remove_ccds": [], "fix_ligands_at_symmetry_centers": True, "fix_arginines": True, "convert_mse_to_met": True, "hydrogen_policy": "remove"}
    cif_parser_args = OmegaConf.to_container(data_cfg.cif_parser_args, resolve=True)

    # Read in the CIF data.
    transformation_id = "1"  # keep only the first assembly
    cif_parser_args["build_assembly"] = [transformation_id]
    input_data = aw_parse(pdb_path, **cif_parser_args)
    atom_array_from_cif = input_data["assemblies"][transformation_id][0] # (1, num_atoms) -> (num_atoms)

    # Run the preprocessing pipeline on the CIF data.
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

    # Add auth_seq_id and auth_asym_id to the example's atom array.
    auth_cif_parser_args = cif_parser_args.copy()
    auth_cif_parser_args["extra_fields"] = "all"
    auth_cif_parser_args["add_missing_atoms"] = False  # True overrides extra_fields
    auth_data = aw_parse(pdb_path, **auth_cif_parser_args)["assemblies"][transformation_id][0]
    mapping = {}
    for atom in auth_data:
        # Create mapping of ("pn_unit_iid", "res_id") to ("auth_asym_id", "auth_seq_id").
        mapping[(atom.pn_unit_iid, atom.res_id)] = (atom.auth_asym_id, int(atom.auth_seq_id))

    # Add auth_asym_id and auth_seq_id to the example's atom array.
    auth_asym_id, auth_seq_id = zip(*map(lambda x: mapping.get((x.pn_unit_iid, x.res_id), ("", const.DUMMY_SEQ_ID)), example["atom_array"]))
    example["atom_array"].set_annotation("auth_asym_id", auth_asym_id)
    example["atom_array"].set_annotation("auth_seq_id", auth_seq_id)

    return example


def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]]) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """
    # Initialize sequence mask: always condition on non-protein or non-standard residues
    standard_prot_mask = batch["is_protein"] & ~batch["is_atomized"]
    batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
    batch["seq_cond_mask"] = torch.where(standard_prot_mask, torch.zeros_like(batch["seq_cond_mask"]), batch["token_resolved_mask"])

    # Initialize atom mask: condition on backbone atoms, non-protein atoms, and non-standard residues
    batch["atom_cond_mask"] = batch["prot_bb_atom_mask"]  # condition on backbone atoms

    ## condition on non-protein atoms and non-standard residues
    atomwise_standard_prot_mask = torch.gather(standard_prot_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
    batch["atom_cond_mask"] = torch.where(atomwise_standard_prot_mask.bool(), batch["atom_cond_mask"], batch["atom_resolved_mask"])

    return batch


def parse_fixed_pos_info(batch: dict[str, TensorType["b ..."]],
                         pos_constraint_df: pd.DataFrame | None,
                         verbose: bool = False) -> dict[str, torch.Tensor]:

    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for seq-level and atom-level conditioning
    - possibly overridden "res_type"

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
        use_label_asym_name = row.get("use_label_asym_name", False)

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(fixed_pos_override_seq, example["atom_array"], use_label_asym_name=use_label_asym_name)
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                batch["restype"][i, abs_pos_i] = F.one_hot(torch.tensor(const.AF3_ENCODING.encode_aa_seq(aa), device=batch["restype"].device), num_classes=const.AF3_ENCODING.n_tokens)

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, example["atom_array"], use_label_asym_name=use_label_asym_name)
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                print("Fixed sequence:")
                visualize_sequences(example["atom_array"], seq_cond_mask[i][example["token_pad_mask"].bool()])
        else:
            if verbose:
                print(f"{example_id}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, example["atom_array"], use_label_asym_name=use_label_asym_name)
            scn_atom_mask = torch.isin(example["atom_to_token_map"], torch.tensor(abs_fixed_pos_scn, device=example["atom_to_token_map"].device))
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            # ensure that we're not fixing sidechains when we override the PDB sequence
            scn_cond_num_atoms = apply_token_wise(example["atom_array"], scn_atom_mask.cpu().numpy(), np.sum)
            if not pd.isna(fixed_pos_override_seq):
                assert (scn_cond_num_atoms[override_abs_pos] == 0).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                print("Fixed sidechains:")
                visualize_sequences(example["atom_array"], scn_cond_num_atoms > 0)
        else:
            if verbose:
                print(f"{example_id}: No fixed sidechain positions specified.")

    # Update batch
    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def parse_pos_restrict_aatype_info(batch: dict[str, TensorType["b ..."]],
                                   pos_constraint_df: Optional[pd.DataFrame],
                                   verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None:
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

        # Make chain_to_asym_id mapping
        chain_to_asym_id = get_chain_to_asym_id_mapping(example, use_label_asym_name=True)

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
                if aa in const.PROT_LETTER_TO_TOKEN:
                    allowed_aatype_mask[i, pos_idx, const.AF3_ENCODING.encode_aa(aa)] = 1.0
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
                        atom_array: AtomArray,
                        use_label_asym_name: bool = False) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A", "B1", "C10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A,B1,C10-25").
        atom_array (AtomArray): AtomArray object containing the atom array.
        use_label_asym_name (bool): Whether to use label_asym_name as the chain name.

    Returns:
        TensorType["k", int]: List of absolute indices to set to 1 in the masks.
    """
    chain_annotation = "pn_unit_iid" if use_label_asym_name else "auth_asym_id"
    residue_index = atom_array.auth_seq_id[get_token_starts(atom_array)]
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

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

            # For the given chain, create a mask for all residues in the desired range
            atomwise_range_mask = (atom_array.get_annotation(chain_annotation) == chain_letter) & (atom_array.auth_seq_id >= start_residue) & (atom_array.auth_seq_id <= end_residue)
            range_mask = apply_token_wise(atom_array, atomwise_range_mask, np.any)  # get per-token mask
            matching_indices = np.where(range_mask)[0]

            # Check that each residue in the requested range; warn if not found
            found_residues = set(residue_index[matching_indices].tolist())

            for r in range(start_residue, end_residue + 1):
                if r not in found_residues:
                    print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

            # Extend our fixed indices with whatever we did find
            fixed_indices.extend(matching_indices.tolist())
        elif match_chain_only:
            chain_letter = match_chain_only.group(1)

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

            # For the given chain, create a mask for all residues
            atomwise_chain_mask = (atom_array.get_annotation(chain_annotation) == chain_letter)
            chain_mask = apply_token_wise(atom_array, atomwise_chain_mask, np.any)
            matching_indices = np.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
        else:
            raise ValueError(f"Invalid position format: {pos}")

    return fixed_indices

def parse_fixed_pos_override_seq_str(override_str: str,
                                     atom_array: AtomArray,
                                     use_label_asym_name: bool = False) -> tuple[list[str], list[int], list[str]]:
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
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array, use_label_asym_name)

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


def visualize_sequences(atom_array: AtomArray,
                        cond_mask: TensorType["n", int]) -> str:
    """
    Visualize the conditioning sequence for a given atom array.
    """
    sequences = {}
    token_array = atom_array[get_token_starts(atom_array)]  # get first atom per token
    get_1to3_fn = np.vectorize(lambda x: aw_sequence.get_1_from_3_letter_code(x.res_name, aw_enums.ChainType(x.chain_type)))

    for asym_id in np.unique(token_array.pn_unit_iid):
        asym_mask = token_array.pn_unit_iid == asym_id
        chain_token_array = token_array[asym_mask]
        chain_cond_mask = cond_mask[asym_mask]

        seq_arr = get_1to3_fn(chain_token_array)
        sequence = "".join([x if chain_cond_mask[i] else "-" for i, x in enumerate(seq_arr)])
        sequences[asym_id] = sequence

    for asym_id, sequence in sequences.items():
        print(f"Chain {asym_id}: {sequence}")
