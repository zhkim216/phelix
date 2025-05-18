"""
Utils for sampling from sequence design models.
"""
import copy
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (atom_center_random_augmentation,
                                      pad_to_max_len, to)
from allatom_design.data.datasets.boltz_sd_dataset import sd_collator
from allatom_design.data.datasets.sd_dataset import process_single_pdb_sd
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    load_input, mmcif_to_pdb)
from allatom_design.data.write.mmcif import write_sd_feats_to_mmcif
from allatom_design.eval.eval_utils import sampling_utils
from allatom_design.eval.eval_utils.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.tokenize.boltz import BoltzTokenizer

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

    if model_name == "atom_mpnn":
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.atom_mpnn.ckpt_path).eval()
        model_cfg, _ = get_cfg_from_ckpt(cfg.atom_mpnn.ckpt_path)
        data_cfg = hydra.utils.instantiate(model_cfg.data)
        sampling_cfg = OmegaConf.load(cfg.atom_mpnn.sampling_cfg)
        sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.to_container(cfg.atom_mpnn.overrides, resolve=True))
        seq_des_model["model"] = lit_sd_model.model
        seq_des_model["data_cfg"] = data_cfg
        seq_des_model["sampling_cfg"] = sampling_cfg

    elif model_name == "proteinmpnn":
        mpnn_cfg = OmegaConf.load(cfg.proteinmpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.proteinmpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.proteinmpnn.mpnn_params_dir, mpnn_cfg, device=device)
        seq_des_model["mpnn_model"] = mpnn_model
        seq_des_model["mpnn_cfg"] = mpnn_cfg

    return seq_des_model


def run_seq_des(model: SeqDenoiser,
                data_cfg: DictConfig,
                cfg: DictConfig,  # sampling config
                struct_file_paths: List[str],
                device: str,
                pos_constraint_df: Optional[pd.DataFrame] = None,  # optional df for specifying fixed positions for a given pdb name (including extensions)
                out_dir: Optional[str] = None,
                ) -> Tuple[Dict[str, Dict[str, torch.Tensor]],
                          Dict]:
    """
    Given a list of processed structure files, run sequence design on them.

    Returns a dictionary mapping from PDB paths to dictionaries containing samples for that PDB, including keys:
    - x_denoised: denoised coordinates
    - seq_mask: sequence mask
    - missing_atom_mask: missing atom mask
    - residue_index: residue index
    - chain_index: chain index
    - pred_aatype: predicted amino acid types
    - pred_seq: predicted sequences

    Also returns a run_aux:
    - If out_dir is specified, save the samples to the given directory and return the paths to the samples in aux.
    """
    # Set up output directory
    run_aux = {}

    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

        run_aux["out_pdbs"] = []  # store output PDB paths
        run_aux["input_struct_files"] = []  # store input PDB names
        run_aux["pred_seqs"] = []  # store predicted sequences as a string for each sample

    # Validate pos_constraint_df
    if pos_constraint_df is not None:
        valid_columns = ["pdb_name", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # columns in input df must be a subset of valid columns
            raise ValueError(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")
        pos_constraint_df = pos_constraint_df.set_index("pdb_name")  # set index to pdb name

        # set empty string to NaN for easier parsing
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids
    if cfg.verbose and cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {cfg.omit_aas}")

    # Process PDBs in batches of size B
    struct_file_paths_repeated = np.repeat(struct_file_paths, cfg.num_seqs_per_pdb)
    pbar = tqdm(total=len(struct_file_paths_repeated), desc=f"Sampling {len(struct_file_paths)} PDBs, {cfg.num_seqs_per_pdb} sequences per PDB...")

    input_pdb_to_samples = defaultdict(list)  # maps from a given input pdb path to its samples
    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel
    with parallel_context as parallel_pool:
        for i in range(0, len(struct_file_paths_repeated), cfg.batch_size):
            batch_struct_files = struct_file_paths_repeated[i:i+cfg.batch_size]
            B = len(batch_struct_files)
            batch, input_structures = get_sd_batch(batch_struct_files, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)

            # TODO: Add fixed positions, etc. For now assume no fixed positions
            batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
            batch["seq_cond_mask"] = torch.where(batch["mol_type"] != const.chain_type_ids["PROTEIN"],
                                                 batch["token_pad_mask"],
                                                 torch.zeros_like(batch["seq_cond_mask"]))  # always condition on non-protein restypes
            batch["atom_cond_mask"] = batch["prot_bb_atom_mask"]  # condition on backbone atoms
            atomwise_prot_mask = torch.bmm(batch["atom_to_token"].float(), (batch["mol_type"] == const.chain_type_ids["PROTEIN"]).float().unsqueeze(-1)).squeeze(-1)  # TODO: make more efficient
            batch["atom_cond_mask"] = torch.where(atomwise_prot_mask.bool(), batch["atom_cond_mask"], torch.ones_like(batch["atom_cond_mask"]))  # always condition on non-protein atoms

            # Run sampling
            res_type_pred = model.sample(batch, sampling_inputs=cfg)

            # Save PDB with predicted sequences
            # TODO: handle cond masks
            output_feats = copy.deepcopy(to(batch, device="cpu"))
            output_feats["res_type"] = F.one_hot(res_type_pred, num_classes=len(const.tokens))
            output_feats["coords"] = output_feats["coords"] * output_feats["atom_cond_mask"].unsqueeze(-1)

            # Save outputs to disk
            if out_dir is not None:
                # Save as cif
                sample_stems = [f"{Path(pdb_file).stem}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_file in enumerate(batch_struct_files)]
                batch_out_files = [f"{sample_out_dir}/{sample_stem}.cif" for sample_stem in sample_stems]  # output PDBs
                write_sd_feats_to_mmcif(output_feats, input_structs=input_structures, filenames=batch_out_files)
                # TODO: temporary fix for converting to PDB
                batch_out_pdb_files = [out_file.replace(".cif", ".pdb") for out_file in batch_out_files]
                [mmcif_to_pdb(cif_file, pdb_file, assign_label_seq_id=False, overwrite=True) for cif_file, pdb_file in zip(batch_out_files, batch_out_pdb_files)]
                run_aux["out_pdbs"].extend(batch_out_pdb_files)
                run_aux["input_struct_files"].extend(batch_struct_files)

            pbar.update(B)
    pbar.close()

    # For each input pdb, aggregate all sequence design samples
    preds = defaultdict(dict)
    for pdb, samples_list in input_pdb_to_samples.items():
        for k in samples_list[0].keys():
            preds[pdb][k] = torch.stack([s[k] for s in samples_list])

        # Get sampled sequences for this PDB as a list of strings
        aatype_denoised = preds[pdb]["res_type_pred"]

        pred_seqs = []
        for i in range(aatype_denoised.shape[0]):
            pred_seq = "".join([rc.restypes_with_x[aatype_denoised[i, j]] for j in range(aatype_denoised.shape[1])])
            pred_seqs.append(pred_seq)
        preds[pdb]["pred_seqs"] = pred_seqs

    return preds, run_aux


def get_sd_batch(struct_file_paths: list[str], device: str,
                 data_cfg: DictConfig,
                 parallel_pool: Parallel | None) -> tuple[dict[str, TensorType["b n ..."]],
                                                          list[str],
                                                          list[Dict[str, int]]]:
    if parallel_pool is None:
        # Load PDBs sequentially
        batch_examples, input_structures = zip(*[get_sd_example(struct_file_path, data_cfg) for struct_file_path in struct_file_paths])
    else:
        # Load PDBs in parallel
        batch_examples, input_structures = zip(*parallel_pool(delayed(get_sd_example)(struct_file_path, data_cfg) for struct_file_path in struct_file_paths))

    # Collate examples
    batch = sd_collator(batch_examples)
    batch = to(batch, device)  # move to device

    return batch, input_structures


def get_sd_example(struct_file_path: str, data_cfg: DictConfig) -> tuple[dict[str, TensorType["b n ..."]],
                                                                         dict[str, Any]]:
    """
    Given a structure file path, return a batch of features and the input structure.
    """
    example = {}

    input_data = load_input(struct_file_path)

    # Tokenize structure (no cropping applied)
    tokenized = data_cfg["tokenizer"].tokenize(input_data)
    feats = data_cfg["featurizer"].process(tokenized,
                                           use_auth_as_residx=False,
                                           atoms_per_window_queries=data_cfg["atoms_per_window_queries"],
                                           num_bins=data_cfg["num_bins"])
    feats["coords"] = feats["coords"].squeeze(0)  # remove batch dimension

    # SE3 augmentation for convenience / scaling
    feats["coords"] = atom_center_random_augmentation(feats["coords"], feats["atom_pad_mask"] * feats["atom_resolved_mask"],
                                                      apply_random_augmentation=True,
                                                      translation_scale=1.0,
                                                      return_transforms=False)

    example["pdb_key"] = Path(struct_file_path).stem
    example.update(feats)
    return example, input_data.structure



def parse_fixed_pos_info(batch: Dict[str, TensorType["b ..."]],
                         pdb_names: List[str],
                         batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
                         pos_constraint_df: Optional[pd.DataFrame],
                         verbose: bool = False) -> Dict[str, torch.Tensor]:

    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for the aatype and sidechain overrides.
    - possibly overridden "aatype"


    The pos_constraint_df should have the following format:
    index: PDB name (including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    aatype_override_mask, scn_override_mask = torch.zeros_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        batch["aatype_override_mask"] = aatype_override_mask
        batch["scn_override_mask"] = scn_override_mask
        return batch

    for i, pdb_name in enumerate(pdb_names):
        if verbose:
            print(f"\n======================== {pdb_name} ========================")

        if pdb_name not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {pdb_name}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[pdb_name]
        fixed_pos_seq, fixed_pos_scn = row.get("fixed_pos_seq", np.nan), row.get("fixed_pos_scn", np.nan)  # get fixed positions for this PDB
        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{pdb_name}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(fixed_pos_override_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            for abs_pos_i, aa in zip(abs_pos, override_aatypes):
                batch["aatype"][i, abs_pos_i] = rc.restype_order_with_x[aa]  # override the aatype at the specified position

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{pdb_name}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            aatype_override_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                fixed_seq_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if aatype_override_mask[i, j] else "-" for j in range(aatype_override_mask.shape[1])])
                print(f"Fixed sequence: {fixed_seq_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{pdb_name}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, chain_id_mapping, example["residue_index"], example["chain_index"])
            scn_override_mask[i, abs_fixed_pos_scn] = 1

            if not pd.isna(fixed_pos_override_seq):
                # ensure that we're not fixing sidechains when we override the PDB sequence
                assert scn_override_mask[i, abs_fixed_pos_seq].sum() == 0, "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                fixed_scn_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if scn_override_mask[i, j] else "-" for j in range(scn_override_mask.shape[1])])
                print(f"Fixed sidechains: {fixed_scn_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sidechain positions specified.")

    # Update batch
    batch["aatype_override_mask"] = aatype_override_mask
    batch["scn_override_mask"] = scn_override_mask

    return batch


def parse_pos_restrict_aatype_info(batch: Dict[str, TensorType["b ..."]],
                                  pdb_names: List[str],
                                  batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
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
    B, N = batch["seq_mask"].shape
    K = len(rc.restype_order_with_x)

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["seq_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["seq_mask"].device)

    if verbose:
        print("\n======================== Position-wise amino acid restrictions ========================")

    for i, pdb_name in enumerate(pdb_names):
        if pdb_name not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_name}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_name]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_name}: No amino acid restrictions specified.")
            continue

        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        if verbose:
            print(f"{pdb_name}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(
            pos_restrict_aatype,
            chain_id_mapping,
            example["residue_index"],
            example["chain_index"]
        )

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in rc.restype_order_with_x:
                    allowed_aatype_mask[i, pos_idx, rc.restype_order_with_x[aa]] = 1.0
                else:
                    print(f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_name} at position {pdb_pos[abs_pos.index(pos_idx)]}")

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
    Parse a list of fixed positions in the format ["A1", "A10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
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
        match = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1)
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

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
