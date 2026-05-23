import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from allatom_design.data.transform.preprocess import preprocess_transform
from atomworks.enums import ChainTypeInfo, ChainType
from atomworks.constants import METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import PEPTIDE_MAX_RESIDUES, NUCLEIC_ACID_LIGANDS_MAX_RESIDUES
from allatom_design.utils.metadata_utils import (compute_contacts_to_proteins_per_pdb, 
                                                 assign_nucleic_acid_chain_clusters_per_pdb, 
                                                 sum_nucleic_acid_cluster_residues_per_pdb,
                                                 compute_nuc_cluster_contacts_to_proteins_per_pdb)

from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.checkpoint_utils import get_cfg_from_ckpt

def get_pdb_files(pdb_dir: str | None,
                  pdb_name_list: str | None,
                  pdb_name_ext: str | None = None,
                  n_subsample: int | None = None,
                  # slurm array parameters for parallelization
                  array_id: int | None = None,
                  num_arrays: int | None = None,
                  skip_pdb_names: list[str] | None = None,
                  # recursive search for nested directory structures (e.g. CCD code subfolders)
                  recursive: bool = False,
                  split_by_subfolder: bool = False,
                  # sample index filtering (for files with pattern {CCD}_len_{L}_{IDX}_model_{M}.cif)
                  sample_indices: list[int] | None = None,
                  # sample length filtering (for files with pattern {CCD}_len_{L}_{IDX}_model_{M}.cif)
                  sample_lengths: list[int] | None = None,
                  ) -> list[str]:
    """
    Retrieve a list of PDB files from a directory, either by specifying a list of pdb_names or by getting all files.

    Args:
        pdb_dir: Directory containing PDB files
        pdb_name_list: Optional path to a file containing PDB keys (one per line)
        pdb_name_ext: Optional extension to append to each key when pdb_name_list is provided
        array_id: Set by Slurm array job. Null means run all.
        num_arrays: Number of total arrays. If array_id is null, this can remain 1.
        skip_pdb_names: List of PDB names to skip
        recursive: If True, recursively search subdirectories for files.
            Useful for nested directory structures (e.g. CCD code subfolders).
        split_by_subfolder: If True and array_id is set, split by top-level subfolder
            instead of splitting the flat file list. Each array task gets one or more
            subfolders. Only used when recursive=True.
        sample_indices: Optional list of sample indices to keep. Filters files whose
            filename matches pattern {PREFIX}_{IDX}_model_{M}.ext, keeping only those
            where IDX is in sample_indices.
        sample_lengths: Optional list of sample lengths to keep. Filters files whose
            filename matches pattern {PREFIX}_len_{L}_{IDX}_model_{M}.ext, keeping only
            those where L is in sample_lengths.

    Returns:
        List of PDB file paths, naturally sorted if retrieving all files

    Raises:
        ValueError: If no PDB files are found in the directory when pdb_name_list is None
    """
    # Read in PDB files from directory or list of PDB names
    if pdb_name_list is not None:
        if isinstance(pdb_name_list, np.ndarray): 
            pdb_name_list = pdb_name_list.tolist()
            pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_name_list]
            pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
            print(f"Found {len(pdb_files)} PDB files from key list")
        else:
            # get PDBs with keys in the list
            with open(pdb_name_list, "r") as f:
                pdb_names = f.read().splitlines()
            if pdb_name_ext:
                # replace extension with pdb_name_ext
                pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_names]
            pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
            print(f"Found {len(pdb_files)} PDB files from key list")
    elif recursive:
        # Recursively search subdirectories for files
        if split_by_subfolder and array_id is not None:
            # Split by top-level subfolder: each array task processes one or more subfolders
            subfolders = natsorted([
                d for d in Path(pdb_dir).iterdir() if d.is_dir()
            ])
            print(f"Found {len(subfolders)} subfolders in {pdb_dir}")
            
            chunk_size = math.ceil(len(subfolders) / num_arrays)
            start_idx = array_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(subfolders))
            selected_subfolders = subfolders[start_idx:end_idx]
            print(f"Array {array_id}/{num_arrays}: processing subfolders {[s.name for s in selected_subfolders]}")
            
            # Collect all files from selected subfolders
            pdb_files = []
            for subfolder in selected_subfolders:
                files = natsorted([str(f) for f in subfolder.iterdir() if f.is_file()])
                pdb_files.extend(files)
        else:
            # Flat recursive search across all subdirectories
            pdb_files = natsorted([
                str(f) for f in Path(pdb_dir).rglob("*") if f.is_file()
            ])
        
        # Filter by extension if pdb_name_ext is provided
        if pdb_name_ext:
            pdb_files = [f for f in pdb_files if f.endswith(pdb_name_ext)]
        
        print(f"Found {len(pdb_files)} PDB files recursively in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found recursively in directory {pdb_dir}")
    else:
        # get all PDBs in the directory
        pdb_files = natsorted(list(glob.glob(f"{pdb_dir}/*")))
        
        # Filter by extension if pdb_name_ext is provided
        if pdb_name_ext:
            pdb_files = [f for f in pdb_files if f.endswith(pdb_name_ext)]
        else:
            # Filter out non-structure files (e.g. .pt, .pkl, .json)
            supported_exts = {".pdb", ".cif", ".mmcif", ".ent"}
            pdb_files = [f for f in pdb_files if Path(f).suffix.lower() in supported_exts]
        
        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {pdb_dir}")

    # Skip existing PDBs
    if skip_pdb_names is not None:
        skip_pdb_names = set(skip_pdb_names)
        pdb_files = [f for f in pdb_files if Path(f).name not in skip_pdb_names]

    # Filter by sample indices (for files with pattern {PREFIX}_{IDX}_model_{M}...)
    if sample_indices is not None:
        sample_idx_pattern = re.compile(r"_(\d+)_model_\d+")
        filtered_files = []
        for f in pdb_files:
            match = sample_idx_pattern.search(Path(f).name)
            if match and int(match.group(1)) in sample_indices:
                filtered_files.append(f)
        print(f"Filtered by sample_indices {sample_indices}: {len(pdb_files)} -> {len(filtered_files)} files")
        pdb_files = filtered_files

    # Filter by sample lengths (for files with pattern {PREFIX}_len_{L}_{IDX}_model_{M}...)
    if sample_lengths is not None:
        sample_len_pattern = re.compile(r"_len_(\d+)_")
        filtered_files = []
        for f in pdb_files:
            match = sample_len_pattern.search(Path(f).name)
            if match and int(match.group(1)) in sample_lengths:
                filtered_files.append(f)
        print(f"Filtered by sample_lengths {sample_lengths}: {len(pdb_files)} -> {len(filtered_files)} files")
        pdb_files = filtered_files

    # Parallelization: split PDB files into chunks based on array id
    # (skip if already split by subfolder above)
    if array_id is not None and not (recursive and split_by_subfolder):
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # Optionally take a random subset, preserving order
    if n_subsample is not None:
        n_subsample = min(n_subsample, len(pdb_files))
        chosen_indices = sorted(np.random.choice(len(pdb_files), n_subsample, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    print(f"Using {len(pdb_files)} PDB files")

    return pdb_files


def get_training_checkpoints(
    denoiser_train_dir: str,
    model_type: str,
    eval_every_n_ckpts: int = 1,
    start_step: int | None = None,
    end_step: int | None = None,
    use_ema: bool = False,
    eval_last_ckpt: bool = True,
) -> list[str]:
    """
    Get model checkpoints from a training directory, preferring EMA checkpoints if available.

    Args:
        denoiser_train_dir: Path to the denoiser training directory
        model_type: Either "atom_denoiser" or "seq_denoiser"
        eval_every_n_ckpts: Only evaluate every nth checkpoint
        start_step: Optional starting step to filter checkpoints (skip checkpoints before this step)
        end_step: Optional ending step to filter checkpoints (skip checkpoints after this step)
        use_ema: Whether to use EMA checkpoints
        eval_last_ckpt: Always include the last checkpoint even if not selected by eval_every_n_ckpts

    Returns:
        List of checkpoint paths, sorted by step/epoch
    """
    # Map model type to checkpoint prefix
    prefix_map = {"atom_denoiser": "ad", "seq_denoiser": "sd"}
    prefix = prefix_map.get(model_type)
    if prefix is None:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'atom_denoiser' or 'seq_denoiser'")

    # Check for EMA checkpoints
    ema_ckpt_dir = f"{denoiser_train_dir}/checkpoints/ema"
    non_ema_ckpt_dir = f"{denoiser_train_dir}/checkpoints"
    if use_ema:
        if Path(ema_ckpt_dir).exists():
            print(f"Using EMA checkpoints from {ema_ckpt_dir}")
            pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)-ema(\\d+\\.\\d+)\\.ckpt$")
            ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
        else:
            print(f"Warning: No EMA checkpoints found in {ema_ckpt_dir}, using non-EMA checkpoints")
            pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
            ckpts = glob.glob(f"{non_ema_ckpt_dir}/*.ckpt")
    else:
        print(f"Using non-EMA checkpoints from {non_ema_ckpt_dir}")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{non_ema_ckpt_dir}/*.ckpt")

    # Filter and sort checkpoints
    all_ckpts = natsorted([ckpt for ckpt in ckpts if pattern.search(Path(ckpt).name)])    

    # Filter by start_step and end_step if provided
    if start_step is not None or end_step is not None:
        filtered_ckpts = []
        for ckpt in all_ckpts:
            match = pattern.search(Path(ckpt).name)
            if match:
                global_step = int(match.group(1))
                if (start_step is None or global_step >= start_step) and (end_step is None or global_step <= end_step):
                    filtered_ckpts.append(ckpt)
            else:
                raise ValueError(f"Unexpected checkpoint filename: {Path(ckpt).name}")
    else:
        filtered_ckpts = all_ckpts

    ckpts = filtered_ckpts[::eval_every_n_ckpts]

    # Include the last checkpoint if eval_last_ckpt is True and it's not already included
    if eval_last_ckpt and all_ckpts and all_ckpts[-1] not in ckpts:
        ckpts.append(all_ckpts[-1])

    return ckpts, pattern

def get_seq_des_model(cfg: DictConfig = None,
                      device: str = None) -> dict[str, Any]:
    """
    Load in a sequence design model.
    Example config:

    seq_des_cfg:
        # MPNN args
        model_name: "atom_mpnn"  # ["atom_mpnn"]
        denoiser_train_dir: /path/to/denoiser_train_dir
        ckpt_cfg:
            start_step: 22500
            end_step: 22500
            eval_every_n_ckpts: 1
            eval_last_ckpt: false
            use_ema: false
        atom_mpnn:
            ckpt_path: null
            sampling_cfg: /path/to/sampling_cfg.yaml
            overrides:
                batch_size: 1
                num_seqs_per_pdb: 1  # number of sequences to sample per pdb
                omit_aas: null  # exclude certain aatypes globally, e.g. ["C", "G"]. "X" is always excluded.
                noise_labels: null
                num_workers: ${num_workers}
                use_potts_sampling: true
                ligand_conditioning: true
                potts_sampling_cfg:
                    potts_sweeps: 500
                    lcp_expand_edge_idx_fix: true
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


def wandb_setup(
    base_out_dir: str,
    no_wandb: bool,
    project: str | None,
    wandb_id: str | None,
    group: str | None,
    exp_name: str | None,
    cfg_dict: dict = None,
) -> str:
    """
    Set up Weights & Biases (wandb) tracking and return the log directory.
    Log directory is set to base_out_dir/exp_name.

    Args:
        no_wandb: If True, disable wandb logging
        project: wandb project name
        wandb_id: wandb entity ID
        group: Group name for the experiment
        exp_name: Name of the experiment
        base_out_dir: Base output directory for logs
        cfg_dict: Configuration dictionary to log

    Returns:
        Path: Log directory path
    """
    if exp_name is None:
        exp_name = "debug"

    # Set up log directory
    log_dir = str(Path(base_out_dir, exp_name))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(base_out_dir, "wandb"))
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(base_out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=project,
            entity=wandb_id,
            group=group,
            name=exp_name,
            config=cfg_dict,
            dir=wandb_dir,
        )
        
        # Define custom x-axis metric to allow non-monotonic step logging
        # This is needed when evaluating multiple checkpoints in different phases
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")

    return log_dir


def get_conformer_dirs(conformer_dir: str,
                       pdb_name_list: str | list[str] | None,
                       # slurm array parameters for parallelization
                       array_id: int | None,
                       num_arrays: int | None,
                       # other options
                       use_lowercase_pdb_names: bool = False,
                       ) -> list[str]:
    """
    Get a list of conformer directories from a directory, either by specifying a list of pdb_names or by getting all files.

    pdb_name_list can be a list of pdb names or a path to a file containing a list of pdb names.
    """
    if pdb_name_list is not None:
        # get conformer directories corresponding to pdb_names in the list
        if isinstance(pdb_name_list, str):
            with open(pdb_name_list, "r") as f:
                pdb_names = f.read().splitlines()
        else:
            pdb_names = pdb_name_list
        if use_lowercase_pdb_names:
            # helpful since outputs of partial diffusion are all lowercase
            pdb_names = [pdb_name.lower() for pdb_name in pdb_names]
        conformer_dirs = [f"{conformer_dir}/{Path(pdb_name).stem}" for pdb_name in pdb_names]
    else:
        # get all directories in the conformer_dir
        conformer_dirs = natsorted(list(glob.glob(f"{conformer_dir}/*")))
        conformer_dirs = [conformer_dir for conformer_dir in conformer_dirs if Path(conformer_dir).is_dir()]

    # Parallelization: split PDB files into chunks based on array id
    if array_id is not None:
        array_id = array_id
        num_arrays = num_arrays
        chunk_size = math.ceil(len(conformer_dirs) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(conformer_dirs))
        conformer_dirs = conformer_dirs[start_idx:end_idx]

    print(f"Using {len(conformer_dirs)} conformer directories")
    return conformer_dirs


def process_conformer_dirs(conformer_dirs: list[str],
                           max_num_conformers: int | None,
                           include_primary_conformer: bool,
                           processed_struct_dir: str,
                           pdb_processing_cfg: DictConfig,
                           ignore_missing_primary_conformer: bool = False,
                           return_original_conformer_files: bool = False,
                           ) -> dict[str, list[str]] | tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Process PDB/CIF structures in all conformer directories.

    If max_num_conformers is None, we will include all conformers found in the conformer directories.
    If include_primary_conformer is True, we will also include the primary conformer, which must share the same PDB name as the conformer directory (either .pdb or .cif).

    For each conformer directory, we will grab all PDB/CIF files, natsort them, and take until we have max_num_conformers files (including the primary conformer if include_primary_conformer is True).

    Returns:
        List of processed structure files, one per conformer directory.
    """
    # First, collect a list of conformers for each PDB
    pdb_to_conformer_list = defaultdict(list)  # maps from a given pdb name to its conformer structure files
    for conformer_dir in conformer_dirs:
        pdb_name = Path(conformer_dir).name
        all_conformers = natsorted(glob.glob(f"{conformer_dir}/*.pdb") + glob.glob(f"{conformer_dir}/*.cif"))

        if max_num_conformers is None:
            max_num_conformers = len(all_conformers)

        # Try to find primary conformer with either .cif or .pdb extension
        primary_conformer_cif = f"{conformer_dir}/{pdb_name}.cif"
        primary_conformer_pdb = f"{conformer_dir}/{pdb_name}.pdb"

        if Path(primary_conformer_cif).exists():
            primary_conformer = primary_conformer_cif
        elif Path(primary_conformer_pdb).exists():
            primary_conformer = primary_conformer_pdb
        elif ignore_missing_primary_conformer:
            print(f"Warning: Primary conformer not found for {pdb_name}, defaulting to first conformer in natsorted list {all_conformers[0]}")
            primary_conformer = None
        else:
            raise FileNotFoundError(f"Primary conformer not found for {pdb_name}. Expected either {primary_conformer_cif} or {primary_conformer_pdb}")

        if primary_conformer is not None:
            all_conformers.remove(primary_conformer)

        # Then, take the first max_num_conformers conformers (including the primary conformer if include_primary_conformer is True)
        if include_primary_conformer and primary_conformer is not None:
            conformers = [primary_conformer] + all_conformers[:max_num_conformers - 1]
        else:
            conformers = all_conformers[:max_num_conformers]
        pdb_to_conformer_list[pdb_name].extend(conformers)

    # To process PDB files, we flatten all conformers for each PDB into a single list
    all_confs_flat = [c for _, group in pdb_to_conformer_list.items() for c in group]
    processed_flat = process_pdb_files(all_confs_flat, processed_struct_dir=processed_struct_dir, **pdb_processing_cfg, keep_order=True)

    # Create a mapping from conformer file to processed structure file
    conf_to_processed_file = {}
    for conf_file, processed_file in zip(all_confs_flat, processed_flat):
        conf_to_processed_file[conf_file] = processed_file
        if processed_file is None:
            # this structure failed to process
            continue

        # sanity check the processed structure file name (making sure the original order was preserved)
        expected_processed_name = Path(conf_file).with_suffix(".npz").name.lower()
        assert Path(processed_file).name == expected_processed_name, f"Processed structure file name mismatch: {processed_file} != {expected_processed_name}"

    # Map from PDB name to valid processed structure files
    pdb_to_processed_conformers = defaultdict(list)
    for pdb_name, conformers in pdb_to_conformer_list.items():
        # filter out conformers that failed to process
        pdb_to_processed_conformers[pdb_name].extend([conf_to_processed_file[k] for k in conformers if conf_to_processed_file[k] is not None])

    # Return original conformer files if requested
    if return_original_conformer_files:
        processed_file_to_conf = {v: k for k, v in conf_to_processed_file.items()}
        pdb_to_conformer_files = {k: [processed_file_to_conf[v_i] for v_i in v] for k, v in pdb_to_processed_conformers.items()}
        return pdb_to_processed_conformers, pdb_to_conformer_files

    return pdb_to_processed_conformers


def get_ensemble_constraint_df(pos_constraint_df: pd.DataFrame,
                               pdb_to_processed_conformers: dict[str, list[str]],
                               ) -> pd.DataFrame:
    """
    Expand a pos_constraint_df to include all conformers
    """
    # expand pdb_key to all conformers
    pos_constraint_df["pdb_key"] = pos_constraint_df["pdb_key"].str.lower()
    conformer_dfs = []
    for pdb_key in pos_constraint_df["pdb_key"].unique():
        if pdb_key not in pdb_to_processed_conformers:
            continue
        conformer_df = pos_constraint_df[pos_constraint_df["pdb_key"] == pdb_key]
        conformer_df = pd.concat([conformer_df] * len(pdb_to_processed_conformers[pdb_key]), ignore_index=True)
        conformer_df["pdb_key"] = [Path(x).stem for x in pdb_to_processed_conformers[pdb_key]]
        conformer_dfs.append(conformer_df)
    pos_constraint_df = pd.concat(conformer_dfs)
    return pos_constraint_df
