import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb
from natsort import natsorted
from omegaconf import DictConfig


def get_training_checkpoints(
    denoiser_train_dir: str,
    model_type: str,
    eval_every_n_ckpts: int = 1,
    start_step: int | None = None,
    end_step: int | None = None
) -> list[str]:
    """
    Get model checkpoints from a training directory, preferring EMA checkpoints if available.

    Args:
        denoiser_train_dir: Path to the denoiser training directory
        model_type: Either "atom_denoiser" or "seq_denoiser"
        eval_every_n_ckpts: Only evaluate every nth checkpoint
        start_step: Optional starting step to filter checkpoints (skip checkpoints before this step)
        end_step: Optional ending step to filter checkpoints (skip checkpoints after this step)

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
    if Path(ema_ckpt_dir).exists():
        # Use EMA checkpoints if they exist
        print(f"Using EMA checkpoints from {ema_ckpt_dir}")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)-ema(\\d+\\.\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
    else:
        print(f"Using non-EMA checkpoints from {denoiser_train_dir}/checkpoints")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{denoiser_train_dir}/checkpoints/*.ckpt")

    # Filter and sort checkpoints
    ckpts = natsorted([ckpt for ckpt in ckpts if pattern.search(Path(ckpt).name)])[::eval_every_n_ckpts]

    # Filter by start_step and end_step if provided
    if start_step is not None or end_step is not None:
        filtered_ckpts = []
        for ckpt in ckpts:
            match = pattern.search(Path(ckpt).name)
            if match:
                global_step = int(match.group(1))
                if (start_step is None or global_step >= start_step) and (end_step is None or global_step <= end_step):
                    filtered_ckpts.append(ckpt)
            else:
                raise ValueError(f"Unexpected checkpoint filename: {Path(ckpt).name}")
        ckpts = filtered_ckpts

    return ckpts, pattern


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
