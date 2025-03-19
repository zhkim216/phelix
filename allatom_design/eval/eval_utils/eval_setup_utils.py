import glob
import math
import re
from pathlib import Path
import os

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm
import wandb

from allatom_design.data.data import get_length_from_pdb


def get_pdb_files(pdb_dir: str,
                  pdb_key_list: str | None,
                  pdb_key_ext: str | None,
                  subset_length_range: tuple[int, int] | None = None,
                  presort_by_length: bool = False,
                  n_subsample: int | None = None,
                  n_jobs: int = 8,
                  # job array parameters for parallelization
                  array_id: int | None = None,
                  num_arrays: int | None = None,
                  skip_pdb_names: list[str] | None = None,
                  ) -> list[str]:
    """
    Retrieve a list of PDB files from a directory, either by specifying a list of keys or by getting all files.

    Args:
        pdb_dir: Directory containing PDB files
        pdb_key_list: Optional path to a file containing PDB keys (one per line)
        pdb_key_ext: Optional extension to append to each key when pdb_key_list is provided
        array_id: Set by Slurm array job. Null means run all.
        num_arrays: Number of total arrays. If array_id is null, this can remain 1.
        skip_pdb_names: List of PDB names to skip


    Returns:
        List of PDB file paths, naturally sorted if retrieving all files

    Raises:
        ValueError: If no PDB files are found in the directory when pdb_key_list is None
    """
    if pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{pdb_dir}/{key}{pdb_key_ext}" for key in pdb_keys]
        print(f"Found {len(pdb_files)} PDB files from key list")
    else:
        # Get all PDBs with .pdb_key_ext extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{pdb_dir}/*")))
        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {pdb_dir}")

    # Skip existing PDBs
    if skip_pdb_names is not None:
        skip_pdb_names = [f"{Path(pdb_key).stem}{pdb_key_ext}" for pdb_key in skip_pdb_names]  # in case pdb_keys instead of pdb_names is passed in, we add the extension
        skip_pdb_names = set(skip_pdb_names)
        pdb_files = [f for f in pdb_files if Path(f).name not in skip_pdb_names]

    # Parallelization: split PDB files into chunks based on array id
    if array_id is not None:
        array_id = array_id
        num_arrays = num_arrays
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # Handle length-dependent options
    if (presort_by_length) or (subset_length_range is not None):
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_length_from_pdb)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths")
        )
        pdb_to_length = dict(results)

        if subset_length_range is not None:
            # filter to length range (inclusive)
            min_len, max_len = subset_length_range
            pdb_files = [f for f in pdb_files if min_len <= pdb_to_length[f] <= max_len]
            print(f"Subsetted to {len(pdb_files)} PDB files in length range [{min_len}, {max_len}]")

        if presort_by_length:
            # sort by length, longest first
            pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)

    # Optionally take a random subset, preserving order
    if n_subsample is not None:
        chosen_indices = sorted(np.random.choice(len(pdb_files), n_subsample, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    print(f"Using {len(pdb_files)} PDB files")

    return pdb_files


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
    no_wandb: bool,
    out_dir: str,
    project: str = None,
    wandb_id: str = None,
    exp_name: str = None,
    group: str = None,
    cfg_dict: dict = None,
) -> Path:
    """
    Set up Weights & Biases (wandb) tracking and return the log directory.

    Args:
        no_wandb: If True, disable wandb logging
        out_dir: Base output directory for logs
        project: wandb project name
        wandb_id: wandb entity ID
        exp_name: Name of the experiment
        group: Group name for the experiment
        cfg_dict: Configuration dictionary to log

    Returns:
        Path: Log directory path
    """
    if no_wandb:
        log_dir = Path(out_dir, "debug")
    else:
        # Create wandb dir
        wandb_dir = str(Path(out_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=project,
            entity=wandb_id,
            name=exp_name,
            group=group,
            config=cfg_dict,
            dir=wandb_dir,
        )
        log_dir = Path(out_dir, wandb.run.name)  # base log dir

    # Set up out directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    return log_dir
