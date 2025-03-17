import glob
import re
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm

from allatom_design.data.data import get_length_from_pdb


def get_pdb_files(pdb_dir: str,
                  pdb_key_list: str | None,
                  pdb_key_ext: str | None,
                  subset_length_range: tuple[int, int] | None = None,
                  presort_by_length: bool = False,
                  n_subsample: int | None = None,
                  n_jobs: int = 8
                  ) -> list[str]:
    """
    Retrieve a list of PDB files from a directory, either by specifying a list of keys or by getting all files.

    Args:
        pdb_dir: Directory containing PDB files
        pdb_key_list: Optional path to a file containing PDB keys (one per line)
        pdb_key_ext: Optional extension to append to each key when pdb_key_list is provided

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
        ckpts = glob.glob(f"{denoiser_train_dir / 'checkpoints' / '*.ckpt'}")

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