import glob
import math
import os
import pickle
import re
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig
from tqdm import tqdm

from functools import partial
from p_tqdm import p_umap
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    Resource, fetch, pdb_to_mmcif, process_structure, PDB)


def get_pdb_files(pdb_dir: str,
                  pdb_name_list: str | None,
                  pdb_name_ext: str | None = None,
                  n_subsample: int | None = None,
                  # slurm array parameters for parallelization
                  array_id: int | None = None,
                  num_arrays: int | None = None,
                  skip_pdb_names: list[str] | None = None,
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

        # if providing a pdb manifest, set options here
        manifest_kwargs:
            pdb_manifest_csv: Optional path to a CSV file containing PDB keys and other metadata


    Returns:
        List of PDB file paths, naturally sorted if retrieving all files

    Raises:
        ValueError: If no PDB files are found in the directory when pdb_name_list is None
    """
    # Read in PDB files from directory or list of PDB names
    if pdb_name_list is not None:
        # get PDBs with keys in the list
        with open(pdb_name_list, "r") as f:
            pdb_names = f.read().splitlines()
        if pdb_name_ext:
            # replace extension with pdb_name_ext
            pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_names]
        pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
        print(f"Found {len(pdb_files)} PDB files from key list")
    else:
        # get all PDBs with .pdb_name_ext extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{pdb_dir}/*")))
        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {pdb_dir}")

    # Skip existing PDBs
    if skip_pdb_names is not None:
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

    # Optionally take a random subset, preserving order
    if n_subsample is not None:
        n_subsample = min(n_subsample, len(pdb_files))
        chosen_indices = sorted(np.random.choice(len(pdb_files), n_subsample, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    print(f"Using {len(pdb_files)} PDB files")

    return pdb_files


def process_pdb_files(pdb_files: list[str],
                      processed_struct_dir: str,
                      num_workers: int,
                      pdb_to_cif_conversion_cfg: DictConfig,
                      ccd_cfg: DictConfig,
                      ) -> list[str]:
    """
    Process PDB files.
    Returns paths to processed structure files (.npz format).
    """
    # Make directories where we'll store preprocessed PDB files
    mmcif_dir = f"{processed_struct_dir}/converted_mmcifs"
    Path(mmcif_dir).mkdir(parents=True, exist_ok=True)

    # Handle PDB -> mmCIF conversion if necessary
    mmcif_files = []
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        if Path(pdb_file).suffix != ".cif":
            # assume PDB file, convert to mmCIF and save to processed_struct_dir/converted_mmcifs
            mmcif_file = Path(mmcif_dir, Path(pdb_file).name.replace(".pdb", ".cif"))
            pdb_to_mmcif(pdb_file, mmcif_file, assign_label_seq_id=pdb_to_cif_conversion_cfg.assign_label_seq_id)
        else:
            mmcif_file = pdb_file
        mmcif_files.append(mmcif_file)

    # Load or seed CCD resource in Redis
    if ccd_cfg.use_redis:
        redis_host, redis_port = "localhost", 7777  # hardcoded but can be changed
        start_redis(redis_host, redis_port, ccd_cfg.software_path, ccd_cfg.ccd_rdb_path)
        resource = Resource(host=redis_host, port=redis_port)
    else:
        resource = pickle.load(open(ccd_cfg.ccd_pkl_path, "rb"))

    # Fetch data
    data = fetch(mmcif_files, max_file_size=None)

    # Process each PDB file
    processed_struct_files = parallel_process_structures(data, resource=resource, outdir=Path(processed_struct_dir), filters=[], clusters={}, num_workers=num_workers)
    return processed_struct_files


def start_redis(redis_host: str, redis_port: int, software_path: str, ccd_rdb_path: str):
    command = [
        f"{software_path}/redis/bin/redis-server",
        "--daemonize", "yes",
        "--dir", str(Path(ccd_rdb_path).parent),
        "--dbfilename", str(Path(ccd_rdb_path).name),
        "--port", str(redis_port),
    ]
    # Start Redis in daemon mode (forks immediately)
    subprocess.run(command, check=True)

    # Poll to see if Redis is accepting connections
    start_time = time.time()
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((redis_host, redis_port))
            sock.close()
            break  # connected successfully
        except ConnectionRefusedError:
            if time.time() - start_time > 60:
                raise TimeoutError("Redis did not start within 60 seconds.")
            time.sleep(1)

    # Now the server is guaranteed to be ready
    print("Redis is up and running.")


def parallel_process_structures(pdbs: list[PDB], resource: Resource, outdir: Path, filters: list, clusters: dict, num_workers: int) -> list[str]:
    """
    Small wrapper around process_structure to parallelize over a list of PDBs. Returns a list of processed structure file paths.
    """
    use_parallel = num_workers > 1
    fn = partial(process_structure, resource=resource, outdir=outdir, filters=filters, clusters=clusters, return_struct_path=True)
    if use_parallel:
        processed_struct_files = p_umap(fn, pdbs, num_cpus=num_workers, desc="Processing mmCIFs")
    else:
        processed_struct_files = [fn(pdb) for pdb in tqdm(pdbs, desc="Processing mmCIFs", total=len(pdbs))]
    return processed_struct_files


def load_pdb_files_from_manifest(pdb_dir: str, **manifest_kwargs) -> list[str]:
    """
    Load PDB files from a manifest CSV file.

    Optional filters:
        phase: str
        subset_length_range: tuple[int, int]
        scrmsd_range: tuple[float, float]
        rel_rog_range: tuple[float, float]
        cluster_sample_first: bool
    """
    pdb_keys_df = pd.read_csv(manifest_kwargs["pdb_manifest_csv"])
    if manifest_kwargs.get("phase") is not None:
        pdb_keys_df = pdb_keys_df[pdb_keys_df["phase"] == manifest_kwargs["phase"]]
    if manifest_kwargs.get("subset_length_range") is not None:
        min_len, max_len = manifest_kwargs["subset_length_range"]
        pdb_keys_df = pdb_keys_df[pdb_keys_df["seq_length"].between(min_len, max_len)]
    if manifest_kwargs.get("scrmsd_range") is not None:
        min_scrmsd, max_scrmsd = manifest_kwargs["scrmsd_range"]
        if min_scrmsd is None:
            min_scrmsd = -np.inf
        if max_scrmsd is None:
            max_scrmsd = np.inf
        pdb_keys_df = pdb_keys_df[pdb_keys_df["sc_ca_rmsd"].between(min_scrmsd, max_scrmsd)]
    if manifest_kwargs.get("rel_rog_range") is not None:
        min_rel_rog, max_rel_rog = manifest_kwargs["rel_rog_range"]
        if min_rel_rog is None:
            min_rel_rog = -np.inf
        if max_rel_rog is None:
            max_rel_rog = np.inf
        pdb_keys_df = pdb_keys_df[pdb_keys_df["rel_rog"].between(min_rel_rog, max_rel_rog)]

    if manifest_kwargs.get("cluster_sample_first") is not None:
        # always take first for reproducibility
        pdb_keys_df = pdb_keys_df.groupby("cluster_id", as_index=False).first().reset_index(drop=True)

    pdb_files = [f"{pdb_dir}/{manifest_kwargs.get('pdb_name_prefix', '')}{pdb_name}" for pdb_name in pdb_keys_df["pdb_name"].tolist()]
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
