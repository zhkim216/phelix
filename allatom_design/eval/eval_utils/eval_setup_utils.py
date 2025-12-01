import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from functools import partial
import pandas as pd
import wandb
from natsort import natsorted
from omegaconf import DictConfig

from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from allatom_design.data.transform.preprocess import preprocess_transform
from atomworks.enums import ChainTypeInfo, ChainType
from atomworks.constants import METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import PEPTIDE_MAX_RESIDUES, NUCLEIC_ACID_LIGANDS_MAX_RESIDUES
from allatom_design.utils.metadata_utils import (compute_contacts_to_proteins_per_pdb, 
                                                 assign_nucleic_acid_chain_clusters_per_pdb, 
                                                 sum_nucleic_acid_cluster_residues_per_pdb,
                                                 compute_nuc_cluster_contacts_to_proteins_per_pdb)


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
        if isinstance(pdb_name_list, np.ndarray): #! From metadata
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
    else:
        # get all PDBs in the directory
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

def get_cached_example_files(cached_example_path: str,
                             pdb_name_list: str | None,
                             pdb_name_ext: str | None = None,
                             n_subsample: int | None = None,
                             # slurm array parameters for parallelization
                             array_id: int | None = None,
                             num_arrays: int | None = None,
                             ) -> list[str]:
    """
    Get a list of cached example files from a directory, either by specifying a list of pdb_names or by getting all files.
    """
    if pdb_name_list is not None:
        if isinstance(pdb_name_list, np.ndarray) or isinstance(pdb_name_list, list): #! From metadata
            if isinstance(pdb_name_list, np.ndarray):
                pdb_name_list = pdb_name_list.tolist()
            pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_name_list]
            pdb_files = [f"{cached_example_path}/{name}" for name in pdb_names]
            print(f"Found {len(pdb_files)} PDB files from key list")
    
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

    ckpts = filtered_ckpts[::eval_every_n_ckpts]
    
    # Include the last checkpoint if eval_last_ckpt is True and it's not already included
    if eval_last_ckpt and all_ckpts and all_ckpts[-1] not in ckpts:
        ckpts.append(all_ckpts[-1])

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

# def filter_validation_set():

# if __name__ == "__main__":
    
#     "NA", "CL", "K", "BR"
    
#     default_cif_parser_args = {
#         "add_missing_atoms": True,
#         "remove_waters": True,
#         "remove_ccds": [
#                         "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE",
#                         "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT",
#                         "BDN", "BEN", "BME", "BO3", "BR", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ",
#                         "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
#                         "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL",
#                         "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
#                         "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH",
#                         "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE",
#                         "MYR", "N", "NA", "NAG", "NH2", "NH4", "NHE", "NO3", "K", "O4B", "OHE", "OLA",
#                         "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP",
#                         "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP", "PVO", "SAR",
#                         "SCN", "SEO", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU",
#                         "TAR", "TBU", "TME", "TRS", "UNK", "UNL", "UNX", "UPL", "URE"
#                       ], 
#         #! LMPNN + AF3 excluded ligands list
#         "fix_ligands_at_symmetry_centers": True,
#         "fix_arginines": True,
#         "convert_mse_to_met": True,
#         "hydrogen_policy": "remove",
#         "add_bond_types_from_struct_conn": ["covale"],        
#     }
    
#     default_data_preprocessor_cfg = {
#         "from_rcsb": True,
#         "build_assembly": "first",
#         "close_distance": 30.0,
#         "contact_distance": 5,
#         "second_shell_distance": 8,
#         "clash_distance": 1.0,
#         "ignore_residues": [],
#         "polymer_pn_unit_limit": 500,
#         **default_cif_parser_args,
#     }
    
#     default_preprocess_transform_cfg = {
#         "undesired_res_names": [],
#         "min_residues_for_polymers": 1,
#     }

    
#     data_preprocessor_cfg = default_data_preprocessor_cfg
#     preprocess_transform_cfg = default_preprocess_transform_cfg
    
#     mmcif_dir = Path("/home/possu/jinho/datasets/val_cifs/lmpnn_val_cifs/pdbs")

#     lmpnn_val_df = pd.read_csv("/home/possu/jinho/datasets/splits/lmpnn_val_list.csv")
#     small_molecule_df = lmpnn_val_df[lmpnn_val_df["type"] == "small_molecule"]

#     pdb_id = "1a28"
#     transformation_id = "1"

#     dp = DataPreprocessor(**data_preprocessor_cfg)
#     cif_path = Path(mmcif_dir, f"{pdb_id}.cif")
#     records, _ = dp.get_rows(cif_path)    
#     metadata_df = pd.DataFrame(records)
    
#     #### Proteins, only consider polypeptide-L chain as a protein
#     protein_chain_type = ChainType.POLYPEPTIDE_L
    
#     ### Peptide-like short-polymer ligands
#     peptide_chain_type = [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L, ChainType.CYCLIC_PSEUDO_PEPTIDE, ChainType.PEPTIDE_NUCLEIC_ACID]
    
#     ### Nucleic acids
#     DNA_chain_type_values = [ChainType.DNA.value]
#     RNA_chain_type_values = [ChainType.RNA.value]
#     RNA_DNA_hybrid_chain_type_values = [ChainType.DNA_RNA_HYBRID.value]
    
#     ### Ligands
#     ligand_chain_types = ChainTypeInfo.NON_POLYMERS
#     ligand_chain_type_values = [chain_type.value for chain_type in ligand_chain_types]
    
#     # add "is_protein" & "is_peptide" column, following the definition in Atomworks
#     metadata_df["q_pn_unit_is_protein"] = (metadata_df["q_pn_unit_type"] == protein_chain_type) & (metadata_df["q_pn_unit_num_resolved_residues"] >= PEPTIDE_MAX_RESIDUES)
#     metadata_df["q_pn_unit_is_peptide"] = (metadata_df["q_pn_unit_type"].isin(peptide_chain_type)) & (metadata_df["q_pn_unit_num_resolved_residues"] < PEPTIDE_MAX_RESIDUES)
    
#     metadata_df["q_pn_unit_is_small_molecule"] = (metadata_df["q_pn_unit_type"].isin(ligand_chain_type_values)) & (metadata_df["q_pn_unit_is_metal"] == False)
    
#     metadata_df["q_pn_unit_is_DNA"] = metadata_df["q_pn_unit_type"].isin(DNA_chain_type_values)
#     metadata_df["q_pn_unit_is_RNA"] = metadata_df["q_pn_unit_type"].isin(RNA_chain_type_values)
#     metadata_df["q_pn_unit_is_RNA_DNA_hybrid"] = metadata_df["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values)
    
#     partial_assign_nucleic_acid_chain_clusters_per_pdb = partial(assign_nucleic_acid_chain_clusters_per_pdb, nucleic_acid_dist_threshold=4.0)
    
#     metadata_df['q_pn_unit_nucleic_acid_chain_cluster'] = metadata_df.groupby('pdb_id', group_keys=False).apply(partial_assign_nucleic_acid_chain_clusters_per_pdb)
#     metadata_df['q_pn_unit_num_resolved_residues_in_nucleic_acid_chain_cluster'] = (metadata_df.groupby('pdb_id', group_keys=False)
#                                                                             .apply(sum_nucleic_acid_cluster_residues_per_pdb)
#                                                                             .astype('int64')
#                                                                             )
    
    
#     tmp_sm = metadata_df.groupby('pdb_id', group_keys=False).apply(compute_contacts_to_proteins_per_pdb)
#     metadata_df['num_contacting_protein'] = tmp_sm['num_contacting_protein']
#     metadata_df['contacting_protein_chains'] = tmp_sm['contacting_protein_chains']
        
#     tmp_nuc = metadata_df.groupby('pdb_id', group_keys=False).apply(compute_nuc_cluster_contacts_to_proteins_per_pdb)
#     na_mask = metadata_df['q_pn_unit_nucleic_acid_chain_cluster'].notna()
#     metadata_df.loc[na_mask, 'num_contacting_protein'] = (
#         tmp_nuc.loc[na_mask, 'num_contacting_protein'].fillna(0).astype('int64')
#     )
#     metadata_df.loc[na_mask, 'contacting_protein_chains'] = (
#         tmp_nuc.loc[na_mask, 'contacting_protein_chains'].fillna("")
#     )

#     print(1)    

#     # # Read in the CIF data.
#     # transformation_id = "1"  # Leep only the first assembly.
#     # cif_parser_args["build_assembly"] = [transformation_id]
#     # input_data = aw_parse(pdb_path, **cif_parser_args)
#     # atom_array_from_cif = input_data["assemblies"][transformation_id][0]  # (1, num_atoms) -> (num_atoms)

#     # Run the preprocessing pipeline on the CIF data.
#     # pipeline = preprocess_transform()
#     # return pipeline(
#     #     data={
#     #         "example_id": Path(pdb_path).stem,
#     #         "atom_array": atom_array_from_cif,
#     #         "chain_info": input_data["chain_info"],
#     #     }
#     # )