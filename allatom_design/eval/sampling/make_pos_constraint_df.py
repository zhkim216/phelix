"""
Make positional constraint DataFrame for ligand pocket or scaffold regions.

Usage:
    python -m allatom_design.eval.sampling.make_pos_constraint_df
    
This script reads CIF files, annotates ligand pockets, and creates a DataFrame
with positional constraints in the format "A1-10,B5-8" for either:
- pocket regions (residues within pocket_distance of ligands)
- scaffold regions (residues NOT within pocket_distance of ligands)
"""

from biotite.structure import AtomArray
from biotite.structure.residues import get_residue_starts
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import hydra

import atomworks.enums as aw_enums

from atomworks.ml.transforms.filters import remove_unresolved_tokens
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets
from allatom_design.eval.eval_utils.seq_des_utils import (
    preprocess_pdb,
    _indices_to_pos_string,
)


def extract_pdb_chain_info_from_metadata(metadata: pd.DataFrame) -> dict[str, dict]:
    """
    Extract pdb_chain_info for each pdb_id from metadata.
    
    Args:
        metadata: DataFrame with columns like q_pn_unit_iid_1, q_pn_unit_is_protein_1, etc.
        
    Returns:
        Dictionary mapping pdb_id to pdb_chain_info dict with keys:
        - protein_pn_unit_iids: list of protein pn_unit_iids
        - ligand_pn_unit_iids: list of ligand pn_unit_iids  
        - ligand_ccd_codes: list of CCD codes for ligands
    """
    pdb_chain_info_dict = {}
    
    for _, row in metadata.iterrows():
        pdb_id = row["pdb_id"]
        
        pdb_chain_info = {
            "protein_chain_iids": [],
            "ligand_chain_iids": [],
            "protein_chain_ids": [],
            "ligand_chain_ids": [],
            "ligand_ccd_codes": []
        }
        
        # Process unit 1 and unit 2
        for suffix in ["1", "2"]:
            pn_unit_iid = row.get(f"q_pn_unit_iid_{suffix}")
            if pn_unit_iid is None or pd.isna(pn_unit_iid):
                continue
                
            chain_iid = pn_unit_iid
            chain_id = pn_unit_iid.split("_")[0]
            
            is_protein = row.get(f"q_pn_unit_is_protein_{suffix}", False)
            is_small_molecule = row.get(f"q_pn_unit_is_small_molecule_{suffix}", False)
            
            if is_protein:
                if chain_iid not in pdb_chain_info["protein_chain_iids"]:
                    pdb_chain_info["protein_chain_iids"].append(chain_iid)
                    pdb_chain_info["protein_chain_ids"].append(chain_id)
            elif is_small_molecule:
                if chain_iid not in pdb_chain_info["ligand_chain_iids"]:
                    pdb_chain_info["ligand_chain_iids"].append(chain_iid)
                    pdb_chain_info["ligand_chain_ids"].append(chain_id)
                    # Get CCD code from non_polymer_res_names
                    ccd_code = row.get(f"q_pn_unit_non_polymer_res_names_{suffix}")
                    if ccd_code is not None and not pd.isna(ccd_code):
                        pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))
                    else:
                        pdb_chain_info["ligand_ccd_codes"].append("")
        
        pdb_chain_info_dict[pdb_id] = pdb_chain_info
    
    return pdb_chain_info_dict


def create_pos_constraint_dict_from_pocket(
    pdb_key: str,
    atom_array: AtomArray,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    receptor_chain_iids: list[str] = None,
    ligand_chain_iids: list[str] = None,
    cif_path: str = None,
    return_ligand_mpnn_format: bool = False,
) -> dict:
    """
    Create a pos_constraint_dict from an atom array based on ligand pocket annotation.
    
    Args:
        pdb_key: Identifier for the PDB entry
        atom_array: AtomArray containing protein and ligand atoms
        pocket_distance: Distance threshold for pocket identification (Angstroms)
        constraint_type: "pocket" to constrain pocket residues, "scaffold" to constrain non-pocket residues
        receptor_chain_iids: List of receptor (protein) chain IIDs
        ligand_chain_iids: List of ligand chain IIDs
        cif_path: Path to the CIF file (required if return_ligand_mpnn_format=True)
        return_ligand_mpnn_format: If True, also include LigandMPNN CSV fields (pdb_path, chains, fixed_residues)
        
    Returns:
        Dictionary with pdb_key, fixed_pos_seq, fixed_pos_scn, and metadata.
        If return_ligand_mpnn_format=True, also includes pdb_path, chains, fixed_residues for LigandMPNN.
    """
    # Annotate ligand pockets
    annotated_atom_array = annotate_ligand_pockets(
        atom_array=atom_array,
        pocket_distance=pocket_distance,
        receptor_chain_iids=receptor_chain_iids,
        ligand_chain_iids=ligand_chain_iids,
        annotation_name="is_ligand_pocket"
    )
    
    # Spread residue-wise: if any atom in a residue is in pocket, mark all atoms in that residue as pocket
    residue_wise_pocket_mask = apply_and_spread_residue_wise(annotated_atom_array, annotated_atom_array.get_annotation("is_ligand_pocket"), function=np.any)
    protein_mask = annotated_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L

    if constraint_type == "pocket":
        constrained_mask = protein_mask & residue_wise_pocket_mask
    elif constraint_type == "scaffold":
        constrained_mask = protein_mask & ~residue_wise_pocket_mask        
    else: 
        raise ValueError(f"Invalid constraint type: {constraint_type}")

    # Get constrained atom array     
    constrained_atom_array = annotated_atom_array[constrained_mask]
    
    # Early return if no constrained residues
    if len(constrained_atom_array) == 0:
        return {
            'pdb_key': pdb_key,
            'fixed_pos_seq': "",
            'fixed_pos_scn': np.nan,
            'pocket_distance': pocket_distance,
            'constraint_type': constraint_type,
            'num_constrained_residues': 0,
        }, {}
    
    # Get residue starts
    res_starts = get_residue_starts(constrained_atom_array)
    chain_ids = constrained_atom_array.chain_id[res_starts]
    res_ids = constrained_atom_array.res_id[res_starts]
    
    result = {
        'pdb_key': pdb_key,
        'fixed_pos_seq': _indices_to_pos_string(chain_ids, res_ids),
        'fixed_pos_scn': np.nan,
        'pocket_distance': pocket_distance,
        'constraint_type': constraint_type,
        'num_constrained_residues': len(res_starts),
    }
    
    # Add LigandMPNN format fields if requested
    results_for_ligand_mpnn = {}
    if return_ligand_mpnn_format:                               
        results_for_ligand_mpnn['pdb_path'] = cif_path if cif_path else ""
        fixed_residues_list = [f"{cid}{rid}" for cid, rid in zip(chain_ids, res_ids)]
        results_for_ligand_mpnn['fixed_residues'] = " ".join(fixed_residues_list)
        # Get chains to parse (protein + ligand)
        protein_chain_ids = list({chain_iid.split("_")[0] for chain_iid in receptor_chain_iids})
        ligand_chain_ids = list({chain_iid.split("_")[0] for chain_iid in ligand_chain_iids})
        results_for_ligand_mpnn['chains'] = ",".join(protein_chain_ids + ligand_chain_ids)                    
                                    
    return result, results_for_ligand_mpnn


def make_pos_constraint_df(
    cif_dir: str,
    pdb_list_file: str = None,
    output_path: str = None,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    data_cfg: DictConfig = None,
    transform_cfg: DictConfig = None,
    pdb_chain_info_dict: dict[str, dict] = None,
    debug: bool = False,
    num_debug_samples: int = 5,
    save_ligand_mpnn_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a positional constraint DataFrame for multiple CIF files.
    
    Args:
        cif_dir: Directory containing CIF files
        pdb_list_file: Text file with list of CIF filenames (one per line). If None, use all CIFs in cif_dir.
        output_path: Path to save the output parquet file
        pocket_distance: Distance threshold for pocket identification
        constraint_type: "pocket" or "scaffold"
        data_cfg: Configuration for CIF parsing
        transform_cfg: Configuration for preprocessing and featurization
        pdb_chain_info_dict: Dictionary mapping pdb_id to pdb_chain_info
        debug: If True, only process num_debug_samples samples
        num_debug_samples: Number of samples to process in debug mode
        save_ligand_mpnn_csv: If True, also save LigandMPNN input CSV
        
    Returns:
        Tuple of (positional constraint DataFrame, LigandMPNN input DataFrame)
    """
    cif_dir = Path(cif_dir)
    
    # Get list of CIF files to process
    if pdb_list_file is not None:
        with open(pdb_list_file, 'r') as f:
            cif_files = [line.strip() for line in f if line.strip()]
        cif_paths = [cif_dir / cif_file for cif_file in cif_files]
    else:
        cif_paths = list(cif_dir.glob("*.cif"))
    
    # Filter existing files
    cif_paths = [p for p in cif_paths if p.exists()]
    
    # Debug mode: limit number of samples
    if debug:
        cif_paths = cif_paths[:num_debug_samples]
        print(f"[DEBUG MODE] Processing only {len(cif_paths)} samples")
    
    print(f"Found {len(cif_paths)} CIF files to process")
    
    rows = []
    failed_pdbs = []
    results_for_ligand_mpnn = []
    
    # Get preprocess config
    preprocess_cfg = transform_cfg.preprocess_cfg if transform_cfg is not None else None
    
    for cif_path in tqdm(cif_paths, desc=f"Processing CIFs ({constraint_type})"):
        # pdb_key is stem of cif_path, e.g., "1a28" from "1a28.cif"
        pdb_key = Path(cif_path).stem        
        
        try:
            # Load CIF file using preprocess_pdb 
            example = preprocess_pdb(
                pdb_path=str(cif_path),
                data_cfg=data_cfg,
                preprocess_transform_cfg=preprocess_cfg,
            )
            
            atom_array = example["atom_array"]
            atom_array = remove_unresolved_tokens(atom_array) 
            #! Remove unresolved tokens before annotating ligand pockets
        
            # Get receptor and ligand chains from pdb_chain_info_dict if available
            if pdb_chain_info_dict is not None and pdb_key in pdb_chain_info_dict:
                chain_info = pdb_chain_info_dict[pdb_key]
                receptor_chain_iids = chain_info["protein_chain_iids"]
                ligand_chain_iids = chain_info["ligand_chain_iids"]                
            else:
                # Fallback: determine chains automatically from atom_array
                protein_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
                non_protein_mask = ~protein_mask
                receptor_chain_iids = list(np.unique(atom_array.chain_iid[protein_mask]))
                ligand_chain_iids = list(np.unique(atom_array.chain_iid[non_protein_mask]))
            
            if len(ligand_chain_iids) == 0:
                print(f"Warning: No ligand found in {pdb_key}, skipping...")
                failed_pdbs.append(pdb_key)
                continue
            
            # Create positional constraint dict
            pos_constraint_dict, ligand_mpnn_dict = create_pos_constraint_dict_from_pocket(
                pdb_key=pdb_key,
                atom_array=atom_array,
                pocket_distance=pocket_distance,
                constraint_type=constraint_type,
                receptor_chain_iids=receptor_chain_iids,
                ligand_chain_iids=ligand_chain_iids,
                cif_path=str(cif_path) if save_ligand_mpnn_csv else None,
                return_ligand_mpnn_format=save_ligand_mpnn_csv,
            )
            
            rows.append(pos_constraint_dict)
            if ligand_mpnn_dict:
                results_for_ligand_mpnn.append(ligand_mpnn_dict)
            
        except Exception as e:
            print(f"Error processing {pdb_key}: {e}")
            failed_pdbs.append(pdb_key)
            continue
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if len(df) > 0:
        df = df.set_index("pdb_key")
    
    # Create LigandMPNN DataFrame
    ligand_mpnn_input_df = pd.DataFrame(results_for_ligand_mpnn) if results_for_ligand_mpnn else pd.DataFrame()
    
    print(f"\nSuccessfully processed {len(df)} CIF files")
    print(f"Failed: {len(failed_pdbs)} CIF files")
    
    if failed_pdbs:
        print(f"Failed PDBs: {failed_pdbs[:10]}{'...' if len(failed_pdbs) > 10 else ''}")
    
    # Save to file if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Drop metadata columns before saving (for minimal version)
        cols_to_drop = ["pocket_distance", "constraint_type", "num_constrained_residues"]
        df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        if output_path.suffix == ".parquet":
            df_to_save.to_parquet(output_path)
        elif output_path.suffix == ".csv":
            df_to_save.to_csv(output_path)
        else:
            # Default to csv
            df_to_save.to_csv(output_path)
        
        print(f"Saved positional constraint DataFrame to {output_path}")
        
        # Also save full version with metadata
        full_output_path = output_path.parent / (output_path.stem + "_full" + output_path.suffix)
        if full_output_path.suffix == ".parquet":
            df.to_parquet(full_output_path)
        elif full_output_path.suffix == ".csv":
            df.to_csv(full_output_path)
        else:
            df.to_parquet(full_output_path)
        
        print(f"Saved full positional constraint DataFrame to {full_output_path}")
        
        # Save LigandMPNN input CSV
        if save_ligand_mpnn_csv and len(ligand_mpnn_input_df) > 0:
            ligand_mpnn_csv_path = output_path.parent / (output_path.stem + "_for_ligandmpnn.csv")
            # Select only required columns for LigandMPNN: pdb_path, chains, fixed_residues
            ligand_mpnn_df_to_save = ligand_mpnn_input_df[['pdb_path', 'chains', 'fixed_residues']]
            ligand_mpnn_df_to_save.to_csv(ligand_mpnn_csv_path, index=False)
            print(f"Saved LigandMPNN input CSV to {ligand_mpnn_csv_path}")
    
    return df, ligand_mpnn_input_df


@hydra.main(config_path="../../configs_local/eval/sampling", config_name="make_pos_constraint_df", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Create positional constraint DataFrame for ligand pocket or scaffold regions.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not cfg.load_designed_samples:
        data_cfg = cfg.data_cfg_for_design
        transform_cfg = cfg.transform_cfg_for_design
    else:
        data_cfg = cfg.data_cfg_for_designed_samples
        transform_cfg = cfg.transform_cfg_for_designed_samples
    
    # Load metadata and extract pdb_chain_info
    metadata = pd.read_parquet(cfg.metadata_path)
    print(f"Loaded metadata with {len(metadata)} entries")
    
    pdb_chain_info_dict = extract_pdb_chain_info_from_metadata(metadata)
    print(f"Extracted pdb_chain_info for {len(pdb_chain_info_dict)} PDBs")
    
    # Determine constraint types to process
    if cfg.constraint_type == "both":
        constraint_types = ["pocket", "scaffold"]
    else:
        constraint_types = [cfg.constraint_type]
    
    for constraint_type in constraint_types:
        print(f"\n{'='*60}")
        print(f"Creating {constraint_type} positional constraint DataFrame")
        print(f"Pocket distance: {cfg.pocket_distance} Å")
        print(f"{'='*60}\n")
        
        if not cfg.debug:
            output_filename = f"pos_constraint_{constraint_type}_{cfg.pocket_distance}A.csv"
        else:
            output_filename = f"debug_pos_constraint_{constraint_type}_{cfg.pocket_distance}A.csv"
        output_path = output_dir / output_filename
        
        df, ligand_mpnn_df = make_pos_constraint_df(
            cif_dir=cfg.cif_dir,
            pdb_list_file=cfg.pdb_list_file,
            output_path=str(output_path),
            pocket_distance=cfg.pocket_distance,
            constraint_type=constraint_type,
            data_cfg=data_cfg,
            transform_cfg=transform_cfg,
            pdb_chain_info_dict=pdb_chain_info_dict,
            debug=cfg.get("debug", False),
            num_debug_samples=cfg.get("num_debug_samples", 5),
            save_ligand_mpnn_csv=cfg.get("save_ligand_mpnn_csv", True),
        )
        
        # Print summary statistics
        if len(df) > 0:
            print(f"\nSummary for {constraint_type}:")
            print(f"  Total entries: {len(df)}")
            print(f"  Entries with constraints: {(df['num_constrained_residues'] > 0).sum()}")
            print(f"  Average constrained residues: {df['num_constrained_residues'].mean():.1f}")
            print(f"  Max constrained residues: {df['num_constrained_residues'].max()}")
            print(f"\nSample entries:")
            print(df.head())
        
        if len(ligand_mpnn_df) > 0:
            print(f"\nLigandMPNN input CSV summary:")
            print(f"  Total entries: {len(ligand_mpnn_df)}")
            print(f"  Entries with fixed_residues: {(ligand_mpnn_df['fixed_residues'] != '').sum()}")
            print(f"\nSample LigandMPNN entries:")
            print(ligand_mpnn_df[['pdb_path', 'chains', 'fixed_residues']].head())
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
