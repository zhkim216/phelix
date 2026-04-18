"""
Make positional constraint DataFrame for ligand pocket or scaffold regions.

Usage:
    python -m allatom_design.eval.sampling.make_pos_constraint_df
    
This script reads CIF files, annotates ligand pockets, and creates a DataFrame
with positional constraints in the format "A1-10,B5-8" for either:
- pocket regions (residues within pocket_distance of ligands)
- scaffold regions (residues NOT within pocket_distance of ligands)
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from allatom_design.eval.eval_utils.seq_des_utils import make_pos_constraint_df    


@hydra.main(config_path="../../configs/eval/sampling", config_name="make_pos_constraint_df", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Create positional constraint DataFrame for ligand pocket or scaffold regions.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.sampling_inputs_csv is not None:
        sampling_inputs_df = pd.read_csv(cfg.sampling_inputs_csv)
    else:
        sampling_inputs_df = None
                                
    if not cfg.source_is_designed:
        cif_parse_cfg = cfg.cif_cfg.parse.native
        preprocess_cfg = cfg.preprocess_cfg.native        
    else:
        cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg = cfg.preprocess_cfg.designed_samples        
                 
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
            pdb_cfg=cfg.pdb_cfg,
            sampling_inputs_df=sampling_inputs_df,
            output_path=str(output_path),
            pocket_distance=cfg.pocket_distance,
            constraint_type=constraint_type,
            cif_parse_cfg=cif_parse_cfg,
            preprocess_cfg=preprocess_cfg,
            sample_is_designed=cfg.get("source_is_designed", False),
            debug=cfg.get("debug", False),
            num_debug_samples=cfg.get("num_debug_samples", 5),
            save_ligand_mpnn_csv=cfg.get("save_ligand_mpnn_csv", True),
            use_pseudocb_for_pocket_annotation=cfg.get("use_pseudocb_for_pocket_annotation", False),
            num_workers=cfg.get("num_workers", 1),
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
