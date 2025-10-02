"""
Codes for caching residue-level data for AtomWorks.
Written by Jinho Kim
"""

import os
import torch
from pathlib import Path
import logging
import sys
from collections import Counter
from rdkit import Chem
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import glob
import torch
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA

from atomworks.ml.transforms.atomize import atomize_by_ccd_name
from atomworks.ml.transforms.rdkit_utils import (sample_rdkit_conformer_for_atom_array,
                                                 atom_array_to_rdkit,
                                                 generate_conformers,
                                                 generate_conformers_with_timeout_from_mol,
                                                 AddRDKitMoleculesForAtomizedMolecules
                                                 )
from atomworks.io.tools.rdkit import (get_morgan_fingerprint_from_rdkit_mol,
                                      atom_array_from_ccd_code,
                                      atom_array_from_rdkit,
                                      get_morgan_fingerprint_from_rdkit_mol,
                                      add_hydrogens,
                                      remove_hydrogens,
                                      )

from allatom_design.data.preprocessing.atomworks.sharding_utils import \
    take_shard, use_sharding

@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="cache_residue_data", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Cache residue-level data for AtomWorks.
    Not written in the original Atomworks code, but written by Jinho Kim.
    """    
    # Create dataset directory + shard dir
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "cached_residue_data"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # Only one shard writes the canonical config.yaml to avoid races
    if not use_sharding(cfg.shard_id, cfg.num_shards) or (cfg.shard_id == 0):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_dict, f)
            
    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem
    
    # Get all CIF paths then take this shard's slice
    ccd_paths_all = get_ccd_paths(cfg.ccd_dir, cfg.max_file_size)
    exclude_path = str(Path(cfg.ccd_dir) / "components.cif")
    ccd_paths_all = [path for path in ccd_paths_all if path != exclude_path]
    
    ccd_paths = take_shard(ccd_paths_all, shard_id=cfg.shard_id, num_shards=cfg.num_shards)    
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(ccd_paths)} ccd files.")

    if use_parallel:         
        worker = partial(
            cache_residue_data,
            seeds=cfg.seeds,
            out_dir=shard_dir,
            cfg=cfg,
            logger=None,
            
        )
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            list(tqdm(executor.map(worker, ccd_paths), total=len(ccd_paths), desc="Caching residues"))
    else:
        worker = partial(
            cache_residue_data,
            seeds=cfg.seeds,
            out_dir=shard_dir,
            cfg=cfg,
            logger=None,
        )
        for path in tqdm(ccd_paths, desc="Caching residues"):
            worker(path)
            
    

def cache_residue_data(ccd_path: str = None,
                       seeds: list[int] = [6, 36, 216],
                       out_dir: str = None,
                       cfg: DictConfig = None,
                       logger: logging.Logger = None,
                       ):
    """
    Cache residue-level data for AtomWorks.
    This is not the most optimal way to do this, but it works.
    """
    res_names_to_ignore = STANDARD_AA + STANDARD_RNA + STANDARD_DNA
    
    # Paths
    ccd_path = Path(ccd_path)
    ccd_code = ccd_path.stem
    save_dir = Path(f"{out_dir}/{ccd_code}/{ccd_code}.pt")
    # if save_dir.exists():
    #     return
    
    # Make save directory
    os.makedirs(save_dir.parent, exist_ok=True)
    
    # Get generate conformers kwargs from cfg
    generate_conformers_kwargs = dict(getattr(cfg, "generate_conformers_kwargs", {}) or {})
    atom_array_to_rdkit_conversion_kwargs = dict(getattr(cfg, "atom_array_to_rdkit_conversion_kwargs", {}) or {})
    add_rdkit_molecules_for_atomized_molecules = AddRDKitMoleculesForAtomizedMolecules(generate_conformers_kwargs["hydrogen_policy"])
    
    # Initialize residue data
    residue_data = {}
    residue_data.setdefault("mol", None)    
    residue_data.setdefault("descriptors", None) #! (JH) Descriptors are generated from neural network potentials, not using it for now
    residue_data.setdefault("atom_names", None)    
    residue_data.setdefault("fingerprint", None)    
        
    # Local logging helper must be defined before first use
    def _logger_wrapper(message: str):
        if logger is not None:
            logger.info(message)
        else:
            print(f"{message}")

    if ccd_path.exists():
        try:
            # Convert CCD code to biotite atom array
            atom_array = atom_array_from_ccd_code(ccd_code)
            atom_array = atomize_by_ccd_name(atom_array, res_names_to_ignore=res_names_to_ignore)           
            
            ### For metals and anions, coords are not provided, so we need to set them to 0s.
            if (len(atom_array.element) == 1) and np.all(np.isnan(atom_array.coord)): #! dealing with metals, and anions
                atom_array.coord = np.zeros_like(atom_array.coord)
                                    
            # Convert biotite atom array to rdkit molecule
            mol = add_rdkit_molecules_for_atomized_molecules._convert_atom_array_to_rdkit_robust(atom_array, conversion_kwargs=atom_array_to_rdkit_conversion_kwargs)            
            add_hydrogens(mol)
            Chem.AssignStereochemistryFrom3D(mol)
            mol = remove_hydrogens(mol)
            
            atom_array = atom_array_from_rdkit(mol) # atom_array with removed hydrogens
            residue_data["atom_names"] = atom_array.atom_name #! Only heavy atoms are kept here
            
            # Generate conformers with retries
            n_trials = int(getattr(cfg, "n_trials", 3))
            timeout_offset = float(getattr(cfg, "timeout_offset", 3.0))
            timeout_slope = float(getattr(cfg, "timeout_slope", 1.0))
            seeds = seeds
            
            for trial in range(n_trials):
                try:
                    mol = generate_conformers_with_timeout_from_mol(
                        mol,
                        ccd_code=ccd_code,
                        n_conformers=cfg.num_conformers,
                        seed=seeds[trial], #! (JH) different seeds for each trial
                        timeout=(timeout_offset, timeout_slope),
                        timeout_strategy="subprocess",
                        **generate_conformers_kwargs,
                    )
                    _logger_wrapper(f"{ccd_code} conformer gen succeeded at trial {trial}")
                    break
                except Exception as e:
                    timeout_slope = timeout_slope + 3.0
                    _logger_wrapper(
                        f"{ccd_code} conformer gen failed at trial {trial}: {e}; retry with timeout=(offset={timeout_offset}, slope={timeout_slope})"
                    )
                                                                                                                                
            residue_data["mol"] = mol
            residue_data["fingerprint"] = get_morgan_fingerprint_from_rdkit_mol(mol)
            
            torch.save(residue_data, save_dir)            
            _logger_wrapper(f"{ccd_code} save done at {save_dir}")
            
                
        except Exception as e:
            _logger_wrapper(f"{ccd_code} error: {e}")
            
    else:
        _logger_wrapper(f"{ccd_code} not found")
        
    return

def get_ccd_paths(ccd_dir: str, max_file_size: int | None = None) -> list[str]:
    ccd_paths = [str(p) for p in Path(ccd_dir).rglob("*.cif")]
    if max_file_size is not None:
        original_len = len(ccd_paths)
        ccd_paths = [path for path in ccd_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(ccd_paths)} files due to size.")
    print(f"Found {len(ccd_paths)} ccd files.")
    return ccd_paths


if __name__ == "__main__":
    main()