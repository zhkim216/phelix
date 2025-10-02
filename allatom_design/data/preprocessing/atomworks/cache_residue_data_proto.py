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
from rdkit.Chem import Mol
import torch
import numpy as np

from atomworks.constants import AF3_EXCLUDED_LIGANDS, GAP, STANDARD_AA, STANDARD_DNA, STANDARD_RNA, METAL_ELEMENTS
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
from atomworks.ml.transforms.atomize import atomize_by_ccd_name


ccd_codes = ["HIS"]
optimize = True

generate_conformers_kwargs = {
    "optimize": optimize,
    "num_threads": 16,    
    "hydrogen_policy": "remove",
}

atom_array_to_rdkit_conversion_kwargs = {
    "hydrogen_policy": "remove",
    "annotations_to_keep": ["chain_id", "res_id", "res_name", "atom_name", "atom_id", "pn_unit_iid"],
    "sanitize": True,
    "set_coord": True,
}

seed = 0

    
parent_dir = "/home/possu/jinho/allatom-design/atomworks_test/250922/residue_cache_data"
log_file = os.path.join(parent_dir, "run.log")

# Configure logger to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)    
logger = logging.getLogger(__name__)

res_names_to_ignore = STANDARD_AA + STANDARD_RNA + STANDARD_DNA
add_rdkit_molecules_for_atomized_molecules = AddRDKitMoleculesForAtomizedMolecules(generate_conformers_kwargs["hydrogen_policy"])

for ccd_code in ccd_codes: 
    test_dir = ccd_code[0]
    save_dir = f"{parent_dir}/{test_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    residue_data = {}
    residue_data.setdefault("mol", None)    
    residue_data.setdefault("descriptors", None) # Descriptors are generated from neural network potentials, not using it for now
    residue_data.setdefault("atom_names", None)    
    residue_data.setdefault("fingerprint", None)    
    path = Path(f"/home/possu/jinho/datasets/ccd_mirror/{test_dir}/{ccd_code}/{ccd_code}.cif")
    if path.exists():
        try:
            atom_array = atom_array_from_ccd_code(ccd_code)          
            atom_array = atomize_by_ccd_name(atom_array, res_names_to_ignore=res_names_to_ignore)           
            ### For metals and anions, coords are not provided, so we need to set them to 0s.
            if (len(atom_array.element) == 1) and np.all(np.isnan(atom_array.coord)): #! dealing with metals, and anions
                atom_array.coord = np.zeros_like(atom_array.coord)
             
            mol = add_rdkit_molecules_for_atomized_molecules._convert_atom_array_to_rdkit_robust(atom_array, conversion_kwargs=atom_array_to_rdkit_conversion_kwargs)            
            #! (JH) Need hydrogens for accurate stereochemistry assignment
            add_hydrogens(mol)
            Chem.AssignStereochemistryFrom3D(mol)
            mol = remove_hydrogens(mol)
            
            atom_array = atom_array_from_rdkit(mol) # atom_array with removed hydrogens
            residue_data["atom_names"] = atom_array.atom_name #! Only heavy atoms are kept
                        
            mol = generate_conformers_with_timeout_from_mol(mol, ccd_code=ccd_code, n_conformers=50, seed=seed, timeout=(3.0, 1.0), timeout_strategy="subprocess", **generate_conformers_kwargs)
                                                                                                                                
            residue_data["mol"] = mol
            residue_data["fingerprint"] = get_morgan_fingerprint_from_rdkit_mol(mol)
            
            os.makedirs(f"{save_dir}/{ccd_code}", exist_ok=True)
            torch.save(residue_data, f"{save_dir}/{ccd_code}/{ccd_code}.pt")
            logger.info(f"{ccd_code} save done at {save_dir}")
        except Exception as e:
            logger.warning(f"{ccd_code} error: {e}")
    else:
        logger.warning(f"{ccd_code} not found")


