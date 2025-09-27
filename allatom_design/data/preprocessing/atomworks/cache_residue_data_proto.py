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
import torch

from atomworks.ml.transforms.rdkit_utils import (sample_rdkit_conformer_for_atom_array,
                                                 atom_array_to_rdkit,
                                                 generate_conformers,
                                                 generate_conformers_with_timeout_from_mol,
                                                 )
from atomworks.io.tools.rdkit import (get_morgan_fingerprint_from_rdkit_mol,
                                      atom_array_from_ccd_code,
                                      atom_array_from_rdkit,
                                      get_morgan_fingerprint_from_rdkit_mol,
                                      add_hydrogens,
                                      remove_hydrogens,
                                      )



ccd_codes = ["ALA", "HEM"]
optimize = True

generate_conformers_kwargs = {
    "optimize": optimize,
    "num_threads": 16,    
    "hydrogen_policy": "remove",
}

seed = 0

def get_atom_names_from_rdkit_mol(mol):
    element_counter = Counter()
    atom_names = []
    for idx, rdatom in enumerate(mol.GetAtoms()):
        element_occurence = element_counter[rdatom.GetAtomicNum()]
        element_counter[rdatom.GetAtomicNum()] += 1
        atom_names.append(f"{rdatom.GetSymbol().upper()}{element_occurence}")
    return atom_names



    
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
            mol = atom_array_to_rdkit(atom_array, hydrogen_policy="remove")
            #! (JH) Need hydrogens for accurate stereochemistry assignment
            add_hydrogens(mol)
            Chem.AssignStereochemistryFrom3D(mol)
            mol = remove_hydrogens(mol)
            
            atom_array = atom_array_from_rdkit(mol) # atom_array with removed hydrogens
            residue_data["atom_names"] = atom_array.atom_name #! Only heavy atoms are kept
                        
            mol = generate_conformers_with_timeout_from_mol(mol, n_conformers=3, seed=seed, timeout=(3.0, 1.0), timeout_strategy="subprocess", **generate_conformers_kwargs)
                                                                                                                                
            residue_data["mol"] = mol
            residue_data["fingerprint"] = get_morgan_fingerprint_from_rdkit_mol(mol)
            
            os.makedirs(f"{save_dir}/{ccd_code}", exist_ok=True)
            torch.save(residue_data, f"{save_dir}/{ccd_code}/{ccd_code}.pt")
            logger.info(f"{ccd_code} save done at {save_dir}")
        except Exception as e:
            logger.warning(f"{ccd_code} error: {e}")
    else:
        logger.warning(f"{ccd_code} not found")


