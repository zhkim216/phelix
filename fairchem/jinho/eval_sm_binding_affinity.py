"""
This script is used to evaluate the binding affinity of small molecules in a protein pocket.
Written by Jinho Kim, 251019
"""

# Import necessary libraries
import pickle
import numpy as np
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.components.calculate.recipes.omol import ligand_pocket
from atomworks.ml.datasets.datasets import PandasDataset
from allatom_design.data.transform.preprocess import preprocess_transform

CIF_PARSE_KWARGS = {
 "add_missing_atoms": True,
 "remove_waters": True,
 "remove_ccds": [
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE",
    "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT",
    "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ",
    "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
    "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL",
    "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH",
    "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE",
    "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA",
    "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP",
    "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP", "PVO", "SAR",
    "SCN", "SEO", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU",
    "TAR", "TBU", "TME", "TRS", "UNK", "UNL", "UNX", "UPL", "URE"
  ], # SEP, TPO are excluded in this list, check atomworks/constants.py
  "fix_ligands_at_symmetry_centers": True,
  "fix_arginines": True,
  "convert_mse_to_met": True,
  "hydrogen_policy": "remove"   
}

# Initialize the dataset in each worker so that the dataset is not pickled
_DATASET: PandasDataset | None = None

def _init_dataset(dataset_cfg: DictConfig):
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg, transform=preprocess_transform())