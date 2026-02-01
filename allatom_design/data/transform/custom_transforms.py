from collections.abc import Sequence
from typing import Any, ClassVar
from typing_extensions import override
from torch import Tensor as TensorType
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import re

import biotite.structure as struc
from biotite.structure import AtomArray
from atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    METAL_ELEMENTS,
    NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA,
    STANDARD_AA_TIP_ATOM_NAMES,
    STANDARD_DNA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)

import atomworks.enums as aw_enums
from atomworks.ml.utils.token import get_token_starts
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms._checks import check_contains_keys, check_is_instance, check_atom_array_annotation
from atomworks.ml.transforms.filters import filter_to_specified_pn_units
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation
from atomworks.ml.utils.token import apply_token_wise, get_af3_token_center_idxs, apply_and_spread_token_wise
from atomworks.ml.conditions.annotator import is_protein_backbone, is_protein_sidechain
from atomworks.ml.utils.token import (
    get_token_count,
    get_token_starts,
    is_glycine,
    is_protein_unknown,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
    spread_token_wise,
)
from atomworks.io.utils.sequence import (
    is_purine,
    is_pyrimidine,
)

import allatom_design.data.const as const
from allatom_design.data.transform.pad import pad_dim
from allatom_design.data.const import TRAINING_SUPPORTED_CHAIN_TYPES
import logging
logger = logging.getLogger(__name__)

# Keep track of the token/atom dimensions of the features for padding & cropping
FEAT_TO_TOKEN_DIM = {
    # Maps feature name to the token dimension
    # token features
    "residue_index": [0],
    "token_index": [0],
    "asym_id": [0],
    "entity_id": [0],
    "sym_id": [0],
    "restype": [0],
    "is_protein": [0],
    "is_rna": [0],
    "is_dna": [0],
    "is_nuc": [0],
    "is_ligand": [0],
    "is_atomized": [0],
    "token_resolved_mask": [0],
    "token_pad_mask": [0],
    "token_chain_type": [0],
    "token_is_polymer": [0],
    "token_is_hetero": [0],
    "token_is_covalent_modification": [0],
    "token_is_ligand_pocket": [0],
    "token_is_protein_chain": [0],
    "token_to_center_atom": [0],
    "tokenwise_atom_idxs": [0],
    "tokenwise_atom_idxs_mask": [0],
    "noised_ca_coords": [0],
    "noised_n_coords": [0],
    "noised_c_coords": [0],
    "noised_o_coords": [0],
    "noised_pseudo_cb_coords": [0],                
    
    # optional features that might not be present
    "seq_cond_mask": [0],
    "token_exists_mask": [0],
}

FEAT_TO_ATOM_DIM = {
    # Maps feature name to the atom dimension
    # atom features
    "coords": [0],
    "noise": [0],
    "noised_coords": [0],
    "atom_to_token_map": [0],
    "atom_resolved_mask": [0],
    "atom_pad_mask": [0],
    "atom_chain_type": [0],
    "atom_is_polymer": [0],
    "atom_is_hetero": [0],
    "atomic_number": [0],
    "atom_charge": [0],
    "atom_is_covalent_modification": [0],
    "atom_is_ligand_pocket": [0],
    "atom_is_protein_chain": [0],    
    "prot_bb_atom_mask": [0],
    "prot_scn_atom_mask": [0],
    "prot_scn_wo_cb_atom_mask": [0],
    "atom_is_atomized": [0],
    
    
    # optional features that might not be present
    "atom_cond_mask": [0],
    "ref_pos": [0],
    "ref_mask": [0],
    "ref_element": [0],
    "ref_charge": [0],
    "ref_atom_name_chars": [0],
    "ref_space_uid": [0],
    "ref_is_atomized_atom_level": [0],
}

# Keep track of data dict keys only included at inference time
# INFERENCE_ONLY_KEYS = ["crop_info", "atom_array", "feat_metadata"]
INFERENCE_ONLY_KEYS = ["crop_info", "atom_array", "feat_metadata"]

class CheckCoordinatesAreNan(Transform):
    """Check if the coordinates are nan."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        if np.isnan(atom_array.coord).all(axis=1).all():
            raise ValueError(f"All coordinates are nan for {data['example_id']}")
        return data

class FeaturizeCoordsAndMasks(Transform):
    """Add coordinates and atom masks to feats."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        feats = data["feats"]

        # Get coordinates
        feats["coords"] = torch.tensor(atom_array.coord)
        if not data["is_inference"]:
            feats["noised_coords"] = feats["coords"] + feats["noise"]
        else:
            feats["noised_coords"] = feats["coords"]

        # Get token and atom resolved masks
        repr_mask = get_af3_token_representative_masks(atom_array)
        feats["token_resolved_mask"] = torch.tensor((atom_array.occupancy > 0)[repr_mask]).float()
        feats["atom_resolved_mask"] = torch.tensor(atom_array.occupancy > 0).float() 
                    
        # Make pad masks
        feats["token_pad_mask"] = torch.ones_like(feats["token_resolved_mask"])
        feats["atom_pad_mask"] = torch.ones_like(feats["atom_resolved_mask"])

        # atomized
        feats["atom_is_atomized"] = torch.tensor(atom_array.atomize).float()

        # Get Chain type features        
        feats["atom_chain_type"] = torch.tensor(atom_array.chain_type).long()
        feats["token_chain_type"] = torch.tensor(atom_array[repr_mask].chain_type)
        feats["atom_is_polymer"] = torch.tensor(atom_array.is_polymer).float()
        feats["token_is_polymer"] = torch.tensor(atom_array[repr_mask].is_polymer).float()        
                
        # hetero flags
        feats["atom_is_hetero"] = torch.tensor(atom_array.hetero).float()
        feats["token_is_hetero"] = torch.tensor(apply_token_wise(atom_array, atom_array.hetero, np.any)).float()
        
        # atomic number and charge
        feats["atomic_number"] = torch.tensor(atom_array.atomic_number).long()
        feats["atom_charge"] = torch.tensor(atom_array.charge).float()
        
        # covalent modification flags
        feats["atom_is_covalent_modification"] = torch.tensor(atom_array.is_covalent_modification).float()
        feats["token_is_covalent_modification"] = torch.tensor(apply_token_wise(atom_array, atom_array.is_covalent_modification, np.any)).float()
    
        # is_ligand_pocket flags
        feats["atom_is_ligand_pocket"] = torch.tensor(atom_array.is_ligand_pocket).float()
        feats["token_is_ligand_pocket"] = torch.tensor(apply_token_wise(atom_array, atom_array.is_ligand_pocket, np.any)).float()

        # Get chain type flags
        polymer_chain_type_enums = [x.value for x in aw_enums.ChainTypeInfo.POLYMERS]
        nucleic_acid_chain_type_enums = [x.value for x in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS]
        non_polymer_chain_type_enums = [x.value for x in aw_enums.ChainTypeInfo.NON_POLYMERS]
        
        # Protein chain flags
        atom_is_protein_chain = np.zeros(len(atom_array), dtype=bool)        
        atom_is_nucleic_acid_chain = np.zeros(len(atom_array), dtype=bool)
        atom_is_metal_chain = np.zeros(len(atom_array), dtype=bool)
        atom_is_small_molecule_chain = np.zeros(len(atom_array), dtype=bool)        
        try:
            for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
                pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
                sel_atom_array = atom_array[pn_unit_mask]
                chain_type = np.unique(sel_atom_array.chain_type)
                if len(chain_type) == 1:
                    if chain_type in polymer_chain_type_enums:
                        if chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value:                    
                            atom_is_protein_chain[pn_unit_mask] = True                                    
                        elif chain_type in nucleic_acid_chain_type_enums:
                            atom_is_nucleic_acid_chain[pn_unit_mask] = True
                    elif np.isin(chain_type, non_polymer_chain_type_enums):
                        if (len(sel_atom_array) == 1):
                            if np.isin(sel_atom_array.element, METAL_ELEMENTS):
                                atom_is_metal_chain[pn_unit_mask] = True
                        else:
                            atom_is_small_molecule_chain[pn_unit_mask] = True                
                elif len(chain_type) > 1: # covalent modification case, e.g. [6, 8]
                    if np.isin(chain_type, non_polymer_chain_type_enums).any():
                        atom_is_small_molecule_chain[pn_unit_mask] = True                    
        except Exception as e:
            print(f"example_id: {data["example_id"]}, {e}")                        
            
        
        token_is_protein_chain = atom_is_protein_chain[repr_mask]
        token_is_nucleic_acid_chain = atom_is_nucleic_acid_chain[repr_mask]
        token_is_metal_chain = atom_is_metal_chain[repr_mask]
        token_is_small_molecule_chain = atom_is_small_molecule_chain[repr_mask]
        
        feats["atom_is_protein_chain"] = torch.tensor(atom_is_protein_chain).float()
        feats["token_is_protein_chain"] = torch.tensor(token_is_protein_chain).float()
        feats["atom_is_nucleic_acid_chain"] = torch.tensor(atom_is_nucleic_acid_chain).float()
        feats["token_is_nucleic_acid_chain"] = torch.tensor(token_is_nucleic_acid_chain).float()
        feats["atom_is_metal_chain"] = torch.tensor(atom_is_metal_chain).float()
        feats["token_is_metal_chain"] = torch.tensor(token_is_metal_chain).float()
        feats["atom_is_small_molecule_chain"] = torch.tensor(atom_is_small_molecule_chain).float()
        feats["token_is_small_molecule_chain"] = torch.tensor(token_is_small_molecule_chain).float()
                
        # atom to tokens map and token to center atom map
        feats["atom_to_token_map"] = feats["atom_to_token_map"].long()
        feats["token_to_center_atom"] = torch.tensor(get_af3_token_center_idxs(atom_array))        

        # protein backbone and sidechain atom masks
        is_prot_bb = (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value) & np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
        is_prot_scn = (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value) & ~np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
        is_prot_scn_wo_cb = (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value) & ~np.isin(atom_array.atom_name, ["N", "CA", "C", "O", "OXT", "CB"])
        feats["prot_bb_atom_mask"] = torch.tensor(is_prot_bb).float()
        feats["prot_scn_atom_mask"] = torch.tensor(is_prot_scn).float()
        feats["prot_scn_wo_cb_atom_mask"] = torch.tensor(is_prot_scn_wo_cb).float()        
                                                                
        # Calculate number of atoms per token
        device = feats["coords"].device
        N_tokens = feats["token_pad_mask"].shape[0]
        n_atoms_per_token = (F.one_hot(feats["atom_to_token_map"], num_classes=N_tokens)).sum(dim=-2)

        # Starting atom index for each token
        tokenwise_atom_idxs = torch.cat([torch.zeros((1,), device=device), n_atoms_per_token.cumsum(dim=-1)[:-1]], dim=-1).long()
        tokenwise_atom_idxs = tokenwise_atom_idxs[..., None] + torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :]
        tokenwise_atom_idxs_mask = torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :] < n_atoms_per_token[..., None]
        tokenwise_atom_idxs = torch.where(tokenwise_atom_idxs_mask, tokenwise_atom_idxs, torch.zeros_like(tokenwise_atom_idxs, dtype=torch.long))
        
        feats["tokenwise_atom_idxs"] = tokenwise_atom_idxs
        feats["tokenwise_atom_idxs_mask"] = tokenwise_atom_idxs_mask                        

        # Get bond features        
        feats["token_bonds"] = torch.tensor(feats["token_bonds"]).float()
        

        # Get ligand related features
        # try:
        #     feats["atom_is_aromatic"] = torch.tensor(atom_array.is_aromatic).float()
        # except: 
        #     # Todo: LMT & DD6 from 6a2w, no attribute error for is_aromatic.
        #     # Todo: Both have hexagon rings, but no aromaticity. Maybe error in parsing?
        #     feats["atom_is_aromatic"] = torch.full((len(atom_array),), False).float()
        #     print(f"Atom array has no attribute 'is_aromatic' for {data['example_id']}")            
                        
        return data
                    

class PadSDFeats(Transform):
    """Pad the token and atom features to the maximum number of tokens and atoms."""

    def __init__(self, max_tokens: int | None, max_atoms: int | None):
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms

    @override
    def forward(self,
                data: dict[str, Any],
                ) -> dict[str, Any]:
        feats = data["feats"]

        # Pad to max tokens if given
        if self.max_tokens is not None:
            token_pad_len = self.max_tokens - len(feats["token_index"])
            if token_pad_len > 0:
                for k, v in FEAT_TO_TOKEN_DIM.items():
                    if k not in feats:
                        continue
                    for dim_to_pad in v:
                        feats[k] = pad_dim(feats[k], dim_to_pad, token_pad_len)

        # Pad to max atoms if given
        if self.max_atoms is not None:
            atom_pad_len = self.max_atoms - len(feats["atom_resolved_mask"])
            if atom_pad_len > 0:
                for k, v in FEAT_TO_ATOM_DIM.items():
                    if k not in feats:
                        continue
                    for dim_to_pad in v:
                        feats[k] = pad_dim(feats[k], dim_to_pad, atom_pad_len)

        return data


class FlattenFeatsDict(Transform):
    """Flatten features into the data dict."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        feats = data.pop("feats")
        for k, v in feats.items():
            data[k] = v
        return data


class CenterRandomAugmentation(Transform):
    """Center the atom array, and optionally apply random rotation and translation.
    If apply_random_augmentation is True, also randomly rotate and translate.
    Update the atom array with the randomly augmented coordinates.
    """

    def __init__(self, apply_random_augmentation: bool, translation_scale: float):
        self.apply_random_augmentation = apply_random_augmentation
        self.translation_scale = translation_scale        

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        
        atom_array = data["atom_array"]
        coords = atom_array.coord
        mask = atom_array.occupancy > 0
        centered_coords = coords.copy()
        centered_coords = masked_center(centered_coords, mask)
        centered_coords = torch.tensor(centered_coords, device = coords.device)
        if self.apply_random_augmentation:
            centered_coords = random_rigid_augmentation(
                centered_coords[None], batch_size=1, s=self.translation_scale
            ).squeeze(0)
                
        data["atom_array"].coord = centered_coords.numpy()

        return data


class AddDataCategory(Transform):
    """Add the sub dataset name to the data dict."""
    def __init__(self):
        pass
    
    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        s = data["example_id"]        
        match = re.search(r"\['[^']+',\s*'([^']+)'\]", s)
        if match:
            dataset_type = match.group(1)  # 'protein_chains' 또는 'complex'
            if dataset_type == "interface":
                data["data_category"] = "interface"
            else:
                data["data_category"] = "protein_monomer_chain"
        else:
            raise ValueError(f"Invalid example_id: {data['example_id']}")
        return data

# class DropOutNonProteinChains(Transform):
#     """Randomly drop out non-protein chains."""
#     def __init__(self, drop_prob: float = 0.1):
#         self.drop_prob = drop_prob

#     @override
#     def forward(self, data: dict[str, Any]) -> dict[str, Any]:
                
#         s = data["example_id"]
#         import re
#         match = re.search(r"\['[^']+',\s*'([^']+)'\]", s)
#         if match:
#             dataset_type = match.group(1)  # 'protein_chains' 또는 'complex'
#             if dataset_type == "complexes":
#                 if len(data["chain_info"].keys()) >= 2:
#                     print(1)
        
        
        
        
        # atom_array = data["atom_array"]
        # return data

class FilterToQueryPNUnits(Transform):
    """Filter the atom array to the query PN units."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        #* From atomworks.ml.datasets.parsers.GenericDFParser: "During VALIDATION, then we do not crop, and query_pn_unit_iids should be None."
        if "query_pn_unit_iids" in data:
            atom_array = filter_to_specified_pn_units(atom_array, data["query_pn_unit_iids"])

        data["atom_array"] = atom_array
        
        return data

class MaskAtomizedTokensInProtein(Transform):
    """Mask atomized tokens from the atom array."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:        
        atom_array = data["atom_array"]        
        isin_protein = np.isin(atom_array.chain_type, aw_enums.ChainTypeInfo.PROTEINS)
        atom_array.occupancy[atom_array.atomize & isin_protein] = 0
        data["atom_array"] = atom_array
        return data

class ErrIfAllUnresolved(Transform):
    """Throw an error if all atoms are unresolved."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        if (atom_array.occupancy == 0).all():
            raise ValueError(f"All atoms are unresolved for {data['example_id']}")
        return data
    
class AddCachedResidueData(Transform): #! (JH) changed 251001
    """Add cached residue data to the data dict."""
    def __init__(self, residue_cache_dir: str):
        self.residue_cache_dir = residue_cache_dir
        self._residue_cache: dict[str, dict] = {}
        
    def _load_residue_cached_entry(self, res_name: str) -> dict | None: 
        if res_name in self._residue_cache:
            return self._residue_cache[res_name]
        path = Path(self.residue_cache_dir) / res_name / f"{res_name}.pt"
        if not path.exists():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=False)
        # entry example: {"mol": rdkit.Chem.Mol, "fingerprint": ..., "atom_names": ...}
        if entry is not None:
            self._residue_cache[res_name] = entry        
        return
    
    @override
    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        res_names = np.unique(atom_array.res_name)
        for rn in res_names:
            self._load_residue_cached_entry(rn)
            
        if self._residue_cache:
            data["cached_residue_level_data"] = {"residues": self._residue_cache}
        return data


#! (JH) 251128 added: Binding site annotation for docking metrics

def annotate_ligand_pockets(
    atom_array: AtomArray = None,
    pocket_distance: float = 5.0,
    n_min_ligand_atoms: int = 5,
    annotation_name: str = "is_ligand_pocket",
    receptor_chain_iids: list[str] = None,
    ligand_chain_iids: list[str] = None,
) -> AtomArray:
    """
    Identify atoms near ligands of sufficient size.
    
    Adapted from atomworks/docs/examples/pocket_conditioning_transform.py
    
    Args:
        atom_array: Input structure
        pocket_distance: Distance threshold for pocket identification (Angstroms)
        n_min_ligand_atoms: Minimum atoms required for a ligand to define pockets
        annotation_name: Name for the boolean annotation
        receptor_chain_iids: List of receptor chain IIDs
        ligand_chain_iids: List of ligand chain IIDs
        receptor_chain_ids: List of receptor chain IDs
        ligand_chain_ids: List of ligand chain IDs
        spread_residue_wise: If True, mark all atoms in a residue as pocket if any atom is in pocket
    Returns:
        AtomArray with ligand pocket annotation added
    """
    atom_array = atom_array.copy()
    
    
    use_chain_iid = False
    use_chain_id = False
    assert hasattr(atom_array, 'chain_iid') or hasattr(atom_array, 'pn_unit_iid'), "atom_array must have chain_iid or pn_unit_iid"
        
    if (receptor_chain_iids is None) or (ligand_chain_iids is None):
        # Used in training time or when there is no specified receptor and ligand chains
        ligand_pn_unit_iids, ligand_counts = np.unique(atom_array.pn_unit_iid[~atom_array.is_polymer], return_counts=True)
        # Todo: handling covalently linked non-polymers
        valid_ligand_mask = ligand_counts >= n_min_ligand_atoms
        valid_ligand_iids = ligand_pn_unit_iids[valid_ligand_mask] #! in this case, pn_unit_iids are used        
        all_valid_ligands_mask = np.isin(atom_array.pn_unit_iid, valid_ligand_iids)                
                            
    else:        
        ligand_chain_iids, ligand_counts = np.unique(atom_array.chain_iid[np.isin(atom_array.chain_iid, ligand_chain_iids)], return_counts=True)                
        valid_ligand_mask = ligand_counts >= n_min_ligand_atoms
        valid_ligand_iids = ligand_chain_iids[valid_ligand_mask]  # Fixed: was ligand_chain_ids
        all_valid_ligands_mask = np.isin(atom_array.chain_iid, valid_ligand_iids)        

    # Initialize pocket annotation
    pocket_annotation = np.zeros(len(atom_array), dtype=bool)

    if len(valid_ligand_iids) == 0:
        atom_array.set_annotation(annotation_name, pocket_annotation)
        return atom_array

    # Build CellList for efficient distance computations
    valid_coords_mask = ~np.isnan(atom_array.coord).any(axis=1)
    assert np.any(valid_coords_mask), "No valid coordinates found"

    valid_coords = atom_array.coord[valid_coords_mask]
    cell_list = struc.CellList(valid_coords, cell_size=pocket_distance)

    # Get coordinates of all valid ligands    
    all_ligand_coords = atom_array.coord[all_valid_ligands_mask]

    # Find atoms within distance of any ligand coordinates
    distance_mask = cell_list.get_atoms(all_ligand_coords, pocket_distance, as_mask=True)
    near_ligand_valid = np.any(distance_mask, axis=0)

    # Map back to full atom array
    near_ligand_full = np.zeros(len(atom_array), dtype=bool)
    near_ligand_full[valid_coords_mask] = near_ligand_valid

    # Only protein atoms can be pocket atoms
    pocket_annotation = atom_array.is_polymer & near_ligand_full
        
    atom_array.set_annotation(annotation_name, pocket_annotation)
    return atom_array


class AnnotateLigandPockets(Transform):
    """Identify atoms near ligands of sufficient size."""

    def __init__(
        self, 
        pocket_distance: float = 8.0, 
        n_min_ligand_atoms: int = 1, 
        annotation_name: str = "is_ligand_pocket"
    ):
        self.pocket_distance = pocket_distance
        self.n_min_ligand_atoms = n_min_ligand_atoms
        self.annotation_name = annotation_name

    @override
    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    @override
    def forward(self, data: dict = None,
                receptor_chain_iids: list[str] = None, 
                ligand_chain_iids: list[str] = None,
            ) -> dict:
        
        data["atom_array"] = annotate_ligand_pockets(
            atom_array=data["atom_array"],
            pocket_distance=self.pocket_distance,
            n_min_ligand_atoms=self.n_min_ligand_atoms,
            annotation_name=self.annotation_name,
            receptor_chain_iids=receptor_chain_iids,
            ligand_chain_iids=ligand_chain_iids,
        )
        return data
        
class GetNCACOAndPseudoCBCoords(Transform):
    """
    Get N, CA, C, O and pseudo CB coordinates for the atom array.
    """
    def __init__(self):
        pass
    
    @override
    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._get_ncaco_and_pseudo_cb_coords(data)

    def _get_ncaco_and_pseudo_cb_coords(self, data: dict[str, Any]):
        """
        Get N, CA, C, O and pseudo CB coordinates for the atom array.
        """
        atom_array = data["atom_array"]           
        
        # token_is_protein_standard_aa = data["feats"]["token_is_protein_chain"] * (1 - data["feats"]["token_is_hetero"])
        # pseudo_cb_valid_mask = token_is_protein_standard_aa * data["feats"]["token_resolved_mask"]
        
        token_len = len(data["feats"]["token_is_protein_chain"])
        
        # Get pseudo CB valid mask. For standard amino acids (not hetero) in protein chains, and all n, ca, c resolved.
        standard_aa_mask = np.isin(atom_array.res_name, STANDARD_AA)
        standard_aa_prot_mask = standard_aa_mask & (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value)
        is_ncaco_resolved = ((np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])) & (atom_array.occupancy > 0)) #! OXT is deleted in preprocessing
        has_all_backbone = apply_and_spread_token_wise(atom_array, is_ncaco_resolved, lambda x: np.sum(x) == 4) #! OXT is deleted in preprocessing
        pseudo_cb_valid_mask = standard_aa_prot_mask & has_all_backbone
        
        # token ids (must be int64 for torch indexing, not uint32)
        token_idxs = torch.tensor(atom_array.token_id, dtype=torch.long)  # [n_atoms]
        
        # Get CA, N, C, O mask for pseudo CB calculation.
        ca_mask = torch.tensor(pseudo_cb_valid_mask & (atom_array.atom_name == "CA"))
        n_mask = torch.tensor(pseudo_cb_valid_mask & (atom_array.atom_name == "N"))
        c_mask = torch.tensor(pseudo_cb_valid_mask & (atom_array.atom_name == "C"))
        o_mask = torch.tensor(pseudo_cb_valid_mask & (atom_array.atom_name == "O"))
        
        # Sanity check: all masks should have the same count
        assert ca_mask.sum() == n_mask.sum() == c_mask.sum() == o_mask.sum(), \
        f"Mask count mismatch: CA={ca_mask.sum()}, N={n_mask.sum()}, C={c_mask.sum()}, O={o_mask.sum()}"
        
        # ca_coords, n_coords, c_coords and token ids
        noised_ca_coords = data["feats"]["noised_coords"][ca_mask]
        noised_n_coords = data["feats"]["noised_coords"][n_mask]
        noised_c_coords = data["feats"]["noised_coords"][c_mask]
        noised_o_coords = data["feats"]["noised_coords"][o_mask]        
        pseudo_cb_token_idxs = token_idxs[ca_mask]
        
        b = noised_ca_coords - noised_n_coords
        c = noised_c_coords - noised_ca_coords
        a = torch.cross(b, c, dim=-1)        
        pseudo_cb_coords = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + noised_ca_coords
        
        ca_coords_new = torch.zeros((token_len, 3), dtype=torch.float32)
        ca_coords_new[pseudo_cb_token_idxs] = noised_ca_coords
        n_coords_new = torch.zeros((token_len, 3), dtype=torch.float32)
        n_coords_new[pseudo_cb_token_idxs] = noised_n_coords
        c_coords_new = torch.zeros((token_len, 3), dtype=torch.float32)
        c_coords_new[pseudo_cb_token_idxs] = noised_c_coords
        o_coords_new = torch.zeros((token_len, 3), dtype=torch.float32)
        o_coords_new[pseudo_cb_token_idxs] = noised_o_coords
        pseudo_cb_coords_new = torch.zeros((token_len, 3), dtype=torch.float32)
        pseudo_cb_coords_new[pseudo_cb_token_idxs] = pseudo_cb_coords
        
        data["feats"] |= {
            "noised_ca_coords": ca_coords_new,
            "noised_n_coords": n_coords_new,
            "noised_c_coords": c_coords_new,
            "noised_o_coords": o_coords_new,
            "noised_pseudo_cb_coords": pseudo_cb_coords_new,
        }   
        
        return data
    
class AddTrainingRandomNoise(Transform):
    """Add training random noise to the atom array."""
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
    
    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._add_training_random_noise(data)
    
    def _add_training_random_noise(self, data: dict[str, Any]):
        """Add training random noise to the atom array."""
        N_atoms = len(data["atom_array"])
        if self.noise_scale <= 0:
            noise = torch.zeros((N_atoms, 3), dtype=torch.float32)
        else: 
            noise = self.noise_scale * torch.randn((N_atoms, 3), dtype=torch.float32)
        
        data["feats"]["noise"] = noise
        
        return data
    

def get_af3_token_representative_masks(
    atom_array: AtomArray, central_atom: str = "CA"
) -> np.ndarray:
    """
    Adapted from foundry/models/rfd3/src/rfd3/transforms/util_transforms.py
    Changed to use CA as the centeral atom, as we're doing backbone-conditioned sequence design.
    """
    
    pyrimidine_representative_atom = is_pyrimidine(atom_array.res_name) & (
        atom_array.atom_name == "C2"
    )
    purine_representative_atom = is_purine(atom_array.res_name) & (
        atom_array.atom_name == "C4"
    )
    unknown_na_representative_atom = is_unknown_nucleotide(atom_array.res_name) & (
        atom_array.atom_name == "C4"
    )

    glycine_representative_atom = is_glycine(atom_array.res_name) & (
        atom_array.atom_name == "CA"
    )
    protein_residue_not_glycine_representative_atom = is_standard_aa_not_glycine(
        atom_array.res_name
    ) & (
        atom_array.atom_name == central_atom  # only change
    )
    unknown_protein_residue_representative_atom = (
        is_protein_unknown(atom_array.res_name)
    ) & (atom_array.atom_name == "CA")
    atoms = atom_array.atomize

    _token_rep_mask = (
        pyrimidine_representative_atom
        | purine_representative_atom
        | unknown_na_representative_atom
        | glycine_representative_atom
        | protein_residue_not_glycine_representative_atom
        | unknown_protein_residue_representative_atom
        | atoms
    )
    return _token_rep_mask


## Remove unsupported chain types transform
def exists(obj: Any) -> bool:
    """Check that obj is not None.

    Args:
        obj: The object to check.

    Returns:
        True if obj is not None, False otherwise.
    """
    return obj is not None

def remove_unsupported_chain_types(
    atom_array: AtomArray,
    query_pn_unit_iids: Sequence[str] | None = None,
    supported_chain_types: Sequence[aw_enums.ChainType] = TRAINING_SUPPORTED_CHAIN_TYPES,
) -> AtomArray:
    """Filter out chains with unsupported chain types from the AtomArray.

    Additionally, asserts that none of the query pn_units are of an unsupported chain type if given.
    (in which case they should have been filtered out upstream, otherwise our example is not valid).

    Args:
        query_pn_unit_iids (Sequence[str] | None): The PN unit IDs to check for unsupported chain types.
        supported_chain_types (Sequence[ChainType]): The chain types to filter out.

    Returns:
        AtomArray: The filtered AtomArray.
    """
    # Convert chain_type to int if stored as string (e.g., from saved cif files) #! (JH) 251201 fixed
    chain_types = atom_array.chain_type
    if chain_types.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
        chain_types = np.array([int(ct) if str(ct).isdigit() else ct for ct in chain_types])
    
    # Convert supported_chain_types to int values for comparison
    supported_chain_type_values = [int(ct) for ct in supported_chain_types]
    
    # We first assert that none of the query pn_units are of an unsupported chain type, which means the example should have been filtered out upstream
    if exists(query_pn_unit_iids):
        query_pn_unit_chain_types = np.unique(
            chain_types[np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)]
        )
        assert np.all(
            np.isin(query_pn_unit_chain_types, supported_chain_type_values)
        ), f"Query PN unit has an unsupported chain type: {query_pn_unit_chain_types}"

    # Then, we filter out chains with unsupported chain types
    is_supported_chain_type = np.isin(chain_types, supported_chain_type_values)
    return atom_array[is_supported_chain_type]

class RemoveUnsupportedChainTypes(Transform):
    """Filter out chains with unsupported chain types from the AtomArray.

    Additionally, asserts that none of the query pn_units are of an unsupported chain type if given.
    (in which case they should have been filtered out upstream, otherwise our example is not valid).
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = []

    def __init__(self, supported_chain_types: Sequence[aw_enums.ChainType] = TRAINING_SUPPORTED_CHAIN_TYPES):
        """
        Initialize the RemoveUnsupportedChainTypes transform.

        Args:
            supported_chain_types (Sequence[ChainType]): The chain types to keep in the AtomArray.
        """
        self.supported_chain_types = supported_chain_types

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["chain_type", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        query_pn_unit_iids = data.get("query_pn_unit_iids")

        # Apply transform
        atom_array = remove_unsupported_chain_types(atom_array, query_pn_unit_iids, self.supported_chain_types)

        # Update data
        data["atom_array"] = atom_array

        return data