from typing import Any
from typing_extensions import override
from torch import Tensor as TensorType
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

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


import allatom_design.data.const as const
from allatom_design.data.transform.pad import pad_dim


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
    "ca_coords": [0],
    "n_coords": [0],
    "c_coords": [0],
    "o_coords": [0],
    "pseudo_cb_coords": [0],
    "token_bonds": [0, 1],            
    
    # optional features that might not be present
    "seq_cond_mask": [0],
    "token_exists_mask": [0],
}

FEAT_TO_ATOM_DIM = {
    # Maps feature name to the atom dimension
    # atom features
    "coords": [0],
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
INFERENCE_ONLY_KEYS = ["crop_info", "atom_array", "feat_metadata"]

class CheckCoordinatesAreNan(Transform):
    """Check if the coordinates are nan."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        return data

class FeaturizeCoordsAndMasks(Transform):
    """Add coordinates and atom masks to feats."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        feats = data["feats"]

        # Get coordinates
        feats["coords"] = torch.tensor(atom_array.coord)

        # Get token and atom resolved masks                
        feats["token_resolved_mask"] = torch.tensor(apply_token_wise(atom_array, atom_array.occupancy > 0, np.any)).float()
        feats["atom_resolved_mask"] = torch.tensor(atom_array.occupancy > 0).float()
        
        # Make pad masks
        feats["token_pad_mask"] = torch.ones_like(feats["token_resolved_mask"])
        feats["atom_pad_mask"] = torch.ones_like(feats["atom_resolved_mask"])

        # atomized
        feats["atom_is_atomized"] = torch.tensor(atom_array.atomize).float()

        # Get Chain type features        
        token_starts = get_token_starts(atom_array)
        feats["atom_chain_type"] = torch.tensor(atom_array.chain_type).long()
        feats["token_chain_type"] = torch.tensor(atom_array[token_starts].chain_type)
        feats["atom_is_polymer"] = torch.tensor(atom_array.is_polymer).float()
        feats["token_is_polymer"] = torch.tensor(atom_array[token_starts].is_polymer).float()        
                
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
        for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
            pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
            sel_atom_array = atom_array[pn_unit_mask]
            chain_type = np.unique(sel_atom_array.chain_type)
            if chain_type in polymer_chain_type_enums:
                if chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value:                    
                    atom_is_protein_chain[pn_unit_mask] = True                                    
                elif chain_type in nucleic_acid_chain_type_enums:
                    atom_is_nucleic_acid_chain[pn_unit_mask] = True
                elif np.isin(sel_atom_array.chain_type, non_polymer_chain_type_enums):
                    if len(sel_atom_array) == 1 & np.isin(sel_atom_array.element, METAL_ELEMENTS):
                        atom_is_metal_chain[pn_unit_mask] = True
                    else:
                        atom_is_small_molecule_chain[pn_unit_mask] = True
        
        token_is_protein_chain = atom_is_protein_chain[get_token_starts(atom_array)]
        token_is_nucleic_acid_chain = atom_is_nucleic_acid_chain[get_token_starts(atom_array)]
        token_is_metal_chain = atom_is_metal_chain[get_token_starts(atom_array)]
        token_is_small_molecule_chain = atom_is_small_molecule_chain[get_token_starts(atom_array)]
        
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
        feats["prot_bb_atom_mask"] = torch.tensor(is_prot_bb).float()
        feats["prot_scn_atom_mask"] = torch.tensor(is_prot_scn).float()        
                        
        # Calculate number of atoms per token
        device = feats["coords"].device
        N_tokens = feats["token_pad_mask"].shape[0]
        n_atoms_per_token = (F.one_hot(feats["atom_to_token_map"], num_classes=N_tokens)).sum(dim=-2)

        # Starting atom index for each token
        tokenwise_atom_idxs = torch.cat([torch.zeros((1,), device=device), n_atoms_per_token.cumsum(dim=-1)[:-1]], dim=-1).long()
        tokenwise_atom_idxs = tokenwise_atom_idxs[..., None] + torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :]
        tokenwise_atom_idxs_mask = torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :] < n_atoms_per_token[..., None]
        tokenwise_atom_idxs = torch.where(tokenwise_atom_idxs_mask, tokenwise_atom_idxs, -1 * torch.ones_like(tokenwise_atom_idxs, dtype=torch.long))
        
        feats["tokenwise_atom_idxs"] = tokenwise_atom_idxs
        feats["tokenwise_atom_idxs_mask"] = tokenwise_atom_idxs_mask                
        feats["ca_coords"], feats["n_coords"], feats["c_coords"], feats["o_coords"], feats["pseudo_cb_coords"] = self._get_pseudo_cb_coords(data)        

        # Get ligand related features
        # try:
        #     feats["atom_is_aromatic"] = torch.tensor(atom_array.is_aromatic).float()
        # except: 
        #     # Todo: LMT & DD6 from 6a2w, no attribute error for is_aromatic.
        #     # Todo: Both have hexagon rings, but no aromaticity. Maybe error in parsing?
        #     feats["atom_is_aromatic"] = torch.full((len(atom_array),), False).float()
        #     print(f"Atom array has no attribute 'is_aromatic' for {data['example_id']}")            
                        
        return data
    
    def _get_pseudo_cb_coords(self, data: dict[str, Any]):
        """
        Get pseudo CB coordinates for the atom array.
        """
        atom_array = data["atom_array"]           
        
        # token_is_protein_standard_aa = data["feats"]["token_is_protein_chain"] * (1 - data["feats"]["token_is_hetero"])
        # pseudo_cb_valid_mask = token_is_protein_standard_aa * data["feats"]["token_resolved_mask"]
        
        token_len = len(data["feats"]["token_is_protein_chain"])
        
        # Get pseudo CB valid mask. For standard amino acids (not hetero) in protein chains, and all n, ca, c resolved.
        standard_aa_mask = np.isin(atom_array.res_name, STANDARD_AA)
        standard_aa_prot_mask = standard_aa_mask & (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L.value)
        is_ncaco_resolved = ((np.isin(atom_array.atom_name, [PROTEIN_BACKBONE_ATOM_NAMES])) & (atom_array.occupancy > 0))
        has_all_backbone = apply_and_spread_token_wise(atom_array, is_ncaco_resolved, lambda x: np.sum(x) == len(PROTEIN_BACKBONE_ATOM_NAMES))
        pseudo_cb_valid_mask = standard_aa_prot_mask & has_all_backbone
        
        # token ids
        token_idxs = atom_array.token_id  # [n_atoms]
        
        # Get ca, n, c mask for pseudo CB calculation.
        ca_mask = pseudo_cb_valid_mask & (atom_array.atom_name == "CA")
        n_mask = pseudo_cb_valid_mask & (atom_array.atom_name == "N")
        c_mask = pseudo_cb_valid_mask & (atom_array.atom_name == "C")
        o_mask = pseudo_cb_valid_mask & (atom_array.atom_name == "O")
        
        # ca_coords, n_coords, c_coords and token ids
        ca_coords = atom_array.coord[ca_mask]
        n_coords = atom_array.coord[n_mask]
        c_coords = atom_array.coord[c_mask]
        o_coords = atom_array.coord[o_mask]
        pseudo_cb_token_idxs = token_idxs[ca_mask]
        
        b = ca_coords - n_coords
        c = c_coords - ca_coords
        a = np.cross(b, c, axis=-1)
        np_cb_coords = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_coords
        
        torch_ca_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        torch_ca_coords[pseudo_cb_token_idxs] = torch.from_numpy(ca_coords).float()
        torch_n_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        torch_n_coords[pseudo_cb_token_idxs] = torch.from_numpy(n_coords).float()
        torch_c_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        torch_c_coords[pseudo_cb_token_idxs] = torch.from_numpy(c_coords).float()
        torch_o_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        torch_o_coords[pseudo_cb_token_idxs] = torch.from_numpy(o_coords).float()
        torch_cb_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        torch_cb_coords[pseudo_cb_token_idxs] = torch.from_numpy(np_cb_coords).float()        
        
        return torch_ca_coords, torch_n_coords, torch_c_coords, torch_o_coords, torch_cb_coords                    

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
    """Randomly augment the center of the atom array."""
    def __init__(self, apply_random_augmentation: bool = True, translation_scale: float = 1.0):
        self.apply_random_augmentation = apply_random_augmentation
        self.translation_scale = translation_scale

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.apply_random_augmentation:
            coords = data["feats"]["coords"]
            mask = data["feats"]["atom_resolved_mask"].bool()
            centered_coords = coords.clone()
            centered_coords = masked_center(centered_coords, mask)
            centered_coords = random_rigid_augmentation(centered_coords[None], batch_size=1, s=self.translation_scale).squeeze(0) #! dummy atom coords masked later?
            data["feats"]["coords"] = centered_coords    
        return data


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

class AtomizeShortPolymers(Transform):
    def __init__(self, peptide_max_res: int = 20, na_max_res: int = 10):
        self.peptide_max_res = peptide_max_res
        self.na_max_res = na_max_res

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "res_id", "chain_type", "is_polymer"])

    @override
    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        if "atomize" not in aa.get_annotation_categories():
            aa.set_annotation("atomize", np.zeros(len(aa), dtype=bool))

        # Use flags from extra_info (supports '', _1, _2, ...)
        extra = data.get("extra_info", {})
        pn_raw = aa.pn_unit_iid
        pn_str = pn_raw.astype(str)  # match by string
        
        # Detect suffixes like AddChainTypeFeatrues
        suffixes = []
        for k in extra.keys():
            if k.startswith("q_pn_unit_iid"):
                suffixes.append(k[len("q_pn_unit_iid"):])
        suffixes = sorted(set(suffixes))
                
        for sfx in suffixes:
            qid = extra.get(f"q_pn_unit_iid{sfx}", None)
            if qid is None:
                continue
            is_pep = bool(extra.get(f"q_pn_unit_is_peptide{sfx}", False))
            is_nuc_lig = bool(extra.get(f"q_pn_unit_is_nuc_ligand{sfx}", False))
            if not (is_pep or is_nuc_lig):
                continue
            
            sel = pn_str == str(qid).strip()
            if sel.any():
                aa.atomize[sel] = True
                aa.is_polymer[sel] = False
                aa.chain_type[sel] = aw_enums.ChainType.NON_POLYMER
                                
        data["atom_array"] = aa
        return data
    
class AddChainTypeFeaturesForTrain(Transform):
    """Add chain type features to the data dict."""
    def __init__(self):
        pass

    @override
    def forward(self, data: dict) -> dict:      
        prot_col = "q_pn_unit_is_protein"
        small_molecule_cols = [
        "q_pn_unit_is_small_molecule",
        "q_pn_unit_is_peptide",
        "q_pn_unit_is_nuc_ligand",
        ]
        nuc_cols = [
        "q_pn_unit_is_nuc_polymer",
        ]
    
        metal_col = "q_pn_unit_is_metal"
    
        all_chain_type_cols = [prot_col, *small_molecule_cols, *nuc_cols, metal_col]        
        asym_id = data["feats"]["asym_id"]
        asym_name = data["feat_metadata"]["asym_name"]
        asym_name_to_id = {name: id for id, name in enumerate(asym_name)}
        
        def _detect_suffix(data: dict[str, Any]):
            suffixes = []
            for key in data['extra_info'].keys():
                if key.startswith('q_pn_unit_iid'):
                    suffix = key[len('q_pn_unit_iid'):]
                    if suffix is None:
                        suffix = ''
                    suffixes.append(suffix)
            return sorted(set(suffixes))
        
        suffixes = _detect_suffix(data)        
        chain_is_protein = np.zeros_like(asym_id, dtype=bool)
        chain_is_nuc = np.zeros_like(asym_id, dtype=bool)
        chain_is_small_molecule = np.zeros_like(asym_id, dtype=bool)
        chain_is_metal = np.zeros_like(asym_id, dtype=bool)                                
        
        for suffix in suffixes:
            key = f"q_pn_unit_iid{suffix}"
            assert key in data["extra_info"], f"q_pn_unit_iid{suffix} not in data['extra_info']"        
            
            q = str(data['extra_info'][key]).strip() 
            idx = asym_name_to_id.get(q)
            if idx is None: #! (JH) because of cropping, one of the q_pn_unit_iid may not be in the data
                continue
                        
            q_pn_unit_iid = data['extra_info'][f'q_pn_unit_iid{suffix}']
            is_prot  = data['extra_info'][f'q_pn_unit_is_protein{suffix}']
            is_metal = data['extra_info'][f'q_pn_unit_is_metal{suffix}']
            is_small = any(data['extra_info'][f'{c}{suffix}'] for c in small_molecule_cols)
            is_nuc   = any(data['extra_info'][f'{c}{suffix}'] for c in nuc_cols)
                        
            chain_is_protein[asym_id == asym_name_to_id[q_pn_unit_iid]] = is_prot
            chain_is_metal[asym_id == asym_name_to_id[q_pn_unit_iid]] = is_metal
            chain_is_small_molecule[asym_id == asym_name_to_id[q_pn_unit_iid]] = is_small
            chain_is_nuc[asym_id == asym_name_to_id[q_pn_unit_iid]] = is_nuc            
              
        
                                                                                        
        data["feats"] |= {
            "chain_is_protein": chain_is_protein,
            "chain_is_nuc": chain_is_nuc,
            "chain_is_small_molecule": chain_is_small_molecule,
            "chain_is_metal": chain_is_metal,
        }
        
        return data
    
class AddChainTypeFeaturesForInference(Transform):
    """Add chain type features to the data dict."""
    def __init__(self):
        pass

    @override
    def forward(self, data: dict) -> dict:      
        
        atom_array = data["atom_array"]
        asym_id = data["feats"]["asym_id"]
        
        chain_is_protein = np.zeros_like(asym_id, dtype=bool)
        chain_is_nuc = np.zeros_like(asym_id, dtype=bool)
        chain_is_small_molecule = np.zeros_like(asym_id, dtype=bool)
        chain_is_metal = np.zeros_like(asym_id, dtype=bool)
        
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        protein_chain_iids = np.unique(token_level_array.chain_iid[token_level_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L])
        non_polymer_chain_iids = np.unique(token_level_array.chain_iid[token_level_array.chain_type == aw_enums.ChainType.NON_POLYMER])
                
        for protein_chain_iid in protein_chain_iids:
            sel = token_level_array.chain_iid == protein_chain_iid
            chain_is_protein[sel] = True
        
        for non_polymer_chain_iid in non_polymer_chain_iids:
            sel = token_level_array.chain_iid == non_polymer_chain_iid
            
            # For single metal ions
            if len(token_level_array[sel]) == 1 and token_level_array[sel].element in METAL_ELEMENTS:
                chain_is_metal[sel] = True            
            else:
                chain_is_small_molecule[sel] = True
            
        # Todo: Need to implement nuc chain type features for inference. Need to consider DNA, RNA, and DNA_RNA_HYBRID as in grouped chains                        
                                                                                        
        data["feats"] |= {
            "chain_is_protein": chain_is_protein,
            "chain_is_nuc": chain_is_nuc,
            "chain_is_small_molecule": chain_is_small_molecule,
            "chain_is_metal": chain_is_metal,
        }
        
        return data
    
# class AddChainTypeAnnotationsToAtomArray(Transform):
#     """Copy chain_is_x (token-wise) to atom_array annotation"""
#     @override
#     def forward(self, data: dict[str, Any]) -> dict[str, Any]:
#         atom_array = data["atom_array"]
#         feats = data["feats"]

#         # (1) atom_to_token_map : [n_atoms]  (already created in ComputeAtomToTokenMap)
#         atom_to_token = feats["atom_to_token_map"].detach().cpu().numpy()

#         # (2) Each chain_is_x (token-wise → atom-wise broadcast)
#         for name in ["chain_is_protein",
#                      "chain_is_small_molecule",
#                      "chain_is_metal",
#                      "chain_is_nuc"]:
#             token_feat = np.asarray(feats[name], dtype=bool)        # [n_tokens]
#             atom_feat = token_feat[atom_to_token]                   # [n_atoms]

#             # (3) Set AtomArray annotation
#             atom_array.set_annotation(name, atom_feat)

#         data["atom_array"] = atom_array
#         return data

# class AssignPNUnitIIDsToAtomArray(Transform):
#     """Assign PN unit IIDs to the atom array for designed samples from other methods."""
#     def __init__(self):
#         pass

#     @override
#     def forward(self, data: dict) -> dict:
#         atom_array = data["atom_array"]
#         if "pn_unit_iid" not in atom_array.get_annotation_categories():
#             pn_unit_id = atom_array.chain_id
            
#             pn_unit_iid = sum_string_arrays(atom_array.pn_unit_id, "_", atom_array.transformation_id)
            
#             atom_array.set_annotation("pn_unit_iid", np.full(len(atom_array), ""))
#         return data


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