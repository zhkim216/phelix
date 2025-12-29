from typing import Any
from typing_extensions import override
from torch import Tensor as TensorType
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

import biotite.structure as struc
from biotite.structure import AtomArray
import atomworks.enums as aw_enums
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms._checks import check_contains_keys, check_is_instance, check_atom_array_annotation
from atomworks.ml.transforms.filters import filter_to_specified_pn_units
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation
from atomworks.ml.utils.token import apply_token_wise, get_af3_token_center_idxs

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
    "is_ligand": [0],
    "is_atomized": [0],
    "token_bonds": [0, 1],
    "token_to_center_atom": [0],
    "token_pad_mask": [0],
    "token_resolved_mask": [0],
    "tokenwise_atom_idxs": [0],
    "tokenwise_atom_idxs_mask": [0],
    "pseudo_cb_coords": [0],
    
    # Ligand related features    
    "token_chain_type": [0], 
    "chain_is_protein": [0],
    "chain_is_small_molecule": [0],   
    "chain_is_metal": [0],
    "chain_is_nuc": [0],
    
    # optional features that might not be present
    "seq_cond_mask": [0],
    "token_exists_mask": [0],
}

FEAT_TO_ATOM_DIM = {
    # Maps feature name to the atom dimension
    # atom features
    "coords": [0],
    "atom_pad_mask": [0],
    "atom_resolved_mask": [0],
    "atom_to_token_map": [0],

    "prot_bb_atom_mask": [0],
    "prot_scn_atom_mask": [0],
    
    # optional features that might not be present
    "atom_cond_mask": [0],
    
    # Ligand related features
    "atomic_number": [0],
    "atom_is_metal": [0],    
    "atom_is_small_molecule": [0],
    "atom_is_ligand": [0],
    
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

        # Get protein backbone and sidechain atom masks
        feats["atom_to_token_map"] = feats["atom_to_token_map"].long()
        atom_is_prot = feats["is_protein"].gather(dim=-1, index=feats["atom_to_token_map"]) #! changed 251101 (JH)
        feats["atom_is_prot"] = atom_is_prot        
        
        atomized = torch.tensor(atom_array.atomize)
        bb_atom_mask = torch.tensor(atom_array.is_backbone_atom) #! changed 251101 (JH)

        feats["prot_bb_atom_mask"] = bb_atom_mask * ~atomized * atom_is_prot
        feats["prot_scn_atom_mask"] = ~bb_atom_mask * ~atomized * atom_is_prot
        feats["token_to_center_atom"] = torch.tensor(get_af3_token_center_idxs(atom_array))

        feats["token_is_polymer"] = torch.tensor(apply_token_wise(atom_array, atom_array.is_polymer, np.any)).float()
        feats["token_chain_type"] = torch.tensor(apply_token_wise(atom_array, atom_array.chain_type, np.any)).float()
        feats["atomic_number"] = torch.tensor(atom_array.atomic_number).long()
                        
        # Convert atomwise to tokenwise (same method as get_tokenwise_coords)        
        # Calculate number of atoms per token
        device = feats["coords"].device
        N_tokens = feats["token_pad_mask"].shape[0]
        n_atoms_per_token = (F.one_hot(feats["atom_to_token_map"], num_classes=N_tokens)).sum(dim=-2)

        # Starting atom index for each token
        tokenwise_atom_idxs = torch.cat([torch.zeros((1,), device=device), n_atoms_per_token.cumsum(dim=-1)[:-1]], dim=-1).long()
        tokenwise_atom_idxs = tokenwise_atom_idxs[..., None] + torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :]
        tokenwise_atom_idxs_mask = torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :] < n_atoms_per_token[..., None]
        tokenwise_atom_idxs = tokenwise_atom_idxs * tokenwise_atom_idxs_mask
        
        feats["tokenwise_atom_idxs"] = tokenwise_atom_idxs
        feats["tokenwise_atom_idxs_mask"] = tokenwise_atom_idxs_mask                
        feats["pseudo_cb_coords"] = self._get_pseudo_cb_coords(data)
                    
        # Ligand related features
        atom_is_dna = feats["is_dna"].gather(dim=-1, index=feats["atom_to_token_map"])
        atom_is_rna = feats["is_rna"].gather(dim=-1, index=feats["atom_to_token_map"])
        atom_is_ligand = feats["is_ligand"].gather(dim=-1, index=feats["atom_to_token_map"])
        
        feats["atom_is_dna"] = atom_is_dna
        feats["atom_is_rna"] = atom_is_rna
        feats["atom_is_ligand"] = atom_is_ligand
        
        feats["token_is_metal"] = feats["chain_is_metal"].clone()
        feats["atom_is_metal"] = feats["chain_is_metal"].gather(dim=-1, index=feats["atom_to_token_map"])
        
        # Ligand pocket features
        if "is_ligand_pocket" in atom_array.get_annotation_categories():    
            feats["atom_is_ligand_pocket"] = torch.tensor(atom_array.is_ligand_pocket)
            feats["pocket_atom_mask"] = feats["atom_is_ligand_pocket"] * feats["atom_resolved_mask"]
            feats["token_is_ligand_pocket"] = torch.tensor(apply_token_wise(atom_array, atom_array.is_ligand_pocket, np.any)).bool()
            feats["pocket_token_mask"] = feats["token_is_ligand_pocket"] * feats["token_resolved_mask"]                
            
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
        token_is_protein = data["feats"]["is_protein"] * data["feats"]["chain_is_protein"]
        token_len = len(token_is_protein)
        
        token_id = atom_array.token_id  # [n_atoms]
        atom_name = atom_array.atom_name  # [n_atoms]
        coords_np = atom_array.coord  # [n_atoms, 3]
                
        atom_is_prot = data["feats"]["atom_is_prot"].detach().to(torch.bool).cpu().numpy() #! Important to convert to numpy and bool, to use boolean mask
        
        ca_mask = atom_is_prot & (atom_name == "CA")
        n_mask = atom_is_prot & (atom_name == "N")
        c_mask = atom_is_prot & (atom_name == "C")
        
        # Positions and their token ids
        pos_ca, toks_ca = np.where(ca_mask)[0], token_id[ca_mask]
        pos_n,  toks_n  = np.where(n_mask)[0],  token_id[n_mask]
        pos_c,  toks_c  = np.where(c_mask)[0],  token_id[c_mask]

        # Allocate and scatter
        ca_coords = torch.zeros((token_len, 3), dtype=torch.float32)
        n_coords  = torch.zeros((token_len, 3), dtype=torch.float32)
        c_coords  = torch.zeros((token_len, 3), dtype=torch.float32)

        ca_coords[toks_ca] = torch.from_numpy(coords_np[pos_ca]).float()
        n_coords[toks_n]   = torch.from_numpy(coords_np[pos_n]).float()
        c_coords[toks_c]   = torch.from_numpy(coords_np[pos_c]).float()
        
        b = ca_coords - n_coords
        c = c_coords - ca_coords
        a = torch.cross(b, c, dim=-1)
        cb_coords = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_coords
        
        return cb_coords                    

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
    
class AddChainTypeFeatrues(Transform):
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
    pocket_distance: float = 8.0,
    n_min_ligand_atoms: int = 1,
    annotation_name: str = "is_ligand_pocket",
    receptor_chains: list[str] = None,
    ligand_chains: list[str] = None,
) -> AtomArray:
    """
    Identify atoms near ligands of sufficient size.
    
    Adapted from atomworks/docs/examples/pocket_conditioning_transform.py
    
    Args:
        atom_array: Input structure
        pocket_distance: Distance threshold for pocket identification (Angstroms)
        n_min_ligand_atoms: Minimum atoms required for a ligand to define pockets
        annotation_name: Name for the boolean annotation
        receptor_chains: List of receptor chain IDs
        ligand_chains: List of ligand chain IDs
    Returns:
        AtomArray with ligand pocket annotation added
    """
    atom_array = atom_array.copy()

    # Find all ligand pn_unit_iids within our structure and their atom counts
    if (receptor_chains is None) or (ligand_chains is None):
        non_protein_mask = ~(atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        ligand_chains = np.unique(atom_array.chain_id[non_protein_mask])
        ligand_counts = np.array([np.sum(atom_array.chain_id[non_protein_mask] == ch) for ch in ligand_chains])
        valid_ligand_mask = ligand_counts >= n_min_ligand_atoms
        valid_ligand_chains = ligand_chains[valid_ligand_mask]
        all_valid_ligands_mask = np.isin(atom_array.chain_id, valid_ligand_chains) & non_protein_mask
    else:
        if not hasattr(atom_array, 'pn_unit_iid'):
            ligand_chain_ids, ligand_counts = np.unique(atom_array.chain_id[np.isin(atom_array.chain_id, ligand_chains)], return_counts=True)
            valid_ligand_mask = ligand_counts >= n_min_ligand_atoms
            valid_ligand_chain_ids = ligand_chain_ids[valid_ligand_mask]
            all_valid_ligands_mask = np.isin(atom_array.chain_id, valid_ligand_chain_ids)
        else:    
            ligand_pn_unit_iids, ligand_counts = np.unique(
                atom_array.pn_unit_iid[np.isin(atom_array.chain_id, ligand_chains)], return_counts=True
            )
            valid_ligand_mask = ligand_counts >= n_min_ligand_atoms
            valid_ligand_pn_unit_iids = ligand_pn_unit_iids[valid_ligand_mask]
            all_valid_ligands_mask = np.isin(atom_array.pn_unit_iid, valid_ligand_pn_unit_iids)

    # Initialize pocket annotation
    pocket_annotation = np.zeros(len(atom_array), dtype=bool)

    if not np.any(all_valid_ligands_mask):
        atom_array.set_annotation(annotation_name, pocket_annotation)
        return atom_array

    # Build CellList for efficient distance computations
    valid_coords_mask = ~np.isnan(atom_array.coord).any(axis=1)
    if not np.any(valid_coords_mask):
        atom_array.set_annotation(annotation_name, pocket_annotation)
        return atom_array

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
    is_protein_chain = (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
    pocket_annotation = is_protein_chain & near_ligand_full

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
    def forward(self, data: dict, receptor_chains: list[str] = None, ligand_chains: list[str] = None) -> dict:
        print(f"pocket_distance: {self.pocket_distance}")
        data["atom_array"] = annotate_ligand_pockets(
            data["atom_array"],
            pocket_distance=self.pocket_distance,
            n_min_ligand_atoms=self.n_min_ligand_atoms,
            annotation_name=self.annotation_name,
            receptor_chains=receptor_chains,
            ligand_chains=ligand_chains,
        )
        return data