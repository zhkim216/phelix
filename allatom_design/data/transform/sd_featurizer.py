from typing import Any, override
from pathlib import Path
import atomworks.enums as aw_enums
import atomworks.constants as aw_const
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as TensorType
import logging
from atomworks.constants import (AF3_EXCLUDED_LIGANDS, STANDARD_AA,
                                    STANDARD_DNA, STANDARD_RNA)
from atomworks.ml.transforms.atom_array import (AddGlobalTokenIdAnnotation,
                                                ComputeAtomToTokenMap)
from atomworks.ml.transforms.base import (AddData, Compose, ConditionalRoute,
                                          ConvertToTorch, Identity,
                                          RandomRoute, RemoveKeys,
                                          SubsetToKeys, Transform)
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.crop import (CropContiguousLikeAF3,
                                          CropSpatialLikeAF3)
from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.encoding import (EncodeAF3TokenLevelFeatures,
                                              EncodeAtomArray)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskResiduesWithSpecificUnresolvedAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence)
from atomworks.ml.transforms.filters import (FilterToProteins,
                                             RemoveUnresolvedTokens,
                                             RemoveUnsupportedChainTypes,
                                             filter_to_specified_pn_units)
from atomworks.ml.utils.geometry import (masked_center,
                                         random_rigid_augmentation)
from atomworks.ml.utils.token import (apply_token_wise,
                                      get_af3_token_center_idxs,
                                      get_af3_token_center_masks,
                                      get_af3_token_representative_masks,
                                      spread_token_wise)

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
    "tokenwise_atom_idxs": [0], #! (JH) changed
    "tokenwise_atom_idxs_mask": [0], #! (JH) changed
    "pseudo_cb_coords": [0],
    
    # Ligand related features    
    "token_chain_type": [0], 
    "token_is_small_molecule": [0],   
    "token_is_metal": [0],
    "token_is_polymer": [0],
    
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
    # "atom_is_aromatic": [0],
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

logger = logging.getLogger(__name__)

def sd_featurizer(
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_p: float = 0.0,
    remove_keys: list[str] = [],
    remove_unresolved_tokens: bool = False,
    residue_cache_dir: str | None = None,
    max_conformers_per_residue: int | None = None,
    apply_random_augmentation: bool = True,
    translation_scale: float = 1.0
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into a training example (including cropping).
    """
    # Featurization that must be done before cropping
    featurization_transforms_pre_crop = [
        MaskResiduesWithSpecificUnresolvedAtoms(chain_type_to_atom_names={
            aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_FRAME_ATOM_NAMES,
            aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_FRAME_ATOM_NAMES,
        }),
        # FilterToProteins(), #! (JH) changed 250926, turn it off for lc-sd
        FilterToQueryPNUnits(),
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        MaskAtomizedTokensInProtein(), #! (JH) changed 250926, only mask atomized tokens (possibly modified residues) in protein
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(), #! (JH) changed        
    ]

    # Cropping
    crop_contiguous_p = 1.0 - crop_spatial_p
    cropping_transform = Identity()
    if max_tokens is not None:
        cropping_transform = RandomRoute(
            transforms=[
                CropContiguousLikeAF3(
                    crop_size=max_tokens,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms,
                ),
                CropSpatialLikeAF3(
                    crop_size=max_tokens,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms,
                ),
            ],
            probs=[crop_contiguous_p, crop_spatial_p],
        )            
    
    # Featurization
    # NOTE: for now, we ignore ref pos features because they are too slow to compute
    featurization_transforms_post_crop = [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        AddCachedResidueData(residue_cache_dir=residue_cache_dir), #! (JH) changed 251001
        # GetAF3ReferenceMoleculeFeatures(
        #     save_rdkit_mols=False,
        #     use_cached_conformers=True,
        #     conformer_generation_timeout=5.0,
        #     use_element_for_atom_names_of_atomized_tokens=False,
        #     max_conformers_per_residue=max_conformers_per_residue,
        # ), #Todo: add automorphisms and chiral features
        ComputeAtomToTokenMap(),
        AddAF3TokenBondFeatures(),
        ConvertToTorch(keys=["encoded", "feats"]),
        # Handle missing atoms and tokens
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"),

        # Add features from the atom_array
        FeaturizeCoordsAndMasks(),
        CenterRandomAugmentation(apply_random_augmentation = apply_random_augmentation, 
                                translation_scale=translation_scale), #! turn on/off depending on train/eval?
    ]
    
    transforms = [
        *featurization_transforms_pre_crop,
        cropping_transform,
        *featurization_transforms_post_crop,
        PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
        SubsetToKeys(keys=["example_id", "feats", *INFERENCE_ONLY_KEYS]),
        FlattenFeatsDict(),
        RemoveKeys(keys=remove_keys),
    ]

    return Compose(transforms)

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
        atom_is_prot = feats["is_protein"].gather(dim=-1, index=feats["atom_to_token_map"])
        feats["atom_is_prot"] = atom_is_prot        
        
        atomized = torch.tensor(atom_array.atomize)
        bb_atom_mask = torch.tensor(atom_array.is_backbone_atom)

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
        n_atoms_per_token = (F.one_hot(feats["atom_to_token_map"], num_classes=N_tokens) * feats["atom_pad_mask"][..., None]).sum(dim=-2)

        # Starting atom index for each token
        tokenwise_atom_idxs = torch.cat([torch.zeros((1,), device=device), n_atoms_per_token.cumsum(dim=-1)[:-1]], dim=-1).long()
        tokenwise_atom_idxs = tokenwise_atom_idxs[..., None] + torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :]
        tokenwise_atom_idxs_mask = torch.arange(const.MAX_NUM_ATOMS, device=device)[None, :] < n_atoms_per_token[..., None]
        tokenwise_atom_idxs = tokenwise_atom_idxs * tokenwise_atom_idxs_mask
        
        feats["tokenwise_atom_idxs"] = tokenwise_atom_idxs
        feats["tokenwise_atom_idxs_mask"] = tokenwise_atom_idxs_mask                
        feats["pseudo_cb_coords"] = self._get_pseudo_cb_coords(data)
                    
    
        ### Ligand related features ###
        # Get ligand atom mask
        atom_is_ligand = feats["is_ligand"].gather(dim=-1, index=feats["atom_to_token_map"])
        feats["atom_is_ligand"] = atom_is_ligand
        
        # Get ligand related features
        # try:
        #     feats["atom_is_aromatic"] = torch.tensor(atom_array.is_aromatic).float()
        # except: 
        #     # Todo: LMT & DD6 from 6a2w, no attribute error for is_aromatic.
        #     # Todo: Both have hexagon rings, but no aromaticity. Maybe error in parsing?
        #     feats["atom_is_aromatic"] = torch.full((len(atom_array),), False).float()
        #     print(f"Atom array has no attribute 'is_aromatic' for {data['example_id']}")            
        
        atom_is_metal = np.isin(atom_array.element, aw_const.METAL_ELEMENTS)
        token_is_metal = torch.tensor(apply_token_wise(atom_array, atom_is_metal, np.any))
        feats["token_is_metal"] = token_is_metal
        
        feats["atom_is_metal"] = torch.tensor(atom_is_metal)
        feats["atom_is_small_molecule"] = feats["atom_is_ligand"] * ~feats["atom_is_metal"]        
        feats["atom_is_metal"] = feats["atom_is_metal"].float()
        feats["atom_is_small_molecule"] = feats["atom_is_small_molecule"].float()
        feats["token_is_small_molecule"] = feats["is_ligand"] * ~feats["token_is_metal"]
        
        # Convert to float
        feats["token_is_metal"] = feats["token_is_metal"].float()
        feats["token_is_small_molecule"] = feats["token_is_small_molecule"].float()        
        
        return data
    
    def _get_pseudo_cb_coords(self, data: dict[str, Any]):
        """
        Get pseudo CB coordinates for the atom array.
        """
        atom_array = data["atom_array"]                           
        token_is_protein = data["feats"]["is_protein"]
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