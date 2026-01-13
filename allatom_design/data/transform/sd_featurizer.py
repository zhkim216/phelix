from typing import Any
from pathlib import Path
from biotite.structure import AtomArray
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as TensorType
import logging
import atomworks.enums as aw_enums
from atomworks.enums import ChainType
import atomworks.constants as aw_const
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

import allatom_design.data.const as const

# Import custom transforms and constants from custom_transforms
from allatom_design.data.transform.custom_transforms import (
    # Constants
    FEAT_TO_TOKEN_DIM,
    FEAT_TO_ATOM_DIM,
    INFERENCE_ONLY_KEYS,
    # Transform classes
    CheckCoordinatesAreNan,
    FeaturizeCoordsAndMasks,
    PadSDFeats,
    FlattenFeatsDict,
    CenterRandomAugmentation,
    FilterToQueryPNUnits,
    MaskAtomizedTokensInProtein,
    ErrIfAllUnresolved,
    AddCachedResidueData,
    AtomizeShortPolymers,
    AddChainTypeFeaturesForTrain,
    AddChainTypeFeaturesForInference,
    AnnotateLigandPockets,
)

logger = logging.getLogger(__name__)


def sd_featurizer(
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_p: float = 0.0,
    remove_keys: list[str] = [],
    remove_unresolved_tokens: bool = False,
    residue_cache_dir: str | None = "/scratch/users/zhkim216/datasets/atomworks/cached_residue_data",
    max_conformers_per_residue: int | None = 50,
    apply_random_augmentation: bool = True,
    translation_scale: float = 1.0,
    pocket_distance: float = 8.0,
    is_inference: bool = False,
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
        FilterToQueryPNUnits(),
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(),
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
    
    check_coordinates_are_nan = CheckCoordinatesAreNan()
    
    # Featurization
    # NOTE: for now, we ignore ref pos features because they are too slow to compute
    featurization_transforms_post_crop = [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        # AddChainTypeFeaturesForTrain() if not is_inference else AddChainTypeFeaturesForInference(),
        # AddCachedResidueData(residue_cache_dir=residue_cache_dir),
        # GetAF3ReferenceMoleculeFeatures(
        #     save_rdkit_mols=False,
        #     use_cached_conformers=True,
        #     conformer_generation_timeout=5.0,
        #     use_element_for_atom_names_of_atomized_tokens=False,
        #     max_conformers_per_residue=max_conformers_per_residue,
        # ),
        ComputeAtomToTokenMap(),        
        AnnotateLigandPockets(pocket_distance=pocket_distance), 
        ConvertToTorch(keys=["encoded", "feats"]),
        # Handle missing atoms and tokens
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"),        
        # Add features from the atom_array
        FeaturizeCoordsAndMasks(),
        CenterRandomAugmentation(apply_random_augmentation=apply_random_augmentation, 
                                translation_scale=translation_scale),
    ]
    
    transforms = [
        *featurization_transforms_pre_crop,
        check_coordinates_are_nan,
        cropping_transform,
        *featurization_transforms_post_crop,
        PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
        SubsetToKeys(keys=["example_id", "feats", *INFERENCE_ONLY_KEYS]),
        FlattenFeatsDict(),
        RemoveKeys(keys=remove_keys),
    ]

    return Compose(transforms)

def featurizer_af3_prediction(
    max_tokens: int | None = None, 
    max_atoms: int | None = None, 
    remove_keys: list[str] = [],
    remove_unresolved_tokens: bool = False,
) -> Transform:
    """
    Build a transform pipeline for AF3 prediction.
    """
    transforms = [        
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        AddGlobalTokenIdAnnotation(),
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        ComputeAtomToTokenMap(),        
    ]
    return Compose(transforms)

def featurizer_designed_samples(
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    remove_keys: list[str] = [], 
    remove_unresolved_tokens: bool = False,
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure from cif files of designed structures from other methods.
    """
    
    transforms = [
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        AddGlobalTokenIdAnnotation(),            
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),              
        ComputeAtomToTokenMap(),                           
    ]
    
    return Compose(transforms)        

# def sd_featurizer_with_load_any(
#     max_tokens: int | None = None,
#     max_atoms: int | None = None,
#     remove_keys: list[str] = [], 
# ) -> Transform:
#     """
#     Build a transform pipeline that transforms a featurized structure from cif files of designed structures loaded with load_any.
#     Assume necessary preprocessing (e.g., removing clashing PN units) has already been done during the design process.
#     """
    
#     transforms = [
#         AddGlobalTokenIdAnnotation(),            
#         EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),      
#         AddChainTypeFeatrues(), 
#         ComputeAtomToTokenMap(),                   
#         ConvertToTorch(keys=["feats"]),                  
#         FeaturizeCoordsAndMasks(),
#         PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
#         FlattenFeatsDict(),
#         RemoveKeys(keys=remove_keys),
#     ]
    
#     return Compose(transforms)        