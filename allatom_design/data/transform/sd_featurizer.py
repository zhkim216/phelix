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
from allatom_design.data.transform.crop import (
    CropContiguousLikeAF3,
    CropSpatialLikeAF3Constrained,
    CropSpatialLikeAF3,
)
from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.encoding import (EncodeAF3TokenLevelFeatures,
                                              EncodeAtomArray)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskResiduesWithSpecificUnresolvedAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence)
from atomworks.ml.transforms.filters import (FilterToProteins,
                                             RemoveUnresolvedTokens,
                                             filter_to_specified_pn_units)
import allatom_design.data.const as const
from allatom_design.data.transform.preprocess import AtomizeByCCDName
# Import custom transforms and constants from custom_transforms
from allatom_design.data.transform.bonds import AddAF3TokenBondFeatures
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
    AnnotateLigandPockets,
    AnnotateLigandPocketsPseudoCB,
    GetNCACOAndPseudoCBCoords,
    AddTrainingRandomNoise,
    AddDataCategory,
    RemoveUnsupportedChainTypes,
    AnnotateChainTypes,
    DropOutNonProteinChains,
    
)

logger = logging.getLogger(__name__)

def TrainingRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )

def sd_featurizer(    
    # Model type and inference flag    
    is_inference: bool = False,    
    # Occupancy thresholds for sidechain and backbone atoms
    # occupancy_threshold_sidechain: float = 0.5,
    occupancy_threshold_protein_backbone: float = 0.8,                
    
    # For training random noise
    training_structure_noise: float = 0.1,
        
    undesired_res_names: list[str] = [],
    remove_keys: list[str] = [],
    
    # For randomly dropping out non-protein chains
    drop_prob_non_protein_chains: float = 0.1,
    
    # For removing unresolved tokens
    remove_unresolved_tokens: bool = False,
    
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_p_protein_monomer_chain: float = 0.5,
    # interface crop constraints
    min_protein_tokens_in_interface_crop: int | None = None,
    
    # For pocket annotation
    pocket_distance: float = 8.0,
    # For pocket-aware RBF (pseudo-CB based pocket annotation)
    use_pocket_rbf: bool = False,
    pocket_rbf_distance: float = 5.0,  # e.g. 3.5, 4.5, 5.5
    
    # For reference molecule features
    residue_cache_dir: str | None = "/scratch/users/zhkim216/datasets/atomworks/cached_residue_data",
    max_conformers_per_residue: int | None = 50,
    
    # For center random augmentation
    apply_random_augmentation: bool = True,
    translation_scale: float = 1.0,    
            
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into a training example (including cropping).
    """
    # Featurization that must be done before cropping
    featurization_transforms_pre_crop = [
        # InferenceRoute(StrtoBoolforIsXFeatures()), # Todo: need this later if we want to use load_any
        AddData({"is_inference": is_inference}),    
        AddDataCategory(),    
        FilterToQueryPNUnits(),       
        AnnotateChainTypes(),
        MaskResiduesWithSpecificUnresolvedAtoms(chain_type_to_atom_names={
            aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_BACKBONE_ATOM_NAMES,            
        }, occupancy_threshold=occupancy_threshold_protein_backbone), # Todo: do some experiment with different occupancy thresholds
        MaskResiduesWithSpecificUnresolvedAtoms(chain_type_to_atom_names={            
            aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
        }, occupancy_threshold=0.0), # Todo: do some experiment with different occupancy thresholds        
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(),
    ]

    # Cropping                
    cropping_transform = ConditionalRoute(
        condition_func=lambda data: data.get("data_category"),                        
                transform_map={
                    "protein_monomer_chain": RandomRoute(
                        transforms = [
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
                            )
                        ],
                        probs = [1.0 - crop_spatial_p_protein_monomer_chain, crop_spatial_p_protein_monomer_chain]
                    ),                    
                    "interface": CropSpatialLikeAF3(
                                crop_size=max_tokens,
                                crop_center_cutoff_distance=crop_center_cutoff_distance,
                                keep_uncropped_atom_array=True,
                                max_atoms_in_crop=max_atoms,
                            )
                }
            )
                    
                
                    
    # Featurization
    # NOTE: for now, we ignore ref pos features because they are too slow to compute
    featurization_transforms_post_crop = [        
        ConditionalRoute(
            condition_func=lambda data: data.get("phase") == "train",
            transform_map={True: DropOutNonProteinChains(drop_prob=drop_prob_non_protein_chains), False: Identity()}),    
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),        
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
        AnnotateLigandPocketsPseudoCB(pocket_distance=pocket_rbf_distance) if use_pocket_rbf else Identity(),
        ConvertToTorch(keys=["encoded", "feats"]),
        AddAF3TokenBondFeatures(distance_cutoff=2.4),
        TrainingRoute(CenterRandomAugmentation(apply_random_augmentation=apply_random_augmentation, 
                            translation_scale=translation_scale)),                                            
        TrainingRoute(AddTrainingRandomNoise(noise_scale=training_structure_noise)),       
        # Handle missing atoms and tokens
        # PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        # PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"), 
        # Add features from the atom_array                            
        FeaturizeCoordsAndMasks(),        
        GetNCACOAndPseudoCBCoords(),        
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

def sd_featurizer_for_design(
    # cropping
    # Model type and inference flag    
    is_inference: bool = False,    
    # Occupancy thresholds for sidechain and backbone atoms
    # occupancy_threshold_sidechain: float = 0.5,
    occupancy_threshold_protein_backbone: float = 0.0,                
    
    # For training random noise
    training_structure_noise: float = 0.0,
        
    undesired_res_names: list[str] = [],
    remove_keys: list[str] = [],
    
    # For removing unresolved tokens
    remove_unresolved_tokens: bool = False,
    
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,    
    
    # For pocket annotation
    pocket_distance: float = 8.0,
    # For pocket-aware RBF (pseudo-CB based pocket annotation)
    use_pocket_rbf: bool = False,
    pocket_rbf_distance: float = 5.0,  # e.g. 3.5, 4.5, 5.5
    
    # For reference molecule features
    residue_cache_dir: str | None = "/scratch/users/zhkim216/datasets/atomworks/cached_residue_data",
    max_conformers_per_residue: int | None = 50,
    
    # For center random augmentation
    apply_random_augmentation: bool = True,
    translation_scale: float = 1.0,
    sample_is_designed: bool = False,    
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into an example for designing samples.
    """
    # Featurization that must be done before cropping
    featurization_transforms_pre_crop = [    
        AddData({"is_inference": is_inference}),        
        FilterToQueryPNUnits(),                
        AnnotateChainTypes(), 
        MaskResiduesWithSpecificUnresolvedAtoms(chain_type_to_atom_names={
            aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_BACKBONE_ATOM_NAMES, #! fixed
            aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_BACKBONE_ATOM_NAMES, #! fixed
        }) if not sample_is_designed else Identity(),
        RemoveUnresolvedTokens() if remove_unresolved_tokens and not sample_is_designed else Identity(),        
        ErrIfAllUnresolved(),
    ]
        
    
    
    # Featurization    
    featurization_transforms_post_crop = [
        
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),        
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
        AnnotateLigandPocketsPseudoCB(pocket_distance=pocket_rbf_distance) if use_pocket_rbf else Identity(),
        ConvertToTorch(keys=["encoded", "feats"]),
        # Handle missing atoms and tokens
        # PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        # PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"), 
        # Add features from the atom_array
        AddAF3TokenBondFeatures(distance_cutoff=2.4),
        TrainingRoute(CenterRandomAugmentation(apply_random_augmentation=apply_random_augmentation, 
                            translation_scale=translation_scale)),     
        TrainingRoute(AddTrainingRandomNoise(noise_scale=training_structure_noise)),       
        FeaturizeCoordsAndMasks(),                
        GetNCACOAndPseudoCBCoords(),        
    ]
    
    transforms = [
        *featurization_transforms_pre_crop,
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
        AnnotateChainTypes(),
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
        AnnotateChainTypes(),
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