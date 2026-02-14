"""
Pocket-only featurizer for sequence denoiser training.

Crops the atom array to ligand-pocket protein residues + all non-protein
tokens, so the model is trained exclusively on pocket regions.
"""
from typing import Any

import logging

import atomworks.enums as aw_enums
import atomworks.constants as aw_const
from atomworks.ml.transforms.atom_array import (
    AddGlobalTokenIdAnnotation,
    ComputeAtomToTokenMap,
)
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RemoveKeys,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskResiduesWithSpecificUnresolvedAtoms,
)
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures
from atomworks.ml.transforms.filters import RemoveUnresolvedTokens

import allatom_design.data.const as const
from allatom_design.data.transform.bonds import AddAF3TokenBondFeatures
from allatom_design.data.transform.crop import CropToPocket
from allatom_design.data.transform.custom_transforms import (
    # Constants (re-exported so the dataset module can import them)
    FEAT_TO_TOKEN_DIM,
    FEAT_TO_ATOM_DIM,
    INFERENCE_ONLY_KEYS,
    # Transforms
    FeaturizeCoordsAndMasks,
    PadSDFeats,
    FlattenFeatsDict,
    CenterRandomAugmentation,
    FilterToQueryPNUnits,
    ErrIfAllUnresolved,
    AnnotateLigandPockets,
    AnnotateLigandPocketsPseudoCB,
    GetNCACOAndPseudoCBCoords,
    AddTrainingRandomNoise,
    RemoveUnsupportedChainTypes,
    AnnotateChainTypes,
    MarkAllProteinAsPocket,
    AnnotateTargetLigandChains,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (same pattern as sd_featurizer.py)
# ---------------------------------------------------------------------------

def TrainingRoute(transform: Transform) -> Transform:
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform: Transform) -> Transform:
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )


# ---------------------------------------------------------------------------
# Main pocket-only featurizer
# ---------------------------------------------------------------------------

def sd_featurizer_pocket_only(
    # Model type and inference flag
    is_inference: bool = False,
    # Occupancy thresholds
    occupancy_threshold_protein_backbone: float = 0.8,
    # Training random noise
    training_structure_noise: float = 0.1,
    # Keys
    remove_keys: list[str] = [],
    # Unresolved tokens
    remove_unresolved_tokens: bool = False,
    # ----- Pocket crop parameters -----
    max_tokens: int = 100,
    max_atoms: int | None = 1500,
    pocket_crop_distance: float = 5.0,
    # ----- Pocket annotation for model features -----    
    # For pocket-aware RBF (pseudo-CB based pocket annotation)
    use_pocket_rbf: bool = False,
    pocket_rbf_distance: float = 5.0,
    # Center random augmentation
    apply_random_augmentation: bool = True,
    translation_scale: float = 1.0,
    # Asymmetric training noise
    asymmetric_noise: bool = False,
    protein_noise_scale: float = 0.1,
    context_noise_scale: float = 0.1,
) -> Transform:
    """Build a pocket-only transform pipeline for training the sequence denoiser.

    Pipeline overview
    -----------------
    1. Pre-crop: filter to query PN units, annotate chain types, mask
       unresolved residues, annotate ligand pockets with
       ``pocket_crop_distance``.
    2. Crop: ``CropToPocket`` keeps all non-protein tokens and only the
       protein tokens within the pocket.  Truncates to ``max_tokens`` by
       distance to ligand centre if needed.
    3. Post-crop: mark all remaining protein residues as pocket, encode
       features, add noise, featurize coordinates, pad.
    """

    # ------------------------------------------------------------------
    # Pre-crop transforms
    # ------------------------------------------------------------------
    featurization_transforms_pre_crop = [
        AddData({"is_inference": is_inference}),
        FilterToQueryPNUnits(),
        AnnotateChainTypes(),
        AnnotateTargetLigandChains(n_min_ligand_atoms=1),
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_BACKBONE_ATOM_NAMES,
            },
            occupancy_threshold=occupancy_threshold_protein_backbone,
        ),
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
            },
            occupancy_threshold=0.0,
        ),
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(),
        # Pocket annotation for cropping (using pocket_crop_distance)
        AnnotateLigandPockets(pocket_distance=pocket_crop_distance),
    ]

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------
    cropping_transform = CropToPocket(
        max_tokens=max_tokens,
        keep_uncropped_atom_array=True,
    )

    # ------------------------------------------------------------------
    # Post-crop transforms
    # ------------------------------------------------------------------
    featurization_transforms_post_crop = [
        # No DropOutNonProteinChains — we need ligand context
        AddGlobalTokenIdAnnotation(),
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        ComputeAtomToTokenMap(),
        # Mark all remaining protein residues as pocket (they all are by
        # construction after CropToPocket).  This replaces the post-crop
        # AnnotateLigandPockets used in the regular featurizer.
        MarkAllProteinAsPocket(),
        ConvertToTorch(keys=["encoded", "feats"]),
        AddAF3TokenBondFeatures(distance_cutoff=2.4),
        TrainingRoute(
            CenterRandomAugmentation(
                apply_random_augmentation=apply_random_augmentation,
                translation_scale=translation_scale,
            )
        ),
        TrainingRoute(
            AddTrainingRandomNoise(
                noise_scale=training_structure_noise,
                asymmetric_noise=asymmetric_noise,
                protein_noise_scale=protein_noise_scale,
                context_noise_scale=context_noise_scale,
            )
        ),
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
