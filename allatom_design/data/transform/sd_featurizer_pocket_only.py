"""
Pocket-only featurizers for training and inference.

Crops the atom array to ligand-pocket protein residues + all non-protein
tokens, so the model operates exclusively on pocket regions.

Provides:
    - ``sd_featurizer_pocket_only``: Training featurizer with random noise,
      augmentation, and max_tokens truncation.
    - ``sd_featurizer_pocket_only_for_design``: Inference featurizer with no
      noise/augmentation and no token truncation.
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
from allatom_design.data.transform.crop import CropToPocket, CropSpatialAroundTargetLigand
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
    # ----- Spatial pre-crop around target ligand -----
    spatial_crop_radius: float = 15.0,
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
    1. Pre-crop: filter to query PN units, annotate chain types and target
       ligand, mask unresolved residues.
    2. Spatial pre-crop: ``CropSpatialAroundTargetLigand`` picks a random
       atom from the target ligand and keeps all tokens within
       ``spatial_crop_radius`` angstroms.  This reduces large structures
       to a local region and provides data augmentation.
    3. Pocket crop: ``AnnotateLigandPockets`` marks pocket residues, then
       ``CropToPocket`` keeps only pocket protein tokens + all non-protein
       tokens.  Truncates to ``max_tokens`` by distance to the target
       ligand if needed.
    4. Post-crop: mark all remaining protein residues as pocket, encode
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
    ]

    # ------------------------------------------------------------------
    # Stage 1: Spatial pre-crop around a random target-ligand atom
    # ------------------------------------------------------------------
    spatial_pre_crop = CropSpatialAroundTargetLigand(
        crop_radius=spatial_crop_radius,
        keep_uncropped_atom_array=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: Pocket annotation + pocket crop
    # ------------------------------------------------------------------
    pocket_annotation = AnnotateLigandPockets(pocket_distance=pocket_crop_distance)

    pocket_crop = CropToPocket(
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
        spatial_pre_crop,       # Stage 1: coarse radius crop around target ligand
        pocket_annotation,      # Annotate pocket residues on the reduced structure
        pocket_crop,            # Stage 2: keep only pocket protein + non-protein
        *featurization_transforms_post_crop,
        PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
        SubsetToKeys(keys=["example_id", "feats", *INFERENCE_ONLY_KEYS]),
        FlattenFeatsDict(),
        RemoveKeys(keys=remove_keys),
    ]

    return Compose(transforms)


# ---------------------------------------------------------------------------
# Pocket-only featurizer for inference / design
# ---------------------------------------------------------------------------

def sd_featurizer_pocket_only_for_design(
    # Keys
    remove_keys: list[str] = [],
    # Unresolved tokens
    remove_unresolved_tokens: bool = True,
    # ----- Pocket crop parameters -----
    pocket_crop_distance: float = 5.0,
    # ----- Spatial pre-crop around target ligand -----
    spatial_crop_radius: float = 20.0,
    # ----- Pocket annotation for model features -----
    pocket_distance: float = 6.0,
    use_pocket_rbf: bool = False,
    pocket_rbf_distance: float = 5.0,
    # ----- Padding (None = no fixed padding) -----
    max_tokens: int | None = None,
    max_atoms: int | None = None,
) -> Transform:
    """Build a pocket-only transform pipeline for inference / sequence design.

    This is the inference counterpart of :func:`sd_featurizer_pocket_only`.
    Key differences from the training featurizer:

    * **No training noise or random augmentation.**
    * **No max_tokens truncation** in ``CropToPocket`` -- all pocket
      residues are kept so that the full pocket is designed.
    * **Wider spatial pre-crop** (default 20 A) to accommodate large ligands.
    * ``is_inference`` is always ``True``.

    For native structures the ``AnnotateTargetLigandChains`` transform falls
    back to treating all non-covalent non-polymer units as the target
    ligand, which is equivalent to using all ligand heavy atoms for pocket
    definition.

    Pipeline overview
    -----------------
    1. Pre-crop: filter to query PN units, annotate chain types and target
       ligand, mask / remove unresolved residues.
    2. Spatial pre-crop (20 A default): coarse reduction around target ligand.
    3. Pocket crop: ``AnnotateLigandPockets`` marks pocket residues, then
       ``CropToPocket`` keeps only pocket protein + all non-protein tokens.
    4. Post-crop: mark all remaining protein as pocket, encode features,
       featurize coordinates, pad.
    """

    # ------------------------------------------------------------------
    # Pre-crop transforms
    # ------------------------------------------------------------------
    featurization_transforms_pre_crop = [
        AddData({"is_inference": True}),
        FilterToQueryPNUnits(),
        AnnotateChainTypes(),
        AnnotateTargetLigandChains(n_min_ligand_atoms=1),
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_BACKBONE_ATOM_NAMES,
                aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
            },
        ),
        RemoveUnresolvedTokens() if remove_unresolved_tokens else Identity(),
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(),
    ]

    # ------------------------------------------------------------------
    # Stage 1: Spatial pre-crop (wider for inference)
    # ------------------------------------------------------------------
    spatial_pre_crop = CropSpatialAroundTargetLigand(
        crop_radius=spatial_crop_radius,
        keep_uncropped_atom_array=False,
    )

    # ------------------------------------------------------------------
    # Stage 2: Pocket annotation + pocket crop (no truncation)
    # ------------------------------------------------------------------
    pocket_annotation = AnnotateLigandPockets(pocket_distance=pocket_crop_distance)

    pocket_crop = CropToPocket(
        max_tokens=None,
        keep_uncropped_atom_array=True,
    )

    # ------------------------------------------------------------------
    # Post-crop transforms (no noise / no augmentation)
    # ------------------------------------------------------------------
    featurization_transforms_post_crop = [
        AddGlobalTokenIdAnnotation(),
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        ComputeAtomToTokenMap(),
        MarkAllProteinAsPocket(),        
        AnnotateLigandPocketsPseudoCB(pocket_distance=pocket_rbf_distance) if use_pocket_rbf else Identity(),
        ConvertToTorch(keys=["encoded", "feats"]),
        AddAF3TokenBondFeatures(distance_cutoff=2.4),
        FeaturizeCoordsAndMasks(),
        GetNCACOAndPseudoCBCoords(),
    ]

    transforms = [
        *featurization_transforms_pre_crop,
        spatial_pre_crop,
        pocket_annotation,
        pocket_crop,
        *featurization_transforms_post_crop,
        PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
        SubsetToKeys(keys=["example_id", "feats", *INFERENCE_ONLY_KEYS]),
        FlattenFeatsDict(),
        RemoveKeys(keys=remove_keys),
    ]

    return Compose(transforms)
