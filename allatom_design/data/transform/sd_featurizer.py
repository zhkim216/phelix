
from atomworks.io.constants import (AF3_EXCLUDED_LIGANDS, STANDARD_AA,
                                    STANDARD_DNA, STANDARD_RNA)
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms.atom_array import (AddGlobalAtomIdAnnotation,
                                                AddGlobalTokenIdAnnotation,
                                                AddWithinChainInstanceResIdx,
                                                AddWithinPolyResIdxAnnotation,
                                                ComputeAtomToTokenMap)
from atomworks.ml.transforms.atomize import (AtomizeByCCDName,
                                             FlagNonPolymersForAtomization)
from atomworks.ml.transforms.base import (AddData, Compose, ConditionalRoute,
                                          ConvertToTorch, Identity,
                                          RandomRoute, SubsetToKeys, Transform)
from atomworks.ml.transforms.bfactor_conditioned_transforms import \
    SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.covalent_modifications import \
    FlagAndReassignCovalentModifications
from atomworks.ml.transforms.encoding import (EncodeAF3TokenLevelFeatures,
                                              EncodeAtomArray)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits, HandleUndesiredResTokens, RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen, RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen, RemoveUnresolvedPNUnits)
from atomworks.ml.transforms.crop import (CropContiguousLikeAF3, CropSpatialLikeAF3)


def sd_featurizer(
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_p: float = 0.0,
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into a training example (including cropping).
    """
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
    featurization_transforms = [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=AF3SequenceEncoding()),
        # NOTE: for now, we ignore ref pos features because they are too slow to compute
        # GetAF3ReferenceMoleculeFeatures(
        #     conformer_generation_timeout=conformer_generation_timeout,
        #     use_element_for_atom_names_of_atomized_tokens=use_element_for_atom_names_of_atomized_tokens,
        # ),
        # FindAutomorphismsWithNetworkX(),  # Adds the  "automorphisms" key to the data dictionary
        ComputeAtomToTokenMap(),
        # GetRDKitChiralCenters(),
        # AddAF3ChiralFeatures(),
        AddAF3TokenBondFeatures(),
        ConvertToTorch(keys=["encoded", "feats"]),

        # handle missing atoms and tokens
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"),
    ]

    keys_to_keep = [
        "example_id",
        "feats",
        "atom_array",
        "extra_info",
    ]
    transforms = [cropping_transform] + featurization_transforms + [SubsetToKeys(keys=keys_to_keep)]

    return Compose(transforms)
