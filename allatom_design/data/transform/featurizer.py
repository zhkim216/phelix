
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


def featurizer(
    # Preprocessing
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    b_factor_min: float | None = None,
    b_factor_max: float | None = None,
    # Featurization
    conformer_generation_timeout: float = 5.0,  # seconds
    use_element_for_atom_names_of_atomized_tokens: bool = True,
) -> Transform:
    """
    Build a transform pipeline for featurizing a structure parsed by the AtomWorks CIF parser.
    """
    af3_sequence_encoding = AF3SequenceEncoding()

    # preprocesing transforms
    preprocessing_transforms = [
        RemoveHydrogens(),
        # filter to non-clashing PN units
        FilterToSpecifiedPNUnits(extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"),
        RemoveTerminalOxygen(),
        SetOccToZeroOnBfactor(b_factor_min, b_factor_max),
        RemoveUnresolvedPNUnits(),
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # NOTE: For inference, we must keep UNL to support ligands that are not in the CCD
        HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # featurization transforms
    featurization_transforms = [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        # TODO: for now, we ignore ref pos features because they are too slow to compute
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
    transforms = preprocessing_transforms + featurization_transforms + [SubsetToKeys(keys=keys_to_keep)]
    return Compose(transforms)
