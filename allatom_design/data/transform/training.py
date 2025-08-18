
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



def sd_featurizer(
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into a training example (including cropping).
    """

    return Identity()
