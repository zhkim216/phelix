#!/usr/bin/env python3
from functools import partial
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from atomworks.ml.datasets.datasets import StructuralDatasetWrapper
from atomworks.io.constants import AF3_EXCLUDED_LIGANDS

import torch

from atomworks.enums import ChainType
from atomworks.io.constants import AF3_EXCLUDED_LIGANDS, GAP, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.common import exists
from atomworks.io.common import md5_hash_string

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, AF3SequenceEncoding
from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
)
from atomworks.ml.transforms.atom_frames import (
    AddAtomFrames,
    AddIsRealAtom,
    AddPolymerFrameIndices,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.center_random_augmentation import CenterRandomAugmentation
from atomworks.ml.transforms.chirals import AddAF3ChiralFeatures
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from atomworks.ml.transforms.diffusion.edm import SampleEDMNoise
from atomworks.ml.transforms.dna.pad_dna import PadDNA
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures, EncodeAtomArray
from atomworks.ml.transforms.feature_aggregation.af3 import AggregateFeaturesLikeAF3
from atomworks.ml.transforms.feature_aggregation.confidence import PackageConfidenceFeats
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)
from atomworks.ml.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeAF3,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from atomworks.ml.transforms.rdkit_utils import GetRDKitChiralCenters
from atomworks.ml.transforms.symmetry import FindAutomorphismsWithNetworkX
from atomworks.ml.transforms.template import (
    AddInputFileTemplate,
    AddRFTemplates,
    FeaturizeTemplatesLikeAF3,
    OneHotTemplateRestype,
    RandomSubsampleTemplates,
)



# tame BLAS threads inside each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="cache_features", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve the original config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem

    ### Cache examples ###
    cached_structure_dir = f"{cfg.out_dir}/cached_structures"
    Path(cached_structure_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.dataset.cif_parser_args["cache_dir"] = cached_structure_dir
        cfg.dataset.dataset.name = dataset_name

    cached_example_dir = f"{cfg.out_dir}/cached_examples"
    Path(cached_example_dir).mkdir(parents=True, exist_ok=True)
    cache_fn = partial(_cache_example, cached_example_dir=cached_example_dir)

    # iterate over the dataset, and the caching will happen automatically
    struct_dataset = hydra.utils.instantiate(cfg.dataset, transform=build_transforms())
    if use_parallel:
        indices = range(len(struct_dataset))
        with ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=mp.get_context("forkserver"),
                                 initializer=_init_dataset, initargs=(cfg.dataset,)) as executor:
            for _ in tqdm(executor.map(cache_fn, indices), total=len(struct_dataset), desc="Caching examples"):
                pass
    else:
        for idx in tqdm(range(len(struct_dataset)), desc="Caching examples"):
            cache_fn(idx, dataset=struct_dataset)

_DATASET: StructuralDatasetWrapper | None = None
def _init_dataset(dataset_cfg: DictConfig):
    # initialize the dataset in each worker so that the dataset is not pickled
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg, transform=build_transforms())


def _cache_example(idx: int,
                   cached_example_dir: str,
                   *,
                   dataset: StructuralDatasetWrapper | None = None) -> str:
    example = dataset[idx] if dataset is not None else _DATASET[idx]  # indexing the dataset triggers structure caching

    # save example to disk
    example_id = example["example_id"]
    example_md5_hash = md5_hash_string(example_id)
    torch.save(example, f"{cached_example_dir}/{example_md5_hash}.pt")
    return example_md5_hash


def build_transforms(
    # Preprocessing
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    b_factor_min: float | None = None,
    b_factor_max: float | None = None,
    # Featurization
    conformer_generation_timeout: float = 0.1,  # seconds
    use_element_for_atom_names_of_atomized_tokens: bool = True,
) -> Transform:
    """
    Build a transform pipeline for the dataset.
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


if __name__ == "__main__":
    main()
