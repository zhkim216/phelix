import gzip
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.crop.cropper import Cropper
from allatom_design.data.data import (atom_apply_random_augmentation,
                                      atom_center_random_augmentation,
                                      subset_tokenized)
from allatom_design.data.feature.ad_featurizer import ADFeaturizer
from allatom_design.data.feature.motif_featurizer import MotifFeaturizer
from allatom_design.data.feature.pad import pad_dim, pad_to_max
from allatom_design.data.motif_selector import MotifSelector
from allatom_design.data.sample.sampler import Sample, Sampler
from allatom_design.data.tokenize.tokenizer import Tokenizer
from allatom_design.data.types import (Connection, Manifest, Record, Structure,
                                       Tokenized, TokenwiseAtomFeats)
from allatom_design.data.data import pad_atom_feats_to_tokenwise
from allatom_design.data.write.mmcif import write_motif_feats_to_mmcif, write_diffusion_inputs_to_mmcif

DIFFUSION_INPUTS_DTYPES = {
    "residue_index": torch.long,
    "chain_index": torch.long,
    "seq_mask": torch.float,
    "token_index": torch.long,
    "sym_id": torch.long,
    "entity_id": torch.long,
    "label_seq_id": torch.long,
    "auth_seq_id": torch.long,
    "pdb_icode": torch.long,
    "x": torch.float,
    "atom_mask": torch.float,
    "bb_atom_mask": torch.float,
}


class BoltzADDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self.task = cfg.task

        # Load in manifest and records for each phase
        manifest = self._load_manifest_from_file()

        records = {"train": [], "eval": [], "eval2": []}
        for record in manifest.records:
            records[record.phase].append(record)

        # Choose filter sets
        filter_map = {
            "train": cfg.augmented_filters if "augmented" in self.pdb_path else cfg.boltz_filters,
            "eval": cfg.augmented_filters if "augmented" in self.pdb_path else cfg.boltz_val_filters,
            "eval2": cfg.augmented_filters if "augmented" in self.pdb_path else cfg.boltz_val_filters,
        }

        # Filter records
        for phase in ["train", "eval", "eval2"]:
            print(f"Number of {phase} records: {len(records[phase])}")
            records[phase] = [record for record in records[phase] if all(f.filter(record) for f in filter_map[phase] if f is not None)]
            print(f"Number of {phase} records after applying filters: {len(records[phase])}")

        # Overfit
        if cfg.overfit > 0:
            records["train"] = records["train"][:cfg.overfit]

        # Create datasets
        datasets = {}
        for phase in ["train", "eval", "eval2"]:
            manifest = Manifest(records=records[phase])
            datasets[phase] = BoltzADDataset(cfg.task, self.pdb_path, manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer,
                                           cfg.motif_cropper, cfg.motif_featurizer, cfg.motif_selector)

        # Print dataset sizes
        for phase in ["train", "eval", "eval2"]:
            print(f"{phase} dataset size: {len(datasets[phase].manifest.records)}")

        # Validate use_auth_as_residx TODO: this should be removed when we make a new augmented dataset
        if "boltz_v2" in self.pdb_path:
            assert not cfg.use_auth_as_residx, "use_auth_as_residx should be false for boltz_v2"
        elif "augmented_af3_monomer_v2_boltz" in self.pdb_path:
            assert cfg.use_auth_as_residx, "use_auth_as_residx should be true for augmented_af3_monomer_v2_boltz"

        dataset_wrapper_fn = partial(ADDataset,
                                     samples_per_epoch=cfg.samples_per_epoch,
                                     max_tokens=cfg.max_tokens,
                                     max_atoms=cfg.max_atoms,
                                     use_auth_as_residx=cfg.use_auth_as_residx,
                                     **cfg.motif_feats,
                                     se3_augment_cfg=cfg.se3_augment_cfg,
                                     )
        self._train_set = dataset_wrapper_fn(dataset=datasets["train"], phase="train")
        self._val_sets = {"eval": dataset_wrapper_fn(dataset=datasets["eval"], phase="eval"),
                          "eval2": dataset_wrapper_fn(dataset=datasets["eval2"], phase="eval2")}


    def train_dataloader(self) -> DataLoader:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        train_loader = DataLoader(self._train_set,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  shuffle=False,  # sampler handles shuffling
                                  drop_last=True,
                                  collate_fn=ad_collator)

        return train_loader


    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        val_loaders = []
        for phase in ["eval", "eval2"]:
            val_loader = DataLoader(self._val_sets[phase],
                                    batch_size=self.cfg.batch_size,
                                    num_workers=self.cfg.num_workers,
                                    pin_memory=True,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=ad_collator)
            val_loaders.append(val_loader)

        return val_loaders


    def _load_manifest_from_file(self) -> Manifest:
        """
        Load manifest from file. Loads from either a compressed file or uncompressed json.
        """
        processed_targets_dir = f"{self.pdb_path}/processed_targets"
        manifest_path = f"{processed_targets_dir}/manifest.json.gz"

        if Path(manifest_path).exists():
            print(f"Loading in manifest from {manifest_path}...")
            with gzip.open(manifest_path, "rt") as f:
                data = json.load(f)
            records = [Record.from_dict(r) for r in tqdm(data, desc="Loading records...")]
            manifest = Manifest(records=records)
        else:
            manifest_path = f"{processed_targets_dir}/manifest.json"
            print(f"Loading in manifest from {manifest_path}...")
            manifest = Manifest.load(Path(manifest_path))

        print(f"Loaded manifest with {len(manifest.records)} records.")
        return manifest


class ADDataset(data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        dataset: "BoltzADDataset",
        samples_per_epoch: int,
        max_tokens: int,
        max_atoms: int,
        use_auth_as_residx: bool,
        phase: str,
        motif_max_tokens: int | None,
        motif_max_atoms: int | None,
        motif_atoms_per_window_queries: int | None,
        motif_num_bins: int | None,
        se3_augment_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.dataset = dataset
        self.probs = dataset.prob
        self.samples_per_epoch = samples_per_epoch

        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.use_auth_as_residx = use_auth_as_residx
        self.phase = phase
        self.se3_augment_cfg = se3_augment_cfg

        # Motif featurization options
        self.requires_motif = self.dataset.task in ["scaffold"]
        self.motif_max_tokens = motif_max_tokens
        self.motif_max_atoms = motif_max_atoms
        self.motif_atoms_per_window_queries = motif_atoms_per_window_queries
        self.motif_num_bins = motif_num_bins

        if self.phase == "train":
            records = dataset.manifest.records
            iterator = dataset.sampler.sample(records, np.random)
            self.samples = iterator


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, torch.Tensor]
            The sampled data features.

        """
        example = {}

        # Load in data and tokenize (+ possible cropping)
        record_id, tokenized = self._load_and_tokenize_input(idx)

        # Featurize input tokens for diffusion (atom23 protein tokens)
        example["diffusion_inputs"] = featurize_diffusion_inputs(tokenized, self.use_auth_as_residx, self.max_tokens)

        # Featurize motif (all 0s if unconditional or empty motif)
        if self.requires_motif:
            try:
                example["motif_inputs"] = self._featurize_motif_inputs(tokenized)
            except Exception as e:
                print(f"Featurizer failed to featurize motif tokens on {record_id} with error {e}. Skipping.")
                return self.__getitem__(idx)

        # Apply random augmentation
        example = self._apply_se3_augmentation(example)

        # # DEBUG: visualize motifs
        # out_dir = f"out_dir/viz_centered/motifs"
        # Path(out_dir).mkdir(parents=True, exist_ok=True)
        # out_file = f"{out_dir}/{record_id}_motif.cif"
        # if example["motif_inputs"]["motif_atom_mask"].sum() > 0:
        #     try:
        #         write_motif_feats_to_mmcif(example["motif_inputs"], [tokenized.structure], [out_file], keep_auth=True)
        #     except Exception as e:
        #         print(f"Failed to write motif features to {out_file} with error {e}. Skipping.")

        # # DEBUG: visualize diffusion inputs
        # out_dir = f"out_dir/viz_centered/diffusion_inputs"
        # Path(out_dir).mkdir(parents=True, exist_ok=True)
        # out_file = f"{out_dir}/{record_id}_diffusion_inputs.cif"
        # try:
        #     write_ad_feats_to_mmcif(example["diffusion_inputs"], [out_file])
        # except Exception as e:
        #     print(f"Failed to write diffusion inputs to {out_file} with error {e}. Skipping.")


        example["pdb_key"] = record_id
        return example


    def _apply_se3_augmentation(self, example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        example = apply_se3_augmentation(example,
                                         center_on_motif=self.requires_motif,
                                         apply_random_augmentation=self.se3_augment_cfg.enabled,
                                         translation_scale=self.se3_augment_cfg.translation_scale,
                                         return_transforms=False)
        return example


    def _featurize_motif_inputs(self, tokenized: Tokenized) -> dict[str, torch.Tensor]:
        """
        Featurize motif.
        """
        motif_data_kwargs = {
            "motif_max_tokens": self.motif_max_tokens,
            "motif_max_atoms": self.motif_max_atoms,
            "motif_atoms_per_window_queries": self.motif_atoms_per_window_queries,
            "motif_num_bins": self.motif_num_bins,
        }
        motif_feats = featurize_motif_inputs(tokenized, self.use_auth_as_residx, self.dataset.motif_selector, self.dataset.motif_cropper, self.dataset.motif_featurizer, motif_data_kwargs)

        return motif_feats


    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.phase == "train":
            # Train set is infinite
            return self.samples_per_epoch
        return len(self.dataset.manifest.records)


    def _load_and_tokenize_input(self, idx: int) -> tuple[str, Tokenized]:
        """Load Boltz features for a given index."""
        dataset = self.dataset

        # Get a sample from the dataset
        if self.phase == "train":
            sample: Sample = next(self.samples)
        else:
            # for validation, use deterministic sampling
            record = self.dataset.manifest.records[idx]
            sample = Sample(record=record, chain_id=None, interface_id=None)

        # Load pre-tokenized data
        tokenized = load_tokenized(sample.record, dataset.pdb_path)

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        return sample.record.id, tokenized


def load_tokenized(record: Record, pdb_path: str) -> Tokenized:
    """
    Load tokenized data for a given record.
    We pre-tokenize the input structure with tokenwise atom feats so we can speed up dataloading.
    """
    if "boltz_v2" in pdb_path:
        tokenized = np.load(f"{pdb_path}/processed_targets/ad_tokenized/{record.id}.npz", allow_pickle=True)
    else:
        tokenized = np.load(f"{pdb_path}/processed_targets/tokenized/{record.id}.npz", allow_pickle=True)
    structure = tokenized["structure"].item()
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )
    return Tokenized(tokens=tokenized["tokens"], bonds=tokenized["bonds"], structure=structure, msa={}, tokenwise_atom_feats=tokenized["tokenwise_atom_feats"])


def add_tokenwise_atom_feats(tokenized: Tokenized, featurizer: ADFeaturizer) -> Tokenized:
    """
    Add tokenwise atom features to the tokenized structure.

    This is done during pretokenization, allowing us to avoid featurizing the whole structure to obtain backbone coords during training.
    """
    # Featurize input tokens as atom23 tokens
    feats = featurizer.process(tokenized,
                               use_auth_as_residx=False)  # doesn't matter here, since we don't use residue indices from this featurizer
    tokenwise_feats = pad_atom_feats_to_tokenwise(feats, max_atoms_per_token=const.max_num_atoms)  # max number of atoms across any token

    # Construct tokenwise atom feats
    tokenwise_atom_feats = np.empty((tokenwise_feats["coords"].shape[:2]), dtype=TokenwiseAtomFeats)
    tokenwise_atom_feats["coords"] = tokenwise_feats["coords"]
    tokenwise_atom_feats["atom_resolved_mask"] = tokenwise_feats["atom_resolved_mask"]

    # Add tokenwise atom feats to tokenized
    tokenized = replace(tokenized, tokenwise_atom_feats=tokenwise_atom_feats)
    return tokenized


def featurize_diffusion_inputs(tokenized: Tokenized, use_auth_as_residx: bool, max_tokens: int | None) -> dict[str, torch.Tensor]:
    """
    Featurize standard protein tokens into diffusion inputs.
    """
    # Get mask for standard protein tokens
    protein_token_mask = (tokenized.tokens["mol_type"] == const.chain_type_ids["PROTEIN"]) & (tokenized.tokens["is_standard"])

    # Also only include resolved tokens
    protein_token_mask = protein_token_mask * tokenized.tokens["resolved_mask"]

    # Subset tokenized data to only include protein tokens
    tokenized = subset_tokenized(tokenized, protein_token_mask)

    # Construct diffusion features
    diffusion_feats = {}
    diffusion_feats["residue_index"] = tokenized.tokens["auth_seq_id"] if use_auth_as_residx else tokenized.tokens["res_idx"]
    diffusion_feats["chain_index"] = tokenized.tokens["asym_id"]
    diffusion_feats["seq_mask"] = np.ones_like(diffusion_feats["residue_index"])  # denotes padding
    diffusion_feats["token_index"] = tokenized.tokens["token_idx"]
    diffusion_feats["sym_id"] = tokenized.tokens["sym_id"]
    diffusion_feats["entity_id"] = tokenized.tokens["entity_id"]

    # optional features, for saving to mmcif
    diffusion_feats["label_seq_id"] = tokenized.tokens["res_idx"]
    diffusion_feats["auth_seq_id"] = tokenized.tokens["auth_seq_id"]
    diffusion_feats["pdb_icode"] = tokenized.tokens["pdb_icode"]

    # Featurize with ground truth coords and atom mask (for training or partial diffusion)
    diffusion_feats["x"] = tokenized.tokenwise_atom_feats["coords"]
    diffusion_feats["atom_mask"] = tokenized.tokenwise_atom_feats["atom_resolved_mask"]
    diffusion_feats["bb_atom_mask"] = diffusion_feats["atom_mask"][..., const.prot_bb_atom14_idxs]

    # Convert to torch and cast to appropriate dtypes
    diffusion_feats = {k: torch.from_numpy(v.copy()).to(DIFFUSION_INPUTS_DTYPES[k]) for k, v in diffusion_feats.items()}

    # Pad to max tokens
    if max_tokens is not None:
        pad_len = max_tokens - len(diffusion_feats["seq_mask"])
        for k, v in diffusion_feats.items():
            diffusion_feats[k] = pad_dim(v, 0, pad_len)

    return diffusion_feats


def featurize_motif_inputs(tokenized: Tokenized,
                           use_auth_as_residx: bool,
                           motif_selector: MotifSelector,
                           motif_cropper: Cropper,
                           motif_featurizer: MotifFeaturizer,
                           motif_data_kwargs: dict[str, Any],
                           motif_cond_type_cfg: DictConfig | None = None):
    """
    Featurize a motif given a tokenized structure and motif selector, cropper, and featurizer.
    """
    # Select motif tokens and subset tokenized data
    motif_token_mask = motif_selector.select_motif_tokens(tokenized)

    is_dummy_motif = False
    if motif_token_mask.sum() == 0:
        # If empty motif, create dummy tokenized data with the first residue
        motif_token_mask = torch.zeros(len(tokenized.tokens))
        motif_token_mask[0] = 1
        is_dummy_motif = True

    tokenized_motif = subset_tokenized(tokenized, motif_token_mask)

    # Crop motif to the max number of motif tokens / atoms
    tokenized_motif = motif_cropper.crop(tokenized_motif,
                                         max_atoms=motif_data_kwargs["motif_max_atoms"],
                                         max_tokens=motif_data_kwargs["motif_max_tokens"],
                                         random=np.random)

    # Featurize (possibly dummy) motif
    motif_feats = motif_featurizer.process(tokenized_motif,
                                           use_auth_as_residx=use_auth_as_residx,
                                           atoms_per_window_queries=motif_data_kwargs["motif_atoms_per_window_queries"],
                                           num_bins=motif_data_kwargs["motif_num_bins"],
                                           max_tokens=motif_data_kwargs["motif_max_tokens"],
                                           max_atoms=motif_data_kwargs["motif_max_atoms"],
                                           motif_selector=motif_selector)

    if is_dummy_motif:
        motif_feats = {k: torch.zeros_like(v) for k, v in motif_feats.items()}  # create dummy features

    return motif_feats


@dataclass
class BoltzADDataset:
    """Data holder."""
    task: str
    pdb_path: str
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: ADFeaturizer
    motif_cropper: Cropper
    motif_featurizer: MotifFeaturizer
    motif_selector: MotifSelector

def ad_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate atom denoiser features into a batch.

    Parameters
    ----------
    data : list[dict[str, torch.Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, torch.Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "pdb_key",
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            if isinstance(values[0], dict):
                # recursively collate dict inputs
                values = ad_collator(values)
            else:
                # Check if all have the same shape
                shape = values[0].shape
                if not all(v.shape == shape for v in values):
                    values, _ = pad_to_max(values, 0)
                else:
                    values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values
    return collated


def apply_se3_augmentation(example: dict[str, torch.Tensor],
                           center_on_motif: bool,
                           apply_random_augmentation: bool,
                           translation_scale: float,
                           return_transforms: bool = False,
                           ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Apply SE3 augmentation to the example.
    If center_on_motif is True and motif_inputs are present, center on motif atoms.
    Otherwise, center on diffusion input atoms.
    """
    N, A, _ = example["diffusion_inputs"]["x"].shape
    if center_on_motif and example["motif_inputs"]["motif_atom_mask"].sum() > 0:
        # Conditional: center on motif atoms
        x_motif, transforms = atom_center_random_augmentation(example["motif_inputs"]["motif_coords"],
                                                              example["motif_inputs"]["motif_atom_mask"],
                                                              apply_random_augmentation=apply_random_augmentation,
                                                              translation_scale=translation_scale,
                                                              return_transforms=True)
        example["motif_inputs"]["motif_coords"] = x_motif

        # apply transforms to diffusion inputs
        x = atom_apply_random_augmentation(example["diffusion_inputs"]["x"].view(N * A, 3),
                                           example["diffusion_inputs"]["atom_mask"].view(N * A),
                                           transforms)
        example["diffusion_inputs"]["x"] = x.view(N, A, 3)
    else:
        # Unconditional: center on diffusion input atoms
        x, transforms = atom_center_random_augmentation(example["diffusion_inputs"]["x"].view(N * A, 3),
                                                        example["diffusion_inputs"]["atom_mask"].view(N * A),
                                                        apply_random_augmentation=apply_random_augmentation,
                                                        translation_scale=translation_scale,
                                                        return_transforms=True)
        example["diffusion_inputs"]["x"] = x.view(N, A, 3)

    if return_transforms:
        return example, transforms
    return example
