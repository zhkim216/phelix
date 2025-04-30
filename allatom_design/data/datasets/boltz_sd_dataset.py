import gzip
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from allatom_design.data.crop.cropper import Cropper
from allatom_design.data.feature.pad import pad_to_max
from allatom_design.data.feature.seq_des_featurizer import \
    SequenceDesignFeaturizer, crop_feats
from allatom_design.data.sample.sampler import Sample, Sampler
from allatom_design.data.tokenize.tokenizer import Tokenized, Tokenizer
from allatom_design.data.types import (Connection, Input, Manifest, Record,
                                       Structure)
from allatom_design.data.data import atom_center_random_augmentation


class BoltzSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path

        # Load in manifest
        manifest = self._load_manifest_from_file()

        # Load in validation split
        with open(f"{self.pdb_path}/splits/validation_ids.txt", "r") as f:
            val_split = {x.lower() for x in f.read().splitlines()}

        train_records = []
        val_records = []
        for record in manifest.records:
            if record.id.lower() in val_split:
                val_records.append(record)
            else:
                train_records.append(record)

        print(f"Number of train records: {len(train_records)}")
        print(f"Number of val records: {len(val_records)}")

        # Filter train records
        train_records = [record for record in train_records if all(f.filter(record) for f in cfg.filters)]
        print(f"Number of train records after applying filters: {len(train_records)}")

        # Create train dataset
        train_manifest = Manifest(records=train_records)
        train_dataset = BoltzSDDataset(self.pdb_path, train_manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer)

        # Create validation dataset
        val_manifest = Manifest(records=val_records)
        val_dataset = BoltzSDDataset(self.pdb_path, val_manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer)

        # Print dataset sizes
        print(f"Training dataset size: {len(train_dataset.manifest.records)}")
        print(f"Validation dataset size: {len(val_dataset.manifest.records)}")

        dataset_wrapper_fn = partial(SDDataset,
                                     samples_per_epoch=cfg.samples_per_epoch,
                                     max_atoms=cfg.max_atoms,
                                     max_tokens=cfg.max_tokens,
                                     pad_to_max_atoms=cfg.pad_to_max_atoms,
                                     pad_to_max_tokens=cfg.pad_to_max_tokens,
                                     atoms_per_window_queries=cfg.atoms_per_window_queries,
                                     num_bins=cfg.num_bins,
                                     )
        self._train_set = dataset_wrapper_fn(dataset=train_dataset, phase="train")
        self._val_set = dataset_wrapper_fn(dataset=val_dataset, phase="val")


    def train_dataloader(self) -> DataLoader:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        train_loader = DataLoader(self._train_set,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  shuffle=False,  # sampler handles shuffling
                                  collate_fn=sd_collator)

        return train_loader


    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        val_loader = DataLoader(self._val_set,
                                batch_size=self.cfg.batch_size,
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=sd_collator)

        return val_loader


    def _load_manifest_from_file(self) -> Manifest:
        """
        Load manifest from file. Preferentially loads from a compressed file, but it if it is not found, will read in an uncompressed json and
        cache the result.
        """
        manifest_path = f"{self.pdb_path}/rcsb_processed_targets/manifest.json.gz"
        if Path(manifest_path).exists():
            print(f"Loading in manifest from {manifest_path}...")
            with gzip.open(manifest_path, "rt") as f:
                data = json.load(f)
            records = [Record.from_dict(r) for r in tqdm(data, desc="Loading records...")]
            # # DEBUG
            # import glob
            # ids = [Path(x).stem for x in glob.glob(f"{self.pdb_path}/processed_targets/featurized/*.npz")]
            # records = [Record.from_dict(r) for r in data if r["id"] in ids]
            manifest = Manifest(records=records)
        else:
            manifest_path = f"{self.pdb_path}/rcsb_processed_targets/manifest.json"
            print(f"Loading in manifest from {manifest_path}...")
            manifest = Manifest.load(Path(manifest_path))
        print(f"Loaded manifest with {len(manifest.records)} records.")
        return manifest


class SDDataset(data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        dataset: "BoltzSDDataset",
        samples_per_epoch: int,
        max_atoms: int,
        max_tokens: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        atoms_per_window_queries: int = 32,
        num_bins: int = 64,
        phase: str = "train",
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.dataset = dataset
        self.probs = dataset.prob
        self.samples_per_epoch = samples_per_epoch
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.atoms_per_window_queries = atoms_per_window_queries
        self.num_bins = num_bins
        self.phase = phase

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
        # Load in data + tokenize (+ possible cropping)
        record_id, feats = self._load_feats(idx)

        # SE3 augmentation for convenience / scaling
        feats["coords"] = atom_center_random_augmentation(feats["coords"], feats["atom_pad_mask"] * feats["atom_resolved_mask"],
                                                          apply_random_augmentation=True,
                                                          translation_scale=1.0,
                                                          return_transforms=False)

        feats["pdb_key"] = record_id
        example = feats
        return example


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


    def _load_feats(self, idx: int) -> tuple[str, dict[str, torch.Tensor]]:
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
                tokenized_cropped, token_crop_mask = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    return_crop_mask=True,
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error: {e}. Skipping.")
            return self._load_feats(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            print(f"No tokens in cropped structure for {sample.record.id}. Skipping.")
            return self._load_feats(idx)

        # Load pre-featurized data and crop
        feats = load_featurized(sample.record, self.dataset.pdb_path)
        feats = crop_feats(feats, token_crop_mask, self.max_tokens, self.max_atoms, self.atoms_per_window_queries)
        feats["coords"] = feats["coords"].squeeze(0)  # squeeze out batch dimension

        return sample.record.id, feats


def load_tokenized(record: Record, pdb_path: str) -> Tokenized:
    """
    Load tokenized data for a given record.
    We pre-tokenize the input structure with tokenwise atom feats so we can speed up dataloading.
    """
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
    return Tokenized(tokens=tokenized["tokens"], bonds=tokenized["bonds"], structure=structure, msa={})


def load_featurized(record: Record, pdb_path: str) -> dict[str, torch.Tensor]:
    """
    Load featurized data for a given record.
    """
    featurized = np.load(f"{pdb_path}/processed_targets/featurized/{record.id}.npz", allow_pickle=True)
    feats = {}
    for k, v in featurized.items():
        feats[k] = torch.from_numpy(v)
    return feats


@dataclass
class BoltzSDDataset:
    """Data holder."""
    pdb_path: str
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: SequenceDesignFeaturizer


def sd_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate sequence denoiser features into a batch.

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
            if key == "tokenwise_feats":
                # recursively collate tokenwise feats
                values = sd_collator(values)
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
