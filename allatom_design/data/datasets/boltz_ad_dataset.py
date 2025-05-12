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
from allatom_design.data.feature.featurizer import SimpleBoltzFeaturizer
from allatom_design.data.feature.motif_featurizer import MotifFeaturizer
from allatom_design.data.feature.pad import pad_dim, pad_to_max
from allatom_design.data.motif_selector import MotifSelector
from allatom_design.data.sample.sampler import Sample, Sampler
from allatom_design.data.tokenize.tokenizer import Tokenizer
from allatom_design.data.types import (Connection, Manifest, Record, Structure,
                                       Tokenized)


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

        # Filter records
        for phase in ["train", "eval", "eval2"]:
            print(f"Number of {phase} records: {len(records[phase])}")
            records[phase] = [record for record in records[phase] if all(f.filter(record) for f in cfg.filters)]
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

        dataset_wrapper_fn = partial(ADDataset,
                                     samples_per_epoch=cfg.samples_per_epoch,
                                     max_tokens=cfg.max_tokens,
                                     max_atoms=cfg.max_atoms,
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
        example["diffusion_inputs"] = featurize_diffusion_inputs(tokenized, self.max_tokens, is_sampling=False)

        # Featurize motif (all 0s if unconditional or empty motif)
        if self.requires_motif:
            try:
                example["motif_inputs"] = self._featurize_motif_inputs(tokenized)
            except Exception as e:
                print(f"Featurizer failed to featurize motif tokens on {record_id} with error {e}. Skipping.")
                return self.__getitem__(idx)

        # Apply random augmentation
        example = self._apply_se3_augmentation(example)

        example["pdb_key"] = record_id
        return example


    def _apply_se3_augmentation(self, example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        N, A, _ = example["diffusion_inputs"]["x"].shape
        if self.requires_motif and example["motif_inputs"]["token_pad_mask"].sum() > 0:
            # Conditional: center on motif atoms
            x_motif, transforms = atom_center_random_augmentation(example["motif_inputs"]["motif_coords"],
                                                                  example["motif_inputs"]["motif_atom_mask"],
                                                                  apply_random_augmentation=self.se3_augment_cfg.enabled,
                                                                  translation_scale=self.se3_augment_cfg.translation_scale,
                                                                  return_transforms=True)
            example["motif_inputs"]["motif_coords"] = x_motif

            # apply transforms to diffusion inputs
            x = atom_apply_random_augmentation(example["diffusion_inputs"]["x"].view(N * A, 3),
                                               example["diffusion_inputs"]["atom_mask"].view(N * A),
                                               transforms)
            example["diffusion_inputs"]["x"] = x.view(N, A, 3)
        else:
            # Unconditional: center on diffusion input atoms
            x = atom_center_random_augmentation(example["diffusion_inputs"]["x"].view(N * A, 3),
                                                example["diffusion_inputs"]["atom_mask"].view(N * A),
                                                apply_random_augmentation=self.se3_augment_cfg.enabled,
                                                translation_scale=self.se3_augment_cfg.translation_scale,
                                                return_transforms=False)
            example["diffusion_inputs"]["x"] = x.view(N, A, 3)

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
        motif_feats = featurize_motif_inputs(tokenized, self.dataset.motif_selector, self.dataset.motif_cropper, self.dataset.motif_featurizer, motif_data_kwargs)

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


def featurize_diffusion_inputs(tokenized: Tokenized, max_tokens: int, is_sampling: bool) -> dict[str, torch.Tensor]:
    """
    Featurize protein tokens into diffusion inputs.
    """
    # Get mask for protein tokens
    protein_token_mask = tokenized.tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
    known_residue_mask = tokenized.tokens["res_type"] != const.token_ids[const.unk_token["PROTEIN"]]
    protein_token_mask = protein_token_mask * known_residue_mask

    # Subset tokenized data to only include protein tokens
    tokenized = subset_tokenized(tokenized, protein_token_mask)

    # Construct diffusion features
    diffusion_feats = {}
    diffusion_feats["residue_index"] = tokenized.tokens["auth_seq_id"]  # we use auth_seq_id since predicted structures won't have SEQRES records
    diffusion_feats["seq_mask"] = np.ones_like(diffusion_feats["residue_index"])  # denotes padding
    if not is_sampling:
        # During training, featurize with ground truth coords and atom mask
        diffusion_feats["x"] = tokenized.tokenwise_atom_feats["coords"]
        diffusion_feats["atom_mask"] = tokenized.tokenwise_atom_feats["atom_resolved_mask"]

    # Convert to torch
    diffusion_feats = {k: torch.from_numpy(v.copy()) for k, v in diffusion_feats.items()}

    # Pad to max tokens
    pad_len = max_tokens - len(diffusion_feats["seq_mask"])
    for k, v in diffusion_feats.items():
        diffusion_feats[k] = pad_dim(v, 0, pad_len)

    return diffusion_feats


def featurize_motif_inputs(tokenized: Tokenized,
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
                                           use_auth_seq_id=True,  # we use auth_seq_id since predicted structures won't have SEQRES records
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
    featurizer: SimpleBoltzFeaturizer
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
