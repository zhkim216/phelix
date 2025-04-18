import gzip
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from allatom_design.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.pad import pad_to_max, pad_dim
from boltz.data.sample.sampler import Sample, Sampler
from allatom_design.data.tokenize.tokenizer import Tokenizer
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data.data import pad_atom_feats_to_tokenwise, get_atom_feat_masks, atom_center_random_augmentation, atom_apply_random_augmentation
from allatom_design.data.feature.featurizer import SimpleBoltzFeaturizer
from allatom_design.data.motif_selector import get_motif_selector
from allatom_design.data.types import Manifest, Record, Tokenized, Structure, Connection


class BoltzADDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path

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

        # Create datasets
        datasets = {}
        for phase in ["train", "eval", "eval2"]:
            manifest = Manifest(records=records[phase])
            datasets[phase] = BoltzDataset(self.pdb_path, manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer)

        # Print dataset sizes
        for phase in ["train", "eval", "eval2"]:
            print(f"{phase} dataset size: {len(datasets[phase].manifest.records)}")

        dataset_wrapper_fn = partial(ADDataset,
                                     samples_per_epoch=cfg.samples_per_epoch,
                                     max_atoms=cfg.max_atoms,
                                     max_tokens=cfg.max_tokens,
                                     pad_to_max_atoms=cfg.pad_to_max_atoms,
                                     pad_to_max_tokens=cfg.pad_to_max_tokens,
                                     atoms_per_window_queries=cfg.atoms_per_window_queries,
                                     min_dist=cfg.min_dist,
                                     max_dist=cfg.max_dist,
                                     num_bins=cfg.num_bins,
                                     motif_selector_cfg=cfg.motif_selector_cfg,
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
                                    collate_fn=ad_collator)
            val_loaders.append(val_loader)

        return val_loaders


    def _load_manifest_from_file(self) -> Manifest:
        """
        Load manifest from file. Preferentially loads from a compressed file, but it if it is not found, will read in an uncompressed json and
        cache the result.
        """
        manifest_path = f"{self.pdb_path}/processed_targets/manifest.json.gz"
        if Path(manifest_path).exists():
            print(f"Loading in manifest from {manifest_path}...")
            with gzip.open(manifest_path, "rt") as f:
                data = json.load(f)
            records = [Record.from_dict(r) for r in tqdm(data, desc="Loading records...")]
            manifest = Manifest(records=records)
        else:
            manifest_path = f"{self.pdb_path}/processed_targets/manifest.json"
            print(f"Loading in manifest from {manifest_path}...")
            manifest = Manifest.load(Path(manifest_path))
        print(f"Loaded manifest with {len(manifest.records)} records.")
        return manifest


class ADDataset(data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        dataset: "BoltzDataset",
        samples_per_epoch: int,
        max_atoms: int,
        max_tokens: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        phase: str = "train",
        motif_selector_cfg: Optional[DictConfig] = None,
        se3_augment_cfg: Optional[DictConfig] = None,
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
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.phase = phase

        self.se3_augment_cfg = se3_augment_cfg
        self.ms = get_motif_selector(motif_selector_cfg)  # for selecting motifs

        if self.phase == "train":
            records = dataset.manifest.records
            iterator = dataset.sampler.sample(records, np.random)
            self.samples = iterator
        else:
            if self.ms is not None:
                self.ms.eval()


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
        example["diffusion_inputs"] = self._featurize_diffusion_inputs(idx, record_id, tokenized)

        # Featurize motif
        example["motif_inputs"] = self._featurize_motif(idx, record_id, tokenized)

        # Apply random augmentation
        example = self._apply_se3_augmentation(example)

        example["pdb_key"] = record_id
        return example


    def _apply_se3_augmentation(self, example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        N, A, _ = example["diffusion_inputs"]["x"].shape
        if len(example["motif_inputs"]["motif_coords"]) > 0:
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


    def _featurize_motif(self,
                         idx: int,
                         record_id: str,
                         tokenized: Tokenized) -> dict[str, torch.Tensor]:
        """
        Featurize motif.
        """
        # Select motif tokens and subset tokenized data
        if self.ms is not None:
            motif_token_mask = self.ms(tokenized)

        # No motif selector or empty motif; create dummy tokenized data with first residue
        is_dummy_motif = False
        if self.ms is None or motif_token_mask.sum() == 0:
            motif_token_mask = torch.zeros(len(tokenized.tokens))
            motif_token_mask[0] = 1
            is_dummy_motif = True

        # Featurize (possibly dummy) motif
        tokenized_motif = subset_tokenized(tokenized, motif_token_mask)
        try:
            motif_feats = self.dataset.featurizer.process(tokenized_motif,
                                                          atoms_per_window_queries=self.atoms_per_window_queries,
                                                          min_dist=self.min_dist,
                                                          max_dist=self.max_dist,
                                                          num_bins=self.num_bins,
                                                          max_tokens=None,
                                                          max_atoms=None)

            motif_feats["motif_coords"] = motif_feats.pop("coords").squeeze(0)  # coords has a batch dimension of 1 for some reason
        except Exception as e:
            print(f"Featurizer failed to featurize motif tokens on {record_id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Create motif atom mask
        motif_feats.update(get_atom_feat_masks(motif_feats))

        # Select motif atoms
        if self.ms is not None:
            motif_feats["motif_atom_mask"] = self.ms.select_motif_atoms(motif_feats)
        else:
            motif_feats["motif_atom_mask"] = torch.zeros_like(motif_feats["atom_resolved_mask"])

        if is_dummy_motif:
            motif_feats = {k: v.new_zeros((0, *v.shape[1:])) for k, v in motif_feats.items()}  # create dummy features of shape (0, ...)

        return motif_feats


    def _featurize_diffusion_inputs(self,
                                    idx: int,
                                    record_id: str,
                                    tokenized: Tokenized) -> dict[str, torch.Tensor]:
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
        diffusion_feats["x"] = tokenized.tokenwise_atom_feats["coords"]
        diffusion_feats["atom_mask"] = tokenized.tokenwise_atom_feats["atom_resolved_mask"]
        diffusion_feats["residue_index"] = tokenized.tokens["res_idx"]
        diffusion_feats["seq_mask"] = np.ones_like(diffusion_feats["residue_index"])  # denotes padding

        # Convert to torch
        diffusion_feats = {k: torch.from_numpy(v.copy()) for k, v in diffusion_feats.items()}

        # Pad to max tokens
        pad_len = self.max_tokens - diffusion_feats["x"].shape[0]
        for k, v in diffusion_feats.items():
            diffusion_feats[k] = pad_dim(v, 0, pad_len)

        return diffusion_feats


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


def subset_tokenized(tokenized: Tokenized,
                     token_mask: TensorType["n", float]) -> Tokenized:
    """
    Subset tokenized data to only include tokens that are 1 in the token_mask.
    """
    # Subset tokens
    if isinstance(token_mask, torch.Tensor):
        token_mask = token_mask.numpy()

    token_mask = token_mask.astype(bool)
    token_data = tokenized.tokens[token_mask]

    # Subset bonds within the cropped tokens
    indices = token_data["token_idx"]
    token_bonds = tokenized.bonds
    token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
    token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

    tokenized = replace(tokenized, tokens=token_data, bonds=token_bonds)
    return tokenized


@dataclass
class BoltzDataset:
    """Data holder."""
    pdb_path: str
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: SimpleBoltzFeaturizer


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
