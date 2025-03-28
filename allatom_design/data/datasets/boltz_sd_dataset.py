from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.pad import pad_to_max
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenized, Tokenizer
from boltz.data.types import (MSA, Connection, Input, Manifest, Record,
                              Structure)
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from allatom_design.data import conversion
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (FEATURES_LONG, atom14_aatype_to_atom37,
                                      atom37_to_atom14,
                                      get_interface_residue_mask)
from allatom_design.data.featurizer import SDFeaturizer


class BoltzSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path

        # Load in manifest
        manifest = Manifest.load(Path(f"{self.pdb_path}/rcsb_processed_targets/manifest.json"))

        # Load in validation split  # TODO: do we need load in test split too?
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
        train_dataset = BoltzDataset(self.pdb_path, train_manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer)

        # Create validation dataset
        val_manifest = Manifest(records=val_records)
        val_dataset = BoltzDataset(self.pdb_path, val_manifest, 1.0, cfg.sampler, cfg.cropper, cfg.tokenizer, cfg.featurizer)

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
                                     min_dist=cfg.min_dist,
                                     max_dist=cfg.max_dist,
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



class SDDataset(data.Dataset):
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
        dataset = self.dataset

        # Get a sample from the dataset
        if self.phase == "train":
            sample: Sample = next(self.samples)
        else:
            # for validation, use deterministic sampling
            record = self.dataset.manifest.records[idx]
            sample = Sample(record=record, chain_id=None, interface_id=None)

        # Get the structure
        input_data = load_input(sample.record, dataset.pdb_path)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

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

        # Compute features
        try:
            features = dataset.featurizer.process(
                tokenized,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
            )
        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Convert Boltz features to seq denoiser features
        try:
            sd_features = self._to_sd_features(input_data, tokenized, features)
        except Exception as e:
            print(f"Failed to convert Boltz features to seq denoiser features for {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        sd_features["pdb_key"] = sample.record.id
        return sd_features


    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


    def _to_sd_features(self,
                        input_data: Input,
                        boltz_tokenized: Tokenized,
                        boltz_feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert Boltz features to seq denoiser features."""
        boltz_restypes = boltz_feats["res_type"].argmax(dim=-1)

        # Pad all tokens to atom23 format (max between 14 for proteins and 23 for nucleic acids)
        tokenwise_feats = pad_atom_feats_to_tokenwise(boltz_feats, max_atoms_per_token=23)

        # TODO: se3 augmentation?

        # === Protein feats === #
        # convert atom mask and coords from atom14 to atom37
        atom14_tokenwise_feats = {k: v[:, :14] for k, v in tokenwise_feats.items()}
        openfold_restypes = torch.tensor([conversion.boltz_token_id_to_restype_id[x.item()] for x in boltz_restypes])  # convert to openfold restypes vocab
        atom_mask = atom14_aatype_to_atom37(atom14_tokenwise_feats["atom_resolved_mask"][..., None], openfold_restypes).squeeze(-1)  # add dummy xyz dimension for conversion
        all_atom_positions = atom14_aatype_to_atom37(atom14_tokenwise_feats["coords"], openfold_restypes) * atom_mask[..., None]
        ref_pos = atom14_aatype_to_atom37(atom14_tokenwise_feats["ref_pos"], openfold_restypes) * atom_mask[..., None]

        # build protein feats in openfold format
        feats = {}
        feats["all_atom_positions"] = all_atom_positions
        feats["all_atom_mask"] = atom_mask
        feats["aatype"] = openfold_restypes
        feats["residue_index"] = boltz_feats["residue_index"]
        feats["chain_index"] = boltz_feats["asym_id"]
        feats["seq_mask"] = torch.ones_like(feats["aatype"], dtype=torch.float32)

        feats["target_feat"] = F.one_hot(feats["aatype"], num_classes=len(rc.restypes_with_x)).float()
        feats["ref_pos"] = ref_pos
        feats["ref_element"] = atom14_tokenwise_feats["ref_element"]
        feats["ref_charge"] = atom14_tokenwise_feats["ref_charge"]

        # subset to protein tokens
        protein_token_mask = boltz_feats["mol_type"] == const.chain_type_ids["PROTEIN"]  # only protein chains
        known_residue_mask = (boltz_restypes != const.token_ids[const.unk_token["PROTEIN"]])  # only known residues; exclude non-standard or unknown residues
        protein_token_mask = protein_token_mask & known_residue_mask  # only protein chains with known residues

        for k, v in feats.items():
            feats[k] = v[protein_token_mask].contiguous()

        # DEBUG: convert back to atom14 and double check that everything looks good
        atom14_coords, atom14_mask = atom37_to_atom14(feats["aatype"], feats["all_atom_positions"], feats["all_atom_mask"])
        protein_tokenwise_mask = atom14_tokenwise_feats["atom_resolved_mask"][protein_token_mask]
        protein_tokenwise_coords = atom14_tokenwise_feats["coords"][protein_token_mask] * protein_tokenwise_mask[..., None]
        if not (atom14_mask == protein_tokenwise_mask).all():
            raise ValueError(f"atom14_mask mismatch")
        if not (atom14_coords * atom14_mask[..., None] == protein_tokenwise_coords).all():
            raise ValueError(f"atom14_coords mismatch")

        # Handle the distinction between missing atoms and ghost atoms in the atom masks
        ghost_atom_mask = 1 - torch.tensor(rc.restype_atom37_mask)[feats["aatype"]]  # 1 for atoms that are not in the residue type; ghost atoms
        missing_atom_mask = (1 - feats["all_atom_mask"]) * (1 - ghost_atom_mask)  # 1 for atoms that are missing in the PDB file; missing if not in atom_mask but not a ghost atom

        feats["ghost_atom_mask"] = ghost_atom_mask  # [n, a]
        feats["missing_atom_mask"] = missing_atom_mask  # [n, a]
        feats["interface_residue_mask"] = get_interface_residue_mask(feats["all_atom_positions"], feats["chain_index"])

        feats["chain_id_mapping"] = {chain_id: asym_id for chain_id, asym_id in zip(input_data.structure.chains["name"],
                                                                                    input_data.structure.chains["asym_id"])}  # get all chain mappings, including invalid chains

        # Save boltz tokenwise feats
        tokenwise_feats_out = {}
        tokenwise_feats_out["atom_positions"] = tokenwise_feats["coords"]
        tokenwise_feats_out["atom_mask"] = tokenwise_feats["atom_resolved_mask"]
        tokenwise_feats_out["res_type"] = boltz_restypes
        tokenwise_feats_out["residue_index"] = boltz_feats["residue_index"]
        tokenwise_feats_out["chain_index"] = boltz_feats["asym_id"]

        tokenwise_feats_out["mol_type"] = boltz_feats["mol_type"]  # [n]
        tokenwise_feats_out["ref_pos"] = tokenwise_feats["ref_pos"]  # [n, a, 3]
        tokenwise_feats_out["ref_element"] = tokenwise_feats["ref_element"]  # [n, a]
        tokenwise_feats_out["ref_charge"] = tokenwise_feats["ref_charge"]  # [n, a]
        tokenwise_feats_out["token_bonds"] = boltz_feats["token_bonds"].squeeze(-1)  # [n, n]

        feats["tokenwise_feats"] = tokenwise_feats_out

        return feats


def load_input(record: Record, pdb_path: str) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    pdb_path : str
        The path to the data directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(f"{pdb_path}/rcsb_processed_targets/structures/{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    return Input(structure, msa={})  # we don't load in the MSAs


@dataclass
class BoltzDataset:
    """Data holder."""
    pdb_path: str
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: SDFeaturizer


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
            "chain_id_mapping",
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


def pad_atom_feats_to_tokenwise(boltz_feats: dict,
                                max_atoms_per_token: int):
    # Build padded atom idxs
    n_atoms_per_token = boltz_feats["atom_to_token"].sum(dim=0)
    # atom_idxs = torch.tensor(tokenized.tokens["atom_idx"])  # this does not work since doesn't account for removal of invalid chains
    atom_idxs = torch.cat([torch.zeros(1), n_atoms_per_token.cumsum(dim=0)[:-1]]).int()
    padded_atom_idxs = atom_idxs[:, None].expand(-1, max_atoms_per_token)
    padded_atom_idxs = padded_atom_idxs + torch.arange(max_atoms_per_token)[None, :]  # [n, 14]
    pad_mask = torch.arange(max_atoms_per_token)[None, :] < n_atoms_per_token[:, None]  # [n, 14]
    padded_atom_idxs = padded_atom_idxs * pad_mask  # mask out ghost atoms

    # Gather from each feature of interest
    tokenwise_feats = {}
    N = padded_atom_idxs.shape[0]
    for k in ["coords", "atom_resolved_mask", "ref_pos", "ref_element", "ref_charge"]:
        v = boltz_feats[k]
        if k == "coords":
            # coords is [1, n_atoms, 3]
            v = v.squeeze(0)
        data_shape = v.shape[1:]
        gather_idxs = padded_atom_idxs.view(-1, *((1,) * len(data_shape))).expand(-1, *data_shape)
        tokenwise_feats[k] = v.gather(0, gather_idxs).view(N, max_atoms_per_token, *data_shape)
        tokenwise_feats[k] = tokenwise_feats[k] * pad_mask.view(N, max_atoms_per_token, *((1,) * len(data_shape)))

    return tokenwise_feats
