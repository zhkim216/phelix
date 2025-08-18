from pathlib import Path
from typing import Any, List, Union

import lightning as L
import torch

from atomworks.ml.datasets.datasets import BaseDataset, PandasDataset
from atomworks.ml.datasets.parsers import MetadataRowParser
from atomworks.ml.transforms.base import Compose, Transform
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader


class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path

        self._train_set = SDDataset(phase="train",
                                    pdb_path=cfg.pdb_path,
                                    dataset=cfg.dataset,
                                    dataset_parser=cfg.dataset_parser,
                                    transform=cfg.transform,
                                    samples_per_epoch=cfg.samples_per_epoch)


    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self._train_set,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  shuffle=False,  # sampler handles shuffling
                                  drop_last=True,
                                  collate_fn=sd_collator)

        return train_loader


class SDDataset(BaseDataset):
    def __init__(
        self,
        *,
        phase: str,
        pdb_path: str,
        dataset: PandasDataset,
        dataset_parser: MetadataRowParser,
        transform: Transform | Compose | None,
        samples_per_epoch: int,
    ):
        """
        Subclass of AtomWorks StructuralDatasetWrapper to load in cached features.

        Args:
            pdb_path (str): Path to PDB files.
            dataset (Dataset): The dataset to wrap. For example, a PandasDataset, PolarsDataset, or standard PyTorch Dataset.
            dataset_parser (MetadataRowParser): Parser to convert dataset metadata rows into a common dictionary format. See `atomworks.ml.datasets.dataframe_parsers`.
            transform (Transform | Compose, optional): Transformation pipeline to apply to the data. See `atomworks.ml.transforms.base`.
        """
        super().__init__()
        self.phase = phase
        self.pdb_path = pdb_path
        self.cached_feats_dir = f"{pdb_path}/cached_feats"
        self.dataset = dataset
        self.dataset_parser = dataset_parser
        self.transform = transform
        self.samples_per_epoch = samples_per_epoch


    def __getitem__(self, idx: int) -> Any:
        """
        Performs the following steps:
            (1) Retrieve the row at the specified index from the dataset using the __getitem__ method.
            (2) Load the cached features based on the PDB ID in the row.
            (3) Apply train-time transforms to the data.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The processed item.
        """
        idx = 0

        # Get example ID and row
        example_id = self.idx_to_id(idx)
        row = self.dataset[idx]

        # Load in cached features
        feats = self._load_cached_feats(row["pdb_id"])

        # Apply train-time transforms
        feats = self.transform(feats)

        return feats


    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.phase == "train":
            # Train set is infinite
            return self.samples_per_epoch
        return len(self.dataset.manifest.records)


    def __contains__(self, example_id: str) -> bool:
        """Pass through the contains method of the wrapped dataset."""
        return example_id in self.dataset


    def id_to_idx(self, example_id: str) -> int:
        """Pass through the id_to_idx method of the wrapped dataset."""
        return self.dataset.id_to_idx(example_id)


    def idx_to_id(self, idx: int) -> str:
        """Pass through the idx_to_id method of the wrapped dataset."""
        return self.dataset.idx_to_id(idx)


    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped dataset."""
        try:
            # `object.__getattribute__(self, "dataset")` bypasses the custom `__getattr__` and safely retrieves the attribute,
            # avoiding infinite recursion.
            dataset = object.__getattribute__(self, "dataset")
            return getattr(dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object (or its wrapped dataset) has no attribute '{name}'")  # noqa: B904


    def _load_cached_feats(self, pdb_id: str) -> dict[str, torch.Tensor]:
        """Load a cached example from the directory."""
        cached_feats_path = f"{self.cached_feats_dir}/{pdb_id}.pt"
        if not Path(cached_feats_path).exists():
            raise FileNotFoundError(f"Cached features for {pdb_id} not found in {self.cached_feats_dir}")
        return torch.load(cached_feats_path, weights_only=False)


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
    # for key in keys:
    #     values = [d[key] for d in data]

    #     if key not in [
    #         "pdb_key",
    #         "all_coords",
    #         "all_resolved_mask",
    #         "crop_to_all_atom_map",
    #         "chain_symmetries",
    #         "amino_acids_symmetries",
    #         "ligand_symmetries",
    #     ]:
    #         if key == "tokenwise_feats":
    #             # recursively collate tokenwise feats
    #             values = sd_collator(values)
    #         else:
    #             # Check if all have the same shape
    #             shape = values[0].shape
    #             if not all(v.shape == shape for v in values):
    #                 values, _ = pad_to_max(values, 0)
    #             else:
    #                 values = torch.stack(values, dim=0)

    #     # Stack the values
    #     collated[key] = values

    return collated
