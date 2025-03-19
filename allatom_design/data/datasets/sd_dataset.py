from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (FEATURES_LONG,
                                      center_random_augmentation,
                                      make_fixed_size_1d,
                                      transform_sidechain_frame)


class LitSDDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: DictConfig, batch_size: int, num_workers: int, cuda: bool):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda


    def train_dataloader(self) -> DataLoader:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        train_loader = self.get_dataloader(phase="train")
        return train_loader


    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        val_loader = self.get_dataloader(phase="eval")
        return val_loader


    def get_dataloader(self, phase: str) -> DataLoader:
        dataset = SDDataset(phase=phase, **self.data_cfg)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=False,
                                shuffle=(phase == "train"),
                                drop_last=(phase == "train")
                                )
        return dataloader


class SDDataset(data.Dataset):
    """
    Dataset used for the sequence denoiser.
    """

    def __init__(
        self,
        pdb_path: str,
        cluster_sample: bool,
        fixed_size: int,
        phase: str,
        overfit: int = -1,
        se3_augment: bool = True,
        translation_scale: float = 1.0,
        subset_length_range: Optional[int] = None,
        spatial_crop_ratio: float = 0.5,
        evaluation_mode: bool = False,
        **kwargs
    ):
        """
        Args:
        - pdb_path: Path to the dataset of PDBs.
        - fixed_size: Input fixed size.
        - phase: "train", "eval", or "test"
        - overfit: Number of examples to overfit on. -1 for all examples.
        - short_epoch: If True, the dataset will only return 500 random examples.
        - n_random_subset: If not None, the dataset will only return a random subset of n examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - subset_length_range: List with with [min, max] length of proteins to subset form training data
        """
        self.pdb_path = pdb_path
        self.cluster_sample = cluster_sample
        self.fixed_size = fixed_size
        self.phase = phase
        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale
        self.subset_length_range = subset_length_range
        self.spatial_crop_ratio = spatial_crop_ratio
        self.evaluation_mode = evaluation_mode
        self.cluster_sample = cluster_sample

        # Read in PDB keys for this phase
        self.pdb_keys_csv = f"{self.pdb_path}/pdb_manifest.csv"
        self.pdb_keys_df = pd.read_csv(self.pdb_keys_csv)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["phase"] == phase]

        # Subset to length range if specified
        if subset_length_range is not None:
            self.subset_to_length_range(*self.subset_length_range)

        # For training on AF3 datasets, we cluster sample the PDB keys
        if Path(pdb_path).stem in ["af3_pdb", "af3_pdb_monomer"]:
            # require cluster sampling for training on AF3 dataset
            assert self.cluster_sample, "Cluster sampling must be enabled for AF3 dataset"

            print(f"Cluster-resampling dataset...")
            self._cluster_sample_pdb_keys(phase=phase)
        else:
            assert not self.cluster_sample, "Cluster sampling must be disabled for non-AF3 dataset"

        # For efficiency set fixed size to max length in the eval or test dataset
        if self.evaluation_mode:
            self.fixed_size = self._get_max_len()

        # For testing overfitting
        if overfit > 0 and phase == "train":
            # Overfit on a subset of the data
            n_data = len(self.pdb_keys_df)
            np.random.seed(0)  # convenient for reproducibility of overfitting dataset
            indices = np.random.choice(n_data, overfit, replace=False).repeat(n_data // overfit)
            self.pdb_keys_df = self.pdb_keys_df.iloc[indices]


    def __len__(self):
        return len(self.pdb_keys_df)


    def __getitem__(self, idx):
        pdb_key = self.pdb_keys_df["pdb_key"].iloc[idx]
        data = self.get_item(pdb_key)
        return data


    def get_item(self, pdb_key):
        data_file = self._get_data_file(pdb_key)
        data = torch.load(data_file, weights_only=True)  # load in cached load_feats_from_pdb() outputs
        example = process_single_pdb(data, convert_types=False)  # process cached data

        # Center on CA, and if enabled, apply random rotation / translation
        example["x"] = center_random_augmentation(example["x"], example["seq_mask"], example["atom_mask"],
                                                  translation_scale=self.translation_scale, apply_random_augmentation=self.se3_augment)

        # Crop example to fixed size
        start_idx = None
        multimer_crop_mask = None
        if not self.evaluation_mode:
            multimer_crop_mask, start_idx = self._crop_examples(example, multimer_crop_mask, start_idx)

        # Make fixed size example
        fixed_size_example = {}
        for k, v in example.items():
            fixed_size_example[k] = make_fixed_size_1d(v, fixed_size=self.fixed_size, start_idx=start_idx, multimer_crop_mask=multimer_crop_mask)

        # Convert data types
        example_out = {}
        for k, v in fixed_size_example.items():
            if k in FEATURES_LONG:
                example_out[k] = v.long()
            else:
                example_out[k] = v.float()

        # Add pdb_key
        example_out["pdb_key"] = pdb_key

        return example_out


    def _crop_examples(self, example, multimer_crop_mask, start_idx):
        # Calculate random cropping start index
        orig_size = example["x"].shape[0]
        extra_len = orig_size - self.fixed_size
        if extra_len > 0:
            if example['chain_ids'] is not None and len(example['chain_ids']) > 1:
                if torch.rand(1) > self.spatial_crop_ratio:
                    chain_1_len, chain_2_len = torch.sum(example['chain_index'] == 0), torch.sum(example['chain_index'] == 1)
                    multimer_crop_mask = self._multimer_contiguous_crop(chain_1_len, chain_2_len)
                else:
                    multimer_crop_mask = self._multimer_spatial_crop(example['x'], example['interface_residue_mask'])
            else:
                start_idx = np.random.choice(np.arange(extra_len + 1))

        return multimer_crop_mask, start_idx


    def _multimer_contiguous_crop(self, chain_1_len: int, chain_2_len: int) -> TensorType["n", bool]:
        """
        AF3 multichain contiguous cropping implementation.
        """
        #init crop masks w/ all false
        total_len = chain_1_len + chain_2_len
        crop_mask = torch.full((total_len,), False)

        #determine crop sizes
        chain_1_crop_max = min(chain_1_len, self.fixed_size)
        chain_1_crop_min = min(chain_1_len, max(0, self.fixed_size - chain_2_len))
        chain_1_crop = torch.randint(chain_1_crop_min, chain_1_crop_max + 1, (1,))
        chain_2_crop = self.fixed_size - chain_1_crop

        #use crop sizes to sample crop indices
        chain_1_crop_start = torch.randint(0, total_len - self.fixed_size, (1,))
        chain_1_crop_end = chain_1_crop_start + chain_1_crop
        crop_mask[chain_1_crop_start: chain_1_crop_end] = True

        #if chain 2 is inlcuded, add its residues to the mask
        if chain_2_crop > 0:
            chain_2_crop_start = torch.randint(chain_1_crop_end, total_len - chain_2_crop, (1,))
            crop_mask[chain_2_crop_start: chain_2_crop_start + chain_2_crop] = True

        return crop_mask


    def _multimer_spatial_crop(self, x: TensorType["n a 3", bool], interface_residue_mask: TensorType["n", bool]):
        """
        AF3 multichain spatial cropping implementation.
        """
        total_len = x.shape[0]
        crop_mask = torch.full((total_len,), False)  # Initialize crop mask with all False
        x_ca = x[:, 1, :]  # Get C-alpha positions of all residues
        interface_residue_idxs = torch.nonzero(interface_residue_mask).squeeze()

        # Choose a random interface residue index
        chosen_interface_residue_idx = torch.randint(0, len(interface_residue_idxs), (1,))
        chosen_interface_residue_ca_pos = x[chosen_interface_residue_idx, 1, :]

        # Calculate distances to the chosen residue's C-alpha position
        d_interface = x_ca - chosen_interface_residue_ca_pos
        d_interface = torch.sqrt(torch.sum(d_interface ** 2, dim=1))  # Euclidean distance

        # Find the indices of the closest `self.fixed_size` residues
        closest_residue_indices = torch.topk(-d_interface, k=self.fixed_size).indices

        # Set the corresponding positions in `crop_mask` to True
        crop_mask[closest_residue_indices] = True

        return crop_mask


    def _cluster_sample_pdb_keys(self, phase: str):
        """
        Cluster sample the PDB keys to ensure that only one PDB key is selected from each cluster.
        """
        if phase == "train":
            # randomly select one PDB key from each cluster
            print(f"Number of PDB keys before cluster sampling: {len(self.pdb_keys_df)}")
            self.pdb_keys_df["cluster_id"] = self.pdb_keys_df["pdb_key"].str.split("_").str[-1]
            self.pdb_keys_df = self.pdb_keys_df.groupby("cluster_id", group_keys=False).apply(lambda g: g.sample(n=1)).reset_index(drop=True)
            print(f"Number of PDB keys after cluster sampling: {len(self.pdb_keys_df)}")
            print(f"First 10 PDB keys after cluster sampling: {self.pdb_keys_df['pdb_key'].head(10).tolist()}")


    def _get_data_file(self, pdb_key: str) -> str:
        """
        For a given pdb_key, return the path to the cached data file.
        """
        data_file = f"{self.pdb_path}/cached_examples/{pdb_key}.pt"
        return data_file


    def _get_max_len(self):
        """
        Reads in cached PDB files and returns max length of all examples.
        This is only done for eval and test datasets where we do no cropping.
        """
        return int(self.pdb_keys_df["seq_length"].max())


    def subset_to_length_range(self, min_len: int, max_len: int):
        """
        Subsets the dataset to only include proteins with sequence length in [min_len, max_len].
        """
        lengths = self.pdb_keys_df["seq_length"]
        self.pdb_keys_df = self.pdb_keys_df[lengths.between(min_len, max_len)]


def process_single_pdb(data, convert_types=True):
    """
    Process raw PDB data into a standardized format.

    Args:
        data: Dictionary containing PDB data with keys like "all_atom_positions", "all_atom_mask", etc.
        convert_types: Whether to convert data types (float/long) based on FEATURES_LONG

    Returns:
        Dictionary with processed data
    """
    example = {}

    # Use raw coordinates
    x = data["all_atom_positions"]  # [n, a, 3]
    atom_mask = data["all_atom_mask"]  # [n, a]
    seq_mask = data["seq_mask"]  # [n]
    x = x * atom_mask[..., None]  # we first ensure missing & ghost atoms are zeroed out

    # per-channel mask for x, used for loss.
    # We only mask out missing atoms from PDB files, not ghost atoms.
    x_mask = rearrange(1 - data["missing_atom_mask"], "n a -> n a 1").expand_as(x)

    # Construct example
    example["x"] = x * atom_mask[..., None]
    example["seq_mask"] = seq_mask
    example["x_mask"] = x_mask
    example["residue_index"] = data["residue_index"]
    example["chain_index"] = data["chain_index"]
    example["aatype"] = data["aatype"]  # not one-hot encoded
    example["ghost_atom_mask"] = data["ghost_atom_mask"]
    example["missing_atom_mask"] = data["missing_atom_mask"]
    example["atom_mask"] = atom_mask
    example["seq_unk_mask"] = (data["aatype"] == rc.restype_order_with_x["X"])
    example["interface_residue_mask"] = data["interface_residue_mask"]
    example["chain_ids"] = data["chain_ids"]

    # Convert data types
    if convert_types:
        example_out = {}
        for k, v in example.items():
            if k in FEATURES_LONG:
                example_out[k] = v.long()
            else:
                example_out[k] = v.float()
        return example_out

    return example


def compute_scale_factors(train_dataloader: DataLoader,
                          n_examples: int = 1000,
                          ) -> Dict[str, Tuple[float, float]]:
    """
    Compute mu and sigma of data based on at least n random examples (rounded up to multiple of batch size).

    Returns a dict mapping from "bb" to (mu, sigma) for backbone features, and "scn" to (mu, sigma) for sidechain

    Adapted from: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py
    """
    # Collect x's
    counter = 0
    # separate backbone and sidechain features
    xs_bb, xs_scn = [], []

    pbar = tqdm(total=n_examples, desc="Computing scale factors")
    for batch in train_dataloader:
        x = batch["x"]

        # Mask out padding and missing atoms
        mask = batch["x_mask"]

        # scale backbone and sidechain features separately
        x_bb = x[..., rc.bb_idxs, :]
        x_bb = x_bb[mask[..., rc.bb_idxs, :].bool()]

        # Subset to sidechain-only atoms
        x_scn = x[..., rc.non_bb_idxs, :]
        scn_mask = mask[..., rc.non_bb_idxs, :]

        ### Transform sidechains to local coordinates
        x_scn, bb_frames_exists = transform_sidechain_frame(x_scn,
                                                            x[..., rc.bb_idxs, :],
                                                            batch["atom_mask"][..., rc.non_bb_idxs],
                                                            batch["atom_mask"][..., rc.bb_idxs],
                                                            to_local=True)
        scn_mask = scn_mask * rearrange(bb_frames_exists, "b n -> b n 1 1")  # mask out sidechain atoms that don't have a frame
        x_scn = x_scn[scn_mask.bool()]

        xs_scn.append(x_scn)
        xs_bb.append(x_bb)

        counter += batch["x"].shape[0]
        if counter >= n_examples:
            break
        pbar.update(batch["x"].shape[0])

    pbar.close()

    # Aggregate and compute mean and std
    xs_scn = torch.cat(xs_scn, dim=0)  # [b, n, a_scn, 3]
    xs_bb = torch.cat(xs_bb, dim=0)  # [b, n, a_bb, 3]

    mean_bb, std_bb = xs_bb.mean().item(), xs_bb.std().item()
    mean_scn, std_scn = xs_scn.mean().item(), xs_scn.std().item()

    return {"bb": (mean_bb, std_bb), "scn": (mean_scn, std_scn)}

