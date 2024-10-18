from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (center_random_augmentation, load_feats_from_pdb,
                                      make_fixed_size_1d)

FEATURES_LONG = ("residue_index", "chain_index", "aatype")


class ADDataset(data.Dataset):
    """
    Dataset used for the atom denoiser and sequence denoiser.
    """

    def __init__(
        self,
        pdb_path: str,
        fixed_size: int,
        phase: str,
        designability_csv: Optional[str] = None,
        overfit: int = -1,
        short_epoch: bool = False,
        n_random_subset: Optional[int] = None,
        se3_augment: bool = True,
        translation_scale: float = 1.0,
        overwrite_cache: bool = False,
        **kwargs
    ):
        """
        Args:
        - pdb_path: Path to the dataset of PDBs.
        - fixed_size: Input fixed size.
        - phase: "train", "eval", or "test"
        - designability_csv: Path to a CSV with designability info for the dataset.
        - overfit: Number of examples to overfit on. -1 for all examples.
        - short_epoch: If True, the dataset will only return 500 random examples.
        - n_random_subset: If not None, the dataset will only return a random subset of n examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - overwrite_cache: If True, overwrite the dataset cache. Useful if the dataset features have been updated.
        """
        self.pdb_path = pdb_path
        self.fixed_size = fixed_size
        self.phase = phase
        self.designability_csv = designability_csv
        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale
        self.overwrite_cache = overwrite_cache

        # Read in PDB keys
        self.pdb_keys_file = f"{self.pdb_path}/{phase}_pdb_keys.list"

        with open(self.pdb_keys_file) as f:
            self.pdb_keys = np.array(f.read().split("\n")[:-1])

        # Cache coordinates for faster loading
        self._cache_examples()

        # Load designability info
        self._load_designability_info()

        # Get dataset source label
        self.dataset_source_label = self._get_dataset_source_label()

        # Subsetting and overfitting
        if overfit > 0 and phase == "train":
            # Overfit on a subset of the data
            n_data = len(self.pdb_keys)
            np.random.seed(0)  # convenient for reproducibility of overfitting dataset
            self.pdb_keys = np.random.choice(self.pdb_keys, overfit, replace=False).repeat(n_data // overfit)

        if short_epoch:
            self.pdb_keys = np.random.choice(self.pdb_keys, min(500, len(self.pdb_keys)), replace=False)

        if n_random_subset is not None:
            self.pdb_keys = np.random.choice(self.pdb_keys, min(n_random_subset, len(self.pdb_keys)), replace=False)


    def __len__(self):
        return len(self.pdb_keys)


    def __getitem__(self, idx):
        pdb_key = self.pdb_keys[idx]
        data = self.get_item(pdb_key)
        return data


    def get_item(self, pdb_key):
        data_file = self._get_data_file(pdb_key)
        data = torch.load(data_file, weights_only=True)

        example = {}

        # Use raw coordinates
        x = data["all_atom_positions"]  # [n, a, 3]
        atom_mask = data["all_atom_mask"]  # [n, a]
        seq_mask = data["seq_mask"]  # [n]

        x = x * atom_mask[..., None]  # we first ensure missing & ghost atoms are zeroed out

        if self.se3_augment:
            # Center on CA and apply random rotation
            x = center_random_augmentation(x, seq_mask, atom_mask, data["missing_atom_mask"],translation_scale=self.translation_scale)

        # per-channel mask for x, used for loss.
        # We only mask out missing atoms from PDB files, not ghost atoms.
        x_mask = rearrange(1 - data["missing_atom_mask"], "n a -> n a 1").expand_as(x)

        # Construct example
        x = x * atom_mask[..., None]  # ensure missing & ghost atoms are zeroed out
        example["x"] = x
        example["seq_mask"] = seq_mask
        example["x_mask"] = x_mask
        example["residue_index"] = data["residue_index"]
        example["chain_index"] = data["chain_index"]
        example["aatype"] = data["aatype"]  # not one-hot encoded
        example["ghost_atom_mask"] = data["ghost_atom_mask"]
        example["missing_atom_mask"] = data["missing_atom_mask"]
        example["atom_mask"] = atom_mask

        # Construct conditioning inputs
        cond_labels_in = {}

        # Add designability info
        cond_labels_in["designability"] = cl.PLACEHOLDER_TOKEN_ID
        if self.designability_csv:
            cond_labels_in["designability"] = self.pdb_to_designability[pdb_key]

        # Add dataset source label
        cond_labels_in["dataset_source"] = cl.TOKEN_TO_ID["dataset_source"][self.dataset_source_label]

        # Calculate random cropping start index
        orig_size = example["x"].shape[0]
        extra_len = orig_size - self.fixed_size
        if extra_len > 0:
            start_idx = np.random.choice(np.arange(extra_len + 1))
            cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["CROPPED"]
        else:
            start_idx = None
            cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["UNCROPPED"]

        # Make fixed size example
        fixed_size_example = {}

        for k, v in example.items():
            fixed_size_example[k] = make_fixed_size_1d(v, fixed_size=self.fixed_size, start_idx=start_idx)

        # Convert data types
        example_out = {}
        for k, v in fixed_size_example.items():
            if k in FEATURES_LONG:
                example_out[k] = v.long()
            else:
                example_out[k] = v.float()

        # Add pdb_key
        example_out["pdb_key"] = pdb_key

        # Add conditioning labels
        example_out["cond_labels_in"] = cond_labels_in

        return example_out


    def _get_data_file(self, pdb_key: str) -> str:
        """
        For a given pdb_key, return the path to the cached data file.
        """
        data_file = f"{self.pdb_path}/cached_examples/{pdb_key}.pt"
        return data_file


    def _get_pdb_data_file(self, pdb_key: str) -> str:
        if self.pdb_path.endswith("ingraham_cath_dataset"):  # ingraham splits
            pdb_data_file = f"{self.pdb_path}/pdb_store/{pdb_key}"
        elif self.pdb_path.endswith("afdb"):  # AFDB augmentation dataset
            pdb_data_file = f"{self.pdb_path}/foldseek_cluster_reps/{pdb_key}.cif"
        elif self.pdb_path.endswith("qfit-test-set/rcsb-pdb"):
            pdb_data_file = f"{self.pdb_path}/all/{pdb_key}.pdb1"  # qfit dataset, use only pdb1s for now
        elif self.pdb_path.endswith("rcsb_test_cases"):
            pdb_data_file = f"{self.pdb_path}/pdbs/{pdb_key}.pdb"
        else:
            assert False, f"Unknown dataset: {self.pdb_path}"
        return pdb_data_file


    def _get_dataset_source_label(self) -> str:
        if self.pdb_path.endswith("ingraham_cath_dataset"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("afdb"):
            dataset_source_label = "SYNTHETIC"
        elif self.pdb_path.endswith("qfit-test-set/rcsb-pdb"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("rcsb_test_cases"):
            dataset_source_label = "EXPERIMENTAL"
        else:
            assert False, f"Unknown dataset: {self.pdb_path}"
        return dataset_source_label


    def _cache_examples(self):
        """
        Reads in PDB files and caches the examples to disk.
        Cached files are stored in cached_examples/ in the pdb_path.
        """
        cache_dir = f"{self.pdb_path}/cached_examples"

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Caching examples to {cache_dir}...")
        for pdb_key in tqdm(self.pdb_keys):
            # Skip if file already exists in cache
            out_file = f"{cache_dir}/{pdb_key}.pt"
            if Path(out_file).exists() and not self.overwrite_cache:
                continue

            # Cache the data
            pdb_data_file = self._get_pdb_data_file(pdb_key)
            example = load_feats_from_pdb(pdb_data_file, chain_residx_gap=None)
            torch.save(example, f"{cache_dir}/{pdb_key}.pt")


    def _load_designability_info(self) -> None:
        """
        If designability info is provided, load it into a mapping from pdb key to designability (0 or 1).
        """
        if not self.designability_csv:
            return
        designability_df = pd.read_csv(self.designability_csv)
        self.pdb_to_designability = designability_df.set_index("pdb")["designable"].to_dict()
        self.pdb_to_designability["3f5hA"] = 0  # 3f5hA is missing from the designability dataset


    def subset_to_length_range(self, min_len: int, max_len: int):
        """
        Subsets the dataset to only include proteins with sequence length in [min_len, max_len].
        """
        pdb_keys = []
        for pdb_key in tqdm(self.pdb_keys, desc=f"Subsetting to length range [{min_len}, {max_len}]", leave=False):
            data_file = self._get_data_file(pdb_key)
            example = torch.load(data_file, weights_only=True)
            seq_len = example["seq_mask"].sum().item()
            if min_len <= seq_len <= max_len:
                pdb_keys.append(pdb_key)

        self.pdb_keys = np.array(pdb_keys)


    @staticmethod
    def index_into_batch(batch: Dict[str, torch.Tensor], idxs: List) -> Dict[str, torch.Tensor]:
        """
        Helper method to index into a batch of data, because batch may contain nested dictionaries.
        """
        if isinstance(batch, dict):
            return {key: ADDataset.index_into_batch(value, idxs) for key, value in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch[idxs]
        elif isinstance(batch, List):
            return [batch[i] for i in idxs]
        else:
            data_type = type(batch)
            raise ValueError(f"Unsupported data type {data_type} in batch. Expected a dict or torch.Tensor.")


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

        ### Center sidechain on CA
        x_scn = x_scn - x[..., 1:2, :]
        scn_missing_atom_mask = batch["missing_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are missing
        x_scn = torch.where(scn_missing_atom_mask[..., None].bool(), 0, x_scn)  # fill missing atoms with zeroes
        scn_ghost_atom_mask = batch["ghost_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are not in the residue type
        x_scn = torch.where(scn_ghost_atom_mask[..., None].bool(), 0, x_scn)  # fill ghost atoms with zeroes

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
