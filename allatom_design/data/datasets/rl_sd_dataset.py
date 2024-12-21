import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (center_random_augmentation,
                                      load_feats_from_pdb, make_fixed_size_1d)

FEATURES_LONG = ("residue_index", "chain_index", "aatype")


class RLSDDataset(data.Dataset):
    """
    Dataset used for RL on sequence denoiser.
    """
    def __init__(
        self,
        pdb_path: str,
        metric: str = "sc_aa_rmsd",
        min_margin: float = 1.0,
        overfit: int = -1,
        short_epoch: bool = False,
        n_random_subset: Optional[int] = None,
        se3_augment: bool = True,
        translation_scale: float = 1.0,
        overwrite_cache: bool = False,
        subset_length_range: Optional[int] = None,
    ):
        """
        Args:
        - pdb_path: Base path to sampled pdbs. Should contain directory "samples" and a self_consistency_metrics.csv file.
        - metric: metric to use for self-consistency
        - min_margin: minimum margin for self-consistency between pairs (should be positive)
        - overfit: Number of examples to overfit on. -1 for all examples.
        - short_epoch: If True, the dataset will only return 500 random examples.
        - n_random_subset: If not None, the dataset will only return a random subset of n examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - overwrite_cache: If True, overwrite the dataset cache. Useful if the dataset features have been updated.
        - subset_length_range: List with with [min, max] length of proteins to subset from training data
        """
        self.pdb_path = pdb_path
        self.self_consistency_csv = f"{pdb_path}/self_consistency_metrics.csv"

        self.metric = metric
        self.lower_is_better = metric in ["sc_ca_rmsd", "sc_aa_rmsd"]
        self.min_margin = min_margin

        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale

        self.overwrite_cache = overwrite_cache
        self.subset_length_range = subset_length_range

        # Read in self-consistency scores and sample keys
        self.sc_df = pd.read_csv(self.self_consistency_csv)
        self.sample_keys = self.sc_df["pdb_name"].values

        # Cache coordinates for faster loading
        self._cache_examples()

        # Subset to length range
        if subset_length_range is not None:
            self.subset_to_length_range(*self.subset_length_range)  # TODO: switch over to use length from CSV

        # Set fixed size to max length
        # self.fixed_size = self._get_max_len()  # TODO: switch over to use length from CSV
        self.fixed_size = 150

        ### Construct paired dataset ###
        self.paired_dataset = self._construct_paired_dataset()

        # Subsetting and overfitting
        if overfit > 0:
            # Overfit on a subset of the data
            n_data = len(self.paired_dataset)
            np.random.seed(0)  # convenient for reproducibility of overfitting dataset
            self.paired_dataset = np.random.choice(self.paired_dataset, overfit, replace=False).repeat(n_data // overfit)

        if short_epoch:
            self.paired_dataset = np.random.choice(self.paired_dataset, min(500, len(self.paired_dataset)), replace=False)

        if n_random_subset is not None:
            self.paired_dataset = np.random.choice(self.paired_dataset, min(n_random_subset, len(self.paired_dataset)), replace=False)


    def __getitem__(self, idx):
        sample_key_w, sample_key_l = self.paired_dataset[idx]
        data_w = self.get_item(sample_key_w)
        data_l = self.get_item(sample_key_l)

        return {"winner": data_w, "loser": data_l}


    def __len__(self):
        return len(self.paired_dataset)

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
        example['interface_residue_mask'] = data['interface_residue_mask']
        example['chain_ids'] = data['chain_ids']

        # Construct conditioning inputs
        cond_labels_in = {}

        # Set cond labels to default
        cond_labels_in["designability"] = cl.PLACEHOLDER_TOKEN_ID
        cond_labels_in["dataset_source"] = cl.DEFAULT_TOKEN_ID['dataset_source']
        cond_labels_in["crop_aug"] = cl.DEFAULT_TOKEN_ID['crop_aug']

        # Make fixed size example: only padding, no cropping
        fixed_size_example = {}

        for k, v in example.items():
            fixed_size_example[k] = make_fixed_size_1d(v, fixed_size=self.fixed_size, start_idx=None, multimer_crop_mask=None)

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


    def _cache_examples(self):
        """
        Reads in PDB files and caches the examples to disk.
        Cached files are stored in cached_examples/ in the pdb_path.
        """
        cache_dir = f"{self.pdb_path}/cached_examples"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Caching examples to {cache_dir}...")

        # Define the number of workers based on CPU count or set manually
        num_workers = 8
        print(f"Using {num_workers} Workers!")

        # Prepare arguments as tuples (sample_key, cache_dir, overwrite_cache) for cache_sample
        task_args = [(sample_key, cache_dir, self.overwrite_cache, self.pdb_path) for sample_key in self.sample_keys]

        # Use a Pool for parallel processing
        with Pool(processes=num_workers) as pool:
            # Use tqdm to display progress
            for _ in tqdm(pool.imap_unordered(cache_sample, task_args), total=len(task_args), desc="Caching PDBs"):
                pass

        print("Caching completed.")

    def _get_max_len(self):
        """
        Reads in cached PDB files and returns max length of all examples.
        """
        max_len = 0
        for sample_key in tqdm(self.sample_keys, desc=f"Getting max length to use as fixed size", leave=False):
            data_file = self._get_data_file(sample_key)
            example = torch.load(data_file, weights_only=True)
            seq_len = example["seq_mask"].sum().item()
            max_len = seq_len if (seq_len > max_len) else max_len

        return int(max_len)


    def subset_to_length_range(self, min_len: int, max_len: int):
        """
        Subsets the dataset to only include proteins with sequence length in [min_len, max_len].
        """
        sample_keys = []
        for sample_key in tqdm(self.sample_keys, desc=f"Subsetting to length range [{min_len}, {max_len}]", leave=False):
            data_file = self._get_data_file(sample_key)
            example = torch.load(data_file, weights_only=True)
            seq_len = example["seq_mask"].sum().item()
            if min_len <= seq_len <= max_len:
                sample_keys.append(sample_key)

        self.sample_keys = np.array(sample_keys)
        self.sc_df = self.sc_df[self.sc_df["pdb_name"].isin(self.sample_keys)]


    def _construct_paired_dataset(self):
        paired_sample_keys = []

        for _, group in self.sc_df.groupby("pdb_key"):
            indices = group.index.tolist()

            for i, j in itertools.combinations(indices, 2):
                val_i = group.loc[i, self.metric]
                val_j = group.loc[j, self.metric]

                sample_i, sample_j = group.loc[i, "pdb_name"], group.loc[j, "pdb_name"]

                if self.lower_is_better:
                    if val_i <= val_j - self.min_margin:
                        paired_sample_keys.append((sample_i, sample_j))  # i is winner, j is loser
                    elif val_j <= val_i - self.min_margin:
                        paired_sample_keys.append((sample_j, sample_i))  # j is winner, i is loser
                else:
                    if val_i >= val_j + self.min_margin:
                        paired_sample_keys.append((sample_i, sample_j))  # i is winner, j is loser
                    elif val_j >= val_i + self.min_margin:
                        paired_sample_keys.append((sample_j, sample_i))  # j is winner, i is loser

        return paired_sample_keys


def get_sample_file(pdb_path: str, sample_key: str) -> str:
    return f"{pdb_path}/samples/{sample_key}.pdb"


def cache_sample(args):
    sample_key, cache_dir, overwrite_cache, pdb_path = args

    out_file = f"{cache_dir}/{sample_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False

    sample_file = get_sample_file(pdb_path, sample_key)
    example = load_feats_from_pdb(sample_file)
    torch.save(example, out_file)


def contrastive_collate_fn(batch):
    # batch is a list dicts:
    # [ {"winner": data_w_1, "loser": data_l_1}, {"winner": data_w_2, "loser": data_l_2}, ... ]

    # Combine winner and losers into a single batch, in order [w1, l1, w2, l2, ...]
    combined = [example for pair in batch for example in [pair["winner"], pair["loser"]]]
    combined_collated = default_collate(combined)
    return combined_collated
