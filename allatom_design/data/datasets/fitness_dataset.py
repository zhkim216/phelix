from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
import os 

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (center_random_augmentation, load_feats_from_pdb,
                                      make_fixed_size_1d)
from allatom_design.data.datasets.ad_dataset import ADDataset

FEATURES_LONG = ("residue_index", "chain_index", "aatype")

class FitDataset(ADDataset):
    def __init__(
        self,
        cfg
    ):
        self.pdb_path = cfg.pdb_path
        self.fixed_size = cfg.fixed_size
        self.phase = cfg.phase
        self.overwrite_cache = cfg.overwrite_cache
        self.csv_path = os.path.join(cfg.pdb_path,'mutations.csv')
        self.mutation_data = self.parse_mutation_csv(self.csv_path)

        # Cache coordinates for faster loading
        self._cache_examples()

    def __len__(self):
        return len(self.mutation_data)

    def _cache_examples(self):
        """
        Cached files are stored in cached_examples/ in the pdb_path.
        """
        cache_dir = f"{self.pdb_path}/cached_examples"

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Caching examples to {cache_dir}...")
        for pdb_key, mut, label, experiment in tqdm(self.mutation_data):
            # Skip if file already exists in cache
            out_file = f"{cache_dir}/{pdb_key}.pt"
            if Path(out_file).exists() and not self.overwrite_cache:
                continue

            # Cache the data
            pdb_data_file = f"{self.pdb_path}/pdbs/{pdb_key}.pdb"
            example = load_feats_from_pdb(pdb_data_file, chain_residx_gap=None)
            torch.save(example, f"{cache_dir}/{pdb_key}.pt")

    def parse_mutation_csv(self, csv_path):
        seq_dataset = pd.read_csv(csv_path)
        mutation_data = [
            (pdb_key, mut, label, experiment)
            for pdb_key, mut, label, experiment in zip(
                seq_dataset['WT_name'],
                seq_dataset['mut_type'],
                seq_dataset['label'],
                seq_dataset.get('experiment', ["none"] * len(seq_dataset)) #optional info to group mutations into their experiments so we can score by experiment
            )]

        return mutation_data

    def get_item(self, pdb_key):
        data_file = self._get_data_file(pdb_key)
        data = torch.load(data_file, weights_only=True)

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
        example["seq_unk_mask"] = (data["aatype"] == rc.restype_order_with_x["X"])
        example['res_b_factors'] = data['res_b_factors']

        # Construct conditioning inputs
        cond_labels_in = {}

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

        return example_out

    def __getitem__(self, idx):
        pdb_key, mut, label, experiment = self.mutation_data[idx]
        pdb_data = self.get_item(pdb_key)
        
        example_out = {
            "pdb_key": pdb_key, 
            "mut": mut, 
            "label": label,
            "experiment": experiment, 
            "pdb_data": pdb_data
        }

        return example_out

        