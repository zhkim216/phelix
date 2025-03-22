import multiprocessing
import random
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (FEATURES_LONG,
                                      center_random_augmentation,
                                      make_fixed_size_1d)
from allatom_design.data.datasets.multi_dataset import MultiDataset
from allatom_design.data.pdb_utils import write_to_pdb
from allatom_design.data.scaffold_manager import (ScaffoldManager,
                                                  get_scaffold_manager)


class LitADDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: DictConfig, batch_size: int, num_workers: int, cuda: bool):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda

        # Data configs
        self.pdb_paths = data_cfg.pdb_paths


    def train_dataloader(self) -> DataLoader:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        return self.get_dataloader(phase="train")


    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        return [self.get_dataloader(phase="eval"), self.get_dataloader(phase="eval2")]


    def get_dataloader(self, phase: str) -> DataLoader:
        num_datasets = len(self.pdb_paths)

        datasets = [ADDataset(pdb_path=self.pdb_paths[i],
                              phase=phase, **self.data_cfg) for i in range(num_datasets)]
        if phase == "train":
            dataset = MultiDataset(datasets, self.data_cfg.dataset_weights, primary_dset_idx=0)
        elif phase in ["eval", "eval2"]:
            # only use the primary dataset for validation
            dataset = datasets[0]
        else:
            raise ValueError(f"Invalid phase: {phase}")

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.cuda,
                                shuffle=(phase == "train"),
                                drop_last=(phase == "train"))

        return dataloader


class ADDataset(data.Dataset):
    """
    Dataset used for the atom denoiser and sequence denoiser.
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
        max_scrmsd: Optional[float] = None,
        max_rel_rog: Optional[float] = None,
        evaluation_mode: bool = False,
        scaffold_manager_cfg: Optional[DictConfig] = None,
        n_train_cluster_resample: int = 1,
        **kwargs
    ):
        """
        Args:
        - pdb_path: Path to the dataset of PDBs.
        - fixed_size: Input fixed size.
        - phase: "train", "eval", or "test"
        - overfit: Number of examples to overfit on. -1 for all examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - subset_length_range: List with with [min, max] length of proteins to subset form training data
        - scrmsd: for training on only designable structures; subset to only pdbs with scRMSD <= max_scrmsd
        - n_train_cluster_resample: Number of times to resample the training dataset when cluster sampling, since epochs can be very short with cluster sampling
        """
        self.pdb_path = pdb_path
        self.dataset_name = Path(pdb_path).stem
        self.cluster_sample = cluster_sample
        self.fixed_size = fixed_size
        self.phase = phase
        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale
        self.subset_length_range = subset_length_range
        self.evaluation_mode = evaluation_mode
        self.n_train_cluster_resample = n_train_cluster_resample

        self.sm = get_scaffold_manager(scaffold_manager_cfg)  # for constructing scaffolding inputs

        # Require cluster sampling for training on AF3 dataset
        if self.dataset_name in ["af3_pdb", "af3_pdb_monomer", "augmented_af3_monomer_v1"]:
            assert self.cluster_sample, "Cluster sampling must be enabled for AF3 dataset"
        else:
            assert not self.cluster_sample, "Cluster sampling must be disabled for non-AF3 dataset"

        # Read in PDB keys for this phase
        self.pdb_keys_csv = f"{self.pdb_path}/pdb_manifest.csv"
        self.pdb_keys_df = pd.read_csv(self.pdb_keys_csv)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["phase"] == phase]

        # Subset to length range
        if subset_length_range is not None:
            self.subset_to_length_range(*self.subset_length_range)

        # Subset based on scRMSD
        if max_scrmsd is not None:
            self.subset_by_scrmsd(max_scrmsd)

        # Subset based on relative radius of gyration
        if max_rel_rog is not None:
            self.subset_by_rel_rog(max_rel_rog)

        # For training on AF3 datasets, we cluster sample the PDB keys
        if self.cluster_sample:
            self._cluster_sample_pdb_keys(phase=phase, n_train_cluster_resample=self.n_train_cluster_resample)

        # For efficiency set fixed size to max length in the eval or test dataset
        if self.evaluation_mode:
            self.fixed_size = self._get_max_len()

        # Subsetting and overfitting
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
        example = process_single_pdb_ad(data, self.sm, convert_types=False)  # process cached data into an example

        # Center on CA, and if enabled, apply random rotation / translation
        example["x"] = center_random_augmentation(example["x"], example["seq_mask"], example["atom_mask"],
                                                  translation_scale=self.translation_scale, apply_random_augmentation=self.se3_augment)

        # Construct conditioning inputs
        cond_labels_in = {}
        cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["UNCROPPED"]  # condition on cropping

        # Disable cropping for evals
        start_idx = None
        if not self.evaluation_mode:
            start_idx, cond_labels_in = self._crop_examples(example, cond_labels_in, start_idx)

        # Make fixed size example
        fixed_size_example = {}

        for k, v in example.items():
            fixed_size_example[k] = make_fixed_size_1d(v, fixed_size=self.fixed_size, start_idx=start_idx, multimer_crop_mask=None)

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

    def _crop_examples(self, example, cond_labels_in, start_idx):
        # Calculate random cropping start index
        orig_size = example["x"].shape[0]
        extra_len = orig_size - self.fixed_size
        if extra_len > 0:
            start_idx = np.random.choice(np.arange(extra_len + 1))
            cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["CROPPED"]

        return start_idx, cond_labels_in

    def _cluster_sample_pdb_keys(self, phase: str, n_train_cluster_resample: int):
        """
        For training on AF3 datasets, we apply stratified sampling by sampling one PDB key from each cluster.
        """
        if phase == "train":
            # For training, randomly resample N times to get different clusters
            # We do this because cluster sampling can be very short with AF3 datasets, which causes overhead
            print(f"Cluster-resampling dataset {self.n_train_cluster_resample} times...")
            pdb_keys_dfs = []
            for _ in range(n_train_cluster_resample):
                pdb_keys_df = self.pdb_keys_df.copy()
                # randomly select one PDB key from each cluster
                pdb_keys_df = pdb_keys_df.groupby("cluster_id", group_keys=False).apply(lambda g: g.sample(n=1)).reset_index(drop=True)
                pdb_keys_dfs.append(pdb_keys_df)
            self.pdb_keys_df = pd.concat(pdb_keys_dfs, ignore_index=True)
        elif phase in ["eval", "eval2"]:
            # For eval, only take the first PDB in each cluster for deterministic evaluation
            self.pdb_keys_df = self.pdb_keys_df.groupby("cluster_id", as_index=False).first().reset_index(drop=True)


    def _get_data_file(self, pdb_key: str) -> str:
        """
        For a given pdb_key, return the path to the cached data file.
        """
        return f"{self.pdb_path}/cached_examples/{pdb_key}.pt"


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

    def subset_by_scrmsd(self, max_scrmsd: float):
        """
        Subsets the dataset to only include proteins with scRMSD <= max_scrmsd.
        """
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["sc_ca_rmsd"] <= max_scrmsd]

    def subset_by_rel_rog(self, max_rel_rog: float):
        """
        Subsets the dataset to only include proteins with relative radius of gyration <= max_rel_rog.
        """
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["rel_rog"] <= max_rel_rog]



def compute_scale_factors(train_dataloader: DataLoader,
                          n_examples: int = 1000,
                          ) -> Dict[str, Tuple[float, float]]:
    """
    Compute mu and sigma of data based on at least n random examples (rounded up to multiple of batch size).

    Returns a dict mapping from "bb" to (mu, sigma) for backbone features.

    Adapted from: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py
    """
    # Collect x's
    counter = 0
    xs_bb = []

    pbar = tqdm(total=n_examples, desc="Computing scale factors")
    for batch in train_dataloader:
        x = batch["x"]

        # Mask out padding and missing atoms
        mask = batch["x_mask"]

        # Extract backbone atoms
        x_bb = x[..., rc.bb_idxs, :]
        x_bb = x_bb[mask[..., rc.bb_idxs, :].bool()]
        xs_bb.append(x_bb)

        counter += batch["x"].shape[0]
        if counter >= n_examples:
            break
        pbar.update(batch["x"].shape[0])

    pbar.close()

    # Aggregate and compute mean and std
    xs_bb = torch.cat(xs_bb, dim=0)  # [b, n, a_bb, 3]

    mean_bb, std_bb = xs_bb.mean().item(), xs_bb.std().item()

    return {"bb": (mean_bb, std_bb)}


def process_single_pdb_ad(data: dict, sm: ScaffoldManager | None = None, convert_types: bool = True):
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
    example['interface_residue_mask'] = data['interface_residue_mask']
    example['chain_ids'] = data['chain_ids']

    # Get scaffolding input with scaffold manager
    example["x_motif"], example["motif_mask"], example["aatype_motif"], example["x"] = get_scaffolding_inputs(sm, example)

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


def get_scaffolding_inputs(sm: Optional[ScaffoldManager],
                           example: Dict[str, TensorType["..."]]) -> Tuple[TensorType["n 37 3"],
                                                                           TensorType["n 37"],
                                                                           TensorType["n"],
                                                                           TensorType["n 37 3"]]:
    """
    Given a scaffold manager and example, return the scaffolded inputs.
    Centers both the motif and the original coordinates on the CA of the scaffolding residues.

    If sm is None, returns unconditional generation inputs.
    """
    x_recentered = example["x"]
    if sm is None:
        x_motif = torch.zeros_like(example["x"])
        motif_mask = torch.zeros_like(example["atom_mask"])
        aatype_motif = torch.full_like(example["residue_index"], fill_value=rc.restype_order_with_x["X"])
    else:
        sm_outputs = sm(example)
        x_motif = sm_outputs["x_motif"]
        motif_mask = sm_outputs["motif_mask"]
        aatype_motif = sm_outputs["aatype_motif"]
        x_recentered = sm_outputs["x_recentered"]

    return x_motif, motif_mask, aatype_motif, x_recentered
