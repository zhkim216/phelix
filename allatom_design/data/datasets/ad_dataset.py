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
                                      get_scaffolding_inputs,
                                      load_feats_from_pdb, make_fixed_size_1d)
from allatom_design.data.datasets.multi_dataset import MultiDataset
from allatom_design.data.scaffold_manager import get_scaffold_manager
from allatom_design.data.pdb_utils import write_to_pdb


class LitADDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: DictConfig, batch_size: int, num_workers: int, cuda: bool):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda

        # Data configs
        self.pdb_paths = data_cfg.pdb_paths
        self.annotation_csvs = data_cfg.annotation_csvs
        self.overwrite_cache = data_cfg.overwrite_cache
        self.run_eval2 = data_cfg.run_eval2
        self.use_struct_preds = data_cfg.use_struct_preds
        self.phases = ["train", "eval"] if not self.run_eval2 else ["train", "eval", "eval2"]


    def prepare_data(self):
        """
        Called only once on rank 0 in distributed mode, so it is safe for multiprocessing.
        """
        # Cache all examples; save lengths to a CSV

        for pdb_path in self.pdb_paths:
            cache_dir = f"{pdb_path}/cached_examples" if not self.use_struct_preds else f"{pdb_path}/cached_esmfold_examples"
            print(f"Caching examples for {pdb_path}...")
            for phase in self.phases:
                eval2_suffix = "_for_eval2" if self.run_eval2 else ""  # if using eval2, we load in a slightly smaller set of training pdb keys
                pdb_keys_csv = f"{pdb_path}/{phase}_pdb_keys{eval2_suffix}.csv"

                if not Path(pdb_keys_csv).exists():
                    # Backwards compatibility; load in PDB keys from list and save to CSV format
                    pdb_keys_file = f"{pdb_path}/{phase}_pdb_keys{eval2_suffix}.list"
                    with open(pdb_keys_file) as f:
                        pdb_keys = np.array(f.read().splitlines())

                    # cache coordinates for faster loading
                    self._cache_examples(cache_dir, pdb_path, pdb_keys, phase)

                    # get lengths; store them and save to csv
                    pdb_key_to_length = get_lengths(pdb_keys, cache_dir)

                    # save to csv
                    pdb_keys_df = pd.DataFrame({"pdb_key": pdb_keys, "seq_length": [pdb_key_to_length[pdb_key] for pdb_key in pdb_keys]})
                    pdb_keys_df.to_csv(pdb_keys_csv, index=False)

                else:
                    # Load from csv
                    pdb_keys_df = pd.read_csv(pdb_keys_csv)

                    # cache coordinates for faster loading
                    self._cache_examples(cache_dir, pdb_path, pdb_keys_df["pdb_key"], phase)


    def setup(self, stage: Optional[str] = None):
        """
        Lightning calls setup() once (per process).
        """
        pass


    def train_dataloader(self) -> DataLoader:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        """
        train_loader = self.get_dataloader(phase="train")
        return train_loader


    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Called each epoch if reload_dataloaders_every_n_epochs > 0.
        If run_eval2 is True, return both the 'eval' and 'eval2' dataloaders.
        """
        val_loader = self.get_dataloader(phase="eval")

        if self.run_eval2:
            val2_loader = self.get_dataloader(phase="eval2")
            return [val_loader, val2_loader]

        return val_loader


    def get_dataloader(self, phase: str) -> DataLoader:
        num_datasets = len(self.pdb_paths)
        if self.annotation_csvs is None:
            self.annotation_csvs = [None] * num_datasets

        datasets = [ADDataset(pdb_path=self.pdb_paths[i],
                              annotation_csv=self.annotation_csvs[i],
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


    def _cache_examples(self, cache_dir: str, pdb_path: str, pdb_keys: List[str], phase: str):
        """
        Reads in PDB files and caches the examples to disk.
        Cached files are stored in cached_examples/ in the pdb_path.
        """
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Caching {phase} examples to {cache_dir}...")

        # Use 8 workers for parallel processing
        num_workers = 8
        print(f"Using {num_workers} workers for caching...")

        # Prepare arguments as tuples for process_pdb_key
        task_args = [(pdb_key, cache_dir, self.overwrite_cache, pdb_path, phase, self.use_struct_preds)
                     for pdb_key in pdb_keys]

        # Use joblib for parallel processing
        parallel = Parallel(n_jobs=num_workers, verbose=0)
        jobs = [delayed(process_pdb_key)(args) for args in task_args]
        parallel(tqdm(jobs, desc="Caching PDBs", total=len(jobs)))

        print("Caching completed.")


class ADDataset(data.Dataset):
    """
    Dataset used for the atom denoiser and sequence denoiser.
    """

    def __init__(
        self,
        pdb_path: str,
        annotation_csv: Optional[str],
        cluster_sample: bool,
        fixed_size: int,
        phase: str,
        run_eval2: bool,
        overfit: int = -1,
        se3_augment: bool = True,
        translation_scale: float = 1.0,
        overwrite_cache: bool = False,
        subset_length_range: Optional[int] = None,
        max_scrmsd: Optional[float] = None,
        max_rel_rog: Optional[float] = None,
        evaluation_mode: bool = False,
        scaffold_manager_cfg: Optional[DictConfig] = None,
        n_train_cluster_resample: int = 1,
        use_struct_preds: bool = False,
        use_first_sample: bool = False,  # for ablation on ai cath
        **kwargs
    ):
        """
        Args:
        - pdb_path: Path to the dataset of PDBs.
        - annotation_csv: If provided, path to csv containing information about pdb keys.
        - fixed_size: Input fixed size.
        - phase: "train", "eval", or "test"
        - run_eval2: if True, run evals on a random subset of train (need train_pdb_keys_eval2.list, eval2_pdb_keys_eval2.list)
        - overfit: Number of examples to overfit on. -1 for all examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - overwrite_cache: If True, overwrite the dataset cache. Useful if the dataset features have been updated.
        - subset_length_range: List with with [min, max] length of proteins to subset form training data
        - scrmsd: for training on only designable structures; subset to only pdbs with scRMSD <= max_scrmsd
        - n_train_cluster_resample: Number of times to resample the training dataset when cluster sampling, since epochs can be very short with cluster sampling
        - use_struct_preds: if True, load from ESMFold structure predictions instead of crystal structures
        """
        self.pdb_path = pdb_path
        self.annotation_csv = annotation_csv
        self.cluster_sample = cluster_sample
        self.fixed_size = fixed_size
        self.phase = phase
        self.run_eval2 = run_eval2
        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale
        self.overwrite_cache = overwrite_cache
        self.subset_length_range = subset_length_range
        self.evaluation_mode = evaluation_mode
        self.n_train_cluster_resample = n_train_cluster_resample
        self.use_struct_preds = use_struct_preds
        self.use_first_sample = use_first_sample

        self.sm = get_scaffold_manager(scaffold_manager_cfg)  # for constructing scaffolding inputs

        # Require cluster sampling for training on AF3 dataset
        if pdb_path.endswith("af3_pdb") or pdb_path.endswith("af3_pdb_monomer"):
            assert self.cluster_sample, "Cluster sampling must be enabled for AF3 dataset"
        else:
            assert not self.cluster_sample, "Cluster sampling must be disabled for non-AF3 dataset"

        if self.use_first_sample:
            assert pdb_path.endswith("augmented_ingraham_cath_bugfree"), "use_first_sample only supported for ai cath dataset"

        # Read in PDB keys
        eval2_suffix = "_for_eval2" if run_eval2 else ""  # if using eval2, we load in a slightly smaller set of training pdb keys
        self.pdb_keys_csv = f"{self.pdb_path}/{phase}_pdb_keys{eval2_suffix}.csv"
        self.pdb_keys_df = pd.read_csv(self.pdb_keys_csv)

        # Load annotation info
        self._load_annotation_info()

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

        # Ablation on AI-CATH: only use the first sample
        if self.use_first_sample:
            self.pdb_keys_df["original_pdb_key"] = self.pdb_keys_df["pdb_key"].str.split("_").str[0]
            print(f"Number of samples before: {len(self.pdb_keys_df)}")
            self.pdb_keys_df = self.pdb_keys_df.groupby("original_pdb_key", as_index=False).first().reset_index(drop=True)
            print(f"Number of samples after taking first: {len(self.pdb_keys_df)}")

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
        data = torch.load(data_file, weights_only=True)

        example = {}

        # Use raw coordinates
        x = data["all_atom_positions"]  # [n, a, 3]
        atom_mask = data["all_atom_mask"]  # [n, a]
        seq_mask = data["seq_mask"]  # [n]

        x = x * atom_mask[..., None]  # we first ensure missing & ghost atoms are zeroed out

        # Center on CA, and if enabled, apply random rotation / translation
        x = center_random_augmentation(x, seq_mask, atom_mask, translation_scale=self.translation_scale, apply_random_augmentation=self.se3_augment)

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
        if not self.evaluation_mode:
            example['interface_residue_mask'] = data['interface_residue_mask']
            example['chain_ids'] = data['chain_ids']

        # Get scaffolding input with scaffold manager
        example["x_motif"], example["motif_mask"], example["aatype_scaffold"], example["x"] = get_scaffolding_inputs(self.sm, example)

        # Construct conditioning inputs
        cond_labels_in = {}

        # Condition on cropping
        cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["UNCROPPED"]

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
        self.pdb_keys_df["cluster_id"] = self.pdb_keys_df["pdb_key"].str.split("_").str[-1]
        if phase == "train":
            # For training, randomly resample N times to get different clusters
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
        if self.use_struct_preds:
            return f"{self.pdb_path}/cached_esmfold_examples/{pdb_key}.pt"
        return f"{self.pdb_path}/cached_examples/{pdb_key}.pt"


    def _get_max_len(self):
        """
        Reads in cached PDB files and returns max length of all examples.
        This is only done for eval and test datasets where we do no cropping.
        """
        return int(self.pdb_keys_df["seq_length"].max())

    def _load_annotation_info(self) -> None:
        """
        If annotation info is provided, load it as a DataFrame with pdb_key as the index.
        """
        if not self.annotation_csv:
            self.annotation_df = None
            return
        self.annotation_df = pd.read_csv(self.annotation_csv)
        self.annotation_df = self.annotation_df.set_index("pdb_key")


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
        keep_pdb_keys = set(self.annotation_df[self.annotation_df["sc_ca_rmsd"] <= max_scrmsd].index)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["pdb_key"].isin(keep_pdb_keys)]

    def subset_by_rel_rog(self, max_rel_rog: float):
        """
        Subsets the dataset to only include proteins with relative radius of gyration <= max_rel_rog.
        """
        keep_pdb_keys = set(self.annotation_df[self.annotation_df["rel_rog"] <= max_rel_rog].index)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["pdb_key"].isin(keep_pdb_keys)]


def get_lengths(pdb_keys: List[str], cache_dir: str) -> Dict[str, int]:
    """
    Computes sequence lengths for given PDB keys in parallel using joblib.
    Args:
        pdb_keys: List of PDB keys to process
        cache_dir: Directory containing cached examples
    Returns:
        Dictionary mapping PDB keys to their sequence lengths.
    """
    # Use 8 workers for parallel processing
    num_workers = 8
    print(f"Computing sequence lengths using {num_workers} workers...")
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(_get_seq_length)(pdb_key, cache_dir) for pdb_key in pdb_keys]
    results = parallel(tqdm(jobs, desc="Getting lengths", total=len(jobs)))
    return dict(results)

def _get_seq_length(pdb_key: str, cache_dir: str) -> Tuple[str, int]:
    """
    Helper function for parallel processing of sequence lengths.
    Args:
        pdb_key: The PDB key to process
        cache_dir: Directory containing cached examples
    Returns:
        Tuple of (pdb_key, sequence_length)
    """
    data_file = f"{cache_dir}/{pdb_key}.pt"
    example = torch.load(data_file, weights_only=True)
    seq_len = example["seq_mask"].sum().long().item()
    return (pdb_key, seq_len)


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


def get_pdb_data_file(pdb_path: str, phase: str, pdb_key: str, use_struct_preds: bool) -> str:
    if pdb_path.endswith("ingraham_cath_dataset"):  # ingraham splits
        pdb_data_file = f"{pdb_path}/pdb_store/{pdb_key}"
    elif pdb_path.endswith("augmented_ingraham_cath_bugfree"):  # tianyu's augmented dataset
        pdb_data_file = f"{pdb_path}/mpnn_esmfold/{pdb_key}"
        if not Path(pdb_data_file).exists():
            pdb_data_file = f"{pdb_path}/dne_mpnn/{pdb_key}"
    elif pdb_path.endswith("af3_pdb"):
        pdb_data_file = f"{pdb_path}/{phase}_mmcifs/{pdb_key[1:3]}/{pdb_key[:4]}-assembly1.cif" #just use first assembly for now
    elif pdb_path.endswith("af3_pdb_monomer"):
        if not use_struct_preds:
            # use original monomer mmcifs
            mmcif_phase = phase
            if phase == "eval2":
                # grab eval2 from train as well
                mmcif_phase = "train"
            pdb_data_file = f"{pdb_path}/{mmcif_phase}_mmcifs/{pdb_key}.cif"
        else:
            # use ESMFold structure predictions
            pdb_data_file = f"{pdb_path}/esmfold_preds/{pdb_key}.pdb"
    elif pdb_path.endswith("afdb"):  # AFDB augmentation dataset
        pdb_data_file = f"{pdb_path}/foldseek_cluster_reps/{pdb_key}.cif"
    else:
        assert False, f"Unknown dataset: {pdb_path}"
    return pdb_data_file


# Modify process_pdb_key to be standalone if necessary (multiprocessing does not work well with methods)
def process_pdb_key(args):
    pdb_key, cache_dir, overwrite_cache, pdb_path, phase, use_struct_preds = args
    out_file = f"{cache_dir}/{pdb_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False

    pdb_data_file = get_pdb_data_file(pdb_path, phase, pdb_key, use_struct_preds)  # Ensure this function can work independently

    #specific to multimeric af3 dataset
    chain_ids_override = None
    if pdb_path.endswith("af3_pdb"):
        chain_ids_override = pdb_key.split('_')[1]

    example = load_feats_from_pdb(pdb_data_file, chain_ids_override=chain_ids_override, max_conformers=1)
    torch.save(example, out_file)


def cached_example_to_pdb(pt_file: str, out_pdb_file: str, mode: str = "aa", conect: bool = False):
    """
    Load a cached PyTorch file (pt_file) with the expected keys:
      - "aatype"
      - "all_atom_positions"
      - "all_atom_mask"
      - "residue_index"
      - "chain_index"
      - optionally "b_factors"

    Write the structure to 'out_pdb_file' as a PDB file.
    """
    # Load the .pt file
    data = torch.load(pt_file, weights_only=True)

    # Extract required fields
    aatype = data["aatype"]  # shape [n]
    atom_positions = data["all_atom_positions"]  # shape [n, 37, 3]
    atom_mask = data["all_atom_mask"]  # shape [n, 37]
    residue_index = data["residue_index"]  # shape [n]
    chain_index = data["chain_index"]      # shape [n]

    # b_factors might not exist
    b_factors = data.get("b_factors", None)

    # # Center for convenience
    # atom_positions = center_random_augmentation(atom_positions, data["seq_mask"], atom_mask, translation_scale=0.0)

    # Call the write_to_pdb function
    write_to_pdb(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
        filename=out_pdb_file,
        mode=mode,
        conect=conect,
    )

