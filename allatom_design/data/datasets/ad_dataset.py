from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from itertools import groupby
import random
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from torchtyping import TensorType
import multiprocessing
from multiprocessing import Pool

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (center_random_augmentation,
                                      load_feats_from_pdb, make_fixed_size_1d, transform_sidechain_frame)


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
        subset_length_range: Optional[int] = None,
        cluster_sample: bool = True,
        afdb_res_plddt_cutoff: float = 0.0,
        spatial_crop_ratio: float = 0.5,
        evaluation_mode: bool = False,
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
        - subset_length_range: List with with [min, max] length of proteins to subset form training data
        - afdb_res_plddt_cutoff: If > 0, for AFDB dataset, cut out any residues with PLDDT < cutoff
        """
        self.pdb_path = pdb_path
        self.fixed_size = fixed_size
        self.phase = phase
        self.designability_csv = designability_csv
        self.overfit = overfit

        self.se3_augment = se3_augment
        self.translation_scale = translation_scale
        self.overwrite_cache = overwrite_cache
        self.subset_length_range = subset_length_range
        self.afdb_res_plddt_cutoff = afdb_res_plddt_cutoff
        self.cluster_sample = cluster_sample
        self.spatial_crop_ratio = spatial_crop_ratio
        self.evaluation_mode = evaluation_mode

        # Read in PDB keys
        self.pdb_keys_file = f"{self.pdb_path}/{phase}_pdb_keys.list"

        with open(self.pdb_keys_file) as f:
            self.pdb_keys = np.array(f.read().split("\n")[:-1])

        if self.cluster_sample:
            if not pdb_path.endswith("af3_pdb"):
                print('Cluster sampling disabled for non AF3 dataset')
            else:
                self._cluster_sample_pdb_keys()

        # Cache coordinates for faster loading
        self._cache_examples()

        # For efficiency set fixed size to max length in the eval or test dataset
        if self.evaluation_mode:
            self.fixed_size = self._get_max_len()

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

        if subset_length_range is not None:
            self.subset_to_length_range(*self.subset_length_range)

    def __len__(self):
        return len(self.pdb_keys)

    def __getitem__(self, idx):
        pdb_key = self.pdb_keys[idx]
        data = self.get_item(pdb_key)
        return data

    def _multimer_contiguous_crop(self, chain_1_len: int, chain_2_len: int) -> TensorType["n", bool]:

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

    def get_item(self, pdb_key):
        data_file = self._get_data_file(pdb_key)
        data = torch.load(data_file, weights_only=True)

        # Remove any residues with PLDDT < cutoff
        data = self._remove_low_plddt_residues(data)

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

        # Add designability info
        cond_labels_in["designability"] = cl.PLACEHOLDER_TOKEN_ID
        if self.designability_csv and self.pdb_path.endswith("ingraham_cath_dataset"):
            cond_labels_in["designability"] = self.pdb_to_designability[pdb_key]

        # Add dataset source label
        cond_labels_in["dataset_source"] = cl.TOKEN_TO_ID["dataset_source"][self.dataset_source_label]
        cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["UNCROPPED"]

        #Disable cropping for specified datasets
        start_idx = None
        multimer_crop_mask = None
        if not self.evaluation_mode:
            multimer_crop_mask, start_idx, cond_labels_in = self._crop_examples(example, cond_labels_in, multimer_crop_mask, start_idx)

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

        # Add conditioning labels
        example_out["cond_labels_in"] = cond_labels_in

        return example_out

    def _crop_examples(self, example, cond_labels_in, multimer_crop_mask, start_idx):
        # Calculate random cropping start index
        orig_size = example["x"].shape[0]
        extra_len = orig_size - self.fixed_size
        if extra_len > 0:
            if len(example['chain_ids']) > 1:
                if torch.rand(1) > self.spatial_crop_ratio:
                    chain_1_len, chain_2_len = torch.sum(example['chain_index'] == 0), torch.sum(example['chain_index'] == 1)
                    multimer_crop_mask = self._multimer_contiguous_crop(chain_1_len, chain_2_len)
                else:
                    multimer_crop_mask = self._multimer_spatial_crop(example['x'], example['interface_residue_mask'])
            else:
                start_idx = np.random.choice(np.arange(extra_len + 1))
            cond_labels_in["crop_aug"] = cl.TOKEN_TO_ID["crop_aug"]["CROPPED"]

        return multimer_crop_mask, start_idx, cond_labels_in

    def _cluster_sample_pdb_keys(self):
        self.pdb_keys = [random.choice(list(group)) for _, group in groupby(sorted(self.pdb_keys), key=lambda x: x.rsplit('_', 1)[-1])]

    def _get_data_file(self, pdb_key: str) -> str:
        """
        For a given pdb_key, return the path to the cached data file.
        """
        data_file = f"{self.pdb_path}/cached_examples/{pdb_key}.pt"
        return data_file

    def _get_dataset_source_label(self) -> str:
        if self.pdb_path.endswith("ingraham_cath_dataset"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("afdb"):
            dataset_source_label = "SYNTHETIC"
        elif self.pdb_path.endswith("af3_pdb"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("qfit-test-set/rcsb-pdb"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("rcsb_test_cases"):
            dataset_source_label = "EXPERIMENTAL"
        elif self.pdb_path.endswith("casp13") or self.pdb_path.endswith("casp14") or self.pdb_path.endswith("casp15"):
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

        # Define the number of workers based on CPU count or set manually
        num_workers = 8
        print(f"Using {num_workers} Workers!")

        # Prepare arguments as tuples (pdb_key, cache_dir, overwrite_cache) for process_pdb_key
        task_args = [(pdb_key, cache_dir, self.overwrite_cache, self.pdb_path, self.phase) for pdb_key in self.pdb_keys]

        # Use a Pool for parallel processing
        with Pool(processes=num_workers) as pool:
            # Use tqdm to display progress
            for _ in tqdm(pool.imap_unordered(process_pdb_key, task_args), total=len(task_args), desc="Caching PDBs"):
                pass

        print("Caching completed.")

    def _get_max_len(self):
        """
        Reads in cached PDB files and returns max length of all examples.
        This is only done for eval and test datasets where we do no cropping.
        """
        max_len = 0
        for pdb_key in tqdm(self.pdb_keys, desc=f"Getting max length in evaluation dataset", leave=False):
            data_file = self._get_data_file(pdb_key)
            example = torch.load(data_file, weights_only=True)
            seq_len = example["seq_mask"].sum().item()
            max_len = seq_len if (seq_len > max_len) else max_len

        return int(max_len)

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


    def _remove_low_plddt_residues(self, data: Dict[str, TensorType["n ..."]]) -> Dict[str, TensorType["n ..."]]:
        """
        If data comes from the afdb dataset, remove residues with plddt lower than self.afdb_res_plddt_cutoff.
        """
        if not self.pdb_path.endswith("afdb"):
            return data

        plddt_mask = data["b_factors"][:, 1] >= self.afdb_res_plddt_cutoff  # filter by C-alpha pLDDT
        for k, v in data.items():
            data[k] = v[plddt_mask]
        return data


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
        # x_scn = x_scn - x[..., 1:2, :]
        # scn_missing_atom_mask = batch["missing_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are missing
        # x_scn = torch.where(scn_missing_atom_mask[..., None].bool(), 0, x_scn)  # fill missing atoms with zeroes
        # scn_ghost_atom_mask = batch["ghost_atom_mask"][..., rc.non_bb_idxs]  # 1 for atoms that are not in the residue type
        # x_scn = torch.where(scn_ghost_atom_mask[..., None].bool(), 0, x_scn)  # fill ghost atoms with zeroes

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

def get_pdb_data_file(pdb_path, phase, pdb_key: str) -> str:
    if pdb_path.endswith("ingraham_cath_dataset"):  # ingraham splits
        pdb_data_file = f"{pdb_path}/pdb_store/{pdb_key}"
    elif pdb_path.endswith("af3_pdb"):
        pdb_data_file = f"{pdb_path}/{phase}_mmcifs/{pdb_key[1:3]}/{pdb_key[:4]}-assembly1.cif" #just use first assembly for now
    elif pdb_path.endswith("afdb"):  # AFDB augmentation dataset
        pdb_data_file = f"{pdb_path}/foldseek_cluster_reps/{pdb_key}.cif"
    elif pdb_path.endswith("qfit-test-set/rcsb-pdb"):
        pdb_data_file = f"{pdb_path}/all/{pdb_key}.pdb1"  # qfit dataset, use only pdb1s for now
    elif pdb_path.endswith("rcsb_test_cases"):
        pdb_data_file = f"{pdb_path}/pdbs/{pdb_key}.pdb"
    elif pdb_path.endswith("casp13") or pdb_path.endswith("casp14") or pdb_path.endswith("casp15"):
        pdb_data_file = f"{pdb_path}/pdbs/{pdb_key}.pdb"
    else:
        assert False, f"Unknown dataset: {pdb_path}"
    return pdb_data_file

# Modify process_pdb_key to be standalone if necessary (multiprocessing does not work well with methods)
def process_pdb_key(args):
    pdb_key, cache_dir, overwrite_cache, pdb_path, phase = args
    out_file = f"{cache_dir}/{pdb_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False

    pdb_data_file = get_pdb_data_file(pdb_path, phase, pdb_key)  # Ensure this function can work independently

    #specific to multimeric af3 dataset
    chain_ids_override = None
    if pdb_path.endswith("af3_pdb"):
        chain_ids_override = pdb_key.split('_')[1]

    example = load_feats_from_pdb(pdb_data_file, chain_ids_override=chain_ids_override, max_conformers=1)
    torch.save(example, out_file)

