import multiprocessing
import random
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (center_random_augmentation,
                                      load_feats_from_pdb, make_fixed_size_1d,
                                      transform_sidechain_frame, get_scaffolding_inputs, FEATURES_LONG)
from allatom_design.data.pdb_utils import write_to_pdb
from allatom_design.data.datasets.scaffold_manager import get_scaffold_manager


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
        short_epoch: bool = False,
        n_random_subset: Optional[int] = None,
        se3_augment: bool = True,
        translation_scale: float = 1.0,
        overwrite_cache: bool = False,
        subset_length_range: Optional[int] = None,
        max_scrmsd: Optional[float] = None,
        max_rel_rog: Optional[float] = None,
        afdb_res_plddt_cutoff: float = 0.0,
        spatial_crop_ratio: float = 0.5,
        evaluation_mode: bool = False,
        scaffold_manager_cfg: Optional[DictConfig] = None,
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
        - short_epoch: If True, the dataset will only return 500 random examples.
        - n_random_subset: If not None, the dataset will only return a random subset of n examples.
        - se3_augment: If True, apply SE3 augmentation to the data.
        - translation_scale: Scale of translation augmentation (when using raw coords or coords feats)
        - overwrite_cache: If True, overwrite the dataset cache. Useful if the dataset features have been updated.
        - subset_length_range: List with with [min, max] length of proteins to subset form training data
        - scrmsd: for training on only designable structures; subset to only pdbs with scRMSD <= max_scrmsd
        - afdb_res_plddt_cutoff: If > 0, for AFDB dataset, cut out any residues with PLDDT < cutoff
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
        self.afdb_res_plddt_cutoff = afdb_res_plddt_cutoff
        self.spatial_crop_ratio = spatial_crop_ratio
        self.evaluation_mode = evaluation_mode

        self.sm = get_scaffold_manager(scaffold_manager_cfg)  # for constructing scaffolding inputs

        # Read in PDB keys
        eval2_suffix = "_with_eval2" if run_eval2 else ""  # if using eval2, we load in a slightly smaller set of training pdb keys
        self.pdb_keys_csv = f"{self.pdb_path}/{phase}_pdb_keys{eval2_suffix}.csv"

        # Load annotation info
        self._load_annotation_info()

        if not Path(self.pdb_keys_csv).exists():
            # backwards compatibility; load in PDB keys from list and save to CSV format
            self.pdb_keys_file = f"{self.pdb_path}/{phase}_pdb_keys{eval2_suffix}.list"
            with open(self.pdb_keys_file) as f:
                self.pdb_keys = np.array(f.read().splitlines())

            # cache coordinates for faster loading
            self._cache_examples(self.pdb_keys)

            # get lengths; store them and save to csv
            pdb_key_to_length = {}
            for pdb_key in tqdm(self.pdb_keys, desc="Getting lengths", leave=False):
                data_file = self._get_data_file(pdb_key)
                example = torch.load(data_file, weights_only=True)
                seq_len = example["seq_mask"].sum().long().item()
                pdb_key_to_length[pdb_key] = seq_len

            # save to csv
            self.pdb_keys_df = pd.DataFrame({"pdb_key": self.pdb_keys, "seq_length": [pdb_key_to_length[pdb_key] for pdb_key in self.pdb_keys]})
            self.pdb_keys_df.to_csv(self.pdb_keys_csv, index=False)

        else:
            # load from csv
            self.pdb_keys_df = pd.read_csv(self.pdb_keys_csv)

            # cache coordinates for faster loading
            self._cache_examples(self.pdb_keys_df["pdb_key"])

        # Subset to length range
        if subset_length_range is not None:
            self.subset_to_length_range(*self.subset_length_range)

        # Subset based on scRMSD
        if max_scrmsd is not None:
            self.subset_by_scrmsd(max_scrmsd)

        if max_rel_rog is not None:
            self.subset_by_rel_rog(max_rel_rog)

        # Cluster sampling for AF3 dataset
        if pdb_path.endswith("af3_pdb") or pdb_path.endswith("af3_pdb_monomer"):
            assert self.cluster_sample, "Cluster sampling must be enabled for AF3 dataset"
            print("Re-sampling dataset based on clusters")
            self._cluster_sample_pdb_keys()
        else:
            assert not self.cluster_sample, "Cluster sampling must be disabled for non-AF3 dataset"

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

        if short_epoch:
            self.pdb_keys_df = self.pdb_keys_df.sample(min(500, len(self.pdb_keys_df)), replace=False)

        if n_random_subset is not None:
            self.pdb_keys_df = self.pdb_keys_df.sample(min(n_random_subset, len(self.pdb_keys_df)), replace=False)


    def __len__(self):
        return len(self.pdb_keys_df)

    def __getitem__(self, idx):
        pdb_key = self.pdb_keys_df["pdb_key"].iloc[idx]
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
            x = center_random_augmentation(x, seq_mask, atom_mask, translation_scale=self.translation_scale)

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
            # example['interface_residue_mask'] = data['interface_residue_mask']
            # example['chain_ids'] = data['chain_ids']
            # DEBUG: remove this once we re-cache the dataset
            example["interface_residue_mask"] = data.get("interface_residue_mask", torch.zeros_like(seq_mask))
            example["chain_ids"] = data.get("chain_ids", torch.tensor([1]))

        # Get scaffolding input with scaffold manager
        example["x_scaffold"], example["scaffold_mask"], example["aatype_scaffold"], example["x"] = get_scaffolding_inputs(self.sm, example)

        # Construct conditioning inputs
        cond_labels_in = {}

        # Condition on cropping
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
            if example['chain_ids'] is not None and len(example['chain_ids']) > 1:
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
        pdb_keys = self.pdb_keys_df["pdb_key"].tolist()
        pdb_keys = [random.choice(list(group)) for _, group in groupby(sorted(pdb_keys), key=lambda x: x.rsplit("_", 1)[-1])]
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["pdb_key"].isin(pdb_keys)]


    def _get_data_file(self, pdb_key: str) -> str:
        """
        For a given pdb_key, return the path to the cached data file.
        """
        data_file = f"{self.pdb_path}/cached_examples/{pdb_key}.pt"
        return data_file


    def _cache_examples(self, pdb_keys: List[str]):
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
        task_args = [(pdb_key, cache_dir, self.overwrite_cache, self.pdb_path, self.phase) for pdb_key in pdb_keys]

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
        keep_pdb_keys = set((self.annotation_csv["sc_ca_rmsd"] <= max_scrmsd).index)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["pdb_key"].isin(keep_pdb_keys)]

    def subset_by_rel_rog(self, max_rel_rog: float):
        """
        Subsets the dataset to only include proteins with relative radius of gyration <= max_rel_rog.
        """
        keep_pdb_keys = set((self.annotation_csv["rel_rog"] <= max_rel_rog).index)
        self.pdb_keys_df = self.pdb_keys_df[self.pdb_keys_df["pdb_key"].isin(keep_pdb_keys)]


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

def get_pdb_data_file(pdb_path: str, phase: str, pdb_key: str) -> str:
    if pdb_path.endswith("ingraham_cath_dataset"):  # ingraham splits
        pdb_data_file = f"{pdb_path}/pdb_store/{pdb_key}"
    elif pdb_path.endswith("augmented_ingraham_cath_bugfree"):  # tianyu's augmented dataset
        pdb_data_file = f"{pdb_path}/mpnn_esmfold/{pdb_key}"
        if not Path(pdb_data_file).exists():
            pdb_data_file = f"{pdb_path}/dne_mpnn/{pdb_key}"
    elif pdb_path.endswith("af3_pdb"):
        pdb_data_file = f"{pdb_path}/{phase}_mmcifs/{pdb_key[1:3]}/{pdb_key[:4]}-assembly1.cif" #just use first assembly for now
    elif pdb_path.endswith("af3_pdb_monomer"):
        mmcif_phase = phase
        if phase == "eval2":
            # grab eval2 from train as well
            mmcif_phase = "train"
        pdb_data_file = f"{pdb_path}/{mmcif_phase}_mmcifs/{pdb_key}.cif"
    elif pdb_path.endswith("afdb"):  # AFDB augmentation dataset
        pdb_data_file = f"{pdb_path}/foldseek_cluster_reps/{pdb_key}.cif"
    elif pdb_path.endswith("qfit-test-set/rcsb-pdb"):
        pdb_data_file = f"{pdb_path}/all/{pdb_key}.pdb1"  # qfit dataset, use only pdb1s for now
    elif pdb_path.endswith("rcsb_test_cases"):
        pdb_data_file = f"{pdb_path}/pdbs/{pdb_key}.pdb"
    elif Path(pdb_path).stem.startswith("casp"):
        pdb_data_file = f"{pdb_path}/pdbs/{pdb_key}.pdb"
    elif pdb_path.endswith("denovo100") or pdb_path.endswith("denovo200") or pdb_path.endswith("denovo300") or pdb_path.endswith("denovo400") or pdb_path.endswith("denovo500"):
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

