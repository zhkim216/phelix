#!/usr/bin/env python3
import glob
import itertools
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import yaml
from atomworks.ml.common import generate_example_id
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import \
    DataPreprocessor
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from atomworks.ml.transforms.encoding import EncodeAtomArray
from p_tqdm import p_umap


@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="build_metadata_parquet", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve the original config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem

    # Get CIF paths
    cif_paths = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)

    # Initialize data preprocessor
    processor = DataPreprocessor(**{
        **cfg.cif_parser_args,
        **cfg.data_preprocessor_cfg
    })

    def _process_cif(cif_path: str):
        """
        Wrapper around the data preprocessor to handle errors.
        """
        try:
            return processor.get_rows(cif_path)
        except Exception as e:
            print(f"Error processing {cif_path}: {e}")
            return []

    ### Process each CIF and save to parquet ###
    if use_parallel:
        df = p_umap(_process_cif, cif_paths, num_cpus=cfg.num_workers, desc="Processing mmCIFs")
    else:
        df = [_process_cif(cif_path) for cif_path in tqdm(cif_paths)]
    df = pd.DataFrame(itertools.chain(*df))  # flatten list of lists
    save_to_parquet(df, dataset_name, cfg.mmcif_dir, f"{cfg.out_dir}/metadata.parquet")  # save to parquet

    # for caching, we need to save a parquet with unique pdb IDs to avoid race conditions
    df_cache = df.groupby("pdb_id").first().reset_index()
    save_to_parquet(df_cache, dataset_name, cfg.mmcif_dir, f"{cfg.out_dir}/metadata_for_caching.parquet")  # save to parquet


def get_cif_paths(mmcif_dir: str, max_file_size: int | None = None) -> list[str]:
    cif_paths = glob.glob(f"{mmcif_dir}/**/*.cif", recursive=True)
    if max_file_size is not None:
        original_len = len(cif_paths)
        cif_paths = [path for path in cif_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(cif_paths)} files due to size.")
    print(f"Found {len(cif_paths)} mmCIFs.")
    return cif_paths


def save_to_parquet(df: pd.DataFrame,
                    dataset_name: str,
                    pdb_in_dir: str,
                    out_path: str):
    """
    Save a dataframe to parquet.
    Also adds an example_id column based on the name of the dataset.

    - df: dataframe to save
    - dataset_name: name of the dataset
    - pdb_in_dir: base path of the input directory
    - out_path: path to save the parquet file
    """
    # Convert all object columns to string to save to parquet
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Add example_id based on the name of this dataset
    df["example_id"] = df.apply(lambda x: generate_example_id(
        [dataset_name],
        x["pdb_id"],
        x["assembly_id"],
        [x["q_pn_unit_iid"]]), axis=1)

    # Add in relative path
    df["rel_path"] = df["path"].apply(lambda x: str(Path(x).relative_to(pdb_in_dir)))

    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
