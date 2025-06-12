#!/usr/bin/env python3
import glob
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.datasets.boltz_ad_dataset import add_tokenwise_atom_feats
from allatom_design.data.feature.ad_featurizer import ADFeaturizer
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.tokenize.tokenizer import Tokenizer
from allatom_design.data.types import (Connection, Input, Structure, Tokenized,
                                       TokenwiseAtomFeats)


@hydra.main(config_path="../../../configs/data/preprocessing/augmented_af3_monomer_v2_boltz", config_name="pretokenize_inputs", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given processed structures, tokenize them and save to disk for faster loading.
    """
    # Create output directory
    out_dir = f"{cfg.pdb_path}/processed_targets/ad_tokenized"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer and featurizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    featurizer = hydra.utils.instantiate(cfg.featurizer)

    # Get all processed structure files
    processed_structure_files = glob.glob(f"{cfg.pdb_path}/processed_targets/structures/*.npz")

    # Tokenize each structure
    use_parallel = cfg.num_workers > 1
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(tokenize_structure_to_disk)(processed_structure_file, out_dir, tokenizer, featurizer) for processed_structure_file in processed_structure_files]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Tokenizing structures")))
    else:
        for processed_structure_file in tqdm(processed_structure_files, desc="Tokenizing structures"):
            tokenize_structure_to_disk(processed_structure_file, out_dir, tokenizer, featurizer)


def tokenize_structure_to_disk(processed_structure_file: str, out_dir: str, tokenizer: Tokenizer, featurizer: ADFeaturizer) -> None:
    """
    Load a processed structure and tokenize it.
    """
    out_file = f"{out_dir}/{Path(processed_structure_file).stem}.npz"
    if Path(out_file).exists():
        # Skip if already tokenized
        return

    # Get structure
    input_data = load_input(processed_structure_file)

    # Tokenize structure
    try:
        tokenized = tokenizer.tokenize(input_data)
    except Exception as e:
        print(f"Error tokenizing structure {processed_structure_file}: {e}. Skipping.")
        return

    if len(tokenized.tokens) == 0:
        print(f"Tokenized structure {processed_structure_file} has no tokens. Skipping.")
        return

    try:
        tokenized = add_tokenwise_atom_feats(tokenized, featurizer)
    except Exception as e:
        print(f"Error adding tokenwise atom feats to structure {processed_structure_file}: {e}. Skipping.")
        return

    # Save tokenized
    np.savez_compressed(out_file, **asdict(tokenized))


if __name__ == "__main__":
    main()
