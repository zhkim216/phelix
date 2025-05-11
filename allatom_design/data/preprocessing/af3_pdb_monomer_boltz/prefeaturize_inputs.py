#!/usr/bin/env python3
import glob
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.feature.seq_des_featurizer import \
    SequenceDesignFeaturizer
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.tokenize.tokenizer import Tokenizer
from functools import partial

@hydra.main(config_path="../../../configs/data/preprocessing/af3_pdb_monomer_boltz", config_name="prefeaturize_inputs", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given processed structures, tokenize them and save to disk for faster loading.
    """
    # Create output directory
    base_out_dir = f"{cfg.pdb_path}/processed_targets"
    out_tokenized_dir = f"{base_out_dir}/tokenized"
    out_featurized_dir = f"{base_out_dir}/featurized"
    Path(out_tokenized_dir).mkdir(parents=True, exist_ok=True)
    Path(out_featurized_dir).mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer and featurizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    featurizer = hydra.utils.instantiate(cfg.featurizer)

    # Get all processed structure files
    processed_structure_files = glob.glob(f"{cfg.pdb_path}/processed_targets/structures/*.npz")

    # Tokenize each structure
    use_parallel = cfg.num_workers > 1
    featurize_fn = partial(featurize_structure_to_disk,
                           tokenizer=tokenizer,
                           featurizer=featurizer,
                           atoms_per_window_queries=cfg.atoms_per_window_queries,
                           num_bins=cfg.num_bins,
                           max_residues_to_process=cfg.max_residues_to_process,
                           out_tokenized_dir=out_tokenized_dir,
                           out_featurized_dir=out_featurized_dir)
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(featurize_fn)(processed_structure_file) for processed_structure_file in processed_structure_files]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Featurizing structures")))
    else:
        for processed_structure_file in tqdm(processed_structure_files, desc="Featurizing structures"):
            featurize_fn(processed_structure_file)


def featurize_structure_to_disk(processed_structure_file: str,
                                tokenizer: Tokenizer,
                                featurizer: SequenceDesignFeaturizer,
                                atoms_per_window_queries: int,
                                num_bins: int,
                                max_residues_to_process: int | None,
                                out_tokenized_dir: str,
                                out_featurized_dir: str,
                                ) -> None:
    """
    Load a processed structure and featurize it.

    - If max_residues_to_process is not None, we skip any structures that have more residues than max_residues_to_process.
    """
    out_tokenized_file = f"{out_tokenized_dir}/{Path(processed_structure_file).stem}.npz"
    out_featurized_file = f"{out_featurized_dir}/{Path(processed_structure_file).stem}.npz"

    if Path(out_tokenized_file).exists() and Path(out_featurized_file).exists():
        # Skip if already tokenized and featurized
        return

    # Get structure
    input_data = load_input(processed_structure_file)

    if max_residues_to_process is not None and len(input_data.structure.residues) > max_residues_to_process:
        print(f"Skipping structure {processed_structure_file} because it has {len(input_data.structure.residues)} residues, which is greater than max_residues_to_process={max_residues_to_process}.")
        return

    # Tokenize structure
    try:
        tokenized = tokenizer.tokenize(input_data)
    except Exception as e:
        print(f"Error tokenizing structure {processed_structure_file}: {e}. Skipping.")
        return

    if len(tokenized.tokens) == 0:
        print(f"Tokenized structure {processed_structure_file} has no tokens. Skipping.")
        return

    # Featurize structure (without padding to max_tokens or max_atoms)
    try:
        feats = featurizer.process(tokenized, use_auth_seq_id=True, atoms_per_window_queries=atoms_per_window_queries, num_bins=num_bins)
    except Exception as e:
        print(f"Error featurizing structure {processed_structure_file}: {e}. Skipping.")
        return

    # Save tokenized
    np.savez_compressed(out_tokenized_file, **asdict(tokenized))

    # Save featurized
    np.savez_compressed(out_featurized_file, **feats)


if __name__ == "__main__":
    main()
