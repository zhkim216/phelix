#!/usr/bin/env python3
from pathlib import Path

import hydra
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    get_polymer_seqs, hash_sequence)
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files)


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_val_cifs", config_name="build_msa_fasta", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script to build fasta files to query the MSA database with colabfold.
    """
    # Create output directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Load in PDB files to make fasta file from
    pdb_files = get_pdb_files(**cfg.input_cfg)
    temp_processed_struct_dir = f"{cfg.out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=temp_processed_struct_dir, **cfg.pdb_processing_cfg)

    # Get all polymer sequences among processed targets
    proteins = set()
    shorts = set()
    nucleotides = set()
    nonpolymer_seqs = set()
    key_to_seq = {}

    use_parallel = cfg.num_workers > 1
    if use_parallel:
        # process in parallel
        results = Parallel(n_jobs=cfg.num_workers)(
            delayed(get_polymer_seqs)(structure_file)
            for structure_file in tqdm(processed_struct_files, desc="Parsing polymer sequences")
        )
        # merge all results
        for p, s, n, npoly, k2s in results:
            proteins.update(p)
            shorts.update(s)
            nucleotides.update(n)
            nonpolymer_seqs.update(npoly)
            key_to_seq.update(k2s)
    else:
        # process sequentially
        for structure_file in tqdm(processed_struct_files, desc="Parsing polymer sequences"):
            p, s, n, npoly, k2s = get_polymer_seqs(structure_file)
            proteins.update(p)
            shorts.update(s)
            nucleotides.update(n)
            nonpolymer_seqs.update(npoly)
            key_to_seq.update(k2s)

    proteins = [f">{hash_sequence(seq)}\n{seq}" for seq in proteins]
    with (Path(cfg.out_dir) / "proteins.fasta").open("w") as f:
        f.write("\n".join(proteins))


if __name__ == "__main__":
    main()
