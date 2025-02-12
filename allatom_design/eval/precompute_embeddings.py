import os
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.esm3_utils import create_esm3_embeddings
from allatom_design.eval.proteinmpnn_utils import (create_mpnn_embeddings,
                                                   load_mpnn)
from esm3.esm.models.esm3 import ESM3

warnings.filterwarnings("ignore")

@hydra.main(config_path="../configs/eval", config_name="precompute_embeddings", version_base="1.3.2")
def main(cfg: DictConfig):
    device = torch.device("cuda")

    # Get pdb keys for each phase
    phase_to_pdb_key_files = {"train": cfg.train_pdb_key_file, "eval": cfg.eval_pdb_key_file, "eval2": cfg.eval2_pdb_key_file}
    pdbs_df = defaultdict(list)
    for phase, file_path in phase_to_pdb_key_files.items():
        if file_path is None:
            # skip if key file is not provided
            continue

        with open(file_path) as f:
            phase_pdbs = np.array(f.read().split("\n")[:-1])

        if cfg.subsample_n is not None:
            # subsample to at max subsample_n pdbs from each phase
            phase_pdbs = np.random.choice(phase_pdbs, min(cfg.subsample_n, len(phase_pdbs)), replace=False)

        pdb_keys = [Path(pdb).stem for pdb in phase_pdbs]  # remove .pdb extension in case it exists in key list
        pdbs_df["pdb_key"].extend(pdb_keys)
        pdbs_df["phase"].extend([phase] * len(pdb_keys))
    pdbs_df = pd.DataFrame(pdbs_df)

    # Dump pdbs with pre-computed embeddings to keys lists
    for phase in pdbs_df["phase"].unique():
        phase_pdbs = pdbs_df[pdbs_df["phase"] == phase]
        with open(Path(cfg.out_dir, f"precomputed_{phase}_pdb_keys.list"), "w") as f:
            f.write("\n".join(phase_pdbs["pdb_key"]))

    # Compute embeddings for FPD
    pdbs_dir = Path(cfg.pdbs_dir)
    pdb_paths = [str(pdbs_dir / f"{pdb}.pdb") for pdb in pdbs_df["pdb_key"]]

    if cfg.compute_mpnn:
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

        create_mpnn_embeddings(mpnn_model, pdb_paths=pdb_paths, out_dir=cfg.mpnn.out_dir, device=device, cfg=mpnn_cfg)

    if cfg.compute_esm3:
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        device = "cuda"
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        vqvae_encoder = model.get_structure_encoder()

        create_esm3_embeddings(vqvae_encoder, pdb_paths=pdb_paths, out_dir=cfg.esm3.out_dir, device=device)


if __name__ == "__main__":
    main()
