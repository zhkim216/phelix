import os
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
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
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get pdb keys for each phase
    if not cfg.use_precomputed_key_files:
        # Subset from pdb key files
        phase_to_pdb_key_files = {"train": cfg.train_pdb_key_file, "eval": cfg.eval_pdb_key_file, "eval2": cfg.eval2_pdb_key_file}
        pdbs_df = defaultdict(list)
        for phase, file_path in phase_to_pdb_key_files.items():
            if file_path is None:
                # skip if key file is not provided
                continue

            with open(file_path) as f:
                phase_pdbs = np.array(f.read().splitlines())

            if cfg.subsample_n is not None:
                # subsample to at max subsample_n pdbs from each phase
                phase_pdbs = np.random.choice(phase_pdbs, min(cfg.subsample_n, len(phase_pdbs)), replace=False)

            pdb_keys = [Path(pdb).stem for pdb in phase_pdbs]  # remove .pdb extension in case it exists in key list
            pdbs_df["pdb_key"].extend(pdb_keys)
            pdbs_df["phase"].extend([phase] * len(pdb_keys))
        pdbs_df = pd.DataFrame(pdbs_df)

        # Dump pdbs with pre-computed embeddings to keys lists
        for phase in pdbs_df["phase"].unique():
            phase_pdbs_df = pdbs_df[pdbs_df["phase"] == phase]
            phase_pdbs_df["pdb_key"].to_csv(Path(cfg.out_dir, f"precomputed_{phase}_pdb_keys.csv"), index=False, header=False)

        all_pdb_keys = pdbs_df["pdb_key"].values
    else:
        # Load precomputed pdb keys
        all_pdb_keys = []
        phases = ["train", "eval", "eval2"]
        precomputed_key_files = [Path(cfg.out_dir, f"precomputed_{phase}_pdb_keys.csv") for phase in phases]
        for phase, file_path in zip(phases, precomputed_key_files):
            if not file_path.exists():
                print(f"Skipping {phase} phase: precomputed key file not found")
                continue

            phase_pdbs = pd.read_csv(file_path, header=None).values.flatten()
            all_pdb_keys.extend(phase_pdbs)

    # Compute embeddings for FPD
    pdbs_dir = Path(cfg.pdbs_dir)
    pdb_paths = [str(pdbs_dir / f"{pdb}.pdb") for pdb in all_pdb_keys]

    if cfg.compute_mpnn:
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

        create_mpnn_embeddings(mpnn_model, pdb_paths=pdb_paths, out_dir=f"{cfg.out_dir}/mpnn", device=device, cfg=mpnn_cfg)

    if cfg.compute_esm3:
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        device = "cuda"
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        vqvae_encoder = model.get_structure_encoder()

        create_esm3_embeddings(vqvae_encoder, pdb_paths=pdb_paths, out_dir=f"{cfg.out_dir}/esm3", device=device)


if __name__ == "__main__":
    main()
