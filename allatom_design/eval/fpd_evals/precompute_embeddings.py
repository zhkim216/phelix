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

from allatom_design.data.datasets.ad_dataset import get_pdb_data_file
from allatom_design.eval.eval_utils.esm3_utils import create_esm3_embeddings
from allatom_design.eval.eval_utils.fampnn_utils import create_fampnn_embeddings
from allatom_design.eval.eval_utils.proteinmpnn_utils import (create_mpnn_embeddings,
                                                   load_mpnn)
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
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

    # Disable gradients globally
    torch.set_grad_enabled(False)

    # Use cluster sampling for AF3 datasets
    use_cluster_sampling = False
    if cfg.pdb_path.endswith("af3_pdb") or cfg.pdb_path.endswith("af3_pdb_monomer"):
        use_cluster_sampling = True

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

            if use_cluster_sampling:
                pdb_keys_df = pd.DataFrame({"pdb_key": phase_pdbs, "cluster_id": [pdb.split("_")[-1] for pdb in phase_pdbs]})
                if phase == "train":
                    # For train, randomly sample one PDB from each cluster
                    pdb_keys_df = pdb_keys_df.groupby("cluster_id", group_keys=False).apply(lambda g: g.sample(n=1, random_state=cfg.seed)).reset_index(drop=True)
                elif phase in ["eval", "eval2"]:
                    # For eval, only take the first PDB in each cluster for deterministic evaluation
                    pdb_keys_df = pdb_keys_df.groupby("cluster_id", as_index=False).first().reset_index(drop=True)
                phase_pdbs = pdb_keys_df["pdb_key"].values

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
    pdb_paths = [get_pdb_data_file(cfg.pdb_path, phase, pdb_key) for phase, pdb_key in zip(pdbs_df["phase"].values, all_pdb_keys)]

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

    if cfg.compute_fampnn:
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.fampnn.checkpoint_path).eval()
        create_fampnn_embeddings(lit_sd_model.model, pdb_paths=pdb_paths,
                                 backbone_only=True,
                                 batch_size=cfg.fampnn.batch_size,
                                 device=lit_sd_model.device,
                                 out_dir=f"{cfg.out_dir}/fampnn")



if __name__ == "__main__":
    main()
