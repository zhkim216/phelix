import os
import warnings
from pathlib import Path

import hydra
import torch
from esm3.esm.models.esm3 import ESM3
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.esm3_utils import create_esm3_embeddings
from allatom_design.eval.proteinmpnn_utils import load_mpnn, create_mpnn_embeddings

warnings.filterwarnings("ignore")

@hydra.main(config_path="../configs/eval", config_name="precompute_embeddings", version_base="1.3.2")
def main(cfg: DictConfig):
    device = torch.device("cuda")

    pdb_key_files = [cfg.train_pdb_key_file, cfg.eval_pdb_key_file, cfg.eval2_pdb_key_file]

    pdbs = []
    for file_path in pdb_key_files:
        with open(file_path) as f:
            pdbs.extend(f.read().split("\n")[:-1])

    pdbs_dir = Path(cfg.pdbs_dir)
    pdb_paths = [str(pdbs_dir / f"{pdb}.pdb") for pdb in pdbs]

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
