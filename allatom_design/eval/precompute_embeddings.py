from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.proteinmpnn_utils import load_mpnn, create_mpnn_embeddings


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

    mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
    mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
    mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    create_mpnn_embeddings(mpnn_model, pdb_paths=pdb_paths, out_dir=cfg.out_dir, device=device, cfg=mpnn_cfg)


if __name__ == "__main__":
    main()
