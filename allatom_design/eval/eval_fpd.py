from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_metrics import fpd
from allatom_design.eval.proteinmpnn_utils import load_mpnn, create_mpnn_embeddings, load_mpnn_embeddings


def subsample_embeddings(embeddings_dir: str, lengths_csv_path: str, frac: float):
    lengths = pd.read_csv(lengths_csv_path)
    lengths_subsample = lengths.groupby("length", group_keys=False).sample(frac=frac)
    embeddings_dir = Path(embeddings_dir)
    pdb_paths = [str(embeddings_dir / f"{pdb_key}.npy") for pdb_key in lengths_subsample["pdb_key"]]
    return pdb_paths


@hydra.main(config_path="../configs/eval", config_name="eval_fpd", version_base="1.3.2")
def main(cfg: DictConfig):
    # TODO: Write sampling code here, might need to subsample embeddings earlier? Not sure though.

    pdbs = ["path/to/pdb1", "path/to/pdb2", "path/to/pdb3"] # List of paths to sampled pdb files (With same length dist)
    out_dir = "" # A master directory under which the folder with the sampled structures is saved

    # Create output directories
    samp_embeddings_dir = Path(out_dir, "mpnn_embeddings")
    samp_embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load MPNN model
    device = torch.device("cuda")
    mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
    mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
    mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    # Run MPNN
    create_mpnn_embeddings(mpnn_model, pdb_paths=pdbs, out_dir=str(samp_embeddings_dir), device=device, cfg=mpnn_cfg)

    # Subsample embeddings
    train_embedding_paths = subsample_embeddings(cfg.embeddings_dir, cfg.train_lengths_csv, cfg.pct_subsample)
    eval_embedding_paths = subsample_embeddings(cfg.embeddings_dir, cfg.eval_lengths_csv, cfg.pct_subsample)
    eval2_embedding_paths = subsample_embeddings(cfg.embeddings_dir, cfg.eval2_lengths_csv, cfg.pct_subsample)

    # Load embeddings
    samp_embedding_paths = [f"{str(samp_embeddings_dir)}/{pdb}.npy" for pdb in pdbs]
    samp_embeddings = load_mpnn_embeddings(samp_embedding_paths)

    train_embeddings = load_mpnn_embeddings(train_embedding_paths)
    eval_embeddings = load_mpnn_embeddings(eval_embedding_paths)
    eval2_embeddings = load_mpnn_embeddings(eval2_embedding_paths)

    # Calculate FPD scores
    train_fpd_scores = []
    eval_fpd_scores = []
    eval2_fpd_scores = []

    for i in range(3): # 3 layers
        train_fpd_score = fpd(samp_embeddings[i], train_embeddings[i])
        train_fpd_scores.append(train_fpd_score)

        eval_fpd_score = fpd(samp_embeddings[i], eval_embeddings[i])
        eval_fpd_scores.append(eval_fpd_score)

        eval2_fpd_score = fpd(samp_embeddings[i], eval2_embeddings[i])
        eval2_fpd_scores.append(eval2_fpd_score)

   # TODO: Add Wandb logging here -- y-axis can be the FPD score, x-axis can be the layer number


if __name__ == "__main__":
    main()
