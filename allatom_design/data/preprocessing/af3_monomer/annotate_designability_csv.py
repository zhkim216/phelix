#!/usr/bin/env python3
"""
Add useful information to the designability statistics csv, such as radius of gyration
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB.MMCIFParser import FastMMCIFParser
from joblib import Parallel, delayed
from tqdm import tqdm

mmcif_parser = FastMMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)

def main():
    parser = argparse.ArgumentParser(description="Parse designability statistics csv")
    parser.add_argument("--designability_stats_csv", type=str, help="Path to designability statistics csv.")
    parser.add_argument("--af3_dir", type=str, help="Path to AF3 dataset directory, containing eval_mmcifs and train_mmcifs.")
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    args = parser.parse_args()

    df = pd.read_csv(args.designability_stats_csv)

    # Parallelize radius of gyration calculation with a progress bar
    radius_of_gyration_list = Parallel(n_jobs=-1)(
        delayed(compute_radius_of_gyration)(row, args.af3_dir)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    df["radius_of_gyration"] = radius_of_gyration_list

    df["ideal_rad"] = 2.24 * (df["seq_length"] ** 0.392)  # Dill et al. https://www.pnas.org/doi/full/10.1073/pnas.1114477108
    df["rel_rog"] = df["radius_of_gyration"] / df["ideal_rad"]  # Verkuil et al.  https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out_dir}/annotated_pdb_keys.csv", index=False, header=True)


def compute_radius_of_gyration(row, dataset_dir, min_dist=1e-3) -> float:
    """
    Computes the radius of gyration for a chain. Adapted from RFDiffusion code: https://github.com/RosettaCommons/RFdiffusion/blob/820bfdfaded8c260b962dc40a3171eae316b6ce0/rfdiffusion/potentials/potentials.py#L24

    Args:
    - row (pd.Series): Row of a dataframe containing a pdb_key and seq_length
    - min_dist (float): Minimum distance.

    Returns:
    - float: Radius of gyration.
    """
    # Load in chain from mmcif
    mmcif_dir = f"{dataset_dir}/{row['phase']}_mmcifs"
    mmcif_path = f"{mmcif_dir}/{row['pdb_key']}.cif"
    model_num = 0
    structure = mmcif_parser.get_structure("s", mmcif_path)[model_num]

    chains = list(structure.get_chains())
    if len(chains) > 1:
        raise ValueError(f"More than one chain found in {mmcif_path}")
    chain = chains[0]

    # Extract CA atom coordinates
    Ca_coords = np.array([atom.get_coord() for residue in chain for atom in residue if atom.get_name() == "CA"])
    Ca = torch.tensor(Ca_coords, dtype=torch.float32)

    centroid = torch.mean(Ca, dim=0, keepdim=True)

    dists = torch.norm((centroid - Ca), dim=-1)
    dists = dists.clamp(min=min_dist)

    rad_of_gyration = torch.sqrt(torch.sum(torch.square(dists)) / Ca.shape[0])
    return rad_of_gyration.item()


if __name__ == "__main__":
    main()
