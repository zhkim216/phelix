from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colorbar import ColorbarBase
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files)
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_example


@hydra.main(version_base=None, config_path="../../configs/eval/plots", config_name="plot_potts_params")
def main(cfg: DictConfig) -> None:
    """
    Plot potts parameters and contact maps
    """
    # Create the base output directory
    Path(cfg.base_out_dir).mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.base_out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB files
    pdb_files = get_pdb_files(**cfg.input_cfg)
    temp_processed_struct_dir = f"{cfg.base_out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=temp_processed_struct_dir, **cfg.pdb_processing_cfg)

    # Prepare directory for contact map plots
    plot_out_dir = f"{cfg.base_out_dir}/plots"
    Path(plot_out_dir).mkdir(parents=True, exist_ok=True)

    # Process each PDB file to extract contact maps
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)
    record_id_to_contact_map = {}
    record_id_to_residue_index = {}
    for struct_file in tqdm(processed_struct_files, desc="Processing PDBs"):
        example, _ = get_sd_example(struct_file, data_cfg)
        X = example["disto_center"]  # CB atoms, CA for glycine
        disto_exists_mask = (example["token_disto_mask"] * example["token_pad_mask"])  # remove pad and unresolved atoms

        # mask out pad and unresolved atoms
        X = X[disto_exists_mask.bool()]
        residue_index = example["residue_index"][disto_exists_mask.bool()]

        # get contact map based on distance cutoff
        contact_map = torch.cdist(X, X, p=2) < cfg.contact_map_cutoff
        record_id_to_contact_map[example["pdb_key"]] = contact_map
        record_id_to_residue_index[example["pdb_key"]] = residue_index


    # Plot and save the contact maps along with potts parameters
    for record_id in tqdm(record_id_to_contact_map.keys(), desc="Plotting results"):
        contact_map = record_id_to_contact_map[record_id]
        residue_index = record_id_to_residue_index[record_id]

        # Plot the ground truth contact map once per record
        plot_contact_map(
            contact_map=contact_map,
            residue_index=residue_index,
            pdb_key=record_id,
            output_dir=plot_out_dir,
        )

        # load in potts parameters
        for model_cfg in cfg.model_csvs:
            model_name = model_cfg.model_name
            model_base_dir = model_cfg.base_dir
            model_plot_name = model_cfg.plot_name

            # load in potts parameters
            potts_params_path = f"{model_base_dir}/{record_id}.pt"
            potts_params = torch.load(potts_params_path)
            J = potts_params["J"]
            mask_ij = potts_params["mask_ij"]
            # restrict to protein-only interactions
            J = J[:, :, 2:22, 2:22]
            J = J * mask_ij[..., None, None]

            # calculate the frobenius norm for each 20x20 residue pair matrix
            fro_norm_J = torch.linalg.norm(J, dim=(-2, -1))

            # scale the frobenius norm from 0 to 1
            min_val = fro_norm_J.min()
            max_val = fro_norm_J.max()
            if (max_val - min_val) > 1e-8: # avoid division by zero
                scaled_fro_norm = (fro_norm_J - min_val) / (max_val - min_val)
            else:
                scaled_fro_norm = torch.zeros_like(fro_norm_J)

            # plot the scaled frobenius norms
            plot_coupling_strengths(
                coupling_matrix=scaled_fro_norm,
                residue_index=residue_index,
                pdb_key=record_id,
                model_plot_name=model_plot_name,
                output_dir=plot_out_dir
            )


def plot_contact_map(
    contact_map: TensorType["n_res", "n_res"],
    residue_index: TensorType["n_res"],
    pdb_key: str,
    output_dir: str,
) -> None:
    """Plots and saves a single protein contact map."""
    fig, ax = plt.subplots()
    ax.imshow(contact_map.numpy(), cmap="gray_r", interpolation="none")
    ax.set_title(f"Contact Map for {pdb_key}")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")

    # Set the axis ticks to be the residue index values instead of array indices
    res_indices_np = residue_index.numpy()
    num_res = len(res_indices_np)
    # Select a subset of indices for ticks to avoid clutter on the axes
    # and handle cases with very few residues.
    num_ticks = min(num_res, 15)  # Show at most 15 ticks
    tick_locs = np.linspace(0, num_res - 1, num=num_ticks, dtype=int)
    tick_labels = res_indices_np[tick_locs]

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(f"{output_dir}/{pdb_key}_contact_map.png", dpi=300)
    plt.close(fig)


def plot_coupling_strengths(
    coupling_matrix: TensorType["n_res", "n_res"],
    residue_index: TensorType["n_res"],
    pdb_key: str,
    model_plot_name: str,
    output_dir: str,
) -> None:
    """Plots and saves a matrix of coupling strengths."""
    fig, ax = plt.subplots()
    # Use a colormap suitable for continuous data and add a colorbar
    im = ax.imshow(coupling_matrix.numpy(), cmap="viridis", interpolation="none", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Scaled Frobenius Norm")

    ax.set_title(f"Coupling Strengths for {pdb_key} ({model_plot_name})")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")

    # Set the axis ticks to be the residue index values instead of array indices
    res_indices_np = residue_index.numpy()
    num_res = len(res_indices_np)
    # Select a subset of indices for ticks to avoid clutter on the axes
    # and handle cases with very few residues.
    num_ticks = min(num_res, 15)  # Show at most 15 ticks
    tick_locs = np.linspace(0, num_res - 1, num=num_ticks, dtype=int)
    tick_labels = res_indices_np[tick_locs]

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(f"{output_dir}/{pdb_key}_{model_plot_name}_couplings.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
