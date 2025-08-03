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

    # Create a mapping from model name to its base directory for easy lookup
    model_name_to_base_dir = {m.model_name: m.base_dir for m in cfg.model_csvs}
    model1_name, model2_name = cfg.comparisons
    model1_dir = model_name_to_base_dir[model1_name]
    model2_dir = model_name_to_base_dir[model2_name]

    # Load in PDB files
    pdb_files = get_pdb_files(**cfg.input_cfg)
    temp_processed_struct_dir = f"{cfg.base_out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=temp_processed_struct_dir, **cfg.pdb_processing_cfg)

    # Prepare directory for plots
    plot_out_dir = f"{cfg.base_out_dir}/plots"
    Path(plot_out_dir).mkdir(parents=True, exist_ok=True)

    # Process each PDB file to extract contact maps
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)
    record_id_to_contact_map = {}
    record_id_to_residue_index = {}
    record_id_to_mask_1d = {}
    for struct_file in tqdm(processed_struct_files, desc="Processing PDBs"):
        example, _ = get_sd_example(struct_file, data_cfg)
        X = example["disto_center"]  # CB atoms, CA for glycine
        mask_1d = (example["token_disto_mask"] * example["token_pad_mask"] * example["token_resolved_mask"]).bool()  # remove pad and unresolved atoms
        record_id_to_mask_1d[example["pdb_key"]] = mask_1d

        # mask out pad and unresolved atoms
        X = X[mask_1d]
        residue_index = example["residue_index"][mask_1d]

        # get contact map based on distance cutoff
        contact_map = torch.cdist(X, X, p=2) < cfg.contact_map_cutoff
        record_id_to_contact_map[example["pdb_key"]] = contact_map
        record_id_to_residue_index[example["pdb_key"]] = residue_index


    # Plot and save the contact maps along with potts parameters
    for record_id in tqdm(record_id_to_contact_map.keys(), desc="Plotting results"):
        contact_map = record_id_to_contact_map[record_id]
        residue_index = record_id_to_residue_index[record_id]
        mask_1d = record_id_to_mask_1d[record_id]

        # Plot the ground truth contact map once per record
        plot_contact_map(
            contact_map=contact_map,
            residue_index=residue_index,
            pdb_key=record_id,
            output_dir=plot_out_dir,
        )

        # Get the scaled frobenius norm for each model
        potts_path1 = f"{model1_dir}/{record_id}.pt"
        norm1 = get_scaled_fro_norm(potts_path1, mask_1d)

        potts_path2 = f"{model2_dir}/{record_id}.pt"
        norm2 = get_scaled_fro_norm(potts_path2, mask_1d)

        # Plot individual coupling strengths for reference
        plot_coupling_strengths(
            coupling_matrix=norm1,
            residue_index=residue_index,
            pdb_key=record_id,
            model_plot_name=model1_name,
            output_dir=plot_out_dir
        )
        plot_coupling_strengths(
            coupling_matrix=norm2,
            residue_index=residue_index,
            pdb_key=record_id,
            model_plot_name=model2_name,
            output_dir=plot_out_dir
        )

        # Calculate and plot the difference (model2 - model1)
        diff_matrix = norm2 - norm1
        plot_coupling_difference(
            diff_matrix=diff_matrix,
            residue_index=residue_index,
            pdb_key=record_id,
            model1_name=model1_name,
            model2_name=model2_name,
            output_dir=plot_out_dir
        )


def get_scaled_fro_norm(potts_params_path: str, mask_1d: TensorType["n_res"]) -> TensorType["n_res", "n_res"]:
    """Loads Potts parameters, calculates, and scales the Frobenius norm."""
    potts_params = torch.load(potts_params_path)
    J = potts_params["J"]
    mask_ij = potts_params["mask_ij"]
    # restrict to protein-only interactions and mask out pad and unresolved atoms
    J = J[:, :, 2:22, 2:22]
    J = J * mask_ij[..., None, None]
    J = J[mask_1d]
    J = J[:, mask_1d]

    # calculate the frobenius norm for each 20x20 residue pair matrix
    fro_norm_J = torch.linalg.norm(J, dim=(-2, -1))

    # scale the frobenius norm from 0 to 1
    min_val = fro_norm_J.min()
    max_val = fro_norm_J.max()
    if (max_val - min_val) > 1e-8: # avoid division by zero
        scaled_fro_norm = (fro_norm_J - min_val) / (max_val - min_val)
    else:
        scaled_fro_norm = torch.zeros_like(fro_norm_J)

    return scaled_fro_norm


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

    ax.set_title(f"Coupling Strengths for {pdb_key}\n({model_plot_name})")
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


def plot_coupling_difference(
    diff_matrix: TensorType["n_res", "n_res"],
    residue_index: TensorType["n_res"],
    pdb_key: str,
    model1_name: str,
    model2_name: str,
    output_dir: str,
) -> None:
    """Plots the difference between two coupling strength matrices."""
    fig, ax = plt.subplots()

    # Center the colormap at 0 for differences
    diff_data = diff_matrix.numpy()
    max_abs_val = np.max(np.abs(diff_data))

    # Use a diverging colormap
    im = ax.imshow(diff_data, cmap="coolwarm", interpolation="none", vmin=-max_abs_val, vmax=max_abs_val)
    fig.colorbar(im, ax=ax, label="Difference in Scaled Norm")

    ax.set_title(f"Coupling Difference for {pdb_key}\n({model2_name} - {model1_name})")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")

    # Set the axis ticks to be the residue index values
    res_indices_np = residue_index.numpy()
    num_res = len(res_indices_np)
    num_ticks = min(num_res, 15)
    tick_locs = np.linspace(0, num_res - 1, num=num_ticks, dtype=int)
    tick_labels = res_indices_np[tick_locs]

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(f"{output_dir}/{pdb_key}_couplings_diff.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()