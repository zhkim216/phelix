import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

import allatom_design.data.residue_constants as rc
from tqdm import tqdm

# Biopython import for FASTA parsing
from Bio import SeqIO


@hydra.main(version_base="1.3.2", config_path="../../configs/eval/plots", config_name="plot_interface_eval")
def main(cfg: DictConfig):
    """
    Loads PKL files from cfg.eval_dirs and computes interface sequence accuracy
    (ignoring any positions where aatype_override_mask=1 or scn_override_mask=1,
     ignoring unknown (X) tokens in predicted or true sequences).
    Then, in a separate loop, loads FASTA files from cfg.ligandmpnn_eval_dirs,
    uses the same interface information, and computes interface sequence accuracy
    for each sampled sequence in the FASTA files (if the matching data is found).
    """
    torch.set_grad_enabled(False)
    L.seed_everything(cfg.seed)

    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # This dictionary will map "some_pdb_id" -> dict of arrays for
    # original_aatype, interface_mask, override_seq_mask, override_scn_mask, etc.
    # We'll store them keyed by base_no_sample0 (pkl stem with "_sample0" removed).
    t_to_pdb_data_dict = defaultdict(dict)

    # Store per-sample info from PKLs in a list
    all_rows = []

    # -------------------------------
    # Parse FAMPNN data
    # -------------------------------
    for path in cfg.eval_dirs:
        method, fraction = Path(path).name.split("_")

        sample_pkl_dir = Path(path) / "sample_pkls"
        if not sample_pkl_dir.is_dir():
            print(f"Warning: {sample_pkl_dir} does not exist, skipping.")
            continue

        pkl_files = list(sample_pkl_dir.glob("*.pkl"))
        if len(pkl_files) == 0:
            print(f"Warning: no PKL files found in {sample_pkl_dir}, skipping.")
            continue

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            pred_aatype = data["pred_aatype"]  # shape [N]
            true_aatype = data["original_aatype"]  # shape [N]
            interface_mask = data["interface_residue_mask"].astype(bool)  # shape [N]
            override_seq_mask = data["aatype_override_mask"].astype(bool) # shape [N]
            override_scn_mask = data["scn_override_mask"].astype(bool)    # shape [N]

            # Get valid mask for interface residues that are not overridden
            x_token = rc.restype_order_with_x["X"]
            valid_mask = interface_mask & (~override_seq_mask) & (~override_scn_mask)
            not_unknown_true = (true_aatype != x_token)
            valid_mask = valid_mask & not_unknown_true

            # We also store these arrays for possible re-use with ligandmpnn
            base_no_sample0 = pkl_file.stem.replace("_sample0", "")
            t_to_pdb_data_dict[fraction][base_no_sample0] = {
                "true_aatype": true_aatype,
                "interface_mask": interface_mask,
                "override_seq_mask": override_seq_mask,
                "override_scn_mask": override_scn_mask,
                "valid_mask": valid_mask,
            }

            # Compute interface seq accuracy from the PKL's predicted sequence

            if np.sum(valid_mask) == 0:
                seq_acc = float("nan")
                pred_seq_str = ""
                true_seq_str = ""
            else:
                correct_positions = (pred_aatype[valid_mask] == true_aatype[valid_mask])
                seq_acc = correct_positions.mean()
                pred_seq_str = "".join(rc.restypes_with_x[a] for a in pred_aatype)
                true_seq_str = "".join(rc.restypes_with_x[a] for a in true_aatype)

            row = {
                "pdb": pkl_file.stem,
                "pred_seq": pred_seq_str,
                "true_seq": true_seq_str,
                "seq_acc": seq_acc,
                "method": method,
                "fraction": fraction
            }
            all_rows.append(row)

    # -------------------------------
    # Parse LigandMPNN FASTAS
    # -------------------------------
    # We treat these sequences as "ligandmpnn" method, extracting fraction
    # from the directory name (e.g. "scn_0.1" or "seq_0.3"), ignoring the "base" method part.
    # For each .fa file in that directory, we look up matching info in pdb_data_dict.
    for path in cfg.ligandmpnn_eval_dirs:
        fraction_dir_name = Path(path).parent.name  # e.g. "scn_0.1" or "seq_0.3"
        method, fraction = fraction_dir_name.split("_")
        pdb_data_dict = t_to_pdb_data_dict[fraction]

        fa_files = list(Path(path).glob("*.fa"))
        if len(fa_files) == 0:
            raise ValueError(f"No FASTA files found in {path}")

        for fa_file in fa_files:
            base_no_sample0 = fa_file.stem  # e.g. "7yji_AB_1710"
            if base_no_sample0 not in pdb_data_dict:
                # If we never saw the matching PKL, skip
                raise ValueError(f"No matching PKL found for {fa_file}")

            arrays = pdb_data_dict[base_no_sample0]
            true_aatype = arrays["true_aatype"]
            interface_mask = arrays["interface_mask"]
            override_seq_mask = arrays["override_seq_mask"]
            override_scn_mask = arrays["override_scn_mask"]

            # We'll parse each FASTA record using Biopython
            fasta_entries = parse_fasta(fa_file)

            # double check that true fasta sequence matches the true_aatype
            _, wt_seq = fasta_entries[0]
            true_aatype_str = "".join(rc.restypes_with_x[a] for a in true_aatype)
            wt_seq = wt_seq.replace(":", "")
            if wt_seq != true_aatype_str:
                print(f"Warning: True sequence mismatch for {fa_file}, skipping...")
                continue

            # grab second entry as the sampled sequence
            _, lig_pred_seq = fasta_entries[1]
            lig_pred_seq = lig_pred_seq.replace(":", "")

            valid_mask = arrays["valid_mask"]

            lig_pred_aatype = np.array(
                [rc.restype_order_with_x.get(aa, x_token) for aa in lig_pred_seq]
            )

            if np.sum(valid_mask) == 0:
                seq_acc_lig = float("nan")
                lig_pred_seq_str = ""
            else:
                correct_positions_lig = (lig_pred_aatype[valid_mask] == true_aatype[valid_mask])
                seq_acc_lig = correct_positions_lig.mean()
                lig_pred_seq_str = lig_pred_seq

                row_lig = {
                    "pdb": f"{base_no_sample0}",
                    "pred_seq": lig_pred_seq_str,
                    "true_seq": "".join(rc.restypes_with_x[a] for a in true_aatype),
                    "seq_acc": seq_acc_lig,
                    "method": f"ligandmpnn_{method}",
                    "fraction": fraction
                }
                all_rows.append(row_lig)

    # Create DataFrame with all results
    df = pd.DataFrame(all_rows)

    # Save raw results
    raw_csv_path = Path(out_dir) / "interface_eval_results_raw.csv"
    df.to_csv(raw_csv_path, index=False)

    # Group by (method, fraction) to compute mean accuracy
    grouped = df.groupby(["method", "fraction"], dropna=False)["seq_acc"].mean().reset_index()

    # Save grouped results
    grouped_csv_path = Path(out_dir) / "interface_eval_results_grouped.csv"
    grouped.to_csv(grouped_csv_path, index=False)

    # Plot the results
    plt.figure(figsize=(6, 4))
    methods = grouped["method"].unique()
    for m in methods:
        subdf = grouped[grouped["method"] == m].sort_values("fraction")
        plt.plot(subdf["fraction"], subdf["seq_acc"], marker="o", label=m)

    plt.xlabel("t")
    plt.ylabel("Interface Sequence Recovery")
    plt.title("Interface Sequence Recovery vs t")
    plt.legend()
    plt.tight_layout()
    plot_path = Path(out_dir) / "interface_seq_recovery.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Done! Raw and grouped results saved to {out_dir}")

    # Optional second style of plotting
    colors = ["#005B96", "#CCCCCC", "#999999", "#555555", "#222222"]
    plt.figure(figsize=(6, 4))
    methods = grouped["method"].unique()
    for i, m in enumerate(methods):
        subdf = grouped[grouped["method"] == m].sort_values("fraction")
        plt.plot(
            subdf["fraction"],
            subdf["seq_acc"],
            marker="o",
            markersize=4,
            color=colors[i % len(colors)],
            label=m
        )

    plt.xlabel("t", fontsize=12)
    plt.ylabel("Interface Sequence Recovery", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    plt.savefig(f"{out_dir}/interface_seq_recovery.pdf", dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(f"{out_dir}/interface_seq_recovery.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    print(f"Done! Plots saved to {out_dir}")


def parse_fasta(fa_file: Path) -> List[Tuple[str, str]]:
    """
    Reads a FASTA file via Biopython and returns a list of (header, sequence_string).
    """
    from Bio import SeqIO
    entries = []
    for record in SeqIO.parse(str(fa_file), "fasta"):
        header = record.description
        seq_str = str(record.seq)
        entries.append((header, seq_str))
    return entries


if __name__ == "__main__":
    main()
