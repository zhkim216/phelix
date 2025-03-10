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

# Ensure you have an import for restypes_with_x or equivalent:
import allatom_design.data.residue_constants as rc
from Bio import SeqIO

@hydra.main(version_base="1.3.2", config_path="../../configs/eval/plots", config_name="plot_monomer_interface_context_eval")
def main(cfg: DictConfig):
    """
    Combines both interface and monomer evaluation into one script.
    We:
      1) Optionally read a previously cached raw dataframe if cfg.read_from_df is true.
         Otherwise, load PKLs from interface_eval_dirs and monomer_eval_dirs
         to compute per-PDB sequence accuracy (predicted vs. true),
         and load the "ligandmpnn" FASTA files from interface_ligandmpnn_eval_dirs
         and monomer_ligandmpnn_eval_dirs to compute sequence accuracy
         using the same reference data from PKLs.
      2) Produce three line plots:
         (a) Combined (interface+monomer) in a single figure (8 lines total).
         (b) Interface-only figure (4 lines).
         (c) Monomer-only figure (4 lines).
      3) Save dataframes as CSV for debugging or subsequent runs.
    """
    torch.set_grad_enabled(False)
    L.seed_everything(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If user wants to read from a cached raw dataframe, do that and skip data loading
    if cfg.read_from_df:
        csv_path = out_dir / "combined_interface_monomer_eval_raw.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Cached dataframe not found at {csv_path}")
        df_all = pd.read_csv(csv_path)
        print(f"[INFO] Loaded cached raw dataframe of shape {df_all.shape} from {csv_path}")
    else:
        # (1a) Load PKLs (interface + monomer)
        df_interface_pkl, interface_dict = load_design_eval_data(
            cfg.interface_eval_dirs, context="interface"
        )
        df_monomer_pkl, monomer_dict = load_design_eval_data(
            cfg.monomer_eval_dirs, context="monomer"
        )

        # (1b) Load ligandmpnn FASTAs
        df_interface_lig = parse_ligandmpnn_fasta_dirs(
            cfg.interface_ligandmpnn_eval_dirs, interface_dict, context="interface"
        )
        df_monomer_lig = parse_ligandmpnn_fasta_dirs(
            cfg.monomer_ligandmpnn_eval_dirs, monomer_dict, context="monomer"
        )

        # Combine
        df_all = pd.concat([df_interface_pkl, df_monomer_pkl, df_interface_lig, df_monomer_lig],
                           ignore_index=True)

        # Save raw CSV
        df_all.to_csv(out_dir / "combined_interface_monomer_eval_raw.csv", index=False)
        print(f"[INFO] Built & saved combined raw df of shape {df_all.shape} to disk.")

    # We'll define color + line-style logic for clarity
    methods = ["FAMPNN (0.3$\\mathrm{\\AA}$), w/ sidechains",
               "FAMPNN (0.3$\\mathrm{\\AA}$)",
               "LigandMPNN (0.3$\\mathrm{\\AA}$), w/ sidechains",
               "LigandMPNN (0.3$\\mathrm{\\AA}$)"]
    color_map = {
        methods[0]: "#005B96",
        methods[1]: "#005B96",
        methods[2]: "#AAAAAA",
        methods[3]: "#AAAAAA",
    }
    df_all["method_name"] = df_all["method"].replace({
        "scn": methods[0],
        "seq": methods[1],
        "ligandmpnn_scn": methods[2],
        "ligandmpnn_seq": methods[3],
    })
    style_map = {
        "interface": "-",
        "monomer": "--",
    }

    # (2) Group by (method, fraction, context) -> mean seq_acc
    grouped = df_all.groupby(["method_name", "fraction", "context"], dropna=False)["seq_acc"].mean().reset_index()

    grouped_csv_path = out_dir / "combined_interface_monomer_eval_grouped.csv"
    grouped.to_csv(grouped_csv_path, index=False)
    print(f"[INFO] Grouped results saved to {grouped_csv_path}")

    ##################################################
    #  2b) Interface-only figure
    ##################################################
    plt.figure(figsize=(6, 4))
    subdf_interface = grouped[grouped["context"] == "interface"].copy()
    for i, m in enumerate(methods):
        sub = subdf_interface[subdf_interface["method_name"] == m].sort_values("fraction")
        if len(sub) == 0:
            continue
        plt.plot(
            sub["fraction"] * 100,
            sub["seq_acc"] * 100,
            marker="o",
            label=m,
            color=color_map[m],
            linestyle="-" if i % 2 == 0 else "--",
            alpha=0.9
        )

    # Add arrows going from seq to seq+sidechain at partial context=90% for each pair
    # FAMPNN
    fampnn_seq = subdf_interface[subdf_interface["method_name"] == methods[1]]
    fampnn_scn = subdf_interface[subdf_interface["method_name"] == methods[0]]
    seq_90 = fampnn_seq.loc[np.isclose(fampnn_seq["fraction"], 0.9)]
    scn_90 = fampnn_scn.loc[np.isclose(fampnn_scn["fraction"], 0.9)]
    if not seq_90.empty and not scn_90.empty:
        x_seq_90 = seq_90["fraction"].values[0] * 100
        y_seq_90 = seq_90["seq_acc"].values[0] * 100
        x_scn_90 = scn_90["fraction"].values[0] * 100
        y_scn_90 = scn_90["seq_acc"].values[0] * 100
        fold_change = (y_scn_90 / y_seq_90)
        plt.annotate(
            f"{fold_change:.2f}x",
            xy=(x_scn_90, y_scn_90),
            xytext=(x_seq_90, y_seq_90),
            arrowprops=dict(arrowstyle="->", color=color_map[methods[0]]),
            fontsize=9,
            color=color_map[methods[0]],
            ha="center"
        )

    # LigandMPNN
    lig_seq = subdf_interface[subdf_interface["method_name"] == methods[3]]
    lig_scn = subdf_interface[subdf_interface["method_name"] == methods[2]]
    seq_90 = lig_seq.loc[np.isclose(lig_seq["fraction"], 0.9)]
    scn_90 = lig_scn.loc[np.isclose(lig_scn["fraction"], 0.9)]
    if not seq_90.empty and not scn_90.empty:
        x_seq_90 = seq_90["fraction"].values[0] * 100
        y_seq_90 = seq_90["seq_acc"].values[0] * 100
        x_scn_90 = scn_90["fraction"].values[0] * 100
        y_scn_90 = scn_90["seq_acc"].values[0] * 100
        fold_change = (y_scn_90 / y_seq_90)
        plt.annotate(
            f"{fold_change:.2f}x",
            xy=(x_scn_90, y_scn_90),
            xytext=(x_seq_90, y_seq_90),
            arrowprops=dict(arrowstyle="->", color=color_map[methods[2]]),
            fontsize=9,
            color=color_map[methods[2]],
            ha="center"
        )

    plt.xlabel("Partial context given (%)", fontsize=12)
    plt.ylabel("Mean sequence recovery", fontsize=12)
    plt.title("Interface sequence recovery", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.xticks(np.arange(0, 101, 10))
    plt.tight_layout()
    out_fig_interface = out_dir / "interface_seq_recovery.png"
    plt.savefig(out_fig_interface, dpi=300)
    plt.savefig(out_fig_interface.with_suffix(".pdf"), dpi=300, transparent=True)
    plt.close()

    ##################################################
    #  2a) Combined figure: interface + monomer
    ##################################################
    plt.figure(figsize=(7, 5))
    for m in methods:
        for c in ["interface", "monomer"]:
            subdf = grouped[(grouped["method_name"] == m) & (grouped["context"] == c)]
            if len(subdf) == 0:
                continue
            subdf = subdf.sort_values("fraction", ascending=True)
            plt.plot(
                subdf["fraction"],
                subdf["seq_acc"],
                label=f"{m}-{c}",
                color=color_map[m],
                linestyle=style_map[c],
                marker="o",
            )

    plt.xlabel("t fraction", fontsize=12)
    plt.ylabel("Mean Sequence Recovery", fontsize=12)
    plt.title("Monomer & Interface Sequence Recovery", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_fig_combined = out_dir / "monomer_interface_seq_recovery.png"
    plt.savefig(out_fig_combined, dpi=150)
    plt.close()

    ##################################################
    #  2c) Monomer-only figure
    ##################################################
    plt.figure(figsize=(6, 4))
    subdf_monomer = grouped[grouped["context"] == "monomer"].copy()
    for m in methods:
        sub = subdf_monomer[subdf_monomer["method_name"] == m].sort_values("fraction")
        if len(sub) == 0:
            continue
        plt.plot(
            sub["fraction"],
            sub["seq_acc"],
            marker="o",
            label=m,
            color=color_map[m],
            linestyle="-",
        )
    plt.xlabel("t fraction", fontsize=12)
    plt.ylabel("Mean Sequence Recovery", fontsize=12)
    plt.title("Monomer Sequence Recovery", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_fig_monomer = out_dir / "monomer_seq_recovery.png"
    plt.savefig(out_fig_monomer, dpi=150)
    plt.close()

    print(f"[DONE] Plots saved in {out_dir}.")


def load_design_eval_data(eval_dirs: List[str], context: str):
    """
    For each directory in eval_dirs:
      1) We parse `method` and `fraction` from the final directory name, e.g. "scn_0.3".
      2) For each PKL in `sample_pkls`, we load predicted aatype, original aatype, override masks, etc.
         Then compute seq_acc with the appropriate valid_mask (depending on interface vs. monomer).
      3) Return:
         - a DataFrame with columns [pdb, pred_seq, true_seq, seq_acc, method, fraction, context]
         - a nested dict pkl_map[fraction][basename] = { "true_aatype":..., "override_seq_mask":..., "override_scn_mask":..., "context_mask":... }
           for future lookups when analyzing ligandmpnn FASTAs.
    """
    all_rows = []
    pkl_map = defaultdict(dict)

    for path in eval_dirs:
        path = Path(path)
        dir_name = path.name  # e.g. "scn_0.3"
        if "_" not in dir_name:
            print(f"Warning: directory name doesn't contain '_': {dir_name}, skipping.")
            continue
        method, frac_str = dir_name.split("_", 1)
        fraction = float(frac_str)

        sample_pkl_dir = path / "sample_pkls"
        if not sample_pkl_dir.is_dir():
            print(f"Warning: {sample_pkl_dir} does not exist, skipping {path}.")
            continue

        pkl_files = list(sample_pkl_dir.glob("*.pkl"))
        if len(pkl_files) == 0:
            print(f"Warning: no PKL files in {sample_pkl_dir}, skipping {path}.")
            continue

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            pred_aatype = data["pred_aatype"]
            true_aatype = data["original_aatype"]

            override_seq_mask = data["aatype_override_mask"].astype(bool)
            override_scn_mask = data["scn_override_mask"].astype(bool)

            if context == "interface":
                if "interface_residue_mask" not in data:
                    print(f"Error: interface_residue_mask missing in {pkl_file}, skipping.")
                    continue
                context_mask = data["interface_residue_mask"].astype(bool)
            else:
                context_mask = np.ones_like(pred_aatype, dtype=bool)

            x_idx = rc.restype_order_with_x["X"]
            not_unknown_pred = (pred_aatype != x_idx)
            not_unknown_true = (true_aatype != x_idx)

            valid_mask = (
                context_mask &
                (~override_seq_mask) &
                (~override_scn_mask) &
                not_unknown_pred &
                not_unknown_true
            )

            if np.sum(valid_mask) == 0:
                seq_acc = float("nan")
            else:
                seq_acc = (pred_aatype[valid_mask] == true_aatype[valid_mask]).mean()

            pred_seq_str = "".join(rc.restypes_with_x[a] for a in pred_aatype)
            true_seq_str = "".join(rc.restypes_with_x[a] for a in true_aatype)

            row = {
                "pdb": pkl_file.stem,
                "pred_seq": pred_seq_str,
                "true_seq": true_seq_str,
                "seq_acc": seq_acc,
                "method": method,
                "fraction": fraction,
                "context": context,
            }
            all_rows.append(row)

            basename = pkl_file.stem.replace("_sample0", "")
            pkl_map[fraction][basename] = {
                "true_aatype": true_aatype,
                "override_seq_mask": override_seq_mask,
                "override_scn_mask": override_scn_mask,
                "context_mask": context_mask,
            }

    df = pd.DataFrame(all_rows)
    return df, pkl_map


def parse_ligandmpnn_fasta_dirs(ligandmpnn_eval_dirs: List[str], pkl_map: Dict, context: str):
    """
    For each directory in ligandmpnn_eval_dirs:
      1) Parse `method` and `fraction` from the parent's parent dir name, e.g. "scn_0.3".
         We'll produce a "ligandmpnn_scn" or "ligandmpnn_seq" method name by prefixing "ligandmpnn_".
      2) For each FASTA file found, load the 2 records:
         - 1st is 'WT' or original
         - 2nd is 'designed'
      3) Match with stored PKL data in pkl_map[fraction], same `basename`.
         Recompute seq_acc with the same valid_mask used by load_design_eval_data.
    """
    rows = []
    for path in ligandmpnn_eval_dirs:
        path = Path(path)
        parent_of_parent = path.parent.name
        if "_" not in parent_of_parent:
            print(f"Warning: {parent_of_parent} doesn't contain '_', skipping {path}")
            continue

        raw_method, frac_str = parent_of_parent.split("_", 1)
        fraction = float(frac_str)
        method = f"ligandmpnn_{raw_method}"

        fa_files = list(path.glob("*.fa")) + list(path.glob("*.fasta"))
        if len(fa_files) == 0:
            print(f"Warning: no FASTA files found in {path}, skipping.")
            continue

        if fraction not in pkl_map:
            print(f"Warning: fraction={fraction} not in pkl_map, skipping {path}")
            continue

        for fa_file in fa_files:
            basename = fa_file.stem
            if basename not in pkl_map[fraction]:
                alt_basename = basename.replace("_sample0", "")
                if alt_basename not in pkl_map[fraction]:
                    print(f"Warning: no matching PKL data for {fa_file}, skipping.")
                    continue
                basename = alt_basename

            arrays = pkl_map[fraction][basename]
            true_aatype = arrays["true_aatype"]
            override_seq_mask = arrays["override_seq_mask"]
            override_scn_mask = arrays["override_scn_mask"]
            context_mask = arrays["context_mask"]

            records = list(SeqIO.parse(str(fa_file), "fasta"))
            if len(records) < 2:
                print(f"Warning: {fa_file} has <2 records, skipping.")
                continue

            wt_record, designed_record = records[0], records[1]
            wt_seq_str = str(wt_record.seq).replace(":", "")
            true_seq_str = "".join(rc.restypes_with_x[a] for a in true_aatype)
            if len(wt_seq_str) != len(true_seq_str):
                print(f"Warning: mismatch in length for {fa_file}, skipping.")
                continue

            designed_seq_str = str(designed_record.seq).replace(":", "")

            x_idx = rc.restype_order_with_x["X"]
            pred_aatype_lig = []
            for aa in designed_seq_str:
                pred_aatype_lig.append(rc.restype_order_with_x.get(aa, x_idx))
            pred_aatype_lig = np.array(pred_aatype_lig, dtype=int)

            not_unknown_true = (true_aatype != x_idx)
            not_unknown_pred = (pred_aatype_lig != x_idx)
            valid_mask = (
                context_mask &
                (~override_seq_mask) &
                (~override_scn_mask) &
                not_unknown_true &
                not_unknown_pred
            )

            if np.sum(valid_mask) == 0:
                seq_acc = float("nan")
            else:
                seq_acc = (pred_aatype_lig[valid_mask] == true_aatype[valid_mask]).mean()

            row = {
                "pdb": basename,
                "pred_seq": designed_seq_str,
                "true_seq": true_seq_str,
                "seq_acc": seq_acc,
                "method": method,
                "fraction": fraction,
                "context": context,
            }
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
