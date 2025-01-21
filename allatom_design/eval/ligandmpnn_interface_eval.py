import csv
import json
import os
from pathlib import Path
import hydra
import lightning as L
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs/eval", config_name="ligandmpnn_interface_eval", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given:
     - pdb_dir, pdb_key_list, fixed_pos_csv
     - fraction, method (seq or scn)

    1) Read PDB keys and create a mapping of:
         {"<absolute_path_to_pdb>": ""}
       for all PDBs => This is for --pdb_path_multi usage.

    2) Read the CSV [pdb_name,fixed_pos_seq,fixed_pos_scn], parse fraction *some* way?
       Actually, you might have already done partial fraction picking, so we just pick the relevant column
       (fixed_pos_seq if method==seq, fixed_pos_scn if method==scn).

    3) Produce "fixed_residues_multi.json", containing e.g.:
       {
         "/abs/path/file1.pdb": "A12 A13 B5",
         "/abs/path/file2.pdb": "A7 A8"
       }

    4) Optionally run ligandmpnn's run.py right away, or just produce the files.

    The final output is stored in:
      cfg.out_base/method_fraction/
    e.g.   /scratch/.../ligandmpnn/scn_0.3
    """

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Construct output directory: e.g. out_base/scn_0.3 or seq_0.8
    subdir_name = f"{cfg.method}_{cfg.fraction}"
    out_dir = Path(cfg.out_base) / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Hydra config for record
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # 1) Read PDB keys
    with open(cfg.pdb_key_list, "r") as f:
        pdb_keys = f.read().splitlines()

    # Build absolute paths
    pdb_paths = []
    for key in pdb_keys:
        pdb_file = os.path.join(cfg.pdb_dir, f"{key}{cfg.pdb_key_ext}")
        pdb_paths.append(os.path.abspath(pdb_file))

    # 2) Create "pdb_ids.json" for run.py with --pdb_path_multi
    #    The recommended format is: { <path> : "" , <path2>: "" , ... }
    pdb_ids_dict = {pdb_path: "" for pdb_path in pdb_paths}
    pdb_ids_json = out_dir / "pdb_ids.json"
    with open(pdb_ids_json, "w") as f:
        json.dump(pdb_ids_dict, f, indent=2)

    # 3) Create "fixed_residues_multi.json"
    #    We'll read the CSV, pick the relevant column (fixed_pos_seq if method="seq"
    #    or fixed_pos_scn if method="scn"), parse them as space-separated strings.

    # If fraction is not literally dictating partial subset in the code, we assume
    # the CSV is already picking e.g. 0.3 fraction. So we just read that CSV, ignoring fraction inside.
    # Or if your CSV includes *multiple* t values in one file, you'll have to filter.
    # We'll assume each CSV is for exactly one fraction + method for clarity.

    # For each row in the CSV: row = [pdb_name, fixed_seq_str, fixed_scn_str]
    # We figure out the relevant column. Then we store => "A12 A13 B5"
    # That means these positions are fixed.
    # We'll match row[0] to the PDB key to produce /abs/path. If method=seq => row[1], else row[2].
    fixed_res_column = 1 if cfg.method == "seq" else 2

    # Read the CSV
    csv_map = {}
    with open(cfg.fixed_pos_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # row: [pdb_name, fixed_seq_str, fixed_scn_str]
            pdb_name = row[0]
            # parse the relevant column
            pos_str = row[fixed_res_column].strip()
            # store in dict
            csv_map[pdb_name] = pos_str

    # Now build the big JSON:
    # e.g. {"/abs/path/1abc.pdb": "A12 A13 B5", "/abs/path/4GYT.pdb": "..."}
    # We'll find row by matching key == row[0].
    fix_res_dict = {}
    for key, pdb_path in zip(pdb_keys, pdb_paths):
        if key in csv_map and csv_map[key]:
            # convert "A1,A2,B3" -> "A1 A2 B3" for run.py
            space_string = csv_map[key].replace(",", " ")
        else:
            space_string = ""
        fix_res_dict[pdb_path] = space_string

    fix_residues_json = out_dir / "fixed_residues_multi.json"
    with open(fix_residues_json, "w") as f:
        json.dump(fix_res_dict, f, indent=2)

    # 4) Optionally run run.py
    if cfg.run_ligandmpnn:
        # We'll build the run.py command
        # E.g. something like:
        #   python run.py \
        #       --model_type "ligand_mpnn" \
        #       --pdb_path_multi "pdb_ids.json" \
        #       --fixed_residues_multi "fixed_residues_multi.json" \
        #       --out_folder . \
        #       --seed 111 \
        #       --ligand_mpnn_use_side_chain_context 1|0 ...
        # We'll set use_side_chain_context=1 if method=="scn", else 0 if method=="seq".
        mpnn_cmd = [
            "conda", "run", "-n", f"{cfg.ligandmpnn_env_name}",
            "python", f"{cfg.ligandmpnn_base_dir}/run.py",
            f"--model_type={cfg.mpnn.model_type}",
            f"--pdb_path_multi={pdb_ids_json}",
            f"--fixed_residues_multi={fix_residues_json}",
            f"--out_folder={out_dir}",  # write outputs here
            f"--seed={cfg.mpnn.seed}",
        ]

        # Specify checkpoint path
        ckpt = f"{cfg.mpnn.mpnn_params_dir}/{cfg.mpnn.checkpoint_name}"
        mpnn_cmd.append(f"--checkpoint_ligand_mpnn={ckpt}")

        # For method "scn", set side_chain_context=1; for "seq", set 0
        side_chain_context_flag = 1 if cfg.method == "scn" else 0
        mpnn_cmd.append(f"--ligand_mpnn_use_side_chain_context={side_chain_context_flag}")

        # If the user specified parse_these_chains_only
        if cfg.mpnn.parse_these_chains_only:
            mpnn_cmd.append(f"--parse_these_chains_only={cfg.mpnn.parse_these_chains_only}")

        # If the user specified chains_to_design
        if cfg.mpnn.chains_to_design:
            mpnn_cmd.append(f"--chains_to_design={cfg.mpnn.chains_to_design}")

        # Print or run the command:
        print(" ".join(mpnn_cmd))
        os.system(" ".join(mpnn_cmd))

        # print the command to output file
        with open(out_dir / "run_command.txt", "w") as f:
            f.write(" ".join(mpnn_cmd))

    else:
        print(f"Prepared JSONs for method={cfg.method}, fraction={cfg.fraction}, out_dir={out_dir}")
        print("Not running run.py automatically (set run_ligandmpnn=true to enable).")


if __name__ == "__main__":
    main()
