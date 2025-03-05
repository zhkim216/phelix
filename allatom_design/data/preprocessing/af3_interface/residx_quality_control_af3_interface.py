"""
Quality control + residue numbering for AF3 mmCIFs
"""

import multiprocessing
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import typer
from Bio.PDB import Chain, Model, Residue, Selection, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import FastMMCIFParser
from tqdm import tqdm
from collections import defaultdict
import fcntl
import os

label_seqid_parser = FastMMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}


def get_pdb_keys(pdb_keys_fp: Path, pdb_store_fp: Path, out_list_name: Path):
    remaining_keys = []
    with open(pdb_keys_fp, 'r') as fp:
        for line in fp.readlines():
            pdb_key = line.strip()
            # We check if file exists in the new mmcif-based store directory
            if (pdb_store_fp / f"{pdb_key}.cif").exists():
                remaining_keys.append(pdb_key)
    with open(pdb_store_fp.parent / out_list_name, 'w') as fp:
        fp.write('\n'.join(remaining_keys) + '\n')


def group_consecutive_idx(nums):
    nums = np.array(nums)
    breaks = np.where(np.diff(nums) > 1)[0] + 1
    result = np.split(nums, breaks)
    return [sublist.tolist() for sublist in result]


def save_residues(
    chain_id_to_residues: dict[str, list[Residue.Residue]],
    save_fp: Path, shift_res: bool = False
):
    """
    Save a list of residues to a PDB structure
    If {shift_res} is True, also shifts the residx to start from 1 (assume monotonically increasing)
    """
    io = MMCIFIO()

    new_structure = Structure.Structure("s")
    new_model = Model.Model(0)

    for chain_id, residues in chain_id_to_residues.items():
        new_chain = Chain.Chain(chain_id)
        idx_offset = residues[0].id[1] - 1
        for residue in residues:
            if shift_res:
                curr_id = list(residue.id)
                curr_id[1] = curr_id[1] - idx_offset
                residue.id = tuple(curr_id)
            new_chain.add(residue)
        new_model.add(new_chain)

    new_structure.add(new_model)
    io.set_structure(new_structure)
    io.save(str(save_fp))


def runner(
    pdb_key: str,
    mmcif_dir: Path = Path("/path/to/your/cif/files"),
    af3_mmcif_dir: Path = Path("/path/to/your/af3/cif/files"),
    save_dir: Path = Path("/path/to/save"),
    shift_res: bool = False,
):
    """
    For each key (assumed to correspond to a .cif file), parse the structure,
    check resolution + R-free, etc. Then optionally save the subset of residues.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_code, chain_id_1, chain_id_2, _ = pdb_key.split("_")
    interface_chain_ids = [chain_id_1, chain_id_2]

    # Load in cif from mmCIF directory; for reading in resolution / rfree
    cif_fp = mmcif_dir / f"{pdb_code[1:3]}/{pdb_code}.cif"
    if not cif_fp.exists():
        return (pdb_key, f"No MMCIF found at {cif_fp}", 0)

    # Load in cif from AF3 directory; for extracting structure
    af3_cif_fp = af3_mmcif_dir / f"{pdb_code[1:3]}/{pdb_code}-assembly1.cif"
    if not af3_cif_fp.exists():
        return (pdb_key, f"No AF3 MMCIF found at {af3_cif_fp}", 0)

    # Filter by resolution / rfree
    mmcif_dict = MMCIF2Dict(str(cif_fp))

    res_cutoff = 3.5
    rfree_cutoff = 0.3

    try:
        # Read in resolution
        resolution = float(mmcif_dict["_refine.ls_d_res_high"][0])
        is_em_structure = False
    except Exception as err:
        # try reading EM resolution
        try:
            resolution = float(mmcif_dict["_em_3d_reconstruction.resolution"][0])
            is_em_structure = True
        except Exception as err:
            return (pdb_key, f"Undefined resolution. Error: {err}", None)

    # Filter by resolution
    if resolution > res_cutoff:
        return (pdb_key, f"Low quality: Resolution={resolution}", None)

    # Filter by rfree if not EM structure
    if not is_em_structure:
        try:
            rfree = float(mmcif_dict["_refine.ls_R_factor_R_free"][0])
        except Exception as err:
            return (pdb_key, f"Undefined rfree. Error: {err}", None)

        if rfree > rfree_cutoff:
            return (pdb_key, f"Low quality: Rfree={rfree}", None)

    # Parse the structure
    model_num = 0
    s1 = label_seqid_parser.get_structure("s", af3_cif_fp)[model_num]

    # Extract residues from chain of interest
    chain_id_to_label_seqids = defaultdict(list)
    chain_id_to_cif_residues = defaultdict(list)

    all_chains = []
    for res in Selection.unfold_entities(s1, "R"):
        # Convert to single-letter code if recognized
        resname = restype_3to1.get(res.resname, "X")
        if resname != "X":
            chain_id = res.get_parent().id
            all_chains.append(chain_id)
            if chain_id in interface_chain_ids:
                chain_id_to_label_seqids[chain_id].append(res.id[1])
                chain_id_to_cif_residues[chain_id].append(res)

    num_chains = len(set(all_chains))

    # Verify that we have both chains of the interface
    if len(chain_id_to_label_seqids) != 2:
        return (pdb_key, f"Only found {len(chain_id_to_label_seqids)} chains", num_chains)

    # Verify that each chain has at least one residue
    for chain, cif_residues in chain_id_to_cif_residues.items():
        if len(cif_residues) == 0:
            return (pdb_key, f"No residues found in chain {chain}", num_chains)

    # If everything looks good, save the residues
    save_residues(chain_id_to_cif_residues, save_dir / f"{pdb_key}.cif", shift_res=shift_res)

    num_gaps = sum([len(group_consecutive_idx(label_seqids)) - 1
                    for label_seqids in chain_id_to_label_seqids.values()])

    return (pdb_key, f"Has {num_gaps} gaps", num_chains)


def multiprocess_runner(
    pdb_keys: Path,
    mmcif_dir: Path = Path("/path/to/your/cif/files"),
    af3_mmcif_dir: Path = Path("/path/to/your/af3/cif/files"),
    max_threads: int = 8,
    save_dir: Path = Path("/path/to/save"),
    out_list_name: Path = Path("/path/to/save/pdb_keys.list"),
    shift_res: bool = False,
    array_id: int = None,
    num_arrays: int = None,
    fix_missing: bool = False,
):
    """
    Main function that reads in PDB keys, optionally splits them among array jobs,
    skips keys that have already been processed, and writes results to a shared CSV
    using file locking for parallel safety.
    """
    # Where we store the output CSV
    out_df_path = save_dir.parent / f"problematic_pdbs_{pdb_keys.stem}.csv"

    # Read the full key list
    all_pdb_keys = []
    with open(pdb_keys, "r") as fp:
        for line in fp.readlines():
            all_pdb_keys.append(line.strip())

    # If fix_missing, load the existing CSV (if it exists) and skip any that are already processed
    if fix_missing and out_df_path.exists():
        existing_df = pd.read_csv(out_df_path)
        done_set = set(existing_df["pdb"].tolist())
        unprocessed = [k for k in all_pdb_keys if k not in done_set]
        all_pdb_keys = unprocessed

    # If array_id is given, slice out the chunk of keys for this job
    if array_id is not None and num_arrays is not None:
        chunk_size = int(np.ceil(len(all_pdb_keys) / num_arrays))
        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_pdb_keys))
        all_pdb_keys = all_pdb_keys[start_idx:end_idx]

    # Run in parallel
    with multiprocessing.Pool(max_threads) as p:
        ret = list(tqdm(
            p.imap(
                partial(
                    runner,
                    mmcif_dir=mmcif_dir,
                    af3_mmcif_dir=af3_mmcif_dir,
                    save_dir=save_dir,
                    shift_res=shift_res
                ),
                all_pdb_keys
            ),
            total=len(all_pdb_keys),
            desc="Processing mmCIFs"
        ))

    # Build a DataFrame from the results
    df = pd.DataFrame()
    df["pdb"] = [r[0] for r in ret]
    df["problem"] = [r[1] for r in ret]
    df["num_chains"] = [r[2] for r in ret]

    # Append results to the CSV with a file lock
    with open(out_df_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        file_empty = (f.tell() == 0)
        df.to_csv(f, index=False, header=file_empty)
        fcntl.flock(f, fcntl.LOCK_UN)

    # Update the filtered pdb_keys list
    get_pdb_keys(pdb_keys, save_dir, out_list_name)


if __name__ == "__main__":
    typer.run(multiprocess_runner)
