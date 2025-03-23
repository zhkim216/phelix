"""
Quality control + residue numbering for AF3 mmCIFs
"""

import fcntl
import glob
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


def get_pdb_keys(pdb_keys_df: pd.DataFrame, filtered_mmcif_dir: Path, save_dir: Path):
    # Filter for keys that exist in the filtered_mmcif_dir
    filtered_pdb_keys = glob.glob(f"{filtered_mmcif_dir}/*.cif")
    filtered_pdb_keys = set([Path(k).stem for k in filtered_pdb_keys])
    pdb_keys_df = pdb_keys_df[pdb_keys_df["pdb_key"].isin(filtered_pdb_keys)]
    pdb_keys_df.to_csv(f"{save_dir}/filtered_pdb_keys.csv", index=False)


def group_consecutive_idx(nums):
    nums = np.array(nums)
    breaks = np.where(np.diff(nums) > 1)[0] + 1
    result = np.split(nums, breaks)
    return [sublist.tolist() for sublist in result]


def save_residues(
    residues: list[Residue.Residue],
    chain_id: str,
    save_fp: Path, shift_res: bool = False
):
    """
    Save a list of residues to a PDB structure
    If {shift_res} is True, also shifts the residx to start from 1 (assume monotonically increasing)
    """
    io = MMCIFIO()

    new_structure = Structure.Structure("s")
    new_model = Model.Model(0)
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
    mmcif_dir: str | Path = Path("/path/to/your/cif/files"),
    af3_mmcif_dir: str | Path = Path("/path/to/your/af3/cif/files"),
    save_dir: str | Path = Path("/path/to/save"),
    shift_res: bool = False,
):
    """
    For each key (assumed to correspond to a .cif file), parse the structure twice:
    - label_seqid_parser (label-based residue numbering)
    - auth_seqid_parser (author-based residue numbering)
    Check the resolution and R-free.
    If valid, compare label_seqids vs auth_seqids to see if they differ, and if not,
    optionally save the subset of residues (using save_residues).
    """
    mmcif_dir, af3_mmcif_dir, save_dir = Path(mmcif_dir), Path(af3_mmcif_dir), Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_code, monomer_chain_id = pdb_key.split("_")

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
    label_seqids = []
    cif_residues = []

    all_chains = []
    for res in Selection.unfold_entities(s1, "R"):
        resname = restype_3to1.get(res.resname, "X")
        if resname != "X":
            chain_id = res.get_parent().id
            all_chains.append(chain_id)
            if chain_id == monomer_chain_id:
                label_seqids.append(res.id[1])
                cif_residues.append(res)

    num_chains = len(set(all_chains))

    # If everything looks good, save the residues
    if len(cif_residues) > 0:
        save_residues(cif_residues, monomer_chain_id, save_dir / f"{pdb_key}.cif", shift_res=shift_res)
    else:
        return (pdb_key, f"No residues found in chain {monomer_chain_id}", num_chains)

    num_gaps = len(group_consecutive_idx(label_seqids)) - 1
    return (pdb_key, f"Has {num_gaps} gaps", num_chains)


def multiprocess_runner(
    pdb_keys_csv: Path,
    mmcif_dir: Path = Path("/path/to/your/cif/files"),
    af3_eval_mmcif_dir: Path = Path("/path/to/your/eval/cif/files"),
    af3_train_mmcif_dir: Path = Path("/path/to/your/train/cif/files"),
    max_threads: int = 8,
    save_dir: Path = Path("/path/to/save"),
    shift_res: bool = False,
    array_id: int = None,
    num_arrays: int = None,
    fix_missing: bool = False
):
    filtered_mmcif_dir = f"{save_dir}/filtered_mmcifs"
    pdb_keys_df = pd.read_csv(pdb_keys_csv)
    for phase in ["eval", "train"]:
        af3_mmcif_dir = af3_eval_mmcif_dir if phase == "eval" else af3_train_mmcif_dir
        pdb_keys = pdb_keys_df[pdb_keys_df["phase"] == phase]["pdb_key"].tolist()
        out_df_path = Path(f"{save_dir}/problematic_pdbs_{phase}.csv")

        if fix_missing and out_df_path.exists():
            existing_df = pd.read_csv(out_df_path)
            done_set = set(existing_df["pdb"].tolist())
            pdb_keys = [k for k in pdb_keys if k not in done_set]

        if array_id is not None and num_arrays is not None:
            chunk_size = int(np.ceil(len(pdb_keys) / num_arrays))
            start_idx = array_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(pdb_keys))
            pdb_keys = pdb_keys[start_idx:end_idx]

        with multiprocessing.Pool(max_threads) as p:
            ret = list(tqdm(
                p.imap(
                    partial(
                        runner,
                        mmcif_dir=mmcif_dir,
                        af3_mmcif_dir=af3_mmcif_dir,
                        save_dir=filtered_mmcif_dir,
                        shift_res=shift_res
                    ),
                    pdb_keys
                ),
                total=len(pdb_keys),
                desc=f"Processing {phase} mmCIFs"
            ))

        df = pd.DataFrame()
        df["pdb"] = [r[0] for r in ret]
        df["problem"] = [r[1] for r in ret]
        df["num_chains"] = [r[2] for r in ret]

        # Write to file using fcntl to avoid race condition
        with open(out_df_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            file_empty = (f.tell() == 0)
            df.to_csv(f, index=False, header=file_empty)
            fcntl.flock(f, fcntl.LOCK_UN)

    # Get the full filtered pdb_keys CSV
    get_pdb_keys(pdb_keys_df, Path(filtered_mmcif_dir), save_dir)


if __name__ == "__main__":
    typer.run(multiprocess_runner)
