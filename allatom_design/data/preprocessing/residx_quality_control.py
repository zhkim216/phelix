"""
Iterate over label seq id and auth seq id, detect where gaps in label seq id differs from auth seq id

Also saves a slice of the mmCIF file that matches the domain parsed by Ingraham CATH of only those that match

Author: Tianyu Lu
"""

from functools import partial
import multiprocessing
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Selection, Structure, Model, Chain, Residue
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import FastMMCIFParser
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm


io = PDBIO()
pdb_parser = PDBParser(QUIET=True)
label_seqid_parser = FastMMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)
auth_seqid_parser = FastMMCIFParser(auth_chains=True, auth_residues=True, QUIET=True)

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


def get_pdb_keys(pdb_keys_fp: Path, pdb_store_fp: Path):
    remaining_keys = []
    with open(pdb_keys_fp, 'r') as fp:
        for line in fp.readlines():
            pdb_key = line.strip()
            if (pdb_store_fp / pdb_key).exists():
                remaining_keys.append(pdb_key)
    with open(pdb_store_fp.parent / f"{pdb_keys_fp.stem}{pdb_keys_fp.suffix}", 'w') as fp:
        fp.write('\n'.join(remaining_keys) + '\n')

def group_consecutive_idx(nums):

    nums = np.array(nums)
    breaks = np.where(np.diff(nums) > 1)[0] + 1

    result = np.split(nums, breaks)

    return [sublist.tolist() for sublist in result]


def resid_gap_differs(id1, id2) -> list[int]:
    id1_zero = np.array(id1) - id1[0]
    id2_zero = np.array(id2) - id2[0]
    gap_diffs, gap_idx = [], []
    for i in range(len(id1)):
        if id1_zero[i] != id2_zero[i]:
            gap_diffs.append(np.abs(id1_zero[i] - id2_zero[i]))
            gap_idx.append(id2[i])
            id1_zero = np.array(id1_zero) - id1_zero[i]
            id2_zero = np.array(id2_zero) - id2_zero[i]
    return gap_diffs, gap_idx


def save_residues(
    residues: list[Residue.Residue], save_fp: Path, shift_res: bool = False
):
    """
    Save a list of residues to a PDB structure
    If {shift_res} is True (default False), also shifts the residx to start from 1 (assume monotonically increasing)
    """
    new_structure = Structure.Structure("s")
    new_model = Model.Model(0)
    new_chain = Chain.Chain("A")

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
    mmcif_dir: Path = Path("/scratch/users/tianyulu/datasets/mmCIF"),
    pdb_store_dir: Path = Path(
        "/oak/stanford/groups/possu/tianyu/ingraham_cath_dataset/pdb_store"
    ),
    save_dir: Path = Path(
        "/scratch/users/tianyulu/datasets/ingraham_cath_dataset_label_seqid/pdb_store"
    ),
    shift_res: bool = False,
):
    # def runner(pdb_key: Path, mmcif_dir: Path = Path("/home/pdl/data/test_data")):
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_code = pdb_key[:4]
    model_num = 0
    if len(pdb_code) == 7:
        model_num = int(pdb_code[5:7])
    cif_fp = mmcif_dir / f"{pdb_code[1:3]}/{pdb_code}.cif"
    pdb_fp = pdb_store_dir / pdb_key
    # cif_fp = mmcif_dir / f"{pdb_code}.cif"
    num_chains = 0
    if cif_fp.exists():
        label_seqids, auth_seqids, pdb_store_seqids = [], [], []

        s1 = label_seqid_parser.get_structure("s", cif_fp)[model_num]
        s2 = auth_seqid_parser.get_structure("s", cif_fp)[model_num]
        s3 = pdb_parser.get_structure("s", pdb_fp)[model_num]

        all_chains = []

        cif_residues = []

        for res in Selection.unfold_entities(s1, "R"):
            resname = restype_3to1.get(res.resname, "X")
            if resname != "X":
                chain_id = res.get_parent().id
                all_chains.append(chain_id)
                if chain_id == pdb_key[4]:
                    label_seqids.append(res.id[1])
                    cif_residues.append(res)

        num_chains = len(set(all_chains))

        #* discard structures with Resolution > 3.0A and Rfree > 0.25
        mmcif_dict = MMCIF2Dict(cif_fp)
        try:
            resolution = float(mmcif_dict.get('_refine.ls_d_res_high', [999])[0])
            rfree = float(mmcif_dict.get('_refine.ls_R_factor_R_free', [999])[0])
        except Exception as err:
            return (pdb_key, f"Undefined: Resolution = {mmcif_dict.get('_refine.ls_d_res_high', [999])[0]} and Rfree = {mmcif_dict.get('_refine.ls_R_factor_R_free', [999])[0]}", num_chains)
        if resolution > 3.0 or rfree > 0.25:
            return (pdb_key, f"Low quality: Resolution = {resolution} and Rfree = {rfree}", num_chains)

        for res in Selection.unfold_entities(s2, "R"):
            resname = restype_3to1.get(res.resname, "X")
            if resname != "X" and res.get_parent().id == pdb_key[4]:
                auth_seqids.append(res.id[1])

        for res in Selection.unfold_entities(s3, "R"):
            resname = restype_3to1.get(res.resname, "X")
            if resname != "X" and res.get_parent().id == pdb_key[4]:
                pdb_store_seqids.append(res.id[1])

        # * subset auth_seqids to those that overlap with pdb_store_seqids (those actually present in ingraham_cath_dataset/pdb_store pdbs)
        if len(pdb_store_seqids) > 0:
            pdb_seqid_start = pdb_store_seqids[0]
            pdb_seqid_end = pdb_store_seqids[-1]
            i_start = (
                auth_seqids.index(pdb_seqid_start)
                if pdb_seqid_start in auth_seqids
                else None
            )
            i_end = (
                auth_seqids.index(pdb_seqid_end)
                if pdb_seqid_end in auth_seqids
                else None
            )
            if i_start is not None and i_end is not None:
                auth_seqids = auth_seqids[i_start : i_end + 1]
                label_seqids = label_seqids[i_start : i_end + 1]
                cif_residues = cif_residues[i_start : i_end + 1]

                # * save the subsetted cif residues
                save_residues(cif_residues, save_dir / pdb_key, shift_res=shift_res)
            else:
                return (pdb_key, f"pdb_seqid not found in .cif file", num_chains)
        else:
            return (pdb_key, f"Empty file pdb_store", num_chains)

        if len(label_seqids) != len(auth_seqids):
            print(label_seqids)
            print(auth_seqids)
            return (pdb_key, "Different lengths", num_chains)

        try:
            gap_diffs, gap_idx = resid_gap_differs(label_seqids, auth_seqids)
        except Exception as err:
            return (pdb_key, err, num_chains)
        if gap_diffs:
            diffs = ",".join([str(gd) for gd in gap_diffs])
            residx = ",".join([str(gi) for gi in gap_idx])
            return (
                pdb_key,
                f"Gaps differ by {diffs} at auth_seq_ids {residx}",
                num_chains,
            )

        num_gaps = len(group_consecutive_idx(label_seqids)) - 1

        return (pdb_key, f"Has {num_gaps} gaps, all matches auth_seq_id", num_chains)

    return (pdb_key, "No MMCIF Found", num_chains)


def multiprocess_runner(
    pdb_keys: Path,
    mmcif_dir: Path = Path("/scratch/users/tianyulu/datasets/mmCIF"),
    pdb_store_dir: Path = Path("/oak/stanford/groups/possu/tianyu/ingraham_cath_dataset/pdb_store"),
    max_threads: int = 8,
    save_dir: Path = Path(
        "/scratch/users/tianyulu/datasets/ingraham_cath_dataset_label_seqid/pdb_store"
    ),
    shift_res: bool = False,
):
    all_pdb_keys = []
    with open(pdb_keys, "r") as fp:
        for line in fp.readlines():
            all_pdb_keys.append(line.strip())

    with multiprocessing.Pool(max_threads) as p:
        ret = list(tqdm(
            p.imap(
                partial(runner, mmcif_dir=mmcif_dir, pdb_store_dir=pdb_store_dir, save_dir=save_dir, shift_res=shift_res),
                all_pdb_keys
            ),
            total=len(all_pdb_keys),
            desc="Processing PDBs"
        ))

    df = pd.DataFrame()
    df["pdb"] = [r[0] for r in ret]
    df["problem"] = [r[1] for r in ret]
    df["num_chains"] = [r[2] for r in ret]
    df.to_csv(f"{Path(save_dir).parent}/problematic_pdbs_{pdb_keys.stem}.csv", index=False)

    # Save the filtered pdb_keys list to the parent of the save_dir
    get_pdb_keys(pdb_keys, save_dir)


if __name__ == "__main__":
    typer.run(multiprocess_runner)
