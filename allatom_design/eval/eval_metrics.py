import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import Bio
import numpy as np
import pandas as pd
import torch
from Bio.PDB.DSSP import DSSP
from colabdesign.af import mk_af_model
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer

import allatom_design.data.residue_constants as rc
from allatom_design.data import data
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import load_feats_from_pdb
from allatom_design.data.pdb_utils import write_batched_to_pdb, write_to_pdb
from allatom_design.eval import eval_metrics
from allatom_design.eval.folding_utils import (run_af2, run_esmfold,
                                               run_esmfold_batched)
from allatom_design.eval.proteinmpnn_utils import run_mpnn
from ligandmpnn.model_utils import ProteinMPNN


def compute_secondary_structure_content(pdbs: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Given a list of PDBs, compute the secondary structure content of each protein.
    Returns a dict mapping from the PDB to a dict containing:
    - pct_alpha: the proportion of residues that are in alpha helices
    - pct_beta: the proportion of residues that are in beta sheets
    """
    dssp_metrics = defaultdict(dict)
    for pdb in pdbs:
        try:
            dssp_str = get_3state_dssp(pdb)
            dssp_metrics[pdb]["pct_alpha"] = np.mean([c == "H" for c in dssp_str]) * 100
            dssp_metrics[pdb]["pct_beta"] = np.mean([c == "E" for c in dssp_str]) * 100
        except Exception:
            dssp_metrics[pdb]["pct_alpha"] = np.nan
            dssp_metrics[pdb]["pct_beta"] = np.nan
    return dssp_metrics


def get_3state_dssp(pdb: str) -> str:
    """
    Given a PDB file, return the DSSP string for the protein, with 3 states: H, E, L.
    """
    dssp_string = get_dssp_string(pdb)
    dssp_string = pool_dssp_symbols(dssp_string, newchar="L", chars=["-", "T", "S", "C", " "])
    dssp_string = pool_dssp_symbols(dssp_string, newchar="H", chars=["H", "G", "I"])
    dssp_string = pool_dssp_symbols(dssp_string, newchar="E", chars=["E", "B"])
    return dssp_string


def pool_dssp_symbols(dssp_string: str,
                      newchar: str,
                      chars: List[str]) -> str:
    """Replaces all instances of chars with newchar. DSSP chars are helix=GHI, strand=EB, loop=- TSC"""
    string_out = dssp_string
    for c in chars:
        string_out = string_out.replace(c, newchar)
    return string_out


def get_dssp_string(pdb: str) -> str:
    """
    Given a PDB file, return the DSSP string for the protein.
    """
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(Path(pdb).stem, pdb)
    dssp = DSSP(structure[0], pdb, dssp="mkdssp")
    dssp_string = "".join([dssp[k][2] for k in dssp.keys()])
    return dssp_string


def run_self_consistency_eval(pdbs: List[str],
                              mpnn_model: Optional[ProteinMPNN],
                              mpnn_cfg: Optional[DictConfig],
                              struct_pred_model: str,  # "af2" or "esmfold"
                              esmfold: Optional[EsmForProteinFolding],
                              esm_tokenizer: Optional[EsmTokenizer],
                              af_model: Optional[mk_af_model],
                              af2_cfg: Optional[DictConfig],
                              device: torch.device,
                              out_dir: str,
                              eval_codesign: bool = False,
                              max_tokens_per_batch: int = 1024,  # for ESMFold
                              ) -> Dict[str, Dict[str, TensorType]]:
    """
    Run self-consistency evaluation on a list of PDBs (MPNN -> AF2 / ESMFold -> eval metrics).

    The number of MPNN sequences per PDB is determined by the mpnn_cfg (batch_size * number of batches).

    Returns a dictionary mapping from PDB file path to a dictionary containing:
    - "mpnn_preds": MPNN predictions
    - "struct_preds": ESMFold or AF2 predictions. Contains:
        - "avg_plddt": average plddt-CA score
    - "sc_metrics": Evaluation metrics

    In out_dir, this function will create:
    - out_dir/mpnn_preds: AF2/ESMFold predicted PDBs
    - out_dir/mpnn_ca_aligned_preds: AF2/ESMFold predicted PDBs, CA aligned to the original PDBs

    If eval_codesign is True, rather than use MPNN predictions, the sequences in the original PDBs will be used.
    - In this case, mpnn_model and mpnn_cfg are not required, and "mpnn_preds" will not be included in the output.
    - Also, the out directories will have the prefix "codesign_"

    Args:
    - max_tokens_per_batch: Maximum number of tokens per batch for ESMFold predictions

    TODO: handle multichain residue index gap when reading in MPNN preds / sampled sequences. For ESMFold, gap should be 1000?
    """
    sc_info = defaultdict(dict)

    # Create output directories
    preds_dir = Path(out_dir, f"{'codesign_' if eval_codesign else 'mpnn_'}preds")
    preds_dir.mkdir(parents=True, exist_ok=True)
    ca_aligned_preds_dir = Path(out_dir, f"{'codesign_' if eval_codesign else 'mpnn_'}ca_aligned_preds")
    ca_aligned_preds_dir.mkdir(parents=True, exist_ok=True)

    # === Run MPNN === #
    if not eval_codesign:
        mpnn_preds_dict = run_mpnn(mpnn_model, pdb_paths=pdbs, device=device, cfg=mpnn_cfg)
        for pdb, mpnn_preds in mpnn_preds_dict.items():
            sc_info[pdb]["mpnn_preds"] = mpnn_preds

    # === Run structure prediction === #
    if not eval_codesign:
        # For backbone eval, run structure prediction on MPNN sequences for each PDB
        for pdb in tqdm(pdbs, desc=f"Running {'ESMFold' if struct_pred_model == 'esmfold' else 'AF2'}", leave=False):
            mpnn_preds = sc_info[pdb]["mpnn_preds"]
            sequences_list, residue_index_list = mpnn_preds["mpnn_seqs"], mpnn_preds["residue_index"]

            if struct_pred_model == "af2":
                # === Run AlphaFold2 === #
                af2_preds, filenames = run_af2(sequences_list=sequences_list,
                                               pdbs=[pdb] * len(sequences_list),
                                               af_model=af_model,
                                               out_dir=preds_dir, **af2_cfg)

                # stack all outputs since they are the same length for a given PDB
                af2_preds = {k: torch.stack(v, dim=0) for k, v in af2_preds.items()}
                sc_info[pdb]["struct_preds"] = af2_preds

            elif struct_pred_model == "esmfold":
                # === Run ESMFold === #
                esm_preds = run_esmfold_batched(sequences_list=sequences_list,
                                                residue_index_list=residue_index_list,
                                                model=esmfold,
                                                tokenizer=esm_tokenizer,
                                                max_tokens_per_batch=max_tokens_per_batch,
                                                )
                # stack all outputs since they are the same length for a given PDB
                esm_preds = {k: torch.stack(v, dim=0) for k, v in esm_preds.items()}
                sc_info[pdb]["struct_preds"] = esm_preds

                # Write to pdb file
                feats = {
                    "aatype": esm_preds["aatype"],
                    "atom_positions": esm_preds["pred_coords"],
                    "atom_mask": esm_preds["atom_mask"],
                    "residue_index": esm_preds["residue_index"],
                    "chain_index": torch.zeros_like(esm_preds["residue_index"]),
                    "b_factors": None,
                }

                B, _, _, _ = esm_preds["pred_coords"].shape
                filenames = [f"{preds_dir}/esmfold_{Path(pdb).stem}_{i}.pdb" for i in range(B)]
                write_batched_to_pdb(**feats, filenames=filenames, mode="aa")
    else:
        # For allatom/co-design eval, run ESMFold on sequences directly from PDBs
        sequences_list, residue_index_list = load_sequence_and_residx_from_pdbs(pdbs)
        if struct_pred_model == "af2":
            # === Run AlphaFold2 === #
            af2_preds, filenames = run_af2(sequences_list=sequences_list,
                                           pdbs=pdbs,
                                           af_model=af_model,
                                           out_dir=preds_dir, **af2_cfg)

            # Add to sc_info
            for i, pdb in enumerate(pdbs):
                sc_info[pdb]["sample_seq"] = sequences_list[i]
                sc_info[pdb]["struct_preds"] = {k: v[i][None] for k, v in af2_preds.items()}  # unpack preds and add batch dim

        elif struct_pred_model == "esmfold":
            esm_preds = run_esmfold_batched(sequences_list=sequences_list,
                                            residue_index_list=residue_index_list,
                                            model=esmfold,
                                            tokenizer=esm_tokenizer,
                                            max_tokens_per_batch=max_tokens_per_batch)
            # Write to pdb file
            for i, pdb in enumerate(pdbs):
                sc_info[pdb]["sample_seq"] = sequences_list[i]
                sc_info[pdb]["struct_preds"] = {k: v[i][None] for k, v in esm_preds.items()}  # unpack preds and add batch dim

                feats = {
                    "aatype": esm_preds["aatype"][i],
                    "atom_positions": esm_preds["pred_coords"][i],
                    "atom_mask": esm_preds["atom_mask"][i],
                    "residue_index": esm_preds["residue_index"][i],
                    "chain_index": torch.zeros_like(esm_preds["residue_index"][i]),
                    "b_factors": None,
                }

                filename = f"{preds_dir}/esmfold_{Path(pdb).stem}.pdb"
                write_to_pdb(**feats, filename=filename, mode="aa")

    # === Compute eval metrics === #
    if not eval_codesign:
        metrics_to_compute = ["sc_ca_rmsd", "sc_ca_tm"]
    else:
        metrics_to_compute = ["sc_ca_rmsd", "sc_ca_tm", "sc_aa_rmsd", "sc_aa_tm"]

    Path(ca_aligned_preds_dir).mkdir(parents=True, exist_ok=True)
    for pdb in tqdm(pdbs, desc="Computing metrics", leave=False):
        # Load in sampled structure
        sampled_pdb_feats = data.load_feats_from_pdb(pdb, chain_residx_gap=None)

        # Retrieve structure predictions
        struct_preds = sc_info[pdb]["struct_preds"]

        # Compute structure metrics
        B, _, _, _ = struct_preds["pred_coords"].shape
        metrics, pred_coords_ca_aligned = eval_metrics.compute_structure_metrics(
            struct_preds["pred_coords"],
            sampled_pdb_feats["all_atom_positions"][None].expand(B, -1, -1, -1),
            sampled_pdb_feats["all_atom_mask"][None].expand(B, -1, -1),
            metrics_to_compute=metrics_to_compute
        )
        sc_info[pdb]["sc_metrics"] = metrics

        # Write aligned coords to pdb file
        feats = {
            "aatype": struct_preds["aatype"],
            "atom_positions": pred_coords_ca_aligned,
            "atom_mask": struct_preds["atom_mask"],
            "residue_index": struct_preds["residue_index"],
            "chain_index": torch.zeros_like(struct_preds["residue_index"]),
            "b_factors": None,
        }

        prefix = "esmfold" if struct_pred_model == "esmfold" else "af2"
        if not eval_codesign:
            filenames = [f"{ca_aligned_preds_dir}/{prefix}_{Path(pdb).stem}_{i}.pdb" for i in range(B)]
        else:
            assert B == 1, "We should only have one prediction per PDB for eval_codesign eval"
            filenames = [f"{ca_aligned_preds_dir}/{prefix}_{Path(pdb).stem}.pdb"]
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")

    return sc_info


def load_sequence_and_residx_from_pdbs(pdbs: List[str]) -> Tuple[List[str],
                                                                 List[TensorType["n_s", int]]]:
    examples = [load_feats_from_pdb(pdb, chain_residx_gap=None) for pdb in pdbs]
    aatypes = [example["aatype"] for example in examples]
    sequences_list = ["".join([rc.restypes[x] for x in aatype]) for aatype in aatypes]
    residue_index_list = [example["residue_index"] for example in examples]
    return sequences_list, residue_index_list


def compute_structure_metrics(coords1: TensorType["b n 37 3"],
                              coords2: TensorType["b n 37 3"],
                              atom_mask: TensorType["b n 37"],
                              metrics_to_compute: List[str],
                              ) -> Tuple[Dict[str, float],
                                         TensorType["b n 37 3"]
                                         ]:
    """
    Compute structure metrics between two sets of coordinates. Batched.
    Allatom metrics assume atom_mask is the same between both sets of coordinates.

    - metrics_to_compute: List of metrics to compute. Options are given below.

    Metrics:
    - sc_ca_rmsd: scRMSD between Ca atoms
    - sc_ca_tm: scTM score between Ca atoms
    - sc_aa_tm: TM score between all atoms, aligned on Ca atoms
    - sc_aa_rmsd: RMSD between all atoms, aligned on all atoms
    - scn_rmsd_per_pos: sidechain RMSD per residue, aligned on backbone atoms

    Returns:
    - structure_metrics: Dict of computed metrics
    - ca_aligned_coords1: Coordinates of coords1 aligned on Ca atoms
    """
    # Check inputs, since we can run into broadcasting issues if not they're not batched
    assert len(coords1.shape) == 4, "coords1 must be of shape [b n 37 3]"
    assert len(coords2.shape) == 4, "coords2 must be of shape [b n 37 3]"

    B, N, _, _ = coords1.shape

    structure_metrics = {}

    # Align by Ca atoms
    ca_atom_mask = torch.zeros_like(atom_mask)
    ca_atom_mask[..., 1] = 1
    ca_atom_mask = ca_atom_mask * atom_mask

    ca_rmsd, (ca_aligned_coords1, _) = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                rearrange(coords2, "b n a x -> b (n a) x"),
                                                                weights=rearrange(ca_atom_mask, "b n a -> b (n a)"),
                                                                return_aligned=True)
    ca_aligned_coords1 = rearrange(ca_aligned_coords1, "b (n a) x -> b n a x", n=N)

    # Compute metrics
    for metric in metrics_to_compute:
        if metric == "sc_ca_rmsd":
            structure_metrics["sc_ca_rmsd"] = ca_rmsd
        elif metric == "sc_ca_tm":
            structure_metrics["sc_ca_tm"] = data.tm_score(ca_aligned_coords1[..., 1:2, :],
                                                          coords2[..., 1:2, :],
                                                          mask=ca_atom_mask[..., 1:2])
        elif metric == "sc_aa_tm":
            # Align on Ca, compute allatom TM
            structure_metrics["sc_aa_tm"] = data.tm_score(ca_aligned_coords1, coords2, mask=atom_mask)
        elif metric == "sc_aa_rmsd":
            # Align on all atoms, compute all-atom RMSD
            structure_metrics["sc_aa_rmsd"] = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                       rearrange(coords2, "b n a x -> b (n a) x"),
                                                                       weights=rearrange(atom_mask, "b n a -> b (n a)"))
        elif metric == "scn_rmsd_per_pos":
            # Align on backbone atoms, compute sidechain RMSD

            # align on backbone atoms
            bb_atom_mask = torch.zeros_like(atom_mask)
            bb_atom_mask[..., rc.bb_idxs] = 1
            bb_atom_mask = bb_atom_mask * atom_mask

            bb_rmsd, (bb_aligned_coords1, _) = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                  rearrange(coords2, "b n a x -> b (n a) x"),
                                                                  weights=rearrange(bb_atom_mask, "b n a -> b (n a)"),
                                                                  return_aligned=True)
            bb_aligned_coords1 = rearrange(bb_aligned_coords1, "b (n a) x -> b n a x", n=N)

            # compute RMSD over sidechain atoms per residue
            scn_atom_mask = torch.zeros_like(atom_mask)
            scn_atom_mask[..., rc.non_bb_idxs] = 1
            scn_atom_mask[..., rc.atom_order["CB"]] = 0  # exclude CB atoms to match LigandMPNN eval
            scn_atom_mask = scn_atom_mask * atom_mask
            scn_atom_mask = scn_atom_mask[..., None].expand_as(bb_aligned_coords1)

            scn_rmsd_per_pos = ((scn_atom_mask * (bb_aligned_coords1 - coords2) ** 2).sum(dim=(-1, -2)) / scn_atom_mask.sum(dim=(-1, -2)).clamp(min=1)).sqrt()
            structure_metrics["scn_rmsd_per_pos"] = scn_rmsd_per_pos
        else:
            assert False, f"Invalid metric: {metric}"

    return structure_metrics, ca_aligned_coords1



def get_sort_key_fn(metric_name: str) -> Callable[[float], float]:
    """
    Returns a key function for sorting based on the metric name.
    Taking the max with this key function will give the best score.

    Supported metrics:
    - 'sc_ca_rmsd': min is best
    - 'sc_aa_rmsd': min is best
    - 'sc_ca_tm': max is best
    - 'sc_aa_tm': max is best

    Args:
    - metric_name (str): The name of the metric.

    Returns:
    - function: A key function for sorting.
    """
    if metric_name in ["sc_ca_rmsd", "sc_aa_rmsd"]:
        # Ascending order, min is best
        return lambda x: -x
    elif metric_name in ["sc_ca_tm", "sc_aa_tm"]:
        # Descending order, max is already best
        return lambda x: x
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def run_nntm_eval(pdbs: List[str],
                  dataset: str,
                  out_dir: str,
                  tsv_prefix: str = "",
                  ) -> Dict[str, float]:
    """
    Compute nnTM scores for a set of PDBs against a dataset.

    Returns a dictionary from PDB ID to nnTM score (0 if no match found).

    In out_dir, we will create:
    - out_dir/nntm: directory containing nnTM scores as well as temporary files
    """
    nntm_out = Path(out_dir, "nntm")
    Path(nntm_out).mkdir(parents=True, exist_ok=True)

    foldseek_tsv = Path(nntm_out, f"{tsv_prefix}foldseek_tm_results.tsv")
    temp_dir = Path(nntm_out, "temp")

    command = [
        "foldseek", "easy-search",
        *pdbs, dataset, str(foldseek_tsv), str(temp_dir),
        "--alignment-type", "1",
        "--format-output", "query,target,alntmscore,qtmscore,ttmscore"
    ]
    subprocess.run(command, check=True)

    # Read results and reformat
    foldseek_df = pd.read_csv(foldseek_tsv, sep="\t", names=["query", "target", "align_tm_score", "query_tm_score", "target_tm_score"])
    foldseek_df["query"] = foldseek_df["query"].replace({Path(pdb).stem: pdb for pdb in pdbs})  # add full path back
    foldseek_df.to_csv(foldseek_tsv, sep="\t", index=False)
    pdb_to_nntm = foldseek_df.groupby("query").agg({"query_tm_score": "max"}).to_dict()["query_tm_score"]

    for pdb in pdbs:
        # if no match, set to 0
        pdb_to_nntm[pdb] = pdb_to_nntm.get(pdb, 0.0)

    return pdb_to_nntm

