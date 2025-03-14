import math
import shutil
import subprocess
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from einops import rearrange
from omegaconf import DictConfig
from scipy import linalg
from scipy.stats import entropy
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.residue_constants as rc
from allatom_design.data import data
from allatom_design.data.data import load_feats_from_pdb
from allatom_design.data.pdb_utils import write_batched_to_pdb, write_to_pdb
from allatom_design.eval import eval_metrics
from allatom_design.eval.dssp_utils import annotate_sse, pdb_to_xyz
from allatom_design.eval.fampnn_utils import run_fampnn
from allatom_design.eval.folding_utils import (run_af2, run_esmfold_batched,
                                               run_omegafold)
from allatom_design.eval.proteinmpnn_utils import run_mpnn
from ligandmpnn.model_utils import ProteinMPNN


def compute_secondary_structure_content(pdbs: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Given a list of PDBs, compute the secondary structure content of each protein using the new method.
    Returns a dict mapping from the PDB to a dict containing:
    - pct_alpha: the proportion of residues that are in alpha helices
    - pct_beta: the proportion of residues that are in beta sheets
    """
    dssp_metrics = defaultdict(dict)
    parser = PDBParser()
    for pdb in tqdm(pdbs, desc="Computing secondary structure content"):
        try:
            structure = parser.get_structure("s", pdb)
            xyz_ca = pdb_to_xyz(structure)
            if len(xyz_ca) == 0:
                raise ValueError("No CA atoms found in the structure.")
            sse = annotate_sse(xyz_ca)
            stats = sse.sum(0) / len(xyz_ca)
            helix = stats[0].item()
            strand = stats[1].item()
            dssp_metrics[pdb]["pct_alpha"] = helix * 100
            dssp_metrics[pdb]["pct_beta"] = strand * 100
            dssp_metrics[pdb]["pct_loop"] = (1 - helix - strand) * 100
        except Exception as e:
            print(f"Error processing {pdb}: {e}")
            dssp_metrics[pdb]["pct_alpha"] = np.nan
            dssp_metrics[pdb]["pct_beta"] = np.nan
            dssp_metrics[pdb]["pct_loop"] = np.nan
    return dssp_metrics


def run_self_consistency_eval(pdbs: List[str],
                              seq_des_model: Dict[str, Any],  # contains sequence design model components
                              struct_pred_model: Dict[str, Any],  # contains struct pred model components
                              device: torch.device,
                              out_dir: str,
                              eval_codesign: bool = False,
                              temp_dir: Optional[str] = None,
                              override_metrics_to_compute: Optional[List[str]] = None,
                              motif_info: dict = {},  # if evaluating motif scaffolding, maps from PDB path to scaffold coordinates and mask
                              ) -> Dict[str, Dict[str, TensorType]]:
    """
    Run self-consistency evaluation on a list of PDBs (MPNN -> struct_pred -> eval metrics).

    The number of MPNN sequences per PDB is determined by the mpnn_cfg (batch_size * number of batches).

    Returns a dictionary mapping from PDB file path to a dictionary containing:
    - "mpnn_preds": MPNN predictions
    - "struct_preds": structure predictions. Contains:
        - "avg_plddt": average plddt-CA score
    - "sc_metrics": Evaluation metrics

    In out_dir, this function will create:
    - out_dir/mpnn_preds: structure predicted PDBs
    - out_dir/mpnn_ca_aligned_preds: structure predicted PDBs, CA aligned to the original PDBs

    If eval_codesign is True, rather than use MPNN predictions, the sequences in the original PDBs will be used.
    - In this case, mpnn_model and mpnn_cfg are not required, and "mpnn_preds" will not be included in the output.
    - Also, the out directories will have the prefix "codesign_"

    TODO: handle multichain residue index gap when reading in MPNN preds / sampled sequences. For ESMFold, gap should be 1000?
    """
    sc_info = defaultdict(dict)

    # Set up struct pred model
    struct_pred_cfg = struct_pred_model["cfg"]
    struct_model_name = struct_pred_model["model_name"]

    # Create output directories
    preds_dir = Path(out_dir, f"{'codesign_' if eval_codesign else 'mpnn_'}preds")
    preds_dir.mkdir(parents=True, exist_ok=True)
    ca_aligned_preds_dir = Path(out_dir, f"{'codesign_' if eval_codesign else 'mpnn_'}ca_aligned_preds")
    ca_aligned_preds_dir.mkdir(parents=True, exist_ok=True)

    # === Run sequence design === #
    if not eval_codesign:
        seq_des_model_name = seq_des_model["model_name"]
        if seq_des_model_name == "proteinmpnn":
            mpnn_model, mpnn_cfg = seq_des_model["mpnn_model"], seq_des_model["mpnn_cfg"]
            mpnn_preds_dict = run_mpnn(mpnn_model, cfg=mpnn_cfg, pdb_paths=pdbs, device=device)
            for pdb, mpnn_preds in mpnn_preds_dict.items():
                sc_info[pdb]["mpnn_preds"] = mpnn_preds
        elif seq_des_model_name == "fampnn":
            fampnn_model, fampnn_cfg = seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"]
            fampnn_preds_dict, _ = run_fampnn(fampnn_model, cfg=fampnn_cfg, pdb_paths=pdbs, device=device)
            for pdb, fampnn_preds in fampnn_preds_dict.items():
                sc_info[pdb]["fampnn_preds"] = fampnn_preds

    # === Run structure prediction === #
    if not eval_codesign:
        # For backbone eval, run structure prediction on the designed sequences for each PDB
        for pdb in tqdm(pdbs, desc=f"Running {struct_model_name}", leave=False):
            # Extract sequences
            if seq_des_model_name == "proteinmpnn":
                mpnn_preds = sc_info[pdb]["mpnn_preds"]
                sequences_list, residue_index_list, chain_index_list = mpnn_preds["mpnn_seqs"], mpnn_preds["residue_index"], mpnn_preds["chain_index"]
            elif seq_des_model_name == "fampnn":
                fampnn_preds = sc_info[pdb]["fampnn_preds"]
                sequences_list, residue_index_list, chain_index_list = fampnn_preds["pred_seqs"], fampnn_preds["residue_index"], fampnn_preds["chain_index"]

            if struct_model_name == "af2":
                # === Run AlphaFold2 === #
                af2_preds, filenames = run_af2(sequences_list=sequences_list,
                                               residue_index_list=residue_index_list,
                                               chain_index_list=chain_index_list,
                                               pdbs=[pdb] * len(sequences_list),
                                               af_model=struct_pred_model["af_model"],
                                               out_dir=preds_dir, **struct_pred_cfg.af2)

                # stack all outputs since they are the same length for a given PDB
                af2_preds = {k: torch.stack(v, dim=0) for k, v in af2_preds.items()}
                sc_info[pdb]["struct_preds"] = af2_preds

            elif struct_model_name == "esmfold":
                # === Run ESMFold === #
                esm_preds = run_esmfold_batched(sequences_list=sequences_list,
                                                residue_index_list=residue_index_list,
                                                chain_index_list=chain_index_list,
                                                model=struct_pred_model["esmfold"],
                                                tokenizer=struct_pred_model["tokenizer"],
                                                max_tokens_per_batch=struct_pred_cfg.esmfold.max_tokens_per_batch,
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

            elif struct_model_name == "omegafold":
                # === Run OmegaFold === #
                of_preds = run_omegafold(sequences_list=sequences_list,
                                         residue_index_list=residue_index_list,
                                         omegafold_model=struct_pred_model["omegafold"],
                                         out_dir=preds_dir, device=struct_pred_model["device"], **struct_pred_cfg.omegafold)

                # stack all outputs since they are the same length for a given PDB
                of_preds = {k: torch.stack(v, dim=0) for k, v in of_preds.items()}
                sc_info[pdb]["struct_preds"] = of_preds

                # Write to pdb file
                feats = {
                    "aatype": of_preds["aatype"],
                    "atom_positions": of_preds["pred_coords"],
                    "atom_mask": of_preds["atom_mask"],
                    "residue_index": of_preds["residue_index"],
                    "chain_index": torch.zeros_like(of_preds["residue_index"]),
                    "b_factors": None,
                }

                B, _, _, _ = of_preds["pred_coords"].shape
                filenames = [f"{preds_dir}/omegafold_{Path(pdb).stem}_{i}.pdb" for i in range(B)]
                write_batched_to_pdb(**feats, filenames=filenames, mode="aa")

    else:
        # For allatom/co-design eval, run ESMFold on sequences directly from PDBs
        sequences_list, residue_index_list, chain_index_list = load_sequence_and_residx_from_pdbs(pdbs)
        if struct_model_name == "af2":
            # === Run AlphaFold2 === #
            af2_preds, filenames = run_af2(sequences_list=sequences_list,
                                           residue_index_list=residue_index_list,
                                           chain_index_list=chain_index_list,
                                           pdbs=pdbs,
                                           af_model=struct_pred_model["af_model"],
                                           out_dir=preds_dir, **struct_pred_cfg.af2)

            # Add to sc_info
            for i, pdb in enumerate(pdbs):
                sc_info[pdb]["sample_seq"] = sequences_list[i]
                sc_info[pdb]["struct_preds"] = {k: v[i][None] for k, v in af2_preds.items()}  # unpack preds and add batch dim

        elif struct_model_name == "esmfold":
            # === Run ESMFold === #
            esm_preds = run_esmfold_batched(sequences_list=sequences_list,
                                            residue_index_list=residue_index_list,
                                            chain_index_list=chain_index_list,
                                            model=struct_pred_model["esmfold"],
                                            tokenizer=struct_pred_model["tokenizer"],
                                            max_tokens_per_batch=struct_pred_cfg.esmfold.max_tokens_per_batch)
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

        elif struct_model_name == "omegafold":
            # === Run OmegaFold === #
            of_preds = run_omegafold(sequences_list=sequences_list,
                                     residue_index_list=residue_index_list,
                                     omegafold_model=struct_pred_model["omegafold"],
                                     out_dir=preds_dir, device=struct_pred_model["device"], **struct_pred_cfg.omegafold)

            # Write to pdb file
            for i, pdb in enumerate(pdbs):
                sc_info[pdb]["sample_seq"] = sequences_list[i]
                sc_info[pdb]["struct_preds"] = {k: v[i][None] for k, v in of_preds.items()}  # unpack preds and add batch dim

                feats = {
                    "aatype": of_preds["aatype"][i],
                    "atom_positions": of_preds["pred_coords"][i],
                    "atom_mask": of_preds["atom_mask"][i],
                    "residue_index": of_preds["residue_index"][i],
                    "chain_index": torch.zeros_like(of_preds["residue_index"][i]),
                    "b_factors": None,
                }
                filename = f"{preds_dir}/omegafold_{Path(pdb).stem}.pdb"
                write_to_pdb(**feats, filename=filename, mode="aa")

    # === Compute eval metrics === #
    if not eval_codesign:
        metrics_to_compute = ["sc_ca_rmsd", "sc_ca_tm"]
    else:
        metrics_to_compute = ["sc_ca_rmsd", "sc_ca_tm", "sc_aa_rmsd"]

    if override_metrics_to_compute is not None:
        metrics_to_compute = override_metrics_to_compute

    Path(ca_aligned_preds_dir).mkdir(parents=True, exist_ok=True)
    for pdb in tqdm(pdbs, desc="Computing metrics", leave=False):
        # Load in sampled structure
        sampled_pdb_feats = data.load_feats_from_pdb(pdb)

        # Retrieve structure predictions
        struct_preds = sc_info[pdb]["struct_preds"]

        # Compute structure metrics
        B, _, _, _ = struct_preds["pred_coords"].shape
        metrics, pred_coords_ca_aligned = eval_metrics.compute_structure_metrics(
            struct_preds["pred_coords"],
            sampled_pdb_feats["all_atom_positions"][None].expand(B, -1, -1, -1),
            sampled_pdb_feats["all_atom_mask"][None].expand(B, -1, -1),
            metrics_to_compute=metrics_to_compute,
            temp_dir=temp_dir,
            motif_info=motif_info.get(pdb, {}),  # if evaluating motif scaffolding, pass in scaffold coordinates and mask
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

        if not eval_codesign:
            filenames = [f"{ca_aligned_preds_dir}/{struct_model_name}_{Path(pdb).stem}_{i}.pdb" for i in range(B)]
        else:
            assert B == 1, "We should only have one prediction per PDB for eval_codesign eval"
            filenames = [f"{ca_aligned_preds_dir}/{struct_model_name}_{Path(pdb).stem}.pdb"]
        write_batched_to_pdb(**feats, filenames=filenames, mode="aa")

    return sc_info


def load_sequence_and_residx_from_pdbs(pdbs: List[str]) -> Tuple[List[str],
                                                                 List[TensorType["n_s", int]],
                                                                 List[TensorType["n_s", int]]]:
    examples = [load_feats_from_pdb(pdb) for pdb in pdbs]
    aatypes = [example["aatype"] for example in examples]
    sequences_list = ["".join([rc.restypes_with_x[x] for x in aatype]) for aatype in aatypes]
    residue_index_list = [example["residue_index"] for example in examples]
    chain_index_list = [example["chain_index"] for example in examples]
    return sequences_list, residue_index_list, chain_index_list


def compute_pairwise_tm_score(coords_list: List[TensorType["n 37 3"]],
                              temp_dir: str,
                              subsample_pairs: Optional[int] = None) -> float:
    """
    Compute the mean CA TM-align -> TM-score among all pairwise comparisons in a batch of structures.

    Averages over the TM-score normalized over either structure, then averages over b(b-1)/2 pairs of structures.
    """
    B = len(coords_list)
    if B == 1:
        return 0.0

    # First, parse coords_list into a tensor with seq_mask for padding
    seq_mask = [torch.ones_like(c[..., 0, 0]) for c in coords_list]
    max_length = max([c.shape[0] for c in coords_list])
    coords = torch.stack([data.make_fixed_size_1d(c, fixed_size=max_length, start_idx=None) for c in coords_list], dim=0)
    seq_mask = torch.stack([data.make_fixed_size_1d(m, fixed_size=max_length, start_idx=None) for m in seq_mask], dim=0)

    # Get all pairs of structures
    pairs = torch.combinations(torch.arange(B), r=2, with_replacement=False)
    if subsample_pairs is not None:
        # randomly subsample pairs
        pairs = pairs[torch.randperm(pairs.shape[0])[:subsample_pairs]]

    i_idxs = pairs[:, 0]  # [num_pairs]
    j_idxs = pairs[:, 1]  # [num_pairs]

    # Extract CA coordinates and atom_mask
    coords_a = coords[i_idxs, :, 1]  # [num_pairs, N, 3]
    coords_b = coords[j_idxs, :, 1]  # [num_pairs, N, 3]
    atom_mask_a = seq_mask[i_idxs, :]  # [num_pairs, N]
    atom_mask_b = seq_mask[j_idxs, :]  # [num_pairs, N]

    # Get TM scores and average
    try:
        pairwise_tm_scores = run_tm_align_coords_batch(coords_a, coords_b,
                                                       atom_mask_a, atom_mask_b,
                                                       temp_dir=temp_dir)
    except Exception as e:
        print(f"Error in compute_pairwise_tm_score: {e}")
        return np.nan
    pairwise_tm_scores = (pairwise_tm_scores[0] + pairwise_tm_scores[1]) / 2  # average TM-score normalized over both structures
    return pairwise_tm_scores.mean().item()


def compute_structure_metrics(coords1: TensorType["b n 37 3"],
                              coords2: TensorType["b n 37 3"],
                              atom_mask: TensorType["b n 37"],
                              metrics_to_compute: List[str],
                              **kwargs,
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
    - sc_aa_rmsd: RMSD between all atoms, aligned on all atoms
    - scn_rmsd_per_pos: sidechain RMSD per residue, aligned on backbone atoms

    If using tm score metrics, kwargs must include:
    - temp_dir: str, to store temporary files

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
            structure_metrics["sc_ca_tm"] = run_tm_align_coords_batch(coords1[..., 1, :], coords2[..., 1, :],
                                                                      ca_atom_mask[..., 1], ca_atom_mask[..., 1],
                                                                      temp_dir=kwargs["temp_dir"])
            structure_metrics["sc_ca_tm"] = structure_metrics["sc_ca_tm"][0]  # get TM-score normalized to coords1 length
        elif metric == "sc_aa_rmsd":
            # Align on all atoms, compute all-atom RMSD
            structure_metrics["sc_aa_rmsd"] = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                       rearrange(coords2, "b n a x -> b (n a) x"),
                                                                       weights=rearrange(atom_mask, "b n a -> b (n a)"))
        elif metric == "motif_bb_rmsd":
            # Align on motif N,CA,C atoms, compute RMSD between the input motif and the predicted motif
            motif_mask = kwargs.get("motif_info", {}).get("motif_mask")  # [b n 37]

            if motif_mask is None:
                raise ValueError("motif_bb_rmsd requires motif_mask in kwargs['motif_info']")

            # Extract N,CA,C mask for the motif
            bb_motif_atom_mask = torch.zeros_like(atom_mask)
            atom_indices = [rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]]  # Zheng et al. MotifBench only uses N, CA, C
            bb_motif_atom_mask[..., atom_indices] = 1
            bb_motif_atom_mask = bb_motif_atom_mask * motif_mask  # [b n 37]

            if bb_motif_atom_mask.sum() == 0:
                structure_metrics["motif_bb_rmsd"] = torch.tensor(np.nan)[None].expand(B)  # no motif atoms to align
                continue

            # Align on motif backbone atoms
            structure_metrics["motif_bb_rmsd"] = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                          rearrange(coords2, "b n a x -> b (n a) x"),
                                                                          weights=rearrange(bb_motif_atom_mask, "b n a -> b (n a)"))

        elif metric.startswith("scn_rmsd_per_pos"):
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
            if metric == "scn_rmsd_per_pos_ligandmpnn":
                # exclude CB atoms to match LigandMPNN eval
                scn_atom_mask[..., rc.atom_order["CB"]] = 0
            scn_atom_mask = scn_atom_mask * atom_mask
            scn_rmsd_per_pos = ((scn_atom_mask[..., None] * (bb_aligned_coords1 - coords2) ** 2).sum(dim=(-1, -2)) / scn_atom_mask.sum(dim=-1).clamp(min=1)).sqrt()

            structure_metrics[metric] = scn_rmsd_per_pos
        elif metric == "sce":
            # Align on backbone atoms, compute sidechain error

            # align on backbone atoms
            bb_atom_mask = torch.zeros_like(atom_mask)
            bb_atom_mask[..., rc.bb_idxs] = 1
            bb_atom_mask = bb_atom_mask * atom_mask

            bb_rmsd, (bb_aligned_coords1, _) = data.torch_rmsd_weighted(rearrange(coords1, "b n a x -> b (n a) x"),
                                                                        rearrange(coords2, "b n a x -> b (n a) x"),
                                                                        weights=rearrange(bb_atom_mask, "b n a -> b (n a)"),
                                                                        return_aligned=True)
            bb_aligned_coords1 = rearrange(bb_aligned_coords1, "b (n a) x -> b n a x", n=N)

            # compute sidechain error
            scn_atom_mask = torch.zeros_like(atom_mask)
            scn_atom_mask[..., rc.non_bb_idxs] = 1
            scn_atom_mask = scn_atom_mask * atom_mask
            sce = torch.where(scn_atom_mask.bool(), torch.norm(bb_aligned_coords1 - coords2, dim=-1), np.nan)  # nan for backbone or missing atoms
            structure_metrics["sce"] = sce[..., rc.non_bb_idxs]

        elif metric == "chi_metrics_per_pos":
            # Compute metrics for sidechain chi angles
            aatype = kwargs["aatype"]

            # Get chi angles in radians
            torsions1, alt_torsions1, torsions_mask1 = data.atom37_to_torsions_rad(aatype, coords1, atom_mask)
            torsions2, alt_torsions2, torsions_mask2 = data.atom37_to_torsions_rad(aatype, coords2, atom_mask)

            # Compute chi angle MAE and accuracy per residue
            chi_metrics_per_pos = metrics_per_chi_per_pos(torsions1[..., 3:], torsions2[..., 3:], alt_torsions2[..., 3:], torsions_mask1[..., 3:])
            structure_metrics["chi_mae_per_pos"] = chi_metrics_per_pos["chi_mae"]
            structure_metrics["chi_acc_per_pos"] = chi_metrics_per_pos["chi_acc"]
            structure_metrics["chi_mask"] = chi_metrics_per_pos["chi_mask"]
        else:
            assert False, f"Invalid metric: {metric}"

    return structure_metrics, ca_aligned_coords1

##########################################
# Adapated from FlowPacker https://gitlab.com/mjslee0921/flowpacker/-/blob/main/utils/metrics.py?ref_type=heads
def angle_ae(pred, target):
    ae = torch.abs(pred - target)
    ae_alt = torch.abs(ae - 2*math.pi)
    ae_min = torch.minimum(ae, ae_alt)
    return ae_min

def angle_mae(pred, target, target_alt, mask, deg=True):
    ae = angle_ae(pred, target)
    ae_alt = angle_ae(pred, target_alt)
    ae_min = torch.minimum(ae, ae_alt)
    mae = ((ae_min*mask).sum() / mask.sum())
    if deg:
        return mae * 180 / math.pi
    return mae

def angle_acc(pred, target, target_alt, mask, threshold=20):
    ae = angle_ae(pred, target)
    ae_alt = angle_ae(pred, target_alt)
    ae_min = torch.minimum(ae, ae_alt)
    acc = torch.logical_and(ae_min <= (threshold * math.pi / 180), mask == 1).sum() / mask.sum()
    return acc

def metrics_per_chi(pred, target, target_alt, chi_mask, threshold=20, deg=True):
    mae_d, acc_d = {}, {}
    for i in range(4):
        mae = angle_mae(pred[..., i], target[...,i], target_alt[...,i], chi_mask[...,i], deg=deg)
        acc = angle_acc(pred[..., i], target[...,i], target_alt[...,i], chi_mask[...,i], threshold=threshold)
        mae_d[f'chi{i+1}'] = mae.item()
        acc_d[f'chi{i+1}'] = acc.item()
    return mae_d, acc_d


def metrics_per_chi_per_pos(pred, target, target_alt, chi_mask, threshold=20):
    mae_d, acc_d = {}, {}
    ae = angle_ae(pred, target)
    ae_alt = angle_ae(pred, target_alt)
    ae_min = torch.minimum(ae, ae_alt) * chi_mask
    ae_min = ae_min * 180 / math.pi
    acc = (ae_min <= threshold) * chi_mask

    chi_metrics_per_pos = {"chi_mae": ae_min, "chi_acc": acc, "chi_mask": chi_mask}
    return chi_metrics_per_pos

##########################################

def get_sort_key_fn(metric_name: str) -> Callable[[float], float]:
    """
    Returns a key function for sorting based on the metric name.
    Taking the max with this key function will give the best score.

    Supported metrics:
    - 'sc_ca_rmsd': min is best
    - 'sc_aa_rmsd': min is best
    - 'sc_ca_tm': max is best

    Args:
    - metric_name (str): The name of the metric.

    Returns:
    - function: A key function for sorting.
    """
    if metric_name in ["sc_ca_rmsd", "sc_aa_rmsd", "motif_bb_rmsd"]:
        # Ascending order, min is best
        return lambda x: -x
    elif metric_name in ["sc_ca_tm"]:
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

    try:
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

    except subprocess.CalledProcessError as e:
        print(f"Error running foldseek: {e}")
        pdb_to_nntm = {pdb: np.nan for pdb in pdbs}

    return pdb_to_nntm


def run_tmalign(pdb_a: str, pdb_b: str) -> Tuple[float, float]:
    """
    Runs TM-align between two PDB files and returns the TM-scores.
    """
    cmd = ["TMalign", pdb_a, pdb_b]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    tm_score_a = None
    tm_score_b = None

    # Parse TM-align output
    for line in output.splitlines():
        if line.startswith("TM-score="):
            parts = line.strip().split()
            score = float(parts[1])
            if "Chain_1" in line:
                tm_score_a = score
            elif "Chain_2" in line:
                tm_score_b = score

    tm_score_a = tm_score_a if tm_score_a is not None else 0.0
    tm_score_b = tm_score_b if tm_score_b is not None else 0.0

    return tm_score_a, tm_score_b


def run_tm_align_coords_batch(a: TensorType["b n 3", float],
                              b: TensorType["b n 3", float],
                              mask_a: TensorType["b n", float],
                              mask_b: TensorType["b n", float],
                              temp_dir: str) -> Tuple[TensorType["b", float], TensorType["b", float]]:
    """
    Given a batch of CA-only atom coordinates, aligns a to b and computes TM-score in parallel.

    Assumes residue_index starts at 0 and is contiguous, and returns TM-score normalized by lengths (a, b).
    """
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    B, N, _ = a.shape

    unique_id = uuid.uuid4().hex  # unique ID for this batch

    # Make sure tensors are on cpu to avoid CUDA issues in threads
    a = a.cpu()
    b = b.cpu()
    mask_a = mask_a.cpu()
    mask_b = mask_b.cpu()

    # Write to temp files
    for prefix, x, mask in zip(["a", "b"], [a, b], [mask_a, mask_b]):
        atom37_positions = torch.zeros((B, N, 37, 3), dtype=a.dtype)
        atom37_positions[..., 1, :] = x
        atom37_mask = torch.zeros((B, N, 37), dtype=mask.dtype)
        atom37_mask[..., 1] = mask

        feats = {
            "aatype": torch.full_like(mask, fill_value=rc.restype_order["G"]).long(),
            "atom_positions": atom37_positions,
            "atom_mask": atom37_mask,
            "residue_index": torch.arange(N, dtype=torch.int64).unsqueeze(0).expand(B, -1),
            "chain_index": torch.zeros((B, N), dtype=torch.int64),
            "b_factors": None,
            "filenames": [f"{temp_dir}/{prefix}_{unique_id}_{i}.pdb" for i in range(B)],
        }
        write_batched_to_pdb(**feats)

    pdb_pairs = [(f"{temp_dir}/a_{unique_id}_{i}.pdb", f"{temp_dir}/b_{unique_id}_{i}.pdb") for i in range(B)]

    # Run TM-align in parallel
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(lambda pair: run_tmalign(*pair), pdb_pairs),
            total=B, desc="Running TM-align", leave=False
        ))

    # Extract TM-scores
    tm_scores_a, tm_scores_b = zip(*results)
    tm_scores_a = torch.tensor(tm_scores_a, dtype=a.dtype)
    tm_scores_b = torch.tensor(tm_scores_b, dtype=a.dtype)

    # Clean up
    for prefix in ["a", "b"]:
        for i in range(B):
            Path(f"{temp_dir}/{prefix}_{i}.pdb").unlink(missing_ok=True)

    return tm_scores_a, tm_scores_b


def compute_ss_kl(p_alpha: List[float], p_beta: List[float],
                  q_alpha: List[float], q_beta: List[float],
                  bin_size: int, pseudocount: float) -> float:
    """
    Compute an approximate to the empirical KL-divergence between the ground truth distribution p
    and the sampled distribution q for secondary structure (helix and strand percentages).

    We bin SS into a 2D grid, and only sum over the lower-triangular part of the grid since helix+strand <= 100%.

    Inputs should be in units of percentage (0-100).

    Parameters:
    - p_alpha: helix percentages for the ground truth distribution p.
    - p_beta: strand percentages for the ground truth distribution p.
    - q_alpha: helix percentages for the sampled distribution q.
    - q_beta: strand percentages for the sampled distribution q.
    - bin_size (int, optional): Size of each bin for helix and strand proportions (default: 10).
    - pseudocount (float, optional): Value to add to each histogram bin to avoid zero probabilities (default: 1.0).

    Returns:
    - kl_div (float): The KL divergence value KL(p || q)
    """
    p_alpha, p_beta, q_alpha, q_beta = np.array(p_alpha), np.array(p_beta), np.array(q_alpha), np.array(q_beta)

    # Define histogram bins
    bins = np.arange(0, 101, bin_size)
    if bins[-1] < 100:
        bins = np.append(bins, 100)

    # 2D histogram for ground truth distribution
    p_counts, _, _ = np.histogram2d(p_beta, p_alpha, bins=[bins, bins])
    p_counts = np.flipud(p_counts)  # flip so that 0,0 is in the bottom-left corner
    p_counts += pseudocount
    p_probs = p_counts / p_counts.sum()

    # 2D histogram for sampled distribution
    q_counts, _, _ = np.histogram2d(q_beta, q_alpha, bins=[bins, bins])
    q_counts = np.flipud(q_counts)  # flip so that 0,0 is in the bottom-left corner
    q_counts += pseudocount
    q_probs = q_counts / q_counts.sum()

    # Compute KL divergence on the lower-triangular part of the matrix, including the diagonal
    tril_indices = np.tril_indices(len(bins) - 1)
    p_probs_flat = p_probs[tril_indices]
    q_probs_flat = q_probs[tril_indices]
    kl_div = entropy(p_probs_flat, q_probs_flat)

    return kl_div


def bootstrap_se(data: List[float], n_samples: int) -> float:
    """
    Perform bootstrapping on the provided data to compute the standard error.

    Args:
        data (List[float]): The data to bootstrap.
        num_samples (int): Number of bootstrap samples.

    Returns:
        float: The bootstrapped standard error.
    """
    if len(data) == 0:
        return np.nan

    data_array = np.array(data)
    bootstrap_means = np.empty(n_samples)

    for i in range(n_samples):
        # Resample with replacement
        resampled = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means[i] = np.mean(resampled)

    boot_se = np.std(bootstrap_means, ddof=1)
    return boot_se


def foldseek_cluster(pdbs: List[str],
                     out_dir: str,
                     temp_dir: str,
                     alignment_type: int,
                     tmscore_threshold: float = 0.6,
                     c: float = 0.8,
                     s: float = 4.0,
                     cluster_reassign: bool = False) -> int:
    """
    Cluster a list of PDBs using Foldseek's easy-cluster command.

    Args:
        pdbs (List[str]): List of PDB files to cluster.
        out_dir (str): Directory to save clustering results.
        alignment-type (int): How to compute the alignment:
            - 0: 3di alignment  (for structure-only / backbone-only)
            - 1: TM alignment
            - 2: 3Di+AA [2]

        tmscore_threshold (float): TM-score threshold for clustering.
        c (float, optional): Fraction of aligned residues required for a match. Defaults to 0.8.
        s (float, optional): Sensitivity level. Defaults to 4.0.
        cluster_reassign (bool, optional): Reassign clusters to correct criteria violations. Defaults to False.

    Returns:
        int: Number of unique clusters.
    """
    if len(pdbs) == 0:
        return 0

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Copy over PDB files to output directory
    pdb_dir = f"{out_dir}/designable_pdbs"
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)
    for pdb in pdbs:
        shutil.copy(pdb, pdb_dir)

    # Run Foldseek clustering
    command = ["foldseek", "easy-cluster",
               "--alignment-type", str(alignment_type),
               *pdbs, f"{out_dir}/foldseek", temp_dir,
               "-c", str(c),
               "--tmscore-threshold", str(tmscore_threshold),
               "-s", str(s)]

    if cluster_reassign:
        command.append("--cluster-reassign")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Foldseek clustering failed with error: {e}")
        return np.nan

    # Read number of unique clusters
    tsv_path = f"{out_dir}/foldseek_cluster.tsv"
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['representative', 'member'])
    num_unique_clusters = df['representative'].nunique()

    return num_unique_clusters


def get_core_surface_mask(coords: TensorType["b n 37 3", float],
                          atom_mask: TensorType["b n 37", float],
                          ) -> Tuple[TensorType["b n", bool], TensorType["b n", bool]]:
    """
    Get a mask for core and surface residues based on the coordinates of a protein, possibly batched.

    Core is fined as residues with at least 20CB atoms within 10A, and surface is defined as residues with at most 15CB atoms within 10A.

    Adapted from FlowPacker: https://gitlab.com/mjslee0921/flowpacker/-/blob/main/sampler_pdb.py?ref_type=heads#L126
    """
    input_coords_shape = coords.shape
    if (len(input_coords_shape) == 3) and (len(atom_mask.shape) == 2):
        # expand to batch dimension
        coords = coords.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)

    assert len(coords.shape) == 4 and len(atom_mask.shape) == 3
    cb_idx = rc.atom_order["CB"]
    cb_exists = atom_mask[:, :, cb_idx]
    cb = coords[:, :, cb_idx, :]

    cb_dist = torch.cdist(cb, cb)
    cb_exists_2d = cb_exists.unsqueeze(-1) * cb_exists.unsqueeze(-2)
    cb_exists_2d = torch.where(torch.eye(cb_exists_2d.shape[-1], device=cb.device).bool(), 0, cb_exists_2d)  # remove diagonal

    cb_dist_w10 = ((cb_dist < 10) * cb_exists_2d).sum(-1)
    core = cb_dist_w10 >= 20
    surface = cb_dist_w10 <= 15

    if len(input_coords_shape) == 3:
        # remove batch dimension if input was not batched
        core = core.squeeze(0)
        surface = surface.squeeze(0)

    return core, surface


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def fpd(
    samp_embeds: np.ndarray,
    gt_embeds: np.ndarray = None,
    gt_mu: np.ndarray = None,
    gt_sigma: np.ndarray = None,
):
    """Entrypoint for computing FPD

    Parameters
    ----------
    samp_embeds
        Array of embeddings of sampled structures, shape (N, D) for N samples with D dimensions each
        If using per-residue embeddings, can be mean-pooled along the sequence dimension to achieve (N, D)
            or treat each residue as a separate embedding, merged into axis=0
    gt_embeds
        Array of embeddings of reference structures
    """
    samp_mu = np.mean(samp_embeds, axis=0)
    samp_sigma = np.cov(samp_embeds, rowvar=False)

    if gt_embeds is not None:
        gt_mu = np.mean(gt_embeds, axis=0)
        gt_sigma = np.cov(gt_embeds, rowvar=False)

    return calculate_frechet_distance(samp_mu, samp_sigma, gt_mu, gt_sigma)



def compute_motif_bb_rmsd(pdb_path: str,
                          motif_coords: TensorType["n a 3", float],
                          motif_mask: TensorType["n a", float]) -> float:
    """
    Given the path to a PDB of length n, and a motif with its mask, compute the RMSD between the motif and the PDB.

    Following MotifBench, the RMSD is computed between the N, CA, C atoms of the motif and the PDB.
    """
    pdb_feats = data.load_feats_from_pdb(pdb_path)

    x = pdb_feats["all_atom_positions"]  # [n 37 3]

    # Construct a mask for the motif backbone atoms
    bb_motif_atom_mask = torch.zeros_like(motif_mask)
    atom_indices = [rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]]  # Zheng et al. MotifBench only uses N, CA, C
    bb_motif_atom_mask[..., atom_indices] = 1
    bb_motif_atom_mask = bb_motif_atom_mask * motif_mask  # [n a]

    if bb_motif_atom_mask.sum() == 0:
        return np.nan

    # Kabsch align the motif to the corresponding positions in the PDB
    bb_rmsd = data.torch_rmsd_weighted(rearrange(x, "n a x -> 1 (n a) x"),
                                       rearrange(motif_coords, "n a x -> 1 (n a) x"),
                                       weights=rearrange(bb_motif_atom_mask, "n a -> 1 (n a)")).squeeze(0)

    return bb_rmsd.item()
