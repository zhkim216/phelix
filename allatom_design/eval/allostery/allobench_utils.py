"""
AlloBench utilities for the Potts allosteric pocket detection pipeline.

Covers:
  - CSV loading and monomer filtering
  - Lightweight (biotite-only) protein sequence extraction from CIF
  - Batch MMseqs2 alignment with CIGAR parsing -> UniProt-to-PDB index map
  - Allosteric modulator atom masking
  - Custom dual-pass batch preparation that preserves orthosteric ligands
  - Bypass-forward-pass helper that skips ``initialize_sampling_masks``
  - Orthosteric ligand inference from active-site proximity
  - Orthosteric pocket residue labeling
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from biotite.structure import filter_canonical_amino_acids, get_residue_starts
from scipy.spatial.distance import cdist

import atomworks.enums as aw_enums


###########################################################
# CSV loading
###########################################################


_ALLO_SITE_RE = re.compile(r"^([^-]+)-([A-Za-z0-9]{1,4})-(-?\d+)$")


def parse_allosteric_site_residue(value) -> set[tuple[str, int]]:
    """Parse e.g. "['A-PHE-157', 'A-TYR-156', ...]" -> {('A', 157), ('A', 156), ...}.

    Returns empty set on NaN / parse failure. Handles 3- or 4-letter residue names
    and negative author residue IDs. Discards malformed tokens silently.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(items, (list, tuple)):
        return set()
    out: set[tuple[str, int]] = set()
    for tok in items:
        m = _ALLO_SITE_RE.match(str(tok).strip())
        if m is None:
            continue
        chain, _resname, rid = m.groups()
        try:
            out.add((chain, int(rid)))
        except ValueError:
            continue
    return out


def load_allobench(
    allobench_csv: str,
    asd_csv: str | None = None,
    monomer_only: bool = True,
) -> pd.DataFrame:
    """Load AlloBench + (optionally) ASD_High_Resolution, filter monomers.

    Returns a DataFrame with columns:
        pdb_id (lowercase),
        modulator_alias, modulator_chain, modulator_resi,
        sequence,
        active_site_uniprot (list[int]),
        allo_site_pdb_keys (set[(chain_id, author_res_id)]),
        oligomeric_state.
    """
    df = pd.read_csv(allobench_csv)
    df["pdb_id"] = df["allosteric_pdb"].astype(str).str.lower()

    oligo_state = None
    if asd_csv is not None:
        asd = pd.read_csv(asd_csv)
        asd["pdb_id"] = asd["PDB ID"].astype(str).str.lower()
        asd_sub = asd[["pdb_id", "Oligomeric State"]].drop_duplicates("pdb_id")
        df = df.merge(asd_sub, on="pdb_id", how="left")
        oligo_state = "Oligomeric State"

    if monomer_only:
        if oligo_state is None:
            raise ValueError("monomer_only=True requires asd_csv")
        df = df[df[oligo_state] == "Monomer"].copy()

    df = df.drop_duplicates(subset=["pdb_id"]).reset_index(drop=True)

    def _parse_active(v):
        if pd.isna(v):
            return []
        try:
            parsed = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
        if not isinstance(parsed, (list, tuple)):
            return []
        out = []
        for x in parsed:
            try:
                out.append(int(x))
            except (ValueError, TypeError):
                continue
        return out

    df["active_site_uniprot"] = df["active_site_residue"].apply(_parse_active)
    df["allo_site_pdb_keys"] = df["allosteric_site_residue"].apply(parse_allosteric_site_residue)

    def _parse_modulator_resi(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if s == "":
            return None
        try:
            return int(s)
        except ValueError:
            return s

    df["modulator_resi_parsed"] = df["modulator_resi"].apply(_parse_modulator_resi)
    df["modulator_alias"] = df["modulator_alias"].astype(str).str.strip().str.upper()
    df["modulator_chain"] = df["modulator_chain"].astype(str).str.strip()

    cols = [
        "pdb_id",
        "modulator_alias",
        "modulator_chain",
        "modulator_resi_parsed",
        "sequence",
        "active_site_uniprot",
        "allo_site_pdb_keys",
    ]
    if oligo_state is not None:
        cols.append(oligo_state)
        df = df.rename(columns={oligo_state: "oligomeric_state"})
        cols[-1] = "oligomeric_state"

    return df[cols].rename(columns={"modulator_resi_parsed": "modulator_resi"})


###########################################################
# Lightweight CIF loading / protein sequence extraction
###########################################################


def light_load_cif(cif_path: str | Path) -> AtomArray:
    """Load a CIF with biotite only (no atomworks pipeline)."""
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(str(cif_path))
    return pdbx.get_structure(cif, model=1)


def extract_protein_chain_sequences(
    atom_array: AtomArray,
) -> dict[str, list[tuple[int, str]]]:
    """Return {chain_id: [(author_res_id, one_letter_aa), ...]} for each protein chain.

    Uses biotite's canonical amino-acid filter. MSE is treated as M via three-to-one.
    Missing residues (gaps in author numbering) are silently omitted.
    """
    ca_mask = (atom_array.atom_name == "CA") & filter_canonical_amino_acids(atom_array)
    if ca_mask.sum() == 0:
        return {}

    out: dict[str, list[tuple[int, str]]] = {}
    chain_ids = atom_array.chain_id[ca_mask]
    res_ids = atom_array.res_id[ca_mask]
    res_names = atom_array.res_name[ca_mask]

    for cid, rid, rname in zip(chain_ids, res_ids, res_names):
        try:
            aa1 = ProteinSequence.convert_letter_3to1(str(rname))
        except KeyError:
            continue
        cid_s = str(cid)
        out.setdefault(cid_s, []).append((int(rid), aa1))

    for cid in list(out.keys()):
        seen: set[int] = set()
        deduped: list[tuple[int, str]] = []
        for rid, aa in out[cid]:
            if rid in seen:
                continue
            seen.add(rid)
            deduped.append((rid, aa))
        deduped.sort(key=lambda t: t[0])
        out[cid] = deduped

    return out


###########################################################
# MMseqs2 batch alignment
###########################################################


_CIGAR_OP_RE = re.compile(r"(\d+)([MIDNSHPX=])")


def _parse_cigar(cigar: str) -> list[tuple[int, str]]:
    """Parse a CIGAR string into list of (length, op)."""
    return [(int(n), op) for n, op in _CIGAR_OP_RE.findall(cigar)]


def _build_uniprot_to_qpos_from_cigar(
    cigar: str,
    qstart: int,
    tstart: int,
) -> dict[int, int]:
    """From MMseqs2 alignment coords + CIGAR, build a UniProt(1-indexed) -> PDB-query(1-indexed) map.

    Query = PDB observed sequence, Target = UniProt sequence.
    Op semantics (MMseqs2 CIGAR, query-centric):
        M / = / X -> aligned (advance both)
        I          -> insertion in query (advance query only)
        D          -> deletion in query (advance target only)
    """
    q = qstart
    t = tstart
    mapping: dict[int, int] = {}
    for length, op in _parse_cigar(cigar):
        if op in ("M", "=", "X"):
            for _ in range(length):
                mapping[t] = q
                q += 1
                t += 1
        elif op == "I":
            q += length
        elif op == "D":
            t += length
        else:
            continue
    return mapping


def run_mmseqs_alignment_batch(
    entries: list[dict],
    mmseqs_bin: str,
    tmp_dir: str | Path,
    min_identity: float = 0.8,
    threads: int = 8,
) -> dict[tuple[str, str], dict]:
    """Batch-align PDB chain seqs (queries) against UniProt seqs (targets).

    Args:
        entries: list of dicts with keys pdb_id, chain_id, pdb_seq, uniprot_seq.
            The same uniprot_seq may appear for multiple entries of the same pdb_id.
        mmseqs_bin: path to the mmseqs binary.
        tmp_dir: writable scratch directory (will be created, left in place for inspection).
        min_identity: minimum fractional identity (0..1) to keep a hit.
        threads: MMseqs2 thread count.

    Returns:
        dict keyed by (pdb_id, chain_id) for each entry whose best hit meets the identity
        threshold. Value contains:
            uniprot_to_qpos (dict int->int)
            qstart, qend, tstart, tend, pident, alnlen
    """
    if not entries:
        return {}

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    queries_fa = tmp_dir / "queries.fasta"
    targets_fa = tmp_dir / "targets.fasta"
    result_tsv = tmp_dir / "result.tsv"
    mmseqs_tmp = tmp_dir / "mmseqs_scratch"
    mmseqs_tmp.mkdir(parents=True, exist_ok=True)

    target_seqs: dict[str, str] = {}
    for e in entries:
        target_seqs.setdefault(e["pdb_id"], str(e["uniprot_seq"]))

    with open(queries_fa, "w") as f:
        for e in entries:
            qid = f"{e['pdb_id']}_{e['chain_id']}"
            f.write(f">{qid}\n{e['pdb_seq']}\n")
    with open(targets_fa, "w") as f:
        for pid, seq in target_seqs.items():
            f.write(f">{pid}\n{seq}\n")

    fmt_output = "query,target,qstart,qend,tstart,tend,alnlen,pident,bits,cigar"
    cmd = [
        str(mmseqs_bin),
        "easy-search",
        str(queries_fa),
        str(targets_fa),
        str(result_tsv),
        str(mmseqs_tmp),
        "--search-type", "1",
        "--format-mode", "4",
        "--format-output", fmt_output,
        "-e", "1e-5",
        "-s", "7.5",
        "--max-seqs", "50",
        "--threads", str(threads),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mmseqs easy-search failed (rc={proc.returncode})\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    shutil.rmtree(mmseqs_tmp, ignore_errors=True)

    if not result_tsv.exists() or result_tsv.stat().st_size == 0:
        return {}

    df = pd.read_csv(result_tsv, sep="\t")
    if df.empty:
        return {}

    out: dict[tuple[str, str], dict] = {}

    for qid, grp in df.groupby("query"):
        if "_" not in qid:
            continue
        pdb_id, chain_id = qid.split("_", 1)

        same_target = grp[grp["target"].astype(str) == pdb_id]
        if same_target.empty:
            continue

        best = same_target.sort_values("bits", ascending=False).iloc[0]
        pident = float(best["pident"])
        if pident < min_identity * 100 and pident < min_identity:
            continue
        pident_frac = pident / 100.0 if pident > 1.0 else pident
        if pident_frac < min_identity:
            continue

        try:
            uniprot_to_qpos = _build_uniprot_to_qpos_from_cigar(
                cigar=str(best["cigar"]),
                qstart=int(best["qstart"]),
                tstart=int(best["tstart"]),
            )
        except Exception:
            continue

        out[(pdb_id, chain_id)] = {
            "uniprot_to_qpos": uniprot_to_qpos,
            "qstart": int(best["qstart"]),
            "qend": int(best["qend"]),
            "tstart": int(best["tstart"]),
            "tend": int(best["tend"]),
            "pident": pident_frac,
            "alnlen": int(best["alnlen"]),
        }
    return out


def uniprot_idx_to_pdb_residue(
    uniprot_idx: int,
    alignment: dict,
    chain_residues: list[tuple[int, str]],
) -> tuple[int, str] | None:
    """Map a UniProt(1-indexed) position to (author_res_id, one_letter)."""
    qpos = alignment["uniprot_to_qpos"].get(uniprot_idx)
    if qpos is None or qpos < 1 or qpos > len(chain_residues):
        return None
    return chain_residues[qpos - 1]


###########################################################
# Allosteric modulator atom masking
###########################################################


def build_allosteric_atom_mask(
    atom_array: AtomArray,
    modulator_alias: str,
    modulator_chain: str,
    modulator_resi: int | str | None,
) -> np.ndarray | None:
    """Per-atom bool mask selecting the allosteric modulator instance.

    Tries (res_name & chain_id & res_id); falls back to (res_name & chain_id) if needed.
    Returns None if zero atoms match.
    """
    rn = str(modulator_alias).upper().strip()
    cid = str(modulator_chain).strip()

    rn_eq = np.array([str(x).upper() == rn for x in atom_array.res_name], dtype=bool)
    cid_eq = np.array([str(x) == cid for x in atom_array.chain_id], dtype=bool)
    base = rn_eq & cid_eq

    if modulator_resi is not None:
        try:
            rid = int(modulator_resi)
            strict = base & (atom_array.res_id == rid)
            if strict.sum() > 0:
                return strict
        except (TypeError, ValueError):
            pass

    if base.sum() > 0:
        return base

    rn_any = rn_eq
    if rn_any.sum() > 0:
        return rn_any

    return None


###########################################################
# Dual-pass batch prep and bypass forward
###########################################################


_PROTECTED_OBJECT_KEYS = {"atom_array", "example_id", "query_pn_unit_iids"}


def _shallow_tensor_clone(batch: dict) -> dict:
    """Shallow-copy a batch dict; clone any tensor values but share Python objects."""
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = copy(v) if k not in _PROTECTED_OBJECT_KEYS else v
    return out


def _zero_atoms_in_cond_mask(batch: dict, atoms_to_zero: np.ndarray) -> None:
    """In-place: zero ``batch['atom_cond_mask']`` at indices flagged by ``atoms_to_zero``.

    Asserts that the per-atom mask length matches the batch's atom dimension. Operates
    on a fresh clone of the cond mask so the caller can share intermediate batches.
    """
    device = batch["atom_cond_mask"].device
    mask_t = torch.from_numpy(atoms_to_zero.astype(np.bool_)).to(device=device)
    n_atoms_in_mask = batch["atom_cond_mask"].shape[-1]
    if mask_t.numel() != n_atoms_in_mask:
        raise ValueError(
            f"Atom mask length {mask_t.numel()} != batch atom mask length "
            f"{n_atoms_in_mask}. The atom ordering differs between the lightweight "
            f"biotite load and the atomworks pipeline; use batch['atom_array'][0]."
        )
    cond = batch["atom_cond_mask"][0].clone()
    cond[mask_t] = 0
    batch["atom_cond_mask"] = cond.unsqueeze(0)


def prepare_forward_passes(
    batch: dict,
    *,
    pass_A_zero_atoms: np.ndarray | None = None,
    pass_B_zero_atoms: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Build two batches that share the initial mask but zero different atom sets.

    Both passes start from ``initialize_sampling_masks(protein_only=False)`` (i.e. all
    non-polymer atoms conditioned), then per-pass-zero atoms are forced to 0 in
    ``atom_cond_mask`` only. Sequence / noise conditioning is identical between passes.

    Args:
        batch: raw batch dict from ``get_sd_batch``.
        pass_A_zero_atoms: per-atom bool array (same length as atom_array). Indices where
            True are zeroed in pass A's ``atom_cond_mask``. ``None`` -> zero nothing
            (full holo).
        pass_B_zero_atoms: same, for pass B.

    Returns:
        (batch_A, batch_B) -- independent shallow-tensor clones ready for forward.
    """
    from allatom_design.eval.eval_utils.seq_des_utils import initialize_sampling_masks

    base = _shallow_tensor_clone(batch)
    base = initialize_sampling_masks(base, protein_only=False)
    base["noise_labels"] = None
    base["noise"] = None

    batch_A = _shallow_tensor_clone(base)
    batch_B = _shallow_tensor_clone(base)

    if pass_A_zero_atoms is not None:
        _zero_atoms_in_cond_mask(batch_A, pass_A_zero_atoms)
    if pass_B_zero_atoms is not None:
        _zero_atoms_in_cond_mask(batch_B, pass_B_zero_atoms)

    return batch_A, batch_B


def run_potts_forward_prepared(
    model: torch.nn.Module,
    batch: dict,
) -> dict[str, torch.Tensor]:
    """Forward pass without re-initializing sampling masks.

    Mirrors ``potts_utils.run_potts_forward`` but skips ``initialize_sampling_masks``,
    so a caller-provided atom_cond_mask is preserved.
    """
    batch["noise_labels"] = None
    batch["noise"] = None
    sampling_inputs = {"batch_size": 1, "add_noise": False}
    potts_decoder_aux, _, _ = model.denoiser.compute_potts_params(
        batch, sampling_inputs=sampling_inputs
    )
    return potts_decoder_aux


###########################################################
# Orthosteric ligand inference + pocket labeling
###########################################################


def _get_protein_chain_ids(atom_array: AtomArray) -> set[str]:
    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    return {str(c) for c in np.unique(atom_array.chain_id[prot_mask])}


def _get_ligand_candidate_mask(
    atom_array: AtomArray,
    allo_atom_mask: np.ndarray,
) -> np.ndarray:
    """Non-polymer, non-covmod, not-allosteric atoms."""
    non_cov = ~atom_array.is_covalent_modification if hasattr(atom_array, "is_covalent_modification") else np.ones(len(atom_array), dtype=bool)
    non_poly = ~atom_array.is_polymer if hasattr(atom_array, "is_polymer") else np.ones(len(atom_array), dtype=bool)
    return non_poly & non_cov & (~allo_atom_mask)


def identify_orthosteric_ligand_atoms(
    atom_array: AtomArray,
    allo_atom_mask: np.ndarray,
    active_site_pdb_keys: set[tuple[str, int]],
    active_site_proximity: float = 5.0,
) -> np.ndarray:
    """Select ligand ENTITIES with any atom within ``active_site_proximity`` Å of active-site atoms.

    Entity granularity: ``pn_unit_iid`` if available, otherwise ``(chain_id, res_id)``.
    Returns a per-atom bool mask. Empty mask if no active site keys or no candidates.
    """
    if len(active_site_pdb_keys) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    cand_mask = _get_ligand_candidate_mask(atom_array, allo_atom_mask)
    if cand_mask.sum() == 0:
        return np.zeros(len(atom_array), dtype=bool)

    active_atom_mask = np.zeros(len(atom_array), dtype=bool)
    chain_ids = np.array([str(c) for c in atom_array.chain_id])
    res_ids = atom_array.res_id
    for cid, rid in active_site_pdb_keys:
        active_atom_mask |= (chain_ids == cid) & (res_ids == rid)
    if active_atom_mask.sum() == 0:
        return np.zeros(len(atom_array), dtype=bool)

    active_coords = atom_array.coord[active_atom_mask]
    active_coords = active_coords[~np.isnan(active_coords).any(axis=1)]
    if len(active_coords) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    cand_indices = np.where(cand_mask)[0]
    cand_coords = atom_array.coord[cand_indices]
    valid_cand = ~np.isnan(cand_coords).any(axis=1)
    cand_indices = cand_indices[valid_cand]
    cand_coords = cand_coords[valid_cand]
    if len(cand_indices) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    min_dists = cdist(cand_coords, active_coords).min(axis=1)
    close_atom_indices = cand_indices[min_dists <= active_site_proximity]
    if len(close_atom_indices) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    if hasattr(atom_array, "pn_unit_iid"):
        close_keys = np.unique(atom_array.pn_unit_iid[close_atom_indices])
        entity_mask = np.isin(atom_array.pn_unit_iid, close_keys) & cand_mask
    else:
        entity_pairs = set()
        for i in close_atom_indices:
            entity_pairs.add((str(atom_array.chain_id[i]), int(atom_array.res_id[i])))
        entity_mask = np.zeros(len(atom_array), dtype=bool)
        for cid, rid in entity_pairs:
            entity_mask |= (chain_ids == cid) & (res_ids == rid) & cand_mask
    return entity_mask


def compute_per_residue_min_dist_to_atoms(
    atom_array: AtomArray,
    target_atom_mask: np.ndarray,
) -> dict[tuple[str, int], float]:
    """Per-residue minimum distance from any heavy protein atom to any TRUE atom in ``target_atom_mask``.

    Returns ``{}`` if the target mask is empty or all NaN-coord. Per-residue distance is
    the minimum over all of that residue's heavy protein atoms.
    """
    if target_atom_mask.sum() == 0:
        return {}

    tgt_coords = atom_array.coord[target_atom_mask]
    tgt_coords = tgt_coords[~np.isnan(tgt_coords).any(axis=1)]
    if len(tgt_coords) == 0:
        return {}

    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    valid_prot = prot_mask & (~np.isnan(atom_array.coord).any(axis=1))
    prot_indices = np.where(valid_prot)[0]
    if len(prot_indices) == 0:
        return {}

    dists = cdist(atom_array.coord[prot_indices], tgt_coords).min(axis=1)

    out: dict[tuple[str, int], float] = {}
    chain_ids = atom_array.chain_id[prot_indices]
    res_ids = atom_array.res_id[prot_indices]
    for i, (cid, rid) in enumerate(zip(chain_ids, res_ids)):
        key = (str(cid), int(rid))
        d = float(dists[i])
        if key not in out or d < out[key]:
            out[key] = d
    return out


def compute_ortho_pocket_labels(
    atom_array: AtomArray,
    ortho_atom_mask: np.ndarray,
    cutoff: float = 6.0,
) -> dict[tuple[str, int], bool]:
    """True if any protein heavy atom of residue is within ``cutoff`` Å of ortho ligand atoms."""
    return {
        key: (d < cutoff)
        for key, d in compute_per_residue_min_dist_to_atoms(atom_array, ortho_atom_mask).items()
    }


def is_single_protein_chain(atom_array: AtomArray) -> bool:
    """True if the atom array has exactly one distinct polypeptide chain id."""
    return len(_get_protein_chain_ids(atom_array)) == 1
