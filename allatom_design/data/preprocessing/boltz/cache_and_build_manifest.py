#!/usr/bin/env python3
import argparse
import fcntl
import glob
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import Connection, Input, Structure
from boltz.data.write.mmcif import to_mmcif
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import conversion
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (atom14_aatype_to_atom37,
                                      atom37_to_atom14,
                                      get_interface_residue_mask)

tokenizer = BoltzTokenizer()
featurizer = BoltzFeaturizer()


@hydra.main(config_path="../../../configs/data/preprocessing/boltz", config_name="cache_and_build_manifest", version_base="1.3.2")
def main(cfg: DictConfig):
    # Load in processed targets
    boltz_processed_files = glob.glob(f"{cfg.pdb_path}/rcsb_processed_targets/structures/*.npz")

    # Sort files by size
    boltz_processed_files.sort(key=lambda x: Path(x).stat().st_size)

    cache_examples(boltz_processed_files, cfg.pdb_path, cfg.overwrite_cache, cfg.num_workers)


def cache_examples(npz_files: list[str], pdb_path: str, overwrite_cache: bool, num_workers: int):
    """
    Reads in PDB files and caches the examples to disk.
    Cached files are stored as {pdb_id}.pt in {pdb_path}/cached_examples.
    """
    cache_dir = f"{pdb_path}/cached_examples"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Caching examples to {cache_dir} with {num_workers} workers...")
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(cache_npz_file)(npz_file, pdb_path, cache_dir, overwrite_cache) for npz_file in npz_files]
    list(parallel(tqdm(jobs, total=len(jobs), desc="Caching PDBs")))
    print("Caching completed.")
    return cache_dir


def cache_npz_file(npz_file: str, pdb_path: str, cache_dir: str, overwrite_cache: bool):
    pdb_key = Path(npz_file).stem
    out_file = f"{cache_dir}/{pdb_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False
    try:
        feats = load_feats_from_boltz_npz(npz_file)
        torch.save(feats, out_file)
    except Exception as e:
        # write to error file with a lock
        print(f"Error caching {npz_file}: {e}")
        with open(f"{pdb_path}/error.txt", "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f"{npz_file}: {e}\n")
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return


def load_feats_from_boltz_npz(npz_file: str) -> dict:
    pdb_id = Path(npz_file).stem

    structure = np.load(npz_file)
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )
    input_data = Input(structure=structure, msa={})  # do not load in MSAs
    tokenized = tokenizer.tokenize(input_data)
    boltz_feats = featurizer.process(tokenized, training=False)
    boltz_restypes = torch.tensor(tokenized.tokens["res_type"], dtype=torch.long)

    # # DEBUG: save mmcif
    # mmcif_file = Path(out_dir, f"{pdb_id}.mmcif")
    # with open(mmcif_file, "w") as f:
    #     f.write(to_mmcif(structure))

    # Pad all tokens to atom23 format (max between 14 for proteins and 23 for nucleic acids)
    tokenwise_feats = pad_atom_feats_to_tokenwise(boltz_feats, max_atoms_per_token=23)

    # === Protein feats === #
    # convert atom mask and coords from atom14 to atom37
    atom14_tokenwise_feats = {k: v[:, :14] for k, v in tokenwise_feats.items()}
    openfold_restypes = torch.tensor([conversion.boltz_token_id_to_restype_id[x.item()] for x in boltz_restypes])  # convert to openfold restypes vocab
    atom_mask = atom14_aatype_to_atom37(atom14_tokenwise_feats["atom_resolved_mask"][..., None], openfold_restypes).squeeze(-1)  # add dummy xyz dimension for conversion
    all_atom_positions = atom14_aatype_to_atom37(atom14_tokenwise_feats["coords"], openfold_restypes) * atom_mask[..., None]
    ref_pos = atom14_aatype_to_atom37(atom14_tokenwise_feats["ref_pos"], openfold_restypes) * atom_mask[..., None]

    # build protein feats in openfold format
    feats = {}
    feats["all_atom_positions"] = all_atom_positions
    feats["all_atom_mask"] = atom_mask
    feats["aatype"] = openfold_restypes
    feats["residue_index"] = boltz_feats["residue_index"]
    feats["chain_index"] = boltz_feats["asym_id"]
    feats["seq_mask"] = torch.ones_like(feats["aatype"], dtype=torch.float32)

    feats["target_feat"] = F.one_hot(feats["aatype"], num_classes=len(rc.restypes_with_x)).float()
    feats["ref_pos"] = ref_pos
    feats["ref_element"] = atom14_tokenwise_feats["ref_element"]
    feats["ref_charge"] = atom14_tokenwise_feats["ref_charge"]

    # subset to protein tokens
    protein_token_mask = boltz_feats["mol_type"] == const.chain_type_ids["PROTEIN"]  # only protein chains
    known_residue_mask = (boltz_restypes != const.token_ids[const.unk_token["PROTEIN"]])  # only known residues; exclude non-standard or unknown residues
    protein_token_mask = protein_token_mask & known_residue_mask  # only protein chains with known residues

    for k, v in feats.items():
        feats[k] = v[protein_token_mask]

    # DEBUG: convert back to atom14 and double check that everything looks good
    atom14_coords, atom14_mask = atom37_to_atom14(feats["aatype"], feats["all_atom_positions"], feats["all_atom_mask"])
    protein_tokenwise_mask = atom14_tokenwise_feats["atom_resolved_mask"][protein_token_mask]
    protein_tokenwise_coords = atom14_tokenwise_feats["coords"][protein_token_mask] * protein_tokenwise_mask[..., None]
    if not (atom14_mask == protein_tokenwise_mask).all():
        raise ValueError(f"atom14_mask mismatch for {pdb_id}")
    if not (atom14_coords * atom14_mask[..., None] == protein_tokenwise_coords).all():
        raise ValueError(f"atom14_coords mismatch for {pdb_id}")

    # Handle the distinction between missing atoms and ghost atoms in the atom mask
    ghost_atom_mask = 1 - torch.tensor(rc.restype_atom37_mask)[feats["aatype"]]  # 1 for atoms that are not in the residue type; ghost atoms
    missing_atom_mask = (1 - feats["all_atom_mask"]) * (1 - ghost_atom_mask)  # 1 for atoms that are missing in the PDB file; missing if not in atom_mask but not a ghost atom

    feats["ghost_atom_mask"] = ghost_atom_mask  # [n, a]
    feats["missing_atom_mask"] = missing_atom_mask  # [n, a]
    feats["interface_residue_mask"] = get_interface_residue_mask(feats["all_atom_positions"], feats["chain_index"])
    feats["chain_id_mapping"] = {chain_id: asym_id for chain_id, asym_id in zip(structure.chains["name"], structure.chains["asym_id"])}  # show all chain mappings, even invalid ones

    # === Nucleic acid feats === #
    # We use boltz vocabulary for nucleic acids
    na_feats = {}
    na_feats["atom_positions"] = tokenwise_feats["coords"]
    na_feats["atom_mask"] = tokenwise_feats["atom_resolved_mask"]
    na_feats["res_type"] = boltz_restypes
    na_feats["residue_index"] = boltz_feats["residue_index"]
    na_feats["chain_index"] = boltz_feats["asym_id"]
    na_feats["seq_mask"] = torch.ones_like(na_feats["res_type"], dtype=torch.float32)

    na_feats["ref_pos"] = tokenwise_feats["ref_pos"]
    na_feats["ref_element"] = tokenwise_feats["ref_element"]
    na_feats["ref_charge"] = tokenwise_feats["ref_charge"]

    # subset to nucleic acid tokens
    dna_token_mask = boltz_feats["mol_type"] == const.chain_type_ids["DNA"]
    known_dna_residue_mask = (boltz_restypes != const.token_ids[const.unk_token["DNA"]])
    dna_token_mask = dna_token_mask & known_dna_residue_mask  # only dna chains with known residues

    rna_token_mask = boltz_feats["mol_type"] == const.chain_type_ids["RNA"]
    known_rna_residue_mask = (boltz_restypes != const.token_ids[const.unk_token["RNA"]])
    rna_token_mask = rna_token_mask & known_rna_residue_mask  # only rna chains with known residues

    na_token_mask = dna_token_mask | rna_token_mask  # either dna or rna

    for k, v in na_feats.items():
        na_feats[k] = v[na_token_mask]
    feats["na_feats"] = na_feats

    # === Save some other features === #
    feats["boltz_feats"] = boltz_feats

    return feats


def pad_atom_feats_to_tokenwise(boltz_feats: dict,
                                max_atoms_per_token: int):
    # Build padded atom idxs
    n_atoms_per_token = boltz_feats["atom_to_token"].sum(dim=0)
    # atom_idxs = torch.tensor(tokenized.tokens["atom_idx"])  # this does not work since doesn't account for removal of invalid chains
    atom_idxs = torch.cat([torch.zeros(1), n_atoms_per_token.cumsum(dim=0)[:-1]]).int()
    padded_atom_idxs = atom_idxs[:, None].expand(-1, max_atoms_per_token)
    padded_atom_idxs = padded_atom_idxs + torch.arange(max_atoms_per_token)[None, :]  # [n, 14]
    pad_mask = torch.arange(max_atoms_per_token)[None, :] < n_atoms_per_token[:, None]  # [n, 14]
    padded_atom_idxs = padded_atom_idxs * pad_mask  # mask out ghost atoms

    # Gather from each feature of interest
    tokenwise_feats = {}
    N = padded_atom_idxs.shape[0]
    for k in ["coords", "atom_resolved_mask", "ref_pos", "ref_element", "ref_charge"]:
        v = boltz_feats[k]
        if k == "coords":
            # coords is [1, n_atoms, 3]
            v = v.squeeze(0)
        data_shape = v.shape[1:]
        gather_idxs = padded_atom_idxs.view(-1, *((1,) * len(data_shape))).expand(-1, *data_shape)
        tokenwise_feats[k] = v.gather(0, gather_idxs).view(N, max_atoms_per_token, *data_shape)
        tokenwise_feats[k] = tokenwise_feats[k] * pad_mask.view(N, max_atoms_per_token, *((1,) * len(data_shape)))

    return tokenwise_feats


if __name__ == "__main__":
    main()