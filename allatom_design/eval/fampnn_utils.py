"""
Utils for sampling from FAMPNN.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data.data import (load_feats_from_pdb, pad_to_max_len,
                                      process_single_pdb)
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser
from allatom_design.data import residue_constants as rc


def get_fampnn_batch(pdb_batch_files: List[str], device: str) -> Tuple[Dict[str, TensorType["b n ..."]],
                                                                       List[str],
                                                                       List[Dict[str, int]],  # maps chain letters to chain index
                                                                       ]:
    # Load and process all PDBs in this batch
    batch_list = []
    batch_chain_id_mapping = []
    for pdb_file in pdb_batch_files:
        data = load_feats_from_pdb(pdb_file)
        single = process_single_pdb(data)
        batch_list.append(single)

        # store chain ID mapping for parsing fixed positions
        batch_chain_id_mapping.append(data["chain_id_mapping"])

    pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

    # Create a batch dictionary from batch_list by stacking
    model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
    max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
    batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
    batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

    # Move to device
    batch = {k: batch[k].to(device) for k in model_input_keys}

    return batch, pdb_names, batch_chain_id_mapping



def create_fampnn_embeddings(model: FAMPNNDenoiser,
                             pdb_paths: List[str],
                             backbone_only: bool,
                             batch_size: int,
                             device: str,
                             out_dir: str):
    """
    Create FAMPNN embeddings for a list of PDB files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pdb_paths), desc="Creating FAMPNN embeddings")
    for i in range(0, len(pdb_paths), batch_size):
        pdb_batch_files = pdb_paths[i:i + batch_size]
        B = len(pdb_batch_files)

        batch, pdb_names, _ = get_fampnn_batch(pdb_batch_files, device)
        with torch.no_grad():
            x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"], batch["aatype"], batch["seq_mask"], batch["missing_atom_mask"], batch["residue_index"], batch["chain_index"]
            if backbone_only:
                # Zero out aatype and sidechains
                aatype = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
                seq_mlm_mask = torch.zeros_like(seq_mask)
                scn_mlm_mask = torch.zeros_like(seq_mask)
            else:
                seq_mlm_mask = torch.ones_like(seq_mask)
                scn_mlm_mask = torch.ones_like(seq_mask)

            _, mpnn_feature_dict = model.score(x=x,
                                               aatype=aatype,
                                               seq_mask=seq_mask,
                                               missing_atom_mask=missing_atom_mask,
                                               scn_mlm_mask=scn_mlm_mask,
                                               residue_index=residue_index,
                                               chain_index=chain_index,
                                               return_embeddings=True)

        # Save FAMPNN feature dict to output directory
        mpnn_feature_dict = {k: v.cpu() for k, v in mpnn_feature_dict.items() if k in ["h_V", "h_V_enc"]}  # prune to node embeddings only
        lengths = seq_mask.sum(dim=-1).long()
        for j in range(B):
            name = pdb_names[j]
            out_file = Path(out_dir) / f"{name}.pt"
            length_j = lengths[j].item()
            mpnn_feature_dict_j = {k: v[j, :length_j].clone() for k, v in mpnn_feature_dict.items()}
            torch.save(mpnn_feature_dict_j, out_file)

        pbar.update(B)
    pbar.close()


