import math
import random
from typing import Optional

import numpy as np
import torch
from boltz.model.modules.utils import \
    center_random_augmentation as boltz_center_random_augmentation
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot

from allatom_design.data import const
from allatom_design.data.feature.pad import pad_dim, crop_dim
from allatom_design.data.tokenize.boltz import Tokenized
from torchtyping import TensorType

# Keep track of the dimensions of the features in the token and atom features for padding & cropping
FEAT_TO_TOKEN_DIM = {
    # token features
    "token_index": [0],
    "residue_index": [0],
    "asym_id": [0],
    "entity_id": [0],
    "sym_id": [0],
    "mol_type": [0],
    "res_type": [0],
    "disto_center": [0],
    "token_pad_mask": [0],
    "token_resolved_mask": [0],
    "token_disto_mask": [0],
    "center_coords": [0],
    "token_bonds": [0, 1],

    # atom features
    "atom_to_token": [1]
}

FEAT_TO_ATOM_DIM = {
    # atom features
    "atom_pad_mask": [0],
    "ref_pos": [0],
    "atom_resolved_mask": [0],
    "ref_element": [0],
    "ref_charge": [0],
    "ref_atom_name_chars": [0],
    "ref_space_uid": [0],
    "coords": [1],
    "atom_to_token": [0],
    "prot_bb_atom_mask": [0],
    "prot_scn_atom_mask": [0],
}


class SequenceDesignFeaturizer:
    """Boltz-based featurizer modified for sequence design featurization."""

    def process(
        self,
        data: Tokenized,
        atoms_per_window_queries: int = 32,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Compute features.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        training : bool
            Whether the model is in training mode.
        max_tokens : int, optional
            The maximum number of tokens.
        max_atoms : int, optional
            The maximum number of atoms
        max_seqs : int, optional
            The maximum number of sequences.

        Returns
        -------
        dict[str, Tensor]
            The features for model training.

        """
        # Compute token features
        token_features = process_sd_token_features(
            data,
        )

        # Compute atom features
        atom_features = process_sd_atom_features(
            data,
            num_bins,
        )

        feats = {**token_features, **atom_features}

        # Pad features
        feats = pad_sd_feats(feats, max_tokens, max_atoms, atoms_per_window_queries)

        return feats


def process_sd_token_features(
    data: Tokenized,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"]).long()
    asym_id = from_numpy(token_data["asym_id"]).long()
    entity_id = from_numpy(token_data["entity_id"]).long()
    sym_id = from_numpy(token_data["sym_id"]).long()
    mol_type = from_numpy(token_data["mol_type"]).long()
    res_type = from_numpy(token_data["res_type"]).long()
    res_type = one_hot(res_type, num_classes=const.num_tokens)
    disto_center = from_numpy(token_data["disto_coords"])
    center_coords = from_numpy(token_data["center_coords"])

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"]).float()
    disto_mask = from_numpy(token_data["disto_mask"]).float()

    # Token bond features
    num_tokens = len(token_data)
    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1

    bonds = bonds.unsqueeze(-1)

    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "center_coords": center_coords,
    }
    return token_features


def process_sd_atom_features(
    data: Tokenized,
    num_bins: int = 64,
) -> dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    dict[str, Tensor]
        The atom features.

    """
    # Filter to tokens' atoms
    atom_data = []
    ref_space_uid = []
    coord_data = []
    prot_bb_atom_mask = []  # 1 if atom is backbone atom of a known protein residue, 0 otherwise
    prot_scn_atom_mask = []  # 1 if atom is sidechain atom of a known protein residue, 0 otherwise
    atom_to_token = []
    atom_idx = 0

    chain_res_ids = {}
    for token_id, token in enumerate(data.tokens):
        # Get the chain residue ids
        chain_idx, res_id = token["asym_id"], token["res_idx"]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Map atoms to token indices
        atom_num = token["atom_num"]
        ref_space_uid.extend([new_idx] * atom_num)
        atom_to_token.extend([token_id] * atom_num)

        # Add atom data
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Get token coordinates
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # Fill in protein backbone and sidechain atom masks
        chain_type = const.chain_types[token["mol_type"]]
        if chain_type == "PROTEIN":
            restype = const.token_ids[const.tokens[token["res_type"]]]
            prot_bb_atom_mask.extend(const.restype_atom_bb[restype].tolist()[:atom_num])
            prot_scn_atom_mask.extend(const.restype_atom_scn[restype].tolist()[:atom_num])
        else:
            prot_bb_atom_mask.extend([0] * atom_num)
            prot_scn_atom_mask.extend([0] * atom_num)

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]  # atom has a copy of first coords
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # Compute features
    ref_atom_name_chars = from_numpy(atom_data["name"]).long()
    ref_element = from_numpy(atom_data["element"]).long()
    ref_charge = from_numpy(atom_data["charge"])
    ref_pos = from_numpy(
        atom_data["conformer"].copy()
    )  # not sure why I need to copy here..
    ref_space_uid = from_numpy(ref_space_uid)
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"])
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    prot_bb_atom_mask = torch.tensor(prot_bb_atom_mask, dtype=torch.float)
    prot_scn_atom_mask = torch.tensor(prot_scn_atom_mask, dtype=torch.float)

    # Convert to one-hot (convert to bool for I/O efficiency)
    ref_atom_name_chars = one_hot(
        ref_atom_name_chars % num_bins, num_classes=num_bins
    ).bool()  # added for lower case letters
    ref_element = one_hot(ref_element, num_classes=const.num_elements).bool()
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1).bool()

    # Apply random roto-translation to the input atoms
    ref_pos = boltz_center_random_augmentation(
        ref_pos[None], resolved_mask[None], centering=False
    )[0]

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "prot_bb_atom_mask": prot_bb_atom_mask,
        "prot_scn_atom_mask": prot_scn_atom_mask,
    }


def select_subset_from_mask(mask, p):
    num_true = np.sum(mask)
    v = np.random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = np.random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def pad_sd_feats(feats: dict[str, Tensor],
                 max_tokens: int | None,
                 max_atoms: int | None,
                 atoms_per_window_queries: int) -> dict[str, Tensor]:
    """Pad the token and atom features to the maximum number of tokens and atoms.
    """
    # Pad to max tokens if given
    if max_tokens is not None:
        token_pad_len = max_tokens - len(feats["token_index"])
        if token_pad_len > 0:
            for k, v in FEAT_TO_TOKEN_DIM.items():
                for dim_to_pad in v:
                    feats[k] = pad_dim(feats[k], dim_to_pad, token_pad_len)

    # Pad to max atoms if given
    if max_atoms is not None:
        assert max_atoms % atoms_per_window_queries == 0
        atom_pad_len = max_atoms - len(feats["atom_resolved_mask"])
    else:
        atom_pad_len = (
            (len(feats["atom_resolved_mask"]) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(feats["atom_resolved_mask"])

    if atom_pad_len > 0:
        for k, v in FEAT_TO_ATOM_DIM.items():
            for dim_to_pad in v:
                feats[k] = pad_dim(feats[k], dim_to_pad, atom_pad_len)

    return feats


def crop_feats(feats: dict[str, Tensor],
               token_crop_mask: TensorType["n", bool],
               max_tokens: int | None,
               max_atoms: int | None,
               atoms_per_window_queries: int) -> dict[str, Tensor]:
    """
    Crop features based on a crop mask specified at the token level.

    Note: after cropping, ref_space_uid and token_index will refer to positions *before* cropping, so make sure to
    account for this when using these features.
    """
    # First, get atoms to crop out as well
    atom_crop_mask = (feats["atom_to_token"] @ token_crop_mask).bool()

    # Subset each feature at the token level
    for k, v in FEAT_TO_TOKEN_DIM.items():
        for dim_to_crop in v:
            feats[k] = crop_dim(feats[k], dim_to_crop, token_crop_mask)

    # Subset each feature at the atom level
    for k, v in FEAT_TO_ATOM_DIM.items():
        for dim_to_crop in v:
            feats[k] = crop_dim(feats[k], dim_to_crop, atom_crop_mask)

    # Pad each feature back to max tokens and atoms, accounting for atoms_per_window_queries
    feats = pad_sd_feats(feats, max_tokens, max_atoms, atoms_per_window_queries)

    return feats
