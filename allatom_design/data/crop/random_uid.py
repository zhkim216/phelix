from dataclasses import replace
from typing import Optional

import numpy as np
from allatom_design.data.crop.cropper import Cropper

from allatom_design.data.types import Tokenized


class RandomUIDCropper(Cropper):
    """
    Random UID cropper. Returns a random crop of the data by randomly selecting UIDs until the crop is the desired size.
    A UID here is defined as a unique (chain id, residue index) tuple, similar to ref_space_uid in the featurizer.

    Used for cropping motifs.
    """

    def __init__(self) -> None:
        """Initialize the cropper."""
        pass

    def crop(
        self,
        data: Tokenized,
        random: np.random.RandomState,
        max_tokens: int,
        max_atoms: int
    ) -> Tokenized:
        """
        TODO: this can technically leak a bit of information about number of atoms per residue?
        """
        # Get UIDs
        uid_pairs = np.stack([data.tokens["asym_id"], data.tokens["res_idx"]], axis=1)
        _, uids = np.unique(uid_pairs, axis=0, return_inverse=True)  # uids are the indices of the unique array that each token belongs to

        # Get the number of tokens and atoms per UID
        tokens_per_uid = np.bincount(uids)
        atoms_per_uid = np.bincount(uids, weights=data.tokens["atom_num"])

        # Randomly permute the UID indices
        uid_indices = np.arange(len(tokens_per_uid))  # number of unique UIDs
        random.shuffle(uid_indices)

        # Compute cumulative sum of tokens and atoms per UID to avoid exceeding max_tokens and max_atoms
        tokens_cumsum = np.cumsum(tokens_per_uid[uid_indices])
        atoms_cumsum = np.cumsum(atoms_per_uid[uid_indices])

        # Find how many UIDs can be included in the crop without exceeding max_tokens and max_atoms
        valid_mask = (tokens_cumsum <= max_tokens) & (atoms_cumsum <= max_atoms)
        uids_to_keep = uid_indices[valid_mask]
        keep_mask = np.isin(uids, uids_to_keep)

        # Filter tokens and tokenwise atom features
        token_data = data.tokens[keep_mask]
        tokenwise_atom_feats = data.tokenwise_atom_feats[keep_mask]

        # Filter bonds
        indices = token_data["token_idx"]
        token_bonds = data.bonds
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        data = replace(data, tokens=token_data, bonds=token_bonds, tokenwise_atom_feats=tokenwise_atom_feats)
        return data
