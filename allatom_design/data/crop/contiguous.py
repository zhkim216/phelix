from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from allatom_design.data import const
from allatom_design.data.crop.cropper import Cropper
from allatom_design.data.data import subset_tokenized
from allatom_design.data.types import Tokenized
from allatom_design.data.crop.boltz import pick_chain_token, pick_interface_token, pick_random_token


class ContiguousCropper(Cropper):
    """Contiguous-only cropping."""

    def __init__(self, subset_chain_types: list[str] | None = None) -> None:
        self.subset_chain_types = subset_chain_types


    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
        return_crop_mask: bool = False,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.
        return_crop_mask : bool, optional
            Whether to return the crop mask.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Check inputs
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds
        mask = data.structure.mask
        chains = data.structure.chains
        interfaces = data.structure.interfaces

        # Filter to a subset of chain types
        if self.subset_chain_types is not None:
            subset_chain_type_ids = [const.chain_type_ids[chain_type] for chain_type in self.subset_chain_types]
            mask = mask & np.isin(chains["mol_type"], subset_chain_type_ids)

        # Filter to valid chains
        valid_chains = chains[mask]

        # Filter to valid interfaces
        valid_interfaces = interfaces
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # Pick a random token: chain or interface
        if chain_id is not None:
            query = pick_chain_token(valid_tokens, chain_id, random)
        elif interface_id is not None:
            interface = interfaces[interface_id]
            query = pick_interface_token(valid_tokens, interface, random)
        elif valid_interfaces.size:
            idx = random.randint(len(valid_interfaces))
            interface = valid_interfaces[idx]
            query = pick_interface_token(valid_tokens, interface, random)
        else:
            idx = random.randint(len(valid_chains))
            chain_id = valid_chains[idx]["asym_id"]
            query = pick_chain_token(valid_tokens, chain_id, random)

        # Select a contiguous subset of tokens around the query token
        cropped: set[int] = set()
        total_atoms = 0

        chain_tokens = token_data[token_data["asym_id"] == query["asym_id"]]

        # Expand by res_idx until we have enough tokens
        for i in range(max_tokens):
            left, right = query["res_idx"] - i, query["res_idx"] + i
            new_tokens = chain_tokens[(chain_tokens["res_idx"] == left) | (chain_tokens["res_idx"] == right)]
            new_atoms = np.sum(new_tokens["atom_num"])

            if len(cropped) + len(new_tokens) > max_tokens:
                # We're about to exceed the max number of tokens
                break
            if (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms):
                # We're about to exceed the max number of atoms
                break

            cropped.update(new_tokens["token_idx"])
            total_atoms += new_atoms

        # Get tokens to crop as a mask based on sorted indices
        keep_mask = np.zeros(len(token_data), dtype=bool)
        keep_mask[sorted(cropped)] = True

        # Return the cropped tokens
        data = subset_tokenized(data, keep_mask)
        if return_crop_mask:
            return data, keep_mask
        return data
