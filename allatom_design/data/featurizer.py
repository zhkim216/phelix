from typing import Optional

import numpy as np
from boltz.data.feature.featurizer import (process_atom_features,
                                           process_token_features)
from boltz.data.tokenize.boltz import Tokenized
from torch import Tensor


class SDFeaturizer:
    """Boltz-based sequence denoiser featurizer."""

    def process(
        self,
        data: Tokenized,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
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
        token_features = process_token_features(
            data,
            max_tokens,
        )

        # Compute atom features
        atom_features = process_atom_features(
            data,
            atoms_per_window_queries,
            min_dist,
            max_dist,
            num_bins,
            max_atoms,
            max_tokens,
        )

        return {
            **token_features,
            **atom_features,
        }
